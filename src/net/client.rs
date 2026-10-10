//! The client side of multiplayer: a [`Connection`] the [`Game`](crate::game)
//! owns while playing on a server, hiding the socket behind a small poll-based
//! API. A background thread does the blocking reads and feeds a channel, so
//! the render loop never stalls on the network. Sends go through a bounded
//! writer thread. Position sends are throttled and heartbeat so a
//! standing-still player still proves they are alive. The join itself
//! ([`Connection::begin_connect`]) runs off the render thread and can be cancelled.
//! Teleport echo (`Position` after `Teleport`) is part of protocol v9.
use std::collections::{HashMap, VecDeque};
use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use quinn::Endpoint;
use tokio::runtime::Runtime;
use glam::DQuat;
use voxel_engine::{Color, DVec3, Vec3};

use crate::coord::Face;
use crate::net::protocol::{self, ClientMessage, ModBytes, ServerMessage};
use crate::net::{MAX_CHAT, MAX_SPEC, PROTOCOL_VERSION, quic};
use crate::presence::{self, Eye, Stance, WireAction};
use crate::sched::RateGate;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Outbound frames waiting on the writer thread. A full queue drops the new
/// frame and leaves the link up; the edit stays pending and expires on its own.
const OUT_QUEUE: usize = 64;
/// Nothing from the server for this long: tell the player the link has gone quiet.
const SILENCE_WARN: Duration = Duration::from_secs(5);
/// Nothing from the server for this long: give up. Matches the QUIC idle timeout.
const SILENCE_GIVE_UP: Duration = Duration::from_secs(12);
/// HUD, console, and the disconnect reason share this phrase.
pub const INTERRUPTED: &str = "connection interrupted";
const MOVE_INTERVAL: Duration = Duration::from_millis(33);
/// So the server's idle timeout never reaps an active-but-idle player.
const HEARTBEAT: Duration = Duration::from_secs(1);
const PING_INTERVAL: Duration = Duration::from_secs(2);
/// Mod-channel traffic is loss-tolerant, so an overrun drops the OLDEST frame
/// on that channel rather than blocking or growing.
const MOD_RING_CAP: usize = 64;
/// Snapshot cells one [`Connection::poll`] hands the game, so a big join overlay
/// is applied over several frames instead of in one long one.
pub(crate) const APPLY_BUDGET: usize = 2048;

struct InboundMod {
    sender: u32,
    seq: u32,
    bytes: ModBytes,
}

/// Snapshotted so we can interpolate between two.
#[derive(Clone, Copy)]
struct Snapshot {
    pos: DVec3,
    yaw: f32,
    pitch: f32,
    frame: DQuat,
    up: Face,
    stance: Stance,
}

pub struct RemotePlayer {
    /// Also the audio runtime's voice `SessionKey`. Kept on the value so
    /// [`peers`](Connection::peers) (which drops the map key) still carries it.
    id: u32,
    /// Shared with the wire message that delivered it and with every draw
    /// record that shows it — a name is cloned as a refcount bump, never a
    /// fresh allocation.
    pub name: Arc<str>,
    pub anim: presence::Animator,
    /// [`presence::peer_color`] of the name, worked out once at the join.
    color: Color,
    /// Joins start hidden (the roster carries names, not positions); the
    /// first pose reveals them and `PeerExited` hides them again — so a
    /// peer who wandered off isn't drawn frozen at their last heard pose.
    visible: bool,
    prev: Snapshot,
    target: Snapshot,
    recv_at: Instant,
    interval: Duration,
    distance: f64,
    /// Occlusion raycast cadence for the name tag (10 Hz is enough; the
    /// result is reused between due steps).
    tag_gate: RateGate,
    tag_occluded: Option<bool>,
}

impl RemotePlayer {
    pub fn id(&self) -> u32 {
        self.id
    }

    pub fn visible(&self) -> bool {
        self.visible
    }

    /// The tint this player is drawn in, the same on every client.
    pub fn color(&self) -> Color {
        self.color
    }

    /// Cached terrain-occlusion bit for the floating name tag. Raycasts on
    /// the first sample and whenever [`RateGate`] says a step is due.
    pub(crate) fn cached_tag_occlusion(&mut self, dt: f32, raycast: impl FnOnce() -> bool) -> bool {
        if self.tag_occluded.is_none() || self.tag_gate.steps(dt) != 0 {
            let hit = raycast();
            self.tag_occluded = Some(hit);
            hit
        } else {
            self.tag_occluded.expect("filled above")
        }
    }
}

pub struct Rendered {
    /// The peer's eye position; drop to [`Feet`](presence::Feet) via
    /// [`Eye::feet`](presence::Eye::feet) with `stance` before rendering.
    pub pos: Eye,
    pub yaw: f32,
    pub pitch: f32,
    pub frame: DQuat,
    /// The up axis of the latest snapshot. Faces don't interpolate.
    pub up: Face,
    pub speed: f32,
    pub phase: f32,
    /// Broadcast stance; the renderer's animator handles the visual blend.
    pub stance: Stance,
}

impl RemotePlayer {
    /// Interpolate this peer's pose at `now`, clamped to the latest packet (no
    /// extrapolation), and report speed/phase for the walk animation.
    pub fn sample(&self, now: Instant) -> Rendered {
        let secs = self.interval.as_secs_f64();
        let alpha = if secs > 0.0 {
            (now.duration_since(self.recv_at).as_secs_f64() / secs).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let pos = self.prev.pos.lerp(self.target.pos, alpha);
        let yaw = lerp_angle(self.prev.yaw, self.target.yaw, alpha as f32);
        let pitch = self.prev.pitch + (self.target.pitch - self.prev.pitch) * alpha as f32;
        let speed = if secs > 0.0 {
            (across_up(self.prev.pos, self.target.pos, self.target.up) / secs) as f32
        } else {
            0.0
        };
        Rendered {
            pos: Eye(pos),
            yaw,
            pitch,
            frame: slerp_frame(self.prev.frame, self.target.frame, alpha),
            up: self.target.up,
            speed,
            phase: (self.distance * presence::STRIDE_FREQ) as f32,
            stance: self.target.stance,
        }
    }
}

/// Distance with the component along `up` removed, so walking on any face
/// swings the gait and a jump along that axis does not.
fn across_up(a: DVec3, b: DVec3, up: Face) -> f64 {
    let mut d = b - a;
    d[up.axis()] = 0.0;
    d.length()
}

/// Frame slerp. Identical frames and the endpoints skip `slerp`, which would
/// drift a stored identity.
fn slerp_frame(a: DQuat, b: DQuat, t: f64) -> DQuat {
    if t <= 0.0 {
        a
    } else if t >= 1.0 {
        b
    } else if a == b {
        a
    } else {
        a.slerp(b, t)
    }
}

/// Shortest-arc angular lerp: wrap `b - a` into `[-π, π]` so a turn across the
/// ±π seam takes the short way round instead of spinning the body.
fn lerp_angle(a: f32, b: f32, t: f32) -> f32 {
    a + presence::wrap_pi(b - a) * t
}

/// Peer presence and movement are applied inside [`Connection::poll`]; these
/// are what the game still has to handle.
pub enum Incoming {
    /// A player's edit (stale revisions were already filtered out by the connection).
    Edit { x: i32, y: i32, z: i32, spec: Arc<str> },
    /// World state from a snapshot: the bootstrap ledger or a reaction commit. Applied
    /// like an edit but nobody placed or broke anything, so no block cue is played.
    Mutation { x: i32, y: i32, z: i32, spec: Arc<str> },
    /// The server accepted our own edit `req`: prediction can forget it.
    EditAccepted { req: u32 },
    /// `restore` is set when no newer authoritative content has landed on the
    /// cell since, so the optimistic apply should roll back.
    EditRejected { req: u32, restore: bool },
    Position { pos: DVec3, frame: DQuat, up: Face },
    Chat { from_name: Arc<str>, channel: u8, text: Arc<str> },
    Joined { name: Arc<str> },
    Left { name: Arc<str> },
    /// Surfaced so the game can react (audio) beyond the local animator
    /// update already applied in `apply()`.
    PeerSwing { id: u32 },
    Time { day: f32, day_secs: f32 },
    /// The server's verdict on our tool use `req` at `cell`: the cell's and the tool's
    /// configurations afterwards (the cell is already applied when `reacted`).
    ToolResult { req: u32, reacted: bool, cell: (i32, i32, i32), cell_spec: Arc<str>, tool_spec: Arc<str> },
    /// `reason` is the server's close phrase when it sent one (empty if the
    /// peer just vanished). "server shutting down" means the process is exiting.
    Disconnected { reason: String },
    /// Nothing from the server for [`SILENCE_WARN`]. Cleared by the next message.
    Interrupted,
}

/// Why [`Connection::connect`] failed. `mods_denied` is empty unless the server
/// answered [`ServerMessage::ModsDenied`](crate::net::protocol::ServerMessage::ModsDenied).
#[derive(Debug)]
pub struct ConnectError {
    message: String,
    /// Package ids the server refuses. The caller disables these and may retry once.
    pub mods_denied: Vec<String>,
}

impl ConnectError {
    fn plain(message: impl Into<String>) -> Self {
        Self { message: message.into(), mods_denied: Vec::new() }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for ConnectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ConnectError {}

/// How long an edit or tool request may sit unanswered before it rolls back.
const PENDING_TTL: Duration = Duration::from_secs(3);

enum PendingKind {
    Edit,
    Tool,
    /// Wait for later requests on this cell before unwinding predictions in reverse order.
    Rejected,
}

struct PendingReq {
    req: u32,
    cell: Cell,
    expect: u32,
    /// Authoritative revision when the prediction was made; `expect` can include earlier requests.
    base: u32,
    sent: Instant,
    kind: PendingKind,
}

/// A world cell, as the wire names it.
type Cell = (i32, i32, i32);

/// What the last `Move` said. An unchanged pose waits for the heartbeat.
#[derive(Clone, Copy, PartialEq)]
struct MovePose {
    pos: DVec3,
    yaw: f32,
    pitch: f32,
    frame: DQuat,
    velocity: Vec3,
    up: Face,
    stance: Stance,
}

/// Everything one joined session knows apart from the socket: the peers, the cells' confirmed
/// revisions, the requests in flight, the ping and the link's health. [`Connection::poll`]
/// feeds it the server's messages in order; tests build one directly.
struct Session {
    /// Where a peer stands until their first pose: the joiner's own spawn.
    spawn: DVec3,
    peers: HashMap<u32, RemotePlayer>,
    /// CONFIRMED cell revisions from the server (snapshot, broadcasts, and
    /// accepted acks) — what future edit expectations are computed against.
    cell_revs: HashMap<Cell, u32>,
    /// In-flight edits and tool uses, in send order. Counted per cell so a
    /// quick break-then-place chain expects the revisions its earlier requests
    /// will commit. Dropped as rejected after [`PENDING_TTL`] with no answer.
    pending_edits: Vec<PendingReq>,
    next_req: u32,
    /// Instant an in-flight `/tp` was sent. Movement is held until a `Position`
    /// verdict lands, or one heartbeat elapses with no reply, so a dropped echo
    /// cannot freeze the client.
    pending_teleport: Option<Instant>,
    ping_sent: Option<(u32, Instant)>,
    ping_seq: u32,
    ping_ms: Option<u32>,
    alive: bool,
    /// [`Incoming::Disconnected`] already surfaced.
    disconnect_emitted: bool,
    /// Last message from the server. Set at connect and on every message.
    last_heard: Instant,
    /// [`Incoming::Interrupted`] already surfaced for the current gap.
    warned: bool,
    /// The join overlay has arrived ([`ServerMessage::SnapshotEnd`]). Stays set.
    snapshot_ready: bool,
}

/// Dropping it closes the QUIC connection, which ends the reader thread and
/// signals the server that this player left.
pub struct Connection {
    conn: quinn::Connection,
    /// Not needed to keep the connection alive (quinn's driver self-sustains
    /// while a connection is open), but required at [`Drop`] to `wait_idle` —
    /// flushing the close frame before the runtime is torn down.
    endpoint: Endpoint,
    rt: Arc<Runtime>,
    /// Bounded handoff to the writer thread. `None` after [`Drop`] takes it.
    writer_tx: Option<SyncSender<Arc<[u8]>>>,
    writer: Option<JoinHandle<()>>,
    inbox: Receiver<ServerMessage>,
    /// Kept OUT of `inbox`. Each channel has its own drop-oldest ring, so one
    /// channel cannot flush another. `HashMap::new` allocates nothing until the
    /// first frame of a channel arrives.
    mod_in: Arc<Mutex<HashMap<Arc<str>, VecDeque<InboundMod>>>>,
    player_id: u32,
    seed: i64,
    worldgen: crate::world::generation::WorldgenKind,
    terrain: crate::world::terrain::TerrainCfg,
    session: Session,
    // Throttling state for outbound moves.
    last_move: Instant,
    last_sent: Option<MovePose>,
    /// Last cruise speed told to the server. `None` means "not cruising" was sent, or nothing yet.
    sent_cruise: Option<f64>,
    wanted_cruise: Option<f64>,
    /// Filled by the reader when the stream ends, before the inbox disconnects.
    close_reason: Arc<Mutex<String>>,
    /// Snapshot cells not handed to the game yet. While any wait, later messages
    /// stay in the inbox, so they still apply after these.
    held: VecDeque<(i32, i32, i32, u32, Arc<str>)>,
}

impl Connection {
    /// `Err` carries a human-readable reason (bad address, refused, wrong
    /// password, version mismatch). No mods are reported.
    pub fn connect(host: &str, port: u16, name: &str, password: &str) -> Result<Self, ConnectError> {
        Self::connect_with(host, port, name, password, &[])
    }

    /// As [`connect`](Self::connect), reporting `mods` as the enabled packages.
    /// That list is the client's own word; the server does not trust it for
    /// anything except the whitelist. Blocks until the attempt finishes;
    /// [`begin_connect`](Self::begin_connect) is the same work off this thread.
    pub fn connect_with(
        host: &str,
        port: u16,
        name: &str,
        password: &str,
        mods: &[(String, String)],
    ) -> Result<Self, ConnectError> {
        Self::begin_connect(host, port, name, password, mods).wait()
    }

    /// DNS, the QUIC handshake, and `Welcome`, on a worker thread.
    /// [`PendingConnect::poll`] is the render thread's view of it.
    pub fn begin_connect(
        host: &str,
        port: u16,
        name: &str,
        password: &str,
        mods: &[(String, String)],
    ) -> PendingConnect {
        let flag = Arc::new(AtomicBool::new(false));
        let notify = Arc::new(tokio::sync::Notify::new());
        let (tx, done) = mpsc::channel();
        let host = host.to_string();
        let name = name.to_string();
        let password = password.to_string();
        let mods = mods.to_vec();
        let flag_worker = Arc::clone(&flag);
        let notify_worker = Arc::clone(&notify);
        thread::spawn(move || {
            let stop = Stop { flag: flag_worker, notify: notify_worker };
            let result = connect_cancellable(&host, port, &name, &password, &mods, &stop);
            let _ = tx.send(result);
        });
        PendingConnect { done, flag, notify }
    }
}

/// One connect attempt the render thread can poll or cancel.
pub struct PendingConnect {
    done: Receiver<Result<Connection, ConnectError>>,
    flag: Arc<AtomicBool>,
    notify: Arc<tokio::sync::Notify>,
}

impl PendingConnect {
    /// `None` while the attempt is still running.
    pub fn poll(&mut self) -> Option<Result<Connection, ConnectError>> {
        match self.done.try_recv() {
            Ok(result) => Some(result),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(Err(ConnectError::plain("connect ended"))),
        }
    }

    /// Ask the worker to stop. A cancel that lands before the worker waits is
    /// kept: [`Notify::notify_one`](tokio::sync::Notify::notify_one) stores a permit.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Relaxed);
        self.notify.notify_one();
    }

    /// Block until the attempt finishes. [`Connection::connect`] is this.
    pub fn wait(self) -> Result<Connection, ConnectError> {
        self.done.recv().unwrap_or_else(|_| Err(ConnectError::plain("connect ended")))
    }
}

/// Shared cancel flag. The worker owns this view; [`PendingConnect`] holds the same arcs.
struct Stop {
    flag: Arc<AtomicBool>,
    notify: Arc<tokio::sync::Notify>,
}

impl Stop {
    fn check(&self) -> Result<(), ConnectError> {
        if self.flag.load(Ordering::Relaxed) {
            Err(ConnectError::plain("cancelled"))
        } else {
            Ok(())
        }
    }
}

async fn until_stopped(stop: &Stop) {
    loop {
        if stop.flag.load(Ordering::Relaxed) {
            return;
        }
        stop.notify.notified().await;
    }
}

fn with_stop<T>(
    rt: &Runtime,
    stop: &Stop,
    timeout_msg: &'static str,
    fut: impl std::future::Future<Output = Result<T, ConnectError>>,
) -> Result<T, ConnectError> {
    stop.check()?;
    rt.block_on(async {
        tokio::select! {
            biased;
            _ = until_stopped(stop) => Err(ConnectError::plain("cancelled")),
            result = tokio::time::timeout(CONNECT_TIMEOUT, fut) => match result {
                Ok(value) => value,
                Err(_) => Err(ConnectError::plain(timeout_msg)),
            },
        }
    })
}

fn connect_cancellable(
    host: &str,
    port: u16,
    name: &str,
    password: &str,
    mods: &[(String, String)],
    stop: &Stop,
) -> Result<Connection, ConnectError> {
    stop.check()?;
    let addrs: Vec<SocketAddr> = (host, port)
        .to_socket_addrs()
        .map_err(|e| ConnectError::plain(format!("bad address: {e}")))?
        .collect();
    stop.check()?;
    if addrs.is_empty() {
        return Err(ConnectError::plain("address resolved to nothing"));
    }
    // One worker drives quinn; the reader, writer and connect threads block on it.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .map_err(|e| ConnectError::plain(format!("runtime: {e}")))?;
    let rt = Arc::new(rt);
    quic::install_crypto();
    let mut last = ConnectError::plain("could not connect");
    for addr in addrs {
        stop.check()?;
        match connect_one(&rt, addr, name, password, mods, stop) {
            Ok(conn) => return Ok(conn),
            Err(e) if e.message == "cancelled" => return Err(e),
            Err(e) => last = e,
        }
    }
    Err(last)
}

fn connect_one(
    rt: &Arc<Runtime>,
    addr: SocketAddr,
    name: &str,
    password: &str,
    mods: &[(String, String)],
    stop: &Stop,
) -> Result<Connection, ConnectError> {
    let (offers, dropped) = protocol::hello_offers(mods);
    if dropped > 0 {
        let omitted = mods.iter().filter(|(id, version)| !offers.iter().any(|o| o.id.as_ref() == id && o.version.as_ref() == version))
            .map(|(id, _)| id.as_str()).collect::<Vec<_>>().join(", ");
        return Err(ConnectError::plain(format!("enabled mods cannot fit the join request: {omitted}")));
    }
    let mods = offers;
    let bind = if addr.is_ipv4() {
        SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0))
    } else {
        SocketAddr::from((Ipv6Addr::UNSPECIFIED, 0))
    };
    let mut endpoint = {
        // Must run inside the runtime: construction spawns quinn's UDP driver.
        let _guard = rt.enter();
        Endpoint::client(bind).map_err(|e| ConnectError::plain(format!("endpoint: {e}")))?
    };
    endpoint.set_default_client_config(quic::client_config());

    let conn = with_stop(rt, stop, "connect timed out", async {
        let connecting = endpoint.connect(addr, "watt").map_err(|e| ConnectError::plain(e.to_string()))?;
        connecting.await.map_err(|e| ConnectError::plain(format!("could not reach {addr}: {e}")))
    })?;
    let (mut send, mut recv) = with_stop(rt, stop, "connect timed out", async {
        conn.open_bi().await.map_err(|e| ConnectError::plain(format!("stream: {e}")))
    })?;

    let id = crate::net::content_id(&crate::block::BlockRegistry::with_builtins());
    let hello = ClientMessage::Hello {
        protocol: PROTOCOL_VERSION,
        worldgen: id.worldgen,
        gravity: id.gravity,
        law: id.law,
        palette: id.palette,
        name: name.into(),
        password: password.into(),
        mods,
    };
    let hello_bytes = hello.encode();
    with_stop(rt, stop, "connect timed out", async {
        protocol::write_frame_async(&mut send, &hello_bytes)
            .await
            .map_err(|e| ConnectError::plain(format!("send failed: {e}")))
    })?;

    // The scratch Vec is reused across frames so the reader loop never allocates.
    let mut frame = Vec::new();
    with_stop(rt, stop, "no reply: timed out", async {
        protocol::read_frame_async(&mut recv, &mut frame)
            .await
            .map_err(|e| ConnectError::plain(format!("no reply: {e}")))
    })?;
        let decoded = ServerMessage::decode(&frame);
        if let Some(ServerMessage::ModsDenied { ids }) = &decoded {
            return Err(ConnectError {
                message: format!(
                    "server refused mods: {}",
                    ids.iter().map(|id| id.as_ref()).collect::<Vec<_>>().join(", ")
                ),
                mods_denied: ids.iter().map(|id| id.to_string()).collect(),
            });
        }
        let (player_id, seed, spawn, worldgen, terrain) = welcome_from(decoded)?;

        let (tx, inbox) = mpsc::channel();
        let mod_in: Arc<Mutex<HashMap<Arc<str>, VecDeque<InboundMod>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let mod_reader = mod_in.clone();
        let reader_rt = rt.clone();
        let close_reason = Arc::new(Mutex::new(String::new()));
        let reason_slot = close_reason.clone();
        let watched = conn.clone();
        thread::spawn(move || {
            while reader_rt.block_on(protocol::read_frame_async(&mut recv, &mut frame)).is_ok() {
                match ServerMessage::decode(&frame) {
                    // A hung-up game side stops draining but channel frames just
                    // roll over; the connection close ends the loop.
                    Some(ServerMessage::PeerModData { channel, sender, seq, bytes }) => {
                        let mut map = mod_reader.lock().unwrap_or_else(PoisonError::into_inner);
                        let ring = map.entry(channel.share()).or_default();
                        if ring.len() >= MOD_RING_CAP {
                            ring.pop_front();
                        }
                        ring.push_back(InboundMod { sender, seq, bytes });
                    }
                    Some(msg) => {
                        if tx.send(msg).is_err() {
                            break; // The game side hung up.
                        }
                    }
                    None => continue, // Skip a junk frame rather than tear down.
                }
            }
            // Record the phrase before dropping `tx`, so `poll` sees it with the disconnect.
            *reason_slot.lock().unwrap_or_else(PoisonError::into_inner) = connection_close_text(&watched);
        });

        let (writer_tx, writer_rx) = mpsc::sync_channel::<Arc<[u8]>>(OUT_QUEUE);
        let writer_rt = Arc::clone(rt);
        let writer = thread::spawn(move || {
            while let Ok(frame) = writer_rx.recv() {
                if writer_rt.block_on(protocol::write_frame_async(&mut send, &frame)).is_err() {
                    break;
                }
            }
        });

        Ok(Connection {
            conn,
            endpoint,
            rt: Arc::clone(rt),
            writer_tx: Some(writer_tx),
            writer: Some(writer),
            inbox,
            mod_in,
            player_id,
            seed,
            worldgen,
            terrain,
            session: Session::new(spawn),
            last_move: Instant::now(),
            last_sent: None,
            sent_cruise: None,
            wanted_cruise: None,
            close_reason,
            held: VecDeque::new(),
        })
}

/// The server's application close phrase, if quinn has it yet. A few retries
/// cover the gap between the read error and the close frame being recorded.
fn connection_close_text(conn: &quinn::Connection) -> String {
    for _ in 0..20 {
        if let Some(err) = conn.close_reason() {
            return match err {
                quinn::ConnectionError::ApplicationClosed(frame) => {
                    String::from_utf8_lossy(&frame.reason).into_owned()
                }
                quinn::ConnectionError::TimedOut => INTERRUPTED.to_string(),
                other => other.to_string(),
            };
        }
        thread::sleep(Duration::from_millis(10));
    }
    String::new()
}

impl Connection {
    pub fn seed(&self) -> i64 {
        self.seed
    }
    pub fn worldgen(&self) -> crate::world::generation::WorldgenKind {
        self.worldgen
    }
    pub fn terrain(&self) -> crate::world::terrain::TerrainCfg {
        self.terrain
    }
    pub fn spawn(&self) -> DVec3 {
        self.session.spawn
    }
    pub fn player_id(&self) -> u32 {
        self.player_id
    }
    pub fn is_alive(&self) -> bool {
        self.session.alive
    }
    pub fn peers(&self) -> impl Iterator<Item = &RemotePlayer> {
        self.session.peers.values()
    }
    pub fn peers_mut(&mut self) -> impl Iterator<Item = &mut RemotePlayer> {
        self.session.peers.values_mut()
    }
    /// The other player with server id `id`, while they are in the session.
    pub fn peer(&self, id: u32) -> Option<&RemotePlayer> {
        self.session.peers.get(&id)
    }
    /// Other players in the session, hidden ones included.
    pub fn peer_count(&self) -> usize {
        self.session.peers.len()
    }
    pub fn ping_ms(&self) -> Option<u32> {
        self.session.ping_ms
    }

    /// True after the join overlay's [`ServerMessage::SnapshotEnd`]. Later snapshots
    /// are reaction batches and do not clear this.
    pub fn snapshot_ready(&self) -> bool {
        self.session.snapshot_ready
    }

    /// True while a silence warning is showing and the link has not been given up.
    pub fn link_interrupted(&self) -> bool {
        self.session.warned && self.session.alive
    }

    /// Peer join/leave/move is applied to the local table here; edits and
    /// chat are returned for the game to handle.
    pub fn poll(&mut self) -> Vec<Incoming> {
        if let Some(nonce) = self.session.ping_due(Instant::now()) {
            self.dispatch(&ClientMessage::Ping { nonce });
        }

        let mut out = Vec::new();
        let mut budget = APPLY_BUDGET;
        self.release_held(&mut budget, &mut out);
        while self.held.is_empty() {
            match self.inbox.try_recv() {
                Ok(ServerMessage::Snapshot { edits }) => {
                    self.session.heard();
                    self.held.extend(edits);
                    self.release_held(&mut budget, &mut out);
                }
                Ok(msg) => self.session.apply(msg, &mut out),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.session.alive = false;
                    break;
                }
            }
        }
        // A budgeted overlay is still being consumed even if its batch arrived earlier.
        if !out.is_empty() {
            self.session.heard();
        }
        let now = Instant::now();
        self.session.expire(now, &mut out);
        let mut reason = self.close_reason.lock().unwrap_or_else(PoisonError::into_inner);
        self.session.silence(now, &mut reason, &mut out);
        drop(reason);
        coalesce_positions(&mut out);
        out
    }

    /// Hand the game up to `budget` held snapshot cells.
    fn release_held(&mut self, budget: &mut usize, out: &mut Vec<Incoming>) {
        while *budget > 0 {
            let Some(cell) = self.held.pop_front() else { return };
            self.session.land_snapshot(cell, out);
            *budget -= 1;
        }
    }
}

impl RemotePlayer {
    /// A peer the roster just announced. prev == target: speed 0 and a stationary phase, no
    /// Option<history> and no special-casing downstream. Hidden until their first pose arrives.
    fn joined(id: u32, name: Arc<str>, at: DVec3) -> Self {
        let at = Snapshot { pos: at, yaw: 0.0, pitch: 0.0, frame: DQuat::IDENTITY, up: Face::PosY, stance: Stance::Standing };
        Self {
            id,
            color: presence::peer_color(&name),
            name,
            anim: presence::Animator::default(),
            visible: false,
            prev: at,
            target: at,
            recv_at: Instant::now(),
            interval: Duration::from_millis(0),
            distance: 0.0,
            tag_gate: RateGate::from_hz(10),
            tag_occluded: None,
        }
    }
}

impl Session {
    fn new(spawn: DVec3) -> Self {
        Self {
            spawn,
            peers: HashMap::new(),
            cell_revs: HashMap::new(),
            pending_edits: Vec::new(),
            next_req: 0,
            pending_teleport: None,
            ping_sent: None,
            ping_seq: 0,
            ping_ms: None,
            alive: true,
            disconnect_emitted: false,
            last_heard: Instant::now(),
            warned: false,
            snapshot_ready: false,
        }
    }

    /// Any message proves the link is alive. A long join's snapshot batches can
    /// hold the Pong back for many seconds.
    fn heard(&mut self) {
        self.last_heard = Instant::now();
        self.warned = false;
    }

    /// The nonce of the ping to send now, if one is due on a live link: one per
    /// [`PING_INTERVAL`], answered or not.
    fn ping_due(&mut self, now: Instant) -> Option<u32> {
        let due = self.ping_sent.is_none_or(|(_, at)| now.saturating_duration_since(at) >= PING_INTERVAL);
        if !due || !self.alive {
            return None;
        }
        self.ping_seq = self.ping_seq.wrapping_add(1);
        self.ping_sent = Some((self.ping_seq, now));
        Some(self.ping_seq)
    }

    /// Note a request about to go out on `cell`: its id and the revision it expects. The
    /// expectation counts the requests in flight on the cell, so a break-then-place chain
    /// lines up with the revisions its earlier requests will commit.
    fn track(&mut self, cell: Cell, kind: PendingKind) -> (u32, u32) {
        let confirmed = self.cell_revs.get(&cell).copied().unwrap_or(0);
        let expect = next_cell_revision(&self.pending_edits, cell, confirmed);
        self.next_req = self.next_req.wrapping_add(1);
        let req = self.next_req;
        self.pending_edits.push(PendingReq { req, cell, expect, base: confirmed, sent: Instant::now(), kind });
        (req, expect)
    }

    /// The server committed revision `rev` on `cell`. Never lowers what is known.
    fn confirm(&mut self, cell: Cell, rev: u32) {
        let known = self.cell_revs.entry(cell).or_insert(0);
        *known = (*known).max(rev);
    }

    /// Authoritative world state (bootstrap ledger, reaction commits): not a
    /// player's act, so it carries no place/break semantics or cue. A revision
    /// older than the one known is stale and dropped.
    fn land_snapshot(&mut self, (x, y, z, rev, spec): (i32, i32, i32, u32, Arc<str>), out: &mut Vec<Incoming>) {
        let cell = (x, y, z);
        if rev >= self.cell_revs.get(&cell).copied().unwrap_or(0) {
            self.cell_revs.insert(cell, rev);
            out.push(Incoming::Mutation { x, y, z, spec });
        }
    }

    /// Requests unanswered for [`PENDING_TTL`]: an edit joins its cell's rejected chain, and a
    /// tool use reports that nothing happened.
    fn expire(&mut self, now: Instant, out: &mut Vec<Incoming>) {
        self.pending_edits.retain_mut(|p| {
            if now.saturating_duration_since(p.sent) < PENDING_TTL {
                return true;
            }
            match p.kind {
                PendingKind::Edit => {
                    p.kind = PendingKind::Rejected;
                    true
                }
                PendingKind::Tool => {
                    let (cell_spec, tool_spec) = (Arc::from(""), Arc::from(""));
                    out.push(Incoming::ToolResult { req: p.req, reacted: false, cell: p.cell, cell_spec, tool_spec });
                    false
                }
                PendingKind::Rejected => true,
            }
        });
        flush_rejections(&mut self.pending_edits, &self.cell_revs, out);
    }

    /// Warn once a gap reaches [`SILENCE_WARN`], give up at [`SILENCE_GIVE_UP`], and surface a
    /// dead link exactly once. `reason` is the close phrase the reader recorded; giving up
    /// without one records [`INTERRUPTED`].
    fn silence(&mut self, now: Instant, reason: &mut String, out: &mut Vec<Incoming>) {
        match link_silence(self.last_heard, now, self.warned) {
            LinkFate::Ok => {}
            LinkFate::Warn => {
                self.warned = true;
                out.push(Incoming::Interrupted);
            }
            LinkFate::GiveUp => {
                // A real close already recorded its reason. Do not overwrite it.
                if self.alive {
                    self.alive = false;
                    if reason.is_empty() {
                        *reason = INTERRUPTED.to_string();
                    }
                }
            }
        }
        self.emit_disconnect(reason, out);
    }

    /// [`Incoming::Disconnected`], once, after the link has died.
    fn emit_disconnect(&mut self, reason: &str, out: &mut Vec<Incoming>) {
        if !self.alive && !self.disconnect_emitted {
            self.disconnect_emitted = true;
            out.push(Incoming::Disconnected { reason: reason.to_string() });
        }
    }

    /// Apply one server message to the session; what the game must handle lands in `out`.
    fn apply(&mut self, msg: ServerMessage, out: &mut Vec<Incoming>) {
        self.heard();
        match msg {
            ServerMessage::Snapshot { edits } => {
                for cell in edits {
                    self.land_snapshot(cell, out);
                }
            }
            ServerMessage::Edit { x, y, z, rev, spec } => {
                // Per-cell revisions make application order-independent: only
                // strictly newer content lands, so a stale or reordered frame
                // can never revert a newer cell.
                let cell = (x, y, z);
                if rev > self.cell_revs.get(&cell).copied().unwrap_or(0) {
                    self.cell_revs.insert(cell, rev);
                    out.push(Incoming::Edit { x, y, z, spec });
                }
            }
            ServerMessage::EditAck { req, accepted, rev } => {
                let Some(at) = self.pending_edits.iter().position(|p| p.req == req) else {
                    return;
                };
                if accepted {
                    let cell = self.pending_edits.remove(at).cell;
                    self.confirm(cell, rev);
                    out.push(Incoming::EditAccepted { req });
                } else {
                    // Restore our optimistic apply only if nothing newer has
                    // confirmed on the cell meanwhile (the race winner's Edit
                    // broadcast may land before or after this ack).
                    self.pending_edits[at].kind = PendingKind::Rejected;
                }
                flush_rejections(&mut self.pending_edits, &self.cell_revs, out);
            }
            ServerMessage::Position { pos, frame, up } => {
                self.pending_teleport = None;
                // TODO: echo a teleport request id so a snap-back Position from an earlier poll cannot still snap the player (wire change).
                out.push(Incoming::Position { pos, frame, up });
            }
            ServerMessage::Chat { from_name, channel, text, .. } => out.push(Incoming::Chat { from_name, channel, text }),
            ServerMessage::Time { day, day_secs } => out.push(Incoming::Time { day, day_secs }),
            ServerMessage::PeerJoined { id, name } => {
                // A duplicate (two joins racing) must not announce the peer twice.
                let std::collections::hash_map::Entry::Vacant(slot) = self.peers.entry(id) else {
                    return;
                };
                out.push(Incoming::Joined { name: name.clone() });
                slot.insert(RemotePlayer::joined(id, name, self.spawn));
            }
            ServerMessage::PeerLeft { id } => {
                if let Some(p) = self.peers.remove(&id) {
                    out.push(Incoming::Left { name: p.name });
                }
            }
            ServerMessage::PeerPoses { poses } => {
                let now = Instant::now();
                for pose in poses.list {
                    apply_pose(&mut self.peers, pose, now);
                }
            }
            ServerMessage::PeerExited { id } => {
                if let Some(p) = self.peers.get_mut(&id) {
                    p.visible = false;
                }
            }
            ServerMessage::PeerSwing { id } => {
                if let Some(p) = self.peers.get_mut(&id) {
                    p.anim.on_action(WireAction::Swing);
                }
                out.push(Incoming::PeerSwing { id });
            }
            ServerMessage::SnapshotEnd => self.snapshot_ready = true,
            ServerMessage::Pong { nonce } => {
                if let Some((sent, at)) = self.ping_sent
                    && sent == nonce
                {
                    self.ping_ms = Some(at.elapsed().as_millis() as u32);
                }
            }
            ServerMessage::Reject { reason } => {
                self.alive = false;
                self.emit_disconnect(&reason, out);
            }
            ServerMessage::ModsDenied { ids } => {
                self.alive = false;
                let list = ids.iter().map(|id| id.as_ref()).collect::<Vec<_>>().join(", ");
                self.emit_disconnect(&format!("mods denied: {list}"), out);
            }
            // A second Welcome is meaningless mid-session.
            ServerMessage::Welcome { .. } => {}
            // Channel frames never reach here: the reader thread routes PeerModData
            // into the per-channel rings, not the inbox this drains.
            ServerMessage::PeerModData { .. } => {}
            ServerMessage::ToolResult { req, reacted, rev, cell_spec, tool_spec } => {
                let Some(at) = self.pending_edits.iter().position(|p| p.req == req) else {
                    return;
                };
                let cell = self.pending_edits.remove(at).cell;
                if reacted {
                    self.confirm(cell, rev);
                }
                out.push(Incoming::ToolResult { req, reacted, cell, cell_spec, tool_spec });
            }
        }
    }
}

fn welcome_from(
    msg: Option<ServerMessage>,
) -> Result<(u32, i64, DVec3, crate::world::generation::WorldgenKind, crate::world::terrain::TerrainCfg), ConnectError> {
    match msg {
        Some(ServerMessage::Welcome { player_id, seed, spawn, worldgen, terrain, law }) => {
            if let Err(ServerMessage::Reject { reason }) = crate::net::protocol::handshake_law(&law) {
                return Err(ConnectError::plain(reason.to_string()));
            }
            Ok((player_id, seed, spawn, worldgen, terrain))
        }
        Some(ServerMessage::Reject { reason }) => Err(ConnectError::plain(reason.to_string())),
        _ => Err(ConnectError::plain("unexpected reply from server")),
    }
}

/// A content confirmation may precede its verdict. Already-confirmed predictions must not
/// be counted twice when calculating the next speculative revision.
fn next_cell_revision(pending: &[PendingReq], cell: Cell, confirmed: u32) -> u32 {
    pending.iter().filter(|p| p.cell == cell && !matches!(p.kind, PendingKind::Rejected))
        .map(|p| p.expect.saturating_add(1)).max().unwrap_or(confirmed).max(confirmed)
}

/// A rejected chain restores its oldest pre-prediction value: unwind newest first. Any newer
/// authoritative content supersedes the entire chain, including its speculative revisions.
fn flush_rejections(
    pending: &mut Vec<PendingReq>,
    cell_revs: &HashMap<Cell, u32>,
    out: &mut Vec<Incoming>,
) {
    let mut i = pending.len();
    while i > 0 {
        i -= 1;
        if !matches!(pending[i].kind, PendingKind::Rejected) {
            continue;
        }
        let cell = pending[i].cell;
        if pending.iter().any(|p| p.cell == cell && !matches!(p.kind, PendingKind::Rejected)) {
            continue;
        }
        let base = pending.iter().filter(|p| p.cell == cell).map(|p| p.base).min().unwrap();
        let restore = cell_revs.get(&cell).copied().unwrap_or(0) <= base;
        // Remove every member together so they share the same authoritative revision floor.
        for at in (0..=i).rev() {
            if pending[at].cell == cell {
                let rejected = pending.remove(at);
                out.push(Incoming::EditRejected { req: rejected.req, restore });
            }
        }
        i = i.min(pending.len());
    }
}

/// What a gap since the last message means. A later message clears `warned`.
enum LinkFate {
    Ok,
    Warn,
    GiveUp,
}

fn link_silence(last_heard: Instant, now: Instant, warned: bool) -> LinkFate {
    let gap = now.saturating_duration_since(last_heard);
    if gap >= SILENCE_GIVE_UP {
        LinkFate::GiveUp
    } else if gap >= SILENCE_WARN && !warned {
        LinkFate::Warn
    } else {
        LinkFate::Ok
    }
}

/// Several `Position` frames can land in one poll (stale snap-back, then the
/// teleport echo). The last one is the server's current pose.
fn coalesce_positions(out: &mut Vec<Incoming>) {
    let mut last_idx = None;
    for (i, e) in out.iter().enumerate() {
        if matches!(e, Incoming::Position { .. }) {
            last_idx = Some(i);
        }
    }
    let Some(last_idx) = last_idx else { return };
    let mut i = 0;
    out.retain(|e| {
        let keep = match e {
            Incoming::Position { .. } => i == last_idx,
            _ => true,
        };
        i += 1;
        keep
    });
}

/// A pose for an unknown id is ignored.
fn apply_pose(peers: &mut HashMap<u32, RemotePlayer>, pose: protocol::PeerPose, now: Instant) {
    let Some(p) = peers.get_mut(&pose.id) else { return };
    let snapshot =
        Snapshot { pos: pose.pos, yaw: pose.yaw, pitch: pose.pitch, frame: pose.frame, up: pose.up, stance: pose.stance };
    if p.visible {
        p.interval = now.saturating_duration_since(p.recv_at);
        p.prev = p.target;
        p.target = snapshot;
        p.distance += across_up(p.prev.pos, p.target.pos, snapshot.up);
    } else {
        // Re-entering interest range: snap, never lerp the
        // avatar across the distance covered while hidden.
        p.visible = true;
        p.interval = Duration::from_millis(0);
        p.prev = snapshot;
        p.target = snapshot;
    }
    p.recv_at = now;
}

/// True while an in-flight `/tp` still has a heartbeat left to hear a
/// `Position` verdict. Past that the hold expires so a dropped echo cannot
/// freeze the client; a later verdict still clears it.
fn teleport_hold_active(pending: Option<Instant>, now: Instant) -> bool {
    pending.is_some_and(|at| now.saturating_duration_since(at) < HEARTBEAT)
}

impl Connection {
    /// Cheap to call every frame; it only actually sends on the movement
    /// cadence or the heartbeat.
    pub fn send_move(&mut self, pos: DVec3, yaw: f32, pitch: f32, frame: DQuat, velocity: Vec3, up: Face, stance: Stance) {
        if !self.session.alive || teleport_hold_active(self.session.pending_teleport, Instant::now()) {
            return;
        }
        let pose = MovePose { pos, yaw, pitch, frame, velocity, up, stance };
        let elapsed = self.last_move.elapsed();
        let changed = self.last_sent != Some(pose);
        let due = (changed && elapsed >= MOVE_INTERVAL) || elapsed >= HEARTBEAT;
        if !due {
            return;
        }
        self.last_move = Instant::now();
        self.last_sent = Some(pose);
        self.dispatch(&ClientMessage::Move { pos, yaw, pitch, frame, velocity, up, stance });
        // A lower declaration follows the last move under the old cap. Keep it until the
        // body's eased velocity fits the new cap, including the final buffered cruise step.
        if self.sent_cruise != self.wanted_cruise
            && self.wanted_cruise.is_none_or(|cap| velocity.as_dvec3().length() <= cap * (1.0 + 2.0 * f32::EPSILON as f64))
        {
            self.sent_cruise = self.wanted_cruise;
            self.dispatch(&ClientMessage::Cruise { speed: self.sent_cruise.unwrap_or(0.0) });
        }
    }

    /// Ordinary moves are envelope-checked server-side; this is the sanctioned
    /// jump, which the server may still refuse with an [`Incoming::Position`]
    /// snap-back.
    pub fn send_teleport(&mut self, pos: DVec3) {
        // So the next `send_move` reports the post-teleport position promptly.
        self.last_sent = None;
        self.session.pending_teleport = Some(Instant::now());
        self.dispatch(&ClientMessage::Teleport { pos });
    }

    pub fn send_swing(&mut self) {
        self.dispatch(&ClientMessage::Swing);
    }

    /// The request id the eventual [`Incoming::EditAccepted`]/[`Incoming::EditRejected`]
    /// verdict will carry, or `None` when the spec will not be sent (too long, or the
    /// link is already dead). Nothing is left pending in that case. The expected
    /// revision counts in-flight requests on the cell, so a break-then-place chain
    /// lines up with the revisions its earlier requests will commit.
    pub fn send_edit(&mut self, x: i32, y: i32, z: i32, spec: Arc<str>) -> Option<u32> {
        if !self.session.alive || spec.len() > MAX_SPEC {
            return None;
        }
        let (req, expect) = self.session.track((x, y, z), PendingKind::Edit);
        self.dispatch(&ClientMessage::Edit { req, x, y, z, expect, spec });
        Some(req)
    }

    /// Tell the server the cruise speed when it changes. `None` sends 0, which ends cruise.
    /// Higher caps precede moves; lower caps follow the last move at the old speed.
    pub fn sync_cruise(&mut self, speed: Option<f64>) {
        let declared = speed.filter(|s| s.is_finite() && *s > 0.0);
        self.wanted_cruise = declared;
        if declared.unwrap_or(0.0) > self.sent_cruise.unwrap_or(0.0) {
            self.sent_cruise = declared;
            self.dispatch(&ClientMessage::Cruise { speed: declared.unwrap_or(0.0) });
        }
    }

    /// Send `bytes` on `channel`. False when the link is down, the name is illegal,
    /// or the payload exceeds the cap. Not throttled: the caller paces itself.
    /// The server stamps the sender.
    pub fn send_channel(&mut self, channel: &str, seq: u32, bytes: &[u8]) -> bool {
        if !self.session.alive {
            return false;
        }
        let Some(channel) = protocol::Channel::parse(channel) else { return false };
        let Ok(bytes) = ModBytes::try_from(bytes.to_vec()) else { return false };
        self.dispatch(&ClientMessage::ModData { channel, seq, bytes });
        true
    }

    /// True when `channel` has a frame waiting. A miss does not allocate.
    pub fn channel_pending(&self, channel: &str) -> bool {
        let map = self.mod_in.lock().unwrap_or_else(PoisonError::into_inner);
        map.get(channel).is_some_and(|ring| !ring.is_empty())
    }

    /// Take every waiting frame on `channel`. An empty or unknown channel returns
    /// an empty vec and allocates nothing.
    pub fn drain_channel(&mut self, channel: &str) -> Vec<(u32, u32, Vec<u8>)> {
        let mut map = self.mod_in.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(ring) = map.get_mut(channel) else { return Vec::new() };
        let mut out = Vec::with_capacity(ring.len());
        while let Some(frame) = ring.pop_front() {
            out.push((frame.sender, frame.seq, frame.bytes.into_vec()));
        }
        out
    }

    pub fn send_chat(&mut self, channel: u8, text: &str) {
        let text: Arc<str> = text.chars().take(MAX_CHAT).collect::<String>().into();
        self.dispatch(&ClientMessage::Chat { channel, text });
    }

    pub fn send_set_time(&mut self, day: f32) {
        self.dispatch(&ClientMessage::SetTime { day });
    }

    /// Use the held configuration as a tool on a cell; the server runs the law and answers with
    /// [`Incoming::ToolResult`] for the returned request id. Nothing is predicted: the law's
    /// outcome is the server's to decide.
    pub fn send_tool_use(&mut self, x: i32, y: i32, z: i32, tool_spec: Arc<str>) -> Option<u32> {
        if !self.session.alive || tool_spec.len() > MAX_SPEC {
            return None;
        }
        let (req, expect) = self.session.track((x, y, z), PendingKind::Tool);
        self.dispatch(&ClientMessage::ToolUse { req, x, y, z, expect, tool_spec });
        Some(req)
    }

    /// Hands the frame to the writer thread. A full queue leaves the link up
    /// (the caller keeps the edit pending). A gone writer is a dead link.
    fn dispatch(&mut self, msg: &ClientMessage) {
        if !self.session.alive {
            return;
        }
        let frame: Arc<[u8]> = msg.encode().into();
        let result = self.writer_tx.as_ref().map(|tx| tx.try_send(frame));
        note_send(result, &mut self.session.alive);
    }
}

fn note_send(result: Option<Result<(), TrySendError<Arc<[u8]>>>>, alive: &mut bool) {
    match result {
        Some(Ok(())) | Some(Err(TrySendError::Full(_))) => {}
        None | Some(Err(TrySendError::Disconnected(_))) => *alive = false,
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        // Close first so a writer blocked in a QUIC write unblocks, then drop
        // the sender so `recv` returns, then join. Joining before the close
        // deadlocks when the QUIC window is full.
        self.conn.close(0u32.into(), b"bye");
        drop(self.writer_tx.take());
        if let Some(handle) = self.writer.take() {
            let _ = handle.join();
        }
        graceful_close(&self.conn, &self.endpoint, &self.rt);
    }
}

/// Sends `CONNECTION_CLOSE` and blocks until quinn has actually transmitted
/// it, so the peer frees this connection promptly instead of waiting out its
/// idle timeout. The one place this lives — [`Connection::drop`] and any
/// other graceful QUIC shutdown (e.g. a test driving raw `quinn::Connection`s
/// below the app handshake) calls this instead of re-deriving it. Bounded so
/// a caller never hitches for long.
pub(crate) fn graceful_close(conn: &quinn::Connection, endpoint: &Endpoint, rt: &Runtime) {
    conn.close(0u32.into(), b"bye");
    // `wait_idle` waits for the close frame to actually send, unlike
    // `closed()`, which resolves before transmit.
    let endpoint = endpoint.clone();
    let _ = rt.block_on(async move { tokio::time::timeout(Duration::from_secs(1), endpoint.wait_idle()).await });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::server::{self, Config};

    /// End-to-end over loopback: two clients on one server see each other join, sync
    /// an edit, and exchange chat. Exercises the real socket path, not just the codec.
    #[test]
    fn two_clients_sync_over_loopback() {
        let handle = server::spawn(
            0,
            Config { password: "pw".into(), seed: 4242, ..Config::default() },
        )
        .unwrap();
        let port = handle.addr().port();

        let mut a = Connection::connect("127.0.0.1", port, "walnutty", "pw").unwrap();
        let mut b = Connection::connect("127.0.0.1", port, "guahlg", "pw").unwrap();
        assert_eq!(a.seed(), 4242);
        assert_eq!(b.seed(), 4242);
        assert_ne!(a.player_id(), b.player_id());

        // Give the join broadcasts time to land, then poll them in.
        thread::sleep(Duration::from_millis(150));
        a.poll();
        b.poll();
        assert_eq!(a.peer_count(), 1, "walnutty should see guahlg");
        assert_eq!(b.peer_count(), 1, "guahlg should see walnutty");
        assert_eq!(a.peer(b.player_id()).map(|p| &*p.name), Some("guahlg"));
        assert!(a.peer(a.player_id()).is_none(), "nobody is their own peer");

        // Walnutty edits a block right next to her spawn; guahlg should receive it.
        let s = a.spawn();
        let (bx, by, bz) = (
            crate::math::block_coord(s.x),
            crate::math::block_coord(s.y),
            crate::math::block_coord(s.z),
        );
        // Report position so the server's reach check passes, then edit.
        a.last_move = Instant::now() - HEARTBEAT; // force the throttle to send
        a.send_move(s, 0.0, 0.0, DQuat::IDENTITY, Vec3::ZERO, Face::PosY, Stance::Standing);
        let _ = a.send_edit(bx, by, bz, "air".into());

        thread::sleep(Duration::from_millis(150));
        let events = b.poll();
        assert!(
            events.iter().any(|e| matches!(e, Incoming::Edit { x, y, z, .. } if (*x, *y, *z) == (bx, by, bz))),
            "guahlg should receive walnutty's edit"
        );

        // Global chat reaches everyone regardless of distance.
        a.send_chat(crate::net::chat::GLOBAL, "hello");
        thread::sleep(Duration::from_millis(150));
        let events = b.poll();
        assert!(
            events.iter().any(|e| matches!(e, Incoming::Chat { text, .. } if &**text == "hello")),
            "guahlg should receive walnutty's global chat"
        );

        handle.stop();
    }

    /// End-to-end: two clients race a break on ONE cell. Exactly one is
    /// accepted; the other is rejected (its rollback signal) and converges on
    /// the winner's authoritative edit.
    #[test]
    fn racing_breaks_on_one_cell_yield_exactly_one_acceptance() {
        let handle =
            server::spawn(0, Config { seed: 4242, ..Config::default() }).unwrap();
        let port = handle.addr().port();
        let mut a = Connection::connect("127.0.0.1", port, "a", "").unwrap();
        let mut b = Connection::connect("127.0.0.1", port, "b", "").unwrap();
        thread::sleep(Duration::from_millis(150));
        a.poll();
        b.poll();

        // A block under a's spawn: both spawns scatter within a few blocks of
        // the origin, so it is within both players' edit reach.
        let s = a.spawn();
        let cell = (
            crate::math::block_coord(s.x),
            crate::math::block_coord(s.y - 3.0),
            crate::math::block_coord(s.z),
        );
        let _ = a.send_edit(cell.0, cell.1, cell.2, "air".into());
        let _ = b.send_edit(cell.0, cell.1, cell.2, "air".into());
        thread::sleep(Duration::from_millis(200));

        let mut accepted = 0;
        let mut rejected = 0;
        let mut loser_saw_authoritative_edit = false;
        for events in [a.poll(), b.poll()] {
            for event in events {
                match event {
                    Incoming::EditAccepted { .. } => accepted += 1,
                    Incoming::EditRejected { .. } => rejected += 1,
                    Incoming::Edit { x, y, z, .. } if (x, y, z) == cell => {
                        loser_saw_authoritative_edit = true;
                    }
                    _ => {}
                }
            }
        }
        assert_eq!((accepted, rejected), (1, 1), "exactly one break wins the cell");
        assert!(
            loser_saw_authoritative_edit,
            "the loser must receive the winner's authoritative edit"
        );
        handle.stop();
    }

    #[test]
    fn wrong_password_is_rejected() {
        let handle = server::spawn(0, Config { password: "secret".into(), seed: 1, ..Config::default() }).unwrap();
        let port = handle.addr().port();
        let err = match Connection::connect("127.0.0.1", port, "eve", "guess") {
            Ok(_) => panic!("a wrong password must be refused"),
            Err(e) => e,
        };
        assert!(err.to_string().to_lowercase().contains("password"), "got: {err}");
        handle.stop();
    }

    fn session() -> Session {
        Session::new(DVec3::new(0.5, 40.0, 0.5))
    }

    impl Session {
        /// One message, as a poll applies it.
        fn feed(&mut self, msg: ServerMessage) -> Vec<Incoming> {
            self.feed_all(std::iter::once(msg))
        }

        /// Messages in order, positions coalesced as a poll coalesces them.
        fn feed_all(&mut self, msgs: impl IntoIterator<Item = ServerMessage>) -> Vec<Incoming> {
            let mut out = Vec::new();
            for msg in msgs {
                self.apply(msg, &mut out);
            }
            coalesce_positions(&mut out);
            out
        }
    }

    #[test]
    fn snapshot_before_welcome_is_an_unexpected_handshake_reply() {
        let snap = ServerMessage::Snapshot { edits: vec![(1, 2, 3, 1, "air".into())] };
        let err = welcome_from(Some(snap)).unwrap_err();
        assert!(err.to_string().contains("unexpected"));
        let edit = ServerMessage::Edit { x: 0, y: 0, z: 0, rev: 1, spec: "air".into() };
        assert!(welcome_from(Some(edit)).unwrap_err().to_string().contains("unexpected"));
        assert!(welcome_from(Some(ServerMessage::SnapshotEnd)).unwrap_err().to_string().contains("unexpected"));
    }

    #[test]
    fn out_of_order_duplicate_poses_and_unknown_exit_never_panic() {
        let mut v = session();
        let pose = |id, x| ServerMessage::PeerPoses {
            poses: protocol::Poses {
                origin: DVec3::new(0.0, 40.0, 0.0),
                list: vec![protocol::PeerPose {
                    id,
                    pos: DVec3::new(x, 40.0, 0.0),
                    yaw: 0.0,
                    pitch: 0.0,
                    frame: DQuat::IDENTITY,
                    velocity: Vec3::ZERO,
                    up: Face::PosY,
                    stance: Stance::Standing,
                }],
            },
        };
        v.feed(pose(7, 3.0));
        assert!(v.peers.is_empty(), "a pose for an unknown id is ignored");
        v.feed(ServerMessage::PeerExited { id: 7 });
        v.feed(ServerMessage::PeerJoined { id: 7, name: "x".into() });
        v.feed(pose(7, 4.0));
        v.feed(pose(7, 4.0));
        v.feed(pose(7, 9.0));
        let p = v.peers.get(&7).unwrap();
        assert!(p.visible());
        assert_eq!(p.target.pos.x, 9.0);
        v.feed(ServerMessage::PeerExited { id: 99 });
        v.feed(ServerMessage::PeerLeft { id: 99 });
    }

    #[test]
    fn position_while_teleport_pending_keeps_the_last_authoritative_pose() {
        let mut v = session();
        let dest = DVec3::new(100.0, 40.0, 0.0);
        let old = DVec3::new(0.5, 40.0, 0.5);
        v.pending_teleport = Some(Instant::now());
        let at = |pos| ServerMessage::Position { pos, frame: DQuat::IDENTITY, up: Face::PosY };
        let events = v.feed_all([at(old), at(dest)]);
        match events.as_slice() {
            [Incoming::Position { pos, .. }] => assert_eq!(*pos, dest),
            other => panic!("expected one coalesced Position(dest), got {} events", other.len()),
        }
        assert!(v.pending_teleport.is_none());

        v.pending_teleport = Some(Instant::now());
        let events = v.feed(at(old));
        match events.as_slice() {
            [Incoming::Position { pos, .. }] => assert_eq!(*pos, old, "a lone Position is the /tp verdict"),
            other => panic!("expected refusal snap-back, got {} events", other.len()),
        }
        assert!(v.pending_teleport.is_none());
    }

    #[test]
    fn dropped_teleport_reply_releases_moves_after_one_heartbeat() {
        let t0 = Instant::now();
        let pending = Some(t0);
        assert!(teleport_hold_active(pending, t0), "a fresh hold must suppress moves");
        assert!(
            teleport_hold_active(pending, t0 + HEARTBEAT - Duration::from_nanos(1)),
            "the hold lasts the full heartbeat"
        );
        assert!(
            !teleport_hold_active(pending, t0 + HEARTBEAT),
            "a dropped echo must resume moves after one heartbeat"
        );
        assert!(
            !teleport_hold_active(None, t0),
            "a Position verdict still clears the hold immediately"
        );

        let handle = server::spawn(0, Config { seed: 1, ..Config::default() }).unwrap();
        let mut a = Connection::connect("127.0.0.1", handle.addr().port(), "a", "").unwrap();
        let pos = a.spawn();
        a.last_move = Instant::now() - HEARTBEAT;
        a.session.pending_teleport = Some(Instant::now());
        let still = |a: &mut Connection, pos| {
            a.send_move(pos, 0.0, 0.0, DQuat::IDENTITY, Vec3::ZERO, Face::PosY, Stance::Standing);
        };
        still(&mut a, pos);
        assert!(a.last_sent.is_none(), "a fresh hold must not send Move");
        a.session.pending_teleport = Some(Instant::now() - HEARTBEAT);
        still(&mut a, pos);
        assert!(a.last_sent.is_some(), "an expired hold must let Move through");
        handle.stop();
    }

    #[test]
    fn snapshot_edits_record_revisions_even_if_fed_directly() {
        let mut v = session();
        let events = v.feed(ServerMessage::Snapshot {
            edits: vec![(1, 2, 3, 4, "air".into())],
        });
        assert!(
            matches!(events.as_slice(), [Incoming::Mutation { x: 1, y: 2, z: 3, .. }]),
            "snapshot cells are world state, not a player's edit"
        );
        assert_eq!(v.cell_revs.get(&(1, 2, 3)), Some(&4));
    }

    #[test]
    fn inbox_disconnect_emits_disconnected_exactly_once() {
        let (tx, inbox) = mpsc::channel::<ServerMessage>();
        let mut v = session();
        drop(tx);
        let mut drain = || {
            let mut out = Vec::new();
            loop {
                match inbox.try_recv() {
                    Ok(_) => {}
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        v.alive = false;
                        break;
                    }
                }
            }
            v.silence(Instant::now(), &mut String::new(), &mut out);
            out
        };
        let first = drain();
        assert_eq!(first.len(), 1);
        assert!(matches!(first[0], Incoming::Disconnected { .. }));
        assert!(drain().is_empty(), "a second drain must not emit again");
        assert!(!v.alive);
    }

    #[test]
    fn dropped_connection_surfaces_disconnected_exactly_once() {
        let handle = server::spawn(0, Config { seed: 1, ..Config::default() }).unwrap();
        let mut a = Connection::connect("127.0.0.1", handle.addr().port(), "a", "").unwrap();
        a.conn.close(0u32.into(), b"bye");
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut got = 0u32;
        while Instant::now() < deadline {
            for e in a.poll() {
                if matches!(e, Incoming::Disconnected { .. }) {
                    got += 1;
                }
            }
            if got > 0 {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(got, 1, "the drop must surface Disconnected once");
        for _ in 0..8 {
            assert!(
                !a.poll().iter().any(|e| matches!(e, Incoming::Disconnected { .. })),
                "subsequent polls must stay quiet"
            );
        }
        handle.stop();
    }

    #[test]
    fn a_joining_peer_keeps_the_colour_of_its_name() {
        let mut v = session();
        v.feed(ServerMessage::PeerJoined { id: 4, name: "ada".into() });
        v.feed(ServerMessage::PeerJoined { id: 9, name: "bob".into() });
        for (id, name) in [(4, "ada"), (9, "bob")] {
            let peer = &v.peers[&id];
            assert_eq!((peer.id(), &*peer.name), (id, name));
            assert_eq!(peer.color(), presence::peer_color(name));
        }
    }

    /// Expiry keeps the survivors in send order and reports tool uses before the rejected edits
    /// they unwound.
    #[test]
    fn expiry_drops_tool_uses_in_order_and_keeps_fresh_requests() {
        let mut v = session();
        let old = Instant::now() - PENDING_TTL;
        let req = |req, cell, sent, kind| PendingReq { req, cell, expect: 0, base: 0, sent, kind };
        v.pending_edits = vec![
            req(1, (0, 0, 0), old, PendingKind::Tool),
            req(2, (0, 0, 0), Instant::now(), PendingKind::Edit),
            req(3, (5, 0, 0), old, PendingKind::Tool),
            req(4, (9, 0, 0), old, PendingKind::Edit),
        ];
        let mut out = Vec::new();
        v.expire(Instant::now(), &mut out);
        assert!(matches!(
            out.as_slice(),
            [Incoming::ToolResult { req: 1, .. }, Incoming::ToolResult { req: 3, .. }, Incoming::EditRejected { req: 4, restore: true }]
        ));
        assert_eq!(v.pending_edits.iter().map(|p| p.req).collect::<Vec<_>>(), [2]);
    }

    #[test]
    fn duplicate_peer_joined_announces_once() {
        let mut v = session();
        let msg = ServerMessage::PeerJoined { id: 4, name: "ada".into() };
        let first = v.feed(msg.clone());
        assert!(matches!(first.as_slice(), [Incoming::Joined { .. }]));
        assert!(v.feed(msg).is_empty(), "a second PeerJoined is not another join");
        assert_eq!(v.peers.len(), 1);
    }

    #[test]
    fn lowering_cruise_accounts_for_wire_velocity_rounding() {
        let handle = server::spawn(0, Config { seed: 1, worldgen: crate::world::generation::WorldgenKind::Flat, ..Config::default() }).unwrap();
        let mut conn = Connection::connect("127.0.0.1", handle.addr().port(), "ada", "").unwrap();
        let cap = 33_333_333.333333332;
        let velocity = Vec3::new(cap as f32, 0.0, 0.0);
        assert!(velocity.x as f64 > cap, "the wire rounds this target upwards");
        conn.sync_cruise(Some(cap * 10.0));
        conn.sync_cruise(Some(cap));
        conn.last_move = Instant::now() - MOVE_INTERVAL;
        conn.send_move(conn.spawn(), 0.0, 0.0, DQuat::IDENTITY, velocity, Face::PosY, Stance::Standing);
        assert_eq!(conn.sent_cruise, Some(cap), "a settled velocity lowers the declared cap");
        handle.stop();
    }

    #[test]
    fn a_confirmation_before_its_ack_does_not_double_count_the_prediction() {
        let cell = (1, 2, 3);
        let mut pending = vec![PendingReq { req: 1, cell, expect: 0, base: 0, sent: Instant::now(), kind: PendingKind::Edit }];
        assert_eq!(next_cell_revision(&pending, cell, 0), 1);
        assert_eq!(next_cell_revision(&pending, cell, 1), 1);
        pending.push(PendingReq { req: 2, cell, expect: 1, base: 0, sent: Instant::now(), kind: PendingKind::Edit });
        assert_eq!(next_cell_revision(&pending, cell, 1), 2);
        assert_eq!(next_cell_revision(&pending, cell, 5), 5);
        pending[1].kind = PendingKind::Rejected;
        assert_eq!(next_cell_revision(&pending, cell, 1), 1);
    }

    #[test]
    fn rejected_edit_after_a_tool_result_keeps_authoritative_content() {
        let mut v = session();
        let cell = (1, 2, 3);
        v.pending_edits.push(PendingReq { req: 1, cell, expect: 0, base: 0, sent: Instant::now(), kind: PendingKind::Tool });
        v.pending_edits.push(PendingReq { req: 2, cell, expect: 1, base: 0, sent: Instant::now(), kind: PendingKind::Edit });
        let out = v.feed(ServerMessage::ToolResult {
            req: 1, reacted: true, rev: 1, cell_spec: "air".into(), tool_spec: "spent".into(),
        });
        assert!(matches!(out.as_slice(), [Incoming::ToolResult { req: 1, .. }]));
        let out = v.feed(ServerMessage::EditAck { req: 2, accepted: false, rev: 1 });
        assert!(matches!(out.as_slice(), [Incoming::EditRejected { req: 2, restore: false }]));
    }

    #[test]
    fn rejected_edit_after_observing_authority_restores_its_current_baseline() {
        let mut v = session();
        let cell = (1, 2, 3);
        v.cell_revs.insert(cell, 1);
        v.pending_edits.push(PendingReq { req: 2, cell, expect: 1, base: 1, sent: Instant::now(), kind: PendingKind::Edit });
        let out = v.feed(ServerMessage::EditAck { req: 2, accepted: false, rev: 1 });
        assert!(matches!(out.as_slice(), [Incoming::EditRejected { req: 2, restore: true }]));
    }

    #[test]
    fn rejected_predictions_wait_for_the_chain_and_unwind_newest_first() {
        let cell = (1, 2, 3);
        for authoritative in [0, 1] {
            let mut v = session();
            v.cell_revs.insert(cell, authoritative);
            for req in 1..=2 {
                v.pending_edits.push(PendingReq { req, cell, expect: req - 1, base: 0, sent: Instant::now(), kind: PendingKind::Edit });
            }
            assert!(v.feed(ServerMessage::EditAck { req: 1, accepted: false, rev: authoritative }).is_empty());
            assert_eq!(v.pending_edits.len(), 2);
            let out = v.feed(ServerMessage::EditAck { req: 2, accepted: false, rev: authoritative });
            assert!(matches!(out.as_slice(), [Incoming::EditRejected { req: 2, restore: a }, Incoming::EditRejected { req: 1, restore: b }] if *a == (authoritative == 0) && a == b));
            assert!(v.pending_edits.is_empty());
        }
    }

    #[test]
    fn a_delayed_snapshot_cannot_revert_a_newer_cell_but_can_confirm_a_prediction() {
        let mut v = session();
        let cell = (1, 2, 3);
        v.cell_revs.insert(cell, 2);
        assert!(v.feed(ServerMessage::Snapshot { edits: vec![(1, 2, 3, 1, "air".into())] }).is_empty());
        assert_eq!(v.cell_revs[&cell], 2);
        let out = v.feed(ServerMessage::Snapshot { edits: vec![(1, 2, 3, 2, "air".into())] });
        assert!(matches!(out.as_slice(), [Incoming::Mutation { .. }]));
    }

    #[test]
    fn unanswered_edit_expires_as_rejected_and_unsent_spec_returns_no_id() {
        let mut v = session();
        v.pending_edits.push(PendingReq {
            req: 3,
            cell: (1, 2, 3),
            expect: 0,
            base: 0,
            sent: Instant::now() - Duration::from_secs(4),
            kind: PendingKind::Edit,
        });
        let mut out = Vec::new();
        v.expire(Instant::now(), &mut out);
        assert!(v.pending_edits.is_empty());
        assert!(matches!(out.as_slice(), [Incoming::EditRejected { req: 3, restore: true }]));

        v.pending_edits.push(PendingReq {
            req: 9,
            cell: (1, 2, 3),
            expect: 4,
            base: 4,
            sent: Instant::now() - PENDING_TTL,
            kind: PendingKind::Tool,
        });
        out.clear();
        v.cell_revs.insert((1, 2, 3), 4);
        v.expire(Instant::now(), &mut out);
        assert!(matches!(
            out.as_slice(),
            [Incoming::ToolResult { req: 9, reacted: false, .. }]
        ));

        let handle = server::spawn(0, Config { seed: 1, ..Config::default() }).unwrap();
        let mut conn = Connection::connect("127.0.0.1", handle.addr().port(), "ada", "").unwrap();
        let huge: Arc<str> = "x".repeat(MAX_SPEC + 1).into();
        assert!(conn.send_edit(0, 0, 0, huge).is_none());
        assert!(conn.session.pending_edits.is_empty());
        assert_eq!(conn.session.next_req, 0);
        // Never sent, so a server ack cannot remove it before the expiry path runs.
        conn.session.pending_edits.push(PendingReq {
            req: 7,
            cell: (1, 2, 3),
            expect: 0,
            base: 0,
            sent: Instant::now() - PENDING_TTL - Duration::from_millis(1),
            kind: PendingKind::Edit,
        });
        let events = conn.poll();
        assert!(
            events.iter().any(|e| matches!(e, Incoming::EditRejected { req: 7, restore: true })),
            "a few seconds with no ack rolls the edit back"
        );
        assert!(conn.session.pending_edits.is_empty());
        // A dead link tracks nothing: neither request is sent or left pending.
        conn.session.alive = false;
        let next = conn.session.next_req;
        assert!(conn.send_edit(1, 2, 3, "air".into()).is_none());
        assert!(conn.send_tool_use(1, 2, 3, "air".into()).is_none());
        assert!(conn.session.pending_edits.is_empty());
        assert_eq!(conn.session.next_req, next);
        handle.stop();
    }

    /// Two-process check: set `WATT_LIVE_ADDR=host:port` (and optional `WATT_LIVE_PW`)
    /// to join a real `watt_server`. Otherwise the server is spawned in-process.
    /// Joins, edits, moves faster than the old 80 m/s cap, and leaves.
    #[test]
    #[ignore]
    fn live_join_edits_fast_move_and_leaves() {
        let password = std::env::var("WATT_LIVE_PW").unwrap_or_default();
        let external = std::env::var("WATT_LIVE_ADDR").ok();
        let owned = if external.is_none() {
            Some(server::spawn(0, Config { seed: 7, ..Config::default() }).unwrap())
        } else {
            None
        };
        let (host, port, pw): (String, u16, String) = if let Some(addr) = &external {
            let (h, p) = addr.rsplit_once(':').expect("WATT_LIVE_ADDR is host:port");
            (h.to_string(), p.parse().expect("port"), password)
        } else {
            let handle = owned.as_ref().unwrap();
            ("127.0.0.1".into(), handle.addr().port(), String::new())
        };
        let mut conn = Connection::connect(&host, port, "live", &pw).expect("join");
        println!(
            "joined id {} seed {} worldgen {:?}",
            conn.player_id(),
            conn.seed(),
            conn.worldgen()
        );
        let s = conn.spawn();
        let (x, y, z) = (
            crate::math::block_coord(s.x),
            crate::math::block_coord(s.y),
            crate::math::block_coord(s.z),
        );
        let req = conn.send_edit(x, y, z, "air".into()).expect("edit sent");
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut edit_accepted = false;
        let mut edit_rejected = false;
        while Instant::now() < deadline && !edit_accepted && !edit_rejected {
            for e in conn.poll() {
                match e {
                    Incoming::EditAccepted { req: r } if r == req => edit_accepted = true,
                    Incoming::EditRejected { req: r, .. } if r == req => edit_rejected = true,
                    _ => {}
                }
            }
            if !edit_accepted && !edit_rejected {
                thread::sleep(Duration::from_millis(10));
            }
        }
        println!("edit req {req} accepted={edit_accepted} rejected={edit_rejected}");
        assert!(edit_accepted, "the spawn-cell edit must be accepted");
        conn.sync_cruise(Some(crate::player::MAX_SPEED));
        let dest = s + DVec3::new(500.0, 0.0, 0.0);
        conn.last_move = Instant::now() - HEARTBEAT;
        conn.send_move(
            dest,
            0.0,
            0.0,
            DQuat::IDENTITY,
            Vec3::new(crate::player::MAX_SPEED as f32, 0.0, 0.0),
            Face::PosY,
            Stance::Standing,
        );
        let deadline = Instant::now() + Duration::from_secs(1);
        let mut snapped = false;
        while Instant::now() < deadline {
            if conn.poll().iter().any(|e| matches!(e, Incoming::Position { .. })) {
                snapped = true;
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        println!("fast move snapped={snapped}");
        assert!(!snapped, "500 blocks at MAX_SPEED must not snap back");
        drop(conn);
        println!("left");
        if let Some(handle) = &owned {
            handle.stop();
        }
    }

    #[test]
    fn connect_to_an_unreachable_address_can_be_cancelled() {
        let pending = Connection::begin_connect("192.0.2.1", 9, "ada", "", &[]);
        pending.cancel();
        let started = Instant::now();
        let err = match pending.wait() {
            Err(err) => err,
            Ok(_) => panic!("cancel should fail the attempt"),
        };
        assert!(started.elapsed() < Duration::from_millis(500), "cancel took {:?}", started.elapsed());
        assert!(err.to_string().contains("cancelled"), "{err}");
    }

    #[test]
    fn silence_warns_once_then_gives_up_at_the_idle_timeout() {
        assert_eq!(INTERRUPTED, "connection interrupted");
        let t0 = Instant::now();
        assert!(matches!(link_silence(t0, t0 + SILENCE_WARN - Duration::from_nanos(1), false), LinkFate::Ok));
        assert!(matches!(link_silence(t0, t0 + SILENCE_WARN, false), LinkFate::Warn));
        assert!(matches!(link_silence(t0, t0 + SILENCE_WARN + Duration::from_secs(1), true), LinkFate::Ok));
        assert!(matches!(link_silence(t0, t0 + SILENCE_GIVE_UP, true), LinkFate::GiveUp));

        let mut v = session();
        v.last_heard = t0;
        let mut reason = String::new();
        let mut out = Vec::new();
        v.silence(t0 + SILENCE_WARN, &mut reason, &mut out);
        assert!(matches!(out.as_slice(), [Incoming::Interrupted]));
        assert!(v.warned && v.alive && reason.is_empty());
        out.clear();
        v.silence(t0 + SILENCE_WARN + Duration::from_secs(1), &mut reason, &mut out);
        assert!(out.is_empty(), "the same gap must not warn again");
        v.silence(t0 + SILENCE_GIVE_UP, &mut reason, &mut out);
        assert!(!v.alive);
        assert_eq!(reason, INTERRUPTED);
        assert!(matches!(out.as_slice(), [Incoming::Disconnected { reason }] if reason == INTERRUPTED));
        out.clear();

        // A real close recorded its phrase first: giving up keeps it.
        let mut v = session();
        v.last_heard = t0;
        v.alive = false;
        let mut reason = String::from("server shutting down");
        v.silence(t0 + SILENCE_GIVE_UP + Duration::from_secs(1), &mut reason, &mut out);
        assert_eq!(reason, "server shutting down");
        assert!(matches!(out.as_slice(), [Incoming::Disconnected { reason }] if reason == "server shutting down"));
    }

    #[test]
    fn a_pong_clears_the_silence_warning() {
        let mut v = session();
        v.warned = true;
        v.last_heard = Instant::now() - SILENCE_WARN;
        v.feed(ServerMessage::Pong { nonce: 1 });
        assert!(!v.warned);
        assert!(v.last_heard.elapsed() < Duration::from_millis(50));
    }

    /// A long join over a slow link: snapshot batches keep arriving and the Pong waits
    /// behind them. Every message is proof of life, so the link neither warns nor gives up.
    #[test]
    fn snapshot_batches_without_a_pong_keep_the_link_alive() {
        let mut v = session();
        let messages = [
            ServerMessage::Snapshot { edits: vec![(1, 2, 3, 1, "air".into())] },
            ServerMessage::Edit { x: 4, y: 5, z: 6, rev: 1, spec: "air".into() },
            ServerMessage::Time { day: 0.5, day_secs: 600.0 },
            ServerMessage::SnapshotEnd,
        ];
        for msg in messages {
            v.warned = true;
            v.last_heard = Instant::now() - (SILENCE_GIVE_UP - Duration::from_secs(1));
            v.feed(msg);
            assert!(!v.warned, "a message clears the silence warning");
            let mut out = Vec::new();
            let mut reason = String::new();
            v.silence(Instant::now() + Duration::from_secs(2), &mut reason, &mut out);
            assert!(v.alive && out.is_empty() && reason.is_empty(), "a link that just spoke is not silent");
        }
    }

    #[test]
    fn snapshot_end_marks_the_overlay_and_a_later_snapshot_does_not_clear_it() {
        let mut v = session();
        assert!(!v.snapshot_ready);
        v.feed(ServerMessage::Snapshot { edits: vec![(1, 2, 3, 1, "air".into())] });
        assert!(!v.snapshot_ready, "a snapshot batch is not the end of the overlay");
        v.feed(ServerMessage::SnapshotEnd);
        assert!(v.snapshot_ready);
        v.feed(ServerMessage::Snapshot { edits: vec![(4, 5, 6, 2, "air".into())] });
        assert!(v.snapshot_ready, "a reaction snapshot must not reopen the loading screen");

        let handle = server::spawn(0, Config { seed: 1, ..Config::default() }).unwrap();
        let mut conn = Connection::connect("127.0.0.1", handle.addr().port(), "ada", "").unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        while !conn.snapshot_ready() && Instant::now() < deadline {
            conn.poll();
            thread::sleep(Duration::from_millis(10));
        }
        assert!(conn.snapshot_ready(), "the server sends SnapshotEnd after the join overlay");
        handle.stop();
    }

    #[test]
    fn a_full_send_queue_keeps_the_link_and_a_dead_one_drops_it() {
        let (tx, rx) = mpsc::sync_channel::<Arc<[u8]>>(1);
        tx.try_send(Arc::from([0u8].as_slice())).unwrap();
        let mut alive = true;
        note_send(Some(tx.try_send(Arc::from([1u8].as_slice()))), &mut alive);
        assert!(alive, "a full queue must not kill the link");
        drop(rx);
        note_send(Some(tx.try_send(Arc::from([2u8].as_slice()))), &mut alive);
        assert!(!alive);
        note_send(None, &mut alive);
        assert!(!alive);
    }
}
