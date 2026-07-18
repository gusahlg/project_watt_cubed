//! The client side of multiplayer: a [`Connection`] the [`Game`](crate::game)
//! owns while playing on a server, hiding the socket behind a small poll-based
//! API. A background thread does the blocking reads and feeds a channel, so
//! the render loop never stalls on the network. Sends happen inline from the
//! game thread (tiny and infrequent). Position sends are throttled and
//! heartbeat so a standing-still player still proves they are alive.
use std::collections::{HashMap, VecDeque};
use std::net::{Ipv4Addr, ToSocketAddrs};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use quinn::{Endpoint, SendStream};
use tokio::runtime::Runtime;
use voxel_engine::DVec3;

use crate::net::protocol::{self, ClientMessage, ServerMessage};
use crate::net::{MAX_CHAT, MAX_SPEC, MAX_VOICE_PAYLOAD, PROTOCOL_VERSION, quic};
use crate::presence::{self, Eye, Stance, WireAction};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const MOVE_INTERVAL: Duration = Duration::from_millis(33);
/// So the server's idle timeout never reaps an active-but-idle player.
const HEARTBEAT: Duration = Duration::from_secs(1);
const PING_INTERVAL: Duration = Duration::from_secs(2);
/// Voice is loss-tolerant, so an overrun drops the OLDEST frame rather than
/// blocking or growing — stale audio is worthless.
const VOICE_RING_CAP: usize = 64;

/// `(speaker id, epoch, seq, opus payload)`.
type VoiceFrame = (u32, u32, u32, Vec<u8>);

/// Snapshotted so we can interpolate between two.
#[derive(Clone, Copy)]
struct Snapshot {
    pos: DVec3,
    yaw: f32,
    pitch: f32,
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
    /// Joins start hidden (the roster carries names, not positions); the
    /// first `PeerMove` reveals them and `PeerExited` hides them again — so a
    /// peer who wandered off isn't drawn frozen at their last heard pose.
    visible: bool,
    prev: Snapshot,
    target: Snapshot,
    recv_at: Instant,
    interval: Duration,
    distance: f64,
}

impl RemotePlayer {
    pub fn id(&self) -> u32 {
        self.id
    }

    pub fn visible(&self) -> bool {
        self.visible
    }
}

pub struct Rendered {
    /// The peer's eye position; drop to [`Feet`](presence::Feet) via
    /// [`Eye::feet`](presence::Eye::feet) with `stance` before rendering.
    pub pos: Eye,
    pub yaw: f32,
    pub pitch: f32,
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
            (horizontal(self.prev.pos, self.target.pos) / secs) as f32
        } else {
            0.0
        };
        Rendered {
            pos: Eye(pos),
            yaw,
            pitch,
            speed,
            phase: (self.distance * presence::STRIDE_FREQ) as f32,
            stance: self.target.stance,
        }
    }
}

/// Horizontal (xz-only) distance between two world positions; jumping/falling on
/// `pos.y` must not drive the gait.
fn horizontal(a: DVec3, b: DVec3) -> f64 {
    let (dx, dz) = (b.x - a.x, b.z - a.z);
    (dx * dx + dz * dz).sqrt()
}

/// Shortest-arc angular lerp: wrap `b - a` into `[-π, π]` so a turn across the
/// ±π seam takes the short way round instead of spinning the body.
fn lerp_angle(a: f32, b: f32, t: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    let delta = (b - a + PI).rem_euclid(TAU) - PI;
    a + delta * t
}

/// Peer presence and movement are applied inside [`Connection::poll`]; these
/// are what the game still has to handle.
pub enum Incoming {
    /// Stale revisions were already filtered out by the connection.
    Edit { x: i32, y: i32, z: i32, spec: Arc<str> },
    /// The server accepted our own edit `req`: prediction can forget it.
    EditAccepted { req: u32 },
    /// `restore` is set when no newer authoritative content has landed on the
    /// cell since, so the optimistic apply should roll back.
    EditRejected { req: u32, restore: bool },
    Position { pos: DVec3 },
    Chat { from_name: Arc<str>, channel: u8, text: Arc<str> },
    Joined { name: Arc<str> },
    Left { name: Arc<str> },
    /// Surfaced so the game can react (audio) beyond the local animator
    /// update already applied in `apply()`.
    PeerSwing { id: u32 },
    Time { day: f32, day_secs: f32 },
    Disconnected,
}

/// Dropping it closes the QUIC connection, which ends the reader thread and
/// signals the server that this player left.
pub struct Connection {
    conn: quinn::Connection,
    send: SendStream,
    /// Not needed to keep the connection alive (quinn's driver self-sustains
    /// while a connection is open), but required at [`Drop`] to `wait_idle` —
    /// flushing the close frame before the runtime is torn down.
    endpoint: Endpoint,
    rt: Arc<Runtime>,
    inbox: Receiver<ServerMessage>,
    /// Kept OUT of `inbox`: voice must not share the reliable, ordered event
    /// channel. `Arc<Mutex<VecDeque>>` rather than a second `mpsc` because std
    /// channels are unbounded and can't drop-oldest — the bound is the point.
    voice_in: Arc<Mutex<VecDeque<VoiceFrame>>>,
    player_id: u32,
    seed: i64,
    spawn: DVec3,
    peers: HashMap<u32, RemotePlayer>,
    alive: bool,
    // Throttling state for outbound moves.
    last_move: Instant,
    last_sent: Option<(DVec3, f32, f32, Stance)>,
    ping_sent: Option<(u32, Instant)>,
    ping_seq: u32,
    ping_ms: Option<u32>,
    /// CONFIRMED cell revisions from the server (snapshot, broadcasts, and
    /// accepted acks) — what future edit expectations are computed against.
    cell_revs: HashMap<(i32, i32, i32), u32>,
    /// Our in-flight edits as `(req, cell, expect)`, in send order. Counted
    /// per cell so a quick break-then-place chain expects the revisions its
    /// own earlier requests will commit.
    pending_edits: Vec<(u32, (i32, i32, i32), u32)>,
    next_req: u32,
}

impl Connection {
    /// `Err` carries a human-readable reason (bad address, refused, wrong
    /// password, version mismatch).
    pub fn connect(host: &str, port: u16, name: &str, password: &str) -> Result<Self, String> {
        let addr = (host, port)
            .to_socket_addrs()
            .map_err(|e| format!("bad address: {e}"))?
            .next()
            .ok_or_else(|| "address resolved to nothing".to_string())?;

        let rt = Arc::new(Runtime::new().map_err(|e| format!("runtime: {e}"))?);
        quic::install_crypto();
        let mut endpoint = {
            // Must run inside the runtime: construction spawns quinn's UDP driver.
            let _guard = rt.enter();
            Endpoint::client((Ipv4Addr::UNSPECIFIED, 0).into())
                .map_err(|e| format!("endpoint: {e}"))?
        };
        endpoint.set_default_client_config(quic::client_config());

        let conn = rt.block_on(async {
            let connecting = endpoint.connect(addr, "watt").map_err(|e| e.to_string())?;
            tokio::time::timeout(CONNECT_TIMEOUT, connecting)
                .await
                .map_err(|_| "connect timed out".to_string())?
                .map_err(|e| format!("could not reach {addr}: {e}"))
        })?;
        let (mut send, mut recv) =
            rt.block_on(conn.open_bi()).map_err(|e| format!("stream: {e}"))?;

        let hello = ClientMessage::Hello {
            protocol: PROTOCOL_VERSION,
            fingerprint: crate::net::content_fingerprint(),
            name: name.into(),
            password: password.into(),
        };
        rt.block_on(protocol::write_frame_async(&mut send, &hello.encode()))
            .map_err(|e| format!("send failed: {e}"))?;

        // The scratch Vec is reused across frames so the reader loop never allocates.
        let mut frame = Vec::new();
        rt.block_on(async {
            tokio::time::timeout(CONNECT_TIMEOUT, protocol::read_frame_async(&mut recv, &mut frame))
                .await
                .map_err(|_| "no reply: timed out".to_string())?
                .map_err(|e| format!("no reply: {e}"))
        })?;
        let (player_id, seed, spawn) = match ServerMessage::decode(&frame) {
            Some(ServerMessage::Welcome { player_id, seed, spawn }) => (player_id, seed, spawn),
            Some(ServerMessage::Reject { reason }) => return Err(reason.to_string()),
            _ => return Err("unexpected reply from server".to_string()),
        };

        let (tx, inbox) = mpsc::channel();
        let voice_in: Arc<Mutex<VecDeque<VoiceFrame>>> = Arc::new(Mutex::new(VecDeque::new()));
        let voice_reader = voice_in.clone();
        let reader_rt = rt.clone();
        thread::spawn(move || {
            while reader_rt.block_on(protocol::read_frame_async(&mut recv, &mut frame)).is_ok() {
                match ServerMessage::decode(&frame) {
                    // A hung-up game side stops draining but voice just rolls
                    // over; the connection close ends the loop.
                    Some(ServerMessage::PeerVoice { id, epoch, seq, payload }) => {
                        let mut ring = voice_reader.lock().unwrap_or_else(PoisonError::into_inner);
                        if ring.len() >= VOICE_RING_CAP {
                            ring.pop_front();
                        }
                        ring.push_back((id, epoch, seq, payload));
                    }
                    Some(msg) => {
                        if tx.send(msg).is_err() {
                            break; // The game side hung up.
                        }
                    }
                    None => continue, // Skip a junk frame rather than tear down.
                }
            }
        });

        Ok(Self {
            conn,
            send,
            endpoint,
            rt,
            inbox,
            voice_in,
            player_id,
            seed,
            spawn,
            peers: HashMap::new(),
            alive: true,
            last_move: Instant::now(),
            last_sent: None,
            ping_sent: None,
            ping_seq: 0,
            ping_ms: None,
            cell_revs: HashMap::new(),
            pending_edits: Vec::new(),
            next_req: 0,
        })
    }

    pub fn seed(&self) -> i64 {
        self.seed
    }
    pub fn spawn(&self) -> DVec3 {
        self.spawn
    }
    pub fn player_id(&self) -> u32 {
        self.player_id
    }
    pub fn is_alive(&self) -> bool {
        self.alive
    }
    pub fn peers(&self) -> impl Iterator<Item = &RemotePlayer> {
        self.peers.values()
    }
    pub fn peers_mut(&mut self) -> impl Iterator<Item = &mut RemotePlayer> {
        self.peers.values_mut()
    }
    pub fn ping_ms(&self) -> Option<u32> {
        self.ping_ms
    }

    /// Peer join/leave/move is applied to the local table here; edits and
    /// chat are returned for the game to handle.
    pub fn poll(&mut self) -> Vec<Incoming> {
        let due = match self.ping_sent {
            None => true,
            Some((_, at)) => at.elapsed() >= PING_INTERVAL,
        };
        if due && self.alive {
            self.ping_seq = self.ping_seq.wrapping_add(1);
            let nonce = self.ping_seq;
            self.dispatch(&ClientMessage::Ping { nonce });
            self.ping_sent = Some((nonce, Instant::now()));
        }

        let mut out = Vec::new();
        loop {
            match self.inbox.try_recv() {
                Ok(msg) => self.apply(msg, &mut out),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    if self.alive {
                        self.alive = false;
                        out.push(Incoming::Disconnected);
                    }
                    break;
                }
            }
        }
        out
    }

    fn apply(&mut self, msg: ServerMessage, out: &mut Vec<Incoming>) {
        match msg {
            ServerMessage::Snapshot { edits } => {
                for (x, y, z, rev, spec) in edits {
                    self.cell_revs.insert((x, y, z), rev);
                    out.push(Incoming::Edit { x, y, z, spec });
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
                let Some(at) = self.pending_edits.iter().position(|&(r, _, _)| r == req) else {
                    return;
                };
                let (_, cell, expect) = self.pending_edits.remove(at);
                if accepted {
                    let known = self.cell_revs.entry(cell).or_insert(0);
                    *known = (*known).max(rev);
                    out.push(Incoming::EditAccepted { req });
                } else {
                    // Restore our optimistic apply only if nothing newer has
                    // confirmed on the cell meanwhile (the race winner's Edit
                    // broadcast may land before or after this ack).
                    let confirmed = self.cell_revs.get(&cell).copied().unwrap_or(0);
                    out.push(Incoming::EditRejected { req, restore: confirmed <= expect });
                }
            }
            ServerMessage::Position { pos } => out.push(Incoming::Position { pos }),
            ServerMessage::Chat { from_name, channel, text, .. } => {
                out.push(Incoming::Chat { from_name, channel, text })
            }
            ServerMessage::Time { day, day_secs } => out.push(Incoming::Time { day, day_secs }),
            ServerMessage::PeerJoined { id, name } => {
                // prev == target on join: speed 0 and a stationary phase, no
                // Option<history> and no special-casing downstream. Hidden
                // until their first PeerMove carries a real pose.
                let spawn =
                    Snapshot { pos: self.spawn, yaw: 0.0, pitch: 0.0, stance: Stance::Standing };
                out.push(Incoming::Joined { name: name.clone() });
                self.peers.entry(id).or_insert(RemotePlayer {
                    id,
                    name,
                    anim: presence::Animator::default(),
                    visible: false,
                    prev: spawn,
                    target: spawn,
                    recv_at: Instant::now(),
                    interval: Duration::from_millis(0),
                    distance: 0.0,
                });
            }
            ServerMessage::PeerLeft { id } => {
                if let Some(p) = self.peers.remove(&id) {
                    out.push(Incoming::Left { name: p.name });
                }
            }
            ServerMessage::PeerMove { id, pos, yaw, pitch, stance } => {
                if let Some(p) = self.peers.get_mut(&id) {
                    let snapshot = Snapshot { pos, yaw, pitch, stance };
                    if p.visible {
                        p.interval = p.recv_at.elapsed();
                        p.prev = p.target;
                        p.target = snapshot;
                        p.distance += horizontal(p.prev.pos, p.target.pos);
                    } else {
                        // Re-entering interest range: snap, never lerp the
                        // avatar across the distance covered while hidden.
                        p.visible = true;
                        p.interval = Duration::from_millis(0);
                        p.prev = snapshot;
                        p.target = snapshot;
                    }
                    p.recv_at = Instant::now();
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
            ServerMessage::Pong { nonce } => {
                if let Some((sent_nonce, at)) = self.ping_sent {
                    if sent_nonce == nonce {
                        self.ping_ms = Some(at.elapsed().as_millis() as u32);
                    }
                }
            }
            ServerMessage::Reject { reason: _ } => {
                self.alive = false;
                out.push(Incoming::Disconnected);
            }
            // A second Welcome is meaningless mid-session.
            ServerMessage::Welcome { .. } => {}
            // Voice never reaches here: the reader thread routes PeerVoice into
            // the dedicated ring (see `connect`), not the `inbox` this drains.
            // The arm exists only to keep the match exhaustive.
            ServerMessage::PeerVoice { .. } => {}
        }
    }

    /// Cheap to call every frame; it only actually sends on the movement
    /// cadence or the heartbeat.
    pub fn send_move(&mut self, pos: DVec3, yaw: f32, pitch: f32, stance: Stance) {
        if !self.alive {
            return;
        }
        let elapsed = self.last_move.elapsed();
        let changed = self.last_sent != Some((pos, yaw, pitch, stance));
        let due = (changed && elapsed >= MOVE_INTERVAL) || elapsed >= HEARTBEAT;
        if !due {
            return;
        }
        self.last_move = Instant::now();
        self.last_sent = Some((pos, yaw, pitch, stance));
        self.dispatch(&ClientMessage::Move { pos, yaw, pitch, stance });
    }

    /// Ordinary moves are envelope-checked server-side; this is the sanctioned
    /// jump, which the server may still refuse with an [`Incoming::Position`]
    /// snap-back.
    pub fn send_teleport(&mut self, pos: DVec3) {
        // So the next `send_move` reports the post-teleport position promptly.
        self.last_sent = None;
        self.dispatch(&ClientMessage::Teleport { pos });
    }

    pub fn send_swing(&mut self) {
        self.dispatch(&ClientMessage::Swing);
    }

    /// Returns the request id the eventual [`Incoming::EditAccepted`]/
    /// [`Incoming::EditRejected`] verdict will carry. The expected revision
    /// counts our own in-flight edits on the cell, so a quick break-then-place
    /// chain lines up with the revisions its earlier requests will commit.
    pub fn send_edit(&mut self, x: i32, y: i32, z: i32, spec: Arc<str>) -> u32 {
        let cell = (x, y, z);
        let confirmed = self.cell_revs.get(&cell).copied().unwrap_or(0);
        let in_flight = self.pending_edits.iter().filter(|&&(_, c, _)| c == cell).count() as u32;
        let expect = confirmed + in_flight;
        self.next_req = self.next_req.wrapping_add(1);
        let req = self.next_req;
        if spec.len() > MAX_SPEC {
            return req; // never sent; no ack will come, nothing pends
        }
        self.pending_edits.push((req, cell, expect));
        self.dispatch(&ClientMessage::Edit { req, x, y, z, expect, spec });
        req
    }

    /// `seq` orders the local stream for the receiver's jitter buffer. Not
    /// throttled — capture already paces frames. An over-cap payload is
    /// dropped rather than sent; capture never produces one.
    pub fn send_voice(&mut self, seq: u32, payload: &[u8]) {
        if payload.len() > MAX_VOICE_PAYLOAD {
            return;
        }
        self.dispatch(&ClientMessage::Voice { seq, payload: payload.to_vec() });
    }

    /// Frames that overran the ring were already dropped (oldest first).
    pub fn drain_voice(&mut self) -> Vec<VoiceFrame> {
        self.voice_in.lock().unwrap_or_else(PoisonError::into_inner).drain(..).collect()
    }

    pub fn send_chat(&mut self, channel: u8, text: &str) {
        let text: Arc<str> = text.chars().take(MAX_CHAT).collect::<String>().into();
        self.dispatch(&ClientMessage::Chat { channel, text });
    }

    pub fn send_set_time(&mut self, day: f32) {
        self.dispatch(&ClientMessage::SetTime { day });
    }

    /// Blocks the game thread on the client runtime — sends are tiny and
    /// infrequent enough that this is fine.
    fn dispatch(&mut self, msg: &ClientMessage) {
        if !self.alive {
            return;
        }
        let encoded = msg.encode();
        if self.rt.block_on(protocol::write_frame_async(&mut self.send, &encoded)).is_err() {
            self.alive = false;
        }
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        // Keep the runtime alive until quinn has actually sent the
        // CONNECTION_CLOSE, so the server frees this player promptly instead
        // of waiting out its idle timeout — otherwise the reader thread
        // unblocks and drops the last runtime ref before the close frame goes
        // out. Bounded so leaving a world never hitches for long.
        self.conn.close(0u32.into(), b"bye");
        // `wait_idle` drives the endpoint driver until the close frame is
        // actually sent (unlike `closed()`, which resolves before transmit).
        let endpoint = self.endpoint.clone();
        let _ = self
            .rt
            .block_on(async move { tokio::time::timeout(Duration::from_secs(1), endpoint.wait_idle()).await });
    }
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
        assert_eq!(a.peers().count(), 1, "walnutty should see guahlg");
        assert_eq!(b.peers().count(), 1, "guahlg should see walnutty");

        // Walnutty edits a block right next to her spawn; guahlg should receive it.
        let s = a.spawn();
        let (bx, by, bz) = (
            crate::math::block_coord(s.x),
            crate::math::block_coord(s.y),
            crate::math::block_coord(s.z),
        );
        // Report position so the server's reach check passes, then edit.
        a.last_move = Instant::now() - HEARTBEAT; // force the throttle to send
        a.send_move(s, 0.0, 0.0, Stance::Standing);
        let _ = a.send_edit(bx, by, bz, "air".into());

        thread::sleep(Duration::from_millis(150));
        let events = b.poll();
        assert!(
            events.iter().any(|e| matches!(e, Incoming::Edit { x, y, z, .. } if (*x, *y, *z) == (bx, by, bz))),
            "guahlg should receive walnutty's edit"
        );

        // Global chat reaches everyone regardless of distance.
        a.send_chat(crate::net::chat::GLOBAL, "hello".into());
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
        assert!(err.to_lowercase().contains("password"), "got: {err}");
        handle.stop();
    }
}
