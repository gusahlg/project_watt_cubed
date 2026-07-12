//! The authoritative, headless multiplayer server: it owns the one true world
//! (seed + edit overlay) and the roster of connected players, and it never trusts a
//! client. It runs with no window, no GPU, and no chunk machinery — terrain is
//! procedural, so the server only tracks the *seed* and the sparse overlay of
//! *edits*, each an opaque portable block spec ([`save`](crate::save)). That makes it
//! tiny to run and lets it scale to many players on a cheap box.
//!
//! **Threading.** One accept thread; per client a blocking reader thread and a
//! bounded-queue writer thread, coordinated through a single [`Mutex`]-guarded
//! [`State`]. The lock is held only for short, allocation-light bursts; the hottest
//! path — move fan-out — snapshots its recipients under the lock and pushes to
//! their queues after releasing it. This comfortably serves hundreds of players;
//! past that the one global lock and the thread-per-client model are still the
//! ceiling (join/leave and global chat remain O(roster) under it), and an
//! event-loop rewrite would be the next step — called out honestly rather than
//! hidden.
//!
//! **Optimisation.** No voxel data is ever sent — a join transfers the seed plus the
//! edit overlay, and live play is just small position/edit/chat frames. Position
//! broadcasts are interest-managed (only players within [`INTEREST_RADIUS`] hear a
//! move) through a 2D bucket grid ([`State::grid`]): a move consults only the
//! mover's 3×3 bucket neighbourhood instead of scanning the roster, so the busiest
//! traffic costs O(nearby players) per move rather than O(everyone online).
//!
//! **Trust.** Joins are password-gated and version-checked; frames are size-capped by
//! the [`protocol`] framing; every client is rate-limited; and every edit is bounds-
//! and reach-validated against the sender's own reported position before it is
//! recorded.
use std::collections::HashMap;
use std::io::{self, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{SyncSender, TryRecvError, TrySendError, sync_channel};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use voxel_engine::DVec3;

use crate::math::block_coord;

use crate::block::registry::BlockRegistry;
use crate::net::protocol::{self, ClientMessage, ServerMessage};
use crate::net::{MAX_CHAT, MAX_NAME, MAX_SPEC, PROTOCOL_VERSION, chat};
use crate::presence::Stance;
use crate::world::generation::{SineHills, TerrainGenerator};

/// Poison-recovering lock: a client thread that panics while holding
/// the state must not take the whole server down with it — [`State`] is plain
/// data, valid at every point a panic could interrupt, so recovery is always
/// sound. The ONE place the recovery policy lives; call sites say
/// `lock_recover()` and can't drift back to a bare `.unwrap()`.
trait LockRecover<T> {
    fn lock_recover(&self) -> MutexGuard<'_, T>;
}

impl<T> LockRecover<T> for Mutex<T> {
    fn lock_recover(&self) -> MutexGuard<'_, T> {
        self.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Largest concurrent roster. A hard bound so a flood of connects can't spawn
/// unbounded threads.
const MAX_PLAYERS: usize = 256;
/// Depth of a client's outbound frame queue. A client that falls this far behind is
/// treated as unresponsive and dropped, so one slow peer can't grow memory without
/// bound.
const OUT_CAPACITY: usize = 1024;
/// A connection that sends nothing for this long is reaped (clients heartbeat well
/// under it). Also bounds how long a stalled handshake can squat a thread.
const IDLE_TIMEOUT: Duration = Duration::from_secs(30);
/// How long a client has to send its `Hello` before we hang up.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
/// Message budget per client per second; excess frames are dropped. Blunts flooding.
const RATE_LIMIT: u32 = 300;
/// A position update is only sent to players within this many world units of the
/// mover — nobody past render distance needs it.
const INTEREST_RADIUS: f64 = 160.0 * crate::math::PER_METER;
/// Squared once so the hot per-listener check in [`on_move`] needs no sqrt.
const INTEREST_RADIUS_SQ: f64 = INTEREST_RADIUS * INTEREST_RADIUS;
/// A client may edit a block at most this far from its own reported eye position;
/// farther edits are rejected as bogus. A little past the client's reach constant.
const EDIT_REACH: f64 = 8.0 * crate::math::PER_METER;
/// Edits are streamed to a joining client in batches this size, so a very built-up
/// world's snapshot never overflows a single frame's size cap. Derived from the
/// worst case per edit — 12 bytes x/y/z + 2-byte length prefix + [`MAX_SPEC`]
/// spec bytes — with headroom for the frame header.
const SNAPSHOT_BATCH: usize = (crate::net::MAX_FRAME - 64) / (12 + 2 + MAX_SPEC);
/// Average terrain height the generator oscillates around — matches the client's
/// [`World`](crate::world::World::new) so server spawn heights land on real ground.
const TERRAIN_BASE: f32 = 20.0;

/// The public knobs for a server. Built by the dedicated binary and the in-game host.
pub struct Config {
    /// Password every client must present. Empty means no password is required.
    pub password: String,
    /// The world seed all clients generate their terrain from.
    pub seed: i64,
}

/// Immutable per-server context shared with every connection handler: the auth
/// password, the seed, and just enough of the generator to place spawns on ground.
struct Ctx {
    password: String,
    seed: i64,
    generator: SineHills,
}

/// One connected player as the server tracks them.
struct PlayerHandle {
    name: String,
    pos: DVec3,
    yaw: f32,
    pitch: f32,
    stance: Stance,
    /// Outbound queue drained by this client's writer thread.
    out: SyncSender<Arc<[u8]>>,
    /// A clone of the socket, kept only to force-close a misbehaving client.
    kick: TcpStream,
    /// False until this player's Welcome/Snapshot bootstrap is fully queued.
    /// Broadcasters must not push into a not-yet-ready queue: a racing frame
    /// would beat Welcome onto the wire (failing the client handshake) or
    /// interleave between snapshot batches (a stale batch would then revert a
    /// newer edit). Instead they buffer into `backlog`, drained in order once
    /// the bootstrap is done — so mid-join edits still arrive, AFTER the
    /// snapshot they must override.
    ready: bool,
    backlog: Vec<Arc<[u8]>>,
}

/// Most frames a joining player can accumulate while their bootstrap queues.
/// Overflow marks them slow (kicked) — matching the outbound-queue policy.
const BOOTSTRAP_BACKLOG: usize = 256;

/// The single piece of shared, mutable server state: the authoritative edit overlay
/// (coordinate → portable block spec), the player roster, and the interest grid
/// that indexes the roster by position.
struct State {
    edits: HashMap<(i32, i32, i32), String>,
    players: HashMap<u32, PlayerHandle>,
    /// Broad-phase interest grid: bucket key → ids of the players standing in it,
    /// keyed by [`bucket_of`] — `(floor(x / INTEREST_RADIUS), floor(z /
    /// INTEREST_RADIUS))`. Buckets are exactly one radius wide, so anyone within
    /// [`INTEREST_RADIUS`] of a mover lives in the mover's 3×3 bucket
    /// neighbourhood; [`on_move`] collects candidates there and still applies the
    /// exact per-player distance check, so the grid only narrows the *candidate*
    /// set, never the audience. Deliberately 2D: interest mirrors render distance,
    /// which is horizontal, and players spread across a sliver of y compared to a
    /// 160-unit radius — a y axis would add bucket churn from every jump and fall
    /// while barely shrinking candidate sets. Ignoring y can only *widen* the
    /// candidate set (3D distance ≥ horizontal distance), never miss a listener.
    /// Invariant: exactly one entry per connected player, updated under the same
    /// lock hold as the roster/position change it mirrors; empty buckets are
    /// removed eagerly so churn can never leak keys.
    grid: HashMap<(i32, i32), Vec<u32>>,
    next_id: u32,
    /// The shared world time as a `[0,1)` day fraction. Set by any client's
    /// `/time`, echoed to everyone, and handed to each joiner so a session shares
    /// one clock. The server does not itself advance it — clients tick locally.
    day: f32,
}

impl State {
    /// Add `id` to the grid bucket containing `pos`. Must run under the same lock
    /// hold as the roster/position change it mirrors, or the grid drifts.
    fn grid_insert(&mut self, id: u32, pos: DVec3) {
        self.grid.entry(bucket_of(pos)).or_default().push(id);
    }

    /// Remove `id` from the grid bucket containing `pos`, dropping the bucket when
    /// it empties so long-running churn can never accumulate dead keys.
    fn grid_remove(&mut self, id: u32, pos: DVec3) {
        let key = bucket_of(pos);
        if let Some(bucket) = self.grid.get_mut(&key) {
            bucket.retain(|&p| p != id);
            if bucket.is_empty() {
                self.grid.remove(&key);
            }
        }
    }
}

/// The interest-grid bucket containing `pos`. Goes through [`block_coord`]'s
/// clamped floor (not truncation) so negative coordinates bucket consistently
/// and a hostile-but-finite huge coordinate can't overflow the i32 key —
/// insert and remove share this one mapping, so the grid stays consistent.
fn bucket_of(pos: DVec3) -> (i32, i32) {
    (block_coord(pos.x / INTEREST_RADIUS), block_coord(pos.z / INTEREST_RADIUS))
}

/// A running server. [`stop`](ServerHandle::stop)ping it takes the listener down;
/// existing clients finish on their own.
pub struct ServerHandle {
    shutdown: Arc<AtomicBool>,
    addr: SocketAddr,
    /// Test-only window into the shared state, for grid-leak assertions.
    #[cfg(test)]
    state: Arc<Mutex<State>>,
}

impl ServerHandle {
    /// The address the server is actually listening on (the resolved port when the
    /// caller bound to port 0).
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Stop accepting new connections. Existing clients finish on their own sockets.
    pub fn stop(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }

    /// Total player entries across every interest-grid bucket. Must always equal
    /// the roster size — the leak the churn test guards against.
    #[cfg(test)]
    fn grid_entries(&self) -> usize {
        self.state.lock_recover().grid.values().map(Vec::len).sum()
    }

    /// Number of live grid buckets. Empty buckets are removed eagerly, so this
    /// must return to zero whenever the roster empties.
    #[cfg(test)]
    fn grid_buckets(&self) -> usize {
        self.state.lock_recover().grid.len()
    }
}

/// Bind `port` and start serving in the background, returning a handle with the
/// resolved address. Bind to port 0 to let the OS pick a free port.
pub fn spawn(port: u16, config: Config) -> io::Result<ServerHandle> {
    let listener = TcpListener::bind(("0.0.0.0", port))?;
    let addr = listener.local_addr()?;
    let shutdown = Arc::new(AtomicBool::new(false));

    // Resolve the generator's palette once so spawn heights match the client terrain.
    let mut registry = BlockRegistry::with_builtins();
    let ctx = Arc::new(Ctx {
        password: config.password,
        seed: config.seed,
        generator: SineHills::new(&mut registry, TERRAIN_BASE, config.seed),
    });
    let shared = Arc::new(Mutex::new(State {
        edits: HashMap::new(),
        players: HashMap::new(),
        grid: HashMap::new(),
        next_id: 1,
        day: 0.3,
    }));

    #[cfg(test)]
    let state = shared.clone();
    let accept_shutdown = shutdown.clone();
    thread::spawn(move || accept_loop(listener, shared, ctx, accept_shutdown));

    Ok(ServerHandle {
        shutdown,
        addr,
        #[cfg(test)]
        state,
    })
}

/// Bind and serve on the current thread until the process exits — the dedicated
/// server's entry point.
pub fn run(port: u16, config: Config) -> io::Result<()> {
    let handle = spawn(port, config)?;
    println!("watt-cubed server listening on {}", handle.addr());
    // The accept loop runs on its own thread; park this one so the process lives.
    loop {
        thread::sleep(Duration::from_secs(3600));
    }
}

/// Accept connections until asked to stop, handing each to its own handler thread.
fn accept_loop(listener: TcpListener, shared: Arc<Mutex<State>>, ctx: Arc<Ctx>, shutdown: Arc<AtomicBool>) {
    // Non-blocking accept so the loop can notice `stop()` between connections.
    let _ = listener.set_nonblocking(true);
    while !shutdown.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, addr)) => {
                let _ = stream.set_nonblocking(false);
                let _ = stream.set_nodelay(true);
                let shared = shared.clone();
                let ctx = ctx.clone();
                thread::spawn(move || {
                    // A dropped connection is routine; the error is the disconnect cause.
                    let _ = handle_client(stream, addr, shared, ctx);
                });
            }
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(50));
            }
            Err(_) => thread::sleep(Duration::from_millis(200)),
        }
    }
}

/// Drive one client: authenticate, register, stream the world snapshot, then relay
/// its messages until it disconnects, tidying up on the way out.
fn handle_client(stream: TcpStream, addr: SocketAddr, shared: Arc<Mutex<State>>, ctx: Arc<Ctx>) -> io::Result<()> {
    // The first frame must be a valid, authenticated Hello within the handshake window.
    // Buffered reads (one buffered read per frame, not two syscalls) plus a scratch
    // Vec reused for every frame this client ever sends — no per-frame allocation.
    stream.set_read_timeout(Some(HANDSHAKE_TIMEOUT))?;
    let mut reader = io::BufReader::new(stream.try_clone()?);
    let mut frame = Vec::new();
    protocol::read_frame(&mut reader, &mut frame)?;
    let name = match ClientMessage::decode(&frame) {
        Some(ClientMessage::Hello { protocol, name, password }) => {
            if protocol != PROTOCOL_VERSION {
                reject(&stream, "protocol version mismatch");
                return Ok(());
            }
            if password != ctx.password {
                reject(&stream, "wrong password");
                return Ok(());
            }
            clean_name(&name)
        }
        _ => {
            reject(&stream, "expected hello");
            return Ok(());
        }
    };

    // Authenticated: switch to the idle timeout and wire up the writer.
    stream.set_read_timeout(Some(IDLE_TIMEOUT))?;
    let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let writer_stream = stream.try_clone()?;
    let writer_shutdown = stream.try_clone()?;
    let writer = thread::spawn(move || {
        // Buffered, flushed once per drained batch: block for the first frame, then
        // opportunistically drain whatever else queued up before paying one flush —
        // a burst of broadcasts costs one syscall instead of one per message.
        //
        // Any write error shuts the socket down so the reader thread unblocks
        // and the player is cleaned up, instead of silently ghosting them.
        let mut w = io::BufWriter::new(writer_stream);
        let fail = |s: &TcpStream| {
            let _ = s.shutdown(Shutdown::Both);
        };
        while let Ok(frame) = rx.recv() {
            if protocol::write_frame(&mut w, &frame).is_err() {
                return fail(&writer_shutdown);
            }
            loop {
                match rx.try_recv() {
                    Ok(frame) => {
                        if protocol::write_frame(&mut w, &frame).is_err() {
                            return fail(&writer_shutdown);
                        }
                    }
                    Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
                }
            }
            if w.flush().is_err() {
                return fail(&writer_shutdown);
            }
        }
    });

    // Register the player and gather what the newcomer needs to bootstrap. Done in
    // one locked scope so the id, spawn, and roster it sees are all consistent.
    let id;
    let spawn;
    let world_day;
    let existing: Vec<(u32, String, DVec3, f32, f32, Stance)>;
    let snapshot: Vec<(i32, i32, i32, String)>;
    {
        let mut state = shared.lock_recover();
        world_day = state.day;
        if state.players.len() >= MAX_PLAYERS {
            drop(state);
            reject(&stream, "server full");
            return Ok(());
        }
        id = state.next_id;
        state.next_id += 1;
        spawn = spawn_point(&ctx.generator, id);

        existing = state
            .players
            .iter()
            .map(|(&pid, h)| (pid, h.name.clone(), h.pos, h.yaw, h.pitch, h.stance))
            .collect();
        snapshot = state
            .edits
            .iter()
            .map(|(&(x, y, z), spec)| (x, y, z, spec.clone()))
            .collect();

        state.players.insert(
            id,
            PlayerHandle {
                name: name.clone(),
                pos: spawn,
                yaw: 0.0,
                pitch: 0.0,
                stance: Stance::Standing,
                out: out.clone(),
                kick: stream.try_clone()?,
                ready: false,
                backlog: Vec::new(),
            },
        );
        // Same lock hold as the roster insert, so the grid never lags the roster.
        state.grid_insert(id, spawn);
    }
    println!("[+] {name} joined as #{id} from {addr} ({} online)", online(&shared));

    // Bootstrap the newcomer: who they are, the world edits, and who else is here.
    // These sends BLOCK (we're on this client's own handler thread): a built-up
    // world or big roster can exceed the outbound queue, and dropping bootstrap
    // frames would ghost the join.
    send_blocking(&out, &ServerMessage::Welcome { player_id: id, seed: ctx.seed, spawn });
    for batch in snapshot.chunks(SNAPSHOT_BATCH) {
        send_blocking(&out, &ServerMessage::Snapshot { edits: batch.to_vec() });
    }
    // Hand the newcomer the shared clock so their sky matches everyone else's.
    send_blocking(&out, &ServerMessage::Time { day: world_day });
    for (pid, pname, ppos, pyaw, ppitch, pstance) in existing {
        send_blocking(&out, &ServerMessage::PeerJoined { id: pid, name: pname });
        send_blocking(
            &out,
            &ServerMessage::PeerMove {
                id: pid,
                pos: ppos,
                yaw: pyaw,
                pitch: ppitch,
                stance: pstance,
            },
        );
    }
    // Bootstrap queued: go live. Frames broadcast during the bootstrap window
    // were buffered; drain them in order (they postdate the snapshot) and only
    // then let broadcasters push directly.
    {
        let mut state = shared.lock_recover();
        let mut slow = false;
        if let Some(h) = state.players.get_mut(&id) {
            for frame in std::mem::take(&mut h.backlog) {
                if h.out.try_send(frame).is_err() {
                    slow = true;
                    break;
                }
            }
            h.ready = true;
        }
        if slow {
            kick_slow(&state, &[id]);
        }
    }

    // Announce the newcomer to everyone already connected.
    broadcast_all(&shared, &ServerMessage::PeerJoined { id, name: name.clone() }, Some(id));

    // Relay loop with a light per-second rate limiter.
    let mut window = Instant::now();
    let mut count: u32 = 0;
    loop {
        if protocol::read_frame(&mut reader, &mut frame).is_err() {
            break; // EOF, timeout, or a malformed length: the client is gone.
        }

        if window.elapsed() >= Duration::from_secs(1) {
            window = Instant::now();
            count = 0;
        }
        count += 1;
        if count > RATE_LIMIT {
            continue; // Over budget this second — drop the frame rather than serve a flood.
        }

        let Some(msg) = ClientMessage::decode(&frame) else {
            continue; // Junk frame; ignore it.
        };
        match msg {
            ClientMessage::Move { pos, yaw, pitch, stance } => {
                on_move(&shared, id, pos, yaw, pitch, stance)
            }
            ClientMessage::Edit { x, y, z, spec } => on_edit(&shared, id, x, y, z, &spec),
            ClientMessage::Chat { channel, text } => on_chat(&shared, id, channel, &text),
            ClientMessage::SetTime { day } => on_set_time(&shared, day),
            // Clients ignore swings for unknown peers, so broadcast to
            // everyone-but-sender is safe.
            ClientMessage::Swing => {
                broadcast_all(&shared, &ServerMessage::PeerSwing { id }, Some(id))
            }
            ClientMessage::Ping { nonce } => {
                let state = shared.lock_recover();
                if let Some(h) = state.players.get(&id) {
                    // Best-effort: a full queue drops the probe, and the
                    // client simply re-sends on its interval.
                    let _ = h.out.try_send(ServerMessage::Pong { nonce }.encode().into());
                }
            }
            ClientMessage::Hello { .. } => {} // Already authenticated; ignore repeats.
        }
    }

    // Cleanup: drop the player (which frees the writer), close the socket, tell peers.
    {
        let mut state = shared.lock_recover();
        if let Some(h) = state.players.remove(&id) {
            // The handle's pos is the last committed one, so it names the exact
            // bucket the grid still holds this id under.
            state.grid_remove(id, h.pos);
        }
    }
    let _ = stream.shutdown(Shutdown::Both);
    drop(out);
    let _ = writer.join();
    broadcast_all(&shared, &ServerMessage::PeerLeft { id }, None);
    println!("[-] {name} (#{id}) left ({} online)", online(&shared));
    Ok(())
}

/// Apply a validated position update and fan it out to interested players only.
///
/// Runs in two phases to keep the global lock hold minimal. Locked: commit the
/// move, keep the grid current, and snapshot the recipients' senders (cheap
/// `SyncSender` clones — one `Arc` bump each). Unlocked: the `try_send`s. Kick
/// semantics are unchanged: `try_send` never blocked even under the lock, the
/// same failures land the same ids on the kick list, [`kick_slow`] already
/// tolerates ids that disconnected in the unlocked window, and ids are never
/// reused, so a late kick can't hit the wrong player.
fn on_move(shared: &Arc<Mutex<State>>, id: u32, pos: DVec3, yaw: f32, pitch: f32, stance: Stance) {
    // Ignore non-finite coordinates outright (a NaN would poison distance checks
    // and the grid keys).
    if !pos.x.is_finite() || !pos.y.is_finite() || !pos.z.is_finite() {
        return;
    }
    // Encode before locking — the frame doesn't depend on shared state.
    let frame: Arc<[u8]> =
        ServerMessage::PeerMove { id, pos, yaw, pitch, stance }.encode().into();

    let mut recipients: Vec<(u32, SyncSender<Arc<[u8]>>)> = Vec::new();
    {
        let mut state = shared.lock_recover();
        let old = match state.players.get_mut(&id) {
            Some(h) => {
                let old = h.pos;
                h.pos = pos;
                h.yaw = yaw;
                h.pitch = pitch;
                h.stance = stance;
                old
            }
            None => return, // Unreachable while the handler thread lives; be safe.
        };
        // Keep the grid honest before collecting from it.
        let (from, to) = (bucket_of(old), bucket_of(pos));
        if from != to {
            state.grid_remove(id, old);
            state.grid_insert(id, pos);
        }
        // Broad phase: buckets are one INTEREST_RADIUS wide, so every player in
        // range is somewhere in the mover's 3×3 neighbourhood. Exact phase: the
        // same per-player squared-distance check as ever — the grid narrows the
        // candidate set, never the audience. `wrapping_add` so a hostile position
        // at the i32 edge can't overflow; a wrapped key at worst nominates
        // candidates the exact check rejects.
        for dx in -1..=1i32 {
            for dz in -1..=1i32 {
                let key = (to.0.wrapping_add(dx), to.1.wrapping_add(dz));
                let Some(bucket) = state.grid.get(&key) else { continue };
                for &pid in bucket {
                    if pid == id {
                        continue;
                    }
                    let Some(h) = state.players.get(&pid) else { continue };
                    // Bootstrapping joiners skip moves: their roster snapshot
                    // carries current positions, and the next move re-delivers.
                    if !h.ready {
                        continue;
                    }
                    // Squared-distance compare: per candidate per move, skip the sqrt.
                    if h.pos.distance_squared(pos) > INTEREST_RADIUS_SQ {
                        continue;
                    }
                    recipients.push((pid, h.out.clone()));
                }
            }
        }
    }

    // Unlocked fan-out; a full (or hung-up) queue marks its owner for the kick pass.
    let mut slow = Vec::new();
    for (pid, out) in &recipients {
        if out.try_send(frame.clone()).is_err() {
            slow.push(*pid);
        }
    }
    if !slow.is_empty() {
        kick_slow(&shared.lock_recover(), &slow);
    }
}

/// Validate and record a block edit, then broadcast it to EVERY player —
/// including the sender — so all overlays converge on the server's ordering.
///
/// Echoing the edit back to its own sender is deliberate. When two players race
/// edits on the same cell (place vs break), the order the server records them
/// in is the one truth, and every client converges by applying the server's
/// stream in that order; the sender's optimistic local apply is then either
/// confirmed by its own echo or overwritten by the later edit. Excluding the
/// sender (the old behavior) left the two editors permanently disagreeing
/// about the cell whenever their edits raced.
fn on_edit(shared: &Arc<Mutex<State>>, id: u32, x: i32, y: i32, z: i32, spec: &str) {
    // Y is unbounded now (infinite world height/depth); reach is the real gate.
    if spec.len() > MAX_SPEC {
        return;
    }
    let mut state = shared.lock_recover();
    // Reach check against the editor's own reported position — no reaching across
    // the map.
    let Some(h) = state.players.get(&id) else { return };
    let target = DVec3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5);
    if h.pos.distance(target) > EDIT_REACH {
        return;
    }
    // The overlay stores the portable spec verbatim; the server never resolves it.
    state.edits.insert((x, y, z), spec.to_string());
    let msg = ServerMessage::Edit { x, y, z, spec: spec.to_string() };
    broadcast(&mut state, &msg, |_, _| true);
}

/// Relay a chat line to its audience: proximity for local, everyone for global.
fn on_chat(shared: &Arc<Mutex<State>>, id: u32, channel: u8, text: &str) {
    let text = clean_chat(text);
    if text.is_empty() {
        return;
    }
    let mut state = shared.lock_recover();
    let Some(sender) = state.players.get(&id) else { return };
    let from_name = sender.name.clone();
    let origin = sender.pos;
    let channel = if channel == chat::GLOBAL { chat::GLOBAL } else { chat::LOCAL };
    println!("<{from_name}> {text}");
    let msg = ServerMessage::Chat { from_id: id, from_name, channel, text };
    broadcast(&mut state, &msg, |_, h| {
        channel == chat::GLOBAL || h.pos.distance(origin) <= chat::RADIUS
    });
}

/// Record and relay a `/time` change: store it as the shared clock so joiners
/// inherit it, then echo it to everyone (the sender included, so all clocks agree).
/// A non-finite value is ignored rather than poisoning the shared time.
fn on_set_time(shared: &Arc<Mutex<State>>, day: f32) {
    if !day.is_finite() {
        return;
    }
    let day = day.rem_euclid(1.0);
    let mut state = shared.lock_recover();
    state.day = day;
    broadcast(&mut state, &ServerMessage::Time { day }, |_, _| true);
}

/// Send one message to every player matching `want`, encoding it just once. Players
/// whose queue is full are force-closed (they've fallen too far behind).
fn broadcast(state: &mut State, msg: &ServerMessage, want: impl Fn(u32, &PlayerHandle) -> bool) {
    let frame: Arc<[u8]> = msg.encode().into();
    let mut slow = Vec::new();
    for (&pid, h) in state.players.iter_mut() {
        if !want(pid, h) {
            continue;
        }
        if !h.ready {
            // Bootstrapping: buffer so the frame lands AFTER the snapshot.
            if h.backlog.len() < BOOTSTRAP_BACKLOG {
                h.backlog.push(frame.clone());
            } else {
                slow.push(pid);
            }
            continue;
        }
        match h.out.try_send(frame.clone()) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => slow.push(pid),
            Err(TrySendError::Disconnected(_)) => {}
        }
    }
    kick_slow(state, &slow);
}

/// Broadcast to everyone, optionally skipping one id (the originator).
fn broadcast_all(shared: &Arc<Mutex<State>>, msg: &ServerMessage, except: Option<u32>) {
    let mut state = shared.lock_recover();
    broadcast(&mut state, msg, |pid, _| Some(pid) != except);
}

/// Force-close clients that couldn't keep up. Their reader threads then wake, error,
/// and run the normal cleanup path (emitting `PeerLeft`).
fn kick_slow(state: &State, ids: &[u32]) {
    for id in ids {
        if let Some(h) = state.players.get(id) {
            let _ = h.kick.shutdown(Shutdown::Both);
        }
    }
}

/// Queue one message, waiting for space. Only safe on the receiving client's
/// own handler thread (used for the join bootstrap, which must not drop frames).
fn send_blocking(out: &SyncSender<Arc<[u8]>>, msg: &ServerMessage) {
    let frame: Arc<[u8]> = msg.encode().into();
    let _ = out.send(frame);
}

/// Reply with a rejection and let the socket close.
fn reject(stream: &TcpStream, reason: &str) {
    let mut w = stream;
    let _ = protocol::write_frame(&mut w, &ServerMessage::Reject { reason: reason.into() }.encode());
    println!("[x] rejected a connection: {reason}");
}

/// A spawn point just above a dry-land surface near the origin, scattered a little
/// per id so players don't stack on the exact same block. Scans outward for the
/// first column above sea level so nobody spawns on the seabed.
fn spawn_point(generator: &SineHills, id: u32) -> DVec3 {
    // A cheap deterministic scatter on a small grid around origin.
    let sx = (id % 8) as i32 - 3;
    let sz = ((id / 8) % 8) as i32 - 3;
    let sea = generator.sea_level();
    // Spiral outward from the scattered start until a land column is found.
    for r in 0..64 {
        for (dx, dz) in [(r, 0), (0, r), (-r, 0), (0, -r), (r, r), (-r, -r), (r, -r), (-r, r)] {
            let (x, z) = (sx + dx * 8, sz + dz * 8);
            let h = generator.height(x, z);
            if h > sea {
                return DVec3::new(x as f64 + 0.5, h as f64 + 3.0, z as f64 + 0.5);
            }
        }
    }
    // Fallback: sit on the water surface at the scattered origin.
    let h = generator.height(sx, sz).max(sea);
    DVec3::new(sx as f64 + 0.5, h as f64 + 3.0, sz as f64 + 0.5)
}

/// Current player count.
fn online(shared: &Arc<Mutex<State>>) -> usize {
    shared.lock_recover().players.len()
}

/// Trim a name to the length cap and strip control characters; fall back to a
/// generic label if nothing usable remains.
fn clean_name(raw: &str) -> String {
    let name: String = raw.chars().filter(|c| !c.is_control()).take(MAX_NAME).collect();
    let name = name.trim().to_string();
    if name.is_empty() { "player".to_string() } else { name }
}

/// Trim a chat line to the length cap and strip control characters.
fn clean_chat(raw: &str) -> String {
    raw.chars().filter(|c| !c.is_control()).take(MAX_CHAT).collect::<String>().trim().to_string()
}

#[cfg(test)]
mod tests {
    // Test setup (bind/connect/spawn) may unwrap: a panic here is a loud test
    // failure, which is exactly what the deny on the PRODUCTION paths exists
    // to prevent (a client thread silently poisoning the shared state).
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn test_generator() -> SineHills {
        SineHills::new(&mut BlockRegistry::with_builtins(), TERRAIN_BASE, 4242)
    }

    #[test]
    fn names_are_capped_and_sanitised() {
        assert_eq!(clean_name("  bob\n "), "bob");
        assert_eq!(clean_name(""), "player");
        assert_eq!(clean_name(&"x".repeat(100)).len(), MAX_NAME);
    }

    #[test]
    fn chat_is_sanitised() {
        assert_eq!(clean_chat("hi\tthere\n"), "hithere");
        assert_eq!(clean_chat(&"a".repeat(500)).len(), MAX_CHAT);
    }

    #[test]
    fn spawn_points_sit_above_the_surface() {
        let terrain = test_generator();
        for id in 1..20 {
            let p = spawn_point(&terrain, id);
            let ground = terrain.height(block_coord(p.x), block_coord(p.z));
            assert!(p.y > ground as f64, "spawn should be above ground");
        }
    }

    /// An out-of-reach edit must be dropped; an in-reach one must be recorded.
    /// Exercised directly against the shared state without a socket.
    #[test]
    fn edit_reach_is_enforced() {
        let (out, _rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        // A throwaway loopback socket just to fill the `kick` handle.
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let stream = TcpStream::connect(listener.local_addr().unwrap()).unwrap();

        let mut players = HashMap::new();
        players.insert(
            1u32,
            PlayerHandle {
                name: "p".into(),
                pos: DVec3::new(8.5, 20.0, 8.5),
                yaw: 0.0,
                pitch: 0.0,
                stance: Stance::Standing,
                out,
                kick: stream,
                ready: true,
                backlog: Vec::new(),
            },
        );
        let shared = Arc::new(Mutex::new(State {
            edits: HashMap::new(),
            players,
            grid: HashMap::new(),
            next_id: 2,
            day: 0.3,
        }));

        on_edit(&shared, 1, 500, 20, 500, "air"); // far away: rejected
        on_edit(&shared, 1, 8, 20, 8, "air"); // in reach: recorded

        let state = shared.lock_recover();
        assert!(state.edits.contains_key(&(8, 20, 8)), "in-reach edit recorded");
        assert!(!state.edits.contains_key(&(500, 20, 500)), "out-of-reach edit dropped");
    }

    /// The server-ordered edit must be echoed back to its own sender — that
    /// echo is what converges racing place-vs-break edits on one cell (see
    /// [`on_edit`]). A rejected edit must echo nothing.
    #[test]
    fn edits_are_echoed_to_the_sender() {
        let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let stream = TcpStream::connect(listener.local_addr().unwrap()).unwrap();

        let mut players = HashMap::new();
        players.insert(
            1u32,
            PlayerHandle {
                name: "p".into(),
                pos: DVec3::new(8.5, 20.0, 8.5),
                yaw: 0.0,
                pitch: 0.0,
                stance: Stance::Standing,
                out,
                kick: stream,
                ready: true,
                backlog: Vec::new(),
            },
        );
        let shared = Arc::new(Mutex::new(State {
            edits: HashMap::new(),
            players,
            grid: HashMap::new(),
            next_id: 2,
            day: 0.3,
        }));

        // Out of reach: rejected, so nothing (not even an echo) is queued.
        on_edit(&shared, 1, 500, 20, 500, "air");
        assert!(rx.try_recv().is_err(), "a rejected edit must not be echoed");

        // In reach: recorded AND echoed to the sender themself.
        on_edit(&shared, 1, 8, 20, 8, "air");
        let frame = rx.try_recv().expect("the sender must receive their own edit");
        match ServerMessage::decode(&frame) {
            Some(ServerMessage::Edit { x: 8, y: 20, z: 8, spec }) if spec == "air" => {}
            other => panic!("expected the sender's edit echoed back, got {other:?}"),
        }
    }

    /// A hand-built state, no sockets: the grid entry must follow the player
    /// across bucket borders, never duplicate within a bucket, and floor (not
    /// truncate) on negative coordinates.
    #[test]
    fn grid_membership_follows_movement_across_bucket_borders() {
        let (out, _rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let stream = TcpStream::connect(listener.local_addr().unwrap()).unwrap();

        let start = DVec3::new(10.0, 20.0, 10.0);
        let mut players = HashMap::new();
        players.insert(
            1u32,
            PlayerHandle {
                name: "p".into(),
                pos: start,
                yaw: 0.0,
                pitch: 0.0,
                stance: Stance::Standing,
                out,
                kick: stream,
                ready: true,
                backlog: Vec::new(),
            },
        );
        let mut state =
            State { edits: HashMap::new(), players, grid: HashMap::new(), next_id: 2, day: 0.3 };
        state.grid_insert(1, start);
        assert_eq!(state.grid.get(&(0, 0)).map(Vec::len), Some(1));
        let shared = Arc::new(Mutex::new(state));

        // Crossing the x border: the entry moves buckets and the emptied bucket
        // is dropped, not left behind as a leaked key.
        on_move(&shared, 1, DVec3::new(INTEREST_RADIUS + 5.0, 20.0, 10.0), 0.0, 0.0, Stance::Standing);
        {
            let s = shared.lock_recover();
            assert_eq!(s.grid.get(&(1, 0)).map(Vec::as_slice), Some(&[1u32][..]));
            assert!(!s.grid.contains_key(&(0, 0)), "emptied bucket must be removed");
        }

        // Moving within the same bucket must not duplicate the entry.
        on_move(&shared, 1, DVec3::new(INTEREST_RADIUS + 6.0, 20.0, 10.0), 0.0, 0.0, Stance::Standing);
        {
            let s = shared.lock_recover();
            assert_eq!(s.grid.get(&(1, 0)).map(Vec::len), Some(1));
            assert_eq!(s.grid.len(), 1);
        }

        // Negative coordinates floor toward -infinity: -1.0 is bucket -1, not 0.
        on_move(&shared, 1, DVec3::new(-1.0, 20.0, -1.0), 0.0, 0.0, Stance::Standing);
        {
            let s = shared.lock_recover();
            assert_eq!(s.grid.get(&(-1, -1)).map(Vec::len), Some(1));
            assert_eq!(s.grid.len(), 1);
        }
    }

    /// End-to-end over loopback: moves are only delivered inside the interest
    /// radius, and delivery resumes when players end up adjacent again — i.e. the
    /// grid entries genuinely follow the players around.
    #[test]
    fn far_players_hear_no_moves_until_adjacent() {
        use crate::net::client::Connection;

        let handle = spawn(0, Config { password: String::new(), seed: 4242 }).unwrap();
        let port = handle.addr().port();
        let mut a = Connection::connect("127.0.0.1", port, "alice", "").unwrap();
        let mut b = Connection::connect("127.0.0.1", port, "bob", "").unwrap();

        let settle = Duration::from_millis(150);
        thread::sleep(settle);
        a.poll();
        b.poll();
        assert_eq!(b.peers().count(), 1, "bob should see alice");

        // Alice teleports many buckets away. Bob (still at spawn) is far outside
        // her interest radius, so his view of her must not update.
        let far = DVec3::new(4000.0, 30.0, 4000.0);
        a.send_move(far, 0.0, 0.0, Stance::Standing);
        thread::sleep(settle);
        b.poll();
        let alice_as_seen = b.peers().next().unwrap().sample(Instant::now() + Duration::from_secs(3600)).pos;
        assert!(
            alice_as_seen.x < 100.0,
            "bob must not hear a move from {} units away (saw x={})",
            far.x,
            alice_as_seen.x
        );

        // Bob moves right next to alice: she is within range of his new position,
        // so she hears it — which requires her grid entry to have followed her.
        b.send_move(DVec3::new(4004.0, 30.0, 4004.0), 0.0, 0.0, Stance::Standing);
        thread::sleep(settle);
        a.poll();
        let bob_as_seen = a.peers().next().unwrap().sample(Instant::now() + Duration::from_secs(3600)).pos;
        assert!(
            bob_as_seen.x > 3900.0,
            "alice should hear bob once adjacent (saw x={})",
            bob_as_seen.x
        );

        // And the reverse direction: bob's entry followed him too.
        a.send_move(DVec3::new(4010.0, 30.0, 4010.0), 0.0, 0.0, Stance::Standing);
        thread::sleep(settle);
        b.poll();
        let alice_as_seen = b.peers().next().unwrap().sample(Instant::now() + Duration::from_secs(3600)).pos;
        assert!(
            alice_as_seen.x > 3900.0,
            "bob should hear alice once adjacent (saw x={})",
            alice_as_seen.x
        );

        handle.stop();
    }

    /// Join, wander across bucket borders, leave — repeatedly. The grid must
    /// always hold exactly one entry per connected player and drain to zero
    /// buckets when everyone is gone: no leaked ids, no leaked keys.
    #[test]
    fn grid_never_leaks_entries_under_churn() {
        use crate::net::client::Connection;

        /// Poll `cond` for up to two seconds (server cleanup runs on its own
        /// threads, so give it a moment rather than a fixed sleep).
        fn eventually(mut cond: impl FnMut() -> bool) -> bool {
            let deadline = Instant::now() + Duration::from_secs(2);
            while Instant::now() < deadline {
                if cond() {
                    return true;
                }
                thread::sleep(Duration::from_millis(10));
            }
            false
        }

        let handle = spawn(0, Config { password: String::new(), seed: 7 }).unwrap();
        let port = handle.addr().port();

        for round in 0..3 {
            let mut a = Connection::connect("127.0.0.1", port, "a", "").unwrap();
            let mut b = Connection::connect("127.0.0.1", port, "b", "").unwrap();
            assert!(
                eventually(|| handle.grid_entries() == 2),
                "round {round}: both joins should land in the grid"
            );

            // March both across several bucket borders (outpacing the client-side
            // move throttle with a small sleep between sends).
            for step in 1..=3 {
                thread::sleep(Duration::from_millis(40));
                let d = (step * 200) as f64; // 200 > INTEREST_RADIUS: a new bucket each step
                a.send_move(DVec3::new(d, 30.0, 0.0), 0.0, 0.0, Stance::Standing);
                b.send_move(DVec3::new(-d, 30.0, -d), 0.0, 0.0, Stance::Standing);
            }
            thread::sleep(Duration::from_millis(150));
            assert_eq!(
                handle.grid_entries(),
                2,
                "round {round}: moving must never grow or shrink membership"
            );

            drop(a);
            drop(b);
            assert!(
                eventually(|| handle.grid_entries() == 0 && handle.grid_buckets() == 0),
                "round {round}: grid must drain to zero entries and zero buckets, got {} entries in {} buckets",
                handle.grid_entries(),
                handle.grid_buckets()
            );
        }

        handle.stop();
    }
}
