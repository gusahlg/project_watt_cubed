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
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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
/// Concurrent not-yet-authenticated connections the server will hold.
/// `MAX_PLAYERS` bounds the roster only AFTER a handshake; without this cap a
/// flood of silent connects would squat one thread and file descriptor each
/// for the whole [`HANDSHAKE_TIMEOUT`]. Connections past the cap are refused
/// in the accept loop, before any thread is spawned for them.
const HANDSHAKE_CAP: usize = 64;
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
/// Fastest plausible legitimate speed (world units/second) for the movement
/// envelope: terminal fall is 60 m/s and default boosted flight well under
/// that, so 80 m/s accepts every stock movement with headroom while making
/// `Move(anywhere)` reach-forging impossible. Deliberate discontinuities go
/// through [`ClientMessage::Teleport`] instead.
const MAX_MOVE_SPEED: f64 = 80.0 * crate::math::PER_METER;
/// Latency slack added to the envelope window, so jitter between a client's
/// send cadence and our receive time never rejects honest movement.
const MOVE_SLACK_SECS: f64 = 0.3;
/// Longest gap credited to the envelope: past this, elapsed time stops buying
/// displacement (an idle client can't bank a cross-map jump allowance).
const MOVE_WINDOW_CAP_SECS: f64 = 2.0;
/// Hard cap on the server's canonical spec pool. Matches the client palette
/// cap ([`format::MAX_SPECS`](crate::save::format)); edits needing a NEW spec
/// past it are rejected, so a hostile client can exhaust neither server
/// memory nor its peers' palettes.
const MAX_SPEC_POOL: usize = 16_384;
/// Edits are streamed to a joining client in batches this size, so a very built-up
/// world's snapshot never overflows a single frame's size cap. Derived from the
/// worst case per edit — 12 bytes x/y/z + 4-byte revision + 2-byte length
/// prefix + [`MAX_SPEC`] spec bytes — with headroom for the frame header.
const SNAPSHOT_BATCH: usize = (crate::net::MAX_FRAME - 64) / (12 + 4 + 2 + MAX_SPEC);
/// Average terrain height the generator oscillates around — matches the client's
/// [`World`](crate::world::World::new) so server spawn heights land on real ground.
const TERRAIN_BASE: f32 = 20.0;

/// The public knobs for a server. Built by the dedicated binary and the in-game host.
pub struct Config {
    /// Password every client must present. Empty means no password is required.
    pub password: String,
    /// The world seed all clients generate their terrain from.
    pub seed: i64,
    /// Real seconds per full day/night cycle, shared with every client.
    pub day_secs: f32,
    /// Whether clients may `/tp` (an explicit [`ClientMessage::Teleport`]).
    /// Off, a teleport is answered with an authoritative snap-back.
    pub allow_teleport: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            password: String::new(),
            seed: 0,
            day_secs: 600.0, // matches the client's default DayLength
            allow_teleport: true,
        }
    }
}

/// Immutable per-server context shared with every connection handler: the auth
/// password, the seed, the content fingerprint joins must match, and just
/// enough of the generator to place spawns on ground.
struct Ctx {
    password: String,
    seed: i64,
    fingerprint: u64,
    day_secs: f32,
    allow_teleport: bool,
    generator: SineHills,
}

/// One connected player as the server tracks them.
struct PlayerHandle {
    name: String,
    pos: DVec3,
    yaw: f32,
    pitch: f32,
    stance: Stance,
    /// When the last accepted `Move`/`Teleport` landed — the movement
    /// envelope's time anchor.
    last_move: Instant,
    /// Ids currently inside mutual interest range (visibility is symmetric,
    /// so `a.visible.contains(b) == b.visible.contains(a)`). Maintained by the
    /// mover's diff in [`on_move`]; drives PeerExited/re-entry pose events.
    visible: std::collections::HashSet<u32>,
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

/// One authoritative overlay cell: its canonical spec and its revision — the
/// optimistic-concurrency token racing edits compare against.
struct Cell {
    spec: std::sync::Arc<str>,
    rev: u32,
}

/// The single piece of shared, mutable server state: the authoritative edit overlay
/// (coordinate → canonical block spec + revision), the player roster, and the
/// interest grid that indexes the roster by position.
struct State {
    edits: HashMap<(i32, i32, i32), Cell>,
    /// Distinct CANONICAL spec strings, shared by every edit naming them: a
    /// thousand broken blocks are a thousand map entries but ONE "air"
    /// allocation. Entries are released as soon as no live cell references
    /// them ([`State::release`]), and the pool is capped at [`MAX_SPEC_POOL`],
    /// so the overlay's per-entry weight — the memory story of a long-lived
    /// world — is bounded by real content, not by attacker-minted strings.
    spec_pool: std::collections::HashSet<std::sync::Arc<str>>,
    /// The same compiled palette clients build, used to validate and
    /// canonicalize incoming edit specs with EXACTLY the rules clients apply.
    registry: BlockRegistry,
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
    /// The shared clock: the `[0,1)` day fraction that was current at
    /// `day_set`. Clients advance locally at the shared cycle length; the
    /// server advances only when asked ([`State::day_now`]), so a late joiner
    /// receives the CURRENT phase rather than whatever `/time` last set.
    day: f32,
    day_set: Instant,
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

    /// The day fraction as of now, advanced from the last set point at the
    /// shared cycle length.
    fn day_now(&self, day_secs: f32) -> f32 {
        let elapsed = self.day_set.elapsed().as_secs_f32();
        (self.day + elapsed / day_secs.max(1.0)).rem_euclid(1.0)
    }

    /// Intern a canonical spec, or `None` at the [`MAX_SPEC_POOL`] cap.
    fn intern(&mut self, spec: &str) -> Option<std::sync::Arc<str>> {
        if let Some(shared) = self.spec_pool.get(spec) {
            return Some(shared.clone());
        }
        if self.spec_pool.len() >= MAX_SPEC_POOL {
            return None;
        }
        let shared: std::sync::Arc<str> = std::sync::Arc::from(spec);
        self.spec_pool.insert(shared.clone());
        Some(shared)
    }

    /// Drop a spec's pool entry once no live cell references it. `old` is the
    /// reference just removed from the overlay: when the pool entry and `old`
    /// are the only two remaining owners, the string is dead content.
    fn release(&mut self, old: std::sync::Arc<str>) {
        if std::sync::Arc::strong_count(&old) == 2 {
            self.spec_pool.remove(&old);
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

    // Resolve the generator's palette once: it doubles as spawn-height terrain,
    // the content identity joins must match, and the edit-spec validator.
    let mut registry = BlockRegistry::with_builtins();
    let generator = SineHills::new(&mut registry, TERRAIN_BASE, config.seed);
    let ctx = Arc::new(Ctx {
        password: config.password,
        seed: config.seed,
        fingerprint: crate::net::fingerprint_of(&registry),
        day_secs: config.day_secs,
        allow_teleport: config.allow_teleport,
        generator,
    });
    let shared = Arc::new(Mutex::new(State {
        edits: HashMap::new(),
        spec_pool: std::collections::HashSet::new(),
        registry,
        players: HashMap::new(),
        grid: HashMap::new(),
        next_id: 1,
        day: 0.3,
        day_set: Instant::now(),
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

/// Decrements the pre-auth connection count when a handshake ends, however it
/// ends — success, rejection, or a dropped socket all release the slot.
struct HandshakeSlot(Arc<AtomicUsize>);

impl Drop for HandshakeSlot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Accept connections until asked to stop, handing each to its own handler thread.
fn accept_loop(listener: TcpListener, shared: Arc<Mutex<State>>, ctx: Arc<Ctx>, shutdown: Arc<AtomicBool>) {
    // Non-blocking accept so the loop can notice `stop()` between connections.
    let _ = listener.set_nonblocking(true);
    let pending = Arc::new(AtomicUsize::new(0));
    while !shutdown.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, addr)) => {
                // Bound pre-auth resources BEFORE spawning: past the cap the
                // connection is refused right here. The reject write goes into
                // an empty send buffer, so it cannot block the accept loop.
                if pending.fetch_add(1, Ordering::Relaxed) >= HANDSHAKE_CAP {
                    pending.fetch_sub(1, Ordering::Relaxed);
                    let _ = stream.set_nonblocking(false);
                    reject(&stream, "server busy");
                    continue;
                }
                let slot = HandshakeSlot(pending.clone());
                let _ = stream.set_nonblocking(false);
                let _ = stream.set_nodelay(true);
                let shared = shared.clone();
                let ctx = ctx.clone();
                thread::spawn(move || {
                    // A dropped connection is routine; the error is the disconnect cause.
                    let _ = handle_client(stream, addr, shared, ctx, slot);
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
fn handle_client(
    stream: TcpStream,
    addr: SocketAddr,
    shared: Arc<Mutex<State>>,
    ctx: Arc<Ctx>,
    slot: HandshakeSlot,
) -> io::Result<()> {
    // The first frame must be a valid, authenticated Hello within the handshake window.
    // Buffered reads (one buffered read per frame, not two syscalls) plus a scratch
    // Vec reused for every frame this client ever sends — no per-frame allocation.
    stream.set_read_timeout(Some(HANDSHAKE_TIMEOUT))?;
    let mut reader = io::BufReader::new(stream.try_clone()?);
    let mut frame = Vec::new();
    protocol::read_frame(&mut reader, &mut frame)?;
    let name = match ClientMessage::decode(&frame) {
        Some(ClientMessage::Hello { protocol, fingerprint, name, password }) => {
            if protocol != PROTOCOL_VERSION {
                reject(&stream, "protocol version mismatch");
                return Ok(());
            }
            if fingerprint != ctx.fingerprint {
                // Same protocol, different generated content (worldgen rules,
                // element table, placement palette): a join would silently
                // build a DIFFERENT world from the shared seed. Refuse loudly.
                reject(&stream, "world content mismatch (different game/content versions)");
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
    // Authenticated: the pre-auth window is over, free the handshake slot (the
    // roster's own MAX_PLAYERS bound takes it from here).
    drop(slot);

    // Switch to the idle timeout and wire up the writer.
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
    let existing: Vec<(u32, String)>;
    let snapshot: Vec<(i32, i32, i32, u32, String)>;
    {
        let mut state = shared.lock_recover();
        world_day = state.day_now(ctx.day_secs);
        if state.players.len() >= MAX_PLAYERS {
            drop(state);
            reject(&stream, "server full");
            return Ok(());
        }
        id = state.next_id;
        state.next_id += 1;
        spawn = spawn_point(&ctx.generator, id);

        // Roster only — poses flow through the visibility machinery once the
        // joiner reports their first move, so a far peer is never bootstrapped
        // as a frozen ghost at a stale position.
        existing = state.players.iter().map(|(&pid, h)| (pid, h.name.clone())).collect();
        snapshot = state
            .edits
            .iter()
            .map(|(&(x, y, z), cell)| (x, y, z, cell.rev, cell.spec.to_string()))
            .collect();

        state.players.insert(
            id,
            PlayerHandle {
                name: name.clone(),
                pos: spawn,
                yaw: 0.0,
                pitch: 0.0,
                stance: Stance::Standing,
                last_move: Instant::now(),
                visible: std::collections::HashSet::new(),
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
    // Hand the newcomer the shared clock — CURRENT phase and cycle length, so
    // a late join lands mid-day exactly where everyone else's sky is.
    send_blocking(&out, &ServerMessage::Time { day: world_day, day_secs: ctx.day_secs });
    for (pid, pname) in existing {
        send_blocking(&out, &ServerMessage::PeerJoined { id: pid, name: pname });
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
            ClientMessage::Teleport { pos } => on_teleport(&shared, &ctx, id, pos),
            ClientMessage::Edit { req, x, y, z, expect, spec } => {
                on_edit(&shared, id, req, x, y, z, expect, &spec)
            }
            ClientMessage::Chat { channel, text } => on_chat(&shared, id, channel, &text),
            ClientMessage::SetTime { day } => on_set_time(&shared, &ctx, day),
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
            // Visibility is symmetric: everyone who could see the leaver holds
            // a reciprocal entry that must not dangle (PeerLeft removes the
            // avatar client-side either way).
            for pid in h.visible {
                if let Some(other) = state.players.get_mut(&pid) {
                    other.visible.remove(&id);
                }
            }
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
/// Validation is a plausibility ENVELOPE, not full physics: a move may cover
/// at most [`MAX_MOVE_SPEED`] × (elapsed + slack) world units and must stay
/// inside the world border. An implausible move is not committed — the server
/// keeps its last accepted position (which edit reach reads), and the client
/// is snapped back with an authoritative [`ServerMessage::Position`]. `/tp`
/// discontinuities arrive as [`ClientMessage::Teleport`] instead.
///
/// Runs in two phases to keep the global lock hold minimal. Locked: commit the
/// move, keep the grid current, diff visibility, and snapshot the recipients'
/// senders (cheap `SyncSender` clones — one `Arc` bump each). Unlocked: the
/// `try_send`s. `try_send` never blocks, failures land their owner on the kick
/// list, [`kick_slow`] tolerates ids that disconnected in the unlocked window,
/// and ids are never reused, so a late kick can't hit the wrong player.
fn on_move(shared: &Arc<Mutex<State>>, id: u32, pos: DVec3, yaw: f32, pitch: f32, stance: Stance) {
    // Ignore non-finite pose data outright. A NaN position poisons distance
    // checks/grid keys; a NaN angle propagates into peer interpolation and render
    // matrices even though the server itself does not otherwise use the angle.
    if !pos.x.is_finite()
        || !pos.y.is_finite()
        || !pos.z.is_finite()
        || !yaw.is_finite()
        || !pitch.is_finite()
    {
        return;
    }
    let mut sends: Vec<(u32, SyncSender<Arc<[u8]>>, Arc<[u8]>)> = Vec::new();
    {
        let mut state = shared.lock_recover();
        let Some(h) = state.players.get(&id) else { return };
        // Envelope: reject a jump the fastest legitimate movement could not
        // have made, and any position outside the border. The client learns
        // its authoritative position instead of silently diverging.
        let elapsed = h.last_move.elapsed().as_secs_f64().min(MOVE_WINDOW_CAP_SECS);
        let allowed = MAX_MOVE_SPEED * (elapsed + MOVE_SLACK_SECS);
        let outside = pos.x.abs() > crate::math::WORLD_BORDER
            || pos.y.abs() > crate::math::WORLD_BORDER
            || pos.z.abs() > crate::math::WORLD_BORDER;
        if outside || h.pos.distance_squared(pos) > allowed * allowed {
            let correction: Arc<[u8]> =
                ServerMessage::Position { pos: h.pos }.encode().into();
            if h.ready {
                sends.push((id, h.out.clone(), correction));
            }
        } else {
            commit_pose(&mut state, id, pos, Some((yaw, pitch, stance)), &mut sends);
        }
    }
    dispatch(shared, sends);
}

/// An explicit `/tp` discontinuity: exempt from the movement envelope, still
/// border-checked, and refused (with an authoritative snap-back) when the
/// server configuration forbids client teleports.
fn on_teleport(shared: &Arc<Mutex<State>>, ctx: &Ctx, id: u32, pos: DVec3) {
    if !pos.x.is_finite() || !pos.y.is_finite() || !pos.z.is_finite() {
        return;
    }
    let mut sends: Vec<(u32, SyncSender<Arc<[u8]>>, Arc<[u8]>)> = Vec::new();
    {
        let mut state = shared.lock_recover();
        let Some(h) = state.players.get(&id) else { return };
        let outside = pos.x.abs() > crate::math::WORLD_BORDER
            || pos.y.abs() > crate::math::WORLD_BORDER
            || pos.z.abs() > crate::math::WORLD_BORDER;
        if outside || !ctx.allow_teleport {
            let correction: Arc<[u8]> =
                ServerMessage::Position { pos: h.pos }.encode().into();
            if h.ready {
                sends.push((id, h.out.clone(), correction));
            }
        } else {
            commit_pose(&mut state, id, pos, None, &mut sends);
        }
    }
    dispatch(shared, sends);
}

/// Commit an accepted position (and optionally angles/stance), keep the grid
/// current, and queue the interest fan-out INCLUDING the visibility diff:
/// peers staying in range get the move, peers entering range get both sides'
/// poses (the "un-hide" signal), and peers leaving range get [`PeerExited`]
/// both ways so nobody keeps drawing a frozen ghost. Must run under the state
/// lock; the queued sends go out after it drops.
///
/// [`PeerExited`]: ServerMessage::PeerExited
fn commit_pose(
    state: &mut State,
    id: u32,
    pos: DVec3,
    angles: Option<(f32, f32, Stance)>,
    sends: &mut Vec<(u32, SyncSender<Arc<[u8]>>, Arc<[u8]>)>,
) {
    let Some(h) = state.players.get_mut(&id) else { return };
    let old = h.pos;
    h.pos = pos;
    if let Some((yaw, pitch, stance)) = angles {
        h.yaw = yaw;
        h.pitch = pitch;
        h.stance = stance;
    }
    h.last_move = Instant::now();
    let (yaw, pitch, stance) = (h.yaw, h.pitch, h.stance);
    // Keep the grid honest before collecting from it.
    let (from, to) = (bucket_of(old), bucket_of(pos));
    if from != to {
        state.grid_remove(id, old);
        state.grid_insert(id, pos);
    }
    let move_frame: Arc<[u8]> =
        ServerMessage::PeerMove { id, pos, yaw, pitch, stance }.encode().into();

    // Broad phase: buckets are one INTEREST_RADIUS wide, so every player in
    // range is somewhere in the mover's 3×3 neighbourhood. Exact phase: the
    // same per-player squared-distance check as ever — the grid narrows the
    // candidate set, never the audience. `wrapping_add` so a hostile position
    // at the i32 edge can't overflow; a wrapped key at worst nominates
    // candidates the exact check rejects.
    let mut now_visible: Vec<u32> = Vec::new();
    for dx in -1..=1i32 {
        for dz in -1..=1i32 {
            let key = (to.0.wrapping_add(dx), to.1.wrapping_add(dz));
            let Some(bucket) = state.grid.get(&key) else { continue };
            for &pid in bucket {
                if pid == id {
                    continue;
                }
                let Some(other) = state.players.get(&pid) else { continue };
                // Bootstrapping joiners wait: visibility forms once they are
                // ready and either side moves.
                if !other.ready {
                    continue;
                }
                // Squared-distance compare: per candidate per move, skip the sqrt.
                if other.pos.distance_squared(pos) > INTEREST_RADIUS_SQ {
                    continue;
                }
                now_visible.push(pid);
            }
        }
    }

    // Departures: in the old set, not the new — both sides hide each other.
    let mover_out = state.players[&id].out.clone();
    let departed: Vec<u32> = state.players[&id]
        .visible
        .iter()
        .copied()
        .filter(|pid| !now_visible.contains(pid))
        .collect();
    for pid in departed {
        state.players.get_mut(&id).map(|h| h.visible.remove(&pid));
        if let Some(other) = state.players.get_mut(&pid) {
            other.visible.remove(&id);
            sends.push((
                pid,
                other.out.clone(),
                ServerMessage::PeerExited { id }.encode().into(),
            ));
            sends.push((
                id,
                mover_out.clone(),
                ServerMessage::PeerExited { id: pid }.encode().into(),
            ));
        }
    }
    // Arrivals and stayers. An arriving peer needs the mover's pose (the move
    // frame doubles as it) AND the mover needs the arriving peer's current
    // pose, or the mover would keep hiding them until they next move.
    for pid in now_visible {
        let entered = !state.players[&id].visible.contains(&pid);
        let Some(other) = state.players.get_mut(&pid) else { continue };
        sends.push((pid, other.out.clone(), move_frame.clone()));
        if entered {
            other.visible.insert(id);
            let pose = ServerMessage::PeerMove {
                id: pid,
                pos: other.pos,
                yaw: other.yaw,
                pitch: other.pitch,
                stance: other.stance,
            };
            sends.push((id, mover_out.clone(), pose.encode().into()));
            if let Some(h) = state.players.get_mut(&id) {
                h.visible.insert(pid);
            }
        }
    }
}

/// Unlocked fan-out of queued `(recipient, sender, frame)` triples; a full (or
/// hung-up) queue marks its owner for the kick pass.
fn dispatch(shared: &Arc<Mutex<State>>, sends: Vec<(u32, SyncSender<Arc<[u8]>>, Arc<[u8]>)>) {
    let mut slow = Vec::new();
    for (pid, out, frame) in sends {
        if out.try_send(frame).is_err() && !slow.contains(&pid) {
            slow.push(pid);
        }
    }
    if !slow.is_empty() {
        kick_slow(&shared.lock_recover(), &slow);
    }
}

/// Validate, order, and record a block edit — the authoritative cell
/// transition. The sender gets an [`EditAck`] verdict (their rollback signal);
/// everyone else gets the committed [`Edit`] at its new revision.
///
/// The gates, in order:
/// - reach, against the sender's last ACCEPTED position (the envelope in
///   [`on_move`] is what makes that position meaningful);
/// - spec validity: parsed and canonicalized by the SAME rules clients apply,
///   so junk never interns and equivalent spellings collapse to one string;
/// - the expected cell revision: when two players race one cell, exactly one
///   expectation matches — the loser is rejected and rolls back (no duplicate
///   loot, no ghost blocks).
///
/// [`EditAck`]: ServerMessage::EditAck
/// [`Edit`]: ServerMessage::Edit
fn on_edit(shared: &Arc<Mutex<State>>, id: u32, req: u32, x: i32, y: i32, z: i32, expect: u32, spec: &str) {
    let mut state = shared.lock_recover();
    let Some(h) = state.players.get(&id) else { return };
    let ack_to = h.ready.then(|| h.out.clone());
    let reject = |state: &State, out: Option<SyncSender<Arc<[u8]>>>| {
        let rev = state.edits.get(&(x, y, z)).map_or(0, |c| c.rev);
        if let Some(out) = out {
            let _ = out.try_send(
                ServerMessage::EditAck { req, accepted: false, rev }.encode().into(),
            );
        }
    };
    // Y is unbounded (infinite world height/depth); reach is the real gate,
    // checked against the editor's own last accepted position — no reaching
    // across the map.
    let target = DVec3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5);
    if spec.len() > MAX_SPEC || h.pos.distance(target) > EDIT_REACH {
        return reject(&state, ack_to);
    }
    // Validate and canonicalize through the shared palette rules. Anything
    // unparseable resolves to AIR; only the literal "air" spec may mean AIR,
    // so junk is rejected instead of silently breaking a block.
    let block = crate::save::registry_parse_block(&mut state.registry, spec);
    if block == crate::block::AIR && spec != "air" {
        return reject(&state, ack_to);
    }
    let canonical = crate::save::registry_block_spec(&state.registry, block);
    // Optimistic concurrency: the sender must have seen the current cell.
    let current = state.edits.get(&(x, y, z)).map_or(0, |c| c.rev);
    if expect != current {
        return reject(&state, ack_to);
    }
    let rev = current + 1;
    let Some(spec) = state.intern(&canonical) else {
        return reject(&state, ack_to); // pool at cap: refuse new content
    };
    if let Some(old) = state.edits.insert((x, y, z), Cell { spec, rev }) {
        state.release(old.spec);
    }
    if let Some(out) = ack_to {
        let _ = out
            .try_send(ServerMessage::EditAck { req, accepted: true, rev }.encode().into());
    }
    let msg = ServerMessage::Edit { x, y, z, rev, spec: canonical };
    broadcast(&mut state, &msg, |pid, _| pid != id);
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

/// Record and relay a `/time` change: anchor the shared clock at the new
/// phase so joiners inherit the CURRENT time, then echo it to everyone (the
/// sender included, so all clocks agree). A non-finite value is ignored
/// rather than poisoning the shared time.
fn on_set_time(shared: &Arc<Mutex<State>>, ctx: &Ctx, day: f32) {
    if !day.is_finite() {
        return;
    }
    let day = day.rem_euclid(1.0);
    let mut state = shared.lock_recover();
    state.day = day;
    state.day_set = Instant::now();
    broadcast(&mut state, &ServerMessage::Time { day, day_secs: ctx.day_secs }, |_, _| true);
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
        assert_eq!(clean_name("  guahlg\n "), "guahlg");
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

    /// A roster entry for direct state tests. `last_move` starts well in the
    /// past so the first envelope window is at its cap (a fresh anchor allows
    /// only ~30 world units); tests re-age it between deliberate big moves.
    fn test_player(pos: DVec3, out: SyncSender<Arc<[u8]>>, kick: TcpStream) -> PlayerHandle {
        PlayerHandle {
            name: "p".into(),
            pos,
            yaw: 0.0,
            pitch: 0.0,
            stance: Stance::Standing,
            last_move: Instant::now() - Duration::from_secs(10),
            visible: std::collections::HashSet::new(),
            out,
            kick,
            ready: true,
            backlog: Vec::new(),
        }
    }

    /// A throwaway loopback socket to fill `kick` handles.
    fn stream_pair() -> TcpStream {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        TcpStream::connect(listener.local_addr().unwrap()).unwrap()
    }

    fn test_state(players: HashMap<u32, PlayerHandle>) -> State {
        State {
            edits: HashMap::new(),
            spec_pool: std::collections::HashSet::new(),
            registry: BlockRegistry::with_builtins(),
            players,
            grid: HashMap::new(),
            next_id: 2,
            day: 0.3,
            day_set: Instant::now(),
        }
    }

    /// Push the player's envelope anchor into the past, buying the next move
    /// the full (capped) displacement window.
    fn age_move(shared: &Arc<Mutex<State>>, id: u32) {
        if let Some(h) = shared.lock_recover().players.get_mut(&id) {
            h.last_move = Instant::now() - Duration::from_secs(10);
        }
    }

    fn test_ctx(allow_teleport: bool) -> Ctx {
        Ctx {
            password: String::new(),
            seed: 4242,
            fingerprint: 0,
            day_secs: 600.0,
            allow_teleport,
            generator: test_generator(),
        }
    }

    /// An out-of-reach edit must be rejected; an in-reach one must be recorded.
    /// Exercised directly against the shared state without a socket.
    #[test]
    fn edit_reach_is_enforced() {
        let (out, _rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, stream_pair()));
        let shared = Arc::new(Mutex::new(test_state(players)));

        on_edit(&shared, 1, 1, 500, 20, 500, 0, "air"); // far away: rejected
        on_edit(&shared, 1, 2, 8, 20, 8, 0, "air"); // in reach: recorded

        let state = shared.lock_recover();
        assert!(state.edits.contains_key(&(8, 20, 8)), "in-reach edit recorded");
        assert!(!state.edits.contains_key(&(500, 20, 500)), "out-of-reach edit dropped");
    }

    #[test]
    fn nonfinite_angles_do_not_enter_authoritative_state() {
        let (out, _rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let start = DVec3::new(8.5, 20.0, 8.5);
        let mut players = HashMap::new();
        let mut p = test_player(start, out, stream_pair());
        p.yaw = 0.25;
        p.pitch = -0.5;
        players.insert(1, p);
        let shared = Arc::new(Mutex::new(test_state(players)));

        let attempted = DVec3::new(9.5, 20.0, 8.5);
        on_move(&shared, 1, attempted, f32::NAN, 0.0, Stance::Sneaking);
        on_move(&shared, 1, attempted, 0.0, f32::INFINITY, Stance::Swimming);

        let state = shared.lock_recover();
        let player = &state.players[&1];
        assert_eq!(player.pos, start);
        assert_eq!(player.yaw, 0.25);
        assert_eq!(player.pitch, -0.5);
        assert_eq!(player.stance, Stance::Standing);
    }

    /// G-02: a jump no legitimate movement could make is NOT committed — the
    /// server keeps the last accepted position (which edit reach reads) and
    /// snaps the client back with an authoritative `Position`.
    #[test]
    fn implausible_moves_are_rejected_and_corrected() {
        let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let start = DVec3::new(8.5, 20.0, 8.5);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(start, out, stream_pair()));
        let shared = Arc::new(Mutex::new(test_state(players)));

        // A plausible walk step commits.
        let step = DVec3::new(10.5, 20.0, 8.5);
        on_move(&shared, 1, step, 0.1, 0.0, Stance::Standing);
        assert_eq!(shared.lock_recover().players[&1].pos, step);
        assert!(rx.try_recv().is_err(), "an accepted move needs no correction");

        // Move(target)+Edit(target) forging: the cross-map hop is refused...
        let forged = DVec3::new(4000.0, 20.0, 4000.0);
        on_move(&shared, 1, forged, 0.0, 0.0, Stance::Standing);
        assert_eq!(shared.lock_recover().players[&1].pos, step, "position unchanged");
        match ServerMessage::decode(&rx.try_recv().expect("a correction is sent")) {
            Some(ServerMessage::Position { pos }) => assert_eq!(pos, step),
            other => panic!("expected a Position snap-back, got {other:?}"),
        }
        // ...so the follow-up edit at the forged position stays out of reach.
        on_edit(&shared, 1, 7, 4000, 20, 4000, 0, "air");
        assert!(!shared.lock_recover().edits.contains_key(&(4000, 20, 4000)));

        // Outside the world border: rejected no matter how slow.
        age_move(&shared, 1);
        on_move(&shared, 1, DVec3::new(2.0e9, 20.0, 8.5), 0.0, 0.0, Stance::Standing);
        assert_eq!(shared.lock_recover().players[&1].pos, step);
    }

    /// `/tp` is an explicit, policy-gated discontinuity: allowed it commits
    /// (envelope exempt), refused it snaps the client back.
    #[test]
    fn teleport_is_permissioned() {
        let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let start = DVec3::new(8.5, 20.0, 8.5);
        let far = DVec3::new(50_000.5, 30.0, -2_000.5);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(start, out, stream_pair()));
        let shared = Arc::new(Mutex::new(test_state(players)));

        on_teleport(&shared, &test_ctx(true), 1, far);
        assert_eq!(shared.lock_recover().players[&1].pos, far, "allowed teleport commits");
        assert!(rx.try_recv().is_err());

        on_teleport(&shared, &test_ctx(false), 1, start);
        assert_eq!(shared.lock_recover().players[&1].pos, far, "refused teleport is not committed");
        match ServerMessage::decode(&rx.try_recv().expect("a correction is sent")) {
            Some(ServerMessage::Position { pos }) => assert_eq!(pos, far),
            other => panic!("expected a Position snap-back, got {other:?}"),
        }
    }

    /// G-01: the cell revision makes racing edits resolve to exactly one
    /// winner, and the sender's ack — not a broadcast echo — carries the
    /// verdict prediction rolls back on.
    #[test]
    fn edit_revisions_arbitrate_races_and_ack_the_sender() {
        let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, stream_pair()));
        let shared = Arc::new(Mutex::new(test_state(players)));
        let ack = |rx: &std::sync::mpsc::Receiver<Arc<[u8]>>| {
            match ServerMessage::decode(&rx.try_recv().expect("an ack is owed")) {
                Some(ServerMessage::EditAck { req, accepted, rev }) => (req, accepted, rev),
                other => panic!("expected an EditAck, got {other:?}"),
            }
        };

        // First break wins at revision 1.
        on_edit(&shared, 1, 10, 8, 20, 8, 0, "air");
        assert_eq!(ack(&rx), (10, true, 1));

        // The racing loser expected revision 0 and is rejected — exactly one
        // reward, and its ack is the rollback signal.
        on_edit(&shared, 1, 11, 8, 20, 8, 0, "air");
        assert_eq!(ack(&rx), (11, false, 1));

        // Building on the current revision succeeds.
        on_edit(&shared, 1, 12, 8, 20, 8, 1, "natural:Stone");
        assert_eq!(ack(&rx), (12, true, 2));

        // Junk specs are rejected before touching the overlay or the pool.
        on_edit(&shared, 1, 13, 8, 20, 8, 2, "banana:zzz");
        assert_eq!(ack(&rx), (13, false, 2));
        assert_eq!(shared.lock_recover().edits[&(8, 20, 8)].spec.as_ref(), "natural:Stone");
    }

    /// G-06: equivalent spec spellings collapse to ONE canonical pool entry,
    /// and a spec no live cell references leaves the pool instead of leaking.
    #[test]
    fn spec_pool_canonicalizes_and_releases_dead_entries() {
        let (out, _rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, stream_pair()));
        let shared = Arc::new(Mutex::new(test_state(players)));

        // Two spellings of the same composition: one canonical entry.
        on_edit(&shared, 1, 1, 8, 20, 8, 0, "natural:Iron,Stone");
        on_edit(&shared, 1, 2, 8, 21, 8, 0, "natural:Stone,Iron");
        {
            let state = shared.lock_recover();
            assert_eq!(state.spec_pool.len(), 1, "equivalent spellings share one entry");
            assert_eq!(
                state.edits[&(8, 20, 8)].spec.as_ref(),
                state.edits[&(8, 21, 8)].spec.as_ref()
            );
        }

        // Overwriting both cells strands the old spec: it must leave the pool.
        on_edit(&shared, 1, 3, 8, 20, 8, 1, "air");
        on_edit(&shared, 1, 4, 8, 21, 8, 1, "air");
        {
            let state = shared.lock_recover();
            assert_eq!(state.spec_pool.len(), 1, "only \"air\" remains interned");
            assert!(state.spec_pool.contains("air"));
        }
    }

    /// G-10: a late joiner reads the CURRENT phase, not the last set value.
    #[test]
    fn shared_clock_advances_between_set_and_join() {
        let mut state = test_state(HashMap::new());
        state.day = 0.25;
        state.day_set = Instant::now() - Duration::from_secs(300);
        let now = state.day_now(600.0);
        assert!((now - 0.75).abs() < 0.01, "half a 600s cycle after 0.25, got {now}");
    }

    /// A hand-built state, no sockets: the grid entry must follow the player
    /// across bucket borders, never duplicate within a bucket, and floor (not
    /// truncate) on negative coordinates.
    #[test]
    fn grid_membership_follows_movement_across_bucket_borders() {
        let (out, _rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let start = DVec3::new(10.0, 20.0, 10.0);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(start, out, stream_pair()));
        let mut state = test_state(players);
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
        // (Aged anchor: the hop back is real distance, and this test is about
        // grid bookkeeping, not the envelope.)
        age_move(&shared, 1);
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

        let handle = spawn(0, Config { password: String::new(), seed: 4242, ..Config::default() }).unwrap();
        let port = handle.addr().port();
        let mut a = Connection::connect("127.0.0.1", port, "walnutty", "").unwrap();
        let mut b = Connection::connect("127.0.0.1", port, "guahlg", "").unwrap();

        let settle = Duration::from_millis(150);
        thread::sleep(settle);
        a.poll();
        b.poll();
        assert_eq!(b.peers().count(), 1, "guahlg should see walnutty");

        // Walnutty teleports many buckets away. guahlg (still at spawn) is far outside
        // her interest radius, so his view of her must not update.
        let far = DVec3::new(4000.0, 30.0, 4000.0);
        a.send_teleport(far);
        thread::sleep(settle);
        b.poll();
        let walnutty_as_seen = b.peers().next().unwrap().sample(Instant::now() + Duration::from_secs(3600)).pos.0;
        assert!(
            walnutty_as_seen.x < 100.0,
            "guahlg must not hear a move from {} units away (saw x={})",
            far.x,
            walnutty_as_seen.x
        );

        // guahlg moves right next to walnutty: she is within range of his new position,
        // so she hears it — which requires her grid entry to have followed her.
        b.send_teleport(DVec3::new(4004.0, 30.0, 4004.0));
        thread::sleep(settle);
        a.poll();
        let guahlg_as_seen = a.peers().next().unwrap().sample(Instant::now() + Duration::from_secs(3600)).pos.0;
        assert!(
            guahlg_as_seen.x > 3900.0,
            "walnutty should hear guahlg once adjacent (saw x={})",
            guahlg_as_seen.x
        );

        // And the reverse direction: guahlg's entry followed him too.
        a.send_move(DVec3::new(4010.0, 30.0, 4010.0), 0.0, 0.0, Stance::Standing);
        thread::sleep(settle);
        b.poll();
        let walnutty_as_seen = b.peers().next().unwrap().sample(Instant::now() + Duration::from_secs(3600)).pos.0;
        assert!(
            walnutty_as_seen.x > 3900.0,
            "guahlg should hear walnutty once adjacent (saw x={})",
            walnutty_as_seen.x
        );

        handle.stop();
    }

    /// G-08: same protocol, different generated content — the handshake must
    /// refuse the join instead of letting two builds silently diverge on one
    /// seed.
    #[test]
    fn mismatched_content_fingerprint_is_rejected() {
        let handle = spawn(0, Config { password: String::new(), seed: 3, ..Config::default() }).unwrap();
        let stream = TcpStream::connect(handle.addr()).unwrap();
        let hello = ClientMessage::Hello {
            protocol: PROTOCOL_VERSION,
            fingerprint: crate::net::content_fingerprint() ^ 1,
            name: "drifted".into(),
            password: String::new(),
        };
        let mut w = &stream;
        protocol::write_frame(&mut w, &hello.encode()).unwrap();
        stream.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
        let mut reader = io::BufReader::new(stream.try_clone().unwrap());
        let mut frame = Vec::new();
        protocol::read_frame(&mut reader, &mut frame).unwrap();
        match ServerMessage::decode(&frame) {
            Some(ServerMessage::Reject { reason }) => {
                assert!(reason.contains("content"), "unexpected reason: {reason}")
            }
            other => panic!("expected a content-mismatch rejection, got {other:?}"),
        }
        handle.stop();
    }

    /// Silent (never-authenticating) connections must be bounded by
    /// [`HANDSHAKE_CAP`]: one past the cap is refused promptly instead of
    /// squatting a thread until the handshake timeout, and releasing squatters
    /// frees slots for a real join.
    #[test]
    fn silent_connections_beyond_the_handshake_cap_are_refused() {
        use crate::net::client::Connection;

        let handle = spawn(0, Config { password: String::new(), seed: 1, ..Config::default() }).unwrap();
        let addr = handle.addr();

        // Fill every pre-auth slot with connections that never send a byte.
        let squatters: Vec<TcpStream> =
            (0..HANDSHAKE_CAP).map(|_| TcpStream::connect(addr).unwrap()).collect();
        // Let the accept loop take them all in before probing past the cap.
        thread::sleep(Duration::from_millis(300));

        // One more must be turned away quickly — a Reject frame or an
        // immediate close, NOT a 10-second handshake squat.
        let extra = TcpStream::connect(addr).unwrap();
        extra.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
        let mut reader = io::BufReader::new(extra.try_clone().unwrap());
        let mut frame = Vec::new();
        match protocol::read_frame(&mut reader, &mut frame) {
            Ok(()) => match ServerMessage::decode(&frame) {
                Some(ServerMessage::Reject { reason }) => {
                    assert!(reason.contains("busy"), "unexpected reason: {reason}")
                }
                other => panic!("expected a busy rejection, got {other:?}"),
            },
            Err(e) => assert_ne!(
                e.kind(),
                io::ErrorKind::WouldBlock,
                "over-cap connection was left squatting instead of refused"
            ),
        }

        // Freeing the squatters must free their slots for a real player.
        drop(squatters);
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match Connection::connect("127.0.0.1", addr.port(), "late", "") {
                Ok(_) => break,
                Err(e) if Instant::now() < deadline => {
                    let _ = e;
                    thread::sleep(Duration::from_millis(100));
                }
                Err(e) => panic!("slots never freed after squatters left: {e}"),
            }
        }

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

        let handle = spawn(0, Config { password: String::new(), seed: 7, ..Config::default() }).unwrap();
        let port = handle.addr().port();

        for round in 0..3 {
            let mut a = Connection::connect("127.0.0.1", port, "a", "").unwrap();
            let mut b = Connection::connect("127.0.0.1", port, "b", "").unwrap();
            assert!(
                eventually(|| handle.grid_entries() == 2),
                "round {round}: both joins should land in the grid"
            );

            // March both across several bucket borders. Teleports rather than
            // moves: a 200-unit hop is beyond the movement envelope, and the
            // grid must follow ACCEPTED discontinuities just as it follows walks.
            for step in 1..=3 {
                thread::sleep(Duration::from_millis(40));
                let d = (step * 200) as f64; // 200 > INTEREST_RADIUS: a new bucket each step
                a.send_teleport(DVec3::new(d, 30.0, 0.0));
                b.send_teleport(DVec3::new(-d, 30.0, -d));
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
