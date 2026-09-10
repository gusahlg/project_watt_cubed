//! Authoritative, headless multiplayer server: owns the seed + edit overlay
//! and the player roster; terrain is procedural, so no voxel data is ever sent.
//!
//! **Threading.** One accept thread; per client a blocking reader thread and a
//! bounded-queue writer thread, coordinated through a single [`Mutex`]-guarded
//! [`State`]. The lock is held only for short bursts — move fan-out snapshots
//! its recipients under the lock and pushes to their queues after releasing
//! it. Comfortably serves hundreds of players; past that the one global lock
//! and thread-per-client model are the ceiling (join/leave and global chat
//! stay O(roster)) — an event-loop rewrite would be the next step.
//!
//! **Interest management.** Position broadcasts only reach players within
//! [`INTEREST_RADIUS`], via a 2D bucket grid ([`State::grid`]): a move consults
//! only the mover's 3×3 bucket neighbourhood instead of scanning the roster.
//!
//! **Trust.** Joins are password-gated and version-checked; frames are size-capped
//! by [`protocol`]; every client is rate-limited; every edit is bounds- and
//! reach-validated against the sender's own reported position.
//!
//! **Server mods.** [`Config::hooks`] is a [`ServerMod`] table (plain-data
//! arguments, no protocol change). Calls run outside the [`State`] lock.
use std::collections::{HashMap, HashSet, VecDeque};
use std::io;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use quinn::{Endpoint, Incoming, SendStream};
use tokio::runtime::Runtime;
use tokio::sync::Notify;
use voxel_engine::DVec3;

use crate::math::block_coord;

use crate::block::registry::{BlockId, BlockRegistry, AIR};
use crate::net::hooks;
use crate::sim::reactions::{self, CellStore, Mutation, Pos, ReactionScheduler};
pub(crate) use crate::net::hooks::{ChatFacts, EditIntent, JoinFacts, ServerMod, Verdict};
use crate::net::protocol::{self, ClientMessage, ServerMessage};
use crate::net::{MAX_CHAT, MAX_NAME, MAX_SPEC, PROTOCOL_VERSION, chat, quic};
use crate::presence::Stance;
use crate::world::diffusion::DiffusionCfg;
use crate::world::generation::{TerrainGenerator, WorldgenKind};

/// A client thread that panics while holding the state must not take the whole
/// server down with it — [`State`] is plain data, valid at every point a panic
/// could interrupt, so recovery is always sound.
trait LockRecover<T> {
    fn lock_recover(&self) -> MutexGuard<'_, T>;
}

impl<T> LockRecover<T> for Mutex<T> {
    fn lock_recover(&self) -> MutexGuard<'_, T> {
        self.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Hard bound so a flood of connects can't spawn unbounded threads.
const MAX_PLAYERS: usize = 256;
/// Reserved player id for scheduler mutations attributed to the world, not a player.
/// `next_id` starts at 1 so this id is never assigned to a joiner.
const WORLD_PLAYER: u32 = 0;
/// `MAX_PLAYERS` bounds the roster only AFTER a handshake; without this cap a
/// flood of silent connects would squat a thread+fd each for the whole
/// [`HANDSHAKE_TIMEOUT`]. Refused in the accept loop, before any thread spawns.
const HANDSHAKE_CAP: usize = 64;
/// A client this far behind is treated as unresponsive and dropped, so one
/// slow peer can't grow memory without bound.
const OUT_CAPACITY: usize = 1024;
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
/// Bounds a client that reads and holds, so a reject still gets delivered
/// before teardown without hanging forever.
const REJECT_DRAIN: Duration = Duration::from_secs(3);
const RATE_LIMIT: u32 = 300;
/// A 20 ms capture cadence is ~50 frames/s sustained; 100 leaves headroom for
/// bursts while capping a voice flood under [`RATE_LIMIT`]. Excess frames are
/// dropped silently — voice is loss-tolerant, never a kick trigger.
const VOICE_RATE_LIMIT: u32 = 100;
/// Workbench applies one connection may send per second: each one evaluates the law and may intern a
/// configuration under the [`State`] lock, so the budget is a human's click rate, not a flood.
const CRAFT_RATE_LIMIT: u32 = 10;
/// Ids the material table keeps for the world's own products (reactions, generation): a spec a
/// CLIENT sends is interned only while at least this many ids are free, so no client can exhaust
/// the table (see [`resolve_client_spec`]).
const CLIENT_INTERN_RESERVE: usize = crate::block::registry::MAX_BLOCK_TYPES / 4;
/// Player ids come from a strictly-incrementing `next_id` and are NEVER
/// reused, so each id has exactly one incarnation and a constant epoch is
/// sound. Reopen if ids ever become reusable: this must become a per-id join
/// generation on `PlayerHandle`.
const VOICE_EPOCH: u32 = 0;
const INTEREST_RADIUS: f64 = 160.0 * crate::math::PER_METER;
/// Squared once so the hot per-listener check in [`on_move`] needs no sqrt.
const INTEREST_RADIUS_SQ: f64 = INTEREST_RADIUS * INTEREST_RADIUS;
/// A little past the client's own reach constant.
const EDIT_REACH: f64 = 8.0 * crate::math::PER_METER;
/// Terminal fall is 60 m/s and default boosted flight well under that, so
/// 80 m/s accepts every stock movement with headroom while making
/// `Move(anywhere)` reach-forging impossible. Deliberate discontinuities go
/// through [`ClientMessage::Teleport`] instead.
const MAX_MOVE_SPEED: f64 = 80.0 * crate::math::PER_METER;
/// So jitter between a client's send cadence and our receive time never
/// rejects honest movement.
const MOVE_SLACK_SECS: f64 = 0.3;
/// Past this, elapsed time stops buying displacement (an idle client can't
/// bank a cross-map jump allowance).
const MOVE_WINDOW_CAP_SECS: f64 = 2.0;
/// Matches the client palette cap ([`format::MAX_SPECS`](crate::save::format));
/// a hostile client can exhaust neither server memory nor peers' palettes.
const MAX_SPEC_POOL: usize = 16_384;
/// Worst case per edit — 12 bytes x/y/z + 4-byte revision + 2-byte length
/// prefix + [`MAX_SPEC`] spec bytes — with headroom for the frame header.
const SNAPSHOT_BATCH: usize = (crate::net::MAX_FRAME - 64) / (12 + 4 + 2 + MAX_SPEC);
/// Matches the client's [`World`](crate::world::World::new) so server spawn
/// heights land on real ground.
pub struct Config {
    /// Empty means no password is required.
    pub password: String,
    pub seed: i64,
    pub day_secs: f32,
    /// Off, a teleport is answered with an authoritative snap-back.
    pub allow_teleport: bool,
    pub worldgen: WorldgenKind,
    pub diffusion: DiffusionCfg,
    /// Server-side mods (`validate_edit`, join/leave, `on_chat`). Empty by
    /// default — this crate ships no implementations. Hook bodies run outside
    /// the roster lock.
    pub hooks: Vec<Box<dyn ServerMod>>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            password: String::new(),
            seed: 0,
            day_secs: 600.0, // matches the client's default DayLength
            allow_teleport: true,
            worldgen: WorldgenKind::Classic,
            diffusion: DiffusionCfg::default(),
            hooks: Vec::new(),
        }
    }
}

/// Same range the client clock uses (`DayLength::clamped`): never zero (which
/// would stall or desync the shared sky) and never a multi-day real-time cycle.
fn clamp_day_secs(s: f32) -> f32 {
    if s.is_nan() { 600.0 } else { s.clamp(10.0, 86_400.0) }
}

/// Sliding 1-second window: a stamp ages out once a full second has passed, so
/// dumping a full budget on both sides of a second boundary cannot double it.
struct RateWindow {
    stamps: VecDeque<Instant>,
    limit: u32,
}

impl RateWindow {
    fn new(limit: u32) -> Self {
        Self { stamps: VecDeque::new(), limit }
    }

    fn allow(&mut self, now: Instant) -> bool {
        const PERIOD: Duration = Duration::from_secs(1);
        while self.stamps.front().is_some_and(|t| now.saturating_duration_since(*t) >= PERIOD) {
            self.stamps.pop_front();
        }
        if self.stamps.len() as u32 >= self.limit {
            return false;
        }
        self.stamps.push_back(now);
        true
    }
}

struct Ctx {
    password: String,
    seed: i64,
    fingerprint: u64,
    day_secs: f32,
    allow_teleport: bool,
    worldgen: WorldgenKind,
    diffusion: DiffusionCfg,
    generator: crate::world::diffusion::Generator,
    /// `None` when [`Config::hooks`] is empty so the default server never
    /// touches a second lock. When `Some`, hook calls happen *outside* the
    /// [`State`] lock: collect facts under it, drop it, then run the table.
    hooks: Option<Mutex<hooks::Table>>,
}

struct PlayerHandle {
    /// Interned once at join; every roster/join/chat broadcast that carries
    /// it is a refcount bump, never a per-recipient allocation.
    name: Arc<str>,
    pos: DVec3,
    yaw: f32,
    pitch: f32,
    stance: Stance,
    /// The movement envelope's time anchor.
    last_move: Instant,
    /// Ids inside mutual interest range (`a.visible.contains(b) ==
    /// b.visible.contains(a)`). Maintained by [`on_move`]'s diff; drives
    /// PeerExited/re-entry pose events.
    visible: HashSet<u32>,
    out: SyncSender<Arc<[u8]>>,
    /// Wakes a misbehaving client's reader out of its blocking read so cleanup
    /// runs. A `Notify` rather than `quinn::Connection` so it's cheap to
    /// fabricate in state-only tests.
    kick: Arc<Notify>,
    /// False until Welcome/Snapshot is fully queued. Broadcasters must not
    /// push into a not-yet-ready queue — a racing frame could beat Welcome
    /// onto the wire or interleave between snapshot batches — so they buffer
    /// into `backlog` instead, drained in order once the bootstrap is done.
    ready: bool,
    backlog: Vec<Arc<[u8]>>,
}

/// A recipient and its encoded frame, queued after releasing the state lock.
type PendingSend = (u32, SyncSender<Arc<[u8]>>, Arc<[u8]>);

impl PlayerHandle {
    fn correct_position(&self, id: u32, sends: &mut Vec<PendingSend>) {
        if self.ready {
            let frame = ServerMessage::Position { pos: self.pos }.encode().into();
            sends.push((id, self.out.clone(), frame));
        }
    }
}

/// Overflow marks a joiner slow (kicked) — matching the outbound-queue policy.
const BOOTSTRAP_BACKLOG: usize = 256;

/// The optimistic-concurrency token racing edits compare against.
struct Cell {
    spec: Arc<str>,
    rev: u32,
}

struct State {
    edits: HashMap<(i32, i32, i32), Cell>,
    /// Distinct CANONICAL spec strings, shared by every edit naming them: a
    /// thousand broken blocks are a thousand map entries but ONE "air"
    /// allocation. Released once no live cell references them
    /// ([`State::release`]) and capped at [`MAX_SPEC_POOL`], so per-entry
    /// weight is bounded by real content, not attacker-minted strings.
    spec_pool: HashSet<Arc<str>>,
    /// The same compiled palette clients build, so specs validate/canonicalize
    /// under EXACTLY the rules clients apply.
    registry: BlockRegistry,
    players: HashMap<u32, PlayerHandle>,
    /// Bucket key → ids standing in it, keyed by [`bucket_of`]. Buckets are
    /// exactly one [`INTEREST_RADIUS`] wide, so anyone in range of a mover
    /// lives in its 3×3 neighbourhood; [`on_move`] still applies the exact
    /// per-player distance check, so the grid only narrows candidates, never
    /// the audience. Deliberately 2D — a y axis would add bucket churn from
    /// every jump/fall while barely shrinking candidate sets, and ignoring y
    /// can only widen the candidate set, never miss a listener. Invariant:
    /// exactly one entry per connected player, updated under the same lock
    /// hold as the position change it mirrors; empty buckets are removed
    /// eagerly so churn can never leak keys.
    grid: HashMap<(i32, i32), Vec<u32>>,
    next_id: u32,
    /// The `[0,1)` day fraction current at `day_set`. The server advances it
    /// only when asked ([`State::day_now`]), so a late joiner receives the
    /// CURRENT phase rather than whatever `/time` last set.
    day: f32,
    day_set: Instant,
    /// Server-authoritative reaction scheduler. Clients never run one.
    reactions: ReactionScheduler,
}

impl State {
    /// Must run under the same lock hold as the roster/position change it
    /// mirrors, or the grid drifts.
    fn grid_insert(&mut self, id: u32, pos: DVec3) {
        self.grid.entry(bucket_of(pos)).or_default().push(id);
    }

    /// Drops the bucket when it empties so churn can never accumulate dead keys.
    fn grid_remove(&mut self, id: u32, pos: DVec3) {
        let key = bucket_of(pos);
        if let Some(bucket) = self.grid.get_mut(&key) {
            bucket.retain(|&p| p != id);
            if bucket.is_empty() {
                self.grid.remove(&key);
            }
        }
    }

    /// The grid narrows candidates; exact distance and readiness decide visibility.
    fn visible_from(&self, id: u32, pos: DVec3) -> HashSet<u32> {
        let at = bucket_of(pos);
        let mut visible = HashSet::new();
        for dx in -1..=1i32 {
            for dz in -1..=1i32 {
                let key = (at.0.wrapping_add(dx), at.1.wrapping_add(dz));
                let Some(bucket) = self.grid.get(&key) else { continue };
                for &pid in bucket {
                    if pid == id {
                        continue;
                    }
                    let Some(other) = self.players.get(&pid) else { continue };
                    if other.ready && other.pos.distance_squared(pos) <= INTEREST_RADIUS_SQ {
                        visible.insert(pid);
                    }
                }
            }
        }
        visible
    }

    fn day_now(&self, day_secs: f32) -> f32 {
        let elapsed = self.day_set.elapsed().as_secs_f32();
        (self.day + elapsed / clamp_day_secs(day_secs)).rem_euclid(1.0)
    }

    /// `None` at the [`MAX_SPEC_POOL`] cap.
    fn intern(&mut self, spec: &str) -> Option<Arc<str>> {
        if let Some(shared) = self.spec_pool.get(spec) {
            return Some(shared.clone());
        }
        if self.spec_pool.len() >= MAX_SPEC_POOL {
            return None;
        }
        let shared: Arc<str> = Arc::from(spec);
        self.spec_pool.insert(shared.clone());
        Some(shared)
    }

    /// `old` is the reference just removed from the overlay: when the pool
    /// entry and `old` are the only two remaining owners, it's dead content.
    fn release(&mut self, old: Arc<str>) {
        if Arc::strong_count(&old) == 2 {
            self.spec_pool.remove(&old);
        }
    }
}

/// Ledger + generator as a [`CellStore`]: a cell not in the ledger reads from
/// the generator, so the infinite world is defined without loading chunks.
struct ServerCells<'a> {
    state: &'a mut State,
    generator: &'a crate::world::diffusion::Generator,
}

fn server_block(
    state: &State,
    generator: &crate::world::diffusion::Generator,
    pos: Pos,
) -> BlockId {
    if let Some(cell) = state.edits.get(&pos) {
        state.registry.lookup_spec(&cell.spec).unwrap_or(AIR)
    } else {
        generator.voxel_at(pos.0, pos.1, pos.2)
    }
}

impl CellStore for ServerCells<'_> {
    fn block_at(&self, pos: Pos) -> Option<BlockId> {
        Some(server_block(self.state, self.generator, pos))
    }

    /// `None` when the spec pool is full: the cell keeps its material and the
    /// scheduler records no mutation for it (a refused write is not a change).
    fn set_block(&mut self, pos: Pos, id: BlockId) -> Option<BlockId> {
        let prev = server_block(self.state, self.generator, pos);
        if prev == id {
            return Some(prev);
        }
        let canonical = crate::save::block_spec(&self.state.registry, id);
        let spec = self.state.intern(&canonical)?;
        let rev = self.state.edits.get(&pos).map_or(0, |c| c.rev).saturating_add(1);
        if let Some(old) = self.state.edits.insert(pos, Cell { spec, rev }) {
            self.state.release(old.spec);
        }
        Some(prev)
    }

    fn registry(&self) -> &BlockRegistry {
        &self.state.registry
    }

    fn registry_mut(&mut self) -> &mut BlockRegistry {
        &mut self.state.registry
    }
}

fn reactions_loop(shared: Arc<Mutex<State>>, ctx: Arc<Ctx>, shutdown: Arc<AtomicBool>) {
    let period = Duration::from_millis(50);
    while !shutdown.load(Ordering::Relaxed) {
        let start = Instant::now();
        run_reactions(&shared, &ctx);
        if let Some(rest) = period.checked_sub(start.elapsed()) {
            thread::sleep(rest);
        }
    }
}

/// One sim tick of the scheduler. Committed mutations are [`ServerMessage::Snapshot`]
/// batches attributed to [`WORLD_PLAYER`], broadcast to every ready client. The
/// scheduler budget already bounds the count; every commit is sent.
fn run_reactions(shared: &Arc<Mutex<State>>, ctx: &Ctx) {
    let mut state = shared.lock_recover();
    if state.reactions.pending() == 0 {
        return;
    }
    let law = *state.registry.law();
    let budget = reactions::Budget::DEFAULT;
    let mut sched = std::mem::take(&mut state.reactions);
    let mutations = {
        let mut cells = ServerCells {
            state: &mut state,
            generator: &ctx.generator,
        };
        sched.tick(&mut cells, &law, budget)
    };
    state.reactions = sched;
    send_reaction_mutations(&mut state, &mutations);
}

/// Authoritative overlay edits from one scheduler tick, as snapshot batches
/// (the client applies [`ServerMessage::Snapshot`] after bootstrap), one entry
/// per distinct cell. One `S_Edit` per mutation would overflow [`OUT_CAPACITY`]
/// on two full ticks.
fn send_reaction_mutations(state: &mut State, mutations: &[Mutation]) {
    if mutations.is_empty() {
        return;
    }
    // A cell committed in both generations of one tick is sent once, with its
    // final content, at the point of its last commit (order is preserved).
    let mut last: HashMap<Pos, usize> = HashMap::with_capacity(mutations.len());
    for (i, m) in mutations.iter().enumerate() {
        last.insert(m.pos, i);
    }
    let mut edits = Vec::with_capacity(last.len());
    for (i, m) in mutations.iter().enumerate() {
        if last[&m.pos] != i {
            continue;
        }
        let Some(cell) = state.edits.get(&m.pos) else { continue };
        edits.push((m.pos.0, m.pos.1, m.pos.2, cell.rev, cell.spec.clone()));
    }
    for batch in edits.chunks(SNAPSHOT_BATCH) {
        broadcast(
            state,
            &ServerMessage::Snapshot {
                edits: batch.to_vec(),
            },
            |pid, _| pid != WORLD_PLAYER,
        );
    }
}

/// The interest-grid bucket containing `pos`. Goes through [`block_coord`]'s
/// clamped floor (not truncation) so negative coordinates bucket consistently
/// and a hostile-but-finite huge coordinate can't overflow the i32 key —
/// insert and remove share this one mapping, so the grid stays consistent.
fn bucket_of(pos: DVec3) -> (i32, i32) {
    (block_coord(pos.x / INTEREST_RADIUS), block_coord(pos.z / INTEREST_RADIUS))
}

fn outside_world(pos: DVec3) -> bool {
    pos.x.abs() > crate::math::WORLD_BORDER
        || pos.y.abs() > crate::math::WORLD_BORDER
        || pos.z.abs() > crate::math::WORLD_BORDER
}

/// A running server. [`stop`](ServerHandle::stop)ping it takes the listener down;
/// existing clients finish on their own.
pub(crate) struct ServerHandle {
    shutdown: Arc<AtomicBool>,
    addr: SocketAddr,
    /// The runtime hosting quinn, held so it outlives the handle. The accept loop
    /// and every client handler thread also hold clones, so a detached server keeps
    /// running and existing clients finish even after the handle is dropped.
    _rt: Arc<Runtime>,
    /// Test-only window into the shared state, for grid-leak assertions.
    #[cfg(test)]
    state: Arc<Mutex<State>>,
    /// Test-only window into the pre-auth handshake slot counter.
    #[cfg(test)]
    handshake_pending: Arc<AtomicUsize>,
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

    /// Must return to zero whenever the roster empties.
    #[cfg(test)]
    fn grid_buckets(&self) -> usize {
        self.state.lock_recover().grid.len()
    }

    /// Live pre-auth connections occupying a [`HANDSHAKE_CAP`] slot.
    #[cfg(test)]
    fn handshake_slots(&self) -> usize {
        self.handshake_pending.load(Ordering::Relaxed)
    }
}

/// Bind to port 0 to let the OS pick a free port.
pub(crate) fn spawn(port: u16, config: Config) -> io::Result<ServerHandle> {
    let rt = Arc::new(Runtime::new()?);
    let endpoint = {
        // Must run inside the runtime: construction spawns quinn's UDP driver.
        let _guard = rt.enter();
        Endpoint::server(quic::server_config()?, (Ipv4Addr::UNSPECIFIED, port).into())?
    };
    let addr = endpoint.local_addr()?;
    let shutdown = Arc::new(AtomicBool::new(false));

    // Doubles as spawn-height terrain, the content identity joins must match,
    // and the edit-spec validator.
    let mut registry = BlockRegistry::with_builtins();
    let generator = match config.worldgen {
        WorldgenKind::Classic => crate::world::diffusion::classic(&mut registry, config.seed),
        WorldgenKind::Diffusion => {
            if config.diffusion.version >= 2 {
                crate::world::diffusion::diffusion_v2(
                    &mut registry,
                    config.seed,
                    config.diffusion,
                )
            } else {
                crate::world::diffusion::diffusion(&mut registry, config.seed, config.diffusion)
            }
        }
    };
    let hooks = if config.hooks.is_empty() {
        None
    } else {
        Some(Mutex::new(hooks::Table::new(config.hooks)))
    };
    let ctx = Arc::new(Ctx {
        password: config.password,
        seed: config.seed,
        fingerprint: crate::net::fingerprint_kind_cfg(&registry, config.worldgen, config.diffusion),
        day_secs: clamp_day_secs(config.day_secs),
        allow_teleport: config.allow_teleport,
        worldgen: config.worldgen,
        diffusion: config.diffusion,
        generator,
        hooks,
    });
    let shared = Arc::new(Mutex::new(State {
        edits: HashMap::new(),
        spec_pool: HashSet::new(),
        registry,
        players: HashMap::new(),
        grid: HashMap::new(),
        next_id: 1,
        day: 0.3,
        day_set: Instant::now(),
        reactions: ReactionScheduler::new(),
    }));
    debug_assert_ne!(WORLD_PLAYER, 1, "player ids start at 1; 0 is the world");

    #[cfg(test)]
    let state = shared.clone();
    let pending = Arc::new(AtomicUsize::new(0));
    #[cfg(test)]
    let handshake_pending = pending.clone();
    let accept_shutdown = shutdown.clone();
    let accept_rt = rt.clone();
    let tick_shutdown = shutdown.clone();
    let tick_shared = shared.clone();
    let tick_ctx = ctx.clone();
    thread::spawn(move || reactions_loop(tick_shared, tick_ctx, tick_shutdown));
    thread::spawn(move || accept_loop(endpoint, accept_rt, shared, ctx, accept_shutdown, pending));

    Ok(ServerHandle {
        shutdown,
        addr,
        _rt: rt,
        #[cfg(test)]
        state,
        #[cfg(test)]
        handshake_pending,
    })
}

/// The dedicated server binary's entry point.
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

fn accept_loop(
    endpoint: Endpoint,
    rt: Arc<Runtime>,
    shared: Arc<Mutex<State>>,
    ctx: Arc<Ctx>,
    shutdown: Arc<AtomicBool>,
    pending: Arc<AtomicUsize>,
) {
    while !shutdown.load(Ordering::Relaxed) {
        // Bounded wait so `stop()` (which only flips the flag) is noticed
        // promptly between connections.
        let incoming = match rt
            .block_on(async { tokio::time::timeout(Duration::from_millis(200), endpoint.accept()).await })
        {
            Ok(Some(incoming)) => incoming,
            Ok(None) => break, // endpoint closed
            Err(_elapsed) => continue,
        };

        // A QUIC CONNECTION_REFUSED past the cap, before any handshake, so a
        // flood of silent connects can't squat handler threads. The accept
        // loop is the only adder, so load-then-add can't overshoot the cap.
        if pending.load(Ordering::Relaxed) >= HANDSHAKE_CAP {
            incoming.refuse();
            continue;
        }
        pending.fetch_add(1, Ordering::Relaxed);
        let slot = HandshakeSlot(pending.clone());
        let shared = shared.clone();
        let ctx = ctx.clone();
        let handler_rt = rt.clone();
        thread::spawn(move || {
            // A dropped connection is routine; the error is the disconnect cause.
            let _ = handle_client(incoming, handler_rt, shared, ctx, slot);
        });
    }
}

fn handle_client(
    incoming: Incoming,
    rt: Arc<Runtime>,
    shared: Arc<Mutex<State>>,
    ctx: Arc<Ctx>,
    slot: HandshakeSlot,
) -> io::Result<()> {
    let Some((conn, mut send, mut recv, mut frame, name, addr)) =
        handshake(incoming, &rt, &ctx, slot)?
    else {
        return Ok(());
    };
    let Some((id, spawn, world_day, existing, snapshot, out, rx, kick)) =
        admit_player(&shared, &ctx, &rt, &mut send, &conn, &name)
    else {
        return Ok(());
    };
    let writer = spawn_writer(rt.clone(), send, rx);
    println!("[+] {name} joined as #{id} from {addr} ({} online)", online(&shared));

    // These sends BLOCK (we're on this client's own handler thread): a built-up
    // world or big roster can exceed the outbound queue, and dropping bootstrap
    // frames would ghost the join.
    send_blocking(
        &out,
        &ServerMessage::Welcome {
            player_id: id,
            seed: ctx.seed,
            spawn,
            worldgen: ctx.worldgen,
            diffusion: ctx.diffusion,
            law: crate::net::protocol::law_stamp(),
        },
    );
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

    broadcast_all(&shared, &ServerMessage::PeerJoined { id, name: name.clone() }, Some(id));

    client_loop(&rt, &mut recv, &mut frame, &kick, &shared, &ctx, id);
    depart(&shared, &ctx, conn, out, writer, id, &name);
    Ok(())
}

fn handshake(
    incoming: Incoming,
    rt: &Runtime,
    ctx: &Ctx,
    slot: HandshakeSlot,
) -> io::Result<Option<(quinn::Connection, SendStream, quinn::RecvStream, Vec<u8>, Arc<str>, SocketAddr)>> {
    // The scratch Vec is reused for every frame this client ever sends.
    let conn = rt.block_on(async { incoming.await }).map_err(io::Error::other)?;
    let addr = conn.remote_address();
    let (mut send, mut recv) = rt
        .block_on(async { tokio::time::timeout(HANDSHAKE_TIMEOUT, conn.accept_bi()).await })
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "handshake timed out"))?
        .map_err(io::Error::other)?;

    let mut frame = Vec::new();
    rt.block_on(async {
        tokio::time::timeout(HANDSHAKE_TIMEOUT, protocol::read_frame_async(&mut recv, &mut frame)).await
    })
    .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "handshake timed out"))??;

    let name = match ClientMessage::decode(&frame) {
        Some(ClientMessage::Hello { protocol, fingerprint, name, password }) => {
            if protocol != PROTOCOL_VERSION {
                reject(rt, &mut send, &conn, "protocol version mismatch");
                return Ok(None);
            }
            if fingerprint != ctx.fingerprint {
                // Same protocol, different generated content: a join would
                // silently build a DIFFERENT world from the shared seed.
                reject(rt, &mut send, &conn, "world content mismatch (different game/content versions)");
                return Ok(None);
            }
            if *password != *ctx.password {
                reject(rt, &mut send, &conn, "wrong password");
                return Ok(None);
            }
            clean_name(&name)
        }
        _ => {
            reject(rt, &mut send, &conn, "expected hello");
            return Ok(None);
        }
    };
    // Pre-auth window is over; the roster's own MAX_PLAYERS bound takes over.
    drop(slot);
    Ok(Some((conn, send, recv, frame, name, addr)))
}

fn admit_player(
    shared: &Arc<Mutex<State>>,
    ctx: &Ctx,
    rt: &Runtime,
    send: &mut SendStream,
    conn: &quinn::Connection,
    name: &Arc<str>,
) -> Option<(
    u32,
    DVec3,
    f32,
    Vec<(u32, Arc<str>)>,
    Vec<(i32, i32, i32, u32, Arc<str>)>,
    SyncSender<Arc<[u8]>>,
    std::sync::mpsc::Receiver<Arc<[u8]>>,
    Arc<Notify>,
)> {
    // Made before the lock, and the writer spawned only after a slot is
    // secured, so the still-owned `send` handles a "server full" reject
    // directly and reliably.
    let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let kick = Arc::new(Notify::new());

    // One locked scope so the id, spawn, and roster snapshot are consistent.
    let id;
    let spawn;
    let world_day;
    let existing: Vec<(u32, Arc<str>)>;
    let snapshot: Vec<(i32, i32, i32, u32, Arc<str>)>;
    {
        let mut state = shared.lock_recover();
        world_day = state.day_now(ctx.day_secs);
        if state.players.len() >= MAX_PLAYERS {
            drop(state);
            reject(rt, send, conn, "server full");
            return None;
        }
        id = state.next_id;
        state.next_id += 1;
        spawn = spawn_point(ctx.generator.as_ref(), id);

        // Roster only — poses flow through the visibility machinery once the
        // joiner reports their first move, so a far peer isn't a frozen ghost.
        existing = state.players.iter().map(|(&pid, h)| (pid, h.name.clone())).collect();
        // The pooled `Arc<str>` spec goes straight onto the wire message: a
        // built-up world's join snapshot clones refcounts, not strings.
        snapshot = state
            .edits
            .iter()
            .map(|(&(x, y, z), cell)| (x, y, z, cell.rev, cell.spec.clone()))
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
                visible: HashSet::new(),
                out: out.clone(),
                kick: kick.clone(),
                ready: false,
                backlog: Vec::new(),
            },
        );
        // Same lock hold as the roster insert, so the grid never lags the roster.
        state.grid_insert(id, spawn);
    }
    if let Some(hooks) = ctx.hooks.as_ref() {
        hooks.lock_recover().on_join(&JoinFacts {
            player: id,
            name: name.clone(),
            x: block_coord(spawn.x),
            y: block_coord(spawn.y),
            z: block_coord(spawn.z),
        });
    }
    Some((id, spawn, world_day, existing, snapshot, out, rx, kick))
}

fn spawn_writer(
    writer_rt: Arc<Runtime>,
    mut send: SendStream,
    rx: std::sync::mpsc::Receiver<Arc<[u8]>>,
) -> thread::JoinHandle<()> {
    // A write error ends the writer; the connection close at cleanup unblocks
    // one stuck on a slow client's flow-control window. QUIC has no user
    // flush — quinn transmits.
    thread::spawn(move || {
        while let Ok(frame) = rx.recv() {
            if writer_rt.block_on(protocol::write_frame_async(&mut send, &frame)).is_err() {
                return;
            }
            while let Ok(frame) = rx.try_recv() {
                if writer_rt.block_on(protocol::write_frame_async(&mut send, &frame)).is_err() {
                    return;
                }
            }
        }
    })
}

fn client_loop(
    rt: &Runtime,
    recv: &mut quinn::RecvStream,
    frame: &mut Vec<u8>,
    kick: &Notify,
    shared: &Arc<Mutex<State>>,
    ctx: &Ctx,
    id: u32,
) {
    // Voice carries a second, tighter per-second budget of its own: it is far
    // chattier than any other message and must not eat a peer's general budget.
    let mut rate = RateWindow::new(RATE_LIMIT);
    let mut voice_rate = RateWindow::new(VOICE_RATE_LIMIT);
    let mut craft_rate = RateWindow::new(CRAFT_RATE_LIMIT);
    loop {
        // A kick (slow client) wakes this out of the blocking read so cleanup
        // runs; the read future is only ever dropped on that teardown path, so
        // no partial frame desyncs a live stream.
        let read = rt.block_on(async {
            tokio::select! {
                r = protocol::read_frame_async(recv, frame) => Some(r),
                _ = kick.notified() => None,
            }
        });
        match read {
            Some(Ok(())) => {}
            _ => break, // EOF, a malformed length, or a kick: the client is gone.
        }

        let now = Instant::now();
        if !rate.allow(now) {
            continue; // Over budget this second — drop the frame rather than serve a flood.
        }

        let Some(msg) = ClientMessage::decode(frame) else {
            continue;
        };
        match msg {
            ClientMessage::Move { pos, yaw, pitch, stance } => {
                on_move(shared, id, pos, yaw, pitch, stance)
            }
            ClientMessage::Teleport { pos } => on_teleport(shared, ctx, id, pos),
            ClientMessage::Edit { req, x, y, z, expect, spec } => {
                on_edit(shared, ctx.hooks.as_ref(), id, req, x, y, z, expect, &spec)
            }
            ClientMessage::Chat { channel, text } => {
                on_chat(shared, ctx.hooks.as_ref(), id, channel, &text)
            }
            ClientMessage::SetTime { day } => on_set_time(shared, ctx, day),
            ClientMessage::Voice { seq, payload } => {
                if !voice_rate.allow(now) {
                    continue; // Over the voice budget this second — drop silently.
                }
                on_voice(shared, id, seq, payload);
            }
            // Clients ignore swings for unknown peers, so broadcast to
            // everyone-but-sender is safe.
            ClientMessage::Swing => {
                broadcast_all(shared, &ServerMessage::PeerSwing { id }, Some(id))
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
            ClientMessage::Craft {
                origin_spec,
                target_spec,
                event,
                repeat,
            } => {
                if !craft_rate.allow(now) {
                    continue; // Over the workbench budget this second — drop silently.
                }
                on_craft(shared, id, &origin_spec, &target_spec, event, repeat)
            }
        }
    }
}

fn depart(
    shared: &Arc<Mutex<State>>,
    ctx: &Ctx,
    conn: quinn::Connection,
    out: SyncSender<Arc<[u8]>>,
    writer: thread::JoinHandle<()>,
    id: u32,
    name: &Arc<str>,
) {
    let mut left: Option<JoinFacts> = None;
    {
        let mut state = shared.lock_recover();
        if let Some(h) = state.players.remove(&id) {
            left = Some(JoinFacts {
                player: id,
                name: h.name.clone(),
                x: block_coord(h.pos.x),
                y: block_coord(h.pos.y),
                z: block_coord(h.pos.z),
            });
            // h.pos is the last committed one, naming the bucket the grid holds it under.
            state.grid_remove(id, h.pos);
            // Everyone who could see the leaver holds a reciprocal entry that
            // must not dangle.
            for pid in h.visible {
                if let Some(other) = state.players.get_mut(&pid) {
                    other.visible.remove(&id);
                }
            }
        }
    }
    if let (Some(hooks), Some(facts)) = (ctx.hooks.as_ref(), left) {
        hooks.lock_recover().on_leave(&facts);
    }
    // Close the connection FIRST: it errors any write the writer is stuck on for a
    // slow client, so dropping `out` and joining actually completes.
    conn.close(0u32.into(), b"bye");
    drop(out);
    let _ = writer.join();
    broadcast_all(shared, &ServerMessage::PeerLeft { id }, None);
    println!("[-] {name} (#{id}) left ({} online)", online(shared));
}

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
    // A NaN position poisons distance checks/grid keys; a NaN angle propagates
    // into peer interpolation and render matrices even though the server
    // itself does not otherwise use the angle.
    if !pos.x.is_finite()
        || !pos.y.is_finite()
        || !pos.z.is_finite()
        || !yaw.is_finite()
        || !pitch.is_finite()
    {
        return;
    }
    let mut sends = Vec::new();
    {
        let mut state = shared.lock_recover();
        let Some(h) = state.players.get(&id) else { return };
        // Envelope: reject a jump the fastest legitimate movement could not
        // have made, and any position outside the border. The client learns
        // its authoritative position instead of silently diverging.
        let elapsed = h.last_move.elapsed().as_secs_f64().min(MOVE_WINDOW_CAP_SECS);
        let allowed = MAX_MOVE_SPEED * (elapsed + MOVE_SLACK_SECS);
        if outside_world(pos) || h.pos.distance_squared(pos) > allowed * allowed {
            h.correct_position(id, &mut sends);
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
    let mut sends = Vec::new();
    {
        let mut state = shared.lock_recover();
        let Some(h) = state.players.get(&id) else { return };
        if outside_world(pos) || !ctx.allow_teleport {
            h.correct_position(id, &mut sends);
        } else {
            commit_pose(&mut state, id, pos, None, &mut sends);
            // Echo so a client with an in-flight `/tp` can tell accept from a
            // stale movement snap-back: the last Position is the committed pose.
            if let Some(h) = state.players.get(&id) {
                h.correct_position(id, &mut sends);
            }
        }
    }
    dispatch(shared, sends);
}

/// Must run under the state lock; the queued sends go out after it drops.
/// Peers entering/leaving range get both sides' poses/[`PeerExited`], so
/// nobody keeps drawing a frozen ghost.
///
/// [`PeerExited`]: ServerMessage::PeerExited
fn commit_pose(
    state: &mut State,
    id: u32,
    pos: DVec3,
    angles: Option<(f32, f32, Stance)>,
    sends: &mut Vec<PendingSend>,
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
    let (from, to) = (bucket_of(old), bucket_of(pos));
    if from != to {
        state.grid_remove(id, old);
        state.grid_insert(id, pos);
    }
    // Set membership keeps the visibility diff linear in the nearby player count.
    let now_visible = state.visible_from(id, pos);
    let mover_out = state.players[&id].out.clone();
    let departed: Vec<u32> = state.players[&id]
        .visible
        .difference(&now_visible)
        .copied()
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
    // An arriving peer needs the mover's pose AND the mover needs theirs, or
    // the mover keeps hiding them until they next move.
    if now_visible.is_empty() {
        return;
    }
    let move_frame: Arc<[u8]> =
        ServerMessage::PeerMove { id, pos, yaw, pitch, stance }.encode().into();
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

/// A full (or hung-up) queue marks its owner for the kick pass.
fn dispatch(shared: &Arc<Mutex<State>>, sends: Vec<PendingSend>) {
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

/// Resolve a spec a client sent. A configuration the server already knows resolves without
/// growing the table; a novel one (a client's offline product) is interned only while the table
/// keeps [`CLIENT_INTERN_RESERVE`] ids free for the world's own products. `None` = malformed, or
/// novel above the reserve line — the caller refuses the message.
fn resolve_client_spec(registry: &mut BlockRegistry, spec: &str) -> Option<BlockId> {
    resolve_client_spec_within(
        registry,
        spec,
        crate::block::registry::MAX_BLOCK_TYPES - CLIENT_INTERN_RESERVE,
    )
}

/// [`resolve_client_spec`] with an explicit line: novel specs intern only while `block_count()`
/// is below `limit`.
fn resolve_client_spec_within(
    registry: &mut BlockRegistry,
    spec: &str,
    limit: usize,
) -> Option<BlockId> {
    if let Some(id) = registry.lookup_spec(spec) {
        return Some(id);
    }
    if registry.block_count() >= limit {
        return None;
    }
    registry.parse_spec(spec)
}


/// Gates, in order: reach (against the sender's last ACCEPTED position, per
/// [`on_move`]'s envelope), spec validity (parsed/canonicalized by the same
/// rules clients apply), installed [`ServerMod::validate_edit`] hooks, then the
/// expected cell revision — when two players race one cell, the loser is
/// rejected and rolls back.
///
/// A hook [`Verdict::Deny`] uses this same reject path (no ledger write, one
/// `EditAck { accepted: false }`, no broadcast), so the client's
/// `EditRejected.restore` is true iff no newer confirmed revision has landed
/// on the cell — identical to a lost race.
///
/// Hook bodies run **outside** the [`State`] lock: facts are collected under
/// it, the lock is dropped, then the table is called. With no hooks installed
/// the lock is never dropped, matching the pre-seam path.
#[allow(clippy::too_many_arguments)] // edit validation takes each protocol field separately
fn on_edit(
    shared: &Arc<Mutex<State>>,
    hooks: Option<&Mutex<hooks::Table>>,
    id: u32,
    req: u32,
    x: i32,
    y: i32,
    z: i32,
    expect: u32,
    spec: &str,
) {
    let mut state = shared.lock_recover();
    let Some(h) = state.players.get(&id) else { return };
    let ack_to = h.ready.then(|| h.out.clone());
    // Cloned before the registry mut-borrow; skipped when no hooks are installed.
    let name = hooks.is_some().then(|| h.name.clone());
    let reject = |state: &State, out: Option<&SyncSender<Arc<[u8]>>>| {
        let rev = state.edits.get(&(x, y, z)).map_or(0, |c| c.rev);
        if let Some(out) = out {
            let _ = out.try_send(
                ServerMessage::EditAck { req, accepted: false, rev }.encode().into(),
            );
        }
    };
    // Y is unbounded (infinite world height/depth); reach is the real gate.
    // `as f64` so i32::MIN never hits signed-abs overflow; cells past the
    // playable border are still reach-checked (a player AT the border can
    // mine the slack column) but a forged i32::MAX coord is out of reach.
    let target = DVec3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5);
    if spec.len() > MAX_SPEC || h.pos.distance(target) > EDIT_REACH {
        return reject(&state, ack_to.as_ref());
    }
    // Known configurations resolve without growing the table; a novel one is
    // interned only below the reserve line. Only the literal "air" spec may
    // mean AIR, so junk (and the void spelled as a configuration) is rejected
    // instead of silently breaking a block.
    let Some(block) = resolve_client_spec(&mut state.registry, spec) else {
        return reject(&state, ack_to.as_ref());
    };
    if block == crate::block::AIR && spec != "air" {
        return reject(&state, ack_to.as_ref());
    }
    let canonical = crate::save::block_spec(&state.registry, block);
    if let (Some(hooks), Some(name)) = (hooks, name) {
        let intent = EditIntent {
            player: id,
            name,
            x,
            y,
            z,
            spec: Arc::from(canonical.as_str()),
            expect,
        };
        drop(state);
        let verdict = hooks.lock_recover().validate_edit(&intent);
        state = shared.lock_recover();
        if let Verdict::Deny { .. } = verdict {
            return reject(&state, ack_to.as_ref());
        }
        if !state.players.contains_key(&id) {
            return;
        }
    }
    let current = state.edits.get(&(x, y, z)).map_or(0, |c| c.rev);
    if expect != current {
        return reject(&state, ack_to.as_ref());
    }
    let rev = current + 1;
    let Some(spec) = state.intern(&canonical) else {
        return reject(&state, ack_to.as_ref()); // pool at cap: refuse new content
    };
    if let Some(old) = state.edits.insert((x, y, z), Cell { spec: spec.clone(), rev }) {
        state.release(old.spec);
    }
    if block == AIR {
        reactions::on_broken(&mut state.reactions, (x, y, z));
    } else {
        reactions::on_placed(&mut state.reactions, (x, y, z));
    }
    if let Some(out) = ack_to {
        let _ = out
            .try_send(ServerMessage::EditAck { req, accepted: true, rev }.encode().into());
    }
    // The broadcast carries the SAME pooled Arc the ledger stores.
    let msg = ServerMessage::Edit { x, y, z, rev, spec };
    broadcast(&mut state, &msg, |pid, _| pid != id);
}

/// Workbench apply. The server checks the shape (specs resolve, event is a
/// workbench kind, repeat in [`protocol::WORKBENCH_REPEAT`]), that the sender is
/// ready, and that the specs are configurations it knows (or, below the reserve
/// line, may learn) — see [`resolve_client_spec`]. There is no holdings ledger
/// (task 67 has not landed), so it does not check that the sender owns the
/// materials. The result is interned only below the same reserve line.
fn on_craft(
    shared: &Arc<Mutex<State>>,
    id: u32,
    origin_spec: &str,
    target_spec: &str,
    event: u8,
    repeat: u8,
) {
    if origin_spec.len() > MAX_SPEC || target_spec.len() > MAX_SPEC {
        return;
    }
    if !protocol::WORKBENCH_REPEAT.contains(&repeat) {
        return;
    }
    let Some(event) = protocol::workbench_event(event) else {
        return;
    };
    let mut state = shared.lock_recover();
    let out = {
        let Some(h) = state.players.get(&id) else { return };
        if !h.ready {
            return;
        }
        h.out.clone()
    };
    let Some(origin_id) = resolve_client_spec(&mut state.registry, origin_spec) else {
        return;
    };
    let Some(target_id) = resolve_client_spec(&mut state.registry, target_spec) else {
        return;
    };
    if origin_id == AIR || target_id == AIR {
        return; // The void is not a material to work.
    }
    let limit = crate::block::registry::MAX_BLOCK_TYPES - CLIENT_INTERN_RESERVE;
    if state.registry.block_count() >= limit {
        return; // A novel product would eat into the world's reserve.
    }
    let origin = state.registry.configuration(origin_id).clone();
    let target = state.registry.configuration(target_id).clone();
    let Some(result_id) = state.registry.apply_interaction(&origin, &target, event, repeat) else {
        return;
    };
    let result_spec: Arc<str> = state.registry.spec(result_id).into();
    let _ = out.try_send(
        ServerMessage::CraftResult {
            origin_spec: origin_spec.into(),
            target_spec: target_spec.into(),
            event: event as u8,
            repeat,
            result_spec,
        }
        .encode()
        .into(),
    );
}

/// A [`Verdict::Deny`] drops the broadcast and delivers `reason` only to the
/// sender (existing `Chat` frame, `from_id` 0). Hook bodies run outside the
/// [`State`] lock, same rule as [`on_edit`].
fn on_chat(
    shared: &Arc<Mutex<State>>,
    hooks: Option<&Mutex<hooks::Table>>,
    id: u32,
    channel: u8,
    text: &str,
) {
    let text = clean_chat(text);
    if text.is_empty() {
        return;
    }
    let mut state = shared.lock_recover();
    let Some(sender) = state.players.get(&id) else { return };
    let from_name = sender.name.clone();
    let origin = sender.pos;
    let channel = if channel == chat::GLOBAL { chat::GLOBAL } else { chat::LOCAL };
    if let Some(hooks) = hooks {
        let facts = ChatFacts {
            player: id,
            name: from_name.clone(),
            channel,
            text: text.clone(),
        };
        let out = sender.ready.then(|| sender.out.clone());
        drop(state);
        let verdict = hooks.lock_recover().on_chat(&facts);
        if let Verdict::Deny { reason } = verdict {
            if let Some(out) = out {
                let _ = out.try_send(
                    ServerMessage::Chat {
                        from_id: 0,
                        from_name: Arc::from("server"),
                        channel,
                        text: reason,
                    }
                    .encode()
                    .into(),
                );
            }
            return;
        }
        state = shared.lock_recover();
        if !state.players.contains_key(&id) {
            return;
        }
    }
    println!("<{from_name}> {text}");
    let msg = ServerMessage::Chat { from_id: id, from_name, channel, text };
    broadcast(&mut state, &msg, |_, h| {
        channel == chat::GLOBAL || h.pos.distance(origin) <= chat::RADIUS
    });
}

/// Voice is loss-tolerant: `try_send` and DROP on a full/closed queue, never
/// counted toward the slow-client kick ([`kick_slow`]/[`OUT_CAPACITY`]) — a
/// voice flood degrades only that listener's own audio.
fn on_voice(shared: &Arc<Mutex<State>>, id: u32, seq: u32, payload: protocol::VoicePayload) {
    let frame: Arc<[u8]> =
        ServerMessage::PeerVoice { id, epoch: VOICE_EPOCH, seq, payload }.encode().into();
    let state = shared.lock_recover();
    let Some(speaker) = state.players.get(&id) else { return };
    // `visible` IS the interest audience; no separate distance scan needed.
    for &pid in &speaker.visible {
        if let Some(other) = state.players.get(&pid) && other.ready {
            let _ = other.out.try_send(frame.clone());
        }
    }
}

/// Anchors the shared clock so joiners inherit the CURRENT time. A non-finite
/// value is ignored rather than poisoning the shared time.
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

/// Encodes `msg` just once for every recipient. Players whose queue is full
/// are force-closed (they've fallen too far behind).
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

fn broadcast_all(shared: &Arc<Mutex<State>>, msg: &ServerMessage, except: Option<u32>) {
    let mut state = shared.lock_recover();
    broadcast(&mut state, msg, |pid, _| Some(pid) != except);
}

/// Force-close clients that couldn't keep up. Their reader threads then wake, error,
/// and run the normal cleanup path (emitting `PeerLeft`).
fn kick_slow(state: &State, ids: &[u32]) {
    for id in ids {
        if let Some(h) = state.players.get(id) {
            // `notify_one` stores a permit if the reader isn't currently
            // awaiting, so a kick is never missed.
            h.kick.notify_one();
        }
    }
}

/// Only safe on the receiving client's own handler thread (used for the join
/// bootstrap, which must not drop frames).
fn send_blocking(out: &SyncSender<Arc<[u8]>>, msg: &ServerMessage) {
    let frame: Arc<[u8]> = msg.encode().into();
    let _ = out.send(frame);
}

/// QUIC (unlike TCP) can discard buffered stream data when a connection closes,
/// so we wait (bounded) for the peer to close after reading — otherwise a
/// rejected client would see "no reply" instead of the reason.
fn reject(rt: &Runtime, send: &mut SendStream, conn: &quinn::Connection, reason: &str) {
    rt.block_on(async {
        let _ =
            protocol::write_frame_async(send, &ServerMessage::Reject { reason: reason.into() }.encode())
                .await;
        let _ = send.finish();
        let _ = tokio::time::timeout(REJECT_DRAIN, conn.closed()).await;
    });
    println!("[x] rejected a connection: {reason}");
}

/// Scattered a little per id so players don't stack on the exact same block;
/// scans outward for the first column above sea level.
fn spawn_point(generator: &dyn TerrainGenerator, id: u32) -> DVec3 {
    let sx = (id % 8) as i32 - 3;
    let sz = ((id / 8) % 8) as i32 - 3;
    let sea = generator.sea_level();
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

fn online(shared: &Arc<Mutex<State>>) -> usize {
    shared.lock_recover().players.len()
}

fn clean_name(raw: &str) -> Arc<str> {
    let name: String = raw.chars().filter(|c| !c.is_control()).take(MAX_NAME).collect();
    let name = name.trim();
    if name.is_empty() { "player".into() } else { name.into() }
}

fn clean_chat(raw: &str) -> Arc<str> {
    raw.chars().filter(|c| !c.is_control()).take(MAX_CHAT).collect::<String>().trim().into()
}

#[cfg(test)]
mod tests {
    // Test setup (bind/connect/spawn) may unwrap: a panic here is a loud test
    // failure, which is exactly what the deny on the PRODUCTION paths exists
    // to prevent (a client thread silently poisoning the shared state).
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn test_generator() -> crate::world::diffusion::Generator {
        crate::world::diffusion::classic(&mut BlockRegistry::with_builtins(), 4242)
    }

    #[test]
    fn names_are_capped_and_sanitised() {
        assert_eq!(&*clean_name("  guahlg\n "), "guahlg");
        assert_eq!(&*clean_name(""), "player");
        assert_eq!(clean_name(&"x".repeat(100)).len(), MAX_NAME);
    }

    #[test]
    fn chat_is_sanitised() {
        assert_eq!(&*clean_chat("hi\tthere\n"), "hithere");
        assert_eq!(clean_chat(&"a".repeat(500)).len(), MAX_CHAT);
    }

    #[test]
    fn spawn_points_sit_above_the_surface() {
        let terrain = test_generator();
        for id in 1..20 {
            let p = spawn_point(terrain.as_ref(), id);
            let ground = terrain.height(block_coord(p.x), block_coord(p.z));
            assert!(p.y > ground as f64, "spawn should be above ground");
        }
    }

    /// A roster entry for direct state tests. `last_move` starts well in the
    /// past so the first envelope window is at its cap (a fresh anchor allows
    /// only ~30 world units); tests re-age it between deliberate big moves.
    fn test_player(pos: DVec3, out: SyncSender<Arc<[u8]>>, kick: Arc<Notify>) -> PlayerHandle {
        PlayerHandle {
            name: "p".into(),
            pos,
            yaw: 0.0,
            pitch: 0.0,
            stance: Stance::Standing,
            last_move: Instant::now() - Duration::from_secs(10),
            visible: HashSet::new(),
            out,
            kick,
            ready: true,
            backlog: Vec::new(),
        }
    }

    /// A throwaway kick handle for state-only players (never notified).
    fn test_kick() -> Arc<Notify> {
        Arc::new(Notify::new())
    }

    /// A client runtime + endpoint for the raw-handshake tests, wired with the same
    /// accept-any-cert config real clients use.
    fn client_endpoint() -> (Runtime, Endpoint) {
        let rt = Runtime::new().unwrap();
        quic::install_crypto();
        let mut ep = {
            let _g = rt.enter();
            Endpoint::client((Ipv4Addr::UNSPECIFIED, 0).into()).unwrap()
        };
        ep.set_default_client_config(quic::client_config());
        (rt, ep)
    }

    /// Dial, open the reliable stream, send one crafted message, and return the
    /// server's first reply — the raw handshake path `Connection::connect` hides.
    fn raw_reply(addr: SocketAddr, hello: &ClientMessage) -> ServerMessage {
        // The server binds 0.0.0.0; quinn refuses to dial the unspecified address, so
        // reach it over loopback (`handle.addr()` carries only the resolved port).
        let target = SocketAddr::from((Ipv4Addr::LOCALHOST, addr.port()));
        let (rt, ep) = client_endpoint();
        rt.block_on(async {
            let conn = ep.connect(target, "watt").unwrap().await.unwrap();
            let (mut s, mut r) = conn.open_bi().await.unwrap();
            protocol::write_frame_async(&mut s, &hello.encode()).await.unwrap();
            let mut buf = Vec::new();
            protocol::read_frame_async(&mut r, &mut buf).await.unwrap();
            ServerMessage::decode(&buf).unwrap()
        })
    }

    fn hello(name: &str, password: &str, protocol: u32, fingerprint: u64) -> ClientMessage {
        ClientMessage::Hello {
            protocol,
            fingerprint,
            name: name.into(),
            password: password.into(),
        }
    }

    fn reject_reason(addr: SocketAddr, msg: &ClientMessage) -> String {
        match raw_reply(addr, msg) {
            ServerMessage::Reject { reason } => reason.to_string(),
            other => panic!("expected Reject, got {other:?}"),
        }
    }

    fn rock_spec() -> String {
        let mut r = BlockRegistry::with_builtins();
        let id = r
            .intern(&material::Configuration::single(material::Element::new([40, 80, 120, 160])))
            .unwrap();
        r.spec(id)
    }

    fn test_state(players: HashMap<u32, PlayerHandle>) -> State {
        State {
            edits: HashMap::new(),
            spec_pool: HashSet::new(),
            registry: BlockRegistry::with_builtins(),
            players,
            grid: HashMap::new(),
            next_id: 2,
            day: 0.3,
            day_set: Instant::now(),
            reactions: ReactionScheduler::new(),
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
            worldgen: WorldgenKind::Classic,
            diffusion: DiffusionCfg::default(),
            generator: test_generator(),
            hooks: None,
        }
    }

    #[test]
    fn player_ids_never_use_the_reserved_world_id() {
        assert_eq!(WORLD_PLAYER, 0);
        let state = test_state(HashMap::new());
        assert!(state.next_id > WORLD_PLAYER);
    }

    #[test]
    fn client_specs_resolve_known_ids_without_growing_the_table() {
        let mut r = BlockRegistry::with_builtins();
        let known = r
            .intern(&material::Configuration::single(material::Element::new([40, 80, 120, 160])))
            .unwrap();
        let spec = r.spec(known);
        let count = r.block_count();
        // A known configuration resolves even with the table "full" (limit at the current count).
        assert_eq!(resolve_client_spec_within(&mut r, &spec, count), Some(known));
        assert_eq!(resolve_client_spec_within(&mut r, "air", 0), Some(AIR));
        assert_eq!(r.block_count(), count);
        // A novel one interns only below the line, and never more than once.
        let novel = "c:0105060708";
        assert_eq!(resolve_client_spec_within(&mut r, novel, count), None, "at the line: refused");
        assert_eq!(r.block_count(), count, "a refusal does not grow the table");
        let id = resolve_client_spec_within(&mut r, novel, count + 1).expect("below the line");
        assert_eq!(r.block_count(), count + 1);
        assert_eq!(resolve_client_spec_within(&mut r, novel, 0), Some(id), "now known");
        assert_eq!(resolve_client_spec_within(&mut r, "c:zz", usize::MAX), None, "malformed");
        assert_eq!(r.block_count(), count + 1);
    }

    #[test]
    fn a_craft_from_an_unready_player_is_not_evaluated() {
        let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let mut players = HashMap::new();
        let mut p = test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick());
        p.ready = false;
        players.insert(1u32, p);
        let shared = Arc::new(Mutex::new(test_state(players)));
        let count = shared.lock_recover().registry.block_count();
        on_craft(&shared, 1, "c:0105060708", "c:0109090909", 2, 1);
        assert_eq!(shared.lock_recover().registry.block_count(), count, "nothing interned");
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn a_craft_of_the_void_is_refused() {
        let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));
        let rock = rock_spec();
        on_craft(&shared, 1, "air", &rock, 2, 1);
        on_craft(&shared, 1, &rock, "air", 2, 1);
        assert!(rx.try_recv().is_err(), "air is not a material to work");
    }

    #[test]
    fn reaction_snapshots_carry_each_cell_once_with_its_final_content() {
        let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
        let mut state = test_state(players);
        let rock = state.registry.lookup_spec(&rock_spec()).or_else(|| state.registry.parse_spec(&rock_spec())).unwrap();
        let spec = state.intern(&state.registry.spec(rock)).unwrap();
        state.edits.insert((1, 2, 3), Cell { spec: spec.clone(), rev: 2 });
        state.edits.insert((4, 5, 6), Cell { spec, rev: 1 });
        let muts = [
            Mutation { pos: (1, 2, 3), from: AIR, to: rock },
            Mutation { pos: (4, 5, 6), from: AIR, to: rock },
            Mutation { pos: (1, 2, 3), from: rock, to: rock },
        ];
        send_reaction_mutations(&mut state, &muts);
        let frame = rx.try_recv().expect("one snapshot batch");
        let ServerMessage::Snapshot { edits } = ServerMessage::decode(&frame).unwrap() else {
            panic!("expected a Snapshot");
        };
        let cells: Vec<(i32, i32, i32, u32)> = edits.iter().map(|e| (e.0, e.1, e.2, e.3)).collect();
        assert_eq!(cells, vec![(4, 5, 6, 1), (1, 2, 3, 2)], "each cell once, at its last commit");
        assert!(rx.try_recv().is_err(), "no second batch");
    }

    #[test]
    fn a_client_craft_does_not_commit_world_reactions() {
        let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));
        let pending_before = shared.lock_recover().reactions.pending();
        on_craft(&shared, 1, "air", "air", 3, 1);
        on_craft(&shared, 1, "air", "air", 99, 1);
        let state = shared.lock_recover();
        assert_eq!(
            state.reactions.pending(),
            pending_before,
            "ExternallyChanged craft must not queue scheduler events"
        );
        assert!(state.edits.is_empty(), "craft must not write the overlay");
        drop(state);
        assert!(rx.try_recv().is_err(), "malformed craft is silent");
    }

    #[test]
    fn on_edit_queues_place_and_break_events() {
        let (out, _rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));
        let rock = rock_spec();
        on_edit(&shared, None, 1, 1, 8, 20, 8, 0, &rock);
        {
            let state = shared.lock_recover();
            assert_eq!(state.reactions.pending(), 1, "place emits NewContact at the cell");
        }
        on_edit(&shared, None, 1, 2, 8, 20, 8, 1, "air");
        let state = shared.lock_recover();
        assert!(
            state.reactions.pending() >= 6,
            "break emits ExternallyChanged on six neighbours: {}",
            state.reactions.pending()
        );
    }

    #[test]
    fn scripted_reactions_match_a_local_world() {
        use crate::render_config::RenderConfig;
        use crate::sim::reactions::{reactive_region_pair, scripted_run, ReactionScheduler};
        use crate::world::World;

        let mut world = World::with_config(42, RenderConfig::default());
        let (wa, wb) = reactive_region_pair(world.registry_mut());
        let y = world.surface_y(0, 0);
        let local = scripted_run(&mut world, &mut ReactionScheduler::new(), wa, wb, y);

        let mut registry = BlockRegistry::with_builtins();
        let generator = crate::world::diffusion::classic(&mut registry, 42);
        let (sa, sb) = reactive_region_pair(&mut registry);
        let mut state = test_state(HashMap::new());
        state.registry = registry;
        let mut cells = ServerCells {
            state: &mut state,
            generator: &generator,
        };
        let server = scripted_run(&mut cells, &mut ReactionScheduler::new(), sa, sb, y);
        assert_eq!(local, server);
        assert!(!local.is_empty(), "scripted pair must react");
    }

    /// An out-of-reach edit must be rejected; an in-reach one must be recorded.
    /// Exercised directly against the shared state without a socket.
    #[test]
    fn edit_reach_is_enforced() {
        let (out, _rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));

        on_edit(&shared, None, 1, 1, 500, 20, 500, 0, "air"); // far away: rejected
        on_edit(&shared, None, 1, 2, 8, 20, 8, 0, "air"); // in reach: recorded

        let state = shared.lock_recover();
        assert!(state.edits.contains_key(&(8, 20, 8)), "in-reach edit recorded");
        assert!(!state.edits.contains_key(&(500, 20, 500)), "out-of-reach edit dropped");
    }

    #[test]
    fn nonfinite_angles_do_not_enter_authoritative_state() {
        let (out, _rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let start = DVec3::new(8.5, 20.0, 8.5);
        let mut players = HashMap::new();
        let mut p = test_player(start, out, test_kick());
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

    /// A jump no legitimate movement could make is NOT committed — the server
    /// keeps the last accepted position (which edit reach reads) and snaps the
    /// client back with an authoritative `Position`.
    #[test]
    fn implausible_moves_are_rejected_and_corrected() {
        let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let start = DVec3::new(8.5, 20.0, 8.5);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(start, out, test_kick()));
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
        on_edit(&shared, None, 1, 7, 4000, 20, 4000, 0, "air");
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
        players.insert(1u32, test_player(start, out, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));

        on_teleport(&shared, &test_ctx(true), 1, far);
        assert_eq!(shared.lock_recover().players[&1].pos, far, "allowed teleport commits");
        match ServerMessage::decode(&rx.try_recv().expect("accepted teleport echoes Position")) {
            Some(ServerMessage::Position { pos }) => assert_eq!(pos, far),
            other => panic!("expected a Position echo, got {other:?}"),
        }

        on_teleport(&shared, &test_ctx(false), 1, start);
        assert_eq!(shared.lock_recover().players[&1].pos, far, "refused teleport is not committed");
        match ServerMessage::decode(&rx.try_recv().expect("a correction is sent")) {
            Some(ServerMessage::Position { pos }) => assert_eq!(pos, far),
            other => panic!("expected a Position snap-back, got {other:?}"),
        }
    }

    /// The cell revision makes racing edits resolve to exactly one winner, and
    /// the sender's ack — not a broadcast echo — carries the verdict prediction
    /// rolls back on.
    #[test]
    fn edit_revisions_arbitrate_races_and_ack_the_sender() {
        let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));
        let ack = |rx: &std::sync::mpsc::Receiver<Arc<[u8]>>| {
            match ServerMessage::decode(&rx.try_recv().expect("an ack is owed")) {
                Some(ServerMessage::EditAck { req, accepted, rev }) => (req, accepted, rev),
                other => panic!("expected an EditAck, got {other:?}"),
            }
        };

        // First break wins at revision 1.
        on_edit(&shared, None, 1, 10, 8, 20, 8, 0, "air");
        assert_eq!(ack(&rx), (10, true, 1));

        // The racing loser expected revision 0 and is rejected — exactly one
        // reward, and its ack is the rollback signal.
        on_edit(&shared, None, 1, 11, 8, 20, 8, 0, "air");
        assert_eq!(ack(&rx), (11, false, 1));

        // Building on the current revision succeeds.
        let rock = rock_spec();
        on_edit(&shared, None, 1, 12, 8, 20, 8, 1, &rock);
        assert_eq!(ack(&rx), (12, true, 2));

        // Junk specs are rejected before touching the overlay or the pool.
        on_edit(&shared, None, 1, 13, 8, 20, 8, 2, "banana:zzz");
        assert_eq!(ack(&rx), (13, false, 2));
        assert_eq!(shared.lock_recover().edits[&(8, 20, 8)].spec.as_ref(), rock.as_str());
    }

    /// Equivalent spec spellings collapse to ONE canonical pool entry, and a
    /// spec no live cell references leaves the pool instead of leaking.
    #[test]
    fn spec_pool_canonicalizes_and_releases_dead_entries() {
        let (out, _rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));

        // The same spec interned twice: one canonical entry.
        let rock = rock_spec();
        on_edit(&shared, None, 1, 1, 8, 20, 8, 0, &rock);
        on_edit(&shared, None, 1, 2, 8, 21, 8, 0, &rock);
        {
            let state = shared.lock_recover();
            assert_eq!(state.spec_pool.len(), 1, "equivalent spellings share one entry");
            assert_eq!(
                state.edits[&(8, 20, 8)].spec.as_ref(),
                state.edits[&(8, 21, 8)].spec.as_ref()
            );
        }

        // Overwriting both cells strands the old spec: it must leave the pool.
        on_edit(&shared, None, 1, 3, 8, 20, 8, 1, "air");
        on_edit(&shared, None, 1, 4, 8, 21, 8, 1, "air");
        {
            let state = shared.lock_recover();
            assert_eq!(state.spec_pool.len(), 1, "only \"air\" remains interned");
            assert!(state.spec_pool.contains("air"));
        }
    }

    /// Voice relays to the speaker's interest set and nobody else, stamps the
    /// server epoch, and — since it rides the shared queue with a plain
    /// try_send — is simply absent from a peer who cannot hear the speaker.
    #[test]
    fn voice_relays_only_to_the_visible_set() {
        let (out1, _rx1) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let (out2, rx2) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let (out3, rx3) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let mut players = HashMap::new();
        // 1 speaks; 2 is in its interest set; 3 is not.
        let mut p1 = test_player(DVec3::new(0.0, 20.0, 0.0), out1, test_kick());
        p1.visible.insert(2);
        players.insert(1u32, p1);
        players.insert(2u32, test_player(DVec3::new(1.0, 20.0, 0.0), out2, test_kick()));
        players.insert(3u32, test_player(DVec3::new(9e3, 20.0, 0.0), out3, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));

        on_voice(&shared, 1, 42, vec![1, 2, 3].try_into().unwrap());

        match ServerMessage::decode(&rx2.try_recv().expect("the visible peer hears it")) {
            Some(ServerMessage::PeerVoice { id, epoch, seq, payload }) => {
                assert_eq!((id, epoch, seq, payload.as_slice()), (1, VOICE_EPOCH, 42, &[1, 2, 3][..]));
            }
            other => panic!("expected PeerVoice, got {other:?}"),
        }
        assert!(rx3.try_recv().is_err(), "a peer outside interest hears nothing");
    }

    /// A late joiner reads the CURRENT phase, not the last set value.
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
        players.insert(1u32, test_player(start, out, test_kick()));
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

    #[test]
    fn visibility_changes_match_full_roster_distance_checks() {
        let radius = INTEREST_RADIUS;
        let positions = [
            DVec3::ZERO,
            DVec3::new(radius, 0.0, 0.0),
            DVec3::new(-radius, 0.0, 0.0),
            DVec3::new(0.0, radius + 1.0, 0.0),
            DVec3::new(4.0 * radius, 0.0, 0.0),
            DVec3::new(0.0, 0.0, radius / 2.0),
            DVec3::new(2.0 * radius, 0.0, 0.0),
        ];
        let (out, _rx) = sync_channel(OUT_CAPACITY);
        let mut state = test_state(HashMap::new());
        for (index, pos) in positions.into_iter().enumerate() {
            let id = index as u32 + 1;
            state.players.insert(id, test_player(pos, out.clone(), test_kick()));
            state.grid_insert(id, pos);
        }
        // A nearby joiner must finish its snapshot before it receives poses.
        state.players.get_mut(&6).unwrap().ready = false;

        let mut previous = HashSet::new();
        for pos in [
            DVec3::ZERO,
            DVec3::ZERO,
            positions[1],
            positions[4],
            DVec3::ZERO,
            DVec3::new(9.0 * radius, 0.0, 0.0),
        ] {
            let expected: HashSet<u32> = state.players
                .iter()
                .filter(|&(&id, player)| {
                    id != 1 && player.ready && player.pos.distance_squared(pos) <= radius * radius
                })
                .map(|(&id, _)| id)
                .collect();
            let mut expected_sends = Vec::new();
            for &id in previous.difference(&expected) {
                expected_sends.push((id, ServerMessage::PeerExited { id: 1 }));
                expected_sends.push((1, ServerMessage::PeerExited { id }));
            }
            for &id in &expected {
                expected_sends.push((
                    id,
                    ServerMessage::PeerMove {
                        id: 1, pos, yaw: 0.0, pitch: 0.0, stance: Stance::Standing,
                    },
                ));
                if !previous.contains(&id) {
                    let player = &state.players[&id];
                    expected_sends.push((
                        1,
                        ServerMessage::PeerMove {
                            id,
                            pos: player.pos,
                            yaw: player.yaw,
                            pitch: player.pitch,
                            stance: player.stance,
                        },
                    ));
                }
            }

            let mut sends = Vec::new();
            commit_pose(&mut state, 1, pos, None, &mut sends);
            assert_eq!(state.players[&1].visible, expected, "mover at {pos:?}");
            for (&id, player) in &state.players {
                assert_eq!(player.visible.contains(&1), expected.contains(&id), "peer {id}");
            }
            let actual: Vec<_> = sends
                .into_iter()
                .map(|(id, _, frame)| (id, ServerMessage::decode(&frame).unwrap()))
                .collect();
            assert_eq!(actual.len(), expected_sends.len());
            for send in expected_sends {
                assert!(actual.contains(&send), "missing {send:?} at {pos:?}");
            }
            previous = expected;
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

    /// Same protocol, different generated content — the handshake must refuse
    /// the join instead of letting two builds silently diverge on one seed.
    #[test]
    fn mismatched_content_fingerprint_is_rejected() {
        let handle = spawn(0, Config { password: String::new(), seed: 3, ..Config::default() }).unwrap();
        let reason = reject_reason(
            handle.addr(),
            &hello(
                "drifted",
                "",
                PROTOCOL_VERSION,
                crate::net::content_fingerprint() ^ 1,
            ),
        );
        assert!(reason.contains("content"), "unexpected reason: {reason}");
        handle.stop();
    }

    #[test]
    fn classic_fingerprint_is_rejected_from_a_diffusion_server() {
        let handle = spawn(
            0,
            Config {
                password: String::new(),
                seed: 3,
                worldgen: WorldgenKind::Diffusion,
                ..Config::default()
            },
        )
        .unwrap();
        let reason = reject_reason(
            handle.addr(),
            &hello(
                "classic",
                "",
                PROTOCOL_VERSION,
                crate::net::content_fingerprint(),
            ),
        );
        assert!(reason.contains("content"), "unexpected reason: {reason}");
        handle.stop();
    }

    #[test]
    fn welcome_carries_the_servers_worldgen_kind_and_cfg() {
        let diffusion = DiffusionCfg::default();
        let handle = spawn(
            0,
            Config {
                password: String::new(),
                seed: 11,
                worldgen: WorldgenKind::Diffusion,
                diffusion,
                ..Config::default()
            },
        )
        .unwrap();
        let hello = hello(
            "guest",
            "",
            PROTOCOL_VERSION,
            crate::net::content_fingerprint_kind_cfg(WorldgenKind::Diffusion, diffusion),
        );
        match raw_reply(handle.addr(), &hello) {
            ServerMessage::Welcome {
                worldgen,
                diffusion: got,
                seed,
                ..
            } => {
                assert_eq!(seed, 11);
                assert_eq!(worldgen, WorldgenKind::Diffusion);
                assert_eq!(got, diffusion);
            }
            other => panic!("expected Welcome with diffusion kind, got {other:?}"),
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

        // Fill every pre-auth slot with connections that complete the QUIC
        // handshake but never open their stream — the server's `accept_bi` blocks,
        // holding the slot exactly as a silent TCP client did.
        let target = SocketAddr::from((Ipv4Addr::LOCALHOST, addr.port()));
        let (squat_rt, squat_ep) = client_endpoint();
        let squatters: Vec<quinn::Connection> = squat_rt.block_on(async {
            let mut v = Vec::new();
            for _ in 0..HANDSHAKE_CAP {
                v.push(squat_ep.connect(target, "watt").unwrap().await.unwrap());
            }
            v
        });
        // Let the accept loop take them all in before probing past the cap.
        thread::sleep(Duration::from_millis(300));

        // One more must be turned away promptly (a QUIC refusal), not left squatting.
        assert!(
            Connection::connect("127.0.0.1", addr.port(), "extra", "").is_err(),
            "a connection past the handshake cap must be refused"
        );

        // Freeing the squatters must free their slots for a real player. Reuse
        // the graceful-shutdown helper (these are raw quinn connections, not
        // our `Connection` wrapper) — merely dropping the handles would strand
        // the server-side handlers until HANDSHAKE_TIMEOUT, racing this
        // assertion's 5s against a 10s timeout.
        for conn in &squatters {
            crate::net::client::graceful_close(conn, &squat_ep, &squat_rt);
        }
        drop(squatters);
        drop(squat_ep);
        drop(squat_rt);
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

    struct XorShift(u64);

    impl XorShift {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn f64(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (self.next() as f64 / u64::MAX as f64) * (hi - lo)
        }
        fn u32(&mut self, max_excl: u32) -> u32 {
            (self.next() as u32) % max_excl.max(1)
        }
    }

    #[test]
    fn spec_pool_is_bounded_under_unique_mints() {
        let mut state = test_state(HashMap::new());
        for i in 0..MAX_SPEC_POOL {
            assert!(state.intern(&format!("spec-{i}")).is_some(), "slot {i} must intern");
        }
        assert!(state.intern("one-too-many").is_none(), "cap must refuse a new spec");
        assert!(state.intern("spec-0").is_some(), "an already-interned spec still resolves");
        let old = state.spec_pool.get("spec-1").cloned().unwrap();
        state.release(old);
        assert!(state.intern("fresh-after-release").is_some(), "release must free a slot");
        assert_eq!(state.spec_pool.len(), MAX_SPEC_POOL);
    }

    #[test]
    fn interest_at_the_radius_bucket_edges_wrap_and_three_bucket_hops() {
        let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let origin = DVec3::new(0.0, 20.0, 0.0);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(origin, out.clone(), test_kick()));
        players.insert(2u32, test_player(DVec3::new(INTEREST_RADIUS, 20.0, 0.0), out.clone(), test_kick()));
        let mut state = test_state(players);
        state.grid_insert(1, origin);
        state.grid_insert(2, DVec3::new(INTEREST_RADIUS, 20.0, 0.0));
        let mut sends = Vec::new();
        commit_pose(&mut state, 1, origin, None, &mut sends);
        assert!(state.players[&1].visible.contains(&2), "exactly INTEREST_RADIUS is visible");
        assert!(state.players[&2].visible.contains(&1));

        // Bucket edge: INTEREST_RADIUS is the first point of bucket 1.
        let on_edge = DVec3::new(INTEREST_RADIUS, 20.0, 0.0);
        let just_inside = DVec3::new(INTEREST_RADIUS - 1.0, 20.0, 0.0);
        assert_eq!(bucket_of(on_edge), (1, 0));
        assert_eq!(bucket_of(just_inside), (0, 0));

        // i32-wrap-like coordinates clamp through block_coord; membership stays 1:1.
        age_move_state(&mut state, 1);
        let wrap = DVec3::new(crate::math::WORLD_BORDER, 20.0, crate::math::WORLD_BORDER);
        sends.clear();
        commit_pose(&mut state, 1, wrap, None, &mut sends);
        let entries: usize = state.grid.values().map(Vec::len).sum();
        assert_eq!(entries, 2, "wrap-range move must not duplicate grid entries");
        assert!(!state.players[&1].visible.contains(&2), "world-border hop leaves interest");
        assert!(!state.players[&2].visible.contains(&1));
        let exited: Vec<_> = sends
            .iter()
            .filter_map(|(_, _, f)| match ServerMessage::decode(f) {
                Some(ServerMessage::PeerExited { id }) => Some(id),
                _ => None,
            })
            .collect();
        assert!(exited.contains(&1) && exited.contains(&2), "PeerExited reaches every peer");

        // Three buckets in one message (teleport-sized hop).
        let start = DVec3::new(10.0, 20.0, 10.0);
        state.players.get_mut(&1).unwrap().pos = start;
        state.grid.clear();
        state.grid_insert(1, start);
        state.grid_insert(2, DVec3::new(INTEREST_RADIUS, 20.0, 0.0));
        let hop = DVec3::new(10.0 + 3.0 * INTEREST_RADIUS, 20.0, 10.0);
        assert_ne!(bucket_of(start), bucket_of(hop));
        sends.clear();
        commit_pose(&mut state, 1, hop, None, &mut sends);
        assert_eq!(state.grid.get(&bucket_of(hop)).map(Vec::as_slice), Some(&[1u32][..]));
        assert!(!state.grid.contains_key(&bucket_of(start)), "emptied start bucket is dropped");
        let _ = rx;
    }

    fn age_move_state(state: &mut State, id: u32) {
        if let Some(h) = state.players.get_mut(&id) {
            h.last_move = Instant::now() - Duration::from_secs(10);
        }
    }

    #[test]
    fn visible_stays_symmetric_across_random_moves_of_twenty_players() {
        let (out, _rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let mut state = test_state(HashMap::new());
        let mut rng = XorShift::new(0x0020_91A7);
        let span = 6.0 * INTEREST_RADIUS;
        for id in 1..=20u32 {
            let pos = DVec3::new(rng.f64(-span, span), 20.0, rng.f64(-span, span));
            state.players.insert(id, test_player(pos, out.clone(), test_kick()));
            state.grid_insert(id, pos);
        }
        for _ in 0..80 {
            let id = rng.u32(20) + 1;
            let pos = DVec3::new(rng.f64(-span, span), 20.0, rng.f64(-span, span));
            let mut sends = Vec::new();
            commit_pose(&mut state, id, pos, None, &mut sends);
            for (&a, ha) in &state.players {
                for (&b, hb) in &state.players {
                    if a >= b {
                        continue;
                    }
                    assert_eq!(
                        ha.visible.contains(&b),
                        hb.visible.contains(&a),
                        "visibility {a}↔{b} broke after moving {id} to {pos:?}"
                    );
                }
            }
            let entries: usize = state.grid.values().map(Vec::len).sum();
            assert_eq!(entries, 20);
        }
    }

    #[test]
    fn burst_faster_than_cap_then_a_legal_move_corrects_once_then_accepts() {
        let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let start = DVec3::new(8.5, 20.0, 8.5);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(start, out, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));
        let forged = DVec3::new(4000.0, 20.0, 4000.0);
        for _ in 0..8 {
            on_move(&shared, 1, forged, 0.0, 0.0, Stance::Standing);
        }
        assert_eq!(shared.lock_recover().players[&1].pos, start);
        let mut corrections = 0;
        while let Ok(frame) = rx.try_recv() {
            match ServerMessage::decode(&frame) {
                Some(ServerMessage::Position { pos }) => {
                    assert_eq!(pos, start);
                    corrections += 1;
                }
                other => panic!("expected Position, got {other:?}"),
            }
        }
        assert!(corrections >= 1, "the burst must snap back at least once");
        let legal = DVec3::new(10.5, 20.0, 8.5);
        on_move(&shared, 1, legal, 0.0, 0.0, Stance::Standing);
        assert_eq!(shared.lock_recover().players[&1].pos, legal);
        assert!(rx.try_recv().is_err(), "a legal follow-up must not snap back");
    }

    #[test]
    fn long_silence_then_a_legitimate_teleport_obeys_the_flag() {
        let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let start = DVec3::new(8.5, 20.0, 8.5);
        let dest = DVec3::new(50_000.5, 30.0, -2_000.5);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(start, out, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));
        age_move(&shared, 1);
        on_teleport(&shared, &test_ctx(true), 1, dest);
        assert_eq!(shared.lock_recover().players[&1].pos, dest);
        let _ = rx.try_recv();
        on_teleport(&shared, &test_ctx(false), 1, start);
        assert_eq!(shared.lock_recover().players[&1].pos, dest);
        match ServerMessage::decode(&rx.try_recv().expect("refused /tp snaps back")) {
            Some(ServerMessage::Position { pos }) => assert_eq!(pos, dest),
            other => panic!("expected Position, got {other:?}"),
        }
    }

    #[test]
    fn edit_at_exact_reach_is_accepted_and_extreme_coords_do_not_panic() {
        let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let center = DVec3::new(8.5, 20.5, 8.5);
        // A hair inside the sphere so f64 rounding cannot push the construction past
        // `>`; a hair outside must still miss.
        let at_reach = DVec3::new(center.x + EDIT_REACH * 0.999, center.y, center.z);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(at_reach, out, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));
        on_edit(&shared, None, 1, 1, 8, 20, 8, 0, "air");
        assert!(shared.lock_recover().edits.contains_key(&(8, 20, 8)), "exact REACH must land");

        let just_out = DVec3::new(center.x + EDIT_REACH * 1.001, center.y, center.z);
        shared.lock_recover().players.get_mut(&1).unwrap().pos = just_out;
        on_edit(&shared, None, 1, 2, 8, 21, 8, 0, "air");
        assert!(!shared.lock_recover().edits.contains_key(&(8, 21, 8)));

        on_edit(&shared, None, 1, 3, i32::MIN, i32::MIN, i32::MIN, 0, "air");
        on_edit(&shared, None, 1, 4, i32::MAX, i32::MAX, i32::MAX, 0, "air");
        let far = crate::math::WORLD_BORDER as i32 + 64;
        on_edit(&shared, None, 1, 5, far, 20, far, 0, "air");
        assert!(!shared.lock_recover().edits.contains_key(&(far, 20, far)));
        let _ = rx;
    }

    #[test]
    fn bootstrap_backlog_overflow_kicks_without_poisoning_the_lock() {
        let (out, _rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let kick = test_kick();
        let mut players = HashMap::new();
        let mut p = test_player(DVec3::new(0.0, 20.0, 0.0), out, kick.clone());
        p.ready = false;
        players.insert(1u32, p);
        let shared = Arc::new(Mutex::new(test_state(players)));
        for i in 0..=BOOTSTRAP_BACKLOG {
            broadcast_all(&shared, &ServerMessage::Pong { nonce: i as u32 }, None);
        }
        assert!(shared.lock().is_ok(), "kick must not poison the state lock");
        assert!(shared.lock_recover().players.contains_key(&1), "overflow notifies, it does not drop the roster");
        let notified = kick.notified();
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            tokio::time::timeout(Duration::from_millis(50), notified).await.expect("kick must notify")
        });
    }

    #[test]
    fn refused_joins_release_the_pre_auth_slot() {
        let handle = spawn(0, Config { password: "pw".into(), seed: 1, ..Config::default() }).unwrap();
        let addr = handle.addr();
        let fp = crate::net::content_fingerprint();
        let reason = reject_reason(addr, &hello("eve", "nope", PROTOCOL_VERSION, fp));
        assert!(reason.to_lowercase().contains("password"));
        let reason = reject_reason(
            addr,
            &hello("eve", "pw", PROTOCOL_VERSION.wrapping_add(1), fp),
        );
        assert!(reason.to_lowercase().contains("protocol"));
        let reason = reject_reason(addr, &hello("eve", "pw", PROTOCOL_VERSION, fp ^ 1));
        assert!(reason.contains("content"));
        let deadline = Instant::now() + Duration::from_secs(5);
        while handle.handshake_slots() != 0 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(handle.handshake_slots(), 0, "refusals must release the pre-auth slot");
        use crate::net::client::Connection;
        Connection::connect("127.0.0.1", addr.port(), "late", "pw").expect("refusals must free the slot");
        handle.stop();
    }

    #[test]
    fn rate_window_resets_across_the_second_and_cannot_be_gamed_at_the_boundary() {
        let t0 = Instant::now();
        let mut w = RateWindow::new(3);
        assert!(w.allow(t0));
        assert!(w.allow(t0 + Duration::from_millis(1)));
        assert!(w.allow(t0 + Duration::from_millis(2)));
        assert!(!w.allow(t0 + Duration::from_millis(3)), "over budget inside the second");
        assert!(!w.allow(t0 + Duration::from_millis(999)), "boundary-1ms still in the window");
        assert!(w.allow(t0 + Duration::from_secs(1)), "the oldest stamp ages out at +1s");
        assert!(!w.allow(t0 + Duration::from_secs(1)), "aging one stamp frees one slot, not a full refill");
        let mut fresh = RateWindow::new(3);
        for i in 0..3 {
            assert!(fresh.allow(t0 + Duration::from_millis(i)));
        }
        let mut gained = 0u32;
        for ms in 1000..=1002 {
            if fresh.allow(t0 + Duration::from_millis(ms)) {
                gained += 1;
            }
        }
        assert_eq!(gained, 3, "a full second later the budget is whole again");
    }

    #[test]
    fn day_secs_clamps_zero_negative_and_huge() {
        assert_eq!(clamp_day_secs(0.0), 10.0);
        assert_eq!(clamp_day_secs(-40.0), 10.0);
        assert_eq!(clamp_day_secs(f32::NAN), 600.0);
        assert_eq!(clamp_day_secs(f32::INFINITY), 86_400.0);
        assert_eq!(clamp_day_secs(1.0e20), 86_400.0);
        assert_eq!(clamp_day_secs(600.0), 600.0);

        let mut state = test_state(HashMap::new());
        state.day = 0.0;
        state.day_set = Instant::now() - Duration::from_secs(10);
        let zero = state.day_now(0.0);
        let neg = state.day_now(-5.0);
        assert!((zero - 1.0).abs() < 0.05 || (zero - 0.0).abs() < 0.05, "10s of a 10s day wraps, got {zero}");
        assert!((zero - neg).abs() < 1e-3, "zero and negative share the clamp");
        let huge = state.day_now(f32::MAX);
        assert!(huge.abs() < 0.01, "a huge cycle barely advances in 10s, got {huge}");
    }

    fn drain_msgs(rx: &std::sync::mpsc::Receiver<Arc<[u8]>>) -> Vec<ServerMessage> {
        let mut out = Vec::new();
        while let Ok(frame) = rx.try_recv() {
            out.push(ServerMessage::decode(&frame).unwrap());
        }
        out
    }

    #[test]
    fn craft_is_evaluated_on_the_server_and_matches_interact() {
        let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));
        let origin = material::Configuration::single(material::Element::new([40, 80, 120, 160]));
        let target = material::Configuration::single(material::Element::new([80, 40, 160, 120]));
        let (origin_spec, target_spec, expected) = {
            let mut state = shared.lock_recover();
            let oid = state.registry.intern(&origin).unwrap();
            let tid = state.registry.intern(&target).unwrap();
            let os = state.registry.spec(oid);
            let ts = state.registry.spec(tid);
            let law = *state.registry.law();
            let result = crate::block::registry::interact_repeat(
                &law,
                &origin,
                &target,
                material::EventKind::Collision,
                3,
            );
            let rid = state.registry.intern(&result).unwrap();
            let rs = state.registry.spec(rid);
            (os, ts, rs)
        };
        on_craft(&shared, 1, &origin_spec, &target_spec, 2, 3);
        match drain_msgs(&rx).as_slice() {
            [ServerMessage::CraftResult {
                origin_spec: o,
                target_spec: t,
                event,
                repeat,
                result_spec,
            }] => {
                assert_eq!(&**o, origin_spec);
                assert_eq!(&**t, target_spec);
                assert_eq!(*event, 2);
                assert_eq!(*repeat, 3);
                assert_eq!(&**result_spec, expected);
            }
            other => panic!("expected one CraftResult, got {other:?}"),
        }
        on_craft(&shared, 1, &origin_spec, &target_spec, 9, 3);
        on_craft(&shared, 1, &origin_spec, &target_spec, 2, 0);
        on_craft(&shared, 1, "nope", &target_spec, 2, 1);
        assert!(
            drain_msgs(&rx).is_empty(),
            "malformed craft is dropped (well-formedness only; no holdings ledger)"
        );
    }

    /// A hook Deny is the same `EditAck { accepted: false }` a lost race sends:
    /// exactly one reject, no ledger write, no broadcast to peers.
    #[test]
    fn hook_denied_edit_is_one_reject_and_no_broadcast() {
        let (out1, rx1) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let (out2, rx2) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out1, test_kick()));
        players.insert(2u32, test_player(DVec3::new(10.5, 20.0, 8.5), out2, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));
        let (mut rec, log) = hooks::Recording::new("deny");
        rec.deny_edit = true;
        let table = Mutex::new(hooks::Table::new(vec![Box::new(rec)]));

        on_edit(&shared, Some(&table), 1, 42, 8, 20, 8, 0, "air");

        let to_editor = drain_msgs(&rx1);
        assert_eq!(to_editor.len(), 1, "exactly one ack");
        match &to_editor[0] {
            ServerMessage::EditAck { req, accepted, rev } => {
                assert_eq!((*req, *accepted, *rev), (42, false, 0));
            }
            other => panic!("expected EditAck reject, got {other:?}"),
        }
        assert!(drain_msgs(&rx2).is_empty(), "denied edit must not broadcast");
        assert!(shared.lock_recover().edits.is_empty(), "ledger untouched");
        assert_eq!(log.lock().unwrap().len(), 1);
    }

    #[test]
    fn panicking_edit_hook_is_neutralised_and_the_edit_commits() {
        let (out1, rx1) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let (out2, rx2) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out1, test_kick()));
        players.insert(2u32, test_player(DVec3::new(10.5, 20.0, 8.5), out2, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));
        let (mut rec, _) = hooks::Recording::new("boom");
        rec.panic_edit = true;
        let table = Mutex::new(hooks::Table::new(vec![Box::new(rec)]));

        on_edit(&shared, Some(&table), 1, 1, 8, 20, 8, 0, "air");

        match &drain_msgs(&rx1)[..] {
            [ServerMessage::EditAck { req, accepted, rev }] => {
                assert_eq!((*req, *accepted, *rev), (1, true, 1));
            }
            other => panic!("expected one accepted ack, got {other:?}"),
        }
        match &drain_msgs(&rx2)[..] {
            [ServerMessage::Edit { x, y, z, rev, .. }] => {
                assert_eq!((*x, *y, *z, *rev), (8, 20, 8, 1));
            }
            other => panic!("expected one broadcast Edit, got {other:?}"),
        }
        assert!(shared.lock_recover().edits.contains_key(&(8, 20, 8)));
    }

    #[test]
    fn hook_denied_chat_reaches_only_the_sender() {
        let (out1, rx1) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let (out2, rx2) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out1, test_kick()));
        players.insert(2u32, test_player(DVec3::new(9.5, 20.0, 8.5), out2, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));
        let (mut rec, _) = hooks::Recording::new("mute");
        rec.deny_chat = true;
        rec.reason = Arc::from("no talking");
        let table = Mutex::new(hooks::Table::new(vec![Box::new(rec)]));

        on_chat(&shared, Some(&table), 1, chat::GLOBAL, "hello");

        match &drain_msgs(&rx1)[..] {
            [ServerMessage::Chat { from_id, from_name, text, .. }] => {
                assert_eq!(*from_id, 0);
                assert_eq!(&**from_name, "server");
                assert_eq!(&**text, "no talking");
            }
            other => panic!("sender should hear the deny reason, got {other:?}"),
        }
        assert!(drain_msgs(&rx2).is_empty(), "denied chat must not reach peers");
    }

    #[test]
    fn join_leave_hooks_fire_in_order() {
        use crate::net::client::Connection;
        use crate::net::hooks::Recorded;

        let (rec, log) = hooks::Recording::new("rec");
        let handle = spawn(
            0,
            Config { seed: 1, hooks: vec![Box::new(rec)], ..Config::default() },
        )
        .unwrap();
        let port = handle.addr().port();
        let a = Connection::connect("127.0.0.1", port, "alice", "").unwrap();
        let b = Connection::connect("127.0.0.1", port, "bob", "").unwrap();
        let wait = |n: usize| {
            for _ in 0..80 {
                if log.lock().unwrap().len() >= n {
                    return;
                }
                thread::sleep(Duration::from_millis(25));
            }
            panic!("timed out waiting for {n} hook events, have {:?}", log.lock().unwrap());
        };
        wait(2);
        drop(a);
        wait(3);
        drop(b);
        wait(4);
        handle.stop();

        let events = log.lock().unwrap().clone();
        let names: Vec<_> = events
            .iter()
            .map(|e| match e {
                Recorded::Join(f) => format!("join {} {}", f.player, f.name),
                Recorded::Leave(f) => format!("leave {} {}", f.player, f.name),
                other => format!("other {other:?}"),
            })
            .collect();
        assert_eq!(
            names,
            vec![
                "join 1 alice".to_string(),
                "join 2 bob".to_string(),
                "leave 1 alice".to_string(),
                "leave 2 bob".to_string(),
            ]
        );
    }

    #[test]
    fn six_hundred_reaction_mutations_reach_the_client_in_order() {
        let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));
        let spec = {
            let mut state = shared.lock_recover();
            state.intern(&rock_spec()).expect("spec pool")
        };
        let mutations: Vec<Mutation> = (0..600)
            .map(|i| Mutation {
                pos: (i, 20, 0),
                from: AIR,
                to: AIR,
            })
            .collect();
        {
            let mut state = shared.lock_recover();
            for m in &mutations {
                state.edits.insert(
                    m.pos,
                    Cell {
                        spec: spec.clone(),
                        rev: (m.pos.0 as u32) + 1,
                    },
                );
            }
            send_reaction_mutations(&mut state, &mutations);
        }
        let mut got = Vec::new();
        for msg in drain_msgs(&rx) {
            match msg {
                ServerMessage::Snapshot { edits } => got.extend(edits),
                other => panic!("expected Snapshot batches, got {other:?}"),
            }
        }
        assert_eq!(got.len(), 600, "every committed mutation must reach the client");
        for (i, (x, y, z, rev, s)) in got.into_iter().enumerate() {
            assert_eq!((x, y, z), (i as i32, 20, 0));
            assert_eq!(rev, i as u32 + 1);
            assert_eq!(&*s, &*spec);
        }
    }

    #[test]
    fn server_block_evaluates_the_column_once_per_probe() {
        use std::hint::black_box;

        let mut registry = BlockRegistry::with_builtins();
        let g = crate::world::diffusion::classic(&mut registry, 4242);
        let probes: Vec<Pos> = (0..60)
            .flat_map(|x| (0..60).map(move |z| (x, 16, z)))
            .collect();
        assert_eq!(probes.len(), 3600);

        let naive = |g: &crate::world::diffusion::Generator| {
            for &(x, y, z) in &probes {
                black_box(g.block_at(x, y, z, g.height(x, z)));
            }
        };
        let once = |g: &crate::world::diffusion::Generator| {
            for &(x, y, z) in &probes {
                black_box(g.voxel_at(x, y, z));
            }
        };
        naive(&g);
        once(&g);
        let mut before = Duration::ZERO;
        let mut after = Duration::ZERO;
        for _ in 0..3 {
            let t0 = Instant::now();
            naive(&g);
            before += t0.elapsed();
            let t1 = Instant::now();
            once(&g);
            after += t1.elapsed();
        }
        println!("server_block column eval: before={before:?} after={after:?}");
        for &(x, y, z) in &probes {
            assert_eq!(
                g.block_at(x, y, z, g.height(x, z)),
                g.voxel_at(x, y, z),
                "voxel_at must match height+block_at at ({x},{y},{z})"
            );
        }
        assert!(
            after < before,
            "one column eval per probe must beat height+block_at ({after:?} vs {before:?})"
        );
    }
}
