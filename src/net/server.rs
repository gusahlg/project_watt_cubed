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
//! [`INTEREST_RADIUS`], via a 3D bucket grid ([`State::grid`]): a move consults
//! only the mover's 3×3×3 bucket neighbourhood instead of scanning the roster.
//!
//! **Trust.** Joins are password-gated and version-checked; frames are size-capped
//! by [`protocol`]; every client is rate-limited; every edit is bounds- and
//! reach-validated against the sender's own reported position.
//!
//! The mod list on `Hello` is what an honest client reports. A modified client
//! can lie about it. What stops mod tools is enforced here: teleport permission,
//! the speed cap (flight, and cruise when the cap is below the game maximum),
//! operator-only time, edit reach, and the movement envelope. A reported move whose
//! body overlaps solid ground is snapped back unless [`NoclipPolicy`] allows it.
//! Gravity is not simulated; the envelope's gravity bound is only a limit on how
//! fast a reported velocity may grow.
//!
//! **Server mods.** [`Config::hooks`] is a [`ServerMod`] table (plain-data
//! arguments, no protocol change). Calls run outside the [`State`] lock.
use std::collections::{HashMap, HashSet, VecDeque};
use std::io;
use std::net::SocketAddr;
#[cfg(test)]
use std::net::Ipv4Addr;
use std::path::PathBuf;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use quinn::{Endpoint, Incoming, SendStream};
use tokio::runtime::Runtime;
use tokio::sync::Notify;
use glam::DQuat;
use voxel_engine::{DVec3, Vec3};

use crate::coord::{BlockCoord, Face};
use crate::gravity::Field;
use crate::math::block_coord;
use crate::player::{self, standing_pose};

use crate::block::registry::{BlockId, BlockRegistry, AIR};
use crate::net::hooks;
use crate::sim::reactions::{self, CellStore, Contact, Mutation, Pos, ReactionScheduler};
pub(crate) use crate::net::hooks::{ChatFacts, EditIntent, JoinFacts, ServerMod, Verdict};
use crate::net::persist::{self, Store};
use crate::net::protocol::{self, ClientMessage, ModOffer, ServerMessage};
use crate::net::{MAX_CHAT, MAX_FRAME, MAX_NAME, MAX_SPEC, PROTOCOL_VERSION, chat, quic};
use crate::presence::Stance;
use crate::world::seam::Seams;
use crate::world::terrain::TerrainCfg;
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
/// Per message type, per second. The client's move cadence is 30 Hz; 40 leaves slack
/// so an honest client is not rejected. Chat, swing, edit, and time are human rates.
const CHAT_RATE: u32 = 5;
const SWING_RATE: u32 = 10;
const EDIT_RATE: u32 = 20;
const SET_TIME_RATE: u32 = 1;
const MOVE_RATE: u32 = 40;
const PING_RATE: u32 = 5;
/// A refused teleport is still answered, so a dropped reply cannot freeze `/tp`.
const TELEPORT_RATE: u32 = 4;
/// A 20 ms capture cadence is ~50 frames/s sustained; 100 leaves headroom for
/// bursts. Excess frames are dropped silently — channel traffic is loss-tolerant.
const CHANNEL_RATE_LIMIT: u32 = 100;
/// Mod frames one connection may send per second over all its channels, charged before
/// the channel's own window, so switching names buys no extra rate.
const MOD_DATA_RATE: u32 = 2 * CHANNEL_RATE_LIMIT;
/// Distinct channel names one connection may use. A new name past this is dropped.
const MAX_CHANNELS: usize = 32;
/// Tool uses one connection may send per second: each one evaluates the law and may intern two
/// configurations under the [`State`] lock, so the budget is a human's swing rate, not a flood.
const TOOL_RATE_LIMIT: u32 = 12;
/// Ids the material table keeps for the world's own products (reactions, generation): a spec a
/// A novel spec a client sends is interned only while at least this many ids stay free,
/// so one client cannot fill the table (see [`take_novel_spec`]).
const CLIENT_INTERN_RESERVE: usize = crate::block::registry::MAX_BLOCK_TYPES / 4;
const INTEREST_RADIUS: f64 = 160.0 * crate::math::PER_METER;
/// Squared once so the hot per-listener check in [`on_move`] needs no sqrt.
const INTEREST_RADIUS_SQ: f64 = INTEREST_RADIUS * INTEREST_RADIUS;
/// A little past the client's own reach constant.
const EDIT_REACH: f64 = 8.0 * crate::math::PER_METER;
/// The distance budget's burst, in seconds at the envelope speed, so a stalled
/// stream that lands several moves at once still drains at speed.
const MOVE_SLACK_SECS: f64 = 0.3;
/// Past this, elapsed time stops buying displacement (an idle client can't
/// bank a cross-map jump allowance).
const MOVE_WINDOW_CAP_SECS: f64 = 2.0;
/// The smallest burst: the old 80 m/s slack, so a standing step is not a teleport.
/// It is spent once, not granted per move.
const MOVE_FLOOR: f64 = 80.0 * crate::math::PER_METER * MOVE_SLACK_SECS;
/// The swept body check samples the path at most this far apart, the client collision's substep.
const SWEEP_STEP: f64 = 0.5;
/// Longest path the swept body check walks. A longer move fails closed, except a cruise move,
/// which passes through everything on the client and is checked at its destination only.
const SWEEP_LIMIT: f64 = 512.0;
/// How fast a reported velocity may grow between moves, above the speed the
/// client already has. Vacuum falls have no terminal speed; this is several
/// times surface gravity so an honest fall fits and a forged one does not.
const GRAVITY_BOUND: f64 = 8.0 * crate::player::STANDARD_GRAVITY;
/// Novel configurations one client may intern. Past this, further novel specs
/// are refused; known specs and `air` are not counted.
const NOVEL_SPEC_QUOTA: u32 = 64;
/// Matches the client palette cap ([`format::MAX_SPECS`](crate::save::format));
/// a hostile client can exhaust neither server memory nor peers' palettes.
const MAX_SPEC_POOL: usize = 16_384;
/// Snapshot payload: tag + u32 count, then each edit is 3×i32 + u32 rev + u16 length + spec.
const SNAPSHOT_HEAD: usize = 1 + 4;
const SNAPSHOT_EDIT_FIXED: usize = 12 + 4 + 2;
/// Who may teleport. `All` is the integrated host and [`Config::default`], so
/// existing sessions keep today's behaviour. A dedicated server passes [`Ops`](Self::Ops).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TeleportPolicy {
    Off,
    Ops,
    All,
}

/// Who may pass through solid ground. Same three settings as [`TeleportPolicy`].
/// [`Config::default`] is [`All`](Self::All) so existing sessions are not suddenly
/// collision-checked. A dedicated server passes [`Ops`](Self::Ops).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoclipPolicy {
    Off,
    Ops,
    All,
}

/// Matches the client's [`World`](crate::world::World::new) so server spawn
/// heights land on real ground.
pub struct Config {
    /// Empty means no password is required.
    pub password: String,
    pub seed: i64,
    pub day_secs: f32,
    pub teleport: TeleportPolicy,
    /// Who may move through solid ground. See [`NoclipPolicy`].
    pub noclip: NoclipPolicy,
    pub worldgen: WorldgenKind,
    pub terrain: TerrainCfg,
    /// Server-side mods (`validate_edit`, join/leave, `on_chat`). Empty by
    /// default — this crate ships no implementations. Hook bodies run outside
    /// the roster lock.
    pub hooks: Vec<Box<dyn ServerMod>>,
    /// Save file. `None` is an unsaved world (tests). A missing file is created
    /// from the flags; an existing file's seed and generator replace them.
    pub world: Option<PathBuf>,
    /// Operator names. Compared case-insensitively after [`clean_name`].
    pub ops: Vec<String>,
    /// Operators who prove it with the chat line `/op <secret>`: (name, secret).
    /// A name here needs its secret even when [`ops`](Self::ops) lists it too.
    pub op_secrets: Vec<(String, String)>,
    /// Movement-envelope cap, in world units per second. The default is
    /// [`crate::player::MAX_SPEED`], which leaves cruise at its own ceiling.
    /// A lower cap also bounds cruise.
    pub max_speed: f64,
    /// When non-empty, an enabled mod must be in this list.
    pub mods_allow: Vec<String>,
    /// Enabled mods in this list are refused.
    pub mods_deny: Vec<String>,
    /// Log when a stored seed or generator differs from the flags.
    pub warn_world_overrides: bool,
    /// How often a world file is written. Also written on shutdown.
    pub autosave_every: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            password: String::new(),
            seed: 0,
            day_secs: 600.0, // matches the client's default DayLength
            teleport: TeleportPolicy::All,
            noclip: NoclipPolicy::All,
            worldgen: WorldgenKind::Diffusion,
            terrain: TerrainCfg::default(),
            hooks: Vec::new(),
            world: None,
            ops: Vec::new(),
            op_secrets: Vec::new(),
            max_speed: crate::player::MAX_SPEED,
            mods_allow: Vec::new(),
            mods_deny: Vec::new(),
            warn_world_overrides: false,
            autosave_every: Duration::from_secs(180),
        }
    }
}

/// Same range the client clock uses (`DayLength::clamped`): never zero (which
/// would stall or desync the shared sky) and never a multi-day real-time cycle.
fn clamp_day_secs(s: f32) -> f32 {
    if s.is_nan() { 600.0 } else { s.clamp(10.0, 86_400.0) }
}

/// The connection's [`MOD_DATA_RATE`] window, then one [`RateWindow`] per channel name.
/// The first packet of a channel allocates the slot, up to [`MAX_CHANNELS`]; later
/// packets scan the small vec.
struct ChannelBudget {
    total: RateWindow,
    windows: Vec<(Arc<str>, RateWindow)>,
}

impl ChannelBudget {
    fn new() -> Self {
        Self { total: RateWindow::new(MOD_DATA_RATE), windows: Vec::new() }
    }

    fn allow(&mut self, channel: &protocol::Channel, now: Instant) -> bool {
        if !self.total.allow(now) {
            return false;
        }
        let name = channel.as_str();
        if let Some((_, window)) = self.windows.iter_mut().find(|(key, _)| key.as_ref() == name) {
            return window.allow(now);
        }
        if self.windows.len() >= MAX_CHANNELS {
            return false;
        }
        let mut window = RateWindow::new(CHANNEL_RATE_LIMIT);
        let ok = window.allow(now);
        self.windows.push((channel.share(), window));
        ok
    }
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

/// One window per message kind. Cruise, hello, tool use, and mod channels are not here.
struct KindBudget {
    chat: RateWindow,
    swing: RateWindow,
    edit: RateWindow,
    set_time: RateWindow,
    movement: RateWindow,
    ping: RateWindow,
    teleport: RateWindow,
}

impl KindBudget {
    fn new() -> Self {
        Self {
            chat: RateWindow::new(CHAT_RATE),
            swing: RateWindow::new(SWING_RATE),
            edit: RateWindow::new(EDIT_RATE),
            set_time: RateWindow::new(SET_TIME_RATE),
            movement: RateWindow::new(MOVE_RATE),
            ping: RateWindow::new(PING_RATE),
            teleport: RateWindow::new(TELEPORT_RATE),
        }
    }
}

/// What to do with one client frame against its kind's budget.
enum Charge {
    /// Under budget, or a kind that is not counted here.
    Pass,
    /// Over budget and safe to ignore.
    Drop,
    /// Over budget, but the client is waiting on an answer.
    Answer,
}

fn charge(budgets: &mut KindBudget, msg: &ClientMessage, now: Instant) -> Charge {
    let (window, answer) = match msg {
        ClientMessage::Cruise { .. }
        | ClientMessage::Hello { .. }
        | ClientMessage::ToolUse { .. }
        | ClientMessage::ModData { .. } => return Charge::Pass,
        ClientMessage::Chat { .. } => (&mut budgets.chat, false),
        ClientMessage::Swing => (&mut budgets.swing, false),
        ClientMessage::Edit { .. } => (&mut budgets.edit, true),
        ClientMessage::SetTime { .. } => (&mut budgets.set_time, false),
        ClientMessage::Move { .. } => (&mut budgets.movement, false),
        ClientMessage::Ping { .. } => (&mut budgets.ping, false),
        ClientMessage::Teleport { .. } => (&mut budgets.teleport, true),
    };
    if window.allow(now) { Charge::Pass } else if answer { Charge::Answer } else { Charge::Drop }
}

struct Ctx {
    password: String,
    seed: i64,
    content: crate::net::ContentId,
    day_secs: f32,
    teleport: TeleportPolicy,
    noclip: NoclipPolicy,
    worldgen: WorldgenKind,
    terrain: TerrainCfg,
    generator: crate::world::terrain::Generator,
    /// The client's seam reads: a storage cell just past a chart's box is the neighbour chart's cell.
    seams: Seams,
    /// `None` when [`Config::hooks`] is empty so the default server never
    /// touches a second lock. When `Some`, hook calls happen *outside* the
    /// [`State`] lock: collect facts under it, drop it, then run the table.
    hooks: Option<Mutex<hooks::Table>>,
    /// Lowercased operator names.
    ops: Vec<String>,
    /// Lowercased names and their `/op` secrets.
    op_secrets: Vec<(String, String)>,
    mods_allow: Vec<String>,
    mods_deny: Vec<String>,
    store: Option<Arc<Store>>,
}

struct PlayerHandle {
    /// Interned once at join; every roster/join/chat broadcast that carries
    /// it is a refcount bump, never a per-recipient allocation.
    name: Arc<str>,
    pos: DVec3,
    yaw: f32,
    pitch: f32,
    frame: DQuat,
    velocity: Vec3,
    up: Face,
    stance: Stance,
    /// The movement envelope's time anchor.
    last_move: Instant,
    /// Distance banked at `last_move`, spent by moves and refilled at the envelope speed.
    budget: f64,
    /// Proved an operator secret with `/op`.
    op: bool,
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
    /// Frames held until bootstrap finishes, with the instant they were queued, oldest first.
    backlog: VecDeque<Queued>,
    backlog_bytes: usize,
    /// Set by [`kick_slow`] and by a bootstrap send that misses its deadline.
    /// The reader is not in [`client_loop`] yet during bootstrap, so the kick
    /// [`Notify`] alone would not be watched.
    kicked: Arc<AtomicBool>,
    /// Physical cells the body occupied on the last accepted move. A later move
    /// only queries cells that are not already here.
    occupied: Vec<(i32, i32, i32)>,
    /// Declared [`ClientMessage::Cruise`]. The envelope uses `cruise_speed` as its cap.
    cruising: bool,
    cruise_speed: f64,
    /// Novel configurations this client has interned. Capped at [`NOVEL_SPEC_QUOTA`].
    novel: u32,
    /// Peer ids whose `PeerJoined` this client has already been queued. A join
    /// both snapshots the roster and may race another joiner's broadcast.
    announced: HashSet<u32>,
}

/// A recipient and its encoded frame, queued after releasing the state lock.
type PendingSend = (u32, SyncSender<Arc<[u8]>>, Arc<[u8]>);

impl PlayerHandle {
    fn correct_position(&self, id: u32, sends: &mut Vec<PendingSend>) {
        if self.ready {
            let frame = ServerMessage::Position { pos: self.pos, frame: self.frame, up: self.up }.encode().into();
            sends.push((id, self.out.clone(), frame));
        }
    }
}

/// A joining client buffers broadcasts until Welcome and the overlay are queued.
/// The bound is bytes and age, not a frame count: cosmetic frames are dropped
/// first, and an essential frame that does not fit or outlives the age kicks.
/// Several full frames fit, so one large reaction batch never kicks on its own.
const BACKLOG_BYTES: usize = 4 * MAX_FRAME;
const BACKLOG_AGE: Duration = Duration::from_secs(15);
/// How long a bootstrap send may wait for queue space before the joiner is dropped.
const SEND_DEADLINE: Duration = Duration::from_secs(8);
/// The reaction thread re-sends the shared clock this often so clients do not drift.
const TIME_BROADCAST: Duration = Duration::from_secs(60);
/// Body cells tested for a solid overlap. A larger box fails closed.
const BODY_CELL_CAP: usize = 64;

/// One frame waiting out a join, and when it was queued.
struct Queued {
    at: Instant,
    frame: Arc<[u8]>,
}

/// The optimistic-concurrency token racing edits compare against.
struct Cell {
    spec: Arc<str>,
    rev: u32,
    /// An edit put back the generated block: kept for its revision, left out of the world file.
    natural: bool,
}

struct State {
    edits: HashMap<(i32, i32, i32), Cell>,
    /// Distinct CANONICAL spec strings and how many live cells name them.
    /// Snapshot clones and broadcasts hold their own `Arc`s and do not count:
    /// an entry leaves when the cell count hits zero. Capped at [`MAX_SPEC_POOL`].
    spec_pool: HashMap<Arc<str>, u32>,
    /// The same compiled palette clients build, so specs validate/canonicalize
    /// under EXACTLY the rules clients apply.
    registry: BlockRegistry,
    players: HashMap<u32, PlayerHandle>,
    /// Bucket key → ids standing in it, keyed by [`bucket_of`]. Buckets are
    /// exactly one [`INTEREST_RADIUS`] wide on each axis, so anyone in range of
    /// a mover lives in its 3×3×3 neighbourhood; [`on_move`] still applies the
    /// exact per-player distance check, so the grid only narrows candidates,
    /// never the audience. Invariant: exactly one entry per connected player,
    /// updated under the same lock hold as the position change it mirrors;
    /// empty buckets are removed eagerly so churn can never leak keys.
    grid: HashMap<(i32, i32, i32), Vec<u32>>,
    next_id: u32,
    /// The `[0,1)` day fraction current at `day_set`. The server advances it
    /// only when asked ([`State::day_now`]), so a late joiner receives the
    /// CURRENT phase rather than whatever `/time` last set.
    day: f32,
    day_set: Instant,
    /// Server-authoritative reaction scheduler. Clients never run one.
    reactions: ReactionScheduler,
    /// Envelope cap copied from [`Config::max_speed`] at spawn.
    max_speed: f64,
    /// Test hook: the next reaction tick panics once, then clears the flag.
    #[cfg(test)]
    panic_tick: bool,
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
            for dy in -1..=1i32 {
                for dz in -1..=1i32 {
                    let key = (at.0.wrapping_add(dx), at.1.wrapping_add(dy), at.2.wrapping_add(dz));
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
        }
        visible
    }

    fn day_now(&self, day_secs: f32) -> f32 {
        let elapsed = self.day_set.elapsed().as_secs_f32();
        (self.day + elapsed / clamp_day_secs(day_secs)).rem_euclid(1.0)
    }

    /// `None` at the [`MAX_SPEC_POOL`] cap. An existing spec still resolves at the cap,
    /// and its cell count goes up by one.
    fn intern(&mut self, spec: &str) -> Option<Arc<str>> {
        if let Some(n) = self.spec_pool.get_mut(spec) {
            *n = n.saturating_add(1);
            return self.spec_pool.get_key_value(spec).map(|(k, _)| k.clone());
        }
        if self.spec_pool.len() >= MAX_SPEC_POOL {
            return None;
        }
        let shared: Arc<str> = Arc::from(spec);
        self.spec_pool.insert(shared.clone(), 1);
        Some(shared)
    }

    /// One live cell stopped naming `old`. The pool entry leaves at zero,
    /// whatever other `Arc` clones (a snapshot list, a test) still exist.
    fn release(&mut self, old: Arc<str>) {
        let Some(n) = self.spec_pool.get_mut(old.as_ref()) else { return };
        if *n <= 1 {
            self.spec_pool.remove(old.as_ref());
        } else {
            *n -= 1;
        }
    }
}

/// Ledger + generator as a [`CellStore`]: a cell not in the ledger reads from
/// the generator, so the infinite world is defined without loading chunks.
struct ServerCells<'a> {
    state: &'a mut State,
    generator: &'a crate::world::terrain::Generator,
}

fn server_block(
    state: &State,
    generator: &crate::world::terrain::Generator,
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
        if let Some(old) = self.state.edits.insert(pos, Cell { spec, rev, natural: false }) {
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
    let mut next_clock = Instant::now() + TIME_BROADCAST;
    while !shutdown.load(Ordering::Relaxed) {
        let start = Instant::now();
        // The tick restores the scheduler itself. This catch covers a panic after that
        // (the broadcast) so the thread, and the scheduler, both stay.
        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| run_reactions(&shared, &ctx))) {
            eprintln!("reaction tick panicked: {}", panic_text(&payload));
        }
        if Instant::now() >= next_clock {
            if let Err(payload) = catch_unwind(AssertUnwindSafe(|| broadcast_clock(&shared, &ctx))) {
                eprintln!("time broadcast panicked: {}", panic_text(&payload));
            }
            next_clock = Instant::now() + TIME_BROADCAST;
        }
        if let Some(rest) = period.checked_sub(start.elapsed()) {
            thread::sleep(rest);
        }
    }
}

fn panic_text(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// One sim tick of the scheduler. Committed mutations are [`ServerMessage::Snapshot`]
/// batches attributed to [`WORLD_PLAYER`], broadcast to every ready client. The
/// scheduler budget already bounds the count; every commit is sent. A panic inside
/// the tick puts the scheduler back before the error propagates.
fn run_reactions(shared: &Arc<Mutex<State>>, ctx: &Ctx) {
    let mut state = shared.lock_recover();
    let force = take_panic_tick(&mut state);
    if state.reactions.pending() == 0 && !force {
        return;
    }
    let budget = reactions::Budget::DEFAULT;
    let mut sched = std::mem::take(&mut state.reactions);
    let tick = catch_unwind(AssertUnwindSafe(|| {
        if force {
            panic!("reaction tick");
        }
        let mut cells = ServerCells {
            state: &mut state,
            generator: &ctx.generator,
        };
        sched.tick(&mut cells, budget)
    }));
    state.reactions = sched;
    match tick {
        Ok(mutations) => send_reaction_mutations(&mut state, &mutations),
        Err(payload) => eprintln!("reaction tick panicked: {}", panic_text(&payload)),
    }
}

fn take_panic_tick(state: &mut State) -> bool {
    #[cfg(test)]
    {
        let force = state.panic_tick;
        state.panic_tick = false;
        force
    }
    #[cfg(not(test))]
    {
        let _ = state;
        false
    }
}

/// The shared clock, sampled now and sent to every connected player.
fn broadcast_clock(shared: &Arc<Mutex<State>>, ctx: &Ctx) {
    let mut state = shared.lock_recover();
    let day = state.day_now(ctx.day_secs);
    broadcast(&mut state, &ServerMessage::Time { day, day_secs: ctx.day_secs }, |_, _| true);
}

/// Authoritative overlay edits from one scheduler tick, as snapshot batches
/// (the client applies [`ServerMessage::Snapshot`] after bootstrap), one entry
/// per distinct cell. One `S_Edit` per mutation would overflow [`OUT_CAPACITY`]
/// on two full ticks.
fn send_reaction_mutations(state: &mut State, mutations: &[Mutation]) {
    if mutations.is_empty() {
        return;
    }
    // A cell committed by several contacts in one turn is sent once, with its
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
    for_snapshot_batches(&edits, |batch| {
        broadcast(
            state,
            &ServerMessage::Snapshot { edits: batch.to_vec() },
            |pid, _| pid != WORLD_PLAYER,
        );
    });
}

/// Split `edits` so each [`ServerMessage::Snapshot`] encodes to at most [`MAX_FRAME`].
/// A single edit is always emitted, so one oversized spec cannot loop forever.
fn for_snapshot_batches(
    edits: &[(i32, i32, i32, u32, Arc<str>)],
    mut emit: impl FnMut(&[(i32, i32, i32, u32, Arc<str>)]),
) {
    let mut start = 0;
    while start < edits.len() {
        let mut end = start + 1;
        let mut size = SNAPSHOT_HEAD + SNAPSHOT_EDIT_FIXED + edits[start].4.len();
        while end < edits.len() {
            let add = SNAPSHOT_EDIT_FIXED + edits[end].4.len();
            if size + add > MAX_FRAME {
                break;
            }
            size += add;
            end += 1;
        }
        emit(&edits[start..end]);
        start = end;
    }
}

/// The interest-grid bucket containing `pos`. Goes through [`block_coord`]'s
/// clamped floor (not truncation) so negative coordinates bucket consistently
/// and a hostile-but-finite huge coordinate can't overflow the i32 key —
/// insert and remove share this one mapping, so the grid stays consistent.
fn bucket_of(pos: DVec3) -> (i32, i32, i32) {
    (
        block_coord(pos.x / INTEREST_RADIUS),
        block_coord(pos.y / INTEREST_RADIUS),
        block_coord(pos.z / INTEREST_RADIUS),
    )
}

fn outside_world(pos: DVec3) -> bool {
    pos.x.abs() > crate::math::WORLD_BORDER
        || pos.y.abs() > crate::math::WORLD_BORDER
        || pos.z.abs() > crate::math::WORLD_BORDER
}

/// A running server. [`stop`](ServerHandle::stop) closes every connection, saves
/// the world, and joins the server threads so the port can be bound again.
/// Dropping the handle does the same.
pub(crate) struct ServerHandle {
    shutdown: Arc<AtomicBool>,
    stopped: AtomicBool,
    addr: SocketAddr,
    /// Dropped at the end of [`stop`](Self::stop), after the endpoint, so quinn's
    /// driver can release the socket while a runtime still exists.
    rt: Mutex<Option<Arc<Runtime>>>,
    endpoint: Mutex<Option<Endpoint>>,
    accept: Mutex<Option<JoinHandle<()>>>,
    reactions: Mutex<Option<JoinHandle<()>>>,
    autosave: Mutex<Option<JoinHandle<()>>>,
    save_gate: Arc<Mutex<()>>,
    state: Arc<Mutex<State>>,
    ctx: Arc<Ctx>,
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

    /// Close every connection with "server shutting down", save, and join threads.
    /// Closing first means no edit is acknowledged after the save captures the ledger:
    /// once the endpoint is closed, no ack reaches a client. A second call does nothing.
    pub fn stop(&self) {
        if self.stopped.swap(true, Ordering::SeqCst) {
            return;
        }
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(endpoint) = self.endpoint.lock_recover().take() {
            endpoint.close(0u32.into(), b"server shutting down");
        }
        self.save_now();
        if let Some(handle) = self.accept.lock_recover().take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.reactions.lock_recover().take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.autosave.lock_recover().take() {
            let _ = handle.join();
        }
        // Last runtime clone: its drop finishes the driver and frees the port.
        drop(self.rt.lock_recover().take());
    }

    /// Write the world file now. No file is configured, or a save is already
    /// in progress, and this waits for that write then writes again.
    pub fn save_now(&self) {
        save_world(&self.state, &self.ctx, &self.save_gate);
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

impl Drop for ServerHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Bind to port 0 to let the OS pick a free port.
pub(crate) fn spawn(port: u16, config: Config) -> io::Result<ServerHandle> {
    let terrain = config.terrain.clamp();
    let flags = persist::Flags {
        seed: config.seed,
        worldgen: config.worldgen,
        terrain,
        warn: config.warn_world_overrides,
    };
    let loaded = match &config.world {
        Some(path) => persist::load(path, &flags).map_err(|e| io::Error::other(e))?,
        None => persist::fresh(&flags),
    };
    let rt = Arc::new(Runtime::new()?);
    let endpoint = {
        // Must run inside the runtime: construction spawns quinn's UDP driver,
        // and `default_runtime` only sees Tokio while that context is entered.
        let _guard = rt.enter();
        let socket = quic::bind_dual_stack(port)?;
        let runtime = quinn::default_runtime()
            .ok_or_else(|| io::Error::other("no async runtime found"))?;
        Endpoint::new(quinn::EndpointConfig::default(), Some(quic::server_config()?), socket, runtime)?
    };
    let addr = endpoint.local_addr()?;
    let shutdown = Arc::new(AtomicBool::new(false));
    let endpoint_for_stop = endpoint.clone();

    // Doubles as spawn-height terrain, the content identity joins must match,
    // and the edit-spec validator. Built from the loaded world, not the flags,
    // once a file has replaced them.
    let mut registry = BlockRegistry::with_builtins();
    let generator: crate::world::terrain::Generator = match loaded.worldgen {
        WorldgenKind::Flat => {
            Arc::new(crate::world::generation::FlatTerrain::new(&mut registry, loaded.seed))
        }
        WorldgenKind::Diffusion => {
            crate::world::terrain::generator(&mut registry, loaded.seed, loaded.terrain)
        }
    };
    let mut state = State {
        edits: HashMap::new(),
        spec_pool: HashMap::new(),
        registry,
        players: HashMap::new(),
        grid: HashMap::new(),
        next_id: 1,
        day: loaded.day,
        day_set: Instant::now(),
        reactions: ReactionScheduler::new(),
        max_speed: finite_speed(config.max_speed),
        #[cfg(test)]
        panic_tick: false,
    };
    let kept = install_edits(&mut state, &loaded.edits);
    state.reactions.restore(&contacts_of(&loaded.pending));
    let content = crate::net::content_id(&state.registry);
    let shared = Arc::new(Mutex::new(state));
    debug_assert_ne!(WORLD_PLAYER, 1, "player ids start at 1; 0 is the world");

    let hooks = if config.hooks.is_empty() {
        None
    } else {
        Some(Mutex::new(hooks::Table::new(config.hooks)))
    };
    let op_secrets: Vec<(String, String)> =
        config.op_secrets.iter().map(|(name, secret)| (canonical_name(name), secret.clone())).collect();
    let ctx = Arc::new(Ctx {
        password: config.password,
        seed: loaded.seed,
        content,
        day_secs: clamp_day_secs(config.day_secs),
        teleport: config.teleport,
        noclip: config.noclip,
        worldgen: loaded.worldgen,
        terrain: loaded.terrain,
        seams: Seams::new(generator.atlases().to_vec()),
        generator,
        hooks,
        ops: canonical_ops(&config.ops, &op_secrets),
        op_secrets,
        mods_allow: config.mods_allow,
        mods_deny: config.mods_deny,
        store: loaded.store.map(Arc::new),
    });
    if let Some(store) = &ctx.store {
        store.set_kept(kept);
        println!(
            "world {}: seed {}, worldgen {}",
            store.path().display(),
            ctx.seed,
            ctx.worldgen.id()
        );
    }

    let pending = Arc::new(AtomicUsize::new(0));
    #[cfg(test)]
    let handshake_pending = pending.clone();
    let save_gate = Arc::new(Mutex::new(()));
    let accept_shutdown = shutdown.clone();
    let accept_rt = rt.clone();
    let tick_shutdown = shutdown.clone();
    let tick_shared = shared.clone();
    let tick_ctx = ctx.clone();
    let reactions = thread::spawn(move || reactions_loop(tick_shared, tick_ctx, tick_shutdown));
    let autosave = ctx.store.as_ref().map(|_| {
        let shared = shared.clone();
        let ctx = ctx.clone();
        let gate = save_gate.clone();
        let shutdown = shutdown.clone();
        let every = config.autosave_every.max(Duration::from_secs(1));
        thread::spawn(move || autosave_loop(shared, ctx, gate, shutdown, every))
    });
    let accept_shared = shared.clone();
    let accept_ctx = ctx.clone();
    let accept = thread::spawn(move || {
        accept_loop(endpoint, accept_rt, accept_shared, accept_ctx, accept_shutdown, pending)
    });

    Ok(ServerHandle {
        shutdown,
        stopped: AtomicBool::new(false),
        addr,
        rt: Mutex::new(Some(rt)),
        endpoint: Mutex::new(Some(endpoint_for_stop)),
        accept: Mutex::new(Some(accept)),
        reactions: Mutex::new(Some(reactions)),
        autosave: Mutex::new(autosave),
        save_gate,
        state: shared,
        ctx,
        #[cfg(test)]
        handshake_pending,
    })
}

fn finite_speed(v: f64) -> f64 {
    if v.is_finite() && v >= 0.0 { v } else { crate::player::MAX_SPEED }
}

/// Name-only operators, minus any name that has a secret.
fn canonical_ops(names: &[String], secrets: &[(String, String)]) -> Vec<String> {
    let mut ops = Vec::new();
    for name in names {
        let canon = canonical_name(name);
        let listed = ops.iter().any(|op: &String| op == &canon) || secrets.iter().any(|(have, _)| have == &canon);
        if canon.is_empty() || listed {
            continue;
        }
        ops.push(canon);
    }
    ops
}

fn canonical_name(raw: &str) -> String {
    clean_name(raw).to_ascii_lowercase()
}

fn is_operator(ctx: &Ctx, h: &PlayerHandle) -> bool {
    h.op || ctx.ops.iter().any(|op| op.eq_ignore_ascii_case(&h.name))
}

/// Skip a spec this build cannot parse or the pool cannot hold, and return those cells
/// so the next save writes them back. Revisions start at 1; the file has none.
fn install_edits(state: &mut State, edits: &[(i32, i32, i32, String)]) -> Vec<(i32, i32, i32, String)> {
    let mut kept = Vec::new();
    let mut over = 0usize;
    for (x, y, z, spec) in edits {
        let Some(id) = state.registry.parse_spec(spec) else {
            eprintln!("skipping unknown block spec at {x},{y},{z}");
            kept.push((*x, *y, *z, spec.clone()));
            continue;
        };
        let canonical = state.registry.spec(id);
        let Some(shared) = state.intern(&canonical) else {
            over += 1;
            kept.push((*x, *y, *z, spec.clone()));
            continue;
        };
        state.edits.insert((*x, *y, *z), Cell { spec: shared, rev: 1, natural: false });
    }
    if over > 0 {
        eprintln!("warning: {over} edits name more than {MAX_SPEC_POOL} distinct blocks; they stay in the file but are not served");
    }
    kept
}

fn contacts_of(pending: &[crate::save::format::PendingContact]) -> Vec<(u32, Contact)> {
    pending
        .iter()
        .map(|contact| (contact.age, Contact { lo: (contact.x, contact.y, contact.z), axis: contact.axis }))
        .collect()
}

fn save_world(state: &Mutex<State>, ctx: &Ctx, gate: &Mutex<()>) {
    let Some(store) = &ctx.store else { return };
    let _gate = gate.lock_recover();
    let snap = {
        let state = state.lock_recover();
        let edits = state
            .edits
            .iter()
            .filter(|(_, cell)| !cell.natural)
            .map(|(&(x, y, z), cell)| (x, y, z, Arc::clone(&cell.spec)))
            .collect();
        let pending = state
            .reactions
            .snapshot()
            .into_iter()
            .map(|(age, contact)| crate::save::format::PendingContact {
                x: contact.lo.0,
                y: contact.lo.1,
                z: contact.lo.2,
                axis: contact.axis,
                age,
            })
            .collect();
        persist::Snapshot {
            seed: ctx.seed,
            worldgen: ctx.worldgen,
            terrain: ctx.terrain,
            day: state.day_now(ctx.day_secs),
            edits,
            pending,
        }
    };
    if let Err(e) = store.write(&snap) {
        eprintln!("could not save {}: {e}", store.path().display());
    }
}

fn autosave_loop(
    state: Arc<Mutex<State>>,
    ctx: Arc<Ctx>,
    gate: Arc<Mutex<()>>,
    shutdown: Arc<AtomicBool>,
    every: Duration,
) {
    let mut next = Instant::now() + every;
    while !shutdown.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(200));
        if shutdown.load(Ordering::SeqCst) {
            break;
        }
        if Instant::now() < next {
            continue;
        }
        save_world(&state, &ctx, &gate);
        next = Instant::now() + every;
    }
}

/// Union `ops.txt` and `mods.toml` beside [`Config::world`] into the flag lists.
/// An `ops.txt` line with a secret goes to [`Config::op_secrets`]. A missing file
/// adds nothing. A `mods.toml` that is not `allow` / `deny` string arrays is an
/// error, so a dedicated server refuses to start open.
pub fn load_world_policy(config: &mut Config) -> Result<(), String> {
    let Some(world) = config.world.clone() else {
        return Ok(());
    };
    let side = persist::read_side_files(&world)?;
    for name in side.ops {
        if !config.ops.iter().any(|op| op.eq_ignore_ascii_case(&name)) {
            config.ops.push(name);
        }
    }
    for (name, secret) in side.op_secrets {
        if !config.op_secrets.iter().any(|(have, _)| have.eq_ignore_ascii_case(&name)) {
            config.op_secrets.push((name, secret));
        }
    }
    push_unique(&mut config.mods_allow, side.allow);
    push_unique(&mut config.mods_deny, side.deny);
    Ok(())
}

fn push_unique(into: &mut Vec<String>, extra: Vec<String>) {
    for id in extra {
        if !into.iter().any(|have| have == &id) {
            into.push(id);
        }
    }
}

/// The dedicated server binary's entry point. SIGINT and SIGTERM save and close.
pub fn run(port: u16, config: Config) -> io::Result<()> {
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    // Listen before the server exists, so a signal during startup is not the default kill.
    #[cfg(unix)]
    let (mut sigint, mut sigterm) = rt.block_on(async {
        use tokio::signal::unix::{SignalKind, signal};
        io::Result::Ok((signal(SignalKind::interrupt())?, signal(SignalKind::terminate())?))
    })?;
    let handle = spawn(port, config)?;
    println!("watt-cubed server listening on {}", handle.addr());
    #[cfg(unix)]
    rt.block_on(async {
        tokio::select! {
            _ = sigint.recv() => {}
            _ = sigterm.recv() => {}
        }
    });
    // Elsewhere Ctrl-C (and Ctrl-Break / console close on Windows) is the stop request.
    #[cfg(not(unix))]
    rt.block_on(tokio::signal::ctrl_c())?;
    handle.stop();
    Ok(())
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
    let mut clients: Vec<JoinHandle<()>> = Vec::new();
    while !shutdown.load(Ordering::Relaxed) {
        clients.retain(|handle| !handle.is_finished());
        let step = catch_unwind(AssertUnwindSafe(|| {
            accept_once(&endpoint, &rt, &shared, &ctx, &pending, &mut clients)
        }));
        match step {
            Ok(true) => break,
            Ok(false) => {}
            Err(payload) => eprintln!("accept loop panicked: {}", panic_text(&payload)),
        }
    }
    for handle in clients {
        let _ = handle.join();
    }
}

/// One accept. `true` when the endpoint has closed. A panic here drops a local
/// [`HandshakeSlot`] if one was taken, so the pre-auth count cannot stick.
fn accept_once(
    endpoint: &Endpoint,
    rt: &Arc<Runtime>,
    shared: &Arc<Mutex<State>>,
    ctx: &Arc<Ctx>,
    pending: &Arc<AtomicUsize>,
    clients: &mut Vec<JoinHandle<()>>,
) -> bool {
    // Bounded wait so `stop()` is noticed between connections. Closing the
    // endpoint makes `accept` return `None` and the loop joins every client.
    let incoming = match rt
        .block_on(async { tokio::time::timeout(Duration::from_millis(200), endpoint.accept()).await })
    {
        Ok(Some(incoming)) => incoming,
        Ok(None) => return true,
        Err(_elapsed) => return false,
    };

    // A stateless retry first: a source must prove it receives our packets before it
    // takes a slot, so spoofed Initials cannot hold the handshake slots.
    if !incoming.remote_address_validated() {
        let _ = incoming.retry();
        return false;
    }
    // A QUIC CONNECTION_REFUSED past the cap, before any handshake, so a
    // flood of silent connects can't squat handler threads. The accept
    // loop is the only adder, so load-then-add can't overshoot the cap.
    if pending.load(Ordering::Relaxed) >= HANDSHAKE_CAP {
        incoming.refuse();
        return false;
    }
    pending.fetch_add(1, Ordering::Relaxed);
    let slot = HandshakeSlot(Arc::clone(pending));
    let shared = Arc::clone(shared);
    let ctx = Arc::clone(ctx);
    let handler_rt = Arc::clone(rt);
    clients.push(thread::spawn(move || {
        // A dropped connection is routine; the error is the disconnect cause.
        let _ = handle_client(incoming, handler_rt, shared, ctx, slot);
    }));
    false
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
    let Some((id, spawn, existing, snapshot, out, rx, kick)) =
        admit_player(&shared, &ctx, &rt, &mut send, &conn, &name)
    else {
        return Ok(());
    };
    let writer = spawn_writer(rt.clone(), send, rx, kick.clone());
    println!("[+] {name} joined as #{id} from {addr} ({} online)", online(&shared));

    let kicked = shared.lock_recover().players.get(&id).map(|h| Arc::clone(&h.kicked));
    let Some(kicked) = kicked else {
        depart(&shared, &ctx, conn, out, writer, id, &name);
        return Ok(());
    };
    // These sends wait, bounded, on this client's handler thread: a built-up
    // world or big roster can exceed the outbound queue, and dropping bootstrap
    // frames would ghost the join. A kick or a missed deadline ends the join.
    if !send_join(&out, &kicked, &shared, &ctx, id, spawn, &snapshot, &existing) {
        kicked.store(true, Ordering::Relaxed);
        depart(&shared, &ctx, conn, out, writer, id, &name);
        return Ok(());
    }
    // Bootstrap queued: go live. Frames broadcast during the bootstrap window
    // were buffered; drain them in order (they postdate the snapshot) and only
    // then let broadcasters push directly.
    {
        let mut state = shared.lock_recover();
        let mut slow = false;
        if let Some(h) = state.players.get_mut(&id) {
            h.backlog_bytes = 0;
            for queued in std::mem::take(&mut h.backlog) {
                if h.out.try_send(queued.frame).is_err() {
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

    let name = match hello_name(&frame, ctx) {
        Ok(name) => clean_name(&name),
        Err(HelloFail::Reject(reason)) => {
            reject(rt, &mut send, &conn, &reason);
            return Ok(None);
        }
        Err(HelloFail::Mods(ids)) => {
            deny_mods(rt, &mut send, &conn, ids);
            return Ok(None);
        }
    };
    // Pre-auth window is over; the roster's own MAX_PLAYERS bound takes over.
    drop(slot);
    Ok(Some((conn, send, recv, frame, name, addr)))
}

enum HelloFail {
    Reject(String),
    Mods(Vec<Arc<str>>),
}

/// Protocol number first, then the content parts. A tag or version mismatch is
/// named without decoding the rest of the payload (a v12 `Hello` is a different shape).
/// Password is checked before the mod list, so a scanner without the password
/// learns nothing about the whitelist.
fn hello_name(frame: &[u8], ctx: &Ctx) -> Result<Arc<str>, HelloFail> {
    let protocol = match protocol::peek_hello(frame) {
        protocol::HelloPeek::NotHello => return Err(HelloFail::Reject("expected hello".into())),
        protocol::HelloPeek::Truncated => return Err(HelloFail::Reject("malformed hello".into())),
        protocol::HelloPeek::Protocol(v) => v,
    };
    if protocol != PROTOCOL_VERSION {
        return Err(HelloFail::Reject(format!(
            "protocol version mismatch: server v{PROTOCOL_VERSION}, client v{protocol}"
        )));
    }
    match ClientMessage::decode(frame) {
        Some(ClientMessage::Hello { worldgen, gravity, law, palette, name, password, mods, .. }) => {
            let client = crate::net::ContentId { worldgen, gravity, law, palette };
            if let Some(why) = crate::net::content_mismatch(ctx.content, client) {
                return Err(HelloFail::Reject(why));
            }
            if *password != *ctx.password {
                return Err(HelloFail::Reject("wrong password".into()));
            }
            let refused = refused_mods(ctx, &mods);
            if !refused.is_empty() {
                return Err(HelloFail::Mods(refused));
            }
            Ok(name)
        }
        _ => Err(HelloFail::Reject("malformed hello".into())),
    }
}

/// Allow list (when set) admits only those ids. Deny always refuses. An id in
/// both is refused. Ids match the registered package id, case-sensitive.
fn refused_mods(ctx: &Ctx, mods: &[ModOffer]) -> Vec<Arc<str>> {
    let mut refused = Vec::new();
    for offer in mods {
        let id = offer.id.as_ref();
        let denied = ctx.mods_deny.iter().any(|d| d == id);
        let blocked = !ctx.mods_allow.is_empty() && !ctx.mods_allow.iter().any(|a| a == id);
        if (denied || blocked) && !refused.iter().any(|have: &Arc<str>| have.as_ref() == id) {
            refused.push(Arc::clone(&offer.id));
        }
    }
    refused
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
    let existing: Vec<(u32, Arc<str>)>;
    let snapshot: Vec<(i32, i32, i32, u32, Arc<str>)>;
    {
        let mut state = shared.lock_recover();
        if reserved_name(name) {
            drop(state);
            reject(rt, send, conn, "the name 'server' is reserved");
            return None;
        }
        if state.players.values().any(|h| h.name.eq_ignore_ascii_case(name)) {
            drop(state);
            reject(rt, send, conn, "that name is already in use");
            return None;
        }
        if state.players.len() >= MAX_PLAYERS {
            drop(state);
            reject(rt, send, conn, "server full");
            return None;
        }
        id = state.next_id;
        state.next_id += 1;
        spawn = spawn_point(ctx.generator.as_ref(), id);
        let (frame, up) = standing_pose(Field::new(ctx.generator.mass()).sample(spawn).accel);

        // Roster only — poses flow through the visibility machinery once the
        // joiner reports their first move, so a far peer isn't a frozen ghost.
        existing = state.players.iter().map(|(&pid, h)| (pid, h.name.clone())).collect();
        let announced: HashSet<u32> = existing.iter().map(|(pid, _)| *pid).collect();
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
                frame,
                velocity: Vec3::ZERO,
                up,
                stance: Stance::Standing,
                last_move: Instant::now(),
                budget: MOVE_FLOOR,
                op: false,
                visible: HashSet::new(),
                out: out.clone(),
                kick: kick.clone(),
                ready: false,
                backlog: VecDeque::new(),
                backlog_bytes: 0,
                kicked: Arc::new(AtomicBool::new(false)),
                occupied: Vec::new(),
                cruising: false,
                cruise_speed: 0.0,
                novel: 0,
                announced,
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
    Some((id, spawn, existing, snapshot, out, rx, kick))
}

fn spawn_writer(
    writer_rt: Arc<Runtime>,
    mut send: SendStream,
    rx: std::sync::mpsc::Receiver<Arc<[u8]>>,
    kick: Arc<Notify>,
) -> thread::JoinHandle<()> {
    // A write error ends the writer and wakes the reader, so the client is
    // dropped instead of left half-open. A clean channel close (depart) does
    // not kick: the reader has already exited. QUIC has no user flush.
    thread::spawn(move || {
        drain_writer(rx, &kick, |frame| {
            writer_rt.block_on(protocol::write_frame_async(&mut send, frame))
        });
    })
}

/// Pull frames until the channel closes. The first write error notifies `kick` and returns.
fn drain_writer(
    rx: std::sync::mpsc::Receiver<Arc<[u8]>>,
    kick: &Notify,
    mut write: impl FnMut(&[u8]) -> io::Result<()>,
) {
    while let Ok(frame) = rx.recv() {
        if write(&frame).is_err() {
            kick.notify_one();
            return;
        }
        while let Ok(frame) = rx.try_recv() {
            if write(&frame).is_err() {
                kick.notify_one();
                return;
            }
        }
    }
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
    // Each kind has its own budget, so a swing flood cannot starve edits.
    // Mod channels and tool uses keep a second, tighter window of their own.
    let mut budgets = KindBudget::new();
    let mut channels = ChannelBudget::new();
    let mut tool_rate = RateWindow::new(TOOL_RATE_LIMIT);
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

        let Some(msg) = ClientMessage::decode(frame) else {
            continue;
        };
        // Cruise is a declared state, not a flood: it is applied even when every
        // other window is spent, and it does not consume a token.
        let now = Instant::now();
        match charge(&mut budgets, &msg, now) {
            Charge::Drop => continue,
            Charge::Answer => {
                match &msg {
                    ClientMessage::Edit { req, x, y, z, .. } => reject_edit(shared, id, *req, *x, *y, *z),
                    ClientMessage::Teleport { .. } => refuse_move(shared, id),
                    _ => {}
                }
                continue;
            }
            Charge::Pass => {}
        }
        match msg {
            ClientMessage::Move { pos, yaw, pitch, frame, velocity, up, stance } => {
                on_move(shared, ctx, id, pos, yaw, pitch, frame, velocity, up, stance)
            }
            ClientMessage::Teleport { pos } => on_teleport(shared, ctx, id, pos),
            ClientMessage::Edit { req, x, y, z, expect, spec } => {
                on_edit(shared, ctx.hooks.as_ref(), &ctx.generator, id, req, x, y, z, expect, &spec)
            }
            ClientMessage::Chat { channel, text } => match op_secret(&text) {
                Some(secret) => on_op_login(shared, ctx, id, &secret),
                None => on_chat(shared, ctx.hooks.as_ref(), id, channel, &text),
            },
            ClientMessage::SetTime { day } => on_set_time(shared, ctx, id, day),
            ClientMessage::ModData { channel, seq, bytes } => {
                if !channels.allow(&channel, now) {
                    continue; // Over this channel's budget this second — drop silently.
                }
                on_mod_data(shared, id, channel, seq, bytes);
            }
            // Only players who can see the swinger, the same audience as voice.
            ClientMessage::Swing => relay_swing(shared, id),
            ClientMessage::Ping { nonce } => {
                let state = shared.lock_recover();
                if let Some(h) = state.players.get(&id) {
                    // Best-effort: a full queue drops the probe, and the
                    // client simply re-sends on its interval.
                    let _ = h.out.try_send(ServerMessage::Pong { nonce }.encode().into());
                }
            }
            ClientMessage::Hello { .. } => {} // Already authenticated; ignore repeats.
            ClientMessage::Cruise { speed } => on_cruise(shared, id, speed),
            ClientMessage::ToolUse { req, x, y, z, expect, tool_spec } => {
                if !tool_rate.allow(now) {
                    // Over the tool budget this second: refuse, so the client's swing resolves.
                    refuse_tool(shared, id, req, x, y, z, &tool_spec);
                    continue;
                }
                on_tool_use(shared, &ctx.generator, id, req, x, y, z, expect, &tool_spec)
            }
        }
    }
}

/// An edit that will not be applied still gets an ack, so the client's request
/// does not stay pending and poison the next expectation on that cell.
fn reject_edit(shared: &Arc<Mutex<State>>, id: u32, req: u32, x: i32, y: i32, z: i32) {
    let state = shared.lock_recover();
    let Some(h) = state.players.get(&id) else { return };
    let rev = state.edits.get(&(x, y, z)).map_or(0, |c| c.rev);
    let _ = h.out.try_send(ServerMessage::EditAck { req, accepted: false, rev }.encode().into());
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

/// Validation is a plausibility ENVELOPE, not full physics. Each player spends a
/// distance budget ([`move_allowance`]) that refills at the envelope speed and holds
/// at most one burst, so splitting a move into many messages gains nothing.
/// An implausible move is not committed — the server keeps its last accepted
/// position (which edit reach reads), and the client is snapped back with an
/// authoritative [`ServerMessage::Position`]. `/tp` discontinuities arrive as
/// [`ClientMessage::Teleport`] instead.
///
/// Runs in two phases to keep the global lock hold minimal. Locked: commit the
/// move, keep the grid current, diff visibility, and snapshot the recipients'
/// senders (cheap `SyncSender` clones — one `Arc` bump each). Unlocked: the
/// `try_send`s. `try_send` never blocks, failures land their owner on the kick
/// list, [`kick_slow`] tolerates ids that disconnected in the unlocked window,
/// and ids are never reused, so a late kick can't hit the wrong player.
/// The orientation a [`ClientMessage::Move`] reports. A teleport passes `None`
/// and keeps whatever the handle already stored.
struct ReportedPose {
    yaw: f32,
    pitch: f32,
    frame: DQuat,
    velocity: Vec3,
    up: Face,
    stance: Stance,
}

fn quat_finite(q: DQuat) -> bool {
    q.x.is_finite() && q.y.is_finite() && q.z.is_finite() && q.w.is_finite()
}

fn on_move(
    shared: &Arc<Mutex<State>>,
    ctx: &Ctx,
    id: u32,
    pos: DVec3,
    yaw: f32,
    pitch: f32,
    frame: DQuat,
    velocity: Vec3,
    up: Face,
    stance: Stance,
) {
    // A NaN position poisons distance checks/grid keys; a NaN angle, frame, or
    // velocity propagates into peer interpolation and render matrices.
    if !pos.x.is_finite()
        || !pos.y.is_finite()
        || !pos.z.is_finite()
        || !yaw.is_finite()
        || !pitch.is_finite()
        || !quat_finite(frame)
        || !velocity.x.is_finite()
        || !velocity.y.is_finite()
        || !velocity.z.is_finite()
    {
        return;
    }
    let mut sends = Vec::new();
    {
        let mut state = shared.lock_recover();
        let max_speed = state.max_speed;
        // Envelope: the speed the client reports (and the one we last accepted),
        // grown by gravity over the gap, capped by the server's speed limit.
        // Cruise raises that cap only after a `Cruise` message, and only up to
        // the game's cruise ceiling unless the server cap is tighter.
        // Outside the border is refused either way. Solid ground is tested along the
        // path unless this player's noclip policy allows the pass.
        let (free, too_far, left, from, cruising, occupied, occupied_n) = {
            let Some(h) = state.players.get(&id) else { return };
            let free = noclip_allowed(ctx, h);
            let elapsed = h.last_move.elapsed().as_secs_f64().min(MOVE_WINDOW_CAP_SECS);
            let available = move_allowance(h, velocity, elapsed, max_speed);
            let distance = h.pos.distance(pos);
            let too_far = outside_world(pos) || distance > available;
            let mut occupied = [(0i32, 0, 0); BODY_CELL_CAP];
            let occupied_n = h.occupied.len().min(BODY_CELL_CAP);
            occupied[..occupied_n].copy_from_slice(&h.occupied[..occupied_n]);
            (free, too_far, available - distance, h.pos, h.cruising, occupied, occupied_n)
        };
        let blocked = !free
            && !too_far
            && move_blocked(&state, ctx, from, pos, cruising, stance, up, &occupied[..occupied_n]);
        if too_far || blocked {
            let Some(h) = state.players.get(&id) else { return };
            h.correct_position(id, &mut sends);
        } else {
            commit_pose(
                &mut state,
                id,
                pos,
                Some(ReportedPose { yaw, pitch, frame, velocity, up, stance }),
                &mut sends,
            );
            if let Some(h) = state.players.get_mut(&id) {
                h.budget = left;
                if !free {
                    remember_occupied(h, pos, stance, up);
                }
            }
        }
    }
    dispatch(shared, sends);
}

fn noclip_allowed(ctx: &Ctx, h: &PlayerHandle) -> bool {
    match ctx.noclip {
        NoclipPolicy::All => true,
        NoclipPolicy::Ops => is_operator(ctx, h),
        NoclipPolicy::Off => false,
    }
}

fn body_stance(stance: Stance) -> player::Stance {
    match stance {
        Stance::Standing => player::Stance::Standing,
        Stance::Sneaking => player::Stance::Sneaking,
    }
}

/// Physical cells the body strictly overlaps. `None` when the box is larger than
/// the stack, which the caller treats as blocked.
fn fill_body_cells(pos: DVec3, stance: Stance, up: Face, out: &mut [(i32, i32, i32); BODY_CELL_CAP]) -> Option<usize> {
    let body = player::collision_box(pos, body_stance(stance), up);
    let mut n = 0;
    for cell in body.voxel_cells() {
        if n >= BODY_CELL_CAP {
            return None;
        }
        out[n] = cell;
        n += 1;
    }
    Some(n)
}

fn remember_occupied(h: &mut PlayerHandle, pos: DVec3, stance: Stance, up: Face) {
    let mut cells = [(0i32, 0, 0); BODY_CELL_CAP];
    let Some(n) = fill_body_cells(pos, stance, up, &mut cells) else {
        return;
    };
    if h.occupied.len() == n && h.occupied.iter().zip(cells[..n].iter()).all(|(have, cell)| have == cell) {
        return;
    }
    h.occupied.clear();
    h.occupied.extend_from_slice(&cells[..n]);
}

/// The body's path from the last accepted pose meets solid ground. A path longer than
/// [`SWEEP_LIMIT`] fails closed, except under cruise, whose destination alone is tested.
#[allow(clippy::too_many_arguments)] // the move's pose fields, passed separately like on_move's
fn move_blocked(
    state: &State,
    ctx: &Ctx,
    from: DVec3,
    to: DVec3,
    cruising: bool,
    stance: Stance,
    up: Face,
    occupied: &[(i32, i32, i32)],
) -> bool {
    let from = if from.distance(to) <= SWEEP_LIMIT {
        from
    } else if cruising {
        to
    } else {
        return true;
    };
    body_blocked(state, &ctx.generator, &ctx.seams, from, to, stance, up, occupied)
}

/// True when the body, swept from `from` to `to` in steps of at most [`SWEEP_STEP`], newly
/// enters a cell that is solid in the edit overlay or the generator. Each step queries only
/// the cells the step before did not hold; the first skips `occupied`, the cells of the last
/// accepted pose. A straight path never re-enters a cell it left, so one step back is enough.
/// Cells are read the way the client's collision reads them: in storage, with a cell just
/// past a chart's box glued to the neighbour chart's cell.
#[allow(clippy::too_many_arguments)] // the move's pose fields, passed separately like on_move's
fn body_blocked(
    state: &State,
    generator: &crate::world::terrain::Generator,
    seams: &Seams,
    from: DVec3,
    to: DVec3,
    stance: Stance,
    up: Face,
    occupied: &[(i32, i32, i32)],
) -> bool {
    let steps = (from.distance(to) / SWEEP_STEP).ceil().max(1.0) as u32;
    let mut held = [(0i32, 0, 0); BODY_CELL_CAP];
    let mut held_n = occupied.len().min(BODY_CELL_CAP);
    held[..held_n].copy_from_slice(&occupied[..held_n]);
    let mut cells = [(0i32, 0, 0); BODY_CELL_CAP];
    for step in 1..=steps {
        let at = if step == steps { to } else { from.lerp(to, f64::from(step) / f64::from(steps)) };
        let Some(n) = fill_body_cells(at, stance, up, &mut cells) else {
            return true;
        };
        for &(x, y, z) in &cells[..n] {
            if held[..held_n].contains(&(x, y, z)) {
                continue;
            }
            let query = seams.glue_cell(BlockCoord::new(x, y, z)).map_or((x, y, z), |g| (g.x, g.y, g.z));
            if state.registry.is_solid(server_block(state, generator, query)) {
                return true;
            }
        }
        std::mem::swap(&mut held, &mut cells);
        held_n = n;
    }
    false
}

/// Snap to the last accepted pose without applying the request. Used when a
/// teleport is over its budget: the client still gets an answer.
fn refuse_move(shared: &Arc<Mutex<State>>, id: u32) {
    let mut sends = Vec::new();
    {
        let state = shared.lock_recover();
        if let Some(h) = state.players.get(&id) {
            h.correct_position(id, &mut sends);
        }
    }
    dispatch(shared, sends);
}

fn speed_of(v: Vec3) -> f64 {
    let s = (v.x as f64).hypot(v.y as f64).hypot(v.z as f64);
    if s.is_finite() { s } else { 0.0 }
}

/// Cruise ceiling. At or above [`crate::player::MAX_SPEED`] the declared cruise
/// may reach [`crate::player::CRUISE_MAX`]. A tighter server cap bounds cruise too.
fn cruise_limit(max_speed: f64) -> f64 {
    if max_speed < crate::player::MAX_SPEED {
        max_speed
    } else {
        crate::player::CRUISE_MAX
    }
}

fn move_cap(h: &PlayerHandle, max_speed: f64) -> f64 {
    if h.cruising {
        h.cruise_speed.min(cruise_limit(max_speed))
    } else {
        max_speed
    }
}

/// Distance this move may cover: the banked budget, at most one burst, plus what `elapsed`
/// refills at the envelope speed. That speed is the reported one (or the last accepted one)
/// grown by gravity over the gap, capped by [`move_cap`]. `reported` is a cap input, not a
/// grant: a cruise declaration does nothing until the velocity the client reports (bounded by
/// that declaration) justifies the hop. The burst is [`MOVE_FLOOR`], or [`MOVE_SLACK_SECS`] at
/// that speed when larger, so over any window the total stays within one burst plus the
/// speed times the window, however the client splits it.
fn move_allowance(h: &PlayerHandle, reported: Vec3, elapsed: f64, max_speed: f64) -> f64 {
    let cap = move_cap(h, max_speed);
    let speed = cap.min(speed_of(reported).max(speed_of(h.velocity)) + GRAVITY_BOUND * elapsed);
    h.budget.min(MOVE_FLOOR.max(speed * MOVE_SLACK_SECS)) + speed * elapsed
}

fn clamp_velocity(v: Vec3, cap: f64) -> Vec3 {
    let s = speed_of(v);
    if s > cap && cap > 0.0 {
        let k = (cap / s) as f32;
        Vec3::new(v.x * k, v.y * k, v.z * k)
    } else {
        v
    }
}

/// `speed` 0 ends cruise. Non-finite or negative is ignored. The stored cap
/// never exceeds [`cruise_limit`].
fn on_cruise(shared: &Arc<Mutex<State>>, id: u32, speed: f64) {
    if !speed.is_finite() || speed < 0.0 {
        return;
    }
    let mut state = shared.lock_recover();
    let limit = cruise_limit(state.max_speed);
    let Some(h) = state.players.get_mut(&id) else { return };
    if speed == 0.0 {
        h.cruising = false;
        h.cruise_speed = 0.0;
    } else {
        h.cruising = true;
        h.cruise_speed = speed.min(limit);
    }
}

/// An explicit `/tp` discontinuity: exempt from the movement envelope, still
/// border-checked, and refused (Position, then a reason) when this player
/// may not teleport.
fn on_teleport(shared: &Arc<Mutex<State>>, ctx: &Ctx, id: u32, pos: DVec3) {
    if !pos.x.is_finite() || !pos.y.is_finite() || !pos.z.is_finite() {
        return;
    }
    let mut sends = Vec::new();
    {
        let mut state = shared.lock_recover();
        let Some(h) = state.players.get(&id) else { return };
        let allowed = match ctx.teleport {
            TeleportPolicy::All => true,
            TeleportPolicy::Ops => is_operator(ctx, h),
            TeleportPolicy::Off => false,
        };
        if outside_world(pos) || !allowed {
            h.correct_position(id, &mut sends);
            if !outside_world(pos) {
                let reason = if ctx.teleport == TeleportPolicy::Off {
                    "teleport is not permitted"
                } else {
                    "only an operator can teleport"
                };
                tell(h, id, reason, &mut sends);
            }
        } else {
            commit_pose(&mut state, id, pos, None, &mut sends);
            // Echo so a client with an in-flight `/tp` can tell accept from a
            // stale movement snap-back: the last Position is the committed pose.
            // The destination's cells are the held ones the next swept move starts from.
            if let Some(h) = state.players.get_mut(&id) {
                h.correct_position(id, &mut sends);
                let (stance, up) = (h.stance, h.up);
                remember_occupied(h, pos, stance, up);
            }
        }
    }
    dispatch(shared, sends);
}

fn tell(h: &PlayerHandle, id: u32, text: &str, sends: &mut Vec<PendingSend>) {
    if !h.ready {
        return;
    }
    let frame = ServerMessage::Chat {
        from_id: WORLD_PLAYER,
        from_name: Arc::from("server"),
        channel: chat::GLOBAL,
        text: text.into(),
    }
    .encode()
    .into();
    sends.push((id, h.out.clone(), frame));
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
    reported: Option<ReportedPose>,
    sends: &mut Vec<PendingSend>,
) {
    let max_speed = state.max_speed;
    let Some(h) = state.players.get_mut(&id) else { return };
    let old = h.pos;
    h.pos = pos;
    if let Some(r) = reported {
        let cap = move_cap(h, max_speed);
        h.yaw = r.yaw;
        h.pitch = r.pitch;
        h.frame = r.frame;
        h.velocity = clamp_velocity(r.velocity, cap);
        h.up = r.up;
        h.stance = r.stance;
    }
    h.last_move = Instant::now();
    let (yaw, pitch, frame, velocity, up, stance) = (h.yaw, h.pitch, h.frame, h.velocity, h.up, h.stance);
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
        ServerMessage::PeerMove { id, pos, yaw, pitch, frame, velocity, up, stance }.encode().into();
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
                frame: other.frame,
                velocity: other.velocity,
                up: other.up,
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

/// Novel specs intern only while `block_count()` is below `limit`. A known spec resolves
/// even at the line. `None` is malformed, or novel at/above the line.
#[cfg(test)]
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

/// A known spec resolves with no quota spend. A novel one interns only while this
/// client is under [`NOVEL_SPEC_QUOTA`] and the table keeps [`CLIENT_INTERN_RESERVE`] free.
fn take_novel_spec(state: &mut State, id: u32, spec: &str) -> Option<BlockId> {
    if let Some(found) = state.registry.lookup_spec(spec) {
        return Some(found);
    }
    let under_quota = state.players.get(&id).is_some_and(|h| h.novel < NOVEL_SPEC_QUOTA);
    let limit = crate::block::registry::MAX_BLOCK_TYPES - CLIENT_INTERN_RESERVE;
    if !under_quota || state.registry.block_count() >= limit {
        return None;
    }
    let before = state.registry.block_count();
    let block = state.registry.parse_spec(spec)?;
    if state.registry.block_count() > before && let Some(h) = state.players.get_mut(&id) {
        h.novel = h.novel.saturating_add(1);
    }
    Some(block)
}


/// Gates, in order: reach (against the sender's last ACCEPTED position, per
/// [`on_move`]'s envelope), a canonical spec (no intern), the expected cell
/// revision, installed [`ServerMod::validate_edit`] hooks, the revision again,
/// then — only if it is still the winner — interning a novel spec under the
/// per-client quota and the world's reserve. A stale novel spec never enters
/// the registry. When two players race one cell, the loser rolls back. An edit that
/// puts back the generated block is a [`Cell::natural`] entry: it keeps the revision
/// but stays out of the world file, so no-op edits cannot grow it.
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
    generator: &crate::world::terrain::Generator,
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
    // A round world's storage cell is judged where its chart embeds it.
    let target = crate::space::atlas::embed_cell(generator.atlases(), (x, y, z))
        .unwrap_or(DVec3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5));
    if spec.len() > MAX_SPEC || h.pos.distance(target) > EDIT_REACH {
        return reject(&state, ack_to.as_ref());
    }
    // Canonical form without touching the registry. Junk and `c:00` fail here,
    // before a revision miss could still have interned them.
    let Some(canonical) = state.registry.canonical_spec(spec) else {
        return reject(&state, ack_to.as_ref());
    };
    let current = state.edits.get(&(x, y, z)).map_or(0, |c| c.rev);
    if expect != current {
        return reject(&state, ack_to.as_ref());
    }
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
    let Some(block) = take_novel_spec(&mut state, id, &canonical) else {
        return reject(&state, ack_to.as_ref());
    };
    if block == crate::block::AIR && canonical != "air" {
        return reject(&state, ack_to.as_ref());
    }
    let rev = current + 1;
    let Some(spec) = state.intern(&canonical) else {
        return reject(&state, ack_to.as_ref()); // pool at cap: refuse new content
    };
    let natural = block == generator.voxel_at(x, y, z);
    if let Some(old) = state.edits.insert((x, y, z), Cell { spec: spec.clone(), rev, natural }) {
        state.release(old.spec);
    }
    // Placed or removed: the cell's contacts wake.
    state.reactions.wake_cell((x, y, z));
    if let Some(out) = ack_to {
        let _ = out
            .try_send(ServerMessage::EditAck { req, accepted: true, rev }.encode().into());
    }
    // The broadcast carries the SAME pooled Arc the ledger stores.
    let msg = ServerMessage::Edit { x, y, z, rev, spec };
    broadcast(&mut state, &msg, |pid, _| pid != id);
}

/// Answer a tool use with "nothing happened": the cell's current content and the tool unchanged.
fn refuse_tool(shared: &Arc<Mutex<State>>, id: u32, req: u32, x: i32, y: i32, z: i32, tool_spec: &str) {
    let state = shared.lock_recover();
    let Some(h) = state.players.get(&id) else { return };
    let cell = state.edits.get(&(x, y, z));
    let (rev, cell_spec) = cell.map_or((0, Arc::from("")), |c| (c.rev, c.spec.clone()));
    let msg = ServerMessage::ToolResult { req, reacted: false, rev, cell_spec, tool_spec: tool_spec.into() };
    let _ = h.out.try_send(msg.encode().into());
}

/// A player uses a held configuration as a tool on a cell. Gates: ready, reach, the tool spec
/// resolves (known, or novel below the reserve line), the cell revision is the one the client
/// expected. Then ONE operation of the law runs between the cell (A) and the tool (B); on a
/// change the cell is written, its contacts wake, the sender gets both results and everyone else
/// the cell edit. There is no holdings ledger, so the server trusts the client about what it holds
/// (as for placement).
#[allow(clippy::too_many_arguments)]
fn on_tool_use(
    shared: &Arc<Mutex<State>>,
    generator: &crate::world::terrain::Generator,
    id: u32,
    req: u32,
    x: i32,
    y: i32,
    z: i32,
    expect: u32,
    tool_spec: &str,
) {
    let mut state = shared.lock_recover();
    let Some(h) = state.players.get(&id) else { return };
    if !h.ready {
        return;
    }
    let out = h.out.clone();
    let target = crate::space::atlas::embed_cell(generator.atlases(), (x, y, z))
        .unwrap_or(DVec3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5));
    let pos = (x, y, z);
    let current = state.edits.get(&pos).map_or(0, |c| c.rev);
    let reply = |state: &State, reacted: bool, rev: u32, tool: Arc<str>| {
        let cell_spec: Arc<str> = state.edits.get(&pos).map_or_else(
            || crate::save::block_spec(&state.registry, generator.voxel_at(x, y, z)).into(),
            |c| c.spec.clone(),
        );
        let msg = ServerMessage::ToolResult { req, reacted, rev, cell_spec, tool_spec: tool };
        let _ = out.try_send(msg.encode().into());
    };
    let unchanged: Arc<str> = tool_spec.into();
    if tool_spec.len() > MAX_SPEC || h.pos.distance(target) > EDIT_REACH || expect != current {
        return reply(&state, false, current, unchanged);
    }
    // Revision already matched, so a novel tool spends quota only for a live request.
    let Some(tool) = take_novel_spec(&mut state, id, tool_spec) else {
        return reply(&state, false, current, unchanged);
    };
    let cell = server_block(&state, generator, pos);
    if tool == AIR || cell == AIR {
        return reply(&state, false, current, unchanged);
    }
    let limit = crate::block::registry::MAX_BLOCK_TYPES - CLIENT_INTERN_RESERVE;
    if state.registry.block_count() + 2 > limit {
        return reply(&state, false, current, unchanged); // products would eat the world's reserve
    }
    let Some((_, new_cell, new_tool)) = state.registry.react(cell, tool) else {
        return reply(&state, false, current, unchanged);
    };
    let canonical = crate::save::block_spec(&state.registry, new_cell);
    let Some(spec) = state.intern(&canonical) else {
        return reply(&state, false, current, unchanged);
    };
    let rev = current + 1;
    if let Some(old) = state.edits.insert(pos, Cell { spec: spec.clone(), rev, natural: false }) {
        state.release(old.spec);
    }
    state.reactions.wake_cell(pos);
    let tool_out: Arc<str> = state.registry.spec(new_tool).into();
    reply(&state, true, rev, tool_out);
    broadcast(&mut state, &ServerMessage::Edit { x, y, z, rev, spec }, |pid, _| pid != id);
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

/// The secret of a `/op <secret>` chat line. Such a line is never relayed, logged, or shown to hooks.
fn op_secret(text: &str) -> Option<Arc<str>> {
    let text = clean_chat(text);
    let rest = text.strip_prefix("/op")?;
    (rest.is_empty() || rest.starts_with(char::is_whitespace)).then(|| rest.trim().into())
}

/// An operator listed with a secret proves it. Either way only the sender hears the answer.
fn on_op_login(shared: &Arc<Mutex<State>>, ctx: &Ctx, id: u32, secret: &str) {
    let mut sends = Vec::new();
    {
        let mut state = shared.lock_recover();
        let Some(h) = state.players.get_mut(&id) else { return };
        let proved = !secret.is_empty()
            && ctx.op_secrets.iter().any(|(name, want)| name.eq_ignore_ascii_case(&h.name) && want == secret);
        h.op |= proved;
        let (reply, log) = if proved {
            ("you are now an operator", "is now an operator")
        } else {
            ("operator secret refused", "sent a wrong operator secret")
        };
        println!("[op] {} (#{id}) {log}", h.name);
        tell(h, id, reply, &mut sends);
    }
    dispatch(shared, sends);
}

/// Channel traffic is loss-tolerant: `try_send` and DROP on a full/closed queue,
/// never counted toward the slow-client kick ([`kick_slow`]/[`OUT_CAPACITY`]).
/// Relayed only to the sender's visible interest set. The sender id is stamped
/// here; the client's own message does not carry it. Size is capped by the codec.
fn on_mod_data(
    shared: &Arc<Mutex<State>>,
    id: u32,
    channel: protocol::Channel,
    seq: u32,
    bytes: protocol::ModBytes,
) {
    let frame: Arc<[u8]> =
        ServerMessage::PeerModData { channel, sender: id, seq, bytes }.encode().into();
    let state = shared.lock_recover();
    let Some(speaker) = state.players.get(&id) else { return };
    for &pid in &speaker.visible {
        if let Some(other) = state.players.get(&pid) && other.ready {
            let _ = other.out.try_send(frame.clone());
        }
    }
}

/// Anchors the shared clock so joiners inherit the CURRENT time. A non-finite
/// value is ignored rather than poisoning the shared time. Only an operator
/// may set it.
fn on_set_time(shared: &Arc<Mutex<State>>, ctx: &Ctx, id: u32, day: f32) {
    if !day.is_finite() {
        return;
    }
    let day = day.rem_euclid(1.0);
    let mut sends = Vec::new();
    {
        let mut state = shared.lock_recover();
        let Some(h) = state.players.get(&id) else { return };
        if !is_operator(ctx, h) {
            tell(h, id, "only an operator can set the time", &mut sends);
            drop(state);
            dispatch(shared, sends);
            return;
        }
        state.day = day;
        state.day_set = Instant::now();
        broadcast(&mut state, &ServerMessage::Time { day, day_secs: ctx.day_secs }, |_, _| true);
    }
}

/// Encodes `msg` just once for every recipient. Players whose queue is full
/// are force-closed (they've fallen too far behind).
fn broadcast(state: &mut State, msg: &ServerMessage, want: impl Fn(u32, &PlayerHandle) -> bool) {
    let joined = match msg {
        ServerMessage::PeerJoined { id, .. } => Some(*id),
        _ => None,
    };
    let frame: Arc<[u8]> = msg.encode().into();
    let mut slow = Vec::new();
    for (&pid, h) in state.players.iter_mut() {
        if !want(pid, h) {
            continue;
        }
        // Already queued for this peer (the join roster, or an earlier broadcast).
        if let Some(jid) = joined && !h.announced.insert(jid) {
            continue;
        }
        if !h.ready {
            // Bootstrapping: buffer so the frame lands AFTER the snapshot.
            // Cosmetic frames are dropped instead of kicking the joiner.
            if !enqueue_backlog(h, Arc::clone(&frame), Instant::now()) {
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
            // awaiting, so a kick is never missed. The flag is what bootstrap
            // sends watch, because that thread is not in `client_loop` yet.
            h.kicked.store(true, Ordering::Relaxed);
            h.kick.notify_one();
        }
    }
}

/// Queue `frame` for a player who is not ready yet. False when an essential frame
/// does not fit after cosmetic frames have been dropped, when an essential frame has
/// waited past [`BACKLOG_AGE`], or when the player is already kicked.
fn enqueue_backlog(h: &mut PlayerHandle, frame: Arc<[u8]>, now: Instant) -> bool {
    if h.kicked.load(Ordering::Relaxed) || !trim_aged(h, now) {
        return false;
    }
    if let Some(subject) = protocol::peer_move_id(&frame) {
        drop_peer_moves(h, subject);
    }
    let cosmetic = is_cosmetic(&frame);
    while h.backlog_bytes + frame.len() > BACKLOG_BYTES {
        if !drop_oldest_cosmetic(h) {
            return cosmetic;
        }
    }
    h.backlog_bytes += frame.len();
    h.backlog.push_back(Queued { at: now, frame });
    true
}

/// A swing or a pose: the joiner loses nothing without it, since a peer entering range
/// sends both poses afresh.
fn is_cosmetic(frame: &[u8]) -> bool {
    protocol::is_peer_swing(frame) || protocol::peer_move_id(frame).is_some()
}

/// Drop frames older than [`BACKLOG_AGE`]. The backlog is in time order, so only the front
/// is read. False when an aged frame is essential: dropping it would desync the joiner.
fn trim_aged(h: &mut PlayerHandle, now: Instant) -> bool {
    while let Some(front) = h.backlog.front()
        && now.saturating_duration_since(front.at) >= BACKLOG_AGE
    {
        if !is_cosmetic(&front.frame) {
            return false;
        }
        let len = front.frame.len();
        h.backlog_bytes = h.backlog_bytes.saturating_sub(len);
        h.backlog.pop_front();
    }
    true
}

fn drop_peer_moves(h: &mut PlayerHandle, subject: u32) {
    let mut freed = 0;
    h.backlog.retain(|queued| {
        let keep = protocol::peer_move_id(&queued.frame) != Some(subject);
        if !keep {
            freed += queued.frame.len();
        }
        keep
    });
    h.backlog_bytes = h.backlog_bytes.saturating_sub(freed);
}

fn drop_oldest_cosmetic(h: &mut PlayerHandle) -> bool {
    let Some(index) = h.backlog.iter().position(|queued| is_cosmetic(&queued.frame)) else {
        return false;
    };
    if let Some(queued) = h.backlog.remove(index) {
        h.backlog_bytes = h.backlog_bytes.saturating_sub(queued.frame.len());
    }
    true
}

/// Swing relay: the swinger's visible set, same as voice. A full queue drops
/// the frame; a swing is cosmetic and never a kick.
fn relay_swing(shared: &Arc<Mutex<State>>, id: u32) {
    let frame: Arc<[u8]> = ServerMessage::PeerSwing { id }.encode().into();
    let state = shared.lock_recover();
    let Some(swinger) = state.players.get(&id) else { return };
    for &pid in &swinger.visible {
        if let Some(other) = state.players.get(&pid) && other.ready {
            let _ = other.out.try_send(Arc::clone(&frame));
        }
    }
}

/// Welcome, the edit overlay, [`ServerMessage::SnapshotEnd`], the clock, and
/// the roster. False when a send misses its deadline or the player was kicked.
fn send_join(
    out: &SyncSender<Arc<[u8]>>,
    kicked: &AtomicBool,
    shared: &Arc<Mutex<State>>,
    ctx: &Ctx,
    id: u32,
    spawn: DVec3,
    snapshot: &[(i32, i32, i32, u32, Arc<str>)],
    existing: &[(u32, Arc<str>)],
) -> bool {
    if !send_blocking(
        out,
        kicked,
        &ServerMessage::Welcome {
            player_id: id,
            seed: ctx.seed,
            spawn,
            worldgen: ctx.worldgen,
            terrain: ctx.terrain,
            law: crate::net::protocol::law_stamp(),
        },
    ) {
        return false;
    }
    let mut ok = true;
    for_snapshot_batches(snapshot, |batch| {
        if ok && !send_blocking(out, kicked, &ServerMessage::Snapshot { edits: batch.to_vec() }) {
            ok = false;
        }
    });
    if !ok || !send_blocking(out, kicked, &ServerMessage::SnapshotEnd) {
        return false;
    }
    // Read the clock at send time. A snapshot of a built-up world can take long
    // enough that the phase captured at admit would already be stale.
    let day = shared.lock_recover().day_now(ctx.day_secs);
    if !send_blocking(out, kicked, &ServerMessage::Time { day, day_secs: ctx.day_secs }) {
        return false;
    }
    for (pid, pname) in existing {
        if !send_blocking(out, kicked, &ServerMessage::PeerJoined { id: *pid, name: Arc::clone(pname) }) {
            return false;
        }
    }
    true
}

/// Only safe on the receiving client's own handler thread (used for the join
/// bootstrap, which must not drop frames). Returns false when `kicked` is set,
/// the queue is gone, or [`SEND_DEADLINE`] passes.
fn send_blocking(out: &SyncSender<Arc<[u8]>>, kicked: &AtomicBool, msg: &ServerMessage) -> bool {
    send_until(out, kicked, msg.encode().into(), Instant::now() + SEND_DEADLINE)
}

fn send_until(out: &SyncSender<Arc<[u8]>>, kicked: &AtomicBool, frame: Arc<[u8]>, deadline: Instant) -> bool {
    loop {
        if kicked.load(Ordering::Relaxed) {
            return false;
        }
        match out.try_send(Arc::clone(&frame)) {
            Ok(()) => return true,
            Err(TrySendError::Disconnected(_)) => return false,
            Err(TrySendError::Full(_)) => {
                if Instant::now() >= deadline {
                    kicked.store(true, Ordering::Relaxed);
                    return false;
                }
                thread::sleep(Duration::from_millis(2));
            }
        }
    }
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
    println!("[x] rejected a connection: {}", console_text(reason));
}

fn deny_mods(rt: &Runtime, send: &mut SendStream, conn: &quinn::Connection, ids: Vec<Arc<str>>) {
    let listed = ids.iter().map(|id| console_text(id)).collect::<Vec<_>>().join(", ");
    rt.block_on(async {
        let _ = protocol::write_frame_async(send, &ServerMessage::ModsDenied { ids }.encode()).await;
        let _ = send.finish();
        let _ = tokio::time::timeout(REJECT_DRAIN, conn.closed()).await;
    });
    println!("[x] refused mods: {listed}");
}

/// Scattered a little per id so players don't stack on the exact same block; scans outward for
/// the first level column (its four neighbours within one block), like the single-player spawn.
fn spawn_point(generator: &dyn TerrainGenerator, id: u32) -> DVec3 {
    if let Some(mut p) = generator.chart_spawn() {
        p.x += (id % 5) as f64 - 2.0;
        p.z += ((id / 5) % 5) as f64 - 2.0;
        return p;
    }
    let sx = (id % 8) as i32 - 3;
    let sz = ((id / 8) % 8) as i32 - 3;
    for r in 0..64 {
        for (dx, dz) in [(r, 0), (0, r), (-r, 0), (0, -r), (r, r), (-r, -r), (r, -r), (-r, r)] {
            let (x, z) = (sx + dx * 8, sz + dz * 8);
            let h = generator.height(x, z);
            let flat = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                .iter()
                .all(|&(ox, oz)| (generator.height(x + ox, z + oz) - h).abs() <= 1);
            if flat {
                return DVec3::new(x as f64 + 0.5, h as f64 + 3.0, z as f64 + 0.5);
            }
        }
    }
    let h = generator.height(sx, sz);
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

/// Peer text for the server console with control characters escaped, so a peer
/// cannot drive the operator's terminal.
fn console_text(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for c in raw.chars() {
        if c.is_control() {
            out.extend(c.escape_default());
        } else {
            out.push(c);
        }
    }
    out
}

/// The server's own chat name: any name whose letters and digits alone spell "server".
fn reserved_name(name: &str) -> bool {
    name.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).eq("server".chars())
}

#[cfg(test)]
mod tests {
    // Test setup (bind/connect/spawn) may unwrap: a panic here is a loud test
    // failure, which is exactly what the deny on the PRODUCTION paths exists
    // to prevent (a client thread silently poisoning the shared state).
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn test_generator() -> crate::world::terrain::Generator {
        crate::world::terrain::generator(&mut BlockRegistry::with_builtins(), 4242, Default::default())
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
        use crate::space::atlas::Patch;
        let terrain = test_generator();
        let centre = terrain.chart_spawn().expect("a charted start world");
        for id in 0..25 {
            let p = spawn_point(terrain.as_ref(), id);
            assert!((p.x - centre.x).abs() <= 2.0 + 1e-6, "id {id} x {}", p.x);
            assert!((p.z - centre.z).abs() <= 2.0 + 1e-6, "id {id} z {}", p.z);
            assert!((p.y - centre.y).abs() < 1e-6, "id {id} y {}", p.y);
        }
        let p = spawn_point(terrain.as_ref(), 12);
        assert!((p.x - centre.x).abs() < 1e-9 && (p.z - centre.z).abs() < 1e-9, "id 12 is the centre");
        let cell = terrain.atlases().iter().find_map(|a| a.storage_of(p)).expect("spawn storage");
        let (patch, local) = terrain.atlases().iter().find_map(|a| a.locate(cell)).expect("located");
        assert!(matches!(patch, Patch::Shell { band: 0, face: Face::PosY }), "{patch:?}");
        let open = i64::from(terrain.height(cell[0] as i32, cell[2] as i32));
        assert!((1..=2).contains(&(local[1] - open)), "local {} open {open}", local[1]);
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
            frame: DQuat::IDENTITY,
            velocity: Vec3::ZERO,
            up: Face::PosY,
            stance: Stance::Standing,
            last_move: Instant::now() - Duration::from_secs(10),
            budget: MOVE_FLOOR,
            op: false,
            visible: HashSet::new(),
            out,
            kick,
            ready: true,
            backlog: VecDeque::new(),
            backlog_bytes: 0,
            kicked: Arc::new(AtomicBool::new(false)),
            occupied: Vec::new(),
            cruising: false,
            cruise_speed: 0.0,
            novel: 0,
            announced: HashSet::new(),
        }
    }

    /// A world with no charts, so an edit's reach is judged at the cell itself.
    fn chartless() -> &'static crate::world::terrain::Generator {
        use std::sync::OnceLock;
        static FLAT: OnceLock<crate::world::terrain::Generator> = OnceLock::new();
        FLAT.get_or_init(|| Arc::new(crate::world::generation::FlatTerrain::new(&mut BlockRegistry::with_builtins(), 1)))
    }

    /// Noclip is open, so movement-envelope tests do not build a collision world
    /// on every step. The generator is built once for the process.
    fn lax_ctx() -> &'static Ctx {
        use std::sync::OnceLock;
        static CTX: OnceLock<Ctx> = OnceLock::new();
        CTX.get_or_init(|| test_ctx(true))
    }

    /// A throwaway kick handle for state-only players (never notified).
    /// A move that leaves the body frame, velocity, and up axis at their defaults.
    fn walk(shared: &Arc<Mutex<State>>, id: u32, pos: DVec3, yaw: f32, pitch: f32, stance: Stance) {
        on_move(shared, lax_ctx(), id, pos, yaw, pitch, DQuat::IDENTITY, Vec3::ZERO, Face::PosY, stance);
    }

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

    fn hello(name: &str, password: &str, protocol: u32, content: crate::net::ContentId) -> ClientMessage {
        ClientMessage::Hello {
            protocol,
            worldgen: content.worldgen,
            gravity: content.gravity,
            law: content.law,
            palette: content.palette,
            name: name.into(),
            password: password.into(),
            mods: vec![],
        }
    }

    fn server_content() -> crate::net::ContentId {
        crate::net::content_id(&BlockRegistry::with_builtins())
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
        // The palette first, exactly as `spawn` builds it, so generator ids mean the same here.
        let mut registry = BlockRegistry::with_builtins();
        crate::world::terrain::Materials::intern(&mut registry);
        State {
            edits: HashMap::new(),
            spec_pool: HashMap::new(),
            registry,
            players,
            grid: HashMap::new(),
            next_id: 2,
            day: 0.3,
            day_set: Instant::now(),
            reactions: ReactionScheduler::new(),
            max_speed: crate::player::MAX_SPEED,
            panic_tick: false,
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
            content: crate::net::content_id(&BlockRegistry::with_builtins()),
            day_secs: 600.0,
            teleport: if allow_teleport { TeleportPolicy::All } else { TeleportPolicy::Off },
            noclip: NoclipPolicy::All,
            worldgen: WorldgenKind::Diffusion,
            terrain: TerrainCfg::default(),
            seams: Seams::new(test_generator().atlases().to_vec()),
            generator: test_generator(),
            hooks: None,
            ops: Vec::new(),
            op_secrets: Vec::new(),
            mods_allow: Vec::new(),
            mods_deny: Vec::new(),
            store: None,
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

    /// The reference destructive pair as specs, with the target written into the ledger at `cell`.
    fn place_pair(shared: &Arc<Mutex<State>>, cell: Pos) -> (String, String) {
        let mut state = shared.lock_recover();
        let (a, e) = crate::sim::reactions::destructive_pair(&mut state.registry);
        let (sa, se) = (state.registry.spec(a), state.registry.spec(e));
        let spec = state.intern(&sa).unwrap();
        state.edits.insert(cell, Cell { spec, rev: 1, natural: false });
        (sa, se)
    }

    #[test]
    fn a_tool_use_from_an_unready_player_is_not_evaluated() {
        let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let mut players = HashMap::new();
        let mut p = test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick());
        p.ready = false;
        players.insert(1u32, p);
        let shared = Arc::new(Mutex::new(test_state(players)));
        let (_, tool) = place_pair(&shared, (8, 20, 8));
        let count = shared.lock_recover().registry.block_count();
        on_tool_use(&shared, &test_generator(), 1, 7, 8, 20, 8, 1, &tool);
        assert_eq!(shared.lock_recover().registry.block_count(), count, "nothing interned");
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn a_tool_use_runs_one_operation_of_the_law_on_the_server() {
        let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let (peer_out, peer_rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
        players.insert(2u32, test_player(DVec3::new(9.5, 20.0, 8.5), peer_out, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));
        let (_, tool) = place_pair(&shared, (8, 20, 8));
        on_tool_use(&shared, &test_generator(), 1, 7, 8, 20, 8, 1, &tool);
        let replies = drain_msgs(&rx);
        let [ServerMessage::ToolResult { req: 7, reacted: true, rev: 2, cell_spec, tool_spec }] = replies.as_slice() else {
            panic!("expected one ToolResult, got {replies:?}");
        };
        let state = shared.lock_recover();
        let cell = state.registry.lookup_spec(cell_spec).unwrap();
        let new_tool = state.registry.lookup_spec(tool_spec).unwrap();
        assert_eq!(state.registry.configuration(cell).len(), 3, "one element left the block");
        assert_eq!(state.registry.configuration(new_tool).len(), 5, "and joined the tool");
        assert_eq!(state.edits[&(8, 20, 8)].rev, 2);
        assert_eq!(state.reactions.pending(), 6, "the changed cell woke its contacts");
        drop(state);
        assert!(
            matches!(drain_msgs(&peer_rx).as_slice(), [ServerMessage::Edit { x: 8, y: 20, z: 8, rev: 2, .. }]),
            "peers see the cell change"
        );
        // A stale revision, a void tool and a tool out of reach are refused with `reacted: false`.
        on_tool_use(&shared, &test_generator(), 1, 8, 8, 20, 8, 1, &tool);
        on_tool_use(&shared, &test_generator(), 1, 9, 8, 20, 8, 2, "air");
        on_tool_use(&shared, &test_generator(), 1, 10, 80, 20, 8, 0, &tool);
        let refused = drain_msgs(&rx);
        assert_eq!(refused.len(), 3);
        assert!(refused.iter().all(|m| matches!(m, ServerMessage::ToolResult { reacted: false, .. })));
    }

    #[test]
    fn reaction_snapshots_carry_each_cell_once_with_its_final_content() {
        let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
        let mut state = test_state(players);
        let rock = state.registry.lookup_spec(&rock_spec()).or_else(|| state.registry.parse_spec(&rock_spec())).unwrap();
        let spec = state.intern(&state.registry.spec(rock)).unwrap();
        state.edits.insert((1, 2, 3), Cell { spec: spec.clone(), rev: 2, natural: false });
        state.edits.insert((4, 5, 6), Cell { spec, rev: 1, natural: false });
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
    fn on_edit_queues_place_and_break_events() {
        let (out, _rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));
        let rock = rock_spec();
        on_edit(&shared, None, chartless(), 1, 1, 8, 20, 8, 0, &rock);
        {
            let state = shared.lock_recover();
            assert_eq!(state.reactions.pending(), 6, "a placement wakes the cell's six contacts");
        }
        on_edit(&shared, None, chartless(), 1, 2, 8, 20, 8, 1, "air");
        let state = shared.lock_recover();
        assert_eq!(state.reactions.pending(), 6, "a removal wakes the same six (deduplicated)");
    }

    #[test]
    fn scripted_reactions_match_a_local_world() {
        use crate::render_config::RenderConfig;
        use crate::sim::reactions::{destructive_pair, Budget, ReactionScheduler};
        use crate::world::World;

        fn run<S: CellStore>(store: &mut S, a: BlockId, e: BlockId, y: i32) -> Vec<(Pos, Vec<u8>, Vec<u8>)> {
            store.set_block((0, y, 0), a);
            store.set_block((1, y, 0), e);
            let mut s = ReactionScheduler::new();
            s.wake_cell((1, y, 0));
            let mut out = Vec::new();
            for _ in 0..50 {
                for m in s.tick(store, Budget::DEFAULT) {
                    let r = store.registry();
                    out.push((m.pos, r.encoding(m.from).as_bytes().to_vec(), r.encoding(m.to).as_bytes().to_vec()));
                }
            }
            out
        }

        let mut world = World::with_kind(42, RenderConfig::default(), crate::world::generation::WorldgenKind::Diffusion, true);
        let (wa, we) = destructive_pair(world.registry_mut());
        let mut registry = BlockRegistry::with_builtins();
        let generator = crate::world::terrain::generator(&mut registry, 42, Default::default());
        let spawn = generator.chart_spawn().expect("a charted start world");
        let y = generator.atlases().iter().find_map(|a| a.storage_of(spawn)).expect("spawn storage")[1] as i32;
        let local = run(&mut world, wa, we, y);

        let (sa, se) = destructive_pair(&mut registry);
        let mut state = test_state(HashMap::new());
        state.registry = registry;
        let mut cells = ServerCells { state: &mut state, generator: &generator };
        let server = run(&mut cells, sa, se, y);
        assert_eq!(local, server);
        assert!(!local.is_empty(), "the pair must react");
    }

    /// An out-of-reach edit must be rejected; an in-reach one must be recorded.
    /// Exercised directly against the shared state without a socket.
    #[test]
    fn edit_reach_is_enforced() {
        let (out, _rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));

        on_edit(&shared, None, chartless(), 1, 1, 500, 20, 500, 0, "air"); // far away: rejected
        on_edit(&shared, None, chartless(), 1, 2, 8, 20, 8, 0, "air"); // in reach: recorded

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
        walk(&shared, 1, attempted, f32::NAN, 0.0, Stance::Sneaking);
        walk(&shared, 1, attempted, 0.0, f32::INFINITY, Stance::Sneaking);

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
        walk(&shared, 1, step, 0.1, 0.0, Stance::Standing);
        assert_eq!(shared.lock_recover().players[&1].pos, step);
        assert!(rx.try_recv().is_err(), "an accepted move needs no correction");

        // Move(target)+Edit(target) forging: the cross-map hop is refused...
        let forged = DVec3::new(4000.0, 20.0, 4000.0);
        walk(&shared, 1, forged, 0.0, 0.0, Stance::Standing);
        assert_eq!(shared.lock_recover().players[&1].pos, step, "position unchanged");
        match ServerMessage::decode(&rx.try_recv().expect("a correction is sent")) {
            Some(ServerMessage::Position { pos, .. }) => assert_eq!(pos, step),
            other => panic!("expected a Position snap-back, got {other:?}"),
        }
        // ...so the follow-up edit at the forged position stays out of reach.
        on_edit(&shared, None, chartless(), 1, 7, 4000, 20, 4000, 0, "air");
        assert!(!shared.lock_recover().edits.contains_key(&(4000, 20, 4000)));

        // Outside the world border: rejected no matter how slow.
        age_move(&shared, 1);
        walk(&shared, 1, DVec3::new(2.0e9, 20.0, 8.5), 0.0, 0.0, Stance::Standing);
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
            Some(ServerMessage::Position { pos, .. }) => assert_eq!(pos, far),
            other => panic!("expected a Position echo, got {other:?}"),
        }

        on_teleport(&shared, &test_ctx(false), 1, start);
        assert_eq!(shared.lock_recover().players[&1].pos, far, "refused teleport is not committed");
        match ServerMessage::decode(&rx.try_recv().expect("a correction is sent")) {
            Some(ServerMessage::Position { pos, .. }) => assert_eq!(pos, far),
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
        on_edit(&shared, None, chartless(), 1, 10, 8, 20, 8, 0, "air");
        assert_eq!(ack(&rx), (10, true, 1));

        // The racing loser expected revision 0 and is rejected — exactly one
        // reward, and its ack is the rollback signal.
        on_edit(&shared, None, chartless(), 1, 11, 8, 20, 8, 0, "air");
        assert_eq!(ack(&rx), (11, false, 1));

        // Building on the current revision succeeds.
        let rock = rock_spec();
        on_edit(&shared, None, chartless(), 1, 12, 8, 20, 8, 1, &rock);
        assert_eq!(ack(&rx), (12, true, 2));

        // Junk specs are rejected before touching the overlay or the pool.
        on_edit(&shared, None, chartless(), 1, 13, 8, 20, 8, 2, "banana:zzz");
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
        on_edit(&shared, None, chartless(), 1, 1, 8, 20, 8, 0, &rock);
        on_edit(&shared, None, chartless(), 1, 2, 8, 21, 8, 0, &rock);
        {
            let state = shared.lock_recover();
            assert_eq!(state.spec_pool.len(), 1, "equivalent spellings share one entry");
            assert_eq!(
                state.edits[&(8, 20, 8)].spec.as_ref(),
                state.edits[&(8, 21, 8)].spec.as_ref()
            );
        }

        // Overwriting both cells strands the old spec: it must leave the pool.
        on_edit(&shared, None, chartless(), 1, 3, 8, 20, 8, 1, "air");
        on_edit(&shared, None, chartless(), 1, 4, 8, 21, 8, 1, "air");
        {
            let state = shared.lock_recover();
            assert_eq!(state.spec_pool.len(), 1, "only \"air\" remains interned");
            assert!(state.spec_pool.contains_key("air"));
        }
    }

    /// A channel relays to the sender's interest set and nobody else, and the
    /// server stamps the sender. A full queue simply drops the frame.
    #[test]
    fn mod_data_relays_only_to_the_visible_set() {
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

        let channel = protocol::Channel::parse("voice").unwrap();
        on_mod_data(&shared, 1, channel, 42, vec![1, 2, 3].try_into().unwrap());

        match ServerMessage::decode(&rx2.try_recv().expect("the visible peer hears it")) {
            Some(ServerMessage::PeerModData { channel, sender, seq, bytes }) => {
                assert_eq!(channel.as_str(), "voice");
                assert_eq!((sender, seq, bytes.as_slice()), (1, 42, &[1, 2, 3][..]));
            }
            other => panic!("expected PeerModData, got {other:?}"),
        }
        assert!(rx3.try_recv().is_err(), "a peer outside interest hears nothing");
    }

    /// 100 frames on one channel, then the next is dropped. A second channel still has its own budget.
    #[test]
    fn channel_budget_is_per_channel_and_drops_the_overflow() {
        let mut budget = ChannelBudget::new();
        let voice = protocol::Channel::parse("voice").unwrap();
        let other = protocol::Channel::parse("other").unwrap();
        let now = Instant::now();
        for _ in 0..CHANNEL_RATE_LIMIT {
            assert!(budget.allow(&voice, now));
        }
        assert!(!budget.allow(&voice, now), "the 101st frame on one channel is dropped");
        assert!(budget.allow(&other, now), "a second channel keeps its own budget");
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
        assert_eq!(state.grid.get(&(0, 0, 0)).map(Vec::len), Some(1));
        let shared = Arc::new(Mutex::new(state));

        // Crossing the x border: the entry moves buckets and the emptied bucket
        // is dropped, not left behind as a leaked key.
        walk(&shared, 1, DVec3::new(INTEREST_RADIUS + 5.0, 20.0, 10.0), 0.0, 0.0, Stance::Standing);
        {
            let s = shared.lock_recover();
            assert_eq!(s.grid.get(&(1, 0, 0)).map(Vec::as_slice), Some(&[1u32][..]));
            assert!(!s.grid.contains_key(&(0, 0, 0)), "emptied bucket must be removed");
        }

        // Moving within the same bucket must not duplicate the entry.
        walk(&shared, 1, DVec3::new(INTEREST_RADIUS + 6.0, 20.0, 10.0), 0.0, 0.0, Stance::Standing);
        {
            let s = shared.lock_recover();
            assert_eq!(s.grid.get(&(1, 0, 0)).map(Vec::len), Some(1));
            assert_eq!(s.grid.len(), 1);
        }

        // Negative coordinates floor toward -infinity: -1.0 is bucket -1, not 0.
        // (Aged anchor: the hop back is real distance, and this test is about
        // grid bookkeeping, not the envelope.)
        age_move(&shared, 1);
        walk(&shared, 1, DVec3::new(-1.0, 20.0, -1.0), 0.0, 0.0, Stance::Standing);
        {
            let s = shared.lock_recover();
            assert_eq!(s.grid.get(&(-1, 0, -1)).map(Vec::len), Some(1));
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
                        id: 1,
                        pos,
                        yaw: 0.0,
                        pitch: 0.0,
                        frame: DQuat::IDENTITY,
                        velocity: Vec3::ZERO,
                        up: Face::PosY,
                        stance: Stance::Standing,
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
                            frame: player.frame,
                            velocity: player.velocity,
                            up: player.up,
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
        a.send_move(DVec3::new(4010.0, 30.0, 4010.0), 0.0, 0.0, DQuat::IDENTITY, Vec3::ZERO, Face::PosY, Stance::Standing);
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
        let id = server_content();
        let reason = reject_reason(
            handle.addr(),
            &hello("drifted", "", PROTOCOL_VERSION, crate::net::ContentId { palette: id.palette ^ 1, ..id }),
        );
        assert!(reason.contains("content"), "unexpected reason: {reason}");
        handle.stop();
    }

    /// Kind is not part of the content id. A client whose local mods would build Flat
    /// still joins a Diffusion server and adopts the kind Welcome carries.
    /// (Re-pinned: this used to reject `content_fingerprint_kind(Flat)`.)
    #[test]
    fn flat_client_joins_a_diffusion_server() {
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
        match raw_reply(handle.addr(), &hello("flat", "", PROTOCOL_VERSION, server_content())) {
            ServerMessage::Welcome { worldgen, .. } => assert_eq!(worldgen, WorldgenKind::Diffusion),
            other => panic!("a code-only fingerprint must be welcomed, got {other:?}"),
        }
        handle.stop();
    }

    #[test]
    fn welcome_carries_the_servers_worldgen_kind_and_cfg() {
        let terrain = TerrainCfg { relief: 175, caves: 50, mines: 0, space: 125, ..Default::default() };
        let handle = spawn(
            0,
            Config {
                password: String::new(),
                seed: 11,
                worldgen: WorldgenKind::Diffusion,
                terrain,
                ..Config::default()
            },
        )
        .unwrap();
        let hello = hello("guest", "", PROTOCOL_VERSION, server_content());
        match raw_reply(handle.addr(), &hello) {
            ServerMessage::Welcome { worldgen, terrain: got, seed, .. } => {
                assert_eq!(seed, 11);
                assert_eq!(worldgen, WorldgenKind::Diffusion);
                assert_eq!(got, terrain);
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
        let old = state.spec_pool.get_key_value("spec-1").unwrap().0.clone();
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
        assert_eq!(bucket_of(on_edge), (1, 0, 0));
        assert_eq!(bucket_of(just_inside), (0, 0, 0));

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
            walk(&shared, 1, forged, 0.0, 0.0, Stance::Standing);
        }
        assert_eq!(shared.lock_recover().players[&1].pos, start);
        let mut corrections = 0;
        while let Ok(frame) = rx.try_recv() {
            match ServerMessage::decode(&frame) {
                Some(ServerMessage::Position { pos, .. }) => {
                    assert_eq!(pos, start);
                    corrections += 1;
                }
                other => panic!("expected Position, got {other:?}"),
            }
        }
        assert!(corrections >= 1, "the burst must snap back at least once");
        let legal = DVec3::new(10.5, 20.0, 8.5);
        walk(&shared, 1, legal, 0.0, 0.0, Stance::Standing);
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
            Some(ServerMessage::Position { pos, .. }) => assert_eq!(pos, dest),
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
        on_edit(&shared, None, chartless(), 1, 1, 8, 20, 8, 0, "air");
        assert!(shared.lock_recover().edits.contains_key(&(8, 20, 8)), "exact REACH must land");

        let just_out = DVec3::new(center.x + EDIT_REACH * 1.001, center.y, center.z);
        shared.lock_recover().players.get_mut(&1).unwrap().pos = just_out;
        on_edit(&shared, None, chartless(), 1, 2, 8, 21, 8, 0, "air");
        assert!(!shared.lock_recover().edits.contains_key(&(8, 21, 8)));

        on_edit(&shared, None, chartless(), 1, 3, i32::MIN, i32::MIN, i32::MIN, 0, "air");
        on_edit(&shared, None, chartless(), 1, 4, i32::MAX, i32::MAX, i32::MAX, 0, "air");
        let far = crate::math::WORLD_BORDER as i32 + 64;
        on_edit(&shared, None, chartless(), 1, 5, far, 20, far, 0, "air");
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
        // Swings are cosmetic: a flood past the byte cap must not kick.
        let swings = BACKLOG_BYTES / 4 + 8;
        for i in 0..swings {
            broadcast_all(&shared, &ServerMessage::PeerSwing { id: i as u32 }, None);
        }
        assert!(!shared.lock_recover().players[&1].kicked.load(Ordering::Relaxed), "swings are dropped, not a kick");
        assert!(shared.lock_recover().players[&1].backlog_bytes <= BACKLOG_BYTES);
        // Pong is essential. Once the swings are gone, a byte-cap overflow kicks.
        let pongs = BACKLOG_BYTES / 4 + 8;
        for i in 0..pongs {
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
        let id = server_content();
        let reason = reject_reason(addr, &hello("eve", "nope", PROTOCOL_VERSION, id));
        assert!(reason.to_lowercase().contains("password"), "{reason}");
        let reason = reject_reason(
            addr,
            &hello("eve", "pw", PROTOCOL_VERSION.wrapping_add(1), id),
        );
        assert!(reason.to_lowercase().contains("protocol"), "{reason}");
        assert!(reason.contains(&format!("server v{PROTOCOL_VERSION}")), "{reason}");
        assert!(reason.contains(&format!("client v{}", PROTOCOL_VERSION.wrapping_add(1))), "{reason}");
        let drifted = crate::net::ContentId { law: id.law ^ 1, ..id };
        let reason = reject_reason(addr, &hello("eve", "pw", PROTOCOL_VERSION, drifted));
        assert!(reason.contains("content"), "{reason}");
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

        on_edit(&shared, Some(&table), chartless(), 1, 42, 8, 20, 8, 0, "air");

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

        on_edit(&shared, Some(&table), chartless(), 1, 1, 8, 20, 8, 0, "air");

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
                        natural: false,
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
    fn server_block_reads_the_generator_like_the_client() {
        use crate::space::atlas::Patch;
        let mut registry = BlockRegistry::with_builtins();
        let g = crate::world::terrain::generator(&mut registry, 4242, Default::default());
        let home = g.cosmos().expect("cosmos").home();
        let atlas = g
            .atlases()
            .iter()
            .find(|a| (a.centre - home.centre_f()).length() < 1.0)
            .expect("the start world is charted");
        let n = atlas.bands[0].n;
        let s = atlas.storage(Patch::Shell { band: 0, face: Face::PosY }, [n / 2, 0, n / 2]);
        let (x0, z0) = (s[0] as i32, s[2] as i32);
        for x in x0..x0 + 40 {
            for z in z0..z0 + 40 {
                let h = g.height(x, z);
                assert_ne!(h, i32::MIN, "({x},{z}) has no chart surface");
                assert_eq!(g.block_at(x, h - 1, z, h), g.voxel_at(x, h - 1, z), "({x},{},{z})", h - 1);
            }
        }
    }

    fn raw_payload_reply(addr: SocketAddr, payload: &[u8]) -> ServerMessage {
        let target = SocketAddr::from((Ipv4Addr::LOCALHOST, addr.port()));
        let (rt, ep) = client_endpoint();
        rt.block_on(async {
            let conn = ep.connect(target, "watt").unwrap().await.unwrap();
            let (mut s, mut r) = conn.open_bi().await.unwrap();
            protocol::write_frame_async(&mut s, payload).await.unwrap();
            let mut buf = Vec::new();
            protocol::read_frame_async(&mut r, &mut buf).await.unwrap();
            ServerMessage::decode(&buf).unwrap()
        })
    }

    fn reject_payload(addr: SocketAddr, payload: &[u8]) -> String {
        match raw_payload_reply(addr, payload) {
            ServerMessage::Reject { reason } => reason.to_string(),
            other => panic!("expected Reject, got {other:?}"),
        }
    }

    #[test]
    fn hello_rejections_name_the_protocol_before_a_full_decode() {
        let handle = spawn(0, Config { seed: 1, ..Config::default() }).unwrap();
        let addr = handle.addr();
        let reason = reject_payload(addr, &[1, 0, 0, 0, 0]);
        assert!(reason.contains("expected hello"), "{reason}");
        let mut old = vec![0u8];
        old.extend_from_slice(&12u32.to_le_bytes());
        old.extend_from_slice(&[0u8; 8]);
        let reason = reject_payload(addr, &old);
        assert!(reason.contains(&format!("server v{PROTOCOL_VERSION}")), "{reason}");
        assert!(reason.contains("client v12"), "{reason}");
        let mut trunc = vec![0u8];
        trunc.extend_from_slice(&PROTOCOL_VERSION.to_le_bytes());
        trunc.extend_from_slice(&[0xff, 0x00]);
        let reason = reject_payload(addr, &trunc);
        assert!(reason.contains("malformed"), "{reason}");
        handle.stop();
    }

    #[test]
    fn snapshot_batches_stay_within_the_frame() {
        let spec: Arc<str> = "s".repeat(MAX_SPEC).into();
        let edits: Vec<_> = (0..40).map(|i| (i, 0, 0, 1u32, spec.clone())).collect();
        let mut n = 0;
        for_snapshot_batches(&edits, |batch| {
            let msg = ServerMessage::Snapshot { edits: batch.to_vec() };
            let encoded = msg.encode();
            let expect = SNAPSHOT_HEAD + batch.iter().map(|e| SNAPSHOT_EDIT_FIXED + e.4.len()).sum::<usize>();
            assert_eq!(encoded.len(), expect);
            assert!(encoded.len() <= MAX_FRAME, "{}", encoded.len());
            n += batch.len();
        });
        assert_eq!(n, edits.len());
        let mut empty_emitted = false;
        for_snapshot_batches(&[], |_| empty_emitted = true);
        assert!(!empty_emitted);
    }

    #[test]
    fn full_configuration_spec_fits_and_is_accepted() {
        let elems: Vec<_> = (0..material::CAPACITY).map(|i| material::Element::new([i as u8, 1, 2, 3])).collect();
        let cfg = material::Configuration::new(elems).unwrap();
        let spec = {
            let mut registry = BlockRegistry::with_builtins();
            let id = registry.intern(&cfg).unwrap();
            registry.spec(id)
        };
        assert_eq!(spec.len(), MAX_SPEC);
        assert!(spec.len() > 256, "the old cap rejected this block");
        let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));
        on_edit(&shared, None, chartless(), 1, 1, 8, 20, 8, 0, &spec);
        match ServerMessage::decode(&rx.try_recv().unwrap()) {
            Some(ServerMessage::EditAck { accepted: true, .. }) => {}
            other => panic!("a full configuration must be accepted, got {other:?}"),
        }
        assert!(shared.lock_recover().edits.contains_key(&(8, 20, 8)));
    }

    #[test]
    fn writer_error_kicks_and_a_clean_close_does_not() {
        let (tx, rx) = sync_channel::<Arc<[u8]>>(4);
        let kick = Arc::new(Notify::new());
        let kick2 = kick.clone();
        let writer = thread::spawn(move || {
            drain_writer(rx, &kick2, |_| Err(io::Error::other("closed")));
        });
        tx.send(Arc::<[u8]>::from([1u8, 2, 3].as_slice())).unwrap();
        writer.join().unwrap();
        let rt = Runtime::new().unwrap();
        let notified = kick.notified();
        rt.block_on(async {
            tokio::time::timeout(Duration::from_millis(50), notified)
                .await
                .expect("a write error must kick the reader");
        });

        let (tx, rx) = sync_channel::<Arc<[u8]>>(4);
        let kick = Arc::new(Notify::new());
        let kick2 = kick.clone();
        let writer = thread::spawn(move || {
            drain_writer(rx, &kick2, |_| Ok(()));
        });
        tx.send(Arc::<[u8]>::from([9u8].as_slice())).unwrap();
        drop(tx);
        writer.join().unwrap();
        let notified = kick.notified();
        rt.block_on(async {
            assert!(
                tokio::time::timeout(Duration::from_millis(50), notified).await.is_err(),
                "channel close is depart, not a kick"
            );
        });
    }

    #[test]
    fn spec_pool_counts_cells_not_arc_clones() {
        let (out, _rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));
        let rock = rock_spec();
        on_edit(&shared, None, chartless(), 1, 1, 8, 20, 8, 0, &rock);
        on_edit(&shared, None, chartless(), 1, 2, 8, 21, 8, 0, &rock);
        let extra = {
            let state = shared.lock_recover();
            assert_eq!(state.spec_pool.get(rock.as_str()).copied(), Some(2));
            state.edits[&(8, 21, 8)].spec.clone()
        };
        on_edit(&shared, None, chartless(), 1, 3, 8, 20, 8, 1, "air");
        assert_eq!(shared.lock_recover().spec_pool.get(rock.as_str()).copied(), Some(1));
        on_edit(&shared, None, chartless(), 1, 4, 8, 21, 8, 1, "air");
        assert!(!shared.lock_recover().spec_pool.contains_key(rock.as_str()), "zero cells frees the entry");
        drop(extra);
    }

    fn numbered_spec(n: u8) -> String {
        let cfg = material::Configuration::single(material::Element::new([200, n, 17, 3]));
        let bytes = cfg.encode();
        let mut s = String::from("c:");
        for b in bytes.as_bytes() {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }

    #[test]
    fn stale_novel_spec_does_not_grow_the_registry_and_quota_holds() {
        let (out, rx) = sync_channel::<Arc<[u8]>>(8);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));
        let before = shared.lock_recover().registry.block_count();
        on_edit(&shared, None, chartless(), 1, 1, 8, 20, 8, 0, &numbered_spec(1));
        assert_eq!(shared.lock_recover().registry.block_count(), before + 1);
        on_edit(&shared, None, chartless(), 1, 2, 8, 20, 8, 0, &numbered_spec(2));
        assert_eq!(shared.lock_recover().registry.block_count(), before + 1, "stale expect must not intern");
        assert_eq!(shared.lock_recover().players[&1].novel, 1);
        let _ = rx;

        let (out, rx) = sync_channel::<Arc<[u8]>>(NOVEL_SPEC_QUOTA as usize + 8);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));
        let before = shared.lock_recover().registry.block_count();
        let mut expect = 0u32;
        for i in 0..NOVEL_SPEC_QUOTA {
            on_edit(&shared, None, chartless(), 1, i + 1, 8, 20, 8, expect, &numbered_spec(i as u8));
            expect += 1;
        }
        assert_eq!(shared.lock_recover().registry.block_count(), before + NOVEL_SPEC_QUOTA as usize);
        assert_eq!(shared.lock_recover().players[&1].novel, NOVEL_SPEC_QUOTA);
        let count = shared.lock_recover().registry.block_count();
        on_edit(&shared, None, chartless(), 1, 1000, 8, 20, 8, expect, &numbered_spec(250));
        assert_eq!(shared.lock_recover().registry.block_count(), count, "past the quota: no intern");
        on_edit(&shared, None, chartless(), 1, 1001, 8, 20, 8, expect, "air");
        assert!(shared.lock_recover().edits[&(8, 20, 8)].spec.as_ref() == "air");
        assert_eq!(shared.lock_recover().players[&1].novel, NOVEL_SPEC_QUOTA, "air is not novel");
        let _ = rx;
    }

    /// Anchor the envelope one realistic move gap ago: the next move's budget refills for that long.
    fn stamp_gap(shared: &Arc<Mutex<State>>, id: u32) {
        if let Some(h) = shared.lock_recover().players.get_mut(&id) {
            h.last_move = Instant::now() - Duration::from_millis(100);
        }
    }

    #[test]
    fn fall_flight_and_cruise_follow_the_reported_speed() {
        let (out, rx) = sync_channel::<Arc<[u8]>>(8);
        let start = DVec3::new(8.5, 20.0, 8.5);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(start, out, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));
        stamp_gap(&shared, 1);

        let fall = DVec3::new(start.x, start.y - 2_000.0, start.z);
        on_move(
            &shared,
            lax_ctx(),
            1,
            fall,
            0.0,
            0.0,
            DQuat::IDENTITY,
            Vec3::new(0.0, -30_000.0, 0.0),
            Face::PosY,
            Stance::Standing,
        );
        assert_eq!(shared.lock_recover().players[&1].pos, fall, "a long fall under MAX_SPEED is not snapped");
        assert!(rx.try_recv().is_err());

        // The fall left a stored speed. Clear it so the next drop is judged on a zero report.
        shared.lock_recover().players.get_mut(&1).unwrap().velocity = Vec3::ZERO;
        stamp_gap(&shared, 1);
        let forged = DVec3::new(fall.x, fall.y - 2_000.0, fall.z);
        on_move(&shared, lax_ctx(), 1, forged, 0.0, 0.0, DQuat::IDENTITY, Vec3::ZERO, Face::PosY, Stance::Standing);
        assert_eq!(shared.lock_recover().players[&1].pos, fall, "the same drop with no speed is a teleport");

        stamp_gap(&shared, 1);
        let flown = DVec3::new(start.x + 1_000.0, start.y, start.z);
        // Fly from `fall`, where the refused drop left the player.
        let speed = crate::player::MAX_SPEED as f32;
        on_move(
            &shared,
            lax_ctx(),
            1,
            DVec3::new(fall.x + 1_000.0, fall.y, fall.z),
            0.0,
            0.0,
            DQuat::IDENTITY,
            Vec3::new(speed, 0.0, 0.0),
            Face::PosY,
            Stance::Standing,
        );
        assert_eq!(shared.lock_recover().players[&1].pos.x, fall.x + 1_000.0, "fast flight is not snapped");
        let _ = flown;

        stamp_gap(&shared, 1);
        let too_far = DVec3::new(fall.x + crate::player::MAX_SPEED * 2.0, fall.y, fall.z);
        on_move(
            &shared,
            lax_ctx(),
            1,
            too_far,
            0.0,
            0.0,
            DQuat::IDENTITY,
            Vec3::new(speed, 0.0, 0.0),
            Face::PosY,
            Stance::Standing,
        );
        assert!(
            (shared.lock_recover().players[&1].pos.x - (fall.x + 1_000.0)).abs() < 1.0,
            "past MAX_SPEED without cruise snaps back"
        );

        on_cruise(&shared, 1, crate::player::CRUISE_MAX);
        stamp_gap(&shared, 1);
        let cruise_to = DVec3::new(fall.x + crate::player::MAX_SPEED * 2.0, fall.y, fall.z);
        on_move(
            &shared,
            lax_ctx(),
            1,
            cruise_to,
            0.0,
            0.0,
            DQuat::IDENTITY,
            Vec3::new(crate::player::CRUISE_MAX as f32, 0.0, 0.0),
            Face::PosY,
            Stance::Standing,
        );
        assert_eq!(shared.lock_recover().players[&1].pos, cruise_to, "a declared cruise is not snapped");
    }

    #[test]
    fn peer_joined_is_delivered_once() {
        let (out1, rx1) = sync_channel::<Arc<[u8]>>(8);
        let (out2, rx2) = sync_channel::<Arc<[u8]>>(8);
        let mut p1 = test_player(DVec3::new(0.0, 20.0, 0.0), out1, test_kick());
        p1.announced.insert(2);
        let p2 = test_player(DVec3::new(1.0, 20.0, 0.0), out2, test_kick());
        let mut players = HashMap::new();
        players.insert(1u32, p1);
        players.insert(2u32, p2);
        let shared = Arc::new(Mutex::new(test_state(players)));
        broadcast_all(&shared, &ServerMessage::PeerJoined { id: 2, name: "b".into() }, None);
        assert!(rx1.try_recv().is_err(), "player 1 was already told about 2");
        assert!(rx2.try_recv().is_ok(), "player 2 had not been told");
        broadcast_all(&shared, &ServerMessage::PeerJoined { id: 3, name: "c".into() }, None);
        assert!(rx1.try_recv().is_ok());
        broadcast_all(&shared, &ServerMessage::PeerJoined { id: 3, name: "c".into() }, None);
        assert!(rx1.try_recv().is_err(), "the second announcement is dropped");
    }

    #[test]
    fn joiner_time_matches_the_clock_at_send() {
        use crate::net::client::Connection;
        let handle = spawn(0, Config { seed: 1, day_secs: 600.0, ..Config::default() }).unwrap();
        {
            let mut state = handle.state.lock_recover();
            state.day = 0.0;
            state.day_set = Instant::now() - Duration::from_secs(300);
        }
        let mut conn = Connection::connect("127.0.0.1", handle.addr().port(), "ada", "").unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut day = None;
        while day.is_none() && Instant::now() < deadline {
            day = conn.poll().into_iter().find_map(|e| match e {
                crate::net::client::Incoming::Time { day, .. } => Some(day),
                _ => None,
            });
            if day.is_none() {
                thread::sleep(Duration::from_millis(10));
            }
        }
        let day = day.expect("Welcome is consumed at connect; Time follows it");
        assert!((day - 0.5).abs() < 0.05, "live clock, got {day}");
        handle.stop();
    }

    #[test]
    fn edits_over_the_rate_budget_are_rejected() {
        use crate::net::client::Connection;
        let handle = spawn(0, Config { seed: 1, ..Config::default() }).unwrap();
        let mut conn = Connection::connect("127.0.0.1", handle.addr().port(), "ada", "").unwrap();
        let s = conn.spawn();
        let (x, y, z) = (
            crate::math::block_coord(s.x),
            crate::math::block_coord(s.y),
            crate::math::block_coord(s.z),
        );
        for _ in 0..(SWING_RATE + 5) {
            conn.send_swing();
        }
        // Swings have their own budget, so the first edit is still accepted.
        let kept = conn.send_edit(x, y, z, "air".into()).expect("air is sent");
        let mut over = kept;
        for i in 0..EDIT_RATE {
            over = conn.send_edit(x, y + 1 + i as i32, z, "air".into()).expect("air is sent");
        }
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut accepted = false;
        let mut rejected = false;
        while !(accepted && rejected) && Instant::now() < deadline {
            for event in conn.poll() {
                match event {
                    crate::net::client::Incoming::EditAccepted { req } if req == kept => accepted = true,
                    crate::net::client::Incoming::EditRejected { req, restore: true } if req == over => rejected = true,
                    _ => {}
                }
            }
            if !(accepted && rejected) {
                thread::sleep(Duration::from_millis(10));
            }
        }
        assert!(accepted, "swings must not spend the edit budget");
        assert!(rejected, "an edit past the edit budget must be acked rejected");
        handle.stop();
    }

    #[test]
    fn joins_over_ipv6_loopback_and_localhost() {
        use crate::net::client::Connection;
        let handle = spawn(0, Config { seed: 1, ..Config::default() }).unwrap();
        let port = handle.addr().port();
        let v6 = Connection::connect("::1", port, "v6", "").expect("::1");
        assert!(v6.is_alive());
        let local = Connection::connect("localhost", port, "local", "").expect("localhost");
        assert!(local.is_alive());
        assert_ne!(v6.player_id(), local.player_id());
        handle.stop();
    }

    fn flat(config: Config) -> ServerHandle {
        spawn(0, Config { seed: 1, worldgen: WorldgenKind::Flat, ..config }).unwrap()
    }

    fn drain(rx: &std::sync::mpsc::Receiver<Arc<[u8]>>) -> Vec<ServerMessage> {
        let mut out = Vec::new();
        while let Ok(frame) = rx.try_recv() {
            if let Some(msg) = ServerMessage::decode(&frame) {
                out.push(msg);
            }
        }
        out
    }

    /// `/tp` and `/time` are operator-only under [`TeleportPolicy::Ops`]. The
    /// reason is a private chat from "server". The speed cap is the same
    /// whether or not the client claimed a mod.
    #[test]
    fn non_operator_time_and_teleport_are_refused_with_a_reason() {
        let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let start = DVec3::new(8.5, 20.0, 8.5);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(start, out, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));
        let mut ops = test_ctx(true);
        ops.teleport = TeleportPolicy::Ops;
        ops.ops = vec!["p".into()];

        on_set_time(&shared, &ops, 1, 0.2);
        assert!((shared.lock_recover().day - 0.2).abs() < 1e-6);
        assert!(drain(&rx).iter().any(|m| matches!(m, ServerMessage::Time { day, .. } if (*day - 0.2).abs() < 1e-4)));

        let far = DVec3::new(80.5, 20.0, 8.5);
        on_teleport(&shared, &ops, 1, far);
        assert_eq!(shared.lock_recover().players[&1].pos, far);

        let mut guest = test_ctx(true);
        guest.teleport = TeleportPolicy::Ops;
        on_set_time(&shared, &guest, 1, 0.9);
        assert!((shared.lock_recover().day - 0.2).abs() < 1e-6, "a guest does not move the clock");
        assert!(
            drain(&rx).iter().any(|m| matches!(
                m,
                ServerMessage::Chat { from_name, text, .. } if from_name.as_ref() == "server" && text.as_ref() == "only an operator can set the time"
            ))
        );

        on_teleport(&shared, &guest, 1, start);
        assert_eq!(shared.lock_recover().players[&1].pos, far, "a guest teleport is not committed");
        let refused = drain(&rx);
        assert!(refused.iter().any(|m| matches!(m, ServerMessage::Position { pos, .. } if *pos == far)));
        assert!(refused.iter().any(|m| matches!(
            m,
            ServerMessage::Chat { text, .. } if text.as_ref() == "only an operator can teleport"
        )));

        let off = test_ctx(false);
        on_teleport(&shared, &off, 1, start);
        assert!(drain(&rx).iter().any(|m| matches!(
            m,
            ServerMessage::Chat { text, .. } if text.as_ref() == "teleport is not permitted"
        )));
    }

    /// A cap below [`crate::player::MAX_SPEED`] bounds flight and cruise. `/tp`
    /// stays refused when the policy is off. Solid overlap is [`noclip_snaps_a_body_in_solid_ground`].
    #[test]
    fn flyspeed_above_the_server_cap_is_snapped() {
        let cap = 30.0 * crate::math::PER_METER;
        let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let start = DVec3::new(8.5, 40.0, 8.5);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(start, out, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));
        shared.lock_recover().max_speed = cap;

        let near = DVec3::new(start.x + 10.0 * crate::math::PER_METER, start.y, start.z);
        on_move(
            &shared, lax_ctx(), 1, near, 0.0, 0.0, DQuat::IDENTITY,
            Vec3::new(cap as f32, 0.0, 0.0), Face::PosY, Stance::Standing,
        );
        assert_eq!(shared.lock_recover().players[&1].pos, near, "10 m under the cap commits");

        age_move(&shared, 1);
        let leap = DVec3::new(near.x + 500.0 * crate::math::PER_METER, near.y, near.z);
        on_move(
            &shared, lax_ctx(), 1, leap, 0.0, 0.0, DQuat::IDENTITY,
            Vec3::new(cap as f32, 0.0, 0.0), Face::PosY, Stance::Standing,
        );
        assert_eq!(shared.lock_recover().players[&1].pos, near, "500 m over the cap snaps");
        assert!(drain(&rx).iter().any(|m| matches!(m, ServerMessage::Position { .. })));

        on_cruise(&shared, 1, crate::player::CRUISE_MAX);
        assert!((shared.lock_recover().players[&1].cruise_speed - cap).abs() < 1e-6);
        age_move(&shared, 1);
        on_move(
            &shared, lax_ctx(), 1, leap, 0.0, 0.0, DQuat::IDENTITY,
            Vec3::new(crate::player::CRUISE_MAX as f32, 0.0, 0.0), Face::PosY, Stance::Standing,
        );
        assert_eq!(shared.lock_recover().players[&1].pos, near, "cruise cannot outrun a lower cap");

        on_teleport(&shared, &test_ctx(false), 1, leap);
        assert_eq!(shared.lock_recover().players[&1].pos, near);
    }

    #[test]
    fn restarted_server_serves_the_same_edits() {
        use crate::net::client::{Connection, Incoming};
        let path = crate::save::store::test_temp_path("restart");
        let _ = std::fs::remove_file(&path);
        let handle = spawn(0, Config {
            seed: 42,
            worldgen: WorldgenKind::Flat,
            world: Some(path.clone()),
            ops: vec!["ada".into()],
            ..Config::default()
        }).unwrap();
        let mut conn = Connection::connect("127.0.0.1", handle.addr().port(), "ada", "").unwrap();
        let s = conn.spawn();
        // The grass under the spawn: breaking it changes the world, so the file keeps it.
        let (x, y, z) = (crate::math::block_coord(s.x), crate::world::generation::FLAT_HEIGHT - 1, crate::math::block_coord(s.z));
        let req = conn.send_edit(x, y, z, "air".into()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);
        let mut accepted = false;
        while !accepted && Instant::now() < deadline {
            accepted = conn.poll().into_iter().any(|e| matches!(e, Incoming::EditAccepted { req: r } if r == req));
            if !accepted {
                thread::sleep(Duration::from_millis(10));
            }
        }
        assert!(accepted, "the edit is committed before shutdown");
        conn.send_set_time(0.2);
        let deadline = Instant::now() + Duration::from_secs(3);
        let mut day = None;
        while day.is_none() && Instant::now() < deadline {
            for event in conn.poll() {
                if let Incoming::Time { day: d, .. } = event {
                    day = Some(d);
                }
            }
            if day.is_none() {
                thread::sleep(Duration::from_millis(10));
            }
        }
        assert!((day.expect("time reply") - 0.2).abs() < 0.02);
        drop(conn);
        handle.stop();

        let again = spawn(0, Config {
            seed: 99,
            worldgen: WorldgenKind::Diffusion,
            world: Some(path.clone()),
            warn_world_overrides: true,
            ..Config::default()
        }).unwrap();
        let mut bob = Connection::connect("127.0.0.1", again.addr().port(), "bob", "").unwrap();
        assert_eq!(bob.seed(), 42);
        assert_eq!(bob.worldgen(), WorldgenKind::Flat);
        let deadline = Instant::now() + Duration::from_secs(3);
        let mut saw_edit = false;
        let mut saw_day = false;
        while Instant::now() < deadline && !(saw_edit && saw_day) {
            for event in bob.poll() {
                match event {
                    Incoming::Mutation { x: mx, y: my, z: mz, spec } if (mx, my, mz) == (x, y, z) && spec.as_ref() == "air" => {
                        saw_edit = true;
                    }
                    Incoming::Time { day, .. } if (day - 0.2).abs() < 0.05 => saw_day = true,
                    _ => {}
                }
            }
            thread::sleep(Duration::from_millis(10));
        }
        assert!(saw_edit, "the restarted world still has the edit");
        assert!(saw_day, "the restarted world still has the clock");
        again.stop();
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn stop_closes_connections_and_frees_the_port() {
        use crate::net::client::{Connection, Incoming};
        let handle = flat(Config::default());
        let port = handle.addr().port();
        let mut conn = Connection::connect("127.0.0.1", port, "ada", "").unwrap();
        handle.stop();
        let deadline = Instant::now() + Duration::from_secs(3);
        let mut reason = None;
        while reason.is_none() && Instant::now() < deadline {
            for event in conn.poll() {
                if let Incoming::Disconnected { reason: text } = event {
                    reason = Some(text);
                }
            }
            if reason.is_none() {
                thread::sleep(Duration::from_millis(20));
            }
        }
        let reason = reason.expect("the close arrives before the idle timeout");
        assert!(
            reason.to_ascii_lowercase().contains("shutting down"),
            "client saw {reason:?}"
        );
        drop(conn);
        let mut rebound = None;
        for _ in 0..50 {
            match spawn(port, Config { seed: 1, worldgen: WorldgenKind::Flat, ..Config::default() }) {
                Ok(handle) => {
                    rebound = Some(handle);
                    break;
                }
                Err(_) => thread::sleep(Duration::from_millis(20)),
            }
        }
        rebound.expect("the port accepts a new server").stop();
    }

    /// Sends SIGTERM to this test process: tokio's handler, installed by `run`, takes it.
    #[cfg(unix)]
    #[test]
    fn sigterm_saves_the_world() {
        use crate::net::client::Connection;
        let path = crate::save::store::test_temp_path("sigterm");
        let _ = std::fs::remove_file(&path);
        let port = std::net::UdpSocket::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let server_path = path.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let server = thread::spawn(move || {
            let result = run(port, Config {
                seed: 42,
                worldgen: WorldgenKind::Flat,
                world: Some(server_path),
                ..Config::default()
            });
            let _ = tx.send(result);
        });
        let deadline = Instant::now() + Duration::from_secs(8);
        let mut conn = None;
        while conn.is_none() && Instant::now() < deadline {
            if let Ok(result) = rx.try_recv() {
                panic!("server exited before a client connected: {result:?}");
            }
            match Connection::connect("127.0.0.1", port, "ada", "") {
                Ok(c) => conn = Some(c),
                Err(_) => thread::sleep(Duration::from_millis(30)),
            }
        }
        let _conn = conn.expect("server accepted a connection");
        assert!(!path.exists(), "nothing is written until a save");
        let _ = std::process::Command::new("kill")
            .args(["-TERM", &std::process::id().to_string()])
            .status();
        match rx.recv_timeout(Duration::from_secs(8)) {
            Ok(Ok(())) => {}
            Ok(Err(e)) => panic!("server failed: {e}"),
            Err(_) => panic!("SIGTERM did not stop the server"),
        }
        server.join().unwrap();
        let bytes = std::fs::read(&path).expect("SIGTERM saved the world");
        let doc = match crate::save::format::decode(&bytes).unwrap() {
            crate::save::format::Decoded::Intact(doc) => doc,
            crate::save::format::Decoded::Salvaged { .. } => panic!("shutdown save must be intact"),
        };
        assert_eq!(doc.meta.seed, 42);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn duplicate_and_reserved_names_are_rejected() {
        use crate::net::client::Connection;
        let handle = flat(Config::default());
        let port = handle.addr().port();
        let _ada = Connection::connect("127.0.0.1", port, "Ada", "").unwrap();
        let dup = match Connection::connect("127.0.0.1", port, "ada", "") {
            Ok(_) => panic!("duplicate name was admitted"),
            Err(err) => err,
        };
        assert!(dup.to_string().contains("already in use"), "{dup}");
        let reserved = match Connection::connect("127.0.0.1", port, "Server", "") {
            Ok(_) => panic!("reserved name was admitted"),
            Err(err) => err,
        };
        assert!(reserved.to_string().contains("reserved"), "{reserved}");
        handle.stop();
    }

    #[test]
    fn allow_list_admits_a_fully_listed_client() {
        use crate::net::client::Connection;
        let handle = flat(Config {
            mods_allow: vec!["pwc.hotbar".into()],
            ..Config::default()
        });
        let port = handle.addr().port();
        let listed = [("pwc.hotbar".into(), "0.1.0".into())];
        let ok = Connection::connect_with("127.0.0.1", port, "ada", "", &listed).expect("listed mod");
        assert!(ok.is_alive());
        drop(ok);
        let extra = [
            ("pwc.hotbar".into(), "0.1.0".into()),
            ("pwc.dev-toolkit".into(), "1.0.0".into()),
        ];
        let err = match Connection::connect_with("127.0.0.1", port, "bea", "", &extra) {
            Ok(_) => panic!("an unlisted mod was admitted"),
            Err(err) => err,
        };
        assert_eq!(err.mods_denied, vec!["pwc.dev-toolkit".to_string()]);
        let none = Connection::connect_with("127.0.0.1", port, "cy", "", &[]).expect("no mods enabled");
        assert!(none.is_alive());
        handle.stop();
    }

    #[test]
    fn denied_mod_is_refused_then_admitted_when_off() {
        use crate::net::client::Connection;
        let handle = flat(Config {
            mods_deny: vec!["pwc.dev-toolkit".into()],
            ..Config::default()
        });
        let port = handle.addr().port();
        let on = [("pwc.dev-toolkit".into(), "1.0.0".into())];
        let err = match Connection::connect_with("127.0.0.1", port, "ada", "", &on) {
            Ok(_) => panic!("a denied mod was admitted"),
            Err(err) => err,
        };
        assert_eq!(err.mods_denied, vec!["pwc.dev-toolkit".to_string()]);
        let retry = Connection::connect_with("127.0.0.1", port, "ada", "", &[]).expect("retry with the mod off");
        assert!(retry.is_alive());
        handle.stop();
    }

    #[test]
    fn no_mod_restriction_admits_everyone() {
        use crate::net::client::Connection;
        let handle = flat(Config::default());
        let port = handle.addr().port();
        let mods = [("pwc.dev-toolkit".into(), "1.0.0".into()), ("pwc.hotbar".into(), "0.1.0".into())];
        let conn = Connection::connect_with("127.0.0.1", port, "ada", "", &mods).expect("default is open");
        assert!(conn.is_alive());
        handle.stop();
    }

    #[test]
    fn world_policy_unions_flags_with_side_files() {
        let dir = crate::save::store::test_temp_path("policy");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("ops.txt"), "cara\n").unwrap();
        std::fs::write(
            dir.join("mods.toml"),
            "allow = [\"pwc.hotbar\"]\ndeny = [\"pwc.dev-toolkit\"]\n",
        ).unwrap();
        let mut config = Config {
            world: Some(dir.join("world.save")),
            ops: vec!["Ada".into()],
            mods_deny: vec!["pwc.other".into()],
            ..Config::default()
        };
        load_world_policy(&mut config).unwrap();
        assert!(config.ops.iter().any(|n| n.eq_ignore_ascii_case("ada")));
        assert!(config.ops.iter().any(|n| n.eq_ignore_ascii_case("cara")));
        assert_eq!(config.mods_allow, vec!["pwc.hotbar".to_string()]);
        assert!(config.mods_deny.iter().any(|id| id == "pwc.dev-toolkit"));
        assert!(config.mods_deny.iter().any(|id| id == "pwc.other"));

        let bad = crate::save::store::test_temp_path("policy-bad");
        std::fs::create_dir_all(&bad).unwrap();
        std::fs::write(bad.join("mods.toml"), "allow = 1\n").unwrap();
        let mut broken = Config { world: Some(bad.join("world.save")), ..Config::default() };
        assert!(load_world_policy(&mut broken).is_err());
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&bad);
    }

    #[test]
    fn each_message_kind_has_its_own_budget() {
        let now = Instant::now();
        let mut budgets = KindBudget::new();
        let chat = ClientMessage::Chat { channel: 0, text: "hi".into() };
        for _ in 0..CHAT_RATE {
            assert!(matches!(charge(&mut budgets, &chat, now), Charge::Pass));
        }
        assert!(matches!(charge(&mut budgets, &chat, now), Charge::Drop));
        let edit = ClientMessage::Edit { req: 1, x: 0, y: 0, z: 0, expect: 0, spec: "air".into() };
        assert!(matches!(charge(&mut budgets, &edit, now), Charge::Pass), "chat does not spend edits");
        for _ in 1..EDIT_RATE {
            assert!(matches!(charge(&mut budgets, &edit, now), Charge::Pass));
        }
        assert!(matches!(charge(&mut budgets, &edit, now), Charge::Answer), "an over-budget edit is still answered");

        let swing = ClientMessage::Swing;
        for _ in 0..SWING_RATE {
            assert!(matches!(charge(&mut budgets, &swing, now), Charge::Pass));
        }
        assert!(matches!(charge(&mut budgets, &swing, now), Charge::Drop));

        let set_time = ClientMessage::SetTime { day: 0.2 };
        assert!(matches!(charge(&mut budgets, &set_time, now), Charge::Pass));
        assert!(matches!(charge(&mut budgets, &set_time, now), Charge::Drop));

        let ping = ClientMessage::Ping { nonce: 1 };
        for _ in 0..PING_RATE {
            assert!(matches!(charge(&mut budgets, &ping, now), Charge::Pass));
        }
        assert!(matches!(charge(&mut budgets, &ping, now), Charge::Drop));

        let teleport = ClientMessage::Teleport { pos: DVec3::ZERO };
        for _ in 0..TELEPORT_RATE {
            assert!(matches!(charge(&mut budgets, &teleport, now), Charge::Pass));
        }
        assert!(matches!(charge(&mut budgets, &teleport, now), Charge::Answer));

        let hop = ClientMessage::Move {
            pos: DVec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            frame: DQuat::IDENTITY,
            velocity: Vec3::ZERO,
            up: Face::PosY,
            stance: Stance::Standing,
        };
        for _ in 0..MOVE_RATE {
            assert!(matches!(charge(&mut budgets, &hop, now), Charge::Pass));
        }
        assert!(matches!(charge(&mut budgets, &hop, now), Charge::Drop));

        let cruise = ClientMessage::Cruise { speed: 1.0 };
        for _ in 0..8 {
            assert!(matches!(charge(&mut budgets, &cruise, now), Charge::Pass), "cruise spends no token");
        }
    }

    #[test]
    fn backlog_drops_superseded_moves_and_aged_frames() {
        let (out, _rx) = sync_channel::<Arc<[u8]>>(4);
        let mut player = test_player(DVec3::ZERO, out, test_kick());
        player.ready = false;
        let now = Instant::now();
        let frame = |id, x| {
            Arc::<[u8]>::from(
                ServerMessage::PeerMove {
                    id,
                    pos: DVec3::new(x, 1.0, 0.0),
                    yaw: 0.0,
                    pitch: 0.0,
                    frame: DQuat::IDENTITY,
                    velocity: Vec3::ZERO,
                    up: Face::PosY,
                    stance: Stance::Standing,
                }
                .encode(),
            )
        };
        assert!(enqueue_backlog(&mut player, frame(7, 1.0), now));
        assert!(enqueue_backlog(&mut player, frame(7, 4.0), now));
        assert_eq!(player.backlog.len(), 1, "a newer move replaces the older one");
        assert_eq!(protocol::peer_move_id(&player.backlog[0].frame), Some(7));
        player.backlog[0].at = now - BACKLOG_AGE;
        let pong: Arc<[u8]> = ServerMessage::Pong { nonce: 1 }.encode().into();
        assert!(enqueue_backlog(&mut player, pong, now));
        assert_eq!(player.backlog.len(), 1, "a frame older than the age bound is dropped");
        assert!(protocol::peer_move_id(&player.backlog[0].frame).is_none());
        assert!(!player.kicked.load(Ordering::Relaxed));
    }

    #[test]
    fn send_blocking_honours_a_kick_and_a_deadline() {
        let kicked = AtomicBool::new(true);
        let (tx, _rx) = sync_channel::<Arc<[u8]>>(1);
        let started = Instant::now();
        assert!(!send_until(&tx, &kicked, Arc::from([0u8].as_slice()), Instant::now() + SEND_DEADLINE));
        assert!(started.elapsed() < Duration::from_millis(50), "a kick returns at once");

        let kicked = AtomicBool::new(false);
        let (tx, rx) = sync_channel::<Arc<[u8]>>(1);
        tx.try_send(Arc::from([0u8].as_slice())).unwrap();
        let started = Instant::now();
        assert!(!send_until(
            &tx,
            &kicked,
            Arc::from([1u8].as_slice()),
            Instant::now() + Duration::from_millis(30),
        ));
        assert!(kicked.load(Ordering::Relaxed), "a missed deadline is a kick");
        assert!(started.elapsed() < Duration::from_millis(500), "the wait is the deadline, not unbounded");
        let _ = rx;
    }

    #[test]
    fn swing_reaches_only_visible_players() {
        let (out_a, rx_a) = sync_channel::<Arc<[u8]>>(4);
        let (out_b, rx_b) = sync_channel::<Arc<[u8]>>(4);
        let (out_c, rx_c) = sync_channel::<Arc<[u8]>>(4);
        let mut swinger = test_player(DVec3::new(0.5, 20.0, 0.5), out_a, test_kick());
        swinger.visible.insert(2);
        swinger.visible.insert(3);
        let seen = test_player(DVec3::new(1.5, 20.0, 0.5), out_b, test_kick());
        let mut hidden = test_player(DVec3::new(2.5, 20.0, 0.5), out_c, test_kick());
        hidden.ready = false;
        let mut players = HashMap::new();
        players.insert(1, swinger);
        players.insert(2, seen);
        players.insert(3, hidden);
        let shared = Arc::new(Mutex::new(test_state(players)));
        relay_swing(&shared, 1);
        assert!(rx_a.try_recv().is_err(), "the swinger does not hear their own swing");
        let frame = rx_b.try_recv().expect("a player who can see the swinger hears it");
        assert!(protocol::is_peer_swing(&frame));
        assert!(rx_c.try_recv().is_err(), "a peer who is not ready is not in the audience");
    }

    #[test]
    fn minute_clock_broadcast_reaches_a_ready_player() {
        assert_eq!(TIME_BROADCAST, Duration::from_secs(60));
        let (out, rx) = sync_channel::<Arc<[u8]>>(4);
        let mut players = HashMap::new();
        players.insert(1, test_player(DVec3::new(0.5, 20.0, 0.5), out, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));
        broadcast_clock(&shared, lax_ctx());
        assert!(drain(&rx).iter().any(|msg| matches!(msg, ServerMessage::Time { .. })));
    }

    #[test]
    fn a_panicking_reaction_tick_keeps_the_server_running() {
        use crate::net::client::Connection;
        let handle = flat(Config::default());
        {
            let mut state = handle.state.lock_recover();
            state.panic_tick = true;
            state.reactions.wake_cell((1, 2, 3));
            assert_eq!(state.reactions.pending(), 6);
        }
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut restored = false;
        while Instant::now() < deadline {
            let state = handle.state.lock_recover();
            if !state.panic_tick && state.reactions.pending() == 6 {
                restored = true;
                break;
            }
            drop(state);
            thread::sleep(Duration::from_millis(5));
        }
        assert!(restored, "a panicking tick must put the scheduler back with its contacts");
        thread::sleep(Duration::from_millis(200));
        assert_eq!(
            handle.state.lock_recover().reactions.pending(),
            0,
            "the following tick still runs"
        );
        let conn = Connection::connect("127.0.0.1", handle.addr().port(), "ada", "").expect("still accepting");
        assert!(conn.is_alive());
        drop(conn);
        handle.stop();
    }

    fn flat_shared(
        players: HashMap<u32, PlayerHandle>,
        noclip: NoclipPolicy,
        ops: &[&str],
    ) -> (Arc<Mutex<State>>, Ctx) {
        let mut registry = BlockRegistry::with_builtins();
        let generator: crate::world::terrain::Generator =
            Arc::new(crate::world::generation::FlatTerrain::new(&mut registry, 1));
        let state = State {
            edits: HashMap::new(),
            spec_pool: HashMap::new(),
            registry,
            players,
            grid: HashMap::new(),
            next_id: 2,
            day: 0.3,
            day_set: Instant::now(),
            reactions: ReactionScheduler::new(),
            max_speed: crate::player::MAX_SPEED,
            panic_tick: false,
        };
        let ctx = Ctx {
            password: String::new(),
            seed: 1,
            content: crate::net::content_id(&BlockRegistry::with_builtins()),
            day_secs: 600.0,
            teleport: TeleportPolicy::All,
            noclip,
            worldgen: WorldgenKind::Flat,
            terrain: TerrainCfg::default(),
            seams: Seams::new(generator.atlases().to_vec()),
            generator,
            hooks: None,
            ops: ops.iter().map(|name| (*name).to_ascii_lowercase()).collect(),
            op_secrets: Vec::new(),
            mods_allow: Vec::new(),
            mods_deny: Vec::new(),
            store: None,
        };
        (Arc::new(Mutex::new(state)), ctx)
    }

    fn pose(pos: DVec3) -> (HashMap<u32, PlayerHandle>, std::sync::mpsc::Receiver<Arc<[u8]>>) {
        let (out, rx) = sync_channel::<Arc<[u8]>>(8);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(pos, out, test_kick()));
        (players, rx)
    }

    #[test]
    fn noclip_snaps_a_body_in_solid_ground() {
        let start = DVec3::new(0.5, 20.0, 0.5);
        let surface = DVec3::new(
            0.5,
            crate::world::generation::FLAT_HEIGHT as f64 + crate::player::Stance::Standing.eye_offset(),
            0.5,
        );
        let buried = DVec3::new(0.5, crate::world::generation::FLAT_HEIGHT as f64, 0.5);
        let step = |shared: &Arc<Mutex<State>>, ctx: &Ctx, pos: DVec3| {
            on_move(shared, ctx, 1, pos, 0.0, 0.0, DQuat::IDENTITY, Vec3::ZERO, Face::PosY, Stance::Standing);
        };

        let (players, _rx) = pose(start);
        let (shared, ctx) = flat_shared(players, NoclipPolicy::Off, &[]);
        step(&shared, &ctx, surface);
        assert_eq!(shared.lock_recover().players[&1].pos, surface, "standing on the flat surface is clear");

        let (players, _rx) = pose(start);
        let (shared, ctx) = flat_shared(players, NoclipPolicy::Off, &[]);
        step(&shared, &ctx, buried);
        assert_eq!(shared.lock_recover().players[&1].pos, start, "a body in solid ground snaps back");

        let (players, _rx) = pose(start);
        let (shared, ctx) = flat_shared(players, NoclipPolicy::All, &[]);
        step(&shared, &ctx, buried);
        assert_eq!(shared.lock_recover().players[&1].pos, buried, "noclip all accepts the buried pose");

        let (players, _rx) = pose(start);
        let (shared, ctx) = flat_shared(players, NoclipPolicy::Ops, &["p"]);
        step(&shared, &ctx, buried);
        assert_eq!(shared.lock_recover().players[&1].pos, buried, "an operator may pass");

        let (players, _rx) = pose(start);
        let (shared, ctx) = flat_shared(players, NoclipPolicy::Ops, &["p"]);
        shared.lock_recover().players.get_mut(&1).unwrap().name = "guest".into();
        step(&shared, &ctx, buried);
        assert_eq!(shared.lock_recover().players[&1].pos, start, "a guest under ops snaps back");

        let (players, _rx) = pose(start);
        let (shared, ctx) = flat_shared(players, NoclipPolicy::Off, &[]);
        step(&shared, &ctx, start);
        let into = DVec3::new(start.x + 1.0, start.y, start.z);
        let mut before = [(0i32, 0, 0); BODY_CELL_CAP];
        let mut after = [(0i32, 0, 0); BODY_CELL_CAP];
        let n0 = fill_body_cells(start, Stance::Standing, Face::PosY, &mut before).unwrap();
        let n1 = fill_body_cells(into, Stance::Standing, Face::PosY, &mut after).unwrap();
        let fresh = after[..n1]
            .iter()
            .copied()
            .find(|cell| !before[..n0].contains(cell))
            .expect("the step enters a new cell");
        {
            let mut state = shared.lock_recover();
            let spec = rock_spec();
            let id = state.registry.parse_spec(&spec).unwrap();
            assert!(state.registry.is_solid(id));
            let canonical = state.registry.spec(id);
            let shared_spec = state.intern(&canonical).unwrap();
            state.edits.insert(fresh, Cell { spec: shared_spec, rev: 1, natural: false });
        }
        step(&shared, &ctx, into);
        assert_eq!(shared.lock_recover().players[&1].pos, start, "a solid edit the body newly enters snaps back");
    }

    /// A body standing on the start world across a chart's edge reads the cells past the edge
    /// through the seam, as the client's collision does: standing on the surface there is clear,
    /// and the same body sunk into the ground is blocked.
    #[test]
    fn noclip_reads_a_chart_edge_like_the_client() {
        use crate::space::atlas::Patch;
        let mut registry = BlockRegistry::with_builtins();
        let generator = crate::world::terrain::generator(&mut registry, 4242, TerrainCfg::default());
        let seams = Seams::new(generator.atlases().to_vec());
        let state = State {
            edits: HashMap::new(),
            spec_pool: HashMap::new(),
            registry,
            players: HashMap::new(),
            grid: HashMap::new(),
            next_id: 2,
            day: 0.3,
            day_set: Instant::now(),
            reactions: ReactionScheduler::new(),
            max_speed: crate::player::MAX_SPEED,
            panic_tick: false,
        };
        let home = generator.cosmos().expect("cosmos").home();
        let atlas = generator
            .atlases()
            .iter()
            .find(|a| (a.centre - home.centre_f()).length() < 1.0)
            .expect("the start world is charted");
        let patch = Patch::Shell { band: 0, face: Face::PosY };
        let (o, size) = atlas.storage_box(patch);
        let edge = (o[0] + size[0] - 1) as i32;
        let top = |x: i32, z: i32| {
            let g = seams.glue_cell(BlockCoord::new(x, 0, z)).map_or((x, z), |g| (g.x, g.z));
            generator.height(g.0, g.1)
        };
        let eye = crate::player::Stance::Standing.eye_offset();
        let mut checked = 0;
        for k in 0..64 {
            let z = (o[2] + size[2] / 2) as i32 + k * 7;
            let (inside, outside) = (generator.height(edge, z), top(edge + 1, z));
            if inside == i32::MIN || outside == i32::MIN || outside > inside {
                continue;
            }
            // Feet on the inside column's top, the body reaching past the box edge.
            let pos = DVec3::new(f64::from(edge) + 0.9, f64::from(inside) + eye, f64::from(z) + 0.5);
            assert!(
                !body_blocked(&state, &generator, &seams, pos, pos, Stance::Standing, Face::PosY, &[]),
                "standing across the edge at z {z} (inside top {inside}, glued top {outside}) was blocked"
            );
            let sunk = pos - DVec3::Y * 1.5;
            assert!(
                body_blocked(&state, &generator, &seams, sunk, sunk, Stance::Standing, Face::PosY, &[]),
                "a body sunk into the edge at z {z} was clear"
            );
            checked += 1;
        }
        assert!(checked >= 8, "only {checked} edge columns were level enough to stand across");
    }

    #[test]
    fn noclip_check_stays_cheap() {
        let mut registry = BlockRegistry::with_builtins();
        let generator = crate::world::terrain::generator(&mut registry, 1, TerrainCfg::default());
        let seams = Seams::new(generator.atlases().to_vec());
        let state = State {
            edits: HashMap::new(),
            spec_pool: HashMap::new(),
            registry,
            players: HashMap::new(),
            grid: HashMap::new(),
            next_id: 2,
            day: 0.3,
            day_set: Instant::now(),
            reactions: ReactionScheduler::new(),
            max_speed: crate::player::MAX_SPEED,
            panic_tick: false,
        };
        let mut pos = DVec3::new(8.5, 80.0, 8.5);
        for _ in 0..40 {
            if !body_blocked(&state, &generator, &seams, pos, pos, Stance::Standing, Face::PosY, &[]) {
                break;
            }
            pos.y += 16.0;
        }
        let mut cells = [(0i32, 0, 0); BODY_CELL_CAP];
        let mut occupied = Vec::new();
        let n = fill_body_cells(pos, Stance::Standing, Face::PosY, &mut cells).expect("body fits");
        occupied.extend_from_slice(&cells[..n]);
        for _ in 0..8 {
            pos.x += 0.5;
            let _ = body_blocked(&state, &generator, &seams, pos, pos, Stance::Standing, Face::PosY, &occupied);
            let n = fill_body_cells(pos, Stance::Standing, Face::PosY, &mut cells).expect("body fits");
            occupied.clear();
            occupied.extend_from_slice(&cells[..n]);
        }
        let steps = 64u32;
        let started = Instant::now();
        for _ in 0..steps {
            let next = DVec3::new(pos.x + 0.5, pos.y, pos.z);
            let blocked = std::hint::black_box(body_blocked(
                &state,
                &generator,
                &seams,
                pos,
                next,
                Stance::Standing,
                Face::PosY,
                &occupied,
            ));
            let _ = blocked;
            let n = fill_body_cells(next, Stance::Standing, Face::PosY, &mut cells).expect("body fits");
            occupied.clear();
            occupied.extend_from_slice(&cells[..n]);
            pos = next;
        }
        let us = started.elapsed().as_secs_f64() * 1.0e6 / f64::from(steps);
        let load_ms = 32.0 * 20.0 * us / 1000.0;
        eprintln!("noclip: {us:.2} µs/move; 32 players at 20 Hz = {load_ms:.2} ms/s");
        let stand = Instant::now();
        for _ in 0..steps {
            let blocked = std::hint::black_box(body_blocked(
                &state,
                &generator,
                &seams,
                pos,
                pos,
                Stance::Standing,
                Face::PosY,
                &occupied,
            ));
            assert!(!blocked, "a body that has not entered a new cell is not retested");
        }
        let stand_us = stand.elapsed().as_secs_f64() * 1.0e6 / f64::from(steps);
        eprintln!("noclip standing: {stand_us:.2} µs/move");
        assert!(us < 1000.0, "newly entered cells took {us:.2} µs/move");
        // Long sweeps, straight up from the spawn so the whole path is walked.
        let spawn = generator.chart_spawn().expect("a charted start world");
        let n = fill_body_cells(spawn, Stance::Standing, Face::PosY, &mut cells).expect("body fits");
        let mut worst = 0.0f64;
        for length in [8.0, 64.0, SWEEP_LIMIT] {
            let to = spawn + DVec3::Y * length;
            let started = Instant::now();
            let blocked = std::hint::black_box(body_blocked(
                &state,
                &generator,
                &seams,
                spawn,
                to,
                Stance::Standing,
                Face::PosY,
                &cells[..n],
            ));
            let sweep_us = started.elapsed().as_secs_f64() * 1.0e6;
            assert!(!blocked, "the sky above the spawn is clear");
            eprintln!("noclip sweep of {length} blocks: {sweep_us:.1} µs ({:.2} µs/block)", sweep_us / length);
            worst = worst.max(sweep_us);
        }
        assert!(worst < 200_000.0, "the longest sweep took {worst:.1} µs");
    }

    #[test]
    fn save_world_exports_the_live_scheduler() {
        let path = crate::save::store::test_temp_path("g25-pending");
        let _ = std::fs::remove_file(&path);
        let flags = persist::Flags {
            seed: 7,
            worldgen: WorldgenKind::Flat,
            terrain: TerrainCfg::default(),
            warn: false,
        };
        let loaded = persist::load(&path, &flags).unwrap();
        let (shared, mut ctx) = flat_shared(HashMap::new(), NoclipPolicy::All, &[]);
        ctx.store = Some(Arc::new(loaded.store.expect("a new file has a store")));
        ctx.seed = 7;
        shared.lock_recover().reactions.wake_cell((4, 5, 6));
        assert_eq!(shared.lock_recover().reactions.pending(), 6);
        save_world(&*shared, &ctx, &Mutex::new(()));
        let again = persist::load(&path, &flags).unwrap();
        assert_eq!(again.pending.len(), 6);
        let mut sched = ReactionScheduler::new();
        sched.restore(&contacts_of(&again.pending));
        assert_eq!(sched.pending(), 6);
        drop(ctx);
        let _ = std::fs::remove_file(&path);
        let mut bak = path.file_name().unwrap().to_os_string();
        bak.push(".bak");
        let _ = std::fs::remove_file(path.with_file_name(bak));
    }

    /// Cycling channel names cannot beat the connection's total, and new names stop at the cap.
    #[test]
    fn mod_channels_share_one_budget_and_a_name_cap() {
        let now = Instant::now();
        let channel = |i: usize| protocol::Channel::parse(&format!("c{i}")).unwrap();
        let mut names = ChannelBudget::new();
        for i in 0..MAX_CHANNELS {
            assert!(names.allow(&channel(i), now));
        }
        assert!(!names.allow(&channel(MAX_CHANNELS), now), "a new name past the cap is dropped");
        assert!(names.allow(&channel(0), now), "a known name keeps its window");
        let mut flood = ChannelBudget::new();
        let passed = (0..4 * MOD_DATA_RATE as usize).filter(|&i| flood.allow(&channel(i % MAX_CHANNELS), now)).count();
        assert_eq!(passed, MOD_DATA_RATE as usize, "switching channels buys no extra rate");
    }

    /// Under a speed cap, a move split into many messages covers no more than one burst plus
    /// the cap over the time taken. A fast stream that stalls and lands at once still passes.
    #[test]
    fn split_moves_gain_nothing_over_the_speed_cap() {
        let cap = 30.0 * crate::math::PER_METER;
        let (out, _rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let start = DVec3::new(8.5, 40.0, 8.5);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(start, out, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));
        shared.lock_recover().max_speed = cap;
        let anchored = Instant::now();
        shared.lock_recover().players.get_mut(&1).unwrap().last_move = anchored;
        for _ in 0..40 {
            let at = shared.lock_recover().players[&1].pos;
            let to = DVec3::new(at.x + 5.0, at.y, at.z);
            on_move(&shared, lax_ctx(), 1, to, 0.0, 0.0, DQuat::IDENTITY, Vec3::new(cap as f32, 0.0, 0.0), Face::PosY, Stance::Standing);
        }
        let covered = shared.lock_recover().players[&1].pos.x - start.x;
        let bound = MOVE_FLOOR + cap * anchored.elapsed().as_secs_f64();
        assert!(covered <= bound + 1e-9, "40 split moves covered {covered} blocks, the bound is {bound}");
        assert!(covered >= MOVE_FLOOR - 5.0, "the burst is spendable, covered {covered}");

        let fast = 1000.0 * crate::math::PER_METER;
        let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let mut players = HashMap::new();
        players.insert(1u32, test_player(start, out, test_kick()));
        let shared = Arc::new(Mutex::new(test_state(players)));
        shared.lock_recover().players.get_mut(&1).unwrap().last_move = Instant::now() - Duration::from_millis(133);
        let step = fast * 0.033;
        for i in 1..=4 {
            let to = DVec3::new(start.x + step * f64::from(i), start.y, start.z);
            on_move(&shared, lax_ctx(), 1, to, 0.0, 0.0, DQuat::IDENTITY, Vec3::new(fast as f32, 0.0, 0.0), Face::PosY, Stance::Standing);
            assert_eq!(shared.lock_recover().players[&1].pos, to, "stalled move {i} at 1 km/s");
        }
        assert!(rx.try_recv().is_err(), "no snap-back");
    }

    /// With noclip refused, a move whose destination is open but whose path crosses a wall
    /// snaps back. A body teleported into ground can still walk out, a path longer than the
    /// sweep fails closed, and a cruise is judged at its destination.
    #[test]
    fn a_move_cannot_pass_through_a_wall() {
        let eye = crate::player::Stance::Standing.eye_offset();
        let ground = crate::world::generation::FLAT_HEIGHT;
        let surface = DVec3::new(0.5, f64::from(ground) + eye, 0.5);
        let beyond = DVec3::new(6.5, surface.y, 0.5);
        let step = |shared: &Arc<Mutex<State>>, ctx: &Ctx, pos: DVec3, speed: f32| {
            on_move(shared, ctx, 1, pos, 0.0, 0.0, DQuat::IDENTITY, Vec3::new(speed, 0.0, 0.0), Face::PosY, Stance::Standing);
        };

        let (players, _rx) = pose(surface);
        let (shared, ctx) = flat_shared(players, NoclipPolicy::Off, &[]);
        step(&shared, &ctx, beyond, 0.0);
        assert_eq!(shared.lock_recover().players[&1].pos, beyond, "an open path is clear");

        let (players, _rx) = pose(surface);
        let (shared, ctx) = flat_shared(players, NoclipPolicy::Off, &[]);
        {
            let mut state = shared.lock_recover();
            let rock = state.registry.parse_spec(&rock_spec()).unwrap();
            let canonical = state.registry.spec(rock);
            for x in 2..=4 {
                for y in ground..ground + 3 {
                    for z in -1..=1 {
                        let spec = state.intern(&canonical).unwrap();
                        state.edits.insert((x, y, z), Cell { spec, rev: 1, natural: false });
                    }
                }
            }
        }
        step(&shared, &ctx, beyond, 0.0);
        assert_eq!(shared.lock_recover().players[&1].pos, surface, "a three-block wall is not crossed");

        let buried = DVec3::new(0.5, f64::from(ground), 0.5);
        let (players, _rx) = pose(surface);
        let (shared, ctx) = flat_shared(players, NoclipPolicy::Off, &[]);
        on_teleport(&shared, &ctx, 1, buried);
        step(&shared, &ctx, surface, 0.0);
        assert_eq!(shared.lock_recover().players[&1].pos, surface, "a body teleported into ground walks out");

        let sky = DVec3::new(surface.x, surface.y + SWEEP_LIMIT + 10.0, surface.z);
        let fast = crate::player::MAX_SPEED as f32;
        let (players, _rx) = pose(surface);
        let (shared, ctx) = flat_shared(players, NoclipPolicy::Off, &[]);
        step(&shared, &ctx, sky, fast);
        assert_eq!(shared.lock_recover().players[&1].pos, surface, "a path past the sweep fails closed");
        on_cruise(&shared, 1, crate::player::CRUISE_MAX);
        step(&shared, &ctx, sky, fast);
        assert_eq!(shared.lock_recover().players[&1].pos, sky, "a cruise is judged at its destination");
    }

    /// The final save waits until every connection has been told to go, so no edit can be
    /// acknowledged after the save has read the ledger.
    #[test]
    fn stop_closes_connections_before_the_final_save() {
        use crate::net::client::{Connection, Incoming};
        let path = crate::save::store::test_temp_path("stop-order");
        let handle = flat(Config { world: Some(path.clone()), ..Config::default() });
        let mut conn = Connection::connect("127.0.0.1", handle.addr().port(), "ada", "").unwrap();
        let gate = handle.save_gate.lock_recover();
        thread::scope(|scope| {
            let stopping = scope.spawn(|| handle.stop());
            let deadline = Instant::now() + Duration::from_secs(3);
            let mut closed = false;
            while !closed && Instant::now() < deadline {
                closed = conn.poll().into_iter().any(|e| matches!(e, Incoming::Disconnected { .. }));
                if !closed {
                    thread::sleep(Duration::from_millis(20));
                }
            }
            assert!(!path.exists(), "the save is still waiting");
            drop(gate);
            stopping.join().unwrap();
            assert!(closed, "connections close before the final save");
        });
        assert!(path.exists(), "stop still saves");
        let _ = std::fs::remove_file(&path);
    }

    /// Chat until a line containing `want` arrives, or three seconds pass.
    fn chat_until(conn: &mut crate::net::client::Connection, want: &str) -> Vec<String> {
        let mut texts = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline && !texts.iter().any(|t: &String| t.contains(want)) {
            for event in conn.poll() {
                if let crate::net::client::Incoming::Chat { text, .. } = event {
                    texts.push(text.to_string());
                }
            }
            thread::sleep(Duration::from_millis(10));
        }
        texts
    }

    /// An `ops.txt` secret beats the bare name: the player is an operator only after `/op` with
    /// that secret. The line is answered privately and reaches nobody else.
    #[test]
    fn an_operator_secret_is_proved_with_op_and_never_relayed() {
        use crate::net::client::Connection;
        let handle = flat(Config {
            teleport: TeleportPolicy::Ops,
            ops: vec!["ada".into()],
            op_secrets: vec![("Ada".into(), "s3cret".into())],
            ..Config::default()
        });
        let port = handle.addr().port();
        let mut ada = Connection::connect("127.0.0.1", port, "ada", "").unwrap();
        let mut bob = Connection::connect("127.0.0.1", port, "bob", "").unwrap();
        let far = DVec3::new(300.5, 30.0, 300.5);
        ada.send_teleport(far);
        assert!(chat_until(&mut ada, "only an operator").iter().any(|t| t == "only an operator can teleport"));
        ada.send_chat(chat::GLOBAL, "/op wrong");
        assert!(chat_until(&mut ada, "refused").iter().any(|t| t == "operator secret refused"));
        ada.send_chat(chat::GLOBAL, "/op s3cret");
        assert!(chat_until(&mut ada, "operator").iter().any(|t| t == "you are now an operator"));
        ada.send_teleport(far);
        let deadline = Instant::now() + Duration::from_secs(3);
        let mut moved = false;
        while !moved && Instant::now() < deadline {
            moved = handle.state.lock_recover().players.values().any(|h| &*h.name == "ada" && h.pos == far);
            thread::sleep(Duration::from_millis(10));
        }
        assert!(moved, "a proved operator may teleport");
        ada.send_chat(chat::GLOBAL, "hello");
        let heard = chat_until(&mut bob, "hello");
        assert!(heard.iter().any(|t| t == "hello"), "ordinary chat still flows: {heard:?}");
        assert!(!heard.iter().any(|t| t.contains("/op") || t.contains("s3cret") || t.contains("wrong")), "{heard:?}");
        assert_eq!(op_secret("/op  s3cret "), Some(Arc::from("s3cret")));
        assert_eq!(op_secret("/opera"), None);
        handle.stop();
    }

    /// One full frame plus another fits the join backlog, and an essential frame that waits
    /// past the age bound kicks the joiner instead of vanishing.
    #[test]
    fn backlog_holds_a_full_frame_and_kicks_on_an_aged_essential_one() {
        let (out, _rx) = sync_channel::<Arc<[u8]>>(4);
        let mut player = test_player(DVec3::ZERO, out, test_kick());
        player.ready = false;
        let now = Instant::now();
        let n = (MAX_FRAME - SNAPSHOT_HEAD) / (SNAPSHOT_EDIT_FIXED + 3);
        let air: Arc<str> = "air".into();
        let full: Arc<[u8]> =
            ServerMessage::Snapshot { edits: (0..n as i32).map(|i| (i, 0, 0, 1, air.clone())).collect() }.encode().into();
        assert!(full.len() > MAX_FRAME - SNAPSHOT_EDIT_FIXED - 3 && full.len() <= MAX_FRAME);
        let edit: Arc<[u8]> = ServerMessage::Edit { x: 1, y: 2, z: 3, rev: 1, spec: air.clone() }.encode().into();
        assert!(enqueue_backlog(&mut player, full, now));
        assert!(enqueue_backlog(&mut player, edit.clone(), now), "a full frame and one more fit");
        assert!(!enqueue_backlog(&mut player, edit, now + BACKLOG_AGE), "an aged edit kicks");
        assert_eq!(player.backlog.len(), 2, "no essential frame was dropped");
    }

    #[test]
    fn edits_past_the_spec_pool_stay_in_the_file() {
        let mut state = test_state(HashMap::new());
        for i in 0..MAX_SPEC_POOL {
            state.intern(&format!("spec-{i}")).unwrap();
        }
        let rock = rock_spec();
        let kept = install_edits(&mut state, &[(1, 2, 3, rock.clone()), (4, 5, 6, "air".into())]);
        assert_eq!(kept, vec![(1, 2, 3, rock), (4, 5, 6, "air".to_string())]);
        assert!(state.edits.is_empty());
    }

    /// An edit that puts back the generated block keeps its revision in memory but is not saved.
    #[test]
    fn a_no_op_edit_stays_out_of_the_world_file() {
        let path = crate::save::store::test_temp_path("no-op");
        let flags = persist::Flags { seed: 1, worldgen: WorldgenKind::Flat, terrain: TerrainCfg::default(), warn: false };
        let ground = crate::world::generation::FLAT_HEIGHT;
        let (players, _rx) = pose(DVec3::new(0.5, f64::from(ground) + 2.0, 0.5));
        let (shared, mut ctx) = flat_shared(players, NoclipPolicy::All, &[]);
        ctx.store = Some(Arc::new(persist::load(&path, &flags).unwrap().store.unwrap()));
        on_edit(&shared, None, &ctx.generator, 1, 1, 0, ground + 1, 0, 0, "air");
        on_edit(&shared, None, &ctx.generator, 1, 2, 1, ground - 1, 0, 0, "air");
        assert_eq!(shared.lock_recover().edits.len(), 2, "both edits hold a revision");
        save_world(&shared, &ctx, &Mutex::new(()));
        let again = persist::load(&path, &flags).unwrap();
        assert_eq!(again.edits, vec![(1, ground - 1, 0, "air".to_string())]);
        let _ = std::fs::remove_file(&path);
    }

    /// Initials from a source that never hears the server (a spoofed address) get a stateless
    /// retry and take no handshake slot; a real client still joins through the retry.
    #[test]
    fn unvalidated_initials_take_no_handshake_slot() {
        use crate::net::client::Connection;
        let handle = flat(Config::default());
        let server = SocketAddr::from((Ipv4Addr::LOCALHOST, handle.addr().port()));
        let relay = std::net::UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        relay.set_read_timeout(Some(Duration::from_millis(20))).unwrap();
        let relay_addr = relay.local_addr().unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let relay_stop = stop.clone();
        // Forward the client's packets to the server; drop every answer.
        let forward = thread::spawn(move || {
            let mut buf = [0u8; 2048];
            while !relay_stop.load(Ordering::Relaxed) {
                if let Ok((n, from)) = relay.recv_from(&mut buf)
                    && from != server
                {
                    let _ = relay.send_to(&buf[..n], server);
                }
            }
        });
        let (rt, ep) = client_endpoint();
        let attempts: Vec<_> = {
            let _g = rt.enter();
            (0..4).map(|_| ep.connect(relay_addr, "watt").unwrap()).collect()
        };
        rt.block_on(async { tokio::time::sleep(Duration::from_millis(400)).await });
        assert_eq!(handle.handshake_slots(), 0, "a source that cannot answer holds no slot");
        stop.store(true, Ordering::Relaxed);
        forward.join().unwrap();
        drop(attempts);
        Connection::connect("127.0.0.1", server.port(), "real", "").expect("a real client passes the retry");
        handle.stop();
    }

    #[test]
    fn console_text_escapes_control_characters() {
        assert_eq!(console_text("\u{1b}[2Jpwc.hotbar\u{7}"), "\\u{1b}[2Jpwc.hotbar\\u{7}");
        assert_eq!(console_text("plain ünïcode"), "plain ünïcode");
    }

    #[test]
    fn names_that_spell_server_are_reserved() {
        for name in ["server", "Server", "S.E.R.V.E.R", "<server>", "server>", " _server_ "] {
            assert!(reserved_name(name), "{name}");
        }
        for name in ["servers", "observer", "server2", "serve", "ada"] {
            assert!(!reserved_name(name), "{name}");
        }
    }
}
