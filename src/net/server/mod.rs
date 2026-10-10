//! Authoritative, headless multiplayer server: owns the seed + edit overlay
//! and the player roster; terrain is procedural, so no voxel data is ever sent.
//!
//! **Threading.** One accept thread; per client a blocking reader thread and a
//! bounded-queue writer thread, coordinated through a single [`Mutex`]-guarded
//! [`State`]. The lock is held only for short bursts — frames are queued in
//! order under it and the writers are woken after it is released, and poses
//! go out once per 20 Hz tick per recipient. Comfortably serves hundreds of
//! players; past that the one global lock and thread-per-client model are the
//! ceiling (join/leave and global chat stay O(roster)) — an event-loop rewrite
//! would be the next step.
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
//!
//! **Layout.** `state` holds what the lock guards (roster, ledger, interest grid,
//! terrain cache); `join` takes a connection from accept to the roster and back out;
//! `session` is its message loop with the chat and clock handlers; `movement`,
//! `edit` and `fanout` handle moves, edits and getting frames out; `budget` rates
//! each connection; `tick` is the 20 Hz thread; `save` is the world file. Tests sit
//! in `tests/` by the same subjects, and `load` is the bot load test.
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::io;
use std::net::SocketAddr;
#[cfg(test)]
use std::net::Ipv4Addr;
use std::path::PathBuf;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, TrySendError, sync_channel};
#[cfg(not(test))]
use std::sync::MutexGuard;
use std::sync::{Arc, LazyLock, Mutex, OnceLock, PoisonError};
use std::thread::{self, JoinHandle, Thread};
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
use crate::net::protocol::{self, ClientMessage, ModId, ModOffer, Pose, PoseBody, PosesWriter, ServerMessage, SnapshotWriter};
use crate::net::{MAX_CHAT, MAX_FRAME, MAX_NAME, MAX_SPEC, PROTOCOL_VERSION, chat, quic};
use crate::presence::Stance;
use crate::world::seam::Seams;
use crate::world::{FastMap, FastSet};
use crate::world::terrain::TerrainCfg;
use crate::world::generation::{TerrainGenerator, WorldgenKind};

mod budget;
mod edit;
mod fanout;
mod join;
mod movement;
mod save;
mod session;
mod state;
mod tick;

use budget::*;
use edit::*;
use fanout::*;
use join::*;
use movement::*;
use save::*;
use session::*;
use state::*;
use tick::*;

pub use save::load_world_policy;

/// A client thread that panics while holding the state must not take the whole
/// server down with it — [`State`] is plain data, valid at every point a panic
/// could interrupt, so recovery is always sound.
trait LockRecover<T> {
    #[cfg_attr(test, track_caller)]
    fn lock_recover(&self) -> Guard<'_, T>;
}

impl<T: 'static> LockRecover<T> for Mutex<T> {
    #[cfg_attr(test, track_caller)]
    fn lock_recover(&self) -> Guard<'_, T> {
        let guard = self.lock().unwrap_or_else(PoisonError::into_inner);
        #[cfg(test)]
        let guard = load::Timed::new(guard);
        guard
    }
}

#[cfg(not(test))]
type Guard<'a, T> = MutexGuard<'a, T>;
/// Test builds time every [`State`] lock hold for the load test.
#[cfg(test)]
type Guard<'a, T> = load::Timed<'a, T>;

#[cfg(test)]
mod load;

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
/// Ids the material table keeps for the world's own products (reactions, generation): a novel
/// spec a client sends is interned only while at least this many ids stay free, so one client
/// cannot fill the table (see [`take_novel_spec`]).
const CLIENT_INTERN_RESERVE: usize = crate::block::registry::MAX_BLOCK_TYPES / 4;
/// The table size past which client specs no longer intern: [`CLIENT_INTERN_RESERVE`] below the cap.
const CLIENT_INTERN_LIMIT: usize = crate::block::registry::MAX_BLOCK_TYPES - CLIENT_INTERN_RESERVE;
const INTEREST_RADIUS: f64 = 160.0 * crate::math::PER_METER;
/// Squared once so the hot per-listener check in [`on_move`] needs no sqrt.
const INTEREST_RADIUS_SQ: f64 = INTEREST_RADIUS * INTEREST_RADIUS;
// Every visible peer's pose fits the offset range of a [`ServerMessage::PeerPoses`] frame.
const _: () = assert!(INTEREST_RADIUS < protocol::POSE_REACH);
/// Peers within this distance get each moved pose; farther ones every [`FAR_EVERY`]th tick.
const NEAR: f64 = 48.0 * crate::math::PER_METER;
const NEAR_SQ: f64 = NEAR * NEAR;
const FAR_EVERY: u64 = 4;
// One tick's frame holds a pose for every player.
const _: () = assert!(protocol::POSES_HEAD + MAX_PLAYERS * protocol::POSE_MAX <= MAX_FRAME);
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
/// Who may do something only some players should: nobody, operators, or everyone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    Off,
    Ops,
    All,
}

impl Policy {
    /// `off`, `ops` or `all`, as the dedicated server's flags spell them.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "off" => Some(Self::Off),
            "ops" => Some(Self::Ops),
            "all" => Some(Self::All),
            _ => None,
        }
    }

    /// The spelling [`parse`](Self::parse) reads.
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Ops => "ops",
            Self::All => "all",
        }
    }

    /// Whether a player, an operator or not, may.
    pub fn allows(self, operator: bool) -> bool {
        match self {
            Self::Off => false,
            Self::Ops => operator,
            Self::All => true,
        }
    }
}

/// Who may teleport. `All` is the integrated host and [`Config::default`], so
/// existing sessions keep today's behaviour. A dedicated server passes [`Ops`](Policy::Ops).
pub type TeleportPolicy = Policy;

/// Who may pass through solid ground. [`Config::default`] is [`All`](Policy::All) so existing
/// sessions are not suddenly collision-checked. A dedicated server passes [`Ops`](Policy::Ops).
pub type NoclipPolicy = Policy;

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

impl Ctx {
    /// The rules `config` sets for a world `generator` builds, whose content identity is `content`.
    fn new(config: Config, generator: crate::world::terrain::Generator, content: crate::net::ContentId, store: Option<Arc<Store>>) -> Self {
        let hooks = if config.hooks.is_empty() { None } else { Some(Mutex::new(hooks::Table::new(config.hooks))) };
        let op_secrets: Vec<(String, String)> =
            config.op_secrets.iter().map(|(name, secret)| (canonical_name(name), secret.clone())).collect();
        Self {
            password: config.password,
            seed: config.seed,
            content,
            day_secs: clamp_day_secs(config.day_secs),
            teleport: config.teleport,
            noclip: config.noclip,
            worldgen: config.worldgen,
            terrain: config.terrain,
            seams: Seams::new(generator.atlases().to_vec()),
            generator,
            hooks,
            ops: canonical_ops(&config.ops, &op_secrets),
            op_secrets,
            mods_allow: config.mods_allow,
            mods_deny: config.mods_deny,
            store,
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
    let mut state = State::new(registry, loaded.day, finite_speed(config.max_speed));
    let kept = install_edits(&mut state, &loaded.edits);
    state.reactions.restore(&contacts_of(&loaded.pending));
    let content = crate::net::content_id(&state.registry);
    let shared = Arc::new(Mutex::new(state));
    debug_assert_ne!(WORLD_PLAYER, 1, "player ids start at 1; 0 is the world");

    let autosave_every = config.autosave_every;
    // The loaded world's seed and generator replace the flags'.
    let config = Config { seed: loaded.seed, worldgen: loaded.worldgen, terrain: loaded.terrain, ..config };
    let ctx = Arc::new(Ctx::new(config, generator, content, loaded.store.map(Arc::new)));
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
        let every = autosave_every.max(Duration::from_secs(1));
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
        if !listed {
            ops.push(canon);
        }
    }
    ops
}

fn canonical_name(raw: &str) -> String {
    clean_name(raw).to_ascii_lowercase()
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

#[cfg(test)]
mod tests;
