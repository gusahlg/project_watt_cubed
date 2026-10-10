//! Multiplayer chaos soak tests. See `super` for the harness.
//!
//! A seeded driver plays many headless clients at once. Honest players fly over a flat world,
//! edit a shared patch and cells of their own, chain edits, chat, send channel frames, swing, use
//! tools, leave and rejoin. The operator proves its secret, sets the time and teleports. Cheaters
//! edit out of reach, send junk specs, speed hack, flood every budget, and forge operator commands
//! and names. What each player does at each step is a pure function of the seed; only the
//! network's timing varies.
//!
//! Every client keeps the world its game would show (optimistic edits, rollbacks, snapshots, tool
//! results). At the end each must equal the ledger a fresh joiner receives, the roster and the
//! clock must agree everywhere, and every edit and tool use must have had a server verdict.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, Once, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use glam::DQuat;
use material::{Configuration, Element};
use voxel_engine::{DVec3, Vec3};

use super::Lobby;
use crate::block::registry::BlockRegistry;
use crate::coord::Face;
use crate::math::{PER_METER, block_coord};
use crate::net::chat;
use crate::net::client::{ConnectError, Connection, Incoming, PendingConnect};
use crate::net::persist;
use crate::net::server::{Config, NoclipPolicy, TeleportPolicy};
use crate::presence::Stance;
use crate::world::generation::{FLAT_HEIGHT, FlatTerrain, TerrainGenerator, WorldgenKind};
use crate::world::terrain::{Materials, TerrainCfg};

type Pos = (i32, i32, i32);

/// One driver step.
const STEP: Duration = Duration::from_millis(10);
const CHANNEL: &str = "chaos";
const OP: &str = "warden";
const SECRET: &str = "s3cret";
/// A client that reports this mod is refused at the door.
const BANNED_MOD: &str = "chaos.wallhack";
/// The client's own wait for a verdict (`client::PENDING_TTL`). A verdict this late was made up by
/// the client because the server never answered.
const VERDICT_TTL: Duration = Duration::from_secs(3);
/// The envelope cap these servers run with, so a speed hack is out of bounds.
const MAX_SPEED: f64 = 20.0;
/// The server's roster cap (`server::MAX_PLAYERS`). More joins than this in one run only fit when
/// every leave frees its slot.
const ROSTER_CAP: u32 = 256;
/// Players fly with their eyes here; the ground's top is y 12 and nobody edits above it.
const CRUISE: f64 = 17.0;
/// Players stay within this many blocks of the origin.
const ROAM: f64 = 5.0;
/// Shared cells: x and z in `-PATCH..=PATCH`, y 10 and 11.
const PATCH: i32 = 4;
const SHARED_Y: [i32; 2] = [FLAT_HEIGHT - 2, FLAT_HEIGHT - 1];
/// The first air layer over the patch is split between the players, a cell each, so the outcome
/// of every edit there is known exactly.
const OWN_Y: i32 = FLAT_HEIGHT;
/// Edits stay this far inside the server's reach (8 m), which absorbs the lag of the last move.
const REACH: f64 = 7.0 * PER_METER;
/// Cheaters' out-of-reach edits land at x of at least this.
const FAR: i32 = 1000;
const JUNK: [&str; 9] = ["c:zz", "c:00", "stone", "", "c:", "AIR", "c:0", " air", "c:g0"];

/// SplitMix64. [`Rng::at`] gives every (player, step) its own stream, so what one step draws
/// never shifts another: the script belongs to the seed, whatever the network does.
struct Rng(u64);

impl Rng {
    fn at(seed: u64, bot: usize, step: u32) -> Self {
        let mut rng = Self(
            seed ^ (bot as u64 + 1).wrapping_mul(0xA24B_AED4_963E_E407)
                ^ u64::from(step).wrapping_mul(0x9FB2_1C65_1E98_DF25),
        );
        rng.next();
        rng
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn span(&mut self, lo: i32, hi: i32) -> i32 {
        lo + self.below((hi - lo + 1) as usize) as i32
    }
}

/// The flat world's generated cells and the blocks players place, as canonical specs.
struct Ground {
    registry: BlockRegistry,
    terrain: FlatTerrain,
    air: Arc<str>,
    blocks: Vec<Arc<str>>,
}

impl Ground {
    fn new(seed: i64) -> Self {
        let mut registry = BlockRegistry::with_builtins();
        let terrain = FlatTerrain::new(&mut registry, seed);
        let m = Materials::intern(&mut registry);
        let blocks: Vec<Arc<str>> =
            [m.grass, m.soil, m.rock[0], m.plank, m.sand].iter().map(|&id| registry.spec(id).into()).collect();
        Self { registry, terrain, air: "air".into(), blocks }
    }

    fn generated(&self, (x, y, z): Pos) -> Arc<str> {
        self.registry.spec(self.terrain.voxel_at(x, y, z)).into()
    }

    /// What a game shows after applying `spec`: the game parses it, and text it cannot parse is air.
    fn shown(&self, spec: &str) -> Arc<str> {
        self.registry.canonical_spec(spec).map_or_else(|| self.air.clone(), Arc::from)
    }

    /// Whether `spec` is in canonical form. Checked once per distinct spec.
    fn canonical(&self, spec: &Arc<str>, checked: &mut HashSet<Arc<str>>) -> bool {
        checked.contains(spec) || {
            let fine = self.registry.canonical_spec(spec).as_deref() == Some(&**spec);
            if fine {
                checked.insert(spec.clone());
            }
            fine
        }
    }

    fn content(&self, cells: &HashMap<Pos, Arc<str>>, at: Pos) -> Arc<str> {
        cells.get(&at).cloned().unwrap_or_else(|| self.generated(at))
    }

    fn block(&self, rng: &mut Rng) -> Arc<str> {
        self.blocks[rng.below(self.blocks.len())].clone()
    }

    /// Air two times in five, else a placeable block.
    fn edit(&self, rng: &mut Rng) -> Arc<str> {
        if rng.unit() < 0.4 { self.air.clone() } else { self.block(rng) }
    }
}

/// A configuration no palette holds, as a spec.
fn novel_spec(rng: &mut Rng) -> Arc<str> {
    let elements: Vec<Element> = (0..1 + rng.below(3))
        .map(|_| Element::new(rng.next().to_le_bytes()[..4].try_into().expect("four bytes")))
        .collect();
    let config = Configuration::new(elements).expect("three elements fit");
    let mut spec = String::from("c:");
    for b in config.encode().as_bytes() {
        spec.push_str(&format!("{b:02x}"));
    }
    spec.into()
}

/// A channel frame: who sent it and its sequence number, then bytes both of those determine.
fn payload(sender: u32, seq: u32) -> Vec<u8> {
    let len = (seq.wrapping_mul(37) ^ sender) as usize % 200;
    let mut bytes = Vec::with_capacity(8 + len);
    bytes.extend_from_slice(&sender.to_le_bytes());
    bytes.extend_from_slice(&seq.to_le_bytes());
    bytes.extend((0..len).map(|k| (sender as usize + seq as usize * 7 + k) as u8));
    bytes
}

/// The player each own cell belongs to.
fn owner((x, _, z): Pos, bots: usize) -> usize {
    ((x + PATCH) * (2 * PATCH + 1) + (z + PATCH)) as usize % bots
}

fn centre((x, y, z): Pos) -> DVec3 {
    DVec3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5)
}

/// Worker threads that panicked (server handlers, client readers and writers, runtimes). Test
/// threads have names; workers have none, or are tokio's. Other suites' staged panics are left out.
static PANICS: Mutex<Vec<String>> = Mutex::new(Vec::new());

fn panics_so_far() -> usize {
    static HOOK: Once = Once::new();
    HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let worker = thread::current().name().is_none_or(|name| name.starts_with("tokio"));
            let payload = info.payload();
            let text = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_default();
            let staged = ["test hook panic", "reaction tick", "simulated", "injected"].iter().any(|s| text.contains(s));
            if worker && !staged {
                let at = info.location().map(|l| format!("{}:{}", l.file(), l.line())).unwrap_or_default();
                PANICS.lock().unwrap_or_else(PoisonError::into_inner).push(format!("{text} at {at}"));
            }
            previous(info);
        }));
    });
    PANICS.lock().unwrap_or_else(PoisonError::into_inner).len()
}

fn config(seed: i64, world: Option<PathBuf>) -> Config {
    Config {
        seed,
        worldgen: WorldgenKind::Flat,
        teleport: TeleportPolicy::Ops,
        noclip: NoclipPolicy::Off,
        op_secrets: vec![(OP.into(), SECRET.into())],
        max_speed: MAX_SPEED,
        mods_deny: vec![BANNED_MOD.into()],
        world,
        ..Config::default()
    }
}

fn begin(port: u16, name: &str, mods: &[(String, String)]) -> PendingConnect {
    Connection::begin_connect("127.0.0.1", port, name, "", mods)
}

#[derive(Clone, Copy)]
struct Plan {
    seed: u64,
    /// Players, the operator and the cheaters included.
    bots: usize,
    cheaters: usize,
    /// Chance per step that a player leaves for a while.
    leave: f64,
    /// Cheaters also place configurations no palette holds. Some start a reaction front that never
    /// settles, so the world is never quiet and only the liveness checks apply.
    novel: bool,
}

impl Plan {
    fn new(seed: u64, bots: usize, cheaters: usize) -> Self {
        Self { seed, bots, cheaters, leave: 0.002, novel: false }
    }

    /// `CHAOS_SEED` replays one seed.
    fn seeded(default: u64, bots: usize, cheaters: usize) -> Self {
        let seed = std::env::var("CHAOS_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(default);
        Self::new(seed, bots, cheaters)
    }
}

/// Steps for a soak: `CHAOS_SECS` overrides the default seconds.
fn soak_steps(default_secs: u32) -> u32 {
    let secs = std::env::var("CHAOS_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(default_secs);
    secs * 100
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Role {
    Honest,
    Op,
    Cheater,
    /// A fresh joiner that only listens: its snapshot is the server's ledger.
    Audit,
}

/// An edit waiting for its verdict, and what the game showed before it (for the rollback).
struct Sent {
    cell: Pos,
    prev: Option<Arc<str>>,
    spec: Arc<str>,
    at: Instant,
}

/// One own cell as far as its player can know: the content last confirmed there, and what
/// requests whose verdict it never heard may have left instead.
struct Owned {
    known: Arc<str>,
    maybe: Vec<Arc<str>>,
}

/// A cheat that touches other players, so the driver runs it.
enum Ask {
    Impersonate,
    BannedMod,
}

struct Bot {
    index: usize,
    name: String,
    role: Role,
    conn: Option<Connection>,
    joining: Option<PendingConnect>,
    /// The step an away player comes back at, and the earliest retry of a refused join.
    back: Option<u32>,
    retry: Instant,
    session: u32,
    /// When the current connection opened: from then on the server has this player on its roster.
    since: Instant,
    ready: bool,
    pos: DVec3,
    heading: f64,
    stance: Stance,
    flown: Instant,
    /// The world as this client's game shows it, where it may differ from generation.
    cells: HashMap<Pos, Arc<str>>,
    edits: HashMap<u32, Sent>,
    tools: HashMap<u32, Instant>,
    /// A `/tp` waiting for its echo.
    teleport: Option<Instant>,
    op: bool,
    asked_op: bool,
    last_set: Option<Instant>,
    /// No requests before this step: a flood may still fill the send queue, which drops frames.
    calm: u32,
    seq: u32,
    chats: u32,
    bursts: u32,
    /// Own global chats and `/time` values waiting for the server's echo.
    said: HashMap<Arc<str>, Instant>,
    set: HashMap<u32, Instant>,
    heard: HashSet<Arc<str>>,
    days: HashSet<u32>,
    clock: Option<(f32, f32, Instant)>,
    /// Last channel seq per sender, last chat number per sender session, chats per flood.
    seqs: HashMap<u32, u32>,
    order: HashMap<String, u32>,
    flood: HashMap<String, u32>,
    owned: HashMap<Pos, Owned>,
    /// A probe is using this player's name: it stays until the probe is answered.
    pinned: bool,
}

impl Bot {
    fn new(index: usize, name: String, role: Role) -> Self {
        let now = Instant::now();
        Self {
            index,
            name,
            role,
            conn: None,
            joining: None,
            back: None,
            retry: now,
            session: 0,
            since: now,
            ready: false,
            pos: DVec3::ZERO,
            heading: index as f64,
            stance: Stance::Standing,
            flown: now,
            cells: HashMap::new(),
            edits: HashMap::new(),
            tools: HashMap::new(),
            teleport: None,
            op: false,
            asked_op: false,
            last_set: None,
            calm: 0,
            seq: 0,
            chats: 0,
            bursts: 0,
            said: HashMap::new(),
            set: HashMap::new(),
            heard: HashSet::new(),
            days: HashSet::new(),
            clock: None,
            seqs: HashMap::new(),
            order: HashMap::new(),
            flood: HashMap::new(),
            owned: HashMap::new(),
            pinned: false,
        }
    }

    fn live(&self) -> bool {
        self.conn.is_some() && self.ready
    }

    fn attach(&mut self, conn: Connection, referee: &mut Referee) {
        referee.joins += 1;
        self.session += 1;
        self.since = Instant::now();
        self.flown = self.since;
        self.pos = conn.spawn();
        self.back = None;
        self.ready = false;
        self.op = false;
        self.asked_op = false;
        self.teleport = None;
        self.seq = 0;
        self.cells.clear();
        self.edits.clear();
        self.tools.clear();
        self.said.clear();
        self.set.clear();
        self.heard.clear();
        self.days.clear();
        self.clock = None;
        self.seqs.clear();
        self.order.clear();
        self.flood.clear();
        self.conn = Some(conn);
    }

    /// Drop the link on its own thread (a close waits for the peer) and come back at step `back`.
    fn leave(&mut self, back: Option<u32>, referee: &mut Referee) {
        if let Some(pending) = self.joining.take() {
            pending.cancel();
        }
        for sent in self.edits.values() {
            if let Some(owned) = self.owned.get_mut(&sent.cell) {
                owned.maybe.push(sent.spec.clone());
            }
            referee.abandoned.entry(sent.cell).or_default().push(sent.spec.clone());
        }
        self.edits.clear();
        self.tools.clear();
        self.ready = false;
        self.back = back;
        if let Some(conn) = self.conn.take() {
            referee.drops.push(thread::spawn(move || drop(conn)));
        }
    }

    /// Take what the server sent: join progress, events, channel frames.
    fn pump(&mut self, ground: &Ground, referee: &mut Referee) {
        if let Some(pending) = self.joining.as_mut()
            && let Some(result) = pending.poll()
        {
            self.joining = None;
            match result {
                Ok(conn) => self.attach(conn, referee),
                Err(e) => self.refused(e, referee),
            }
        }
        let Some(conn) = self.conn.as_mut() else { return };
        let events = conn.poll();
        let frames = conn.drain_channel(CHANNEL);
        let landed = conn.snapshot_ready();
        let alive = conn.is_alive();
        for event in events {
            self.take(event, ground, referee);
        }
        for frame in frames {
            self.frame(frame, referee);
        }
        if landed && !self.ready {
            self.ready = true;
            for (&cell, owned) in &mut self.owned {
                owned.known = ground.content(&self.cells, cell);
                owned.maybe.clear();
            }
        }
        if !alive && let Some(conn) = self.conn.take() {
            self.ready = false;
            referee.drops.push(thread::spawn(move || drop(conn)));
        }
    }

    fn refused(&mut self, e: ConnectError, referee: &mut Referee) {
        if e.message().contains("already in use") {
            // The server has not yet seen this player's last session leave.
            referee.retries += 1;
            self.retry = Instant::now() + Duration::from_millis(20);
        } else if e.message() != "cancelled" {
            referee.fault(format!("{} could not join: {}", self.name, e.message()));
            self.retry = Instant::now() + Duration::from_millis(200);
        }
    }

    fn take(&mut self, event: Incoming, ground: &Ground, referee: &mut Referee) {
        match event {
            Incoming::Edit { x, y, z, spec } => {
                referee.churn += 1;
                self.world((x, y, z), spec, ground, referee);
            }
            Incoming::Mutation { x, y, z, spec } => {
                referee.churn += 1;
                if self.ready {
                    // Past the join overlay a snapshot is a reaction's work.
                    referee.reactions += 1;
                    referee.stirred.insert((x, y, z));
                }
                self.world((x, y, z), spec, ground, referee);
            }
            Incoming::EditAccepted { req } => {
                if let Some(sent) = self.edits.remove(&req) {
                    self.answered(sent.at, "edit", referee);
                    if let Some(owned) = self.owned.get_mut(&sent.cell) {
                        owned.known = sent.spec;
                    }
                }
            }
            Incoming::EditRejected { req, restore } => {
                if let Some(sent) = self.edits.remove(&req) {
                    self.answered(sent.at, "edit", referee);
                    if restore {
                        match sent.prev {
                            Some(prev) => self.cells.insert(sent.cell, prev),
                            None => self.cells.remove(&sent.cell),
                        };
                    }
                }
            }
            Incoming::ToolResult { req, reacted, cell, cell_spec, .. } => {
                if let Some(at) = self.tools.remove(&req) {
                    self.answered(at, "tool use", referee);
                    if reacted {
                        referee.churn += 1;
                        referee.reacted += 1;
                        self.world(cell, cell_spec, ground, referee);
                    }
                }
            }
            Incoming::Position { pos, .. } => {
                let expected = match self.role {
                    Role::Cheater => true,
                    Role::Op => self.teleport.take().is_some(),
                    Role::Honest | Role::Audit => false,
                };
                if expected {
                    referee.corrections += 1;
                } else {
                    referee.fault(format!("{} was corrected from {:.2?} to {pos:.2?}", self.name, self.pos));
                }
                self.pos = pos;
            }
            Incoming::Chat { from_name, text, .. } => self.chat(&from_name, text, referee),
            Incoming::Time { day, day_secs } => {
                let bits = day.to_bits();
                if referee.forged.contains(&bits) {
                    referee.fault(format!("{} heard a refused /time {day}", self.name));
                }
                if let Some(at) = self.set.remove(&bits) {
                    referee.sets.push((bits, at));
                }
                self.days.insert(bits);
                self.clock = Some((day, day_secs, Instant::now()));
            }
            Incoming::Disconnected { reason } => {
                if !(referee.stopping && reason == "server shutting down") {
                    referee.fault(format!("{} was disconnected: {reason:?}", self.name));
                }
            }
            Incoming::Interrupted => {
                if !referee.stopping {
                    referee.fault(format!("{}'s link went quiet", self.name));
                }
            }
            Incoming::Joined { .. } | Incoming::Left { .. } | Incoming::PeerSwing { .. } => {}
        }
    }

    /// Authoritative content for a cell.
    fn world(&mut self, cell: Pos, spec: Arc<str>, ground: &Ground, referee: &mut Referee) {
        if !ground.canonical(&spec, &mut referee.specs) {
            referee.fault(format!("{} got the non-canonical spec {spec:?} at {cell:?}", self.name));
        }
        if cell.0 >= FAR {
            referee.fault(format!("{} got an out-of-reach edit at {cell:?}", self.name));
        }
        self.cells.insert(cell, spec);
    }

    fn answered(&self, at: Instant, what: &str, referee: &mut Referee) {
        let took = at.elapsed();
        if took + Duration::from_millis(50) >= VERDICT_TTL {
            referee.fault(format!("{}: a {what} went unanswered ({took:.2?})", self.name));
        } else {
            referee.verdicts += 1;
        }
    }

    fn chat(&mut self, from: &str, text: Arc<str>, referee: &mut Referee) {
        if text.contains("/op") {
            referee.fault(format!("{} heard an operator login: {text:?}", self.name));
        }
        if from == "server" {
            match (self.role, &*text) {
                (Role::Op, "you are now an operator") => self.op = true,
                (Role::Cheater, _) => {}
                _ => referee.fault(format!("{} was told {text:?}", self.name)),
            }
            return;
        }
        if let Some(at) = self.said.remove(&text) {
            referee.chats.push((text.clone(), at));
        }
        if !self.heard.insert(text.clone()) {
            referee.fault(format!("{} heard {text:?} twice", self.name));
        }
        let parts: Vec<&str> = text.split(':').collect();
        match parts.as_slice() {
            ["g" | "l", name, session, n] => {
                if *name != from {
                    referee.fault(format!("{} heard {text:?} from {from:?}", self.name));
                }
                let n: u32 = n.parse().unwrap_or(0);
                let key = format!("{name}:{session}");
                if self.order.get(&key).is_some_and(|&last| n <= last) {
                    referee.fault(format!("{} heard {text:?} out of order", self.name));
                }
                self.order.insert(key, n);
            }
            ["f", name, session, burst, _] => {
                let count = self.flood.entry(format!("{name}:{session}:{burst}")).or_insert(0);
                *count += 1;
                if *count > 5 {
                    referee.fault(format!("{} heard more of one chat flood than the budget", self.name));
                }
            }
            _ => referee.fault(format!("{} heard an unknown chat {text:?}", self.name)),
        }
    }

    fn frame(&mut self, (sender, seq, bytes): (u32, u32, Vec<u8>), referee: &mut Referee) {
        referee.frames += 1;
        if bytes != payload(sender, seq) {
            referee.fault(format!("{}: frame {seq} stamped #{sender} does not match its payload", self.name));
        }
        if self.seqs.get(&sender).is_some_and(|&last| seq <= last) {
            referee.fault(format!("{}: frame {seq} from #{sender} out of order", self.name));
        }
        self.seqs.insert(sender, seq);
    }

    /// Fly one step: a wandering heading over the patch at walking speed, eyes at [`CRUISE`].
    fn fly(&mut self, rng: &mut Rng) {
        let now = Instant::now();
        let dt = now.duration_since(self.flown).as_secs_f64().min(0.05);
        self.flown = now;
        self.heading += (rng.unit() - 0.5) * 0.5;
        let speed = 2.0 + 3.0 * rng.unit();
        let mut v = DVec3::new(
            self.heading.cos() * speed,
            ((CRUISE - self.pos.y) * 4.0).clamp(-3.0, 3.0),
            self.heading.sin() * speed,
        );
        if (self.pos.x + v.x * dt).abs() > ROAM {
            v.x = -v.x;
        }
        if (self.pos.z + v.z * dt).abs() > ROAM {
            v.z = -v.z;
        }
        self.heading = v.z.atan2(v.x);
        if rng.unit() < 0.01 {
            self.stance = if self.stance == Stance::Standing { Stance::Sneaking } else { Stance::Standing };
        }
        self.pos += v * dt;
        let Some(conn) = self.conn.as_mut() else { return };
        conn.send_move(self.pos, self.heading as f32, 0.0, DQuat::IDENTITY, v.as_vec3(), Face::PosY, self.stance);
    }

    fn reaches(&self, cell: Pos) -> bool {
        self.pos.distance(centre(cell)) <= REACH
    }

    /// A shared cell in reach, if a few tries find one.
    fn shared_cell(&self, rng: &mut Rng) -> Option<Pos> {
        (0..12)
            .map(|_| (rng.span(-PATCH, PATCH), SHARED_Y[rng.below(2)], rng.span(-PATCH, PATCH)))
            .find(|&cell| self.reaches(cell))
    }

    fn own_cell(&self, rng: &mut Rng, bots: usize) -> Option<Pos> {
        (0..16)
            .map(|_| (rng.span(-PATCH, PATCH), OWN_Y, rng.span(-PATCH, PATCH)))
            .find(|&cell| owner(cell, bots) == self.index && self.reaches(cell))
    }

    /// Edit like the game: show the result at once and remember what to roll back to.
    fn send(&mut self, cell: Pos, spec: Arc<str>, ground: &Ground, bots: usize) -> Option<u32> {
        let conn = self.conn.as_mut()?;
        let req = conn.send_edit(cell.0, cell.1, cell.2, spec.clone())?;
        let shown = ground.shown(&spec);
        if cell.1 == OWN_Y && owner(cell, bots) == self.index && self.role != Role::Cheater {
            let known = ground.content(&self.cells, cell);
            self.owned.entry(cell).or_insert(Owned { known, maybe: Vec::new() });
        }
        let prev = self.cells.insert(cell, shown.clone());
        self.edits.insert(req, Sent { cell, prev, spec: shown, at: Instant::now() });
        Some(req)
    }

    /// Break then place on one cell before either verdict.
    fn chain(&mut self, cell: Pos, rng: &mut Rng, ground: &Ground, bots: usize) {
        self.send(cell, ground.air.clone(), ground, bots);
        let spec = ground.block(rng);
        self.send(cell, spec, ground, bots);
    }

    fn tool(&mut self, cell: Pos, spec: Arc<str>, referee: &mut Referee) {
        let Some(conn) = self.conn.as_mut() else { return };
        if let Some(req) = conn.send_tool_use(cell.0, cell.1, cell.2, spec) {
            self.tools.insert(req, Instant::now());
            referee.stirred.insert(cell);
        }
    }

    fn say(&mut self, global: bool) {
        let Some(conn) = self.conn.as_mut() else { return };
        let kind = if global { "g" } else { "l" };
        let text: Arc<str> = format!("{kind}:{}:{}:{}", self.name, self.session, self.chats).into();
        self.chats += 1;
        if global {
            conn.send_chat(chat::GLOBAL, &text);
            // A cheater's own floods spend its chat budget, so only honest chats must arrive.
            if self.role != Role::Cheater {
                self.said.insert(text, Instant::now());
            }
        } else {
            conn.send_chat(chat::LOCAL, &text);
        }
    }

    fn send_frames(&mut self, n: u32) {
        let Some(conn) = self.conn.as_mut() else { return };
        let id = conn.player_id();
        for _ in 0..n {
            conn.send_channel(CHANNEL, self.seq, &payload(id, self.seq));
            self.seq += 1;
        }
    }

    /// The legal acts every player has, at most one per step. `roll` is spent as it is read; false
    /// when it chose none.
    fn play(&mut self, roll: &mut f64, rng: &mut Rng, step: u32, plan: &Plan, ground: &Ground, referee: &mut Referee) -> bool {
        let mut chance = |p: f64| {
            let hit = *roll < p;
            *roll -= p;
            hit
        };
        let bots = plan.bots;
        let honest = self.role != Role::Cheater;
        let calm = step >= self.calm;
        if chance(0.03) {
            let cell = if honest && rng.unit() < 0.35 { self.own_cell(rng, bots) } else { self.shared_cell(rng) };
            if let Some(cell) = cell.filter(|_| calm) {
                let spec = ground.edit(rng);
                self.send(cell, spec, ground, bots);
            }
        } else if chance(0.004) {
            if let Some(cell) = self.own_cell(rng, bots).filter(|_| honest && calm) {
                self.chain(cell, rng, ground, bots);
            }
        } else if chance(0.006) {
            self.say(rng.unit() < 0.7);
        } else if chance(0.05) {
            self.send_frames(1);
        } else if chance(0.02) {
            if let Some(conn) = self.conn.as_mut() {
                conn.send_swing();
            }
        } else if chance(0.006) {
            if let Some(cell) = self.shared_cell(rng).filter(|_| calm) {
                let spec = ground.block(rng);
                self.tool(cell, spec, referee);
            }
        } else if chance(if self.role == Role::Op { plan.leave / 3.0 } else { plan.leave }) {
            if !self.pinned {
                self.leave(Some(step + 20 + rng.below(60) as u32), referee);
            }
        } else {
            return false;
        }
        true
    }

    /// The operator's powers, once its secret is proved.
    fn rule(&mut self, rng: &mut Rng) {
        let Some(conn) = self.conn.as_mut() else { return };
        if !self.asked_op {
            self.asked_op = true;
            conn.send_chat(chat::GLOBAL, &format!("/op {SECRET}"));
            return;
        }
        let roll = rng.unit();
        if !self.op {
            return;
        }
        if roll < 0.01 {
            if self.last_set.is_some_and(|t| t.elapsed() < Duration::from_millis(1500)) {
                return;
            }
            let day = rng.below(5000) as f32 / 10_000.0;
            conn.send_set_time(day);
            self.set.insert(day.to_bits(), Instant::now());
            self.last_set = Some(Instant::now());
        } else if roll < 0.015 && self.teleport.is_none() {
            let to = DVec3::new(rng.span(-PATCH, PATCH) as f64 + 0.5, CRUISE, rng.span(-PATCH, PATCH) as f64 + 0.5);
            conn.send_teleport(to);
            self.teleport = Some(Instant::now());
            self.pos = to;
        }
    }

    /// One illegal act. Each must be refused or dropped without a kick, and must never reach the
    /// ledger, the clock, or anyone's chat.
    fn cheat(&mut self, rng: &mut Rng, step: u32, plan: &Plan, ground: &Ground, referee: &mut Referee) -> Option<Ask> {
        if step < self.calm {
            return None;
        }
        let bots = plan.bots;
        let pos = self.pos;
        match rng.below(15) {
            0 => {
                let cell = (FAR + rng.below(100) as i32, FLAT_HEIGHT - 1, rng.span(-50, 50));
                let spec = ground.block(rng);
                self.send(cell, spec, ground, bots);
            }
            1 => {
                if let Some(cell) = self.shared_cell(rng) {
                    self.send(cell, JUNK[rng.below(JUNK.len())].into(), ground, bots);
                }
            }
            2 => {
                // A speed hack: the cheater keeps flying from the far point until snapped back.
                let lunge = 300.0 + 100.0 * rng.unit();
                self.pos += if rng.unit() < 0.5 { DVec3::new(lunge, 0.0, 0.0) } else { DVec3::new(0.0, 0.0, -lunge) };
            }
            3 => {
                let day = 0.5 + rng.below(4900) as f32 / 10_000.0;
                referee.forged.insert(day.to_bits());
                self.conn.as_mut()?.send_set_time(day);
            }
            4 => self.conn.as_mut()?.send_teleport(pos + DVec3::new(2.0, 0.0, 0.0)),
            5 => {
                let cells: HashSet<Pos> = (0..40).filter_map(|_| self.shared_cell(rng)).collect();
                for cell in cells.into_iter().take(25) {
                    let spec = ground.edit(rng);
                    self.send(cell, spec, ground, bots);
                }
            }
            6 => {
                self.bursts += 1;
                let (name, session, burst) = (self.name.clone(), self.session, self.bursts);
                let conn = self.conn.as_mut()?;
                for k in 0..12 {
                    conn.send_chat(chat::GLOBAL, &format!("f:{name}:{session}:{burst}:{k}"));
                }
            }
            7 => {
                self.send_frames(55);
                self.calm = step + 10;
            }
            8 => {
                let conn = self.conn.as_mut()?;
                for k in 0..40 {
                    conn.send_channel(&format!("spray{k}"), 0, &[]);
                }
                self.calm = step + 10;
            }
            9 => self.conn.as_mut()?.send_chat(chat::GLOBAL, "/op hunter2"),
            10 => {
                if let Some(cell) = self.shared_cell(rng) {
                    let spec = if plan.novel { novel_spec(rng) } else { JUNK[rng.below(JUNK.len())].into() };
                    self.send(cell, spec, ground, bots);
                }
            }
            11 => {
                for _ in 0..16 {
                    if let Some(cell) = self.shared_cell(rng) {
                        let spec = ground.block(rng);
                        self.tool(cell, spec, referee);
                    }
                }
            }
            12 => {
                let stance = self.stance;
                let conn = self.conn.as_mut()?;
                conn.send_move(DVec3::new(f64::NAN, pos.y, pos.z), f32::NAN, 0.0, DQuat::IDENTITY, Vec3::ZERO, Face::PosY, stance);
                conn.send_teleport(DVec3::new(2.0e9, CRUISE, 0.0));
            }
            13 => return Some(Ask::Impersonate),
            _ => return Some(Ask::BannedMod),
        }
        None
    }
}

/// A join that must be refused, and the player whose name it borrows.
struct Probe {
    pending: PendingConnect,
    kind: Ask,
    target: Option<usize>,
}

/// Faults found so far and the facts the end checks need.
#[derive(Default)]
struct Referee {
    faults: Vec<String>,
    verdicts: u32,
    joins: u32,
    retries: u32,
    frames: u32,
    corrections: u32,
    reactions: u32,
    reacted: u32,
    /// World events seen: the world is quiet when this stops moving.
    churn: u64,
    /// Specs already found canonical.
    specs: HashSet<Arc<str>>,
    /// Cells a tool use or a reaction touched: what they hold is the law's to say.
    stirred: HashSet<Pos>,
    /// Edits whose player left before the verdict.
    abandoned: HashMap<Pos, Vec<Arc<str>>>,
    /// Honest global chats the server relayed, and when they were sent.
    chats: Vec<(Arc<str>, Instant)>,
    /// The operator's `/time` values the server applied, and when they were sent.
    sets: Vec<(u32, Instant)>,
    /// `/time` values the server must refuse.
    forged: HashSet<u32>,
    /// The server is shutting down: its close is expected.
    stopping: bool,
    probes: Vec<Probe>,
    drops: Vec<JoinHandle<()>>,
}

impl Referee {
    fn fault(&mut self, text: String) {
        self.faults.push(text);
    }
}

/// What the players knew when the server stopped. The restarted server must hold a world that
/// fits it.
struct Before {
    /// Per cell, every content a client showed and every request without a verdict.
    seen: HashMap<Pos, HashSet<Arc<str>>>,
    /// Per own cell, its owner and what the save may hold there.
    owned: Vec<(String, Pos, HashSet<Arc<str>>)>,
    /// The day fraction at the stop.
    day: Option<f64>,
    day_secs: f32,
}

struct Chaos {
    plan: Plan,
    ground: Ground,
    bots: Vec<Bot>,
    audit: Option<Bot>,
    referee: Referee,
    port: u16,
    step: u32,
    started: Instant,
}

impl Chaos {
    fn new(plan: Plan, port: u16) -> Self {
        let bots = (0..plan.bots)
            .map(|i| match i {
                0 => Bot::new(0, OP.into(), Role::Op),
                i if i <= plan.cheaters => Bot::new(i, format!("cheat{i}"), Role::Cheater),
                i => Bot::new(i, format!("p{i}"), Role::Honest),
            })
            .collect();
        let ground = Ground::new(plan.seed as i64);
        Self { plan, ground, bots, audit: None, referee: Referee::default(), port, step: 0, started: Instant::now() }
    }

    fn pump_all(&mut self) {
        let Self { ground, bots, audit, referee, port, step, .. } = self;
        let now = Instant::now();
        for bot in bots.iter_mut() {
            if bot.conn.is_none() && bot.joining.is_none() && bot.back.is_some_and(|b| *step >= b) && now >= bot.retry {
                bot.joining = Some(begin(*port, &bot.name, &[]));
            }
            bot.pump(ground, referee);
        }
        if let Some(audit) = audit {
            audit.pump(ground, referee);
        }
        let mut probes = std::mem::take(&mut referee.probes);
        probes.retain_mut(|probe| {
            let Some(result) = probe.pending.poll() else { return true };
            if let Some(target) = probe.target {
                bots[target].pinned = false;
            }
            match (result, &probe.kind) {
                (Ok(conn), kind) => {
                    let what = match kind {
                        Ask::Impersonate => "the operator's name in capitals",
                        Ask::BannedMod => "a banned mod",
                    };
                    referee.fault(format!("a join with {what} got in"));
                    referee.drops.push(thread::spawn(move || drop(conn)));
                }
                (Err(e), Ask::Impersonate) if e.message().contains("already in use") => {}
                (Err(e), Ask::BannedMod) if e.mods_denied == [BANNED_MOD] => {}
                (Err(e), _) if e.message() == "cancelled" => {}
                (Err(e), _) => referee.fault(format!("a refused join gave the wrong reason: {}", e.message())),
            }
            false
        });
        referee.probes.append(&mut probes);
    }

    fn act(&mut self, i: usize) {
        let step = self.step;
        let plan = self.plan;
        let Self { ground, bots, referee, port, .. } = self;
        let bot = &mut bots[i];
        if !bot.live() {
            return;
        }
        let mut rng = Rng::at(plan.seed, i, step);
        bot.fly(&mut rng);
        if bot.role == Role::Op {
            bot.rule(&mut rng);
        }
        let mut roll = rng.unit();
        if bot.play(&mut roll, &mut rng, step, &plan, ground, referee) || bot.role != Role::Cheater || roll >= 0.015 {
            return;
        }
        match bot.cheat(&mut rng, step, &plan, ground, referee) {
            Some(Ask::Impersonate) => {
                // The operator's name in capitals, while the operator is on.
                if bots[0].live() && !bots[0].pinned {
                    bots[0].pinned = true;
                    let pending = begin(*port, &OP.to_uppercase(), &[]);
                    referee.probes.push(Probe { pending, kind: Ask::Impersonate, target: Some(0) });
                }
            }
            Some(Ask::BannedMod) => {
                let pending = begin(*port, &format!("mallory{step}"), &[(BANNED_MOD.into(), "1.0".into())]);
                referee.probes.push(Probe { pending, kind: Ask::BannedMod, target: None });
            }
            None => {}
        }
    }

    /// Pump everyone until `done` holds. A timeout is a fault naming `what`.
    fn until(&mut self, timeout: Duration, what: &str, done: impl Fn(&Self) -> bool) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            self.pump_all();
            if done(self) {
                return true;
            }
            if Instant::now() >= deadline {
                let report = self.report();
                self.referee.fault(format!("timed out waiting until {what}: {report}"));
                return false;
            }
            thread::sleep(Duration::from_millis(5));
        }
    }

    /// Who is still waiting on what.
    fn report(&self) -> String {
        self.bots
            .iter()
            .filter(|b| !b.live() || !b.edits.is_empty() || !b.tools.is_empty() || b.teleport.is_some() || !b.said.is_empty() || !b.set.is_empty())
            .map(|b| {
                format!(
                    "{} (live {}, joining {}, edits {}, tools {}, teleport {}, chats {}, times {})",
                    b.name,
                    b.live(),
                    b.joining.is_some(),
                    b.edits.len(),
                    b.tools.len(),
                    b.teleport.is_some(),
                    b.said.len(),
                    b.set.len()
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// Everyone connects and lands a join snapshot.
    fn gather(&mut self) {
        let step = self.step;
        for bot in &mut self.bots {
            bot.pinned = false;
            if bot.conn.is_none() {
                bot.back = Some(step);
            }
        }
        self.until(Duration::from_secs(10), "everyone is on", |c| c.bots.iter().all(Bot::live));
    }

    /// The chaos itself, on a fixed step clock.
    fn run(&mut self, steps: u32) {
        let start = Instant::now();
        for n in 0..steps {
            self.pump_all();
            for i in 0..self.bots.len() {
                self.act(i);
            }
            self.step += 1;
            if let Some(rest) = (start + STEP * (n + 1)).checked_duration_since(Instant::now()) {
                thread::sleep(rest);
            }
        }
    }

    /// Bring everyone back, wait for every verdict and echo, then for the world to go quiet.
    fn settle(&mut self) {
        self.gather();
        self.until(Duration::from_secs(6), "every request is answered", |c| {
            c.referee.probes.is_empty()
                && c.bots.iter().all(|b| {
                    b.live()
                        && b.edits.is_empty()
                        && b.tools.is_empty()
                        && b.teleport.is_none()
                        && b.said.is_empty()
                        && b.set.is_empty()
                })
        });
        if !self.plan.novel {
            self.quiet();
        }
    }

    fn quiet(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(3);
        let mut seen = self.referee.churn;
        let mut still = Instant::now();
        while still.elapsed() < Duration::from_millis(150) {
            if Instant::now() >= deadline {
                self.referee.fault("the world never went quiet".into());
                return;
            }
            thread::sleep(Duration::from_millis(5));
            self.pump_all();
            if self.referee.churn != seen {
                seen = self.referee.churn;
                still = Instant::now();
            }
        }
    }

    /// A fresh joiner: its snapshot is the server's ledger and its roster the server's.
    fn audit(&mut self) {
        let name = format!("audit{}", self.step);
        let mut audit = Bot::new(usize::MAX, name.clone(), Role::Audit);
        audit.joining = Some(begin(self.port, &name, &[]));
        self.audit = Some(audit);
        let live = self.bots.iter().filter(|b| b.live()).count();
        self.until(Duration::from_secs(5), "the auditor and the roster are in", |c| {
            let Some(conn) = c.audit.as_ref().filter(|a| a.live()).and_then(|a| a.conn.as_ref()) else { return false };
            let id = conn.player_id();
            conn.peers().count() == live
                && c.bots.iter().filter(|b| b.live()).all(|b| b.conn.as_ref().is_some_and(|c| c.peers().any(|p| p.id() == id)))
        });
        if !self.plan.novel {
            self.quiet();
        }
    }

    /// Every client shows the ledger, sees the roster, keeps the clock, and heard what it should.
    fn check(&mut self) {
        let Some(audit) = self.audit.as_ref() else { return };
        let Some(audit_conn) = audit.conn.as_ref() else {
            self.referee.fault("the auditor is not on".into());
            return;
        };
        let ground = &self.ground;
        let referee = &self.referee;
        let ledger = &audit.cells;
        let mut faults = Vec::new();
        for cell in ledger.keys().filter(|c| c.0 >= FAR) {
            faults.push(format!("the ledger holds an out-of-reach edit at {cell:?}"));
        }
        let roster: HashSet<(u32, String)> = self
            .bots
            .iter()
            .filter(|b| b.live())
            .filter_map(|b| b.conn.as_ref().map(|c| (c.player_id(), b.name.clone())))
            .collect();
        let seen: HashSet<(u32, String)> = audit_conn.peers().map(|p| (p.id(), p.name.to_string())).collect();
        if seen != roster {
            faults.push(format!("the auditor's roster {seen:?} is not the live set {roster:?}"));
        }
        let reference = audit.clock.map(|(day, secs, at)| day as f64 + at.elapsed().as_secs_f64() / secs as f64);
        for bot in self.bots.iter().filter(|b| b.live()) {
            let Some(conn) = bot.conn.as_ref() else { continue };
            let mut wrong = Vec::new();
            for &cell in ledger.keys().chain(bot.cells.keys()).collect::<HashSet<_>>() {
                let want = ground.content(ledger, cell);
                let have = ground.content(&bot.cells, cell);
                if want != have {
                    wrong.push(format!("{cell:?} shows {have} where the server has {want}"));
                }
            }
            if !wrong.is_empty() && !self.plan.novel {
                wrong.sort();
                faults.push(format!("{} is out of step on {} cells: {}", bot.name, wrong.len(), wrong.join(", ")));
            }
            let mut expected: HashSet<(u32, String)> =
                roster.iter().filter(|(id, _)| *id != conn.player_id()).cloned().collect();
            expected.insert((audit_conn.player_id(), audit.name.clone()));
            let peers: HashSet<(u32, String)> = conn.peers().map(|p| (p.id(), p.name.to_string())).collect();
            if peers != expected {
                faults.push(format!("{} sees {peers:?}, not {expected:?}", bot.name));
            }
            for (text, at) in &referee.chats {
                if bot.since < *at && !bot.heard.contains(text) {
                    faults.push(format!("{} never heard {text:?}", bot.name));
                }
            }
            for (bits, at) in &referee.sets {
                if bot.since < *at && !bot.days.contains(bits) {
                    faults.push(format!("{} never heard /time {}", bot.name, f32::from_bits(*bits)));
                }
            }
            match (bot.clock, reference) {
                (Some((day, secs, at)), Some(want)) => {
                    let have = day as f64 + at.elapsed().as_secs_f64() / secs as f64;
                    let off = (have - want).rem_euclid(1.0);
                    if off.min(1.0 - off) > 0.002 {
                        faults.push(format!("{}'s clock reads {have:.4}, the server's {want:.4}", bot.name));
                    }
                }
                _ => faults.push(format!("{} never heard the time", bot.name)),
            }
        }
        self.referee.faults.extend(faults);
    }

    /// Everyone leaves; a newcomer then finds nobody: the roster shrank back to empty.
    fn disband(&mut self) {
        for bot in &mut self.bots {
            bot.leave(None, &mut self.referee);
        }
        if let Some(mut audit) = self.audit.take() {
            audit.leave(None, &mut self.referee);
        }
        for handle in self.referee.drops.drain(..) {
            let _ = handle.join();
        }
        let deadline = Instant::now() + Duration::from_secs(4);
        for attempt in 0.. {
            let name = format!("last{attempt}");
            let mut last = Bot::new(usize::MAX, name.clone(), Role::Audit);
            last.joining = Some(begin(self.port, &name, &[]));
            let wait = Instant::now() + Duration::from_secs(3);
            while Instant::now() < wait && !last.clock.is_some_and(|(_, _, at)| at.elapsed() > Duration::from_millis(50)) {
                last.pump(&self.ground, &mut self.referee);
                thread::sleep(Duration::from_millis(5));
            }
            let names = last.conn.as_ref().map(|c| c.peers().map(|p| p.name.to_string()).collect::<Vec<_>>());
            last.leave(None, &mut self.referee);
            match names {
                Some(names) if names.is_empty() => break,
                _ if Instant::now() < deadline => thread::sleep(Duration::from_millis(100)),
                Some(names) => {
                    self.referee.fault(format!("{names:?} stayed on the roster after everyone left"));
                    break;
                }
                None => {
                    self.referee.fault("a newcomer could not join after everyone left".into());
                    break;
                }
            }
        }
    }

    /// Stop the server mid-traffic and collect what every client heard before the close.
    fn stop(&mut self, lobby: Lobby) -> Before {
        self.referee.stopping = true;
        for bot in &mut self.bots {
            bot.back = None;
            bot.pinned = false;
            if let Some(pending) = bot.joining.take() {
                pending.cancel();
            }
        }
        for probe in self.referee.probes.drain(..) {
            probe.pending.cancel();
        }
        lobby.server.stop();
        drop(lobby);
        self.until(Duration::from_secs(5), "every client saw the server stop", |c| c.bots.iter().all(|b| b.conn.is_none()));
        let mut seen: HashMap<Pos, HashSet<Arc<str>>> = HashMap::new();
        let mut owned = Vec::new();
        let mut clock = None;
        for bot in &self.bots {
            for (&cell, spec) in &bot.cells {
                seen.entry(cell).or_default().insert(spec.clone());
            }
            for sent in bot.edits.values() {
                seen.entry(sent.cell).or_default().insert(sent.spec.clone());
            }
            for (&cell, own) in &bot.owned {
                let mut may: HashSet<Arc<str>> = own.maybe.iter().cloned().collect();
                may.insert(own.known.clone());
                may.extend(bot.edits.values().filter(|s| s.cell == cell).map(|s| s.spec.clone()));
                owned.push((bot.name.clone(), cell, may));
            }
            if clock.is_none() {
                clock = bot.clock;
            }
        }
        for (cell, specs) in &self.referee.abandoned {
            seen.entry(*cell).or_default().extend(specs.iter().cloned());
        }
        // A client that never heard of a cell shows it as generated.
        for (cell, specs) in seen.iter_mut() {
            if self.bots.iter().any(|b| !b.cells.contains_key(cell)) {
                specs.insert(self.ground.generated(*cell));
            }
        }
        let day = clock.map(|(day, secs, at)| day as f64 + at.elapsed().as_secs_f64() / secs as f64);
        let day_secs = clock.map_or(600.0, |(_, secs, _)| secs);
        Before { seen, owned, day, day_secs }
    }

    /// The restarted server's ledger against what the clients knew at the stop.
    fn check_saved(&mut self, before: &Before, started: Instant) {
        let Some(audit) = self.audit.as_ref() else { return };
        let ground = &self.ground;
        let referee = &mut self.referee;
        for (name, cell, may) in &before.owned {
            let saved = ground.content(&audit.cells, *cell);
            if !referee.stirred.contains(cell) && !may.contains(&saved) {
                referee.fault(format!("{name}'s cell {cell:?} came back as {saved}, not one of {may:?}"));
            }
        }
        for &cell in audit.cells.keys().chain(before.seen.keys()).collect::<HashSet<_>>() {
            if referee.stirred.contains(&cell) {
                continue;
            }
            let saved = ground.content(&audit.cells, cell);
            let fits = match before.seen.get(&cell) {
                Some(may) => may.contains(&saved),
                None => saved == ground.generated(cell),
            };
            if !fits {
                referee.fault(format!("{cell:?} came back as {saved}, which no client showed or asked for"));
            }
        }
        match (before.day, audit.clock) {
            (Some(day), Some((now, secs, at))) => {
                let want = day + at.duration_since(started).as_secs_f64() / secs as f64;
                let off = (now as f64 - want).rem_euclid(1.0);
                if off.min(1.0 - off) > 0.002 + 1.0 / before.day_secs as f64 {
                    referee.fault(format!("the clock came back at {now:.4}, not {want:.4}"));
                }
            }
            _ => referee.fault("no clock to compare across the restart".into()),
        }
    }

    fn finish(mut self, panics: usize, label: &str) {
        for handle in self.referee.drops.drain(..) {
            let _ = handle.join();
        }
        let since = PANICS.lock().unwrap_or_else(PoisonError::into_inner)[panics..].to_vec();
        for panic in since {
            self.referee.fault(format!("a worker thread panicked: {panic}"));
        }
        let r = &self.referee;
        eprintln!(
            "{label} seed {} in {:.1?}: {} joins ({} refused while the last session left), {} verdicts, {} frames, \
             {} chats, {} time sets, {} corrections, {} tool reactions, {} reaction cells",
            self.plan.seed,
            self.started.elapsed(),
            r.joins,
            r.retries,
            r.verdicts,
            r.frames,
            r.chats.len(),
            r.sets.len(),
            r.corrections,
            r.reacted,
            r.reactions
        );
        assert!(
            r.verdicts > 0 && r.joins >= self.plan.bots as u32,
            "{label} seed {} did nothing: {} verdicts, {} joins",
            self.plan.seed,
            r.verdicts,
            r.joins
        );
        let shown = r.faults.iter().take(40).cloned().collect::<Vec<_>>().join("\n");
        assert!(r.faults.is_empty(), "{label} seed {}: {} faults\n{shown}", self.plan.seed, r.faults.len());
    }
}

/// Join, play `steps`, settle, audit, check, leave.
fn soak(plan: Plan, steps: u32) -> Chaos {
    let lobby = Lobby::start(config(plan.seed as i64, None));
    let mut chaos = Chaos::new(plan, lobby.port);
    chaos.gather();
    chaos.run(steps);
    chaos.settle();
    chaos.audit();
    chaos.check();
    chaos.disband();
    lobby.server.stop();
    chaos
}

/// Play, stop the server mid-traffic, restart it from its world file, check what came back,
/// play again, and check everything.
fn restart(plan: Plan, first: u32, second: u32) -> Chaos {
    let path = crate::save::store::test_temp_path("chaos-restart");
    let lobby = Lobby::start(config(plan.seed as i64, Some(path.clone())));
    let mut chaos = Chaos::new(plan, lobby.port);
    chaos.gather();
    chaos.run(first);
    let before = chaos.stop(lobby);
    let started = Instant::now();
    // The file's seed wins over the flag.
    let lobby = Lobby::start(config(plan.seed as i64 + 1, Some(path.clone())));
    chaos.port = lobby.port;
    chaos.referee.stopping = false;
    chaos.audit();
    chaos.check_saved(&before, started);
    if let Some(mut audit) = chaos.audit.take() {
        audit.leave(None, &mut chaos.referee);
    }
    chaos.gather();
    chaos.run(second);
    chaos.settle();
    chaos.audit();
    chaos.check();
    chaos.disband();
    lobby.server.stop();
    for suffix in ["", ".bak", ".tmp"] {
        let mut name = path.clone().into_os_string();
        name.push(suffix);
        let _ = std::fs::remove_file(name);
    }
    chaos
}

#[test]
fn eight_players_three_seconds_of_chaos() {
    let panics = panics_so_far();
    let plan = Plan { leave: 0.004, ..Plan::seeded(0xC4A0_5EED, 8, 2) };
    soak(plan, 250).finish(panics, "chaos");
}

#[test]
fn a_server_restart_mid_traffic_keeps_every_answered_edit() {
    let panics = panics_so_far();
    let plan = Plan { leave: 0.004, ..Plan::seeded(0x5EED_0B07, 8, 2) };
    restart(plan, 100, 100).finish(panics, "restart");
}

#[test]
#[ignore = "soak: twelve seeds of the suite's runs, every third one a restart (CHAOS_RUNS)"]
fn a_dozen_seeds() {
    let runs = std::env::var("CHAOS_RUNS").ok().and_then(|s| s.parse().ok()).unwrap_or(12u64);
    for seed in 1..=runs {
        let panics = panics_so_far();
        let plan = Plan { leave: 0.004, ..Plan::new(seed, 8, 2) };
        let chaos = if seed % 3 == 0 { restart(plan, 100, 100) } else { soak(plan, 250) };
        chaos.finish(panics, "seeds");
    }
}

#[test]
#[ignore = "soak: 16 players for 30 s (CHAOS_SECS, CHAOS_SEED)"]
fn sixteen_players_soak() {
    let panics = panics_so_far();
    let plan = Plan::seeded(0x50A4_0016, 16, 3);
    soak(plan, soak_steps(30)).finish(panics, "soak");
}

#[test]
#[ignore = "soak: 16 players churning past the roster cap for 40 s (CHAOS_SECS, CHAOS_SEED)"]
fn churn_past_the_roster_cap_frees_every_slot() {
    let panics = panics_so_far();
    let plan = Plan { leave: 0.012, ..Plan::seeded(0xC4E2_0256, 16, 3) };
    let chaos = soak(plan, soak_steps(40));
    let joins = chaos.referee.joins;
    chaos.finish(panics, "churn");
    assert!(joins > ROSTER_CAP, "only {joins} joins: the run must outnumber the roster cap to show leaves free slots");
}

#[test]
#[ignore = "soak: 16 players, a restart after 20 s, then 20 s more (CHAOS_SECS, CHAOS_SEED)"]
fn sixteen_players_restart_soak() {
    let panics = panics_so_far();
    let plan = Plan::seeded(0x5EED_1616, 16, 3);
    let steps = soak_steps(20);
    restart(plan, steps, steps).finish(panics, "restart soak");
}

#[test]
#[ignore = "soak: cheaters' novel blocks start reaction fronts for 15 s; liveness checks only (CHAOS_SECS, CHAOS_SEED)"]
fn players_keep_up_through_a_reaction_storm() {
    let panics = panics_so_far();
    let plan = Plan { novel: true, ..Plan::seeded(0x5704_4A11, 8, 3) };
    soak(plan, soak_steps(15)).finish(panics, "storm");
}

#[test]
fn a_chain_with_no_rival_lands_whole() {
    let panics = panics_so_far();
    let lobby = Lobby::start(config(8, None));
    let mut chaos = Chaos::new(Plan::new(8, 3, 0), lobby.port);
    chaos.gather();
    let at = chaos.bots[1].pos;
    let cell = (block_coord(at.x), FLAT_HEIGHT - 1, block_coord(at.z));
    let sand = chaos.ground.blocks[4].clone();
    chaos.bots[1].send(cell, chaos.ground.air.clone(), &chaos.ground, 3);
    chaos.bots[1].send(cell, sand.clone(), &chaos.ground, 3);
    chaos.settle();
    chaos.audit();
    let held = chaos.audit.as_ref().and_then(|a| a.cells.get(&cell).cloned());
    chaos.check();
    chaos.disband();
    lobby.server.stop();
    assert_eq!(held, Some(sand), "the place lands on top of the break");
    chaos.finish(panics, "chain");
}

#[test]
fn a_chain_that_loses_its_first_edit_to_a_peer_leaves_the_chainer_out_of_step() {
    let panics = panics_so_far();
    let lobby = Lobby::start(config(7, None));
    let mut chaos = Chaos::new(Plan::new(7, 3, 0), lobby.port);
    chaos.gather();
    let at = chaos.bots[1].pos;
    let cell = (block_coord(at.x), FLAT_HEIGHT - 1, block_coord(at.z));
    let plank = chaos.ground.blocks[3].clone();
    let sand = chaos.ground.blocks[4].clone();
    // The second player wins the cell while the first is not looking.
    chaos.bots[2].send(cell, plank, &chaos.ground, 3);
    while !chaos.bots[2].edits.is_empty() {
        chaos.bots[2].pump(&chaos.ground, &mut chaos.referee);
        thread::sleep(Duration::from_millis(5));
    }
    // The first breaks then places, both against the cell as she last saw it.
    chaos.bots[1].send(cell, chaos.ground.air.clone(), &chaos.ground, 3);
    chaos.bots[1].send(cell, sand, &chaos.ground, 3);
    chaos.settle();
    chaos.audit();
    chaos.check();
    chaos.disband();
    lobby.server.stop();
    chaos.finish(panics, "chain race");
}

#[test]
#[ignore = "BUG: one novel block in the soil starts a reaction front that never settles; the ledger grows ~1000 cells/s without bound"]
fn one_novel_block_settles() {
    let panics = panics_so_far();
    let lobby = Lobby::start(config(1, None));
    let mut chaos = Chaos::new(Plan::new(1, 1, 0), lobby.port);
    chaos.gather();
    let spec = novel_spec(&mut Rng::at(0, 0, 0));
    chaos.bots[0].send((0, FLAT_HEIGHT - 1, 0), spec, &chaos.ground, 1);
    let start = Instant::now();
    let mut early = 0;
    while start.elapsed() < Duration::from_secs(4) {
        chaos.pump_all();
        if start.elapsed() < Duration::from_secs(2) {
            early = chaos.referee.reactions;
        }
        thread::sleep(Duration::from_millis(5));
    }
    let late = chaos.referee.reactions - early;
    let cells = chaos.referee.stirred.len();
    chaos.disband();
    lobby.server.stop();
    chaos.finish(panics, "front");
    assert_eq!(late, 0, "the front still moved {late} cells between 2 s and 4 s, {cells} cells so far");
}

#[test]
fn a_long_join_overlay_is_not_a_quiet_link() {
    const CELLS: i32 = 60_000;
    let path = crate::save::store::test_temp_path("chaos-overlay");
    let flags = persist::Flags { seed: 1, worldgen: WorldgenKind::Flat, terrain: TerrainCfg::default(), warn: false };
    let store = persist::load(&path, &flags).expect("a fresh world").store.expect("a world file");
    let plank = Ground::new(1).blocks[3].clone();
    let edits = (0..CELLS).map(|i| (i % 300, OWN_Y, i / 300, plank.clone())).collect();
    let world = persist::Snapshot {
        seed: 1,
        worldgen: WorldgenKind::Flat,
        terrain: TerrainCfg::default(),
        day: 0.3,
        edits,
        pending: Vec::new(),
    };
    store.write(&world).expect("the world is written");
    let lobby = Lobby::start(config(1, Some(path.clone())));
    let mut conn = Connection::connect("127.0.0.1", lobby.port, "slow", "").expect("joins");
    // Four polls a second stand in for a world fifteen times the size at sixty frames a second.
    let start = Instant::now();
    let mut cells = 0;
    let mut quiet = None;
    while quiet.is_none() && !conn.snapshot_ready() && start.elapsed() < Duration::from_secs(20) {
        for event in conn.poll() {
            match event {
                Incoming::Mutation { .. } => cells += 1,
                Incoming::Interrupted | Incoming::Disconnected { .. } => quiet = quiet.or(Some(cells)),
                _ => {}
            }
        }
        thread::sleep(Duration::from_millis(250));
    }
    drop(conn);
    lobby.server.stop();
    for suffix in ["", ".bak", ".tmp"] {
        let mut name = path.clone().into_os_string();
        name.push(suffix);
        let _ = std::fs::remove_file(name);
    }
    assert_eq!(quiet, None, "the link was called quiet after {cells} of {CELLS} overlay cells, the rest still arriving");
}
