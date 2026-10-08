//! Multiplayer world tests. See `super` for the harness.
//!
//! A [`Peer`] is a headless client: a [`Connection`] and a client [`World`] that applies
//! [`Incoming`] the way `game.rs` does, optimistic edits and their rollbacks included. The
//! server's ledger is what a fresh joiner receives as its overlay.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use voxel_engine::DVec3;

use super::{Lobby, listen};
use crate::block::{AIR, BlockId};
use crate::net::client::{APPLY_BUDGET, Connection, Incoming};
use crate::net::hooks::{self, EditIntent, ServerMod};
use crate::net::persist;
use crate::net::server::Config;
use crate::render_config::RenderConfig;
use crate::save;
use crate::sim::reactions::{CellStore, destructive_pair};
use crate::world::World;
use crate::world::generation::{FLAT_HEIGHT, WorldgenKind};

type Cell = (i32, i32, i32);

/// Inside the server's edit reach (eight metres), with a margin.
const REACH: f64 = 7.0 * crate::math::PER_METER;
/// Shorter than the client's own three-second expiry, so a verdict seen within it came from the server.
const ANSWER: Duration = Duration::from_millis(2500);
/// The server's edit budget per second.
const EDIT_RATE: usize = 20;
/// The server's tool-use budget per second.
const TOOL_RATE: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    Accepted,
    Rejected { restore: bool },
}

struct ToolOutcome {
    reacted: bool,
    cell: Cell,
    cell_spec: String,
    tool_spec: String,
}

struct Peer {
    name: String,
    conn: Connection,
    world: World,
    /// Edits in flight as `game.rs` keeps them: request, cell, and the block before the prediction.
    predicted: HashMap<u32, (Cell, BlockId)>,
    verdicts: HashMap<u32, Verdict>,
    tools: HashMap<u32, ToolOutcome>,
    /// Other players' edits, in arrival order.
    edits: Vec<(Cell, Arc<str>)>,
    /// Snapshot cells: the join overlay, then reaction commits.
    mutations: Vec<(Cell, Arc<str>)>,
    /// How many of `mutations` had arrived when `SnapshotEnd` did.
    overlay: Option<usize>,
    /// Most snapshot cells one poll handed over.
    widest_poll: usize,
    positions: usize,
}

impl Peer {
    /// Join and apply the overlay through `SnapshotEnd`.
    fn join(lobby: &Lobby, name: &str) -> Self {
        let mut peer = Self::connect(lobby, name);
        peer.settle(Duration::from_secs(10), |p| p.overlay.is_some());
        peer
    }

    /// Connected, nothing polled yet.
    fn connect(lobby: &Lobby, name: &str) -> Self {
        let conn = Connection::connect("127.0.0.1", lobby.port, name, "").expect("joins");
        let world = World::with_kind_cfg(conn.seed(), RenderConfig::default(), conn.worldgen(), conn.terrain(), false);
        Self {
            name: name.to_string(),
            conn,
            world,
            predicted: HashMap::new(),
            verdicts: HashMap::new(),
            tools: HashMap::new(),
            edits: Vec::new(),
            mutations: Vec::new(),
            overlay: None,
            widest_poll: 0,
            positions: 0,
        }
    }

    /// One poll, applied like `Game::apply_net_events`.
    fn pump(&mut self) {
        let mut handed = 0;
        for event in self.conn.poll() {
            match event {
                Incoming::Edit { x, y, z, spec } => {
                    let id = save::parse_block(self.world.registry_mut(), &spec);
                    self.world.set_block(x, y, z, id);
                    self.edits.push(((x, y, z), spec));
                }
                Incoming::Mutation { x, y, z, spec } => {
                    let id = save::parse_block(self.world.registry_mut(), &spec);
                    self.world.set_block(x, y, z, id);
                    self.mutations.push(((x, y, z), spec));
                    handed += 1;
                }
                Incoming::EditAccepted { req } => {
                    self.predicted.remove(&req);
                    self.verdicts.insert(req, Verdict::Accepted);
                }
                Incoming::EditRejected { req, restore } => {
                    if let Some(((x, y, z), prev)) = self.predicted.remove(&req)
                        && restore
                    {
                        self.world.set_block(x, y, z, prev);
                    }
                    self.verdicts.insert(req, Verdict::Rejected { restore });
                }
                Incoming::ToolResult { req, reacted, cell, cell_spec, tool_spec } => {
                    if reacted {
                        let id = save::parse_block(self.world.registry_mut(), &cell_spec);
                        self.world.set_block(cell.0, cell.1, cell.2, id);
                    }
                    let (cell_spec, tool_spec) = (cell_spec.to_string(), tool_spec.to_string());
                    self.tools.insert(req, ToolOutcome { reacted, cell, cell_spec, tool_spec });
                }
                Incoming::Position { .. } => self.positions += 1,
                Incoming::Disconnected { reason } => panic!("{} was disconnected: {reason}", self.name),
                _ => {}
            }
        }
        self.widest_poll = self.widest_poll.max(handed);
        if self.overlay.is_none() && self.conn.snapshot_ready() {
            self.overlay = Some(self.mutations.len());
        }
    }

    fn settle(&mut self, timeout: Duration, done: impl FnMut(&Peer) -> bool) {
        settle_peers(std::slice::from_mut(self), timeout, done);
    }

    /// Predict `spec` at `cell` and send it, as `game.rs` breaks and places.
    fn edit(&mut self, cell: Cell, spec: &str) -> u32 {
        let id = save::parse_block(self.world.registry_mut(), spec);
        let prev = self.block(cell);
        self.world.set_block(cell.0, cell.1, cell.2, id);
        let req = self.send(cell, spec);
        self.predicted.insert(req, (cell, prev));
        req
    }

    /// An edit request with no prediction behind it.
    fn send(&mut self, cell: Cell, spec: &str) -> u32 {
        self.conn.send_edit(cell.0, cell.1, cell.2, spec.into()).expect("edit sent")
    }

    fn tool(&mut self, cell: Cell, tool: &str) -> u32 {
        self.conn.send_tool_use(cell.0, cell.1, cell.2, tool.into()).expect("tool use sent")
    }

    fn in_flight(&self, cell: Cell) -> bool {
        self.predicted.values().any(|(c, _)| *c == cell)
    }

    /// The server's verdict on each request.
    fn answers(&mut self, reqs: &[u32]) -> Vec<Verdict> {
        self.settle(ANSWER, |p| reqs.iter().all(|r| p.verdicts.contains_key(r)));
        reqs.iter().map(|r| self.verdicts[r]).collect()
    }

    fn answer(&mut self, req: u32) -> Verdict {
        self.answers(&[req])[0]
    }

    fn tool_answer(&mut self, req: u32) -> &ToolOutcome {
        self.settle(ANSWER, |p| p.tools.contains_key(&req));
        &self.tools[&req]
    }

    /// The client world's block: loaded chunk, else the overlay, else the generator.
    fn block(&self, cell: Cell) -> BlockId {
        CellStore::block_at(&self.world, cell).expect("every cell reads")
    }

    fn view(&self, cell: Cell) -> String {
        self.world.registry().spec(self.block(cell))
    }

    fn generated(&self, cell: Cell) -> String {
        self.world.registry().spec(self.world.terrain().voxel_at(cell.0, cell.1, cell.2))
    }

    fn spec_of(&self, block: BlockId) -> String {
        self.world.registry().spec(block)
    }

    /// The reference reacting pair as specs: the block, and the tool that empties it.
    fn pair(&mut self) -> (String, String) {
        let (a, e) = destructive_pair(self.world.registry_mut());
        (self.spec_of(a), self.spec_of(e))
    }

    /// The overlay this peer joined with. Panics on a cell sent twice.
    fn overlay_map(&self) -> HashMap<Cell, Arc<str>> {
        let n = self.overlay.expect("joined");
        let mut map = HashMap::new();
        for (cell, spec) in &self.mutations[..n] {
            assert!(map.insert(*cell, spec.clone()).is_none(), "{cell:?} is in the overlay twice");
        }
        map
    }

    fn edits_at(&self, cell: Cell) -> Vec<&str> {
        self.edits.iter().filter(|(c, _)| *c == cell).map(|(_, s)| s.as_ref()).collect()
    }

    fn mutated(&self) -> HashSet<Cell> {
        self.mutations.iter().map(|(c, _)| *c).collect()
    }

    /// Where the server judges an edit of `cell` from: a chart's storage cell sits where it embeds.
    fn at(&self, cell: Cell) -> DVec3 {
        crate::space::atlas::embed_cell(self.world.terrain().atlases(), cell)
            .unwrap_or(DVec3::new(cell.0 as f64 + 0.5, cell.1 as f64 + 0.5, cell.2 as f64 + 0.5))
    }

    fn reaches(&self, cell: Cell) -> bool {
        self.conn.spawn().distance(self.at(cell)) <= REACH
    }
}

/// Pump every peer until `done` holds for all of them. Panics on timeout, naming the stragglers.
fn settle_peers(peers: &mut [Peer], timeout: Duration, mut done: impl FnMut(&Peer) -> bool) {
    let deadline = Instant::now() + timeout;
    loop {
        for p in peers.iter_mut() {
            p.pump();
        }
        let pending: Vec<&str> = peers.iter().filter(|p| !done(p)).map(|p| p.name.as_str()).collect();
        if pending.is_empty() {
            return;
        }
        assert!(Instant::now() < deadline, "{pending:?} not done after {timeout:?}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Pump until no request is in flight and nothing has arrived for `quiet`.
fn quiesce(peers: &mut [Peer], quiet: Duration, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    let heard = |peers: &[Peer]| -> usize {
        peers.iter().map(|p| p.edits.len() + p.mutations.len() + p.verdicts.len() + p.tools.len()).sum()
    };
    let mut seen = heard(peers);
    let mut since = Instant::now();
    loop {
        for p in peers.iter_mut() {
            p.pump();
        }
        let now = heard(peers);
        if now != seen || peers.iter().any(|p| !p.predicted.is_empty()) {
            seen = now;
            since = Instant::now();
        } else if since.elapsed() >= quiet {
            return;
        }
        assert!(Instant::now() < deadline, "the world did not settle within {timeout:?}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Every peer's view of `cells`, and of every cell the server has written, comes to equal the
/// server's ledger: a fresh joiner's overlay, else the generated block. The joiner's own world must
/// agree. Peers keep polling for what was already on its way when the joiner was admitted.
fn assert_converged(lobby: &Lobby, peers: &mut [Peer], cells: &[Cell]) -> HashMap<Cell, Arc<str>> {
    let witness = Peer::join(lobby, "witness");
    let ledger = witness.overlay_map();
    let want: Vec<(Cell, String)> = cells
        .iter()
        .copied()
        .chain(ledger.keys().copied())
        .collect::<BTreeSet<Cell>>()
        .into_iter()
        .map(|cell| (cell, ledger.get(&cell).map_or_else(|| witness.generated(cell), |s| s.to_string())))
        .collect();
    let deadline = Instant::now() + ANSWER;
    loop {
        for p in peers.iter_mut() {
            p.pump();
        }
        let mut wrong = Vec::new();
        for (cell, want) in &want {
            for p in peers.iter().chain(std::iter::once(&witness)) {
                let got = p.view(*cell);
                if got != *want {
                    wrong.push(format!("{} at {cell:?}: {got}, server {want}", p.name));
                }
            }
        }
        if wrong.is_empty() {
            return ledger;
        }
        assert!(Instant::now() < deadline, "{} views of {} cells diverge:\n{}", wrong.len(), want.len(), wrong.join("\n"));
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Splitmix64: the same choices on every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn pick<T: Clone>(&mut self, from: &[T]) -> T {
        from[(self.next() % from.len() as u64) as usize].clone()
    }
}

fn diffusion_lobby() -> Lobby {
    Lobby::start(Config { seed: 4242, worldgen: WorldgenKind::Diffusion, ..Config::default() })
}

fn file_lobby(path: &Path) -> Lobby {
    Lobby::start(Config { seed: 1, worldgen: WorldgenKind::Flat, world: Some(path.to_path_buf()), ..Config::default() })
}

fn peers(lobby: &Lobby, names: &[&str]) -> Vec<Peer> {
    names.iter().map(|n| Peer::join(lobby, n)).collect()
}

/// Cells in the box `centre ± r` that every peer reaches from its spawn.
fn reachable(peers: &[Peer], centre: Cell, r: Cell) -> Vec<Cell> {
    let mut cells = Vec::new();
    for dx in -r.0..=r.0 {
        for dy in -r.1..=r.1 {
            for dz in -r.2..=r.2 {
                let cell = (centre.0 + dx, centre.1 + dy, centre.2 + dz);
                if peers.iter().all(|p| p.reaches(cell)) {
                    cells.push(cell);
                }
            }
        }
    }
    cells
}

/// Soil, grass, and the air above them, around the flat spawns.
fn flat_cells(peers: &[Peer]) -> Vec<Cell> {
    reachable(peers, (-1, FLAT_HEIGHT - 1, -3), (2, 1, 1))
}

/// The ground under the first spawn and the air above it, in storage cells (a chart's up is +Y).
fn chart_cells(peers: &[Peer]) -> Vec<Cell> {
    let spawn = peers[0].conn.spawn();
    let s = peers[0].world.terrain().atlases().iter().find_map(|a| a.storage_of(spawn)).expect("spawn on a chart");
    let mut ground = (s[0] as i32, s[1] as i32, s[2] as i32);
    for _ in 0..64 {
        if peers[0].block(ground) != AIR {
            break;
        }
        ground.1 -= 1;
    }
    assert_ne!(peers[0].block(ground), AIR, "ground under the spawn");
    reachable(peers, ground, (2, 1, 2))
}

/// Air and every block generated in `cells`, as specs.
fn specs_in(peer: &Peer, cells: &[Cell]) -> Vec<Arc<str>> {
    let mut specs: Vec<Arc<str>> = vec!["air".into()];
    for &cell in cells {
        let spec: Arc<str> = peer.generated(cell).into();
        if !specs.contains(&spec) {
            specs.push(spec);
        }
    }
    specs
}

/// The flat world's grass, soil and rock.
fn flat_specs(peer: &Peer) -> (String, String, String) {
    let grass = peer.generated((0, FLAT_HEIGHT - 1, 0));
    let soil = peer.generated((0, FLAT_HEIGHT - 2, 0));
    let rock = peer.generated((0, FLAT_HEIGHT - 6, 0));
    assert!(grass != "air" && soil != "air" && rock != "air" && grass != soil && soil != rock);
    (grass, soil, rock)
}

/// `rounds` rounds in which every peer sends three random edits before anyone polls. Chained, each
/// edit is a break and then a placement on one cell; unchained, a peer never edits a cell its own
/// earlier edit is still in flight on.
fn burst(peers: &mut [Peer], cells: &[Cell], specs: &[Arc<str>], rounds: usize, chained: bool, seed: u64) {
    let mut rng = Rng(seed);
    for _ in 0..rounds {
        for p in peers.iter_mut() {
            for _ in 0..3 {
                let cell = rng.pick(cells);
                let spec = rng.pick(specs);
                if chained {
                    p.edit(cell, "air");
                    p.edit(cell, &spec);
                } else if !p.in_flight(cell) {
                    p.edit(cell, &spec);
                }
            }
        }
        for p in peers.iter_mut() {
            p.pump();
        }
        std::thread::sleep(Duration::from_millis(15));
    }
}

fn converge_after_a_burst(lobby: &Lobby, chained: bool, seed: u64) {
    let mut ps = peers(lobby, &["ada", "bob", "cyd"]);
    let cells = if ps[0].conn.worldgen() == WorldgenKind::Flat { flat_cells(&ps) } else { chart_cells(&ps) };
    assert!(cells.len() >= 24, "only {} cells in reach of every spawn", cells.len());
    let specs = specs_in(&ps[0], &cells);
    assert!(specs.len() >= 2, "{specs:?}");
    burst(&mut ps, &cells, &specs, 10, chained, seed);
    quiesce(&mut ps, Duration::from_millis(300), Duration::from_secs(4));
    let accepted: usize = ps.iter().map(|p| p.verdicts.values().filter(|v| **v == Verdict::Accepted).count()).sum();
    let rejected: usize = ps.iter().map(|p| p.verdicts.len()).sum::<usize>() - accepted;
    assert!(accepted >= 30 && rejected > 0, "{accepted} accepted, {rejected} rejected");
    assert_converged(lobby, &mut ps, &cells);
}

/// A world file at `path` holding `cells`.
fn write_world(path: &Path, cells: &[(Cell, String)]) {
    remove_world(path);
    let flags = persist::Flags { seed: 1, worldgen: WorldgenKind::Flat, terrain: Default::default(), warn: false };
    let store = persist::load(path, &flags).expect("a fresh world").store.expect("a file to write");
    let edits = cells.iter().map(|((x, y, z), s)| (*x, *y, *z, Arc::from(s.as_str()))).collect();
    let snap = persist::Snapshot { seed: 1, worldgen: WorldgenKind::Flat, terrain: Default::default(), day: 0.3, edits, pending: Vec::new() };
    store.write(&snap).expect("world written");
}

fn saved_edits(path: &Path) -> Vec<(i32, i32, i32, String)> {
    let flags = persist::Flags { seed: 1, worldgen: WorldgenKind::Flat, terrain: Default::default(), warn: false };
    persist::load(path, &flags).expect("the saved world loads").edits
}

fn remove_world(path: &Path) {
    for suffix in ["", ".bak", ".tmp"] {
        let mut name = path.as_os_str().to_os_string();
        name.push(suffix);
        let _ = std::fs::remove_file(PathBuf::from(name));
    }
}

#[test]
fn places_and_breaks_reach_every_client() {
    let lobby = Lobby::flat();
    let mut ps = peers(&lobby, &["ada", "bob", "cyd"]);
    let (grass, soil, rock) = flat_specs(&ps[0]);
    let ground = (-1, FLAT_HEIGHT - 1, -3);
    let above = (0, FLAT_HEIGHT, -4);
    let floating = (-2, FLAT_HEIGHT + 1, -2);
    assert_eq!(ps[0].view(ground), grass);
    assert_eq!(ps[0].view(above), "air");

    let reqs = [ps[0].edit(ground, "air"), ps[1].edit(above, &rock), ps[2].edit(floating, &soil)];
    for (p, req) in ps.iter_mut().zip(reqs) {
        assert_eq!(p.answer(req), Verdict::Accepted, "{}", p.name);
    }
    settle_peers(&mut ps, ANSWER, |p| p.edits.len() == 2);
    let made = [(ground, "air"), (above, rock.as_str()), (floating, soil.as_str())];
    for (i, p) in ps.iter().enumerate() {
        for (by, &(cell, spec)) in made.iter().enumerate() {
            assert_eq!(p.view(cell), spec, "{} at {cell:?}", p.name);
            let heard = p.edits_at(cell);
            if by == i {
                assert!(heard.is_empty(), "{} hears its own edit back: {heard:?}", p.name);
            } else {
                assert_eq!(heard, [spec], "{} hears {cell:?} once", p.name);
            }
        }
    }

    // Someone who saw the cell broken fills it again.
    let refill = ps[2].edit(ground, &grass);
    assert_eq!(ps[2].answer(refill), Verdict::Accepted);
    settle_peers(&mut ps[..2], ANSWER, |p| p.edits_at(ground).len() == usize::from(p.name != "ada") + 1);
    let ledger = assert_converged(&lobby, &mut ps, &[ground, above, floating]);
    assert_eq!(ledger.len(), 3, "{ledger:?}");
    assert!(ps.iter().all(|p| p.mutations.is_empty()), "no reaction stirred");
}

#[test]
fn racing_edits_on_one_cell_have_one_winner_and_the_loser_rolls_back() {
    let lobby = Lobby::flat();
    let mut ps = peers(&lobby, &["ada", "bob", "cyd"]);
    let (_, soil, rock) = flat_specs(&ps[0]);
    let specs = [rock.as_str(), soil.as_str()];
    let cells: Vec<Cell> = (0..6).map(|i| (-3 + i, FLAT_HEIGHT, -2)).filter(|&c| ps.iter().all(|p| p.reaches(c))).collect();
    assert!(cells.len() >= 4);
    for (n, &cell) in cells.iter().enumerate() {
        // Both place into the same empty cell before either polls; who sends first alternates.
        let (first, second) = if n % 2 == 0 { (0, 1) } else { (1, 0) };
        let a = ps[first].edit(cell, specs[first]);
        let b = ps[second].edit(cell, specs[second]);
        let winner = match (ps[first].answer(a), ps[second].answer(b)) {
            (Verdict::Accepted, Verdict::Rejected { restore: false }) => first,
            (Verdict::Rejected { restore: false }, Verdict::Accepted) => second,
            other => panic!("{cell:?}: want one winner and an unrestored loser, got {other:?}"),
        };
        assert_eq!(ps[1 - winner].edits_at(cell), [specs[winner]], "the loser hears the winner");
        assert!(ps[winner].edits_at(cell).is_empty());
        settle_peers(&mut ps[2..], ANSWER, |p| !p.edits_at(cell).is_empty());
        assert_eq!(ps[2].edits_at(cell), [specs[winner]], "a bystander hears only the winner");
        for p in &ps {
            assert_eq!(p.view(cell), specs[winner], "{} at {cell:?}", p.name);
        }
    }

    // A loser that never polled: bob's placement lands first, ada still expects revision 0.
    let cell = (-1, FLAT_HEIGHT, -4);
    let won = ps[1].edit(cell, &soil);
    assert_eq!(ps[1].answer(won), Verdict::Accepted);
    let lost = ps[0].edit(cell, &rock);
    assert_eq!(ps[0].answer(lost), Verdict::Rejected { restore: false });
    assert_eq!(ps[0].view(cell), soil, "the loser shows the winner's block, not its prediction");
    let mut all = cells.clone();
    all.push(cell);
    assert_converged(&lobby, &mut ps, &all);
}

#[test]
#[ignore = "BUG: a break-then-place chain whose break loses to a peer keeps the placement on the server but shows air"]
fn a_chain_that_loses_its_first_edit_to_a_peer_converges() {
    let lobby = Lobby::flat();
    let mut ps = peers(&lobby, &["ada", "bob"]);
    let (_, _, rock) = flat_specs(&ps[0]);
    let cell = (-1, FLAT_HEIGHT - 1, -3);
    // Bob breaks the grass first. Ada has not heard: she breaks it too, then places rock there.
    let won = ps[1].edit(cell, "air");
    assert_eq!(ps[1].answer(won), Verdict::Accepted);
    let broke = ps[0].edit(cell, "air");
    let placed = ps[0].edit(cell, &rock);
    let verdicts = ps[0].answers(&[broke, placed]);
    assert_eq!(verdicts[0], Verdict::Rejected { restore: false }, "bob's break won the cell");
    assert_converged(&lobby, &mut ps, &[cell]);
}

#[test]
fn edits_over_the_rate_budget_are_answered_and_rolled_back() {
    let lobby = Lobby::flat();
    let mut ps = peers(&lobby, &["ada", "bob"]);
    let (_, _, rock) = flat_specs(&ps[0]);
    let cells: Vec<Cell> = reachable(&ps, (-1, FLAT_HEIGHT + 2, -3), (3, 1, 3)).into_iter().take(2 * EDIT_RATE).collect();
    assert_eq!(cells.len(), 2 * EDIT_RATE);
    let reqs: Vec<u32> = cells.iter().map(|&c| ps[0].edit(c, &rock)).collect();
    let verdicts = ps[0].answers(&reqs);
    // Every answer was stamped on the server before it arrived here.
    let answered = Instant::now();
    assert!(verdicts[..EDIT_RATE].iter().all(|v| *v == Verdict::Accepted), "the first {EDIT_RATE} fit: {verdicts:?}");
    assert!(verdicts.iter().all(|v| matches!(v, Verdict::Accepted | Verdict::Rejected { restore: true })));
    let refused: Vec<Cell> = cells.iter().zip(&verdicts).filter(|(_, v)| **v != Verdict::Accepted).map(|(c, _)| *c).collect();
    assert!(!refused.is_empty(), "a burst past the budget is refused in part");
    for &cell in &refused {
        assert_eq!(ps[0].view(cell), "air", "the refused placement at {cell:?} rolls back");
    }
    let accepted = cells.len() - refused.len();
    settle_peers(&mut ps[1..], ANSWER, |p| p.edits.len() == accepted);
    assert!(refused.iter().all(|&c| ps[1].edits_at(c).is_empty()), "refused edits are not relayed");

    // The window empties a second after the burst.
    std::thread::sleep(Duration::from_millis(1010).saturating_sub(answered.elapsed()));
    let retry = ps[0].edit(refused[0], &rock);
    assert_eq!(ps[0].answer(retry), Verdict::Accepted, "the budget refills");
    let ledger = assert_converged(&lobby, &mut ps, &cells);
    assert_eq!(ledger.len(), accepted + 1);
}

#[test]
#[ignore = "BUG: two refused edits on one cell roll back in arrival order, leaving the first edit's prediction"]
fn a_refused_chain_on_one_cell_rolls_back_to_the_server_block() {
    let lobby = Lobby::flat();
    let mut ps = peers(&lobby, &["ada"]);
    let (grass, _, rock) = flat_specs(&ps[0]);
    // Out of reach: the server refuses the break and the placement that follows it.
    let cell = (40, FLAT_HEIGHT - 1, 40);
    assert!(!ps[0].reaches(cell));
    let reqs = [ps[0].edit(cell, "air"), ps[0].edit(cell, &rock)];
    let verdicts = ps[0].answers(&reqs);
    assert!(verdicts.iter().all(|v| matches!(v, Verdict::Rejected { .. })), "{verdicts:?}");
    assert_eq!(ps[0].view(cell), grass, "both predictions roll back to the server's grass");
}

/// Holds every edit before allowing it, as a slow mod or a stalled server would.
struct Stall(Duration);

impl ServerMod for Stall {
    fn id(&self) -> &'static str {
        "stall"
    }

    fn validate_edit(&mut self, _: &EditIntent) -> hooks::Verdict {
        std::thread::sleep(self.0);
        hooks::Verdict::Allow
    }
}

#[test]
#[ignore = "BUG: an edit accepted after the client's 3 s expiry stays on the server but is rolled back on the sender"]
fn an_edit_accepted_after_the_client_gave_up_converges() {
    // Past the client's three-second expiry.
    let stall = Stall(Duration::from_millis(3500));
    let lobby = Lobby::start(Config { seed: 1, worldgen: WorldgenKind::Flat, hooks: vec![Box::new(stall)], ..Config::default() });
    let mut ps = peers(&lobby, &["ada", "bob"]);
    let (_, _, rock) = flat_specs(&ps[0]);
    let cell = (-1, FLAT_HEIGHT, -3);
    let req = ps[0].edit(cell, &rock);
    ps[0].settle(Duration::from_secs(4), |p| p.verdicts.contains_key(&req));
    settle_peers(&mut ps[1..], Duration::from_secs(2), |p| !p.edits.is_empty());
    assert_eq!(ps[1].edits_at(cell), [rock.as_str()], "the server accepted it");
    assert_converged(&lobby, &mut ps, &[cell]);
}

#[test]
fn out_of_reach_and_malformed_edits_are_refused() {
    let lobby = Lobby::flat();
    let mut ps = peers(&lobby, &["ada", "bob"]);
    let (grass, _, rock) = flat_specs(&ps[0]);
    let far = (30, FLAT_HEIGHT - 1, 30);
    assert!(!ps[0].reaches(far));
    let req = ps[0].edit(far, "air");
    assert_eq!(ps[0].answer(req), Verdict::Rejected { restore: true });
    assert_eq!(ps[0].view(far), grass, "the refused break rolls back");

    let near = (-2, FLAT_HEIGHT, -3);
    let mut reqs = vec![ps[0].send(near, "c:zz"), ps[0].send(near, "c:00"), ps[0].send(near, "")];
    for cell in [(i32::MAX, i32::MAX, i32::MAX), (i32::MIN, i32::MIN, i32::MIN), (-1, i32::MIN, -3)] {
        reqs.push(ps[0].send(cell, &rock));
    }
    for (v, req) in ps[0].answers(&reqs).into_iter().zip(&reqs) {
        assert!(matches!(v, Verdict::Rejected { .. }), "request {req}: {v:?}");
    }

    // The server still serves, and reach follows an accepted teleport.
    ps[0].conn.send_teleport(DVec3::new(30.5, f64::from(FLAT_HEIGHT) + 2.0, 30.5));
    ps[0].settle(ANSWER, |p| p.positions > 0);
    let retry = ps[0].edit(far, "air");
    assert_eq!(ps[0].answer(retry), Verdict::Accepted, "in reach after the teleport");
    settle_peers(&mut ps[1..], ANSWER, |p| !p.edits.is_empty());
    assert_eq!(ps[1].edits, [(far, Arc::from("air"))], "only the accepted break is relayed");
    let ledger = assert_converged(&lobby, &mut ps, &[far, near]);
    assert_eq!(ledger.keys().collect::<Vec<_>>(), [&far]);
}

#[test]
fn a_tool_use_runs_the_law_on_the_server_and_every_client_sees_the_cell() {
    let lobby = Lobby::flat();
    let mut ps = peers(&lobby, &["ada", "bob", "cyd"]);
    let (block, tool) = ps[0].pair();
    let cell = (-1, FLAT_HEIGHT + 2, -3);
    let placed = ps[1].edit(cell, &block);
    assert_eq!(ps[1].answer(placed), Verdict::Accepted);
    settle_peers(&mut ps, ANSWER, |p| p.view(cell) == block);

    let req = ps[0].tool(cell, &tool);
    let (reacted, at, cell_spec, tool_spec) = {
        let out = ps[0].tool_answer(req);
        (out.reacted, out.cell, out.cell_spec.clone(), out.tool_spec.clone())
    };
    assert!(reacted, "the reference pair reacts");
    assert_eq!(at, cell);
    let registry = ps[0].world.registry_mut();
    let (after, held) = (registry.parse_spec(&cell_spec).expect("cell spec"), registry.parse_spec(&tool_spec).expect("tool spec"));
    assert_eq!(registry.configuration(after).len(), 3, "one element left the block");
    assert_eq!(registry.configuration(held).len(), 5, "and joined the tool");
    assert_eq!(ps[0].view(cell), cell_spec);
    settle_peers(&mut ps[1..], ANSWER, |p| p.edits_at(cell).contains(&cell_spec.as_str()));
    assert_eq!(ps[0].edits_at(cell), [block.as_str()], "ada hears bob's placement as an edit, and her own use only as its result");

    // Refused, each answered with the tool unchanged and nothing written: a void tool, a cell
    // out of reach, and a revision that is gone (bob replaced the block; cyd has not polled).
    let void = ps[0].tool(cell, "air");
    assert!(!ps[0].tool_answer(void).reacted);
    let far = (30, FLAT_HEIGHT - 1, 30);
    let reach = ps[0].tool(far, &tool);
    let out = ps[0].tool_answer(reach);
    assert!(!out.reacted);
    assert_eq!(out.tool_spec, tool);
    let replaced = ps[1].edit(cell, &block);
    assert_eq!(ps[1].answer(replaced), Verdict::Accepted);
    let stale = ps[2].tool(cell, &tool);
    let out = ps[2].tool_answer(stale);
    assert!(!out.reacted, "a tool aimed at a revision that is gone does nothing");
    assert_eq!(out.tool_spec, tool);
    quiesce(&mut ps, Duration::from_millis(300), Duration::from_secs(3));
    let ledger = assert_converged(&lobby, &mut ps, &[cell, far]);
    assert_eq!(ledger.get(&cell).map(|s| s.as_ref()), Some(block.as_str()));
    assert!(!ledger.contains_key(&far));
}

#[test]
fn tool_uses_over_the_budget_are_answered() {
    let lobby = Lobby::flat();
    let mut ps = peers(&lobby, &["ada", "bob"]);
    let (_, tool) = ps[0].pair();
    let cells: Vec<Cell> = reachable(&ps, (-1, FLAT_HEIGHT - 1, -3), (3, 0, 3)).into_iter().take(2 * TOOL_RATE).collect();
    assert_eq!(cells.len(), 2 * TOOL_RATE);
    let reqs: Vec<u32> = cells.iter().map(|&c| ps[0].tool(c, &tool)).collect();
    ps[0].settle(ANSWER, |p| reqs.iter().all(|r| p.tools.contains_key(r)));
    for (n, (req, &cell)) in reqs.iter().zip(&cells).enumerate() {
        let out = &ps[0].tools[req];
        assert_eq!(out.cell, cell);
        if n >= TOOL_RATE {
            assert!(!out.reacted, "use {n} is past the budget");
        }
        if !out.reacted {
            assert_eq!(out.tool_spec, tool, "a refusal hands the tool back unchanged");
        }
    }
    quiesce(&mut ps, Duration::from_millis(300), Duration::from_secs(3));
    assert_converged(&lobby, &mut ps, &cells);
}

#[test]
#[ignore = "BUG: a tool use refused over the budget names an empty cell spec for an unedited cell"]
fn a_tool_use_refused_over_the_budget_names_the_cell() {
    let lobby = Lobby::flat();
    let mut ps = peers(&lobby, &["ada"]);
    let (_, tool) = ps[0].pair();
    let cell = (-2, FLAT_HEIGHT + 3, -3);
    let reqs: Vec<u32> = (0..2 * TOOL_RATE).map(|_| ps[0].tool(cell, &tool)).collect();
    ps[0].settle(ANSWER, |p| reqs.iter().all(|r| p.tools.contains_key(r)));
    for req in reqs {
        let out = &ps[0].tools[&req];
        assert!(!out.reacted, "a tool on air does nothing");
        assert_eq!(out.cell_spec, "air", "request {req} names the cell as it is");
    }
}

#[test]
fn a_tool_use_racing_a_break_has_one_winner() {
    let lobby = Lobby::flat();
    let mut ps = peers(&lobby, &["ada", "bob"]);
    let (block, tool) = ps[0].pair();
    let cells: Vec<Cell> = (0..4).map(|i| (-3 + 2 * i, FLAT_HEIGHT + 2, -3)).collect();
    let placed: Vec<u32> = cells.iter().map(|&c| ps[0].edit(c, &block)).collect();
    assert!(ps[0].answers(&placed).iter().all(|v| *v == Verdict::Accepted));
    settle_peers(&mut ps[1..], ANSWER, |p| p.edits.len() == cells.len());
    let mut tool_wins = 0;
    for (n, &cell) in cells.iter().enumerate() {
        let (used, broke) = if n % 2 == 0 {
            let used = ps[0].tool(cell, &tool);
            (used, ps[1].edit(cell, "air"))
        } else {
            let broke = ps[1].edit(cell, "air");
            (ps[0].tool(cell, &tool), broke)
        };
        let reacted = ps[0].tool_answer(used).reacted;
        match (reacted, ps[1].answer(broke)) {
            (true, Verdict::Rejected { restore: false }) => tool_wins += 1,
            (false, Verdict::Accepted) => {}
            other => panic!("{cell:?}: want exactly one winner, got {other:?}"),
        }
    }
    quiesce(&mut ps, Duration::from_millis(300), Duration::from_secs(3));
    let ledger = assert_converged(&lobby, &mut ps, &cells);
    let air = cells.iter().filter(|c| ledger[c].as_ref() == "air").count();
    assert_eq!(air, cells.len() - tool_wins, "each break that won left air");
}

#[test]
#[ignore = "BUG: a break sent before a tool use's result is accepted on the server, but the result overwrites the predicted air"]
fn a_break_right_after_a_tool_use_converges() {
    let lobby = Lobby::flat();
    let mut ps = peers(&lobby, &["ada", "bob"]);
    let (block, tool) = ps[0].pair();
    let cell = (-1, FLAT_HEIGHT + 2, -3);
    let placed = ps[0].edit(cell, &block);
    assert_eq!(ps[0].answer(placed), Verdict::Accepted);
    // The tool reacts; the break, sent before its result, expects the revision the tool commits.
    let used = ps[0].tool(cell, &tool);
    let broke = ps[0].edit(cell, "air");
    assert!(ps[0].tool_answer(used).reacted);
    assert_eq!(ps[0].answer(broke), Verdict::Accepted);
    settle_peers(&mut ps[1..], ANSWER, |p| p.edits_at(cell).len() == 3);
    assert_converged(&lobby, &mut ps, &[cell]);
}

#[test]
fn server_reactions_reach_every_client() {
    let lobby = Lobby::flat();
    let mut ps = peers(&lobby, &["ada", "bob", "cyd"]);
    let (a, e) = ps[0].pair();
    // The law's A is the lower cell of a contact.
    let lo = (-1, FLAT_HEIGHT + 2, -3);
    let hi = (0, FLAT_HEIGHT + 2, -3);
    let first = ps[0].edit(lo, &a);
    assert_eq!(ps[0].answer(first), Verdict::Accepted);
    let second = ps[1].edit(hi, &e);
    assert_eq!(ps[1].answer(second), Verdict::Accepted);
    settle_peers(&mut ps, Duration::from_secs(3), |p| p.mutated().contains(&lo) && p.mutated().contains(&hi));
    quiesce(&mut ps, Duration::from_millis(400), Duration::from_secs(4));
    assert!(ps.iter().all(|p| p.overlay == Some(0)), "everyone joined an empty world");
    let cells = ps[2].mutated();
    for p in &ps[..2] {
        assert_eq!(p.mutated(), cells, "{} hears the same reaction cells", p.name);
    }
    let ledger = assert_converged(&lobby, &mut ps, &[lo, hi]);
    assert_ne!(ledger[&lo].as_ref(), a, "the reaction changed the placed block");
    assert_ne!(ledger[&hi].as_ref(), e);
}

#[test]
fn a_late_joiner_gets_the_whole_overlay_before_snapshot_end() {
    let lobby = Lobby::flat();
    let mut ps = peers(&lobby, &["ada", "bob", "cyd"]);
    let (grass, soil, rock) = flat_specs(&ps[0]);
    let cells = flat_cells(&ps);
    let mut want: HashMap<Cell, String> = HashMap::new();
    let mut rng = Rng(7);
    for n in 0..15 {
        let cell = rng.pick(&cells);
        let spec = [grass.as_str(), soil.as_str(), rock.as_str(), "air"][n % 4];
        let req = ps[n % 3].edit(cell, spec);
        assert_eq!(ps[n % 3].answer(req), Verdict::Accepted);
        want.insert(cell, spec.to_string());
        settle_peers(&mut ps, ANSWER, |p| p.view(cell) == spec);
    }
    quiesce(&mut ps, Duration::from_millis(200), Duration::from_secs(3));

    let mut late = Peer::connect(&lobby, "late");
    late.settle(Duration::from_secs(5), |p| p.overlay.is_some());
    let got: HashMap<Cell, String> = late.overlay_map().into_iter().map(|(c, s)| (c, s.to_string())).collect();
    assert_eq!(got, want, "the overlay is every edited cell with its last content");
    assert_eq!(late.mutations.len(), want.len(), "nothing after SnapshotEnd");
    assert!(late.edits.is_empty(), "the overlay arrives as world state, not as edits");
    let after = listen(std::slice::from_mut(&mut late.conn), Duration::from_millis(100));
    assert!(!after[0].iter().any(|e| matches!(e, Incoming::Mutation { .. } | Incoming::Edit { .. })));
    for (&cell, spec) in &want {
        assert_eq!(&late.view(cell), spec);
    }
}

#[test]
fn an_overlay_bigger_than_one_poll_arrives_whole_over_several_polls() {
    let path = crate::save::store::test_temp_path("mp-overlay");
    let n = 2 * APPLY_BUDGET + 900;
    // The ground under an 80 × 80 patch, rock or air, top layer first.
    let file: Vec<(Cell, String)> = (0..n as i32)
        .map(|i| ((i % 80 - 40, FLAT_HEIGHT - 1 - i / 6400, i / 80 % 80 - 40), if i % 3 == 0 { "rock" } else { "air" }.to_string()))
        .collect();
    let rock = {
        let world = World::with_kind_cfg(1, RenderConfig::default(), WorldgenKind::Flat, Default::default(), false);
        world.registry().spec(world.terrain().voxel_at(0, 0, 0))
    };
    let file: Vec<(Cell, String)> = file.into_iter().map(|(c, s)| (c, if s == "rock" { rock.clone() } else { s })).collect();
    write_world(&path, &file);
    let lobby = file_lobby(&path);
    let mut host = Peer::join(&lobby, "host");
    assert_eq!(host.overlay, Some(n));
    assert!(host.widest_poll <= APPLY_BUDGET);

    // Two joiners bootstrap at once while the host rewrites a cell of the overlay.
    let mut late = vec![Peer::connect(&lobby, "ada"), Peer::connect(&lobby, "bob")];
    let (target, _) = file.iter().find(|(c, s)| s == "air" && host.reaches(*c)).expect("an air cell in reach").clone();
    let rewrite = host.edit(target, &rock);
    assert_eq!(host.answer(rewrite), Verdict::Accepted);
    settle_peers(&mut late, Duration::from_secs(5), |p| p.overlay.is_some());
    settle_peers(&mut late, ANSWER, |p| p.view(target) == rock);
    for p in &late {
        assert_eq!(p.overlay, Some(n), "{}", p.name);
        assert_eq!(p.mutations.len(), n, "{}: nothing after SnapshotEnd", p.name);
        assert!(p.widest_poll <= APPLY_BUDGET, "{} took {} cells in one poll", p.name, p.widest_poll);
        let map = p.overlay_map();
        for (cell, spec) in &file {
            if *cell == target {
                continue;
            }
            assert_eq!(map.get(cell).map(|s| s.as_ref()), Some(spec.as_str()), "{} overlay at {cell:?}", p.name);
            assert_eq!(p.view(*cell), *spec, "{} at {cell:?}", p.name);
        }
    }
    drop(late);
    drop(host);
    lobby.server.stop();
    remove_world(&path);
}

#[test]
fn no_op_edits_are_answered_relayed_and_stay_out_of_the_file() {
    let path = crate::save::store::test_temp_path("mp-noop");
    remove_world(&path);
    let lobby = file_lobby(&path);
    let mut ps = peers(&lobby, &["ada", "bob"]);
    let (grass, _, rock) = flat_specs(&ps[0]);
    let sky = (-2, FLAT_HEIGHT + 1, -3);
    let lawn = (-1, FLAT_HEIGHT - 1, -3);
    let kept = (0, FLAT_HEIGHT, -3);
    let reqs = [ps[0].edit(sky, "air"), ps[0].edit(lawn, &grass), ps[0].edit(kept, &rock)];
    assert!(ps[0].answers(&reqs).iter().all(|v| *v == Verdict::Accepted));
    settle_peers(&mut ps[1..], ANSWER, |p| p.edits.len() == 3);
    assert_eq!(ps[1].edits_at(sky), ["air"], "a no-op is still relayed");
    assert_eq!(ps[1].edits_at(lawn), [grass.as_str()]);
    // The no-op took a revision; bob heard it, so his expectation is current.
    let over = ps[1].edit(sky, &rock);
    assert_eq!(ps[1].answer(over), Verdict::Accepted);
    let ledger = assert_converged(&lobby, &mut ps, &[sky, lawn, kept]);
    assert_eq!(ledger.len(), 3);

    // Put back the generated air: a natural cell again, which the file leaves out.
    let back = ps[0].edit(sky, "air");
    assert_eq!(ps[0].answer(back), Verdict::Accepted);
    lobby.server.save_now();
    assert_eq!(saved_edits(&path), vec![(kept.0, kept.1, kept.2, rock.clone())], "only the real change is saved");
    drop(ps);
    lobby.server.stop();
    remove_world(&path);
}

#[test]
fn every_client_converges_after_a_burst_on_a_flat_world() {
    converge_after_a_burst(&Lobby::flat(), false, 11);
}

#[test]
fn every_client_converges_after_a_burst_on_a_diffusion_world() {
    converge_after_a_burst(&diffusion_lobby(), false, 12);
}

#[test]
#[ignore = "BUG: break-then-place chains diverge after a refusal or a lost race (rollback restores a prediction)"]
fn every_client_converges_after_a_chained_burst() {
    converge_after_a_burst(&Lobby::flat(), true, 13);
}
