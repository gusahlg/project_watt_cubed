//! The reaction-event scheduler: turns world events into local material interactions, evaluated in
//! generations (every reaction of a generation reads the same state; mutations commit together, in
//! position order), budgeted per tick. Nothing here runs because a chunk loaded, meshed or saved —
//! only gameplay hands in events. In multiplayer the server owns the scheduler and broadcasts the
//! committed mutations; a client never evaluates reactions itself.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use material::{interact_many, EventKind, Law};

use crate::block::{BlockId, BlockRegistry, AIR};
use crate::world::World;

use super::Tick;

/// A world position of one voxel.
pub type Pos = (i32, i32, i32);

/// A gameplay event that may cause local reactions.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MaterialEvent {
    /// The voxel the event happened to.
    pub at: Pos,
    /// What happened.
    pub kind: EventKind,
}

/// What the scheduler needs from the world: read/write cells and intern configurations. Keeping this a
/// trait makes the generation semantics testable on a plain map.
pub trait CellStore {
    /// The material at a position. `None` for unloaded space, which never reacts.
    fn block_at(&self, pos: Pos) -> Option<BlockId>;
    /// Overwrite a cell; returns the previous id, or `None` when the store refused the write (nothing
    /// changed — the scheduler then commits nothing for that cell).
    fn set_block(&mut self, pos: Pos, id: BlockId) -> Option<BlockId>;
    /// The material table.
    fn registry(&self) -> &BlockRegistry;
    /// The material table, for interning results.
    fn registry_mut(&mut self) -> &mut BlockRegistry;
}

/// One committed change: the scheduler's output (an ordinary overlay edit attributed to the world).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Mutation {
    /// Where.
    pub pos: Pos,
    /// Before.
    pub from: BlockId,
    /// After.
    pub to: BlockId,
}

/// Budget per generation and per tick, so a rule cannot cascade unboundedly in one frame. Events
/// beyond the budget are not dropped: they wait, oldest first, for the next generation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Budget {
    /// Events evaluated per generation (the rest wait for the next generation).
    pub events_per_generation: usize,
    /// Generations run per tick.
    pub generations_per_tick: u32,
}

impl Budget {
    /// Provisional defaults: small, so gameplay stays responsive while the law is explored.
    pub const DEFAULT: Budget = Budget {
        events_per_generation: 256,
        generations_per_tick: 2,
    };
}

/// Distinct pending events a scheduler holds before it refuses more (a memory guard for a law that
/// never comes to rest on an infinite world; the count of refusals is the `dropped` gauge).
pub const DEFAULT_CAPACITY: usize = 16_384;

const FACES: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

/// Place → `NewContact` at the placed cell. The one gameplay place rule.
pub fn on_placed(sched: &mut ReactionScheduler, at: Pos) {
    sched.push(MaterialEvent {
        at,
        kind: EventKind::NewContact,
    });
}

/// Break → `ExternallyChanged` on the six neighbours of the broken cell. The one gameplay break rule.
pub fn on_broken(sched: &mut ReactionScheduler, at: Pos) {
    for f in FACES {
        sched.push(MaterialEvent {
            at: (at.0 + f.0, at.1 + f.1, at.2 + f.2),
            kind: EventKind::ExternallyChanged,
        });
    }
}

/// `EventKind` from its `repr(u8)` value (the queue keys on the byte so it orders without `Ord`).
fn kind_from_u8(k: u8) -> EventKind {
    const KINDS: [EventKind; 4] = [
        EventKind::Moved,
        EventKind::NewContact,
        EventKind::Collision,
        EventKind::ExternallyChanged,
    ];
    KINDS[k as usize]
}

/// The scheduler state: pending events (deduplicated per position and kind) and counters.
///
/// The queue is keyed by `(arrival generation, position, kind)`: a generation evaluates the OLDEST
/// events first and breaks ties in position order. Follow-ups of a generation arrive for the next
/// one, so a cascade larger than the budget is worked through in arrival order instead of being
/// truncated by coordinate. A generation defines simultaneity (its targets see the mean of every
/// origin acting in it), so the batch size is part of the dynamics: every peer runs
/// [`Budget::DEFAULT`], which keeps results identical across machines.
pub struct ReactionScheduler {
    queue: BTreeSet<(u64, Pos, u8)>,
    seen: HashSet<(Pos, u8)>,
    capacity: usize,
    /// Generations run so far (a clock for tests and gauges).
    pub generations: u64,
    /// Mutations committed so far.
    pub mutations: u64,
    /// Events refused because the queue was full (a gauge for the console and the lab).
    pub dropped: u64,
}

impl Default for ReactionScheduler {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }
}

impl ReactionScheduler {
    /// An empty scheduler holding at most [`DEFAULT_CAPACITY`] distinct pending events.
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty scheduler holding at most `capacity` distinct pending events.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            queue: BTreeSet::new(),
            seen: HashSet::new(),
            capacity,
            generations: 0,
            mutations: 0,
            dropped: 0,
        }
    }

    /// Hand in a gameplay event. Duplicates (same position and kind) collapse until they are evaluated.
    pub fn push(&mut self, ev: MaterialEvent) {
        self.push_at(self.generations, ev);
    }

    fn push_at(&mut self, arrival: u64, ev: MaterialEvent) {
        let key = (ev.at, ev.kind as u8);
        if self.seen.contains(&key) {
            return;
        }
        if self.queue.len() >= self.capacity {
            self.dropped += 1;
            return;
        }
        self.seen.insert(key);
        self.queue.insert((arrival, ev.at, ev.kind as u8));
    }

    /// Events waiting for evaluation.
    pub fn pending(&self) -> usize {
        self.queue.len()
    }

    /// Run up to `budget.generations_per_tick` generations. Each generation: take the oldest batch of
    /// events, gather for every face neighbour (target) of every event cell (origin) the origins acting
    /// on it, evaluate each target ONCE with all its origins at the state at the START of the generation
    /// (`interact_many`: order-independent), commit the mutations in position order, then queue an
    /// `ExternallyChanged` follow-up for every changed cell's neighbours, for the next generation.
    /// A store that refuses a write (returns `None`) commits nothing for that cell: no mutation, no
    /// follow-ups. Returns the commits.
    pub fn tick<S: CellStore>(&mut self, store: &mut S, law: &Law, budget: Budget) -> Vec<Mutation> {
        let mut committed = Vec::new();
        for _ in 0..budget.generations_per_tick {
            if self.queue.is_empty() {
                break;
            }
            let batch: Vec<(u64, Pos, u8)> =
                self.queue.iter().take(budget.events_per_generation).copied().collect();
            for key in &batch {
                self.queue.remove(key);
                self.seen.remove(&(key.1, key.2));
            }
            // Gather phase: every target with the origins acting on it, all read from the
            // generation's starting state.
            let mut acting: BTreeMap<Pos, (BlockId, Vec<(BlockId, EventKind)>)> = BTreeMap::new();
            for &(_, at, kind) in &batch {
                let Some(origin_id) = store.block_at(at) else { continue };
                if origin_id == AIR {
                    continue;
                }
                let kind = kind_from_u8(kind);
                for f in FACES {
                    let tp = (at.0 + f.0, at.1 + f.1, at.2 + f.2);
                    let Some(target_id) = store.block_at(tp) else { continue };
                    if target_id == AIR {
                        continue;
                    }
                    acting
                        .entry(tp)
                        .or_insert_with(|| (target_id, Vec::new()))
                        .1
                        .push((origin_id, kind));
                }
            }
            // Evaluate phase: one interaction per target.
            let mut results: BTreeMap<Pos, (BlockId, BlockId)> = BTreeMap::new();
            for (tp, (target_id, origins)) in acting {
                let res = {
                    let r = store.registry();
                    let os: Vec<(&material::Configuration, EventKind)> =
                        origins.iter().map(|(id, k)| (r.configuration(*id), *k)).collect();
                    interact_many(law, &os, r.configuration(target_id))
                };
                if !res.changed {
                    continue;
                }
                let Some(new_id) = store.registry_mut().intern(&res.target) else { continue };
                if new_id != target_id {
                    results.insert(tp, (target_id, new_id));
                }
            }
            // Write phase: position order; follow-ups arrive for the next generation.
            let next = self.generations + 1;
            for (pos, (from, to)) in results {
                let Some(prev) = store.set_block(pos, to) else { continue };
                debug_assert_eq!(prev, from, "generation read a stale cell");
                committed.push(Mutation { pos, from, to });
                self.mutations += 1;
                for f in FACES {
                    self.push_at(
                        next,
                        MaterialEvent {
                            at: (pos.0 + f.0, pos.1 + f.1, pos.2 + f.2),
                            kind: EventKind::ExternallyChanged,
                        },
                    );
                }
            }
            self.generations += 1;
        }
        committed
    }
}

/// Sim system `Tick("reactions")`: one budgeted scheduler step at the 20 Hz sim tick.
pub struct Reactions;

/// Two interned configurations that change under Collision, drawn from the
/// worldgen regions (or a 40-unit one-axis shift of the first centre).
#[cfg(test)]
pub(crate) fn reactive_region_pair(reg: &mut BlockRegistry) -> (BlockId, BlockId) {
    use material::{interact, Configuration};
    let law = *reg.law();
    let regions = reg.regions().to_vec();
    for ra in &regions {
        for rb in &regions {
            let ca = Configuration::single(ra.centre);
            let cb = Configuration::single(rb.centre);
            if interact(&law, &ca, &cb, EventKind::Collision).changed
                || interact(&law, &cb, &ca, EventKind::Collision).changed
            {
                return (reg.intern(&ca).unwrap(), reg.intern(&cb).unwrap());
            }
        }
    }
    let centre = regions[0].centre;
    let ca = Configuration::single(centre);
    for axis in 0..4 {
        for &delta in &[40u8, 16, 50, 30] {
            let mut e = centre;
            e.0[axis] = if centre.0[axis] <= 255 - delta {
                centre.0[axis] + delta
            } else {
                centre.0[axis].saturating_sub(delta)
            };
            if e == centre {
                continue;
            }
            let cb = Configuration::single(e);
            if interact(&law, &ca, &cb, EventKind::Collision).changed
                || interact(&law, &cb, &ca, EventKind::Collision).changed
            {
                return (reg.intern(&ca).unwrap(), reg.intern(&cb).unwrap());
            }
        }
    }
    panic!("no reactive pair from the worldgen regions under Collision");
}

/// Place `a` and `b` at `(0, y, 0)` / `(1, y, 0)`, Collision both, tick 10 times.
#[cfg(test)]
pub(crate) fn scripted_run<S: CellStore>(
    store: &mut S,
    sched: &mut ReactionScheduler,
    a: BlockId,
    b: BlockId,
    y: i32,
) -> Vec<(Pos, Vec<u8>, Vec<u8>)> {
    store.set_block((0, y, 0), a);
    store.set_block((1, y, 0), b);
    sched.push(MaterialEvent {
        at: (0, y, 0),
        kind: EventKind::Collision,
    });
    sched.push(MaterialEvent {
        at: (1, y, 0),
        kind: EventKind::Collision,
    });
    let law = *store.registry().law();
    let mut keys = Vec::new();
    for _ in 0..10 {
        for m in sched.tick(store, &law, Budget::DEFAULT) {
            keys.push((
                m.pos,
                store.registry().encoding(m.from).as_bytes().to_vec(),
                store.registry().encoding(m.to).as_bytes().to_vec(),
            ));
        }
    }
    keys
}

impl Tick for Reactions {
    fn name(&self) -> &'static str {
        "reactions"
    }

    fn tick(&mut self, world: &mut World, _dt: f32) {
        let _ = world.tick_reactions();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_config::RenderConfig;
    use crate::world::World;
    use material::{interact_many, Configuration, Element};
    use std::collections::HashMap;

    struct Map {
        cells: HashMap<Pos, BlockId>,
        loaded: HashSet<Pos>,
        all_loaded: bool,
        reg: BlockRegistry,
    }
    impl CellStore for Map {
        fn block_at(&self, pos: Pos) -> Option<BlockId> {
            if !self.all_loaded && !self.loaded.contains(&pos) {
                return None;
            }
            Some(*self.cells.get(&pos).unwrap_or(&AIR))
        }
        fn set_block(&mut self, pos: Pos, id: BlockId) -> Option<BlockId> {
            self.loaded.insert(pos);
            Some(self.cells.insert(pos, id).unwrap_or(AIR))
        }
        fn registry(&self) -> &BlockRegistry {
            &self.reg
        }
        fn registry_mut(&mut self) -> &mut BlockRegistry {
            &mut self.reg
        }
    }
    fn map() -> Map {
        Map {
            cells: HashMap::new(),
            loaded: HashSet::new(),
            all_loaded: true,
            reg: BlockRegistry::with_builtins(),
        }
    }
    fn single(reg: &mut BlockRegistry, c: [u8; 4]) -> BlockId {
        reg.intern(&Configuration::single(Element::new(c))).unwrap()
    }

    #[test]
    fn uniform_matter_at_rest_never_reacts() {
        let mut m = map();
        let rock = single(&mut m.reg, [120, 130, 140, 150]);
        for x in 0..4 {
            m.set_block((x, 0, 0), rock);
        }
        let mut s = ReactionScheduler::new();
        s.push(MaterialEvent {
            at: (1, 0, 0),
            kind: EventKind::Collision,
        });
        let law = Law::v0();
        let out = s.tick(&mut m, &law, Budget::DEFAULT);
        assert!(out.is_empty());
        assert_eq!(s.pending(), 0);
    }

    #[test]
    fn a_reaction_commits_in_position_order_and_queues_followups() {
        let mut m = map();
        let a = single(&mut m.reg, [40, 40, 40, 40]);
        let b = single(&mut m.reg, [90, 40, 40, 40]); // 50 apart on one axis: attractive range
        m.set_block((0, 0, 0), a);
        m.set_block((1, 0, 0), b);
        m.set_block((-1, 0, 0), b);
        let mut s = ReactionScheduler::new();
        s.push(MaterialEvent {
            at: (0, 0, 0),
            kind: EventKind::Collision,
        });
        let law = Law::v0();
        let out = s.tick(
            &mut m,
            &law,
            Budget {
                generations_per_tick: 1,
                ..Budget::DEFAULT
            },
        );
        assert_eq!(out.len(), 2);
        assert!(out[0].pos < out[1].pos, "position order");
        assert!(out.iter().all(|mu| mu.from == b && mu.to != b));
        assert!(s.pending() > 0, "changed cells queue follow-ups for the next generation");
        assert_eq!(s.generations, 1);
    }

    #[test]
    fn duplicate_events_collapse_and_air_never_reacts() {
        let mut m = map();
        let mut s = ReactionScheduler::new();
        for _ in 0..10 {
            s.push(MaterialEvent {
                at: (5, 5, 5),
                kind: EventKind::Moved,
            });
        }
        assert_eq!(s.pending(), 1);
        let law = Law::v0();
        assert!(s.tick(&mut m, &law, Budget::DEFAULT).is_empty());
    }

    #[test]
    fn generations_are_jacobi_not_gauss_seidel() {
        // Two origins acting on one target in the same generation both read the ORIGINAL target and
        // act together (order-independent), never one after the other.
        let mut m = map();
        let o1 = single(&mut m.reg, [40, 40, 40, 40]);
        let o2 = single(&mut m.reg, [40, 40, 40, 90]);
        let t = single(&mut m.reg, [80, 40, 40, 60]);
        m.set_block((0, 0, 0), t);
        m.set_block((1, 0, 0), o1);
        m.set_block((0, 1, 0), o2);
        let mut s = ReactionScheduler::new();
        s.push(MaterialEvent {
            at: (1, 0, 0),
            kind: EventKind::Collision,
        });
        s.push(MaterialEvent {
            at: (0, 1, 0),
            kind: EventKind::Collision,
        });
        let law = Law::v0();
        let out = s.tick(
            &mut m,
            &law,
            Budget {
                generations_per_tick: 1,
                ..Budget::DEFAULT
            },
        );
        // The target changed at most once this generation, from its original id, by both origins.
        assert!(out.len() <= 1);
        if let Some(mu) = out.first() {
            assert_eq!(mu.from, t);
            let c1 = m.reg.configuration(o1).clone();
            let c2 = m.reg.configuration(o2).clone();
            let orig = Configuration::single(Element::new([80, 40, 40, 60]));
            let expect =
                interact_many(&law, &[(&c1, EventKind::Collision), (&c2, EventKind::Collision)], &orig)
                    .target;
            assert_eq!(m.reg.configuration(mu.to), &expect);
        }
    }

    #[test]
    fn budgets_bound_the_work_per_tick() {
        let mut m = map();
        let a = single(&mut m.reg, [40, 40, 40, 40]);
        let b = single(&mut m.reg, [90, 40, 40, 40]);
        for x in 0..64 {
            m.set_block((x, 0, 0), if x % 2 == 0 { a } else { b });
        }
        let mut s = ReactionScheduler::new();
        for x in 0..64 {
            s.push(MaterialEvent {
                at: (x, 0, 0),
                kind: EventKind::Collision,
            });
        }
        let law = Law::v0();
        let budget = Budget {
            events_per_generation: 8,
            generations_per_tick: 1,
        };
        let _ = s.tick(&mut m, &law, budget);
        assert!(s.pending() >= 56, "unevaluated events stay queued");
    }

    #[test]
    fn unloaded_space_never_reacts() {
        let mut m = map();
        m.all_loaded = false;
        let a = single(&mut m.reg, [40, 40, 40, 40]);
        let b = single(&mut m.reg, [90, 40, 40, 40]);
        m.set_block((0, 0, 0), a);
        // Neighbour is not in `loaded`, so block_at returns None even if we stuffed the map.
        m.cells.insert((1, 0, 0), b);
        let mut s = ReactionScheduler::new();
        s.push(MaterialEvent {
            at: (0, 0, 0),
            kind: EventKind::Collision,
        });
        let out = s.tick(&mut m, &Law::v0(), Budget::DEFAULT);
        assert!(out.is_empty());
        assert_eq!(m.cells.get(&(1, 0, 0)).copied(), Some(b));
    }

    #[test]
    fn reactions_tick_is_named_reactions() {
        assert_eq!(Reactions.name(), "reactions");
    }

    fn scripted_world_mutations() -> Vec<(Pos, Vec<u8>, Vec<u8>)> {
        let mut world = World::with_config(42, RenderConfig::default());
        let (a, b) = reactive_region_pair(world.registry_mut());
        let y = world.surface_y(0, 0);
        scripted_run(&mut world, &mut ReactionScheduler::new(), a, b, y)
    }

    #[test]
    fn scripted_world_mutations_are_bit_identical_across_runs() {
        let a = scripted_world_mutations();
        let b = scripted_world_mutations();
        assert_eq!(a, b);
        assert!(
            !a.is_empty(),
            "the scripted pair must actually react so the determinism check is not vacuously empty"
        );
    }

    #[test]
    fn world_cell_store_defines_unloaded_chunks_from_the_generator() {
        let world = World::with_config_lazy(1, RenderConfig::default());
        let y = world.surface_y(0, 0);
        let below = CellStore::block_at(&world, (0, y - 4, 0));
        assert!(below.is_some_and(|id| id != AIR), "generated ground reads without a chunk");
        assert_eq!(CellStore::block_at(&world, (0, y + 40, 0)), Some(AIR), "generated sky");
        // The world's own query keeps its contract: unloaded space reads as AIR for gameplay.
        assert_eq!(world.block_at(0, y - 4, 0), AIR);
    }

    #[test]
    fn world_cell_store_set_block_goes_through_the_overlay() {
        let mut world = World::with_config(1, RenderConfig::default());
        let (a, _) = reactive_region_pair(world.registry_mut());
        let y = world.surface_y(8, 8);
        let prev = CellStore::set_block(&mut world, (8, y, 8), a).expect("a world never refuses");
        assert_ne!(prev, a);
        assert_eq!(world.block_at(8, y, 8), a);
        assert!(
            world.edits().any(|(p, id)| p == (8, y, 8) && id == a),
            "mutation must persist in the edit overlay"
        );
    }

    #[test]
    fn a_client_does_not_run_the_scheduler() {
        let mut world = World::with_config(1, RenderConfig::default());
        let (a, b) = reactive_region_pair(world.registry_mut());
        let y = world.surface_y(0, 0);
        world.set_block(0, y, 0, a);
        world.set_block(1, y, 0, b);
        world.set_reactions_authority(false);
        world.push_material_event((0, y, 0), EventKind::Collision);
        assert_eq!(world.reactions().pending(), 0);
        assert!(world.tick_reactions().is_empty());
    }

    #[test]
    fn cascades_read_unloaded_cells_through_the_overlay_and_generator() {
        // No chunk is loaded: the pair lives in the overlay only, yet the scheduler must see it —
        // the world is defined without loading, so results never depend on streaming state.
        let mut world = World::with_config_lazy(1, RenderConfig::default());
        let (a, b) = reactive_region_pair(world.registry_mut());
        let y = 200; // well above any terrain: generated space there is air
        world.set_block(0, y, 0, a);
        world.set_block(1, y, 0, b);
        assert_eq!(CellStore::block_at(&world, (0, y, 0)), Some(a), "overlay read without a chunk");
        assert_eq!(CellStore::block_at(&world, (1, y, 0)), Some(b));
        assert_eq!(CellStore::block_at(&world, (0, y - 1, 0)), Some(AIR), "generated read");
        world.push_material_event((0, y, 0), EventKind::Collision);
        world.push_material_event((1, y, 0), EventKind::Collision);
        let out = world.tick_reactions();
        assert!(!out.is_empty(), "the pair reacts even though no chunk is loaded");
        for m in &out {
            assert_eq!(CellStore::block_at(&world, m.pos), Some(m.to), "the commit is readable");
            assert_ne!(m.from, AIR);
        }
    }

    fn shuffle_events(events: &mut [MaterialEvent], seed: u64) {
        let mut s = seed | 1;
        for i in (1..events.len()).rev() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            events.swap(i, (s as usize) % (i + 1));
        }
    }

    #[test]
    fn scheduler_is_order_independent_under_shuffle() {
        let mut events = Vec::new();
        for x in 0..16 {
            events.push(MaterialEvent {
                at: (x, 0, 0),
                kind: EventKind::Collision,
            });
        }
        let run = |seed: u64| {
            let mut m = map();
            let a = single(&mut m.reg, [40, 40, 40, 40]);
            let b = single(&mut m.reg, [90, 40, 40, 40]);
            for x in 0..16 {
                m.set_block((x, 0, 0), if x % 2 == 0 { a } else { b });
            }
            let mut evs = events.clone();
            shuffle_events(&mut evs, seed);
            let mut s = ReactionScheduler::new();
            for e in evs {
                s.push(e);
            }
            let law = Law::v0();
            let budget = Budget {
                events_per_generation: 5,
                generations_per_tick: 8,
            };
            let mut committed = Vec::new();
            for _ in 0..8 {
                committed.extend(s.tick(&mut m, &law, budget));
            }
            committed.sort_by_key(|mu| (mu.pos, mu.from.0, mu.to.0));
            let world: Vec<_> = (0..16)
                .map(|x| (x, m.block_at((x, 0, 0)).unwrap()))
                .collect();
            (committed, world)
        };
        let a = run(1);
        let b = run(0x9E37_79B1);
        let c = run(0xA5A5_A5A5);
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert!(
            !a.0.is_empty(),
            "the shuffled pair must actually react so the check is not vacuous"
        );
    }

    /// Alternating reactive cells along +x with Collision on every one.
    fn reactive_line(m: &mut Map, s: &mut ReactionScheduler, len: i32) {
        let a = single(&mut m.reg, [40, 40, 40, 40]);
        let b = single(&mut m.reg, [90, 40, 40, 40]);
        for x in 0..len {
            m.set_block((x, 0, 0), if x % 2 == 0 { a } else { b });
        }
        for x in 0..len {
            s.push(MaterialEvent {
                at: (x, 0, 0),
                kind: EventKind::Collision,
            });
        }
    }

    #[test]
    fn followups_are_deferred_never_dropped() {
        let mut m = map();
        let mut s = ReactionScheduler::new();
        reactive_line(&mut m, &mut s, 32);
        let budget = Budget {
            events_per_generation: 4,
            generations_per_tick: 1,
        };
        let mut committed = 0;
        for _ in 0..400 {
            committed += s.tick(&mut m, &Law::v0(), budget).len();
            if s.pending() == 0 {
                break;
            }
        }
        assert!(committed > 0, "the line must react");
        assert_eq!(s.dropped, 0, "a budget spreads work over generations, it never loses causality");
        assert_eq!(s.pending(), 0, "the cascade came to rest within 400 generations");
    }

    #[test]
    fn oldest_events_go_first_whatever_their_position() {
        // Two reactive pairs far apart: (100,101) queued first, (0,1) queued in the same generation
        // by a later push; with one event per generation the first generation evaluates x=0 (position
        // order within an arrival generation) and its follow-ups arrive for generation 2 — the still
        // older event at x=100 must be evaluated before them even though 0..2 < 100.
        let mut m = map();
        let a = single(&mut m.reg, [40, 40, 40, 40]);
        let b = single(&mut m.reg, [90, 40, 40, 40]);
        for base in [0, 100] {
            m.set_block((base, 0, 0), a);
            m.set_block((base + 1, 0, 0), b);
        }
        let mut s = ReactionScheduler::new();
        s.push(MaterialEvent {
            at: (100, 0, 0),
            kind: EventKind::Collision,
        });
        s.push(MaterialEvent {
            at: (0, 0, 0),
            kind: EventKind::Collision,
        });
        let budget = Budget {
            events_per_generation: 1,
            generations_per_tick: 1,
        };
        let law = Law::v0();
        let g1 = s.tick(&mut m, &law, budget);
        assert!(g1.iter().all(|mu| mu.pos.0 <= 2), "generation 1 works the x=0 pair: {g1:?}");
        assert!(!g1.is_empty(), "the pair must react");
        let g2 = s.tick(&mut m, &law, budget);
        assert!(
            g2.iter().all(|mu| mu.pos.0 >= 99),
            "generation 2 works the older x=100 event before the newer follow-ups: {g2:?}"
        );
    }

    #[test]
    fn capacity_refuses_and_counts_instead_of_growing() {
        let mut m = map();
        let mut s = ReactionScheduler::with_capacity(7);
        reactive_line(&mut m, &mut s, 32);
        assert_eq!(s.pending(), 7);
        assert_eq!(s.dropped, 25);
        let _ = s.tick(&mut m, &Law::v0(), Budget::DEFAULT);
        assert!(s.pending() <= 7, "follow-ups respect the capacity too");
    }

    #[test]
    fn a_refused_write_commits_nothing_and_queues_nothing() {
        struct Refusing(Map);
        impl CellStore for Refusing {
            fn block_at(&self, pos: Pos) -> Option<BlockId> {
                self.0.block_at(pos)
            }
            fn set_block(&mut self, _pos: Pos, _id: BlockId) -> Option<BlockId> {
                None
            }
            fn registry(&self) -> &BlockRegistry {
                &self.0.reg
            }
            fn registry_mut(&mut self) -> &mut BlockRegistry {
                &mut self.0.reg
            }
        }
        let mut m = map();
        let mut s = ReactionScheduler::new();
        reactive_line(&mut m, &mut s, 8);
        let mut r = Refusing(m);
        let out = s.tick(&mut r, &Law::v0(), Budget::DEFAULT);
        assert!(out.is_empty());
        assert_eq!(s.mutations, 0);
        assert_eq!(s.pending(), 0, "no follow-ups for changes that did not happen");
    }

    #[test]
    fn empty_pending_tick_is_a_noop() {
        let mut m = map();
        let before = m.reg.block_count();
        let mut s = ReactionScheduler::new();
        let out = s.tick(&mut m, &Law::v0(), Budget::DEFAULT);
        assert!(out.is_empty());
        assert_eq!(s.generations, 0);
        assert_eq!(m.reg.block_count(), before);
    }

    #[test]
    fn world_quiet_tick_interns_nothing() {
        let mut world = World::with_config(1, RenderConfig::default());
        let before = world.registry().block_count();
        assert!(world.tick_reactions().is_empty());
        assert_eq!(world.registry().block_count(), before);
        assert_eq!(world.reactions().pending(), 0);
    }

    #[test]
    fn quiet_sim_tick_empty_scheduler_allocates_nothing_and_touches_no_chunk() {
        use crate::alloc_count;
        struct PanicStore(BlockRegistry);
        impl CellStore for PanicStore {
            fn block_at(&self, pos: Pos) -> Option<BlockId> {
                panic!("empty scheduler tick read cell {pos:?}");
            }
            fn set_block(&mut self, pos: Pos, _id: BlockId) -> Option<BlockId> {
                panic!("empty scheduler tick wrote cell {pos:?}");
            }
            fn registry(&self) -> &BlockRegistry {
                &self.0
            }
            fn registry_mut(&mut self) -> &mut BlockRegistry {
                &mut self.0
            }
        }
        let mut store = PanicStore(BlockRegistry::with_builtins());
        let mut s = ReactionScheduler::new();
        crate::alloc_count::reset();
        let out = s.tick(&mut store, &Law::v0(), Budget::DEFAULT);
        assert!(out.is_empty());
        assert_eq!(alloc_count::alloc_count(), 0);
        assert_eq!(alloc_count::alloc_bytes(), 0);
        assert_eq!(s.generations, 0);

        let mut world = World::with_config(1, RenderConfig::default());
        assert_eq!(world.reactions().pending(), 0);
        alloc_count::reset();
        assert!(world.tick_reactions().is_empty());
        assert_eq!(alloc_count::alloc_count(), 0);
        assert_eq!(alloc_count::alloc_bytes(), 0);
        assert_eq!(alloc_count::cell_reads(), 0);
        assert_eq!(alloc_count::cell_writes(), 0);
    }

    #[test]
    #[ignore]
    fn scheduler_tick_cost_at_budget() {
        let mut m = map();
        let a = single(&mut m.reg, [40, 40, 40, 40]);
        let b = single(&mut m.reg, [90, 40, 40, 40]);
        for x in 0..256 {
            m.set_block((x, 0, 0), if x % 2 == 0 { a } else { b });
        }
        let mut s = ReactionScheduler::new();
        for x in 0..256 {
            s.push(MaterialEvent {
                at: (x, 0, 0),
                kind: EventKind::Collision,
            });
        }
        let law = Law::v0();
        let t0 = std::time::Instant::now();
        let _ = s.tick(&mut m, &law, Budget::DEFAULT);
        let us = t0.elapsed().as_secs_f64() * 1e6;
        println!(
            "scheduler tick at default budget: {us:.0} µs (pending left {})",
            s.pending()
        );
        assert!(us < 50_000.0, "tick {us:.0} µs is past the sanity ceiling");
    }
}
