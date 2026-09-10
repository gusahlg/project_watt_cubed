//! The reaction-event scheduler: turns world events into local material interactions, evaluated in
//! generations (every reaction of a generation reads the same state; mutations commit together, in
//! position order), budgeted per tick. Nothing here runs because a chunk loaded, meshed or saved —
//! only gameplay hands in events. In multiplayer the server owns the scheduler and broadcasts the
//! committed mutations; a client never evaluates reactions itself.

use std::collections::{BTreeMap, HashSet};

use material::{interact_many, EventKind, Law};

use crate::block::{BlockId, BlockRegistry, AIR};

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
    /// The material at a position (AIR when unloaded: unloaded space never reacts).
    fn block_at(&self, pos: Pos) -> Option<BlockId>;
    /// Overwrite a cell; returns the previous id.
    fn set_block(&mut self, pos: Pos, id: BlockId) -> BlockId;
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

/// Budget per generation and per tick, so a rule cannot cascade unboundedly in one frame.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Budget {
    /// Events evaluated per generation (the rest wait for the next generation).
    pub events_per_generation: usize,
    /// Generations run per tick.
    pub generations_per_tick: u32,
    /// Follow-up events a single generation may emit (cells that changed re-enter the queue).
    pub max_followups: usize,
}

impl Budget {
    /// Provisional defaults: small, so gameplay stays responsive while the law is explored.
    pub const DEFAULT: Budget = Budget { events_per_generation: 256, generations_per_tick: 2, max_followups: 512 };
}

const FACES: [(i32, i32, i32); 6] = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)];

/// The scheduler state: pending events (deduplicated per position and kind) and counters.
#[derive(Default)]
pub struct ReactionScheduler {
    pending: Vec<MaterialEvent>,
    seen: HashSet<(Pos, u8)>,
    /// Generations run so far (a clock for tests and gauges).
    pub generations: u64,
    /// Mutations committed so far.
    pub mutations: u64,
}

impl ReactionScheduler {
    /// An empty scheduler.
    pub fn new() -> Self {
        Self::default()
    }

    /// Hand in a gameplay event. Duplicates (same position and kind) collapse until they are evaluated.
    pub fn push(&mut self, ev: MaterialEvent) {
        if self.seen.insert((ev.at, ev.kind as u8)) {
            self.pending.push(ev);
        }
    }

    /// Events waiting for evaluation.
    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    /// Run up to `budget.generations_per_tick` generations. Each generation: take a batch of events,
    /// gather for every face neighbour (target) of every event cell (origin) the origins acting on it,
    /// evaluate each target ONCE with all its origins at the state at the START of the generation
    /// (`interact_many`: order-independent), commit the mutations in position order, then queue an
    /// `ExternallyChanged` follow-up for every changed cell's neighbours (bounded). Returns the commits.
    pub fn tick<S: CellStore>(&mut self, store: &mut S, law: &Law, budget: Budget) -> Vec<Mutation> {
        let mut committed = Vec::new();
        for _ in 0..budget.generations_per_tick {
            if self.pending.is_empty() {
                break;
            }
            let n = self.pending.len().min(budget.events_per_generation);
            let mut batch: Vec<MaterialEvent> = self.pending.drain(..n).collect();
            for ev in &batch {
                self.seen.remove(&(ev.at, ev.kind as u8));
            }
            // Deterministic evaluation order: by origin position, then kind.
            batch.sort_by_key(|e| (e.at, e.kind as u8));
            // Gather phase: every target with the origins acting on it, all read from the
            // generation's starting state.
            let mut acting: BTreeMap<Pos, (BlockId, Vec<(BlockId, EventKind)>)> = BTreeMap::new();
            for ev in &batch {
                let Some(origin_id) = store.block_at(ev.at) else { continue };
                if origin_id == AIR {
                    continue;
                }
                for f in FACES {
                    let tp = (ev.at.0 + f.0, ev.at.1 + f.1, ev.at.2 + f.2);
                    let Some(target_id) = store.block_at(tp) else { continue };
                    if target_id == AIR {
                        continue;
                    }
                    acting.entry(tp).or_insert_with(|| (target_id, Vec::new())).1.push((origin_id, ev.kind));
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
            // Write phase: position order.
            let mut followups = 0usize;
            for (pos, (from, to)) in results {
                let prev = store.set_block(pos, to);
                debug_assert_eq!(prev, from, "generation read a stale cell");
                committed.push(Mutation { pos, from, to });
                self.mutations += 1;
                if followups < budget.max_followups {
                    for f in FACES {
                        let np = (pos.0 + f.0, pos.1 + f.1, pos.2 + f.2);
                        self.push(MaterialEvent { at: np, kind: EventKind::ExternallyChanged });
                    }
                    followups += 6;
                }
            }
            self.generations += 1;
        }
        committed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use material::{Configuration, Element};
    use std::collections::HashMap;

    struct Map {
        cells: HashMap<Pos, BlockId>,
        reg: BlockRegistry,
    }
    impl CellStore for Map {
        fn block_at(&self, pos: Pos) -> Option<BlockId> {
            Some(*self.cells.get(&pos).unwrap_or(&AIR))
        }
        fn set_block(&mut self, pos: Pos, id: BlockId) -> BlockId {
            self.cells.insert(pos, id).unwrap_or(AIR)
        }
        fn registry(&self) -> &BlockRegistry {
            &self.reg
        }
        fn registry_mut(&mut self) -> &mut BlockRegistry {
            &mut self.reg
        }
    }
    fn map() -> Map {
        Map { cells: HashMap::new(), reg: BlockRegistry::with_builtins() }
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
        s.push(MaterialEvent { at: (1, 0, 0), kind: EventKind::Collision });
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
        s.push(MaterialEvent { at: (0, 0, 0), kind: EventKind::Collision });
        let law = Law::v0();
        let out = s.tick(&mut m, &law, Budget { generations_per_tick: 1, ..Budget::DEFAULT });
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
            s.push(MaterialEvent { at: (5, 5, 5), kind: EventKind::Moved });
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
        s.push(MaterialEvent { at: (1, 0, 0), kind: EventKind::Collision });
        s.push(MaterialEvent { at: (0, 1, 0), kind: EventKind::Collision });
        let law = Law::v0();
        let out = s.tick(&mut m, &law, Budget { generations_per_tick: 1, ..Budget::DEFAULT });
        // The target changed at most once this generation, from its original id, by both origins.
        assert!(out.len() <= 1);
        if let Some(mu) = out.first() {
            assert_eq!(mu.from, t);
            let c1 = m.reg.configuration(o1).clone();
            let c2 = m.reg.configuration(o2).clone();
            let orig = Configuration::single(Element::new([80, 40, 40, 60]));
            let expect = interact_many(&law, &[(&c1, EventKind::Collision), (&c2, EventKind::Collision)], &orig).target;
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
            s.push(MaterialEvent { at: (x, 0, 0), kind: EventKind::Collision });
        }
        let law = Law::v0();
        let budget = Budget { events_per_generation: 8, generations_per_tick: 1, max_followups: 12 };
        let _ = s.tick(&mut m, &law, budget);
        assert!(s.pending() >= 56, "unevaluated events stay queued");
    }
}
