//! The reaction scheduler: a deterministic queue of active face contacts, worked through a few
//! hundred per simulation turn, one law operation per contact per turn.
//!
//! Reactions start only when contact or material state changes — a block is placed, removed, moved,
//! or its configuration changes (a tool, a machine, another reaction). The affected voxel's six
//! face-sharing contacts are queued, empty neighbours included (a mixture that wants to come apart
//! can shed into empty space); for a move both the old and the new location are. Diagonals never
//! interact. When a contact's turn comes, selective transfer v1 runs ONE operation on it
//! ([`BlockRegistry::react`]); if both configurations changed, every contact of both cells is queued
//! for the NEXT turn (the cascade advances one hop per turn, visibly, instead of completing at once).
//! A contact that produces no change goes dormant: it leaves the queue until something wakes it.
//!
//! Chunk loading, generation, meshing and rendering never queue anything. The queue order —
//! `(arrival turn, contact)` with contacts ordered by their lower cell and axis — and the per-turn
//! quota are part of the world's dynamics: every peer runs [`Budget::DEFAULT`]. A no-op attempt costs
//! quota like any other. In multiplayer the server owns the scheduler and broadcasts every committed
//! mutation; a client never evaluates reactions.

use std::collections::{BTreeSet, HashSet};

use crate::block::{BlockId, BlockRegistry};
use crate::world::World;

use super::Tick;

/// A world position of one voxel.
pub type Pos = (i32, i32, i32);

/// One face contact: the lower cell and the axis (0 = x, 1 = y, 2 = z) toward its `+1` neighbour.
/// Ordering is `(lower cell, axis)`, the canonical physical order of the law's A/B roles: A is
/// always the lower cell.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct Contact {
    /// The lower cell (the law's A).
    pub lo: Pos,
    /// Axis toward the upper cell.
    pub axis: u8,
}

impl Contact {
    /// The upper cell (the law's B).
    pub fn hi(self) -> Pos {
        let (x, y, z) = self.lo;
        match self.axis {
            0 => (x + 1, y, z),
            1 => (x, y + 1, z),
            _ => (x, y, z + 1),
        }
    }

    /// The six contacts of a cell, in canonical order.
    pub fn around(p: Pos) -> [Contact; 6] {
        let (x, y, z) = p;
        [
            Contact { lo: (x - 1, y, z), axis: 0 },
            Contact { lo: (x, y - 1, z), axis: 1 },
            Contact { lo: (x, y, z - 1), axis: 2 },
            Contact { lo: p, axis: 0 },
            Contact { lo: p, axis: 1 },
            Contact { lo: p, axis: 2 },
        ]
    }
}

/// What the scheduler needs from the world: read/write cells and the material table. A trait so the
/// queue semantics are testable on a plain map.
pub trait CellStore {
    /// The material at a position. `None` for space that cannot react (never generated).
    fn block_at(&self, pos: Pos) -> Option<BlockId>;
    /// Overwrite a cell; returns the previous id, or `None` when the store refused the write.
    fn set_block(&mut self, pos: Pos, id: BlockId) -> Option<BlockId>;
    /// The material table.
    fn registry(&self) -> &BlockRegistry;
    /// The material table, for interning results.
    fn registry_mut(&mut self) -> &mut BlockRegistry;
}

/// One committed cell change (an ordinary overlay edit attributed to the world).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Mutation {
    /// Where.
    pub pos: Pos,
    /// Before.
    pub from: BlockId,
    /// After.
    pub to: BlockId,
}

/// Work quota per simulation turn. Part of the simulation configuration, not a render setting.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Budget {
    /// Contact attempts per turn (no-op attempts count).
    pub contacts_per_turn: usize,
}

impl Budget {
    /// Every peer's quota: 384 attempts per turn at 20 turns per second.
    pub const DEFAULT: Budget = Budget { contacts_per_turn: 384 };
}

/// Distinct active contacts the scheduler holds before it refuses more (a memory guard; refusals are
/// counted in the `dropped` gauge and never happen in ordinary play).
pub const DEFAULT_CAPACITY: usize = 1 << 16;

/// The scheduler state: active contacts in processing order, and counters.
pub struct ReactionScheduler {
    queue: BTreeSet<(u64, Contact)>,
    active: HashSet<Contact>,
    capacity: usize,
    /// Turns run so far.
    pub turns: u64,
    /// Law operations committed so far.
    pub operations: u64,
    /// Contacts refused because the queue was full.
    pub dropped: u64,
}

impl Default for ReactionScheduler {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }
}

impl ReactionScheduler {
    /// An empty scheduler.
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty scheduler holding at most `capacity` active contacts.
    pub fn with_capacity(capacity: usize) -> Self {
        Self { queue: BTreeSet::new(), active: HashSet::new(), capacity, turns: 0, operations: 0, dropped: 0 }
    }

    fn enqueue(&mut self, arrival: u64, c: Contact) {
        if self.active.contains(&c) {
            return;
        }
        if self.active.len() >= self.capacity {
            self.dropped += 1;
            return;
        }
        self.active.insert(c);
        self.queue.insert((arrival, c));
    }

    /// Something changed at `p` (placed, removed, configuration changed): wake its six contacts.
    pub fn wake_cell(&mut self, p: Pos) {
        for c in Contact::around(p) {
            self.enqueue(self.turns, c);
        }
    }

    /// A block moved from `from` to `to`: both locations' contacts.
    pub fn wake_move(&mut self, from: Pos, to: Pos) {
        self.wake_cell(from);
        self.wake_cell(to);
    }

    /// Active contacts waiting for a turn.
    pub fn pending(&self) -> usize {
        self.queue.len()
    }

    /// The active contacts in processing order, each with its age in turns (how long it has been
    /// waiting): the pending work a save carries. Order and ages restore exactly.
    pub fn snapshot(&self) -> Vec<(u32, Contact)> {
        self.queue.iter().map(|&(t, c)| (self.turns.saturating_sub(t).min(u32::MAX as u64) as u32, c)).collect()
    }

    /// Restore saved pending work with its order, without inventing events.
    pub fn restore(&mut self, contacts: &[(u32, Contact)]) {
        let oldest = contacts.iter().map(|&(age, _)| age as u64).max().unwrap_or(0);
        self.turns = self.turns.max(oldest);
        for &(age, c) in contacts {
            self.enqueue(self.turns - age as u64, c);
        }
    }

    /// Run one turn: attempt up to `budget.contacts_per_turn` contacts that arrived before this turn,
    /// oldest first. A changed contact commits both cells and queues every contact of both cells for
    /// the next turn; an unchanged one goes dormant. Returns the commits in order.
    pub fn tick<S: CellStore>(&mut self, store: &mut S, budget: Budget) -> Vec<Mutation> {
        let mut committed = Vec::new();
        let this_turn = self.turns;
        let next = this_turn + 1;
        for _ in 0..budget.contacts_per_turn {
            let Some(&(arrival, c)) = self.queue.first() else { break };
            if arrival > this_turn {
                break;
            }
            self.queue.pop_first();
            self.active.remove(&c);
            let (lo, hi) = (c.lo, c.hi());
            let (Some(a), Some(b)) = (store.block_at(lo), store.block_at(hi)) else { continue };
            if a == b {
                // Identical configurations (and two voids) have no candidate: dormant.
                continue;
            }
            let Some((_, na, nb)) = store.registry_mut().react(a, b) else { continue };
            // Commit both cells as one change; a refused write commits nothing more.
            let Some(prev_a) = store.set_block(lo, na) else { continue };
            debug_assert_eq!(prev_a, a);
            committed.push(Mutation { pos: lo, from: a, to: na });
            if let Some(prev_b) = store.set_block(hi, nb) {
                debug_assert_eq!(prev_b, b);
                committed.push(Mutation { pos: hi, from: b, to: nb });
            }
            self.operations += 1;
            for p in [lo, hi] {
                for w in Contact::around(p) {
                    self.enqueue(next, w);
                }
            }
        }
        self.turns = next;
        committed
    }
}

/// Sim system `Tick("reactions")`: one budgeted scheduler turn at the 20 Hz sim tick.
pub struct Reactions;

impl Tick for Reactions {
    fn name(&self) -> &'static str {
        "reactions"
    }

    fn tick(&mut self, world: &mut World, _dt: f32) {
        let _ = world.tick_reactions();
    }
}

/// The reference destructive pair (selective transfer v1 §6): `A` is emptied by `E` in four
/// transfers. Interned into `reg` as `(A, E)`.
#[cfg(test)]
pub(crate) fn destructive_pair(reg: &mut BlockRegistry) -> (BlockId, BlockId) {
    use material::{Configuration, Element};
    let cfg = |e: [[u8; 4]; 4]| Configuration::new(e.map(Element::new).to_vec()).unwrap();
    let a = cfg([[73, 145, 162, 161], [71, 77, 157, 208], [34, 125, 217, 144], [8, 85, 210, 206]]);
    let e = cfg([[83, 135, 211, 195], [51, 125, 144, 147], [11, 72, 167, 145], [25, 80, 211, 204]]);
    (reg.intern(&a).unwrap(), reg.intern(&e).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::AIR;
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
        fn set_block(&mut self, pos: Pos, id: BlockId) -> Option<BlockId> {
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
        Map { cells: HashMap::new(), reg: BlockRegistry::with_builtins() }
    }

    fn run_to_rest(m: &mut Map, s: &mut ReactionScheduler) -> Vec<Mutation> {
        let mut all = Vec::new();
        for _ in 0..10_000 {
            if s.pending() == 0 {
                return all;
            }
            all.extend(s.tick(m, Budget::DEFAULT));
        }
        panic!("did not come to rest");
    }

    #[test]
    fn placing_a_counter_material_empties_the_resistant_block_over_several_turns() {
        let mut m = map();
        let (a, e) = destructive_pair(&mut m.reg);
        m.set_block((0, 0, 0), a);
        let mut s = ReactionScheduler::new();
        // Place E beside A: queue E's six contacts.
        m.set_block((1, 0, 0), e);
        s.wake_cell((1, 0, 0));
        let first = s.tick(&mut m, Budget::DEFAULT);
        assert_eq!(first.len(), 2, "one transfer per contact per turn: both cells change once");
        assert_eq!(m.reg.configuration(m.cells[&(0, 0, 0)]).len(), 3);
        assert_eq!(m.reg.configuration(m.cells[&(1, 0, 0)]).len(), 5);
        run_to_rest(&mut m, &mut s);
        assert_eq!(m.cells[&(0, 0, 0)], AIR, "further transfers emptied A");
        assert_eq!(m.reg.configuration(m.cells[&(1, 0, 0)]).len(), 8);
    }

    #[test]
    fn contacts_without_change_go_dormant_and_identical_neighbours_never_react() {
        let mut m = map();
        let rock = m.reg.intern(&Configuration::single(Element::new([120, 130, 140, 150]))).unwrap();
        for x in 0..4 {
            m.set_block((x, 0, 0), rock);
        }
        let mut s = ReactionScheduler::new();
        s.wake_cell((1, 0, 0));
        assert_eq!(s.pending(), 6);
        assert!(s.tick(&mut m, Budget::DEFAULT).is_empty());
        assert_eq!(s.pending(), 0, "every contact went dormant");
    }

    #[test]
    fn duplicate_wakes_collapse_and_the_order_is_lower_cell_then_axis() {
        let mut s = ReactionScheduler::new();
        for _ in 0..10 {
            s.wake_cell((5, 5, 5));
        }
        assert_eq!(s.pending(), 6);
        let order = s.snapshot();
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(order, sorted);
        assert_eq!(order[0], (0, Contact { lo: (4, 5, 5), axis: 0 }));
    }

    #[test]
    fn a_mixture_that_wants_to_separate_sheds_into_empty_space() {
        let mut m = map();
        let mixed = m
            .reg
            .intern(&Configuration::new(vec![Element::new([0; 4]), Element::new([128; 4])]).unwrap())
            .unwrap();
        m.set_block((0, 0, 0), mixed);
        let mut s = ReactionScheduler::new();
        s.wake_cell((0, 0, 0));
        run_to_rest(&mut m, &mut s);
        let occupied: Vec<_> = m.cells.iter().filter(|(_, id)| **id != AIR).collect();
        assert_eq!(occupied.len(), 2, "the two repelling elements ended in two cells: {occupied:?}");
    }

    #[test]
    fn a_smaller_budget_takes_more_turns_and_stays_reproducible() {
        let build = || {
            let mut m = map();
            let (a, e) = destructive_pair(&mut m.reg);
            for x in 0..12 {
                m.set_block((x, 0, 0), if x % 3 == 0 { e } else { a });
            }
            m
        };
        let mut fast = build();
        let mut slow = build();
        let (mut sf, mut ss) = (ReactionScheduler::new(), ReactionScheduler::new());
        for x in (0..12).step_by(3) {
            sf.wake_cell((x, 0, 0));
            ss.wake_cell((x, 0, 0));
        }
        let small = Budget { contacts_per_turn: 7 };
        let mut turns = 0;
        while ss.pending() > 0 {
            let _ = ss.tick(&mut slow, small);
            turns += 1;
            assert!(turns < 100_000);
        }
        run_to_rest(&mut fast, &mut sf);
        assert!(ss.turns > sf.turns, "a smaller quota takes more turns");
        // Within a turn the work is FIFO; a smaller quota moves turn boundaries, so outcomes may
        // legitimately differ — but each run is reproducible.
        let mut again = build();
        let mut sa = ReactionScheduler::new();
        for x in (0..12).step_by(3) {
            sa.wake_cell((x, 0, 0));
        }
        while sa.pending() > 0 {
            let _ = sa.tick(&mut again, small);
        }
        let mut cs: Vec<_> = slow.cells.into_iter().collect();
        let mut ca: Vec<_> = again.cells.into_iter().collect();
        cs.sort();
        ca.sort();
        assert_eq!(cs, ca);
    }

    #[test]
    fn element_occurrences_are_conserved_by_every_cascade() {
        let mut m = map();
        let (a, e) = destructive_pair(&mut m.reg);
        let mut total = 0;
        for x in 0..6 {
            for z in 0..3 {
                let id = if (x + z) % 4 == 0 { e } else { a };
                m.set_block((x, 0, z), id);
                total += m.reg.configuration(id).len();
            }
        }
        let mut s = ReactionScheduler::new();
        for x in 0..6 {
            s.wake_cell((x, 0, 1));
        }
        let muts = run_to_rest(&mut m, &mut s);
        assert!(!muts.is_empty());
        let after: usize = m.cells.values().map(|&id| m.reg.configuration(id).len()).sum();
        assert_eq!(after, total);
    }

    #[test]
    fn saved_pending_work_restores_in_order() {
        let mut s = ReactionScheduler::new();
        s.wake_cell((0, 0, 0));
        s.wake_move((3, 0, 0), (3, 1, 0));
        let saved = s.snapshot();
        let mut r = ReactionScheduler::new();
        r.restore(&saved);
        assert_eq!(r.snapshot(), saved);
    }

    #[test]
    fn a_quiet_tick_allocates_nothing_and_touches_no_cell() {
        use crate::alloc_count;
        use crate::render_config::RenderConfig;
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
        alloc_count::reset();
        assert!(s.tick(&mut store, Budget::DEFAULT).is_empty());
        assert_eq!(alloc_count::alloc_count(), 0);
        assert_eq!(alloc_count::alloc_bytes(), 0);

        let mut world = World::with_config(1, RenderConfig::default());
        assert_eq!(world.reactions().pending(), 0);
        alloc_count::reset();
        assert!(world.tick_reactions().is_empty());
        assert_eq!(alloc_count::alloc_count(), 0);
        assert_eq!(alloc_count::cell_reads(), 0);
        assert_eq!(alloc_count::cell_writes(), 0);
    }

    #[test]
    fn reactions_tick_is_named_reactions() {
        assert_eq!(Reactions.name(), "reactions");
    }
}
