//! Optional authoritative simulation host. Queueing and interaction laws live in native modules;
//! core keeps the cell adapter, identity, lifecycle, persistence and common budget vocabulary.
use super::Tick;
use super::model::{Model, ModelSpec, Sample, Stats};
use crate::block::{BlockId, BlockRegistry};
use crate::world::World;
use std::collections::BTreeMap;

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
            Contact {
                lo: (x - 1, y, z),
                axis: 0,
            },
            Contact {
                lo: (x, y - 1, z),
                axis: 1,
            },
            Contact {
                lo: (x, y, z - 1),
                axis: 2,
            },
            Contact { lo: p, axis: 0 },
            Contact { lo: p, axis: 1 },
            Contact { lo: p, axis: 2 },
        ]
    }
}

/// A simulation model's access to cells and the material table, also usable by a headless map.
pub trait CellStore {
    /// The material at a position. `None` for space that cannot react (never generated).
    fn block_at(&self, pos: Pos) -> Option<BlockId>;
    /// Overwrite a cell; returns the previous id, or `None` when the store refused the write.
    fn set_block(&mut self, pos: Pos, id: BlockId) -> Option<BlockId>;
    /// Resolve a virtual seam cell to its real storage identity.
    fn resolve(&self, pos: Pos) -> Pos {
        pos
    }
    /// Atomically replace two cells after checking both inputs; `false` changes neither. The
    /// default writes through [`set_block`](Self::set_block) and puts `lo` back when the store
    /// refuses `hi`. Stores that can refuse a write after accepting one (a bounded spec pool)
    /// admit both products up front instead.
    fn set_pair(&mut self, lo: Pos, a: BlockId, na: BlockId, hi: Pos, b: BlockId, nb: BlockId) -> bool {
        if lo == hi || self.block_at(lo) != Some(a) || self.block_at(hi) != Some(b) {
            return false;
        }
        if self.set_block(lo, na).is_none() {
            return false;
        }
        if self.set_block(hi, nb).is_none() {
            let undone = self.set_block(lo, a);
            debug_assert!(undone.is_some(), "a store takes back the material it just held");
            return false;
        }
        true
    }
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

/// Work quota per simulation turn. Models document their unit and bounded cost; whole-phase
/// field models may count sites, while selective transfer counts contact attempts.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Budget {
    /// Historical field name: work units per turn (no-op contact attempts count).
    pub contacts_per_turn: usize,
}

impl Budget {
    /// Default authority quota: 384 work units per turn at 20 turns per second.
    pub const DEFAULT: Budget = Budget {
        contacts_per_turn: 384,
    };
}

/// The extracted legacy module's default active-contact guard. Refusals increment `dropped`;
/// other models choose and document their own state bounds.
pub const DEFAULT_CAPACITY: usize = 1 << 16;

/// The longest simulation record value a save holds: the save's cap on one module-state value.
const MAX_RECORD: usize = crate::save::format::MAX_MODULE_STATE as usize;

#[derive(Clone)]
struct Saved {
    version: u32,
    payload: String,
}

/// The selected native model, plus frozen state retained for models switched off or absent.
/// Historical counters remain public for the existing inspection/benchmark API.
pub struct ReactionScheduler {
    model: Option<Box<dyn Model>>,
    spec: Option<ModelSpec>,
    saved: BTreeMap<String, Saved>,
    saved_selection: Option<(String, u32)>,
    legacy: Vec<(u32, Contact)>,
    pub turns: u64,
    pub operations: u64,
    pub dropped: u64,
}

impl Default for ReactionScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl ReactionScheduler {
    /// Production vanilla is inert. Core tests use the actual extracted legacy implementation
    /// through a test-only fixture; explicit no-model tests use `disabled`.
    pub fn new() -> Self {
        #[allow(unused_mut)]
        let mut host = Self::disabled();
        #[cfg(test)]
        host.configure(super::model::test_default())
            .expect("legacy test model");
        host
    }
    pub fn disabled() -> Self {
        Self {
            model: None,
            spec: None,
            saved: BTreeMap::new(),
            saved_selection: None,
            legacy: Vec::new(),
            turns: 0,
            operations: 0,
            dropped: 0,
        }
    }
    /// Kept for existing scheduler fixtures. Real models declare their own bounded queues.
    pub fn with_capacity(capacity: usize) -> Self {
        #[allow(unused_mut)]
        let mut host = Self::disabled();
        #[cfg(test)]
        {
            host.spec = super::model::test_default();
            host.model = Some(super::model::test_with_capacity(capacity));
        }
        #[cfg(not(test))]
        let _ = capacity;
        host
    }
    pub fn is_enabled(&self) -> bool {
        self.model.is_some()
    }
    pub fn model_id(&self) -> Option<&str> {
        self.spec.map(|s| s.id)
    }
    /// The running model, `None` while paused or frozen.
    pub fn spec(&self) -> Option<ModelSpec> {
        self.spec
    }
    pub fn selected_record(&self) -> Option<(&str, u32)> {
        self.saved_selection
            .as_ref()
            .map(|(id, version)| (id.as_str(), *version))
    }
    fn sync_stats(&mut self) {
        let s = self.model.as_ref().map_or(Stats::default(), |m| m.stats());
        self.turns = s.turns;
        self.operations = s.operations;
        self.dropped = s.dropped;
    }
    pub fn stats(&self) -> Stats {
        self.model.as_ref().map_or(Stats::default(), |m| m.stats())
    }
    pub fn pending(&self) -> usize {
        self.stats().pending
    }
    pub fn sample(&self, pos: Pos) -> Sample {
        self.model
            .as_ref()
            .map_or(Sample::default(), |m| m.sample(pos))
    }
    /// Switch/pause implementations without discarding another model's state. Version mismatch
    /// is an error, never a reinterpretation. Model factories must match their descriptor id.
    pub fn configure(&mut self, spec: Option<ModelSpec>) -> Result<(), String> {
        if let (Some((id, version)), Some(next)) = (&self.saved_selection, spec)
            && id == next.id
            && *version != next.version
        {
            return Err(format!(
                "simulation {id} selection is v{version}, model is v{}",
                next.version
            ));
        }
        if self.spec.map(|s| (s.id, s.version)) == spec.map(|s| (s.id, s.version)) {
            return Ok(());
        }
        let next = if let Some(s) = spec {
            let mut next = (s.create)();
            if next.id() != s.id {
                return Err("simulation factory returned a different model id".into());
            }
            if let Some(saved) = self.saved.get(s.id) {
                if saved.version != s.version {
                    return Err(format!(
                        "simulation {} state is v{}, model is v{}",
                        s.id, saved.version, s.version
                    ));
                }
                next.load(&saved.payload)?;
            }
            if s.id == super::model::LEGACY_ID && !self.legacy.is_empty() {
                next.restore_legacy(&self.legacy)?;
            }
            Some(next)
        } else {
            None
        };
        if let (Some(old), Some(s)) = (&self.model, self.spec) {
            self.saved.insert(
                s.id.into(),
                Saved {
                    version: s.version,
                    payload: old.save(),
                },
            );
        }
        self.model = next;
        self.spec = spec;
        self.saved_selection = spec.map(|s| (s.id.to_string(), s.version));
        if spec.is_some_and(|s| s.id == super::model::LEGACY_ID) {
            self.legacy.clear();
        }
        self.sync_stats();
        Ok(())
    }
    pub fn wake_cell(&mut self, pos: Pos) {
        if let Some(m) = &mut self.model {
            m.wake_cell(pos);
        }
        self.sync_stats();
    }
    pub fn wake_move(&mut self, from: Pos, to: Pos) {
        self.wake_cell(from);
        self.wake_cell(to);
    }
    pub fn perturb(&mut self, pos: Pos, amount: i64) {
        if let Some(m) = &mut self.model {
            m.perturb(pos, amount);
        }
        self.sync_stats();
    }
    pub fn tool(
        &mut self,
        registry: &mut BlockRegistry,
        cell: BlockId,
        tool: BlockId,
    ) -> Option<(BlockId, BlockId)> {
        self.model.as_mut()?.tool(registry, cell, tool)
    }
    pub fn tick<S: CellStore>(&mut self, store: &mut S, budget: Budget) -> Vec<Mutation> {
        let mut out = Vec::new();
        self.tick_into(store, budget, &mut out);
        out
    }
    pub fn tick_into<S: CellStore>(
        &mut self,
        store: &mut S,
        budget: Budget,
        out: &mut Vec<Mutation>,
    ) {
        if let Some(m) = &mut self.model {
            m.tick(store, budget, out);
        }
        self.sync_stats();
    }
    pub fn snapshot(&self) -> Vec<(u32, Contact)> {
        if self.spec.is_some_and(|s| s.id == super::model::LEGACY_ID) {
            self.model
                .as_ref()
                .map_or_else(|| self.legacy.clone(), |m| m.legacy_snapshot())
        } else {
            self.legacy.clone()
        }
    }
    pub fn restore(&mut self, contacts: &[(u32, Contact)]) {
        if self.spec.is_some_and(|s| s.id == super::model::LEGACY_ID) {
            if let Some(m) = &mut self.model
                && let Err(e) = m.restore_legacy(contacts)
            {
                eprintln!("could not restore legacy contacts: {e}");
            }
        } else {
            self.legacy = contacts.to_vec();
        }
        self.sync_stats();
    }
    /// Reserved records travel in the existing save's opaque module-state section.
    /// Each payload is copied once, in id order, with the live model's state in place of its
    /// frozen copy. A record over the save's module-state cap is left out with a notice, so one
    /// oversized model never fails the whole save.
    pub fn state_records(&self) -> Vec<(String, String)> {
        // An unavailable provider is frozen, not an intentional conversion to an off-world.
        let mut out = vec![(
            "@simulation.selected".into(),
            self.saved_selection
                .as_ref()
                .map_or_else(|| "off".into(), |(id, version)| format!("{id}@{version}")),
        )];
        let live = match (&self.model, self.spec) {
            (Some(m), Some(s)) => Some((s.id, s.version, m.save())),
            _ => None,
        };
        let mut push = |id: &str, version: u32, payload: &str| {
            let head = format!("v{version};");
            if head.len() + payload.len() > MAX_RECORD {
                eprintln!(
                    "simulation {id} state is {} bytes, over the save limit: it is not saved",
                    payload.len()
                );
                return;
            }
            let mut value = String::with_capacity(head.len() + payload.len());
            value.push_str(&head);
            value.push_str(payload);
            out.push((format!("@simulation.state.{id}"), value));
        };
        let mut unwritten = live.as_ref();
        for (id, s) in &self.saved {
            if let Some(&(live_id, version, ref payload)) = unwritten
                && live_id < id.as_str()
            {
                push(live_id, version, payload);
                unwritten = None;
            }
            if live.as_ref().is_some_and(|l| l.0 == id) {
                continue;
            }
            push(id, s.version, &s.payload);
        }
        if let Some(&(live_id, version, ref payload)) = unwritten {
            push(live_id, version, payload);
        }
        out
    }
    pub fn load_records(&mut self, records: &[(String, String)]) -> Result<(), String> {
        let mut saved = BTreeMap::new();
        let mut selection = None;
        let mut selected_seen = false;
        for (key, value) in records {
            if key == "@simulation.selected" {
                if selected_seen {
                    return Err("duplicate simulation selection".into());
                }
                selected_seen = true;
                selection = if value == "off" {
                    None
                } else {
                    let (id, version) = value
                        .rsplit_once('@')
                        .ok_or("invalid simulation selection")?;
                    if id.is_empty() {
                        return Err("empty simulation id".into());
                    }
                    Some((
                        id.into(),
                        version.parse().map_err(|_| "invalid simulation version")?,
                    ))
                };
            } else if let Some(id) = key.strip_prefix("@simulation.state.") {
                if id.is_empty() {
                    return Err("empty simulation state id".into());
                }
                let (version, payload) = value
                    .strip_prefix('v')
                    .and_then(|v| v.split_once(';'))
                    .ok_or("invalid simulation state")?;
                if value.len() > MAX_RECORD {
                    return Err("simulation state exceeds save limit".into());
                }
                if saved
                    .insert(
                        id.into(),
                        Saved {
                            version: version
                                .parse()
                                .map_err(|_| "invalid simulation state version")?,
                            payload: payload.into(),
                        },
                    )
                    .is_some()
                {
                    return Err("duplicate simulation state".into());
                }
            }
        }
        self.saved = saved;
        self.saved_selection = selection;
        // Force a later configure to hydrate from the records, even in a legacy test fixture.
        self.model = None;
        self.spec = None;
        self.sync_stats();
        Ok(())
    }
}

/// Generic fixed-tick adapter; the selected module supplies all simulation work.
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
    let a = cfg([
        [73, 145, 162, 161],
        [71, 77, 157, 208],
        [34, 125, 217, 144],
        [8, 85, 210, 206],
    ]);
    let e = cfg([
        [83, 135, 211, 195],
        [51, 125, 144, 147],
        [11, 72, 167, 145],
        [25, 80, 211, 204],
    ]);
    (reg.intern(&a).unwrap(), reg.intern(&e).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::AIR;
    use material::{Configuration, Element};
    use std::collections::HashMap;

    #[test]
    fn no_model_has_no_implicit_contact_or_tool_law() {
        let mut host = ReactionScheduler::disabled();
        host.wake_cell((0, 0, 0));
        assert_eq!(host.pending(), 0);
        let mut reg = BlockRegistry::with_builtins();
        let (a, e) = destructive_pair(&mut reg);
        assert_eq!(host.tool(&mut reg, a, e), None);
        assert_eq!(host.model_id(), None);
    }

    #[test]
    fn switching_off_preserves_the_real_module_state_for_later() {
        let mut host = ReactionScheduler::disabled();
        host.configure(crate::sim::model::test_default()).unwrap();
        host.wake_cell((2, 3, 4));
        let before = host.snapshot();
        host.configure(None).unwrap();
        let records = host.state_records();
        assert_eq!(host.pending(), 0);
        let mut loaded = ReactionScheduler::disabled();
        loaded.load_records(&records).unwrap();
        loaded.configure(crate::sim::model::test_default()).unwrap();
        assert_eq!(loaded.snapshot(), before);
    }

    #[test]
    fn unavailable_model_selection_survives_and_invalid_records_are_transactional() {
        let mut host = ReactionScheduler::disabled();
        host.configure(crate::sim::model::test_default()).unwrap();
        host.wake_cell((2, 3, 4));
        let records = host.state_records();
        let before = host.snapshot();
        assert!(
            host.load_records(&[("@simulation.selected".into(), "bad".into())])
                .is_err()
        );
        assert_eq!(host.snapshot(), before);
        let mut missing = ReactionScheduler::disabled();
        missing.load_records(&records).unwrap();
        missing.configure(None).unwrap();
        assert_eq!(missing.state_records(), records);
    }

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
        Map {
            cells: HashMap::new(),
            reg: BlockRegistry::with_builtins(),
        }
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
        assert_eq!(
            first.len(),
            2,
            "one transfer per contact per turn: both cells change once"
        );
        assert_eq!(m.reg.configuration(m.cells[&(0, 0, 0)]).len(), 3);
        assert_eq!(m.reg.configuration(m.cells[&(1, 0, 0)]).len(), 5);
        run_to_rest(&mut m, &mut s);
        assert_eq!(m.cells[&(0, 0, 0)], AIR, "further transfers emptied A");
        assert_eq!(m.reg.configuration(m.cells[&(1, 0, 0)]).len(), 8);
    }

    #[test]
    fn contacts_without_change_go_dormant_and_identical_neighbours_never_react() {
        let mut m = map();
        let rock = m
            .reg
            .intern(&Configuration::single(Element::new([120, 130, 140, 150])))
            .unwrap();
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
        assert_eq!(
            order[0],
            (
                0,
                Contact {
                    lo: (4, 5, 5),
                    axis: 0
                }
            )
        );
    }

    #[test]
    fn a_mixture_that_wants_to_separate_sheds_into_empty_space() {
        let mut m = map();
        let mixed = m
            .reg
            .intern(
                &Configuration::new(vec![Element::new([0; 4]), Element::new([128; 4])]).unwrap(),
            )
            .unwrap();
        m.set_block((0, 0, 0), mixed);
        let mut s = ReactionScheduler::new();
        s.wake_cell((0, 0, 0));
        run_to_rest(&mut m, &mut s);
        let occupied: Vec<_> = m.cells.iter().filter(|(_, id)| **id != AIR).collect();
        assert_eq!(
            occupied.len(),
            2,
            "the two repelling elements ended in two cells: {occupied:?}"
        );
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
        let small = Budget {
            contacts_per_turn: 7,
        };
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
        let after: usize = m
            .cells
            .values()
            .map(|&id| m.reg.configuration(id).len())
            .sum();
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
