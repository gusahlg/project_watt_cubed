//! The configurations a player carries. Owned by the core; mods only present and spend it.
use crate::block::registry::BlockId;

/// Starting capacity. Large-looking, but with no stacking it is modest — and meant
/// to be upgraded over time.
pub const START_CAPACITY: usize = 100;

/// The block counts the player is carrying — the single source of truth.
/// Mods read, spend, and display it; they do not own it.
pub struct ElementStash {
    /// Per-block counts in first-seen order, so display rows are stable as
    /// counts change.
    counts: Vec<(BlockId, u32)>,
    /// Cached sum of `counts`; pickups and crafting read it every frame.
    total: u32,
    /// Soft cap on total held units — the old inventory capacity, upgradeable.
    capacity: usize,
    /// Bumped on every content change; caches (like the inventory's display rows)
    /// rebuild when they see a rev they haven't.
    rev: u64,
}

impl ElementStash {
    pub fn new(capacity: usize) -> Self {
        Self {
            counts: Vec::new(),
            total: 0,
            capacity,
            rev: 0,
        }
    }

    /// Add `n` units of `id` while there is room. Units past the capacity are
    /// dropped, and the return value is `false` if any were. Bumps `rev` when
    /// anything landed.
    pub fn add(&mut self, id: BlockId, n: u32) -> bool {
        if n == 0 {
            return true;
        }
        let room = self.capacity.saturating_sub(self.total as usize) as u32;
        let take = n.min(room);
        if take == 0 {
            return false;
        }
        match self.counts.iter_mut().find(|(e, _)| *e == id) {
            Some((_, count)) => *count += take,
            None => self.counts.push((id, take)),
        }
        self.total += take;
        self.rev += 1;
        take == n
    }

    /// How many of one block are held.
    pub fn count(&self, id: BlockId) -> u32 {
        self.counts
            .iter()
            .find(|(e, _)| *e == id)
            .map_or(0, |&(_, c)| c)
    }

    /// Spend `n` of `id`, all or nothing. Emptied rows drop out of the display
    /// order. Bumps `rev` on success.
    pub fn consume(&mut self, id: BlockId, n: u32) -> bool {
        if n == 0 {
            return true;
        }
        if self.count(id) < n {
            return false;
        }
        if let Some((_, count)) = self.counts.iter_mut().find(|(e, _)| *e == id) {
            *count -= n;
            self.total -= n;
        }
        self.counts.retain(|&(_, count)| count > 0);
        self.rev += 1;
        true
    }

    /// Take back up to `n` of `id`, best-effort. Unlike [`consume`](Self::consume)
    /// this is NOT all-or-nothing — it is the rollback path for a server-rejected
    /// break, where whatever was already spent elsewhere simply can't be revoked.
    pub fn revoke(&mut self, id: BlockId, n: u32) {
        if n == 0 {
            return;
        }
        let mut removed = false;
        if let Some((_, count)) = self.counts.iter_mut().find(|(e, _)| *e == id) {
            let take = (*count).min(n);
            if take > 0 {
                *count -= take;
                self.total -= take;
                removed = true;
            }
        }
        if removed {
            self.counts.retain(|&(_, count)| count > 0);
            self.rev += 1;
        }
    }

    /// Total units held, across all kinds.
    pub fn total(&self) -> u32 {
        self.total
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The current content revision (see the field docs).
    pub fn rev(&self) -> u64 {
        self.rev
    }

    /// The held `(id, count)` pairs in stable first-seen order.
    pub fn iter(&self) -> impl Iterator<Item = (BlockId, u32)> + '_ {
        self.counts.iter().copied()
    }

    /// Drop everything (used when loading a save into this stash).
    pub fn clear(&mut self) {
        self.counts.clear();
        self.total = 0;
        self.rev += 1;
    }

    /// Portable `(spec, count)` pairs in first-seen order.
    pub fn to_portable(&self, spec_of: impl Fn(BlockId) -> String) -> Vec<(String, u32)> {
        self.iter()
            .map(|(id, count)| (spec_of(id), count))
            .collect()
    }

    /// Replace contents from portable `(spec, count)` pairs. Unknown specs are
    /// skipped. Capacity still applies, so overflow is dropped the same as [`add`].
    pub fn load_portable(
        &mut self,
        items: &[(String, u32)],
        mut parse: impl FnMut(&str) -> Option<BlockId>,
    ) {
        let mut pairs = Vec::new();
        for (spec, count) in items {
            if let Some(id) = parse(spec) {
                pairs.push((id, *count));
            }
        }
        self.clear();
        for (id, count) in pairs {
            self.add(id, count);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::registry::BlockRegistry;
    use crate::world::placement;
    use crate::world::World;

    fn rock() -> BlockId {
        let mut reg = BlockRegistry::with_builtins();
        placement::builtin().compile(&mut reg);
        reg.id_by_label("rock").unwrap()
    }

    fn soil() -> BlockId {
        let mut reg = BlockRegistry::with_builtins();
        placement::builtin().compile(&mut reg);
        reg.id_by_label("soil").unwrap()
    }

    fn clay() -> BlockId {
        let mut reg = BlockRegistry::with_builtins();
        placement::builtin().compile(&mut reg);
        reg.id_by_label("clay").unwrap()
    }

    #[test]
    fn stash_add_respects_capacity_per_item() {
        let mut stash = ElementStash::new(3);
        let rock = rock();
        let soil = soil();
        let clay = clay();
        assert!(stash.add(rock, 1));
        assert!(stash.add(soil, 1));
        // Room for one more: one of two lands, the rest is dropped.
        assert!(!stash.add(rock, 2));
        assert_eq!(stash.total(), 3);
        assert_eq!(stash.count(rock), 2);
        assert_eq!(stash.count(clay), 0);
    }

    #[test]
    fn stash_consume_is_all_or_nothing() {
        let mut stash = ElementStash::new(10);
        let rock = rock();
        let soil = soil();
        stash.add(rock, 2);
        stash.add(soil, 1);
        let rev = stash.rev();
        assert!(!stash.consume(soil, 2));
        assert_eq!(stash.rev(), rev, "a failed consume changes nothing");
        assert_eq!(stash.total(), 3);
        assert!(stash.consume(rock, 1));
        assert!(stash.consume(soil, 1));
        assert_eq!(stash.count(rock), 1);
        assert_eq!(stash.count(soil), 0);
        assert!(stash.rev() > rev);
    }

    #[test]
    fn stash_iteration_keeps_first_seen_order() {
        let mut stash = ElementStash::new(10);
        let rock = rock();
        let soil = soil();
        stash.add(soil, 1);
        stash.add(rock, 1);
        stash.add(soil, 1);
        let order: Vec<_> = stash.iter().collect();
        assert_eq!(order, vec![(soil, 2), (rock, 1)]);
    }

    #[test]
    fn load_portable_replaces_previous_contents() {
        let mut world = World::new(1);
        let rock = world.registry().id_by_label("rock").unwrap();
        let soil = world.registry().id_by_label("soil").unwrap();
        let mut stash = ElementStash::new(10);
        stash.add(rock, 2);
        let spec = world.registry().spec(soil);
        stash.load_portable(&[(spec, 1)], |s| world.registry_mut().parse_spec(s));
        assert_eq!(stash.total(), 1);
        assert_eq!(stash.count(soil), 1);
        assert_eq!(stash.count(rock), 0);
    }

    #[test]
    fn stash_overflow_unchanged() {
        let mut stash = ElementStash::new(2);
        let rock = rock();
        assert!(stash.add(rock, 2));
        assert!(!stash.add(rock, 1));
        assert_eq!(stash.total(), 2);
        assert_eq!(stash.count(rock), 2);
    }

    #[test]
    fn portable_round_trips_through_specs() {
        let mut world = World::new(1);
        let rock = world.registry().id_by_label("rock").unwrap();
        let mut stash = ElementStash::new(10);
        stash.add(rock, 3);
        let portable = stash.to_portable(|id| world.registry().spec(id));
        let mut other = ElementStash::new(10);
        other.load_portable(&portable, |s| world.registry_mut().parse_spec(s));
        assert_eq!(other.count(rock), 3);
        assert_eq!(other.total(), 3);
    }
}
