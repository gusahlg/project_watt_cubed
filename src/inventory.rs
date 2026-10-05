//! The configurations a player carries. Owned by the core; mods only present and spend it.
use crate::block::registry::BlockId;

/// Starting capacity, in units. Counts of one configuration add together.
pub const START_CAPACITY: usize = 100;

/// The block counts the player is carrying — the single source of truth.
/// Mods read, spend, and display it; they do not own it.
pub struct Inventory {
    /// Per-block counts in first-seen order, so display rows are stable as
    /// counts change.
    counts: Vec<(BlockId, u32)>,
    /// Cached sum of `counts`; pickups and crafting read it every frame.
    total: u32,
    /// Soft cap on total units.
    capacity: usize,
    /// Bumped on every content change; caches (like the inventory's display rows)
    /// rebuild when they see a rev they haven't.
    rev: u64,
}

impl Inventory {
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

    /// Drop everything (used when loading a save into this inventory).
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
    /// Returns how many entries were skipped as unknown.
    pub fn load_portable(
        &mut self,
        items: &[(String, u32)],
        mut parse: impl FnMut(&str) -> Option<BlockId>,
    ) -> u32 {
        let mut pairs = Vec::new();
        let mut skipped = 0u32;
        for (spec, count) in items {
            if let Some(id) = parse(spec) {
                pairs.push((id, *count));
            } else {
                skipped += 1;
            }
        }
        self.clear();
        for (id, count) in pairs {
            self.add(id, count);
        }
        skipped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::registry::BlockRegistry;
    use crate::world::World;

    fn rock() -> BlockId {
        let mut reg = BlockRegistry::with_builtins();
        crate::world::terrain::Materials::intern(&mut reg);
        reg.id_by_label("rock").unwrap()
    }

    fn soil() -> BlockId {
        let mut reg = BlockRegistry::with_builtins();
        crate::world::terrain::Materials::intern(&mut reg);
        reg.id_by_label("soil").unwrap()
    }

    fn clay() -> BlockId {
        let mut reg = BlockRegistry::with_builtins();
        crate::world::terrain::Materials::intern(&mut reg);
        reg.id_by_label("gravel").unwrap()
    }

    #[test]
    fn inventory_add_respects_capacity_per_item() {
        let mut inventory = Inventory::new(3);
        let rock = rock();
        let soil = soil();
        let clay = clay();
        assert!(inventory.add(rock, 1));
        assert!(inventory.add(soil, 1));
        // Room for one more: one of two lands, the rest is dropped.
        assert!(!inventory.add(rock, 2));
        assert_eq!(inventory.total(), 3);
        assert_eq!(inventory.count(rock), 2);
        assert_eq!(inventory.count(clay), 0);
    }

    #[test]
    fn inventory_consume_is_all_or_nothing() {
        let mut inventory = Inventory::new(10);
        let rock = rock();
        let soil = soil();
        inventory.add(rock, 2);
        inventory.add(soil, 1);
        let rev = inventory.rev();
        assert!(!inventory.consume(soil, 2));
        assert_eq!(inventory.rev(), rev, "a failed consume changes nothing");
        assert_eq!(inventory.total(), 3);
        assert!(inventory.consume(rock, 1));
        assert!(inventory.consume(soil, 1));
        assert_eq!(inventory.count(rock), 1);
        assert_eq!(inventory.count(soil), 0);
        assert!(inventory.rev() > rev);
    }

    #[test]
    fn inventory_iteration_keeps_first_seen_order() {
        let mut inventory = Inventory::new(10);
        let rock = rock();
        let soil = soil();
        inventory.add(soil, 1);
        inventory.add(rock, 1);
        inventory.add(soil, 1);
        let order: Vec<_> = inventory.iter().collect();
        assert_eq!(order, vec![(soil, 2), (rock, 1)]);
    }

    #[test]
    fn load_portable_replaces_previous_contents() {
        let mut world = World::new(1);
        let rock = world.registry().id_by_label("rock").unwrap();
        let soil = world.registry().id_by_label("soil").unwrap();
        let mut inventory = Inventory::new(10);
        inventory.add(rock, 2);
        let spec = world.registry().spec(soil);
        inventory.load_portable(&[(spec, 1)], |s| world.registry_mut().parse_spec(s));
        assert_eq!(inventory.total(), 1);
        assert_eq!(inventory.count(soil), 1);
        assert_eq!(inventory.count(rock), 0);
    }

    #[test]
    fn load_portable_counts_unknown_specs() {
        let mut world = World::new(1);
        let rock = world.registry().id_by_label("rock").unwrap();
        let spec = world.registry().spec(rock);
        let mut inventory = Inventory::new(10);
        let skipped = inventory.load_portable(
            &[
                ("natural:Stone".into(), 2),
                (spec, 1),
                ("nope".into(), 3),
            ],
            |s| world.registry_mut().parse_spec(s),
        );
        assert_eq!(skipped, 2);
        assert_eq!(inventory.count(rock), 1);
        assert_eq!(inventory.total(), 1);
    }

    #[test]
    fn inventory_overflow_unchanged() {
        let mut inventory = Inventory::new(2);
        let rock = rock();
        assert!(inventory.add(rock, 2));
        assert!(!inventory.add(rock, 1));
        assert_eq!(inventory.total(), 2);
        assert_eq!(inventory.count(rock), 2);
    }

    #[test]
    fn inventory_overflow_with_many_configurations() {
        let mut world = World::new(1);
        let mut ids = Vec::new();
        for i in 0..8u8 {
            let c = material::Configuration::single(material::Element::new([i, 10, 20, 30]));
            ids.push(world.registry_mut().intern(&c).unwrap());
        }
        let mut inventory = Inventory::new(5);
        for &id in &ids {
            let _ = inventory.add(id, 1);
        }
        assert_eq!(inventory.total(), 5);
        assert_eq!(inventory.iter().count(), 5, "capacity is units, first-seen kinds");
        assert!(!inventory.add(ids[7], 1));
        assert_eq!(inventory.total(), 5);
        assert_eq!(inventory.count(ids[7]), 0);
    }

    #[test]
    fn portable_round_trips_through_specs() {
        let mut world = World::new(1);
        let rock = world.registry().id_by_label("rock").unwrap();
        let mut inventory = Inventory::new(10);
        inventory.add(rock, 3);
        let portable = inventory.to_portable(|id| world.registry().spec(id));
        let mut other = Inventory::new(10);
        other.load_portable(&portable, |s| world.registry_mut().parse_spec(s));
        assert_eq!(other.count(rock), 3);
        assert_eq!(other.total(), 3);
    }
}
