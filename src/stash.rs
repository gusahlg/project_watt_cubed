//! The elements a player carries. Owned by the core; mods only present and spend it.
//!
//! Server-side validation of holdings is a follow-up; nothing on the wire
//! changes with this move.
use crate::block::ElementId;

/// Starting capacity. Large-looking, but with no stacking it is modest — and meant
/// to be upgraded over time.
pub const START_CAPACITY: usize = 100;

/// The element counts the player is carrying — the single source of truth.
/// Mods read, spend, and display it; they do not own it.
pub struct ElementStash {
    /// Per-element counts in first-seen order, so display rows are stable as
    /// counts change (matching the old inventory's grouped view).
    counts: Vec<(ElementId, u32)>,
    /// Cached sum of `counts`; pickups and crafting read it every frame.
    total: u32,
    /// Soft cap on total held elements — the old inventory capacity, upgradeable.
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

    /// Add elements one by one while there is room, exactly like the old
    /// inventory's per-item add: elements past the capacity are dropped, and the
    /// return value is `false` if any were. Bumps `rev` when anything landed.
    pub fn add(&mut self, elements: &[ElementId]) -> bool {
        let mut all = true;
        let mut added = false;
        for &element in elements {
            if self.total as usize >= self.capacity {
                all = false;
                continue;
            }
            match self.counts.iter_mut().find(|(e, _)| *e == element) {
                Some((_, count)) => *count += 1,
                None => self.counts.push((element, 1)),
            }
            self.total += 1;
            added = true;
        }
        if added {
            self.rev += 1;
        }
        all
    }

    /// How many of one element are held.
    pub fn count(&self, element: ElementId) -> u32 {
        self.counts
            .iter()
            .find(|(e, _)| *e == element)
            .map_or(0, |&(_, c)| c)
    }

    /// Spend elements, all or nothing: `elements` is consumed only if every entry
    /// is covered (listing an element twice requires two of it). Emptied elements
    /// drop out of the display order. Bumps `rev` on success.
    pub fn consume(&mut self, elements: &[ElementId]) -> bool {
        // Check the full multiplicity first so a failure changes nothing.
        for &element in elements {
            let needed = elements.iter().filter(|&&e| e == element).count() as u32;
            if self.count(element) < needed {
                return false;
            }
        }
        for &element in elements {
            if let Some((_, count)) = self.counts.iter_mut().find(|(e, _)| *e == element) {
                *count -= 1;
                self.total -= 1;
            }
        }
        self.counts.retain(|&(_, count)| count > 0);
        self.rev += 1;
        true
    }

    /// Take back elements, best-effort: each entry removes one of that element
    /// if any are held. Unlike [`consume`](Self::consume) this is NOT
    /// all-or-nothing — it is the rollback path for a server-rejected break,
    /// where whatever was already spent elsewhere simply can't be revoked.
    pub fn revoke(&mut self, elements: &[ElementId]) {
        let mut removed = false;
        for &element in elements {
            if let Some((_, count)) = self.counts.iter_mut().find(|(e, _)| *e == element) {
                if *count > 0 {
                    *count -= 1;
                    self.total -= 1;
                    removed = true;
                }
            }
        }
        if removed {
            self.counts.retain(|&(_, count)| count > 0);
            self.rev += 1;
        }
    }

    /// Total elements held, across all kinds.
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

    /// The held `(element, count)` pairs in stable first-seen order.
    pub fn iter(&self) -> impl Iterator<Item = (ElementId, u32)> + '_ {
        self.counts.iter().copied()
    }

    /// Drop everything (used when loading a save into this stash).
    pub fn clear(&mut self) {
        self.counts.clear();
        self.total = 0;
        self.rev += 1;
    }

    /// Portable `(name, count)` pairs in first-seen order.
    pub fn to_portable<'a>(&self, name_of: impl Fn(ElementId) -> &'a str) -> Vec<(String, u32)> {
        self.iter()
            .map(|(id, count)| (name_of(id).to_string(), count))
            .collect()
    }

    /// Replace contents from portable `(name, count)` pairs. Unknown names are
    /// skipped. Capacity still applies, so overflow is dropped the same as [`add`].
    pub fn load_portable(
        &mut self,
        items: &[(String, u32)],
        id_of: impl Fn(&str) -> Option<ElementId>,
    ) {
        let mut ids = Vec::new();
        for (name, count) in items {
            if let Some(id) = id_of(name) {
                ids.extend(std::iter::repeat_n(id, *count as usize));
            }
        }
        self.clear();
        self.add(&ids);
    }

    /// Old inventory save line: one element name per held unit, pickup order.
    /// Unknown names are skipped.
    pub fn load_names(&mut self, data: &str, id_of: impl Fn(&str) -> Option<ElementId>) {
        let mut ids = Vec::new();
        for name in data.split(',').filter(|s| !s.is_empty()) {
            if let Some(id) = id_of(name) {
                ids.push(id);
            }
        }
        self.clear();
        self.add(&ids);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::element::El;
    use crate::world::World;

    #[test]
    fn stash_add_respects_capacity_per_item() {
        let mut stash = ElementStash::new(3);
        assert!(stash.add(&[El::Stone.id(), El::Soil.id()]));
        // Room for one more: the first lands, the second is dropped.
        assert!(!stash.add(&[El::Stone.id(), El::Clay.id()]));
        assert_eq!(stash.total(), 3);
        assert_eq!(stash.count(El::Stone.id()), 2);
        assert_eq!(stash.count(El::Clay.id()), 0);
    }

    #[test]
    fn stash_consume_is_all_or_nothing() {
        let mut stash = ElementStash::new(10);
        stash.add(&[El::Stone.id(), El::Stone.id(), El::Iron.id()]);
        let rev = stash.rev();
        assert!(!stash.consume(&[El::Stone.id(), El::Copper.id()]));
        assert_eq!(stash.rev(), rev, "a failed consume changes nothing");
        assert_eq!(stash.total(), 3);
        assert!(stash.consume(&[El::Stone.id(), El::Iron.id()]));
        assert_eq!(stash.count(El::Stone.id()), 1);
        assert_eq!(stash.count(El::Iron.id()), 0);
        assert!(stash.rev() > rev);
    }

    #[test]
    fn stash_consume_counts_multiplicity() {
        let mut stash = ElementStash::new(10);
        stash.add(&[El::Stone.id()]);
        // Listing Stone twice needs two Stones; only one is held.
        assert!(!stash.consume(&[El::Stone.id(), El::Stone.id()]));
        assert!(stash.consume(&[El::Stone.id()]));
        assert_eq!(stash.total(), 0);
    }

    #[test]
    fn stash_iteration_keeps_first_seen_order() {
        let mut stash = ElementStash::new(10);
        stash.add(&[El::Iron.id(), El::Stone.id(), El::Iron.id()]);
        let order: Vec<_> = stash.iter().collect();
        assert_eq!(order, vec![(El::Iron.id(), 2), (El::Stone.id(), 1)]);
    }

    #[test]
    fn load_names_skips_unknown_and_groups_by_first_seen() {
        let world = World::new(1);
        let elements = world.registry().elements();
        let mut stash = ElementStash::new(10);
        stash.load_names("Stone,Soil,Stone,Bogus", |n| elements.id_by_name(n));
        assert_eq!(stash.total(), 3);
        assert_eq!(stash.count(El::Stone.id()), 2);
        assert_eq!(stash.count(El::Soil.id()), 1);
        let portable = stash.to_portable(|id| elements.get(id).name.as_ref());
        assert_eq!(
            portable,
            vec![("Stone".into(), 2), ("Soil".into(), 1)]
        );
    }

    #[test]
    fn load_portable_replaces_previous_contents() {
        let world = World::new(1);
        let elements = world.registry().elements();
        let mut stash = ElementStash::new(10);
        stash.load_names("Stone,Stone", |n| elements.id_by_name(n));
        stash.load_portable(&[("Iron".into(), 1)], |n| elements.id_by_name(n));
        assert_eq!(stash.total(), 1);
        assert_eq!(stash.count(El::Iron.id()), 1);
        assert_eq!(stash.count(El::Stone.id()), 0);
    }
}
