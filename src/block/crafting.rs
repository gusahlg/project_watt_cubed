//! Natural-tier crafting: turn a *set* of elements into a registered block.
//!
//! This is the thin canonicalising layer between "what the player selected" and
//! the registry: the selection is sorted and deduplicated (a natural block is a
//! set — element multiplicity is a mixture-tier concern), so any ordering of the
//! same elements resolves to one composition, one name, one [`BlockId`]. The
//! registry already dedups identical compositions, so crafting the same set twice
//! hands back the existing id instead of growing the palette.
use crate::block::element::ElementId;
use crate::block::registry::{BlockId, BlockRegistry};

/// Craft (or look up) the natural block made of exactly this set of elements.
/// Order and duplicates in `elements` are irrelevant; the block is named by
/// joining the element names in id order, e.g. `Stone+Iron`.
pub fn craft_natural(registry: &mut BlockRegistry, elements: &[ElementId]) -> Option<BlockId> {
    let mut set: Vec<ElementId> = elements.to_vec();
    set.sort_unstable_by_key(|e| e.0);
    set.dedup();
    if set.is_empty() {
        return None;
    }
    // Existing compositions come back without growing the palette; new ones
    // are refused once the palette hits its texture-layer-bounded cap.
    let composition = crate::block::Composition::natural(&set);
    if let Some(existing) = registry.lookup(&composition) {
        return Some(existing);
    }
    // `natural` auto-names by joining element names with "+" (now sorted and
    // deduped), dedups the composition, and returns `None` if the palette is
    // at capacity.
    registry.natural(&set)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::element::El;

    #[test]
    fn same_set_crafts_to_same_id_once() {
        let mut reg = BlockRegistry::with_builtins();
        let a = craft_natural(&mut reg, &[El::Stone.id(), El::Iron.id()]).unwrap();
        let count = reg.block_count();
        let b = craft_natural(&mut reg, &[El::Stone.id(), El::Iron.id()]).unwrap();
        assert_eq!(a, b);
        assert_eq!(reg.block_count(), count, "re-crafting must not grow the palette");
    }

    #[test]
    fn order_and_duplicates_do_not_matter() {
        let mut reg = BlockRegistry::with_builtins();
        let a = craft_natural(&mut reg, &[El::Iron.id(), El::Stone.id()]).unwrap();
        let b = craft_natural(&mut reg, &[El::Stone.id(), El::Iron.id()]).unwrap();
        let c = craft_natural(&mut reg, &[El::Stone.id(), El::Iron.id(), El::Stone.id()]).unwrap();
        assert_eq!(a, b);
        assert_eq!(b, c, "duplicates collapse: a natural block is a set");
    }

    #[test]
    fn name_joins_element_names_in_id_order() {
        let mut reg = BlockRegistry::with_builtins();
        // Passed Glass-first, but Copper has the lower element id, so it leads.
        // (Stone+Iron would dedup into the builtin IronVein and keep its name.)
        let id = craft_natural(&mut reg, &[El::Glass.id(), El::Copper.id()]).unwrap();
        assert_eq!(reg.block(id).name.as_ref(), "Copper+Glass");
    }

    #[test]
    fn single_element_set_matches_builtin() {
        let mut reg = BlockRegistry::with_builtins();
        let id = craft_natural(&mut reg, &[El::Stone.id(), El::Stone.id()]).unwrap();
        assert_eq!(reg.id_by_name("Stone"), Some(id));
    }
}
