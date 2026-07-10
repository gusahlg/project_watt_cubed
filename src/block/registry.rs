//! The block registry: the single source of truth that turns element compositions
//! into the [`BlockId`]s the world stores in every voxel. It is built once at
//! startup and read-only afterwards.
//!
//! Layout is deliberately split hot from cold (the "performance ahead of
//! readability" mandate). Two parallel arrays — `solid` and `color`, indexed by
//! `BlockId` — are *all* the per-voxel mesh/collision path ever reads, so they stay
//! small and cache-resident. Everything else (composition, derived properties,
//! specials, reactions, names) lives in the cold `blocks` vector that only
//! inspection, crafting, and the future simulation touch.
use std::collections::HashMap;

use voxel_engine::{Color, Pass};

use crate::block::composition::{Composition, MixError};
use crate::block::derive;
use crate::block::element::{CoreProperties, El, ElementId, ElementRegistry, SpecialKind};
use crate::block::reaction::{ActiveReaction, ReactionRegistry, apply_reactions};
use crate::macros::blocks;

/// A compact handle to a registered block. Voxels store this (1 byte), so a chunk
/// is just a flat array of ids into the registry. The [`MAX_BLOCK_TYPES`] cap keeps
/// every id inside the `u8` space.
///
/// [`MAX_BLOCK_TYPES`]: BlockRegistry::MAX_BLOCK_TYPES
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId(pub u8);

/// Empty space. Always id `0`, the only non-solid block.
pub const AIR: BlockId = BlockId(0);

/// A fully-derived block: what it is made of and everything that follows from it.
/// Cold data — read by inspection and crafting, never per voxel.
pub struct Block {
    pub name: Box<str>,
    pub composition: Composition,
    /// Core properties after reaction bonuses are folded in.
    pub core: CoreProperties,
    /// Special behaviours and their block-level strengths, sorted by kind.
    pub specials: Box<[(SpecialKind, u8)]>,
    /// Reactions that fired in this block, with their strengths.
    pub reactions: Box<[ActiveReaction]>,
}

/// Owns the element and reaction tables and every registered block. Resolves a
/// composition to a stable [`BlockId`], deduplicating identical compositions so the
/// palette stays bounded as terrain registers the same natural block over and over.
pub struct BlockRegistry {
    elements: ElementRegistry,
    reactions: ReactionRegistry,
    blocks: Vec<Block>,   // cold records
    solid: Vec<bool>,     // HOT, indexed by BlockId — collision key ("is there a block")
    opaque: Vec<bool>,    // HOT — mesher cull/AO key (solid & transparency == 0)
    layer: Vec<Pass>,     // HOT — mesher routing key (draw technique); Opaque iff `opaque`
    emission: Vec<u8>,    // HOT — blocklight seed, 0..=15
    color: Vec<Color>,    // HOT, indexed by BlockId
    dedup: HashMap<CompKey, BlockId>,
    names: HashMap<Box<str>, BlockId>,
}

/// A cache-resident snapshot of the registry's hot per-voxel tables, indexed by
/// [`BlockId`]. Bundled so meshing and light propagation read one immutable view
/// at one revision (the registry is append-only, so `block_count()` stamps it) —
/// no window in which `solid` is fresh but `opaque` is stale. Handed to worker
/// mesh jobs behind an `Arc` via [`crate::derived::Derived`].
#[derive(Default)]
pub struct HotTables {
    pub solid: Box<[bool]>,
    pub opaque: Box<[bool]>,
    /// Draw technique per block — the mesher's routing key. `layer[id] == Opaque`
    /// exactly when `opaque[id]` (both derived from `transparency`); the bool is
    /// kept for the branchless cull/AO hot loop, this for pass routing.
    pub layer: Box<[Pass]>,
    pub emission: Box<[u8]>,
}

impl BlockRegistry {
    /// A registry preloaded with the built-in elements, reactions, and blocks.
    /// `AIR` is registered first, so it is always [`BlockId(0)`](BlockId).
    pub fn with_builtins() -> Self {
        let mut registry = Self {
            elements: ElementRegistry::with_builtins(),
            reactions: ReactionRegistry::with_builtins(),
            blocks: Vec::new(),
            solid: Vec::new(),
            opaque: Vec::new(),
            layer: Vec::new(),
            emission: Vec::new(),
            color: Vec::new(),
            dedup: HashMap::new(),
            names: HashMap::new(),
        };
        register_builtins(&mut registry);
        registry
    }

    /// Whether the block is solid. The single hottest query in the game (collision
    /// every frame, neighbour culling while meshing) — one array load, no branch.
    #[inline]
    pub fn is_solid(&self, id: BlockId) -> bool {
        self.solid[id.0 as usize]
    }

    /// Whether the block hides the faces behind it — the mesher's cull key. A
    /// translucent solid (glass) is solid but NOT opaque, so faces behind it
    /// still draw. One array load, no branch, like [`is_solid`](Self::is_solid).
    #[inline]
    pub fn is_opaque(&self, id: BlockId) -> bool {
        self.opaque[id.0 as usize]
    }

    /// The block's blocklight output, 0..=15. Read only during light propagation.
    #[inline]
    pub fn emission(&self, id: BlockId) -> u8 {
        self.emission[id.0 as usize]
    }

    /// A fresh snapshot of the hot per-voxel tables for meshing/light jobs. Cheap
    /// (three small array copies); rebuilt only when the palette grows, behind the
    /// [`Derived`](crate::derived::Derived) revision cache on the world.
    pub fn hot_tables(&self) -> HotTables {
        HotTables {
            solid: self.solid.clone().into_boxed_slice(),
            opaque: self.opaque.clone().into_boxed_slice(),
            layer: self.layer.clone().into_boxed_slice(),
            emission: self.emission.clone().into_boxed_slice(),
        }
    }

    /// The block's render colour. Read once per emitted mesh face.
    #[inline]
    pub fn color(&self, id: BlockId) -> Color {
        self.color[id.0 as usize]
    }

    /// The full cold record for a block, for inspection and crafting.
    pub fn block(&self, id: BlockId) -> &Block {
        &self.blocks[id.0 as usize]
    }

    /// The element table, so callers can resolve element ids to names/properties.
    pub fn elements(&self) -> &ElementRegistry {
        &self.elements
    }

    /// How many distinct blocks are registered.
    /// Hard cap on distinct block types: the mesher stores the texture-array
    /// layer as a u8 (vertex color alpha), and it also bounds what remote
    /// network specs can make a client's palette (and texture memory) grow to.
    pub const MAX_BLOCK_TYPES: usize = 256;

    /// Whether the palette can still take a NEW composition.
    pub fn at_capacity(&self) -> bool {
        self.blocks.len() >= Self::MAX_BLOCK_TYPES
    }

    /// The already-registered block for this composition, if any (no growth).
    pub fn lookup(&self, composition: &Composition) -> Option<BlockId> {
        self.dedup.get(&CompKey::of(composition, self.blocks.len())).copied()
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Look up a block by its (case-sensitive) name.
    pub fn id_by_name(&self, name: &str) -> Option<BlockId> {
        self.names.get(name).copied()
    }

    /// Register a block from its composition, deriving every property and filling
    /// the hot arrays. Returns the existing id if an identical composition is
    /// already registered, so equal blocks share one id.
    ///
    /// Returns `None` if the composition is new and the palette is already at its
    /// [`MAX_BLOCK_TYPES`](Self::MAX_BLOCK_TYPES) cap — the single gate that keeps
    /// ids inside the `u8` voxel space rather than silently truncating.
    ///
    /// Name uniqueness is caller-enforced, not type-checked: if `name` was already
    /// used for a *different* composition (e.g. two `Configuration`s with the same
    /// mix but different `Layout`, since [`auto_name`](Self::auto_name) ignores
    /// layout), the first registration wins and `id_by_name` keeps resolving to it.
    pub fn register(&mut self, name: &str, composition: Composition) -> Option<BlockId> {
        let key = CompKey::of(&composition, self.blocks.len());
        if let Some(&existing) = self.dedup.get(&key) {
            return Some(existing);
        }

        if self.at_capacity() {
            return None;
        }

        // Reduce the composition to its weight multiset once, then share it
        // across reaction matching and every derivation instead of re-walking
        // it each time. (`derive_solid` only needs emptiness, not the weights.)
        let weights = composition.weights();
        let reactions = self.reactions.active_for_weights(&weights);
        let core = apply_reactions(
            derive::derive_core_from(&self.elements, &weights),
            &reactions,
        );
        let color = derive::derive_color_from(&self.elements, &weights);
        let solid = derive::derive_solid(&composition);
        let opaque = derive::derive_opaque(&core, solid);
        let layer = derive::derive_layer(&core, solid);
        let emission = derive::derive_emission(&core);
        let specials = derive::derive_specials_from(&self.elements, &weights);

        let id = self.push_block(
            Block {
                name: name.into(),
                composition,
                core,
                specials,
                reactions,
            },
            solid,
            opaque,
            layer,
            emission,
            color,
        );
        self.dedup.insert(key, id);
        debug_assert!(
            self.names.get(name).is_none_or(|&existing| existing == id),
            "block name collision: {name:?} already maps to a different id"
        );
        self.names.entry(name.into()).or_insert(id);
        Some(id)
    }

    /// Append one block to the parallel SoA arrays in lockstep, returning its
    /// freshly assigned [`BlockId`]. The single place the hot `solid`/`color`
    /// arrays and the cold `blocks` vector grow together, so they can never
    /// desync.
    fn push_block(
        &mut self,
        block: Block,
        solid: bool,
        opaque: bool,
        layer: Pass,
        emission: u8,
        color: Color,
    ) -> BlockId {
        let id = BlockId(self.blocks.len() as u8);
        self.blocks.push(block);
        self.solid.push(solid);
        self.opaque.push(opaque);
        self.layer.push(layer);
        self.emission.push(emission);
        self.color.push(color);
        id
    }

    /// Craft a natural block from a set of elements (equal parts). The natural-tier
    /// crafter the player uses without a machine; also the modding entry point.
    /// Returns `None` if the palette is at capacity (see [`register`](Self::register)).
    pub fn natural(&mut self, elements: &[ElementId]) -> Option<BlockId> {
        let composition = Composition::natural(elements);
        let name = self.auto_name(&composition);
        self.register(&name, composition)
    }

    /// Craft a mixture block from exact element percentages, which must sum to 100.
    /// The first machine-crafted tier. Errors if the shares are invalid or the
    /// palette is at capacity.
    pub fn mixture(&mut self, parts: &[(ElementId, u8)]) -> Result<BlockId, MixError> {
        let composition = Composition::mixture(parts)?;
        let name = self.auto_name(&composition);
        self.register(&name, composition).ok_or(MixError::Full)
    }

    /// A readable default name built from a composition's element names, e.g.
    /// `Soil+Clay` or `Copper60+Iron40`.
    fn auto_name(&self, composition: &Composition) -> String {
        match composition {
            Composition::Natural(els) => els
                .iter()
                .map(|&e| self.elements.get(e).name.to_string())
                .collect::<Vec<_>>()
                .join("+"),
            Composition::Mixture(mix) | Composition::Configuration { mix, .. } => mix
                .parts()
                .iter()
                .map(|&(e, p)| format!("{}{}", self.elements.get(e).name, p))
                .collect::<Vec<_>>()
                .join("+"),
            Composition::Computational(_) => "Computer".to_string(),
        }
    }
}

/// Which composition tier a [`CompKey::Reduced`] came from, so a `Mixture` and
/// a `Configuration` with the same weights never collide even though they'd
/// derive the same properties (layout aside).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Tier {
    Natural,
    Mixture,
    Configuration,
}

/// A hashable key for deduplicating compositions. Element order is
/// normalised so two blocks with the same tier and elements collapse to one id.
#[derive(Clone, PartialEq, Eq, Hash)]
enum CompKey {
    /// `Natural`/`Mixture`/`Configuration`, keyed by their weight multiset
    /// (sorted by id, duplicates merged into a count/share).
    Reduced(Tier, Vec<(u16, u32)>),
    /// Computational blocks are opaque, so they never dedup — keyed by a unique
    /// registration index instead.
    Computational(usize),
}

impl CompKey {
    fn of(composition: &Composition, fresh_index: usize) -> Self {
        match composition {
            Composition::Computational(_) => CompKey::Computational(fresh_index),
            _ => {
                let tier = match composition {
                    Composition::Natural(_) => Tier::Natural,
                    Composition::Mixture(_) => Tier::Mixture,
                    Composition::Configuration { .. } => Tier::Configuration,
                    Composition::Computational(_) => unreachable!("handled above"),
                };
                let parts =
                    composition.weights().parts().iter().map(|&(e, w)| (e.0, w)).collect();
                CompKey::Reduced(tier, parts)
            }
        }
    }
}

// The built-in block palette. `AIR` must be first (id 0). Built-in blocks are
// defined *as element compositions* — they dogfood the whole pipeline, so their
// solidity and colour are derived, not hardcoded.
blocks! {
    // Empty space: the one block with no elements, hence the only non-solid one.
    Air => Composition::natural(&[]),
    // Solid rock: a single element, so it keeps stone's exact grey.
    Stone => Composition::natural(&[El::Stone.id()]),
    // Packed earth: mostly soil, bound with clay.
    Dirt => Composition::mixture(&[(El::Soil.id(), 70), (El::Clay.id(), 30)])
        .expect("builtin Dirt sums to 100"),
    // Topsoil under a layer of growth: green over earthy brown.
    Grass => Composition::mixture(&[(El::Organic.id(), 65), (El::Soil.id(), 35)])
        .expect("builtin Grass sums to 100"),
    // --- Ore veins: stone flecked with a payload element. Natural blocks, so
    // the texture pipeline paints the flecks for free. Appended after the
    // original palette — ids must never reorder. ---
    // The starter fuel: shallow, common, and gone in a puff of smoke.
    CoalVein => Composition::natural(&[El::Stone.id(), El::Coal.id()]),
    // The workhorse metal, still wearing its rock.
    IronVein => Composition::natural(&[El::Stone.id(), El::Iron.id()]),
    // Wiring in the rough.
    CopperVein => Composition::natural(&[El::Stone.id(), El::Copper.id()]),
    // Deep glitter; heavy pockets for patient miners.
    GoldVein => Composition::natural(&[El::Stone.id(), El::Gold.id()]),
    // Dull grey seams that weigh more than they look.
    LeadVein => Composition::natural(&[El::Stone.id(), El::Lead.id()]),
    // Pale crystal veins with a charge-hoarding streak.
    QuartzVein => Composition::natural(&[El::Stone.id(), El::Quartz.id()]),
    // Yellow streaks best mined from a respectful distance.
    SulfurVein => Composition::natural(&[El::Stone.id(), El::Sulfur.id()]),
    // Rock with a faint glow seeping through the cracks.
    LuminVein => Composition::natural(&[El::Stone.id(), El::Lumin.id()]),
    // The deepest prize: aerospace-grade ore under miles of rock.
    TitanVein => Composition::natural(&[El::Stone.id(), El::Titan.id()]),
    // Sky-stone: only ever found up in the flying islands.
    AeriumVein => Composition::natural(&[El::Stone.id(), El::Aerium.id()]),
    // Pure volcanic glass pockets in the deep dark.
    Obsidian => Composition::natural(&[El::Obsidian.id()]),
    // Lowland beaches: what valleys have instead of grass.
    Sand => Composition::natural(&[El::Sand.id()]),
    // High-altitude island frosting.
    Ice => Composition::natural(&[El::Ice.id()]),
    // Oceans, rivers, and lakes: a translucent solid you can stand on (the glass
    // render path), filling every column up to sea level.
    Water => Composition::natural(&[El::Water.id()]),
    // Biome dressing on cold or high ground.
    Snow => Composition::natural(&[El::Snow.id()]),
    // Tree trunk: woody brown, distinct from packed dirt.
    Wood => Composition::mixture(&[(El::Soil.id(), 55), (El::Coal.id(), 25), (El::Clay.id(), 20)])
        .expect("builtin Wood sums to 100"),
    // Tree canopy: pure living green.
    Leaves => Composition::natural(&[El::Organic.id()]),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::element::El;

    #[test]
    fn air_is_zero_and_not_solid() {
        let reg = BlockRegistry::with_builtins();
        assert_eq!(Blk::Air.id(), AIR);
        assert_eq!(AIR, BlockId(0));
        assert!(!reg.is_solid(AIR));
        assert!(reg.is_solid(Blk::Stone.id()));
    }

    #[test]
    fn hot_arrays_agree_with_cold_records() {
        let reg = BlockRegistry::with_builtins();
        let hot = reg.hot_tables();
        for i in 0..reg.block_count() {
            let id = BlockId(i as u8);
            let block = reg.block(id);
            let solid = derive::derive_solid(&block.composition);
            assert_eq!(reg.is_solid(id), solid);
            assert_eq!(reg.is_opaque(id), derive::derive_opaque(&block.core, solid));
            assert_eq!(hot.layer[i], derive::derive_layer(&block.core, solid));
            // The routing enum and the branchless cull bool are the same fact.
            assert_eq!(hot.layer[i] == Pass::Opaque, reg.is_opaque(id));
            assert_eq!(reg.emission(id), derive::derive_emission(&block.core));
            assert_eq!(reg.color(id), derive::derive_color(reg.elements(), &block.composition));
        }
    }

    #[test]
    fn stone_keeps_its_grey() {
        let reg = BlockRegistry::with_builtins();
        assert_eq!(reg.color(Blk::Stone.id()), Color::new(128, 128, 128, 255));
    }

    #[test]
    fn identical_compositions_dedup() {
        let mut reg = BlockRegistry::with_builtins();
        let a = reg.natural(&[El::Stone.id()]).unwrap();
        // Same as the built-in Stone — must resolve to the existing id, not a new one.
        assert_eq!(a, Blk::Stone.id());
        let before = reg.block_count();
        let b = reg.natural(&[El::Iron.id(), El::Copper.id()]).unwrap();
        let c = reg.natural(&[El::Copper.id(), El::Iron.id()]).unwrap(); // order-independent
        assert_eq!(b, c);
        assert_eq!(reg.block_count(), before + 1);
    }

    #[test]
    fn duplicated_natural_element_does_not_collapse_to_singleton() {
        let mut reg = BlockRegistry::with_builtins();
        let before = reg.block_count();
        // [Stone, Stone] carries a different weight (count 2) than plain [Stone]
        // (count 1), so it must register as a distinct block, not dedup with Stone.
        let doubled = reg.natural(&[El::Stone.id(), El::Stone.id()]).unwrap();
        assert_ne!(doubled, Blk::Stone.id());
        assert_eq!(reg.block_count(), before + 1);
        // Registering the same doubled composition again dedups with itself.
        let doubled_again = reg.natural(&[El::Stone.id(), El::Stone.id()]).unwrap();
        assert_eq!(doubled, doubled_again);
        assert_eq!(reg.block_count(), before + 1);
    }

    #[test]
    fn registering_past_cap_refuses_rather_than_wraps() {
        use crate::block::element::ElementId;
        let mut reg = BlockRegistry::with_builtins();
        let n = reg.elements().len() as u16;
        // Fill the palette to its cap with distinct three-element natural blocks.
        'fill: for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    if reg.at_capacity() {
                        break 'fill;
                    }
                    reg.natural(&[ElementId(i), ElementId(j), ElementId(k)]);
                }
            }
        }
        assert!(reg.at_capacity(), "test needs enough elements to fill the palette");
        assert_eq!(reg.block_count(), BlockRegistry::MAX_BLOCK_TYPES);
        // A brand-new composition past the cap is refused, not truncated into a
        // colliding u8 voxel id.
        assert_eq!(reg.natural(&[ElementId(0), ElementId(1), ElementId(2), ElementId(3)]), None);
        assert_eq!(reg.mixture(&[(ElementId(0), 60), (ElementId(1), 40)]), Err(MixError::Full));
        assert_eq!(reg.block_count(), BlockRegistry::MAX_BLOCK_TYPES);
    }

    #[test]
    fn mixture_rejects_bad_percentages() {
        let mut reg = BlockRegistry::with_builtins();
        assert!(reg.mixture(&[(El::Soil.id(), 70), (El::Clay.id(), 20)]).is_err());
    }

    #[test]
    fn names_resolve() {
        let reg = BlockRegistry::with_builtins();
        assert_eq!(reg.id_by_name("Grass"), Some(Blk::Grass.id()));
        assert_eq!(reg.id_by_name("Nonexistent"), None);
    }
}
