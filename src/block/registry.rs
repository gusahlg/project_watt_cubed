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

use raylib::prelude::*;

use crate::block::composition::{Composition, MixError};
use crate::block::derive;
use crate::block::element::{CoreProperties, El, ElementId, ElementRegistry, SpecialKind};
use crate::block::reaction::{ActiveReaction, ReactionRegistry, apply_reactions};
use crate::macros::blocks;

/// A compact handle to a registered block. Voxels store this (2 bytes), so a chunk
/// is just a flat array of ids into the registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId(pub u16);

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
    blocks: Vec<Block>, // cold records
    solid: Vec<bool>,   // HOT, indexed by BlockId
    color: Vec<Color>,  // HOT, indexed by BlockId
    dedup: HashMap<CompKey, BlockId>,
    names: HashMap<String, BlockId>,
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
    /// Panics only if the palette would exceed the `u16` id space (65 536 blocks) —
    /// a hard ceiling, reported rather than silently wrapped.
    pub fn register(&mut self, name: &str, composition: Composition) -> BlockId {
        let key = CompKey::of(&composition, self.blocks.len());
        if let Some(&existing) = self.dedup.get(&key) {
            return existing;
        }

        assert!(
            self.blocks.len() <= u16::MAX as usize,
            "block registry is full ({} blocks); BlockId is a u16",
            self.blocks.len()
        );

        let reactions = self.reactions.active_for(&composition);
        let core = apply_reactions(derive::derive_core(&self.elements, &composition), &reactions);
        let color = derive::derive_color(&self.elements, &composition);
        let solid = derive::derive_solid(&composition);
        let specials = derive::derive_specials(&self.elements, &composition);

        let id = BlockId(self.blocks.len() as u16);
        self.blocks.push(Block {
            name: name.into(),
            composition,
            core,
            specials,
            reactions,
        });
        self.solid.push(solid);
        self.color.push(color);
        self.dedup.insert(key, id);
        self.names.entry(name.to_string()).or_insert(id);
        id
    }

    /// Craft a natural block from a set of elements (equal parts). The natural-tier
    /// crafter the player uses without a machine; also the modding entry point.
    pub fn natural(&mut self, elements: &[ElementId]) -> BlockId {
        let composition = Composition::natural(elements);
        let name = self.auto_name(&composition);
        self.register(&name, composition)
    }

    /// Craft a mixture block from exact element percentages, which must sum to 100.
    /// The first machine-crafted tier.
    pub fn mixture(&mut self, parts: &[(ElementId, u8)]) -> Result<BlockId, MixError> {
        let composition = Composition::mixture(parts)?;
        let name = self.auto_name(&composition);
        Ok(self.register(&name, composition))
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
                .0
                .iter()
                .map(|&(e, p)| format!("{}{}", self.elements.get(e).name, p))
                .collect::<Vec<_>>()
                .join("+"),
            Composition::Computational(_) => "Computer".to_string(),
        }
    }
}

/// A canonical, hashable key for deduplicating compositions. Element order is
/// normalised so two natural blocks with the same elements collapse to one id.
#[derive(Clone, PartialEq, Eq, Hash)]
enum CompKey {
    Natural(Vec<u16>),
    Mix(Vec<(u16, u8)>),
    Config(Vec<(u16, u8)>),
    /// Computational blocks are opaque, so they never dedup — keyed by a unique
    /// registration index instead.
    Computational(usize),
}

impl CompKey {
    fn of(composition: &Composition, fresh_index: usize) -> Self {
        match composition {
            Composition::Natural(els) => {
                let mut v: Vec<u16> = els.iter().map(|e| e.0).collect();
                v.sort_unstable(); // duplicates kept: they change derived weights
                CompKey::Natural(v)
            }
            Composition::Mixture(mix) => CompKey::Mix(sorted_parts(mix)),
            Composition::Configuration { mix, .. } => CompKey::Config(sorted_parts(mix)),
            Composition::Computational(_) => CompKey::Computational(fresh_index),
        }
    }
}

fn sorted_parts(mix: &crate::block::composition::Mix) -> Vec<(u16, u8)> {
    let mut v: Vec<(u16, u8)> = mix.0.iter().map(|&(e, p)| (e.0, p)).collect();
    v.sort_unstable();
    v
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
        for i in 0..reg.block_count() {
            let id = BlockId(i as u16);
            let block = reg.block(id);
            assert_eq!(reg.is_solid(id), derive::derive_solid(&block.composition));
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
        let a = reg.natural(&[El::Stone.id()]);
        // Same as the built-in Stone — must resolve to the existing id, not a new one.
        assert_eq!(a, Blk::Stone.id());
        let before = reg.block_count();
        let b = reg.natural(&[El::Iron.id(), El::Copper.id()]);
        let c = reg.natural(&[El::Copper.id(), El::Iron.id()]); // order-independent
        assert_eq!(b, c);
        assert_eq!(reg.block_count(), before + 1);
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
