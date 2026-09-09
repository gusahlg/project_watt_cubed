//! The block registry: the single source of truth that turns element compositions
//! into the [`BlockId`]s the world stores in every voxel. It is built once at
//! startup and read-only afterwards.
//!
//! Layout is deliberately split hot from cold (the "performance ahead of
//! readability" mandate). Packed flags and compact property arrays indexed by
//! `BlockId` are all the per-voxel mesh/collision path ever reads, so they stay
//! cache-resident. Everything else (composition, derived properties,
//! specials, reactions, names) lives in the cold `blocks` vector that only
//! inspection, crafting, and the future simulation touch.
use std::collections::HashMap;

use voxel_engine::{Color, Pass};

use crate::block::composition::{Composition, MixError};
use crate::block::derive;
use crate::block::derive::SoundClass;
use crate::block::element::{CoreProperties, ElementId, ElementRegistry, SpecialKind};
use crate::block::reaction::{ActiveReaction, ReactionRegistry, apply_reactions};

/// A compact handle to a registered block. Chunks store these through per-chunk
/// palettes (cells stay one byte — see [`ChunkData`](crate::world::chunk::ChunkData)),
/// so widening the id space costs no chunk memory. The [`MAX_BLOCK_TYPES`] cap
/// keeps every id inside the mesh vertex's 14-bit texture-layer field.
///
/// [`MAX_BLOCK_TYPES`]: BlockRegistry::MAX_BLOCK_TYPES
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
    blocks: Vec<Block>,   // cold records
    flags: Vec<u8>,       // HOT — packed collision/cull/liquid/material predicates
    buoyancy: Vec<u8>,    // HOT — ordinary core property; 0 = not a liquid
    layer: Vec<Pass>,     // HOT — mesher routing key; Blend iff solid && !opaque (air's slot is inert)
    emission: Vec<u8>,    // HOT — blocklight seed, 0..=15
    sound: Vec<SoundClass>, // HOT — acoustic class; drives cue naming + absorption
    color: Vec<Color>,    // HOT, indexed by BlockId
    dedup: HashMap<CompKey, BlockId>,
    names: HashMap<Box<str>, BlockId>,
}

/// A cache-resident snapshot of the registry's hot per-voxel tables, indexed by
/// [`BlockId`]. Bundled so meshing and light propagation read one immutable view
/// at one revision (the registry is append-only, so `block_count()` stamps it) —
/// no window in which one property is fresh but another is stale. Handed to worker
/// mesh jobs behind an `Arc` via [`crate::derived::Derived`].
pub struct HotTables {
    /// The boolean properties the meshers probe per face — solid,
    /// opaque, fluid-surface — PACKED one byte per block (see the `FLAG_*` bits
    /// and the [`solid`]/[`opaque`]/[`fluid_surface`] accessors). One array means the
    /// ~14 probes a face sample makes (cull + AO stencil) all hit the same
    /// L1-resident table instead of three parallel ones.
    ///
    /// [`solid`]: HotTables::solid
    /// [`opaque`]: HotTables::opaque
    /// [`fluid_surface`]: HotTables::fluid_surface
    flags: Box<[u8]>,
    /// Draw technique per block — the mesher's routing key. `layer[id] == Opaque`
    /// exactly when the opaque flag (both derived from `transparency`); the flag
    /// serves the branchless cull/AO hot loop, this the pass routing.
    pub layer: Box<[Pass]>,
    pub emission: Box<[u8]>,
    /// Per-block acoustic absorption per metre (`0..=255`), indexed by `BlockId`.
    /// The acoustic occlusion DDA (`World::capture_acoustic_window`) reads this once
    /// per traced cell; open blocks (air, water) are `0`. Derived from the same
    /// physics as the render tables so it shares their snapshot revision.
    pub absorption: Box<[u8]>,
    /// The device's texture-array layer ceiling; the mesher emits
    /// `id % layer_cap` as the vertex layer. Identity while the palette fits
    /// (every id < cap — the common case). Never zero: defaults to `u16::MAX`
    /// and the world stamps the real cap when it refreshes tables.
    pub layer_cap: u16,
    /// Baked corner ambient occlusion — a meshing input the world stamps from
    /// its settings (like `layer_cap`); off reads every corner unoccluded.
    pub ao: bool,
}

const FLAG_SOLID: u8 = 1 << 0;
const FLAG_OPAQUE: u8 = 1 << 1;
const FLAG_LIQUID: u8 = 1 << 2;
/// Liquid AND translucent (`Pass::Blend`) — the engine's animated fluid material.
/// Transparent non-liquids share the pass without receiving this bit.
const FLAG_FLUID_SURFACE: u8 = 1 << 3;

impl HotTables {
    /// Pack one block's flag byte from its boolean properties.
    fn pack(solid: bool, opaque: bool, liquid: bool, fluid_surface: bool) -> u8 {
        ((solid as u8) * FLAG_SOLID)
            | ((opaque as u8) * FLAG_OPAQUE)
            | ((liquid as u8) * FLAG_LIQUID)
            | ((fluid_surface as u8) * FLAG_FLUID_SURFACE)
    }

    /// Build from parallel boolean tables (tests and the registry snapshot).
    pub fn from_parts(
        solid: &[bool],
        opaque: &[bool],
        fluid_surface: &[bool],
        layer: Box<[Pass]>,
        emission: Box<[u8]>,
        absorption: Box<[u8]>,
    ) -> Self {
        debug_assert!(solid.len() == opaque.len() && solid.len() == fluid_surface.len());
        let flags = solid
            .iter()
            .zip(opaque)
            .zip(fluid_surface)
            .map(|((&s, &o), &f)| Self::pack(s, o, f, f))
            .collect();
        Self { flags, layer, emission, absorption, layer_cap: u16::MAX, ao: true }
    }

    /// Whether `id` blocks movement (the mesher's own-cell gate).
    #[inline]
    pub fn solid(&self, id: BlockId) -> bool {
        self.flags[id.0 as usize] & FLAG_SOLID != 0
    }

    /// Whether `id` hides a neighbouring face (the cull/AO probe).
    #[inline]
    pub fn opaque(&self, id: BlockId) -> bool {
        self.flags[id.0 as usize] & FLAG_OPAQUE != 0
    }

    /// Whether `id` takes the animated fluid material bit.
    #[inline]
    pub fn fluid_surface(&self, id: BlockId) -> bool {
        self.flags[id.0 as usize] & FLAG_FLUID_SURFACE != 0
    }

    /// Acoustic absorption per metre (`0..=255`) for `id` — the occlusion
    /// DDA's per-cell probe.
    #[inline]
    pub fn absorption(&self, id: BlockId) -> u8 {
        self.absorption[id.0 as usize]
    }
}

impl Default for HotTables {
    fn default() -> Self {
        Self {
            flags: Box::default(),
            layer: Box::default(),
            emission: Box::default(),
            absorption: Box::default(),
            layer_cap: u16::MAX,
            ao: true,
        }
    }
}

impl BlockRegistry {
    /// A registry preloaded with the built-in elements, reactions, and blocks.
    /// `AIR` is registered first, so it is always [`BlockId(0)`](BlockId).
    pub fn with_builtins() -> Self {
        let mut registry = Self {
            elements: ElementRegistry::with_builtins(),
            reactions: ReactionRegistry::with_builtins(),
            blocks: Vec::new(),
            flags: Vec::new(),
            buoyancy: Vec::new(),
            layer: Vec::new(),
            emission: Vec::new(),
            sound: Vec::new(),
            color: Vec::new(),
            dedup: HashMap::new(),
            names: HashMap::new(),
        };
        registry
            .register("Air", Composition::natural(&[]))
            .expect("air fits in an empty block palette");
        for i in 0..registry.elements.len() {
            let element = ElementId(i as u16);
            let name = registry.elements.get(element).name.clone();
            registry
                .register(&name, Composition::natural(&[element]))
                .expect("pure element blocks fit within the palette cap");
        }
        registry
    }

    /// Whether the block is solid. The single hottest query in the game (collision
    /// every frame, neighbour culling while meshing) — one array load, no branch.
    #[inline]
    pub fn is_solid(&self, id: BlockId) -> bool {
        self.flags[id.0 as usize] & FLAG_SOLID != 0
    }

    /// Whether the block is a passable liquid — collision's third axis. A liquid
    /// is still `solid` (so it meshes), but the movement code swims *through* it
    /// instead of colliding, so [`collides`](crate::world::World::collides) skips
    /// it. One array load, like [`is_solid`](Self::is_solid).
    #[inline]
    pub fn is_liquid(&self, id: BlockId) -> bool {
        self.flags[id.0 as usize] & FLAG_LIQUID != 0
    }

    /// The block's buoyancy strength (`0` for non-liquids) — the upward push and
    /// inverse viscosity a swimmer feels. Read a handful of times per frame while
    /// sampling the water around the player, not per voxel.
    #[inline]
    pub fn buoyancy(&self, id: BlockId) -> u8 {
        self.buoyancy[id.0 as usize]
    }

    /// Whether the block obstructs movement and clearance rays: a solid that is
    /// *not* a passable liquid. Mining deliberately keys off solidity instead, so
    /// liquids can be broken without becoming collidable. Two array loads.
    #[inline]
    pub fn is_obstacle(&self, id: BlockId) -> bool {
        self.is_solid(id) && !self.is_liquid(id)
    }

    /// Whether the block hides the faces behind it — the mesher's cull key. A
    /// translucent solid (glass) is solid but NOT opaque, so faces behind it
    /// still draw. One array load, no branch, like [`is_solid`](Self::is_solid).
    #[inline]
    pub fn is_opaque(&self, id: BlockId) -> bool {
        self.flags[id.0 as usize] & FLAG_OPAQUE != 0
    }

    /// The block's blocklight output, 0..=15. Read only during light propagation.
    #[inline]
    pub fn emission(&self, id: BlockId) -> u8 {
        self.emission[id.0 as usize]
    }

    /// The block's acoustic absorption per metre (`0..=255`) — the occlusion DDA's
    /// per-cell weight. Read once per traced cell by the acoustic window capture.
    #[inline]
    pub fn absorption(&self, id: BlockId) -> u8 {
        self.sound[id.0 as usize].absorption()
    }

    /// The block's acoustic material class as a cue-name stem (`"stone"`, `"soil"`,
    /// `"wood"`, `"glass"`, `"foliage"`, `"water"`) — the naming key the block-cue
    /// table resolves `break_<class>`/`place_<class>`/`step_<class>` against. Same
    /// classification as [`absorption`](Self::absorption), so the two never drift.
    #[inline]
    pub fn sound_class(&self, id: BlockId) -> &'static str {
        self.sound[id.0 as usize].as_str()
    }

    /// A fresh snapshot of the hot per-voxel tables for meshing/light jobs. Cheap
    /// (three small array copies); rebuilt only when the palette grows, behind the
    /// [`Derived`](crate::derived::Derived) revision cache on the world.
    pub fn hot_tables(&self) -> HotTables {
        // A liquid that is also translucent (⇒ `Pass::Blend`) takes the engine's
        // fluid material bit: that shader assumes a see-through reflective
        // surface, so an opaque liquid (e.g. lava, `transparency == 0` ⇒
        // `Pass::Opaque`) must NOT take it.
        HotTables {
            flags: self.flags.clone().into_boxed_slice(),
            layer: self.layer.clone().into_boxed_slice(),
            emission: self.emission.clone().into_boxed_slice(),
            absorption: self.sound.iter().map(|c| c.absorption()).collect(),
            // The registry owns no device or settings knowledge; the world
            // stamps the real cap and AO choice right after (`refresh_tables`).
            layer_cap: u16::MAX,
            ao: true,
        }
    }

    /// The block's render colour. Read once per emitted mesh face.
    #[inline]
    pub fn color(&self, id: BlockId) -> Color {
        self.color[id.0 as usize]
    }

    /// Immutable colour table for background jobs. The element-worldgen
    /// compiler registers additional natural compositions after the builtins,
    /// so recreating a builtin-only registry on a worker is not equivalent.
    pub(crate) fn color_snapshot(&self) -> Box<[Color]> {
        self.color.clone().into_boxed_slice()
    }

    /// The full cold record for a block, for inspection and crafting.
    pub fn block(&self, id: BlockId) -> &Block {
        &self.blocks[id.0 as usize]
    }

    /// The element table, so callers can resolve element ids to names/properties.
    pub fn elements(&self) -> &ElementRegistry {
        &self.elements
    }

    /// Hard cap on distinct block types: the packed mesh vertex carries the
    /// texture-array layer in a 14-bit field (the engine's `MASK_LAYER`), and it
    /// also bounds what remote network specs can make a client's palette (and
    /// texture memory) grow to. Distinct *looks* saturate earlier at the GPU's
    /// `maxImageArrayLayers` (commonly 2048) — past that, layers wrap with a loud
    /// log but registration never fails.
    pub const MAX_BLOCK_TYPES: usize = 16_384;

    /// Whether the palette can still take a NEW composition.
    pub fn at_capacity(&self) -> bool {
        self.blocks.len() >= Self::MAX_BLOCK_TYPES
    }

    /// The already-registered block for this composition, if any (no growth).
    pub fn lookup(&self, composition: &Composition) -> Option<BlockId> {
        self.dedup.get(&CompKey::of(composition)).copied()
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
    /// used for a *different* composition, the first registration wins and
    /// `id_by_name` keeps resolving to it.
    pub fn register(&mut self, name: &str, composition: Composition) -> Option<BlockId> {
        let key = CompKey::of(&composition);
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
        let buoyancy = core.buoyancy;
        let liquid = buoyancy > 0;
        let sound = derive::derive_sound_class(&core, solid);
        let flags = HotTables::pack(solid, opaque, liquid, liquid && layer == Pass::Blend);

        let id = self.push_block(
            Block {
                name: name.into(),
                composition,
                core,
                specials,
                reactions,
            },
            flags,
            buoyancy,
            layer,
            emission,
            sound,
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
    /// freshly assigned [`BlockId`]. The single place the hot flags/property
    /// arrays and the cold `blocks` vector grow together, so they can never desync.
    #[allow(clippy::too_many_arguments)] // SoA append keeps flags/properties in lockstep
    fn push_block(
        &mut self,
        block: Block,
        flags: u8,
        buoyancy: u8,
        layer: Pass,
        emission: u8,
        sound: SoundClass,
        color: Color,
    ) -> BlockId {
        let id = BlockId(self.blocks.len() as u16);
        self.blocks.push(block);
        self.flags.push(flags);
        self.buoyancy.push(buoyancy);
        self.layer.push(layer);
        self.emission.push(emission);
        self.sound.push(sound);
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
            Composition::Mixture(mix) => mix
                .parts()
                .iter()
                .map(|&(e, p)| format!("{}{}", self.elements.get(e).name, p))
                .collect::<Vec<_>>()
                .join("+"),
        }
    }
}

/// Which composition tier a [`CompKey`] came from, so a `Natural` and a
/// `Mixture` with the same weights never collide even though they'd derive the
/// same properties.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Tier {
    Natural,
    Mixture,
}

/// A hashable key for deduplicating compositions: the tier plus its weight
/// multiset (sorted by id, duplicates merged), so two blocks with the same tier
/// and elements collapse to one id.
#[derive(Clone, PartialEq, Eq, Hash)]
struct CompKey(Tier, Vec<(u16, u32)>);

impl CompKey {
    fn of(composition: &Composition) -> Self {
        let tier = match composition {
            Composition::Natural(_) => Tier::Natural,
            Composition::Mixture(_) => Tier::Mixture,
        };
        let parts = composition.weights().parts().iter().map(|&(e, w)| (e.0, w)).collect();
        CompKey(tier, parts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::element::El;

    #[test]
    fn air_is_zero_and_not_solid() {
        let reg = BlockRegistry::with_builtins();
        assert_eq!(AIR, BlockId(0));
        assert!(!reg.is_solid(AIR));
        assert!(reg.is_solid(reg.id_by_name("Stone").unwrap()));
    }

    #[test]
    fn hot_arrays_agree_with_cold_records() {
        let reg = BlockRegistry::with_builtins();
        let hot = reg.hot_tables();
        for i in 0..reg.block_count() {
            let id = BlockId(i as u16);
            let block = reg.block(id);
            let solid = derive::derive_solid(&block.composition);
            assert_eq!(reg.is_solid(id), solid);
            assert_eq!(reg.is_opaque(id), derive::derive_opaque(&block.core, solid));
            assert_eq!(hot.layer[i], derive::derive_layer(&block.core, solid));
            // Blend routing exactly marks translucent solids; air and opaque
            // solids both route Opaque (air's slot is inert — it is never meshed,
            // so "layer == Opaque iff opaque" was a false invariant for non-solids).
            assert_eq!(hot.layer[i] == Pass::Blend, reg.is_solid(id) && !reg.is_opaque(id));
            assert_eq!(reg.emission(id), derive::derive_emission(&block.core));
            assert_eq!(reg.color(id), derive::derive_color(reg.elements(), &block.composition));
        }
    }

    #[test]
    fn stone_keeps_its_slate() {
        // A pure single-element block carries its element's colour exactly —
        // the derivation adds nothing for a one-part composition.
        let reg = BlockRegistry::with_builtins();
        assert_eq!(reg.color(reg.id_by_name("Stone").unwrap()), Color::new(112, 118, 128, 255));
    }

    #[test]
    fn identical_compositions_dedup() {
        let mut reg = BlockRegistry::with_builtins();
        let a = reg.natural(&[El::Stone.id()]).unwrap();
        // Same as the built-in Stone — must resolve to the existing id, not a new one.
        assert_eq!(a, reg.id_by_name("Stone").unwrap());
        let before = reg.block_count();
        let b = reg.natural(&[El::Iron.id(), El::Copper.id()]).unwrap();
        let c = reg.natural(&[El::Copper.id(), El::Iron.id()]).unwrap(); // order-independent
        assert_eq!(b, c);
        assert_eq!(reg.block_count(), before + 1);
    }

    #[test]
    fn duplicated_natural_element_collapses_to_the_set() {
        let mut reg = BlockRegistry::with_builtins();
        let before = reg.block_count();
        // Naturals are sets: [Stone, Stone] canonicalizes to [Stone],
        // so it dedups with the builtin instead of minting a duplicate block
        // that would fail to round-trip through save/network specs.
        let doubled = reg.natural(&[El::Stone.id(), El::Stone.id()]).unwrap();
        assert_eq!(doubled, reg.id_by_name("Stone").unwrap());
        assert_eq!(reg.block_count(), before, "no duplicate variant registered");
    }

    #[test]
    fn registering_past_cap_refuses_rather_than_wraps() {
        use crate::block::element::ElementId;
        let mut reg = BlockRegistry::with_builtins();
        let n = reg.elements().len() as u16;
        // Fill the palette to its cap with distinct two-element mixtures: element
        // pairs × 99 split ratios comfortably exceeds MAX_BLOCK_TYPES.
        'fill: for i in 0..n {
            for j in (i + 1)..n {
                for p in 1..=99u8 {
                    if reg.at_capacity() {
                        break 'fill;
                    }
                    reg.mixture(&[(ElementId(i), p), (ElementId(j), 100 - p)])
                        .expect("distinct mixture registers below the cap");
                }
            }
        }
        assert!(reg.at_capacity(), "test needs enough element pairs to fill the palette");
        assert_eq!(reg.block_count(), BlockRegistry::MAX_BLOCK_TYPES);
        // A brand-new composition past the cap is refused, not truncated into a
        // colliding voxel id. (A natural and a three-way mixture — neither shape
        // was registered by the pair fill.)
        assert_eq!(reg.natural(&[ElementId(0), ElementId(1), ElementId(2), ElementId(3)]), None);
        assert_eq!(
            reg.mixture(&[(ElementId(0), 50), (ElementId(1), 30), (ElementId(2), 20)]),
            Err(MixError::Full)
        );
        assert_eq!(reg.block_count(), BlockRegistry::MAX_BLOCK_TYPES);
    }

    #[test]
    fn mixture_rejects_bad_percentages() {
        let mut reg = BlockRegistry::with_builtins();
        assert!(reg.mixture(&[(El::Soil.id(), 70), (El::Clay.id(), 20)]).is_err());
    }

    #[test]
    fn absorption_classes_and_open_cells() {
        let reg = BlockRegistry::with_builtins();
        let hot = reg.hot_tables();
        // Snapshot mirrors the accessor, indexed by BlockId.
        for i in 0..reg.block_count() {
            assert_eq!(hot.absorption[i], reg.absorption(BlockId(i as u16)));
        }
        // Open cells: air (non-solid) and water (passable liquid) absorb nothing,
        // so `solid && absorption > 0` never fires for them.
        assert_eq!(reg.absorption(AIR), 0);
        let water = reg.id_by_name("Water").unwrap();
        assert!(reg.is_solid(water) && reg.is_liquid(water));
        assert_eq!(reg.absorption(water), 0);
        // Every other solid is an occluding wall (non-zero class).
        for i in 0..reg.block_count() {
            let id = BlockId(i as u16);
            if reg.is_solid(id) && !reg.is_liquid(id) {
                assert!(reg.absorption(id) > 0, "solid wall {} absorbs", reg.block(id).name);
            }
        }
        // Spot-check the intended classes.
        let stone = reg.id_by_name("Stone").unwrap();
        assert_eq!(reg.absorption(stone), 200); // dense
        assert_eq!(reg.sound_class(stone), "stone");
        assert_eq!(reg.absorption(reg.id_by_name("Ice").unwrap()), 60); // translucent
        assert_eq!(reg.sound_class(reg.id_by_name("Ice").unwrap()), "glass");
        assert_eq!(reg.sound_class(AIR), "water"); // non-solid folds into the open class
        assert_eq!(reg.sound_class(water), "water");
        // Every block resolves to one of the six catalog stems game.rs names cues
        // against — guards a future SoundClass variant added without a cue name.
        const STEMS: [&str; 6] = ["stone", "soil", "wood", "glass", "foliage", "water"];
        for i in 0..reg.block_count() {
            let stem = reg.sound_class(BlockId(i as u16));
            assert!(STEMS.contains(&stem), "block {i} has uncatalogued sound class {stem}");
        }
    }

    #[test]
    fn names_resolve() {
        let mut reg = BlockRegistry::with_builtins();
        assert_eq!(reg.id_by_name("Stone"), reg.natural(&[El::Stone.id()]));
        assert_eq!(reg.id_by_name("Nonexistent"), None);
    }

    #[test]
    fn liquid_behavior_is_an_averaged_core_property() {
        let mut reg = BlockRegistry::with_builtins();
        let water = reg.id_by_name("Water").unwrap();
        let wet_stone = reg.natural(&[El::Stone.id(), El::Water.id()]).unwrap();
        assert_eq!(reg.buoyancy(water), 200);
        assert_eq!(reg.buoyancy(wet_stone), 100);
        assert!(reg.is_liquid(wet_stone));
        assert!(!reg.is_obstacle(wet_stone));
        assert_eq!(reg.sound_class(wet_stone), "water");
        assert!(reg.hot_tables().fluid_surface(wet_stone));
        assert!(reg.block(wet_stone).specials.is_empty());
    }
}
