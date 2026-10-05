//! The world's material table: a dynamic intern table of configurations. Voxels store a compact
//! [`BlockId`]; the table maps it to the exact configuration and its kernel record (the law's cached
//! internal supports, so a reaction never rebuilds them), to the hot per-voxel readings the meshers,
//! light, physics and audio consume (SoA, observed once per distinct configuration), to its own
//! texture-array layer (every configuration has a unique look while the device has layers), and to
//! the names a naming mod gave it.

use std::collections::HashMap;

use material::{
    observe, react_once, visual_with, Block, Configuration, Encoding, Law, Observation, Operation,
    Visual,
};
use voxel_engine::{Color, Pass};

use super::naming::{fallback_names, MaterialNamer, MaterialNames, NamingSource};

/// Compact per-voxel material id: the index of a configuration in the world's table.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct BlockId(pub u16);

/// The void configuration: always id 0.
pub const AIR: BlockId = BlockId(0);

/// Configurations one world can hold (the id width).
pub const MAX_BLOCK_TYPES: usize = u16::MAX as usize;
/// Texture-array layers the vertex format can address (14-bit layer field).
pub const MAX_DESCRIPTORS: usize = 16_384;

/// The acoustic class a configuration reads as (audio cue stems and occlusion absorption).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SoundClass {
    /// Dense, hard matter.
    Stone,
    /// Packed earth.
    Soil,
    /// Organic aggregates.
    Wood,
    /// Transparent solids.
    Glass,
    /// Very light matter.
    Foliage,
    /// Air: no wall.
    Open,
}

impl SoundClass {
    /// The cue-catalog stem.
    pub fn as_str(self) -> &'static str {
        match self {
            SoundClass::Stone => "stone",
            SoundClass::Soil => "soil",
            SoundClass::Wood => "wood",
            SoundClass::Glass => "glass",
            SoundClass::Foliage => "foliage",
            SoundClass::Open => "open",
        }
    }

    /// Acoustic absorption per metre for the occlusion DDA.
    pub fn absorption(self) -> u8 {
        match self {
            SoundClass::Stone => 200,
            SoundClass::Soil => 140,
            SoundClass::Wood => 90,
            SoundClass::Glass => 60,
            SoundClass::Foliage => 30,
            SoundClass::Open => 0,
        }
    }

    /// Read the class off an observation: no wall for the void, glass when see-through, then by how
    /// firmly the matter holds together.
    pub fn of(obs: &Observation) -> SoundClass {
        if !obs.solid {
            SoundClass::Open
        } else if obs.transparency >= 128 {
            SoundClass::Glass
        } else if obs.hardness >= 170 {
            SoundClass::Stone
        } else if obs.hardness >= 90 {
            SoundClass::Soil
        } else if obs.hardness >= 40 {
            SoundClass::Wood
        } else {
            SoundClass::Foliage
        }
    }
}

const FLAG_SOLID: u8 = 1 << 0;
const FLAG_OPAQUE: u8 = 1 << 1;

/// A cache-resident snapshot of the hot per-voxel tables, indexed by [`BlockId`]. Bundled so meshing
/// and light propagation read one immutable view at one revision (the table is append-only, so
/// `block_count()` stamps it). Handed to worker mesh jobs behind an `Arc`.
pub struct HotTables {
    /// Packed solid/opaque bits, one byte per block.
    flags: Box<[u8]>,
    /// Draw technique per block — the mesher's routing key.
    pub layer: Box<[Pass]>,
    /// Block-light level 0..15 per block.
    pub emission: Box<[u8]>,
    /// Acoustic absorption per metre per block.
    pub absorption: Box<[u8]>,
    /// Texture-array layer per block.
    render_layer: Box<[u16]>,
    /// The device's texture-array layer ceiling; `render_layer(id)` saturates at `cap - 1`.
    /// Never zero: defaults to `u16::MAX` and the world stamps the real cap when it refreshes tables.
    pub layer_cap: u16,
    /// Baked corner ambient occlusion — a meshing input the world stamps from its settings.
    pub ao: bool,
}

impl HotTables {
    fn pack(solid: bool, opaque: bool) -> u8 {
        ((solid as u8) * FLAG_SOLID) | ((opaque as u8) * FLAG_OPAQUE)
    }

    /// Build from parallel tables (tests).
    pub fn from_parts(
        solid: &[bool],
        opaque: &[bool],
        layer: Box<[Pass]>,
        emission: Box<[u8]>,
        absorption: Box<[u8]>,
        render_layer: Box<[u16]>,
    ) -> Self {
        debug_assert!(solid.len() == opaque.len());
        let flags = solid.iter().zip(opaque).map(|(&s, &o)| Self::pack(s, o)).collect();
        Self { flags, layer, emission, absorption, render_layer, layer_cap: u16::MAX, ao: true }
    }

    /// Whether `id` blocks movement (the mesher's own-cell gate).
    #[inline]
    pub fn solid(&self, id: BlockId) -> bool {
        self.flags[id.0 as usize] & FLAG_SOLID != 0
    }

    /// Whether `id` hides what is behind it (face culling, AO, light opacity).
    #[inline]
    pub fn opaque(&self, id: BlockId) -> bool {
        self.flags[id.0 as usize] & FLAG_OPAQUE != 0
    }

    /// Acoustic absorption per metre.
    #[inline]
    pub fn absorption(&self, id: BlockId) -> u8 {
        self.absorption[id.0 as usize]
    }

    /// The vertex layer for `id`, saturating at the device cap.
    #[inline]
    pub fn render_layer(&self, id: BlockId) -> u16 {
        let layer = self.render_layer[id.0 as usize];
        let cap = self.layer_cap;
        debug_assert!(layer < cap, "layer {layer} exceeds layer cap {cap}");
        layer.min(cap.saturating_sub(1))
    }

    /// Number of blocks covered.
    pub fn len(&self) -> usize {
        self.flags.len()
    }

    /// True when no block is covered.
    pub fn is_empty(&self) -> bool {
        self.flags.is_empty()
    }
}

impl Default for HotTables {
    fn default() -> Self {
        Self {
            flags: Box::default(),
            layer: Box::default(),
            emission: Box::default(),
            absorption: Box::default(),
            render_layer: Box::default(),
            layer_cap: u16::MAX,
            ao: true,
        }
    }
}

/// The dynamic material table of one world. Append-only: an id never changes meaning, so snapshots
/// held by workers stay valid while new configurations are interned on the main thread.
pub struct BlockRegistry {
    law: Law,
    // cold
    configs: Vec<Configuration>,
    blocks: Vec<Block>,
    observations: Vec<Observation>,
    intern: HashMap<Encoding, BlockId>,
    labels: HashMap<BlockId, Box<str>>,
    label_ids: HashMap<Box<str>, BlockId>,
    // hot SoA
    flags: Vec<u8>,
    layer: Vec<Pass>,
    sound: Vec<SoundClass>,
    color: Vec<Color>,
    visuals: Vec<Visual>,
    render_layer: Vec<u16>,
    // render identity: each texture layer shows one configuration
    layer_source: Vec<BlockId>,
    layer_cap: usize,
    layer_cap_set: bool,
    // presentation names
    names: Vec<Option<MaterialNames>>,
    names_revision: Option<u32>,
    /// Ids below this were offered to the current namer.
    named_upto: usize,
}

impl BlockRegistry {
    /// An empty table under `law`, holding only the void as [`AIR`].
    pub fn new(law: Law) -> Self {
        let mut r = Self {
            law,
            configs: Vec::new(),
            blocks: Vec::new(),
            observations: Vec::new(),
            intern: HashMap::new(),
            labels: HashMap::new(),
            label_ids: HashMap::new(),
            flags: Vec::new(),
            layer: Vec::new(),
            sound: Vec::new(),
            color: Vec::new(),
            visuals: Vec::new(),
            render_layer: Vec::new(),
            layer_source: Vec::new(),
            layer_cap: MAX_DESCRIPTORS,
            layer_cap_set: false,
            names: Vec::new(),
            names_revision: None,
            named_upto: 1,
        };
        let air = r.intern(&Configuration::void()).expect("empty table has room");
        debug_assert_eq!(air, AIR);
        r.set_label(AIR, "air");
        r
    }

    /// The table every world starts from: the void under the current law (nothing else is authored).
    pub fn with_builtins() -> Self {
        Self::new(Law::current())
    }

    /// The physics this table observes under.
    pub fn law(&self) -> &Law {
        &self.law
    }

    /// Set the texture-layer ceiling to `min(MAX_DESCRIPTORS, cap)`. First call wins; later calls are
    /// ignored. Default is [`MAX_DESCRIPTORS`] (tests and the server).
    pub fn set_descriptor_cap(&mut self, cap: u16) {
        if self.layer_cap_set {
            return;
        }
        self.layer_cap_set = true;
        self.layer_cap = (cap as usize).clamp(1, MAX_DESCRIPTORS);
    }

    /// Intern a configuration: the existing id if it was seen, else a new one with its kernel record,
    /// readings and texture layer computed once. `None` when the id space is exhausted.
    pub fn intern(&mut self, c: &Configuration) -> Option<BlockId> {
        if let Some(&id) = self.intern.get(&c.encode()) {
            return Some(id);
        }
        self.insert(c.clone(), Block::of(c))
    }

    /// Intern a canonical kernel record (a reaction result), keeping its cached supports.
    pub fn intern_block(&mut self, block: Block) -> Option<BlockId> {
        let c = block.configuration();
        if let Some(&id) = self.intern.get(&c.encode()) {
            return Some(id);
        }
        self.insert(c, block)
    }

    fn insert(&mut self, c: Configuration, block: Block) -> Option<BlockId> {
        if self.configs.len() >= MAX_BLOCK_TYPES {
            return None;
        }
        let id = BlockId(self.configs.len() as u16);
        let obs = observe(&self.law, &block);
        let vis = visual_with(&self.law, &block, &obs);
        let opaque = obs.solid && obs.transparency == 0;
        self.flags.push(HotTables::pack(obs.solid, opaque));
        self.layer.push(if opaque || !obs.solid { Pass::Opaque } else { Pass::Blend });
        self.sound.push(SoundClass::of(&obs));
        self.color.push(Color { r: vis.rgb[0], g: vis.rgb[1], b: vis.rgb[2], a: 255 });
        let layer = self.layer_for(id, &vis);
        self.render_layer.push(layer);
        self.visuals.push(vis);
        self.observations.push(obs);
        self.intern.insert(c.encode(), id);
        self.configs.push(c);
        self.blocks.push(block);
        self.names.push(None);
        Some(id)
    }

    /// A new configuration's texture layer: its own while the device has layers, else the existing
    /// layer whose look is nearest (render identity degrades before material identity). The void
    /// owns layer 0, which the engine requires to be white.
    fn layer_for(&mut self, id: BlockId, vis: &Visual) -> u16 {
        if self.layer_source.len() < self.layer_cap {
            self.layer_source.push(id);
            return (self.layer_source.len() - 1) as u16;
        }
        let mut best = (u32::MAX, 1u16);
        for (layer, &src) in self.layer_source.iter().enumerate().skip(1) {
            let v = &self.visuals[src.0 as usize];
            let dist: u32 = (0..3).map(|k| (v.rgb[k] as i32 - vis.rgb[k] as i32).unsigned_abs()).sum::<u32>()
                + (v.alpha as i32 - vis.alpha as i32).unsigned_abs();
            if dist < best.0 {
                best = (dist, layer as u16);
            }
        }
        best.1
    }

    /// Run ONE operation of the law between `a` and `b` (in canonical physical order: `a` is the
    /// lower cell, or the world cell when `b` is a held tool) and intern both results. `None` when the
    /// contact is quiescent, or when the table cannot hold a result (then nothing changed).
    pub fn react(&mut self, a: BlockId, b: BlockId) -> Option<(Operation, BlockId, BlockId)> {
        let (mut ba, mut bb) = (self.blocks[a.0 as usize].clone(), self.blocks[b.0 as usize].clone());
        let op = react_once(&mut ba, &mut bb)?;
        let na = self.intern_block(ba)?;
        let nb = self.intern_block(bb)?;
        Some((op, na, nb))
    }

    /// True when the contact of `a` and `b` would do nothing (a dormant contact).
    pub fn quiescent(&self, a: BlockId, b: BlockId) -> bool {
        if a == b || (a == AIR && b == AIR) {
            // Identical configurations have zero gain everywhere; two voids have no candidates.
            return true;
        }
        material::Contact::new(&self.blocks[a.0 as usize], &self.blocks[b.0 as usize]).peek().is_none()
    }

    /// The id of a configuration already in the table.
    pub fn lookup(&self, c: &Configuration) -> Option<BlockId> {
        self.intern.get(&c.encode()).copied()
    }

    /// The exact configuration behind an id.
    pub fn configuration(&self, id: BlockId) -> &Configuration {
        &self.configs[id.0 as usize]
    }

    /// The amount of matter in a cell of this id: its occurrence count (0 for air). Conserved by the
    /// law, and the mass gravity sees.
    pub fn amount(&self, id: BlockId) -> u8 {
        self.configs[id.0 as usize].len() as u8
    }

    /// The law's kernel record of an id (cached internal supports).
    pub fn block(&self, id: BlockId) -> &Block {
        &self.blocks[id.0 as usize]
    }

    /// The canonical bytes behind an id.
    pub fn encoding(&self, id: BlockId) -> Encoding {
        self.configs[id.0 as usize].encode()
    }

    /// The readings of an id.
    pub fn observation(&self, id: BlockId) -> Observation {
        self.observations[id.0 as usize]
    }

    /// The visual of an id.
    pub fn visual(&self, id: BlockId) -> Visual {
        self.visuals[id.0 as usize]
    }

    /// The configuration a texture layer shows.
    pub fn layer_source(&self, layer: u16) -> BlockId {
        self.layer_source[layer as usize]
    }

    /// Number of texture layers in use.
    pub fn descriptor_count(&self) -> usize {
        self.layer_source.len()
    }

    /// The texture layer of an id.
    #[inline]
    pub fn render_layer(&self, id: BlockId) -> u16 {
        self.render_layer[id.0 as usize]
    }

    #[inline]
    /// Blocks movement (every non-void configuration).
    pub fn is_solid(&self, id: BlockId) -> bool {
        self.flags[id.0 as usize] & FLAG_SOLID != 0
    }

    #[inline]
    /// Hides what is behind it.
    pub fn is_opaque(&self, id: BlockId) -> bool {
        self.flags[id.0 as usize] & FLAG_OPAQUE != 0
    }

    #[inline]
    /// Block-light level 0..15.
    pub fn emission(&self, id: BlockId) -> u8 {
        self.observations[id.0 as usize].emission
    }

    #[inline]
    /// Acoustic absorption per metre.
    pub fn absorption(&self, id: BlockId) -> u8 {
        self.sound[id.0 as usize].absorption()
    }

    #[inline]
    /// The cue-catalog stem of the sound class.
    pub fn sound_class(&self, id: BlockId) -> &'static str {
        self.sound[id.0 as usize].as_str()
    }

    #[inline]
    /// How firmly the matter holds together, 0..255.
    pub fn hardness(&self, id: BlockId) -> u8 {
        self.observations[id.0 as usize].hardness
    }

    #[inline]
    /// Friction reading, 0..255.
    pub fn friction(&self, id: BlockId) -> u8 {
        self.observations[id.0 as usize].friction
    }

    #[inline]
    /// 0 opaque … 255 fully transparent.
    pub fn transparency(&self, id: BlockId) -> u8 {
        self.observations[id.0 as usize].transparency
    }

    /// Snapshot the hot tables (the world stamps `layer_cap` and `ao` afterwards).
    pub fn hot_tables(&self) -> HotTables {
        HotTables {
            flags: self.flags.clone().into_boxed_slice(),
            layer: self.layer.clone().into_boxed_slice(),
            emission: self.observations.iter().map(|o| o.emission).collect(),
            absorption: self.sound.iter().map(|c| c.absorption()).collect(),
            render_layer: self.render_layer.clone().into_boxed_slice(),
            layer_cap: u16::MAX,
            ao: true,
        }
    }

    #[inline]
    /// The material's base colour (minimap, height mip, HUD swatches).
    pub fn color(&self, id: BlockId) -> Color {
        self.color[id.0 as usize]
    }

    /// All colours by id.
    pub(crate) fn color_snapshot(&self) -> Box<[Color]> {
        self.color.clone().into_boxed_slice()
    }

    /// Number of configurations interned (the revision of every snapshot).
    pub fn block_count(&self) -> usize {
        self.configs.len()
    }

    /// True when no further configuration can be interned.
    pub fn at_capacity(&self) -> bool {
        self.configs.len() >= MAX_BLOCK_TYPES
    }

    /// Attach an internal annotation to an id (worldgen palette roles, tests). Labels never drive
    /// mechanics and are never shown to players. The first label on an id sticks; later aliases still
    /// resolve through [`id_by_label`](Self::id_by_label).
    pub fn set_label(&mut self, id: BlockId, label: &str) {
        let b: Box<str> = label.into();
        self.label_ids.insert(b.clone(), id);
        self.labels.entry(id).or_insert(b);
    }

    /// The label of an id, if any.
    pub fn label(&self, id: BlockId) -> Option<&str> {
        self.labels.get(&id).map(|s| &**s)
    }

    /// The id a label was attached to.
    pub fn id_by_label(&self, label: &str) -> Option<BlockId> {
        self.label_ids.get(label).copied()
    }

    /// Name every configuration interned since the last call (all of them when the namer's revision
    /// changed). Presentation only; cheap when nothing is new.
    pub fn refresh_names(&mut self, namer: Option<&dyn MaterialNamer>) {
        let rev = namer.map(|n| n.revision());
        if rev != self.names_revision {
            self.names_revision = rev;
            self.names.iter_mut().for_each(|n| *n = None);
            self.named_upto = 1;
        }
        let Some(namer) = namer else { return };
        for i in self.named_upto..self.configs.len() {
            let src = NamingSource { law: &self.law, config: &self.configs[i], obs: &self.observations[i] };
            self.names[i] = Some(namer.names(&src));
        }
        self.named_upto = self.configs.len();
    }

    /// What a player calls the block: the naming mod's name, else words read off the observation.
    pub fn display_name(&self, id: BlockId) -> String {
        match &self.names[id.0 as usize] {
            Some(n) if id != AIR => n.block.clone(),
            _ => fallback_names(&self.observations[id.0 as usize]).block,
        }
    }

    /// What a player calls the same configuration held as a tool.
    pub fn tool_name(&self, id: BlockId) -> String {
        if id == AIR {
            return "hand".to_string();
        }
        match &self.names[id.0 as usize] {
            Some(n) => n.tool.clone(),
            None => fallback_names(&self.observations[id.0 as usize]).tool,
        }
    }

    /// Text form of an id for saves and the wire: `air` or `c:<hex of the encoding>`.
    pub fn spec(&self, id: BlockId) -> String {
        if id == AIR {
            return "air".to_string();
        }
        let bytes = self.encoding(id);
        let mut s = String::with_capacity(2 + bytes.as_bytes().len() * 2);
        s.push_str("c:");
        for b in bytes.as_bytes() {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }

    /// Inverse of [`BlockRegistry::spec`]: interns the configuration. `None` for malformed or legacy
    /// (named) specs, for `c:00` (only `air` spells the void), and when the table is full.
    pub fn parse_spec(&mut self, spec: &str) -> Option<BlockId> {
        match self.read_spec(spec) {
            SpecKind::Ok(id) => Some(id),
            _ => None,
        }
    }

    /// Canonical wire spelling of `spec` without interning. `Some("air")`, or `Some("c:…")`
    /// in lowercase hex for a decodable configuration (already known or not). `None` when
    /// the text is malformed or is `c:00` (only `air` spells the void).
    pub fn canonical_spec(&self, spec: &str) -> Option<String> {
        if spec == "air" {
            return Some("air".to_string());
        }
        if let Some(id) = self.lookup_spec(spec) {
            return Some(self.spec(id));
        }
        let cfg = decode_spec(spec)?;
        if cfg.is_empty() {
            return None;
        }
        let bytes = cfg.encode();
        let mut s = String::with_capacity(2 + bytes.as_bytes().len() * 2);
        s.push_str("c:");
        for b in bytes.as_bytes() {
            s.push_str(&format!("{b:02x}"));
        }
        Some(s)
    }

    /// Look up a spec already in the table without interning. `None` if the spec is malformed,
    /// `c:00`, or the configuration has not been interned yet.
    pub fn lookup_spec(&self, spec: &str) -> Option<BlockId> {
        let c = decode_spec(spec)?;
        if c.is_empty() {
            return (spec == "air").then_some(AIR);
        }
        self.lookup(&c)
    }

    /// Classify a spec so the save loader can word its notice by cause.
    pub(crate) fn read_spec(&mut self, spec: &str) -> SpecKind {
        if spec == "air" {
            return SpecKind::Ok(AIR);
        }
        match decode_spec(spec) {
            None => {
                if spec.starts_with("c:") {
                    SpecKind::Bad
                } else {
                    SpecKind::Legacy
                }
            }
            Some(c) if c.is_empty() => SpecKind::Bad,
            Some(c) => match self.intern(&c) {
                Some(id) => SpecKind::Ok(id),
                None => SpecKind::Full,
            },
        }
    }
}

/// Why [`BlockRegistry::read_spec`] could not intern a spec.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SpecKind {
    Ok(BlockId),
    /// Named form from before the material model (`natural:Stone`, …).
    Legacy,
    /// Well-formed encoding that could not be interned (id space full).
    Full,
    /// Malformed (`c:00`, a leading `+`, odd nibble, …).
    Bad,
}

/// `air` or `c:<hex of the encoding>`. `None` for legacy names, odd nibbles, non-hex (including a
/// leading `+`), non-ASCII, or a truncated/oversize payload — hostile wire/save input must not panic.
fn decode_spec(spec: &str) -> Option<Configuration> {
    if spec == "air" {
        return Some(Configuration::void());
    }
    let hex = spec.strip_prefix("c:")?;
    if hex.len() % 2 != 0 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let bytes: Option<Vec<u8>> = hex
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| std::str::from_utf8(pair).ok().and_then(|s| u8::from_str_radix(s, 16).ok()))
        .collect();
    Configuration::decode(&bytes?).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use material::Element;

    fn cfg(elems: &[[u8; 4]]) -> Configuration {
        Configuration::new(elems.iter().map(|c| Element::new(*c)).collect::<Vec<_>>()).unwrap()
    }

    #[test]
    fn air_is_id_zero_and_reads_as_air() {
        let r = BlockRegistry::with_builtins();
        assert_eq!(r.lookup(&Configuration::void()), Some(AIR));
        assert!(!r.is_solid(AIR) && !r.is_opaque(AIR));
        assert_eq!(r.emission(AIR), 0);
        assert_eq!(r.sound_class(AIR), "open");
        assert_eq!(r.label(AIR), Some("air"));
        assert_eq!(r.spec(AIR), "air");
        assert_eq!(r.render_layer(AIR), 0, "the void owns the white layer");
        assert_eq!(r.block_count(), 1);
        assert_eq!(r.display_name(AIR), "air");
    }

    #[test]
    fn intern_dedups_multisets_and_keeps_multiplicity() {
        let mut r = BlockRegistry::with_builtins();
        let a = cfg(&[[10, 20, 30, 40], [50, 60, 70, 80]]);
        let b = cfg(&[[50, 60, 70, 80], [10, 20, 30, 40]]);
        let aa = cfg(&[[10, 20, 30, 40], [10, 20, 30, 40]]);
        let ia = r.intern(&a).unwrap();
        assert_eq!(r.intern(&b), Some(ia), "storage order is not identity");
        let iaa = r.intern(&aa).unwrap();
        assert_ne!(ia, iaa);
        assert_eq!(r.configuration(ia), &a);
        assert_eq!(r.block_count(), 3);
    }

    #[test]
    fn every_configuration_gets_its_own_layer_until_the_cap() {
        let mut r = BlockRegistry::with_builtins();
        r.set_descriptor_cap(4);
        let ids: Vec<_> = (0..6u8).map(|i| r.intern(&cfg(&[[i * 40, 3, 9, 27]])).unwrap()).collect();
        assert_eq!(r.render_layer(ids[0]), 1);
        assert_eq!(r.render_layer(ids[2]), 3);
        assert_eq!(r.descriptor_count(), 4);
        for &id in &ids[3..] {
            let l = r.render_layer(id);
            assert!((1..4).contains(&l), "past the cap: nearest existing layer, never the void's");
        }
        assert_eq!(r.layer_source(2), ids[1]);
    }

    #[test]
    fn hot_tables_match_the_accessors() {
        let mut r = BlockRegistry::with_builtins();
        let mut ids = Vec::new();
        for i in 0..40u8 {
            ids.push(r.intern(&cfg(&[[i * 6, 200 - i * 3, i * 5, 90 + i], [i, i, 7, 3]])).unwrap());
        }
        let hot = r.hot_tables();
        assert_eq!(hot.len(), r.block_count());
        for id in ids {
            assert_eq!(hot.solid(id), r.is_solid(id));
            assert!(r.is_solid(id), "every non-void configuration is solid");
            assert_eq!(hot.opaque(id), r.is_opaque(id));
            assert_eq!(hot.absorption(id), r.absorption(id));
            assert_eq!(hot.render_layer(id), r.render_layer(id));
            assert_eq!(hot.emission[id.0 as usize], r.emission(id));
            assert_eq!(hot.opaque(id), hot.layer[id.0 as usize] == Pass::Opaque);
            assert_eq!(r.observation(id), observe(r.law(), r.block(id)));
        }
    }

    #[test]
    fn react_interns_both_results_and_conserves_elements() {
        let mut r = BlockRegistry::with_builtins();
        // The reference destructive exception: A loses all four constituents to E.
        let a = r
            .intern(&cfg(&[[73, 145, 162, 161], [71, 77, 157, 208], [34, 125, 217, 144], [8, 85, 210, 206]]))
            .unwrap();
        let e = r
            .intern(&cfg(&[[83, 135, 211, 195], [51, 125, 144, 147], [11, 72, 167, 145], [25, 80, 211, 204]]))
            .unwrap();
        assert!(!r.quiescent(a, e));
        let (mut ca, mut ce) = (a, e);
        let mut ops = 0;
        while let Some((_, na, ne)) = r.react(ca, ce) {
            assert_eq!(
                r.configuration(na).len() + r.configuration(ne).len(),
                8,
                "occurrences are conserved"
            );
            (ca, ce) = (na, ne);
            ops += 1;
        }
        assert_eq!(ops, 4);
        assert_eq!(ca, AIR, "A emptied into E");
        assert_eq!(r.configuration(ce).len(), 8);
        assert!(r.quiescent(ca, ce));
        assert!(r.quiescent(e, e), "identical configurations never react");
    }

    #[test]
    fn spec_text_round_trips_and_rejects_hostile_input() {
        let mut r = BlockRegistry::with_builtins();
        let c = cfg(&[[1, 2, 3, 4], [250, 251, 252, 253], [1, 2, 3, 4]]);
        let id = r.intern(&c).unwrap();
        let spec = r.spec(id);
        assert!(spec.starts_with("c:03"));
        let mut r2 = BlockRegistry::with_builtins();
        let parsed = r2.parse_spec(&spec).unwrap();
        assert_eq!(r2.configuration(parsed), &c);
        assert_eq!(r2.parse_spec("air"), Some(AIR));
        assert_eq!(r2.parse_spec("natural:Stone,Iron"), None);
        for bad in ["c:zz", "c:0201020304", "c:aéa", "c:é", "c:", "c:0", "c:00", "c:00ff", "c:21", " c:00", "c:01+f+f+f+f"] {
            assert_eq!(r2.parse_spec(bad), None, "{bad:?}");
        }
        assert_eq!(r2.read_spec("natural:Stone"), SpecKind::Legacy);
        assert_eq!(r2.read_spec("c:00"), SpecKind::Bad);
        let hex = spec.trim_start_matches("c:");
        assert_eq!(r.parse_spec(&format!("c:{}", hex.to_ascii_uppercase())), Some(id));
        assert_eq!(r.lookup_spec("c:00"), None);
    }

    #[test]
    fn labels_are_annotations_only() {
        let mut r = BlockRegistry::with_builtins();
        let id = r.intern(&cfg(&[[120, 130, 140, 150]])).unwrap();
        r.set_label(id, "rock");
        assert_eq!(r.id_by_label("rock"), Some(id));
        assert!(r.spec(id).starts_with("c:"), "the spec never carries the label");
        assert!(!r.display_name(id).contains("rock"), "labels are never shown");
    }

    struct Fixed(u32);
    impl MaterialNamer for Fixed {
        fn names(&self, src: &NamingSource) -> MaterialNames {
            MaterialNames { block: format!("m{}", src.config.len()), tool: format!("t{}", self.0) }
        }
        fn revision(&self) -> u32 {
            self.0
        }
    }

    #[test]
    fn names_come_from_the_namer_and_follow_its_revision() {
        let mut r = BlockRegistry::with_builtins();
        let id = r.intern(&cfg(&[[1, 1, 1, 1], [2, 2, 2, 2]])).unwrap();
        let described = r.display_name(id);
        r.refresh_names(Some(&Fixed(1)));
        assert_eq!(r.display_name(id), "m2");
        assert_eq!(r.tool_name(id), "t1");
        let late = r.intern(&cfg(&[[3, 1, 1, 1]])).unwrap();
        r.refresh_names(Some(&Fixed(1)));
        assert_eq!(r.display_name(late), "m1", "new configurations are named on the next refresh");
        r.refresh_names(Some(&Fixed(2)));
        assert_eq!(r.tool_name(id), "t2", "a revision bump renames everything");
        r.refresh_names(None);
        assert_eq!(r.display_name(id), described, "no namer: the description");
        assert_eq!(r.tool_name(AIR), "hand");
    }

    #[test]
    fn intern_returns_none_at_u16_cap() {
        let mut r = BlockRegistry::with_builtins();
        let mut n = 1u32;
        while r.block_count() < MAX_BLOCK_TYPES {
            let e = Element::new([n as u8, (n >> 8) as u8, (n >> 16) as u8, 7]);
            assert!(r.intern(&Configuration::single(e)).is_some(), "slot {n}");
            n += 1;
        }
        assert!(r.at_capacity());
        assert!(r.intern(&cfg(&[[9, 8, 7, 6], [1, 2, 3, 4]])).is_none(), "id space is exhausted");
    }
}
