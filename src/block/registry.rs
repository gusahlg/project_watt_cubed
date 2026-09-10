//! The world's material table under the emergent model: a dynamic intern table of configurations.
//! Voxels store a compact [`BlockId`]; the table maps it to the exact configuration (cold), to the
//! hot per-voxel readings the meshers, light, physics and audio consume (SoA, observed once per
//! distinct configuration through the law's probes), and to a render descriptor shared by every
//! configuration that looks alike (the texture-array layer index — render identity is decoupled from
//! material identity, so the 14-bit vertex field never caps the number of materials).

use std::collections::HashMap;

use material::{
    interact, observe, visual, Configuration, DescriptorKey, Encoding, EventKind, Law, Observation,
    Visual,
};
use voxel_engine::{Color, Pass};

/// Compact per-voxel material id: the index of a configuration in the world's table.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct BlockId(pub u16);

/// The void configuration: always id 0.
pub const AIR: BlockId = BlockId(0);

/// Repeat `interact` `repeat` times; origin is unchanged (law v0).
pub fn interact_repeat(
    law: &Law,
    origin: &Configuration,
    target: &Configuration,
    event: EventKind,
    repeat: u8,
) -> Configuration {
    let mut t = target.clone();
    for _ in 0..repeat {
        t = interact(law, origin, &t, event).target;
    }
    t
}

/// Configurations one world can hold (the id width).
pub const MAX_BLOCK_TYPES: usize = u16::MAX as usize;
/// Render descriptors (texture-array layers) — the engine vertex carries a 14-bit layer.
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
    /// Air and liquids: no wall.
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
            // The checked-in cue catalog retains its historical `water` stem.
            SoundClass::Open => "water",
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

    /// Read the class off an observation: no wall for void/liquid, glass when see-through, then by
    /// how stable the material is under contact.
    pub fn of(obs: &Observation) -> SoundClass {
        if !obs.solid || obs.liquid {
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
const FLAG_LIQUID: u8 = 1 << 2;
/// Liquid AND translucent (`Pass::Blend`) — the engine's animated fluid material.
const FLAG_FLUID_SURFACE: u8 = 1 << 3;

/// A cache-resident snapshot of the hot per-voxel tables, indexed by [`BlockId`]. Bundled so meshing
/// and light propagation read one immutable view at one revision (the table is append-only, so
/// `block_count()` stamps it). Handed to worker mesh jobs behind an `Arc`.
pub struct HotTables {
    /// Packed solid/opaque/liquid/fluid-surface bits, one byte per block.
    flags: Box<[u8]>,
    /// Draw technique per block — the mesher's routing key.
    pub layer: Box<[Pass]>,
    /// Block-light level 0..15 per block.
    pub emission: Box<[u8]>,
    /// Acoustic absorption per metre per block.
    pub absorption: Box<[u8]>,
    /// Render descriptor (texture-array layer) per block.
    render_layer: Box<[u16]>,
    /// The device's texture-array layer ceiling; `render_layer(id)` is reduced modulo it. Never zero:
    /// defaults to `u16::MAX` and the world stamps the real cap when it refreshes tables.
    pub layer_cap: u16,
    /// Baked corner ambient occlusion — a meshing input the world stamps from its settings.
    pub ao: bool,
}

impl HotTables {
    fn pack(solid: bool, opaque: bool, liquid: bool, fluid_surface: bool) -> u8 {
        ((solid as u8) * FLAG_SOLID)
            | ((opaque as u8) * FLAG_OPAQUE)
            | ((liquid as u8) * FLAG_LIQUID)
            | ((fluid_surface as u8) * FLAG_FLUID_SURFACE)
    }

    /// Build from parallel tables (tests).
    pub fn from_parts(
        solid: &[bool],
        opaque: &[bool],
        fluid_surface: &[bool],
        layer: Box<[Pass]>,
        emission: Box<[u8]>,
        absorption: Box<[u8]>,
        render_layer: Box<[u16]>,
    ) -> Self {
        debug_assert!(solid.len() == opaque.len() && solid.len() == fluid_surface.len());
        let flags = solid
            .iter()
            .zip(opaque)
            .zip(fluid_surface)
            .map(|((&s, &o), &f)| Self::pack(s, o, f, f))
            .collect();
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

    /// Whether `id` draws with the engine's animated fluid material.
    #[inline]
    pub fn fluid_surface(&self, id: BlockId) -> bool {
        self.flags[id.0 as usize] & FLAG_FLUID_SURFACE != 0
    }

    /// Acoustic absorption per metre.
    #[inline]
    pub fn absorption(&self, id: BlockId) -> u8 {
        self.absorption[id.0 as usize]
    }

    /// The vertex layer for `id`: its render descriptor, reduced modulo the device cap.
    #[inline]
    pub fn render_layer(&self, id: BlockId) -> u16 {
        self.render_layer[id.0 as usize] % self.layer_cap
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
    intern: HashMap<Encoding, BlockId>,
    labels: HashMap<BlockId, Box<str>>,
    label_ids: HashMap<Box<str>, BlockId>,
    // hot SoA
    flags: Vec<u8>,
    layer: Vec<Pass>,
    emission: Vec<u8>,
    sound: Vec<SoundClass>,
    color: Vec<Color>,
    hardness: Vec<u8>,
    friction: Vec<u8>,
    transparency: Vec<u8>,
    buoyancy: Vec<u8>,
    visuals: Vec<Visual>,
    render_layer: Vec<u16>,
    // render identity
    descriptors: Vec<Visual>,
    descriptor_intern: HashMap<DescriptorKey, u16>,
}

impl BlockRegistry {
    /// An empty table under `law`, holding only the void as [`AIR`].
    pub fn new(law: Law) -> Self {
        let mut r = Self {
            law,
            configs: Vec::new(),
            intern: HashMap::new(),
            labels: HashMap::new(),
            label_ids: HashMap::new(),
            flags: Vec::new(),
            layer: Vec::new(),
            emission: Vec::new(),
            sound: Vec::new(),
            color: Vec::new(),
            hardness: Vec::new(),
            friction: Vec::new(),
            transparency: Vec::new(),
            buoyancy: Vec::new(),
            visuals: Vec::new(),
            render_layer: Vec::new(),
            descriptors: Vec::new(),
            descriptor_intern: HashMap::new(),
        };
        let air = r.intern(&Configuration::void()).expect("empty table has room");
        debug_assert_eq!(air, AIR);
        r.set_label(AIR, "air");
        r
    }

    /// The table every world starts from: the void under the current law (nothing else is authored).
    pub fn with_builtins() -> Self {
        Self::new(Law::v0())
    }

    /// The physics this table observes under.
    pub fn law(&self) -> &Law {
        &self.law
    }

    /// Apply `event` from `origin` onto `target`, `repeat` times, and intern the result.
    /// Origin is unchanged (law v0). `None` when the id space is exhausted.
    pub fn apply_interaction(
        &mut self,
        origin: &Configuration,
        target: &Configuration,
        event: EventKind,
        repeat: u8,
    ) -> Option<BlockId> {
        let law = self.law;
        let result = interact_repeat(&law, origin, target, event, repeat);
        self.intern(&result)
    }

    /// Parse origin/target specs, apply [`apply_interaction`], return the interned result.
    /// `None` when a spec is malformed or the table is full.
    pub fn apply_specs(
        &mut self,
        origin_spec: &str,
        target_spec: &str,
        event: EventKind,
        repeat: u8,
    ) -> Option<BlockId> {
        let origin_id = self.parse_spec(origin_spec)?;
        let target_id = self.parse_spec(target_spec)?;
        let origin = self.configs[origin_id.0 as usize].clone();
        let target = self.configs[target_id.0 as usize].clone();
        self.apply_interaction(&origin, &target, event, repeat)
    }

    /// What a player is told about a material: a description read off the law (its observed
    /// properties), never an authored name. Labels are worldgen annotations and stay internal; the
    /// names players give their own products live in their journals.
    pub fn display_name(&self, id: BlockId) -> String {
        describe(&self.observation(id))
    }

    /// Intern a configuration: the existing id if it was seen, else a new one with its readings and
    /// render descriptor computed once. `None` when the id space is exhausted.
    pub fn intern(&mut self, c: &Configuration) -> Option<BlockId> {
        let key = c.encode();
        if let Some(&id) = self.intern.get(&key) {
            return Some(id);
        }
        if self.configs.len() >= MAX_BLOCK_TYPES {
            return None;
        }
        let id = BlockId(self.configs.len() as u16);
        let obs = observe(&self.law, c);
        let vis = visual(&self.law, c);
        // Liquids mesh (the engine's fluid surface) but stay passable: `is_solid`
        // is the mesher/mining gate, `is_obstacle` strips liquids for collision.
        let meshes = obs.solid || obs.liquid;
        let opaque = obs.solid && obs.transparency == 0;
        let pass = if opaque { Pass::Opaque } else { Pass::Blend };
        let fluid_surface = obs.liquid && pass == Pass::Blend;
        let sound = SoundClass::of(&obs);
        self.flags.push(HotTables::pack(meshes, opaque, obs.liquid, fluid_surface));
        self.layer.push(pass);
        self.emission.push(obs.emission);
        self.sound.push(sound);
        self.color.push(Color { r: vis.rgb[0], g: vis.rgb[1], b: vis.rgb[2], a: 255 });
        self.hardness.push(obs.hardness);
        self.friction.push(obs.friction);
        self.transparency.push(obs.transparency);
        self.buoyancy.push(if obs.liquid { obs.flow.max(64) } else { 0 });
        let descriptor = self.descriptor_for(vis);
        self.render_layer.push(descriptor);
        self.visuals.push(vis);
        self.configs.push(c.clone());
        self.intern.insert(key, id);
        Some(id)
    }

    /// The descriptor id for a visual: an existing key, a new one while the layer space lasts, else
    /// the nearest existing descriptor by colour (render identity degrades before material identity).
    fn descriptor_for(&mut self, vis: Visual) -> u16 {
        let key = vis.quantize();
        if let Some(&d) = self.descriptor_intern.get(&key) {
            return d;
        }
        if self.descriptors.len() < MAX_DESCRIPTORS {
            let d = self.descriptors.len() as u16;
            self.descriptors.push(Visual::dequantize(key));
            self.descriptor_intern.insert(key, d);
            return d;
        }
        let mut best = (u32::MAX, 0u16);
        for (i, v) in self.descriptors.iter().enumerate() {
            let dist: u32 = (0..3).map(|k| (v.rgb[k] as i32 - vis.rgb[k] as i32).unsigned_abs()).sum();
            if dist < best.0 {
                best = (dist, i as u16);
            }
        }
        // Remember the miss so a later intern of the same visual is O(1).
        self.descriptor_intern.insert(key, best.1);
        best.1
    }

    /// The id of a configuration already in the table.
    pub fn lookup(&self, c: &Configuration) -> Option<BlockId> {
        self.intern.get(&c.encode()).copied()
    }

    /// The exact configuration behind an id.
    pub fn configuration(&self, id: BlockId) -> &Configuration {
        &self.configs[id.0 as usize]
    }

    /// The canonical bytes behind an id.
    pub fn encoding(&self, id: BlockId) -> Encoding {
        self.configs[id.0 as usize].encode()
    }

    /// The readings of an id (recomputed from the stored tables; not for hot paths).
    pub fn observation(&self, id: BlockId) -> Observation {
        observe(&self.law, &self.configs[id.0 as usize])
    }

    /// The exact (unquantized) visual of an id.
    pub fn visual(&self, id: BlockId) -> Visual {
        self.visuals[id.0 as usize]
    }

    /// The representative visual of a render descriptor (texture-array layer).
    pub fn descriptor(&self, layer: u16) -> Visual {
        self.descriptors[layer as usize]
    }

    /// Number of render descriptors in use.
    pub fn descriptor_count(&self) -> usize {
        self.descriptors.len()
    }

    /// The render descriptor of an id.
    #[inline]
    pub fn render_layer(&self, id: BlockId) -> u16 {
        self.render_layer[id.0 as usize]
    }

    #[inline]
    /// Blocks movement.
    pub fn is_solid(&self, id: BlockId) -> bool {
        self.flags[id.0 as usize] & FLAG_SOLID != 0
    }

    #[inline]
    /// Flows and floats things.
    pub fn is_liquid(&self, id: BlockId) -> bool {
        self.flags[id.0 as usize] & FLAG_LIQUID != 0
    }

    #[inline]
    /// Buoyancy strength (0 for non-liquids).
    pub fn buoyancy(&self, id: BlockId) -> u8 {
        self.buoyancy[id.0 as usize]
    }

    #[inline]
    /// Solid and not liquid: the collision obstacle test.
    pub fn is_obstacle(&self, id: BlockId) -> bool {
        self.is_solid(id) && !self.is_liquid(id)
    }

    #[inline]
    /// Hides what is behind it.
    pub fn is_opaque(&self, id: BlockId) -> bool {
        self.flags[id.0 as usize] & FLAG_OPAQUE != 0
    }

    #[inline]
    /// Block-light level 0..15.
    pub fn emission(&self, id: BlockId) -> u8 {
        self.emission[id.0 as usize]
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
    /// Stability under contact, 0..255 (mining time and impact resistance read this).
    pub fn hardness(&self, id: BlockId) -> u8 {
        self.hardness[id.0 as usize]
    }

    #[inline]
    /// Friction reading, 0..255.
    pub fn friction(&self, id: BlockId) -> u8 {
        self.friction[id.0 as usize]
    }

    #[inline]
    /// 0 opaque … 255 fully transparent.
    pub fn transparency(&self, id: BlockId) -> u8 {
        self.transparency[id.0 as usize]
    }

    /// Snapshot the hot tables (the world stamps `layer_cap` and `ao` afterwards).
    pub fn hot_tables(&self) -> HotTables {
        HotTables {
            flags: self.flags.clone().into_boxed_slice(),
            layer: self.layer.clone().into_boxed_slice(),
            emission: self.emission.clone().into_boxed_slice(),
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

    /// Attach a debug/semantic label to an id (worldgen regions, tests). Labels never drive mechanics.
    /// The first label on an id is its display name; later aliases still resolve through
    /// [`id_by_label`](Self::id_by_label) without overwriting it.
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

    /// Text form of an id for saves and the wire: `air` or `c:<hex of the encoding>`.
    pub fn spec(&self, id: BlockId) -> String {
        if id == AIR {
            return "air".to_string();
        }
        let mut s = String::with_capacity(2 + self.configs[id.0 as usize].len() * 8 + 2);
        s.push_str("c:");
        for b in self.encoding(id).as_bytes() {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }

    /// Inverse of [`BlockRegistry::spec`]: interns the configuration. `None` for malformed or legacy
    /// (named) specs and when the table is full.
    pub fn parse_spec(&mut self, spec: &str) -> Option<BlockId> {
        self.intern(&decode_spec(spec)?)
    }

    /// Look up a spec already in the table without interning. `None` if the spec is
    /// malformed or the configuration has not been interned yet.
    pub fn lookup_spec(&self, spec: &str) -> Option<BlockId> {
        self.lookup(&decode_spec(spec)?)
    }
}

/// Words for an observation: phase, hardness band, clarity, glow, grip — each a threshold on a probe
/// response, so two configurations that read alike are described alike and a law change re-words
/// the world by itself. The void is "air".
pub fn describe(obs: &Observation) -> String {
    if !obs.solid && !obs.liquid {
        return "air".to_string();
    }
    let mut words: Vec<&str> = Vec::with_capacity(5);
    if obs.emission > 0 {
        words.push("glowing");
    }
    match obs.transparency {
        200..=255 => words.push("clear"),
        100..=199 => words.push("hazy"),
        _ => {}
    }
    if obs.liquid {
        words.push(if obs.flow >= 200 { "thin liquid" } else { "liquid" });
    } else {
        words.push(match obs.hardness {
            200..=255 => "hard",
            100..=199 => "firm",
            _ => "soft",
        });
        match obs.friction {
            192..=255 => words.push("rough"),
            0..=63 => words.push("slick"),
            _ => {}
        }
        words.push("solid");
    }
    words.join(" ")
}

/// `air` or `c:<hex of the encoding>`. `None` for legacy names, odd nibbles, non-ASCII, or
/// a truncated/oversize payload — hostile wire/save input must not panic.
fn decode_spec(spec: &str) -> Option<Configuration> {
    if spec == "air" {
        return Some(Configuration::void());
    }
    let hex = spec.strip_prefix("c:")?;
    if !hex.is_ascii() || hex.len() % 2 != 0 {
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
    use material::{observe, Element, Visual};

    fn cfg(elems: &[[u8; 4]]) -> Configuration {
        Configuration::new(elems.iter().map(|c| Element::new(*c)).collect::<Vec<_>>()).unwrap()
    }

    #[test]
    fn air_is_id_zero_and_reads_as_air() {
        let r = BlockRegistry::with_builtins();
        assert_eq!(r.lookup(&Configuration::void()), Some(AIR));
        assert!(!r.is_solid(AIR) && !r.is_liquid(AIR) && !r.is_opaque(AIR));
        assert_eq!(r.emission(AIR), 0);
        assert_eq!(r.sound_class(AIR), "water");
        assert_eq!(r.label(AIR), Some("air"));
        assert_eq!(r.spec(AIR), "air");
        assert_eq!(r.block_count(), 1);
    }

    #[test]
    fn intern_dedups_and_keeps_order_and_multiplicity() {
        let mut r = BlockRegistry::with_builtins();
        let a = cfg(&[[10, 20, 30, 40], [50, 60, 70, 80]]);
        let b = cfg(&[[50, 60, 70, 80], [10, 20, 30, 40]]);
        let aa = cfg(&[[10, 20, 30, 40], [10, 20, 30, 40]]);
        let ia = r.intern(&a).unwrap();
        assert_eq!(r.intern(&a), Some(ia));
        let ib = r.intern(&b).unwrap();
        let iaa = r.intern(&aa).unwrap();
        assert!(ia != ib && ib != iaa && ia != iaa);
        assert_eq!(r.configuration(ia), &a);
        assert_eq!(r.block_count(), 4);
    }

    #[test]
    fn hot_tables_match_the_accessors() {
        let mut r = BlockRegistry::with_builtins();
        let mut ids = Vec::new();
        for i in 0..40u8 {
            ids.push(r.intern(&cfg(&[[i * 6, 200 - i * 3, i * 5, 90 + i]])).unwrap());
        }
        let hot = r.hot_tables();
        assert_eq!(hot.len(), r.block_count());
        for id in ids {
            assert_eq!(hot.solid(id), r.is_solid(id));
            assert_eq!(hot.opaque(id), r.is_opaque(id));
            assert_eq!(hot.absorption(id), r.absorption(id));
            assert_eq!(hot.render_layer(id), r.render_layer(id));
            assert_eq!(hot.fluid_surface(id), r.is_liquid(id) && hot.layer[id.0 as usize] == Pass::Blend);
            assert_eq!(hot.opaque(id), hot.layer[id.0 as usize] == Pass::Opaque);
        }
    }

    #[test]
    fn material_families_share_render_descriptors() {
        // A worldgen region = a centre and variants within spread 8; a family must not need a
        // descriptor per member, or render identity would proliferate with material identity.
        let mut r = BlockRegistry::with_builtins();
        let mut layers_per_family = 0usize;
        for f in 0..40u8 {
            let centre = [f.wrapping_mul(37), f.wrapping_mul(91), f.wrapping_mul(53), f.wrapping_mul(17)];
            let mut layers = std::collections::HashSet::new();
            for v in 0..7u8 {
                let mut e = centre;
                e[(v % 4) as usize] = e[(v % 4) as usize].saturating_add(v * 2 % 9);
                e[((v + 1) % 4) as usize] = e[((v + 1) % 4) as usize].saturating_sub(v % 5);
                let id = r.intern(&cfg(&[e])).unwrap();
                layers.insert(r.render_layer(id));
            }
            layers_per_family += layers.len();
        }
        // Measured for law v0 with 4-4-4 colour bins: ~5.3 of 7. The colour drift per lattice step
        // (≤ 32/765) is the feature — related materials look related — while render identity still
        // grows slower than material identity (the cap-management goal).
        let avg = layers_per_family as f64 / 40.0;
        assert!(avg <= 6.0, "a family of 7 variants used {avg:.2} descriptors on average");
        assert!(r.descriptor_count() * 100 <= r.block_count() * 85, "{} descriptors for {} materials", r.descriptor_count(), r.block_count());
    }

    #[test]
    fn spec_text_round_trips_and_rejects_legacy_names() {
        let mut r = BlockRegistry::with_builtins();
        let c = cfg(&[[1, 2, 3, 4], [250, 251, 252, 253], [1, 2, 3, 4]]);
        let id = r.intern(&c).unwrap();
        let spec = r.spec(id);
        assert!(spec.starts_with("c:03"));
        let mut r2 = BlockRegistry::with_builtins();
        let id2 = r2.parse_spec(&spec).unwrap();
        assert_eq!(r2.configuration(id2), &c);
        assert_eq!(r2.parse_spec("air"), Some(AIR));
        assert_eq!(r2.parse_spec("natural:Stone,Iron"), None);
        assert_eq!(r2.parse_spec("c:zz"), None);
        assert_eq!(r2.parse_spec("c:0201020304"), None, "truncated");
        assert_eq!(r2.parse_spec("c:aéa"), None, "non-ASCII must not panic (wire/save input)");
        assert_eq!(r2.parse_spec("c:é"), None);
        assert_eq!(r2.parse_spec("c:"), None, "no bytes at all");
    }

    #[test]
    fn labels_are_annotations_only() {
        let mut r = BlockRegistry::with_builtins();
        let id = r.intern(&cfg(&[[120, 130, 140, 150]])).unwrap();
        r.set_label(id, "rock-like");
        assert_eq!(r.id_by_label("rock-like"), Some(id));
        assert_eq!(r.label(id), Some("rock-like"));
        assert!(r.spec(id).starts_with("c:"), "the spec never carries the label");
        assert!(
            !r.display_name(id).contains("rock"),
            "a label is a worldgen annotation, never shown to the player: {}",
            r.display_name(id)
        );
    }

    #[test]
    fn display_names_are_read_off_the_observation() {
        let r = BlockRegistry::with_builtins();
        assert_eq!(r.display_name(AIR), "air");
        let mut obs = Observation::AIR;
        obs.solid = true;
        obs.transparency = 0;
        obs.hardness = 230;
        obs.friction = 128;
        assert_eq!(describe(&obs), "hard solid");
        obs.emission = 9;
        obs.transparency = 210;
        obs.friction = 20;
        assert_eq!(describe(&obs), "glowing clear hard slick solid");
        let mut liq = Observation::AIR;
        liq.liquid = true;
        liq.transparency = 60;
        liq.flow = 240;
        assert_eq!(describe(&liq), "thin liquid");
        liq.transparency = 230;
        assert_eq!(describe(&liq), "clear thin liquid");
        // Two configurations with equal readings get equal words: the description is a reading.
        let mut r = BlockRegistry::with_builtins();
        let a = r.intern(&cfg(&[[120, 130, 140, 150]])).unwrap();
        let b = r.intern(&cfg(&[[121, 130, 140, 150]])).unwrap();
        if r.observation(a) == r.observation(b) {
            assert_eq!(r.display_name(a), r.display_name(b));
        }
    }

    #[test]
    fn apply_interaction_matches_interact_and_interns_once() {
        let mut r = BlockRegistry::with_builtins();
        let law = *r.law();
        let origin = cfg(&[[40, 80, 120, 160]]);
        let target = cfg(&[[80, 40, 160, 120]]);
        let once = interact(&law, &origin, &target, EventKind::Collision).target;
        assert_eq!(
            interact_repeat(&law, &origin, &target, EventKind::Collision, 1),
            once
        );
        let twice = interact(&law, &origin, &once, EventKind::Collision).target;
        assert_eq!(
            interact_repeat(&law, &origin, &target, EventKind::Collision, 2),
            twice
        );
        let before = r.block_count();
        let a = r
            .apply_interaction(&origin, &target, EventKind::Collision, 1)
            .unwrap();
        let after_first = r.block_count();
        let b = r
            .apply_interaction(&origin, &target, EventKind::Collision, 1)
            .unwrap();
        assert_eq!(a, b);
        assert_eq!(r.configuration(a), &once);
        assert_eq!(r.block_count(), after_first, "the same result interned once");
        assert!(after_first == before || after_first == before + 1);
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 32) as u32
        }
        fn config(&mut self) -> Configuration {
            let n = 1 + (self.next() as usize % 6);
            cfg(&(0..n)
                .map(|_| {
                    let x = self.next();
                    [
                        x as u8,
                        (x >> 8) as u8,
                        (x >> 16) as u8,
                        (x >> 24) as u8,
                    ]
                })
                .collect::<Vec<_>>())
        }
    }

    #[test]
    fn intern_is_a_bijection_on_encodings() {
        let mut r = BlockRegistry::with_builtins();
        let mut rng = Rng(0xC0FF_EE42);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..400 {
            let c = rng.config();
            let enc = c.encode();
            let id = r.intern(&c).unwrap();
            assert_eq!(r.encoding(id).as_bytes(), enc.as_bytes());
            assert_eq!(r.lookup(&c), Some(id));
            assert_eq!(r.configuration(id), &c);
            assert_eq!(
                Configuration::decode(enc.as_bytes()).unwrap(),
                c,
                "encode/decode is the intern key"
            );
            assert_eq!(r.intern(&c), Some(id), "second intern is identity");
            seen.insert(enc.as_bytes().to_vec());
        }
        assert_eq!(r.block_count(), seen.len() + 1, "air plus each distinct encoding");
    }

    #[test]
    fn hot_tables_agree_with_observe() {
        let mut r = BlockRegistry::with_builtins();
        let law = *r.law();
        let mut rng = Rng(11);
        let mut ids = Vec::new();
        for _ in 0..80 {
            ids.push(r.intern(&rng.config()).unwrap());
        }
        let hot = r.hot_tables();
        for id in ids {
            let obs = observe(&law, r.configuration(id));
            assert_eq!(r.emission(id), obs.emission);
            assert_eq!(hot.emission[id.0 as usize], obs.emission);
            assert_eq!(r.is_liquid(id), obs.liquid);
            assert_eq!(r.is_opaque(id), obs.solid && obs.transparency == 0);
            assert_eq!(hot.opaque(id), r.is_opaque(id));
            assert_eq!(r.hardness(id), obs.hardness);
            assert_eq!(r.friction(id), obs.friction);
            assert_eq!(r.transparency(id), obs.transparency);
            // Liquids mesh (FLAG_SOLID) but observe as non-solid.
            assert_eq!(r.is_solid(id), obs.solid || obs.liquid);
            assert_eq!(hot.solid(id), r.is_solid(id));
        }
    }

    #[test]
    fn parse_spec_rejects_hostile_wire_and_save_input() {
        let mut r = BlockRegistry::with_builtins();
        let c = cfg(&[[1, 2, 3, 4]]);
        let id = r.intern(&c).unwrap();
        let spec = r.spec(id);
        assert!(spec.starts_with("c:"));
        let hex = spec.trim_start_matches("c:");
        assert_eq!(r.parse_spec(&format!("C:{hex}")), None, "uppercase prefix");
        assert_eq!(r.parse_spec(&format!(" {spec}")), None, "leading space");
        assert_eq!(r.parse_spec(&format!("{spec} ")), None, "trailing space");
        assert_eq!(r.parse_spec("c:0"), None, "odd nibble");
        assert_eq!(r.parse_spec("c:gg"), None);
        assert_eq!(r.parse_spec("c:\0"), None);
        assert_eq!(r.parse_spec("c:00ff"), None, "trailing byte on void");
        assert_eq!(r.parse_spec("c:11"), None, "len 17 > CONFIG_MAX");
        assert_eq!(r.parse_spec("natural:Stone"), None);
        assert_eq!(r.parse_spec("air\0"), None);
        let upper = format!("c:{}", hex.to_ascii_uppercase());
        assert_eq!(r.parse_spec(&upper), Some(id), "uppercase hex is still a configuration");
        assert_eq!(r.parse_spec("c:00"), Some(AIR), "void encoding is air");
    }

    #[test]
    fn intern_returns_none_at_u16_cap() {
        let mut r = BlockRegistry::with_builtins();
        let mut n = 1u32;
        while r.block_count() < MAX_BLOCK_TYPES {
            let e = Element::new([
                n as u8,
                (n >> 8) as u8,
                (n >> 16) as u8,
                (n >> 24) as u8,
            ]);
            assert!(r.intern(&Configuration::single(e)).is_some(), "slot {n}");
            n += 1;
        }
        assert!(r.at_capacity());
        assert_eq!(r.block_count(), MAX_BLOCK_TYPES);
        let extra = cfg(&[[9, 8, 7, 6], [1, 2, 3, 4]]);
        assert!(r.intern(&extra).is_none(), "id space is exhausted");
        assert_eq!(r.parse_spec(&format!("c:{:02x}0908070601020304", 2)), None);
    }

    #[test]
    fn nearest_descriptor_fallback_at_layer_cap() {
        let mut r = BlockRegistry::with_builtins();
        // Occupy the 14-bit layer space with unique quantized visuals.
        let mut n = 0usize;
        for red in 0..16u8 {
            for green in 0..16u8 {
                for blue in 0..16u8 {
                    for alpha in 0..16u8 {
                        if n >= MAX_DESCRIPTORS {
                            break;
                        }
                        let vis = Visual {
                            rgb: [red << 4 | red, green << 4 | green, blue << 4 | blue],
                            rgb2: [0, 0, 0],
                            frequency: 0,
                            roughness: 0,
                            alpha: alpha << 4 | alpha,
                            glow: 0,
                        };
                        let _ = r.descriptor_for(vis);
                        n += 1;
                    }
                    if n >= MAX_DESCRIPTORS {
                        break;
                    }
                }
                if n >= MAX_DESCRIPTORS {
                    break;
                }
            }
            if n >= MAX_DESCRIPTORS {
                break;
            }
        }
        assert_eq!(r.descriptor_count(), MAX_DESCRIPTORS);
        let before = r.descriptor_count();
        let id = r.intern(&cfg(&[[7, 9, 11, 13], [200, 10, 30, 40]])).unwrap();
        assert_eq!(r.descriptor_count(), before, "material identity grows; render identity does not");
        assert!((r.render_layer(id) as usize) < MAX_DESCRIPTORS);
        let again = r.intern(&cfg(&[[7, 9, 11, 13], [200, 10, 30, 40]])).unwrap();
        assert_eq!(again, id);
        assert_eq!(r.render_layer(again), r.render_layer(id));
    }

    #[test]
    #[ignore]
    fn intern_observe_cost_per_new_configuration() {
        let mut r = BlockRegistry::with_builtins();
        let mut rng = Rng(99);
        let warmup = (0..32).map(|_| rng.config()).collect::<Vec<_>>();
        for c in &warmup {
            let _ = r.intern(c);
        }
        const N: u32 = 2_000;
        let configs: Vec<_> = (0..N).map(|_| rng.config()).collect();
        let t0 = std::time::Instant::now();
        for c in &configs {
            let _ = r.intern(c);
        }
        let us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(N);
        println!("intern+observe per new configuration: {us:.2} µs");
        assert!(us < 500.0, "intern+observe {us:.1} µs is past the sanity ceiling");
    }
}
