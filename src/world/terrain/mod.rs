//! InfiniteDiffusion: the world generator.
//!
//! The [`cosmos`] lists every body. A cube cell is that cube's face; an asteroid cell is the rock
//! that contains it. The twins are six faces of today's terrain — shape, caves, mines, veins,
//! trees, landmarks, structures — in face-local coordinates, one salt per face, on a cube grid bent
//! by the sag (the cells live in storage). The start world is
//! the same six faces laid on cube-sphere charts (home +Y keeps the v3 salts). Provinces theme
//! every column: a realm per face, regions and provinces on the shared surface point. The twins'
//! facing faces also carry spires and arches across the canyon, inside the relief bound. Below the
//! crust the bulk is a coarse mix whose mean amount is [`cosmos::BULK_DENSITY`], carved by the
//! interior ([`deep`]). Round bodies live on curved charts in storage ([`storage`]): storage
//! coordinates answer from their painters, and physical space holds none of their cells. Empty
//! space classifies as air and is never sampled.
//!
//! Every material is a configuration the [`palette`] found in the law; nothing here names an
//! element. The arithmetic is bit-identical on every peer (see [`noise`]).

pub mod cosmos;
pub mod noise;
pub mod palette;
mod cube;
mod deep;
mod features;
mod marks;
mod province;
pub mod round;
mod shape;
mod space;
mod span;
mod structures;
pub mod storage;
mod trees;
mod underground;

use std::sync::Arc;

use super::chunk::{CHUNK_SIZE, CHUNK_VOLUME, Chunk, ChunkData};
use super::generation::{Classify, ColumnHeights, TerrainGenerator};
use super::layout::{ColumnKey, Sky};
use crate::block::registry::{AIR, BlockId, BlockRegistry};
use crate::coord::{ChunkCoord, Face};
use crate::space::FaceFrame;

use noise::hash3;
use shape::{Column, Shape};
use trees::Trees;
use underground::{Grid, Underground};

/// Bumped whenever the same `(seed, coordinate)` can generate different materials than before.
/// Folded into the content fingerprint and recorded in saves. History: 1-5 the classic and
/// diffusion v1/v2 generators over authored and then emergent materials; 6 = InfiniteDiffusion v3
/// over the selective-transfer palette (2026-10-02); 7 = the cube planet (2026-10-03): bodies from
/// the cosmos, six faces, empty space. Home +Y keeps the v3 salts; provinces theme the field;
/// 8 = surface landmarks, and the ruins, monuments and watchers on them;
/// 9 = the start world on cube-sphere charts (unreleased, so the number stays);
/// 10 = the twins on a warped cube grid in storage.
pub const WORLDGEN_VERSION: u16 = 10;

/// The old v3 space floor. No longer a realm boundary; the fade and tests still name it.
pub const SPACE_FLOOR: i32 = 640;
/// Ground never rises above this (inside the far-LOD window `[0, 512)`; landmarks above it are not in the far field).
pub const MAX_GROUND: i32 = 470;
/// Ground never sinks below this (the far-LOD floor is 0).
pub const MIN_GROUND: i32 = 6;

/// Atlas index of a chart body id, if `body` is one.
fn chart_body(body: u16) -> Option<usize> {
    let base = super::section::CHART_BODY_BASE;
    (body >= base).then(|| (body - base) as usize)
}

/// The generator's knobs, in percent of the default (100 = as designed).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TerrainCfg {
    /// Mountain height.
    pub relief: u16,
    /// Cave density.
    pub caves: u16,
    /// Mine density.
    pub mines: u16,
    /// Planet and asteroid density.
    pub space: u16,
    /// Column variety.
    pub variety: u16,
    /// Surface feature density.
    pub features: u16,
    /// Structure density.
    pub structures: u16,
    /// Deep-strata density.
    pub deep: u16,
}

impl Default for TerrainCfg {
    fn default() -> Self {
        Self {
            relief: 100,
            caves: 100,
            mines: 100,
            space: 100,
            variety: 100,
            features: 100,
            structures: 100,
            deep: 100,
        }
    }
}

impl TerrainCfg {
    /// Every knob steps by this many percent.
    pub const STEP: u16 = 25;
    /// Relief range.
    pub const RELIEF: (u16, u16) = (25, 200);
    /// Range of the density knobs.
    pub const DENSITY: (u16, u16) = (0, 200);

    /// Snap every knob onto its stepper.
    pub fn clamp(mut self) -> Self {
        let snap = |v: u16, (lo, hi): (u16, u16)| ((v.clamp(lo, hi) + Self::STEP / 2) / Self::STEP * Self::STEP).clamp(lo, hi);
        self.relief = snap(self.relief, Self::RELIEF);
        self.caves = snap(self.caves, Self::DENSITY);
        self.mines = snap(self.mines, Self::DENSITY);
        self.space = snap(self.space, Self::DENSITY);
        self.variety = snap(self.variety, Self::DENSITY);
        self.features = snap(self.features, Self::DENSITY);
        self.structures = snap(self.structures, Self::DENSITY);
        self.deep = snap(self.deep, Self::DENSITY);
        self
    }

    /// Text form (`relief=100,...,deep=100`): saves, the wire, mod state.
    pub fn to_text(self) -> String {
        format!(
            "relief={},caves={},mines={},space={},variety={},features={},structures={},deep={}",
            self.relief, self.caves, self.mines, self.space, self.variety, self.features, self.structures, self.deep,
        )
    }

    /// Parse a full or partial knob string, starting from the defaults.
    pub fn from_text(data: &str) -> Self {
        Self::default().overlay(data)
    }

    /// Overlay keys from `data` onto `self`, then clamp. Unknown keys are ignored.
    pub fn overlay(mut self, data: &str) -> Self {
        for part in data.split(',') {
            let Some((k, v)) = part.split_once('=') else { continue };
            let slot = match k.trim() {
                "relief" => &mut self.relief,
                "caves" => &mut self.caves,
                "mines" => &mut self.mines,
                "space" => &mut self.space,
                "variety" => &mut self.variety,
                "features" => &mut self.features,
                "structures" => &mut self.structures,
                "deep" => &mut self.deep,
                _ => continue,
            };
            *slot = v.trim().parse().unwrap_or(*slot);
        }
        self.clamp()
    }

    /// The eight knobs in wire order.
    pub fn to_wire(self) -> [u16; 8] {
        [self.relief, self.caves, self.mines, self.space, self.variety, self.features, self.structures, self.deep]
    }

    /// Inverse of [`to_wire`](Self::to_wire).
    pub fn from_wire(v: [u16; 8]) -> Self {
        Self {
            relief: v[0],
            caves: v[1],
            mines: v[2],
            space: v[3],
            variety: v[4],
            features: v[5],
            structures: v[6],
            deep: v[7],
        }
        .clamp()
    }
}

/// The palette interned into one world's table, as ids the generator can paint with.
#[derive(Clone, Debug)]
pub struct Materials {
    pub grass: BlockId,
    pub meadow: BlockId,
    pub soil: BlockId,
    pub sand: BlockId,
    pub redsand: BlockId,
    pub snow: BlockId,
    pub ice: BlockId,
    pub gravel: BlockId,
    pub rock: [BlockId; 4],
    pub sandstone: [BlockId; 4],
    pub deeprock: BlockId,
    pub abyss: BlockId,
    pub timber: BlockId,
    pub leaves: BlockId,
    pub pine: BlockId,
    pub blossom: BlockId,
    pub autumn: BlockId,
    pub plank: BlockId,
    pub rail: BlockId,
    pub lamp: BlockId,
    pub rubble: BlockId,
    pub bone: BlockId,
    pub glowcap: BlockId,
    pub crystal: BlockId,
    pub copper: BlockId,
    pub azurite: BlockId,
    pub gold: BlockId,
    pub regolith: BlockId,
    pub basalt: BlockId,
    pub frost: BlockId,
    pub moss: BlockId,
    pub ochre: BlockId,
    pub violet: BlockId,
    pub magma: BlockId,
    pub core: BlockId,
    pub star: BlockId,
    pub flower_red: BlockId,
    pub flower_yellow: BlockId,
    pub flower_blue: BlockId,
    pub flower_white: BlockId,
    pub cap_red: BlockId,
    pub cap_brown: BlockId,
    pub stem: BlockId,
    pub ash: BlockId,
    pub obsidian: BlockId,
    pub salt: BlockId,
    pub clay: BlockId,
    pub limestone: BlockId,
    pub marble: BlockId,
    pub jade: BlockId,
    pub rust: BlockId,
    pub mud: BlockId,
    pub lichen: BlockId,
    pub darkwood: BlockId,
    pub bark: BlockId,
    pub amber: BlockId,
    pub slate: BlockId,
    pub cinder: BlockId,
    pub petrified: BlockId,
    pub tundra: BlockId,
    pub glowshroom: BlockId,
    /// Reagents with the strata they may sit in (dormant hosts).
    pub reagents: Vec<Reagent>,
}

/// A reagent and the strata its veins may sit in.
#[derive(Clone, Debug)]
pub struct Reagent {
    pub id: BlockId,
    /// The material this reagent empties (read by tests and the `/reagents` console listing).
    #[allow(dead_code)]
    pub target: BlockId,
    /// Strata (rock and sandstone bands, deep rock) this reagent is dormant against.
    pub hosts: Vec<BlockId>,
}

impl Materials {
    /// Intern the law's palette into `reg` (in role order, labelled) and resolve the ids.
    pub fn intern(reg: &mut BlockRegistry) -> Self {
        let entries = palette::of(reg.law());
        let mut ids = Vec::with_capacity(entries.len());
        for e in &entries {
            let id = reg.intern(&e.config).expect("a fresh table holds the palette");
            reg.set_label(id, e.label);
            ids.push(id);
        }
        let id = |label: &str| ids[entries.iter().position(|e| e.label == label).expect("palette role")];
        let rock = [id("rock"), id("rock1"), id("rock2"), id("rock3")];
        let sandstone = [id("sandstone"), id("sandstone1"), id("sandstone2"), id("sandstone3")];
        let deeprock = id("deeprock");
        let strata: Vec<BlockId> = rock.iter().chain(&sandstone).copied().chain([deeprock, id("abyss")]).collect();
        let reagents = palette::ROLES
            .iter()
            .enumerate()
            .filter_map(|(i, r)| match r.need {
                palette::Need::Reagent(target) => Some((ids[i], id(target))),
                _ => None,
            })
            .map(|(rid, target)| Reagent {
                id: rid,
                target,
                hosts: strata.iter().copied().filter(|&s| s != target && reg.quiescent(s, rid) && reg.quiescent(rid, s)).collect(),
            })
            .collect();
        Self {
            grass: id("grass"),
            meadow: id("meadow"),
            soil: id("soil"),
            sand: id("sand"),
            redsand: id("redsand"),
            snow: id("snow"),
            ice: id("ice"),
            gravel: id("gravel"),
            rock,
            sandstone,
            deeprock,
            abyss: id("abyss"),
            timber: id("timber"),
            leaves: id("leaves"),
            pine: id("pine"),
            blossom: id("blossom"),
            autumn: id("autumn"),
            plank: id("plank"),
            rail: id("rail"),
            lamp: id("lamp"),
            rubble: id("rubble"),
            bone: id("bone"),
            glowcap: id("glowcap"),
            crystal: id("crystal"),
            copper: id("copper"),
            azurite: id("azurite"),
            gold: id("gold"),
            regolith: id("regolith"),
            basalt: id("basalt"),
            frost: id("frost"),
            moss: id("moss"),
            ochre: id("ochre"),
            violet: id("violet"),
            magma: id("magma"),
            core: id("core"),
            star: id("star"),
            flower_red: id("flower_red"),
            flower_yellow: id("flower_yellow"),
            flower_blue: id("flower_blue"),
            flower_white: id("flower_white"),
            cap_red: id("cap_red"),
            cap_brown: id("cap_brown"),
            stem: id("stem"),
            ash: id("ash"),
            obsidian: id("obsidian"),
            salt: id("salt"),
            clay: id("clay"),
            limestone: id("limestone"),
            marble: id("marble"),
            jade: id("jade"),
            rust: id("rust"),
            mud: id("mud"),
            lichen: id("lichen"),
            darkwood: id("darkwood"),
            bark: id("bark"),
            amber: id("amber"),
            slate: id("slate"),
            cinder: id("cinder"),
            petrified: id("petrified"),
            tundra: id("tundra"),
            glowshroom: id("glowshroom"),
            reagents,
        }
    }
}


/// The bulk mix as one material: fractions weighted by the prototype response, density forced to
/// the amount the gravity primitive uses.
fn bulk_matter(registry: &BlockRegistry, bulk: &cube::Bulk) -> crate::mechanics::material::Params {
    use crate::mechanics::material;
    let parts: Vec<(f64, material::Params)> = bulk
        .fractions()
        .iter()
        .filter(|(_, f)| *f > 0.0)
        .map(|&(id, f)| {
            let cohesion = registry.observation(id).cohesion as i32;
            (f, material::mechanical_response(registry.amount(id) as u32, material::cohesion01(cohesion)))
        })
        .collect();
    let mut matter = material::Params::mix(&parts);
    matter.density = cosmos::BULK_DENSITY;
    matter
}

/// Sagging cubes: a `Layout::Cube` body whose warp moves any point by more than half a block.
fn cube_warps(
    cosmos: &cosmos::Cosmos,
    matter: &crate::mechanics::material::Params,
) -> Vec<(u16, Arc<crate::space::warp::Warp>)> {
    use crate::mechanics::genesis;
    use crate::space::warp::{Warp, STORAGE_MOVE};
    let mut out = Vec::new();
    for b in cosmos.bodies() {
        let cosmos::Shape::Cube { half } = b.shape else { continue };
        let half_f = half as f64;
        let pi = genesis::pi_g(matter, half_f);
        let Some((layout, _)) = genesis::tabulated_layout(pi, half_f) else { continue };
        if !matches!(layout, genesis::Layout::Cube) {
            continue;
        }
        let Some(solved) = genesis::tabulated(pi, half_f) else { continue };
        let warp = Warp::from_solved(&solved, b.centre_f());
        if warp.max_displacement() <= STORAGE_MOVE {
            continue;
        }
        out.push((b.id, Arc::new(warp)));
    }
    out
}

/// Blended columns of one start-world chart chunk. Face altitude of local `y0 + la` is
/// `h_base + la` (the atlas embedding lifts band 0 onto the fitted datum, so storage altitude is
/// altitude above the datum).
#[derive(Clone)]
struct ChartCols {
    cols: Vec<Column>,
    u0: i32,
    v0: i32,
    h_base: i32,
    max_h: i32,
    min_h: i32,
    max_terrain: i32,
    face: Face,
    y0: i64,
}

/// One cube face's copy of today's terrain, with its own salt.
struct FacePaint {
    shape: Shape,
    under: Underground,
    trees: Trees,
    features: features::Features,
    structures: structures::Structures,
}

/// The generator.
pub struct Terrain {
    seed: i64,
    /// Every body in the universe; also the generator's mass oracle.
    cosmos: Arc<cosmos::Cosmos>,
    /// Indexed by `body.id * 6 + face`. `None` for bodies that are not cubes and not the start world.
    paints: Vec<Option<FacePaint>>,
    bulk: cube::Bulk,
    /// Caverns, chambers, mantle bubbles and the Heart. Quiet deep chunks never consult it per voxel.
    deep: deep::Deep,
    /// The round bodies, painted on curved charts in storage.
    storage: storage::StorageWorlds,
    m: Arc<Materials>,
}

/// Whether world cell x `x` lies in the storage region (curved charts), not physical space.
#[inline]
fn stored(x: i32) -> bool {
    x as i64 >= crate::space::atlas::STORAGE_X0
}

/// Centres of the other big bodies, the ones an observatory can point at. Moons are not big.
fn sky_aims(cosmos: &cosmos::Cosmos, id: u16) -> ([[i64; 3]; 8], usize) {
    let mut aims = [[0i64; 3]; 8];
    let mut n = 0;
    for o in cosmos.bodies() {
        if o.kind == cosmos::Kind::Moon || o.id == id || n == aims.len() {
            continue;
        }
        aims[n] = o.centre;
        n += 1;
    }
    (aims, n)
}

fn rel_box(centre: [i64; 3], lo: [i64; 3], hi: [i64; 3]) -> ([i64; 3], [i64; 3]) {
    (
        [lo[0] - centre[0], lo[1] - centre[1], lo[2] - centre[2]],
        [hi[0] - centre[0], hi[1] - centre[1], hi[2] - centre[2]],
    )
}

/// Shared handle workers clone.
pub type Generator = Arc<dyn TerrainGenerator>;

/// Build the generator for `seed`, interning its materials into `registry`.
pub fn generator(registry: &mut BlockRegistry, seed: i64, cfg: TerrainCfg) -> Generator {
    Arc::new(Terrain::with_cfg(registry, seed, cfg))
}

impl Terrain {
    /// The generator with default knobs.
    #[cfg(test)]
    pub fn new(registry: &mut BlockRegistry, seed: i64) -> Self {
        Self::with_cfg(registry, seed, TerrainCfg::default())
    }

    /// The generator with explicit knobs.
    pub fn with_cfg(registry: &mut BlockRegistry, seed: i64, cfg: TerrainCfg) -> Self {
        let cfg = cfg.clamp();
        let s = (seed as u64 ^ (seed as u64 >> 32)) as u32 ^ 0x1D1F_F051;
        let m = Arc::new(Materials::intern(registry));
        let scale = cfg.deep as f32 / 100.0;
        let mut cosmos = cosmos::Cosmos::with_deep(s, cfg.space as f32 / 100.0, scale);
        let bulk = cube::choose_bulk(registry, &m, s ^ 0xB01C_D3E5);
        // Sagging cubes keep their grid and record how far the warp moves it, so gravity can bound
        // the missing shell. The start world stays the chart sphere: fitting that chart to the
        // relaxed datum would move spawn by the pole relief.
        let matter = bulk_matter(registry, &bulk);
        let warps = cube_warps(&cosmos, &matter);
        for (id, warp) in &warps {
            cosmos.set_warp_sag(*id, warp.max_displacement());
        }
        let cosmos = Arc::new(cosmos);
        let relief = cfg.relief as f32 / 100.0;
        let variety = cfg.variety as f32 / 100.0;
        // The twin with the smaller seed is lush; the other is crystalline. One twin keeps lush.
        let lush = cosmos
            .bodies()
            .iter()
            .filter(|b| b.kind == cosmos::Kind::Twin)
            .min_by_key(|b| (b.seed, b.id))
            .map(|b| b.id);
        let n = cosmos.bodies().iter().map(|b| b.id as usize).max().unwrap_or(0) + 1;
        let mut paints = Vec::new();
        paints.resize_with(n * 6, || None);
        for b in cosmos.bodies() {
            let half = match b.shape {
                cosmos::Shape::Cube { half } => half,
                cosmos::Shape::Ball { .. } if b.kind == cosmos::Kind::Home => {
                    crate::space::atlas::surface_n(cosmos::HOME_RADIUS) / 2
                }
                _ => continue,
            };
            let twin = b.kind == cosmos::Kind::Twin;
            let (aims, n_aims) = sky_aims(&cosmos, b.id);
            let facing_face = span::facing_face(&cosmos, b);
            for face in Face::ALL {
                // Home +Y keeps the v3 salts. Every other face is a fresh field.
                let s_face = if b.kind == cosmos::Kind::Home && face == Face::PosY {
                    s
                } else {
                    hash3(s ^ 0x5A17, i32::from(b.id), face.index() as i32, 0x6A1E)
                };
                let realm = if twin {
                    if Some(b.id) == lush { province::Realm::Lush } else { province::Realm::Crystal }
                } else {
                    province::Realm::of_home(face)
                };
                let garden = b.kind == cosmos::Kind::Home && face == Face::PosY;
                // The facing canyon already carries spires. Structures stay on the other faces.
                let struct_scale = if facing_face == Some(face) { 0.0 } else { cfg.structures as f32 / 100.0 };
                let i = b.id as usize * 6 + face.index();
                paints[i] = Some(FacePaint {
                    shape: Shape::new(s_face, relief, m.clone(), face, half, realm, b.seed, variety, garden),
                    under: Underground::new(s_face ^ 0x0BAD_CAFE, cfg, m.clone()),
                    trees: Trees::new(s_face ^ 0x7EE5_0000, m.clone()),
                    features: features::Features::new(
                        s_face ^ 0x5A1E_5A1E,
                        cfg.features as f32 / 100.0,
                        m.clone(),
                    ),
                    structures: structures::Structures::new(
                        s_face ^ 0x57E0_C700,
                        struct_scale,
                        m.clone(),
                        face,
                        half,
                        b.centre,
                        &aims[..n_aims],
                    ),
                });
            }
        }
        // Majority of the 24-bit mix: the uniform block under the start world's band 0.
        let home_fill = bulk.majority();
        let storage = storage::StorageWorlds::new(&cosmos, &m, home_fill, &warps);
        Self {
            seed,
            storage,
            cosmos,
            paints,
            bulk,
            deep: deep::Deep::new(scale, m.clone()),
            m,
        }
    }

    /// The interned palette.
    #[cfg(test)]
    pub fn materials(&self) -> &Materials {
        &self.m
    }

    fn paint(&self, body: &cosmos::Body, face: Face) -> &FacePaint {
        self.paints[body.id as usize * 6 + face.index()].as_ref().expect("cube face")
    }

    /// The body that owns cell `p`: closest datum, then the smaller id. A stored cube is skipped
    /// unless `include` (the reference painter, reached by translating a storage cell back).
    fn owner_in(&self, p: [i64; 3], include: bool) -> Option<cosmos::Body> {
        let q = glam::DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64);
        let mut best: Option<(f64, cosmos::Body)> = None;
        for b in self.cosmos.bodies() {
            // Round bodies are charted: their cells live in storage. A sagging cube's cells do too.
            if !matches!(b.shape, cosmos::Shape::Cube { .. }) || !b.touches(p, p) {
                continue;
            }
            if !include && self.storage.holds_cube(b.id) {
                continue;
            }
            let alt = b.altitude(q).abs();
            let take = match best {
                None => true,
                Some((a, prev)) => alt < a || (alt == a && b.id < prev.id),
            };
            if take {
                best = Some((alt, *b));
            }
        }
        best.map(|(_, b)| b)
    }

    /// Reference chunk of a storage chunk that sits in a warped-cube box.
    #[inline]
    fn cube_chunk(&self, coord: ChunkCoord) -> Option<ChunkCoord> {
        if (coord.x as i64) * (CHUNK_SIZE as i64) < self.storage.cube_x0() {
            return None;
        }
        self.storage.cube_ref_chunk(coord)
    }

    /// Home +Y column at world `(x, z)`, with the rim blend applied to its height.
    fn posy_hit(&self, wx: i32, wz: i32) -> Option<Posy<'_>> {
        let mut best: Option<(i32, Column, usize, i32, i32, i64, [i64; 3])> = None;
        let n = self.cosmos.bodies().len();
        for i in 0..n {
            let body = self.cosmos.bodies()[i];
            let cosmos::Shape::Cube { half } = body.shape else { continue };
            let Some(centre) = cube::centre_i32(body.centre) else { continue };
            let (ub, vb) = cube::tangents(Face::PosY, centre, wx, wz);
            if cube::edge_inside(half, ub, vb) < -cosmos::RELIEF {
                continue;
            }
            let (Ok(u), Ok(v)) = (i32::try_from(ub), i32::try_from(vb)) else { continue };
            let paint = self.paint(&body, Face::PosY);
            let mut col = paint.shape.column(u, v);
            col.height = cube::blend_height(col.height, cube::rim_seed(&body), Face::PosY, half, ub, vb);
            let a_body = half + i64::from(col.height) - 1;
            let Ok(a_i) = i32::try_from(a_body) else { continue };
            if cube::face_of(cube::local_to_rel(Face::PosY, u, a_i, v)) != Face::PosY {
                continue;
            }
            let Some(world_a) = cube::world_a(half, col.height, cube::normal_dot(body.centre, Face::PosY)) else {
                continue;
            };
            if best.as_ref().is_none_or(|hit| world_a > hit.0) {
                best = Some((world_a, col, i, u, v, half, body.centre));
            }
        }
        let (world_a, col, i, u, v, half, centre) = best?;
        let body = self.cosmos.bodies()[i];
        Some(Posy { world_a, col, paint: self.paint(&body, Face::PosY), u, v, half, centre })
    }

    /// One named cube's column on `face`. PosY of every body stays on [`posy_hit`](Self::posy_hit)
    /// (the highest surface), so the home +Y bytes do not move.
    fn named_face_hit(&self, body_id: u16, face: Face, u: i32, v: i32) -> Option<Posy<'_>> {
        let body = self.cosmos.bodies().iter().find(|b| b.id == body_id)?;
        let cosmos::Shape::Cube { half } = body.shape else { return None };
        let centre_i = cube::centre_i32(body.centre)?;
        let (ub, vb) = cube::tangents(face, centre_i, u, v);
        if cube::edge_inside(half, ub, vb) < -cosmos::RELIEF {
            return None;
        }
        let (Ok(ui), Ok(vi)) = (i32::try_from(ub), i32::try_from(vb)) else { return None };
        let paint = self.paint(body, face);
        let mut col = paint.shape.column(ui, vi);
        col.height = cube::blend_height(col.height, cube::rim_seed(body), face, half, ub, vb);
        let a_body = half + i64::from(col.height) - 1;
        let Ok(a_i) = i32::try_from(a_body) else { return None };
        if cube::face_of(cube::local_to_rel(face, ui, a_i, vi)) != face {
            return None;
        }
        let world_a = cube::world_a(half, col.height, cube::normal_dot(body.centre, face))?;
        Some(Posy { world_a, col, paint, u: ui, v: vi, half, centre: body.centre })
    }

    /// Bulk, or the interior feature at `rel`. `depth` is the true depth when the caller already
    /// has the column; otherwise it is derived, and inside one band the block does not depend on it.
    fn deep_at(&self, body: &cosmos::Body, rel: [i64; 3], depth: Option<i32>) -> BlockId {
        let bulk = cube::bulk_id(&self.bulk, body, rel);
        // A known depth means the chunk already intersects a feature; `might` would only repeat that test.
        if depth.is_none() && !self.deep.might(body, rel) {
            return bulk;
        }
        let depth = depth.unwrap_or_else(|| self.depth_of(body, rel));
        self.deep.block(body, rel, depth, bulk)
    }

    /// A depth in the same band as the true one. The column is sampled only when the surface-height
    /// range straddles a band boundary.
    fn depth_of(&self, body: &cosmos::Body, rel: [i64; 3]) -> i32 {
        let half = cube::half_of(body);
        let pd = deep::plane_depth(half, rel);
        let lo = pd + i64::from(MIN_GROUND);
        let hi = pd + i64::from(MAX_GROUND);
        let inside = |a: i64, b: i64| lo > a && hi <= b;
        if inside(i64::from(cube::CRUST), i64::from(deep::DEEP_HI))
            || inside(i64::from(deep::DEEP_HI), i64::from(deep::UNDER_HI))
            || lo > i64::from(deep::UNDER_HI)
        {
            return lo as i32;
        }
        self.surface_depth(body, rel)
    }

    /// `column height − face altitude`, the same subtraction [`cube_cell`](Self::cube_cell) uses.
    fn surface_depth(&self, body: &cosmos::Body, rel: [i64; 3]) -> i32 {
        let half = cube::half_of(body);
        let fallback = (deep::plane_depth(half, rel) + i64::from(MIN_GROUND)) as i32;
        let (Ok(x), Ok(y), Ok(z)) = (i32::try_from(rel[0]), i32::try_from(rel[1]), i32::try_from(rel[2])) else {
            return fallback;
        };
        let face = cube::face_of(rel);
        let (u, a, v) = FaceFrame::new(face).cell_to_local((x, y, z));
        let h = a - half as i32;
        let paint = self.paint(body, face);
        let mut col = paint.shape.column(u, v);
        col.height = cube::blend_height(col.height, cube::rim_seed(body), face, half, i64::from(u), i64::from(v));
        col.height - h
    }

    fn cube_cell(&self, body: &cosmos::Body, p: [i64; 3]) -> BlockId {
        let rel = [p[0] - body.centre[0], p[1] - body.centre[1], p[2] - body.centre[2]];
        let half = cube::half_of(body);
        if cube::in_deep(rel, half) {
            return self.deep_at(body, rel, None);
        }
        let (Ok(x), Ok(y), Ok(z)) = (i32::try_from(rel[0]), i32::try_from(rel[1]), i32::try_from(rel[2])) else {
            return AIR;
        };
        let face = cube::face_of(rel);
        let paint = self.paint(body, face);
        let (u, a, v) = FaceFrame::new(face).cell_to_local((x, y, z));
        let h = a - half as i32;
        let mut col = paint.shape.column(u, v);
        col.height = cube::blend_height(col.height, cube::rim_seed(body), face, half, i64::from(u), i64::from(v));
        self.face_voxel(body, face, u, h, v, &col, rel)
    }

    /// One face cell once `(u, h, v)` and the blended column are known. `rel` is the virtual-cube cell.
    fn face_voxel(
        &self,
        body: &cosmos::Body,
        face: Face,
        u: i32,
        h: i32,
        v: i32,
        col: &Column,
        rel: [i64; 3],
    ) -> BlockId {
        let paint = self.paint(body, face);
        let half = cube::half_of(body);
        if h >= col.height {
            // Structures own their footprint, then landmarks. A dug air cell stays empty.
            if h - col.height <= features::MAX_ABOVE {
                if let Some(st) = paint.structures.block_at(&paint.shape, &paint.under, u, h, v, col.height) {
                    if st.id != AIR {
                        return st.id;
                    }
                    if st.dig {
                        return AIR;
                    }
                }
                if paint.structures.owns(&paint.shape, &paint.under, u, v) {
                    return AIR;
                }
                if let Some(st) = paint.features.block_at(&paint.shape, u, h, v, col.height) {
                    if st.id != AIR {
                        return st.id;
                    }
                    if st.dig {
                        return AIR;
                    }
                }
            }
            if h - col.height <= trees::MAX_TREE_HEIGHT {
                if paint.features.cuts_surface(&paint.shape, u, col.height, v) {
                    return AIR;
                }
                if let Some(id) = paint.trees.block_at(&paint.shape, u, h, v) {
                    return id;
                }
                if h == col.height {
                    if let Some(id) = paint.shape.flower_at(&col, u, v) {
                        return id;
                    }
                }
            }
            // Home and the outward faces never ask. The facing face's spires sit above the ground.
            if body.kind == cosmos::Kind::Twin && span::facing_face(&self.cosmos, body) == Some(face) {
                if let Some(id) = span::block(&self.m, span::lush(&self.cosmos, body), span::face_seed(body), half, u, h, v) {
                    return id;
                }
            }
            return AIR;
        }
        if col.height - h > cube::CRUST {
            return self.deep_at(body, rel, Some(col.height - h));
        }
        // A dig replaces the crust cell, so a crater core is not opened back into a cave.
        // Structure shafts reach a mine level; landmark digs stay in their shallower band.
        let depth = col.height - h;
        if depth <= structures::DIG_LIMIT {
            if let Some(st) = paint.structures.block_at(&paint.shape, &paint.under, u, h, v, col.height) {
                if st.dig {
                    return st.id;
                }
            }
        }
        if depth <= features::MAX_BELOW && !paint.structures.owns(&paint.shape, &paint.under, u, v) {
            if let Some(st) = paint.features.block_at(&paint.shape, u, h, v, col.height) {
                if st.dig {
                    return st.id;
                }
            }
        }
        let field = |yy: i32| Grid::interp_corners(&paint.under.corners(u, yy, v), u, yy, v);
        let ground = paint.shape.ground(&col, u, h, v);
        paint.under.finish(&col, u, h, v, ground, field(h), || field(h - 1), || field(h + 1))
    }

    fn cell(&self, x: i32, y: i32, z: i32) -> BlockId {
        self.cell_in(x, y, z, false)
    }

    /// One cell. `include` paints a stored cube at its reference coordinates; without it that
    /// cube's physical reach is air, the same as a charted round body.
    fn cell_in(&self, x: i32, y: i32, z: i32, include: bool) -> BlockId {
        if let Some((id, reference)) = self.storage.cube_ref_cell(x, y, z) {
            let Some(body) = self.cosmos.bodies().iter().find(|b| b.id == id) else { return AIR };
            return self.cube_cell(body, reference);
        }
        if stored(x) {
            if let Some(id) = self.home_block(x, y, z) {
                return id;
            }
            return self.storage.voxel(x, y, z);
        }
        let p = [i64::from(x), i64::from(y), i64::from(z)];
        if let Some(body) = self.owner_in(p, include) {
            return self.cube_cell(&body, p);
        }
        // A round body's reach, and a stored cube's old physical cells, are empty here.
        if self.cosmos.bodies().iter().any(|b| {
            let charted = !matches!(b.shape, cosmos::Shape::Cube { .. }) || (!include && self.storage.holds_cube(b.id));
            charted && b.touches(p, p)
        }) {
            return AIR;
        }
        space::block(&self.cosmos, &self.m, p)
    }

    /// Chunk wholly inside one cube's deep limit: the mix, and nothing else.
    fn fast_bulk_in(&self, coord: ChunkCoord, include: bool) -> Option<ChunkData> {
        let (lo, hi) = cube::chunk_bounds(coord);
        let mut only: Option<cosmos::Body> = None;
        for b in self.cosmos.bodies_touching(lo, hi) {
            if !include && self.storage.holds_cube(b.id) {
                continue;
            }
            if only.is_some() {
                return None;
            }
            only = Some(*b);
        }
        let body = only?;
        let cosmos::Shape::Cube { .. } = body.shape else { return None };
        let half = cube::half_of(&body);
        let rels = cube::corners(lo, hi).map(|p| [p[0] - body.centre[0], p[1] - body.centre[1], p[2] - body.centre[2]]);
        if !rels.iter().copied().all(|r| cube::in_deep(r, half)) {
            return None;
        }
        let (rlo, rhi) = rel_box(body.centre, lo, hi);
        if self.deep.all_air(&body, rlo, rhi) {
            return Some(ChunkData::Uniform(AIR));
        }
        if self.deep.hits(&body, rlo, rhi) {
            return None;
        }
        let id = cube::bulk_uniform(&self.bulk, &body, &rels)?;
        Some(ChunkData::Uniform(id))
    }

    fn fill_slow_in(&self, cx: i32, cy: i32, cz: i32, include: bool) -> ChunkData {
        let coord = ChunkCoord::new(cx, cy, cz);
        match self.classify_in(coord, include) {
            Classify::Air => return ChunkData::Uniform(AIR),
            Classify::Uniform(id) => return ChunkData::Uniform(id),
            Classify::Mixed => {}
        }
        if let Some(data) = self.fast_bulk_in(coord, include) {
            return data;
        }
        let n = CHUNK_SIZE as i32;
        let (x0, y0, z0) = (cx * n, cy * n, cz * n);
        // Only rocks here: paint them from one list instead of looking them up per cell.
        let (lo, hi) = cube::chunk_bounds(coord);
        let touches = self.cosmos.bodies_touching(lo, hi).any(|b| include || !self.storage.holds_cube(b.id));
        if !touches {
            return space::fill(&self.cosmos, &self.m, lo);
        }
        let mut cells = Box::new([AIR; CHUNK_VOLUME]);
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                for ly in 0..CHUNK_SIZE {
                    cells[Chunk::index(lx, ly, lz)] = self.cell_in(x0 + lx as i32, y0 + ly as i32, z0 + lz as i32, include);
                }
            }
        }
        ChunkData::from_cells(cells)
    }

    /// The cube whose `face` covers this footprint outside the sky's edge band, outermost along
    /// the normal.
    fn face_column_body_in(&self, key: ColumnKey, include: bool) -> Option<cosmos::Body> {
        let samples = [(0i32, 0i32), (0, 15), (15, 0), (15, 15), (8, 8)];
        let mut best: Option<(cosmos::Body, i32)> = None;
        let n = self.cosmos.bodies().len();
        for i in 0..n {
            let body = self.cosmos.bodies()[i];
            if !include && self.storage.holds_cube(body.id) {
                continue;
            }
            let cosmos::Shape::Cube { half } = body.shape else { continue };
            let Some(centre) = cube::centre_i32(body.centre) else { continue };
            let mut min_in = i64::MAX;
            for (lu, lv) in samples {
                let (u, v) = key.column_cell_uv(lu, lv);
                let (ub, vb) = cube::tangents(key.face, centre, u, v);
                min_in = min_in.min(cube::edge_inside(half, ub, vb));
            }
            if min_in < cube::SKY_EDGE {
                continue;
            }
            let (u, v) = key.column_cell_uv(8, 8);
            let (ub, vb) = cube::tangents(key.face, centre, u, v);
            let (Ok(ui), Ok(vi)) = (i32::try_from(ub), i32::try_from(vb)) else { continue };
            let paint = self.paint(&body, key.face);
            let terrain = paint.shape.height(ui, vi);
            let h = cube::blend_height(terrain, cube::rim_seed(&body), key.face, half, ub, vb);
            let a_body = half + i64::from(h) - 1;
            let Ok(a_i) = i32::try_from(a_body) else { continue };
            if cube::face_of(cube::local_to_rel(key.face, ui, a_i, vi)) != key.face {
                continue;
            }
            let Some(world_a) = cube::world_a(half, h, cube::normal_dot(body.centre, key.face)) else { continue };
            if best.as_ref().is_none_or(|(_, a)| world_a > *a) {
                best = Some((body, world_a));
            }
        }
        best.map(|(b, _)| b)
    }

    fn outward_blocked(&self, body: &cosmos::Body, face: Face, coord: ChunkCoord) -> bool {
        let s = 16i64;
        let origin = [coord.x as i64 * s, coord.y as i64 * s, coord.z as i64 * s];
        let mut lo = origin;
        let mut hi = [origin[0] + 15, origin[1] + 15, origin[2] + 15];
        let (nx, ny, nz) = face.normal();
        let n = [i64::from(nx), i64::from(ny), i64::from(nz)];
        let far = 1_200_000_000i64;
        for a in 0..3 {
            if n[a] > 0 {
                hi[a] = far;
            } else if n[a] < 0 {
                lo[a] = -far;
            }
        }
        self.cosmos.bodies_touching(lo, hi).any(|b| b.id != body.id)
    }

    /// One deep chunk: eight coarse-cell samples, then the cheap mix if they disagree.
    /// `IDENTITY` writes storage `(lu, la, lv)` (a chart). A cube face permutes through its frame.
    fn bulk_chunk<const IDENTITY: bool>(
        &self,
        body: &cosmos::Body,
        face: Face,
        u0: i32,
        h0: i32,
        v0: i32,
    ) -> ChunkData {
        let half = cube::half_of(body) as i32;
        let at = |lu: i32, la: i32, lv: i32| cube::local_to_rel(face, u0 + lu, half + h0 + la, v0 + lv);
        let rels = [
            at(0, 0, 0),
            at(15, 0, 0),
            at(0, 0, 15),
            at(15, 0, 15),
            at(0, 15, 0),
            at(15, 15, 0),
            at(0, 15, 15),
            at(15, 15, 15),
        ];
        if let Some(id) = cube::bulk_uniform(&self.bulk, body, &rels) {
            return ChunkData::Uniform(id);
        }
        let frame = FaceFrame::new(face);
        let mut cells = Box::new([AIR; CHUNK_VOLUME]);
        for la in 0..CHUNK_SIZE {
            for lv in 0..CHUNK_SIZE {
                for lu in 0..CHUNK_SIZE {
                    let rel = at(lu as i32, la as i32, lv as i32);
                    let id = cube::bulk_id(&self.bulk, body, rel);
                    let (lx, ly, lz) = if IDENTITY { (lu, la, lv) } else { frame.index_to_world(lu, la, lv) };
                    cells[Chunk::index(lx, ly, lz)] = id;
                }
            }
        }
        ChunkData::from_cells(cells)
    }

    /// A chunk wholly below the crust. Quiet chunks stay on the mix; a feature is filled per column.
    fn fill_deep<const IDENTITY: bool>(
        &self,
        body: &cosmos::Body,
        face: Face,
        cols: &[Column],
        u0: i32,
        h0: i32,
        v0: i32,
    ) -> ChunkData {
        let half = cube::half_of(body) as i32;
        let at = |lu: i32, la: i32, lv: i32| cube::local_to_rel(face, u0 + lu, half + h0 + la, v0 + lv);
        let mut lo = at(0, 0, 0);
        let mut hi = lo;
        for (lu, la, lv) in [(15, 0, 0), (0, 15, 0), (0, 0, 15), (15, 15, 0), (15, 0, 15), (0, 15, 15), (15, 15, 15)] {
            let p = at(lu, la, lv);
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        if self.deep.all_air(body, lo, hi) {
            return ChunkData::Uniform(AIR);
        }
        if !self.deep.hits(body, lo, hi) {
            return self.bulk_chunk::<IDENTITY>(body, face, u0, h0, v0);
        }
        let frame = FaceFrame::new(face);
        let mut cells = Box::new([AIR; CHUNK_VOLUME]);
        for la in 0..CHUNK_SIZE {
            for lv in 0..CHUNK_SIZE {
                for lu in 0..CHUNK_SIZE {
                    let h = h0 + la as i32;
                    let rel = at(lu as i32, la as i32, lv as i32);
                    let depth = cols[lu + lv * CHUNK_SIZE].height - h;
                    let (lx, ly, lz) = if IDENTITY { (lu, la, lv) } else { frame.index_to_world(lu, la, lv) };
                    cells[Chunk::index(lx, ly, lz)] = self.deep_at(body, rel, Some(depth));
                }
            }
        }
        ChunkData::from_cells(cells)
    }

    /// Fill one chunk of a face column. `cols[lu + lv * 16].height` is the blended face-local surface.
    /// `IDENTITY` stores the cell at `(lu, la, lv)` (a chart chunk). Cube faces use `false`.
    fn fill_face<const IDENTITY: bool>(
        &self,
        body: &cosmos::Body,
        face: Face,
        cols: &[Column],
        u0: i32,
        v0: i32,
        h0: i32,
        max_terrain: i32,
        min_h: i32,
        tree_blocks: &[(i32, i32, i32, BlockId)],
    ) -> ChunkData {
        let n = CHUNK_SIZE as i32;
        // Unblended max keeps the v3 early-out. Blended max covers a rim that rose above it.
        // The facing canyon is taller: spires stop at `span::CLEAR`, still inside the relief.
        let max_h = cols.iter().map(|c| c.height).max().unwrap_or(i32::MIN);
        let facing = body.kind == cosmos::Kind::Twin && span::facing_face(&self.cosmos, body) == Some(face);
        let landmarks = max_terrain.max(max_h) + features::MAX_ABOVE;
        let clearance = if facing { span::CLEAR.max(landmarks) } else { landmarks };
        if h0 >= clearance {
            return ChunkData::Uniform(AIR);
        }
        if i64::from(h0) + i64::from(n) <= i64::from(min_h) - i64::from(cube::CRUST) {
            return self.fill_deep::<IDENTITY>(body, face, cols, u0, h0, v0);
        }
        let paint = self.paint(body, face);
        // The batch grid covers a 4-aligned 16³. PosY is aligned; a flipped axis is not,
        // and those faces sample the same corners the per-voxel path does.
        let aligned = u0.rem_euclid(4) == 0 && h0.rem_euclid(4) == 0 && v0.rem_euclid(4) == 0;
        // Floor within `CRUST` of the highest column: every solid cell is still crust.
        let crust_only = i64::from(h0) >= i64::from(max_h) - i64::from(cube::CRUST);
        let grid = (aligned && h0 < max_h).then(|| paint.under.grid(u0, h0, v0));
        let mut cells = Box::new([AIR; CHUNK_VOLUME]);
        // `h0 >= max_h`: every column tops out at or below this chunk, so only landmarks and trees write.
        if h0 < max_h && face == Face::PosY && crust_only {
            // Identity frame, the v3 surface loop: no permute and no bulk test.
            if let Some(g) = &grid {
                fill_posy_surface(paint, cols, g, u0, v0, h0, &mut cells);
            } else {
                fill_posy_corners(paint, cols, u0, v0, h0, &mut cells);
            }
        } else if h0 < max_h {
            let frame = FaceFrame::new(face);
            let half = cube::half_of(body) as i32;
            for lv in 0..CHUNK_SIZE {
                for lu in 0..CHUNK_SIZE {
                    let col = &cols[lu + lv * CHUNK_SIZE];
                    let (u, v) = (u0 + lu as i32, v0 + lv as i32);
                    for la in 0..CHUNK_SIZE {
                        let h = h0 + la as i32;
                        let (lx, ly, lz) = if IDENTITY { (lu, la, lv) } else { frame.index_to_world(lu, la, lv) };
                        let id = if h >= col.height {
                            AIR
                        } else if !crust_only && col.height - h > cube::CRUST {
                            let rel = cube::local_to_rel(face, u, half + h, v);
                            self.deep_at(body, rel, Some(col.height - h))
                        } else {
                            let la_i = la as i32;
                            let ground = paint.shape.ground(col, u, h, v);
                            if let Some(g) = &grid {
                                paint.under.finish(
                                    col,
                                    u,
                                    h,
                                    v,
                                    ground,
                                    g.at(lu, la_i, lv),
                                    || g.at(lu, la_i - 1, lv),
                                    || g.at(lu, la_i + 1, lv),
                                )
                            } else {
                                let field = |yy: i32| {
                                    Grid::interp_corners(&paint.under.corners(u, yy, v), u, yy, v)
                                };
                                paint.under.finish(
                                    col,
                                    u,
                                    h,
                                    v,
                                    ground,
                                    field(h),
                                    || field(h - 1),
                                    || field(h + 1),
                                )
                            }
                        };
                        cells[Chunk::index(lx, ly, lz)] = id;
                    }
                }
            }
        }
        let frame = FaceFrame::new(face);
        let owned = paint.structures.owned(&paint.shape, &paint.under, cols, u0, v0, n);
        let stamps = paint.features.blocks_in(&paint.shape, cols, u0, v0, n, h0, h0 + n);
        let mut claim = [false; CHUNK_VOLUME];
        for &(u, h, v, id, dig) in &stamps {
            let (lu, la, lv) = (u - u0, h - h0, v - v0);
            if !(0..n).contains(&lu) || !(0..n).contains(&la) || !(0..n).contains(&lv) {
                continue;
            }
            let col_i = lu as usize + lv as usize * CHUNK_SIZE;
            if !owned.is_empty() && owned[col_i] {
                continue;
            }
            let above = h >= cols[col_i].height;
            let (lx, ly, lz) = if IDENTITY {
                (lu as usize, la as usize, lv as usize)
            } else {
                frame.index_to_world(lu as usize, la as usize, lv as usize)
            };
            let i = Chunk::index(lx, ly, lz);
            if above {
                if id != AIR && cells[i] == AIR {
                    cells[i] = id;
                }
                if dig && id == AIR {
                    claim[i] = true;
                }
            } else if dig {
                cells[i] = id;
            }
        }
        let built = paint.structures.blocks_in(&paint.shape, &paint.under, cols, u0, v0, n, h0, h0 + n);
        for &(u, h, v, id, dig) in &built {
            let (lu, la, lv) = (u - u0, h - h0, v - v0);
            if !(0..n).contains(&lu) || !(0..n).contains(&la) || !(0..n).contains(&lv) {
                continue;
            }
            let col_i = lu as usize + lv as usize * CHUNK_SIZE;
            let above = h >= cols[col_i].height;
            let (lx, ly, lz) = if IDENTITY {
                (lu as usize, la as usize, lv as usize)
            } else {
                frame.index_to_world(lu as usize, la as usize, lv as usize)
            };
            let i = Chunk::index(lx, ly, lz);
            if above {
                if id != AIR {
                    cells[i] = id;
                } else if dig {
                    cells[i] = AIR;
                    claim[i] = true;
                }
            } else if dig {
                cells[i] = id;
            }
        }
        let open = paint.features.surface_open(&paint.shape, cols, u0, v0, n);
        for &(u, h, v, id) in tree_blocks {
            let (lu, la, lv) = (u - u0, h - h0, v - v0);
            if !(0..n).contains(&lu) || !(0..n).contains(&la) || !(0..n).contains(&lv) {
                continue;
            }
            let col_i = lu as usize + lv as usize * CHUNK_SIZE;
            if open[col_i] || (!owned.is_empty() && owned[col_i]) {
                continue;
            }
            let (lx, ly, lz) = if IDENTITY {
                (lu as usize, la as usize, lv as usize)
            } else {
                frame.index_to_world(lu as usize, la as usize, lv as usize)
            };
            let i = Chunk::index(lx, ly, lz);
            if cells[i] == AIR && !claim[i] && h >= cols[col_i].height {
                cells[i] = id;
            }
        }
        // A flower is one block on the ground cell. A tree or landmark already there stays.
        for lv in 0..CHUNK_SIZE {
            for lu in 0..CHUNK_SIZE {
                let col_i = lu + lv * CHUNK_SIZE;
                if open[col_i] || (!owned.is_empty() && owned[col_i]) {
                    continue;
                }
                let col = &cols[col_i];
                let la = col.height - h0;
                if !(0..n).contains(&la) {
                    continue;
                }
                let (u, v) = (u0 + lu as i32, v0 + lv as i32);
                let Some(id) = paint.shape.flower_at(col, u, v) else { continue };
                let (lx, ly, lz) = if IDENTITY { (lu, la as usize, lv) } else { frame.index_to_world(lu, la as usize, lv) };
                let i = Chunk::index(lx, ly, lz);
                if cells[i] == AIR && !claim[i] {
                    cells[i] = id;
                }
            }
        }
        if facing && h0 < span::CLEAR {
            let half = cube::half_of(body);
            let lush = span::lush(&self.cosmos, body);
            let seed = span::face_seed(body);
            let min_col = cols.iter().map(|c| c.height).min().unwrap_or(i32::MAX);
            if h0 + n > min_col {
                for lv in 0..CHUNK_SIZE {
                    for lu in 0..CHUNK_SIZE {
                        let col_h = cols[lu + lv * CHUNK_SIZE].height;
                        if h0 + n <= col_h {
                            continue;
                        }
                        let (u, v) = (u0 + lu as i32, v0 + lv as i32);
                        for la in 0..CHUNK_SIZE {
                            let h = h0 + la as i32;
                            if h < col_h {
                                continue;
                            }
                            let (lx, ly, lz) = if IDENTITY { (lu, la, lv) } else { frame.index_to_world(lu, la, lv) };
                            let i = Chunk::index(lx, ly, lz);
                            // As `cube_cell`: a landmark's dug air and a cut column stay open.
                            let cut = open[lu + lv * CHUNK_SIZE] && h - col_h <= trees::MAX_TREE_HEIGHT;
                            if cells[i] != AIR || claim[i] || cut {
                                continue;
                            }
                            if let Some(id) = span::block(&self.m, lush, seed, half, u, h, v) {
                                cells[i] = id;
                            }
                        }
                    }
                }
            }
        }
        ChunkData::from_cells(cells)
    }

    /// Virtual-cube `(rel, face altitude, u, v)` of a band-0 chart cell.
    fn chart_sample(&self, face: Face, i: i64, y: i64, j: i64) -> Option<([i64; 3], i32, i32, i32)> {
        let atlas = self.storage.home_atlas()?;
        let b = atlas.bands[0];
        let half = b.n / 2;
        let h64 = b.r_lo + y - atlas.radius;
        let h = i32::try_from(h64).ok()?;
        let a = i32::try_from(half + h64).ok()?;
        let u = i32::try_from(i - half).ok()?;
        let v = i32::try_from(j - half).ok()?;
        Some((cube::local_to_rel(face, u, a, v), h, u, v))
    }

    /// Per-voxel block of a start-world band-0 cell. Same decisions as [`cube_cell`](Self::cube_cell).
    fn chart_voxel(&self, face: Face, local: [i64; 3]) -> BlockId {
        let Some((rel, h, u, v)) = self.chart_sample(face, local[0], local[1], local[2]) else { return AIR };
        let body = self.cosmos.home();
        let half = cube::half_of(body);
        if cube::in_deep(rel, half) {
            return self.deep_at(body, rel, None);
        }
        let paint = self.paint(body, face);
        let mut col = paint.shape.column(u, v);
        col.height = cube::blend_height(col.height, cube::rim_seed(body), face, half, i64::from(u), i64::from(v));
        self.face_voxel(body, face, u, h, v, &col, rel)
    }

    fn chart_columns(&self, face: Face, local: [i64; 3]) -> ChartCols {
        let atlas = self.storage.home_atlas().expect("the start world is charted");
        let b = atlas.bands[0];
        let half = b.n / 2;
        let (i0, y0, j0) = (local[0], local[1], local[2]);
        let (u0, v0) = ((i0 - half) as i32, (j0 - half) as i32);
        let h_base = (b.r_lo + y0 - atlas.radius) as i32;
        let body = self.cosmos.home();
        let paint = self.paint(body, face);
        let seed = cube::rim_seed(body);
        let raw = paint.shape.columns_16(u0, v0);
        let mut cols = Vec::with_capacity(CHUNK_SIZE * CHUNK_SIZE);
        let mut max_h = i32::MIN;
        let mut min_h = i32::MAX;
        let mut max_terrain = i32::MIN;
        for (k, mut col) in raw.into_iter().enumerate() {
            let lu = (k % CHUNK_SIZE) as i32;
            let lv = (k / CHUNK_SIZE) as i32;
            max_terrain = max_terrain.max(col.height);
            let (ub, vb) = (i64::from(u0) + i64::from(lu), i64::from(v0) + i64::from(lv));
            col.height = cube::blend_height(col.height, seed, face, half, ub, vb);
            max_h = max_h.max(col.height);
            min_h = min_h.min(col.height);
            cols.push(col);
        }
        ChartCols { cols, u0, v0, h_base, max_h, min_h, max_terrain, face, y0 }
    }

    /// Air, or one deep material, when the chunk cannot meet the crust. `None` when it might.
    fn home_fast_uniform(&self, face: Face, local: [i64; 3]) -> Option<BlockId> {
        let atlas = self.storage.home_atlas()?;
        let b = atlas.bands[0];
        let (i0, y0, j0) = (local[0], local[1], local[2]);
        let h_base = b.r_lo + y0 - atlas.radius;
        if h_base >= i64::from(MAX_GROUND) + i64::from(features::MAX_ABOVE) {
            return Some(AIR);
        }
        let h_top = h_base + i64::from(CHUNK_SIZE as i32 - 1);
        if h_top >= i64::from(MIN_GROUND) - i64::from(cube::CRUST) {
            return None;
        }
        let body = self.cosmos.home();
        let at = |di: i64, dy: i64, dj: i64| self.chart_sample(face, i0 + di, y0 + dy, j0 + dj).map(|(rel, _, _, _)| rel);
        let rels = [
            at(0, 0, 0)?,
            at(15, 0, 0)?,
            at(0, 0, 15)?,
            at(15, 0, 15)?,
            at(0, 15, 0)?,
            at(15, 15, 0)?,
            at(0, 15, 15)?,
            at(15, 15, 15)?,
        ];
        let mut lo = rels[0];
        let mut hi = rels[0];
        for p in &rels[1..] {
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        if self.deep.all_air(body, lo, hi) {
            return Some(AIR);
        }
        if !self.deep.hits(body, lo, hi)
            && let Some(id) = cube::bulk_uniform(&self.bulk, body, &rels)
        {
            return Some(id);
        }
        None
    }

    /// What [`fill_face`](Self::fill_face) would collapse to, so classify and generate agree.
    fn columns_uniform(&self, c: &ChartCols) -> Option<BlockId> {
        let landmarks = c.max_terrain.max(c.max_h) + features::MAX_ABOVE;
        let h0 = c.h_base;
        if h0 >= landmarks {
            return Some(AIR);
        }
        let n = CHUNK_SIZE as i32;
        if i64::from(h0) + i64::from(n) > i64::from(c.min_h) - i64::from(cube::CRUST) {
            return None;
        }
        let body = self.cosmos.home();
        let half = cube::half_of(body) as i32;
        let at = |lu: i32, la: i32, lv: i32| cube::local_to_rel(c.face, c.u0 + lu, half + h0 + la, c.v0 + lv);
        let rels = [
            at(0, 0, 0),
            at(15, 0, 0),
            at(0, 0, 15),
            at(15, 0, 15),
            at(0, 15, 0),
            at(15, 15, 0),
            at(0, 15, 15),
            at(15, 15, 15),
        ];
        let mut lo = rels[0];
        let mut hi = rels[0];
        for p in &rels[1..] {
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        if self.deep.all_air(body, lo, hi) {
            return Some(AIR);
        }
        if !self.deep.hits(body, lo, hi) { cube::bulk_uniform(&self.bulk, body, &rels) } else { None }
    }

    fn fill_chart(&self, c: &ChartCols) -> ChunkData {
        let body = self.cosmos.home();
        let paint = self.paint(body, c.face);
        let trees = paint.trees.blocks_in(&paint.shape, c.u0, c.v0, CHUNK_SIZE as i32);
        self.fill_face::<true>(body, c.face, &c.cols, c.u0, c.v0, c.h_base, c.max_terrain, c.min_h, &trees)
    }

    fn home_chunk_data(&self, coord: ChunkCoord) -> Option<ChunkData> {
        let (patch, local) = self.storage.home_chunk(coord)?;
        let crate::space::atlas::Patch::Shell { band: 0, face } = patch else {
            return Some(ChunkData::Uniform(self.storage.home_fill()));
        };
        if let Some(id) = self.home_fast_uniform(face, local) {
            return Some(ChunkData::Uniform(id));
        }
        let cols = self.chart_columns(face, local);
        if let Some(id) = self.columns_uniform(&cols) {
            return Some(ChunkData::Uniform(id));
        }
        Some(self.fill_chart(&cols))
    }

    /// A storage column of the start world's band 0, painted from one set of column samples (the
    /// columns depend on the chart column only, not on the chunk's altitude).
    fn home_column(&self, key: ColumnKey, range: std::ops::RangeInclusive<i32>) -> Option<(Vec<(i32, ChunkData)>, ColumnHeights)> {
        let first = key.chunk(*range.start());
        let (patch, local) = self.storage.home_chunk(first)?;
        let crate::space::atlas::Patch::Shell { band: 0, face } = patch else { return None };
        let atlas = self.storage.home_atlas()?;
        let b = atlas.bands[0];
        let mut cols: Option<ChartCols> = None;
        let mut chunks = Vec::with_capacity(range.clone().count());
        for alt in range {
            let coord = key.chunk(alt);
            let Some((p, l)) = self.storage.home_chunk(coord) else {
                chunks.push((alt, self.storage.generate(coord)));
                continue;
            };
            debug_assert_eq!(p, patch, "boxes are never stacked: one patch per storage column");
            if let Some(id) = self.home_fast_uniform(face, l) {
                chunks.push((alt, ChunkData::Uniform(id)));
                continue;
            }
            let base = cols.get_or_insert_with(|| self.chart_columns(face, [local[0], 0, local[2]]));
            let mut c = base.clone();
            c.y0 = l[1];
            c.h_base = (b.r_lo + l[1] - atlas.radius) as i32;
            let data = match self.columns_uniform(&c) {
                Some(id) => ChunkData::Uniform(id),
                None => self.fill_chart(&c),
            };
            chunks.push((alt, data));
        }
        let base = match cols {
            Some(c) => c,
            None => self.chart_columns(face, [local[0], 0, local[2]]),
        };
        let rise = atlas.radius - b.r_lo;
        let mut heights = [0i32; CHUNK_SIZE * CHUNK_SIZE];
        for k in 0..CHUNK_SIZE * CHUNK_SIZE {
            let y = i64::from(base.cols[k].height) + rise;
            heights[k] = y.clamp(i32::MIN as i64 + 1, storage::BURIED as i64 - 1) as i32;
        }
        Some((chunks, heights))
    }

    fn home_class(&self, coord: ChunkCoord) -> Option<Classify> {
        let (patch, local) = self.storage.home_chunk(coord)?;
        let crate::space::atlas::Patch::Shell { band: 0, face } = patch else {
            return Some(Classify::Uniform(self.storage.home_fill()));
        };
        // Bounds only: classification runs on the streaming thread for every candidate chunk, and
        // a near-surface chunk the bounds cannot settle is generated (and collapsed) on a worker.
        Some(self.home_fast_uniform(face, local).map_or(Classify::Mixed, Classify::Uniform))
    }

    /// The start-world patch covering storage column `(x, z)`, with its local column.
    fn home_patch_column(&self, x: i32, z: i32) -> Option<(crate::space::atlas::Patch, i64, i64)> {
        let atlas = self.storage.home_atlas()?;
        let (x, z) = (i64::from(x), i64::from(z));
        let hit = |patch: crate::space::atlas::Patch| {
            let (o, size) = atlas.storage_box(patch);
            (x >= o[0] && x < o[0] + size[0] && z >= o[2] && z < o[2] + size[2]).then_some((patch, x - o[0], z - o[2]))
        };
        for face in Face::ALL {
            for band in 0..atlas.bands.len() {
                if let Some(found) = hit(crate::space::atlas::Patch::Shell { band: band as u8, face }) {
                    return Some(found);
                }
            }
            if atlas.inner.is_some()
                && let Some(found) = hit(crate::space::atlas::Patch::Transition { face })
            {
                return Some(found);
            }
        }
        if atlas.inner.is_some() { hit(crate::space::atlas::Patch::Core) } else { None }
    }

    /// Storage y of the first open cell on a start-world column, or [`storage::BURIED`] under it.
    fn home_surface_y(&self, x: i32, z: i32) -> Option<i32> {
        let (patch, i, j) = self.home_patch_column(x, z)?;
        let crate::space::atlas::Patch::Shell { band: 0, face } = patch else {
            return Some(storage::BURIED);
        };
        let atlas = self.storage.home_atlas()?;
        let b = atlas.bands[0];
        let half = b.n / 2;
        let (u, v) = ((i - half) as i32, (j - half) as i32);
        let body = self.cosmos.home();
        let paint = self.paint(body, face);
        let h = cube::blend_height(
            paint.shape.height(u, v),
            cube::rim_seed(body),
            face,
            half,
            i64::from(u),
            i64::from(v),
        );
        let y = i64::from(h) + atlas.radius - b.r_lo;
        Some(y.clamp(i32::MIN as i64 + 1, storage::BURIED as i64 - 1) as i32)
    }

    fn home_heights_16(&self, cx: i32, cz: i32) -> Option<ColumnHeights> {
        let (patch, i0, j0) = self.home_patch_column(cx * CHUNK_SIZE as i32, cz * CHUNK_SIZE as i32)?;
        let crate::space::atlas::Patch::Shell { band: 0, face } = patch else {
            return Some([storage::BURIED; CHUNK_SIZE * CHUNK_SIZE]);
        };
        let cols = self.chart_columns(face, [i0, 0, j0]);
        let atlas = self.storage.home_atlas()?;
        let rise = atlas.radius - atlas.bands[0].r_lo;
        let mut out = [0i32; CHUNK_SIZE * CHUNK_SIZE];
        for k in 0..CHUNK_SIZE * CHUNK_SIZE {
            let y = i64::from(cols.cols[k].height) + rise;
            out[k] = y.clamp(i32::MIN as i64 + 1, storage::BURIED as i64 - 1) as i32;
        }
        Some(out)
    }

    fn home_bounds(&self, x0: i32, z0: i32, span: i32) -> Option<(i32, i32)> {
        if span <= 0 {
            return None;
        }
        let atlas = self.storage.home_atlas()?;
        let (x1, z1) = (i64::from(x0) + i64::from(span) - 1, i64::from(z0) + i64::from(span) - 1);
        for face in Face::ALL {
            let patch = crate::space::atlas::Patch::Shell { band: 0, face };
            let (o, size) = atlas.storage_box(patch);
            if i64::from(x0) < o[0] || x1 >= o[0] + size[0] || i64::from(z0) < o[2] || z1 >= o[2] + size[2] {
                continue;
            }
            let rise = atlas.radius - atlas.bands[0].r_lo;
            let lo = i32::try_from(i64::from(MIN_GROUND) + rise).ok()?;
            let hi = i32::try_from(i64::from(MAX_GROUND) + rise).unwrap_or(i32::MAX);
            return Some((lo, hi.max(lo.saturating_add(1))));
        }
        None
    }

    /// Far field of a start-world column. `false` when `(x, z)` is not the start world.
    fn home_lod_column(&self, x: i32, z: i32, ys: &[i32], out: &mut [BlockId]) -> bool {
        let Some((patch, i, j)) = self.home_patch_column(x, z) else { return false };
        let n = out.len().min(ys.len());
        let crate::space::atlas::Patch::Shell { band: 0, face } = patch else {
            out[..n].fill(self.storage.home_fill());
            return true;
        };
        let Some(atlas) = self.storage.home_atlas() else { return false };
        let b = atlas.bands[0];
        let half = b.n / 2;
        let (u, v) = ((i - half) as i32, (j - half) as i32);
        let body = self.cosmos.home();
        let paint = self.paint(body, face);
        let mut col = paint.shape.column(u, v);
        col.height = cube::blend_height(col.height, cube::rim_seed(body), face, half, i64::from(u), i64::from(v));
        let surf = i64::from(col.height) + atlas.radius - b.r_lo;
        for (o, &y) in out.iter_mut().zip(ys).take(n) {
            if i64::from(y) >= surf {
                *o = AIR;
            } else {
                let h = (b.r_lo + i64::from(y) - atlas.radius) as i32;
                *o = paint.shape.ground(&col, u, h, v);
            }
        }
        if super::generation::coarse_floor_samples(ys) {
            super::generation::paint_lod_top(out, n, col.surface);
        }
        true
    }

    fn home_block(&self, x: i32, y: i32, z: i32) -> Option<BlockId> {
        let (patch, local) = self.storage.home_cell(x, y, z)?;
        Some(match patch {
            crate::space::atlas::Patch::Shell { band: 0, face } => self.chart_voxel(face, local),
            _ => self.storage.home_fill(),
        })
    }

    /// Physical spawn: the +Y chart's centre column, two blocks above its first open cell.
    fn home_spawn(&self) -> Option<glam::DVec3> {
        let atlas = self.storage.home_atlas()?;
        let b = atlas.bands[0];
        let face = Face::PosY;
        let patch = crate::space::atlas::Patch::Shell { band: 0, face };
        let (i, j) = (b.n / 2, b.n / 2);
        let half = b.n / 2;
        let body = self.cosmos.home();
        let paint = self.paint(body, face);
        let h = cube::blend_height(paint.shape.height(0, 0), cube::rim_seed(body), face, half, 0, 0);
        let y = i64::from(h) + atlas.radius - b.r_lo;
        Some(atlas.embed(patch, glam::DVec3::new(i as f64 + 0.5, y as f64 + 2.0, j as f64 + 0.5)))
    }

    /// Replace the start world's spherical datum. Tests use this; a relaxed field is data.
    #[cfg(test)]
    pub fn set_home_datum(&mut self, field: std::sync::Arc<crate::space::datum::DatumField>) {
        self.storage.set_home_datum(field);
    }

    fn face_columns(
        &self,
        body: &cosmos::Body,
        key: ColumnKey,
        range: std::ops::RangeInclusive<i32>,
    ) -> (Vec<(i32, ChunkData)>, ColumnHeights) {
        let face = key.face;
        let half = cube::half_of(body);
        let centre = cube::centre_i32(body.centre).expect("cube centre fits i32");
        let (wu, wv) = key.column_cell_uv(0, 0);
        let (u0, v0) = cube::tangents(face, centre, wu, wv);
        let (u0, v0) = (u0 as i32, v0 as i32);
        let paint = self.paint(body, face);
        let seed = cube::rim_seed(body);
        let n_dot = cube::normal_dot(body.centre, face);
        let raw = paint.shape.columns_16(u0, v0);
        let mut heights = [0i32; CHUNK_SIZE * CHUNK_SIZE];
        let mut cols = Vec::with_capacity(CHUNK_SIZE * CHUNK_SIZE);
        let mut max_terrain = i32::MIN;
        let mut min_h = i32::MAX;
        for (i, mut col) in raw.into_iter().enumerate() {
            let lu = (i % CHUNK_SIZE) as i32;
            let lv = (i / CHUNK_SIZE) as i32;
            let (ub, vb) = (i64::from(u0) + i64::from(lu), i64::from(v0) + i64::from(lv));
            max_terrain = max_terrain.max(col.height);
            col.height = cube::blend_height(col.height, seed, face, half, ub, vb);
            min_h = min_h.min(col.height);
            heights[i] = cube::world_a(half, col.height, n_dot).unwrap_or(i32::MIN);
            cols.push(col);
        }
        if range.is_empty() {
            return (Vec::new(), heights);
        }
        let tree_blocks = paint.trees.blocks_in(&paint.shape, u0, v0, CHUNK_SIZE as i32);
        let frame = FaceFrame::new(face);
        let chunks = range
            .map(|alt| {
                let coord = key.chunk(alt);
                let h0 = cube::face_h(half, n_dot, frame.chunk_alt0(coord));
                let data = self.fill_face::<false>(body, face, &cols, u0, v0, h0, max_terrain, min_h, &tree_blocks);
                (alt, data)
            })
            .collect();
        (chunks, heights)
    }

    fn column_heights_of(&self, key: ColumnKey) -> ColumnHeights {
        let mut heights = [0i32; CHUNK_SIZE * CHUNK_SIZE];
        for lv in 0..CHUNK_SIZE {
            for lu in 0..CHUNK_SIZE {
                let (u, v) = key.column_cell_uv(lu as i32, lv as i32);
                heights[lu + lv * CHUNK_SIZE] = self.surface(key.face, u, v);
            }
        }
        heights
    }

    /// How high a uniform-air shortcut may start. The canyon's facing face keeps room for spires;
    /// every other face stops at the trees. Home never takes the tall path.
    fn face_clear(&self, body: &cosmos::Body, lo: [i64; 3], hi: [i64; 3]) -> i32 {
        if body.kind != cosmos::Kind::Twin {
            return cube::TREE_CLEAR;
        }
        let Some(face) = span::facing_face(&self.cosmos, body) else { return cube::TREE_CLEAR };
        let on_face = cube::corners(lo, hi).into_iter().all(|p| {
            let rel = [p[0] - body.centre[0], p[1] - body.centre[1], p[2] - body.centre[2]];
            cube::face_of(rel) == face
        });
        if on_face { span::CLEAR.max(cube::TREE_CLEAR) } else { cube::TREE_CLEAR }
    }

    /// The cube-face sky rule. `include` counts stored cubes (their reference chunks).
    fn sky_of(&self, coord: ChunkCoord, include: bool) -> Sky {
        let (lo, hi) = cube::chunk_bounds(coord);
        let mut owned: Option<(cosmos::Body, Face)> = None;
        let mut min_inside = i64::MAX;
        for p in cube::corners(lo, hi) {
            let Some(body) = self.owner_in(p, include) else { return Sky::Open };
            let cosmos::Shape::Cube { .. } = body.shape else { return Sky::Open };
            let rel = [p[0] - body.centre[0], p[1] - body.centre[1], p[2] - body.centre[2]];
            let (Ok(x), Ok(y), Ok(z)) = (i32::try_from(rel[0]), i32::try_from(rel[1]), i32::try_from(rel[2])) else {
                return Sky::Open;
            };
            let face = cube::face_of(rel);
            match owned {
                None => owned = Some((body, face)),
                Some((b, f)) if b.id != body.id || f != face => return Sky::Open,
                _ => {}
            }
            let (u, _, v) = FaceFrame::new(face).cell_to_local((x, y, z));
            min_inside = min_inside.min(cube::edge_inside(cube::half_of(&body), i64::from(u), i64::from(v)));
        }
        let Some((body, face)) = owned else { return Sky::Open };
        if min_inside < cube::SKY_EDGE || self.outward_blocked(&body, face, coord) {
            return Sky::Open;
        }
        Sky::Axis(face)
    }

    fn classify_in(&self, coord: ChunkCoord, include: bool) -> Classify {
        let (lo, hi) = cube::chunk_bounds(coord);
        if !self.cosmos.may_hold(lo, hi) {
            return Classify::Air;
        }
        let mut only: Option<cosmos::Body> = None;
        for b in self.cosmos.bodies_touching(lo, hi) {
            if !include && self.storage.holds_cube(b.id) {
                continue;
            }
            if only.is_some() {
                return Classify::Mixed;
            }
            only = Some(*b);
        }
        let Some(body) = only else {
            // A rock's sub-cell is huge. Only a chunk the reserved box actually meets is mixed.
            return if space::any_overlap(&self.cosmos, lo, hi) { Classify::Mixed } else { Classify::Air };
        };
        match body.shape {
            cosmos::Shape::Cube { .. } => {
                let half = cube::half_of(&body);
                let clear = self.face_clear(&body, lo, hi);
                if cube::min_reach(body.centre, lo, hi) - half >= i64::from(clear) {
                    return Classify::Uniform(AIR);
                }
                let rels = cube::corners(lo, hi)
                    .map(|p| [p[0] - body.centre[0], p[1] - body.centre[1], p[2] - body.centre[2]]);
                if rels.iter().copied().all(|r| cube::in_deep(r, half)) {
                    let (rlo, rhi) = rel_box(body.centre, lo, hi);
                    if self.deep.all_air(&body, rlo, rhi) {
                        return Classify::Uniform(AIR);
                    }
                    if !self.deep.hits(&body, rlo, rhi)
                        && let Some(id) = cube::bulk_uniform(&self.bulk, &body, &rels)
                    {
                        return Classify::Uniform(id);
                    }
                }
                Classify::Mixed
            }
            // Charted: nothing of a round body is in physical space.
            cosmos::Shape::Ball { .. } | cosmos::Shape::Shell { .. } => Classify::Air,
        }
    }

    fn generate_in(&self, coord: ChunkCoord, include: bool) -> ChunkData {
        match self.classify_in(coord, include) {
            Classify::Air => return ChunkData::Uniform(AIR),
            Classify::Uniform(id) => return ChunkData::Uniform(id),
            Classify::Mixed => {}
        }
        match self.sky_of(coord, include) {
            Sky::Axis(face) => {
                let (key, alt) = ColumnKey::of(face, coord);
                let (mut chunks, _) = self.generate_column_in(key, alt..=alt, include);
                chunks.pop().expect("the requested layer").1
            }
            Sky::Open => self.fill_slow_in(coord.x, coord.y, coord.z, include),
        }
    }

    fn generate_column_in(
        &self,
        key: ColumnKey,
        range: std::ops::RangeInclusive<i32>,
        include: bool,
    ) -> (Vec<(i32, ChunkData)>, ColumnHeights) {
        // Storage columns: the chart painters, chunk by chunk; heights are the charts' surfaces.
        if key.face == Face::PosY && storage::StorageWorlds::owns(key.chunk(*range.start())) {
            let c = key.chunk(*range.start());
            if let Some(column) = self.home_column(key, range.clone()) {
                return column;
            }
            if let Some(heights) = self.home_heights_16(c.x, c.z) {
                let chunks = range
                    .map(|alt| {
                        let coord = key.chunk(alt);
                        let data = self.home_chunk_data(coord).unwrap_or_else(|| self.storage.generate(coord));
                        (alt, data)
                    })
                    .collect();
                return (chunks, heights);
            }
            let chunks = range.map(|alt| (alt, self.storage.generate(key.chunk(alt)))).collect();
            return (chunks, self.storage.heights_16(c.x, c.z));
        }
        if let Some(body) = self.face_column_body_in(key, include) {
            if range.clone().all(|alt| self.sky_of(key.chunk(alt), include) == Sky::Axis(key.face)) {
                return self.face_columns(&body, key, range);
            }
        }
        let heights = self.column_heights_of(key);
        if range.is_empty() {
            return (Vec::new(), heights);
        }
        let chunks = range
            .map(|alt| {
                let c = key.chunk(alt);
                (alt, self.fill_slow_in(c.x, c.y, c.z, include))
            })
            .collect();
        (chunks, heights)
    }
}

/// +Y surface chunk on the batch grid. Same loop as v3: identity axes, crust only.
fn fill_posy_surface(
    paint: &FacePaint,
    cols: &[Column],
    g: &Grid,
    u0: i32,
    v0: i32,
    h0: i32,
    cells: &mut [BlockId; CHUNK_VOLUME],
) {
    for lz in 0..CHUNK_SIZE {
        for lx in 0..CHUNK_SIZE {
            let col = &cols[lx + lz * CHUNK_SIZE];
            let (x, z) = (u0 + lx as i32, v0 + lz as i32);
            for ly in 0..CHUNK_SIZE {
                let y = h0 + ly as i32;
                let id = if y >= col.height {
                    AIR
                } else {
                    let ly_i = ly as i32;
                    let ground = paint.shape.ground(col, x, y, z);
                    paint.under.finish(
                        col,
                        x,
                        y,
                        z,
                        ground,
                        g.at(lx, ly_i, lz),
                        || g.at(lx, ly_i - 1, lz),
                        || g.at(lx, ly_i + 1, lz),
                    )
                };
                cells[Chunk::index(lx, ly, lz)] = id;
            }
        }
    }
}

/// +Y surface chunk whose origin is not 4-aligned, so it cannot use the batch grid.
fn fill_posy_corners(
    paint: &FacePaint,
    cols: &[Column],
    u0: i32,
    v0: i32,
    h0: i32,
    cells: &mut [BlockId; CHUNK_VOLUME],
) {
    for lz in 0..CHUNK_SIZE {
        for lx in 0..CHUNK_SIZE {
            let col = &cols[lx + lz * CHUNK_SIZE];
            let (x, z) = (u0 + lx as i32, v0 + lz as i32);
            for ly in 0..CHUNK_SIZE {
                let y = h0 + ly as i32;
                if y >= col.height {
                    continue;
                }
                let field = |yy: i32| Grid::interp_corners(&paint.under.corners(x, yy, z), x, yy, z);
                let ground = paint.shape.ground(col, x, y, z);
                cells[Chunk::index(lx, ly, lz)] =
                    paint.under.finish(col, x, y, z, ground, field(y), || field(y - 1), || field(y + 1));
            }
        }
    }
}

/// A PosY column resolved to one cube face.
struct Posy<'a> {
    world_a: i32,
    col: Column,
    paint: &'a FacePaint,
    u: i32,
    v: i32,
    half: i64,
    centre: [i64; 3],
}

impl TerrainGenerator for Terrain {
    fn seed(&self) -> i64 {
        self.seed
    }

    fn mass(&self) -> Arc<dyn crate::gravity::MassOracle> {
        self.cosmos.clone()
    }

    fn cosmos(&self) -> Option<&cosmos::Cosmos> {
        Some(self.cosmos.as_ref())
    }

    fn cosmos_arc(&self) -> Option<Arc<cosmos::Cosmos>> {
        Some(self.cosmos.clone())
    }

    fn kind(&self) -> &'static str {
        "diffusion"
    }

    fn height(&self, wx: i32, wz: i32) -> i32 {
        self.surface(Face::PosY, wx, wz)
    }

    fn atlases(&self) -> &[Arc<crate::space::atlas::Atlas>] {
        self.storage.atlases()
    }

    fn chart_spawn(&self) -> Option<voxel_engine::DVec3> {
        self.home_spawn()
    }

    fn sky(&self, coord: ChunkCoord) -> Sky {
        // A warped cube keeps a sky per face. Every chart's up is storage +Y.
        if let Some(reference) = self.cube_chunk(coord) {
            return self.sky_of(reference, true);
        }
        if storage::StorageWorlds::owns(coord) {
            return Sky::Axis(Face::PosY);
        }
        self.sky_of(coord, false)
    }

    fn classify(&self, coord: ChunkCoord) -> Classify {
        if let Some(reference) = self.cube_chunk(coord) {
            return self.classify_in(reference, true);
        }
        if storage::StorageWorlds::owns(coord) {
            if let Some(k) = self.home_class(coord) {
                return k;
            }
            return match self.storage.uniform(coord) {
                Some(AIR) => Classify::Air,
                Some(id) => Classify::Uniform(id),
                None => Classify::Mixed,
            };
        }
        self.classify_in(coord, false)
    }

    fn surface(&self, face: Face, u: i32, v: i32) -> i32 {
        if face == Face::PosY && stored(u) {
            if let Some(y) = self.home_surface_y(u, v) {
                return y;
            }
            return self.storage.surface(u, v);
        }
        let mut best = i32::MIN;
        let n = self.cosmos.bodies().len();
        for i in 0..n {
            let body = self.cosmos.bodies()[i];
            let cosmos::Shape::Cube { half } = body.shape else { continue };
            let Some(centre) = cube::centre_i32(body.centre) else { continue };
            let (ub, vb) = cube::tangents(face, centre, u, v);
            if cube::edge_inside(half, ub, vb) < -cosmos::RELIEF {
                continue;
            }
            let (Ok(ui), Ok(vi)) = (i32::try_from(ub), i32::try_from(vb)) else { continue };
            let paint = self.paint(&body, face);
            let terrain = paint.shape.height(ui, vi);
            let h = cube::blend_height(terrain, cube::rim_seed(&body), face, half, ub, vb);
            let a_body = half + i64::from(h) - 1;
            let Ok(a_i) = i32::try_from(a_body) else { continue };
            if cube::face_of(cube::local_to_rel(face, ui, a_i, vi)) != face {
                continue;
            }
            let Some(world_a) = cube::world_a(half, h, cube::normal_dot(body.centre, face)) else { continue };
            if world_a > best {
                best = world_a;
            }
        }
        best
    }

    fn heights_16(&self, cx: i32, cz: i32) -> ColumnHeights {
        if stored(cx * CHUNK_SIZE as i32) {
            if let Some(h) = self.home_heights_16(cx, cz) {
                return h;
            }
            return self.storage.heights_16(cx, cz);
        }
        super::generation::sample_column_heights(self, cx, cz)
    }

    fn surface_at(&self, wx: i32, wz: i32) -> BlockId {
        if stored(wx) {
            if let Some(s) = self.home_surface_y(wx, wz) {
                return if s == i32::MIN || s >= storage::BURIED {
                    AIR
                } else {
                    self.home_block(wx, s - 1, wz).unwrap_or(AIR)
                };
            }
            let s = self.storage.surface(wx, wz);
            return if s == i32::MIN || s >= storage::BURIED { AIR } else { self.storage.voxel(wx, s - 1, wz) };
        }
        self.posy_hit(wx, wz).map(|hit| hit.col.surface).unwrap_or(AIR)
    }

    fn deep(&self) -> BlockId {
        self.m.rock[0]
    }

    fn block_at(&self, wx: i32, wy: i32, wz: i32, _height: i32) -> BlockId {
        self.cell(wx, wy, wz)
    }

    fn voxel_at(&self, wx: i32, wy: i32, wz: i32) -> BlockId {
        self.cell(wx, wy, wz)
    }

    fn lod_block_at(&self, wx: i32, wy: i32, wz: i32) -> BlockId {
        if stored(wx) {
            if let Some(id) = self.home_block(wx, wy, wz) {
                return id;
            }
            return self.storage.voxel(wx, wy, wz);
        }
        let Some(hit) = self.posy_hit(wx, wz) else { return AIR };
        if wy >= hit.world_a {
            return AIR;
        }
        let h = cube::face_h(hit.half, cube::normal_dot(hit.centre, Face::PosY), wy);
        hit.paint.shape.ground(&hit.col, hit.u, h, hit.v)
    }

    fn lod_column(&self, wx: i32, wz: i32, ys: &[i32], out: &mut [BlockId]) {
        if stored(wx) {
            if self.home_lod_column(wx, wz, ys, out) {
                return;
            }
            self.storage.lod_column(wx, wz, ys, out);
            return;
        }
        let Some(hit) = self.posy_hit(wx, wz) else {
            for o in out.iter_mut().take(ys.len()) {
                *o = AIR;
            }
            return;
        };
        let n_dot = cube::normal_dot(hit.centre, Face::PosY);
        for (o, &wy) in out.iter_mut().zip(ys) {
            *o = if wy >= hit.world_a {
                AIR
            } else {
                let h = cube::face_h(hit.half, n_dot, wy);
                hit.paint.shape.ground(&hit.col, hit.u, h, hit.v)
            };
        }
        // The hit is already in hand: painting here does not sample the column again.
        if super::generation::coarse_floor_samples(ys) {
            super::generation::paint_lod_top(out, ys.len(), hit.col.surface);
        }
    }

    fn lod_column_face(&self, body: u16, face: Face, u: i32, v: i32, alts: &[i32], out: &mut [BlockId]) {
        if let Some(index) = chart_body(body) {
            if face != Face::PosY {
                out.iter_mut().take(alts.len()).for_each(|o| *o = AIR);
                return;
            }
            if self.storage.home_index() == Some(index) {
                let _ = self.home_lod_column(u, v, alts, out);
                return;
            }
            self.storage.lod_column(u, v, alts, out);
            return;
        }
        if face == Face::PosY {
            self.lod_column(u, v, alts, out);
            return;
        }
        let Some(hit) = self.named_face_hit(body, face, u, v) else {
            for o in out.iter_mut().take(alts.len()) {
                *o = AIR;
            }
            return;
        };
        let n_dot = cube::normal_dot(hit.centre, face);
        for (o, &a) in out.iter_mut().zip(alts) {
            *o = if a >= hit.world_a {
                AIR
            } else {
                let h = cube::face_h(hit.half, n_dot, a);
                hit.paint.shape.ground(&hit.col, hit.u, h, hit.v)
            };
        }
        if super::generation::coarse_floor_samples(alts) {
            super::generation::paint_lod_top(out, alts.len(), hit.col.surface);
        }
    }

    fn surface_bounds(&self, body: u16, face: Face, u0: i32, v0: i32, span: i32) -> Option<(i32, i32)> {
        if let Some(index) = chart_body(body) {
            if face != Face::PosY {
                return None;
            }
            if self.storage.home_index() == Some(index) {
                return self.home_bounds(u0, v0, span);
            }
            return self.storage.bounds(u0, v0, span);
        }
        let body = self.cosmos.bodies().iter().find(|b| b.id == body)?;
        let cosmos::Shape::Cube { half } = body.shape else { return None };
        let centre = cube::centre_i32(body.centre)?;
        let (cu, _, cv) = FaceFrame::new(face).cell_to_local(centre);
        let lim = half + cosmos::RELIEF;
        let u_lo = i64::from(u0) - i64::from(cu);
        let v_lo = i64::from(v0) - i64::from(cv);
        let u_hi = u_lo + i64::from(span);
        let v_hi = v_lo + i64::from(span);
        if u_lo >= lim || u_hi <= -lim || v_lo >= lim || v_hi <= -lim {
            return None;
        }
        let n = cube::normal_dot(body.centre, face);
        let lo = cube::world_a(half, MIN_GROUND, n)?;
        let hi = cube::world_a(half, MAX_GROUND, n)?;
        Some((lo.min(hi), lo.max(hi)))
    }

    fn face_datum(&self, body: u16, face: Face) -> i32 {
        if let Some(index) = chart_body(body) {
            return if face == Face::PosY { self.storage.datum(index).unwrap_or(0) } else { 0 };
        }
        let Some(body) = self.cosmos.bodies().iter().find(|b| b.id == body) else { return 0 };
        let cosmos::Shape::Cube { half } = body.shape else { return 0 };
        cube::world_a(half, 0, cube::normal_dot(body.centre, face)).unwrap_or(0)
    }

    fn generate(&self, cx: i32, cy: i32, cz: i32) -> ChunkData {
        let coord = ChunkCoord::new(cx, cy, cz);
        // One compare before the start-world painter: cube boxes begin past every chart.
        if let Some(reference) = self.cube_chunk(coord) {
            return self.generate_in(reference, true);
        }
        if storage::StorageWorlds::owns(coord) {
            if let Some(data) = self.home_chunk_data(coord) {
                return data;
            }
            return self.storage.generate(coord);
        }
        self.generate_in(coord, false)
    }

    fn generate_column(
        &self,
        key: ColumnKey,
        range: std::ops::RangeInclusive<i32>,
    ) -> (Vec<(i32, ChunkData)>, ColumnHeights) {
        // A stored cube's column is the reference painter, shifted into the storage altitude frame.
        // This runs before the chart +Y path: a cube box is storage too, and its +Y is not a chart.
        if let Some((reference, shift)) = self.storage.cube_column(key) {
            let lo = range.start().saturating_add(shift);
            let hi = range.end().saturating_add(shift);
            let (chunks, heights) = self.generate_column_in(reference, lo..=hi, true);
            let cells = shift.saturating_mul(CHUNK_SIZE as i32);
            let chunks = chunks.into_iter().map(|(alt, data)| (alt.saturating_sub(shift), data)).collect();
            let heights = heights.map(|h| {
                if h == i32::MIN || h == i32::MAX { h } else { h.saturating_sub(cells) }
            });
            return (chunks, heights);
        }
        self.generate_column_in(key, range, false)
    }
}

#[cfg(test)]
mod tests;
