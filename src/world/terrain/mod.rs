//! InfiniteDiffusion (v3): the world generator.
//!
//! One pure function of `(seed, coordinate)` paints three realms:
//!
//! * **The surface** — continents of lowland basins, rolling hills and great mountain ranges.
//!   Ridged multifractal crests over derivative-eroded slopes, cut by long carved valleys that run
//!   through the ranges, with stepped mesas in the arid belts and banded strata on every cliff.
//!   Biomes follow altitude, temperature and moisture: meadows, forests (broadleaf, pine, autumn,
//!   blossom), deserts and mesas, alpine scree, snowfields.
//! * **The underground** — spaghetti tunnels, caverns that widen with depth, glowing fungus on
//!   deep floors and crystal on the deepest ceilings; ore veins; and abandoned **mines**: timbered
//!   corridors on several levels, rails, rare lamps, collapsed stretches, rooms and shafts.
//! * **Space** — above [`SPACE_FLOOR`]: planets (rocky, icy, verdant, desert, crystal, molten)
//!   with crusts, mantles and glowing cores, rings and moons, drifting asteroids and stars.
//!
//! Every material is a configuration the [`palette`] found in the law; nothing here names an
//! element. The arithmetic is bit-identical on every peer (see [`noise`]).

pub mod cosmos;
pub mod noise;
pub mod palette;
pub mod round;
mod shape;
mod space;
mod trees;
mod underground;

use std::sync::Arc;

use super::chunk::{CHUNK_SIZE, CHUNK_VOLUME, Chunk, ChunkData};
use super::generation::{self, ColumnHeights, TerrainGenerator};
use super::layout::ColumnKey;
use crate::coord::Face;
use crate::block::registry::{AIR, BlockId, BlockRegistry};

use shape::{Column, Shape};
use space::Space;
use trees::Trees;
use underground::{Grid, Underground};

/// Bumped whenever the same `(seed, coordinate)` can generate different materials than before.
/// Folded into the content fingerprint and recorded in saves. History: 1-5 the classic and
/// diffusion v1/v2 generators over authored and then emergent materials; 6 = InfiniteDiffusion v3
/// over the selective-transfer palette (2026-10-02).
pub const WORLDGEN_VERSION: u16 = 6;

/// Everything at or above this height is space (above the far-LOD window, so planets never coarsen
/// the terrain's distant meshes).
pub const SPACE_FLOOR: i32 = 640;
/// Ground never rises above this (inside the far-LOD window `[0, 512)`).
pub const MAX_GROUND: i32 = 470;
/// Ground never sinks below this (the far-LOD floor is 0).
pub const MIN_GROUND: i32 = 6;

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
}

impl Default for TerrainCfg {
    fn default() -> Self {
        Self { relief: 100, caves: 100, mines: 100, space: 100 }
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
        self
    }

    /// Text form (`relief=100,caves=100,mines=100,space=100`): saves, the wire, mod state.
    pub fn to_text(self) -> String {
        format!("relief={},caves={},mines={},space={}", self.relief, self.caves, self.mines, self.space)
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
                _ => continue,
            };
            *slot = v.trim().parse().unwrap_or(*slot);
        }
        self.clamp()
    }

    /// The four knobs in wire order.
    pub fn to_wire(self) -> [u16; 4] {
        [self.relief, self.caves, self.mines, self.space]
    }

    /// Inverse of [`to_wire`](Self::to_wire).
    pub fn from_wire(v: [u16; 4]) -> Self {
        Self { relief: v[0], caves: v[1], mines: v[2], space: v[3] }.clamp()
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
            reagents,
        }
    }
}

/// The generator.
pub struct Terrain {
    seed: i64,
    /// Every body in the universe; also the generator's mass oracle.
    cosmos: Arc<cosmos::Cosmos>,
    shape: Shape,
    under: Underground,
    trees: Trees,
    space: Space,
    m: Arc<Materials>,
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
        Self {
            seed,
            cosmos: Arc::new(cosmos::Cosmos::new(s, cfg.space as f32 / 100.0)),
            shape: Shape::new(s, cfg.relief as f32 / 100.0, m.clone()),
            under: Underground::new(s ^ 0x0BAD_CAFE, cfg, m.clone()),
            trees: Trees::new(s ^ 0x7EE5_0000, m.clone()),
            space: Space::new(s ^ 0x5BAC_E000, cfg.space as f32 / 100.0, m.clone()),
            m,
        }
    }

    /// The interned palette.
    #[cfg(test)]
    pub fn materials(&self) -> &Materials {
        &self.m
    }

    /// The full column description at `(x, z)` (height, biome, surface layers).
    fn column(&self, x: i32, z: i32) -> Column {
        self.shape.column(x, z)
    }

    /// Ground below the surface, before caves, mines and ores carve or replace it.
    fn ground(&self, col: &Column, x: i32, y: i32, z: i32) -> BlockId {
        self.shape.ground(col, x, y, z)
    }

    /// One voxel, from its column. The single source of truth: batch fills reproduce it exactly.
    fn voxel(&self, col: &Column, x: i32, y: i32, z: i32) -> BlockId {
        if y >= SPACE_FLOOR {
            return self.space.block(x, y, z);
        }
        if y >= col.height {
            return self.trees.block_at(&self.shape, x, y, z).unwrap_or(AIR);
        }
        let field = |y: i32| Grid::interp_corners(&self.under.corners(x, y, z), x, y, z);
        let ground = self.ground(col, x, y, z);
        self.under.finish(col, x, y, z, ground, field(y), || field(y - 1), || field(y + 1))
    }

    /// Fill one 16³ chunk densely from a column window (the batch path; equal to [`Self::voxel`]).
    fn fill_chunk(&self, cols: &ColumnWindow, cx: i32, cy: i32, cz: i32, tree_blocks: &[(i32, i32, i32, BlockId)]) -> ChunkData {
        let n = CHUNK_SIZE as i32;
        let (x0, y0, z0) = (cx * n, cy * n, cz * n);
        if y0 >= SPACE_FLOOR && !self.space.may_touch(x0, y0, z0, n) {
            return ChunkData::Uniform(AIR);
        }
        let top_tree = cols.max_height + trees::MAX_TREE_HEIGHT;
        if y0 >= top_tree && y0 + n <= SPACE_FLOOR {
            return ChunkData::Uniform(AIR);
        }
        let mut cells = Box::new([AIR; CHUNK_VOLUME]);
        let underground = y0 < cols.max_height;
        let grid = underground.then(|| self.under.grid(x0, y0, z0));
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let col = cols.at(lx, lz);
                let (x, z) = (x0 + lx as i32, z0 + lz as i32);
                for ly in 0..CHUNK_SIZE {
                    let y = y0 + ly as i32;
                    let id = if y >= SPACE_FLOOR {
                        self.space.block(x, y, z)
                    } else if y >= col.height {
                        AIR
                    } else {
                        let g = grid.as_ref().expect("underground chunk has a grid");
                        let ly = ly as i32;
                        let ground = self.ground(col, x, y, z);
                        self.under.finish(col, x, y, z, ground, g.at(lx, ly, lz), || g.at(lx, ly - 1, lz), || g.at(lx, ly + 1, lz))
                    };
                    cells[Chunk::index(lx, ly, lz)] = id;
                }
            }
        }
        for &(x, y, z, id) in tree_blocks {
            let (lx, ly, lz) = (x - x0, y - y0, z - z0);
            if (0..n).contains(&lx) && (0..n).contains(&ly) && (0..n).contains(&lz) {
                let i = Chunk::index(lx as usize, ly as usize, lz as usize);
                // Trees fill open air above the ground only (never a cave under a neighbour's hill).
                if cells[i] == AIR && y >= cols.at(lx as usize, lz as usize).height {
                    cells[i] = id;
                }
            }
        }
        ChunkData::from_cells(cells)
    }
}

/// The 16×16 columns of one chunk column (computed once per batch).
struct ColumnWindow {
    cols: Vec<Column>,
    max_height: i32,
}

impl ColumnWindow {
    fn at(&self, lx: usize, lz: usize) -> &Column {
        &self.cols[lx + lz * CHUNK_SIZE]
    }
}

impl TerrainGenerator for Terrain {
    fn seed(&self) -> i64 {
        self.seed
    }

    fn mass(&self) -> Arc<dyn crate::gravity::MassOracle> {
        self.cosmos.clone()
    }

    fn kind(&self) -> &'static str {
        "diffusion"
    }

    fn height(&self, wx: i32, wz: i32) -> i32 {
        self.shape.height(wx, wz)
    }

    fn surface_at(&self, wx: i32, wz: i32) -> BlockId {
        self.column(wx, wz).surface
    }

    fn deep(&self) -> BlockId {
        self.m.rock[0]
    }

    fn block_at(&self, wx: i32, wy: i32, wz: i32, _height: i32) -> BlockId {
        let col = self.column(wx, wz);
        self.voxel(&col, wx, wy, wz)
    }

    fn voxel_at(&self, wx: i32, wy: i32, wz: i32) -> BlockId {
        let col = self.column(wx, wz);
        self.voxel(&col, wx, wy, wz)
    }

    fn lod_block_at(&self, wx: i32, wy: i32, wz: i32) -> BlockId {
        let col = self.column(wx, wz);
        if wy >= col.height { AIR } else { self.ground(&col, wx, wy, wz) }
    }

    fn lod_column(&self, wx: i32, wz: i32, ys: &[i32], out: &mut [BlockId]) {
        let col = self.column(wx, wz);
        for (o, &wy) in out.iter_mut().zip(ys) {
            *o = if wy >= col.height { AIR } else { self.ground(&col, wx, wy, wz) };
        }
    }

    fn generate(&self, cx: i32, cy: i32, cz: i32) -> ChunkData {
        let key = ColumnKey { face: Face::PosY, a: cx, b: cz };
        let (mut chunks, _) = self.generate_column(key, cy..=cy);
        chunks.pop().expect("one chunk").1
    }

    fn generate_column(
        &self,
        key: ColumnKey,
        cy: std::ops::RangeInclusive<i32>,
    ) -> (Vec<(i32, ChunkData)>, ColumnHeights) {
        if key.face != Face::PosY {
            return generation::generate_column_default(self, key, cy);
        }
        let (cx, cz) = (key.a, key.b);
        let n = CHUNK_SIZE as i32;
        let (x0, z0) = (cx * n, cz * n);
        let mut heights = [0i32; CHUNK_SIZE * CHUNK_SIZE];
        let cols: Vec<Column> = (0..CHUNK_SIZE * CHUNK_SIZE)
            .map(|i| self.column(x0 + (i % CHUNK_SIZE) as i32, z0 + (i / CHUNK_SIZE) as i32))
            .collect();
        for (h, c) in heights.iter_mut().zip(&cols) {
            *h = c.height;
        }
        if cy.is_empty() {
            return (Vec::new(), heights);
        }
        let max_height = cols.iter().map(|c| c.height).max().unwrap_or(0);
        let window = ColumnWindow { cols, max_height };
        let tree_blocks = self.trees.blocks_in(&self.shape, x0, z0, n);
        let chunks = cy.map(|cyy| (cyy, self.fill_chunk(&window, cx, cyy, cz, &tree_blocks))).collect();
        (chunks, heights)
    }
}

#[cfg(test)]
mod tests;
