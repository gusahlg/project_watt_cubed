//! Terrain generation decoupled from chunk storage so the algorithm can be swapped.
//!
//! [`TerrainGenerator`] is the seam: a pure function of `(seed, coordinate)` that worker threads
//! and multiplayer clients reproduce bit for bit. The shipped generator is InfiniteDiffusion
//! ([`super::terrain`]); the core fallback is [`FlatTerrain`] (used when no worldgen mod is
//! enabled, and by tests that want a plain world).
use std::ops::RangeInclusive;

use super::chunk::{CHUNK_SIZE, CHUNK_VOLUME, Chunk, ChunkData};
use super::layout::{ColumnKey, Sky};
use crate::block::registry::{AIR, BlockId, BlockRegistry};
use crate::coord::{ChunkCoord, Face};

/// Ground height per cell of a 16×16 chunk column — identical to [`TerrainGenerator::height`].
pub type ColumnHeights = [i32; CHUNK_SIZE * CHUNK_SIZE];

/// Sample [`TerrainGenerator::height`] across a chunk column. Used by the
/// default [`TerrainGenerator::generate_column`] (test gens that do not batch).
fn sample_column_heights(g: &(impl TerrainGenerator + ?Sized), cx: i32, cz: i32) -> ColumnHeights {
    let x0 = cx * CHUNK_SIZE as i32;
    let z0 = cz * CHUNK_SIZE as i32;
    let mut heights = [0i32; CHUNK_SIZE * CHUNK_SIZE];
    for lz in 0..CHUNK_SIZE {
        for lx in 0..CHUNK_SIZE {
            heights[lx + lz * CHUNK_SIZE] = g.height(x0 + lx as i32, z0 + lz as i32);
        }
    }
    heights
}

/// Face-local altitudes for a non-PosY column, via [`TerrainGenerator::surface`].
fn sample_face_heights(g: &(impl TerrainGenerator + ?Sized), key: ColumnKey) -> ColumnHeights {
    let mut heights = [0i32; CHUNK_SIZE * CHUNK_SIZE];
    for lv in 0..CHUNK_SIZE {
        for lu in 0..CHUNK_SIZE {
            let (u, v) = key.column_cell_uv(lu as i32, lv as i32);
            heights[lu + lv * CHUNK_SIZE] = g.surface(key.face, u, v);
        }
    }
    heights
}

/// Default [`TerrainGenerator::generate_column`]. PosY calls [`TerrainGenerator::generate`]
/// per layer and samples [`TerrainGenerator::height`]; any other face does the same
/// through [`TerrainGenerator::surface`] and `key.chunk`. Diffusion keeps its own
/// PosY batch and delegates here for the other faces, so this must not call back
/// into an override with a non-PosY key.
pub(super) fn generate_column_default(
    g: &(impl TerrainGenerator + ?Sized),
    key: ColumnKey,
    range: RangeInclusive<i32>,
) -> (Vec<(i32, ChunkData)>, ColumnHeights) {
    let heights = if key.face == Face::PosY {
        sample_column_heights(g, key.a, key.b)
    } else {
        sample_face_heights(g, key)
    };
    if range.is_empty() {
        return (Vec::new(), heights);
    }
    let chunks = if key.face == Face::PosY {
        range.map(|alt| (alt, g.generate(key.a, alt, key.b))).collect()
    } else {
        range
            .map(|alt| {
                let c = key.chunk(alt);
                (alt, g.generate(c.x, c.y, c.z))
            })
            .collect()
    };
    (chunks, heights)
}

/// Which generator a world is built with. Folded into the content fingerprint.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum WorldgenKind {
    /// The core fallback: a flat world.
    Flat,
    /// InfiniteDiffusion: mountains and valleys, caves and mines, space.
    #[default]
    Diffusion,
}

impl WorldgenKind {
    pub fn id(self) -> &'static str {
        match self {
            Self::Flat => "flat",
            Self::Diffusion => "diffusion",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "flat" => Some(Self::Flat),
            "diffusion" => Some(Self::Diffusion),
            _ => None,
        }
    }

    pub fn wire(self) -> u8 {
        match self {
            Self::Flat => 0,
            Self::Diffusion => 1,
        }
    }

    pub fn from_wire(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Flat),
            1 => Some(Self::Diffusion),
            _ => None,
        }
    }
}

/// Produces terrain for absolute world coordinates.
pub trait TerrainGenerator: Send + Sync {
    /// World seed this generator was built from.
    fn seed(&self) -> i64 {
        0
    }
    /// Stable id folded into the content fingerprint (`flat`, `diffusion`, …).
    fn kind(&self) -> &'static str {
        "flat"
    }
    /// Topmost non-ground cell in this column (the PosY surface).
    fn height(&self, wx: i32, wz: i32) -> i32;

    /// Which way skylight falls in `c`. Default is everywhere +Y.
    fn sky(&self, _c: ChunkCoord) -> Sky {
        Sky::Axis(Face::PosY)
    }

    /// Altitude of the first open cell above the ground along `face` at face-local `(u, v)`.
    /// PosY is [`height`](Self::height). Any other face is open (`i32::MIN`) unless overridden.
    fn surface(&self, face: Face, u: i32, v: i32) -> i32 {
        if face == Face::PosY { self.height(u, v) } else { i32::MIN }
    }

    /// Ground height for every cell of the 16×16 chunk column at `(cx, cz)`.
    /// Default walks [`height`](Self::height); Diffusion fills the rectangle
    /// from the field in one tile load.
    fn heights_16(&self, cx: i32, cz: i32) -> ColumnHeights {
        sample_column_heights(self, cx, cz)
    }

    /// Surface block (biome-dependent).
    fn surface_at(&self, wx: i32, wz: i32) -> BlockId;

    /// Deep block (underground / far-LOD sides).
    fn deep(&self) -> BlockId;

    /// Block at world coordinate; default is surface/deep/air; Terrain overrides.
    fn block_at(&self, wx: i32, wy: i32, wz: i32, height: i32) -> BlockId {
        if wy >= height {
            AIR
        } else if wy >= height - 1 {
            self.surface_at(wx, wz)
        } else {
            self.deep()
        }
    }

    /// Block at a world cell, sampling the column once (not `height` then `block_at`).
    fn voxel_at(&self, wx: i32, wy: i32, wz: i32) -> BlockId {
        self.block_at(wx, wy, wz, self.height(wx, wz))
    }

    /// The block a coarse far-LOD tile shows at a cell. Distinct from
    /// [`block_at`](Self::block_at) because a tile samples at a `2^k`-metre stride
    /// where sub-cell detail would alias to noise: only the ground silhouette
    /// matters (no caves, trees or ores).
    fn lod_block_at(&self, wx: i32, wy: i32, wz: i32) -> BlockId {
        self.block_at(wx, wy, wz, self.height(wx, wz))
    }

    /// Fill a vertical run of LOD cells; default per-cell, generators batch.
    fn lod_column(&self, wx: i32, wz: i32, ys: &[i32], out: &mut [BlockId]) {
        for (o, &wy) in out.iter_mut().zip(ys) {
            *o = self.lod_block_at(wx, wy, wz);
        }
    }

    /// Generate chunk; default dense then collapse; generators shortcut.
    fn generate(&self, cx: i32, cy: i32, cz: i32) -> ChunkData {
        let y0 = cy * CHUNK_SIZE as i32;
        let mut cells = Box::new([AIR; CHUNK_VOLUME]);
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let wx = cx * CHUNK_SIZE as i32 + lx as i32;
                let wz = cz * CHUNK_SIZE as i32 + lz as i32;
                let height = self.height(wx, wz);
                for ly in 0..CHUNK_SIZE {
                    cells[Chunk::index(lx, ly, lz)] = self.block_at(wx, y0 + ly as i32, wz, height);
                }
            }
        }
        ChunkData::from_cells(cells)
    }

    /// Generate a run of chunks along `key`'s normal, plus 256 altitudes.
    ///
    /// `range` is local altitude chunk indices. Returned [`ChunkData`] is in
    /// world cell order ([`Chunk::index`](super::chunk::Chunk::index)). Heights
    /// are altitudes along `key.face`, indexed `lu + lv * 16` in face-local
    /// order, and are produced even when `range` is empty. PosY keeps the
    /// per-chunk [`generate`](Self::generate) loop (Diffusion overrides that
    /// case with its batch). Other faces use the same loop through `key.chunk`.
    fn generate_column(
        &self,
        key: ColumnKey,
        range: RangeInclusive<i32>,
    ) -> (Vec<(i32, ChunkData)>, ColumnHeights) {
        generate_column_default(self, key, range)
    }
}

/// The flat world: grass on soil on banded rock at a fixed height. The core fallback generator.
pub struct FlatTerrain {
    seed: i64,
    grass: BlockId,
    soil: BlockId,
    rock: BlockId,
}

/// Height of the flat world's ground (first air cell).
pub const FLAT_HEIGHT: i32 = 12;

impl FlatTerrain {
    pub fn new(registry: &mut BlockRegistry, seed: i64) -> Self {
        let m = super::terrain::Materials::intern(registry);
        Self { seed, grass: m.grass, soil: m.soil, rock: m.rock[0] }
    }
}

impl TerrainGenerator for FlatTerrain {
    fn seed(&self) -> i64 {
        self.seed
    }

    fn height(&self, _wx: i32, _wz: i32) -> i32 {
        FLAT_HEIGHT
    }

    fn surface_at(&self, _wx: i32, _wz: i32) -> BlockId {
        self.grass
    }

    fn deep(&self) -> BlockId {
        self.rock
    }

    fn block_at(&self, _wx: i32, wy: i32, _wz: i32, height: i32) -> BlockId {
        if wy >= height {
            AIR
        } else if wy == height - 1 {
            self.grass
        } else if wy >= height - 3 {
            self.soil
        } else {
            self.rock
        }
    }

    fn generate(&self, cx: i32, cy: i32, cz: i32) -> ChunkData {
        let y0 = cy * CHUNK_SIZE as i32;
        if y0 >= FLAT_HEIGHT {
            return ChunkData::Uniform(AIR);
        }
        if y0 + (CHUNK_SIZE as i32) <= FLAT_HEIGHT - 3 {
            return ChunkData::Uniform(self.rock);
        }
        let mut cells = Box::new([AIR; CHUNK_VOLUME]);
        for ly in 0..CHUNK_SIZE {
            let id = self.block_at(0, y0 + ly as i32, 0, FLAT_HEIGHT);
            for lz in 0..CHUNK_SIZE {
                for lx in 0..CHUNK_SIZE {
                    cells[Chunk::index(lx, ly, lz)] = id;
                }
            }
        }
        let _ = (cx, cz);
        ChunkData::from_cells(cells)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::reversed_empty_ranges)] // `1..=0`: the height field with no chunk layers
    fn flat_generate_matches_block_at_and_heights() {
        let mut reg = BlockRegistry::with_builtins();
        let g = FlatTerrain::new(&mut reg, 3);
        for cy in 2..5 {
            let data = g.generate(1, cy, -2);
            for ly in 0..CHUNK_SIZE {
                let y = cy * CHUNK_SIZE as i32 + ly as i32;
                assert_eq!(data.get(Chunk::index(3, ly, 7)), g.block_at(19, y, -25, FLAT_HEIGHT));
            }
        }
        let key = ColumnKey { face: Face::PosY, a: 0, b: 0 };
        let (_, heights) = g.generate_column(key, 1..=0);
        assert!(heights.iter().all(|&h| h == FLAT_HEIGHT));
        assert_eq!(g.sky(ChunkCoord::new(0, 0, 0)), Sky::Axis(Face::PosY));
        assert_eq!(g.surface(Face::PosY, 3, 4), FLAT_HEIGHT);
        assert_eq!(g.surface(Face::PosX, 3, 4), i32::MIN);
    }

    #[test]
    fn worldgen_kind_ids_and_wire_round_trip() {
        for k in [WorldgenKind::Flat, WorldgenKind::Diffusion] {
            assert_eq!(WorldgenKind::from_id(k.id()), Some(k));
            assert_eq!(WorldgenKind::from_wire(k.wire()), Some(k));
        }
        assert_eq!(WorldgenKind::default(), WorldgenKind::Diffusion);
    }
}
