//! chunk.rs stores one 16x16x16 cube of voxels. It owns no generation logic of
//! its own — it asks a [`TerrainGenerator`] to fill itself.
//!
//! A cell is a [`BlockId`] — a compact index into the world's
//! [`BlockRegistry`](crate::block::BlockRegistry), not a block itself. Storage
//! is the memory backbone of the infinite-Y world: most chunks are all air or
//! all stone, so [`ChunkData::Uniform`] stores those as one id (~a dozen bytes)
//! instead of a 4 KiB array. Dense cells are `u8` — safe because the palette is
//! hard-capped at 256 block types ([`BlockRegistry::MAX_BLOCK_TYPES`]).
use crate::block::registry::BlockId;
use crate::world::generation::TerrainGenerator;

/// Chunk edge length along every world axis (chunks are cubes).
pub const CHUNK_SIZE: usize = 16;
/// Cells per chunk.
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

/// A chunk's voxel storage: one id for a uniform chunk, or a dense cube of
/// `u8` cells (block ids — the palette never exceeds 256). `Uniform` is what
/// makes an infinite-Y world affordable: sky and deep rock cost no array.
#[derive(Clone, PartialEq, Debug)]
pub enum ChunkData {
    /// Every cell is this block.
    Uniform(BlockId),
    /// One `u8` block id per cell, flat-indexed by [`Chunk::index`].
    Dense(Box<[u8; CHUNK_VOLUME]>),
}

/// A 16-cube region of the world. `Clone` copies at most the 4 KiB dense array
/// (uniform chunks clone for free) — used to snapshot a chunk for a
/// worker-thread mesh job (see [`pipeline`](super::pipeline)), never on a
/// per-frame hot path.
#[derive(Clone)]
pub struct Chunk {
    /// Chunk coordinate on the X axis (world X = cx * CHUNK_SIZE + local x).
    pub cx: i32,
    /// Chunk coordinate on the Y axis (world Y = cy * CHUNK_SIZE + local y).
    pub cy: i32,
    /// Chunk coordinate on the Z axis (world Z = cz * CHUNK_SIZE + local z).
    pub cz: i32,
    data: ChunkData,
}

impl Chunk {
    /// Create a chunk at the given chunk coordinate and fill it using `generator`.
    pub fn new<G: TerrainGenerator>(cx: i32, cy: i32, cz: i32, generator: &G) -> Self {
        Self {
            cx,
            cy,
            cz,
            data: generator.generate(cx, cy, cz),
        }
    }

    /// Flat array index of a chunk-local coordinate: `x + z*16 + y*256`.
    pub const fn index(x: usize, y: usize, z: usize) -> usize {
        x + z * CHUNK_SIZE + y * CHUNK_SIZE * CHUNK_SIZE
    }

    /// The chunk-local coordinate a flat index maps back to (inverse of [`index`]).
    pub const fn local_of(index: usize) -> (usize, usize, usize) {
        let x = index % CHUNK_SIZE;
        let z = (index / CHUNK_SIZE) % CHUNK_SIZE;
        let y = index / (CHUNK_SIZE * CHUNK_SIZE);
        (x, y, z)
    }

    /// The raw storage, for the mesher's uniform fast paths and flat reads.
    #[inline]
    pub fn data(&self) -> &ChunkData {
        &self.data
    }

    /// The single block filling this chunk, if it is uniform.
    #[inline]
    pub fn uniform(&self) -> Option<BlockId> {
        match self.data {
            ChunkData::Uniform(id) => Some(id),
            ChunkData::Dense(_) => None,
        }
    }

    /// Read a voxel by flat index.
    #[inline]
    pub fn get_index(&self, index: usize) -> BlockId {
        match &self.data {
            ChunkData::Uniform(id) => *id,
            ChunkData::Dense(cells) => BlockId(cells[index] as u16),
        }
    }

    /// Read a voxel using chunk-local coordinates.
    #[inline]
    pub fn get_local(&self, x: usize, y: usize, z: usize) -> BlockId {
        self.get_index(Self::index(x, y, z))
    }

    /// Write a voxel using chunk-local coordinates.
    pub fn set_local(&mut self, x: usize, y: usize, z: usize, v: BlockId) {
        self.set_index(Self::index(x, y, z), v);
    }

    /// Overwrite a voxel by flat index — used to replay saved/broken-block
    /// edits. The first write that differs from a uniform chunk's block
    /// promotes it to dense storage; a matching write stays uniform for free.
    pub fn set_index(&mut self, index: usize, v: BlockId) {
        debug_assert!(v.0 < 256, "dense cells are u8: palette must stay under 256");
        match &mut self.data {
            ChunkData::Uniform(id) => {
                if *id == v {
                    return;
                }
                let mut cells = Box::new([id.0 as u8; CHUNK_VOLUME]);
                cells[index] = v.0 as u8;
                self.data = ChunkData::Dense(cells);
            }
            ChunkData::Dense(cells) => cells[index] = v.0 as u8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::registry::{AIR, BlockRegistry};
    use crate::world::generation::SineHills;

    /// The generator plus the registry-resolved ids its terrain is made of.
    fn hills(seed: i64) -> (SineHills, BlockId, BlockId) {
        let registry = BlockRegistry::with_builtins();
        let stone = registry.id_by_name("Stone").unwrap();
        let dirt = registry.id_by_name("Dirt").unwrap();
        (SineHills::new(&registry, 20.0, seed), stone, dirt)
    }

    #[test]
    fn index_and_local_of_are_inverses() {
        for index in [0, 1, 255, 256, 4095] {
            let (x, y, z) = Chunk::local_of(index);
            assert_eq!(Chunk::index(x, y, z), index);
        }
        assert_eq!(Chunk::index(1, 2, 3), 1 + 3 * 16 + 2 * 256, "x + z*16 + y*256");
    }

    #[test]
    fn get_set_roundtrip_on_dense() {
        let (g, _, dirt) = hills(7);
        let mut chunk = Chunk::new(0, 0, 0, &g); // ground chunk: dense
        assert!(chunk.uniform().is_none(), "surface chunks hold mixed cells");
        chunk.set_local(3, 4, 5, dirt);
        assert_eq!(chunk.get_local(3, 4, 5), dirt);
        chunk.set_local(3, 4, 5, AIR);
        assert_eq!(chunk.get_local(3, 4, 5), AIR);
    }

    #[test]
    fn uniform_promotes_to_dense_on_first_differing_write() {
        let (g, stone, _) = hills(7);
        let mut chunk = Chunk::new(0, -10, 0, &g); // deep rock: uniform stone
        assert_eq!(chunk.uniform(), Some(stone));

        // Writing the same block keeps the cheap representation.
        chunk.set_local(0, 0, 0, stone);
        assert_eq!(chunk.uniform(), Some(stone), "matching write stays uniform");

        // The first differing write promotes, preserving every other cell.
        chunk.set_local(8, 8, 8, AIR);
        assert!(chunk.uniform().is_none(), "differing write goes dense");
        assert_eq!(chunk.get_local(8, 8, 8), AIR);
        assert_eq!(chunk.get_local(0, 0, 0), stone);
        assert_eq!(chunk.get_local(15, 15, 15), stone);
    }

    #[test]
    fn generated_sky_chunk_is_uniform_air() {
        let (g, _, _) = hills(7);
        // Above the terrain (h <= 31) and below the island band (y >= 64).
        let sky = Chunk::new(0, 3, 0, &g);
        assert_eq!(sky.uniform(), Some(AIR), "sky chunk stores one id, not 4 KiB");
        // The uniform representation really is tiny: the enum is pointer-sized
        // plus a tag, nowhere near CHUNK_VOLUME bytes.
        assert!(std::mem::size_of::<ChunkData>() <= 16);
    }

    #[test]
    fn edit_replay_on_uniform_chunk_promotes_correctly() {
        let (g, stone, dirt) = hills(7);
        let mut chunk = Chunk::new(2, 3, 2, &g); // uniform air sky chunk
        assert_eq!(chunk.uniform(), Some(AIR));
        // Replaying an edit overlay (flat index -> id) like the world does.
        for (index, id) in [(Chunk::index(1, 2, 3), stone), (Chunk::index(0, 0, 0), dirt)] {
            chunk.set_index(index, id);
        }
        assert_eq!(chunk.get_local(1, 2, 3), stone);
        assert_eq!(chunk.get_local(0, 0, 0), dirt);
        assert_eq!(chunk.get_local(5, 5, 5), AIR, "untouched cells keep the old fill");
    }
}
