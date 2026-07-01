//! chunk.rs stores a fixed-size column of voxels. It owns no generation logic of
//! its own — it asks a [`TerrainGenerator`] to fill itself.
//!
//! A cell is a [`BlockId`] — a compact index into the world's
//! [`BlockRegistry`](crate::block::BlockRegistry), not a block itself — so the
//! storage stays 2 bytes per voxel while the blocks they name can be arbitrarily
//! rich.
use crate::block::registry::{AIR, BlockId};
use crate::world::generation::TerrainGenerator;

pub const CHUNK_WIDTH: usize = 16; // along world X
pub const CHUNK_HEIGHT: usize = 64; // along world Y
pub const CHUNK_DEPTH: usize = 16; // along world Z

/// A region of the world holding its own flat array of voxels.
pub struct Chunk {
    /// Chunk coordinate on the X axis (world X = cx * CHUNK_WIDTH + local x).
    pub cx: i32,
    /// Chunk coordinate on the Z axis (world Z = cz * CHUNK_DEPTH + local z).
    pub cz: i32,
    voxels: Vec<BlockId>,
}

impl Chunk {
    /// Create a chunk at the given chunk coordinate and fill it using `generator`.
    pub fn new<G: TerrainGenerator>(cx: i32, cz: i32, generator: &G) -> Self {
        let mut chunk = Self {
            cx,
            cz,
            voxels: vec![AIR; CHUNK_WIDTH * CHUNK_HEIGHT * CHUNK_DEPTH],
        };
        chunk.generate(generator);
        chunk
    }

    /// Flat array index of a chunk-local coordinate.
    pub const fn index(x: usize, y: usize, z: usize) -> usize {
        x + z * CHUNK_WIDTH + y * CHUNK_WIDTH * CHUNK_DEPTH
    }

    /// The chunk-local coordinate a flat index maps back to (inverse of [`index`]).
    pub const fn local_of(index: usize) -> (usize, usize, usize) {
        let x = index % CHUNK_WIDTH;
        let z = (index / CHUNK_WIDTH) % CHUNK_DEPTH;
        let y = index / (CHUNK_WIDTH * CHUNK_DEPTH);
        (x, y, z)
    }

    /// Read a voxel using chunk-local coordinates.
    pub fn get_local(&self, x: usize, y: usize, z: usize) -> BlockId {
        self.voxels[Self::index(x, y, z)]
    }

    /// Write a voxel using chunk-local coordinates.
    pub fn set_local(&mut self, x: usize, y: usize, z: usize, v: BlockId) {
        self.voxels[Self::index(x, y, z)] = v;
    }

    /// Overwrite a voxel by flat index — used to replay saved/broken-block edits.
    pub fn set_index(&mut self, index: usize, v: BlockId) {
        self.voxels[index] = v;
    }

    fn generate<G: TerrainGenerator>(&mut self, generator: &G) {
        for lx in 0..CHUNK_WIDTH {
            for lz in 0..CHUNK_DEPTH {
                let wx = self.cx * CHUNK_WIDTH as i32 + lx as i32;
                let wz = self.cz * CHUNK_DEPTH as i32 + lz as i32;
                let height = generator.height(wx, wz);

                for ly in 0..CHUNK_HEIGHT {
                    let v = generator.block_at(wx, ly as i32, wz, height);
                    self.set_local(lx, ly, lz, v);
                }
            }
        }
    }
}
