//! chunk.rs stores a fixed-size column of voxels. It owns no generation logic of
//! its own — it asks a [`TerrainGenerator`] to fill itself.
use crate::world::generation::TerrainGenerator;
use crate::world::voxel::Voxel;

pub const CHUNK_WIDTH: usize = 16; // along world X
pub const CHUNK_HEIGHT: usize = 64; // along world Y
pub const CHUNK_DEPTH: usize = 16; // along world Z

/// A region of the world holding its own flat array of voxels.
pub struct Chunk {
    /// Chunk coordinate on the X axis (world X = cx * CHUNK_WIDTH + local x).
    pub cx: i32,
    /// Chunk coordinate on the Z axis (world Z = cz * CHUNK_DEPTH + local z).
    pub cz: i32,
    voxels: Vec<Voxel>,
}

impl Chunk {
    /// Create a chunk at the given chunk coordinate and fill it using `generator`.
    pub fn new<G: TerrainGenerator>(cx: i32, cz: i32, generator: &G) -> Self {
        let mut chunk = Self {
            cx,
            cz,
            voxels: vec![Voxel::Air; CHUNK_WIDTH * CHUNK_HEIGHT * CHUNK_DEPTH],
        };
        chunk.generate(generator);
        chunk
    }

    fn index(x: usize, y: usize, z: usize) -> usize {
        x + z * CHUNK_WIDTH + y * CHUNK_WIDTH * CHUNK_DEPTH
    }

    /// Read a voxel using chunk-local coordinates.
    pub fn get_local(&self, x: usize, y: usize, z: usize) -> Voxel {
        self.voxels[Self::index(x, y, z)]
    }

    fn set_local(&mut self, x: usize, y: usize, z: usize, v: Voxel) {
        self.voxels[Self::index(x, y, z)] = v;
    }

    fn generate<G: TerrainGenerator>(&mut self, generator: &G) {
        for lx in 0..CHUNK_WIDTH {
            for lz in 0..CHUNK_DEPTH {
                let wx = self.cx * CHUNK_WIDTH as i32 + lx as i32;
                let wz = self.cz * CHUNK_DEPTH as i32 + lz as i32;
                let height = generator.height(wx, wz);

                for ly in 0..CHUNK_HEIGHT {
                    let v = generator.voxel_at(wx, ly as i32, wz, height);
                    self.set_local(lx, ly, lz, v);
                }
            }
        }
    }
}
