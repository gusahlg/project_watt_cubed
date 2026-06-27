//! The world module owns every chunk and answers questions about the voxels in
//! them: what block is at a position, whether a box collides with terrain, and how
//! to draw the visible surface.
pub mod chunk;
pub mod generation;
pub mod mesh;
pub mod voxel;

use raylib::prelude::*;

use crate::math::Aabb;
use crate::render::Render;
use chunk::{CHUNK_DEPTH, CHUNK_HEIGHT, CHUNK_WIDTH, Chunk};
use generation::{SineHills, TerrainGenerator};
use voxel::Voxel;

/// How many chunks to generate along each horizontal axis.
pub const WORLD_CHUNKS_X: i32 = 4;
pub const WORLD_CHUNKS_Z: i32 = 4;

/// The collection of generated chunks plus the queries the rest of the game needs.
///
/// Chunks live in a flat `Vec` indexed directly by chunk coordinate rather than a
/// `HashMap`: the grid is a known, fixed size, so an index is a single multiply
/// instead of a hash. That matters because [`is_solid`](Self::is_solid) is on the
/// per-frame collision hot path and is hammered again while meshing.
pub struct World {
    chunks: Vec<Chunk>,
    /// GPU-resident geometry, built once by [`build_meshes`](Self::build_meshes).
    /// World-space positions are baked into the vertices, so every model is drawn
    /// at the origin. Usually one model per chunk (more only if a chunk overflows
    /// a u16 index buffer).
    meshes: Vec<Model>,
}

impl World {
    /// Generate the default world (a grid of [`SineHills`] chunks). This builds the
    /// voxel data only; call [`build_meshes`](Self::build_meshes) once a window
    /// exists to upload the renderable geometry to the GPU.
    pub fn generate() -> Self {
        Self::with_generator(&SineHills::default())
    }

    /// Generate a square grid of chunks using any [`TerrainGenerator`].
    pub fn with_generator<G: TerrainGenerator>(generator: &G) -> Self {
        let mut chunks = Vec::with_capacity((WORLD_CHUNKS_X * WORLD_CHUNKS_Z) as usize);
        // Pushed in (cx outer, cz inner) order so the index matches `chunk_index`.
        for cx in 0..WORLD_CHUNKS_X {
            for cz in 0..WORLD_CHUNKS_Z {
                chunks.push(Chunk::new(cx, cz, generator));
            }
        }
        Self {
            chunks,
            meshes: Vec::new(),
        }
    }

    /// Build (or rebuild) the GPU meshes for every chunk. Requires a live window,
    /// so it is kept separate from [`generate`](Self::generate) to keep the world's
    /// voxel logic testable without a GPU.
    pub fn build_meshes(&mut self, rl: &mut RaylibHandle, thread: &RaylibThread) {
        let mut meshes = Vec::new();
        // Read chunks by index so `&self` stays free to pass into the mesher for
        // cross-chunk neighbour culling; `self.meshes` is untouched until assigned.
        for i in 0..self.chunks.len() {
            let chunk = &self.chunks[i];
            meshes.append(&mut mesh::build_chunk_models(chunk, self, rl, thread));
        }
        self.meshes = meshes;
    }

    /// Flat index of a chunk coordinate, or `None` if it is outside the grid.
    fn chunk_index(cx: i32, cz: i32) -> Option<usize> {
        if cx < 0 || cx >= WORLD_CHUNKS_X || cz < 0 || cz >= WORLD_CHUNKS_Z {
            None
        } else {
            Some((cx * WORLD_CHUNKS_Z + cz) as usize)
        }
    }

    /// Look up the voxel at an absolute world voxel coordinate. Anything outside
    /// generated chunks (or above/below the world) reads as `Air`.
    pub fn voxel_at(&self, x: i32, y: i32, z: i32) -> Voxel {
        if y < 0 || y >= CHUNK_HEIGHT as i32 {
            return Voxel::Air;
        }

        let cx = x.div_euclid(CHUNK_WIDTH as i32);
        let cz = z.div_euclid(CHUNK_DEPTH as i32);

        match Self::chunk_index(cx, cz) {
            Some(i) => {
                let lx = x.rem_euclid(CHUNK_WIDTH as i32) as usize;
                let lz = z.rem_euclid(CHUNK_DEPTH as i32) as usize;
                self.chunks[i].get_local(lx, y as usize, lz)
            }
            None => Voxel::Air,
        }
    }

    /// Whether the block at a world voxel coordinate is solid.
    pub fn is_solid(&self, x: i32, y: i32, z: i32) -> bool {
        self.voxel_at(x, y, z).is_solid()
    }

    /// Collision test: does the given box overlap any solid voxel?
    pub fn collides(&self, aabb: &Aabb) -> bool {
        aabb.voxel_cells().any(|(x, y, z)| self.is_solid(x, y, z))
    }
}

impl Render for World {
    /// Draw the prebuilt chunk meshes. All per-voxel work happened once at load;
    /// a frame is now just one `draw_model` per chunk mesh.
    fn render<D: RaylibDraw3D>(&self, d: &mut D) {
        for model in &self.meshes {
            d.draw_model(model, Vector3::zero(), 1.0, Color::WHITE);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ground_is_solid_and_sky_is_air() {
        let world = World::generate();
        assert!(world.is_solid(8, 0, 8), "deep ground should be solid");
        assert!(
            !world.is_solid(8, CHUNK_HEIGHT as i32 - 1, 8),
            "top of the world should be air"
        );
    }

    #[test]
    fn collision_agrees_with_solidity() {
        let world = World::generate();
        let in_ground = Aabb::new(Vector3::new(8.5, 0.5, 8.5), Vector3::new(0.3, 0.3, 0.3));
        let in_sky = Aabb::new(
            Vector3::new(8.5, CHUNK_HEIGHT as f32 - 0.5, 8.5),
            Vector3::new(0.3, 0.3, 0.3),
        );
        assert!(world.collides(&in_ground));
        assert!(!world.collides(&in_sky));
    }
}
