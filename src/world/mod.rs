//! The world module owns every chunk and answers questions about the voxels in
//! them: what block is at a position, whether a box collides with terrain, and how
//! to draw the visible surface.
pub mod chunk;
pub mod generation;
pub mod voxel;

use std::collections::HashMap;

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
pub struct World {
    chunks: HashMap<(i32, i32), Chunk>,
}

impl World {
    /// Generate the default world (a grid of [`SineHills`] chunks).
    pub fn generate() -> Self {
        Self::with_generator(&SineHills::default())
    }

    /// Generate a square grid of chunks using any [`TerrainGenerator`].
    pub fn with_generator<G: TerrainGenerator>(generator: &G) -> Self {
        let mut chunks = HashMap::new();
        for cx in 0..WORLD_CHUNKS_X {
            for cz in 0..WORLD_CHUNKS_Z {
                chunks.insert((cx, cz), Chunk::new(cx, cz, generator));
            }
        }
        Self { chunks }
    }

    /// Look up the voxel at an absolute world voxel coordinate. Anything outside
    /// generated chunks (or above/below the world) reads as `Air`.
    pub fn voxel_at(&self, x: i32, y: i32, z: i32) -> Voxel {
        if y < 0 || y >= CHUNK_HEIGHT as i32 {
            return Voxel::Air;
        }

        let cx = x.div_euclid(CHUNK_WIDTH as i32);
        let cz = z.div_euclid(CHUNK_DEPTH as i32);
        let lx = x.rem_euclid(CHUNK_WIDTH as i32) as usize;
        let lz = z.rem_euclid(CHUNK_DEPTH as i32) as usize;

        match self.chunks.get(&(cx, cz)) {
            Some(chunk) => chunk.get_local(lx, y as usize, lz),
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

    /// A voxel surrounded by solid neighbours on all six sides is never visible, so
    /// it can be skipped when drawing.
    fn is_enclosed(&self, x: i32, y: i32, z: i32) -> bool {
        self.is_solid(x + 1, y, z)
            && self.is_solid(x - 1, y, z)
            && self.is_solid(x, y + 1, z)
            && self.is_solid(x, y - 1, z)
            && self.is_solid(x, y, z + 1)
            && self.is_solid(x, y, z - 1)
    }
}

impl Render for World {
    /// Draw every solid voxel that has at least one exposed face. Fully buried
    /// voxels are culled so chunk interiors cost nothing to render.
    fn render<D: RaylibDraw3D>(&self, d: &mut D) {
        for chunk in self.chunks.values() {
            for ly in 0..CHUNK_HEIGHT {
                for lz in 0..CHUNK_DEPTH {
                    for lx in 0..CHUNK_WIDTH {
                        let voxel = chunk.get_local(lx, ly, lz);
                        if !voxel.is_solid() {
                            continue;
                        }

                        let wx = chunk.cx * CHUNK_WIDTH as i32 + lx as i32;
                        let wy = ly as i32;
                        let wz = chunk.cz * CHUNK_DEPTH as i32 + lz as i32;

                        if self.is_enclosed(wx, wy, wz) {
                            continue;
                        }

                        // Voxel cube spans [w, w+1], so its centre is offset by half.
                        let center =
                            Vector3::new(wx as f32 + 0.5, wy as f32 + 0.5, wz as f32 + 0.5);
                        d.draw_cube(center, 1.0, 1.0, 1.0, voxel.color());
                    }
                }
            }
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
