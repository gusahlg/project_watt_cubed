//! Terrain generation, decoupled from chunk storage so the algorithm can be
//! swapped without touching how voxels are stored or drawn.
use super::chunk::CHUNK_HEIGHT;
use super::voxel::Voxel;

/// Produces terrain for absolute world coordinates.
///
/// An implementor only has to provide a surface [`height`](Self::height); the
/// default [`voxel_at`](Self::voxel_at) turns that into layered terrain. A richer
/// generator (caves, ores, biomes) can override `voxel_at` instead.
pub trait TerrainGenerator {
    /// Surface height for a world column: the number of solid layers stacked from
    /// `y = 0` upward.
    fn height(&self, wx: i32, wz: i32) -> i32;

    /// The voxel at a world coordinate, given the column's surface `height`.
    ///
    /// Default layering: grass on the surface, dirt just below, stone deeper, and
    /// air above the surface.
    fn voxel_at(&self, _wx: i32, wy: i32, _wz: i32, height: i32) -> Voxel {
        if wy >= height {
            Voxel::Air
        } else if wy >= height - 1 {
            Voxel::Grass
        } else if wy >= height - 3 {
            Voxel::Dirt
        } else {
            Voxel::Stone
        }
    }
}

/// Rolling hills built from layered sine waves — deterministic and dependency-free.
pub struct SineHills {
    /// Average terrain height that the waves oscillate around.
    pub base: f32,
}

impl Default for SineHills {
    fn default() -> Self {
        Self { base: 20.0 }
    }
}

impl TerrainGenerator for SineHills {
    fn height(&self, wx: i32, wz: i32) -> i32 {
        let x = wx as f32;
        let z = wz as f32;

        let h = self.base
            + 6.0 * (x * 0.08).sin() * (z * 0.08).cos()
            + 3.0 * (x * 0.21 + z * 0.13).sin()
            + 2.0 * (z * 0.30).cos();

        h.round().clamp(1.0, (CHUNK_HEIGHT - 1) as f32) as i32
    }
}
