//! Terrain generation, decoupled from chunk storage so the algorithm can be
//! swapped without touching how voxels are stored or drawn.
//!
//! A generator works in [`BlockId`]s, not raw element compositions: it resolves the
//! handful of blocks it places against the [`BlockRegistry`] once, up front, so
//! filling a cell stays a cheap id copy with no per-voxel allocation.
use super::chunk::CHUNK_HEIGHT;
use crate::block::registry::{AIR, BlockId, BlockRegistry};

/// Produces terrain for absolute world coordinates.
///
/// An implementor provides a surface [`height`](Self::height) and the three blocks
/// it layers (surface, subsoil, deep); the default [`block_at`](Self::block_at)
/// turns those into stacked terrain. A richer generator (caves, ores, biomes) can
/// override `block_at` instead.
pub trait TerrainGenerator {
    /// Surface height for a world column: the number of solid layers stacked from
    /// `y = 0` upward.
    fn height(&self, wx: i32, wz: i32) -> i32;

    /// The block placed on the surface (the topmost solid layer).
    fn surface(&self) -> BlockId;
    /// The block placed just below the surface.
    fn subsoil(&self) -> BlockId;
    /// The block placed deep underground.
    fn deep(&self) -> BlockId;

    /// The block at a world coordinate, given the column's surface `height`.
    ///
    /// Default layering: surface block on top, subsoil just below, deep block
    /// further down, and air above the surface.
    fn block_at(&self, _wx: i32, wy: i32, _wz: i32, height: i32) -> BlockId {
        if wy >= height {
            AIR
        } else if wy >= height - 1 {
            self.surface()
        } else if wy >= height - 3 {
            self.subsoil()
        } else {
            self.deep()
        }
    }
}

/// Rolling hills built from layered sine waves — deterministic and dependency-free.
/// A `seed` shifts the wave phases so each world looks different while staying fully
/// reproducible from that one number.
pub struct SineHills {
    /// Average terrain height that the waves oscillate around.
    pub base: f32,
    /// The world seed this generator was built from (saved and restored verbatim).
    pub seed: i64,
    // Phase offsets derived from the seed, so different seeds sample different terrain.
    offset_x: f32,
    offset_z: f32,
    // Block palette, resolved once against the registry so generation never looks
    // anything up per cell.
    grass: BlockId,
    dirt: BlockId,
    stone: BlockId,
}

impl SineHills {
    /// Build the generator for a seed, resolving its grass/dirt/stone palette against
    /// the registry. Panics if those built-in blocks are missing — they are part of
    /// every [`BlockRegistry::with_builtins`].
    pub fn new(registry: &BlockRegistry, base: f32, seed: i64) -> Self {
        let resolve = |name: &str| {
            registry
                .id_by_name(name)
                .unwrap_or_else(|| panic!("SineHills needs the built-in '{name}' block"))
        };
        // Spread the seed's bits into two large, unrelated phase offsets.
        let offset_x = (seed.wrapping_mul(0x2545F491_4F6CDD1D) as u32 as f32) * 0.000_01;
        let offset_z = (seed.wrapping_mul(0x9E3779B9_7F4A7C15u64 as i64) as u32 as f32) * 0.000_01;
        Self {
            base,
            seed,
            offset_x,
            offset_z,
            grass: resolve("Grass"),
            dirt: resolve("Dirt"),
            stone: resolve("Stone"),
        }
    }
}

impl TerrainGenerator for SineHills {
    fn height(&self, wx: i32, wz: i32) -> i32 {
        let x = wx as f32 + self.offset_x;
        let z = wz as f32 + self.offset_z;

        let h = self.base
            + 6.0 * (x * 0.08).sin() * (z * 0.08).cos()
            + 3.0 * (x * 0.21 + z * 0.13).sin()
            + 2.0 * (z * 0.30).cos();

        h.round().clamp(1.0, (CHUNK_HEIGHT - 1) as f32) as i32
    }

    fn surface(&self) -> BlockId {
        self.grass
    }
    fn subsoil(&self) -> BlockId {
        self.dirt
    }
    fn deep(&self) -> BlockId {
        self.stone
    }
}
