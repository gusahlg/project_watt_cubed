//! Read-only queries: block lookups, solidity, box collision, surface height,
//! coordinate mapping, and the registry/seed accessors. Code motion only:
//! these are `World` methods; the struct itself lives in `mod.rs`.

use crate::block::registry::{AIR, BlockId, BlockRegistry};
use crate::math::{Aabb, block_coord};

use super::chunk::CHUNK_SIZE;
use super::generation::TerrainGenerator;
use super::{Coord, World};

impl World {
    /// The seed this world was generated from.
    pub fn seed(&self) -> i64 {
        self.generator.seed
    }

    /// The block palette, for resolving ids to names, properties, and the hot
    /// solidity/colour arrays.
    pub fn registry(&self) -> &BlockRegistry {
        &self.registry
    }

    /// Mutable access to the palette, so crafting and mods can register new blocks.
    pub fn registry_mut(&mut self) -> &mut BlockRegistry {
        &mut self.registry
    }

    /// Surface height of a column, for placing the player on spawn.
    pub fn surface_y(&self, x: i32, z: i32) -> i32 {
        self.generator.height(x, z)
    }

    /// Look up the block id at an absolute world voxel coordinate. Anything
    /// outside the loaded region reads as [`AIR`] — Y is unbounded, so there
    /// is no world floor or ceiling anymore.
    pub fn block_at(&self, x: i32, y: i32, z: i32) -> BlockId {
        match self.chunks.get(&Self::chunk_of(x, y, z)) {
            Some(loaded) => {
                let s = CHUNK_SIZE as i32;
                loaded.chunk.get_local(
                    x.rem_euclid(s) as usize,
                    y.rem_euclid(s) as usize,
                    z.rem_euclid(s) as usize,
                )
            }
            None => AIR,
        }
    }

    /// Whether the block at a world voxel coordinate is solid. The per-frame
    /// collision hot path: a fast-hashed chunk lookup plus one registry array load.
    pub fn is_solid(&self, x: i32, y: i32, z: i32) -> bool {
        self.registry.is_solid(self.block_at(x, y, z))
    }

    /// Collision test: does the given box overlap any solid voxel?
    ///
    /// Cells are visited grouped by owning chunk — one map probe per chunk the
    /// box touches (1–8 for anything player-sized) instead of one per cell,
    /// and a uniform chunk answers for all its cells with one solidity load.
    pub fn collides(&self, aabb: &Aabb) -> bool {
        // Same cell range as `Aabb::voxel_cells`: block_coord(min)..=block_coord(max)
        // (the shared clamped floor, so a box at the world border stays in i32).
        let (min, max) = (aabb.min(), aabb.max());
        let (x0, x1) = (block_coord(min.x), block_coord(max.x));
        let (y0, y1) = (block_coord(min.y), block_coord(max.y));
        let (z0, z1) = (block_coord(min.z), block_coord(max.z));

        let s = CHUNK_SIZE as i32;
        for cx in x0.div_euclid(s)..=x1.div_euclid(s) {
            for cy in y0.div_euclid(s)..=y1.div_euclid(s) {
                for cz in z0.div_euclid(s)..=z1.div_euclid(s) {
                    let Some(loaded) = self.chunks.get(&(cx, cy, cz)) else {
                        continue; // unloaded chunks read as air
                    };
                    // Uniform chunks: one lookup answers every cell in the box.
                    if let Some(id) = loaded.chunk.uniform() {
                        if self.registry.is_solid(id) {
                            return true;
                        }
                        continue;
                    }
                    let xs = x0.max(cx * s)..=x1.min((cx + 1) * s - 1);
                    let ys = y0.max(cy * s)..=y1.min((cy + 1) * s - 1);
                    let zs = z0.max(cz * s)..=z1.min((cz + 1) * s - 1);
                    for x in xs {
                        let lx = x.rem_euclid(s) as usize;
                        for z in zs.clone() {
                            let lz = z.rem_euclid(s) as usize;
                            for y in ys.clone() {
                                let ly = y.rem_euclid(s) as usize;
                                let id = loaded.chunk.get_local(lx, ly, lz);
                                if self.registry.is_solid(id) {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
        }
        false
    }

    /// The chunk coordinate an absolute world position falls in.
    pub(in crate::world) fn chunk_of(x: i32, y: i32, z: i32) -> Coord {
        let s = CHUNK_SIZE as i32;
        (x.div_euclid(s), y.div_euclid(s), z.div_euclid(s))
    }
}
