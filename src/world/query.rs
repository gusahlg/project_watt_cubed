//! Read-only queries: block lookups, solidity, box collision, surface height,
//! coordinate mapping, and the registry/seed accessors. Code motion only:
//! these are `World` methods; the struct itself lives in `mod.rs`.

use crate::block::registry::{AIR, BlockId, BlockRegistry};
use crate::coord::BlockCoord;
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
        let (chunk, local) = BlockCoord::new(x, y, z).split();
        match self.chunks.get(&chunk) {
            Some(loaded) => loaded.chunk.get_local(local.lx(), local.ly(), local.lz()),
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
        // TODO Stage 1: dedupe with Aabb::voxel_cells — this re-derives the same
        // cell range but needs it grouped-by-chunk (one map probe per chunk),
        // which voxel_cells' flat per-cell iterator doesn't provide; routing
        // through it would change the iteration order/perf, so defer.
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
                    let Some(loaded) = self.chunks.get(&Coord::new(cx, cy, cz)) else {
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
                        for z in zs.clone() {
                            for y in ys.clone() {
                                let (_, local) = BlockCoord::new(x, y, z).split();
                                let id = loaded.chunk.get_local(local.lx(), local.ly(), local.lz());
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
        // Single-sourced through the split iso in `coord.rs`, so the negative
        // `div_euclid` semantics live in exactly one place.
        BlockCoord::new(x, y, z).split().0
    }
}
