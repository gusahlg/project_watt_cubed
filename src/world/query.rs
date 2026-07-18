//! Read-only queries: block lookups, solidity, box collision, surface height,
//! coordinate mapping, and the registry/seed accessors. Code motion only:
//! these are `World` methods; the struct itself lives in `mod.rs`.

use std::sync::Arc;

use glam::{IVec3, UVec3};

use crate::audio::acoustics::{AcousticWindow, Cell};
use crate::block::registry::{AIR, BlockId, BlockRegistry};
use crate::coord::BlockCoord;
use crate::math::{Aabb, block_coord, block_coord_end};
use voxel_engine::Color;

use super::chunk::CHUNK_SIZE;
use super::generation::TerrainGenerator;
use super::{Coord, World};

impl World {
    /// The seed this world was generated from.
    pub fn seed(&self) -> i64 {
        self.generator.seed
    }

    /// Incremented when blocks are edited.
    pub fn edit_generation(&self) -> u64 {
        self.edit_generation
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

    /// Sea level, so spawn logic can tell dry land from seabed/ocean columns.
    pub fn sea_level(&self) -> i32 {
        self.generator.sea_level()
    }

    /// Highest solid block's Y in column (x, z) from loaded chunks, or None if empty.
    pub fn top_solid(&self, x: i32, z: i32) -> Option<i32> {
        let s = CHUNK_SIZE as i32;
        let (cx, cz) = (x.div_euclid(s), z.div_euclid(s));
        let (lx, lz) = (x.rem_euclid(s) as usize, z.rem_euclid(s) as usize);

        // Scan loaded chunks top-down (Y unbounded).
        let mut cys: Vec<i32> = self
            .chunks
            .keys()
            .filter(|c| c.x == cx && c.z == cz)
            .map(|c| c.y)
            .collect();
        cys.sort_unstable_by(|a, b| b.cmp(a));

        for cy in cys {
            let loaded = &self.chunks[&Coord::new(cx, cy, cz)];
            if let Some(id) = loaded.chunk.uniform() {
                if self.registry.is_solid(id) {
                    return Some(cy * s + s - 1);
                }
                continue; // uniform air
            }
            for ly in (0..s).rev() {
                let id = loaded.chunk.get_local(lx, ly as usize, lz);
                if self.registry.is_solid(id) {
                    return Some(cy * s + ly);
                }
            }
        }
        None
    }

    /// Calls paint for the top solid block of each column in the x/z rectangle.
    pub fn for_surface_columns(
        &self,
        x0: i32,
        z0: i32,
        x1: i32,
        z1: i32,
        mut paint: impl FnMut(i32, i32, i32, Color),
    ) {
        // Bucket the footprint's vertical stacks in one pass over `chunks` and
        // one sort, keyed by dense grid cell — no hashing.
        let s = CHUNK_SIZE as i32;
        let (cx0, cx1) = (x0.div_euclid(s), x1.div_euclid(s));
        let (cz0, cz1) = (z0.div_euclid(s), z1.div_euclid(s));
        let grid_w = cx1 - cx0 + 1;
        let mut stacks: Vec<(i32, i32, &super::Loaded)> = Vec::new();
        for (&coord, loaded) in &self.chunks {
            if (cx0..=cx1).contains(&coord.x) && (cz0..=cz1).contains(&coord.z) {
                let cell = (coord.z - cz0) * grid_w + (coord.x - cx0);
                stacks.push((cell, coord.y, loaded));
            }
        }
        // Group by cell, top chunk first within each stack.
        stacks.sort_unstable_by_key(|&(cell, cy, _)| (cell, std::cmp::Reverse(cy)));

        for stack in stacks.chunk_by(|a, b| a.0 == b.0) {
            let (ccx, ccz) = (cx0 + stack[0].0 % grid_w, cz0 + stack[0].0 / grid_w);
            let xs = x0.max(ccx * s)..=x1.min((ccx + 1) * s - 1);
            let zs = z0.max(ccz * s)..=z1.min((ccz + 1) * s - 1);
            for x in xs {
                let lx = x.rem_euclid(s) as usize;
                for z in zs.clone() {
                    let lz = z.rem_euclid(s) as usize;
                    for &(_, cy, loaded) in stack {
                        let top = match loaded.chunk.uniform() {
                            Some(id) if self.registry.is_solid(id) => Some((cy * s + s - 1, id)),
                            Some(_) => None,
                            None => (0..s).rev().find_map(|ly| {
                                let id = loaded.chunk.get_local(lx, ly as usize, lz);
                                self.registry.is_solid(id).then_some((cy * s + ly, id))
                            }),
                        };
                        if let Some((top_y, id)) = top {
                            paint(x, z, top_y, self.registry.color(id));
                            break; // topmost hit
                        }
                    }
                }
            }
        }
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

    /// Whether the block at a world voxel obstructs movement and the aim ray — a
    /// solid that is not a passable liquid. The predicate collision and interaction
    /// share, so water stops neither.
    pub fn is_obstacle(&self, x: i32, y: i32, z: i32) -> bool {
        self.registry.is_obstacle(self.block_at(x, y, z))
    }

    /// The buoyancy of the liquid at a world voxel coordinate, or `0` if the cell
    /// is air or a non-liquid solid. The movement code samples this at the swimmer's
    /// feet and eye to decide whether — and how strongly — to swim.
    pub fn buoyancy_at(&self, x: i32, y: i32, z: i32) -> u8 {
        self.registry.buoyancy(self.block_at(x, y, z))
    }

    /// Collision test: does the given box overlap any solid, non-liquid voxel?
    /// Liquids are `solid` (so they mesh) but passable, so the player swims through
    /// them; only genuine obstacles block movement here.
    ///
    /// Cells are visited grouped by owning chunk — one map probe per chunk the
    /// box touches (1–8 for anything player-sized) instead of one per cell,
    /// and a uniform chunk answers for all its cells with one solidity load.
    pub fn collides(&self, aabb: &Aabb) -> bool {
        // Same cell range as `Aabb::voxel_cells` (shared clamped floor and
        // exclusive-upper-edge helpers, so a box at the world border stays in
        // i32 and exact face contact does not visit the touching next voxel),
        // but grouped by owning chunk — voxel_cells' flat per-cell iterator
        // can't provide the one-map-probe-per-chunk order this hot path needs.
        let (min, max) = (aabb.min(), aabb.max());
        let (x0, x1) = (block_coord(min.x), block_coord_end(max.x));
        let (y0, y1) = (block_coord(min.y), block_coord_end(max.y));
        let (z0, z1) = (block_coord(min.z), block_coord_end(max.z));

        let s = CHUNK_SIZE as i32;
        for cx in x0.div_euclid(s)..=x1.div_euclid(s) {
            for cy in y0.div_euclid(s)..=y1.div_euclid(s) {
                for cz in z0.div_euclid(s)..=z1.div_euclid(s) {
                    let Some(loaded) = self.chunks.get(&Coord::new(cx, cy, cz)) else {
                        continue; // unloaded chunks read as air
                    };
                    // Uniform chunks: one lookup answers every cell in the box.
                    if let Some(id) = loaded.chunk.uniform() {
                        if self.registry.is_obstacle(id) {
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
                                if self.registry.is_obstacle(id) {
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

    /// An immutable acoustic snapshot of the cube `[center − r, center + r]³`, for
    /// the audio kernel's occlusion DDA. `radius` is clamped so the window edge stays
    /// within [`MAX_WINDOW_DIM`](crate::audio::acoustics::MAX_WINDOW_DIM): `dim =
    /// 2r + 1 ≤ 96` ⇒ `r ≤ 47`.
    ///
    /// Cell mapping (one [`HotTables`](crate::block::registry::HotTables) snapshot at
    /// entry — never per-voxel registry calls): a whole missing chunk reads
    /// [`Unloaded`](Cell::Unloaded); otherwise a solid, non-passable block is
    /// [`Solid`](Cell::Solid) with its derived absorption and everything else
    /// (air, water) is [`Open`](Cell::Open). Passable liquids derive absorption `0`,
    /// so `solid && absorption > 0` exactly selects occluding walls; water/air stay
    /// Open for occlusion and the listener's medium is decided elsewhere.
    ///
    /// Cell layout is z-outer, y-mid, x-inner with `origin = center − r`:
    /// `index = (dz · dim + dy) · dim + dx`, `d* = world − origin` — matching
    /// [`AcousticWindow::cell`](crate::audio::acoustics::AcousticWindow::cell).
    pub fn capture_acoustic_window(&self, center: IVec3, radius: u32) -> Arc<AcousticWindow> {
        let r = radius.min(47) as i32;
        let dim = (2 * r + 1) as usize;
        let origin = center - IVec3::splat(r);
        // Missing chunks stay Unloaded by leaving their cells untouched.
        let mut cells = vec![Cell::Unloaded; dim * dim * dim].into_boxed_slice();

        let hot = self.registry.hot_tables();
        let occlude = |id: BlockId| {
            let i = id.0 as usize;
            if hot.solid(crate::block::registry::BlockId(i as u16)) && hot.absorption[i] > 0 {
                Cell::Solid { absorption: hot.absorption[i] }
            } else {
                Cell::Open
            }
        };

        let s = CHUNK_SIZE as i32;
        let hi = origin + IVec3::splat(dim as i32 - 1); // inclusive far corner
        for cx in origin.x.div_euclid(s)..=hi.x.div_euclid(s) {
            for cy in origin.y.div_euclid(s)..=hi.y.div_euclid(s) {
                for cz in origin.z.div_euclid(s)..=hi.z.div_euclid(s) {
                    let Some(loaded) = self.chunks.get(&Coord::new(cx, cy, cz)) else {
                        continue; // whole chunk unloaded → cells remain Unloaded
                    };
                    let uniform = loaded.chunk.uniform().map(&occlude);
                    let xs = origin.x.max(cx * s)..=hi.x.min((cx + 1) * s - 1);
                    let ys = origin.y.max(cy * s)..=hi.y.min((cy + 1) * s - 1);
                    let zs = origin.z.max(cz * s)..=hi.z.min((cz + 1) * s - 1);
                    for wz in zs {
                        let dz = (wz - origin.z) as usize;
                        for wy in ys.clone() {
                            let dy = (wy - origin.y) as usize;
                            for wx in xs.clone() {
                                let dx = (wx - origin.x) as usize;
                                let cell = uniform.unwrap_or_else(|| {
                                    let (_, l) = BlockCoord::new(wx, wy, wz).split();
                                    occlude(loaded.chunk.get_local(l.lx(), l.ly(), l.lz()))
                                });
                                cells[(dz * dim + dy) * dim + dx] = cell;
                            }
                        }
                    }
                }
            }
        }

        let win = AcousticWindow::new(origin, UVec3::splat(dim as u32), cells)
            .expect("dim ≤ MAX_WINDOW_DIM and size·product == cells.len() by construction");
        Arc::new(win)
    }

    /// The chunk coordinate an absolute world position falls in.
    pub(in crate::world) fn chunk_of(x: i32, y: i32, z: i32) -> Coord {
        // Single-sourced through the split iso in `coord.rs`, so the negative
        // `div_euclid` semantics live in exactly one place.
        BlockCoord::new(x, y, z).split().0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn indexed_surface_walk_matches_point_queries_across_chunk_edges() {
        let world = World::new(73);
        let (x0, z0, x1, z1) = (-18, -19, 21, 20);
        let mut actual = BTreeMap::new();
        world.for_surface_columns(x0, z0, x1, z1, |x, z, y, color| {
            actual.insert((x, z), (y, [color.r, color.g, color.b, color.a]));
        });

        for x in x0..=x1 {
            for z in z0..=z1 {
                let expected = world.top_solid(x, z).map(|y| {
                    let color = world.registry.color(world.block_at(x, y, z));
                    (y, [color.r, color.g, color.b, color.a])
                });
                assert_eq!(actual.get(&(x, z)).copied(), expected, "column ({x}, {z})");
            }
        }
    }
}
