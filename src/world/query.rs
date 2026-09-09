//! Read-only block, surface, collision, and acoustic queries over loaded chunks.

use std::ops::Range;
use std::sync::Arc;

use glam::{IVec3, UVec3};

use crate::audio::acoustics::{AcousticWindow, Cell};
use crate::block::registry::{AIR, BlockId, BlockRegistry};
use crate::coord::BlockCoord;
use crate::math::{Aabb, block_coord, block_coord_end};
use voxel_engine::Color;

use super::chunk::CHUNK_SIZE;
use super::{Coord, World};

/// Clip an inclusive world interval to a chunk and return its local cell range.
/// The interval must intersect the chunk. Saturation keeps endpoint chunks valid
/// even at the i32 world limits.
fn local_range(min: i32, max: i32, chunk: i32) -> Range<usize> {
    let origin = chunk * CHUNK_SIZE as i32;
    min.saturating_sub(origin).max(0) as usize
        ..max.saturating_sub(origin).min(CHUNK_SIZE as i32 - 1) as usize + 1
}

impl World {
    /// The seed this world was generated from.
    pub fn seed(&self) -> i64 {
        self.generator.seed()
    }

    /// Worldgen algorithm id (`classic`, `diffusion`, …).
    pub fn worldgen_kind(&self) -> &'static str {
        self.generator.kind()
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

    /// Ground height for every cell of a 16×16 chunk column.
    pub fn heights_16(&self, cx: i32, cz: i32) -> super::generation::ColumnHeights {
        self.generator.heights_16(cx, cz)
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

    /// Whether the block at a world voxel coordinate contains material. Rendering
    /// and mining use this query; movement uses [`is_obstacle`](Self::is_obstacle)
    /// so liquids remain passable.
    pub fn is_solid(&self, x: i32, y: i32, z: i32) -> bool {
        self.registry.is_solid(self.block_at(x, y, z))
    }

    /// Whether the block at a world voxel obstructs movement and clearance rays: a
    /// solid that is not a passable liquid. Mining uses [`is_solid`](Self::is_solid)
    /// so liquids can still be broken.
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
                    let xs = local_range(x0, x1, cx);
                    let ys = local_range(y0, y1, cy);
                    let zs = local_range(z0, z1, cz);
                    // X is contiguous in chunk storage.
                    for y in ys {
                        for z in zs.clone() {
                            for x in xs.clone() {
                                let id = loaded.chunk.get_local(x, y, z);
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
            let absorption = hot.absorption(id);
            if hot.solid(id) && absorption > 0 {
                Cell::Solid { absorption }
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
                    let xs = local_range(origin.x, hi.x, cx);
                    let ys = local_range(origin.y, hi.y, cy);
                    let zs = local_range(origin.z, hi.z, cz);
                    let dx = (cx * s + xs.start as i32 - origin.x) as usize;
                    let mut row = [AIR; CHUNK_SIZE];
                    for z in zs {
                        let dz = (cz * s + z as i32 - origin.z) as usize;
                        for y in ys.clone() {
                            let dy = (cy * s + y as i32 - origin.y) as usize;
                            let start = (dz * dim + dy) * dim + dx;
                            let out = &mut cells[start..start + xs.len()];
                            if let Some(cell) = uniform {
                                out.fill(cell);
                            } else {
                                // One storage dispatch per row, shared with mesh
                                // snapshot capture, instead of one per voxel.
                                let row = &mut row[..xs.len()];
                                loaded.chunk.copy_row_from(xs.start, y, z, row);
                                for (cell, &id) in out.iter_mut().zip(row.iter()) {
                                    *cell = occlude(id);
                                }
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
    use crate::render_config::RenderConfig;
    use crate::world::chunk::{CHUNK_VOLUME, Chunk, ChunkData};
    use crate::world::{Loaded, MeshState};
    use std::collections::BTreeMap;
    use voxel_engine::DVec3;

    fn insert_chunk(world: &mut World, chunk: Chunk) {
        world.chunks.insert(
            Coord::new(chunk.cx, chunk.cy, chunk.cz),
            Loaded {
                chunk: Arc::new(chunk),
                state: MeshState::needs_mesh(),
                rev: 0,
                connectivity: None,
                visible: true,
                light: None,
                has_blocklight: false,
            },
        );
    }

    fn query_world() -> World {
        let mut world = World::with_config_lazy(73, RenderConfig::default());
        let stone = world.registry.id_by_name("Stone").unwrap();
        let water = world.registry.id_by_name("Water").unwrap();
        let ice = world.registry.id_by_name("Ice").unwrap();
        for (coord, id) in [
            (Coord::new(-1, -1, -1), stone),
            (Coord::new(0, -1, -1), AIR),
            (Coord::new(-1, 0, -1), water),
        ] {
            insert_chunk(
                &mut world,
                Chunk::from_uniform(coord.x, coord.y, coord.z, id),
            );
        }
        let ids = [AIR, stone, water, ice];
        let mut cells = Box::new([AIR; CHUNK_VOLUME]);
        for (i, id) in cells.iter_mut().enumerate() {
            let (x, y, z) = Chunk::local_of(i);
            *id = ids[(x + 2 * y + 3 * z) % ids.len()];
        }
        insert_chunk(&mut world, Chunk::from_cells(0, 0, -1, cells.clone()));
        insert_chunk(
            &mut world,
            Chunk::from_data(-1, 0, 0, ChunkData::Dense(cells)),
        );
        world
    }

    #[test]
    fn acoustic_rows_match_per_cell_queries_for_every_storage_shape() {
        let world = query_world();
        for (center, radius) in [
            (IVec3::new(-1, 0, -1), 17),
            (IVec3::new(15, 15, -16), 1),
            (IVec3::new(-16, 0, 15), 2),
            (IVec3::new(1, 2, -3), 0),
        ] {
            let window = world.capture_acoustic_window(center, radius);
            let r = radius as i32;
            for z in center.z - r..=center.z + r {
                for y in center.y - r..=center.y + r {
                    for x in center.x - r..=center.x + r {
                        let expected = if !world.chunks.contains_key(&World::chunk_of(x, y, z)) {
                            Cell::Unloaded
                        } else if world.is_obstacle(x, y, z) {
                            Cell::Solid {
                                absorption: world.registry.absorption(world.block_at(x, y, z)),
                            }
                        } else {
                            Cell::Open
                        };
                        let pos = IVec3::new(x, y, z);
                        assert_eq!(window.cell(pos), expected, "{pos:?} in {center:?}, r={r}");
                    }
                }
            }
            assert_eq!(window.cell(center + IVec3::X * (r + 1)), Cell::Unloaded);
        }
    }

    #[test]
    fn local_collision_walk_matches_per_cell_queries_across_chunk_edges() {
        let world = query_world();
        let positions = [-16.0, -0.1, 0.0, 15.9, 16.0];
        for x in positions {
            for y in positions {
                for z in positions {
                    for half in [DVec3::splat(0.5), DVec3::new(1.0, 0.75, 2.0)] {
                        let aabb = Aabb::new(DVec3::new(x, y, z), half);
                        let expected = aabb
                            .voxel_cells()
                            .any(|(x, y, z)| world.is_obstacle(x, y, z));
                        assert_eq!(
                            world.collides(&aabb),
                            expected,
                            "at ({x}, {y}, {z}), {half:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn collision_local_ranges_include_world_border_cells() {
        let mut world = World::with_config_lazy(73, RenderConfig::default());
        let stone = world.registry.id_by_name("Stone").unwrap();
        // Most-negative clamp cell (local 0) and most-positive `block_coord_end`
        // cell (local 15): the reachable endpoints, where saturating clip matters.
        for x in [block_coord(f64::NEG_INFINITY), block_coord_end(f64::INFINITY)] {
            let (coord, local) = BlockCoord::new(x, 0, 0).split();
            let mut chunk = Chunk::from_uniform(coord.x, coord.y, coord.z, AIR);
            chunk.set_local(local.lx(), local.ly(), local.lz(), stone);
            insert_chunk(&mut world, chunk);
            let center = DVec3::new(f64::from(x) + 0.5, 0.5, 0.5);
            assert!(world.collides(&Aabb::new(center, DVec3::splat(0.25))));
            assert!(!world.collides(&Aabb::new(center + DVec3::Y, DVec3::splat(0.25))));
        }
    }

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
