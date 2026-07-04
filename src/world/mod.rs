//! The world owns the block palette and an *infinite*, streamed field of chunks:
//! it keeps the chunks near the player loaded (generated and meshed), discards
//! distant ones, and answers what block is at a position, whether a box collides
//! with terrain, and how to draw the visible surface.
//!
//! Two design choices serve the "optimisation ahead of readability" mandate:
//! chunks live in a `HashMap` behind a tiny multiplicative hasher (the default
//! SipHash is far too slow for a per-frame collision hot path), and player edits
//! live in a compact overlay so a chunk can be regenerated identically after it
//! streams out and back in.
pub mod chunk;
pub mod generation;
pub mod mesh;

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

use voxel_engine::{Color, Engine, Frame3D, MeshData, MeshHandle, Vec3};

use crate::block::registry::{AIR, BlockId, BlockRegistry};
use crate::math::Aabb;
use crate::render::Render;
use chunk::{CHUNK_DEPTH, CHUNK_HEIGHT, CHUNK_WIDTH, Chunk};
use generation::{SineHills, TerrainGenerator};

/// Default number of chunk rings meshed and drawn around the player.
const DEFAULT_VIEW_RADIUS: i32 = 6;
/// The range a runtime render-distance change is clamped to.
const VIEW_RADIUS_RANGE: std::ops::RangeInclusive<i32> = 3..=10;
/// One extra ring of *data* (not meshed) so edge chunks can cull faces against
/// their neighbours without re-meshing when those neighbours later load.
const DATA_MARGIN: i32 = 1;
/// How far past the view radius chunks survive before they are freed, so
/// walking back and forth across the boundary doesn't thrash.
const UNLOAD_MARGIN: i32 = 3;
/// How many fresh chunk meshes to build per stream so entering a world grows the
/// terrain in over a few frames instead of freezing on one.
const MESH_BUDGET: usize = 6;
/// How many *dirty* (edited) chunks may remesh per frame. Processed nearest
/// first, so a locally broken block still vanishes the same frame while a
/// multiplayer join snapshot flood spreads over a few frames instead of one hitch.
const DIRTY_BUDGET: usize = 8;
/// The seed a default (`generate`) world uses when none is chosen.
pub const DEFAULT_SEED: i64 = 1;

/// A chunk coordinate: `(cx, cz)` where world X = `cx * CHUNK_WIDTH + local x`.
type Coord = (i32, i32);

/// Fast identity-ish hasher for the small integer keys the chunk/edit maps use.
/// The keys are already well-distributed grid coordinates, so a couple of
/// multiplies beat a general-purpose hash by a wide margin on the hot path.
#[derive(Default)]
struct FastHasher(u64);

impl Hasher for FastHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x0100_0000_01b3);
        }
    }
    fn write_i32(&mut self, i: i32) {
        self.0 = (self.0 ^ i as u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    fn write_usize(&mut self, i: usize) {
        self.0 = (self.0 ^ i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;
type FastSet<K> = HashSet<K, BuildHasherDefault<FastHasher>>;

/// A loaded chunk: its voxel data plus the GPU mesh built from it. `meshed`
/// distinguishes a chunk that only has data (a margin chunk, or one awaiting its
/// turn in the mesh budget) from one ready to draw; a meshed all-air chunk has
/// `meshed == true` with `mesh == None`.
struct Loaded {
    chunk: Chunk,
    mesh: Option<MeshHandle>,
    meshed: bool,
}

/// The streamed world: the block palette, the terrain generator, the currently
/// loaded chunks, and the overlay of player edits that outlive chunk unloads.
pub struct World {
    /// The block palette every voxel indexes into. Built once and read-only; the
    /// hot solidity/colour arrays it owns are what meshing and collision read.
    registry: BlockRegistry,
    generator: SineHills,
    chunks: FastMap<Coord, Loaded>,
    /// Player edits, grouped by chunk so regenerating a chunk can replay just its
    /// own. Inner key is the flat voxel index within the chunk.
    edits: FastMap<Coord, FastMap<usize, BlockId>>,
    /// Chunks whose mesh is stale (an edit changed them) and must rebuild,
    /// nearest first, ahead of any fresh meshing.
    dirty: FastSet<Coord>,
    /// The chunk the player was last centred on, so streaming only reacts to
    /// crossing a chunk boundary. Invalidated to force a full pass.
    center: Coord,
    /// Runtime render distance in chunk rings (clamped to [`VIEW_RADIUS_RANGE`]).
    view_radius: i32,
    /// Whether a fresh-mesh scan might still find unmeshed chunks. Cleared when
    /// a scan finishes with nothing left, set again by anything that could
    /// create work (centre/radius change, edits, new chunk data).
    pending_fresh: bool,
    /// Set when the render distance shrank; the next stream frees meshes
    /// beyond the new radius instead of leaving them drawn until movement.
    radius_shrunk: bool,
    /// Reusable CPU-side mesh scratch; `upload_mesh` copies out of it, so one
    /// buffer serves every chunk build without per-chunk allocations.
    scratch: MeshData,
    /// Snapshots of the registry's hot per-block arrays as plain slices for the
    /// mesher. Refreshed when the palette grows (it is append-only).
    solid_table: Vec<bool>,
    color_table: Vec<Color>,
}

impl World {
    /// A fresh world for `seed`, with the region around the origin pre-generated
    /// (data only — no GPU) so spawning and headless queries work before the first
    /// [`stream`](Self::stream).
    pub fn new(seed: i64) -> Self {
        let registry = BlockRegistry::with_builtins();
        let generator = SineHills::new(&registry, 20.0, seed);
        let mut world = Self {
            registry,
            generator,
            chunks: FastMap::default(),
            edits: FastMap::default(),
            dirty: FastSet::default(),
            center: (i32::MIN, i32::MIN),
            view_radius: DEFAULT_VIEW_RADIUS,
            pending_fresh: true,
            radius_shrunk: false,
            scratch: MeshData::default(),
            solid_table: Vec::new(),
            color_table: Vec::new(),
        };
        world.ensure_region_data((0, 0));
        world
    }

    /// The default world (seed [`DEFAULT_SEED`]).
    pub fn generate() -> Self {
        Self::new(DEFAULT_SEED)
    }

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

    /// Current render distance in chunk rings.
    pub fn view_radius(&self) -> i32 {
        self.view_radius
    }

    /// Change the render distance (clamped to 3..=10). Marks streaming dirty so
    /// the next [`stream`](Self::stream) unloads past the new radius or resumes
    /// meshing out to it.
    pub fn set_view_radius(&mut self, radius: i32) {
        let radius = radius.clamp(*VIEW_RADIUS_RANGE.start(), *VIEW_RADIUS_RANGE.end());
        if radius != self.view_radius {
            let shrunk = radius < self.view_radius;
            self.view_radius = radius;
            // Invalidate the centre so the next stream reruns the full
            // unload/ensure/scan pass even though the player hasn't moved.
            self.center = (i32::MIN, i32::MIN);
            self.pending_fresh = true;
            // On shrink, meshes between the new radius and the (also shrunk)
            // unload ring would otherwise stay drawn until the player moves;
            // flag them so the next stream frees them immediately.
            self.radius_shrunk = shrunk;
        }
    }

    /// Bring the world up to date around `center` (the player's position): load and
    /// mesh nearby chunks, free distant ones. Requires the engine (it uploads
    /// meshes), so it runs from the game update, not from headless logic.
    ///
    /// Steady-state cost is near zero: the unload/generate pass only runs when
    /// the player crosses a chunk boundary (or the radius changed), and the
    /// fresh-mesh scan is skipped once a scan has found nothing left to build.
    pub fn stream(&mut self, center: Vec3, eng: &mut Engine) {
        let center_chunk = (
            (center.x.floor() as i32).div_euclid(CHUNK_WIDTH as i32),
            (center.z.floor() as i32).div_euclid(CHUNK_DEPTH as i32),
        );
        if center_chunk != self.center {
            self.center = center_chunk;
            self.unload_far(center_chunk, eng);
            self.ensure_region_data(center_chunk);
            self.pending_fresh = true;
        }
        if self.radius_shrunk {
            self.radius_shrunk = false;
            // Meshes between the new view radius and the unload ring survive
            // unload_far's hysteresis; free them now (data stays loaded).
            for (&(cx, cz), loaded) in self.chunks.iter_mut() {
                let ring = (cx - center_chunk.0).abs().max((cz - center_chunk.1).abs());
                if ring > self.view_radius && loaded.meshed {
                    loaded.meshed = false;
                    if let Some(handle) = loaded.mesh.take() {
                        eng.free_mesh(handle);
                    }
                }
            }
        }
        self.build_meshes(center_chunk, eng);
    }

    /// Draw the meshed chunks. All per-voxel work happened when each chunk was
    /// built; a frame is one `draw_mesh` per chunk (the engine frustum-culls
    /// each against its AABB internally).
    pub fn render(&self, f: &mut Frame3D) {
        for loaded in self.chunks.values() {
            if let Some(handle) = loaded.mesh {
                f.draw_mesh(handle);
            }
        }
    }

    /// Free every chunk's GPU mesh and clear the meshed flags — used when
    /// leaving a world. The voxel data stays; a later [`stream`](Self::stream)
    /// would rebuild the meshes from scratch.
    pub fn free_meshes(&mut self, eng: &mut Engine) {
        for loaded in self.chunks.values_mut() {
            if let Some(handle) = loaded.mesh.take() {
                eng.free_mesh(handle);
            }
            loaded.meshed = false;
        }
        self.dirty.clear();
        self.center = (i32::MIN, i32::MIN);
        self.pending_fresh = true;
    }

    /// Ensure every chunk within the data radius of `center` exists (voxel data
    /// only). Cheap and GPU-free, so it also seeds headless queries.
    fn ensure_region_data(&mut self, center: Coord) {
        let radius = self.view_radius + DATA_MARGIN;
        for cx in (center.0 - radius)..=(center.0 + radius) {
            for cz in (center.1 - radius)..=(center.1 + radius) {
                self.ensure_data((cx, cz));
            }
        }
    }

    /// Generate a chunk's data if it isn't loaded, replaying any saved edits on it.
    fn ensure_data(&mut self, coord: Coord) {
        if self.chunks.contains_key(&coord) {
            return;
        }
        let mut chunk = Chunk::new(coord.0, coord.1, &self.generator);
        if let Some(edits) = self.edits.get(&coord) {
            for (&index, &id) in edits {
                chunk.set_index(index, id);
            }
        }
        self.chunks.insert(
            coord,
            Loaded {
                chunk,
                mesh: None,
                meshed: false,
            },
        );
        // New data means new mesh work next scan.
        self.pending_fresh = true;
    }

    /// Free chunks past the unload radius, releasing their GPU meshes.
    fn unload_far(&mut self, center: Coord, eng: &mut Engine) {
        let radius = self.view_radius + UNLOAD_MARGIN;
        // Collect-then-remove instead of `retain`: freeing needs `&mut eng`,
        // which can't be borrowed inside a retain closure over `self.chunks`.
        let far: Vec<Coord> = self
            .chunks
            .keys()
            .copied()
            .filter(|&(cx, cz)| (cx - center.0).abs() > radius || (cz - center.1).abs() > radius)
            .collect();
        for coord in far {
            if let Some(loaded) = self.chunks.remove(&coord)
                && let Some(handle) = loaded.mesh
            {
                eng.free_mesh(handle);
            }
        }
    }

    /// Remesh edited chunks (nearest first, budgeted), then build up to
    /// [`MESH_BUDGET`] fresh chunks in the view radius, nearest first. A chunk
    /// only meshes once its four orthogonal neighbours have data, so border
    /// faces are culled correctly the first time.
    fn build_meshes(&mut self, center: Coord, eng: &mut Engine) {
        if !self.dirty.is_empty() {
            let mut dirty: Vec<Coord> = self.dirty.iter().copied().collect();
            dirty.sort_by_key(|&(cx, cz)| (cx - center.0).abs().max((cz - center.1).abs()));
            for coord in dirty.into_iter().take(DIRTY_BUDGET) {
                self.dirty.remove(&coord);
                // No neighbour-data gate here: an edited chunk must remesh even
                // when a far neighbour has no data (the mesher reads missing
                // neighbours as air, exactly like the original world lookup).
                // Gating would leave a stale mesh with a hole at the border in
                // margin chunks whose outer neighbour never loads.
                if self.chunks.contains_key(&coord) {
                    self.mesh_chunk(coord, eng);
                }
            }
        }

        // Fresh chunks, nearest first, capped by the frame budget. Skipped
        // entirely once a scan came up empty, until something re-flags work.
        if !self.pending_fresh {
            return;
        }
        let mut pending: Vec<Coord> = self
            .chunks
            .iter()
            .filter(|(_, loaded)| !loaded.meshed)
            .map(|(&coord, _)| coord)
            .filter(|&(cx, cz)| {
                (cx - center.0).abs() <= self.view_radius
                    && (cz - center.1).abs() <= self.view_radius
            })
            .filter(|&coord| self.neighbours_have_data(coord))
            .collect();
        pending.sort_by_key(|&(cx, cz)| (cx - center.0).abs().max((cz - center.1).abs()));

        if pending.len() <= MESH_BUDGET {
            // This pass finishes the backlog; don't scan again until new work appears.
            self.pending_fresh = false;
        }
        for coord in pending.into_iter().take(MESH_BUDGET) {
            self.mesh_chunk(coord, eng);
        }
    }

    /// Whether the four orthogonal neighbours of a chunk have voxel data loaded.
    fn neighbours_have_data(&self, coord: Coord) -> bool {
        let (cx, cz) = coord;
        self.chunks.contains_key(&(cx - 1, cz))
            && self.chunks.contains_key(&(cx + 1, cz))
            && self.chunks.contains_key(&(cx, cz - 1))
            && self.chunks.contains_key(&(cx, cz + 1))
    }

    /// Build (or rebuild) one chunk's GPU mesh and mark it drawable, freeing any
    /// previous mesh. An all-air chunk ends up `meshed` with no handle.
    fn mesh_chunk(&mut self, coord: Coord, eng: &mut Engine) {
        self.refresh_tables();
        // Move the scratch out so the build can borrow `self.chunks` shared
        // (for cross-chunk neighbour culling) while filling it.
        let mut scratch = std::mem::take(&mut self.scratch);
        {
            let loaded = &self.chunks[&coord];
            let neighbours = mesh::Neighbours {
                neg_x: self.chunks.get(&(coord.0 - 1, coord.1)).map(|l| &l.chunk),
                pos_x: self.chunks.get(&(coord.0 + 1, coord.1)).map(|l| &l.chunk),
                neg_z: self.chunks.get(&(coord.0, coord.1 - 1)).map(|l| &l.chunk),
                pos_z: self.chunks.get(&(coord.0, coord.1 + 1)).map(|l| &l.chunk),
            };
            mesh::build_chunk_mesh(
                &loaded.chunk,
                &neighbours,
                &self.solid_table,
                &self.color_table,
                &mut scratch,
            );
        }
        let handle = eng.upload_mesh(&scratch); // None when the chunk is all air
        self.scratch = scratch;
        if let Some(loaded) = self.chunks.get_mut(&coord) {
            if let Some(old) = loaded.mesh.take() {
                eng.free_mesh(old);
            }
            loaded.mesh = handle;
            loaded.meshed = true;
        }
    }

    /// Re-snapshot the registry's hot arrays if blocks were registered since the
    /// last build. The palette is append-only, so a length check suffices.
    fn refresh_tables(&mut self) {
        let count = self.registry.block_count();
        if self.solid_table.len() != count {
            self.solid_table = (0..count)
                .map(|i| self.registry.is_solid(BlockId(i as u16)))
                .collect();
            self.color_table = (0..count)
                .map(|i| self.registry.color(BlockId(i as u16)))
                .collect();
        }
    }

    /// Look up the block id at an absolute world voxel coordinate. Anything outside
    /// the loaded region (or above/below the world) reads as [`AIR`].
    pub fn block_at(&self, x: i32, y: i32, z: i32) -> BlockId {
        if y < 0 || y >= CHUNK_HEIGHT as i32 {
            return AIR;
        }
        let coord = Self::chunk_of(x, z);
        match self.chunks.get(&coord) {
            Some(loaded) => {
                let lx = x.rem_euclid(CHUNK_WIDTH as i32) as usize;
                let lz = z.rem_euclid(CHUNK_DEPTH as i32) as usize;
                loaded.chunk.get_local(lx, y as usize, lz)
            }
            None => AIR,
        }
    }

    /// Whether the block at a world voxel coordinate is solid. The per-frame
    /// collision hot path: a fast-hashed chunk lookup plus one registry array load.
    pub fn is_solid(&self, x: i32, y: i32, z: i32) -> bool {
        self.registry.is_solid(self.block_at(x, y, z))
    }

    /// Replace the block at a world coordinate, recording the change in the edit
    /// overlay (so it survives streaming and can be saved) and marking the affected
    /// chunk — and any neighbour across a shared face — for remeshing. Returns the
    /// block that was there.
    pub fn set_block(&mut self, x: i32, y: i32, z: i32, id: BlockId) -> BlockId {
        if y < 0 || y >= CHUNK_HEIGHT as i32 {
            return AIR;
        }
        let coord = Self::chunk_of(x, z);
        let lx = x.rem_euclid(CHUNK_WIDTH as i32) as usize;
        let lz = z.rem_euclid(CHUNK_DEPTH as i32) as usize;
        let ly = y as usize;
        let index = Chunk::index(lx, ly, lz);

        let previous = self.block_at(x, y, z);
        self.edits.entry(coord).or_default().insert(index, id);

        if let Some(loaded) = self.chunks.get_mut(&coord) {
            loaded.chunk.set_index(index, id);
            loaded.meshed = false;
            self.dirty.insert(coord);
            self.pending_fresh = true;
            // A block on a chunk edge also changes the neighbour's exposed faces.
            if lx == 0 {
                self.mark_dirty((coord.0 - 1, coord.1));
            }
            if lx == CHUNK_WIDTH - 1 {
                self.mark_dirty((coord.0 + 1, coord.1));
            }
            if lz == 0 {
                self.mark_dirty((coord.0, coord.1 - 1));
            }
            if lz == CHUNK_DEPTH - 1 {
                self.mark_dirty((coord.0, coord.1 + 1));
            }
        }
        previous
    }

    /// Mark a loaded chunk stale so the next stream remeshes it.
    fn mark_dirty(&mut self, coord: Coord) {
        if let Some(loaded) = self.chunks.get_mut(&coord) {
            loaded.meshed = false;
            self.dirty.insert(coord);
            // In case the dirty pass drops it (missing neighbour data), the
            // fresh scan must be able to pick it back up later.
            self.pending_fresh = true;
        }
    }

    /// Collision test: does the given box overlap any solid voxel?
    ///
    /// Cells are visited grouped by owning chunk — one map probe per chunk the
    /// box touches (1–4 for anything player-sized) instead of one per cell.
    pub fn collides(&self, aabb: &Aabb) -> bool {
        // Same cell range as `Aabb::voxel_cells`: floor(min)..=floor(max).
        let (min, max) = (aabb.min(), aabb.max());
        let (x0, x1) = (min.x.floor() as i32, max.x.floor() as i32);
        let (y0, y1) = (min.y.floor() as i32, max.y.floor() as i32);
        let (z0, z1) = (min.z.floor() as i32, max.z.floor() as i32);

        // Out-of-world layers read as air, exactly like `is_solid`.
        let (y0, y1) = (y0.max(0), y1.min(CHUNK_HEIGHT as i32 - 1));
        if y0 > y1 {
            return false;
        }

        let (cx0, cz0) = Self::chunk_of(x0, z0);
        let (cx1, cz1) = Self::chunk_of(x1, z1);
        for cx in cx0..=cx1 {
            for cz in cz0..=cz1 {
                let Some(loaded) = self.chunks.get(&(cx, cz)) else {
                    continue; // unloaded chunks read as air
                };
                let xs = x0.max(cx * CHUNK_WIDTH as i32)..=x1.min((cx + 1) * CHUNK_WIDTH as i32 - 1);
                let zs = z0.max(cz * CHUNK_DEPTH as i32)..=z1.min((cz + 1) * CHUNK_DEPTH as i32 - 1);
                for x in xs {
                    let lx = x.rem_euclid(CHUNK_WIDTH as i32) as usize;
                    for z in zs.clone() {
                        let lz = z.rem_euclid(CHUNK_DEPTH as i32) as usize;
                        for y in y0..=y1 {
                            let id = loaded.chunk.get_local(lx, y as usize, lz);
                            if self.registry.is_solid(id) {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        false
    }

    /// Every recorded edit as `((x, y, z), block)`, for saving.
    pub fn edits(&self) -> impl Iterator<Item = ((i32, i32, i32), BlockId)> + '_ {
        self.edits.iter().flat_map(|(&(cx, cz), cells)| {
            cells.iter().map(move |(&index, &id)| {
                let (lx, ly, lz) = Chunk::local_of(index);
                let x = cx * CHUNK_WIDTH as i32 + lx as i32;
                let z = cz * CHUNK_DEPTH as i32 + lz as i32;
                ((x, ly as i32, z), id)
            })
        })
    }

    /// The chunk coordinate an absolute world `(x, z)` falls in.
    fn chunk_of(x: i32, z: i32) -> Coord {
        (
            x.div_euclid(CHUNK_WIDTH as i32),
            z.div_euclid(CHUNK_DEPTH as i32),
        )
    }
}

impl Render for World {
    fn render(&self, f: &mut Frame3D) {
        World::render(self, f);
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
        let in_ground = Aabb::new(Vec3::new(8.5, 0.5, 8.5), Vec3::new(0.3, 0.3, 0.3));
        let in_sky = Aabb::new(
            Vec3::new(8.5, CHUNK_HEIGHT as f32 - 0.5, 8.5),
            Vec3::new(0.3, 0.3, 0.3),
        );
        assert!(world.collides(&in_ground));
        assert!(!world.collides(&in_sky));
    }

    #[test]
    fn collision_grouped_lookup_matches_per_cell_path() {
        // Boxes straddling chunk boundaries exercise the multi-chunk grouping;
        // the grouped fast path must agree with a per-cell `is_solid` sweep.
        let world = World::generate();
        for center in [
            Vec3::new(15.9, 18.0, 15.9), // corner of four chunks
            Vec3::new(0.1, 21.5, 8.0),   // one X boundary
            Vec3::new(-3.2, 19.0, -16.4),
            Vec3::new(4.0, -1.0, 4.0), // below the world
        ] {
            let aabb = Aabb::new(center, Vec3::new(0.4, 0.9, 0.4));
            let reference = aabb.voxel_cells().any(|(x, y, z)| world.is_solid(x, y, z));
            assert_eq!(world.collides(&aabb), reference, "at {center:?}");
        }
    }

    #[test]
    fn column_is_layered_grass_dirt_stone() {
        let world = World::generate();
        let reg = world.registry();
        let (grass, dirt, stone) = (
            reg.id_by_name("Grass").unwrap(),
            reg.id_by_name("Dirt").unwrap(),
            reg.id_by_name("Stone").unwrap(),
        );

        let (x, z) = (8, 8);
        let h = (0..CHUNK_HEIGHT as i32)
            .rev()
            .find(|&y| world.is_solid(x, y, z))
            .expect("the column has solid ground");

        assert_eq!(world.block_at(x, h + 1, z), AIR);
        assert_eq!(world.block_at(x, h, z), grass);
        assert_eq!(world.block_at(x, h - 1, z), dirt);
        assert_eq!(world.block_at(x, h - 3, z), stone);
    }

    #[test]
    fn edits_persist_across_unload() {
        let mut world = World::generate();
        // Break the surface block far enough out that it will stream away, then be
        // regenerated when we ask again — the edit must replay.
        let (x, z) = (8, 8);
        let h = (0..CHUNK_HEIGHT as i32)
            .rev()
            .find(|&y| world.is_solid(x, y, z))
            .unwrap();
        world.set_block(x, h, z, AIR);
        assert_eq!(world.block_at(x, h, z), AIR);

        // Drop the chunk and regenerate its data; the recorded edit should return.
        world.chunks.clear();
        world.ensure_data(World::chunk_of(x, z));
        assert_eq!(world.block_at(x, h, z), AIR, "edit survived reload");
    }

    #[test]
    fn distinct_seeds_differ() {
        let a = World::new(1);
        let b = World::new(9_999);
        let ha: Vec<i32> = (0..16).map(|x| a.surface_y(x, 0)).collect();
        let hb: Vec<i32> = (0..16).map(|x| b.surface_y(x, 0)).collect();
        assert_ne!(ha, hb, "different seeds should sculpt different terrain");
    }

    #[test]
    fn view_radius_clamps_and_flags_streaming() {
        let mut world = World::generate();
        assert_eq!(world.view_radius(), DEFAULT_VIEW_RADIUS);
        world.set_view_radius(99);
        assert_eq!(world.view_radius(), 10);
        world.set_view_radius(1);
        assert_eq!(world.view_radius(), 3);
        // The change must force the next stream to rescan.
        assert!(world.pending_fresh);
        assert_eq!(world.center, (i32::MIN, i32::MIN));
    }
}
