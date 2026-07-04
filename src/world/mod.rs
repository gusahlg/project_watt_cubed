//! The world owns the block palette and an *infinite*, streamed field of
//! chunks — infinite along all three axes: 16-cube chunks stack upward through
//! the flying-island band and downward through bottomless stone. It keeps the
//! chunks near the player loaded (generated and meshed), discards distant
//! ones, and answers what block is at a position, whether a box collides with
//! terrain, and how to draw the visible surface.
//!
//! Two design choices serve the "optimisation ahead of readability" mandate:
//! chunks live in a `HashMap` behind a tiny multiplicative hasher (the default
//! SipHash is far too slow for a per-frame collision hot path), and player edits
//! live in a compact overlay so a chunk can be regenerated identically after it
//! streams out and back in. The third is inherited from the storage layer:
//! most of the 3D streaming volume is uniform air or stone
//! ([`ChunkData::Uniform`](chunk::ChunkData)), which costs no voxel array and
//! — for air — no mesh job at all.
//!
//! Heavy chunk work is off the render thread: generation and fresh meshing run
//! on a small worker pool (see [`pipeline`]), while *edited* chunks keep a
//! synchronous remesh so a broken block never lags a frame.
pub mod chunk;
pub mod generation;
pub mod mesh;
pub mod pipeline;

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;

use voxel_engine::{Engine, Frame3D, MeshData, MeshHandle, Vec3};

use crate::block::registry::{AIR, BlockId, BlockRegistry};
use crate::math::Aabb;
use crate::render::Render;
use chunk::{CHUNK_SIZE, Chunk};
use generation::{SineHills, TerrainGenerator};

/// Default number of chunk rings meshed and drawn around the player.
const DEFAULT_VIEW_RADIUS: i32 = 6;
/// The range a runtime render-distance change is clamped to.
const VIEW_RADIUS_RANGE: std::ops::RangeInclusive<i32> = 3..=10;
/// One extra shell of *data* (not meshed) in all three axes so edge chunks can
/// cull faces against their neighbours without re-meshing when those
/// neighbours later load.
const DATA_MARGIN: i32 = 1;
/// How far past the view radius chunks survive horizontally before they are
/// freed, so walking back and forth across the boundary doesn't thrash.
const UNLOAD_MARGIN: i32 = 3;
/// The vertical unload hysteresis (vertical radii are smaller, so is this).
const UNLOAD_MARGIN_V: i32 = 2;
/// How many fresh-chunk *mesh jobs* may be handed to the worker pool per
/// stream. Bounds the enqueue-time snapshot cost (a few KiB copy each) and
/// keeps the queue from flooding when a world is entered.
const MESH_ENQUEUE_BUDGET: usize = 8;
/// How many finished worker meshes may be uploaded to the GPU per stream —
/// the upload is the only part of the async path the render thread still pays.
const UPLOAD_BUDGET: usize = 4;
/// How many *dirty* (edited) chunks may remesh per frame. Processed nearest
/// first, so a locally broken block still vanishes the same frame while a
/// multiplayer join snapshot flood spreads over a few frames instead of one hitch.
const DIRTY_BUDGET: usize = 8;
/// The seed a default (`generate`) world uses when none is chosen.
pub const DEFAULT_SEED: i64 = 1;

/// A chunk coordinate: `(cx, cy, cz)` where world X = `cx * CHUNK_SIZE +
/// local x`, and likewise for Y and Z.
type Coord = (i32, i32, i32);

/// The "no centre yet" sentinel that forces the next stream to run a full pass.
const NO_CENTER: Coord = (i32::MIN, i32::MIN, i32::MIN);

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
/// `meshed == true` with `mesh == None` (uniform-air chunks are *born* that
/// way, skipping the worker round-trip entirely).
struct Loaded {
    chunk: Chunk,
    mesh: Option<MeshHandle>,
    meshed: bool,
    /// Mesh-input revision: bumped whenever this chunk's mesh inputs change —
    /// a direct edit, or an edit on a neighbour's touching border (which flips
    /// this chunk's exposed faces). A worker mesh result carries the rev its
    /// snapshot was taken at; a result whose rev no longer matches is stale
    /// and dropped (the chunk is in `dirty` or gets re-scanned anyway).
    rev: u32,
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
    /// own. Inner key is the flat voxel index within the chunk. (Saves store
    /// absolute coordinates; only this in-memory keying is per-chunk.)
    edits: FastMap<Coord, FastMap<usize, BlockId>>,
    /// Chunks whose mesh is stale (an edit changed them) and must rebuild,
    /// nearest first, ahead of any fresh meshing.
    dirty: FastSet<Coord>,
    /// The chunk the player was last centred on, so streaming only reacts to
    /// crossing a chunk boundary. Invalidated to force a full pass.
    center: Coord,
    /// Runtime render distance in chunk rings (clamped to [`VIEW_RADIUS_RANGE`]).
    /// The vertical streaming radius is derived from it — see
    /// [`vertical_radius`](Self::vertical_radius).
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
    /// Snapshot of the registry's hot solidity array for the mesher, behind an
    /// `Arc` so worker mesh jobs share it without copying. Refreshed when the
    /// palette grows (it is append-only): a refresh builds a *new* Arc, and
    /// in-flight jobs keep the old one harmlessly.
    solid_table: Arc<Vec<bool>>,
    /// Background generate/mesh workers, spawned lazily on the first
    /// [`stream`](Self::stream) so headless worlds (server, tests) never start
    /// threads. Dropped with the world: closing the job queue makes every
    /// worker exit, then the handles are joined (bounded — workers never touch
    /// the GPU, so nothing can wedge the join).
    workers: Option<pipeline::Workers>,
    /// Coords with a worker job in flight — either kind, one entry per coord
    /// (a generate job implies no data, a mesh job requires data, so the two
    /// never coexist). Blocks the centre-change scan from re-enqueueing a
    /// generate and the fresh scan from re-enqueueing a mesh; cleared per
    /// coord when its result drains, whatever becomes of the result.
    in_flight: FastSet<Coord>,
    /// Finished worker meshes awaiting their turn in the per-frame upload
    /// budget. Every entry re-validates its rev at upload time — it may have
    /// gone stale while queued.
    upload_queue: VecDeque<(Coord, u32, MeshData)>,
    /// Reusable buffer for draining worker results, so the drain neither
    /// borrows the channel across the processing loop nor allocates per frame.
    done_scratch: Vec<pipeline::Done>,
    /// How many blocks the last uploaded block-texture array covered. When the
    /// palette outgrows it (0 -> N on the first stream, +1 when crafting mints
    /// a new block type), the next stream rebuilds and re-uploads the array.
    textures_built: usize,
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
            center: NO_CENTER,
            view_radius: DEFAULT_VIEW_RADIUS,
            pending_fresh: true,
            radius_shrunk: false,
            scratch: MeshData::default(),
            solid_table: Arc::new(Vec::new()),
            workers: None,
            in_flight: FastSet::default(),
            upload_queue: VecDeque::new(),
            done_scratch: Vec::new(),
            textures_built: 0,
        };
        // Centre the pre-generated box on the origin's surface chunk, the
        // spawn point's own layer.
        let cy = world.generator.height(0, 0).div_euclid(CHUNK_SIZE as i32);
        world.ensure_region_data((0, cy, 0));
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

    /// Vertical streaming radius in chunk layers: half the horizontal view
    /// radius, clamped to 2..=5 — interesting terrain is mostly lateral, so
    /// the streamed volume stays a flat box rather than a cube.
    fn vertical_radius(&self) -> i32 {
        (self.view_radius / 2).clamp(2, 5)
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
            self.center = NO_CENTER;
            self.pending_fresh = true;
            // On shrink, meshes between the new radius and the (also shrunk)
            // unload ring would otherwise stay drawn until the player moves;
            // flag them so the next stream frees them immediately.
            self.radius_shrunk = shrunk;
            // In-flight worker jobs are NOT cancelled: results now outside the
            // radius are dropped by the range checks when they drain.
        }
    }

    /// Bring the world up to date around `center` (the player's position): land
    /// finished background work, queue new generation/meshing for nearby chunks,
    /// free distant ones. Requires the engine (it uploads meshes), so it runs
    /// from the game update, not from headless logic.
    ///
    /// Steady-state cost is near zero: the result drain is one non-blocking
    /// channel poll, the unload/generate pass only runs when the player crosses
    /// a chunk boundary (or the radius changed), and the fresh-mesh scan is
    /// skipped once a scan has found nothing left to hand out.
    pub fn stream(&mut self, center: Vec3, eng: &mut Engine) {
        // Palette growth re-uploads the block texture array before any meshing
        // this frame, so vertices never reference a layer that isn't there.
        // Covers the initial upload too (0 tracked -> N on the first stream).
        self.refresh_textures(eng);
        let s = CHUNK_SIZE as i32;
        let center_chunk = (
            (center.x.floor() as i32).div_euclid(s),
            (center.y.floor() as i32).div_euclid(s),
            (center.z.floor() as i32).div_euclid(s),
        );
        // Adopt the real centre BEFORE draining: after a radius change or
        // world reset the stored centre is a far-away sentinel, and draining
        // against it would discard every landed result - even in-range ones -
        // only to regenerate them moments later.
        let full_pass = center_chunk != self.center;
        self.center = center_chunk;
        // Land worker results before the scans below, so freshly generated
        // chunks count as data this frame and finished meshes draw this frame.
        self.drain_results(eng);
        if full_pass {
            self.unload_far(center_chunk, eng);
            self.request_region_data(center_chunk);
            self.pending_fresh = true;
        }
        if self.radius_shrunk {
            self.radius_shrunk = false;
            // Meshes between the new view radius and the unload ring survive
            // unload_far's hysteresis; free them now (data stays loaded).
            // Chunks that are `meshed` with no handle (all-air) stay as they
            // are — there is nothing to free and nothing drawn.
            let (rh, rv) = (self.view_radius as i64, self.vertical_radius() as i64);
            for (&coord, loaded) in self.chunks.iter_mut() {
                if Self::ring(coord, center_chunk) > rh || Self::updown(coord, center_chunk) > rv {
                    if let Some(handle) = loaded.mesh.take() {
                        loaded.meshed = false;
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
        // Drop the pipeline bookkeeping too: buffered worker meshes are for a
        // world we are leaving, and in-flight jobs may re-run from scratch if
        // we come back. Results still flying land against the invalidated
        // centre below and are dropped by the range/rev checks — at worst a
        // coord gets generated or meshed twice, never wrongly.
        self.in_flight.clear();
        self.upload_queue.clear();
        self.center = NO_CENTER;
        self.pending_fresh = true;
    }

    /// Land finished worker results: insert generated chunks, then upload
    /// finished meshes under [`UPLOAD_BUDGET`]. Strictly non-blocking — an
    /// idle frame costs one failed `try_recv`.
    ///
    /// Every drained result removes its coord from `in_flight`, uncondition-
    /// ally. A result that cannot apply (chunk unloaded, out of range, stale
    /// rev) is dropped, and for mesh results `pending_fresh` is re-set so the
    /// fresh scan can re-enqueue the coord if it still qualifies — that is the
    /// invariant that lets the scan clear `pending_fresh` while jobs still fly.
    fn drain_results(&mut self, eng: &mut Engine) {
        if let Some(workers) = &self.workers {
            while let Some(done) = workers.try_recv() {
                self.done_scratch.push(done);
            }
        }
        if self.done_scratch.is_empty() && self.upload_queue.is_empty() {
            return;
        }
        // Process outside the drain loop (the borrow checker aside, accepting
        // a result mutates half the world); the swap keeps the capacity.
        let mut done = std::mem::take(&mut self.done_scratch);
        for result in done.drain(..) {
            match result {
                pipeline::Done::Chunk { coord, chunk } => {
                    self.in_flight.remove(&coord);
                    self.accept_chunk(coord, chunk);
                }
                pipeline::Done::Mesh { coord, rev, data } => {
                    // NOT removed from in_flight yet: the coord stays claimed
                    // until its budgeted upload resolves, so the fresh scan
                    // can't re-enqueue a duplicate build meanwhile.
                    self.accept_mesh(coord, rev, data);
                }
            }
        }
        self.done_scratch = done;

        // Budgeted uploads. Re-validate at the moment of upload: an entry may
        // have sat queued across frames while an edit bumped the chunk's rev
        // (the synchronous dirty remesh has it covered in that case).
        let mut uploads = 0;
        while uploads < UPLOAD_BUDGET {
            let Some((coord, rev, data)) = self.upload_queue.pop_front() else {
                break;
            };
            self.in_flight.remove(&coord);
            if !self.mesh_result_applies(coord, rev) {
                self.pending_fresh = true; // went stale while queued: rescan
                continue;
            }
            let handle = eng.upload_mesh(&data); // None when the chunk is all air
            if let Some(loaded) = self.chunks.get_mut(&coord) {
                if let Some(old) = loaded.mesh.take() {
                    eng.free_mesh(old);
                }
                loaded.mesh = handle;
                loaded.meshed = true;
            }
            uploads += 1;
        }
    }

    /// A worker finished generating `coord`. Discard it if the world moved on
    /// (outside the data box) or the coord already has data (the centre
    /// safety floor generated it synchronously); otherwise insert it via
    /// [`store_chunk`](Self::store_chunk), which replays the edit overlay once
    /// more — idempotent over the worker's own replay, and it catches edits
    /// that arrived while the job flew.
    fn accept_chunk(&mut self, coord: Coord, chunk: Chunk) {
        let (rh, rv) = (
            (self.view_radius + DATA_MARGIN) as i64,
            (self.vertical_radius() + DATA_MARGIN) as i64,
        );
        if Self::ring(coord, self.center) > rh
            || Self::updown(coord, self.center) > rv
            || self.chunks.contains_key(&coord)
        {
            return;
        }
        self.store_chunk(coord, chunk);
    }

    /// A worker finished meshing `coord` at `rev`. Queue it for a budgeted
    /// upload if it can still apply; otherwise drop it and re-arm the fresh
    /// scan, which re-enqueues the coord if it still qualifies.
    fn accept_mesh(&mut self, coord: Coord, rev: u32, data: MeshData) {
        if self.mesh_result_applies(coord, rev) {
            self.upload_queue.push_back((coord, rev, data));
        } else {
            self.in_flight.remove(&coord);
            self.pending_fresh = true;
        }
    }

    /// Whether a worker mesh built at `rev` is still the right mesh for
    /// `coord`: the chunk is loaded, within view range of the current centre,
    /// and nothing bumped its rev since the snapshot. Checked when the result
    /// lands *and* again at upload time — it can go stale in between.
    fn mesh_result_applies(&self, coord: Coord, rev: u32) -> bool {
        Self::ring(coord, self.center) <= self.view_radius as i64
            && Self::updown(coord, self.center) <= self.vertical_radius() as i64
            && self.chunks.get(&coord).is_some_and(|l| l.rev == rev)
    }

    /// Horizontal Chebyshev ring distance between two chunk coords, widened to
    /// i64 so the invalidated-centre sentinel (`i32::MIN`) can never overflow
    /// a subtract.
    fn ring(a: Coord, b: Coord) -> i64 {
        (a.0 as i64 - b.0 as i64).abs().max((a.2 as i64 - b.2 as i64).abs())
    }

    /// Vertical (chunk-layer) distance between two chunk coords.
    fn updown(a: Coord, b: Coord) -> i64 {
        (a.1 as i64 - b.1 as i64).abs()
    }

    /// Streaming priority: 3D Chebyshev with the vertical axis weighted
    /// double, so the lateral terrain around the player streams in before the
    /// sky above it.
    fn order(a: Coord, b: Coord) -> i64 {
        Self::ring(a, b).max(2 * Self::updown(a, b))
    }

    /// Queue generation jobs for every missing chunk in the data box, nearest
    /// first — enqueue order is the pool's priority order. Safety floor: the
    /// centre chunk (the one holding the player) generates synchronously via
    /// [`ensure_data`](Self::ensure_data), so collision there never reads air
    /// while a job flies.
    fn request_region_data(&mut self, center: Coord) {
        self.ensure_data(center);
        let rh = self.view_radius + DATA_MARGIN;
        let rv = self.vertical_radius() + DATA_MARGIN;
        let mut missing: Vec<Coord> = Vec::new();
        for cx in (center.0 - rh)..=(center.0 + rh) {
            for cz in (center.2 - rh)..=(center.2 + rh) {
                for cy in (center.1 - rv)..=(center.1 + rv) {
                    let coord = (cx, cy, cz);
                    if !self.chunks.contains_key(&coord) && !self.in_flight.contains(&coord) {
                        missing.push(coord);
                    }
                }
            }
        }
        if missing.is_empty() {
            return;
        }
        missing.sort_by_key(|&coord| Self::order(coord, center));
        let workers = self
            .workers
            .get_or_insert_with(|| pipeline::Workers::spawn(pipeline::Workers::default_threads()));
        for coord in missing {
            let edits = self
                .edits
                .get(&coord)
                .map(|cells| cells.iter().map(|(&index, &id)| (index, id)).collect())
                .unwrap_or_default();
            let accepted = workers.submit(pipeline::Job::Generate {
                coord,
                generator: self.generator.clone(),
                edits,
            });
            if accepted {
                self.in_flight.insert(coord);
            }
        }
    }

    /// Ensure every chunk within the data box of `center` exists (voxel data
    /// only). Cheap and GPU-free, so it also seeds headless queries.
    fn ensure_region_data(&mut self, center: Coord) {
        let rh = self.view_radius + DATA_MARGIN;
        let rv = self.vertical_radius() + DATA_MARGIN;
        for cx in (center.0 - rh)..=(center.0 + rh) {
            for cz in (center.2 - rh)..=(center.2 + rh) {
                for cy in (center.1 - rv)..=(center.1 + rv) {
                    self.ensure_data((cx, cy, cz));
                }
            }
        }
    }

    /// Synchronously generate the small box of chunks around a position, so
    /// the first physics steps after entering a world land on real terrain.
    /// World::new only pre-generates around the ORIGIN; a save (or server
    /// spawn) can restore the player anywhere, and the async pipeline needs a
    /// few frames to catch up — during which collision would read the void as
    /// air and embed the player in late-arriving ground. Uniform fast paths
    /// make this box cheap (sky/deep-rock chunks are proven uniform).
    pub fn prepare_around(&mut self, pos: Vec3) {
        let c = Self::chunk_of(pos.x.floor() as i32, pos.y.floor() as i32, pos.z.floor() as i32);
        for cx in (c.0 - 1)..=(c.0 + 1) {
            for cz in (c.2 - 1)..=(c.2 + 1) {
                for cy in (c.1 - 2)..=(c.1 + 1) {
                    self.ensure_data((cx, cy, cz));
                }
            }
        }
    }

    /// Generate a chunk's data if it isn't loaded, replaying any saved edits on it.
    fn ensure_data(&mut self, coord: Coord) {
        if self.chunks.contains_key(&coord) {
            return;
        }
        let chunk = Chunk::new(coord.0, coord.1, coord.2, &self.generator);
        self.store_chunk(coord, chunk);
    }

    /// Insert freshly generated data: replay the edit overlay, then register
    /// the chunk. A uniform-air chunk (after replay) can never produce
    /// geometry, so it is born `meshed` with no mesh — no worker job, no
    /// upload, nothing drawn.
    fn store_chunk(&mut self, coord: Coord, mut chunk: Chunk) {
        if let Some(edits) = self.edits.get(&coord) {
            for (&index, &id) in edits {
                chunk.set_index(index, id);
            }
        }
        let meshed = chunk.uniform() == Some(AIR);
        self.chunks.insert(
            coord,
            Loaded {
                chunk,
                mesh: None,
                meshed,
                rev: 0,
            },
        );
        // New data means new mesh work next scan (its neighbours may have
        // been waiting on this chunk even when it is itself uniform air).
        self.pending_fresh = true;
    }

    /// Free chunks past the unload box, releasing their GPU meshes.
    fn unload_far(&mut self, center: Coord, eng: &mut Engine) {
        let rh = (self.view_radius + UNLOAD_MARGIN) as i64;
        let rv = (self.vertical_radius() + UNLOAD_MARGIN_V) as i64;
        // Collect-then-remove instead of `retain`: freeing needs `&mut eng`,
        // which can't be borrowed inside a retain closure over `self.chunks`.
        let far: Vec<Coord> = self
            .chunks
            .keys()
            .copied()
            .filter(|&coord| Self::ring(coord, center) > rh || Self::updown(coord, center) > rv)
            .collect();
        for coord in far {
            if let Some(loaded) = self.chunks.remove(&coord)
                && let Some(handle) = loaded.mesh
            {
                eng.free_mesh(handle);
            }
        }
    }

    /// Remesh edited chunks (nearest first, budgeted, synchronously — an edit
    /// must be visible the same frame), then hand up to [`MESH_ENQUEUE_BUDGET`]
    /// fresh chunks to the worker pool, nearest first. A fresh chunk is only
    /// snapshotted once its six orthogonal neighbours have data, so border
    /// faces are culled correctly the first time.
    fn build_meshes(&mut self, center: Coord, eng: &mut Engine) {
        if !self.dirty.is_empty() {
            let mut dirty: Vec<Coord> = self.dirty.iter().copied().collect();
            dirty.sort_by_key(|&coord| Self::order(coord, center));
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

        // Fresh chunks: snapshot and hand to the worker pool, nearest first,
        // capped by the frame budget. Skipped entirely once a scan came up
        // empty, until something re-flags work.
        if !self.pending_fresh {
            return;
        }
        let (rh, rv) = (self.view_radius as i64, self.vertical_radius() as i64);
        let mut pending: Vec<Coord> = self
            .chunks
            .iter()
            .filter(|(_, loaded)| !loaded.meshed)
            .map(|(&coord, _)| coord)
            .filter(|&coord| {
                Self::ring(coord, center) <= rh && Self::updown(coord, center) <= rv
            })
            .filter(|&coord| self.neighbours_have_data(coord))
            .filter(|coord| !self.in_flight.contains(coord))
            // Dirty chunks are the sync remesh path's job; queueing them here
            // too would double-mesh them (identical result, wasted snapshot,
            // worker build, and upload-budget slot).
            .filter(|coord| !self.dirty.contains(coord))
            .collect();
        pending.sort_by_key(|&coord| Self::order(coord, center));

        if pending.len() <= MESH_ENQUEUE_BUDGET {
            // Every candidate below gets enqueued, so the scan has nothing
            // left. Clearing while jobs still fly is sound: an in-flight coord
            // is *not* scan work — its result either lands as a mesh (`meshed`
            // flips true, nothing to scan) or fails to apply, which re-sets
            // `pending_fresh` after the coord left `in_flight`, so the next
            // scan sees it again. See `drain_results`.
            self.pending_fresh = false;
        }
        if pending.is_empty() {
            return;
        }
        self.refresh_tables(); // the snapshots below share the solid-table Arc
        for coord in pending.into_iter().take(MESH_ENQUEUE_BUDGET) {
            let (rev, snapshot) = self.snapshot(coord);
            let workers = self.workers.get_or_insert_with(|| {
                pipeline::Workers::spawn(pipeline::Workers::default_threads())
            });
            if workers.submit(pipeline::Job::Mesh { coord, rev, snapshot }) {
                self.in_flight.insert(coord);
            }
        }
    }

    /// Copy everything a worker mesh job needs for `coord`: the chunk's
    /// storage (a dense chunk's 4 KiB cells; a uniform chunk clones for free),
    /// the six neighbour border planes, and the shared solidity table. Returns
    /// the rev the snapshot represents. Runs on the main thread at enqueue
    /// time — cheap at [`MESH_ENQUEUE_BUDGET`] per frame.
    fn snapshot(&self, coord: Coord) -> (u32, pipeline::ChunkSnapshot) {
        let loaded = &self.chunks[&coord];
        let neighbours = self.neighbours(coord);
        (
            loaded.rev,
            pipeline::ChunkSnapshot {
                chunk: loaded.chunk.clone(),
                borders: mesh::BorderPlanes::capture(&neighbours),
                solid: Arc::clone(&self.solid_table),
            },
        )
    }

    /// Borrow the six orthogonal neighbours' chunks for a mesh build.
    fn neighbours(&self, (cx, cy, cz): Coord) -> mesh::Neighbours<'_> {
        let get = |c: Coord| self.chunks.get(&c).map(|l| &l.chunk);
        mesh::Neighbours {
            neg_x: get((cx - 1, cy, cz)),
            pos_x: get((cx + 1, cy, cz)),
            neg_z: get((cx, cy, cz - 1)),
            pos_z: get((cx, cy, cz + 1)),
            neg_y: get((cx, cy - 1, cz)),
            pos_y: get((cx, cy + 1, cz)),
        }
    }

    /// Whether the six orthogonal neighbours of a chunk have voxel data loaded.
    fn neighbours_have_data(&self, (cx, cy, cz): Coord) -> bool {
        self.chunks.contains_key(&(cx - 1, cy, cz))
            && self.chunks.contains_key(&(cx + 1, cy, cz))
            && self.chunks.contains_key(&(cx, cy, cz - 1))
            && self.chunks.contains_key(&(cx, cy, cz + 1))
            && self.chunks.contains_key(&(cx, cy - 1, cz))
            && self.chunks.contains_key(&(cx, cy + 1, cz))
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
            let neighbours = self.neighbours(coord);
            mesh::build_chunk_mesh(&loaded.chunk, &neighbours, &self.solid_table, &mut scratch);
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

    /// Re-snapshot the registry's hot solidity array if blocks were registered
    /// since the last build. The palette is append-only, so a length check
    /// suffices; a rebuild makes a *new* Arc, so worker jobs holding the old
    /// one are unaffected. (Colour needs no table anymore: the texture array
    /// carries it, keyed by block id — see
    /// [`refresh_textures`](Self::refresh_textures).)
    fn refresh_tables(&mut self) {
        let count = self.registry.block_count();
        if self.solid_table.len() != count {
            self.solid_table = Arc::new(
                (0..count)
                    .map(|i| self.registry.is_solid(BlockId(i as u16)))
                    .collect(),
            );
        }
    }

    /// Rebuild and upload the block texture array when the palette has grown
    /// since the last upload. Fires at most once per growth: on world entry
    /// (0 -> N) and when crafting registers a brand-new block type.
    /// `set_block_textures` waits for GPU idle — fine at this rarity.
    fn refresh_textures(&mut self, eng: &mut Engine) {
        let count = self.registry.block_count();
        if self.textures_built != count {
            let layers = crate::block::texture::build_block_textures(&self.registry);
            eng.set_block_textures(crate::block::texture::TEXTURE_SIZE, &layers);
            self.textures_built = count;
        }
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

    /// Replace the block at a world coordinate, recording the change in the edit
    /// overlay (so it survives streaming and can be saved) and marking the affected
    /// chunk — and any neighbour across a shared face — for remeshing. Returns the
    /// block that was there.
    pub fn set_block(&mut self, x: i32, y: i32, z: i32, id: BlockId) -> BlockId {
        let coord = Self::chunk_of(x, y, z);
        let s = CHUNK_SIZE as i32;
        let lx = x.rem_euclid(s) as usize;
        let ly = y.rem_euclid(s) as usize;
        let lz = z.rem_euclid(s) as usize;
        let index = Chunk::index(lx, ly, lz);

        let previous = self.block_at(x, y, z);
        self.edits.entry(coord).or_default().insert(index, id);

        if let Some(loaded) = self.chunks.get_mut(&coord) {
            loaded.chunk.set_index(index, id);
            loaded.meshed = false;
            // Any in-flight worker mesh of this chunk is now stale.
            loaded.rev = loaded.rev.wrapping_add(1);
            self.dirty.insert(coord);
            self.pending_fresh = true;
        }
        // A block on a chunk face also changes that neighbour's exposed
        // faces — even when the edited chunk itself has no data (a remote
        // edit landing in an unloaded chunk must still invalidate a loaded,
        // still-drawn neighbour, or its culled border face becomes a hole).
        let (cx, cy, cz) = coord;
        if lx == 0 {
            self.mark_dirty((cx - 1, cy, cz));
        }
        if lx == CHUNK_SIZE - 1 {
            self.mark_dirty((cx + 1, cy, cz));
        }
        if ly == 0 {
            self.mark_dirty((cx, cy - 1, cz));
        }
        if ly == CHUNK_SIZE - 1 {
            self.mark_dirty((cx, cy + 1, cz));
        }
        if lz == 0 {
            self.mark_dirty((cx, cy, cz - 1));
        }
        if lz == CHUNK_SIZE - 1 {
            self.mark_dirty((cx, cy, cz + 1));
        }
        previous
    }

    /// Mark a loaded chunk stale so the next stream remeshes it.
    fn mark_dirty(&mut self, coord: Coord) {
        if let Some(loaded) = self.chunks.get_mut(&coord) {
            loaded.meshed = false;
            // The neighbour's border edit changed this chunk's exposed faces,
            // so any in-flight worker mesh of it is stale too.
            loaded.rev = loaded.rev.wrapping_add(1);
            self.dirty.insert(coord);
            // In case the dirty pass drops it (missing neighbour data), the
            // fresh scan must be able to pick it back up later.
            self.pending_fresh = true;
        }
    }

    /// Collision test: does the given box overlap any solid voxel?
    ///
    /// Cells are visited grouped by owning chunk — one map probe per chunk the
    /// box touches (1–8 for anything player-sized) instead of one per cell,
    /// and a uniform chunk answers for all its cells with one solidity load.
    pub fn collides(&self, aabb: &Aabb) -> bool {
        // Same cell range as `Aabb::voxel_cells`: floor(min)..=floor(max).
        let (min, max) = (aabb.min(), aabb.max());
        let (x0, x1) = (min.x.floor() as i32, max.x.floor() as i32);
        let (y0, y1) = (min.y.floor() as i32, max.y.floor() as i32);
        let (z0, z1) = (min.z.floor() as i32, max.z.floor() as i32);

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

    /// Every recorded edit as `((x, y, z), block)`, for saving. Coordinates
    /// are absolute — the save format is independent of the chunk keying.
    pub fn edits(&self) -> impl Iterator<Item = ((i32, i32, i32), BlockId)> + '_ {
        self.edits.iter().flat_map(|(&(cx, cy, cz), cells)| {
            cells.iter().map(move |(&index, &id)| {
                let (lx, ly, lz) = Chunk::local_of(index);
                let x = cx * CHUNK_SIZE as i32 + lx as i32;
                let y = cy * CHUNK_SIZE as i32 + ly as i32;
                let z = cz * CHUNK_SIZE as i32 + lz as i32;
                ((x, y, z), id)
            })
        })
    }

    /// The chunk coordinate an absolute world position falls in.
    fn chunk_of(x: i32, y: i32, z: i32) -> Coord {
        let s = CHUNK_SIZE as i32;
        (x.div_euclid(s), y.div_euclid(s), z.div_euclid(s))
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
        assert!(world.is_solid(8, 0, 8), "surface-band ground should be solid");
        assert!(
            world.is_solid(8, -200, 8) || world.block_at(8, -200, 8) == AIR,
            "deep query must not panic"
        );
        // Deep rock is stone forever down (within the pre-generated region).
        assert!(world.is_solid(8, -40, 8), "no world floor: stone all the way down");
        // Above the hills and below the island band: air.
        assert!(!world.is_solid(8, 40, 8), "sky between terrain and islands is air");
    }

    #[test]
    fn collision_agrees_with_solidity() {
        let world = World::generate();
        let in_ground = Aabb::new(Vec3::new(8.5, 0.5, 8.5), Vec3::new(0.3, 0.3, 0.3));
        let in_sky = Aabb::new(Vec3::new(8.5, 40.0, 8.5), Vec3::new(0.3, 0.3, 0.3));
        let in_deep = Aabb::new(Vec3::new(8.5, -30.0, 8.5), Vec3::new(0.3, 0.3, 0.3));
        assert!(world.collides(&in_ground));
        assert!(!world.collides(&in_sky));
        assert!(world.collides(&in_deep), "uniform stone chunks collide");
    }

    #[test]
    fn collision_grouped_lookup_matches_per_cell_path() {
        // Boxes straddling chunk boundaries exercise the multi-chunk grouping
        // (including vertical boundaries now); the grouped fast path must
        // agree with a per-cell `is_solid` sweep.
        let world = World::generate();
        for center in [
            Vec3::new(15.9, 18.0, 15.9), // corner of four chunks
            Vec3::new(0.1, 21.5, 8.0),   // one X boundary
            Vec3::new(-3.2, 19.0, -16.4),
            Vec3::new(4.0, -1.0, 4.0),  // below the surface band: solid now
            Vec3::new(4.0, 15.9, 4.0),  // straddles a vertical chunk boundary
            Vec3::new(4.0, 200.0, 4.0), // unloaded high sky: air on both paths
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
        let h = (0..64)
            .rev()
            .find(|&y| world.is_solid(x, y, z))
            .expect("the column has solid ground");

        assert_eq!(world.block_at(x, h + 1, z), AIR);
        assert_eq!(world.block_at(x, h, z), grass);
        assert_eq!(world.block_at(x, h - 1, z), dirt);
        assert_eq!(world.block_at(x, h - 3, z), stone);
        // And no bottom anymore: the deep layer continues below y = 0.
        assert_eq!(world.block_at(x, -25, z), stone);
    }

    #[test]
    fn edits_persist_across_unload() {
        let mut world = World::generate();
        // Break the surface block, then regenerate the chunk from scratch —
        // the edit must replay. Also place a block above the old ceiling
        // (y >= 64 is legal now) and expect the same.
        let (x, z) = (8, 8);
        let h = (0..64)
            .rev()
            .find(|&y| world.is_solid(x, y, z))
            .unwrap();
        world.set_block(x, h, z, AIR);
        assert_eq!(world.block_at(x, h, z), AIR);
        let stone = world.registry().id_by_name("Stone").unwrap();
        world.set_block(x, 70, z, stone);

        // Drop the chunks and regenerate; the recorded edits should return.
        world.chunks.clear();
        world.ensure_data(World::chunk_of(x, h, z));
        world.ensure_data(World::chunk_of(x, 70, z));
        assert_eq!(world.block_at(x, h, z), AIR, "edit survived reload");
        assert_eq!(world.block_at(x, 70, z), stone, "high edit survived reload");
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
    fn stale_rev_mesh_results_are_dropped() {
        let mut world = World::generate();
        world.center = (0, 0, 0); // pretend the player streamed here
        let coord = (0, 0, 0);
        let rev = world.chunks[&coord].rev;
        assert!(world.mesh_result_applies(coord, rev));

        // An edit bumps the rev: the snapshot a worker holds is now stale.
        world.set_block(3, 3, 3, AIR);
        assert!(!world.mesh_result_applies(coord, rev));

        // A stale landing is dropped and re-arms the fresh scan.
        world.pending_fresh = false;
        world.accept_mesh(coord, rev, MeshData::default());
        assert!(world.upload_queue.is_empty(), "stale result never queues");
        assert!(world.pending_fresh, "drop re-arms the scan");

        // A current-rev landing queues for upload.
        let rev = world.chunks[&coord].rev;
        world.accept_mesh(coord, rev, MeshData::default());
        assert_eq!(world.upload_queue.len(), 1);
        world.upload_queue.clear();

        // Unloaded / out-of-range coords are rejected too — horizontally and
        // vertically (the vertical radius is tighter).
        assert!(!world.mesh_result_applies((99, 0, 99), 0));
        assert!(!world.mesh_result_applies((0, 99, 0), 0));
        let rv = world.vertical_radius();
        assert!(!world.mesh_result_applies((0, rv + 1, 0), 0), "just past vertical range");
    }

    #[test]
    fn neighbour_edits_bump_the_bordering_chunks_rev() {
        let mut world = World::generate();
        // An edit at x == 0 of chunk (0, 0, 0) touches chunk (-1, 0, 0)'s border.
        let before = world.chunks[&(-1, 0, 0)].rev;
        world.set_block(0, 5, 8, AIR);
        assert_eq!(world.chunks[&(-1, 0, 0)].rev, before + 1, "border neighbour");
        assert_eq!(world.chunks[&(0, 0, 0)].rev, 1, "edited chunk itself");
        assert_eq!(world.chunks[&(1, 0, 0)].rev, 0, "far side untouched");

        // Vertical borders count too: an edit at y == 16 (bottom of chunk
        // layer 1) touches the chunk below.
        let below = world.chunks[&(0, 0, 0)].rev;
        world.set_block(8, 16, 8, AIR);
        assert_eq!(world.chunks[&(0, 0, 0)].rev, below + 1, "chunk below bumped");
        assert_eq!(world.chunks[&(0, 1, 0)].rev, 1, "edited vertical chunk");
    }

    #[test]
    fn landed_chunks_replay_edits_that_arrived_mid_flight() {
        let mut world = World::generate();
        world.center = (0, 0, 0);
        let coord = (2, 0, 2);
        let (x, z) = (coord.0 * CHUNK_SIZE as i32 + 3, coord.2 * CHUNK_SIZE as i32 + 4);
        // Simulate the coord being in flight: no data yet, edit lands meanwhile
        // (recorded in the overlay only).
        world.chunks.remove(&coord);
        world.set_block(x, 5, z, AIR);
        // The worker's result was built before that edit existed.
        let raw = Chunk::new(coord.0, coord.1, coord.2, &world.generator);
        assert_ne!(raw.get_local(3, 5, 4), AIR, "terrain is solid there");
        world.pending_fresh = false;
        world.accept_chunk(coord, raw);
        assert_eq!(world.block_at(x, 5, z), AIR, "overlay replayed on landing");
        assert!(world.pending_fresh, "new data re-arms the fresh scan");

        // Results for coords the world has moved past are discarded —
        // horizontally or vertically.
        let far = (100, 0, 100);
        world.accept_chunk(far, Chunk::new(far.0, far.1, far.2, &world.generator));
        assert!(!world.chunks.contains_key(&far), "out-of-range chunk dropped");
        let high = (0, 100, 0);
        world.accept_chunk(high, Chunk::new(high.0, high.1, high.2, &world.generator));
        assert!(!world.chunks.contains_key(&high), "out-of-height chunk dropped");
    }

    #[test]
    fn uniform_air_chunks_are_born_meshed() {
        let world = World::generate();
        // A sky chunk between the hills and the island band: uniform air,
        // meshed on arrival with no mesh and no worker job ever queued.
        let sky = &world.chunks[&(0, 3, 0)];
        assert_eq!(sky.chunk.uniform(), Some(AIR));
        assert!(sky.meshed, "uniform air needs no mesh job");
        assert!(sky.mesh.is_none());
        // A ground chunk still goes through the normal mesh path.
        let ground = &world.chunks[&(0, 0, 0)];
        assert!(!ground.meshed, "dense terrain waits for a real mesh");
    }

    #[test]
    fn vertical_radius_derives_from_view_radius() {
        let mut world = World::generate();
        for (view, vertical) in [(3, 2), (4, 2), (6, 3), (8, 4), (10, 5)] {
            world.set_view_radius(view);
            assert_eq!(world.vertical_radius(), vertical, "view {view}");
        }
    }

    #[test]
    fn streaming_order_weights_vertical_double() {
        let c = (0, 0, 0);
        assert_eq!(World::order((4, 0, 0), c), 4);
        assert_eq!(World::order((0, 2, 0), c), 4, "2 layers up ranks like 4 rings out");
        assert!(
            World::order((0, 3, 0), c) > World::order((5, 0, 0), c),
            "lateral terrain streams before the sky"
        );
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
        assert_eq!(world.center, NO_CENTER);
    }
}
