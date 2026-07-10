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
//!
//! [`World`] is ONE struct, but its methods are grouped by concern across
//! sibling files (multiple `impl World` blocks — pure code motion):
//!
//! * `mod.rs` (this file) — constants, the fast hash maps, [`Loaded`], the
//!   `World` struct itself, construction, and rendering.
//! * `streaming.rs` — the per-frame [`stream`](World::stream) pass: worker
//!   result draining, generation/mesh job queueing, budgeted uploads,
//!   unloading, and the radius/centre bookkeeping.
//! * `query.rs` — read-only queries: block lookup, solidity, collision,
//!   surface height, coordinate mapping, registry/seed accessors.
//! * `edits.rs` — player edits: block placement, the edit overlay, dirty
//!   marking, mesh freeing, and the render-distance setting.
pub mod chunk;
pub mod connectivity;
pub mod generation;
pub mod light;
pub mod lod;
pub mod mesh;
pub mod pipeline;

mod edits;
mod query;
mod streaming;

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{BuildHasherDefault, Hasher};

use voxel_engine::{DVec3, Engine, Frame3D, MeshData, MeshHandle, Vec3};

use crate::block::registry::{BlockId, BlockRegistry, HotTables};
use crate::coord::{ByPass, ChunkBox, ChunkCoord};
use crate::render::Render;
use chunk::{CHUNK_SIZE, Chunk};
use generation::{SineHills, TerrainGenerator};
use light::LightGrid;
use lod::{Tile, TileState};
use mesh::{ChunkMeshData, new_chunk_mesh_data};

/// Default number of chunk rings meshed and drawn around the player.
const DEFAULT_VIEW_RADIUS: i32 = 6;
/// The range a runtime render-distance change is clamped to. Shared: the
/// settings model and the settings menu stepper clamp to the same bounds.
pub const VIEW_RADIUS_RANGE: std::ops::RangeInclusive<i32> = 3..=20;
/// One extra shell of *data* (not meshed) in all three axes so edge chunks can
/// cull faces against their neighbours without re-meshing when those
/// neighbours later load.
const DATA_MARGIN: i32 = 1;
/// How far past the view radius chunks survive before they are freed, so
/// walking back and forth across the boundary doesn't thrash. Isotropic: the
/// streamed volume is a cube (see [`ViewVolume`]), so one margin serves every axis.
const UNLOAD_MARGIN: i32 = 3;
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
/// How many chunks the main-thread light-settle pass relaxes per stream. Each
/// `propagate` is ~µs (it only floods a 16³ grid), so this can dwarf the mesh
/// budget: settling must stay well ahead of meshing, which gates on it, or fresh
/// meshes stall waiting for their neighbourhood light to converge.
const LIGHT_SETTLE_BUDGET: usize = 64;
/// The seed a default (`generate`) world uses when none is chosen.
pub const DEFAULT_SEED: i64 = 1;
/// The far-tile ring tracks `view_radius`: the LOD pyramid fills the band from
/// render distance out to twice render distance ([`level_at`](lod::level_at)),
/// finer levels near and coarser far, no matter the setting.
const LOD_REACH: i32 = 2;
/// Tile mesh jobs handed to the pool per stream, and tile uploads per frame —
/// their own budgets so a world-entry tile flood can't starve chunk meshing.
const TILE_ENQUEUE_BUDGET: usize = 2;
const TILE_UPLOAD_BUDGET: usize = 2;

/// A chunk-coordinate map key. The [`ChunkCoord`] newtype owns the
/// `chunk * CHUNK_SIZE + local` relationship (see [`crate::coord`]); the
/// `Coord` alias keeps the shorter name the `world` module already used.
use crate::coord::ChunkCoord as Coord;
use connectivity::{Connectivity, Occlusion};

/// The streamed chunk volume around the player: a horizontal ring radius and a
/// vertical layer radius, in chunks. **Isotropic by default** (`vertical ==
/// horizontal`, a cube): the terrain here spans hundreds of blocks vertically
/// (sea, mountains, the flying-island band, the ice cap), so the old flat disk
/// popped in-view full-res chunks on ordinary vertical movement (jumping,
/// cliffs, flight). The two axes stay separable so a future anisotropic tuning
/// is a value change behind [`set_view_radius`](World::set_view_radius), not a
/// change to the box-building shape.
#[derive(Clone, Copy)]
pub(in crate::world) struct ViewVolume {
    horizontal: i32,
    vertical: i32,
}

impl ViewVolume {
    /// A cube: vertical radius equals horizontal.
    fn cube(radius: i32) -> Self {
        Self { horizontal: radius, vertical: radius }
    }
    /// The mesh box grown by `margin` chunks on every axis. Isotropy is what
    /// collapses the old horizontal/vertical margin pair into one scalar.
    fn box_at(self, center: Coord, margin: i32) -> ChunkBox {
        ChunkBox::new(center, self.horizontal + margin, self.vertical + margin)
    }
    /// Chunks meshed and drawn around `center`.
    fn mesh(self, center: Coord) -> ChunkBox {
        self.box_at(center, 0)
    }
    /// The mesh box plus one [`DATA_MARGIN`] shell of voxel data, so edge chunks
    /// cull against neighbours that are loaded but unmeshed.
    fn data(self, center: Coord) -> ChunkBox {
        self.box_at(center, DATA_MARGIN)
    }
    /// The mesh box plus the [`UNLOAD_MARGIN`] hysteresis, past which chunks free.
    fn unload(self, center: Coord) -> ChunkBox {
        self.box_at(center, UNLOAD_MARGIN)
    }
}

/// Fast multiply-based hasher for well-distributed grid coordinate keys (chunk hot path).
#[derive(Default)]
pub(in crate::world) struct FastHasher(u64);

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

pub(in crate::world) type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;
pub(in crate::world) type FastSet<K> = HashSet<K, BuildHasherDefault<FastHasher>>;

/// Raise-then-consume flag: can be raised or consumed, never lowered (so a queued
/// shrink survives an intervening grow; a queued raise can't be silently clobbered).
#[derive(Default, Clone, Copy)]
pub struct Sticky(bool);

impl Sticky {
    /// A flag already raised (for fields that start armed).
    pub fn raised() -> Self {
        Self(true)
    }
    /// Raise iff `v` — OR it in, never lowering an already-raised flag.
    pub fn raise(&mut self, v: bool) {
        self.0 |= v;
    }
    /// Raise unconditionally.
    pub fn set(&mut self) {
        self.0 = true;
    }
    /// Read and clear, returning whether it was raised.
    pub fn take(&mut self) -> bool {
        std::mem::take(&mut self.0)
    }
    /// Peek without clearing.
    pub fn get(&self) -> bool {
        self.0
    }
}

/// A loaded chunk: its voxel data plus the [`MeshState`] tracking the GPU mesh
/// built from it. The state captures both the lifecycle stage and the handle
/// ownership: a born-air chunk is `Air` (nothing drawn, no worker job), a
/// drawable one is `Ready(handle)`, and so on — see [`MeshState`].
struct Loaded {
    chunk: Chunk,
    state: MeshState,
    /// Mesh-input revision: bumped whenever this chunk's mesh inputs change —
    /// a direct edit, or an edit on a neighbour's touching border (which flips
    /// this chunk's exposed faces). A worker mesh result carries the rev its
    /// snapshot was taken at; a result whose rev no longer matches is stale
    /// and dropped (the chunk is `Dirty` or gets re-scanned anyway).
    rev: u32,
    /// Which of this chunk's faces a sightline can pass between, for the
    /// occlusion BFS. `None` until first needed and after a direct edit — it is
    /// computed **lazily**, only when the occlusion gate is active, so a world
    /// that never runs occlusion (the common, CPU-bound case) never pays the
    /// flood-fill. Depends only on the chunk's own voxels, so a neighbour edit
    /// (which bumps `rev`) leaves it valid.
    connectivity: Option<Connectivity>,
    /// The chunk's settled light grid, published by the main-thread
    /// [`settle_light`](World::settle_light) pass (decoupled from meshing). Read
    /// as part of the neighbour shell ([`light::PaddedLight`]) when an adjacent
    /// chunk meshes or settles, and directly queryable for gameplay (mob spawns,
    /// plant growth) with no mesh. `None` until the chunk has first settled.
    light: Option<LightGrid>,
}

impl Loaded {
    /// Transition to `next`, freeing the mesh this chunk was drawing unless
    /// that mesh is carried into `next`. This is the single place that frees a
    /// chunk's mesh: every GPU-freeing transition — unload, radius shrink, sync
    /// remesh, world-leave — routes through here, so there's one place to check
    /// for double frees or leaks.
    fn retire(&mut self, next: MeshState, eng: &mut Engine) {
        std::mem::replace(&mut self.state, next).free_owned(eng);
    }
}

/// Sole owner of one GPU mesh allocation. Deliberately NOT `Copy`/`Clone`:
/// holding an `OwnedMesh` *is* holding the live handle, so the only ways to
/// dispose of it are [`free`](OwnedMesh::free) (which releases the GPU
/// allocation) or moving it into another [`MeshState`] (carrying the same mesh
/// forward). `free` consumes `self`, so the same allocation can't be released
/// twice, and the handle can't be silently overwritten and leaked.
#[derive(Debug, PartialEq, Eq)]
pub(in crate::world) struct OwnedMesh(MeshHandle);

impl OwnedMesh {
    /// Wrap a freshly uploaded handle as its sole owner.
    fn new(handle: MeshHandle) -> Self {
        Self(handle)
    }
    /// The `Copy` handle id, for recording a draw — a borrow, so ownership stays put.
    fn id(&self) -> MeshHandle {
        self.0
    }
    /// Release the GPU allocation. Consumes `self`, so it can't be double-freed.
    fn free(self, eng: &mut Engine) {
        eng.free_mesh(self.0);
    }
}

/// The GPU meshes of one chunk, split by draw pass — the resident dual of
/// [`ChunkMeshData`]. A chunk yields an opaque mesh, a transparent mesh, or both;
/// the smart constructor enforces **at least one present** (an all-empty chunk is
/// [`MeshState::Air`], never a `Ready` with nothing to draw). Freeing frees both,
/// so this stays the sole GPU-free chokepoint, now over a `ByPass`.
#[derive(Debug, PartialEq, Eq)]
pub(in crate::world) struct ChunkMeshes(ByPass<Option<OwnedMesh>>);

impl ChunkMeshes {
    /// `None` when no pass has a mesh (the caller uses [`MeshState::Air`]).
    fn new(passes: ByPass<Option<OwnedMesh>>) -> Option<Self> {
        let any = passes.iter().any(|(_, m)| m.is_some());
        any.then_some(Self(passes))
    }
    /// Record a draw for each present pass at `offset`/`scale`.
    fn draw(&self, f: &mut Frame3D, offset: Vec3, scale: f32) {
        for (_, m) in self.0.iter() {
            if let Some(mesh) = m {
                f.draw_mesh(mesh.id(), offset, scale);
            }
        }
    }
    /// Free every present pass's GPU allocation. Consumes `self`.
    fn free(self, eng: &mut Engine) {
        for (_, m) in self.0.into_iter_passes() {
            if let Some(mesh) = m {
                mesh.free(eng);
            }
        }
    }
    /// Whether any pass draws `handle` — for the render/ownership tests.
    #[cfg(test)]
    fn draws(&self, handle: MeshHandle) -> bool {
        self.0.iter().any(|(_, m)| m.as_ref().is_some_and(|o| o.id() == handle))
    }
}

/// Mesh-lifecycle state of a loaded chunk. Owns GPU mesh via [`OwnedMesh`] token;
/// `rev` bumped when mesh inputs stale. Handle ownership rides the state machine:
/// moves on edit (`Dirty.prev`) or frees on unload/shrink/remesh. `Meshing` owns none.
#[derive(Debug, PartialEq, Eq)]
enum MeshState {
    /// Uniform-air, born meshed: nothing to draw, no worker job ever queued.
    Air,
    /// Dense data with no mesh yet and no job outstanding: awaiting the fresh scan.
    NeedsMesh,
    /// A fresh mesh job is outstanding on the worker pool (owns no handle;
    /// `rev` on `Loaded` referees its result; the `Meshing` state IS the
    /// "mesh in flight" claim, held until the budgeted upload resolves).
    Meshing,
    /// Drawable: owns the live GPU mesh(es) (up to one per pass).
    Ready(ChunkMeshes),
    /// Edited, awaiting the synchronous remesh. `prev` is the previously-drawn
    /// mesh of an edited-`Ready` chunk — kept drawn until the remesh replaces
    /// it — or `None` if the chunk had no mesh when edited.
    Dirty { prev: Option<ChunkMeshes> },
}

impl MeshState {
    /// The mesh(es) currently drawn, if any: a `Ready` chunk, or an edited
    /// `Dirty` one still showing its previous mesh. Borrows, so ownership stays
    /// in the state (the render/draw gate).
    fn live_meshes(&self) -> Option<&ChunkMeshes> {
        match self {
            MeshState::Ready(m) | MeshState::Dirty { prev: Some(m) } => Some(m),
            _ => None,
        }
    }
    /// Move the owned meshes out for freeing. Consumes `self`; the caller owns
    /// the token afterwards and must `free` it (or carry it on).
    #[must_use]
    fn into_owned(self) -> Option<ChunkMeshes> {
        match self {
            MeshState::Ready(m) | MeshState::Dirty { prev: Some(m) } => Some(m),
            _ => None,
        }
    }
    /// Free the owned meshes, if any. Consuming `self` here means a caller can't
    /// hold on to a state and free it twice. Every GPU-freeing removal —
    /// unload, world-leave, [`Loaded::retire`] — routes through this.
    fn free_owned(self, eng: &mut Engine) {
        if let Some(meshes) = self.into_owned() {
            meshes.free(eng);
        }
    }
    /// The state a fresh upload produces: `Ready` if any pass yielded a handle,
    /// else `Air` (an all-air chunk uploads to nothing). This is the D-2
    /// "≥1 present" invariant site.
    fn from_upload(handles: ByPass<Option<MeshHandle>>) -> MeshState {
        let meshes = ByPass::from_fn(|p| handles[p].map(OwnedMesh::new));
        match ChunkMeshes::new(meshes) {
            Some(m) => MeshState::Ready(m),
            None => MeshState::Air,
        }
    }
    /// Invalidate to `Dirty`, carrying the currently-drawn mesh forward as
    /// `prev` so it keeps drawing until the sync remesh. Nothing is freed here
    /// — the token just moves. `Ready(m) -> Dirty{Some(m)}`; an already-`Dirty`
    /// chunk keeps its `prev`; handle-less states -> `Dirty{None}`.
    fn invalidate(&mut self) {
        let prev = std::mem::replace(self, MeshState::NeedsMesh).into_owned();
        *self = MeshState::Dirty { prev };
    }
    fn is_needs_mesh(&self) -> bool {
        matches!(self, MeshState::NeedsMesh)
    }
    fn is_dirty(&self) -> bool {
        matches!(self, MeshState::Dirty { .. })
    }
}

/// The streamed world: the block palette, the terrain generator, the currently
/// loaded chunks, and the overlay of player edits that outlive chunk unloads.
pub struct World {
    /// Read-only block palette; meshing/collision read its hot solidity arrays.
    registry: BlockRegistry,
    generator: SineHills,
    chunks: FastMap<Coord, Loaded>,
    /// Player edits grouped by chunk (inner key: flat voxel index for replay on regenerate).
    edits: FastMap<Coord, FastMap<usize, BlockId>>,
    /// Last chunk centre; `None` forces a full stream pass. Streams only react to boundary crosses.
    center: Option<Coord>,
    /// The streamed chunk volume (horizontal ring + vertical layer radii),
    /// isotropic by default. Its horizontal radius is the render-distance
    /// setting (clamped to [`VIEW_RADIUS_RANGE`]); see [`ViewVolume`].
    view: ViewVolume,
    /// Fresh meshes may exist. Cleared on idle scan, set by edits/new chunks/radius change.
    pending_fresh: Sticky,
    /// Edited chunks may need remesh (hint; authoritative set is `Dirty` fiber in `MeshState`).
    pending_dirty: Sticky,
    /// Render distance shrank; next stream frees meshes beyond new radius.
    radius_shrunk: Sticky,
    /// Reusable CPU-side mesh scratch (one [`MeshData`] per pass); `upload_mesh`
    /// copies out of it, so one buffer set serves every sync chunk build.
    scratch: ChunkMeshData,
    /// Hot tables (solid/opaque/emission) behind `Arc` for worker sharing. Rebuilt on palette growth.
    tables: crate::derived::Derived<HotTables>,
    /// Background workers spawned lazily on first stream (headless worlds skip threads).
    workers: Option<pipeline::Workers>,
    /// Coords with generate jobs in flight. Blocks re-enqueue; cleared on drain.
    generating: FastSet<Coord>,
    /// Finished meshes awaiting budgeted upload (re-validated at upload time for staleness).
    upload_queue: VecDeque<(Coord, u32, ChunkMeshData)>,
    /// Chunks needing light settling (budgeted, seeded on load/edit/border moves).
    light_worklist: FastSet<Coord>,
    /// Reusable buffer for draining worker results, so the drain neither
    /// borrows the channel across the processing loop nor allocates per frame.
    done_scratch: Vec<pipeline::Done>,
    /// Block count last uploaded. Rebuilds/re-uploads when palette grows.
    textures_built: usize,
    /// Occlusion visible set (rebuilt at stream sync point, read by render).
    occlusion: Occlusion,
    /// Occlusion visible set needs rebuild (input-triggered on centre/chunk/connectivity change).
    occlusion_dirty: Sticky,
    /// Whether occlusion was active last stream (render honours visible set if active).
    occlusion_active: bool,
    /// Manual occlusion override (`VOXEL_OCCLUSION=1`, defaults off when GPU-bound signal unavailable).
    occlusion_forced: bool,
    /// Far LOD tiles (parallel lane to chunks; meet at occlusion-skip predicate).
    tiles: FastMap<Tile, TileState>,
    /// Finished tile meshes awaiting upload (never stale by edit, only by unload).
    tile_upload_queue: VecDeque<(Tile, MeshData)>,
    /// Whether desired tiles still need enqueueing (the enqueue budget spreads a
    /// world-entry flood across frames).
    pending_tiles: Sticky,
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
            center: None,
            view: ViewVolume::cube(DEFAULT_VIEW_RADIUS),
            pending_fresh: Sticky::raised(),
            pending_dirty: Sticky::default(),
            radius_shrunk: Sticky::default(),
            scratch: new_chunk_mesh_data(),
            tables: crate::derived::Derived::default(),
            workers: None,
            generating: FastSet::default(),
            upload_queue: VecDeque::new(),
            light_worklist: FastSet::default(),
            done_scratch: Vec::new(),
            textures_built: 0,
            occlusion: Occlusion::default(),
            occlusion_dirty: Sticky::default(),
            occlusion_active: false,
            occlusion_forced: matches!(std::env::var("VOXEL_OCCLUSION").as_deref(), Ok("1")),
            tiles: FastMap::default(),
            tile_upload_queue: VecDeque::new(),
            pending_tiles: Sticky::default(),
        };
        // Centre the pre-generated box on the origin's surface chunk, the
        // spawn point's own layer.
        let cy = world.generator.height(0, 0).div_euclid(CHUNK_SIZE as i32);
        world.ensure_region_data(ChunkCoord::new(0, cy, 0));
        world
    }

    /// The default world (seed [`DEFAULT_SEED`]).
    pub fn generate() -> Self {
        Self::new(DEFAULT_SEED)
    }

    /// Whether `coord` is a loaded chunk awaiting its fresh mesh — dense data,
    /// no mesh, no job outstanding, not edited. The fresh-scan meshed-ness test.
    fn is_needs_mesh(&self, coord: Coord) -> bool {
        self.chunks.get(&coord).is_some_and(|l| l.state.is_needs_mesh())
    }

    /// Sanity check: a coord claimed as a live generate job (`generating`)
    /// should have no `Loaded` entry yet — it's only in that set because it's
    /// still absent. A generating coord that already has data would block its
    /// own regeneration and never re-enter the mesh pipeline. Runs behind
    /// `cfg(debug_assertions)` (on in test builds).
    #[cfg(debug_assertions)]
    fn debug_assert_liveness(&self) {
        for coord in &self.generating {
            debug_assert!(
                !self.chunks.contains_key(coord),
                "generating coord {coord:?} already has data — a stuck generate claim"
            );
        }
    }

    /// Draw the meshed chunks. All per-voxel work happened when each chunk was
    /// built; a frame is one `draw_mesh` per chunk (the engine frustum-culls
    /// each against its offset AABB internally).
    ///
    /// Meshes are CHUNK-LOCAL (vertices in 0..=16), so each draw carries the
    /// camera-relative offset `chunk_origin - cam`, computed here in `f64` and
    /// only then narrowed to `f32`: near the camera — the only place precision
    /// is visible — the difference is small and exact, no matter how far from
    /// the world origin both sit.
    pub fn render(&self, f: &mut Frame3D, cam: DVec3) {
        let s = CHUNK_SIZE as f64;
        // A pure projection of prepared state. When occlusion is active the
        // visible set was rebuilt in `stream`; when it is off (the default, no
        // GPU-bound signal) every frustum-visible chunk draws — the baseline
        // path, no occlusion cost. The engine frustum-culls each drawn chunk.
        // Layer 1 — full-res chunks, drawn first so they fill depth before the
        // tile backdrop. No ownership cull: the tiles under them are pushed back by
        // depth bias, not skipped, so there is no boundary to align.
        for (&coord, loaded) in &self.chunks {
            if self.occlusion_active && !self.occlusion.is_visible(coord) {
                continue;
            }
            if let Some(meshes) = loaded.state.live_meshes() {
                let origin =
                    DVec3::new(coord.x as f64 * s, coord.y as f64 * s, coord.z as f64 * s);
                // Full-res chunks are unit-scale: their local 0..=16 coords are
                // already metres. LOD tiles pass 2^k here.
                meshes.draw(f, (origin - cam).as_vec3(), 1.0);
            }
        }

        // Layer 2 — the coarse tile backdrop: every loaded tile, drawn depth-biased
        // so full-res wins wherever both cover the same ground (no z-fight) and the
        // tile shows only where full-res is absent (no gap, no notch). Tiles fully
        // hidden by full-res are never loaded (see `desired_tiles`), so nothing
        // wasted meshes; the rest are early-Z discarded under the chunks above.
        for (&tile, state) in &self.tiles {
            let Some(handle) = state.drawable() else { continue };
            let origin = DVec3::new(
                tile.origin_x() as f64,
                tile.origin_y() as f64,
                tile.origin_z() as f64,
            );
            f.draw_mesh_biased(handle, (origin - cam).as_vec3(), tile.lod.cell() as f32);
        }
    }

    /// Whether the full-res slab *fully* contains `tile`, so the chunks draw all of
    /// it and the tile is provably never visible — a pure LOADING skip (it no longer
    /// runs in `render`; the tile backdrop is drawn depth-biased instead, so a loaded
    /// tile that overlaps the slab is simply hidden by depth, not culled here).
    /// Containment on all three axes, NOT overlap: a tile that only clips the slab
    /// boundary is not fully covered, so it must be loaded — dropping it would leave
    /// a hole the biased backdrop can't fill because it was never meshed. Now that a
    /// tile carries its own `y`, the vertical term is symmetric with x/z: the tile's
    /// chunk-layer span must sit wholly inside the vertical slab, else it loads (which
    /// is what keeps the ground/islands when the player flies away from them). `false`
    /// before the first stream (no centre yet).
    fn tile_occluded(&self, tile: Tile) -> bool {
        let Some(c) = self.center else { return false };
        let r = self.view.horizontal;
        let vr = self.view.vertical;
        let cps = tile.lod.chunks_per_side();
        let (cx0, cy0, cz0) = (tile.x * cps, tile.y * cps, tile.z * cps);
        let (cx1, cy1, cz1) = (cx0 + cps - 1, cy0 + cps - 1, cz0 + cps - 1);
        cx0 >= c.x - r && cx1 <= c.x + r
            && cy0 >= c.y - vr && cy1 <= c.y + vr
            && cz0 >= c.z - r && cz1 <= c.z + r
    }

    /// Whether the occlusion gate is active this frame — the adaptive decision
    /// to spend CPU culling in order to save GPU draw time. Occlusion only pays
    /// off when the frame is GPU/overdraw-bound; that signal lives in the engine
    /// (GPU timestamps) and isn't wired yet, so this is off unless force-enabled
    /// with `VOXEL_OCCLUSION=1`. When the engine exposes it, OR the GPU-bound
    /// signal in here and every GPU-side optimisation can share this one gate.
    fn occlusion_enabled(&self) -> bool {
        self.occlusion_forced
    }

    /// Rebuild the occlusion visible set: lazily fill any missing per-chunk
    /// connectivity (only floods chunks not yet classified — so a world that
    /// never activates occlusion never pays it), then BFS from the camera's
    /// chunk. Called only when the gate is active and the inputs changed.
    fn rebuild_occlusion(&mut self, origin: Coord) {
        let registry = &self.registry;
        for loaded in self.chunks.values_mut() {
            if loaded.connectivity.is_none() {
                loaded.connectivity = Some(Connectivity::compute(&loaded.chunk, |id| registry.is_solid(id)));
            }
        }
        self.occlusion
            .rebuild(origin, |c| self.chunks.get(&c).and_then(|l| l.connectivity));
    }
}

impl Render for World {
    fn render(&self, f: &mut Frame3D, cam: DVec3) {
        World::render(self, f, cam);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::registry::AIR;
    use crate::math::Aabb;
    use voxel_engine::{DVec3, Pass};

    /// A `Ready` state drawing a single opaque mesh `h` (the common test shape).
    fn ready(h: MeshHandle) -> MeshState {
        MeshState::from_upload(ByPass::from_fn(|p| (p == Pass::Opaque).then_some(h)))
    }
    /// The `ChunkMeshes` for a single opaque mesh `h`.
    fn meshes(h: MeshHandle) -> ChunkMeshes {
        match ready(h) {
            MeshState::Ready(m) => m,
            _ => unreachable!("opaque handle yields Ready"),
        }
    }

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
        let in_ground = Aabb::new(DVec3::new(8.5, 0.5, 8.5), DVec3::new(0.3, 0.3, 0.3));
        let in_sky = Aabb::new(DVec3::new(8.5, 40.0, 8.5), DVec3::new(0.3, 0.3, 0.3));
        let in_deep = Aabb::new(DVec3::new(8.5, -30.0, 8.5), DVec3::new(0.3, 0.3, 0.3));
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
            DVec3::new(15.9, 18.0, 15.9), // corner of four chunks
            DVec3::new(0.1, 21.5, 8.0),   // one X boundary
            DVec3::new(-3.2, 19.0, -16.4),
            DVec3::new(4.0, -1.0, 4.0),  // below the surface band: solid now
            DVec3::new(4.0, 15.9, 4.0),  // straddles a vertical chunk boundary
            DVec3::new(4.0, 200.0, 4.0), // unloaded high sky: air on both paths
        ] {
            let aabb = Aabb::new(center, DVec3::new(0.4, 0.9, 0.4));
            let reference = aabb.voxel_cells().any(|(x, y, z)| world.is_solid(x, y, z));
            assert_eq!(world.collides(&aabb), reference, "at {center:?}");
        }
    }

    #[test]
    fn column_is_layered_grass_dirt_stone() {
        let mut world = World::generate();
        let reg = world.registry();
        let (grass, dirt, stone) = (
            reg.id_by_name("Grass").unwrap(),
            reg.id_by_name("Dirt").unwrap(),
            reg.id_by_name("Stone").unwrap(),
        );

        // Biome dressing now varies the surface block (grass / sand / snow) and
        // band stone can carry ore flecks — those rules have their own tests in
        // generation.rs. Here we scan a small grid for a genuinely grass-topped
        // column whose shallow stone rolled clean, and check the usual layering.
        let (x, z, h) = (0..64)
            .flat_map(|x| (0..64).map(move |z| (x, z)))
            .find_map(|(x, z)| {
                let h = (0..96).rev().find(|&y| world.is_solid(x, y, z))?;
                (world.block_at(x, h, z) == grass && world.block_at(x, h - 3, z) == stone)
                    .then_some((x, z, h))
            })
            .expect("a grass-topped column with clean shallow stone near spawn");

        assert_eq!(world.block_at(x, h + 1, z), AIR);
        assert_eq!(world.block_at(x, h, z), grass);
        assert_eq!(world.block_at(x, h - 1, z), dirt);
        assert_eq!(world.block_at(x, h - 3, z), stone);
        // And no bottom anymore: the deep layer continues below y = 0 —
        // checked below the ore band (depth > 64), where stone is always
        // pure. That chunk sits outside the pregenerated region, so load it.
        world.ensure_data(World::chunk_of(x, h - 70, z));
        assert_eq!(world.block_at(x, h - 70, z), stone);
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
        world.center = Some(ChunkCoord::new(0, 0, 0)); // pretend the player streamed here
        let coord = ChunkCoord::new(0, 0, 0);
        let rev = world.chunks[&coord].rev;
        assert!(world.mesh_result_applies(coord, rev));

        // An edit bumps the rev: the snapshot a worker holds is now stale.
        world.set_block(3, 3, 3, AIR);
        assert!(!world.mesh_result_applies(coord, rev));

        // A stale landing is dropped and re-arms the fresh scan.
        world.pending_fresh.take();
        world.accept_mesh(coord, rev, new_chunk_mesh_data());
        assert!(world.upload_queue.is_empty(), "stale result never queues");
        assert!(world.pending_fresh.get(), "drop re-arms the scan");

        // A current-rev landing queues for upload.
        let rev = world.chunks[&coord].rev;
        world.accept_mesh(coord, rev, new_chunk_mesh_data());
        assert_eq!(world.upload_queue.len(), 1);
        world.upload_queue.clear();

        // Unloaded / out-of-range coords are rejected too — horizontally and
        // vertically (the volume is a cube, so the vertical bound equals the
        // horizontal one).
        assert!(!world.mesh_result_applies(ChunkCoord::new(99, 0, 99), 0));
        assert!(!world.mesh_result_applies(ChunkCoord::new(0, 99, 0), 0));
        let rv = world.view.vertical;
        assert!(!world.mesh_result_applies(ChunkCoord::new(0, rv + 1, 0), 0), "just past vertical range");
    }

    #[test]
    fn neighbour_edits_bump_the_bordering_chunks_rev() {
        let mut world = World::generate();
        // An edit at x == 0 of chunk (0, 0, 0) touches chunk (-1, 0, 0)'s border.
        let before = world.chunks[&ChunkCoord::new(-1, 0, 0)].rev;
        world.set_block(0, 5, 8, AIR);
        assert_eq!(world.chunks[&ChunkCoord::new(-1, 0, 0)].rev, before + 1, "border neighbour");
        assert_eq!(world.chunks[&ChunkCoord::new(0, 0, 0)].rev, 1, "edited chunk itself");
        assert_eq!(world.chunks[&ChunkCoord::new(1, 0, 0)].rev, 0, "far side untouched");

        // Vertical borders count too: an edit at y == 16 (bottom of chunk
        // layer 1) touches the chunk below. That cell is air here, so PLACE a
        // solid block — a real change (a no-op placement is now skipped whole).
        let stone = world.registry().id_by_name("Stone").unwrap();
        let below = world.chunks[&ChunkCoord::new(0, 0, 0)].rev;
        world.set_block(8, 16, 8, stone);
        assert_eq!(world.chunks[&ChunkCoord::new(0, 0, 0)].rev, below + 1, "chunk below bumped");
        assert_eq!(world.chunks[&ChunkCoord::new(0, 1, 0)].rev, 1, "edited vertical chunk");
    }

    #[test]
    fn landed_chunks_replay_edits_that_arrived_mid_flight() {
        let mut world = World::generate();
        world.center = Some(ChunkCoord::new(0, 0, 0));
        let coord = ChunkCoord::new(2, 0, 2);
        let (x, z) = (coord.x * CHUNK_SIZE as i32 + 3, coord.z * CHUNK_SIZE as i32 + 4);
        // Simulate the coord being in flight: no data yet, edit lands meanwhile
        // (recorded in the overlay only).
        world.chunks.remove(&coord);
        world.set_block(x, 5, z, AIR);
        // The worker's result was built before that edit existed.
        let raw = Chunk::new(coord.x, coord.y, coord.z, &world.generator);
        assert_ne!(raw.get_local(3, 5, 4), AIR, "terrain is solid there");
        world.pending_fresh.take();
        world.accept_chunk(coord, raw);
        assert_eq!(world.block_at(x, 5, z), AIR, "overlay replayed on landing");
        assert!(world.pending_fresh.get(), "new data re-arms the fresh scan");

        // Results for coords the world has moved past are discarded —
        // horizontally or vertically.
        let far = ChunkCoord::new(100, 0, 100);
        world.accept_chunk(far, Chunk::new(far.x, far.y, far.z, &world.generator));
        assert!(!world.chunks.contains_key(&far), "out-of-range chunk dropped");
        let high = ChunkCoord::new(0, 100, 0);
        world.accept_chunk(high, Chunk::new(high.x, high.y, high.z, &world.generator));
        assert!(!world.chunks.contains_key(&high), "out-of-height chunk dropped");
    }

    #[test]
    fn uniform_air_chunks_are_born_meshed() {
        let world = World::generate();
        // A sky chunk between the hills and the island band: uniform air,
        // meshed on arrival with no mesh and no worker job ever queued.
        let sky = &world.chunks[&ChunkCoord::new(0, 3, 0)];
        assert_eq!(sky.chunk.uniform(), Some(AIR));
        assert_eq!(sky.state, MeshState::Air, "uniform air is born Air, no mesh job");
        assert!(sky.state.live_meshes().is_none());
        // A ground chunk still goes through the normal mesh path.
        let ground = &world.chunks[&ChunkCoord::new(0, 0, 0)];
        assert_eq!(ground.state, MeshState::NeedsMesh, "dense terrain waits for a real mesh");
    }

    // === MeshState handle-ownership tests ===
    //
    // The GPU-touching write paths (mesh_chunk/unload_far/free_meshes/upload)
    // take a live `&mut Engine`, which is not headlessly constructible — so
    // these tests fabricate handles (`MeshHandle::from_raw_parts`), wrap them in
    // an `OwnedMesh` token, and drive the pure state-machine transitions that
    // gate every `eng.free_mesh` call. They check: the mesh is surfaced by
    // `draw_id` (or carried as `Dirty.prev`) exactly once, and moved out on the
    // transition that frees it.

    #[test]
    fn handle_is_tracked_exactly_once_across_edits() {
        // The single mesh must survive an edit and a re-edit without being
        // dropped or duplicated (no leak, no double-free). `invalidate` moves
        // the token rather than cloning or dropping it.
        let mut world = World::generate();
        let coord = ChunkCoord::new(0, 0, 0);
        let h = MeshHandle::from_raw_parts(42, 3);
        world.chunks.get_mut(&coord).unwrap().state = ready(h);

        world.set_block(3, 3, 3, AIR); // Ready(h) → Dirty{Some(h)} (not freed here)
        assert_eq!(world.chunks[&coord].state, MeshState::Dirty { prev: Some(meshes(h)) });

        world.set_block(4, 4, 4, AIR); // re-edit a still-Dirty chunk
        assert_eq!(
            world.chunks[&coord].state,
            MeshState::Dirty { prev: Some(meshes(h)) },
            "re-edit preserves the single handle (no leak, no duplicate)"
        );

        // `live_meshes` is the selector the render path uses: it must surface the
        // handle exactly once for the live-mesh states and nothing otherwise.
        // An edited-but-not-yet-remeshed chunk still draws its old handle here,
        // so an edit never blanks the chunk for a frame (no black frame).
        assert!(world.chunks[&coord].state.live_meshes().unwrap().draws(h));
        for s in [
            MeshState::Air,
            MeshState::NeedsMesh,
            MeshState::Meshing,
            MeshState::Dirty { prev: None },
        ] {
            world.chunks.get_mut(&coord).unwrap().state = s;
            let state = &world.chunks[&coord].state;
            assert!(state.live_meshes().is_none(), "nothing to free for {state:?}");
        }
    }

    #[test]
    fn edit_during_meshing_drops_the_stale_async_result() {
        // An edit while a fresh mesh job flies makes that job stale; its
        // result must be dropped by the rev check (the sync path remeshes).
        let mut world = World::generate();
        world.center = Some(ChunkCoord::new(0, 0, 0));
        let coord = ChunkCoord::new(0, 0, 0);
        world.chunks.get_mut(&coord).unwrap().state = MeshState::Meshing;
        let rev = world.chunks[&coord].rev;

        world.set_block(2, 2, 2, AIR); // Meshing → Dirty, rev bumped
        assert!(world.chunks[&coord].state.is_dirty(), "edit turns Meshing into Dirty");
        assert_ne!(world.chunks[&coord].rev, rev, "edit bumps rev, stranding the job");

        // The worker's result lands at the OLD rev: dropped, never uploaded. The
        // `Dirty` state (not a side set) is now the claim; the sync remesh owns it.
        world.pending_fresh.take();
        world.accept_mesh(coord, rev, new_chunk_mesh_data());
        assert!(world.upload_queue.is_empty(), "stale mesh result never queues");
        assert!(world.chunks[&coord].state.is_dirty(), "chunk stays Dirty for the sync remesh");
        assert!(world.pending_fresh.get(), "drop re-arms the fresh scan");
    }

    #[test]
    fn view_volume_is_isotropic() {
        // The streamed volume is a cube: the vertical radius tracks the
        // horizontal one exactly, on every render-distance setting. This is the
        // near-field thrash fix — a flat disk popped in-view chunks above/below.
        let mut world = World::generate();
        for view in [3, 4, 6, 8, 10, 20] {
            world.set_view_radius(view);
            assert_eq!(world.view.horizontal, view, "view {view}");
            assert_eq!(world.view.vertical, world.view.horizontal, "cube at view {view}");
        }
    }

    #[test]
    fn streaming_order_weights_vertical_double() {
        let c = ChunkCoord::new(0, 0, 0);
        assert_eq!(World::order(ChunkCoord::new(4, 0, 0), c), 4);
        assert_eq!(World::order(ChunkCoord::new(0, 2, 0), c), 4, "2 layers up ranks like 4 rings out");
        assert!(
            World::order(ChunkCoord::new(0, 3, 0), c) > World::order(ChunkCoord::new(5, 0, 0), c),
            "lateral terrain streams before the sky"
        );
    }

    #[test]
    fn view_radius_clamps_and_flags_streaming() {
        let mut world = World::generate();
        assert_eq!(world.view_radius(), DEFAULT_VIEW_RADIUS);
        world.set_view_radius(99);
        assert_eq!(world.view_radius(), 20);
        world.set_view_radius(1);
        assert_eq!(world.view_radius(), 3);
        // The change must force the next stream to rescan.
        assert!(world.pending_fresh.get());
        assert_eq!(world.center, None);
    }

    #[test]
    fn queued_shrink_survives_an_intervening_grow() {
        // A shrink then a grow before the next stream: the shrink flag must
        // still be raised (the old `= shrunk` clobbered it back to false, so
        // the freed-mesh pass never ran and stale meshes stayed drawn).
        let mut world = World::generate();
        world.set_view_radius(4); // shrink from default 6: raises radius_shrunk
        world.set_view_radius(8); // grow: must NOT lower the queued shrink
        assert!(world.radius_shrunk.get(), "grow must not clobber a queued shrink");
        assert!(world.radius_shrunk.take());
        assert!(!world.radius_shrunk.get(), "take consumes it");
    }

    #[test]
    fn invalidate_carries_or_clears_the_owned_mesh() {
        let h = MeshHandle::from_raw_parts(5, 2);
        // Ready → Dirty{Some}: the mesh is CARRIED (moved), never freed.
        let mut s = ready(h);
        s.invalidate();
        assert_eq!(s, MeshState::Dirty { prev: Some(meshes(h)) });
        // Re-invalidating a Dirty{Some} keeps the same single token.
        s.invalidate();
        assert_eq!(s, MeshState::Dirty { prev: Some(meshes(h)) });
        // Every handle-less state → Dirty{None}.
        for empty in [
            MeshState::Air,
            MeshState::NeedsMesh,
            MeshState::Meshing,
            MeshState::Dirty { prev: None },
        ] {
            let mut s = empty;
            s.invalidate();
            assert_eq!(s, MeshState::Dirty { prev: None });
        }
    }

    #[test]
    fn from_upload_and_into_owned_agree_on_handle_ownership() {
        let h = MeshHandle::from_raw_parts(9, 1);
        // A handle uploads to Ready, draws it, and yields it back exactly once.
        let state = ready(h);
        assert!(state.live_meshes().unwrap().draws(h));
        assert!(matches!(state.into_owned(), Some(m) if m.draws(h)));
        // No handle → Air, which owns and draws nothing.
        let air = MeshState::from_upload(ByPass::from_fn(|_| None));
        assert_eq!(air, MeshState::Air);
        assert!(air.live_meshes().is_none());
        assert!(air.into_owned().is_none());
    }

    #[test]
    fn edit_raises_pending_dirty_and_enters_the_dirty_fiber() {
        // There's no separate `dirty` side set: an edit raises the
        // `pending_dirty` hint and the chunk enters the `Dirty` state directly.
        let mut world = World::generate();
        assert!(!world.pending_dirty.get(), "a clean world has no dirty hint");
        let coord = ChunkCoord::new(0, 0, 0);
        world.set_block(1, 1, 1, AIR);
        assert!(world.pending_dirty.get(), "an edit raises the dirty hint");
        let in_fiber = world.chunks.iter().any(|(&c, l)| c == coord && l.state.is_dirty());
        assert!(in_fiber, "the edited chunk is in the Dirty fiber");
    }

    #[test]
    fn generating_never_shadows_loaded_data() {
        // A coord claimed as a live generate job must be absent — a loaded
        // coord left in `generating` would block its own regeneration.
        let mut world = World::generate();
        let absent = ChunkCoord::new(500, 0, 500);
        assert!(!world.chunks.contains_key(&absent));
        world.generating.insert(absent);
        world.debug_assert_liveness(); // holds: the claim shadows no data
    }

    #[test]
    fn tiles_over_the_full_res_box_are_occluded_and_far_ones_are_not() {
        // The occlusion-skip is 3-D *containment*: a tile whose whole footprint
        // sits inside the full-res slab (all three axes) is hidden; a tile the
        // slab only partly covers still draws (else its uncovered part is a hole).
        // A level-2 tile spans 4 chunk columns; default view radius 6, and the
        // volume is a cube, so the box is Y in [-6, 6] chunk layers.
        let mut world = World::generate();
        world.center = Some(ChunkCoord::new(0, 0, 0));
        let lod = lod::Lod(2);
        // Tile (0,0,0): chunk cols/layers 0..=3 fully in [-6, 6] on every axis → occluded.
        assert!(world.tile_occluded(Tile { lod, x: 0, y: 0, z: 0 }));
        // Tile (1,0,0) covers columns 4..=7 — column 7 is past the slab, so it is
        // only partly covered → it must draw (the boundary-gap regression).
        assert!(!world.tile_occluded(Tile { lod, x: 1, y: 0, z: 0 }));
        // Tile (2,0,0) covers columns 8..=11 — clear of [-6, 6] entirely → drawn.
        assert!(!world.tile_occluded(Tile { lod, x: 2, y: 0, z: 0 }));
        // A tile stacked above (layers 4..=7) pokes past the vertical slab → drawn.
        assert!(!world.tile_occluded(Tile { lod, x: 0, y: 1, z: 0 }));
        // Player flown high: the ground tile now sits far below the vertical box,
        // so the chunks no longer cover it → it must draw.
        world.center = Some(ChunkCoord::new(0, 100, 0));
        assert!(!world.tile_occluded(Tile { lod, x: 0, y: 0, z: 0 }));
        // No centre yet ⇒ nothing is occluded (nothing has streamed).
        world.center = None;
        assert!(!world.tile_occluded(Tile { lod, x: 0, y: 0, z: 0 }));
    }
}
