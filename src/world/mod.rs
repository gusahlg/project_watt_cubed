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
pub mod pyramid;
pub mod section;

mod edits;
mod quadtree;
mod query;
mod streaming;

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::sync::Arc;

use voxel_engine::{DVec3, Engine, Frame3D, MeshHandle, Vec3};

use crate::block::registry::{BlockId, BlockRegistry, HotTables};
use crate::coord::{ByPass, ChunkBox, ChunkCoord};
use crate::render::Render;
use chunk::{CHUNK_SIZE, Chunk};
use generation::{SineHills, TerrainGenerator};
use light::LightGrid;
use mesh::{ChunkMeshData, new_chunk_mesh_data};
use section::{SectionMeshData, SectionPos};

/// Default number of chunk rings meshed and drawn around the player.
const DEFAULT_VIEW_RADIUS: i32 = 6;
/// The range a runtime render-distance change is clamped to. Shared: the
/// settings model and the settings menu stepper clamp to the same bounds.
pub const VIEW_RADIUS_RANGE: std::ops::RangeInclusive<i32> = 3..=20;
/// One extra shell of *data* (not meshed) in all three axes so edge chunks can
/// cull faces against their neighbours without re-meshing when those
/// neighbours later load.
const DATA_MARGIN: i32 = 1;
/// How far past the view radius chunks survive horizontally before they are
/// freed, so walking back and forth across the boundary doesn't thrash.
const UNLOAD_MARGIN: i32 = 3;
/// The vertical unload hysteresis. Smaller than [`UNLOAD_MARGIN`] because the
/// vertical streaming radius is itself smaller (see [`ViewVolume`]).
const UNLOAD_MARGIN_V: i32 = 2;
/// How many finished worker meshes may be uploaded to the GPU per stream —
/// the upload is the only part of the async path the render thread still pays.
const UPLOAD_BUDGET: usize = 4;
/// How many *dirty* (edited) chunks may remesh per frame. Processed nearest
/// first, so a locally broken block still vanishes the same frame while a
/// multiplayer join snapshot flood spreads over a few frames instead of one hitch.
const DIRTY_BUDGET: usize = 8;
/// How many chunks the occlusion rebuild may flood-fill (`Connectivity::compute`)
/// per frame. A boundary cross can newly load a whole shell of unclassified
/// chunks; capping the fill keeps a cross from BFS-flooding O(cube) in one frame.
/// On a partial fill the `occlusion_dirty` flag is left set so the rebuild
/// re-runs next frame — convergence over frames, no correctness cost.
const OCCLUSION_FILL_BUDGET: usize = 64;
/// The seed a default (`generate`) world uses when none is chosen.
pub const DEFAULT_SEED: i64 = 1;
/// Column-section uploads per frame; separate budget so a world-entry
/// flood doesn't starve chunk uploads.
const SECTION_UPLOAD_BUDGET: usize = 2;

/// A chunk-coordinate map key. The [`ChunkCoord`] newtype owns the
/// `chunk * CHUNK_SIZE + local` relationship (see [`crate::coord`]); the
/// `Coord` alias keeps the shorter name the `world` module already used.
use crate::coord::ChunkCoord as Coord;
use connectivity::{Connectivity, Occlusion};

/// The streamed chunk volume around the player: a horizontal ring radius and a
/// (smaller) vertical layer radius, in chunks. **Anisotropic**: interesting
/// terrain is mostly lateral, so a full cube would load a tall column of empty
/// sky and deep rock that never produces drawable geometry — tripling the
/// loaded set (and every O(loaded) streaming pass) for nothing on screen. The
/// vertical radius is derived from the horizontal one ([`vertical_for`]), tall
/// enough to keep the ground under vertical movement (jumping, cliffs, flight)
/// without paying for the whole render sphere.
///
/// [`vertical_for`]: ViewVolume::vertical_for
#[derive(Clone, Copy)]
pub(in crate::world) struct ViewVolume {
    horizontal: i32,
    vertical: i32,
}

impl ViewVolume {
    /// Vertical streaming radius for a horizontal view radius: half of it,
    /// clamped to 2..=5. Keeps the streamed volume a flat box, not a cube.
    fn vertical_for(horizontal: i32) -> i32 {
        (horizontal / 2).clamp(2, 5)
    }
    /// The streamed volume for a horizontal view radius, with the vertical
    /// radius derived from it.
    fn view(horizontal: i32) -> Self {
        Self { horizontal, vertical: Self::vertical_for(horizontal) }
    }
    /// The mesh box grown by `dh` rings horizontally and `dv` layers vertically.
    fn box_at(self, center: Coord, dh: i32, dv: i32) -> ChunkBox {
        ChunkBox::new(center, self.horizontal + dh, self.vertical + dv)
    }
    /// Chunks meshed and drawn around `center`.
    fn mesh(self, center: Coord) -> ChunkBox {
        self.box_at(center, 0, 0)
    }
    /// The mesh box plus one [`DATA_MARGIN`] shell of voxel data, so edge chunks
    /// cull against neighbours that are loaded but unmeshed.
    fn data(self, center: Coord) -> ChunkBox {
        self.box_at(center, DATA_MARGIN, DATA_MARGIN)
    }
    /// The mesh box plus the unload hysteresis, past which chunks free. Vertical
    /// uses the tighter [`UNLOAD_MARGIN_V`] to match the tighter vertical radius.
    fn unload(self, center: Coord) -> ChunkBox {
        self.box_at(center, UNLOAD_MARGIN, UNLOAD_MARGIN_V)
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
    /// Shared voxel storage by refcount; edits via `Arc::make_mut`.
    chunk: Arc<Chunk>,
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
    /// The chunk's settled light grid, published ([`settle_light`](World::settle_light))
    /// either analytically (the trivial fast path) or when a worker-pool flood
    /// lands (the flood runs off-thread, decoupled from meshing). Read
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
    /// Wrap freshly uploaded per-pass handles, same "≥1 present" rule as [`Self::new`].
    pub(in crate::world) fn from_upload_handles(handles: ByPass<Option<MeshHandle>>) -> Option<Self> {
        Self::new(ByPass::from_fn(|p| handles[p].map(OwnedMesh::new)))
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

/// A section is either meshing or ready. Blocks are pre-positioned at upload,
/// so draw needs only camera-relative offset arithmetic.
pub(in crate::world) enum SectionState {
    Meshing,
    /// Each block is positioned at a world min-corner, drawn at `scale = cell`.
    Ready { cell: f32, blocks: Vec<([f64; 3], ChunkMeshes)> },
}

impl SectionState {
    /// Upload block meshes with pre-baked world positions so draw needs only
    /// the camera-relative subtract. Empty sections upload to no handles.
    fn from_upload(pos: SectionPos, meshes: SectionMeshData, eng: &mut Engine) -> SectionState {
        let cell = pos.cell_size();
        let (mx, mz) = (pos.min_x() as f64, pos.min_z() as f64);
        let mut blocks = Vec::new();
        for (block_origin, data) in meshes {
            let handles = ByPass::from_fn(|p| eng.upload_mesh(&data[p]));
            if let Some(meshes) = ChunkMeshes::from_upload_handles(handles) {
                // Block origin is in CELLS; the section floor is world-Y 0.
                let wmin = [
                    mx + block_origin.x as f64 * cell as f64,
                    block_origin.y as f64 * cell as f64,
                    mz + block_origin.z as f64 * cell as f64,
                ];
                blocks.push((wmin, meshes));
            }
        }
        SectionState::Ready { cell: cell as f32, blocks }
    }

    fn is_ready(&self) -> bool {
        matches!(self, SectionState::Ready { .. })
    }
    /// Draw all blocks; the dither band in the shader hands the near ground to
    /// full-res chunks, so no depth bias is needed.
    fn draw(&self, f: &mut Frame3D, cam: DVec3) {
        if let SectionState::Ready { cell, blocks } = self {
            for (wmin, meshes) in blocks {
                let offset = (DVec3::new(wmin[0], wmin[1], wmin[2]) - cam).as_vec3();
                meshes.draw(f, offset, *cell);
            }
        }
    }
    fn free(self, eng: &mut Engine) {
        if let SectionState::Ready { blocks, .. } = self {
            for (_, meshes) in blocks {
                meshes.free(eng);
            }
        }
    }
}

/// Mesh-lifecycle state of a loaded chunk. Owns GPU mesh via [`OwnedMesh`] token;
/// `rev` bumped when mesh inputs stale. Handle ownership rides the state machine:
/// moves on edit (`Dirty.prev`) or frees on unload/shrink/remesh.
///
/// "Needs a mesh" and "a build job is outstanding" are orthogonal, so the second
/// is a `building` refinement of `NeedsMesh` — NOT a separate `Meshing` state
/// mutually exclusive with it. That fusion was the old wedge: an async result
/// that went stale purely because the view moved (no edit, no unload — the one
/// transition carrying no event) had no way to un-claim a `Meshing` state, so
/// the chunk stuck claimed-but-never-ready forever. Now the claim is a bool that
/// only `Air`/`Ready` structurally cannot carry, and every result-consumption
/// path clears it, so the wedge is unrepresentable.
#[derive(Debug, PartialEq, Eq)]
enum MeshState {
    /// Uniform-air, born meshed: nothing to draw, no worker job ever queued.
    Air,
    /// Dense data with no mesh yet. `building` is the in-flight claim: `true`
    /// once a fresh mesh job is outstanding on the worker pool (owns no handle;
    /// `rev` on `Loaded` referees its result), held until the budgeted upload
    /// resolves or the result is dropped. A `building` chunk still draws
    /// nothing and is still "needs mesh" — it is just also claimed.
    NeedsMesh { building: bool },
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
    /// chunk keeps its `prev`; handle-less states (incl. a `building` chunk,
    /// whose in-flight claim is dropped — the sync remesh takes over and the
    /// orphan async result is refereed out by `rev`) -> `Dirty{None}`.
    fn invalidate(&mut self) {
        let prev = std::mem::replace(self, MeshState::NeedsMesh { building: false }).into_owned();
        *self = MeshState::Dirty { prev };
    }
    /// Release the in-flight mesh claim if this chunk is still awaiting its
    /// build. A no-op once the chunk has moved on (`Dirty` via an edit,
    /// `Ready`/`Air` via a prior consume): those states carry no claim. Called
    /// at every mesh-result-consumption site whose result did NOT apply, so a
    /// stale result (view moved, chunk left the box) can never wedge the claim.
    fn release_build(&mut self) {
        if let MeshState::NeedsMesh { building } = self {
            *building = false;
        }
    }
    fn is_needs_mesh(&self) -> bool {
        matches!(self, MeshState::NeedsMesh { .. })
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
    /// Monotone counter bumped on every recorded edit; the autosaver compares
    /// it against the last written generation to know the world is dirty.
    pub(crate) edit_generation: u64,
    /// Last chunk centre; `None` forces a full stream pass. Streams only react to boundary crosses.
    center: Option<Coord>,
    /// The streamed chunk volume (horizontal ring + a smaller, derived vertical
    /// layer radius). Its horizontal radius is the render-distance setting
    /// (clamped to [`VIEW_RADIUS_RANGE`]); see [`ViewVolume`].
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
    /// Chunks needing a *fresh* mesh (the [`MeshLane`] seed set — replaces the
    /// old whole-map rescan `pending_fresh` armed). Seeded on load (self + 6
    /// neighbours), on a light publish that moved a border, and on an
    /// accept_mesh stale drop. Drained nearest-first by the mesh lane, so the
    /// enqueue scan is O(shell) not O(cube).
    mesh_worklist: FastSet<Coord>,
    /// Whether the [`LightLane`] still has seeds/in-flight to drain (its
    /// `pending` gate — the [`LaneSpec::pending`] accessor). Raised when the
    /// worklist or in-flight set is non-empty, cleared when both drain.
    light_pending: Sticky,
    /// Chunks needing light settling (budgeted, seeded on load/edit/border moves).
    light_worklist: FastSet<Coord>,
    /// Chunks with a light-settle job in flight on the worker pool. A settle is
    /// claimed out of `light_worklist` at submit and released here when its grid
    /// lands, so at most one flood per chunk is in flight and the mesh gate
    /// ([`light_ready`](World::light_ready)) treats an in-flight chunk as not yet
    /// settled. An edit landing mid-flight re-seeds the worklist, so the next
    /// stream resubmits with the fresh voxels once the current job drains.
    light_inflight: FastSet<Coord>,
    /// Settled light grids landed from the worker pool, awaiting budgeted
    /// application ([`LightLane::integrate`] → [`settle_light`]). The light
    /// *drain* lane previously had no per-frame budget: ~80 floods could land
    /// and all apply in one frame. Buffering here and applying ≤
    /// a per-frame time budget ([`pipeline::LIGHT_APPLY_BUDGET`]) caps that
    /// main-thread bookkeeping spike; leftovers apply next frame (each grid is absolute).
    light_apply_queue: VecDeque<(Coord, light::LightGrid)>,
    /// Light-gate degraded-path state defined in [`streaming`]: per-chunk
    /// timers for how long a fresh mesh has waited on neighbour light, plus the set
    /// of chunks currently showing a degraded (known-not-final) mesh awaiting relight.
    /// Kept in one struct so the feature's footprint on `World` is a single field.
    light_gate: streaming::LightGate,
    /// Skylight ceiling per `(x, z)` chunk column — the surface heightmap the
    /// settle pass seeds skylight from. A pure generator function (independent of
    /// y and of edits), so it is computed once per column and reused across every
    /// vertical chunk and every re-settle instead of re-sampling 256 noise columns
    /// per settle. Pruned when a column fully unloads.
    ceilings: FastMap<(i32, i32), light::CeilingWindow>,
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
    /// Manual occlusion override (from [`RenderConfig::occlusion`]), on by default when GPU-bound signal unavailable.
    occlusion_forced: bool,
    /// Cross-chunk lighting enable flag. Driven by the `lighting` graphics
    /// setting via [`set_lighting`](World::set_lighting); the initial value only
    /// governs pre-`enter_game` generation and is overridden on world entry.
    lighting: bool,
    /// Generation stamp for asynchronous light jobs. Toggling lighting advances
    /// it so a result captured under the previous mode cannot publish later.
    light_epoch: u32,
    /// LOD2 column-section far field enable.
    lod2: bool,
    /// LOD pyramid config with `unit` in metres.
    section_pyramid: pyramid::PyramidCfg,
    /// Loaded sections.
    sections: FastMap<SectionPos, SectionState>,
    /// Finished section meshes awaiting budgeted upload.
    section_upload_queue: VecDeque<(SectionPos, SectionMeshData)>,
    /// Whether desired sections still need enqueueing (budget spreads a flood).
    pending_sections: Sticky,
    /// Sections invalidated by edits, freed and re-admitted from the generator.
    dirty_sections: FastSet<SectionPos>,
    /// Visible sections (covering-resolved).
    section_visible: Vec<SectionPos>,
}

impl World {
    /// A fresh world for `seed`, with the region around the origin pre-generated
    /// (data only — no GPU) so spawning and headless queries work before the first
    /// [`stream`](Self::stream).
    pub fn new(seed: i64) -> Self {
        Self::with_config(seed, crate::render_config::RenderConfig::default())
    }

    /// A fresh world for `seed` with explicit render config — the sole source
    /// for lod2/occlusion gates. [`new`](Self::new) is this with defaults.
    ///
    /// [`RenderConfig`]: crate::render_config::RenderConfig
    pub fn with_config(seed: i64, render: crate::render_config::RenderConfig) -> Self {
        let registry = BlockRegistry::with_builtins();
        let generator = SineHills::new(&registry, 20.0, seed);
        // The section ladder's innermost ring begins where the full-res box ends,
        // so its `unit` is the render distance in metres.
        let unit = (DEFAULT_VIEW_RADIUS * CHUNK_SIZE as i32) as f32;
        let lod2 = render.lod2;
        let mut world = Self {
            registry,
            generator,
            chunks: FastMap::default(),
            ceilings: FastMap::default(),
            edits: FastMap::default(),
            edit_generation: 0,
            center: None,
            view: ViewVolume::view(DEFAULT_VIEW_RADIUS),
            pending_fresh: Sticky::raised(),
            pending_dirty: Sticky::default(),
            radius_shrunk: Sticky::default(),
            scratch: new_chunk_mesh_data(),
            tables: crate::derived::Derived::default(),
            workers: None,
            generating: FastSet::default(),
            upload_queue: VecDeque::new(),
            mesh_worklist: FastSet::default(),
            light_pending: Sticky::default(),
            light_worklist: FastSet::default(),
            light_inflight: FastSet::default(),
            light_apply_queue: VecDeque::new(),
            light_gate: streaming::LightGate::default(),
            done_scratch: Vec::new(),
            textures_built: 0,
            occlusion: Occlusion::default(),
            occlusion_dirty: Sticky::default(),
            occlusion_active: false,
            occlusion_forced: render.occlusion,
            lighting: true,
            light_epoch: 0,
            lod2,
            section_pyramid: pyramid::PyramidCfg::sections(unit),
            sections: FastMap::default(),
            section_upload_queue: VecDeque::new(),
            pending_sections: Sticky::default(),
            dirty_sections: FastSet::default(),
            section_visible: Vec::new(),
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
    /// own regeneration and never re-enter the mesh pipeline. Test-only: the
    /// invariant is exercised directly without adding release work.
    #[cfg(any(debug_assertions, test))]
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
        let mut live = 0u64;
        for (&coord, loaded) in &self.chunks {
            if self.occlusion_active && !self.occlusion.is_visible(coord) {
                continue;
            }
            if let Some(meshes) = loaded.state.live_meshes() {
                live += 1;
                let origin =
                    DVec3::new(coord.x as f64 * s, coord.y as f64 * s, coord.z as f64 * s);
                // Full-res chunks are unit-scale: their local 0..=16 coords are
                // already metres. LOD sections pass 2^k here (see `SectionState::draw`).
                meshes.draw(f, (origin - cam).as_vec3(), 1.0);
            }
        }
        // Set-size gauges: `list.world` cost is O(these). A spike here localizes a
        // regression to a grown set (view volume / section frontier), not a
        // per-item slowdown. See `profile::Gauge`.
        use voxel_engine::profile::{gauge, Gauge};
        gauge(Gauge::WorldChunks, self.chunks.len() as u64);
        gauge(Gauge::WorldChunksLive, live);

        // Layer 2 — section far field: the covering-resolved visible set built
        // in `stream`. The shader discards LOD-section fragments inside the
        // full-res radius (chunks own the near ground) and fades in the sections
        // beyond it. Skipped at zero cost if the section lane is inactive.
        f.set_lod_clip((self.view.horizontal * CHUNK_SIZE as i32) as f32);
        for pos in &self.section_visible {
            if let Some(state) = self.sections.get(pos) {
                state.draw(f, cam);
            }
        }
    }

    /// Whether the occlusion gate is active this frame — the adaptive decision
    /// to spend CPU culling in order to save GPU draw time. Occlusion only pays
    /// off when the frame is GPU/overdraw-bound; that signal lives in the engine
    /// (GPU timestamps) and isn't wired yet, so this is off unless force-enabled
    /// via [`RenderConfig::occlusion`](crate::render_config::RenderConfig). When the
    /// engine exposes the signal, OR it in here and every GPU-side optimisation can
    /// share this one gate.
    fn occlusion_enabled(&self) -> bool {
        self.occlusion_forced
    }

    /// Rebuild the occlusion visible set: lazily fill any missing per-chunk
    /// connectivity (only floods chunks not yet classified — so a world that
    /// never activates occlusion never pays it), then BFS from the camera's
    /// chunk. Called only when the gate is active and the inputs changed.
    fn rebuild_occlusion(&mut self, origin: Coord) {
        let registry = &self.registry;
        // Cap the per-frame connectivity flood: a boundary cross can newly load a
        // whole shell of unclassified chunks, and filling them all in one frame
        // is the O(cube) spike. Stop at the budget and re-arm below so the fill
        // resumes next frame (partial classification only under-occludes — draws
        // a few extra chunks — never a hole).
        let mut filled = 0;
        let mut capped = false;
        for loaded in self.chunks.values_mut() {
            if loaded.connectivity.is_none() {
                if filled >= OCCLUSION_FILL_BUDGET {
                    capped = true;
                    break;
                }
                // Sightlines pass through anything not opaque — water/glass are
                // solid (collision) but see-through, so they must NOT seal chunks
                // behind them, or terrain under water gets occlusion-culled.
                loaded.connectivity = Some(Connectivity::compute(&loaded.chunk, |id| registry.is_opaque(id)));
                filled += 1;
            }
        }
        // Leave `occlusion_dirty` set on a partial fill so the gate re-runs.
        if capped {
            self.occlusion_dirty.set();
        }
        // A *loaded* chunk whose connectivity the budget hasn't reached yet
        // defaults to OPEN — drawn and passed through — so a partial fill only
        // *weakens* the cull (temporary over-draw) and never punches a hole by
        // culling a visible chunk. `None` stays reserved for genuinely unloaded
        // chunks, which bound the BFS frontier.
        self.occlusion.rebuild(origin, |c| {
            self.chunks.get(&c).map(|l| l.connectivity.unwrap_or(Connectivity::OPEN))
        });
    }
}

// ---------------------------------------------------------------------------
// Streaming lanes — the unified budgeted enqueue/integrate spine.
// ---------------------------------------------------------------------------
//
// Every async streaming lane (fresh chunk meshing, far LOD sections,
// cross-chunk light) is the same shape: gather candidates near the player,
// drop the ones already in flight, order nearest-first, submit up to a
// per-frame budget to the worker pool, and integrate finished results. The
// lanes differed only in *where their state lives*, so that state is
// reached through a [`LaneSpec`] of accessors and the loop is written once
// ([`lane_enqueue`]/[`lane_integrate`]).
//
// Each lane is handed a per-frame time [`Deadline`](pipeline::Deadline) at its
// enqueue call site (minted fresh from its `pipeline::*_BUDGET` const), checked
// BETWEEN admitted items — the time-based admission control that replaced the
// old integer count budgets (a burst of cheap items and a burst of expensive
// ones no longer share one integer, and no lane can flood a frame unbounded).

/// How a lane names the work it wants to do this frame. A *geometry* lane
/// derives its keys from the player centre each frame (sections: a desired
/// covering frontier minus what's already loaded). A *worklist* lane reads an explicit seed
/// set accumulated on the `World` (mesh, light: coords poked dirty by loads and
/// edits) — [`LaneSpec::seed_set`] returns `Some` for exactly those.
pub(in crate::world) enum Candidates<K> {
    /// The full candidate key list, recomputed from the centre this frame.
    Geometry(Vec<K>),
    /// Read the lane's seed set (`LaneSpec::seed_set`) for candidates.
    Worklist,
}

/// One streaming lane. Zero-sized marker types (`MeshLane`, …) implement it;
/// all mutable state lives on [`World`] behind these accessors, so the lane
/// itself carries nothing and the generic loop stays allocation-free.
pub(in crate::world) trait LaneSpec {
    /// The lane's work key (a chunk `Coord`, a `SectionPos`).
    /// The loop sorts by the [`order`](LaneSpec::order) metric, so the key
    /// itself needs only `Copy + Eq + Hash` (set membership + move).
    type Key: Copy + Eq + Hash;

    /// Forward-progress floor: admissions guaranteed per call BEFORE the frame
    /// deadline is consulted (see the rationale in [`lane_enqueue`]). REQUIRED
    /// per lane — a floor sized for cheap admits (light seeding) would repeal
    /// the time budget for expensive ones (mesh snapshot captures), so each
    /// lane states its own.
    const MIN_ADMIT: usize;

    /// The candidate keys for this frame (see [`Candidates`]).
    fn candidates(world: &World, center: Coord) -> Candidates<Self::Key>;
    /// The worklist seed set, for worklist lanes (`None` for geometry lanes).
    fn seed_set(world: &mut World) -> Option<&mut FastSet<Self::Key>>;
    /// This lane's raise-then-consume "has pending work" gate.
    fn pending(world: &mut World) -> &mut Sticky;
    /// Nearest-first metric for `key` relative to `center` (lower = sooner).
    fn order(center: Coord, key: Self::Key) -> i32;
    /// Whether `key` already has a job in flight (skip re-submitting it).
    fn in_flight(world: &World, key: Self::Key) -> bool;
    /// Whether `key` may be submitted yet (data/light gates). Default: always.
    fn ready(world: &World, key: Self::Key) -> bool {
        let _ = (world, key);
        true
    }
    /// Squared euclidean METRES from `key`'s world-space centre to the player,
    /// for a far (distance-ordered) lane — the [`FarQueue`](pipeline::FarQueue)
    /// key. `None` (the default) marks a near lane, which submits FIFO.
    fn dist2(world: &World, center: Coord, key: Self::Key) -> Option<u64> {
        let _ = (world, center, key);
        None
    }
    /// Build the worker job for `key`, or `None` to drop it (unloaded/covered).
    /// Takes `&mut World` so a lane may warm a cache while snapshotting (light
    /// warms `ceilings` via `capture_ceiling`); it must not mutate lane state.
    fn submit(world: &mut World, key: Self::Key) -> Option<pipeline::Job>;
    /// Mark `key` in flight: remove it from the seed set (if any) and claim it
    /// (a state transition or an in-flight-set insert), so it isn't re-submitted.
    fn claim(world: &mut World, key: Self::Key);
    /// Fold a finished result back into the world (upload a mesh, publish light).
    fn integrate(world: &mut World, done: pipeline::Done);
}

/// The unified enqueue loop: gather → drop in-flight/unready → nearest-first →
/// submit until the frame `deadline` expires (checked between admitted items,
/// never mid-item) → claim; clear `pending` once the whole ready backlog drained.
pub(in crate::world) fn lane_enqueue<S: LaneSpec>(
    world: &mut World,
    center: Coord,
    deadline: pipeline::Deadline,
) {
    if !S::pending(world).get() {
        return;
    }
    let worklist = S::seed_set(world).is_some();
    let candidates: Vec<S::Key> = match S::candidates(world, center) {
        Candidates::Geometry(v) => v,
        Candidates::Worklist => {
            S::seed_set(world).map(|s| s.iter().copied().collect()).unwrap_or_default()
        }
    };
    // ONE `ready()` pass: partition into actionable (ready, not in flight) and
    // blocked. For a WORKLIST lane, evict the blocked ones from the seed set —
    // a worklist lane guarantees it re-seeds a key on its unblock event
    // (`store_chunk` seeds a chunk's neighbours when data lands; `settle_light`
    // seeds a chunk + moved-border neighbours when light lands), so a blocked
    // seed is re-added exactly when it becomes actionable. Persisting it instead
    // would force an O(accumulated backlog) rescan every frame — the `stream.mesh`
    // spike. Evicting makes the per-frame cost O(fresh seeds this frame).
    let mut ready_keys = Vec::new();
    let mut blocked = Vec::new();
    for k in candidates {
        if S::in_flight(world, k) {
            continue;
        }
        if S::ready(world, k) {
            ready_keys.push(k);
        } else if worklist {
            blocked.push(k);
        }
    }
    if worklist {
        if let Some(set) = S::seed_set(world) {
            for k in &blocked {
                set.remove(k);
            }
        }
    }
    ready_keys.sort_by_key(|&k| S::order(center, k));
    // Admission is time-gated: check the deadline BETWEEN items (never abort an
    // admitted item mid-work). `exhausted` stays true only if every ready key
    // was processed before the clock ran out.
    //
    // Forward-progress floor: the per-frame gather/partition/sort above is O(worklist)
    // and is charged against the same `deadline`. For a large worklist under a small
    // budget (e.g. the 1 ms light lane in a debug build), that setup alone can exhaust
    // the budget before the FIRST submit — starving the lane to zero admissions every
    // frame, forever (the seed set never shrinks). Guaranteeing at least
    // `S::MIN_ADMIT` admissions per call makes the seed set strictly shrink so
    // the lane always converges; the deadline then gates only ADDITIONAL
    // admissions.
    let mut exhausted = true;
    let mut admitted = 0usize;
    for key in ready_keys {
        if admitted >= S::MIN_ADMIT && deadline.expired() {
            exhausted = false;
            break;
        }
        let Some(job) = S::submit(world, key) else {
            // Unloaded/covered: drop the stale seed so it isn't retried forever.
            if let Some(set) = S::seed_set(world) {
                set.remove(&key);
            }
            continue;
        };
        // Far (distance-ordered) lanes push to the FarQueue keyed by dist²; near
        // lanes keep their FIFO. A closed pool declines — leave the seed to retry.
        // Compute dist² (immutable borrow) before taking the workers pool (mutable).
        let far = S::dist2(world, center, key);
        let workers = world.workers.get_or_insert_with(|| {
            pipeline::Workers::spawn(pipeline::Workers::default_threads())
        });
        let accepted = match far {
            Some(dist2) => workers.submit_far(job, dist2),
            None => workers.submit(job),
        };
        if accepted {
            S::claim(world, key);
            admitted += 1;
        } else {
            // The pool declined (shutting down, or the far queue is at its
            // admission cap). Nothing later in this pass can be admitted either
            // — stop instead of building and discarding a job per remaining
            // key. The unclaimed keys stay candidates and retry next frame.
            exhausted = false;
            break;
        }
    }
    // Clear the gate once the whole ready backlog submitted AND no seeds remain
    // (only ready-but-time-sliced seeds can remain now — blocked ones were evicted).
    let drained = exhausted && S::seed_set(world).map_or(true, |set| set.is_empty());
    if drained {
        S::pending(world).take();
    }
}

/// Fold one finished [`pipeline::Done`] back into the world via its lane. The
/// caller picks `S` by matching the `Done` variant; this is the single dispatch
/// point that replaced the four ad-hoc drain arms.
pub(in crate::world) fn lane_integrate<S: LaneSpec>(world: &mut World, done: pipeline::Done) {
    S::integrate(world, done);
}

/// Squared euclidean metres from the player-chunk centre to a world-space point,
/// the [`FarQueue`](pipeline::FarQueue) distance key. The player position is
/// taken at chunk-centre granularity — the same granularity the lane's `order`
/// metric already uses.
fn player_dist2(center: Coord, wx: i64, wy: i64, wz: i64) -> u64 {
    let s = CHUNK_SIZE as i64;
    let half = s / 2;
    let (px, py, pz) = (center.x as i64 * s + half, center.y as i64 * s + half, center.z as i64 * s + half);
    let (dx, dy, dz) = (wx - px, wy - py, wz - pz);
    (dx * dx + dy * dy + dz * dz) as u64
}

/// Fresh full-res chunk meshing. Worklist lane (seed set `mesh_worklist`);
/// in-flight is the `NeedsMesh { building: true }` claim; ready is the 4-predicate gate.
pub(in crate::world) struct MeshLane;
impl LaneSpec for MeshLane {
    type Key = Coord;
    /// Low floor: each admit captures a padded voxel+light snapshot (tens of
    /// KiB of copies), so a big floor would repeal the 2 ms window.
    const MIN_ADMIT: usize = 4;
    fn candidates(_world: &World, _center: Coord) -> Candidates<Coord> {
        Candidates::Worklist
    }
    fn seed_set(world: &mut World) -> Option<&mut FastSet<Coord>> {
        Some(&mut world.mesh_worklist)
    }
    fn pending(world: &mut World) -> &mut Sticky {
        &mut world.pending_fresh
    }
    fn order(center: Coord, key: Coord) -> i32 {
        World::order(key, center)
    }
    fn in_flight(world: &World, key: Coord) -> bool {
        matches!(
            world.chunks.get(&key).map(|l| &l.state),
            Some(MeshState::NeedsMesh { building: true })
        )
    }
    fn ready(world: &World, key: Coord) -> bool {
        // Awaiting a fresh mesh, in view, all neighbour data present, and EITHER
        // the neighbourhood light settled (mesh once with final smooth light) OR
        // the chunk has waited on neighbour light past the degrade deadline —
        // then it meshes DEGRADED now and remeshes when real light lands.
        world.is_needs_mesh(key)
            && world.in_mesh_box(key)
            && world.neighbours_have_data(key)
            && (world.light_ready(key) || world.light_wait_expired(key))
    }
    fn submit(world: &mut World, key: Coord) -> Option<pipeline::Job> {
        world.refresh_tables(); // the snapshot shares the solid-table Arc
        // A key admitted with light not yet ready is meshed DEGRADED: missing
        // neighbour light planes stand in as fully-lit open-sky, and the chunk is
        // recorded so it remeshes (and clears) once its real light arrives.
        let degraded = !world.light_ready(key);
        let (rev, snapshot) = world.snapshot(key, degraded);
        world.mark_degraded(key, degraded);
        Some(pipeline::Job::Mesh { coord: key, rev, snapshot })
    }
    fn claim(world: &mut World, key: Coord) {
        // NeedsMesh{false} → NeedsMesh{true}: setting `building` IS the
        // mesh-in-flight claim (held until the budgeted upload retires it, or a
        // stale result releases it). Out of the worklist too.
        world.mesh_worklist.remove(&key);
        if let Some(loaded) = world.chunks.get_mut(&key) {
            debug_assert!(loaded.state.is_needs_mesh(), "mesh submit for non-NeedsMesh {key:?}");
            loaded.state = MeshState::NeedsMesh { building: true };
        }
    }
    fn integrate(world: &mut World, done: pipeline::Done) {
        // The coord stays claimed (`building: true`) until the budgeted upload
        // resolves; `accept_mesh` queues it (or drops+re-seeds if stale).
        if let pipeline::Done::Mesh { coord, rev, data } = done {
            world.accept_mesh(coord, rev, data);
        }
    }
}

/// LOD2 column sections. Geometry lane (covering frontier); in-flight is the
/// map entry, claimed as `SectionState::Meshing`. Distance-ordered via `FarQueue`.
pub(in crate::world) struct SectionLane;
impl LaneSpec for SectionLane {
    type Key = SectionPos;
    /// An admit is a generator clone + (rarely) an edit-overlay copy; the heavy
    /// extract + mesh runs off-thread, so a modest floor keeps the enqueue cheap.
    const MIN_ADMIT: usize = 4;
    fn candidates(world: &World, center: Coord) -> Candidates<SectionPos> {
        Candidates::Geometry(
            world.desired_sections(center).into_iter().filter(|s| !world.sections.contains_key(s)).collect(),
        )
    }
    fn seed_set(_world: &mut World) -> Option<&mut FastSet<SectionPos>> {
        None
    }
    fn pending(world: &mut World) -> &mut Sticky {
        &mut world.pending_sections
    }
    fn order(center: Coord, key: SectionPos) -> i32 {
        let cs = CHUNK_SIZE as i32;
        let span = key.span();
        let (psx, psz) = ((center.x * cs).div_euclid(span), (center.z * cs).div_euclid(span));
        (key.x - psx).abs().max((key.z - psz).abs())
    }
    fn dist2(_world: &World, center: Coord, key: SectionPos) -> Option<u64> {
        // A 2-D far field (columns span the whole vertical domain): pass the
        // player's own centre Y so the vertical term drops out.
        let span = key.span() as i64;
        let py = center.y as i64 * CHUNK_SIZE as i64 + CHUNK_SIZE as i64 / 2;
        Some(player_dist2(center, key.min_x() as i64 + span / 2, py, key.min_z() as i64 + span / 2))
    }
    fn in_flight(world: &World, key: SectionPos) -> bool {
        world.sections.contains_key(&key)
    }
    fn submit(world: &mut World, key: SectionPos) -> Option<pipeline::Job> {
        // The section mesher reads the hot solidity/opacity tables like the tile
        // mesher; refresh them before the snapshot.
        world.refresh_tables();
        // Each section extracts its own footprint from the generator, replaying the
        // player edits over that footprint (empty for a never-edited section).
        Some(pipeline::Job::Section {
            pos: key,
            generator: world.generator.clone(),
            edits: world.edits_for_section(key),
            tables: world.tables.get(),
        })
    }
    fn claim(world: &mut World, key: SectionPos) {
        // Claim out of the dirty set (if it was there) as it goes in flight, so a
        // later edit re-marks it for a fresh re-extract.
        world.dirty_sections.remove(&key);
        world.sections.insert(key, SectionState::Meshing);
    }
    fn integrate(world: &mut World, done: pipeline::Done) {
        // Sections never go stale by edit (an edit frees + re-admits); a landing
        // for an unloaded section is dropped at upload time.
        if let pipeline::Done::Section { pos, meshes } = done {
            world.section_upload_queue.push_back((pos, meshes));
        }
    }
}

/// Cross-chunk light settling. Worklist lane (seed set `light_worklist`);
/// in-flight is `light_inflight`. Its enqueue is time-budgeted from the frame's
/// `light_apply` deadline; the apply drain in `streaming.rs` shares that deadline.
pub(in crate::world) struct LightLane;
impl LaneSpec for LightLane {
    type Key = Coord;
    /// High floor: an admit is a light-shell capture, far cheaper than a mesh
    /// snapshot, and the settle worklist must drain fast for meshing to start.
    const MIN_ADMIT: usize = 32;
    fn candidates(_world: &World, _center: Coord) -> Candidates<Coord> {
        Candidates::Worklist
    }
    fn seed_set(world: &mut World) -> Option<&mut FastSet<Coord>> {
        Some(&mut world.light_worklist)
    }
    fn pending(world: &mut World) -> &mut Sticky {
        &mut world.light_pending
    }
    fn order(center: Coord, key: Coord) -> i32 {
        World::order(key, center)
    }
    fn in_flight(world: &World, key: Coord) -> bool {
        world.light_inflight.contains(&key)
    }
    fn submit(world: &mut World, key: Coord) -> Option<pipeline::Job> {
        if !world.lighting || !world.chunks.contains_key(&key) {
            return None;
        }
        world.refresh_tables();
        // Capture the frozen neighbourhood the flood reads; `capture_ceiling`
        // warms the per-column ceiling cache (hence `&mut World`).
        let shell = world.capture_face_shell(key);
        let ceiling = world.capture_ceiling(key);
        let snapshot = pipeline::LightSnapshot {
            chunk: Arc::clone(&world.chunks[&key].chunk),
            shell,
            ceiling,
            world_y0: key.y * CHUNK_SIZE as i32,
            tables: world.tables.get(),
        };
        Some(pipeline::Job::Light {
            coord: key,
            epoch: world.light_epoch,
            snapshot: Box::new(snapshot),
        })
    }
    fn claim(world: &mut World, key: Coord) {
        // Out of the worklist, into the in-flight set (one flood per chunk).
        world.light_worklist.remove(&key);
        world.light_inflight.insert(key);
    }
    fn integrate(world: &mut World, done: pipeline::Done) {
        // Buffer for budgeted application; the chunk stays in `light_inflight`
        // (so `light_ready` keeps gating meshing) until it is actually applied.
        if let pipeline::Done::Light { coord, epoch, grid } = done {
            if world.lighting
                && epoch == world.light_epoch
                && world.chunks.contains_key(&coord)
                && world.light_inflight.contains(&coord)
            {
                world.light_apply_queue.push_back((coord, grid));
            }
        }
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
    use crate::render_config::RenderConfig;
    use voxel_engine::{DVec3, Pass};

    /// A headless lod2 world with the section far field enabled.
    fn lod2_world() -> World {
        World::with_config(DEFAULT_SEED, RenderConfig::default())
    }

    #[test]
    fn lod2_is_the_default_and_near_only_leaves_sections_dormant() {
        assert!(World::generate().lod2, "lod2 far field on by default");
        // A near-only world (`lod2: false`) leaves the section lane dormant.
        let d = World::with_config(DEFAULT_SEED, RenderConfig { lod2: false, ..RenderConfig::default() });
        assert!(!d.lod2 && d.sections.is_empty() && d.section_visible.is_empty());
    }

    #[test]
    fn section_lane_claim_and_integrate_parity() {
        // Claim marks in-flight, integrate queues the finished result.
        let mut world = lod2_world();
        let center = ChunkCoord::new(0, 0, 0);
        world.center = Some(center);
        let pos = world.desired_sections(center)[0];
        assert!(!<SectionLane as LaneSpec>::in_flight(&world, pos));
        <SectionLane as LaneSpec>::claim(&mut world, pos);
        assert!(matches!(world.sections.get(&pos), Some(SectionState::Meshing)));
        assert!(<SectionLane as LaneSpec>::in_flight(&world, pos), "claimed ⇒ in flight");
        <SectionLane as LaneSpec>::integrate(
            &mut world,
            pipeline::Done::Section { pos, meshes: Vec::new() },
        );
        assert_eq!(world.section_upload_queue.len(), 1, "landing queued for upload");
    }

    #[test]
    fn section_covering_gates_on_a_ready_ancestor_or_self() {
        // A cell is covered by a Ready self or by a Ready ancestor.
        let mut world = lod2_world();
        let center = ChunkCoord::new(0, 0, 0);
        let cell = world.desired_sections(center)[0];
        assert!(!world.section_covered(cell), "nothing loaded ⇒ uncovered");
        let empty_ready = || SectionState::Ready { cell: 1.0, blocks: Vec::new() };
        world.sections.insert(cell, empty_ready());
        assert!(world.section_covered(cell), "a Ready self covers");
        // Test ancestor covering.
        world.sections.remove(&cell);
        world.sections.insert(cell.parent(), empty_ready());
        assert!(world.section_covered(cell), "a Ready ancestor covers the finer cell");
    }

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
            // Collision skips liquids (they are solid to the mesher but passable),
            // so the reference keys off the same obstacle predicate.
            let reference = aabb.voxel_cells().any(|(x, y, z)| world.is_obstacle(x, y, z));
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
    fn lighting_toggle_reseeds_only_stale_work_and_rejects_old_results() {
        let mut world = World::generate();
        let coord = ChunkCoord::new(0, 0, 0);
        let missing = ChunkCoord::new(1, 0, 0);

        // Pretend this chunk already had a grid while a newer settle was in
        // flight. Disabling must not preserve that known-stale grid.
        world.light_worklist.clear();
        world.chunks.get_mut(&coord).unwrap().light = Some(light::LightGrid::dark());
        world.light_inflight.insert(coord);
        assert!(world.transition_lighting(false));
        let off_epoch = world.light_epoch;
        assert!(!world.lighting());
        assert!(world.light_worklist.is_empty());
        assert!(world.light_inflight.is_empty());
        assert!(world.chunks[&coord].light.is_none());

        // An edit made while disabled remains dormant, and a chunk loaded while
        // disabled is represented by an absent grid.
        let old = world.block_at(3, 3, 3);
        let stone = world.registry().id_by_name("Stone").unwrap();
        world.set_block(3, 3, 3, if old == AIR { stone } else { AIR });
        world.chunks.get_mut(&missing).unwrap().light = None;
        assert!(world.light_worklist.contains(&coord));

        assert!(world.transition_lighting(true));
        assert_eq!(world.light_epoch, off_epoch.wrapping_add(1));
        assert!(world.light_worklist.contains(&coord));
        assert!(world.light_worklist.contains(&missing));
        assert!(world.light_pending.get());

        // A worker from the disabled generation cannot publish after re-enable;
        // a result stamped with the current generation can.
        world.light_inflight.insert(coord);
        <LightLane as LaneSpec>::integrate(
            &mut world,
            pipeline::Done::Light { coord, epoch: off_epoch, grid: light::LightGrid::dark() },
        );
        assert!(world.light_apply_queue.is_empty());
        let current_epoch = world.light_epoch;
        <LightLane as LaneSpec>::integrate(
            &mut world,
            pipeline::Done::Light {
                coord,
                epoch: current_epoch,
                grid: light::LightGrid::dark(),
            },
        );
        assert_eq!(world.light_apply_queue.len(), 1);
        assert!(!world.transition_lighting(true), "same value is a no-op");
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
        // vertically (the vertical bound is the tighter, derived vertical radius).
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
        // layer 1) touches the chunk below.
        // Edit to whatever the cell is NOT (the generator owns what's there —
        // the origin surface sits around y=25, so this cell is usually solid),
        // so the no-op-placement skip can't swallow the change.
        let stone = world.registry().id_by_name("Stone").unwrap();
        let other =
            if world.block_at(8, 16, 8) == crate::block::AIR { stone } else { crate::block::AIR };
        let below = world.chunks[&ChunkCoord::new(0, 0, 0)].rev;
        world.set_block(8, 16, 8, other);
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
        assert_eq!(
            ground.state,
            MeshState::NeedsMesh { building: false },
            "dense terrain waits for a real mesh"
        );
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
            MeshState::NeedsMesh { building: false },
            MeshState::NeedsMesh { building: true },
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
        world.chunks.get_mut(&coord).unwrap().state = MeshState::NeedsMesh { building: true };
        let rev = world.chunks[&coord].rev;

        world.set_block(2, 2, 2, AIR); // building → Dirty (claim dropped), rev bumped
        assert!(world.chunks[&coord].state.is_dirty(), "edit turns a building chunk into Dirty");
        assert_ne!(world.chunks[&coord].rev, rev, "edit bumps rev, stranding the job");

        // The worker's result lands at the OLD rev: dropped, never uploaded. The
        // `Dirty` state is now the claim; the sync remesh owns it.
        world.pending_fresh.take();
        world.accept_mesh(coord, rev, new_chunk_mesh_data());
        assert!(world.upload_queue.is_empty(), "stale mesh result never queues");
        assert!(world.chunks[&coord].state.is_dirty(), "chunk stays Dirty for the sync remesh");
        assert!(world.pending_fresh.get(), "drop re-arms the fresh scan");
    }

    #[test]
    fn mesh_result_stale_by_box_exit_releases_the_claim() {
        // The wedge regression. A fresh mesh result that no longer applies
        // because the chunk left the mesh box — the view moved, with NO edit and
        // NO rev bump — must release the in-flight claim. The old `Meshing` state
        // had no un-claim on this path (only success or an edit cleared it), so a
        // fast fly-by that caught a chunk mid-flight at the box edge wedged it
        // claimed-but-never-ready forever: transparent, never re-meshed on
        // landing. Now the claim is a `building` bool that the stale-drop path
        // clears, so the chunk falls back to a re-meshable `NeedsMesh`.
        let mut world = World::generate();
        let coord = ChunkCoord::new(0, 0, 0);
        world.center = Some(coord);
        let rev = world.chunks[&coord].rev;
        // Claim it, exactly as `MeshLane::claim` would when a job is submitted.
        world.chunks.get_mut(&coord).unwrap().state = MeshState::NeedsMesh { building: true };
        // Player teleports far: the chunk is now outside the mesh box, so its
        // in-flight result is stale by BOX (rev is untouched — no edit happened).
        world.center = Some(ChunkCoord::new(1000, 0, 0));
        assert!(!world.mesh_result_applies(coord, rev), "out-of-box result is stale");

        world.pending_fresh.take();
        world.accept_mesh(coord, rev, new_chunk_mesh_data());
        assert!(world.upload_queue.is_empty(), "stale result never queues");
        assert_eq!(
            world.chunks[&coord].state,
            MeshState::NeedsMesh { building: false },
            "claim released — the chunk is re-meshable, not wedged in a Meshing state"
        );
        assert!(world.mesh_worklist.contains(&coord), "re-seeded for a later mesh");
        assert!(world.pending_fresh.get(), "drop re-arms the fresh scan");
    }

    #[test]
    fn view_volume_vertical_is_derived_and_flatter() {
        // The streamed volume is a flat box: the vertical radius is half the
        // horizontal one, clamped to 2..=5, on every render-distance setting.
        // A full cube would load a tall column of sky/rock that never draws.
        let mut world = World::generate();
        for (view, vertical) in [(3, 2), (4, 2), (6, 3), (8, 4), (10, 5), (20, 5)] {
            world.set_view_radius(view);
            assert_eq!(world.view.horizontal, view, "view {view}");
            assert_eq!(world.view.vertical, vertical, "vertical at view {view}");
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
        // Every handle-less state → Dirty{None} (a `building` chunk drops its
        // claim in the process).
        for empty in [
            MeshState::Air,
            MeshState::NeedsMesh { building: false },
            MeshState::NeedsMesh { building: true },
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
    fn lod2_far_field_drives_to_covering_complete() {
        // Run the real section lane for the whole desired frontier on real
        // terrain, simulating GPU upload as an empty `Ready`, and assert
        // covering resolution terminates (every cell covered by an ancestor or self).
        let mut world = lod2_world();
        let center = ChunkCoord::new(0, 0, 0);
        world.center = Some(center);
        let desired = world.desired_sections(center);
        assert!(!desired.is_empty(), "a lod2 world wants a far frontier at spawn");

        let workers = pipeline::Workers::spawn(2);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while world.desired_sections(center).iter().any(|&c| !world.section_covered(c)) {
            assert!(std::time::Instant::now() < deadline, "covering did not converge: {}", world.entry_debug());
            // Submit every desired-but-unloaded section (the real lane path).
            for pos in world.desired_sections(center) {
                if <SectionLane as LaneSpec>::in_flight(&world, pos) {
                    continue;
                }
                <SectionLane as LaneSpec>::claim(&mut world, pos);
                if let Some(job) = <SectionLane as LaneSpec>::submit(&mut world, pos) {
                    assert!(workers.submit(job), "worker pool admits the section job");
                }
            }
            // Drain finished jobs and yield briefly when the pool is empty.
            let mut got = false;
            while let Some(done) = workers.try_recv() {
                <SectionLane as LaneSpec>::integrate(&mut world, done);
                got = true;
            }
            if !got {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            while let Some((pos, _meshes)) = world.section_upload_queue.pop_front() {
                if let Some(s @ SectionState::Meshing) = world.sections.get_mut(&pos) {
                    *s = SectionState::Ready { cell: pos.cell_size() as f32, blocks: Vec::new() };
                }
            }
        }
        assert!(world.sections.values().any(|s| s.is_ready()), "at least one Ready section");
        assert!(
            world.desired_sections(center).iter().all(|&c| world.section_covered(c)),
            "terminal state: every desired cell has a drawable ancestor-or-self",
        );
    }

    #[test]
    fn light_settle_to_identical_grid_reseeds_evicted_mesh_seed() {
        // Regression: `lane_enqueue::<MeshLane>` evicts a light-blocked seed from
        // `mesh_worklist`, trusting `settle_light` to re-seed it once the block
        // clears. `settle_light` used to early-return without re-seeding when the
        // settled grid was IDENTICAL to the old one, stranding a chunk that
        // settled to the light it already had (ready-but-unreachable — the golden
        // idle stall). The fix re-arms this chunk's mesh readiness UNCONDITIONALLY,
        // before the `self_changed` early-return; this pins that.
        let mut world = World::generate();
        let c = ChunkCoord::new(0, 0, 0);
        world.center = Some(c);

        // C and its 6 face neighbours must have data (generate() pregenerates
        // near spawn) and each must carry a published light grid so `light_ready`
        // is gateable purely by `light_inflight`/`light_worklist` membership.
        for n in std::iter::once(c).chain(crate::coord::Face::ALL.iter().map(|&f| c.step(f))) {
            world.chunks.get_mut(&n).expect("neighbourhood pregenerated near spawn").light =
                Some(light::LightGrid::dark());
        }
        world.chunks.get_mut(&c).unwrap().state = MeshState::NeedsMesh { building: false };
        assert!(world.neighbours_have_data(c));
        assert!(world.in_mesh_box(c));

        // Block C on light: still awaiting its own settle.
        world.light_worklist.clear();
        world.light_inflight.insert(c);
        assert!(!world.light_ready(c), "in-flight light blocks readiness");
        assert!(!<MeshLane as LaneSpec>::ready(&world, c), "blocked: not mesh-ready yet");

        // Seed C, then run the real eviction path: lane_enqueue partitions
        // candidates into ready/blocked and evicts every blocked worklist seed.
        world.mesh_worklist.insert(c);
        world.pending_fresh.set();
        lane_enqueue::<MeshLane>(
            &mut world,
            c,
            pipeline::Deadline::from_budget(std::time::Duration::from_secs(1)),
        );
        assert!(!world.mesh_worklist.contains(&c), "lane_enqueue evicted the blocked seed");

        // C's light settles — but to the SAME grid it already had (e.g. a flood
        // that changes nothing new). `settle_light` is the atomic terminal
        // transition: it releases the in-flight claim ITSELF (no separate
        // `light_inflight.remove` at the call site), publishes, and re-arms mesh.
        world.settle_light(c, light::LightGrid::dark());

        // Fixed: light is ready AND the chunk is back on the worklist with a scan
        // armed, so the mesh lane will claim it next stream instead of stranding it.
        assert!(world.light_ready(c), "light_inflight cleared, grid published: ready now");
        assert!(<MeshLane as LaneSpec>::ready(&world, c), "every MeshLane::ready predicate holds");
        assert!(world.mesh_worklist.contains(&c), "re-seeded: ready chunk restored to the seed set");
        assert!(world.pending_fresh.get(), "re-armed: the fresh scan will pick it up");
    }
}
