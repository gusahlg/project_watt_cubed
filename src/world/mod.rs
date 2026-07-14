//! The world owns the block palette and an *infinite*, streamed field of
//! chunks — infinite along all three axes: 16-cube chunks stack upward through
//! the flying-island band and downward through bottomless stone. It keeps the
//! chunks near the player loaded (generated and meshed), discards distant
//! ones, and answers what block is at a position, whether a box collides with
//! terrain, and how to draw the visible surface.
//!
//! Two design choices for speed: chunks use a fast multiplicative hasher (not the
//! default SipHash, which is too slow for per-frame collision), and player edits
//! live in a compact overlay so regenerated chunks match originals after streaming
//! out and back in. Most of the 3D volume is uniform air or stone
//! ([`ChunkData::Uniform`](chunk::ChunkData)), costing no voxel array and
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
pub mod placement;
pub mod pyramid;
pub mod section;

mod edits;
mod heightmip;
mod metric;
mod quadtree;
mod query;
mod streaming;
mod summary;
mod swapfade;

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::sync::Arc;
use std::sync::mpsc::Receiver;
use std::time::Instant;

use voxel_engine::{CoverageVolume, DVec3, Detail, Engine, Frame3D, MeshHandle, Vec3};

use crate::block::registry::{BlockId, BlockRegistry, HotTables};
use crate::coord::{ByPass, ChunkBox, ChunkCoord};
use crate::render::Render;
use chunk::{CHUNK_SIZE, Chunk};
use generation::{SineHills, TerrainGenerator};
use heightmip::HeightMip;
use light::LightGrid;
use mesh::{ChunkMeshData, new_chunk_mesh_data};
use quadtree::QuadrantMask;
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
    /// The full-res coverage slab in metres: the shader's LOD-cull volume, which
    /// must equal this streamed full-res volume (one source of truth for both).
    fn coverage(&self) -> CoverageVolume {
        CoverageVolume {
            radius: (self.horizontal * CHUNK_SIZE as i32) as f32,
            half_height: (self.vertical * CHUNK_SIZE as i32) as f32,
        }
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
    /// Wrap freshly uploaded per-pass handles, same "at least 1 present" rule as [`Self::new`].
    pub(in crate::world) fn from_upload_handles(handles: ByPass<Option<MeshHandle>>) -> Option<Self> {
        Self::new(ByPass::from_fn(|p| handles[p].map(OwnedMesh::new)))
    }
    /// Record a draw for each present pass at `offset`/`detail`.
    fn draw(&self, f: &mut Frame3D, offset: Vec3, detail: Detail) {
        for (_, m) in self.0.iter() {
            if let Some(mesh) = m {
                f.draw_mesh(mesh.id(), offset, detail);
            }
        }
    }
    /// Draw each present pass with cross-fade and far-material styling.
    /// `(1.0, 0, 0)` is identical to [`draw`](Self::draw).
    fn draw_faded(&self, f: &mut Frame3D, offset: Vec3, detail: Detail, fade: f32, mode: u32, flat_rgba: u32) {
        for (_, m) in self.0.iter() {
            if let Some(mesh) = m {
                f.draw_mesh_faded(mesh.id(), offset, detail, fade, mode, flat_rgba);
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

/// Per-draw `mode` bitflags: flat palette-average colour instead of texture,
/// and fade-the-complement for an outgoing cell.
const DRAW_FLAT_COLOR: u32 = 1;
const DRAW_FADE_OUT: u32 = 2;
/// Detail threshold for flat palette-average colour (far-material optimization).
/// The outer rings lose per-texel detail to sub-pixel shimmer, so average colour
/// reduces bandwidth; nearer rings keep texture. Inert until the mip bake lands.
const FLAT_DETAIL: u8 = section::FINEST_DETAIL + 4;

/// A section is either meshing or ready. Blocks are pre-positioned at upload,
/// so draw needs only camera-relative offset arithmetic.
pub(in crate::world) enum SectionState {
    Meshing,
    /// Blocks grouped by quadrant (indexed by [`SectionPos::quadrant`]); the draw
    /// path renders only the quadrants a [`QuadrantMask`] selects. Each block is
    /// positioned at a world min-corner, drawn at the section's [`Detail`]
    /// (cell size `2^detail`) — derived from the map key `pos.detail`, not stored.
    Ready { quadrants: [Vec<([f64; 3], ChunkMeshes)>; 4] },
}

impl SectionState {
    /// Upload each quadrant's block meshes with pre-baked world positions so draw
    /// needs only the camera-relative subtract. Empty quadrants upload to no handles.
    fn from_upload(pos: SectionPos, meshes: [SectionMeshData; 4], eng: &mut Engine) -> SectionState {
        let cell = pos.cell_size();
        let (mx, mz) = (pos.min_x() as f64, pos.min_z() as f64);
        let quadrants = meshes.map(|quad| {
            let mut blocks = Vec::new();
            for (block_origin, data) in quad {
                let handles = ByPass::from_fn(|p| eng.upload_mesh(&data[p]));
                if let Some(meshes) = ChunkMeshes::from_upload_handles(handles) {
                    // Block origin is in section-space CELLS; the section floor is world-Y 0.
                    let wmin = [
                        mx + block_origin.x as f64 * cell as f64,
                        block_origin.y as f64 * cell as f64,
                        mz + block_origin.z as f64 * cell as f64,
                    ];
                    blocks.push((wmin, meshes));
                }
            }
            blocks
        });
        SectionState::Ready { quadrants }
    }

    fn is_ready(&self) -> bool {
        matches!(self, SectionState::Ready { .. })
    }
    /// Draw the quadrants `mask` selects at `detail`, with cross-fade and far-material
    /// styling via `fade`/`mode`/`flat_rgba`. `(1.0, 0, 0)` is flat draw.
    /// The shader hands the near ground to full-res chunks via coverage dither.
    fn draw(&self, f: &mut Frame3D, cam: DVec3, mask: QuadrantMask, detail: Detail, fade: f32, mode: u32, flat_rgba: u32) {
        if let SectionState::Ready { quadrants } = self {
            for q in mask.iter() {
                for (wmin, meshes) in &quadrants[q.index()] {
                    let offset = (DVec3::new(wmin[0], wmin[1], wmin[2]) - cam).as_vec3();
                    meshes.draw_faded(f, offset, detail, fade, mode, flat_rgba);
                }
            }
        }
    }
    fn free(self, eng: &mut Engine) {
        if let SectionState::Ready { quadrants, .. } = self {
            for quad in quadrants {
                for (_, meshes) in quad {
                    meshes.free(eng);
                }
            }
        }
    }
}

/// Mesh-lifecycle state of a loaded chunk. Owns GPU mesh via [`OwnedMesh`] token.
/// Handle ownership rides the state machine: moves on edit or frees on unload.
///
/// `NeedsMesh` has a `building` flag (not a separate state) to represent in-flight
/// mesh jobs. This avoids wedging when an async result goes stale due to a view
/// move: the claim is a bool that only `Air`/`Ready` cannot carry, and clearing
/// it on any result-consumption prevents chunks from getting stuck.
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
    /// else `Air` (an all-air chunk uploads to nothing).
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
    upload_queue: VecDeque<(Coord, u32, Box<ChunkMeshData>)>,
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
    /// application via [`settle_light`]. Buffering here and applying at most
    /// a per-frame time budget ([`pipeline::LIGHT_APPLY_BUDGET`]) caps
    /// main-thread bookkeeping spikes; leftovers apply next frame.
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
    /// Panic counts per failed claim, for the bounded-retry policy in
    /// [`fail_job`](World::fail_job). Rare by construction (a strike is a
    /// worker panic), so the map stays tiny.
    job_strikes: FastMap<streaming::FailKey, u8>,
    /// Claims that kept panicking: permanently parked so one poison input is a
    /// bounded hole in the world, not an infinite resubmit-panic loop. Every
    /// scan that would re-request the work consults this set.
    quarantined: FastSet<streaming::FailKey>,
    /// Block count last uploaded. Rebuilds/re-uploads when palette grows.
    textures_built: usize,
    /// Built texture layers by id, kept so palette growth (crafting registers
    /// one block at a time) appends new layers instead of regenerating all.
    texture_cache: Vec<Vec<u8>>,
    /// Device texture-array layer ceiling, stamped into `HotTables::layer_cap`
    /// so the meshers wrap vertex layers past it. `u16::MAX` until the first
    /// stream pass reads the engine cap (identity in practice — ids start tiny).
    texture_layer_cap: u16,
    /// Baked corner AO in the mesher — stamped into `HotTables::ao`. A meshing
    /// input like `lighting`: toggling remeshes the world.
    ao: bool,
    /// Bumped whenever a chunk gains or loses a drawable mesh (retire installs
    /// and frees, unloads, the free-meshes passes). `invalidate()` keeps the
    /// previous mesh drawing, so per-edit dirtying never bumps.
    draw_set_rev: u64,
    /// Drawable chunk coords, rebuilt by `sync_draw_cache` when `draw_set_rev`
    /// moved — render walks this (~live set) instead of every loaded chunk.
    draw_cache: Vec<Coord>,
    draw_cache_rev: u64,
    /// Bumped whenever a non-registry meshing input stamped onto the hot
    /// tables changes (AO today), so `refresh_tables` rebuilds even though the
    /// block count didn't move — the count and epoch fold into one revision.
    tables_epoch: u32,
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
    /// The eye altitude captured each `stream()` before chunk-coord floor rounds it.
    /// Feeds the vertical LOD selection. XZ selection uses chunk centre only.
    section_eye_y: f64,
    /// Previous stream eye + timestamp for velocity computation.
    /// Reset to None on teleport or first stream.
    section_eye_prev: Option<(DVec3, Instant)>,
    /// Eye velocity (m/s) from successive stream centres.
    /// Zero at rest or after teleport.
    section_vel: DVec3,
    /// Per-cell relief drives error-driven LOD selection for the far field.
    /// `None` until the background bake lands; selection falls back to default LOD.
    section_mip: Option<HeightMip>,
    /// Receiver for the in-flight background bake, taken once it lands. `None` before
    /// the bake is spawned and after it is installed.
    section_mip_rx: Option<Receiver<HeightMip>>,
    /// Loaded sections.
    sections: FastMap<SectionPos, SectionState>,
    /// Finished section meshes awaiting budgeted upload.
    section_upload_queue: VecDeque<(SectionPos, [SectionMeshData; 4])>,
    /// Whether desired sections still need enqueueing (budget spreads a flood).
    pending_sections: Sticky,
    /// Sections invalidated by edits, freed and re-admitted from the generator.
    dirty_sections: FastSet<SectionPos>,
    /// Visible sections (covering-resolved) — the desired cut this frame, before the
    /// cross-fade. Feeds `unload_sections` and the fade.
    section_visible: Vec<(SectionPos, QuadrantMask)>,
    /// Temporal cross-fade over successive LOD cuts with per-cell dissolve state.
    /// Presentation only; never affects selection logic.
    section_fade: swapfade::SwapFade,
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
        // `mut` for the placement compile: the generator registers every block
        // terrain can emit here at startup, then keeps only resolved ids.
        let mut registry = BlockRegistry::with_builtins();
        let generator = SineHills::new(&mut registry, 20.0, seed);
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
            job_strikes: FastMap::default(),
            quarantined: FastSet::default(),
            textures_built: 0,
            texture_cache: Vec::new(),
            texture_layer_cap: u16::MAX,
            ao: true,
            tables_epoch: 0,
            draw_set_rev: 0,
            draw_cache: Vec::new(),
            // Differs from `draw_set_rev` so the first stream builds the cache.
            draw_cache_rev: u64::MAX,
            occlusion: Occlusion::default(),
            occlusion_dirty: Sticky::default(),
            occlusion_active: false,
            occlusion_forced: render.occlusion,
            lighting: true,
            light_epoch: 0,
            lod2,
            section_pyramid: pyramid::PyramidCfg::sections(unit),
            section_eye_y: 0.0,
            section_eye_prev: None,
            section_vel: DVec3::ZERO,
            section_mip: None,
            section_mip_rx: None,
            sections: FastMap::default(),
            section_upload_queue: VecDeque::new(),
            pending_sections: Sticky::default(),
            dirty_sections: FastSet::default(),
            section_visible: Vec::new(),
            section_fade: swapfade::SwapFade::default(),
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
        // The drawable-coords cache (synced at the end of `stream`, the frame's
        // last mutation point) — the walk is O(live), not O(loaded).
        debug_assert_eq!(
            self.draw_cache.len(),
            self.chunks.values().filter(|l| l.state.live_meshes().is_some()).count(),
            "draw cache went stale: a mesh-set mutation missed its draw_set_rev bump"
        );
        let mut live = 0u64;
        for &coord in &self.draw_cache {
            if self.occlusion_active && !self.occlusion.is_visible(coord) {
                continue;
            }
            let Some(loaded) = self.chunks.get(&coord) else { continue };
            if let Some(meshes) = loaded.state.live_meshes() {
                live += 1;
                let origin =
                    DVec3::new(coord.x as f64 * s, coord.y as f64 * s, coord.z as f64 * s);
                // Full-res chunks are unit-scale: their local 0..=16 coords are
                // already metres. LOD sections pass 2^k here (see `SectionState::draw`).
                meshes.draw(f, (origin - cam).as_vec3(), Detail::FULL);
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
        // Vertical extent: the streamed slab guarantees `vertical` chunks above
        // and below the eye even when the eye sits at a chunk boundary.
        f.set_lod_clip(self.view.coverage());
        // Draw the cross-fade view with per-cell dissolve. Bands screen-door-dissolve
        // instead of popping. At rest every cell is fade 1 / mode 0 (unless flat).
        for (pos, mask, fade, out) in self.section_fade.draws() {
            // Occlusion cull sections hidden behind terrain. Under-claiming means
            // a miss wastes a draw but never creates holes.
            if self.occlusion_active {
                if let Some(mip) = &self.section_mip {
                    if mip.occludes(cam, pos) {
                        continue;
                    }
                }
            }
            if let Some(state) = self.sections.get(&pos) {
                let (flat_mode, flat_rgba) = self.section_material(pos);
                let mode = flat_mode | if out { DRAW_FADE_OUT } else { 0 };
                state.draw(f, cam, mask, Detail::new(pos.detail), fade, mode, flat_rgba);
            }
        }
    }

    /// Far-material style: flat palette-average past [`FLAT_DETAIL`] if available,
    /// else textured. Returns `mode` bit and packed sRGB colour.
    fn section_material(&self, pos: SectionPos) -> (u32, u32) {
        if pos.detail < FLAT_DETAIL {
            return (0, 0);
        }
        match self.section_mip.as_ref().and_then(|m| m.color(pos)) {
            Some(c) => (
                DRAW_FLAT_COLOR,
                c.r as u32 | (c.g as u32) << 8 | (c.b as u32) << 16 | (c.a as u32) << 24,
            ),
            None => (0, 0),
        }
    }

    /// Skip near-field LOD load if the section's footprint is provably inside the
    /// coverage clip slab, so the shader discards it anyway. Only filters the load lane,
    /// not the desired set; selection stays isotropic.
    ///
    /// Skip only if all backing chunks are settled (drawable or born-air, never
    /// in-flight), to avoid holes during fast descent.
    fn coverage_skips(&self, center: Coord, key: SectionPos) -> bool {
        let cov = self.view.coverage();
        let cs = CHUNK_SIZE as i32;
        let (h_lim, v_lim) = (0.75 * cov.radius, 0.75 * cov.half_height);
        // The f64 eye XZ was floored to `center` before this lane; inflate the reach
        // by one chunk half-diagonal so the true eye can't sit outside our bound.
        let margin = cs as f32 * 0.5 * std::f32::consts::SQRT_2;
        let (ex, ez) = (center.x * cs + cs / 2, center.z * cs + cs / 2);
        let span = key.span();
        let (x0, z0) = (key.min_x(), key.min_z());
        let fx = (x0 - ex).abs().max((x0 + span - ex).abs()) as f32;
        let fz = (z0 - ez).abs().max((z0 + span - ez).abs()) as f32;
        if (fx * fx + fz * fz).sqrt() + margin > h_lim {
            return false;
        }
        // Vertical: the section's terrain must sit inside the eye's slab, else
        // clip draws the part that pokes out. If unbaked, can't prove, so don't skip.
        let Some(mip) = &self.section_mip else { return false };
        let Some((lo, hi)) = mip.relief_band(key) else { return false };
        let ey = self.section_eye_y as f32;
        if lo < ey - v_lim || hi > ey + v_lim {
            return false;
        }
        // Every chunk backing the footprint is settled, so the near area is
        // actually covered now, not just in-range.
        let (cx_lo, cx_hi) = (x0.div_euclid(cs), (x0 + span - 1).div_euclid(cs));
        let (cz_lo, cz_hi) = (z0.div_euclid(cs), (z0 + span - 1).div_euclid(cs));
        let (cy_lo, cy_hi) = ((lo.floor() as i32).div_euclid(cs), (hi.floor() as i32).div_euclid(cs));
        for cy in cy_lo..=cy_hi {
            for cz in cz_lo..=cz_hi {
                for cx in cx_lo..=cx_hi {
                    let settled = self.chunks.get(&Coord::new(cx, cy, cz)).is_some_and(|l| {
                        l.state.live_meshes().is_some() || matches!(l.state, MeshState::Air)
                    });
                    if !settled {
                        return false;
                    }
                }
            }
        }
        true
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

/// How a lane defines its work this frame. A geometry lane derives candidates
/// from the player centre. A worklist lane reads an explicit dirty set
/// accumulated on the `World` (mesh, light: coords marked dirty by loads/edits).
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
    /// The lane's work key: chunk `Coord` or `SectionPos`.
    /// Must be `Copy + Eq + Hash`.
    type Key: Copy + Eq + Hash;

    /// Minimum admissions guaranteed per call, before deadline checks. Each lane
    /// sets this based on typical cost; a small floor for cheap work prevents
    /// budget theft by expensive work.
    const MIN_ADMIT: usize;

    /// The candidate keys for this frame (see [`Candidates`]).
    fn candidates(world: &World, center: Coord) -> Candidates<Self::Key>;
    /// The worklist seed set, for worklist lanes (`None` for geometry lanes).
    fn seed_set(world: &mut World) -> Option<&mut FastSet<Self::Key>>;
    /// This lane's raise-then-consume "has pending work" gate.
    fn pending(world: &mut World) -> &mut Sticky;
    /// Nearest-first ordering metric (lower is sooner).
    fn order(center: Coord, key: Self::Key) -> i32;
    /// Whether `key` is already in flight.
    fn in_flight(world: &World, key: Self::Key) -> bool;
    /// Whether `key` can be submitted now. Default: always true.
    fn ready(world: &World, key: Self::Key) -> bool {
        let _ = (world, key);
        true
    }
    /// Squared distance in metres from `key` to player, for far-lane distance ordering.
    /// `None` (default) marks a near lane using FIFO ordering.
    fn dist2(world: &World, center: Coord, key: Self::Key) -> Option<u64> {
        let _ = (world, center, key);
        None
    }
    /// Build the worker job for `key`, or `None` to drop it.
    /// May warm caches but must not mutate lane state.
    fn submit(world: &mut World, key: Self::Key) -> Option<pipeline::Job>;
    /// Mark `key` in flight: remove from seed set and claim it to prevent re-submission.
    fn claim(world: &mut World, key: Self::Key);
    /// Fold a finished result back into the world (upload a mesh, publish light).
    fn integrate(world: &mut World, done: pipeline::Done);
}

/// The unified enqueue loop: gather, filter unready/in-flight, sort nearest-first,
/// submit until deadline expires (checked between items, never mid-item), then claim.
/// Clear pending once the ready backlog drains.
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
    // Partition into ready (not in-flight) and blocked. For worklist lanes, evict
    // blocked seeds since they'll be re-added when unblocked. This avoids
    // rescanning the accumulated backlog every frame.
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
    // Admission is time-gated: check deadline between items, never mid-work.
    // `exhausted` stays true only if all ready keys were processed before timeout.
    //
    // Forward-progress floor: large worklists under tight budgets can starve if
    // setup (gather/partition/sort) alone exhausts the budget before the first
    // submit. Guaranteeing at least MIN_ADMIT admissions makes the seed set
    // shrink strictly, so the lane converges.
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
        // Far lanes use distance ordering; near lanes use FIFO.
        // Compute dist² before taking the mutable workers pool.
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
            // Pool declined: stop and retry next frame instead of wasting work.
            exhausted = false;
            break;
        }
    }
    // Clear the gate once all ready items are submitted and no seeds remain.
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

/// Squared distance in metres from player-chunk centre to a world point.
fn player_dist2(center: Coord, wx: i64, wy: i64, wz: i64) -> u64 {
    let s = CHUNK_SIZE as i64;
    let half = s / 2;
    let (px, py, pz) = (center.x as i64 * s + half, center.y as i64 * s + half, center.z as i64 * s + half);
    let (dx, dy, dz) = (wx - px, wy - py, wz - pz);
    (dx * dx + dy * dy + dz * dz) as u64
}

/// Below this speed (m/s), motion bias is disabled.
const MOTION_BIAS_MIN_SPEED: f64 = 0.5;
/// Max bias fraction: cells ahead sort up to 30% nearer, cells behind 30% farther.
const MOTION_BIAS_STRENGTH: f64 = 0.3;

/// Bias priority by eye velocity: cells ahead sort sooner, behind later.
/// Affects ordering only, never the desired set. Identity at rest.
fn motion_biased_dist2(base: u64, vel: DVec3, dx: f64, dz: f64) -> u64 {
    let speed = (vel.x * vel.x + vel.z * vel.z).sqrt();
    let disp = (dx * dx + dz * dz).sqrt();
    if speed < MOTION_BIAS_MIN_SPEED || disp < 1.0 {
        return base;
    }
    let align = (vel.x * dx + vel.z * dz) / (speed * disp); // cosine in [-1, 1]
    (base as f64 * (1.0 - MOTION_BIAS_STRENGTH * align)).max(0.0) as u64
}

/// Fresh full-res chunk meshing.
pub(in crate::world) struct MeshLane;
impl LaneSpec for MeshLane {
    type Key = Coord;
    /// Low floor: admits are relatively cheap.
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
        // Ready if needs mesh, in view, has all neighbour data, and either light
        // is settled or wait timeout expired (then mesh degraded and remesh later).
        // Quarantined (repeatedly panicking) meshes are never ready.
        world.is_needs_mesh(key)
            && !world.quarantined.contains(&streaming::FailKey::Mesh { coord: key })
            && world.in_mesh_box(key)
            && world.neighbours_have_data(key)
            && (world.light_ready(key) || world.light_wait_expired(key))
    }
    fn submit(world: &mut World, key: Coord) -> Option<pipeline::Job> {
        world.refresh_tables();
        // If light isn't ready, mesh degraded with assumed-lit neighbours,
        // then remesh when real light arrives.
        let degraded = !world.light_ready(key);
        let (rev, snapshot) = world.snapshot(key, degraded);
        world.mark_degraded(key, degraded);
        Some(pipeline::Job::Mesh { coord: key, rev, snapshot })
    }
    fn claim(world: &mut World, key: Coord) {
        // Set building flag to claim the mesh job. Held until upload retires
        // it or a stale result releases it. Remove from worklist.
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

/// LOD2 column sections.
pub(in crate::world) struct SectionLane;
impl LaneSpec for SectionLane {
    type Key = SectionPos;
    /// Modest floor: heavy work runs off-thread.
    const MIN_ADMIT: usize = 4;
    fn candidates(world: &World, center: Coord) -> Candidates<SectionPos> {
        // Start with desired sections, filter those already loaded and those
        // provably inside the coverage clip (skip load).
        Candidates::Geometry(
            world
                .desired_sections(center)
                .into_iter()
                .filter(|s| {
                    !world.sections.contains_key(s)
                        && !world.quarantined.contains(&streaming::FailKey::Section { pos: *s })
                        && !world.coverage_skips(center, *s)
                })
                .collect(),
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
    fn dist2(world: &World, center: Coord, key: SectionPos) -> Option<u64> {
        // 2-D far field: use player's Y to drop vertical term.
        let span = key.span() as i64;
        let py = center.y as i64 * CHUNK_SIZE as i64 + CHUNK_SIZE as i64 / 2;
        let (cx, cz) = (key.min_x() as i64 + span / 2, key.min_z() as i64 + span / 2);
        let base = player_dist2(center, cx, py, cz);
        // Bias by motion direction so leading edge fills first.
        let cs = CHUNK_SIZE as i64;
        let (px, pz) = (center.x as i64 * cs + cs / 2, center.z as i64 * cs + cs / 2);
        Some(motion_biased_dist2(base, world.section_vel, (cx - px) as f64, (cz - pz) as f64))
    }
    fn in_flight(world: &World, key: SectionPos) -> bool {
        world.sections.contains_key(&key)
    }
    fn submit(world: &mut World, key: SectionPos) -> Option<pipeline::Job> {
        world.refresh_tables();
        Some(pipeline::Job::Section {
            pos: key,
            generator: world.generator.clone(),
            edits: world.edits_for_section(key),
            tables: world.tables.get(),
        })
    }
    fn claim(world: &mut World, key: SectionPos) {
        world.dirty_sections.remove(&key);
        world.sections.insert(key, SectionState::Meshing);
    }
    fn integrate(world: &mut World, done: pipeline::Done) {
        if let pipeline::Done::Section { pos, meshes } = done {
            world.section_upload_queue.push_back((pos, meshes));
        }
    }
}

/// Cross-chunk light settling.
pub(in crate::world) struct LightLane;
impl LaneSpec for LightLane {
    type Key = Coord;
    /// High floor: settle must drain fast so meshing can start.
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
        if !world.lighting
            || !world.chunks.contains_key(&key)
            || world.quarantined.contains(&streaming::FailKey::Light { coord: key })
        {
            return None;
        }
        world.refresh_tables();
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
        world.light_worklist.remove(&key);
        world.light_inflight.insert(key);
    }
    fn integrate(world: &mut World, done: pipeline::Done) {
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

    fn lod2_world() -> World {
        World::with_config(DEFAULT_SEED, RenderConfig::default())
    }

    #[test]
    fn lod2_is_the_default_and_near_only_leaves_sections_dormant() {
        assert!(World::generate().lod2, "lod2 far field on by default");
        let d = World::with_config(DEFAULT_SEED, RenderConfig { lod2: false, ..RenderConfig::default() });
        assert!(!d.lod2 && d.sections.is_empty() && d.section_visible.is_empty());
    }

    #[test]
    fn section_lane_claim_and_integrate_parity() {
        let mut world = lod2_world();
        let center = ChunkCoord::new(0, 0, 0);
        world.center = Some(center);
        let pos = world.desired_sections(center)[0];
        assert!(!<SectionLane as LaneSpec>::in_flight(&world, pos));
        <SectionLane as LaneSpec>::claim(&mut world, pos);
        assert!(matches!(world.sections.get(&pos), Some(SectionState::Meshing)));
        assert!(<SectionLane as LaneSpec>::in_flight(&world, pos), "claim marks in-flight");
        <SectionLane as LaneSpec>::integrate(
            &mut world,
            pipeline::Done::Section { pos, meshes: Default::default() },
        );
        assert_eq!(world.section_upload_queue.len(), 1, "landing queued for upload");
    }

    #[test]
    fn section_covering_gates_on_a_ready_ancestor_or_self() {
        // A cell is covered by a Ready self or by a Ready ancestor.
        let mut world = lod2_world();
        let center = ChunkCoord::new(0, 0, 0);
        let cell = world.desired_sections(center)[0];
        assert!(!world.section_covered(cell), "nothing loaded means uncovered");
        let empty_ready = || SectionState::Ready { quadrants: Default::default() };
        world.sections.insert(cell, empty_ready());
        assert!(world.section_covered(cell), "a Ready self covers");
        world.sections.remove(&cell);
        world.sections.insert(cell.parent(), empty_ready());
        assert!(world.section_covered(cell), "a Ready ancestor covers the finer cell");
    }

    /// A settled born-air chunk for testing coverage skip logic.
    fn air_chunk(cx: i32, cy: i32, cz: i32) -> Loaded {
        Loaded {
            chunk: std::sync::Arc::new(Chunk::from_uniform(cx, cy, cz, AIR)),
            state: MeshState::Air,
            rev: 0,
            connectivity: None,
            light: None,
        }
    }

    /// Verify coverage_skips skips only provably-covered sections inside the slab,
    /// and only when backed by settled chunks.
    #[test]
    fn coverage_skip_is_sound_and_backed() {
        use super::heightmip::BakeExtent;
        let mut world = lod2_world();
        let center = ChunkCoord::new(0, 0, 0);
        world.center = Some(center);
        world.view = ViewVolume::view(20);
        world.section_mip = Some(HeightMip::bake(
            &world.generator,
            &world.registry.color_snapshot(),
            BakeExtent::new(2048, section::FINEST_DETAIL + 3),
        ));
        let mip = world.section_mip.clone().unwrap();

        let cell = SectionPos { detail: section::FINEST_DETAIL, x: 0, z: 0 };
        let (lo, hi) = mip.relief_band(cell).expect("near cell is baked");
        // Centre the eye on the cell's relief so its terrain sits inside the slab.
        world.section_eye_y = ((lo + hi) * 0.5) as f64;

        // Selection stays total: skip is invisible to covering.
        assert!(world.desired_sections(center).contains(&cell), "cell still desired");

        // With nothing loaded, footprint is unbacked, so don't skip.
        assert!(!world.coverage_skips(center, cell), "unbacked near disc must not be skipped");

        // Back the footprint with settled chunks.
        let cs = CHUNK_SIZE as i32;
        let nchunks = cell.span() / cs;
        let (cy_lo, cy_hi) = ((lo.floor() as i32).div_euclid(cs), (hi.floor() as i32).div_euclid(cs));
        for cy in cy_lo..=cy_hi {
            for cz in 0..nchunks {
                for cx in 0..nchunks {
                    world.chunks.insert(ChunkCoord::new(cx, cy, cz), air_chunk(cx, cy, cz));
                }
            }
        }
        assert!(world.coverage_skips(center, cell), "backed near disc inside the slab is skipped");

        // If any chunk is in-flight, don't skip (fast-descent guard).
        world.chunks.insert(ChunkCoord::new(0, cy_lo, 0), Loaded {
            state: MeshState::NeedsMesh { building: true },
            ..air_chunk(0, cy_lo, 0)
        });
        assert!(!world.coverage_skips(center, cell), "an in-flight covering chunk blocks the skip");
    }

    /// Verify sections outside the skipped core are never skipped.
    #[test]
    fn coverage_skip_never_skips_outside_the_core() {
        use super::heightmip::BakeExtent;
        let mut world = lod2_world();
        let center = ChunkCoord::new(0, 0, 0);
        world.center = Some(center);
        world.view = ViewVolume::view(20);
        world.section_mip = Some(HeightMip::bake(
            &world.generator,
            &world.registry.color_snapshot(),
            BakeExtent::new(2048, section::FINEST_DETAIL + 3),
        ));
        let mip = world.section_mip.clone().unwrap();

        // Far section: clip draws it, so don't skip it.
        let far = SectionPos { detail: section::FINEST_DETAIL, x: 5, z: 0 };
        if let Some((lo, hi)) = mip.relief_band(far) {
            world.section_eye_y = ((lo + hi) * 0.5) as f64;
        }
        assert!(!world.coverage_skips(center, far), "a far section is never skipped");

        // Near section but eye is high above it: terrain pokes out of slab, so don't skip.
        let near = SectionPos { detail: section::FINEST_DETAIL, x: 0, z: 0 };
        world.section_eye_y = 5000.0;
        assert!(!world.coverage_skips(center, near), "high eye over low ground is not skipped");
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
        assert!(world.is_solid(8, -40, 8), "no world floor: stone all the way down");
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
        let world = World::generate();
        for center in [
            DVec3::new(15.9, 18.0, 15.9),
            DVec3::new(0.1, 21.5, 8.0),
            DVec3::new(-3.2, 19.0, -16.4),
            DVec3::new(4.0, -1.0, 4.0),
            DVec3::new(4.0, 15.9, 4.0),
            DVec3::new(4.0, 200.0, 4.0),
        ] {
            let aabb = Aabb::new(center, DVec3::new(0.4, 0.9, 0.4));
            // Collision skips liquids (solid to mesher, passable to collision).
            let reference = aabb.voxel_cells().any(|(x, y, z)| world.is_obstacle(x, y, z));
            assert_eq!(world.collides(&aabb), reference, "at {center:?}");
        }
    }

    #[test]
    fn column_is_layered_grass_dirt_stone() {
        let mut world = World::generate();
        let reg = world.registry();
        // Terrain speaks elements now: the crust blocks are the natural unions
        // the placement table derives, not the authored Grass/Dirt mixtures
        // (which remain registered for crafting and old saves).
        let (grass, dirt, stone) = (
            reg.id_by_name("Soil+Organic").unwrap(),
            reg.id_by_name("Soil+Clay").unwrap(),
            reg.id_by_name("Stone").unwrap(),
        );

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
        // Deep stone persists below y = 0.
        world.ensure_data(World::chunk_of(x, h - 70, z));
        assert_eq!(world.block_at(x, h - 70, z), stone);
    }

    #[test]
    fn restoring_the_generated_block_compacts_the_overlay() {
        // Edit-and-revert must leave NO overlay weight: regeneration produces
        // the reverted block anyway, so saves and join transfers stay
        // proportional to the world's real difference from its seed.
        let mut world = World::generate();
        let (x, z) = (8, 8);
        let h = (0..96).rev().find(|&y| world.is_solid(x, y, z)).unwrap();
        let original = world.block_at(x, h, z);
        assert_eq!(world.edits().count(), 0);
        world.set_block(x, h, z, AIR);
        assert_eq!(world.edits().count(), 1, "a real edit is recorded");
        world.set_block(x, h, z, original);
        assert_eq!(world.edits().count(), 0, "restoring generation drops the entry");
    }

    #[test]
    fn edits_persist_across_unload() {
        let mut world = World::generate();
        let (x, z) = (8, 8);
        let h = (0..64)
            .rev()
            .find(|&y| world.is_solid(x, y, z))
            .unwrap();
        world.set_block(x, h, z, AIR);
        assert_eq!(world.block_at(x, h, z), AIR);
        let stone = world.registry().id_by_name("Stone").unwrap();
        world.set_block(x, 70, z, stone);

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

        world.light_worklist.clear();
        world.chunks.get_mut(&coord).unwrap().light = Some(light::LightGrid::dark());
        world.light_inflight.insert(coord);
        assert!(world.transition_lighting(false));
        let off_epoch = world.light_epoch;
        assert!(!world.lighting());
        assert!(world.light_worklist.is_empty());
        assert!(world.light_inflight.is_empty());
        assert!(world.chunks[&coord].light.is_none());

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

        // Results from old generation don't publish after re-enable.
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
    fn failed_jobs_release_claims_then_quarantine_after_repeated_strikes() {
        let mut world = World::generate();
        let coord = *world.chunks.keys().next().unwrap();

        // Mesh lane: a panicked build releases the claim and re-seeds the worklist.
        world.chunks.get_mut(&coord).unwrap().state = MeshState::NeedsMesh { building: true };
        world.mesh_worklist.remove(&coord);
        world.fail_job(pipeline::JobKey::Mesh { coord });
        assert!(
            matches!(world.chunks[&coord].state, MeshState::NeedsMesh { building: false }),
            "the build claim must be released"
        );
        assert!(world.mesh_worklist.contains(&coord), "released work is re-seeded");

        // Strike out: the third failure quarantines and stops re-seeding.
        world.fail_job(pipeline::JobKey::Mesh { coord });
        world.mesh_worklist.remove(&coord);
        world.fail_job(pipeline::JobKey::Mesh { coord });
        assert!(world.quarantined.contains(&streaming::FailKey::Mesh { coord }));
        assert!(!world.mesh_worklist.contains(&coord), "quarantined claims are not re-seeded");
        assert!(
            !<MeshLane as LaneSpec>::ready(&world, coord),
            "the mesh lane skips a quarantined coord"
        );

        // Light lane: claim released and re-seeded, then quarantined likewise.
        world.light_inflight.insert(coord);
        world.light_worklist.remove(&coord);
        world.fail_job(pipeline::JobKey::Light { coord });
        assert!(!world.light_inflight.contains(&coord));
        assert!(world.light_worklist.contains(&coord));
        for _ in 0..2 {
            world.light_inflight.insert(coord);
            world.fail_job(pipeline::JobKey::Light { coord });
        }
        assert!(!world.light_inflight.contains(&coord), "claim always releases");
        assert!(world.quarantined.contains(&streaming::FailKey::Light { coord }));
        assert!(
            <LightLane as LaneSpec>::submit(&mut world, coord).is_none(),
            "a quarantined light claim never resubmits"
        );

        // Generate lane: every claimed coord in the failed column span clears.
        let (cx, cz) = (100, 100);
        for cy in 0..=2 {
            world.generating.insert(ChunkCoord::new(cx, cy, cz));
        }
        world.fail_job(pipeline::JobKey::Column { col: (cx, cz), cy: 0..=2 });
        assert!((0..=2).all(|cy| !world.generating.contains(&ChunkCoord::new(cx, cy, cz))));

        // Section lane: a panicked Meshing claim is dropped so selection retries.
        let pos = SectionPos { detail: 2, x: 9, z: 9 };
        world.sections.insert(pos, SectionState::Meshing);
        world.fail_job(pipeline::JobKey::Section { pos });
        assert!(!world.sections.contains_key(&pos), "the Meshing claim must clear");
    }

    /// Descheduled (left-behind) jobs release their claims like panics do,
    /// but with NO strike, NO quarantine, and no forced requeue — coming back
    /// later must re-request the work as if it had never been claimed.
    #[test]
    fn cancelled_jobs_release_claims_without_strikes() {
        let mut world = World::generate();
        let coord = *world.chunks.keys().next().unwrap();

        world.chunks.get_mut(&coord).unwrap().state = MeshState::NeedsMesh { building: true };
        world.cancel_job(pipeline::JobKey::Mesh { coord });
        assert!(matches!(world.chunks[&coord].state, MeshState::NeedsMesh { building: false }));

        world.light_inflight.insert(coord);
        world.cancel_job(pipeline::JobKey::Light { coord });
        assert!(!world.light_inflight.contains(&coord));

        let (cx, cz) = (200, 200);
        for cy in 0..=1 {
            world.generating.insert(ChunkCoord::new(cx, cy, cz));
        }
        world.cancel_job(pipeline::JobKey::Column { col: (cx, cz), cy: 0..=1 });
        assert!((0..=1).all(|cy| !world.generating.contains(&ChunkCoord::new(cx, cy, cz))));

        assert!(world.quarantined.is_empty(), "cancellation is not a failure");
        assert!(world.job_strikes.is_empty(), "cancellation earns no strikes");
    }

    #[test]
    fn stale_rev_mesh_results_are_dropped() {
        let mut world = World::generate();
        world.center = Some(ChunkCoord::new(0, 0, 0));
        let coord = ChunkCoord::new(0, 0, 0);
        let rev = world.chunks[&coord].rev;
        assert!(world.mesh_result_applies(coord, rev));

        world.set_block(3, 3, 3, AIR);
        assert!(!world.mesh_result_applies(coord, rev));

        world.pending_fresh.take();
        world.accept_mesh(coord, rev, Box::new(new_chunk_mesh_data()));
        assert!(world.upload_queue.is_empty(), "stale result never queues");
        assert!(world.pending_fresh.get(), "drop re-arms the scan");

        let rev = world.chunks[&coord].rev;
        world.accept_mesh(coord, rev, Box::new(new_chunk_mesh_data()));
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
        let before = world.chunks[&ChunkCoord::new(-1, 0, 0)].rev;
        world.set_block(0, 5, 8, AIR);
        assert_eq!(world.chunks[&ChunkCoord::new(-1, 0, 0)].rev, before + 1, "border neighbour");
        assert_eq!(world.chunks[&ChunkCoord::new(0, 0, 0)].rev, 1, "edited chunk itself");
        assert_eq!(world.chunks[&ChunkCoord::new(1, 0, 0)].rev, 0, "far side untouched");

        // Vertical borders also bump rev on the neighbour below.
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
        world.chunks.remove(&coord);
        world.set_block(x, 5, z, AIR);
        let raw = Chunk::new(coord.x, coord.y, coord.z, &world.generator);
        assert_ne!(raw.get_local(3, 5, 4), AIR, "terrain is solid there");
        world.pending_fresh.take();
        world.accept_chunk(coord, raw);
        assert_eq!(world.block_at(x, 5, z), AIR, "overlay replayed on landing");
        assert!(world.pending_fresh.get(), "new data re-arms the fresh scan");

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
        let sky = &world.chunks[&ChunkCoord::new(0, 3, 0)];
        assert_eq!(sky.chunk.uniform(), Some(AIR));
        assert_eq!(sky.state, MeshState::Air, "uniform air is born Air");
        assert!(sky.state.live_meshes().is_none());
        let ground = &world.chunks[&ChunkCoord::new(0, 0, 0)];
        assert_eq!(
            ground.state,
            MeshState::NeedsMesh { building: false },
            "dense terrain waits for a real mesh"
        );
    }

    #[test]
    fn handle_is_tracked_exactly_once_across_edits() {
        let mut world = World::generate();
        let coord = ChunkCoord::new(0, 0, 0);
        let h = MeshHandle::from_raw_parts(42, 3);
        world.chunks.get_mut(&coord).unwrap().state = ready(h);

        world.set_block(3, 3, 3, AIR);
        assert_eq!(world.chunks[&coord].state, MeshState::Dirty { prev: Some(meshes(h)) });

        world.set_block(4, 4, 4, AIR);
        assert_eq!(
            world.chunks[&coord].state,
            MeshState::Dirty { prev: Some(meshes(h)) },
            "re-edit preserves the single handle"
        );

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
        let mut world = World::generate();
        world.center = Some(ChunkCoord::new(0, 0, 0));
        let coord = ChunkCoord::new(0, 0, 0);
        world.chunks.get_mut(&coord).unwrap().state = MeshState::NeedsMesh { building: true };
        let rev = world.chunks[&coord].rev;

        world.set_block(2, 2, 2, AIR);
        assert!(world.chunks[&coord].state.is_dirty(), "edit turns a building chunk into Dirty");
        assert_ne!(world.chunks[&coord].rev, rev, "edit bumps rev");

        world.pending_fresh.take();
        world.accept_mesh(coord, rev, Box::new(new_chunk_mesh_data()));
        assert!(world.upload_queue.is_empty(), "stale mesh result never queues");
        assert!(world.chunks[&coord].state.is_dirty(), "chunk stays Dirty for the sync remesh");
        assert!(world.pending_fresh.get(), "drop re-arms the fresh scan");
    }

    #[test]
    fn mesh_result_stale_by_box_exit_releases_the_claim() {
        // Stale result when chunk left the mesh box (no edit, no rev bump)
        // must release the in-flight claim so the chunk is re-meshable.
        let mut world = World::generate();
        let coord = ChunkCoord::new(0, 0, 0);
        world.center = Some(coord);
        let rev = world.chunks[&coord].rev;
        world.chunks.get_mut(&coord).unwrap().state = MeshState::NeedsMesh { building: true };
        world.center = Some(ChunkCoord::new(1000, 0, 0));
        assert!(!world.mesh_result_applies(coord, rev), "out-of-box result is stale");

        world.pending_fresh.take();
        world.accept_mesh(coord, rev, Box::new(new_chunk_mesh_data()));
        assert!(world.upload_queue.is_empty(), "stale result never queues");
        assert_eq!(
            world.chunks[&coord].state,
            MeshState::NeedsMesh { building: false },
            "claim released, chunk is re-meshable"
        );
        assert!(world.mesh_worklist.contains(&coord), "re-seeded for a later mesh");
        assert!(world.pending_fresh.get(), "drop re-arms the fresh scan");
    }

    #[test]
    fn view_volume_vertical_is_derived_and_flatter() {
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
    fn motion_bias_reorders_toward_heading_and_is_identity_at_rest() {
        let base = 1_000_000u64;
        let vx = DVec3::new(1.0, 0.0, 0.0);
        let ahead = motion_biased_dist2(base, vx, 500.0, 0.0);
        let behind = motion_biased_dist2(base, vx, -500.0, 0.0);
        assert!(ahead < base, "cell ahead of motion sorts sooner");
        assert!(behind > base, "cell behind motion sorts later");
        assert_eq!(base - ahead, behind - base, "symmetric about the base key");
        assert_eq!(motion_biased_dist2(base, vx, 0.0, 500.0), base, "perpendicular is unbiased");
        assert_eq!(motion_biased_dist2(base, DVec3::ZERO, 500.0, 0.0), base, "identity at rest");
        assert_eq!(
            motion_biased_dist2(base, DVec3::new(0.1, 0.0, 0.0), 500.0, 0.0),
            base,
            "below speed floor is identity"
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
        assert!(world.pending_fresh.get());
        assert_eq!(world.center, None);
    }

    #[test]
    fn queued_shrink_survives_an_intervening_grow() {
        let mut world = World::generate();
        world.set_view_radius(4);
        world.set_view_radius(8);
        assert!(world.radius_shrunk.get(), "grow must not clobber a queued shrink");
        assert!(world.radius_shrunk.take());
        assert!(!world.radius_shrunk.get(), "take consumes it");
    }

    #[test]
    fn invalidate_carries_or_clears_the_owned_mesh() {
        let h = MeshHandle::from_raw_parts(5, 2);
        let mut s = ready(h);
        s.invalidate();
        assert_eq!(s, MeshState::Dirty { prev: Some(meshes(h)) });
        s.invalidate();
        assert_eq!(s, MeshState::Dirty { prev: Some(meshes(h)) });
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
        let state = ready(h);
        assert!(state.live_meshes().unwrap().draws(h));
        assert!(matches!(state.into_owned(), Some(m) if m.draws(h)));
        let air = MeshState::from_upload(ByPass::from_fn(|_| None));
        assert_eq!(air, MeshState::Air);
        assert!(air.live_meshes().is_none());
        assert!(air.into_owned().is_none());
    }

    #[test]
    fn edit_raises_pending_dirty_and_enters_the_dirty_fiber() {
        let mut world = World::generate();
        assert!(!world.pending_dirty.get());
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
                    *s = SectionState::Ready { quadrants: Default::default() };
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
