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
pub mod brick;
pub mod chunk;
pub mod connectivity;
pub mod generation;
pub mod light;
pub mod lod;
pub mod mesh;
mod neighborhood;
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
mod coverage;
pub(crate) mod lanes;

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::sync::Arc;
use std::sync::mpsc::Receiver;
use std::time::Instant;

use voxel_engine::producer::Progress;
use voxel_engine::{CoverageVolume, DVec3, Detail, Engine, FadeStyle, Frame3D, MeshHandle};

use crate::block::registry::{BlockId, BlockRegistry, HotTables};
use crate::coord::{ByPass, ChunkBox, ChunkCoord};
use crate::render::Render;
use chunk::{CHUNK_SIZE, Chunk};
use generation::{SineHills, TerrainGenerator};
use heightmip::HeightMip;
use light::LightGrid;
use mesh::{ChunkMeshData, new_chunk_mesh_data};
use quadtree::QuadrantMask;
use section::Quadrant;
use section::{SectionMeshData, SectionPos};

/// Default number of chunk rings meshed and drawn around the player.
const DEFAULT_VIEW_RADIUS: i32 = 6;
/// The range a runtime render-distance change is clamped to. Shared: the
/// settings model and the settings menu stepper clamp to the same bounds.
/// Zero draws only the current chunk column; the [`DATA_MARGIN`] shell still
/// keeps an unmeshed collision/data halo around the player.
pub const VIEW_RADIUS_RANGE: std::ops::RangeInclusive<i32> = 0..=20;
/// The range a runtime vertical-distance change is clamped to: chunk layers
/// streamed above and below the eye. Shared with the settings model so the
/// menu stepper and the world clamp to the same bounds.
pub const VERTICAL_RADIUS_RANGE: std::ops::RangeInclusive<i32> = 1..=10;
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

/// Live streaming-queue depths — the numeric twin of
/// [`entry_debug`](World::entry_debug)'s formatted counters, for the harness's
/// stress metrics (peak backlog depths, settle progress) where parsing a
/// debug string would be absurd.
#[derive(Clone, Copy, Debug, Default)]
pub struct StreamGauges {
    pub chunks: usize,
    pub generating: usize,
    pub mesh_worklist: usize,
    pub upload_queue: usize,
    pub light_worklist: usize,
    pub light_inflight: usize,
    pub light_apply_queue: usize,
}
use connectivity::{Connectivity, Occlusion};

/// The streamed chunk volume around the player: a horizontal ring radius and a
/// (smaller) vertical layer radius, in chunks. **Anisotropic**: interesting
/// terrain is mostly lateral, so a full cube would load a tall column of empty
/// sky and deep rock that never produces drawable geometry — tripling the
/// loaded set (and every O(loaded) streaming pass) for nothing on screen. The
/// vertical radius can be configured independently, tall enough to keep the
/// ground under vertical movement (jumping, cliffs, flight) without paying
/// for the whole render sphere. The compatibility render-radius setter still
/// uses [`vertical_for`] to preserve its historical derivation.
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
    /// Explicit anisotropic volume. Runtime setters clamp before construction.
    fn new(horizontal: i32, vertical: i32) -> Self {
        Self { horizontal, vertical }
    }
    /// The streamed volume for a horizontal view radius, with the vertical
    /// radius derived from it.
    fn view(horizontal: i32) -> Self {
        Self::new(horizontal, Self::vertical_for(horizontal))
    }
    /// The full-res coverage slab in metres: the shader's LOD-cull volume, which
    /// must equal this streamed full-res volume (one source of truth for both).
    fn coverage(&self) -> CoverageVolume {
        CoverageVolume {
            radius: (self.horizontal * CHUNK_SIZE as i32) as f32,
            half_height: (self.vertical * CHUNK_SIZE as i32) as f32,
        }
    }

    /// The far-field LOD ring unit in metres. Cannot be zero even when the
    /// stripped near field draws only the current chunk column.
    pub(in crate::world) fn lod_unit(&self) -> f32 {
        (self.horizontal.max(1) * CHUNK_SIZE as i32) as f32
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
    /// Push far-material style onto each present pass's resident mesh (LOD
    /// sections). Placement and detail are pinned at upload; the engine delta-gates
    /// identical style. Full-res chunks never call it (default textured style).
    fn set_style(&self, eng: &mut Engine, style: FadeStyle, flat_rgba: u32) {
        for (_, m) in self.0.iter() {
            if let Some(mesh) = m {
                eng.set_mesh_style(mesh.id(), style, flat_rgba);
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
    /// Patch every present pass's slot in the GPU visibility mask.
    fn set_visible(&self, eng: &mut Engine, on: bool) {
        for (_, m) in self.0.iter() {
            if let Some(mesh) = m {
                eng.set_visible(mesh.id(), on);
            }
        }
    }
    /// Whether any pass draws `handle` — for the render/ownership tests.
    #[cfg(test)]
    fn draws(&self, handle: MeshHandle) -> bool {
        self.0.iter().any(|(_, m)| m.as_ref().is_some_and(|o| o.id() == handle))
    }
}

// Per-draw style (flat palette-average colour) is the typed engine `FadeStyle`
// now — no raw mode bits.
/// Detail threshold for flat palette-average colour (far-material optimization).
/// The outer rings lose per-texel detail to sub-pixel shimmer, so average colour
/// reduces bandwidth; nearer rings keep texture. Inert until the mip bake lands.
const FLAT_DETAIL: Detail = Detail(section::FINEST_DETAIL.0 + 4);

/// A section is either meshing or ready. Blocks are pre-positioned at upload,
/// so draw needs only camera-relative offset arithmetic.
pub(in crate::world) enum SectionState {
    /// Claimed by an in-flight worker job carrying this unique token (see
    /// [`pipeline::ClaimToken`]). A result, cancellation, or failure that
    /// presents a different token belongs to a superseded claim and must not
    /// touch this entry.
    Meshing { token: pipeline::ClaimToken },
    /// Block meshes grouped by quadrant (indexed by [`SectionPos::quadrant`]).
    /// Position and detail (cell size `2^detail`, from the map key `pos.detail`)
    /// are pinned into each resident mesh at upload; visibility is a per-quadrant
    /// `set_visible` mask and style a `set_style` push, so nothing per-block is
    /// stored beyond the meshes themselves.
    Ready {
        quadrants: [Vec<ChunkMeshes>; 4],
        /// Last `(style, flat_rgba)` pushed via [`Self::push_style`], so a value
        /// re-observed next frame (the steady case) sends nothing.
        last_style: Option<(FadeStyle, u32)>,
    },
}

impl SectionState {
    /// Upload each quadrant's block meshes at their absolute world origin and
    /// detail (pinned once — the engine recovers camera-relative position). Empty
    /// quadrants upload to no handles.
    fn from_upload(pos: SectionPos, meshes: [SectionMeshData; 4], eng: &mut Engine) -> SectionState {
        let cell = pos.cell_size();
        let detail = pos.detail;
        let quadrants = meshes.map(|quad| {
            let mut blocks = Vec::new();
            for (block_origin, data) in quad {
                let placement = voxel_engine::MeshPlacement::terrain(
                    voxel_engine::IVec3::new(
                        pos.min_x() + block_origin.x as i32 * cell,
                        block_origin.y as i32 * cell,
                        pos.min_z() + block_origin.z as i32 * cell,
                    ),
                    detail,
                );
                let handles = ByPass::from_fn(|p| eng.upload_mesh_placed(&data[p], placement));
                if let Some(meshes) = ChunkMeshes::from_upload_handles(handles) {
                    blocks.push(meshes);
                }
            }
            blocks
        });
        SectionState::Ready { quadrants, last_style: None }
    }

    fn is_ready(&self) -> bool {
        matches!(self, SectionState::Ready { .. })
    }

    /// Project `mask` onto this region's slots: the quadrants it selects are
    /// visible, the rest not. `None` draws nothing. Walks EVERY quadrant, so a
    /// quadrant leaving the mask is cleared rather than stranded at its last value.
    fn set_visible(&self, eng: &mut Engine, mask: Option<QuadrantMask>) {
        let SectionState::Ready { quadrants, .. } = self else {
            return;
        };
        for q in Quadrant::ALL {
            let on = mask.is_some_and(|m| m.contains(q));
            for meshes in &quadrants[q.index()] {
                meshes.set_visible(eng, on);
            }
        }
    }
    /// Push the far-material style onto every block mesh of this section
    /// (visibility decides which actually draw). The engine delta-gates unchanged style.
    fn set_style(&self, eng: &mut Engine, style: FadeStyle, flat_rgba: u32) {
        let SectionState::Ready { quadrants, .. } = self else {
            return;
        };
        for quad in quadrants {
            for meshes in quad {
                meshes.set_style(eng, style, flat_rgba);
            }
        }
    }
    /// [`Self::set_style`], gated on the pushed tuple actually changing since last
    /// time — the DrawDyn contract ("at rest, zero writes"): the engine delta-gates
    /// per-mesh too, but a settled far field must not even attempt the walk.
    fn push_style(&mut self, eng: &mut Engine, style: FadeStyle, flat_rgba: u32) {
        let key = (style, flat_rgba);
        let SectionState::Ready { last_style, .. } = self else {
            return;
        };
        if *last_style == Some(key) {
            return;
        }
        *last_style = Some(key);
        self.set_style(eng, style, flat_rgba);
    }
    fn free(self, eng: &mut Engine) {
        if let SectionState::Ready { quadrants, .. } = self {
            for quad in quadrants {
                for meshes in quad {
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
    /// Dense data awaiting a fresh ASYNC mesh. `building` is the in-flight
    /// claim: `true` once a mesh job is outstanding on the worker pool (`rev`
    /// on `Loaded` referees its result), held until the budgeted upload
    /// resolves or the result is dropped. `prev` is the still-drawn old mesh
    /// of a chunk whose rebuild was requested asynchronously (a light grid
    /// landed, a degraded mesh's real light arrived — see
    /// [`World::remesh_async`]): it keeps drawing until the fresh upload
    /// retires it, so an async relight never blanks the chunk. `None` for a
    /// never-meshed chunk.
    NeedsMesh { building: bool, prev: Option<ChunkMeshes> },
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
            MeshState::Ready(m)
            | MeshState::Dirty { prev: Some(m) }
            | MeshState::NeedsMesh { prev: Some(m), .. } => Some(m),
            _ => None,
        }
    }

    /// Whether this chunk contributes its final ground truth to the frame:
    /// something is drawn for it, or there is provably nothing to draw
    /// (born-air). THE handoff predicate — near-section admission skipping
    /// and the settled LOD clip both gate on it, so "the chunks cover this"
    /// can never mean two different things.
    fn settled(&self) -> bool {
        self.live_meshes().is_some() || matches!(self, MeshState::Air)
    }
    /// Move the owned meshes out for freeing. Consumes `self`; the caller owns
    /// the token afterwards and must `free` it (or carry it on).
    #[must_use]
    fn into_owned(self) -> Option<ChunkMeshes> {
        match self {
            MeshState::Ready(m)
            | MeshState::Dirty { prev: Some(m) }
            | MeshState::NeedsMesh { prev: Some(m), .. } => Some(m),
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
    /// A fresh never-meshed (or reset) state: not building, nothing carried.
    fn needs_mesh() -> MeshState {
        MeshState::NeedsMesh { building: false, prev: None }
    }
    /// Invalidate to `Dirty`, carrying the currently-drawn mesh forward as
    /// `prev` so it keeps drawing until the sync remesh. Nothing is freed here
    /// — the token just moves. `Ready(m) -> Dirty{Some(m)}`; an already-`Dirty`
    /// chunk keeps its `prev`, and so does a `NeedsMesh` carrying one (an edit
    /// landing mid-async-rebuild keeps drawing the old mesh); other handle-less
    /// states (incl. a bare `building` chunk, whose in-flight claim is dropped
    /// — the sync remesh takes over and the orphan async result is refereed
    /// out by `rev`) -> `Dirty{None}`.
    fn invalidate(&mut self) {
        let prev = std::mem::replace(self, MeshState::needs_mesh()).into_owned();
        *self = MeshState::Dirty { prev };
    }
    /// Release the in-flight mesh claim if this chunk is still awaiting its
    /// build (a carried `prev` keeps drawing). A no-op once the chunk has
    /// moved on (`Dirty` via an edit, `Ready`/`Air` via a prior consume):
    /// those states carry no claim. Called at every mesh-result-consumption
    /// site whose result did NOT apply, so a stale result (view moved, chunk
    /// left the box) can never wedge the claim.
    fn release_build(&mut self) {
        if let MeshState::NeedsMesh { building, .. } = self {
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
    generator: std::sync::Arc<SineHills>,
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
    /// The [`GenerateLane`](lanes::GenerateLane)'s raise-then-consume gate: the
    /// data box has columns to request. Raised on a boundary cross and by a
    /// generate strike-out re-request; drained when the box is fully requested.
    pending_gen: Sticky,
    /// Finished meshes awaiting budgeted upload (re-validated at upload time for staleness).
    upload_queue: VecDeque<(Coord, u32, pipeline::MeshOutput)>,
    /// Chunks needing a *fresh* mesh (the [`MeshLane`] seed set — replaces the
    /// old whole-map rescan `pending_fresh` armed). Seeded on load (self + 6
    /// neighbours), on a light publish that moved a border, and on an
    /// accept_mesh stale drop. Drained nearest-first by the mesh lane, so the
    /// enqueue scan is O(shell) not O(cube).
    mesh_worklist: FastSet<Coord>,
    /// Whether the [`LightLane`] still has seeds/in-flight to drain (its
    /// `pending` gate — the [`StreamLane::pending`] accessor). Raised when the
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
    /// Scheduler handles for the `stream` call-point CPU lanes, set by
    /// `Game::new` after it registers them. `None` only before
    /// that wiring (a bare `World` with no scheduler never calls `stream`).
    stream_lanes: Option<lanes::StreamLanes>,
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
    /// Finished section meshes awaiting budgeted upload, tagged with the claim
    /// token that produced them (re-validated at the moment of upload).
    section_upload_queue: VecDeque<(SectionPos, pipeline::ClaimToken, [SectionMeshData; 4])>,
    /// Whether desired sections still need enqueueing (budget spreads a flood).
    pending_sections: Sticky,
    /// Sections invalidated by edits, freed and re-admitted from the generator.
    dirty_sections: FastSet<SectionPos>,
    /// Per-section edit generation: bumped (alongside `dirty_sections`) whenever an
    /// edit lands in a section's footprint at any active detail. Stamps
    /// `section_overlay_cache` so a cell is only ever re-derived when ITS OWN
    /// footprint changed (locality) — never on an edit elsewhere.
    section_edit_rev: FastMap<SectionPos, u64>,
    /// The edited chunk columns inside each ever-edited section's footprint —
    /// the index that makes per-section edit collection O(own edited chunks)
    /// instead of a scan of the WHOLE edit map per section (which the overlay
    /// refresh and every far-job submit used to pay). Entries whose edits
    /// compact away are filtered at read time (the chunk's edit map is gone),
    /// so stale coords cost a lookup, never wrong data. Edit-lifetime, like
    /// `section_edit_rev` — survives unload and ladder changes.
    section_edit_chunks: FastMap<SectionPos, FastSet<Coord>>,
    /// Sections whose edit overlay must be re-derived — the exact positions
    /// edits touched since the last overlay refresh. Empty on a quiet frame,
    /// so the refresh lane is a set-emptiness check and nothing more.
    section_overlay_dirty: FastSet<SectionPos>,
    /// Consecutive fully-[`settled`](MeshState::settled) chunk rings around
    /// the centre, `0..=horizontal+1` (`horizontal+1` = the whole mesh box).
    /// THE input to the settled LOD clip: far sections keep drawing over any
    /// column whose chunks are not yet on screen, so a loading edge shows
    /// coarse terrain instead of a hole. Maintained incrementally — grown
    /// outward on upload/air events (each ring re-scanned at most once per
    /// loading wave), reset by centre moves, unloads, and mesh teardown.
    lod_clip_rings: i32,
    /// A settle event landed (chunk mesh upload, born-air store): try to
    /// extend [`lod_clip_rings`](Self::lod_clip_rings) outward.
    lod_clip_grow: Sticky,
    /// The ring geometry or settledness regressed (centre moved, chunks
    /// unloaded or meshes freed): restart the ring scan from zero.
    lod_clip_shrunk: Sticky,
    /// The far covering must be re-resolved: set by every event that can move
    /// it (a section landing Ready, an unload/free, a claim release, a
    /// frontier change, a ladder change). While clear AND no admission is
    /// pending, the per-pass covering walk and visibility diff are skipped
    /// entirely; `pending_sections` doubles as the level-triggered backstop
    /// that keeps the old staleness fix (holes keep the lane re-arming).
    section_cover_dirty: Sticky,
    /// Memoized edit-folded cell, keyed by `section_edit_rev`. `None` means the
    /// cell's footprint currently has no edits (overlay compaction can revert
    /// this after having been `Some`); the pure bake applies unchanged.
    section_overlay_cache: voxel_engine::rev::DerivedMap<SectionPos, Option<heightmip::MipCell>>,
    /// This frame's resolved overlay snapshot (refreshed in `stream`'s `&mut`
    /// pass): the only edit-staleness fix `render`'s `&self` readers consult —
    /// occlusion, section colour, and the coverage-skip relief test all prefer
    /// this over `section_mip`'s immutable bake when a cell is present here.
    section_overlay: FastMap<SectionPos, heightmip::MipCell>,
    /// The desired section frontier computed ONCE per stream frame and shared
    /// by unloading, the load lane, and the covering rebuild — the selection
    /// sweep (grid walk + relief coarsening) used to run up to three times a
    /// frame for identical inputs.
    section_desired: Vec<SectionPos>,
    /// Visible sections (covering-resolved) — the desired cut this frame. Feeds
    /// `unload_sections` and the coverage projection.
    section_visible: Vec<(SectionPos, QuadrantMask)>,
    /// Drawn-coverage projection over successive LOD cuts (hard pop). Presentation
    /// only; never affects selection logic.
    section_fade: coverage::Coverage,
    /// The exact inputs [`section_desired`](Self::section_desired) was computed
    /// from. While they are unchanged the frontier is retained across streaming
    /// passes — a still camera repeats no grid walk or relief coarsening.
    section_frontier_key: Option<SectionFrontierKey>,
    /// Far-lane configuration epoch: bumped by every live ladder change so an
    /// in-flight worker result from a retired configuration can never land.
    section_epoch: u32,
    /// Monotone claim-token source for section jobs (see
    /// [`pipeline::ClaimToken`]); `section_pending_claim` carries the
    /// freshly minted token from the lane's `submit` to its `claim`.
    section_claim_seq: u64,
    section_pending_claim: Option<(SectionPos, pipeline::ClaimToken)>,
}

/// The exact inputs the desired-section frontier depends on, as cheap bit
/// patterns. Equality means the cached frontier is still the right answer.
/// Edits are handled separately: a dirty section forces a recompute (see
/// `World::stream`) because relief coarsening consults the edit overlay.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct SectionFrontierKey {
    center_xz: [i32; 2],
    eye_y: u64,
    velocity: [u64; 3],
    unit: u32,
    finest: i8,
    levels: u8,
    step: u8,
    mip_ready: bool,
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
        Self::with_config_inner(seed, render, true)
    }

    /// Construct without synchronously generating the full origin data box.
    /// Interactive startup uses this path and calls
    /// [`prepare_around`](Self::prepare_around) for the small collision-safe
    /// spawn slab; streaming fills the remainder asynchronously. Existing
    /// constructors retain eager data for tests and headless callers that
    /// query the origin before their first stream.
    pub fn with_config_lazy(seed: i64, render: crate::render_config::RenderConfig) -> Self {
        Self::with_config_inner(seed, render, false)
    }

    fn with_config_inner(
        seed: i64,
        render: crate::render_config::RenderConfig,
        pregenerate_origin: bool,
    ) -> Self {
        // `mut` for the placement compile: the generator registers every block
        // terrain can emit here at startup, then keeps only resolved ids.
        let mut registry = BlockRegistry::with_builtins();
        // Shared immutably with every worker job — an `Arc` bump instead of a
        // deep clone of the whole compiled terrain per job.
        let generator = std::sync::Arc::new(SineHills::new(&mut registry, 20.0, seed));
        // The section ladder's innermost ring begins where the full-res box ends,
        // so its `unit` is the render distance in metres.
        let unit = (DEFAULT_VIEW_RADIUS * CHUNK_SIZE as i32) as f32;
        let lod2 = render.lod2;
        let (lod_levels, lod_detail) = render.normalized_lod();
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
            pending_gen: Sticky::default(),
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
            occlusion: Occlusion::default(),
            occlusion_dirty: Sticky::default(),
            occlusion_active: false,
            occlusion_forced: render.occlusion,
            stream_lanes: None,
            lighting: true,
            light_epoch: 0,
            lod2,
            section_pyramid: pyramid::PyramidCfg::sections_with(unit, lod_levels, lod_detail),
            section_eye_y: 0.0,
            section_eye_prev: None,
            section_vel: DVec3::ZERO,
            section_mip: None,
            section_mip_rx: None,
            sections: FastMap::default(),
            section_upload_queue: VecDeque::new(),
            pending_sections: Sticky::default(),
            dirty_sections: FastSet::default(),
            section_edit_rev: FastMap::default(),
            section_edit_chunks: FastMap::default(),
            section_overlay_dirty: FastSet::default(),
            section_cover_dirty: Sticky::raised(),
            section_overlay_cache: voxel_engine::rev::DerivedMap::new(),
            section_overlay: FastMap::default(),
            section_desired: Vec::new(),
            section_visible: Vec::new(),
            section_fade: coverage::Coverage::default(),
            section_frontier_key: None,
            lod_clip_rings: 0,
            lod_clip_grow: Sticky::default(),
            lod_clip_shrunk: Sticky::raised(),
            section_epoch: 0,
            section_claim_seq: 0,
            section_pending_claim: None,
        };
        if pregenerate_origin {
            // Centre the pre-generated box on the origin's surface chunk, the
            // spawn point's own layer.
            let cy = world.generator.height(0, 0).div_euclid(CHUNK_SIZE as i32);
            world.ensure_region_data(ChunkCoord::new(0, cy, 0));
        }
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
        // A queued settled grid is a TRANSFERRED light claim: the in-flight
        // entry must be held until `settle_light` releases it, or `light_ready`
        // would admit a mesh against a grid that is about to change.
        // (`section_pending_claim` is deliberately NOT asserted `None` here: a
        // far-cap-rejected submit leaves it set until the next submit
        // overwrites it — a benign leftover, not a stranded claim.)
        for (coord, _) in &self.light_apply_queue {
            debug_assert!(
                self.light_inflight.contains(coord),
                "queued light grid for {coord:?} without its in-flight claim"
            );
        }
    }

    /// Draw the meshed chunks. All per-voxel work happened when each chunk was
    /// built; a frame is one `draw_mesh` per chunk (the engine frustum-culls
    /// each against its offset AABB internally).
    ///
    /// Terrain chunks and LOD sections are RESIDENT meshes: placement and detail
    /// pinned at upload, visibility a `set_visible` mask, style a `set_style`
    /// push (both maintained in `stream`). The engine draws every visible
    /// resident mesh itself, so `render` submits no per-mesh draws — it only sets
    /// the frame's LOD-cull volume and reports the set-size gauge.
    pub fn render(&self, f: &mut Frame3D, _cam: DVec3) {
        // The shader discards LOD-section fragments inside the SETTLED radius:
        // the rings whose chunks are actually drawn (or born-air). While a
        // loading edge is still meshing, the clip stays behind it and the far
        // sections keep covering the gap — coarse terrain instead of a hole —
        // then hands off ring by ring as uploads land. Fully settled, this is
        // exactly the old full-res radius.
        f.set_lod_clip(self.lod_clip());
        // Set-size gauge: a spike localizes a regression to a grown set (view
        // volume / section frontier). See `profile::Gauge`.
        use voxel_engine::profile::{gauge, Gauge};
        gauge(Gauge::WorldChunks, self.chunks.len() as u64);
    }

    /// The LOD-cull volume for this frame: the full-res slab shrunk to the
    /// settled rings. `rings` counts settled rings from the centre, so the
    /// nearest possibly-unsettled column sits at chess distance `rings`; its
    /// closest face is at least `(rings - 1) * 16` m from any eye position
    /// inside the centre chunk — the conservative discard radius. With every
    /// ring settled this is bit-identical to [`ViewVolume::coverage`].
    fn lod_clip(&self) -> CoverageVolume {
        let full = self.view.coverage();
        let radius_m = ((self.lod_clip_rings - 1).max(0) * CHUNK_SIZE as i32) as f32;
        CoverageVolume { radius: radius_m.min(full.radius), half_height: full.half_height }
    }

    /// Advance the settled-ring scan at a `&mut` sync point (end of `pump`
    /// and of `stream`). Self-gates on the two event flags: a converged,
    /// still frame is two flag checks. Growth re-scans only from the current
    /// frontier ring, so a loading wave costs each ring once, not per frame.
    pub(in crate::world) fn refresh_lod_clip(&mut self) {
        if self.lod_clip_shrunk.take() {
            self.lod_clip_rings = 0;
            self.lod_clip_grow.set();
        }
        if !self.lod_clip_grow.take() {
            return;
        }
        let Some(center) = self.center else { return };
        let max_rings = self.view.horizontal + 1;
        while self.lod_clip_rings < max_rings && self.ring_settled(center, self.lod_clip_rings) {
            self.lod_clip_rings += 1;
        }
    }

    /// Whether every column of the chess-distance `ring` around `center` is
    /// fully settled across the streamed vertical range.
    fn ring_settled(&self, center: Coord, ring: i32) -> bool {
        let v = self.view.vertical;
        let column = |cx: i32, cz: i32| {
            (center.y - v..=center.y + v)
                .all(|cy| self.chunks.get(&Coord::new(cx, cy, cz)).is_some_and(|l| l.state.settled()))
        };
        if ring == 0 {
            return column(center.x, center.z);
        }
        let r = ring;
        (-r..=r).all(|d| column(center.x + d, center.z - r) && column(center.x + d, center.z + r))
            && (1 - r..r).all(|d| column(center.x - r, center.z + d) && column(center.x + r, center.z + d))
    }

    /// Far-material style: flat palette-average past [`FLAT_DETAIL`] if available,
    /// else textured. Returns `mode` bit and packed sRGB colour.
    fn section_material(&self, pos: SectionPos) -> (bool, u32) {
        if pos.detail < FLAT_DETAIL {
            return (false, 0);
        }
        let color = self
            .section_overlay
            .get(&pos)
            .map(|c| c.color)
            .or_else(|| self.section_mip.as_ref().and_then(|m| m.color(pos)));
        match color {
            Some(c) => (
                true,
                c.r as u32 | (c.g as u32) << 8 | (c.b as u32) << 16 | (c.a as u32) << 24,
            ),
            None => (false, 0),
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
        // An edited footprint prefers the fresh overlay over the (possibly stale) bake.
        let (lo, hi) = if let Some(c) = self.section_overlay.get(&key) {
            (c.lo, c.hi)
        } else {
            let Some(mip) = &self.section_mip else { return false };
            let Some(band) = mip.relief_band(key) else { return false };
            band
        };
        let ey = self.section_eye_y as f32;
        if lo < ey - v_lim || hi > ey + v_lim {
            return false;
        }
        // Every chunk backing the footprint is settled, so the near area is
        // actually covered now, not just in-range.
        self.backing_chunks_ready(key)
    }

    /// Whether every full-res chunk backing `key`'s footprint is settled
    /// (drawable or born-air, never in-flight). Used by [`Self::coverage_skips`]
    /// to skip the near-LOD load only where full-res provably covers. A footprint
    /// whose vertical band can't be proven (no overlay, no baked mip) reads as
    /// NOT ready — fail toward keeping the section loaded, never toward a hole.
    fn backing_chunks_ready(&self, key: SectionPos) -> bool {
        let cs = CHUNK_SIZE as i32;
        let span = key.span();
        let (x0, z0) = (key.min_x(), key.min_z());
        let (lo, hi) = if let Some(c) = self.section_overlay.get(&key) {
            (c.lo, c.hi)
        } else {
            let Some(mip) = &self.section_mip else { return false };
            let Some(band) = mip.relief_band(key) else { return false };
            band
        };
        let (cx_lo, cx_hi) = (x0.div_euclid(cs), (x0 + span - 1).div_euclid(cs));
        let (cz_lo, cz_hi) = (z0.div_euclid(cs), (z0 + span - 1).div_euclid(cs));
        let (cy_lo, cy_hi) = ((lo.floor() as i32).div_euclid(cs), (hi.floor() as i32).div_euclid(cs));
        for cy in cy_lo..=cy_hi {
            for cz in cz_lo..=cz_hi {
                for cx in cx_lo..=cx_hi {
                    let settled =
                        self.chunks.get(&Coord::new(cx, cy, cz)).is_some_and(|l| l.state.settled());
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

    /// Wire the scheduler handles for the `stream` CPU lanes.
    /// Called once by `Game::new` after registering the producers.
    pub fn set_stream_lanes(&mut self, lanes: lanes::StreamLanes) {
        self.stream_lanes = Some(lanes);
    }

    /// The registered stream-lane handles (panics if `stream` runs before
    /// `Game::new` wired them — see [`World::stream_lanes`]).
    fn lanes(&self) -> lanes::StreamLanes {
        self.stream_lanes.expect("stream lanes registered by Game::new")
    }

    /// Rebuild the occlusion visible set: lazily fill any missing per-chunk
    /// connectivity (only floods chunks not yet classified — so a world that
    /// never activates occlusion never pays it), then BFS from the camera's
    /// chunk, then patch each drawable chunk's GPU visibility mask to its
    /// occlusion bit (`apply_occlusion_masks`). Nothing filters at draw time.
    ///
    /// The [`OcclusionLane`](lanes::OcclusionLane) producer's body: self-gates
    /// each call on the adaptive `occlusion_dirty`/`occlusion_active` state
    /// and reports `Progress::Partial` while the connectivity fill is budget-capped.
    pub(in crate::world) fn rebuild_occlusion(&mut self, eng: &mut Engine) -> Progress {
        // Recompute due-ness from current state every call; never cache it.
        let on = self.occlusion_enabled();
        let was_active = self.occlusion_active;
        let due = on && (self.occlusion_dirty.take() || !was_active);
        self.occlusion_active = on;
        if !on {
            // Gate off (or never on): restore every mask the last active pass may
            // have hidden, once, so render draws everything.
            if was_active {
                self.reveal_all(eng);
            }
            return Progress::Idle;
        }
        if !due {
            return Progress::Idle;
        }
        let Some(origin) = self.center else {
            return Progress::Idle;
        };
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
        self.apply_occlusion_masks(eng);
        if capped {
            let remaining = self.chunks.values().filter(|l| l.connectivity.is_none()).count();
            Progress::Partial { remaining: remaining as u32 }
        } else {
            Progress::Idle
        }
    }

    /// Patch every drawable chunk's GPU visibility mask to its occlusion bit —
    /// the recast of the BFS visible-set into `set_visible` (nothing filters at
    /// draw time). Runs only after a real rebuild (mask changed), so steady
    /// state pays nothing. Imperfect masking is only ever a wasted draw
    /// (occluded geometry is depth-culled), never a hole, so no cross-frame diff
    /// is kept: a freshly-remeshed chunk is re-hidden on the next rebuild.
    fn apply_occlusion_masks(&mut self, eng: &mut Engine) {
        for (&coord, loaded) in self.chunks.iter() {
            if let Some(meshes) = loaded.state.live_meshes() {
                meshes.set_visible(eng, self.occlusion.is_visible(coord));
            }
        }
    }

    /// Reveal every drawable chunk (set its mask visible) — the one-shot restore
    /// when the occlusion gate turns off, since only occlusion ever hides a
    /// resident chunk mesh.
    fn reveal_all(&mut self, eng: &mut Engine) {
        for loaded in self.chunks.values() {
            if let Some(meshes) = loaded.state.live_meshes() {
                meshes.set_visible(eng, true);
            }
        }
    }
}

// Streaming lanes: async admission producers.
// Every async streaming lane (fresh chunk meshing, far LOD sections,
// cross-chunk light, column generation) is the same shape: gather candidates
// near the player, drop the ones already in flight, order nearest-first, submit
// up to a per-frame budget to the worker pool, and integrate finished results.
// The lanes differ only in *where their state lives*, reached through a
// [`StreamLane`] of accessors so the admission loop ([`admit`]) is written once.
//
// Each lane is a `sched::Run` producer registered in [`lanes`]; the scheduler
// hands it its `Budget::Millis` and it derives its [`Deadline`](pipeline::Deadline)
// from that (never a private `pipeline::*_BUDGET` const — one budget locus, the
// manifest). The deadline is checked BETWEEN admitted items, and the forward-
// progress floor ([`admission_exhausted`]) is the single definition of "admit at
// least `MIN_ADMIT` before the clock can stop you", shared with the generation
// lane so no producer hand-rolls its own budget/floor.

/// The one admission-stop rule (forward-progress floor + time budget): stop only
/// once at least `min_admit` items have been admitted AND the deadline has
/// passed. The single definition every streaming producer's admit loop shares.
pub(in crate::world) fn admission_exhausted(
    admitted: usize,
    min_admit: usize,
    deadline: pipeline::Deadline,
) -> bool {
    admitted >= min_admit && deadline.expired()
}

/// How a lane defines its work this frame. A geometry lane derives candidates
/// from the player centre. A worklist lane reads an explicit dirty set
/// accumulated on the `World` (mesh, light: coords marked dirty by loads/edits).
pub(in crate::world) enum Candidates<K> {
    /// The full candidate key list, recomputed from the centre this frame.
    Geometry(Vec<K>),
    /// Read the lane's seed set (`StreamLane::seed_set`) for candidates.
    Worklist,
}

/// The accessor surface of one streaming lane. Zero-sized marker types
/// (`MeshLane`, …) implement it AND `sched::Run` (in [`lanes`]); all mutable
/// state lives on [`World`] behind these accessors, so the lane carries nothing
/// and the shared [`admit`] loop stays allocation-free. This is *not* a
/// scheduler: budget and forward-progress come from the scheduler that drives
/// the producer, never from here.
pub(in crate::world) trait StreamLane {
    /// The lane's work key: chunk `Coord` or `SectionPos`.
    /// Must be `Copy + Eq + Hash`.
    type Key: Copy + Eq + Hash;

    /// Minimum admissions before the deadline can stop the loop — the forward-
    /// progress floor, so setup cost (gather/sort) alone can't starve a lane
    /// under a tight budget. One value per lane; enforced once, in [`admit`].
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

/// The shared admission loop: gather, filter unready/in-flight, sort
/// nearest-first, submit until the deadline expires (checked between items,
/// never mid-item, and never before `MIN_ADMIT`), then claim. Clears the lane's
/// pending gate once the ready backlog drains. `deadline` comes from the
/// scheduler's per-producer budget — this loop owns no budget of its own.
pub(in crate::world) fn admit<S: StreamLane>(
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
    // `exhausted` stays true only if all ready keys were processed before the
    // deadline stopped the loop (the floor lives in `admission_exhausted`).
    let mut exhausted = true;
    let mut admitted = 0usize;
    for key in ready_keys {
        if admission_exhausted(admitted, S::MIN_ADMIT, deadline) {
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
impl StreamLane for MeshLane {
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
            Some(MeshState::NeedsMesh { building: true, .. })
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
        // Set the building flag IN PLACE to claim the mesh job — a whole-state
        // overwrite would silently drop a carried `prev` mesh (leaking its GPU
        // handle and blanking the chunk mid-rebuild). Held until upload retires
        // it or a stale result releases it. Remove from worklist.
        world.mesh_worklist.remove(&key);
        if let Some(loaded) = world.chunks.get_mut(&key) {
            debug_assert!(loaded.state.is_needs_mesh(), "mesh submit for non-NeedsMesh {key:?}");
            if let MeshState::NeedsMesh { building, .. } = &mut loaded.state {
                *building = true;
            }
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
impl StreamLane for SectionLane {
    type Key = SectionPos;
    /// Modest floor: heavy work runs off-thread.
    const MIN_ADMIT: usize = 4;
    fn candidates(world: &World, center: Coord) -> Candidates<SectionPos> {
        // Start with the frame's cached desired set, filter those already
        // loaded and those provably inside the coverage clip (skip load).
        Candidates::Geometry(
            world
                .section_desired
                .iter()
                .copied()
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
        // Mint the claim token here; `claim` (which always follows an accepted
        // submit for the same key) installs it on the `Meshing` entry.
        world.section_claim_seq = world.section_claim_seq.wrapping_add(1);
        let token = pipeline::ClaimToken(world.section_claim_seq);
        world.section_pending_claim = Some((key, token));
        Some(pipeline::Job::Section {
            pos: key,
            epoch: world.section_epoch,
            token,
            generator: world.generator.clone(),
            edits: world.edits_for_section(key),
            tables: world.tables.get(),
        })
    }
    fn claim(world: &mut World, key: SectionPos) {
        let token = match world.section_pending_claim.take() {
            Some((pos, token)) if pos == key => token,
            other => {
                debug_assert!(false, "section claim for {key:?} without its submit ({other:?})");
                pipeline::ClaimToken(world.section_claim_seq)
            }
        };
        world.dirty_sections.remove(&key);
        world.sections.insert(key, SectionState::Meshing { token });
    }
    fn integrate(world: &mut World, done: pipeline::Done) {
        if let pipeline::Done::Section { pos, epoch, token, meshes } = done {
            // A result from a retired ladder epoch, or for a claim replaced
            // after unload/re-admission, is dropped here — it must not queue an
            // upload that would capture a same-position replacement.
            let live = epoch == world.section_epoch
                && matches!(world.sections.get(&pos),
                    Some(SectionState::Meshing { token: t }) if *t == token);
            if live {
                world.section_upload_queue.push_back((pos, token, meshes));
            }
        }
    }
}

/// Cross-chunk light settling.
pub(in crate::world) struct LightLane;
impl StreamLane for LightLane {
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
        // `accept_light` owns the claim rule (release-or-transfer on every
        // consumed result) — see its doc for the epoch soundness argument.
        if let pipeline::Done::Light { coord, epoch, grid } = done {
            world.accept_light(coord, epoch, grid);
        }
    }
}

impl Render for World {
    fn render(&self, f: &mut Frame3D, cam: DVec3) {
        World::render(self, f, cam);
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
