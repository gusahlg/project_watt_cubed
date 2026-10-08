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
//! * `mod.rs` (this file) — constants, the fast hash maps, the `World` struct
//!   itself, and construction.
//! * `resident.rs` — [`Loaded`] chunks, their mesh lifecycle and GPU meshes,
//!   sections, and slot counts.
//! * `streaming.rs` (and `streaming/`) — the per-frame [`stream`](World::stream)
//!   pass: pacing, generation, claims, worker result draining, budgeted
//!   uploads, unloading, light, meshing, the far field, and gauges.
//! * `admit.rs` — the admission loop and the mesh, section and light lanes.
//! * `clip.rs` — rendering's LOD clip, settled rings and coverage proofs.
//! * `occlusion.rs` — the occlusion gate and visible-set rebuild.
//! * `textures.rs` — block texture layers.
//! * `query.rs` — read-only queries: block lookup, solidity, collision,
//!   surface height, coordinate mapping, registry/seed accessors.
//! * `settings.rs` — view distances, the render config, lighting and AO.
//! * `edits.rs` — player edits: block placement, the edit overlay, and dirty
//!   marking.
pub mod brick;
pub mod chunk;
pub mod connectivity;
pub mod generation;
mod layout;
pub use layout::{ColumnKey, Sky};
pub mod light;
pub mod lod;
pub mod mesh;
mod neighborhood;
pub mod pipeline;
pub mod pyramid;
pub mod section;
pub mod terrain;

mod admit;
mod census;
mod clip;
mod coverage;
mod edits;
#[cfg(test)]
mod fixtures;
mod heightmip;
mod lanes;
mod metric;
mod occlusion;
mod quadtree;
mod query;
mod resident;
pub(crate) mod seam;
mod settings;
mod streaming;
mod summary;
mod textures;
mod worklist;

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::sync::Arc;
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use voxel_engine::producer::{Budget, Progress};
use voxel_engine::{CoverageVolume, DVec3, Engine, FadeStyle, Frame3D, MeshHandle, Vec3};
use crate::ident::Detail;

use crate::block::registry::{BlockId, BlockRegistry, HotTables};
use crate::coord::{ByPass, ChunkBox, ChunkCoord, Face};
use crate::space::FaceFrame;
use chunk::{CHUNK_SIZE, Chunk};
use generation::WorldgenKind;
use heightmip::HeightMip;
use light::LightGrid;
use mesh::{ChunkMeshData, new_chunk_mesh_data};
use quadtree::QuadrantMask;
use section::{SectionMeshData, SectionPos};

/// Engine CPU-cull live-count threshold (`mesh_stats().cpu_cull_max` fallback).
/// Above this the engine uses its GPU cull dispatch (~10 µs); it is a cost
/// knob, not a hard slot limit. The far lane budgets section slots against
/// `max(SECTION_SLOT_FLOOR, cpu_cull_max - near_chunk_slots)`.
const CPU_CULL_MAX: u32 = 1024;
/// Floor on the far lane's section-slot budget so a large near field cannot
/// starve covering: chunks never count against sections.
pub(in crate::world) const SECTION_SLOT_FLOOR: usize = 512;

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
/// Per-frame BYTE budget for chunk-mesh GPU uploads — the upload is the only
/// part of the async path the render thread still pays, and its real cost is
/// bytes staged, not mesh count (the old 4-mesh budget charged a tiny stale
/// drop the same as a full surface chunk, so a post-flight queue of stale
/// entries starved real uploads for dozens of frames). Half the engine's
/// 8 MiB/frame transfer window, leaving room for section uploads that share it.
const UPLOAD_BUDGET_BYTES: usize = 4 << 20;
/// Hard cap on upload-queue pops per drain, so even a mountain of
/// free-to-drop stale entries takes bounded time.
const UPLOAD_SCAN_MAX: usize = 256;
/// Upload-queue depth past which MESH ADMISSION pauses: the workers are
/// outproducing the byte budget, so more jobs would only grow the backlog
/// (and its memory — each entry owns a pooled mesh buffer). The run-site gate
/// leaves `pending_fresh` raised and the worklist intact, so admission
/// self-resumes as the queue drains. Also the backpressure that bounds the
/// pool's unbounded result channel under the scaled worker count.
pub(in crate::world) const UPLOAD_QUEUE_MAX: usize = 96;
/// How many *dirty* (edited) chunks may remesh per frame. Processed nearest
/// first, so a locally broken block still vanishes the same frame while a
/// multiplayer join snapshot flood spreads over a few frames instead of one hitch.
const DIRTY_BUDGET: usize = 8;
/// Floor on connectivity fills per occlusion pass so a tight time budget still
/// makes strict progress during a load flood. The lane itself is time-budgeted
/// (`Budget::Millis(0.5)`); a partial fill no longer forces an immediate BFS.
const OCCLUSION_FILL_FLOOR: usize = 16;
/// Minimum spacing between TOPOLOGY-triggered occlusion rebuilds (chunk
/// loads/unloads). Their staleness is over-draw only — a not-yet-hidden fresh
/// chunk — never a hole, so a load flood no longer pays a full BFS + mask
/// walk every stream pass. Root moves and edits rebuild immediately (a stale
/// BFS root or stale connectivity CAN hide a visible chunk).
const OCCLUSION_DEBOUNCE: Duration = Duration::from_millis(100);
/// The seed a default (`generate`) world uses when none is chosen.
pub const DEFAULT_SEED: i64 = 1;
/// Base column-section uploads per frame. Unlike the old queue-depth scaler,
/// this is a hard ceiling: a backlog is evidence that producers should slow
/// down, not permission to make the render thread do progressively more work.
const SECTION_UPLOAD_BUDGET: usize = 2;

/// A chunk-coordinate map key. The [`ChunkCoord`] newtype owns the
/// `chunk * CHUNK_SIZE + local` relationship (see [`crate::coord`]); the
/// `Coord` alias keeps the shorter name the `world` module already used.
use crate::coord::ChunkCoord as Coord;

/// `seed_light` insert attempts by source. Degrade / terminal / neighbour-remesh
/// stay zero unless those paths start seeding the light worklist.
#[derive(Clone, Copy, Debug, Default)]
pub struct LightSeedSplit {
    pub store: u64,
    pub border: u64,
    pub edit: u64,
    pub degrade: u64,
    pub terminal: u64,
    pub remesh: u64,
}

/// Why a coord was inserted into `light_worklist`.
#[derive(Clone, Copy)]
#[allow(dead_code)] // Degrade/Terminal/Remesh are report buckets; tests construct them.
pub(in crate::world) enum LightSeed {
    Store,
    Border,
    Edit,
    Degrade,
    Terminal,
    Remesh,
}

impl LightSeedSplit {
    pub(in crate::world) fn add(&mut self, kind: LightSeed) {
        *match kind {
            LightSeed::Store => &mut self.store,
            LightSeed::Border => &mut self.border,
            LightSeed::Edit => &mut self.edit,
            LightSeed::Degrade => &mut self.degrade,
            LightSeed::Terminal => &mut self.terminal,
            LightSeed::Remesh => &mut self.remesh,
        } += 1;
    }
}

pub use census::MemoryCensus;
pub use streaming::StreamGauges;
use admit::{
    AdmitScratch, LightLane, MeshLane, SectionLane, StreamLane, admission_exhausted, admit, bias_order,
    motion_biased_dist2,
};
#[cfg(test)]
use admit::player_dist2;
use resident::{ChartBend, Loaded, MeshState, SectionState, adjust_count};
#[cfg(test)]
use resident::{ChunkMeshes, mesh_free_log, vis_log};
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
        Self {
            horizontal,
            vertical,
        }
    }
    /// The streamed volume for a horizontal view radius, with the vertical
    /// radius derived from it.
    fn view(horizontal: i32) -> Self {
        Self::new(horizontal, Self::vertical_for(horizontal))
    }
    /// The full-res coverage box in metres (horizontal half `rh` chunks, vertical
    /// `rv`). The shader's LOD-cull volume must equal this streamed volume.
    fn coverage(&self) -> CoverageVolume {
        let r = (self.horizontal * CHUNK_SIZE as i32) as f32;
        let rv = (self.vertical * CHUNK_SIZE as i32) as f32;
        CoverageVolume {
            half: Vec3::new(r, rv, r),
        }
    }

    /// The far-field LOD ring unit in metres. Cannot be zero even when the
    /// stripped near field draws only the current chunk column.
    pub(in crate::world) fn lod_unit(&self) -> f32 {
        (self.horizontal.max(1) * CHUNK_SIZE as i32) as f32
    }
    /// The mesh box grown by `dh` across `up` and `dv` along it.
    /// `None` ignores `dv`: every axis uses the horizontal radius plus `dh`.
    fn box_at(self, center: Coord, dh: i32, dv: i32, up: Option<Face>) -> ChunkBox {
        match up {
            None => ChunkBox::with_up(center, self.horizontal + dh, 0, None),
            Some(face) => {
                ChunkBox::with_up(center, self.horizontal + dh, self.vertical + dv, Some(face))
            }
        }
    }
    /// Chunks meshed and drawn around `center`.
    fn mesh(self, center: Coord, up: Option<Face>) -> ChunkBox {
        self.box_at(center, 0, 0, up)
    }
    /// The mesh box plus one [`DATA_MARGIN`] shell of voxel data, so edge chunks
    /// cull against neighbours that are loaded but unmeshed.
    fn data(self, center: Coord, up: Option<Face>) -> ChunkBox {
        self.box_at(center, DATA_MARGIN, DATA_MARGIN, up)
    }
    /// The mesh box plus the unload hysteresis, past which chunks free. Vertical
    /// uses the tighter [`UNLOAD_MARGIN_V`] to match the tighter vertical radius.
    /// `None` grows every axis by [`UNLOAD_MARGIN`].
    fn unload(self, center: Coord, up: Option<Face>) -> ChunkBox {
        self.box_at(center, UNLOAD_MARGIN, UNLOAD_MARGIN_V, up)
    }

    /// Ring-worklist bucket count: one past the data box's heaviest
    /// [`World::order`], the box stretched `stretch` layers along the up axis.
    /// `None` is plain 3-D chess, so the count is the horizontal data radius
    /// plus one. Keys past the last ring clamp there.
    fn worklist_rings(self, up: Option<Face>, stretch: i32) -> usize {
        let rh = self.horizontal + DATA_MARGIN;
        let rv = self.vertical + stretch + DATA_MARGIN;
        let span = match up {
            None => rh,
            Some(_) => rh.max(2 * rv),
        };
        (span as usize).saturating_add(1)
    }
}

/// Fast multiply-based hasher for well-distributed grid coordinate keys (chunk hot path).
#[derive(Default)]
pub(crate) struct FastHasher(u64);

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

pub(crate) type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;
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

/// The streamed world: the block palette, the terrain generator, the currently
/// loaded chunks, and the overlay of player edits that outlive chunk unloads.
pub struct World {
    /// Read-only block palette; meshing/collision read its hot solidity arrays.
    registry: BlockRegistry,
    generator: terrain::Generator,
    /// Gravity from the generator's matter plus every committed edit.
    gravity: crate::gravity::Field,
    /// The seams of the generator's round bodies (curved-chart storage boxes).
    seams: seam::Seams,
    /// The charts around the streaming centre unfolded into one net (identity off round worlds).
    fold: seam::Unfold,
    /// The engine cage each loaded storage chunk's meshes are drawn through (freed on unload).
    cages: FastMap<Coord, voxel_engine::CageHandle>,
    kind: WorldgenKind,
    terrain_cfg: terrain::TerrainCfg,
    chunks: FastMap<Coord, Loaded>,
    /// Player edits grouped by chunk (inner key: flat voxel index for replay on regenerate).
    edits: FastMap<Coord, FastMap<usize, BlockId>>,
    /// Monotone counter bumped on every recorded edit; the autosaver compares
    /// it against the last written generation to know the world is dirty.
    pub(crate) edit_generation: u64,
    /// Last chunk centre; `None` forces a full stream pass. Streams only react to boundary crosses.
    center: Option<Coord>,
    /// The near window's span along the up axis around [`center`](Self::center): the eye band,
    /// grown over the near square's terrain.
    window: streaming::Window,
    /// Surface bounds of the near square's chunk columns, kept while they stay in the square.
    window_ground: streaming::NearBounds,
    /// The unload box of a window frame just left (another up face), in its chart net: its chunks
    /// stay loaded until the far field draws the new window.
    retired: Option<(ChunkBox, seam::Unfold)>,
    /// Chunk-y extent of the edits in each chunk column `(x, z)`, never shrunk. The generator's
    /// surface bounds miss a pit dug below them or a tower built above.
    edit_columns: FastMap<(i32, i32), [i32; 2]>,
    /// The far field's chunk centre: the chart column under the eye, which outlasts the near
    /// window's chart reach; the streaming centre elsewhere. Set by [`stream`](Self::stream).
    far_center: Option<Coord>,
    /// The chart net around [`far_center`](Self::far_center), which the job gate measures far work
    /// in; the identity off charts.
    far_fold: seam::Unfold,
    /// Atlas whose chart the far field stood on last pass (its reach gets the hold margin).
    far_atlas: Option<usize>,
    /// Doublings of the chart ring unit for the eye's height above the ground.
    far_scale: u8,
    /// The far eye's storage column `(x, z)` and its generated ground.
    far_ground: Option<((i32, i32), i32)>,
    /// Up face committed for the streaming centre. `None` is isotropic.
    /// Meaningful only once [`stream_up_set`](Self::stream_up_set) is true.
    stream_up: Option<Face>,
    /// False until the first centre resolve. Until then [`live_up`](Self::live_up)
    /// is +Y, so pre-stream orders match the historical volume.
    stream_up_set: bool,
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
    /// Reused by Coord admission lanes (mesh, light) so [`admit`] never allocates.
    admit_coords: AdmitScratch<Coord>,
    /// Reused by the section admission lane.
    admit_sections: AdmitScratch<SectionPos>,
    /// Generate runs for the current data box, and what they were gathered for.
    gen_cursor: streaming::GenCursor,
    /// Radii new generation, light and mesh may cover. `-1` until the first
    /// stream, which means the full view. Already-loaded chunks outside this
    /// stay until the unload box frees them.
    load_h: i32,
    load_v: i32,
    load_moved: bool,
    heading_changed: bool,
    load_heading: i8,
    /// Loaded chunks whose light seed or claim the reduced loading window
    /// dropped. They re-seed once the window covers them again; unload forgets
    /// them.
    light_owed: FastSet<Coord>,
    /// Coords with generate jobs in flight. Blocks re-enqueue; cleared on drain.
    generating: FastSet<Coord>,
    /// The flight bench's and the stress gauges' counters.
    counters: streaming::StreamCounters,
    /// `NeedsMesh { building: true }` claims. Counter so idle `pump` never scans chunks.
    building_meshes: usize,
    /// `SectionState::Meshing` claims. Counter so idle `pump` never scans sections.
    meshing_sections: usize,
    /// The [`GenerateLane`](lanes::GenerateLane)'s raise-then-consume gate: the
    /// data box has columns to request. Raised on a boundary cross and by a
    /// generate strike-out re-request; drained when the box is fully requested.
    pending_gen: Sticky,
    /// Outstanding spawn/teleport collision slab. `None` once every chunk has
    /// loaded (or never requested). Physics waits on [`spawn_ready`](Self::spawn_ready).
    spawn_slab: Option<ChunkBox>,
    /// Finished meshes awaiting budgeted upload (re-validated at upload time for staleness).
    upload_queue: VecDeque<(Coord, u32, pipeline::MeshPayload)>,
    /// Chunks needing a *fresh* mesh (the [`MeshLane`] seed set — replaces the
    /// old whole-map rescan `pending_fresh` armed). Seeded on load (self + 6
    /// neighbours), on a light publish that moved a border, and on an
    /// accept_mesh stale drop. Drained nearest-first by the mesh lane, so the
    /// enqueue scan is O(shell) not O(cube).
    mesh_worklist: worklist::RingWorklist,
    /// Whether the [`LightLane`] still has seeds/in-flight to drain (its
    /// `pending` gate — the [`StreamLane::pending`] accessor). Raised when the
    /// worklist or in-flight set is non-empty, cleared when both drain.
    light_pending: Sticky,
    /// Chunks needing light settling (budgeted, seeded on load/edit/border moves).
    light_worklist: worklist::RingWorklist,
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
    /// Remesh/stale-drop samples for the stress C3 gauges.
    remesh_stats: streaming::RemeshStats,
    /// Chunks whose missing neighbour light will never arrive, so a mesh
    /// snapshot must read missing planes as settled dark (not open-sky).
    light_terminal: FastSet<Coord>,
    /// Degraded-snapshot flag carried from mesh submit to claim, so a rejected
    /// submit does not mutate the degraded or terminal sets.
    mesh_pending_degraded: Option<(Coord, bool)>,
    /// Skylight ceiling per [`ColumnKey`] — the surface heightmap the settle
    /// pass seeds skylight from. A pure generator function (independent of
    /// altitude and of edits), so it is computed once per column and reused
    /// across every chunk of the run and every re-settle. Pruned when a column
    /// fully unloads. `Open` chunks are not entered.
    ceilings: FastMap<ColumnKey, Arc<light::CeilingWindow>>,
    /// Panic counts per failed claim, for the bounded-retry policy in
    /// [`fail_job`](World::fail_job). Rare by construction (a strike is a
    /// worker panic), so the map stays tiny.
    job_strikes: FastMap<streaming::FailKey, u8>,
    /// Claims that kept panicking: permanently parked so one poison input is a
    /// bounded hole in the world, not an infinite resubmit-panic loop. Every
    /// scan that would re-request the work consults this set.
    quarantined: FastSet<streaming::FailKey>,
    /// The block texture array as last built and sent to the GPU.
    textures: textures::BlockTextures,
    /// Baked corner AO in the mesher — stamped into `HotTables::ao`. A meshing
    /// input like `lighting`: toggling remeshes the world.
    ao: bool,
    /// Bumped whenever a non-registry meshing input stamped onto the hot
    /// tables changes (AO today), so `refresh_tables` rebuilds even though the
    /// block count didn't move — the count and epoch fold into one revision.
    tables_epoch: u32,
    /// Occlusion visible set (rebuilt at stream sync point, read by render).
    occlusion: Occlusion,
    /// Occlusion needs an IMMEDIATE rebuild: the BFS root moved or an edit
    /// changed connectivity — staleness of either can hide a visible chunk (a
    /// hole), so these never debounce.
    occlusion_dirty: Sticky,
    /// Occlusion needs a TOPOLOGY rebuild (a chunk loaded/unloaded): staleness
    /// is over-draw only, so these debounce at [`OCCLUSION_DEBOUNCE`] — a load
    /// flood pays one rebuild per window instead of one per pass.
    occlusion_topo_dirty: Sticky,
    /// The last occlusion rebuild's instant — the debounce clock.
    last_occlusion_rebuild: Option<Instant>,
    /// Chunks needing a connectivity fill for the occlusion BFS — fed by
    /// loads and connectivity-invalidating edits (only while the gate is on;
    /// activation reseeds from scratch), drained up to
    /// [`OCCLUSION_FILL_FLOOR`] per pass. Replaces the per-rebuild
    /// all-chunks missing-connectivity scan; unloaded entries drop lazily at pop.
    conn_fill_queue: VecDeque<Coord>,
    /// Membership set of the SYNC `Dirty` fiber, maintained by
    /// [`invalidate_mesh`](World::invalidate_mesh) and pruned on
    /// unload/free/drain — the dirty pass drains this set instead of
    /// filtering every loaded chunk each frame `pending_dirty` is up.
    dirty_worklist: FastSet<Coord>,
    /// The mesh box of the previous full pass: a boundary cross probes only
    /// the entered shell (new ∖ old) for NeedsMesh re-seeding. `None` after a
    /// radius change, so the next cross probes the whole box.
    prev_mesh_box: Option<ChunkBox>,
    /// The unload box of the previous full pass: a boundary cross frees only
    /// the left shell (old ∖ new). `None` after a radius change, so the next
    /// cross scans every loaded chunk.
    prev_unload_box: Option<ChunkBox>,
    /// Settled chunks across a seam kept past the unload box until they leave the turn-back
    /// skirt. The shell diff would not see them on a later pass.
    far_wait: FastSet<Coord>,
    /// Loaded altitude-chunk indices per [`ColumnKey`], highest first. Empty
    /// vec is pruned so a column's cached ceiling drops exactly when its last
    /// chunk unloads. PosY altitudes are chunk Y. `Open` chunks are not entered.
    column_chunks: FastMap<ColumnKey, Vec<i32>>,
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
    /// Per-`Loaded` light-claim identity (see [`Loaded::light_gen`]).
    light_claim_seq: u32,
    /// LOD2 column-section far field enable.
    lod2: bool,
    /// LOD pyramid config with `unit` in metres.
    section_pyramid: pyramid::PyramidCfg,
    /// The far field's eye altitude captured each `stream()` before chunk-coord floor rounds it.
    /// Feeds the vertical LOD selection. XZ selection uses chunk centre only.
    section_eye_y: f64,
    /// Body and face the far field is selecting, with hysteresis at edges.
    /// `None` once a pass has decided the camera is over no cube (space, storage, a round body).
    section_lod_face: Option<(u16, Face)>,
    /// Set by [`stream`](Self::stream). Until then selection uses the dominant face
    /// (PosY at the origin), so tests that never stream keep today's frontier.
    section_face_set: bool,
    /// The bake's `(body, face, anchor_u, anchor_v)`. `None` until a bake is spawned;
    /// readers then do not filter summaries by face.
    section_mip_anchor: Option<(u16, Face, i32, i32)>,
    /// Previous far-field eye + timestamp for velocity computation.
    /// Reset to None on teleport or first stream.
    section_eye_prev: Option<(DVec3, Instant)>,
    /// Previous near-window eye + timestamp: the pacer's travel sample.
    near_eye_prev: Option<(DVec3, Instant)>,
    /// Eye velocity (m/s) from successive stream centres.
    /// Zero at rest or after teleport.
    section_vel: DVec3,
    /// Velocity-aware load controller. Fast travel shrinks the loading window
    /// and lowers admission, result-integration, and upload pressure; both
    /// recover gradually after stopping so the first stationary frame cannot hitch.
    stream_pacer: streaming::StreamPacer,
    /// Wall time of the previous [`World::stream`] topology pass. The rest-time
    /// worker boost uses it as the frame-headroom signal.
    last_stream_secs: f64,
    /// Per-cell relief drives error-driven LOD selection for the far field.
    /// `None` until the background bake lands; selection falls back to default LOD.
    section_mip: Option<HeightMip>,
    /// Receiver for the in-flight background bake, taken once it lands. `None` before
    /// the bake is spawned and after it is installed.
    section_mip_rx: Option<Receiver<HeightMip>>,
    /// Loaded sections.
    sections: FastMap<SectionPos, SectionState>,
    /// Ready finer tiles under a desired cell nothing Ready draws yet. Counted at the
    /// start of a section admission that has already filled [`sections_allowed`](Self::sections_allowed).
    /// Those tiles hold slots the holes still need, so the lane may pass the floor by this many.
    section_standin_slack: usize,
    /// Finished section meshes awaiting budgeted upload, tagged with the claim
    /// token that produced them (re-validated at the moment of upload) and the
    /// vertex-byte charge computed at queue time.
    section_upload_queue:
        VecDeque<(SectionPos, pipeline::ClaimToken, usize, pipeline::SectionPayload)>,
    /// Last engine `mesh_stats().live_slots` sampled at `stream`. Zero until
    /// the first stream (tests without a GPU).
    gpu_live_slots: u32,
    /// CPU-cull live-count ceiling (`mesh_stats().cpu_cull_max`, else 1024).
    /// Cost knob for the engine's GPU cull dispatch; the far lane's section
    /// budget is [`sections_allowed`](Self::sections_allowed), not this raw value.
    slot_ceiling: u32,
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
    /// The chunk range along the up axis the settled rings are proven over: the
    /// near window's, or `None` for the centre's eye band.
    lod_clip_span: Option<[i32; 2]>,
    /// A grown window span still being proven, with its settled rings. The clip
    /// keeps to the proven span until these catch up.
    lod_clip_next: Option<([i32; 2], i32)>,
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
    /// Surface bounds the near-window punch read, kept across frontier recomputes.
    near_bounds: streaming::NearBounds,
    /// Desired sections the near window holds whose ground has not all settled, with the chunk
    /// layers that ground spans. They keep drawing until those chunks do. Admission skips one
    /// only once a Ready section already draws it.
    section_held: FastMap<SectionPos, [i32; 2]>,
    /// A chunk settled since the held sections were last checked.
    held_recheck: Sticky,
    /// Far-lane configuration epoch: bumped by every live ladder change so an
    /// in-flight worker result from a retired configuration can never land.
    section_epoch: u32,
    /// Monotone claim-token source for section jobs (see
    /// [`pipeline::ClaimToken`]); `section_pending_claim` carries the
    /// freshly minted token from the lane's `submit` to its `claim`.
    section_claim_seq: u64,
    section_pending_claim: Option<(SectionPos, pipeline::ClaimToken)>,
    /// Gameplay reaction events. Ticked by the sim `reactions` system when this
    /// instance is the authority (single-player or the dedicated server).
    reactions: crate::sim::reactions::ReactionScheduler,
    /// Single-player (and a hosting server) run the scheduler; a client connected
    /// to a server does not.
    reactions_authority: bool,
}

/// The exact inputs the desired-section frontier depends on, as cheap bit
/// patterns. Equality means the cached frontier is still the right answer.
/// Edits are handled separately: a dirty section forces a recompute (see
/// `World::stream`) because relief coarsening consults the edit overlay.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct SectionFrontierKey {
    center_xz: [i32; 2],
    center_y: i32,
    body: u16,
    face: u8,
    /// The altitude the selection reads: the exact eye on a cube face, the block on a chart.
    eye_y: u64,
    /// The prediction the selection reads (see `World::refresh_frontier`).
    velocity: [u64; 3],
    /// The full-res window's height and up face; a chart punches sections the window holds.
    vertical: i32,
    up: Option<Face>,
    /// The span a chart's punch tests against (`None` off charts).
    window: Option<[i32; 2]>,
    unit: u32,
    finest: i8,
    levels: u8,
    step: u8,
    mip_ready: bool,
    /// Far-lane section-slot budget the selection coarsens to.
    allowed: u32,
}

impl World {
    /// A fresh FLAT world for `seed` (the core fallback generator: ground at
    /// [`generation::FLAT_HEIGHT`]), with the region around the origin pre-generated
    /// (data only — no GPU) so headless queries work before the first
    /// [`stream`](Self::stream). The plain world tests and tools build on; the game builds
    /// its worlds with [`with_kind_cfg`](Self::with_kind_cfg).
    pub fn new(seed: i64) -> Self {
        Self::with_config(seed, crate::render_config::RenderConfig::default())
    }

    /// A fresh world for `seed` with explicit render config — the sole source
    /// for lod2/occlusion gates. [`new`](Self::new) is this with defaults.
    ///
    /// [`RenderConfig`]: crate::render_config::RenderConfig
    pub fn with_config(seed: i64, render: crate::render_config::RenderConfig) -> Self {
        Self::with_kind(seed, render, WorldgenKind::Flat, true)
    }

    /// Construct without synchronously generating the full origin data box.
    /// Interactive startup uses this path and calls
    /// [`prepare_around`](Self::prepare_around) to *request* the collision-safe
    /// spawn slab (worker-pool columns; [`spawn_ready`](Self::spawn_ready)
    /// becomes true once they land). Streaming fills the remainder.
    /// Existing constructors retain eager data for tests and headless callers
    /// that query the origin before their first stream.
    pub fn with_config_lazy(seed: i64, render: crate::render_config::RenderConfig) -> Self {
        Self::with_kind(seed, render, WorldgenKind::Flat, false)
    }

    /// Construct with an explicit worldgen kind (flat or InfiniteDiffusion) and default knobs.
    pub fn with_kind(
        seed: i64,
        render: crate::render_config::RenderConfig,
        kind: WorldgenKind,
        pregenerate_origin: bool,
    ) -> Self {
        Self::with_kind_cfg(seed, render, kind, terrain::TerrainCfg::default(), pregenerate_origin)
    }

    pub fn with_kind_cfg(
        seed: i64,
        render: crate::render_config::RenderConfig,
        kind: WorldgenKind,
        cfg: terrain::TerrainCfg,
        pregenerate_origin: bool,
    ) -> Self {
        let cfg = cfg.clamp();
        let mut registry = BlockRegistry::with_builtins();
        let generator: terrain::Generator = match kind {
            WorldgenKind::Flat => Arc::new(generation::FlatTerrain::new(&mut registry, seed)),
            WorldgenKind::Diffusion => terrain::generator(&mut registry, seed, cfg),
        };
        // The section ladder's innermost ring begins where the full-res box ends,
        // so its `unit` is the render distance in metres.
        let unit = (DEFAULT_VIEW_RADIUS * CHUNK_SIZE as i32) as f32;
        let lod2 = render.lod2;
        let (lod_levels, lod_detail) = render.normalized_lod();
        let gravity = crate::gravity::Field::new(generator.mass());
        let seams = seam::Seams::new(generator.atlases().to_vec());
        let mut world = Self {
            registry,
            gravity,
            seams,
            fold: seam::Unfold::IDENTITY,
            cages: FastMap::default(),
            generator,
            kind,
            terrain_cfg: cfg,
            chunks: FastMap::default(),
            ceilings: FastMap::default(),
            edits: FastMap::default(),
            edit_generation: 0,
            center: None,
            window: streaming::Window::default(),
            window_ground: streaming::NearBounds::default(),
            retired: None,
            edit_columns: FastMap::default(),
            far_center: None,
            far_fold: seam::Unfold::IDENTITY,
            far_atlas: None,
            far_scale: 0,
            far_ground: None,
            stream_up: None,
            stream_up_set: false,
            view: ViewVolume::view(DEFAULT_VIEW_RADIUS),
            pending_fresh: Sticky::raised(),
            pending_dirty: Sticky::default(),
            radius_shrunk: Sticky::default(),
            scratch: new_chunk_mesh_data(),
            tables: crate::derived::Derived::default(),
            workers: None,
            admit_coords: AdmitScratch::default(),
            admit_sections: AdmitScratch::default(),
            gen_cursor: streaming::GenCursor::default(),
            load_h: -1,
            load_v: -1,
            load_moved: false,
            heading_changed: false,
            load_heading: 0,
            light_owed: FastSet::default(),
            generating: FastSet::default(),
            counters: streaming::StreamCounters::default(),
            building_meshes: 0,
            meshing_sections: 0,
            pending_gen: Sticky::default(),
            spawn_slab: None,
            upload_queue: VecDeque::new(),
            mesh_worklist: worklist::RingWorklist::new(
                Coord::new(0, 0, 0),
                ViewVolume::view(DEFAULT_VIEW_RADIUS).worklist_rings(Some(Face::PosY), 0),
            ),
            light_pending: Sticky::default(),
            light_worklist: worklist::RingWorklist::new(
                Coord::new(0, 0, 0),
                ViewVolume::view(DEFAULT_VIEW_RADIUS).worklist_rings(Some(Face::PosY), 0),
            ),
            light_inflight: FastSet::default(),
            light_apply_queue: VecDeque::new(),
            light_gate: streaming::LightGate::default(),
            remesh_stats: streaming::RemeshStats::default(),
            light_terminal: FastSet::default(),
            mesh_pending_degraded: None,
            job_strikes: FastMap::default(),
            quarantined: FastSet::default(),
            textures: textures::BlockTextures::new(),
            ao: true,
            tables_epoch: 0,
            occlusion: Occlusion::default(),
            occlusion_dirty: Sticky::default(),
            occlusion_topo_dirty: Sticky::default(),
            last_occlusion_rebuild: None,
            conn_fill_queue: VecDeque::new(),
            dirty_worklist: FastSet::default(),
            prev_mesh_box: None,
            prev_unload_box: None,
            far_wait: FastSet::default(),
            column_chunks: FastMap::default(),
            occlusion_active: false,
            occlusion_forced: render.occlusion,
            lighting: true,
            light_epoch: 0,
            light_claim_seq: 0,
            lod2,
            section_pyramid: pyramid::PyramidCfg::sections_with(unit, lod_levels, lod_detail),
            section_eye_y: 0.0,
            section_lod_face: None,
            section_face_set: false,
            section_mip_anchor: None,
            section_eye_prev: None,
            near_eye_prev: None,
            section_vel: DVec3::ZERO,
            stream_pacer: streaming::StreamPacer::default(),
            last_stream_secs: 0.0,
            section_mip: None,
            section_mip_rx: None,
            sections: FastMap::default(),
            section_standin_slack: 0,
            section_upload_queue: VecDeque::new(),
            gpu_live_slots: 0,
            slot_ceiling: CPU_CULL_MAX,
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
            near_bounds: streaming::NearBounds::default(),
            section_held: FastMap::default(),
            held_recheck: Sticky::default(),
            lod_clip_rings: 0,
            lod_clip_span: None,
            lod_clip_next: None,
            lod_clip_grow: Sticky::default(),
            lod_clip_shrunk: Sticky::raised(),
            section_epoch: 0,
            section_claim_seq: 0,
            section_pending_claim: None,
            reactions: crate::sim::reactions::ReactionScheduler::new(),
            reactions_authority: true,
        };
        if pregenerate_origin {
            // The charted start world spawns on its +Y chart; a flat world keeps the origin column.
            if let Some(p) = world.generator.chart_spawn()
                && let Some(s) = world.generator.atlases().iter().find_map(|a| a.storage_of(p))
            {
                let sdiv = |v: i64| v.div_euclid(CHUNK_SIZE as i64) as i32;
                world.ensure_region_data(ChunkCoord::new(sdiv(s[0]), sdiv(s[1]), sdiv(s[2])));
            } else {
                let cy = world.generator.height(0, 0).div_euclid(CHUNK_SIZE as i32);
                world.ensure_region_data(ChunkCoord::new(0, cy, 0));
            }
        }
        world
    }

    /// The default world (seed [`DEFAULT_SEED`]).
    pub fn generate() -> Self {
        Self::new(DEFAULT_SEED)
    }

    /// The one lazily created worker pool. Keeping construction here prevents
    /// direct lane/test entry points from each restating the spawn policy. It starts with the
    /// chart net already adopted: adoption publishes only a change, so a net adopted before the
    /// pool existed would otherwise never reach its gate.
    pub(in crate::world) fn worker_pool(&mut self) -> &mut pipeline::Workers {
        let fold = self.fold;
        self.workers.get_or_insert_with(|| {
            let workers = pipeline::Workers::spawn(pipeline::Workers::default_threads());
            workers.set_fold(fold);
            workers
        })
    }

    /// Whether `coord` is a loaded chunk awaiting its fresh mesh — dense data,
    /// no mesh, no job outstanding, not edited. The fresh-scan meshed-ness test.
    fn is_needs_mesh(&self, coord: Coord) -> bool {
        self.chunks
            .get(&coord)
            .is_some_and(|l| l.state.is_needs_mesh())
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

    /// Any generate/mesh/light/section claim or upload still outstanding.
    /// Counter reads only — idle `pump` uses this to skip the drain lane.
    pub fn anything_in_flight(&self) -> bool {
        self.building_meshes != 0
            || self.meshing_sections != 0
            || !self.generating.is_empty()
            || !self.light_inflight.is_empty()
            || !self.upload_queue.is_empty()
            || !self.light_apply_queue.is_empty()
            || !self.section_upload_queue.is_empty()
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
