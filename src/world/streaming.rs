//! Streaming: the [`World::stream`] pass, the eyes and the view boxes it stands on. Its stages
//! (pacing, generation, claims, the result drain, unloading, light, meshing, the far field and
//! its sections, gauges) are `World` methods in the child modules (struct lives in `mod.rs`).

use std::time::{Duration, Instant};

use voxel_engine::producer::{Budget, Progress};
use voxel_engine::{DVec3, Engine, FadeStyle};

use crate::block::appearance::BlockAppearance;

use crate::coord::{ByPass, ChunkBox, Face};
use crate::space::FaceFrame;
use crate::derived::Revision;
use crate::math::block_coord;

use super::chunk::{CHUNK_SIZE, Chunk, ChunkData};
use super::generation::{Classify, ColumnHeights};
use crate::block::registry::{HotTables, AIR};
use super::brick::ChunkPayload;
use super::heightmip::{BakeExtent, HeightMip};
use super::metric::{DyCap, EyeDist, EyeMetric, HeightEnvelope};
use super::section::SectionPos;
use super::summary::{CellError, CellSummary, SseBudget};
use super::{
    ColumnKey, Coord, DIRTY_BUDGET, FastMap, FastSet, LightLane, Loaded, MeshLane, MeshState,
    SECTION_UPLOAD_BUDGET, SectionFrontierKey, SectionLane, SectionState, Sky, StreamLane,
    UPLOAD_BUDGET_BYTES, UPLOAD_QUEUE_MAX, UPLOAD_SCAN_MAX, World, pipeline, pyramid, quadtree,
};
// What the stage modules name through `super::`.
use super::{
    ChartBend, DATA_MARGIN, LightSeed, SECTION_SLOT_FLOOR, adjust_count, admission_exhausted,
    bias_order, heightmip, lanes, seam, section, terrain,
};
#[cfg(test)]
use super::vis_log;

mod claims;
mod drain;
mod far;
mod gauges;
mod generate;
mod light;
mod mesh;
mod pacer;
mod sections;
mod unload;
mod window;

pub use gauges::{RemeshDistribution, StreamGauges};
pub(in crate::world) use far::{NearBounds, span_reach};
pub(in crate::world) use gauges::StreamCounters;
pub(in crate::world) use generate::{FailKey, GenCursor};
pub(in crate::world) use light::{LightGate, RemeshStats};
pub(in crate::world) use pacer::{StreamPacer, chunk_behind};
pub(in crate::world) use window::Window;
#[cfg(test)]
pub(in crate::world) use generate::GenRun;
use far::{eye_chunk, section_key};
use pacer::{LoadWindow, travel_heading};
#[cfg(test)]
use far::inside_near;
#[cfg(test)]
use pacer::FULL_EFFORT_SPEED_MPS;

/// Exact upload placement for a chunk mesh: integer chunk origin, full detail.
/// Pinned at upload so the GPU record is written once; draws only mark
/// visibility.
fn chunk_placement(coord: Coord) -> voxel_engine::MeshPlacement {
    voxel_engine::MeshPlacement::terrain(
        voxel_engine::IVec3::new(coord.x, coord.y, coord.z) * CHUNK_SIZE as i32,
        crate::ident::Detail::FULL,
    )
}

/// The GPU bytes a finished mesh will stage on upload (direction-major
/// vertices across every pass) — what the byte-based upload budget charges.
/// Counts both staged regions and the `Vec` fallback.
pub(in crate::world) fn mesh_output_bytes(data: &pipeline::MeshPayload) -> usize {
    data.vertex_bytes()
}

/// Vertex bytes a finished section mesh will stage (every slab × pass).
#[cfg(test)]
pub(in crate::world) fn section_output_bytes(data: &super::SectionMeshData) -> usize {
    data.vertex_bytes()
}

/// Edits whose chunk falls inside `pos`'s footprint and height domain. Free
/// function (not a `World` method) so callers needing only `&self.edits` — the
/// heightmip overlay refresh among them — don't have to borrow the rest of `World`.
#[cfg(test)]
pub(in crate::world) fn edits_in_footprint(
    edits: &FastMap<Coord, FastMap<usize, crate::block::registry::BlockId>>,
    pos: SectionPos,
) -> Vec<(Coord, Vec<(usize, crate::block::registry::BlockId)>)> {
    let cs = CHUNK_SIZE as i32;
    let span = pos.span();
    let (cx0, cz0) = (pos.min_x().div_euclid(cs), pos.min_z().div_euclid(cs));
    let cn = span / cs; // chunk columns per section side
    let cy_hi = super::section::DOMAIN_H / cs; // vertical chunk-layer count
    edits
        .iter()
        .filter(|(c, _)| {
            (cx0..cx0 + cn).contains(&c.x)
                && (cz0..cz0 + cn).contains(&c.z)
                && (0..cy_hi).contains(&c.y)
        })
        .map(|(&c, cells)| (c, cells.iter().map(|(&i, &b)| (i, b)).collect()))
        .collect()
}

/// Velocity prediction horizon: pre-loads sections ahead of eye motion so they're
/// ready by the time the eye reaches them. Conservative; tuning it larger preloads more
/// (safer for fast motion, low cost) but never shrinks the view.
const TAU_STREAM: f64 = 1.0;

/// Chart prediction lands on whole chunks. A flight's velocity jitters frame to frame; an exact
/// lookahead would move the predicted eye, and so recompute the frontier, every frame. Adding
/// zero folds a rounded `-0` into `+0`, so equal deltas key equal bits.
fn chart_delta(vel: DVec3) -> DVec3 {
    let step = CHUNK_SIZE as f64;
    (vel * TAU_STREAM / step).round() * step + DVec3::ZERO
}

/// Above this sample gap, treat eye motion as pause/teleport; discard velocity
/// to zero prediction. Generous (streaming may legitimately run at 15 Hz under
/// `stream_hz`); the speed cap below catches genuine discontinuities.
const MAX_PREDICT_SAMPLE_GAP: f64 = 0.5;

/// Above this apparent speed (m/s), treat the motion as a teleport rather than
/// travel: prediction is zeroed AND queued far work is purged, because its
/// admission-time priorities are stale where the eye is now.
const MAX_PREDICT_SPEED: f64 = 512.0;

/// An eye's velocity since its last sample, and the sample interval. A pause
/// past [`MAX_PREDICT_SAMPLE_GAP`], a non-finite eye or a non-positive interval
/// reads as rest, so nothing downstream fires on garbage input.
fn eye_sample(prev: Option<(DVec3, Instant)>, eye: DVec3, now: Instant) -> (DVec3, f64) {
    let Some((prev, t)) = prev else {
        return (DVec3::ZERO, 0.0);
    };
    let dt = now.duration_since(t).as_secs_f64();
    let v = (eye - prev) / dt;
    if eye.is_finite() && dt > 0.0 && dt <= MAX_PREDICT_SAMPLE_GAP && v.is_finite() {
        (v, dt)
    } else {
        (DVec3::ZERO, dt.min(MAX_PREDICT_SAMPLE_GAP))
    }
}

/// Above this speed the far-field key is bucketed so the sweep is not redone
/// every chunk step. The sweep costs the same at any speed; running it on
/// every step is what made a fast frame grow with speed. Slower flight keeps
/// the exact key, so rest and the spawn frontier stay bit-identical.
const FRONTIER_COARSE_SPEED: f64 = 1000.0;

/// Fixed chunk grid for that key. A speed-scaled grid moves its edges when the
/// sampled speed jitters, which rebuilds the sweep every frame.
const FRONTIER_CHUNK_QUANTUM: i32 = 32;

/// Fixed altitude grid, metres. Same reason: the eye's storage height is large,
/// and a quantum that tracks speed does not land on one value.
const FRONTIER_EYE_QUANTUM: f64 = 512.0;

impl World {
    /// Up face of the streaming centre. +Y until the first resolve, so
    /// pre-stream orders match the historical volume.
    pub(in crate::world) fn live_up(&self) -> Option<Face> {
        if self.stream_up_set { self.stream_up } else { Some(Face::PosY) }
    }

    /// Up face for `center`. An `Open` centre keeps the previous face while it
    /// is within one mesh-box radius (3-D chess of [`ViewVolume`]'s horizontal
    /// radius) of a chunk with that face, so an edge band does not flip the
    /// box every chunk. The first centre, with nothing committed, does not
    /// inherit +Y: `Open` there is isotropic.
    pub(in crate::world) fn resolve_stream_up(&self, center: Coord) -> Option<Face> {
        match self.generator.sky(center) {
            Sky::Axis(face) => Some(face),
            Sky::Open => {
                if !self.stream_up_set {
                    return None;
                }
                let Some(prev) = self.stream_up else {
                    return None;
                };
                let r = self.view.horizontal;
                if r > 0 && self.open_near_face(center, prev, r) {
                    Some(prev)
                } else {
                    None
                }
            }
        }
    }

    fn open_near_face(&self, center: Coord, face: Face, r: i32) -> bool {
        let want = Sky::Axis(face);
        for dx in -r..=r {
            for dy in -r..=r {
                for dz in -r..=r {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let c = Coord::new(center.x + dx, center.y + dy, center.z + dz);
                    if self.generator.sky(c) == want {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// The mesh box: chunks meshed and drawn around `center`, over the held window.
    fn mesh_box(&self, center: Coord) -> ChunkBox {
        self.view.mesh(center, self.live_up()).stretched(self.window_grow(center))
    }

    /// Radii new work may cover. Before the first stream this is the full view.
    fn load_volume(&self) -> super::ViewVolume {
        if self.loading_full() {
            self.view
        } else {
            super::ViewVolume::new(self.load_h, self.load_v.max(0))
        }
    }

    /// The loading window is the whole view (rest, walking, or not yet streamed).
    fn loading_full(&self) -> bool {
        self.load_h < 0
            || (self.load_h >= self.view.horizontal && self.load_v >= self.view.vertical)
    }

    /// Chunks new mesh admission will spend workers on: the mesh box, grown window included, while
    /// the whole view loads; a reduced window keeps to the eye's band.
    fn load_mesh_box(&self, center: Coord) -> ChunkBox {
        if self.loading_full() {
            self.mesh_box(center)
        } else {
            self.load_volume().mesh(center, self.live_up())
        }
    }

    /// Voxel data the loading window generates and lights: the load mesh box plus one data shell.
    fn load_data_box(&self, center: Coord) -> ChunkBox {
        if self.loading_full() {
            self.data_box(center)
        } else {
            self.load_volume().data(center, self.live_up())
        }
    }

    /// Publish the speed-reduced radii and the travel heading. A change is a
    /// view event: generation rebuilds and the mesh/light seed follows. Only a
    /// reduced window takes a heading, so the trail filter never runs while
    /// the whole view is loading.
    fn apply_loading_radius(&mut self) {
        let h = self.stream_pacer.load_radius(self.view.horizontal);
        let v = self.stream_pacer.load_radius(self.view.vertical);
        if self.load_h != h || self.load_v != v {
            self.load_h = h;
            self.load_v = v;
            self.load_moved = true;
        }
        let heading = if self.loading_full() {
            0
        } else {
            travel_heading(self.stream_pacer.travel(), self.live_up(), self.load_heading)
        };
        if heading != self.load_heading {
            self.load_heading = heading;
            self.heading_changed = true;
        }
    }

    /// The speed-reduced loading window. `None` while it is the whole view or
    /// before the first stream: then every lane keeps its full-view rules.
    fn load_window(&self) -> Option<LoadWindow> {
        if self.loading_full() {
            return None;
        }
        let center = self.center?;
        let up = self.live_up();
        let volume = self.load_volume();
        Some(LoadWindow {
            center,
            mesh: volume.mesh(center, up),
            data: volume.data(center, up),
            travel: self.stream_pacer.travel(),
            up,
            trail: self.load_heading != 0,
        })
    }

    /// Whether a new mesh job for `coord` is worth admitting: always for the
    /// full window (the mesh lane checks the draw box itself), else inside the
    /// reduced mesh box and not behind the player.
    pub(in crate::world) fn admits_mesh(&self, coord: Coord) -> bool {
        self.load_window()
            .is_none_or(|w| w.covers(self.fold.fold(coord), false))
    }

    /// Whether a light settle for `coord` is worth running. Light covers the
    /// data box, so a mesh-box edge chunk sees lit neighbours.
    pub(in crate::world) fn admits_light(&self, coord: Coord) -> bool {
        self.load_window()
            .is_none_or(|w| w.covers(self.fold.fold(coord), true))
    }

    /// `will_accept_chunk`'s region: the spawn slab always, otherwise the load
    /// data box and not behind the player. No centre and no slab accepts nothing.
    fn admits_new(&self, coord: Coord) -> bool {
        if self.spawn_slab.is_some_and(|slab| self.view_contains(slab, coord)) {
            return true;
        }
        let Some(center) = self.center else {
            return false;
        };
        let folded = self.fold.fold(coord);
        match self.load_window() {
            Some(window) => window.covers(folded, true),
            None => self.data_box(center).contains(folded),
        }
    }

    /// Record that `coord` is owed a light settle the reduced window dropped.
    pub(in crate::world) fn owe_light(&mut self, coord: Coord) {
        if self.chunks.contains_key(&coord) {
            self.light_owed.insert(coord);
        }
    }

    /// Seed the owed settles the loading window covers again. The full window
    /// covers all of them.
    fn reseed_owed_light(&mut self) {
        if self.light_owed.is_empty() {
            return;
        }
        let mut owed = std::mem::take(&mut self.light_owed);
        let mut seeded = false;
        owed.retain(|&coord| {
            if !self.admits_light(coord) {
                return true;
            }
            self.seed_light(coord, super::LightSeed::Store);
            seeded = true;
            false
        });
        self.light_owed = owed;
        if seeded {
            self.light_pending.set();
        }
    }

    /// The data box: the mesh box plus one [`DATA_MARGIN`] shell of voxel data,
    /// so edge chunks can cull against neighbours that are loaded but unmeshed.
    fn data_box(&self, center: Coord) -> ChunkBox {
        self.view.data(center, self.live_up()).stretched(self.window_grow(center))
    }

    /// The unload box: the mesh box plus the unload hysteresis, past which
    /// chunks are freed.
    pub(in crate::world) fn unload_box(&self, center: Coord) -> ChunkBox {
        self.view.unload(center, self.live_up()).stretched(self.window_grow(center))
    }

    /// Ring buckets for the worklists around `center` (see `ViewVolume::worklist_rings`).
    pub(in crate::world) fn worklist_rings(&self, center: Coord) -> usize {
        let [below, above] = self.window_grow(center);
        self.view.worklist_rings(self.live_up(), below.max(above))
    }

    /// Whether `coord` is inside the current mesh box. The single mesh-view
    /// check: the enqueue gate (the [`MeshLane`] ready predicate) and the
    /// apply gate ([`mesh_result_applies`](Self::mesh_result_applies)) both call
    /// this, so a chunk is enqueued only if its result would be accepted.
    /// `false` before the first stream (no centre yet).
    pub(in crate::world) fn in_mesh_box(&self, coord: Coord) -> bool {
        self.center
            .is_some_and(|c| self.view_contains(self.mesh_box(c), coord))
    }

    /// Whether `coord` lies in view box `b` once the charts around a storage centre are unfolded
    /// into one net (SPACE-ARCHITECTURE §7); plain containment elsewhere.
    #[inline]
    pub(in crate::world) fn view_contains(&self, b: ChunkBox, coord: Coord) -> bool {
        b.contains(self.fold.fold(coord))
    }

    /// The real chunks of view box `b`: across the unfolded chart net around a storage centre
    /// (storage that holds nothing is skipped), `b` itself elsewhere.
    pub(in crate::world) fn view_coords(&self, b: ChunkBox) -> impl Iterator<Item = Coord> + use<> {
        let fold = self.fold;
        b.coords().filter_map(move |v| fold.unfold(v))
    }

    /// The real chunks of view box `b` that view box `other` does not hold. Both boxes are in the
    /// current net, and a real chunk folds back to the net cell it unfolded from.
    pub(in crate::world) fn view_shell(&self, b: ChunkBox, other: ChunkBox) -> impl Iterator<Item = Coord> + use<> {
        let fold = self.fold;
        b.coords_outside(other).filter_map(move |v| {
            let c = fold.unfold(v)?;
            // The walk skips `other` by net cell: a chunk folding elsewhere would be misjudged.
            debug_assert_eq!(fold.fold(c), v, "net cell {v:?} unfolds to {c:?}, which folds elsewhere");
            Some(c)
        })
    }

    /// Upload placement of chunk `coord`'s meshes: a storage chunk of a round world is drawn bent
    /// through its chart cage (made once per loaded chunk; corners relative to an anchor block, so
    /// they stay precise in `f32`), every other chunk at its integer origin.
    fn placement_of(&mut self, coord: Coord, eng: &mut Engine) -> voxel_engine::MeshPlacement {
        if let Some(&cage) = self.cages.get(&coord) {
            return voxel_engine::MeshPlacement::caged(cage, crate::ident::Detail::FULL);
        }
        let Some(corners) = self.seams.cage(coord) else { return chunk_placement(coord) };
        let a = corners[0].floor();
        let anchor = voxel_engine::IVec3::new(a.x as i32, a.y as i32, a.z as i32);
        let rel = corners.map(|c| (c - a).as_vec3());
        match eng.create_cage(anchor, rel) {
            Some(cage) => {
                self.cages.insert(coord, cage);
                voxel_engine::MeshPlacement::caged(cage, crate::ident::Detail::FULL)
            }
            None => chunk_placement(coord),
        }
    }

    /// Peek the dirty-remesh hint without consuming it.
    pub(in crate::world) fn dirty_pending(&self) -> bool {
        self.pending_dirty.get()
    }

    /// The every-frame half of streaming: land finished worker results, run
    /// the budgeted uploads, and remesh edited chunks — everything whose
    /// LATENCY the player sees directly. The game runs this every frame no
    /// matter how `stream_hz` throttles [`stream`](Self::stream), so a 15 Hz
    /// Minimum profile still publishes finished terrain the frame it lands
    /// and a mined block still vanishes the same frame it was clicked.
    /// Steady-state cost is a channel poll and two sticky-flag checks.
    /// `eng` is `None` only in headless tests; GPU work panics without it.
    pub fn pump(
        &mut self,
        mut eng: Option<&mut Engine>,
        sched: &mut crate::sched::Scheduler,
        appearance: &dyn BlockAppearance,
    ) {
        self.remesh_stats.drop_stale_this_frame = 0;
        // Palette growth appends new block texture layers before any upload
        // this frame references a new layer.
        if let Some(eng) = eng.as_deref_mut() {
            self.refresh_textures(eng, appearance);
        }
        // Idle: no claim can produce a `Done`, so skip try_recv and the
        // deadline Instant. Dirty-remesh and lod-clip stay (flag checks).
        if self.anything_in_flight() {
            let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::StreamDrain);
            let drain_lane = self.lanes().drain;
            sched.run_manual(drain_lane, self, eng.as_deref_mut());
        }
        // The synchronous edit remesh: self-gates on `pending_dirty`, so an
        // editless frame pays one flag check and does not need the engine.
        let dirty_lane = self.lanes().dirty_remesh;
        sched.run_manual(dirty_lane, self, eng.as_deref_mut());
        // Fold any settle events into the LOD clip the moment they land.
        self.refresh_lod_clip();
    }

    /// Land worker results, queue generation/meshing, free distant chunks —
    /// the topology half, run at `stream_hz` (or out of band on forced
    /// refreshes). [`pump`](Self::pump) covers the every-frame latency half;
    /// the drain/dirty lanes here are second-run no-ops on a pumped frame.
    /// Steady-state zero cost: one channel poll, lazy unload/generate on boundary cross.
    pub fn stream(
        &mut self,
        center: DVec3,
        mut eng: Option<&mut Engine>,
        sched: &mut crate::sched::Scheduler,
        appearance: &dyn BlockAppearance,
    ) {
        if let Some(eng) = eng.as_deref() {
            let stats = eng.mesh_stats();
            self.gpu_live_slots = stats.live_slots;
            self.slot_ceiling = stats.cpu_cull_max.max(1);
        }
        let stager = eng.as_ref().map(|e| e.mesh_stager());
        let (center_chunk, far_chunk, full_pass, far_moved) = self.begin_stream(center, stager);
        // Each lane creates its own budget window, not shared: lanes run
        // sequentially, so a single frame-start snapshot would starve lanes
        // after the first.
        // Textures, worker results, and edit remeshes land through the ONE
        // owner of that trio — after the centre update above, so a
        // boundary-cross frame's results drain against the live centre, not
        // the one they'd be discarded by. `stream_phase` pumps only on frames
        // the topology pass does not run; this is the pump on stream-due frames.
        self.pump(eng.as_deref_mut(), sched, appearance);
        if full_pass {
            self.unload_far(
                center_chunk,
                eng.as_deref_mut()
                    .expect("unload on a boundary cross needs the engine"),
            );
            self.cross_boundary(center_chunk);
        }
        self.finish_load_window(center_chunk, full_pass);
        // Runs every frame to drain a boundary-cross flood across frames;
        // self-gates on `pending_gen` so a settled world pays one flag check.
        // Placed after unload so freed slots can regenerate.
        {
            let gen_lane = self.lanes().generate;
            sched.run_manual(gen_lane, self, None);
        }
        if self.radius_shrunk.take() {
            // Meshes are about to be freed: the settled scan must restart.
            self.lod_clip_shrunk.set();
            // Free meshes between new radius and unload ring (data stays).
            // Air/NeedsMesh own no handle.
            // Ready chunks drop to NeedsMesh; Dirty chunks stay dirty
            // (prev: None) so same-frame dirty pass still remeshes them.
            let keep = self.mesh_box(center_chunk);
            let fold = self.fold;
            for (&coord, loaded) in self.chunks.iter_mut() {
                if keep.contains(fold.fold(coord)) {
                    continue;
                }
                // Ready becomes NeedsMesh; Dirty stays Dirty (prev: None); a
                // NeedsMesh carrying a rebuild's old mesh drops it (keeping
                // its claim truthful — the in-flight result resolves as
                // stale). Same-frame dirty pass remeshes Dirty chunks.
                // Retire frees the old mesh.
                let next = match loaded.state {
                    MeshState::Ready(_) => MeshState::needs_mesh(),
                    MeshState::Dirty { prev: Some(_) } => MeshState::Dirty { prev: None },
                    MeshState::NeedsMesh {
                        building,
                        prev: Some(_),
                    } => MeshState::NeedsMesh {
                        building,
                        prev: None,
                    },
                    _ => continue,
                };
                let stays_dirty = matches!(next, MeshState::Dirty { .. });
                loaded.retire(
                    next,
                    eng.as_deref_mut()
                        .expect("radius shrink frees GPU meshes"),
                );
                // A retired `Dirty` chunk (its drawn mesh just freed) still needs
                // the same-frame dirty pass to remesh it — which only runs when
                // `pending_dirty` is set. Set it explicitly here rather than
                // hoping some other path already did.
                if stays_dirty {
                    self.pending_dirty.set();
                }
            }
        }
        // Light settling: worklist lane. Trivial grids publish synchronously;
        // only the dense surface band reaches the worker pool.
        {
            let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::StreamLight);
            if self.lighting {
                let light_lane = self.lanes().light_admit;
                sched.run_manual(light_lane, self, None);
                // No level-triggered mesh-lane forcing here: every event that
                // can flip a chunk's mesh-readiness arms `pending_fresh` WITH
                // a seed (`settle_light` seeds self + moved-border neighbours,
                // `store_chunk` seeds self + 6, the light gate re-seeds timed
                // chunks, fail/cancel re-seed). The old unconditional forcing
                // while any light work existed papered over wedged
                // `light_inflight` claims — fixed at the root in `accept_light`
                // — at the cost of a full admission pass every frame of a flood.
            } else {
                // No flood. Edit seeds stay dormant so re-enabling lighting only
                // settles chunks changed while it was off; `light_ready` bypasses
                // this worklist while disabled.
            }
        }
        // Sync dirty remesh (edited chunks, budgeted) then the fresh mesh lane
        // (worklist seeded on load/light-move, O(shell) not a whole-map rescan).
        {
            let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::StreamMesh);
            let dirty_lane = self.lanes().dirty_remesh;
            sched.run_manual(dirty_lane, self, eng.as_deref_mut());
            // Advance the light-gate degrade timers and keep still-waiting chunks on
            // the worklist (their degrade fires on the clock, which raises no re-seed
            // event) BEFORE the mesh lane reads them.
            self.tick_light_gate();
            // The mesh lane evicts blocked/stale seeds itself (see `admit`),
            // so the worklist stays O(fresh work) with no separate prune here.
            // A deep upload queue pauses the RUN (never `ready` — that would
            // evict the whole worklist with no re-seed event): `pending_fresh`
            // stays raised and admission self-resumes as uploads drain.
            if !self.upload_backlogged() {
                let mesh_lane = self.lanes().mesh_admit;
                sched.run_manual(mesh_lane, self, None);
            }
            // Level-triggered backstop to the edge-triggered degraded clear: once
            // ALL light work is quiescent, any chunk still degraded is owed a
            // remesh that no future light-arrival event will ever deliver (its
            // missing neighbour is already terminal). Promote it to final now.
            self.flush_degraded_terminal();
        }
        // LOD2 section far field: skipped entirely when disabled (zero cost). Visible
        // set rebuilt every pass because sections become Ready asynchronously.
        if self.lod2 {
            let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::StreamTiles);
            // Update pyramid unit to track the current view distance.
            self.section_pyramid.unit = self.view.lod_unit();
            self.update_lod_face(far_chunk);
            // Until the bake lands, selection uses the worst-case ladder;
            // mip only coarsens, no upward pops during bake.
            let mip_lane = self.lanes().mip;
            sched.run_manual(mip_lane, self, None);
            // Section overlay lane: refresh the edit overlay BEFORE
            // selection/occlusion/material read it (the frontier's error
            // coarsening below already consults it).
            let overlay_lane = self.lanes().section_overlay;
            sched.run_manual(overlay_lane, self, None);
            self.refresh_frontier(far_chunk);
            if full_pass || far_moved {
                self.unload_sections(
                    far_chunk,
                    eng.as_deref_mut()
                        .expect("section unload on a boundary cross needs the engine"),
                );
                self.pending_sections.set();
            }
            // Section dirty-remesh lane: free GPU meshes of edited sections so
            // they re-extract from the updated generator overlay.
            let section_remesh_lane = self.lanes().section_remesh;
            sched.run_manual(section_remesh_lane, self, eng.as_deref_mut());
            // The floor may be full of sections this frontier no longer draws.
            // Unload runs only on a boundary cross, so a still camera never
            // drops them and admission stays refused.
            self.reclaim_blocked_sections(far_chunk, eng.as_deref_mut());
            let section_lane = self.lanes().section_admit;
            sched.run_manual(section_lane, self, None);
            // Section visible-set lane: re-resolve the covering only when an
            // event moved it (upload/unload/free/claim release/frontier or
            // ladder change) or while admission is still pending — the
            // level-triggered backstop that keeps holes re-arming the lane.
            // A converged, still far field pays a flag check, no covering walk.
            if self.section_cover_dirty.take() || self.pending_sections.get() {
                let visible_lane = self.lanes().section_visible;
                sched.run_manual(visible_lane, self, eng.as_deref_mut());
            }
        }
        // Occlusion is derived state, rebuilt here at the `&mut` sync point (never
        // in the `&self` render). The lane self-gates: it rebuilds only when the
        // adaptive gate is active AND an input changed (or it was just
        // activated), and always updates `occlusion_active` (so turning the gate
        // off lets render draw everything). A CPU-bound world pays nothing.
        {
            let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::StreamOcclusion);
            let occ_lane = self.lanes().occlusion;
            sched.run_manual(occ_lane, self, eng.as_deref_mut());
        }
        // Unloads/boundary crossings above may have shrunk the settled rings;
        // fold them in before this frame renders.
        self.refresh_lod_clip();
        #[cfg(debug_assertions)]
        self.debug_assert_liveness();
    }

    /// The prologue of [`stream`](Self::stream): the near and far eyes, velocity and pacing, the
    /// chart net and up face, the worklist rings, and the worker view. Returns the centre chunk,
    /// the far field's centre chunk, whether this is a full pass (centre, up face or chart net
    /// moved) and whether the far centre moved.
    pub(in crate::world) fn begin_stream(
        &mut self,
        center: DVec3,
        stager: Option<voxel_engine::MeshStager>,
    ) -> (Coord, Coord, bool, bool) {
        // On a round world streaming stands in the chart's storage cells, and the far field on the
        // chart under the eye (also above the near window's reach).
        let (center, far) = self.place_eyes(center);
        let now = crate::sched::now();
        // Far-field prediction treats >512 m/s as a teleport and predicts
        // nothing. Far jobs left behind by a jump are re-keyed and descheduled
        // by the pool's per-epoch sync, so no separate purge is needed here.
        let (far_vel, sample_dt) = eye_sample(self.section_eye_prev, far, now);
        self.section_vel = if far_vel.length() <= MAX_PREDICT_SPEED { far_vel } else { DVec3::ZERO };
        self.section_eye_prev = far.is_finite().then_some((far, now));
        // The pacer sees the near eye at any finite speed, in the frame the
        // near window streams in, so its trail filter points the right way.
        let (travel, travel_dt) = eye_sample(self.near_eye_prev, center, now);
        self.near_eye_prev = center.is_finite().then_some((center, now));
        self.counters.light_admitted_last = 0;
        let center_chunk = eye_chunk(center);
        let far_chunk = eye_chunk(far);
        let far_moved = self.set_far_center(far_chunk);
        // Update centre before draining: old centre may be a sentinel, so draining
        // against it would discard all results and regenerate them immediately.
        // An up-face change is the same kind of pass: the box changed shape.
        let prev_center = self.center;
        let leaving = (self.prev_unload_box, self.fold);
        let center_moved = Some(center_chunk) != self.center;
        let fold_changed = center_moved && self.adopt_fold(center_chunk);
        let up_changed = if center_moved || !self.stream_up_set {
            let up = self.resolve_stream_up(center_chunk);
            let changed = if self.stream_up_set {
                up != self.stream_up
            } else {
                // Pre-stream orders assume +Y. A different first face reshapes.
                up != Some(Face::PosY)
            };
            self.stream_up = up;
            self.stream_up_set = true;
            changed
        } else {
            false
        };
        if up_changed || fold_changed {
            self.follow_frame(center_chunk, leaving);
        }
        let window_moved = self.place_window(center_chunk, center_moved || up_changed || fold_changed);
        let full_pass = center_moved || up_changed || fold_changed || window_moved;
        self.center = Some(center_chunk);
        // A new chart net means the near eye changed frames: that sample's
        // velocity mixes two storage origins.
        self.stream_pacer.observe(travel, travel_dt, fold_changed);
        let queued_near = !self.light_worklist.is_empty()
            || !self.mesh_worklist.is_empty()
            || !self.light_inflight.is_empty()
            || !self.light_apply_queue.is_empty();
        self.stream_pacer.set_boost(queued_near, self.last_stream_secs);
        self.last_stream_secs = sample_dt;
        // Re-bucket worklists around the live centre before any lane (or pump
        // insert) runs. O(n) once per boundary cross; a no-op when the rings,
        // centre, and up face already match.
        if full_pass {
            let up = self.live_up();
            let rings = self.worklist_rings(center_chunk);
            self.mesh_worklist.fit(center_chunk, rings, up);
            self.light_worklist.fit(center_chunk, rings, up);
        }
        self.publish_view(center_chunk, far_chunk, stager);
        // Crossing a chunk boundary moves the BFS root, so the visible set is stale.
        self.occlusion_dirty.raise(full_pass);
        // The ring geometry is centred on the eye: a boundary cross SHIFTS the
        // settled rings by the move's chess distance across the up axis (a
        // move along that axis, an up-face change, or the first pass restarts
        // the scan) — see `shift_lod_clip`.
        if full_pass {
            if up_changed || fold_changed {
                self.lod_clip_shrunk.set();
            } else {
                self.shift_lod_clip(prev_center, center_chunk);
            }
        }
        (center_chunk, far_chunk, full_pass, far_moved)
    }

    /// The engine-free rest of a full pass, after [`unload_far`](Self::unload_far).
    pub(in crate::world) fn cross_boundary(&mut self, center: Coord) {
        // Stale queued uploads (the trailing edge of fast movement) release
        // in ONE pass here instead of trickling through the drain budget.
        self.prune_upload_queue();
        // Sync-generate the centre only when it is missing and not already
        // claimed: a claimed job is imminent and the previous centre's
        // collision halo still exists.
        self.ensure_data(center);
        self.pending_gen.set();
        self.seed_load_window(center);
    }

    /// Mesh and light seeds for a pass that did not already cross a boundary.
    /// `seeded` means [`cross_boundary`](Self::cross_boundary) ran.
    fn finish_load_window(&mut self, center: Coord, seeded: bool) {
        if !seeded && (self.load_moved || self.heading_changed) {
            self.seed_load_window(center);
            self.pending_gen.set();
        }
        self.load_moved = false;
        self.heading_changed = false;
    }

    /// Seed meshes for chunks the loading window just took on, and the light
    /// settles it dropped that it covers again. Only the part of the window the
    /// last seeded box did not hold is new; a changed heading rescans the whole
    /// window, since the trail it skipped may be wanted now. A reduced window
    /// first drops queued seeds it no longer covers.
    fn seed_load_window(&mut self, center: Coord) {
        if !self.loading_full() {
            self.prune_admission_worklists();
        }
        let window = self.load_mesh_box(center);
        let fresh: Vec<Coord> = match self.prev_mesh_box {
            Some(prev) if !self.heading_changed => self
                .view_shell(window, prev)
                .filter(|&c| self.awaits_mesh(c) && self.admits_mesh(c))
                .collect(),
            _ => self
                .view_coords(window)
                .filter(|&c| self.awaits_mesh(c) && self.admits_mesh(c))
                .collect(),
        };
        self.mesh_worklist.extend(fresh);
        self.pending_fresh.set();
        self.prev_mesh_box = Some(window);
        self.reseed_owed_light();
    }

    /// Drop queued mesh and light seeds the reduced window does not cover. A
    /// dropped light seed is owed: [`reseed_owed_light`](Self::reseed_owed_light)
    /// brings it back once the window covers the chunk again.
    fn prune_admission_worklists(&mut self) {
        let Some(window) = self.load_window() else {
            return;
        };
        let fold = self.fold;
        self.mesh_worklist.retain(|c| window.covers(fold.fold(*c), false));
        let (chunks, owed) = (&self.chunks, &mut self.light_owed);
        self.light_worklist.retain(|c| {
            let keep = window.covers(fold.fold(*c), true);
            if !keep && chunks.contains_key(c) {
                owed.insert(*c);
            }
            keep
        });
    }

    /// Streaming priority. `Some(face)`: tangent chess, distance along the
    /// face ×2 (terrain before sky). `None`: plain 3-D chess. +Y is
    /// `ring.max(2 * updown)`.
    pub(in crate::world) fn order(a: Coord, b: Coord, up: Option<Face>) -> i32 {
        match up {
            None => a.chess3(b),
            Some(face) => a.across(b, face).max(2 * a.along(b, face)),
        }
    }

    /// Headless settle: centre on `pos`, fill the mesh box with data, mark every
    /// in-view chunk as a final Air mesh, and drain worklists so
    /// [`entry_complete`](Self::entry_complete) holds without an Engine.
    #[cfg(test)]
    pub fn settle_around(&mut self, pos: DVec3) {
        // Same eyes as `stream`: a charted body stands in storage, so a later
        // quiet frame does not see a boundary cross and demand an engine.
        let (pos, far) = self.place_eyes(pos);
        let center = eye_chunk(pos);
        self.center = Some(center);
        self.set_far_center(eye_chunk(far));
        let _ = self.adopt_fold(center);
        self.place_window(center, true);
        for coord in self.mesh_box(center).coords() {
            self.ensure_data(coord);
            if let Some(loaded) = self.chunks.get_mut(&coord)
                && !matches!(loaded.state, MeshState::Air | MeshState::Ready(_))
            {
                loaded.state = MeshState::Air;
            }
        }
        self.mesh_worklist.clear();
        self.light_worklist.clear();
        self.generating.clear();
        self.light_inflight.clear();
        self.upload_queue.clear();
        self.light_apply_queue.clear();
        self.section_upload_queue.clear();
        self.section_desired.clear();
        self.light_gate.degraded.clear();
        self.light_gate.dirty.clear();
        self.light_gate.blocked_since.clear();
        self.pending_fresh.take();
        self.pending_gen.take();
        self.pending_dirty.take();
        self.pending_sections.take();
        let _ = self.worker_pool();
    }
}

#[cfg(test)]
mod flight_bench;
#[cfg(test)]
mod ground_walk;
#[cfg(test)]
mod tests;
