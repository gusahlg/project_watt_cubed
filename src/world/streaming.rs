//! Streaming: the [`World::stream`] pass and everything it drives — draining
//! worker results, queueing generation and mesh jobs, budgeted uploads,
//! unloading far chunks, and the radius/centre bookkeeping. Code motion only:
//! these are `World` methods; the struct itself lives in `mod.rs`.

use std::time::{Duration, Instant};

use voxel_engine::{DVec3, Engine};

use crate::coord::{ByPass, ChunkBox, ChunkCoord, Face};
use crate::derived::Revision;
use crate::math::block_coord;

use super::chunk::{CHUNK_SIZE, Chunk};
use super::generation::TerrainGenerator;
use super::mesh::ChunkMeshData;
use super::section::SectionPos;
use super::{
    Coord, DIRTY_BUDGET, FastMap, FastSet, LightLane, Loaded, MeshLane, MeshState,
    SECTION_UPLOAD_BUDGET, SectionLane, SectionState, UPLOAD_BUDGET, World, lane_enqueue,
    lane_integrate, light, mesh, pipeline, pyramid, quadtree,
};

/// How long a chunk's fresh mesh may wait on neighbour light before it is meshed
/// DEGRADED — missing neighbour light planes stand in as fully-lit open-sky — and
/// later remeshed through the existing `Dirty` machinery once real light lands.
///
// Tuned against `time_to_first_full_render`. Wait-time gating (rather than
// meshing unconditionally) is load-bearing: at a cold world entry the
// overwhelming majority of chunks receive neighbour light well within this window
// and mesh once with final smooth light, so only the few stragglers ever degrade —
// which is what keeps the ≤3-thread worker pool (≈46 ms/mesh job) from a remesh
// storm where every chunk meshes twice. This is its final home.
const LIGHT_WAIT_DEGRADE: Duration = Duration::from_millis(150);

/// State backing the light-gate degraded path. Bundled into one struct so
/// the feature adds a single field to [`World`] (`world/mod.rs` is being edited
/// concurrently — this keeps the merge surface to one line there).
///
/// `blocked_since` records, per chunk, the first frame its fresh mesh was observed
/// blocked purely on neighbour light (data present, light not settled). `degraded`
/// is the set of chunks currently drawing a degraded (known-not-final) mesh, still
/// owed a remesh once their real light arrives.
#[derive(Default)]
pub(in crate::world) struct LightGate {
    blocked_since: FastMap<Coord, Instant>,
    degraded: FastSet<Coord>,
}

impl World {
    /// The mesh box: chunks meshed and drawn around `center`.
    fn mesh_box(&self, center: Coord) -> ChunkBox {
        self.view.mesh(center)
    }

    /// The data box: the mesh box plus one [`DATA_MARGIN`] shell of voxel data,
    /// so edge chunks can cull against neighbours that are loaded but unmeshed.
    fn data_box(&self, center: Coord) -> ChunkBox {
        self.view.data(center)
    }

    /// The unload box: the mesh box plus the unload hysteresis, past which
    /// chunks are freed.
    fn unload_box(&self, center: Coord) -> ChunkBox {
        self.view.unload(center)
    }

    /// Whether `coord` is inside the current mesh box. The single mesh-view
    /// check: the enqueue gate (the [`MeshLane`] ready predicate) and the
    /// apply gate ([`mesh_result_applies`](Self::mesh_result_applies)) both call
    /// this, so a chunk is enqueued only if its result would be accepted.
    /// `false` before the first stream (no centre yet).
    pub(in crate::world) fn in_mesh_box(&self, coord: Coord) -> bool {
        self.center.is_some_and(|c| self.mesh_box(c).contains(coord))
    }

    /// Land worker results, queue generation/meshing, free distant chunks.
    /// Steady-state zero cost: one channel poll, lazy unload/generate on boundary cross.
    pub fn stream(&mut self, center: DVec3, eng: &mut Engine) {
        // Palette growth re-uploads the block texture array before any meshing
        // this frame, so vertices never reference a layer that isn't there.
        // Covers the initial upload too (0 tracked -> N on the first stream).
        self.refresh_textures(eng);
        let s = CHUNK_SIZE as i32;
        let center_chunk = ChunkCoord::new(
            block_coord(center.x).div_euclid(s),
            block_coord(center.y).div_euclid(s),
            block_coord(center.z).div_euclid(s),
        );
        // Adopt the real centre BEFORE draining: after a radius change or
        // world reset the stored centre is a far-away sentinel, and draining
        // against it would discard every landed result - even in-range ones -
        // only to regenerate them moments later.
        let full_pass = Some(center_chunk) != self.center;
        self.center = Some(center_chunk);
        // Crossing a chunk boundary moves the BFS root, so the visible set is stale.
        self.occlusion_dirty.raise(full_pass);
        // Time budgets, not counts: each loop below mints its OWN fresh
        // admission window at the instant it starts — the lanes run
        // sequentially, so one shared frame-start snapshot would leave every
        // lane after the first pre-expired (world-entry starvation).
        // Land worker results before the scans below, so freshly generated
        // chunks count as data this frame and finished meshes draw this frame.
        {
            let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::StreamDrain);
            let apply = pipeline::Deadline::from_budget(pipeline::LIGHT_APPLY_BUDGET);
            self.drain_results(eng, apply);
        }
        if full_pass {
            self.unload_far(center_chunk, eng);
            self.request_region_data(center_chunk);
            // The cross moved the mesh box: re-seed every already-loaded chunk
            // still awaiting a fresh mesh, so one that was loaded earlier as data
            // margin and just entered the box meshes now. Cheap (a boundary-cross
            // O(loaded) pass, like unload/generate), and it is what lets the mesh
            // worklist replace the old whole-map rescan without leaving a hole.
            let fresh: Vec<Coord> = self
                .chunks
                .iter()
                .filter(|(_, l)| l.state.is_needs_mesh())
                .map(|(&c, _)| c)
                .filter(|&c| self.in_mesh_box(c))
                .collect();
            self.mesh_worklist.extend(fresh);
            self.pending_fresh.set();
        }
        if self.radius_shrunk.take() {
            // Meshes between the new view radius and the unload ring survive
            // unload_far's hysteresis; free them now (data stays loaded).
            // `Air`/`NeedsMesh` (building or not) own no handle — nothing to free.
            //
            // Behaviour-preserving mapping (the old loop touched only the
            // handle, never the `dirty` index): a `Ready` chunk drops to
            // `NeedsMesh`, but a `Dirty` chunk STAYS dirty (→ `prev: None`) so
            // the same-frame dirty pass still remeshes it exactly as before.
            let keep = self.mesh_box(center_chunk);
            for (&coord, loaded) in self.chunks.iter_mut() {
                if keep.contains(coord) {
                    continue;
                }
                // A `Ready` chunk drops to `NeedsMesh`; a `Dirty` chunk STAYS
                // dirty (→ `prev: None`) so the same-frame dirty pass still
                // remeshes it. `retire` frees the outgoing mesh exactly once;
                // handle-less states are left untouched.
                let next = match loaded.state {
                    MeshState::Ready(_) => MeshState::NeedsMesh { building: false },
                    MeshState::Dirty { prev: Some(_) } => MeshState::Dirty { prev: None },
                    _ => continue,
                };
                let stays_dirty = matches!(next, MeshState::Dirty { .. });
                loaded.retire(next, eng);
                // A retired `Dirty` chunk (its drawn mesh just freed) still needs
                // the same-frame dirty pass to remesh it — which only runs when
                // `pending_dirty` is set. Set it explicitly here rather than
                // hoping some other path already did.
                if stays_dirty {
                    self.pending_dirty.set();
                }
            }
        }
        // Light settling: a worklist lane. Analytic-trivial grids (deep opaque,
        // above-surface open air) publish synchronously in `store_chunk`, so only
        // the residual Dense surface band reaches the pool. StreamLight times the
        // main-thread snapshot/submit only (the flood shows under `workers:light`).
        {
            let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::StreamLight);
            if self.lighting {
                lane_enqueue::<LightLane>(self, center_chunk, pipeline::Deadline::from_budget(pipeline::LIGHT_APPLY_BUDGET));
                // While light is unsettled, keep the mesh lane armed so it
                // re-checks `light_ready` as grids land.
                if !self.light_worklist.is_empty() || !self.light_inflight.is_empty() {
                    self.pending_fresh.set();
                }
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
            self.remesh_dirty(center_chunk, eng);
            // Advance the light-gate degrade timers and keep still-waiting chunks on
            // the worklist (their degrade fires on the clock, which raises no re-seed
            // event) BEFORE the mesh lane reads them.
            self.tick_light_gate();
            // The mesh lane evicts blocked/stale seeds itself (see `lane_enqueue`),
            // so the worklist stays O(fresh work) with no separate prune here.
            lane_enqueue::<MeshLane>(self, center_chunk, pipeline::Deadline::from_budget(pipeline::STREAM_BUDGET));
            // Level-triggered backstop to the edge-triggered degraded clear: once
            // ALL light work is quiescent, any chunk still degraded is owed a
            // remesh that no future light-arrival event will ever deliver (its
            // missing neighbour is already terminal). Promote it to final now.
            self.flush_degraded_terminal(eng);
        }
        // LOD2 section far field: skipped entirely when disabled (zero cost). Visible
        // set rebuilt every pass because sections become Ready asynchronously.
        if self.lod2 {
            let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::StreamTiles);
            // Update pyramid unit to track the current view distance.
            self.section_pyramid.unit = (self.view.horizontal * CHUNK_SIZE as i32) as f32;
            if full_pass {
                self.unload_sections(center_chunk, eng);
                self.pending_sections.set();
            }
            // Free GPU meshes of edited sections so they re-extract from the
            // updated generator overlay (edits land every frame).
            self.remesh_dirty_sections(eng);
            lane_enqueue::<SectionLane>(
                self,
                center_chunk,
                pipeline::Deadline::from_budget(pipeline::LOD_ENQUEUE_BUDGET),
            );
            self.rebuild_section_visible(center_chunk);
        }
        // Occlusion is derived state, rebuilt here at the `&mut` sync point (never
        // in the `&self` render) — and only when the adaptive gate is active AND
        // an input changed (or it was just activated). When the gate is off this
        // is skipped entirely and render draws everything, so a CPU-bound world
        // pays nothing for occlusion.
        let occlusion_on = self.occlusion_enabled();
        if occlusion_on && (self.occlusion_dirty.take() || !self.occlusion_active) {
            let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::StreamOcclusion);
            self.rebuild_occlusion(center_chunk);
        }
        self.occlusion_active = occlusion_on;
        #[cfg(debug_assertions)]
        self.debug_assert_liveness();
    }

    /// Land finished worker results (non-blocking). Generate results clear `generating`.
    /// Stale results drop and re-arm fresh scan. Budgeted mesh upload to GPU.
    fn drain_results(&mut self, eng: &mut Engine, light_apply: pipeline::Deadline) {
        if let Some(workers) = &self.workers {
            while let Some(done) = workers.try_recv() {
                self.done_scratch.push(done);
            }
        }
        if self.done_scratch.is_empty()
            && self.upload_queue.is_empty()
            && self.section_upload_queue.is_empty()
            && self.light_apply_queue.is_empty()
        {
            return;
        }
        // Process outside the drain loop (the borrow checker aside, accepting
        // a result mutates half the world); the swap keeps the capacity. Each
        // finished result routes through its lane's `integrate` (light/mesh/
        // section) or, for the carved-out generation path, `accept_column`.
        let mut done = std::mem::take(&mut self.done_scratch);
        for result in done.drain(..) {
            match result {
                pipeline::Done::Column { col, chunks } => self.accept_column(col, chunks),
                m @ pipeline::Done::Mesh { .. } => lane_integrate::<MeshLane>(self, m),
                l @ pipeline::Done::Light { .. } => lane_integrate::<LightLane>(self, l),
                sc @ pipeline::Done::Section { .. } => lane_integrate::<SectionLane>(self, sc),
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
            // Charge the budget per CONSUMED entry, whether it uploads or is
            // dropped as stale — otherwise a queue full of stale entries drains
            // entirely in one frame, defeating UPLOAD_BUDGET.
            uploads += 1;
            if !self.mesh_result_applies(coord, rev) {
                // Went stale while queued (an edit turned it `Dirty`, or it left
                // the box). Release the in-flight claim so a still-`NeedsMesh`
                // chunk falls back to `{ building: false }` and re-meshes when
                // re-seeded — the wedge fix. A no-op if an edit already moved it
                // to `Dirty` (the sync remesh owns it then).
                if let Some(loaded) = self.chunks.get_mut(&coord) {
                    loaded.state.release_build();
                }
                self.pending_fresh.set();
                self.mesh_worklist.insert(coord);
                continue;
            }
            // Upload each present pass; an empty pass yields no handle. Both
            // passes of a chunk upload together under one budget charge — they
            // share the chunk's fate (same rev), so never show half its geometry.
            let handles = ByPass::from_fn(|p| eng.upload_mesh(&data[p]));
            // Light is decoupled — it was settled and published before this mesh
            // was ever enqueued, so the upload is a pure GPU handoff.
            if let Some(loaded) = self.chunks.get_mut(&coord) {
                // The rev referee above guarantees this chunk is still
                // `NeedsMesh { building: true }` (an edit would have bumped rev →
                // dropped), so it owns no handle — but route through `retire`
                // anyway: it frees any stray token for free (and clears the
                // claim as the state moves to `Ready`/`Air`), keeping the async
                // path correct by construction rather than by assertion.
                loaded.retire(MeshState::from_upload(handles), eng);
            }
        }

        // Budgeted light application — the drain lane that previously had none.
        // A worker flood lands here as a grid; applying it (publish + border-diff
        // + mesh-invalidate) is cheap but ~80 could land at once on entry, so it
        // is capped. The chunk stays in `light_inflight` (so `light_ready` still
        // blocks meshing) until it is actually applied here. Order-independent:
        // each grid is absolute, so leftovers apply next frame with no seam.
        // Time-budgeted (checked between grids): an admitted grid always
        // publishes; leftovers apply next frame (each grid is absolute, no seam).
        while !light_apply.expired() {
            let Some((coord, grid)) = self.light_apply_queue.pop_front() else { break };
            self.settle_light(coord, grid);
        }

        // Section uploads, on their own budget. Dropped if unloaded since landing (no rev tracking).
        let mut section_uploads = 0;
        while section_uploads < SECTION_UPLOAD_BUDGET {
            let Some((pos, meshes)) = self.section_upload_queue.pop_front() else { break };
            section_uploads += 1;
            if let Some(state @ SectionState::Meshing) = self.sections.get_mut(&pos) {
                *state = SectionState::from_upload(pos, meshes, eng);
            }
        }
    }

    /// Generated chunk result: discard if out-of-range/already loaded; else store (replays edits).
    pub(in crate::world) fn accept_chunk(&mut self, coord: Coord, chunk: Chunk) {
        let Some(center) = self.center else { return };
        if !self.data_box(center).contains(coord) || self.chunks.contains_key(&coord) {
            return;
        }
        self.store_chunk(coord, chunk);
    }

    /// Mesh result at `rev`: queue for upload if still applies; else drop and re-arm scan.
    pub(in crate::world) fn accept_mesh(&mut self, coord: Coord, rev: u32, data: ChunkMeshData) {
        if self.mesh_result_applies(coord, rev) {
            self.upload_queue.push_back((coord, rev, data));
        } else {
            // Stale: the chunk was edited (now `Dirty`, sync-remesh territory)
            // or left the box. Release the in-flight claim so a still-`NeedsMesh`
            // coord drops to `{ building: false }` and can be picked up again
            // once re-seeded (a no-op if it is already `Dirty`).
            if let Some(loaded) = self.chunks.get_mut(&coord) {
                loaded.state.release_build();
            }
            self.pending_fresh.set();
            self.mesh_worklist.insert(coord);
        }
    }

    /// Mesh result still applies: chunk loaded, in view range, rev not bumped.
    pub(in crate::world) fn mesh_result_applies(&self, coord: Coord, rev: u32) -> bool {
        self.in_mesh_box(coord) && self.chunks.get(&coord).is_some_and(|l| l.rev == rev)
    }

    /// Streaming priority: chessboard distance with vertical axis weighted 2x (terrain before sky).
    pub(in crate::world) fn order(a: Coord, b: Coord) -> i32 {
        a.ring(b).max(2 * a.updown(b))
    }

    /// Queue generation for missing chunks in the data box, grouped into vertical
    /// columns so the `cy`-invariant column profile is sampled once per column
    /// instead of once per chunk. Nearest column first; the centre is generated
    /// synchronously first for collision safety.
    fn request_region_data(&mut self, center: Coord) {
        self.ensure_data(center);
        // Collect missing chunks grouped by horizontal column, tracking each
        // column's cy extent so one job covers the whole vertical run.
        let mut columns: super::FastMap<(i32, i32), (i32, i32)> = super::FastMap::default();
        for coord in self.data_box(center).coords() {
            if self.chunks.contains_key(&coord) || self.generating.contains(&coord) {
                continue;
            }
            let entry = columns.entry((coord.x, coord.z)).or_insert((coord.y, coord.y));
            entry.0 = entry.0.min(coord.y);
            entry.1 = entry.1.max(coord.y);
        }
        if columns.is_empty() {
            return;
        }
        // Nearest column first (horizontal ring distance to the centre column).
        let mut cols: Vec<((i32, i32), (i32, i32))> = columns.into_iter().collect();
        cols.sort_by_key(|&((cx, cz), _)| (cx - center.x).abs().max((cz - center.z).abs()));
        let workers = self
            .workers
            .get_or_insert_with(|| pipeline::Workers::spawn(pipeline::Workers::default_threads()));
        for ((cx, cz), (cy_lo, cy_hi)) in cols {
            // Every chunk-coord in the column's span carries its edit overlay, so
            // a landing chunk replays its edits on the worker thread.
            let edits: Vec<(Coord, Vec<(usize, crate::block::registry::BlockId)>)> = (cy_lo..=cy_hi)
                .filter_map(|cy| {
                    let coord = ChunkCoord::new(cx, cy, cz);
                    self.edits
                        .get(&coord)
                        .map(|cells| (coord, cells.iter().map(|(&i, &id)| (i, id)).collect()))
                })
                .collect();
            let accepted = workers.submit(pipeline::Job::GenerateColumn {
                col: (cx, cz),
                cy: cy_lo..=cy_hi,
                generator: self.generator.clone(),
                edits,
            });
            if accepted {
                // Claim every missing coord in the span so it isn't re-requested.
                for cy in cy_lo..=cy_hi {
                    let coord = ChunkCoord::new(cx, cy, cz);
                    if !self.chunks.contains_key(&coord) {
                        self.generating.insert(coord);
                    }
                }
            }
        }
    }

    /// Land a generated column: register each not-yet-loaded, in-range chunk
    /// (edits already replayed on the worker) and clear its generate claim.
    pub(in crate::world) fn accept_column(&mut self, _col: (i32, i32), chunks: Vec<(Coord, Chunk)>) {
        for (coord, chunk) in chunks {
            self.generating.remove(&coord);
            self.accept_chunk(coord, chunk);
        }
    }

    /// Ensure every chunk within the data box of `center` exists (voxel data
    /// only). Cheap and GPU-free, so it also seeds headless queries.
    pub(in crate::world) fn ensure_region_data(&mut self, center: Coord) {
        for coord in self.data_box(center).coords() {
            self.ensure_data(coord);
        }
    }

    /// Sync-generate small box around spawn position (collision safety before async catches up).
    pub fn prepare_around(&mut self, pos: DVec3) {
        let c = Self::chunk_of(block_coord(pos.x), block_coord(pos.y), block_coord(pos.z));
        for cx in (c.x - 1)..=(c.x + 1) {
            for cz in (c.z - 1)..=(c.z + 1) {
                for cy in (c.y - 2)..=(c.y + 1) {
                    self.ensure_data(ChunkCoord::new(cx, cy, cz));
                }
            }
        }
    }

    /// Generate a chunk's data if it isn't loaded, replaying any saved edits on it.
    pub(in crate::world) fn ensure_data(&mut self, coord: Coord) {
        if self.chunks.contains_key(&coord) {
            return;
        }
        let chunk = Chunk::new(coord.x, coord.y, coord.z, &self.generator);
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
        // Born-air chunks can never produce geometry, so they start `Air` (no
        // worker job, nothing drawn); dense chunks await the fresh scan. A
        // uniform chunk of ANY non-solid block (air is the only one today) is
        // equally empty of geometry — key off solidity, not the AIR id, so a
        // future transparent/non-solid block is handled without a hole.
        // (Computed before the insert so `registry` and `chunks` don't clash.)
        let born_air = chunk.uniform().is_some_and(|id| !self.registry.is_solid(id));
        let state =
            if born_air { MeshState::Air } else { MeshState::NeedsMesh { building: false } };
        // Connectivity is computed lazily by the occlusion rebuild (only if the
        // gate is active), so generation pays no flood-fill when occlusion is off.
        let chunk = std::sync::Arc::new(chunk);
        // Liveness invariant, checked O(1) at the sole point a coord enters
        // `chunks`: a stored coord must not still be claimed in `generating`
        // (a stuck generate claim shadowing live data). Replaces the old
        // per-accept full scan of `generating`, which was O(|generating|) per
        // chunk — quadratic over a world-entry drain burst (debug builds only).
        debug_assert!(
            !self.generating.contains(&coord),
            "storing {coord:?} still claimed in generating — a stuck generate claim"
        );
        self.chunks.insert(coord, Loaded { chunk: std::sync::Arc::clone(&chunk), state, rev: 0, connectivity: None, light: None });
        // Light: try the analytic fast path first — a uniform-opaque chunk settles
        // to all-dark and an above-surface uniform-air chunk to full sky with no
        // flood. A trivial grid publishes synchronously (which fans the border to
        // its neighbours); only the residual Dense band seeds the settle worklist.
        if self.lighting {
            match self.trivial_light(coord, &chunk) {
                Some(grid) => self.settle_light(coord, grid),
                None => {
                    self.light_worklist.insert(coord);
                    self.light_pending.set();
                }
            }
        }
        // A new chunk changes what the BFS can reach.
        self.occlusion_dirty.set();
        // Fresh mesh work: seed this chunk and its 6 neighbours into the mesh
        // lane's worklist (a neighbour may have been blocked waiting on this
        // chunk's data even when it is itself uniform air). O(shell), not a
        // whole-map rescan.
        self.mesh_worklist.insert(coord);
        for face in Face::ALL {
            self.mesh_worklist.insert(coord.step(face));
        }
        self.pending_fresh.set();
    }

    /// Free chunks past the unload box, releasing their GPU meshes.
    fn unload_far(&mut self, center: Coord, eng: &mut Engine) {
        let unload = self.unload_box(center);
        // Collect-then-remove instead of `retain`: freeing needs `&mut eng`,
        // which can't be borrowed inside a retain closure over `self.chunks`.
        let far: Vec<Coord> = self
            .chunks
            .keys()
            .copied()
            .filter(|&coord| !unload.contains(coord))
            .collect();
        // A removed chunk changes what the BFS can reach.
        self.occlusion_dirty.raise(!far.is_empty());
        for coord in far {
            // Free whatever mesh the chunk was drawing (Ready, or an edited
            // Dirty still showing its old mesh); Air/NeedsMesh own none.
            // The removed `Loaded` owns its token, so `free_owned` consumes it
            // to be freed exactly once as the entry is discarded.
            if let Some(loaded) = self.chunks.remove(&coord) {
                loaded.state.free_owned(eng);
            }
        }
        // Drop cached ceilings for columns that no longer have any loaded chunk;
        // the heightmap is pure, so a re-entered column simply recomputes once.
        if !self.ceilings.is_empty() {
            let live: FastSet<(i32, i32)> =
                self.chunks.keys().map(|c| (c.x, c.z)).collect();
            self.ceilings.retain(|col, _| live.contains(col));
        }
    }

    /// Remesh edited (`Dirty`) chunks synchronously, budgeted, nearest first —
    /// carved out from the async mesh lane so a broken block never lags a frame.
    /// The fresh-mesh half is now the [`MeshLane`] worklist lane.
    fn remesh_dirty(&mut self, center: Coord, eng: &mut Engine) {
        // The `Dirty` fiber of `MeshState`, materialised into a scratch Vec (the
        // pass mutates each chunk via `mesh_chunk`, so it can't hold the filter
        // borrow). Gated by `pending_dirty` so an idle frame does no scan.
        if !self.pending_dirty.take() {
            return;
        }
        let mut dirty: Vec<Coord> = self
            .chunks
            .iter()
            .filter(|(_, l)| l.state.is_dirty())
            .map(|(&coord, _)| coord)
            .collect();
        dirty.sort_by_key(|&coord| Self::order(coord, center));
        // Leftovers past the budget stay `Dirty` (still in the fiber); re-arm
        // the hint so the next frame drains them.
        if dirty.len() > DIRTY_BUDGET {
            self.pending_dirty.set();
        }
        for coord in dirty.into_iter().take(DIRTY_BUDGET) {
            // No neighbour-data gate here: an edited chunk must remesh even when
            // a far neighbour has no data (the mesher reads missing neighbours as
            // air). The coord came from `chunks`, so it is still present.
            self.mesh_chunk(coord, eng);
        }
    }

    /// Snapshot for mesh job: chunk storage, neighbour shell, solidity table, and rev.
    pub(in crate::world) fn snapshot(
        &self,
        coord: Coord,
        degraded: bool,
    ) -> (u32, pipeline::ChunkSnapshot) {
        let loaded = &self.chunks[&coord];
        (
            loaded.rev,
            pipeline::ChunkSnapshot {
                padded: self.capture_padded(coord),
                uniform: loaded.chunk.uniform(),
                light: self.capture_padded_light(coord, degraded),
                tables: self.tables.get(),
            },
        )
    }

    /// Settled light shell for chunk and 26 neighbours (18³). A missing grid reads
    /// dark for a normal mesh; for a `degraded` mesh it stands in as fully-lit
    /// open-sky, so an unsettled neighbourhood fails toward visible-and-plausible.
    fn capture_padded_light(&self, coord: Coord, degraded: bool) -> light::PaddedLight {
        if !self.lighting {
            return light::PaddedLight::full();
        }
        let fallback = degraded.then(light::LightGrid::open_sky);
        light::PaddedLight::capture(|dx, dy, dz| {
            self.chunks
                .get(&Coord::new(coord.x + dx, coord.y + dy, coord.z + dz))
                .and_then(|l| l.light.as_ref())
                .or(fallback.as_ref())
        })
    }

    /// Near-face light from six neighbours (input for light flood seeds).
    pub(in crate::world) fn capture_face_shell(&self, coord: Coord) -> light::FaceShell {
        if !self.lighting {
            return light::FaceShell::dark();
        }
        light::FaceShell::capture(|face| {
            self.chunks.get(&coord.step(face)).and_then(|l| l.light.as_ref())
        })
    }

    /// Skylight ceiling: surface height per column (pure generator fn, caves dark
    /// consistently). Keyed by `(x, z)` chunk column — the surface heightmap is
    /// independent of `y` and of edits, so it is computed once per column and
    /// shared across every vertical chunk and every re-settle. `capture_ceiling`
    /// therefore samples 256 noise columns *once per column ever*, not per settle.
    pub(in crate::world) fn capture_ceiling(&mut self, coord: Coord) -> light::CeilingWindow {
        if let Some(ceiling) = self.ceilings.get(&(coord.x, coord.z)) {
            return ceiling.clone();
        }
        let x0 = coord.x * CHUNK_SIZE as i32;
        let z0 = coord.z * CHUNK_SIZE as i32;
        let generator = &self.generator;
        let ceiling = light::CeilingWindow::from_heights(|lx, lz| {
            generator.height(x0 + lx as i32, z0 + lz as i32)
        });
        self.ceilings.insert((coord.x, coord.z), ceiling.clone());
        ceiling
    }

    /// The analytic light grid for a chunk whose settled light is provable
    /// without a flood, or `None` if it must go through the worker settle. The
    /// two trivial cases collapse the load-time light-job burst to the thin
    /// Dense surface band (see [`store_chunk`](Self::store_chunk)):
    /// - a uniform-*opaque* chunk settles to all-dark (no light enters);
    /// - a uniform-*air* chunk fully above every column's surface, with no near
    ///   blocklight from a loaded neighbour, settles to full sky / dark block.
    ///
    /// Correctness anchor: the returned grid equals `propagate(uniform, dark
    /// shell, ceiling, world_y0, tables)`.
    ///
    /// `&mut self` so it can warm the `ceilings` column cache and the hot tables
    /// while probing — it mutates no lane state.
    fn trivial_light(&mut self, coord: Coord, chunk: &Chunk) -> Option<light::LightGrid> {
        if !self.lighting {
            return None;
        }
        self.refresh_tables();
        let tables = self.tables.get();
        // A full block of opaque rock settles to all-dark: no skylight column
        // stays open through it and no neighbour light can relax into an opaque
        // cell — so this holds regardless of neighbours (dark unconditionally).
        if chunk.is_uniform_opaque(&tables) {
            return Some(light::LightGrid::dark());
        }
        // A uniform-air chunk that sits fully above every column's surface is
        // full sky — *if* no loaded neighbour has near-border blocklight that
        // would bleed in (skylight can't exceed FULL, so only blocklight breaks
        // the analytic result). `propagate` with a dark shell yields exactly
        // `open_sky()` here; the neighbour check is what makes the dark shell sound.
        if chunk.uniform() == Some(crate::block::registry::AIR) {
            let world_y0 = coord.y * CHUNK_SIZE as i32;
            let ceiling = self.capture_ceiling(coord);
            let all_open = (0..CHUNK_SIZE)
                .all(|lz| (0..CHUNK_SIZE).all(|lx| ceiling.open_above(lx, lz, world_y0)));
            if all_open && !self.neighbour_blocklight_near(coord) {
                return Some(light::LightGrid::open_sky());
            }
        }
        None
    }

    /// Whether any loaded face-neighbour carries near-border blocklight `> 1`
    /// (light level 1 attenuates to 0 crossing in, so it can't seed). Used by
    /// [`trivial_light`](Self::trivial_light) to reject the dark-shell fast path
    /// when a torch next door would actually bleed across the border.
    fn neighbour_blocklight_near(&self, coord: Coord) -> bool {
        let shell = self.capture_face_shell(coord);
        Face::ALL.iter().any(|&face| {
            (0..CHUNK_SIZE).any(|b| (0..CHUNK_SIZE).any(|a| shell.at(face, a, b).block.get() > 1))
        })
    }

    /// The single terminal light transition: release the in-flight claim, publish
    /// the settled grid, and re-arm this chunk's mesh readiness — atomically, so no
    /// caller can perform a PARTIAL settle (the identical-grid stall class).
    ///
    /// Diffs the grid against the previously published one to find which shared
    /// borders moved (→ the neighbours to re-settle) and whether anything changed
    /// (→ this chunk's own mesh, if any, is stale). The cheap main-thread
    /// bookkeeping half of a settle, shared by the sync (trivial) and async
    /// (drained) paths.
    pub(in crate::world) fn settle_light(&mut self, coord: Coord, grid: light::LightGrid) {
        // Release the claim FIRST. Harmless no-op on the trivial sync path (which
        // never entered `light_inflight`); on the async path it absorbs the removal
        // that used to sit at the drain call site, so "settled" and "still in
        // flight" can never diverge for an observer between two statements.
        // `FastSet::remove` on an absent key returns `false` harmlessly.
        self.light_inflight.remove(&coord);
        // Unloaded while the flood flew (or before a trivial publish): drop it.
        if !self.chunks.contains_key(&coord) {
            return;
        }
        let (self_changed, moved): (bool, Vec<Face>) = match &self.chunks[&coord].light {
            None => (true, Face::ALL.to_vec()),
            Some(old) => (
                *old != grid,
                Face::ALL.into_iter().filter(|&f| light::border_changed(old, &grid, f)).collect(),
            ),
        };
        self.chunks.get_mut(&coord).unwrap().light = Some(grid);
        // Re-arm THIS chunk's mesh readiness UNCONDITIONALLY — and before the
        // `self_changed` early-return below. Reaching here means the chunk just
        // left `light_worklist`/`light_inflight` (its caller cleared the claim),
        // which flips its `light_ready` gate even when the settled grid is
        // byte-identical to the old one. The mesh lane evicts light-blocked
        // seeds on the contract that publish RE-SEEDS them; a fixpoint re-settle
        // that skipped this seed stranded the chunk off the worklist forever
        // (`ready` but unreachable — the golden idle stall). The early-return
        // gates only the light-VALUE-driven work (neighbour propagation and the
        // stale-self-mesh invalidation), never this readiness re-arm.
        self.pending_fresh.set();
        self.mesh_worklist.insert(coord);
        if !self_changed {
            return;
        }
        // A neighbour may now be meshable too (this chunk's FIRST grid completes
        // their neighbourhood — `moved` is all faces then); re-settle the
        // neighbours whose shared border moved.
        for face in &moved {
            let n = coord.step(*face);
            self.light_worklist.insert(n);
            self.mesh_worklist.insert(n);
            // A DEGRADED neighbour meshed with fake open-sky light across this
            // border; now that real light has crossed it, force its remesh through
            // the Dirty machinery — seeding the worklist alone can't, since the
            // neighbour is already `Ready` and so fails the mesh lane's
            // `is_needs_mesh` gate (remesh-on-arrival).
            if self.light_gate.degraded.contains(&n) {
                if let Some(loaded) = self.chunks.get_mut(&n) {
                    if matches!(
                        loaded.state,
                        MeshState::Ready(_) | MeshState::NeedsMesh { building: true }
                    ) {
                        loaded.state.invalidate();
                        loaded.rev = loaded.rev.wrapping_add(1);
                        self.pending_dirty.set();
                    }
                }
            }
        }
        if !self.light_worklist.is_empty() {
            self.light_pending.set();
        }
        // My light changed and I already show (or am building) a mesh → that mesh
        // is stale. Reuse the edit invalidation: keep the old mesh drawn, bump rev
        // to strand any in-flight build, re-mesh with the new light. A not-yet-meshed,
        // not-yet-building chunk (`NeedsMesh { building: false }`/`Air`) needs
        // nothing here — it will mesh fresh against the new light.
        let loaded = self.chunks.get_mut(&coord).unwrap();
        if matches!(
            loaded.state,
            MeshState::Ready(_) | MeshState::Dirty { .. } | MeshState::NeedsMesh { building: true }
        ) {
            loaded.state.invalidate();
            loaded.rev = loaded.rev.wrapping_add(1);
            self.pending_dirty.set();
        }
    }

    /// Chunk + 1-voxel neighbour shell for mesh build (shared by worker and sync paths).
    fn capture_padded(&self, coord: Coord) -> mesh::Padded {
        mesh::Padded::capture(|dx, dy, dz| {
            self.chunks
                .get(&Coord::new(coord.x + dx, coord.y + dy, coord.z + dz))
                .map(|l| &*l.chunk)
        })
    }

    // --- Column-LOD sections -----------------------------------------

    /// The desired quadtree frontier around the player's XZ position.
    pub(in crate::world) fn desired_sections(&self, center: Coord) -> Vec<SectionPos> {
        let cs = CHUNK_SIZE as i32;
        let (pcx, pcz) = (center.x * cs + cs / 2, center.z * cs + cs / 2);
        quadtree::desired_sections(pcx, pcz, &self.section_pyramid)
    }

    /// True if the cell or a Ready ancestor covers it.
    pub(in crate::world) fn section_covered(&self, cell: SectionPos) -> bool {
        let max = self.section_pyramid.coarsest();
        let ready = |p: SectionPos| self.sections.get(&p).is_some_and(|s| s.is_ready());
        quadtree::drawable_cover(cell, max, &ready).is_some()
    }

    /// Edits affecting this section: chunks within its footprint and height domain.
    /// Used when re-extracting after an edit.
    pub(in crate::world) fn edits_for_section(
        &self,
        pos: SectionPos,
    ) -> Vec<(Coord, Vec<(usize, crate::block::registry::BlockId)>)> {
        let cs = CHUNK_SIZE as i32;
        let span = pos.span();
        let (cx0, cz0) = (pos.min_x().div_euclid(cs), pos.min_z().div_euclid(cs));
        let cn = span / cs; // chunk columns per section side
        let cy_hi = super::section::DOMAIN_H / cs; // vertical chunk-layer count
        self.edits
            .iter()
            .filter(|(c, _)| {
                (cx0..cx0 + cn).contains(&c.x)
                    && (cz0..cz0 + cn).contains(&c.z)
                    && (0..cy_hi).contains(&c.y)
            })
            .map(|(&c, cells)| (c, cells.iter().map(|(&i, &b)| (i, b)).collect()))
            .collect()
    }

    /// Rebuild the visible set every frame; as sections become Ready, the covering
    /// changes and stale entries would draw incorrectly.
    fn rebuild_section_visible(&mut self, center: Coord) {
        let desired = self.desired_sections(center);
        let max = self.section_pyramid.coarsest();
        let ready = |p: SectionPos| self.sections.get(&p).is_some_and(|s| s.is_ready());
        let visible = quadtree::resolve_covering(&desired, max, &ready);
        self.section_visible = visible;
    }

    /// Unload sections outside desired, visible, and hysteresis bands (boundary cross).
    /// Hysteresis prevents thrashing at view edges.
    fn unload_sections(&mut self, center: Coord, eng: &mut Engine) {
        let desired: FastSet<SectionPos> = self.desired_sections(center).into_iter().collect();
        let visible: FastSet<SectionPos> = self.section_visible.iter().copied().collect();
        let cs = CHUNK_SIZE as i32;
        let (pcx, pcz) = (center.x * cs + cs / 2, center.z * cs + cs / 2);
        let cfg = &self.section_pyramid;
        let stale: Vec<SectionPos> = self
            .sections
            .keys()
            .copied()
            .filter(|s| {
                if desired.contains(s) || visible.contains(s) {
                    return false;
                }
                let span = s.span();
                let (cx, cz) = (s.x * span + span / 2, s.z * span + span / 2);
                let dist = ((cx - pcx) as f32).hypot((cz - pcz) as f32);
                !pyramid::acceptable(dist, super::lod::Lod(s.detail), cfg)
            })
            .collect();
        for s in stale {
            if let Some(state) = self.sections.remove(&s) {
                state.free(eng);
            }
        }
    }

    /// Free GPU meshes so edited sections re-extract from the updated overlay.
    fn remesh_dirty_sections(&mut self, eng: &mut Engine) {
        if self.dirty_sections.is_empty() {
            return;
        }
        let dirty: Vec<SectionPos> = self.dirty_sections.iter().copied().collect();
        let mut freed = false;
        for s in dirty {
            match self.sections.get(&s) {
                Some(SectionState::Ready { .. }) => {
                    if let Some(state) = self.sections.remove(&s) {
                        state.free(eng);
                    }
                    self.dirty_sections.remove(&s);
                    freed = true;
                }
                Some(SectionState::Meshing) => {} // in flight: free once it lands Ready
                None => {
                    self.dirty_sections.remove(&s);
                }
            }
        }
        if freed {
            self.pending_sections.set();
        }
    }

    /// Six orthogonal neighbours have data loaded.
    pub(in crate::world) fn neighbours_have_data(&self, coord: Coord) -> bool {
        Face::ALL.iter().all(|&f| self.chunks.contains_key(&coord.step(f)))
    }

    /// Light settled enough to mesh: chunk and face neighbours have grids, and the
    /// chunk is neither seeded nor being settled (in-flight counts as not-yet-final,
    /// so a chunk never meshes against a flood that's still running for it).
    pub(in crate::world) fn light_ready(&self, coord: Coord) -> bool {
        if !self.lighting {
            // Nothing to settle: gate meshing on data alone (checked separately).
            return self.chunks.contains_key(&coord);
        }
        !self.light_worklist.contains(&coord)
            && !self.light_inflight.contains(&coord)
            && self.chunks.get(&coord).is_some_and(|l| l.light.is_some())
            && Face::ALL.iter().all(|&f| {
                self.chunks.get(&coord.step(f)).is_some_and(|l| l.light.is_some())
            })
    }

    /// A chunk waiting purely on neighbour light: it has data and is in view and
    /// awaiting a fresh mesh, but its neighbourhood light has not settled. The
    /// [`LightGate`] times exactly these chunks.
    fn chunk_light_blocked(&self, coord: Coord) -> bool {
        self.is_needs_mesh(coord)
            && self.in_mesh_box(coord)
            && self.neighbours_have_data(coord)
            && !self.light_ready(coord)
    }

    /// Whether `coord` has waited on neighbour light past [`LIGHT_WAIT_DEGRADE`] —
    /// the mesh-lane predicate that admits a DEGRADED mesh.
    pub(in crate::world) fn light_wait_expired(&self, coord: Coord) -> bool {
        self.light_gate.blocked_since.get(&coord).is_some_and(|t| t.elapsed() >= LIGHT_WAIT_DEGRADE)
    }

    /// Record (or clear) that `coord` is currently drawing a degraded, known-not-
    /// final mesh. The set is queryable by [`entry_complete`](Self::entry_complete)
    /// ("none pending").
    pub(in crate::world) fn mark_degraded(&mut self, coord: Coord, degraded: bool) {
        if degraded {
            self.light_gate.degraded.insert(coord);
        } else {
            self.light_gate.degraded.remove(&coord);
        }
    }

    /// Advance the light-gate before the mesh lane runs: start a timer for
    /// each newly light-blocked mesh candidate, reap timers whose chunk stopped
    /// waiting, drop degraded entries for unloaded chunks, and re-seed still-waiting
    /// chunks onto the mesh worklist so a wait-time degrade (which raises no re-seed
    /// event of its own) is never stranded by worklist eviction.
    fn tick_light_gate(&mut self) {
        // `LightGate` is `Default`, so move it out to break the self-borrow while
        // the predicates below read the chunk map.
        let mut gate = std::mem::take(&mut self.light_gate);
        gate.degraded.retain(|c| self.chunks.contains_key(c));
        gate.blocked_since.retain(|c, _| self.chunk_light_blocked(*c));
        // Remesh-on-arrival safety net: a DEGRADED chunk clears only when it is
        // re-invalidated after its light settles. The event-driven path
        // (`settle_light` border-move) MISSES a degraded chunk whose neighbour's
        // FINAL light publish doesn't move their shared border — it would then stay
        // degraded forever though `light_ready` is now true (the world-entry stall at
        // `degraded=N`). So sweep the bounded, shrinking set: any entry now light-ready
        // and still drawing a `Ready` mesh is invalidated to remesh, which clears its
        // flag in `mesh_chunk`. One remesh per chunk, so it converges (no re-arm once
        // out of the set).
        let relit: Vec<Coord> =
            gate.degraded.iter().copied().filter(|&c| self.light_ready(c)).collect();
        for c in relit {
            if let Some(loaded) = self.chunks.get_mut(&c) {
                if matches!(loaded.state, MeshState::Ready(_)) {
                    loaded.state.invalidate();
                    loaded.rev = loaded.rev.wrapping_add(1);
                    self.pending_dirty.set();
                }
            }
        }
        let now = Instant::now();
        let fresh: Vec<Coord> = self
            .mesh_worklist
            .iter()
            .copied()
            .filter(|c| self.chunk_light_blocked(*c) && !gate.blocked_since.contains_key(c))
            .collect();
        for c in fresh {
            gate.blocked_since.insert(c, now);
        }
        if !gate.blocked_since.is_empty() {
            for &c in gate.blocked_since.keys() {
                self.mesh_worklist.insert(c);
            }
            self.pending_fresh.set();
        }
        self.light_gate = gate;
    }

    /// Forward-progress floor for the degraded set. The edge-triggered
    /// clear (`settle_light` → invalidate-on-arrival, and the `tick_light_gate`
    /// remesh-on-arrival sweep) only fires while light is still *moving*: a
    /// degraded chunk whose missing neighbour has already reached its TERMINAL
    /// light state (unloaded, or settled with a border that never moved) gets no
    /// further arrival, so it stays degraded forever and `entry_complete` (which
    /// requires an empty degraded set) hangs.
    ///
    /// This is the level-triggered backstop. It fires ONLY at true light
    /// quiescence — no generate, mesh, or light work of any kind outstanding — at
    /// which point every remaining degraded chunk's neighbourhood is provably
    /// final, so re-meshing against the REAL current light (missing planes read
    /// dark, which is the correct terminal input for a never-lit neighbour — NOT
    /// the degraded open-sky fallback) is strictly more correct than the mesh it
    /// currently draws. Each chunk is promoted to a FINAL mesh and dropped from
    /// the set, so the pass runs once and `degraded` drains to empty. It cannot
    /// fire early (the AND-guard) and cannot re-degrade its own output
    /// (`remesh_terminal` marks final unconditionally).
    fn flush_degraded_terminal(&mut self, eng: &mut Engine) {
        let quiescent = self.generating.is_empty()
            && self.mesh_worklist.is_empty()
            && self.light_worklist.is_empty()
            && self.light_inflight.is_empty()
            && self.light_apply_queue.is_empty()
            && !self.light_gate.degraded.is_empty();
        if !quiescent {
            return;
        }
        // Promote SETTLED degraded chunks, a few per frame: `remesh_terminal`
        // is a synchronous main-thread mesh build (milliseconds each), so an
        // unbudgeted pass over N stuck chunks would be one big hitch. The world
        // is quiescent here (nothing else re-degrades), so the set drains
        // monotonically across frames either way; a still-building/Dirty chunk
        // is left for a later flush once its own path settles it.
        const TERMINAL_FLUSH_BUDGET: usize = 2;
        let stuck: Vec<Coord> = self.light_gate.degraded.iter().copied().collect();
        let mut promoted = 0usize;
        for coord in stuck {
            if promoted >= TERMINAL_FLUSH_BUDGET {
                break;
            }
            match self.chunks.get(&coord).map(|l| &l.state) {
                // Settled on a degraded mesh — the stuck case. Promote to final.
                Some(MeshState::Ready(_) | MeshState::Air) => {
                    self.remesh_terminal(coord, eng);
                    promoted += 1;
                }
                // Unloaded out from under the set between marking and here.
                None => self.mark_degraded(coord, false),
                // Still building (in-flight degraded result pending) or Dirty (a
                // sync remesh owns it): another path is about to resolve it. Leave
                // it in the set; a later frame's flush promotes it once settled, so
                // the flush never races an in-flight upload for the same chunk.
                Some(MeshState::NeedsMesh { .. } | MeshState::Dirty { .. }) => {}
            }
        }
    }

    /// Re-mesh a degraded chunk against its neighbours' REAL final light and mark
    /// it FINAL — the terminal-flush counterpart to the degraded mesh-lane path.
    /// Unlike [`mesh_chunk`](Self::mesh_chunk) (which recomputes `degraded` from
    /// `light_ready` and so would re-degrade a chunk with a terminally-missing
    /// neighbour), this forces `degraded = false`: at the quiescence the caller
    /// guarantees, a missing plane is a settled neighbour's real (possibly dark)
    /// light, so the mesh IS final. The chunk is `Ready`/`Air` here; `retire`
    /// frees its degraded mesh exactly once.
    fn remesh_terminal(&mut self, coord: Coord, eng: &mut Engine) {
        self.refresh_tables();
        let mut scratch = std::mem::replace(&mut self.scratch, mesh::new_chunk_mesh_data());
        let tables = self.tables.get();
        let uniform = self.chunks[&coord].chunk.uniform();
        let padded = self.capture_padded(coord);
        self.mark_degraded(coord, false);
        let light = self.capture_padded_light(coord, false);
        mesh::build_chunk_mesh(&padded, uniform, &tables, &light, &mut scratch);
        let handles = ByPass::from_fn(|p| eng.upload_mesh(&scratch[p]));
        self.scratch = scratch;
        if let Some(loaded) = self.chunks.get_mut(&coord) {
            loaded.retire(MeshState::from_upload(handles), eng);
        }
    }

    /// World-entry completeness predicate: true once, within the view
    /// radius, every chunk shows a FINAL-light mesh (`Ready`/`Air`, none degraded
    /// and none still waiting on light), no near generate/mesh/light work is queued
    /// or in flight, and (under lod2) the section far field is covering-complete.
    /// Reads private streaming state — its home here.
    pub fn entry_complete(&self) -> bool {
        let Some(center) = self.center else { return false };
        // No near work queued or in flight, and nothing owed a final-light remesh.
        if !self.generating.is_empty()
            || !self.mesh_worklist.is_empty()
            || !self.upload_queue.is_empty()
            || !self.light_worklist.is_empty()
            || !self.light_inflight.is_empty()
            || !self.light_apply_queue.is_empty()
            || !self.light_gate.degraded.is_empty()
            || !self.light_gate.blocked_since.is_empty()
        {
            return false;
        }
        // Every in-view chunk has a final mesh (data loaded, not building/dirty).
        for coord in self.mesh_box(center).coords() {
            match self.chunks.get(&coord).map(|l| &l.state) {
                Some(MeshState::Air | MeshState::Ready(_)) => {}
                _ => return false,
            }
        }
        // LOD2 far field: all desired cells covered and no uploads pending. Skipped
        // when disabled (no far field in near-only mode).
        if self.lod2 {
            if !self.section_upload_queue.is_empty() {
                return false;
            }
            if self.desired_sections(center).into_iter().any(|c| !self.section_covered(c)) {
                return false;
            }
        }
        true
    }

    /// Every desired far-field section is itself Ready — the strongest far-field
    /// state. `entry_complete` accepts a Ready *ancestor* as covering (right for
    /// playability), but a coarse cover moves the horizon's pixels — and through
    /// the exposure meter, the whole frame's brightness — as refinement lands.
    /// The golden harness gates captures on this so blessed shots are the
    /// converged frame; gameplay never waits on it.
    pub fn far_field_refined(&self) -> bool {
        let Some(center) = self.center else { return false };
        if !self.lod2 {
            return true;
        }
        if !self.section_upload_queue.is_empty() {
            return false;
        }
        self.desired_sections(center)
            .into_iter()
            .all(|c| self.sections.get(&c).is_some_and(|s| s.is_ready()))
    }

    /// How many desired far-field sections still lack their own mesh — the
    /// harness's progress signal while it waits on
    /// [`far_field_refined`](Self::far_field_refined).
    pub fn far_field_pending(&self) -> usize {
        let Some(center) = self.center else { return 0 };
        if !self.lod2 {
            return 0;
        }
        self.desired_sections(center)
            .into_iter()
            .filter(|c| !self.sections.get(c).is_some_and(|s| s.is_ready()))
            .count()
    }

    /// Human-readable reason `entry_complete` is not yet true — the first
    /// unsatisfied clause with a count, so a stalled bless/harness run says WHICH
    /// streaming stage is stuck instead of hanging silently. Clause order mirrors
    /// [`entry_complete`](Self::entry_complete).
    pub fn entry_debug(&self) -> String {
        let Some(center) = self.center else { return "no stream centre yet".into() };
        let near: [(&str, usize); 8] = [
            ("generating", self.generating.len()),
            ("mesh_worklist", self.mesh_worklist.len()),
            ("upload_queue", self.upload_queue.len()),
            ("light_worklist", self.light_worklist.len()),
            ("light_inflight", self.light_inflight.len()),
            ("light_apply_queue", self.light_apply_queue.len()),
            ("degraded", self.light_gate.degraded.len()),
            ("light_blocked", self.light_gate.blocked_since.len()),
        ];
        let pending: Vec<String> =
            near.iter().filter(|(_, n)| *n != 0).map(|(k, n)| format!("{k}={n}")).collect();
        if !pending.is_empty() {
            let mut msg = format!("near work pending: {}", pending.join(", "));
            // If the fresh-mesh lane is the blocker, tally WHICH ready()-predicate the
            // stuck chunks fail — the four gates from `MeshLane::ready`.
            if !self.mesh_worklist.is_empty() {
                let (mut not_needs, mut out_box, mut no_neigh, mut lit_or_expired) = (0, 0, 0, 0);
                for &c in self.mesh_worklist.iter() {
                    if !self.is_needs_mesh(c) {
                        not_needs += 1;
                    } else if !self.in_mesh_box(c) {
                        out_box += 1;
                    } else if !self.neighbours_have_data(c) {
                        no_neigh += 1;
                    } else if self.light_ready(c) || self.light_wait_expired(c) {
                        lit_or_expired += 1;
                    }
                }
                msg.push_str(&format!(
                    " | mesh_worklist stuck-on: not_needs_mesh={not_needs} out_of_box={out_box} \
                     no_neighbour_data={no_neigh} ready_but_unclaimed={lit_or_expired} \
                     (lighting={})",
                    self.lighting
                ));
            }
            return msg;
        }
        // Terminal wedge fingerprint: all near-work queues are empty (the block
        // above returned nothing), yet some in-box chunk is neither `Air` nor
        // `Ready`. Break the residual down by state — and, for an idle
        // `NeedsMesh { building: false }` (the missed-enqueue suspect), by which
        // mesh-lane gate it would fail — so a stalled run names the exact wedge
        // instead of a bare count. `in_wl` is whether the coord is (wrongly, since
        // the worklist is drained) still tracked in `mesh_worklist`.
        let (mut missing, mut idle, mut building, mut dirty, mut queued) = (0, 0, 0, 0, 0);
        let (mut idle_no_neigh, mut idle_unlit) = (0, 0);
        for c in self.mesh_box(center).coords() {
            let in_wl = self.mesh_worklist.contains(&c);
            match self.chunks.get(&c).map(|l| &l.state) {
                Some(MeshState::Air | MeshState::Ready(_)) => {}
                None => missing += 1,
                Some(MeshState::Dirty { .. }) => dirty += 1,
                Some(MeshState::NeedsMesh { building: true }) => building += 1,
                Some(MeshState::NeedsMesh { building: false }) => {
                    if in_wl {
                        queued += 1;
                    } else {
                        idle += 1;
                        if !self.neighbours_have_data(c) {
                            idle_no_neigh += 1;
                        } else if !(self.light_ready(c) || self.light_wait_expired(c)) {
                            idle_unlit += 1;
                        }
                    }
                }
            }
        }
        let unmeshed = missing + idle + building + dirty + queued;
        if unmeshed != 0 {
            return format!(
                "chunks without a final mesh: {unmeshed} \
                 [missing={missing} idle={idle} building={building} dirty={dirty} \
                 queued_but_worklist_drained={queued}] \
                 idle stuck-on: no_neighbour_data={idle_no_neigh} unlit={idle_unlit}"
            );
        }
        if self.lod2 {
            if !self.section_upload_queue.is_empty() {
                return format!("section_upload_queue = {}", self.section_upload_queue.len());
            }
            let desired = self.desired_sections(center);
            let uncovered = desired.iter().filter(|&&c| !self.section_covered(c)).count();
            if uncovered != 0 {
                return format!(
                    "column sections uncovered: {uncovered} of {} desired",
                    desired.len()
                );
            }
        }
        "entry complete".into()
    }

    /// Build chunk GPU mesh (sync dirty-remesh). Frees old handle exactly once; all-air → Air.
    fn mesh_chunk(&mut self, coord: Coord, eng: &mut Engine) {
        self.refresh_tables();
        // Move the scratch out so the build can borrow `self.chunks` shared
        // (for cross-chunk neighbour culling) while filling it. `MeshData` has no
        // `Default` (it carries a `Pass`), so swap in a fresh opaque scratch
        // rather than `mem::take`; `build_chunk_mesh` clears it first anyway.
        let mut scratch = std::mem::replace(&mut self.scratch, mesh::new_chunk_mesh_data());
        let tables = self.tables.get();
        let uniform = self.chunks[&coord].chunk.uniform();
        let padded = self.capture_padded(coord);
        // Mesh with the CURRENTLY settled light (possibly stale after an edit):
        // geometry updates this frame for responsiveness, and the relit result
        // lands a frame or two later when the light lane reconverges and marks
        // this chunk dirty again — the visible light lag Minecraft also shows.
        // A dirty remesh runs against the currently-published light. If that light
        // is now final, the chunk is no longer degraded; if a neighbour is still
        // unsettled it stays degraded (missing planes read dark here — the sync
        // path keeps its stale-but-plausible behaviour). This is the remesh-on-
        // arrival that clears a chunk degraded by the mesh lane.
        let degraded = !self.light_ready(coord);
        self.mark_degraded(coord, degraded);
        let light = self.capture_padded_light(coord, degraded);
        mesh::build_chunk_mesh(&padded, uniform, &tables, &light, &mut scratch);
        let handles = ByPass::from_fn(|p| eng.upload_mesh(&scratch[p]));
        self.scratch = scratch;
        // `retire` frees the edited-Ready chunk's old mesh (`Dirty.prev`) exactly
        // once and installs the fresh `Ready`/`Air` state.
        if let Some(loaded) = self.chunks.get_mut(&coord) {
            debug_assert!(loaded.state.is_dirty(), "sync remesh of non-Dirty {coord:?}");
            loaded.retire(MeshState::from_upload(handles), eng);
        }
    }

    /// Re-snapshot hot solidity array if palette grew (append-only, new Arc, old jobs unaffected).
    pub(in crate::world) fn refresh_tables(&mut self) {
        // Split the borrow: `sync`'s rebuild closure needs `&self.registry`
        // while `&mut self.tables` is held, so bind `registry` separately.
        let count = self.registry.block_count();
        let registry = &self.registry;
        let layer_cap = self.texture_layer_cap;
        self.tables.sync(Revision::from_count(count), || {
            let mut tables = registry.hot_tables();
            tables.layer_cap = layer_cap;
            tables
        });
    }

    /// Rebuild/upload block texture array on palette growth (rare: world entry or new block type).
    /// The per-id layer cache makes growth O(new blocks), not O(palette).
    fn refresh_textures(&mut self, eng: &mut Engine) {
        // Never zero (modulo divisor) and never past the vertex field's u16.
        self.texture_layer_cap = eng.max_texture_array_layers().clamp(1, u16::MAX as u32) as u16;
        let count = self.registry.block_count();
        if self.textures_built != count {
            for i in self.texture_cache.len()..count {
                self.texture_cache.push(crate::block::texture::build_block_texture(
                    &self.registry,
                    crate::block::registry::BlockId(i as u16),
                ));
            }
            let visible = count.min(self.texture_layer_cap as usize);
            if count > visible && self.textures_built <= visible {
                eprintln!(
                    "block palette ({count}) exceeds the device texture-layer cap \
                     ({visible}); further block textures wrap onto existing layers"
                );
            }
            eng.set_block_textures(crate::block::texture::TEXTURE_SIZE, &self.texture_cache[..visible]);
            self.textures_built = count;
        }
    }
}
