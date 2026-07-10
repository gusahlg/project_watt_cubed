//! Streaming: the [`World::stream`] pass and everything it drives — draining
//! worker results, queueing generation and mesh jobs, budgeted uploads,
//! unloading far chunks, and the radius/centre bookkeeping. Code motion only:
//! these are `World` methods; the struct itself lives in `mod.rs`.

use voxel_engine::{DVec3, Engine};

use crate::coord::{ByPass, ChunkBox, ChunkCoord, Face};
use crate::derived::Revision;
use crate::math::block_coord;

use super::chunk::{CHUNK_SIZE, Chunk};
use super::generation::TerrainGenerator;
use super::lod::{TILE_LOD, Tile, TileState};
use super::mesh::ChunkMeshData;
use super::skin::{self, SkinColumn, SkinState};
use super::{
    Coord, DIRTY_BUDGET, FastSet, LIGHT_APPLY_BUDGET, LOD_REACH, LightLane, Loaded,
    MeshLane, MeshState, SKIN_REACH, SKIN_UPLOAD_BUDGET, SkinLane, TILE_UPLOAD_BUDGET, TileLane,
    UPLOAD_BUDGET, World, lane_enqueue, lane_integrate, light, mesh, pipeline,
};

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
        // Land worker results before the scans below, so freshly generated
        // chunks count as data this frame and finished meshes draw this frame.
        {
            let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::StreamDrain);
            self.drain_results(eng);
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
                lane_enqueue::<LightLane>(self, center_chunk);
                // While light is unsettled, keep the mesh lane armed so it
                // re-checks `light_ready` as grids land.
                if !self.light_worklist.is_empty() || !self.light_inflight.is_empty() {
                    self.pending_fresh.set();
                }
            } else {
                // No flood: drop seeds so `light_ready` never blocks meshing.
                self.light_worklist.clear();
            }
        }
        // Sync dirty remesh (edited chunks, budgeted) then the fresh mesh lane
        // (worklist seeded on load/light-move, O(shell) not a whole-map rescan).
        {
            let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::StreamMesh);
            self.remesh_dirty(center_chunk, eng);
            // The mesh lane evicts blocked/stale seeds itself (see `lane_enqueue`),
            // so the worklist stays O(fresh work) with no separate prune here.
            lane_enqueue::<MeshLane>(self, center_chunk);
        }
        // Far LOD tiles + Zone-3 far skin: geometry lanes. Select/unload only on a
        // boundary cross; the lane's budget spreads a world-entry flood.
        {
            let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::StreamTiles);
            if self.tiles_enabled {
                if full_pass {
                    self.unload_tiles(center_chunk, eng);
                    self.pending_tiles.set();
                }
                lane_enqueue::<TileLane>(self, center_chunk);
            } else {
                // Lane disabled: free any lingering tiles and stop enqueueing.
                for (_, state) in self.tiles.drain() {
                    state.free(eng);
                }
            }
            if self.skins_enabled {
                if full_pass {
                    self.unload_skins(center_chunk, eng);
                    self.pending_skins.set();
                }
                lane_enqueue::<SkinLane>(self, center_chunk);
            } else {
                for (_, state) in self.skins.drain() {
                    state.free(eng);
                }
            }
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
    fn drain_results(&mut self, eng: &mut Engine) {
        if let Some(workers) = &self.workers {
            while let Some(done) = workers.try_recv() {
                self.done_scratch.push(done);
            }
        }
        if self.done_scratch.is_empty()
            && self.upload_queue.is_empty()
            && self.tile_upload_queue.is_empty()
            && self.skin_upload_queue.is_empty()
            && self.light_apply_queue.is_empty()
        {
            return;
        }
        // Process outside the drain loop (the borrow checker aside, accepting
        // a result mutates half the world); the swap keeps the capacity. Each
        // finished result routes through its lane's `integrate` (light/mesh/
        // tile/skin) or, for the carved-out generation path, `accept_column`.
        let mut done = std::mem::take(&mut self.done_scratch);
        for result in done.drain(..) {
            match result {
                pipeline::Done::Column { col, chunks } => self.accept_column(col, chunks),
                m @ pipeline::Done::Mesh { .. } => lane_integrate::<MeshLane>(self, m),
                l @ pipeline::Done::Light { .. } => lane_integrate::<LightLane>(self, l),
                t @ pipeline::Done::Tile { .. } => lane_integrate::<TileLane>(self, t),
                s @ pipeline::Done::Skin { .. } => lane_integrate::<SkinLane>(self, s),
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
        let mut applied = 0;
        while applied < LIGHT_APPLY_BUDGET {
            let Some((coord, grid)) = self.light_apply_queue.pop_front() else { break };
            applied += 1;
            self.light_inflight.remove(&coord);
            self.publish_light(coord, grid);
        }

        // Tile uploads, on their own budget so a world-entry tile flood can't
        // starve chunk uploads. A landing for a tile that has since unloaded (or
        // already uploaded) is dropped — tiles carry no rev, so "still Meshing"
        // is the whole staleness check.
        let mut tile_uploads = 0;
        while tile_uploads < TILE_UPLOAD_BUDGET {
            let Some((tile, data)) = self.tile_upload_queue.pop_front() else { break };
            tile_uploads += 1;
            if let Some(state @ TileState::Meshing) = self.tiles.get_mut(&tile) {
                // An empty coarse sample (all-air sky or fully-buried stone)
                // uploads to no handle on any pass → the tile is born `Air`,
                // never drawn. Each present pass uploads together, same as chunks.
                let handles = ByPass::from_fn(|p| eng.upload_mesh(&data[p]));
                *state = TileState::from_upload(handles);
            }
        }

        // Skin uploads, on their own budget. A landing for a column that has
        // since unloaded is dropped (columns carry no rev — "still Meshing" is
        // the whole staleness check). Unlike a tile, a skin has NO `Air`
        // fallback: every column has a surface, so an `upload_surface` alloc
        // failure (`None`) leaves it `Meshing` to retry next frame rather than
        // wedging it in a broken empty state.
        let mut skin_uploads = 0;
        while skin_uploads < SKIN_UPLOAD_BUDGET {
            let Some((col, data)) = self.skin_upload_queue.pop_front() else { break };
            skin_uploads += 1;
            if let Some(state @ SkinState::Meshing) = self.skins.get_mut(&col) {
                if let Some(h) = eng.upload_surface(&data) {
                    *state = SkinState::ready(h);
                }
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
        #[cfg(debug_assertions)]
        self.debug_assert_liveness();
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
        #[cfg(debug_assertions)]
        self.debug_assert_liveness();
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
        self.chunks.insert(coord, Loaded { chunk: std::sync::Arc::clone(&chunk), state, rev: 0, connectivity: None, light: None });
        // Light: try the analytic fast path first — a uniform-opaque chunk settles
        // to all-dark and an above-surface uniform-air chunk to full sky with no
        // flood. A trivial grid publishes synchronously (which fans the border to
        // its neighbours); only the residual Dense band seeds the settle worklist.
        if self.lighting {
            match self.trivial_light(coord, &chunk) {
                Some(grid) => self.publish_light(coord, grid),
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
    pub(in crate::world) fn snapshot(&self, coord: Coord) -> (u32, pipeline::ChunkSnapshot) {
        let loaded = &self.chunks[&coord];
        (
            loaded.rev,
            pipeline::ChunkSnapshot {
                padded: self.capture_padded(coord),
                uniform: loaded.chunk.uniform(),
                light: self.capture_padded_light(coord),
                tables: self.tables.get(),
            },
        )
    }

    /// Settled light shell for chunk and 26 neighbours (18³). Missing reads dark, ensures seamless light.
    fn capture_padded_light(&self, coord: Coord) -> light::PaddedLight {
        if !self.lighting {
            return light::PaddedLight::full();
        }
        light::PaddedLight::capture(|dx, dy, dz| {
            self.chunks
                .get(&Coord::new(coord.x + dx, coord.y + dy, coord.z + dz))
                .and_then(|l| l.light.as_ref())
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

    /// Publish a settled light grid (from the analytic path or a worker flood):
    /// diff it against the previously published grid to find which shared borders
    /// moved (→ the neighbours to re-settle) and whether anything changed (→ this
    /// chunk's own mesh, if any, is stale). The cheap main-thread bookkeeping half
    /// of a settle, shared by the sync (trivial) and async (drained) paths.
    pub(in crate::world) fn publish_light(&mut self, coord: Coord, grid: light::LightGrid) {
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
        if !self_changed {
            return;
        }
        // A settled chunk may now be meshable, or a neighbour may be; seed the
        // mesh lane and re-settle the neighbours whose shared border moved.
        self.pending_fresh.set();
        self.mesh_worklist.insert(coord);
        for face in &moved {
            let n = coord.step(*face);
            self.light_worklist.insert(n);
            self.mesh_worklist.insert(n);
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

    /// Tiles to load: LOD ring out to LOD_REACH × view_radius, minus fully-covered by slab.
    pub(in crate::world) fn desired_tiles(&self, center: Coord) -> Vec<Tile> {
        let lod = TILE_LOD;
        let cps = lod.chunks_per_side();
        let (ptx, pty, ptz) =
            (center.x.div_euclid(cps), center.y.div_euclid(cps), center.z.div_euclid(cps));
        // `+1` covers the partial tile the centre sits inside. A cube volume, so
        // the vertical reach mirrors the horizontal — the shell hangs islands and
        // overhangs above/below, and buried/air tiles cost only an empty sample.
        let tr = (self.view.horizontal * LOD_REACH).div_euclid(cps) + 1;
        let tvr = (self.view.vertical * LOD_REACH).div_euclid(cps) + 1;
        let mut out = Vec::new();
        for tx in (ptx - tr)..=(ptx + tr) {
            for ty in (pty - tvr)..=(pty + tvr) {
                for tz in (ptz - tr)..=(ptz + tr) {
                    let tile = Tile { lod, x: tx, y: ty, z: tz };
                    if !self.tile_occluded(tile) {
                        out.push(tile);
                    }
                }
            }
        }
        out
    }

    /// Free tiles outside the desired ring (runs on a boundary cross).
    fn unload_tiles(&mut self, center: Coord, eng: &mut Engine) {
        let keep: FastSet<Tile> = self.desired_tiles(center).into_iter().collect();
        let stale: Vec<Tile> = self.tiles.keys().copied().filter(|t| !keep.contains(t)).collect();
        for tile in stale {
            if let Some(state) = self.tiles.remove(&tile) {
                state.free(eng);
            }
        }
    }

    /// Skin columns to load: a 2-D `(x, z)` disk on the [`SKIN_LOD`] grid out to
    /// `SKIN_REACH × view_radius`, independent of player y (a horizon backdrop).
    /// Mirrors [`desired_tiles`](Self::desired_tiles) with the vertical axis
    /// removed. No near-field cull: a `SKIN_LOD` column (1024 m) always dwarfs
    /// the full-res box, so per-column occlusion could never fire — the render's
    /// fragment clip hides the near part of the disk instead (see `set_skin_clip`).
    pub(in crate::world) fn desired_columns(&self, center: Coord) -> Vec<SkinColumn> {
        let cps = skin::SKIN_LOD.chunks_per_side();
        let (pcx, pcz) = (center.x.div_euclid(cps), center.z.div_euclid(cps));
        // `+1` covers the partial column the centre sits inside.
        let cr = (self.view.horizontal * SKIN_REACH).div_euclid(cps) + 1;
        let mut out = Vec::new();
        for cx in (pcx - cr)..=(pcx + cr) {
            for cz in (pcz - cr)..=(pcz + cr) {
                out.push(SkinColumn { x: cx, z: cz });
            }
        }
        out
    }

    /// Free skin columns outside the desired ring (runs on a boundary cross).
    fn unload_skins(&mut self, center: Coord, eng: &mut Engine) {
        let keep: FastSet<SkinColumn> = self.desired_columns(center).into_iter().collect();
        let stale: Vec<SkinColumn> =
            self.skins.keys().copied().filter(|c| !keep.contains(c)).collect();
        for col in stale {
            if let Some(state) = self.skins.remove(&col) {
                state.free(eng);
            }
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
        let light = self.capture_padded_light(coord);
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
        self.tables.sync(Revision::from_count(count), || registry.hot_tables());
    }

    /// Rebuild/upload block texture array on palette growth (rare: world entry or new block type).
    fn refresh_textures(&mut self, eng: &mut Engine) {
        let count = self.registry.block_count();
        if self.textures_built != count {
            let layers = crate::block::texture::build_block_textures(&self.registry);
            eng.set_block_textures(crate::block::texture::TEXTURE_SIZE, &layers);
            self.textures_built = count;
        }
    }
}
