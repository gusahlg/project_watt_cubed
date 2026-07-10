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
use super::lod::{self, TILE_LOD, Tile, TileState};
use super::mesh::ChunkMeshData;
use super::{
    Coord, DIRTY_BUDGET, FastSet, LIGHT_SETTLE_BUDGET, LOD_REACH, Loaded,
    MESH_ENQUEUE_BUDGET, MeshState, TILE_ENQUEUE_BUDGET, TILE_UPLOAD_BUDGET, UPLOAD_BUDGET, World,
    light, mesh, pipeline,
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
    /// check: the enqueue gate ([`build_meshes`](Self::build_meshes)) and the
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
        self.drain_results(eng);
        if full_pass {
            self.unload_far(center_chunk, eng);
            self.request_region_data(center_chunk);
            self.pending_fresh.set();
        }
        if self.radius_shrunk.take() {
            // Meshes between the new view radius and the unload ring survive
            // unload_far's hysteresis; free them now (data stays loaded).
            // `Air`/`NeedsMesh`/`Meshing` own no handle — nothing to free.
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
                    MeshState::Ready(_) => MeshState::NeedsMesh,
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
        // Settle light BEFORE meshing: it is cheap (main-thread Gauss-Seidel over
        // the worklist) and the mesh gate waits on it, so relaxing first lets a
        // freshly settled chunk mesh the same frame.
        self.settle_light(center_chunk);
        self.build_meshes(center_chunk, eng);
        // Far LOD tiles: a parallel lane. Select/unload only on a boundary cross;
        // enqueue is budgeted, so `pending_tiles` spreads a world-entry flood.
        if full_pass {
            self.unload_tiles(center_chunk, eng);
            self.pending_tiles.set();
        }
        if self.pending_tiles.get() {
            self.enqueue_tiles(center_chunk);
        }
        // Occlusion is derived state, rebuilt here at the `&mut` sync point (never
        // in the `&self` render) — and only when the adaptive gate is active AND
        // an input changed (or it was just activated). When the gate is off this
        // is skipped entirely and render draws everything, so a CPU-bound world
        // pays nothing for occlusion.
        let occlusion_on = self.occlusion_enabled();
        if occlusion_on && (self.occlusion_dirty.take() || !self.occlusion_active) {
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
        {
            return;
        }
        // Process outside the drain loop (the borrow checker aside, accepting
        // a result mutates half the world); the swap keeps the capacity.
        let mut done = std::mem::take(&mut self.done_scratch);
        for result in done.drain(..) {
            match result {
                pipeline::Done::Chunk { coord, chunk } => {
                    self.generating.remove(&coord);
                    self.accept_chunk(coord, chunk);
                }
                pipeline::Done::Mesh { coord, rev, data } => {
                    // The coord stays claimed by its `Meshing` state until the
                    // budgeted upload resolves, so the fresh scan can't
                    // re-enqueue a duplicate build meanwhile.
                    self.accept_mesh(coord, rev, data);
                }
                pipeline::Done::Tile { tile, data } => {
                    // Tiles never go stale by edit; a landing for an unloaded tile
                    // is simply dropped at upload time.
                    self.tile_upload_queue.push_back((tile, data));
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
            // Charge the budget per CONSUMED entry, whether it uploads or is
            // dropped as stale — otherwise a queue full of stale entries drains
            // entirely in one frame, defeating UPLOAD_BUDGET.
            uploads += 1;
            if !self.mesh_result_applies(coord, rev) {
                // Went stale while queued (an edit turned it `Dirty`, or it left
                // the box). Drop it; the sync remesh or a later fresh scan
                // covers the chunk. `Meshing` is cleared by whichever path owns
                // it now, not here.
                self.pending_fresh.set();
                continue;
            }
            // Upload each present pass; an empty pass yields no handle. Both
            // passes of a chunk upload together under one budget charge — they
            // share the chunk's fate (same rev), so never show half its geometry.
            let handles = ByPass::from_fn(|p| eng.upload_mesh(&data[p]));
            // Light is decoupled — it was settled and published before this mesh
            // was ever enqueued, so the upload is a pure GPU handoff.
            if let Some(loaded) = self.chunks.get_mut(&coord) {
                // The rev referee above guarantees this chunk is still `Meshing`
                // (an edit would have bumped rev → dropped), so it owns no handle
                // — but route through `retire` anyway: it frees any stray token
                // for free, keeping the async path correct by construction rather
                // than by assertion.
                loaded.retire(MeshState::from_upload(handles), eng);
            }
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
                // uploads to no handle → the tile is born `Air`, never drawn.
                *state = match eng.upload_mesh(&data) {
                    Some(h) => TileState::ready(h),
                    None => TileState::Air,
                };
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
            // or left the box. Nothing to release — the `Meshing` claim, if the
            // chunk still holds one, is a separate concern; just re-arm the
            // fresh scan so a still-`NeedsMesh` coord can be picked up again.
            self.pending_fresh.set();
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

    /// Queue generation for missing chunks in data box (nearest first). Centre generates sync for safety.
    fn request_region_data(&mut self, center: Coord) {
        self.ensure_data(center);
        let mut missing: Vec<Coord> = Vec::new();
        for coord in self.data_box(center).coords() {
            if !self.chunks.contains_key(&coord) && !self.generating.contains(&coord) {
                missing.push(coord);
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
                self.generating.insert(coord);
            }
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
        let state = if born_air { MeshState::Air } else { MeshState::NeedsMesh };
        // Connectivity is computed lazily by the occlusion rebuild (only if the
        // gate is active), so generation pays no flood-fill when occlusion is off.
        self.chunks.insert(coord, Loaded { chunk, state, rev: 0, connectivity: None, light: None });
        // New voxel data is a light source that must be settled — for the chunk
        // itself and (once its border resolves) for its neighbours.
        self.light_worklist.insert(coord);
        // A new chunk changes what the BFS can reach.
        self.occlusion_dirty.set();
        // New data means new mesh work next scan (its neighbours may have
        // been waiting on this chunk even when it is itself uniform air).
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
            // Dirty still showing its old mesh); Air/NeedsMesh/Meshing own none.
            // The removed `Loaded` owns its token, so `free_owned` consumes it
            // to be freed exactly once as the entry is discarded.
            if let Some(loaded) = self.chunks.remove(&coord) {
                loaded.state.free_owned(eng);
            }
        }
    }

    /// Remesh dirty chunks (sync, budgeted, nearest first). Enqueue fresh chunks when neighbours ready.
    fn build_meshes(&mut self, center: Coord, eng: &mut Engine) {
        // Remesh edited chunks: the `Dirty` fiber of `MeshState`, materialised
        // into a scratch Vec (the pass mutates each chunk via `mesh_chunk`, so
        // it can't hold the filter borrow). Gated by `pending_dirty` so an idle
        // frame does no scan; nearest first, capped by `DIRTY_BUDGET`.
        if self.pending_dirty.take() {
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
                // No neighbour-data gate here: an edited chunk must remesh even
                // when a far neighbour has no data (the mesher reads missing
                // neighbours as air, exactly like the original world lookup).
                // Gating would leave a stale mesh with a hole at the border in
                // margin chunks whose outer neighbour never loads. The coord came
                // from `chunks`, so it is still present.
                self.mesh_chunk(coord, eng);
            }
        }

        // Fresh chunks: snapshot and hand to the worker pool, nearest first,
        // capped by the frame budget. Skipped entirely once a scan came up
        // empty, until something re-flags work.
        if !self.pending_fresh.get() {
            return;
        }
        let mut pending: Vec<Coord> = self
            .chunks
            .keys()
            .copied()
            // The meshed-ness test IS the `NeedsMesh` fiber: `is_needs_mesh`
            // excludes `Dirty` (sync-remesh path's job) and `Meshing` (already in
            // flight) by construction, and a chunk with data is never in
            // `generating` — so the state machine alone is the index, no
            // `∉dirty`/`∉generating` guard needed.
            .filter(|&coord| self.is_needs_mesh(coord))
            // Same mesh-view check as `mesh_result_applies`: a chunk is
            // enqueued only if its result would be accepted.
            .filter(|&coord| self.in_mesh_box(coord))
            .filter(|&coord| self.neighbours_have_data(coord))
            // Mesh only once the neighbourhood light has settled, so a chunk is
            // meshed with its final smooth light instead of remeshed per step.
            .filter(|&coord| self.light_ready(coord))
            .collect();
        pending.sort_by_key(|&coord| Self::order(coord, center));

        if pending.len() <= MESH_ENQUEUE_BUDGET {
            // Every candidate below gets enqueued, so the scan has nothing
            // left. Clearing while jobs still fly is sound: a `Meshing` coord is
            // *not* scan work — its result either lands as a mesh (state flips to
            // `Ready`/`Air`, nothing to scan) or fails to apply, which re-sets
            // `pending_fresh`, so the next scan sees it again. See `drain_results`.
            self.pending_fresh.take();
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
                // NeedsMesh → Meshing: this transition IS the mesh-in-flight
                // claim (held until the budgeted upload retires it). The
                // candidate set is `is_needs_mesh`, so the chunk owns no handle.
                if let Some(loaded) = self.chunks.get_mut(&coord) {
                    debug_assert!(
                        loaded.state.is_needs_mesh(),
                        "fresh-scan submit for non-NeedsMesh {coord:?}: {:?}",
                        loaded.state
                    );
                    loaded.state = MeshState::Meshing;
                }
            }
        }
    }

    /// Snapshot for mesh job: chunk storage, neighbour shell, solidity table, and rev.
    fn snapshot(&self, coord: Coord) -> (u32, pipeline::ChunkSnapshot) {
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
        light::PaddedLight::capture(|dx, dy, dz| {
            self.chunks
                .get(&Coord::new(coord.x + dx, coord.y + dy, coord.z + dz))
                .and_then(|l| l.light.as_ref())
        })
    }

    /// Skylight ceiling: surface height per column (pure generator fn, caves dark consistently).
    fn capture_ceiling(&self, coord: Coord) -> light::CeilingWindow {
        let x0 = coord.x * CHUNK_SIZE as i32;
        let z0 = coord.z * CHUNK_SIZE as i32;
        light::CeilingWindow::from_heights(|lx, lz| {
            self.generator.height(x0 + lx as i32, z0 + lz as i32)
        })
    }

    /// Relax light field (budgeted, nearest-first Gauss-Seidel). Enqueues neighbours
    /// with moved borders; marks existing meshes dirty if this chunk's light changed.
    fn settle_light(&mut self, center: Coord) {
        self.light_worklist.retain(|c| self.chunks.contains_key(c));
        if self.light_worklist.is_empty() {
            return;
        }
        self.refresh_tables();
        let tables = self.tables.get();
        let mut ready: Vec<Coord> = self.light_worklist.iter().copied().collect();
        ready.sort_by_key(|&coord| Self::order(coord, center));
        for coord in ready.into_iter().take(LIGHT_SETTLE_BUDGET) {
            self.light_worklist.remove(&coord);
            let padded = self.capture_padded(coord);
            let shell = self.capture_padded_light(coord);
            let ceiling = self.capture_ceiling(coord);
            let mut grid = light::LightGrid::dark();
            light::propagate(&padded, &shell, &ceiling, coord.y * CHUNK_SIZE as i32, &tables, &mut grid);

            // Diff against the published grid: which shared borders moved (→ the
            // neighbours to re-settle), and did anything at all change (→ this
            // chunk's own mesh, if any, is stale).
            let loaded = &self.chunks[&coord];
            let (self_changed, moved): (bool, Vec<Face>) = match &loaded.light {
                None => (true, Face::ALL.to_vec()),
                Some(old) => (
                    *old != grid,
                    Face::ALL.into_iter().filter(|&f| light::border_changed(old, &grid, f)).collect(),
                ),
            };
            self.chunks.get_mut(&coord).unwrap().light = Some(grid);
            if !self_changed {
                continue;
            }
            // A settled chunk may now be meshable, or a neighbour may be; re-arm
            // the fresh scan so the mesh gate re-examines the neighbourhood.
            self.pending_fresh.set();
            for face in moved {
                self.light_worklist.insert(coord.step(face));
            }
            // My light changed and I already show (or am building) a mesh → that
            // mesh is stale. Reuse the edit invalidation: keep the old mesh drawn,
            // bump rev to strand any in-flight build, re-mesh with the new light.
            // A not-yet-meshed chunk (`NeedsMesh`/`Air`) needs nothing here — it
            // meshes correctly once light is ready.
            let loaded = self.chunks.get_mut(&coord).unwrap();
            if matches!(loaded.state, MeshState::Ready(_) | MeshState::Dirty { .. } | MeshState::Meshing) {
                loaded.state.invalidate();
                loaded.rev = loaded.rev.wrapping_add(1);
                self.pending_dirty.set();
            }
        }
        // Anything left in the worklist settles next stream; the fresh scan must
        // keep running until it drains (a settled neighbour may unblock a mesh).
        if !self.light_worklist.is_empty() {
            self.pending_fresh.set();
        }
    }

    /// Chunk + 1-voxel neighbour shell for mesh build (shared by worker and sync paths).
    fn capture_padded(&self, coord: Coord) -> mesh::Padded {
        mesh::Padded::capture(|dx, dy, dz| {
            self.chunks
                .get(&Coord::new(coord.x + dx, coord.y + dy, coord.z + dz))
                .map(|l| &l.chunk)
        })
    }

    /// Tiles to load: LOD ring out to LOD_REACH × view_radius, minus fully-covered by slab.
    fn desired_tiles(&self, center: Coord) -> Vec<Tile> {
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

    /// Enqueue missing desired tiles (budgeted, nearest first). Clear flag when done.
    fn enqueue_tiles(&mut self, center: Coord) {
        let mut missing: Vec<Tile> = self
            .desired_tiles(center)
            .into_iter()
            .filter(|t| !self.tiles.contains_key(t))
            .collect();
        let remaining = missing.len();
        if remaining == 0 {
            self.pending_tiles.take();
            return;
        }
        missing.sort_by_key(|t| lod::tile_order(*t, center));
        // The coarse mesher reads the hot solidity/opacity tables like the chunk
        // mesher; make sure they reflect the current palette before the snapshot.
        self.refresh_tables();
        let tables = self.tables.get();
        let generator = self.generator.clone();
        let workers = self
            .workers
            .get_or_insert_with(|| pipeline::Workers::spawn(pipeline::Workers::default_threads()));
        for tile in missing.into_iter().take(TILE_ENQUEUE_BUDGET) {
            let job = pipeline::Job::Tile {
                tile,
                generator: generator.clone(),
                tables: tables.clone(),
            };
            if workers.submit(job) {
                self.tiles.insert(tile, TileState::Meshing);
            }
        }
        // Clear the hint only when this pass drained the whole backlog.
        if remaining <= TILE_ENQUEUE_BUDGET {
            self.pending_tiles.take();
        }
    }

    /// Six orthogonal neighbours have data loaded.
    fn neighbours_have_data(&self, coord: Coord) -> bool {
        Face::ALL.iter().all(|&f| self.chunks.contains_key(&coord.step(f)))
    }

    /// Light settled enough to mesh: chunk and face neighbours have grids, chunk not in worklist.
    fn light_ready(&self, coord: Coord) -> bool {
        !self.light_worklist.contains(&coord)
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
        // lands a frame or two later when `settle_light` reconverges and marks
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
    fn refresh_tables(&mut self) {
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
