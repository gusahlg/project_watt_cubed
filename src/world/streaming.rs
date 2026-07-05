//! Streaming: the [`World::stream`] pass and everything it drives — draining
//! worker results, queueing generation and mesh jobs, budgeted uploads,
//! unloading far chunks, and the radius/centre bookkeeping. Code motion only:
//! these are `World` methods; the struct itself lives in `mod.rs`.

use std::sync::Arc;

use voxel_engine::{Engine, MeshData, Vec3};

use crate::block::registry::{AIR, BlockId};

use super::chunk::{CHUNK_SIZE, Chunk};
use super::{
    Coord, DATA_MARGIN, DIRTY_BUDGET, Loaded, MESH_ENQUEUE_BUDGET, UNLOAD_MARGIN, UNLOAD_MARGIN_V,
    UPLOAD_BUDGET, World, mesh, pipeline,
};

impl World {
    /// Vertical streaming radius in chunk layers: half the horizontal view
    /// radius, clamped to 2..=5 — interesting terrain is mostly lateral, so
    /// the streamed volume stays a flat box rather than a cube.
    pub(in crate::world) fn vertical_radius(&self) -> i32 {
        (self.view_radius / 2).clamp(2, 5)
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
    pub(in crate::world) fn accept_chunk(&mut self, coord: Coord, chunk: Chunk) {
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
    pub(in crate::world) fn accept_mesh(&mut self, coord: Coord, rev: u32, data: MeshData) {
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
    pub(in crate::world) fn mesh_result_applies(&self, coord: Coord, rev: u32) -> bool {
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
    pub(in crate::world) fn order(a: Coord, b: Coord) -> i64 {
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
    pub(in crate::world) fn ensure_region_data(&mut self, center: Coord) {
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
    pub(in crate::world) fn ensure_data(&mut self, coord: Coord) {
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
}
