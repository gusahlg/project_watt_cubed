//! Landing worker results: the result drain, integration, and budgeted chunk uploads.

use super::*;
use super::pacer::RESULT_INTEGRATE_FLOOR;
use crate::world::{light, mesh};

impl World {
    /// Land finished worker results (non-blocking). Generate results clear
    /// `generating`; stale results release their exact claims. Result
    /// integration used to drain the unbounded channel in one frame, making a
    /// productive worker burst a main-thread hitch. It now shares the adaptive
    /// effort signal and keeps a small forward-progress floor.
    pub(in crate::world) fn drain_results(&mut self, eng: &mut Engine, result_budget: Duration) {
        self.integrate_results(result_budget);
        self.counters.section_upload_bytes = 0;
        self.counters.drain_upload_bytes = 0;
        if self.upload_queue.is_empty()
            && self.section_upload_queue.is_empty()
            && self.light_apply_queue.is_empty()
        {
            return;
        }

        // Budgeted uploads, charged in BYTES (the actual staging cost — see
        // `UPLOAD_BUDGET_BYTES`). A stale entry costs nothing but a bounded
        // pop (`UPLOAD_SCAN_MAX`), so a post-flight queue of stale entries no
        // longer starves real uploads for dozens of frames. The byte check
        // sits at the loop head, so at least one real upload always lands —
        // the same forward-progress floor the admission lanes keep.
        // Re-validate at the moment of upload: an entry may have sat queued
        // across frames while an edit bumped the chunk's rev.
        let pacer = self.stream_pacer;
        let upload_budget = pacer.upload_bytes();
        let mut upload_bytes = 0usize;
        let mut uploads = 0usize;
        let mut pops = 0usize;
        while (uploads == 0 || upload_bytes < upload_budget) && pops < UPLOAD_SCAN_MAX {
            let Some((coord, rev, data)) = self.upload_queue.pop_front() else {
                break;
            };
            pops += 1;
            if !self.mesh_result_applies(coord, rev) {
                // Stale while queued: edit made it Dirty or it left the box.
                data.release_staging(eng);
                self.drop_stale_upload(coord);
                continue;
            }
            upload_bytes += mesh_output_bytes(&data);
            uploads += 1;
            // Both passes upload together under one budget charge (same rev).
            // Staged payloads install through the worker-written ring; the
            // Vec fallback uses the existing main-thread copy. (The rev
            // check above guarantees the state is NeedsMesh { building: true }.)
            self.upload_chunk_payload(coord, data, eng);
        }

        self.apply_light_queue();

        // Section uploads share the chunk byte counter. A section is binary
        // (`SectionState` Ready-or-not), so the byte gate sits before the pop:
        // the next whole tile lands only while the counter has room. The count
        // cap stays as a secondary ceiling. Re-validated by claim token at the
        // moment of upload: an entry that sat queued across an unload or a
        // re-admission must not capture the replacement claim.
        let section_budget = pacer.section_uploads();
        let mut section_uploads = 0;
        while section_uploads < section_budget {
            let Some((_, _, _, _)) = self.section_upload_queue.front() else {
                break;
            };
            if upload_bytes >= upload_budget {
                break;
            }
            let (pos, token, bytes, meshes) = self
                .section_upload_queue
                .pop_front()
                .expect("front was Some");
            section_uploads += 1;
            // `section_material` borrows all of `self`, so it must run before
            // `self.sections.get_mut` below takes an overlapping mutable borrow.
            let (flat_color, flat_rgba) = self.section_material(pos);
            let bend = self.chart_bend(pos);
            if let Some(state @ SectionState::Meshing { .. }) = self.sections.get_mut(&pos)
                && matches!(state, SectionState::Meshing { token: t } if *t == token)
            {
                super::adjust_count(&mut self.meshing_sections, true, false);
                upload_bytes += bytes;
                self.counters.section_upload_bytes += bytes;
                *state = SectionState::from_upload_payload(pos, meshes, eng, bend.as_ref());
                // Slots are born visible (residency implies it for everything but the
                // far field), so a section that Coverage does not draw — or draws only
                // in part — must be corrected here, at the transition that gave it slots
                // to correct. No frame intervenes: patches flush at submit.
                state.set_visible(eng, self.section_fade.drawn_mask(pos));
                // Push the section's far-material style so it doesn't draw one frame at
                // the engine's post-upload default.
                state.push_style(eng, FadeStyle { flat_color }, flat_rgba);
                // A new Ready section moves the covering: re-arm the lane so
                // any refinement it exposes loads immediately.
                self.pending_sections.set();
                self.section_cover_dirty.set();
            } else {
                meshes.release_staging(eng);
            }
        }
        self.counters.drain_upload_bytes = upload_bytes;
    }

    /// The first block of [`drain_results`](Self::drain_results): integrate finished worker
    /// results within the paced `budget`, past a small forward-progress floor.
    pub(in crate::world) fn integrate_results(&mut self, budget: Duration) {
        let pacer = self.stream_pacer;
        let deadline = pipeline::Deadline::from_budget(pacer.duration(budget));
        let floor = pacer.floor(RESULT_INTEGRATE_FLOOR);
        let mut integrated = 0usize;
        while integrated < floor || !deadline.expired() {
            let Some(result) = self.workers.as_ref().and_then(pipeline::Workers::try_recv) else {
                break;
            };
            self.integrate_worker_result(result);
            integrated += 1;
        }
    }

    /// Budgeted light application, after the chunk uploads of
    /// [`drain_results`](Self::drain_results). Order-independent: each grid is absolute,
    /// leftovers apply next frame with no seam.
    pub(in crate::world) fn apply_light_queue(&mut self) {
        let deadline =
            pipeline::Deadline::from_budget(self.stream_pacer.duration(pipeline::LIGHT_APPLY_BUDGET));
        let mut applied = 0usize;
        while applied == 0 || !deadline.expired() {
            let Some((coord, grid)) = self.light_apply_queue.pop_front() else {
                break;
            };
            self.settle_light(coord, grid);
            applied += 1;
        }
    }

    /// Route one completed worker payload through its owning lane. This is the
    /// claim-resolution chokepoint: every accepted claim is owed exactly one
    /// payload, cancellation, or failure, and consuming it must release or
    /// transfer that claim even when the result became stale in flight.
    pub(super) fn integrate_worker_result(&mut self, result: pipeline::Done) {
        #[cfg(debug_assertions)]
        let light_audit = match &result {
            pipeline::Done::Light { coord, epoch, light_gen, .. } => {
                Some((*coord, *epoch, *light_gen))
            }
            _ => None,
        };
        #[cfg(debug_assertions)]
        let section_audit = match &result {
            pipeline::Done::Section {
                pos, epoch, token, ..
            } => Some((*pos, *epoch, *token)),
            _ => None,
        };
        match result {
            pipeline::Done::Column { key, chunks, heights } => {
                self.counters.jobs_completed += 1;
                self.accept_column(key, chunks, heights)
            }
            m @ pipeline::Done::Mesh { .. } => {
                self.counters.jobs_completed += 1;
                MeshLane::integrate(self, m)
            }
            l @ pipeline::Done::Light { .. } => {
                self.counters.jobs_completed += 1;
                LightLane::integrate(self, l)
            }
            sc @ pipeline::Done::Section { .. } => {
                self.counters.jobs_completed += 1;
                SectionLane::integrate(self, sc)
            }
            pipeline::Done::Failed(key) => self.fail_job(*key),
            pipeline::Done::Cancelled(keys) => {
                self.counters.jobs_cancelled += keys.len() as u64;
                for key in keys {
                    self.cancel_job(key);
                }
            }
        }
        // A consumed CURRENT-epoch light result must have released its claim
        // or transferred it into the apply queue.
        #[cfg(debug_assertions)]
        if let Some((coord, epoch, light_gen)) = light_audit {
            let live = self.chunks.get(&coord).map(|l| l.light_gen) == Some(light_gen);
            debug_assert!(
                epoch != self.light_epoch
                    || !live
                    || !self.light_inflight.contains(&coord)
                    || self.light_apply_queue.iter().any(|(c, _)| *c == coord),
                "light Done for {coord:?} left its claim neither released nor transferred"
            );
        }
        // A current section result matching the live token must likewise have
        // transferred to the upload queue.
        #[cfg(debug_assertions)]
        if let Some((pos, epoch, token)) = section_audit {
            debug_assert!(
                epoch != self.section_epoch
                    || !matches!(self.sections.get(&pos),
                        Some(SectionState::Meshing { token: t }) if *t == token)
                    || self
                        .section_upload_queue
                        .iter()
                        .any(|(p, t, _, _)| *p == pos && *t == token),
                "section Done for {pos:?} matched the live claim but was not transferred"
            );
        }
    }

    /// Generated chunk result: discard if out-of-range/already loaded; else store (replays edits).
    pub(in crate::world) fn accept_chunk(&mut self, coord: Coord, chunk: Chunk) {
        if !self.will_accept_chunk(coord) {
            if !self.chunks.contains_key(&coord) {
                self.counters.gen_discarded += 1;
            }
            return;
        }
        self.counters.gen_landed += 1;
        if self.chunk_landed_behind(coord) {
            self.counters.gen_landed_behind += 1;
        }
        self.store_chunk(coord, chunk);
    }

    /// A just-stored chunk sits strictly behind the travel direction. Rest and
    /// walking never count: the whole window is wanted.
    fn chunk_landed_behind(&self, coord: Coord) -> bool {
        let Some(center) = self.center else { return false };
        chunk_behind(center, self.fold.fold(coord), self.stream_pacer.travel(), self.live_up())
    }

    /// Mesh result at `rev`: queue for upload if still applies; else drop and re-arm scan.
    /// Staged regions release on drop of a rejected payload.
    pub(in crate::world) fn accept_mesh(
        &mut self,
        coord: Coord,
        rev: u32,
        data: impl Into<pipeline::MeshPayload>,
    ) {
        let data = data.into();
        if self.mesh_result_applies(coord, rev) {
            self.upload_queue.push_back((coord, rev, data));
        } else {
            // Stale: chunk edited (Dirty) or left box. Staging Drop releases.
            self.drop_stale_upload(coord);
        }
    }

    /// Release a stale mesh result's build claim and re-seed the coord so it
    /// can mesh again later — the one stale-drop path, shared by the accept
    /// site, the pop-time re-validation, and the boundary-cross prune. A chunk
    /// an edit made `Dirty` belongs to the dirty lane and takes no seed.
    pub(super) fn drop_stale_upload(&mut self, coord: Coord) {
        self.remesh_stats.note_drop_stale();
        // An unloaded chunk is not re-seeded: its next load seeds it.
        let Some(loaded) = self.chunks.get_mut(&coord) else { return };
        if loaded.state.release_build() {
            super::adjust_count(&mut self.building_meshes, true, false);
        }
        self.pending_fresh.set();
        self.seed_mesh(coord);
    }

    /// One-pass prune of stale upload entries (boundary cross): each is
    /// released and re-seeded exactly like the pop-time stale path — without
    /// letting a deep post-flight backlog of left-behind meshes trickle out
    /// at drain speed while real uploads wait behind it.
    pub(in crate::world) fn prune_upload_queue(&mut self) {
        if self.upload_queue.is_empty() {
            return;
        }
        let mut queue = std::mem::take(&mut self.upload_queue);
        let mut stale: Vec<Coord> = Vec::new();
        queue.retain(|(coord, rev, _)| {
            let live = self.mesh_result_applies(*coord, *rev);
            if !live {
                stale.push(*coord);
            }
            live
        });
        self.upload_queue = queue;
        for coord in stale {
            self.drop_stale_upload(coord);
        }
    }

    /// Whether mesh admission should pause this pass (see [`UPLOAD_QUEUE_MAX`]).
    pub(in crate::world) fn upload_backlogged(&self) -> bool {
        self.upload_queue.len() >= UPLOAD_QUEUE_MAX
    }

    /// Upload one built chunk mesh (every pass) and retire the chunk's state
    /// to the fresh `Ready`/`Air` — the one upload+install step shared by the
    /// async drain, the sync edit remesh, and the terminal degraded promotion.
    /// `retire` frees whatever the old state carried, exactly once.
    /// `hash` is `Some` only on the sync edit path; async passes `None` and
    /// clears `mesh_hash` (no `content_hash` on the main thread).
    pub(super) fn upload_chunk(
        &mut self,
        coord: Coord,
        data: &mesh::ChunkMeshData,
        hash: Option<u64>,
        eng: &mut Engine,
    ) {
        self.upload_chunk_inner(coord, Some((data, eng)), hash);
    }

    /// Worker payload: staged regions install through the ring; the `Vec`
    /// fallback uses the existing main-thread copy. Async, so `mesh_hash` is
    /// cleared (`None`).
    fn upload_chunk_payload(
        &mut self,
        coord: Coord,
        data: pipeline::MeshPayload,
        eng: &mut Engine,
    ) {
        match data {
            pipeline::MeshPayload::Cpu(data) => {
                self.upload_chunk(coord, &data, None, eng);
            }
            pipeline::MeshPayload::Staged(mut staged) => {
                let placement = self.placement_of(coord, eng);
                let wanted = ByPass::from_fn(|p| staged.passes[p].is_some());
                let handles = ByPass::from_fn(|p| {
                    staged.passes[p].take().and_then(|pass| {
                        eng.upload_mesh_staged(pass.staging, pass.quad_counts, p, placement)
                    })
                });
                self.install_chunk_handles(coord, handles, wanted, None, eng);
            }
        }
    }

    fn upload_chunk_inner(
        &mut self,
        coord: Coord,
        gpu: Option<(&mesh::ChunkMeshData, &mut Engine)>,
        hash: Option<u64>,
    ) {
        if let Some(h) = hash
            && self.chunks.get(&coord).is_some_and(|l| l.mesh_hash == Some(h))
        {
            self.keep_resident_mesh(coord, gpu.map(|(_, eng)| eng));
            return;
        }
        if let Some((data, eng)) = gpu {
            let placement = self.placement_of(coord, eng);
            let handles = ByPass::from_fn(|p| eng.upload_mesh_placed(&data[p], placement));
            self.install_chunk_handles(coord, handles, ByPass::from_fn(|p| !data[p].is_empty()), hash, eng);
            return;
        }
        if let Some(loaded) = self.chunks.get_mut(&coord) {
            loaded.mesh_hash = hash;
        }
    }

    /// Install an upload's `handles`. A pass with geometry (`wanted`) that got no handle failed:
    /// the chunk keeps whatever it drew before and is not settled, so the far field keeps
    /// drawing it.
    fn install_chunk_handles(
        &mut self,
        coord: Coord,
        handles: ByPass<Option<voxel_engine::MeshHandle>>,
        wanted: ByPass<bool>,
        hash: Option<u64>,
        eng: &mut Engine,
    ) {
        let failed = handles.iter().any(|(p, h)| wanted[p] && h.is_none());
        let vis = !self.occlusion_active || self.occlusion.is_visible(coord);
        if let Some(loaded) = self.chunks.get_mut(&coord) {
            let was = loaded.state.is_building();
            if failed {
                for h in handles.into_iter_passes().filter_map(|(_, h)| h) {
                    eng.free_mesh(h);
                }
                loaded.state.release_build();
            } else {
                loaded.retire(MeshState::from_upload(handles), eng);
                loaded.mesh_hash = hash;
                loaded.visible = vis;
                if !vis && let Some(meshes) = loaded.state.live_meshes() {
                    meshes.set_visible(eng, false);
                }
            }
            super::adjust_count(&mut self.building_meshes, was, loaded.state.is_building());
        }
        if !failed {
            self.note_settled();
        }
    }

    /// An edit remesh whose vertex bytes match the resident GPU mesh: keep the
    /// existing handles and drop the Dirty/NeedsMesh claim, no upload.
    pub(super) fn keep_resident_mesh(&mut self, coord: Coord, eng: Option<&mut Engine>) {
        let vis = !self.occlusion_active || self.occlusion.is_visible(coord);
        if let Some(loaded) = self.chunks.get_mut(&coord) {
            let was = loaded.state.is_building();
            let next = match std::mem::replace(&mut loaded.state, MeshState::Air) {
                MeshState::Ready(m)
                | MeshState::Dirty { prev: Some(m) }
                | MeshState::NeedsMesh { prev: Some(m), .. } => MeshState::Ready(m),
                MeshState::Dirty { prev: None }
                | MeshState::NeedsMesh { prev: None, .. }
                | MeshState::Air => MeshState::Air,
            };
            loaded.state = next;
            super::adjust_count(&mut self.building_meshes, was, false);
            self.remesh_stats.note_upload(coord);
            if vis != loaded.visible {
                loaded.visible = vis;
                if let Some(meshes) = loaded.state.live_meshes() {
                    #[cfg(test)]
                    super::vis_log::record(meshes.handles(), vis);
                    if let Some(eng) = eng {
                        meshes.set_visible(eng, vis);
                    }
                }
            }
        }
        self.note_settled();
    }

    #[cfg(test)]
    pub(super) fn upload_chunk_without_gpu(&mut self, coord: Coord, hash: Option<u64>) {
        self.upload_chunk_inner(coord, None, hash);
    }

    /// Light result at `epoch`: release-or-transfer the claim, then queue the
    /// grid if it still applies. The claim rule at this consumption site: a
    /// claimed key is owed exactly one `Done`, and consuming that `Done` must
    /// release or transfer the claim — silently dropping a result used to
    /// wedge its coord's `light_inflight` entry forever (a chunk re-loaded at
    /// that coord could never settle light again: `light_ready` read the
    /// stale claim as still-in-flight, the mesh lane skipped it as in-flight,
    /// and quiescence — degraded promotion, `entry_complete` — never came).
    ///
    /// Epoch and generation reasoning (what makes the release sound):
    /// [`transition_lighting`](Self::transition_lighting) is the only
    /// `light_epoch` bump and it clears `light_inflight` in the same breath, so
    /// - a CURRENT-epoch result is the unique owner of any in-flight entry at
    ///   its coord (releasing can never steal a newer claim), while
    /// - a STALE-epoch result's claim was already wiped at the bump — an entry
    ///   present now belongs to a post-bump job and must not be touched.
    /// `light_gen` is the per-`Loaded` stamp: unload then regenerate at the same
    /// coord does not bump the epoch, so a current-epoch result for the *old*
    /// resident must not publish onto the new voxels (and store skips trivial
    /// settle while the old claim is still in flight, so this Done remains
    /// that claim's unique owner).
    pub(in crate::world) fn accept_light(
        &mut self,
        coord: Coord,
        epoch: u32,
        light_gen: u32,
        grid: light::LightGrid,
    ) {
        if epoch != self.light_epoch {
            return;
        }
        let live_gen = self.chunks.get(&coord).map(|l| l.light_gen);
        if live_gen != Some(light_gen) {
            // Unusable: unloaded, or a later Loaded at this coord. Release
            // the leftover claim and re-seed the new resident so it can
            // settle against its own voxels.
            self.light_inflight.remove(&coord);
            if live_gen.is_some() && self.lighting {
                self.seed_light(coord, super::LightSeed::Store);
                self.light_pending.set();
            }
            return;
        }
        if !self.lighting {
            self.light_inflight.remove(&coord);
            return;
        }
        // TRANSFER: the claim stays held through the apply queue (so
        // `light_ready` keeps treating the chunk as unsettled);
        // [`settle_light`](Self::settle_light) is the release point.
        self.light_apply_queue.push_back((coord, grid));
    }

    /// Mesh result still applies: chunk loaded, in view range, rev not bumped.
    pub(in crate::world) fn mesh_result_applies(&self, coord: Coord, rev: u32) -> bool {
        self.in_mesh_box(coord) && self.chunks.get(&coord).is_some_and(|l| l.rev == rev)
    }
}
