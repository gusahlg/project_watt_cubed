//! World entry readiness and the streaming gauges the harness and the debug overlay read.

use super::*;
use super::super::LightSeedSplit;

/// Live streaming-queue depths — the numeric twin of
/// [`entry_debug`](World::entry_debug)'s formatted counters, for the harness's
/// stress metrics (peak backlog depths, settle progress) where parsing a
/// debug string would be absurd. Cheap enough to sample every frame; the
/// remesh distributions, which sort, are a separate [`RemeshDistribution`]
/// taken at report time.
#[derive(Clone, Copy, Debug, Default)]
pub struct StreamGauges {
    pub chunks: usize,
    pub generating: usize,
    pub mesh_worklist: usize,
    pub upload_queue: usize,
    pub light_worklist: usize,
    pub light_inflight: usize,
    pub light_apply_queue: usize,
    /// Near/far jobs waiting in the shared worker queue (running jobs excluded).
    pub worker_near_queue: usize,
    pub worker_far_queue: usize,
    /// Worker threads in the pool.
    pub worker_capacity: usize,
    /// Current horizontal travel speed and normalized streaming effort.
    pub travel_speed_mps: f64,
    pub effort: f32,
    /// Cumulative light jobs admitted to the pool, and the count from the
    /// most recent light-admit pass (0 if that pass did not run).
    pub light_admitted: u64,
    pub light_admitted_last: usize,
    /// Cumulative `light_worklist` insert attempts (including already-queued).
    pub light_seed_inserts: u64,
    /// Live GPU mesh slots (engine gauge when streamed, else a local handle count).
    pub mesh_slots: usize,
    /// CPU-cull live-count ceiling (`cpu_cull_max`); a cost knob, not a hard cap.
    pub slot_ceiling: usize,
    /// Ready far-LOD sections (each is one mesh per pass).
    pub section_ready: usize,
    /// Insert attempts split by `seed_light` source (stress report at stop).
    pub light_seed_split: LightSeedSplit,
    /// `remesh_async` calls (rev-bumping rebuilds) this world has issued.
    pub remesh_async_calls: u64,
    /// Vertex bytes of section (LOD tile) meshes uploaded this stream pass.
    pub section_upload_bytes: usize,
    /// Vertex bytes of chunk + section meshes uploaded this drain (harness peak).
    pub drain_upload_bytes: usize,
    /// Worker staging vs CPU-fallback counts (cumulative for the world).
    pub mesh_staged: u64,
    pub mesh_fallback: u64,
    pub mesh_ring_full: u64,
    pub section_staged: u64,
    pub section_fallback: u64,
    pub section_ring_full: u64,
    /// Stale mesh drops (accept-time, pop-time, and prune).
    pub drop_stale_uploads: u64,
    /// Stale drops during the current stream/pump frame.
    pub drop_stale_this_frame: u32,
    /// Reaction-event scheduler queue depth at sample time.
    pub reactions_pending: usize,
    /// Cumulative reaction mutations committed by the scheduler.
    pub reactions_mutations: u64,
}

/// Per-chunk remesh counts over the world's life ([`World::remesh_distribution`]).
#[derive(Clone, Copy, Debug, Default)]
pub struct RemeshDistribution {
    /// `remesh_async` calls per coord between successful uploads.
    pub remesh_between_upload_mean: f32,
    pub remesh_between_upload_p95: f32,
    pub remesh_between_upload_n: u64,
    /// Mesh jobs claimed for a chunk before its 27-neighbourhood light fixpoint.
    pub mesh_jobs_before_fixpoint_mean: f32,
    pub mesh_jobs_before_fixpoint_p95: f32,
    pub mesh_jobs_before_fixpoint_n: u64,
}

/// Cumulative counters the flight bench and the stress gauges read.
#[derive(Default)]
pub(in crate::world) struct StreamCounters {
    /// Worker results integrated, and claim keys released by deschedule.
    /// The flight bench reads these; a quiet frame does not touch them.
    pub(in crate::world) jobs_completed: u64,
    pub(in crate::world) jobs_cancelled: u64,
    /// Chunks stored from a generate result, how many of those were already
    /// behind the player, and results dropped because the loading window had
    /// left them (the flight bench's wasted-work counts).
    pub(in crate::world) gen_landed: u64,
    pub(in crate::world) gen_landed_behind: u64,
    pub(in crate::world) gen_discarded: u64,
    /// Generation run-list rebuilds (the flight bench's regrow cost).
    pub(in crate::world) gen_cursor_rebuilds: u64,
    /// Cumulative light-worklist insert attempts (stress: seeds per chunk).
    pub(in crate::world) light_seed_inserts: u64,
    pub(in crate::world) light_seed_split: LightSeedSplit,
    /// Cumulative light jobs accepted by the worker pool.
    pub(in crate::world) light_admitted: u64,
    /// Jobs accepted by the most recent [`admit`](crate::world::admit)`<LightLane>` pass.
    pub(in crate::world) light_admitted_last: usize,
    /// Vertex bytes uploaded for sections in the current drain (harness peak).
    pub(in crate::world) section_upload_bytes: usize,
    /// Chunk + section vertex bytes uploaded in the current drain (harness peak).
    pub(in crate::world) drain_upload_bytes: usize,
}

impl World {
    /// World-entry completeness predicate: true once, within the view
    /// radius, every chunk shows a FINAL-light mesh (`Ready`/`Air`, none degraded
    /// and none still waiting on light), no near generate/mesh/light work is queued
    /// or in flight, and (under lod2) the section far field is covering-complete.
    /// Reads private streaming state — its home here.
    pub fn entry_complete(&self) -> bool {
        let Some(center) = self.center else {
            return false;
        };
        // No near work queued or in flight, and nothing owed a final-light remesh.
        if !self.near_quiescent()
            || !self.upload_queue.is_empty()
            || !self.light_gate.degraded.is_empty()
            || !self.light_gate.dirty.is_empty()
            || !self.light_gate.blocked_since.is_empty()
        {
            return false;
        }
        // Every in-view chunk has a final mesh (data loaded, not building/dirty).
        for coord in self.view_coords(self.mesh_box(center)) {
            match self.chunks.get(&coord).map(|l| &l.state) {
                Some(MeshState::Air | MeshState::Ready(_)) => {}
                _ => return false,
            }
        }
        // LOD2 far field: all desired cells covered and no uploads pending. Skipped
        // when disabled (no far field in near-only mode). A cell the settled chunks
        // already draw is not admitted, so it counts as done without a section mesh.
        if self.desired_unrefined().is_some() {
            if !self.section_upload_queue.is_empty() {
                return false;
            }
            if self
                .section_desired
                .iter()
                .any(|&c| !self.section_covered(c) && !self.full_res_covers(center, c))
            {
                return false;
            }
        }
        true
    }

    /// Near generate/mesh/light queues empty — the five-queue rest predicate.
    pub(super) fn near_quiescent(&self) -> bool {
        self.generating.is_empty()
            && self.mesh_worklist.is_empty()
            && self.light_worklist.is_empty()
            && self.light_inflight.is_empty()
            && self.light_apply_queue.is_empty()
            && self.light_owed.is_empty()
    }

    /// `None` when LOD2 is off. `Some(n)` is how many desired far cells lack
    /// their own Ready mesh.
    fn desired_unrefined(&self) -> Option<usize> {
        if !self.lod2 {
            return None;
        }
        Some(
            self.section_desired
                .iter()
                .filter(|c| !self.sections.get(c).is_some_and(|s| s.is_ready()))
                .count(),
        )
    }

    /// Every desired far-field section is itself Ready — the strongest far-field
    /// state. `entry_complete` accepts a Ready *ancestor* as covering (right for
    /// playability), but a coarse cover moves the horizon's pixels — and through
    /// the exposure meter, the whole frame's brightness — as refinement lands.
    /// The golden harness gates captures on this so blessed shots are the
    /// converged frame; gameplay never waits on it.
    pub fn far_field_refined(&self) -> bool {
        if self.center.is_none() {
            return false;
        }
        match self.desired_unrefined() {
            None => true,
            Some(n) => self.section_upload_queue.is_empty() && n == 0,
        }
    }

    /// How many desired far-field sections still lack their own mesh — the
    /// harness's progress signal while it waits on
    /// [`far_field_refined`](Self::far_field_refined).
    pub fn far_field_pending(&self) -> usize {
        if self.center.is_none() {
            return 0;
        }
        self.desired_unrefined().unwrap_or(0)
    }

    /// Snapshot the streaming-queue depths (see [`super::StreamGauges`]).
    pub fn stream_gauges(&self) -> super::StreamGauges {
        let (worker_near_queue, worker_far_queue, worker_capacity) = self
            .workers
            .as_ref()
            .map(|workers| {
                let (near, far) = workers.queue_depths();
                (near, far, workers.worker_capacity())
            })
            .unwrap_or_default();
        let staging = self
            .workers
            .as_ref()
            .map(pipeline::Workers::staging_snapshot)
            .unwrap_or_default();
        super::StreamGauges {
            chunks: self.chunks.len(),
            generating: self.generating.len(),
            mesh_worklist: self.mesh_worklist.len(),
            upload_queue: self.upload_queue.len(),
            light_worklist: self.light_worklist.len(),
            light_inflight: self.light_inflight.len(),
            light_apply_queue: self.light_apply_queue.len(),
            worker_near_queue,
            worker_far_queue,
            worker_capacity,
            travel_speed_mps: self.stream_pacer.speed_mps(),
            effort: self.stream_pacer.effort(),
            light_admitted: self.counters.light_admitted,
            light_admitted_last: self.counters.light_admitted_last,
            light_seed_inserts: self.counters.light_seed_inserts,
            mesh_slots: if self.gpu_live_slots != 0 {
                self.gpu_live_slots as usize
            } else {
                self.local_mesh_slots()
            },
            slot_ceiling: self.slot_ceiling as usize,
            section_ready: self.sections.values().filter(|s| s.is_ready()).count(),
            light_seed_split: self.counters.light_seed_split,
            remesh_async_calls: self.remesh_stats.remesh_async_calls,
            drop_stale_uploads: self.remesh_stats.drop_stale_uploads,
            drop_stale_this_frame: self.remesh_stats.drop_stale_this_frame,
            section_upload_bytes: self.counters.section_upload_bytes,
            drain_upload_bytes: self.counters.drain_upload_bytes,
            mesh_staged: staging.chunk_staged,
            mesh_fallback: staging.chunk_fallback,
            mesh_ring_full: staging.chunk_ring_full,
            section_staged: staging.section_staged,
            section_fallback: staging.section_fallback,
            section_ring_full: staging.section_ring_full,
            reactions_pending: self.reactions.pending(),
            reactions_mutations: self.reactions.operations,
        }
    }

    /// The remesh distributions so far. Sorts up to 65,536 samples per
    /// distribution, so it belongs at report time, not in a frame.
    pub fn remesh_distribution(&self) -> RemeshDistribution {
        let (ru_mean, ru_p95, ru_n) = self.remesh_stats.between_upload_mean_p95();
        let (jf_mean, jf_p95, jf_n) = self.remesh_stats.jobs_before_fixpoint_mean_p95();
        RemeshDistribution {
            remesh_between_upload_mean: ru_mean,
            remesh_between_upload_p95: ru_p95,
            remesh_between_upload_n: ru_n,
            mesh_jobs_before_fixpoint_mean: jf_mean,
            mesh_jobs_before_fixpoint_p95: jf_p95,
            mesh_jobs_before_fixpoint_n: jf_n,
        }
    }

    /// Human-readable reason `entry_complete` is not yet true — the first
    /// unsatisfied clause with a count, so a stalled bless/harness run says WHICH
    /// streaming stage is stuck instead of hanging silently. Clause order mirrors
    /// [`entry_complete`](Self::entry_complete).
    pub fn entry_debug(&self) -> String {
        if let Some(slab) = self.spawn_slab {
            let missing = self
                .view_coords(slab)
                .filter(|c| !self.chunks.contains_key(c))
                .count();
            if missing > 0 {
                return format!(
                    "spawn slab loading: {missing} chunks, generating={}",
                    self.generating.len()
                );
            }
        }
        let Some(center) = self.center else {
            return "no stream centre yet".into();
        };
        if !self.quarantined.is_empty() {
            return format!(
                "{} claim(s) quarantined after repeated worker panics: {:?}",
                self.quarantined.len(),
                self.quarantined.iter().take(4).collect::<Vec<_>>()
            );
        }
        // Share the one queue-depth source with the harness gauge, so the two
        // can never drift; the gate counters have no gauge field, so stay local.
        let g = self.stream_gauges();
        let near: [(&str, usize); 10] = [
            ("generating", g.generating),
            ("mesh_worklist", g.mesh_worklist),
            ("upload_queue", g.upload_queue),
            ("light_worklist", g.light_worklist),
            ("light_inflight", g.light_inflight),
            ("light_apply_queue", g.light_apply_queue),
            ("degraded", self.light_gate.degraded.len()),
            ("light_dirty", self.light_gate.dirty.len()),
            ("terminal", self.light_terminal.len()),
            ("light_blocked", self.light_gate.blocked_since.len()),
        ];
        let pending: Vec<String> = near
            .iter()
            .filter(|(_, n)| *n != 0)
            .map(|(k, n)| format!("{k}={n}"))
            .collect();
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
                    } else if !self.neighbours_have_data(c) && !self.light_terminal.contains(&c) {
                        no_neigh += 1;
                    } else if self.light_ready(c)
                        || self.light_wait_expired(c)
                        || self.light_terminal.contains(&c)
                    {
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
        // Terminal wedge: near-work queues empty but some in-box chunk not final.
        // Tally by state and (for idle NeedsMesh) by which gate would block.
        let (mut missing, mut idle, mut building, mut dirty, mut queued) = (0, 0, 0, 0, 0);
        let (mut idle_no_neigh, mut idle_unlit) = (0, 0);
        for c in self.view_coords(self.mesh_box(center)) {
            let in_wl = self.mesh_worklist.contains(&c);
            match self.chunks.get(&c).map(|l| &l.state) {
                Some(MeshState::Air | MeshState::Ready(_)) => {}
                None => missing += 1,
                Some(MeshState::Dirty { .. }) => dirty += 1,
                Some(MeshState::NeedsMesh { building: true, .. }) => building += 1,
                Some(MeshState::NeedsMesh {
                    building: false, ..
                }) => {
                    if in_wl {
                        queued += 1;
                    } else {
                        idle += 1;
                        if !self.neighbours_have_data(c) && !self.light_terminal.contains(&c) {
                            idle_no_neigh += 1;
                        } else if !(self.light_ready(c)
                            || self.light_wait_expired(c)
                            || self.light_terminal.contains(&c))
                        {
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
            let uncovered = self
                .section_desired
                .iter()
                .filter(|&&c| !self.section_covered(c) && !self.full_res_covers(center, c))
                .count();
            if uncovered != 0 {
                return format!(
                    "column sections uncovered: {uncovered} of {} desired",
                    self.section_desired.len()
                );
            }
        }
        "entry complete".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The per-frame gauge snapshot neither sorts nor allocates, however many remesh samples the
    /// world holds: the distributions come from `remesh_distribution`, at report time.
    #[test]
    fn stream_gauges_do_not_allocate() {
        let mut world = World::with_config_lazy(1, crate::render_config::RenderConfig::default());
        for x in 0..1000 {
            let c = Coord::new(x, 0, 0);
            world.remesh_stats.note_upload(c);
            world.remesh_stats.note_mesh_job(c, true);
        }
        crate::alloc_count::reset();
        let gauges = world.stream_gauges();
        assert_eq!(crate::alloc_count::alloc_count(), 0, "a gauge snapshot allocated");
        assert_eq!(gauges.chunks, 0);
        let remesh = world.remesh_distribution();
        assert_eq!((remesh.remesh_between_upload_n, remesh.mesh_jobs_before_fixpoint_n), (1000, 1000));
    }
}
