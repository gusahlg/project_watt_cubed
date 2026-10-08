//! Payload-less claim resolution: cancelled and panicked worker jobs, strikes and quarantine.

use super::*;

/// Panics tolerated per claim before it is quarantined. A panic is a real bug
/// in job code, usually deterministic for one input — retrying a couple of
/// times absorbs flukes (allocation pressure, a racing palette snapshot)
/// without looping forever on poison.
const MAX_JOB_STRIKES: u8 = 3;

/// Why a claim is being resolved WITHOUT a payload — see
/// [`World::resolve_claim`]. Cancelled: descheduled at the pool, no strike.
/// Failed: the job panicked; strikes accumulate toward quarantine.
enum ClaimOutcome {
    Cancelled,
    Failed,
}

impl World {
    /// A queued job was DESCHEDULED at the pool: its region left the live view
    /// while it waited (fast movement). Release the exact claim with no strike.
    /// Work the player has left is not requeued — the boundary-cross scans
    /// re-request it on return. A mesh still inside the mesh box is re-seeded:
    /// that scan misses a cancel that lands after it. A still-loaded chunk the
    /// loading window still wants is owed its light settle, so that claim re-seeds.
    pub(in crate::world) fn cancel_job(&mut self, key: pipeline::JobKey) {
        self.resolve_claim(key, ClaimOutcome::Cancelled);
    }

    /// A worker job PANICKED: release its exact claim so streaming can
    /// converge, then retry (the normal scans re-request freed work) up to
    /// [`MAX_JOB_STRIKES`] times. Past that the claim is quarantined — a
    /// bounded hole instead of an infinite panic loop — and every enqueue path
    /// skips it via `quarantined`.
    pub(in crate::world) fn fail_job(&mut self, key: pipeline::JobKey) {
        self.resolve_claim(key, ClaimOutcome::Failed);
    }

    /// The ONE payload-less claim-resolution path (cancel and fail shared the
    /// whole per-kind release; only the strike/re-arm policy differed).
    /// RELEASE is unconditional per kind; RE-ARM follows `outcome`: a
    /// cancellation re-arms the light settle it still owes and a mesh still
    /// inside the mesh box, a non-quarantined failure re-arms its lane.
    fn resolve_claim(&mut self, key: pipeline::JobKey, outcome: ClaimOutcome) {
        let rearm = match outcome {
            ClaimOutcome::Cancelled => matches!(key, pipeline::JobKey::Light { .. }),
            ClaimOutcome::Failed => {
                let fail_key = FailKey::of(&key);
                let strikes = self.job_strikes.entry(fail_key).or_insert(0);
                *strikes = strikes.saturating_add(1);
                let quarantine = *strikes >= MAX_JOB_STRIKES;
                if quarantine {
                    self.quarantined.insert(fail_key);
                    eprintln!(
                        "streaming: {fail_key:?} panicked {MAX_JOB_STRIKES} times — quarantined"
                    );
                }
                !quarantine
            }
        };
        match key {
            pipeline::JobKey::Column { key, range } => {
                self.release_run(GenRun::Column { key, lo: *range.start(), hi: *range.end() }, rearm);
            }
            pipeline::JobKey::Open { coord } => self.release_run(GenRun::Open { coord }, rearm),
            pipeline::JobKey::Mesh { coord } => {
                if let Some(loaded) = self.chunks.get_mut(&coord) {
                    if loaded.state.release_build() {
                        super::adjust_count(&mut self.building_meshes, true, false);
                    }
                }
                // The shell scan cannot see a cancel that lands after it, and nothing else
                // re-seeds an interior chunk. Outside the box the work stays unqueued.
                if rearm || self.in_mesh_box(coord) {
                    self.seed_mesh(coord);
                    self.pending_fresh.set();
                }
            }
            pipeline::JobKey::Light { coord } => {
                self.light_inflight.remove(&coord);
                // Outside the loading window the settle is owed rather than
                // re-seeded: reseeding a cancel would queue the same job forever.
                if rearm && self.admits_light(coord) {
                    self.seed_light(coord, super::LightSeed::Store);
                    self.light_pending.set();
                } else if rearm {
                    self.owe_light(coord);
                }
                // Quarantined light: the chunk never settles, so the mesh
                // lane's degrade timeout takes over and the terminal flush
                // promotes it — the world converges on fallback light.
            }
            pipeline::JobKey::Section { pos, epoch, token } => {
                // Release only the exact claim: a same-position replacement
                // minted after this job was queued keeps its own claim.
                let held = epoch == self.section_epoch
                    && matches!(self.sections.get(&pos),
                        Some(SectionState::Meshing { token: t }) if *t == token);
                if held {
                    self.sections.remove(&pos);
                    super::adjust_count(&mut self.meshing_sections, true, false);
                    self.section_cover_dirty.set();
                }
                if rearm {
                    self.pending_sections.set();
                }
            }
        }
    }

    /// Release a generate run's claims.
    fn release_run(&mut self, run: GenRun, rearm: bool) {
        // A cancelled spawn-slab run must be re-requested even when the pool
        // dropped it as out-of-view: physics is frozen on it.
        let in_slab = self.spawn_slab.is_some_and(|slab| run.coords().any(|c| slab.contains(c)));
        for c in run.coords() {
            self.generating.remove(&c);
        }
        // Freed generate claims are otherwise only re-requested on a
        // boundary cross; a retryable failure re-arms the lane so a
        // standing-still player still converges.
        if rearm || in_slab {
            self.pending_gen.set();
            // The run already left the queue when it was submitted.
            self.gen_cursor.dirty = true;
        }
    }
}
