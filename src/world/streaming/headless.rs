//! The stream pass without a GPU, for the tests and the headless benches: chunk uploads land as
//! one fake handle, sections as empty `Ready` states, frees go to the test log and occlusion
//! touches no engine. [`Laps`] times each [`Phase`] of the real pass.

use std::time::{Duration, Instant};

use voxel_engine::{MeshHandle, MeshStager, Pass};

use super::drain::Landing;
use super::*;
use crate::world::mesh_free_log;

/// Headless [`StreamSteps`], timing each phase into `laps` when set.
#[derive(Default)]
pub(in crate::world) struct Headless {
    pub(in crate::world) laps: Option<Laps>,
}

impl Headless {
    pub(in crate::world) fn timed() -> Self {
        Self { laps: Some(Laps::default()) }
    }
}

impl Landing for Headless {
    fn chunk(&mut self, world: &mut World, coord: Coord, data: pipeline::MeshPayload) {
        let drawn = data.vertex_bytes() > 0;
        let handles = ByPass::from_fn(|p| (drawn && p == Pass::Opaque).then(|| MeshHandle::from_raw_parts(1, 1)));
        let visible = !world.occlusion_active || world.occlusion.is_visible(coord);
        if let Some(loaded) = world.chunks.get_mut(&coord) {
            let was = loaded.state.is_building();
            loaded.retire_logged(MeshState::from_upload(handles));
            loaded.mesh_hash = None;
            loaded.visible = visible;
            adjust_count(&mut world.building_meshes, was, false);
        }
        world.note_settled();
    }

    fn section(&mut self, _world: &mut World, _pos: SectionPos, _data: pipeline::SectionPayload) -> SectionState {
        crate::world::fixtures::ready_section()
    }
}

impl StreamSteps for Headless {
    fn begin(&mut self, world: &mut World) -> Option<MeshStager> {
        // A real engine reports its live slots each pass; headless sections carry none.
        world.gpu_live_slots = world.local_mesh_slots() as u32;
        mesh_free_log::take();
        if let Some(laps) = &mut self.laps {
            laps.start();
        }
        None
    }

    fn textures(&mut self, _world: &mut World) {}

    fn drain(&mut self, world: &mut World) {
        world.drain_with(lanes::duration(lanes::DrainLane::BUDGET), self);
    }

    fn remesh_dirty(&mut self, world: &mut World) {
        assert!(!world.dirty_pending(), "a headless pass remeshes no edit");
    }

    fn unload(&mut self, world: &mut World, center: Coord) {
        world.unload_far_with(center, |state, _| state.free_logged());
    }

    fn retire(&mut self, loaded: &mut Loaded, next: MeshState) {
        loaded.retire_logged(next);
    }

    fn unload_sections(&mut self, world: &mut World, far: Coord) {
        world.unload_sections_with(far, |_| {});
    }

    fn remesh_sections(&mut self, world: &mut World) {
        // Headless worlds edit before they stream, so an edited section is never Ready: there
        // is nothing to free, and its mark clears when the section is claimed.
        assert!(
            world.dirty_sections.iter().all(|s| !world.sections.get(s).is_some_and(SectionState::is_ready)),
            "a headless pass frees no edited section"
        );
    }

    fn reclaim_sections(&mut self, world: &mut World, far: Coord) {
        world.reclaim_blocked_sections(far, None);
    }

    fn section_visible(&mut self, world: &mut World) {
        world.rebuild_section_visible(None);
    }

    fn occlusion(&mut self, world: &mut World) {
        world.rebuild_occlusion(None, lanes::OcclusionLane::BUDGET);
    }

    fn lap(&mut self, phase: Phase) {
        if let Some(laps) = &mut self.laps {
            laps.lap(phase);
        }
    }
}

/// How many [`Phase`]s a pass has.
const PHASES: usize = Phase::Occlusion as usize + 1;

/// [`Phase`] names, for reports.
pub(in crate::world) const PHASE_NAMES: [&str; PHASES] = [
    "begin",
    "drain",
    "unload",
    "cross",
    "generate",
    "light",
    "mesh",
    "lod_face",
    "frontier",
    "sec_unload",
    "reclaim",
    "sec_admit",
    "sec_visible",
    "occlusion",
];

/// Wall time per [`Phase`], summed over every pass and over full passes only, and the last pass.
pub(in crate::world) struct Laps {
    at: Instant,
    start: Instant,
    pass: [Duration; PHASES],
    full_pass: bool,
    pub(in crate::world) sum: [Duration; PHASES],
    pub(in crate::world) full: [Duration; PHASES],
    /// The last finished pass: its wall time and whether it was a full pass.
    pub(in crate::world) last: (Duration, bool),
}

impl Default for Laps {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            at: now,
            start: now,
            pass: [Duration::ZERO; PHASES],
            full_pass: false,
            sum: [Duration::ZERO; PHASES],
            full: [Duration::ZERO; PHASES],
            last: (Duration::ZERO, false),
        }
    }
}

impl Laps {
    fn start(&mut self) {
        self.start = Instant::now();
        self.at = self.start;
        self.pass = [Duration::ZERO; PHASES];
        self.full_pass = false;
    }

    fn lap(&mut self, phase: Phase) {
        let now = Instant::now();
        self.pass[phase as usize] += now - self.at;
        self.at = now;
        self.full_pass |= phase == Phase::Unload;
        if phase == Phase::Occlusion {
            for (i, d) in self.pass.iter().enumerate() {
                self.sum[i] += *d;
                if self.full_pass {
                    self.full[i] += *d;
                }
            }
            self.last = (now - self.start, self.full_pass);
        }
    }

    /// Forget the sums (the passes before a measured run).
    pub(in crate::world) fn reset(&mut self) {
        self.sum = [Duration::ZERO; PHASES];
        self.full = [Duration::ZERO; PHASES];
    }
}

/// One headless pass at `eye`.
pub(in crate::world) fn step(world: &mut World, eye: DVec3) {
    world.stream_steps(eye, &mut Headless::default());
}

/// [`step`], then [`finish_jobs`]: a pass bounded by the work it started rather than by the
/// wall clock, so a starved worker pool slows the pass down but cannot change what lands.
pub(in crate::world) fn step_finished(world: &mut World, eye: DVec3) {
    step(world, eye);
    finish_jobs(world, |_, _| {});
}

/// No result for this long while a claim waits on one is a stuck claim, not a slow pool.
const STUCK: Duration = Duration::from_secs(120);

/// Block until every job a claim waits on has landed, integrating each result through the
/// claim chokepoint (`seen` looks at it first).
pub(in crate::world) fn finish_jobs(world: &mut World, mut seen: impl FnMut(&World, &pipeline::Done)) {
    let mut last = Instant::now();
    while owed(world) {
        match world.workers.as_ref().and_then(pipeline::Workers::try_recv) {
            Some(done) => {
                seen(world, &done);
                world.integrate_worker_result(done);
                last = Instant::now();
            }
            None => {
                assert!(last.elapsed() < STUCK, "no job landed for {STUCK:?}: {}", world.entry_debug());
                std::thread::sleep(Duration::from_micros(100));
            }
        }
    }
}

/// Whether a claim still waits on a worker: one not yet carried into an upload or apply queue.
fn owed(world: &World) -> bool {
    world.workers.is_some()
        && (!world.generating.is_empty()
            || world.light_inflight.len() > world.light_apply_queue.len()
            || world.building_meshes > world.upload_queue.len()
            || world.meshing_sections > world.section_upload_queue.len())
}
