//! Stream pacing: the velocity-aware load controller, the speed-reduced loading window, and the
//! view the worker pool is told.

use super::*;

/// Square-root loading radius through this speed, then inverse. 100 m/s stays
/// on the wide shoulder; past it the radius falls with speed.
const LOAD_SQRT_KNEE_MPS: f64 = 120.0;

/// A sample this many times faster than the last accepted speed (and well
/// above walking) waits for the next sample to confirm it. A teleport or a
/// warp moves the eye once; real flight keeps moving.
const SPEED_JUMP_RATIO: f64 = 4.0;

/// The held heading axis stays until another axis leads it by this factor, so
/// a near-diagonal flight does not flip the heading (and rescan every queue)
/// each sample.
const HEADING_HYSTERESIS: f64 = 1.2;

/// Travel up to this speed gets the full streaming budget. It is comfortably
/// above ordinary walking/sprinting, so normal play and world entry retain
/// maximum convergence speed. Above it, useful chunk lifetime falls roughly
/// inversely with velocity, and effort follows the same curve.
pub(super) const FULL_EFFORT_SPEED_MPS: f64 = 24.0;

/// Keep a small progress floor even during extreme travel. Stopping is never
/// required for the centre/collision neighbourhood to advance, while the cap
/// leaves most CPU and transfer time to the frame loop.
const MIN_STREAM_EFFORT: f32 = 0.15;

/// Once travel stops, restore background capacity over this time constant.
/// Shedding is immediate (a hitch should stop now); recovery is deliberately
/// damped so the first stationary frame cannot release a catch-up avalanche.
const STREAM_RECOVERY_SECS: f64 = 0.75;

/// Last topology pass cheaper than this: leftover light/mesh work at rest may
/// run at full worker/admission capacity. Half a 60 Hz frame — the post-flight
/// frames on this branch sit well below it, while an already-expensive pass
/// keeps travel shedding.
const STREAM_HEADROOM_SECS: f64 = 0.008;

/// Hysteresis: once rest-boosted, stay boosted until a pass exceeds this so a
/// slightly heavier first stationary admit cannot immediately re-shed.
const STREAM_HEADROOM_EXIT_SECS: f64 = 0.014;

/// Near-queue cap at rest when leftover light/mesh work remains. Travel keeps
/// `active * 4` because queued jobs go stale; at rest they will still be wanted,
/// and cheap light jobs otherwise idle the pool for the rest of a 16 ms frame.
/// Same order as [`pipeline::FAR_QUEUE_CAP`].
const NEAR_REST_QUEUE_CAP: usize = 256;

/// A real mesh upload always makes progress even when the scaled byte budget is
/// tiny. Most chunks fit below this; an unusually large first mesh is allowed
/// to overrun it once, just as it may overrun the normal byte budget once.
const MIN_UPLOAD_BUDGET_BYTES: usize = 256 << 10;

/// Minimum completed results integrated per drain before its time budget may
/// stop it. This releases claims promptly without reverting to the old
/// unbounded channel drain.
pub(super) const RESULT_INTEGRATE_FLOOR: usize = 8;

/// Velocity-aware streaming load controller. `effort` bounds main-thread
/// admission, result integration and GPU uploads. The loading window
/// (`load_fraction` of the view radius) is what shrinks with speed, so every
/// worker stays on the nearest chunks ahead of the player instead of a few
/// workers starving on the whole view. `boost` is the rest-time override:
/// leftover light/mesh work on a frame with headroom runs at full admission
/// so the travel floor cannot idle a catch-up.
#[derive(Clone, Copy, Debug)]
pub(in crate::world) struct StreamPacer {
    speed_mps: f64,
    /// Last accepted near-eye velocity, in the frame the near window streams
    /// in, including speeds the far-field predictor treats as a teleport. The
    /// loading window aims with this.
    travel: DVec3,
    effort: f32,
    /// Smoothed fraction of the view radius new work may cover. `1` at rest.
    load_fraction: f32,
    boost: bool,
    /// The last sample was a speed jump waiting for confirmation.
    held: bool,
}

impl Default for StreamPacer {
    fn default() -> Self {
        Self {
            speed_mps: 0.0,
            travel: DVec3::ZERO,
            effort: 1.0,
            load_fraction: 1.0,
            boost: false,
            held: false,
        }
    }
}

impl StreamPacer {
    /// The useful-work fraction at `speed_mps`. Inverse scaling models the
    /// shrinking time a chunk remains in view; it is continuous at the full
    /// effort threshold and bounded away from zero for forward progress.
    fn target_effort(speed_mps: f64) -> f32 {
        if speed_mps.is_nan() || speed_mps <= FULL_EFFORT_SPEED_MPS {
            return 1.0;
        }
        if speed_mps.is_infinite() {
            return MIN_STREAM_EFFORT;
        }
        (FULL_EFFORT_SPEED_MPS / speed_mps).max(f64::from(MIN_STREAM_EFFORT)) as f32
    }

    /// Fraction of the view radius new work may cover. Full at walking speed,
    /// wide at 100 m/s (the leading face is then a small share of the window),
    /// a few chunks near 600 m/s, almost nothing at several km/s. Square root
    /// through [`LOAD_SQRT_KNEE_MPS`], then inverse, so the two fast-flight
    /// speeds do not keep a window the main thread cannot service.
    fn target_load_fraction(speed_mps: f64) -> f32 {
        if speed_mps.is_nan() || speed_mps <= FULL_EFFORT_SPEED_MPS {
            return 1.0;
        }
        if !speed_mps.is_finite() {
            return 0.0;
        }
        if speed_mps <= LOAD_SQRT_KNEE_MPS {
            return (FULL_EFFORT_SPEED_MPS / speed_mps).sqrt() as f32;
        }
        let at_knee = (FULL_EFFORT_SPEED_MPS / LOAD_SQRT_KNEE_MPS).sqrt();
        (at_knee * LOAD_SQRT_KNEE_MPS / speed_mps) as f32
    }

    /// Shrink the moment speed rises so a jump does not admit the window it
    /// just left. Grow back on the effort-recovery time constant so stopping
    /// cannot release the whole view in one frame. A steady speed holds one
    /// integer radius.
    fn smooth_load(&mut self, sample_dt: f64) {
        let target = Self::target_load_fraction(self.speed_mps);
        if sample_dt <= 0.0 || target <= self.load_fraction {
            self.load_fraction = target;
            return;
        }
        let dt = sample_dt.min(MAX_PREDICT_SAMPLE_GAP);
        let alpha = 1.0 - (-dt / STREAM_RECOVERY_SECS).exp();
        self.load_fraction += (target - self.load_fraction) * alpha as f32;
        if (target - self.load_fraction).abs() < 0.001 {
            self.load_fraction = target;
        }
    }

    /// Integer loading radius for a view radius of `full` chunks.
    pub(super) fn load_radius(self, full: i32) -> i32 {
        if full <= 0 {
            return 0;
        }
        let r = (full as f32 * self.load_fraction).round();
        if r <= 0.0 {
            0
        } else if r >= full as f32 {
            full
        } else {
            r as i32
        }
    }

    /// Feed one near-eye sample. `discontinuity` (the near window's frame
    /// changed under the eye: a chart seam, an altitude band, entering or
    /// leaving a chart) drops it: its velocity mixes two frames. A jump to a
    /// speed the last accepted one does not support is held and applies only
    /// when the next sample is fast too.
    pub(super) fn observe(&mut self, velocity: DVec3, sample_dt: f64, discontinuity: bool) {
        if discontinuity {
            self.held = false;
            return;
        }
        let speed = velocity.x.hypot(velocity.y).hypot(velocity.z);
        if speed > SPEED_JUMP_RATIO * self.speed_mps.max(FULL_EFFORT_SPEED_MPS) && !self.held {
            self.held = true;
            return;
        }
        self.held = false;
        self.update(velocity, sample_dt);
    }

    pub(in crate::world) fn update(&mut self, velocity: DVec3, sample_dt: f64) {
        // `hypot(x, 0) = |x|` and `hypot` is even, so `vy == 0` matches the
        // old horizontal speed bit for bit.
        self.travel = velocity;
        self.speed_mps = velocity.x.hypot(velocity.y).hypot(velocity.z);
        self.smooth_load(sample_dt);
        let target = Self::target_effort(self.speed_mps);
        if target <= self.effort {
            // Load shedding has to beat the next expensive frame.
            self.effort = target;
            return;
        }
        // Recovery is a time-based exponential, independent of stream_hz.
        let dt = sample_dt.clamp(0.0, MAX_PREDICT_SAMPLE_GAP);
        let alpha = 1.0 - (-dt / STREAM_RECOVERY_SECS).exp();
        self.effort += (target - self.effort) * alpha as f32;
        if (target - self.effort).abs() < 0.001 {
            self.effort = target;
        }
    }

    /// Rest-time override: full admission deadlines and the deep near-queue cap
    /// while light/mesh work remains, the eye is at walking speed, and the last
    /// pass had frame-time headroom. Travel keeps the reduced deadlines.
    pub(super) fn set_boost(&mut self, queued_near: bool, last_stream_secs: f64) {
        let at_rest = self.speed_mps <= FULL_EFFORT_SPEED_MPS;
        if !queued_near || !at_rest {
            self.boost = false;
            return;
        }
        if self.boost {
            self.boost = last_stream_secs < STREAM_HEADROOM_EXIT_SECS;
        } else {
            self.boost = last_stream_secs < STREAM_HEADROOM_SECS;
        }
    }

    pub(in crate::world) fn boosting(self) -> bool {
        self.boost
    }

    /// A speed jump waits for confirmation: the next pass may collapse the
    /// loading window.
    pub(in crate::world) fn holding(self) -> bool {
        self.held
    }

    /// Effort applied to admission deadlines, floors and uploads. The worker
    /// count does not follow it.
    fn applied_effort(self) -> f32 {
        if self.boost { 1.0 } else { self.effort }
    }

    pub(in crate::world) fn effort(self) -> f32 {
        self.effort
    }

    pub(in crate::world) fn speed_mps(self) -> f64 {
        self.speed_mps
    }

    pub(in crate::world) fn travel(self) -> DVec3 {
        self.travel
    }

    pub(in crate::world) fn duration(self, base: Duration) -> Duration {
        base.mul_f32(self.applied_effort())
    }

    pub(in crate::world) fn floor(self, base: usize) -> usize {
        ((base as f32 * self.applied_effort()).ceil() as usize).clamp(1, base.max(1))
    }

    pub(super) fn upload_bytes(self) -> usize {
        ((UPLOAD_BUDGET_BYTES as f32 * self.applied_effort()) as usize).max(MIN_UPLOAD_BUDGET_BYTES)
    }

    pub(super) fn section_uploads(self) -> usize {
        ((SECTION_UPLOAD_BUDGET as f32 * self.applied_effort()).round() as usize)
            .clamp(1, SECTION_UPLOAD_BUDGET)
    }

    /// Every worker stays available. Speed shrinks the loading window, not the
    /// pool: a few workers on the full radius never finish the chunks that matter.
    fn active_workers(self, capacity: usize) -> usize {
        capacity.max(1)
    }

    /// Near-queue lookahead. Travel keeps `capacity * 4` so a short queue can
    /// still be dropped when the window moves; at rest the deeper cap keeps
    /// cheap light jobs from idling the pool.
    fn near_queue_cap(self, capacity: usize) -> usize {
        let travel = (capacity.max(1) * 4).max(8);
        if self.boost {
            travel.max(NEAR_REST_QUEUE_CAP)
        } else {
            travel
        }
    }
}

/// A chunk is strictly behind the direction of travel. Rest, walking, and a
/// mostly vertical climb do not count: the whole window stays wanted, and a
/// climb must not drop the ground. `coord` is already in the same net as
/// `center`. The up-axis component is ignored, so distance along the face
/// never looks like trailing the player.
pub(in crate::world) fn chunk_behind(
    center: Coord,
    coord: Coord,
    vel: DVec3,
    up: Option<Face>,
) -> bool {
    let mut v = [vel.x, vel.y, vel.z];
    let mut d = [
        (i64::from(coord.x) - i64::from(center.x)) as f64,
        (i64::from(coord.y) - i64::from(center.y)) as f64,
        (i64::from(coord.z) - i64::from(center.z)) as f64,
    ];
    if let Some(face) = up {
        let a = face.axis();
        v[a] = 0.0;
        d[a] = 0.0;
    }
    let speed2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    let along = d[0] * v[0] + d[1] * v[1] + d[2] * v[2];
    // `along / speed < -0.5` without the square root. Non-finite input fails
    // a comparison and keeps the chunk.
    speed2 > FULL_EFFORT_SPEED_MPS * FULL_EFFORT_SPEED_MPS
        && along < 0.0
        && along * along > 0.25 * speed2
}

/// Dominant-axis sign of planar travel: `+1`/`-1` on X, `+2`/`-2` on Y, `+3`/`-3`
/// on Z, `0` at rest. A straight flight holds one value, so the job gate's epoch
/// does not bump every frame; a reversal does, and behind jobs drop immediately.
/// The `held` axis is kept until another leads it by [`HEADING_HYSTERESIS`].
pub(super) fn travel_heading(vel: DVec3, up: Option<Face>, held: i8) -> i8 {
    let mut v = [vel.x, vel.y, vel.z];
    if let Some(face) = up {
        v[face.axis()] = 0.0;
    }
    let mut axis = 0usize;
    for i in 1..3 {
        if v[i].abs() > v[axis].abs() {
            axis = i;
        }
    }
    let mag = v[axis].abs();
    if !(mag > FULL_EFFORT_SPEED_MPS) {
        return 0;
    }
    if held != 0 {
        let h = usize::from(held.unsigned_abs()) - 1;
        if v[h].abs() * HEADING_HYSTERESIS >= mag {
            axis = h;
        }
    }
    let sign = if v[axis] >= 0.0 { 1i8 } else { -1 };
    sign * (axis as i8 + 1)
}

/// A speed-reduced loading window, copied out of the world so a worklist can
/// prune itself while borrowed.
#[derive(Clone, Copy)]
pub(super) struct LoadWindow {
    pub(super) center: Coord,
    pub(super) mesh: ChunkBox,
    pub(super) data: ChunkBox,
    pub(super) travel: DVec3,
    pub(super) up: Option<Face>,
    /// A heading is set: chunks strictly behind the player take no new work.
    pub(super) trail: bool,
}

impl LoadWindow {
    /// Whether new work at `folded` (a chunk already placed in the centre's
    /// net) is worth starting: inside the mesh box, or the data box for
    /// generation and light, and not behind the player.
    pub(super) fn covers(self, folded: Coord, data: bool) -> bool {
        let b = if data { self.data } else { self.mesh };
        b.contains(folded) && !(self.trail && chunk_behind(self.center, folded, self.travel, self.up))
    }
}

impl World {
    /// The worker-pool half of [`begin_stream`](Self::begin_stream): the loading radii, then the
    /// stager, the spawn slab, the live view and the pacing the pool runs on.
    pub(super) fn publish_view(
        &mut self,
        center_chunk: Coord,
        far_chunk: Coord,
        stager: Option<voxel_engine::MeshStager>,
    ) {
        // Publish the live view to the worker pool: queued jobs re-key toward
        // the player's CURRENT position on every view change, and entries left
        // behind by fast movement — far sections included — are descheduled
        // instead of run. The far horizon covers the whole frontier plus the
        // velocity lookahead, so prediction-desired sections survive it.
        let speed3 = self.section_vel.x.hypot(self.section_vel.y).hypot(self.section_vel.z);
        let far_m = self.far_horizon() + speed3 * TAU_STREAM;
        let far_view = self.far_view(far_chunk);
        // Configure the pool before any lane can submit this frame. On the
        // first stream this avoids one permissive/full-capacity burst from a
        // lazily spawned pool before the pacer catches it on the next pass.
        self.apply_loading_radius();
        let pacer = self.stream_pacer;
        let up = self.live_up();
        let load_h = self.load_h;
        let load_v = self.load_v;
        let load_heading = self.load_heading;
        let horizontal = self.view.horizontal;
        let section_vel = self.section_vel;
        let tight = !self.loading_full();
        let slab = self.spawn_slab;
        let fold = self.fold;
        let workers = self.worker_pool();
        if let Some(stager) = stager {
            workers.set_stager(stager);
        }
        workers.set_slab(slab);
        if tight {
            // The far predictor zeros velocity above its teleport cap. The near
            // window still aims with the real travel, or nothing at several
            // km/s would know which way is ahead.
            let travel = pacer.travel();
            workers.set_load_view(
                center_chunk.x,
                center_chunk.y,
                center_chunk.z,
                far_view,
                load_h,
                load_v,
                super::DATA_MARGIN,
                load_heading,
                far_m,
                travel.x,
                travel.y,
                travel.z,
                up,
                fold,
            );
        } else {
            workers.set_view(
                center_chunk.x,
                center_chunk.y,
                center_chunk.z,
                far_view,
                horizontal,
                far_m,
                section_vel.x,
                section_vel.y,
                section_vel.z,
                up,
                fold,
            );
        }
        let capacity = workers.worker_capacity();
        workers.set_pacing(
            pacer.active_workers(capacity),
            pacer.near_queue_cap(capacity),
        );
    }

    /// The view a spawn request publishes: at rest around near centre `c` and far centre `f`, with
    /// the collision slab.
    pub(super) fn publish_spawn_view(&mut self, c: Coord, f: Coord, up: Option<Face>, slab: ChunkBox) {
        let (far_m, far_view) = (self.far_horizon(), self.far_view(f));
        let view_r = self.view.horizontal;
        let fold = self.fold;
        let workers = self.worker_pool();
        workers.set_view(c.x, c.y, c.z, far_view, view_r, far_m, 0.0, 0.0, 0.0, up, fold);
        workers.set_slab(Some(slab));
    }

    /// A view at rest, published the way the far-field tests stand it.
    #[cfg(test)]
    pub(super) fn publish_rest_view(
        &mut self,
        c: Coord,
        far_view: pipeline::FarView,
        radius: i32,
        far_m: f64,
        up: Option<Face>,
        fold: seam::Unfold,
    ) {
        let workers = self.worker_pool();
        workers.set_view(c.x, c.y, c.z, far_view, radius, far_m, 0.0, 0.0, 0.0, up, fold);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_pacer_scales_with_useful_chunk_lifetime_and_recovers_gradually() {
        assert_eq!(StreamPacer::target_effort(0.0), 1.0);
        assert_eq!(StreamPacer::target_effort(FULL_EFFORT_SPEED_MPS), 1.0);
        assert!((StreamPacer::target_effort(48.0) - 0.5).abs() < f32::EPSILON);
        assert_eq!(StreamPacer::target_effort(10_000.0), MIN_STREAM_EFFORT);
        assert_eq!(StreamPacer::target_effort(f64::INFINITY), MIN_STREAM_EFFORT);
        assert_eq!(StreamPacer::target_effort(f64::NAN), 1.0);

        let mut pacer = StreamPacer::default();
        pacer.update(DVec3::new(200.0, 0.0, 0.0), 1.0 / 60.0);
        assert_eq!(pacer.effort(), MIN_STREAM_EFFORT, "shedding is immediate");
        assert_eq!(pacer.active_workers(12), 12, "speed shrinks the window, not the pool");
        assert_eq!(pacer.floor(32), 5);
        assert_eq!(pacer.section_uploads(), 1);

        pacer.update(DVec3::ZERO, 0.1);
        assert!(
            pacer.effort() > MIN_STREAM_EFFORT && pacer.effort() < 1.0,
            "recovery ramps instead of releasing a one-frame catch-up burst"
        );
        for _ in 0..100 {
            pacer.update(DVec3::ZERO, 0.1);
        }
        assert_eq!(pacer.effort(), 1.0);
    }

    #[test]
    fn stream_pacer_runs_full_workers_for_queued_work_at_rest() {
        let mut pacer = StreamPacer::default();
        pacer.update(DVec3::new(200.0, 0.0, 0.0), 1.0 / 60.0);
        assert_eq!(pacer.active_workers(12), 12, "travel keeps every worker");
        pacer.set_boost(true, 0.001);
        assert!(
            !pacer.boosting(),
            "queued work during travel must not lift the floor"
        );
        assert_eq!(pacer.active_workers(12), 12);
        assert_eq!(pacer.near_queue_cap(12), 48);
        assert!(pacer.floor(32) < 32, "travel still sheds admission");

        pacer.update(DVec3::ZERO, 1.0 / 60.0);
        pacer.set_boost(true, 0.001);
        assert!(pacer.boosting(), "cheap rest frame with leftover work");
        assert_eq!(pacer.active_workers(12), 12);
        assert_eq!(pacer.near_queue_cap(12), NEAR_REST_QUEUE_CAP);
        assert_eq!(pacer.floor(32), 32);
        assert!(
            pacer.duration(Duration::from_millis(1)) >= Duration::from_millis(1),
            "rest boost restores the full admission window"
        );

        pacer.set_boost(true, 0.020);
        assert!(
            !pacer.boosting(),
            "an already-expensive pass keeps the travel floor"
        );
        assert!(pacer.effort() < 1.0, "one rest frame does not restore effort");
        assert_eq!(pacer.active_workers(12), 12, "worker count does not follow the effort floor");
        assert!(pacer.floor(32) < 32);
    }

    #[test]
    fn loading_radius_shrinks_with_speed_and_recovers_gradually() {
        assert_eq!(StreamPacer::target_load_fraction(0.0), 1.0);
        assert_eq!(StreamPacer::target_load_fraction(FULL_EFFORT_SPEED_MPS), 1.0);
        assert_eq!(StreamPacer::target_load_fraction(f64::NAN), 1.0);
        assert_eq!(StreamPacer::target_load_fraction(f64::INFINITY), 0.0);

        let mut prev = 1.0f32;
        for speed in 24..=8000 {
            let fraction = StreamPacer::target_load_fraction(speed as f64);
            assert!(fraction <= prev + 1.0e-5, "not monotone at {speed} m/s: {fraction} > {prev}");
            assert!(prev - fraction < 0.05, "step at {speed} m/s is {prev} -> {fraction}");
            prev = fraction;
        }

        let mut pacer = StreamPacer::default();
        assert_eq!(pacer.load_radius(16), 16, "full at rest");
        pacer.update(DVec3::new(600.0, 0.0, 0.0), 0.0);
        let at_600 = pacer.load_radius(16);
        assert!((1..=3).contains(&at_600), "a few chunks at 600 m/s, got {at_600}");
        assert!(pacer.load_radius(5) <= 2);
        pacer.update(DVec3::new(2000.0, 0.0, 0.0), 0.0);
        let at_2000 = pacer.load_radius(16);
        assert!(
            at_2000 <= at_600 && at_2000 <= 2,
            "2000 m/s stays inside 600's window, got {at_2000}"
        );
        pacer.update(DVec3::new(5000.0, 0.0, 0.0), 0.0);
        assert!(pacer.load_radius(16) <= 1, "almost nothing at 5 km/s");

        pacer.update(DVec3::new(600.0, 0.0, 0.0), 0.0);
        let held = pacer.load_radius(16);
        pacer.update(DVec3::ZERO, 1.0 / 60.0);
        let nudged = pacer.load_radius(16);
        assert!(
            nudged <= held + 1 && nudged < 16,
            "one stopped frame does not restore the view, got {held} -> {nudged}"
        );
        for _ in 0..30 {
            pacer.update(DVec3::new(600.0, 0.0, 0.0), 1.0 / 60.0);
            assert_eq!(pacer.load_radius(16), held, "a steady speed holds one radius");
        }
        for _ in 0..80 {
            pacer.update(DVec3::ZERO, 0.1);
        }
        assert_eq!(pacer.load_radius(16), 16, "standing still grows the window back");
    }

    /// A teleport, a warp or a frame change moves the eye once: that sample
    /// does not shrink the window. Flight that keeps moving does.
    #[test]
    fn a_one_sample_jump_does_not_shrink_the_loading_window() {
        let dt = 1.0 / 60.0;
        let mut pacer = StreamPacer::default();
        pacer.observe(DVec3::new(3.0, 0.0, 0.0), dt, false);
        pacer.observe(DVec3::new(5000.0, 0.0, 0.0), dt, false);
        assert_eq!(pacer.load_radius(16), 16, "a jump waits for confirmation");
        pacer.observe(DVec3::new(3.0, 0.0, 0.0), dt, false);
        assert_eq!(pacer.load_radius(16), 16, "walking on drops the jump");
        pacer.observe(DVec3::new(5000.0, 0.0, 0.0), dt, true);
        pacer.observe(DVec3::new(3.0, 0.0, 0.0), dt, false);
        assert_eq!(pacer.load_radius(16), 16, "a frame change is no velocity");
        assert!(pacer.speed_mps() <= FULL_EFFORT_SPEED_MPS);
        pacer.observe(DVec3::new(600.0, 0.0, 0.0), dt, false);
        pacer.observe(DVec3::new(600.0, 0.0, 0.0), dt, false);
        assert!(pacer.load_radius(16) <= 3, "sustained flight shrinks it");
    }

    #[test]
    fn chunk_behind_is_travel_only_and_ignores_the_up_axis() {
        let center = Coord::new(0, 0, 0);
        let up = Some(Face::PosY);
        let fast = DVec3::new(100.0, 0.0, 0.0);
        assert!(!chunk_behind(center, Coord::new(3, 0, 0), fast, up), "ahead stays");
        assert!(!chunk_behind(center, Coord::new(0, 0, 2), fast, up), "beside stays");
        assert!(chunk_behind(center, Coord::new(-1, 0, 0), fast, up), "behind drops");
        assert!(!chunk_behind(center, center, fast, up), "the player's chunk stays");
        assert!(
            !chunk_behind(center, Coord::new(-1, 0, 0), DVec3::new(10.0, 0.0, 0.0), up),
            "walking keeps the trail"
        );
        assert!(
            !chunk_behind(center, Coord::new(0, -4, 0), DVec3::new(0.0, 100.0, 0.0), up),
            "a vertical climb does not drop the ground"
        );
    }
}
