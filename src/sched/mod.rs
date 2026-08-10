//! Runs registered [`Producer`]s serially against `&mut World`, in
//! registration order. Owns the fixed-tick accumulator and its catch-up cap,
//! and wall-clock interval gates for throttled lanes (autosave).
//!
//! Lives app-side, not in `voxel-engine`, because [`Ctx`] names the concrete
//! app `World` type and the engine crate has no dependency back on the app.
//!
//! Conflict-graph scheduling (footprint interference, claim tracking) was
//! removed: every real producer declares `FootprintKey::Global`, so ordering
//! was always plain registration order in practice. See
//! [`Scheduler::register_manual`] for how real lanes are actually driven.

use voxel_engine::producer::{Budget, Cadence, Clocks, Producer, Progress, SourceId, TickReport};
use voxel_engine::profile::{self, Meter};
use voxel_engine::{Engine, Rev};

use crate::world::World;

/// The [`Cadence::FixedTick`] period, in seconds — one definition of the tick
/// rate, shared with the sim systems that step at it (`sim::TICK_SECONDS`).
const FIXED_TICK_SECONDS: f32 = crate::sim::TICK_SECONDS;

/// A long stall (stutter, breakpoint) can owe at most this many seconds of
/// catch-up, so the next frame runs a bounded number of fixed ticks instead
/// of hundreds ("spiral of death"). At 20 Hz this is 5 ticks. One bank cap
/// shared by the fixed-tick clock and every [`RateGate`].
const MAX_FIXED_ACCUM: f32 = 0.25;

/// A bounded fixed-rate clock: banks frame time (capped at
/// [`MAX_FIXED_ACCUM`]) and converts it into whole due steps. THE one
/// accumulator implementation — the scheduler's `Cadence::Hz` lanes and the
/// game's throttled clocks (physics/stream/sky/mod rates) all run on it, so
/// the catch-up bound and the zero-rate convention cannot drift apart.
///
/// Rate zero is the explicit every-frame mode: always one step due, nothing
/// banked. The reciprocal is cached at construction so the hot not-due path
/// is one add, one min, one compare.
#[derive(Clone, Copy, Debug)]
pub struct RateGate {
    /// The configured rate; `0` is the every-frame mode.
    hz: u32,
    /// Seconds per step; `0.0` encodes the every-frame mode.
    interval: f32,
    accum: f32,
}

impl RateGate {
    /// A gate stepping `hz` times per second; `0` steps every frame.
    pub fn from_hz(hz: u32) -> Self {
        let interval = if hz == 0 { 0.0 } else { 1.0 / hz as f32 };
        RateGate { hz, interval, accum: 0.0 }
    }

    /// Reconfigure the rate. A no-op at the same rate; a real change drops the
    /// banked time so the new cadence starts clean instead of replaying a
    /// burst measured against the old interval.
    pub fn set_hz(&mut self, hz: u32) {
        if self.hz != hz {
            *self = Self::from_hz(hz);
        }
    }

    /// Whether this gate is in the every-frame mode.
    pub fn every_frame(&self) -> bool {
        self.interval == 0.0
    }

    /// Bank `dt` and return how many whole steps are due. Non-finite or
    /// negative `dt` banks nothing; the bank is capped so a stall replays a
    /// bounded burst, never an unbounded one.
    pub fn steps(&mut self, dt: f32) -> u32 {
        if self.interval == 0.0 {
            self.accum = 0.0;
            return 1;
        }
        let dt = if dt.is_finite() { dt.clamp(0.0, MAX_FIXED_ACCUM) } else { 0.0 };
        self.accum = (self.accum + dt).min(MAX_FIXED_ACCUM);
        if self.accum < self.interval {
            return 0;
        }
        let steps = (self.accum / self.interval) as u32;
        self.accum = (self.accum - self.interval * steps as f32).max(0.0);
        steps
    }

    /// The fixed step length in seconds, or `frame_dt` in every-frame mode —
    /// what a caller integrating real time should advance per due step.
    pub fn step_dt(&self, frame_dt: f32) -> f32 {
        if self.interval == 0.0 { frame_dt } else { self.interval }
    }

    /// Drop any banked time (the owning lane just ran out of band — a forced
    /// refresh — so the next step is a full interval out).
    pub fn reset(&mut self) {
        self.accum = 0.0;
    }
}

/// Single-clock replacement for a lane's hand-rolled `Instant` throttle (the
/// autosave gate's `last_attempt`). Accumulates `frame_dt` and reports "due"
/// once a whole `period` has elapsed; the owning lane resets it when it acts.
/// `accum` is clamped to `period` so "due" is sticky (a dirty-but-throttled
/// lane fires the instant it is allowed) and the float never grows unbounded.
struct Interval {
    period: f32,
    accum: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct IntervalHandle(usize);

/// The `&mut World` (+ optional engine handle) every producer runs against.
pub struct Ctx<'a> {
    pub world: &'a mut World,
    /// `Some` only for main-thread producers that also touch the renderer
    /// (e.g. an upload); worker-pool and pure-CPU producers see `None`.
    pub eng: Option<&'a mut Engine>,
}

impl<'a> Ctx<'a> {
    pub fn new(world: &'a mut World, eng: Option<&'a mut Engine>) -> Self {
        Ctx { world, eng }
    }
}

pub trait Run {
    /// Process due work within `budget`. Must make progress — the
    /// forward-progress floor is enforced once by the scheduler, never by a
    /// lane. Recompute the ready-set from current state every call; never
    /// cache actionability across ticks.
    fn run(&mut self, ctx: &mut Ctx<'_>, budget: Budget) -> Progress;
}

/// One registered producer: its manifest, its runner, and scheduler-side
/// bookkeeping (stamp, forward-progress floor, profiler wiring) that lives
/// with the scheduler, never inside the producer.
struct Registered {
    manifest: Producer,
    runner: Box<dyn Run>,
    /// The revision this producer last stamped its output at.
    stamp: Rev,
    has_run: bool,
    /// Consecutive ticks this producer was skipped by cadence gating (not by
    /// choice) — the forward-progress floor forces admission once this
    /// reaches `floor`, so a starved `OnRevision` producer can't stall
    /// forever behind an upstream that never advances.
    ticks_skipped: u32,
    floor: u32,
    /// `None` until the owning lane wires a profiler row for it.
    meter: Option<Meter>,
    /// The per-producer wall clock backing `Cadence::Hz`; `None` for every
    /// other cadence. Advanced once per [`Scheduler::tick`] by `frame_dt`.
    hz_gate: Option<RateGate>,
    /// A disabled producer is skipped entirely — no run, no starvation
    /// accounting. The lever a settings gate (simulation off) pulls instead
    /// of tearing the producer out of the registration order.
    enabled: bool,
}

impl Registered {
    /// Fresh bookkeeping around a manifest/runner pair; `floor`/`hz_gate` are
    /// the only fields that vary by registration path.
    fn new(manifest: Producer, runner: Box<dyn Run>, floor: u32, hz_gate: Option<RateGate>) -> Self {
        Registered {
            manifest,
            runner,
            stamp: Rev::START,
            has_run: false,
            ticks_skipped: 0,
            floor,
            meter: None,
            hz_gate,
            enabled: true,
        }
    }
}

/// Executes the manifest graph: ready-set from cadence ∧ rev, registration
/// order, budget+floor enforced once, one profiler row per producer once a
/// `Meter` is wired.
pub struct Scheduler {
    producers: Vec<Registered>,
    /// Each producer's most recently published [`Rev`], indexed by its own
    /// registration-order [`SourceId`] — the only data `Cadence::OnRevision`
    /// needs. A producer that hasn't run yet stays at `Rev::START`.
    source_revs: Vec<Rev>,
    /// Real time banked toward the next [`Cadence::FixedTick`] step, capped
    /// at [`MAX_FIXED_ACCUM`].
    fixed_accum: f32,
    /// Wall-clock interval gates, advanced by the same frame clock as
    /// `fixed_accum`.
    intervals: Vec<Interval>,
    /// Producers driven at a fixed call point inside an unmigrated serial pass
    /// (today: the occlusion + dirty-remesh lanes inside `World::stream`),
    /// invoked by [`Scheduler::run_manual`] rather than [`Scheduler::tick`].
    /// They stay off the tick loop because their exact position relative to the
    /// other, not-yet-migrated `stream` passes is load-bearing (dirty-remesh
    /// must precede the draw-set cache; occlusion must read the post-load chunk
    /// set) — a single tick slot cannot honour both. As those passes migrate,
    /// these move onto `tick`.
    manual: Vec<Registered>,
}

/// Handle to a [`Scheduler::register_manual`] producer, kept by the caller (the
/// `World` streaming lanes) to invoke it at its call point.
#[derive(Clone, Copy, Debug)]
pub struct ManualHandle(usize);

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl Scheduler {
    pub fn new() -> Self {
        Scheduler {
            producers: Vec::new(),
            source_revs: Vec::new(),
            fixed_accum: 0.0,
            intervals: Vec::new(),
            manual: Vec::new(),
        }
    }

    /// Register a call-point-driven producer (see [`Scheduler::manual`]).
    /// `tick` never runs these; the owner drives them with
    /// [`Scheduler::run_manual`] at the exact point their order requires.
    pub fn register_manual(&mut self, manifest: Producer, runner: Box<dyn Run>) -> ManualHandle {
        // A pure call-point lane is driven every frame by its owner; the
        // forward-progress floor (a starvation backstop for cadence-gated
        // producers) is inapplicable, so it never force-fires on its own.
        self.manual.push(Registered::new(manifest, runner, u32::MAX, None));
        ManualHandle(self.manual.len() - 1)
    }

    /// Run one call-point producer now, against `world` (+ `eng` for the
    /// lanes that upload to the renderer).
    pub fn run_manual(
        &mut self,
        handle: ManualHandle,
        world: &mut World,
        eng: Option<&mut Engine>,
    ) -> Progress {
        let reg = &mut self.manual[handle.0];
        let budget = reg.manifest.budget;
        let mut ctx = Ctx::new(world, eng);

        let progress = {
            let _guard = reg.meter.map(profile::scope);
            reg.runner.run(&mut ctx, budget)
        };

        reg.has_run = true;
        if let Progress::UpTo(rev) = progress {
            reg.stamp = rev;
        }
        progress
    }

    /// Starts un-elapsed, so a fresh lane waits a full period before its
    /// first fire (matching the old autosaver's construction-time grace).
    pub fn register_interval(&mut self, period_secs: f32) -> IntervalHandle {
        self.intervals.push(Interval { period: period_secs, accum: 0.0 });
        IntervalHandle(self.intervals.len() - 1)
    }

    /// Whether the gate's period has elapsed since it was last reset.
    pub fn interval_due(&self, handle: IntervalHandle) -> bool {
        let iv = &self.intervals[handle.0];
        iv.accum >= iv.period
    }

    /// Clear a gate — the lane calls this when it acts on a `due` gate, so the
    /// next fire is a full period out.
    pub fn interval_reset(&mut self, handle: IntervalHandle) {
        self.intervals[handle.0].accum = 0.0;
    }

    /// Advance the scheduler's clock by one frame and derive this tick's
    /// [`Clocks`]. Banks `frame_dt` (capped at [`MAX_FIXED_ACCUM`]) and
    /// converts it into whole `fixed_ticks_due` — the bounded catch-up count
    /// [`Scheduler::tick`] replays each `FixedTick` producer. The remainder
    /// stays banked for the next frame; the cap lives here, not in any lane.
    pub fn clocks(&mut self, frame_dt: f32) -> Clocks {
        for iv in &mut self.intervals {
            iv.accum = (iv.accum + frame_dt).min(iv.period);
        }
        self.fixed_accum = (self.fixed_accum + frame_dt).min(MAX_FIXED_ACCUM);
        let due = (self.fixed_accum / FIXED_TICK_SECONDS) as u32;
        self.fixed_accum -= due as f32 * FIXED_TICK_SECONDS;
        Clocks { frame_dt, fixed_ticks_due: due.min(u8::MAX as u32) as u8 }
    }

    /// `floor` is a tick count, declared by the caller rather than encoded
    /// in `Budget`. Returns this producer's `SourceId`, for wiring a
    /// dependent's `Cadence::OnRevision`.
    pub fn register(&mut self, manifest: Producer, runner: Box<dyn Run>, floor: u32) -> SourceId {
        let id = SourceId(self.producers.len() as u32);
        let hz_gate = match manifest.cadence {
            Cadence::Hz(hz) => Some(RateGate::from_hz(hz as u32)),
            _ => None,
        };
        self.producers.push(Registered::new(manifest, runner, floor, hz_gate));
        self.source_revs.push(Rev::START);
        id
    }

    /// Wire an already-registered producer's profiler row.
    pub fn set_meter(&mut self, id: SourceId, meter: Meter) {
        self.producers[id.0 as usize].meter = Some(meter);
    }

    /// Enable or disable a producer. Disabled producers are skipped by `tick`
    /// with no starvation accounting; re-enabling clears the banked skip count
    /// and any banked `Hz` time, so a lane returning from a long off period
    /// resumes on its cadence instead of replaying a catch-up burst.
    pub fn set_enabled(&mut self, id: SourceId, enabled: bool) {
        let p = &mut self.producers[id.0 as usize];
        if p.enabled == enabled {
            return;
        }
        p.enabled = enabled;
        p.ticks_skipped = 0;
        if let Some(gate) = &mut p.hz_gate {
            gate.reset();
        }
    }

    /// Runs registered producers in registration order — the only tie-break
    /// available since there's no derived dependency order — budget+floor
    /// enforced once, one profiler row per producer once a `Meter` is wired.
    pub fn tick(&mut self, ctx: &mut Ctx<'_>, clocks: &Clocks) -> TickReport {
        let mut quiescent = true;
        let mut spent_ms = 0.0f32;
        for i in 0..self.producers.len() {
            if !self.producers[i].enabled {
                continue;
            }
            let reps = self.reps(i, clocks);
            if reps == 0 {
                self.producers[i].ticks_skipped += 1;
                continue;
            }
            self.producers[i].ticks_skipped = 0;
            let budget = self.producers[i].manifest.budget;

            // A `FixedTick` producer replays once per whole tick due this
            // frame (bounded catch-up, capped by `clocks`); every other
            // cadence runs once. The catch-up loop lives here, in the
            // scheduler — never in a lane-local accumulator.
            for _ in 0..reps {
                let progress = {
                    let _guard = self.producers[i].meter.map(profile::scope);
                    self.producers[i].runner.run(ctx, budget)
                };

                self.producers[i].has_run = true;
                match progress {
                    Progress::UpTo(rev) => {
                        self.producers[i].stamp = rev;
                        self.source_revs[i] = rev;
                    }
                    Progress::Partial { .. } => quiescent = false,
                    Progress::Idle => {}
                }
                if let Budget::Millis(ms) = budget {
                    spent_ms += ms;
                }
            }
        }
        TickReport { quiescent, spent: Budget::Millis(spent_ms) }
    }

    /// Recomputed every call, never cached. Cadence ∧ rev gives the count —
    /// 1 for the once-per-tick cadences, `fixed_ticks_due` for `FixedTick`,
    /// the per-producer [`RateGate`]'s due steps for `Hz` (both bounded
    /// catch-up) — and the forward-progress floor forces one admission when a
    /// starved producer has been skipped `floor` consecutive ticks. A
    /// `FixedTick`/`Hz` lane whose clock has nothing due is not starved, so
    /// such lanes register with an effectively-infinite floor and never
    /// force-fire.
    fn reps(&mut self, i: usize, clocks: &Clocks) -> u32 {
        let p = &mut self.producers[i];
        let n = match &p.manifest.cadence {
            Cadence::Frame => 1,
            Cadence::FixedTick => clocks.fixed_ticks_due as u32,
            Cadence::Hz(_) => {
                p.hz_gate.as_mut().expect("Hz producer registered its gate").steps(clocks.frame_dt)
            }
            Cadence::OnRevision(SourceId(id)) => {
                self.source_revs.get(*id as usize).is_some_and(|r| *r > p.stamp) as u32
            }
            Cadence::Once => (!p.has_run) as u32,
        };
        if n == 0 && p.ticks_skipped >= p.floor { 1 } else { n }
    }
}

#[cfg(test)]
mod tests {
    use super::{Ctx, MAX_FIXED_ACCUM, RateGate, Run, Scheduler};
    use std::cell::Cell;
    use std::rc::Rc;
    use voxel_engine::Rev;
    use voxel_engine::producer::{Budget, Cadence, Footprint, FootprintKey, Producer, Progress};

    /// A counting producer that reports `progress` every run.
    struct Probe {
        runs: Rc<Cell<u32>>,
        progress: fn(u32) -> Progress,
    }

    impl Run for Probe {
        fn run(&mut self, _ctx: &mut Ctx<'_>, _budget: Budget) -> Progress {
            let n = self.runs.get() + 1;
            self.runs.set(n);
            (self.progress)(n)
        }
    }

    fn manifest(cadence: Cadence) -> Producer {
        Producer {
            name: "probe",
            footprint: Footprint { reads: vec![FootprintKey::Global], writes: vec![FootprintKey::Global] },
            cadence,
            budget: Budget::Millis(1.0),
        }
    }

    fn probe(progress: fn(u32) -> Progress) -> (Rc<Cell<u32>>, Box<Probe>) {
        let runs = Rc::new(Cell::new(0));
        (runs.clone(), Box::new(Probe { runs, progress }))
    }

    fn tick_n(sched: &mut Scheduler, world: &mut crate::world::World, frames: u32, dt: f32) {
        for _ in 0..frames {
            let clocks = sched.clocks(dt);
            let mut ctx = Ctx::new(world, None);
            sched.tick(&mut ctx, &clocks);
        }
    }

    #[test]
    fn hz_cadence_actually_throttles_and_disable_gates() {
        let mut world = crate::world::World::with_config_lazy(
            crate::world::DEFAULT_SEED,
            crate::render_config::RenderConfig::default(),
        );
        let mut sched = Scheduler::new();
        let (runs, lane) = probe(|_| Progress::Idle);
        // An Hz lane whose clock has nothing due is not starved: infinite floor.
        let id = sched.register(manifest(Cadence::Hz(10)), lane, u32::MAX);

        // 60 frames at ~60 fps ≈ 1 s: a 10 Hz lane runs ~10 times, not 60.
        tick_n(&mut sched, &mut world, 60, 1.0 / 60.0);
        let after_second = runs.get();
        assert!(
            (8..=12).contains(&after_second),
            "10 Hz lane ran {after_second} times over one second"
        );

        // Disabled: not ticked at all, and re-enabling drops banked time.
        sched.set_enabled(id, false);
        tick_n(&mut sched, &mut world, 60, 1.0 / 60.0);
        assert_eq!(runs.get(), after_second, "a disabled producer must not run");
        sched.set_enabled(id, true);
        let clocks = sched.clocks(1.0 / 60.0);
        sched.tick(&mut Ctx::new(&mut world, None), &clocks);
        assert!(runs.get() <= after_second + 1, "re-enable must not replay a banked burst");
    }

    #[test]
    fn on_revision_fires_on_upstream_advance_and_floor_backstops_starvation() {
        let mut world = crate::world::World::with_config_lazy(
            crate::world::DEFAULT_SEED,
            crate::render_config::RenderConfig::default(),
        );
        let mut sched = Scheduler::new();
        // Upstream publishes revision 1 on its first run and never advances
        // again — downstream must fire exactly once for it, then only the
        // starvation floor (10 skipped ticks) may force further admissions.
        let (_up_runs, up) = probe(|_| Progress::UpTo(Rev(1)));
        let up_id = sched.register(manifest(Cadence::Frame), up, u32::MAX);
        // Downstream STAMPS the revision it consumed (`UpTo`) — a lane that
        // returned `Idle` would rightly keep being offered the unconsumed rev.
        let (down_runs, down) = probe(|_| Progress::UpTo(Rev(1)));
        sched.register(manifest(Cadence::OnRevision(up_id)), down, 10);

        tick_n(&mut sched, &mut world, 5, 1.0 / 60.0);
        assert_eq!(down_runs.get(), 1, "one upstream advance = one downstream fire");

        // 30 more frozen ticks: the floor force-fires once per 10 skips.
        tick_n(&mut sched, &mut world, 30, 1.0 / 60.0);
        let forced = down_runs.get() - 1;
        assert!(
            (2..=4).contains(&forced),
            "the starvation floor should force ~3 admissions over 30 idle ticks, got {forced}"
        );
    }


    #[test]
    fn zero_rate_preserves_every_frame_behavior() {
        let mut gate = RateGate::from_hz(0);
        gate.accum = 0.123;
        assert!(gate.every_frame());
        assert_eq!(gate.steps(0.0), 1);
        assert_eq!(gate.accum, 0.0);
        assert_eq!(gate.step_dt(0.016), 0.016);
    }

    #[test]
    fn fixed_rate_accumulates_and_keeps_remainder() {
        let mut gate = RateGate::from_hz(10);
        assert_eq!(gate.steps(0.04), 0);
        assert_eq!(gate.steps(0.11), 1);
        assert!((gate.accum - 0.05).abs() < 1e-6);
        assert_eq!(gate.step_dt(0.016), 0.1);
    }

    #[test]
    fn fixed_rate_catch_up_is_bounded() {
        let mut gate = RateGate::from_hz(1_000);
        let steps = gate.steps(10.0);
        assert!(steps <= (MAX_FIXED_ACCUM * 1_000.0) as u32);
        assert!(gate.accum <= 1.0 / 1_000.0 + f32::EPSILON);
    }

    #[test]
    fn non_finite_dt_banks_nothing() {
        let mut gate = RateGate::from_hz(60);
        assert_eq!(gate.steps(f32::NAN), 0);
        assert_eq!(gate.steps(f32::INFINITY), 0);
        assert_eq!(gate.accum, 0.0);
    }
}
