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
/// of hundreds ("spiral of death"). At 20 Hz this is 5 ticks.
const MAX_FIXED_ACCUM: f32 = 0.25;

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
    /// reaches `floor`, so a starved `OnRevision`/`Hz` producer can't stall
    /// forever behind an upstream that never advances.
    ticks_skipped: u32,
    floor: u32,
    /// `None` until the owning lane wires a profiler row for it.
    meter: Option<Meter>,
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
        self.manual.push(Registered {
            manifest,
            runner,
            stamp: Rev::START,
            has_run: false,
            ticks_skipped: 0,
            // A pure call-point lane is driven every frame by its owner; the
            // forward-progress floor (a starvation backstop for cadence-gated
            // producers) is inapplicable, so it never force-fires on its own.
            floor: u32::MAX,
            meter: None,
        });
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
        self.producers.push(Registered {
            manifest,
            runner,
            stamp: Rev::START,
            has_run: false,
            ticks_skipped: 0,
            floor,
            meter: None,
        });
        self.source_revs.push(Rev::START);
        id
    }

    /// Wire an already-registered producer's profiler row.
    pub fn set_meter(&mut self, id: SourceId, meter: Meter) {
        self.producers[id.0 as usize].meter = Some(meter);
    }

    /// Runs registered producers in registration order — the only tie-break
    /// available since there's no derived dependency order — budget+floor
    /// enforced once, one profiler row per producer once a `Meter` is wired.
    pub fn tick(&mut self, ctx: &mut Ctx<'_>, clocks: &Clocks) -> TickReport {
        let mut quiescent = true;
        let mut spent_ms = 0.0f32;
        for i in 0..self.producers.len() {
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
    /// 1 for the once-per-tick cadences, `fixed_ticks_due` for `FixedTick`
    /// (bounded catch-up) — and the forward-progress floor forces one
    /// admission when a starved producer has been skipped `floor`
    /// consecutive ticks. A `FixedTick` lane whose clock has nothing due is
    /// not starved, so it registers with an effectively-infinite floor and
    /// never force-fires.
    fn reps(&self, i: usize, clocks: &Clocks) -> u32 {
        let p = &self.producers[i];
        let n = match &p.manifest.cadence {
            Cadence::Frame => 1,
            Cadence::FixedTick => clocks.fixed_ticks_due as u32,
            // Real Hz throttling needs per-producer wall-clock state this
            // scheduler doesn't track yet (`Clocks` carries only
            // frame_dt/fixed_ticks_due, not a per-producer elapsed-time
            // accumulator), so this falls back to once-per-tick.
            Cadence::Hz(_) => 1,
            Cadence::OnRevision(SourceId(id)) => {
                self.source_revs.get(*id as usize).is_some_and(|r| *r > p.stamp) as u32
            }
            Cadence::Once => (!p.has_run) as u32,
        };
        if n == 0 && p.ticks_skipped >= p.floor { 1 } else { n }
    }
}
