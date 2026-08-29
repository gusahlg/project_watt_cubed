//! The fixed-timestep simulation seam. Systems implement [`Tick`] and register
//! with [`Simulation`]; unfinished physics belongs in design notes until it has a
//! real active-cell model and observable behaviour.

use crate::sched::{Ctx, Run};
use crate::world::World;
use voxel_engine::Rev;
use voxel_engine::producer::{
    Budget, Cadence, Footprint, FootprintKey, Producer, Progress,
};

/// How often the simulation steps, in seconds (20 Hz). Fixed so behaviour is
/// independent of frame rate. Also the period of the scheduler's
/// `Cadence::FixedTick` clock, so the tick rate has one definition, not two.
pub const TICK_SECONDS: f32 = 1.0 / 20.0;

/// One simulation system: advanced by a fixed `dt` each tick and free to read and
/// mutate the world. Implementors are the documented physics systems (thermal,
/// electrical, …).
pub trait Tick {
    /// Advance this system by one fixed step of `dt` seconds.
    fn tick(&mut self, world: &mut World, dt: f32);
}

/// Drives every registered [`Tick`] system on a fixed timestep. Registered as
/// a [`Cadence::FixedTick`] producer: the scheduler owns the real-time
/// accumulator and the catch-up cap, invoking [`Run::run`] once per whole
/// tick due this frame — this driver holds no accumulator of its own.
pub struct Simulation {
    systems: Vec<Box<dyn Tick>>,
    /// Whole ticks run so far; stamps this producer's output revision.
    ticks: u64,
}

impl Simulation {
    /// An empty simulation ready for concrete systems to register.
    pub fn new() -> Self {
        Self { systems: Vec::new(), ticks: 0 }
    }

    /// Add a system. The registration seam for future physics and mods.
    pub fn add(&mut self, system: Box<dyn Tick>) {
        self.systems.push(system);
    }

    /// A physics step may read and write any block in any chunk
    /// (thermal/electrical diffuse across the loaded world), so the
    /// footprint declares `Global` reads and writes rather than a tighter
    /// region. Budget is a soft per-tick CPU floor — inert today, so
    /// unmeasured.
    pub fn manifest() -> Producer {
        Producer {
            name: "sim",
            footprint: Footprint {
                reads: vec![FootprintKey::Global],
                writes: vec![FootprintKey::Global],
            },
            cadence: Cadence::FixedTick,
            budget: Budget::Millis(2.0),
        }
    }
}

impl Run for Simulation {
    /// One whole fixed tick: advance every system by [`TICK_SECONDS`]. The
    /// scheduler calls this once per tick due (bounded catch-up capped by the
    /// scheduler's clock), so there is no lane-local accumulator or loop here.
    fn run(&mut self, ctx: &mut Ctx<'_>, _budget: Budget) -> Progress {
        self.ticks += 1;
        for system in self.systems.iter_mut() {
            system.tick(ctx.world, TICK_SECONDS);
        }
        Progress::UpTo(Rev(self.ticks))
    }
}

impl Default for Simulation {
    fn default() -> Self {
        Self::new()
    }
}
