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
    /// Stable name used to order systems. Install order must not decide tick
    /// sequence across clients.
    fn name(&self) -> &'static str;

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
        Self::with_systems(Vec::new())
    }

    /// Install `systems` in ascending-name order.
    pub fn with_systems(mut systems: Vec<Box<dyn Tick>>) -> Self {
        // Name order, not install order: two clients that register the same
        // systems in different sequences still tick them identically.
        systems.sort_by(|a, b| a.name().cmp(b.name()));
        Self { systems, ticks: 0 }
    }

    /// Add a system, inserting it into the name-sorted sequence.
    pub fn add(&mut self, system: Box<dyn Tick>) {
        let i = self.systems.partition_point(|s| s.name() <= system.name());
        self.systems.insert(i, system);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::World;

    struct Named(&'static str);

    impl Tick for Named {
        fn name(&self) -> &'static str {
            self.0
        }

        fn tick(&mut self, _world: &mut World, _dt: f32) {}
    }

    fn names(sim: &Simulation) -> Vec<&'static str> {
        sim.systems.iter().map(|s| s.name()).collect()
    }

    #[test]
    fn add_in_any_order_yields_the_same_iteration_order() {
        let mut added = Simulation::new();
        added.add(Box::new(Named("thermal")));
        added.add(Box::new(Named("electrical")));
        added.add(Box::new(Named("fluid")));

        let built = Simulation::with_systems(vec![
            Box::new(Named("fluid")),
            Box::new(Named("thermal")),
            Box::new(Named("electrical")),
        ]);

        let expect = ["electrical", "fluid", "thermal"];
        assert_eq!(names(&added), expect);
        assert_eq!(names(&built), expect);

        let mut reverse = Simulation::new();
        reverse.add(Box::new(Named("fluid")));
        reverse.add(Box::new(Named("electrical")));
        reverse.add(Box::new(Named("thermal")));
        assert_eq!(names(&reverse), expect);
    }
}
