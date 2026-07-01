//! The simulation seam: where the world's *physics* will live — heat flowing along
//! thermal conductivity, electricity along conductivity, and the rest of the
//! documented systems that read the properties derived in [`block`](crate::block).
//!
//! None of that is implemented yet. What is real here is the *shape*: a [`Tick`]
//! trait, a fixed-timestep [`Simulation`] driver, and the wiring into the app loop.
//! The built-in systems are inert (gated off), so the game behaves exactly as
//! before — but adding a real system later is implementing `Tick`, with the
//! timestep and registration already in place and the hot block data already
//! reachable through [`World`](crate::world::World).
pub mod electrical;
pub mod thermal;

use crate::world::World;

/// How often the simulation steps, in seconds (20 Hz). Fixed so behaviour is
/// independent of frame rate.
pub const TICK_SECONDS: f32 = 1.0 / 20.0;

/// Upper bound on accumulated time, so a long stall (a stutter, a breakpoint)
/// doesn't make the next frame run hundreds of catch-up ticks ("spiral of death").
const MAX_ACCUMULATED: f32 = 0.25;

/// One simulation system: advanced by a fixed `dt` each tick and free to read and
/// mutate the world. Implementors are the documented physics systems (thermal,
/// electrical, …).
pub trait Tick {
    /// Advance this system by one fixed step of `dt` seconds.
    fn tick(&mut self, world: &mut World, dt: f32);
}

/// Drives every registered [`Tick`] system on a fixed timestep, decoupled from the
/// render frame rate by accumulating real time and spending it in whole ticks.
pub struct Simulation {
    systems: Vec<Box<dyn Tick>>,
    /// Real time banked but not yet spent on a whole tick.
    accumulator: f32,
}

impl Simulation {
    /// A simulation preloaded with the built-in systems (currently inert).
    pub fn new() -> Self {
        Self {
            systems: vec![
                Box::new(thermal::ThermalSystem::new()),
                Box::new(electrical::ElectricalSystem::new()),
            ],
            accumulator: 0.0,
        }
    }

    /// Add a system. The registration seam for future physics and mods.
    pub fn add(&mut self, system: Box<dyn Tick>) {
        self.systems.push(system);
    }

    /// Bank `dt` seconds of real time and run as many fixed ticks as it now affords.
    /// With only inert systems registered this is effectively free.
    pub fn advance(&mut self, world: &mut World, dt: f32) {
        self.accumulator = (self.accumulator + dt).min(MAX_ACCUMULATED);
        while self.accumulator >= TICK_SECONDS {
            self.accumulator -= TICK_SECONDS;
            for system in &mut self.systems {
                system.tick(world, TICK_SECONDS);
            }
        }
    }
}

impl Default for Simulation {
    fn default() -> Self {
        Self::new()
    }
}
