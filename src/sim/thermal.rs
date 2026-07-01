//! Heat flow — the future thermal simulation. A block's
//! [`thermal_conductivity`](crate::block::CoreProperties::thermal_conductivity)
//! sets how fast heat crosses into it, and its
//! [`temperature_resistance`](crate::block::CoreProperties::temperature_resistance)
//! sets when it degrades. None of that runs yet: this system is wired into the tick
//! loop but gated off, so it has no effect on the game.
use crate::sim::Tick;
use crate::world::World;

/// Propagates heat between neighbouring blocks (eventually). Inert until enabled.
pub struct ThermalSystem {
    /// Off until propagation is implemented; the tick is wired but does nothing.
    enabled: bool,
}

impl ThermalSystem {
    pub fn new() -> Self {
        Self { enabled: false }
    }

    /// How readily heat enters the block at a world coordinate. The accessor the
    /// real propagation step will sweep over neighbours — kept here to prove the
    /// derived block data is reachable from the simulation.
    fn thermal_conductivity_at(&self, world: &World, x: i32, y: i32, z: i32) -> u8 {
        let id = world.block_at(x, y, z);
        world.registry().block(id).core.thermal_conductivity
    }
}

impl Tick for ThermalSystem {
    fn tick(&mut self, world: &mut World, _dt: f32) {
        if !self.enabled {
            return;
        }
        // TODO: sweep blocks, moving heat toward equilibrium across faces, weighted
        // by each neighbour's thermal conductivity; degrade blocks past their
        // temperature resistance.
        let _ = self.thermal_conductivity_at(world, 0, 0, 0);
    }
}

impl Default for ThermalSystem {
    fn default() -> Self {
        Self::new()
    }
}
