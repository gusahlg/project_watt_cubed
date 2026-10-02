//! InfiniteDiffusion worldgen mod: mountains and valleys on the surface, caves and abandoned mines
//! below, planets in space above (see [`crate::world::terrain`]). Knobs apply to new worlds.

use crate::mods::{Knob, Mod};
use crate::world::generation::WorldgenKind;
use crate::world::terrain::TerrainCfg;
use crate::world::World;

pub struct InfiniteDiffusionMod {
    cfg: TerrainCfg,
}

impl InfiniteDiffusionMod {
    pub fn new() -> Self {
        Self { cfg: TerrainCfg::default() }
    }

    #[cfg(test)]
    pub fn cfg(&self) -> TerrainCfg {
        self.cfg
    }

    fn apply_cfg_text(&mut self, data: &str) {
        self.cfg = self.cfg.overlay(data);
    }
}

impl Default for InfiniteDiffusionMod {
    fn default() -> Self {
        Self::new()
    }
}

impl Mod for InfiniteDiffusionMod {
    fn name(&self) -> &str {
        "InfiniteDiffusion"
    }

    fn id(&self) -> &'static str {
        WorldgenKind::Diffusion.id()
    }

    fn description(&self) -> &str {
        "Mountain ranges and carved valleys, caves and abandoned mines below, planets in space above (new worlds)."
    }

    fn group(&self) -> &'static str {
        crate::mods::ESSENTIALS
    }

    fn worldgen(&self) -> Option<WorldgenKind> {
        Some(WorldgenKind::Diffusion)
    }

    fn worldgen_config(&self) -> Option<String> {
        Some(self.cfg.to_text())
    }

    fn knobs(&self) -> Vec<Knob> {
        let (rlo, rhi) = TerrainCfg::RELIEF;
        let (dlo, dhi) = TerrainCfg::DENSITY;
        vec![
            Knob { label: "Relief", value: format!("{}%", self.cfg.relief), hint: format!("{rlo}..{rhi}%") },
            Knob { label: "Caves", value: format!("{}%", self.cfg.caves), hint: format!("{dlo}..{dhi}%") },
            Knob { label: "Mines", value: format!("{}%", self.cfg.mines), hint: format!("{dlo}..{dhi}%") },
            Knob { label: "Space", value: format!("{}%", self.cfg.space), hint: format!("{dlo}..{dhi}%") },
        ]
    }

    fn step_knob(&mut self, index: usize, delta: i32) {
        let step = |v: u16| (v as i32 + delta * TerrainCfg::STEP as i32).max(0) as u16;
        match index {
            0 => self.cfg.relief = step(self.cfg.relief),
            1 => self.cfg.caves = step(self.cfg.caves),
            2 => self.cfg.mines = step(self.cfg.mines),
            3 => self.cfg.space = step(self.cfg.space),
            _ => {}
        }
        self.cfg = self.cfg.clamp();
    }

    fn save_state(&self, _world: &World) -> Option<(u16, String)> {
        Some((2, self.cfg.to_text()))
    }

    fn load_state(&mut self, _version: u16, data: &str, _world: &mut World) -> u32 {
        self.apply_cfg_text(data);
        0
    }

    fn save_choice_state(&self) -> Option<String> {
        Some(self.cfg.to_text())
    }

    fn load_choice_state(&mut self, data: &str) {
        self.apply_cfg_text(data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knobs_step_snap_and_round_trip_through_text() {
        let mut m = InfiniteDiffusionMod::new();
        assert_eq!(m.id(), WorldgenKind::Diffusion.id());
        m.step_knob(0, 2);
        assert_eq!(m.cfg().relief, 150);
        m.step_knob(2, -10);
        assert_eq!(m.cfg().mines, 0, "density knobs bottom out at 0");
        m.step_knob(0, 10);
        assert_eq!(m.cfg().relief, TerrainCfg::RELIEF.1);
        let text = m.save_choice_state().unwrap();
        let mut n = InfiniteDiffusionMod::new();
        n.load_choice_state(&text);
        assert_eq!(n.cfg(), m.cfg());
        n.load_choice_state("relief=37,unknown=9");
        assert_eq!(n.cfg().relief, 25, "a stray value snaps onto the stepper");
    }
}
