//! InfiniteDiffusion worldgen mod: overlapping-window fields instead of fBm.
//! Off by default so classic noise (and its goldens/fingerprint) stay the core.

use crate::mods::{Knob, Mod};
use crate::world::diffusion::DiffusionCfg;
use crate::world::generation::WorldgenKind;
use crate::world::World;

pub struct InfiniteDiffusionMod {
    cfg: DiffusionCfg,
}

impl InfiniteDiffusionMod {
    pub fn new() -> Self {
        Self {
            cfg: DiffusionCfg::default(),
        }
    }

    pub fn cfg(&self) -> DiffusionCfg {
        self.cfg
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

    fn description(&self) -> &str {
        "Replace fBm worldgen with overlapping-window InfiniteDiffusion fields (new worlds)."
    }

    fn worldgen(&self) -> Option<WorldgenKind> {
        Some(WorldgenKind::Diffusion)
    }

    fn diffusion_cfg(&self) -> Option<DiffusionCfg> {
        Some(self.cfg)
    }

    fn knobs(&self) -> Vec<Knob> {
        vec![
            Knob {
                label: "Tile",
                value: self.cfg.tile.to_string(),
            },
            Knob {
                label: "Stride",
                value: self.cfg.stride.to_string(),
            },
            Knob {
                label: "Phases",
                value: self.cfg.phases.to_string(),
            },
            Knob {
                label: "Relief",
                value: format!("{:.2}", self.cfg.relief),
            },
        ]
    }

    fn step_knob(&mut self, index: usize, delta: i32) {
        match index {
            0 => {
                let next = [16, 32, 64];
                self.cfg.tile = step_choice(&next, self.cfg.tile as i32, delta) as u32;
                if self.cfg.stride > self.cfg.tile {
                    self.cfg.stride = self.cfg.tile / 2;
                }
            }
            1 => {
                self.cfg.stride = (self.cfg.stride as i32 + delta * 8).clamp(8, self.cfg.tile as i32) as u32;
            }
            2 => {
                self.cfg.phases = (self.cfg.phases as i32 + delta).clamp(2, 8) as u32;
            }
            3 => {
                let next = [0.5, 1.0, 1.5, 2.0, 4.0];
                let bits = (self.cfg.relief * 100.0f32).round() as i32;
                let cur = next.iter().position(|v| (*v * 100.0f32).round() as i32 == bits).unwrap_or(1);
                let i = (cur as i32 + delta).rem_euclid(next.len() as i32) as usize;
                self.cfg.relief = next[i];
            }
            _ => {}
        }
        self.cfg = self.cfg.clamp();
    }

    fn save_state(&self, _world: &World) -> Option<String> {
        Some(format!(
            "tile={},stride={},phases={},relief={:.2}",
            self.cfg.tile, self.cfg.stride, self.cfg.phases, self.cfg.relief
        ))
    }

    fn load_state(&mut self, data: &str, _world: &mut World) {
        let mut cfg = self.cfg;
        for part in data.split(',') {
            let Some((k, v)) = part.split_once('=') else { continue };
            match k.trim() {
                "tile" => cfg.tile = v.parse().unwrap_or(cfg.tile),
                "stride" => cfg.stride = v.parse().unwrap_or(cfg.stride),
                "phases" => cfg.phases = v.parse().unwrap_or(cfg.phases),
                "relief" => cfg.relief = v.parse().unwrap_or(cfg.relief),
                _ => {}
            }
        }
        self.cfg = cfg.clamp();
    }
}

fn step_choice(list: &[u32], cur: i32, delta: i32) -> u32 {
    let at = list.iter().position(|&v| v as i32 == cur).unwrap_or(0);
    let i = (at as i32 + delta).rem_euclid(list.len() as i32) as usize;
    list[i]
}
