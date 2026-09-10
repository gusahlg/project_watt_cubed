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

    #[cfg(test)]
    pub fn cfg(&self) -> DiffusionCfg {
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
        "Replace fBm worldgen with overlapping-window InfiniteDiffusion fields (new worlds)."
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
        vec![
            Knob {
                label: "Tile",
                value: self.cfg.tile.to_string(),
                hint: DiffusionCfg::TILES
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(" / "),
            },
            Knob {
                label: "Stride",
                value: self.cfg.stride.to_string(),
                hint: format!(
                    "{}..={} step {}",
                    DiffusionCfg::MIN_STRIDE,
                    self.cfg.tile,
                    DiffusionCfg::STRIDE_STEP
                ),
            },
            Knob {
                label: "Phases",
                value: self.cfg.phases.to_string(),
                hint: format!(
                    "{}..={}",
                    DiffusionCfg::PHASES_MIN, DiffusionCfg::PHASES_MAX
                ),
            },
            Knob {
                label: "Relief",
                value: format!("{:.2}", self.cfg.relief),
                hint: DiffusionCfg::RELIEFS
                    .iter()
                    .map(|v| format!("{v:.1}"))
                    .collect::<Vec<_>>()
                    .join(" / "),
            },
        ]
    }

    fn step_knob(&mut self, index: usize, delta: i32) {
        match index {
            0 => {
                self.cfg.tile =
                    step_choice(&DiffusionCfg::TILES, self.cfg.tile as i32, delta) as u32;
                if self.cfg.stride > self.cfg.tile {
                    self.cfg.stride = self.cfg.tile / 2;
                }
            }
            1 => {
                self.cfg.stride = (self.cfg.stride as i32
                    + delta * DiffusionCfg::STRIDE_STEP as i32)
                    .clamp(
                        DiffusionCfg::MIN_STRIDE as i32,
                        self.cfg.tile as i32,
                    ) as u32;
            }
            2 => {
                self.cfg.phases = (self.cfg.phases as i32 + delta).clamp(
                    DiffusionCfg::PHASES_MIN as i32,
                    DiffusionCfg::PHASES_MAX as i32,
                ) as u32;
            }
            3 => {
                let next = DiffusionCfg::RELIEFS;
                let bits = (self.cfg.relief * 100.0f32).round() as i32;
                let cur = next
                    .iter()
                    .position(|v| (*v * 100.0f32).round() as i32 == bits)
                    .unwrap_or(1);
                let i = (cur as i32 + delta).rem_euclid(next.len() as i32) as usize;
                self.cfg.relief = next[i];
            }
            _ => {}
        }
        self.cfg = self.cfg.clamp();
    }

    fn save_state(&self, _world: &World) -> Option<(u16, String)> {
        Some((1, self.cfg.to_text()))
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

fn step_choice(list: &[u32], cur: i32, delta: i32) -> u32 {
    let at = list.iter().position(|&v| v as i32 == cur).unwrap_or_else(|| {
        list.iter()
            .enumerate()
            .min_by_key(|(_, v)| (**v as i32 - cur).unsigned_abs())
            .map(|(i, _)| i)
            .unwrap_or(0)
    });
    let i = (at as i32 + delta).rem_euclid(list.len() as i32) as usize;
    list[i]
}

#[cfg(test)]
mod knob_tests {
    use super::*;
    use crate::world::World;

    #[test]
    fn stepping_a_tile_not_in_the_choice_list_lands_on_a_neighbour() {
        assert_eq!(step_choice(&DiffusionCfg::TILES, 48, 1), 64);
        assert_eq!(step_choice(&DiffusionCfg::TILES, 48, -1), 16);
    }

    #[test]
    fn clamp_snaps_loaded_tile_to_a_stepper_choice() {
        let mut m = InfiniteDiffusionMod::new();
        let mut world = World::new(1);
        m.load_state(0, "tile=48,stride=16,phases=2,relief=1.00", &mut world);
        assert_eq!(m.cfg().tile, 32, "48 snaps to nearest choice 32");
        m.step_knob(0, 1);
        assert_eq!(m.cfg().tile, 64);
        m.load_state(0, "tile=48,stride=16,phases=2,relief=1.00", &mut world);
        m.step_knob(0, -1);
        assert_eq!(m.cfg().tile, 16);
    }

    #[test]
    fn default_is_on_the_stepper_and_id_is_the_worldgen_kind() {
        let m = InfiniteDiffusionMod::new();
        assert_eq!(m.id(), WorldgenKind::Diffusion.id());
        assert_eq!(m.name(), "InfiniteDiffusion");
        let cfg = m.cfg();
        assert!(DiffusionCfg::TILES.contains(&cfg.tile));
        assert_eq!(cfg.stride % DiffusionCfg::STRIDE_STEP, 0);
        assert!((DiffusionCfg::PHASES_MIN..=DiffusionCfg::PHASES_MAX).contains(&cfg.phases));
        assert!(
            DiffusionCfg::RELIEFS
                .iter()
                .any(|v| (*v - cfg.relief).abs() < f32::EPSILON)
        );
        let knobs = m.knobs();
        assert!(knobs.iter().all(|k| !k.hint.is_empty()));
    }
}
