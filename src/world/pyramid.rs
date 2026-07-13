//! Distance-driven LOD selection: map XZ distance to LOD level and keep tolerance.
use std::num::NonZeroU8;

use super::lod::Lod;
use super::metric::EyeDist;

/// Number of LOD rings: 7 gives base-2 cells, reaching 256m at the farthest ring.
pub(in crate::world) const SECTION_LEVELS: u8 = 7;

/// LOD choice at a given XZ distance. No `Chunks` case because sections totally cover
/// the plane; the near-field overlap with full-res chunks is handled by the clip volume.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LodChoice {
    Level(Lod),
    BeyondHorizon,
}

/// LOD pyramid configuration. `unit`: ring 0 radius in meters; `base`: cell-size ratio per ring.
pub struct PyramidCfg {
    pub finest: Lod,
    pub levels: NonZeroU8,
    pub unit: f32,
    /// ≥ 2.0 for exponential falloff.
    pub base: f32,
}

impl PyramidCfg {
    /// Standard config: base 2, 7 rings starting at finest LOD.
    pub fn sections(unit: f32) -> PyramidCfg {
        PyramidCfg {
            finest: Lod(2),
            levels: NonZeroU8::new(SECTION_LEVELS).unwrap(),
            unit,
            base: 2.0,
        }
    }

    /// LOD value increment per ring (log2 of base, floored at 1 for degenerate bases).
    pub fn step(&self) -> u8 {
        (self.base.log2().round() as i32).max(1) as u8
    }

    pub fn coarsest(&self) -> u8 {
        self.finest.0 + (self.levels.get() - 1) * self.step()
    }

    pub fn active_lods(&self) -> impl Iterator<Item = Lod> + '_ {
        (0..self.levels.get()).map(move |r| Lod(self.finest.0 + r * self.step()))
    }

    /// Pyramid's outer edge in metres — single authority for all zone boundaries.
    pub fn outer_m(&self) -> f32 {
        self.unit * self.base.powi(self.levels.get() as i32)
    }
}

/// Select LOD for a given distance: log-falloff to rings, clamped near and far.
/// Returns a Level or BeyondHorizon; NaN/negatives clamp to nearest ring.
pub(in crate::world) fn level_for(dist: EyeDist, cfg: &PyramidCfg) -> LodChoice {
    let dist_xz = dist.get();
    // `!(>=)` catches NaN and negatives: clamp to finest ring.
    if !(dist_xz >= cfg.unit) {
        return LodChoice::Level(cfg.finest);
    }
    let ring = (dist_xz / cfg.unit).log(cfg.base).floor();
    // Non-finite ring (corrupt cfg) saturates past horizon.
    let ring = if ring.is_finite() { ring as u32 } else { u32::MAX };
    if ring >= cfg.levels.get() as u32 {
        LodChoice::BeyondHorizon
    } else {
        LodChoice::Level(Lod(cfg.finest.0 + ring as u8 * cfg.step()))
    }
}

/// Keep-side tolerance for hysteresis: whether `lod` is drawable at this distance.
/// Accepts the ring's level and one ring finer (one step() apart) to avoid thrashing edges.
/// Past the horizon, nothing is acceptable.
pub(in crate::world) fn acceptable(dist: EyeDist, lod: Lod, cfg: &PyramidCfg) -> bool {
    match level_for(dist, cfg) {
        LodChoice::Level(expected) => {
            lod.0 == expected.0 || lod.0 + cfg.step() == expected.0
        }
        LodChoice::BeyondHorizon => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d1() -> PyramidCfg {
        PyramidCfg { finest: Lod(2), levels: NonZeroU8::new(2).unwrap(), unit: 256.0, base: 4.0 }
    }

    /// `level_for` is monotone and always returns a valid level; tolerance prevents thrashing.
    #[test]
    fn pyramid_selection_is_total_monotone_and_tolerant() {
        let cfg = d1();
        let mut last_coarseness = 0u8;
        for m in 0..40_000u32 {
            let d = EyeDist::new(m as f32);
            let c = level_for(d, &cfg); // total: never panics
            if let LodChoice::Level(l) = c {
                assert!(l.0 >= last_coarseness, "never finer with distance");
                last_coarseness = l.0;
                assert!(acceptable(d, l, &cfg), "chosen level acceptable at its distance");
                // One ring finer (one step apart) is also acceptable.
                if l.0 > cfg.finest.0 {
                    assert!(
                        acceptable(d, Lod(l.0 - cfg.step()), &cfg),
                        "one ring finer also acceptable"
                    );
                    assert!(
                        !acceptable(d, Lod(l.0 + cfg.step()), &cfg),
                        "one ring coarser is not"
                    );
                }
            }
        }
    }

    /// D1 bands: finest ring inside unit, Lod2 in [unit, unit*base), Lod4 beyond, then skin.
    #[test]
    fn d1_bands_are_lod2_then_lod4_then_horizon() {
        let cfg = d1();
        let d = EyeDist::new;
        assert_eq!(level_for(d(0.0), &cfg), LodChoice::Level(Lod(2)));
        assert_eq!(level_for(d(255.0), &cfg), LodChoice::Level(Lod(2)));
        assert_eq!(level_for(d(256.0), &cfg), LodChoice::Level(Lod(2)));
        assert_eq!(level_for(d(1023.0), &cfg), LodChoice::Level(Lod(2)));
        assert_eq!(level_for(d(1024.0), &cfg), LodChoice::Level(Lod(4)));
        assert_eq!(level_for(d(4095.0), &cfg), LodChoice::Level(Lod(4)));
        assert_eq!(level_for(d(4096.0), &cfg), LodChoice::BeyondHorizon);
        // Infinity clamps to finest ring (EyeDist enforces finite values).
        assert_eq!(level_for(d(f32::INFINITY), &cfg), LodChoice::Level(Lod(2)));
    }

}
