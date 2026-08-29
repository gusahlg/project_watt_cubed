//! Distance-driven LOD selection: map XZ distance to LOD level and keep tolerance.
use std::num::NonZeroU8;

use crate::ident::Detail;
use crate::render_config::{LOD_DETAIL_RANGE, LOD_LEVELS_RANGE, max_lod_levels};

use super::metric::EyeDist;

/// Number of LOD rings: 7 gives base-2 cells, reaching 256m at the farthest ring.
pub(in crate::world) const SECTION_LEVELS: u8 = 7;

/// LOD choice at a given XZ distance. No `Chunks` case because sections totally cover
/// the plane; the near-field overlap with full-res chunks is handled by the clip volume.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LodChoice {
    Level(Detail),
    BeyondHorizon,
}

/// LOD pyramid configuration. `unit`: ring 0 radius in meters; `base`: cell-size ratio per ring.
pub struct PyramidCfg {
    pub finest: Detail,
    pub levels: NonZeroU8,
    pub unit: f32,
    /// ≥ 2.0 for exponential falloff.
    pub base: f32,
    /// Cached integer log2(base). Selection and covering query this for every
    /// section, so deriving it once avoids repeated floating-point logarithms.
    step: u8,
}

impl PyramidCfg {
    /// Standard config: base 2, 7 rings starting at finest LOD.
    pub fn sections(unit: f32) -> PyramidCfg {
        Self::sections_with(unit, SECTION_LEVELS, super::section::FINEST_DETAIL.0 as u8)
    }

    /// Configurable section ladder. Inputs are clamped defensively even though
    /// [`RenderConfig`](crate::render_config::RenderConfig) normalizes them at
    /// the settings boundary: construction from tests and internal callers
    /// must preserve the same coarsest-detail hierarchy invariant.
    pub fn sections_with(unit: f32, levels: u8, detail: u8) -> PyramidCfg {
        let detail = detail.clamp(*LOD_DETAIL_RANGE.start(), *LOD_DETAIL_RANGE.end());
        let levels = levels
            .clamp(*LOD_LEVELS_RANGE.start(), *LOD_LEVELS_RANGE.end())
            .min(max_lod_levels(detail));
        PyramidCfg {
            finest: Detail(detail as i8),
            levels: NonZeroU8::new(levels).expect("levels clamped to a nonzero range"),
            unit,
            base: 2.0,
            step: 1,
        }
    }

    /// LOD value increment per ring (integer log2 of base, cached at construction).
    pub fn step(&self) -> u8 {
        self.step
    }

    pub fn coarsest(&self) -> Detail {
        ringed_detail(self.finest, (self.levels.get() - 1) as u32, self.step())
    }

    pub fn active_lods(&self) -> impl Iterator<Item = Detail> + '_ {
        (0..self.levels.get()).map(move |r| ringed_detail(self.finest, r as u32, self.step()))
    }

    /// Pyramid's outer edge in metres — single authority for all zone boundaries.
    pub fn outer_m(&self) -> f32 {
        self.unit * self.base.powi(self.levels.get() as i32)
    }
}

/// `finest.0 + ring*step` widened to i64 before summing, then clamped into
/// `Detail`'s i8 range — a large ring count (corrupt cfg, or `step` pushing the
/// u8 product past 127) must saturate at the coarsest representable Detail,
/// never silently wrap through `i8::MAX` into a negative (impossibly fine) one.
fn ringed_detail(finest: Detail, ring: u32, step: u8) -> Detail {
    let off = ring as i64 * step as i64;
    Detail((finest.0 as i64 + off).clamp(i8::MIN as i64, i8::MAX as i64) as i8)
}

/// Select LOD for a given distance: log-falloff to rings, clamped near and far.
/// Returns a Level or BeyondHorizon; NaN/negatives clamp to nearest ring.
pub(in crate::world) fn level_for(dist: EyeDist, cfg: &PyramidCfg) -> LodChoice {
    let dist_xz = dist.get();
    // `!(>=)` catches NaN and negatives: clamp to finest ring.
    if !(dist_xz >= cfg.unit) {
        return LodChoice::Level(cfg.finest);
    }
    // At most eight multiply/compare steps beat a transcendental logarithm on
    // the per-section visibility path, while preserving exact band boundaries.
    // Bounded by the levels cap even for a degenerate (non-growing) base.
    let mut ring = 0u32;
    let mut upper = cfg.unit * cfg.base;
    while dist_xz >= upper {
        ring += 1;
        if ring >= cfg.levels.get() as u32 {
            return LodChoice::BeyondHorizon;
        }
        upper *= cfg.base;
    }
    LodChoice::Level(ringed_detail(cfg.finest, ring, cfg.step()))
}

/// Keep-side tolerance for hysteresis: whether `lod` is drawable at this distance.
/// Accepts the ring's level and one ring finer (one step() apart) to avoid thrashing edges.
/// Past the horizon, nothing is acceptable.
pub(in crate::world) fn acceptable(dist: EyeDist, lod: Detail, cfg: &PyramidCfg) -> bool {
    match level_for(dist, cfg) {
        LodChoice::Level(expected) => {
            // Widen to i32 for the compare: `lod.0` near `i8::MAX` plus `step`
            // must not wrap through the i8 range and produce a false match.
            lod.0 == expected.0 || lod.0 as i32 + cfg.step() as i32 == expected.0 as i32
        }
        LodChoice::BeyondHorizon => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d1() -> PyramidCfg {
        PyramidCfg {
            finest: Detail(2),
            levels: NonZeroU8::new(2).unwrap(),
            unit: 256.0,
            base: 4.0,
            step: 2,
        }
    }

    /// The configurable constructor clamps into the supported ladder and
    /// shortens levels so the coarsest ring never exceeds the hierarchy cap.
    #[test]
    fn configurable_ladder_clamps_and_respects_the_coarsest_cap() {
        let cfg = PyramidCfg::sections_with(256.0, 8, 6);
        assert_eq!(cfg.finest, Detail(6));
        assert_eq!(cfg.levels.get(), 4, "detail 6 exposes only levels 6 through 9");
        assert_eq!(cfg.coarsest(), Detail(9));

        let default = PyramidCfg::sections_with(256.0, SECTION_LEVELS, 2);
        assert_eq!(default.finest, super::super::section::FINEST_DETAIL);
        assert_eq!(default.levels.get(), SECTION_LEVELS);
        assert_eq!(default.step(), 1);
    }

    /// `level_for` is monotone and always returns a valid level; tolerance prevents thrashing.
    #[test]
    fn pyramid_selection_is_total_monotone_and_tolerant() {
        let cfg = d1();
        let mut last_coarseness = 0i8;
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
                        acceptable(d, Detail(l.0 - cfg.step() as i8), &cfg),
                        "one ring finer also acceptable"
                    );
                    assert!(
                        !acceptable(d, Detail(l.0 + cfg.step() as i8), &cfg),
                        "one ring coarser is not"
                    );
                }
            }
        }
    }

    /// Finest ring inside unit yields LOD2, then LOD4 beyond unit*base, then skin (horizon).
    #[test]
    fn d1_bands_are_lod2_then_lod4_then_horizon() {
        let cfg = d1();
        let d = EyeDist::new;
        assert_eq!(level_for(d(0.0), &cfg), LodChoice::Level(Detail(2)));
        assert_eq!(level_for(d(255.0), &cfg), LodChoice::Level(Detail(2)));
        assert_eq!(level_for(d(256.0), &cfg), LodChoice::Level(Detail(2)));
        assert_eq!(level_for(d(1023.0), &cfg), LodChoice::Level(Detail(2)));
        assert_eq!(level_for(d(1024.0), &cfg), LodChoice::Level(Detail(4)));
        assert_eq!(level_for(d(4095.0), &cfg), LodChoice::Level(Detail(4)));
        assert_eq!(level_for(d(4096.0), &cfg), LodChoice::BeyondHorizon);
        // Infinity clamps to finest ring (EyeDist enforces finite values).
        assert_eq!(level_for(d(f32::INFINITY), &cfg), LodChoice::Level(Detail(2)));
    }

}
