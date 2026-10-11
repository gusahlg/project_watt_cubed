//! The far-field ladder a render distance's settings give: its unit, finest detail and levels,
//! which size the relief bake.
use std::num::NonZeroU8;

use crate::ident::Detail;
use crate::render_config::{LOD_DETAIL_RANGE, LOD_LEVELS_RANGE, max_lod_levels};


/// Number of LOD rings: 7 gives base-2 cells, reaching 256m at the farthest ring.
#[cfg(test)]
pub(in crate::world) const SECTION_LEVELS: u8 = 7;

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
    #[cfg(test)]
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

    /// Pyramid's outer edge in metres — single authority for all zone boundaries.
    pub fn outer_m(&self) -> f32 {
        self.unit * self.base.powi(self.levels.get() as i32)
    }
}

/// `finest.0 + ring*step` widened to i64 before summing, then clamped into `Detail`'s i8 range: a
/// large ring count must saturate at the coarsest representable detail, never wrap.
fn ringed_detail(finest: Detail, ring: u32, step: u8) -> Detail {
    let off = ring as i64 * step as i64;
    Detail((finest.0 as i64 + off).clamp(i8::MIN as i64, i8::MAX as i64) as i8)
}

#[cfg(test)]
mod tests {
    use super::*;

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

}
