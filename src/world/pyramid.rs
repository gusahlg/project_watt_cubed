//! The distance-driven LOD ladder: monotone map from XZ distance to the
//! [`Lod`] level drawn there ([`level_for`]), plus the keep-side tolerance
//! ([`acceptable`]). Used only by column-section far field ([`PyramidCfg::sections`]).
use std::num::NonZeroU8;

use super::lod::Lod;

/// Matches the old tile+skin far field horizon distance.
pub(in crate::world) const SECTION_LEVELS: u8 = 5;

/// What a given XZ distance from the player wants drawn there. `Option<Lod>`
/// was underspecified — "no tile" meant two different things (chunks own it vs.
/// the skin owns it), so it becomes three explicit cases.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LodChoice {
    /// Nearer than the innermost ring: full-res chunks own it.
    Chunks,
    /// This ring's tile level owns it.
    Level(Lod),
    /// Beyond the outermost ring: the far skin owns it.
    BeyondHorizon,
}

/// D1 configuration. `unit` is the chunk view radius in
/// metres (the innermost ring starts where the full-res box ends), `base` the
/// distance falloff — the cell-size ratio between adjacent rings.
pub struct PyramidCfg {
    pub finest: Lod,
    pub levels: NonZeroU8,
    pub unit: f32,
    /// ≥ 2.0 (log falloff — DH's load-bearing rule).
    pub base: f32,
}

impl PyramidCfg {
    /// Sized for progressive-covering and matching the old tile+skin horizon.
    pub fn sections(unit: f32) -> PyramidCfg {
        PyramidCfg {
            finest: Lod(2),
            levels: NonZeroU8::new(SECTION_LEVELS).unwrap(),
            unit,
            base: 2.0,
        }
    }

    /// LOD-value delta per ring: `log2(base)` (base 4 ⇒ 2 lod steps ⇒ Lod2→Lod4).
    /// Floored at 1 so a degenerate `base < 4` still advances a level per ring.
    pub fn step(&self) -> u8 {
        (self.base.log2().round() as i32).max(1) as u8
    }

    /// The outermost ring's LOD value.
    pub fn coarsest(&self) -> u8 {
        self.finest.0 + (self.levels.get() - 1) * self.step()
    }

    /// Every active ring's LOD, finest → coarsest (D1: `Lod(2)`, `Lod(4)`).
    pub fn active_lods(&self) -> impl Iterator<Item = Lod> + '_ {
        (0..self.levels.get()).map(move |r| Lod(self.finest.0 + r * self.step()))
    }

    /// The pyramid's outer edge in metres (`unit·base^levels`) — the ONE
    /// authority every dependent zone radius derives from: the last ring's
    /// band ends here, the render's skin clip starts here, and the Zone-3 skin
    /// ring is sized from here. Deriving them all from this method is what
    /// keeps the zones from drifting apart (Zone 3 strictly outside Zone 2 by
    /// construction, never by convention).
    pub fn outer_m(&self) -> f32 {
        self.unit * self.base.powi(self.levels.get() as i32)
    }
}

/// The log-falloff rule. `ring = floor(log_base(dist/unit))`; the ring's
/// level is `finest + ring·step`. `dist < unit` → [`Chunks`](LodChoice::Chunks);
/// past the last ring → [`BeyondHorizon`](LodChoice::BeyondHorizon). TOTAL (every
/// finite distance maps somewhere; NaN/negative fall to `Chunks`) and MONOTONE
/// (never finer with distance) — see [`tests`].
pub fn level_for(dist_xz: f32, cfg: &PyramidCfg) -> LodChoice {
    // `!(>=)` catches NaN and negatives too — they resolve to the near case.
    if !(dist_xz >= cfg.unit) {
        return LodChoice::Chunks;
    }
    let ring = (dist_xz / cfg.unit).log(cfg.base).floor();
    // `ring >= 0` since `dist >= unit` and `base >= 2`; a non-finite ring (only
    // reachable with a corrupt cfg) saturates past the horizon.
    let ring = if ring.is_finite() { ring as u32 } else { u32::MAX };
    if ring >= cfg.levels.get() as u32 {
        LodChoice::BeyondHorizon
    } else {
        LodChoice::Level(Lod(cfg.finest.0 + ring as u8 * cfg.step()))
    }
}

/// DH's `expected − 1` tolerance, in RINGS: may `lod` be DRAWN at this
/// distance? True for the ring's own level and one *ring* finer (i.e. one
/// `cfg.step()` of LOD value — with base 4 the rings step by 2, so a one-VALUE
/// tolerance would only ever match a level that doesn't exist), so a ring
/// breathes one step at its edges instead of thrashing. Outside the ring band
/// (chunks / skin) no tile is acceptable.
///
/// KEEP-side predicate: loading is exact-band (`level_for` equality in
/// `desired_tiles`) — this tolerance exists for unload/keep hysteresis, so a
/// just-crossed ring edge doesn't immediately drop the one-ring-finer tile the
/// player was looking at. Not yet consulted by an unload path.
pub fn acceptable(dist_xz: f32, lod: Lod, cfg: &PyramidCfg) -> bool {
    match level_for(dist_xz, cfg) {
        LodChoice::Level(expected) => {
            lod.0 == expected.0 || lod.0 + cfg.step() == expected.0
        }
        LodChoice::Chunks | LodChoice::BeyondHorizon => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d1() -> PyramidCfg {
        PyramidCfg { finest: Lod(2), levels: NonZeroU8::new(2).unwrap(), unit: 256.0, base: 4.0 }
    }

    /// Contract test (formerly `skeleton::pyramid::tests`):
    /// `level_for` is total + monotone, and every chosen level plus one finer LOD
    /// is `acceptable` at its own distance (rings breathe, never thrash).
    #[test]
    fn pyramid_selection_is_total_monotone_and_tolerant() {
        let cfg = d1();
        let mut last_coarseness = 0u8;
        for m in 0..40_000u32 {
            let d = m as f32;
            let c = level_for(d, &cfg); // total: never panics
            if let LodChoice::Level(l) = c {
                assert!(l.0 >= last_coarseness, "never finer with distance");
                last_coarseness = l.0;
                assert!(acceptable(d, l, &cfg), "chosen level acceptable at its distance");
                // The tolerance is one RING finer (one cfg.step() of LOD value)
                // — the level an adjacent ring actually draws.
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

    /// The D1 bands: chunks inside `unit`, `Lod(2)` in `[unit, unit·base)`,
    /// `Lod(4)` in `[unit·base, unit·base²)`, skin past that.
    #[test]
    fn d1_bands_are_lod2_then_lod4_then_horizon() {
        let cfg = d1();
        assert_eq!(level_for(0.0, &cfg), LodChoice::Chunks);
        assert_eq!(level_for(255.0, &cfg), LodChoice::Chunks);
        assert_eq!(level_for(256.0, &cfg), LodChoice::Level(Lod(2)));
        assert_eq!(level_for(1023.0, &cfg), LodChoice::Level(Lod(2)));
        assert_eq!(level_for(1024.0, &cfg), LodChoice::Level(Lod(4)));
        assert_eq!(level_for(4095.0, &cfg), LodChoice::Level(Lod(4)));
        assert_eq!(level_for(4096.0, &cfg), LodChoice::BeyondHorizon);
        assert_eq!(level_for(f32::INFINITY, &cfg), LodChoice::BeyondHorizon);
    }

}
