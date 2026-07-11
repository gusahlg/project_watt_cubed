//! Phase D1 — the far-terrain LOD *pyramid*: a distance-driven ladder of
//! coarse [`Lod`] tile rings between the full-res chunks and the Zone-3 skin.
//!
//! D1 is the ONE-new-level slice: exactly two rings, `Lod(2)` (today's tier) then
//! `Lod(4)`, spaced by a log-falloff distance rule. The k-level generalisation is
//! D2 — an explicit go/no-go gate, deliberately NOT built here.
//!
//! Two derivations live here, both "derived, not guessed":
//! * [`level_for`] — the total, monotone distance→[`LodChoice`] map.
//! * [`DroopTable`] — per-level geometric droop, *calibrated* by a deterministic
//!   in-crate sweep of the generator, never a bare constant. Coarser levels
//!   droop further into the ground, so where two rings overlap the finer one sits
//!   higher and wins the depth test — the same trick the skin uses under the tiles
//!   ([`SKIN_DROOP`](super::skin)), now a whole ladder.
use std::num::NonZeroU8;

use super::generation::TerrainGenerator;
use super::lod::Lod;

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
    /// The D1 slice: `finest = Lod(2)`, two rings, `base = 4.0` (a 4× cell-size
    /// ratio ⇒ the second ring is `Lod(4)`), `unit` = the chunk view radius in m.
    pub fn d1(unit: f32) -> PyramidCfg {
        PyramidCfg { finest: Lod(2), levels: NonZeroU8::new(2).unwrap(), unit, base: 4.0 }
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

/// Ring overlap in COARSER-level tiles (≥ 1): the coarse parent is loaded one
/// tile past the fine ring's outer edge so it draws *under* that edge —
/// coarse-over-fine, never a hole — set to 1, acceptance-tested by
/// `SkyHoleCount`.
pub fn ring_overlap(_lod: Lod) -> i32 {
    1
}

pub const CALIBRATION_COLUMNS: usize = 4096;
/// Droop cap in METRES — the sweep measures height disagreement in world metres
/// and `render` sinks tiles by world metres, so the cap lives in the same unit.
/// Note: the cap is in metres, not cells — a per-cell cap would scale with LOD and mean nothing at render.
/// A pathological generator saturates here and [`ring_overlap`] carries the
/// residual; the worst-seed golden shot is the check.
pub const DROOP_CAP: i32 = 32;

/// Per-level geometric droop, generalising the skin's single `SKIN_DROOP`.
/// "Derived, not guessed" is the API: the only constructors are
/// [`calibrate`](Self::calibrate) (evidence from a sweep) and
/// [`manual`](Self::manual) (evidence in writing).
pub struct DroopTable {
    /// Indexed by `lod.0 − finest.0`, so `Lod(2)`→`[0]`, `Lod(4)`→`[2]`; the
    /// unused intermediate index (`Lod(3)`) is filled but never queried.
    per_level: Vec<i32>,
}

impl DroopTable {
    /// Sweep [`CALIBRATION_COLUMNS`] columns (a deterministic in-crate LCG,
    /// seeded from `cfg` — no `rand`, no `Date::now`) across ±2²⁰ m. For each
    /// active LOD value `k`, `droop(k)` is the max over the sweep of the height
    /// disagreement between the generator sampled on the level-`k` cell grid and
    /// on the grid it must sit UNDER at a ring seam — the next finer ACTIVE
    /// level (`k − step`), or the exact per-block height for the finest ring
    /// (its seam is against the full-res chunks). Comparing against `k−1` would
    /// measure a grid no ring ever draws and so systematically under-droop the real seams.
    /// Capped at [`DROOP_CAP`].
    /// Run once at startup/config change and cached on the `World`.
    pub fn calibrate<G: TerrainGenerator>(generator: &G, cfg: &PyramidCfg) -> DroopTable {
        const RANGE: i64 = 1 << 20;
        let coarsest = cfg.coarsest();
        let span = (coarsest - cfg.finest.0) as usize + 1;
        let mut per_level = vec![0i32; span];

        // Seed the LCG from the cfg so calibration is reproducible per config.
        let mut lcg: u64 = 0x2545_F491_4F6C_DD1D
            ^ (cfg.unit.to_bits() as u64)
            ^ ((cfg.base.to_bits() as u64) << 32);
        let mut next = || {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            lcg
        };
        let rnd = |r: u64| ((r >> 11) as i64).rem_euclid(2 * RANGE) - RANGE;

        for _ in 0..CALIBRATION_COLUMNS {
            let (wx, wz) = (rnd(next()), rnd(next()));
            for lod in cfg.active_lods() {
                let k = lod.0;
                // The height this level must not poke through: exact terrain
                // for the finest ring, the next finer ring's grid otherwise.
                let finer = if k == cfg.finest.0 {
                    generator.height(wx as i32, wz as i32)
                } else {
                    sample_snapped(generator, wx, wz, k - cfg.step())
                };
                let d = (sample_snapped(generator, wx, wz, k) - finer).abs().min(DROOP_CAP);
                let i = (k - cfg.finest.0) as usize;
                per_level[i] = per_level[i].max(d);
            }
        }
        DroopTable { per_level }
    }

    /// Escape hatch — demands its evidence in writing.
    pub fn manual(per_level: Vec<i32>, _reason: &'static str) -> DroopTable {
        DroopTable { per_level }
    }

    /// Metres of downward droop for `lod` (0 for levels below `finest`, e.g. the
    /// full-res chunks, which never droop).
    pub fn droop(&self, lod: Lod, cfg: &PyramidCfg) -> i32 {
        let i = lod.0.saturating_sub(cfg.finest.0) as usize;
        self.per_level.get(i).copied().unwrap_or(0)
    }
}

/// The generator height at `(wx, wz)` snapped to the centre of its level-`k`
/// (`2^k`-metre) cell — the coarse height a level-`k` tile would show.
fn sample_snapped<G: TerrainGenerator>(generator: &G, wx: i64, wz: i64, k: u8) -> i32 {
    let cell = 1i64 << k;
    let snap = |v: i64| (v.div_euclid(cell) * cell + cell / 2) as i32;
    generator.height(snap(wx), snap(wz))
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

    /// Calibration is deterministic, capped, and coarser-drooping-further. A flat
    /// generator (no disagreement between grids) calibrates to zero droop.
    #[test]
    fn droop_is_deterministic_capped_and_monotone() {
        use crate::block::registry::{AIR, BlockId, BlockRegistry};

        struct Flat;
        impl TerrainGenerator for Flat {
            fn height(&self, _: i32, _: i32) -> i32 {
                40
            }
            fn surface_at(&self, _: i32, _: i32) -> BlockId {
                AIR
            }
            fn deep(&self) -> BlockId {
                AIR
            }
        }
        let cfg = d1();
        let t = DroopTable::calibrate(&Flat, &cfg);
        assert_eq!(t.droop(Lod(2), &cfg), 0, "a flat generator never droops");
        assert_eq!(t.droop(Lod(4), &cfg), 0);

        // A real generator: droop is bounded by the cap and non-negative, and the
        // sweep is reproducible run to run.
        let generator =
            crate::world::generation::SineHills::new(&BlockRegistry::with_builtins(), 20.0, 7);
        let a = DroopTable::calibrate(&generator, &cfg);
        let b = DroopTable::calibrate(&generator, &cfg);
        for lod in cfg.active_lods() {
            assert_eq!(a.droop(lod, &cfg), b.droop(lod, &cfg), "deterministic");
            assert!((0..=DROOP_CAP).contains(&a.droop(lod, &cfg)), "within cap");
        }
    }
}
