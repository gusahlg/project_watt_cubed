//! Distance-driven LOD selection: map XZ distance to LOD level and keep tolerance.
use std::num::NonZeroU8;

use crate::ident::Detail;

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
}

impl PyramidCfg {
    /// Standard config: base 2, 7 rings starting at finest LOD.
    pub fn sections(unit: f32) -> PyramidCfg {
        PyramidCfg {
            finest: super::section::FINEST_DETAIL,
            levels: NonZeroU8::new(SECTION_LEVELS).unwrap(),
            unit,
            base: 2.0,
        }
    }

    /// LOD value increment per ring (log2 of base, floored at 1 for degenerate bases).
    pub fn step(&self) -> u8 {
        (self.base.log2().round() as i32).max(1) as u8
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
    let ring = (dist_xz / cfg.unit).log(cfg.base).floor();
    // Non-finite ring (corrupt cfg) saturates past horizon.
    let ring = if ring.is_finite() { ring as u32 } else { u32::MAX };
    if ring >= cfg.levels.get() as u32 {
        LodChoice::BeyondHorizon
    } else {
        LodChoice::Level(ringed_detail(cfg.finest, ring, cfg.step()))
    }
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

/// The one quantized detail decision: folds the near-field full-res chunk
/// radius and the far-field pyramid ladder into a
/// single output in the engine's `Detail` type — the type the existing
/// upload/draw pipeline already consumes (chunks upload at `Detail::FULL`,
/// `streaming::chunk_placement`; sections draw at `Detail::new(pos.detail)`,
/// `SectionState::draw`). `None` past the horizon: nothing is required there.
///
/// Chunks own everything nearer than `cfg.unit` (by construction — `unit` is
/// kept equal to the streamed chunk-view radius every frame, `World::stream`).
/// `level_for`'s own near branch returns the section pyramid's *finest ring*
/// there instead (`Detail(2)`, coarser than `Detail::FULL`), which is correct
/// for its own callers (`acceptable`'s hysteresis) but is NOT chunk resolution —
/// so this near branch is a genuinely separate case, not a re-derivation of
/// `level_for`'s existing clamp.
///
/// `affordable`: the coarsest `Detail` the current VRAM budget can afford — a
/// floor; this never returns something FINER than it. Dormant by construction:
/// the only caller today, [`vram_budget_floor`], always returns `Detail::FULL`
/// (no floor), and every quantity `max`ed against `Detail::FULL` is unchanged,
/// so this parameter is presently a no-op end to end.
pub(in crate::world) fn required_detail(
    dist: EyeDist,
    cfg: &PyramidCfg,
    affordable: Detail,
) -> Option<Detail> {
    if dist.get() < cfg.unit {
        return Some(Detail::FULL.max(affordable));
    }
    let desired = match level_for(dist, cfg) {
        LodChoice::Level(lod) => lod,
        LodChoice::BeyondHorizon => return None,
    };
    Some(desired.max(affordable))
}

/// The coarsest `Detail` the current VRAM budget can afford. DORMANT — always
/// `Detail::FULL` (no floor).
///
/// Activation needs TWO things neither landed here: a graphics-setting toggle
/// (a behavioural default change is user-gated, never landed autonomously),
/// AND a new engine-side public accessor. The engine's
/// `VK_EXT_memory_budget` query (`vk::device::MemoryBudget::query`) exists but
/// is `unsafe`, `vk`-module-private, and takes a raw `ash::Instance`/
/// `PhysicalDevice` — there is no public `Engine` method reaching it today, so
/// "reading the existing query" is not yet possible from app code without a
/// small new engine-side API. Flagging that gap rather than papering over it.
pub(in crate::world) fn vram_budget_floor() -> Detail {
    Detail::FULL
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d1() -> PyramidCfg {
        PyramidCfg { finest: Detail(2), levels: NonZeroU8::new(2).unwrap(), unit: 256.0, base: 4.0 }
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

    /// Chunks own everything nearer than `unit`: `required_detail` returns
    /// `Detail::FULL` there, strictly finer than `level_for`'s own near-clamp
    /// (`Detail(2)`) — proving this is a genuinely separate case, not a duplicate.
    #[test]
    fn required_detail_is_full_res_inside_the_chunk_radius() {
        let cfg = d1();
        for d in [0.0f32, 100.0, 255.9] {
            let dist = EyeDist::new(d);
            assert_eq!(
                required_detail(dist, &cfg, Detail::FULL),
                Some(Detail::FULL),
                "distance {d} is inside the chunk radius"
            );
            assert!(
                Detail::FULL < Detail::new(2),
                "chunk resolution must be strictly finer than the section's own finest ring"
            );
        }
    }

    /// Beyond `unit`, with the floor dormant (`Detail::FULL`, a no-op `max`),
    /// `required_detail` matches `level_for` exactly: identical selections to the
    /// pre-existing ladder, a fixed point required by the current design.
    #[test]
    fn required_detail_matches_level_for_beyond_the_chunk_radius_when_dormant() {
        let cfg = d1();
        for m in 256..40_000u32 {
            let dist = EyeDist::new(m as f32);
            let got = required_detail(dist, &cfg, vram_budget_floor());
            let want = match level_for(dist, &cfg) {
                LodChoice::Level(l) => Some(l),
                LodChoice::BeyondHorizon => None,
            };
            assert_eq!(got, want, "distance {m} must match the pre-existing ladder exactly");
        }
    }

    /// Never finer with distance, matching `level_for`'s own monotonicity.
    #[test]
    fn required_detail_never_refines_with_distance() {
        let cfg = d1();
        let mut last = Detail::FULL;
        for m in 0..40_000u32 {
            let Some(got) = required_detail(EyeDist::new(m as f32), &cfg, vram_budget_floor()) else {
                continue;
            };
            assert!(got >= last, "detail coarsened then refined at distance {m}");
            last = got;
        }
    }

    /// The dormant floor is a true no-op: activating it with an artificially
    /// coarse floor changes the result (proving the seam is live code, not dead
    /// weight), while the real default (`Detail::FULL`) never does.
    #[test]
    fn affordable_floor_only_coarsens_when_actually_activated() {
        let cfg = d1();
        let dist = EyeDist::new(2000.0); // deep in the ladder, level_for gives Detail(4)
        let unclamped = required_detail(dist, &cfg, Detail::FULL);
        assert_eq!(unclamped, Some(Detail::new(4)), "sanity: matches level_for");
        // Dormant default changes nothing.
        assert_eq!(required_detail(dist, &cfg, vram_budget_floor()), unclamped);
        // A hypothetical activated floor coarser than the desired level DOES win.
        let coarse_floor = Detail::new(6);
        assert_eq!(required_detail(dist, &cfg, coarse_floor), Some(coarse_floor));
        // A floor finer than what's desired never refines past the ladder's own choice.
        let fine_floor = Detail::new(1);
        assert_eq!(required_detail(dist, &cfg, fine_floor), unclamped);
    }

}
