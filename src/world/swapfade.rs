//! Temporal cross-fade for the section far field — presentation only. Selection
//! (the desired cut) is unchanged; this dissolves between successive cuts over
//! [`FADE_SECS`] instead of popping a band swap in one frame. Incoming cells fade in,
//! outgoing cells fade out, and the shader uses dithering to keep the coverage exact —
//! no holes, no overlap — during the transition.
//!
//! T1/H1 are untouched: this is a view of the selection state, not part of it. A cell
//! is kept loaded while it fades out ([`SwapFade::tracked`]), and the fade converges to
//! the new cut regardless of how many swaps interrupt it.

use std::time::Instant;

use super::FastMap;
use super::quadtree::QuadrantMask;
use super::section::SectionPos;

/// Cross-fade duration. ~150 ms reads as a smooth screen-door dissolve without lagging
/// the LOD change perceptibly.
pub(in crate::world) const FADE_SECS: f32 = 0.15;

/// One tracked cell: its draw mask, fade progress (0 to 1), and whether it's fading out.
#[derive(Clone, Copy, Debug, PartialEq)]
struct FadeCell {
    mask: QuadrantMask,
    fade: f32,
    out: bool,
}

/// The far-field cross-fade state: cells currently drawing with their fade progress.
#[derive(Default)]
pub(in crate::world) struct SwapFade {
    cells: FastMap<SectionPos, FadeCell>,
    last: Option<Instant>,
}

impl SwapFade {
    /// Advance by wall-clock time since the last call, then adopt `cut`. First call steps
    /// by zero, so no sudden fade jumps on startup.
    pub fn update_now(&mut self, cut: &[(SectionPos, QuadrantMask)]) {
        let now = Instant::now();
        let dt = self.last.map_or(0.0, |t| now.duration_since(t).as_secs_f32());
        self.last = Some(now);
        self.update(cut, dt);
    }

    /// Adopt a new desired `cut` and advance every cell's fade by `dt` seconds. New cells
    /// fade in; vanished cells fade out (kept until they reach 0). If a cell is re-desired
    /// before fading fully out, it reverses direction and fades back in.
    pub fn update(&mut self, cut: &[(SectionPos, QuadrantMask)], dt: f32) {
        let step = if FADE_SECS > 0.0 { (dt / FADE_SECS).max(0.0) } else { 1.0 };
        let desired: FastMap<SectionPos, QuadrantMask> = cut.iter().copied().collect();
        // Desired cells fade in (reversing if they were fading out) and get fresh masks.
        for (&pos, &mask) in &desired {
            let e = self.cells.entry(pos).or_insert(FadeCell { mask, fade: 0.0, out: false });
            e.mask = mask;
            e.out = false;
        }
        for (pos, c) in self.cells.iter_mut() {
            if !desired.contains_key(pos) {
                c.out = true;
            }
        }
        self.cells.retain(|_, c| {
            if c.out {
                c.fade -= step;
                c.fade > 0.0
            } else {
                c.fade = (c.fade + step).min(1.0);
                true
            }
        });
    }

    /// The cells to draw this frame: `(pos, mask, fade, fade_out)`. `fade_out` selects the
    /// complementary dither phase in the shader.
    pub fn draws(&self) -> impl Iterator<Item = (SectionPos, QuadrantMask, f32, bool)> + '_ {
        self.cells.iter().map(|(&p, c)| (p, c.mask, c.fade, c.out))
    }

    /// Every tracked cell (fading or not) — the set selection must keep loaded so an
    /// outgoing mesh is not unloaded mid-dissolve.
    pub fn tracked(&self) -> impl Iterator<Item = SectionPos> + '_ {
        self.cells.keys().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::section::FINEST_DETAIL;

    fn cell(x: i32, detail: u8) -> (SectionPos, QuadrantMask) {
        (SectionPos { detail, x, z: 0 }, QuadrantMask::ALL)
    }
    fn drawn(f: &SwapFade) -> std::collections::HashMap<SectionPos, (f32, bool)> {
        f.draws().map(|(p, _m, fade, out)| (p, (fade, out))).collect()
    }
    /// Advance a step big enough to complete any fade in one call.
    const DONE: f32 = FADE_SECS * 2.0;

    /// A completed fade leaves EXACTLY the new cut drawn, every cell solid (fade 1, not
    /// out) — the swap-fade end state.
    #[test]
    fn fade_completes_to_exactly_the_new_cut() {
        let mut f = SwapFade::default();
        let old = [cell(0, FINEST_DETAIL + 1), cell(1, FINEST_DETAIL + 1)];
        f.update(&old, DONE);
        let new = [cell(5, FINEST_DETAIL + 2), cell(6, FINEST_DETAIL + 2)];
        f.update(&new, DONE);
        let d = drawn(&f);
        assert_eq!(d.len(), new.len(), "exactly the new cut draws after the fade");
        for (pos, _) in new {
            assert_eq!(d.get(&pos), Some(&(1.0, false)), "new cell solid");
        }
        for (pos, _) in old {
            assert!(!d.contains_key(&pos), "old cell fully gone");
        }
    }

    /// Mid-fade both cuts draw: incoming partway in (not out), outgoing partway out.
    #[test]
    fn mid_fade_draws_both_with_opposite_directions() {
        let mut f = SwapFade::default();
        f.update(&[cell(0, FINEST_DETAIL + 1)], DONE);
        f.update(&[cell(9, FINEST_DETAIL + 1)], FADE_SECS * 0.5);
        let d = drawn(&f);
        let (in_fade, in_out) = d[&SectionPos { detail: FINEST_DETAIL + 1, x: 9, z: 0 }];
        let (out_fade, out_out) = d[&SectionPos { detail: FINEST_DETAIL + 1, x: 0, z: 0 }];
        assert!(!in_out && in_fade > 0.0 && in_fade < 1.0, "incoming is fading in");
        assert!(out_out && out_fade > 0.0 && out_fade < 1.0, "outgoing is fading out");
    }

    /// An interrupted swap converges: swap to B partway, then back to A before B finishes
    /// fading in. A recovers to solid and B is removed.
    #[test]
    fn interrupted_swap_converges_to_the_latest_cut() {
        let mut f = SwapFade::default();
        let a = [cell(0, FINEST_DETAIL + 1)];
        let b = [cell(1, FINEST_DETAIL + 1)];
        f.update(&a, DONE);
        f.update(&b, FADE_SECS * 0.4);
        f.update(&b, FADE_SECS * 0.2);
        f.update(&a, DONE); // complete the fade back to A
        let d = drawn(&f);
        assert_eq!(d.len(), 1, "converged to a single cut");
        assert_eq!(d.get(&SectionPos { detail: FINEST_DETAIL + 1, x: 0, z: 0 }), Some(&(1.0, false)));
    }

    /// At rest (the same cut every frame) nothing fades: all draws are solid, fade 1.
    #[test]
    fn steady_cut_has_no_active_fade() {
        let mut f = SwapFade::default();
        let cut = [cell(0, FINEST_DETAIL + 1), cell(2, FINEST_DETAIL + 3)];
        for _ in 0..4 {
            f.update(&cut, FADE_SECS); // several frames, same cut
        }
        for (_p, _m, fade, out) in f.draws() {
            assert_eq!((fade, out), (1.0, false), "steady state is all solid, no fade");
        }
        assert_eq!(f.tracked().count(), cut.len());
    }

    /// Equal inputs (state, cut, dt) produce equal draw sets (order-independent).
    #[test]
    fn update_is_deterministic() {
        let seq = [
            (vec![cell(0, 3), cell(1, 3)], 0.05f32),
            (vec![cell(1, 3), cell(2, 4)], 0.05),
            (vec![cell(2, 4)], 0.05),
        ];
        let run = || {
            let mut f = SwapFade::default();
            for (cut, dt) in &seq {
                f.update(cut, *dt);
            }
            let mut v: Vec<_> = f.draws().map(|(p, _m, fade, out)| (p, fade.to_bits(), out)).collect();
            v.sort_by_key(|(p, _, _)| (p.detail, p.x, p.z));
            v
        };
        assert_eq!(run(), run());
    }
}
