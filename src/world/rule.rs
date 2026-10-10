//! The level-of-detail rule. The eye's physical distance to a node's physical box decides its
//! detail: within the render distance `R` the full-res chunks draw, and beyond it detail `d`
//! reaches out to `R·2^(d − finest)`, so every level is a shell around the eye. Charts, chart
//! seams, cube faces, twins and flat worlds are one case: every box is measured where it really is.

use voxel_engine::DVec3;

use super::World;
use super::section::{CHART_BODY_BASE, SectionPos};
use crate::space::FaceFrame;

/// Render distance per finest cell edge: the finest far level is the one whose section span is
/// `R / 2` to `R / 4` (`2^d_f ≤ R / 64`), so its tile count stays the same at every distance.
const FINEST_RATIO: f64 = 64.0;

/// Blocks a node's terrain bounds are widened by below (digs) and above (plants) the sampled tops.
const DRESS: f64 = 8.0;

/// Doublings of the render distance the far field reaches: its horizon is `R·2^8`.
const HORIZON_DOUBLINGS: i32 = 8;

/// The rule at one render distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::world) struct Rule {
    /// Render distance in blocks.
    pub r: f64,
    /// Finest far detail.
    pub finest: i8,
    /// Coarsest detail any node takes.
    pub coarsest: i8,
}

impl Rule {
    /// The rule at render distance `r` blocks, never coarser than `coarsest`.
    pub fn new(r: f64, coarsest: i8) -> Rule {
        let finest = ((r / FINEST_RATIO).log2().floor().max(0.0) as i8).min(coarsest);
        Rule { r, finest, coarsest }
    }

    /// Distance out to which detail `d` stays split into detail `d − 1`.
    pub fn range(&self, d: i8) -> f64 {
        self.r * 2f64.powi(i32::from(d) - i32::from(self.finest))
    }

    /// Whether a node of detail `d` at distance `dist` splits.
    pub fn splits(&self, d: i8, dist: f64) -> bool {
        d > self.finest && dist < self.range(d)
    }

    /// Distance past which nothing of the far field draws.
    pub fn horizon(&self) -> f64 {
        self.r * 2f64.powi(HORIZON_DOUBLINGS)
    }

    /// This rule never coarser than `coarsest` (a chart's chord rule).
    pub fn capped(self, coarsest: i8) -> Rule {
        Rule { coarsest: self.coarsest.min(coarsest).max(self.finest), ..self }
    }
}

/// An axis-aligned box in physical space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::world) struct PhysBox {
    pub lo: DVec3,
    pub hi: DVec3,
}

impl PhysBox {
    /// The smallest box holding `points`, widened by `pad`.
    fn around(points: impl IntoIterator<Item = DVec3>, pad: f64) -> Option<PhysBox> {
        let mut it = points.into_iter();
        let first = it.next()?;
        let (lo, hi) = it.fold((first, first), |(lo, hi), p| (lo.min(p), hi.max(p)));
        let pad = DVec3::splat(pad);
        (lo.is_finite() && hi.is_finite()).then_some(PhysBox { lo: lo - pad, hi: hi + pad })
    }

    /// Euclidean distance from `p` to the nearest point of the box (`0` inside).
    pub fn dist(&self, p: DVec3) -> f64 {
        (self.lo - p).max(p - self.hi).max(DVec3::ZERO).length()
    }
}

impl World {
    /// The terrain altitudes of `s`'s footprint, in its own frame (chart storage `y`, a cube face's
    /// local altitude), widened for digs and plants. An edited square reads its overlay, a baked one
    /// the relief bake, the rest the generator's bounds.
    fn node_bounds(&self, s: SectionPos) -> Option<(f64, f64)> {
        let baked = self.section_relief_band(s).map(|(lo, hi)| (f64::from(self.baked_world_y(s, lo)), f64::from(self.baked_world_y(s, hi))));
        let (lo, hi) = match baked {
            Some(b) => b,
            None => {
                let (lo, hi) = self.generator.surface_bounds(s.body, s.face, s.min_x(), s.min_z(), s.span())?;
                (f64::from(lo), f64::from(hi))
            }
        };
        Some((lo.min(hi) - DRESS, lo.max(hi) + DRESS))
    }

    /// Section `s`'s physical box: its footprint between its lowest and highest terrain, placed
    /// through its chart, the warp of its cube, or its face frame.
    pub(in crate::world) fn node_box(&self, s: SectionPos) -> Option<PhysBox> {
        let (lo, hi) = self.node_bounds(s)?;
        let span = f64::from(s.span());
        let (x0, z0) = (f64::from(s.min_x()), f64::from(s.min_z()));
        let corners = (0..8).map(|c| DVec3::new(x0 + span * f64::from(c & 1), if c & 2 == 0 { lo } else { hi }, z0 + span * f64::from((c >> 2) & 1)));
        let bend = self.chart_bend(s);
        if s.body >= CHART_BODY_BASE {
            // A chart tile is curved between its corners: its sagitta `L²/8R` stays within one
            // block by the chord rule, but a far tile may not.
            let bend = bend?;
            let pad = span * span / (8.0 * bend.atlas.radius.max(1) as f64) + 1.0;
            return PhysBox::around(corners.map(|p| bend.atlas.embed_storage(bend.patch, p)), pad);
        }
        let frame = FaceFrame::new(s.face);
        let world = |p: DVec3| {
            let (x, y, z) = frame.cell_to_world((p.x as i32, p.y as i32, p.z as i32));
            DVec3::new(f64::from(x), f64::from(y), f64::from(z))
        };
        // A warp moves every point by at most its largest displacement.
        let pad = bend.as_ref().and_then(|b| b.atlas.warp.as_ref()).map_or(0.0, |w| w.max_displacement()) + 1.0;
        PhysBox::around(corners.map(world), pad)
    }
}

#[cfg(test)]
mod probes;

#[cfg(test)]
mod tests {
    use super::*;

    /// The finest far detail keeps two to four of its section spans inside the render distance.
    #[test]
    fn finest_far_detail_follows_the_render_distance() {
        for (rd, finest) in [(2, 0), (4, 0), (6, 0), (8, 1), (12, 1), (16, 2), (24, 2), (32, 3)] {
            let rule = Rule::new(f64::from(rd * 16), 9);
            assert_eq!(rule.finest, finest, "render distance {rd}");
            let span = 32.0 * 2f64.powi(i32::from(finest));
            assert!(rule.r >= span && (rule.r < 4.0 * span || finest == 0), "render distance {rd}: span {span}");
        }
        let rule = Rule::new(256.0, 8);
        assert_eq!((rule.range(2), rule.range(3), rule.range(8)), (256.0, 512.0, 16_384.0));
        assert!(rule.splits(3, 511.0) && !rule.splits(3, 512.0) && !rule.splits(2, 0.0));
    }

    /// Distance to a box is zero inside and Euclidean outside.
    #[test]
    fn box_distance_is_euclidean() {
        let b = PhysBox { lo: DVec3::new(0.0, 0.0, 0.0), hi: DVec3::new(10.0, 2.0, 10.0) };
        assert_eq!(b.dist(DVec3::new(5.0, 1.0, 5.0)), 0.0);
        assert_eq!(b.dist(DVec3::new(13.0, 6.0, 5.0)), 5.0);
        assert_eq!(b.dist(DVec3::new(-3.0, 1.0, -4.0)), 5.0);
    }
}
