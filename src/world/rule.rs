//! The level-of-detail rule. The eye's physical distance to a node's physical box decides its
//! detail: within the render distance `R` the full-res chunks draw, and beyond it a tile of detail
//! `d` (span `s = 32·2^d`) splits while the eye is within `2s`, so every level is a shell twice as
//! far out and twice as coarse as the one inside it, and each shell holds the same count of tiles
//! at every render distance. Charts, chart seams, cube faces, twins and flat worlds are one case:
//! every box is measured where it really is.
//!
//! The eye is the 16-block physical cell that holds it ([`PhysBox::eye_cell`]): the cut and the
//! near columns are measured from the nearest point of that cell, so they change only when the
//! eye crosses into another cell, and anything within `R` of the eye itself is within `R` of it.

use voxel_engine::DVec3;

use super::World;
use super::section::{CHART_BODY_BASE, SectionPos};
use crate::space::FaceFrame;

/// Section spans a tile reaches before it splits: detail `d` holds out to `2·32·2^d` blocks.
const SPLIT_SPANS: f64 = 2.0;

/// The finest far level is the one whose span is `R / 4` to `R / 2`: it splits no further, and its
/// parent's reach (`4` of its spans) clears the render distance.
const FINEST_RATIO: f64 = 64.0;

/// Blocks a node's terrain bounds are widened by below (digs) and above (plants) the sampled tops.
pub(in crate::world) const DRESS: f64 = 8.0;

/// Doublings of the render distance the far field reaches: its horizon is `R·2^8`.
const HORIZON_DOUBLINGS: i32 = 8;

/// Widest tile whose terrain bounds are read from the sampled surface rather than the generator's
/// band: nearer tiles need tight boxes, farther ones are cheap to over-split.
const TIGHT_SPAN: i32 = 256;

/// Side of the physical cell the eye is measured from.
pub(in crate::world) const EYE_CELL: f64 = 16.0;

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

    /// Distance out to which a tile of detail `d` and nominal span `32·2^d` splits into detail
    /// `d − 1`. Tied to the tile's size; it equals `R·2^(d − finest)` when `R = 64·2^k`.
    #[cfg(test)]
    pub fn range(&self, d: i8) -> f64 {
        self.range_of(32.0 * 2f64.powi(i32::from(d)))
    }

    /// Distance out to which a tile `span` blocks across (physically) splits.
    pub fn range_of(&self, span: f64) -> f64 {
        SPLIT_SPANS * span
    }

    /// Whether a node of detail `d`, `span` blocks across where it really is, at distance `dist`
    /// splits. A chart's cells are not all one block: a tile is as large as its footprint.
    pub fn splits(&self, d: i8, dist: f64, span: f64) -> bool {
        d > self.finest && dist < self.range_of(span)
    }

    /// Distance past which nothing of the far field draws.
    pub fn horizon(&self) -> f64 {
        self.r * 2f64.powi(HORIZON_DOUBLINGS)
    }

    /// This rule never coarser than `coarsest` (a chart's chord rule): a ladder that short ends
    /// there at both ends.
    pub fn capped(self, coarsest: i8) -> Rule {
        let coarsest = self.coarsest.min(coarsest);
        Rule { coarsest, finest: self.finest.min(coarsest), ..self }
    }
}

/// An axis-aligned box in physical space around a node, with how far across the node's
/// footprint really is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::world) struct PhysBox {
    pub lo: DVec3,
    pub hi: DVec3,
    /// The footprint's physical side in blocks.
    pub span: f64,
}

impl PhysBox {
    /// The smallest box holding `points`, widened by `pad`, around a footprint `span` across.
    pub fn around(points: impl IntoIterator<Item = DVec3>, pad: f64, span: f64) -> Option<PhysBox> {
        let mut it = points.into_iter();
        let first = it.next()?;
        let (lo, hi) = it.fold((first, first), |(lo, hi), p| (lo.min(p), hi.max(p)));
        let pad = DVec3::splat(pad);
        (lo.is_finite() && hi.is_finite()).then_some(PhysBox { lo: lo - pad, hi: hi + pad, span })
    }

    /// The box holding only `p`.
    #[cfg(test)]
    pub fn point(p: DVec3) -> PhysBox {
        PhysBox { lo: p, hi: p, span: 0.0 }
    }

    /// The [`EYE_CELL`] cell holding physical eye `eye`.
    pub fn eye_cell(eye: DVec3) -> PhysBox {
        let lo = (eye / EYE_CELL).floor() * EYE_CELL;
        PhysBox { lo, hi: lo + DVec3::splat(EYE_CELL), span: EYE_CELL }
    }

    /// Euclidean distance from `p` to the nearest point of the box (`0` inside).
    #[cfg(test)]
    pub fn dist(&self, p: DVec3) -> f64 {
        (self.lo - p).max(p - self.hi).max(DVec3::ZERO).length()
    }

    /// Euclidean distance between the nearest points of the two boxes (`0` when they meet).
    pub fn dist_box(&self, other: &PhysBox) -> f64 {
        (self.lo - other.hi).max(other.lo - self.hi).max(DVec3::ZERO).length()
    }

    pub fn centre(&self) -> DVec3 {
        (self.lo + self.hi) * 0.5
    }
}

impl World {
    /// The terrain altitudes of `s`'s footprint in its own frame (chart storage `y`, a cube face's
    /// local altitude) from an edit overlay or the relief bake, widened for digs and plants. `None`
    /// where neither has the square.
    pub(in crate::world) fn baked_bounds(&self, s: SectionPos) -> Option<(f64, f64)> {
        let (lo, hi) = self.section_relief_band(s)?;
        let (lo, hi) = (f64::from(self.baked_world_y(s, lo)), f64::from(self.baked_world_y(s, hi)));
        Some((lo.min(hi) - DRESS, lo.max(hi) + DRESS))
    }

    /// [`baked_bounds`](Self::baked_bounds) from the generator's surface bounds: what the square
    /// can hold before anything is baked or edited. Pure, so it may be kept. A tile up to
    /// [`TIGHT_SPAN`] across reads the sampled surface; a larger one the generator's band.
    pub(in crate::world) fn generator_bounds(&self, s: SectionPos) -> Option<(f64, f64)> {
        let (x0, z0, w) = (s.min_x(), s.min_z(), s.span());
        let (lo, hi) = if w <= TIGHT_SPAN {
            self.generator.surface_rect(s.body, s.face, x0, z0, x0 + w, z0 + w)?
        } else {
            self.generator.surface_bounds(s.body, s.face, x0, z0, w)?
        };
        let (lo, hi) = (f64::from(lo), f64::from(hi));
        Some((lo.min(hi) - DRESS, lo.max(hi) + DRESS))
    }

    /// Section `s`'s physical box: its footprint between its lowest and highest terrain, placed
    /// through its chart, the warp of its cube, or its face frame.
    pub(in crate::world) fn node_box(&self, s: SectionPos) -> Option<PhysBox> {
        let mut memo = self.node_boxes.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        self.node_box_in(&mut memo, s)
    }

    /// [`node_box`](Self::node_box) through a memo the caller already holds.
    pub(in crate::world) fn node_box_in(&self, memo: &mut RuleMemo, s: SectionPos) -> Option<PhysBox> {
        if let Some(&b) = memo.boxes.get(&s) {
            return b;
        }
        if memo.boxes.len() > MEMO_CAP {
            memo.boxes.clear();
        }
        let b = self.baked_bounds(s).or_else(|| self.generator_bounds(s)).and_then(|(lo, hi)| self.place_box(s, lo, hi));
        memo.boxes.insert(s, b);
        b
    }

    /// Forget every measured box: a node's bounds changed.
    pub(in crate::world) fn forget_node_boxes(&self) {
        self.node_boxes.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clear();
    }

    /// `s`'s footprint between frame altitudes `lo` and `hi`, in physical space.
    pub(in crate::world) fn place_box(&self, s: SectionPos, lo: f64, hi: f64) -> Option<PhysBox> {
        let span = f64::from(s.span());
        let (cx, cz) = self.tile_centre(s);
        let (x0, z0) = (cx as f64 - span / 2.0, cz as f64 - span / 2.0);
        let corners = (0..8).map(|c| DVec3::new(x0 + span * f64::from(c & 1), if c & 2 == 0 { lo } else { hi }, z0 + span * f64::from((c >> 2) & 1)));
        let bend = self.chart_bend(s);
        if s.body >= CHART_BODY_BASE {
            // A chart tile is curved between its corners: its sagitta `L²/8R` stays within one
            // block by the chord rule, but a far tile may not.
            let bend = bend?;
            let pad = span * span / (8.0 * bend.atlas.radius.max(1) as f64) + 1.0;
            let embedded: Vec<DVec3> = corners.map(|p| bend.atlas.embed_storage(bend.patch, p)).collect();
            // Its cells are not one block everywhere: the footprint's longer edge, at its floor.
            let side = (embedded[1] - embedded[0]).length().max((embedded[4] - embedded[0]).length());
            return PhysBox::around(embedded, pad, side);
        }
        let frame = FaceFrame::new(s.face);
        let world = |p: DVec3| {
            let (x, y, z) = frame.cell_to_world((p.x as i32, p.y as i32, p.z as i32));
            DVec3::new(f64::from(x), f64::from(y), f64::from(z))
        };
        let b = PhysBox::around(corners.map(world), 1.0, span)?;
        // A warped cube: the box its warp can move the tile to.
        Some(match bend.as_ref().and_then(|b| b.atlas.warp.as_ref()) {
            Some(w) => {
                let (lo, hi) = w.bound_box(b.lo, b.hi);
                PhysBox { lo, hi, span }
            }
            None => b,
        })
    }
}

mod select;
mod surface;
#[cfg(test)]
mod probes;

pub(in crate::world) use select::Leaf;
pub(in crate::world) use surface::{Ground, RuleMemo};
use surface::MEMO_CAP;

#[cfg(test)]
mod tests {
    use super::*;

    /// The finest far detail keeps two to four of its section spans inside the render distance,
    /// so its parent always splits inside it.
    #[test]
    fn finest_far_detail_follows_the_render_distance() {
        for (rd, finest) in [(2, 0), (4, 0), (6, 0), (8, 1), (12, 1), (16, 2), (24, 2), (32, 3)] {
            let rule = Rule::new(f64::from(rd * 16), 9);
            assert_eq!(rule.finest, finest, "render distance {rd}");
            let span = 32.0 * 2f64.powi(i32::from(finest));
            assert!(rule.r >= span && (rule.r < 4.0 * span || finest == 0), "render distance {rd}: span {span}");
            assert!(rule.range(finest + 1) >= rule.r, "render distance {rd}: the finest level's parent splits short of R");
        }
        let rule = Rule::new(256.0, 8);
        assert_eq!((rule.range(2), rule.range(3), rule.range(8)), (256.0, 512.0, 16_384.0));
        assert!(rule.splits(3, 511.0, 256.0) && !rule.splits(3, 512.0, 256.0) && !rule.splits(2, 0.0, 128.0));
    }

    /// Distance to a box is zero inside and Euclidean outside.
    #[test]
    fn box_distance_is_euclidean() {
        let b = PhysBox { lo: DVec3::new(0.0, 0.0, 0.0), hi: DVec3::new(10.0, 2.0, 10.0), span: 10.0 };
        assert_eq!(b.dist(DVec3::new(5.0, 1.0, 5.0)), 0.0);
        assert_eq!(b.dist(DVec3::new(13.0, 6.0, 5.0)), 5.0);
        assert_eq!(b.dist(DVec3::new(-3.0, 1.0, -4.0)), 5.0);
    }
}
