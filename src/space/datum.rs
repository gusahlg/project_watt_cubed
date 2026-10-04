//! A body's datum surface: how far its relaxed shape departs from a sphere, as radius offsets over the
//! six equiangular cube-sphere faces.
//!
//! Physics decides a body's shape (its cube of matter relaxes under its own gravity, see
//! `mechanics::genesis`); the layout is then fitted to that shape (owner decision 2026-10-04,
//! guide §10.2): an atlas of radius `R` plus these offsets. Samples sit on a regular grid of the
//! equiangular face parameters `(ξ, η) ∈ [−1, 1]²` (the same parameters the atlas maps cells
//! through), so the two faces of a seam sample the same physical directions along their shared edge
//! and bilinear interpolation is continuous across every seam and corner.

use glam::DVec3;

use crate::space::atlas::FACES;
use crate::space::chart::Map;
use crate::space::chart::basis;

/// Radius offsets of a body's datum surface.
#[derive(Clone, Debug, PartialEq)]
pub struct DatumField {
    /// Samples per face edge (`g × g` per face, including the edges).
    pub g: usize,
    /// Offsets in blocks, [`FACES`] order, index `face · g² + j · g + i` (`i` along ξ, `j` along η).
    pub offsets: Vec<f32>,
}

impl DatumField {
    /// A spherical datum (all offsets zero).
    pub fn flat(g: usize) -> Self {
        assert!(g >= 2);
        Self { g, offsets: vec![0.0; 6 * g * g] }
    }

    /// Build from a radius function of direction, relative to `radius`.
    pub fn sample(g: usize, radius: f64, mut r_of: impl FnMut(DVec3) -> f64) -> Self {
        let mut offsets = Vec::with_capacity(6 * g * g);
        for f in 0..6 {
            for j in 0..g {
                for i in 0..g {
                    let d = Self::direction(f, g, i, j);
                    offsets.push((r_of(d) - radius) as f32);
                }
            }
        }
        Self { g, offsets }
    }

    /// The unit direction of sample `(i, j)` on face `f` ([`FACES`] order).
    pub fn direction(f: usize, g: usize, i: usize, j: usize) -> DVec3 {
        let step = 2.0 / (g - 1) as f64;
        let d = Map::Equiangular.dir(-1.0 + i as f64 * step, -1.0 + j as f64 * step);
        let (tu, nn, tv) = basis(FACES[f]);
        (tu * d.x + nn * d.y + tv * d.z).normalize()
    }

    /// Offset (blocks) at face parameters `(ξ, η)` of face `f` ([`FACES`] order), bilinear.
    pub fn offset(&self, f: usize, xi: f64, eta: f64) -> f64 {
        let g = self.g;
        let s = (g - 1) as f64;
        let u = ((xi + 1.0) * 0.5 * s).clamp(0.0, s);
        let v = ((eta + 1.0) * 0.5 * s).clamp(0.0, s);
        let (i0, j0) = ((u.floor() as usize).min(g - 2), (v.floor() as usize).min(g - 2));
        let (fu, fv) = (u - i0 as f64, v - j0 as f64);
        let at = |i: usize, j: usize| self.offsets[f * g * g + j * g + i] as f64;
        let a = at(i0, j0) * (1.0 - fu) + at(i0 + 1, j0) * fu;
        let b = at(i0, j0 + 1) * (1.0 - fu) + at(i0 + 1, j0 + 1) * fu;
        a * (1.0 - fv) + b * fv
    }

    /// Offset (blocks) along unit direction `d` from the body's centre.
    pub fn at(&self, d: DVec3) -> f64 {
        let face = crate::coord::Face::from_dominant(d);
        let (tu, nn, tv) = basis(face);
        let (xi, eta) = Map::Equiangular.inverse(DVec3::new(d.dot(tu), d.dot(nn), d.dot(tv)));
        let f = FACES.iter().position(|&g| g == face).expect("a face");
        self.offset(f, xi, eta)
    }

    /// Smallest and largest offset.
    pub fn range(&self) -> (f64, f64) {
        let lo = self.offsets.iter().copied().fold(f32::INFINITY, f32::min) as f64;
        let hi = self.offsets.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        (lo, hi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_seam_samples_the_same_direction_from_both_faces() {
        let g = 9;
        // The +X face's ξ = +1 edge and every neighbour edge: some sample of another face has the
        // same direction as each edge sample.
        for j in 0..g {
            let d = DatumField::direction(0, g, g - 1, j);
            let twin = (1..6).flat_map(|f| (0..g).flat_map(move |jj| (0..g).map(move |ii| (f, ii, jj)))).any(|(f, ii, jj)| (DatumField::direction(f, g, ii, jj) - d).length() < 1e-9);
            assert!(twin, "edge sample {j} has no twin");
        }
    }

    #[test]
    fn a_radius_function_round_trips_through_the_field() {
        let r = |d: DVec3| 1000.0 + 50.0 * d.x + 20.0 * d.y * d.z;
        let field = DatumField::sample(33, 1000.0, r);
        let (lo, hi) = field.range();
        assert!(lo < -40.0 && hi > 40.0);
        // At a sample point the offset is exact; between samples bilinear is close for a smooth field.
        let f = 2;
        let d = DatumField::direction(f, 33, 16, 16);
        assert!((field.offset(f, 0.0, 0.0) - (r(d) - 1000.0)).abs() < 1e-3);
        let between = field.offset(f, 0.03, -0.41);
        assert!(between.is_finite());
    }
}
