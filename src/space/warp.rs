//! Displacement of a cube body that kept its cube grid: the relaxed lattice minus its undeformed
//! positions. `apply(x) = x + u(x)` is trilinear inside each element and the identity outside the
//! lattice box. The table lattice has one element of void around the matter, so the field tapers
//! off one element outside the cube.

use glam::{DMat3, DVec3};

use crate::mechanics::genesis::Solved;
use crate::mechanics::lattice::{det_lower_bound, invert_trilinear, local_jacobian, trilinear};

/// A body whose warp moves any point by more than this (half a block) keeps its cells in storage.
pub const STORAGE_MOVE: f64 = 0.5;

/// Displacement over a regular reference lattice in physical space.
#[derive(Clone, Debug, PartialEq)]
pub struct Warp {
    /// Physical position of node `(0, 0, 0)`.
    origin: DVec3,
    /// Physical edge of one element.
    cell: f64,
    /// Elements per axis.
    dims: [usize; 3],
    /// Displacement of every node, index [`Warp::node`].
    nodes: Vec<DVec3>,
}

impl Warp {
    /// Scale a tabulated lattice onto a body centred at `centre` (the table's `Solved::scale`
    /// already carries the half-size). Node `(m, m, m)` of a genesis lattice is the origin.
    pub fn from_solved(solved: &Solved, centre: DVec3) -> Self {
        let lat = &solved.lattice;
        let dims = lat.dims;
        let m = [dims[0] / 2, dims[1] / 2, dims[2] / 2];
        let cell_c = lat.cell as f64;
        let cell = cell_c * solved.scale;
        let origin = centre - DVec3::new(m[0] as f64, m[1] as f64, m[2] as f64) * cell;
        let mut nodes = Vec::with_capacity(lat.nodes.len());
        for k in 0..=dims[2] {
            for j in 0..=dims[1] {
                for i in 0..=dims[0] {
                    let undeformed = DVec3::new(
                        (i as f64 - m[0] as f64) * cell_c,
                        (j as f64 - m[1] as f64) * cell_c,
                        (k as f64 - m[2] as f64) * cell_c,
                    );
                    let relaxed = lat.nodes[lat.node(i, j, k)];
                    nodes.push((relaxed - undeformed) * solved.scale);
                }
            }
        }
        Self { origin, cell, dims, nodes }
    }

    /// Largest nodal displacement, in blocks.
    pub fn max_displacement(&self) -> f64 {
        self.nodes.iter().fold(0.0_f64, |m, u| m.max(u.length()))
    }

    #[inline]
    fn node(&self, i: usize, j: usize, k: usize) -> usize {
        i + (self.dims[0] + 1) * (j + (self.dims[1] + 1) * k)
    }

    fn hi(&self) -> DVec3 {
        self.origin + DVec3::new(self.dims[0] as f64, self.dims[1] as f64, self.dims[2] as f64) * self.cell
    }

    /// Whether `x` lies in the lattice box `[origin, origin + dims·cell)`.
    pub fn contains(&self, x: DVec3) -> bool {
        let hi = self.hi();
        (0..3).all(|a| x[a] >= self.origin[a] && x[a] < hi[a])
    }

    /// `p` clamped just inside the box, or `None` when it is further than the field can reach.
    fn pull_inside(&self, p: DVec3) -> Option<DVec3> {
        let hi = self.hi();
        let margin = self.max_displacement() + 1.0;
        if (0..3).any(|a| p[a] < self.origin[a] - margin || p[a] > hi[a] + margin) {
            return None;
        }
        let eps = (self.cell * 1e-12).max(1e-6);
        let mut x = p;
        for a in 0..3 {
            x[a] = x[a].clamp(self.origin[a] + eps, hi[a] - eps);
        }
        Some(x)
    }

    /// Inverse of one element's deformed trilinear, as a reference point.
    fn element_inverse(&self, ijk: [usize; 3], p: DVec3) -> Option<DVec3> {
        let base = self.origin + DVec3::new(ijk[0] as f64, ijk[1] as f64, ijk[2] as f64) * self.cell;
        let u = self.corners(ijk);
        let deformed = std::array::from_fn(|c| {
            let d = DVec3::new((c & 1) as f64, ((c >> 1) & 1) as f64, ((c >> 2) & 1) as f64);
            base + d * self.cell + u[c]
        });
        let t = invert_trilinear(&deformed, p)?;
        let x = base + t * self.cell;
        self.contains(x).then_some(x)
    }

    /// Element and local coordinates of a reference point inside the box.
    fn locate(&self, x: DVec3) -> Option<([usize; 3], DVec3)> {
        if !self.contains(x) {
            return None;
        }
        let mut ijk = [0usize; 3];
        let mut t = DVec3::ZERO;
        for a in 0..3 {
            let u = (x[a] - self.origin[a]) / self.cell;
            let e = (u.floor() as usize).min(self.dims[a] - 1);
            ijk[a] = e;
            t[a] = u - e as f64;
        }
        Some((ijk, t))
    }

    fn corners(&self, ijk: [usize; 3]) -> [DVec3; 8] {
        let [i, j, k] = ijk;
        std::array::from_fn(|c| {
            self.nodes[self.node(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1))]
        })
    }

    fn displacement(&self, x: DVec3) -> DVec3 {
        let Some((ijk, t)) = self.locate(x) else { return DVec3::ZERO };
        trilinear(&self.corners(ijk), t)
    }

    /// `x + u(x)`. The identity outside the lattice box.
    pub fn apply(&self, x: DVec3) -> DVec3 {
        x + self.displacement(x)
    }

    /// `∂apply/∂x`. The identity outside the lattice box.
    pub fn jacobian(&self, x: DVec3) -> DMat3 {
        let Some((ijk, t)) = self.locate(x) else { return DMat3::IDENTITY };
        DMat3::IDENTITY + local_jacobian(&self.corners(ijk), t) * (1.0 / self.cell)
    }

    /// Reference point whose image is `p`. Newton from `p − u(p)`. A face can bow past the lattice
    /// box, where `apply` is the identity, so an outside start is pulled just inside. `None` when
    /// it does not converge or the solution leaves the lattice.
    pub fn invert(&self, p: DVec3) -> Option<DVec3> {
        let guess = p - self.displacement(p);
        let mut x = if self.contains(guess) { guess } else { self.pull_inside(p)? };
        for _ in 0..12 {
            let Some((ijk, _)) = self.locate(x) else { return None };
            if let Some(hit) = self.element_inverse(ijk, p) {
                let hr = (self.apply(hit) - p).length();
                if hr <= 1e-6 {
                    return Some(hit);
                }
                if hr < (self.apply(x) - p).length() {
                    x = hit;
                }
            }
            let f = self.apply(x) - p;
            let residual = f.length();
            if residual <= 1e-6 {
                return Some(x);
            }
            let j = self.jacobian(x);
            let det = j.determinant();
            if !det.is_finite() || det.abs() < 1e-18 {
                return None;
            }
            let step = j.inverse() * f;
            if !step.is_finite() {
                return None;
            }
            let mut alpha = 1.0;
            let mut improved = false;
            while alpha >= 1.0 / 1024.0 {
                let next = x - step * alpha;
                if !next.is_finite() {
                    return None;
                }
                let Some(next) = self.contains(next).then_some(next).or_else(|| self.pull_inside(next)) else {
                    alpha *= 0.5;
                    continue;
                };
                if (self.apply(next) - p).length() < residual {
                    x = next;
                    improved = true;
                    break;
                }
                alpha *= 0.5;
            }
            if !improved {
                break;
            }
        }
        let f = self.apply(x) - p;
        (f.length() <= 1e-6 && self.contains(x)).then_some(x)
    }

    /// Every element's deformed corners have a positive Jacobian-determinant lower bound.
    pub fn certified(&self) -> bool {
        for k in 0..self.dims[2] {
            for j in 0..self.dims[1] {
                for i in 0..self.dims[0] {
                    let u = self.corners([i, j, k]);
                    let base = self.origin + DVec3::new(i as f64, j as f64, k as f64) * self.cell;
                    let deformed = std::array::from_fn(|c| {
                        let d = DVec3::new((c & 1) as f64, ((c >> 1) & 1) as f64, ((c >> 2) & 1) as f64);
                        base + d * self.cell + u[c]
                    });
                    if det_lower_bound(&deformed) <= 0.0 {
                        return false;
                    }
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mechanics::genesis;

    fn warp_at(half: f64) -> Warp {
        let solved = genesis::tabulated(2.6, half).expect("genesis table");
        Warp::from_solved(&solved, DVec3::ZERO)
    }

    fn lcg(state: &mut u64) -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*state >> 33) as f64) / (1u64 << 31) as f64
    }

    #[test]
    fn round_trip_identity_outside_certified_and_jacobian() {
        let warp = warp_at(8_000.0);
        assert!(warp.certified(), "a sagging cube stays injective");
        let hi = warp.origin + DVec3::new(warp.dims[0] as f64, warp.dims[1] as f64, warp.dims[2] as f64) * warp.cell;
        let outside = [warp.origin - DVec3::splat(10.0), hi, hi + DVec3::new(1.0, -3.0, 4.0)];
        for p in outside {
            assert!((warp.apply(p) - p).length() < 1e-9, "identity outside at {p}");
            assert!(warp.jacobian(p).abs_diff_eq(DMat3::IDENTITY, 1e-12));
        }
        let mut state = 0xC0FFEE_u64;
        let span = hi - warp.origin;
        for _ in 0..24 {
            let x = warp.origin + DVec3::new(lcg(&mut state), lcg(&mut state), lcg(&mut state)) * span * 0.999;
            let image = warp.apply(x);
            let back = warp.invert(image).unwrap_or_else(|| panic!("invert missed {x} -> {image}"));
            assert!((back - x).length() < 1e-6, "{x} -> {image} -> {back}");
            let j = warp.jacobian(x);
            let h = 1.0;
            for axis in 0..3 {
                let mut d = DVec3::ZERO;
                d[axis] = h;
                let fd = (warp.apply(x + d) - warp.apply(x - d)) / (2.0 * h);
                let col = j.col(axis);
                assert!((fd - col).length() < 1e-6, "axis {axis}: fd {fd} jacobian {col}");
            }
        }
    }

    #[test]
    fn a_twin_scale_warp_moves_more_than_half_a_block() {
        let warp = warp_at(6_000_000.0);
        assert!(warp.max_displacement() > STORAGE_MOVE);
        assert!(warp.certified());
        let x = DVec3::splat(1_000_000.0);
        let back = warp.invert(warp.apply(x)).expect("twin-scale invert");
        assert!((back - x).length() < 1e-4, "{}", (back - x).length());
    }
}
