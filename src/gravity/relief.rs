//! The gravity of a round body's relief (guide §6, §10.2): the relaxed surface's departure from its
//! datum sphere, as a thin layer of the body's bulk density on the sphere, expanded in real
//! spherical harmonics. Added to the ball of the datum radius it gives the field of the shape
//! physics produced, smooth at every altitude (no point masses near the ground).
//!
//! The single-layer potential of surface density `σ(Ω) = ρ h(Ω)` on radius `R`:
//! `V = −Σ K_lm (Rˡ / rˡ⁺¹) Y_lm` outside, `−Σ K_lm (rˡ / Rˡ⁺¹) Y_lm` inside, `K_lm = 4πGR²ρ h_lm / (2l+1)`.

use glam::DVec3;

use super::G;
use crate::space::atlas::FACES;
use crate::space::chart::{basis, Map};
use crate::space::datum::DatumField;

/// Degree of the expansion (a relaxed body's relief is dominated by degrees 4–8).
pub const L_MAX: usize = 24;
/// Quadrature cells per face edge when projecting a datum.
const QUADRATURE: usize = 64;

/// The relief layer of one body.
#[derive(Clone, Debug)]
pub struct Relief {
    pub centre: DVec3,
    pub radius: f64,
    /// `K_lm`, index `l² + l + m`.
    coeff: Vec<f64>,
}

impl Relief {
    /// The layer of a body of datum `radius` and bulk `density` whose surface sits `datum` above the
    /// sphere.
    pub fn new(centre: DVec3, radius: f64, density: f64, datum: &DatumField) -> Self {
        let mut h = vec![0.0f64; (L_MAX + 1) * (L_MAX + 1)];
        let mut y = vec![0.0f64; h.len()];
        let step = 2.0 / QUADRATURE as f64;
        let mut cells = Vec::with_capacity(6 * QUADRATURE * QUADRATURE);
        for f in 0..6 {
            let (tu, nn, tv) = basis(FACES[f]);
            for j in 0..QUADRATURE {
                for i in 0..QUADRATURE {
                    let (xi, eta) = (-1.0 + (i as f64 + 0.5) * step, -1.0 + (j as f64 + 0.5) * step);
                    let d = Map::Equiangular.dir(xi, eta);
                    let dir = (tu * d.x + nn * d.y + tv * d.z).normalize();
                    // Solid angle of an equiangular cell: dΩ = (1+x²)(1+y²)/(1+x²+y²)^{3/2} dα dβ.
                    let (x, z) = (d.x / d.y, d.z / d.y);
                    let w = (1.0 + x * x) * (1.0 + z * z) / (1.0 + x * x + z * z).powf(1.5);
                    cells.push((dir, w, datum.offset(f, xi, eta)));
                }
            }
        }
        let total: f64 = cells.iter().map(|c| c.1).sum();
        let norm = 4.0 * std::f64::consts::PI / total;
        for (dir, w, offset) in cells {
            harmonics(dir, &mut y);
            for (hk, yk) in h.iter_mut().zip(&y) {
                *hk += offset * yk * w * norm;
            }
        }
        let mut coeff = h;
        for l in 0..=L_MAX {
            let k = 4.0 * std::f64::consts::PI * G * radius * radius * density / (2 * l + 1) as f64;
            for m in 0..=2 * l {
                coeff[l * l + m] *= k;
            }
        }
        Self { centre, radius, coeff }
    }

    /// Potential at `p`, blocks²/s².
    pub fn potential(&self, p: DVec3) -> f64 {
        let d = p - self.centre;
        let r = d.length();
        if r < 1e-9 {
            return -self.coeff[0] * (1.0 / self.radius) * Y00;
        }
        let mut y = vec![0.0f64; self.coeff.len()];
        harmonics(d / r, &mut y);
        let (ratio, outside) = if r >= self.radius { (self.radius / r, true) } else { (r / self.radius, false) };
        let mut sum = 0.0;
        let mut radial = if outside { 1.0 / r } else { 1.0 / self.radius };
        for l in 0..=L_MAX {
            let mut s = 0.0;
            for m in 0..=2 * l {
                s += self.coeff[l * l + m] * y[l * l + m];
            }
            sum += s * radial;
            radial *= ratio;
        }
        -sum
    }

    /// Acceleration at `p` (central differences of the potential, a millionth of the radius apart).
    pub fn accel(&self, p: DVec3) -> DVec3 {
        let e = self.radius * 1e-6;
        let mut g = DVec3::ZERO;
        for a in 0..3 {
            let mut step = DVec3::ZERO;
            step[a] = e;
            g[a] = -(self.potential(p + step) - self.potential(p - step)) / (2.0 * e);
        }
        g
    }
}

const Y00: f64 = 0.282_094_791_773_878_14;

/// Orthonormal real spherical harmonics `Y_lm(dir)` for `l ≤ L_MAX`, index `l² + l + m`
/// (`m < 0` the sine terms). Polar axis +Z.
fn harmonics(dir: DVec3, out: &mut [f64]) {
    let ct = dir.z.clamp(-1.0, 1.0);
    let st = (1.0 - ct * ct).max(0.0).sqrt();
    let phi = dir.y.atan2(dir.x);
    // Fully normalised associated Legendre functions P̄_lm(cos θ), m ≥ 0.
    let n = L_MAX + 1;
    let mut p = vec![0.0f64; n * n];
    p[0] = Y00;
    for m in 1..n {
        p[m * n + m] = -((2 * m + 1) as f64 / (2 * m) as f64).sqrt() * st * p[(m - 1) * n + (m - 1)];
    }
    for m in 0..n - 1 {
        p[(m + 1) * n + m] = ((2 * m + 3) as f64).sqrt() * ct * p[m * n + m];
    }
    for m in 0..n {
        for l in m + 2..n {
            let (lf, mf) = (l as f64, m as f64);
            let a = ((4.0 * lf * lf - 1.0) / (lf * lf - mf * mf)).sqrt();
            let b = (((lf - 1.0) * (lf - 1.0) - mf * mf) / (4.0 * (lf - 1.0) * (lf - 1.0) - 1.0)).sqrt();
            p[l * n + m] = a * (ct * p[(l - 1) * n + m] - b * p[(l - 2) * n + m]);
        }
    }
    let sqrt2 = std::f64::consts::SQRT_2;
    for l in 0..n {
        out[l * l + l] = p[l * n];
        for m in 1..=l {
            let (s, c) = (m as f64 * phi).sin_cos();
            out[l * l + l + m] = sqrt2 * p[l * n + m] * c;
            out[l * l + l - m] = sqrt2 * p[l * n + m] * s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harmonics_are_orthonormal_over_the_sphere() {
        // Gauss–Legendre in cos θ would be exact; a fine midpoint grid is close enough.
        let (nt, np) = (200, 400);
        let count = (L_MAX + 1) * (L_MAX + 1);
        let mut gram = vec![0.0f64; 25 * 25];
        let mut y = vec![0.0f64; count];
        for it in 0..nt {
            let ct = -1.0 + (it as f64 + 0.5) * 2.0 / nt as f64;
            let st = (1.0 - ct * ct).sqrt();
            for ip in 0..np {
                let phi = (ip as f64 + 0.5) * std::f64::consts::TAU / np as f64;
                harmonics(DVec3::new(st * phi.cos(), st * phi.sin(), ct), &mut y);
                let w = 2.0 / nt as f64 * std::f64::consts::TAU / np as f64;
                for a in 0..25 {
                    for b in 0..25 {
                        gram[a * 25 + b] += y[a] * y[b] * w;
                    }
                }
            }
        }
        for a in 0..25 {
            for b in 0..25 {
                let want = if a == b { 1.0 } else { 0.0 };
                assert!((gram[a * 25 + b] - want).abs() < 2e-3, "⟨Y{a}, Y{b}⟩ = {}", gram[a * 25 + b]);
            }
        }
    }

    #[test]
    fn a_uniform_layer_pulls_like_its_mass_outside_and_not_at_all_inside() {
        let g = 9;
        let datum = DatumField { g, offsets: vec![100.0; 6 * g * g] };
        let (radius, density) = (1.0e6, 5.0);
        let relief = Relief::new(DVec3::ZERO, radius, density, &datum);
        let mass = 4.0 * std::f64::consts::PI * radius * radius * 100.0 * density;
        let p = DVec3::new(0.3, -2.0, 1.1).normalize() * 2.5 * radius;
        let want = -p.normalize() * G * mass / p.length_squared();
        let got = relief.accel(p);
        assert!((got - want).length() < 1e-4 * want.length(), "{got} vs {want}");
        let inside = relief.accel(DVec3::new(0.1, 0.2, -0.3) * radius);
        assert!(inside.length() < 1e-4 * want.length(), "{inside}");
    }

    #[test]
    fn a_cube_symmetric_relief_matches_its_point_masses_far_away() {
        // Highlands toward the cube corners, as a relaxed cube leaves them.
        let g = 33;
        let radius = 1.0e6;
        let datum = DatumField::sample(g, 0.0, |d| 2.0e4 * (d.x.abs() + d.y.abs() + d.z.abs() - 1.4));
        let relief = Relief::new(DVec3::ZERO, radius, 5.0, &datum);
        // Direct sum over a fine surface sampling at three radii out.
        let p = DVec3::new(1.0, 2.0, 2.5).normalize() * 3.0 * radius;
        let mut direct = DVec3::ZERO;
        let n = 120;
        let step = 2.0 / n as f64;
        let mut total_w = 0.0;
        let mut pts = Vec::new();
        for f in 0..6 {
            let (tu, nn, tv) = basis(FACES[f]);
            for j in 0..n {
                for i in 0..n {
                    let (xi, eta) = (-1.0 + (i as f64 + 0.5) * step, -1.0 + (j as f64 + 0.5) * step);
                    let d = Map::Equiangular.dir(xi, eta);
                    let dir = (tu * d.x + nn * d.y + tv * d.z).normalize();
                    let (x, z) = (d.x / d.y, d.z / d.y);
                    let w = (1.0 + x * x) * (1.0 + z * z) / (1.0 + x * x + z * z).powf(1.5);
                    total_w += w;
                    pts.push((dir, w, datum.offset(f, xi, eta)));
                }
            }
        }
        for (dir, w, h) in pts {
            let dm = 5.0 * h * radius * radius * w * 4.0 * std::f64::consts::PI / total_w;
            let q = dir * radius;
            let r = p - q;
            direct -= r * (G * dm / r.length().powi(3));
        }
        let got = relief.accel(p);
        assert!((got - direct).length() < 2e-3 * direct.length(), "{got} vs {direct}");
    }
}
