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
/// Coefficients of an expansion to [`L_MAX`].
const COUNT: usize = (L_MAX + 1) * (L_MAX + 1);
/// Quadrature cells per face edge when projecting a datum (a datum has 33 samples per edge; 128
/// cells around a great circle resolve degree 24 comfortably).
const QUADRATURE: usize = 32;

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
        let mut h = vec![0.0f64; COUNT];
        let mut y = [0.0f64; COUNT];
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
        let mut y = [0.0f64; COUNT];
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

    /// Acceleration at `p` (see [`field`](Self::field)).
    pub fn accel(&self, p: DVec3) -> DVec3 {
        self.field(p).0
    }

    /// Acceleration and potential at `p` in one pass: the analytic gradient of the expansion
    /// (spherical components), or central differences of the potential within a thousandth of a
    /// radian of the polar axis.
    pub fn field(&self, p: DVec3) -> (DVec3, f64) {
        let d = p - self.centre;
        let r = d.length();
        let rho = (d.x * d.x + d.y * d.y).sqrt();
        if r < 1e-9 || rho < 1e-3 * r {
            return (self.accel_numeric(p), self.potential(p));
        }
        let (ct, st) = (d.z / r, rho / r);
        let (cp, sp) = (d.x / rho, d.y / rho);
        let mut y = [0.0f64; COUNT];
        let mut dy = [0.0f64; COUNT];
        harmonics_with_theta(d / r, &mut y, &mut dy);
        let outside = r >= self.radius;
        let ratio = if outside { self.radius / r } else { r / self.radius };
        let mut g = if outside { 1.0 / r } else { 1.0 / self.radius };
        let (mut v, mut dv_dr, mut dv_dt, mut dv_dp) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for l in 0..=L_MAX {
            let (mut sl, mut tl, mut ul) = (0.0, 0.0, 0.0);
            for m in -(l as i64)..=(l as i64) {
                let k = l * l + (l as i64 + m) as usize;
                let c = self.coeff[k];
                sl += c * y[k];
                tl += c * dy[k];
                // ∂/∂φ: cos(mφ) → −m sin(mφ) (the m < 0 partner), sin(|m|φ) → |m| cos(|m|φ).
                let partner = l * l + (l as i64 - m) as usize;
                ul += c * (-(m as f64)) * y[partner];
            }
            let dg = if outside { -((l + 1) as f64) * g / r } else { l as f64 * g / r };
            v -= g * sl;
            dv_dr -= dg * sl;
            dv_dt -= g * tl;
            dv_dp -= g * ul;
            g *= ratio;
        }
        let r_hat = DVec3::new(st * cp, st * sp, ct);
        let t_hat = DVec3::new(ct * cp, ct * sp, -st);
        let p_hat = DVec3::new(-sp, cp, 0.0);
        (-(r_hat * dv_dr + t_hat * (dv_dt / r) + p_hat * (dv_dp / (r * st))), v)
    }

    /// [`accel`](Self::accel) by central differences of the potential, a millionth of the radius
    /// apart.
    fn accel_numeric(&self, p: DVec3) -> DVec3 {
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
    // Fully normalised associated Legendre functions P̄_lm(cos θ), m ≥ 0 (on the stack: gravity
    // samples run every physics step and allocate nothing).
    let n = L_MAX + 1;
    let mut p = [0.0f64; COUNT];
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

/// [`harmonics`] and their derivatives with respect to the colatitude θ (off the polar axis:
/// `dP̄_lm/dθ = (l cos θ P̄_lm − √((2l+1)(l²−m²)/(2l−1)) P̄_(l−1)m) / sin θ`).
fn harmonics_with_theta(dir: DVec3, out: &mut [f64], d_theta: &mut [f64]) {
    let ct = dir.z.clamp(-1.0, 1.0);
    let st = (1.0 - ct * ct).max(1e-300).sqrt();
    let phi = dir.y.atan2(dir.x);
    let n = L_MAX + 1;
    let mut p = [0.0f64; COUNT];
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
    let mut dp = [0.0f64; COUNT];
    for l in 0..n {
        for m in 0..=l {
            let (lf, mf) = (l as f64, m as f64);
            let below = if l > m { ((2.0 * lf + 1.0) * (lf * lf - mf * mf) / (2.0 * lf - 1.0)).sqrt() * p[(l - 1) * n + m] } else { 0.0 };
            dp[l * n + m] = (lf * ct * p[l * n + m] - below) / st;
        }
    }
    let sqrt2 = std::f64::consts::SQRT_2;
    for l in 0..n {
        out[l * l + l] = p[l * n];
        d_theta[l * l + l] = dp[l * n];
        for m in 1..=l {
            let (s, c) = (m as f64 * phi).sin_cos();
            out[l * l + l + m] = sqrt2 * p[l * n + m] * c;
            out[l * l + l - m] = sqrt2 * p[l * n + m] * s;
            d_theta[l * l + l + m] = sqrt2 * dp[l * n + m] * c;
            d_theta[l * l + l - m] = sqrt2 * dp[l * n + m] * s;
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
        let mut gram = vec![0.0f64; 25 * 25];
        let mut y = [0.0f64; COUNT];
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
    fn theta_derivatives_match_finite_differences() {
        let (theta, phi) = (1.1f64, 0.7f64);
        let dir = |t: f64| DVec3::new(t.sin() * phi.cos(), t.sin() * phi.sin(), t.cos());
        let mut y = [0.0f64; COUNT];
        let mut dy = [0.0f64; COUNT];
        harmonics_with_theta(dir(theta), &mut y, &mut dy);
        let (mut a, mut b) = ([0.0f64; COUNT], [0.0f64; COUNT]);
        let h = 1e-6;
        harmonics(dir(theta + h), &mut a);
        harmonics(dir(theta - h), &mut b);
        for k in 0..COUNT {
            let fd = (a[k] - b[k]) / (2.0 * h);
            assert!((fd - dy[k]).abs() < 1e-5 * (1.0 + fd.abs()), "k {k}: {} vs {fd}", dy[k]);
        }
    }

    #[test]
    fn the_analytic_gradient_matches_central_differences() {
        let g = 33;
        let radius = 1.0e6;
        let datum = DatumField::sample(g, 0.0, |d| 3.0e4 * (d.x.abs() + d.y.abs() + d.z.abs() - 1.4) + 5.0e3 * d.x * d.y);
        let relief = Relief::new(DVec3::new(1.0e5, -2.0e5, 3.0e5), radius, 5.0, &datum);
        // Not on the layer itself: its normal pull jumps there, which a central difference straddles.
        for (i, f) in [0.4f64, 0.97, 0.999, 1.001, 1.03, 2.5].iter().enumerate() {
            let dir = DVec3::new(0.3 + i as f64 * 0.17, -0.8 + i as f64 * 0.3, 0.5 - i as f64 * 0.21).normalize();
            let p = relief.centre + dir * radius * f;
            let (a, n) = (relief.accel(p), relief.accel_numeric(p));
            assert!((a - n).length() < 1e-5 * n.length().max(1e-12), "at {f}: {a} vs {n}");
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
