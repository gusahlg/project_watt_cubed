//! The force law: a softened inverse square with a smooth finite range (guide §14.3), identical for
//! every pair of masses. Inside `R_IN` it is exactly Newtonian (softened by `EPS` for point samples),
//! beyond `R_G` it is zero, and in between the force is tapered by a smoothstep window. The potential
//! is the integral of the force, so the two always agree.

use std::sync::LazyLock;

use glam::DVec3;

/// The universal gravitational constant in block units: blocks³ per amount unit per second².
/// Chosen once so the designed start planet pulls `24 m/s²` at its +Y face centre.
pub const G: f64 = 4.349e-8;
/// Range below which the law is exactly Newtonian.
pub const R_IN: f64 = 1.25e8;
/// Range at and beyond which no force acts.
pub const R_G: f64 = 2.5e8;
/// Softening length of a point sample (half a block: a sample stands for a cell).
pub const EPS: f64 = 0.5;

/// The range window: `1` up to `R_IN`, smoothstep down to `0` at `R_G`.
#[inline]
pub fn window(r: f64) -> f64 {
    if r <= R_IN {
        1.0
    } else if r >= R_G {
        0.0
    } else {
        let t = (r - R_IN) / (R_G - R_IN);
        1.0 - t * t * (3.0 - 2.0 * t)
    }
}

/// `∫_r^{R_G} w(s)/s² ds` for `R_IN <= r <= R_G` (the softening is below f64 resolution there).
/// The window expands to a cubic in `s`, so the integral has a closed form.
fn tail(r: f64) -> f64 {
    let (a, l) = (R_IN, R_G - R_IN);
    let (l2, l3) = (l * l, l * l * l);
    let c0 = 1.0 - 3.0 * a * a / l2 - 2.0 * a * a * a / l3;
    let c1 = 6.0 * a / l2 + 6.0 * a * a / l3;
    let c2 = -3.0 / l2 - 6.0 * a / l3;
    let c3 = 2.0 / l3;
    let f = |s: f64| -c0 / s + c1 * (s / R_G).ln() + c2 * s + c3 * s * s / 2.0;
    f(R_G) - f(r)
}

/// The constant that joins the Newtonian potential inside `R_IN` to the tapered tail:
/// `Φ_point(r) = −G m (1/√(r²+ε²) + OFFSET)` for `r <= R_IN`.
static OFFSET: LazyLock<f64> = LazyLock::new(|| tail(R_IN) - 1.0 / (R_IN * R_IN + EPS * EPS).sqrt());

/// The potential offset per unit `G·mass` that every source inside `R_IN` adds to its Newtonian
/// closed form (extended sources included), so potentials of all sources share one reference.
#[inline]
pub fn offset() -> f64 {
    *OFFSET
}

/// Acceleration and potential at `p` from a point mass `m` at `at`, per unit `G`.
/// Returns `(accel, potential)`; the acceleration points toward the mass.
#[inline]
pub fn point(at: DVec3, m: f64, p: DVec3) -> (DVec3, f64) {
    let d = at - p;
    let r2 = d.length_squared();
    let r = r2.sqrt();
    if r >= R_G {
        return (DVec3::ZERO, 0.0);
    }
    let soft = r2 + EPS * EPS;
    let inv = 1.0 / soft.sqrt();
    if r <= R_IN {
        return (d * (m * inv * inv * inv), -m * (inv + offset()));
    }
    (d * (m * inv * inv * inv * window(r)), -m * tail(r))
}

/// Unsoftened Newtonian monopole of an extended source seen from inside `R_IN` (softening is for
/// point samples standing for cells, not for a body's far field), per unit `G`.
#[inline]
pub fn newton(at: DVec3, m: f64, p: DVec3) -> (DVec3, f64) {
    let d = at - p;
    let r = d.length();
    let inv = 1.0 / r;
    (d * (m * inv * inv * inv), -m * (inv + offset()))
}

/// The tidal tensor (∂accel/∂p) of a point mass, per unit `G`, inside `R_IN` (zero beyond `R_G`;
/// the window region uses the Newtonian tensor scaled by the window, a declared approximation).
pub fn point_tidal(at: DVec3, m: f64, p: DVec3) -> glam::DMat3 {
    let d = p - at;
    let r2 = d.length_squared() + EPS * EPS;
    let r = r2.sqrt();
    if d.length() >= R_G {
        return glam::DMat3::ZERO;
    }
    let w = window(d.length());
    let k = m * w / (r2 * r);
    let outer = glam::DMat3::from_cols(d * d.x, d * d.y, d * d.z);
    (outer * (3.0 / r2) - glam::DMat3::IDENTITY) * k
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Force magnitude of a unit point mass at distance `r` (per unit G).
    fn force(r: f64) -> f64 {
        point(DVec3::ZERO, 1.0, DVec3::new(r, 0.0, 0.0)).0.length()
    }

    fn potential(r: f64) -> f64 {
        point(DVec3::ZERO, 1.0, DVec3::new(r, 0.0, 0.0)).1
    }

    #[test]
    fn newtonian_inside_the_inner_range() {
        for r in [1.0, 10.0, 1.0e4, 2.5e7, 1.0e8] {
            let soft = (r * r + EPS * EPS).sqrt();
            let want = r / (soft * soft * soft);
            assert!((force(r) - want).abs() <= want * 1e-14, "r={r}");
        }
    }

    #[test]
    fn force_and_potential_vanish_beyond_the_range() {
        assert_eq!(force(R_G), 0.0);
        assert_eq!(potential(R_G * 1.5), 0.0);
        assert!(force(R_G * 0.999) > 0.0);
    }

    #[test]
    fn force_is_continuous_across_both_window_edges() {
        for edge in [R_IN, R_G] {
            let (a, b) = (force(edge * (1.0 - 1e-9)), force(edge * (1.0 + 1e-9)));
            assert!((a - b).abs() <= 1e-6 * a.max(1e-30), "edge {edge}: {a} vs {b}");
        }
        let (a, b) = (potential(R_IN * (1.0 - 1e-12)), potential(R_IN * (1.0 + 1e-12)));
        assert!((a - b).abs() <= 1e-9 * a.abs(), "potential jumps at R_IN: {a} vs {b}");
    }

    #[test]
    fn force_is_minus_the_potential_gradient_everywhere() {
        for r in [3.0, 1.0e3, 1.0e6, 9.0e7, 1.3e8, 1.8e8, 2.3e8, 2.49e8] {
            let h = r * 1e-5;
            let grad = (potential(r + h) - potential(r - h)) / (2.0 * h);
            let f = force(r);
            assert!((grad - f).abs() <= 1e-5 * f + 1e-30, "r={r}: dΦ/dr={grad} force={f}");
        }
    }

    #[test]
    fn tidal_tensor_matches_the_numerical_derivative() {
        let at = DVec3::new(3.0, -2.0, 7.0);
        let p = DVec3::new(40.0, 15.0, -9.0);
        let t = point_tidal(at, 5.0, p);
        let h = 1e-3;
        for (axis, col) in [DVec3::X, DVec3::Y, DVec3::Z].into_iter().zip([t.x_axis, t.y_axis, t.z_axis]) {
            let num = (point(at, 5.0, p + axis * h).0 - point(at, 5.0, p - axis * h).0) / (2.0 * h);
            assert!((num - col).length() <= 1e-9 * col.length().max(1e-12), "{num} vs {col}");
        }
    }
}
