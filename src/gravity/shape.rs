//! Analytic mass sources: uniform boxes (the closed-form prism field), uniform balls and point
//! masses. Densities may be negative, so a layered or hollow body is a sum of these with density
//! increments. Each source evaluates its exact Newtonian field when it lies wholly inside the
//! law's inner range, and falls back to tapered point masses where it straddles the window.

use glam::{DMat3, DVec3};

use super::kernel::{self, R_G, R_IN};

/// One uniform source. `density` is amount per block³ (for `Point` it is the mass).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Primitive {
    pub shape: Shape,
    pub density: f64,
}

/// The geometry of a [`Primitive`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    /// Axis-aligned box `[lo, hi]`.
    Box { lo: DVec3, hi: DVec3 },
    /// Ball of radius `r` about `c`.
    Ball { c: DVec3, r: f64 },
    /// A point mass.
    Point { at: DVec3 },
}

/// Beyond this many bounding radii a source is evaluated by its multipoles (the box closed form
/// loses digits to cancellation far away; monopole + quadrupole is accurate to (r/d)⁴ there).
const FAR_RATIO: f64 = 100.0;
/// Pieces of a source that straddle the range window are split until their bounding radius is
/// below this fraction of their distance, then evaluated as tapered point masses.
const SPLIT_RATIO: f64 = 0.02;
/// Recursion cap for the window split (each level splits a box in eight).
const SPLIT_DEPTH: u32 = 7;

impl Primitive {
    pub const fn new(shape: Shape, density: f64) -> Self {
        Self { shape, density }
    }

    /// Bounding sphere `(centre, radius)`.
    pub fn bounds(&self) -> (DVec3, f64) {
        match self.shape {
            Shape::Box { lo, hi } => ((lo + hi) * 0.5, (hi - lo).length() * 0.5),
            Shape::Ball { c, r } => (c, r),
            Shape::Point { at } => (at, 0.0),
        }
    }

    /// Total (signed) mass.
    pub fn mass(&self) -> f64 {
        match self.shape {
            Shape::Box { lo, hi } => {
                let e = hi - lo;
                e.x * e.y * e.z * self.density
            }
            Shape::Ball { r, .. } => 4.0 / 3.0 * std::f64::consts::PI * r * r * r * self.density,
            Shape::Point { .. } => self.density,
        }
    }

    /// Centre of mass (uniform sources: the geometric centre).
    pub fn centre(&self) -> DVec3 {
        self.bounds().0
    }

    /// Acceleration and potential at `p`, per unit `G` (multiply both by `G`).
    pub fn field(&self, p: DVec3) -> (DVec3, f64) {
        let (c, rad) = self.bounds();
        let d = (p - c).length();
        if d - rad >= R_G {
            return (DVec3::ZERO, 0.0);
        }
        if d + rad <= R_IN {
            let m = self.mass();
            if rad == 0.0 {
                return kernel::point(c, m, p);
            }
            if d > FAR_RATIO * rad {
                let (a, phi) = kernel::newton(c, m, p);
                return match self.shape {
                    Shape::Box { lo, hi } => {
                        let (qa, qphi) = box_quadrupole(lo, hi, m, p);
                        (a + qa, phi + qphi)
                    }
                    _ => (a, phi),
                };
            }
            let (a, phi) = match self.shape {
                Shape::Box { lo, hi } => box_field(lo, hi, p),
                Shape::Ball { c, r } => ball_field(c, r, p),
                Shape::Point { .. } => unreachable!("points have zero radius"),
            };
            return (a * self.density, phi * self.density - m * kernel::offset());
        }
        self.split_field(p, 0)
    }

    /// The tidal tensor at `p`, per unit `G`: exact for points and balls, central differences of
    /// the closed form for boxes.
    pub fn tidal(&self, p: DVec3) -> DMat3 {
        match self.shape {
            Shape::Point { at } => kernel::point_tidal(at, self.density, p),
            _ => {
                let h = 1e-3 * (1.0 + (p - self.centre()).length() * 1e-6);
                let col = |e: DVec3| (self.field(p + e * h).0 - self.field(p - e * h).0) / (2.0 * h);
                DMat3::from_cols(col(DVec3::X), col(DVec3::Y), col(DVec3::Z))
            }
        }
    }

    /// Field of a source straddling the window: split into octants until each piece is either
    /// wholly inside `R_IN`, wholly beyond `R_G`, or small enough to be a tapered point mass.
    fn split_field(&self, p: DVec3, depth: u32) -> (DVec3, f64) {
        let (c, rad) = self.bounds();
        let d = (p - c).length();
        if d - rad >= R_G {
            return (DVec3::ZERO, 0.0);
        }
        if d + rad <= R_IN && depth > 0 {
            return self.field(p);
        }
        if depth >= SPLIT_DEPTH || rad <= SPLIT_RATIO * d {
            return kernel::point(c, self.mass(), p);
        }
        let (lo, hi) = match self.shape {
            Shape::Box { lo, hi } => (lo, hi),
            // A ball's mass is evaluated as its centre point mass outside the inner range: exact
            // for the Newtonian part, the window's variation across it is the declared error.
            Shape::Ball { .. } | Shape::Point { .. } => return kernel::point(c, self.mass(), p),
        };
        let mid = (lo + hi) * 0.5;
        let mut sum = (DVec3::ZERO, 0.0);
        for k in 0..8 {
            let pick = |bit: usize, a: f64, m: f64, b: f64| if k >> bit & 1 == 0 { (a, m) } else { (m, b) };
            let (x0, x1) = pick(0, lo.x, mid.x, hi.x);
            let (y0, y1) = pick(1, lo.y, mid.y, hi.y);
            let (z0, z1) = pick(2, lo.z, mid.z, hi.z);
            let piece = Primitive::new(
                Shape::Box { lo: DVec3::new(x0, y0, z0), hi: DVec3::new(x1, y1, z1) },
                self.density,
            );
            let (a, phi) = piece.split_field(p, depth + 1);
            sum.0 += a;
            sum.1 += phi;
        }
        sum
    }
}

/// `ln(a + r)` with `r = √(a² + q)`, robust when `a < 0` and `a + r → 0` (then `a + r = q/(r − a)`).
#[inline]
fn ln_sum(a: f64, r: f64, q: f64) -> f64 {
    if a >= 0.0 { (a + r).ln() } else { q.ln() - (r - a).ln() }
}

/// Newtonian field of a unit-density box `[lo, hi]` at `p` (Nagy's closed form for the prism):
/// `(accel toward the mass, potential)` per unit `G·ρ`. Faces, edges and corners are handled
/// (every singular log is multiplied by a coordinate that vanishes with it).
pub fn box_field(lo: DVec3, hi: DVec3, p: DVec3) -> (DVec3, f64) {
    let (mut g, mut phi) = (DVec3::ZERO, 0.0);
    let (xs, ys, zs) = ([lo.x - p.x, hi.x - p.x], [lo.y - p.y, hi.y - p.y], [lo.z - p.z, hi.z - p.z]);
    for (i, &x) in xs.iter().enumerate() {
        for (j, &y) in ys.iter().enumerate() {
            for (k, &z) in zs.iter().enumerate() {
                let sign = if (i + j + k) % 2 == 1 { 1.0 } else { -1.0 };
                let (x2, y2, z2) = (x * x, y * y, z * z);
                let r = (x2 + y2 + z2).sqrt();
                if r == 0.0 {
                    continue;
                }
                // ln(c + r) appears multiplied by a coordinate that is zero whenever the log is
                // singular (c + r = 0 needs the other two coordinates to vanish).
                let lx = if y2 + z2 > 0.0 { ln_sum(x, r, y2 + z2) } else { 0.0 };
                let ly = if x2 + z2 > 0.0 { ln_sum(y, r, x2 + z2) } else { 0.0 };
                let lz = if x2 + y2 > 0.0 { ln_sum(z, r, x2 + y2) } else { 0.0 };
                let at = |a: f64, b: f64, c: f64| if c != 0.0 { (a * b / (c * r)).atan() } else { 0.0 };
                let (ax, ay, az) = (at(y, z, x), at(z, x, y), at(x, y, z));
                g.x += sign * (y * lz + z * ly - x * ax);
                g.y += sign * (z * lx + x * lz - y * ay);
                g.z += sign * (x * ly + y * lx - z * az);
                phi += sign * (x * y * lz + y * z * lx + z * x * ly - 0.5 * (x2 * ax + y2 * ay + z2 * az));
            }
        }
    }
    (-g, -phi)
}

/// Quadrupole correction of a uniform box of mass `m` at `p` (zero for a cube), per unit `G`:
/// `Φ_q = −x·Q·x / (2r⁵)` with the box's diagonal `Q_ii = 3 S_i − ΣS`, `S_i = m e_i²/3`.
fn box_quadrupole(lo: DVec3, hi: DVec3, m: f64, p: DVec3) -> (DVec3, f64) {
    let e = (hi - lo) * 0.5;
    let s = e * e * (m / 3.0);
    let q = s * 3.0 - DVec3::splat(s.x + s.y + s.z);
    let x = p - (lo + hi) * 0.5;
    let r2 = x.length_squared();
    let r = r2.sqrt();
    let inv5 = 1.0 / (r2 * r2 * r);
    let qx = q * x;
    let xqx = x.dot(qx);
    (qx * inv5 - x * (2.5 * xqx * inv5 / r2), -0.5 * xqx * inv5)
}

/// Newtonian field of a unit-density ball at `p`: `(accel, potential)` per unit `G·ρ`.
pub fn ball_field(c: DVec3, r: f64, p: DVec3) -> (DVec3, f64) {
    let d = p - c;
    let dist = d.length();
    let m = 4.0 / 3.0 * std::f64::consts::PI * r * r * r;
    if dist >= r {
        let inv = 1.0 / dist;
        (-d * (m * inv * inv * inv), -m * inv)
    } else {
        let inv3 = 1.0 / (r * r * r);
        (-d * (m * inv3), -m * (3.0 * r * r - dist * dist) * 0.5 * inv3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Midpoint-rule field of a unit-density box (pure Newtonian, no softening). Cells within
    /// `CORE` of `p` are skipped: a uniform ball pulls nothing at its own centre, so interior points
    /// stay accurate as long as the core lies inside the box (its potential is added analytically).
    fn brute_box(lo: DVec3, hi: DVec3, p: DVec3, n: usize) -> (DVec3, f64) {
        const CORE: f64 = 0.25;
        let inside = (0..3).all(|a| p[a] - CORE > lo[a] && p[a] + CORE < hi[a]);
        let step = (hi - lo) / n as f64;
        let cell = step.x * step.y * step.z;
        let (mut g, mut phi) = (DVec3::ZERO, 0.0);
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let q = lo + step * DVec3::new(i as f64 + 0.5, j as f64 + 0.5, k as f64 + 0.5);
                    let d = q - p;
                    let r = d.length();
                    if inside && r < CORE {
                        continue;
                    }
                    g += d / (r * r * r);
                    phi -= 1.0 / r;
                }
            }
        }
        // The skipped core's potential at its centre: −2π CORE² (its pull there is zero).
        let core = if inside { -2.0 * std::f64::consts::PI * CORE * CORE } else { 0.0 };
        (g * cell, phi * cell + core)
    }

    #[test]
    fn box_closed_form_matches_known_values_and_brute_force() {
        let (lo, hi) = (DVec3::splat(-1.0), DVec3::splat(1.0));
        // Reference values computed independently (unit cube of half-size 1, G = ρ = 1).
        let (g, phi) = box_field(lo, hi, DVec3::new(0.0, 2.0, 0.0));
        assert!((g.y + 1.885_995_524_385_673_6).abs() < 1e-12 && g.x.abs() < 1e-12, "{g}");
        assert!((phi + 3.950_369_616_696_249).abs() < 1e-12, "{phi}");
        let (g, _) = box_field(lo, hi, DVec3::new(0.0, 1.0, 0.0));
        assert!((g.y + 5.193_793_156_516_73).abs() < 1e-10, "face centre {g}");
        for p in [DVec3::new(0.5, 1.5, 0.3), DVec3::new(2.0, 2.0, 1.0), DVec3::new(0.2, -0.4, 0.6), DVec3::new(-3.0, 0.1, 0.0)] {
            let (g, phi) = box_field(lo, hi, p);
            let (bg, bphi) = brute_box(lo, hi, p, 64);
            assert!((g - bg).length() <= 5e-3 * bg.length().max(0.5), "{p}: {g} vs {bg}");
            assert!((phi - bphi).abs() <= 5e-3 * bphi.abs(), "{p}: {phi} vs {bphi}");
        }
    }

    #[test]
    fn box_field_is_finite_on_faces_edges_and_corners() {
        let (lo, hi) = (DVec3::ZERO, DVec3::new(2.0, 3.0, 5.0));
        for p in [lo, hi, DVec3::new(1.0, 0.0, 0.0), DVec3::new(2.0, 3.0, 2.5), DVec3::new(1.0, 1.5, 0.0), DVec3::new(-1.0, 0.0, 0.0), DVec3::new(0.0, -2.0, 0.0)] {
            let (g, phi) = box_field(lo, hi, p);
            assert!(g.is_finite() && phi.is_finite(), "{p}: {g} {phi}");
        }
        // Approaching a face from outside and inside meets the same value (the field is continuous).
        let a = box_field(lo, hi, DVec3::new(1.0, 1.5, -1e-9)).0;
        let b = box_field(lo, hi, DVec3::new(1.0, 1.5, 1e-9)).0;
        assert!((a - b).length() < 1e-6, "{a} vs {b}");
    }

    #[test]
    fn box_far_field_tends_to_its_monopole() {
        let (lo, hi) = (DVec3::new(10.0, 20.0, 30.0), DVec3::new(14.0, 23.0, 31.0));
        let m = 4.0 * 3.0;
        let c = (lo + hi) * 0.5;
        let p = c + DVec3::new(300.0, -120.0, 75.0);
        let (g, _) = box_field(lo, hi, p);
        let d = c - p;
        let mono = d * (m / d.length().powi(3));
        let ratio = 3.0 / d.length();
        assert!((g - mono).length() <= ratio * ratio * mono.length(), "{g} vs {mono}");
    }

    #[test]
    fn ball_is_linear_inside_zero_at_centre_and_a_point_outside() {
        let c = DVec3::new(5.0, -3.0, 2.0);
        assert_eq!(ball_field(c, 10.0, c).0, DVec3::ZERO);
        let half = ball_field(c, 10.0, c + DVec3::new(5.0, 0.0, 0.0)).0.length();
        let surface = ball_field(c, 10.0, c + DVec3::new(10.0, 0.0, 0.0)).0.length();
        assert!((half * 2.0 - surface).abs() < 1e-9 * surface, "interior field is linear in r");
        let m = 4.0 / 3.0 * std::f64::consts::PI * 1000.0;
        let out = ball_field(c, 10.0, c + DVec3::new(0.0, 20.0, 0.0)).0.length();
        assert!((out - m / 400.0).abs() < 1e-9 * out);
    }

    #[test]
    fn hollow_shell_cavity_feels_nothing() {
        let c = DVec3::new(1.0e6, 2.0, -3.0);
        let outer = Primitive::new(Shape::Ball { c, r: 5000.0 }, 5.0);
        let inner = Primitive::new(Shape::Ball { c, r: 4000.0 }, -5.0);
        for off in [DVec3::ZERO, DVec3::new(100.0, -2000.0, 3000.0), DVec3::new(3999.0, 0.0, 0.0)] {
            let a = outer.field(c + off).0 + inner.field(c + off).0;
            assert!(a.length() < 1e-9 * outer.field(c + DVec3::X * 5000.0).0.length(), "{off}: {a}");
        }
    }

    #[test]
    fn far_multipoles_agree_with_the_closed_form() {
        // Past the switch the multipole expansion stands in for the closed form: both at one point.
        for (lo, hi) in [(DVec3::ZERO, DVec3::splat(2.0)), (DVec3::new(-3.0, 0.0, 1.0), DVec3::new(5.0, 1.0, 2.0))] {
            let prim = Primitive::new(Shape::Box { lo, hi }, 3.0);
            let (c, r) = prim.bounds();
            for dir in [DVec3::X, DVec3::new(0.3, -0.8, 0.52).normalize()] {
                let p = c + dir * (FAR_RATIO * r * 1.2);
                let far = prim.field(p).0;
                let exact = box_field(lo, hi, p).0 * 3.0;
                assert!((far - exact).length() <= 1e-7 * exact.length(), "{far} vs {exact}");
            }
        }
    }

    #[test]
    fn a_source_straddling_the_window_splits_without_blowing_up() {
        let h = 2.5e7;
        let prim = Primitive::new(Shape::Box { lo: DVec3::splat(-h), hi: DVec3::splat(h) }, 5.0);
        let near = prim.field(DVec3::new(0.0, h + 10.0, 0.0)).0.length();
        let mid = prim.field(DVec3::new(0.0, R_IN + h * 0.5, 0.0)).0.length();
        let far = prim.field(DVec3::new(0.0, R_G + 2.0 * h, 0.0)).0.length();
        assert!(near > mid && mid > 0.0 && far == 0.0, "{near} {mid} {far}");
    }
}
