//! The gravitational field of a world: the generator's matter (through its oracle, analytic) plus
//! the exact edit corrections. A query sums every source within the law's range; summarised groups
//! are accepted as single masses when they are small enough as seen from the query (Barnes–Hut).

use std::sync::Arc;

use glam::{DMat3, DVec3};

use super::kernel::{self, G, R_G};
use super::ledger::Ledger;
use super::oracle::{MassOracle, Summary, Visitor};
use super::shape::Primitive;
use super::Sample;

/// Opening angle: a group is accepted as one mass when its radius over its distance is below this.
pub const THETA: f64 = 0.35;

/// A world's field.
pub struct Field {
    oracle: Arc<dyn MassOracle>,
    ledger: Ledger,
    epoch: u64,
}

/// Accumulates one query.
struct Query {
    p: DVec3,
    theta: f64,
    tidal: bool,
    accel: DVec3,
    potential: f64,
    tensor: DMat3,
    error: f64,
}

impl Visitor for Query {
    fn group(&mut self, g: &Summary) -> bool {
        let d = (g.com - self.p).length();
        let to_centre = (g.centre - self.p).length();
        if to_centre <= g.radius || g.radius > self.theta * d {
            return true;
        }
        let (a, phi) = if d + g.radius <= kernel::R_IN { kernel::newton(g.com, g.mass, self.p) } else { kernel::point(g.com, g.mass, self.p) };
        self.accel += a;
        self.potential += phi;
        if self.tidal {
            self.tensor += kernel::point_tidal(g.com, g.mass, self.p);
        }
        // Quadrupole remainder of a positive mass seen from outside its bounding sphere.
        let q = 1.0 - g.radius / to_centre;
        self.error += 3.0 * g.mass.abs() * g.radius * g.radius / (d * d * d * d * q * q * q * q);
        false
    }

    fn primitive(&mut self, prim: &Primitive) {
        let (a, phi) = prim.field(self.p);
        self.accel += a;
        self.potential += phi;
        if self.tidal {
            self.tensor += prim.tidal(self.p);
        }
    }

    fn error(&mut self, e: f64) {
        self.error += e;
    }
}

impl Field {
    pub fn new(oracle: Arc<dyn MassOracle>) -> Self {
        Self { oracle, ledger: Ledger::default(), epoch: 0 }
    }

    /// Record a committed change of `delta` amount at a world cell (`amount(new) − amount(old)`).
    pub fn record(&mut self, cell: (i32, i32, i32), delta: i32) {
        if delta != 0 {
            self.ledger.record(cell, delta);
            self.epoch += 1;
        }
    }

    /// The source epoch: bumps with every committed amount change.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// The field at `p` (no tidal tensor).
    pub fn sample(&self, p: DVec3) -> Sample {
        self.query(p, THETA, false)
    }

    /// The field at `p` with the tidal tensor.
    pub fn sample_tidal(&self, p: DVec3) -> Sample {
        self.query(p, THETA, true)
    }

    /// The field at `p` with an explicit opening angle.
    pub fn sample_with(&self, p: DVec3, theta: f64, tidal: bool) -> Sample {
        self.query(p, theta, tidal)
    }

    fn query(&self, p: DVec3, theta: f64, tidal: bool) -> Sample {
        let mut q = Query { p, theta, tidal, accel: DVec3::ZERO, potential: 0.0, tensor: DMat3::ZERO, error: 0.0 };
        self.oracle.visit(p, R_G, &mut q);
        if !self.ledger.is_empty() {
            let (a, phi, err) = self.ledger.field(p);
            q.accel += a;
            q.potential += phi;
            q.error += err;
        }
        Sample {
            accel: q.accel * G,
            potential: q.potential * G,
            tidal: tidal.then(|| q.tensor * G),
            error: q.error * G,
            epoch: self.epoch,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::oracle::Primitives;
    use super::super::shape::Shape;
    use super::*;

    const H: f64 = 2.5e7;

    fn cube() -> Field {
        let prim = Primitive::new(Shape::Box { lo: DVec3::new(-H, -2.0 * H, -H), hi: DVec3::new(H, 0.0, H) }, 5.0);
        Field::new(Arc::new(Primitives(vec![prim])))
    }

    #[test]
    fn cube_face_centre_pulls_straight_down_with_the_designed_strength() {
        let s = cube().sample(DVec3::new(0.0, 0.0, 0.0));
        assert!(s.accel.x.abs() < 1e-9 && s.accel.z.abs() < 1e-9, "{}", s.accel);
        // 5.19379 G ρ H is the face-centre field of a uniform cube; 24 m/s² is the designed spawn pull.
        let want = 24.0 * crate::math::PER_METER;
        assert!((-s.accel.y - want).abs() < 0.01 * want, "spawn gravity {} vs {want}", -s.accel.y);
    }

    #[test]
    fn cube_faces_tilt_toward_the_centre_as_derived() {
        // Halfway to an edge the field leans 16.27° off the face normal (independent derivation).
        let s = cube().sample(DVec3::new(0.5 * H, 0.0, 0.0));
        let tilt = (s.accel.x.abs() / s.accel.y.abs()).atan().to_degrees();
        assert!((tilt - 16.271).abs() < 0.01, "tilt {tilt}");
        assert!(s.accel.x < 0.0, "leans toward the face centre");
        // On an edge midpoint it is exactly 45°.
        let e = cube().sample(DVec3::new(H, 0.0, 0.0));
        assert!((e.accel.x.abs() - e.accel.y.abs()).abs() < 1e-9 * e.accel.length(), "{}", e.accel);
    }

    #[test]
    fn the_core_of_the_cube_is_weightless() {
        let s = cube().sample(DVec3::new(0.0, -H, 0.0));
        assert!(s.accel.length() < 1e-9, "{}", s.accel);
        assert!(s.up(0.1).is_none(), "no up direction in zero g");
    }

    #[test]
    fn equal_masses_cancel_at_the_midpoint_without_nan() {
        let ball = |x: f64| Primitive::new(Shape::Ball { c: DVec3::new(x, 0.0, 0.0), r: 1000.0 }, 5.0);
        let f = Field::new(Arc::new(Primitives(vec![ball(-5000.0), ball(5000.0)])));
        let s = f.sample_tidal(DVec3::ZERO);
        assert!(s.accel.length() < 1e-12 && s.accel.is_finite(), "{}", s.accel);
        let t = s.tidal.unwrap();
        assert!(t.x_axis.x > 0.0, "the pair stretches along their axis");
        assert!(s.up(1e-6).is_none());
    }

    #[test]
    fn edits_are_corrections_that_never_double_count() {
        let mut f = cube();
        let p = DVec3::new(3.0, 10.5, -2.0);
        let base = f.sample(p);
        // Place a 3x3x3 lump of amount-5 cells just below the eye and remove it again.
        for x in 0..3 {
            for y in 0..3 {
                for z in 0..3 {
                    f.record((x, y, z), 5);
                }
            }
        }
        let lumped = f.sample(p);
        assert!(lumped.accel.y < base.accel.y, "the lump pulls down");
        assert!(lumped.epoch > base.epoch);
        for x in 0..3 {
            for y in 0..3 {
                for z in 0..3 {
                    f.record((x, y, z), -5);
                }
            }
        }
        assert_eq!(f.sample(p).accel, base.accel, "undoing restores the baseline exactly");
    }

    #[test]
    fn far_bodies_beyond_the_range_are_ignored() {
        let far = Primitive::new(Shape::Ball { c: DVec3::new(6.0e8, 0.0, 0.0), r: 1.0e7 }, 5.0);
        let f = Field::new(Arc::new(Primitives(vec![far])));
        assert_eq!(f.sample(DVec3::ZERO).accel, DVec3::ZERO);
    }
}
