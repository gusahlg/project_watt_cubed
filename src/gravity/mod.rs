//! Gravity derived from matter (guide §6). Every cell's mass is its amount (occurrence count); the
//! field is the vector sum over all matter within the law's range: the generator's matter through
//! its analytic oracle, plus exact corrections for every edit. Nothing here knows what a planet is.

mod field;
mod kernel;
mod ledger;
mod oracle;
pub mod polyhedron;
pub mod relief;
mod shape;

use glam::{DMat3, DVec3};

pub use field::Field;
pub use kernel::{window, EPS, G, R_G, R_IN};
pub use oracle::{Empty, MassOracle, Primitives, Summary, Visitor};
pub use polyhedron::Polyhedron;
pub use shape::{Primitive, Shape};

/// Bumped whenever the law, its constants or the amount rule change; folded into the content
/// fingerprint so peers and saves agree on the physics.
pub const PHYSICS_VERSION: u16 = 1;

/// One evaluation of the field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    /// Acceleration, blocks/s².
    pub accel: DVec3,
    /// Potential, blocks²/s² (zero far from all matter).
    pub potential: f64,
    /// ∂accel/∂position, when asked for.
    pub tidal: Option<DMat3>,
    /// Bound on the error of `accel` from declared approximations, blocks/s².
    pub error: f64,
    /// The source epoch the sample was computed from.
    pub epoch: u64,
}

impl Sample {
    /// A uniform field (tests and the vanilla fallback).
    pub fn uniform(accel: DVec3) -> Self {
        Self { accel, potential: 0.0, tidal: None, error: 0.0, epoch: 0 }
    }

    /// The direction opposite to gravity, or `None` where gravity is weaker than `min` (never
    /// normalises a near-zero vector).
    pub fn up(&self, min: f64) -> Option<DVec3> {
        let g = self.accel.length();
        (g > min && g.is_finite()).then(|| -self.accel / g)
    }
}

/// The constants that define the law, folded into fingerprints.
pub fn law_digest() -> [u64; 5] {
    [PHYSICS_VERSION as u64, G.to_bits(), R_IN.to_bits(), R_G.to_bits(), EPS.to_bits()]
}
