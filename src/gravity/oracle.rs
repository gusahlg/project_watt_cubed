//! The mass oracle: how the world's generated matter is described to gravity without generating
//! voxels (guide §6.5). The generator hands out a hierarchy — summarised groups that a query may
//! accept as one mass or open into their children, down to analytic primitives — and visits only
//! what lies within the law's range of the query. Edits are not the oracle's business: they are
//! exact corrections kept by the field.

use glam::DVec3;

use super::shape::Primitive;

/// A summarised group of sources (a cluster of bodies, a body seen from afar).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Summary {
    /// Centre of the bounding sphere.
    pub centre: DVec3,
    /// Bounding radius around `centre`.
    pub radius: f64,
    /// Total mass (amount).
    pub mass: f64,
    /// Centre of mass.
    pub com: DVec3,
}

/// What a query does with the hierarchy the oracle walks.
pub trait Visitor {
    /// Offered a group: return `true` to open it (the oracle then visits its children), `false`
    /// to accept the summary as one mass (the visitor has used it).
    fn group(&mut self, g: &Summary) -> bool;
    /// One analytic primitive.
    fn primitive(&mut self, p: &Primitive);
    /// A declared approximation error (blocks/s² per unit `G`) the oracle adds for detail it does not
    /// model near this query (surface relief, caves).
    fn error(&mut self, e: f64);
}

/// The generator's description of its matter.
pub trait MassOracle: Send + Sync {
    /// Visit every source whose bounds intersect the sphere `(centre, reach)`.
    fn visit(&self, centre: DVec3, reach: f64, v: &mut dyn Visitor);
}

/// A fixed list of primitives (the flat world, tests).
pub struct Primitives(pub Vec<Primitive>);

impl MassOracle for Primitives {
    fn visit(&self, centre: DVec3, reach: f64, v: &mut dyn Visitor) {
        for p in &self.0 {
            let (c, r) = p.bounds();
            if (c - centre).length() - r < reach {
                v.primitive(p);
            }
        }
    }
}

/// No matter at all.
pub struct Empty;

impl MassOracle for Empty {
    fn visit(&self, _: DVec3, _: f64, _: &mut dyn Visitor) {}
}
