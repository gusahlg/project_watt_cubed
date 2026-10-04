//! Generation-time relaxation (owner decision 2026-10-04: "physics shape, fitted grid").
//!
//! A generated body is placed as a cube of matter (optionally hollow). It relaxes under its own
//! gravity by the same law as everything else, and its equilibrium shape decides its layout:
//!
//! - a surface that stays close to the cube (strong matter: a large face stays at a slope to the
//!   radial direction) keeps its cube grid — [`Layout::Cube`];
//! - a surface that rounds is laid out on cube-sphere charts fitted to it — [`Layout::Round`]: a
//!   datum radius and the relaxed surface's offsets from it ([`DatumField`]), so blocks stay nearly
//!   cubic and level while the shape is the one physics produced.
//!
//! The rule reads only the relaxed geometry (no body kinds). Matter still flowing when the coarse
//! cube grid reaches its representation limit, with `Π_g = Gρ²L²/Y` far above one, is taken to its
//! hydrostatic figure (a sphere for a non-rotating body); that limit is reported, not hidden.
//!
//! Lattice edges are powers of two, so a body is relaxed at a nearby computational size with yield
//! and shear scaled by `(L′/L)²`: `Π_g` is unchanged, the flow pattern is the same, and the shape
//! scales back by `L/L′`.

use glam::DVec3;

use super::lattice::Lattice;
use super::material::Params;
use super::solver::{Body, Relax, Report};
use crate::gravity::G;
use crate::space::datum::DatumField;

/// A relaxed surface tilting more than this from the radial direction keeps its cube grid.
pub const MAX_TILT_DEG: f64 = 25.0;
/// Datum samples per face edge.
pub const DATUM_SAMPLES: usize = 33;
/// Above this `Π_g`, flow stopped by the representation limit is taken to the hydrostatic figure.
pub const FLUID_PI: f64 = 100.0;

/// What to relax.
#[derive(Clone, Copy, Debug)]
pub struct Spec {
    /// Half-size of the cube of matter, blocks.
    pub half: f64,
    /// Half-size of a cubic cavity at the centre (a hollow world), blocks.
    pub cavity_half: Option<f64>,
    /// The (homogenised) matter.
    pub matter: Params,
    /// Matter elements per axis.
    pub elements: usize,
}

/// The layout the relaxed shape asks for.
#[derive(Clone, Debug)]
pub enum Layout {
    /// Keeps its cube grid.
    Cube,
    /// Rounded: an atlas of `radius` with the relaxed outer surface's offsets; a hollow body also
    /// gets its inner surface.
    Round { radius: f64, datum: DatumField, inner: Option<(f64, DatumField)> },
}

/// The outcome of a relaxation.
pub struct Relaxed {
    pub layout: Layout,
    pub report: Report,
    /// `Π_g` of the body.
    pub pi: f64,
    /// Largest angle (degrees) between the relaxed outer surface's normal and the radial direction.
    pub max_tilt: f64,
    /// Corner over face-centre radius of the outer surface (1.732 for a cube, 1 for a ball).
    pub roundness: f64,
    /// Whether the hydrostatic figure stood in for an unfinished flow.
    pub hydrostatic: bool,
}

/// Relax a body and choose its layout.
pub fn relax(spec: &Spec) -> Relaxed {
    let n = spec.elements.max(2);
    let want = 2.0 * spec.half / n as f64;
    let cell = 1i64 << (want.log2().round().max(4.0) as u32);
    let half_c = n as f64 * cell as f64 / 2.0;
    let scale = spec.half / half_c;
    // Π-preserving computational matter.
    let s2 = 1.0 / (scale * scale);
    let mut matter = spec.matter;
    matter.yield_stress *= s2;
    matter.shear *= s2;
    let void = Params::void(&matter);
    let dims = n + 2;
    let centre_c = dims as f64 * cell as f64 / 2.0;
    let cavity_c = spec.cavity_half.map(|c| c / scale);
    let params: Vec<Params> = (0..dims * dims * dims)
        .map(|e| {
            let ijk = [e % dims, (e / dims) % dims, e / (dims * dims)];
            if ijk.iter().any(|&q| q == 0 || q == dims - 1) {
                return void;
            }
            if let Some(cav) = cavity_c {
                let mid = ijk.map(|q| (q as f64 + 0.5) * cell as f64 - centre_c);
                if mid.iter().all(|m| m.abs() < cav) {
                    return void;
                }
            }
            matter
        })
        .collect();
    let lattice = Lattice::undeformed([0, 0, 0], cell, [dims; 3], DVec3::splat(-centre_c));
    let mut body = Body::new(lattice, params);
    let report = body.relax(&Relax { max_iterations: 40_000, tolerance: 2e-4, ..Relax::default() });
    let pi = G * spec.matter.density * spec.matter.density * spec.half * spec.half / spec.matter.yield_stress;

    // Outer surface of the matter block: element layers 1..=n, nodes 1..=n+1.
    let lat = &body.lattice;
    let outer = block_surface(lat, 1, n + 1);
    let centre = centroid(lat, 1, n + 1);
    let roundness = {
        let (c, f) = (lat.node(n + 1, n + 1, n + 1), lat.node(dims / 2, n + 1, dims / 2));
        (lat.nodes[c] - centre).length() / (lat.nodes[f] - centre).length()
    };
    let max_tilt = outer
        .iter()
        .map(|q| {
            let mid = (q[0] + q[1] + q[2] + q[3]) * 0.25;
            let normal = (q[3] - q[0]).cross(q[2] - q[1]).normalize_or_zero();
            let radial = (mid - centre).normalize_or_zero();
            normal.dot(radial).abs().clamp(0.0, 1.0).acos().to_degrees()
        })
        .fold(0.0f64, f64::max);
    let hydrostatic = !report.converged && pi > FLUID_PI;
    let volume = 8.0 * spec.half.powi(3) - spec.cavity_half.map_or(0.0, |c| 8.0 * c.powi(3));
    let layout = if hydrostatic {
        // Still flowing at the representation limit, far into the fluid regime: the end state is the
        // hydrostatic figure — a sphere (a spherical shell for a hollow body) of the same volume.
        let outer_r = match spec.cavity_half {
            None => (3.0 * volume / (4.0 * std::f64::consts::PI)).cbrt(),
            Some(c) => (3.0 * (volume + 8.0 * c.powi(3)) / (4.0 * std::f64::consts::PI)).cbrt(),
        };
        let inner = spec.cavity_half.map(|c| ((3.0 * 8.0 * c.powi(3)) / (4.0 * std::f64::consts::PI)).cbrt());
        Layout::Round {
            radius: outer_r,
            datum: DatumField::flat(DATUM_SAMPLES),
            inner: inner.map(|r| (r, DatumField::flat(DATUM_SAMPLES))),
        }
    } else if max_tilt > MAX_TILT_DEG {
        Layout::Cube
    } else {
        let radius_of = |quads: &[[DVec3; 4]], d: DVec3| cast(quads, centre, d).map(|t| t * scale);
        let fit = |quads: &[[DVec3; 4]]| {
            let mut samples = Vec::new();
            let raw = DatumField::sample(DATUM_SAMPLES, 0.0, |d| {
                let r = radius_of(quads, d).unwrap_or(f64::NAN);
                samples.push(r);
                r
            });
            let good: Vec<f64> = samples.iter().copied().filter(|r| r.is_finite()).collect();
            let mean = good.iter().sum::<f64>() / good.len().max(1) as f64;
            let offsets = raw.offsets.iter().map(|&o| if o.is_finite() { (o as f64 - mean) as f32 } else { 0.0 }).collect();
            (mean, DatumField { g: raw.g, offsets })
        };
        let (radius, datum) = fit(&outer);
        let inner = cavity_c.map(|cav| {
            // The cavity's walls: element layers just outside the void core.
            let k = ((centre_c - cav) / cell as f64).round() as usize;
            let quads = block_surface(lat, k, dims - k);
            fit(&quads)
        });
        Layout::Round { radius, datum, inner }
    };
    Relaxed { layout, report, pi, max_tilt, roundness, hydrostatic }
}

/// Physical quads of the surface of the node block `[lo, hi]³` (outward winding not required).
fn block_surface(l: &Lattice, lo: usize, hi: usize) -> Vec<[DVec3; 4]> {
    let mut quads = Vec::new();
    for a in lo..hi {
        for b in lo..hi {
            for (fixed, axis) in [(lo, 0usize), (hi, 0), (lo, 1), (hi, 1), (lo, 2), (hi, 2)] {
                let at = |u: usize, v: usize| {
                    let ijk = match axis {
                        0 => [fixed, u, v],
                        1 => [u, fixed, v],
                        _ => [u, v, fixed],
                    };
                    l.nodes[l.node(ijk[0], ijk[1], ijk[2])]
                };
                quads.push([at(a, b), at(a + 1, b), at(a, b + 1), at(a + 1, b + 1)]);
            }
        }
    }
    quads
}

/// Mean of the nodes of the block `[lo, hi]³`.
fn centroid(l: &Lattice, lo: usize, hi: usize) -> DVec3 {
    let mut sum = DVec3::ZERO;
    let mut count = 0.0;
    for k in lo..=hi {
        for j in lo..=hi {
            for i in lo..=hi {
                sum += l.nodes[l.node(i, j, k)];
                count += 1.0;
            }
        }
    }
    sum / count
}

/// Distance from `origin` along unit `d` to the outermost crossing of the quad surface.
fn cast(quads: &[[DVec3; 4]], origin: DVec3, d: DVec3) -> Option<f64> {
    let mut best: Option<f64> = None;
    for q in quads {
        for tri in [[q[0], q[1], q[3]], [q[0], q[3], q[2]]] {
            if let Some(t) = ray_triangle(origin, d, tri) {
                best = Some(best.map_or(t, |b: f64| b.max(t)));
            }
        }
    }
    best
}

/// Möller–Trumbore: the ray parameter of `origin + t d` crossing the triangle, if `t > 0`.
fn ray_triangle(origin: DVec3, d: DVec3, tri: [DVec3; 3]) -> Option<f64> {
    let e1 = tri[1] - tri[0];
    let e2 = tri[2] - tri[0];
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-12 * e1.length() * e2.length() {
        return None;
    }
    let inv = 1.0 / det;
    let s = origin - tri[0];
    let u = s.dot(p) * inv;
    if !(-1e-9..=1.0 + 1e-9).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = d.dot(q) * inv;
    if v < -1e-9 || u + v > 1.0 + 1e-9 {
        return None;
    }
    let t = e2.dot(q) * inv;
    (t > 0.0).then_some(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_strong_cube_keeps_its_grid_and_a_weak_one_is_charted() {
        // Strong: barely deforms, its faces stay at up to 45° to the radial direction.
        let strong = relax(&Spec { half: 200_000.0, cavity_half: None, matter: Params::from_yield(5.0, 1e9), elements: 4 });
        assert!(matches!(strong.layout, Layout::Cube), "tilt {}", strong.max_tilt);
        assert!(strong.pi < 1.0);
        // Weak: Π ≫ 1, it rounds; the layout is charted with the radius of a ball of its volume (or
        // its relaxed surface's mean radius when the flow finished).
        let weak = relax(&Spec { half: 200_000.0, cavity_half: None, matter: Params::from_yield(5.0, 1e2), elements: 4 });
        let Layout::Round { radius, .. } = weak.layout else { panic!("weak cube kept its grid: tilt {}", weak.max_tilt) };
        let ball = (3.0 * 8.0 * 200_000f64.powi(3) / (4.0 * std::f64::consts::PI)).cbrt();
        assert!((radius - ball).abs() / ball < 0.08, "radius {radius} vs ball {ball}");
        assert!(weak.pi > FLUID_PI);
    }

    #[test]
    fn a_ray_crosses_a_triangle() {
        let tri = [DVec3::new(1.0, -1.0, -1.0), DVec3::new(1.0, 1.0, -1.0), DVec3::new(1.0, 0.0, 1.0)];
        assert!((ray_triangle(DVec3::ZERO, DVec3::X, tri).unwrap() - 1.0).abs() < 1e-12);
        assert!(ray_triangle(DVec3::ZERO, -DVec3::X, tri).is_none());
    }
}
