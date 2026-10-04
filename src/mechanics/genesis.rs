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
/// Above this `Π_g`, flow still unfinished after every re-gridding stage is taken to the
/// hydrostatic figure.
pub const FLUID_PI: f64 = 100.0;
/// Re-grid when the worst element's certified quality falls below this fraction of what the stage
/// started with (before the surface can fold).
const REGRID_FRACTION: f64 = 0.4;
/// A re-gridded lattice worse than this is re-gridded again from a filtered surface.
const SMOOTH_BELOW: f64 = 0.2;
/// Equilibrium: the largest nodal residual over the largest nodal weight.
const TOLERANCE: f64 = 2e-4;
/// Matter elements per edge of a volume-preserving block (see `Body::group_blocks`).
const VOLUME_BLOCK: usize = 2;

/// What to relax.
#[derive(Clone, Copy, Debug)]
pub struct Spec {
    /// Half-size of the cube of matter, blocks.
    pub half: f64,
    /// Half-size of a cubic cavity at the centre (a hollow world), blocks.
    pub cavity_half: Option<f64>,
    /// The (homogenised) matter.
    pub matter: Params,
    /// Matter elements per axis (rounded up to even).
    pub elements: usize,
    /// Print the solver's progress every this many iterations, and each re-gridding (0: quiet).
    pub trace: usize,
    /// Re-griddings and relaxation iterations allowed ([`Patience::QUICK`] for live use).
    pub patience: Patience,
}

/// How long a relaxation may run.
#[derive(Clone, Copy, Debug)]
pub struct Patience {
    pub stages: usize,
    pub iterations: usize,
}

impl Patience {
    /// Live relaxation.
    pub const QUICK: Patience = Patience { stages: 8, iterations: 60_000 };
    /// Offline tables (`genesis_table`).
    pub const TABLE: Patience = Patience { stages: 40, iterations: 600_000 };
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

/// A relaxed body on its computational lattice.
pub struct Solved {
    /// The lattice: matter elements `1..=n` per axis inside one element of void, centred on the
    /// origin, in computational units (multiply by `scale` for blocks).
    pub lattice: Lattice,
    /// Blocks per computational unit.
    pub scale: f64,
    /// Matter elements per axis.
    pub elements: usize,
    pub report: Report,
    /// `Π_g` of the body.
    pub pi: f64,
    /// Re-griddings used.
    pub stages: usize,
    /// The flow folded the lattice past what re-gridding can follow (a cavity closing, a surface
    /// overturning); the lattice is the last valid state.
    pub folded: bool,
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
    /// See [`Solved::folded`].
    pub folded: bool,
}

/// `Π_g = Gρ²L²/Y` of a body of half-size `half`.
pub fn pi_g(matter: &Params, half: f64) -> f64 {
    G * matter.density * matter.density * half * half / matter.yield_stress
}

/// Relax a body and choose its layout.
pub fn relax(spec: &Spec) -> Relaxed {
    choose(&solve(spec), spec.half, spec.cavity_half)
}

/// Relax a body's cube of matter to equilibrium under its own gravity.
pub fn solve(spec: &Spec) -> Solved {
    let n = spec.elements.max(2).div_ceil(2) * 2;
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
    let fresh = |lattice: Lattice| {
        let mut body = Body::new(lattice, params.clone());
        body.group_blocks(VOLUME_BLOCK, 1);
        body
    };
    let mut body = fresh(lattice);
    // Relax in stages: when the cube grid's elements degrade before the flow ends, fit a fresh
    // lattice radially onto the current matter surface (stresses rebuild within a fraction of a
    // percent of strain) and keep flowing, so the equilibrium is reached rather than assumed.
    let cav_k = cavity_c.map(|c| c / half_c);
    let (mut spent, mut stages, mut folded) = (0usize, 0usize, false);
    let report = loop {
        let floor = REGRID_FRACTION * body.min_quality();
        let budget = spec.patience.iterations;
        let rep = body.relax(&Relax { max_iterations: budget - spent, tolerance: TOLERANCE, min_quality: floor, plastic_relax: 1.0, trace: spec.trace });
        spent += rep.iterations;
        let report = Report { iterations: spent, ..rep };
        if rep.converged || stages == spec.patience.stages || spent >= budget || !rep.residual.is_finite() {
            folded |= !rep.residual.is_finite();
            break report;
        }
        // Re-grid onto the surface as it is; only when that leaves poor elements (the surface has
        // grown a grid-scale wrinkle) filter the wrinkle out first.
        let mut next = fresh(regrid(&body.lattice, n, cell, centre_c, cav_k, false));
        let mut q = next.min_quality();
        if q < SMOOTH_BELOW {
            let smoothed = fresh(regrid(&body.lattice, n, cell, centre_c, cav_k, true));
            let qs = smoothed.min_quality();
            if qs > q {
                (next, q) = (smoothed, qs);
            }
        }
        if spec.trace > 0 {
            eprintln!("  stage {stages}: {} its, quality {:.3} → re-gridded quality {q:.3}, residual {:.2e}", rep.iterations, rep.min_quality, rep.residual);
        }
        if !(q > 0.0) {
            folded = true;
            break report;
        }
        body = next;
        stages += 1;
    };
    let lattice = body.lattice;
    Solved { lattice, scale, elements: n, report, pi: pi_g(&spec.matter, spec.half), stages, folded }
}

/// Choose the layout of a relaxed body of half-size `half` (blocks) from its shape.
pub fn choose(solved: &Solved, half: f64, cavity_half: Option<f64>) -> Relaxed {
    let n = solved.elements;
    let dims = n + 2;
    let scale = solved.scale;
    let report = solved.report;
    let pi = solved.pi;
    // Outer surface of the matter block: element layers 1..=n, nodes 1..=n+1.
    let lat = &solved.lattice;
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
    let volume = 8.0 * half.powi(3) - cavity_half.map_or(0.0, |c| 8.0 * c.powi(3));
    let layout = if hydrostatic {
        // Still flowing at the representation limit, far into the fluid regime: the end state is the
        // hydrostatic figure — a sphere (a spherical shell for a hollow body) of the same volume.
        let outer_r = match cavity_half {
            None => (3.0 * volume / (4.0 * std::f64::consts::PI)).cbrt(),
            Some(c) => (3.0 * (volume + 8.0 * c.powi(3)) / (4.0 * std::f64::consts::PI)).cbrt(),
        };
        let inner = cavity_half.map(|c| ((3.0 * 8.0 * c.powi(3)) / (4.0 * std::f64::consts::PI)).cbrt());
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
        let inner = cavity_half.map(|cav| {
            // The cavity's walls: element layers just outside the void core.
            let cell = lat.cell as f64;
            let k = ((dims as f64 * cell / 2.0 - cav / scale) / cell).round() as usize;
            let quads = block_surface(lat, k, dims - k);
            fit(&quads)
        });
        Layout::Round { radius, datum, inner }
    };
    Relaxed { layout, report, pi, max_tilt, roundness, hydrostatic, folded: solved.folded }
}

/// A fresh lattice over the same reference box, fitted radially onto `old`'s matter surface (and
/// cavity wall): a node at cube-norm fraction `k` of the matter half-size sits at `k · r(d)` from
/// the centre (between the cavity wall and the surface for a hollow body), along the equiangular
/// direction `d` of its cube-surface point — the cube-sphere layout's own directions, so a round
/// body's cells stay within a third of each other in size instead of the fivefold spread of a
/// gnomonic projection. `r` is sampled at the fresh surface's nodes and interpolated bilinearly.
fn regrid(old: &Lattice, n: usize, cell: i64, centre_c: f64, cav_k: Option<f64>, smooth: bool) -> Lattice {
    let dims = n + 2;
    let half_c = n as f64 * cell as f64 / 2.0;
    let centre = centroid(old, 1, n + 1);
    let outer = SurfaceRadius::fit(&block_surface(old, 1, n + 1), centre, n, half_c, smooth);
    let inner = cav_k.map(|k| {
        let layer = ((centre_c - k * half_c) / cell as f64).round() as usize;
        SurfaceRadius::fit(&block_surface(old, layer, dims - layer), centre, n, half_c, smooth)
    });
    let mut fresh = Lattice::undeformed([0, 0, 0], cell, [dims; 3], DVec3::splat(-centre_c));
    for idx in 0..fresh.nodes.len() {
        let x = fresh.nodes[idx];
        let k = x.abs().max_element() / half_c;
        if k < 1e-12 {
            fresh.nodes[idx] = centre;
            continue;
        }
        let on_cube = x / k;
        let d = equiangular(on_cube, half_c);
        let r_out = outer.at(on_cube);
        let r = match (cav_k, &inner) {
            (Some(kc), Some(walls)) => {
                let r_in = walls.at(on_cube);
                if k <= kc {
                    r_in * k / kc
                } else if k <= 1.0 {
                    r_in + (r_out - r_in) * (k - kc) / (1.0 - kc)
                } else {
                    r_out * k
                }
            }
            _ => r_out * k,
        };
        fresh.nodes[idx] = centre + d * r;
    }
    // A radial map pinches the elements at the centre (the cube's corner directions close up onto
    // the ball's): relax every node strictly inside the outer surface, except the cavity wall, to
    // the mean of its six neighbours — the discrete harmonic map onto the fitted surfaces.
    let mid = dims / 2;
    let layer = |i: usize, j: usize, k: usize| i.abs_diff(mid).max(j.abs_diff(mid)).max(k.abs_diff(mid));
    let wall = cav_k.map(|kc| (kc * n as f64 / 2.0).round() as usize);
    let free: Vec<usize> = (1..dims)
        .flat_map(|k| (1..dims).flat_map(move |j| (1..dims).map(move |i| (i, j, k))))
        .filter(|&(i, j, k)| {
            let l = layer(i, j, k);
            l < n / 2 && Some(l) != wall
        })
        .map(|(i, j, k)| fresh.node(i, j, k))
        .collect();
    // Jacobi sweeps (order independent, so a symmetric body stays symmetric).
    let stride = [1, dims + 1, (dims + 1) * (dims + 1)];
    let mut next = vec![DVec3::ZERO; free.len()];
    for _ in 0..4 * n {
        for (slot, &v) in next.iter_mut().zip(&free) {
            let mut sum = DVec3::ZERO;
            for s in stride {
                sum += fresh.nodes[v - s] + fresh.nodes[v + s];
            }
            *slot = sum / 6.0;
        }
        for (&v, &p) in free.iter().zip(&next) {
            fresh.nodes[v] = p;
        }
    }
    // Fitting facets onto a facetted surface loses a sliver of volume each time: scale about the
    // centre so the matter keeps its volume exactly.
    let matter = |l: &Lattice| -> f64 {
        let mut v = 0.0;
        for k in 1..=n {
            for j in 1..=n {
                for i in 1..=n {
                    let core = cav_k.is_some_and(|kc| [i, j, k].iter().all(|&q| ((q as f64 + 0.5) - dims as f64 / 2.0).abs() < kc * n as f64 / 2.0));
                    if !core {
                        v += super::solver::element_volume(&l.corners(l.element(i, j, k)));
                    }
                }
            }
        }
        v
    };
    let factor = (matter(old) / matter(&fresh)).cbrt();
    for p in fresh.nodes.iter_mut() {
        *p = centre + (*p - centre) * factor;
    }
    fresh.reindex();
    fresh
}

/// A surface's radius along the equiangular directions of the cube of half-size `half`, on an
/// `n × n` grid per face.
struct SurfaceRadius {
    n: usize,
    half: f64,
    /// Per face (axis, sign) row-major `(n + 1)²`.
    faces: Vec<Vec<f64>>,
}

impl SurfaceRadius {
    fn fit(quads: &[[DVec3; 4]], centre: DVec3, n: usize, half: f64, smooth: bool) -> Self {
        let side = n + 1;
        let h = 2.0 * half / n as f64;
        let point = |f: usize, u: usize, v: usize| -> DVec3 {
            let (axis, sign) = (f / 2, if f % 2 == 0 { 1.0 } else { -1.0 });
            let (a, b) = (-half + u as f64 * h, -half + v as f64 * h);
            match axis {
                0 => DVec3::new(sign * half, a, b),
                1 => DVec3::new(a, sign * half, b),
                _ => DVec3::new(a, b, sign * half),
            }
        };
        // The surface grid as one graph: points shared by two or three faces are one node.
        let key = |p: DVec3| (p / h).round().as_i64vec3();
        let mut ids = std::collections::BTreeMap::new();
        let mut pts = Vec::new();
        let slot: Vec<usize> = (0..6)
            .flat_map(|f| (0..side).flat_map(move |v| (0..side).map(move |u| (f, u, v))))
            .map(|(f, u, v)| {
                let p = point(f, u, v);
                *ids.entry(key(p).to_array()).or_insert_with(|| {
                    pts.push(p);
                    pts.len() - 1
                })
            })
            .collect();
        let mut r: Vec<f64> = pts.iter().map(|&p| cast(quads, centre, equiangular(p, half)).unwrap_or(f64::NAN)).collect();
        let good: Vec<f64> = r.iter().copied().filter(|v| v.is_finite()).collect();
        let mean = good.iter().sum::<f64>() / good.len().max(1) as f64;
        for v in r.iter_mut() {
            if !v.is_finite() {
                *v = mean;
            }
        }
        // Grid neighbours: one step along the surface (three at a cube corner, four elsewhere).
        let neighbours: Vec<Vec<usize>> = pts
            .iter()
            .map(|&p| {
                let k = key(p);
                pts.iter().enumerate().filter(|&(_, &q)| (key(q) - k).abs().element_sum() == 1).map(|(j, _)| j).collect()
            })
            .collect();
        // A ray through a fold reads the wrong sheet: a sample far off its neighbours' median takes
        // that median. With `smooth`, one pass of the grid filter `½ r + ½ mean(neighbours)`
        // follows: it removes the two-cell wrinkle a free surface grows at the grid scale (which
        // the representation cannot resolve) and keeps a feature four cells wide at three quarters,
        // eight wide at 0.93.
        let spacing = mean * std::f64::consts::FRAC_PI_2 / n as f64;
        let repaired: Vec<f64> = neighbours
            .iter()
            .enumerate()
            .map(|(i, nb)| {
                let mut v: Vec<f64> = nb.iter().map(|&j| r[j]).collect();
                v.sort_by(f64::total_cmp);
                let median = v[v.len() / 2];
                if (r[i] - median).abs() > spacing { median } else { r[i] }
            })
            .collect();
        r = if smooth {
            neighbours.iter().enumerate().map(|(i, nb)| 0.5 * repaired[i] + 0.5 * nb.iter().map(|&j| repaired[j]).sum::<f64>() / nb.len() as f64).collect()
        } else {
            repaired
        };
        let faces = (0..6).map(|f| (0..side * side).map(|k| r[slot[f * side * side + k]]).collect()).collect();
        Self { n, half, faces }
    }

    /// Radius at point `p` of the reference cube surface.
    fn at(&self, p: DVec3) -> f64 {
        let a = p.abs();
        let axis = if a.x >= a.y && a.x >= a.z { 0 } else if a.y >= a.z { 1 } else { 2 };
        let f = axis * 2 + usize::from(p[axis] < 0.0);
        let (pa, pb) = match axis {
            0 => (p.y, p.z),
            1 => (p.x, p.z),
            _ => (p.x, p.y),
        };
        let s = self.n as f64;
        let u = ((pa + self.half) / (2.0 * self.half) * s).clamp(0.0, s);
        let v = ((pb + self.half) / (2.0 * self.half) * s).clamp(0.0, s);
        let (u0, v0) = ((u.floor() as usize).min(self.n - 1), (v.floor() as usize).min(self.n - 1));
        let (fu, fv) = (u - u0 as f64, v - v0 as f64);
        let side = self.n + 1;
        let g = &self.faces[f];
        let at = |i: usize, j: usize| g[j * side + i];
        let a0 = at(u0, v0) * (1.0 - fu) + at(u0 + 1, v0) * fu;
        let a1 = at(u0, v0 + 1) * (1.0 - fu) + at(u0 + 1, v0 + 1) * fu;
        a0 * (1.0 - fv) + a1 * fv
    }
}

/// The equiangular cube-sphere direction of a point `p` on the surface of the cube of half-size
/// `half`: face coordinates `(a, b) ∈ [−1, 1]²` go to `(tan(πa/4), tan(πb/4))`.
fn equiangular(p: DVec3, half: f64) -> DVec3 {
    let m = p.abs();
    let axis = if m.x >= m.y && m.x >= m.z { 0 } else if m.y >= m.z { 1 } else { 2 };
    let mut d = DVec3::ZERO;
    for a in 0..3 {
        d[a] = if a == axis { p[a].signum() } else { (std::f64::consts::FRAC_PI_4 * (p[a] / half).clamp(-1.0, 1.0)).tan() };
    }
    d.normalize()
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

/// Distance from `origin` along unit `d` to the outermost crossing of the quad surface. A curved
/// quad is crossed as the mean of its two triangulations, so the answer does not depend on which
/// diagonal a quad was split along (a mirror-symmetric surface casts mirror-symmetric radii).
fn cast(quads: &[[DVec3; 4]], origin: DVec3, d: DVec3) -> Option<f64> {
    let mut best: Option<f64> = None;
    let hit = |a: [DVec3; 3], b: [DVec3; 3]| ray_triangle(origin, d, a).or_else(|| ray_triangle(origin, d, b));
    for q in quads {
        let one = hit([q[0], q[1], q[3]], [q[0], q[3], q[2]]);
        let other = hit([q[0], q[1], q[2]], [q[1], q[3], q[2]]);
        let t = match (one, other) {
            (Some(a), Some(b)) => Some(0.5 * (a + b)),
            (a, b) => a.or(b),
        };
        if let Some(t) = t {
            best = Some(best.map_or(t, |b: f64| b.max(t)));
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

/// Relaxed solid cubes by `Π_g`, computed offline by `genesis_table` (see that binary): one entry
/// per half octave of `Π_g`, each with its layout figures and datum (one face; a cube-symmetric
/// shape has the same field on every face) and the lattice's nodes in the fundamental domain of
/// the cube's symmetry. All lengths in units of the cube's half-size.
static TABLE: &[u8] = include_bytes!("genesis_table.bin");
/// Format version of [`TABLE`] ([`read_table`] also reads version 1, nodes only).
pub const TABLE_VERSION: u32 = 2;
/// Half-size of the table's computational cube: 8 matter elements of 2^20 blocks each side.
pub const TABLE_HALF: f64 = 8.0 * (1u64 << 20) as f64;
/// Matter elements per axis of the table's lattices.
pub const TABLE_ELEMENTS: usize = 16;

/// One tabulated relaxed cube.
#[derive(Clone, Debug, Default)]
pub struct TableEntry {
    pub log2_pi: f64,
    pub converged: bool,
    pub max_tilt: f64,
    pub roundness: f64,
    /// Datum radius; zero when the shape keeps its cube grid.
    pub radius: f64,
    /// Datum offsets of one face (`g × g`, [`DatumField`] order).
    pub offsets: Vec<f32>,
    /// Canonical nodes ([`canonical_nodes`] order).
    pub nodes: Vec<DVec3>,
}

/// A table of relaxed cubes.
#[derive(Clone, Debug, Default)]
pub struct Table {
    pub elements: usize,
    pub samples: usize,
    pub entries: Vec<TableEntry>,
}

/// Parse a table (version 1: nodes only, no layout figures).
pub fn read_table(bytes: &[u8]) -> Option<Table> {
    let mut r = Reader { bytes, at: 0 };
    if bytes.len() < 16 || r.take(4)? != b"PWCG" {
        return None;
    }
    let version = r.u32()?;
    if version != 1 && version != 2 {
        return None;
    }
    let elements = r.u32()? as usize;
    let count = r.u32()? as usize;
    let samples = if version >= 2 { r.u32()? as usize } else { 0 };
    let canon = canonical_nodes((elements + 2) / 2).len();
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let mut e = TableEntry { log2_pi: r.f32()? as f64, converged: r.u32()? != 0, ..TableEntry::default() };
        if version >= 2 {
            e.max_tilt = r.f32()? as f64;
            e.roundness = r.f32()? as f64;
            e.radius = r.f32()? as f64;
            e.offsets = (0..samples * samples).map(|_| r.f32()).collect::<Option<_>>()?;
        }
        e.nodes = (0..canon).map(|_| Some(DVec3::new(r.f32()? as f64, r.f32()? as f64, r.f32()? as f64))).collect::<Option<_>>()?;
        entries.push(e);
    }
    Some(Table { elements, samples, entries })
}

/// Serialise a table in the current format.
pub fn write_table(t: &Table) -> Vec<u8> {
    let mut out = Vec::new();
    let u = |v: u32, out: &mut Vec<u8>| out.extend_from_slice(&v.to_le_bytes());
    out.extend_from_slice(b"PWCG");
    u(TABLE_VERSION, &mut out);
    u(t.elements as u32, &mut out);
    u(t.entries.len() as u32, &mut out);
    u(t.samples as u32, &mut out);
    let f = |v: f64, out: &mut Vec<u8>| out.extend_from_slice(&(v as f32).to_le_bytes());
    for e in &t.entries {
        f(e.log2_pi, &mut out);
        out.extend_from_slice(&u32::from(e.converged).to_le_bytes());
        f(e.max_tilt, &mut out);
        f(e.roundness, &mut out);
        f(e.radius, &mut out);
        for &o in &e.offsets {
            out.extend_from_slice(&o.to_le_bytes());
        }
        for p in &e.nodes {
            for a in 0..3 {
                f(p[a], &mut out);
            }
        }
    }
    out
}

/// Lattice nodes `(p, q, r)` with `0 ≤ p ≤ q ≤ r ≤ m` (offsets from the centre node; `m` is half the
/// node count per axis): every node is one of these up to the cube's 48 symmetries.
pub fn canonical_nodes(m: usize) -> Vec<[usize; 3]> {
    let mut out = Vec::new();
    for r in 0..=m {
        for q in 0..=r {
            for p in 0..=q {
                out.push([p, q, r]);
            }
        }
    }
    out
}

/// The canonical node of offset `o` and the map back: `x[axis] = sign[axis] · canonical[slot[axis]]`.
pub fn canonical_of(o: [i64; 3]) -> ([usize; 3], [usize; 3], [f64; 3]) {
    let mut order = [0usize, 1, 2];
    order.sort_by_key(|&a| (o[a].unsigned_abs(), a));
    let canon = order.map(|a| o[a].unsigned_abs() as usize);
    let mut slot = [0usize; 3];
    for (s, &a) in order.iter().enumerate() {
        slot[a] = s;
    }
    let sign = o.map(|v| if v < 0 { -1.0 } else { 1.0 });
    (canon, slot, sign)
}

/// A relaxed body from canonical node positions (units of the half-size) for half-size `half`.
pub fn entry_solved(elements: usize, nodes: &[DVec3], converged: bool, pi: f64, half: f64) -> Solved {
    let dims = elements + 2;
    let m = dims / 2;
    let index: std::collections::HashMap<[usize; 3], usize> = canonical_nodes(m).into_iter().enumerate().map(|(i, c)| (c, i)).collect();
    let cell = 1i64 << 20;
    let half_c = elements as f64 * cell as f64 / 2.0;
    let mut lattice = Lattice::undeformed([0, 0, 0], cell, [dims; 3], DVec3::splat(-(m as f64) * cell as f64));
    for k in 0..=dims {
        for j in 0..=dims {
            for i in 0..=dims {
                let o = [i as i64 - m as i64, j as i64 - m as i64, k as i64 - m as i64];
                let (c, slot, sign) = canonical_of(o);
                let p = nodes[index[&c]];
                let node = lattice.node(i, j, k);
                lattice.nodes[node] = DVec3::new(sign[0] * p[slot[0]], sign[1] * p[slot[1]], sign[2] * p[slot[2]]) * half_c;
            }
        }
    }
    lattice.reindex();
    let min_quality = (0..lattice.elements()).map(|e| lattice.certify(e)).fold(f64::INFINITY, f64::min);
    let report = Report { converged, min_quality, ..Report::default() };
    Solved { lattice, scale: half / half_c, elements, report, pi, stages: 0, folded: false }
}

/// The bracketing entries of `log₂ Π` and the weight of the upper one (clamped at the ends).
fn bracket(t: &Table, pi: f64) -> Option<(&TableEntry, &TableEntry, f64)> {
    let first = t.entries.first()?;
    let last = t.entries.last()?;
    let x = pi.max(1e-300).log2().clamp(first.log2_pi, last.log2_pi);
    let count = t.entries.len();
    if count == 1 {
        return Some((first, first, 0.0));
    }
    let hi = t.entries.iter().position(|e| e.log2_pi >= x).unwrap_or(count - 1).clamp(1, count - 1);
    let (a, b) = (&t.entries[hi - 1], &t.entries[hi]);
    Some((a, b, ((x - a.log2_pi) / (b.log2_pi - a.log2_pi)).clamp(0.0, 1.0)))
}

/// The tabulated relaxed cube for `Π_g = pi`, scaled to half-size `half`: node positions
/// interpolated linearly in `log₂ Π` between the bracketing entries. `None` without a table.
pub fn tabulated(pi: f64, half: f64) -> Option<Solved> {
    let t = read_table(TABLE)?;
    let (a, b, w) = bracket(&t, pi)?;
    let nodes: Vec<DVec3> = a.nodes.iter().zip(&b.nodes).map(|(p, q)| *p * (1.0 - w) + *q * w).collect();
    Some(entry_solved(t.elements, &nodes, a.converged && b.converged, pi, half))
}

/// The layout the table gives a solid cube of half-size `half` at `Π_g = pi`, with its interpolated
/// surface tilt (the same rule as [`choose`], read from the figures stored per entry). `None`
/// without a table carrying layouts.
pub fn tabulated_layout(pi: f64, half: f64) -> Option<(Layout, f64)> {
    let t = read_table(TABLE)?;
    if t.samples < 2 {
        return None;
    }
    let (a, b, w) = bracket(&t, pi)?;
    let lerp = |x: f64, y: f64| x * (1.0 - w) + y * w;
    let tilt = lerp(a.max_tilt, b.max_tilt);
    if tilt > MAX_TILT_DEG || a.radius <= 0.0 || b.radius <= 0.0 {
        return Some((Layout::Cube, tilt));
    }
    let face: Vec<f32> = a.offsets.iter().zip(&b.offsets).map(|(&p, &q)| (lerp(p as f64, q as f64) * half) as f32).collect();
    let g = t.samples;
    let offsets = (0..6).flat_map(|_| face.iter().copied()).collect();
    Some((Layout::Round { radius: lerp(a.radius, b.radius) * half, datum: DatumField { g, offsets }, inner: None }, tilt))
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        let s = self.bytes.get(self.at..self.at + n)?;
        self.at += n;
        Some(s)
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn f32(&mut self) -> Option<f32> {
        Some(f32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_strong_cube_keeps_its_grid_and_a_weak_one_is_charted() {
        // Strong: barely deforms, its faces stay at up to 45° to the radial direction.
        let strong = relax(&Spec { half: 200_000.0, cavity_half: None, matter: Params::from_yield(5.0, 1e9), elements: 4, trace: 0, patience: Patience::QUICK });
        assert!(matches!(strong.layout, Layout::Cube), "tilt {}", strong.max_tilt);
        assert!(strong.pi < 1.0);
        // Weak: Π ≫ 1, it rounds; the layout is charted with the radius of a ball of its volume (or
        // its relaxed surface's mean radius when the flow finished).
        let weak = relax(&Spec { half: 200_000.0, cavity_half: None, matter: Params::from_yield(5.0, 1e2), elements: 4, trace: 0, patience: Patience::QUICK });
        let Layout::Round { radius, .. } = weak.layout else { panic!("weak cube kept its grid: tilt {}", weak.max_tilt) };
        let ball = (3.0 * 8.0 * 200_000f64.powi(3) / (4.0 * std::f64::consts::PI)).cbrt();
        assert!((radius - ball).abs() / ball < 0.08, "radius {radius} vs ball {ball}");
        assert!(weak.pi > FLUID_PI);
    }

    #[test]
    fn every_node_maps_to_a_canonical_node_and_back() {
        let m = 3usize;
        let canon = canonical_nodes(m);
        assert_eq!(canon.len(), 20);
        // A symmetric field: position = offset. The canonical node's own position mapped back
        // through slot and sign reproduces every offset.
        for k in -3i64..=3 {
            for j in -3i64..=3 {
                for i in -3i64..=3 {
                    let (c, slot, sign) = canonical_of([i, j, k]);
                    assert!(canon.contains(&c));
                    let back = [0, 1, 2].map(|a| sign[a] * c[slot[a]] as f64);
                    assert_eq!(back, [i as f64, j as f64, k as f64]);
                }
            }
        }
    }

    #[test]
    fn a_ray_crosses_a_triangle() {
        let tri = [DVec3::new(1.0, -1.0, -1.0), DVec3::new(1.0, 1.0, -1.0), DVec3::new(1.0, 0.0, 1.0)];
        assert!((ray_triangle(DVec3::ZERO, DVec3::X, tri).unwrap() - 1.0).abs() < 1e-12);
        assert!(ray_triangle(DVec3::ZERO, -DVec3::X, tri).is_none());
    }
}
