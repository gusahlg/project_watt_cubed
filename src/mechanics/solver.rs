//! The mechanical solve of one body (guide §8): explicit dynamic relaxation to equilibrium under
//! self-gravity, with J2 viscoplasticity and incompressible matter.
//!
//! - **Elements** are the lattice hexes, trilinear, 2×2×2 Gauss points for the deviatoric response.
//! - **Constitutive update** (Wilkins): the deviator follows the Jaumann rate `ṡ = 2μD' + Ws − sW`
//!   with radial return onto the von Mises yield surface (a `relax` fraction below one returns only
//!   part of the overstress: Perzyna creep over a world-time step; one is rate independent).
//! - **Incompressible matter:** planetary pressure compresses matter by ~0.2 % (the material bulk
//!   modulus), which the representation does not resolve, so matter is taken as incompressible.
//!   Every step projects the acceleration so each matter element keeps its volume — one multiplier
//!   per element from a warm-started conjugate-gradient solve, which is the element's pressure at
//!   equilibrium — with a small drift correction back to the reference volume. The explicit part
//!   then sees only shear stiffness, so gravity-driven flow is not throttled by a stiff volumetric
//!   mode. Void (air) is compressible and unconstrained: a neo-Hookean skin that rides along.
//! - **Dynamic relaxation:** fictitious nodal masses scaled for a unit pseudo-time step (Gershgorin
//!   bound of the shear stiffness), Underwood's adaptive damping (critical for the lowest active
//!   mode, estimated from the residual change along the last step), and the rigid-body part of the
//!   residual projected out (bodies are anchored).
//! - **Gravity:** the deformed elements' masses through a Barnes–Hut tree, refreshed every
//!   [`GRAVITY_EVERY`] iterations (staggered coupling).
//!
//! Deterministic for any thread count: elements are processed in fixed contiguous ranges, each
//! writes only its own state and slots, and every gather and reduction runs in a fixed order.

use glam::{DMat3, DVec3};

use super::lattice::{local_jacobian, Lattice};
use super::material::Params;
use super::selfgrav::{Mass, Tree};

/// Gauss points per element.
const GP: usize = 8;
/// Iterations between gravity refreshes.
pub const GRAVITY_EVERY: usize = 8;
/// Safety factor on the fictitious masses.
const MASS_SAFETY: f64 = 2.0;
/// Fraction of an element's volume error corrected per step.
const VOLUME_DRIFT: f64 = 0.2;
/// Conjugate-gradient iterations per projection (warm started from the last step).
const CG_MAX: usize = 80;
/// Relative residual at which a projection stops.
const CG_TOL: f64 = 1e-6;

/// Local coordinates of the 2×2×2 Gauss points on `[0, 1]³`, in corner order.
fn gauss_points() -> [DVec3; GP] {
    let a = 0.5 - 0.5 / 3f64.sqrt();
    let b = 0.5 + 0.5 / 3f64.sqrt();
    std::array::from_fn(|c| DVec3::new(if c & 1 == 0 { a } else { b }, if c & 2 == 0 { a } else { b }, if c & 4 == 0 { a } else { b }))
}

/// ∂N_c/∂t of the eight trilinear shape functions at `t`.
fn shape_gradients(t: DVec3) -> [DVec3; 8] {
    std::array::from_fn(|c| {
        let (sx, sy, sz) = (c & 1 != 0, c & 2 != 0, c & 4 != 0);
        let fx = if sx { t.x } else { 1.0 - t.x };
        let fy = if sy { t.y } else { 1.0 - t.y };
        let fz = if sz { t.z } else { 1.0 - t.z };
        let dx = if sx { 1.0 } else { -1.0 };
        let dy = if sy { 1.0 } else { -1.0 };
        let dz = if sz { 1.0 } else { -1.0 };
        DVec3::new(dx * fy * fz, fx * dy * fz, fx * fy * dz)
    })
}

/// Symmetric stress as a matrix.
#[inline]
fn mat(s: &[f64; 6]) -> DMat3 {
    DMat3::from_cols_array(&[s[0], s[3], s[5], s[3], s[1], s[4], s[5], s[4], s[2]])
}

#[inline]
fn voigt(m: &DMat3) -> [f64; 6] {
    [m.x_axis.x, m.y_axis.y, m.z_axis.z, 0.5 * (m.y_axis.x + m.x_axis.y), 0.5 * (m.z_axis.y + m.y_axis.z), 0.5 * (m.z_axis.x + m.x_axis.z)]
}

#[inline]
fn outer(a: DVec3, b: DVec3) -> DMat3 {
    DMat3::from_cols(a * b.x, a * b.y, a * b.z)
}

/// Result of a relaxation run.
#[derive(Clone, Copy, Debug, Default)]
pub struct Report {
    pub iterations: usize,
    /// Final max nodal residual (after the pressure) over max nodal weight.
    pub residual: f64,
    /// Largest equivalent plastic strain anywhere.
    pub max_plastic: f64,
    /// Smallest certified `det J` (relative to the undeformed element) over all elements.
    pub min_quality: f64,
    /// Largest matter-element volume error `|V/V₀ − 1|`.
    pub volume_error: f64,
    /// Largest node displacement from the undeformed lattice, blocks.
    pub max_displacement: f64,
    pub converged: bool,
}

/// Options of a relaxation run.
#[derive(Clone, Copy, Debug)]
pub struct Relax {
    pub max_iterations: usize,
    /// Stop when the residual falls below this.
    pub tolerance: f64,
    /// Fraction of the overstress returned per pseudo-step (1 = rate independent).
    pub plastic_relax: f64,
    /// Stop (unconverged) rather than let any element's certified quality fall below this.
    pub min_quality: f64,
    /// Print progress to stderr every this many iterations (0 = quiet).
    pub trace: usize,
}

impl Default for Relax {
    fn default() -> Self {
        Self { max_iterations: 20_000, tolerance: 1e-4, plastic_relax: 1.0, min_quality: 0.02, trace: 0 }
    }
}

/// The mechanical state of one body on its lattice.
pub struct Body {
    pub lattice: Lattice,
    /// Per element.
    pub params: Vec<Params>,
    /// Per Gauss point (`element · 8 + gp`): deviatoric Cauchy stress (xx, yy, zz, xy, yz, zx).
    pub stress: Vec<[f64; 6]>,
    /// Per Gauss point: equivalent plastic strain.
    pub plastic: Vec<f64>,
    /// Per element: the incompressibility multiplier (minus the element's pressure).
    lambda: Vec<f64>,
    /// Per element: matter (volume preserving) or void.
    constrained: Vec<bool>,
    /// Per element: reference volume.
    ref_volume: Vec<f64>,
    /// Per node.
    velocity: Vec<DVec3>,
    /// Per node: physical (lumped) mass.
    pub mass: Vec<f64>,
    /// Per node: held in place (tests: supports).
    pub fixed: Vec<bool>,
    /// A uniform external field added to self-gravity (tests).
    pub uniform_g: DVec3,
    pub self_gravity: bool,
    gravity: Vec<DVec3>,
    scaled: Vec<f64>,
    undeformed: Vec<DVec3>,
    threads: usize,
}

impl Body {
    /// A body on `lattice` with per-element `params` (density 0 = void).
    pub fn new(lattice: Lattice, params: Vec<Params>) -> Self {
        assert_eq!(params.len(), lattice.elements());
        let n = lattice.nodes.len();
        let e = lattice.elements();
        let mut mass = vec![0.0; n];
        let vol = (lattice.cell as f64).powi(3);
        for el in 0..e {
            let m = params[el].density * vol / 8.0;
            for node in lattice.element_nodes(el) {
                mass[node] += m;
            }
        }
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(16);
        let undeformed = lattice.nodes.clone();
        let constrained = params.iter().map(|p| p.density > 0.0).collect();
        let ref_volume = (0..e).map(|el| element_volume(&lattice.corners(el))).collect();
        Self {
            lattice,
            params,
            stress: vec![[0.0; 6]; e * GP],
            plastic: vec![0.0; e * GP],
            lambda: vec![0.0; e],
            constrained,
            ref_volume,
            velocity: vec![DVec3::ZERO; n],
            mass,
            fixed: vec![false; n],
            uniform_g: DVec3::ZERO,
            self_gravity: true,
            gravity: vec![DVec3::ZERO; n],
            scaled: vec![0.0; n],
            undeformed,
            threads,
        }
    }

    /// Process elements on this many threads (results are identical for any count).
    pub fn set_threads(&mut self, threads: usize) {
        self.threads = threads.max(1);
    }

    /// Total mass.
    pub fn total_mass(&self) -> f64 {
        self.mass.iter().sum()
    }

    /// Pressure of element `e` (positive in compression; zero for void).
    pub fn pressure(&self, e: usize) -> f64 {
        if self.constrained[e] { -self.lambda[e] } else { 0.0 }
    }

    /// Mean von Mises stress of element `e`.
    pub fn von_mises(&self, e: usize) -> f64 {
        let s = &self.stress[e * GP..(e + 1) * GP];
        s.iter().map(|v| equivalent(&mat(v))).sum::<f64>() / GP as f64
    }

    /// Refresh the gravitational field at every node.
    fn refresh_gravity(&mut self) {
        if !self.self_gravity {
            self.gravity.iter_mut().for_each(|g| *g = DVec3::ZERO);
            return;
        }
        let vol = (self.lattice.cell as f64).powi(3);
        let points: Vec<Mass> = (0..self.lattice.elements())
            .filter(|&e| self.params[e].density > 0.0)
            .map(|e| {
                let c = self.lattice.corners(e);
                Mass { at: c.iter().copied().sum::<DVec3>() / 8.0, mass: self.params[e].density * vol }
            })
            .collect();
        let tree = Tree::build(&points, 0.5 * self.lattice.cell as f64);
        let nodes = &self.lattice.nodes;
        let mut out = vec![DVec3::ZERO; nodes.len()];
        std::thread::scope(|scope| {
            let mut rest = out.as_mut_slice();
            for (from, to) in split_ranges(nodes.len(), self.threads) {
                let (mine, tail) = rest.split_at_mut(to - from);
                rest = tail;
                let tree = &tree;
                scope.spawn(move || {
                    for (i, g) in mine.iter_mut().enumerate() {
                        *g = tree.accel(nodes[from + i]);
                    }
                });
            }
        });
        self.gravity = out;
    }

    /// Recompute the fictitious masses for a unit pseudo-time step (shear stiffness for matter, the
    /// full wave modulus for void).
    fn rescale(&mut self) {
        let n = self.lattice.nodes.len();
        let mut scaled = vec![0.0f64; n];
        for e in 0..self.lattice.elements() {
            let c = self.lattice.corners(e);
            let mut h2 = f64::INFINITY;
            for (a, b) in EDGES {
                h2 = h2.min((c[a] - c[b]).length_squared());
            }
            let vol = element_volume(&c).abs().max(1e-30);
            let p = &self.params[e];
            let modulus = if self.constrained[e] { 4.0 / 3.0 * p.shear } else { p.wave_modulus() };
            let k = modulus * vol / h2.max(1e-30);
            for node in self.lattice.element_nodes(e) {
                scaled[node] += MASS_SAFETY * 4.0 * k;
            }
        }
        for (s, m) in scaled.iter_mut().zip(&self.mass) {
            *s = s.max(*m).max(1e-30);
        }
        self.scaled = scaled;
    }

    /// One element pass: advance the Gauss-point deviators by the current velocities (pseudo-step
    /// `dt = 1`); return per-element nodal internal forces and volume gradients `∂V/∂x_c`.
    fn element_pass(&mut self, plastic_relax: f64) -> (Vec<[DVec3; 8]>, Vec<[DVec3; 8]>) {
        let ne = self.lattice.elements();
        let mut forces = vec![[DVec3::ZERO; 8]; ne];
        let mut vgrads = vec![[DVec3::ZERO; 8]; ne];
        let lattice = &self.lattice;
        let params = &self.params;
        let velocity = &self.velocity;
        let cell = self.lattice.cell as f64;
        let gps = gauss_points();
        let grads: [[DVec3; 8]; GP] = std::array::from_fn(|g| shape_gradients(gps[g]));
        std::thread::scope(|scope| {
            let mut stress = self.stress.as_mut_slice();
            let mut plastic = self.plastic.as_mut_slice();
            let mut out = forces.as_mut_slice();
            let mut vg = vgrads.as_mut_slice();
            for (from, to) in split_ranges(ne, self.threads) {
                let (s_mine, s_rest) = stress.split_at_mut((to - from) * GP);
                let (p_mine, p_rest) = plastic.split_at_mut((to - from) * GP);
                let (f_mine, f_rest) = out.split_at_mut(to - from);
                let (g_mine, g_rest) = vg.split_at_mut(to - from);
                stress = s_rest;
                plastic = p_rest;
                out = f_rest;
                vg = g_rest;
                let grads = &grads;
                let gps = &gps;
                scope.spawn(move || {
                    for (k, e) in (from..to).enumerate() {
                        let nodes = lattice.element_nodes(e);
                        let x = nodes.map(|n| lattice.nodes[n]);
                        let v = nodes.map(|n| velocity[n]);
                        element(
                            &x,
                            &v,
                            &params[e],
                            gps,
                            grads,
                            &mut s_mine[k * GP..(k + 1) * GP],
                            &mut p_mine[k * GP..(k + 1) * GP],
                            &mut f_mine[k],
                            &mut g_mine[k],
                            plastic_relax,
                            cell,
                        );
                    }
                });
            }
        });
        (forces, vgrads)
    }

    /// Relax towards equilibrium.
    pub fn relax(&mut self, opts: &Relax) -> Report {
        let n = self.lattice.nodes.len();
        let ne = self.lattice.elements();
        let mut report = Report::default();
        let free: Vec<usize> = (0..n).filter(|&i| !self.fixed[i]).collect();
        let pinned = self.fixed.iter().any(|&f| f);
        let mut weight_scale = 1e-300f64;
        let mut prev_a: Option<Vec<DVec3>> = None;
        let mut damping = 1.0f64;
        for iter in 0..opts.max_iterations {
            if iter % GRAVITY_EVERY == 0 {
                self.refresh_gravity();
                self.rescale();
                weight_scale = self
                    .mass
                    .iter()
                    .zip(&self.gravity)
                    .map(|(m, g)| m * (*g + self.uniform_g).length())
                    .fold(1e-300f64, f64::max);
            }
            let (forces, vgrads) = self.element_pass(opts.plastic_relax);
            // Residual = external − internal, gathered in element order.
            let mut r: Vec<DVec3> = (0..n).map(|i| self.mass[i] * (self.gravity[i] + self.uniform_g)).collect();
            for (e, f) in forces.iter().enumerate() {
                for (c, node) in self.lattice.element_nodes(e).into_iter().enumerate() {
                    r[node] -= f[c];
                }
            }
            for i in 0..n {
                if self.fixed[i] {
                    r[i] = DVec3::ZERO;
                }
            }
            if !pinned {
                self.project_rigid(&mut r);
            }
            // Damped update coefficients (Underwood), then the incompressibility projection of the
            // acceleration so that the new velocity keeps each matter element's volume (and pulls a
            // drifted element back towards its reference volume).
            let alpha = if prev_a.is_none() { 0.0 } else { (2.0 - damping) / (2.0 + damping) };
            let beta = if prev_a.is_none() { 0.5 } else { 2.0 / (2.0 + damping) };
            let mut target = vec![0.0f64; ne];
            let mut worst_volume = 0.0f64;
            for e in 0..ne {
                if !self.constrained[e] {
                    continue;
                }
                let vol: f64 = element_volume(&self.lattice.corners(e));
                let err = vol - self.ref_volume[e];
                worst_volume = worst_volume.max((err / self.ref_volume[e]).abs());
                let current: f64 = self.lattice.element_nodes(e).iter().enumerate().map(|(c, &node)| vgrads[e][c].dot(self.velocity[node])).sum();
                // B a = (−drift − α B v) / β.
                target[e] = (-VOLUME_DRIFT * err - alpha * current) / beta;
            }
            self.project(&vgrads, &r, &target);
            // Pressure forces and the accelerations they leave.
            let mut f_p = vec![DVec3::ZERO; n];
            for e in 0..ne {
                if self.constrained[e] {
                    for (c, node) in self.lattice.element_nodes(e).into_iter().enumerate() {
                        f_p[node] += vgrads[e][c] * self.lambda[e];
                    }
                }
            }
            let mut a = vec![DVec3::ZERO; n];
            let mut res = 0.0f64;
            for &i in &free {
                let net = r[i] - f_p[i];
                res = res.max(net.length());
                a[i] = net / self.scaled[i];
            }
            let res = res / weight_scale;
            report.residual = res;
            report.iterations = iter + 1;
            report.volume_error = worst_volume;
            if opts.trace > 0 && (iter + 1) % opts.trace == 0 {
                let disp = self.lattice.nodes.iter().zip(&self.undeformed).map(|(p, q)| (*p - *q).length()).fold(0.0, f64::max);
                eprintln!(
                    "  iter {:>6}  residual {:.3e}  damping {:.3}  volume {:.2e}  max disp {:.3e}  quality {:.3}  plastic {:.3}",
                    iter + 1,
                    res,
                    damping,
                    worst_volume,
                    disp,
                    self.min_quality(),
                    self.plastic.iter().copied().fold(0.0, f64::max)
                );
            }
            if res < opts.tolerance && worst_volume < 1e-3 && iter > GRAVITY_EVERY {
                report.converged = true;
                break;
            }
            // Underwood: the lowest active frequency as a Rayleigh quotient along the last step.
            if let Some(prev) = &prev_a {
                if iter % GRAVITY_EVERY != 0 {
                    let (mut num, mut den) = (0.0f64, 0.0f64);
                    for &i in &free {
                        let dx = self.velocity[i];
                        num -= self.scaled[i] * dx.dot(a[i] - prev[i]);
                        den += self.scaled[i] * dx.length_squared();
                    }
                    if den > 0.0 {
                        damping = (2.0 * (num / den).max(0.0).sqrt()).min(1.9);
                    }
                }
            }
            for &i in &free {
                self.velocity[i] = self.velocity[i] * alpha + a[i] * beta;
                self.lattice.nodes[i] += self.velocity[i];
            }
            prev_a = Some(a);
            if iter % 64 == 63 {
                let q = self.min_quality();
                if q < opts.min_quality {
                    break;
                }
            }
        }
        self.velocity.iter_mut().for_each(|v| *v = DVec3::ZERO);
        self.lattice.reindex();
        report.max_plastic = self.plastic.iter().copied().fold(0.0, f64::max);
        report.min_quality = self.min_quality();
        report.max_displacement = self.lattice.nodes.iter().zip(&self.undeformed).map(|(p, q)| (*p - *q).length()).fold(0.0, f64::max);
        report
    }

    /// Solve `B M⁻¹ Bᵀ λ = B M⁻¹ r − target` for the element multipliers (preconditioned conjugate
    /// gradients, warm started from the last λ). `B` holds each matter element's `∂V/∂x_c`.
    fn project(&mut self, vgrads: &[[DVec3; 8]], r: &[DVec3], target: &[f64]) {
        let ne = self.lattice.elements();
        let n = self.lattice.nodes.len();
        let inv_m: Vec<f64> = (0..n).map(|i| if self.fixed[i] { 0.0 } else { 1.0 / self.scaled[i] }).collect();
        let elems: Vec<usize> = (0..ne).filter(|&e| self.constrained[e]).collect();
        if elems.is_empty() {
            return;
        }
        let nodes_of: Vec<[usize; 8]> = elems.iter().map(|&e| self.lattice.element_nodes(e)).collect();
        // y = B M⁻¹ Bᵀ x over the constrained elements (indexed by position in `elems`).
        let apply = |x: &[f64], y: &mut [f64], scratch: &mut [DVec3]| {
            scratch.iter_mut().for_each(|v| *v = DVec3::ZERO);
            for (k, &e) in elems.iter().enumerate() {
                for c in 0..8 {
                    scratch[nodes_of[k][c]] += vgrads[e][c] * x[k];
                }
            }
            for (k, &e) in elems.iter().enumerate() {
                let mut s = 0.0;
                for c in 0..8 {
                    let node = nodes_of[k][c];
                    s += vgrads[e][c].dot(scratch[node]) * inv_m[node];
                }
                y[k] = s;
            }
        };
        let m = elems.len();
        let mut scratch = vec![DVec3::ZERO; n];
        let mut rhs = vec![0.0f64; m];
        for (k, &e) in elems.iter().enumerate() {
            let mut s = 0.0;
            for c in 0..8 {
                let node = nodes_of[k][c];
                s += vgrads[e][c].dot(r[node]) * inv_m[node];
            }
            rhs[k] = s - target[e];
        }
        let diag: Vec<f64> = elems
            .iter()
            .enumerate()
            .map(|(k, &e)| (0..8).map(|c| vgrads[e][c].length_squared() * inv_m[nodes_of[k][c]]).sum::<f64>().max(1e-300))
            .collect();
        let mut x: Vec<f64> = elems.iter().map(|&e| self.lambda[e]).collect();
        let mut ax = vec![0.0f64; m];
        apply(&x, &mut ax, &mut scratch);
        let mut res: Vec<f64> = rhs.iter().zip(&ax).map(|(b, a)| b - a).collect();
        let norm_b = rhs.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-300);
        let mut z: Vec<f64> = res.iter().zip(&diag).map(|(r, d)| r / d).collect();
        let mut p = z.clone();
        let mut rz: f64 = res.iter().zip(&z).map(|(a, b)| a * b).sum();
        let mut ap = vec![0.0f64; m];
        for _ in 0..CG_MAX {
            let rn = res.iter().map(|v| v * v).sum::<f64>().sqrt();
            if rn <= CG_TOL * norm_b {
                break;
            }
            apply(&p, &mut ap, &mut scratch);
            let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
            if pap.abs() <= 1e-300 {
                break;
            }
            let step = rz / pap;
            for k in 0..m {
                x[k] += step * p[k];
                res[k] -= step * ap[k];
            }
            for k in 0..m {
                z[k] = res[k] / diag[k];
            }
            let rz_new: f64 = res.iter().zip(&z).map(|(a, b)| a * b).sum();
            let beta = rz_new / rz.max(1e-300);
            rz = rz_new;
            for k in 0..m {
                p[k] = z[k] + beta * p[k];
            }
        }
        for (k, &e) in elems.iter().enumerate() {
            self.lambda[e] = x[k];
        }
    }

    /// Smallest certified relative `det J` over all elements.
    pub fn min_quality(&self) -> f64 {
        (0..self.lattice.elements()).map(|e| self.lattice.certify(e)).fold(f64::INFINITY, f64::min)
    }

    /// Remove the net force and torque of the residual (with respect to the fictitious masses), so
    /// an unsupported body neither drifts nor spins.
    fn project_rigid(&self, r: &mut [DVec3]) {
        let total: f64 = self.scaled.iter().sum();
        let com = self.lattice.nodes.iter().zip(&self.scaled).map(|(x, m)| *x * *m).sum::<DVec3>() / total;
        let net = r.iter().copied().sum::<DVec3>();
        let a = net / total;
        let mut torque = DVec3::ZERO;
        let mut inertia = DMat3::ZERO;
        for (i, x) in self.lattice.nodes.iter().enumerate() {
            let d = *x - com;
            torque += d.cross(r[i] - self.scaled[i] * a);
            inertia += (DMat3::IDENTITY * d.length_squared() - outer(d, d)) * self.scaled[i];
        }
        let alpha = if inertia.determinant().abs() > 1e-300 { inertia.inverse() * torque } else { DVec3::ZERO };
        for (i, x) in self.lattice.nodes.iter().enumerate() {
            let d = *x - com;
            r[i] -= self.scaled[i] * (a + alpha.cross(d));
        }
    }
}

const EDGES: [(usize, usize); 12] = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)];

/// Volume of a trilinear element (2×2×2 Gauss quadrature of `det J`, exact for the trilinear map).
fn element_volume(x: &[DVec3; 8]) -> f64 {
    gauss_points().iter().map(|&t| local_jacobian(x, t).determinant()).sum::<f64>() / GP as f64
}

/// von Mises equivalent of a stress matrix.
#[inline]
fn equivalent(s: &DMat3) -> f64 {
    let p = (s.x_axis.x + s.y_axis.y + s.z_axis.z) / 3.0;
    let d = *s - DMat3::IDENTITY * p;
    let dd = d.x_axis.length_squared() + d.y_axis.length_squared() + d.z_axis.length_squared();
    (1.5 * dd).sqrt()
}

/// One element: velocity gradients, the constitutive update at every Gauss point, the nodal
/// internal forces `∫ s ∇N dV` (matter: the deviator; the pressure comes from the projection) and
/// the volume gradients `∫ ∇N dV`.
#[allow(clippy::too_many_arguments)]
fn element(
    x: &[DVec3; 8],
    v: &[DVec3; 8],
    p: &Params,
    gps: &[DVec3; GP],
    grads: &[[DVec3; 8]; GP],
    stress: &mut [[f64; 6]],
    plastic: &mut [f64],
    out: &mut [DVec3; 8],
    volume_grad: &mut [DVec3; 8],
    relax: f64,
    cell: f64,
) {
    let mut b = [[DVec3::ZERO; 8]; GP];
    let mut w = [0.0f64; GP];
    let mut l = [DMat3::ZERO; GP];
    for g in 0..GP {
        let j = local_jacobian(x, gps[g]);
        let det = j.determinant();
        let jit = if det.abs() > 1e-300 { j.inverse().transpose() } else { DMat3::ZERO };
        w[g] = det / GP as f64;
        let mut lg = DMat3::ZERO;
        for c in 0..8 {
            b[g][c] = jit * grads[g][c];
            lg += outer(v[c], b[g][c]);
        }
        l[g] = lg;
    }
    *volume_grad = std::array::from_fn(|c| (0..GP).map(|g| b[g][c] * w[g]).sum());
    *out = [DVec3::ZERO; 8];
    let mu = p.shear;
    if p.density <= 0.0 {
        // Void carries no history: a total-Lagrangian compressible neo-Hookean skin over the
        // undeformed cube, whose ln J term grows without bound as the element nears collapse, so the
        // air the matter drags along stays a valid map.
        let lambda = p.bulk - 2.0 / 3.0 * mu;
        for g in 0..GP {
            let f = local_jacobian(x, gps[g]) * (1.0 / cell);
            let jdet = f.determinant().max(1e-9);
            let bmat = f * f.transpose();
            let s = (bmat - DMat3::IDENTITY) * (mu / jdet) + DMat3::IDENTITY * (lambda * jdet.ln() / jdet);
            stress[g] = voigt(&s);
            for c in 0..8 {
                out[c] += (s * b[g][c]) * w[g];
            }
        }
        return;
    }
    for g in 0..GP {
        let lg = l[g];
        let d = (lg + lg.transpose()) * 0.5;
        let wspin = (lg - lg.transpose()) * 0.5;
        let trd = d.x_axis.x + d.y_axis.y + d.z_axis.z;
        let ddev = d - DMat3::IDENTITY * (trd / 3.0);
        let mut dev = mat(&stress[g]);
        dev += ddev * (2.0 * mu) + wspin * dev - dev * wspin;
        // Keep it a deviator (the spin terms preserve the trace only to rounding).
        let tr = dev.x_axis.x + dev.y_axis.y + dev.z_axis.z;
        dev -= DMat3::IDENTITY * (tr / 3.0);
        // Radial return.
        let dd = dev.x_axis.length_squared() + dev.y_axis.length_squared() + dev.z_axis.length_squared();
        let eq = (1.5 * dd).sqrt();
        if eq > p.yield_stress {
            let target = eq - relax * (eq - p.yield_stress);
            dev *= target / eq;
            plastic[g] += (eq - target) / (3.0 * mu.max(1e-300));
        }
        stress[g] = voigt(&dev);
        for c in 0..8 {
            out[c] += (dev * b[g][c]) * w[g];
        }
    }
}

/// Split `n` items into at most `parts` contiguous, near-equal ranges.
fn split_ranges(n: usize, parts: usize) -> Vec<(usize, usize)> {
    let parts = parts.clamp(1, n.max(1));
    let base = n / parts;
    let extra = n % parts;
    let mut out = Vec::with_capacity(parts);
    let mut at = 0;
    for p in 0..parts {
        let len = base + usize::from(p < extra);
        out.push((at, at + len));
        at += len;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(dims: [usize; 3], cell: i64) -> Lattice {
        Lattice::undeformed([0, 0, 0], cell, dims, DVec3::ZERO)
    }

    /// A cantilever under a uniform field, clamped at one end: the tip deflection matches
    /// Euler–Bernoulli `w = q L⁴ / (8 E I)` with the incompressible `E = 3μ`, within the coarse
    /// discretisation's error.
    #[test]
    fn a_cantilever_bends_like_a_beam() {
        let (nx, ny, nz, h) = (16usize, 2usize, 2usize, 16i64);
        let l = block([nx, ny, nz], h);
        let p = Params { density: 1.0, shear: 4.0e6, bulk: f64::INFINITY, yield_stress: f64::INFINITY, creep_time: 1.0 };
        let mut body = Body::new(l, vec![p; nx * ny * nz]);
        body.self_gravity = false;
        body.uniform_g = DVec3::new(0.0, -10.0, 0.0);
        for k in 0..=nz {
            for j in 0..=ny {
                let n = body.lattice.node(0, j, k);
                body.fixed[n] = true;
            }
        }
        let rep = body.relax(&Relax { max_iterations: 40_000, tolerance: 1e-6, ..Relax::default() });
        assert!(rep.converged, "{rep:?}");
        let (len, height, width) = ((nx as i64 * h) as f64, (ny as i64 * h) as f64, (nz as i64 * h) as f64);
        let e_mod = 3.0 * p.shear;
        let inertia = width * height.powi(3) / 12.0;
        let q = p.density * 10.0 * width * height;
        let want = q * len.powi(4) / (8.0 * e_mod * inertia);
        let tip = body.lattice.nodes[body.lattice.node(nx, ny / 2, nz / 2)];
        let got = -(tip.y - (ny as f64 / 2.0) * h as f64);
        // Trilinear hexes are stiff in bending at two elements through the depth: accept 40%.
        assert!(got > 0.0 && (got - want).abs() / want < 0.4, "tip {got} vs beam theory {want} ({rep:?})");
        assert!(rep.volume_error < 1e-3, "{rep:?}");
    }

    /// Strength decides who rounds: a strong cube keeps its corners, a weak one flows towards a ball.
    #[test]
    fn a_strong_cube_keeps_its_corners_and_a_weak_one_rounds() {
        let n = 6usize;
        let cell = 1i64 << 16;
        let make = |yield_stress: f64| {
            let l = block([n, n, n], cell);
            let matter = Params::from_yield(5.0, yield_stress);
            let mut params = vec![matter; n * n * n];
            for (e, p) in params.iter_mut().enumerate() {
                let [i, j, k] = [e % n, (e / n) % n, e / (n * n)];
                if [i, j, k].iter().any(|&q| q == 0 || q == n - 1) {
                    *p = Params::void(&matter);
                }
            }
            Body::new(l, params)
        };
        let shape = |b: &Body| {
            let c = b.lattice.nodes[b.lattice.node(3, 3, 3)];
            let corner = (b.lattice.nodes[b.lattice.node(5, 5, 5)] - c).length();
            let face = (b.lattice.nodes[b.lattice.node(5, 3, 3)] - c).length();
            corner / face
        };
        let mut strong = make(1e12);
        let rs = strong.relax(&Relax { max_iterations: 4_000, ..Relax::default() });
        assert!(rs.max_plastic == 0.0, "{rs:?}");
        assert!((shape(&strong) - 3f64.sqrt()).abs() < 0.01, "a strong cube keeps its corners: {}", shape(&strong));
        let mut weak = make(1e3);
        let rw = weak.relax(&Relax { max_iterations: 8_000, ..Relax::default() });
        assert!(rw.max_plastic > 0.05, "{rw:?}");
        assert!(rw.min_quality > 0.0, "{rw:?}");
        assert!(rw.volume_error < 0.01, "{rw:?}");
        assert!(shape(&weak) < 3f64.sqrt() - 0.1, "a weak cube rounds: {} ({rw:?})", shape(&weak));
    }

    #[test]
    fn relaxation_is_independent_of_the_thread_count() {
        let l = block([4, 4, 4], 1 << 14);
        let params = vec![Params::from_yield(5.0, 1e3); 64];
        let run = |threads: usize| {
            let mut b = Body::new(l.clone(), params.clone());
            b.set_threads(threads);
            b.relax(&Relax { max_iterations: 300, ..Relax::default() });
            b.lattice.nodes.iter().flat_map(|n| n.to_array()).map(f64::to_bits).collect::<Vec<_>>()
        };
        assert_eq!(run(1), run(5));
    }

    #[test]
    fn mass_is_the_reference_mass() {
        let l = block([3, 3, 3], 16);
        let b = Body::new(l, vec![Params::from_yield(4.0, 1e5); 27]);
        assert!((b.total_mass() - 4.0 * (48.0f64).powi(3)).abs() < 1e-6);
    }
}
