//! Exact Newtonian field of a uniform-density closed polyhedron with triangular faces
//! (Werner & Scheeres 1997): a solid-angle term per face and a line-integral term per edge,
//! valid inside and outside. Face normals and edge dyads are built once; a sample walks them
//! and allocates nothing. Far from the body the sum cancels, so beyond [`FAR_RATIO`] bounding
//! radii the monopole and quadrupole take over.

use std::collections::HashMap;

use glam::{DMat3, DVec3};

use super::kernel::{self, R_G, R_IN};
use super::shape::{self, Primitive, Shape};

/// Beyond this many bounding radii the face/edge sum loses digits and the monopole plus
/// quadrupole replaces it (the error of that truncation is of order `(R/d)⁴`).
const FAR_RATIO: f64 = 50.0;
/// A body this small against its distance, while the range window cuts it, is one tapered point.
const SPLIT_RATIO: f64 = 0.02;
/// Quads along one edge of each face when a warped cube is meshed: every other node of the warp's
/// 16-element lattice. A twin-face sample costs about 57 µs (16 quads: 213 µs, 64: 3.6 ms) and
/// differs from the 16-quad mesh by 0.05 %.
pub const FACE_QUADS: usize = 8;
/// Werner summation inside this many bounding radii of a shape standing in for a cube.
/// A face sample walks both twin meshes and costs milliseconds, so farther out the cube's closed
/// form keeps the higher multipoles and the shape adds only its monopole and quadrupole difference.
/// The other twin lies inside this ratio of a surface sample, so the ground stays on the polyhedron.
pub const EXACT_RATIO: f64 = 3.0;

/// One uniform polyhedron.
pub struct Polyhedron {
    density: f64,
    /// Centre of mass.
    com: DVec3,
    /// Bounding radius about `com`.
    radius: f64,
    mass: f64,
    /// Traceless quadrupole about `com`: `Q_ij = ∫ (3 y_i y_j − δ_ij |y|²) ρ dV`.
    q: DMat3,
    lo: DVec3,
    hi: DVec3,
    faces: Vec<Face>,
    edges: Vec<Edge>,
}

struct Face {
    /// Outward unit normal.
    n: DVec3,
    v0: DVec3,
    v1: DVec3,
    v2: DVec3,
}

struct Edge {
    a: DVec3,
    b: DVec3,
    len: f64,
    /// The two faces' outward normals and their in-plane outward edge normals.
    n1: DVec3,
    ne1: DVec3,
    n2: DVec3,
    ne2: DVec3,
}

impl std::fmt::Debug for Polyhedron {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Polyhedron")
            .field("faces", &self.faces.len())
            .field("edges", &self.edges.len())
            .field("mass", &self.mass)
            .field("com", &self.com)
            .field("radius", &self.radius)
            .finish()
    }
}

impl Polyhedron {
    /// The surface of the axis-aligned box `[lo, hi]`, two triangles per face.
    pub fn from_box(lo: DVec3, hi: DVec3, density: f64) -> Self {
        Self::from_surface(lo, hi, 1, density, |p| p)
    }

    /// The box `[lo, hi]` with `n` quads along each edge (two triangles each), every surface
    /// point sent through `map`. `map` keeps orientation: outward stays outward.
    pub fn from_surface(lo: DVec3, hi: DVec3, n: usize, density: f64, map: impl Fn(DVec3) -> DVec3) -> Self {
        let n = n.max(1);
        let mut index_of: HashMap<(u64, u64, u64), u32> = HashMap::new();
        let mut verts: Vec<DVec3> = Vec::new();
        let mut tris: Vec<[u32; 3]> = Vec::with_capacity(12 * n * n);
        let mut put = |p: DVec3| -> u32 {
            let key = (p.x.to_bits(), p.y.to_bits(), p.z.to_bits());
            if let Some(&i) = index_of.get(&key) {
                return i;
            }
            let i = verts.len() as u32;
            verts.push(p);
            index_of.insert(key, i);
            i
        };
        // (fixed axis, u, v) with u × v pointing to the +fixed side. The sign chooses the side.
        let sides = [
            (0usize, 1usize, 2usize, 1.0f64),
            (0, 2, 1, -1.0),
            (1, 2, 0, 1.0),
            (1, 0, 2, -1.0),
            (2, 0, 1, 1.0),
            (2, 1, 0, -1.0),
        ];
        for (fixed, u, v, sign) in sides {
            let mut grid = vec![0u32; (n + 1) * (n + 1)];
            for j in 0..=n {
                for i in 0..=n {
                    let mut p = DVec3::ZERO;
                    p[fixed] = if sign > 0.0 { hi[fixed] } else { lo[fixed] };
                    p[u] = lo[u] + (hi[u] - lo[u]) * (i as f64 / n as f64);
                    p[v] = lo[v] + (hi[v] - lo[v]) * (j as f64 / n as f64);
                    grid[i + j * (n + 1)] = put(map(p));
                }
            }
            let at = |i: usize, j: usize| grid[i + j * (n + 1)];
            for j in 0..n {
                for i in 0..n {
                    let (a, b, c, d) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
                    tris.push([a, b, c]);
                    tris.push([a, c, d]);
                }
            }
        }
        if signed_volume(&verts, &tris) < 0.0 {
            for t in &mut tris {
                t.swap(1, 2);
            }
        }
        Self::assemble(verts, tris, density)
    }

    fn assemble(verts: Vec<DVec3>, tris: Vec<[u32; 3]>, density: f64) -> Self {
        let mut faces = Vec::with_capacity(tris.len());
        let mut half: HashMap<(u32, u32), (Option<Half>, Option<Half>)> = HashMap::with_capacity(tris.len() * 3 / 2);
        for t in &tris {
            let (v0, v1, v2) = (verts[t[0] as usize], verts[t[1] as usize], verts[t[2] as usize]);
            let cr = (v1 - v0).cross(v2 - v0);
            let area = cr.length();
            if area == 0.0 || !area.is_finite() {
                continue;
            }
            let n = cr / area;
            let ids = [t[0], t[1], t[2]];
            for k in 0..3 {
                let (i, j) = (ids[k], ids[(k + 1) % 3]);
                let dir = verts[j as usize] - verts[i as usize];
                let ne = dir.cross(n);
                let nel = ne.length();
                if nel == 0.0 {
                    continue;
                }
                let ne = ne / nel;
                let key = if i < j { (i, j) } else { (j, i) };
                let slot = half.entry(key).or_insert((None, None));
                let h = Half { n, ne };
                if slot.0.is_none() {
                    slot.0 = Some(h);
                } else {
                    slot.1 = Some(h);
                }
            }
            faces.push(Face { n, v0, v1, v2 });
        }
        let mut edges = Vec::with_capacity(half.len());
        for ((i, j), (h0, h1)) in half {
            let (Some(h0), Some(h1)) = (h0, h1) else { continue };
            let (a, b) = (verts[i as usize], verts[j as usize]);
            let len = (b - a).length();
            if len == 0.0 || !len.is_finite() {
                continue;
            }
            edges.push(Edge { a, b, len, n1: h0.n, ne1: h0.ne, n2: h1.n, ne2: h1.ne });
        }
        let (com, radius, mass, q, lo, hi) = moments(&verts, &tris, density);
        Self { density, com, radius, mass, q, lo, hi, faces, edges }
    }

    /// Total mass (amount).
    pub fn mass(&self) -> f64 {
        self.mass
    }

    /// Centre of mass.
    pub fn centre(&self) -> DVec3 {
        self.com
    }

    /// Bounding radius about the centre of mass.
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Acceleration and potential at `p`, per unit `G`.
    pub fn field(&self, p: DVec3) -> (DVec3, f64) {
        if self.faces.is_empty() || self.mass == 0.0 {
            return (DVec3::ZERO, 0.0);
        }
        let d = (p - self.com).length();
        if d - self.radius >= R_G {
            return (DVec3::ZERO, 0.0);
        }
        if d + self.radius <= R_IN {
            if d > FAR_RATIO * self.radius {
                let (a, phi) = self.multipole(p);
                return (a, phi - self.mass * kernel::offset());
            }
            let (a, phi) = self.werner(p);
            return (a, phi - self.mass * kernel::offset());
        }
        // The range window cuts the body. A surface sample sees the whole body inside `R_IN`;
        // only a query near `R_G` meets the window, so the equal-mass bounding box, split like
        // any box, carries the taper. A twin uses `field_over_cube`, which keeps its reference cube.
        if self.radius <= SPLIT_RATIO * d.max(1.0) {
            return kernel::point(self.com, self.mass, p);
        }
        let vol = (self.hi - self.lo).x * (self.hi - self.lo).y * (self.hi - self.lo).z;
        let density = if vol > 0.0 { self.mass / vol } else { 0.0 };
        Primitive::new(Shape::Box { lo: self.lo, hi: self.hi }, density).field(p)
    }

    /// [`field`](Self::field) of a shape that sags from the cube `[lo, hi]` of `density`.
    /// Inside [`EXACT_RATIO`] bounding radii (and inside `R_IN`) this is the polyhedron.
    /// Farther, but still inside `R_IN`, it is that cube's closed form plus the monopole and
    /// quadrupole difference. In the range window the cube is split as a box and the mass
    /// difference is a tapered point: a twin's window queries are ≥ 14 bounding radii out.
    pub fn field_over_cube(&self, lo: DVec3, hi: DVec3, density: f64, p: DVec3) -> (DVec3, f64) {
        let d = (p - self.com).length();
        if self.mass == 0.0 || self.faces.is_empty() || d - self.radius >= R_G {
            return (DVec3::ZERO, 0.0);
        }
        let inside = d + self.radius <= R_IN;
        if inside && d <= EXACT_RATIO * self.radius {
            return self.field(p);
        }
        let cube = Primitive::new(Shape::Box { lo, hi }, density);
        if !inside {
            let (ba, bp) = cube.field(p);
            let dm = self.mass - cube.mass();
            if dm == 0.0 {
                return (ba, bp);
            }
            let (da, dp) = kernel::point(self.com, dm, p);
            return (ba + da, bp + dp);
        }
        if d > FAR_RATIO * self.radius {
            let (a, phi) = self.multipole(p);
            return (a, phi - self.mass * kernel::offset());
        }
        let (ba, bp) = cube.field(p);
        let (da, dp) = self.multipole_minus_cube(lo, hi, cube.mass(), (lo + hi) * 0.5, p);
        (ba + da, bp + dp)
    }

    /// Newtonian monopole + quadrupole, potential zero at infinity (no kernel offset).
    fn multipole(&self, p: DVec3) -> (DVec3, f64) {
        let x = p - self.com;
        let r2 = x.length_squared();
        if r2 == 0.0 {
            return (DVec3::ZERO, 0.0);
        }
        let r = r2.sqrt();
        let inv = 1.0 / r;
        let inv3 = inv / r2;
        let inv5 = inv3 / r2;
        let mono_a = (self.com - p) * (self.mass * inv3);
        let mono_p = -self.mass * inv;
        let qx = self.q * x;
        let xqx = x.dot(qx);
        let quad_a = qx * inv5 - x * (2.5 * xqx * inv5 / r2);
        let quad_p = -0.5 * xqx * inv5;
        (mono_a + quad_a, mono_p + quad_p)
    }

    /// This shape's monopole and quadrupole minus the cube's, with the offset of the mass
    /// difference so it can be added to the cube primitive's field.
    fn multipole_minus_cube(&self, lo: DVec3, hi: DVec3, cube_mass: f64, cube_com: DVec3, p: DVec3) -> (DVec3, f64) {
        let (pa, pp) = self.multipole(p);
        let x = p - cube_com;
        let r2 = x.length_squared();
        let r = r2.sqrt();
        let inv = 1.0 / r;
        let ca = (cube_com - p) * (cube_mass * inv / r2);
        let cp = -cube_mass * inv;
        let (qa, qp) = shape::box_quadrupole(lo, hi, cube_mass, p);
        let dm = self.mass - cube_mass;
        (pa - ca - qa, pp - cp - qp - dm * kernel::offset())
    }

    /// Face and edge sum, per unit `G`, potential zero at infinity.
    fn werner(&self, p: DVec3) -> (DVec3, f64) {
        let (mut accel, mut pot) = (DVec3::ZERO, 0.0);
        for f in &self.faces {
            let r1 = f.v0 - p;
            let r2 = f.v1 - p;
            let r3 = f.v2 - p;
            let height = f.n.dot(r1);
            let w = solid_angle(r1, r2, r3);
            if !w.is_finite() || !height.is_finite() {
                continue;
            }
            accel += f.n * (height * w);
            pot -= 0.5 * height * height * w;
        }
        for e in &self.edges {
            let r1 = e.a - p;
            let r2 = e.b - p;
            let a = r1.length();
            let b = r2.length();
            if a == 0.0 || b == 0.0 {
                continue;
            }
            // `L = ln((a+b+e)/(a+b−e))`, written as `ln(1 + e(a+b+e)/(ab + r1·r2))` so a far
            // edge does not subtract two huge logs.
            let prod = a * b + r1.dot(r2);
            if prod <= 0.0 {
                continue;
            }
            let l = (e.len * (a + b + e.len) / prod).ln_1p();
            if !l.is_finite() {
                continue;
            }
            let er = e.n1 * e.ne1.dot(r1) + e.n2 * e.ne2.dot(r1);
            let rer = e.ne1.dot(r1) * e.n1.dot(r1) + e.ne2.dot(r1) * e.n2.dot(r1);
            accel -= er * l;
            pot += 0.5 * rer * l;
        }
        // `pot` is the negation of the Nagy potential (zero at infinity); the acceleration
        // above already points toward the mass, matching `−∇Φ`.
        (accel * self.density, -pot * self.density)
    }
}

struct Half {
    n: DVec3,
    ne: DVec3,
}

/// Signed solid angle of the triangle `(r1, r2, r3)` as seen from the origin.
fn solid_angle(r1: DVec3, r2: DVec3, r3: DVec3) -> f64 {
    let n1 = r1.length();
    let n2 = r2.length();
    let n3 = r3.length();
    if n1 == 0.0 || n2 == 0.0 || n3 == 0.0 {
        return 0.0;
    }
    let num = r1.dot(r2.cross(r3));
    let den = n1 * n2 * n3 + r1.dot(r2) * n3 + r2.dot(r3) * n1 + r3.dot(r1) * n2;
    2.0 * num.atan2(den)
}

fn signed_volume(verts: &[DVec3], tris: &[[u32; 3]]) -> f64 {
    if verts.is_empty() {
        return 0.0;
    }
    let o = verts[0];
    let mut det = 0.0;
    for t in tris {
        let a = verts[t[0] as usize] - o;
        let b = verts[t[1] as usize] - o;
        let c = verts[t[2] as usize] - o;
        det += a.dot(b.cross(c));
    }
    det / 6.0
}

/// `(a ⊗ b)` as a matrix, `(a ⊗ b) v = a (b · v)`.
fn outer(a: DVec3, b: DVec3) -> DMat3 {
    DMat3::from_cols(a * b.x, a * b.y, a * b.z)
}

/// Volume moments about the centre of mass. Vertices may sit far from the origin; the sums run
/// about the box centre so they do not cancel leading digits.
fn moments(verts: &[DVec3], tris: &[[u32; 3]], density: f64) -> (DVec3, f64, f64, DMat3, DVec3, DVec3) {
    let (mut lo, mut hi) = (DVec3::splat(f64::MAX), DVec3::splat(f64::MIN));
    for v in verts {
        lo = lo.min(*v);
        hi = hi.max(*v);
    }
    if verts.is_empty() {
        return (DVec3::ZERO, 0.0, 0.0, DMat3::ZERO, DVec3::ZERO, DVec3::ZERO);
    }
    let o = (lo + hi) * 0.5;
    let (mut det_sum, mut com_acc, mut second) = (0.0, DVec3::ZERO, DMat3::ZERO);
    for t in tris {
        let a = verts[t[0] as usize] - o;
        let b = verts[t[1] as usize] - o;
        let c = verts[t[2] as usize] - o;
        let det = a.dot(b.cross(c));
        det_sum += det;
        com_acc += (a + b + c) * det;
        let mut mm = (outer(a, a) + outer(b, b) + outer(c, c)) * 2.0;
        mm += outer(a, b) + outer(b, a);
        mm += outer(a, c) + outer(c, a);
        mm += outer(b, c) + outer(c, b);
        second += mm * det;
    }
    if det_sum == 0.0 {
        return (o, 0.0, 0.0, DMat3::ZERO, lo, hi);
    }
    let volume = det_sum / 6.0;
    let com_rel = com_acc / (4.0 * det_sum);
    let com = o + com_rel;
    // `∫ y_i y_j dV` about the centre of mass.
    let about = second / 120.0 - outer(com_rel, com_rel) * volume;
    let mass_moment = about * density;
    let trace = mass_moment.x_axis.x + mass_moment.y_axis.y + mass_moment.z_axis.z;
    let q = (mass_moment * 3.0 - DMat3::IDENTITY * trace).let_sym();
    let mass = volume * density;
    let mut r2 = 0.0f64;
    for v in verts {
        r2 = r2.max((*v - com).length_squared());
    }
    (com, r2.sqrt(), mass, q, lo, hi)
}

trait Sym {
    fn let_sym(self) -> Self;
}

impl Sym for DMat3 {
    fn let_sym(self) -> Self {
        (self + self.transpose()) * 0.5
    }
}

/// Largest distance from the mapped box surface to its triangle mesh, sampled inside each quad.
#[cfg(test)]
fn surface_chord(lo: DVec3, hi: DVec3, n: usize, map: impl Fn(DVec3) -> DVec3) -> f64 {
    let n = n.max(1);
    let sides = [
        (0usize, 1usize, 2usize, 1.0f64),
        (0, 2, 1, -1.0),
        (1, 2, 0, 1.0),
        (1, 0, 2, -1.0),
        (2, 0, 1, 1.0),
        (2, 1, 0, -1.0),
    ];
    let samples = [(0.5, 0.25), (0.5, 0.5), (0.25, 0.5), (0.75, 0.25), (0.25, 0.75), (0.5, 0.75)];
    let mut worst = 0.0f64;
    for (fixed, u, v, sign) in sides {
        for j in 0..n {
            for i in 0..n {
                let corner = |ii: usize, jj: usize| {
                    let mut p = DVec3::ZERO;
                    p[fixed] = if sign > 0.0 { hi[fixed] } else { lo[fixed] };
                    p[u] = lo[u] + (hi[u] - lo[u]) * (ii as f64 / n as f64);
                    p[v] = lo[v] + (hi[v] - lo[v]) * (jj as f64 / n as f64);
                    p
                };
                let (r00, r10, r11, r01) = (corner(i, j), corner(i + 1, j), corner(i + 1, j + 1), corner(i, j + 1));
                let (c00, c10, c11, c01) = (map(r00), map(r10), map(r11), map(r01));
                for (s, t) in samples {
                    let mut ref_p = DVec3::ZERO;
                    ref_p[fixed] = r00[fixed];
                    ref_p[u] = r00[u] + (r10[u] - r00[u]) * s;
                    ref_p[v] = r00[v] + (r01[v] - r00[v]) * t;
                    let true_p = map(ref_p);
                    let mesh = if s >= t {
                        c00 * (1.0 - s) + c10 * (s - t) + c11 * t
                    } else {
                        c00 * (1.0 - t) + c11 * s + c01 * (t - s)
                    };
                    worst = worst.max((true_p - mesh).length());
                }
            }
        }
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(got: DVec3, want: DVec3) -> f64 {
        (got - want).length() / want.length().max(1e-30)
    }

    #[test]
    fn a_cube_polyhedron_matches_the_nagy_box_inside_and_outside() {
        let (lo, hi) = (DVec3::splat(-1.0), DVec3::splat(1.0));
        let poly = Polyhedron::from_box(lo, hi, 2.5);
        assert_eq!(poly.faces.len(), 12, "two triangles on each of six faces");
        assert_eq!(poly.edges.len(), 18, "twelve edges plus six face diagonals");
        let volume = 8.0;
        assert!((poly.mass() - 2.5 * volume).abs() <= 1e-9 * poly.mass(), "mass {}", poly.mass());
        assert!(poly.centre().length() < 1e-12, "centre {}", poly.centre());
        // A cube's quadrupole vanishes.
        let qmax = poly.q.x_axis.abs().max_element().max(poly.q.y_axis.abs().max_element()).max(poly.q.z_axis.abs().max_element());
        assert!(qmax < 1e-9, "Q {}", poly.q);
        let prim = Primitive::new(Shape::Box { lo, hi }, 2.5);
        let points = [
            DVec3::new(0.0, 3.0, 0.0),
            DVec3::new(2.0, -1.5, 0.7),
            DVec3::new(0.2, -0.4, 0.15),
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(-0.3, 1.4, 0.5),
            DVec3::new(4.0, 4.0, -2.0),
        ];
        for p in points {
            let (a, phi) = poly.field(p);
            let (b, bphi) = prim.field(p);
            assert!(rel(a, b) <= 1e-9 || (a - b).length() <= 1e-9, "{p}: {a} vs {b}");
            let scale = bphi.abs().max(1e-30);
            assert!((phi - bphi).abs() <= 1e-9 * scale, "{p}: {phi} vs {bphi}");
        }
        // A rectangular box exercises a non-zero quadrupole and off-centre coordinates.
        let (lo, hi) = (DVec3::new(-2.0, 1.0, 4.0), DVec3::new(2.0, 3.0, 5.0));
        let poly = Polyhedron::from_box(lo, hi, 3.0);
        let want_m = 4.0 * 2.0 * 1.0 * 3.0;
        assert!((poly.mass() - want_m).abs() <= 1e-9 * want_m, "mass {}", poly.mass());
        let prim = Primitive::new(Shape::Box { lo, hi }, 3.0);
        for p in [DVec3::new(0.0, 6.0, 4.5), DVec3::new(-1.0, 2.0, 4.3), DVec3::new(5.0, 0.0, 9.0), DVec3::new(0.5, 1.5, 4.2)] {
            let (a, phi) = poly.field(p);
            let (b, bphi) = prim.field(p);
            assert!(rel(a, b) <= 1e-9 || (a - b).length() <= 1e-9, "{p}: {a} vs {b}");
            assert!((phi - bphi).abs() <= 1e-9 * bphi.abs().max(1.0), "{p}: {phi} vs {bphi}");
        }
        // Far enough that both sides are the monopole + quadrupole.
        let c = (lo + hi) * 0.5;
        let p = c + DVec3::new(80.0, -50.0, 30.0).normalize() * (FAR_RATIO * poly.radius() * 2.5);
        let (a, phi) = poly.field(p);
        let (b, bphi) = prim.field(p);
        assert!(rel(a, b) <= 1e-8, "far {a} vs {b}");
        assert!((phi - bphi).abs() <= 1e-8 * bphi.abs(), "far potential {phi} vs {bphi}");

        // `field_over_cube` of the same box: Werner near, the closed form past `EXACT_RATIO`
        // (its multipoles cancel), and that box's window split where the range cuts it.
        let rad = poly.radius();
        for p in [c + DVec3::new(0.4, -0.2, 0.1), c + DVec3::Y * (EXACT_RATIO * rad * 4.0), c + DVec3::new(crate::gravity::R_IN, 0.0, 0.0)] {
            let (a, phi) = poly.field_over_cube(lo, hi, 3.0, p);
            let (b, bphi) = prim.field(p);
            assert!(rel(a, b) <= 1e-8 || (a - b).length() <= 1e-8, "over cube {p}: {a} vs {b}");
            assert!((phi - bphi).abs() <= 1e-8 * bphi.abs().max(1.0), "over cube {p}: {phi} vs {bphi}");
        }
    }

    #[test]
    fn a_sphere_polyhedron_converges_to_the_ball() {
        let radius = 2.0f64;
        let density = 1.5;
        let centre = DVec3::new(10.0, -4.0, 3.0);
        let ball_mass = 4.0 / 3.0 * std::f64::consts::PI * radius.powi(3) * density;
        let mut prev = f64::MAX;
        for n in [2usize, 4, 8] {
            let poly = Polyhedron::from_surface(centre - DVec3::splat(1.0), centre + DVec3::splat(1.0), n, density, |p| {
                centre + (p - centre).normalize() * radius
            });
            assert!((poly.mass() - ball_mass).abs() < prev * ball_mass || n == 2);
            let mass_err = (poly.mass() - ball_mass).abs() / ball_mass;
            assert!(mass_err < prev, "n={n} mass error {mass_err} did not fall");
            prev = mass_err;
            let outside = centre + DVec3::new(0.0, radius * 2.0, 0.0);
            let (a, _) = poly.field(outside);
            let want = shape::ball_field(centre, radius, outside).0 * density;
            let err = rel(a, want);
            if n == 8 {
                assert!(err < 0.02, "n=8 outside error {err}: {a} vs {want}");
                assert!(mass_err < 0.02, "n=8 mass error {mass_err}");
            }
            let inside = centre + DVec3::new(radius * 0.4, 0.0, 0.0);
            let (ai, _) = poly.field(inside);
            let want_i = shape::ball_field(centre, radius, inside).0 * density;
            if n == 8 {
                assert!(rel(ai, want_i) < 0.05, "n=8 inside {ai} vs {want_i}");
            }
        }
    }

    #[test]
    fn acceleration_matches_the_potential_gradient() {
        let poly = Polyhedron::from_box(DVec3::splat(-1.0), DVec3::splat(1.0), 4.0);
        for p in [DVec3::new(0.3, -0.2, 0.4), DVec3::new(1.6, 0.4, -0.5), DVec3::new(-2.0, 0.2, 0.7)] {
            let h = 1e-5;
            let a = poly.field(p).0;
            for axis in 0..3 {
                let mut step = DVec3::ZERO;
                step[axis] = h;
                let dphi = (poly.field(p + step).1 - poly.field(p - step).1) / (2.0 * h);
                let got = a[axis];
                assert!((got + dphi).abs() <= 1e-6 * (got.abs() + dphi.abs()).max(1e-8), "axis {axis} at {p}: a={got} ∂Φ={dphi}");
            }
        }
    }

    #[test]
    fn the_field_is_continuous_across_a_face() {
        let (lo, hi) = (DVec3::new(-1.0, -2.0, -0.5), DVec3::new(1.5, 0.5, 2.0));
        let poly = Polyhedron::from_box(lo, hi, 1.0);
        let eps = 1e-7;
        // Across the +Y face, clear of every edge.
        let under = DVec3::new(0.2, hi.y - eps, 0.3);
        let over = DVec3::new(0.2, hi.y + eps, 0.3);
        let (a, pa) = poly.field(under);
        let (b, pb) = poly.field(over);
        assert!((a - b).length() <= 1e-6 * a.length().max(1.0), "{a} vs {b}");
        assert!((pa - pb).abs() <= 1e-6 * pa.abs().max(1.0), "{pa} vs {pb}");
        // And finite on the face itself.
        let (c, pc) = poly.field(DVec3::new(0.2, hi.y, 0.3));
        assert!(c.is_finite() && pc.is_finite(), "{c} {pc}");
        assert!((c - a).length() <= 1e-5 * a.length().max(1.0), "on face {c} vs {a}");
    }

    #[test]
    fn a_sagging_cube_surface_changes_the_field() {
        use crate::mechanics::genesis;
        use crate::space::warp::Warp;
        let half = 6_000_000.0;
        let solved = genesis::tabulated(2.6, half).expect("genesis table");
        let warp = Warp::from_solved(&solved, DVec3::ZERO);
        let h = DVec3::splat(half);
        let chord = surface_chord(-h, h, FACE_QUADS, |p| warp.apply(p));
        // Trilinear twist, not the smooth sag. ~725 blocks here; under a few blocks needs ~900
        // quads per edge (~10⁷ triangles), which a gravity sample cannot walk.
        assert!(
            (400.0..1_200.0).contains(&chord),
            "chord error {chord} blocks at {FACE_QUADS}×{FACE_QUADS}"
        );
        let poly = Polyhedron::from_surface(-h, h, FACE_QUADS, 5.0, |p| warp.apply(p));
        let face_ref = DVec3::new(half, 0.0, 0.0);
        let face = warp.apply(face_ref);
        assert!(face.length() > face_ref.length(), "the face bows out");
        // Between the old face and the bowed face: inside the polyhedron, outside the box.
        let mid = (face_ref + face) * 0.5;
        let (a, _) = poly.field(mid);
        let (b, _) = shape::box_field(-h, h, mid);
        let b = b * 5.0;
        let inward = -mid.normalize();
        assert!(a.dot(inward) > 0.0 && b.dot(inward) > 0.0, "both pull inward");
        assert!((a - b).length() > 1e-3 * b.length(), "the sag moves the field: {a} vs {b}");
        // Past a few radii the sample is the cube plus the monopole and quadrupole difference.
        let far = face.normalize() * (poly.radius() * (EXACT_RATIO + 2.0));
        let (af, pf) = poly.field_over_cube(-h, h, 5.0, far);
        assert!(af.is_finite() && pf.is_finite() && af.dot(-far) > 0.0, "far approx {af} {pf}");
    }
}
