//! Curved patches (guide §§5, 9, 10): cube-sphere charts. A round body's shell is covered by six
//! charts per depth band; a chart maps cell coordinates `(i, j)` across its face and `k` along the
//! radius to physical points. Three maps are offered so the lab can measure their distortion: the
//! normalised (gnomonic) projection, the equiangular one, and the "spherified cube".
//!
//! Charts are layout — world state chosen once — never physics: gravity and mass never read them.

use glam::{DMat3, DVec3};

use crate::coord::Face;

/// A cube-to-sphere map of one face's parameter square `[-1, 1]²`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Map {
    /// The normalised cube: equal steps on the cube face (the deliberately imperfect baseline).
    Gnomonic,
    /// Equal angles from the centre along both face axes.
    Equiangular,
    /// The "spherified cube" (each axis shrunk by the other two), the most uniform in area.
    Spherified,
}

/// The right-handed basis `(t_u, n, t_v)` of a face (same convention as `space::FaceFrame`).
pub fn basis(face: Face) -> (DVec3, DVec3, DVec3) {
    let (x, y, z) = (DVec3::X, DVec3::Y, DVec3::Z);
    match face {
        Face::PosY => (x, y, z),
        Face::NegY => (x, -y, -z),
        Face::PosX => (-y, x, z),
        Face::NegX => (y, -x, z),
        Face::PosZ => (x, z, -y),
        Face::NegZ => (x, -z, y),
    }
}

impl Map {
    /// The unit direction of parameters `(xi, eta)` in face-local axes (x along `t_u`, y along the
    /// normal, z along `t_v`).
    pub fn dir(self, xi: f64, eta: f64) -> DVec3 {
        match self {
            Map::Gnomonic => DVec3::new(xi, 1.0, eta).normalize(),
            Map::Equiangular => {
                let q = std::f64::consts::FRAC_PI_4;
                DVec3::new((xi * q).tan(), 1.0, (eta * q).tan()).normalize()
            }
            Map::Spherified => {
                let (x2, z2) = (xi * xi, eta * eta);
                DVec3::new(
                    xi * (1.0 - 0.5 - z2 * 0.5 + z2 / 3.0).sqrt(),
                    (1.0 - x2 * 0.5 - z2 * 0.5 + x2 * z2 / 3.0).sqrt(),
                    eta * (1.0 - x2 * 0.5 - 0.5 + x2 / 3.0).sqrt(),
                )
            }
        }
    }

    /// The parameters of a face-local direction (`d.y > 0`, inside this face's pyramid).
    pub fn inverse(self, d: DVec3) -> (f64, f64) {
        let (s, t) = (d.x / d.y, d.z / d.y);
        match self {
            Map::Gnomonic => (s, t),
            Map::Equiangular => {
                let k = 4.0 / std::f64::consts::PI;
                (s.atan() * k, t.atan() * k)
            }
            Map::Spherified => {
                // Newton on the forward map, started from the gnomonic guess.
                let target = d.normalize();
                let (mut xi, mut eta) = (s.clamp(-1.0, 1.0), t.clamp(-1.0, 1.0));
                for _ in 0..24 {
                    let f = self.dir(xi, eta);
                    let h = 1e-7;
                    let fx = (self.dir(xi + h, eta) - self.dir(xi - h, eta)) / (2.0 * h);
                    let fz = (self.dir(xi, eta + h) - self.dir(xi, eta - h)) / (2.0 * h);
                    // Solve the 2×2 least-squares step in the tangent plane.
                    let r = target - f;
                    let (a, b, c) = (fx.dot(fx), fx.dot(fz), fz.dot(fz));
                    let (p, q) = (fx.dot(r), fz.dot(r));
                    let det = a * c - b * b;
                    if det.abs() < 1e-30 {
                        break;
                    }
                    let (dx, dz) = ((c * p - b * q) / det, (a * q - b * p) / det);
                    xi += dx;
                    eta += dz;
                    if dx.abs() + dz.abs() < 1e-15 {
                        break;
                    }
                }
                (xi, eta)
            }
        }
    }
}

/// One chart: a face of a spherical shell band, `n × n` cells across, `layers` cells thick.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Chart {
    pub centre: DVec3,
    pub face: Face,
    pub map: Map,
    /// Cells per face edge.
    pub n: u32,
    /// Radius of the band's inner surface.
    pub r0: f64,
    /// Radial cells (one block each).
    pub layers: u32,
}

impl Chart {
    /// The physical point of continuous cell coordinates (corners at integers).
    pub fn point(&self, i: f64, j: f64, k: f64) -> DVec3 {
        let step = 2.0 / self.n as f64;
        let (xi, eta) = (-1.0 + i * step, -1.0 + j * step);
        let d = self.map.dir(xi, eta);
        let (tu, n, tv) = basis(self.face);
        self.centre + (tu * d.x + n * d.y + tv * d.z) * (self.r0 + k)
    }

    /// Continuous cell coordinates of a physical point (inverse of [`point`](Self::point)), or `None`
    /// when the point is not in this face's pyramid.
    pub fn cell_of(&self, p: DVec3) -> Option<DVec3> {
        let rel = p - self.centre;
        let (tu, n, tv) = basis(self.face);
        let local = DVec3::new(rel.dot(tu), rel.dot(n), rel.dot(tv));
        if local.y <= 0.0 || local.x.abs() > local.y * 1.0001 || local.z.abs() > local.y * 1.0001 {
            return None;
        }
        let (xi, eta) = self.map.inverse(local);
        let step = 2.0 / self.n as f64;
        Some(DVec3::new((xi + 1.0) / step, (eta + 1.0) / step, local.length() - self.r0))
    }

    /// The eight corners of cell `(i, j, k)` (bit 0 → +i, bit 1 → +j, bit 2 → +k).
    pub fn corners(&self, i: u32, j: u32, k: u32) -> [DVec3; 8] {
        std::array::from_fn(|c| {
            self.point((i + (c as u32 & 1)) as f64, (j + (c as u32 >> 1 & 1)) as f64, (k + (c as u32 >> 2 & 1)) as f64)
        })
    }

    /// The Jacobian `∂x/∂(i, j, k)` at continuous cell coordinates (central differences).
    pub fn jacobian(&self, i: f64, j: f64, k: f64) -> DMat3 {
        let h = 1e-3;
        let col = |d: DVec3| (self.point(i + d.x * h, j + d.y * h, k + d.z * h) - self.point(i - d.x * h, j - d.y * h, k - d.z * h)) / (2.0 * h);
        DMat3::from_cols(col(DVec3::X), col(DVec3::Y), col(DVec3::Z))
    }
}

/// Quality of one hexahedral cell (guide §9.2).
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Quality {
    /// Longest over shortest of the 12 edges.
    pub edge_ratio: f64,
    /// Largest deviation of a face corner angle from 90°, degrees.
    pub skew_deg: f64,
    /// Volume in blocks³ (1 is nominal).
    pub volume: f64,
    /// Singular values of the centre Jacobian, ascending.
    pub singular: [f64; 3],
}

/// The quality of a hexahedron given its corners in the [`Chart::corners`] order, plus its centre
/// Jacobian.
pub fn quality(c: &[DVec3; 8], jacobian: DMat3) -> Quality {
    const EDGES: [(usize, usize); 12] = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)];
    let lens = EDGES.map(|(a, b)| (c[a] - c[b]).length());
    let (lo, hi) = lens.iter().fold((f64::MAX, 0.0f64), |(lo, hi), &l| (lo.min(l), hi.max(l)));
    // Faces as corner cycles.
    const FACES: [[usize; 4]; 6] = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]];
    let mut skew = 0.0f64;
    for f in FACES {
        for v in 0..4 {
            let (p, a, b) = (c[f[v]], c[f[(v + 1) % 4]], c[f[(v + 3) % 4]]);
            let ang = (a - p).angle_between(b - p).to_degrees();
            skew = skew.max((ang - 90.0).abs());
        }
    }
    // Volume: six tetrahedra around the 0–7 diagonal.
    let t = |p: [usize; 4]| ((c[p[1]] - c[p[0]]).cross(c[p[2]] - c[p[0]])).dot(c[p[3]] - c[p[0]]).abs() / 6.0;
    let volume = [[0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7], [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7]].map(t).iter().sum();
    Quality { edge_ratio: hi / lo, skew_deg: skew, volume, singular: singular_values(jacobian) }
}

/// Singular values of a 3×3 matrix (square roots of the eigenvalues of `MᵀM`, by Jacobi rotations).
pub fn singular_values(m: DMat3) -> [f64; 3] {
    let mut a = (m.transpose() * m).to_cols_array_2d();
    for _ in 0..32 {
        let (mut p, mut q, mut big) = (0, 1, 0.0f64);
        for i in 0..3 {
            for j in (i + 1)..3 {
                if a[i][j].abs() > big {
                    big = a[i][j].abs();
                    (p, q) = (i, j);
                }
            }
        }
        if big < 1e-18 {
            break;
        }
        let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let (cs, sn) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
        let mut r = [[0.0; 3]; 3];
        for (i, row) in r.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        r[p][p] = cs;
        r[q][q] = cs;
        r[p][q] = sn;
        r[q][p] = -sn;
        // a = rᵀ a r
        let mul = |x: &[[f64; 3]; 3], y: &[[f64; 3]; 3]| {
            let mut z = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    z[i][j] = (0..3).map(|k| x[i][k] * y[k][j]).sum();
                }
            }
            z
        };
        let rt = [[r[0][0], r[1][0], r[2][0]], [r[0][1], r[1][1], r[2][1]], [r[0][2], r[1][2], r[2][2]]];
        a = mul(&mul(&rt, &a), &r);
    }
    let mut s = [a[0][0].max(0.0).sqrt(), a[1][1].max(0.0).sqrt(), a[2][2].max(0.0).sqrt()];
    s.sort_by(|x, y| x.total_cmp(y));
    s
}

/// The cells across a face edge for a shell of radius `r` whose surface cells should be about one
/// block wide (a quarter great circle of arc).
pub fn cells_per_face(r: f64) -> u32 {
    (std::f64::consts::FRAC_PI_2 * r).round() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_map_round_trips_and_covers_the_face() {
        for map in [Map::Gnomonic, Map::Equiangular, Map::Spherified] {
            for &(xi, eta) in &[(0.0, 0.0), (0.3, -0.7), (-1.0, 1.0), (0.999, -0.2), (0.5, 0.5)] {
                let d = map.dir(xi, eta);
                assert!((d.length() - 1.0).abs() < 1e-12, "{map:?} not unit");
                let (a, b) = map.inverse(d);
                assert!((a - xi).abs() < 1e-9 && (b - eta).abs() < 1e-9, "{map:?} ({xi},{eta}) -> ({a},{b})");
            }
            // The face corner lands on the cube diagonal for every map.
            let c = map.dir(1.0, 1.0);
            assert!((c - DVec3::splat(1.0).normalize()).length() < 1e-12, "{map:?} corner {c}");
        }
    }

    #[test]
    fn charts_of_neighbouring_faces_share_their_edge() {
        // The +Y chart's xi = +1 edge is the +X chart's matching edge: identical points, so cells
        // conform face to face across chart seams.
        for map in [Map::Gnomonic, Map::Equiangular, Map::Spherified] {
            let mk = |face| Chart { centre: DVec3::new(10.0, -5.0, 3.0), face, map, n: 400, r0: 1000.0, layers: 50 };
            let (top, side) = (mk(Face::PosY), mk(Face::PosX));
            for j in [0.0, 37.0, 200.0, 400.0] {
                let p = top.point(400.0, j, 10.0);
                let q = side.cell_of(p).expect("on the shared edge");
                let back = side.point(q.x, q.y, q.z);
                assert!((back - p).length() < 1e-6, "{map:?}: {p} vs {back}");
                assert!((q.x - 0.0).abs() < 1e-6 || (q.x - 400.0).abs() < 1e-6, "{map:?} edge column {q}");
            }
        }
    }

    #[test]
    fn the_face_centre_cell_is_nearly_a_unit_cube() {
        let r = 2.0e6;
        let n = cells_per_face(r);
        for map in [Map::Gnomonic, Map::Equiangular, Map::Spherified] {
            let chart = Chart { centre: DVec3::ZERO, face: Face::PosY, map, n, r0: r - 10.0, layers: 20 };
            let (i, j) = (n / 2, n / 2);
            let q = quality(&chart.corners(i, j, 10), chart.jacobian(i as f64 + 0.5, j as f64 + 0.5, 10.5));
            assert!(q.skew_deg < 1e-3, "{map:?} skew {}", q.skew_deg);
            // Equal parameter steps make gnomonic centre cells 4/π wide; the others are ~1.
            assert!(q.volume > 0.8 && q.volume < 1.7, "{map:?} volume {}", q.volume);
        }
    }

    #[test]
    fn singular_values_of_a_known_matrix() {
        let m = DMat3::from_cols(DVec3::new(3.0, 0.0, 0.0), DVec3::new(0.0, 2.0, 0.0), DVec3::new(0.0, 0.0, 1.0));
        let s = singular_values(m);
        assert!((s[0] - 1.0).abs() < 1e-12 && (s[1] - 2.0).abs() < 1e-12 && (s[2] - 3.0).abs() < 1e-12, "{s:?}");
        let rot = DMat3::from_rotation_z(0.7) * m;
        let s2 = singular_values(rot);
        assert!((s2[2] - 3.0).abs() < 1e-9, "{s2:?}");
    }
}
