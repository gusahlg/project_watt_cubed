//! A body's deformation map φ: a regular hex lattice over its reference box whose nodes carry
//! physical positions. φ is trilinear inside each element, which makes it the one authoritative
//! geometry (guide §9.3): a chunk lying inside an element is reproduced exactly by a cage through
//! its eight embedded corners, the local Jacobian drives motion and collision, and each element is
//! certified injective from the Bernstein coefficients of its Jacobian determinant.
//!
//! Reference coordinates are storage blocks (continuous, `f64`); physical coordinates are universe
//! blocks (`f64`, absolute). Corner order everywhere: bit 0 = +x, bit 1 = +y, bit 2 = +z.

use glam::{DMat3, DVec3};

/// Smallest element edge: one chunk, so elements tile chunks exactly.
pub const MIN_CELL: i64 = 16;

/// The lattice of one body.
#[derive(Clone, Debug)]
pub struct Lattice {
    /// Reference coordinate (blocks) of node `(0, 0, 0)`.
    pub origin: [i64; 3],
    /// Reference edge of one element, blocks: a power of two, at least [`MIN_CELL`].
    pub cell: i64,
    /// Elements per axis.
    pub dims: [usize; 3],
    /// Physical position of every node, index [`Lattice::node`].
    pub nodes: Vec<DVec3>,
    /// Physical point location: elements bucketed by their physical bounds.
    index: Index,
}

/// Elements bucketed on a uniform physical grid by their bounding boxes.
#[derive(Clone, Debug, Default)]
struct Index {
    lo: DVec3,
    step: f64,
    dims: [usize; 3],
    /// `starts[b]..starts[b + 1]` indexes `items` for bucket `b`.
    starts: Vec<u32>,
    items: Vec<u32>,
}

impl Lattice {
    /// An undeformed lattice: node `(i, j, k)` sits at `at + (i, j, k)·cell` (a pure translation
    /// of the reference box to `at`, the physical position of the reference origin).
    pub fn undeformed(origin: [i64; 3], cell: i64, dims: [usize; 3], at: DVec3) -> Self {
        assert!(cell >= MIN_CELL && cell.count_ones() == 1, "element edge {cell} is not a power of two ≥ {MIN_CELL}");
        assert!(dims.iter().all(|&d| d > 0), "a lattice has elements");
        let n = (dims[0] + 1) * (dims[1] + 1) * (dims[2] + 1);
        let mut nodes = Vec::with_capacity(n);
        for k in 0..=dims[2] {
            for j in 0..=dims[1] {
                for i in 0..=dims[0] {
                    nodes.push(at + DVec3::new(i as f64, j as f64, k as f64) * cell as f64);
                }
            }
        }
        let mut lattice = Self { origin, cell, dims, nodes, index: Index::default() };
        lattice.reindex();
        lattice
    }

    /// Node index of `(i, j, k)`.
    #[inline]
    pub fn node(&self, i: usize, j: usize, k: usize) -> usize {
        i + (self.dims[0] + 1) * (j + (self.dims[1] + 1) * k)
    }

    /// Number of elements.
    pub fn elements(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    /// Element index of `(i, j, k)`.
    #[inline]
    pub fn element(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.dims[0] * (j + self.dims[1] * k)
    }

    /// `(i, j, k)` of element `e`.
    #[inline]
    pub fn element_ijk(&self, e: usize) -> [usize; 3] {
        let (nx, ny) = (self.dims[0], self.dims[1]);
        [e % nx, (e / nx) % ny, e / (nx * ny)]
    }

    /// Node indices of element `e` in corner order.
    #[inline]
    pub fn element_nodes(&self, e: usize) -> [usize; 8] {
        let [i, j, k] = self.element_ijk(e);
        std::array::from_fn(|c| self.node(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1)))
    }

    /// Physical corners of element `e`.
    #[inline]
    pub fn corners(&self, e: usize) -> [DVec3; 8] {
        self.element_nodes(e).map(|n| self.nodes[n])
    }

    /// Reference box `[lo, hi)` in blocks.
    pub fn reference_box(&self) -> ([i64; 3], [i64; 3]) {
        let hi = std::array::from_fn(|a| self.origin[a] + self.dims[a] as i64 * self.cell);
        (self.origin, hi)
    }

    /// Whether a reference point lies in the lattice's box.
    pub fn contains(&self, s: DVec3) -> bool {
        let (lo, hi) = self.reference_box();
        (0..3).all(|a| s[a] >= lo[a] as f64 && s[a] < hi[a] as f64)
    }

    /// The element holding reference point `s` (clamped to the box) and `s` in its local `[0, 1]³`
    /// coordinates (outside the box the local coordinates extrapolate the edge element).
    #[inline]
    pub fn locate_reference(&self, s: DVec3) -> (usize, DVec3) {
        let mut ijk = [0usize; 3];
        let mut t = DVec3::ZERO;
        for a in 0..3 {
            let u = (s[a] - self.origin[a] as f64) / self.cell as f64;
            let e = (u.floor() as i64).clamp(0, self.dims[a] as i64 - 1);
            ijk[a] = e as usize;
            t[a] = u - e as f64;
        }
        (self.element(ijk[0], ijk[1], ijk[2]), t)
    }

    /// φ: physical position of reference point `s`.
    #[inline]
    pub fn embed(&self, s: DVec3) -> DVec3 {
        let (e, t) = self.locate_reference(s);
        trilinear(&self.corners(e), t)
    }

    /// ∂φ/∂s at reference point `s` (columns: the physical images of the reference axes per block).
    #[inline]
    pub fn jacobian(&self, s: DVec3) -> DMat3 {
        let (e, t) = self.locate_reference(s);
        local_jacobian(&self.corners(e), t) * (1.0 / self.cell as f64)
    }

    /// Physical corners of the reference box `[lo, lo + extent]` (cube), which must lie inside
    /// one element: then the trilinear cage through them reproduces φ exactly over the box.
    pub fn box_corners(&self, lo: [i64; 3], extent: i64) -> [DVec3; 8] {
        std::array::from_fn(|c| {
            let d = [(c & 1) as i64, ((c >> 1) & 1) as i64, ((c >> 2) & 1) as i64];
            self.embed(DVec3::new(
                (lo[0] + d[0] * extent) as f64,
                (lo[1] + d[1] * extent) as f64,
                (lo[2] + d[2] * extent) as f64,
            ))
        })
    }

    /// Whether the reference box `[lo, lo + extent]` lies inside a single element.
    pub fn box_in_one_element(&self, lo: [i64; 3], extent: i64) -> bool {
        (0..3).all(|a| {
            let rel = lo[a] - self.origin[a];
            let e = rel.div_euclid(self.cell);
            (rel + extent - 1).div_euclid(self.cell) == e && e >= 0 && e < self.dims[a] as i64
        })
    }

    /// φ⁻¹: the reference point whose image is physical point `p`, if `p` lies in the lattice.
    pub fn locate(&self, p: DVec3) -> Option<DVec3> {
        let b = self.index.bucket(p)?;
        let (from, to) = (self.index.starts[b] as usize, self.index.starts[b + 1] as usize);
        for &e in &self.index.items[from..to] {
            let e = e as usize;
            if let Some(t) = invert_trilinear(&self.corners(e), p) {
                let [i, j, k] = self.element_ijk(e);
                let ijk = [i, j, k];
                return Some(DVec3::new(
                    self.origin[0] as f64 + (ijk[0] as f64 + t.x) * self.cell as f64,
                    self.origin[1] as f64 + (ijk[1] as f64 + t.y) * self.cell as f64,
                    self.origin[2] as f64 + (ijk[2] as f64 + t.z) * self.cell as f64,
                ));
            }
        }
        None
    }

    /// Physical bounding box of all nodes.
    pub fn physical_bounds(&self) -> (DVec3, DVec3) {
        let mut lo = DVec3::splat(f64::INFINITY);
        let mut hi = DVec3::splat(f64::NEG_INFINITY);
        for &n in &self.nodes {
            lo = lo.min(n);
            hi = hi.max(n);
        }
        (lo, hi)
    }

    /// Rebuild the physical point-location index (after the nodes moved).
    pub fn reindex(&mut self) {
        let (lo, hi) = self.physical_bounds();
        let n = self.elements();
        // About one element per bucket on the longest axis.
        let longest = (hi - lo).max_element().max(1.0);
        let per_axis = (n as f64).cbrt().ceil().max(1.0);
        let step = (longest / per_axis).max(1.0);
        let dims = std::array::from_fn(|a| (((hi[a] - lo[a]) / step).floor() as usize + 1).max(1));
        let buckets = dims[0] * dims[1] * dims[2];
        let mut counts = vec![0u32; buckets + 1];
        let range = |e: usize| -> ([usize; 3], [usize; 3]) {
            let c = self.corners(e);
            let mut a_lo = c[0];
            let mut a_hi = c[0];
            for p in &c[1..] {
                a_lo = a_lo.min(*p);
                a_hi = a_hi.max(*p);
            }
            let cell = |v: f64, a: usize| (((v - lo[a]) / step).floor().max(0.0) as usize).min(dims[a] - 1);
            ([cell(a_lo.x, 0), cell(a_lo.y, 1), cell(a_lo.z, 2)], [cell(a_hi.x, 0), cell(a_hi.y, 1), cell(a_hi.z, 2)])
        };
        let bucket = |x: usize, y: usize, z: usize| x + dims[0] * (y + dims[1] * z);
        for e in 0..n {
            let (a, b) = range(e);
            for z in a[2]..=b[2] {
                for y in a[1]..=b[1] {
                    for x in a[0]..=b[0] {
                        counts[bucket(x, y, z) + 1] += 1;
                    }
                }
            }
        }
        for b in 0..buckets {
            counts[b + 1] += counts[b];
        }
        let mut fill = counts.clone();
        let mut items = vec![0u32; counts[buckets] as usize];
        for e in 0..n {
            let (a, b) = range(e);
            for z in a[2]..=b[2] {
                for y in a[1]..=b[1] {
                    for x in a[0]..=b[0] {
                        let k = bucket(x, y, z);
                        items[fill[k] as usize] = e as u32;
                        fill[k] += 1;
                    }
                }
            }
        }
        self.index = Index { lo, step, dims, starts: counts, items };
    }

    /// Certified lower bound of `det J` over element `e` relative to the undeformed `cell³`
    /// (positive means the element is injective and orientation preserving everywhere inside).
    pub fn certify(&self, e: usize) -> f64 {
        det_lower_bound(&self.corners(e)) / (self.cell as f64).powi(3)
    }
}

impl Index {
    fn bucket(&self, p: DVec3) -> Option<usize> {
        if self.starts.is_empty() {
            return None;
        }
        let mut c = [0usize; 3];
        for a in 0..3 {
            let u = ((p[a] - self.lo[a]) / self.step).floor();
            if u < 0.0 || u >= self.dims[a] as f64 {
                return None;
            }
            c[a] = u as usize;
        }
        Some(c[0] + self.dims[0] * (c[1] + self.dims[1] * c[2]))
    }
}

/// Trilinear interpolation of eight corners at local `t ∈ [0, 1]³`.
#[inline]
pub fn trilinear(c: &[DVec3; 8], t: DVec3) -> DVec3 {
    let x00 = c[0].lerp(c[1], t.x);
    let x10 = c[2].lerp(c[3], t.x);
    let x01 = c[4].lerp(c[5], t.x);
    let x11 = c[6].lerp(c[7], t.x);
    let y0 = x00.lerp(x10, t.y);
    let y1 = x01.lerp(x11, t.y);
    y0.lerp(y1, t.z)
}

/// ∂x/∂t of the trilinear map at `t` (columns per local axis).
#[inline]
pub fn local_jacobian(c: &[DVec3; 8], t: DVec3) -> DMat3 {
    let (u, v, w) = (t.x, t.y, t.z);
    let (iu, iv, iw) = (1.0 - u, 1.0 - v, 1.0 - w);
    let du = (c[1] - c[0]) * (iv * iw) + (c[3] - c[2]) * (v * iw) + (c[5] - c[4]) * (iv * w) + (c[7] - c[6]) * (v * w);
    let dv = (c[2] - c[0]) * (iu * iw) + (c[3] - c[1]) * (u * iw) + (c[6] - c[4]) * (iu * w) + (c[7] - c[5]) * (u * w);
    let dw = (c[4] - c[0]) * (iu * iv) + (c[5] - c[1]) * (u * iv) + (c[6] - c[2]) * (iu * v) + (c[7] - c[3]) * (u * v);
    DMat3::from_cols(du, dv, dw)
}

/// Local coordinates of `p` inside the trilinear element, if it lies inside (Newton from the
/// centre, with a small tolerance at the faces so a point on a shared face is found in either).
pub fn invert_trilinear(c: &[DVec3; 8], p: DVec3) -> Option<DVec3> {
    const TOL: f64 = 1e-9;
    let mut t = DVec3::splat(0.5);
    let scale = (c[7] - c[0]).length().max(1e-12);
    for _ in 0..32 {
        let r = trilinear(c, t) - p;
        if r.length() <= 1e-9 * scale {
            break;
        }
        let j = local_jacobian(c, t);
        let det = j.determinant();
        if !det.is_finite() || det.abs() < 1e-300 {
            return None;
        }
        let step = j.inverse() * r;
        t -= step;
        // A point far outside drives the iterate away: give up early.
        if t.abs().max_element() > 4.0 {
            return None;
        }
    }
    let r = trilinear(c, t) - p;
    if r.length() > 1e-6 * scale {
        return None;
    }
    (t.min_element() >= -TOL && t.max_element() <= 1.0 + TOL).then(|| t.clamp(DVec3::ZERO, DVec3::ONE))
}

/// A certified lower bound of `det(∂x/∂t)` over `[0, 1]³`: the determinant of a trilinear map is
/// tri-quadratic, so its Bernstein coefficients (degree 2 per axis) bound it from below.
pub fn det_lower_bound(c: &[DVec3; 8]) -> f64 {
    // Lagrange samples at t ∈ {0, ½, 1}³ …
    let mut f = [[[0.0f64; 3]; 3]; 3];
    for (k, w) in [0.0, 0.5, 1.0].into_iter().enumerate() {
        for (j, v) in [0.0, 0.5, 1.0].into_iter().enumerate() {
            for (i, u) in [0.0, 0.5, 1.0].into_iter().enumerate() {
                f[k][j][i] = local_jacobian(c, DVec3::new(u, v, w)).determinant();
            }
        }
    }
    // … to Bernstein, one axis at a time: b0 = f(0), b1 = 2 f(½) − (f(0) + f(1)) / 2, b2 = f(1).
    let to_bernstein = |a: f64, h: f64, b: f64| [a, 2.0 * h - 0.5 * (a + b), b];
    for k in 0..3 {
        for j in 0..3 {
            let [a, h, b] = f[k][j];
            f[k][j] = to_bernstein(a, h, b);
        }
    }
    for k in 0..3 {
        for i in 0..3 {
            let b = to_bernstein(f[k][0][i], f[k][1][i], f[k][2][i]);
            for j in 0..3 {
                f[k][j][i] = b[j];
            }
        }
    }
    for j in 0..3 {
        for i in 0..3 {
            let b = to_bernstein(f[0][j][i], f[1][j][i], f[2][j][i]);
            for k in 0..3 {
                f[k][j][i] = b[k];
            }
        }
    }
    f.iter().flatten().flatten().copied().fold(f64::INFINITY, f64::min)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sheared(l: &mut Lattice, amount: f64) {
        for n in l.nodes.iter_mut() {
            let y = n.y;
            n.x += amount * y;
        }
        l.reindex();
    }

    #[test]
    fn an_undeformed_lattice_is_a_translation() {
        let l = Lattice::undeformed([1_000, -32, 64], 32, [3, 2, 4], DVec3::new(-500.0, 7.0, 0.0));
        let s = DVec3::new(1_010.5, -20.25, 100.0);
        let p = l.embed(s);
        assert!((p - (s + DVec3::new(-1_500.0, 39.0, -64.0))).length() < 1e-9);
        assert!((l.jacobian(s) - DMat3::IDENTITY).abs_diff_eq(DMat3::ZERO, 1e-12));
        let back = l.locate(p).expect("inside");
        assert!((back - s).length() < 1e-6);
        assert!(l.locate(DVec3::new(1e6, 0.0, 0.0)).is_none());
        for e in 0..l.elements() {
            assert!((l.certify(e) - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn embed_locate_round_trip_on_a_curved_lattice() {
        let mut l = Lattice::undeformed([0, 0, 0], 16, [4, 4, 4], DVec3::ZERO);
        // Bend the lattice: a smooth radial bulge.
        for n in l.nodes.iter_mut() {
            let d = *n - DVec3::splat(32.0);
            *n += d * (0.002 * d.length());
        }
        l.reindex();
        for s in [DVec3::new(3.0, 5.0, 7.0), DVec3::new(40.0, 17.5, 63.0), DVec3::new(32.0, 32.0, 32.0)] {
            let p = l.embed(s);
            let back = l.locate(p).expect("a point of the lattice is found");
            assert!((back - s).length() < 1e-5, "{s} -> {p} -> {back}");
        }
        assert!((0..l.elements()).all(|e| l.certify(e) > 0.0));
    }

    #[test]
    fn the_jacobian_matches_finite_differences() {
        let mut l = Lattice::undeformed([0, 0, 0], 16, [2, 2, 2], DVec3::ZERO);
        sheared(&mut l, 0.3);
        for n in l.nodes.iter_mut() {
            n.z += 0.01 * n.x * n.y;
        }
        let s = DVec3::new(9.0, 21.0, 5.0);
        let j = l.jacobian(s);
        let h = 1e-4;
        for a in 0..3 {
            let mut d = DVec3::ZERO;
            d[a] = h;
            let fd = (l.embed(s + d) - l.embed(s - d)) / (2.0 * h);
            assert!((fd - j.col(a)).length() < 1e-6, "axis {a}: {fd} vs {}", j.col(a));
        }
    }

    #[test]
    fn certification_rejects_an_inverted_corner() {
        let mut c: [DVec3; 8] = std::array::from_fn(|i| DVec3::new((i & 1) as f64, ((i >> 1) & 1) as f64, ((i >> 2) & 1) as f64));
        assert!((det_lower_bound(&c) - 1.0).abs() < 1e-12);
        // Push corner 7 through the opposite face: the element folds near that corner.
        c[7] = DVec3::new(0.2, 0.2, 0.2);
        assert!(det_lower_bound(&c) < 0.0);
        // A mild move keeps it valid.
        c[7] = DVec3::new(1.1, 0.9, 1.05);
        assert!(det_lower_bound(&c) > 0.0);
    }

    #[test]
    fn a_box_inside_one_element_is_reproduced_by_its_corners() {
        let mut l = Lattice::undeformed([0, 0, 0], 64, [2, 2, 2], DVec3::ZERO);
        for n in l.nodes.iter_mut() {
            *n += DVec3::new(0.01 * n.y * n.y, 0.003 * n.x * n.z, 0.0);
        }
        l.reindex();
        let (lo, ext) = ([16, 32, 48], 16);
        assert!(l.box_in_one_element(lo, ext));
        assert!(!l.box_in_one_element([56, 0, 0], 16));
        let c = l.box_corners(lo, ext);
        let s = DVec3::new(21.0, 40.0, 50.0);
        let t = (s - DVec3::new(16.0, 32.0, 48.0)) / 16.0;
        assert!((trilinear(&c, t) - l.embed(s)).length() < 1e-9);
    }
}
