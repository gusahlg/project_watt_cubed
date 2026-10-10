//! The grids a field runs on: a finite box ([`Box3`]) and a cube-sphere ([`Sphere`]). Both list a
//! cell's neighbours in a fixed order, so a rule that folds them (ties to the first) is exact.

use std::sync::OnceLock;

/// Most neighbours any topology reports for one cell.
pub const MAX_NEIGHBOURS: usize = 26;

/// A finite set of cells and who neighbours whom.
pub trait Topology: Sync {
    /// Number of cells.
    fn len(&self) -> usize;
    /// Writes the neighbours of cell `i` into `out` in a fixed order and returns how many.
    fn neighbours(&self, i: usize, out: &mut [u32; MAX_NEIGHBOURS]) -> usize;
    /// Whether a pass computes cell `i`. Other cells are copies of an owner and are never anyone's
    /// neighbour; [`Topology::glue`] fills them after the last pass.
    fn owns(&self, _i: usize) -> bool {
        true
    }
    /// `(copy, owner)` pairs.
    fn glue(&self) -> &[(u32, u32)] {
        &[]
    }
}

/// A box of `n[0] · n[1] · n[2]` cells, x fastest. Boundary cells have fewer neighbours.
#[derive(Clone, Copy, Debug)]
pub struct Box3 {
    pub n: [u32; 3],
    offsets: [[i32; 3]; MAX_NEIGHBOURS],
    /// The offsets as index steps, for interior cells.
    steps: [i32; MAX_NEIGHBOURS],
    count: usize,
}

impl Box3 {
    /// A box of `n` cells per axis; `full` selects the 26-neighbourhood, otherwise the 6 faces.
    pub fn new(n: [u32; 3], full: bool) -> Self {
        let mut offsets = [[0i32; 3]; MAX_NEIGHBOURS];
        let mut count = 0;
        for dz in -1..=1i32 {
            for dy in -1..=1i32 {
                for dx in -1..=1i32 {
                    let taxicab = dx.abs() + dy.abs() + dz.abs();
                    if taxicab != 0 && (full || taxicab == 1) {
                        offsets[count] = [dx, dy, dz];
                        count += 1;
                    }
                }
            }
        }
        let stride = [1, n[0] as i32, (n[0] * n[1]) as i32];
        let steps = offsets.map(|o| o[0] * stride[0] + o[1] * stride[1] + o[2] * stride[2]);
        Self { n, offsets, steps, count }
    }

    /// Index of cell `p`.
    #[inline]
    pub fn index(&self, p: [u32; 3]) -> usize {
        (p[0] + self.n[0] * (p[1] + self.n[1] * p[2])) as usize
    }

    /// Coordinates of cell `i`.
    #[inline]
    pub fn coords(&self, i: usize) -> [u32; 3] {
        let i = i as u32;
        let xy = self.n[0] * self.n[1];
        [i % self.n[0], (i % xy) / self.n[0], i / xy]
    }
}

impl Topology for Box3 {
    fn len(&self) -> usize {
        (self.n[0] * self.n[1] * self.n[2]) as usize
    }

    #[inline]
    fn neighbours(&self, i: usize, out: &mut [u32; MAX_NEIGHBOURS]) -> usize {
        let p = self.coords(i).map(|v| v as i32);
        let n = self.n.map(|v| v as i32);
        if (0..3).all(|a| p[a] > 0 && p[a] < n[a] - 1) {
            for (o, &s) in out.iter_mut().zip(&self.steps[..self.count]) {
                *o = (i as i32 + s) as u32;
            }
            return self.count;
        }
        let mut k = 0;
        for (o, &s) in self.offsets[..self.count].iter().zip(&self.steps) {
            if (0..3).all(|a| p[a] + o[a] >= 0 && p[a] + o[a] < n[a]) {
                out[k] = (i as i32 + s) as u32;
                k += 1;
            }
        }
        k
    }
}

/// `(axis, sign)` of the six faces, in the atlas order +X, −X, +Y, −Y, +Z, −Z.
pub const FACE_AXES: [(usize, i32); 6] = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)];

/// A cube-sphere of `6 · (G+1)²` nodes: per face (in [`FACE_AXES`] order) a `(G+1)²` grid of rows
/// `j` of columns `i`. Node `(f, i, j)` is the direction of the cube point
/// `n_f·G + u_f·(2i − G) + v_f·(2j − G)` (gnomonic: equal steps of `p_u / p_n`), with `u` and `v` the
/// next two axes after the face's own. A node's neighbours are its 3×3 grid neighbours on every face
/// that holds it (8, or 6 at the eight corners). A node on an edge or corner belongs to the lowest
/// face that holds it; its copies on the other faces are glued.
pub struct Sphere {
    pub g: u32,
    /// Unit direction of every node.
    pub dirs: Vec<[f64; 3]>,
    /// Chord length on the unit sphere of every edge, in [`Sphere::edge`] order.
    pub chord: Vec<f32>,
    start: Vec<u32>,
    list: Vec<u32>,
    owner: Vec<u32>,
    glue: Vec<(u32, u32)>,
}

impl Sphere {
    /// The sphere of resolution `g` (a power of two, 2 to 256), built once per process.
    pub fn get(g: u32) -> &'static Sphere {
        static SPHERES: [OnceLock<Sphere>; 9] = [const { OnceLock::new() }; 9];
        assert!(g.is_power_of_two() && (2..=256).contains(&g), "sphere resolution {g}");
        SPHERES[g.trailing_zeros() as usize].get_or_init(|| Sphere::build(g))
    }

    /// Index of node `(face, i, j)`.
    #[inline]
    pub fn node(&self, face: usize, i: u32, j: u32) -> usize {
        let side = (self.g + 1) as usize;
        face * side * side + j as usize * side + i as usize
    }

    /// Index of the first edge of node `i`: neighbour `k` of the list [`Topology::neighbours`]
    /// writes is edge `edge(i) + k`.
    #[inline]
    pub fn edge(&self, i: usize) -> usize {
        self.start[i] as usize
    }

    /// The owner of node `i` (itself unless it is a glued copy).
    #[inline]
    pub fn owner(&self, i: usize) -> usize {
        self.owner[i] as usize
    }

    /// The face and continuous grid position `(i, j)` in `[0, G]²` of a direction: two divisions,
    /// no trigonometry. Ties between faces go to the lowest face.
    pub fn locate(&self, d: [f64; 3]) -> (usize, f64, f64) {
        let mut face = 0;
        let mut best = -1.0;
        for (f, &(axis, sign)) in FACE_AXES.iter().enumerate() {
            let v = d[axis] * sign as f64;
            if v > best {
                best = v;
                face = f;
            }
        }
        let (axis, _) = FACE_AXES[face];
        let half = self.g as f64 * 0.5;
        let inv = 1.0 / best;
        (face, half * (1.0 + d[(axis + 1) % 3] * inv), half * (1.0 + d[(axis + 2) % 3] * inv))
    }

    /// Copy every owner's value onto its glued copies.
    pub fn glue_cells<C: Copy>(&self, cells: &mut [C]) {
        for &(copy, owner) in &self.glue {
            cells[copy as usize] = cells[owner as usize];
        }
    }

    fn build(g: u32) -> Sphere {
        let side = g + 1;
        let len = 6 * (side * side) as usize;
        let gi = g as i32;
        let point = |f: usize, i: u32, j: u32| {
            let (axis, sign) = FACE_AXES[f];
            let mut p = [0i32; 3];
            p[axis] = sign * gi;
            p[(axis + 1) % 3] = 2 * i as i32 - gi;
            p[(axis + 2) % 3] = 2 * j as i32 - gi;
            p
        };
        let owner_of = |p: [i32; 3]| {
            let f = FACE_AXES.iter().position(|&(axis, sign)| p[axis] * sign == gi).expect("a surface point");
            let (axis, _) = FACE_AXES[f];
            let i = ((p[(axis + 1) % 3] + gi) / 2) as u32;
            let j = ((p[(axis + 2) % 3] + gi) / 2) as u32;
            (f * (side * side) as usize + (j * side + i) as usize) as u32
        };
        let mut dirs = Vec::with_capacity(len);
        let mut owner = Vec::with_capacity(len);
        let mut start = Vec::with_capacity(len + 1);
        let mut list = Vec::with_capacity(len * 8);
        let mut glue = Vec::new();
        for f in 0..6 {
            for j in 0..side {
                for i in 0..side {
                    let p = point(f, i, j);
                    let pf = p.map(|v| v as f64);
                    let l = (pf[0] * pf[0] + pf[1] * pf[1] + pf[2] * pf[2]).sqrt();
                    dirs.push(pf.map(|v| v / l));
                    let me = owner_of(p);
                    let idx = dirs.len() as u32 - 1;
                    owner.push(me);
                    if me != idx {
                        glue.push((idx, me));
                    }
                    start.push(list.len() as u32);
                    if me != idx {
                        continue;
                    }
                    // The union of the in-face neighbours on every face holding this point.
                    let first = list.len();
                    for (g_face, &(axis, sign)) in FACE_AXES.iter().enumerate() {
                        if p[axis] * sign != gi {
                            continue;
                        }
                        let fi = (p[(axis + 1) % 3] + gi) / 2;
                        let fj = (p[(axis + 2) % 3] + gi) / 2;
                        for dj in -1..=1 {
                            for di in -1..=1 {
                                let (ni, nj) = (fi + di, fj + dj);
                                if (di, dj) == (0, 0) || ni < 0 || nj < 0 || ni > gi || nj > gi {
                                    continue;
                                }
                                let q = owner_of(point(g_face, ni as u32, nj as u32));
                                if !list[first..].contains(&q) {
                                    list.push(q);
                                }
                            }
                        }
                    }
                }
            }
        }
        start.push(list.len() as u32);
        let mut chord = vec![0.0f32; list.len()];
        for i in 0..len {
            for e in start[i] as usize..start[i + 1] as usize {
                let (a, b) = (dirs[i], dirs[list[e] as usize]);
                let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
                chord[e] = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() as f32;
            }
        }
        Sphere { g, dirs, chord, start, list, owner, glue }
    }
}

impl Topology for Sphere {
    fn len(&self) -> usize {
        self.dirs.len()
    }

    #[inline]
    fn neighbours(&self, i: usize, out: &mut [u32; MAX_NEIGHBOURS]) -> usize {
        let s = &self.list[self.start[i] as usize..self.start[i + 1] as usize];
        out[..s.len()].copy_from_slice(s);
        s.len()
    }

    #[inline]
    fn owns(&self, i: usize) -> bool {
        self.owner[i] as usize == i
    }

    fn glue(&self) -> &[(u32, u32)] {
        &self.glue
    }
}
