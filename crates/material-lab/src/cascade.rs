//! 3-D lab grid and Jacobi-commit cascade generations.

use material::{interact, Configuration, EventKind, Law};

use crate::rng::Rng;
use crate::variants;

/// Face-adjacent offsets of a cube.
const FACE: [(i32, i32, i32); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

/// Cubic grid of configurations. Index is `x + n*(y + n*z)` (x fastest).
pub struct Grid {
    /// Edge length.
    pub n: usize,
    /// `n³` cells in position order.
    pub cells: Vec<Configuration>,
}

impl Grid {
    /// Fill every cell with `fill`.
    pub fn new(n: usize, fill: Configuration) -> Self {
        let k = n * n * n;
        Self {
            n,
            cells: vec![fill; k],
        }
    }

    /// Linear index of `(x,y,z)`.
    pub fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        x + self.n * (y + self.n * z)
    }

    /// Write a cell.
    pub fn set(&mut self, x: usize, y: usize, z: usize, c: Configuration) {
        let i = self.idx(x, y, z);
        self.cells[i] = c;
    }

    /// Face neighbours that lie inside the cube.
    pub fn neighbours(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        let n = self.n as i32;
        let x = (i % self.n) as i32;
        let y = ((i / self.n) % self.n) as i32;
        let z = (i / (self.n * self.n)) as i32;
        FACE.iter().filter_map(move |&(dx, dy, dz)| {
            let nx = x + dx;
            let ny = y + dy;
            let nz = z + dz;
            if nx < 0 || ny < 0 || nz < 0 || nx >= n || ny >= n || nz >= n {
                return None;
            }
            Some(nx as usize + self.n * (ny as usize + self.n * nz as usize))
        })
    }

    /// One Jacobi generation: every `changed` cell is origin against its 6 neighbours.
    /// Mutations are computed from the pre-generation state and committed together;
    /// two writes to the same target keep the last origin in position order.
    ///
    /// Returns the cells whose committed value differs from the pre-generation value.
    pub fn generation(&mut self, law: &Law, changed: &[usize]) -> Vec<usize> {
        let ncells = self.cells.len();
        let mut origins = changed.to_vec();
        origins.sort_unstable();
        origins.dedup();
        let mut proposed: Vec<Option<Configuration>> = vec![None; ncells];
        for &oi in &origins {
            for ni in self.neighbours(oi) {
                let r = interact(law, &self.cells[oi], &self.cells[ni], EventKind::NewContact);
                if r.changed {
                    proposed[ni] = Some(r.target);
                }
            }
        }
        let mut next = Vec::new();
        for (i, slot) in proposed.into_iter().enumerate() {
            if let Some(c) = slot {
                if c != self.cells[i] {
                    self.cells[i] = c;
                    next.push(i);
                }
            }
        }
        next
    }
}

/// One cascade run: fill, place at the centre, iterate generations.
pub struct CascadeRun {
    /// Jacobi generations after the centre event until quiet, or `max_gen` if still moving.
    pub generations: u32,
    /// True when the last generation produced no mutations.
    pub quiescent: bool,
    /// Cells mutated by the centre event.
    pub initial_changed: u32,
    /// Cells mutated per Jacobi generation (length `generations`).
    pub cells_per_gen: Vec<u32>,
    /// Largest of `initial_changed` and `cells_per_gen`.
    pub max_changed: u32,
}

/// Fill a cube from 3 random regions (centre + spread-8 variants) plus 10% random configs,
/// apply `NewContact` at the centre, then Jacobi-iterate.
pub fn run_one(law: &Law, rng: &mut Rng, n: usize, max_gen: u32) -> CascadeRun {
    let mut grid = fill(rng, n);
    let centre = grid.idx(n / 2, n / 2, n / 2);
    let mut changed = grid.generation(law, &[centre]);
    let initial_changed = changed.len() as u32;
    let mut max_changed = initial_changed;
    let mut cells_per_gen = Vec::new();
    let mut generations = 0u32;
    while !changed.is_empty() && generations < max_gen {
        changed = grid.generation(law, &changed);
        let c = changed.len() as u32;
        cells_per_gen.push(c);
        max_changed = max_changed.max(c);
        generations += 1;
    }
    CascadeRun {
        generations,
        quiescent: changed.is_empty(),
        initial_changed,
        cells_per_gen,
        max_changed,
    }
}

fn fill(rng: &mut Rng, n: usize) -> Grid {
    struct RegionFill {
        seed: (i32, i32, i32),
        members: [Configuration; 7],
    }
    let regions: [RegionFill; 3] = std::array::from_fn(|_| {
        let centre = rng.element();
        let vars = variants(centre, crate::SPREAD);
        let mut members: [Configuration; 7] = std::array::from_fn(|_| Configuration::void());
        members[0] = Configuration::single(centre);
        for (i, v) in vars.into_iter().enumerate() {
            members[i + 1] = Configuration::single(v);
        }
        RegionFill {
            seed: (
                rng.index(n) as i32,
                rng.index(n) as i32,
                rng.index(n) as i32,
            ),
            members,
        }
    });

    let mut grid = Grid::new(n, Configuration::void());
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let mut best = 0usize;
                let mut best_d = i32::MAX;
                for (ri, r) in regions.iter().enumerate() {
                    let d = (x as i32 - r.seed.0).abs()
                        + (y as i32 - r.seed.1).abs()
                        + (z as i32 - r.seed.2).abs();
                    if d < best_d {
                        best_d = d;
                        best = ri;
                    }
                }
                let pick = (x + 3 * y + 7 * z) % 7;
                grid.set(x, y, z, regions[best].members[pick].clone());
            }
        }
    }
    let ncells = n * n * n;
    let n_rand = ncells / 10;
    for _ in 0..n_rand {
        let i = rng.index(ncells);
        grid.cells[i] = rng.config(6);
    }
    grid
}
