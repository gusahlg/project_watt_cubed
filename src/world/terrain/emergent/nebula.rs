//! Stages 1 and 2: the nebula and its systems. A 32³ field of mass and composition (offsets from a
//! base element) collapses for a few Jacobi phases: each cell sends a quarter of its mass to its
//! neighbour of largest smoothed mass, so mass is conserved exactly and filaments drain into knots.
//! A watershed of the result gives basins; basins far enough apart become systems, the rest join a
//! near system or become residual debris.

use field::{hash32_3, map, mul_q16, ridged_q16, Box3, Fbm, Field, NoiseBox, Rule, Topology, HALF, ONE};
use material::Element;

use super::Params;
use crate::world::terrain::TerrainCfg;

/// Cells per axis.
pub const N: u32 = 32;
/// Edge of a cell in blocks (2^26): the grid spans ±1.07e9.
pub const CELL: f64 = (1u64 << 26) as f64;
/// Noise lattice units per cell (the noise runs on integers).
const SUB: i32 = 64;
/// Pointer-jumping rounds of the watershed (2^15 covers any path in 32³ cells).
const JUMPS: u32 = 15;
/// Range of a composition offset per axis.
const COMP: i32 = 96;

/// One nebula cell: mass (Q16), smoothed mass, the neighbour it sends to (itself: none) and its
/// composition.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cell {
    pub m: u64,
    pub phi: u64,
    pub to: u32,
    pub comp: [i8; 4],
}

/// A system: a basin (or several) of the collapsed nebula far from every other system.
#[derive(Clone, Debug)]
pub struct System {
    /// Raw mass (Q16 cell units).
    pub mass: f64,
    /// Mass-weighted centroid of its first basin, in blocks.
    pub centre: [f64; 3],
    /// Member cells and their masses, in cell order.
    pub cells: Vec<(u32, u64)>,
    /// The sink cell and mass of every basin in it (the first basin first).
    pub sinks: Vec<(u32, u64)>,
    /// Holds the origin cell: the start system.
    pub origin: bool,
}

/// The collapsed nebula.
pub struct Nebula {
    /// The element compositions are offsets from.
    pub base: Element,
    pub cells: Vec<Cell>,
    /// Each cell's watershed sink.
    pub sink: Vec<u32>,
    pub basins: u32,
    /// Systems, the start system first.
    pub systems: Vec<System>,
    /// Raw mass of basins too close to a system to stand alone and too far to join one.
    pub residual: f64,
    pub total: f64,
}

/// Mean of a cell and its neighbours (mass or smoothed mass).
struct Blur {
    from_mass: bool,
}

impl Rule<Cell> for Blur {
    const RADIUS: u8 = 1;
    fn apply(&self, prev: &[Cell], i: usize, nb: &[u32]) -> Cell {
        let v = |c: &Cell| if self.from_mass { c.m } else { c.phi };
        let sum = nb.iter().fold(v(&prev[i]), |s, &n| s + v(&prev[n as usize]));
        Cell { phi: sum / (nb.len() as u64 + 1), ..prev[i] }
    }
}

/// Point at the neighbour of largest smoothed mass when it exceeds the cell's own (ties: lowest index).
struct Steepest;

impl Rule<Cell> for Steepest {
    const RADIUS: u8 = 1;
    fn apply(&self, prev: &[Cell], i: usize, nb: &[u32]) -> Cell {
        let mut best = (prev[i].phi, i as u32);
        for &n in nb {
            if prev[n as usize].phi > best.0 {
                best = (prev[n as usize].phi, n);
            }
        }
        Cell { to: best.1, ..prev[i] }
    }
}

/// Send `m >> shift` along the pointer, receive from every neighbour pointing here; composition
/// becomes the mass-weighted mean.
struct Transfer {
    shift: u32,
}

impl Rule<Cell> for Transfer {
    const RADIUS: u8 = 1;
    fn apply(&self, prev: &[Cell], i: usize, nb: &[u32]) -> Cell {
        let c = prev[i];
        let keep = if c.to as usize != i { c.m - (c.m >> self.shift) } else { c.m };
        let mut m = keep;
        let mut acc = c.comp.map(|v| v as i64 * keep as i64);
        for &n in nb {
            let o = prev[n as usize];
            if o.to as usize == i {
                let sent = o.m >> self.shift;
                m += sent;
                for a in 0..4 {
                    acc[a] += o.comp[a] as i64 * sent as i64;
                }
            }
        }
        let comp = if m == 0 { c.comp } else { acc.map(|v| (v / m as i64) as i8) };
        Cell { m, comp, ..c }
    }
}

/// Centre of cell coordinate `v` in blocks (cell 16 starts at the origin).
fn centre_of(v: u32) -> f64 {
    (v as f64 - (N / 2) as f64 + 0.5) * CELL
}

impl Nebula {
    /// The nebula of `seed`, collapsed, with its systems.
    pub fn new(seed: u32, cfg: &TerrainCfg, p: &Params, threads: usize) -> Nebula {
        let base = Element::new(hash32_3(seed, 0, 0, 0, 0xBA5E).to_le_bytes());
        let faces = Box3::new([N; 3], false);
        let full = Box3::new([N; 3], true);
        let origin = faces.index([N / 2; 3]);
        let amp = (p.prior_amp as i64 * cfg.variety as i64 / 100) as i32;
        let (lo, hi) = ([0; 3], [(N as i32 - 1) * SUB; 3]);
        let lumps = Fbm::new(seed, 16 * SUB, p.octaves, HALF, 1, lo, hi);
        let filaments = NoiseBox::new(seed, 8 * SUB, 9, lo, hi);
        let chem: [NoiseBox; 4] = std::array::from_fn(|k| NoiseBox::new(seed, 12 * SUB, 20 + k as u32, lo, hi));
        let prior = |i: usize| {
            let c = full.coords(i);
            let q = c.map(|v| v as i32 * SUB);
            let n = lumps.at(q);
            let r = ridged_q16(filaments.at(q));
            let t = (3 * n + r) / 4;
            let noise = (amp as i64 * mul_q16(t, t) as i64) * cfg.space as i64 / 100;
            let d2: i32 = (0..3).map(|a| (c[a] as i32 - (N / 2) as i32) * (c[a] as i32 - (N / 2) as i32)).sum();
            let r2 = p.well_r * p.well_r;
            let well = if d2 < r2 { p.well as i64 * ONE as i64 * (r2 - d2) as i64 / r2 as i64 } else { 0 };
            let comp = std::array::from_fn(|k| {
                let v = chem[k].at(q) - HALF;
                (v * 2 * COMP / ONE).clamp(-COMP, COMP) as i8
            });
            Cell { m: (ONE as i64 + noise + well) as u64, phi: 0, to: i as u32, comp }
        };
        let mut f = Field::new(map(full.len(), prior, threads));
        let send = Transfer { shift: p.send };
        for _ in 0..p.collapse {
            f.run(&faces, &Blur { from_mass: true }, 1, threads);
            f.run(&faces, &Blur { from_mass: false }, 1, threads);
            f.run(&full, &Steepest, 1, threads);
            f.run(&full, &send, 1, threads);
        }
        // The watershed reads the collapsed mass smoothed by `p.blur` sweeps.
        f.run(&faces, &Blur { from_mass: true }, 1, threads);
        f.run(&faces, &Blur { from_mass: false }, p.blur.max(1) - 1, threads);
        f.run(&full, &Steepest, 1, threads);
        let cells = f.into_cells();
        let mut sink: Vec<u32> = cells.iter().map(|c| c.to).collect();
        for _ in 0..JUMPS {
            sink = sink.iter().map(|&s| sink[s as usize]).collect();
        }
        // Basins in order of their first cell, accumulated in cell order.
        let mut basin_of = vec![u32::MAX; cells.len()];
        let mut basins: Vec<(u64, [u64; 3], Vec<(u32, u64)>, u32)> = Vec::new();
        for (i, c) in cells.iter().enumerate() {
            let s = sink[i] as usize;
            if basin_of[s] == u32::MAX {
                basin_of[s] = basins.len() as u32;
                basins.push((0, [0; 3], Vec::new(), s as u32));
            }
            let b = &mut basins[basin_of[s] as usize];
            b.0 += c.m;
            let at = full.coords(i);
            for a in 0..3 {
                b.1[a] += at[a] as u64 * 2 * c.m + c.m;
            }
            b.2.push((i as u32, c.m));
        }
        let origin_basin = basin_of[sink[origin] as usize] as usize;
        let mut order: Vec<usize> = (0..basins.len()).collect();
        order.sort_by_key(|&k| (k != origin_basin, std::cmp::Reverse(basins[k].0), basins[k].3));
        let total = cells.iter().map(|c| c.m as f64).sum();
        let mut systems: Vec<System> = Vec::new();
        let mut residual = 0.0;
        let floor = p.system_min * basins[origin_basin].0 as f64;
        for k in order {
            let (mass, moment, members, sink) = &basins[k];
            let sinks = vec![(*sink, *mass)];
            let centre: [f64; 3] = std::array::from_fn(|a| {
                (moment[a] as f64 / (2.0 * *mass as f64) - (N / 2) as f64) * CELL
            });
            let dist = |s: &System| (0..3).map(|a| (s.centre[a] - centre[a]) * (s.centre[a] - centre[a])).sum::<f64>().sqrt();
            let nearest = systems.iter().enumerate().map(|(i, s)| (dist(s), i)).min_by(|a, b| a.0.total_cmp(&b.0));
            let inside = centre.iter().all(|v| v.abs() <= p.system_bound);
            match nearest {
                None => systems.push(System { mass: *mass as f64, centre, cells: members.clone(), sinks, origin: true }),
                Some((d, _)) if d >= p.d_sep && inside && *mass as f64 >= floor => {
                    systems.push(System { mass: *mass as f64, centre, cells: members.clone(), sinks, origin: false })
                }
                Some((d, s)) if d < 2.0 * p.d_sep => {
                    let sys = &mut systems[s];
                    sys.mass += *mass as f64;
                    sys.cells.extend_from_slice(members);
                    sys.cells.sort_unstable();
                    sys.sinks.push((*sink, *mass));
                }
                Some(_) => residual += *mass as f64,
            }
        }
        Nebula { base, cells, sink, basins: basins.len() as u32, systems, residual, total }
    }

    /// Centre of cell `i` in blocks.
    pub fn cell_centre(i: u32) -> [f64; 3] {
        [i % N, i / N % N, i / (N * N)].map(centre_of)
    }
}
