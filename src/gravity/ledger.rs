//! The edit ledger: exact corrections to the generated matter (guide §6.6–6.7). Every committed
//! cell change records `amount(new) − amount(old)`; the ledger keeps the per-cell correction
//! against the generated baseline, per-chunk and per-region signed sums with first moments about
//! fixed centres (never a centre of mass — a correction can have zero net mass and a real dipole),
//! and the absolute mass that bounds the approximation error. All bookkeeping is integer, so the
//! state is identical whatever order the edits arrive in.

use std::collections::{BTreeMap, HashMap};

use glam::DVec3;

use super::kernel::{self, R_G, R_IN};

/// Chunk edge in cells (matches the world's chunks).
const CHUNK: i32 = 16;
/// Region edge in chunks.
const REGION: i32 = 16;
/// A node is evaluated by its monopole and dipole when its half-diagonal over its distance is
/// below this; otherwise it is opened.
const THETA: f64 = 0.5;

type Key = (i32, i32, i32);

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ChunkMass {
    mass: i64,
    /// Σ Δ·(2·local − 15) per axis: the first moment about the chunk centre in half-blocks.
    moment2: [i64; 3],
    abs: i64,
    /// Nonzero corrections, sorted by in-chunk index (`x + 16 z + 256 y`).
    cells: Vec<(u16, i16)>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct RegionMass {
    mass: i64,
    /// First moment about the region centre in half-blocks.
    moment2: [i64; 3],
    abs: i64,
    chunks: Vec<Key>,
}

/// Exact mass corrections from edits.
#[derive(Default, Debug)]
pub struct Ledger {
    chunks: HashMap<Key, ChunkMass>,
    /// Sorted, so every query sums in the same order whatever order edits arrived in.
    regions: BTreeMap<Key, RegionMass>,
}

/// Bound on the field dropped by truncating a node of absolute mass `abs` and half-diagonal `h`
/// after its dipole, seen from distance `d > h` (the quadrupole remainder, worst case).
fn truncation(abs: i64, h: f64, d: f64) -> f64 {
    let q = 1.0 - h / d;
    3.0 * abs as f64 * h * h / (d * d * d * d * q * q * q * q)
}

/// Monopole + dipole field of a correction node about the fixed centre `c`, per unit `G`.
fn multipole(c: DVec3, mass: f64, dipole: DVec3, p: DVec3) -> (DVec3, f64) {
    let d = p - c;
    let r2 = d.length_squared();
    let r = r2.sqrt();
    if r >= R_IN {
        // Far nodes: tapered monopole; the dipole there is far below the declared error.
        return kernel::point(c, mass, p);
    }
    let inv = 1.0 / r;
    let inv3 = inv * inv * inv;
    let pd = dipole.dot(d);
    let a = -d * (mass * inv3) + dipole * inv3 - d * (3.0 * pd * inv3 * inv * inv);
    (a, -mass * (inv + kernel::offset()) - pd * inv3)
}

impl Ledger {
    /// Record a change of `delta` amount at world cell `cell`.
    pub fn record(&mut self, cell: Key, delta: i32) {
        if delta == 0 {
            return;
        }
        let chunk = (cell.0.div_euclid(CHUNK), cell.1.div_euclid(CHUNK), cell.2.div_euclid(CHUNK));
        let local = [cell.0.rem_euclid(CHUNK), cell.1.rem_euclid(CHUNK), cell.2.rem_euclid(CHUNK)];
        let index = (local[0] + CHUNK * local[2] + CHUNK * CHUNK * local[1]) as u16;
        let region = (chunk.0.div_euclid(REGION), chunk.1.div_euclid(REGION), chunk.2.div_euclid(REGION));
        let k = [chunk.0 - region.0 * REGION, chunk.1 - region.1 * REGION, chunk.2 - region.2 * REGION];

        let cm = self.chunks.entry(chunk).or_default();
        let before = match cm.cells.binary_search_by_key(&index, |c| c.0) {
            Ok(i) => cm.cells[i].1 as i64,
            Err(_) => 0,
        };
        let after = before + delta as i64;
        match cm.cells.binary_search_by_key(&index, |c| c.0) {
            Ok(i) if after == 0 => {
                cm.cells.remove(i);
            }
            Ok(i) => cm.cells[i].1 = after as i16,
            Err(i) => cm.cells.insert(i, (index, after as i16)),
        }
        let d = delta as i64;
        let dabs = after.abs() - before.abs();
        cm.mass += d;
        cm.abs += dabs;
        for a in 0..3 {
            cm.moment2[a] += d * (2 * local[a] as i64 - (CHUNK as i64 - 1));
        }
        let emptied = cm.cells.is_empty();
        if emptied {
            debug_assert!(cm.mass == 0 && cm.abs == 0 && cm.moment2 == [0; 3]);
            self.chunks.remove(&chunk);
        }

        let rm = self.regions.entry(region).or_default();
        rm.mass += d;
        rm.abs += dabs;
        let half = (CHUNK * REGION) as i64; // region half-size in half-blocks
        for a in 0..3 {
            let chunk_off = 2 * CHUNK as i64 * k[a] as i64 + CHUNK as i64 - half;
            rm.moment2[a] += d * (chunk_off + 2 * local[a] as i64 - (CHUNK as i64 - 1));
        }
        match rm.chunks.binary_search(&chunk) {
            Ok(i) if emptied => {
                rm.chunks.remove(i);
            }
            Ok(_) => {}
            Err(i) if !emptied => rm.chunks.insert(i, chunk),
            Err(_) => {}
        }
        if rm.chunks.is_empty() {
            debug_assert!(rm.mass == 0 && rm.abs == 0 && rm.moment2 == [0; 3]);
            self.regions.remove(&region);
        }
    }

    /// Whether no correction is recorded.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Field of every correction at `p`, per unit `G`: `(accel, potential, error)`.
    pub fn field(&self, p: DVec3) -> (DVec3, f64, f64) {
        let (mut acc, mut phi, mut err) = (DVec3::ZERO, 0.0, 0.0);
        let rspan = (CHUNK * REGION) as f64;
        let rhalf = rspan * 0.5 * 3f64.sqrt();
        for (key, rm) in &self.regions {
            let c = DVec3::new(key.0 as f64, key.1 as f64, key.2 as f64) * rspan + DVec3::splat(rspan * 0.5);
            let dist = (p - c).length();
            if dist - rhalf >= R_G {
                continue;
            }
            if rhalf < THETA * dist {
                let dip = DVec3::new(rm.moment2[0] as f64, rm.moment2[1] as f64, rm.moment2[2] as f64) * 0.5;
                let (a, f) = multipole(c, rm.mass as f64, dip, p);
                acc += a;
                phi += f;
                err += truncation(rm.abs, rhalf, dist);
                continue;
            }
            for chunk in &rm.chunks {
                let cm = &self.chunks[chunk];
                let cs = CHUNK as f64;
                let origin = DVec3::new(chunk.0 as f64, chunk.1 as f64, chunk.2 as f64) * cs;
                let c = origin + DVec3::splat(cs * 0.5);
                let chalf = cs * 0.5 * 3f64.sqrt();
                let dist = (p - c).length();
                if chalf < THETA * dist {
                    let dip = DVec3::new(cm.moment2[0] as f64, cm.moment2[1] as f64, cm.moment2[2] as f64) * 0.5;
                    let (a, f) = multipole(c, cm.mass as f64, dip, p);
                    acc += a;
                    phi += f;
                    err += truncation(cm.abs, chalf, dist);
                    continue;
                }
                for &(index, m) in &cm.cells {
                    let i = index as i32;
                    let l = DVec3::new((i % CHUNK) as f64, (i / (CHUNK * CHUNK)) as f64, (i / CHUNK % CHUNK) as f64);
                    let (a, f) = kernel::point(origin + l + DVec3::splat(0.5), m as f64, p);
                    acc += a;
                    phi += f;
                }
            }
        }
        (acc, phi, err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direct(cells: &HashMap<Key, i64>, p: DVec3) -> DVec3 {
        cells
            .iter()
            .map(|(&(x, y, z), &m)| kernel::point(DVec3::new(x as f64, y as f64, z as f64) + DVec3::splat(0.5), m as f64, p).0)
            .sum()
    }

    /// A tiny deterministic generator for test edits.
    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *seed >> 33
    }

    fn random_edits(n: usize, seed: u64, spread: i64) -> Vec<(Key, i32)> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                let c = |s: &mut u64| (lcg(s) as i64 % (2 * spread) - spread) as i32;
                let cell = (c(&mut s), c(&mut s), c(&mut s));
                let delta = (lcg(&mut s) % 13) as i32 - 6;
                (cell, delta)
            })
            .collect()
    }

    #[test]
    fn ledger_field_matches_direct_summation_within_its_error() {
        let edits = random_edits(3000, 7, 600);
        let mut ledger = Ledger::default();
        let mut cells: HashMap<Key, i64> = HashMap::new();
        for &(cell, d) in &edits {
            ledger.record(cell, d);
            *cells.entry(cell).or_default() += d as i64;
        }
        cells.retain(|_, m| *m != 0);
        for p in [DVec3::new(0.3, 0.7, 0.1), DVec3::new(2000.0, -50.0, 900.0), DVec3::new(-10.0, 400.0, 30.0), DVec3::new(9.0e4, 0.0, 0.0)] {
            let (a, _, err) = ledger.field(p);
            let want = direct(&cells, p);
            assert!((a - want).length() <= err + 1e-9 * want.length().max(1e-12), "{p}: {a} vs {want} (err {err})");
        }
    }

    #[test]
    fn edit_order_does_not_change_the_state() {
        let edits = random_edits(2000, 11, 200);
        let mut a = Ledger::default();
        let mut b = Ledger::default();
        for &(c, d) in &edits {
            a.record(c, d);
        }
        for &(c, d) in edits.iter().rev() {
            b.record(c, d);
        }
        assert_eq!(a.chunks, b.chunks);
        assert_eq!(a.regions, b.regions);
        let p = DVec3::new(33.0, -7.5, 120.25);
        assert_eq!(a.field(p).0, b.field(p).0);
    }

    #[test]
    fn undoing_every_edit_empties_the_ledger_exactly() {
        let edits = random_edits(1500, 3, 300);
        let mut l = Ledger::default();
        for &(c, d) in &edits {
            l.record(c, d);
        }
        assert!(!l.is_empty());
        for &(c, d) in &edits {
            l.record(c, -d);
        }
        assert!(l.is_empty() && l.regions.is_empty());
        assert_eq!(l.field(DVec3::new(1.0, 2.0, 3.0)).0, DVec3::ZERO);
    }

    #[test]
    fn a_zero_mass_dipole_still_pulls() {
        // Mining one cell and placing it next door: zero net mass, a real dipole.
        let mut l = Ledger::default();
        l.record((0, 0, 0), -5);
        l.record((1, 0, 0), 5);
        let (a, _, _) = l.field(DVec3::new(40.0, 0.0, 0.0));
        assert!(a.x < 0.0 && a.length() > 0.0, "the moved mass is closer, so the pull grows toward it: {a}");
    }
}
