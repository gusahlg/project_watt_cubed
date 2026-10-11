//! InfiniteDiffusion v5: integer fields refined by local rules in Jacobi passes.
//!
//! A field is one value per cell of a [`Topology`]. A [`Rule`] computes a cell's next value from the
//! previous pass alone (its own value and its neighbours'), so the order cells, tiles or threads run
//! in cannot change a byte, and a tile padded by `passes · RADIUS` cells equals the whole grid bit
//! for bit ([`tile`]). There is no learned denoiser: the "scores" are small integer stencils.
//! Generation built on this crate uses integers and IEEE basic operations only, no libm.
#![forbid(unsafe_code)]

mod noise;
mod tile;
mod topology;

pub use noise::{
    div_floor, fbm3_q16, hash32_3, isqrt_u64, lerp_q16, mul_q16, rem_floor, ridged_q16, smoothstep_q16, uniform_q16,
    value_noise3_q16, Fbm, NoiseBox, HALF, ONE,
};
pub use tile::{point, tile};
pub use topology::{Box3, FACE_AXES, MAX_NEIGHBOURS, Sphere, Topology};

/// Fewest cells a thread is given; smaller grids run on fewer threads.
const MIN_SPLIT: usize = 4096;

/// One local update.
pub trait Rule<C: Copy>: Sync {
    /// How many cells away one pass reads (1 for a rule that reads only `nb`).
    const RADIUS: u8;
    /// Whether `apply` reads `nb` (a rule with its own neighbour lists leaves the copy out).
    const NEIGHBOURS: bool = true;
    /// The next value of cell `i` from the previous pass `prev` and the cell's neighbours `nb`.
    fn apply(&self, prev: &[C], i: usize, nb: &[u32]) -> C;
}

/// A field with its second buffer, for running several rules in a row without reallocating.
pub struct Field<C> {
    cells: Vec<C>,
    spare: Vec<C>,
}

impl<C: Copy + Send + Sync> Field<C> {
    /// A field holding `cells`.
    pub fn new(cells: Vec<C>) -> Self {
        let spare = cells.clone();
        Self { cells, spare }
    }

    /// The current values.
    pub fn cells(&self) -> &[C] {
        &self.cells
    }

    /// The current values, for a global pass between rules.
    pub fn cells_mut(&mut self) -> &mut [C] {
        &mut self.cells
    }

    /// The current values, consuming the field.
    pub fn into_cells(self) -> Vec<C> {
        self.cells
    }

    /// Run `passes` Jacobi passes of `rule` on up to `threads` threads. Each pass writes the spare
    /// buffer from the current one; threads write disjoint index ranges, so the bytes do not depend
    /// on the thread count. Glued copies take their owners' values at the end.
    pub fn run<T: Topology, R: Rule<C>>(&mut self, top: &T, rule: &R, passes: u32, threads: usize) {
        assert_eq!(self.cells.len(), top.len(), "one value per cell");
        if passes == 0 {
            return;
        }
        let threads = threads.clamp(1, (self.cells.len() / MIN_SPLIT).max(1));
        if threads == 1 || passes == 1 {
            for _ in 0..passes {
                pass(top, &self.cells, &mut self.spare, rule, threads);
                std::mem::swap(&mut self.cells, &mut self.spare);
            }
        } else {
            self.run_workers(top, rule, passes, threads);
        }
        for &(copy, owner) in top.glue() {
            self.cells[copy as usize] = self.cells[owner as usize];
        }
    }

    /// Several passes on persistent workers: each computes its index range into a local buffer
    /// while every worker reads the shared previous buffer, then (after a barrier) copies it into the
    /// shared next buffer, then waits again. No threads are spawned per pass.
    fn run_workers<T: Topology, R: Rule<C>>(&mut self, top: &T, rule: &R, passes: u32, threads: usize) {
        use std::sync::{Barrier, RwLock};
        let len = self.cells.len();
        let chunk = len.div_ceil(threads);
        let bufs = [RwLock::new(std::mem::take(&mut self.cells)), RwLock::new(std::mem::take(&mut self.spare))];
        let barrier = Barrier::new(threads);
        let work = |t: usize| {
            let range = t * chunk..((t + 1) * chunk).min(len);
            let mut local = bufs[0].read().expect("field buffer")[range.clone()].to_vec();
            for k in 0..passes as usize {
                {
                    let prev = bufs[k % 2].read().expect("field buffer");
                    fill(top, &prev, &mut local, range.start, rule);
                }
                barrier.wait();
                bufs[1 - k % 2].write().expect("field buffer")[range.clone()].copy_from_slice(&local);
                barrier.wait();
            }
        };
        std::thread::scope(|s| {
            for t in 1..threads {
                let work = &work;
                s.spawn(move || work(t));
            }
            work(0);
        });
        let [a, b] = bufs.map(|l| l.into_inner().expect("field buffer"));
        (self.cells, self.spare) = if passes % 2 == 1 { (b, a) } else { (a, b) };
    }
}

/// [`Field::run`] on a plain vector.
pub fn run<C, T, R>(top: &T, cells: &mut Vec<C>, rule: &R, passes: u32, threads: usize)
where
    C: Copy + Send + Sync,
    T: Topology,
    R: Rule<C>,
{
    let mut f = Field::new(std::mem::take(cells));
    f.run(top, rule, passes, threads);
    *cells = f.into_cells();
}

/// `f(i)` for every index below `len`, on up to `threads` threads over disjoint ranges (a prior, or
/// any per-cell pass that reads no neighbours).
pub fn map<C, F>(len: usize, f: F, threads: usize) -> Vec<C>
where
    C: Copy + Default + Send,
    F: Fn(usize) -> C + Sync,
{
    let mut out = vec![C::default(); len];
    let threads = threads.clamp(1, (len / MIN_SPLIT).max(1));
    let chunk = len.div_ceil(threads).max(1);
    let fill = &|k: usize, part: &mut [C]| {
        for (j, o) in part.iter_mut().enumerate() {
            *o = f(k * chunk + j);
        }
    };
    std::thread::scope(|s| {
        let mut parts = out.chunks_mut(chunk).enumerate();
        let first = parts.next();
        for (k, part) in parts {
            s.spawn(move || fill(k, part));
        }
        if let Some((_, part)) = first {
            fill(0, part);
        }
    });
    out
}

fn pass<C, T, R>(top: &T, prev: &[C], next: &mut [C], rule: &R, threads: usize)
where
    C: Copy + Send + Sync,
    T: Topology,
    R: Rule<C>,
{
    if threads == 1 {
        fill(top, prev, next, 0, rule);
        return;
    }
    let chunk = prev.len().div_ceil(threads);
    std::thread::scope(|s| {
        let mut parts = next.chunks_mut(chunk).enumerate();
        let (_, first) = parts.next().expect("a non-empty field");
        for (k, part) in parts {
            s.spawn(move || fill(top, prev, part, k * chunk, rule));
        }
        fill(top, prev, first, 0, rule);
    });
}

fn fill<C, T, R>(top: &T, prev: &[C], out: &mut [C], from: usize, rule: &R)
where
    C: Copy,
    T: Topology,
    R: Rule<C>,
{
    let mut nb = [0u32; MAX_NEIGHBOURS];
    for (k, o) in out.iter_mut().enumerate() {
        let i = from + k;
        if top.owns(i) {
            let n = if R::NEIGHBOURS { top.neighbours(i, &mut nb) } else { 0 };
            *o = rule.apply(prev, i, &nb[..n]);
        }
    }
}

#[cfg(test)]
mod tests;
