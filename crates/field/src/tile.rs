//! Tiles of an infinite 3-D lattice with exact aprons. After `k` passes a cell depends only on the
//! priors within `k · RADIUS` cells, so a tile computed on a box padded by that margin equals the
//! infinite field on its interior: no blending, no seams, no shared cache.

use crate::topology::{Box3, Topology};
use crate::{run, Rule};

/// The field after `passes` passes of `rule` on the box of `size` cells at `lo`, with priors from
/// `prior(cell)`. Returned x fastest.
pub fn tile<C, R, P>(rule: &R, prior: P, lo: [i64; 3], size: [u32; 3], passes: u32, threads: usize) -> Vec<C>
where
    C: Copy + Send + Sync,
    R: Rule<C>,
    P: Fn([i64; 3]) -> C,
{
    let pad = passes * R::RADIUS as u32;
    let top = Box3::new(size.map(|s| s + 2 * pad), true);
    let mut cells = Vec::with_capacity(top.len());
    for i in 0..top.len() {
        let p = top.coords(i);
        cells.push(prior(std::array::from_fn(|a| lo[a] - pad as i64 + p[a] as i64)));
    }
    run(&top, &mut cells, rule, passes, threads);
    let mut out = Vec::with_capacity((size[0] * size[1] * size[2]) as usize);
    for z in 0..size[2] {
        for y in 0..size[1] {
            let row = top.index([pad, y + pad, z + pad]);
            out.extend_from_slice(&cells[row..row + size[0] as usize]);
        }
    }
    out
}

/// One lattice cell after `passes` passes: the cone of priors under it, as a one-cell tile.
pub fn point<C, R, P>(rule: &R, prior: P, at: [i64; 3], passes: u32) -> C
where
    C: Copy + Send + Sync,
    R: Rule<C>,
    P: Fn([i64; 3]) -> C,
{
    tile(rule, prior, at, [1, 1, 1], passes, 1)[0]
}
