//! The planet's interior below the crust: the Deep, the Underdark, mantle bubbles and the Heart.
//!
//! Depth below the local surface is `surface height + half − ||rel||∞` (face-normal altitude and
//! the L∞ distance to the centre are the same coordinate). Features sit on site grids sized to
//! themselves and stay inside their cells, so a chunk that misses every conservative bound is
//! untouched bulk. `depth` selects a band only; inside one band the block does not depend on which
//! value in that band the caller passes.

mod cavern;
mod chamber;
mod glow;
mod hall;
mod mass;
mod mushroom;

use super::cosmos::Body;
use super::cube;
use super::noise::{hash3, unit};
use super::{Materials, MAX_GROUND, MIN_GROUND};
use crate::block::registry::{AIR, BlockId};
use std::sync::Arc;

/// Deep band: below the crust, down to this depth.
pub(super) const DEEP_HI: i32 = 4_000;
/// Underdark band: below [`DEEP_HI`], down to this depth.
pub(super) const UNDER_HI: i32 = 60_000;

/// Face normal of the body-relative vector (dominant axis, the cube's tie break).
#[derive(Clone, Copy)]
pub(super) struct Up {
    pub axis: usize,
    pub sign: i32,
}

impl Up {
    pub fn of(rel: [i64; 3]) -> Self {
        let (x, y, z) = cube::face_of(rel).normal();
        let n = [x, y, z];
        let axis = n.iter().position(|&c| c != 0).unwrap_or(1);
        Self { axis, sign: n[axis] }
    }

    /// Increases toward the face.
    #[inline]
    pub fn outward(self, rel: [i64; 3]) -> i64 {
        rel[self.axis] * i64::from(self.sign)
    }
}

#[derive(Clone, Copy)]
pub(super) struct Ctx<'a> {
    pub seed: u32,
    pub scale: f32,
    pub half: i64,
    pub m: &'a Materials,
}

/// Distance below the face plane. True depth is this plus the column's surface height.
#[inline]
pub(super) fn plane_depth(half: i64, rel: [i64; 3]) -> i64 {
    half - rel.iter().copied().map(i64::abs).max().unwrap_or(0)
}

#[inline]
pub(super) fn dist2(a: [i64; 3], b: [i64; 3]) -> i64 {
    let mut s = 0i64;
    for i in 0..3 {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

#[inline]
pub(super) fn tangent(rel: [i64; 3], axis: usize) -> (i64, i64) {
    match axis {
        0 => (rel[1], rel[2]),
        1 => (rel[0], rel[2]),
        _ => (rel[0], rel[1]),
    }
}

#[inline]
pub(super) fn place(axis: usize, sign: i32, t: (i64, i64), outward_v: i64) -> [i64; 3] {
    let mut p = [0i64; 3];
    p[axis] = outward_v * i64::from(sign);
    match axis {
        0 => {
            p[1] = t.0;
            p[2] = t.1;
        }
        1 => {
            p[0] = t.0;
            p[2] = t.1;
        }
        _ => {
            p[0] = t.0;
            p[1] = t.1;
        }
    }
    p
}

#[inline]
pub(super) fn roll(h: u32, p: f32) -> bool {
    p > 0.0 && unit(h) < p
}

#[inline]
pub(super) fn idx_i32(v: i64) -> Option<i32> {
    i32::try_from(v).ok()
}

/// Centre of a ball of radius `r` kept inside the cell.
pub(super) fn confine(origin: i64, cell: i64, r: i64, u: f32) -> i64 {
    let span = cell - 2 * r;
    if span <= 1 {
        origin + cell / 2
    } else {
        origin + r + (f64::from(u) * (span - 1) as f64) as i64
    }
}

pub(super) fn for_cells(lo: [i64; 3], hi: [i64; 3], cell: i64, mut f: impl FnMut([i64; 3]) -> bool) -> bool {
    let s = |v: i64| v.div_euclid(cell);
    for z in s(lo[2])..=s(hi[2]) {
        for y in s(lo[1])..=s(hi[1]) {
            for x in s(lo[0])..=s(hi[0]) {
                if f([x, y, z]) {
                    return true;
                }
            }
        }
    }
    false
}

/// Cells that can hold a confined shaft meeting the box.
///
/// A shaft stays inside its own cell in the tangent plane (`confine` insets the centre by the
/// radius), and it only travels along the face normal. Every such cell shares a tangent index
/// with the box and lies within `pad` cells along one axis.
pub(super) fn for_shafts(lo: [i64; 3], hi: [i64; 3], cell: i64, pad: i64, mut f: impl FnMut([i64; 3]) -> bool) -> bool {
    let s = |v: i64| v.div_euclid(cell);
    for axis in 0..3 {
        let (t, u) = match axis {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        for ia in (s(lo[axis]) - pad)..=(s(hi[axis]) + pad) {
            for it in s(lo[t])..=s(hi[t]) {
                for iu in s(lo[u])..=s(hi[u]) {
                    let mut idx = [0i64; 3];
                    idx[axis] = ia;
                    idx[t] = it;
                    idx[u] = iu;
                    if f(idx) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Shallowest and deepest true depth any surface height can give a point of the box.
pub(super) fn depth_span(half: i64, lo: [i64; 3], hi: [i64; 3]) -> (i64, i64) {
    let min_pd = half - max_linf(lo, hi);
    let max_pd = half - cube::min_reach([0, 0, 0], lo, hi);
    (min_pd + i64::from(MIN_GROUND), max_pd + i64::from(MAX_GROUND))
}

/// Every surface height puts the whole box inside `(band_lo, band_hi]`.
pub(super) fn band_certain(half: i64, lo: [i64; 3], hi: [i64; 3], band_lo: i32, band_hi: i32) -> bool {
    let (shallow, deep) = depth_span(half, lo, hi);
    shallow > i64::from(band_lo) && deep <= i64::from(band_hi)
}

#[inline]
pub(super) fn min_outward(up: Up, lo: [i64; 3], hi: [i64; 3]) -> i64 {
    if up.sign > 0 { lo[up.axis] } else { -hi[up.axis] }
}

pub(super) fn hash_rel(salt: u32, rel: [i64; 3]) -> u32 {
    match (idx_i32(rel[0]), idx_i32(rel[1]), idx_i32(rel[2])) {
        (Some(x), Some(y), Some(z)) => hash3(salt, x, y, z),
        _ => 0,
    }
}

/// The closest integer point of `[lo, hi]` to `c` lies inside the sphere.
pub(super) fn sphere_hits(c: [i64; 3], r: i64, lo: [i64; 3], hi: [i64; 3]) -> bool {
    if r <= 0 {
        return false;
    }
    let p = std::array::from_fn(|a| c[a].clamp(lo[a], hi[a]));
    dist2(p, c) <= r * r
}

pub(super) fn box_hits(a0: [i64; 3], a1: [i64; 3], b0: [i64; 3], b1: [i64; 3]) -> bool {
    (0..3).all(|i| a0[i] <= b1[i] && b0[i] <= a1[i])
}

fn max_linf(lo: [i64; 3], hi: [i64; 3]) -> i64 {
    let mut m = 0i64;
    for i in 0..8 {
        let p = [
            if i & 1 == 0 { lo[0] } else { hi[0] },
            if i & 2 == 0 { lo[1] } else { hi[1] },
            if i & 4 == 0 { lo[2] } else { hi[2] },
        ];
        m = m.max(p.iter().copied().map(i64::abs).max().unwrap_or(0));
    }
    m
}

/// Whether any surface height in range puts some point of the box in `(lo, hi]`.
fn band_possible(half: i64, lo: [i64; 3], hi: [i64; 3], band_lo: i32, band_hi: i32) -> bool {
    let min_pd = half - max_linf(lo, hi);
    let max_pd = half - cube::min_reach([0, 0, 0], lo, hi);
    let shallow = min_pd + i64::from(MIN_GROUND);
    let deep = max_pd + i64::from(MAX_GROUND);
    deep > i64::from(band_lo) && shallow <= i64::from(band_hi)
}

/// Interior features for one generator. Placement is a function of the body's own seed.
pub(super) struct Deep {
    scale: f32,
    m: Arc<Materials>,
}

impl Deep {
    pub fn new(scale: f32, m: Arc<Materials>) -> Self {
        Self { scale: scale.clamp(0.0, 2.0), m }
    }

    fn ctx<'a>(&'a self, body: &Body) -> Ctx<'a> {
        Ctx { seed: body.seed, scale: self.scale, half: cube::half_of(body), m: &self.m }
    }

    /// Every voxel of the box is air (a hollow this chunk sits inside, clear of dressing).
    pub fn all_air(&self, body: &Body, lo: [i64; 3], hi: [i64; 3]) -> bool {
        let ctx = self.ctx(body);
        if hall::hits(&ctx, lo, hi) {
            return false;
        }
        mass::heart_all_air(&ctx, lo, hi)
            || mass::bubble_all_air(&ctx, lo, hi)
            || cavern::all_air(&ctx, lo, hi)
            || chamber::all_air(&ctx, lo, hi)
    }

    /// A feature's conservative bound meets the box.
    pub fn hits(&self, body: &Body, lo: [i64; 3], hi: [i64; 3]) -> bool {
        let ctx = self.ctx(body);
        mass::heart_hits(&ctx, lo, hi)
            || mass::bubble_hits(&ctx, lo, hi)
            || (band_possible(ctx.half, lo, hi, cube::CRUST, DEEP_HI) && (hall::hits(&ctx, lo, hi) || cavern::hits(&ctx, lo, hi)))
            || (band_possible(ctx.half, lo, hi, DEEP_HI, UNDER_HI) && chamber::hits(&ctx, lo, hi))
            || (band_possible(ctx.half, lo, hi, cube::CRUST, DEEP_HI) && chamber::shaft_hits(&ctx, lo, hi))
    }

    #[inline]
    pub fn might(&self, body: &Body, rel: [i64; 3]) -> bool {
        self.all_air(body, rel, rel) || self.hits(body, rel, rel)
    }

    /// The block at `rel`, or `bulk` where no feature writes.
    pub fn block(&self, body: &Body, rel: [i64; 3], depth: i32, bulk: BlockId) -> BlockId {
        let ctx = self.ctx(body);
        if let Some(id) = mass::heart_block(&ctx, rel) {
            return id;
        }
        if let Some(id) = mass::bubble_block(&ctx, rel) {
            return id;
        }
        if depth > cube::CRUST && depth <= DEEP_HI {
            if let Some(id) = hall::block(&ctx, rel) {
                return id;
            }
            if let Some(id) = cavern::block(&ctx, rel) {
                return id;
            }
            if let Some(AIR) = chamber::shaft_block(&ctx, rel) {
                return AIR;
            }
        } else if depth > DEEP_HI && depth <= UNDER_HI
            && let Some(id) = chamber::block(&ctx, rel)
        {
            return id;
        }
        bulk
    }
}

/// Mean air fraction of the deep shell and of the underdark shell, at this `deep` scale.
pub(super) fn porosity(scale: f32) -> (f64, f64) {
    (cavern::porosity(scale) + hall::porosity(scale), chamber::porosity(scale))
}

/// The analytic masses [`super::cosmos`] subtracts. Re-exported so the oracle can see it.
pub(super) use mass::apply;

/// Coarse-field sample shared by site radii (one 3-D noise value per site).
pub(super) fn field_at(seed: u32, p: [i64; 3]) -> f32 {
    super::noise::perlin3(seed, p[0] as f64 / 480.0, p[1] as f64 / 480.0, p[2] as f64 / 480.0)
}

pub(super) fn hash_site(seed: u32, idx: [i64; 3]) -> Option<u32> {
    Some(hash3(seed, idx_i32(idx[0])?, idx_i32(idx[1])?, idx_i32(idx[2])?))
}

/// Palette roles that carry block light. Crystal is clear and does not emit.
#[cfg(test)]
pub(super) fn emits_light(m: &Materials, id: BlockId) -> bool {
    id == m.lamp || id == m.glowcap || id == m.glowshroom || id == m.magma || id == m.star || id == m.core
}

/// Emissive cells, queried as "any within Euclidean 12".
///
/// A sparse dressing stays in buckets of width 12. A lake or a crystal lining is a solid region:
/// those cells go into a bitset, and the query walks shells outward so a neighbour hits immediately.
#[cfg(test)]
struct NearLights {
    buckets: std::collections::HashMap<[i64; 3], Vec<[i64; 3]>>,
    dense: Option<LightBits>,
}

#[cfg(test)]
struct LightBits {
    origin: [i64; 3],
    dim: [i64; 3],
    bits: Vec<u64>,
}

#[cfg(test)]
impl LightBits {
    fn build(lights: &[[i64; 3]]) -> Option<Self> {
        let mut lo = [i64::MAX; 3];
        let mut hi = [i64::MIN; 3];
        for p in lights {
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        let dim = std::array::from_fn(|a| hi[a] - lo[a] + 1);
        if dim.iter().any(|&d| d <= 0 || d > 900) {
            return None;
        }
        let vol = dim[0].checked_mul(dim[1])?.checked_mul(dim[2])?;
        if vol > 300_000_000 {
            return None;
        }
        let mut bits = vec![0u64; (vol as usize + 63) / 64];
        let mut out = Self { origin: lo, dim, bits: Vec::new() };
        for p in lights {
            if let Some(i) = out.index(*p) {
                bits[i >> 6] |= 1u64 << (i & 63);
            }
        }
        out.bits = bits;
        Some(out)
    }

    fn index(&self, p: [i64; 3]) -> Option<usize> {
        let x = p[0] - self.origin[0];
        let y = p[1] - self.origin[1];
        let z = p[2] - self.origin[2];
        if x < 0 || y < 0 || z < 0 || x >= self.dim[0] || y >= self.dim[1] || z >= self.dim[2] {
            return None;
        }
        Some((x + self.dim[0] * (y + self.dim[1] * z)) as usize)
    }

    fn has(&self, p: [i64; 3]) -> bool {
        let Some(i) = self.index(p) else { return false };
        (self.bits[i >> 6] >> (i & 63)) & 1 == 1
    }

    fn covers(&self, p: [i64; 3]) -> bool {
        if self.has(p) {
            return true;
        }
        for rad in 1..=12 {
            let r2 = rad * rad;
            let prev = (rad - 1) * (rad - 1);
            for dz in -rad..=rad {
                for dy in -rad..=rad {
                    let base = dz * dz + dy * dy;
                    if base > r2 {
                        continue;
                    }
                    let max_dx = isqrt_144(r2 - base);
                    let min_dx = if base > prev { 0 } else { isqrt_144(prev - base) + 1 };
                    if min_dx > max_dx {
                        continue;
                    }
                    for sign in [-1i64, 1] {
                        for adx in min_dx..=max_dx {
                            let dx = if adx == 0 { 0 } else { sign * adx };
                            if adx == 0 && sign == 1 {
                                continue;
                            }
                            if self.has([p[0] + dx, p[1] + dy, p[2] + dz]) {
                                return true;
                            }
                            if adx == 0 {
                                break;
                            }
                        }
                    }
                }
            }
        }
        false
    }
}

/// `floor(sqrt(n))` for `n` in `0..=144`.
#[cfg(test)]
fn isqrt_144(n: i64) -> i64 {
    let mut x = 0i64;
    while x < 12 && (x + 1) * (x + 1) <= n {
        x += 1;
    }
    x
}

#[cfg(test)]
impl NearLights {
    const B: i64 = 12;

    fn new(lights: Vec<[i64; 3]>) -> Self {
        // Lakes and linings are dense. A few thousand cap-glows spread over a cavern are not:
        // a bitset query then walks the whole ball for every dark floor cell.
        if lights.len() > 16_384
            && let Some(dense) = LightBits::build(&lights)
        {
            return Self { buckets: std::collections::HashMap::new(), dense: Some(dense) };
        }
        let mut buckets: std::collections::HashMap<[i64; 3], Vec<[i64; 3]>> = std::collections::HashMap::new();
        for p in lights {
            buckets.entry(p.map(|v| v.div_euclid(Self::B))).or_default().push(p);
        }
        Self { buckets, dense: None }
    }

    /// Euclidean distance ≤ 12. Bucket width equals that radius, so a hit is in the 3³ neighbourhood.
    fn covers(&self, p: [i64; 3]) -> bool {
        if let Some(dense) = &self.dense {
            return dense.covers(p);
        }
        let q = p.map(|v| v.div_euclid(Self::B));
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let Some(list) = self.buckets.get(&[q[0] + dx, q[1] + dy, q[2] + dz]) else { continue };
                    for &l in list {
                        if dist2(l, p) <= 12 * 12 {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    fn count(&self, floors: &[[i64; 3]]) -> u32 {
        floors.iter().filter(|&&p| self.covers(p)).count() as u32
    }
}

/// One surveyed hollow: how many floor cells sit within 12 of an emissive block.
#[cfg(test)]
pub(super) struct Cover {
    pub kind: u32,
    pub r: i64,
    pub center: [i64; 3],
    pub up_axis: usize,
    pub up_sign: i32,
    pub floor: u32,
    pub near: u32,
    /// Column stride (1 = every column).
    pub step: i64,
}

#[cfg(test)]
impl Deep {
    pub(super) fn locate_cavern(&self, body: &Body) -> Option<[i64; 3]> {
        cavern::locate(&self.ctx(body))
    }

    pub(super) fn survey_caverns(&self, body: &Body, n: usize) -> Vec<Cover> {
        cavern::survey(&self.ctx(body), n)
    }

    pub(super) fn survey_cavern_at(&self, body: &Body, rel: [i64; 3]) -> Option<Cover> {
        cavern::survey_at(&self.ctx(body), rel)
    }

    pub(super) fn survey_chambers(&self, body: &Body, n: usize) -> Vec<Cover> {
        chamber::survey(&self.ctx(body), n)
    }

    pub(super) fn survey_halls(&self, body: &Body, n: usize) -> Vec<Cover> {
        hall::survey(&self.ctx(body), n)
    }

    /// Smallest cavern of each biome. The assert measures these rather than the largest halls.
    pub(super) fn cover_kinds(&self, body: &Body) -> Vec<Cover> {
        cavern::measure_each_kind(&self.ctx(body))
    }

    /// Smallest underdark chamber, floor band only.
    pub(super) fn cover_chamber(&self, body: &Body) -> Option<Cover> {
        chamber::measure_smallest(&self.ctx(body))
    }

    pub(super) fn locate_hall(&self, body: &Body) -> Option<[i64; 3]> {
        hall::locate(&self.ctx(body))
    }

    pub(super) fn locate_chamber(&self, body: &Body) -> Option<[i64; 3]> {
        chamber::locate(&self.ctx(body))
    }

    pub(super) fn locate_bubble(&self, body: &Body) -> Option<([i64; 3], i64)> {
        mass::locate_bubble(&self.ctx(body))
    }
}
