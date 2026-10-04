//! Surface landmarks. One candidate per site cell, themed by the province feature vector, scaled
//! by the `features` knob. The batch path and the per-voxel path share one site order.

mod bones;
mod craters;
mod crystals;
mod dunes;
mod flora;
mod giants;
mod ice;
mod islands;
mod mushrooms;
mod petrified;
mod sinkholes;
mod spires;
mod volcanic;

use std::cell::RefCell;
use std::sync::Arc;

use super::cube;
use super::noise::{self, hash2, unit};
use super::province::{self, FEAT};
use super::shape::{Column, Shape};
use super::Materials;
use crate::block::registry::{AIR, BlockId};

/// Tallest landmark above the column it stands on, sky islands included.
pub const MAX_ABOVE: i32 = 320;
/// Deepest a landmark may replace ground. Stays inside the crust.
pub const MAX_BELOW: i32 = 72;

/// Eight horizontal steps, then the diagonals. Shared by the families.
pub(super) const DIRS: [(i32, i32); 8] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
];

/// A block a landmark wants to write. `dig` replaces ground; otherwise it only fills air.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stamp {
    pub id: BlockId,
    pub dig: bool,
}

pub(super) struct Ctx<'a> {
    pub s: u32,
    pub scale: f32,
    pub m: &'a Materials,
    shape: &'a Shape,
    cols: Option<(&'a [Column], i32, i32, i32)>,
}

impl<'a> Ctx<'a> {
    fn feats(&self, x: i32, z: i32) -> [f32; FEAT] {
        if let Some(col) = self.saved(x, z) {
            return col.feats;
        }
        let key = self.shape.cache_key();
        feats_cached(key, x, z, || self.shape.feats_at(x, z))
    }

    fn column(&self, x: i32, z: i32) -> Column {
        if let Some(col) = self.saved(x, z) {
            return col;
        }
        let key = self.shape.cache_key();
        column_cached(key, x, z, || self.shape.column(x, z))
    }

    fn saved(&self, x: i32, z: i32) -> Option<Column> {
        let (cols, u0, v0, n) = self.cols?;
        if x < u0 || z < v0 || x >= u0 + n || z >= v0 + n {
            return None;
        }
        let i = (x - u0) as usize + (z - v0) as usize * n as usize;
        Some(cols[i])
    }

    pub(super) fn inland(&self, x: i32, z: i32, reach: i32) -> bool {
        self.shape.inset(x, z) >= i64::from(cube::RIM) + i64::from(reach)
    }
}

/// Keep a site when `roll` falls inside `dens * scale`. A scale of 0 plants nothing.
pub(super) fn keep(scale: f32, dens: f32, roll: u32) -> bool {
    if dens <= 0.0 || scale <= 0.0 {
        return false;
    }
    let t = dens * scale;
    t >= 1.0 || unit(roll) < t
}

/// Anchor inside the site cell `(cx, cz)`, plus the salt that shapes it.
pub(super) fn site_anchor(salt: u32, cell: i32, cx: i32, cz: i32) -> (i32, i32, u32) {
    let h = hash2(salt, cx, cz);
    let room = (cell - 2).max(1) as u32;
    let x = cx.wrapping_mul(cell).wrapping_add(1).wrapping_add((h % room) as i32);
    let z = cz.wrapping_mul(cell).wrapping_add(1).wrapping_add(((h >> 8) % room) as i32);
    (x, z, h)
}

pub(super) fn disk(dx: i32, dz: i32, r: i32) -> bool {
    if r < 0 {
        return false;
    }
    let (dx, dz, r) = (i64::from(dx), i64::from(dz), i64::from(r));
    dx * dx + dz * dz <= r * r
}

#[derive(Clone, Copy)]
struct Slot<T: Copy> {
    key: u64,
    cx: i32,
    cz: i32,
    live: bool,
    val: Option<T>,
}

fn feats_cached(key: u64, cx: i32, cz: i32, f: impl FnOnce() -> [f32; FEAT]) -> [f32; FEAT] {
    thread_local! {
        static RING: RefCell<Vec<Slot<[f32; FEAT]>>> = RefCell::new(Vec::new());
    }
    if let Some(v) = RING.with(|r| lookup(r.borrow().as_slice(), key, cx, cz)) {
        return v;
    }
    let val = f();
    RING.with(|r| remember(r, 2048, key, cx, cz, val));
    val
}

fn column_cached(key: u64, cx: i32, cz: i32, f: impl FnOnce() -> Column) -> Column {
    thread_local! {
        static RING: RefCell<Vec<Slot<Column>>> = RefCell::new(Vec::new());
    }
    if let Some(v) = RING.with(|r| lookup(r.borrow().as_slice(), key, cx, cz)) {
        return v;
    }
    let val = f();
    RING.with(|r| remember(r, 1024, key, cx, cz, val));
    val
}

fn remember<T: Copy>(ring: &RefCell<Vec<Slot<T>>>, n: usize, key: u64, cx: i32, cz: i32, val: T) {
    let mut g = ring.borrow_mut();
    if g.is_empty() {
        g.resize(n, Slot { key: 0, cx: 0, cz: 0, live: false, val: None });
    }
    let mask = n - 1;
    let i0 = mix(key, cx, cz) & mask;
    let mut slot = i0;
    for k in 0..4 {
        let i = (i0 + k) & mask;
        let s = &g[i];
        if !s.live || (s.key == key && s.cx == cx && s.cz == cz) {
            slot = i;
            break;
        }
    }
    g[slot] = Slot { key, cx, cz, live: true, val: Some(val) };
}

fn lookup<T: Copy>(slots: &[Slot<T>], key: u64, cx: i32, cz: i32) -> Option<T> {
    if slots.is_empty() {
        return None;
    }
    let mask = slots.len() - 1;
    let i0 = mix(key, cx, cz) & mask;
    for k in 0..4 {
        let s = &slots[(i0 + k) & mask];
        if s.live && s.key == key && s.cx == cx && s.cz == cz {
            return s.val;
        }
    }
    None
}

fn mix(key: u64, cx: i32, cz: i32) -> usize {
    let mut h = key ^ (cx as u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= (cz as u32 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h.wrapping_mul(0x1000_0000_01B3) as usize
}

struct Rec<T> {
    spec: T,
    cx: i32,
    cz: i32,
}

fn gather<T, S>(ctx: &Ctx, cell: i32, reach: i32, u0: i32, v0: i32, n: i32, spawn: S) -> Vec<Rec<T>>
where
    T: Copy + 'static,
    S: Fn(&Ctx, i32, i32) -> Option<T>,
{
    let x1 = u0 + n - 1;
    let z1 = v0 + n - 1;
    let c0x = (u0 - reach).div_euclid(cell);
    let c1x = (x1 + reach).div_euclid(cell);
    let c0z = (v0 - reach).div_euclid(cell);
    let c1z = (z1 + reach).div_euclid(cell);
    let mut out = Vec::new();
    for cz in c0z..=c1z {
        for cx in c0x..=c1x {
            if let Some(spec) = spawn(ctx, cx, cz) {
                out.push(Rec { spec, cx, cz });
            }
        }
    }
    out
}

fn in_window(cx: i32, cz: i32, x: i32, z: i32, cell: i32, reach: i32) -> bool {
    let c0x = (x - reach).div_euclid(cell);
    let c1x = (x + reach).div_euclid(cell);
    let c0z = (z - reach).div_euclid(cell);
    let c1z = (z + reach).div_euclid(cell);
    (c0x..=c1x).contains(&cx) && (c0z..=c1z).contains(&cz)
}

fn hit_recs<T: Copy>(
    recs: &[Rec<T>],
    m: &Materials,
    x: i32,
    y: i32,
    z: i32,
    ground: i32,
    cell: i32,
    reach: i32,
    bounds: fn(&T) -> (i32, i32),
    paint: fn(&T, &Materials, i32, i32, i32, i32) -> Option<Stamp>,
) -> Option<Stamp> {
    for rec in recs {
        if !in_window(rec.cx, rec.cz, x, z, cell, reach) {
            continue;
        }
        let (y0, y1) = bounds(&rec.spec);
        if y < y0 || y >= y1 {
            continue;
        }
        if let Some(st) = paint(&rec.spec, m, x, y, z, ground) {
            return Some(st);
        }
    }
    None
}

fn hit_spawn<T, S>(
    ctx: &Ctx,
    x: i32,
    y: i32,
    z: i32,
    ground: i32,
    cell: i32,
    reach: i32,
    spawn: S,
    bounds: fn(&T) -> (i32, i32),
    paint: fn(&T, &Materials, i32, i32, i32, i32) -> Option<Stamp>,
) -> Option<Stamp>
where
    T: Copy + 'static,
    S: Fn(&Ctx, i32, i32) -> Option<T>,
{
    let c0x = (x - reach).div_euclid(cell);
    let c1x = (x + reach).div_euclid(cell);
    let c0z = (z - reach).div_euclid(cell);
    let c1z = (z + reach).div_euclid(cell);
    for cz in c0z..=c1z {
        for cx in c0x..=c1x {
            let Some(spec) = spawn(ctx, cx, cz) else { continue };
            let (y0, y1) = bounds(&spec);
            if y < y0 || y >= y1 {
                continue;
            }
            if let Some(st) = paint(&spec, ctx.m, x, y, z, ground) {
                return Some(st);
            }
        }
    }
    None
}

/// Themed landmarks for one face.
pub struct Features {
    s: u32,
    scale: f32,
    m: Arc<Materials>,
}

impl Features {
    pub fn new(s: u32, scale: f32, m: Arc<Materials>) -> Self {
        Self { s, scale, m }
    }

    fn ctx<'a>(&'a self, shape: &'a Shape, cols: Option<(&'a [Column], i32, i32, i32)>) -> Ctx<'a> {
        Ctx { s: self.s, scale: self.scale, m: &self.m, shape, cols }
    }

    fn hit(&self, ctx: &Ctx, x: i32, y: i32, z: i32, ground: i32) -> Option<Stamp> {
        if let Some(st) = hit_spawn(ctx, x, y, z, ground, craters::CELL, craters::REACH, craters::spawn, craters::bounds, craters::paint) {
            return Some(st);
        }
        if let Some(st) = hit_spawn(ctx, x, y, z, ground, sinkholes::CELL, sinkholes::REACH, sinkholes::spawn, sinkholes::bounds, sinkholes::paint) {
            return Some(st);
        }
        if let Some(st) = hit_spawn(ctx, x, y, z, ground, volcanic::CELL, volcanic::REACH, volcanic::spawn, volcanic::bounds, volcanic::paint) {
            return Some(st);
        }
        if let Some(st) = hit_spawn(ctx, x, y, z, ground, islands::CELL, islands::REACH, islands::spawn, islands::bounds, islands::paint) {
            return Some(st);
        }
        if let Some(st) = hit_spawn(ctx, x, y, z, ground, spires::CELL, spires::REACH, spires::spawn, spires::bounds, spires::paint) {
            return Some(st);
        }
        if let Some(st) = hit_spawn(ctx, x, y, z, ground, dunes::CELL, dunes::REACH, dunes::spawn, dunes::bounds, dunes::paint) {
            return Some(st);
        }
        if let Some(st) = hit_spawn(ctx, x, y, z, ground, ice::CELL, ice::REACH, ice::spawn, ice::bounds, ice::paint) {
            return Some(st);
        }
        if let Some(st) = hit_spawn(ctx, x, y, z, ground, bones::CELL, bones::REACH, bones::spawn, bones::bounds, bones::paint) {
            return Some(st);
        }
        if let Some(st) = hit_spawn(ctx, x, y, z, ground, giants::CELL, giants::REACH, giants::spawn, giants::bounds, giants::paint) {
            return Some(st);
        }
        if let Some(st) = hit_spawn(ctx, x, y, z, ground, petrified::CELL, petrified::REACH, petrified::spawn, petrified::bounds, petrified::paint) {
            return Some(st);
        }
        if let Some(st) = hit_spawn(ctx, x, y, z, ground, mushrooms::CELL, mushrooms::REACH, mushrooms::spawn, mushrooms::bounds, mushrooms::paint) {
            return Some(st);
        }
        if let Some(st) = hit_spawn(ctx, x, y, z, ground, crystals::CELL, crystals::REACH, crystals::spawn, crystals::bounds, crystals::paint) {
            return Some(st);
        }
        hit_spawn(ctx, x, y, z, ground, flora::CELL, flora::REACH, flora::spawn, flora::bounds, flora::paint)
    }

    /// The landmark at one cell, if it claims the cell. `ground` is that column's surface.
    pub(super) fn block_at(&self, shape: &Shape, x: i32, y: i32, z: i32, ground: i32) -> Option<Stamp> {
        let ctx = self.ctx(shape, None);
        self.hit(&ctx, x, y, z, ground)
    }

    /// True when a landmark has removed the ground cell under this column.
    pub(super) fn cuts_surface(&self, shape: &Shape, x: i32, ground: i32, z: i32) -> bool {
        matches!(self.block_at(shape, x, ground - 1, z, ground), Some(st) if st.dig && st.id == AIR)
    }

    /// Per-column "the surface cell was dug away". Empty when this square has no carver.
    pub(super) fn surface_open(&self, shape: &Shape, cols: &[Column], u0: i32, v0: i32, n: i32) -> Vec<bool> {
        let ctx = self.ctx(shape, Some((cols, u0, v0, n)));
        let mut open = vec![false; (n * n) as usize];
        let craters = gather(&ctx, craters::CELL, craters::REACH, u0, v0, n, craters::spawn);
        let sinks = gather(&ctx, sinkholes::CELL, sinkholes::REACH, u0, v0, n, sinkholes::spawn);
        let dunes = gather(&ctx, dunes::CELL, dunes::REACH, u0, v0, n, dunes::spawn);
        let ice = gather(&ctx, ice::CELL, ice::REACH, u0, v0, n, ice::spawn);
        if craters.is_empty() && sinks.is_empty() && dunes.is_empty() && ice.is_empty() {
            return open;
        }
        for lv in 0..n {
            for lu in 0..n {
                let x = u0 + lu;
                let z = v0 + lv;
                let g = cols[(lu + lv * n) as usize].height;
                let cut = matches!(self.hit(&ctx, x, g - 1, z, g), Some(st) if st.dig && st.id == AIR);
                open[(lu + lv * n) as usize] = cut;
            }
        }
        open
    }

    /// Landmark cells in the square `(u0, v0)` and the half-open height window `[y0, y1)`.
    /// `(u, h, v, id, dig)`, in column-major order. Same winner as [`block_at`](Self::block_at).
    pub(super) fn blocks_in(
        &self,
        shape: &Shape,
        cols: &[Column],
        u0: i32,
        v0: i32,
        n: i32,
        y0: i32,
        y1: i32,
    ) -> Vec<(i32, i32, i32, BlockId, bool)> {
        let ctx = self.ctx(shape, Some((cols, u0, v0, n)));
        let craters = gather(&ctx, craters::CELL, craters::REACH, u0, v0, n, craters::spawn);
        let sinks = gather(&ctx, sinkholes::CELL, sinkholes::REACH, u0, v0, n, sinkholes::spawn);
        let volcanic = gather(&ctx, volcanic::CELL, volcanic::REACH, u0, v0, n, volcanic::spawn);
        let islands = gather(&ctx, islands::CELL, islands::REACH, u0, v0, n, islands::spawn);
        let spires = gather(&ctx, spires::CELL, spires::REACH, u0, v0, n, spires::spawn);
        let dunes = gather(&ctx, dunes::CELL, dunes::REACH, u0, v0, n, dunes::spawn);
        let ice = gather(&ctx, ice::CELL, ice::REACH, u0, v0, n, ice::spawn);
        let bones = gather(&ctx, bones::CELL, bones::REACH, u0, v0, n, bones::spawn);
        let giants = gather(&ctx, giants::CELL, giants::REACH, u0, v0, n, giants::spawn);
        let petrified = gather(&ctx, petrified::CELL, petrified::REACH, u0, v0, n, petrified::spawn);
        let mushrooms = gather(&ctx, mushrooms::CELL, mushrooms::REACH, u0, v0, n, mushrooms::spawn);
        let crystals = gather(&ctx, crystals::CELL, crystals::REACH, u0, v0, n, crystals::spawn);
        let flora = gather(&ctx, flora::CELL, flora::REACH, u0, v0, n, flora::spawn);
        if craters.is_empty()
            && sinks.is_empty()
            && volcanic.is_empty()
            && islands.is_empty()
            && spires.is_empty()
            && dunes.is_empty()
            && ice.is_empty()
            && bones.is_empty()
            && giants.is_empty()
            && petrified.is_empty()
            && mushrooms.is_empty()
            && crystals.is_empty()
            && flora.is_empty()
        {
            return Vec::new();
        }
        let m = ctx.m;
        let mut out = Vec::new();
        for z in v0..v0 + n {
            for x in u0..u0 + n {
                let ground = ctx.column(x, z).height;
                let mut lo = i32::MAX;
                let mut hi = i32::MIN;
                acc(&craters, x, z, craters::CELL, craters::REACH, craters::bounds, &mut lo, &mut hi);
                acc(&sinks, x, z, sinkholes::CELL, sinkholes::REACH, sinkholes::bounds, &mut lo, &mut hi);
                acc(&volcanic, x, z, volcanic::CELL, volcanic::REACH, volcanic::bounds, &mut lo, &mut hi);
                acc(&islands, x, z, islands::CELL, islands::REACH, islands::bounds, &mut lo, &mut hi);
                acc(&spires, x, z, spires::CELL, spires::REACH, spires::bounds, &mut lo, &mut hi);
                acc(&dunes, x, z, dunes::CELL, dunes::REACH, dunes::bounds, &mut lo, &mut hi);
                acc(&ice, x, z, ice::CELL, ice::REACH, ice::bounds, &mut lo, &mut hi);
                acc(&bones, x, z, bones::CELL, bones::REACH, bones::bounds, &mut lo, &mut hi);
                acc(&giants, x, z, giants::CELL, giants::REACH, giants::bounds, &mut lo, &mut hi);
                acc(&petrified, x, z, petrified::CELL, petrified::REACH, petrified::bounds, &mut lo, &mut hi);
                acc(&mushrooms, x, z, mushrooms::CELL, mushrooms::REACH, mushrooms::bounds, &mut lo, &mut hi);
                acc(&crystals, x, z, crystals::CELL, crystals::REACH, crystals::bounds, &mut lo, &mut hi);
                acc(&flora, x, z, flora::CELL, flora::REACH, flora::bounds, &mut lo, &mut hi);
                if lo >= hi {
                    continue;
                }
                let y_start = y0.max(ground - MAX_BELOW).max(lo);
                let y_end = y1.min(ground + MAX_ABOVE + 1).min(hi);
                for y in y_start..y_end {
                    let st = hit_recs(&craters, m, x, y, z, ground, craters::CELL, craters::REACH, craters::bounds, craters::paint)
                        .or_else(|| hit_recs(&sinks, m, x, y, z, ground, sinkholes::CELL, sinkholes::REACH, sinkholes::bounds, sinkholes::paint))
                        .or_else(|| hit_recs(&volcanic, m, x, y, z, ground, volcanic::CELL, volcanic::REACH, volcanic::bounds, volcanic::paint))
                        .or_else(|| hit_recs(&islands, m, x, y, z, ground, islands::CELL, islands::REACH, islands::bounds, islands::paint))
                        .or_else(|| hit_recs(&spires, m, x, y, z, ground, spires::CELL, spires::REACH, spires::bounds, spires::paint))
                        .or_else(|| hit_recs(&dunes, m, x, y, z, ground, dunes::CELL, dunes::REACH, dunes::bounds, dunes::paint))
                        .or_else(|| hit_recs(&ice, m, x, y, z, ground, ice::CELL, ice::REACH, ice::bounds, ice::paint))
                        .or_else(|| hit_recs(&bones, m, x, y, z, ground, bones::CELL, bones::REACH, bones::bounds, bones::paint))
                        .or_else(|| hit_recs(&giants, m, x, y, z, ground, giants::CELL, giants::REACH, giants::bounds, giants::paint))
                        .or_else(|| hit_recs(&petrified, m, x, y, z, ground, petrified::CELL, petrified::REACH, petrified::bounds, petrified::paint))
                        .or_else(|| hit_recs(&mushrooms, m, x, y, z, ground, mushrooms::CELL, mushrooms::REACH, mushrooms::bounds, mushrooms::paint))
                        .or_else(|| hit_recs(&crystals, m, x, y, z, ground, crystals::CELL, crystals::REACH, crystals::bounds, crystals::paint))
                        .or_else(|| hit_recs(&flora, m, x, y, z, ground, flora::CELL, flora::REACH, flora::bounds, flora::paint));
                    if let Some(st) = st {
                        let above = y >= ground;
                        if (above && st.id != AIR) || (!above && st.dig) || (above && st.dig) {
                            out.push((x, y, z, st.id, st.dig));
                        }
                    }
                }
            }
        }
        out
    }
}

fn acc<T: Copy>(recs: &[Rec<T>], x: i32, z: i32, cell: i32, reach: i32, bounds: fn(&T) -> (i32, i32), lo: &mut i32, hi: &mut i32) {
    for rec in recs {
        if !in_window(rec.cx, rec.cz, x, z, cell, reach) {
            continue;
        }
        let (a, b) = bounds(&rec.spec);
        *lo = (*lo).min(a);
        *hi = (*hi).max(b);
    }
}
