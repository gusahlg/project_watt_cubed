//! Ruins, monuments and watchers. One candidate per site cell, themed by the province, scaled by
//! the `structures` knob. A structure owns its footprint: landmarks, trees and flowers yield there.
//! The batch path and the per-voxel path share one site order.

mod bridges;
mod circles;
mod mines;
mod monoliths;
mod observatories;
mod pyramids;
mod temples;
mod towers;
mod waystones;

use std::cell::RefCell;
use std::sync::Arc;

use super::cube;
use super::province::ThemeId;
use super::shape::{Column, Shape};
use super::underground::Underground;
use super::Materials;
use crate::block::registry::{AIR, BlockId};
use crate::coord::Face;

pub(super) use super::features::{disk, keep, site_anchor, Stamp};

/// Deepest a structure may replace crust. A mine shaft stops on a mine level, still above the bulk.
pub(super) const DIG_LIMIT: i32 = cube::CRUST - 1;

/// One concrete ring per family. A generic `thread_local` cannot name its type parameter.
macro_rules! cache {
    ($ty:ty, $key:expr, $cx:expr, $cz:expr, $make:expr) => {{
        thread_local! {
            static RING: ::std::cell::RefCell<Vec<super::Entry<$ty>>> = ::std::cell::RefCell::new(Vec::new());
        }
        let key = $key;
        let cx = $cx;
        let cz = $cz;
        if let Some(found) = RING.with(|ring| super::lookup(ring.borrow().as_slice(), key, cx, cz)) {
            found
        } else {
            let val = $make;
            RING.with(|ring| super::remember(&mut ring.borrow_mut(), key, cx, cz, val));
            val
        }
    }};
}
pub(super) use cache;

pub(super) fn site_key(ctx: &Ctx, salt: u32) -> u64 {
    ctx.shape
        .cache_key()
        .wrapping_mul(0x1000_0000_01B3)
        ^ (ctx.s as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (salt as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ ctx.scale.to_bits() as u64
}

#[derive(Clone, Copy)]
struct Entry<T: Copy> {
    key: u64,
    cx: i32,
    cz: i32,
    live: bool,
    val: Option<T>,
}

fn lookup<T: Copy>(slots: &[Entry<T>], key: u64, cx: i32, cz: i32) -> Option<Option<T>> {
    if slots.is_empty() {
        return None;
    }
    let mask = slots.len() - 1;
    let i0 = mix(key, cx, cz) & mask;
    for k in 0..4 {
        let s = &slots[(i0 + k) & mask];
        if s.live && s.key == key && s.cx == cx && s.cz == cz {
            return Some(s.val);
        }
    }
    None
}

fn remember<T: Copy>(g: &mut Vec<Entry<T>>, key: u64, cx: i32, cz: i32, val: Option<T>) {
    const N: usize = 1024;
    if g.is_empty() {
        g.resize(N, Entry { key: 0, cx: 0, cz: 0, live: false, val: None });
    }
    let mask = N - 1;
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
    g[slot] = Entry { key, cx, cz, live: true, val };
}

fn mix(key: u64, cx: i32, cz: i32) -> usize {
    let mut h = key ^ (cx as u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= (cz as u32 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h.wrapping_mul(0x1000_0000_01B3) as usize
}

fn column_cached(key: u64, x: i32, z: i32, f: impl FnOnce() -> Column) -> Column {
    thread_local! {
        static RING: RefCell<Vec<Entry<Column>>> = RefCell::new(Vec::new());
    }
    if let Some(Some(col)) = RING.with(|r| lookup(r.borrow().as_slice(), key, x, z)) {
        return col;
    }
    let col = f();
    RING.with(|r| remember(&mut r.borrow_mut(), key, x, z, Some(col)));
    col
}

pub(super) struct Ctx<'a> {
    pub s: u32,
    pub scale: f32,
    pub m: &'a Materials,
    pub face: Face,
    pub half: i64,
    pub centre: [i64; 3],
    pub aims: &'a [[i64; 3]],
    shape: &'a Shape,
    under: &'a Underground,
    cols: Option<(&'a [Column], i32, i32, i32)>,
}

impl<'a> Ctx<'a> {
    pub(super) fn column(&self, x: i32, z: i32) -> Column {
        if let Some(col) = self.saved(x, z) {
            return col;
        }
        column_cached(self.shape.cache_key(), x, z, || self.shape.column(x, z))
    }

    /// Surface height only. A bridge transect asks for a whole site line of these.
    pub(super) fn ground(&self, x: i32, z: i32) -> i32 {
        if let Some(col) = self.saved(x, z) {
            return col.height;
        }
        self.shape.height(x, z)
    }

    fn saved(&self, x: i32, z: i32) -> Option<Column> {
        let (cols, u0, v0, n) = self.cols?;
        if x < u0 || z < v0 || x >= u0 + n || z >= v0 + n {
            return None;
        }
        Some(cols[(x - u0) as usize + (z - v0) as usize * n as usize])
    }

    pub(super) fn inland(&self, x: i32, z: i32, reach: i32) -> bool {
        self.inset(x, z) >= i64::from(cube::RIM) + i64::from(reach)
    }

    pub(super) fn inset(&self, x: i32, z: i32) -> i64 {
        self.shape.inset(x, z)
    }

    /// Footprint samples stay off the rim blend and within a storey of the pad.
    pub(super) fn flat(&self, x: i32, z: i32, pad: i32, samples: &[(i32, i32)]) -> bool {
        for &(dx, dz) in samples {
            let (sx, sz) = (x + dx, z + dz);
            if self.inset(sx, sz) < i64::from(cube::RIM) {
                return false;
            }
            if (self.column(sx, sz).height - pad).abs() > 12 {
                return false;
            }
        }
        true
    }

    pub(super) fn mine_floor(&self, x: i32, z: i32, surface: i32) -> Option<i32> {
        self.under.mine_floor(x, z, surface)
    }
}

pub(super) fn solid(id: BlockId, y: i32, ground: i32) -> Stamp {
    Stamp { id, dig: y < ground }
}

pub(super) fn carved() -> Stamp {
    Stamp { id: AIR, dig: true }
}

/// Fill below the pad and carve above it. Only the caller decides `inside`.
pub(super) fn foundation(pad: i32, ground: i32, y: i32, stone: BlockId, inside: bool) -> Option<Stamp> {
    if !inside {
        return None;
    }
    if ground < pad && (ground..pad).contains(&y) {
        return Some(Stamp { id: stone, dig: false });
    }
    if ground > pad && y == pad - 1 {
        return Some(Stamp { id: stone, dig: true });
    }
    if ground > pad && (pad..ground).contains(&y) {
        return Some(Stamp { id: AIR, dig: true });
    }
    None
}

/// Wall stone for a province. Monoliths and the telescope frame pick their own.
pub(super) fn masonry(m: &Materials, theme: ThemeId, salt: u32) -> BlockId {
    match theme {
        ThemeId::Taiga | ThemeId::Alpine | ThemeId::Tundra | ThemeId::Glacier => m.slate,
        ThemeId::Canyon | ThemeId::Mesa | ThemeId::Dune | ThemeId::Badlands | ThemeId::Petrified => {
            m.sandstone[(salt as usize) % 4]
        }
        ThemeId::Crystal => m.marble,
        ThemeId::Volcanic | ThemeId::Ash | ThemeId::Crater => m.obsidian,
        ThemeId::Fungal | ThemeId::GlowMoss => m.plank,
        _ => m.limestone,
    }
}

struct Rec<T> {
    spec: T,
    cx: i32,
    cz: i32,
}

fn gather<T, S, A>(ctx: &Ctx, cell: i32, reach: i32, u0: i32, v0: i32, n: i32, spawn: S, anchor: A) -> Vec<Rec<T>>
where
    T: Copy + 'static,
    S: Fn(&Ctx, i32, i32) -> Option<T>,
    A: Fn(&T) -> (i32, i32),
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
            let Some(spec) = spawn(ctx, cx, cz) else { continue };
            // The site cell is kilometres wide. Skip a spec whose anchor cannot reach this square.
            let (ax, az) = anchor(&spec);
            let dx = if ax < u0 { u0 - ax } else if ax > x1 { ax - x1 } else { 0 };
            let dz = if az < v0 { v0 - az } else if az > z1 { az - z1 } else { 0 };
            if dx <= reach && dz <= reach {
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

fn covers_recs<T: Copy>(
    recs: &[Rec<T>],
    x: i32,
    z: i32,
    cell: i32,
    reach: i32,
    covers: fn(&T, i32, i32) -> bool,
) -> bool {
    for rec in recs {
        if in_window(rec.cx, rec.cz, x, z, cell, reach) && covers(&rec.spec, x, z) {
            return true;
        }
    }
    false
}

fn acc<T: Copy>(
    recs: &[Rec<T>],
    x: i32,
    z: i32,
    cell: i32,
    reach: i32,
    bounds: fn(&T) -> (i32, i32),
    lo: &mut i32,
    hi: &mut i32,
) {
    for rec in recs {
        if !in_window(rec.cx, rec.cz, x, z, cell, reach) {
            continue;
        }
        let (a, b) = bounds(&rec.spec);
        *lo = (*lo).min(a);
        *hi = (*hi).max(b);
    }
}

macro_rules! families {
    ($m:ident) => {
        $m!(towers);
        $m!(pyramids);
        $m!(circles);
        $m!(monoliths);
        $m!(temples);
        $m!(observatories);
        $m!(mines);
        $m!(bridges);
        $m!(waystones);
    };
}

/// Themed structures for one face. Scale 0 (and the twins' facing face) builds nothing.
pub struct Structures {
    s: u32,
    scale: f32,
    m: Arc<Materials>,
    face: Face,
    half: i64,
    centre: [i64; 3],
    aims: [[i64; 3]; 8],
    n_aims: usize,
}

impl Structures {
    pub fn new(
        s: u32,
        scale: f32,
        m: Arc<Materials>,
        face: Face,
        half: i64,
        centre: [i64; 3],
        aims: &[[i64; 3]],
    ) -> Self {
        let mut buf = [[0i64; 3]; 8];
        let n = aims.len().min(buf.len());
        buf[..n].copy_from_slice(&aims[..n]);
        Self { s, scale, m, face, half, centre, aims: buf, n_aims: n }
    }

    fn ctx<'a>(
        &'a self,
        shape: &'a Shape,
        under: &'a Underground,
        cols: Option<(&'a [Column], i32, i32, i32)>,
    ) -> Ctx<'a> {
        Ctx {
            s: self.s,
            scale: self.scale,
            m: &self.m,
            face: self.face,
            half: self.half,
            centre: self.centre,
            aims: &self.aims[..self.n_aims],
            shape,
            under,
            cols,
        }
    }

    fn hit(&self, ctx: &Ctx, x: i32, y: i32, z: i32, ground: i32) -> Option<Stamp> {
        macro_rules! one {
            ($fam:ident) => {
                if let Some(st) = hit_spawn(
                    ctx,
                    x,
                    y,
                    z,
                    ground,
                    $fam::CELL,
                    $fam::REACH,
                    $fam::spawn,
                    $fam::bounds,
                    $fam::paint,
                ) {
                    return Some(st);
                }
            };
        }
        families!(one);
        None
    }

    /// The structure block at one cell, if it claims the cell. `ground` is that column's surface.
    pub(super) fn block_at(
        &self,
        shape: &Shape,
        under: &Underground,
        x: i32,
        y: i32,
        z: i32,
        ground: i32,
    ) -> Option<Stamp> {
        if self.scale <= 0.0 {
            return None;
        }
        let ctx = self.ctx(shape, under, None);
        self.hit(&ctx, x, y, z, ground)
    }

    /// The column is inside some structure's footprint, so landmarks and trees stay out.
    pub(super) fn owns(&self, shape: &Shape, under: &Underground, x: i32, z: i32) -> bool {
        if self.scale <= 0.0 {
            return false;
        }
        let ctx = self.ctx(shape, under, None);
        macro_rules! one {
            ($fam:ident) => {
                let c0x = (x - $fam::REACH).div_euclid($fam::CELL);
                let c1x = (x + $fam::REACH).div_euclid($fam::CELL);
                let c0z = (z - $fam::REACH).div_euclid($fam::CELL);
                let c1z = (z + $fam::REACH).div_euclid($fam::CELL);
                for cz in c0z..=c1z {
                    for cx in c0x..=c1x {
                        if let Some(spec) = $fam::spawn(&ctx, cx, cz) {
                            if $fam::covers(&spec, x, z) {
                                return true;
                            }
                        }
                    }
                }
            };
        }
        families!(one);
        false
    }

    /// Per-column footprint mask. An empty vec means nothing in this square stands.
    pub(super) fn owned(
        &self,
        shape: &Shape,
        under: &Underground,
        cols: &[Column],
        u0: i32,
        v0: i32,
        n: i32,
    ) -> Vec<bool> {
        if self.scale <= 0.0 {
            return Vec::new();
        }
        let ctx = self.ctx(shape, under, Some((cols, u0, v0, n)));
        let towers = gather(&ctx, towers::CELL, towers::REACH, u0, v0, n, towers::spawn, towers::anchor);
        let pyramids = gather(&ctx, pyramids::CELL, pyramids::REACH, u0, v0, n, pyramids::spawn, pyramids::anchor);
        let circles = gather(&ctx, circles::CELL, circles::REACH, u0, v0, n, circles::spawn, circles::anchor);
        let monoliths = gather(&ctx, monoliths::CELL, monoliths::REACH, u0, v0, n, monoliths::spawn, monoliths::anchor);
        let temples = gather(&ctx, temples::CELL, temples::REACH, u0, v0, n, temples::spawn, temples::anchor);
        let observatories = gather(&ctx, observatories::CELL, observatories::REACH, u0, v0, n, observatories::spawn, observatories::anchor);
        let mines = gather(&ctx, mines::CELL, mines::REACH, u0, v0, n, mines::spawn, mines::anchor);
        let bridges = gather(&ctx, bridges::CELL, bridges::REACH, u0, v0, n, bridges::spawn, bridges::anchor);
        let waystones = gather(&ctx, waystones::CELL, waystones::REACH, u0, v0, n, waystones::spawn, waystones::anchor);
        if towers.is_empty()
            && pyramids.is_empty()
            && circles.is_empty()
            && monoliths.is_empty()
            && temples.is_empty()
            && observatories.is_empty()
            && mines.is_empty()
            && bridges.is_empty()
            && waystones.is_empty()
        {
            return Vec::new();
        }
        let mut mask = vec![false; (n * n) as usize];
        for z in v0..v0 + n {
            for x in u0..u0 + n {
                let hit = covers_recs(&towers, x, z, towers::CELL, towers::REACH, towers::covers)
                    || covers_recs(&pyramids, x, z, pyramids::CELL, pyramids::REACH, pyramids::covers)
                    || covers_recs(&circles, x, z, circles::CELL, circles::REACH, circles::covers)
                    || covers_recs(&monoliths, x, z, monoliths::CELL, monoliths::REACH, monoliths::covers)
                    || covers_recs(&temples, x, z, temples::CELL, temples::REACH, temples::covers)
                    || covers_recs(&observatories, x, z, observatories::CELL, observatories::REACH, observatories::covers)
                    || covers_recs(&mines, x, z, mines::CELL, mines::REACH, mines::covers)
                    || covers_recs(&bridges, x, z, bridges::CELL, bridges::REACH, bridges::covers)
                    || covers_recs(&waystones, x, z, waystones::CELL, waystones::REACH, waystones::covers);
                mask[(x - u0) as usize + (z - v0) as usize * n as usize] = hit;
            }
        }
        mask
    }

    /// Structure cells in the square `(u0, v0)` and the half-open height window `[y0, y1)`.
    /// `(u, h, v, id, dig)`, same winner as [`block_at`](Self::block_at).
    pub(super) fn blocks_in(
        &self,
        shape: &Shape,
        under: &Underground,
        cols: &[Column],
        u0: i32,
        v0: i32,
        n: i32,
        y0: i32,
        y1: i32,
    ) -> Vec<(i32, i32, i32, BlockId, bool)> {
        if self.scale <= 0.0 {
            return Vec::new();
        }
        let ctx = self.ctx(shape, under, Some((cols, u0, v0, n)));
        let towers = gather(&ctx, towers::CELL, towers::REACH, u0, v0, n, towers::spawn, towers::anchor);
        let pyramids = gather(&ctx, pyramids::CELL, pyramids::REACH, u0, v0, n, pyramids::spawn, pyramids::anchor);
        let circles = gather(&ctx, circles::CELL, circles::REACH, u0, v0, n, circles::spawn, circles::anchor);
        let monoliths = gather(&ctx, monoliths::CELL, monoliths::REACH, u0, v0, n, monoliths::spawn, monoliths::anchor);
        let temples = gather(&ctx, temples::CELL, temples::REACH, u0, v0, n, temples::spawn, temples::anchor);
        let observatories = gather(&ctx, observatories::CELL, observatories::REACH, u0, v0, n, observatories::spawn, observatories::anchor);
        let mines = gather(&ctx, mines::CELL, mines::REACH, u0, v0, n, mines::spawn, mines::anchor);
        let bridges = gather(&ctx, bridges::CELL, bridges::REACH, u0, v0, n, bridges::spawn, bridges::anchor);
        let waystones = gather(&ctx, waystones::CELL, waystones::REACH, u0, v0, n, waystones::spawn, waystones::anchor);
        if towers.is_empty()
            && pyramids.is_empty()
            && circles.is_empty()
            && monoliths.is_empty()
            && temples.is_empty()
            && observatories.is_empty()
            && mines.is_empty()
            && bridges.is_empty()
            && waystones.is_empty()
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
                acc(&towers, x, z, towers::CELL, towers::REACH, towers::bounds, &mut lo, &mut hi);
                acc(&pyramids, x, z, pyramids::CELL, pyramids::REACH, pyramids::bounds, &mut lo, &mut hi);
                acc(&circles, x, z, circles::CELL, circles::REACH, circles::bounds, &mut lo, &mut hi);
                acc(&monoliths, x, z, monoliths::CELL, monoliths::REACH, monoliths::bounds, &mut lo, &mut hi);
                acc(&temples, x, z, temples::CELL, temples::REACH, temples::bounds, &mut lo, &mut hi);
                acc(&observatories, x, z, observatories::CELL, observatories::REACH, observatories::bounds, &mut lo, &mut hi);
                acc(&mines, x, z, mines::CELL, mines::REACH, mines::bounds, &mut lo, &mut hi);
                acc(&bridges, x, z, bridges::CELL, bridges::REACH, bridges::bounds, &mut lo, &mut hi);
                acc(&waystones, x, z, waystones::CELL, waystones::REACH, waystones::bounds, &mut lo, &mut hi);
                if lo >= hi {
                    continue;
                }
                let y_start = y0.max(ground - DIG_LIMIT).max(lo);
                let y_end = y1.min(ground + super::features::MAX_ABOVE + 1).min(hi);
                for y in y_start..y_end {
                    let st = hit_recs(&towers, m, x, y, z, ground, towers::CELL, towers::REACH, towers::bounds, towers::paint)
                        .or_else(|| hit_recs(&pyramids, m, x, y, z, ground, pyramids::CELL, pyramids::REACH, pyramids::bounds, pyramids::paint))
                        .or_else(|| hit_recs(&circles, m, x, y, z, ground, circles::CELL, circles::REACH, circles::bounds, circles::paint))
                        .or_else(|| hit_recs(&monoliths, m, x, y, z, ground, monoliths::CELL, monoliths::REACH, monoliths::bounds, monoliths::paint))
                        .or_else(|| hit_recs(&temples, m, x, y, z, ground, temples::CELL, temples::REACH, temples::bounds, temples::paint))
                        .or_else(|| hit_recs(&observatories, m, x, y, z, ground, observatories::CELL, observatories::REACH, observatories::bounds, observatories::paint))
                        .or_else(|| hit_recs(&mines, m, x, y, z, ground, mines::CELL, mines::REACH, mines::bounds, mines::paint))
                        .or_else(|| hit_recs(&bridges, m, x, y, z, ground, bridges::CELL, bridges::REACH, bridges::bounds, bridges::paint))
                        .or_else(|| hit_recs(&waystones, m, x, y, z, ground, waystones::CELL, waystones::REACH, waystones::bounds, waystones::paint));
                    if let Some(st) = st {
                        let above = y >= ground;
                        if (above && st.id != AIR) || st.dig {
                            out.push((x, y, z, st.id, st.dig));
                        }
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
#[derive(Clone, Copy)]
pub(super) struct Desc {
    pub x: i32,
    pub z: i32,
    pub pad: i32,
    pub a: i32,
    pub b: i32,
    pub qu: i32,
    pub qa: i32,
    pub qv: i32,
}

#[cfg(test)]
#[derive(Clone, Copy)]
pub(super) struct Found {
    pub kind: u8,
    pub x: i32,
    pub z: i32,
    pub pad: i32,
    pub a: i32,
    pub b: i32,
    pub qu: i32,
    pub qa: i32,
    pub qv: i32,
}

#[cfg(test)]
pub(super) const KIND_TOWER: u8 = 0;
#[cfg(test)]
pub(super) const KIND_PYRAMID: u8 = 1;
#[cfg(test)]
pub(super) const KIND_CIRCLE: u8 = 2;
#[cfg(test)]
pub(super) const KIND_MONOLITH: u8 = 3;
#[cfg(test)]
pub(super) const KIND_TEMPLE: u8 = 4;
#[cfg(test)]
pub(super) const KIND_OBSERVATORY: u8 = 5;
#[cfg(test)]
pub(super) const KIND_MINE: u8 = 6;
#[cfg(test)]
pub(super) const KIND_BRIDGE: u8 = 7;
#[cfg(test)]
pub(super) const KIND_WAYSTONE: u8 = 8;

#[cfg(test)]
fn sweep<T: Copy + 'static>(
    ctx: &Ctx,
    cell: i32,
    u: i32,
    v: i32,
    cells: i32,
    spawn: fn(&Ctx, i32, i32) -> Option<T>,
    describe: fn(&T) -> Desc,
    kind: u8,
    out: &mut Vec<Found>,
) {
    let cu = u.div_euclid(cell);
    let cv = v.div_euclid(cell);
    for cz in (cv - cells)..=(cv + cells) {
        for cx in (cu - cells)..=(cu + cells) {
            let Some(spec) = spawn(ctx, cx, cz) else { continue };
            let d = describe(&spec);
            out.push(Found {
                kind,
                x: d.x,
                z: d.z,
                pad: d.pad,
                a: d.a,
                b: d.b,
                qu: d.qu,
                qa: d.qa,
                qv: d.qv,
            });
        }
    }
}

/// Buildings near `(u, v)`, a few site cells out. Waystones are [`survey_roads`](survey_roads).
#[cfg(test)]
pub(super) fn survey(
    st: &Structures,
    shape: &Shape,
    under: &Underground,
    u: i32,
    v: i32,
    cells: i32,
) -> Vec<Found> {
    let ctx = st.ctx(shape, under, None);
    let mut out = Vec::new();
    sweep(&ctx, towers::CELL, u, v, cells, towers::spawn, towers::describe, KIND_TOWER, &mut out);
    sweep(&ctx, pyramids::CELL, u, v, cells, pyramids::spawn, pyramids::describe, KIND_PYRAMID, &mut out);
    sweep(&ctx, circles::CELL, u, v, cells, circles::spawn, circles::describe, KIND_CIRCLE, &mut out);
    sweep(&ctx, monoliths::CELL, u, v, cells, monoliths::spawn, monoliths::describe, KIND_MONOLITH, &mut out);
    sweep(&ctx, temples::CELL, u, v, cells, temples::spawn, temples::describe, KIND_TEMPLE, &mut out);
    sweep(&ctx, observatories::CELL, u, v, cells, observatories::spawn, observatories::describe, KIND_OBSERVATORY, &mut out);
    sweep(&ctx, mines::CELL, u, v, cells, mines::spawn, mines::describe, KIND_MINE, &mut out);
    sweep(&ctx, bridges::CELL, u, v, cells, bridges::spawn, bridges::describe, KIND_BRIDGE, &mut out);
    out
}

#[cfg(test)]
pub(super) fn survey_roads(
    st: &Structures,
    shape: &Shape,
    under: &Underground,
    u: i32,
    v: i32,
    cells: i32,
) -> Vec<Found> {
    let ctx = st.ctx(shape, under, None);
    let mut out = Vec::new();
    sweep(&ctx, waystones::CELL, u, v, cells, waystones::spawn, waystones::describe, KIND_WAYSTONE, &mut out);
    out
}

#[cfg(test)]
pub(super) fn aim_at(st: &Structures, u: i32, h: i32, v: i32, span: i32) -> (i32, i32, i32) {
    observatories::aim(st.face, st.half, st.centre, &st.aims[..st.n_aims], u, h, v, span)
}

#[cfg(test)]
mod tests {
    use super::cube;
    use super::{foundation, AIR, BlockId, DIG_LIMIT};

    #[test]
    fn foundation_flattens_only_inside_the_footprint() {
        let stone = BlockId(9);
        assert!(foundation(10, 10, 9, stone, true).is_none());
        assert!(foundation(10, 6, 7, stone, true).is_some_and(|s| s.id == stone && !s.dig));
        assert!(foundation(10, 6, 7, stone, false).is_none());
        assert!(foundation(10, 6, 5, stone, true).is_none());
        assert!(foundation(10, 14, 12, stone, true).is_some_and(|s| s.id == AIR && s.dig));
        assert!(foundation(10, 14, 9, stone, true).is_some_and(|s| s.id == stone && s.dig));
        assert!(foundation(10, 14, 9, stone, false).is_none());
        assert!(foundation(10, 14, 14, stone, true).is_none());
    }

    #[test]
    fn shafts_stop_inside_the_crust() {
        assert_eq!(DIG_LIMIT, cube::CRUST - 1);
    }
}
