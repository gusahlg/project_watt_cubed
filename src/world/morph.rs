//! Morph bands for the drawn far field, one per engine mesh detail.
//!
//! Recomputed when that cut or the eye that selected it changes. The shader
//! blends `a = smoothstep(start, end, d)` on the unmorphed post-cage distance.
//! A level with `start >= end` stays put (`a = 0`).

use voxel_engine::{DVec3, LodMorph};

use super::chunk::CHUNK_SIZE;
use super::section::{CHART_BODY_BASE, CLIP_MAX_SPAN, ColumnClip, DOMAIN_H, LOD_CEIL_Y, LOD_FLOOR_Y, SectionPos, section_span};
use super::seam::{ChartSeat, SeamAcross};
use super::terrain::{DIG, LOD_DRESS_CELL, PLANT_REACH};
use super::{Coord, FastMap, World};
use crate::coord::Face;
use crate::ident::Detail;

/// Nesting uploads section detail `d` at engine detail `d + MESH_SHIFT`
/// (`pack_shift` of a 32-column section is 1). Band index is that engine detail.
const MESH_SHIFT: usize = 1;

/// Storage-versus-physical slack (curvature and cages): 1% of the distance.
const MARGIN: f32 = 0.01;

/// The bands are measured from the centre chunk's middle and kept while the centre stays, but the
/// shader measures from the camera, which can stand anywhere in that chunk: half its diagonal.
const EYE_SLACK: f32 = CHUNK_SIZE as f32 * 0.5 * std::f32::consts::SQRT_2;

/// One drawn detail's nearest and farthest camera distance, in blocks. `flat`: some section of it
/// was meshed without `dy` ([`World::section_morphs`]), so the level never moves.
#[derive(Clone, Copy)]
struct LevelReach {
    near: f32,
    far: f32,
    on: bool,
    flat: bool,
}

/// Column tops sampled for one section. `stamp` is the edit overlay; `pass` is the
/// sweep that last used the entry, so a sweep can drop sections that left the cut.
#[derive(Clone, Copy)]
pub(in crate::world) struct CachedTops {
    pass: u32,
    stamp: u64,
    y0: f64,
    y1: f64,
}

impl LevelReach {
    const OFF: Self = Self { near: f32::MAX, far: 0.0, on: false, flat: false };
}

/// One edge of a drawn tile where its level meets a coarser drawn level (`coarse`), or the
/// full-res window (`coarse: None`). The segment lies in `fine`'s frame, at the shared boundary.
#[derive(Clone, Copy)]
pub(in crate::world) struct Frontier {
    fine: SectionPos,
    coarse: Option<SectionPos>,
    x0: f64,
    x1: f64,
    z0: f64,
    z1: f64,
}

/// Per engine detail: the farthest point where a finer level (or the full-res window) meets this
/// one, and the nearest point where this one meets a coarser level.
#[derive(Clone, Copy)]
struct Meets {
    finer_far: Option<f32>,
    coarser_near: Option<f32>,
}

impl Meets {
    const NONE: Self = Self { finer_far: None, coarser_near: None };
}

/// Lower limit of a band: the distance grown by the margin, one cell and the eye slack.
pub(in crate::world) fn grow_start(limit: f32, cell: f32) -> f32 {
    let s = limit * (1.0 + MARGIN) + cell + EYE_SLACK;
    if s.is_finite() && s >= 0.0 { s } else { f32::MAX }
}

/// Upper limit of a band: the distance shrunk by the margin, one cell and the eye slack.
/// Negative when the next level reaches inside that of the eye; `end > start` then fails.
pub(in crate::world) fn shrink_end(limit: f32, cell: f32) -> f32 {
    let e = limit * (1.0 - MARGIN) - cell - EYE_SLACK;
    if e.is_finite() { e } else { f32::MIN }
}

/// Ground tops widened to the vertices on them. Digs cut below the surface; plants
/// stand on a cell only while it is still dressed.
fn dress_tops(lo: f64, hi: f64, cell: i32) -> (f64, f64) {
    let (lo, mut hi) = (lo.min(hi), lo.max(hi));
    let lo = lo - DIG as f64;
    if i64::from(cell) <= LOD_DRESS_CELL {
        hi += PLANT_REACH as f64;
    }
    (lo, hi)
}

/// `a = smoothstep(start, end, d)`, or 0 unless `start >= 0`, `end > start`, and both are finite.
/// Twin of the engine's `lod_morph_a`.
#[cfg(test)]
pub(in crate::world) fn morph_a(start: f32, end: f32, d: f32) -> f32 {
    if !(start >= 0.0 && end > start && start.is_finite() && end.is_finite()) {
        return 0.0;
    }
    let t = ((d - start) / (end - start)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Nearest and farthest distance from `(ex, ey, ez)` to the closed box.
fn aabb_reach(ex: f64, ey: f64, ez: f64, x0: f64, x1: f64, y0: f64, y1: f64, z0: f64, z1: f64) -> (f32, f32) {
    let axis = |e: f64, lo: f64, hi: f64| -> (f64, f64) {
        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
        let near = if e < lo {
            lo - e
        } else if e > hi {
            e - hi
        } else {
            0.0
        };
        let far = (e - lo).abs().max((e - hi).abs());
        (near, far)
    };
    let (nx, fx) = axis(ex, x0, x1);
    let (ny, fy) = axis(ey, y0, y1);
    let (nz, fz) = axis(ez, z0, z1);
    let pack = |x: f64, y: f64, z: f64, far: bool| {
        let d = (x * x + y * y + z * z).sqrt() as f32;
        if d.is_finite() && d >= 0.0 {
            d
        } else if far {
            f32::MAX
        } else {
            0.0
        }
    };
    (pack(nx, ny, nz, false), pack(fx, fy, fz, true))
}

/// Widest bands inside the no-crack limits. Index is the engine mesh detail.
///
/// At every frontier the finer level is fully morphed and the coarser one unmoved: `end_d` is the
/// shrunk nearest point where detail `d` meets a coarser level, `start_d` the grown farthest point
/// where a finer level or the full-res window meets `d`. With no finer neighbour a level starts at
/// its own nearest point (`full_res_far` adds the full-res window where its contact is not traced);
/// with no coarser neighbour, or meshed without `dy`, it stays unmorphed.
fn bands_from_meets(levels: &[LevelReach; 16], meets: &[Meets; 16], full_res_far: Option<f32>) -> [LodMorph; 16] {
    let mut out = [LodMorph::default(); 16];
    let finest = (0..16).find(|&d| levels[d].on);
    for d in 0..16 {
        if !levels[d].on {
            continue;
        }
        let cell = (1u32 << d) as f32;
        let mut start = grow_start(meets[d].finer_far.unwrap_or(levels[d].near), cell);
        if let (Some(full), Some(f)) = (full_res_far, finest)
            && f == d
        {
            start = start.max(grow_start(full, cell));
        }
        let end = match meets[d].coarser_near {
            Some(near) if !levels[d].flat => shrink_end(near, cell),
            _ => start,
        };
        let k = d + MESH_SHIFT;
        if k < 16 {
            out[k] = LodMorph { start, end };
        }
    }
    out
}

/// Home-chart eye, plus neighbour seams whose storage frame a drawn section may use.
struct ChartEyes {
    eye: (i64, i64, i64),
    seat: ChartSeat,
    seams: Vec<SeamAcross>,
}

impl ChartEyes {
    /// Storage `(x, z)` in the chart that holds it, starting from section `s`'s chart. `None` past
    /// every known chart (no neighbour is traced there).
    fn locate(&self, s: SectionPos, x: i64, z: i64) -> Option<(i64, i64)> {
        let inside = |seat: &ChartSeat, x: i64, z: i64| x >= seat.lo[0] && x < seat.hi[0] && z >= seat.lo[2] && z < seat.hi[2];
        if in_seat(s, self.seat.lo, self.seat.hi) {
            if inside(&self.seat, x, z) {
                return Some((x, z));
            }
            return self.seams.iter().find_map(|a| {
                let (nx, nz) = a.storage_xz(x, z);
                inside(&a.seat, nx, nz).then_some((nx, nz))
            });
        }
        let across = self.seams.iter().find(|a| in_seat(s, a.seat.lo, a.seat.hi))?;
        if inside(&across.seat, x, z) {
            return Some((x, z));
        }
        let (hx, hz) = across.home_xz(x, z);
        inside(&self.seat, hx, hz).then_some((hx, hz))
    }

    fn home(&self, s: SectionPos) -> bool {
        in_seat(s, self.seat.lo, self.seat.hi)
    }

    fn for_section(&self, s: SectionPos) -> (f64, f64, f64) {
        let (ex, ey, ez) = self.eye;
        if in_seat(s, self.seat.lo, self.seat.hi) {
            return (ex as f64, ey as f64, ez as f64);
        }
        for across in &self.seams {
            if in_seat(s, across.seat.lo, across.seat.hi) {
                let (x, z) = across.storage_xz(ex, ez);
                return (x as f64, ey as f64, z as f64);
            }
        }
        (ex as f64, ey as f64, ez as f64)
    }
}

/// The drawn sections and the chunk columns each leaves out. The engine draws a section's whole
/// tile for any quadrant the cut gives it, so the trace does too. `layered`: some drawn tile lies
/// over a drawn descendant (a stand-in while its children load).
struct Drawn {
    tiles: FastMap<SectionPos, ColumnClip>,
    finest: i8,
    coarsest: i8,
    layered: bool,
}

impl Drawn {
    fn new(cut: &[(SectionPos, super::quadtree::QuadrantMask)], clip: impl Fn(SectionPos) -> ColumnClip) -> Self {
        let mut d = Drawn { tiles: FastMap::default(), finest: i8::MAX, coarsest: i8::MIN, layered: false };
        for &(s, mask) in cut {
            if !mask.is_empty() {
                d.tiles.insert(s, clip(s));
                d.finest = d.finest.min(s.detail.0);
                d.coarsest = d.coarsest.max(s.detail.0);
            }
        }
        d.layered = d.tiles.keys().any(|&s| {
            let mut a = s;
            (s.detail.0..d.coarsest).any(|_| {
                a = a.parent();
                d.tiles.contains_key(&a)
            })
        });
        d
    }

    /// Calls `hit` with each drawn tile at detail `from` or coarser that draws storage point
    /// `(x, z)` (read in the chart that holds it, from `s`'s frame): the first, unless tiles
    /// overlap. Returns whether one did.
    fn covering(&self, s: SectionPos, x: i64, z: i64, from: i8, chart: Option<&ChartEyes>, mut hit: impl FnMut(SectionPos)) -> bool {
        let Some((x, z)) = (match chart {
            Some(c) => c.locate(s, x, z),
            None => Some((x, z)),
        }) else {
            return false;
        };
        let mut any = false;
        for d in from.max(self.finest)..=self.coarsest {
            let span = i64::from(section_span(Detail(d)));
            let cell = SectionPos { detail: Detail(d), body: s.body, face: s.face, x: x.div_euclid(span) as i32, z: z.div_euclid(span) as i32 };
            if let Some(clip) = self.tiles.get(&cell)
                && !clip.masks_point(cell, x, z)
            {
                any = true;
                hit(cell);
                if !self.layered {
                    break;
                }
            }
        }
        any
    }
}

/// Add frontier edge `f` to `out`, extending a run from `start` on that it continues.
fn push_run(out: &mut Vec<Frontier>, start: usize, f: Frontier) {
    let joins = |e: &Frontier| {
        e.coarse == f.coarse
            && ((e.x0 == e.x1 && f.x0 == f.x1 && e.x0 == f.x0 && (e.z1 == f.z0 || f.z1 == e.z0))
                || (e.z0 == e.z1 && f.z0 == f.z1 && e.z0 == f.z0 && (e.x1 == f.x0 || f.x1 == e.x0)))
    };
    match out[start..].iter_mut().rev().find(|e| joins(e)) {
        Some(e) => {
            (e.x0, e.x1, e.z0, e.z1) = (e.x0.min(f.x0), e.x1.max(f.x1), e.z0.min(f.z0), e.z1.max(f.z1));
        }
        None => out.push(f),
    }
}

fn reach_box(ex: f64, ey: f64, ez: f64, s: SectionPos, y0: f64, y1: f64) -> (f32, f32) {
    let span = s.span() as f64;
    let x0 = s.min_x() as f64;
    let z0 = s.min_z() as f64;
    aabb_reach(ex, ey, ez, x0, x0 + span, y0, y1, z0, z0 + span)
}

/// Edit-overlay fingerprint. `0` when the section has no overlay, which is also the
/// stamp of a pure generator sample.
fn overlay_stamp(world: &World, s: SectionPos) -> u64 {
    let Some(cell) = world.section_overlay.get(&s) else { return 0 };
    mix(mix(0x6f76_6572_6c61_7901, cell.lo.to_bits() as u64), cell.hi.to_bits() as u64)
}

fn in_seat(s: SectionPos, lo: [i64; 3], hi: [i64; 3]) -> bool {
    let span = s.span() as i64;
    let (x, z) = (s.min_x() as i64, s.min_z() as i64);
    x >= lo[0] && x + span <= hi[0] && z >= lo[2] && z + span <= hi[2]
}

fn mix(h: u64, v: u64) -> u64 {
    let mut x = h ^ v.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

impl World {
    /// Bands for the current cut. A still cut returns before any allocation; a moving eye over the
    /// same cut reuses the traced frontier and only remeasures it.
    pub(in crate::world) fn recompute_lod_morph(&mut self) {
        let Some(center) = self.section_center() else {
            if self.lod_morph_set && self.lod_morph_key == 0 {
                return;
            }
            self.lod_morph = [LodMorph::default(); 16];
            self.lod_morph_key = 0;
            self.lod_frontier.clear();
            self.lod_frontier_key = 0;
            self.lod_morph_set = true;
            return;
        };
        let cut = self.cut_key(center);
        let key = mix(cut, self.section_eye_y.to_bits()) | 1;
        if self.lod_morph_set && self.lod_morph_key == key {
            return;
        }
        let chart = if self.section_on_chart(center) { self.chart_eyes(center) } else { None };
        if self.lod_frontier_key != cut {
            self.trace_frontier(center, chart.as_ref());
            self.lod_frontier_key = cut;
        }
        let (levels, meets, full) = self.gather_reaches(center, chart.as_ref());
        self.lod_morph = bands_from_meets(&levels, &meets, full);
        self.lod_morph_key = key;
        self.lod_morph_set = true;
    }

    /// Fingerprint of the drawn cut and the window it meets (not the eye's height). Odd, so a
    /// cleared world (`0`) never matches a real cut.
    fn cut_key(&self, center: Coord) -> u64 {
        let mut h = 0x6D6F_7270_685F_6B31;
        h = mix(h, center.x as u64);
        h = mix(h, center.y as u64);
        h = mix(h, center.z as u64);
        h = mix(h, self.view.horizontal as u64);
        h = mix(h, self.view.vertical as u64);
        h = mix(h, self.live_up().map(|f| f as u64).unwrap_or(0xFF));
        match self.window.punch {
            Some([lo, hi]) => {
                h = mix(h, lo as u64);
                h = mix(h, hi as u64);
            }
            None => h = mix(h, 0xFFFF_FFFF),
        }
        h = mix(h, self.section_visible.len() as u64);
        let in_cube = self.lod_place(center).2;
        for &(s, mask) in &self.section_visible {
            h = mix(h, s.detail.0 as u64);
            h = mix(h, s.body as u64);
            h = mix(h, s.face as u64);
            h = mix(h, s.x as u64);
            h = mix(h, s.z as u64);
            h = mix(h, mask.bits() as u64);
            if s.span() <= CLIP_MAX_SPAN {
                h = mix(h, self.drawn_clip(s).0);
            }
            if let Some((lo, hi)) = self.cover_band(s, in_cube) {
                h = mix(h, lo.to_bits() as u64);
                h = mix(h, hi.to_bits() as u64);
            }
        }
        h | 1
    }

    /// Every edge where a drawn tile meets a coarser drawn tile or the full-res window, as runs of
    /// collinear segments. Rebuilt only when the cut changes. Each tile counts whole, less the
    /// columns it leaves to the window (whose edges meet the window), and a point drawn by several
    /// coarser tiles meets each. Across a chart seam the neighbour is found in its own storage.
    fn trace_frontier(&mut self, center: Coord, chart: Option<&ChartEyes>) {
        let drawn = Drawn::new(&self.section_visible, |s| self.drawn_clip(s));
        // The full-res window, in the home frame, where its contact can be traced.
        let near = (chart.is_some() || !self.lod_place(center).2).then(|| self.near_block_box(center));
        let in_near = |x: i64, z: i64| near.is_some_and(|(x0, x1, z0, z1)| x >= x0 && x < x1 && z >= z0 && z < z1);
        let cs = i64::from(CHUNK_SIZE as i32);
        let mut out = std::mem::take(&mut self.lod_frontier);
        out.clear();
        for (&s, &clip) in &drawn.tiles {
            let start = out.len();
            let home = chart.is_none_or(|c| c.home(s));
            let (span, x0, z0) = (i64::from(s.span()), i64::from(s.min_x()), i64::from(s.min_z()));
            // Chunk-column steps where a tile can meet the window's columns: clipped, at the near
            // square, or a fine one across a seam. Elsewhere every coarser tile aligns with a
            // quadrant side.
            let at_square = near.is_some_and(|(nx0, nx1, nz0, nz1)| x0 - cs < nx1 && x0 + span + cs > nx0 && z0 - cs < nz1 && z0 + span + cs > nz0);
            let by_column = !clip.is_empty() || at_square || (s.span() <= CLIP_MAX_SPAN && !home);
            let step = if by_column { cs } else { span / 2 };
            let classify = |out: &mut Vec<Frontier>, px: i64, pz: i64, edge: (i64, i64, i64, i64), window: bool| {
                let at = |coarse| Frontier { fine: s, coarse, x0: edge.0 as f64, x1: edge.1 as f64, z0: edge.2 as f64, z1: edge.3 as f64 };
                let any = drawn.covering(s, px, pz, s.detail.0, chart, |c| {
                    if c.detail > s.detail {
                        push_run(out, start, at(Some(c)));
                    }
                });
                // The window draws where no tile does, finer ones included.
                if window || (!any && home && in_near(px, pz) && !drawn.covering(s, px, pz, drawn.finest, chart, |_| {})) {
                    push_run(out, start, at(None));
                }
            };
            // The border, segment by segment, where the column inside is drawn.
            let mut a = 0;
            while a < span {
                let (m, b) = (a + step / 2, a + step);
                for (ix, iz, px, pz, edge) in [
                    (x0, z0 + m, x0 - 1, z0 + m, (x0, x0, z0 + a, z0 + b)),
                    (x0 + span - 1, z0 + m, x0 + span, z0 + m, (x0 + span, x0 + span, z0 + a, z0 + b)),
                    (x0 + m, z0, x0 + m, z0 - 1, (x0 + a, x0 + b, z0, z0)),
                    (x0 + m, z0 + span - 1, x0 + m, z0 + span, (x0 + a, x0 + b, z0 + span, z0 + span)),
                ] {
                    if !clip.masks_point(s, ix, iz) {
                        classify(&mut out, px, pz, edge, false);
                    }
                }
                a = b;
            }
            // Edges between kept and clipped columns meet the window, and whatever else draws there.
            for ((kx, kz), (mx, mz)) in clip.inner_edges(s) {
                let (px, pz) = (x0 + i64::from(mx) * cs + cs / 2, z0 + i64::from(mz) * cs + cs / 2);
                let (ex, ez) = (x0 + i64::from(kx.max(mx)) * cs, z0 + i64::from(kz.max(mz)) * cs);
                let edge = if kx != mx { (ex, ex, z0 + i64::from(kz) * cs, z0 + i64::from(kz + 1) * cs) } else { (x0 + i64::from(kx) * cs, x0 + i64::from(kx + 1) * cs, ez, ez) };
                classify(&mut out, px, pz, edge, true);
            }
        }
        self.lod_frontier = out;
    }

    /// The chunk columns Ready section `s`'s mesh leaves out.
    pub(in crate::world) fn drawn_clip(&self, s: SectionPos) -> ColumnClip {
        match self.sections.get(&s) {
            Some(super::SectionState::Ready { clip, .. }) if s.span() <= CLIP_MAX_SPAN => clip.drawn,
            _ => ColumnClip::NONE,
        }
    }

    fn gather_reaches(&mut self, center: Coord, chart: Option<&ChartEyes>) -> ([LevelReach; 16], [Meets; 16], Option<f32>) {
        let in_cube = self.lod_place(center).2;
        self.lod_relief_gen = self.lod_relief_gen.wrapping_add(1);
        let pass = self.lod_relief_gen;
        let mut levels = [LevelReach::OFF; 16];
        let mut morphs = (u16::MAX, 0..0);
        let n = self.section_visible.len();
        for i in 0..n {
            let (s, mask) = self.section_visible[i];
            if mask.is_empty() {
                continue;
            }
            let d = s.detail.0;
            if !(0..16).contains(&d) {
                continue;
            }
            let (near, far) = self.section_reach_cached(center, s, chart, in_cube, pass);
            if morphs.0 != s.body {
                morphs = (s.body, self.morph_levels(s.body));
            }
            let flat = !morphs.1.contains(&d);
            let slot = &mut levels[d as usize];
            slot.near = slot.near.min(near);
            slot.far = slot.far.max(far);
            slot.on = true;
            slot.flat |= flat;
        }
        let mut meets = [Meets::NONE; 16];
        let frontier = std::mem::take(&mut self.lod_frontier);
        for f in &frontier {
            let (near, far) = self.frontier_reach(center, f, chart, in_cube, pass);
            match f.coarse {
                Some(c) => {
                    let lo = &mut meets[f.fine.detail.0 as usize].coarser_near;
                    *lo = Some(lo.map_or(near, |v| v.min(near)));
                    let hi = &mut meets[c.detail.0 as usize].finer_far;
                    *hi = Some(hi.map_or(far, |v| v.max(far)));
                }
                None => {
                    let hi = &mut meets[f.fine.detail.0 as usize].finer_far;
                    *hi = Some(hi.map_or(far, |v| v.max(far)));
                }
            }
        }
        self.lod_frontier = frontier;
        self.lod_relief.retain(|_, tops| tops.pass == pass);
        // On a cube the full-res contact is not traced: the finest level keeps the old bound.
        let full = in_cube.then(|| self.full_res_far(center));
        (levels, meets, full)
    }

    /// The two distances a frontier edge constrains: the nearest of the finer side's vertices on it
    /// (they must be fully morphed) and the farthest of the side that must stay unmoved, the coarser
    /// level's vertices or, at the full-res window, the drawn level's own.
    fn frontier_reach(&mut self, center: Coord, f: &Frontier, chart: Option<&ChartEyes>, in_cube: bool, pass: u32) -> (f32, f32) {
        let (ex, ey, ez) = self.section_eye(center, f.fine, chart);
        let (a0, a1) = self.cached_tops(f.fine, ey, in_cube, pass);
        let near = aabb_reach(ex, ey, ez, f.x0, f.x1, a0, a1, f.z0, f.z1).0;
        let (b0, b1) = match f.coarse {
            Some(c) => self.cached_tops(c, ey, in_cube, pass),
            None => (a0, a1),
        };
        let far = aabb_reach(ex, ey, ez, f.x0, f.x1, b0, b1, f.z0, f.z1).1;
        (near, far)
    }

    /// Whether `s`'s level can morph (see [`morph_levels`](Self::morph_levels)).
    pub(in crate::world) fn section_morphs(&self, s: SectionPos) -> bool {
        self.morph_levels(s.body).contains(&s.detail.0)
    }

    /// The details of `body` that can morph: the selection's rings with a coarser ring beyond
    /// them. The coarsest ring and the slivers finer than the finest are meshed without `dy`, and
    /// their bands stay empty.
    fn morph_levels(&self, body: u16) -> std::ops::Range<i8> {
        let coarsest = if body >= CHART_BODY_BASE {
            let atlas = self.seams.atlases().get(usize::from(body - CHART_BODY_BASE));
            match atlas.and_then(|a| self.chart_pyramid(a.radius)) {
                Some((_, max_d)) => max_d,
                None => return 0..0,
            }
        } else {
            self.section_pyramid.coarsest().0
        };
        self.section_pyramid.finest.0..coarsest
    }

    fn chart_eyes(&self, center: Coord) -> Option<ChartEyes> {
        let seat = self.seams.chart_seat(center)?;
        let eye = super::streaming::storage_eye_block(center, self.section_eye_y, DVec3::ZERO);
        let band = self.far_horizon().max(0.0) as i64;
        let seams = self.seams.seam_across(seat, [eye.0, eye.1, eye.2], band);
        Some(ChartEyes { eye, seat, seams })
    }

    /// Eye in the frame the section is keyed in: chart storage, the reference face
    /// through [`lod_place`](Self::lod_place), or world axes on a flat world.
    fn section_eye(&self, center: Coord, s: SectionPos, chart: Option<&ChartEyes>) -> (f64, f64, f64) {
        if let Some(chart) = chart {
            chart.for_section(s)
        } else if self.section_on_chart(center) {
            let (x, y, z) = super::streaming::storage_eye_block(center, self.section_eye_y, DVec3::ZERO);
            (x as f64, y as f64, z as f64)
        } else {
            let datum = self.generator.face_datum(s.body, s.face);
            let (x, y, z) = self.section_metric_on(center, DVec3::ZERO, s.face, datum).pos();
            (x, y, z)
        }
    }

    fn section_reach_cached(
        &mut self,
        center: Coord,
        s: SectionPos,
        chart: Option<&ChartEyes>,
        in_cube: bool,
        pass: u32,
    ) -> (f32, f32) {
        let (ex, ey, ez) = self.section_eye(center, s, chart);
        let (y0, y1) = self.cached_tops(s, ey, in_cube, pass);
        reach_box(ex, ey, ez, s, y0, y1)
    }

    /// Reuse tops while the section and its edit overlay are unchanged. The eye slab
    /// (unknown relief) is not stored: it moves with the eye.
    fn cached_tops(&mut self, s: SectionPos, eye_y: f64, in_cube: bool, pass: u32) -> (f64, f64) {
        let stamp = overlay_stamp(self, s);
        let estimate = if let Some(hit) = self.lod_relief.get_mut(&s)
            && hit.stamp == stamp
        {
            hit.pass = pass;
            (hit.y0, hit.y1)
        } else {
            let (y0, y1, stable) = self.estimated_tops(s, eye_y, in_cube);
            if stable {
                self.lod_relief.insert(s, CachedTops { pass, stamp, y0, y1 });
            }
            (y0, y1)
        };
        self.narrowed(s, in_cube, estimate)
    }

    /// `tops` cut to the span the section's slabs draw, when both are known. Each bounds the
    /// drawn vertices, so their overlap does too.
    fn narrowed(&self, s: SectionPos, in_cube: bool, (y0, y1): (f64, f64)) -> (f64, f64) {
        match self.drawn_span(s, in_cube) {
            Some((d0, d1)) if d0.max(y0) < d1.min(y1) => (d0.max(y0), d1.min(y1)),
            Some(drawn) => drawn,
            None => (y0, y1),
        }
    }

    /// The altitudes the section's uploaded slabs span, where they are storage y: a chart, or a
    /// flat world's +Y window.
    fn drawn_span(&self, s: SectionPos, in_cube: bool) -> Option<(f64, f64)> {
        if s.body < CHART_BODY_BASE && (in_cube || s.face != Face::PosY) {
            return None;
        }
        match self.sections.get(&s) {
            Some(super::SectionState::Ready { alt: Some((lo, hi)), .. }) => Some((f64::from(*lo), f64::from(*hi))),
            _ => None,
        }
    }

    /// Column tops from the overlay, the chart's ground window, the cube bake or the domain.
    fn estimated_tops(&self, s: SectionPos, eye_y: f64, in_cube: bool) -> (f64, f64, bool) {
        let cell = s.cell_size();
        if let Some(cell_m) = self.section_overlay.get(&s) {
            let (y0, y1) = dress_tops(f64::from(cell_m.lo), f64::from(cell_m.hi), cell);
            return (y0, y1, true);
        }
        if s.body >= CHART_BODY_BASE {
            // The ground window, not a strided surface sample. The window is the whole
            // face band, so the bound stays a superset and the cut stays O(sections).
            let (u0, v0) = (s.min_x(), s.min_z());
            if let Some((lo, hi)) = self.generator.surface_bounds(s.body, Face::PosY, u0, v0, s.span()) {
                let (y0, y1) = dress_tops(f64::from(lo), f64::from(hi), cell);
                return (y0, y1, true);
            }
        } else if let Some((lo, hi)) = self.cover_band(s, in_cube) {
            let (y0, y1) = dress_tops(f64::from(lo), f64::from(hi), cell);
            return (y0, y1, true);
        } else if let Some((lo, hi)) = self.generator.surface_bounds(s.body, s.face, s.min_x(), s.min_z(), s.span()) {
            if (lo, hi) != (LOD_FLOOR_Y, LOD_CEIL_Y) && (lo, hi) != (LOD_CEIL_Y, LOD_FLOOR_Y) {
                let y0 = f64::from(self.baked_height(s, i64::from(lo)));
                let y1 = f64::from(self.baked_height(s, i64::from(hi)));
                return (y0.min(y1), y0.max(y1), true);
            }
            if let Some(top) = self.domain_top(s) {
                return (f64::from(LOD_FLOOR_Y), top, true);
            }
        }
        let h = f64::from(DOMAIN_H);
        (eye_y - h, eye_y + h, false)
    }

    /// First open cell at the section's corners and centre. `None` when every sample is open.
    fn domain_top(&self, s: SectionPos) -> Option<f64> {
        let span = s.span();
        if span <= 0 {
            return None;
        }
        let x0 = s.min_x();
        let z0 = s.min_z();
        let x1 = x0.saturating_add(span - 1);
        let z1 = z0.saturating_add(span - 1);
        let xm = x0.saturating_add(span / 2);
        let zm = z0.saturating_add(span / 2);
        let mut top = i32::MIN;
        for (u, v) in [(x0, z0), (x1, z0), (x0, z1), (x1, z1), (xm, zm)] {
            let h = self.generator.surface(s.face, u, v);
            if h != i32::MIN {
                top = top.max(h);
            }
        }
        (top != i32::MIN).then_some(f64::from(top))
    }

    /// Farthest point of the near window's box, in the same frame as the finest sections.
    fn full_res_far(&self, center: Coord) -> f32 {
        if self.section_on_chart(center) {
            let (ex, ey, ez) = super::streaming::storage_eye_block(center, self.section_eye_y, DVec3::ZERO);
            let (x0, x1, z0, z1) = self.near_block_box(center);
            let (y0, y1) = self.near_y_range(center);
            return aabb_reach(
                ex as f64,
                ey as f64,
                ez as f64,
                x0 as f64,
                x1 as f64,
                y0 as f64,
                y1 as f64,
                z0 as f64,
                z1 as f64,
            )
            .1;
        }
        let face = self.live_up().unwrap_or(Face::PosY);
        let body = self
            .section_visible
            .iter()
            .find(|(s, _)| s.face == face)
            .map(|(s, _)| s.body)
            .unwrap_or(0);
        let datum = self.generator.face_datum(body, face);
        let (ex, ey, ez) = self.section_metric_on(center, DVec3::ZERO, face, datum).pos();
        let (reference, _, in_cube) = self.lod_place(center);
        let half = self.view.horizontal as f64 * CHUNK_SIZE as f64 + CHUNK_SIZE as f64 * 0.5;
        let (y0, y1) = if let Some([a0, a1]) = self.window_alts() {
            let shift = if in_cube { Self::face_alt_shift(center, reference, face) } else { 0 };
            let y = |a: i64| {
                let b = a + shift;
                if face == Face::PosY && datum == 0 { b as f64 } else { (b - i64::from(datum)) as f64 }
            };
            (y(a0), y(a1))
        } else {
            let v = self.view.vertical as f64 * CHUNK_SIZE as f64 + CHUNK_SIZE as f64;
            (ey - v, ey + v)
        };
        aabb_reach(ex, ey, ez, ex - half, ex + half, y0, y1, ez - half, ez + half).1
    }
}

/// Two levels meeting at a frontier ring: the finer one is fully morphed where it meets the
/// coarser one, which is unmoved there and has no coarser neighbour (blend 0). The full-res
/// window starts the finer level. Crossed limits pop.
#[test]
fn lod_morph_bands_follow_synthetic_frontiers() {
    let mut levels = [LevelReach::OFF; 16];
    levels[2] = LevelReach { near: 400.0, far: 900.0, on: true, flat: false };
    levels[3] = LevelReach { near: 800.0, far: 1800.0, on: true, flat: false };
    let mut meets = [Meets::NONE; 16];
    meets[2] = Meets { finer_far: Some(420.0), coarser_near: Some(800.0) };
    meets[3] = Meets { finer_far: Some(870.0), coarser_near: None };
    let bands = bands_from_meets(&levels, &meets, None);
    let (b2, b3) = (bands[2 + MESH_SHIFT], bands[3 + MESH_SHIFT]);
    assert_eq!(b2.start, grow_start(420.0, 4.0));
    assert_eq!(b2.end, shrink_end(800.0, 4.0));
    assert!(b2.end > b2.start, "inner band {b2:?}");
    assert_eq!(morph_a(b2.start, b2.end, 800.0), 1.0, "fully morphed at its coarser frontier");
    assert_eq!(morph_a(b2.start, b2.end, 420.0), 0.0, "unmoved where the window meets it");
    assert_eq!(b3.start, grow_start(870.0, 8.0));
    assert_eq!(b3.end, b3.start, "no coarser neighbour: unmorphed");
    assert_eq!(morph_a(b3.start, b3.end, 870.0), 0.0, "unmoved at its finer frontier");

    // No traced finer neighbour: the level starts at its own nearest point.
    meets[2].finer_far = None;
    assert_eq!(bands_from_meets(&levels, &meets, None)[2 + MESH_SHIFT].start, grow_start(400.0, 4.0));
    // An untraced full-res window still bounds the finest level.
    assert_eq!(bands_from_meets(&levels, &meets, Some(600.0))[2 + MESH_SHIFT].start, grow_start(600.0, 4.0));

    // A level meshed without `dy` cannot move, coarser neighbour or not.
    levels[2].flat = true;
    let flat = bands_from_meets(&levels, &meets, None)[2 + MESH_SHIFT];
    assert_eq!(flat.end, flat.start, "a flat level stays put: {flat:?}");
    levels[2].flat = false;

    // Crossed limits: the level pops.
    meets[2] = Meets { finer_far: Some(900.0), coarser_near: Some(700.0) };
    let mid = bands_from_meets(&levels, &meets, None)[2 + MESH_SHIFT];
    assert!(mid.start >= mid.end, "no room: {mid:?}");
    for d in [0.0, 700.0, 900.0, 10_000.0] {
        assert_eq!(morph_a(mid.start, mid.end, d), 0.0, "pop at {d}");
    }
}

/// The owner's world and view: render distance 16, vertical 5.
#[cfg(test)]
pub(in crate::world) fn owner_world(seed: i64) -> World {
    use crate::render_config::{RenderConfig, lod_for};
    let (lod_levels, lod_detail) = lod_for(16);
    let render = RenderConfig { lod2: true, occlusion: true, lod_levels, lod_detail, ..RenderConfig::default() };
    let mut world = World::with_kind(seed, render, super::generation::WorldgenKind::Diffusion, false);
    world.set_view_distances(16, 5);
    world.section_pyramid.unit = world.view.lod_unit();
    world
}

/// The owner's camera on the start world, and the seed he plays.
#[cfg(test)]
pub(in crate::world) const OWNER_EYE: DVec3 = DVec3::new(-19.1, 390.8, -0.6);
#[cfg(test)]
pub(in crate::world) const OWNER_SEED: i64 = 1_791_184_794_939_118_871;

/// Select the frontier the settled punch keeps: held columns are dropped without
/// generating their chunks, and clipped tiles take every column they want. That is
/// the cut `settle_held` leaves once the near window's ground is final. Returns how
/// many held sections were removed.
#[cfg(test)]
fn select_settled(world: &mut World, eye: DVec3) -> usize {
    let (_, far, _, _) = world.begin_stream(eye, None);
    world.refresh_frontier(far);
    let held: std::collections::HashSet<SectionPos> = world.section_held.keys().copied().collect();
    let dropped = held.len();
    world.section_held.clear();
    if !held.is_empty() {
        world.section_desired.retain(|s| !held.contains(s));
    }
    world.settle_clip_waits();
    dropped
}

/// [`select_settled`] and draw it without meshing: every tile leaves out its clip and spans no
/// known altitudes.
#[cfg(test)]
fn arm_cut(world: &mut World, eye: DVec3) {
    select_settled(world, eye);
    for s in world.section_desired.clone() {
        let asked = world.clip_target(s);
        let clip = super::SectionClip { asked, drawn: asked };
        world.sections.insert(s, super::SectionState::Ready { meshes: Vec::new(), cages: Vec::new(), last_style: None, alt: None, clip });
    }
    world.rebuild_section_visible(None);
}

/// [`select_settled`] and draw every section of it.
#[cfg(test)]
fn arm_drawn(world: &mut World, eye: DVec3) -> usize {
    let dropped = select_settled(world, eye);
    // Mesh each section on this thread as a worker would, for the altitudes its slabs span; the
    // GPU meshes stay empty.
    world.refresh_tables();
    let tables = world.tables.get();
    for s in world.section_desired.clone() {
        let asked = world.clip_target(s);
        let mesh = super::section::extract_section_mesh_with(s, world.generator.as_ref(), &[], &tables, asked, world.section_morphs(s));
        let detail = crate::ident::Detail(s.detail.0.saturating_add(mesh.shift as i8));
        let alt = super::SectionState::slab_span(s, detail, mesh.altitude_floor, mesh.slabs.iter().map(|slab| slab.origin_y));
        let clip = super::SectionClip { asked, drawn: mesh.clip_drawn };
        world.sections.insert(s, super::SectionState::Ready { meshes: Vec::new(), cages: Vec::new(), last_style: None, alt, clip });
    }
    world.rebuild_section_visible(None);
    dropped
}

/// One forced recompute, then a second call that must not allocate.
#[cfg(test)]
fn time_recompute(world: &mut World, label: &str) {
    let n = world.section_visible.iter().filter(|(_, m)| !m.is_empty()).count();
    world.lod_relief.clear();
    world.lod_frontier_key = 0;
    world.lod_morph_set = false;
    let started = std::time::Instant::now();
    world.recompute_lod_morph();
    let cold = started.elapsed().as_micros();
    world.lod_morph_set = false;
    let started = std::time::Instant::now();
    world.recompute_lod_morph();
    let warm = started.elapsed().as_micros();
    crate::alloc_count::reset();
    world.recompute_lod_morph();
    let quiet = crate::alloc_count::alloc_count();
    println!(
        "{label}: cold recompute {cold} µs, cached {warm} µs, over {n} sections (debug); quiet allocs {quiet}"
    );
    assert_eq!(quiet, 0, "{label}: a repeated cut allocated");
}

#[cfg(test)]
fn assert_lod_bands(world: &mut World, label: &str) {
    let center = world.section_center().unwrap_or_else(|| panic!("{label}: no centre"));
    let chart = if world.section_on_chart(center) { world.chart_eyes(center) } else { None };
    let (levels, meets, full) = world.gather_reaches(center, chart.as_ref());
    assert_eq!(world.lod_morph, bands_from_meets(&levels, &meets, full), "{label}: stored bands");
    let drawn = world.section_visible.iter().filter(|(_, m)| !m.is_empty()).count();
    assert!(drawn > 0, "{label}: nothing drawn");
    assert!(!world.lod_frontier.is_empty(), "{label}: no frontier traced");
    let in_cube = world.lod_place(center).2;
    let band = |w: &World, s: SectionPos| w.lod_morph[s.detail.0 as usize + MESH_SHIFT];
    // The no-crack condition, edge by edge: the finer side fully morphed (or its level pops), the
    // coarser side and the full-res side's neighbour unmoved.
    let frontier = world.lod_frontier.clone();
    let pass = world.lod_relief_gen;
    for f in &frontier {
        let (near, far) = world.frontier_reach(center, f, chart.as_ref(), in_cube, pass);
        let fine = band(world, f.fine);
        match f.coarse {
            Some(c) => {
                let popped = !(fine.end > fine.start);
                assert!(popped || morph_a(fine.start, fine.end, near) == 1.0, "{label}: {:?} not morphed at {near}", f.fine);
                let coarse = band(world, c);
                assert_eq!(morph_a(coarse.start, coarse.end, far), 0.0, "{label}: {c:?} moves at its finer frontier {far}");
            }
            None => assert_eq!(morph_a(fine.start, fine.end, far), 0.0, "{label}: {:?} moves at the window {far}", f.fine),
        }
    }
    let mut empty = 0usize;
    let mut shown = 0usize;
    for d in 0..16 {
        if !levels[d].on {
            continue;
        }
        shown += 1;
        let b = world.lod_morph[d + MESH_SHIFT];
        let popped = !(b.end > b.start);
        empty += usize::from(popped);
        println!(
            "{label} detail {d} engine {} start {:.1} end {:.1} near {:.1} far {:.1} finer_far {:?} coarser_near {:?} empty {popped}",
            d + MESH_SHIFT,
            b.start,
            b.end,
            levels[d].near,
            levels[d].far,
            meets[d].finer_far,
            meets[d].coarser_near,
        );
    }
    // Drawn whole over a drawn descendant: the engine has no per-quadrant draw.
    let shown_set: std::collections::HashSet<SectionPos> = world.section_visible.iter().filter(|(_, m)| !m.is_empty()).map(|(s, _)| *s).collect();
    let partial = world
        .section_visible
        .iter()
        .filter(|(s, m)| !m.is_empty() && m.bits() != 0b1111 && shown_set.iter().any(|c| c.detail < s.detail && {
            let mut a = *c;
            while a.detail < s.detail {
                a = a.parent();
            }
            a == *s
        }))
        .count();
    let mut per = [0usize; 16];
    for (s, _) in world.section_visible.iter().filter(|(_, m)| !m.is_empty()) {
        per[s.detail.0 as usize] += 1;
    }
    println!("{label}: empty {empty} of {shown}, {drawn} sections {per:?}, {} frontier edges, {partial} partial parents drawn whole", frontier.len());
}

/// Probe: how many quads the morph's dy splits add, per detail, over the drawn cut at the owner
/// site: with `dy` everywhere, and gated to the levels that can morph (what the workers mesh).
#[test]
#[ignore]
fn morph_quad_overhead_probe() {
    let mut world = owner_world(42);
    arm_drawn(&mut world, OWNER_EYE);
    world.recompute_lod_morph();
    let tables = world.tables.get();
    let quads = |m: &super::section::SectionMeshData| -> u64 {
        m.slabs.iter().map(|slab| slab.data.iter().map(|(_, d)| d.quad_counts().iter().map(|&q| u64::from(q)).sum::<u64>()).sum::<u64>()).sum()
    };
    let pct = |a: u64, b: u64| (a as f64 / b.max(1) as f64 - 1.0) * 100.0;
    let mut per = [(0u64, 0u64, 0u64, 0usize); 16];
    for &(s, mask) in &world.section_visible {
        if mask.is_empty() {
            continue;
        }
        let mesh = |morph| quads(&super::section::extract_section_mesh_with(s, world.generator.as_ref(), &[], &tables, world.clip_target(s), morph));
        let (off, on) = (mesh(false), mesh(true));
        let e = &mut per[s.detail.0 as usize];
        e.0 += off;
        e.1 += on;
        e.2 += if world.section_morphs(s) { on } else { off };
        e.3 += 1;
    }
    let (mut t0, mut t1, mut t2) = (0u64, 0u64, 0u64);
    for (d, &(off, on, gated, n)) in per.iter().enumerate() {
        if n == 0 {
            continue;
        }
        let band = world.lod_morph[d + MESH_SHIFT];
        (t0, t1, t2) = (t0 + off, t1 + on, t2 + gated);
        println!(
            "detail {d}: {n} sections, quads {off} plain, {on} with dy (+{:.1}%), {gated} gated (+{:.1}%), band morphs {}",
            pct(on, off),
            pct(gated, off),
            band.end > band.start
        );
    }
    println!("total quads {t0} plain, {t1} with dy (+{:.1}%), {t2} gated (+{:.1}%)", pct(t1, t0), pct(t2, t0));
}

/// Chart, RD16/V5, the drawn far field after the near window has taken its columns.
#[test]
fn lod_morph_bands_hold_on_charted_worlds() {
    for seed in [42_i64, OWNER_SEED] {
        let mut world = owner_world(seed);
        // The game's budget at this view: the near field takes the rest of the cull knob.
        world.slot_ceiling = super::SECTION_SLOT_FLOOR as u32;
        let label = format!("chart seed {seed}");
        let dropped = arm_drawn(&mut world, OWNER_EYE);
        println!("{label}: dropped {dropped} held sections");
        let center = world.section_center().expect("centre");
        assert!(world.section_on_chart(center), "seed {seed} is not charted at the owner site");
        assert!(world.section_held.is_empty(), "seed {seed} still holds a punched section");
        assert_lod_bands(&mut world, &label);
        time_recompute(&mut world, &label);
        // The window clips the finest ring's straddlers: nothing finer is drawn, the cut fits the
        // floor, and every drawn level but the coarsest morphs.
        let drawn = |d: usize| world.section_visible.iter().any(|(s, m)| !m.is_empty() && s.detail.0 as usize == d);
        assert!(!drawn(0) && !drawn(1), "{label}: slivers drawn beside the window");
        assert!(world.section_desired.len() <= super::SECTION_SLOT_FLOOR, "{label}: over the section floor");
        let coarsest = (0..16).rev().find(|&d| drawn(d)).expect("a drawn level");
        for d in (2..coarsest).filter(|&d| drawn(d)) {
            let b = world.lod_morph[d + MESH_SHIFT];
            assert!(b.end > b.start, "{label}: detail {d} has no band {b:?}");
        }
    }
}

/// Twin 2's +Y stand, and a flat world. The cube punch needs real chunks, so the
/// stand keeps every desired section. The flat world is not charted.
#[test]
fn lod_morph_bands_hold_on_a_twin_and_a_flat_world() {
    use crate::render_config::{RenderConfig, lod_for};
    use super::generation::{WorldgenKind, FLAT_HEIGHT};

    let (lod_levels, lod_detail) = lod_for(16);
    let render = RenderConfig { lod2: true, occlusion: true, lod_levels, lod_detail, ..RenderConfig::default() };
    let mut twin = World::with_kind(42, render, WorldgenKind::Diffusion, false);
    twin.set_view_distances(16, 5);
    twin.section_pyramid.unit = twin.view.lod_unit();
    arm_drawn(&mut twin, DVec3::new(296_989_696.0, -308_421_885.0, -370_431_824.0));
    assert!(twin.lod_place(twin.section_center().expect("centre")).2, "twin stand is not on a cube");
    assert_lod_bands(&mut twin, "twin +Y");
    time_recompute(&mut twin, "twin +Y");

    let mut flat = World::with_kind(1, render, WorldgenKind::Flat, false);
    flat.set_view_distances(16, 5);
    flat.section_pyramid.unit = flat.view.lod_unit();
    arm_drawn(&mut flat, DVec3::new(8.5, f64::from(FLAT_HEIGHT) + 2.0, 8.5));
    assert!(!flat.section_on_chart(flat.section_center().expect("centre")), "flat is charted");
    assert_lod_bands(&mut flat, "flat");
    time_recompute(&mut flat, "flat");
}

/// Probe: the owner walks sixteen 16-block steps, then climbs from the ground to 900 blocks above
/// it in 8-block steps. Prints each step's bands (`EMPTY` marks a level that can morph but has no
/// band) and every finest tile that a coarser one replaced before it had fully morphed where it
/// was nearest.
#[test]
#[ignore]
fn lod_morph_walk_and_climb_probe() {
    let mut world = owner_world(OWNER_SEED);
    let ground = |world: &World| -> f64 {
        let (ex, _, ez) = super::streaming::storage_eye_block(world.section_center().expect("centre"), world.section_eye_y, DVec3::ZERO);
        f64::from(world.generator.surface(Face::PosY, ex as i32, ez as i32))
    };
    arm_cut(&mut world, OWNER_EYE);
    let floor = OWNER_EYE.y - (world.section_eye_y - ground(&world));
    let mut steps: Vec<DVec3> = (1..=16).map(|i| OWNER_EYE + DVec3::new(16.0 * f64::from(i), 0.0, 0.0)).collect();
    steps.extend((0..=112).map(|i| DVec3::new(OWNER_EYE.x, floor + 2.0 + 8.0 * f64::from(i), OWNER_EYE.z)));
    let mut prev: Option<([LodMorph; 16], FastMap<SectionPos, f32>)> = None;
    let (mut early, mut empty) = (0usize, 0usize);
    for eye in steps {
        arm_cut(&mut world, eye);
        let center = world.section_center().expect("centre");
        let chart = world.chart_eyes(center);
        let (levels, _, _) = world.gather_reaches(center, chart.as_ref());
        let above = world.section_eye_y - ground(&world);
        let coarsest = (0..16).rev().find(|&d| levels[d].on).expect("a drawn level");
        let mut line = format!("above {above:.0}:");
        for d in (0..coarsest).filter(|&d| levels[d].on && !levels[d].flat) {
            let b = world.lod_morph[d + MESH_SHIFT];
            line.push_str(&format!(" d{d} {:.0}..{:.0}", b.start, b.end));
            if b.end <= b.start {
                line.push_str(" EMPTY");
                empty += 1;
            }
        }
        println!("{line}");
        // Nearest distance of each finest tile drawn now.
        let finest = world.section_pyramid.finest;
        let pass = world.lod_relief_gen;
        let in_cube = world.lod_place(center).2;
        let drawn: std::collections::HashSet<SectionPos> = world.section_visible.iter().filter(|(_, m)| !m.is_empty()).map(|(s, _)| *s).collect();
        let mut near = FastMap::default();
        for &s in drawn.iter().filter(|s| s.detail == finest) {
            near.insert(s, world.section_reach_cached(center, s, chart.as_ref(), in_cube, pass).0);
        }
        if let Some((bands, was)) = &prev {
            let b = bands[finest.0 as usize + MESH_SHIFT];
            for (s, &d) in was {
                let coarser = std::iter::successors(Some(s.parent()), |a| (a.detail.0 < 16).then(|| a.parent())).take(8).any(|a| drawn.contains(&a));
                if !near.contains_key(s) && coarser && morph_a(b.start, b.end, d) < 0.99 {
                    early += 1;
                    println!("  {s:?} replaced at {d:.0} under {:.0}..{:.0}", b.start, b.end);
                }
            }
        }
        prev = Some((world.lod_morph, near));
    }
    println!("{empty} empty bands, {early} finest tiles replaced before fully morphed");
}
