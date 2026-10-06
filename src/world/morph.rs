//! Morph bands for the drawn far field, one per engine mesh detail.
//!
//! Recomputed when that cut or the eye that selected it changes. The shader
//! blends `a = smoothstep(start, end, d)` on the unmorphed post-cage distance.
//! A level with `start >= end` stays put (`a = 0`).

use voxel_engine::{DVec3, LodMorph};

use super::chunk::CHUNK_SIZE;
use super::section::{CHART_BODY_BASE, DOMAIN_H, LOD_CEIL_Y, LOD_FLOOR_Y, SectionPos};
use super::seam::{ChartSeat, SeamAcross};
use super::terrain::{DIG, LOD_DRESS_CELL, PLANT_REACH};
use super::{Coord, World};
use crate::coord::Face;

/// Nesting uploads section detail `d` at engine detail `d + MESH_SHIFT`
/// (`pack_shift` of a 32-column section is 1). Band index is that engine detail.
const MESH_SHIFT: usize = 1;

/// Storage-versus-physical slack (curvature and cages): 1% of the distance.
const MARGIN: f32 = 0.01;

/// The bands are measured from the centre chunk's middle and kept while the centre stays, but the
/// shader measures from the camera, which can stand anywhere in that chunk: half its diagonal.
const EYE_SLACK: f32 = CHUNK_SIZE as f32 * 0.5 * std::f32::consts::SQRT_2;

/// One drawn detail's nearest and farthest camera distance, in blocks.
#[derive(Clone, Copy)]
struct LevelReach {
    near: f32,
    far: f32,
    on: bool,
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
    const OFF: Self = Self { near: f32::MAX, far: 0.0, on: false };
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
/// `end_d` is the shrunk nearest point of drawn detail `d + 1`, so detail `d` is
/// fully morphed on that frontier. `start_d` is the grown farthest point of drawn
/// detail `d - 1` (and, for the finest drawn detail, of the full-res box). The
/// coarsest drawn detail has no coarser neighbour on screen, so it stays unmorphed.
fn bands_from_reaches(levels: &[LevelReach; 16], full_res_far: f32) -> [LodMorph; 16] {
    let mut out = [LodMorph::default(); 16];
    let Some(finest) = (0..16).find(|&d| levels[d].on) else {
        return out;
    };
    for d in 0..16 {
        if !levels[d].on {
            continue;
        }
        let cell = (1u32 << d) as f32;
        let mut start = if d == finest { grow_start(full_res_far, cell) } else { 0.0 };
        if d > 0 && levels[d - 1].on {
            start = start.max(grow_start(levels[d - 1].far, cell));
        }
        let end = if d + 1 < 16 && levels[d + 1].on {
            shrink_end(levels[d + 1].near, cell)
        } else {
            start
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
    /// Bands for the current cut. A still cut returns before any allocation.
    pub(in crate::world) fn recompute_lod_morph(&mut self) {
        let Some(center) = self.section_center() else {
            if self.lod_morph_set && self.lod_morph_key == 0 {
                return;
            }
            self.lod_morph = [LodMorph::default(); 16];
            self.lod_morph_key = 0;
            self.lod_morph_set = true;
            return;
        };
        let key = self.morph_key(center);
        if self.lod_morph_set && self.lod_morph_key == key {
            return;
        }
        let (levels, full) = self.gather_reaches(center);
        self.lod_morph = bands_from_reaches(&levels, full);
        self.lod_morph_key = key;
        self.lod_morph_set = true;
    }

    /// Odd fingerprint, so a cleared world (`key == 0`) never matches a real cut.
    fn morph_key(&self, center: Coord) -> u64 {
        let mut h = 0x6D6F_7270_685F_6B31;
        h = mix(h, center.x as u64);
        h = mix(h, center.y as u64);
        h = mix(h, center.z as u64);
        h = mix(h, self.section_eye_y.to_bits());
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
        for &(s, _) in &self.section_visible {
            h = mix(h, s.detail.0 as u64);
            h = mix(h, s.body as u64);
            h = mix(h, s.face as u64);
            h = mix(h, s.x as u64);
            h = mix(h, s.z as u64);
            if let Some((lo, hi)) = self.cover_band(s, in_cube) {
                h = mix(h, lo.to_bits() as u64);
                h = mix(h, hi.to_bits() as u64);
            }
        }
        h | 1
    }

    fn gather_reaches(&mut self, center: Coord) -> ([LevelReach; 16], f32) {
        let chart = if self.section_on_chart(center) { self.chart_eyes(center) } else { None };
        let in_cube = self.lod_place(center).2;
        self.lod_relief_gen = self.lod_relief_gen.wrapping_add(1);
        let pass = self.lod_relief_gen;
        let mut levels = [LevelReach::OFF; 16];
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
            let (near, far) = self.section_reach_cached(center, s, chart.as_ref(), in_cube, pass);
            let slot = &mut levels[d as usize];
            slot.near = slot.near.min(near);
            slot.far = slot.far.max(far);
            slot.on = true;
        }
        self.lod_relief.retain(|_, tops| tops.pass == pass);
        (levels, self.full_res_far(center))
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

    fn section_reach(&self, center: Coord, s: SectionPos, chart: Option<&ChartEyes>, in_cube: bool) -> (f32, f32) {
        let (ex, ey, ez) = self.section_eye(center, s, chart);
        let (y0, y1) = self.morph_relief(s, ey, in_cube);
        reach_box(ex, ey, ez, s, y0, y1)
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

    /// Column tops in the section's frame, widened to the vertices that stand on them.
    ///
    /// An edit overlay wins. A chart takes its face's ground window: every column top
    /// sits in it, and a surface walk would rerun the field for each section of a cut.
    /// A cube without a bake takes its face window in bake height. The legacy `[0, 512]`
    /// answer is a flat slab: the window floor up to the sampled top, not the whole domain.
    /// Unknown relief is a slab around the eye, so the near distance stays 0 and the far
    /// distance only grows.
    fn morph_relief(&self, s: SectionPos, eye_y: f64, in_cube: bool) -> (f64, f64) {
        let (y0, y1, _) = self.morph_relief_stable(s, eye_y, in_cube);
        (y0, y1)
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

    /// `(lo, hi, stable)`, narrowed to the drawn slabs. `stable` is false only for the eye slab,
    /// which moves with the camera.
    fn morph_relief_stable(&self, s: SectionPos, eye_y: f64, in_cube: bool) -> (f64, f64, bool) {
        let (y0, y1, stable) = self.estimated_tops(s, eye_y, in_cube);
        let (y0, y1) = self.narrowed(s, in_cube, (y0, y1));
        (y0, y1, stable)
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

#[cfg(test)]
fn ring_reach(inner: f64, outer: f64, y0: f64, y1: f64) -> (f32, f32) {
    let boxes = [
        (inner, outer, -outer, outer),
        (-outer, -inner, -outer, outer),
        (-outer, outer, inner, outer),
        (-outer, outer, -outer, -inner),
    ];
    let mut near = f32::MAX;
    let mut far = 0.0f32;
    for (x0, x1, z0, z1) in boxes {
        let (n, f) = aabb_reach(0.0, 0.0, 0.0, x0, x1, y0, y1, z0, z1);
        near = near.min(n);
        far = far.max(f);
    }
    (near, far)
}

/// Two box rings around the eye. The inner band sits on the margin limits; the outer
/// level has no coarser neighbour, so its blend is 0. A crossed middle level pops.
#[test]
fn lod_morph_bands_follow_synthetic_rings() {
    let (n2, f2) = ring_reach(400.0, 800.0, -8.0, 8.0);
    let (n3, f3) = ring_reach(800.0, 1600.0, -8.0, 8.0);
    assert!((n2 - 400.0).abs() < 1e-3, "inner near {n2}");
    assert!((n3 - 800.0).abs() < 1e-3, "outer near {n3}");
    assert!(f2 > 800.0 && f3 > f2, "far grows with the ring: {f2} {f3}");

    let full = aabb_reach(0.0, 0.0, 0.0, -200.0, 200.0, -32.0, 32.0, -200.0, 200.0).1;
    let mut levels = [LevelReach::OFF; 16];
    levels[2] = LevelReach { near: n2, far: f2, on: true };
    levels[3] = LevelReach { near: n3, far: f3, on: true };
    let bands = bands_from_reaches(&levels, full);
    let (b2, b3) = (bands[2 + MESH_SHIFT], bands[3 + MESH_SHIFT]);
    let cell2 = 4.0f32;
    let cell3 = 8.0f32;
    assert_eq!(b2.start, grow_start(full, cell2));
    assert_eq!(b2.end, shrink_end(n3, cell2));
    assert!(b2.end <= n3 && b2.start >= full && b2.end > b2.start, "inner band {b2:?}");
    assert_eq!(b3.start, grow_start(f2, cell3));
    assert_eq!(b3.end, b3.start, "coarsest stays unmorphed");
    assert!(b3.start >= f2);
    for d in [0.0, b3.start, b3.start + 10_000.0] {
        assert_eq!(morph_a(b3.start, b3.end, d), 0.0, "empty at {d}");
    }
    assert_eq!(morph_a(b2.start, b2.end, b2.start), 0.0);
    assert_eq!(morph_a(b2.start, b2.end, b2.end), 1.0);

    // The middle level's limits cross: it pops, and the stored limits still nest.
    levels[4] = LevelReach { near: 80.0, far: 400.0, on: true };
    levels[2].far = 10_000.0;
    let bands = bands_from_reaches(&levels, full);
    let mid = bands[3 + MESH_SHIFT];
    assert!(mid.start >= mid.end, "no room: {mid:?}");
    assert!(mid.end <= levels[4].near && mid.start >= levels[2].far);
    for d in [0.0, 80.0, 10_000.0, mid.start] {
        assert_eq!(morph_a(mid.start, mid.end, d), 0.0, "pop at {d}");
    }
}

/// Select the frontier and draw every section the settled punch keeps.
/// Held columns are dropped without generating their chunks: that is the cut
/// `settle_held` leaves once the near window's ground is final. Returns how
/// many held sections were removed.
#[cfg(test)]
fn arm_drawn(world: &mut World, eye: DVec3) -> usize {
    let (_, far, _, _) = world.begin_stream(eye, None);
    world.refresh_frontier(far);
    let held: std::collections::HashSet<SectionPos> = world.section_held.keys().copied().collect();
    let dropped = held.len();
    world.section_held.clear();
    if !held.is_empty() {
        world.section_desired.retain(|s| !held.contains(s));
    }
    // Mesh each section on this thread for the altitudes its slabs span; the GPU meshes stay empty.
    world.refresh_tables();
    let tables = world.tables.get();
    for s in world.section_desired.clone() {
        let mesh = super::section::extract_section_mesh(s, world.generator.as_ref(), &[], &tables);
        let detail = crate::ident::Detail(s.detail.0.saturating_add(mesh.shift as i8));
        let alt = super::SectionState::slab_span(s, detail, mesh.altitude_floor, mesh.slabs.iter().map(|slab| slab.origin_y));
        world.sections.insert(s, super::SectionState::Ready { meshes: Vec::new(), cages: Vec::new(), last_style: None, alt });
    }
    world.rebuild_section_visible(None);
    dropped
}

/// One forced recompute, then a second call that must not allocate.
#[cfg(test)]
fn time_recompute(world: &mut World, label: &str) {
    let n = world.section_visible.iter().filter(|(_, m)| !m.is_empty()).count();
    world.lod_relief.clear();
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
    let (levels, full) = world.gather_reaches(center);
    assert_eq!(world.lod_morph, bands_from_reaches(&levels, full), "{label}: stored bands");
    let drawn = world.section_visible.iter().filter(|(_, m)| !m.is_empty()).count();
    assert!(drawn > 0, "{label}: nothing drawn");
    let chart = if world.section_on_chart(center) { world.chart_eyes(center) } else { None };
    let in_cube = world.lod_place(center).2;
    let mut empty = 0usize;
    let mut shown = 0usize;
    for d in 0..16 {
        if !levels[d].on {
            continue;
        }
        shown += 1;
        let band = world.lod_morph[d + MESH_SHIFT];
        let popped = !(band.end > band.start);
        if popped {
            empty += 1;
        }
        println!(
            "{label} detail {d} engine {} start {:.2} end {:.2} near {:.1} far {:.1} empty {popped}",
            d + MESH_SHIFT,
            band.start,
            band.end,
            levels[d].near,
            levels[d].far
        );
        if d + 1 < 16 && levels[d + 1].on {
            let next = world.lod_morph[d + 1 + MESH_SHIFT];
            assert!(
                band.end <= levels[d + 1].near,
                "{label}: end_{d} {} > near_{} {}",
                band.end,
                d + 1,
                levels[d + 1].near
            );
            assert!(
                next.start >= levels[d].far,
                "{label}: start_{} {} < far_{d} {}",
                d + 1,
                next.start,
                levels[d].far
            );
        }
    }
    let finest = (0..16).find(|&d| levels[d].on).expect("a level");
    let start = world.lod_morph[finest + MESH_SHIFT].start;
    assert_eq!(start, grow_start(full, (1u32 << finest) as f32), "{label}: finest start");
    for &(s, mask) in &world.section_visible {
        if mask.is_empty() {
            continue;
        }
        let (near, _) = world.section_reach(center, s, chart.as_ref(), in_cube);
        let d = s.detail.0 as usize;
        let band = world.lod_morph[d + MESH_SHIFT];
        let a = morph_a(band.start, band.end, near);
        assert!(
            near + 1e-2 >= band.start || a == 0.0,
            "{label}: {s:?} near {near} start {} a {a}",
            band.start
        );
    }
    println!("{label}: empty {empty} of {shown}, {drawn} sections, full_res_far {full:.1}");
}

/// Chart, RD16/V5, the drawn far field after the near window has taken its columns.
#[test]
fn lod_morph_bands_hold_on_charted_worlds() {
    use crate::render_config::{RenderConfig, lod_for};
    use super::generation::WorldgenKind;

    let eye = DVec3::new(-19.1, 390.8, -0.6);
    for seed in [42_i64, 1_791_184_794_939_118_871] {
        let (lod_levels, lod_detail) = lod_for(16);
        let render = RenderConfig { lod2: true, occlusion: true, lod_levels, lod_detail, ..RenderConfig::default() };
        let mut world = World::with_kind(seed, render, WorldgenKind::Diffusion, false);
        world.set_view_distances(16, 5);
        world.section_pyramid.unit = world.view.lod_unit();
        let label = format!("chart seed {seed}");
        let dropped = arm_drawn(&mut world, eye);
        println!("{label}: dropped {dropped} held sections");
        let center = world.section_center().expect("centre");
        assert!(world.section_on_chart(center), "seed {seed} is not charted at the owner site");
        assert!(world.section_held.is_empty(), "seed {seed} still holds a punched section");
        assert_lod_bands(&mut world, &label);
        time_recompute(&mut world, &label);
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
