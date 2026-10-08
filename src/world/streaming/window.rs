//! The near window along the up axis. It always holds the eye band (`eye ± vertical`), so nothing
//! the player stands on or builds beside unloads, and it grows over the terrain of the near square:
//! down to one chunk under its lowest ground and up to one chunk over its highest, within a cap
//! that keeps the data box inside [`CHUNK_BUDGET`]. Far sections the window holds give way only
//! once the chunks under them have settled (on a chart the punch waits per section; on a warped
//! cube the section is skipped once those chunks are final; in physical space the engine's clip
//! box keeps to the span its settled rings prove), and a shrink that gives up ground waits for the
//! far field.

use super::*;
use super::super::section::{CHART_BODY_BASE, FINEST_DETAIL, section_span};
use super::super::{DATA_MARGIN, UNLOAD_MARGIN, VERTICAL_RADIUS_RANGE, VIEW_RADIUS_RANGE, seam};

/// Most layers the punch window spans, eye band included.
const WINDOW_CAP: i32 = 32;

/// Layers the held window may keep past the punch window while a shrink waits for the far field.
const HELD_SLACK: i32 = 4;

/// Chunks the data box may hold: the largest settings' box, with the held slack to spare.
const CHUNK_BUDGET: i32 = {
    let side = 2 * (*VIEW_RADIUS_RANGE.end() + DATA_MARGIN) + 1;
    side * side * (2 * (*VERTICAL_RADIUS_RANGE.end() + DATA_MARGIN) + 1 + HELD_SLACK)
};

/// The near window's span along the up axis, in up-local chunk altitudes (inclusive). All `None`
/// where the window does not grow: it is the eye band there.
#[derive(Clone, Copy, Debug, Default)]
pub(in crate::world) struct Window {
    /// The span the far field's punch tests against.
    pub(in crate::world) punch: Option<[i32; 2]>,
    /// The span the chunk boxes hold: `punch`, or wider while ground a shrink gives up is not yet
    /// drawn by the far field.
    held: Option<[i32; 2]>,
    /// The near square's terrain, margins included, as last read.
    ground: Option<[i32; 2]>,
    /// The window reaches the ground.
    grounded: bool,
    /// The ground bounds are to be read again on the next pass (an edit, a relief bake).
    pub(in crate::world) stale: bool,
    /// The up face and chart atlas the altitudes are measured in.
    frame: Option<(Option<Face>, Option<usize>)>,
    /// `punch` was placed under a speed-reduced loading window: the eye band, ground unread.
    reduced: bool,
}

/// The punch and held caps at render distance `h`: [`WINDOW_CAP`] and the held slack past it,
/// shrunk so the data box stays within [`CHUNK_BUDGET`].
fn caps(h: i32) -> (i32, i32) {
    let side = 2 * (h + DATA_MARGIN) + 1;
    let punch = WINDOW_CAP.min(CHUNK_BUDGET / (side * side) - 2 * DATA_MARGIN - HELD_SLACK);
    (punch, punch + HELD_SLACK)
}

/// `core` grown toward `hull` (which holds it) by at most `spare` layers in all. Each side keeps
/// what lies nearest the core, at least half the spare layers when it needs them.
fn grow_within(core: [i32; 2], hull: [i32; 2], spare: i32) -> [i32; 2] {
    let spare = spare.max(0);
    let (down, up) = ((core[0] - hull[0]).max(0), (hull[1] - core[1]).max(0));
    let d = down.min(spare - up.min(spare / 2));
    let u = up.min(spare - d);
    [core[0] - d, core[1] + u]
}

/// The smallest span holding both.
fn hull(a: [i32; 2], b: [i32; 2]) -> [i32; 2] {
    [a[0].min(b[0]), a[1].max(b[1])]
}

/// Layers in span `s`.
fn layers(s: [i32; 2]) -> i32 {
    s[1] - s[0] + 1
}

/// The span the window wants: the eye band `eye ± v`, grown over the terrain `ground` within
/// `cap` layers.
fn wanted(eye: i32, v: i32, ground: Option<[i32; 2]>, cap: i32) -> [i32; 2] {
    let band = [eye - v, eye + v];
    ground.map_or(band, |g| grow_within(band, hull(band, g), cap - layers(band)))
}

/// `want` through a one-layer hysteresis on `prev`: a bound follows a growing span at once, and a
/// shrinking one only from two layers in, stopping one short, so bobbing across a chunk boundary
/// moves nothing.
fn hold(prev: [i32; 2], want: [i32; 2]) -> [i32; 2] {
    let lo = if want[0] < prev[0] {
        want[0]
    } else if want[0] > prev[0] + 1 {
        want[0] - 1
    } else {
        prev[0]
    };
    let hi = if want[1] > prev[1] {
        want[1]
    } else if want[1] < prev[1] - 1 {
        want[1] + 1
    } else {
        prev[1]
    };
    [lo, hi]
}

/// Whether the spans meet.
fn meets(a: [i32; 2], b: [i32; 2]) -> bool {
    a[0] <= b[1] && b[0] <= a[1]
}

impl World {
    /// The near window grows past the eye band wherever the far field gives way to it: on a round
    /// world's chart, punched by key, on a warped cube's storage when the centre's own sky is that
    /// cube face (the section is skipped once its chunks are final; the clip stays off), and in
    /// physical space, where the engine's clip box follows it
    /// ([`lod_clip_box`](Self::lod_clip_box)). Open space has no up axis.
    fn window_grows(&self, center: Coord) -> bool {
        match self.live_up() {
            Some(_) if self.fold.is_identity() => true,
            Some(Face::PosY) if self.section_on_chart(center) => true,
            Some(face) if self.on_cube_face(center, face) => true,
            _ => false,
        }
    }

    /// `center` is inside a warped cube's storage and its sky is `face` (not an open edge kept
    /// only by the up-face hysteresis).
    fn on_cube_face(&self, center: Coord, face: Face) -> bool {
        self.lod_place(center).2 && self.generator.sky(center) == Sky::Axis(face)
    }

    /// The frame the window's altitudes are measured in: the up face, and a chart's atlas.
    fn window_frame(&self, center: Coord) -> (Option<Face>, Option<usize>) {
        let atlas = if self.fold.is_identity() { None } else { self.seams.chart_seat(center).map(|s| s.index) };
        (self.live_up(), atlas)
    }

    /// The up face or chart net changed under the window. The charts of one atlas share storage
    /// +Y and one datum, so a seam crossing keeps the window as it is. Another frame starts the
    /// window afresh, and the box chunks were held to (`held`, in its net) stays loaded until the
    /// far field draws the new window, when the eye is still inside it: a walk over a cube's edge,
    /// not a jump.
    pub(in crate::world) fn follow_frame(&mut self, center: Coord, (held, fold): (Option<ChunkBox>, seam::Unfold)) {
        if self.window.frame == Some(self.window_frame(center)) {
            return;
        }
        self.window = Window::default();
        (self.lod_clip_span, self.lod_clip_next) = (None, None);
        self.retired = held.filter(|b| b.contains(fold.fold(center))).map(|b| (b, fold));
        // Whatever was kept for an earlier frame is unloaded by the next full scan.
        self.prev_unload_box = None;
    }

    /// The punch window in world chunk coordinates along the up axis (inclusive).
    pub(in crate::world) fn window_raw(&self) -> Option<[i32; 2]> {
        let ([lo, hi], face) = (self.window.punch?, self.live_up()?);
        let frame = FaceFrame::new(face);
        let raw = |alt: i32| {
            let c = frame.chunk_to_world((0, alt, 0));
            [c.x, c.y, c.z][face.axis()]
        };
        let (a, b) = (raw(lo), raw(hi));
        Some([a.min(b), a.max(b)])
    }

    /// Point the settled rings at a new punch window. Inside the span they prove, they keep it; a
    /// window reaching past it keeps clipping the part already proven while the whole is proven.
    /// Without a window they go back to the eye band, proven afresh. A chart clips nothing, so
    /// there is nothing to follow.
    fn clip_follow(&mut self, center: Coord) {
        if !self.fold.is_identity() {
            return;
        }
        let (Some(face), Some(p)) = (self.live_up(), self.window_raw()) else {
            (self.lod_clip_span, self.lod_clip_next) = (None, None);
            self.lod_clip_shrunk.set();
            return;
        };
        let c = [center.x, center.y, center.z][face.axis()];
        let v = self.view.vertical;
        let proven = self.lod_clip_span.unwrap_or([c - v, c + v]);
        let cut = [p[0].max(proven[0]), p[1].min(proven[1])];
        // Rings proven over a span hold over any part of it.
        let rings = match self.lod_clip_next {
            Some((n, r)) if p[0] >= n[0] && p[1] <= n[1] => r,
            _ => 0,
        };
        if cut == p {
            (self.lod_clip_span, self.lod_clip_next) = (Some(p), None);
        } else if cut[0] > cut[1] {
            (self.lod_clip_span, self.lod_clip_next) = (Some(p), None);
            self.lod_clip_shrunk.set();
        } else {
            (self.lod_clip_span, self.lod_clip_next) = (Some(cut), Some((p, rings)));
            self.lod_clip_grow.set();
        }
    }

    /// Layers the held window adds below and above `center`'s eye band (none off charts).
    pub(in crate::world) fn window_grow(&self, center: Coord) -> [i32; 2] {
        let (Some([lo, hi]), Some(face)) = (self.window.held, self.live_up()) else { return [0, 0] };
        let (_, eye) = ColumnKey::of(face, center);
        let v = self.view.vertical;
        [(eye - v - lo).max(0), (hi - eye - v).max(0)]
    }

    /// Place the window around streaming centre `center`; returns whether the held span changed
    /// or a retired box was let go. The punch span is recomputed only when the centre `moved` (or
    /// the ground went stale), and the settled rings follow it. The held span takes a grown punch
    /// span at once, keeps what it held within the held cap (nearest the punch first), and gives
    /// up layers once they hold no ground or once the far field draws the ground they held. A jump
    /// past the unload box leaves nothing to keep.
    pub(in crate::world) fn place_window(&mut self, center: Coord, moved: bool) -> bool {
        let prev = self.window.held;
        let reduced = !self.loading_full();
        if moved || reduced != self.window.reduced || std::mem::take(&mut self.window.stale) {
            let before = self.window.punch;
            self.window.punch = self.punch_window(center, reduced);
            self.window.reduced = reduced;
            self.window.frame = Some(self.window_frame(center));
            if self.window.punch != before {
                self.clip_follow(center);
            }
            let reach = self.view.horizontal + UNLOAD_MARGIN;
            let jumped = self.center.zip(self.live_up()).is_some_and(|(c, f)| self.fold.fold(c).across(center, f) > reach);
            let (_, held_cap) = caps(self.view.horizontal);
            self.window.held = match (prev, self.window.punch) {
                (Some(h), Some(p)) if !jumped => Some(grow_within(p, hull(h, p), held_cap - layers(p))),
                (_, p) => p,
            };
        }
        if let (Some(h), Some(p)) = (self.window.held, self.window.punch)
            && h != p
        {
            let ground = self.window.ground;
            let gives_up = |part: [i32; 2]| part[0] <= part[1] && ground.is_some_and(|g| meets(part, g));
            let ground_left = gives_up([h[0], p[0] - 1]) || gives_up([p[1] + 1, h[1]]);
            if !ground_left || self.far_covers_near(center) {
                self.window.held = Some(p);
            }
        }
        let released = self.retired.is_some() && self.far_covers_near(center);
        if released {
            self.retired = None;
            self.prev_unload_box = None;
        }
        released || self.window.held != prev
    }

    /// The span the punch tests against for `center`, through the hysteresis. The window reaches
    /// for the ground while the eye is within the window's width over the highest solid top, and
    /// within what the cap lets it add, and while the eye band is not buried under the lowest one
    /// (one layer of hysteresis each way); otherwise it is the eye band and the far field draws the
    /// ground. A `reduced` loading window would not load what the window grows over, so it keeps
    /// to the eye band and reads no ground; what the window held stays until the far field draws
    /// it ([`place_window`](Self::place_window)).
    fn punch_window(&mut self, center: Coord, reduced: bool) -> Option<[i32; 2]> {
        if !self.window_grows(center) {
            self.window = Window::default();
            return None;
        }
        let (_, eye) = ColumnKey::of(self.live_up()?, center);
        let v = self.view.vertical;
        if reduced {
            self.window.grounded = false;
            let band = [eye - v, eye + v];
            return Some(self.window.punch.map_or(band, |prev| hold(prev, band)));
        }
        let ground = self.ground_span(center);
        let (cap, _) = caps(self.view.horizontal);
        let reach = (2 * self.view.horizontal + 1).min(cap - v - 1);
        let slack = if self.window.grounded { 1 } else { -1 };
        let grounded = ground.is_some_and(|[lo, hi]| {
            let (over, under) = (eye - (hi - 1), lo + 1 - (eye + v));
            over <= reach + slack && under <= slack
        });
        self.window.ground = ground;
        self.window.grounded = grounded;
        let want = wanted(eye, v, ground.filter(|_| grounded), cap);
        Some(self.window.punch.map_or(want, |prev| hold(prev, want)))
    }

    /// Chunk layers of the near square's terrain: one under the layer of its lowest solid top to
    /// one over the layer of its highest, edits included.
    fn ground_span(&mut self, center: Coord) -> Option<[i32; 2]> {
        if self.live_up().is_some_and(|face| self.on_cube_face(center, face)) {
            self.cube_span(center)
        } else if self.fold.is_identity() {
            self.relief_span(center)
        } else {
            self.chart_span(center)
        }
    }

    /// [`ground_span`](Self::ground_span) on a warped cube. The relief is read in the reference
    /// face frame (where the bake lives) and returned as storage up-local chunk altitudes, the
    /// frame [`window_raw`](Self::window_raw) already uses. `None` until the bake has landed.
    fn cube_span(&self, center: Coord) -> Option<[i32; 2]> {
        let (reference, _, in_cube) = self.lod_place(center);
        if !in_cube {
            return None;
        }
        let face = self.live_up()?;
        let (body, _) = self.section_lod_face.filter(|&(_, f)| f == face)?;
        let frame = FaceFrame::new(face);
        let (cu, _, cv) = frame.chunk_to_local(reference);
        let h = self.view.horizontal;
        let span = section_span(FINEST_DETAIL);
        // Face-local chunk `c` does not cover cells `[c·16, c·16+16)` on a flipped tangent.
        let cells = |c0: i32, c1: i32, along_u: bool| -> (i32, i32) {
            let mut lo = i32::MAX;
            let mut hi = i32::MIN;
            let cs = CHUNK_SIZE as i32;
            for c in [c0, c1] {
                let w = if along_u { frame.chunk_to_world((c, 0, 0)) } else { frame.chunk_to_world((0, 0, c)) };
                for o in [0, cs - 1] {
                    let wx = (i64::from(w.x) * i64::from(cs) + i64::from(o)) as i32;
                    let wy = (i64::from(w.y) * i64::from(cs) + i64::from(o)) as i32;
                    let wz = (i64::from(w.z) * i64::from(cs) + i64::from(o)) as i32;
                    let (u, _, v) = frame.cell_to_local((wx, wy, wz));
                    let t = if along_u { u } else { v };
                    lo = lo.min(t);
                    hi = hi.max(t);
                }
            }
            (lo, hi)
        };
        let (u0, u1) = cells(cu - h, cu + h, true);
        let (v0, v1) = cells(cv - h, cv + h, false);
        let tiles = |a: i32, b: i32| a.div_euclid(span)..=b.div_euclid(span);
        let at = |x: i32, z: i32| SectionPos { detail: FINEST_DETAIL, body, face, x, z };
        let mut band: Option<(f32, f32)> = None;
        for z in tiles(v0, v1) {
            for x in tiles(u0, u1) {
                let (lo, hi) = self.section_relief_band(at(x, z))?;
                band = Some(band.map_or((lo, hi), |(a, b)| (a.min(lo), b.max(hi))));
            }
        }
        let (lo, hi) = band?;
        let probe = at(0, 0);
        let layer = |h: f32| -> i64 {
            let (x, y, z) = frame.cell_to_world((0, self.baked_world_y(probe, h), 0));
            i64::from(ColumnKey::of(face, World::chunk_of(x, y, z)).1)
        };
        let shift = Self::face_alt_shift(center, reference, face) / i64::from(CHUNK_SIZE as i32);
        let (a, b) = (layer(lo) - 1 - shift, layer(hi) + 1 - shift);
        Some([i32::try_from(a).ok()?, i32::try_from(b).ok()?])
    }

    /// [`ground_span`](Self::ground_span) off a chart, from the far field's baked relief over the
    /// near square (edits folded in by its overlay), per finest section of its face: no generator
    /// reads at all. `None` until that bake has landed, or where it does not reach.
    fn relief_span(&self, center: Coord) -> Option<[i32; 2]> {
        let face = self.live_up()?;
        let (body, _) = self.section_lod_face.filter(|&(_, f)| f == face)?;
        let frame = FaceFrame::new(face);
        let (cu, _, cv) = frame.chunk_to_local(center);
        let (h, cs) = (self.view.horizontal, CHUNK_SIZE as i32);
        let span = section_span(FINEST_DETAIL);
        let tiles = |c: i32| ((c - h) * cs).div_euclid(span)..=((c + h + 1) * cs - 1).div_euclid(span);
        let at = |x: i32, z: i32| SectionPos { detail: FINEST_DETAIL, body, face, x, z };
        let mut band: Option<(f32, f32)> = None;
        for z in tiles(cv) {
            for x in tiles(cu) {
                let (lo, hi) = self.section_relief_band(at(x, z))?;
                band = Some(band.map_or((lo, hi), |(a, b)| (a.min(lo), b.max(hi))));
            }
        }
        let (lo, hi) = band?;
        let layer = |h: f32| {
            let (x, y, z) = frame.cell_to_world((0, self.baked_world_y(at(0, 0), h), 0));
            ColumnKey::of(face, World::chunk_of(x, y, z)).1
        };
        Some([layer(lo) - 1, layer(hi) + 1])
    }

    /// [`ground_span`](Self::ground_span) on a chart: per chunk column of the near square, through
    /// the chart net into a neighbour chart across a seam, the generator's surface bounds (the far
    /// field's per-rect memo, kept while the column stays in the square) widened by that column's
    /// edits. Every punch rect is a union of whole columns, so its bounds lie inside these.
    fn chart_span(&mut self, center: Coord) -> Option<[i32; 2]> {
        let seat = self.seams.chart_seat(center)?;
        let body = CHART_BODY_BASE + seat.index as u16;
        let (h, n) = (self.view.horizontal, CHUNK_SIZE as i32);
        let floor = seat.lo[1].div_euclid(CHUNK_SIZE as i64) as i32;
        let (fold, mut memo) = (self.fold, std::mem::take(&mut self.window_ground));
        let bounds = memo.sweep(|memo| {
            let mut out: Option<(i32, i32)> = None;
            for cz in center.z - h..=center.z + h {
                for cx in center.x - h..=center.x + h {
                    let Some(c) = fold.unfold(Coord::new(cx, floor, cz)) else { continue };
                    let rect = [c.x * n, c.z * n, c.x * n + n, c.z * n + n];
                    let surface = || self.generator.surface_rect(body, Face::PosY, rect[0], rect[1], rect[2], rect[3]);
                    if let Some(b) = memo.get((body, rect), surface) {
                        let (lo, hi) = self.with_edits(b, rect);
                        out = Some(out.map_or((lo, hi), |(a, b)| (a.min(lo), b.max(hi))));
                    }
                }
            }
            out
        });
        self.window_ground = memo;
        let (lo, hi) = bounds?;
        Some([lo.saturating_sub(1).div_euclid(n) - 1, hi.saturating_sub(1).div_euclid(n) + 1])
    }

    /// Surface bounds `(lo, hi)` (first open cells) of storage rect `[u0, v0, u1, v1)` widened to
    /// every chunk an edit touched in its columns: a pit or a tower may sit anywhere in it.
    pub(in crate::world) fn with_edits(&self, (mut lo, mut hi): (i32, i32), [u0, v0, u1, v1]: [i32; 4]) -> (i32, i32) {
        let n = CHUNK_SIZE as i32;
        let (cx, cz) = (u0.div_euclid(n)..=(u1 - 1).div_euclid(n), v0.div_euclid(n)..=(v1 - 1).div_euclid(n));
        let mut widen = |[a, b]: [i32; 2]| {
            lo = lo.min(a * n + 1);
            hi = hi.max(b * n + n);
        };
        let columns = (cx.end() - cx.start() + 1) as usize * (cz.end() - cz.start() + 1) as usize;
        if self.edit_columns.len() < columns {
            for (&(x, z), &span) in &self.edit_columns {
                if cx.contains(&x) && cz.contains(&z) {
                    widen(span);
                }
            }
        } else {
            for z in cz.clone() {
                for x in cx.clone() {
                    if let Some(&span) = self.edit_columns.get(&(x, z)) {
                        widen(span);
                    }
                }
            }
        }
        (lo, hi)
    }

    /// Every far section the frontier wants over the near square has something drawn, or is not
    /// loaded because the window draws it. On a chart the frontier must be selected for the current
    /// punch span.
    fn far_covers_near(&self, center: Coord) -> bool {
        if !self.lod2 {
            return true;
        }
        // The frontier key records the punch only on a chart, where selection reads it. A cube
        // face does not, so a shrink waits on coverage instead of a key that stays `None`.
        if !self.fold.is_identity()
            && self.section_on_chart(center)
            && self.section_frontier_key.is_none_or(|k| k.window != self.window.punch)
        {
            return false;
        }
        self.section_desired
            .iter()
            .all(|&s| !self.near_meets(center, s) || self.coverage_skips(center, s) || self.section_covered(s))
    }

    /// Whether far section `s` reaches over the near square around `center`. On a chart a section
    /// of a neighbour chart is placed through the net; a section of another face, or outside the
    /// net, counts wherever it lies.
    fn near_meets(&self, center: Coord, s: SectionPos) -> bool {
        let h = self.view.horizontal;
        let n = CHUNK_SIZE as i32;
        let (c0, c1) = ((s.min_x().div_euclid(n), s.min_z().div_euclid(n)), ((s.min_x() + s.span() - 1).div_euclid(n), (s.min_z() + s.span() - 1).div_euclid(n)));
        let (lo, hi, eye) = if self.fold.is_identity() {
            let Some(face) = self.live_up() else { return true };
            if s.face != face || self.section_lod_face != Some((s.body, face)) {
                return true;
            }
            let (cu, _, cv) = FaceFrame::new(face).chunk_to_local(center);
            (c0, c1, (cu, cv))
        } else if self.lod_place(center).2 {
            let Some(face) = self.live_up() else { return true };
            if s.face != face || self.section_lod_face != Some((s.body, face)) {
                return true;
            }
            let (reference, _, _) = self.lod_place(center);
            let frame = FaceFrame::new(face);
            let (cu, _, cv) = frame.chunk_to_local(reference);
            let chunk_uv = |u: i32, v: i32| {
                let (x, y, z) = frame.cell_to_world((u, 0, v));
                let (tu, _, tv) = frame.chunk_to_local(World::chunk_of(x, y, z));
                (tu, tv)
            };
            let (a, b) = (chunk_uv(s.min_x(), s.min_z()), chunk_uv(s.min_x() + s.span() - 1, s.min_z() + s.span() - 1));
            ((a.0.min(b.0), a.1.min(b.1)), (a.0.max(b.0), a.1.max(b.1)), (cu, cv))
        } else {
            let Some(seat) = self.seams.chart_seat(center) else { return true };
            let floor = seat.lo[1].div_euclid(CHUNK_SIZE as i64) as i32;
            let place = |(x, z): (i32, i32)| {
                let v = self.fold.fold(Coord::new(x, floor, z));
                (v.x, v.z)
            };
            let (a, b) = (place(c0), place(c1));
            ((a.0.min(b.0), a.1.min(b.1)), (a.0.max(b.0), a.1.max(b.1)), (center.x, center.z))
        };
        lo.0 <= eye.0 + h && hi.0 >= eye.0 - h && lo.1 <= eye.1 + h && hi.1 >= eye.1 - h
    }

    /// Face-local altitudes `[a0, a1)` of the punch window.
    pub(in crate::world) fn window_alts(&self) -> Option<[i64; 2]> {
        let face = self.live_up()?;
        let [lo, hi] = self.window_raw()?;
        let cs = CHUNK_SIZE as i64;
        let (r0, r1) = (i64::from(lo) * cs, (i64::from(hi) + 1) * cs);
        Some(if face.sign() > 0 { [r0, r1] } else { [1 - r1, 1 - r0] })
    }

    /// A chunk settled: the settled rings may grow, a held section may be done waiting, and off a
    /// chart a far section its chunks now draw hands over.
    pub(in crate::world) fn note_settled(&mut self) {
        self.lod_clip_grow.set();
        self.held_recheck.set();
        // Charts punch by key. A cube has no clip, so a settled chunk has to drop the section
        // the same pass, which the visible rebuild does only when this flag is set.
        if self.fold.is_identity() || self.center.is_some_and(|c| self.lod_place(c).2) {
            self.section_cover_dirty.set();
        }
    }

    /// Every chunk under `s` in `layers` is final (see [`chunk_final`](Self::chunk_final)).
    fn backing_settled(&self, s: SectionPos, layers: [i32; 2]) -> bool {
        let n = CHUNK_SIZE as i32;
        let (x0, z0, w) = (s.min_x().div_euclid(n), s.min_z().div_euclid(n), s.span() / n);
        (layers[0]..=layers[1])
            .rev()
            .all(|y| (z0..z0 + w).all(|z| (x0..x0 + w).all(|x| self.chunk_final(Coord::new(x, y, z)))))
    }

    /// Keep drawing each held section whose ground has not all settled: it joins the desired
    /// frontier. Admission skips it only once a Ready section already draws it; skipping earlier
    /// leaves a hole. Whatever draws it (itself or an ancestor) stays drawn.
    pub(super) fn wait_held(&mut self, held: Vec<(SectionPos, [i32; 2])>) {
        self.section_held.clear();
        for (s, layers) in held {
            let selected = self.section_desired.binary_search_by_key(&section_key(&s), section_key).is_ok();
            if !selected && !self.backing_settled(s, layers) {
                self.section_held.insert(s, layers);
            }
        }
        if !self.section_held.is_empty() {
            self.section_desired.extend(self.section_held.keys().copied());
            self.section_desired.sort_unstable_by_key(section_key);
        }
    }

    /// Drop the held sections whose ground has now settled from the frontier: the punch applies.
    pub(super) fn settle_held(&mut self) {
        let mut settled: Vec<SectionPos> =
            self.section_held.iter().filter(|&(&s, &layers)| self.backing_settled(s, layers)).map(|(&s, _)| s).collect();
        if settled.is_empty() {
            return;
        }
        for s in &settled {
            self.section_held.remove(s);
        }
        settled.sort_unstable_by_key(section_key);
        self.section_desired.retain(|s| settled.binary_search_by_key(&section_key(s), section_key).is_err());
        self.section_cover_dirty.set();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The band stands alone without terrain, grows over terrain within the cap, and past it gives
    /// each side at least half the spare layers.
    #[test]
    fn wanted_span_grows_over_terrain_within_the_cap() {
        assert_eq!(wanted(10, 5, None, 32), [5, 15]);
        assert_eq!(wanted(10, 5, Some([7, 9]), 32), [5, 15], "terrain inside the band");
        assert_eq!(wanted(10, 5, Some([-6, 2]), 32), [-6, 15], "hovering over the ground");
        assert_eq!(wanted(10, 5, Some([8, 40]), 32), [5, 36], "canyon walls above, capped");
        assert_eq!(wanted(10, 5, Some([-40, 60]), 32), [-6, 25], "both sides past the cap");
        assert_eq!(wanted(10, 5, Some([-40, 16]), 32), [-15, 16], "short side first, rest below");
        assert_eq!(wanted(28, 5, Some([-40, 2]), 32), [2, 33], "the floor reaches the terrain top");
        assert_eq!(wanted(10, 20, Some([-40, 2]), 32), [-10, 30], "a band past the cap stays whole");
        for (eye, ground) in [(10, Some([-40, 60])), (0, Some([3, 90])), (7, None)] {
            let w = wanted(eye, 5, ground, 32);
            assert!(w[0] <= eye - 5 && w[1] >= eye + 5, "the band is always held");
            assert!(w[1] - w[0] < 32, "{w:?} past the cap");
        }
    }

    /// Bobbing one layer either way moves no bound; a steady climb or descent moves each bound
    /// once per layer and never back.
    #[test]
    fn hold_ignores_a_bob_and_follows_a_move() {
        let mut w = [0, 10];
        for want in [[1, 11], [0, 10], [1, 11], [0, 10]] {
            w = hold(w, want);
        }
        assert_eq!(w, [0, 11], "a bob grows the top once and then holds");
        let mut changes = 0;
        for step in 0..20 {
            let next = hold(w, [step + 1, step + 11]);
            assert!(next[0] >= w[0] && next[1] >= w[1], "a climb never lowers a bound");
            changes += i32::from(next != w);
            w = next;
        }
        assert!(changes <= 20, "{changes} changes over 20 layers");
        assert_eq!(w, [19, 30]);
        for step in (0..20).rev() {
            let next = hold(w, [step, step + 10]);
            assert!(next[0] <= w[0] && next[1] <= w[1], "a descent never raises a bound");
            w = next;
        }
        assert_eq!(w, [0, 11]);
    }

    use super::super::headless::{step, step_finished};
    use crate::world::fixtures::{air_loaded, ready_section, round_world};
    use crate::render_config::{RenderConfig, lod_for};
    use crate::world::generation::WorldgenKind;
    use std::time::{Duration, Instant};

    /// Seed 42's start world at render distance `h` and vertical distance `v` (see [`seeded`]).
    fn chart_world(h: i32, v: i32) -> World {
        seeded(42, h, v)
    }

    /// The start world of `seed` at render distance `h` and vertical distance `v`, with the far
    /// ladder the settings give that distance.
    fn seeded(seed: i64, h: i32, v: i32) -> World {
        let (lod_levels, lod_detail) = lod_for(h);
        let render = RenderConfig { lod2: true, occlusion: true, lod_levels, lod_detail, ..RenderConfig::default() };
        let mut world = round_world(seed, render);
        world.set_view_distances(h, v);
        world.section_pyramid.unit = world.view.lod_unit();
        world
    }

    /// From the start world's centre through spawn.
    fn spawn_dir(world: &World) -> DVec3 {
        let centre = world.generator.cosmos().expect("cosmos").home().centre_f();
        world.chart_spawn().expect("the start world is charted") - centre
    }

    /// Spawn in storage.
    fn spawn_storage(world: &World) -> DVec3 {
        world.chart_eye(world.chart_spawn().expect("charted")).expect("spawn stands on a chart")
    }

    /// The physical point of storage point `s`.
    fn physical(world: &World, s: DVec3) -> DVec3 {
        let cell = [s.x.floor() as i64, s.y.floor() as i64, s.z.floor() as i64];
        world
            .generator
            .atlases()
            .iter()
            .find_map(|a| a.locate(cell).map(|(patch, _)| a.embed_storage(patch, s)))
            .expect("a charted storage point")
    }

    /// The near square's chunk-centre columns around `center`, across a seam in the neighbour
    /// chart's storage: `(x, z)` and the solid top there.
    fn near_columns(world: &World, center: Coord) -> Vec<(i32, i32, i32)> {
        let h = world.view.horizontal;
        let floor = world.seams.chart_seat(center).map_or(center.y, |s| s.lo[1].div_euclid(16) as i32);
        let mut out = Vec::new();
        for cz in center.z - h..=center.z + h {
            for cx in center.x - h..=center.x + h {
                let Some(c) = world.fold.unfold(Coord::new(cx, floor, cz)) else { continue };
                let (x, z) = (c.x * 16 + 8, c.z * 16 + 8);
                out.push((x, z, world.generator.surface(Face::PosY, x, z) - 1));
            }
        }
        out
    }

    fn in_rect(s: SectionPos, x: i32, z: i32) -> bool {
        let span = s.span();
        (s.min_x()..s.min_x() + span).contains(&x) && (s.min_z()..s.min_z() + span).contains(&z)
    }

    /// Columns a span-32 tile or more inside the near square (the tile across its edge draws the
    /// sliver beside it by design) whose solid top the punch window holds and a section of
    /// `desired` covers.
    fn overlap(world: &World, center: Coord, desired: &[SectionPos], cols: &[(i32, i32, i32)]) -> usize {
        let (y0, y1) = world.near_y_range(center);
        let (x0, x1, z0, z1) = world.near_block_box(center);
        cols.iter()
            .filter(|&&(x, z, top)| {
                let (bx, bz) = (i64::from(x), i64::from(z));
                let deep = bx - x0 >= 32 && x1 - bx > 32 && bz - z0 >= 32 && z1 - bz > 32;
                deep && (y0..y1).contains(&i64::from(top)) && desired.iter().any(|&s| in_rect(s, x, z))
            })
            .count()
    }

    /// A far section on screen draws column `(x, z)`.
    fn far_drawn(world: &World, x: i32, z: i32) -> bool {
        world.section_visible.iter().any(|&(s, mask)| {
            let half = s.span() / 2;
            in_rect(s, x, z) && mask.iter().any(|q| (x - s.min_x()) / half == q.dx() && (z - s.min_z()) / half == q.dz())
        })
    }

    /// Columns drawn by neither a settled chunk holding their solid top nor a far section on screen.
    fn bare(world: &World, cols: &[(i32, i32, i32)]) -> usize {
        cols.iter()
            .filter(|&&(x, z, top)| {
                let c = Coord::new(x.div_euclid(16), top.div_euclid(16), z.div_euclid(16));
                !world.chunks.get(&c).is_some_and(|l| l.state.settled()) && !far_drawn(world, x, z)
            })
            .count()
    }

    /// Off a chart, columns whose ground the engine's clip box about `cam` hides the far field over
    /// while no settled chunk holds it, or that nothing draws.
    fn clip_bare(world: &World, cam: DVec3, cols: &[(i32, i32, i32)]) -> usize {
        let (min, max) = world.lod_clip_box(cam);
        cols.iter()
            .filter(|&&(x, z, top)| {
                let rel = [f64::from(x) - cam.x, f64::from(top) + 0.5 - cam.y, f64::from(z) - cam.z];
                let clipped = (0..3).all(|a| f64::from(min[a]) < rel[a] && rel[a] < f64::from(max[a]));
                let c = Coord::new(x.div_euclid(16), top.div_euclid(16), z.div_euclid(16));
                !world.chunks.get(&c).is_some_and(|l| l.state.settled()) && (clipped || !far_drawn(world, x, z))
            })
            .count()
    }

    /// Stream at `eye` until the pacer is back to the whole view, the world is complete and the held
    /// window is the punch window, asserting `bare` finds nothing on any pass. Each pass lands the
    /// work it started, so the passes are bounded by work, not by the wall clock. Returns the
    /// passes taken.
    fn settle(world: &mut World, eye: DVec3, bare: &dyn Fn(&World) -> usize, name: &str) -> usize {
        for pass in 0.. {
            if world.loading_full() && world.entry_complete() && world.window.held == world.window.punch {
                return pass;
            }
            assert!(pass < 200_000, "{name}: did not settle: {}", world.entry_debug());
            step_finished(world, eye);
            assert_eq!(bare(world), 0, "{name}: bare ground on settling pass {pass}");
        }
        unreachable!()
    }

    const NONE: &dyn Fn(&World) -> usize = &|_| 0;

    /// The next pass moves the eye at walking pace whatever the step: the pacer sees no travel, so
    /// the loading window stays the whole view.
    fn walk(world: &mut World) {
        world.near_eye_prev = None;
    }

    /// Owner view (16/5), seed 42: 150 and 300 blocks over spawn the window reaches down over the
    /// near square's ground and keeps the eye band, the eye's chunk loads, and once that ground
    /// has settled no far section draws over it.
    #[test]
    fn hovering_window_holds_the_ground() {
        let mut world = chart_world(16, 5);
        let dir = spawn_dir(&world);
        for above in [150.0, 300.0] {
            let (center, far, _, _) = world.begin_stream(world.home_eye(dir, above), None);
            world.cross_boundary(center);
            assert!(world.chunks.contains_key(&center), "+{above}: the eye's chunk is not loaded");
            let [lo, hi] = world.window.punch.expect("a chart window");
            let held = world.window.held.expect("a held window");
            assert!(lo <= center.y - 5 && hi >= center.y + 5, "+{above}: {lo}..={hi} drops the eye band");
            assert!(held[0] <= lo && held[1] >= hi, "+{above}: chunks hold less than the punch");
            let cols = near_columns(&world, center);
            let (y0, y1) = world.near_y_range(far);
            let missed = cols.iter().filter(|c| !(y0..y1).contains(&i64::from(c.2))).count();
            assert_eq!(missed, 0, "+{above}: ground columns outside the window");
            let desired = world.desired_sections(far);
            assert_eq!(overlap(&world, center, &desired, &cols), 0, "+{above}: far sections over held ground");
        }
    }

    /// 200 blocks under the spawn ground the eye band stands alone: the surface over a buried
    /// player is the far field's, not 20 layers of rock to load.
    #[test]
    fn buried_eye_keeps_the_band() {
        let mut world = chart_world(16, 5);
        let s = spawn_storage(&world);
        let ground = world.generator.surface(Face::PosY, s.x.floor() as i32, s.z.floor() as i32);
        let eye = physical(&world, DVec3::new(s.x, f64::from(ground) - 200.0, s.z));
        let (center, ..) = world.begin_stream(eye, None);
        assert!(!world.window.grounded, "a buried eye reached for the surface");
        assert_eq!(world.window.punch, Some([center.y - 5, center.y + 5]));
    }

    /// A platform 200 blocks over the natural ground stays loaded and solid under a player on it
    /// while the window reaches down to the ground, also while the player bobs across a chunk
    /// border: the window always keeps the eye band.
    #[test]
    fn platform_over_the_ground_stays_loaded() {
        let mut world = chart_world(16, 5);
        let spawn = spawn_storage(&world);
        let (x, z) = (spawn.x.floor() as i32, spawn.z.floor() as i32);
        let ground = world.generator.surface(Face::PosY, x, z);
        let deck = ground + 200;
        let rock = world.registry.id_by_label("rock").expect("rock");
        for dz in -2..=2 {
            for dx in -2..=2 {
                world.set_block(x + dx, deck, z + dz, rock);
            }
        }
        for lift in [1.62, 17.62, 1.62, 17.62, 1.62] {
            let eye = physical(&world, DVec3::new(f64::from(x) + 0.5, f64::from(deck + 1) + lift, f64::from(z) + 0.5));
            world.ensure_around(eye);
            let (center, ..) = world.begin_stream(eye, None);
            world.unload_far_with(center, |_, _| {});
            world.cross_boundary(center);
            let held = world.window.held.expect("a held window");
            assert!(held[0] <= (ground - 1).div_euclid(16), "+{lift}: the window does not reach the ground");
            let slab = World::collision_slab(center, world.live_up());
            assert!(world.view_coords(slab).all(|c| world.chunks.contains_key(&c)), "+{lift}: the player's surroundings unloaded");
            assert!(world.is_solid(x, deck, z) && world.is_solid(x + 2, deck, z - 2), "+{lift}: fall-through");
            assert!(!world.is_solid(x, deck + 1, z));
        }
    }

    /// Passes of [`World::begin_stream`] over spawn at each height: centre crossings, full passes,
    /// punch-window changes, and the punch window and grounding after each.
    #[derive(Default)]
    struct Track {
        moves: usize,
        full: usize,
        changes: usize,
        punches: Vec<[i32; 2]>,
        grounded: Vec<bool>,
    }

    fn fly_over(world: &mut World, dir: DVec3, heights: impl Iterator<Item = f64>) -> Track {
        let mut t = Track::default();
        for above in heights {
            let prev = (world.center, world.window.punch);
            walk(world);
            let (center, _, full, _) = world.begin_stream(world.home_eye(dir, above), None);
            let punch = world.window.punch.expect("a chart window");
            t.moves += usize::from(prev.0 != Some(center));
            t.full += usize::from(full);
            t.changes += usize::from(prev.1 != Some(punch));
            t.punches.push(punch);
            t.grounded.push(world.window.grounded);
        }
        t
    }

    /// A slow descent from 600 over spawn to the ground and back, four blocks a pass at the owner
    /// view: each window bound moves one way only, at most once per chunk crossed, so the full
    /// passes are the centre crossings. Bobbing a chunk where the window holds the ground, and
    /// where the climb let go of it, moves the window at most once.
    #[test]
    fn slow_flight_moves_the_window_by_the_hysteresis_only() {
        let mut world = chart_world(16, 5);
        let dir = spawn_dir(&world);
        fly_over(&mut world, dir, std::iter::once(600.0));
        let down = fly_over(&mut world, dir, (1..=145).map(|k| 600.0 - 4.0 * f64::from(k)));
        let up = fly_over(&mut world, dir, (1..=145).map(|k| 20.0 + 4.0 * f64::from(k)));
        println!(
            "descent: {} crossings, {} full passes, {} window changes; ascent: {} {} {}",
            down.moves, down.full, down.changes, up.moves, up.full, up.changes
        );
        let monotone = |t: &Track, sign: i32| t.punches.windows(2).all(|w| (0..2).all(|i| (w[1][i] - w[0][i]) * sign >= 0));
        assert!(monotone(&down, -1), "a bound rose during the descent");
        assert!(monotone(&up, 1), "a bound fell during the ascent");
        for (name, t) in [("descent", &down), ("ascent", &up)] {
            assert!(t.changes <= t.moves, "{name}: {} changes over {} crossings", t.changes, t.moves);
            assert_eq!(t.full, t.moves, "{name}: full passes past the crossings");
        }
        assert!(down.grounded.last() == Some(&true) && up.grounded.first() == Some(&true));
        let release = up.grounded.iter().position(|&g| !g).expect("the climb let go of the ground");
        let release = 20.0 + 4.0 * (release + 1) as f64;
        for (name, base) in [("held", 300.0), ("reach", release)] {
            fly_over(&mut world, dir, std::iter::once(base));
            let grounded = world.window.grounded;
            let bob = fly_over(&mut world, dir, (0..20).map(|k| if k % 2 == 0 { base - 16.0 } else { base }));
            assert!(bob.changes <= 1, "{name} +{base}: bobbing moved the window {} times", bob.changes);
            assert!(bob.grounded.iter().all(|&g| g == grounded), "{name} +{base}: bobbing flipped the reach");
        }
    }

    /// A deep canyon 640 blocks from spawn on seed 42: the column at storage
    /// (1124117888, 120586336) has ground within the near radius rising over 300 blocks above it.
    /// Standing on its floor at the owner view the window holds every wall (full resolution), and
    /// once that ground has settled no far section draws over it.
    #[test]
    fn canyon_walls_are_full_resolution() {
        let mut world = chart_world(16, 5);
        let (x, z) = (1_124_117_888, 120_586_336);
        let floor = world.generator.surface(Face::PosY, x, z);
        let eye = physical(&world, DVec3::new(f64::from(x) + 0.5, f64::from(floor) + 1.62, f64::from(z) + 0.5));
        let (center, far, _, _) = world.begin_stream(eye, None);
        let cols = near_columns(&world, center);
        let rim = cols.iter().map(|c| c.2).max().expect("columns");
        assert!(rim - (floor - 1) > 300, "the walls rise only {} blocks", rim - (floor - 1));
        let (y0, y1) = world.near_y_range(far);
        let missed = cols.iter().filter(|c| !(y0..y1).contains(&i64::from(c.2))).count();
        assert_eq!(missed, 0, "wall columns outside the window");
        assert_eq!(overlap(&world, center, &world.desired_sections(far), &cols), 0, "far sections over held walls");
    }

    /// Hovering 320 over spawn at render distance 6 (past the window's reach) the far field draws
    /// the ground. Descending to 60 the window grows down over it, and climbing back it lets go.
    /// On every pass of both transitions and their settling each near-square column is drawn by a
    /// settled chunk holding its ground or by a far section on screen.
    #[test]
    fn window_transitions_never_bare_the_ground() {
        let mut world = chart_world(6, 3);
        let dir = spawn_dir(&world);
        let (high, low) = (world.home_eye(dir, 320.0), world.home_eye(dir, 60.0));
        world.prepare_around(high);
        world.drive_spawn_ready();
        settle(&mut world, high, NONE, "hover");
        assert!(!world.window.grounded, "320 is past the reach");
        let cols = near_columns(&world, world.center.expect("a centre"));
        assert_eq!(bare(&world, &cols), 0, "hover: bare ground");
        for k in 1..=33 {
            let eye = world.home_eye(dir, 320.0 - 8.0 * f64::from(k));
            step(&mut world, eye);
            assert_eq!(bare(&world, &cols), 0, "descent pass {k}: bare ground");
        }
        let passes = settle(&mut world, low, &|w| bare(w, &cols), "engage");
        assert!(world.window.grounded && world.section_held.is_empty(), "engaged and settled");
        println!("engage settled in {passes} passes, window {:?}", world.window.punch);
        for k in 1..=33 {
            let eye = world.home_eye(dir, 60.0 + 8.0 * f64::from(k));
            step(&mut world, eye);
            assert_eq!(bare(&world, &cols), 0, "ascent pass {k}: bare ground");
        }
        let passes = settle(&mut world, high, &|w| bare(w, &cols), "retract");
        assert!(!world.window.grounded, "retracted");
        println!("retract settled in {passes} passes, window {:?}", world.window.punch);
    }

    /// Flying 1.2 km at eight blocks a pass 100 over the ground at render distance 6, then
    /// stopping: the flight's reduced loading window keeps the near window to the eye band, and
    /// once stopped the world settles with the window grown back over the ground, nothing left
    /// waiting on it, every column drawn, and no far section over ground the window holds.
    #[test]
    fn flight_then_stop_settles() {
        let mut world = chart_world(6, 3);
        let start = spawn_storage(&world);
        let ground = world.generator.surface(Face::PosY, start.x.floor() as i32, start.z.floor() as i32);
        let at = |k: i32| DVec3::new(start.x + 8.0 * f64::from(k), f64::from(ground) + 100.0, start.z);
        world.prepare_around(physical(&world, at(0)));
        world.drive_spawn_ready();
        let mut reduced = 0;
        for k in 0..150 {
            let eye = physical(&world, at(k));
            step(&mut world, eye);
            if world.window.reduced {
                reduced += 1;
                assert!(!world.window.grounded, "pass {k}: a reduced window read the ground");
            }
        }
        assert!(reduced > 100, "the flight reduced the window on {reduced} passes of 150");
        let stop = physical(&world, at(150));
        let passes = settle(&mut world, stop, NONE, "stop");
        assert!(world.window.grounded && !world.window.reduced, "stopped 100 up, the window holds the ground");
        let center = world.center.expect("a centre");
        let cols = near_columns(&world, center);
        println!("settled {passes} passes after stopping, {} chunks", world.chunks.len());
        assert!(world.section_held.is_empty(), "held sections still waiting");
        assert_eq!(bare(&world, &cols), 0, "bare ground after settling");
        assert_eq!(overlap(&world, center, &world.section_desired, &cols), 0, "far sections over held ground");
    }

    /// Near-square columns within its radius of the eye's column that a far section on screen draws.
    fn far_near(world: &World, center: Coord, cols: &[(i32, i32, i32)]) -> usize {
        let r = f64::from(16 * world.view.horizontal);
        let (ex, ez) = (f64::from(center.x * 16 + 8), f64::from(center.z * 16 + 8));
        cols.iter()
            .filter(|&&(x, z, _)| (f64::from(x) - ex).hypot(f64::from(z) - ez) < r && far_drawn(world, x, z))
            .count()
    }

    /// Owner view (16/5): chunks, memory, far-drawn columns near the player and settle time
    /// standing at spawn, hovering 300 over it, on the canyon floor, and at the owner's mountainside.
    /// `cargo test --release --lib near_window_cost -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn near_window_cost() {
        let canyon = |w: &World| {
            let (x, z) = (1_124_117_888, 120_586_336);
            let floor = w.generator.surface(Face::PosY, x, z);
            physical(w, DVec3::new(f64::from(x) + 0.5, f64::from(floor) + 1.62, f64::from(z) + 0.5))
        };
        let sites: [(&str, i64, &dyn Fn(&World) -> DVec3); 4] = [
            ("spawn", 42, &|w: &World| w.home_eye(spawn_dir(w), 1.62)),
            ("hover 300", 42, &|w: &World| w.home_eye(spawn_dir(w), 300.0)),
            ("canyon", 42, &canyon),
            ("owner", 1_791_184_794_939_118_871, &|_: &World| DVec3::new(-19.1, 390.8, -0.6)),
        ];
        for (name, seed, eye_of) in sites {
            let mut world = seeded(seed, 16, 5);
            let eye = eye_of(&world);
            world.prepare_around(eye);
            world.drive_spawn_ready();
            let t = Instant::now();
            let passes = settle(&mut world, eye, NONE, name);
            let center = world.center.expect("a centre");
            let census = world.memory_census();
            let cols = near_columns(&world, center);
            let mib = |b: usize| b as f64 / f64::from(1 << 20);
            println!(
                "{name}: {} chunks, mesh box {:?}, {:.1} MiB (chunks {:.1}, light {:.1}), {} sections, \
                 far-drawn near columns {} of {}, settled in {passes} passes {:.1} s",
                world.chunks.len(),
                world.mesh_box(center).size(),
                mib(census.total),
                mib(census.chunk_uniform_bytes + census.chunk_paletted_bytes + census.chunk_dense_bytes),
                mib(census.light_uniform_bytes + census.light_cells_bytes),
                world.sections.len(),
                far_near(&world, center, &cols),
                cols.len(),
                t.elapsed().as_secs_f64()
            );
        }
    }

    /// The owner's report: seed 1791184794939118871 at render distance 16 and vertical 5, standing
    /// on a mountainside at physical (-19.1, 390.8, -0.6), the valley 300 blocks below beside him
    /// drew as 4-block far field (the near square's relief spans 391 blocks, 25 chunk layers).
    /// Arriving from 600 above, no near-square column goes bare on any pass while the grown window
    /// loads; settled, the window holds that whole relief and no far section is drawn within the
    /// near square's radius of the eye.
    /// `cargo test --release --lib owner_mountainside -- --ignored --nocapture` (RD16 streaming
    /// with real workers: a release bench, not a default test).
    #[test]
    #[ignore]
    fn owner_mountainside_is_full_resolution_around_the_player() {
        let mut world = seeded(1_791_184_794_939_118_871, 16, 5);
        let eye = DVec3::new(-19.1, 390.8, -0.6);
        let up = (eye - world.generator.cosmos().expect("cosmos").home().centre_f()).normalize();
        let above = |h: f64| eye + up * h;
        world.prepare_around(above(600.0));
        world.drive_spawn_ready();
        settle(&mut world, above(600.0), NONE, "above");
        assert!(!world.window.grounded, "600 over the mountainside is past the reach");
        let center = eye_chunk(world.stream_eye(eye));
        let cols = near_columns(&world, center);
        for k in 1..=75 {
            step(&mut world, above(600.0 - 8.0 * f64::from(k)));
            assert_eq!(bare(&world, &cols), 0, "descent pass {k}: bare ground");
        }
        settle(&mut world, eye, &|w| bare(w, &cols), "mountainside");
        assert_eq!(world.center, Some(center));
        assert!(world.section_held.is_empty(), "held sections still waiting");
        let (y0, y1) = world.near_y_range(center);
        let missed = cols.iter().filter(|c| !(y0..y1).contains(&i64::from(c.2))).count();
        assert_eq!(missed, 0, "ground columns outside the window");
        assert_eq!(far_near(&world, center, &cols), 0, "far sections drawn within the near radius");
        let census = world.memory_census();
        println!(
            "owner site: window {:?}, mesh box {:?}, {} chunks, {:.1} MiB",
            world.window.punch,
            world.mesh_box(center).size(),
            world.chunks.len(),
            census.total as f64 / f64::from(1 << 20)
        );
    }

    /// Off a chart the engine's clip box is the window. Over a flat world at render distance 6,
    /// hovering 400 up (past the reach) the far field draws the ground; descending to 60 the
    /// window grows over it and climbing back it lets go. On every pass no column's ground is
    /// hidden by the box while unsettled, nor left undrawn; settled low, the box spans the window.
    #[test]
    fn flat_clip_box_follows_the_window() {
        use crate::world::generation::FLAT_HEIGHT;

        let render = RenderConfig { lod2: true, occlusion: true, ..RenderConfig::default() };
        let mut world = World::with_kind(1, render, WorldgenKind::Flat, false);
        world.set_view_distances(6, 3);
        let at = |above: f64| DVec3::new(8.5, f64::from(FLAT_HEIGHT) + above, 8.5);
        world.prepare_around(at(400.0));
        world.drive_spawn_ready();
        settle(&mut world, at(400.0), NONE, "hover");
        assert!(world.window.punch.is_some() && !world.window.grounded, "400 is past the reach");
        let cols = near_columns(&world, world.center.expect("a centre"));
        for (name, heights) in [("descent", (1..=43).map(|k| 400.0 - 8.0 * f64::from(k)).collect::<Vec<_>>()), ("ascent", (1..=43).map(|k| 56.0 + 8.0 * f64::from(k)).collect())] {
            for (k, &above) in heights.iter().enumerate() {
                step(&mut world, at(above));
                assert_eq!(clip_bare(&world, at(above), &cols), 0, "{name} pass {k} at +{above}: bare ground");
            }
            let end = *heights.last().expect("heights");
            settle(&mut world, at(end), &|w| clip_bare(w, at(end), &cols), name);
            if name == "descent" {
                let [lo, hi] = world.window_raw().expect("a window");
                assert!(world.window.grounded && lo <= (FLAT_HEIGHT - 1).div_euclid(16) - 1, "the window misses the ground");
                assert_eq!(world.lod_clip_span, Some([lo, hi]), "the rings prove the whole window");
                let (min, max) = world.lod_clip_box(at(end));
                let cam_y = at(end).y;
                assert_eq!((f64::from(min.y), f64::from(max.y)), (f64::from(lo * 16) - cam_y, f64::from((hi + 1) * 16) - cam_y));
                assert_eq!(max.x, (16 * world.view.horizontal) as f32, "the rings settle the whole square");
            }
        }
        assert!(!world.window.grounded, "climbed past the reach");
    }

    /// The owner's site (see above) without streaming: the window placed there holds every column
    /// of the near square, and the selection once that ground settles draws no far section within
    /// the near radius of the eye.
    #[test]
    fn owner_mountainside_selection_is_full_resolution() {
        let mut world = seeded(1_791_184_794_939_118_871, 16, 5);
        let (center, far, _, _) = world.begin_stream(DVec3::new(-19.1, 390.8, -0.6), None);
        world.cross_boundary(center);
        assert!(world.chunks.contains_key(&center), "the eye's chunk is not loaded");
        let cols = near_columns(&world, center);
        let (y0, y1) = world.near_y_range(far);
        assert_eq!(cols.iter().filter(|c| !(y0..y1).contains(&i64::from(c.2))).count(), 0, "ground outside the window");
        let desired = world.desired_sections(far);
        let r = f64::from(16 * world.view.horizontal);
        let (ex, ez) = (f64::from(center.x * 16 + 8), f64::from(center.z * 16 + 8));
        let drawn = cols
            .iter()
            .filter(|&&(x, z, _)| (f64::from(x) - ex).hypot(f64::from(z) - ez) < r && desired.iter().any(|&s| in_rect(s, x, z)))
            .count();
        assert_eq!(drawn, 0, "far sections within the near radius");
    }

    /// The data box never passes the budget, at any render and vertical distance: the window's cap
    /// shrinks with the render distance. Hovering 300 over spawn at 16/5 the window grows to its
    /// cap; at 20/10 the eye band already fills the budget.
    #[test]
    fn data_box_stays_within_the_budget() {
        for h in VIEW_RADIUS_RANGE {
            for v in VERTICAL_RADIUS_RANGE {
                let side = 2 * (h + DATA_MARGIN) + 1;
                let held = caps(h).1.max(2 * v + 1);
                assert!(side * side * (held + 2 * DATA_MARGIN) <= CHUNK_BUDGET, "{h}/{v}: past the budget");
            }
        }
        for (h, v) in [(16, 5), (20, 10)] {
            let mut world = chart_world(h, v);
            let (center, ..) = world.begin_stream(world.home_eye(spawn_dir(&world), 300.0), None);
            let (x, y, z) = world.data_box(center).size();
            println!("{h}/{v} hovering 300: window {:?}, data box {} chunks of {CHUNK_BUDGET}", world.window.punch, x * y * z);
            assert!(x * y * z <= CHUNK_BUDGET);
        }
    }

    /// Flying 100 over the start world from the +Y chart across its seam into the +Z chart at
    /// render distance 6: the grown window survives the crossing (the charts share storage +Y)
    /// and no near-square column, on either side of the seam, goes bare on any pass.
    #[test]
    fn seam_crossing_keeps_the_grown_window() {
        let mut world = chart_world(6, 3);
        let e = 6.5e-6;
        let at = |world: &World, t: f64| world.home_eye(DVec3::new(0.0, 1.0, 1.0 - e + 2.0 * e * t), 100.0);
        let start = at(&world, 0.0);
        world.prepare_around(start);
        world.drive_spawn_ready();
        settle(&mut world, start, NONE, "before the seam");
        assert!(world.window.grounded, "100 up holds the ground");
        let mut crossed = false;
        // Forty-first parts never land on the seam itself, where the faces tie.
        for k in 1..=40 {
            let fold = world.fold;
            let eye = at(&world, f64::from(k) / 41.0);
            walk(&mut world);
            step_finished(&mut world, eye);
            crossed |= world.fold != fold;
            assert!(world.window.grounded, "pass {k}: the window let go of the ground");
            let cols = near_columns(&world, world.center.expect("a centre"));
            assert_eq!(bare(&world, &cols), 0, "pass {k}: bare ground");
        }
        assert!(crossed, "the flight did not cross the seam");
        let end = at(&world, 1.0);
        let cols = near_columns(&world, world.center.expect("a centre"));
        settle(&mut world, end, &|w| bare(w, &cols), "past the seam");
    }

    /// Eye `t` of the way from the +Y chart across its seam into the +Z chart, 100 over the ground.
    /// `t` of 0 and 1 sit off the seam; 0.5 is the seam itself.
    fn seam_eye(world: &World, t: f64) -> DVec3 {
        let e = 6.5e-6;
        world.home_eye(DVec3::new(0.0, 1.0, 1.0 - e + 2.0 * e * t), 100.0)
    }

    /// `steps` samples from `from` to `to` that never land on the seam. `flight` plants an 80 m/s
    /// sample so the loading window shrinks; the seam step drops its sample and must keep that speed.
    fn cross_seam(world: &mut World, from: f64, to: f64, steps: i32, flight: bool) -> bool {
        let mut crossed = false;
        for k in 1..=steps {
            let fold = world.fold;
            let t = from + (to - from) * f64::from(k) / f64::from(steps + 1);
            let eye = seam_eye(world, t);
            if flight {
                let storage = world.stream_eye(eye);
                let prev = world.near_eye_prev.map(|(p, _)| p).unwrap_or(storage);
                let delta = storage - prev;
                let dir = if delta.length_squared() > 1.0 { delta.normalize() } else { DVec3::Z };
                let dt = Duration::from_millis(100);
                world.near_eye_prev = Some((storage - dir * 80.0 * dt.as_secs_f64(), crate::sched::now() - dt));
            } else {
                walk(world);
            }
            step(world, eye);
            crossed |= world.fold != fold;
            let cols = near_columns(world, world.center.expect("a centre"));
            assert_eq!(bare(world, &cols), 0, "t {t}: bare ground");
            if flight {
                let speed = world.stream_pacer.speed_mps();
                assert!(speed > FULL_EFFORT_SPEED_MPS && speed < 200.0, "t {t}: flight speed {speed}");
                assert!(!world.loading_full(), "t {t}: flight kept the whole loading window");
                if k >= 2 {
                    assert!(!world.window.grounded, "t {t}: a reduced window still reached for the ground");
                }
            } else {
                assert!(world.loading_full(), "t {t}: the walk shrank the loading window");
                assert!(world.window.grounded, "t {t}: the window let go of the ground");
            }
        }
        crossed
    }

    /// Walk the +Y/+Z seam three times, then fly back. No near column goes bare, the walk keeps the
    /// whole loading window, and flight stays reduced across the seam (the step drops its sample).
    #[test]
    fn seam_crossings_at_walk_and_flight_speed_leave_no_bare_ground() {
        let mut world = chart_world(6, 3);
        let start = seam_eye(&world, 0.0);
        world.prepare_around(start);
        world.drive_spawn_ready();
        settle(&mut world, start, NONE, "before the crossings");
        assert!(world.window.grounded, "100 up holds the ground");
        // Same pace as `seam_crossing_keeps_the_grown_window`: coarser steps cross the seam in one
        // frame, before the new chart's chunks exist, and that frame is bare.
        const STEPS: i32 = 40;
        let edge = f64::from(STEPS) / f64::from(STEPS + 1);
        for (from, to) in [(0.0, 1.0), (1.0, 0.0), (0.0, 1.0)] {
            assert!(cross_seam(&mut world, from, to, STEPS, false), "walk {from} -> {to} did not cross");
        }
        let walked = seam_eye(&world, edge);
        let cols = near_columns(&world, world.center.expect("a centre"));
        settle(&mut world, walked, &|w| bare(w, &cols), "after the walks");
        assert!(cross_seam(&mut world, 1.0, 0.0, STEPS, true), "the flight did not cross");
        walk(&mut world);
        let end = seam_eye(&world, 1.0 - edge);
        let cols = near_columns(&world, world.center.expect("a centre"));
        settle(&mut world, end, &|w| bare(w, &cols), "after the flight");
        assert!(world.window.grounded, "stopping did not grow the window back over the ground");
        assert!(world.loading_full(), "stopping left the loading window reduced");
    }

    /// A neighbour chart's section is millions of blocks away in its own storage and a few sections
    /// away in the net. Unload keeps that section; a section at the storage origin goes. A punched
    /// section is admitted until something draws it, and skipped once its ancestor does.
    #[test]
    fn neighbour_sections_are_measured_in_the_chart_net() {
        let mut world = chart_world(6, 3);
        let (_, far, _, _) = world.begin_stream(seam_eye(&world, 0.66), None);
        world.update_lod_face(far);
        world.refresh_frontier(far);
        let seat = world.seams.chart_seat(far).expect("the eye stands on a chart");
        let outside = |s: SectionPos| {
            let span = s.span() as i64;
            let (x, z) = (s.min_x() as i64, s.min_z() as i64);
            x + span <= seat.lo[0] || x >= seat.hi[0] || z + span <= seat.lo[2] || z >= seat.hi[2]
        };
        let neighbour = world
            .section_desired
            .iter()
            .copied()
            .filter(|s| outside(*s) && s.detail.0 <= FINEST_DETAIL.0)
            .min_by_key(|s| <SectionLane as StreamLane>::order(&world, far, *s))
            .expect("a neighbour section in the frontier");
        let span = neighbour.span() as i64;
        let cs = 16i64;
        let (psx, psz) = ((far.x as i64 * cs).div_euclid(span), (far.z as i64 * cs).div_euclid(span));
        let (sx, sz) = (
            (neighbour.min_x() as i64 + span / 2).div_euclid(span),
            (neighbour.min_z() as i64 + span / 2).div_euclid(span),
        );
        let raw = (sx - psx).unsigned_abs().max((sz - psz).unsigned_abs());
        let folded = <SectionLane as StreamLane>::order(&world, far, neighbour);
        assert!(raw > 10_000, "raw grid gap {raw} of {neighbour:?}");
        assert!(folded < 32, "folded order {folded} of {neighbour:?}");
        let mut ancestor = neighbour;
        while ancestor.detail.0 < FINEST_DETAIL.0 {
            ancestor = ancestor.parent();
        }
        let ancestor_order = <SectionLane as StreamLane>::order(&world, far, ancestor);
        assert!(ancestor_order < 32, "ancestor {ancestor:?} folds to order {ancestor_order}");
        let ready = ready_section;
        world.sections.insert(ancestor, ready());
        world.section_desired.retain(|s| *s != ancestor);
        world.section_visible.retain(|(s, _)| *s != ancestor);
        world.section_held.remove(&ancestor);
        world.unload_sections_with(far, |_| {});
        assert!(world.sections.contains_key(&ancestor), "unload dropped near neighbour {ancestor:?}");
        let origin = SectionPos { detail: ancestor.detail, body: ancestor.body, face: ancestor.face, x: 0, z: 0 };
        assert_ne!(origin, ancestor);
        world.sections.insert(origin, ready());
        world.unload_sections_with(far, |_| {});
        assert!(!world.sections.contains_key(&origin), "a section at the storage origin stayed resident");
        assert!(world.sections.contains_key(&ancestor), "the second unload dropped the neighbour");
        world.sections.remove(&ancestor);
        world.section_held.insert(neighbour, [0, 0]);
        assert!(!world.coverage_skips(far, neighbour), "a held section with nothing drawing it was skipped");
        world.sections.insert(ancestor, ready());
        assert!(world.section_covered(neighbour), "the ancestor does not cover {neighbour:?}");
        assert!(world.coverage_skips(far, neighbour), "a held section already drawn was still a hole");
    }

    /// A pit dug 100 blocks down from the surface near spawn reaches below the window's natural
    /// floor. Coming down from 320 to stand beside it, the window reaches the pit's floor, and no
    /// column (the pit's own counted at its floor) goes bare on any pass.
    #[test]
    fn pit_below_the_window_floor_is_never_bare() {
        let mut world = chart_world(6, 3);
        let dir = spawn_dir(&world);
        let low = world.home_eye(dir, 1.62);
        let s = world.chart_eye(low).expect("charted");
        let (px, pz) = ((s.x.floor() as i32).div_euclid(16) * 16 + 56, (s.z.floor() as i32).div_euclid(16) * 16 + 8);
        let top = world.generator.surface(Face::PosY, px, pz) - 1;
        let floor = top - 100;
        for x in px - 1..=px + 1 {
            for z in pz - 1..=pz + 1 {
                for y in floor..=world.generator.surface(Face::PosY, x, z) {
                    world.set_block(x, y, z, AIR);
                }
            }
        }
        let high = world.home_eye(dir, 320.0);
        world.prepare_around(high);
        world.drive_spawn_ready();
        settle(&mut world, high, NONE, "hover");
        let mut cols = near_columns(&world, eye_chunk(s));
        let pit = cols.iter_mut().find(|c| (c.0, c.1) == (px, pz)).expect("the pit is in the near square");
        pit.2 = floor - 1;
        for k in 1..=40 {
            let eye = world.home_eye(dir, 320.0 - 8.0 * f64::from(k));
            step(&mut world, eye);
            assert_eq!(bare(&world, &cols), 0, "descent pass {k}: bare ground");
        }
        settle(&mut world, low, &|w| bare(w, &cols), "beside the pit");
        let [lo, _] = world.window.punch.expect("a window");
        assert!(lo <= (floor - 1).div_euclid(16), "the window floor {lo} misses the pit floor");
    }

    /// A held section whose ground has settled but for one chunk parked by quarantine stops
    /// waiting: the parked chunk is a bounded hole, not a far section held over it for good.
    #[test]
    fn quarantine_does_not_hold_a_section() {
        let mut world = chart_world(6, 3);
        let (center, far, _, _) = world.begin_stream(world.home_eye(spawn_dir(&world), 1.62), None);
        world.update_lod_face(far);
        world.refresh_frontier(far);
        let (&s, &layers) = world.section_held.iter().min_by_key(|(s, _)| (s.span(), s.x, s.z)).expect("held sections wait");
        let n = 16;
        let mut chunks: Vec<Coord> = Vec::new();
        for y in layers[0]..=layers[1] {
            for z in s.min_z().div_euclid(n)..(s.min_z() + s.span()).div_euclid(n) {
                for x in s.min_x().div_euclid(n)..(s.min_x() + s.span()).div_euclid(n) {
                    chunks.push(Coord::new(x, y, z));
                }
            }
        }
        // Load first: a neighbour landing later can reopen a buried chunk.
        for &c in &chunks {
            world.ensure_data(c);
        }
        for &c in &chunks {
            world.chunks.get_mut(&c).expect("loaded").state = MeshState::Air;
        }
        let parked = chunks[0];
        world.chunks.get_mut(&parked).expect("loaded").state = MeshState::needs_mesh();
        world.held_recheck.set();
        world.refresh_frontier(far);
        assert!(world.section_held.contains_key(&s), "an unsettled chunk keeps the section waiting");
        world.quarantined.insert(FailKey::Mesh { coord: parked });
        world.held_recheck.set();
        world.refresh_frontier(far);
        assert!(!world.section_held.contains_key(&s) && !world.section_desired.contains(&s), "quarantine held the section");
        assert_eq!(world.center, Some(center));
    }

    /// Off a chart, landing from 400 on a flat world at render distance 6: far sections over
    /// ground the settled chunks draw hand over at once, on no pass is one drawn, and once every
    /// chunk of the window has settled the rest of the overlap inside the clip's reach (the
    /// outermost ring stays the far field's by design) ends within a few passes.
    #[test]
    fn flat_overlap_ends_once_the_ground_settles() {
        use crate::world::generation::FLAT_HEIGHT;

        let render = RenderConfig { lod2: true, occlusion: true, ..RenderConfig::default() };
        let mut world = World::with_kind(1, render, WorldgenKind::Flat, false);
        world.set_view_distances(6, 3);
        let at = |above: f64| DVec3::new(8.5, f64::from(FLAT_HEIGHT) + above, 8.5);
        world.prepare_around(at(400.0));
        world.drive_spawn_ready();
        settle(&mut world, at(400.0), NONE, "hover");
        let c = world.center.expect("a centre");
        let inner = world.view.horizontal - 1;
        let cols: Vec<_> = near_columns(&world, c)
            .into_iter()
            .filter(|&(x, z, _)| (x.div_euclid(16) - c.x).abs() <= inner && (z.div_euclid(16) - c.z).abs() <= inner)
            .collect();
        let overlap = |w: &World, cam: DVec3| {
            let (min, max) = w.lod_clip_box(cam);
            cols.iter()
                .filter(|&&(x, z, top)| {
                    let rel = [f64::from(x) - cam.x, f64::from(top) + 0.5 - cam.y, f64::from(z) - cam.z];
                    let clipped = (0..3).all(|a| f64::from(min[a]) < rel[a] && rel[a] < f64::from(max[a]));
                    w.chunk_final(Coord::new(x.div_euclid(16), top.div_euclid(16), z.div_euclid(16))) && !clipped && far_drawn(w, x, z)
                })
                .count()
        };
        let handed = |w: &World| {
            let c = w.center.expect("a centre");
            w.section_visible.iter().filter(|&&(s, _)| w.full_res_covers(c, s)).count()
        };
        let mut settled_at = None;
        for pass in 0..20_000usize {
            let above = (400.0 - 8.0 * pass as f64).max(60.0);
            step(&mut world, at(above));
            assert_eq!(handed(&world), 0, "pass {pass}: a section the chunks draw is still drawn");
            let center = world.center.expect("a centre");
            let window_settled = world.view_coords(world.mesh_box(center)).all(|c| world.chunk_final(c));
            if above == 60.0 && window_settled {
                let since = *settled_at.get_or_insert(pass);
                let left = overlap(&world, at(above));
                assert!(left == 0 || pass - since < 8, "{left} columns overlap {} passes after the window settled", pass - since);
                if left == 0 {
                    println!("overlap ended {} passes after the window settled", pass - since);
                    return;
                }
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        panic!("the window never settled: {}", world.entry_debug());
    }

    /// Mesh teardown (a lighting or AO toggle frees every mesh) while a grown window's span is
    /// still being proven: neither span keeps its old rings, so the clip box never hides the far
    /// field over a chunk that is not drawn while everything remeshes.
    #[test]
    fn mesh_teardown_during_a_pending_span_is_never_bare() {
        use crate::world::generation::FLAT_HEIGHT;

        let render = RenderConfig { lod2: true, occlusion: true, ..RenderConfig::default() };
        let mut world = World::with_kind(1, render, WorldgenKind::Flat, false);
        world.set_view_distances(6, 3);
        let at = |above: f64| DVec3::new(8.5, f64::from(FLAT_HEIGHT) + above, 8.5);
        world.prepare_around(at(400.0));
        world.drive_spawn_ready();
        settle(&mut world, at(400.0), NONE, "hover");
        let mut k = 0;
        while world.lod_clip_next.is_none() {
            k += 1;
            assert!(k <= 45, "the descent never left a span to prove");
            step(&mut world, at(400.0 - 8.0 * f64::from(k)));
        }
        let eye = at(400.0 - 8.0 * f64::from(k));
        let cols = near_columns(&world, world.center.expect("a centre"));
        let coords: Vec<Coord> = world.chunks.keys().copied().collect();
        for c in coords {
            let loaded = world.chunks.get_mut(&c).expect("loaded");
            loaded.rev = loaded.rev.wrapping_add(1);
            loaded.retire_logged(MeshState::needs_mesh());
        }
        world.building_meshes = 0;
        world.mesh_worklist.extend(world.chunks.keys().copied().collect::<Vec<_>>());
        world.pending_fresh.set();
        world.lod_clip_shrunk.set();
        world.refresh_lod_clip();
        assert!(world.lod_clip_next.is_none_or(|(_, r)| r == 0), "the pending span kept its rings");
        assert_eq!(clip_bare(&world, eye, &cols), 0, "bare right after the teardown");
        settle(&mut world, eye, &|w| clip_bare(w, eye, &cols), "remesh");
    }

    fn gentle_world() -> World {
        let gentle = crate::world::terrain::TerrainCfg { relief: 25, ..Default::default() };
        World::with_kind_cfg(
            crate::world::DEFAULT_SEED,
            RenderConfig::default(),
            WorldgenKind::Diffusion,
            gentle,
            true,
        )
    }

    /// Reference-frame chunks under `key`, or the storage chunks that hold them when `storage` is set.
    fn footprint_chunks(world: &World, key: SectionPos, storage: Option<Coord>) -> Vec<Coord> {
        let frame = FaceFrame::new(key.face);
        let (lo, hi) = world.section_relief_band(key).expect("baked");
        let (a0, a1) = (world.baked_world_y(key, lo), world.baked_world_y(key, hi));
        let span = key.span();
        let (u0, v0) = (key.min_x(), key.min_z());
        let (u1, v1) = (u0 + span - 1, v0 + span - 1);
        let mut lo_c = [i32::MAX; 3];
        let mut hi_c = [i32::MIN; 3];
        for a in [a0, a1] {
            for u in [u0, u1] {
                for v in [v0, v1] {
                    let (x, y, z) = frame.cell_to_world((u, a, v));
                    let c = World::chunk_of(x, y, z);
                    let p = [c.x, c.y, c.z];
                    for i in 0..3 {
                        lo_c[i] = lo_c[i].min(p[i]);
                        hi_c[i] = hi_c[i].max(p[i]);
                    }
                }
            }
        }
        let d = storage
            .map(|s| {
                let (reference, _, _) = world.lod_place(s);
                [
                    i64::from(reference.x) - i64::from(s.x),
                    i64::from(reference.y) - i64::from(s.y),
                    i64::from(reference.z) - i64::from(s.z),
                ]
            })
            .unwrap_or([0; 3]);
        let mut out = Vec::new();
        for y in lo_c[1]..=hi_c[1] {
            for z in lo_c[2]..=hi_c[2] {
                for x in lo_c[0]..=hi_c[0] {
                    out.push(Coord::new(
                        (i64::from(x) - d[0]) as i32,
                        (i64::from(y) - d[1]) as i32,
                        (i64::from(z) - d[2]) as i32,
                    ));
                }
            }
        }
        out
    }

    /// A warped cube's near section is proved in the reference face frame and backed by storage
    /// chunks. Reference-frame coords, which the old proof looked up, do not cover it.
    #[test]
    fn cube_coverage_reads_storage_chunks() {
        use crate::ident::Detail;
        use crate::world::heightmip::BakeExtent;
        use crate::world::terrain::cosmos::{Kind, Shape};

        for face in [Face::PosY, Face::PosX] {
            let mut world = gentle_world();
            // A finest section is 128 blocks wide. The proof wants its farthest corner inside
            // 0.75 of the near radius, which a view of 8 (96 blocks) can never show. 18 reaches
            // 216. The bake's finest radius is half_m >> 3, so 4096 covers the near square.
            world.set_view_distances(18, 5);
            let twin = world
                .generator
                .cosmos()
                .expect("cosmos")
                .bodies()
                .iter()
                .copied()
                .filter(|b| b.kind == Kind::Twin)
                .nth(1)
                .expect("twin 2");
            let Shape::Cube { .. } = twin.shape else { panic!("twin is a cube") };
            let centre = (
                i32::try_from(twin.centre[0]).unwrap(),
                i32::try_from(twin.centre[1]).unwrap(),
                i32::try_from(twin.centre[2]).unwrap(),
            );
            let (cu, _, cv) = FaceFrame::new(face).cell_to_local(centre);
            let open = world.generator.surface(face, cu, cv);
            assert_ne!(open, i32::MIN, "{face:?} has no surface");
            let grid = world
                .generator
                .atlases()
                .iter()
                .find_map(|a| a.grid.filter(|g| g.body == twin.id))
                .expect("cube box");
            let (rx, ry, rz) = FaceFrame::new(face).cell_to_world((cu, open, cv));
            let cs = CHUNK_SIZE as i64;
            let d = [
                (grid.ref_min[0] - grid.origin[0]) / cs,
                (grid.ref_min[1] - grid.origin[1]) / cs,
                (grid.ref_min[2] - grid.origin[2]) / cs,
            ];
            let reference = World::chunk_of(rx, ry, rz);
            let storage = Coord::new(
                (i64::from(reference.x) - d[0]) as i32,
                (i64::from(reference.y) - d[1]) as i32,
                (i64::from(reference.z) - d[2]) as i32,
            );
            world.section_eye_y = (i64::from(ry) - (grid.ref_min[1] - grid.origin[1])) as f64 + 1.62;
            world.adopt_fold(storage);
            world.stream_up = world.resolve_stream_up(storage);
            world.stream_up_set = true;
            world.center = Some(storage);
            assert!(!world.fold.is_identity(), "{face:?} box is physical space");
            assert_eq!(world.live_up(), Some(face), "{face:?} centre sky");
            let (placed, _, in_cube) = world.lod_place(storage);
            assert!(in_cube && placed == reference, "{face:?} lod_place {placed:?} != {reference:?}");
            assert_ne!(World::face_alt_shift(storage, placed, face), 0, "{face:?} box did not translate");
            world.section_lod_face = world.dominant_lod_face(storage);
            world.section_face_set = true;
            assert_eq!(world.section_lod_face, Some((twin.id, face)), "{face:?} dominant face");
            let (au, av) = world.face_tangent_centre(storage, face);
            world.section_mip = Some(HeightMip::bake_at(
                &*world.generator,
                &world.registry.color_snapshot(),
                BakeExtent::new(4096, Detail(FINEST_DETAIL.0 + 3)),
                au,
                av,
                face,
                twin.id,
            ));
            assert!(world.place_window(storage, true));
            assert!(world.window.ground.is_some() && world.window.grounded, "{face:?} window did not read the face");
            let punch = world.window.punch.expect("punch");
            let (_, eye) = ColumnKey::of(face, storage);
            assert!(punch[0] <= eye - 4 && punch[1] >= eye + 4, "{face:?} punch {punch:?} dropped the eye");
            assert_eq!(world.lod_clip().half.x, 0.0, "{face:?} clip came on");
            // A coarse tile must not cross the skip disk: the part over the player is a wholly
            // inside piece (the chunks draw it) and the part outside stops at detail 0.
            let h_lim = 0.75 * world.view.coverage().half.x;
            let margin = (CHUNK_SIZE as f32) * 0.5 * std::f32::consts::SQRT_2;
            let n = CHUNK_SIZE as i32;
            let mid = |c: i32| (i64::from(c) * i64::from(n) + i64::from(n / 2)) as i32;
            let (eu, _, ev) = FaceFrame::new(face).cell_to_local((mid(placed.x), mid(placed.y), mid(placed.z)));
            world.gpu_live_slots = 6000;
            let desired = world.desired_sections(storage);
            assert!(
                desired.len() <= crate::world::SECTION_SLOT_FLOOR,
                "{face:?} frontier {} exceeds the section floor",
                desired.len()
            );
            let shift = World::face_alt_shift(storage, placed, face);
            let [a0, a1] = world.window_alts().expect("window");
            for s in desired.iter().copied().filter(|s| s.face == face && s.detail.0 > 0) {
                let (far, near) = super::super::span_reach(s, eu, ev);
                assert!(
                    !(near < h_lim - margin && far + margin > h_lim),
                    "{face:?} coarse tile {s:?} still crosses the near disk"
                );
                if far + margin <= h_lim
                    && let Some((lo, hi)) = world.cover_band(s, true)
                {
                    let (floor, ceil) = (world.baked_height(s, a0 + shift), world.baked_height(s, a1 + shift));
                    assert!(
                        lo >= floor && hi <= ceil,
                        "{face:?} {s:?} is inside the disk with relief outside the window"
                    );
                }
            }
            let span = section_span(FINEST_DETAIL);
            let cell = SectionPos {
                body: twin.id,
                face,
                detail: FINEST_DETAIL,
                x: cu.div_euclid(span),
                z: cv.div_euclid(span),
            };
            assert!(world.section_relief_band(cell).is_some(), "{face:?} stand section is not baked");
            assert!(!world.full_res_covers(storage, cell), "{face:?} unbacked section was skipped");
            for c in footprint_chunks(&world, cell, None) {
                world.chunks.insert(c, air_loaded(c.x, c.y, c.z));
            }
            assert!(!world.full_res_covers(storage, cell), "{face:?} reference chunks covered a storage section");
            world.chunks.clear();
            let stored = footprint_chunks(&world, cell, Some(storage));
            assert!(!stored.is_empty());
            let stand = {
                let rc = World::chunk_of(rx, ry, rz);
                Coord::new((i64::from(rc.x) - d[0]) as i32, (i64::from(rc.y) - d[1]) as i32, (i64::from(rc.z) - d[2]) as i32)
            };
            assert!(stored.contains(&stand), "{face:?} stand chunk {stand:?} is outside the footprint");
            for c in &stored {
                world.chunks.insert(*c, air_loaded(c.x, c.y, c.z));
            }
            assert!(world.full_res_covers(storage, cell), "{face:?} storage-backed section was kept");
            world.chunks.insert(stand, Loaded {
                state: MeshState::NeedsMesh { building: true, prev: None },
                ..air_loaded(stand.x, stand.y, stand.z)
            });
            assert!(!world.full_res_covers(storage, cell), "{face:?} an in-flight chunk still skipped");
        }
    }

    /// Columns of the near square in the streaming centre's face frame: face-local `(u, v)` and the
    /// storage chunk holding the solid top.
    fn cube_columns(world: &World) -> Vec<(i32, i32, Coord)> {
        let Some(center) = world.center else { return Vec::new() };
        let Some(face) = world.live_up() else { return Vec::new() };
        let (reference, _, in_cube) = world.lod_place(center);
        if !in_cube {
            return Vec::new();
        }
        let frame = FaceFrame::new(face);
        let (cu, _, cv) = frame.chunk_to_local(reference);
        let h = world.view.horizontal;
        let cs = CHUNK_SIZE as i32;
        let (dx, dy, dz) = (
            i64::from(reference.x) - i64::from(center.x),
            i64::from(reference.y) - i64::from(center.y),
            i64::from(reference.z) - i64::from(center.z),
        );
        let mut out = Vec::new();
        for dv in -h..=h {
            for du in -h..=h {
                let w = frame.chunk_to_world((cu + du, 0, cv + dv));
                let p = (
                    (i64::from(w.x) * i64::from(cs) + i64::from(cs / 2)) as i32,
                    (i64::from(w.y) * i64::from(cs) + i64::from(cs / 2)) as i32,
                    (i64::from(w.z) * i64::from(cs) + i64::from(cs / 2)) as i32,
                );
                let (u, _, v) = frame.cell_to_local(p);
                let open = world.generator.surface(face, u, v);
                if open == i32::MIN {
                    continue;
                }
                let (x, y, z) = frame.cell_to_world((u, open - 1, v));
                let rc = World::chunk_of(x, y, z);
                out.push((
                    u,
                    v,
                    Coord::new((i64::from(rc.x) - dx) as i32, (i64::from(rc.y) - dy) as i32, (i64::from(rc.z) - dz) as i32),
                ));
            }
        }
        out
    }

    fn cube_far(world: &World, face: Face, body: u16, u: i32, v: i32) -> bool {
        world.section_visible.iter().any(|&(s, mask)| {
            if s.face != face || s.body != body {
                return false;
            }
            let half = s.span() / 2;
            in_rect(s, u, v) && mask.iter().any(|q| (u - s.min_x()) / half == q.dx() && (v - s.min_z()) / half == q.dz())
        })
    }

    fn cube_bare(world: &World) -> usize {
        let Some((body, face)) = world.section_lod_face else { return 0 };
        if world.live_up() != Some(face) {
            return 0;
        }
        cube_columns(world)
            .into_iter()
            .filter(|&(u, v, c)| !world.chunks.get(&c).is_some_and(|l| l.state.settled()) && !cube_far(world, face, body, u, v))
            .count()
    }

    /// Visible sections on `face` whose footprint lies inside the proof (farthest corner within
    /// 0.75 of the near radius, relief inside the window). Also prints how many remain inside the
    /// full near radius, and why the proof does not cover them.
    fn proof_hits(world: &World, face: Face) -> (usize, usize) {
        let center = world.center.expect("a centre");
        let (reference, _, _) = world.lod_place(center);
        let frame = FaceFrame::new(face);
        let cs = CHUNK_SIZE as i32;
        let mid = |c: i32| (i64::from(c) * i64::from(cs) + i64::from(cs / 2)) as i32;
        let (eu, _, ev) = frame.cell_to_local((mid(reference.x), mid(reference.y), mid(reference.z)));
        let radius = world.view.horizontal * cs;
        let h_lim = 0.75 * radius as f32;
        let margin = cs as f32 * 0.5 * std::f32::consts::SQRT_2;
        let shift = World::face_alt_shift(center, reference, face);
        let window = world.window_alts().map(|[a0, a1]| (a0 + shift, a1 + shift));
        let (mut within, mut proof, mut straddling, mut relief_out, mut other_face) = (0, 0, 0, 0, 0);
        let mut by_detail = [0usize; 10];
        let mut nearest = i32::MAX;
        let mut nearest_coarse = i32::MAX;
        for &(s, _) in &world.section_visible {
            if s.face != face {
                other_face += 1;
                continue;
            }
            let span = s.span();
            let (x0, z0) = (s.min_x(), s.min_z());
            let dx = if eu < x0 { x0 - eu } else if eu > x0 + span - 1 { eu - (x0 + span - 1) } else { 0 };
            let dz = if ev < z0 { z0 - ev } else if ev > z0 + span - 1 { ev - (z0 + span - 1) } else { 0 };
            let gap = dx.max(dz);
            if gap >= radius {
                continue;
            }
            within += 1;
            nearest = nearest.min(gap);
            if s.detail.0 >= 2 {
                nearest_coarse = nearest_coarse.min(gap);
            }
            if (s.detail.0 as usize) < by_detail.len() {
                by_detail[s.detail.0 as usize] += 1;
            }
            let fx = (x0 - eu).abs().max((x0 + span - eu).abs()) as f32;
            let fz = (z0 - ev).abs().max((z0 + span - ev).abs()) as f32;
            let inside = (fx * fx + fz * fz).sqrt() + margin <= h_lim;
            let relief_in = match (window, world.cover_band(s, true)) {
                (Some((a0, a1)), Some((lo, hi))) => {
                    lo >= world.baked_height(s, a0) && hi <= world.baked_height(s, a1)
                }
                _ => false,
            };
            if inside && relief_in {
                proof += 1;
                println!("  proof hit {s:?} span {span} gap {gap}");
            } else if !inside && gap * 4 < radius * 3 {
                straddling += 1;
            } else if inside {
                relief_out += 1;
            }
        }
        println!(
            "{face:?}: {within} visible within {radius} blocks ({proof} inside the proof, {straddling} reaching the 0.75 core but not wholly inside, {relief_out} wholly inside with relief outside the window); nearest {nearest} nearest detail>=2 {nearest_coarse} by detail {by_detail:?}; {other_face} visible on another face; window {:?} grounded {}",
            world.window.punch, world.window.grounded
        );
        (within, proof)
    }

    fn twin_under(world: &World, p: DVec3) -> crate::world::terrain::cosmos::Body {
        use crate::world::terrain::cosmos::Kind;
        let cosmos = world.generator.cosmos().expect("cosmos");
        cosmos
            .bodies()
            .iter()
            .copied()
            .filter(|b| b.kind == Kind::Twin)
            .min_by(|a, b| cosmos.altitude(a, p).abs().total_cmp(&cosmos.altitude(b, p).abs()))
            .expect("a twin")
    }

    /// Physical point on `dir` from `body`'s centre at the same altitude above the cube datum as
    /// the +Y face-centre eye (`surface + 2`).
    fn physical_on(world: &World, body: &crate::world::terrain::cosmos::Body, dir: DVec3) -> DVec3 {
        use crate::world::terrain::cosmos::Shape;
        let cosmos = world.generator.cosmos().expect("cosmos");
        let Shape::Cube { half } = body.shape else { panic!("cube") };
        let centre = (
            i32::try_from(body.centre[0]).unwrap(),
            i32::try_from(body.centre[1]).unwrap(),
            i32::try_from(body.centre[2]).unwrap(),
        );
        let (cu, _, cv) = FaceFrame::new(Face::PosY).cell_to_local(centre);
        let open = world.generator.surface(Face::PosY, cu, cv);
        let above = f64::from(open + 2) - (body.centre[1] as f64 + half as f64);
        let (mut lo, mut hi) = (0.0, 2.0 * half as f64);
        let c = body.centre_f();
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if cosmos.altitude(body, c + dir * mid) < above { lo = mid } else { hi = mid }
        }
        c + dir * lo
    }

    fn settle_twin(world: &mut World, eye: DVec3, face: Face) {
        world.prepare_around(eye);
        world.drive_spawn_ready();
        let passes = settle(world, eye, NONE, "stand");
        for _ in 0..8 {
            if world.window.grounded && !world.window.stale {
                break;
            }
            step(world, eye);
        }
        let center = world.center.expect("a centre");
        println!(
            "settled in {passes} at {eye:?}: centre {center:?} up {:?} face {:?} window {:?} grounded {} bare {}",
            world.live_up(),
            world.section_lod_face,
            world.window.punch,
            world.window.grounded,
            cube_bare(world)
        );
        assert_eq!(world.live_up(), Some(face), "stood on {face:?}");
        assert!(world.window.grounded, "{face:?} window never reached the face");
        assert_eq!(world.lod_clip().half, voxel_engine::Vec3::ZERO, "{face:?} clip came on");
        assert_eq!(cube_bare(world), 0, "{face:?} bare ground after settling");
        let (within, proof) = proof_hits(world, face);
        assert_eq!(proof, 0, "{face:?}: {within} sections within the near radius, {proof} inside the proof");
        let covered = world.section_visible.iter().filter(|&&(s, _)| world.full_res_covers(center, s)).count();
        assert_eq!(covered, 0, "{face:?}: a section the chunks draw is still drawn");
    }

    /// Seed 42, RD16/V5, twin 2. Before the frame fix, nine detail-2 sections stayed visible 0–120
    /// blocks from the +Y eye. After settling, none that the proof covers (footprint wholly inside
    /// 0.75 of the near radius, relief inside the window) is still drawn.
    /// `cargo test --release --lib twin_cube_lod -- --ignored --nocapture --test-threads=1`
    #[test]
    #[ignore]
    fn twin_cube_lod_stays_off_the_player() {
        let y_eye = DVec3::new(296_989_696.0, -308_421_885.0, -370_431_824.0);
        let mut world = seeded(42, 16, 5);
        let twin = twin_under(&world, y_eye);
        let y_bisect = physical_on(&world, &twin, DVec3::Y);
        let dy = (y_eye - y_bisect).length();
        println!(
            "+Y given {y_eye:?} bisect {y_bisect:?} altitude {} delta {dy} body {} centre {:?}",
            world.generator.cosmos().unwrap().altitude(&twin, y_eye),
            twin.id,
            twin.centre
        );
        // Same column as the face-centre ray. The published eye sits a few blocks above
        // `surface + 2` once the cube's warp is inverted; both are the +Y stand.
        assert!(y_bisect.x == y_eye.x && y_bisect.z == y_eye.z && dy < 8.0, "+Y stand drifted from the bisection by {dy}");
        settle_twin(&mut world, y_eye, Face::PosY);

        let x_eye = physical_on(&world, &twin, DVec3::X);
        println!("+X stand {x_eye:?} altitude {}", world.generator.cosmos().unwrap().altitude(&twin, x_eye));
        let mut world = seeded(42, 16, 5);
        settle_twin(&mut world, x_eye, Face::PosX);
    }

    /// From 600 above twin 2's +Y stand, eight blocks a pass: every near column is drawn by a
    /// settled chunk or a far section. Same bare rule as the chart descent, in the cube's frames.
    #[test]
    #[ignore]
    fn twin_cube_descent_never_bares_the_near_columns() {
        let stand = DVec3::new(296_989_696.0, -308_421_885.0, -370_431_824.0);
        let mut world = seeded(42, 16, 5);
        let high = stand + DVec3::new(0.0, 600.0, 0.0);
        world.prepare_around(high);
        world.drive_spawn_ready();
        settle(&mut world, high, NONE, "high");
        assert_eq!(world.live_up(), Some(Face::PosY), "the descent is not on +Y");
        assert_eq!(world.section_lod_face.map(|(_, f)| f), Some(Face::PosY), "the hover has no face");
        assert_eq!(cube_bare(&world), 0, "bare at 600 above");
        for k in 1..=75 {
            let eye = stand + DVec3::new(0.0, 600.0 - 8.0 * f64::from(k), 0.0);
            step(&mut world, eye);
            assert_eq!(cube_bare(&world), 0, "descent pass {k}: bare ground");
        }
        let passes = settle(&mut world, stand, &|w| cube_bare(w), "land");
        println!("descent settled in {passes}, window {:?} bare {}", world.window.punch, cube_bare(&world));
        assert_eq!(cube_bare(&world), 0, "bare on the stand");
    }

    /// Seed 42, RD16/V5, LOD on. One ignored measurement per body kind the chart, cube and flat
    /// suites do not already stand on. A failure is a visible far section wholly inside the covered
    /// disk whose relief the window holds. Bare columns after settling must be none.
    /// `cargo test --release --lib near_lod_ -- --ignored --nocapture --test-threads=1`

    fn say(line: &str) {
        println!("{line}");
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }

    fn body_of(world: &World, kind: crate::world::terrain::cosmos::Kind) -> crate::world::terrain::cosmos::Body {
        world.generator.cosmos().expect("cosmos").bodies().iter().copied().find(|b| b.kind == kind).expect("body")
    }

    /// The round chart of `body`: outward ball, or the shell surface `inward` names.
    fn round_atlas<'a>(world: &'a World, body: &crate::world::terrain::cosmos::Body, inward: bool) -> &'a crate::space::atlas::Atlas {
        use crate::world::terrain::cosmos::Shape;
        let want = match body.shape {
            Shape::Ball { r } => r,
            Shape::Shell { outer, inner } => {
                if inward {
                    inner
                } else {
                    outer
                }
            }
            Shape::Cube { .. } => panic!("a cube has no round chart"),
        };
        world
            .generator
            .atlases()
            .iter()
            .find(|a| a.grid.is_none() && a.inward == inward && (a.centre - body.centre_f()).length() < 1.0 && (a.radius - want).abs() < 16)
            .expect("chart")
            .as_ref()
    }

    struct ChartStand {
        x: i32,
        z: i32,
        /// First open cell of the column (the block above the solid top).
        open: i32,
        storage: DVec3,
        physical: DVec3,
        roundtrip: f64,
        relief: i32,
    }

    /// A clear column on the +Y chart, in the interior sample whose ground varies most.
    fn chart_stand(world: &World, atlas: &crate::space::atlas::Atlas) -> ChartStand {
        use crate::space::atlas::Patch;
        let patch = Patch::Shell { band: 0, face: Face::PosY };
        let (_, size) = atlas.storage_box(patch);
        let n = size[0];
        let margin = 1024i64.min(n / 4);
        let mid = n / 2;
        let anchors = [(0i64, 0i64), (2048, 0), (-2048, 1536), (0, -3072), (4096, 4096)];
        let mut best: Option<(i32, Vec<(i32, i32, i32)>)> = None;
        let mut seen = Vec::new();
        for (dx, dz) in anchors {
            let ax = (mid + dx).clamp(margin, n - margin - 1);
            let az = (mid + dz).clamp(margin, n - margin - 1);
            if seen.contains(&(ax, az)) {
                continue;
            }
            seen.push((ax, az));
            let mut samples = Vec::new();
            let (mut lo, mut hi) = (i32::MAX, i32::MIN);
            for oz in (-256i64..=256).step_by(64) {
                for ox in (-256i64..=256).step_by(64) {
                    let lx = (ax + ox).clamp(margin, n - margin - 1);
                    let lz = (az + oz).clamp(margin, n - margin - 1);
                    let s = atlas.storage(patch, [lx, 0, lz]);
                    let (x, z) = (i32::try_from(s[0]).expect("x"), i32::try_from(s[2]).expect("z"));
                    let y = world.generator.surface(Face::PosY, x, z);
                    if y == i32::MIN || y >= crate::world::terrain::storage::BURIED {
                        continue;
                    }
                    lo = lo.min(y);
                    hi = hi.max(y);
                    samples.push((x, z, y));
                }
            }
            let relief = if samples.is_empty() { 0 } else { hi - lo };
            if best.as_ref().is_none_or(|(r, _)| relief > *r) {
                best = Some((relief, samples));
            }
        }
        let (relief, mut samples) = best.expect("a chart sample");
        samples.sort_by_key(|s| s.2);
        let air = crate::block::registry::AIR;
        let mut chosen = None;
        for (x, z, open) in samples {
            if !world.registry.is_solid(world.generator.voxel_at(x, open - 1, z)) {
                continue;
            }
            for lift in 0..8 {
                let y = open + lift;
                let clear = world.generator.voxel_at(x, y, z) == air
                    && world.generator.voxel_at(x, y + 1, z) == air
                    && world.generator.voxel_at(x, y + 2, z) == air;
                if clear {
                    chosen = Some((x, z, open, y));
                    break;
                }
            }
            if chosen.is_some() {
                break;
            }
        }
        let (x, z, open, eye_block) = chosen.expect("a clear column");
        let cell = [i64::from(x), i64::from(open), i64::from(z)];
        let (patch, _) = atlas.locate(cell).unwrap_or_else(|| panic!("surface {cell:?} left the chart"));
        let eye_y = if atlas.locate([i64::from(x), i64::from(eye_block) + 1, i64::from(z)]).is_some() {
            f64::from(eye_block) + 1.62
        } else {
            f64::from(eye_block) + 0.5
        };
        let storage = DVec3::new(f64::from(x) + 0.5, eye_y, f64::from(z) + 0.5);
        let physical = atlas.embed_storage(patch, storage);
        let roundtrip = world.chart_eye(physical).map(|p| (p - storage).length()).unwrap_or(f64::MAX);
        ChartStand { x, z, open, storage, physical, roundtrip, relief }
    }

    /// Settle on a chart and print the near-LOD census. Returns passes.
    fn audit_chart(world: &mut World, name: &str, stand: &ChartStand, inward: bool, radius: i64) -> usize {
        say(&format!(
            "STAND {name} phys {:.1} {:.1} {:.1} storage {:.1} {:.1} {:.1} relief {} roundtrip {:.3} open {}",
            stand.physical.x, stand.physical.y, stand.physical.z, stand.storage.x, stand.storage.y, stand.storage.z, stand.relief, stand.roundtrip, stand.open
        ));
        world.prepare_around(stand.physical);
        world.drive_spawn_ready();
        let passes = settle(world, stand.physical, NONE, name);
        for _ in 0..8 {
            if world.window.grounded && !world.window.stale {
                break;
            }
            step(world, stand.physical);
        }
        let center = world.center.expect("a centre");
        let cols = near_columns(world, center);
        let bare_n = bare(world, &cols);
        let (y0, y1) = world.near_y_range(center);
        let near = world.near_block_box(center);
        let radius_blocks = 16 * world.view.horizontal;
        let (ex, ez) = (stand.x, stand.z);
        let mut near_n = 0usize;
        let mut failures = 0usize;
        let mut other_face = 0usize;
        let mut by_detail = [0usize; 12];
        let mut nearest = i32::MAX;
        let mut hits = String::new();
        for &(s, _) in &world.section_visible {
            if s.face != Face::PosY {
                other_face += 1;
                continue;
            }
            let span = s.span();
            let (x0, z0) = (s.min_x(), s.min_z());
            let dx = if ex < x0 {
                x0 - ex
            } else if ex > x0 + span - 1 {
                ex - (x0 + span - 1)
            } else {
                0
            };
            let dz = if ez < z0 {
                z0 - ez
            } else if ez > z0 + span - 1 {
                ez - (z0 + span - 1)
            } else {
                0
            };
            let dist = (f64::from(dx) * f64::from(dx) + f64::from(dz) * f64::from(dz)).sqrt();
            if dist >= f64::from(radius_blocks) {
                continue;
            }
            near_n += 1;
            nearest = nearest.min(dx.max(dz));
            let slot = (s.detail.0 as usize).min(by_detail.len() - 1);
            by_detail[slot] += 1;
            let wholly = super::super::inside_near(s, near, None);
            let mut relief_in = false;
            if wholly
                && let Some((lo, hi)) = world.generator.surface_rect(s.body, Face::PosY, x0, z0, x0 + span, z0 + span)
            {
                let (top_lo, top_hi) = (lo.saturating_sub(1), hi.saturating_sub(1));
                relief_in = i64::from(top_lo) >= y0 && i64::from(top_hi) < y1;
            }
            if wholly && relief_in {
                failures += 1;
                if failures <= 24 {
                    hits.push_str(&format!("HIT {name} detail {} dist {dist:.1} span {span}\n", s.detail.0));
                }
            }
        }
        let missed = cols.iter().filter(|c| !(y0..y1).contains(&i64::from(c.2))).count();
        let far = far_near(world, center, &cols);
        let (x0, x1, z0, z1) = near;
        let far_inner = cols
            .iter()
            .filter(|&&(x, z, _)| {
                let (bx, bz) = (i64::from(x), i64::from(z));
                bx - x0 >= 32 && x1 - bx > 32 && bz - z0 >= 32 && z1 - bz > 32 && far_drawn(world, x, z)
            })
            .count();
        let top = stand.open - 1;
        let feet_in = (y0..y1).contains(&i64::from(top));
        let feet_far = far_drawn(world, stand.x, stand.z);
        let seat = world.seams.chart_seat(center);
        let (seat_i, seat_in, seat_r) = seat
            .map(|s| {
                let a = &world.generator.atlases()[s.index];
                (s.index, a.inward, a.radius)
            })
            .unwrap_or((usize::MAX, false, 0));
        say(&format!(
            "NEAR {name} passes {passes} bare {bare_n} near {near_n} failures {failures} missed {missed} far_near {far} far_inner {far_inner} \
             feet_in {feet_in} feet_far {feet_far} grounded {} window {:?} up {:?} seat {seat_i} inward {seat_in} radius {seat_r} \
             other_face {other_face} nearest {nearest} by_detail {by_detail:?} cols {} chunks {}",
            world.window.grounded,
            world.window.punch,
            world.live_up(),
            cols.len(),
            world.chunks.len()
        ));
        if !hits.is_empty() {
            print!("{hits}");
        }
        assert!(stand.roundtrip < 1.0, "{name}: storage roundtrip {}", stand.roundtrip);
        assert!(seat_in == inward && (seat_r - radius).abs() < 16, "{name}: stood on atlas {seat_i} inward {seat_in} radius {seat_r}");
        assert_eq!(world.live_up(), Some(Face::PosY), "{name}: chart up");
        assert!(feet_in, "{name}: the column underfoot is outside the window");
        assert!(!feet_far, "{name}: a far section draws the column underfoot");
        assert!(world.window.grounded, "{name}: the window never reached the ground");
        assert_eq!(failures, 0, "{name}: {failures} sections inside the near square with relief in the window");
        assert_eq!(bare_n, 0, "{name}: {bare_n} bare columns");
        passes
    }

    fn audit_round(kind: crate::world::terrain::cosmos::Kind, inward: bool, name: &str) {
        let mut world = seeded(42, 16, 5);
        let body = body_of(&world, kind);
        let atlas = round_atlas(&world, &body, inward);
        let radius = atlas.radius;
        let stand = chart_stand(&world, atlas);
        audit_chart(&mut world, name, &stand, inward, radius);
    }

    fn cell_solid(world: &World, p: DVec3) -> bool {
        let (x, y, z) = (crate::math::block_coord(p.x), crate::math::block_coord(p.y), crate::math::block_coord(p.z));
        world.registry.is_solid(world.generator.voxel_at(x, y, z))
    }

    fn tangents(dir: DVec3) -> (DVec3, DVec3) {
        let up = if dir.y.abs() < 0.9 { DVec3::Y } else { DVec3::X };
        let u = dir.cross(up).normalize();
        (u, dir.cross(u).normalize())
    }

    /// First solid point marching from `from` along unit `into`, or `None` when the ray stays air.
    fn first_solid(world: &World, from: DVec3, into: DVec3, steps: i32, step: f64) -> Option<DVec3> {
        if cell_solid(world, from) {
            return Some(from);
        }
        for i in 1..=steps {
            let p = from + into * (step * f64::from(i));
            if cell_solid(world, p) {
                let mut lo = from + into * (step * f64::from(i - 1));
                let mut hi = p;
                for _ in 0..24 {
                    let mid = (lo + hi) * 0.5;
                    if cell_solid(world, mid) {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                return Some(hi);
            }
        }
        None
    }

    /// Eye just outside the great rock, on the axis whose surface varies most inside the near radius.
    fn rock_stand(world: &World, rock: &crate::world::terrain::cosmos::Rock) -> (DVec3, DVec3, i32) {
        let c = DVec3::new(f64::from(rock.centre[0]), f64::from(rock.centre[1]), f64::from(rock.centre[2]));
        let reach = f64::from(rock.r) * 1.35 + 2.0;
        let dirs = [DVec3::X, DVec3::NEG_X, DVec3::Y, DVec3::NEG_Y, DVec3::Z, DVec3::NEG_Z];
        let mut best: Option<(DVec3, DVec3, i32)> = None;
        for dir in dirs {
            let Some(solid) = first_solid(world, c + dir * reach, -dir, 400, 16.0) else { continue };
            let (u, v) = tangents(dir);
            let (mut lo, mut hi) = (f64::MAX, f64::MIN);
            for iv in -4..=4 {
                for iu in -4..=4 {
                    let off = u * (f64::from(iu) * 64.0) + v * (f64::from(iv) * 64.0);
                    if let Some(hit) = first_solid(world, c + off + dir * reach, -dir, 400, 16.0) {
                        let h = (hit - c).dot(dir);
                        lo = lo.min(h);
                        hi = hi.max(h);
                    }
                }
            }
            let relief = if lo > hi { 0 } else { (hi - lo).round() as i32 };
            let mut eye = solid + dir * 1.62;
            for _ in 0..8 {
                if !cell_solid(world, eye) {
                    break;
                }
                eye += dir;
            }
            if best.as_ref().is_none_or(|b| relief > b.2) {
                best = Some((eye, dir, relief));
            }
        }
        best.expect("a rock surface")
    }

    /// Solid-top chunks of the rock within 256 blocks of the eye, in the tangent plane.
    fn rock_chunks(world: &World, eye: DVec3, dir: DVec3) -> Vec<Coord> {
        let (u, v) = tangents(dir);
        let mut out = Vec::new();
        for iv in -16..=16 {
            for iu in -16..=16 {
                let off = u * (f64::from(iu) * 16.0) + v * (f64::from(iv) * 16.0);
                if off.length_squared() > 256.0 * 256.0 {
                    continue;
                }
                let Some(hit) = first_solid(world, eye + off + dir * 64.0, -dir, 48, 8.0) else { continue };
                let c = World::chunk_of(crate::math::block_coord(hit.x), crate::math::block_coord(hit.y), crate::math::block_coord(hit.z));
                if !out.contains(&c) {
                    out.push(c);
                }
            }
        }
        out
    }

    #[test]
    #[ignore]
    fn near_lod_inventory() {
        let world = seeded(42, 16, 5);
        let cosmos = world.generator.cosmos().expect("cosmos");
        let atlases = world.generator.atlases();
        for (i, a) in atlases.iter().enumerate() {
            let grid = a.grid.map(|g| g.body);
            say(&format!(
                "ATLAS {i} radius {} inward {} grid {grid:?} warp {} bands {} centre {:.0} {:.0} {:.0}",
                a.radius,
                a.inward,
                a.warp.is_some(),
                a.bands.len(),
                a.centre.x,
                a.centre.y,
                a.centre.z
            ));
        }
        for b in cosmos.bodies() {
            let charts: Vec<usize> = atlases
                .iter()
                .enumerate()
                .filter(|(_, a)| {
                    if a.grid.is_some_and(|g| g.body == b.id) {
                        return true;
                    }
                    if a.grid.is_some() || (a.centre - b.centre_f()).length() >= 1.0 {
                        return false;
                    }
                    match b.shape {
                        crate::world::terrain::cosmos::Shape::Ball { r } => (a.radius - r).abs() < 16,
                        crate::world::terrain::cosmos::Shape::Shell { outer, inner } => (a.radius - outer).abs() < 16 || (a.radius - inner).abs() < 16,
                        crate::world::terrain::cosmos::Shape::Cube { .. } => false,
                    }
                })
                .map(|(i, _)| i)
                .collect();
            let warped = atlases.iter().any(|a| a.grid.is_some_and(|g| g.body == b.id) && a.warp.is_some());
            say(&format!("BODY {} {} {:?} centre {:?} charts {charts:?} warped {warped}", b.id, b.kind.name(), b.shape, b.centre));
        }
        match cosmos.great_rock() {
            Some(r) => say(&format!("ROCK great centre {:?} r {} axes {:?} {:?}", r.centre, r.r, r.axes, r.kind)),
            None => say("ROCK none"),
        }
    }

    #[test]
    #[ignore]
    fn near_lod_verdant() {
        audit_round(crate::world::terrain::cosmos::Kind::Verdant, false, "verdant");
    }

    #[test]
    #[ignore]
    fn near_lod_hollow_outer() {
        audit_round(crate::world::terrain::cosmos::Kind::Hollow, false, "hollow_outer");
    }

    #[test]
    #[ignore]
    fn near_lod_hollow_inner() {
        audit_round(crate::world::terrain::cosmos::Kind::Hollow, true, "hollow_inner");
    }

    #[test]
    #[ignore]
    fn near_lod_ember() {
        audit_round(crate::world::terrain::cosmos::Kind::Ember, false, "ember");
    }

    #[test]
    #[ignore]
    fn near_lod_moon() {
        audit_round(crate::world::terrain::cosmos::Kind::Moon, false, "moon");
    }

    #[test]
    #[ignore]
    fn near_lod_rock() {
        let mut world = seeded(42, 16, 5);
        let rock = world.generator.cosmos().expect("cosmos").great_rock().expect("a great rock");
        say(&format!("ROCK stand-on centre {:?} r {} axes {:?} {:?}", rock.centre, rock.r, rock.axes, rock.kind));
        let (eye, dir, relief) = rock_stand(&world, &rock);
        let chunks = rock_chunks(&world, eye, dir);
        say(&format!(
            "STAND rock phys {:.1} {:.1} {:.1} dir {:.3} {:.3} {:.3} relief {} samples {}",
            eye.x, eye.y, eye.z, dir.x, dir.y, dir.z, relief, chunks.len()
        ));
        let cosmos = world.generator.cosmos().expect("cosmos");
        for b in cosmos.bodies() {
            let alt = cosmos.altitude(b, eye);
            if alt.abs() < b.reach() * 1.5 {
                say(&format!("ROCK near-body {} {} alt {:.1}", b.id, b.kind.name(), alt));
            }
        }
        world.prepare_around(eye);
        world.drive_spawn_ready();
        let passes = settle(&mut world, eye, NONE, "rock");
        let bare_n = chunks.iter().filter(|c| !world.chunks.get(c).is_some_and(|l| l.state.settled())).count();
        let (near_n, failures) = if let Some((_, face)) = world.section_lod_face {
            proof_hits(&world, face)
        } else {
            let n = world.section_visible.len();
            (n, n)
        };
        say(&format!(
            "NEAR rock passes {passes} bare {bare_n} near {near_n} failures {failures} missed 0 far_near 0 far_inner 0 \
             feet_in 1 feet_far 0 grounded {} window {:?} up {:?} face {:?} chunks {} visible {}",
            world.window.grounded,
            world.window.punch,
            world.live_up(),
            world.section_lod_face,
            world.chunks.len(),
            world.section_visible.len()
        ));
        assert!(!cell_solid(&world, eye), "rock: the eye is inside the rock");
        assert_eq!(failures, 0, "rock: coarse sections next to the eye");
        assert_eq!(bare_n, 0, "rock: {bare_n} surface chunks inside the near radius are not settled");
    }
}
