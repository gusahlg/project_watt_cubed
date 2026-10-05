//! The near window along the up axis. It always holds the eye band (`eye ± vertical`), so nothing
//! the player stands on or builds beside unloads, and it grows over the terrain of the near square:
//! down to one chunk under its lowest ground and up to one chunk over its highest, within
//! [`WINDOW_CAP`] layers. Far sections the window holds give way only once the chunks under them
//! have settled (on a chart the punch waits per section; elsewhere the engine's clip box keeps to
//! the span its settled rings prove), and a shrink that gives up ground waits for the far field.

use super::*;

/// Layers the near window may span along the up axis, eye band included.
const WINDOW_CAP: i32 = 32;

/// Layers the held window may reach while a shrink waits for the far field.
const HELD_CAP: i32 = 2 * WINDOW_CAP;

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
    /// The ground bounds are to be read again on the next pass: the far field's relief bake landed.
    pub(in crate::world) stale: bool,
}

/// The span the window wants: the eye band `eye ± v`, grown over the terrain `ground` within
/// `cap` layers. Past the cap each side keeps the terrain nearest the band, at least half the
/// spare layers when it needs them.
fn wanted(eye: i32, v: i32, ground: Option<[i32; 2]>, cap: i32) -> [i32; 2] {
    let band = [eye - v, eye + v];
    let Some([lo, hi]) = ground else { return band };
    let spare = spare(v, cap);
    let (down, up) = ((band[0] - lo).max(0), (hi - band[1]).max(0));
    let d = down.min(spare - up.min(spare / 2));
    let u = up.min(spare - d);
    [band[0] - d, band[1] + u]
}

/// Layers the window may add to an eye band of radius `v` within `cap`.
fn spare(v: i32, cap: i32) -> i32 {
    (cap - 2 * v - 1).max(0)
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

/// Every chunk under `s` in `layers` is settled (drawn, or nothing to draw).
fn backing_settled(chunks: &FastMap<Coord, Loaded>, s: SectionPos, layers: [i32; 2]) -> bool {
    let cs = CHUNK_SIZE as i32;
    let (x0, z0, n) = (s.min_x().div_euclid(cs), s.min_z().div_euclid(cs), s.span() / cs);
    (layers[0]..=layers[1]).rev().all(|y| {
        (z0..z0 + n).all(|z| (x0..x0 + n).all(|x| chunks.get(&Coord::new(x, y, z)).is_some_and(|l| l.state.settled())))
    })
}

impl World {
    /// The near window grows past the eye band wherever the far field gives way to it: on a round
    /// world's chart, punched by key, and in physical space, where the engine's clip box follows
    /// it ([`lod_clip_box`](Self::lod_clip_box)). A warped cube's storage clips nothing, and open
    /// space has no up axis.
    fn window_grows(&self, center: Coord) -> bool {
        match self.live_up() {
            Some(_) if self.fold.is_identity() => true,
            Some(Face::PosY) => self.section_on_chart(center),
            _ => false,
        }
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
    /// Without a window they go back to the eye band, proven afresh.
    fn clip_follow(&mut self, center: Coord) {
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

    /// Place the window around streaming centre `center`; returns whether the held span changed.
    /// The punch span is recomputed only when the centre `moved` (or the ground went stale), and
    /// the settled rings follow it. The held span takes a grown punch span at once and gives up
    /// layers once they hold no ground, or once the far field draws the ground they held.
    pub(in crate::world) fn place_window(&mut self, center: Coord, moved: bool) -> bool {
        let prev = self.window.held;
        if moved || std::mem::take(&mut self.window.stale) {
            let before = self.window.punch;
            self.window.punch = self.punch_window(center);
            if self.window.punch != before {
                self.clip_follow(center);
            }
            self.window.held = match (prev, self.window.punch) {
                (Some(h), Some(p)) => {
                    let hull = [h[0].min(p[0]), h[1].max(p[1])];
                    Some(if hull[1] - hull[0] < HELD_CAP { hull } else { p })
                }
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
        self.window.held != prev
    }

    /// The span the punch tests against for `center`, through the hysteresis. The window reaches
    /// for the ground while the eye is within the window's width over the highest solid top, and
    /// within what the cap lets it add, and while the eye band is not buried under the lowest one
    /// (one layer of hysteresis each way); otherwise it is the eye band and the far field draws the
    /// ground.
    fn punch_window(&mut self, center: Coord) -> Option<[i32; 2]> {
        if !self.window_grows(center) {
            self.window = Window::default();
            return None;
        }
        let ground = self.ground_span(center);
        let (_, eye) = ColumnKey::of(self.live_up()?, center);
        let v = self.view.vertical;
        let reach = (2 * self.view.horizontal + 1).min(v + spare(v, WINDOW_CAP));
        let slack = if self.window.grounded { 1 } else { -1 };
        let grounded = ground.is_some_and(|[lo, hi]| {
            let (over, under) = (eye - (hi - 1), lo + 1 - (eye + v));
            over <= reach + slack && under <= slack
        });
        self.window.ground = ground;
        self.window.grounded = grounded;
        let want = wanted(eye, v, ground.filter(|_| grounded), WINDOW_CAP);
        Some(self.window.punch.map_or(want, |prev| hold(prev, want)))
    }

    /// Chunk layers of the near square's terrain: one under the layer of its lowest solid top to
    /// one over the layer of its highest.
    fn ground_span(&mut self, center: Coord) -> Option<[i32; 2]> {
        if self.fold.is_identity() { self.relief_span(center) } else { self.chart_span(center) }
    }

    /// [`ground_span`](Self::ground_span) off a chart, from the far field's baked relief over the
    /// near square, per finest section of its face: no generator reads at all. `None` until that
    /// bake has landed, or where it does not reach.
    fn relief_span(&self, center: Coord) -> Option<[i32; 2]> {
        let face = self.live_up()?;
        let (body, _) = self.section_lod_face.filter(|&(_, f)| f == face)?;
        let frame = FaceFrame::new(face);
        let (cu, _, cv) = frame.chunk_to_local(center);
        let (h, cs) = (self.view.horizontal, CHUNK_SIZE as i32);
        let span = super::super::section::section_span(super::super::section::FINEST_DETAIL);
        let tiles = |c: i32| ((c - h) * cs).div_euclid(span)..=((c + h + 1) * cs - 1).div_euclid(span);
        let at = |x: i32, z: i32| SectionPos { detail: super::super::section::FINEST_DETAIL, body, face, x, z };
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

    /// [`ground_span`](Self::ground_span) on a chart, per chunk column of the home chart through
    /// the generator's surface bounds (the far field's per-rect memo), kept while the column stays
    /// in the square. Every punch rect is a union of whole columns, so its bounds lie inside these.
    fn chart_span(&mut self, center: Coord) -> Option<[i32; 2]> {
        let seat = self.seams.chart_seat(center)?;
        let body = super::super::section::CHART_BODY_BASE + seat.index as u16;
        let cs = CHUNK_SIZE as i64;
        let (x0, x1, z0, z1) = self.near_block_box(center);
        let (cx0, cx1) = (x0.max(seat.lo[0]).div_euclid(cs), x1.min(seat.hi[0]).div_euclid(cs));
        let (cz0, cz1) = (z0.max(seat.lo[2]).div_euclid(cs), z1.min(seat.hi[2]).div_euclid(cs));
        let n = CHUNK_SIZE as i32;
        let mut memo = std::mem::take(&mut self.window_ground);
        let bounds = memo.sweep(|memo| {
            let mut out: Option<(i32, i32)> = None;
            for cz in cz0..cz1 {
                for cx in cx0..cx1 {
                    let (Ok(u), Ok(v)) = (i32::try_from(cx * cs), i32::try_from(cz * cs)) else { continue };
                    let surface = || self.generator.surface_rect(body, Face::PosY, u, v, u + n, v + n);
                    if let Some((lo, hi)) = memo.get((body, [u, v, u + n, v + n]), surface) {
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

    /// Every far section the frontier wants over the near square has something drawn, or is not
    /// loaded because the window draws it. On a chart the frontier must be selected for the current
    /// punch span.
    fn far_covers_near(&self, center: Coord) -> bool {
        if !self.lod2 {
            return true;
        }
        if !self.fold.is_identity() && self.section_frontier_key.is_none_or(|k| k.window != self.window.punch) {
            return false;
        }
        self.section_desired
            .iter()
            .all(|&s| !self.near_meets(center, s) || self.coverage_skips(center, s) || self.section_covered(s))
    }

    /// Whether far section `s` reaches over the near square around `center`. A section of another
    /// chart or face counts wherever it lies.
    fn near_meets(&self, center: Coord, s: SectionPos) -> bool {
        let near = if self.fold.is_identity() {
            let Some(face) = self.live_up() else { return true };
            if s.face != face || self.section_lod_face != Some((s.body, face)) {
                return true;
            }
            let (cu, _, cv) = FaceFrame::new(face).chunk_to_local(center);
            let (h, cs) = (i64::from(self.view.horizontal), CHUNK_SIZE as i64);
            let (u, v) = (i64::from(cu), i64::from(cv));
            ((u - h) * cs, (u + h + 1) * cs, (v - h) * cs, (v + h + 1) * cs)
        } else {
            match self.seams.chart_seat(center) {
                Some(seat) if inside_xz(s, seat.lo, seat.hi) => self.near_block_box(center),
                _ => return true,
            }
        };
        covers_near(s, near, None)
    }

    /// Face-local altitudes `[a0, a1)` of the punch window.
    pub(in crate::world) fn window_alts(&self) -> Option<[i64; 2]> {
        let face = self.live_up()?;
        let [lo, hi] = self.window_raw()?;
        let cs = CHUNK_SIZE as i64;
        let (r0, r1) = (i64::from(lo) * cs, (i64::from(hi) + 1) * cs);
        Some(if face.sign() > 0 { [r0, r1] } else { [1 - r1, 1 - r0] })
    }

    /// A chunk settled: the settled rings may grow and a held section may be done waiting.
    pub(in crate::world) fn note_settled(&mut self) {
        self.lod_clip_grow.set();
        self.held_recheck.set();
    }

    /// Keep drawing each held section whose ground has not all settled: it joins the desired
    /// frontier, admission skips it, and whatever draws it (itself or an ancestor) stays drawn.
    pub(super) fn wait_held(&mut self, held: Vec<(SectionPos, [i32; 2])>) {
        self.section_held.clear();
        for (s, layers) in held {
            let selected = self.section_desired.binary_search_by_key(&section_key(&s), section_key).is_ok();
            if !selected && !backing_settled(&self.chunks, s, layers) {
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
        let chunks = &self.chunks;
        let mut settled = Vec::new();
        self.section_held.retain(|&s, &mut layers| {
            let done = backing_settled(chunks, s, layers);
            if done {
                settled.push(s);
            }
            !done
        });
        if settled.is_empty() {
            return;
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

    use super::super::flight_bench::step;
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
        let mut world = World::with_kind(seed, render, WorldgenKind::Diffusion, false);
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

    /// The near square's chunk-centre columns around `center`: `(x, z)` and the solid top there.
    fn near_columns(world: &World, center: Coord) -> Vec<(i32, i32, i32)> {
        let h = world.view.horizontal;
        let mut out = Vec::new();
        for cz in center.z - h..=center.z + h {
            for cx in center.x - h..=center.x + h {
                let (x, z) = (cx * 16 + 8, cz * 16 + 8);
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

    /// Stream at `eye` until the world is complete and the held window is the punch window,
    /// asserting `bare` finds nothing on any pass. Returns the passes taken.
    fn settle(world: &mut World, eye: DVec3, bare: &dyn Fn(&World) -> usize, name: &str) -> usize {
        for pass in 0.. {
            if world.entry_complete() && world.window.held == world.window.punch {
                return pass;
            }
            assert!(pass < 200_000, "{name}: did not settle: {}", world.entry_debug());
            step(world, eye);
            assert_eq!(bare(world), 0, "{name}: bare ground on settling pass {pass}");
            std::thread::sleep(Duration::from_millis(1));
        }
        unreachable!()
    }

    const NONE: &dyn Fn(&World) -> usize = &|_| 0;

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
    /// stopping: the world settles, nothing is left waiting on the ground, every column is drawn,
    /// and no far section draws over ground the window holds.
    #[test]
    fn flight_then_stop_settles() {
        let mut world = chart_world(6, 3);
        let start = spawn_storage(&world);
        let ground = world.generator.surface(Face::PosY, start.x.floor() as i32, start.z.floor() as i32);
        let at = |k: i32| DVec3::new(start.x + 8.0 * f64::from(k), f64::from(ground) + 100.0, start.z);
        world.prepare_around(physical(&world, at(0)));
        world.drive_spawn_ready();
        for k in 0..150 {
            let eye = physical(&world, at(k));
            step(&mut world, eye);
        }
        let stop = physical(&world, at(150));
        let passes = settle(&mut world, stop, NONE, "stop");
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
    #[test]
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
}
