//! The near field. A chunk column is near when the distance from the eye's cell to its terrain box
//! (its footprint between its lowest and highest ground, in physical space) is under the render
//! distance `R` ([`super::super::rule`]). A near column's terrain shell, one chunk under its lowest
//! solid top to one over its highest, loads and draws at full resolution, and the far tile over it
//! leaves it out ([`ColumnClip`]). Around the eye, the eye band (the eye's layer ± the vertical
//! distance) holds voxel data across the whole square for collision and building, and draws where
//! it lies within `R` and away from the ground of a column that is not near. Nothing else loads:
//! no window grows, caps or waits, and a column draws either full-res or far, never both once its
//! chunks have settled.

use super::*;
use super::super::rule::PhysBox;
use super::super::section::{CHART_BODY_BASE, ColumnClip};
use super::super::sampler::{Recent, Sampler};
use super::super::generation::TerrainGenerator;
use super::super::{DATA_MARGIN, UNLOAD_MARGIN, UNLOAD_MARGIN_V};

/// Where the near field reads its columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::world) enum Frame {
    /// A round world's chart storage, folded into one net around the centre; storage +Y is up.
    Chart { index: usize, body: u16, floor: i32 },
    /// A warped cube's storage: `shift` chunks take a storage chunk to its reference chunk.
    Cube { body: u16, face: Face, shift: [i32; 3] },
    /// A flat world's plane.
    Flat,
}

/// One column of the near square.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Col {
    /// The real chunk column it reads (unfolded).
    real: Option<ColumnKey>,
    /// Chunk layers from one under the lowest solid top to one over the highest, edits included.
    shell: Option<[i32; 2]>,
    /// Its terrain box in physical space.
    bx: Option<PhysBox>,
    /// The terrain box lies within `R` of the eye cell: the shell draws at full resolution.
    near: bool,
    /// Within `R` plus the unload margin: the shell stays loaded.
    keep: bool,
    /// Layers new work may load beside the eye band: the near shells of the column and its eight
    /// neighbours, one layer wider each way (meshing and light read the face neighbours).
    load: Option<[i32; 2]>,
    /// The same over the kept shells: what unloading leaves in place.
    hold: Option<[i32; 2]>,
}

/// The near field around the streaming centre.
#[derive(Default)]
pub(in crate::world) struct NearField {
    frame: Option<Frame>,
    up: Option<Face>,
    /// Folded centre column `(a, b)` and the square's half side in columns.
    centre: (i32, i32),
    half: i32,
    /// The eye band (up-local chunk layers).
    band: [i32; 2],
    /// Render distance in chunks, squared, for the eye band's reach.
    reach2: f64,
    cols: Vec<Col>,
    /// The span along up every box holds: the eye band and every kept shell.
    hull: Option<[i32; 2]>,
    /// The eye cell (block corner) the flags were measured from.
    eye_cell: Option<[i64; 3]>,
    /// Bumped whenever a span or flag changes.
    pub(in crate::world) rev: u32,
    /// First-open-cell bounds `(lo, hi)` in frame altitude, by real column.
    ground: Recent<ColumnKey, Option<(i32, i32)>>,
    /// Physical terrain boxes by real column, with the ground (edits included) they were placed for.
    boxes: Recent<ColumnKey, ((i32, i32), Option<PhysBox>)>,
    /// The ground is to be read again (an edit landed).
    pub(in crate::world) stale: bool,
    /// The farthest a point of a near column's box lies from the eye's cell: where the finest far
    /// level meets the full-res chunks.
    pub(in crate::world) reach: f64,
    /// The thread reading every column's ground: the square's own as it is first met, and the
    /// columns just outside it before the centre moves onto them.
    sampler: Option<Sampler<(Frame, ColumnKey), Option<(i32, i32)>>>,
    /// Columns of the square whose ground was not in at the last layout: none of them is near or
    /// kept until it is (the eye band still loads across them).
    unknown: usize,
}

/// Most column grounds one pass keeps.
const LAND_MAX: usize = 1 << 12;

/// Rings of columns outside the square whose ground is read ahead.
const LEAD_RINGS: i32 = 2;

/// First-open-cell bounds of real column `real` in its frame's altitude: the generator's bounds
/// over a chart's storage rect, or the ground sampled at nine points of a cube face's or a flat
/// world's column, eight blocks wider each way. `None` where it holds no ground. Pure, so the near
/// field's sampler reads it ahead.
fn column_ground(generator: &dyn TerrainGenerator, frame: Frame, real: ColumnKey) -> Option<(i32, i32)> {
    let cs = CHUNK_SIZE as i32;
    match frame {
        Frame::Chart { body, .. } => {
            let (x0, z0) = (real.a * cs, real.b * cs);
            let (ox, oz) = generator.chart_origin(body, Face::PosY).unwrap_or((0, 0));
            generator.surface_rect(body, Face::PosY, x0 - ox, z0 - oz, x0 + cs - ox, z0 + cs - oz)
        }
        Frame::Cube { face, shift, .. } => {
            let (u0, u1, v0, v1) = cube_cells(real, shift)?;
            let mut out: Option<(i32, i32)> = None;
            for u in [u0, (u0 + u1) / 2, u1] {
                for v in [v0, (v0 + v1) / 2, v1] {
                    let s = generator.surface(face, u, v);
                    if s != i32::MIN {
                        out = Some(out.map_or((s, s), |(a, b)| (a.min(s), b.max(s))));
                    }
                }
            }
            out.map(|(lo, hi)| (lo - 8, hi + 8))
        }
        Frame::Flat => {
            let (x0, z0) = (real.a * cs, real.b * cs);
            let mut out: Option<(i32, i32)> = None;
            for x in [x0, x0 + cs / 2, x0 + cs - 1] {
                for z in [z0, z0 + cs / 2, z0 + cs - 1] {
                    let s = generator.height(x, z);
                    out = Some(out.map_or((s, s), |(a, b)| (a.min(s), b.max(s))));
                }
            }
            out
        }
    }
}

/// The reference-face cell range `(u0, u1, v0, v1)` of storage column `real` shifted by `shift`
/// chunks.
fn cube_cells(real: ColumnKey, shift: [i32; 3]) -> Option<(i32, i32, i32, i32)> {
    let frame = FaceFrame::new(real.face);
    let c = real.chunk(0);
    let r = Coord::new(c.x + shift[0], c.y + shift[1], c.z + shift[2]);
    let cs = CHUNK_SIZE as i32;
    let (mut u0, mut u1, mut v0, mut v1) = (i32::MAX, i32::MIN, i32::MAX, i32::MIN);
    for dx in [0, cs - 1] {
        for dz in [0, cs - 1] {
            for dy in [0, cs - 1] {
                let (u, _, v) = frame.cell_to_local((r.x.checked_mul(cs)? + dx, r.y.checked_mul(cs)? + dy, r.z.checked_mul(cs)? + dz));
                (u0, u1, v0, v1) = (u0.min(u), u1.max(u), v0.min(v), v1.max(v));
            }
        }
    }
    Some((u0, u1, v0, v1))
}

/// `a ∪ b`.
fn hull(a: Option<[i32; 2]>, b: [i32; 2]) -> Option<[i32; 2]> {
    Some(a.map_or(b, |a| [a[0].min(b[0]), a[1].max(b[1])]))
}

fn holds(span: Option<[i32; 2]>, alt: i32) -> bool {
    span.is_some_and(|[lo, hi]| lo <= alt && alt <= hi)
}

impl NearField {
    /// Whether the near field has columns: off them every box keeps its eye band and loads whole.
    pub(in crate::world) fn active(&self) -> bool {
        self.frame.is_some()
    }

    /// The column index and altitude of folded chunk `c`, inside the square.
    fn locate(&self, c: Coord) -> Option<(usize, i32)> {
        let up = self.up?;
        let (key, alt) = ColumnKey::of(up, c);
        let (da, db) = (key.a - self.centre.0, key.b - self.centre.1);
        let n = 2 * self.half + 1;
        ((-self.half..=self.half).contains(&da) && (-self.half..=self.half).contains(&db))
            .then(|| (((db + self.half) * n + da + self.half) as usize, alt))
    }

    /// The eye band's layers of column `i` within `R` of the eye's chunk.
    fn band_drawn(&self, i: usize) -> Option<[i32; 2]> {
        let n = 2 * self.half + 1;
        let (da, db) = ((i as i32 % n) - self.half, (i as i32 / n) - self.half);
        let gap = |d: i32| f64::from((d.abs() - 1).max(0));
        let left = self.reach2 - gap(da).powi(2) - gap(db).powi(2);
        if left <= 0.0 {
            return None;
        }
        let k = 1 + (left.sqrt() - 1e-9).floor() as i32;
        let eye = (self.band[0] + self.band[1]) / 2;
        let (lo, hi) = (self.band[0].max(eye - k), self.band[1].min(eye + k));
        (lo <= hi).then_some([lo, hi])
    }

    /// Folded chunk `c` draws at full resolution: in a near column's shell, or in the eye band
    /// within `R` and clear of the ground of a column that is not near.
    pub(in crate::world) fn draws(&self, c: Coord) -> bool {
        if !self.active() {
            return true;
        }
        let Some((i, alt)) = self.locate(c) else { return false };
        let col = &self.cols[i];
        let in_shell = holds(col.shell, alt);
        (col.near && in_shell) || ((col.near || !in_shell) && holds(self.band_drawn(i), alt))
    }

    /// Folded chunk `c` may load voxel data: the eye band and a data margin, or beside a near shell.
    pub(in crate::world) fn loads(&self, c: Coord) -> bool {
        if !self.active() {
            return true;
        }
        let Some(up) = self.up else { return true };
        let (_, alt) = ColumnKey::of(up, c);
        if (self.band[0] - DATA_MARGIN..=self.band[1] + DATA_MARGIN).contains(&alt) {
            return true;
        }
        self.locate(c).is_some_and(|(i, alt)| holds(self.cols[i].load, alt))
    }

    /// Folded chunk `c` stays loaded: [`loads`](Self::loads) over the kept shells, with the
    /// vertical unload margin.
    pub(in crate::world) fn keeps(&self, c: Coord) -> bool {
        if !self.active() {
            return true;
        }
        let Some(up) = self.up else { return true };
        let (_, alt) = ColumnKey::of(up, c);
        let m = DATA_MARGIN + UNLOAD_MARGIN_V;
        if (self.band[0] - m..=self.band[1] + m).contains(&alt) {
            return true;
        }
        self.locate(c).is_some_and(|(i, alt)| self.cols[i].hold.is_some_and(|[lo, hi]| lo - UNLOAD_MARGIN_V <= alt && alt <= hi + UNLOAD_MARGIN_V))
    }

    /// Layers every box adds below and above the eye band.
    pub(in crate::world) fn grow(&self) -> [i32; 2] {
        match self.hull {
            Some([lo, hi]) => [(self.band[0] - lo).max(0), (hi - self.band[1]).max(0)],
            None => [0, 0],
        }
    }

    /// The near column holding folded chunk column `(a, b)` along up, with its real column and
    /// shell, when it is near.
    pub(in crate::world) fn near_column(&self, folded: Coord) -> Option<(ColumnKey, [i32; 2])> {
        let (i, _) = self.locate(folded)?;
        let col = &self.cols[i];
        (col.near).then_some(())?;
        Some((col.real?, col.shell?))
    }

    /// Every near column: its real chunk column and shell.
    #[cfg(test)]
    pub(in crate::world) fn near_columns(&self) -> impl Iterator<Item = (ColumnKey, [i32; 2])> + '_ {
        self.cols.iter().filter(|c| c.near).filter_map(|c| Some((c.real?, c.shell?)))
    }

    /// The up face the columns stand along.
    pub(in crate::world) fn up(&self) -> Option<Face> {
        self.up
    }

    pub(in crate::world) fn frame(&self) -> Option<Frame> {
        self.frame
    }
}

impl World {
    /// The frame the near field reads at streaming centre `center`: a chart under the near window,
    /// a warped cube's storage whose sky is the up face, or a flat world. `None` in open space and
    /// in physical space over a round world, where no ground is near.
    pub(in crate::world) fn near_frame(&self, center: Coord) -> Option<Frame> {
        let up = self.live_up()?;
        let (reference, _, in_cube) = self.lod_place(center);
        if !in_cube && !self.fold.is_identity() {
            let seat = self.seams.chart_seat(center)?;
            let floor = seat.lo[1].div_euclid(CHUNK_SIZE as i64) as i32;
            return (up == Face::PosY).then_some(Frame::Chart { index: seat.index, body: CHART_BODY_BASE + seat.index as u16, floor });
        }
        if in_cube {
            if self.generator.sky(center) != Sky::Axis(up) {
                return None;
            }
            let cosmos = self.generator.cosmos()?;
            let cs = CHUNK_SIZE as f64;
            let p = DVec3::new((f64::from(reference.x) + 0.5) * cs, (f64::from(reference.y) + 0.5) * cs, (f64::from(reference.z) + 0.5) * cs);
            let body = cosmos.body_at(p)?;
            let shift = [reference.x - center.x, reference.y - center.y, reference.z - center.z];
            return Some(Frame::Cube { body: body.id, face: up, shift });
        }
        (self.generator.cosmos().is_none() && up == Face::PosY).then_some(Frame::Flat)
    }

    /// Lay the near field out around streaming centre `center` for physical eye `eye`; returns
    /// whether any span or flag changed. Columns are read again when the centre `moved`, the eye
    /// crossed into another cell, or the ground went stale.
    pub(in crate::world) fn place_near(&mut self, center: Coord, eye: DVec3, moved: bool) -> bool {
        let frame = self.near_frame(center);
        let up = self.live_up();
        let cell = PhysBox::eye_cell(eye);
        let cell_key = [cell.lo.x as i64, cell.lo.y as i64, cell.lo.z as i64];
        if let Some(frame) = frame {
            let landed = self.land_near_ground(frame);
            if self.near.unknown > 0 && (landed > 0 || self.near.sampler.as_ref().is_none_or(|s| s.waiting() == 0)) {
                // Ground the square waited on is in, or nothing it asked for is still out (an
                // answer for another frame was dropped): lay it out again, asking for what is
                // missing.
                self.near.stale = true;
            }
        }
        let near = &self.near;
        let same = near.frame == frame && near.up == up && near.eye_cell == Some(cell_key) && !near.stale && !moved;
        if same && frame.is_some() {
            return false;
        }
        let mut near = std::mem::take(&mut self.near);
        // Only the eye's cell moved: every column's shell and box stand.
        let held = !moved && !near.stale && near.frame == frame && near.up == up;
        near.stale = false;
        let prev = (near.frame, near.hull, std::mem::take(&mut near.cols), near.band);
        near.frame = frame;
        near.up = up;
        near.eye_cell = Some(cell_key);
        let changed = match (frame, up) {
            (Some(frame), Some(up)) => {
                self.lay_out(&mut near, frame, up, center, &cell, held.then_some(&prev.2[..]));
                prev.0 != near.frame || prev.1 != near.hull || prev.2 != near.cols || prev.3 != near.band
            }
            _ => {
                near.hull = None;
                prev.0.is_some()
            }
        };
        if changed {
            near.rev = near.rev.wrapping_add(1);
        }
        self.near = near;
        changed
    }

    /// [`place_near`](Self::place_near) on a frame: every column's shell, box and flags, then the
    /// data spans beside them. `held` is the last layout's columns when only the eye's cell moved
    /// since: their shells and boxes are kept if the square is the same.
    fn lay_out(&self, near: &mut NearField, frame: Frame, up: Face, center: Coord, eye: &PhysBox, held: Option<&[Col]>) {
        let (h, v) = (self.view.horizontal, self.view.vertical);
        let (key, alt) = ColumnKey::of(up, center);
        let half = h + UNLOAD_MARGIN + DATA_MARGIN;
        let band = [alt - v, alt + v];
        let held = held.filter(|p| near.centre == (key.a, key.b) && near.half == half && near.band == band && p.len() == ((2 * half + 1) * (2 * half + 1)) as usize);
        near.centre = (key.a, key.b);
        near.half = half;
        near.band = band;
        let r = f64::from(h * CHUNK_SIZE as i32);
        near.reach2 = f64::from(h).powi(2);
        if near.sampler.is_none() {
            let generator = self.generator.clone();
            near.sampler = Sampler::start("near-ground", move |(frame, real)| column_ground(&*generator, frame, real));
        }
        let mut want: Vec<(i32, ColumnKey)> = Vec::new();
        let n = 2 * near.half + 1;
        let keep_r = r + f64::from(UNLOAD_MARGIN * CHUNK_SIZE as i32);
        let mut cols = vec![Col::default(); (n * n) as usize];
        near.reach = 0.0;
        for db in -near.half..=near.half {
            for da in -near.half..=near.half {
                let i = ((db + near.half) * n + da + near.half) as usize;
                let col = &mut cols[i];
                if let Some(p) = held {
                    *col = Col { real: p[i].real, shell: p[i].shell, bx: p[i].bx, ..Col::default() };
                    if col.shell.is_none() {
                        continue;
                    }
                } else {
                    let folded = match frame {
                        Frame::Chart { floor, .. } => Coord::new(key.a + da, floor, key.b + db),
                        _ => ColumnKey { face: up, a: key.a + da, b: key.b + db }.chunk(alt),
                    };
                    let Some(real_chunk) = self.fold.unfold(folded) else { continue };
                    let (real, _) = ColumnKey::of(up, real_chunk);
                    col.real = Some(real);
                    // The ground is read off the main thread; until it is in, the column waits.
                    let Some(ground) = near.ground.get(&real) else {
                        want.push((da.abs().max(db.abs()), real));
                        continue;
                    };
                    let Some((lo, hi)) = ground.map(|b| self.column_edits(frame, real, b)) else { continue };
                    let (a, b) = (self.frame_layer(frame, real, lo.saturating_sub(1)), self.frame_layer(frame, real, hi.saturating_sub(1)));
                    col.shell = Some([a.min(b) - 1, a.max(b) + 1]);
                    col.bx = match near.boxes.get(&real) {
                        Some((g, b)) if g == (lo, hi) => b,
                        _ => {
                            let b = self.column_box(frame, real, Some((lo, hi)));
                            near.boxes.insert(real, ((lo, hi), b));
                            b
                        }
                    };
                }
                let b = col.bx;
                // Without a far field the chunks are all the ground there is: every column of the
                // near square draws.
                // The view square holds the near columns: the mesh box is built from it.
                let chess = da.abs().max(db.abs());
                let d = if self.lod2 && chess <= h + UNLOAD_MARGIN {
                    let d = b.map_or(f64::INFINITY, |b| b.dist_box(eye));
                    if chess <= h { d } else { d.max(r) }
                } else if self.lod2 {
                    f64::INFINITY
                } else if chess <= h {
                    0.0
                } else if chess <= h + UNLOAD_MARGIN {
                    r
                } else {
                    f64::INFINITY
                };
                col.near = d < r;
                col.keep = d < keep_r;
                if col.near && let Some(b) = b {
                    near.reach = near.reach.max(d + (b.hi - b.lo).length());
                }
            }
        }
        // The data beside each column: its 3×3 neighbours' shells, one layer wider.
        let mut hull_span = Some(near.band);
        for db in -near.half..=near.half {
            for da in -near.half..=near.half {
                let i = ((db + near.half) * n + da + near.half) as usize;
                let (mut load, mut hold) = (None, None);
                for nb in (db - 1).max(-near.half)..=(db + 1).min(near.half) {
                    for na in (da - 1).max(-near.half)..=(da + 1).min(near.half) {
                        let c = cols[((nb + near.half) * n + na + near.half) as usize];
                        let Some([lo, hi]) = c.shell else { continue };
                        let wide = [lo - DATA_MARGIN, hi + DATA_MARGIN];
                        if c.near {
                            load = hull(load, wide);
                        }
                        if c.keep {
                            hold = hull(hold, wide);
                        }
                    }
                }
                cols[i].load = load;
                cols[i].hold = hold;
                if let Some([lo, hi]) = cols[i].shell.filter(|_| cols[i].keep) {
                    hull_span = hull(hull_span, [lo, hi]);
                }
            }
        }
        near.cols = cols;
        near.hull = hull_span;
        if held.is_none() {
            near.unknown = want.len();
            want.sort_unstable_by_key(|&(chess, _)| chess);
            if let Some(sampler) = &mut near.sampler {
                let ask: Vec<_> = want.into_iter().filter_map(|(_, real)| sampler.mark((frame, real)).then_some((frame, real))).collect();
                sampler.send(ask);
            }
            self.read_ahead(near, frame, up, key, alt);
        }
    }

    /// Keep the column grounds the near field's sampler has answered for `frame`; how many landed.
    fn land_near_ground(&mut self, frame: Frame) -> usize {
        let near = &mut self.near;
        let Some(sampler) = &mut near.sampler else { return 0 };
        let mut landed = 0;
        sampler.land(LAND_MAX, |(f, real), ground| {
            if f == frame && !near.ground.contains(&real) {
                near.ground.insert(real, ground);
                landed += 1;
            }
        });
        landed
    }

    /// [`place_near`](Self::place_near), then again until every column's ground is in: the near
    /// field a settled pass stands on.
    #[cfg(test)]
    pub(in crate::world) fn place_near_settled(&mut self, center: Coord, eye: DVec3, moved: bool) -> bool {
        let mut changed = self.place_near(center, eye, moved);
        let start = std::time::Instant::now();
        while self.near_ground_pending() {
            assert!(start.elapsed().as_secs() < 120, "the near field's ground never came");
            std::thread::sleep(std::time::Duration::from_micros(200));
            changed |= self.place_near(center, eye, false);
        }
        changed
    }

    /// Whether the near field's sampler has answered every column sent to it.
    #[cfg(test)]
    pub(in crate::world) fn near_sampler_answered(&self) -> bool {
        self.near.sampler.as_ref().is_none_or(|s| s.answered())
    }

    /// Column grounds the near field waits on.
    pub(in crate::world) fn near_ground_pending(&self) -> bool {
        self.near.unknown > 0 || self.near.sampler.as_ref().is_some_and(|s| s.waiting() > 0)
    }

    /// Ask the near field's sampler for the ground of the [`LEAD_RINGS`] rings of columns just
    /// outside the square around folded centre column `key` (altitude `alt`), nearest first.
    fn read_ahead(&self, near: &mut NearField, frame: Frame, up: Face, key: ColumnKey, alt: i32) {
        let Some(sampler) = &mut near.sampler else { return };
        let mut want = Vec::new();
        for ring in near.half + 1..=near.half + LEAD_RINGS {
            for db in -ring..=ring {
                let step = if db.abs() == ring { 1 } else { 2 * ring as usize };
                for da in (-ring..=ring).step_by(step) {
                    let folded = match frame {
                        Frame::Chart { floor, .. } => Coord::new(key.a + da, floor, key.b + db),
                        _ => ColumnKey { face: up, a: key.a + da, b: key.b + db }.chunk(alt),
                    };
                    let Some(real_chunk) = self.fold.unfold(folded) else { continue };
                    let (real, _) = ColumnKey::of(up, real_chunk);
                    if !near.ground.contains(&real) && sampler.mark((frame, real)) {
                        want.push((frame, real));
                    }
                }
            }
        }
        sampler.send(want);
    }

    /// [`column_ground`] widened to every chunk an edit touched in a chart's
    /// or a flat world's column (storage +Y).
    fn column_edits(&self, frame: Frame, real: ColumnKey, (lo, hi): (i32, i32)) -> (i32, i32) {
        match frame {
            Frame::Cube { .. } => (lo, hi),
            _ => {
                let cs = CHUNK_SIZE as i32;
                let (x0, z0) = (real.a * cs, real.b * cs);
                self.with_edits((lo, hi), [x0, z0, x0 + cs, z0 + cs])
            }
        }
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

    /// The up-local chunk layer of frame altitude `a` in real column `real`.
    fn frame_layer(&self, frame: Frame, real: ColumnKey, a: i32) -> i32 {
        match frame {
            Frame::Cube { face, shift, .. } => {
                // Reference altitude `a` along the face normal, in the reference column, then back
                // to the storage column it is stored in.
                let f = FaceFrame::new(face);
                let c = real.chunk(0);
                let r = Coord::new(c.x + shift[0], c.y + shift[1], c.z + shift[2]);
                let (ru, _, rv) = f.chunk_to_local(r);
                let (x, y, z) = f.cell_to_world((ru * CHUNK_SIZE as i32, a, rv * CHUNK_SIZE as i32));
                let (_, ref_alt) = ColumnKey::of(face, World::chunk_of(x, y, z));
                let (_, ref_alt0) = ColumnKey::of(face, r);
                let (_, sto_alt0) = ColumnKey::of(face, c);
                ref_alt - ref_alt0 + sto_alt0
            }
            _ => a.div_euclid(CHUNK_SIZE as i32),
        }
    }

    /// Real column `real`'s physical terrain box from its ground `(lo, hi)` (first open cells):
    /// through its chart, its cube's warp, or as it is.
    fn column_box(&self, frame: Frame, real: ColumnKey, ground: Option<(i32, i32)>) -> Option<PhysBox> {
        let (lo, hi) = ground?;
        let (y0, y1) = (f64::from(lo - 1), f64::from(hi));
        let cs = CHUNK_SIZE as i32;
        match frame {
            Frame::Chart { index, .. } => {
                let atlas = self.seams.atlases().get(index)?;
                let (x0, z0) = (real.a * cs, real.b * cs);
                let (patch, _, _) = atlas.column(i64::from(x0 + cs / 2), i64::from(z0 + cs / 2))?;
                // Each corner's column, at the floor and the top.
                let column = |c: i32| atlas.embed_storage_column(patch, f64::from(x0 + cs * (c & 1)), f64::from(z0 + cs * (c >> 1)), [y0, y1]);
                PhysBox::around([0, 1, 2, 3].into_iter().flat_map(column), 1.0, f64::from(cs))
            }
            Frame::Cube { body, face, shift } => {
                let (u0, u1, v0, v1) = cube_cells(real, shift)?;
                let f = FaceFrame::new(face);
                let warp = self.seams.atlases().iter().find(|a| a.grid.as_ref().is_some_and(|g| g.body == body)).and_then(|a| a.warp.clone());
                let corners = (0..8).map(|c| {
                    let u = if c & 1 == 0 { u0 } else { u1 + 1 };
                    let v = if c & 4 == 0 { v0 } else { v1 + 1 };
                    f.point_to_world(DVec3::new(f64::from(u), if c & 2 == 0 { y0 } else { y1 }, f64::from(v)))
                });
                let b = PhysBox::around(corners, 1.0, f64::from(cs))?;
                Some(match warp {
                    Some(w) => {
                        let (lo, hi) = w.bound_box(b.lo, b.hi);
                        PhysBox { lo, hi, span: b.span }
                    }
                    None => b,
                })
            }
            Frame::Flat => {
                let (x0, z0) = (f64::from(real.a * cs), f64::from(real.b * cs));
                Some(PhysBox { lo: DVec3::new(x0, y0, z0), hi: DVec3::new(x0 + f64::from(cs), y1, z0 + f64::from(cs)), span: f64::from(cs) })
            }
        }
    }

    /// The clip each far tile within `R` wants: each of its 16-block columns that only near columns
    /// hold, with the real columns and shells whose chunks must be final first. Only the finest far
    /// level reaches inside `R` (its parent splits there), and it is never wider than a clip.
    pub(in crate::world) fn wanted_clips(&self, leaves: &[super::super::rule::Leaf]) -> Vec<(SectionPos, Vec<super::clip::ClipBit>)> {
        let mut out = Vec::new();
        if !self.near.active() {
            return out;
        }
        let r = f64::from(self.view.horizontal * CHUNK_SIZE as i32);
        for leaf in leaves.iter().filter(|l| l.dist < r) {
            let s = leaf.pos;
            let Some(side) = ColumnClip::side(s) else { continue };
            let mut bits = Vec::new();
            for cz in 0..side {
                for cx in 0..side {
                    let Some(chunks) = self.tile_column_chunks(s, cx, cz) else { continue };
                    let near: Option<Vec<_>> = chunks.iter().map(|&c| self.near.near_column(self.fold.fold(c))).collect();
                    if let Some(cols) = near.filter(|c| !c.is_empty()) {
                        bits.push((ColumnClip::of_column(cx, cz), cols));
                    }
                }
            }
            if !bits.is_empty() {
                out.push((s, bits));
            }
        }
        out
    }

    /// The real storage chunk columns (any layer) tile `s`'s 16-block column `(cx, cz)` covers:
    /// one on a chart or a flat world; on a cube face whose tangent is flipped, the two its cells
    /// straddle along that tangent.
    pub(in crate::world) fn tile_column_chunks(&self, s: SectionPos, cx: i32, cz: i32) -> Option<Vec<Coord>> {
        let cs = CHUNK_SIZE as i32;
        let (x0, z0) = (s.min_x() + cx * cs, s.min_z() + cz * cs);
        if s.body >= CHART_BODY_BASE {
            let (ox, oz) = self.generator.chart_origin(s.body, s.face)?;
            let (x, z) = ((ox + x0).div_euclid(cs), (oz + z0).div_euclid(cs));
            let y = match self.near.frame() {
                Some(Frame::Chart { floor, .. }) => floor,
                _ => 0,
            };
            return Some(vec![Coord::new(x, y, z)]);
        }
        match self.near.frame() {
            Some(Frame::Flat) => Some(vec![Coord::new(x0.div_euclid(cs), 0, z0.div_euclid(cs))]),
            Some(Frame::Cube { body, face, shift }) if s.body == body && s.face == face => {
                let f = FaceFrame::new(face);
                let mut out = Vec::new();
                for (du, dv) in [(0, 0), (cs - 1, 0), (0, cs - 1), (cs - 1, cs - 1)] {
                    let (x, y, z) = f.cell_to_world((x0 + du, 0, z0 + dv));
                    let r = World::chunk_of(x, y, z);
                    let mut c = [r.x - shift[0], r.y - shift[1], r.z - shift[2]];
                    // At the centre's layer, inside the cube's storage: a column is any layer of it.
                    if let Some(centre) = self.center {
                        c[face.axis()] = [centre.x, centre.y, centre.z][face.axis()];
                    }
                    let c = Coord::new(c[0], c[1], c[2]);
                    let (key, _) = ColumnKey::of(face, c);
                    if !out.iter().any(|&o: &Coord| ColumnKey::of(face, o).0 == key) {
                        out.push(c);
                    }
                }
                Some(out)
            }
            _ => None,
        }
    }

    /// Every chunk of real column `real`'s shell `[lo, hi]` is final.
    pub(in crate::world) fn column_settled(&self, real: ColumnKey, [lo, hi]: [i32; 2]) -> bool {
        (lo..=hi).all(|alt| {
            let c = real.chunk(alt);
            self.chunk_final(c)
        })
    }
}
