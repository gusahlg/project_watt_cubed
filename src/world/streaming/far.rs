//! Far-field selection: the far eyes, the section frontier, chart seats and the near-window punch.

use super::*;

/// Blocks past a chart's stored top that still stream on that chart. The band ends
/// `RELIEF` above the datum and the crust tops out at `MAX_GROUND`, so this clears
/// three thousand blocks of flight over the highest crust.
const CHART_FLIGHT: f64 = 2_048.0;

/// Blocks above a round world's stored top within which the far field still stands on its chart.
/// Higher up the body's impostor alone draws it.
const FAR_FLIGHT: f64 = 262_144.0;

/// Hysteresis of the far field's thresholds: crossing one back takes this factor more height than
/// crossing it did (the far reach is left at `FAR_FLIGHT · FAR_HOLD`, a ring doubling dropped at
/// `1 / FAR_HOLD` of the height that took it, and the chart stood on is left for another round
/// world's only when that top is this factor nearer).
const FAR_HOLD: f64 = 1.25;

/// The chart rings reach this many times the eye's height above the ground.
const FAR_VIEW: f64 = 3.0;

/// Candidates the coarsest chart ring may sweep (its square of sections), twice the section floor.
const FAR_CANDIDATES: f64 = (2 * super::SECTION_SLOT_FLOOR) as f64;

/// The chunk holding point `p`.
pub(super) fn eye_chunk(p: DVec3) -> Coord {
    World::chunk_of(block_coord(p.x), block_coord(p.y), block_coord(p.z))
}

/// Sections from the eye's out to the edge of the square the coarsest ring sweeps, for a ring
/// reaching `outer` metres in sections of `span` (`quadtree::desired_sections`).
fn ring_reach(outer: f64, span: f64) -> f64 {
    (outer / span).ceil() + 1.0
}

/// Storage block the chart frontier treats as the eye: the centre chunk's middle, plus the
/// prediction delta. Y is the streamed altitude, not the chunk layer.
pub(super) fn storage_eye_block(center: Coord, eye_y: f64, delta: DVec3) -> (i64, i64, i64) {
    let cs = CHUNK_SIZE as i64;
    let x = center.x as i64 * cs + cs / 2 + delta.x.round() as i64;
    let z = center.z as i64 * cs + cs / 2 + delta.z.round() as i64;
    let y = (eye_y + delta.y).round() as i64;
    (x, y, z)
}

/// Generator surface bounds of the storage rects the near-window punch tested, stamped with the
/// frontier sweep that last read them. The terrain is immutable, so an answer never changes; a
/// sweep keeps only the rects it read.
#[derive(Default)]
pub(in crate::world) struct NearBounds {
    pub(super) rects: FastMap<(u16, [i32; 4]), (Option<(i32, i32)>, u32)>,
    pub(super) pass: u32,
}

impl NearBounds {
    /// Run one frontier sweep over the memo, then drop the rects it did not read.
    pub(super) fn sweep<R>(&mut self, select: impl FnOnce(&mut Self) -> R) -> R {
        self.pass = self.pass.wrapping_add(1);
        let out = select(self);
        let pass = self.pass;
        self.rects.retain(|_, e| e.1 == pass);
        out
    }

    /// The bounds of `rect`, read through `surface` the first time.
    pub(super) fn get(&mut self, rect: (u16, [i32; 4]), surface: impl FnOnce() -> Option<(i32, i32)>) -> Option<(i32, i32)> {
        let pass = self.pass;
        let e = self.rects.entry(rect).or_insert_with(|| (surface(), pass));
        e.1 = pass;
        e.0
    }
}

/// Sections of a chart seat, already filtered. Distance rings chose the detail: collapsing every
/// complete quad would flatten those rings onto the chord cap, so a quad merges only while the
/// frontier is over `budget`, and only when the parent passes `keep`.
fn coarsen_chart(
    sections: Vec<SectionPos>,
    max_detail: i8,
    budget: usize,
    mut keep: impl FnMut(SectionPos) -> bool,
) -> Vec<SectionPos> {
    let mut set: FastSet<SectionPos> = sections.into_iter().filter(|s| s.detail.0 <= max_detail).collect();
    if set.len() <= budget || set.is_empty() {
        return set.into_iter().collect();
    }
    let finest = set.iter().map(|s| s.detail.0).min().unwrap();
    for child_d in (finest..max_detail).rev() {
        if set.len() <= budget {
            break;
        }
        let mut merges: Vec<SectionPos> =
            count_children(&set, child_d).into_iter().filter(|&(p, n)| n == 4 && keep(p)).map(|(p, _)| p).collect();
        merges.sort_unstable_by_key(section_key);
        for p in merges {
            if set.len() <= budget {
                break;
            }
            for q in super::section::Quadrant::ALL {
                set.remove(&p.child(q));
            }
            set.insert(p);
        }
    }
    set.into_iter().collect()
}

/// How many tiles of `set` at detail `child_d` each parent has.
fn count_children(set: &FastSet<SectionPos>, child_d: i8) -> FastMap<SectionPos, u8> {
    let mut kids: FastMap<SectionPos, u8> = FastMap::default();
    for &c in set {
        if c.detail.0 == child_d {
            *kids.entry(c.parent()).or_insert(0) += 1;
        }
    }
    kids
}

/// Home-chart footprint of `s` (`hi` exclusive). A neighbour section unfolds across the seam.
fn home_rect(s: SectionPos, across: Option<&super::seam::SeamAcross>) -> (i64, i64, i64, i64) {
    let span = s.span() as i64;
    let (x0, z0) = (s.min_x() as i64, s.min_z() as i64);
    if let Some(m) = across {
        let (a, c) = m.home_xz(x0, z0);
        let (b, d) = m.home_xz(x0 + span, z0 + span);
        (a.min(b), c.min(d), a.max(b), c.max(d))
    } else {
        (x0, z0, x0 + span, z0 + span)
    }
}

/// Whether `s` meets the full-res chunk box. A neighbour section is tested in the home chart,
/// unfolded past the seam.
fn covers_near(s: SectionPos, near: (i64, i64, i64, i64), across: Option<&super::seam::SeamAcross>) -> bool {
    let (x0, z0, x1, z1) = home_rect(s, across);
    x0 < near.1 && x1 > near.0 && z0 < near.3 && z1 > near.2
}

/// Farthest and nearest distance from `(eu, ev)` to the closed span square of `s`.
/// The farthest corner is what [`World::full_res_covers`](super::World::full_res_covers) tests.
pub(in crate::world) fn span_reach(s: SectionPos, eu: i32, ev: i32) -> (f32, f32) {
    // In f64: block distances to far tiles reach millions, and their squares overflow i32.
    let span = f64::from(s.span());
    let (x0, z0) = (f64::from(s.min_x()), f64::from(s.min_z()));
    let (x1, z1) = (x0 + span, z0 + span);
    let (eu, ev) = (f64::from(eu), f64::from(ev));
    let fx = (x0 - eu).abs().max((x1 - eu).abs());
    let fz = (z0 - ev).abs().max((z1 - ev).abs());
    let dx = (x0 - eu).max(eu - x1).max(0.0);
    let dz = (z0 - ev).max(ev - z1).max(0.0);
    (fx.hypot(fz) as f32, dx.hypot(dz) as f32)
}

/// Pieces of a cube-face section against the full-view skip disk. A tile that crosses the disk, or
/// sits wholly inside it while its relief leaves the near window, is replaced by the largest
/// descendants that do not, down to detail 0 (span 32). A piece wholly inside whose relief fits
/// stays, so the chunks can take it over once they are final. A tile still crossing at detail 0
/// stays: dropping it would hole the sliver outside the disk.
fn push_split(
    s: SectionPos,
    eu: i32,
    ev: i32,
    h_lim: f32,
    margin: f32,
    out: &mut Vec<SectionPos>,
    sticks: &impl Fn(SectionPos) -> bool,
) {
    let (far, near) = span_reach(s, eu, ev);
    let inside = far + margin <= h_lim;
    let crosses = near < h_lim - margin && !inside;
    if s.detail.0 == 0 || !(crosses || (inside && sticks(s))) {
        out.push(s);
        return;
    }
    for q in super::section::Quadrant::ALL {
        push_split(s.child(q), eu, ev, h_lim, margin, out, sticks);
    }
}

/// Merge tiles that miss the skip disk until `set` fits `budget`. A parent that meets the disk
/// is left split: merging it would put a coarse tile back over the player.
fn coarsen_off_disk(set: &mut FastSet<SectionPos>, budget: usize, eu: i32, ev: i32, limit: f32) {
    let cap = crate::render_config::LOD_COARSEST_DETAIL as i8;
    while set.len() > budget {
        let mut merged = false;
        for child_d in (0..cap).rev() {
            if set.len() <= budget {
                break;
            }
            let mut parents: Vec<_> = count_children(set, child_d).into_iter().filter(|&(_, n)| n >= 2).collect();
            parents.sort_unstable_by_key(|(p, n)| (std::cmp::Reverse(*n), p.body, p.face as u8, p.x, p.z));
            for (p, _) in parents {
                if set.len() <= budget {
                    break;
                }
                if span_reach(p, eu, ev).1 < limit {
                    continue;
                }
                let mut removed = 0u8;
                for q in super::section::Quadrant::ALL {
                    if set.remove(&p.child(q)) {
                        removed += 1;
                    }
                }
                if removed > 0 {
                    set.insert(p);
                    merged = true;
                }
            }
        }
        if !merged {
            break;
        }
    }
}

/// Whether `s` lies wholly inside the full-res chunk box.
pub(super) fn inside_near(s: SectionPos, near: (i64, i64, i64, i64), across: Option<&super::seam::SeamAcross>) -> bool {
    let (x0, z0, x1, z1) = home_rect(s, across);
    x0 >= near.0 && x1 <= near.1 && z0 >= near.2 && z1 <= near.3
}

/// Storage rectangle of `s` inside the full-res box, `(u0, v0, u1, v1)` exclusive.
/// `None` when that overlap is empty or does not land back inside `s`.
fn overlap_storage(
    s: SectionPos,
    near: (i64, i64, i64, i64),
    across: Option<&super::seam::SeamAcross>,
) -> Option<(i32, i32, i32, i32)> {
    let span = s.span() as i64;
    let (sx0, sz0) = (s.min_x() as i64, s.min_z() as i64);
    let (sx1, sz1) = (sx0 + span, sz0 + span);
    let (hx0, hz0, hx1, hz1) = home_rect(s, across);
    let ix0 = hx0.max(near.0);
    let iz0 = hz0.max(near.2);
    let ix1 = hx1.min(near.1);
    let iz1 = hz1.min(near.3);
    if ix0 >= ix1 || iz0 >= iz1 {
        return None;
    }
    let (u0, v0, u1, v1) = if let Some(m) = across {
        // Inclusive corners: a reflected exclusive edge is off by one.
        let (a, c) = m.storage_xz(ix0, iz0);
        let (b, d) = m.storage_xz(ix1 - 1, iz1 - 1);
        (
            a.min(b).max(sx0),
            c.min(d).max(sz0),
            (a.max(b) + 1).min(sx1),
            (c.max(d) + 1).min(sz1),
        )
    } else {
        (ix0, iz0, ix1, iz1)
    };
    if u0 >= u1 || v0 >= v1 {
        return None;
    }
    Some((
        i32::try_from(u0).ok()?,
        i32::try_from(v0).ok()?,
        i32::try_from(u1).ok()?,
        i32::try_from(v1).ok()?,
    ))
}

/// The section's storage square lies wholly inside the chart box (`hi` exclusive).
pub(super) fn inside_xz(s: SectionPos, lo: [i64; 3], hi: [i64; 3]) -> bool {
    let span = s.span() as i64;
    let (x, z) = (s.min_x() as i64, s.min_z() as i64);
    x >= lo[0] && x + span <= hi[0] && z >= lo[2] && z + span <= hi[2]
}

/// The section's storage square meets the chart box.
fn overlaps_xz(s: SectionPos, lo: [i64; 3], hi: [i64; 3]) -> bool {
    let span = s.span() as i64;
    let (x, z) = (s.min_x() as i64, s.min_z() as i64);
    x < hi[0] && x + span > lo[0] && z < hi[2] && z + span > lo[2]
}

/// Wholly-inside pieces of `s`. A tile that crosses an edge is replaced by the largest descendants
/// that do not: the edge is not a multiple of the coarser spans, so dropping the straddler leaves
/// a strip of the chart with nothing drawn. Finest tiles meet a 128-aligned edge and are not split.
fn cover_chart(s: SectionPos, lo: [i64; 3], hi: [i64; 3], out: &mut Vec<SectionPos>) {
    if !overlaps_xz(s, lo, hi) {
        return;
    }
    if inside_xz(s, lo, hi) {
        out.push(s);
        return;
    }
    if s.detail.0 <= super::section::FINEST_DETAIL.0 {
        return;
    }
    for q in super::section::Quadrant::ALL {
        cover_chart(s.child(q), lo, hi, out);
    }
}

/// Pieces of `s` against the full-res box. A tile that crosses the edge is replaced by the largest
/// descendants that do not. Wholly outside pieces stay. Wholly inside pieces stay for the punch.
/// Descent continues past the finest far level: that span is wider than the full-res box, so a tile
/// kept there redraws the whole near field. It stops at detail 0 (span 32), the last grid coarser
/// than a chunk. A tile still crossing the edge stays, so the sliver beside the box is drawn.
fn cover_near(
    s: SectionPos,
    near: (i64, i64, i64, i64),
    across: Option<&super::seam::SeamAcross>,
    out: &mut Vec<SectionPos>,
) {
    let crosses = covers_near(s, near, across) && !inside_near(s, near, across);
    if !crosses || s.detail.0 <= 0 {
        out.push(s);
        return;
    }
    for q in super::section::Quadrant::ALL {
        cover_near(s.child(q), near, across, out);
    }
}

/// The far field's eye metric at `eye` on ladder `cfg`, over the LOD height envelope.
fn lod_metric(eye: DVec3, cfg: &pyramid::PyramidCfg) -> EyeMetric {
    let env = HeightEnvelope::new(super::section::LOD_FLOOR_Y as f32, super::section::LOD_CEIL_Y as f32);
    EyeMetric::new(eye, env, DyCap::new(cfg.outer_m(), cfg.base))
}

fn section_dist2(s: SectionPos, ex: f64, ez: f64) -> f64 {
    let span = s.span() as f64;
    let dx = s.min_x() as f64 + span * 0.5 - ex;
    let dz = s.min_z() as f64 + span * 0.5 - ez;
    dx * dx + dz * dz
}

pub(super) fn section_key(s: &SectionPos) -> (u16, u8, crate::ident::Detail, i32, i32) {
    (s.body, s.face as u8, s.detail, s.x, s.z)
}

impl World {
    /// Storage position of an eye on or above a round body, including flight past the stored
    /// top ([`CHART_FLIGHT`], wider than the stream window). `None` on a flat world and away
    /// from every chart.
    pub(crate) fn chart_eye(&self, eye: DVec3) -> Option<DVec3> {
        if self.seams.is_empty() {
            return None;
        }
        let window = (self.view.horizontal.max(self.view.vertical) + super::DATA_MARGIN + 2) as f64
            * CHUNK_SIZE as f64;
        self.seams.storage_eye(eye, window.max(CHART_FLIGHT))
    }

    /// The point streaming stands on: the eye's storage position on (or above) a round world's
    /// chart, else the eye itself.
    pub(crate) fn stream_eye(&self, eye: DVec3) -> DVec3 {
        self.chart_eye(eye).unwrap_or(eye)
    }

    /// The points streaming stands on for physical eye `eye`: the near window's
    /// ([`stream_eye`](Self::stream_eye)) and the far field's, whose altitude it captures.
    pub(in crate::world) fn place_eyes(&mut self, eye: DVec3) -> (DVec3, DVec3) {
        let chart = self.chart_eye(eye);
        let near = chart.unwrap_or(eye);
        let far = if self.lod2 { self.adopt_far_eye(eye, chart) } else { near };
        self.section_eye_y = far.y;
        (near, far)
    }

    /// The far field's eye: the near window's chart eye `chart`, else the chart column under the
    /// nearest round world within [`FAR_FLIGHT`] of its top, else `eye`. Commits the atlas it
    /// stands on and the altitude scale of its rings.
    fn adopt_far_eye(&mut self, eye: DVec3, chart: Option<DVec3>) -> DVec3 {
        let far = chart.or_else(|| self.seams.far_eye(eye, FAR_FLIGHT, self.far_atlas, FAR_HOLD));
        let seat = far.and_then(|p| self.seams.chart_seat(eye_chunk(p)));
        self.far_atlas = seat.map(|s| s.index);
        self.far_scale = match (far, seat) {
            (Some(p), Some(s)) => {
                let ground = self.far_ground(p);
                let h = if ground == i32::MIN { 0.0 } else { (p.y - f64::from(ground)).max(0.0) };
                self.far_scale_at(h, s.radius)
            }
            _ => 0,
        };
        far.unwrap_or(eye)
    }

    /// The generated ground of storage point `p`'s column, kept until the column changes.
    fn far_ground(&mut self, p: DVec3) -> i32 {
        let column = (block_coord(p.x), block_coord(p.z));
        if let Some((c, ground)) = self.far_ground
            && c == column
        {
            return ground;
        }
        let ground = self.generator.surface(Face::PosY, column.0, column.1);
        self.far_ground = Some((column, ground));
        ground
    }

    /// Unit doublings of the chart rings at height `h` over the ground on a chart of datum radius
    /// `radius`: the rings reach [`FAR_VIEW`] times the height, a doubling is dropped only
    /// [`FAR_HOLD`] lower than it was taken, and the coarsest ring's square stays within
    /// [`FAR_CANDIDATES`] (its detail is capped by the chord rule, so only its reach can grow).
    fn far_scale_at(&self, h: f64, radius: i64) -> u8 {
        let Some((cfg, max_d)) = self.chart_pyramid(radius) else { return 0 };
        let outer = f64::from(cfg.outer_m()) / f64::from(1u32 << self.far_scale);
        let span = f64::from(super::section::section_span(crate::ident::Detail(max_d)));
        let rows = |s: u8| 2.0 * ring_reach(outer * f64::from(1u32 << s), span) + 1.0;
        let mut cap = 0u8;
        while cap < 16 && rows(cap + 1).powi(2) <= FAR_CANDIDATES {
            cap += 1;
        }
        let level = |reach: f64| (0..cap).find(|&s| outer * f64::from(1u32 << s) >= reach).unwrap_or(cap);
        self.far_scale.clamp(level(FAR_VIEW * h), level(FAR_VIEW * FAR_HOLD * h))
    }

    /// The section ladder with its unit doubled [`far_scale`](World::far_scale) times (the
    /// ladder itself off charts, where the scale is 0).
    pub(super) fn far_pyramid(&self) -> pyramid::PyramidCfg {
        let src = &self.section_pyramid;
        let unit = src.unit * (1u32 << self.far_scale) as f32;
        pyramid::PyramidCfg::sections_with(unit, src.levels.get(), src.finest.0 as u8)
    }

    /// The far field's descheduling horizon in metres from its eye: the ladder's outer edge, and on
    /// a chart the half-diagonal of the coarsest ring's square of sections, whose corners reach
    /// far past that edge. The selection is untouched; only the worker gate reads this.
    pub(super) fn far_horizon(&self) -> f64 {
        let outer = f64::from(self.far_pyramid().outer_m());
        let chart = self.far_atlas.and_then(|i| self.chart_pyramid(self.seams.atlases()[i].radius));
        let Some((cfg, max_d)) = chart else { return outer };
        let span = f64::from(super::section::section_span(crate::ident::Detail(max_d)));
        outer.max(ring_reach(f64::from(cfg.outer_m()), span) * span * std::f64::consts::SQRT_2)
    }

    /// The far field's centre chunk (the streaming centre unless it stands on a chart the near
    /// window has left). `None` before the first stream.
    pub(in crate::world) fn section_center(&self) -> Option<Coord> {
        self.center.map(|c| self.far_center.unwrap_or(c))
    }

    /// Move the far field's centre to `centre`, with the chart net the gate measures far work in.
    /// Returns whether it moved.
    pub(in crate::world) fn set_far_center(&mut self, centre: Coord) -> bool {
        if self.far_center.replace(centre) == Some(centre) {
            return false;
        }
        self.far_fold = self.far_unfold(centre);
        true
    }

    /// The chart net around far-field centre `centre`: the identity off charts.
    fn far_unfold(&self, centre: Coord) -> super::seam::Unfold {
        if self.section_on_chart(centre) { self.seams.unfold_at(centre) } else { super::seam::Unfold::IDENTITY }
    }

    /// The far field standing on chunk `centre`, as the job gate is told it (the net is the cached
    /// one when that is the far centre).
    pub(super) fn far_view(&self, centre: Coord) -> pipeline::FarView {
        let fold = if self.far_center == Some(centre) { self.far_fold } else { self.far_unfold(centre) };
        pipeline::FarView { x: centre.x, z: centre.z, fold }
    }

    /// Storage column `(x, z)` in the chart net around `center`. Home columns stay themselves and a
    /// neighbour's lands just past the seam. A column outside the net stays put: it is far, and a
    /// non-finite stand-in would collapse to distance zero and be kept.
    pub(in crate::world) fn net_column(&self, center: Coord, x: i64, z: i64) -> (i64, i64) {
        let fold = if self.far_center == Some(center) {
            self.far_fold
        } else if self.center == Some(center) {
            self.fold
        } else {
            self.seams.unfold_at(center)
        };
        fold.fold_column(x, z).unwrap_or((x, z))
    }

    /// Whether far-field centre `center` stands on a round world's chart: in or above a storage
    /// box that is not a warped cube's.
    pub(super) fn section_on_chart(&self, center: Coord) -> bool {
        let (_, _, in_cube) = self.lod_place(center);
        !in_cube && self.seams.in_column(center)
    }

    /// Adopt the chart net around streaming centre `centre`; returns whether it changed. A new net
    /// drops the previous boxes' diffs (they were measured in the old net) and re-buckets the
    /// worklists. The worker gate hears the net with the centre, in that publish — not here.
    /// Publishing the net while the centre is still the previous chart's chunk deschedules every
    /// queued near job.
    pub(in crate::world) fn adopt_fold(&mut self, centre: Coord) -> bool {
        let fold = self.seams.unfold_at(centre);
        if fold == self.fold {
            return false;
        }
        self.fold = fold;
        self.prev_mesh_box = None;
        self.prev_unload_box = None;
        self.mesh_worklist.set_fold(fold);
        self.light_worklist.set_fold(fold);
        true
    }

    /// Chunk bucket for the far-field key during very fast flight.
    pub(super) fn frontier_bucket(&self, v: i32) -> i32 {
        if self.stream_pacer.speed_mps() < FRONTIER_COARSE_SPEED {
            return v;
        }
        let q = FRONTIER_CHUNK_QUANTUM;
        v.div_euclid(q) * q
    }

    /// Altitude the far-field key reads. The selection still runs at the live
    /// altitude on the frame the key changes.
    fn frontier_eye_y(&self) -> f64 {
        if self.stream_pacer.speed_mps() < FRONTIER_COARSE_SPEED {
            return self.section_eye_y;
        }
        let q = FRONTIER_EYE_QUANTUM;
        (self.section_eye_y / q).round() * q
    }

    /// Recompute the far-field selection when its inputs moved (see [`SectionFrontierKey`]).
    pub(in crate::world) fn refresh_frontier(&mut self, center: Coord) {
        // ONE selection sweep, retained across passes: unloading, the load
        // lane, and the covering rebuild below all read this cache. The
        // frontier is a pure function of the key's inputs (eye, velocity,
        // ladder, relief-mip readiness), so while they are bit-identical —
        // a still camera — the sweep (grid walk + relief coarsening) is
        // skipped entirely. Edits force a recompute: relief coarsening
        // consults the edit overlay, which the key cannot cheaply cover.
        let (body, face_u8, cu, cv) = match self.section_lod_face {
            Some((b, f)) => {
                let (cu, _, cv) = FaceFrame::new(f).chunk_to_local(center);
                (b, f as u8, cu, cv)
            }
            // A chart has no cube face. The storage centre still has to invalidate the frontier.
            None if self.section_on_chart(center) => (u16::MAX, u8::MAX, center.x, center.z),
            None => (u16::MAX, u8::MAX, 0, 0),
        };
        // A chart reads whole blocks (`storage_eye_block`), so its key is exact on them: an
        // eye that moves within one block keeps the frontier. A cube face reads the exact eye;
        // its velocity is quantised to 0.25 m/s so a continuously changing flight velocity does
        // not recompute the frontier every pass.
        let on_chart = self.section_on_chart(center);
        let (eye_y, velocity) = if on_chart {
            let d = chart_delta(self.section_vel);
            let y = self.frontier_eye_y();
            (y.round().to_bits(), [d.x.to_bits(), (y + d.y).round().to_bits(), d.z.to_bits()])
        } else {
            let v = (self.section_vel * 4.0).round();
            (self.frontier_eye_y().to_bits(), [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()])
        };
        // Slot churn of a fast flight would otherwise change `allowed` every
        // frame and rebuild the sweep. A 64-slot bucket still tracks a real
        // budget change. Below [`FRONTIER_COARSE_SPEED`] the count is exact.
        let mut allowed = self.sections_allowed() as u32;
        if self.stream_pacer.speed_mps() >= FRONTIER_COARSE_SPEED {
            allowed &= !63;
        }
        let frontier_key = SectionFrontierKey {
            center_xz: [self.frontier_bucket(cu), self.frontier_bucket(cv)],
            center_y: self.frontier_bucket(center.y),
            body,
            face: face_u8,
            eye_y,
            velocity,
            vertical: self.view.vertical,
            up: self.live_up(),
            window: self.window.punch.filter(|_| on_chart),
            unit: self.far_pyramid().unit.to_bits(),
            finest: self.section_pyramid.finest.0,
            levels: self.section_pyramid.levels.get(),
            step: self.section_pyramid.step(),
            mip_ready: self.section_mip.is_some(),
            allowed,
        };
        if self.section_frontier_key != Some(frontier_key) || !self.dirty_sections.is_empty() {
            let mut memo = std::mem::take(&mut self.near_bounds);
            let mut held = Vec::new();
            self.section_desired = memo.sweep(|memo| self.desired_sections_with(center, memo, &mut held));
            self.near_bounds = memo;
            self.wait_held(held);
            self.held_recheck.take();
            self.section_frontier_key = Some(frontier_key);
            self.section_cover_dirty.set();
        } else if self.held_recheck.take() && !self.section_held.is_empty() {
            self.settle_held();
        }
    }

    // Column-LOD section selection and streaming.

    /// Per-frame selection metric: chunk-centre tangents, `dy` from eye altitude to the LOD envelope.
    /// Tangents stay on the chunk centre (not the raw eye) so PosY `dy=0` stays bit-identical.
    /// Altitude is relative to the face datum, so the envelope stays `[0, 512]` on every face.
    /// Streaming centre in a warped-cube box, mapped back to the reference cube. Outside every
    /// cube box this is the centre unchanged, so an identity fold stays bit-identical. A reference
    /// centre is not inside a box, so a second call does not translate again.
    pub(in crate::world) fn lod_place(&self, center: Coord) -> (Coord, f64, bool) {
        let cs = CHUNK_SIZE as i64;
        let cell = [center.x as i64 * cs, center.y as i64 * cs, center.z as i64 * cs];
        for atlas in self.generator.atlases() {
            let Some(g) = atlas.grid else { continue };
            let inside = (0..3).all(|a| cell[a] >= g.origin[a] && cell[a] < g.origin[a] + g.size[a]);
            if !inside {
                continue;
            }
            let d = [
                (g.ref_min[0] - g.origin[0]) / cs,
                (g.ref_min[1] - g.origin[1]) / cs,
                (g.ref_min[2] - g.origin[2]) / cs,
            ];
            let reference = Coord::new(
                (center.x as i64 + d[0]) as i32,
                (center.y as i64 + d[1]) as i32,
                (center.z as i64 + d[2]) as i32,
            );
            let eye_y = self.section_eye_y + (g.ref_min[1] - g.origin[1]) as f64;
            return (reference, eye_y, true);
        }
        (center, self.section_eye_y, false)
    }

    /// Cells added to a storage-frame altitude on `face` to reach the reference cube.
    /// Zero when `reference` is `storage` (anything outside a cube box).
    pub(in crate::world) fn face_alt_shift(storage: Coord, reference: Coord, face: Face) -> i64 {
        let (_, sa) = ColumnKey::of(face, storage);
        let (_, ra) = ColumnKey::of(face, reference);
        (i64::from(ra) - i64::from(sa)) * i64::from(CHUNK_SIZE as i32)
    }

    pub(super) fn section_metric_on(&self, center: Coord, delta: DVec3, face: Face, datum: i32) -> EyeMetric {
        let (center, eye_y, _) = self.lod_place(center);
        let cs = CHUNK_SIZE as i32;
        let cfg = &self.section_pyramid;
        if face == Face::PosY && datum == 0 {
            let (pcx, pcz) = (center.x * cs + cs / 2, center.z * cs + cs / 2);
            return lod_metric(DVec3::new(pcx as f64 + delta.x, eye_y + delta.y, pcz as f64 + delta.z), cfg);
        }
        let frame = FaceFrame::new(face);
        let (cu, _, cv) = frame.chunk_to_local(center);
        let (u, v) = (cu * cs + cs / 2, cv * cs + cs / 2);
        let d = frame.point_to_local(delta);
        let eye = DVec3::new((center.x * cs + cs / 2) as f64, eye_y, (center.z * cs + cs / 2) as f64);
        let rel = frame.point_to_local(eye).y + d.y - datum as f64;
        lod_metric(DVec3::new(u as f64 + d.x, rel, v as f64 + d.z), cfg)
    }

    /// Desired frontier at one metric, stamped with `body`/`face` before coarsening
    /// so a parent keeps the frame and the mip lookup hits the right bake.
    fn frontier(&self, metric: &EyeMetric, body: u16, face: Face) -> Vec<SectionPos> {
        let cfg = &self.section_pyramid;
        let mut radial = quadtree::desired_sections(metric, cfg);
        for s in &mut radial {
            s.body = body;
            s.face = face;
        }
        let anchor = self.section_mip_anchor;
        let selected = match &self.section_mip {
            Some(mip) => {
                let summary_at = |c: SectionPos| {
                    if let Some((b, f, _, _)) = anchor
                        && (c.body != b || c.face != f)
                    {
                        return CellSummary {
                            env: HeightEnvelope::new(0.0, super::section::DOMAIN_H as f32),
                            err: CellError::worst_case(c.detail),
                        };
                    }
                    match self.section_overlay.get(&c) {
                        Some(ov) => CellSummary {
                            env: HeightEnvelope::new(ov.lo, ov.hi),
                            err: CellError::from_metres(ov.hi - ov.lo),
                        },
                        None => mip.summary(c),
                    }
                };
                quadtree::coarsen_by_error(radial, metric, cfg, &summary_at, &self.sse_budget())
            }
            None => radial,
        };
        // Prefer coarser tiles over dropping coverage when the far field
        // would exceed its section-slot budget (outermost ring first).
        quadtree::coarsen_to_budget(selected, self.sections_allowed(), cfg)
    }

    /// Ladder-pinned SSE budget for current view radius. Rebuilt per query
    /// as `unit` tracks view distance.
    fn sse_budget(&self) -> SseBudget {
        SseBudget::ladder(self.section_pyramid.unit, self.section_pyramid.finest.0)
    }

    /// Far sections of the home chart and, where the far field reaches a side, its neighbours.
    /// Storage +Y is the chart's up, so the sections are [`Face::PosY`] over storage `(x, z)`.
    /// Sections the near window holds go to `held` instead, with the chunk layers of their ground.
    fn chart_sections(&self, center: Coord, memo: &mut NearBounds, held: &mut Vec<(SectionPos, [i32; 2])>) -> Vec<SectionPos> {
        let Some(seat) = self.seams.chart_seat(center) else { return Vec::new() };
        let Some((cfg, max_d)) = self.chart_pyramid(seat.radius) else { return Vec::new() };
        let base = self.chart_pick(center, DVec3::ZERO, &seat, &cfg, max_d, memo, held);
        let delta = chart_delta(self.section_vel);
        if delta == DVec3::ZERO {
            return base;
        }
        quadtree::union_frontiers(base, self.chart_pick(center, delta, &seat, &cfg, max_d, memo, held))
    }

    /// Pyramid stopped at the largest detail whose span still satisfies `L² ≤ 8R`, on the
    /// altitude-scaled unit.
    fn chart_pyramid(&self, radius: i64) -> Option<(pyramid::PyramidCfg, i8)> {
        let src = self.far_pyramid();
        let mut levels = 0u8;
        let mut max_d = src.finest.0;
        for ring in 0..src.levels.get() {
            let detail = src.finest.0 + ring as i8 * src.step() as i8;
            let span = super::section::section_span(crate::ident::Detail(detail));
            if !super::section::section_fits(span, radius) {
                break;
            }
            levels += 1;
            max_d = detail;
        }
        (levels > 0).then(|| (pyramid::PyramidCfg::sections_with(src.unit, levels, src.finest.0 as u8), max_d))
    }

    /// The chart eye's storage block and its height above the column under it (`None` where that
    /// column has no ground).
    fn chart_eye_rel(&self, center: Coord, delta: DVec3) -> ((i64, i64, i64), Option<f64>) {
        let (ex, ey, ez) = storage_eye_block(center, self.section_eye_y, delta);
        let ground = self.generator.surface(Face::PosY, ex as i32, ez as i32);
        let rel = (ground != i32::MIN).then(|| (ey as f64 - ground as f64).clamp(0.0, 1.0e7));
        ((ex, ey, ez), rel)
    }

    /// Eye metric in the storage frame. Altitude is height above the column under the eye, so
    /// standing on a mountain still selects the finest ring (the cube envelope is `[0, 512]`).
    pub(super) fn chart_metric(&self, center: Coord, delta: DVec3, cfg: &pyramid::PyramidCfg) -> EyeMetric {
        let ((ex, _, ez), rel) = self.chart_eye_rel(center, delta);
        lod_metric(DVec3::new(ex as f64, rel.unwrap_or(0.0), ez as f64), cfg)
    }

    fn chart_pick(
        &self,
        center: Coord,
        delta: DVec3,
        seat: &super::seam::ChartSeat,
        cfg: &pyramid::PyramidCfg,
        max_d: i8,
        memo: &mut NearBounds,
        held: &mut Vec<(SectionPos, [i32; 2])>,
    ) -> Vec<SectionPos> {
        let ((ex, ey, ez), Some(rel)) = self.chart_eye_rel(center, delta) else {
            return Vec::new();
        };
        let body = super::section::CHART_BODY_BASE + seat.index as u16;
        let (y0, y1) = self.near_y_range(center);
        let near = self.chart_near(center, body, y0);
        let mut tagged =
            self.seat_sections(seat, ex as f64, ez as f64, rel, body, cfg, max_d, near, y0, y1, None, memo, held);
        // The neighbour is visible as far as the coarsest ring's square reaches, not merely the
        // two finest sections. Its rings run on from the eye unfolded beyond its edge.
        let span = super::section::section_span(crate::ident::Detail(max_d)) as f64;
        let band = (ring_reach(f64::from(cfg.outer_m()), span) * span) as i64;
        for across in self.seams.seam_across(*seat, [ex, ey, ez], band) {
            let (ix, iz) = across.storage_xz(ex, ez);
            tagged.extend(self.seat_sections(
                &across.seat,
                ix as f64,
                iz as f64,
                rel,
                body,
                cfg,
                max_d,
                near,
                y0,
                y1,
                Some(&across),
                memo,
                held,
            ));
        }
        let budget = self.sections_allowed();
        if tagged.len() > budget {
            // Nearest first, so the cap drops the far rim rather than the ground beside the eye.
            tagged.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| section_key(&a.0).cmp(&section_key(&b.0))));
            tagged.truncate(budget);
        }
        let mut out: Vec<_> = tagged.into_iter().map(|(s, _)| s).collect();
        out.sort_unstable_by_key(section_key);
        out
    }

    fn seat_sections(
        &self,
        seat: &super::seam::ChartSeat,
        ex: f64,
        ez: f64,
        rel: f64,
        body: u16,
        cfg: &pyramid::PyramidCfg,
        max_d: i8,
        near: (i64, i64, i64, i64),
        y0: i64,
        y1: i64,
        across: Option<&super::seam::SeamAcross>,
        memo: &mut NearBounds,
        held: &mut Vec<(SectionPos, [i32; 2])>,
    ) -> Vec<(SectionPos, f64)> {
        let metric = lod_metric(DVec3::new(ex, rel, ez), cfg);
        let mut radial = quadtree::desired_sections(&metric, cfg);
        for s in &mut radial {
            s.body = body;
            s.face = Face::PosY;
        }
        let mut clipped = Vec::new();
        for s in radial {
            cover_chart(s, seat.lo, seat.hi, &mut clipped);
        }
        // A straddler is replaced before the punch. Dropping it whole would leave the part
        // outside the box — up to one ring of span — drawn by nothing.
        let mut edged = Vec::new();
        for s in clipped {
            cover_near(s, near, across, &mut edged);
        }
        // Charts have no shader clip. A section wholly inside the near square is dropped only
        // when that square's surface sits inside the full-res window; a valley or a hilltop
        // outside it stays, so the far field draws what the window misses.
        let on_seat = |s: SectionPos| inside_xz(s, seat.lo, seat.hi) && super::section::section_fits(s.span(), seat.radius);
        let mut kept = Vec::with_capacity(edged.len());
        for s in edged.into_iter().filter(|&s| on_seat(s)) {
            match self.held_layers(s, near, y0, y1, across, memo) {
                Some(layers) => held.push((s, layers)),
                None => kept.push(s),
            }
        }
        // A straddling parent must not merge back: that tile is what the descent just replaced.
        let merge = |p: SectionPos| {
            on_seat(p)
                && self.held_layers(p, near, y0, y1, across, memo).is_none()
                && !(covers_near(p, near, across) && !inside_near(p, near, across))
        };
        coarsen_chart(kept, max_d, self.sections_allowed(), merge)
            .into_iter()
            .map(|s| (s, section_dist2(s, ex, ez)))
            .collect()
    }

    /// The chunk layers of `s`'s solid tops, when the full-res window draws every one of them and
    /// `s` lies wholly inside the near square. A section that only crosses the edge is not
    /// punched: the part outside the square would be drawn by nothing. Storage altitude:
    /// `surface` is the first open cell, so the solid top is the block below it. A bound we cannot
    /// place is kept (punched nowhere) so a missed column is not a sky hole.
    fn held_layers(
        &self,
        s: SectionPos,
        near: (i64, i64, i64, i64),
        y0: i64,
        y1: i64,
        across: Option<&super::seam::SeamAcross>,
        memo: &mut NearBounds,
    ) -> Option<[i32; 2]> {
        if !inside_near(s, near, across) {
            return None;
        }
        let (u0, v0, u1, v1) = overlap_storage(s, near, across)?;
        let surface = || self.generator.surface_rect(s.body, Face::PosY, u0, v0, u1, v1);
        let (lo, hi) = self.with_edits(memo.get((s.body, [u0, v0, u1, v1]), surface)?, [u0, v0, u1, v1]);
        let (top_lo, top_hi) = (lo.saturating_sub(1), hi.saturating_sub(1));
        let cs = CHUNK_SIZE as i32;
        (i64::from(top_lo) >= y0 && i64::from(top_hi) < y1).then(|| [top_lo.div_euclid(cs), top_hi.div_euclid(cs)])
    }

    /// The full-res square the punch tests chart sections against. Empty when the near window
    /// stands in physical space, or when its floor `y0` is above every surface of the square: it
    /// draws no ground there, so splitting the far field around it would punch nothing.
    fn chart_near(&self, center: Coord, body: u16, y0: i64) -> (i64, i64, i64, i64) {
        const NONE: (i64, i64, i64, i64) = (0, 0, 0, 0);
        if self.fold.is_identity() {
            return NONE;
        }
        let near = self.near_block_box(center);
        let span = i32::try_from((near.1 - near.0).max(near.3 - near.2)).unwrap_or(i32::MAX);
        match self.generator.surface_bounds(body, Face::PosY, near.0 as i32, near.2 as i32, span) {
            Some((_, hi)) if i64::from(hi) - 1 < y0 => NONE,
            _ => near,
        }
    }

    /// Full-res vertical block range (`y1` exclusive) the punch tests against: the punch window on
    /// a chart, else the mesh box. Up is storage Y.
    pub(super) fn near_y_range(&self, center: Coord) -> (i64, i64) {
        let cs = CHUNK_SIZE as i64;
        if let Some([lo, hi]) = self.window.punch {
            return (i64::from(lo) * cs, (i64::from(hi) + 1) * cs);
        }
        let b = self.mesh_box(center);
        let y0 = i64::from(b.min().y) * cs;
        (y0, y0 + i64::from(b.size().1) * cs)
    }

    /// Full-res chunk box in storage blocks (`hi` exclusive), wide on x/z. Up is storage Y, so this
    /// is the footprint a chart section overlaps when it meets the near square.
    pub(super) fn near_block_box(&self, center: Coord) -> (i64, i64, i64, i64) {
        let cs = CHUNK_SIZE as i64;
        let h = self.view.horizontal as i64;
        (
            (center.x as i64 - h) * cs,
            (center.x as i64 + h + 1) * cs,
            (center.z as i64 - h) * cs,
            (center.z as i64 + h + 1) * cs,
        )
    }

    /// [`desired_sections_with`](Self::desired_sections_with) reading every surface afresh: the
    /// selection once every held section's ground has settled.
    #[cfg(test)]
    pub(in crate::world) fn desired_sections(&self, center: Coord) -> Vec<SectionPos> {
        self.desired_sections_with(center, &mut NearBounds::default(), &mut Vec::new())
    }

    /// Desired frontier: union of static eye and velocity-predicted eye position.
    /// Pulls sections ahead of player motion. At rest, velocity is zero so returns
    /// static frontier bit-for-bit. Open space and a round body seen from past the far reach
    /// select nothing; a far-field centre on a chart (in or above its box) selects that chart's
    /// sections, reading chart surfaces through `memo`; the sections its near window holds go to
    /// `held`. On a warped cube, up-face tiles that cross the near disk are split to detail 0.
    fn desired_sections_with(
        &self,
        center: Coord,
        memo: &mut NearBounds,
        held: &mut Vec<(SectionPos, [i32; 2])>,
    ) -> Vec<SectionPos> {
        if self.section_on_chart(center) {
            return self.chart_sections(center, memo, held);
        }
        let focus = if self.section_face_set { self.section_lod_face } else { self.dominant_lod_face(center) };
        let Some((body, face)) = focus else { return Vec::new() };
        let mut out = self.frontier_union(center, body, face);
        for nface in self.edge_faces(center, body, face) {
            out = quadtree::union_frontiers(out, self.frontier_union(center, body, nface));
        }
        // The disk is the full view, not the reduced loading window: the frontier key has no
        // load radius, and a fast flight must not bake the small disk in.
        if self.lod_place(center).2 {
            out = self.split_near_cube(center, out);
        }
        out
    }

    /// [`desired_sections_with`](Self::desired_sections_with) on a warped cube: sections of the up
    /// face that cross the skip disk are split down to detail 0, then tiles clear of the disk are
    /// merged back until the frontier fits the section budget.
    fn split_near_cube(&self, center: Coord, sections: Vec<SectionPos>) -> Vec<SectionPos> {
        let Some(face) = self.live_up() else {
            return sections;
        };
        let (reference, _, _) = self.lod_place(center);
        let cs = CHUNK_SIZE as i32;
        let h_lim = 0.75 * self.view.coverage().half.x;
        let margin = cs as f32 * 0.5 * std::f32::consts::SQRT_2;
        let frame = FaceFrame::new(face);
        let mid = |c: i32| (i64::from(c) * i64::from(cs) + i64::from(cs / 2)) as i32;
        let (eu, _, ev) = frame.cell_to_local((mid(reference.x), mid(reference.y), mid(reference.z)));
        let shift = Self::face_alt_shift(center, reference, face);
        let band = self.window_alts().map(|[a0, a1]| (a0 + shift, a1 + shift));
        let sticks = |s: SectionPos| {
            let Some((a0, a1)) = band else { return false };
            let Some((lo, hi)) = self.cover_band(s, true) else { return false };
            lo < self.baked_height(s, a0) || hi > self.baked_height(s, a1)
        };
        let mut split = Vec::with_capacity(sections.len());
        for s in sections {
            if s.face == face {
                push_split(s, eu, ev, h_lim, margin, &mut split, &sticks);
            } else {
                split.push(s);
            }
        }
        let budget = self.sections_allowed();
        if split.len() > budget {
            let mut set: FastSet<SectionPos> = split.into_iter().collect();
            coarsen_off_disk(&mut set, budget, eu, ev, h_lim - margin);
            split = set.into_iter().collect();
        }
        split.sort_unstable_by_key(section_key);
        split
    }

    fn frontier_union(&self, center: Coord, body: u16, face: Face) -> Vec<SectionPos> {
        let datum = self.generator.face_datum(body, face);
        let base = self.frontier(&self.section_metric_on(center, DVec3::ZERO, face, datum), body, face);
        let delta = self.section_vel * TAU_STREAM;
        if delta == DVec3::ZERO {
            return base;
        }
        let predicted = self.frontier(&self.section_metric_on(center, delta, face, datum), body, face);
        quadtree::union_frontiers(base, predicted)
    }

    /// Chunk-centre sample the far field treats as the eye (tangents quantised, altitude exact on Y).
    fn lod_eye_point(&self, center: Coord) -> DVec3 {
        let (center, eye_y, _) = self.lod_place(center);
        let cs = CHUNK_SIZE as i32;
        DVec3::new((center.x * cs + cs / 2) as f64, eye_y, (center.z * cs + cs / 2) as f64)
    }

    /// Face-local chunk-centre tangents.
    pub(super) fn face_tangent_centre(&self, center: Coord, face: Face) -> (i32, i32) {
        let (center, _, _) = self.lod_place(center);
        let cs = CHUNK_SIZE as i32;
        let (cu, _, cv) = FaceFrame::new(face).chunk_to_local(center);
        (cu * cs + cs / 2, cv * cs + cs / 2)
    }

    /// The cube face under the camera. `None` in a chart's storage, in open space, or over a round
    /// body. A streaming centre inside a warped cube still names that cube's face.
    pub(super) fn dominant_lod_face(&self, center: Coord) -> Option<(u16, Face)> {
        if self.section_on_chart(center) {
            return None;
        }
        let Some(cosmos) = self.generator.cosmos() else {
            return Some((0, Face::PosY));
        };
        let eye = self.lod_eye_point(center);
        let body = cosmos.body_at(eye)?;
        if !matches!(body.shape, super::terrain::cosmos::Shape::Cube { .. }) {
            return None;
        }
        Some((body.id, Face::from_dominant(eye - body.centre_f())))
    }

    /// Commit the face for this pass. The previous face sticks while its component
    /// is within two finest sections of the dominant one (the edge).
    pub(super) fn update_lod_face(&mut self, center: Coord) {
        let dominant = self.dominant_lod_face(center);
        self.section_lod_face = match (self.section_lod_face, dominant) {
            (Some((id, prev)), Some((bid, _))) if id == bid && self.face_holds(center, id, prev) => Some((id, prev)),
            _ => dominant,
        };
        self.section_face_set = true;
    }

    /// Cube body `id` of the cosmos, with its half size. `None` off a cosmos or for another shape.
    fn cube_body(&self, id: u16) -> Option<(super::terrain::cosmos::Body, i64)> {
        let cosmos = self.generator.cosmos()?;
        let body = *cosmos.bodies().iter().find(|b| b.id == id)?;
        let super::terrain::cosmos::Shape::Cube { half } = body.shape else { return None };
        Some((body, half))
    }

    /// Whether cube `body_id`'s held face `prev` sticks (see [`update_lod_face`](Self::update_lod_face)).
    fn face_holds(&self, center: Coord, body_id: u16, prev: Face) -> bool {
        let Some((body, _)) = self.cube_body(body_id) else { return false };
        let rel = self.lod_eye_point(center) - body.centre_f();
        let comps = [rel.x.abs(), rel.y.abs(), rel.z.abs()];
        let max = comps[0].max(comps[1]).max(comps[2]);
        let band = (super::section::section_span(super::section::FINEST_DETAIL) * 2) as f64;
        max - comps[prev.axis()] <= band
    }

    /// Neighbouring faces whose squares are within two finest sections of the eye.
    fn edge_faces(&self, center: Coord, body_id: u16, face: Face) -> Vec<Face> {
        let Some((body, half)) = self.cube_body(body_id) else { return Vec::new() };
        let frame = FaceFrame::new(face);
        let local = frame.point_to_local(self.lod_eye_point(center) - body.centre_f());
        let band = (super::section::section_span(super::section::FINEST_DETAIL) * 2) as f64;
        let mut out = Vec::new();
        let mut push = |du: i32, dv: i32| {
            let (x, y, z) = frame.cell_to_world((du, 0, dv));
            let n = Face::from_dominant(DVec3::new(x as f64, y as f64, z as f64));
            if n != face && !out.contains(&n) {
                out.push(n);
            }
        };
        if half as f64 - local.x.abs() < band && local.x != 0.0 {
            push(local.x.signum() as i32, 0);
        }
        if half as f64 - local.z.abs() < band && local.z != 0.0 {
            push(0, local.z.signum() as i32);
        }
        out
    }

    /// Start-world sites the far field is checked from high above the ground: the direction from
    /// the centre and the heights. A face centre, a highland, a seam (77 km from the +X/+Y edge)
    /// and a cube corner (46 km from both edges of the +X chart).
    #[cfg(test)]
    pub(in crate::world) const FAR_SITES: [(&'static str, DVec3, &'static [f64]); 4] = [
        ("plus-y", DVec3::new(0.0, 1.0, 0.0), &[10_000.0, 50_000.0, 150_000.0]),
        ("highland", DVec3::new(1.0, 0.9, 0.8), &[10_000.0, 50_000.0, 150_000.0]),
        ("seam", DVec3::new(1.0, 0.995, 0.3), &[10_000.0, 50_000.0, 150_000.0]),
        ("corner", DVec3::new(1.0, 0.997, 0.997), &[50_000.0]),
    ];

    /// A physical eye `above` blocks out along the local up (the radial) from the start world's
    /// ground in direction `dir` from its centre.
    #[cfg(test)]
    pub(in crate::world) fn home_eye(&self, dir: DVec3, above: f64) -> DVec3 {
        use crate::space::atlas::Patch;
        use crate::space::chart::{self, Map};
        let centre = self.generator.cosmos().expect("cosmos").home().centre_f();
        let atlas = self
            .generator
            .atlases()
            .iter()
            .find(|a| (a.centre - centre).length() < 1.0)
            .expect("the start world is charted");
        let dir = dir.normalize();
        let face = Face::from_dominant(dir);
        let (tu, nn, tv) = chart::basis(face);
        let (xi, eta) = Map::Equiangular.inverse(DVec3::new(dir.dot(tu), dir.dot(nn), dir.dot(tv)));
        let n = atlas.bands[0].n;
        let step = 2.0 / n as f64;
        let (i, j) = (((xi + 1.0) / step).floor() as i64, ((eta + 1.0) / step).floor() as i64);
        assert!((0..n).contains(&i) && (0..n).contains(&j), "({i},{j}) leaves the {face:?} chart");
        let patch = Patch::Shell { band: 0, face };
        let (origin, _) = atlas.storage_box(patch);
        let stored = atlas.storage(patch, [i, 0, j]);
        let ground = self.generator.surface(Face::PosY, stored[0] as i32, stored[2] as i32);
        assert_ne!(ground, i32::MIN, "{face:?} column has no surface");
        let local_y = ground as f64 - origin[1] as f64;
        let surf = atlas.embed(patch, DVec3::new(i as f64 + 0.5, local_y, j as f64 + 0.5));
        surf + (surf - atlas.centre).normalize() * above
    }
}
