//! Far-field selection: the far eyes, the rule's cut and the clips it takes, chart seats and
//! cube faces.

use super::*;
use super::super::rule::{Ground, Leaf, PhysBox};

/// Blocks past a chart's stored top that still stream on that chart. The band ends
/// `RELIEF` above the datum and the crust tops out at `MAX_GROUND`, so this clears
/// three thousand blocks of flight over the highest crust.
const CHART_FLIGHT: f64 = 2_048.0;

/// Blocks above a round world's stored top within which the far field still stands on its chart.
/// Higher up the body's impostor alone draws it.
const FAR_FLIGHT: f64 = 262_144.0;

/// Hysteresis of the far field's chart: the chart stood on is left for another round world's only
/// when that top is this factor nearer, and the far reach is left at `FAR_FLIGHT · FAR_HOLD`.
const FAR_HOLD: f64 = 1.25;

/// The chunk holding point `p`.
pub(super) fn eye_chunk(p: DVec3) -> Coord {
    World::chunk_of(block_coord(p.x), block_coord(p.y), block_coord(p.z))
}

/// Storage block of the centre chunk's middle, plus `delta`. Y is the streamed altitude, not the
/// chunk layer.
#[cfg(test)]
pub(super) fn storage_eye_block(center: Coord, eye_y: f64, delta: DVec3) -> (i64, i64, i64) {
    let cs = CHUNK_SIZE as i64;
    let x = center.x as i64 * cs + cs / 2 + delta.x.round() as i64;
    let z = center.z as i64 * cs + cs / 2 + delta.z.round() as i64;
    let y = (eye_y + delta.y).round() as i64;
    (x, y, z)
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
    /// ([`stream_eye`](Self::stream_eye)) and the far field's, whose altitude it captures. The
    /// physical eye is kept with the chunks both stand on: the rule measures from it.
    pub(in crate::world) fn place_eyes(&mut self, eye: DVec3) -> (DVec3, DVec3) {
        let chart = self.chart_eye(eye);
        let near = chart.unwrap_or(eye);
        let far = if self.lod2 { self.adopt_far_eye(eye, chart) } else { near };
        self.section_eye_y = far.y;
        self.eye_at = Some((eye_chunk(near), eye_chunk(far), eye));
        (near, far)
    }

    /// The far field's eye: the near window's chart eye `chart`, else the chart column under the
    /// nearest round world within [`FAR_FLIGHT`] of its top, else `eye`. Commits the atlas it
    /// stands on.
    fn adopt_far_eye(&mut self, eye: DVec3, chart: Option<DVec3>) -> DVec3 {
        let far = chart.or_else(|| self.seams.far_eye(eye, FAR_FLIGHT, self.far_atlas, FAR_HOLD));
        let seat = far.and_then(|p| self.seams.chart_seat(eye_chunk(p)));
        self.far_atlas = seat.map(|s| s.index);
        far.unwrap_or(eye)
    }

    /// The physical eye that stood on streaming centre `center` (`far` for the far field's centre),
    /// or the physical point of the centre chunk's middle at the far eye's altitude when no eye
    /// stood there (a centre placed by hand).
    pub(in crate::world) fn phys_eye(&self, center: Coord, far: bool) -> DVec3 {
        if let Some((near_c, far_c, eye)) = self.eye_at
            && (if far { far_c } else { near_c }) == center
        {
            return eye;
        }
        let cs = CHUNK_SIZE as f64;
        let p = DVec3::new((f64::from(center.x) + 0.5) * cs, self.section_eye_y, (f64::from(center.z) + 0.5) * cs);
        let (reference, _, in_cube) = self.lod_place(center);
        if in_cube {
            let d = DVec3::new(f64::from(reference.x - center.x), f64::from(reference.y - center.y), f64::from(reference.z - center.z)) * cs;
            let r = p + d;
            let warp = self.seams.atlases().iter().find(|a| {
                a.grid.as_ref().is_some_and(|g| (0..3).all(|i| {
                    let c = [center.x, center.y, center.z][i] as i64 * CHUNK_SIZE as i64;
                    c >= g.origin[i] && c < g.origin[i] + g.size[i]
                }))
            }).and_then(|a| a.warp.clone());
            return warp.map_or(r, |w| w.apply(r));
        }
        let (x, z) = (p.x.floor() as i64, p.z.floor() as i64);
        self.seams
            .atlases()
            .iter()
            .find_map(|a| a.column(x, z).map(|(patch, _, _)| a.embed_storage(patch, p)))
            .unwrap_or(p)
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

    /// The far field's descheduling horizon in metres from its eye: the rule's horizon with the
    /// half-diagonal of a coarsest tile, whose centre the gate measures. Only the worker gate
    /// reads this.
    pub(super) fn far_horizon(&self) -> f64 {
        let rule = super::super::rule::Rule::new(f64::from(self.view.horizontal * CHUNK_SIZE as i32), crate::render_config::LOD_COARSEST_DETAIL as i8);
        let span = f64::from(super::section::section_span(crate::ident::Detail(rule.coarsest)));
        rule.horizon() + span * std::f64::consts::SQRT_2
    }

    /// The ground the far field stands on around far centre `center`: the chart it is on or over,
    /// the cube body it is in, or a flat world. `None` in open space.
    pub(in crate::world) fn far_ground(&self, center: Coord) -> Option<Ground> {
        if self.section_on_chart(center) {
            return self.seams.chart_seat(center).map(|s| Ground::Chart { index: s.index });
        }
        let Some(cosmos) = self.generator.cosmos() else { return Some(Ground::Flat) };
        let body = cosmos.body_at(self.lod_eye_point(center))?;
        matches!(body.shape, super::terrain::cosmos::Shape::Cube { .. }).then_some(Ground::Cube { body: body.id })
    }

    /// Recompute the far field's cut when its inputs moved (see [`SectionFrontierKey`]): the
    /// ground, the eye's cell, the render distance, the relief bake and the near field's columns
    /// the clips are taken from. An edit forces it (its bounds read the overlay).
    pub(in crate::world) fn refresh_frontier(&mut self, center: Coord) {
        let ground = self.far_ground(center);
        let eye = self.phys_eye(center, true);
        let cell = PhysBox::eye_cell(eye);
        let key = SectionFrontierKey {
            ground,
            eye_cell: [cell.lo.x as i64, cell.lo.y as i64, cell.lo.z as i64],
            horizontal: self.view.horizontal,
            mip_ready: self.section_mip.is_some(),
            near: self.near.rev,
        };
        if self.section_frontier_key != Some(key) || !self.dirty_sections.is_empty() {
            let mut leaves = Vec::new();
            let mut memo = std::mem::take(&mut self.rule_memo);
            self.section_desired = self.desired_sections_with(center, &mut memo, &mut leaves);
            self.rule_memo = memo;
            let wanted = self.wanted_clips(&leaves);
            self.section_dist = leaves.iter().map(|l| (l.pos, l.dist)).collect();
            self.wait_clips(wanted);
            self.clip_recheck.take();
            self.section_frontier_key = Some(key);
            self.section_cover_dirty.set();
        } else if self.clip_recheck.take() && self.clips_waiting() {
            self.settle_clips();
        }
    }

    /// [`desired_sections_with`](Self::desired_sections_with) alone.
    #[cfg(test)]
    pub(in crate::world) fn desired_sections(&self, center: Coord) -> Vec<SectionPos> {
        self.desired_sections_with(center, &mut Default::default(), &mut Vec::new())
    }

    /// The cut around far centre `center` and the clip each tile takes once every near column's
    /// chunks have settled (the near field as last placed).
    #[cfg(test)]
    pub(in crate::world) fn desired_and_clips(&self, center: Coord) -> (Vec<SectionPos>, FastMap<SectionPos, ColumnClip>) {
        let mut leaves = Vec::new();
        let desired = self.desired_sections_with(center, &mut Default::default(), &mut leaves);
        let clips = self
            .wanted_clips(&leaves)
            .into_iter()
            .map(|(s, bits)| (s, bits.iter().fold(ColumnClip::NONE, |m, (b, _)| m.union(*b))))
            .collect();
        (desired, clips)
    }

    /// The rule's cut around far centre `center`, sorted by [`section_key`], with its leaves (and
    /// their distances) in `leaves`. Open space selects nothing. Prediction never changes it: a
    /// moving eye only reorders what loads first.
    fn desired_sections_with(&self, center: Coord, memo: &mut super::super::rule::RuleMemo, leaves: &mut Vec<Leaf>) -> Vec<SectionPos> {
        leaves.clear();
        let Some(ground) = self.far_ground(center) else { return Vec::new() };
        let eye = self.phys_eye(center, true);
        self.rule_cut(ground, eye, memo, leaves);
        let mut out: Vec<SectionPos> = leaves.iter().map(|l| l.pos).collect();
        out.sort_unstable_by_key(section_key);
        debug_assert!(out.len() <= super::SECTION_SLOT_FLOOR, "the rule's cut holds {} tiles, past the section floor", out.len());
        out
    }
}
