//! The render clip: the settled chunk rings the LOD clip follows, and the proofs that full-res
//! chunks cover a far section.

use super::*;
use super::resident::FLAT_DETAIL;

impl World {
    /// Draw the meshed chunks. All per-voxel work happened when each chunk was
    /// built; a frame is one `draw_mesh` per chunk (the engine frustum-culls
    /// each against its offset AABB internally).
    ///
    /// Terrain chunks and LOD sections are RESIDENT meshes: placement and detail
    /// pinned at upload, visibility a `set_visible` mask, style a `set_style`
    /// push (both maintained in `stream`). The engine draws every visible
    /// resident mesh itself, so `render` submits no per-mesh draws — it only sets
    /// the frame's LOD-cull volume and reports the set-size gauge.
    pub fn render(&self, f: &mut Frame3D, cam: DVec3) {
        // The shader discards LOD-section fragments inside the SETTLED radius:
        // the rings whose chunks are actually drawn (or born-air). While a
        // loading edge is still meshing, the clip stays behind it and the far
        // sections keep covering the gap — coarse terrain instead of a hole —
        // then hands off ring by ring as uploads land. Fully settled, this is
        // exactly the old full-res radius.
        let (min, max) = self.lod_clip_box(cam);
        f.set_lod_clip_box(min, max);
        // Set-size gauge: a spike localizes a regression to a grown set (view
        // volume / section frontier). See `profile::Gauge`.
        use voxel_engine::profile::{Gauge, gauge};
        gauge(Gauge::WorldChunks, self.chunks.len() as u64);
    }

    /// The LOD-cull volume for this frame: the full-res slab shrunk to the
    /// settled rings. `rings` counts settled rings from the centre, so the
    /// nearest possibly-unsettled column sits at chess distance `rings`; its
    /// closest face is at least `(rings - 1) * 16` m from any eye position
    /// inside the centre chunk — the conservative discard radius. With every
    /// ring settled this is bit-identical to [`ViewVolume::coverage`].
    pub(super) fn lod_clip(&self) -> CoverageVolume {
        // A chart's full-res box is a storage square bent through its cages. The engine clip is an
        // axis-aligned box in camera space after that bend, and storage up is always +Y, so the box
        // would punch world Y on a tilted chart. Chart sections are clipped by key instead.
        if !self.fold.is_identity() {
            return CoverageVolume { half: Vec3::ZERO };
        }
        let full = self.view.coverage();
        let radius_m = ((self.lod_clip_rings - 1).max(0) * CHUNK_SIZE as i32) as f32;
        let hx = radius_m.min(full.half.x);
        let hv = full.half.y;
        // Settled radius on the two tangents, the full vertical extent on the up axis.
        // PosY (the default before a stream commits an up face) is (hx, hv, hx).
        // No up face: a cube of the settled radius, not the tall vertical slab.
        let (mut x, mut y, mut z) = (hx, hx, hx);
        if let Some(face) = self.live_up() {
            match face.axis() {
                0 => x = hv,
                1 => y = hv,
                _ => z = hv,
            }
        }
        CoverageVolume { half: Vec3::new(x, y, z) }
    }

    /// [`lod_clip`](Self::lod_clip) as extents about camera `cam`. Along the up axis a near window
    /// off a chart clips the span its rings are proven over, which may reach past the eye band.
    pub(super) fn lod_clip_box(&self, cam: DVec3) -> (Vec3, Vec3) {
        let half = self.lod_clip().half;
        let (mut min, mut max) = (-half, half);
        if let (Some(face), Some([lo, hi])) = (self.live_up(), self.lod_clip_span)
            && self.fold.is_identity()
        {
            let (a, cs) = (face.axis(), CHUNK_SIZE as f64);
            min[a] = (f64::from(lo) * cs - cam[a]) as f32;
            max[a] = ((f64::from(hi) + 1.0) * cs - cam[a]) as f32;
        }
        (min, max)
    }

    /// Fold a streaming-centre move into the settled-ring count WITHOUT
    /// restarting the scan. Proven-settled rings survive a move across the up
    /// axis, shifted down by its tangent chess distance `d`: a column at
    /// distance ρ ≤ rings−d−1 from the NEW centre lies at distance ≤ ρ+d ≤
    /// rings−1 from the old one, over the SAME span along the up axis. A move
    /// along that axis resets (`ring_settled` scans `center ± vertical` on the
    /// up axis, and the proof does not transfer across layers). With no up
    /// face the volume is a cube, so any move shifts by 3-D chess. The first
    /// pass (no previous centre) resets.
    ///
    /// Settledness cannot have regressed on this pass either: `unload_box` ⊇
    /// mesh box, so a boundary-cross unload never removes a chunk inside a
    /// countable ring; every OTHER regression (radius shrink, mesh teardown,
    /// `free_meshes`) still raises `lod_clip_shrunk`, and a raised shrink wins
    /// over the shift (the reset in [`refresh_lod_clip`](Self::refresh_lod_clip)
    /// runs after).
    ///
    /// Resetting on every boundary cross collapsed the far clip each crossed
    /// boundary: far LOD popped back over the whole settled near field for
    /// frames (the flying flicker) and every ring was re-proven from scratch.
    pub(in crate::world) fn shift_lod_clip(&mut self, prev: Option<Coord>, new: Coord) {
        let Some(p) = prev else {
            self.lod_clip_shrunk.set();
            return;
        };
        let up = self.live_up();
        // A window span is fixed in the world, so its proof survives a move along the axis.
        let along = match up {
            Some(face) if self.lod_clip_span.is_none() => p.along(new, face),
            _ => 0,
        };
        if along != 0 {
            self.lod_clip_shrunk.set();
            return;
        }
        let d = match up {
            Some(face) => p.across(new, face),
            None => p.chess3(new),
        };
        self.lod_clip_rings = (self.lod_clip_rings - d).max(0);
        if let Some((_, r)) = &mut self.lod_clip_next {
            *r = (*r - d).max(0);
        }
        self.lod_clip_grow.set();
    }

    /// Advance the settled-ring scan at a `&mut` sync point (end of `pump`
    /// and of `stream`). Self-gates on the two event flags: a converged,
    /// still frame is two flag checks. Growth re-scans only from the current
    /// frontier ring, so a loading wave costs each ring once, not per frame.
    pub(in crate::world) fn refresh_lod_clip(&mut self) {
        if self.lod_clip_shrunk.take() {
            self.lod_clip_rings = 0;
            if let Some((_, r)) = &mut self.lod_clip_next {
                *r = 0;
            }
            self.lod_clip_grow.set();
        }
        // A chart clips nothing (`lod_clip`), so its rings are never read: no scan.
        if !self.lod_clip_grow.take() || !self.fold.is_identity() {
            return;
        }
        let Some(center) = self.center else { return };
        let max_rings = self.view.horizontal + 1;
        while self.lod_clip_rings < max_rings && self.ring_settled(center, self.lod_clip_rings) {
            self.lod_clip_rings += 1;
        }
        if let Some((span, mut r)) = self.lod_clip_next {
            while r < max_rings && self.ring_settled_over(center, r, Some(span)) {
                r += 1;
            }
            self.lod_clip_next = Some((span, r));
            if r >= self.lod_clip_rings {
                (self.lod_clip_span, self.lod_clip_rings, self.lod_clip_next) = (Some(span), r, None);
            }
        }
    }

    /// Whether every column of chess-distance `ring` around `center` is fully
    /// settled across the proven span along the up axis. `None` (a cube)
    /// settles the 3-D chess shell instead of a column.
    pub(super) fn ring_settled(&self, center: Coord, ring: i32) -> bool {
        self.ring_settled_over(center, ring, self.lod_clip_span)
    }

    /// [`ring_settled`](Self::ring_settled) across chunk range `span` along the
    /// up axis, `None` being `center ± vertical`.
    fn ring_settled_over(&self, center: Coord, ring: i32, span: Option<[i32; 2]>) -> bool {
        match self.live_up() {
            Some(face) => self.column_ring_settled(center, ring, face, span),
            None => self.cube_ring_settled(center, ring),
        }
    }

    /// +Y is the XZ ring, each column spanning `span` (`center.y ± vertical`
    /// when `None`).
    fn column_ring_settled(&self, center: Coord, ring: i32, face: Face, span: Option<[i32; 2]>) -> bool {
        let v = self.view.vertical;
        let axis = face.axis();
        let (t0, t1) = match axis {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        let origin = [center.x, center.y, center.z];
        let [lo, hi] = span.unwrap_or([origin[axis] - v, origin[axis] + v]);
        let settled = |tu: i32, tv: i32| {
            (lo..=hi).all(|a| {
                let mut p = origin;
                p[t0] = tu;
                p[t1] = tv;
                p[axis] = a;
                // Through the chart net; storage that holds nothing is settled by definition.
                self.fold
                    .unfold(Coord::new(p[0], p[1], p[2]))
                    .is_none_or(|c| self.chunk_final(c))
            })
        };
        let c0 = origin[t0];
        let c1 = origin[t1];
        if ring == 0 {
            return settled(c0, c1);
        }
        let r = ring;
        (-r..=r).all(|d| settled(c0 + d, c1 - r) && settled(c0 + d, c1 + r))
            && (1 - r..r).all(|d| settled(c0 - r, c1 + d) && settled(c0 + r, c1 + d))
    }

    /// Whether chunk `c` is final for the far field's handover: settled (drawn, or nothing to
    /// draw), or parked by quarantine, a bounded hole that would otherwise hold the far field
    /// over it for good.
    pub(in crate::world) fn chunk_final(&self, c: Coord) -> bool {
        if self.chunks.get(&c).is_some_and(|l| l.state.settled()) {
            return true;
        }
        if self.quarantined.is_empty() {
            return false;
        }
        let parked = |key| self.quarantined.contains(&key);
        parked(streaming::FailKey::Mesh { coord: c })
            || parked(streaming::GenRun::of_chunk(c, self.generator.sky(c)).fail_key())
    }

    fn cube_ring_settled(&self, center: Coord, ring: i32) -> bool {
        let r = ring;
        for dx in -r..=r {
            for dy in -r..=r {
                for dz in -r..=r {
                    if dx.abs().max(dy.abs()).max(dz.abs()) != r {
                        continue;
                    }
                    let c = Coord::new(center.x + dx, center.y + dy, center.z + dz);
                    let Some(c) = self.fold.unfold(c) else { continue };
                    if !self.chunk_final(c) {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Far-material style: flat palette-average past [`FLAT_DETAIL`] if available,
    /// else textured. Returns `mode` bit and packed sRGB colour.
    pub(super) fn section_material(&self, pos: SectionPos) -> (bool, u32) {
        if pos.detail < FLAT_DETAIL {
            return (false, 0);
        }
        let color = self
            .section_overlay
            .get(&pos)
            .map(|c| c.color)
            .or_else(|| self.section_mip.as_ref().and_then(|m| m.color(pos)));
        match color {
            Some(c) => (
                true,
                c.r as u32 | (c.g as u32) << 8 | (c.b as u32) << 16 | (c.a as u32) << 24,
            ),
            None => (false, 0),
        }
    }

    /// Current vertical relief for one section footprint. Live edit-derived
    /// data takes precedence over the immutable background bake.
    pub(super) fn section_relief_band(&self, key: SectionPos) -> Option<(f32, f32)> {
        self.section_overlay
            .get(&key)
            .map(|cell| (cell.lo, cell.hi))
            .or_else(|| self.section_mip.as_ref()?.relief_band(key))
    }

    /// [`section_relief_band`](Self::section_relief_band), or the baked ancestor's envelope when
    /// `in_cube` and `key` is finer than the bake. The ancestor covers this tile, so a band that
    /// fits the window proves the tile does too. Off a cube the bake's own cell is the only proof,
    /// which keeps charts and flat worlds on the bands they had.
    pub(in crate::world) fn cover_band(&self, key: SectionPos, in_cube: bool) -> Option<(f32, f32)> {
        if let Some(band) = self.section_relief_band(key) {
            return Some(band);
        }
        if !in_cube {
            return None;
        }
        let mut p = key;
        for _ in 0..8 {
            if p.detail.0 >= crate::render_config::LOD_COARSEST_DETAIL as i8 {
                return None;
            }
            p = p.parent();
            if let Some(band) = self.section_relief_band(p) {
                return Some(band);
            }
        }
        None
    }

    /// Eye altitude in the bake's height space. A zero datum on +Y stores world Y;
    /// every other bake stores height above the face datum.
    fn baked_eye(&self, key: SectionPos) -> f32 {
        let datum = self.generator.face_datum(key.body, key.face);
        if key.face == Face::PosY && datum == 0 {
            self.section_eye_y as f32
        } else {
            (self.section_eye_y - f64::from(datum)) as f32
        }
    }

    /// Face-local altitude `a` in the bake's height space (see [`baked_eye`](Self::baked_eye)).
    pub(super) fn baked_height(&self, key: SectionPos, a: i64) -> f32 {
        let datum = self.generator.face_datum(key.body, key.face);
        if key.face == Face::PosY && datum == 0 { a as f32 } else { (a - i64::from(datum)) as f32 }
    }

    /// Geometric altitude of one baked height. +Y with a zero datum stores world Y;
    /// every other bake stores height above the face datum.
    pub(super) fn baked_world_y(&self, key: SectionPos, h: f32) -> i32 {
        let datum = self.generator.face_datum(key.body, key.face);
        let rel = h.floor() as i32;
        if key.face == Face::PosY && datum == 0 { rel } else { datum.saturating_add(rel) }
    }

    /// Skip near-field LOD load if the section's footprint is provably inside the
    /// coverage clip slab, so the shader discards it anyway. Only filters the load lane,
    /// not the desired set; selection stays isotropic.
    ///
    /// Skip only if all backing chunks are settled (drawable or born-air, never
    /// in-flight), to avoid holes during fast descent. A chart section the near
    /// window holds is not loaded either once something already draws it: that cover
    /// stays until its chunks have settled, then the punch drops it. Until a cover
    /// exists the section is a hole, and admission has to load it.
    pub(super) fn coverage_skips(&self, center: Coord, key: SectionPos) -> bool {
        (self.section_held.contains_key(&key) && self.section_covered(key)) || self.full_res_covers(center, key)
    }

    /// The full-res chunks draw everything `key` would: its footprint lies well inside the near
    /// square, its relief inside the near window, and every chunk under it has settled. Such a
    /// section is neither loaded nor drawn. The footprint is face-local on every cube face; off a
    /// cube the reference frame is the streaming centre, so a flat world and a chart stay as they
    /// were.
    pub(super) fn full_res_covers(&self, center: Coord, key: SectionPos) -> bool {
        let cov = self.view.coverage();
        let cs = CHUNK_SIZE as i32;
        let (mut h_lim, mut v_lim) = (0.75 * cov.half.x, 0.75 * cov.half.y);
        // A speed-reduced window is not covering the view. Only chunks it
        // actually loads can prove a section redundant; the rest of the view
        // keeps its far-field draw without a backing-chunk scan.
        if self.load_h >= 0 && self.load_h < self.view.horizontal {
            h_lim = 0.75 * (self.load_h.max(0) * cs) as f32;
        }
        if self.load_v >= 0 && self.load_v < self.view.vertical {
            v_lim = 0.75 * (self.load_v.max(0) * cs) as f32;
        }
        // The f64 eye tangents were floored to the centre chunk; inflate the reach
        // by one chunk half-diagonal so the true eye can't sit outside our bound.
        let margin = cs as f32 * 0.5 * std::f32::consts::SQRT_2;
        let (reference, eye_y, in_cube) = self.lod_place(center);
        let frame = FaceFrame::new(key.face);
        let mid = |c: i32| (i64::from(c) * i64::from(cs) + i64::from(cs / 2)) as i32;
        let (eu, _, ev) = frame.cell_to_local((mid(reference.x), mid(reference.y), mid(reference.z)));
        if streaming::span_reach(key, eu, ev).0 + margin > h_lim {
            return false;
        }
        // Vertical: the section's terrain must sit inside the near window (the eye's
        // slab without one). If unbaked, can't prove, so don't skip. An edited footprint
        // prefers the fresh overlay over the (possibly stale) bake. Window altitudes are
        // storage-frame; the bake is reference-frame.
        let Some((lo, hi)) = self.cover_band(key, in_cube) else {
            return false;
        };
        let (floor, ceil) = match self.window_alts() {
            Some([a0, a1]) if self.live_up() == Some(key.face) => {
                let shift = if in_cube { Self::face_alt_shift(center, reference, key.face) } else { 0 };
                (self.baked_height(key, a0 + shift), self.baked_height(key, a1 + shift))
            }
            _ if in_cube => {
                let eye = DVec3::new(f64::from(mid(reference.x)), eye_y, f64::from(mid(reference.z)));
                let altitude = frame.point_to_local(eye).y;
                let datum = self.generator.face_datum(key.body, key.face);
                let ey = if key.face == Face::PosY && datum == 0 {
                    altitude as f32
                } else {
                    (altitude - f64::from(datum)) as f32
                };
                (ey - v_lim, ey + v_lim)
            }
            _ => {
                let ey = self.baked_eye(key);
                (ey - v_lim, ey + v_lim)
            }
        };
        if lo < floor || hi > ceil {
            return false;
        }
        // Every chunk backing the footprint is settled, so the near area is
        // actually covered now, not just in-range.
        self.backing_chunks_ready(center, reference, in_cube, key)
    }

    /// Whether every full-res chunk backing `key`'s footprint is settled
    /// (drawable or born-air, never in-flight). Used by [`Self::coverage_skips`]
    /// to skip the near-LOD load only where full-res provably covers. A footprint
    /// whose vertical band can't be proven (no overlay, no baked mip) reads as
    /// NOT ready — fail toward keeping the section loaded, never toward a hole.
    /// Chunks of a warped cube are loaded in its storage box; `key` is reference-frame.
    fn backing_chunks_ready(&self, storage: Coord, reference: Coord, in_cube: bool, key: SectionPos) -> bool {
        let Some((lo, hi)) = self.cover_band(key, in_cube) else {
            return false;
        };
        let frame = FaceFrame::new(key.face);
        let span = key.span();
        let (u0, v0) = (key.min_x(), key.min_z());
        let (u1, v1) = (u0 + span - 1, v0 + span - 1);
        let (a_lo, a_hi) = (self.baked_world_y(key, lo), self.baked_world_y(key, hi));
        let mut lo_c = [i32::MAX; 3];
        let mut hi_c = [i32::MIN; 3];
        for a in [a_lo, a_hi] {
            for u in [u0, u1] {
                for v in [v0, v1] {
                    let (x, y, z) = frame.cell_to_world((u, a, v));
                    let c = Self::chunk_of(x, y, z);
                    let p = [c.x, c.y, c.z];
                    for i in 0..3 {
                        lo_c[i] = lo_c[i].min(p[i]);
                        hi_c[i] = hi_c[i].max(p[i]);
                    }
                }
            }
        }
        let (dx, dy, dz) = if in_cube {
            (
                i64::from(reference.x) - i64::from(storage.x),
                i64::from(reference.y) - i64::from(storage.y),
                i64::from(reference.z) - i64::from(storage.z),
            )
        } else {
            (0, 0, 0)
        };
        for y in lo_c[1]..=hi_c[1] {
            for z in lo_c[2]..=hi_c[2] {
                for x in lo_c[0]..=hi_c[0] {
                    let c = Coord::new((i64::from(x) - dx) as i32, (i64::from(y) - dy) as i32, (i64::from(z) - dz) as i32);
                    if !self.chunk_final(c) {
                        return false;
                    }
                }
            }
        }
        true
    }
}
