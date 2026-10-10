//! The window clip. A finest-ring tile that crosses the full-res window stays whole, meshed
//! without the chunk columns the window draws ([`ColumnClip`]). A column joins a tile's clip once
//! its ground fits the window and its chunks are final; a column the window gives up leaves the
//! clip at once. A clip change re-extracts the tile while its old mesh keeps drawing, and the
//! chunks of a column a landed mesh still leaves out stay resident until it is replaced.

use super::*;
use super::super::section::ColumnClip;

/// The far lane's clip state.
#[derive(Default)]
pub(in crate::world) struct Clips {
    /// The clip each tile should draw now.
    target: FastMap<SectionPos, ColumnClip>,
    /// Tiles whose wanted clip still has unsettled columns: that clip and its chunk layers.
    wait: FastMap<SectionPos, (ColumnClip, [i32; 2])>,
    /// Ready tiles re-extracting in place, by claim token.
    reclip: FastMap<SectionPos, pipeline::ClaimToken>,
    /// Ready tiles extracted with a clip. Entries that left that state are dropped on use.
    clipped: FastSet<SectionPos>,
    /// Chunks past the unload box kept for a column a landed clip still leaves to them.
    retained: Vec<Coord>,
}

impl World {
    /// Chunk layers of storage rect `rect`'s solid tops, when they all lie in `[y0, y1)`. Storage
    /// altitude: `surface` is the first open cell, so the solid top is the block below it.
    pub(super) fn fit_layers(&self, body: u16, rect: [i32; 4], y0: i64, y1: i64, memo: &mut NearBounds) -> Option<[i32; 2]> {
        let [u0, v0, u1, v1] = rect;
        let surface = || self.generator.surface_rect(body, Face::PosY, u0, v0, u1, v1);
        let (lo, hi) = self.with_edits(memo.get((body, rect), surface)?, rect);
        let (top_lo, top_hi) = (lo.saturating_sub(1), hi.saturating_sub(1));
        let cs = CHUNK_SIZE as i32;
        (i64::from(top_lo) >= y0 && i64::from(top_hi) < y1).then(|| [top_lo.div_euclid(cs), top_hi.div_euclid(cs)])
    }

    /// The columns of tile `s` the window can draw, with their chunk layers: its overlap with the
    /// near square, whole when that ground fits the window, else column by column. `None` when no
    /// column fits, or the overlap does not fall on chunk columns (a seam's unfold).
    pub(super) fn clip_columns(
        &self,
        s: SectionPos,
        near: (i64, i64, i64, i64),
        y0: i64,
        y1: i64,
        across: Option<&super::super::seam::SeamAcross>,
        memo: &mut NearBounds,
    ) -> Option<(ColumnClip, [i32; 2])> {
        let (u0, v0, u1, v1) = overlap_storage(s, near, across)?;
        let cs = CHUNK_SIZE as i32;
        if [u0, v0, u1, v1].iter().any(|e| e.rem_euclid(cs) != 0) {
            return None;
        }
        if let Some(layers) = self.fit_layers(s.body, [u0, v0, u1, v1], y0, y1, memo) {
            return Some((ColumnClip::of_rect(s, [u0, v0, u1, v1]), layers));
        }
        let (mut clip, mut hull) = (ColumnClip::NONE, None::<[i32; 2]>);
        for v in (v0..v1).step_by(cs as usize) {
            for u in (u0..u1).step_by(cs as usize) {
                let rect = [u, v, u + cs, v + cs];
                let Some(l) = self.fit_layers(s.body, rect, y0, y1, memo) else { continue };
                clip = clip.union(ColumnClip::of_rect(s, rect));
                hull = Some(hull.map_or(l, |h| [h[0].min(l[0]), h[1].max(l[1])]));
            }
        }
        hull.map(|h| (clip, h))
    }

    /// The clip tile `s` should draw now.
    pub(in crate::world) fn clip_target(&self, s: SectionPos) -> ColumnClip {
        self.clips.target.get(&s).copied().unwrap_or_default()
    }

    /// The columns of `mask` whose chunks in `layers` are all final.
    fn settled_columns(&self, s: SectionPos, mask: ColumnClip, layers: [i32; 2]) -> ColumnClip {
        let cs = CHUNK_SIZE as i32;
        let (x0, z0) = (s.min_x().div_euclid(cs), s.min_z().div_euclid(cs));
        let settled = |&(x, z): &(i32, i32)| (layers[0]..=layers[1]).rev().all(|y| self.chunk_final(Coord::new(x, y, z)));
        ColumnClip(mask.columns(s).filter(settled).fold(0, |m, (x, z)| m | 1 << ((z - z0) * 8 + (x - x0))))
    }

    /// Take the clips the selection wants. A tile keeps the part of its clip still wanted and
    /// adds the wanted columns whose chunks are final; the rest wait for them.
    pub(super) fn wait_clips(&mut self, wanted: Vec<(SectionPos, ColumnClip, [i32; 2])>) {
        let prev = std::mem::take(&mut self.clips.target);
        self.clips.wait.clear();
        for (s, want, layers) in wanted {
            let had = prev.get(&s).copied().unwrap_or_default().intersect(want);
            let now = had.union(self.settled_columns(s, want.without(had), layers));
            if now != want {
                self.clips.wait.insert(s, (want, layers));
            }
            if !now.is_empty() {
                self.clips.target.insert(s, now);
            }
        }
        if prev != self.clips.target {
            self.pending_sections.set();
        }
    }

    /// Add the waiting columns whose chunks have settled since.
    pub(super) fn settle_clips(&mut self) {
        if self.clips.wait.is_empty() {
            return;
        }
        let waits: Vec<_> = self.clips.wait.iter().map(|(&s, &w)| (s, w)).collect();
        let mut grew = false;
        for (s, (want, layers)) in waits {
            let had = self.clip_target(s);
            let now = had.union(self.settled_columns(s, want.without(had), layers));
            if now != had {
                self.clips.target.insert(s, now);
                grew = true;
            }
            if now == want {
                self.clips.wait.remove(&s);
            }
        }
        if grew {
            self.pending_sections.set();
        }
    }

    /// Every waiting clip as if its chunks had settled.
    #[cfg(test)]
    pub(in crate::world) fn settle_clip_waits(&mut self) {
        for (s, (want, _)) in std::mem::take(&mut self.clips.wait) {
            self.clips.target.insert(s, want);
        }
    }

    /// The selection once every held section's ground has settled, and the clip each tile wants.
    #[cfg(test)]
    pub(in crate::world) fn desired_and_clips(&self, center: Coord) -> (Vec<SectionPos>, FastMap<SectionPos, ColumnClip>) {
        let mut clips = Vec::new();
        let desired = self.desired_sections_with(center, &mut NearBounds::default(), &mut Vec::new(), &mut clips);
        (desired, clips.into_iter().map(|(s, c, _)| (s, c)).collect())
    }

    /// Every clip settled and landed: nothing waits on chunks or is re-extracting, and every drawn
    /// tile draws its target.
    #[cfg(test)]
    pub(in crate::world) fn clips_at_target(&self) -> bool {
        self.clips.wait.is_empty()
            && self.clips.reclip.is_empty()
            && self.section_visible.iter().all(|&(s, m)| m.is_empty() || !self.sections.get(&s).is_some_and(|st| st.is_ready()) || !self.needs_reclip(s))
    }

    /// Some tile whose clip waits on chunks: the clip it wants and their layers.
    #[cfg(test)]
    pub(in crate::world) fn clip_wait(&self) -> Option<(SectionPos, ColumnClip, [i32; 2])> {
        self.clips.wait.iter().map(|(&s, &(c, l))| (s, c, l)).min_by_key(|(s, _, _)| (s.x, s.z))
    }

    /// Whether a clip still waits on chunks.
    pub(super) fn clips_waiting(&self) -> bool {
        !self.clips.wait.is_empty()
    }

    /// Whether a landed clip leaves out a column the window no longer draws: the window keeps
    /// what it holds until the re-extracts have landed. Only drawn tiles count.
    pub(super) fn clips_landed(&self) -> bool {
        self.clips.clipped.iter().all(|&s| match self.sections.get(&s) {
            Some(SectionState::Ready { clip, .. }) => {
                clip.drawn.is_subset_of(self.clip_target(s)) || self.section_fade.drawn_mask(s).is_none()
            }
            _ => true,
        })
    }

    /// A Ready tile whose mesh is not for its current clip, or that an edit dirtied (once its
    /// overlay is in), with no re-extract in flight.
    pub(in crate::world) fn needs_reclip(&self, s: SectionPos) -> bool {
        let Some(SectionState::Ready { clip, .. }) = self.sections.get(&s) else { return false };
        let target = self.clip_target(s);
        let dirty = !clip.asked.union(target).is_empty() && self.dirty_sections.contains(&s) && !self.section_overlay_dirty.contains(&s);
        (clip.asked != target || dirty) && !self.clips.reclip.contains_key(&s)
    }

    /// Whether `s` draws now or is about to: drawn, or selected.
    pub(in crate::world) fn in_cut(&self, s: SectionPos) -> bool {
        self.section_fade.drawn_mask(s).is_some() || self.section_desired.binary_search_by_key(&section_key(&s), section_key).is_ok()
    }

    /// Visit the tiles in the cut to re-extract in place.
    pub(in crate::world) fn for_each_reclip(&self, mut visit: impl FnMut(SectionPos)) {
        let target = self.clips.target.keys().filter(|s| !self.clips.clipped.contains(s));
        for &s in self.clips.clipped.iter().chain(target) {
            if self.needs_reclip(s) && self.in_cut(s) {
                visit(s);
            }
        }
    }

    /// Record (`Some`) or drop the in-place re-extract claim on `s`.
    pub(in crate::world) fn claim_reclip(&mut self, s: SectionPos, token: Option<pipeline::ClaimToken>) {
        match token {
            Some(t) => {
                self.clips.reclip.insert(s, t);
            }
            None => {
                self.clips.reclip.remove(&s);
            }
        }
    }

    /// A cancelled or failed job releases the re-extract claim it held.
    pub(super) fn release_reclip(&mut self, s: SectionPos, token: pipeline::ClaimToken) {
        if self.clips.reclip.get(&s) == Some(&token) {
            self.clips.reclip.remove(&s);
            self.pending_sections.set();
        }
    }

    pub(in crate::world) fn reclips_in_flight(&self) -> bool {
        !self.clips.reclip.is_empty()
    }

    /// Whether `token` is the live claim on `s`: a fresh extract or an in-place one.
    pub(in crate::world) fn section_claim_live(&self, s: SectionPos, token: pipeline::ClaimToken) -> bool {
        matches!(self.sections.get(&s), Some(SectionState::Meshing { token: t }) if *t == token)
            || self.clips.reclip.get(&s) == Some(&token)
    }

    /// Forget the lane's clips with the lane. Retained chunks go at the next unload pass.
    pub(in crate::world) fn clear_clips(&mut self) {
        let retained = std::mem::take(&mut self.clips.retained);
        self.clips = Clips { retained, ..Clips::default() };
    }

    /// Land one finished section job. A fresh mesh installs as a new Ready section; an in-place
    /// one swaps in for the mesh it replaces, which kept drawing until now. A result whose clip
    /// leaves out a column neither its target nor the mesh it replaces does is dropped: a fresh
    /// claim is released, an in-place one is asked again. Returns whether it installed.
    pub(in crate::world) fn land_section(&mut self, up: pipeline::SectionUpload, mut eng: Option<&mut Engine>) -> bool {
        let pipeline::SectionUpload { pos, token, clip: asked, meshes, .. } = up;
        let reclip = self.clips.reclip.get(&pos) == Some(&token);
        if reclip {
            self.clips.reclip.remove(&pos);
        }
        let landed = match self.sections.get(&pos) {
            Some(SectionState::Meshing { token: t }) if *t == token => Some(ColumnClip::NONE),
            Some(SectionState::Ready { clip, .. }) if reclip => Some(clip.drawn),
            _ => None,
        };
        let drawn = meshes.clip_drawn();
        if !landed.is_some_and(|l| drawn.is_subset_of(self.clip_target(pos).union(l))) {
            if let Some(e) = eng {
                meshes.release_staging(e);
            }
            if landed.is_some() && !reclip {
                self.sections.remove(&pos);
                super::super::adjust_count(&mut self.meshing_sections, true, false);
                self.section_cover_dirty.set();
            }
            self.pending_sections.set();
            return false;
        }
        let state = match eng.as_deref_mut() {
            Some(e) => {
                let bend = self.chart_bend(pos);
                SectionState::from_upload_payload(pos, meshes, asked, e, bend.as_ref())
            }
            None => SectionState::Ready {
                meshes: Vec::new(),
                cages: Vec::new(),
                last_style: None,
                alt: None,
                clip: super::super::SectionClip { asked, drawn },
            },
        };
        if !reclip {
            super::super::adjust_count(&mut self.meshing_sections, true, false);
        }
        // `section_material` borrows all of `self`: read it before the entry.
        let style = eng.is_some().then(|| self.section_material(pos));
        let old = self.sections.insert(pos, state);
        if let (Some(e), Some((flat_color, flat_rgba))) = (eng.as_deref_mut(), style) {
            if let Some(old) = old {
                old.free(e);
            }
            let mask = self.section_fade.drawn_mask(pos);
            let state = self.sections.get_mut(&pos).expect("just installed");
            // Slots are born visible: a section that Coverage does not draw, or draws only in
            // part, is corrected at the transition that gave it slots.
            state.set_visible(e, mask);
            state.push_style(e, FadeStyle { flat_color }, flat_rgba);
        }
        if !asked.is_empty() {
            self.clips.clipped.insert(pos);
        }
        // A new Ready section moves the covering, and a new clip the morph cut: re-arm the lane
        // so any refinement or later clip it exposes loads immediately.
        self.pending_sections.set();
        self.section_cover_dirty.set();
        if let (true, Some(e)) = (reclip, eng) {
            self.release_clip_retained(e);
        }
        true
    }

    /// Chunk columns some landed clip leaves out, in the chunks' own storage frame.
    fn landed_columns(&mut self) -> FastSet<(i32, i32)> {
        let sections = &self.sections;
        self.clips.clipped.retain(|s| matches!(sections.get(s), Some(SectionState::Ready { clip, .. }) if !clip.asked.is_empty()));
        let mut out = FastSet::default();
        for s in &self.clips.clipped {
            if let Some(SectionState::Ready { clip, .. }) = sections.get(s) {
                out.extend(clip.drawn.columns(*s));
            }
        }
        out
    }

    /// Of the chunks `far` leaving the unload box, keep those whose column a landed clip leaves
    /// to them; add to `far` the retained ones no clip needs past the box any more.
    pub(super) fn retain_clipped(&mut self, unload: ChunkBox, far: &mut Vec<Coord>) {
        if self.clips.retained.is_empty() && self.clips.clipped.is_empty() {
            return;
        }
        let masked = self.landed_columns();
        let mut retained = std::mem::take(&mut self.clips.retained);
        retained.retain(|&c| {
            let keep = masked.contains(&(c.x, c.z));
            if !keep && !self.view_contains(unload, c) && self.chunks.contains_key(&c) {
                far.push(c);
            }
            keep
        });
        far.retain(|&c| {
            let keep = masked.contains(&(c.x, c.z)) && !retained.contains(&c);
            if keep {
                retained.push(c);
            }
            !keep && !retained.contains(&c)
        });
        self.clips.retained = retained;
    }

    /// Free the retained chunks no landed clip needs any more.
    fn release_clip_retained(&mut self, eng: &mut Engine) {
        let Some(center) = self.center.filter(|_| !self.clips.retained.is_empty()) else { return };
        let mut far = Vec::new();
        self.retain_clipped(self.unload_box(center), &mut far);
        self.drop_chunks(&far, |state, cage| {
            state.free_owned(eng);
            if let Some(cage) = cage {
                eng.free_cage(cage);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::morph::{OWNER_EYE, OWNER_SEED, owner_world};

    /// Over every chunk-centre eye of the owner's 128-block cell, with every clip settled, each
    /// chunk column of the near square and the 128 blocks around it is drawn exactly once: by
    /// the window where its ground fits, else by one section that does not clip it. Finer slivers
    /// appear only where no column of a crossing tile fits.
    #[test]
    fn window_partition_holds_over_the_owner_cell() {
        let cs = CHUNK_SIZE as i64;
        for seed in [42, OWNER_SEED] {
            let mut world = owner_world(seed);
            let base = (OWNER_EYE / 128.0).floor() * 128.0;
            let (mut slivers, mut clipped) = (0usize, 0usize);
            for i in 0..64 {
                let eye = DVec3::new(base.x + (16 * (i % 8) + 8) as f64, OWNER_EYE.y, base.z + (16 * (i / 8) + 8) as f64);
                // Each eye stands still: no travel shrinks the loading window.
                world.near_eye_prev = None;
                let (_, center, _, _) = world.begin_stream(eye, None);
                let (mut held, mut clips) = (Vec::new(), Vec::new());
                let mut memo = NearBounds::default();
                let desired = world.desired_sections_with(center, &mut memo, &mut held, &mut clips);
                let seat = world.seams.chart_seat(center).expect("the owner cell is charted");
                let body = crate::world::section::CHART_BODY_BASE + seat.index as u16;
                let clip_of: FastMap<SectionPos, ColumnClip> = clips.iter().map(|&(s, c, _)| (s, c)).collect();
                let finest = world.chart_pyramid(seat.radius).expect("pyramid").0.finest;
                slivers += desired.iter().filter(|s| s.detail < finest).count();
                clipped += clip_of.len();
                let near = world.near_block_box(center);
                let (y0, y1) = world.near_y_range(center);
                let mut x = near.0 - 128;
                while x < near.1 + 128 {
                    let mut z = near.2 - 128;
                    while z < near.3 + 128 {
                        let (px, pz) = (x + cs / 2, z + cs / 2);
                        if px < seat.lo[0] || px >= seat.hi[0] || pz < seat.lo[2] || pz >= seat.hi[2] {
                            z += cs;
                            continue;
                        }
                        let inside = px >= near.0 && px < near.1 && pz >= near.2 && pz < near.3;
                        let rect = [x as i32, z as i32, (x + cs) as i32, (z + cs) as i32];
                        let window = inside && world.fit_layers(body, rect, y0, y1, &mut memo).is_some();
                        let drawn: Vec<SectionPos> = desired
                            .iter()
                            .filter(|s| {
                                let (sx, sz, w) = (i64::from(s.min_x()), i64::from(s.min_z()), i64::from(s.span()));
                                px >= sx && px < sx + w && pz >= sz && pz < sz + w && !clip_of.get(s).is_some_and(|c| c.masks_point(**s, px, pz))
                            })
                            .copied()
                            .collect();
                        assert_eq!(
                            drawn.len() + usize::from(window),
                            1,
                            "seed {seed} eye {eye:?}: column ({x}, {z}) drawn by {drawn:?}, window {window}"
                        );
                        z += cs;
                    }
                    x += cs;
                }
                if (eye.x - OWNER_EYE.x).abs() < 8.0 && (eye.z - OWNER_EYE.z).abs() < 8.0 {
                    assert_eq!(desired.iter().filter(|s| s.detail < finest).count(), 0, "seed {seed}: slivers at the owner eye");
                }
            }
            println!("seed {seed}: {clipped} clipped tiles and {slivers} slivers over 64 eyes");
        }
    }

    /// A Ready tile at the chart's origin with clip `drawn` (asked for and wanted), drawn whole.
    fn clipped_tile(world: &mut World, drawn: ColumnClip) -> SectionPos {
        use crate::world::section::{CHART_BODY_BASE, FINEST_DETAIL};
        let pos = SectionPos { body: CHART_BODY_BASE, face: Face::PosY, detail: FINEST_DETAIL, x: 0, z: 0 };
        let clip = crate::world::SectionClip { asked: drawn, drawn };
        world.sections.insert(pos, SectionState::Ready { meshes: Vec::new(), cages: Vec::new(), last_style: None, alt: None, clip });
        world.clips.clipped.insert(pos);
        if !drawn.is_empty() {
            world.clips.target.insert(pos, drawn);
        }
        world.section_fade.update_now(&[(pos, crate::world::quadtree::QuadrantMask::ALL)]);
        pos
    }

    /// Submit and claim the lane's job for `pos`; returns its claim and asked clip.
    fn submit(world: &mut World, pos: SectionPos) -> (pipeline::ClaimToken, ColumnClip) {
        let Some(pipeline::Job::Section { token, clip, .. }) = <SectionLane as StreamLane>::submit(world, pos) else {
            panic!("no section job");
        };
        <SectionLane as StreamLane>::claim(world, pos);
        (token, clip)
    }

    /// Integrate a finished job for `pos` whose mesh leaves out `drawn`; returns its queued upload.
    fn finish(world: &mut World, pos: SectionPos, token: pipeline::ClaimToken, asked: ColumnClip, drawn: ColumnClip) -> pipeline::SectionUpload {
        let meshes = crate::world::section::SectionMeshData { clip_drawn: drawn, ..Default::default() };
        let done = pipeline::Done::Section { pos, epoch: world.section_epoch, token, clip: asked, meshes: meshes.into() };
        <SectionLane as StreamLane>::integrate(world, done);
        world.section_upload_queue.pop_front().expect("queued")
    }

    fn drawn_clip(world: &World, pos: SectionPos) -> Option<crate::world::SectionClip> {
        match world.sections.get(&pos) {
            Some(SectionState::Ready { clip, .. }) => Some(*clip),
            _ => None,
        }
    }

    /// A clip change re-extracts a drawn tile in place: it stays Ready, its mesh keeps drawing
    /// until the new one lands, and the landing swaps it. A result that leaves out a column
    /// neither the target nor the landed mesh does is dropped and asked again; cancel, failure and
    /// a cleared lane release the claim.
    #[test]
    fn a_clip_change_replaces_the_mesh_on_upload() {
        let mut world = owner_world(42);
        let center = Coord::new(0, 0, 0);
        let (wide, narrow) = (ColumnClip(0x0F), ColumnClip(0x03));
        let pos = clipped_tile(&mut world, wide);
        assert!(<SectionLane as StreamLane>::in_flight(&world, pos), "a tile at its target is done");
        world.clips.target.insert(pos, narrow);
        assert!(!<SectionLane as StreamLane>::in_flight(&world, pos));
        let mut visited = Vec::new();
        <SectionLane as StreamLane>::for_each_geometry(&world, center, &mut Vec::new(), |s| visited.push(s));
        assert_eq!(visited, vec![pos], "the lane visits the tile to re-extract");
        let (token, asked) = submit(&mut world, pos);
        assert_eq!(asked, narrow, "the job asks for the target");
        assert_eq!(drawn_clip(&world, pos).map(|c| c.drawn), Some(wide), "the old mesh keeps drawing");
        assert!(<SectionLane as StreamLane>::in_flight(&world, pos) && world.anything_in_flight());
        let up = finish(&mut world, pos, token, narrow, narrow);
        assert!(world.land_section(up, None), "the re-extract lands");
        assert_eq!(drawn_clip(&world, pos), Some(crate::world::SectionClip { asked: narrow, drawn: narrow }));
        assert!(!world.reclips_in_flight() && <SectionLane as StreamLane>::in_flight(&world, pos));

        // The window gives up another column while a result that still leaves it out is in flight.
        world.clips.target.insert(pos, ColumnClip(0x01));
        let (token, _) = submit(&mut world, pos);
        let up = finish(&mut world, pos, token, ColumnClip(0x01), ColumnClip(0x07));
        world.pending_sections.take();
        assert!(!world.land_section(up, None), "a result leaving out a column neither the target nor the old mesh does is dropped");
        assert_eq!(drawn_clip(&world, pos).map(|c| c.drawn), Some(narrow), "the old mesh stays");
        assert!(world.pending_sections.get() && world.needs_reclip(pos), "and the tile is asked again");

        // Cancel and failure release the in-place claim; the tile stays Ready.
        for fail in [false, true] {
            let (token, _) = submit(&mut world, pos);
            let key = pipeline::JobKey::Section { pos, epoch: world.section_epoch, token };
            if fail {
                world.fail_job(key);
            } else {
                world.cancel_job(key);
            }
            assert!(!world.reclips_in_flight() && world.needs_reclip(pos), "fail {fail}: the claim is released");
            assert!(drawn_clip(&world, pos).is_some(), "fail {fail}: the tile keeps drawing");
        }
        submit(&mut world, pos);
        world.clear_clips();
        assert!(!world.reclips_in_flight(), "a cleared lane drops the claim");
    }

    /// A fresh tile installs only a clip its target allows: one past it is released.
    #[test]
    fn a_fresh_tile_lands_only_inside_its_target() {
        let mut world = owner_world(42);
        let pos = clipped_tile(&mut world, ColumnClip::NONE);
        world.sections.remove(&pos);
        world.clips.target.insert(pos, ColumnClip(0x03));
        let (token, asked) = submit(&mut world, pos);
        assert!(matches!(world.sections.get(&pos), Some(SectionState::Meshing { .. })) && asked == ColumnClip(0x03));
        world.clips.target.insert(pos, ColumnClip(0x01));
        let up = finish(&mut world, pos, token, asked, asked);
        assert!(!world.land_section(up, None));
        assert!(!world.sections.contains_key(&pos) && world.meshing_sections == 0, "the claim is released");
        let (token, asked) = submit(&mut world, pos);
        let up = finish(&mut world, pos, token, asked, asked);
        assert!(world.land_section(up, None));
        assert_eq!(drawn_clip(&world, pos).map(|c| c.drawn), Some(ColumnClip(0x01)));
    }

    /// Chunks leaving the unload box stay while a landed clip leaves their column to them, and go
    /// at the next pass once no landed clip does.
    #[test]
    fn clipped_columns_keep_their_chunks_until_the_clip_lands() {
        let mut world = World::generate();
        let kept = Coord::new(0, 0, 0);
        assert!(world.chunks.contains_key(&kept), "the origin chunk is loaded");
        let pos = clipped_tile(&mut world, ColumnClip(0x01));
        let other = Coord::new(1, 0, 0);
        let away = world.unload_box(Coord::new(10_000, 0, 10_000));
        let mut far = vec![kept, other];
        world.retain_clipped(away, &mut far);
        assert_eq!(far, vec![other], "the clipped column's chunk is retained");
        let mut again = Vec::new();
        world.retain_clipped(away, &mut again);
        assert!(again.is_empty(), "still retained while the clip stands");
        if let Some(SectionState::Ready { clip, .. }) = world.sections.get_mut(&pos) {
            clip.drawn = ColumnClip::NONE;
        }
        world.retain_clipped(away, &mut again);
        assert_eq!(again, vec![kept], "released once no landed clip leaves it out");
    }
}
