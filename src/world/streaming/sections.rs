//! Far-field sections: the relief bake, the edit overlay, residency and the visible covering.

use super::*;

impl World {
    /// Spawn background max-mip bake around the eye's face. Re-bakes when the eye
    /// leaves the inner half of the baked square, or the body/face changes.
    /// A still camera (anchor held, bake landed or in flight) allocates nothing.
    pub(in crate::world) fn ensure_mip_bake(&mut self) {
        if self.section_face_set && self.section_lod_face.is_none() {
            return;
        }
        let (body, face) = self.section_lod_face.unwrap_or((0, Face::PosY));
        let (au, av) = match (self.section_face_set, self.center) {
            (true, Some(c)) => self.face_tangent_centre(c, face),
            _ => (0, 0),
        };
        let cfg = &self.section_pyramid;
        let extent = BakeExtent::new(cfg.outer_m() as i32, cfg.coarsest());
        let half = extent.half_m() / 2;
        let face_changed = self.section_mip_anchor.is_some_and(|(b, f, _, _)| b != body || f != face);
        if face_changed {
            self.section_mip = None;
            self.section_mip_rx = None;
            self.section_frontier_key = None;
        }
        let moved = self.section_mip_anchor.is_some_and(|(_, _, u, v)| (au - u).abs() > half || (av - v).abs() > half);
        if !face_changed && !moved && (self.section_mip.is_some() || self.section_mip_rx.is_some()) {
            return;
        }
        if self.section_mip_rx.is_some() {
            return;
        }
        self.section_mip_anchor = Some((body, face, au, av));
        let generator = self.generator.clone();
        // The generator stores resolved IDs for every element-worldgen
        // composition registered during `World::new`. A fresh builtin registry
        // is too short for those IDs; snapshot the matching color table instead.
        let colors = self.registry.color_snapshot();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(HeightMip::bake_at(&*generator, &colors, extent, au, av, face, body));
        });
        self.section_mip_rx = Some(rx);
    }

    /// Install background bake if landed. Newly-arrived mip only coarsens
    /// far field, so just re-arm section pass to re-select.
    pub(in crate::world) fn poll_mip(&mut self) {
        if let Some(rx) = &self.section_mip_rx
            && let Ok(mip) = rx.try_recv()
        {
            self.section_mip = Some(mip);
            self.section_mip_rx = None;
            self.pending_sections.set();
            // Off a chart the window's ground comes from this bake.
            self.window.stale = true;
        }
    }

    /// Re-derive the edit-folded cell for every section touched by a live edit,
    /// fixing the immutable bake's edit-staleness (a mined-out feature would
    /// otherwise keep occluding/colouring/measuring error as if still solid).
    /// The `&mut` sync point `render`'s `&self` readers may never recompute
    /// (occlusion-class derived state is rebuilt here, not in render).
    ///
    /// Cost is bounded by `section_edit_rev`'s size (sections an edit has EVER
    /// touched), not by view distance or total edit count: untouched cells never
    /// enter the loop, so an unedited world pays nothing (`section_overlay` stays
    /// empty and every reader falls back to the pure bake, bit-identical to
    /// before this cache existed).
    /// A quiet frame is one set-emptiness check: only the exact positions
    /// edits touched since the last refresh (`section_overlay_dirty`) are
    /// re-derived, and the resolved map is maintained incrementally instead
    /// of cleared and rebuilt every pass.
    ///
    /// Each position costs about one far section's extract, and one edit dirties a position per
    /// active detail, so the work is spread over frames: at least one position per pass, more while
    /// `budget` lasts. [`Self::remesh_dirty_sections`] holds a square back until its overlay is in.
    pub(in crate::world) fn refresh_section_overlay(&mut self, budget: Budget) -> Progress {
        if self.section_overlay_dirty.is_empty() {
            return Progress::Idle;
        }
        let deadline = super::lanes::paced_deadline(self, budget);
        let positions: Vec<SectionPos> = self.section_overlay_dirty.iter().copied().collect();
        for (done, pos) in positions.into_iter().enumerate() {
            if done > 0 && deadline.expired() {
                break;
            }
            self.section_overlay_dirty.remove(&pos);
            let rev = voxel_engine::Rev(self.section_edit_rev.get(&pos).copied().unwrap_or(0));
            let touched = self.edits_for_section(pos);
            if touched.is_empty() {
                // Reverted back to what the generator would produce (overlay
                // compaction, edits.rs): no override, the pure bake applies.
                self.section_overlay.remove(&pos);
                continue;
            }
            let cell = *self.section_overlay_cache.get_or_recompute(pos, rev, || {
                let colors = self.registry.color_snapshot();
                super::heightmip::resample_cell(pos, &*self.generator, &touched, &colors)
            });
            match cell {
                Some(cell) => {
                    self.section_overlay.insert(pos, cell);
                }
                None => {
                    self.section_overlay.remove(&pos);
                }
            }
            // Off a chart the window's ground is read from this overlay.
            self.window.stale = true;
        }
        match self.section_overlay_dirty.len() {
            0 => Progress::Idle,
            n => Progress::Partial { remaining: n as u32 },
        }
    }

    /// The atlas patch a chart section is bent through. `None` for a cube section or a square
    /// that is not inside a storage box.
    pub(super) fn chart_bend(&self, pos: SectionPos) -> Option<super::ChartBend> {
        if pos.body < super::section::CHART_BODY_BASE {
            let atlas = self.seams.atlases().iter().find(|a| {
                a.grid.as_ref().is_some_and(|g| g.body == pos.body) && a.warp.is_some()
            })?;
            return Some(super::ChartBend { atlas: atlas.clone(), patch: crate::space::atlas::Patch::Grid });
        }
        let index = (pos.body - super::section::CHART_BODY_BASE) as usize;
        let atlas = self.seams.atlases().get(index)?.clone();
        let (patch, _) = atlas.locate([pos.min_x() as i64, 0, pos.min_z() as i64])?;
        Some(super::ChartBend { atlas, patch })
    }

    /// Ready tiles under an uncovered desired cell, once the section floor is full.
    /// Zero below the floor, so a quiet or under-budget admission does not scan.
    pub(in crate::world) fn count_section_standins(&self) -> usize {
        if self.section_budget_used() < self.sections_allowed() {
            return 0;
        }
        let desired: FastSet<SectionPos> = self.section_desired.iter().copied().collect();
        self.sections
            .iter()
            .filter(|&(&s, st)| st.is_ready() && self.stands_under_hole(s, &desired))
            .count()
    }

    /// `s` is a Ready tile strictly under a desired cell that nothing Ready draws yet.
    /// Unload and reclaim would otherwise drop it in the frame before the visible set
    /// can stand it back in.
    pub(super) fn stands_under_hole(&self, s: SectionPos, desired: &FastSet<SectionPos>) -> bool {
        if !self.sections.get(&s).is_some_and(|st| st.is_ready()) {
            return false;
        }
        let max = crate::ident::Detail(crate::render_config::LOD_COARSEST_DETAIL as i8);
        let mut p = s;
        while p.detail < max {
            p = p.parent();
            if desired.contains(&p) && !self.section_covered(p) {
                return true;
            }
        }
        false
    }

    /// Ready sections strictly under a drawn cell that has no Ready self or ancestor.
    /// [`resolve_covering`](quadtree::resolve_covering) only walks up, so a tile that just
    /// left the cut would pop off before its replacement can draw.
    fn ready_tiles_under(
        sections: &FastMap<SectionPos, SectionState>,
        drawn: &[SectionPos],
        max: crate::ident::Detail,
    ) -> Vec<SectionPos> {
        let ready = |p: SectionPos| sections.get(&p).is_some_and(|s| s.is_ready());
        let mut holes = FastSet::default();
        for &c in drawn {
            if quadtree::drawable_cover(c, max, &ready).is_none() {
                holes.insert(c);
            }
        }
        if holes.is_empty() {
            return Vec::new();
        }
        let mut extra = Vec::new();
        for (&s, state) in sections {
            if !state.is_ready() {
                continue;
            }
            let mut p = s;
            while p.detail < max {
                p = p.parent();
                if holes.contains(&p) {
                    extra.push(s);
                    break;
                }
            }
        }
        extra
    }

    /// True if the cell or a Ready ancestor covers it.
    pub(in crate::world) fn section_covered(&self, cell: SectionPos) -> bool {
        let max = crate::ident::Detail(crate::render_config::LOD_COARSEST_DETAIL as i8);
        let ready = |p: SectionPos| self.sections.get(&p).is_some_and(|s| s.is_ready());
        quadtree::drawable_cover(cell, max, &ready).is_some()
    }

    /// Edits affecting this section: chunks within its footprint and height domain.
    /// Used when re-extracting after an edit. Reads the `section_edit_chunks`
    /// index — O(this section's edited chunks), not a scan of every edit in
    /// the world (the reference scan survives as `edits_in_footprint`, pinned
    /// equivalent by test). Compacted-away chunks fall out at the lookup.
    pub(in crate::world) fn edits_for_section(
        &self,
        pos: SectionPos,
    ) -> Vec<(Coord, Vec<(usize, crate::block::registry::BlockId)>)> {
        let Some(chunks) = self.section_edit_chunks.get(&pos) else {
            return Vec::new();
        };
        chunks.iter().filter_map(|&c| Some((c, self.chunk_edits(c)?))).collect()
    }

    /// Rebuild the visible set every frame; as sections become Ready, the covering
    /// changes and stale entries would draw incorrectly.
    ///
    /// Also the LEVEL-TRIGGERED load arming (the fast-movement staleness fix):
    /// while ANY desired cell is unloaded, uncovered by a Ready self/ancestor,
    /// and not skipped as chunk-covered near field, the section lane stays
    /// armed. The old edge-triggered arming (boundary crossings and a few
    /// events) could go quiet with holes still open — flying far up left the
    /// covering permanently behind the live frontier, drawing a couple of
    /// stale coarse cubes over an otherwise missing far field.
    pub(in crate::world) fn rebuild_section_visible(&mut self, eng: Option<&mut Engine>) {
        let Some(center) = self.section_center() else {
            return;
        };
        let desired = std::mem::take(&mut self.section_desired);
        let max = crate::ident::Detail(crate::render_config::LOD_COARSEST_DETAIL as i8);
        let ready = |p: SectionPos| self.sections.get(&p).is_some_and(|s| s.is_ready());
        // A cell the settled full-res chunks already draw hands over at once, ahead of the clip.
        let mut drawn: Vec<SectionPos> = desired.iter().copied().filter(|&c| !self.full_res_covers(center, c)).collect();
        // A finer Ready tile keeps drawing while the cut that replaces it is still meshing.
        // Dropping it first opens a hole; the coarser tile takes over the frame it lands.
        let standins = Self::ready_tiles_under(&self.sections, &drawn, max);
        drawn.extend(standins);
        let cut = quadtree::resolve_covering(&drawn, max, &ready);
        let backlog = desired.iter().any(|&c| {
            !self.sections.contains_key(&c)
                && quadtree::drawable_cover(c, max, &ready).is_none()
                && !self.coverage_skips(center, c)
        });
        self.section_desired = desired;
        if backlog {
            self.pending_sections.set();
        }
        self.section_visible = cut.iter().copied().collect();
        // Adopt the new cut (hard pop); it decides what actually draws.
        let changed = self.section_fade.update_now(&self.section_visible);
        // The mask is a projection of that decision, so it follows the same diff —
        // a region whose slots have not landed yet is caught by the upload site instead.
        if let Some(eng) = eng {
            for (pos, mask) in changed {
                if let Some(state) = self.sections.get(&pos) {
                    state.set_visible(eng, mask);
                }
            }
        }
    }

    /// Free Ready sections that no desired cell draws, when the section floor is
    /// full and a desired cell is still unloaded. A still camera does not unload,
    /// so those extras would block covering for good.
    pub(super) fn reclaim_blocked_sections(&mut self, center: Coord, mut eng: Option<&mut Engine>) {
        let allowed = self.sections_allowed();
        let used = self.section_budget_used();
        if used < allowed {
            return;
        }
        // A still, converged floor has neither flag. Holes keep admission
        // pending (a full budget no longer clears it), and a frontier change
        // raises the cover flag before this runs, so the scan below stays off
        // the quiet path.
        if !self.pending_sections.get() && !self.section_cover_dirty.get() {
            return;
        }
        let holes = self
            .section_desired
            .iter()
            .filter(|&&c| {
                !self.sections.contains_key(&c)
                    && !self.section_covered(c)
                    && !self.coverage_skips(center, c)
                    && !self.quarantined.contains(&FailKey::Section { pos: c })
            })
            .count();
        if holes == 0 {
            return;
        }
        let need = holes + (used - allowed);
        let max = crate::ident::Detail(crate::render_config::LOD_COARSEST_DETAIL as i8);
        let ready = |p: SectionPos| self.sections.get(&p).is_some_and(|s| s.is_ready());
        let desired: FastSet<SectionPos> = self.section_desired.iter().copied().collect();
        let mut covers: FastSet<SectionPos> = FastSet::default();
        for &c in &self.section_desired {
            if let Some(p) = quadtree::drawable_cover(c, max, &ready) {
                covers.insert(p);
            }
        }
        let spare: Vec<SectionPos> = self
            .sections
            .iter()
            .filter_map(|(&s, state)| {
                if !matches!(state, SectionState::Ready { .. }) {
                    return None;
                }
                (!desired.contains(&s) && !covers.contains(&s) && !self.stands_under_hole(s, &desired)).then_some(s)
            })
            .collect();
        let mut victims: Vec<(u64, SectionPos)> = spare
            .into_iter()
            .map(|s| {
                let rank = <SectionLane as StreamLane>::dist2(self, center, s).unwrap_or(0);
                (rank, s)
            })
            .collect();
        if victims.is_empty() {
            return;
        }
        victims.sort_unstable_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| section_key(&a.1).cmp(&section_key(&b.1)))
        });
        let mut freed = 0usize;
        for (_, s) in victims.into_iter().take(need) {
            let gpu = match self.sections.get(&s) {
                Some(SectionState::Ready { meshes, cages, .. }) => !meshes.is_empty() || !cages.is_empty(),
                _ => false,
            };
            if gpu && eng.is_none() {
                continue;
            }
            if let Some(state) = self.sections.remove(&s) {
                if let Some(eng) = eng.as_deref_mut() {
                    state.free(eng);
                }
                freed += 1;
            }
        }
        if freed > 0 {
            self.pending_sections.set();
            self.section_cover_dirty.set();
        }
    }

    /// Unload sections outside desired, visible, and hysteresis bands (boundary cross).
    /// Hysteresis prevents thrashing at view edges.
    pub(super) fn unload_sections(&mut self, center: Coord, eng: &mut Engine) {
        self.unload_sections_with(center, |state| state.free(eng));
    }

    /// [`unload_sections`](Self::unload_sections) with the GPU release passed in.
    pub(in crate::world) fn unload_sections_with(&mut self, center: Coord, mut free: impl FnMut(SectionState)) {
        // KEEP reads the frame's cached frontier (already velocity-unioned),
        // so sections stay kept even as a fast-moving eye passes.
        let desired: FastSet<SectionPos> = self.section_desired.iter().copied().collect();
        let visible: FastSet<SectionPos> = self.section_visible.iter().map(|(p, _)| *p).collect();
        // Fading sections still draw this frame. Keep meshes until fade completes
        // or outgoing section vanishes mid-fade.
        let fading: FastSet<SectionPos> = self.section_fade.tracked().collect();
        let cfg = &self.far_pyramid();
        let (metric_body, metric_face, metric) = if self.section_on_chart(center) {
            let body = self
                .seams
                .chart_seat(center)
                .map(|s| super::section::CHART_BODY_BASE + s.index as u16)
                .unwrap_or(u16::MAX);
            (body, Face::PosY, self.chart_metric(center, DVec3::ZERO, cfg))
        } else {
            let metric_face = self.section_lod_face.map(|(_, f)| f).unwrap_or(Face::PosY);
            let metric_body = self.section_lod_face.map(|(b, _)| b).unwrap_or(0);
            let metric = self.section_metric_on(
                center,
                DVec3::ZERO,
                metric_face,
                self.generator.face_datum(metric_body, metric_face),
            );
            (metric_body, metric_face, metric)
        };
        let stale: Vec<SectionPos> = self
            .sections
            .keys()
            .copied()
            .filter(|s| {
                if visible.contains(s) || fading.contains(s) {
                    return false;
                }
                if desired.contains(s) {
                    // Settled full-res chunks draw it: hand it over (a chart punches it instead).
                    return self.full_res_covers(center, *s);
                }
                // A finer tile still drawing a hole. The visible rebuild stands it in after this.
                if self.stands_under_hole(*s, &desired) {
                    return false;
                }
                let span = s.span() as i64;
                // In the chart net. A neighbour's own storage is a different box, so the raw
                // column reads as past the horizon and the fold deletes a section the player
                // is about to walk back onto.
                let (cx, cz) = self.net_column(center, s.min_x() as i64 + span / 2, s.min_z() as i64 + span / 2);
                // A section on another face is kept only while it is still desired.
                let dist = if s.body == metric_body && s.face == metric_face {
                    metric.point(cx as f64, cz as f64)
                } else {
                    EyeDist::new(f32::MAX)
                };
                !pyramid::acceptable(dist, s.detail, cfg)
            })
            .collect();
        for s in &stale {
            if let Some(state) = self.sections.remove(s) {
                super::adjust_count(
                    &mut self.meshing_sections,
                    matches!(state, SectionState::Meshing { .. }),
                    false,
                );
                free(state);
            }
        }
        // Removals move the covering (a freed cell may re-expose an ancestor).
        self.section_cover_dirty.raise(!stale.is_empty());
    }

    /// Free GPU meshes so edited sections re-extract from the updated overlay.
    pub(in crate::world) fn remesh_dirty_sections(&mut self, eng: &mut Engine) {
        if self.dirty_sections.is_empty() {
            return;
        }
        let dirty: Vec<SectionPos> = self.dirty_sections.iter().copied().collect();
        let mut freed = false;
        for s in dirty {
            // Re-extract only once the square's overlay is in: its upload reads the overlay colour.
            if self.section_overlay_dirty.contains(&s) {
                continue;
            }
            match self.sections.get(&s) {
                Some(SectionState::Ready { .. }) => {
                    if let Some(state) = self.sections.remove(&s) {
                        state.free(eng);
                    }
                    self.dirty_sections.remove(&s);
                    freed = true;
                }
                Some(SectionState::Meshing { .. }) => {} // in flight: free once it lands Ready
                None => {
                    self.dirty_sections.remove(&s);
                }
            }
        }
        if freed {
            self.pending_sections.set();
            self.section_cover_dirty.set();
        }
    }
}
