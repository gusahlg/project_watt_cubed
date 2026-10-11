//! Far-field sections: the relief bake, the edit overlay, residency and the visible covering.

use super::*;

/// The coarsest section detail: a Ready ancestor walk stops here.
const COARSEST: crate::ident::Detail = crate::ident::Detail(crate::render_config::LOD_COARSEST_DETAIL as i8);

/// `s` lies strictly under one of `holes`.
fn under_hole(s: SectionPos, holes: &FastSet<SectionPos>) -> bool {
    let mut p = s;
    while p.detail < COARSEST {
        p = p.parent();
        if holes.contains(&p) {
            return true;
        }
    }
    false
}

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
        }
        match self.section_overlay_dirty.len() {
            0 => Progress::Idle,
            n => Progress::Partial { remaining: n as u32 },
        }
    }

    /// The atlas patch a chart section is bent through. `None` for a cube section or a square
    /// that is not inside a storage box.
    pub(in crate::world) fn chart_bend(&self, pos: SectionPos) -> Option<super::ChartBend> {
        if pos.body < super::section::CHART_BODY_BASE {
            let atlas = self.seams.atlases().iter().find(|a| {
                a.grid.as_ref().is_some_and(|g| g.body == pos.body) && a.warp.is_some()
            })?;
            return Some(super::ChartBend { atlas: atlas.clone(), patch: crate::space::atlas::Patch::Grid, origin: (0, 0) });
        }
        let index = (pos.body - super::section::CHART_BODY_BASE) as usize;
        let atlas = self.seams.atlases().get(index)?.clone();
        let origin = self.generator.chart_origin(pos.body, pos.face)?;
        let patch = crate::space::atlas::Patch::Shell { band: 0, face: pos.face };
        Some(super::ChartBend { atlas, patch, origin })
    }

    /// Storage `(x, z)` of `pos`'s centre: a chart tile counts from its patch origin, any other
    /// tile is in its own frame already.
    pub(in crate::world) fn tile_centre(&self, pos: SectionPos) -> (i64, i64) {
        let (ox, oz) = self.generator.chart_origin(pos.body, pos.face).unwrap_or((0, 0));
        let half = i64::from(pos.span()) / 2;
        (i64::from(ox) + i64::from(pos.min_x()) + half, i64::from(oz) + i64::from(pos.min_z()) + half)
    }

    /// Tile `s`'s footprint `[x0, z0, x1, z1)`: storage blocks on a chart, its face frame elsewhere.
    #[cfg(test)]
    pub(in crate::world) fn tile_rect(&self, s: SectionPos) -> [i64; 4] {
        let (ox, oz) = self.generator.chart_origin(s.body, s.face).unwrap_or((0, 0));
        let (x0, z0, w) = (i64::from(ox) + i64::from(s.min_x()), i64::from(oz) + i64::from(s.min_z()), i64::from(s.span()));
        [x0, z0, x0 + w, z0 + w]
    }

    /// The tile of chart `body` at `detail` holding storage column `(x, z)`, counted from the
    /// band-0 patch that holds it. `None` off its patches.
    #[cfg(test)]
    pub(in crate::world) fn tile_at(&self, body: u16, detail: crate::ident::Detail, x: i64, z: i64) -> Option<SectionPos> {
        let atlas = self.seams.atlases().get(usize::from(body.checked_sub(super::section::CHART_BODY_BASE)?))?;
        let (crate::space::atlas::Patch::Shell { band: 0, face }, _, _) = atlas.column(x, z)? else { return None };
        let (ox, oz) = self.generator.chart_origin(body, face)?;
        let span = i64::from(super::section::section_span(detail));
        let (lx, lz) = ((x - i64::from(ox)).div_euclid(span), (z - i64::from(oz)).div_euclid(span));
        Some(SectionPos { detail, body, face, x: i32::try_from(lx).ok()?, z: i32::try_from(lz).ok()? })
    }

    /// The generator's surface over storage rect `[x0, x1) × [z0, z1)` of chart `body`: callers on
    /// the storage grid read through its +Y patch's frame, which spans all of storage.
    pub(in crate::world) fn storage_rect(&self, body: u16, x0: i32, z0: i32, x1: i32, z1: i32) -> Option<(i32, i32)> {
        let (ox, oz) = self.generator.chart_origin(body, Face::PosY).unwrap_or((0, 0));
        self.generator.surface_rect(body, Face::PosY, x0 - ox, z0 - oz, x1 - ox, z1 - oz)
    }

    /// Ready tiles under an uncovered desired cell, once the section floor is full.
    /// Zero below the floor, so a quiet or under-budget admission does not scan.
    pub(in crate::world) fn count_section_standins(&self) -> usize {
        if self.section_budget_used() < self.sections_allowed() {
            return 0;
        }
        let holes = self.section_holes(&self.section_desired);
        if holes.is_empty() {
            return 0;
        }
        self.sections.iter().filter(|&(&s, st)| st.is_ready() && under_hole(s, &holes)).count()
    }

    /// The cells of `cells` that nothing Ready draws yet: no Ready self or ancestor.
    fn section_holes(&self, cells: &[SectionPos]) -> FastSet<SectionPos> {
        cells.iter().copied().filter(|&c| !self.section_covered(c)).collect()
    }

    /// `s` is a Ready tile strictly under one of `holes` (see [`section_holes`](Self::section_holes)).
    /// Unload and reclaim would otherwise drop it in the frame before the visible set
    /// can stand it back in.
    pub(super) fn stands_under_hole(&self, s: SectionPos, holes: &FastSet<SectionPos>) -> bool {
        self.section_ready(s) && under_hole(s, holes)
    }

    /// Section `p` is loaded and Ready.
    pub(in crate::world) fn section_ready(&self, p: SectionPos) -> bool {
        self.sections.get(&p).is_some_and(|s| s.is_ready())
    }

    /// True if the cell or a Ready ancestor covers it.
    pub(in crate::world) fn section_covered(&self, cell: SectionPos) -> bool {
        quadtree::drawable_cover(cell, COARSEST, &|p| self.section_ready(p)).is_some()
    }

    /// Remove section `pos`, keeping the meshing count and the cover flag in step. The caller
    /// frees what it returns.
    pub(in crate::world) fn drop_section(&mut self, pos: SectionPos) -> Option<SectionState> {
        let state = self.sections.remove(&pos)?;
        super::adjust_count(&mut self.meshing_sections, matches!(state, SectionState::Meshing { .. }), false);
        self.section_cover_dirty.set();
        Some(state)
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
        if self.section_center().is_none() {
            return;
        }
        let ready = |p: SectionPos| self.section_ready(p);
        let cut = quadtree::whole_cut(&self.section_desired, COARSEST, &ready);
        let backlog = self.section_desired.iter().any(|&c| !self.sections.contains_key(&c) && quadtree::drawable_cover(c, COARSEST, &ready).is_none());
        if backlog {
            self.pending_sections.set();
        }
        self.section_visible = cut;
        // Adopt the new cut (hard pop); it decides what actually draws.
        let changed = self.section_fade.update_now(&self.section_visible);
        // Visibility follows the same diff; a tile whose slots have not landed yet is caught at
        // its upload instead.
        if let Some(eng) = eng {
            for (pos, on) in changed {
                if let Some(state) = self.sections.get(&pos) {
                    state.set_visible(eng, on);
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
                    && !self.quarantined.contains(&FailKey::Section { pos: c })
            })
            .count();
        if holes == 0 {
            return;
        }
        let need = holes + (used - allowed);
        // What each desired cell is drawn by, or that nothing draws it yet. A Ready desired cell
        // draws itself, so the covers hold every one.
        let mut covers: FastSet<SectionPos> = FastSet::default();
        let mut uncovered: FastSet<SectionPos> = FastSet::default();
        for &c in &self.section_desired {
            match quadtree::drawable_cover(c, COARSEST, &|p| self.section_ready(p)) {
                Some(p) => covers.insert(p),
                None => uncovered.insert(c),
            };
        }
        let spare: Vec<SectionPos> = self
            .sections
            .iter()
            .filter_map(|(&s, state)| {
                if !matches!(state, SectionState::Ready { .. }) {
                    return None;
                }
                (!covers.contains(&s) && !under_hole(s, &uncovered) && !self.section_ancestors.contains(&s)).then_some(s)
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
            if let Some(state) = self.drop_section(s) {
                if let Some(eng) = eng.as_deref_mut() {
                    state.free(eng);
                }
                freed += 1;
            }
        }
        if freed > 0 {
            self.pending_sections.set();
        }
    }

    /// Unload sections outside desired, visible, and hysteresis bands (boundary cross).
    /// Hysteresis prevents thrashing at view edges.
    pub(super) fn unload_sections(&mut self, center: Coord, eng: &mut Engine) {
        self.unload_sections_with(center, |state| state.free(eng));
    }

    /// [`unload_sections`](Self::unload_sections) with the GPU release passed in. A section stays
    /// while it is desired, drawn or fading, stands in under a hole, or is within a quarter of the
    /// rule's ranges of being the detail its distance gives (so turning back reuses it).
    pub(in crate::world) fn unload_sections_with(&mut self, center: Coord, mut free: impl FnMut(SectionState)) {
        let desired: FastSet<SectionPos> = self.section_desired.iter().copied().collect();
        let holes = self.section_holes(&self.section_desired);
        let visible: FastSet<SectionPos> = self.section_visible.iter().copied().collect();
        // Fading sections still draw this frame. Keep meshes until fade completes
        // or outgoing section vanishes mid-fade.
        let fading: FastSet<SectionPos> = self.section_fade.tracked().collect();
        let ground = self.far_ground(center);
        let eye = super::super::rule::PhysBox::eye_cell(self.phys_eye(center, true));
        let stale: Vec<SectionPos> = self
            .sections
            .keys()
            .copied()
            .filter(|s| {
                if visible.contains(s) || fading.contains(s) || desired.contains(s) || self.section_ancestors.contains(s) {
                    return false;
                }
                // A finer tile still drawing a hole. The visible rebuild stands it in after this.
                if self.stands_under_hole(*s, &holes) {
                    return false;
                }
                !ground.is_some_and(|g| self.rule_keeps(g, *s, &eye))
            })
            .collect();
        // Removals move the covering (a freed cell may re-expose an ancestor).
        for &s in &stale {
            if let Some(state) = self.drop_section(s) {
                free(state);
            }
        }
    }

    /// Section `s` of `ground` is within the hold of what the rule draws from `eye`: inside its
    /// horizon, its parent would split, and it would not split well inside its own range.
    fn rule_keeps(&self, ground: super::super::rule::Ground, s: SectionPos, eye: &super::super::rule::PhysBox) -> bool {
        use super::super::rule::Ground;
        const HOLD: f64 = 1.25;
        let on = match ground {
            Ground::Chart { index } => s.body == super::section::CHART_BODY_BASE + index as u16,
            Ground::Cube { body } => s.body == body,
            Ground::Flat => s.body == 0,
        };
        if !on {
            return false;
        }
        let rule = self.rule_on(ground);
        let Some(b) = self.node_box(s) else { return false };
        let d = b.dist_box(eye);
        if d >= rule.horizon() * HOLD || s.detail.0 > rule.coarsest {
            return false;
        }
        if s.detail.0 < rule.coarsest && self.node_box(s.parent()).is_some_and(|p| p.dist_box(eye) >= rule.range_of(p.span) * HOLD) {
            return false;
        }
        !(s.detail.0 > rule.finest && d < rule.range_of(b.span) / HOLD)
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
                    if let Some(state) = self.drop_section(s) {
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
        }
    }
}
