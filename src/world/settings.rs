//! Runtime world settings: view distances, the render config and its lanes, the lighting and AO
//! toggles, and freeing meshes on world leave.

use voxel_engine::Engine;

use crate::render_config::RenderConfig;

use super::{MeshState, VERTICAL_RADIUS_RANGE, VIEW_RADIUS_RANGE, World};

impl World {
    /// Current render distance in chunk rings.
    pub fn view_radius(&self) -> i32 {
        self.view.horizontal
    }

    /// Current vertical streaming distance in chunk layers above and below the eye.
    #[cfg(test)]
    pub fn vertical_radius(&self) -> i32 {
        self.view.vertical
    }

    /// Compatibility setter for callers with a single render-distance value.
    /// The vertical distance retains its historical half-horizontal derivation.
    #[cfg(test)]
    pub fn set_view_radius(&mut self, radius: i32) {
        let horizontal = radius.clamp(*VIEW_RADIUS_RANGE.start(), *VIEW_RADIUS_RANGE.end());
        let view = super::ViewVolume::view(horizontal);
        self.set_view_distances(view.horizontal, view.vertical);
    }

    /// Set the anisotropic full-resolution streaming volume. Independent axes
    /// let minimum mode keep only the collision-relevant vertical slab. Marks
    /// streaming dirty so the next [`stream`](Self::stream) unloads past the
    /// new radius or resumes meshing out to it.
    pub fn set_view_distances(&mut self, horizontal: i32, vertical: i32) {
        let horizontal = horizontal.clamp(*VIEW_RADIUS_RANGE.start(), *VIEW_RADIUS_RANGE.end());
        let vertical =
            vertical.clamp(*VERTICAL_RADIUS_RANGE.start(), *VERTICAL_RADIUS_RANGE.end());
        if horizontal != self.view.horizontal || vertical != self.view.vertical {
            let shrunk = horizontal < self.view.horizontal || vertical < self.view.vertical;
            self.view = super::ViewVolume::new(horizontal, vertical);
            let rings = self.view.worklist_rings(self.live_up(), 0);
            self.mesh_worklist.resize(rings);
            self.light_worklist.resize(rings);
            // Unit re-pinned on stream; invalidate centre for rescan.
            // unload/ensure/scan pass even though the player hasn't moved.
            self.center = None;
            // The next full pass must probe the WHOLE new box (a grown radius
            // exposes chunks the old shell diff would skip).
            self.prev_mesh_box = None;
            self.prev_unload_box = None;
            self.pending_fresh.set();
            // On shrink, meshes between the new radius and the (also shrunk)
            // unload ring would otherwise stay drawn until the player moves;
            // flag them so the next stream frees them immediately. OR it in so a
            // shrink queued before the next stream survives a later grow.
            self.radius_shrunk.raise(shrunk);
            // In-flight worker jobs are NOT cancelled: results now outside the
            // radius are dropped by the range checks when they drain.
        }
    }

    /// Set render lanes (occlusion/lod2). Entry-only; mesh teardown not needed.
    /// Live settings must use [`set_render_config`](Self::set_render_config)
    /// so GPU state is retired.
    pub fn set_render_lanes(&mut self, occlusion: bool, lod2: bool) {
        self.occlusion_forced = occlusion;
        self.lod2 = lod2;
    }

    /// Apply the live world-owned subset of render settings. A far-field ladder
    /// transition retires every old section allocation and all derived state;
    /// enabling then re-arms streaming to build the new hierarchy.
    pub fn set_render_config(&mut self, render: RenderConfig, eng: &mut Engine) {
        if self.occlusion_forced != render.occlusion {
            self.occlusion_forced = render.occlusion;
            self.occlusion_dirty.set();
            if !render.occlusion {
                // Stop consulting an old visible set immediately, before the
                // next mutable stream sync point.
                self.occlusion_active = false;
            }
        }

        if !self.section_config_changed(render) {
            return;
        }
        let pyramid_changed = self.section_pyramid_changed(render);
        let (levels, detail) = render.normalized_lod();
        let unit = self.view.lod_unit();
        // A pure on/off transition retires draw/claim state but preserves the
        // immutable generator mip (and an in-flight bake). Ladder/range changes
        // alter its extent and must rebuild it.
        self.clear_section_lane(eng, pyramid_changed);
        self.lod2 = render.lod2;
        self.section_pyramid = super::pyramid::PyramidCfg::sections_with(unit, levels, detail);
        if self.lod2 {
            self.pending_sections.set();
        }
    }

    /// Pure change detector kept separate so distance-only LOD invalidation is
    /// regression-testable without constructing a renderer/Engine.
    pub(in crate::world) fn section_config_changed(&self, render: RenderConfig) -> bool {
        self.lod2 != render.lod2 || self.section_pyramid_changed(render)
    }

    pub(in crate::world) fn section_pyramid_changed(&self, render: RenderConfig) -> bool {
        let (levels, detail) = render.normalized_lod();
        let unit = self.view.lod_unit();
        self.section_pyramid.levels.get() != levels
            || self.section_pyramid.finest.0 != detail as i8
            || self.section_pyramid.unit != unit
    }

    /// Retire the whole far-section lane: free every GPU allocation, drop all
    /// derived selection/cover state, purge queued far jobs, and advance the
    /// epoch so in-flight worker results from the old configuration can never
    /// land. `reset_mip` additionally discards the relief bake (its extent
    /// depends on the ladder).
    pub(in crate::world) fn clear_section_lane(&mut self, eng: &mut Engine, reset_mip: bool) {
        // Queued section snapshots belong to the old epoch/configuration. Drop
        // them immediately instead of letting a queue of obsolete, heavyweight
        // jobs monopolize workers after a live LOD change.
        if let Some(workers) = &self.workers {
            let _ = workers.clear_far();
        }
        self.section_epoch = self.section_epoch.wrapping_add(1);
        self.section_pending_claim = None;
        for (_, state) in self.sections.drain() {
            state.free(eng);
        }
        self.meshing_sections = 0;
        self.section_upload_queue.clear();
        self.pending_sections.take();
        self.dirty_sections.clear();
        self.section_desired.clear();
        self.section_frontier_key = None;
        self.section_visible.clear();
        self.section_fade = Default::default();
        self.section_cover_dirty.set();
        if reset_mip {
            // Ladder/distance changes alter the required bake extent and levels.
            self.section_mip = None;
            self.section_mip_rx = None;
        }
        self.section_eye_prev = None;
        self.section_vel = voxel_engine::DVec3::ZERO;
        self.job_strikes.retain(|key, _| !matches!(key, super::streaming::FailKey::Section { .. }));
        self.quarantined.retain(|key| !matches!(key, super::streaming::FailKey::Section { .. }));
    }

    /// Whether cross-chunk lighting is currently enabled.
    pub fn lighting(&self) -> bool {
        self.lighting
    }

    /// Toggle cross-chunk lighting. On a real change, drops every mesh and
    /// re-scans from scratch (via [`free_meshes`](Self::free_meshes)) so the next
    /// [`stream`](Self::stream) rebuilds them with — or without — settled light.
    /// A no-op when the value is unchanged, so it is cheap to push every frame.
    pub fn set_lighting(&mut self, on: bool, eng: &mut Engine) {
        if !self.transition_lighting(on) {
            return;
        }
        self.free_meshes(eng);
    }

    /// Toggle baked corner AO. A meshing input like lighting: the hot tables
    /// restamp (epoch bump) and every mesh rebuilds with the new corners.
    pub fn set_ao(&mut self, on: bool, eng: &mut Engine) {
        if !self.set_ao_flag(on) {
            return;
        }
        self.free_meshes(eng);
    }

    /// Stamp AO without freeing GPU meshes (headless tests, already-empty worlds).
    pub(crate) fn set_ao_flag(&mut self, on: bool) -> bool {
        if on == self.ao {
            return false;
        }
        self.ao = on;
        self.tables_epoch = self.tables_epoch.wrapping_add(1);
        true
    }

    /// Move the CPU lighting pipeline between enabled and full-bright modes.
    /// Kept separate from GPU mesh retirement so the asynchronous state machine
    /// can be tested without constructing an engine.
    pub(crate) fn transition_lighting(&mut self, on: bool) -> bool {
        if on == self.lighting {
            return false;
        }

        self.lighting = on;
        self.light_epoch = self.light_epoch.wrapping_add(1);
        if !on {
            // Work captured but not yet published has no trustworthy settled
            // grid. Mark those chunks missing so re-enable discovers them without
            // retaining a second dormant work set.
            let unsettled: Vec<_> = self
                .light_worklist
                .iter()
                .chain(&self.light_inflight)
                .copied()
                .chain(self.light_apply_queue.iter().map(|(coord, _)| *coord))
                .collect();
            for coord in unsettled {
                if let Some(loaded) = self.chunks.get_mut(&coord) {
                    loaded.light = None;
                }
            }
            self.light_worklist.clear();
        }
        self.light_inflight.clear();
        self.light_apply_queue.clear();
        self.light_pending.take();
        self.light_terminal.clear();
        // `light_gate` (degraded/blocked_since) is left untouched on purpose: the
        // per-frame `tick_light_gate` reconciles it against live predicates. With
        // lighting off, `light_ready` is data-only, so blocked timers drain and any
        // degraded chunk is re-meshed full-bright and cleared.

        if on {
            // Edits made while off remain in the dormant worklist. Chunks loaded
            // while off have no grid, so add only those; unchanged settled grids
            // remain valid and avoid a whole-volume relight.
            self.light_worklist.extend(
                self.chunks
                    .iter()
                    .filter_map(|(&coord, loaded)| loaded.light.is_none().then_some(coord)),
            );
            if !self.light_worklist.is_empty() {
                self.light_pending.set();
            }
        }
        true
    }

    /// Free every chunk's GPU mesh and reset every chunk to `NeedsMesh` — used
    /// when leaving a world. The voxel data stays; a later
    /// [`stream`](Self::stream) rebuilds the meshes from scratch. (Resetting to
    /// `NeedsMesh` — rather than back to `Air` for born-air chunks — matches the
    /// old unconditional `meshed = false`; the next scan re-derives `Air`.)
    pub fn free_meshes(&mut self, eng: &mut Engine) {
        // Every drawn mesh is going away: the settled-ring scan restarts.
        self.lod_clip_shrunk.set();
        for loaded in self.chunks.values_mut() {
            // Any worker mesh captured before this reset must not be accepted if
            // it lands after the next stream establishes a new centre.
            loaded.rev = loaded.rev.wrapping_add(1);
            loaded.retire(MeshState::needs_mesh(), eng);
        }
        for (_, cage) in self.cages.drain() {
            eng.free_cage(cage);
        }
        self.building_meshes = 0;
        // Every chunk is now `NeedsMesh`, so the `Dirty` fiber is empty; drop
        // the membership set and the stale hint with it.
        self.dirty_worklist.clear();
        self.pending_dirty.take();
        // Drop the pipeline bookkeeping too: buffered worker meshes are for a
        // world we are leaving, and in-flight jobs may re-run from scratch if
        // we come back. Results still flying land against the invalidated
        // centre below and are dropped by the range/rev checks — at worst a
        // coord gets generated or meshed twice, never wrongly.
        self.generating.clear();
        self.upload_queue.clear();
        // Sections belong to the world being left: epoch bump + far-job purge
        // so in-flight results cannot land after we come back.
        self.clear_section_lane(eng, false);
        self.center = None;
        // Every chunk is back to `NeedsMesh`; re-seed the mesh lane's worklist so
        // the next stream rebuilds them (the worklist is the fresh-mesh index now).
        self.mesh_worklist.clear();
        self.mesh_worklist.extend(self.chunks.keys().copied());
        self.pending_fresh.set();
        self.light_terminal.clear();
    }
}
