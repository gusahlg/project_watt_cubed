//! Player edits and world-leaving cleanup: block placement/breaking, the
//! edit-overlay save iterator, dirty-marking for remesh, freeing meshes, and
//! the render-distance setting. Code motion only: these are `World` methods;
//! the struct itself lives in `mod.rs`.

use voxel_engine::{DVec3, Engine};

use crate::block::registry::BlockId;
use crate::coord::{BlockCoord, Face, Local};
use crate::render_config::RenderConfig;

use super::chunk::Chunk;
use super::generation::TerrainGenerator;
use super::{Coord, MeshState, VERTICAL_RADIUS_RANGE, VIEW_RADIUS_RANGE, World};

impl World {
    /// Current render distance in chunk rings.
    pub fn view_radius(&self) -> i32 {
        self.view.horizontal
    }

    /// Current vertical streaming distance in chunk layers above and below the eye.
    pub fn vertical_radius(&self) -> i32 {
        self.view.vertical
    }

    /// Compatibility setter for callers with a single render-distance value.
    /// The vertical distance retains its historical half-horizontal derivation.
    pub fn set_view_radius(&mut self, radius: i32) {
        let horizontal = radius.clamp(*VIEW_RADIUS_RANGE.start(), *VIEW_RADIUS_RANGE.end());
        let view = super::ViewVolume::view(horizontal);
        self.set_view_distances(view.horizontal, view.vertical);
    }

    /// Set the anisotropic full-resolution streaming volume. Independent axes
    /// let minimum mode keep only the collision-relevant vertical slab.
    pub fn set_view_distances(&mut self, horizontal: i32, vertical: i32) {
        let horizontal = horizontal.clamp(*VIEW_RADIUS_RANGE.start(), *VIEW_RADIUS_RANGE.end());
        let vertical = vertical.clamp(
            *VERTICAL_RADIUS_RANGE.start(),
            *VERTICAL_RADIUS_RANGE.end(),
        );
        if horizontal != self.view.horizontal || vertical != self.view.vertical {
            let shrunk = horizontal < self.view.horizontal || vertical < self.view.vertical;
            self.view = super::ViewVolume::new(horizontal, vertical);
            // Unit re-pinned on stream; invalidate centre for rescan.
            // unload/ensure/scan pass even though the player hasn't moved.
            self.center = None;
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

        let section_changed = self.section_config_changed(render);
        let pyramid_changed = self.section_pyramid_changed(render);
        let (levels, detail) = render.normalized_lod();
        let unit = self.view.lod_unit();
        if !section_changed {
            return;
        }

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
            || self.section_pyramid.finest.0 != detail
            || self.section_pyramid.unit != unit
    }

    /// Compatibility setter for construction-time callers. Live settings must
    /// use [`set_render_config`](Self::set_render_config) so GPU state is retired.
    pub fn set_render_lanes(&mut self, occlusion: bool, lod2: bool) {
        self.occlusion_forced = occlusion;
        self.lod2 = lod2;
    }

    fn clear_section_lane(&mut self, eng: &mut Engine, reset_mip: bool) {
        // Queued section snapshots belong to the old epoch/configuration. Drop
        // them immediately instead of letting up to FAR_QUEUE_CAP obsolete,
        // heavyweight jobs monopolize workers after a live LOD change.
        if let Some(workers) = &self.workers {
            let _ = workers.clear_far();
        }
        self.section_epoch = self.section_epoch.wrapping_add(1);
        self.section_pending_claim = None;
        for (_, state) in self.sections.drain() {
            state.free(eng);
        }
        self.section_upload_queue.clear();
        self.section_draw_cache.clear();
        self.section_block_draws.clear();
        self.pending_sections.take();
        self.dirty_sections.clear();
        self.section_desired.clear();
        self.section_frontier_key = None;
        self.section_visible.clear();
        self.section_cover_dirty = true;
        self.section_fade = Default::default();
        if reset_mip {
            // Ladder/distance changes alter the required bake extent and levels.
            self.section_mip = None;
            self.section_mip_rx = None;
        }
        self.section_eye_prev = None;
        self.section_vel = DVec3::ZERO;
        self.job_strikes
            .retain(|key, _| !matches!(key, super::streaming::FailKey::Section { .. }));
        self.quarantined
            .retain(|key| !matches!(key, super::streaming::FailKey::Section { .. }));
    }

    /// Whether cross-chunk lighting is currently enabled.
    pub fn lighting(&self) -> bool {
        self.lighting
    }

    /// Toggle cross-chunk lighting. On a real change, drops every near mesh and
    /// re-scans that lane so the next [`stream`](Self::stream) rebuilds it with —
    /// or without — settled light. Independent far sections stay resident.
    /// A no-op when the value is unchanged, so it is cheap to push every frame.
    pub fn set_lighting(&mut self, on: bool, eng: &mut Engine) {
        if !self.transition_lighting(on) {
            return;
        }
        self.invalidate_near_meshes(eng);
    }

    /// Toggle baked corner AO. A meshing input like lighting: the hot tables
    /// restamp (epoch bump) and every mesh rebuilds with the new corners.
    pub fn set_ao(&mut self, on: bool, eng: &mut Engine) {
        if !self.transition_ao(on) {
            return;
        }
        self.invalidate_near_meshes(eng);
    }

    /// Apply both CPU meshing inputs as one transition. Settings profiles often
    /// change lighting and AO together; retiring the near mesh volume once avoids
    /// a duplicate full-map scan/free/reseed. Far sections use neither settled
    /// chunk light nor corner AO and remain valid.
    pub fn set_meshing_config(&mut self, lighting: bool, ao: bool, eng: &mut Engine) {
        let changed = self.transition_lighting(lighting) | self.transition_ao(ao);
        if changed {
            self.invalidate_near_meshes(eng);
        }
    }

    fn transition_ao(&mut self, on: bool) -> bool {
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
    pub(in crate::world) fn transition_lighting(&mut self, on: bool) -> bool {
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

    /// Retire only full-resolution chunk meshes after a live meshing-input
    /// change. Generation jobs and the independent far-field hierarchy remain
    /// valid, avoiding duplicate terrain work and visible LOD re-convergence.
    fn invalidate_near_meshes(&mut self, eng: &mut Engine) {
        self.invalidate_near_mesh_states(|meshes| meshes.free(eng));
    }

    /// Apply the CPU-side half of [`invalidate_near_meshes`](Self::invalidate_near_meshes).
    /// The callback consumes each retired GPU allocation; keeping that operation
    /// injectable makes the complete state transition regression-testable without
    /// constructing a renderer.
    fn invalidate_near_mesh_states(&mut self, mut free_meshes: impl FnMut(super::ChunkMeshes)) {
        self.mesh_worklist.clear();
        let mut removed_draws = false;
        for (&coord, loaded) in &mut self.chunks {
            // Any worker mesh captured before this reset must not be accepted if
            // it lands after the settings transition. This includes `Air`: it
            // should have no claim, but bumping all revisions makes a stale queued
            // result harmless even if an earlier state transition produced it.
            loaded.rev = loaded.rev.wrapping_add(1);

            // AO and light alter vertices, not voxel occupancy. A chunk already
            // proven to be uniform air therefore remains fully settled `Air` and
            // must not consume a worker job. Every other state is invalidated:
            // an outstanding build claim is released, and Ready/Dirty ownership
            // moves into `retired` before its handles are freed by the callback.
            if matches!(loaded.state, MeshState::Air) {
                continue;
            }
            let retired = std::mem::replace(
                &mut loaded.state,
                MeshState::NeedsMesh { building: false },
            );
            removed_draws |= retired.live_meshes().is_some();
            if let Some(meshes) = retired.into_owned() {
                free_meshes(meshes);
            }
            self.mesh_worklist.insert(coord);
        }

        if removed_draws {
            // Live AO/lighting changes may run between stream and render.
            // Prepared records copy handles, so retire them immediately instead
            // of waiting for the next end-of-stream sync.
            self.draw_set_rev = self.draw_set_rev.wrapping_add(1);
            self.draw_cache.clear();
            self.draw_cache_rev = self.draw_set_rev;
        }
        // Every Dirty chunk moved to `NeedsMesh`; drop its stale fiber hint.
        self.pending_dirty.take();
        // Buffered meshes captured the old AO/light inputs. In-flight results
        // carry the bumped chunk revision and are rejected when they land.
        self.upload_queue.clear();
        if self.mesh_worklist.is_empty() {
            // An all-air volume is already converged. Do not pay even an empty
            // mesh-lane pass after changing AO or lighting.
            self.pending_fresh.take();
        } else {
            self.pending_fresh.set();
        }
    }

    /// Free every GPU mesh when leaving a world. Voxel data and the expensive
    /// terrain-summary mip stay reusable, but near/far draw ownership and all
    /// outstanding lane claims are retired before the world is dropped.
    pub fn free_meshes(&mut self, eng: &mut Engine) {
        self.invalidate_near_meshes(eng);
        self.generating.clear();
        self.clear_section_lane(eng, false);
        self.center = None;
    }

    /// Set block at world coord; record in edit overlay and mark chunk(s) for remesh.
    /// Returns previous block.
    pub fn set_block(&mut self, x: i32, y: i32, z: i32, id: BlockId) -> BlockId {
        let (coord, local) = BlockCoord::new(x, y, z).split();
        let (lx, ly, lz) = (local.lx(), local.ly(), local.lz());
        let index = Chunk::index(lx, ly, lz);

        let previous = self.block_at(x, y, z);
        // A no-op placement (same block already there) changes no exposed face,
        // so skip recording the edit, bumping revs, and the synchronous remesh
        // of up to four chunks it would otherwise trigger. Gate on the chunk
        // being loaded: `block_at` reads an unloaded chunk as AIR regardless of
        // its true generated/edited contents, so `previous` is only an
        // authoritative "what's there" for a loaded chunk — a remote edit into
        // an unloaded chunk must still be recorded in the overlay.
        if previous == id && self.chunks.contains_key(&coord) {
            return previous;
        }
        // Overlay compaction: a write that restores what generation would
        // produce is pure weight in the overlay — regeneration yields it
        // anyway. Drop the entry instead of storing it, so the overlay (and
        // every save and join transfer built from it) stays proportional to
        // the world's real difference from its seed. One generator query per
        // edit: user-click/network rate, never the voxel hot path.
        let old_edit = self
            .edits
            .get(&coord)
            .and_then(|cells| cells.get(&index))
            .copied();
        let generated = self
            .generator
            .block_at(x, y, z, self.generator.height(x, z));
        let new_edit = if id == generated {
            if let Some(cells) = self.edits.get_mut(&coord) {
                cells.remove(&index);
                if cells.is_empty() {
                    self.edits.remove(&coord);
                }
            }
            None
        } else {
            self.edits.entry(coord).or_default().insert(index, id);
            Some(id)
        };
        self.edit_generation += 1;
        // Skylight ceiling upkeep (G-03): a roof appearing above a column's
        // current ceiling raises it; the topmost edited roof disappearing
        // lowers it. Either way the cached window is stale, and every loaded
        // chunk at or below the edit seeds skylight from it — re-settle them
        // so a constructed roof actually darkens the world underneath.
        if let Some(ceiling) = self.ceilings.get(&(coord.x, coord.z)) {
            let cell = ceiling.surface_at(lx, lz);
            let raises = new_edit.is_some_and(|id| self.registry.is_opaque(id)) && y + 1 > cell;
            let lowers = old_edit.is_some_and(|id| self.registry.is_opaque(id)) && y + 1 == cell;
            if raises || lowers {
                self.ceilings.remove(&(coord.x, coord.z));
                if self.lighting {
                    let shadowed: Vec<Coord> = self
                        .chunks
                        .keys()
                        .copied()
                        .filter(|c| c.x == coord.x && c.z == coord.z && c.y <= coord.y)
                        .collect();
                    for c in shadowed {
                        self.light_worklist.insert(c);
                    }
                    self.light_pending.set();
                }
            }
        }
        // Invalidate section to re-extract from overlay.
        if self.lod2 {
            self.mark_dirty_sections_from_edit(x, y, z);
        }

        if let Some(loaded) = self.chunks.get_mut(&coord) {
            std::sync::Arc::make_mut(&mut loaded.chunk).set_index(index, id);
            // Editing this chunk's own voxels can open or seal an interior pocket,
            // so its connectivity is stale — invalidate it (the occlusion rebuild
            // recomputes lazily if the gate is active) and flag the visible set.
            loaded.connectivity = None;
            self.occlusion_dirty.set();
            // Keep whatever is currently drawn as `prev` so the old mesh shows
            // until the sync remesh: Ready(m) → Dirty{Some(m)}, and re-editing
            // an already-Dirty{Some} chunk preserves its mesh (the token MOVES,
            // no free). NeedsMesh (building or not)/Air draw nothing → Dirty{None}.
            loaded.state.invalidate();
            // Any in-flight worker mesh of this chunk is now stale.
            loaded.rev = loaded.rev.wrapping_add(1);
            self.pending_dirty.set();
            self.pending_fresh.set();
            // The edited voxels are a changed light source/occluder: re-settle
            // this chunk (border diffs then fan the change to neighbours).
            self.light_worklist.insert(coord);
            self.light_pending.set();
        }
        // A block on a chunk face also changes that neighbour's exposed
        // faces — even when the edited chunk itself has no data (a remote
        // edit landing in an unloaded chunk must still invalidate a loaded,
        // still-drawn neighbour, or its culled border face becomes a hole).
        // `Face::touches` is the face-boundary encoding shared with the mesher.
        for face in Face::ALL {
            if face.touches(local) {
                self.mark_dirty(coord.step(face));
            }
        }
        previous
    }

    /// Mark a loaded chunk stale so the next stream remeshes it.
    fn mark_dirty(&mut self, coord: Coord) {
        if let Some(loaded) = self.chunks.get_mut(&coord) {
            // Same transition as `set_block`'s own chunk: carry the drawn mesh
            // forward as `prev` (Ready → Dirty{Some}, already-Dirty keeps it).
            loaded.state.invalidate();
            // The neighbour's border edit changed this chunk's exposed faces,
            // so any in-flight worker mesh of it is stale too.
            loaded.rev = loaded.rev.wrapping_add(1);
            self.pending_dirty.set();
            // In case the dirty pass drops it (missing neighbour data), the
            // fresh scan must be able to pick it back up later.
            self.pending_fresh.set();
            // A border edit can change this chunk's light directly (an emitter on
            // the shared face); re-settle it too.
            self.light_worklist.insert(coord);
            self.light_pending.set();
        }
    }

    /// Mark sections covering this voxel dirty at every active detail so they
    /// re-extract from the edit overlay. Sections span the full vertical domain
    /// (Y-independent), so edits outside [0, DOMAIN_H) don't touch any section.
    fn mark_dirty_sections_from_edit(&mut self, x: i32, y: i32, z: i32) {
        if !(0..super::section::DOMAIN_H).contains(&y) {
            return;
        }
        let details: Vec<u8> = self.section_pyramid.active_lods().map(|l| l.0).collect();
        for detail in details {
            let span = (super::section::SECTION_N as i32) << detail;
            let pos = super::section::SectionPos {
                detail,
                x: x.div_euclid(span),
                z: z.div_euclid(span),
            };
            self.dirty_sections.insert(pos);
        }
        self.pending_sections.set();
    }

    /// All edits as world coordinates and blocks for saving.
    pub fn edits(&self) -> impl Iterator<Item = ((i32, i32, i32), BlockId)> + '_ {
        self.edits.iter().flat_map(|(&coord, cells)| {
            cells.iter().map(move |(&index, &id)| {
                let (lx, ly, lz) = Chunk::local_of(index);
                // `local_of` splits a valid chunk index, so every component is
                // `< CHUNK_SIZE` — the checked ctor can't fail here.
                let local = Local::new(lx as u8, ly as u8, lz as u8)
                    .expect("chunk-local index is < CHUNK_SIZE");
                (BlockCoord::join(coord, local).to_tuple(), id)
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use voxel_engine::{MeshHandle, Pass};

    use crate::block::registry::AIR;
    use crate::coord::{ByPass, ChunkCoord};

    use super::*;
    use super::super::{ChunkMeshes, DEFAULT_SEED, Loaded};

    fn loaded(coord: ChunkCoord, state: MeshState, rev: u32) -> Loaded {
        Loaded {
            chunk: Arc::new(Chunk::from_uniform(coord.x, coord.y, coord.z, AIR)),
            state,
            rev,
            connectivity: None,
            light: None,
        }
    }

    fn meshes(handle: MeshHandle) -> ChunkMeshes {
        ChunkMeshes::from_upload_handles(ByPass::from_fn(|pass| {
            (pass == Pass::Opaque).then_some(handle)
        }))
        .expect("one opaque handle is a drawable chunk mesh")
    }

    #[test]
    fn meshing_input_invalidation_preserves_air_and_reseeds_only_non_air() {
        let mut world = World::with_config_lazy(DEFAULT_SEED, RenderConfig::default());
        let air = ChunkCoord::new(0, 0, 0);
        let ready = ChunkCoord::new(1, 0, 0);
        let dirty = ChunkCoord::new(2, 0, 0);
        let building = ChunkCoord::new(3, 0, 0);
        let stale_seed = ChunkCoord::new(99, 0, 0);
        let ready_handle = MeshHandle::from_raw_parts(7, 1);
        let dirty_handle = MeshHandle::from_raw_parts(8, 2);

        world
            .chunks
            .insert(air, loaded(air, MeshState::Air, u32::MAX));
        world.chunks.insert(
            ready,
            loaded(ready, MeshState::Ready(meshes(ready_handle)), 10),
        );
        world.chunks.insert(
            dirty,
            loaded(
                dirty,
                MeshState::Dirty {
                    prev: Some(meshes(dirty_handle)),
                },
                20,
            ),
        );
        world.chunks.insert(
            building,
            loaded(building, MeshState::NeedsMesh { building: true }, 30),
        );

        // Seed stale work/cache/upload state to exercise the whole transition,
        // including handle ownership and buffered-result rejection.
        world.mesh_worklist.extend([air, stale_seed]);
        world.pending_fresh.take();
        world.pending_dirty.set();
        world.upload_queue.push_back((
            ready,
            10,
            super::super::pipeline::MeshOutput::new(),
        ));
        world.draw_set_rev = 40;
        world.sync_draw_cache();
        assert_eq!(world.draw_cache.len(), 2);

        let mut freed = Vec::new();
        world.invalidate_near_mesh_states(|owned| freed.push(owned));

        assert!(matches!(&world.chunks[&air].state, MeshState::Air));
        assert_eq!(world.chunks[&air].rev, 0, "Air still rejects stale results");
        for (coord, expected_rev) in [(ready, 11), (dirty, 21), (building, 31)] {
            assert!(matches!(
                &world.chunks[&coord].state,
                MeshState::NeedsMesh { building: false }
            ));
            assert_eq!(world.chunks[&coord].rev, expected_rev);
            assert!(world.mesh_worklist.contains(&coord));
        }
        assert_eq!(world.mesh_worklist.len(), 3);
        assert!(!world.mesh_worklist.contains(&air));
        assert!(!world.mesh_worklist.contains(&stale_seed));
        assert!(world.pending_fresh.get());
        assert!(!world.pending_dirty.get());
        assert!(world.upload_queue.is_empty());

        assert_eq!(freed.len(), 2, "Ready and Dirty-prev each retire once");
        assert!(freed.iter().any(|owned| owned.draws(ready_handle)));
        assert!(freed.iter().any(|owned| owned.draws(dirty_handle)));
        assert_eq!(world.draw_set_rev, 41);
        assert_eq!(world.draw_cache_rev, world.draw_set_rev);
        assert!(world.draw_cache.is_empty());
    }

    #[test]
    fn all_air_invalidation_keeps_mesh_lane_and_draw_cache_idle() {
        let mut world = World::with_config_lazy(DEFAULT_SEED, RenderConfig::default());
        let air = ChunkCoord::new(0, 4, 0);
        let stale_seed = ChunkCoord::new(99, 4, 0);
        world.chunks.insert(air, loaded(air, MeshState::Air, 4));
        world.mesh_worklist.extend([air, stale_seed]);
        world.pending_fresh.set();
        world.pending_dirty.set();
        world.upload_queue.push_back((
            air,
            4,
            super::super::pipeline::MeshOutput::new(),
        ));
        world.draw_set_rev = 12;
        world.draw_cache_rev = 12;

        let mut retired = 0;
        world.invalidate_near_mesh_states(|_| retired += 1);

        assert!(matches!(&world.chunks[&air].state, MeshState::Air));
        assert_eq!(world.chunks[&air].rev, 5);
        assert_eq!(retired, 0);
        assert!(world.mesh_worklist.is_empty());
        assert!(!world.pending_fresh.get());
        assert!(!world.pending_dirty.get());
        assert!(world.upload_queue.is_empty());
        assert_eq!(world.draw_set_rev, 12);
        assert_eq!(world.draw_cache_rev, 12);
        assert!(world.draw_cache.is_empty());
    }
}
