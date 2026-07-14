//! Player edits and world-leaving cleanup: block placement/breaking, the
//! edit-overlay save iterator, dirty-marking for remesh, freeing meshes, and
//! the render-distance setting. Code motion only: these are `World` methods;
//! the struct itself lives in `mod.rs`.

use voxel_engine::Engine;

use crate::block::registry::BlockId;
use crate::coord::{BlockCoord, Face, Local};

use super::chunk::Chunk;
use super::generation::TerrainGenerator;
use super::{Coord, MeshState, VIEW_RADIUS_RANGE, World};

impl World {
    /// Current render distance in chunk rings.
    pub fn view_radius(&self) -> i32 {
        self.view.horizontal
    }

    /// Change the render distance (clamped to 3..=10). Marks streaming dirty so
    /// the next [`stream`](Self::stream) unloads past the new radius or resumes
    /// meshing out to it.
    pub fn set_view_radius(&mut self, radius: i32) {
        let radius = radius.clamp(*VIEW_RADIUS_RANGE.start(), *VIEW_RADIUS_RANGE.end());
        if radius != self.view.horizontal {
            let shrunk = radius < self.view.horizontal;
            self.view = super::ViewVolume::view(radius);
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

    /// Set render lanes (occlusion/lod2). Entry-only; mesh teardown not needed.
    pub fn set_render_lanes(&mut self, occlusion: bool, lod2: bool) {
        self.occlusion_forced = occlusion;
        self.lod2 = lod2;
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
        if on == self.ao {
            return;
        }
        self.ao = on;
        self.tables_epoch = self.tables_epoch.wrapping_add(1);
        self.free_meshes(eng);
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

    /// Free every chunk's GPU mesh and reset every chunk to `NeedsMesh` — used
    /// when leaving a world. The voxel data stays; a later
    /// [`stream`](Self::stream) rebuilds the meshes from scratch. (Resetting to
    /// `NeedsMesh` — rather than back to `Air` for born-air chunks — matches the
    /// old unconditional `meshed = false`; the next scan re-derives `Air`.)
    pub fn free_meshes(&mut self, eng: &mut Engine) {
        for loaded in self.chunks.values_mut() {
            // Any worker mesh captured before this reset must not be accepted if
            // it lands after the next stream establishes a new centre.
            loaded.rev = loaded.rev.wrapping_add(1);
            loaded.retire(MeshState::NeedsMesh { building: false }, eng);
        }
        // Every drawn mesh is gone: the drawable set is empty until remeshes land.
        self.draw_set_rev += 1;
        // Every chunk is now `NeedsMesh`, so the `Dirty` fiber is empty; drop the
        // stale hint (a raised `pending_dirty` would just scan an empty fiber).
        self.pending_dirty.take();
        // Drop the pipeline bookkeeping too: buffered worker meshes are for a
        // world we are leaving, and in-flight jobs may re-run from scratch if
        // we come back. Results still flying land against the invalidated
        // centre below and are dropped by the range/rev checks — at worst a
        // coord gets generated or meshed twice, never wrongly.
        self.generating.clear();
        self.upload_queue.clear();
        // Sections belong to the world being left.
        for (_, state) in self.sections.drain() {
            state.free(eng);
        }
        self.section_upload_queue.clear();
        self.pending_sections.take();
        self.dirty_sections.clear();
        self.section_visible.clear();
        self.center = None;
        // Every chunk is back to `NeedsMesh`; re-seed the mesh lane's worklist so
        // the next stream rebuilds them (the worklist is the fresh-mesh index now).
        self.mesh_worklist = self.chunks.keys().copied().collect();
        self.pending_fresh.set();
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
        let old_edit = self.edits.get(&coord).and_then(|cells| cells.get(&index)).copied();
        let generated = self.generator.block_at(x, y, z, self.generator.height(x, z));
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
