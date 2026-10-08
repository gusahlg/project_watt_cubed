//! Player edits: block placement/breaking, the edit-overlay save iterator,
//! dirty-marking for remesh, and gameplay reactions. Code motion only: these
//! are `World` methods; the struct itself lives in `mod.rs`.

use crate::block::registry::BlockId;
use crate::coord::{BlockCoord, Face, Local};
use crate::sim::reactions::{self, CellStore, Mutation, Pos, ReactionScheduler};
use crate::space::FaceFrame;

use super::chunk::Chunk;
use super::{ColumnKey, Coord, Sky, World};

impl World {
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
        let generated = self.generator.voxel_at(x, y, z);
        // Gravity follows the matter: the cell's amount changes from what was really there (the
        // overlay, else generation — never `previous`, which reads unloaded chunks as air).
        let delta = self.registry.amount(id) as i32 - self.registry.amount(old_edit.unwrap_or(generated)) as i32;
        // A storage cell's matter sits where its chart embeds it.
        let at = self.seams.physical_cell(BlockCoord::new(x, y, z)).unwrap_or((x, y, z));
        self.gravity.record(at, delta);
        let sky = self.generator.sky(coord);
        let new_edit = if id == generated {
            if let Some(cells) = self.edits.get_mut(&coord) {
                cells.remove(&index);
                if cells.is_empty() {
                    self.edits.remove(&coord);
                }
            }
            None
        } else {
            let cells = self.edits.entry(coord).or_default();
            let first = cells.is_empty();
            cells.insert(index, id);
            if first {
                self.index_edited_chunk(coord, sky);
            }
            let span = self.edit_columns.entry((coord.x, coord.z)).or_insert([coord.y, coord.y]);
            *span = [span[0].min(coord.y), span[1].max(coord.y)];
            Some(id)
        };
        self.edit_generation += 1;
        // The window's ground may have moved with this edit.
        self.window.stale = true;
        // Skylight ceiling upkeep: a roof appearing above a column's
        // current ceiling raises it; the topmost edited roof disappearing
        // lowers it. Either way the cached window is stale, and every loaded
        // chunk at or below the edit seeds skylight from it — re-settle them
        // so a constructed roof actually darkens the world underneath.
        if let Sky::Axis(face) = sky {
            let (key, alt_chunk) = ColumnKey::of(face, coord);
            if let Some(ceiling) = self.ceilings.get(&key) {
                let frame = FaceFrame::new(face);
                let (lu, _, lv) = frame.index_to_local(lx, ly, lz);
                let alt = frame.cell_to_local((x, y, z)).1;
                let cell = ceiling.surface_at(lu, lv);
                let raises = new_edit.is_some_and(|id| self.registry.is_opaque(id)) && alt + 1 > cell;
                let lowers = old_edit.is_some_and(|id| self.registry.is_opaque(id)) && alt + 1 == cell;
                if raises || lowers {
                    self.ceilings.remove(&key);
                    if self.lighting {
                        let shadowed: Vec<Coord> = self
                            .column_chunks
                            .get(&key)
                            .map(|alts| {
                                alts.iter()
                                    .filter(|&&a| a <= alt_chunk)
                                    .map(|&a| key.chunk(a))
                                    .collect()
                            })
                            .unwrap_or_default();
                        for c in shadowed {
                            self.seed_light(c, super::LightSeed::Edit);
                        }
                        self.light_pending.set();
                    }
                }
            }
        }
        // Invalidate section to re-extract from overlay.
        if self.lod2 {
            self.mark_dirty_sections_from_edit(coord, x, y, z);
        }

        if let Some(loaded) = self.chunks.get_mut(&coord) {
            std::sync::Arc::make_mut(&mut loaded.chunk).set_index(index, id);
            // Editing this chunk's own voxels can open or seal an interior pocket,
            // so its connectivity is stale — invalidate it (the occlusion rebuild
            // recomputes lazily if the gate is active) and flag the visible set.
            // IMMEDIATE class: stale connectivity can hide a visible chunk.
            loaded.connectivity = None;
            self.occlusion_dirty.set();
            if self.occlusion_enabled() {
                self.conn_fill_queue.push_back(coord);
            }
            self.invalidate_mesh(coord);
            self.pending_fresh.set();
            // The edited voxels are a changed light source/occluder: re-settle
            // this chunk (border diffs then fan the change to neighbours).
            self.seed_light(coord, super::LightSeed::Edit);
            self.light_pending.set();
        }
        // A block on a chunk face also changes that neighbour's exposed
        // faces — even when the edited chunk itself has no data (a remote
        // edit landing in an unloaded chunk must still invalidate a loaded,
        // still-drawn neighbour, or its culled border face becomes a hole).
        // `Face::touches` is the face-boundary encoding shared with the mesher.
        for face in Face::ALL {
            if face.touches(local) {
                self.mark_dirty(self.neighbour(coord, face));
            }
        }
        previous
    }

    /// Enter chunk `coord`'s first edit in its column's roof index.
    fn index_edited_chunk(&mut self, coord: Coord, sky: Sky) {
        if let Sky::Axis(face) = sky {
            let (key, alt) = ColumnKey::of(face, coord);
            let alts = self.edit_column_chunks.entry(key).or_default();
            if !alts.contains(&alt) {
                alts.push(alt);
            }
        }
    }

    /// Invalidate a chunk's mesh into the SYNC edit path: state → `Dirty`
    /// (carrying the drawn mesh — see [`MeshState::invalidate`]), rev bump to
    /// strand in-flight builds, dirty-worklist membership, and the hint. THE
    /// one edit-class invalidation path, so `remesh_dirty` can drain the
    /// membership set instead of filtering every loaded chunk.
    pub(in crate::world) fn invalidate_mesh(&mut self, coord: Coord) {
        if let Some(loaded) = self.chunks.get_mut(&coord) {
            let was = loaded.state.is_building();
            loaded.state.invalidate();
            super::adjust_count(&mut self.building_meshes, was, false);
            loaded.rev = loaded.rev.wrapping_add(1);
            self.dirty_worklist.insert(coord);
            self.pending_dirty.set();
        }
    }

    /// Mark a loaded chunk stale so the next stream remeshes it.
    fn mark_dirty(&mut self, coord: Coord) {
        if self.chunks.contains_key(&coord) {
            // Same transition as `set_block`'s own chunk: carry the drawn mesh
            // forward as `prev`, bump rev (the neighbour's border edit changed
            // this chunk's exposed faces, so in-flight meshes are stale too).
            self.invalidate_mesh(coord);
            // In case the dirty pass drops it (missing neighbour data), the
            // fresh scan must be able to pick it back up later.
            self.pending_fresh.set();
            // A border edit can change this chunk's light directly (an emitter on
            // the shared face); re-settle it too.
            self.seed_light(coord, super::LightSeed::Edit);
            self.light_pending.set();
        }
    }

    /// Mark sections covering this voxel dirty at every active detail so they
    /// re-extract from the edit overlay. A storage cell dirties its chart section
    /// (storage +Y is the chart's up). A cube face maps the cell through the face
    /// frame. A flat world keeps the `[0, 512)` window.
    fn edit_face_cell(&self, x: i32, y: i32, z: i32) -> Option<(u16, Face, i32, i32, i32)> {
        let s = [i64::from(x), i64::from(y), i64::from(z)];
        for (index, atlas) in self.generator.atlases().iter().enumerate() {
            if let Some(g) = atlas.grid {
                if atlas.locate(s).is_none() {
                    continue;
                }
                let p = [
                    s[0] - g.origin[0] + g.ref_min[0],
                    s[1] - g.origin[1] + g.ref_min[1],
                    s[2] - g.origin[2] + g.ref_min[2],
                ];
                let (Ok(px), Ok(py), Ok(pz)) = (i32::try_from(p[0]), i32::try_from(p[1]), i32::try_from(p[2])) else {
                    return None;
                };
                let q = voxel_engine::DVec3::new(p[0] as f64 + 0.5, p[1] as f64 + 0.5, p[2] as f64 + 0.5);
                let face = Face::from_dominant(q - atlas.centre);
                let (u, a, v) = FaceFrame::new(face).cell_to_local((px, py, pz));
                return Some((g.body, face, u, v, a));
            }
            if atlas.locate(s).is_some() {
                let body = super::section::CHART_BODY_BASE + index as u16;
                return Some((body, Face::PosY, x, z, y));
            }
        }
        let Some(cosmos) = self.generator.cosmos() else {
            if !(0..super::section::DOMAIN_H).contains(&y) {
                return None;
            }
            return Some((0, Face::PosY, x, z, y));
        };
        let p = voxel_engine::DVec3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5);
        let body = cosmos.body_at(p)?;
        if !matches!(body.shape, super::terrain::cosmos::Shape::Cube { .. }) {
            return None;
        }
        let face = Face::from_dominant(p - body.centre_f());
        let (u, a, v) = FaceFrame::new(face).cell_to_local((x, y, z));
        Some((body.id, face, u, v, a))
    }

    fn edit_in_window(&self, pos: super::section::SectionPos, a: i32) -> bool {
        let Some((lo, hi)) = self.generator.surface_bounds(pos.body, pos.face, pos.min_x(), pos.min_z(), pos.span()) else {
            return false;
        };
        let (wlo, whi) = super::section::sample_window(lo, hi, pos.cell_size());
        (wlo..whi).contains(&a)
    }

    fn mark_dirty_sections_from_edit(&mut self, chunk: Coord, x: i32, y: i32, z: i32) {
        let Some((body, face, u, v, a)) = self.edit_face_cell(x, y, z) else { return };
        let details: Vec<_> = self.section_pyramid.active_lods().collect();
        let mut any = false;
        for detail in details {
            let span = super::section::section_span(detail);
            let pos = super::section::SectionPos {
                body,
                face,
                detail,
                x: u.div_euclid(span),
                z: v.div_euclid(span),
            };
            if !self.edit_in_window(pos, a) {
                continue;
            }
            any = true;
            self.dirty_sections.insert(pos);
            // The heightmip edit overlay (streaming.rs `refresh_section_overlay`)
            // keys its cache on this same per-section counter, so it re-derives
            // exactly the cells this edit could have changed.
            *self.section_edit_rev.entry(pos).or_insert(0) += 1;
            // Index the edited chunk under every footprint that contains it,
            // and queue the exact overlay re-derivation this edit requires.
            self.section_edit_chunks.entry(pos).or_default().insert(chunk);
            self.section_overlay_dirty.insert(pos);
        }
        if any {
            self.pending_sections.set();
        }
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

    /// Cheap autosave snapshot of the overlay: a HashMap clone, no spec strings.
    pub(crate) fn clone_edit_overlay(
        &self,
    ) -> super::FastMap<Coord, super::FastMap<usize, BlockId>> {
        self.edits.clone()
    }

    /// Contact or material state changed at a cell (placed, removed, configuration changed): wake
    /// its six contacts. No-op when this instance is not the authority (a client connected to a
    /// server).
    pub fn note_cell_changed(&mut self, x: i32, y: i32, z: i32) {
        if self.reactions_authority {
            self.reactions.wake_cell((x, y, z));
        }
    }

    /// A block moved: wake the contacts of both locations.
    #[allow(dead_code)] // no gameplay moves blocks yet; machines and physics will
    pub fn note_block_moved(&mut self, from: (i32, i32, i32), to: (i32, i32, i32)) {
        if self.reactions_authority {
            self.reactions.wake_move(from, to);
        }
    }

    /// One budgeted scheduler turn. Empty when this instance is not the authority.
    pub fn tick_reactions(&mut self) -> Vec<Mutation> {
        if !self.reactions_authority {
            return Vec::new();
        }
        let mut sched = std::mem::take(&mut self.reactions);
        let out = sched.tick(self, reactions::Budget::DEFAULT);
        self.reactions = sched;
        out
    }

    /// Resume saved reaction work (`(age, contact)` in processing order). Authority only.
    pub fn restore_reactions(&mut self, pending: &[(u32, reactions::Contact)]) {
        if self.reactions_authority {
            self.reactions.restore(pending);
        }
    }

    /// Single-player and the hosting server are the authority; a connected client is not.
    pub fn set_reactions_authority(&mut self, yes: bool) {
        self.reactions_authority = yes;
        if !yes {
            self.reactions = ReactionScheduler::new();
        }
    }

    /// Scheduler counters for the console and bench gauges.
    pub fn reactions(&self) -> &ReactionScheduler {
        &self.reactions
    }
}

impl CellStore for World {
    /// Loaded chunk, else the edit overlay, else the generator: the infinite world is defined
    /// without loading, so a cascade reads the same cells whatever the streaming state (the server's
    /// store does the same).
    fn block_at(&self, pos: Pos) -> Option<BlockId> {
        #[cfg(test)]
        crate::alloc_count::note_cell_read();
        let (chunk, local) = BlockCoord::new(pos.0, pos.1, pos.2).split();
        if let Some(loaded) = self.chunks.get(&chunk) {
            return Some(loaded.chunk.get_local(local.lx(), local.ly(), local.lz()));
        }
        let index = Chunk::index(local.lx(), local.ly(), local.lz());
        if let Some(id) = self.edits.get(&chunk).and_then(|cells| cells.get(&index)) {
            return Some(*id);
        }
        Some(self.generator.voxel_at(pos.0, pos.1, pos.2))
    }

    fn set_block(&mut self, pos: Pos, id: BlockId) -> Option<BlockId> {
        // `World::set_block` reports an unloaded cell as AIR; the scheduler wants the true previous
        // material (overlay or generated), which the read above defines.
        let prev = CellStore::block_at(self, pos)?;
        #[cfg(test)]
        crate::alloc_count::note_cell_write();
        World::set_block(self, pos.0, pos.1, pos.2, id);
        Some(prev)
    }

    fn registry(&self) -> &crate::block::BlockRegistry {
        &self.registry
    }

    fn registry_mut(&mut self) -> &mut crate::block::BlockRegistry {
        &mut self.registry
    }
}

#[cfg(test)]
impl World {
    /// Pack `n` overlay entries without going through [`World::set_block`] —
    /// used by the autosave snapshot/encode timing probe.
    pub(crate) fn test_fill_overlay(&mut self, n: usize, id: BlockId) {
        use super::chunk::CHUNK_VOLUME;
        let mut placed = 0;
        let mut cx = 0i32;
        while placed < n {
            let coord = Coord::new(cx, 20, 0);
            let inner = self.edits.entry(coord).or_default();
            let room = CHUNK_VOLUME.min(n - placed);
            for index in 0..room {
                inner.insert(index, id);
            }
            placed += room;
            self.edit_generation += room as u64;
            let sky = self.generator.sky(coord);
            self.index_edited_chunk(coord, sky);
            cx += 1;
        }
    }
}
