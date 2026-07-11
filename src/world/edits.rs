//! Player edits and world-leaving cleanup: block placement/breaking, the
//! edit-overlay save iterator, dirty-marking for remesh, freeing meshes, and
//! the render-distance setting. Code motion only: these are `World` methods;
//! the struct itself lives in `mod.rs`.

use voxel_engine::Engine;

use crate::block::registry::BlockId;
use crate::coord::{BlockCoord, Face, Local};

use super::chunk::Chunk;
use super::lod::{Lod, Tile};
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
            self.view = super::ViewVolume::cube(radius);
            // The pyramid's innermost ring begins where the full-res box ends, so
            // its `unit` tracks the render distance in metres. Droop needs no
            // recalibration — it depends only on the LOD cell grids, not `unit`.
            self.pyramid.unit = (radius * super::chunk::CHUNK_SIZE as i32) as f32;
            // Invalidate the centre so the next stream reruns the full
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

    /// Whether cross-chunk lighting is currently enabled.
    pub fn lighting(&self) -> bool {
        self.lighting
    }

    /// Toggle cross-chunk lighting. On a real change, drops every mesh and
    /// re-scans from scratch (via [`free_meshes`](Self::free_meshes)) so the next
    /// [`stream`](Self::stream) rebuilds them with — or without — settled light.
    /// A no-op when the value is unchanged, so it is cheap to push every frame.
    pub fn set_lighting(&mut self, on: bool, eng: &mut Engine) {
        if on == self.lighting {
            return;
        }
        self.lighting = on;
        self.free_meshes(eng);
    }

    /// Free every chunk's GPU mesh and reset every chunk to `NeedsMesh` — used
    /// when leaving a world. The voxel data stays; a later
    /// [`stream`](Self::stream) rebuilds the meshes from scratch. (Resetting to
    /// `NeedsMesh` — rather than back to `Air` for born-air chunks — matches the
    /// old unconditional `meshed = false`; the next scan re-derives `Air`.)
    pub fn free_meshes(&mut self, eng: &mut Engine) {
        for loaded in self.chunks.values_mut() {
            loaded.retire(MeshState::NeedsMesh { building: false }, eng);
        }
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
        // Far LOD tiles belong to the world we are leaving; free them too.
        for (_, state) in self.tiles.drain() {
            state.free(eng);
        }
        self.tile_upload_queue.clear();
        self.pending_tiles.take();
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
        self.edits.entry(coord).or_default().insert(index, id);
        // The edit also invalidates the far LOD tiles that cover this voxel (at
        // every active pyramid level), so they remesh from the overlay.
        self.mark_dirty_tiles_from_edit(x, y, z);

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

    /// Mark the far tile that contains world voxel `(x, y, z)` dirty at every
    /// active pyramid level, PLUS the cardinal-neighbour tile whenever the edit
    /// lies within one CELL of a tile face — a border cell feeds the neighbour
    /// tile's padded shell, so that neighbour must remesh too. World→tile is
    /// `div_euclid(lod.span())`; the within-a-cell test is on the in-tile remainder.
    fn mark_dirty_tiles_from_edit(&mut self, x: i32, y: i32, z: i32) {
        // Snapshot the active levels first: `active_lods` borrows `self.pyramid`,
        // and marking borrows `self.dirty_tiles` mutably (two entries: Lod2, Lod4).
        let lods: Vec<Lod> = self.pyramid.active_lods().collect();
        for lod in lods {
            let (span, cell) = (lod.span(), lod.cell());
            let (tx, ty, tz) = (x.div_euclid(span), y.div_euclid(span), z.div_euclid(span));
            self.dirty_tiles.mark(Tile { lod, x: tx, y: ty, z: tz });
            let (rx, ry, rz) = (x.rem_euclid(span), y.rem_euclid(span), z.rem_euclid(span));
            let mut border = |dx: i32, dy: i32, dz: i32| {
                self.dirty_tiles.mark(Tile { lod, x: tx + dx, y: ty + dy, z: tz + dz });
            };
            if rx < cell {
                border(-1, 0, 0);
            }
            if rx >= span - cell {
                border(1, 0, 0);
            }
            if ry < cell {
                border(0, -1, 0);
            }
            if ry >= span - cell {
                border(0, 1, 0);
            }
            if rz < cell {
                border(0, 0, -1);
            }
            if rz >= span - cell {
                border(0, 0, 1);
            }
        }
    }

    /// Project the edit overlay onto one tile — every edited chunk whose
    /// coord lies in the tile's chunk span, in `GenerateColumn`-shaped form. The
    /// far-tile mesher's coarse-cell reducer replays these onto the downsample.
    pub(in crate::world) fn edits_for_tile(&self, tile: Tile) -> Vec<(Coord, Vec<(usize, BlockId)>)> {
        let cps = tile.lod.chunks_per_side();
        let (x0, y0, z0) = (tile.x * cps, tile.y * cps, tile.z * cps);
        self.edits
            .iter()
            .filter(|(c, _)| {
                (x0..x0 + cps).contains(&c.x)
                    && (y0..y0 + cps).contains(&c.y)
                    && (z0..z0 + cps).contains(&c.z)
            })
            .map(|(&c, cells)| (c, cells.iter().map(|(&i, &b)| (i, b)).collect()))
            .collect()
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
