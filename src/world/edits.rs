//! Player edits and world-leaving cleanup: block placement/breaking, the
//! edit-overlay save iterator, dirty-marking for remesh, freeing meshes, and
//! the render-distance setting. Code motion only: these are `World` methods;
//! the struct itself lives in `mod.rs`.

use voxel_engine::Engine;

use crate::block::registry::BlockId;

use super::chunk::{CHUNK_SIZE, Chunk};
use super::{Coord, NO_CENTER, VIEW_RADIUS_RANGE, World};

impl World {
    /// Current render distance in chunk rings.
    pub fn view_radius(&self) -> i32 {
        self.view_radius
    }

    /// Change the render distance (clamped to 3..=10). Marks streaming dirty so
    /// the next [`stream`](Self::stream) unloads past the new radius or resumes
    /// meshing out to it.
    pub fn set_view_radius(&mut self, radius: i32) {
        let radius = radius.clamp(*VIEW_RADIUS_RANGE.start(), *VIEW_RADIUS_RANGE.end());
        if radius != self.view_radius {
            let shrunk = radius < self.view_radius;
            self.view_radius = radius;
            // Invalidate the centre so the next stream reruns the full
            // unload/ensure/scan pass even though the player hasn't moved.
            self.center = NO_CENTER;
            self.pending_fresh = true;
            // On shrink, meshes between the new radius and the (also shrunk)
            // unload ring would otherwise stay drawn until the player moves;
            // flag them so the next stream frees them immediately.
            self.radius_shrunk = shrunk;
            // In-flight worker jobs are NOT cancelled: results now outside the
            // radius are dropped by the range checks when they drain.
        }
    }

    /// Free every chunk's GPU mesh and clear the meshed flags — used when
    /// leaving a world. The voxel data stays; a later [`stream`](Self::stream)
    /// would rebuild the meshes from scratch.
    pub fn free_meshes(&mut self, eng: &mut Engine) {
        for loaded in self.chunks.values_mut() {
            if let Some(handle) = loaded.mesh.take() {
                eng.free_mesh(handle);
            }
            loaded.meshed = false;
        }
        self.dirty.clear();
        // Drop the pipeline bookkeeping too: buffered worker meshes are for a
        // world we are leaving, and in-flight jobs may re-run from scratch if
        // we come back. Results still flying land against the invalidated
        // centre below and are dropped by the range/rev checks — at worst a
        // coord gets generated or meshed twice, never wrongly.
        self.in_flight.clear();
        self.upload_queue.clear();
        self.center = NO_CENTER;
        self.pending_fresh = true;
    }

    /// Replace the block at a world coordinate, recording the change in the edit
    /// overlay (so it survives streaming and can be saved) and marking the affected
    /// chunk — and any neighbour across a shared face — for remeshing. Returns the
    /// block that was there.
    pub fn set_block(&mut self, x: i32, y: i32, z: i32, id: BlockId) -> BlockId {
        let coord = Self::chunk_of(x, y, z);
        let s = CHUNK_SIZE as i32;
        let lx = x.rem_euclid(s) as usize;
        let ly = y.rem_euclid(s) as usize;
        let lz = z.rem_euclid(s) as usize;
        let index = Chunk::index(lx, ly, lz);

        let previous = self.block_at(x, y, z);
        self.edits.entry(coord).or_default().insert(index, id);

        if let Some(loaded) = self.chunks.get_mut(&coord) {
            loaded.chunk.set_index(index, id);
            loaded.meshed = false;
            // Any in-flight worker mesh of this chunk is now stale.
            loaded.rev = loaded.rev.wrapping_add(1);
            self.dirty.insert(coord);
            self.pending_fresh = true;
        }
        // A block on a chunk face also changes that neighbour's exposed
        // faces — even when the edited chunk itself has no data (a remote
        // edit landing in an unloaded chunk must still invalidate a loaded,
        // still-drawn neighbour, or its culled border face becomes a hole).
        let (cx, cy, cz) = coord;
        if lx == 0 {
            self.mark_dirty((cx - 1, cy, cz));
        }
        if lx == CHUNK_SIZE - 1 {
            self.mark_dirty((cx + 1, cy, cz));
        }
        if ly == 0 {
            self.mark_dirty((cx, cy - 1, cz));
        }
        if ly == CHUNK_SIZE - 1 {
            self.mark_dirty((cx, cy + 1, cz));
        }
        if lz == 0 {
            self.mark_dirty((cx, cy, cz - 1));
        }
        if lz == CHUNK_SIZE - 1 {
            self.mark_dirty((cx, cy, cz + 1));
        }
        previous
    }

    /// Mark a loaded chunk stale so the next stream remeshes it.
    fn mark_dirty(&mut self, coord: Coord) {
        if let Some(loaded) = self.chunks.get_mut(&coord) {
            loaded.meshed = false;
            // The neighbour's border edit changed this chunk's exposed faces,
            // so any in-flight worker mesh of it is stale too.
            loaded.rev = loaded.rev.wrapping_add(1);
            self.dirty.insert(coord);
            // In case the dirty pass drops it (missing neighbour data), the
            // fresh scan must be able to pick it back up later.
            self.pending_fresh = true;
        }
    }

    /// Every recorded edit as `((x, y, z), block)`, for saving. Coordinates
    /// are absolute — the save format is independent of the chunk keying.
    pub fn edits(&self) -> impl Iterator<Item = ((i32, i32, i32), BlockId)> + '_ {
        self.edits.iter().flat_map(|(&(cx, cy, cz), cells)| {
            cells.iter().map(move |(&index, &id)| {
                let (lx, ly, lz) = Chunk::local_of(index);
                let x = cx * CHUNK_SIZE as i32 + lx as i32;
                let y = cy * CHUNK_SIZE as i32 + ly as i32;
                let z = cz * CHUNK_SIZE as i32 + lz as i32;
                ((x, y, z), id)
            })
        })
    }
}
