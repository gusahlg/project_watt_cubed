//! Unloading chunks past the unload box, and the turn-back hold near a seam.

use super::*;

impl World {
    /// `c` is still close enough that walking back across a seam would show its column. The far
    /// edge of the near square is a view-radius inland; a view-radius on the next chart puts that
    /// column `3 * horizontal + 1` away, counting the two edge chunks.
    /// Within the skirt on every axis: a turn-back can bring the column back, and a dig or a climb
    /// cannot pile up layers past it.
    fn within_skirt(&self, center: Coord, c: Coord) -> bool {
        let up = self.live_up().unwrap_or(Face::PosY);
        let folded = self.fold.fold(c);
        folded.across(center, up) <= self.skirt() && folded.along(center, up) <= self.skirt()
    }

    pub(super) fn skirt(&self) -> i32 {
        self.view.horizontal * 3 + 1
    }

    /// `center` stands on a chart within the skirt of one of its edges, where a crossing and a
    /// turn-back can bring the trailing rows back before a far section draws them.
    pub(super) fn near_a_seam(&self, center: Coord) -> bool {
        let Some(seat) = self.seams.chart_seat(center) else { return false };
        let cs = CHUNK_SIZE as i64;
        let (x, z) = (i64::from(center.x), i64::from(center.z));
        let inland = (x - seat.lo[0] / cs)
            .min(seat.hi[0] / cs - 1 - x)
            .min(z - seat.lo[2] / cs)
            .min(seat.hi[2] / cs - 1 - z);
        inland <= i64::from(self.skirt())
    }

    /// `c` draws something, is inside the skirt, and either lies on a neighbour chart or the centre
    /// is near a seam: it starts waiting for a turn-back. Air and buried rock draw nothing, so their
    /// unloading can never show as bare ground. Away from every seam nothing waits: the
    /// unload box and the far field already cover a turn-back there. Once waiting, a chunk stays
    /// (through a crossing, when it may become home) until it is inside the unload box again or
    /// past the skirt.
    fn keeps_for_far(&self, center: Coord, c: Coord, near_seam: bool) -> bool {
        (near_seam || self.fold.fold(c) != c)
            && self.chunks.get(&c).is_some_and(|l| l.state.settled() && l.state.live_meshes().is_some())
            && self.within_skirt(center, c)
    }

    /// Coords in the previous unload box that have left `new_box`, or every
    /// loaded chunk past `new_box` when there is no previous box (first pass
    /// or a radius change). Spawn-slab chunks stay.
    pub(in crate::world) fn unload_leaving(&self, new_box: ChunkBox) -> Vec<Coord> {
        let keep_spawn = |coord| self.spawn_slab.is_some_and(|slab| self.view_contains(slab, coord));
        match self.prev_unload_box {
            Some(prev) => self
                .view_shell(prev, new_box)
                .filter(|&coord| !keep_spawn(coord) && self.chunks.contains_key(&coord))
                .collect(),
            None => self
                .chunks
                .keys()
                .copied()
                .filter(|&coord| !self.view_contains(new_box, coord) && !keep_spawn(coord))
                .collect(),
        }
    }

    /// Remove chunk `coord` and every per-chunk entry kept beside it, returning the mesh state and
    /// cage the GPU still holds for it. Seeds of a chunk that is gone are garbage: its next load
    /// seeds afresh. Left in, they pile up in the clamped last ring during flight (never visited,
    /// re-bucketed on every centre move).
    pub(in crate::world) fn forget_chunk(
        &mut self,
        coord: Coord,
    ) -> Option<(MeshState, Option<voxel_engine::CageHandle>)> {
        self.dirty_worklist.remove(&coord);
        self.light_worklist.remove(&coord);
        self.light_owed.remove(&coord);
        self.mesh_worklist.remove(&coord);
        self.light_terminal.remove(&coord);
        self.light_gate.forget(coord);
        self.remesh_stats.forget(coord);
        // Column layers: the last chunk out drops the cached ceiling.
        if let Sky::Axis(face) = self.generator.sky(coord) {
            let (key, alt) = ColumnKey::of(face, coord);
            if let Some(ys) = self.column_chunks.get_mut(&key) {
                if let Some(i) = ys.iter().position(|&y| y == alt) {
                    ys.remove(i);
                }
                if ys.is_empty() {
                    self.column_chunks.remove(&key);
                    self.ceilings.remove(&key);
                }
            }
        }
        let loaded = self.chunks.remove(&coord)?;
        super::adjust_count(&mut self.building_meshes, loaded.state.is_building(), false);
        Some((loaded.state, self.cages.remove(&coord)))
    }

    /// Free chunks past the unload box, releasing their GPU meshes.
    pub(super) fn unload_far(&mut self, center: Coord, eng: &mut Engine) {
        self.unload_far_with(center, |state, cage| {
            state.free_owned(eng);
            if let Some(cage) = cage {
                eng.free_cage(cage);
            }
        });
    }

    /// [`unload_far`](Self::unload_far) with the GPU release passed in: `free` gets each removed
    /// chunk's mesh state and cage.
    pub(super) fn unload_far_with(
        &mut self,
        center: Coord,
        mut free: impl FnMut(MeshState, Option<voxel_engine::CageHandle>),
    ) {
        let unload = self.unload_box(center);
        // Collect-then-remove instead of `retain`: freeing borrows the caller's
        // engine, which can't be borrowed inside a retain closure over `self.chunks`.
        let mut far = self.unload_leaving(unload);
        if let Some((kept, fold)) = self.retired {
            far.retain(|&c| !kept.contains(fold.fold(c)));
        }
        // Keep a settled chunk across a seam out to the turn-back skirt. A section on screen is not
        // a reason to drop it: that section unloads as the player walks on, and the column is bare
        // on the way back.
        // Waiting chunks leave once back inside the unload box (they stay loaded) or past the skirt.
        // With the far field off, or off every chart, nothing waits: the rest unload now.
        let holding = self.lod2 && !self.fold.is_identity();
        if !self.far_wait.is_empty() {
            let mut waiting = std::mem::take(&mut self.far_wait);
            waiting.retain(|&c| {
                if !self.chunks.contains_key(&c) || self.view_contains(unload, c) {
                    return false;
                }
                if holding && self.within_skirt(center, c) {
                    return true;
                }
                far.push(c);
                false
            });
            self.far_wait = waiting;
        }
        if holding && !far.is_empty() {
            let near_seam = self.near_a_seam(center);
            let mut waiting = std::mem::take(&mut self.far_wait);
            far.retain(|&c| {
                if waiting.contains(&c) || self.keeps_for_far(center, c, near_seam) {
                    waiting.insert(c);
                    false
                } else {
                    true
                }
            });
            self.far_wait = waiting;
        }
        self.prev_unload_box = Some(unload);
        // A removed chunk changes what the BFS can reach — topology class.
        self.occlusion_topo_dirty.raise(!far.is_empty());
        for &coord in &far {
            // Free the mesh handle (Ready or Dirty); Air/NeedsMesh own none.
            // `far` holds loaded chunks only (see `unload_leaving`).
            if let Some((state, cage)) = self.forget_chunk(coord) {
                free(state, cage);
            }
        }
        // Settled grids still queued for removed chunks describe the world
        // being unloaded: applying one to a LATER re-generated chunk would
        // publish stale light past every epoch check. Drop them and release
        // the claims they were carrying (one retain pass, not per-coord scans).
        if !self.light_apply_queue.is_empty() && !far.is_empty() {
            let removed: FastSet<Coord> = far.iter().copied().collect();
            let inflight = &mut self.light_inflight;
            self.light_apply_queue.retain(|(c, _)| {
                let gone = removed.contains(c);
                if gone {
                    inflight.remove(c);
                }
                !gone
            });
        }
        // (Ceilings for fully-unloaded columns dropped by the refcount above;
        // the heightmap is pure, so a re-entered column simply recomputes once.)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An unloaded chunk leaves no mark behind: the light gate's timers, degraded and dirty sets,
    /// and the terminal set hold no unloaded coord.
    #[test]
    fn unload_forgets_light_gate_marks() {
        let mut world = World::generate();
        world.lod2 = false;
        world.set_view_distances(2, 2);
        let cy = world.generator.height(0, 0).div_euclid(CHUNK_SIZE as i32);
        let marked: Vec<Coord> = world.chunks.keys().copied().collect();
        for &c in &marked {
            world.light_gate.note_blocked(c);
            world.light_gate.degraded.insert(c);
            world.light_gate.dirty.insert(c, crate::sched::now());
            world.light_terminal.insert(c);
        }
        world.unload_far_with(Coord::new(1_000, cy, 0), |_, _| {});
        assert!(world.chunks.len() < marked.len(), "the move unloaded chunks");
        let gate = &world.light_gate;
        let held = |c: &Coord| {
            gate.blocked_since.contains_key(c)
                || gate.degraded.contains(c)
                || gate.dirty.contains_key(c)
                || world.light_terminal.contains(c)
        };
        let stale: Vec<&Coord> = marked.iter().filter(|c| !world.chunks.contains_key(c) && held(c)).collect();
        assert!(stale.is_empty(), "unloaded chunks still marked: {stale:?}");
    }
}
