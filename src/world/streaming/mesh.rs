//! Chunk meshing on the main thread: the edit remesh, mesh snapshots, and mesh seeds.

use super::*;
use crate::world::{light, mesh};

impl World {
    /// Remesh edited (`Dirty`) chunks synchronously, budgeted, nearest first —
    /// carved out from the async mesh lane so a broken block never lags a frame.
    /// Reports `Progress::Partial { remaining }` when more `Dirty` chunks are
    /// queued than the per-frame [`DIRTY_BUDGET`] (they stay `Dirty`, drained
    /// next frame).
    pub(in crate::world) fn remesh_dirty(&mut self, eng: &mut Engine) -> Progress {
        // Gated by pending_dirty so idle frames pay one flag check.
        if !self.pending_dirty.take() {
            return Progress::Idle;
        }
        let Some(center) = self.center else {
            return Progress::Idle;
        };
        // Drain the MAINTAINED membership set (entries whose chunk moved on —
        // unloaded, or resolved by another path — drop right here), instead of
        // filtering every loaded chunk each frame the hint is up: during a
        // light flood that was an O(world) iteration per frame.
        let chunks = &self.chunks;
        self.dirty_worklist
            .retain(|c| chunks.get(c).is_some_and(|l| l.state.is_dirty()));
        let mut dirty: Vec<Coord> = self.dirty_worklist.iter().copied().collect();
        let (up, fold) = (self.live_up(), self.fold);
        dirty.sort_by_key(|&coord| Self::order(fold.fold(coord), center, up));
        // Leftovers past the budget stay `Dirty` (still in the fiber); re-arm
        // the hint so the next frame drains them.
        let remaining = dirty.len().saturating_sub(DIRTY_BUDGET);
        if remaining > 0 {
            self.pending_dirty.set();
        }
        for coord in dirty.into_iter().take(DIRTY_BUDGET) {
            self.dirty_worklist.remove(&coord);
            // No neighbour-data gate: edited chunks remesh even with missing
            // neighbour data (mesher reads them as air).
            self.mesh_chunk(coord, eng);
        }
        if remaining > 0 {
            Progress::Partial {
                remaining: remaining as u32,
            }
        } else {
            Progress::Idle
        }
    }

    /// Snapshot for mesh job: chunk storage, neighbour shell, solidity table, and rev.
    pub(in crate::world) fn snapshot(
        &self,
        coord: Coord,
        degraded: bool,
    ) -> (u32, pipeline::ChunkSnapshot) {
        let loaded = &self.chunks[&coord];
        // One 3×3×3 lookup feeds both the voxel shell and the light shell.
        let nhood = self.loaded_neighbourhood(coord);
        let fallback = (self.lighting && degraded).then(light::LightGrid::open_sky);
        let mut padded = mesh::Padded::capture(|dx, dy, dz| {
            Self::nhood_at(&nhood, dx, dy, dz).map(|l| &*l.chunk)
        });
        self.seam_halo(coord, &mut padded);
        (
            loaded.rev,
            pipeline::ChunkSnapshot {
                padded,
                uniform: loaded.chunk.uniform(),
                // Lighting off omits the 18³ shell entirely — the mesher's
                // unlit path reads constant full light instead. `open_sky` is
                // Uniform (task 05): the degraded fallback does not allocate.
                light: self.lighting.then(|| {
                    let mut shell = light::PaddedLight::capture(|dx, dy, dz| {
                        Self::nhood_at(&nhood, dx, dy, dz)
                            .and_then(|l| l.light.as_ref())
                            .or(fallback.as_ref())
                    });
                    self.seam_light_halo(coord, &mut shell, fallback.as_ref());
                    shell
                }),
                tables: self.tables.get(),
            },
        )
    }

    /// The 3×3×3 neighbourhood of `coord`, dx-fast then dy then dz.
    fn loaded_neighbourhood(&self, coord: Coord) -> [Option<&Loaded>; 27] {
        let mut nhood = [None; 27];
        let mut i = 0;
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    nhood[i] = self
                        .chunks
                        .get(&Coord::new(coord.x + dx, coord.y + dy, coord.z + dz));
                    i += 1;
                }
            }
        }
        nhood
    }

    fn nhood_at<'a>(nhood: &[Option<&'a Loaded>; 27], dx: i32, dy: i32, dz: i32) -> Option<&'a Loaded> {
        nhood[((dz + 1) * 9 + (dy + 1) * 3 + (dx + 1)) as usize]
    }

    /// Every cell is opaque. A paletted chunk's entries are exactly the ids in
    /// use, so the palette decides it without a cell walk.
    fn chunk_all_opaque(chunk: &Chunk, tables: &HotTables) -> bool {
        match &chunk.data().payload {
            ChunkPayload::Uniform(v) => tables.opaque(v.id),
            ChunkPayload::Paletted { palette, .. } => palette.iter().all(|p| tables.opaque(p.id)),
            ChunkPayload::Dense(cells) => cells.iter().all(|c| tables.opaque(c.id)),
        }
    }

    /// The face that touches `face` is solid opaque. Uniform and all-opaque
    /// payloads answer without walking the face.
    fn chunk_face_opaque(chunk: &Chunk, face: Face, tables: &HotTables) -> bool {
        if let Some(id) = chunk.uniform() {
            return tables.opaque(id);
        }
        if Self::chunk_all_opaque(chunk, tables) {
            return true;
        }
        let edge = if face.sign() > 0 { CHUNK_SIZE - 1 } else { 0 };
        for a in 0..CHUNK_SIZE {
            for b in 0..CHUNK_SIZE {
                let (x, y, z) = match face.axis() {
                    0 => (edge, a, b),
                    1 => (a, edge, b),
                    _ => (a, b, edge),
                };
                if !tables.opaque(chunk.get_local(x, y, z)) {
                    return false;
                }
            }
        }
        true
    }

    /// A fresh mesh whose chunk is fully opaque and whose six neighbour faces
    /// are too has nothing to draw. Settle it `Air` (the empty-mesh result)
    /// and skip the snapshot. A carried GPU mesh still goes through the worker
    /// so the old handle is freed, and a missing neighbour reads as air and
    /// would emit faces.
    pub(in crate::world) fn bury_solid_mesh(&mut self, coord: Coord) -> bool {
        let fresh = matches!(
            self.chunks.get(&coord).map(|l| &l.state),
            Some(MeshState::NeedsMesh {
                building: false,
                prev: None,
            })
        );
        if !fresh || self.light_terminal.contains(&coord) || !self.neighbours_have_data(coord) {
            return false;
        }
        self.refresh_tables();
        let tables = self.tables.get();
        if !self
            .chunks
            .get(&coord)
            .is_some_and(|l| Self::chunk_all_opaque(&l.chunk, &tables))
        {
            return false;
        }
        for &face in &Face::ALL {
            let ncoord = self.neighbour(coord, face);
            let covered = self.chunks.get(&ncoord).is_some_and(|l| {
                Self::chunk_face_opaque(&l.chunk, face.opposite(), &tables)
            });
            if !covered {
                return false;
            }
        }
        if let Some(loaded) = self.chunks.get_mut(&coord) {
            loaded.state = MeshState::Air;
        }
        self.note_settled();
        true
    }

    /// Put `coord` on the mesh worklist, or settle it `Air` when it is already
    /// walled in. Only a chunk the mesh lane could admit is seeded: an unloaded
    /// coord seeds itself when it loads, an in-flight build re-seeds if its
    /// result goes stale (`drop_stale_upload`), and a `Ready`, `Dirty` or `Air`
    /// chunk is never admitted (a relight remeshes through the light gate, an
    /// edit through the dirty lane). Seeding those parked them in rings the
    /// admission walk never reached, re-bucketed on every centre move.
    pub(super) fn seed_mesh(&mut self, coord: Coord) {
        if self.awaits_mesh(coord) && self.admits_mesh(coord) && !self.bury_solid_mesh(coord) {
            self.mesh_worklist.insert(coord);
        }
    }

    /// A loaded chunk awaiting a mesh with no build in flight: the only state the mesh lane admits.
    pub(super) fn awaits_mesh(&self, coord: Coord) -> bool {
        matches!(
            self.chunks.get(&coord).map(|l| &l.state),
            Some(MeshState::NeedsMesh { building: false, .. })
        )
    }

    /// Six orthogonal neighbours have data loaded.
    pub(in crate::world) fn neighbours_have_data(&self, coord: Coord) -> bool {
        Face::ALL
            .iter()
            .all(|&f| self.chunks.contains_key(&self.neighbour(coord, f)))
    }

    /// Build chunk GPU mesh (sync dirty-remesh). Frees old handle exactly once.
    fn mesh_chunk(&mut self, coord: Coord, eng: &mut Engine) {
        // Move the scratch out so the build can borrow `self` while filling it.
        // `MeshData` has no `Default` (it carries a `Pass`), so swap in a fresh
        // opaque scratch rather than `mem::take`; the build clears it first anyway.
        let mut scratch = std::mem::replace(&mut self.scratch, mesh::new_chunk_mesh_data());
        self.build_dirty_mesh(coord, &mut scratch);
        // `upload_chunk`'s retire frees the edited-Ready chunk's old mesh
        // (`Dirty.prev`) exactly once and installs the fresh `Ready`/`Air`.
        let hash = mesh::content_hash(&scratch);
        self.upload_chunk(coord, &scratch, Some(hash), eng);
        self.scratch = scratch;
    }

    /// The edit remesh's mesh of `coord`, built into `out` from the snapshot a mesh job carries.
    /// Uses the published light, which may be stale after an edit: geometry updates this frame,
    /// the relit mesh lands once light reconverges.
    pub(in crate::world) fn build_dirty_mesh(&mut self, coord: Coord, out: &mut mesh::ChunkMeshData) {
        self.refresh_tables();
        let degraded = !self.light_ready(coord);
        self.mark_degraded(coord, degraded);
        let (_, snap) = self.snapshot(coord, degraded);
        match &snap.light {
            Some(light) => mesh::build_chunk_mesh(&snap.padded, snap.uniform, &snap.tables, light, out),
            None => mesh::build_chunk_mesh_unlit(&snap.padded, snap.uniform, &snap.tables, out),
        }
        debug_assert!(
            self.chunks.get(&coord).is_some_and(|l| l.state.is_dirty()),
            "sync remesh of non-Dirty {coord:?}"
        );
    }

    /// Re-snapshot hot solidity array if palette grew (append-only, new Arc, old jobs unaffected)
    /// or a stamped meshing input (AO) flipped — the epoch folds into the revision's high bits
    /// (block count stays far below 2^32, so the two never collide).
    pub(in crate::world) fn refresh_tables(&mut self) {
        // Split the borrow: `sync`'s rebuild closure needs `&self.registry`
        // while `&mut self.tables` is held, so bind `registry` separately.
        let count = self.registry.block_count();
        let registry = &self.registry;
        let layer_cap = self.textures.layer_cap;
        let ao = self.ao;
        let rev = Revision::from_count(count | (self.tables_epoch as usize) << 32);
        self.tables.sync(rev, || {
            let mut tables = registry.hot_tables();
            tables.layer_cap = layer_cap;
            tables.ao = ao;
            tables
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Breaks one surface block of the flat world and returns its chunk, now `Dirty`.
    fn edit_surface(world: &mut World) -> Coord {
        let (x, z) = (5, 9);
        let h = (0..64).rev().find(|&y| world.is_solid(x, y, z)).expect("ground");
        world.set_block(x, h, z, AIR);
        let coord = World::chunk_of(x, h, z);
        assert!(world.chunks[&coord].state.is_dirty());
        coord
    }

    /// The content hash of the mesh a worker builds for `coord`'s mesh job.
    fn worker_mesh(world: &mut World, coord: Coord, degraded: bool) -> u64 {
        let (rev, snapshot) = world.snapshot(coord, degraded);
        assert!(world.worker_pool().submit(pipeline::Job::Mesh { coord, rev, snapshot }));
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            match world.workers.as_ref().and_then(pipeline::Workers::try_recv) {
                Some(pipeline::Done::Mesh { coord: c, data: pipeline::MeshPayload::Cpu(out), .. }) if c == coord => {
                    return mesh::content_hash(&out);
                }
                Some(pipeline::Done::Cancelled(_) | pipeline::Done::Failed(_)) => panic!("the mesh job did not run"),
                Some(_) => {}
                None => {
                    assert!(Instant::now() < deadline, "the mesh job never landed");
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        }
    }

    /// The content hash of the edit remesh's old build: the voxel shell and the light shell
    /// captured by two separate neighbourhood walks.
    fn captured_mesh(world: &World, coord: Coord, degraded: bool) -> u64 {
        let at = |dx: i32, dy: i32, dz: i32| world.chunks.get(&Coord::new(coord.x + dx, coord.y + dy, coord.z + dz));
        let mut padded = mesh::Padded::capture(|dx, dy, dz| at(dx, dy, dz).map(|l| &*l.chunk));
        world.seam_halo(coord, &mut padded);
        let fallback = degraded.then(light::LightGrid::open_sky);
        let mut shell =
            light::PaddedLight::capture(|dx, dy, dz| at(dx, dy, dz).and_then(|l| l.light.as_ref()).or(fallback.as_ref()));
        world.seam_light_halo(coord, &mut shell, fallback.as_ref());
        let mut out = mesh::new_chunk_mesh_data();
        let uniform = world.chunks[&coord].chunk.uniform();
        mesh::build_chunk_mesh(&padded, uniform, &world.tables.get(), &shell, &mut out);
        mesh::content_hash(&out)
    }

    /// With lighting off an edit remeshes full-bright, exactly as a mesh job does.
    #[test]
    fn unlit_edit_remesh_matches_the_worker_mesh() {
        let mut world = World::generate();
        assert!(world.transition_lighting(false));
        let coord = edit_surface(&mut world);
        let mut out = mesh::new_chunk_mesh_data();
        world.build_dirty_mesh(coord, &mut out);
        assert!(out.iter().any(|(_, m)| !m.is_empty()), "the edited surface draws");
        assert_eq!(mesh::content_hash(&out), worker_mesh(&mut world, coord, false));
    }

    /// With lighting on the edit remesh is the old two-walk capture, degraded and settled.
    #[test]
    fn lit_edit_remesh_matches_the_capture() {
        let mut world = World::generate();
        let coord = edit_surface(&mut world);
        let mut out = mesh::new_chunk_mesh_data();
        assert!(!world.light_ready(coord), "the edit re-seeded its light");
        let expect = captured_mesh(&world, coord, true);
        world.build_dirty_mesh(coord, &mut out);
        assert_eq!(mesh::content_hash(&out), expect, "degraded");
        assert!(world.light_gate.degraded.contains(&coord));

        world.light_worklist.clear();
        world.light_inflight.clear();
        let mut i = 0;
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if let Some(l) = world.chunks.get_mut(&Coord::new(coord.x + dx, coord.y + dy, coord.z + dz)) {
                        l.light = Some(if i % 3 == 0 { light::LightGrid::dark() } else { light::LightGrid::open_sky() });
                    }
                    i += 1;
                }
            }
        }
        assert!(world.light_ready(coord));
        let expect = captured_mesh(&world, coord, false);
        world.build_dirty_mesh(coord, &mut out);
        assert_eq!(mesh::content_hash(&out), expect, "settled");
        assert!(!world.light_gate.degraded.contains(&coord));
    }
}
