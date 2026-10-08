//! The light gate and light settling: ceilings, trivial grids, settles and their seeds, and the
//! degraded-mesh timers.

use super::*;
use crate::world::light;

/// Timeout before meshing a chunk with missing neighbour light as degraded.
/// Degraded chunks remesh once real light arrives.
///
// Wait-time gating avoids a remesh storm at cold-world entry: most chunks
// receive light within this window and mesh once with final light. Only
// stragglers degrade. Without this, the worker pool remeshes every chunk twice.
pub(super) const LIGHT_WAIT_DEGRADE: Duration = Duration::from_millis(150);

/// Tracks degraded meshes waiting for neighbour light to settle.
/// `blocked_since`: per-chunk timer for when it became light-blocked.
/// `degraded`: set of chunks currently drawing a degraded mesh, owed a remesh.
/// `dirty`: changed-light chunks waiting for a 27-neighbourhood fixpoint (or
/// the degrade timer) before their next mesh job.
#[derive(Default)]
pub(in crate::world) struct LightGate {
    pub(in crate::world) blocked_since: FastMap<Coord, Instant>,
    pub(in crate::world) degraded: FastSet<Coord>,
    pub(in crate::world) dirty: FastMap<Coord, Instant>,
}

impl LightGate {
    /// Start the wait timer for a light-blocked chunk (keeps an existing
    /// timer — re-eviction must not push the degrade horizon out).
    pub(in crate::world) fn note_blocked(&mut self, coord: Coord) {
        self.blocked_since.entry(coord).or_insert_with(crate::sched::now);
    }

    /// Mark a changed-light chunk; the first mark starts the degrade clock.
    fn mark_dirty(&mut self, coord: Coord) {
        self.dirty.entry(coord).or_insert_with(crate::sched::now);
    }
}

/// Cap on per-chunk remesh/job samples kept for the stress mean/p95 gauges.
const REMESH_SAMPLE_CAP: usize = 1 << 16;

/// Flight counters for light-convergence remeshes (stress C3).
#[derive(Default)]
pub(in crate::world) struct RemeshStats {
    pub remesh_async_calls: u64,
    pub drop_stale_uploads: u64,
    pub drop_stale_this_frame: u32,
    remesh_since_upload: FastMap<Coord, u16>,
    remesh_between_upload_samples: Vec<u16>,
    mesh_jobs_until_fixpoint: FastMap<Coord, u16>,
    mesh_jobs_fixpoint_done: FastSet<Coord>,
    mesh_jobs_before_fixpoint_samples: Vec<u16>,
}

impl RemeshStats {
    fn note_remesh(&mut self, coord: Coord) {
        self.remesh_async_calls += 1;
        let n = self.remesh_since_upload.entry(coord).or_insert(0);
        *n = n.saturating_add(1);
    }

    pub(super) fn note_upload(&mut self, coord: Coord) {
        let n = self.remesh_since_upload.remove(&coord).unwrap_or(0);
        if self.remesh_between_upload_samples.len() < REMESH_SAMPLE_CAP {
            self.remesh_between_upload_samples.push(n);
        }
    }

    pub(super) fn note_drop_stale(&mut self) {
        self.drop_stale_uploads += 1;
        self.drop_stale_this_frame = self.drop_stale_this_frame.saturating_add(1);
    }

    pub(in crate::world) fn note_mesh_job(&mut self, coord: Coord, nhood_quiet: bool) {
        if self.mesh_jobs_fixpoint_done.contains(&coord) {
            return;
        }
        if nhood_quiet {
            let n = self.mesh_jobs_until_fixpoint.remove(&coord).unwrap_or(0);
            if self.mesh_jobs_before_fixpoint_samples.len() < REMESH_SAMPLE_CAP {
                self.mesh_jobs_before_fixpoint_samples.push(n);
            }
            self.mesh_jobs_fixpoint_done.insert(coord);
        } else {
            let n = self.mesh_jobs_until_fixpoint.entry(coord).or_insert(0);
            *n = n.saturating_add(1);
        }
    }

    pub(super) fn forget(&mut self, coord: Coord) {
        self.remesh_since_upload.remove(&coord);
        self.mesh_jobs_until_fixpoint.remove(&coord);
        self.mesh_jobs_fixpoint_done.remove(&coord);
    }

    pub(super) fn between_upload_mean_p95(&self) -> (f32, f32, u64) {
        sample_mean_p95(&self.remesh_between_upload_samples)
    }

    pub(super) fn jobs_before_fixpoint_mean_p95(&self) -> (f32, f32, u64) {
        sample_mean_p95(&self.mesh_jobs_before_fixpoint_samples)
    }
}

fn sample_mean_p95(samples: &[u16]) -> (f32, f32, u64) {
    let n = samples.len() as u64;
    if samples.is_empty() {
        return (0.0, 0.0, 0);
    }
    let sum: u64 = samples.iter().map(|&v| u64::from(v)).sum();
    let mean = sum as f32 / n as f32;
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let p95 = sorted[((n - 1) as f32 * 0.95).round() as usize] as f32;
    (mean, p95, n)
}

impl World {
    /// Near-face light from six neighbours (input for light flood seeds).
    pub(in crate::world) fn capture_face_shell(&self, coord: Coord) -> light::FaceShell {
        if !self.lighting {
            return light::FaceShell::dark();
        }
        let mut shell = light::FaceShell::capture(|face| {
            self.chunks
                .get(&coord.step(face))
                .and_then(|l| l.light.as_ref())
        });
        self.seam_face_shell(coord, &mut shell);
        shell
    }

    /// Skylight ceiling: ground altitude per face-local column (caves dark
    /// consistently) RAISED by edited opaque roofs, so a player-built ceiling
    /// shadows the chunks below it. Keyed by [`ColumnKey`] and cached. `Open`
    /// chunks have no ceiling; this returns the ignored window and does not
    /// cache it.
    ///
    /// Async columns install the window in [`accept_column`](Self::accept_column)
    /// before store; [`ensure_data`](Self::ensure_data) does the same from the
    /// synchronous `generate_column`. This miss path is the remainder (edit
    /// invalidation, tests) and still reads heights from `generate_column`,
    /// never `height()`.
    ///
    /// Generated volumetrics (overhang shelves, flying islands) are still NOT
    /// part of the ceiling: `height()` deliberately describes ground only, so
    /// they don't shadow the columns beneath them — a known model limit that
    /// needs a generator-side occupancy summary to lift.
    pub(in crate::world) fn capture_ceiling(
        &mut self,
        coord: Coord,
    ) -> std::sync::Arc<light::CeilingWindow> {
        let Sky::Axis(face) = self.generator.sky(coord) else {
            return light::ignored_ceiling();
        };
        let (key, _) = ColumnKey::of(face, coord);
        if let Some(ceiling) = self.ceilings.get(&key) {
            return std::sync::Arc::clone(ceiling);
        }
        // Empty altitude range: both generators sample the 256 column profiles
        // before iterating the chunk layers, so this is the height field
        // without a voxel fill.
        let heights = self.generator.generate_column(key, 1..=0).1;
        let ceiling = std::sync::Arc::new(self.ceiling_from_heights(key, &heights));
        self.ceilings.insert(key, std::sync::Arc::clone(&ceiling));
        ceiling
    }

    /// Rebuild the PosY ceiling from `height()` plus edited roofs, ignoring the
    /// cache — equality check against production `generate_column` heights.
    #[cfg(test)]
    pub(in crate::world) fn capture_ceiling_slow(&self, coord: Coord) -> light::CeilingWindow {
        let x0 = coord.x * CHUNK_SIZE as i32;
        let z0 = coord.z * CHUNK_SIZE as i32;
        let generator = &self.generator;
        let mut ceiling = light::CeilingWindow::from_heights(Face::PosY, |lx, lz| {
            generator.height(x0 + lx as i32, z0 + lz as i32)
        });
        let key = ColumnKey { face: Face::PosY, a: coord.x, b: coord.z };
        self.raise_edited_roofs(key, &mut ceiling);
        ceiling
    }

    pub(super) fn install_ceiling(&mut self, key: ColumnKey, heights: &ColumnHeights) {
        if self.ceilings.contains_key(&key) {
            return;
        }
        let ceiling = std::sync::Arc::new(self.ceiling_from_heights(key, heights));
        self.ceilings.insert(key, ceiling);
    }

    fn ceiling_from_heights(
        &self,
        key: ColumnKey,
        heights: &ColumnHeights,
    ) -> light::CeilingWindow {
        let mut ceiling =
            light::CeilingWindow::from_heights(key.face, |lu, lv| heights[lu + lv * CHUNK_SIZE]);
        self.raise_edited_roofs(key, &mut ceiling);
        ceiling
    }

    fn raise_edited_roofs(&self, key: ColumnKey, ceiling: &mut light::CeilingWindow) {
        let frame = FaceFrame::new(key.face);
        let s = CHUNK_SIZE as i32;
        for (&c, cells) in &self.edits {
            if !matches!(self.generator.sky(c), Sky::Axis(face) if face == key.face) {
                continue;
            }
            if ColumnKey::of(key.face, c).0 != key {
                continue;
            }
            for (&index, &id) in cells {
                if !self.registry.is_opaque(id) {
                    continue;
                }
                let (lx, ly, lz) = Chunk::local_of(index);
                let (lu, _, lv) = frame.index_to_local(lx, ly, lz);
                let world = (c.x * s + lx as i32, c.y * s + ly as i32, c.z * s + lz as i32);
                let alt = frame.cell_to_local(world).1;
                ceiling.raise(lu, lv, alt + 1);
            }
        }
    }

    /// The analytic light grid for a chunk whose settled light is provable
    /// without a flood, or `None` if it must go through the worker settle. The
    /// two trivial cases collapse the load-time light-job burst to the thin
    /// Dense surface band (see [`store_chunk`](Self::store_chunk)):
    /// - a uniform opaque, non-emissive chunk settles to all-dark (no light enters);
    /// - a uniform-*air* chunk fully above every column's surface, with no near
    ///   blocklight from a loaded neighbour, settles to full sky / dark block.
    ///   `Open` air with no near blocklight is full sky with no ceiling.
    ///
    /// Correctness anchor: the returned grid equals `propagate(uniform, dark
    /// shell, ceiling, sky, alt0, tables)`.
    ///
    /// `&mut self` so it can warm the `ceilings` column cache and the hot tables
    /// while probing — it mutates no lane state.
    pub(super) fn trivial_light(&mut self, coord: Coord, chunk: &Chunk) -> Option<light::LightGrid> {
        if !self.lighting {
            return None;
        }
        self.refresh_tables();
        let tables = self.tables.get();
        // A full block of inert opaque rock settles to all-dark: no skylight
        // column stays open through it and no neighbour light can relax into an
        // opaque cell. Emissive opaque blocks must take the flood path so they
        // can seed their own blocklight.
        if chunk.is_uniform_opaque(&tables) {
            return Some(light::LightGrid::dark());
        }
        // A uniform-air chunk that sits fully above every column's surface is
        // full sky — *if* no loaded neighbour has near-border blocklight that
        // would bleed in (skylight can't exceed FULL, so only blocklight breaks
        // the analytic result). `propagate` with a dark shell yields exactly
        // `open_sky()` here; the neighbour check is what makes the dark shell sound.
        if chunk.uniform() == Some(crate::block::registry::AIR) {
            match self.generator.sky(coord) {
                Sky::Open => {
                    if !self.neighbour_blocklight_near(coord) {
                        return Some(light::LightGrid::open_sky());
                    }
                }
                Sky::Axis(face) => {
                    let alt0 = FaceFrame::new(face).chunk_alt0(coord);
                    let ceiling = self.capture_ceiling(coord);
                    if alt0 >= ceiling.min_surface() && !self.neighbour_blocklight_near(coord) {
                        return Some(light::LightGrid::open_sky());
                    }
                }
            }
        }
        None
    }

    /// Whether any loaded face-neighbour carries near-border blocklight `> 1`
    /// (light level 1 attenuates to 0 crossing in, so it can't seed). Used by
    /// [`trivial_light`](Self::trivial_light) to reject the dark-shell fast path
    /// when a torch next door would actually bleed across the border.
    pub(super) fn neighbour_blocklight_near(&self, coord: Coord) -> bool {
        Face::ALL.iter().any(|&face| {
            self.chunks
                .get(&self.neighbour(coord, face))
                .is_some_and(|l| l.has_blocklight)
        })
    }

    /// Schedule an ASYNC rebuild for a chunk whose mesh inputs changed off the
    /// edit path (a light grid landed, a degraded mesh's real light arrived).
    /// The chunk keeps drawing its current mesh — carried as `NeedsMesh.prev`
    /// — until the fresh worker result uploads, and the rev bump both strands
    /// any in-flight build against the old inputs and stales any queued
    /// upload. This replaces the old routing of light arrivals through the
    /// SYNC `Dirty` machinery, which built up to `DIRTY_BUDGET` full greedy
    /// meshes per frame ON THE MAIN THREAD during load floods (nearly every
    /// chunk meshes degraded first under the 150 ms light gate, then relights)
    /// — the "still laggy seconds after stopping" stall. The sync path stays
    /// for player edits only, where same-frame response is the point.
    pub(super) fn remesh_async(&mut self, coord: Coord) {
        let skip = match self.chunks.get(&coord).map(|l| &l.state) {
            None => return,
            Some(MeshState::Dirty { .. } | MeshState::Air) => true,
            Some(_) => false,
        };
        if skip {
            self.light_gate.dirty.remove(&coord);
            return;
        }
        {
            let Some(loaded) = self.chunks.get_mut(&coord) else {
                return;
            };
            if let MeshState::Ready(_) = &loaded.state {
                // Carry the drawn mesh into the rebuild. A NeedsMesh already
                // awaiting/mid-build just takes the rev bump, which strands the
                // in-flight result.
                let prev = std::mem::replace(&mut loaded.state, MeshState::needs_mesh()).into_owned();
                loaded.state = MeshState::NeedsMesh {
                    building: false,
                    prev,
                };
            }
            loaded.rev = loaded.rev.wrapping_add(1);
        }
        self.seed_mesh(coord);
        self.pending_fresh.set();
        self.light_gate.dirty.remove(&coord);
        self.remesh_stats.note_remesh(coord);
    }

    /// Faces whose border lumels differ. A first publish compares against dark
    /// (the shell missing neighbours already assumed).
    fn face_moves(prev: Option<&light::LightGrid>, grid: &light::LightGrid) -> u8 {
        let dark = light::LightGrid::dark();
        let old = prev.unwrap_or(&dark);
        let mut bits = 0u8;
        for &face in &Face::ALL {
            if light::border_changed(old, grid, face) {
                bits |= 1 << (face as u8);
            }
        }
        bits
    }

    /// Count a light-worklist insert (the stress harness's seeds-per-chunk signal).
    pub(in crate::world) fn seed_light(&mut self, coord: Coord, source: super::LightSeed) {
        self.counters.light_seed_inserts += 1;
        self.counters.light_seed_split.add(source);
        if let Some(loaded) = self.chunks.get_mut(&coord) {
            loaded.light_reseed = false;
        }
        if !self.light_owed.is_empty() {
            self.light_owed.remove(&coord);
        }
        self.light_worklist.insert(coord);
    }

    /// Publish settled light, re-arm mesh readiness, and seed neighbours to re-settle.
    /// Shared by sync (trivial) and async settle paths.
    pub(in crate::world) fn settle_light(&mut self, coord: Coord, grid: light::LightGrid) {
        // Release the claim first. No-op on sync path (trivial never enters inflight);
        // absorbs async removal, keeping settled/inflight state consistent.
        self.light_inflight.remove(&coord);
        // Unloaded while the flood flew (or before a trivial publish): drop it.
        let reseed = match self.chunks.get_mut(&coord) {
            Some(loaded) => {
                let r = loaded.light_reseed;
                loaded.light_reseed = false;
                r
            }
            None => return,
        };
        let self_changed = self.chunks[&coord]
            .light
            .as_ref()
            .is_none_or(|old| *old != grid);
        // Re-arm mesh readiness unconditionally, even on identical grids.
        // A fixpoint re-settle that skipped this seed would strand the chunk
        // off the worklist forever (ready but unreachable, idle stall).
        self.pending_fresh.set();
        self.seed_mesh(coord);
        if !self_changed {
            if reseed {
                self.seed_light(coord, super::LightSeed::Border);
                self.light_pending.set();
            }
            return;
        }
        // Face bitmask (bit = `Face` discriminant). First publish compares
        // against dark — missing neighbours already assumed that shell for the
        // flood. Neighbours still need a *mesh* seed: first publish is what
        // makes `light_ready` true for them.
        let first = self.chunks[&coord].light.is_none();
        let moved = Self::face_moves(self.chunks[&coord].light.as_ref(), &grid);
        let has_blocklight = grid.has_border_blocklight();
        let open_sky = grid == light::LightGrid::open_sky();
        {
            let loaded = self.chunks.get_mut(&coord).unwrap();
            loaded.light = Some(grid);
            loaded.has_blocklight = has_blocklight;
        }
        // Light-seed only neighbours whose shared border moved and that already
        // have data. An in-flight neighbour is marked, not re-inserted: at most
        // one extra flood when its result integrates. Mesh-seed on first publish
        // too (waiting neighbours become ready). Mark dirty instead of remeshing
        // immediately: `tick_light_gate` promotes once the 27-neighbourhood
        // has no pending light work (or the degrade timer expires).
        self.light_gate.mark_dirty(coord);
        for &face in &Face::ALL {
            let face_moved = moved & (1 << (face as u8)) != 0;
            if !face_moved && !first {
                continue;
            }
            let n = self.neighbour(coord, face);
            if !self.chunks.contains_key(&n) {
                continue;
            }
            if face_moved {
                // Open-sky next to open-sky cannot change. Mesh-seed only:
                // this publish can complete their light_ready.
                let n_sky = self.chunks[&n].light.as_ref() == Some(&light::LightGrid::open_sky());
                if open_sky && n_sky {
                    self.seed_mesh(n);
                    continue;
                }
                if self.light_inflight.contains(&n) {
                    self.chunks.get_mut(&n).unwrap().light_reseed = true;
                } else if !self.light_worklist.contains(&n) {
                    // Pending floods read live neighbour grids at admit.
                    self.seed_light(n, super::LightSeed::Border);
                }
            }
            self.seed_mesh(n);
            self.light_gate.mark_dirty(n);
        }
        if reseed {
            self.seed_light(coord, super::LightSeed::Border);
        }
        if !self.light_worklist.is_empty() {
            self.light_pending.set();
        }
    }

    /// Light settled enough to mesh: chunk and face neighbours have grids, and the
    /// chunk is neither seeded nor being settled (in-flight counts as not-yet-final,
    /// so a chunk never meshes against a flood that's still running for it).
    /// A neighbour the reduced loading window will not light counts as settled:
    /// that face stays dark, matching a missing chunk, instead of holding the
    /// surface for [`LIGHT_WAIT_DEGRADE`]. The full window still waits.
    pub(in crate::world) fn light_ready(&self, coord: Coord) -> bool {
        if !self.lighting {
            // Nothing to settle: gate meshing on data alone (checked separately).
            return self.chunks.contains_key(&coord);
        }
        !self.light_worklist.contains(&coord)
            && !self.light_inflight.contains(&coord)
            && self.chunks.get(&coord).is_some_and(|l| l.light.is_some())
            && Face::ALL.iter().all(|&f| self.neighbour_light_ready(self.neighbour(coord, f)))
    }

    /// `coord` has a grid, or the reduced window will not schedule its flood.
    fn neighbour_light_ready(&self, coord: Coord) -> bool {
        self.chunks.get(&coord).is_some_and(|l| l.light.is_some()) || !self.admits_light(coord)
    }

    /// True when the 27-neighbourhood has no pending light work. Apply-queue
    /// coords keep their inflight claim until `settle_light`, so inflight
    /// covers the queue; the empty-queue check is the cheap global fast path.
    pub(in crate::world) fn light_nhood_quiet(&self, coord: Coord) -> bool {
        if self.light_worklist.is_empty()
            && self.light_inflight.is_empty()
            && self.light_apply_queue.is_empty()
        {
            return true;
        }
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let n = Coord::new(coord.x + dx, coord.y + dy, coord.z + dz);
                    if self.light_worklist.contains(&n) || self.light_inflight.contains(&n) {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Turn a `light_dirty` mark into a mesh job. Returns true if the mark
    /// can drop. Skips an in-flight first mesh (no rev bump) so the early
    /// degraded mesh still appears on the same clock as before.
    fn promote_dirty_mesh(&mut self, coord: Coord) -> bool {
        let Some(loaded) = self.chunks.get(&coord) else {
            return true;
        };
        let action = match &loaded.state {
            MeshState::Air | MeshState::Dirty { .. } => 0u8,
            MeshState::NeedsMesh {
                building: true, ..
            } => 1,
            MeshState::NeedsMesh {
                building: false,
                prev: None,
            } => 2,
            MeshState::Ready(_)
            | MeshState::NeedsMesh {
                building: false,
                prev: Some(_),
            } => 3,
        };
        match action {
            0 => true,
            1 => false,
            2 => self.light_nhood_quiet(coord),
            _ => {
                self.remesh_async(coord);
                true
            }
        }
    }

    /// A chunk waiting purely on neighbour light: it has data and is in view and
    /// awaiting a fresh mesh, but its neighbourhood light has not settled. The
    /// [`LightGate`] times exactly these chunks.
    pub(in crate::world) fn chunk_light_blocked(&self, coord: Coord) -> bool {
        self.is_needs_mesh(coord)
            && self.in_mesh_box(coord)
            && self.admits_mesh(coord)
            && self.neighbours_have_data(coord)
            && !self.light_ready(coord)
    }

    /// Whether `coord` has waited on neighbour light past [`LIGHT_WAIT_DEGRADE`] —
    /// the mesh-lane predicate that admits a DEGRADED mesh.
    pub(in crate::world) fn light_wait_expired(&self, coord: Coord) -> bool {
        self.light_gate
            .blocked_since
            .get(&coord)
            .is_some_and(|t| t.elapsed() >= LIGHT_WAIT_DEGRADE)
    }

    /// Record (or clear) that `coord` is currently drawing a degraded, known-not-
    /// final mesh. The set is queryable by [`entry_complete`](Self::entry_complete)
    /// ("none pending").
    pub(in crate::world) fn mark_degraded(&mut self, coord: Coord, degraded: bool) {
        if degraded {
            self.light_gate.degraded.insert(coord);
        } else {
            self.light_gate.degraded.remove(&coord);
            self.light_terminal.remove(&coord);
        }
    }

    /// Advance the light-gate before the mesh lane runs: reap timers whose
    /// chunk stopped waiting, drop degraded/dirty entries for unloaded chunks,
    /// promote `light_dirty` (and relit-degraded) chunks whose 27-neighbourhood
    /// has no pending light work or whose degrade timer expired, and re-seed
    /// exactly the chunks whose DEGRADE TIMER expired — expiry raises no event
    /// of its own, so this sweep (over ONLY the timed/dirty maps, never the
    /// world) is what un-strands them. Timers START at the admit loop's
    /// blocked-eviction event ([`MeshLane::on_blocked`]); every pre-expiry
    /// re-seed comes from a real event (a grid landing via `settle_light`,
    /// neighbour data via `store_chunk`).
    pub(in crate::world) fn tick_light_gate(&mut self) {
        // `LightGate` is `Default`, so move it out to break the self-borrow while
        // the predicates below read the chunk map. Empty maps skip `retain`
        // (it still walks capacity); a drained flood `shrink_to_fit`s once.
        let mut gate = std::mem::take(&mut self.light_gate);
        if !gate.degraded.is_empty() {
            gate.degraded.retain(|c| self.chunks.contains_key(c));
            if gate.degraded.is_empty() {
                gate.degraded.shrink_to_fit();
            }
        }
        if !gate.dirty.is_empty() {
            gate.dirty.retain(|c, _| self.chunks.contains_key(c));
            if gate.dirty.is_empty() {
                gate.dirty.shrink_to_fit();
            }
        }
        if !self.light_terminal.is_empty() {
            self.light_terminal.retain(|c| self.chunks.contains_key(c));
            if self.light_terminal.is_empty() {
                self.light_terminal.shrink_to_fit();
            }
        }
        if !gate.blocked_since.is_empty() {
            gate.blocked_since
                .retain(|c, _| self.chunk_light_blocked(*c));
            if gate.blocked_since.is_empty() {
                gate.blocked_since.shrink_to_fit();
            }
        }
        // One clock read for the sweep, and only when a timer exists to compare.
        // Timers are tested before the 27-neighbourhood scan: an expired entry
        // promotes whatever the neighbours say.
        let now = (!gate.dirty.is_empty() || !gate.blocked_since.is_empty()).then(crate::sched::now);
        let waited = |t: &Instant| now.is_some_and(|n| n.duration_since(*t) >= LIGHT_WAIT_DEGRADE);
        // Promote dirty (and relit-degraded) chunks once the 27-neighbourhood
        // has no pending light work, or the degrade timer has expired. One
        // pass over the dirty/degraded sets, never the world.
        let mut promote: Vec<Coord> = gate
            .dirty
            .iter()
            .filter(|(c, t)| waited(t) || self.light_nhood_quiet(**c))
            .map(|(c, _)| *c)
            .collect();
        for &c in &gate.degraded {
            if gate.dirty.contains_key(&c) {
                continue;
            }
            if self.light_ready(c)
                && (gate.blocked_since.get(&c).is_some_and(waited) || self.light_nhood_quiet(c))
            {
                promote.push(c);
            }
        }
        for c in promote {
            if self.promote_dirty_mesh(c) {
                gate.dirty.remove(&c);
            } else {
                gate.dirty.entry(c).or_insert_with(crate::sched::now);
            }
        }
        // The expiry sweep: a chunk past LIGHT_WAIT_DEGRADE is mesh-ready via
        // `light_wait_expired` but was evicted from the worklist when it
        // blocked — re-seed it now that the clock (not an event) unblocked it.
        // A timed chunk whose build is in flight takes no seed.
        let mut expired = false;
        for (&c, t) in &gate.blocked_since {
            if waited(t) {
                self.seed_mesh(c);
                expired = true;
            }
        }
        if expired {
            self.pending_fresh.set();
        }
        self.light_gate = gate;
    }

    /// Level-triggered backstop for the degraded set: per-coord, once that
    /// chunk's 27-neighbourhood has no pending light work. Event-driven paths
    /// miss degraded chunks whose missing neighbour settled without moving the
    /// shared border; this sweep promotes them to final so entry_complete
    /// doesn't hang. A still-building/Dirty chunk is left for a later flush.
    pub(in crate::world) fn flush_degraded_terminal(&mut self) {
        if self.light_gate.degraded.is_empty() {
            return;
        }
        // Taken out so each chunk's flush can touch the rest of the world.
        let mut degraded = std::mem::take(&mut self.light_gate.degraded);
        degraded.retain(|&coord| self.flush_degraded(coord));
        self.light_gate.degraded = degraded;
    }

    /// One chunk of [`flush_degraded_terminal`](Self::flush_degraded_terminal).
    /// Returns whether it stays marked degraded.
    fn flush_degraded(&mut self, coord: Coord) -> bool {
        match self.chunks.get(&coord).map(|l| &l.state) {
            // Still building (in-flight result pending) or Dirty (a sync
            // remesh owns it): another path is about to resolve it.
            Some(MeshState::NeedsMesh { building: true, .. } | MeshState::Dirty { .. }) => true,
            // Seeded in the box: the mesh lane admits it.
            Some(MeshState::NeedsMesh { .. })
                if self.in_mesh_box(coord) && self.mesh_worklist.contains(&coord) =>
            {
                true
            }
            // Every arm below acts, and only once the 27-neighbourhood has
            // no pending light work.
            _ if !self.light_nhood_quiet(coord) => true,
            // Unloaded out from under the set between marking and here, or
            // nothing to draw: drop the degraded flag.
            None | Some(MeshState::Air) => {
                self.light_terminal.remove(&coord);
                false
            }
            // Past the mesh box (unload hysteresis): not drawn, and a
            // rebuild would fail `in_mesh_box` / be dropped at apply. Drop
            // the flag so quiescence is not wedged.
            Some(_) if !self.in_mesh_box(coord) => {
                self.light_terminal.remove(&coord);
                false
            }
            // Outside the reduced loading window: the window's rescan seeds it
            // once it covers the chunk, and the mark keeps the remesh owed.
            Some(_) if !self.admits_mesh(coord) => true,
            // Settled on a degraded mesh — the stuck case. Rebuild async;
            // if neighbour light is still missing it will never arrive, so
            // the terminal set makes the snapshot read missing planes dark.
            Some(MeshState::Ready(_)) => {
                if !self.light_ready(coord) {
                    self.light_terminal.insert(coord);
                }
                self.remesh_async(coord);
                true
            }
            // Admit evicted the seed (a missing neighbour used to fail
            // `ready`, or this flush ran after that pass's admit). Re-seed
            // it; mark terminal if neighbour light will not arrive.
            Some(MeshState::NeedsMesh { .. }) => {
                if !self.light_ready(coord) {
                    self.light_terminal.insert(coord);
                }
                self.mesh_worklist.insert(coord);
                self.pending_fresh.set();
                true
            }
        }
    }
}
