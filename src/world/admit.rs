//! Streaming lanes: async admission producers.
//! Every async streaming lane (fresh chunk meshing, far LOD sections,
//! cross-chunk light, column generation) is the same shape: gather candidates
//! near the player, drop the ones already in flight, order nearest-first, submit
//! up to a per-frame budget to the worker pool, and integrate finished results.
//! The lanes differ only in *where their state lives*, reached through a
//! [`StreamLane`] of accessors so the admission loop ([`admit`]) is written once.
//!
//! Each lane is a `sched::Run` producer registered in [`lanes`]; the scheduler
//! hands it its `Budget::Millis` and it derives its [`Deadline`](pipeline::Deadline)
//! from that (never a private `pipeline::*_BUDGET` const — one budget locus, the
//! manifest). The deadline is checked BETWEEN admitted items, and the forward-
//! progress floor ([`admission_exhausted`]) is the single definition of "admit at
//! least `MIN_ADMIT` before the clock can stop you", shared with the generation
//! lane so no producer hand-rolls its own budget/floor.

use super::*;

/// The one admission-stop rule (forward-progress floor + time budget): stop only
/// once at least `min_admit` items have been admitted AND the deadline has
/// passed. The single definition every streaming producer's admit loop shares.
pub(in crate::world) fn admission_exhausted(
    admitted: usize,
    min_admit: usize,
    deadline: pipeline::Deadline,
) -> bool {
    admitted >= min_admit && deadline.expired()
}

/// How a lane defines its work this frame. A geometry lane derives candidates
/// from the player centre. A worklist lane reads an explicit dirty set
/// accumulated on the `World` (mesh, light: coords marked dirty by loads/edits).
pub(in crate::world) enum Candidates {
    /// Walk the frame's derived candidate keys (see [`StreamLane::for_each_geometry`]).
    Geometry,
    /// Read the lane's seed set (`StreamLane::seed_set`) for candidates.
    Worklist,
}

/// Reused gather buffers for one [`admit`] pass. Held on [`World`] so the
/// loop never allocates per pass.
pub(in crate::world) struct AdmitScratch<K> {
    keys: Vec<(u64, K)>,
    blocked: Vec<K>,
    /// A geometry lane's bounded nearest-first shortlist.
    rank: Vec<(u64, K)>,
}

impl<K> Default for AdmitScratch<K> {
    fn default() -> Self {
        Self {
            keys: Vec::new(),
            blocked: Vec::new(),
            rank: Vec::new(),
        }
    }
}

/// The accessor surface of one streaming lane. Zero-sized marker types
/// (`MeshLane`, …) implement it AND `sched::Run` (in [`lanes`]); all mutable
/// state lives on [`World`] behind these accessors, so the lane carries nothing
/// and the shared [`admit`] loop stays allocation-free. This is *not* a
/// scheduler: budget and forward-progress come from the scheduler that drives
/// the producer, never from here.
pub(in crate::world) trait StreamLane {
    /// The lane's work key: chunk `Coord` or `SectionPos`.
    /// Must be `Copy + Eq + Hash`.
    type Key: Copy + Eq + Hash;

    /// Minimum admissions before the deadline can stop the loop — the forward-
    /// progress floor, so setup cost (gather/sort) alone can't starve a lane
    /// under a tight budget. One value per lane; enforced once, in [`admit`].
    const MIN_ADMIT: usize;

    /// Far LOD lanes enqueue through `submit_far`; everyone else is near.
    const FAR: bool = false;

    /// The candidate keys for this frame (see [`Candidates`]).
    fn candidates(world: &World, center: Coord) -> Candidates;
    /// The worklist seed set, for worklist lanes (`None` for geometry lanes).
    fn seed_set(world: &mut World) -> Option<&mut worklist::RingWorklist>;
    /// Worklist-lane admission. Geometry lanes leave the default (unreachable).
    fn admit_worklist(_world: &mut World, _center: Coord, _budget: Budget) {
        debug_assert!(false, "geometry lane has no worklist");
    }
    /// This lane's reusable gather buffers.
    fn scratch(world: &mut World) -> &mut AdmitScratch<Self::Key>;
    /// This lane's raise-then-consume "has pending work" gate.
    fn pending(world: &mut World) -> &mut Sticky;
    /// Nearest-first ordering metric (lower is sooner). It receives the world
    /// so travel-sensitive lanes can spend their reduced budget ahead of the
    /// player rather than on equally near trailing work.
    fn order(world: &World, center: Coord, key: Self::Key) -> u64;
    /// Whether `key` is already in flight.
    fn in_flight(world: &World, key: Self::Key) -> bool;
    /// Whether `key` can be submitted now. Default: always true.
    fn ready(world: &World, key: Self::Key) -> bool {
        let _ = (world, key);
        true
    }
    /// Lane-specific budget refresh before geometry candidates are filtered.
    /// Default: nothing. Runs only when the worker queue has a free slot.
    fn prepare_admit(_world: &mut World) {}
    /// A worklist seed was evicted as BLOCKED by this pass's [`admit`] — the
    /// lane's chance to register it with an event source that will re-seed it
    /// (the mesh lane starts its light-degrade timer here). Default: no-op.
    fn on_blocked(world: &mut World, key: Self::Key) {
        let _ = (world, key);
    }
    /// Squared distance in metres from `key` to player, for far-lane distance ordering.
    /// `None` (default) marks a near lane using FIFO ordering.
    fn dist2(world: &World, center: Coord, key: Self::Key) -> Option<u64> {
        let _ = (world, center, key);
        None
    }
    /// Geometry-lane candidate visit; `rank` is a reusable shortlist buffer.
    /// Worklist lanes leave this unused.
    fn for_each_geometry(
        world: &World,
        center: Coord,
        rank: &mut Vec<(u64, Self::Key)>,
        visit: impl FnMut(Self::Key),
    ) {
        let _ = (world, center, rank, visit);
    }
    /// Build the worker job for `key`, or `None` to drop it.
    /// May warm caches but must not mutate lane state.
    fn submit(world: &mut World, key: Self::Key) -> Option<pipeline::Job>;
    /// Mark `key` in flight: remove from seed set and claim it to prevent re-submission.
    fn claim(world: &mut World, key: Self::Key);
    /// Fold a finished result back into the world (upload a mesh, publish light).
    fn integrate(world: &mut World, done: pipeline::Done);
    /// Record how many jobs this pass admitted. Light counts it for the stress
    /// harness; everyone else is a no-op.
    fn note_admitted(world: &mut World, n: usize) {
        let _ = (world, n);
    }
}

fn queue_slots<S: StreamLane>(world: &World) -> usize {
    let Some(workers) = world.workers.as_ref() else {
        return usize::MAX;
    };
    if S::FAR {
        workers.far_slots_free()
    } else {
        workers.near_slots_free()
    }
}

/// The shared admission loop: gather, filter unready/in-flight, select the
/// nearest `want`, submit until the deadline expires (checked between items,
/// never mid-item, and never before `MIN_ADMIT`), then claim. Worklist lanes
/// walk ring buckets nearest-first and stop once they have `want` ready keys
/// (far buckets are not visited). Clears the lane's pending gate once the
/// ready backlog drains. `budget` becomes a [`pipeline::Deadline`] only after
/// the pending gate, so an idle frame never samples `Instant::now`.
pub(in crate::world) fn admit<S: StreamLane>(
    world: &mut World,
    center: Coord,
    budget: Budget,
) {
    if !S::pending(world).get() {
        return;
    }
    match S::candidates(world, center) {
        Candidates::Worklist => S::admit_worklist(world, center, budget),
        Candidates::Geometry => admit_geometry::<S>(world, center, budget),
    }
}

/// Geometry-lane admission: walk this frame's derived candidates, select the
/// nearest `want` in O(n), submit.
fn admit_geometry<S: StreamLane>(world: &mut World, center: Coord, budget: Budget) {
    let deadline = lanes::paced_deadline(world, budget);
    let slots = queue_slots::<S>(world);
    let min_admit = world.stream_pacer.floor(S::MIN_ADMIT);

    let mut scratch = std::mem::take(S::scratch(world));
    scratch.keys.clear();
    scratch.blocked.clear();

    if slots == 0 {
        *S::scratch(world) = scratch;
        return;
    }
    // A full section budget refuses every candidate. That is not a drained
    // backlog: clearing pending here left the holes disarmed on a still camera.
    S::prepare_admit(world);
    let mut refused = false;
    S::for_each_geometry(world, center, &mut scratch.rank, |k| {
        if S::in_flight(world, k) {
            return;
        }
        if !S::ready(world, k) {
            refused = true;
            return;
        }
        scratch.keys.push((S::order(world, center, k), k));
    });

    let n = scratch.keys.len();
    if n == 0 {
        if !refused {
            S::pending(world).take();
        }
        *S::scratch(world) = scratch;
        return;
    }

    let want = n.min(slots.max(min_admit));
    if want < n {
        scratch.keys.select_nth_unstable_by_key(want - 1, |e| e.0);
        scratch.keys[..want].sort_by_key(|e| e.0);
    } else {
        scratch.keys.sort_by_key(|e| e.0);
    }

    let mut exhausted = want == n;
    let mut admitted = 0usize;
    let fill_slots = world.stream_pacer.boosting();
    for i in 0..want {
        let key = scratch.keys[i].1;
        if admission_exhausted(admitted, min_admit, deadline)
            && !(fill_slots && admitted < slots)
        {
            exhausted = false;
            break;
        }
        let Some(job) = S::submit(world, key) else {
            continue;
        };
        let far = S::dist2(world, center, key);
        let workers = world.worker_pool();
        let accepted = match far {
            Some(dist2) => workers.submit_far(job, dist2),
            None => workers.submit(job),
        };
        if accepted {
            S::claim(world, key);
            admitted += 1;
        } else {
            exhausted = false;
            break;
        }
    }
    S::note_admitted(world, admitted);
    if exhausted {
        S::pending(world).take();
    }
    *S::scratch(world) = scratch;
}

/// Worklist-lane admission: walk ring buckets nearest-first, evict blocked
/// seeds only in visited buckets, stop once `want` ready keys are in hand.
/// Motion bias is the tie-break within those buckets (it never reorders rings).
fn admit_coord_worklist<S: StreamLane<Key = Coord>>(
    world: &mut World,
    center: Coord,
    budget: Budget,
) {
    let deadline = lanes::paced_deadline(world, budget);
    let slots = queue_slots::<S>(world);
    let min_admit = world.stream_pacer.floor(S::MIN_ADMIT);
    let up = world.live_up();
    let rings = world.worklist_rings(center);

    let mut list = std::mem::take(S::seed_set(world).expect("worklist"));
    list.fit(center, rings, up);

    if slots == 0 {
        let empty = list.is_empty();
        *S::seed_set(world).expect("worklist") = list;
        if empty {
            S::pending(world).take();
        }
        return;
    }

    let want = slots.max(min_admit);
    let mut scratch = std::mem::take(S::scratch(world));
    scratch.keys.clear();
    scratch.blocked.clear();

    let visited_all = list.walk_nearest(|bucket| {
        if scratch.keys.len() >= want {
            return false;
        }
        for &k in bucket {
            if S::in_flight(world, k) {
                continue;
            }
            if S::ready(world, k) {
                scratch.keys.push((S::order(world, center, k), k));
            } else {
                scratch.blocked.push(k);
            }
        }
        true
    });
    for k in scratch.blocked.drain(..) {
        list.remove(&k);
        S::on_blocked(world, k);
    }
    *S::seed_set(world).expect("worklist") = list;

    let n = scratch.keys.len();
    if n == 0 {
        let drained = visited_all && S::seed_set(world).expect("worklist").is_empty();
        if drained {
            S::pending(world).take();
        }
        *S::scratch(world) = scratch;
        return;
    }

    let take = n.min(want);
    if take < n {
        scratch.keys.select_nth_unstable_by_key(take - 1, |e| e.0);
        scratch.keys[..take].sort_by_key(|e| e.0);
    } else {
        scratch.keys.sort_by_key(|e| e.0);
    }

    let mut exhausted = visited_all && take == n;
    let mut admitted = 0usize;
    let fill_slots = world.stream_pacer.boosting();
    for i in 0..take {
        let key = scratch.keys[i].1;
        if admission_exhausted(admitted, min_admit, deadline)
            && !(fill_slots && admitted < slots)
        {
            exhausted = false;
            break;
        }
        let Some(job) = S::submit(world, key) else {
            if let Some(set) = S::seed_set(world) {
                set.remove(&key);
            }
            continue;
        };
        let far = S::dist2(world, center, key);
        let workers = world.worker_pool();
        let accepted = match far {
            Some(dist2) => workers.submit_far(job, dist2),
            None => workers.submit(job),
        };
        if accepted {
            S::claim(world, key);
            admitted += 1;
        } else {
            exhausted = false;
            break;
        }
    }
    S::note_admitted(world, admitted);
    let drained = exhausted && S::seed_set(world).expect("worklist").is_empty();
    if drained {
        S::pending(world).take();
    }
    *S::scratch(world) = scratch;
}

/// Squared distance in metres from player-chunk centre to a world point.
pub(super) fn player_dist2(center: Coord, wx: i64, wy: i64, wz: i64) -> u64 {
    let s = CHUNK_SIZE as i64;
    let half = s / 2;
    let (px, py, pz) = (
        center.x as i64 * s + half,
        center.y as i64 * s + half,
        center.z as i64 * s + half,
    );
    let (dx, dy, dz) = (wx - px, wy - py, wz - pz);
    (dx * dx + dy * dy + dz * dz) as u64
}

/// Below this speed (m/s), motion bias is disabled.
const MOTION_BIAS_MIN_SPEED: f64 = 0.5;

/// Max bias fraction: cells ahead sort up to 30% nearer, cells behind 30% farther.
const MOTION_BIAS_STRENGTH: f64 = 0.3;

/// Bias priority by eye velocity: cells ahead sort sooner, behind later.
/// Affects ordering only, never the desired set. Identity at rest.
/// The Y component of `vel` is ignored (the +Y tangent plane is XZ).
pub(super) fn motion_biased_dist2(base: u64, vel: DVec3, dx: f64, dz: f64) -> u64 {
    let speed = (vel.x * vel.x + vel.z * vel.z).sqrt();
    let disp = (dx * dx + dz * dz).sqrt();
    if speed < MOTION_BIAS_MIN_SPEED || disp < 1.0 {
        return base;
    }
    let align = (vel.x * dx + vel.z * dz) / (speed * disp); // cosine in [-1, 1]
    (base as f64 * (1.0 - MOTION_BIAS_STRENGTH * align)).max(0.0) as u64
}

/// Same bias in all three axes. Used when the streaming volume has no up face.
fn motion_biased_dist3(base: u64, vel: DVec3, dx: f64, dy: f64, dz: f64) -> u64 {
    let speed = (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z).sqrt();
    let disp = (dx * dx + dy * dy + dz * dz).sqrt();
    if speed < MOTION_BIAS_MIN_SPEED || disp < 1.0 {
        return base;
    }
    let align = (vel.x * dx + vel.y * dy + vel.z * dz) / (speed * disp);
    (base as f64 * (1.0 - MOTION_BIAS_STRENGTH * align)).max(0.0) as u64
}

/// Motion bias in the plane perpendicular to `up`. `None` biases in 3-D.
/// Axis Y passes `(dx, dz)` and the raw velocity into [`motion_biased_dist2`]
/// (which ignores `vel.y`), so a +Y caller stays bit-identical.
pub(super) fn bias_order(base: u64, vel: DVec3, dx: f64, dy: f64, dz: f64, up: Option<Face>) -> u64 {
    match up {
        None => motion_biased_dist3(base, vel, dx, dy, dz),
        Some(face) => match face.axis() {
            0 => motion_biased_dist2(base, DVec3::new(vel.y, 0.0, vel.z), dy, dz),
            1 => motion_biased_dist2(base, vel, dx, dz),
            _ => motion_biased_dist2(base, DVec3::new(vel.x, 0.0, vel.y), dx, dy),
        },
    }
}

/// Near-lane ordering with the same leading-edge bias as the worker queue.
/// Distance along the up axis stays encoded by [`World::order`]; velocity
/// only reweights candidates in the tangent plane (`None`: all three axes).
fn near_motion_order(world: &World, center: Coord, key: Coord) -> u64 {
    let key = world.fold.fold(key);
    let up = world.live_up();
    let ring = World::order(key, center, up).max(0) as u64;
    let base = ring.saturating_mul(ring).saturating_mul(1024);
    let scale = CHUNK_SIZE as f64;
    bias_order(
        base,
        world.stream_pacer.travel(),
        (key.x - center.x) as f64 * scale,
        (key.y - center.y) as f64 * scale,
        (key.z - center.z) as f64 * scale,
        up,
    )
}

/// Fresh full-res chunk meshing.
pub(in crate::world) struct MeshLane;

impl StreamLane for MeshLane {
    type Key = Coord;
    /// Low floor: admits are relatively cheap.
    const MIN_ADMIT: usize = 4;
    fn candidates(_world: &World, _center: Coord) -> Candidates {
        Candidates::Worklist
    }
    fn seed_set(world: &mut World) -> Option<&mut worklist::RingWorklist> {
        Some(&mut world.mesh_worklist)
    }
    fn admit_worklist(world: &mut World, center: Coord, budget: Budget) {
        admit_coord_worklist::<Self>(world, center, budget);
    }
    fn scratch(world: &mut World) -> &mut AdmitScratch<Coord> {
        &mut world.admit_coords
    }
    fn pending(world: &mut World) -> &mut Sticky {
        &mut world.pending_fresh
    }
    fn order(world: &World, center: Coord, key: Coord) -> u64 {
        near_motion_order(world, center, key)
    }
    fn in_flight(world: &World, key: Coord) -> bool {
        matches!(
            world.chunks.get(&key).map(|l| &l.state),
            Some(MeshState::NeedsMesh { building: true, .. })
        )
    }
    fn on_blocked(world: &mut World, key: Coord) {
        // Start the degrade timer at the eviction EVENT (the gate used to
        // discover blocked chunks by re-scanning the whole worklist against
        // ~15 hash probes each, every pass of a flood). Only light-blockage is
        // timed — a seed evicted for missing neighbour data or leaving the box
        // is re-seeded by its own events, not by a degrade clock.
        if world.chunk_light_blocked(key) {
            world.light_gate.note_blocked(key);
        }
    }
    fn ready(world: &World, key: Coord) -> bool {
        // Cheapest-first: hash, arithmetic, quarantine set, neighbours, light.
        world.is_needs_mesh(key)
            && world.in_mesh_box(key)
            && world.admits_mesh(key)
            && !world
                .quarantined
                .contains(&streaming::FailKey::Mesh { coord: key })
            // Terminal promotion is unconditional: missing neighbour planes
            // read as dark/air, matching the old sync path. Snapshot already
            // fills an absent neighbour that way (`Padded` → AIR, `PaddedLight`
            // → DARK when not degraded).
            && (world.neighbours_have_data(key) || world.light_terminal.contains(&key))
            && (world.light_ready(key)
                || world.light_wait_expired(key)
                || world.light_terminal.contains(&key))
    }
    fn submit(world: &mut World, key: Coord) -> Option<pipeline::Job> {
        // Fully walled solid: the worker mesh would be empty. `None` drops the
        // seed; an edit turns `Air` back into `Dirty`.
        if world.bury_solid_mesh(key) {
            return None;
        }
        world.refresh_tables();
        // If light isn't ready, mesh degraded with assumed-lit neighbours,
        // then remesh when real light arrives — unless the chunk is terminal
        // (missing neighbour light will never arrive: missing planes are dark).
        let degraded = !world.light_ready(key) && !world.light_terminal.contains(&key);
        let (rev, snapshot) = world.snapshot(key, degraded);
        world.mesh_pending_degraded = Some((key, degraded));
        Some(pipeline::Job::Mesh {
            coord: key,
            rev,
            snapshot,
        })
    }
    fn claim(world: &mut World, key: Coord) {
        // Apply the snapshot's degraded flag now that the pool has accepted
        // the job — submit must not mutate lane state (a rejected submit
        // would otherwise park the chunk in `degraded` with nothing in flight,
        // or drop a terminal mark that the retry still needs).
        let degraded = match world.mesh_pending_degraded.take() {
            Some((coord, degraded)) if coord == key => degraded,
            pending => {
                if let Some(leftover) = pending {
                    world.mesh_pending_degraded = Some(leftover);
                }
                debug_assert!(
                    pending.is_none(),
                    "mesh claim for {key:?} with pending for {pending:?}"
                );
                !world.light_ready(key) && !world.light_terminal.contains(&key)
            }
        };
        world.mark_degraded(key, degraded);
        let nhood_quiet = world.light_nhood_quiet(key);
        world.remesh_stats.note_mesh_job(key, nhood_quiet);
        // Set the building flag IN PLACE to claim the mesh job — a whole-state
        // overwrite would silently drop a carried `prev` mesh (leaking its GPU
        // handle and blanking the chunk mid-rebuild). Held until upload retires
        // it or a stale result releases it. Remove from worklist.
        world.mesh_worklist.remove(&key);
        if let Some(loaded) = world.chunks.get_mut(&key) {
            debug_assert!(
                loaded.state.is_needs_mesh(),
                "mesh submit for non-NeedsMesh {key:?}"
            );
            if let MeshState::NeedsMesh { building, .. } = &mut loaded.state {
                if !*building {
                    *building = true;
                    adjust_count(&mut world.building_meshes, false, true);
                }
            }
        }
    }
    fn integrate(world: &mut World, done: pipeline::Done) {
        // The coord stays claimed (`building: true`) until the budgeted upload
        // resolves; `accept_mesh` queues it (or drops+re-seeds if stale).
        if let pipeline::Done::Mesh { coord, rev, data } = done {
            world.accept_mesh(coord, rev, data);
        }
    }
}

/// LOD2 column sections.
pub(in crate::world) struct SectionLane;

impl StreamLane for SectionLane {
    type Key = SectionPos;
    /// Modest floor: heavy work runs off-thread.
    const MIN_ADMIT: usize = 4;
    const FAR: bool = true;
    fn candidates(_world: &World, _center: Coord) -> Candidates {
        Candidates::Geometry
    }
    fn seed_set(_world: &mut World) -> Option<&mut worklist::RingWorklist> {
        None
    }
    fn scratch(world: &mut World) -> &mut AdmitScratch<SectionPos> {
        &mut world.admit_sections
    }
    fn pending(world: &mut World) -> &mut Sticky {
        &mut world.pending_sections
    }
    fn for_each_geometry(
        world: &World,
        center: Coord,
        rank: &mut Vec<(u64, SectionPos)>,
        mut visit: impl FnMut(SectionPos),
    ) {
        let hole = |s: &SectionPos| {
            !world.sections.contains_key(s)
                && !world.quarantined.contains(&streaming::FailKey::Section { pos: *s })
        };
        // At rest every hole is a candidate. While moving, the deadline submits
        // only a few, so keep the nearest handful of holes. Residency is the
        // cheap filter and runs first: the nearest sections are almost always
        // resident. The coverage proof runs only on a hole that would enter
        // the shortlist.
        if world.stream_pacer.effort() >= 1.0 || world.stream_pacer.boosting() {
            for s in world.section_desired.iter().copied().filter(hole) {
                if !world.coverage_skips(center, s) {
                    visit(s);
                }
            }
            return;
        }
        const CAP: usize = 32;
        rank.clear();
        for s in world.section_desired.iter().copied().filter(hole) {
            let d = Self::order(world, center, s);
            if rank.len() == CAP && d >= rank[CAP - 1].0 {
                continue;
            }
            if world.coverage_skips(center, s) {
                continue;
            }
            if rank.len() == CAP {
                rank[CAP - 1] = (d, s);
            } else {
                rank.push((d, s));
            }
            let mut i = rank.len() - 1;
            while i > 0 && rank[i].0 < rank[i - 1].0 {
                rank.swap(i, i - 1);
                i -= 1;
            }
        }
        for &(_, s) in rank.iter() {
            visit(s);
        }
    }
    fn order(world: &World, center: Coord, key: SectionPos) -> u64 {
        let span = key.span() as i64;
        let (x, z) = world.net_column(center, key.min_x() as i64 + span / 2, key.min_z() as i64 + span / 2);
        let cs = CHUNK_SIZE as i64;
        let (psx, psz) = ((center.x as i64 * cs).div_euclid(span), (center.z as i64 * cs).div_euclid(span));
        let (sx, sz) = (x.div_euclid(span), z.div_euclid(span));
        (sx - psx).unsigned_abs().max((sz - psz).unsigned_abs())
    }
    fn dist2(world: &World, center: Coord, key: SectionPos) -> Option<u64> {
        // Sections are an XZ heightfield: pass the eye altitude as `wy` so
        // `dy` is 0 and a vertical move does not reshuffle them. A neighbour
        // chart's column is measured where the net puts it, or its storage
        // distance admits the far side of the box ahead of the seam.
        let span = key.span() as i64;
        let py = center.y as i64 * CHUNK_SIZE as i64 + CHUNK_SIZE as i64 / 2;
        let (cx, cz) = world.net_column(center, key.min_x() as i64 + span / 2, key.min_z() as i64 + span / 2);
        let base = player_dist2(center, cx, py, cz);
        // Bias by motion direction so leading edge fills first.
        let cs = CHUNK_SIZE as i64;
        let (px, pz) = (center.x as i64 * cs + cs / 2, center.z as i64 * cs + cs / 2);
        Some(motion_biased_dist2(
            base,
            world.section_vel,
            (cx - px) as f64,
            (cz - pz) as f64,
        ))
    }
    fn in_flight(world: &World, key: SectionPos) -> bool {
        world.sections.contains_key(&key)
    }
    fn prepare_admit(world: &mut World) {
        world.section_standin_slack = world.count_section_standins();
    }
    fn ready(world: &World, _key: SectionPos) -> bool {
        // Sections only. Near-field chunks never consume this budget; a large
        // view must not starve covering. In-flight claims count now. A Ready
        // finer tile under a hole keeps its slot, and the hole is admitted
        // past the floor by however many such tiles there are.
        world.section_budget_used() < world.sections_allowed() + world.section_standin_slack
    }
    fn submit(world: &mut World, key: SectionPos) -> Option<pipeline::Job> {
        world.refresh_tables();
        // Mint the claim token here; `claim` (which always follows an accepted
        // submit for the same key) installs it on the `Meshing` entry.
        world.section_claim_seq = world.section_claim_seq.wrapping_add(1);
        let token = pipeline::ClaimToken(world.section_claim_seq);
        world.section_pending_claim = Some((key, token));
        Some(pipeline::Job::Section {
            pos: key,
            epoch: world.section_epoch,
            token,
            generator: world.generator.clone(),
            edits: world.edits_for_section(key),
            tables: world.tables.get(),
        })
    }
    fn claim(world: &mut World, key: SectionPos) {
        let token = match world.section_pending_claim.take() {
            Some((pos, token)) if pos == key => token,
            other => {
                // A far-cap rejection leaves the pending token set; a later
                // claim for a different key must not consume it.
                if let Some(pending) = other {
                    world.section_pending_claim = Some(pending);
                }
                debug_assert!(
                    false,
                    "section claim for {key:?} without its submit ({other:?})"
                );
                pipeline::ClaimToken(world.section_claim_seq)
            }
        };
        world.dirty_sections.remove(&key);
        let prev = world.sections.insert(key, SectionState::Meshing { token });
        adjust_count(
            &mut world.meshing_sections,
            matches!(prev, Some(SectionState::Meshing { .. })),
            true,
        );
    }
    fn integrate(world: &mut World, done: pipeline::Done) {
        if let pipeline::Done::Section {
            pos,
            epoch,
            token,
            meshes,
        } = done
        {
            // A result from a retired ladder epoch, or for a claim replaced
            // after unload/re-admission, is dropped here — it must not queue an
            // upload that would capture a same-position replacement.
            let live = epoch == world.section_epoch
                && matches!(world.sections.get(&pos),
                    Some(SectionState::Meshing { token: t }) if *t == token);
            if live {
                let bytes = meshes.vertex_bytes();
                world
                    .section_upload_queue
                    .push_back((pos, token, bytes, meshes));
            }
        }
    }
}

/// Cross-chunk light settling.
pub(in crate::world) struct LightLane;

impl StreamLane for LightLane {
    type Key = Coord;
    /// High floor: settle must drain fast so meshing can start.
    const MIN_ADMIT: usize = 32;
    fn candidates(_world: &World, _center: Coord) -> Candidates {
        Candidates::Worklist
    }
    fn seed_set(world: &mut World) -> Option<&mut worklist::RingWorklist> {
        Some(&mut world.light_worklist)
    }
    fn admit_worklist(world: &mut World, center: Coord, budget: Budget) {
        admit_coord_worklist::<Self>(world, center, budget);
    }
    fn scratch(world: &mut World) -> &mut AdmitScratch<Coord> {
        &mut world.admit_coords
    }
    fn pending(world: &mut World) -> &mut Sticky {
        &mut world.light_pending
    }
    fn order(world: &World, center: Coord, key: Coord) -> u64 {
        near_motion_order(world, center, key)
    }
    fn in_flight(world: &World, key: Coord) -> bool {
        world.light_inflight.contains(&key)
    }
    fn ready(world: &World, key: Coord) -> bool {
        world.admits_light(key)
    }
    fn on_blocked(world: &mut World, key: Coord) {
        // Outside the reduced loading window: the chunk still owes this settle.
        world.owe_light(key);
    }
    fn submit(world: &mut World, key: Coord) -> Option<pipeline::Job> {
        // `trivial_light` is decided at store time. A worklist seed here is a
        // real re-settle (neighbour border / edit) and must run the flood.
        if !world.lighting
            || !world.chunks.contains_key(&key)
            || world
                .quarantined
                .contains(&streaming::FailKey::Light { coord: key })
        {
            return None;
        }
        world.refresh_tables();
        let shell = world.capture_face_shell(key);
        let sky = world.generator.sky(key);
        let (alt0, ceiling) = match sky {
            Sky::Open => (0, light::ignored_ceiling()),
            Sky::Axis(face) => (FaceFrame::new(face).chunk_alt0(key), world.capture_ceiling(key)),
        };
        let snapshot = pipeline::LightSnapshot {
            chunk: Arc::clone(&world.chunks[&key].chunk),
            shell,
            ceiling,
            sky,
            alt0,
            tables: world.tables.get(),
        };
        Some(pipeline::Job::Light {
            coord: key,
            epoch: world.light_epoch,
            light_gen: world.chunks[&key].light_gen,
            snapshot: Box::new(snapshot),
        })
    }
    fn claim(world: &mut World, key: Coord) {
        world.light_worklist.remove(&key);
        world.light_inflight.insert(key);
    }
    fn note_admitted(world: &mut World, n: usize) {
        world.counters.light_admitted += n as u64;
        world.counters.light_admitted_last = n;
    }
    fn integrate(world: &mut World, done: pipeline::Done) {
        // `accept_light` owns the claim rule (release-or-transfer on every
        // consumed result) — see its doc for the epoch soundness argument.
        if let pipeline::Done::Light {
            coord,
            epoch,
            light_gen,
            grid,
        } = done
        {
            world.accept_light(coord, epoch, light_gen, grid);
        }
    }
}
