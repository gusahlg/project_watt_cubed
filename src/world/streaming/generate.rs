//! Generation: gathering missing chunks into column runs, submitting them, the spawn slab, and
//! storing landed data.

use super::*;

/// The strike/quarantine identity of a panicked job — the per-lane key
/// [`World::fail_job`] counts strikes against. A generate failure is keyed by
/// its whole column: the failing chunk inside a column job is unknown, and the
/// span requested for a column varies with the view, so per-span keys would
/// never accumulate strikes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(in crate::world) enum FailKey {
    Column { key: ColumnKey },
    /// One `Open` chunk. Not a [`Column`](Self::Column): a PosY column at the
    /// same `(cx, cz)` must keep its own strike count.
    Open { coord: Coord },
    Mesh { coord: Coord },
    Light { coord: Coord },
    Section { pos: SectionPos },
}

impl FailKey {
    pub(super) fn of(key: &pipeline::JobKey) -> FailKey {
        match key {
            pipeline::JobKey::Column { key, .. } => FailKey::Column { key: *key },
            pipeline::JobKey::Open { coord } => FailKey::Open { coord: *coord },
            pipeline::JobKey::Mesh { coord } => FailKey::Mesh { coord: *coord },
            pipeline::JobKey::Light { coord } => FailKey::Light { coord: *coord },
            pipeline::JobKey::Section { pos, .. } => FailKey::Section { pos: *pos },
        }
    }
}

/// One generate admission: a face column, or a single `Open` chunk.
/// `Open` is not encoded as a one-chunk PosY column — that key collided with
/// a real PosY column's [`FailKey`] at the same `(cx, cz)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::world) enum GenRun {
    Column { key: ColumnKey, lo: i32, hi: i32 },
    Open { coord: Coord },
}

impl GenRun {
    /// The run that generates chunk `coord` alone, under its sky `sky`.
    pub(in crate::world) fn of_chunk(coord: Coord, sky: Sky) -> GenRun {
        match sky {
            Sky::Axis(face) => {
                let (key, alt) = ColumnKey::of(face, coord);
                GenRun::Column { key, lo: alt, hi: alt }
            }
            Sky::Open => GenRun::Open { coord },
        }
    }

    /// The strike/quarantine identity of this run's job.
    pub(in crate::world) fn fail_key(self) -> FailKey {
        match self {
            GenRun::Column { key, .. } => FailKey::Column { key },
            GenRun::Open { coord } => FailKey::Open { coord },
        }
    }

    /// The column the worker fills and its inclusive altitude range. `Open` is the one-layer
    /// PosY column `generate_column` round-trips.
    fn span(self) -> (ColumnKey, i32, i32) {
        match self {
            GenRun::Column { key, lo, hi } => (key, lo, hi),
            GenRun::Open { coord } => (ColumnKey { face: Face::PosY, a: coord.x, b: coord.z }, coord.y, coord.y),
        }
    }

    /// Every chunk the run generates, lowest altitude first.
    pub(super) fn coords(self) -> impl Iterator<Item = Coord> {
        let (key, lo, hi) = self.span();
        (lo..=hi).map(move |alt| key.chunk(alt))
    }

    /// Chunk the run is ordered from. A column uses its low end: every layer
    /// shares the tangent coordinates, and +Y ordering ignores altitude.
    fn anchor(self) -> Coord {
        let (key, lo, _) = self.span();
        key.chunk(lo)
    }
}

/// The generate run list. Kept across frames: rebuilding it walks every coord, and a 2 ms
/// budget that pays that walk each pass admits only the floor.
#[derive(Default)]
pub(in crate::world) struct GenCursor {
    /// Runs for the current data box not yet submitted, the next one last.
    pub(in crate::world) runs: Vec<(u64, GenRun)>,
    /// What `runs` was gathered for. A mismatch, or `dirty`, rebuilds the queue instead of
    /// scanning the data box again.
    key: Option<GenKey>,
    pub(in crate::world) dirty: bool,
    /// Velocity the queued runs were last ordered with. A change re-sorts; a standing eye does not.
    vel: DVec3,
    ranked: bool,
}

/// The data box a run list was gathered for, with the loading radii, the travel heading and the
/// spawn slab.
#[derive(PartialEq)]
struct GenKey {
    center: Coord,
    data_box: ChunkBox,
    lh: i32,
    lv: i32,
    heading: i8,
    slab: Option<ChunkBox>,
}

/// Forward-progress floor for the generation lane: admit at least this many
/// columns before the deadline can stop it, so a boundary-cross flood still
/// makes strict progress each frame under a tight budget (the same floor role
/// [`super::StreamLane::MIN_ADMIT`] plays for the per-chunk lanes).
const GEN_MIN_ADMIT: usize = 8;

/// Group chunk coords into generate runs.
///
/// `Axis(f)` chunks that share a [`ColumnKey`] become one inclusive altitude
/// run. With `span_loaded` (the spawn slab) every coord of that column in the
/// input is recorded, loaded or not, and a run is emitted only when something
/// in a contiguous group is missing — a uniform PosY slab therefore submits
/// its full altitude span. Without it (`request_region_data`) only missing
/// coords are recorded, and a gap stays inside the run when every chunk
/// between the ends has the same sky (a loaded hole of the same face). A gap
/// whose sky differs splits the run, so a PosY job never generates an Open coord.
///
/// `Open` is one job per missing chunk ([`GenRun::Open`]), not merged with
/// its `(cx, cz)` neighbours and not encoded as a PosY column. Voxel
/// generation still calls `generate_column` with that PosY encoding so
/// `key.chunk(cy)` round-trips; only the claim and quarantine key differ.
/// `accept_column` does not install a ceiling and `store_chunk` does not
/// record the chunk when `sky` is `Open`.
///
/// `skip_quarantine` drops a quarantined run before it can take a slot.
/// The slab path leaves it false so the submit rejects the run and the
/// caller keeps `pending_gen` set.
fn gather_column_runs(
    coords: impl IntoIterator<Item = Coord>,
    mut sky_of: impl FnMut(Coord) -> Sky,
    mut present: impl FnMut(Coord) -> bool,
    mut quarantined: impl FnMut(FailKey) -> bool,
    span_loaded: bool,
    skip_quarantine: bool,
) -> Vec<GenRun> {
    let mut axis: FastMap<ColumnKey, Vec<(i32, bool)>> = FastMap::default();
    let mut open_seen: FastSet<Coord> = FastSet::default();
    let mut runs: Vec<GenRun> = Vec::new();
    for coord in coords {
        let missing = !present(coord);
        // Without `span_loaded` a present chunk records nothing, whatever its sky.
        if !span_loaded && !missing {
            continue;
        }
        match sky_of(coord) {
            Sky::Open => {
                if skip_quarantine && quarantined(FailKey::Open { coord }) {
                    continue;
                }
                if missing && open_seen.insert(coord) {
                    runs.push(GenRun::Open { coord });
                }
            }
            Sky::Axis(face) => {
                let (key, alt) = ColumnKey::of(face, coord);
                if skip_quarantine && quarantined(FailKey::Column { key }) {
                    continue;
                }
                axis.entry(key).or_default().push((alt, missing));
            }
        }
    }
    for (key, mut alts) in axis {
        alts.sort_unstable_by_key(|p| p.0);
        let mut deduped: Vec<(i32, bool)> = Vec::new();
        for (alt, missing) in alts {
            if let Some(last) = deduped.last_mut() {
                if last.0 == alt {
                    last.1 |= missing;
                    continue;
                }
            }
            deduped.push((alt, missing));
        }
        if span_loaded {
            let mut i = 0;
            while i < deduped.len() {
                let mut j = i;
                while j + 1 < deduped.len() && deduped[j + 1].0 == deduped[j].0 + 1 {
                    j += 1;
                }
                if deduped[i..=j].iter().any(|p| p.1) {
                    runs.push(GenRun::Column { key, lo: deduped[i].0, hi: deduped[j].0 });
                }
                i = j + 1;
            }
        } else {
            let mut start = 0;
            for i in 0..deduped.len() {
                let split = i + 1 == deduped.len()
                    || !gap_same_sky(key, deduped[i].0, deduped[i + 1].0, &mut sky_of);
                if split {
                    runs.push(GenRun::Column {
                        key,
                        lo: deduped[start].0,
                        hi: deduped[i].0,
                    });
                    start = i + 1;
                }
            }
        }
    }
    // Open occupies the slot the old PosY one-chunk encoding sorted into, so
    // a +Y gather keeps today's order.
    runs.sort_unstable_by_key(|run| match run {
        GenRun::Column { key, lo, .. } => (key.face.index(), key.a, key.b, *lo),
        GenRun::Open { coord } => (Face::PosY.index(), coord.x, coord.z, coord.y),
    });
    runs
}

fn gap_same_sky(key: ColumnKey, lo: i32, hi: i32, sky_of: &mut impl FnMut(Coord) -> Sky) -> bool {
    let want = Sky::Axis(key.face);
    (lo + 1..hi).all(|alt| sky_of(key.chunk(alt)) == want)
}

/// Column priority: tangent chess (not the ×2 along-axis weight — a column
/// job is the whole run). `None` is 3-D chess of the anchor. +Y reads only
/// XZ, matching the old `(cx, cz)` key.
fn column_order(center: Coord, vel: DVec3, anchor: Coord, up: Option<Face>) -> u64 {
    let across = match up {
        None => anchor.chess3(center) as u64,
        Some(face) => anchor.across(center, face) as u64,
    };
    let scale = CHUNK_SIZE as f64;
    super::bias_order(
        across.saturating_mul(across).saturating_mul(1024),
        vel,
        f64::from(anchor.x - center.x) * scale,
        f64::from(anchor.y - center.y) * scale,
        f64::from(anchor.z - center.z) * scale,
        up,
    )
}

impl World {
    /// The [`GenerateLane`](lanes::GenerateLane) producer's body: queue worker
    /// jobs for missing chunks in the data box, grouped into vertical columns so
    /// the `cy`-invariant column profile is sampled once per column, nearest
    /// column first, up to `deadline`. Self-gates on `pending_gen` (raised on a
    /// boundary cross and by a generate strike-out re-request). Column
    /// granularity means it keeps its own gather/claim rather than the per-chunk
    /// [`admit`](super::admit) loop, but the forward-progress floor + time budget
    /// are the one shared rule ([`admission_exhausted`](super::admission_exhausted)).
    /// The centre chunk is generated synchronously in `stream` only when it is
    /// missing and not already claimed; `accept_column` lands these results.
    ///
    /// The run list is gathered once per data box and drained across frames.
    /// Walking every coord each pass (classify, then sky) blows the 2 ms budget
    /// before any job is submitted, so an open asteroid admits only the floor
    /// and spends the frame on the walk.
    pub(in crate::world) fn request_region_data(
        &mut self,
        center: Coord,
        budget: Budget,
    ) -> Progress {
        if !self.pending_gen.take() {
            return Progress::Idle;
        }
        // Generation for the whole view would land behind a window the
        // confirmed jump collapses next pass; a dropped jump costs one pass.
        if self.stream_pacer.holding() {
            self.pending_gen.set();
            return Progress::Partial {
                remaining: self.gen_cursor.runs.len() as u32,
            };
        }
        if !self.gen_cursor_matches(center) {
            self.rebuild_gen_cursor(center);
        }
        if self.gen_cursor.runs.is_empty() {
            return Progress::Idle;
        }
        let slots = match self.workers.as_ref() {
            Some(w) => w.near_slots_free(),
            None => usize::MAX,
        };
        if slots == 0 {
            self.pending_gen.set();
            return Progress::Partial {
                remaining: self.gen_cursor.runs.len() as u32,
            };
        }
        let deadline = super::lanes::paced_deadline(self, budget);
        let vel = self.stream_pacer.travel();
        if !self.gen_cursor.ranked || self.gen_cursor.vel != vel {
            let up = self.live_up();
            let fold = self.fold;
            for entry in &mut self.gen_cursor.runs {
                let anchor = entry.1.anchor();
                entry.0 = column_order(center, vel, fold.fold(anchor), up);
            }
            // Farthest first, so the nearest pops off the end; ties keep their submission order.
            self.gen_cursor.runs.sort_by(|a, b| b.0.cmp(&a.0));
            self.gen_cursor.vel = vel;
            self.gen_cursor.ranked = true;
        }
        let min_admit = self.stream_pacer.floor(GEN_MIN_ADMIT);
        let mut admitted = 0usize;
        while let Some(&(_, run)) = self.gen_cursor.runs.last() {
            if super::admission_exhausted(admitted, min_admit, deadline) {
                break;
            }
            // A quarantined run drops, matching `gather_column_runs`'s `skip_quarantine`, as does
            // one already claimed or left behind by the loading window. Pool backpressure is a
            // different `false` from submit and leaves the run queued.
            let drop = self.quarantined.contains(&run.fail_key())
                || run.coords().all(|c| self.claimed(c))
                || !self.run_in_load(run);
            if !drop {
                if !self.submit_run(run) {
                    break;
                }
                admitted += 1;
            }
            self.gen_cursor.runs.pop();
        }
        if self.gen_cursor.runs.is_empty() {
            Progress::Idle
        } else {
            self.pending_gen.set();
            Progress::Partial {
                remaining: self.gen_cursor.runs.len() as u32,
            }
        }
    }

    /// Whether the run list is still the one for `center`'s data box.
    fn gen_cursor_matches(&self, center: Coord) -> bool {
        !self.gen_cursor.dirty && self.gen_cursor.key == Some(self.gen_key(center))
    }

    /// What a run list gathered now for `center` is for. The data box carries the up, the view and
    /// the grown window; the loading window rides on top.
    fn gen_key(&self, center: Coord) -> GenKey {
        GenKey {
            center,
            data_box: self.data_box(center),
            lh: self.load_h,
            lv: self.load_v,
            heading: self.load_heading,
            slab: self.spawn_slab,
        }
    }

    /// Classify the data box once, store uniform chunks, and queue the rest.
    fn rebuild_gen_cursor(&mut self, center: Coord) {
        self.counters.gen_cursor_rebuilds += 1;
        let mut coords: Vec<Coord> = self.view_coords(self.load_data_box(center)).collect();
        if let Some(slab) = self.spawn_slab {
            coords.extend(self.view_coords(slab));
        }
        if !self.loading_full() {
            let window = self.load_window();
            coords.retain(|&c| self.admits_new_in(window, c));
        }
        let mut stored = false;
        coords.retain(|c| {
            if self.store_if_free(*c) {
                stored = true;
                false
            } else {
                true
            }
        });
        if stored {
            self.refresh_spawn_slab();
        }
        let runs = gather_column_runs(
            coords,
            |c| self.generator.sky(c),
            |c| self.claimed(c),
            |fail| self.quarantined.contains(&fail),
            false,
            true,
        );
        self.gen_cursor.runs.clear();
        self.gen_cursor.runs.extend(runs.into_iter().rev().map(|run| (0, run)));
        self.gen_cursor.key = Some(self.gen_key(center));
        self.gen_cursor.dirty = false;
        self.gen_cursor.ranked = false;
    }

    /// Chunk `coord` is loaded, or a generate job claims it.
    fn claimed(&self, coord: Coord) -> bool {
        self.chunks.contains_key(&coord) || self.generating.contains(&coord)
    }

    /// Land a generated column: install the skylight ceiling from the worker's
    /// heights (plus any edited-roof raise) before storing, so `trivial_light`
    /// hits the cache instead of sampling the generator on this thread.
    pub(in crate::world) fn accept_column(
        &mut self,
        key: ColumnKey,
        chunks: Vec<(Coord, Chunk)>,
        heights: Box<ColumnHeights>,
    ) {
        // Only cache when at least one axis chunk will actually land — an install
        // with no `column_chunks` bump would leak in `ceilings` forever. Open
        // runs encode a PosY key and must not install a ceiling.
        if chunks.iter().any(|(coord, _)| {
            self.will_accept_chunk(*coord) && matches!(self.generator.sky(*coord), Sky::Axis(_))
        }) {
            self.install_ceiling(key, &heights);
        }
        for (coord, chunk) in chunks {
            self.generating.remove(&coord);
            self.accept_chunk(coord, chunk);
        }
        self.refresh_spawn_slab();
    }

    /// `accept_chunk`'s store predicate: inside the loading data window or the
    /// requested spawn slab, not behind the player, and not yet loaded.
    pub(super) fn will_accept_chunk(&self, coord: Coord) -> bool {
        if self.chunks.contains_key(&coord) {
            return false;
        }
        self.admits_new(coord)
    }

    /// A gathered run is still inside the loading window. The full window's
    /// cursor was built from the full data box, so every run is.
    fn run_in_load(&self, run: GenRun) -> bool {
        if self.loading_full() {
            return true;
        }
        let window = self.load_window();
        run.coords().any(|c| self.admits_new_in(window, c))
    }

    /// [`admits_new`](Self::admits_new) under a reduced loading window the caller already built:
    /// the spawn slab, or inside `window` (`None` before the first stream).
    fn admits_new_in(&self, window: Option<LoadWindow>, coord: Coord) -> bool {
        self.spawn_slab.is_some_and(|slab| self.view_contains(slab, coord))
            || window.is_some_and(|w| w.covers(self.fold.fold(coord), true))
    }

    /// Ensure every chunk within the data box of `center` exists (voxel data
    /// only). Cheap and GPU-free, so it also seeds headless queries.
    pub(in crate::world) fn ensure_region_data(&mut self, center: Coord) {
        let coords: Vec<Coord> = self.view_coords(self.data_box(center)).collect();
        for coord in coords {
            self.ensure_data(coord);
        }
    }

    /// Collision halo around an eye chunk. `Some(face)`: ±1 across the face,
    /// ±2 along it (+Y is the old 3×3 columns, two layers below through two
    /// above). `None`: ±2 on every axis.
    pub(in crate::world) fn collision_slab(center: Coord, up: Option<Face>) -> ChunkBox {
        match up {
            None => ChunkBox::with_up(center, 2, 2, None),
            Some(face) => ChunkBox::with_up(center, 1, 2, Some(face)),
        }
    }

    fn slab_up(&self, center: Coord) -> Option<Face> {
        match self.generator.sky(center) {
            Sky::Axis(face) => Some(face),
            Sky::Open => None,
        }
    }

    /// Request the collision slab around `pos` from the worker pool. Does not
    /// generate on this thread — [`spawn_ready`](Self::spawn_ready) is true
    /// once every chunk of the box has loaded. Teleports and net snaps use
    /// the same request (physics freezes until it lands).
    pub fn prepare_around(&mut self, pos: DVec3) {
        let (near, far) = self.place_eyes(pos);
        let (c, f) = (eye_chunk(near), eye_chunk(far));
        self.adopt_fold(c);
        let up = self.slab_up(c);
        let slab = Self::collision_slab(c, up);
        self.publish_spawn_view(c, f, up, slab);
        self.submit_slab_columns(slab);
        self.pending_gen.set();
        self.spawn_slab = (!self.slab_loaded(slab)).then_some(slab);
    }

    /// Synchronously generate the collision slab. Headless callers (tests,
    /// anything that queries voxels before a stream pass).
    pub fn ensure_around(&mut self, pos: DVec3) {
        let c = eye_chunk(self.stream_eye(pos));
        self.adopt_fold(c);
        let coords: Vec<Coord> = self.view_coords(Self::collision_slab(c, self.slab_up(c))).collect();
        for coord in coords {
            self.ensure_data(coord);
        }
    }

    /// True once every chunk of the requested spawn/teleport slab is loaded,
    /// or no slab is outstanding.
    pub fn spawn_ready(&self) -> bool {
        self.spawn_slab.is_none_or(|slab| self.slab_loaded(slab))
    }

    /// Every chunk of `slab` is loaded.
    fn slab_loaded(&self, slab: ChunkBox) -> bool {
        self.view_coords(slab).all(|c| self.chunks.contains_key(&c))
    }

    /// Drive in-flight generate jobs until the spawn slab is loaded. Tests
    /// only — the live path drains through [`stream`](Self::stream).
    #[cfg(test)]
    pub fn drive_spawn_ready(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(30);
        while !self.spawn_ready() {
            assert!(
                Instant::now() < deadline,
                "spawn slab did not land: {}",
                self.entry_debug()
            );
            if let Some(slab) = self.spawn_slab {
                self.submit_slab_columns(slab);
            }
            let mut got = false;
            while let Some(done) = self.workers.as_ref().and_then(pipeline::Workers::try_recv) {
                self.integrate_worker_result(done);
                got = true;
            }
            self.refresh_spawn_slab();
            if !got {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }

    fn refresh_spawn_slab(&mut self) {
        if self.spawn_slab.is_some_and(|slab| self.slab_loaded(slab)) {
            self.spawn_slab = None;
        }
    }

    fn submit_slab_columns(&mut self, slab: ChunkBox) {
        let coords: Vec<Coord> = self.view_coords(slab).collect();
        let runs = gather_column_runs(
            coords,
            |c| self.generator.sky(c),
            |c| self.claimed(c),
            |fail| self.quarantined.contains(&fail),
            true,
            false,
        );
        let mut remaining = false;
        for run in runs {
            if !self.submit_run(run) {
                remaining = true;
            }
        }
        if remaining {
            self.pending_gen.set();
        }
    }

    /// Submit one run's job and claim its missing chunks. `false` means the
    /// pool rejected it (backpressure, shutdown, or quarantine) so the caller
    /// must retry.
    fn submit_run(&mut self, run: GenRun) -> bool {
        if self.quarantined.contains(&run.fail_key()) {
            return false;
        }
        let generator = self.generator.clone();
        let job = match run {
            GenRun::Column { key, lo, hi } => pipeline::Job::GenerateColumn {
                key,
                range: lo..=hi,
                generator,
                edits: run.coords().filter_map(|c| Some((c, self.chunk_edits(c)?))).collect(),
            },
            // The worker still fills it through the PosY one-chunk
            // `generate_column` encoding; the claim key does not.
            GenRun::Open { coord } => pipeline::Job::GenerateOpen {
                coord,
                generator,
                edits: self.chunk_edits(coord).unwrap_or_default(),
            },
        };
        let accepted = self.worker_pool().submit(job);
        if accepted {
            for c in run.coords() {
                if !self.chunks.contains_key(&c) {
                    self.generating.insert(c);
                }
            }
        }
        accepted
    }

    /// Chunk `coord`'s edit overlay as `(index, block)` pairs, for a job that replays it.
    pub(super) fn chunk_edits(&self, coord: Coord) -> Option<Vec<(usize, crate::block::registry::BlockId)>> {
        self.edits.get(&coord).map(|cells| cells.iter().map(|(&i, &id)| (i, id)).collect())
    }

    /// Air and uniform bulk never take a worker slot: same `Chunk`, same edit replay,
    /// same light fast path as a generated uniform chunk.
    fn store_if_free(&mut self, coord: Coord) -> bool {
        if self.claimed(coord) {
            return false;
        }
        let id = match self.generator.classify(coord) {
            Classify::Mixed => return false,
            Classify::Air => AIR,
            Classify::Uniform(id) => id,
        };
        let chunk = Chunk::from_data(coord.x, coord.y, coord.z, ChunkData::Uniform(id));
        self.store_chunk(coord, chunk);
        true
    }

    /// Generate a chunk's data if it isn't loaded, replaying any saved edits on it.
    /// Uses `generate_column` so the ceiling heights come from the same sample
    /// the voxels did — never a second `height()` walk on this thread.
    /// Skips coords already claimed in `generating`: the async result is imminent.
    pub(in crate::world) fn ensure_data(&mut self, coord: Coord) {
        if self.claimed(coord) {
            return;
        }
        if self.store_if_free(coord) {
            self.refresh_spawn_slab();
            return;
        }
        let sky = self.generator.sky(coord);
        // `Open` takes the one-layer PosY encoding and no ceiling.
        let (key, alt, _) = GenRun::of_chunk(coord, sky).span();
        let (chunks, heights) = self.generator.generate_column(key, alt..=alt);
        if matches!(sky, Sky::Axis(_)) {
            self.install_ceiling(key, &heights);
        }
        let data = chunks
            .into_iter()
            .next()
            .map(|(_, data)| data)
            .expect("generate_column emits the requested layer");
        let placed = key.chunk(alt);
        let chunk = Chunk::from_data(placed.x, placed.y, placed.z, data);
        self.store_chunk(coord, chunk);
        self.refresh_spawn_slab();
    }

    /// Insert freshly generated data: replay the edit overlay, then register
    /// the chunk. A uniform-air chunk (after replay) can never produce
    /// geometry, so it is born `meshed` with no mesh — no worker job, no
    /// upload, nothing drawn.
    pub(super) fn store_chunk(&mut self, coord: Coord, mut chunk: Chunk) {
        if let Some(edits) = self.edits.get(&coord) {
            for (&index, &id) in edits {
                chunk.set_index(index, id);
            }
        }
        // Uniform non-solid chunks produce no geometry, so start Air.
        // Check solidity, not AIR id, for future non-solid blocks.
        let born_air = chunk
            .uniform()
            .is_some_and(|id| !self.registry.is_solid(id));
        let state = if born_air {
            MeshState::Air
        } else {
            MeshState::needs_mesh()
        };
        // Born-air is already settled; a sky ring can complete without a
        // single upload.
        if born_air {
            self.note_settled();
        }
        // No flood-fill here; occlusion rebuild computes connectivity lazily.
        let chunk = std::sync::Arc::new(chunk);
        // Liveness check: coord must not be claimed in generating (would shadow data).
        debug_assert!(
            !self.generating.contains(&coord),
            "storing {coord:?} still claimed in generating — a stuck generate claim"
        );
        self.light_claim_seq = self.light_claim_seq.wrapping_add(1);
        let light_gen = self.light_claim_seq;
        self.chunks.insert(
            coord,
            Loaded {
                chunk: std::sync::Arc::clone(&chunk),
                state,
                rev: 0,
                connectivity: None,
                visible: true,
                light: None,
                has_blocklight: false,
                light_reseed: false,
                light_gen,
                mesh_hash: None,
            },
        );
        // Ceiling-cache lifetime: the column's last layer out drops the entry.
        // Open chunks have no ceiling and are not recorded here.
        if let Sky::Axis(face) = self.generator.sky(coord) {
            let (key, alt) = ColumnKey::of(face, coord);
            let ys = self.column_chunks.entry(key).or_default();
            if !ys.contains(&alt) {
                ys.push(alt);
                ys.sort_unstable_by(|a, b| b.cmp(a));
            }
        }
        // Occlusion learns of the new chunk through the fill queue (bounded
        // drain per rebuild) — no per-rebuild missing-connectivity scan.
        if self.occlusion_enabled() {
            self.conn_fill_queue.push_back(coord);
        }
        // Light: try the analytic fast path first — a uniform-opaque chunk settles
        // to all-dark and an above-surface uniform-air chunk to full sky with no
        // flood. A trivial grid publishes synchronously (which fans the border to
        // its neighbours); only the residual Dense band seeds the settle worklist.
        if self.lighting {
            if self.light_inflight.contains(&coord) {
                // A previous Loaded at this coord still owns the inflight
                // claim. Skip trivial publish (it would steal that claim via
                // settle_light) and seed so we resettle after the stale Done
                // is consumed against the old generation.
                self.seed_light(coord, super::LightSeed::Store);
                self.light_pending.set();
            } else {
                match self.trivial_light(coord, &chunk) {
                    Some(grid) => self.settle_light(coord, grid),
                    None => {
                        self.seed_light(coord, super::LightSeed::Store);
                        self.light_pending.set();
                    }
                }
            }
        }
        // A new chunk changes what the BFS can reach — topology class:
        // debounced (an unclassified fresh chunk is over-draw, never a hole).
        self.occlusion_topo_dirty.set();
        // And it changes the near-field coverage picture the section skip
        // reads: re-arm the far-field lane so LOD reacts to ANY chunk
        // creation instead of waiting for a boundary crossing.
        if self.lod2 {
            self.pending_sections.set();
        }
        // Seed this chunk and 6 neighbours; a neighbour may have been
        // blocked waiting on this data even if itself uniform air.
        // A buried solid is `Air` with no mesh. New neighbour voxels (edit
        // replay on load) can open a face, so that chunk meshes again.
        // Uniform non-solid `Air` is born with nothing to draw.
        self.seed_mesh(coord);
        for face in Face::ALL {
            let n = self.neighbour(coord, face);
            let (is_air, fill) = match self.chunks.get(&n) {
                Some(l) if matches!(l.state, MeshState::Air) => (true, l.chunk.uniform()),
                _ => (false, None),
            };
            let buried = is_air && !fill.is_some_and(|id| !self.registry.is_solid(id));
            if buried {
                if let Some(loaded) = self.chunks.get_mut(&n) {
                    loaded.state = MeshState::needs_mesh();
                }
            }
            let ready = matches!(
                self.chunks.get(&n).map(|l| &l.state),
                Some(MeshState::Ready(_))
            );
            self.seed_mesh(n);
            // A Ready neighbour meshed without this chunk (terminal promotion
            // at a load-set edge, or the neighbour unloaded after the mesh).
            // Rebuild so the final look picks up the new border; worklist
            // seeding alone cannot, since Ready fails `is_needs_mesh`.
            if ready {
                self.remesh_async(n);
            }
        }
        self.pending_fresh.set();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same-sky holes stay one run. A foreign sky splits. Open chunks that share
    /// `(cx, cz)` stay separate jobs. A slab span includes the loaded layers.
    #[test]
    fn gather_column_runs_splits_foreign_sky_and_keeps_open_separate() {
        let sky = |c: Coord| -> Sky {
            if c.x == 0 && c.z == 0 && (c.y == 1 || c.y == 3) {
                Sky::Open
            } else if c.x >= 4 {
                Sky::Axis(Face::PosX)
            } else {
                Sky::Axis(Face::PosY)
            }
        };
        let missing = [
            Coord::new(1, 0, 0),
            Coord::new(1, 2, 0),
            Coord::new(0, 0, 0),
            Coord::new(0, 2, 0),
            Coord::new(0, 1, 0),
            Coord::new(0, 3, 0),
            Coord::new(4, 0, 0),
            Coord::new(6, 0, 0),
        ];
        let runs = gather_column_runs(missing, sky, |_| false, |_| false, false, false);
        let hole = ColumnKey { face: Face::PosY, a: 1, b: 0 };
        assert!(
            runs.contains(&GenRun::Column { key: hole, lo: 0, hi: 2 }),
            "a loaded PosY hole stays one run: {runs:?}"
        );
        let split = ColumnKey { face: Face::PosY, a: 0, b: 0 };
        assert!(runs.contains(&GenRun::Column { key: split, lo: 0, hi: 0 }), "{runs:?}");
        assert!(runs.contains(&GenRun::Column { key: split, lo: 2, hi: 2 }), "{runs:?}");
        assert!(
            runs.contains(&GenRun::Open { coord: Coord::new(0, 1, 0) }),
            "the Open layer is its own run: {runs:?}"
        );
        assert!(
            !runs.contains(&GenRun::Column { key: split, lo: 0, hi: 2 }),
            "Open in the gap splits the PosY run"
        );
        assert!(
            runs.contains(&GenRun::Open { coord: Coord::new(0, 3, 0) }),
            "Open chunks that share xz are not merged"
        );
        let posx = ColumnKey { face: Face::PosX, a: -1, b: 0 };
        assert!(
            runs.contains(&GenRun::Column { key: posx, lo: 4, hi: 6 }),
            "a same-sky gap along +X merges: {runs:?}"
        );

        let slab = [Coord::new(2, 0, 3), Coord::new(2, 1, 3), Coord::new(2, 2, 3)];
        let slab_runs = gather_column_runs(slab, sky, |c| c.y == 1, |_| false, true, false);
        assert_eq!(
            slab_runs,
            vec![GenRun::Column { key: ColumnKey { face: Face::PosY, a: 2, b: 3 }, lo: 0, hi: 2 }],
            "a slab span includes the loaded middle"
        );

        let gapped = [Coord::new(2, 0, 4), Coord::new(2, 2, 4)];
        let gapped_runs = gather_column_runs(gapped, sky, |_| false, |_| false, true, false);
        let gk = ColumnKey { face: Face::PosY, a: 2, b: 4 };
        assert_eq!(
            gapped_runs,
            vec![
                GenRun::Column { key: gk, lo: 0, hi: 0 },
                GenRun::Column { key: gk, lo: 2, hi: 2 },
            ]
        );
    }

    #[test]
    fn open_fail_key_does_not_collide_with_a_pos_y_column() {
        let coord = Coord::new(3, 1, 4);
        let open = FailKey::Open { coord };
        let column = FailKey::Column {
            key: ColumnKey { face: Face::PosY, a: coord.x, b: coord.z },
        };
        assert_ne!(open, column);
        let mut set = FastSet::default();
        set.insert(column);
        assert!(!set.contains(&open));
        let runs = gather_column_runs(
            [coord],
            |_| Sky::Open,
            |_| false,
            |k| k == column,
            false,
            true,
        );
        assert_eq!(runs, vec![GenRun::Open { coord }]);
    }
}
