//! The shared [`State`] behind the one lock: the roster, the edit ledger, the interest grid and the terrain cache.
use super::*;

pub(super) struct PlayerHandle {
    /// Interned once at join; every roster/join/chat broadcast that carries
    /// it is a refcount bump, never a per-recipient allocation.
    pub(super) name: Arc<str>,
    /// The last accepted pose: edit reach, interest and the envelope read it.
    pub(super) pose: Pose,
    /// The movement envelope's time anchor.
    pub(super) last_move: Instant,
    /// Distance banked at `last_move`, spent by moves and refilled at the envelope speed.
    pub(super) budget: f64,
    /// Burst capacity used to express banked credit as a fraction across speed changes.
    pub(super) burst: f64,
    /// An operator: listed by name when admitted, or proved a secret with `/op` since.
    pub(super) op: bool,
    /// Ids inside mutual interest range (`a.visible.contains(b) ==
    /// b.visible.contains(a)`). Maintained by [`on_move`]'s diff; drives
    /// PeerExited/re-entry pose events.
    pub(super) visible: FastSet<u32>,
    /// The pose in wire form, refreshed when it changes.
    pub(super) body: PoseBody,
    /// The pose tick ([`State::tick`]) the pose last changed before.
    pub(super) moved: u64,
    pub(super) out: Outbox,
    /// Wakes a misbehaving client's reader out of its blocking read so cleanup
    /// runs. A `Notify` rather than `quinn::Connection` so it's cheap to
    /// fabricate in state-only tests.
    pub(super) kick: Arc<Notify>,
    /// False until Welcome/Snapshot is fully queued. Broadcasters must not
    /// push into a not-yet-ready queue — a racing frame could beat Welcome
    /// onto the wire or interleave between snapshot batches — so they buffer
    /// into `backlog` instead, drained in order once the bootstrap is done.
    pub(super) ready: bool,
    /// Frames held until bootstrap finishes, with the instant they were queued, oldest first.
    pub(super) backlog: VecDeque<Queued>,
    pub(super) backlog_bytes: usize,
    /// Set by [`kick_slow`] and by a bootstrap send that misses its deadline.
    /// The reader is not in [`client_loop`] yet during bootstrap, so the kick
    /// [`Notify`] alone would not be watched.
    pub(super) kicked: Arc<AtomicBool>,
    /// Physical cells the body occupied on the last accepted move. A later move
    /// only queries cells that are not already here.
    pub(super) occupied: Vec<(i32, i32, i32)>,
    /// Declared [`ClientMessage::Cruise`]. The envelope uses `cruise_speed` as its cap.
    pub(super) cruising: bool,
    pub(super) cruise_speed: f64,
    /// Novel configurations this client has interned. Capped at [`NOVEL_SPEC_QUOTA`].
    pub(super) novel: u32,
    /// Peers named in this client's join roster whose own `PeerJoined` broadcast has not
    /// come yet: that broadcast skips this client once and takes the id out. A peer that
    /// leaves first is taken out by [`State::remove_player`], so the set never outgrows
    /// the roster.
    pub(super) announced: FastSet<u32>,
}

impl PlayerHandle {
    /// A player admitted at `spawn`, standing still in the pose `frame` and `up`, and not
    /// ready until its bootstrap is queued.
    pub(super) fn new(name: Arc<str>, spawn: DVec3, frame: DQuat, up: Face, op: bool, out: Outbox, kick: Arc<Notify>) -> Self {
        let pose = Pose::standing(spawn, frame, up);
        Self {
            name,
            pose,
            last_move: Instant::now(),
            budget: MOVE_FLOOR,
            burst: MOVE_FLOOR,
            op,
            visible: FastSet::default(),
            body: pose.body(),
            moved: 0,
            out,
            kick,
            ready: false,
            backlog: VecDeque::new(),
            backlog_bytes: 0,
            kicked: Arc::new(AtomicBool::new(false)),
            occupied: Vec::new(),
            cruising: false,
            cruise_speed: 0.0,
            novel: 0,
            announced: FastSet::default(),
        }
    }
}

/// Buffers [`commit_pose`] reuses so a move allocates nothing.
#[derive(Default)]
pub(super) struct Scratch {
    pub(super) near: FastSet<u32>,
    pub(super) gone: Vec<u32>,
    pub(super) fresh: Vec<u32>,
}

/// One edited cell: its block (so reads never parse the spec), the registry's own spec text
/// for it (the wire and saves carry it), and the revision racing edits compare against.
pub(super) struct Cell {
    pub(super) block: BlockId,
    pub(super) spec: Arc<str>,
    pub(super) rev: u32,
    /// An edit put back the generated block: kept for its revision, left out of the world file.
    pub(super) natural: bool,
}

/// How many live cells name each block, by id, and how many blocks are named at all, which
/// [`MAX_SPEC_POOL`] caps. The spec text is the registry's (`spec_ref`), so the pool holds no
/// strings. Snapshot clones and broadcasts hold their own `Arc`s and are not counted: a block
/// leaves when its cell count reaches zero.
#[derive(Default)]
pub(super) struct SpecPool {
    cells: Vec<u32>,
    named: usize,
}

impl SpecPool {
    /// Count one more cell naming `id`. False when no live cell names `id` yet and the pool is
    /// full; a block already named still resolves at the cap.
    pub(super) fn take(&mut self, id: BlockId) -> bool {
        let at = usize::from(id.0);
        if at >= self.cells.len() {
            self.cells.resize(at + 1, 0);
        }
        let n = &mut self.cells[at];
        if *n == 0 {
            if self.named >= MAX_SPEC_POOL {
                return false;
            }
            self.named += 1;
        }
        *n = n.saturating_add(1);
        true
    }

    /// One live cell stopped naming `id`.
    pub(super) fn release(&mut self, id: BlockId) {
        let Some(n) = self.cells.get_mut(usize::from(id.0)) else { return };
        match *n {
            0 => {}
            1 => {
                *n = 0;
                self.named -= 1;
            }
            _ => *n -= 1,
        }
    }

    /// Distinct blocks named by live cells.
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.named
    }

    /// Live cells naming `id`.
    #[cfg(test)]
    pub(super) fn cells(&self, id: BlockId) -> u32 {
        self.cells.get(usize::from(id.0)).copied().unwrap_or(0)
    }
}

pub(super) struct State {
    /// Keyed by cells clients choose, so it keeps SipHash.
    pub(super) edits: HashMap<(i32, i32, i32), Cell>,
    pub(super) spec_pool: SpecPool,
    /// The same compiled palette clients build, so specs validate/canonicalize
    /// under EXACTLY the rules clients apply.
    pub(super) registry: BlockRegistry,
    /// Keyed by ids the server assigns, so a fast hash is safe.
    pub(super) players: FastMap<u32, PlayerHandle>,
    /// Bucket key → ids standing in it, keyed by [`bucket_of`]. Buckets are
    /// exactly one [`INTEREST_RADIUS`] wide on each axis, so anyone in range of
    /// a mover lives in its 3×3×3 neighbourhood; [`on_move`] still applies the
    /// exact per-player distance check, so the grid only narrows candidates,
    /// never the audience. Invariant: exactly one entry per connected player,
    /// updated under the same lock hold as the position change it mirrors;
    /// empty buckets are removed eagerly so churn can never leak keys.
    pub(super) grid: FastMap<(i32, i32, i32), Vec<u32>>,
    pub(super) next_id: u32,
    /// Ids and names held by joins between their checks and their roster insert ([`reserve`]).
    pub(super) joining: Vec<(u32, Arc<str>)>,
    /// The `[0,1)` day fraction current at `day_set`. The server advances it
    /// only when asked ([`State::day_now`]), so a late joiner receives the
    /// CURRENT phase rather than whatever `/time` last set.
    pub(super) day: f32,
    pub(super) day_set: Instant,
    /// Server-authoritative reaction scheduler. Clients never run one.
    pub(super) reactions: ReactionScheduler,
    /// The reaction tick's commits, reused from tick to tick.
    pub(super) mutations: Vec<Mutation>,
    /// Each committed cell's last commit in a tick, reused from tick to tick. Keyed by cells
    /// clients can choose, so it keeps SipHash.
    pub(super) latest: HashMap<Pos, usize>,
    /// The next pose tick ([`broadcast_poses`]).
    pub(super) tick: u64,
    pub(super) poses: PosesWriter,
    pub(super) scratch: Scratch,
    pub(super) terrain: TerrainCache,
    /// Envelope cap copied from [`Config::max_speed`] at spawn.
    pub(super) max_speed: f64,
    /// Test hook: the next reaction tick panics once, then clears the flag.
    #[cfg(test)]
    pub(super) panic_tick: bool,
}

impl State {
    /// An empty world on `registry`, with no players and the clock at `day`.
    pub(super) fn new(registry: BlockRegistry, day: f32, max_speed: f64) -> Self {
        Self {
            edits: HashMap::new(),
            spec_pool: SpecPool::default(),
            registry,
            players: FastMap::default(),
            grid: FastMap::default(),
            next_id: 1,
            joining: Vec::new(),
            day,
            day_set: Instant::now(),
            reactions: ReactionScheduler::new(),
            mutations: Vec::new(),
            latest: HashMap::new(),
            tick: 1,
            poses: PosesWriter::new(),
            scratch: Scratch::default(),
            terrain: TerrainCache::new(),
            max_speed,
            #[cfg(test)]
            panic_tick: false,
        }
    }

    /// Put an admitted player on the roster and the grid, and return the roster it joins
    /// (each peer's `PeerJoined` goes out in the bootstrap) and the overlay to send. One lock
    /// hold, so the roster, the overlay and the grid agree.
    pub(super) fn admit(&mut self, id: u32, mut handle: PlayerHandle) -> (Vec<(u32, Arc<str>)>, Overlay) {
        // Roster only — poses flow through the visibility machinery once the
        // joiner reports their first move, so a far peer isn't a frozen ghost.
        let existing: Vec<(u32, Arc<str>)> = self.players.iter().map(|(&pid, h)| (pid, h.name.clone())).collect();
        handle.announced.clear();
        handle.announced.extend(existing.iter().map(|&(pid, _)| pid));
        // Coordinates, revisions and block ids only: specs are named once per
        // block, and the frames are sorted and encoded after the lock.
        let overlay = Overlay::of(self);
        let spawn = handle.pose.pos;
        self.players.insert(id, handle);
        // Same lock hold as the roster insert, so the grid never lags the roster.
        self.grid_insert(id, spawn);
        (existing, overlay)
    }

    /// Take a player off the roster and the grid, and out of every peer's interest and
    /// announcement sets, so nothing names a player who has gone.
    pub(super) fn remove_player(&mut self, id: u32) -> Option<PlayerHandle> {
        let h = self.players.remove(&id)?;
        // The last committed position names the bucket the grid holds it under.
        self.grid_remove(id, h.pose.pos);
        for other in self.players.values_mut() {
            other.visible.remove(&id);
            other.announced.remove(&id);
        }
        Some(h)
    }

    /// Must run under the same lock hold as the roster/position change it
    /// mirrors, or the grid drifts.
    pub(super) fn grid_insert(&mut self, id: u32, pos: DVec3) {
        self.grid.entry(bucket_of(pos)).or_default().push(id);
    }

    /// Drops the bucket when it empties so churn can never accumulate dead keys.
    pub(super) fn grid_remove(&mut self, id: u32, pos: DVec3) {
        let key = bucket_of(pos);
        if let Some(bucket) = self.grid.get_mut(&key) {
            bucket.retain(|&p| p != id);
            if bucket.is_empty() {
                self.grid.remove(&key);
            }
        }
    }

    /// The grid narrows candidates; exact distance and readiness decide visibility.
    pub(super) fn visible_from(&self, id: u32, pos: DVec3, visible: &mut FastSet<u32>) {
        let at = bucket_of(pos);
        visible.clear();
        for dx in -1..=1i32 {
            for dy in -1..=1i32 {
                for dz in -1..=1i32 {
                    let key = (at.0.wrapping_add(dx), at.1.wrapping_add(dy), at.2.wrapping_add(dz));
                    let Some(bucket) = self.grid.get(&key) else { continue };
                    for &pid in bucket {
                        if pid == id {
                            continue;
                        }
                        let Some(other) = self.players.get(&pid) else { continue };
                        if other.ready && other.pose.pos.distance_squared(pos) <= INTEREST_RADIUS_SQ {
                            visible.insert(pid);
                        }
                    }
                }
            }
        }
    }

    pub(super) fn day_now(&self, day_secs: f32) -> f32 {
        let elapsed = self.day_set.elapsed().as_secs_f32();
        (self.day + elapsed / clamp_day_secs(day_secs)).rem_euclid(1.0)
    }

    /// The registry's spec text for `id`, counted as one more live cell naming it. `None` at the
    /// [`MAX_SPEC_POOL`] cap for a block no live cell names yet.
    pub(super) fn intern(&mut self, id: BlockId) -> Option<Arc<str>> {
        self.spec_pool.take(id).then(|| Arc::clone(self.registry.spec_ref(id)))
    }

    /// The revision of the cell at `at`: 0 for a cell never edited.
    pub(super) fn rev(&self, at: Pos) -> u32 {
        self.edits.get(&at).map_or(0, |c| c.rev)
    }

    /// Commit `block` at `at` at the next revision, and return that revision and the spec.
    /// `None`, with nothing written, when the spec pool cannot name `block`.
    pub(super) fn write(&mut self, at: Pos, block: BlockId, natural: bool) -> Option<(u32, Arc<str>)> {
        let spec = self.intern(block)?;
        let rev = self.rev(at).saturating_add(1);
        if let Some(old) = self.edits.insert(at, Cell { block, spec: Arc::clone(&spec), rev, natural }) {
            self.spec_pool.release(old.block);
        }
        Some((rev, spec))
    }
}

/// Ledger + generator as a [`CellStore`]: a cell not in the ledger reads from
/// the generator, so the infinite world is defined without loading chunks.
pub(super) struct ServerCells<'a> {
    pub(super) state: &'a mut State,
    pub(super) generator: &'a crate::world::terrain::Generator,
}

pub(super) fn server_block(
    state: &State,
    generator: &crate::world::terrain::Generator,
    pos: Pos,
) -> BlockId {
    if let Some(cell) = state.edits.get(&pos) {
        cell.block
    } else {
        state.terrain.get(pos, || generator.voxel_at(pos.0, pos.1, pos.2))
    }
}

pub(super) const TERRAIN_SLOTS: usize = 1 << 14;
pub(super) const TERRAIN_WAYS: usize = 4;

/// Generated cells read lately (a generator read costs microseconds), four ways
/// per set. Terrain never changes under the overlay, so a slot stays right until
/// another cell takes it. Only touched under the state lock.
pub(super) struct TerrainCache(RefCell<Box<[Option<(Pos, BlockId)>]>>);

impl TerrainCache {
    pub(super) fn new() -> Self {
        Self(RefCell::new(vec![None; TERRAIN_SLOTS].into_boxed_slice()))
    }

    pub(super) fn get(&self, pos: Pos, generate: impl FnOnce() -> BlockId) -> BlockId {
        if let Some(id) = self.peek(pos) {
            return id;
        }
        let id = generate();
        self.store(pos, id);
        id
    }

    pub(super) fn peek(&self, pos: Pos) -> Option<BlockId> {
        let (set, _) = Self::slot(pos);
        let cache = self.0.borrow();
        cache[set..set + TERRAIN_WAYS].iter().find_map(|w| w.filter(|(at, _)| *at == pos).map(|(_, id)| id))
    }

    pub(super) fn store(&self, pos: Pos, id: BlockId) {
        let (set, way) = Self::slot(pos);
        self.0.borrow_mut()[set + way] = Some((pos, id));
    }

    /// The set's first slot, and the way a new entry takes.
    pub(super) fn slot(pos: Pos) -> (usize, usize) {
        let h = u64::from(pos.0 as u32).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ u64::from(pos.1 as u32).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
            ^ u64::from(pos.2 as u32).wrapping_mul(0x1656_67B1_9E37_79F9);
        let sets = (TERRAIN_SLOTS / TERRAIN_WAYS).trailing_zeros();
        ((h >> (64 - sets)) as usize * TERRAIN_WAYS, h as usize % TERRAIN_WAYS)
    }
}

impl CellStore for ServerCells<'_> {
    fn block_at(&self, pos: Pos) -> Option<BlockId> {
        Some(server_block(self.state, self.generator, pos))
    }

    /// `None` when the spec pool is full: the cell keeps its material and the
    /// scheduler records no mutation for it (a refused write is not a change).
    fn set_block(&mut self, pos: Pos, id: BlockId) -> Option<BlockId> {
        let prev = server_block(self.state, self.generator, pos);
        if prev == id {
            return Some(prev);
        }
        self.state.write(pos, id, false)?;
        Some(prev)
    }

    fn registry(&self) -> &BlockRegistry {
        &self.state.registry
    }

    fn registry_mut(&mut self) -> &mut BlockRegistry {
        &mut self.state.registry
    }
}

/// The interest-grid bucket containing `pos`. Goes through [`block_coord`]'s
/// clamped floor (not truncation) so negative coordinates bucket consistently
/// and a hostile-but-finite huge coordinate can't overflow the i32 key —
/// insert and remove share this one mapping, so the grid stays consistent.
pub(super) fn bucket_of(pos: DVec3) -> (i32, i32, i32) {
    (
        block_coord(pos.x / INTEREST_RADIUS),
        block_coord(pos.y / INTEREST_RADIUS),
        block_coord(pos.z / INTEREST_RADIUS),
    )
}

/// Where a storage cell sits in the world: a round world's cell is judged where its chart embeds
/// it, any other at its own centre.
pub(super) fn cell_centre(generator: &crate::world::terrain::Generator, (x, y, z): Pos) -> DVec3 {
    crate::space::atlas::embed_cell(generator.atlases(), (x, y, z))
        .unwrap_or(DVec3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5))
}

pub(super) fn outside_world(pos: DVec3) -> bool {
    pos.x.abs() > crate::math::WORLD_BORDER
        || pos.y.abs() > crate::math::WORLD_BORDER
        || pos.z.abs() > crate::math::WORLD_BORDER
}
