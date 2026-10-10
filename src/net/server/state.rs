//! The shared [`State`] behind the one lock: the roster, the edit ledger, the interest grid and the terrain cache.
use super::*;

pub(super) struct PlayerHandle {
    /// Interned once at join; every roster/join/chat broadcast that carries
    /// it is a refcount bump, never a per-recipient allocation.
    pub(super) name: Arc<str>,
    pub(super) pos: DVec3,
    pub(super) yaw: f32,
    pub(super) pitch: f32,
    pub(super) frame: DQuat,
    pub(super) velocity: Vec3,
    pub(super) up: Face,
    pub(super) stance: Stance,
    /// The movement envelope's time anchor.
    pub(super) last_move: Instant,
    /// Distance banked at `last_move`, spent by moves and refilled at the envelope speed.
    pub(super) budget: f64,
    /// Burst capacity used to express banked credit as a fraction across speed changes.
    pub(super) burst: f64,
    /// Proved an operator secret with `/op`.
    pub(super) op: bool,
    /// Ids inside mutual interest range (`a.visible.contains(b) ==
    /// b.visible.contains(a)`). Maintained by [`on_move`]'s diff; drives
    /// PeerExited/re-entry pose events.
    pub(super) visible: HashSet<u32, Ids>,
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
    /// Peer ids whose `PeerJoined` this client has already been queued. A join
    /// both snapshots the roster and may race another joiner's broadcast.
    pub(super) announced: HashSet<u32, Ids>,
}

/// Hashes server-assigned player ids. No client picks them, so one multiply
/// spreads them well and costs a fraction of SipHash on the per-move paths.
#[derive(Default)]
pub(super) struct IdHasher(u64);

impl Hasher for IdHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u32(u32::from(b));
        }
    }

    fn write_u32(&mut self, v: u32) {
        self.0 = (self.0 ^ u64::from(v)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

pub(super) type Ids = BuildHasherDefault<IdHasher>;

/// Buffers [`commit_pose`] reuses so a move allocates nothing.
#[derive(Default)]
pub(super) struct Scratch {
    pub(super) near: HashSet<u32, Ids>,
    pub(super) gone: Vec<u32>,
    pub(super) fresh: Vec<u32>,
}

/// One edited cell: its block (so reads never parse the spec), the pooled spec
/// the wire and saves carry, and the revision racing edits compare against.
pub(super) struct Cell {
    pub(super) block: BlockId,
    pub(super) spec: Arc<str>,
    pub(super) rev: u32,
    /// An edit put back the generated block: kept for its revision, left out of the world file.
    pub(super) natural: bool,
}

pub(super) struct State {
    pub(super) edits: HashMap<(i32, i32, i32), Cell>,
    /// Distinct CANONICAL spec strings and how many live cells name them.
    /// Snapshot clones and broadcasts hold their own `Arc`s and do not count:
    /// an entry leaves when the cell count hits zero. Capped at [`MAX_SPEC_POOL`].
    pub(super) spec_pool: HashMap<Arc<str>, u32>,
    /// The same compiled palette clients build, so specs validate/canonicalize
    /// under EXACTLY the rules clients apply.
    pub(super) registry: BlockRegistry,
    pub(super) players: HashMap<u32, PlayerHandle, Ids>,
    /// Bucket key → ids standing in it, keyed by [`bucket_of`]. Buckets are
    /// exactly one [`INTEREST_RADIUS`] wide on each axis, so anyone in range of
    /// a mover lives in its 3×3×3 neighbourhood; [`on_move`] still applies the
    /// exact per-player distance check, so the grid only narrows candidates,
    /// never the audience. Invariant: exactly one entry per connected player,
    /// updated under the same lock hold as the position change it mirrors;
    /// empty buckets are removed eagerly so churn can never leak keys.
    pub(super) grid: HashMap<(i32, i32, i32), Vec<u32>>,
    pub(super) next_id: u32,
    /// The `[0,1)` day fraction current at `day_set`. The server advances it
    /// only when asked ([`State::day_now`]), so a late joiner receives the
    /// CURRENT phase rather than whatever `/time` last set.
    pub(super) day: f32,
    pub(super) day_set: Instant,
    /// Server-authoritative reaction scheduler. Clients never run one.
    pub(super) reactions: ReactionScheduler,
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
    pub(super) fn visible_from(&self, id: u32, pos: DVec3, visible: &mut HashSet<u32, Ids>) {
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
                        if other.ready && other.pos.distance_squared(pos) <= INTEREST_RADIUS_SQ {
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

    /// `None` at the [`MAX_SPEC_POOL`] cap. An existing spec still resolves at the cap,
    /// and its cell count goes up by one.
    pub(super) fn intern(&mut self, spec: &str) -> Option<Arc<str>> {
        if let Some(n) = self.spec_pool.get_mut(spec) {
            *n = n.saturating_add(1);
            return self.spec_pool.get_key_value(spec).map(|(k, _)| k.clone());
        }
        if self.spec_pool.len() >= MAX_SPEC_POOL {
            return None;
        }
        let shared: Arc<str> = Arc::from(spec);
        self.spec_pool.insert(shared.clone(), 1);
        Some(shared)
    }

    /// One live cell stopped naming `old`. The pool entry leaves at zero,
    /// whatever other `Arc` clones (a snapshot list, a test) still exist.
    pub(super) fn release(&mut self, old: Arc<str>) {
        let Some(n) = self.spec_pool.get_mut(old.as_ref()) else { return };
        if *n <= 1 {
            self.spec_pool.remove(old.as_ref());
        } else {
            *n -= 1;
        }
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
        let canonical = crate::save::block_spec(&self.state.registry, id);
        let spec = self.state.intern(&canonical)?;
        let rev = self.state.edits.get(&pos).map_or(0, |c| c.rev).saturating_add(1);
        if let Some(old) = self.state.edits.insert(pos, Cell { block: id, spec, rev, natural: false }) {
            self.state.release(old.spec);
        }
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

pub(super) fn outside_world(pos: DVec3) -> bool {
    pos.x.abs() > crate::math::WORLD_BORDER
        || pos.y.abs() > crate::math::WORLD_BORDER
        || pos.z.abs() > crate::math::WORLD_BORDER
}
