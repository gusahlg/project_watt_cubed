//! Shared fixtures: hand-built players and states, contexts, raw handshakes and frame drains.
use super::super::*;
pub(super) use crate::net::client::Connection;
use crate::net::client::Incoming;
pub(super) use crate::net::test_util::eventually;
pub(super) use std::collections::HashSet;
pub(super) use std::sync::mpsc::Receiver;

pub(super) fn test_generator() -> crate::world::terrain::Generator {
    crate::world::terrain::generator(&mut BlockRegistry::with_builtins(), 4242, Default::default())
}

/// A roster entry for direct state tests. `last_move` starts well in the
/// past so the first envelope window is at its cap (a fresh anchor allows
/// only ~30 world units); tests re-age it between deliberate big moves.
pub(super) fn test_player(pos: DVec3, tx: SyncSender<Arc<[u8]>>, kick: Arc<Notify>) -> PlayerHandle {
    let out = Outbox { tx: Some(tx), writer: Arc::default() };
    PlayerHandle {
        last_move: Instant::now() - Duration::from_secs(10),
        ready: true,
        ..PlayerHandle::new("p".into(), pos, DQuat::IDENTITY, Face::PosY, false, out, kick)
    }
}

/// A world with no charts, so an edit's reach is judged at the cell itself.
pub(super) fn chartless() -> &'static crate::world::terrain::Generator {
    use std::sync::OnceLock;
    static FLAT: OnceLock<crate::world::terrain::Generator> = OnceLock::new();
    FLAT.get_or_init(|| Arc::new(crate::world::generation::FlatTerrain::new(&mut BlockRegistry::with_builtins(), 1)))
}

/// Noclip is open, so movement-envelope tests do not build a collision world
/// on every step. The generator is built once for the process.
pub(super) fn lax_ctx() -> &'static Ctx {
    use std::sync::OnceLock;
    static CTX: OnceLock<Ctx> = OnceLock::new();
    CTX.get_or_init(|| test_ctx(true))
}

/// One [`on_move`] with a buffer of its own for the frames it queues.
#[allow(clippy::too_many_arguments)] // the fields of a Move, as the message carries them
pub(super) fn move_once(
    shared: &Arc<Mutex<State>>,
    ctx: &Ctx,
    id: u32,
    pos: DVec3,
    yaw: f32,
    pitch: f32,
    frame: DQuat,
    velocity: Vec3,
    up: Face,
    stance: Stance,
) {
    on_move(shared, ctx, id, Pose { pos, yaw, pitch, frame, velocity, up, stance }, &mut Vec::new());
}

/// A move that leaves the body frame, velocity, and up axis at their defaults.
pub(super) fn walk(shared: &Arc<Mutex<State>>, id: u32, pos: DVec3, yaw: f32, pitch: f32, stance: Stance) {
    move_once(shared, lax_ctx(), id, pos, yaw, pitch, DQuat::IDENTITY, Vec3::ZERO, Face::PosY, stance);
}

/// A throwaway kick handle for state-only players (never notified).
pub(super) fn test_kick() -> Arc<Notify> {
    Arc::new(Notify::new())
}

/// A client runtime + endpoint for the raw-handshake tests, wired with the same
/// accept-any-cert config real clients use.
pub(super) fn client_endpoint() -> (Runtime, Endpoint) {
    let rt = Runtime::new().unwrap();
    quic::install_crypto();
    let mut ep = {
        let _g = rt.enter();
        Endpoint::client((Ipv4Addr::UNSPECIFIED, 0).into()).unwrap()
    };
    ep.set_default_client_config(quic::client_config());
    (rt, ep)
}

/// Dial, open the reliable stream, send one crafted payload, and return the
/// server's first reply — the raw handshake path `Connection::connect` hides.
pub(super) fn raw_reply(addr: SocketAddr, payload: &[u8]) -> ServerMessage {
    // The server binds 0.0.0.0; quinn refuses to dial the unspecified address, so
    // reach it over loopback (`handle.addr()` carries only the resolved port).
    let target = SocketAddr::from((Ipv4Addr::LOCALHOST, addr.port()));
    let (rt, ep) = client_endpoint();
    rt.block_on(async {
        let conn = ep.connect(target, "watt").unwrap().await.unwrap();
        let (mut s, mut r) = conn.open_bi().await.unwrap();
        protocol::write_frame_async(&mut s, payload).await.unwrap();
        let mut buf = Vec::new();
        protocol::read_frame_async(&mut r, &mut buf).await.unwrap();
        ServerMessage::decode(&buf).unwrap()
    })
}

/// The reason the server refuses `payload` with.
pub(super) fn reject_reason(addr: SocketAddr, payload: &[u8]) -> String {
    match raw_reply(addr, payload) {
        ServerMessage::Reject { reason } => reason.to_string(),
        other => panic!("expected Reject, got {other:?}"),
    }
}

/// A `Hello` from `name`, encoded for [`raw_reply`].
pub(super) fn hello(name: &str, password: &str, protocol: u32, content: crate::net::ContentId) -> Vec<u8> {
    ClientMessage::Hello {
        protocol,
        worldgen: content.worldgen,
        gravity: content.gravity,
        law: content.law,
        palette: content.palette,
        name: name.into(),
        password: password.into(),
        mods: vec![],
    }
    .encode()
}

pub(super) fn server_content() -> crate::net::ContentId {
    crate::net::content_id(&BlockRegistry::with_builtins())
}

pub(super) fn rock_spec() -> String {
    let mut r = BlockRegistry::with_builtins();
    let id = r
        .intern(&material::Configuration::single(material::Element::new([40, 80, 120, 160])))
        .unwrap();
    r.spec(id)
}

pub(super) fn test_state(players: HashMap<u32, PlayerHandle>) -> State {
    // The palette first, exactly as `spawn` builds it, so generator ids mean the same here.
    let mut registry = BlockRegistry::with_builtins();
    crate::world::terrain::Materials::intern(&mut registry);
    let mut state = State::new(registry, 0.3, crate::player::MAX_SPEED);
    state.players = players.into_iter().collect();
    state.next_id = 2;
    state
}

/// A [`test_state`] holding a ready [`test_player`] at each `(id, position)`, each with its own
/// outbox, and the receiving ends in the same order.
pub(super) fn roster<const N: usize>(at: [(u32, DVec3); N]) -> (State, [Receiver<Arc<[u8]>>; N]) {
    let mut players = HashMap::new();
    let inboxes = at.map(|(id, pos)| {
        let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        players.insert(id, test_player(pos, out, test_kick()));
        rx
    });
    (test_state(players), inboxes)
}

/// [`roster`] behind the lock the handlers take.
pub(super) fn lobby<const N: usize>(at: [(u32, DVec3); N]) -> (Arc<Mutex<State>>, [Receiver<Arc<[u8]>>; N]) {
    let (state, inboxes) = roster(at);
    (Arc::new(Mutex::new(state)), inboxes)
}

/// Anchor the player's envelope `ago` in the past: the next move's budget refills for that long.
pub(super) fn age_state(state: &mut State, id: u32, ago: Duration) {
    if let Some(h) = state.players.get_mut(&id) {
        h.last_move = Instant::now() - ago;
    }
}

/// Push the player's envelope anchor into the past, buying the next move
/// the full (capped) displacement window.
pub(super) fn age_move(shared: &Arc<Mutex<State>>, id: u32) {
    age_state(&mut shared.lock_recover(), id, Duration::from_secs(10));
}

/// Anchor the envelope one realistic move gap ago: the next move's budget refills for that long.
pub(super) fn stamp_gap(shared: &Arc<Mutex<State>>, id: u32) {
    age_state(&mut shared.lock_recover(), id, Duration::from_millis(100));
}

/// The 4242 world, with noclip open and teleport allowed or not.
pub(super) fn test_ctx(allow_teleport: bool) -> Ctx {
    let teleport = if allow_teleport { Policy::All } else { Policy::Off };
    Ctx::new(Config { seed: 4242, teleport, ..Config::default() }, test_generator(), server_content(), None)
}

/// The reference destructive pair as specs, with the target written into the ledger at `cell`.
pub(super) fn place_pair(shared: &Arc<Mutex<State>>, cell: Pos) -> (String, String) {
    let mut state = shared.lock_recover();
    let (a, e) = crate::sim::reactions::destructive_pair(&mut state.registry);
    let (sa, se) = (state.registry.spec(a), state.registry.spec(e));
    let spec = state.intern(a).unwrap();
    state.edits.insert(cell, Cell { block: a, spec, rev: 1, natural: false });
    (sa, se)
}

pub(super) struct XorShift(u64);

impl XorShift {
    pub(super) fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    pub(super) fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub(super) fn f64(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (self.next() as f64 / u64::MAX as f64) * (hi - lo)
    }
    pub(super) fn u32(&mut self, max_excl: u32) -> u32 {
        (self.next() as u32) % max_excl.max(1)
    }
}

pub(super) fn numbered_spec(n: u8) -> String {
    let cfg = material::Configuration::single(material::Element::new([200, n, 17, 3]));
    let bytes = cfg.encode();
    let mut s = String::from("c:");
    for b in bytes.as_bytes() {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

pub(super) fn flat(config: Config) -> ServerHandle {
    spawn(0, Config { seed: 1, worldgen: WorldgenKind::Flat, ..config }).unwrap()
}

/// Every frame waiting in an outbox, decoded. A frame that does not decode fails the test.
pub(super) fn drain(rx: &Receiver<Arc<[u8]>>) -> Vec<ServerMessage> {
    rx.try_iter().map(|frame| ServerMessage::decode(&frame).expect("the server sends frames that decode")).collect()
}

pub(super) fn flat_shared(
    players: HashMap<u32, PlayerHandle>,
    noclip: NoclipPolicy,
    ops: &[&str],
) -> (Arc<Mutex<State>>, Ctx) {
    let mut registry = BlockRegistry::with_builtins();
    let generator: crate::world::terrain::Generator =
        Arc::new(crate::world::generation::FlatTerrain::new(&mut registry, 1));
    let state = {
        let mut state = State::new(registry, 0.3, crate::player::MAX_SPEED);
        state.players = players.into_iter().collect();
        state.next_id = 2;
        // Admission makes a player listed by name an operator.
        for h in state.players.values_mut() {
            h.op |= ops.iter().any(|op| op.eq_ignore_ascii_case(&h.name));
        }
        state
    };
    let ops = ops.iter().map(|name| name.to_string()).collect();
    let config = Config { seed: 1, worldgen: WorldgenKind::Flat, noclip, ops, ..Config::default() };
    (Arc::new(Mutex::new(state)), Ctx::new(config, generator, server_content(), None))
}

pub(super) fn pose(pos: DVec3) -> (HashMap<u32, PlayerHandle>, Receiver<Arc<[u8]>>) {
    let (out, rx) = sync_channel::<Arc<[u8]>>(8);
    let mut players = HashMap::new();
    players.insert(1u32, test_player(pos, out, test_kick()));
    (players, rx)
}

/// Chat until a line containing `want` arrives, or three seconds pass.
pub(super) fn chat_until(conn: &mut Connection, want: &str) -> Vec<String> {
    let mut texts = Vec::new();
    eventually(Duration::from_secs(3), || {
        for event in conn.poll() {
            if let Incoming::Chat { text, .. } = event {
                texts.push(text.to_string());
            }
        }
        texts.iter().any(|t| t.contains(want))
    });
    texts
}

/// Poll `conn` until `pick` takes an event, or `within` passes.
pub(super) fn await_event<T>(conn: &mut Connection, within: Duration, mut pick: impl FnMut(Incoming) -> Option<T>) -> Option<T> {
    let mut found = None;
    eventually(within, || {
        for event in conn.poll() {
            if found.is_none() {
                found = pick(event);
            }
        }
        found.is_some()
    });
    found
}

/// A generator that counts its reads (cells and column heights), and the reads made while the
/// state lock was held.
pub(super) struct Watched {
    inner: crate::world::terrain::Generator,
    state: std::sync::Weak<Mutex<State>>,
    pub(super) reads: AtomicUsize,
    pub(super) locked: AtomicUsize,
}

impl Watched {
    pub(super) fn new(inner: crate::world::terrain::Generator, state: &Arc<Mutex<State>>) -> Arc<Self> {
        Arc::new(Self { inner, state: Arc::downgrade(state), reads: AtomicUsize::new(0), locked: AtomicUsize::new(0) })
    }

    fn read(&self) {
        self.reads.fetch_add(1, Ordering::Relaxed);
        if self.state.upgrade().is_some_and(|s| s.try_lock().is_err()) {
            self.locked.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl TerrainGenerator for Watched {
    fn chart_spawn(&self) -> Option<DVec3> {
        self.inner.chart_spawn()
    }

    fn height(&self, x: i32, z: i32) -> i32 {
        self.read();
        self.inner.height(x, z)
    }

    fn surface_at(&self, x: i32, z: i32) -> BlockId {
        self.inner.surface_at(x, z)
    }

    fn deep(&self) -> BlockId {
        self.inner.deep()
    }

    fn voxel_at(&self, x: i32, y: i32, z: i32) -> BlockId {
        self.read();
        self.inner.voxel_at(x, y, z)
    }
}
