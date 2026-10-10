//! Shared fixtures: hand-built players and states, contexts, raw handshakes and frame drains.
use super::super::*;

pub(super) fn test_generator() -> crate::world::terrain::Generator {
    crate::world::terrain::generator(&mut BlockRegistry::with_builtins(), 4242, Default::default())
}

/// A roster entry for direct state tests. `last_move` starts well in the
/// past so the first envelope window is at its cap (a fresh anchor allows
/// only ~30 world units); tests re-age it between deliberate big moves.
pub(super) fn test_player(pos: DVec3, tx: SyncSender<Arc<[u8]>>, kick: Arc<Notify>) -> PlayerHandle {
    PlayerHandle {
        name: "p".into(),
        pos,
        yaw: 0.0,
        pitch: 0.0,
        frame: DQuat::IDENTITY,
        velocity: Vec3::ZERO,
        up: Face::PosY,
        stance: Stance::Standing,
        last_move: Instant::now() - Duration::from_secs(10),
        budget: MOVE_FLOOR,
            burst: MOVE_FLOOR,
        op: false,
        visible: HashSet::default(),
        body: PoseBody::new(0.0, 0.0, DQuat::IDENTITY, Vec3::ZERO, Face::PosY, Stance::Standing),
        moved: 0,
        out: Outbox { tx: Some(tx), writer: Arc::default() },
        kick,
        ready: true,
        backlog: VecDeque::new(),
        backlog_bytes: 0,
        kicked: Arc::new(AtomicBool::new(false)),
        occupied: Vec::new(),
        cruising: false,
        cruise_speed: 0.0,
        novel: 0,
        announced: HashSet::default(),
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

/// A throwaway kick handle for state-only players (never notified).
/// A move that leaves the body frame, velocity, and up axis at their defaults.
pub(super) fn walk(shared: &Arc<Mutex<State>>, id: u32, pos: DVec3, yaw: f32, pitch: f32, stance: Stance) {
    on_move(shared, lax_ctx(), id, pos, yaw, pitch, DQuat::IDENTITY, Vec3::ZERO, Face::PosY, stance);
}

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

/// Dial, open the reliable stream, send one crafted message, and return the
/// server's first reply — the raw handshake path `Connection::connect` hides.
pub(super) fn raw_reply(addr: SocketAddr, hello: &ClientMessage) -> ServerMessage {
    // The server binds 0.0.0.0; quinn refuses to dial the unspecified address, so
    // reach it over loopback (`handle.addr()` carries only the resolved port).
    let target = SocketAddr::from((Ipv4Addr::LOCALHOST, addr.port()));
    let (rt, ep) = client_endpoint();
    rt.block_on(async {
        let conn = ep.connect(target, "watt").unwrap().await.unwrap();
        let (mut s, mut r) = conn.open_bi().await.unwrap();
        protocol::write_frame_async(&mut s, &hello.encode()).await.unwrap();
        let mut buf = Vec::new();
        protocol::read_frame_async(&mut r, &mut buf).await.unwrap();
        ServerMessage::decode(&buf).unwrap()
    })
}

pub(super) fn hello(name: &str, password: &str, protocol: u32, content: crate::net::ContentId) -> ClientMessage {
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
}

pub(super) fn server_content() -> crate::net::ContentId {
    crate::net::content_id(&BlockRegistry::with_builtins())
}

pub(super) fn reject_reason(addr: SocketAddr, msg: &ClientMessage) -> String {
    match raw_reply(addr, msg) {
        ServerMessage::Reject { reason } => reason.to_string(),
        other => panic!("expected Reject, got {other:?}"),
    }
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
    State {
        edits: HashMap::new(),
        spec_pool: HashMap::new(),
        registry,
        players: players.into_iter().collect(),
        grid: HashMap::new(),
        next_id: 2,
        day: 0.3,
        day_set: Instant::now(),
        reactions: ReactionScheduler::new(),
        tick: 1,
        poses: PosesWriter::new(),
        scratch: Scratch::default(),
        terrain: TerrainCache::new(),
        max_speed: crate::player::MAX_SPEED,
        panic_tick: false,
    }
}

/// Push the player's envelope anchor into the past, buying the next move
/// the full (capped) displacement window.
pub(super) fn age_move(shared: &Arc<Mutex<State>>, id: u32) {
    if let Some(h) = shared.lock_recover().players.get_mut(&id) {
        h.last_move = Instant::now() - Duration::from_secs(10);
    }
}

pub(super) fn test_ctx(allow_teleport: bool) -> Ctx {
    Ctx {
        password: String::new(),
        seed: 4242,
        content: crate::net::content_id(&BlockRegistry::with_builtins()),
        day_secs: 600.0,
        teleport: if allow_teleport { TeleportPolicy::All } else { TeleportPolicy::Off },
        noclip: NoclipPolicy::All,
        worldgen: WorldgenKind::Diffusion,
        terrain: TerrainCfg::default(),
        seams: Seams::new(test_generator().atlases().to_vec()),
        generator: test_generator(),
        hooks: None,
        ops: Vec::new(),
        op_secrets: Vec::new(),
        mods_allow: Vec::new(),
        mods_deny: Vec::new(),
        store: None,
    }
}

/// The reference destructive pair as specs, with the target written into the ledger at `cell`.
pub(super) fn place_pair(shared: &Arc<Mutex<State>>, cell: Pos) -> (String, String) {
    let mut state = shared.lock_recover();
    let (a, e) = crate::sim::reactions::destructive_pair(&mut state.registry);
    let (sa, se) = (state.registry.spec(a), state.registry.spec(e));
    let spec = state.intern(&sa).unwrap();
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

pub(super) fn age_move_state(state: &mut State, id: u32) {
    if let Some(h) = state.players.get_mut(&id) {
        h.last_move = Instant::now() - Duration::from_secs(10);
    }
}

pub(super) fn drain_msgs(rx: &std::sync::mpsc::Receiver<Arc<[u8]>>) -> Vec<ServerMessage> {
    let mut out = Vec::new();
    while let Ok(frame) = rx.try_recv() {
        out.push(ServerMessage::decode(&frame).unwrap());
    }
    out
}

pub(super) fn raw_payload_reply(addr: SocketAddr, payload: &[u8]) -> ServerMessage {
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

pub(super) fn reject_payload(addr: SocketAddr, payload: &[u8]) -> String {
    match raw_payload_reply(addr, payload) {
        ServerMessage::Reject { reason } => reason.to_string(),
        other => panic!("expected Reject, got {other:?}"),
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

/// Anchor the envelope one realistic move gap ago: the next move's budget refills for that long.
pub(super) fn stamp_gap(shared: &Arc<Mutex<State>>, id: u32) {
    if let Some(h) = shared.lock_recover().players.get_mut(&id) {
        h.last_move = Instant::now() - Duration::from_millis(100);
    }
}

pub(super) fn flat(config: Config) -> ServerHandle {
    spawn(0, Config { seed: 1, worldgen: WorldgenKind::Flat, ..config }).unwrap()
}

pub(super) fn drain(rx: &std::sync::mpsc::Receiver<Arc<[u8]>>) -> Vec<ServerMessage> {
    let mut out = Vec::new();
    while let Ok(frame) = rx.try_recv() {
        if let Some(msg) = ServerMessage::decode(&frame) {
            out.push(msg);
        }
    }
    out
}

pub(super) fn flat_shared(
    players: HashMap<u32, PlayerHandle>,
    noclip: NoclipPolicy,
    ops: &[&str],
) -> (Arc<Mutex<State>>, Ctx) {
    let mut registry = BlockRegistry::with_builtins();
    let generator: crate::world::terrain::Generator =
        Arc::new(crate::world::generation::FlatTerrain::new(&mut registry, 1));
    let state = State {
        edits: HashMap::new(),
        spec_pool: HashMap::new(),
        registry,
        players: players.into_iter().collect(),
        grid: HashMap::new(),
        next_id: 2,
        day: 0.3,
        day_set: Instant::now(),
        reactions: ReactionScheduler::new(),
        tick: 1,
        poses: PosesWriter::new(),
        scratch: Scratch::default(),
        terrain: TerrainCache::new(),
        max_speed: crate::player::MAX_SPEED,
        panic_tick: false,
    };
    let ctx = Ctx {
        password: String::new(),
        seed: 1,
        content: crate::net::content_id(&BlockRegistry::with_builtins()),
        day_secs: 600.0,
        teleport: TeleportPolicy::All,
        noclip,
        worldgen: WorldgenKind::Flat,
        terrain: TerrainCfg::default(),
        seams: Seams::new(generator.atlases().to_vec()),
        generator,
        hooks: None,
        ops: ops.iter().map(|name| (*name).to_ascii_lowercase()).collect(),
        op_secrets: Vec::new(),
        mods_allow: Vec::new(),
        mods_deny: Vec::new(),
        store: None,
    };
    (Arc::new(Mutex::new(state)), ctx)
}

pub(super) fn pose(pos: DVec3) -> (HashMap<u32, PlayerHandle>, std::sync::mpsc::Receiver<Arc<[u8]>>) {
    let (out, rx) = sync_channel::<Arc<[u8]>>(8);
    let mut players = HashMap::new();
    players.insert(1u32, test_player(pos, out, test_kick()));
    (players, rx)
}

/// Chat until a line containing `want` arrives, or three seconds pass.
pub(super) fn chat_until(conn: &mut crate::net::client::Connection, want: &str) -> Vec<String> {
    let mut texts = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline && !texts.iter().any(|t: &String| t.contains(want)) {
        for event in conn.poll() {
            if let crate::net::client::Incoming::Chat { text, .. } = event {
                texts.push(text.to_string());
            }
        }
        thread::sleep(Duration::from_millis(10));
    }
    texts
}
