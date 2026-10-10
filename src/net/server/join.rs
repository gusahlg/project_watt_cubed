//! A connection's life outside its message loop: accept, handshake, admit, bootstrap and depart.
use super::*;

/// A joiner's copy of the edit overlay, taken under the lock without touching a spec.
pub(super) struct Overlay {
    pub(super) cells: Vec<(i32, i32, i32, u32, BlockId)>,
    /// The spec of each block the cells name, indexed by block id.
    pub(super) specs: Vec<Option<Arc<str>>>,
}

impl Overlay {
    pub(super) fn of(state: &State) -> Self {
        let mut specs: Vec<Option<Arc<str>>> = Vec::new();
        let mut cells = Vec::with_capacity(state.edits.len());
        for (&(x, y, z), cell) in &state.edits {
            let at = usize::from(cell.block.0);
            if at >= specs.len() {
                specs.resize(at + 1, None);
            }
            if specs[at].is_none() {
                specs[at] = Some(cell.spec.clone());
            }
            cells.push((x, y, z, cell.rev, cell.block));
        }
        Self { cells, specs }
    }
}

/// Decrements the pre-auth connection count when a handshake ends, however it
/// ends — success, rejection, or a dropped socket all release the slot.
pub(super) struct HandshakeSlot(Arc<AtomicUsize>);

impl Drop for HandshakeSlot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

pub(super) fn accept_loop(
    endpoint: Endpoint,
    rt: Arc<Runtime>,
    shared: Arc<Mutex<State>>,
    ctx: Arc<Ctx>,
    shutdown: Arc<AtomicBool>,
    pending: Arc<AtomicUsize>,
) {
    let mut clients: Vec<JoinHandle<()>> = Vec::new();
    while !shutdown.load(Ordering::Relaxed) {
        clients.retain(|handle| !handle.is_finished());
        let step = catch_unwind(AssertUnwindSafe(|| {
            accept_once(&endpoint, &rt, &shared, &ctx, &pending, &mut clients)
        }));
        match step {
            Ok(true) => break,
            Ok(false) => {}
            Err(payload) => eprintln!("accept loop panicked: {}", panic_text(&payload)),
        }
    }
    for handle in clients {
        let _ = handle.join();
    }
}

/// One accept. `true` when the endpoint has closed. A panic here drops a local
/// [`HandshakeSlot`] if one was taken, so the pre-auth count cannot stick.
pub(super) fn accept_once(
    endpoint: &Endpoint,
    rt: &Arc<Runtime>,
    shared: &Arc<Mutex<State>>,
    ctx: &Arc<Ctx>,
    pending: &Arc<AtomicUsize>,
    clients: &mut Vec<JoinHandle<()>>,
) -> bool {
    // Bounded wait so `stop()` is noticed between connections. Closing the
    // endpoint makes `accept` return `None` and the loop joins every client.
    let incoming = match rt
        .block_on(async { tokio::time::timeout(Duration::from_millis(200), endpoint.accept()).await })
    {
        Ok(Some(incoming)) => incoming,
        Ok(None) => return true,
        Err(_elapsed) => return false,
    };

    // A stateless retry first: a source must prove it receives our packets before it
    // takes a slot, so spoofed Initials cannot hold the handshake slots.
    if !incoming.remote_address_validated() {
        let _ = incoming.retry();
        return false;
    }
    // A QUIC CONNECTION_REFUSED past the cap, before any handshake, so a
    // flood of silent connects can't squat handler threads. The accept
    // loop is the only adder, so load-then-add can't overshoot the cap.
    if pending.load(Ordering::Relaxed) >= HANDSHAKE_CAP {
        incoming.refuse();
        return false;
    }
    pending.fetch_add(1, Ordering::Relaxed);
    let slot = HandshakeSlot(Arc::clone(pending));
    let shared = Arc::clone(shared);
    let ctx = Arc::clone(ctx);
    let handler_rt = Arc::clone(rt);
    clients.push(thread::spawn(move || {
        // A dropped connection is routine; the error is the disconnect cause.
        let _ = handle_client(incoming, handler_rt, shared, ctx, slot);
    }));
    false
}

pub(super) fn handle_client(
    incoming: Incoming,
    rt: Arc<Runtime>,
    shared: Arc<Mutex<State>>,
    ctx: Arc<Ctx>,
    slot: HandshakeSlot,
) -> io::Result<()> {
    let Some((conn, mut send, mut recv, mut frame, name, addr)) =
        handshake(incoming, &rt, &ctx, slot)?
    else {
        return Ok(());
    };
    let Some((id, spawn, existing, snapshot, out, rx, kick)) =
        admit_player(&shared, &ctx, &rt, &mut send, &conn, &name)
    else {
        return Ok(());
    };
    let writer = spawn_writer(rt.clone(), send, rx, kick.clone());
    println!("[+] {name} joined as #{id} from {addr} ({} online)", online(&shared));

    let kicked = shared.lock_recover().players.get(&id).map(|h| Arc::clone(&h.kicked));
    let Some(kicked) = kicked else {
        depart(&shared, &ctx, conn, out, writer, id, &name);
        return Ok(());
    };
    // These sends wait, bounded, on this client's handler thread: a built-up
    // world or big roster can exceed the outbound queue, and dropping bootstrap
    // frames would ghost the join. A kick or a missed deadline ends the join.
    if !send_join(&out, &kicked, &shared, &ctx, id, spawn, snapshot, &existing) {
        kicked.store(true, Ordering::Relaxed);
        depart(&shared, &ctx, conn, out, writer, id, &name);
        return Ok(());
    }
    // Bootstrap queued: go live. Frames broadcast during the bootstrap window
    // were buffered; drain them in order (they postdate the snapshot) and only
    // then let broadcasters push directly.
    {
        let mut state = shared.lock_recover();
        let mut slow = false;
        if let Some(h) = state.players.get_mut(&id) {
            h.backlog_bytes = 0;
            for queued in std::mem::take(&mut h.backlog) {
                if h.out.try_send(queued.frame).is_err() {
                    slow = true;
                    break;
                }
            }
            h.ready = true;
        }
        if slow {
            kick_slow(&state, &[id]);
        }
    }

    broadcast_all(&shared, &ServerMessage::PeerJoined { id, name: name.clone() }, Some(id));

    client_loop(&rt, &mut recv, &mut frame, &kick, &shared, &ctx, id);
    depart(&shared, &ctx, conn, out, writer, id, &name);
    Ok(())
}

pub(super) fn handshake(
    incoming: Incoming,
    rt: &Runtime,
    ctx: &Ctx,
    slot: HandshakeSlot,
) -> io::Result<Option<(quinn::Connection, SendStream, quinn::RecvStream, Vec<u8>, Arc<str>, SocketAddr)>> {
    // The scratch Vec is reused for every frame this client ever sends.
    let conn = rt.block_on(async { incoming.await }).map_err(io::Error::other)?;
    let addr = conn.remote_address();
    let (mut send, mut recv) = rt
        .block_on(async { tokio::time::timeout(HANDSHAKE_TIMEOUT, conn.accept_bi()).await })
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "handshake timed out"))?
        .map_err(io::Error::other)?;

    let mut frame = Vec::new();
    rt.block_on(async {
        tokio::time::timeout(HANDSHAKE_TIMEOUT, protocol::read_frame_async(&mut recv, &mut frame)).await
    })
    .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "handshake timed out"))??;

    let name = match hello_name(&frame, ctx) {
        Ok(name) => clean_name(&name),
        Err(HelloFail::Reject(reason)) => {
            reject(rt, &mut send, &conn, &reason);
            return Ok(None);
        }
        Err(HelloFail::Mods(ids)) => {
            deny_mods(rt, &mut send, &conn, ids);
            return Ok(None);
        }
    };
    // Pre-auth window is over; the roster's own MAX_PLAYERS bound takes over.
    drop(slot);
    Ok(Some((conn, send, recv, frame, name, addr)))
}

pub(super) enum HelloFail {
    Reject(String),
    Mods(Vec<ModId>),
}

/// Protocol number first, then the content parts. A tag or version mismatch is
/// named without decoding the rest of the payload (a v12 `Hello` is a different shape).
/// Password is checked before the mod list, so a scanner without the password
/// learns nothing about the whitelist.
pub(super) fn hello_name(frame: &[u8], ctx: &Ctx) -> Result<Arc<str>, HelloFail> {
    let protocol = match protocol::peek_hello(frame) {
        protocol::HelloPeek::NotHello => return Err(HelloFail::Reject("expected hello".into())),
        protocol::HelloPeek::Truncated => return Err(HelloFail::Reject("malformed hello".into())),
        protocol::HelloPeek::Protocol(v) => v,
    };
    if protocol != PROTOCOL_VERSION {
        return Err(HelloFail::Reject(format!(
            "protocol version mismatch: server v{PROTOCOL_VERSION}, client v{protocol}"
        )));
    }
    match ClientMessage::decode(frame) {
        Some(ClientMessage::Hello { worldgen, gravity, law, palette, name, password, mods, .. }) => {
            let client = crate::net::ContentId { worldgen, gravity, law, palette };
            if let Some(why) = crate::net::content_mismatch(ctx.content, client) {
                return Err(HelloFail::Reject(why));
            }
            if *password != *ctx.password {
                return Err(HelloFail::Reject("wrong password".into()));
            }
            let refused = refused_mods(ctx, &mods);
            if !refused.is_empty() {
                return Err(HelloFail::Mods(refused));
            }
            Ok(name)
        }
        _ => Err(HelloFail::Reject("malformed hello".into())),
    }
}

/// Allow list (when set) admits only those ids. Deny always refuses. An id in
/// both is refused. Ids match the registered package id, case-sensitive.
pub(super) fn refused_mods(ctx: &Ctx, mods: &[ModOffer]) -> Vec<ModId> {
    let mut refused = Vec::new();
    for offer in mods {
        let id = offer.id.as_str();
        let denied = ctx.mods_deny.iter().any(|d| d == id);
        let blocked = !ctx.mods_allow.is_empty() && !ctx.mods_allow.iter().any(|a| a == id);
        if (denied || blocked) && !refused.iter().any(|have: &ModId| have.as_str() == id) {
            refused.push(offer.id.clone());
        }
    }
    refused
}

pub(super) fn admit_player(
    shared: &Arc<Mutex<State>>,
    ctx: &Ctx,
    rt: &Runtime,
    send: &mut SendStream,
    conn: &quinn::Connection,
    name: &Arc<str>,
) -> Option<(
    u32,
    DVec3,
    Vec<(u32, Arc<str>)>,
    Overlay,
    Outbox,
    Outgoing,
    Arc<Notify>,
)> {
    // Made before the lock, and the writer spawned only after a slot is
    // secured, so the still-owned `send` handles a "server full" reject
    // directly and reliably.
    let (out, rx) = outbox(OUT_CAPACITY);
    let kick = Arc::new(Notify::new());

    // One locked scope so the id, spawn, and roster snapshot are consistent.
    let id;
    let spawn;
    let existing: Vec<(u32, Arc<str>)>;
    let snapshot: Overlay;
    {
        let mut state = shared.lock_recover();
        if reserved_name(name) {
            drop(state);
            reject(rt, send, conn, "the name 'server' is reserved");
            return None;
        }
        if state.players.values().any(|h| h.name.eq_ignore_ascii_case(name)) {
            drop(state);
            reject(rt, send, conn, "that name is already in use");
            return None;
        }
        if state.players.len() >= MAX_PLAYERS {
            drop(state);
            reject(rt, send, conn, "server full");
            return None;
        }
        id = state.next_id;
        state.next_id += 1;
        spawn = spawn_point(ctx.generator.as_ref(), id);
        let (frame, up) = standing_pose(Field::new(ctx.generator.mass()).sample(spawn).accel);
        let op = ctx.ops.iter().any(|op| op.eq_ignore_ascii_case(name));
        (existing, snapshot) = state.admit(id, PlayerHandle::new(name.clone(), spawn, frame, up, op, out.clone(), kick.clone()));
    }
    if let Some(hooks) = ctx.hooks.as_ref() {
        hooks.lock_recover().on_join(&JoinFacts::at(id, name.clone(), spawn));
    }
    Some((id, spawn, existing, snapshot, out, rx, kick))
}

pub(super) fn depart(
    shared: &Arc<Mutex<State>>,
    ctx: &Ctx,
    conn: quinn::Connection,
    out: Outbox,
    writer: thread::JoinHandle<()>,
    id: u32,
    name: &Arc<str>,
) {
    let mut left: Option<JoinFacts> = None;
    {
        let mut state = shared.lock_recover();
        if let Some(h) = state.remove_player(id) {
            left = Some(JoinFacts::at(id, h.name.clone(), h.pose.pos));
        }
    }
    if let (Some(hooks), Some(facts)) = (ctx.hooks.as_ref(), left) {
        hooks.lock_recover().on_leave(&facts);
    }
    // Close the connection FIRST: it errors any write the writer is stuck on for a
    // slow client, so dropping `out` and joining actually completes.
    conn.close(0u32.into(), b"bye");
    drop(out);
    let _ = writer.join();
    broadcast_all(shared, &ServerMessage::PeerLeft { id }, None);
    println!("[-] {name} (#{id}) left ({} online)", online(shared));
}

/// Welcome, the edit overlay, [`ServerMessage::SnapshotEnd`], the clock, and
/// the roster. False when a send misses its deadline or the player was kicked.
pub(super) fn send_join(
    out: &Outbox,
    kicked: &AtomicBool,
    shared: &Arc<Mutex<State>>,
    ctx: &Ctx,
    id: u32,
    spawn: DVec3,
    overlay: Overlay,
    existing: &[(u32, Arc<str>)],
) -> bool {
    if !send_blocking(
        out,
        kicked,
        &ServerMessage::Welcome {
            player_id: id,
            seed: ctx.seed,
            spawn,
            worldgen: ctx.worldgen,
            terrain: ctx.terrain,
            law: crate::net::protocol::law_stamp(),
        },
    ) {
        return false;
    }
    let Overlay { mut cells, specs } = overlay;
    // Chunk by chunk, so the coordinate deltas stay small.
    cells.sort_unstable_by_key(|&(x, y, z, _, _)| (x >> 4, z >> 4, y >> 4, x, z, y));
    let failed = std::cell::Cell::new(false);
    let mut emit = |frame| {
        if !failed.get() && !send_until(out, kicked, frame, Instant::now() + SEND_DEADLINE) {
            failed.set(true);
        }
    };
    let mut writer = SnapshotWriter::new();
    for &(x, y, z, rev, block) in &cells {
        let spec = specs.get(usize::from(block.0)).and_then(|s| s.as_deref()).unwrap_or("air");
        writer.push((x, y, z), rev, block.0, spec, &mut emit);
        if failed.get() {
            return false;
        }
    }
    writer.finish(&mut emit);
    if failed.get() || !send_blocking(out, kicked, &ServerMessage::SnapshotEnd) {
        return false;
    }
    // Read the clock at send time. A snapshot of a built-up world can take long
    // enough that the phase captured at admit would already be stale.
    let day = shared.lock_recover().day_now(ctx.day_secs);
    if !send_blocking(out, kicked, &ServerMessage::Time { day, day_secs: ctx.day_secs }) {
        return false;
    }
    for (pid, pname) in existing {
        if !send_blocking(out, kicked, &ServerMessage::PeerJoined { id: *pid, name: Arc::clone(pname) }) {
            return false;
        }
    }
    true
}

/// Only safe on the receiving client's own handler thread (used for the join
/// bootstrap, which must not drop frames). Returns false when `kicked` is set,
/// the queue is gone, or [`SEND_DEADLINE`] passes.
pub(super) fn send_blocking(out: &Outbox, kicked: &AtomicBool, msg: &ServerMessage) -> bool {
    send_until(out, kicked, msg.frame(), Instant::now() + SEND_DEADLINE)
}

pub(super) fn send_until(out: &Outbox, kicked: &AtomicBool, frame: Arc<[u8]>, deadline: Instant) -> bool {
    loop {
        if kicked.load(Ordering::Relaxed) {
            return false;
        }
        match out.try_send(Arc::clone(&frame)) {
            Ok(()) => return true,
            Err(TrySendError::Disconnected(_)) => return false,
            Err(TrySendError::Full(_)) => {
                if Instant::now() >= deadline {
                    kicked.store(true, Ordering::Relaxed);
                    return false;
                }
                thread::sleep(Duration::from_millis(2));
            }
        }
    }
}

pub(super) fn reject(rt: &Runtime, send: &mut SendStream, conn: &quinn::Connection, reason: &str) {
    refuse(rt, send, conn, &ServerMessage::Reject { reason: reason.into() });
    println!("[x] rejected a connection: {}", console_text(reason));
}

pub(super) fn deny_mods(rt: &Runtime, send: &mut SendStream, conn: &quinn::Connection, ids: Vec<ModId>) {
    let listed = ids.iter().map(|id| console_text(id)).collect::<Vec<_>>().join(", ");
    refuse(rt, send, conn, &ServerMessage::ModsDenied { ids });
    println!("[x] refused mods: {listed}");
}

/// Send the last word of a refused connection and close the stream. QUIC (unlike TCP) can
/// discard buffered stream data when a connection closes, so we wait (bounded) for the peer
/// to close after reading — otherwise a refused client would see "no reply" instead of why.
fn refuse(rt: &Runtime, send: &mut SendStream, conn: &quinn::Connection, last: &ServerMessage) {
    rt.block_on(async {
        let _ = protocol::write_frame_async(send, &last.encode()).await;
        let _ = send.finish();
        let _ = tokio::time::timeout(REJECT_DRAIN, conn.closed()).await;
    });
}

/// Scattered a little per id so players don't stack on the exact same block; scans outward for
/// the first level column (its four neighbours within one block), like the single-player spawn.
pub(super) fn spawn_point(generator: &dyn TerrainGenerator, id: u32) -> DVec3 {
    if let Some(mut p) = generator.chart_spawn() {
        p.x += (id % 5) as f64 - 2.0;
        p.z += ((id / 5) % 5) as f64 - 2.0;
        return p;
    }
    let sx = (id % 8) as i32 - 3;
    let sz = ((id / 8) % 8) as i32 - 3;
    for r in 0..64 {
        for (dx, dz) in [(r, 0), (0, r), (-r, 0), (0, -r), (r, r), (-r, -r), (r, -r), (-r, r)] {
            let (x, z) = (sx + dx * 8, sz + dz * 8);
            let h = generator.height(x, z);
            let flat = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                .iter()
                .all(|&(ox, oz)| (generator.height(x + ox, z + oz) - h).abs() <= 1);
            if flat {
                return DVec3::new(x as f64 + 0.5, h as f64 + 3.0, z as f64 + 0.5);
            }
        }
    }
    let h = generator.height(sx, sz);
    DVec3::new(sx as f64 + 0.5, h as f64 + 3.0, sz as f64 + 0.5)
}

pub(super) fn online(shared: &Arc<Mutex<State>>) -> usize {
    shared.lock_recover().players.len()
}

pub(super) fn clean_name(raw: &str) -> Arc<str> {
    let name: String = raw.chars().filter(|c| !c.is_control()).take(MAX_NAME).collect();
    let name = name.trim();
    if name.is_empty() { "player".into() } else { name.into() }
}

/// Peer text for the server console with control characters escaped, so a peer
/// cannot drive the operator's terminal.
pub(super) fn console_text(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for c in raw.chars() {
        if c.is_control() {
            out.extend(c.escape_default());
        } else {
            out.push(c);
        }
    }
    out
}

/// The server's own chat name: any name whose letters and digits alone spell "server".
pub(super) fn reserved_name(name: &str) -> bool {
    name.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).eq("server".chars())
}
