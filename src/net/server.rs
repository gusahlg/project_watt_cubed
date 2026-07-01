//! The authoritative, headless multiplayer server: it owns the one true world
//! (seed + edit overlay) and the roster of connected players, and it never trusts a
//! client. It runs with no window, no GPU, and no chunk machinery — terrain is
//! procedural, so the server only tracks the *seed* and the sparse overlay of
//! *edits*, each an opaque portable block spec ([`save`](crate::save)). That makes it
//! tiny to run and lets it scale to many players on a cheap box.
//!
//! **Threading.** One accept thread; per client a blocking reader thread and a
//! bounded-queue writer thread, coordinated through a single [`Mutex`]-guarded
//! [`State`]. The lock is held only for short, allocation-light bursts. This
//! comfortably serves hundreds of players; past that the single lock and
//! thread-per-client model become the ceiling, and an event-loop rewrite would be
//! the next step — called out honestly rather than hidden.
//!
//! **Optimisation.** No voxel data is ever sent — a join transfers the seed plus the
//! edit overlay, and live play is just small position/edit/chat frames. Position
//! broadcasts are interest-managed (only players within [`INTEREST_RADIUS`] hear a
//! move), which keeps the busiest traffic sub-quadratic as the roster grows.
//!
//! **Trust.** Joins are password-gated and version-checked; frames are size-capped by
//! the [`protocol`] framing; every client is rate-limited; and every edit is bounds-
//! and reach-validated against the sender's own reported position before it is
//! recorded.
use std::collections::HashMap;
use std::io;
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use raylib::prelude::*;

use crate::block::registry::BlockRegistry;
use crate::net::protocol::{self, ClientMessage, ServerMessage};
use crate::net::{MAX_CHAT, MAX_NAME, MAX_SPEC, PROTOCOL_VERSION, chat};
use crate::world::chunk::CHUNK_HEIGHT;
use crate::world::generation::{SineHills, TerrainGenerator};

/// Largest concurrent roster. A hard bound so a flood of connects can't spawn
/// unbounded threads.
const MAX_PLAYERS: usize = 256;
/// Depth of a client's outbound frame queue. A client that falls this far behind is
/// treated as unresponsive and dropped, so one slow peer can't grow memory without
/// bound.
const OUT_CAPACITY: usize = 1024;
/// A connection that sends nothing for this long is reaped (clients heartbeat well
/// under it). Also bounds how long a stalled handshake can squat a thread.
const IDLE_TIMEOUT: Duration = Duration::from_secs(30);
/// How long a client has to send its `Hello` before we hang up.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
/// Message budget per client per second; excess frames are dropped. Blunts flooding.
const RATE_LIMIT: u32 = 300;
/// A position update is only sent to players within this many world units of the
/// mover — nobody past render distance needs it.
const INTEREST_RADIUS: f32 = 160.0;
/// A client may edit a block at most this far from its own reported eye position;
/// farther edits are rejected as bogus. A little past the client's reach constant.
const EDIT_REACH: f32 = 8.0;
/// Edits are streamed to a joining client in batches this size, so a very built-up
/// world's snapshot never overflows a single frame's size cap.
const SNAPSHOT_BATCH: usize = 512;
/// Average terrain height the generator oscillates around — matches the client's
/// [`World`](crate::world::World::new) so server spawn heights land on real ground.
const TERRAIN_BASE: f32 = 20.0;

/// The public knobs for a server. Built by the dedicated binary and the in-game host.
pub struct Config {
    /// Password every client must present. Empty means no password is required.
    pub password: String,
    /// The world seed all clients generate their terrain from.
    pub seed: i64,
}

/// Immutable per-server context shared with every connection handler: the auth
/// password, the seed, and just enough of the generator to place spawns on ground.
struct Ctx {
    password: String,
    seed: i64,
    generator: SineHills,
}

/// One connected player as the server tracks them.
struct PlayerHandle {
    name: String,
    pos: Vector3,
    yaw: f32,
    pitch: f32,
    /// Outbound queue drained by this client's writer thread.
    out: SyncSender<Arc<[u8]>>,
    /// A clone of the socket, kept only to force-close a misbehaving client.
    kick: TcpStream,
}

/// The single piece of shared, mutable server state: the authoritative edit overlay
/// (coordinate → portable block spec) and the player roster.
struct State {
    edits: HashMap<(i32, i32, i32), String>,
    players: HashMap<u32, PlayerHandle>,
    next_id: u32,
}

/// A running server. [`stop`](ServerHandle::stop)ping it takes the listener down;
/// existing clients finish on their own.
pub struct ServerHandle {
    shutdown: Arc<AtomicBool>,
    addr: SocketAddr,
}

impl ServerHandle {
    /// The address the server is actually listening on (the resolved port when the
    /// caller bound to port 0).
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Stop accepting new connections. Existing clients finish on their own sockets.
    pub fn stop(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }
}

/// Bind `port` and start serving in the background, returning a handle with the
/// resolved address. Bind to port 0 to let the OS pick a free port.
pub fn spawn(port: u16, config: Config) -> io::Result<ServerHandle> {
    let listener = TcpListener::bind(("0.0.0.0", port))?;
    let addr = listener.local_addr()?;
    let shutdown = Arc::new(AtomicBool::new(false));

    // Resolve the generator's palette once so spawn heights match the client terrain.
    let registry = BlockRegistry::with_builtins();
    let ctx = Arc::new(Ctx {
        password: config.password,
        seed: config.seed,
        generator: SineHills::new(&registry, TERRAIN_BASE, config.seed),
    });
    let shared = Arc::new(Mutex::new(State {
        edits: HashMap::new(),
        players: HashMap::new(),
        next_id: 1,
    }));

    let accept_shutdown = shutdown.clone();
    thread::spawn(move || accept_loop(listener, shared, ctx, accept_shutdown));

    Ok(ServerHandle { shutdown, addr })
}

/// Bind and serve on the current thread until the process exits — the dedicated
/// server's entry point.
pub fn run(port: u16, config: Config) -> io::Result<()> {
    let handle = spawn(port, config)?;
    println!("watt-cubed server listening on {}", handle.addr());
    // The accept loop runs on its own thread; park this one so the process lives.
    loop {
        thread::sleep(Duration::from_secs(3600));
    }
}

/// Accept connections until asked to stop, handing each to its own handler thread.
fn accept_loop(listener: TcpListener, shared: Arc<Mutex<State>>, ctx: Arc<Ctx>, shutdown: Arc<AtomicBool>) {
    // Non-blocking accept so the loop can notice `stop()` between connections.
    let _ = listener.set_nonblocking(true);
    while !shutdown.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, addr)) => {
                let _ = stream.set_nonblocking(false);
                let _ = stream.set_nodelay(true);
                let shared = shared.clone();
                let ctx = ctx.clone();
                thread::spawn(move || {
                    // A dropped connection is routine; the error is the disconnect cause.
                    let _ = handle_client(stream, addr, shared, ctx);
                });
            }
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(50));
            }
            Err(_) => thread::sleep(Duration::from_millis(200)),
        }
    }
}

/// Drive one client: authenticate, register, stream the world snapshot, then relay
/// its messages until it disconnects, tidying up on the way out.
fn handle_client(stream: TcpStream, addr: SocketAddr, shared: Arc<Mutex<State>>, ctx: Arc<Ctx>) -> io::Result<()> {
    // The first frame must be a valid, authenticated Hello within the handshake window.
    stream.set_read_timeout(Some(HANDSHAKE_TIMEOUT))?;
    let mut reader = stream.try_clone()?;
    let hello = protocol::read_frame(&mut reader)?;
    let name = match ClientMessage::decode(&hello) {
        Some(ClientMessage::Hello { protocol, name, password }) => {
            if protocol != PROTOCOL_VERSION {
                reject(&stream, "protocol version mismatch");
                return Ok(());
            }
            if password != ctx.password {
                reject(&stream, "wrong password");
                return Ok(());
            }
            clean_name(&name)
        }
        _ => {
            reject(&stream, "expected hello");
            return Ok(());
        }
    };

    // Authenticated: switch to the idle timeout and wire up the writer.
    stream.set_read_timeout(Some(IDLE_TIMEOUT))?;
    let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let mut writer_stream = stream.try_clone()?;
    let writer = thread::spawn(move || {
        for frame in rx.iter() {
            if protocol::write_frame(&mut writer_stream, &frame).is_err() {
                break;
            }
        }
    });

    // Register the player and gather what the newcomer needs to bootstrap. Done in
    // one locked scope so the id, spawn, and roster it sees are all consistent.
    let id;
    let spawn;
    let existing: Vec<(u32, String, Vector3, f32, f32)>;
    let snapshot: Vec<(i32, i32, i32, String)>;
    {
        let mut state = shared.lock().unwrap();
        if state.players.len() >= MAX_PLAYERS {
            drop(state);
            reject(&stream, "server full");
            return Ok(());
        }
        id = state.next_id;
        state.next_id += 1;
        spawn = spawn_point(&ctx.generator, id);

        existing = state
            .players
            .iter()
            .map(|(&pid, h)| (pid, h.name.clone(), h.pos, h.yaw, h.pitch))
            .collect();
        snapshot = state
            .edits
            .iter()
            .map(|(&(x, y, z), spec)| (x, y, z, spec.clone()))
            .collect();

        state.players.insert(
            id,
            PlayerHandle {
                name: name.clone(),
                pos: spawn,
                yaw: 0.0,
                pitch: 0.0,
                out: out.clone(),
                kick: stream.try_clone()?,
            },
        );
    }
    println!("[+] {name} joined as #{id} from {addr} ({} online)", online(&shared));

    // Bootstrap the newcomer: who they are, the world edits, and who else is here.
    send(&out, &ServerMessage::Welcome { player_id: id, seed: ctx.seed, spawn });
    for batch in snapshot.chunks(SNAPSHOT_BATCH) {
        send(&out, &ServerMessage::Snapshot { edits: batch.to_vec() });
    }
    for (pid, pname, ppos, pyaw, ppitch) in existing {
        send(&out, &ServerMessage::PeerJoined { id: pid, name: pname });
        send(&out, &ServerMessage::PeerMove { id: pid, pos: ppos, yaw: pyaw, pitch: ppitch });
    }
    // Announce the newcomer to everyone already connected.
    broadcast_all(&shared, &ServerMessage::PeerJoined { id, name: name.clone() }, Some(id));

    // Relay loop with a light per-second rate limiter.
    let mut window = Instant::now();
    let mut count: u32 = 0;
    loop {
        let frame = match protocol::read_frame(&mut reader) {
            Ok(f) => f,
            Err(_) => break, // EOF, timeout, or a malformed length: the client is gone.
        };

        if window.elapsed() >= Duration::from_secs(1) {
            window = Instant::now();
            count = 0;
        }
        count += 1;
        if count > RATE_LIMIT {
            continue; // Over budget this second — drop the frame rather than serve a flood.
        }

        let Some(msg) = ClientMessage::decode(&frame) else {
            continue; // Junk frame; ignore it.
        };
        match msg {
            ClientMessage::Move { pos, yaw, pitch } => on_move(&shared, id, pos, yaw, pitch),
            ClientMessage::Edit { x, y, z, spec } => on_edit(&shared, id, x, y, z, &spec),
            ClientMessage::Chat { channel, text } => on_chat(&shared, id, channel, &text),
            ClientMessage::Hello { .. } => {} // Already authenticated; ignore repeats.
        }
    }

    // Cleanup: drop the player (which frees the writer), close the socket, tell peers.
    {
        let mut state = shared.lock().unwrap();
        state.players.remove(&id);
    }
    let _ = stream.shutdown(Shutdown::Both);
    drop(out);
    let _ = writer.join();
    broadcast_all(&shared, &ServerMessage::PeerLeft { id }, None);
    println!("[-] {name} (#{id}) left ({} online)", online(&shared));
    Ok(())
}

/// Apply a validated position update and fan it out to interested players only.
fn on_move(shared: &Arc<Mutex<State>>, id: u32, pos: Vector3, yaw: f32, pitch: f32) {
    // Ignore non-finite coordinates outright (a NaN would poison distance checks).
    if !pos.x.is_finite() || !pos.y.is_finite() || !pos.z.is_finite() {
        return;
    }
    let mut state = shared.lock().unwrap();
    if let Some(h) = state.players.get_mut(&id) {
        h.pos = pos;
        h.yaw = yaw;
        h.pitch = pitch;
    }
    let frame: Arc<[u8]> = ServerMessage::PeerMove { id, pos, yaw, pitch }.encode().into();
    let mut slow = Vec::new();
    for (&pid, h) in &state.players {
        if pid == id || h.pos.distance(pos) > INTEREST_RADIUS {
            continue;
        }
        if h.out.try_send(frame.clone()).is_err() {
            slow.push(pid);
        }
    }
    kick_slow(&state, &slow);
}

/// Validate and record a block edit, then broadcast it to every other player so all
/// overlays stay in agreement.
fn on_edit(shared: &Arc<Mutex<State>>, id: u32, x: i32, y: i32, z: i32, spec: &str) {
    if spec.len() > MAX_SPEC || y < 0 || y >= CHUNK_HEIGHT as i32 {
        return;
    }
    let mut state = shared.lock().unwrap();
    // Reach check against the editor's own reported position — no reaching across
    // the map.
    let Some(h) = state.players.get(&id) else { return };
    let target = Vector3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
    if h.pos.distance(target) > EDIT_REACH {
        return;
    }
    // The overlay stores the portable spec verbatim; the server never resolves it.
    state.edits.insert((x, y, z), spec.to_string());
    let msg = ServerMessage::Edit { x, y, z, spec: spec.to_string() };
    broadcast(&state, &msg, |pid, _| pid != id);
}

/// Relay a chat line to its audience: proximity for local, everyone for global.
fn on_chat(shared: &Arc<Mutex<State>>, id: u32, channel: u8, text: &str) {
    let text = clean_chat(text);
    if text.is_empty() {
        return;
    }
    let state = shared.lock().unwrap();
    let Some(sender) = state.players.get(&id) else { return };
    let from_name = sender.name.clone();
    let origin = sender.pos;
    let channel = if channel == chat::GLOBAL { chat::GLOBAL } else { chat::LOCAL };
    println!("<{from_name}> {text}");
    let msg = ServerMessage::Chat { from_id: id, from_name, channel, text };
    broadcast(&state, &msg, |_, h| {
        channel == chat::GLOBAL || h.pos.distance(origin) <= chat::RADIUS
    });
}

/// Send one message to every player matching `want`, encoding it just once. Players
/// whose queue is full are force-closed (they've fallen too far behind).
fn broadcast(state: &State, msg: &ServerMessage, want: impl Fn(u32, &PlayerHandle) -> bool) {
    let frame: Arc<[u8]> = msg.encode().into();
    let mut slow = Vec::new();
    for (&pid, h) in &state.players {
        if !want(pid, h) {
            continue;
        }
        match h.out.try_send(frame.clone()) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => slow.push(pid),
            Err(TrySendError::Disconnected(_)) => {}
        }
    }
    kick_slow(state, &slow);
}

/// Broadcast to everyone, optionally skipping one id (the originator).
fn broadcast_all(shared: &Arc<Mutex<State>>, msg: &ServerMessage, except: Option<u32>) {
    let state = shared.lock().unwrap();
    broadcast(&state, msg, |pid, _| Some(pid) != except);
}

/// Force-close clients that couldn't keep up. Their reader threads then wake, error,
/// and run the normal cleanup path (emitting `PeerLeft`).
fn kick_slow(state: &State, ids: &[u32]) {
    for id in ids {
        if let Some(h) = state.players.get(id) {
            let _ = h.kick.shutdown(Shutdown::Both);
        }
    }
}

/// Queue one message to a single client (best-effort; a full queue drops it).
fn send(out: &SyncSender<Arc<[u8]>>, msg: &ServerMessage) {
    let frame: Arc<[u8]> = msg.encode().into();
    let _ = out.try_send(frame);
}

/// Reply with a rejection and let the socket close.
fn reject(stream: &TcpStream, reason: &str) {
    let mut w = stream;
    let _ = protocol::write_frame(&mut w, &ServerMessage::Reject { reason: reason.into() }.encode());
    println!("[x] rejected a connection: {reason}");
}

/// A spawn point just above the origin surface, scattered a little per id so players
/// don't stack on the exact same block.
fn spawn_point(generator: &SineHills, id: u32) -> Vector3 {
    // A cheap deterministic scatter on a small grid around origin.
    let x = (id % 8) as i32 - 3;
    let z = ((id / 8) % 8) as i32 - 3;
    let surface = generator.height(x, z);
    Vector3::new(x as f32 + 0.5, surface as f32 + 3.0, z as f32 + 0.5)
}

/// Current player count.
fn online(shared: &Arc<Mutex<State>>) -> usize {
    shared.lock().unwrap().players.len()
}

/// Trim a name to the length cap and strip control characters; fall back to a
/// generic label if nothing usable remains.
fn clean_name(raw: &str) -> String {
    let name: String = raw.chars().filter(|c| !c.is_control()).take(MAX_NAME).collect();
    let name = name.trim().to_string();
    if name.is_empty() { "player".to_string() } else { name }
}

/// Trim a chat line to the length cap and strip control characters.
fn clean_chat(raw: &str) -> String {
    raw.chars().filter(|c| !c.is_control()).take(MAX_CHAT).collect::<String>().trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_generator() -> SineHills {
        let registry = BlockRegistry::with_builtins();
        SineHills::new(&registry, TERRAIN_BASE, 4242)
    }

    #[test]
    fn names_are_capped_and_sanitised() {
        assert_eq!(clean_name("  bob\n "), "bob");
        assert_eq!(clean_name(""), "player");
        assert_eq!(clean_name(&"x".repeat(100)).len(), MAX_NAME);
    }

    #[test]
    fn chat_is_sanitised() {
        assert_eq!(clean_chat("hi\tthere\n"), "hithere");
        assert_eq!(clean_chat(&"a".repeat(500)).len(), MAX_CHAT);
    }

    #[test]
    fn spawn_points_sit_above_the_surface() {
        let terrain = test_generator();
        for id in 1..20 {
            let p = spawn_point(&terrain, id);
            let ground = terrain.height(p.x.floor() as i32, p.z.floor() as i32);
            assert!(p.y > ground as f32, "spawn should be above ground");
        }
    }

    /// An out-of-reach edit must be dropped; an in-reach one must be recorded.
    /// Exercised directly against the shared state without a socket.
    #[test]
    fn edit_reach_is_enforced() {
        let (out, _rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        // A throwaway loopback socket just to fill the `kick` handle.
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let stream = TcpStream::connect(listener.local_addr().unwrap()).unwrap();

        let mut players = HashMap::new();
        players.insert(
            1u32,
            PlayerHandle {
                name: "p".into(),
                pos: Vector3::new(8.5, 20.0, 8.5),
                yaw: 0.0,
                pitch: 0.0,
                out,
                kick: stream,
            },
        );
        let shared = Arc::new(Mutex::new(State { edits: HashMap::new(), players, next_id: 2 }));

        on_edit(&shared, 1, 500, 20, 500, "air"); // far away: rejected
        on_edit(&shared, 1, 8, 20, 8, "air"); // in reach: recorded

        let state = shared.lock().unwrap();
        assert!(state.edits.contains_key(&(8, 20, 8)), "in-reach edit recorded");
        assert!(!state.edits.contains_key(&(500, 20, 500)), "out-of-reach edit dropped");
    }
}
