//! Multiplayer session tests: joining, leaving and rejoining, the roster, names and passwords,
//! version and content refusals, the mod whitelist, operators and the shared clock, disconnect
//! reasons, the silence watchdog, and the world file across restarts. See `super` for the harness.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::net::{Ipv4Addr, SocketAddr, UdpSocket};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use quinn::Endpoint;
use tokio::runtime::Runtime;
use voxel_engine::DVec3;

use super::{Lobby, listen, settle};
use crate::block::AIR;
use crate::net::client::{ConnectError, Connection, INTERRUPTED, Incoming};
use crate::net::protocol::{self, ClientMessage, ModOffer, ServerMessage};
use crate::net::server::{self, Config};
use crate::net::{ContentId, PROTOCOL_VERSION, chat, quic};
use crate::save::format::{self, Decoded};
use crate::world::World;
use crate::world::generation::{FLAT_HEIGHT, WorldgenKind};
use crate::world::terrain::TerrainCfg;

const WAIT: Duration = Duration::from_secs(5);
const HOST: &str = "127.0.0.1";
const REFUSED_TIME: &str = "only an operator can set the time";
/// The clock of a server without a world file, at its start.
const FRESH_DAY: f32 = 0.3;
/// `SetTime` is budgeted one a second from when the server reads it, which is before its answer
/// arrives: this long after the answer, the next one is always in budget.
const SET_TIME_GAP: Duration = Duration::from_millis(1050);

/// Join and wait for the overlay and the clock, keeping every event the join produced.
fn enter(port: u16, name: &str, password: &str, mods: &[(String, String)]) -> (Connection, Vec<Incoming>) {
    let conn = Connection::connect_with(HOST, port, name, password, mods)
        .unwrap_or_else(|e| panic!("{name} should join: {e}"));
    let mut clients = [conn];
    let mut events = settle(&mut clients, WAIT, |c, seen| c.snapshot_ready() && clock(seen).is_some());
    let [conn] = clients;
    (conn, events.remove(0))
}

fn refused(port: u16, name: &str, password: &str, mods: &[(String, String)]) -> ConnectError {
    match Connection::connect_with(HOST, port, name, password, mods) {
        Ok(_) => panic!("{name:?} was admitted"),
        Err(e) => e,
    }
}

/// The game's join: a refusal that names mods turns those off and tries once more.
fn join_like_the_game(port: u16, name: &str, mods: &[(String, String)]) -> (Result<Connection, ConnectError>, Vec<String>) {
    match Connection::connect_with(HOST, port, name, "", mods) {
        Err(e) if !e.mods_denied.is_empty() => {
            let kept: Vec<(String, String)> =
                mods.iter().filter(|(id, _)| !e.mods_denied.contains(id)).cloned().collect();
            (Connection::connect_with(HOST, port, name, "", &kept), e.mods_denied)
        }
        other => (other, Vec::new()),
    }
}

fn offer(ids: &[&str]) -> Vec<(String, String)> {
    ids.iter().map(|id| (id.to_string(), "1.0.0".to_string())).collect()
}

fn flat(config: Config) -> Lobby {
    Lobby::start(Config { seed: 1, worldgen: WorldgenKind::Flat, ..config })
}

fn joined(events: &[Incoming]) -> Vec<String> {
    events.iter().filter_map(|e| if let Incoming::Joined { name } = e { Some(name.to_string()) } else { None }).collect()
}

fn left(events: &[Incoming]) -> Vec<String> {
    events.iter().filter_map(|e| if let Incoming::Left { name } = e { Some(name.to_string()) } else { None }).collect()
}

/// `(from, text)` of every chat line.
fn chats(events: &[Incoming]) -> Vec<(String, String)> {
    events
        .iter()
        .filter_map(|e| match e {
            Incoming::Chat { from_name, text, .. } => Some((from_name.to_string(), text.to_string())),
            _ => None,
        })
        .collect()
}

fn told(events: &[Incoming], want: &str) -> bool {
    chats(events).iter().any(|(_, text)| text == want)
}

/// The last `(day, day_secs)` heard.
fn clock(events: &[Incoming]) -> Option<(f32, f32)> {
    events.iter().rev().find_map(|e| if let Incoming::Time { day, day_secs } = e { Some((*day, *day_secs)) } else { None })
}

fn times(events: &[Incoming]) -> usize {
    events.iter().filter(|e| matches!(e, Incoming::Time { .. })).count()
}

fn reasons(events: &[Incoming]) -> Vec<String> {
    events.iter().filter_map(|e| if let Incoming::Disconnected { reason } = e { Some(reason.clone()) } else { None }).collect()
}

fn interruptions(events: &[Incoming]) -> usize {
    events.iter().filter(|e| matches!(e, Incoming::Interrupted)).count()
}

fn roster(conn: &Connection) -> Vec<String> {
    sorted(conn.peers().map(|p| p.name.to_string()).collect())
}

fn sorted(mut names: Vec<String>) -> Vec<String> {
    names.sort();
    names
}

/// Distance between two day fractions, the short way round the clock.
fn near(a: f32, b: f32, tol: f32) -> bool {
    let d = (a - b).rem_euclid(1.0);
    d.min(1.0 - d) <= tol
}

fn heard_time(seen: &[Incoming], day: f32) -> bool {
    clock(seen).is_some_and(|(d, _)| near(d, day, 0.01))
}

fn sleep_until(at: Instant) {
    if let Some(rest) = at.checked_duration_since(Instant::now()) {
        thread::sleep(rest);
    }
}

/// Named connections and everything each has received, for assertions across a session.
struct Room {
    port: u16,
    password: String,
    names: Vec<String>,
    clients: Vec<Connection>,
    seen: Vec<Vec<Incoming>>,
}

impl Room {
    fn new(port: u16) -> Self {
        Self::locked(port, "")
    }

    fn locked(port: u16, password: &str) -> Self {
        Self { port, password: password.into(), names: Vec::new(), clients: Vec::new(), seen: Vec::new() }
    }

    fn join(&mut self, name: &str) {
        let entered = enter(self.port, name, &self.password, &[]);
        self.add(name, entered);
    }

    fn add(&mut self, name: &str, (conn, seen): (Connection, Vec<Incoming>)) {
        self.names.push(name.into());
        self.clients.push(conn);
        self.seen.push(seen);
    }

    fn at(&self, name: &str) -> usize {
        self.names.iter().position(|n| n == name).unwrap_or_else(|| panic!("{name} is not in the room"))
    }

    fn leave(&mut self, name: &str) {
        let at = self.at(name);
        self.names.remove(at);
        self.seen.remove(at);
        drop(self.clients.remove(at));
    }

    fn conn(&mut self, name: &str) -> &mut Connection {
        let at = self.at(name);
        &mut self.clients[at]
    }

    fn seen(&self, name: &str) -> &[Incoming] {
        &self.seen[self.at(name)]
    }

    fn clear(&mut self) {
        self.seen.iter_mut().for_each(Vec::clear);
    }

    /// Poll everyone until `done(name, connection, everything it has seen)` holds for all.
    fn until(&mut self, timeout: Duration, what: &str, mut done: impl FnMut(&str, &Connection, &[Incoming]) -> bool) {
        let deadline = Instant::now() + timeout;
        loop {
            for (c, seen) in self.clients.iter_mut().zip(self.seen.iter_mut()) {
                seen.extend(c.poll());
            }
            let pending: Vec<&str> = (0..self.clients.len())
                .filter(|&i| !done(&self.names[i], &self.clients[i], &self.seen[i]))
                .map(|i| self.names[i].as_str())
                .collect();
            if pending.is_empty() {
                return;
            }
            assert!(Instant::now() < deadline, "{what}: {pending:?} not done after {timeout:?}");
            thread::sleep(Duration::from_millis(5));
        }
    }

    fn hear(&mut self, span: Duration) {
        for (seen, more) in self.seen.iter_mut().zip(listen(&mut self.clients, span)) {
            seen.extend(more);
        }
    }
}

fn ours() -> ContentId {
    crate::net::content_id(&crate::block::BlockRegistry::with_builtins())
}

fn hello(name: &str, password: &str, protocol: u32, content: ContentId, mods: &[&str]) -> ClientMessage {
    ClientMessage::Hello {
        protocol,
        worldgen: content.worldgen,
        gravity: content.gravity,
        law: content.law,
        palette: content.palette,
        name: name.into(),
        password: password.into(),
        mods: mods.iter().map(|id| ModOffer { id: (*id).into(), version: "1.0.0".into() }).collect(),
    }
}

/// Dial, send one crafted `Hello`, and return the server's first reply: the raw handshake
/// [`Connection`] hides, for clients that speak another version or content.
fn raw_reply(port: u16, hello: &ClientMessage) -> ServerMessage {
    let rt = Runtime::new().expect("runtime");
    quic::install_crypto();
    let mut endpoint = {
        let _guard = rt.enter();
        Endpoint::client(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0))).expect("client endpoint")
    };
    endpoint.set_default_client_config(quic::client_config());
    let reply = rt.block_on(async {
        let conn = endpoint
            .connect(SocketAddr::from((Ipv4Addr::LOCALHOST, port)), "watt")
            .expect("dial")
            .await
            .expect("handshake");
        let (mut send, mut recv) = conn.open_bi().await.expect("stream");
        protocol::write_frame_async(&mut send, &hello.encode()).await.expect("hello");
        let mut frame = Vec::new();
        protocol::read_frame_async(&mut recv, &mut frame).await.expect("a reply");
        conn.close(0u32.into(), b"bye");
        ServerMessage::decode(&frame).expect("a server message")
    });
    rt.block_on(async {
        let _ = tokio::time::timeout(Duration::from_secs(1), endpoint.wait_idle()).await;
    });
    reply
}

fn reject_reason(port: u16, hello: &ClientMessage) -> String {
    match raw_reply(port, hello) {
        ServerMessage::Reject { reason } => reason.to_string(),
        other => panic!("expected a Reject, got {other:?}"),
    }
}

fn welcome(law: [u8; material::STAMP_LEN]) -> ServerMessage {
    ServerMessage::Welcome {
        player_id: 1,
        seed: 5,
        spawn: DVec3::new(0.5, 15.0, 0.5),
        worldgen: WorldgenKind::Flat,
        terrain: TerrainCfg::default(),
        law,
    }
}

fn mute_join() -> Vec<ServerMessage> {
    vec![welcome(protocol::law_stamp()), ServerMessage::SnapshotEnd, ServerMessage::Time { day: 0.5, day_secs: 600.0 }]
}

/// A server that admits one client with `opening`, then says only what the test hands it and
/// answers nothing, while its QUIC link stays up: a server that has stopped responding.
struct Mute {
    port: u16,
    say: Option<mpsc::Sender<ServerMessage>>,
    thread: Option<JoinHandle<()>>,
}

impl Mute {
    fn start(opening: Vec<ServerMessage>) -> Self {
        let (port_tx, port_rx) = mpsc::channel();
        let (say, heard) = mpsc::channel::<ServerMessage>();
        let thread = thread::spawn(move || {
            let rt = Runtime::new().expect("runtime");
            let endpoint = {
                let _guard = rt.enter();
                let config = quic::server_config().expect("server cert");
                Endpoint::server(config, SocketAddr::from((Ipv4Addr::LOCALHOST, 0))).expect("binds")
            };
            port_tx.send(endpoint.local_addr().expect("bound").port()).expect("port");
            let admitted = rt.block_on(async {
                let incoming = tokio::time::timeout(Duration::from_secs(10), endpoint.accept()).await.ok()??;
                let conn = incoming.await.ok()?;
                let (send, mut recv) = conn.accept_bi().await.ok()?;
                let mut hello = Vec::new();
                protocol::read_frame_async(&mut recv, &mut hello).await.ok()?;
                Some((conn, send, recv))
            });
            let Some((conn, mut send, _recv)) = admitted else { return };
            for msg in opening.into_iter().chain(heard.iter()) {
                if rt.block_on(protocol::write_frame_async(&mut send, &msg.encode())).is_err() {
                    break;
                }
            }
            conn.close(0u32.into(), b"bye");
            rt.block_on(async {
                let _ = tokio::time::timeout(Duration::from_secs(1), endpoint.wait_idle()).await;
            });
        });
        let port = port_rx.recv().expect("the mute server binds");
        Self { port, say: Some(say), thread: Some(thread) }
    }

    fn say(&self, msg: ServerMessage) {
        self.say.as_ref().expect("open").send(msg).expect("the mute server is up");
    }
}

impl Drop for Mute {
    fn drop(&mut self) {
        drop(self.say.take());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// A UDP relay in front of a server. Cutting it drops every datagram both ways, as a dead
/// network or a hung server host does.
struct Relay {
    port: u16,
    cut: Arc<AtomicBool>,
    done: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Relay {
    fn to(server: u16) -> Self {
        let front = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).expect("relay front");
        let back = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).expect("relay back");
        back.connect((Ipv4Addr::LOCALHOST, server)).expect("relay target");
        front.set_nonblocking(true).expect("nonblocking");
        back.set_nonblocking(true).expect("nonblocking");
        let port = front.local_addr().expect("relay addr").port();
        let cut = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));
        let (cutting, ending) = (cut.clone(), done.clone());
        let thread = thread::spawn(move || {
            let mut buf = vec![0u8; 65_536];
            let mut client = None;
            while !ending.load(Ordering::Relaxed) {
                let mut moved = false;
                while let Ok((n, from)) = front.recv_from(&mut buf) {
                    moved = true;
                    client = Some(from);
                    if !cutting.load(Ordering::Relaxed) {
                        let _ = back.send(&buf[..n]);
                    }
                }
                while let Ok(n) = back.recv(&mut buf) {
                    moved = true;
                    if let (false, Some(to)) = (cutting.load(Ordering::Relaxed), client) {
                        let _ = front.send_to(&buf[..n], to);
                    }
                }
                if !moved {
                    thread::sleep(Duration::from_micros(200));
                }
            }
        });
        Self { port, cut, done, thread: Some(thread) }
    }

    fn cut(&self) {
        self.cut.store(true, Ordering::Relaxed);
    }
}

impl Drop for Relay {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// A directory removed on drop, so a failing test leaves no world files behind.
struct Dir(PathBuf);

impl Dir {
    fn new(tag: &str) -> Self {
        let dir = crate::save::store::test_temp_path(tag);
        fs::create_dir_all(&dir).expect("temp dir");
        Self(dir)
    }

    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

type Cell = (i32, i32, i32);

fn persistent(path: &Path, seed: i64) -> Lobby {
    Lobby::start(Config {
        seed,
        worldgen: WorldgenKind::Flat,
        world: Some(path.to_path_buf()),
        ops: vec!["ada".into()],
        ..Config::default()
    })
}

/// Edit `cell` to `spec` and wait for the verdict: true when accepted.
fn edit(conn: &mut Connection, (x, y, z): Cell, spec: &str) -> bool {
    let req = conn.send_edit(x, y, z, spec.into()).expect("the edit is sent");
    let events = settle(std::slice::from_mut(conn), WAIT, |_, seen| {
        seen.iter().any(|e| matches!(e, Incoming::EditAccepted { req: r } | Incoming::EditRejected { req: r, .. } if *r == req))
    });
    events[0].iter().any(|e| matches!(e, Incoming::EditAccepted { req: r } if *r == req))
}

/// The cell under the spawn and its two neighbours along x, all inside edit reach.
fn ground_cells(conn: &Connection) -> [Cell; 3] {
    let s = conn.spawn();
    let (x, z) = (crate::math::block_coord(s.x), crate::math::block_coord(s.z));
    [0, 1, 2].map(|dx| (x + dx, FLAT_HEIGHT - 1, z))
}

/// Overlay cells a join delivered, as `cell → spec`.
fn overlay(events: &[Incoming]) -> HashMap<Cell, String> {
    events
        .iter()
        .filter_map(|e| match e {
            Incoming::Mutation { x, y, z, spec } => Some(((*x, *y, *z), spec.to_string())),
            _ => None,
        })
        .collect()
}

/// A headless client world for `seed` with `events` applied the way the game applies them.
fn client_world(seed: i64, events: &[Incoming]) -> World {
    let mut world = World::new(seed);
    for e in events {
        if let Incoming::Mutation { x, y, z, spec } | Incoming::Edit { x, y, z, spec } = e {
            let id = crate::save::parse_block(world.registry_mut(), spec);
            world.set_block(*x, *y, *z, id);
        }
    }
    world
}

fn saved_cells(path: &Path) -> Vec<Cell> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    match format::decode(&bytes).expect("a world file") {
        Decoded::Intact(doc) => sorted_cells(doc.edits.iter().map(|e| (e.x, e.y, e.z))),
        Decoded::Salvaged { .. } => panic!("{} is not intact", path.display()),
    }
}

fn sorted_cells(cells: impl IntoIterator<Item = Cell>) -> Vec<Cell> {
    let mut cells: Vec<Cell> = cells.into_iter().collect();
    cells.sort();
    cells
}

/// One server life on `path`: join as ada, check the overlay holds exactly the broken cells
/// `expect`, break cell `then`, and stop. Returns the cells (the same every life: ada is id 1).
fn life(path: &Path, expect: &[usize], then: usize) -> [Cell; 3] {
    let lobby = persistent(path, 7);
    let (mut ada, events) = enter(lobby.port, "ada", "", &[]);
    let cells = ground_cells(&ada);
    let want = sorted_cells(expect.iter().map(|&i| cells[i]));
    assert_eq!(sorted_cells(overlay(&events).into_keys()), want, "the world this life loaded");
    assert!(edit(&mut ada, cells[then], "air"), "break cell {then}");
    drop(ada);
    lobby.server.stop();
    cells
}

#[test]
fn a_joiner_gets_the_world_the_clock_and_everyone_already_here() {
    let lobby = Lobby::start(Config { seed: 31, worldgen: WorldgenKind::Flat, day_secs: 120.0, ..Config::default() });
    let mut room = Room::new(lobby.port);
    room.join("ada");
    {
        let ada = &room.clients[0];
        assert_eq!((ada.seed(), ada.worldgen()), (31, WorldgenKind::Flat));
        assert!(ada.is_alive() && ada.snapshot_ready());
    }
    assert_eq!(clock(room.seen("ada")).map(|(_, secs)| secs), Some(120.0));
    assert!(joined(room.seen("ada")).is_empty(), "nobody else is here yet");
    room.join("bob");
    room.join("cy");
    room.until(WAIT, "full roster", |_, c, _| c.peers().count() == 2);
    assert_eq!(roster(room.conn("ada")), ["bob", "cy"]);
    assert_eq!(roster(room.conn("bob")), ["ada", "cy"]);
    assert_eq!(roster(room.conn("cy")), ["ada", "bob"]);
    let ids: HashSet<u32> = room.clients.iter().map(Connection::player_id).collect();
    assert_eq!(ids.len(), 3, "every player has their own id");
    assert!(!ids.contains(&0), "id 0 is the world's");
    for c in &room.clients {
        assert!(c.peers().all(|p| p.id() != c.player_id()), "nobody is their own peer");
        assert!(c.peers().all(|p| !p.visible()), "a roster entry stays hidden until its first pose");
    }
}

#[test]
fn joins_and_leaves_reach_everyone_else_exactly_once() {
    let lobby = Lobby::flat();
    let mut room = Room::new(lobby.port);
    room.join("ada");
    room.join("bob");
    room.join("cy");
    room.until(WAIT, "full roster", |_, c, _| c.peers().count() == 2);
    room.hear(Duration::from_millis(200));
    assert_eq!(sorted(joined(room.seen("ada"))), ["bob", "cy"]);
    assert_eq!(sorted(joined(room.seen("bob"))), ["ada", "cy"], "ada from the join roster, cy live");
    assert_eq!(sorted(joined(room.seen("cy"))), ["ada", "bob"]);
    room.leave("bob");
    room.until(WAIT, "bob's leave", |_, c, _| c.peers().count() == 1);
    room.hear(Duration::from_millis(200));
    for (name, other) in [("ada", "cy"), ("cy", "ada")] {
        assert_eq!(left(room.seen(name)), ["bob"], "{name}");
        assert_eq!(roster(room.conn(name)), [other]);
        assert!(reasons(room.seen(name)).is_empty() && room.conn(name).is_alive());
    }
}

#[test]
fn a_player_who_leaves_can_rejoin_at_once_under_the_same_name() {
    let lobby = Lobby::flat();
    let mut room = Room::new(lobby.port);
    room.join("bob");
    let mut ids = HashSet::new();
    for round in 0..5 {
        let (ada, _) = enter(lobby.port, "ada", "", &[]);
        assert!(ids.insert(ada.player_id()), "round {round}: ids are not reused");
        room.add("ada", (ada, Vec::new()));
        room.until(WAIT, "ada and bob see each other", |_, c, _| c.peers().count() == 1);
        assert_eq!(roster(room.conn("ada")), ["bob"]);
        room.leave("ada");
    }
    room.until(WAIT, "the last leave", |_, c, _| c.peers().count() == 0);
    room.hear(Duration::from_millis(200));
    let order: Vec<&str> = room
        .seen("bob")
        .iter()
        .filter_map(|e| match e {
            Incoming::Joined { .. } => Some("joined"),
            Incoming::Left { .. } => Some("left"),
            _ => None,
        })
        .collect();
    assert_eq!(order, ["joined", "left"].repeat(5), "bob hears each visit in order");
    assert!(joined(room.seen("bob")).iter().chain(&left(room.seen("bob"))).all(|n| n == "ada"));
}

#[test]
fn a_taken_name_is_refused_whatever_its_case_or_padding() {
    let lobby = Lobby::flat();
    let mut room = Room::new(lobby.port);
    room.join("Ada");
    room.join("bob");
    room.until(WAIT, "roster", |_, c, _| c.peers().count() == 1);
    room.clear();
    for name in ["Ada", "ada", "ADA", "  ada ", "a\u{7}da", "ada\n"] {
        assert_eq!(refused(lobby.port, name, "", &[]).message(), "that name is already in use", "{name:?}");
    }
    room.hear(Duration::from_millis(300));
    for name in ["Ada", "bob"] {
        assert!(joined(room.seen(name)).is_empty() && left(room.seen(name)).is_empty(), "{name} saw a refused join");
    }
    assert_eq!(roster(room.conn("bob")), ["Ada"]);
    assert!(room.clients.iter().all(Connection::is_alive));
}

#[test]
fn racing_joins_under_one_name_admit_exactly_one() {
    let lobby = Lobby::flat();
    let port = lobby.port;
    let results: Vec<Result<Connection, ConnectError>> = thread::scope(|s| {
        let racers: Vec<_> = (0..6).map(|_| s.spawn(move || Connection::connect(HOST, port, "twin", ""))).collect();
        racers.into_iter().map(|r| r.join().expect("racer")).collect()
    });
    let (won, lost): (Vec<_>, Vec<_>) = results.into_iter().partition(Result::is_ok);
    assert_eq!(won.len(), 1, "one twin is admitted");
    for err in lost.into_iter().filter_map(Result::err) {
        assert_eq!(err.message(), "that name is already in use");
    }
    let (late, _) = enter(port, "late", "", &[]);
    let mut late = [late];
    settle(&mut late, WAIT, |c, _| c.peers().count() == 1);
    assert_eq!(roster(&late[0]), ["twin"], "no losing twin lingers in the roster");
}

#[test]
fn names_that_spell_server_are_reserved() {
    let lobby = Lobby::flat();
    for name in ["server", "Server", "SERVER", "s.e.r.v.e.r", " server!", "Ser ver"] {
        assert_eq!(refused(lobby.port, name, "", &[]).message(), "the name 'server' is reserved", "{name:?}");
    }
    let mut room = Room::new(lobby.port);
    room.join("observer");
    room.join("servers");
    room.until(WAIT, "both admitted", |_, c, _| c.peers().count() == 1);
}

#[test]
fn names_are_cleaned_and_capped_before_anyone_sees_them() {
    let lobby = Lobby::flat();
    let mut room = Room::new(lobby.port);
    room.join("bob");
    let long = "abcdefghijklmnopqrstuvwxyz0123";
    let capped = &long[..crate::net::MAX_NAME];
    let _zed = enter(lobby.port, "  zed\u{1b}[31m ", "", &[]);
    let _long = enter(lobby.port, long, "", &[]);
    let _blank = enter(lobby.port, "", "", &[]);
    assert_eq!(
        refused(lobby.port, &format!("{capped}-other"), "", &[]).message(),
        "that name is already in use",
        "names that share the capped prefix collide"
    );
    assert_eq!(refused(lobby.port, "   ", "", &[]).message(), "that name is already in use", "a blank name is 'player'");
    room.until(WAIT, "three joiners", |_, c, _| c.peers().count() == 3);
    assert_eq!(roster(room.conn("bob")), sorted(vec![capped.into(), "player".into(), "zed[31m".into()]));
}

#[test]
fn the_password_gates_the_join() {
    let lobby = flat(Config { password: "hunter2".into(), ..Config::default() });
    let mut room = Room::locked(lobby.port, "hunter2");
    room.join("ada");
    room.clear();
    for password in ["", "hunter3", "HUNTER2"] {
        assert_eq!(refused(lobby.port, "bob", password, &[]).message(), "wrong password", "{password:?}");
    }
    room.hear(Duration::from_millis(300));
    assert!(joined(room.seen("ada")).is_empty() && left(room.seen("ada")).is_empty());
    room.join("bob");
    room.until(WAIT, "bob joins with the password", |_, c, _| c.peers().count() == 1);
}

#[test]
fn another_protocol_version_is_refused_naming_both_versions() {
    let lobby = Lobby::flat();
    for theirs in [PROTOCOL_VERSION - 1, PROTOCOL_VERSION + 1, 0] {
        assert_eq!(
            reject_reason(lobby.port, &hello("ada", "", theirs, ours(), &[])),
            format!("protocol version mismatch: server v{PROTOCOL_VERSION}, client v{theirs}")
        );
    }
    let (ada, events) = enter(lobby.port, "ada", "", &[]);
    assert!(ada.is_alive());
    assert!(joined(&events).is_empty(), "no refused client is in the roster");
}

#[test]
fn different_world_content_is_refused_naming_what_differs() {
    let lobby = Lobby::flat();
    let id = ours();
    let cases = [
        (
            ContentId { worldgen: id.worldgen + 1, ..id },
            format!("generator version differs: server v{}, client v{}", id.worldgen, id.worldgen + 1),
        ),
        (ContentId { gravity: id.gravity ^ 1, ..id }, "gravity law differs".into()),
        (ContentId { law: id.law ^ 1, ..id }, "material law differs".into()),
        (ContentId { palette: id.palette ^ 1, ..id }, "palette differs".into()),
    ];
    for (content, why) in cases {
        assert_eq!(
            reject_reason(lobby.port, &hello("ada", "", PROTOCOL_VERSION, content, &[])),
            format!("world content mismatch: {why}")
        );
    }
    assert!(matches!(raw_reply(lobby.port, &hello("ada", "", PROTOCOL_VERSION, id, &[])), ServerMessage::Welcome { .. }));
}

#[test]
fn version_and_content_come_before_the_password_and_the_password_before_mods() {
    let lobby = flat(Config { password: "pw".into(), mods_deny: vec!["pwc.dev-toolkit".into()], ..Config::default() });
    let port = lobby.port;
    let old = reject_reason(port, &hello("ada", "nope", PROTOCOL_VERSION - 1, ours(), &[]));
    assert!(old.starts_with("protocol version mismatch"), "{old}");
    let drifted = ContentId { palette: ours().palette ^ 1, ..ours() };
    let drifted = reject_reason(port, &hello("ada", "nope", PROTOCOL_VERSION, drifted, &[]));
    assert!(drifted.starts_with("world content mismatch"), "{drifted}");
    let hidden = reject_reason(port, &hello("ada", "nope", PROTOCOL_VERSION, ours(), &["pwc.dev-toolkit"]));
    assert_eq!(hidden, "wrong password", "a stranger learns nothing about the mod policy");
    match raw_reply(port, &hello("ada", "pw", PROTOCOL_VERSION, ours(), &["pwc.dev-toolkit"])) {
        ServerMessage::ModsDenied { ids } => assert_eq!(ids, [Arc::<str>::from("pwc.dev-toolkit")]),
        other => panic!("expected ModsDenied, got {other:?}"),
    }
}

#[test]
fn a_server_running_another_law_is_refused_by_the_client() {
    let mut law = protocol::law_stamp();
    let last = law.len() - 1;
    law[last] ^= 0xff;
    let mute = Mute::start(vec![welcome(law)]);
    let err = refused(mute.port, "ada", "", &[]);
    assert!(err.message().contains("law"), "the refusal names the law: {err}");
    assert!(err.mods_denied.is_empty());
}

#[test]
fn a_denied_mod_is_named_and_the_client_rejoins_once_without_it() {
    let lobby = flat(Config { mods_deny: vec!["pwc.dev-toolkit".into()], ..Config::default() });
    let mut room = Room::new(lobby.port);
    room.join("bob");
    room.clear();
    let mods = offer(&["pwc.dev-toolkit", "pwc.hotbar"]);
    let first = refused(lobby.port, "ada", "", &mods);
    assert_eq!(first.mods_denied, ["pwc.dev-toolkit"]);
    assert_eq!(first.message(), "server refused mods: pwc.dev-toolkit");
    let (retry, denied) = join_like_the_game(lobby.port, "ada", &mods);
    assert_eq!(denied, ["pwc.dev-toolkit"]);
    let ada = retry.unwrap_or_else(|e| panic!("the retry without the toolkit joins: {e}"));
    let mut ada = [ada];
    settle(&mut ada, WAIT, |c, _| c.snapshot_ready() && c.peers().count() == 1);
    room.until(WAIT, "bob sees ada", |_, c, _| c.peers().count() == 1);
    room.hear(Duration::from_millis(200));
    assert_eq!(joined(room.seen("bob")), ["ada"], "two refusals and one join: one announcement");
    assert!(left(room.seen("bob")).is_empty());
}

/// `hello_offers` leaves out an offer whose version is over 64 bytes, and every offer past
/// the 128th, with only a console warning. The server never hears of those mods, so an honest
/// client joins with a denied mod still on.
#[test]
#[ignore = "BUG: a mod the Hello cannot carry (long version, or past 128 offers) is not reported, so a denied mod joins enabled"]
fn a_denied_mod_the_hello_cannot_carry_still_keeps_the_client_out() {
    let lobby = flat(Config { mods_deny: vec!["pwc.dev-toolkit".into()], ..Config::default() });
    let long = vec![("pwc.dev-toolkit".to_string(), format!("1.0.0+{}", "f".repeat(64)))];
    let mut crowded: Vec<(String, String)> = (0..128).map(|i| (format!("pkg.n{i}"), "1.0.0".to_string())).collect();
    crowded.push(("pwc.dev-toolkit".into(), "1.0.0".into()));
    let admitted: Vec<&str> = [("long version", long), ("129th offer", crowded)]
        .into_iter()
        .filter_map(|(case, mods)| match Connection::connect_with(HOST, lobby.port, case, "", &mods) {
            Ok(_) => Some(case),
            Err(e) => {
                assert!(e.message().contains("pwc.dev-toolkit"), "{case}: the refusal names the mod: {e}");
                None
            }
        })
        .collect();
    assert!(admitted.is_empty(), "joined with the denied toolkit enabled: {admitted:?}");
}

#[test]
fn an_allow_list_admits_only_what_it_names() {
    let lobby = flat(Config { mods_allow: vec!["pwc.hotbar".into(), "pwc.sounds".into()], ..Config::default() });
    let port = lobby.port;
    let (listed, _) = enter(port, "ada", "", &offer(&["pwc.hotbar", "pwc.sounds"]));
    let (bare, _) = enter(port, "bob", "", &[]);
    let mods = offer(&["pwc.hotbar", "pwc.dev-toolkit", "x.extra"]);
    let err = refused(port, "cy", "", &mods);
    assert_eq!(err.mods_denied, ["pwc.dev-toolkit", "x.extra"], "the unlisted ones, in the client's order");
    assert_eq!(err.message(), "server refused mods: pwc.dev-toolkit, x.extra");
    let (retry, _) = join_like_the_game(port, "cy", &mods);
    let cy = retry.unwrap_or_else(|e| panic!("cy joins with only the hotbar: {e}"));
    assert_eq!(refused(port, "dee", "", &offer(&["PWC.hotbar"])).mods_denied, ["PWC.hotbar"], "ids are case-sensitive");
    assert!(listed.is_alive() && bare.is_alive() && cy.is_alive());
}

#[test]
fn deny_wins_over_allow_and_a_repeated_id_is_named_once() {
    let lobby = flat(Config {
        mods_allow: vec!["a.one".into(), "a.two".into()],
        mods_deny: vec!["a.two".into()],
        ..Config::default()
    });
    assert_eq!(refused(lobby.port, "ada", "", &offer(&["a.one", "a.two", "a.two"])).mods_denied, ["a.two"]);
    let open = flat(Config { mods_deny: vec!["pwc.Toolkit".into()], ..Config::default() });
    let (ada, _) = enter(open.port, "ada", "", &offer(&["pwc.toolkit"]));
    assert!(ada.is_alive(), "a deny entry matches its exact id only");
}

#[test]
fn a_name_only_operator_sets_the_clock_for_everyone() {
    let lobby = flat(Config { ops: vec![" Ada ".into()], ..Config::default() });
    let mut room = Room::new(lobby.port);
    room.join("ada");
    room.join("bob");
    room.until(WAIT, "roster", |_, c, _| c.peers().count() == 1);
    room.clear();
    room.conn("ada").send_set_time(0.75);
    room.until(WAIT, "the new time", |_, _, seen| heard_time(seen, 0.75));
    assert!(room.seen.iter().all(|seen| clock(seen).is_some_and(|(_, secs)| secs == 600.0)));
    let (_, events) = enter(lobby.port, "cy", "", &[]);
    assert!(heard_time(&events, 0.75), "a later joiner inherits the time: {:?}", clock(&events));
}

#[test]
fn a_guest_cannot_set_the_clock_and_only_they_hear_why() {
    let lobby = flat(Config { ops: vec!["ada".into()], ..Config::default() });
    let mut room = Room::new(lobby.port);
    room.join("ada");
    room.join("bob");
    room.until(WAIT, "roster", |_, c, _| c.peers().count() == 1);
    room.clear();
    room.conn("bob").send_set_time(0.9);
    room.until(WAIT, "the refusal", |name, _, seen| name == "ada" || !chats(seen).is_empty());
    room.hear(Duration::from_millis(300));
    assert_eq!(chats(room.seen("bob")), [("server".to_string(), REFUSED_TIME.to_string())]);
    assert!(chats(room.seen("ada")).is_empty() && times(room.seen("ada")) == 0, "the refusal is private");
    assert!(!heard_time(room.seen("bob"), 0.9));
    let (_, events) = enter(lobby.port, "cy", "", &[]);
    let day = clock(&events).map(|(d, _)| d);
    assert!(day.is_some_and(|d| near(d, FRESH_DAY, 0.02)), "the shared clock did not move: {day:?}");
}

/// The game moves its own sky before it sends `SetTime` (`Game::run_line`), so a refusal that
/// carries only a chat line leaves the guest's sky off the shared clock until the next minute
/// broadcast. A refused teleport is answered with the authoritative `Position`; this is its twin.
#[test]
#[ignore = "BUG: a refused SetTime gets no Time back, so the guest's own sky stays where they set it"]
fn a_refused_set_time_puts_the_guests_clock_back() {
    let lobby = flat(Config { ops: vec!["ada".into()], ..Config::default() });
    let (bob, _) = enter(lobby.port, "bob", "", &[]);
    let mut bob = [bob];
    bob[0].send_set_time(0.9);
    let events = settle(&mut bob, Duration::from_secs(2), |_, seen| told(seen, REFUSED_TIME) && clock(seen).is_some());
    let (day, _) = clock(&events[0]).expect("a Time with the shared clock");
    assert!(near(day, FRESH_DAY, 0.02), "bob is put back on the shared clock, got {day}");
}

/// `SetTime` over its one-per-second budget is dropped without an answer (`charge` returns
/// `Drop`), unlike an edit or a teleport over budget, so the operator's sky, already moved
/// locally, leaves everyone else's.
#[test]
#[ignore = "BUG: an operator's second SetTime within a second is dropped silently, so their sky leaves the shared clock"]
fn a_set_time_over_its_budget_still_gets_an_answer() {
    let lobby = flat(Config { ops: vec!["ada".into()], ..Config::default() });
    let (ada, _) = enter(lobby.port, "ada", "", &[]);
    let mut ada = [ada];
    ada[0].send_set_time(0.2);
    settle(&mut ada, WAIT, |_, seen| heard_time(seen, 0.2));
    thread::sleep(Duration::from_millis(300));
    ada[0].send_set_time(0.6);
    let events = settle(&mut ada, Duration::from_secs(2), |_, seen| clock(seen).is_some());
    let (day, _) = clock(&events[0]).expect("an answer");
    assert!(near(day, 0.6, 0.01) || near(day, 0.2, 0.01), "{day}");
}

#[test]
fn set_time_wraps_into_one_day_and_ignores_nan() {
    let lobby = flat(Config { ops: vec!["ada".into()], ..Config::default() });
    let mut room = Room::new(lobby.port);
    room.join("ada");
    room.join("bob");
    room.until(WAIT, "roster", |_, c, _| c.peers().count() == 1);
    room.clear();
    let nan = Instant::now();
    room.conn("ada").send_set_time(f32::NAN);
    room.hear(Duration::from_millis(300));
    assert!(room.seen.iter().all(|seen| times(seen) == 0), "NaN never reaches the clock");
    sleep_until(nan + Duration::from_millis(1500));
    room.conn("ada").send_set_time(1.25);
    room.until(WAIT, "1.25 wraps to 0.25", |_, _, seen| heard_time(seen, 0.25));
    thread::sleep(SET_TIME_GAP);
    room.conn("ada").send_set_time(-0.25);
    room.until(WAIT, "-0.25 wraps to 0.75", |_, _, seen| heard_time(seen, 0.75));
}

#[test]
fn an_operator_with_a_secret_proves_it_with_op_and_nobody_else_hears() {
    let lobby = flat(Config {
        ops: vec!["ada".into()],
        op_secrets: vec![("ADA".into(), "s3cret".into())],
        ..Config::default()
    });
    let mut room = Room::new(lobby.port);
    for name in ["ada", "bob", "cy"] {
        room.join(name);
    }
    room.until(WAIT, "roster", |_, c, _| c.peers().count() == 2);
    room.clear();
    room.conn("ada").send_set_time(0.6);
    room.until(WAIT, "a listed name alone is not enough", |n, _, seen| n != "ada" || told(seen, REFUSED_TIME));
    let refused_at = Instant::now();
    room.conn("ada").send_chat(chat::GLOBAL, "/op");
    room.conn("ada").send_chat(chat::GLOBAL, "/op wrong");
    room.conn("bob").send_chat(chat::GLOBAL, "/op s3cret");
    room.until(WAIT, "three refusals", |name, _, seen| {
        let refusals = chats(seen).iter().filter(|(_, text)| text == "operator secret refused").count();
        refusals
            == match name {
                "ada" => 2,
                "bob" => 1,
                _ => 0,
            }
    });
    room.conn("ada").send_chat(chat::GLOBAL, "/op s3cret");
    room.until(WAIT, "proved", |n, _, seen| n != "ada" || told(seen, "you are now an operator"));
    sleep_until(refused_at + SET_TIME_GAP);
    room.conn("ada").send_set_time(0.6);
    room.until(WAIT, "the proved operator's time", |_, _, seen| heard_time(seen, 0.6));
    room.hear(Duration::from_millis(200));
    for name in ["ada", "bob", "cy"] {
        for (from, text) in chats(room.seen(name)) {
            assert_eq!(from, "server", "{name} heard {text:?} from {from}");
            assert!(!text.contains("s3cret") && !text.contains("wrong") && !text.contains("/op"), "{name} heard {text:?}");
        }
    }
    assert!(chats(room.seen("cy")).is_empty(), "the answers are private");
}

#[test]
fn operator_status_ends_with_the_session() {
    let lobby = flat(Config { op_secrets: vec![("ada".into(), "s3cret".into())], ..Config::default() });
    let mut room = Room::new(lobby.port);
    room.join("ada");
    room.conn("ada").send_chat(chat::GLOBAL, "/op s3cret");
    room.until(WAIT, "proved", |_, _, seen| told(seen, "you are now an operator"));
    room.leave("ada");
    room.join("ada");
    room.conn("ada").send_set_time(0.4);
    room.until(WAIT, "refused again", |_, _, seen| told(seen, REFUSED_TIME));
    assert!(!heard_time(room.seen("ada"), 0.4));
}

#[test]
fn operators_and_mod_policy_beside_the_world_file_apply() {
    let dir = Dir::new("side-files");
    fs::write(dir.file("ops.txt"), "# operators\nbob\nada s3cret\n").expect("ops.txt");
    fs::write(dir.file("mods.toml"), "deny = [\"pwc.dev-toolkit\"]\n").expect("mods.toml");
    let mut config = Config { seed: 1, worldgen: WorldgenKind::Flat, world: Some(dir.file("world.save")), ..Config::default() };
    server::load_world_policy(&mut config).expect("side files load");
    let lobby = Lobby::start(config);
    assert_eq!(refused(lobby.port, "cy", "", &offer(&["pwc.dev-toolkit"])).mods_denied, ["pwc.dev-toolkit"]);
    let mut room = Room::new(lobby.port);
    room.join("ada");
    room.join("bob");
    room.until(WAIT, "roster", |_, c, _| c.peers().count() == 1);
    room.clear();
    room.conn("bob").send_set_time(0.1);
    room.until(WAIT, "a name-only operator sets the time", |_, _, seen| heard_time(seen, 0.1));
    room.conn("ada").send_set_time(0.5);
    room.until(WAIT, "ada must prove it", |name, _, seen| name == "bob" || told(seen, REFUSED_TIME));
    let refused_at = Instant::now();
    room.conn("ada").send_chat(chat::GLOBAL, "/op s3cret");
    room.until(WAIT, "proved", |name, _, seen| name == "bob" || told(seen, "you are now an operator"));
    sleep_until(refused_at + SET_TIME_GAP);
    room.conn("ada").send_set_time(0.5);
    room.until(WAIT, "the proved operator's time", |_, _, seen| heard_time(seen, 0.5));
}

/// `parse_ops` splits a line at its last space, so a name-only line for a player whose name has
/// a space lists the first word as an operator with the last word as its secret. `--ops "Big Ada"`
/// trusts the whole name.
#[test]
#[ignore = "BUG: an ops.txt line naming a player whose name has a space is read as name + secret, so that player is never an operator"]
fn an_operator_whose_name_has_a_space_can_be_listed_alone_in_ops_txt() {
    let dir = Dir::new("spaced-op");
    fs::write(dir.file("ops.txt"), "Big Ada\n").expect("ops.txt");
    let mut config = Config { seed: 1, worldgen: WorldgenKind::Flat, world: Some(dir.file("world.save")), ..Config::default() };
    server::load_world_policy(&mut config).expect("side files load");
    let lobby = Lobby::start(config);
    let (big_ada, _) = enter(lobby.port, "Big Ada", "", &[]);
    let mut big_ada = [big_ada];
    big_ada[0].send_set_time(0.7);
    let events = settle(&mut big_ada, WAIT, |_, seen| heard_time(seen, 0.7) || told(seen, REFUSED_TIME));
    assert!(heard_time(&events[0], 0.7), "Big Ada is not an operator: {:?}", chats(&events[0]));
}

#[test]
fn the_shared_clock_runs_at_the_servers_day_length() {
    let lobby = flat(Config { day_secs: 10.0, ..Config::default() });
    let (_ada, first) = enter(lobby.port, "ada", "", &[]);
    let at = Instant::now();
    thread::sleep(Duration::from_secs(1));
    let (_bob, second) = enter(lobby.port, "bob", "", &[]);
    let elapsed = at.elapsed().as_secs_f32();
    let (d0, s0) = clock(&first).expect("ada's clock");
    let (d1, s1) = clock(&second).expect("bob's clock");
    assert_eq!((s0, s1), (10.0, 10.0));
    let advance = (d1 - d0).rem_euclid(1.0);
    assert!((advance - elapsed / 10.0).abs() < 0.03, "advanced {advance} over {elapsed}s of a 10 s day");
}

#[test]
fn day_length_is_clamped_to_the_clients_range() {
    for (asked, served) in [(0.0, 10.0), (-5.0, 10.0), (1.0e9, 86_400.0), (f32::NAN, 600.0)] {
        let lobby = flat(Config { day_secs: asked, ..Config::default() });
        let (_conn, events) = enter(lobby.port, "ada", "", &[]);
        assert_eq!(clock(&events).map(|(_, secs)| secs), Some(served), "asked for {asked}");
    }
}

#[test]
#[ignore = "slow: waits out the one-minute clock broadcast"]
fn the_clock_is_rebroadcast_every_minute() {
    let lobby = flat(Config { day_secs: 600.0, ..Config::default() });
    let (ada, events) = enter(lobby.port, "ada", "", &[]);
    let (day, _) = clock(&events).expect("join clock");
    let joined_at = Instant::now();
    let mut ada = [ada];
    let later = settle(&mut ada, Duration::from_secs(65), |_, seen| clock(seen).is_some());
    let (again, secs) = clock(&later[0]).expect("a broadcast");
    let elapsed = joined_at.elapsed().as_secs_f32();
    assert_eq!(secs, 600.0);
    assert!(near(again, day + elapsed / 600.0, 0.005), "the broadcast carries the running clock: {day} then {again}");
}

#[test]
fn stop_tells_every_client_the_server_is_shutting_down() {
    let lobby = Lobby::flat();
    let mut room = Room::new(lobby.port);
    for name in ["ada", "bob", "cy"] {
        room.join(name);
    }
    room.until(WAIT, "roster", |_, c, _| c.peers().count() == 2);
    room.clear();
    lobby.server.stop();
    lobby.server.stop();
    room.until(WAIT, "the shutdown", |_, c, _| !c.is_alive());
    room.hear(Duration::from_millis(300));
    for (name, seen) in room.names.iter().zip(&room.seen) {
        assert_eq!(reasons(seen), ["server shutting down"], "{name}");
        assert_eq!(interruptions(seen), 0, "{name}");
    }
    let ada = room.conn("ada");
    assert!(!ada.link_interrupted());
    assert_eq!(ada.send_edit(0, FLAT_HEIGHT - 1, 0, "air".into()), None, "a dead link sends nothing");
    ada.send_chat(chat::GLOBAL, "anyone?");
    assert!(ada.poll().is_empty(), "the disconnect is reported once");
}

#[test]
fn a_silent_server_raises_interrupted_and_its_next_message_clears_it() {
    let mute = Mute::start(mute_join());
    let (conn, _) = enter(mute.port, "ada", "", &[]);
    let quiet = Instant::now();
    let mut conn = [conn];
    let events = settle(&mut conn, Duration::from_secs(8), |_, seen| interruptions(seen) > 0);
    assert!(quiet.elapsed() >= Duration::from_millis(4500), "warned after {:?}", quiet.elapsed());
    assert_eq!(interruptions(&events[0]), 1);
    assert!(reasons(&events[0]).is_empty());
    assert!(conn[0].is_alive() && conn[0].link_interrupted());
    mute.say(ServerMessage::Time { day: 0.6, day_secs: 600.0 });
    settle(&mut conn, WAIT, |c, _| !c.link_interrupted());
    let after = listen(&mut conn, Duration::from_millis(300));
    assert!(conn[0].is_alive());
    assert_eq!(interruptions(&after[0]), 0);
    assert!(reasons(&after[0]).is_empty());
}

/// The server reads nothing from a joiner until its overlay is queued, so over a slow link
/// the join's snapshot batches are all the client hears, and no Pong comes back until the
/// end. `Connection::poll` holds a `Snapshot` aside without going through `apply`, the one
/// place `last_heard` is refreshed, so the batches never count as proof of life.
#[test]
#[ignore = "BUG: snapshot batches do not refresh the silence watchdog, so a long join warns Interrupted (and gives up at 12 s) while data flows"]
fn snapshot_batches_alone_keep_a_joining_link_alive() {
    let mute = Mute::start(vec![welcome(protocol::law_stamp())]);
    let mut conn = [Connection::connect(HOST, mute.port, "ada", "").expect("the welcome admits ada")];
    let quiet_until = Instant::now() + Duration::from_millis(5600);
    let mut seen = Vec::new();
    let mut batches = 0;
    while Instant::now() < quiet_until {
        batches += 1;
        mute.say(ServerMessage::Snapshot { edits: vec![(batches, FLAT_HEIGHT - 1, 0, 1, "air".into())] });
        seen.extend(listen(&mut conn, Duration::from_millis(250)).remove(0));
    }
    mute.say(ServerMessage::SnapshotEnd);
    seen.extend(settle(&mut conn, WAIT, |c, _| c.snapshot_ready()).remove(0));
    assert_eq!(overlay(&seen).len(), batches as usize, "every batch arrived");
    assert_eq!(interruptions(&seen), 0, "a link that delivered a batch every 250 ms was called silent");
    assert!(conn[0].is_alive() && !conn[0].link_interrupted());
}

#[test]
fn an_idle_player_on_a_live_server_is_never_warned() {
    let lobby = Lobby::flat();
    let (ada, _) = enter(lobby.port, "ada", "", &[]);
    let mut ada = [ada];
    let events = listen(&mut ada, Duration::from_millis(5500));
    assert_eq!(interruptions(&events[0]), 0, "pongs keep the watchdog quiet");
    assert!(reasons(&events[0]).is_empty());
    assert!(ada[0].is_alive() && !ada[0].link_interrupted());
    assert!(ada[0].ping_ms().is_some(), "the server answered a ping");
}

#[test]
#[ignore = "slow: waits out the twelve-second give-up"]
fn a_silent_server_is_given_up_as_interrupted() {
    let mute = Mute::start(mute_join());
    let (conn, _) = enter(mute.port, "ada", "", &[]);
    let quiet = Instant::now();
    let mut conn = [conn];
    let events = settle(&mut conn, Duration::from_secs(16), |c, _| !c.is_alive());
    let events = &events[0];
    assert!(quiet.elapsed() >= Duration::from_millis(11_500), "gave up after {:?}", quiet.elapsed());
    assert_eq!(interruptions(events), 1);
    assert_eq!(reasons(events), [INTERRUPTED]);
    let warned = events.iter().position(|e| matches!(e, Incoming::Interrupted));
    let gone = events.iter().position(|e| matches!(e, Incoming::Disconnected { .. }));
    assert!(warned < gone, "the warning comes first");
    assert!(!conn[0].link_interrupted(), "a given-up link is no longer 'interrupted'");
}

#[test]
#[ignore = "slow: waits out the client's give-up and the server's idle timeout"]
fn a_cut_link_is_given_up_by_the_client_and_reaped_by_the_server() {
    let lobby = Lobby::flat();
    let relay = Relay::to(lobby.port);
    let mut room = Room::new(lobby.port);
    room.join("bob");
    let ada = enter(relay.port, "ada", "", &[]);
    room.add("ada", ada);
    room.until(WAIT, "roster", |_, c, _| c.peers().count() == 1);
    room.clear();
    relay.cut();
    room.until(Duration::from_secs(25), "ada gives up and bob sees her go", |name, c, seen| match name {
        "ada" => !c.is_alive(),
        _ => left(seen) == ["ada"],
    });
    assert_eq!(interruptions(room.seen("ada")), 1);
    assert_eq!(reasons(room.seen("ada")).len(), 1);
    assert!(room.seen("bob").iter().all(|e| !matches!(e, Incoming::Disconnected { .. } | Incoming::Interrupted)));
    room.leave("ada");
    room.join("ada");
    room.until(WAIT, "ada is back", |_, c, _| c.peers().count() == 1);
}

/// The watchdog and quinn's idle timeout both give up 12 s after the last packet. When the
/// reader thread sees quinn's timeout first, `connection_close_text` records quinn's own
/// "timed out" and the watchdog keeps it. Polling every frame only narrows the race (the test
/// above saw both reasons); a client that is not polling when the link dies always loses it.
#[test]
#[ignore = "BUG: a dead link's disconnect reason is quinn's \"timed out\", not \"connection interrupted\", whenever the transport idle timeout is noticed first"]
fn a_dead_link_is_reported_as_interrupted_whichever_timer_notices_first() {
    let lobby = Lobby::flat();
    let relay = Relay::to(lobby.port);
    let (ada, _) = enter(relay.port, "ada", "", &[]);
    let mut ada = [ada];
    relay.cut();
    thread::sleep(Duration::from_secs(13));
    let events = settle(&mut ada, WAIT, |c, _| !c.is_alive());
    assert_eq!(reasons(&events[0]), [INTERRUPTED]);
}

#[test]
fn edits_and_the_clock_survive_a_restart_on_the_same_world_file() {
    let dir = Dir::new("restart");
    let path = dir.file("world.save");
    let lobby = persistent(&path, 7);
    let (mut ada, _) = enter(lobby.port, "ada", "", &[]);
    let [ground, beside, _] = ground_cells(&ada);
    let pillar = (beside.0, FLAT_HEIGHT, beside.2);
    let local = World::new(ada.seed());
    let grass_id = local.block_at(ground.0, ground.1, ground.2);
    assert_ne!(grass_id, AIR);
    let grass = local.registry().spec(grass_id);
    assert!(edit(&mut ada, ground, "air"));
    assert!(edit(&mut ada, pillar, &grass));
    ada.send_set_time(0.2);
    let mut one = [ada];
    settle(&mut one, WAIT, |_, seen| heard_time(seen, 0.2));
    drop(one);
    let port = lobby.port;
    lobby.server.stop();
    assert!(path.exists(), "stop saves the world");

    let config = Config { seed: 99, worldgen: WorldgenKind::Diffusion, world: Some(path.clone()), ..Config::default() };
    let lobby = Lobby { server: server::spawn(port, config).expect("stop frees the port"), port };
    let (mut bob, events) = enter(lobby.port, "bob", "", &[]);
    assert_eq!((bob.seed(), bob.worldgen()), (7, WorldgenKind::Flat), "the file's world wins over the flags");
    assert_eq!(overlay(&events), HashMap::from([(ground, "air".to_string()), (pillar, grass.clone())]));
    let world = client_world(bob.seed(), &events);
    assert_eq!(world.block_at(ground.0, ground.1, ground.2), AIR);
    assert_eq!(world.registry().spec(world.block_at(pillar.0, pillar.1, pillar.2)), grass);
    assert!(heard_time(&events, 0.2), "the clock came back: {:?}", clock(&events));
    assert!(edit(&mut bob, pillar, "air"), "a restored cell takes a new edit");
    assert!(edit(&mut bob, ground, &grass));
}

#[test]
fn a_corrupt_world_file_loads_its_backup_and_the_next_save_keeps_that_backup() {
    let dir = Dir::new("backup");
    let path = dir.file("world.save");
    let backup = dir.file("world.save.bak");
    let cells = life(&path, &[], 0);
    assert!(!backup.exists(), "the first save has nothing to rotate");
    life(&path, &[0], 1);
    assert_eq!(saved_cells(&backup), [cells[0]], "the second save rotated the first to .bak");
    fs::write(&path, b"not a world").expect("corrupt the live file");
    life(&path, &[0], 2);
    assert_eq!(saved_cells(&backup), [cells[0]], "the save after a fallback does not rotate over the backup");
    assert_eq!(saved_cells(&path), sorted_cells([cells[0], cells[2]]));
    life(&path, &[0, 2], 1);
    assert_eq!(saved_cells(&backup), sorted_cells([cells[0], cells[2]]), "a good live file rotates again");
    fs::remove_file(&path).expect("lose the live file");
    life(&path, &[0, 2], 1);
    assert_eq!(saved_cells(&backup), sorted_cells([cells[0], cells[2]]), "a missing live file loads the backup and keeps it");
    assert_eq!(saved_cells(&path), sorted_cells(cells));
}

#[test]
fn a_cell_put_back_to_its_generated_block_is_left_out_of_the_world_file() {
    let dir = Dir::new("natural");
    let path = dir.file("world.save");
    let cells = life(&path, &[], 0);
    let lobby = persistent(&path, 7);
    let (mut ada, events) = enter(lobby.port, "ada", "", &[]);
    assert_eq!(overlay(&events), HashMap::from([(cells[0], "air".to_string())]));
    let local = World::new(ada.seed());
    let (x, y, z) = cells[0];
    let ground = local.registry().spec(local.block_at(x, y, z));
    assert!(edit(&mut ada, cells[0], &ground), "put the ground back");
    drop(ada);
    lobby.server.stop();
    assert!(saved_cells(&path).is_empty(), "a generated block is not an edit");

    let lobby = persistent(&path, 7);
    let (ada, events) = enter(lobby.port, "ada", "", &[]);
    assert!(overlay(&events).is_empty());
    let world = client_world(ada.seed(), &events);
    assert_eq!(world.registry().spec(world.block_at(x, y, z)), ground);
}

#[test]
fn autosave_writes_the_world_while_it_runs() {
    let dir = Dir::new("autosave");
    let path = dir.file("world.save");
    let lobby = Lobby::start(Config {
        seed: 7,
        worldgen: WorldgenKind::Flat,
        world: Some(path.clone()),
        autosave_every: Duration::from_secs(1),
        ..Config::default()
    });
    let (mut ada, _) = enter(lobby.port, "ada", "", &[]);
    let [cell, ..] = ground_cells(&ada);
    assert!(edit(&mut ada, cell, "air"));
    let deadline = Instant::now() + Duration::from_secs(3);
    while !(path.exists() && saved_cells(&path) == [cell]) {
        assert!(Instant::now() < deadline, "no autosave held the edit after 3 s");
        thread::sleep(Duration::from_millis(50));
    }
    assert!(ada.is_alive(), "saving does not disturb the session");
}

#[test]
fn a_corrupt_world_file_without_a_backup_is_refused_and_kept() {
    let dir = Dir::new("corrupt");
    let path = dir.file("world.save");
    fs::write(&path, b"not a world").expect("corrupt file");
    let started = server::spawn(0, Config { seed: 1, worldgen: WorldgenKind::Flat, world: Some(path.clone()), ..Config::default() });
    let err = match started {
        Ok(_) => panic!("a corrupt world was served"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains(&path.display().to_string()), "the error names the file: {err}");
    assert_eq!(fs::read(&path).expect("still there"), b"not a world");
}

#[test]
fn many_joins_and_leaves_leave_no_one_behind() {
    let lobby = Lobby::flat();
    let mut room = Room::new(lobby.port);
    room.join("ada");
    room.join("bob");
    room.until(WAIT, "roster", |_, c, _| c.peers().count() == 1);
    room.clear();
    const ROUNDS: usize = 12;
    for _ in 0..ROUNDS {
        let visitor = enter(lobby.port, "visitor", "", &[]);
        room.until(WAIT, "the visitor arrives", |_, c, _| c.peers().count() == 2);
        drop(visitor);
        room.until(WAIT, "the visitor leaves", |_, c, _| c.peers().count() == 1);
    }
    room.hear(Duration::from_millis(200));
    for name in ["ada", "bob"] {
        assert_eq!(joined(room.seen(name)), vec!["visitor"; ROUNDS], "{name}");
        assert_eq!(left(room.seen(name)), vec!["visitor"; ROUNDS], "{name}");
    }
    let (last, mut events) = enter(lobby.port, "last", "", &[]);
    let mut last = [last];
    events.extend(settle(&mut last, WAIT, |c, _| c.peers().count() == 2).remove(0));
    events.extend(listen(&mut last, Duration::from_millis(200)).remove(0));
    assert_eq!(roster(&last[0]), ["ada", "bob"]);
    assert_eq!(sorted(joined(&events)), ["ada", "bob"], "no visitor is left in the roster");
}

#[test]
fn joiners_who_leave_before_reading_their_join_leave_no_ghost_and_free_their_name() {
    let lobby = Lobby::flat();
    let mut room = Room::new(lobby.port);
    room.join("host");
    room.clear();
    let names: Vec<String> = (0..8).map(|i| format!("v{i}")).collect();
    for name in &names {
        drop(Connection::connect(HOST, lobby.port, name, "").expect("joins"));
    }
    room.join("last");
    room.until(WAIT, "only host and last remain", |name, c, _| {
        roster(c) == [if name == "host" { "last" } else { "host" }]
    });
    room.hear(Duration::from_millis(200));
    let mut came = sorted(joined(room.seen("host")));
    came.retain(|n| n != "last");
    assert!(came.iter().all(|n| names.contains(n)), "{came:?}");
    assert_eq!(came, sorted(left(room.seen("host"))), "every visitor host heard arrive was heard leaving");
    for name in &names {
        let (back, _) = enter(lobby.port, name, "", &[]);
        assert!(back.is_alive(), "{name} is free again");
    }
}

#[test]
fn a_crowd_joining_and_leaving_at_once_settles_to_an_exact_roster() {
    let lobby = Lobby::flat();
    let port = lobby.port;
    let mut room = Room::new(port);
    room.join("host");
    room.clear();
    const CROWD: usize = 10;
    let names: Vec<String> = (0..CROWD).map(|i| format!("p{i}")).collect();
    let mut crowd: Vec<Connection> = thread::scope(|s| {
        let joins: Vec<_> = names.iter().map(|n| s.spawn(move || Connection::connect(HOST, port, n, ""))).collect();
        joins.into_iter().map(|j| j.join().expect("joiner").expect("joins")).collect()
    });
    let mut seen = settle(&mut crowd, WAIT, |c, _| c.snapshot_ready() && c.peers().count() == CROWD);
    for (s, more) in seen.iter_mut().zip(listen(&mut crowd, Duration::from_millis(200))) {
        s.extend(more);
    }
    room.until(WAIT, "the host sees the crowd", |_, c, _| c.peers().count() == CROWD);
    let ids: HashSet<u32> = crowd.iter().map(Connection::player_id).chain([room.clients[0].player_id()]).collect();
    assert_eq!(ids.len(), CROWD + 1);
    for (i, (c, events)) in crowd.iter().zip(&seen).enumerate() {
        let mut others: Vec<String> = names.iter().filter(|n| **n != names[i]).cloned().collect();
        others.push("host".into());
        let others = sorted(others);
        assert_eq!(roster(c), others, "{}", names[i]);
        assert_eq!(sorted(joined(events)), others, "{} hears each arrival once", names[i]);
    }
    thread::scope(|s| {
        for c in crowd {
            s.spawn(move || drop(c));
        }
    });
    room.until(WAIT, "the crowd is gone", |_, c, _| c.peers().count() == 0);
    room.hear(Duration::from_millis(200));
    assert_eq!(sorted(joined(room.seen("host"))), names);
    assert_eq!(sorted(left(room.seen("host"))), names);
    let (last, _) = enter(port, "last", "", &[]);
    let mut last = [last];
    settle(&mut last, WAIT, |c, _| c.peers().count() == 1);
    listen(&mut last, Duration::from_millis(200));
    assert_eq!(roster(&last[0]), ["host"]);
}
