//! Multiplayer presence tests: poses and their quantisation, interest range, the movement envelope
//! against honest and lying clients, teleport policy, chat, swings, mod channels and voice. See
//! `super` for the harness.
//!
//! [`Raw`] speaks the protocol with no throttle, so a test sees every frame the server sends and
//! can forge any move. [`Body`] is the game's own player physics on a flat world like the
//! server's, reporting each frame as `Game::net_phase` does. A frame sent after a request is a
//! sentinel: the server handles one client's frames in order and queues what each causes before
//! it reads the next, so whatever the request caused arrives before the sentinel's answer.

use std::f32::consts::{PI, TAU};
use std::net::{Ipv4Addr, SocketAddr};
use std::ops::Range;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use glam::DQuat;
use quinn::{Endpoint, SendStream};
use tokio::runtime::Runtime;
use voxel_engine::{DVec3, Vec3};

use super::Lobby;
use crate::audio::{ModFrame, ModLink};
use crate::block::BlockRegistry;
use crate::coord::Face;
use crate::input::movement::{self, MoveInput};
use crate::math::PER_METER;
use crate::net::client::{self, Connection, Incoming, RemotePlayer};
use crate::net::protocol::{self, Channel, ClientMessage, ModBytes, POSE_STEP, PeerPose, ServerMessage};
use crate::net::server::{Config, NoclipPolicy, TeleportPolicy};
use crate::net::{MAX_MOD_BYTES, PROTOCOL_VERSION, chat, quic};
use crate::player::{CRUISE_DEFAULT, MAX_SPEED, Motion, Player};
use crate::presence::Stance;
use crate::world::World;
use crate::world::generation::{FLAT_HEIGHT, WorldgenKind};

/// The server's tuning (`server.rs`) in blocks: interest range, the band that hears every
/// tick, the burst a move may spend, and the per-second budgets.
const INTEREST: f64 = 160.0 * PER_METER;
const NEAR: f64 = 48.0 * PER_METER;
const MOVE_FLOOR: f64 = 80.0 * PER_METER * 0.3;
const CHAT_RATE: usize = 5;
const SWING_RATE: usize = 10;
const TELEPORT_RATE: usize = 4;
const CHANNEL_RATE: usize = 100;
const MOD_DATA_RATE: usize = 200;
const MAX_CHANNELS: usize = 32;
/// The client's receive ring per channel (`client.rs`).
const MOD_RING: usize = 64;
/// The most one coordinate may move on the wire: half of [`POSE_STEP`].
const HALF_STEP: f64 = POSE_STEP / 2.0;
/// The most yaw or pitch may move on the wire: half a 16-bit fraction of a turn.
const HALF_TURN_STEP: f32 = TAU / 65536.0 / 2.0;
const WAIT: Duration = Duration::from_secs(3);

/// A protocol-level client: no throttle, no prediction, every server frame kept with its arrival.
struct Raw {
    id: u32,
    spawn: DVec3,
    rt: Arc<Runtime>,
    endpoint: Endpoint,
    conn: quinn::Connection,
    send: SendStream,
    inbox: Receiver<(Instant, ServerMessage)>,
    log: Vec<(Instant, ServerMessage)>,
    nonce: u32,
}

impl Raw {
    /// Join as `name`, then wait for a Pong: the server answers Pings only once the player is live.
    fn join(lobby: &Lobby, name: &str) -> Self {
        let rt = Arc::new(
            tokio::runtime::Builder::new_multi_thread().worker_threads(1).enable_all().build().expect("runtime"),
        );
        quic::install_crypto();
        let mut endpoint = {
            let _guard = rt.enter();
            Endpoint::client(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0))).expect("endpoint")
        };
        endpoint.set_default_client_config(quic::client_config());
        let content = crate::net::content_id(&BlockRegistry::with_builtins());
        let hello = ClientMessage::Hello {
            protocol: PROTOCOL_VERSION,
            worldgen: content.worldgen,
            gravity: content.gravity,
            law: content.law,
            palette: content.palette,
            name: name.into(),
            password: "".into(),
            mods: Vec::new(),
        };
        let server = SocketAddr::from((Ipv4Addr::LOCALHOST, lobby.port));
        let (conn, send, mut recv, welcome) = rt.block_on(async {
            let conn = endpoint.connect(server, "watt").expect("dial").await.expect("handshake");
            let (mut send, mut recv) = conn.open_bi().await.expect("stream");
            protocol::write_frame_async(&mut send, &hello.encode()).await.expect("hello");
            let mut frame = Vec::new();
            protocol::read_frame_async(&mut recv, &mut frame).await.expect("welcome");
            (conn, send, recv, ServerMessage::decode(&frame))
        });
        let Some(ServerMessage::Welcome { player_id, spawn, .. }) = welcome else {
            panic!("{name} was not welcomed: {welcome:?}");
        };
        let (tx, inbox) = mpsc::channel();
        let reader = rt.clone();
        thread::spawn(move || {
            let mut frame = Vec::new();
            while reader.block_on(protocol::read_frame_async(&mut recv, &mut frame)).is_ok() {
                if let Some(msg) = ServerMessage::decode(&frame)
                    && tx.send((Instant::now(), msg)).is_err()
                {
                    break;
                }
            }
        });
        let mut raw = Self { id: player_id, spawn, rt, endpoint, conn, send, inbox, log: Vec::new(), nonce: 0 };
        raw.sync();
        raw
    }

    fn send(&mut self, msg: &ClientMessage) {
        let frame = msg.encode();
        self.rt.block_on(protocol::write_frame_async(&mut self.send, &frame)).expect("send");
    }

    /// A move with a level body: identity frame, up +Y, standing, facing yaw 0.
    fn walk(&mut self, pos: DVec3, velocity: Vec3) {
        self.send(&ClientMessage::Move {
            pos,
            yaw: 0.0,
            pitch: 0.0,
            frame: DQuat::IDENTITY,
            velocity,
            up: Face::PosY,
            stance: Stance::Standing,
        });
    }

    fn mod_data(&mut self, channel: &str, seq: u32, bytes: &[u8]) {
        self.send(&ClientMessage::ModData {
            channel: Channel::parse(channel).expect("a legal name"),
            seq,
            bytes: ModBytes::try_from(bytes.to_vec()).expect("within the cap"),
        });
    }

    fn say(&mut self, text: &str) {
        self.send(&ClientMessage::Chat { channel: chat::GLOBAL, text: text.into() });
    }

    fn pump(&mut self) {
        while let Ok(entry) = self.inbox.try_recv() {
            self.log.push(entry);
        }
    }

    /// The log position now; later waits and counts start here.
    fn mark(&mut self) -> usize {
        self.pump();
        self.log.len()
    }

    /// Index of the first message from `mark` on that `want` accepts.
    fn wait(&mut self, mark: usize, what: &str, want: impl Fn(&ServerMessage) -> bool) -> usize {
        let deadline = Instant::now() + WAIT;
        loop {
            self.pump();
            if let Some(i) = self.log[mark..].iter().position(|(_, m)| want(m)) {
                return mark + i;
            }
            assert!(Instant::now() < deadline, "player {} never got {what}", self.id);
            thread::sleep(Duration::from_millis(2));
        }
    }

    /// Round-trip a Ping and return the log length at the Pong: everything the server queued for
    /// this player before reading the Ping is logged by then. A Ping over the server's budget is
    /// dropped, so an unanswered one is sent again.
    fn sync(&mut self) -> usize {
        let deadline = Instant::now() + WAIT;
        loop {
            self.nonce += 1;
            let nonce = self.nonce;
            let mark = self.mark();
            self.send(&ClientMessage::Ping { nonce });
            let resend = Instant::now() + Duration::from_millis(300);
            while Instant::now() < resend {
                self.pump();
                if let Some(i) = self.log[mark..]
                    .iter()
                    .position(|(_, m)| matches!(m, ServerMessage::Pong { nonce: n } if *n == nonce))
                {
                    return mark + i + 1;
                }
                thread::sleep(Duration::from_millis(2));
            }
            assert!(Instant::now() < deadline, "player {} got no pong", self.id);
        }
    }

    /// The answer to a `Teleport` to `pos`: the pose the server holds afterwards.
    fn teleport(&mut self, pos: DVec3) -> DVec3 {
        let mark = self.mark();
        self.send(&ClientMessage::Teleport { pos });
        let i = self.wait(mark, "a teleport answer", is_position);
        position(&self.log[i].1)
    }

    /// Where the server holds this player: the snap-back from a move past the world border.
    fn held(&mut self) -> DVec3 {
        let mark = self.mark();
        self.walk(DVec3::new(2.0e9, 0.0, 0.0), Vec3::ZERO);
        let i = self.wait(mark, "a snap-back", is_position);
        position(&self.log[i].1)
    }

    /// Poses of player `id` logged from `mark` to `end`: arrival, the frame's origin, the record.
    fn poses(&mut self, mark: usize, end: usize, id: u32) -> Vec<(Instant, DVec3, PeerPose)> {
        self.pump();
        let mut out = Vec::new();
        for (at, msg) in &self.log[mark..end] {
            if let ServerMessage::PeerPoses { poses } = msg {
                out.extend(poses.list.iter().filter(|p| p.id == id).map(|p| (*at, poses.origin, *p)));
            }
        }
        out
    }

    /// How many messages from `mark` to `end` `want` accepts.
    fn count(&mut self, mark: usize, end: usize, want: impl Fn(&ServerMessage) -> bool) -> usize {
        self.pump();
        self.log[mark..end].iter().filter(|(_, m)| want(m)).count()
    }

    /// Mod frames logged from `mark` to `end`: channel, sender, seq.
    fn mod_frames(&mut self, mark: usize, end: usize) -> Vec<(String, u32, u32)> {
        self.pump();
        self.log[mark..end]
            .iter()
            .filter_map(|(_, m)| match m {
                ServerMessage::PeerModData { channel, sender, seq, .. } => {
                    Some((channel.as_str().to_string(), *sender, *seq))
                }
                _ => None,
            })
            .collect()
    }

    /// Wait for the global line `text`; the log index just past it.
    fn heard(&mut self, mark: usize, text: &str) -> usize {
        self.wait(mark, text, |m| matches!(m, ServerMessage::Chat { text: t, .. } if t.as_ref() == text)) + 1
    }
}

impl Drop for Raw {
    fn drop(&mut self) {
        client::graceful_close(&self.conn, &self.endpoint, &self.rt);
    }
}

fn position(msg: &ServerMessage) -> DVec3 {
    match msg {
        ServerMessage::Position { pos, .. } => *pos,
        other => panic!("expected Position, got {other:?}"),
    }
}

fn is_position(msg: &ServerMessage) -> bool {
    matches!(msg, ServerMessage::Position { .. })
}

fn has_pose(id: u32) -> impl Fn(&ServerMessage) -> bool {
    move |m| matches!(m, ServerMessage::PeerPoses { poses } if poses.list.iter().any(|p| p.id == id))
}

fn exited(id: u32) -> impl Fn(&ServerMessage) -> bool {
    move |m| matches!(m, ServerMessage::PeerExited { id: gone } if *gone == id)
}

/// The game's player physics on a flat world like the server's. Each frame reports the way
/// `Game::net_phase` does (cruise, then the move) and then steps by the time since the last
/// frame, at most `max_dt`. A server `Position` snaps it as the game does and is kept as a
/// correction.
struct Body {
    player: Player,
    world: World,
    last: Instant,
    /// The longest step. A sixtieth of a second by default: a slow frame then moves the body less
    /// than the time that passed, which an honest client may always do.
    max_dt: f32,
    /// The position the last report carried.
    reported: DVec3,
    travelled: f64,
    corrections: Vec<DVec3>,
}

impl Body {
    fn at(pos: DVec3) -> Self {
        let mut world = World::new(1);
        world.ensure_around(pos);
        Self {
            player: Player::new(pos),
            world,
            last: Instant::now(),
            max_dt: 1.0 / 60.0,
            reported: pos,
            travelled: 0.0,
            corrections: Vec::new(),
        }
    }

    /// Apply server corrections, then report: the start of `Game::net_phase`.
    fn report(&mut self, conn: &mut Connection) {
        for event in conn.poll() {
            if let Incoming::Position { pos, frame, up } = event {
                self.corrections.push(pos);
                self.player.position = pos;
                self.player.orientation.frame = frame;
                self.player.up_axis = up;
                self.player.cancel_fall();
            }
        }
        conn.sync_cruise(self.player.cruise.map(|c| c.speed));
        let p = &self.player;
        conn.send_move(
            p.position,
            p.orientation.yaw,
            p.orientation.pitch,
            p.orientation.frame,
            p.velocity().as_vec3(),
            p.up_axis,
            Stance::of_player(p),
        );
        self.reported = p.position;
    }

    /// One physics step over the time since the last.
    fn step(&mut self, input: &MoveInput) {
        let now = Instant::now();
        let dt = now.duration_since(self.last).as_secs_f32().min(self.max_dt);
        self.last = now;
        if !self.player.flying() {
            self.world.ensure_around(self.player.position);
        }
        let gravity =
            if self.player.cruising() { DVec3::ZERO } else { self.world.gravity_at(self.player.position).accel };
        let before = self.player.position;
        movement::update_player(&mut self.player, &self.world, input, dt, gravity);
        self.travelled += self.player.position.distance(before);
    }

    fn frame(&mut self, conn: &mut Connection, input: &MoveInput) {
        self.report(conn);
        self.step(input);
    }

    /// `input` at about 60 frames a second for `span`.
    fn hold(&mut self, conn: &mut Connection, input: MoveInput, span: Duration) {
        let end = Instant::now() + span;
        while Instant::now() < end {
            self.frame(conn, &input);
            thread::sleep(Duration::from_millis(16));
        }
    }

    /// `/tp` as the game sends it, arriving at rest.
    fn teleport(&mut self, conn: &mut Connection, pos: DVec3) {
        teleport(conn, pos);
        self.player.position = pos;
        self.player.motion = match self.player.motion {
            Motion::Flying { noclip, .. } => Motion::Flying { velocity: DVec3::ZERO, noclip },
            Motion::Walking { .. } => Motion::Walking { velocity: DVec3::ZERO, on_ground: false },
        };
        self.last = Instant::now();
    }

    /// A last report after a pause long enough that the client's throttle lets it through.
    fn settle(&mut self, conn: &mut Connection) -> DVec3 {
        thread::sleep(Duration::from_millis(40));
        self.report(conn);
        self.reported
    }
}

fn idle() -> MoveInput {
    MoveInput::keys(0.0, 0.0, false, false, false)
}

fn forward() -> MoveInput {
    MoveInput::keys(0.0, 1.0, false, false, false)
}

/// A flat server like the client's [`World::new`], with `config`'s policies.
fn lobby(config: Config) -> Lobby {
    Lobby::start(Config { seed: 1, worldgen: WorldgenKind::Flat, ..config })
}

/// Join and wait for a Pong, so the server already treats the player as live.
fn join(lobby: &Lobby, name: &str) -> Connection {
    let mut conn = lobby.join(name);
    poll_until(&mut conn, "a pong", |c, _| c.ping_ms().is_some());
    conn
}

/// Poll `conn` until `done` holds for it and its events so far.
fn poll_until(
    conn: &mut Connection,
    what: &str,
    mut done: impl FnMut(&Connection, &[Incoming]) -> bool,
) -> Vec<Incoming> {
    let deadline = Instant::now() + WAIT;
    let mut events = Vec::new();
    loop {
        events.extend(conn.poll());
        if done(conn, &events) {
            return events;
        }
        assert!(Instant::now() < deadline, "player {} never saw {what}", conn.player_id());
        thread::sleep(Duration::from_millis(2));
    }
}

fn until(conn: &mut Connection, what: &str, want: impl Fn(&Incoming) -> bool) -> Vec<Incoming> {
    poll_until(conn, what, |_, events| events.iter().any(&want))
}

/// Poll until the global line `text` arrives; everything polled up to it.
fn hear(conn: &mut Connection, text: &str) -> Vec<Incoming> {
    until(conn, text, |e| matches!(e, Incoming::Chat { text: t, .. } if t.as_ref() == text))
}

/// Each client reports standing at its spawn until every one sees every other.
fn meet(conns: &mut [Connection]) {
    let others = conns.len() - 1;
    let deadline = Instant::now() + WAIT;
    loop {
        for c in conns.iter_mut() {
            let at = c.spawn();
            c.send_move(at, 0.0, 0.0, DQuat::IDENTITY, Vec3::ZERO, Face::PosY, Stance::Standing);
            c.poll();
        }
        if conns.iter().all(|c| c.peers().filter(|p| p.visible()).count() == others) {
            return;
        }
        assert!(Instant::now() < deadline, "the players never met");
        thread::sleep(Duration::from_millis(5));
    }
}

/// `/tp` and its echo.
fn teleport(conn: &mut Connection, pos: DVec3) -> Vec<Incoming> {
    conn.send_teleport(pos);
    until(conn, "its teleport echo", |e| matches!(e, Incoming::Position { pos: p, .. } if *p == pos))
}

fn peer(conn: &Connection, id: u32) -> &RemotePlayer {
    conn.peers().find(|p| p.id() == id).expect("a known peer")
}

/// The latest pose `conn` holds for peer `id`, past any interpolation.
fn latest(conn: &Connection, id: u32) -> DVec3 {
    peer(conn, id).sample(Instant::now() + Duration::from_secs(3600)).pos.0
}

/// An eye standing on the flat ground.
fn ground(x: f64, z: f64) -> DVec3 {
    DVec3::new(x, FLAT_HEIGHT as f64 + Stance::Standing.eye_offset(), z)
}

fn chats(events: &[Incoming]) -> Vec<(String, u8, String)> {
    events
        .iter()
        .filter_map(|e| match e {
            Incoming::Chat { from_name, channel, text } => Some((from_name.to_string(), *channel, text.to_string())),
            _ => None,
        })
        .collect()
}

fn line(from: &str, channel: u8, text: &str) -> (String, u8, String) {
    (from.to_string(), channel, text.to_string())
}

fn snaps(events: &[Incoming]) -> Vec<DVec3> {
    events
        .iter()
        .filter_map(|e| match e {
            Incoming::Position { pos, .. } => Some(*pos),
            _ => None,
        })
        .collect()
}

fn within_step(got: DVec3, want: DVec3) -> bool {
    (got - want).abs().max_element() <= HALF_STEP + 1e-9
}

/// Angles compared around the circle.
fn turn_gap(a: f32, b: f32) -> f32 {
    let d = (a - b).rem_euclid(TAU);
    d.min(TAU - d)
}

/// Each move to a visible peer goes out on the next pose tick, inside the documented
/// quantisation: 1/128 block per axis from the recipient's own position, yaw and pitch to a
/// 16-bit fraction of a turn, the frame as smallest-three, the velocity as f16. A peer that then
/// stands still is not sent again.
#[test]
fn a_move_reaches_a_near_peer_on_the_next_tick_within_the_quantisation_bound() {
    let lobby = Lobby::flat();
    let mut ada = Raw::join(&lobby, "ada");
    let mut bob = Raw::join(&lobby, "bob");
    let mark = bob.mark();
    let start = ada.spawn;
    ada.walk(start, Vec3::ZERO);
    bob.wait(mark, "ada entering range", has_pose(ada.id));
    thread::sleep(Duration::from_millis(120));
    let ups = [Face::PosY, Face::NegX, Face::PosZ, Face::NegY, Face::PosX, Face::NegZ];
    let mut latency = Vec::new();
    for i in 1..=10u32 {
        let k = f64::from(i);
        let pos = start + DVec3::new(0.0123 * k + 0.3, 0.0071 * k, -0.0191 * k);
        let yaw = -6.0 + 1.3 * i as f32;
        let pitch = -1.4 + 0.28 * i as f32;
        let frame = if i == 3 {
            DQuat::IDENTITY
        } else {
            DQuat::from_axis_angle(glam::DVec3::new(1.0, k, -0.5).normalize(), 0.4 * k)
        };
        let velocity = if i % 4 == 0 { Vec3::ZERO } else { Vec3::new(1.5, -0.25 * i as f32, 700.0 / i as f32) };
        let up = ups[i as usize % ups.len()];
        let stance = if i % 2 == 1 { Stance::Sneaking } else { Stance::Standing };
        let mark = bob.mark();
        let sent = Instant::now();
        ada.send(&ClientMessage::Move { pos, yaw, pitch, frame, velocity, up, stance });
        let at = bob.wait(mark, "ada's move", |m| matches!(m, ServerMessage::PeerPoses { .. }));
        let poses = bob.poses(mark, at + 1, ada.id);
        assert_eq!(poses.len(), 1, "move {i}: the next frame carries ada once");
        let (arrived, origin, got) = poses[0];
        latency.push(arrived.duration_since(sent));
        assert_eq!(origin, bob.spawn, "move {i}: the origin is bob's own position");
        assert!(within_step(got.pos, pos), "move {i}: {:?} is off by {:?}", got.pos, got.pos - pos);
        assert!(turn_gap(got.yaw, yaw) <= HALF_TURN_STEP + 1e-6, "move {i}: yaw {} for {yaw}", got.yaw);
        assert!((got.pitch - pitch).abs() <= HALF_TURN_STEP + 1e-6, "move {i}: pitch {} for {pitch}", got.pitch);
        assert!(got.frame.dot(frame).abs() >= 0.999, "move {i}: frame {:?} for {frame:?}", got.frame);
        for (g, w) in [(got.velocity.x, velocity.x), (got.velocity.y, velocity.y), (got.velocity.z, velocity.z)] {
            assert!((g - w).abs() <= w.abs() / 1024.0 + 1e-7, "move {i}: velocity {g} for {w}");
        }
        assert_eq!((got.up, got.stance), (up, stance), "move {i}");
        thread::sleep(Duration::from_millis(20));
    }
    let mark = bob.mark();
    thread::sleep(Duration::from_millis(250));
    let end = bob.mark();
    assert!(bob.poses(mark, end, ada.id).is_empty(), "a still peer is not sent again");
    latency.sort();
    assert!(latency[latency.len() / 2] < Duration::from_millis(80), "a move waits for the next tick only: {latency:?}");
    assert!(latency[latency.len() - 1] < Duration::from_secs(1), "{latency:?}");
}

/// Far peers (past 48 m) are sent every fourth tick while near ones go every tick, a frame names
/// a peer once, and a far peer's last pose still arrives after it stops.
#[test]
fn far_peers_are_sent_every_fourth_tick_and_still_peers_not_at_all() {
    let lobby = Lobby::flat();
    let mut ada = Raw::join(&lobby, "ada");
    let mut carol = Raw::join(&lobby, "carol");
    let mut bob = Raw::join(&lobby, "bob");
    let home = ada.spawn;
    let near_at = home + DVec3::new(3.0, 0.0, 0.0);
    let far_at = home + DVec3::new(NEAR + 40.0, 0.0, 0.0);
    assert!(far_at.distance(home) < INTEREST);
    let mark = ada.mark();
    assert_eq!(carol.teleport(near_at), near_at);
    assert_eq!(bob.teleport(far_at), far_at);
    ada.wait(mark, "carol in range", has_pose(carol.id));
    ada.wait(mark, "bob in range", has_pose(bob.id));
    thread::sleep(Duration::from_millis(250));

    let mark = ada.mark();
    let (mut near_pos, mut far_pos) = (near_at, far_at);
    for i in 0..36 {
        let side = if i % 2 == 0 { 0.25 } else { -0.25 };
        near_pos = near_at + DVec3::new(0.0, 0.0, side);
        far_pos = far_at + DVec3::new(0.0, 0.0, side);
        carol.walk(near_pos, Vec3::new(0.0, 0.0, 20.0));
        bob.walk(far_pos, Vec3::new(0.0, 0.0, 20.0));
        thread::sleep(Duration::from_millis(25));
    }
    thread::sleep(Duration::from_millis(400));
    let end = ada.mark();
    let near = ada.poses(mark, end, carol.id);
    let far = ada.poses(mark, end, bob.id);
    for (_, msg) in &ada.log[mark..end] {
        if let ServerMessage::PeerPoses { poses } = msg {
            let mut ids: Vec<u32> = poses.list.iter().map(|p| p.id).collect();
            ids.sort();
            ids.dedup();
            assert_eq!(ids.len(), poses.list.len(), "a frame names each peer once");
            assert_eq!(poses.origin, home, "the origin is ada's own position");
        }
    }
    assert!(near.len() >= 12, "a near peer every tick: {}", near.len());
    let quarter = near.len() as f64 / 4.0;
    assert!(
        far.len() >= 2 && (far.len() as f64 - quarter).abs() <= 2.5,
        "a far peer every fourth tick: {} far, {} near",
        far.len(),
        near.len()
    );
    assert!(within_step(near.last().expect("near").2.pos, near_pos), "carol's last pose");
    assert!(within_step(far.last().expect("far").2.pos, far_pos), "bob's last pose arrives after he stops");

    let mark = ada.mark();
    thread::sleep(Duration::from_millis(300));
    let end = ada.mark();
    assert_eq!(ada.count(mark, end, |m| matches!(m, ServerMessage::PeerPoses { .. })), 0, "still peers are not sent");
}

/// Next to the world border the offset from the recipient still lands within the step.
#[test]
fn poses_near_the_world_border_stay_within_the_step() {
    let lobby = Lobby::flat();
    let mut ada = Raw::join(&lobby, "ada");
    let mut bob = Raw::join(&lobby, "bob");
    let far = DVec3::new(987_654_321.123_456, 40.0, -765_432_109.876_543);
    assert_eq!(ada.teleport(far), far);
    let mark = bob.mark();
    let watch = far + DVec3::new(2.0, 0.0, 1.0);
    assert_eq!(bob.teleport(watch), watch);
    bob.wait(mark, "ada in range", has_pose(ada.id));
    thread::sleep(Duration::from_millis(120));
    for step in [DVec3::new(0.123_456_789, 0.010_1, -0.987_654_321), DVec3::new(-15.007, 3.3, 9.996)] {
        let pos = far + step;
        let mark = bob.mark();
        ada.walk(pos, Vec3::new(1.0, 0.0, 0.0));
        let at = bob.wait(mark, "ada's move", |m| matches!(m, ServerMessage::PeerPoses { .. }));
        let &(_, origin, got) = bob.poses(mark, at + 1, ada.id).last().expect("ada");
        assert_eq!(origin, watch);
        assert!(within_step(got.pos, pos), "{:?} is off by {:?}", got.pos, got.pos - pos);
        thread::sleep(Duration::from_millis(60));
    }
}

/// Interest range reaches exactly 160 m: a quarter block inside is shown, with its pose inside
/// the step even that far from the origin, and a quarter block outside is hidden from both sides.
#[test]
fn the_interest_edge_is_inclusive() {
    let lobby = Lobby::flat();
    let mut ada = Raw::join(&lobby, "ada");
    let mut bob = Raw::join(&lobby, "bob");
    let home = ada.spawn;
    let inside = home + DVec3::new(INTEREST - 0.25, 0.0, 0.0);
    let outside = home + DVec3::new(INTEREST + 0.25, 0.0, 0.0);
    for _ in 0..2 {
        let mark = ada.mark();
        assert_eq!(bob.teleport(inside), inside);
        let at = ada.wait(mark, "bob just inside range", has_pose(bob.id));
        let &(_, origin, got) = ada.poses(mark, at + 1, bob.id).last().expect("bob");
        assert_eq!(origin, home);
        assert!(within_step(got.pos, inside), "{:?} is off by {:?}", got.pos, got.pos - inside);
        let (mark_a, mark_b) = (ada.mark(), bob.mark());
        assert_eq!(bob.teleport(outside), outside);
        ada.wait(mark_a, "bob just outside range", exited(bob.id));
        bob.wait(mark_b, "ada just outside range", exited(ada.id));
    }
}

/// Teleporting out of range hides both sides at once. Coming back shows both again at once, even
/// the side that never moved, and the returning avatar snaps to its new pose instead of sliding
/// across the gap.
#[test]
fn leaving_range_hides_both_sides_and_returning_snaps_both_back() {
    let lobby = Lobby::flat();
    let mut players = vec![join(&lobby, "ada"), join(&lobby, "bob")];
    meet(&mut players);
    let [ada, bob] = &mut players[..] else { unreachable!() };
    let (ada_id, bob_id) = (ada.player_id(), bob.player_id());
    let gone = ada.spawn() + DVec3::new(INTEREST + 200.0, 0.0, 0.0);
    teleport(bob, gone);
    poll_until(ada, "bob leaving range", |c, _| !peer(c, bob_id).visible());
    poll_until(bob, "ada leaving range", |c, _| !peer(c, ada_id).visible());

    let back = ada.spawn() + DVec3::new(5.0, 0.0, 1.0);
    teleport(bob, back);
    poll_until(ada, "bob back in range", |c, _| peer(c, bob_id).visible());
    poll_until(bob, "ada back in range", |c, _| peer(c, ada_id).visible());
    let shown = peer(ada, bob_id).sample(Instant::now()).pos.0;
    assert!(within_step(shown, back), "bob snaps in at {shown:?}, not on a slide from {gone:?}");
    assert!(within_step(latest(bob, ada_id), ada.spawn()), "ada, who never moved, shows where she stands");
}

/// An observer who flies out of range and back sees a peer who never moved again at once, that
/// peer sees the observer, and the honest flight is never corrected.
#[test]
fn an_observer_flying_away_and_back_sees_a_still_peer_again() {
    let lobby = Lobby::flat();
    let mut ada = Raw::join(&lobby, "ada");
    let mut bob = Raw::join(&lobby, "bob");
    let (mark_a, mark_b) = (ada.mark(), bob.mark());
    let mut at = ada.spawn;
    ada.walk(at, Vec3::ZERO);
    ada.wait(mark_a, "bob in range", has_pose(bob.id));
    // A standing start: the time spent standing is what the first fast move may cover.
    thread::sleep(Duration::from_millis(300));
    let speed = 1500.0;
    for out in [1.0, -1.0] {
        for _ in 0..8 {
            at.x += out * speed * 0.04;
            ada.walk(at, Vec3::new((out * speed) as f32, 0.0, 0.0));
            thread::sleep(Duration::from_millis(40));
        }
    }
    let end_a = ada.sync();
    assert_eq!(ada.count(mark_a, end_a, is_position), 0, "an honest flight is not corrected");
    let left = ada.wait(mark_a, "bob leaving range", exited(bob.id));
    ada.wait(left, "bob again", has_pose(bob.id));
    let poses = ada.poses(left, ada.log.len(), bob.id);
    assert!(within_step(poses[0].2.pos, bob.spawn), "bob shows where he stood all along");
    let left = bob.wait(mark_b, "ada leaving range", exited(ada.id));
    bob.wait(left, "ada again", has_pose(ada.id));
}

/// A late joiner sees the players already standing there after its first report, they see it,
/// and when it leaves they drop it.
#[test]
fn a_late_joiner_sees_the_room_and_its_leaving_is_seen() {
    let lobby = Lobby::flat();
    let mut players = vec![join(&lobby, "ada"), join(&lobby, "bob")];
    meet(&mut players);
    players.push(join(&lobby, "carol"));
    meet(&mut players);
    let carol = players.pop().expect("carol");
    let carol_id = carol.player_id();
    drop(carol);
    for p in &mut players {
        until(p, "carol leaving", |e| matches!(e, Incoming::Left { name } if name.as_ref() == "carol"));
        assert!(p.peers().all(|q| q.id() != carol_id), "carol is dropped from the roster");
    }
}

/// The game's own physics walking, sprinting, jumping, turning round, sneaking and strafing on a
/// server that checks every body against the ground is never corrected, and a watcher sees where
/// the walker really ends up.
#[test]
fn an_honest_walker_is_never_corrected() {
    let lobby = lobby(Config { noclip: NoclipPolicy::Off, ..Config::default() });
    let mut players = vec![join(&lobby, "ada"), join(&lobby, "bob")];
    meet(&mut players);
    let [ada, bob] = &mut players[..] else { unreachable!() };
    let ada_id = ada.player_id();
    let mut body = Body::at(ada.spawn());
    let started = Instant::now();
    while !body.player.on_ground() {
        assert!(started.elapsed() < WAIT, "still falling from the spawn height at {:?}", body.player.position);
        body.hold(ada, idle(), Duration::from_millis(50));
    }
    body.hold(ada, forward(), Duration::from_millis(500));
    body.hold(ada, MoveInput::keys(0.0, 1.0, false, true, false), Duration::from_millis(400));
    body.hold(ada, MoveInput::keys(0.0, 1.0, true, true, false), Duration::from_millis(100));
    body.hold(ada, forward(), Duration::from_millis(400));
    body.player.orientation.yaw += PI;
    body.hold(ada, forward(), Duration::from_millis(400));
    body.hold(ada, MoveInput::keys(0.0, 1.0, false, false, true), Duration::from_millis(400));
    body.hold(ada, MoveInput::keys(1.0, 0.0, false, false, false), Duration::from_millis(300));
    body.hold(ada, idle(), Duration::from_millis(300));
    let last = body.settle(ada);
    assert!(body.corrections.is_empty(), "an honest walk was corrected to {:?}", body.corrections);
    assert!(body.travelled > 8.0, "the walk covered ground: {}", body.travelled);
    poll_until(bob, "ada's last pose", |c, _| within_step(latest(c, ada_id), last));
}

/// A fall from forty blocks under the game's gravity is never corrected and lands on the ground.
#[test]
fn an_honest_fall_is_never_corrected() {
    let lobby = lobby(Config { noclip: NoclipPolicy::Off, ..Config::default() });
    let mut ada = join(&lobby, "ada");
    let floor = ground(ada.spawn().x, ada.spawn().z);
    let mut body = Body::at(ada.spawn());
    body.teleport(&mut ada, floor + DVec3::new(0.0, 40.0, 0.0));
    let started = Instant::now();
    while !body.player.on_ground() {
        assert!(started.elapsed() < Duration::from_secs(4), "still falling at {:?}", body.player.position);
        body.frame(&mut ada, &idle());
        thread::sleep(Duration::from_millis(16));
    }
    body.hold(&mut ada, idle(), Duration::from_millis(200));
    body.settle(&mut ada);
    assert!(body.corrections.is_empty(), "an honest fall was corrected to {:?}", body.corrections);
    assert!((body.player.position.y - floor.y).abs() < 0.01, "landed at {:?}", body.player.position);
}

/// Flight at a kilometre a second, there and back, past every collision check, is never corrected.
#[test]
fn an_honest_flight_at_a_kilometre_a_second_is_never_corrected() {
    let lobby = lobby(Config { noclip: NoclipPolicy::Off, ..Config::default() });
    let (mut ada, mut body) = airborne(&lobby);
    body.player.fly_speed = 1000.0 * PER_METER;
    body.hold(&mut ada, forward(), Duration::from_millis(800));
    body.player.orientation.yaw += PI;
    body.hold(&mut ada, forward(), Duration::from_millis(600));
    body.hold(&mut ada, idle(), Duration::from_millis(200));
    body.settle(&mut ada);
    assert!(body.corrections.is_empty(), "an honest flight was corrected to {:?}", body.corrections);
    assert!(body.travelled > 500.0, "the flight covered {}", body.travelled);
}

/// Flight at 20 km/s, under the server's speed cap; the corrections it drew.
fn fly_fast(noclip: NoclipPolicy) -> Vec<DVec3> {
    let lobby = lobby(Config { noclip, ..Config::default() });
    let (mut ada, mut body) = airborne(&lobby);
    body.player.fly_speed = 20_000.0 * PER_METER;
    body.hold(&mut ada, forward(), Duration::from_millis(1200));
    body.hold(&mut ada, idle(), Duration::from_millis(200));
    body.settle(&mut ada);
    assert!(body.travelled > 5_000.0, "the flight covered {}", body.travelled);
    body.corrections
}

/// Where noclip is open, fast flight is judged by the speed envelope alone, and passes.
#[test]
fn an_honest_flight_at_twenty_km_s_passes_the_speed_envelope() {
    let corrections = fly_fast(NoclipPolicy::All);
    assert!(corrections.is_empty(), "an honest flight was corrected to {corrections:?}");
}

/// The same flight where bodies are checked against the ground, through open sky. A report at
/// 20 km/s covers about 780 blocks, past the 512-block sweep.
#[test]
#[ignore = "BUG: a non-cruise move longer than SWEEP_LIMIT fails closed, so flight above ~13 km/s is snapped back every report"]
fn an_honest_flight_at_twenty_km_s_passes_the_collision_check() {
    let corrections = fly_fast(NoclipPolicy::Off);
    assert!(corrections.is_empty(), "an honest flight was corrected {} times", corrections.len());
}

/// Ada joined and flying at rest in open sky above her spawn.
fn airborne(lobby: &Lobby) -> (Connection, Body) {
    let mut ada = join(lobby, "ada");
    let sky = ada.spawn() + DVec3::new(0.0, 25.0, 0.0);
    let mut body = Body::at(ada.spawn());
    body.player.set_flying(true);
    body.teleport(&mut ada, sky);
    (ada, body)
}

/// A cruise in open sky, held and then sped up fourfold, is never corrected.
#[test]
fn an_honest_cruise_through_open_sky_is_never_corrected() {
    let lobby = lobby(Config { noclip: NoclipPolicy::Off, ..Config::default() });
    let (mut ada, mut body) = airborne(&lobby);
    body.player.start_cruise(CRUISE_DEFAULT);
    body.hold(&mut ada, forward(), Duration::from_millis(600));
    body.player.start_cruise(CRUISE_DEFAULT * 4.0);
    body.hold(&mut ada, forward(), Duration::from_millis(300));
    body.settle(&mut ada);
    assert!(body.corrections.is_empty(), "an honest cruise was corrected to {:?}", body.corrections);
    assert!(body.travelled > 1.0e7, "the cruise covered {}", body.travelled);
}

/// A cruise slowed tenfold eases down to its new speed over a few frames, as the game's flight
/// does. Noclip is open, so only the speed envelope judges it.
#[test]
fn slowing_a_cruise_is_never_corrected() {
    let lobby = Lobby::flat();
    let (mut ada, mut body) = airborne(&lobby);
    body.player.start_cruise(CRUISE_DEFAULT);
    body.hold(&mut ada, forward(), Duration::from_millis(600));
    body.player.start_cruise(CRUISE_DEFAULT / 10.0);
    body.hold(&mut ada, forward(), Duration::from_millis(500));
    body.settle(&mut ada);
    assert!(body.corrections.is_empty(), "slowing a cruise was corrected to {:?}", body.corrections);
}

/// The player ends a cruise in a frame whose move the client's throttle held back (about every
/// other frame at 60 Hz). `Game::net_phase` then sends the end of the cruise before the move that
/// still carries the last cruise step. Noclip is open, so only the speed envelope judges it.
#[test]
fn ending_a_cruise_is_never_corrected() {
    let lobby = Lobby::flat();
    let (mut ada, mut body) = airborne(&lobby);
    body.player.start_cruise(CRUISE_DEFAULT);
    body.hold(&mut ada, forward(), Duration::from_millis(600));
    assert!(body.corrections.is_empty(), "the cruise itself was corrected to {:?}", body.corrections);
    // A frame whose move goes out: the last one was at least 40 ms ago. Its step is a cruise step.
    thread::sleep(Duration::from_millis(40));
    body.frame(&mut ada, &forward());
    // The next frame at once: the throttle holds its move back, and the cruise ends in it.
    thread::sleep(Duration::from_millis(2));
    body.report(&mut ada);
    assert!(body.player.end_cruise());
    body.step(&forward());
    thread::sleep(Duration::from_millis(40));
    body.hold(&mut ada, idle(), Duration::from_millis(300));
    body.settle(&mut ada);
    assert!(body.corrections.is_empty(), "ending a cruise was corrected back to {:?}", body.corrections);
}

/// Flight at 4 km/s reached at 20 frames a second goes on when the frame rate rises to 60. Each
/// report then covers the long step before it while the server credits only the short gap, a
/// shortfall of a few tens of milliseconds at speed. Noclip is open, so only the speed envelope
/// judges it.
#[test]
fn fast_flight_survives_a_rising_frame_rate() {
    let lobby = Lobby::flat();
    let (mut ada, mut body) = airborne(&lobby);
    body.max_dt = 0.1;
    body.player.fly_speed = 4000.0 * PER_METER;
    let slow = Instant::now() + Duration::from_secs(1);
    while Instant::now() < slow {
        body.frame(&mut ada, &forward());
        thread::sleep(Duration::from_millis(50));
    }
    assert!(body.corrections.is_empty(), "flight at 20 frames a second was corrected to {:?}", body.corrections);
    body.hold(&mut ada, forward(), Duration::from_millis(500));
    body.settle(&mut ada);
    assert!(body.corrections.is_empty(), "a rising frame rate was corrected to {:?}", body.corrections);
}

/// With noclip for operators only, a guest's jump through a wall snaps back, and so does every
/// small step into it; the same move beside the wall passes, and an operator walks through.
#[test]
fn a_jump_through_a_wall_is_snapped_back_unless_noclip_allows_it() {
    let lobby = lobby(Config { noclip: NoclipPolicy::Ops, ops: vec!["ada".into()], ..Config::default() });
    let mut ada = Raw::join(&lobby, "ada");
    let mut bob = Raw::join(&lobby, "bob");
    let (a0, b0) = (ground(ada.spawn.x, ada.spawn.z), ground(bob.spawn.x, bob.spawn.z));
    ada.walk(a0, Vec3::ZERO);
    bob.walk(b0, Vec3::ZERO);
    let mut registry = BlockRegistry::with_builtins();
    let rock = crate::world::terrain::Materials::intern(&mut registry).rock[0];
    let spec: Arc<str> = registry.spec(rock).into();
    // One block thick at x = 2, three high on the ground, across both paths along +x.
    let wall: Vec<(i32, i32, i32)> = (FLAT_HEIGHT..FLAT_HEIGHT + 3).flat_map(|y| (-5..=0).map(move |z| (2, y, z))).collect();
    let mark = bob.mark();
    for (req, &(x, y, z)) in wall.iter().enumerate() {
        bob.send(&ClientMessage::Edit { req: req as u32, x, y, z, expect: 0, spec: spec.clone() });
    }
    let end = bob.sync();
    let built = bob.count(mark, end, |m| matches!(m, ServerMessage::EditAck { accepted: true, .. }));
    assert_eq!(built, wall.len(), "the wall stands");
    thread::sleep(Duration::from_millis(150));
    let end = bob.mark();
    for (_, msg) in &bob.log[mark..end] {
        if let ServerMessage::Snapshot { edits } = msg {
            assert!(edits.iter().all(|&(x, y, z, rev, ref content)| !wall.contains(&(x, y, z)) || (rev == 1 && content == &spec)), "the wall reacted: {edits:?}");
        }
    }

    let mark = bob.mark();
    bob.walk(DVec3::new(5.5, b0.y, b0.z), Vec3::new(7.0, 0.0, 0.0));
    let i = bob.wait(mark, "a snap-back", is_position);
    assert_eq!(position(&bob.log[i].1), b0, "a jump through the wall snaps back");

    let mark = bob.mark();
    let mut x = b0.x;
    while x < 5.5 {
        x += 0.4;
        bob.walk(DVec3::new(x, b0.y, b0.z), Vec3::new(12.0, 0.0, 0.0));
        thread::sleep(Duration::from_millis(40));
    }
    let end = bob.sync();
    assert!(bob.count(mark, end, is_position) > 0, "stepping into the wall is corrected");
    let held = bob.held();
    assert!(held.x <= 2.0 - 0.3 + 1e-9, "never past the wall's face: held at {held:?}");

    let mark = bob.mark();
    bob.walk(DVec3::new(b0.x, b0.y, 3.5), Vec3::new(0.0, 0.0, 9.0));
    thread::sleep(Duration::from_millis(40));
    bob.walk(DVec3::new(5.5, b0.y, 3.5), Vec3::new(7.0, 0.0, 0.0));
    let end = bob.sync();
    assert_eq!(bob.count(mark, end, is_position), 0, "the same move beside the wall passes");

    let mark = ada.mark();
    ada.walk(DVec3::new(5.5, a0.y, a0.z), Vec3::new(7.0, 0.0, 0.0));
    let end = ada.sync();
    assert_eq!(ada.count(mark, end, is_position), 0, "an operator walks through");
}

/// Under a 10 m/s cap, a client that splits a long move into many small ones, restarting from
/// each snap-back, claiming the cap and then a declared cruise at a huge speed, covers no more
/// than one burst plus the cap over the time it took.
#[test]
fn split_moves_gain_nothing_over_the_speed_cap() {
    let cap = 10.0 * PER_METER;
    let before = Instant::now();
    let lobby = lobby(Config { max_speed: cap, ..Config::default() });
    let mut liar = Raw::join(&lobby, "liar");
    let start = liar.spawn;
    let mut x = start.x;
    let mut snapped = 0;
    let mut mark = liar.mark();
    for i in 0..60 {
        if i == 40 {
            liar.send(&ClientMessage::Cruise { speed: crate::player::CRUISE_MAX });
        }
        let claimed = if i < 40 { cap as f32 } else { 1.0e9 };
        x += 2.0;
        liar.walk(DVec3::new(x, start.y, start.z), Vec3::new(claimed, 0.0, 0.0));
        thread::sleep(Duration::from_millis(34));
        let now = liar.mark();
        for (_, msg) in &liar.log[mark..now] {
            if let ServerMessage::Position { pos, .. } = msg {
                x = pos.x;
                snapped += 1;
            }
        }
        mark = now;
    }
    let gained = liar.held().x - start.x;
    let bound = MOVE_FLOOR + cap * before.elapsed().as_secs_f64();
    assert!(gained <= bound, "split moves covered {gained} blocks, the bound is {bound}");
    assert!(gained >= MOVE_FLOOR * 0.5, "the burst is spendable: {gained}");
    assert!(snapped >= 10, "the lie was corrected only {snapped} times");
}

/// Moves no envelope allows snap back, non-finite ones are dropped unanswered, and none of them
/// reaches a peer.
#[test]
fn forged_moves_are_refused_and_never_reach_a_peer() {
    let lobby = Lobby::flat();
    let mut liar = Raw::join(&lobby, "liar");
    let mut bob = Raw::join(&lobby, "bob");
    let home = liar.spawn;
    let mark = bob.mark();
    liar.walk(home, Vec3::ZERO);
    bob.wait(mark, "the liar in range", has_pose(liar.id));
    thread::sleep(Duration::from_millis(150));
    let (mark, mark_b) = (liar.mark(), bob.mark());
    let top = Vec3::new(MAX_SPEED as f32, 0.0, 0.0);
    liar.walk(home + DVec3::new(1.0e6, 0.0, 0.0), top);
    liar.walk(DVec3::new(1.5e9, home.y, home.z), Vec3::ZERO);
    liar.walk(DVec3::new(f64::NAN, home.y, home.z), Vec3::ZERO);
    liar.walk(home + DVec3::new(1.0, 0.0, 0.0), Vec3::new(f32::INFINITY, 0.0, 0.0));
    liar.send(&ClientMessage::Move {
        pos: home + DVec3::new(1.0, 0.0, 0.0),
        yaw: f32::NAN,
        pitch: 0.0,
        frame: DQuat::IDENTITY,
        velocity: Vec3::ZERO,
        up: Face::PosY,
        stance: Stance::Standing,
    });
    let end = liar.sync();
    let answers: Vec<DVec3> =
        liar.log[mark..end].iter().filter(|(_, m)| is_position(m)).map(|(_, m)| position(m)).collect();
    assert_eq!(answers, vec![home, home], "the jump and the border move snap back; non-finite moves are dropped");
    thread::sleep(Duration::from_millis(150));
    let end_b = bob.mark();
    assert!(bob.poses(mark_b, end_b, liar.id).is_empty(), "no forged pose reaches a peer");
}

/// The default policy lets anyone teleport: the echo lands at the target, and peers see the
/// player leave range and arrive.
#[test]
fn teleport_policy_all_commits_and_echoes() {
    let lobby = Lobby::flat();
    let mut players = vec![join(&lobby, "ada"), join(&lobby, "bob")];
    meet(&mut players);
    let [ada, bob] = &mut players[..] else { unreachable!() };
    let ada_id = ada.player_id();
    let away = ada.spawn() + DVec3::new(300.0, 7.0, -40.0);
    let events = teleport(ada, away);
    assert!(chats(&events).is_empty(), "no refusal: {:?}", chats(&events));
    poll_until(bob, "ada leaving range", |c, _| !peer(c, ada_id).visible());
    let beside = bob.spawn() + DVec3::new(2.0, 0.0, 2.0);
    teleport(ada, beside);
    poll_until(bob, "ada arriving", |c, _| peer(c, ada_id).visible() && within_step(latest(c, ada_id), beside));
}

/// Teleport off refuses everyone, an operator too, with a reason only the asker hears and a
/// snap-back to where the server holds them; nothing reaches the peers.
#[test]
fn teleport_policy_off_refuses_everyone_with_a_reason() {
    let lobby = lobby(Config { teleport: TeleportPolicy::Off, ops: vec!["ada".into()], ..Config::default() });
    let mut players = vec![join(&lobby, "ada"), join(&lobby, "bob")];
    meet(&mut players);
    let [ada, bob] = &mut players[..] else { unreachable!() };
    let ada_id = ada.player_id();
    let home = ada.spawn();
    ada.send_teleport(home + DVec3::new(300.0, 0.0, 0.0));
    let events = hear(ada, "teleport is not permitted");
    assert_eq!(snaps(&events), vec![home]);
    assert_eq!(chats(&events), vec![line("server", chat::GLOBAL, "teleport is not permitted")]);
    ada.send_chat(chat::GLOBAL, "still here");
    let events = hear(bob, "still here");
    assert_eq!(chats(&events), vec![line("ada", chat::GLOBAL, "still here")], "the refusal is private");
    assert!(peer(bob, ada_id).visible(), "a refused teleport moves nobody out of range");
}

/// Teleport for operators admits a listed name in any case and a player who proved their secret
/// with `/op`; a guest, and the secret holder before proving it, are refused with a reason.
#[test]
fn teleport_policy_ops_admits_listed_and_proven_operators_only() {
    let lobby = lobby(Config {
        teleport: TeleportPolicy::Ops,
        ops: vec!["Ada".into()],
        op_secrets: vec![("cy".into(), "s3cret".into())],
        ..Config::default()
    });
    let mut ada = join(&lobby, "ada");
    let mut bob = join(&lobby, "bob");
    let mut cy = join(&lobby, "cy");
    let refused = |conn: &mut Connection| {
        let home = conn.spawn();
        conn.send_teleport(home + DVec3::new(30.0, 5.0, 0.0));
        let events = hear(conn, "only an operator can teleport");
        assert_eq!(snaps(&events), vec![home], "refused back to where the server holds them");
    };
    let to = ada.spawn() + DVec3::new(30.0, 5.0, 0.0);
    assert!(chats(&teleport(&mut ada, to)).is_empty(), "a listed operator teleports");
    refused(&mut bob);
    refused(&mut cy);
    cy.send_chat(chat::GLOBAL, "/op wrong");
    hear(&mut cy, "operator secret refused");
    refused(&mut cy);
    cy.send_chat(chat::GLOBAL, "/op s3cret");
    hear(&mut cy, "you are now an operator");
    let to = cy.spawn() + DVec3::new(30.0, 5.0, 0.0);
    assert!(chats(&teleport(&mut cy, to)).is_empty(), "a proven operator teleports");
}

/// Every teleport gets a Position, even past the per-second budget, so a `/tp` never hangs: the
/// budget's worth commit, the rest and a jump past the world border snap to where the player is,
/// with no reason given.
#[test]
fn every_teleport_is_answered_even_over_the_budget() {
    let lobby = Lobby::flat();
    let mut ada = Raw::join(&lobby, "ada");
    let home = ada.spawn;
    let targets: Vec<DVec3> = (1..=7).map(|i| home + DVec3::new(10.0 * f64::from(i), 0.0, 0.0)).collect();
    let mark = ada.mark();
    ada.send(&ClientMessage::Teleport { pos: DVec3::new(0.0, 0.0, 2.0e9) });
    for &pos in &targets {
        ada.send(&ClientMessage::Teleport { pos });
    }
    let end = ada.sync();
    let answers: Vec<DVec3> =
        ada.log[mark..end].iter().filter(|(_, m)| is_position(m)).map(|(_, m)| position(m)).collect();
    let committed = TELEPORT_RATE - 1;
    let mut want = vec![home];
    want.extend_from_slice(&targets[..committed]);
    want.resize(targets.len() + 1, targets[committed - 1]);
    assert_eq!(answers, want);
    assert_eq!(ada.count(mark, end, |m| matches!(m, ServerMessage::Chat { .. })), 0, "no reason is given");
}

/// Global chat reaches every player, the speaker too. Local chat, and a channel byte the server
/// does not know, reach only players within 48 m of the speaker, inclusive, and arrive as local.
#[test]
fn local_chat_reaches_only_players_within_its_radius() {
    let lobby = Lobby::flat();
    let mut players: Vec<Connection> = ["ada", "bob", "cy", "dan"].iter().map(|name| join(&lobby, name)).collect();
    let home = players[0].spawn();
    let spots = [
        home + DVec3::new(chat::RADIUS - 0.25, 0.0, 0.0),
        home + DVec3::new(chat::RADIUS + 0.25, 0.0, 0.0),
        home + DVec3::new(5000.0, 0.0, 0.0),
    ];
    for (p, at) in players[1..].iter_mut().zip(spots) {
        teleport(p, at);
    }
    players[0].send_chat(chat::LOCAL, "near");
    players[0].send_chat(7, "odd channel");
    players[0].send_chat(chat::GLOBAL, "everyone");
    for (i, p) in players.iter_mut().enumerate() {
        let lines = chats(&hear(p, "everyone"));
        let mut want = Vec::new();
        if i < 2 {
            want.push(line("ada", chat::LOCAL, "near"));
            want.push(line("ada", chat::LOCAL, "odd channel"));
        }
        want.push(line("ada", chat::GLOBAL, "everyone"));
        assert_eq!(lines, want, "player {i}");
    }
}

/// Chat past five lines a second is dropped, and the budget refills.
#[test]
fn chat_past_the_rate_budget_is_dropped() {
    let lobby = Lobby::flat();
    let mut ada = join(&lobby, "ada");
    let mut bob = join(&lobby, "bob");
    for i in 0..8 {
        ada.send_chat(chat::GLOBAL, &format!("line {i}"));
    }
    thread::sleep(Duration::from_millis(1050));
    ada.send_chat(chat::GLOBAL, "after");
    let lines: Vec<String> = chats(&hear(&mut bob, "after")).into_iter().map(|(_, _, text)| text).collect();
    let mut want: Vec<String> = (0..CHAT_RATE).map(|i| format!("line {i}")).collect();
    want.push("after".into());
    assert_eq!(lines, want);
}

/// A swing reaches the players who can see the swinger, not the swinger, not a player out of range.
#[test]
fn swings_reach_only_players_who_can_see_the_swinger() {
    let lobby = Lobby::flat();
    let mut players = vec![join(&lobby, "ada"), join(&lobby, "bob"), join(&lobby, "cy")];
    meet(&mut players);
    let away = players[2].spawn() + DVec3::new(-(INTEREST + 300.0), 0.0, 0.0);
    teleport(&mut players[2], away);
    let ada_id = players[0].player_id();
    players[0].send_swing();
    players[0].send_chat(chat::GLOBAL, "swung");
    for (i, p) in players.iter_mut().enumerate() {
        let events = hear(p, "swung");
        let swings = events.iter().filter(|e| matches!(e, Incoming::PeerSwing { id } if *id == ada_id)).count();
        assert_eq!(swings, usize::from(i == 1), "player {i}");
    }
}

/// Swings past ten a second are dropped.
#[test]
fn swings_past_the_rate_budget_are_dropped() {
    let lobby = Lobby::flat();
    let mut players = vec![join(&lobby, "ada"), join(&lobby, "bob")];
    meet(&mut players);
    let [ada, bob] = &mut players[..] else { unreachable!() };
    for _ in 0..15 {
        ada.send_swing();
    }
    ada.send_chat(chat::GLOBAL, "done");
    let swings = hear(bob, "done").iter().filter(|e| matches!(e, Incoming::PeerSwing { .. })).count();
    assert_eq!(swings, SWING_RATE);
}

/// Channel bytes reach the sender's visible set unchanged, stamped with the sender's id, and no
/// one else; illegal names and oversized payloads are refused before they leave the client.
#[test]
fn mod_data_reaches_only_the_visible_set_with_the_sender_stamped() {
    let lobby = Lobby::flat();
    let mut players = vec![join(&lobby, "ada"), join(&lobby, "bob"), join(&lobby, "cy")];
    meet(&mut players);
    let away = players[2].spawn() + DVec3::new(-(INTEREST + 300.0), 0.0, 0.0);
    teleport(&mut players[2], away);
    let ada = &mut players[0];
    let ada_id = ada.player_id();
    assert!(!ada.send_channel("", 0, b"x"));
    assert!(!ada.send_channel("seventeen-letters", 0, b"x"));
    assert!(!ada.send_channel("chan", 0, &[0; MAX_MOD_BYTES + 1]));
    let full: Vec<u8> = (0..MAX_MOD_BYTES).map(|i| i as u8).collect();
    assert!(ada.send_channel("chan", 7, b"hello"));
    assert!(ada.send_channel("chan", 8, &full));
    ada.send_chat(chat::GLOBAL, "sent");
    for (i, p) in players.iter_mut().enumerate() {
        hear(p, "sent");
        let got = p.drain_channel("chan");
        let want = if i == 1 { vec![(ada_id, 7, b"hello".to_vec()), (ada_id, 8, full.clone())] } else { Vec::new() };
        assert_eq!(got, want, "player {i}");
    }
}

/// Two players who see each other, the first having just reported.
fn pair(lobby: &Lobby) -> (Raw, Raw) {
    let mut ada = Raw::join(lobby, "ada");
    let mut bob = Raw::join(lobby, "bob");
    let mark = bob.mark();
    let at = ada.spawn;
    ada.walk(at, Vec3::ZERO);
    bob.wait(mark, "ada in range", has_pose(ada.id));
    (ada, bob)
}

/// One channel passes a hundred frames a second, in order, and drops the rest.
#[test]
fn one_channel_is_held_to_its_budget() {
    let lobby = Lobby::flat();
    let (mut ada, mut bob) = pair(&lobby);
    let mark = bob.mark();
    for seq in 0..150 {
        ada.mod_data("a", seq, &[seq as u8]);
    }
    ada.say("done");
    let end = bob.heard(mark, "done");
    let want: Vec<(String, u32, u32)> = (0..CHANNEL_RATE as u32).map(|seq| ("a".to_string(), ada.id, seq)).collect();
    assert_eq!(bob.mod_frames(mark, end), want);
}

/// All of a connection's channels share two hundred frames a second.
#[test]
fn all_channels_share_one_budget() {
    let lobby = Lobby::flat();
    let (mut ada, mut bob) = pair(&lobby);
    let mark = bob.mark();
    for seq in 0..80 {
        for name in ["a", "b", "c"] {
            ada.mod_data(name, seq, &[]);
        }
    }
    ada.say("done");
    let end = bob.heard(mark, "done");
    let frames = bob.mod_frames(mark, end);
    assert_eq!(frames.len(), MOD_DATA_RATE, "three channels within their own windows still share one");
    for name in ["a", "b", "c"] {
        let seqs: Vec<u32> = frames.iter().filter(|f| f.0 == name).map(|f| f.2).collect();
        assert_eq!(seqs, (0..seqs.len() as u32).collect::<Vec<_>>(), "channel {name} keeps its order");
    }
}

/// A connection may name 32 channels; a 33rd name is dropped while the known ones still pass.
#[test]
fn a_connection_names_at_most_thirty_two_channels() {
    let lobby = Lobby::flat();
    let (mut ada, mut bob) = pair(&lobby);
    let mark = bob.mark();
    for n in 0..=MAX_CHANNELS {
        ada.mod_data(&format!("n{n}"), 0, &[]);
    }
    ada.mod_data("n0", 1, &[]);
    ada.say("done");
    let end = bob.heard(mark, "done");
    let mut want: Vec<(String, u32, u32)> = (0..MAX_CHANNELS).map(|n| (format!("n{n}"), ada.id, 0)).collect();
    want.push(("n0".to_string(), ada.id, 1));
    assert_eq!(bob.mod_frames(mark, end), want);
}

/// A channel nobody drains keeps its newest 64 frames in order, and its overflow spares the others.
#[test]
fn a_full_channel_ring_keeps_the_newest_frames() {
    let lobby = Lobby::flat();
    let mut ada = Raw::join(&lobby, "ada");
    let mut bob = join(&lobby, "bob");
    let ada_id = ada.id;
    let at = ada.spawn;
    ada.walk(at, Vec3::ZERO);
    poll_until(&mut bob, "ada in range", |c, _| c.peers().any(|p| p.id() == ada_id && p.visible()));
    for seq in 0..CHANNEL_RATE as u32 {
        ada.mod_data("ring", seq, &seq.to_le_bytes());
    }
    ada.mod_data("other", 0, b"kept");
    ada.say("done");
    hear(&mut bob, "done");
    let ring = bob.drain_channel("ring");
    let first = (CHANNEL_RATE - MOD_RING) as u32;
    let want: Vec<(u32, u32, Vec<u8>)> = (first..CHANNEL_RATE as u32).map(|s| (ada_id, s, s.to_le_bytes().to_vec())).collect();
    assert_eq!(ring, want);
    assert_eq!(bob.drain_channel("other"), vec![(ada_id, 0, b"kept".to_vec())]);
    assert!(!bob.channel_pending("ring"), "drained");
}

/// A 20 ms opus-sized frame, numbered.
fn opus(seq: u32) -> Vec<u8> {
    (0..120).map(|i| (seq as u8).wrapping_mul(31).wrapping_add(i)).collect()
}

/// Speak `seqs` on the voice channel at the capture cadence, then say `after`.
fn speak(conn: &mut Connection, seqs: Range<u32>, after: &str) {
    for seq in seqs {
        assert!(ModLink::new(Some(&mut *conn)).send("voice", seq, &opus(seq)));
        thread::sleep(Duration::from_millis(20));
    }
    conn.send_chat(chat::GLOBAL, after);
}

/// The voice frames that arrived before the line `after`.
fn listen_voice(conn: &mut Connection, after: &str) -> Vec<(u32, u32)> {
    hear(conn, after);
    let mut frames: Vec<ModFrame> = Vec::new();
    ModLink::new(Some(conn)).drain("voice", &mut frames);
    frames
        .into_iter()
        .map(|f| {
            assert_eq!(f.bytes, opus(f.seq), "frame {} arrives intact", f.seq);
            (f.sender, f.seq)
        })
        .collect()
}

/// Voice frames through the mod link reach a listener in range, in order and stamped with the
/// speaker, never one out of range or the speaker; they stop when the listener walks off and
/// resume when the listener comes back.
#[test]
fn voice_follows_the_listener_out_of_range_and_back() {
    let lobby = Lobby::flat();
    let mut players = vec![join(&lobby, "ada"), join(&lobby, "bob"), join(&lobby, "cy")];
    meet(&mut players);
    let [ada, bob, cy] = &mut players[..] else { unreachable!() };
    let ada_id = ada.player_id();
    let away = cy.spawn() + DVec3::new(-(INTEREST + 300.0), 0.0, 0.0);
    teleport(cy, away);
    speak(ada, 0..25, "one");
    assert_eq!(listen_voice(bob, "one"), (0..25).map(|s| (ada_id, s)).collect::<Vec<_>>());
    assert!(listen_voice(cy, "one").is_empty(), "out of range");
    assert!(listen_voice(ada, "one").is_empty(), "the speaker does not hear herself");

    let home = bob.spawn();
    teleport(bob, home + DVec3::new(INTEREST + 300.0, 0.0, 0.0));
    speak(ada, 25..35, "two");
    assert!(listen_voice(bob, "two").is_empty(), "a listener who walked off hears nothing");
    teleport(bob, home);
    speak(ada, 35..45, "three");
    assert_eq!(listen_voice(bob, "three"), (35..45).map(|s| (ada_id, s)).collect::<Vec<_>>());
}

/// An observer draws a peer who turns round across ±π turning the short way, and moving along
/// the segment between the two poses.
#[test]
fn a_peer_turning_round_is_drawn_turning_the_short_way() {
    let lobby = Lobby::flat();
    let mut ada = Raw::join(&lobby, "ada");
    let mut bob = join(&lobby, "bob");
    let ada_id = ada.id;
    let home = ada.spawn;
    let to = home + DVec3::new(0.5, 0.0, 0.0);
    let facing = |pos: DVec3, yaw: f32| ClientMessage::Move {
        pos,
        yaw,
        pitch: 0.0,
        frame: DQuat::IDENTITY,
        velocity: Vec3::ZERO,
        up: Face::PosY,
        stance: Stance::Standing,
    };
    let latest_yaw = |c: &Connection| peer(c, ada_id).sample(Instant::now() + Duration::from_secs(3600)).yaw;
    ada.send(&facing(home, 3.0));
    poll_until(&mut bob, "ada facing one way", |c, _| {
        c.peers().any(|p| p.id() == ada_id && p.visible()) && turn_gap(latest_yaw(c), 3.0) < 1e-3
    });
    thread::sleep(Duration::from_millis(300));
    bob.poll();
    ada.send(&facing(to, -3.0));
    poll_until(&mut bob, "ada facing the other way", |c, _| turn_gap(latest_yaw(c), -3.0) < 1e-3);
    let now = Instant::now();
    let p = peer(&bob, ada_id);
    let arc = turn_gap(3.0, -3.0);
    let mut halfway = false;
    for k in 0..200 {
        let r = p.sample(now + Duration::from_millis(5 * k));
        let (from_start, to_end) = (turn_gap(r.yaw, 3.0), turn_gap(r.yaw, -3.0));
        assert!(from_start + to_end <= arc + 1e-3, "yaw {} leaves the short arc", r.yaw);
        halfway |= from_start > 0.25 * arc && to_end > 0.25 * arc;
        let x = r.pos.0.x;
        assert!(x >= home.x - HALF_STEP - 1e-9 && x <= to.x + HALF_STEP + 1e-9, "x {x} leaves the segment");
    }
    assert!(halfway, "the turn is drawn between the poses, not snapped");
}
