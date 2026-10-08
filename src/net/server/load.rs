//! Bot load test: raw QUIC bots on one shared runtime against an in-process
//! server, plus the test-build instrumentation it reads (State lock holds and
//! out-queue depth). `cargo test --release --lib bot_load_clusters -- --ignored --nocapture`.
// Setup may unwrap: a panic here is a loud test failure.
#![allow(clippy::unwrap_used)]
use std::any::TypeId;
use std::net::{Ipv4Addr, SocketAddr};
use std::ops::{Deref, DerefMut};
use std::panic::Location;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use glam::DQuat;
use quinn::{Endpoint, RecvStream, SendStream};
use tokio::runtime::Runtime;
use voxel_engine::{DVec3, Vec3};

use super::{Config, LockRecover, NoclipPolicy, ServerHandle, State, TeleportPolicy, install_edits, spawn};
use crate::block::registry::BlockId;
use crate::coord::Face;
use crate::net::protocol::{self, ClientMessage, ServerMessage};
use crate::net::{PROTOCOL_VERSION, chat, quic};
use crate::presence::Stance;
use crate::space::atlas::Atlas;

const BUCKETS: usize = 8 + 61 * 8;

/// Log-linear histogram, eight sub-buckets per power of two (about 12% wide).
pub(super) struct Histogram {
    buckets: [AtomicU64; BUCKETS],
    max: AtomicU64,
}

#[derive(Clone, Copy, Default)]
struct Summary {
    count: u64,
    p50: u64,
    p99: u64,
    max: u64,
}

impl Histogram {
    const fn new() -> Self {
        Self { buckets: [const { AtomicU64::new(0) }; BUCKETS], max: AtomicU64::new(0) }
    }

    pub(super) fn record(&self, v: u64) {
        self.buckets[bucket(v)].fetch_add(1, Ordering::Relaxed);
        self.max.fetch_max(v, Ordering::Relaxed);
    }

    fn reset(&self) {
        for b in &self.buckets {
            b.store(0, Ordering::Relaxed);
        }
        self.max.store(0, Ordering::Relaxed);
    }

    fn summary(&self) -> Summary {
        let counts: Vec<u64> = self.buckets.iter().map(|b| b.load(Ordering::Relaxed)).collect();
        let count: u64 = counts.iter().sum();
        let max = self.max.load(Ordering::Relaxed);
        let at = |q: f64| {
            let want = ((count as f64 * q).ceil() as u64).max(1);
            let mut seen = 0;
            for (i, n) in counts.iter().enumerate() {
                seen += n;
                if seen >= want && i + 1 < BUCKETS {
                    return (floor_of(i + 1) - 1).min(max);
                }
            }
            max
        };
        Summary { count, p50: at(0.5), p99: at(0.99), max }
    }
}

fn bucket(v: u64) -> usize {
    if v < 8 {
        return v as usize;
    }
    let e = 63 - v.leading_zeros() as usize;
    (e - 2) * 8 + ((v >> (e - 3)) & 7) as usize
}

fn floor_of(i: usize) -> u64 {
    if i < 8 {
        return i as u64;
    }
    (8 + (i % 8) as u64) << (i / 8 - 1)
}

/// Nanoseconds each [`State`] lock was held.
pub(super) static LOCK_HOLD: Histogram = Histogram::new();
/// Frames waiting in an out-queue, sampled at every push.
pub(super) static QUEUE: Histogram = Histogram::new();
/// Ping to pong, in nanoseconds, as the bots see it.
static RTT: Histogram = Histogram::new();

/// Per call site of a [`State`] lock: (site, holds, total ns, max ns).
static SITES: Mutex<Vec<(&'static Location<'static>, u64, u64, u64)>> = Mutex::new(Vec::new());

fn sites_reset() {
    SITES.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clear();
}

/// The `n` sites with the most total hold time.
fn sites_top(n: usize) -> String {
    let mut sites = SITES.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
    sites.sort_by_key(|s| std::cmp::Reverse(s.2));
    sites
        .iter()
        .take(n)
        .map(|(at, count, total, max)| {
            format!("line {} ×{count} total {:.1} ms max {:.0} µs", at.line(), *total as f64 / 1e6, *max as f64 / 1e3)
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// A lock guard that records how long a [`State`] lock was held, and where.
pub(super) struct Timed<'a, T: 'static> {
    /// `None` only inside `drop`, which unlocks before recording.
    guard: Option<MutexGuard<'a, T>>,
    since: Instant,
    at: &'static Location<'static>,
}

impl<'a, T: 'static> Timed<'a, T> {
    #[track_caller]
    pub(super) fn new(guard: MutexGuard<'a, T>) -> Self {
        Self { guard: Some(guard), since: Instant::now(), at: Location::caller() }
    }
}

impl<T: 'static> Deref for Timed<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.guard.as_ref().unwrap()
    }
}

impl<T: 'static> DerefMut for Timed<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.guard.as_mut().unwrap()
    }
}

impl<T: 'static> Drop for Timed<'_, T> {
    fn drop(&mut self) {
        let ns = self.since.elapsed().as_nanos() as u64;
        drop(self.guard.take());
        if TypeId::of::<T>() == TypeId::of::<State>() {
            LOCK_HOLD.record(ns);
            let mut sites = SITES.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            match sites.iter_mut().find(|s| std::ptr::eq(s.0, self.at)) {
                Some(s) => {
                    s.1 += 1;
                    s.2 += ns;
                    s.3 = s.3.max(ns);
                }
                None => sites.push((self.at, 1, ns, ns)),
            }
        }
    }
}

const BOT_THREAD: &str = "loadbot";
const MOVE_PERIOD: Duration = Duration::from_millis(33);
const RADIUS: f64 = 1.5;
const SPIN: f64 = 2.0;
/// Bots float this far above their spawn so a hill beside it never blocks a body.
const LIFT: f64 = 3.0;
/// Altitude between layers, so a cluster mixes near and far peers inside interest range.
const LAYER_GAP: f64 = 40.0;
const CLIMB: f64 = 60.0;
const JOIN_STAGGER: Duration = Duration::from_millis(10);
/// After the run, how long bots keep reading so their last acks land.
const GRACE: Duration = Duration::from_millis(500);

#[derive(Clone, Copy)]
struct Pace {
    edit_every: Duration,
    chat_every: Duration,
    /// Bots spread over this many altitude layers [`LAYER_GAP`] apart.
    layers: usize,
}

/// One bot's counters. Each bot owns its own so the bots do not share cache lines.
#[derive(Default)]
struct Tally {
    joined: AtomicU64,
    failed: AtomicU64,
    kicked: AtomicU64,
    join_ns: AtomicU64,
    join_bytes: AtomicU64,
    join_frames: AtomicU64,
    bytes: AtomicU64,
    frames: AtomicU64,
    poses: AtomicU64,
    corrections: AtomicU64,
    edits: AtomicU64,
    answered: AtomicU64,
    rejected: AtomicU64,
}

macro_rules! counts {
    ($($field:ident),+ $(,)?) => {
        /// Summed [`Tally`] values at one instant.
        #[derive(Clone, Copy, Default, Debug)]
        struct Counts {
            $($field: u64,)+
            join_max_ns: u64,
        }

        impl Counts {
            fn of(tallies: &[Arc<Tally>]) -> Self {
                let mut c = Counts::default();
                for t in tallies {
                    $(c.$field += t.$field.load(Ordering::Relaxed);)+
                    c.join_max_ns = c.join_max_ns.max(t.join_ns.load(Ordering::Relaxed));
                }
                c
            }

            fn since(self, before: Counts) -> Self {
                Counts { $($field: self.$field - before.$field,)+ join_max_ns: self.join_max_ns }
            }
        }
    };
}
counts!(joined, failed, kicked, join_ns, join_bytes, join_frames, bytes, frames, poses, corrections, edits, answered, rejected);

/// What the bot reader learned about the one cell this bot edits, and when the
/// last ping left (nanoseconds after `origin`).
struct Mine {
    rev: AtomicU32,
    acked: AtomicU32,
    origin: Instant,
    pinged: AtomicU64,
}

struct Bot {
    index: usize,
    target: SocketAddr,
    hello: Vec<u8>,
    rock: Arc<str>,
    atlases: Arc<[Arc<Atlas>]>,
    pace: Pace,
    tally: Arc<Tally>,
    stop: Arc<AtomicBool>,
}

struct Session {
    _endpoint: Endpoint,
    conn: quinn::Connection,
    send: SendStream,
    recv: RecvStream,
    id: u32,
    spawn: DVec3,
}

fn hello(name: &str) -> Vec<u8> {
    let content = crate::net::content_id(&crate::block::BlockRegistry::with_builtins());
    ClientMessage::Hello {
        protocol: PROTOCOL_VERSION,
        worldgen: content.worldgen,
        gravity: content.gravity,
        law: content.law,
        palette: content.palette,
        name: name.into(),
        password: "".into(),
        mods: vec![],
    }
    .encode()
}

fn cell_of(atlases: &[Arc<Atlas>], p: DVec3) -> (i32, i32, i32) {
    atlases.iter().find_map(|a| a.storage_of(p)).map_or_else(
        || (crate::math::block_coord(p.x), crate::math::block_coord(p.y), crate::math::block_coord(p.z)),
        |c| (c[0] as i32, c[1] as i32, c[2] as i32),
    )
}

/// Above the bot's own circle, one block higher per 25 ids: chart spawns repeat every 25 ids,
/// so no two bots ever race one cell.
fn edit_cell(atlases: &[Arc<Atlas>], spawn: DVec3, height: f64, id: u32) -> (i32, i32, i32) {
    cell_of(atlases, spawn + DVec3::new(0.0, height + 2.0 + f64::from(id / 25), 0.0))
}

async fn join(bot: &Bot) -> Option<Session> {
    let mut endpoint = Endpoint::client((Ipv4Addr::UNSPECIFIED, 0).into()).ok()?;
    endpoint.set_default_client_config(quic::client_config());
    let started = Instant::now();
    let conn = endpoint.connect(bot.target, "watt").ok()?.await.ok()?;
    let (mut send, mut recv) = conn.open_bi().await.ok()?;
    protocol::write_frame_async(&mut send, &bot.hello).await.ok()?;
    let mut buf = Vec::new();
    let mut welcome = None;
    loop {
        protocol::read_frame_async(&mut recv, &mut buf).await.ok()?;
        bot.tally.join_bytes.fetch_add(4 + buf.len() as u64, Ordering::Relaxed);
        bot.tally.join_frames.fetch_add(1, Ordering::Relaxed);
        match ServerMessage::decode(&buf)? {
            ServerMessage::Welcome { player_id, spawn, .. } => welcome = Some((player_id, spawn)),
            ServerMessage::SnapshotEnd => break,
            _ => {}
        }
    }
    let (id, spawn) = welcome?;
    bot.tally.join_ns.store(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    bot.tally.joined.store(1, Ordering::Relaxed);
    Some(Session { _endpoint: endpoint, conn, send, recv, id, spawn })
}

async fn read_loop(mut recv: RecvStream, cell: (i32, i32, i32), mine: Arc<Mine>, tally: Arc<Tally>, stop: Arc<AtomicBool>) {
    let mut buf = Vec::new();
    while protocol::read_frame_async(&mut recv, &mut buf).await.is_ok() {
        tally.bytes.fetch_add(4 + buf.len() as u64, Ordering::Relaxed);
        tally.frames.fetch_add(1, Ordering::Relaxed);
        match ServerMessage::decode(&buf) {
            Some(ServerMessage::PeerPoses { poses }) => {
                tally.poses.fetch_add(poses.list.len() as u64, Ordering::Relaxed);
            }
            Some(ServerMessage::Position { .. }) => {
                tally.corrections.fetch_add(1, Ordering::Relaxed);
            }
            Some(ServerMessage::Pong { .. }) => {
                let now = mine.origin.elapsed().as_nanos() as u64;
                RTT.record(now.saturating_sub(mine.pinged.load(Ordering::Relaxed)));
            }
            Some(ServerMessage::EditAck { accepted, rev, .. }) => {
                mine.rev.store(rev, Ordering::Relaxed);
                mine.acked.fetch_add(1, Ordering::Relaxed);
                tally.answered.fetch_add(1, Ordering::Relaxed);
                if !accepted {
                    tally.rejected.fetch_add(1, Ordering::Relaxed);
                }
            }
            Some(ServerMessage::Edit { x, y, z, rev, .. }) if (x, y, z) == cell => mine.rev.store(rev, Ordering::Relaxed),
            Some(ServerMessage::Snapshot { edits }) => {
                for &(x, y, z, rev, _) in &edits {
                    if (x, y, z) == cell {
                        mine.rev.store(rev, Ordering::Relaxed);
                    }
                }
            }
            _ => {}
        }
    }
    if !stop.load(Ordering::Relaxed) {
        tally.kicked.fetch_add(1, Ordering::Relaxed);
    }
}

/// Climbs to its layer, then moves at 30 Hz on a small circle, toggles one cell every
/// `edit_every`, chats every `chat_every`, and pings every 2 s like the real client.
async fn play(bot: &Bot, session: Session) {
    let Session { _endpoint, conn, mut send, recv, id, spawn } = session;
    let layer = (bot.index % bot.pace.layers) as f64 * LAYER_GAP;
    let cell = edit_cell(&bot.atlases, spawn, LIFT + layer, id);
    let mine = Arc::new(Mine { rev: AtomicU32::new(0), acked: AtomicU32::new(0), origin: Instant::now(), pinged: AtomicU64::new(0) });
    let reader = tokio::spawn(read_loop(recv, cell, mine.clone(), bot.tally.clone(), bot.stop.clone()));
    let mut tick = tokio::time::interval(MOVE_PERIOD);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let start = Instant::now();
    let stagger = (bot.index % 16) as f64 / 16.0;
    let phase = bot.index as f64 * 0.37;
    let mut next_edit = start + bot.pace.edit_every.mul_f64(0.5 + stagger);
    let mut next_chat = start + bot.pace.chat_every.mul_f64(0.5 + stagger);
    let mut next_ping = start;
    let (mut edits, mut nonce) = (0u32, 0u32);
    'run: while !bot.stop.load(Ordering::Relaxed) {
        tick.tick().await;
        let now = Instant::now();
        let t = start.elapsed().as_secs_f64();
        let a = phase + t * SPIN;
        let (sin, cos) = a.sin_cos();
        let climbed = CLIMB * t >= layer;
        let rise = if climbed { 0.0 } else { CLIMB };
        let pos = spawn + DVec3::new(RADIUS * cos, LIFT + layer.min(CLIMB * t), RADIUS * sin);
        let velocity = Vec3::new((-RADIUS * SPIN * sin) as f32, rise as f32, (RADIUS * SPIN * cos) as f32);
        let mut frames = vec![
            ClientMessage::Move {
                pos,
                yaw: a as f32,
                pitch: 0.0,
                frame: DQuat::IDENTITY,
                velocity,
                up: Face::PosY,
                stance: Stance::Standing,
            },
        ];
        if climbed && now >= next_edit && mine.acked.load(Ordering::Relaxed) == edits {
            let spec = if edits % 2 == 0 { bot.rock.clone() } else { Arc::from("air") };
            let expect = mine.rev.load(Ordering::Relaxed);
            frames.push(ClientMessage::Edit { req: edits + 1, x: cell.0, y: cell.1, z: cell.2, expect, spec });
            edits += 1;
            bot.tally.edits.fetch_add(1, Ordering::Relaxed);
            next_edit += bot.pace.edit_every;
        }
        if now >= next_chat {
            frames.push(ClientMessage::Chat { channel: chat::LOCAL, text: "hello from a bot".into() });
            next_chat += bot.pace.chat_every;
        }
        if now >= next_ping {
            mine.pinged.store(mine.origin.elapsed().as_nanos() as u64, Ordering::Relaxed);
            nonce += 1;
            frames.push(ClientMessage::Ping { nonce });
            next_ping += Duration::from_secs(2);
        }
        for msg in frames {
            if protocol::write_frame_async(&mut send, &msg.encode()).await.is_err() {
                break 'run;
            }
        }
    }
    tokio::time::sleep(GRACE).await;
    conn.close(0u32.into(), b"bye");
    let _ = reader.await;
}

async fn run_bot(bot: Bot) {
    tokio::time::sleep(JOIN_STAGGER * bot.index as u32).await;
    match join(&bot).await {
        Some(session) => play(&bot, session).await,
        None => {
            bot.tally.failed.store(1, Ordering::Relaxed);
        }
    }
}

fn bot_runtime(workers: usize) -> Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .thread_name(BOT_THREAD)
        .enable_all()
        .build()
        .unwrap()
}

/// CPU nanoseconds of every live thread: (server, bots). The test thread itself sleeps.
fn cpu() -> (u64, u64) {
    let (mut server, mut bots) = (0, 0);
    let Ok(dir) = std::fs::read_dir("/proc/self/task") else { return (0, 0) };
    for entry in dir.flatten() {
        let path = entry.path();
        let comm = std::fs::read_to_string(path.join("comm")).unwrap_or_default();
        let ns = std::fs::read_to_string(path.join("schedstat"))
            .ok()
            .and_then(|s| s.split_whitespace().next()?.parse::<u64>().ok())
            .unwrap_or(0);
        if comm.trim_end() == BOT_THREAD {
            bots += ns;
        } else {
            server += ns;
        }
    }
    (server, bots)
}

/// The dedicated server's defaults: teleport and noclip for operators only, so every bot
/// move runs the body check.
fn dedicated() -> ServerHandle {
    spawn(
        0,
        Config { seed: 4242, teleport: TeleportPolicy::Ops, noclip: NoclipPolicy::Ops, ..Config::default() },
    )
    .unwrap()
}

/// One measured run.
struct Run {
    bots: usize,
    secs: f64,
    server_cpu: f64,
    bot_cpu: f64,
    /// Lock holds while the bots were joining.
    joining: Summary,
    lock: Summary,
    queue: Summary,
    rtt: Summary,
    /// The busiest lock sites in the window, and while joining.
    sites: String,
    join_sites: String,
    window: Counts,
    total: Counts,
}

fn launch(
    rt: &Runtime,
    handle: &ServerHandle,
    bots: usize,
    pace: Pace,
    stop: &Arc<AtomicBool>,
) -> (Vec<Arc<Tally>>, Vec<tokio::task::JoinHandle<()>>) {
    let target = SocketAddr::from((Ipv4Addr::LOCALHOST, handle.addr().port()));
    let rock: Arc<str> = {
        let state = handle.state.lock_recover();
        let id = state.registry.id_by_label("rock").unwrap_or(BlockId(1));
        state.registry.spec(id).into()
    };
    let atlases: Arc<[Arc<Atlas>]> = handle.ctx.generator.atlases().into();
    let tallies: Vec<Arc<Tally>> = (0..bots).map(|_| Arc::default()).collect();
    let tasks = tallies
        .iter()
        .enumerate()
        .map(|(i, tally)| {
            let bot = Bot {
                index: i,
                target,
                hello: hello(&format!("bot{i}")),
                rock: rock.clone(),
                atlases: atlases.clone(),
                pace,
                tally: tally.clone(),
                stop: stop.clone(),
            };
            rt.spawn(run_bot(bot))
        })
        .collect();
    (tallies, tasks)
}

fn wait_joined(tallies: &[Arc<Tally>]) {
    let deadline = Instant::now() + Duration::from_secs(120);
    while Instant::now() < deadline {
        let c = Counts::of(tallies);
        if (c.joined + c.failed) as usize >= tallies.len() {
            return;
        }
        thread::sleep(Duration::from_millis(20));
    }
}

fn finish(rt: &Runtime, stop: &AtomicBool, tasks: Vec<tokio::task::JoinHandle<()>>) {
    stop.store(true, Ordering::Relaxed);
    rt.block_on(async {
        let _ = tokio::time::timeout(Duration::from_secs(20), async {
            for task in tasks {
                let _ = task.await;
            }
        })
        .await;
    });
}

fn cluster(rt: &Runtime, bots: usize, pace: Pace, warm: Duration, window: Duration) -> Run {
    let handle = dedicated();
    let stop = Arc::new(AtomicBool::new(false));
    LOCK_HOLD.reset();
    sites_reset();
    let (tallies, tasks) = launch(rt, &handle, bots, pace, &stop);
    wait_joined(&tallies);
    let joining = LOCK_HOLD.summary();
    let join_sites = sites_top(3);
    thread::sleep(warm);
    let before = Counts::of(&tallies);
    let (server0, bots0) = cpu();
    LOCK_HOLD.reset();
    QUEUE.reset();
    RTT.reset();
    sites_reset();
    let t0 = Instant::now();
    thread::sleep(window);
    let secs = t0.elapsed().as_secs_f64();
    let (server1, bots1) = cpu();
    let lock = LOCK_HOLD.summary();
    let queue = QUEUE.summary();
    let rtt = RTT.summary();
    let sites = sites_top(4);
    let window = Counts::of(&tallies).since(before);
    finish(rt, &stop, tasks);
    let total = Counts::of(&tallies);
    handle.stop();
    Run {
        bots,
        secs,
        server_cpu: (server1 - server0) as f64 / 1e9 / secs,
        bot_cpu: (bots1 - bots0) as f64 / 1e9 / secs,
        joining,
        lock,
        queue,
        rtt,
        sites,
        join_sites,
        window,
        total,
    }
}

fn print_header() {
    println!(
        "| bots | server CPU (cores) | bot CPU | lock p50 / p99 / max (µs) | holds/s | join lock max (µs) | join mean (ms) | KB/s per bot | frames/s per bot | poses/s per bot | queue p99 / max | ping p50 / p99 (ms) | kicks | corrections | edits sent / answered / rejected |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
}

fn print_row(r: &Run) {
    let per_bot = |v: u64| v as f64 / r.secs / r.bots as f64;
    println!(
        "| {} | {:.2} | {:.2} | {:.1} / {:.1} / {:.1} | {:.0} | {:.1} | {:.1} | {:.1} | {:.0} | {:.0} | {} / {} | {:.2} / {:.2} | {} | {} | {} / {} / {} |",
        r.bots,
        r.server_cpu,
        r.bot_cpu,
        r.lock.p50 as f64 / 1e3,
        r.lock.p99 as f64 / 1e3,
        r.lock.max as f64 / 1e3,
        r.lock.count as f64 / r.secs,
        r.joining.max as f64 / 1e3,
        r.total.join_ns as f64 / 1e6 / r.bots as f64,
        per_bot(r.window.bytes) / 1024.0,
        per_bot(r.window.frames),
        per_bot(r.window.poses),
        r.queue.p99,
        r.queue.max,
        r.rtt.p50 as f64 / 1e6,
        r.rtt.p99 as f64 / 1e6,
        r.total.kicked,
        r.total.corrections,
        r.total.edits,
        r.total.answered,
        r.total.rejected,
    );
}

/// `edits` cells in a 100 × 20 × 50 block under the spawn, half air and half a
/// spread of materials, as a built-up world's overlay.
fn fill(handle: &ServerHandle, edits: usize) {
    let mut state = handle.state.lock_recover();
    let mut specs = vec!["air".to_string()];
    for i in 1..state.registry.block_count().min(16) {
        specs.push(state.registry.spec(BlockId(i as u16)));
    }
    let spawn = handle.ctx.generator.chart_spawn().unwrap_or(DVec3::ZERO);
    let (bx, by, bz) = cell_of(handle.ctx.generator.atlases(), spawn);
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut cells = Vec::with_capacity(edits);
    'fill: for dx in 0..100 {
        for dz in 0..50 {
            for dy in 0..20 {
                if cells.len() == edits {
                    break 'fill;
                }
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let spec = if rng % 2 == 0 { &specs[0] } else { &specs[1 + (rng >> 8) as usize % (specs.len() - 1)] };
                cells.push((bx + dx - 50, by - 4 - dy, bz + dz - 25, spec.clone()));
            }
        }
    }
    let skipped = install_edits(&mut state, &cells);
    assert!(skipped.is_empty());
}

/// One bot joins an idle server holding `edits` cells.
fn join_big_world(rt: &Runtime, edits: usize) {
    let handle = dedicated();
    fill(&handle, edits);
    assert_eq!(handle.state.lock_recover().edits.len(), edits);
    let stop = Arc::new(AtomicBool::new(false));
    let pace = Pace { edit_every: Duration::from_secs(3600), chat_every: Duration::from_secs(3600), layers: 1 };
    LOCK_HOLD.reset();
    let (tallies, tasks) = launch(rt, &handle, 1, pace, &stop);
    wait_joined(&tallies);
    let lock = LOCK_HOLD.summary();
    finish(rt, &stop, tasks);
    let c = Counts::of(&tallies);
    println!(
        "join with {edits} edits: {:.1} ms, {:.0} KB in {} frames, lock max {:.2} ms",
        c.join_max_ns as f64 / 1e6,
        c.join_bytes as f64 / 1024.0,
        c.join_frames,
        lock.max as f64 / 1e6,
    );
    client_join(&handle, edits);
    handle.stop();
}

fn threads() -> usize {
    std::fs::read_dir("/proc/self/task").map_or(0, |d| d.count())
}

/// The game's view of the same join: a real [`Connection`](crate::net::client::Connection)
/// polled once per 16 ms frame until the overlay is in.
fn client_join(handle: &ServerHandle, edits: usize) {
    use crate::net::client::{Connection, Incoming};
    let before = threads();
    let started = Instant::now();
    let mut conn = Connection::connect("127.0.0.1", handle.addr().port(), "client", "").unwrap();
    let added = threads() - before;
    let (mut most, mut polls, mut cells) = (0, 0, 0);
    while !conn.snapshot_ready() && started.elapsed() < Duration::from_secs(30) {
        let n = conn.poll().iter().filter(|e| matches!(e, Incoming::Mutation { .. })).count();
        most = most.max(n);
        cells += n;
        polls += 1;
        thread::sleep(Duration::from_millis(16));
    }
    assert_eq!(cells, edits, "the client applied the whole overlay");
    println!(
        "client join: {added} threads for the connection; overlay of {cells} cells in {:.0} ms over {polls} polls, at most {most} cells in one poll",
        started.elapsed().as_secs_f64() * 1e3,
    );
}

/// Four bots for two seconds: nobody is kicked or corrected and every edit is answered.
#[test]
fn bot_load_smoke() {
    let rt = bot_runtime(2);
    let pace = Pace { edit_every: Duration::from_millis(300), chat_every: Duration::from_millis(700), layers: 1 };
    let run = cluster(&rt, 4, pace, Duration::from_millis(200), Duration::from_secs(2));
    assert_eq!((run.total.joined, run.total.failed), (4, 0), "{:?}", run.total);
    assert_eq!(run.total.kicked, 0, "no bot is kicked");
    assert_eq!(run.total.corrections, 0, "no honest move is corrected");
    assert!(run.total.edits >= 4, "the bots edited: {:?}", run.total);
    assert_eq!(run.total.answered, run.total.edits, "every edit is answered");
    assert_eq!(run.total.rejected, 0, "no bot races another for a cell");
    assert!(run.window.poses > 0, "the bots see each other move");
}

#[test]
#[ignore = "load test: cargo test --release --lib bot_load_clusters -- --ignored --nocapture"]
fn bot_load_clusters() {
    let rt = bot_runtime(4);
    let pace = Pace { edit_every: Duration::from_secs(3), chat_every: Duration::from_secs(10), layers: 4 };
    print_header();
    let mut sites = Vec::new();
    for bots in [8, 32, 64, 128] {
        let run = cluster(&rt, bots, pace, Duration::from_millis(2500), Duration::from_secs(5));
        print_row(&run);
        sites.push(format!("{bots} bots, server.rs lock sites: {}", run.sites));
        sites.push(format!("{bots} bots joining: {}", run.join_sites));
    }
    for line in sites {
        println!("{line}");
    }
    join_big_world(&rt, 100_000);
}

#[test]
fn histogram_buckets_bound_their_values() {
    for v in [0u64, 1, 7, 8, 9, 15, 16, 17, 1000, 123_456_789, u64::MAX] {
        let i = bucket(v);
        assert!(floor_of(i) <= v, "{v} below its bucket {i}");
        if i + 1 < BUCKETS {
            assert!(v < floor_of(i + 1), "{v} past its bucket {i}");
        }
    }
}
