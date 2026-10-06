//! Headless ground walk with real streaming, for a player who sticks to the ground.
//!
//! Ignored. The short probe measures rounding, chart round-trip and a few seconds of
//! walking; the long walk is the regression harness. Streaming frames are paced to `hz`:
//! the loader samples wall-clock speed, and an unpaced loop looks like flight fast enough
//! to drop the ground.
//!
//! `WALK_SECS` (default 180), `WALK_HZ` (60), `WALK_SITES` (`flat,spawn,owner`),
//! `WALK_VIEWS` (`2,1;16,5`), `WALK_LOD` (1), `WALK_STRICT` (1).
//! `cargo test --release --lib walking_on_the_ground_never_gets_stuck -- --ignored --nocapture`

use std::io::Write;

use voxel_engine::DVec3;

use super::flight_bench::step;
use crate::coord::{BlockCoord, ChunkCoord, Face};
use crate::input::movement::{update_player, update_player_in, MoveInput};
use crate::math::{block_coord, Aabb, Bounded};
use crate::player::{collision_box, Motion, Player, Stance};
use crate::render_config::RenderConfig;
use crate::world::chunk::CHUNK_SIZE;
use crate::world::generation::{WorldgenKind, FLAT_HEIGHT};
use crate::world::World;

const OWNER_SEED: i64 = 1_791_184_794_939_118_871;
const SPAWN_SEED: i64 = 42;

fn stamp(msg: &str) {
    println!("WALK {msg}");
    let _ = std::io::stdout().flush();
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

fn env_flag(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => v != "0",
        Err(_) => default,
    }
}

/// How far the box minimum sits below the mathematical feet. Negative clips the cell underfoot.
fn feet_dip(feet_y: f64, stance: Stance) -> f64 {
    let eye = DVec3::new(0.5, feet_y + stance.eye_offset(), 0.5);
    collision_box(eye, stance, Face::PosY).min().y - feet_y
}

fn chunk_of_pos(p: DVec3) -> ChunkCoord {
    World::chunk_of(block_coord(p.x), block_coord(p.y), block_coord(p.z))
}

/// Storage eye collision will use this step. Physical positions on a chart are not in the voxel grid.
fn storage_of(world: &World, physical: DVec3) -> DVec3 {
    world.atlas_at(physical).and_then(|a| a.local(physical)).map(|h| h.storage).unwrap_or(physical)
}

fn horiz_speed(player: &Player) -> f64 {
    let v = player.velocity();
    let up = player.up();
    (v - up * v.dot(up)).length()
}

/// Signed gap from the feet face to the top of the first solid within six blocks. Negative is inside it.
fn ground_gap(world: &World, player: &Player, storage: DVec3) -> f64 {
    let b = collision_box(storage, player.stance, player.up_axis);
    let a = player.up_axis.axis();
    let sign = player.up_axis.sign() as f64;
    let face = if sign > 0.0 { b.min()[a] } else { b.max()[a] };
    let mut cell = [block_coord(storage.x), block_coord(storage.y), block_coord(storage.z)];
    cell[a] = block_coord(face);
    for k in 0..6 {
        let c = cell[a] - sign as i32 * k;
        let (x, y, z) = match a {
            0 => (c, cell[1], cell[2]),
            1 => (cell[0], c, cell[2]),
            _ => (cell[0], cell[1], c),
        };
        if world.is_solid(x, y, z) {
            let top = if sign > 0.0 { c as f64 + 1.0 } else { c as f64 };
            return (face - top) * sign;
        }
    }
    1.0e9
}

fn ground_samples(player: &Player, storage: DVec3) -> [DVec3; 2] {
    let b = collision_box(storage, player.stance, player.up_axis);
    let a = player.up_axis.axis();
    let sign = player.up_axis.sign() as f64;
    let face = if sign > 0.0 { b.min()[a] } else { b.max()[a] };
    let mut at = storage;
    let mut below = storage;
    at[a] = face;
    below[a] = face - sign * 0.5;
    [below, at]
}

fn box_chunks(storage: DVec3, player: &Player) -> Vec<ChunkCoord> {
    let b = collision_box(storage, player.stance, player.up_axis);
    let (min, max) = (b.min(), b.max());
    let s = CHUNK_SIZE as i32;
    let (x0, x1) = (block_coord(min.x).div_euclid(s), block_coord_end_chunk(max.x, s));
    let (y0, y1) = (block_coord(min.y).div_euclid(s), block_coord_end_chunk(max.y, s));
    let (z0, z1) = (block_coord(min.z).div_euclid(s), block_coord_end_chunk(max.z, s));
    let mut out = Vec::with_capacity(8);
    for x in x0..=x1 {
        for y in y0..=y1 {
            for z in z0..=z1 {
                out.push(ChunkCoord::new(x, y, z));
            }
        }
    }
    out
}

fn block_coord_end_chunk(v: f64, s: i32) -> i32 {
    crate::math::block_coord_end(v).div_euclid(s)
}

fn solids_touching(world: &World, aabb: &Aabb, prev: &[ChunkCoord]) -> String {
    let mut text = String::new();
    let mut n = 0;
    for (x, y, z) in aabb.voxel_cells() {
        if !world.is_solid(x, y, z) {
            continue;
        }
        let c = BlockCoord::new(x, y, z).split().0;
        let state = if world.chunks.contains_key(&c) {
            "loaded"
        } else if world.generating.contains(&c) {
            "generating"
        } else {
            "absent"
        };
        let when = if prev.is_empty() {
            "first"
        } else if prev.contains(&c) {
            "held"
        } else {
            "entered"
        };
        if n > 0 {
            text.push(' ');
        }
        text.push_str(&format!("({x},{y},{z}) {state}/{when} chunk ({},{},{})", c.x, c.y, c.z));
        n += 1;
        if n == 6 {
            text.push_str(" ...");
            break;
        }
    }
    if n == 0 {
        text.push_str("none");
    }
    text
}

/// Storage-frame heading of the body's forward, up component removed. `None` when it vanishes.
fn storage_forward(world: &World, player: &Player) -> Option<DVec3> {
    let (fwd, _) = player.movement_basis();
    let mut dir = fwd;
    if let Some(here) = world.atlas_at(player.position).and_then(|a| a.local(player.position)) {
        dir = here.rotation().inverse() * dir;
    }
    let a = player.up_axis.axis();
    dir[a] = 0.0;
    let len = dir.length();
    if !len.is_finite() || len < 1e-9 {
        return None;
    }
    dir /= len;
    Some(dir)
}

/// A solid within one walking step, at the current height. A lip counts as a wall.
fn wall_ahead(world: &World, player: &Player, storage: DVec3) -> bool {
    let Some(dir) = storage_forward(world, player) else {
        return false;
    };
    // One accelerated step on the ground is well under half a block.
    let probe = storage + dir * 0.5;
    world.collides(&collision_box(probe, player.stance, player.up_axis))
}

/// Which forward distances overlap a solid, for a stall log.
fn probe_reach(world: &World, player: &Player, storage: DVec3) -> String {
    let Some(dir) = storage_forward(world, player) else {
        return "nodir".to_string();
    };
    let mut text = String::new();
    for dist in [0.2, 0.5, 1.0, 1.5] {
        let hit = world.collides(&collision_box(storage + dir * dist, player.stance, player.up_axis));
        if !text.is_empty() {
            text.push(' ');
        }
        text.push_str(&format!("{dist}:{hit}"));
    }
    text
}

/// Solid tops on the 3×3 around the feet, relative to the feet face. `.` is air for four blocks.
fn columns_around(world: &World, player: &Player, storage: DVec3) -> String {
    let b = collision_box(storage, player.stance, player.up_axis);
    let a = player.up_axis.axis();
    let sign = player.up_axis.sign() as f64;
    let face = if sign > 0.0 { b.min()[a] } else { b.max()[a] };
    let mut cell = [block_coord(storage.x), block_coord(storage.y), block_coord(storage.z)];
    cell[a] = block_coord(face);
    let mut text = String::new();
    for dx in -1..=1 {
        for dz in -1..=1 {
            let mut top = None;
            for k in 0..4 {
                let c = cell[a] + sign as i32 - k;
                let (x, y, z) = match a {
                    0 => (c, cell[1] + dx, cell[2] + dz),
                    1 => (cell[0] + dx, c, cell[2] + dz),
                    _ => (cell[0] + dx, cell[1] + dz, c),
                };
                if world.is_solid(x, y, z) {
                    let plane = if sign > 0.0 { c as f64 + 1.0 } else { c as f64 };
                    top = Some((plane - face) * sign);
                    break;
                }
            }
            if !text.is_empty() {
                text.push(' ');
            }
            match top {
                Some(h) => text.push_str(&format!("{h:.2}")),
                None => text.push('.'),
            }
        }
    }
    text
}

fn closure_y(world: &World, physical: DVec3) -> Option<f64> {
    let atlas = world.atlas_at(physical)?;
    let here = atlas.local(physical)?;
    let back = atlas.embed_storage(here.patch, here.storage);
    let again = atlas.local(back)?;
    Some(again.storage.y - here.storage.y)
}

fn physics(world: &mut World, player: &mut Player, input: &MoveInput, dt: f32) {
    let g = world.gravity_at(player.position).accel;
    if let Some(atlas) = world.atlas_at(player.position).cloned() {
        update_player_in(player, world, &atlas, input, dt, g);
    } else {
        update_player(player, world, input, dt, g);
    }
}

struct Keys {
    yaw: f32,
    pushing: bool,
    jump: bool,
    sprint: bool,
    sneak: bool,
    mode: u64,
    rng: u64,
    frame: u64,
    period: u64,
}

impl Keys {
    fn new(hz: f64, salt: u64) -> Self {
        Self {
            yaw: 0.0,
            pushing: true,
            jump: false,
            sprint: false,
            sneak: false,
            mode: 0,
            rng: salt | 1,
            frame: 0,
            period: (hz * 2.0).max(1.0) as u64,
        }
    }

    fn advance(&mut self) {
        if self.frame.is_multiple_of(self.period) {
            let mut x = self.rng;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.rng = x;
            self.yaw = ((x % 628) as f32) / 100.0;
            self.mode = (x >> 10) % 6;
            self.sprint = self.mode == 1 || self.mode == 5;
            self.sneak = self.mode == 2;
            self.pushing = self.mode != 4;
            if self.mode == 5 {
                self.yaw += std::f32::consts::PI;
            }
        }
        self.jump = self.mode == 3 && self.frame % self.period < 4;
        self.frame += 1;
    }

    fn input(&self) -> MoveInput {
        MoveInput::keys(0.0, if self.pushing { 1.0 } else { 0.0 }, self.jump, self.sprint, self.sneak)
    }
}

struct Report {
    frames: u64,
    overlaps: u32,
    stalls: u32,
    unloads: u32,
    absent: u32,
    generating: u32,
    min_gap: f64,
    min_dy: f64,
    max_dy: f64,
    min_closure: f64,
    max_closure: f64,
    landed: bool,
    max_speed: f64,
    min_load_h: i32,
    min_load_v: i32,
    hits: Vec<String>,
}

impl Report {
    fn new() -> Self {
        Self {
            frames: 0,
            overlaps: 0,
            stalls: 0,
            unloads: 0,
            absent: 0,
            generating: 0,
            min_gap: f64::MAX,
            min_dy: f64::MAX,
            max_dy: f64::MIN,
            min_closure: f64::MAX,
            max_closure: f64::MIN,
            landed: false,
            max_speed: 0.0,
            min_load_h: i32::MAX,
            min_load_v: i32::MAX,
            hits: Vec::new(),
        }
    }

    fn hit(&mut self, line: String) {
        if self.hits.len() < 8 {
            self.hits.push(line);
        }
    }

    fn print(&self, tag: &str) {
        let gap = if self.min_gap > 1.0e8 { f64::NAN } else { self.min_gap };
        let (min_dy, max_dy) = if self.min_dy > 1.0e8 { (f64::NAN, f64::NAN) } else { (self.min_dy, self.max_dy) };
        let (min_c, max_c) =
            if self.min_closure > 1.0e8 { (f64::NAN, f64::NAN) } else { (self.min_closure, self.max_closure) };
        let (lh, lv) = if self.min_load_h == i32::MAX {
            (-1, -1)
        } else {
            (self.min_load_h, self.min_load_v)
        };
        stamp(&format!(
            "{tag} frames={} overlaps={} stalls={} unloads={} absent={} generating={} landed={} min_gap={gap:.3e} dy=[{min_dy:.3e},{max_dy:.3e}] closure=[{min_c:.3e},{max_c:.3e}] speed={:.2} load=({lh},{lv})",
            self.frames, self.overlaps, self.stalls, self.unloads, self.absent, self.generating, self.landed, self.max_speed
        ));
        for h in &self.hits {
            stamp(&format!("HIT {h}"));
        }
    }
}

struct Watch {
    report: Report,
    prev_chunks: Vec<ChunkCoord>,
    prev_storage: Option<DVec3>,
    prev_ground: [Option<ChunkCoord>; 2],
    prev_ground_loaded: [bool; 2],
    slow_run: u32,
    edit: u64,
}

impl Watch {
    fn new(world: &World) -> Self {
        Self {
            report: Report::new(),
            prev_chunks: Vec::new(),
            prev_storage: None,
            prev_ground: [None, None],
            prev_ground_loaded: [false, false],
            slow_run: 0,
            edit: world.edit_generation(),
        }
    }

    fn observe(&mut self, world: &World, player: &Player, keys: Option<&Keys>, tag: &str) {
        let storage = storage_of(world, player.position);
        let samples = ground_samples(player, storage);
        let ground_chunks = [chunk_of_pos(samples[0]), chunk_of_pos(samples[1])];
        for i in 0..2 {
            let loaded = world.chunks.contains_key(&ground_chunks[i]);
            if self.prev_ground[i] == Some(ground_chunks[i]) && self.prev_ground_loaded[i] && !loaded {
                self.report.unloads += 1;
                let c = ground_chunks[i];
                self.report.hit(format!(
                    "{tag} frame {} ground chunk ({},{},{}) unloaded edit {}",
                    self.report.frames, c.x, c.y, c.z, world.edit_generation()
                ));
            }
            if player.on_ground() && !loaded {
                if world.generating.contains(&ground_chunks[i]) {
                    self.report.generating += 1;
                } else {
                    self.report.absent += 1;
                }
            }
            self.prev_ground[i] = Some(ground_chunks[i]);
            self.prev_ground_loaded[i] = loaded;
        }

        let gap = ground_gap(world, player, storage);
        if player.on_ground() && gap < self.report.min_gap {
            self.report.min_gap = gap;
        }
        if let Some(prev) = self.prev_storage
            && player.on_ground()
            && keys.is_none_or(|k| !k.jump)
        {
            let dy = storage[player.up_axis.axis()] - prev[player.up_axis.axis()];
            let sign = player.up_axis.sign() as f64;
            let along = dy * sign;
            self.report.min_dy = self.report.min_dy.min(along);
            self.report.max_dy = self.report.max_dy.max(along);
        }
        if let Some(c) = closure_y(world, player.position) {
            self.report.min_closure = self.report.min_closure.min(c);
            self.report.max_closure = self.report.max_closure.max(c);
        }
        self.report.max_speed = self.report.max_speed.max(world.stream_pacer.speed_mps());
        if world.load_h >= 0 {
            self.report.min_load_h = self.report.min_load_h.min(world.load_h);
            self.report.min_load_v = self.report.min_load_v.min(world.load_v);
        }
        self.prev_storage = Some(storage);

        let aabb = collision_box(storage, player.stance, player.up_axis);
        let overlapped = world.collides(&aabb);
        if overlapped {
            self.report.overlaps += 1;
            let edit = world.edit_generation();
            self.report.hit(format!(
                "{tag} overlap frame {} gap {gap:.4e} storage {:?} box_min {:?} edit {edit} was {} {}",
                self.report.frames,
                storage,
                aabb.min(),
                self.edit,
                solids_touching(world, &aabb, &self.prev_chunks)
            ));
            self.edit = edit;
        } else if let Some(keys) = keys
            && keys.pushing
            && player.on_ground()
            && !keys.jump
            && horiz_speed(player) < 0.15
            && !wall_ahead(world, player, storage)
        {
            // One rejected step at a corner is not stuck. The owner cannot move again without
            // jumping, which takes a third of a second of held forward with nowhere to go.
            self.slow_run += 1;
            if self.slow_run > 20 {
                self.report.stalls += 1;
                let speed = horiz_speed(player);
                self.report.hit(format!(
                    "{tag} stall frame {} horiz {speed:.4} gap {gap:.4e} h {:.2} up {:?} load ({},{}) speed {:.1} probes {} cols {} storage {storage:?}",
                    self.report.frames,
                    player.stance.height(),
                    player.up_axis,
                    world.load_h,
                    world.load_v,
                    world.stream_pacer.speed_mps(),
                    probe_reach(world, player, storage),
                    columns_around(world, player, storage),
                ));
            }
        } else {
            self.slow_run = 0;
        }

        self.prev_chunks = box_chunks(storage, player)
            .into_iter()
            .filter(|c| world.chunks.contains_key(c))
            .collect();
        if player.on_ground() {
            self.report.landed = true;
        }
        self.report.frames += 1;
    }
}

fn pace(start: std::time::Instant, period: std::time::Duration) {
    if let Some(rest) = period.checked_sub(start.elapsed()) {
        std::thread::sleep(rest);
    }
}

/// Random headings, jumps, sprints, sneaks, stops and turn-backs. `stream` runs [`step`] after physics,
/// paced so the loader's wall-clock speed stays a walk.
fn walk(world: &mut World, player: &mut Player, secs: f64, hz: f64, stream: bool, tag: &str) -> Report {
    let dt = (1.0 / hz) as f32;
    let frames = (secs * hz).round().max(1.0) as u64;
    let period = std::time::Duration::from_secs_f64(1.0 / hz);
    if stream {
        world.prepare_around(player.position);
        world.drive_spawn_ready();
    } else {
        world.ensure_around(player.position);
    }
    // A spawn inside a tree reads as solid once the slab exists. Step up until the box is free.
    for _ in 0..24 {
        let storage = storage_of(world, player.position);
        if !world.collides(&collision_box(storage, player.stance, player.up_axis)) {
            break;
        }
        player.position += player.up();
    }

    let mut watch = Watch::new(world);
    let idle = MoveInput::keys(0.0, 0.0, false, false, false);
    let settle = (hz * 6.0).round() as u64;
    let mut grounded = 0u32;
    for _ in 0..settle {
        let start = std::time::Instant::now();
        if stream {
            if world.spawn_ready() {
                physics(world, player, &idle, dt);
            }
            step(world, player.position);
            pace(start, period);
        } else {
            world.ensure_around(player.position);
            physics(world, player, &idle, dt);
        }
        watch.observe(world, player, None, tag);
        if player.on_ground() {
            grounded += 1;
            if grounded > (hz as u32 / 5).max(4) {
                break;
            }
        } else {
            grounded = 0;
        }
    }
    if !player.on_ground() {
        watch.report.hit(format!("{tag} never landed at {:?}", player.position));
    }

    let mut keys = Keys::new(hz, tag.bytes().fold(0u64, |h, b| h.wrapping_mul(131).wrapping_add(b as u64)));
    for _ in 0..frames {
        let start = std::time::Instant::now();
        keys.advance();
        player.orientation.yaw = keys.yaw;
        // Observe the box the step is about to use, then move, then stream (the game's order).
        watch.observe(world, player, Some(&keys), tag);
        physics(world, player, &keys.input(), dt);
        if stream {
            step(world, player.position);
            pace(start, period);
        } else {
            world.ensure_around(player.position);
        }
    }
    watch.report.print(tag);
    watch.report
}

fn render(lod2: bool) -> RenderConfig {
    RenderConfig { lod2, occlusion: lod2, ..RenderConfig::default() }
}

fn chart_world(seed: i64, lod2: bool) -> World {
    World::with_kind(seed, render(lod2), WorldgenKind::Diffusion, false)
}

fn flat_world(lod2: bool) -> World {
    World::with_kind(1, render(lod2), WorldgenKind::Flat, false)
}

/// Physical eye two blocks above the chart column under `hint`.
fn eye_above_column(world: &World, hint: DVec3) -> Option<DVec3> {
    let atlas = world.atlas_at(hint)?.clone();
    let here = atlas.local(hint)?;
    let ground = world.terrain().surface(Face::PosY, block_coord(here.storage.x), block_coord(here.storage.z));
    if ground == i32::MIN || ground > 1_000_000_000 {
        return None;
    }
    let stand = DVec3::new(
        here.storage.x,
        ground as f64 + Stance::Standing.eye_offset() + 2.0,
        here.storage.z,
    );
    Some(atlas.embed_storage(here.patch, stand))
}

fn place_chart(world: &World, hint: DVec3) -> Player {
    let at = eye_above_column(world, hint).unwrap_or(hint);
    let mut player = Player::new(at);
    let g = world.gravity_at(at).accel;
    if g.length() > 1.0 {
        player.snap_up((-g).normalize());
    }
    player
}

fn place_flat(world: &World) -> Player {
    let y = FLAT_HEIGHT as f64 + Stance::Standing.eye_offset() + 2.0;
    let mut player = Player::new(DVec3::new(0.5, y, 0.5));
    let g = world.gravity_at(player.position).accel;
    if g.length() > 1.0 {
        player.snap_up((-g).normalize());
    }
    player
}

fn measure_roundtrip(world: &World, name: &str, hint: DVec3) {
    let Some(atlas) = world.atlas_at(hint).cloned() else {
        stamp(&format!("{name} roundtrip: no atlas at {hint:?}"));
        return;
    };
    let Some(here) = atlas.local(hint) else {
        stamp(&format!("{name} roundtrip: local failed at {hint:?}"));
        return;
    };
    let ground = world.terrain().surface(Face::PosY, block_coord(here.storage.x), block_coord(here.storage.z));
    let rest = DVec3::new(here.storage.x, ground as f64 + Stance::Standing.eye_offset(), here.storage.z);
    let dip = collision_box(rest, Stance::Standing, Face::PosY).min().y - ground as f64;
    let mut min_e = f64::MAX;
    let mut max_e = f64::MIN;
    for dx in [-16.0, -1.0, 0.0, 1.0, 16.0] {
        for dz in [-16.0, 0.0, 16.0] {
            for dy in [0.0, 0.001, 1.0] {
                let s = DVec3::new(rest.x + dx, rest.y + dy, rest.z + dz);
                let p = atlas.embed_storage(here.patch, s);
                let Some(back) = atlas.local(p) else { continue };
                if back.patch != here.patch {
                    continue;
                }
                let e = back.storage.y - s.y;
                min_e = min_e.min(e);
                max_e = max_e.max(e);
            }
        }
    }
    let mut s = rest;
    let patch = here.patch;
    let mut acc = 0.0;
    let mut broke = false;
    for _ in 0..20_000 {
        let p = atlas.embed_storage(patch, s);
        let Some(back) = atlas.local(p) else {
            broke = true;
            break;
        };
        if back.patch != patch {
            broke = true;
            break;
        }
        acc += back.storage.y - s.y;
        s = back.storage;
    }
    stamp(&format!(
        "{name} ground={ground} storage=({:.3},{:.3},{:.3}) feet_dip={dip:.3e} one_shot_y=[{min_e:.3e},{max_e:.3e}] drift_20k={acc:.3e} broke={broke}",
        rest.x, rest.y, rest.z
    ));
}

fn print_ulp() {
    for stance in [Stance::Standing, Stance::Sneaking] {
        let mut bad = Vec::new();
        for y in 0..8192 {
            let d = feet_dip(y as f64, stance);
            if d < 0.0 {
                bad.push((y, d));
            }
        }
        let name = match stance {
            Stance::Standing => "standing",
            Stance::Sneaking => "sneaking",
        };
        let show: Vec<_> = bad.iter().take(16).map(|(y, d)| format!("{y}:{d:.3e}")).collect();
        stamp(&format!("ulp {name} bad_in_0..8192={} first=[{}]", bad.len(), show.join(" ")));
    }
}

/// A stone floor whose top is the integer plane `surface`, wide enough for a one-second walk.
/// The flat world's own ground is already that plane at [`FLAT_HEIGHT`].
fn carve_surface(world: &mut World, origin: DVec3, surface: i32, len: i32) {
    world.ensure_around(origin);
    world.ensure_around(origin + DVec3::new(len as f64, 0.0, 0.0));
    if surface == FLAT_HEIGHT {
        return;
    }
    let stone = world.registry().id_by_label("rock").expect("rock");
    let (x0, z0) = (block_coord(origin.x), block_coord(origin.z));
    for x in (x0 - 2)..(x0 + len) {
        for z in (z0 - 2)..=(z0 + 2) {
            world.set_block(x, surface - 1, z, stone);
        }
    }
}

/// Planted exactly on an integer surface, then idle and a short walk. Reports a clip before any step,
/// a clip gravity introduces, and whether walking still moves.
fn plant_and_walk(world: &mut World, surface: i32, fps: f64) {
    let origin = DVec3::new(2_000.0 + surface as f64, surface as f64 + 2.0, 4.0);
    carve_surface(world, origin, surface, 48);
    let feet = surface as f64;
    let eye = feet + Stance::Standing.eye_offset();
    let mut player = Player::new(DVec3::new(origin.x, eye, origin.z));
    player.snap_up(DVec3::Y);
    player.motion = Motion::Walking { velocity: DVec3::ZERO, on_ground: true };
    let planted = world.collides(&player.aabb());
    let dt = (1.0 / fps) as f32;
    let idle = MoveInput::keys(0.0, 0.0, false, false, false);
    let mut idle_overlap = 0u32;
    let mut min_gap = f64::MAX;
    for _ in 0..(fps as i32) {
        if world.collides(&player.aabb()) {
            idle_overlap += 1;
        }
        min_gap = min_gap.min(ground_gap(world, &player, player.position));
        physics(world, &mut player, &idle, dt);
    }
    player.orientation.yaw = 0.0;
    let forward = MoveInput::keys(0.0, 1.0, false, false, false);
    let x0 = player.position.x;
    let mut walk_overlap = 0u32;
    for _ in 0..(fps as i32) {
        if world.collides(&player.aabb()) {
            walk_overlap += 1;
        }
        min_gap = min_gap.min(ground_gap(world, &player, player.position));
        physics(world, &mut player, &forward, dt);
    }
    let moved = player.position.x - x0;
    stamp(&format!(
        "plant surface={surface} fps={fps} planted_overlap={planted} idle_overlaps={idle_overlap} walk_overlaps={walk_overlap} moved={moved:.3} min_gap={min_gap:.3e} on_ground={}",
        player.on_ground()
    ));
}

fn drop_onto(world: &mut World, surface: i32, fps: f64) {
    let origin = DVec3::new(8_000.0 + surface as f64, surface as f64 + 6.0, 4.0);
    carve_surface(world, origin, surface, 48);
    let mut player = Player::new(origin);
    player.snap_up(DVec3::Y);
    let dt = (1.0 / fps) as f32;
    let idle = MoveInput::keys(0.0, 0.0, false, false, false);
    let mut overlaps = 0u32;
    let mut min_gap = f64::MAX;
    let mut landed = false;
    for _ in 0..(fps as i32 * 3) {
        if world.collides(&player.aabb()) {
            overlaps += 1;
        }
        if player.on_ground() {
            landed = true;
            min_gap = min_gap.min(ground_gap(world, &player, player.position));
        }
        physics(world, &mut player, &idle, dt);
    }
    let gap = if min_gap > 1.0e8 { f64::NAN } else { min_gap };
    stamp(&format!(
        "drop surface={surface} fps={fps} landed={landed} overlaps={overlaps} min_gap={gap:.3e} eye_y={:.6}",
        player.position.y
    ));
}

fn bad_walk(report: &Report) -> bool {
    report.overlaps > 0 || report.stalls > 0 || !report.landed
}

/// Short measurement: integer-feet rounding, gravity on a flat floor, chart round-trip, and a few
/// seconds of walking with and without streaming. Panics if a walk sticks or never lands.
#[test]
#[ignore]
fn ground_contact_probe() {
    print_ulp();
    stamp("flat gravity");
    let mut flat = flat_world(false);
    for surface in [FLAT_HEIGHT, 40, 255, 256, 511, 512] {
        plant_and_walk(&mut flat, surface, 60.0);
        drop_onto(&mut flat, surface, 60.0);
        drop_onto(&mut flat, surface, 240.0);
    }
    drop_onto(&mut flat, 40, 1000.0);
    plant_and_walk(&mut flat, 40, 1000.0);

    stamp("flat stream");
    let mut flat = flat_world(false);
    flat.set_view_distances(2, 1);
    let mut player = place_flat(&flat);
    let flat_stream = walk(&mut flat, &mut player, 4.0, 60.0, true, "flat-stream-2x1");

    stamp("spawn chart");
    let mut spawn = chart_world(SPAWN_SEED, false);
    let hint = spawn.chart_spawn().expect("charted spawn");
    measure_roundtrip(&spawn, "spawn", hint);
    let mut player = place_chart(&spawn, hint);
    let spawn_sync = walk(&mut spawn, &mut player, 3.0, 60.0, false, "spawn-sync");
    let mut player = place_chart(&spawn, hint);
    let spawn_fast = walk(&mut spawn, &mut player, 2.0, 240.0, false, "spawn-sync-240");
    spawn.set_view_distances(2, 1);
    let mut player = place_chart(&spawn, hint);
    let spawn_stream = walk(&mut spawn, &mut player, 4.0, 60.0, true, "spawn-stream-2x1");

    stamp("owner chart");
    let owner_hint = DVec3::new(-19.1, 390.8, -0.6);
    let mut owner = chart_world(OWNER_SEED, false);
    measure_roundtrip(&owner, "owner", owner_hint);
    let mut player = place_chart(&owner, owner_hint);
    let owner_sync = walk(&mut owner, &mut player, 3.0, 60.0, false, "owner-sync");
    owner.set_view_distances(2, 1);
    let mut player = place_chart(&owner, owner_hint);
    let owner_stream = walk(&mut owner, &mut player, 4.0, 60.0, true, "owner-stream-2x1");

    let reports = [
        ("flat-stream", &flat_stream),
        ("spawn-sync", &spawn_sync),
        ("spawn-sync-240", &spawn_fast),
        ("spawn-stream", &spawn_stream),
        ("owner-sync", &owner_sync),
        ("owner-stream", &owner_stream),
    ];
    let bad: Vec<_> = reports.iter().filter(|(_, r)| bad_walk(r)).map(|(n, _)| *n).collect();
    assert!(bad.is_empty(), "stuck or never landed: {}", bad.join(", "));
}

fn parse_sites() -> Vec<&'static str> {
    let raw = std::env::var("WALK_SITES").unwrap_or_else(|_| "flat,spawn,owner".to_string());
    let mut sites = Vec::new();
    for part in raw.split(',') {
        match part.trim() {
            "flat" => sites.push("flat"),
            "spawn" => sites.push("spawn"),
            "owner" => sites.push("owner"),
            "" => {}
            other => panic!("unknown WALK_SITES entry {other}"),
        }
    }
    sites
}

fn parse_views() -> Vec<(i32, i32)> {
    let raw = std::env::var("WALK_VIEWS").unwrap_or_else(|_| "2,1;16,5".to_string());
    let mut views = Vec::new();
    for part in raw.split(';') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (h, v) = part.split_once(',').unwrap_or_else(|| panic!("WALK_VIEWS entry {part} wants h,v"));
        views.push((h.trim().parse().expect("view h"), v.trim().parse().expect("view v")));
    }
    views
}

/// Minutes of walking on flat ground, at spawn, and at the owner's seed, with streaming.
/// Fails on a step that starts inside a solid, or on forward input that does not move with no wall ahead.
#[test]
#[ignore]
fn walking_on_the_ground_never_gets_stuck() {
    let secs = env_f64("WALK_SECS", 180.0);
    let hz = env_f64("WALK_HZ", 60.0);
    let lod = env_flag("WALK_LOD", true);
    let strict = env_flag("WALK_STRICT", true);
    let mut bad = Vec::new();
    for site in parse_sites() {
        for (h, v) in parse_views() {
            let tag = format!("{site}-{h}x{v}");
            stamp(&format!("begin {tag} secs={secs} hz={hz} lod={lod}"));
            let (mut world, mut player) = match site {
                "flat" => {
                    let world = flat_world(lod);
                    let player = place_flat(&world);
                    (world, player)
                }
                "spawn" => {
                    let world = chart_world(SPAWN_SEED, lod);
                    let hint = world.chart_spawn().expect("charted spawn");
                    let player = place_chart(&world, hint);
                    (world, player)
                }
                "owner" => {
                    let world = chart_world(OWNER_SEED, lod);
                    let player = place_chart(&world, DVec3::new(-19.1, 390.8, -0.6));
                    (world, player)
                }
                _ => unreachable!(),
            };
            world.set_view_distances(h, v);
            let report = walk(&mut world, &mut player, secs, hz, true, &tag);
            if strict && bad_walk(&report) {
                bad.push(tag);
            }
        }
    }
    assert!(bad.is_empty(), "stuck or never landed: {}", bad.join(", "));
}
