//! Test fixtures shared by the world's test modules: start worlds, chunk and section states,
//! and the knobs and pacing of the headless benches.

use std::sync::Arc;
use std::time::{Duration, Instant};

use voxel_engine::DVec3;

use super::chunk::Chunk;
use super::generation::WorldgenKind;
use super::{Loaded, MeshState, SectionState, World};
use crate::block::registry::AIR;
use crate::render_config::RenderConfig;

/// The round (diffusion) start world of `seed`, nothing generated yet.
pub(in crate::world) fn round_world(seed: i64, render: RenderConfig) -> World {
    World::with_kind(seed, render, WorldgenKind::Diffusion, false)
}

/// Seed 42's round start world at view `(h, v)` with `render`, its spawn slab landed, and the
/// spawn eye.
pub(in crate::world) fn spawned_round_world(render: RenderConfig, h: i32, v: i32) -> (World, DVec3) {
    let mut world = round_world(42, render);
    world.set_view_distances(h, v);
    let spawn = world.chart_spawn().expect("the start world is charted");
    world.prepare_around(spawn);
    world.drive_spawn_ready();
    (world, spawn)
}

/// A settled born-air chunk at `(cx, cy, cz)`.
pub(in crate::world) fn air_loaded(cx: i32, cy: i32, cz: i32) -> Loaded {
    Loaded::new(Arc::new(Chunk::from_uniform(cx, cy, cz, AIR)), MeshState::Air, 0)
}

/// A resident section with no slabs: what a headless upload lands as.
pub(in crate::world) fn ready_section() -> SectionState {
    SectionState::Ready { meshes: Vec::new(), cages: Vec::new(), last_style: None }
}

/// The environment variable `name` parsed, else `default`.
pub(in crate::world) fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

/// Sleep out the rest of a frame `period` that began at `start`.
pub(in crate::world) fn pace(start: Instant, period: Duration) {
    if let Some(rest) = period.checked_sub(start.elapsed()) {
        std::thread::sleep(rest);
    }
}
