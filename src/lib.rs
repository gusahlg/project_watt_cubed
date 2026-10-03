//! project_watt_cubed — a voxel game on `voxel-engine`: the PWC runtime.
//!
//! The game is a library: [`run`] starts it with a [`modding::GameBuild`] naming the mod
//! packages compiled into this executable. Mods compile against the `pwc-mod-api` crate
//! (`crates/pwc-mod-api`), which re-exports the mod-facing part of this crate.
//!
//! One module per subsystem. Worldgen, save bytes, protocol bytes, and mesh
//! vertices are bit-identical unless a change explicitly says otherwise.
#[cfg(test)]
mod alloc_count;
#[cfg(test)]
#[global_allocator]
static ALLOC: alloc_count::Counting = alloc_count::Counting;
pub mod app;
pub(crate) mod audio;
pub(crate) mod avatar;
pub(crate) mod benchmark;
pub mod block;
pub(crate) mod camera;
pub(crate) mod command;
pub(crate) mod console;
pub(crate) mod coord;
pub mod derived;
pub(crate) mod frame_snapshot;
pub(crate) mod game;
pub mod harness;
pub(crate) mod hash;
pub(crate) mod ident;
pub mod input;
pub(crate) mod interact;
pub(crate) mod macros;
pub(crate) mod math;
pub mod menu;
pub(crate) mod minimap;
pub mod modding;
pub mod net;
pub mod paths;
pub mod player;
pub(crate) mod presence;
pub mod render_config;
pub(crate) mod save;
pub(crate) mod sched;
pub mod session;
pub mod settings;
pub mod sim;
pub mod space;
pub(crate) mod sky;
pub mod stash;
pub mod ui;
pub mod world;

/// The renderer, re-exported for mods (colours, math types).
pub use voxel_engine as engine;
/// The material law, re-exported for mods (configurations, observations, presentation).
pub use material;

/// Run the game with the mod packages of `build` (`GameBuild::vanilla()` for the bare game).
/// Returns when the window closes.
pub fn run(build: modding::GameBuild) {
    app::App::new(&build).run();
}
