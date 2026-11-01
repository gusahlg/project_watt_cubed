//! project_watt_cubed — a voxel game on `voxel-engine`: the PWC runtime.
//!
//! The game is a library: [`run`] starts it with a [`modules::GameBuild`] naming the module
//! packages compiled into this executable. Modules compile against the `pwc-module-api` crate
//! (`crates/pwc-module-api`), which re-exports the module-facing part of this crate.
//!
//! One module per subsystem. Worldgen, save bytes, protocol bytes, and mesh
//! vertices are bit-identical unless a change explicitly says otherwise.
#[cfg(test)]
mod alloc_count;
#[cfg(test)]
extern crate self as pwc_module_api;
#[cfg(test)]
#[global_allocator]
static ALLOC: alloc_count::Counting = alloc_count::Counting;
pub mod app;
pub mod audio;
pub(crate) mod avatar;
pub(crate) mod benchmark;
pub mod block;
pub(crate) mod camera;
pub(crate) mod coord;
pub mod derived;
pub mod gravity;
pub(crate) mod frame_snapshot;
pub(crate) mod game;
pub mod harness;
pub mod headless;
pub(crate) mod hash;
#[cfg(feature = "dev-tools")]
pub use hash::splitmix_next;
pub(crate) mod ident;
pub mod input;
pub(crate) mod interact;
pub(crate) mod macros;
pub mod math;
pub mod mechanics;
pub mod screen;
pub(crate) mod minimap;
pub mod modules;
pub mod net;
pub mod paths;
pub mod player;
pub(crate) mod play;
pub(crate) mod presence;
pub mod render_config;
pub(crate) mod save;
pub(crate) mod sched;
pub mod session;
pub mod settings;
pub mod sim;
pub mod space;
pub mod sky;
pub mod inventory;
pub mod ui;
pub mod world;

/// The renderer, re-exported for modules (colours, math types).
pub use voxel_engine as engine;
/// The material law, re-exported for modules (configurations, observations, presentation).
pub use material;

/// Run the game with the module packages of `build` (`GameBuild::vanilla()` for the bare game).
/// Returns when the window closes.
pub fn run(build: modules::GameBuild) {
    let args: Vec<String> = std::env::args().collect();
    if let Some(code) = modules::dispatch(&build, &args) {
        std::process::exit(code);
    }
    // The material palette is searched once per process (~0.5 s). Start it now, off the main thread,
    // so the first world finds it ready instead of stalling its first frame.
    let _ = std::thread::Builder::new()
        .name("palette".into())
        .spawn(|| {
            world::terrain::palette::current();
        });
    app::App::new(&build).run();
}
