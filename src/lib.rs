//! project_watt_cubed — a voxel game on `voxel-engine`.
//!
//! One module per subsystem. Worldgen, save bytes, protocol bytes, and mesh
//! vertices are bit-identical unless a change explicitly says otherwise.
pub mod app;
pub(crate) mod audio;
pub(crate) mod avatar;
pub(crate) mod benchmark;
pub(crate) mod block;
pub(crate) mod camera;
pub(crate) mod command;
pub(crate) mod console;
pub(crate) mod coord;
pub(crate) mod derived;
pub(crate) mod frame_snapshot;
pub(crate) mod game;
pub mod harness;
pub(crate) mod hash;
pub(crate) mod ident;
pub(crate) mod input;
pub(crate) mod interact;
pub(crate) mod macros;
pub(crate) mod math;
pub(crate) mod menu;
pub(crate) mod minimap;
pub(crate) mod mods;
pub mod net;
pub mod paths;
pub(crate) mod player;
pub(crate) mod presence;
pub(crate) mod render;
pub(crate) mod render_config;
pub(crate) mod save;
pub(crate) mod sched;
pub(crate) mod session;
pub(crate) mod settings;
pub(crate) mod sim;
pub(crate) mod sky;
pub(crate) mod stash;
pub(crate) mod ui;
pub(crate) mod world;
