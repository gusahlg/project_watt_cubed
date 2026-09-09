//! project_watt_cubed — a voxel game on `voxel-engine`.
//!
//! One module per subsystem. Worldgen, save bytes, protocol bytes, and mesh
//! vertices are bit-identical unless a change explicitly says otherwise.
pub mod app;
pub mod audio;
pub mod avatar;
pub mod benchmark;
pub mod block;
pub mod camera;
pub mod command;
pub mod console;
pub mod coord;
pub mod derived;
pub mod frame_snapshot;
pub mod game;
pub mod harness;
pub mod hash;
pub mod ident;
pub mod input;
pub mod interact;
pub mod macros;
pub mod math;
pub mod menu;
pub mod minimap;
pub mod mods;
pub mod net;
pub mod paths;
pub mod player;
pub mod presence;
pub mod render;
pub mod render_config;
pub mod save;
pub mod sched;
pub mod session;
pub mod settings;
pub mod sim;
pub mod sky;
pub mod stash;
pub mod ui;
pub mod world;
