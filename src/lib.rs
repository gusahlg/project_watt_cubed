//! project_watt_cubed — a small voxel prototype.
//!
//! The crate is split into focused modules so each concern can evolve on its own:
//!
//! - [`app`] — owns the window and the menu/play/mods state machine.
//! - [`game`] — the in-world state (world, player, physics, console) and its frame.
//! - [`block`] — the element/block core: elements, compositions, property
//!   derivation, reactions, and the block registry the world indexes into.
//! - [`world`] — the infinite streamed chunk field, terrain generation, collision,
//!   and rendering.
//! - [`player`] — player state and the camera derived from it.
//! - [`stash`] — the elements a player carries (core-owned; mods present it).
//! - [`input`] — keyboard movement and mouse look.
//! - [`interact`] — the aim raycast that turns looking into breaking blocks.
//! - [`console`] — the in-game console / chat line and its text input.
//! - [`command`] — parsing and dispatch for console commands.
//! - [`menu`] — the start menu and mod menu screens.
//! - [`mods`] — the runtime-toggleable mod system and the default inventory mod.
//! - [`net`] — multiplayer: the authoritative server and the client connection.
//! - [`paths`] — data/config roots (worlds, settings, session, mod choices).
//! - [`save`] — saving and loading worlds.
//! - [`settings`] — persistent graphics settings (settings menu + `/gfx`).
//! - [`sim`] — the fixed-timestep seam for registered simulation systems.
//! - [`math`] — geometry shared across systems (the [`Aabb`](math::Aabb) and
//!   [`Bounded`](math::Bounded) trait).
//! - [`render`] — the [`Render`](render::Render) trait for drawable things.
//! - [`macros`] — declarative macros that generate repetitive code.
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
