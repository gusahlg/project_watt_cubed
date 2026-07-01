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
//! - [`input`] — keyboard movement and mouse look.
//! - [`interact`] — the aim raycast that turns looking into breaking blocks.
//! - [`console`] — the in-game console / chat line and its text input.
//! - [`command`] — parsing and dispatch for console commands.
//! - [`menu`] — the start menu and mod menu screens.
//! - [`mods`] — the runtime-toggleable mod system and the default inventory mod.
//! - [`net`] — multiplayer: the authoritative server and the client connection.
//! - [`save`] — saving and loading worlds.
//! - [`sim`] — the fixed-timestep simulation seam (thermal/electrical, inert for now).
//! - [`math`] — geometry shared across systems (the [`Aabb`](math::Aabb) and
//!   [`Bounded`](math::Bounded) trait).
//! - [`render`] — the [`Render`](render::Render) trait for drawable things.
//! - [`macros`] — declarative macros that generate repetitive code.
pub mod app;
pub mod block;
pub mod command;
pub mod console;
pub mod game;
pub mod input;
pub mod interact;
pub mod macros;
pub mod math;
pub mod menu;
pub mod mods;
pub mod net;
pub mod player;
pub mod render;
pub mod save;
pub mod sim;
pub mod world;
