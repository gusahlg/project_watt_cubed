//! project_watt_cubed — a small voxel prototype.
//!
//! The crate is split into focused modules so each concern can evolve on its own:
//!
//! - [`app`] — owns the window and runs the update/draw loop.
//! - [`world`] — voxels, chunks, terrain generation, collision, and rendering.
//! - [`player`] — player state and the camera derived from it.
//! - [`input`] — keyboard movement and mouse look.
//! - [`console`] — the in-game console / chat line and its text input.
//! - [`command`] — parsing and dispatch for console commands.
//! - [`math`] — geometry shared across systems (the [`Aabb`](math::Aabb) and
//!   [`Bounded`](math::Bounded) trait).
//! - [`render`] — the [`Render`](render::Render) trait for drawable things.
//! - [`macros`] — declarative macros that generate repetitive code.
pub mod app;
pub mod command;
pub mod console;
pub mod input;
pub mod macros;
pub mod math;
pub mod player;
pub mod render;
pub mod world;
