//! # The PWC mod API
//!
//! The one crate a PWC mod depends on. A mod package (`.pwcmod`, managed by the PWC package
//! manager) is a Rust library whose `src/lib.rs` defines
//!
//! ```ignore
//! pub fn register(registrar: &mut pwc_mod_api::ModRegistrar) {
//!     registrar.add(MyMod::default());
//! }
//! ```
//!
//! and implements [`Mod`] for the values it adds. The PWC builder generates the executable that
//! calls every package's `register` in dependency order (see [`GameBuild`]).
//!
//! **Versioning.** This crate's version is the *mod API version* (`pwc-api` in `mod.toml`).
//! Everything re-exported here is the API surface; breaking it requires a major version bump.
//! The [`prelude`] covers what most mods need; the module re-exports give access to the game's
//! subsystems a mod may read or drive.
//!
//! **Rules of the game** (see the package manager's POLICY.md): matter is not a mod — blocks are
//! configurations of elements under one law; names and looks are presentation and never feed
//! back into the law, world generation, saves or the network; anything that affects the world is
//! deterministic; in multiplayer the server is the authority; hooks run at frame/event
//! granularity, never per voxel.

pub use project_watt_cubed::modding::{
    annotate_setting, forced_off_marker, ChoicesFlush, GameBuild, Group, Knob, Mod, ModContext, ModDescriptor,
    ModRegistrar, Mods, VisualMask, ESSENTIALS, ESSENTIALS_GROUP,
};

/// The game's subsystems, as far as mods may use them.
pub use project_watt_cubed::{block, derived, engine, input, material, menu, net, player, render_config, session, settings, sim, stash, ui, world};

/// The mod API version this crate provides (`pwc-api` requirements are checked against it).
pub const API_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The names almost every mod uses.
pub mod prelude {
    pub use crate::block::{BlockId, BlockRegistry};
    pub use crate::player::Player;
    pub use crate::ui::{Anchor, HudElement, Line, Panel, Role, Row};
    pub use crate::world::World;
    pub use crate::{Group, Knob, Mod, ModContext, ModRegistrar, ESSENTIALS};
}
