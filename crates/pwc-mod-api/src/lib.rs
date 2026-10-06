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
//! Everything re-exported here is the API surface; a break old mods cannot compile through
//! requires a major version bump. The [`prelude`] covers what most mods need; the module
//! re-exports give access to the game's subsystems a mod may read or drive.
//!
//! # 2.1.0
//!
//! Additive on 2.0.0 (`^2.0` still matches):
//!
//! - [`Action::held`] keeps an action on for every frame its chord is down. Menus do not sample
//!   it. An `Action` literal names `held`; `repeat` is ignored while it is set.
//! - [`Mod::on_game_event`] reports a block edit, a step, a swing, a menu click, the voice-test
//!   cue, and entering or leaving a world. [`Mod::on_audio`] runs every frame, menus included,
//!   with the listener and the peer roster.
//! - [`audio`] plays catalog cues, reads the mix, and opens voice sessions and the microphone.
//!   The device, the codecs and the mixer stay in the core.
//!
//! # 2.0.0
//!
//! Breaking changes from 1.x:
//!
//! - The `stash` module is gone. A player carries an [`inventory::Inventory`] at `Player::inventory`.
//!   Save bytes are unchanged.
//! - `Mod::held` is [`Mod::tool`]. `None` means no tool: a primary action breaks the block into
//!   the inventory.
//! - Mods declare input with [`Mod::actions`]. [`ModContext::action`] reports which fired this
//!   frame and [`ModContext::wheel`] is the signed scroll. `hotbar_key`, `hotbar_cycle` and
//!   `toggle_inventory` are gone. A core binding wins a chord clash.
//! - [`ModContext`] is `#[non_exhaustive]`. Tests build it with [`ModContext::new`] and
//!   [`ModContext::set_action`].
//! - [`Mod::on_tool_used`] reports a [`ToolUse`]. The core draws nothing for it.
//! - `GameplayEvent` no longer has hand, number-key, wheel or inventory events.
//!
//! There is no controls screen. [`Action::label`] is what a future one will show; rebinding is
//! left for later.
//!
//! **Rules of the game** (see the package manager's POLICY.md): matter is not a mod — blocks are
//! configurations of elements under one law; names and looks are presentation and never feed
//! back into the law, world generation, saves or the network; anything that affects the world is
//! deterministic; in multiplayer the server is the authority; hooks run at frame/event
//! granularity, never per voxel.

pub use project_watt_cubed::modding::{
    annotate_setting, forced_off_marker, Action, ActionSet, ChoicesFlush, Command, CommandContext, GameBuild, Group,
    Knob, Mod, ModContext, ModDescriptor, ModRegistrar, Mods, ToolUse, VisualMask, ESSENTIALS, ESSENTIALS_GROUP,
};

/// Catalog cues, the mix, voice sessions and the microphone. No device, codec or mixer type.
pub mod audio {
    pub use project_watt_cubed::audio::{
        AudioApi, AudioBench, AudioView, BlockSound, CapturedFrame, GameEvent, ModFrame, ModLink, PeerAudio, Play,
    };
}

/// The game's subsystems, as far as mods may use them.
pub use project_watt_cubed::{
    block, derived, engine, gravity, input, material, math, menu, net, player, render_config, session, settings, sim,
    inventory, sky, ui, world,
};

/// The mod API version this crate provides (`pwc-api` requirements are checked against it).
pub const API_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The names almost every mod uses.
pub mod prelude {
    pub use crate::block::{BlockId, BlockRegistry};
    pub use crate::player::Player;
    pub use crate::ui::{Anchor, HudElement, Line, Panel, Role, Row};
    pub use crate::world::World;
    pub use crate::{Command, CommandContext, Group, Knob, Mod, ModContext, ModRegistrar, ESSENTIALS};
}
