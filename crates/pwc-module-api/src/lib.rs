//! # The PWC module API
//!
//! The one crate a PWC module depends on. A module package (`.pwcmodule`, managed by the PWC package
//! manager) is a Rust library whose `src/lib.rs` defines
//!
//! ```ignore
//! pub fn register(registrar: &mut pwc_module_api::ModuleRegistrar) {
//!     registrar.add(MyModule::default());
//! }
//! ```
//!
//! and implements [`Module`] for the values it adds. The PWC builder generates the executable that
//! calls every package's `register` in dependency order (see [`GameBuild`]).
//!
//! **Versioning.** This crate's version is the *module API version* (`pwc-api` in `module.toml`).
//! Everything re-exported here is the API surface; a break old modules cannot compile through
//! requires a major version bump. The [`prelude`] covers what most modules need; the module
//! re-exports give access to the game's subsystems a module may read or drive.
//!
//! # 3.4.0
//!
//! Mods are modules (a rename; `^3.0` requirements still match). The crate is `pwc-module-api`
//! (lib `pwc_module_api`); `pwc-mod-api` (lib `pwc_mod_api`) stays as a shim that re-exports it.
//! - `Mod` is [`Module`], `ModContext` is [`ModuleContext`], `ModDescriptor` is [`ModuleDescriptor`],
//!   `ModRegistrar` is [`ModuleRegistrar`], `audio::{ModFrame, ModLink}` are
//!   [`audio::ModuleFrame`] and [`audio::ModuleLink`], `PackageKind::Mod` is [`PackageKind::Module`]
//!   and `GameBuild::with_mod` is [`GameBuild::with_module`]. The old names remain as hidden
//!   aliases (not deprecated), so a package written for 3.3 compiles unchanged.
//! - Fields cannot be aliased. `ModuleContext::mod_ui` is [`ModuleContext::module_ui`] and
//!   `headless::ConnectError::mods_denied` is [`headless::ConnectError::modules_denied`]; the old
//!   fields stay as hidden copies of the new ones, so packages that read them still compile.
//!   One break: `headless::JoinOptions::mods` is [`headless::JoinOptions::modules`] (it shipped in
//!   3.3, used only by `pwc.exploraswarm` 0.1.0, which 0.1.1 replaces). Methods keep their names.
//! - Package manifests are `module.toml` with `kind = "module"` (the package manager still
//!   reads `mod.toml` and `kind = "mod"`). Settings `mod_hz` and `mod_logic` are `module_hz` and
//!   `module_logic`; the old keys still load.
//!
//! # 3.3.0
//!
//! Additive on 3.2 (`^3.0` still matches):
//! - [`headless::HeadlessClient`] is one player on a server with the game's world mirror and
//!   physics and none of the window. [`HeadlessClient::connect`](headless::HeadlessClient::connect)
//!   blocks until a frame can run; [`step`](headless::HeadlessClient::step) polls, moves, acts
//!   and sends. The caller names the held material for a place or a tool use. `target` and `ray`
//!   use the game's raycast; `cell_center` embeds a round world's storage cell in world space.
//! - [`ModRegistrar::cli`] registers a command-line entry point ([`Cli`], [`CliContext`]).
//!   `help` and `--commands` list the registered commands. Any other first argument starts the game.
//!
//! # 3.2.0
//!
//! Additive native simulation providers on 3.1 (`^3.0` still matches):
//! - [`Mod::reaction_model`] offers `sim::model::ModelSpec { id, version, create }`.
//!   The first active provider wins; no provider means no automatic or held-material reactions.
//! - `sim::model::Model: Send` owns wakeups, budgeted stepping, optional tool interactions,
//!   counters, samples, and saved dynamical state. `sim::reactions::CellStore` provides the
//!   world adapter and atomic pair admission. The common material catalog remains in core.
//! - `World::configure_simulation`, `simulation_model_id`, `simulation_stats`,
//!   `simulation_sample`, `simulation_cell`, and `perturb_simulation` support selection and
//!   laboratory inspection/input. Switching or disabling a model preserves its saved payload.
//! - The original selective-transfer queue/tool law is the separate `pwc.physics-selective`
//!   package. Experimental wave, packet, reservoir and rotor packages are independent providers.
//! - Protocol 18 adds the selected model id/version fingerprint (informational: only the
//!   server runs the model, so it never refuses a join); server configuration accepts an
//!   explicit factory. Save v10 uses reserved opaque records without a binary format bump.
//!   Experimental per-cell fields and local perturbations are not replicated to clients yet.
//!
//! # 3.1.0
//!
//! The emergent cosmos (worldgen 11): every seed grows its own bodies. Additions only.
//! - `world::terrain::cosmos::Body` carries `traits` (`Traits`: rank, form, parent, partner, core,
//!   mass, gravity, heat, glow, air top, temperature, wetness, life, suite, name) and `name()`;
//!   `Traits::class()` is a label of what the body is ("start world", "binary cube", "living moon",
//!   ...). `Cosmos::suites` and `Cosmos::suite` give each body's minerals.
//! - `Cosmos::new` and `Cosmos::with_deep` take the full 64-bit seed.
//! - `Body::kind` (`Kind`) stays one release, derived from the traits (`Traits::legacy_kind`): a
//!   body with id 0 is `Home`, a satellite `Moon`, a cube partner `Twin`, a shell `Hollow`, a free
//!   core `Ember`, anything else `Verdant`. The Twins, Verdance, the Hollow and the Ember are
//!   outcomes now, not fixtures.
//!
//! # 3.0.0
//!
//! The mods-v3 release: chat, commands, every menu and the mod list are mods, the build decides
//! what is installed, and every tunable lives in one registry. Breaking changes from 2.x:
//!
//! **Builds and the installed set** (the build decides; nothing is switched in the game)
//! - Mods have no on/off state. `Group`, `ESSENTIALS`/`ESSENTIALS_GROUP`, `ModRegistrar::declare_group`,
//!   `ModRegistrar::add_disabled`, `Knob`, `Mod::knobs`/`step_knob`/`save_choice_state`/
//!   `load_choice_state`, `Mod::on_enable`/`on_disable`, `ChoicesFlush`, `annotate_setting`,
//!   `forced_off_marker` and `mods.cfg` are gone.
//! - `Mod::description` and `Mod::group` are gone: a package's `mod.toml` is what menus show.
//!   [`Mod::name`] defaults to the id and only names old saves and logs.
//! - The core can *suspend* a package for a session (a server that refuses it; benchmarks pin
//!   it). A suspended mod's hooks do not run. Nothing is saved and there is no UI for it.
//!   [`VisualMask::strips`] says whether a lane's visual group is unprovided.
//! - The host type `Mods` is no longer part of the API. Tests drive packages through
//!   [`testing::Harness`].
//!
//! **Options** (one registry for every tunable)
//! - [`ModRegistrar::option`] declares a package's option ([`settings::OptionSpec`]: toggle,
//!   choice, percent or float, its label, settings page, whether it applies to the next world, an
//!   optional legacy key) and returns an [`settings::OptionId`]. [`Mod::on_options`] delivers the
//!   values at registration, after `settings.cfg` loads, and on every change; reads are by index.
//!   Values persist in `settings.cfg` as `<package-id>.<key>=`.
//! - [`settings::OptionsView`] lists the core's settings (owner `"core"`) and every package's
//!   options through one interface. [`GameContext::options_mut`] and the screens'
//!   `ScreenContext::options_mut` change them.
//! - The core settings `mod_hud`, `simulation` and `voice_enabled` are gone (`WATT_SIMULATION=0`
//!   is the bench override; Voice Chat is `pwc.proximity-chat`'s option).
//!
//! **Screens** (every menu is a mod, see [`screen`])
//! - [`screen::Screen`] reads `MenuInput` and `ScreenFacts` and answers a `ScreenOutcome`
//!   (`Stay`, `Back`, `Push`, `Open(entry)`, `Request(AppRequest)`); it draws [`screen::UiElement`]s.
//! - Slots [`Mod::root_screen`] (out of a world; replaces `start_screen` and `menu_theme`) and
//!   [`Mod::pause_screen`] (Esc in a world). [`ModRegistrar::screen_entry`] offers a screen on the
//!   main menu and/or the pause screen. Without a root screen the game enters the newest world and
//!   Esc saves and quits.
//! - The `menu` module is gone: the menu framework, the default look and the text widgets
//!   (`TextInput`, `EditBuf`, `Ring`) live in the `pwc.ui-kit` library package.
//!
//! **Frame input, chat and messages**
//! - [`Mod::on_frame`] runs every in-world frame with a [`FrameContext`]: immediate actions
//!   ([`Action::immediate`], sampled even with mod logic off; every `Action` literal names it),
//!   the keyboard capture ([`FrameContext::capture_text`], [`TextFrame`]) and the game state in a
//!   [`GameContext`] (player, world, sky, settings, options, `send_chat`, `notice`). The core follows
//!   up on what a hook changed.
//! - [`Mod::on_message`] gets chat lines, joins, leaves and the core's notices ([`Message`],
//!   [`Notice`]).
//! - `Command`, `CommandContext`, `Mod::command`/`run_command`/`commands` and `Mod::on_toggle_fly`
//!   are gone: `pwc.chat`, `pwc.commands` and `pwc.dev-toolkit` provide chat, commands and flight.
//!
//! **HUD**
//! - [`Mod::hud`] takes [`HudFacts`] (screen, frame rate, ping, players, loading, the link, the HUD
//!   mode, the UI scale, cruise) and is asked in every HUD mode, Off included. The core draws no HUD
//!   text: `pwc.game-ui` draws the reticle and readouts.
//!
//! **Build metadata** (from 2.2, unchanged)
//! - [`PackageInfo`], [`PackageKind`], [`BuildInfo`], [`GameBuild::from_static`],
//!   [`ModRegistrar::build`] and [`bundles_of`].
//!
//! # 2.2.0
//!
//! Additive on 2.1.0 (`^2.0` still matches):
//!
//! - The build lists every package compiled in, of every kind: one [`PackageInfo`] each (id,
//!   name, version, description, [`PackageKind`], direct dependencies, and the entry point for
//!   mods). [`ModRegistrar::build`] gives the whole list to any package as a [`BuildInfo`], so a
//!   package can show what is installed without the core naming anything.
//! - [`bundles_of`] works out which bundles include a package, from the bundles' own dependency
//!   lists.
//! - Generated builds use [`GameBuild::from_static`]. [`ModDescriptor`] and
//!   [`GameBuild::with_mod`] still work and make a mod package with no description and no
//!   dependencies. [`ModRegistrar::package`] returns the [`PackageInfo`], which has the same `id`,
//!   `name` and `version` fields.
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
//! **Rules of the game** (see the package manager's POLICY.md): blocks are canonical element
//! configurations, with no named special cases. Optional native models supply universal laws;
//! names and looks never feed back into world state. World-affecting code is deterministic and
//! the multiplayer server is authoritative. Frame/event hooks remain cheap; budgeted model ticks
//! handle local simulation work and document its cost.

pub use project_watt_cubed::modules::{
    Action, ActionSet, BuildInfo, Cli, CliContext, GameBuild, Module, ModuleContext, ModuleDescriptor, ModuleRegistrar,
    PackageInfo, PackageKind, ToolUse, VisualMask,
};

/// The pre-3.4 names of the module types, so packages written before the rename still compile.
#[doc(hidden)]
pub use project_watt_cubed::modules::{
    Module as Mod, ModuleContext as ModContext, ModuleDescriptor as ModDescriptor, ModuleRegistrar as ModRegistrar,
};

pub mod testing;
/// The frame hook, text capture, the chat send queue, the message stream and the HUD facts
/// (track B of 3.0).
pub use project_watt_cubed::modules::{
    Channel, FrameContext, GameContext, HudFacts, Message, Notice, NoticeLevel, Notices, TextFrame,
};

/// Every bundle in `packages` that includes the package `id`, in the order `packages` lists them
/// (for a build: registration order). A bundle includes its own dependencies, and everything a
/// bundle among them includes. A module's dependencies are not members of the bundles that list the
/// module, and a bundle is not its own member.
///
/// ```
/// use pwc_module_api::{bundles_of, PackageInfo, PackageKind};
/// const fn package(id: &'static str, kind: PackageKind, dependencies: &'static [&'static str]) -> PackageInfo {
///     PackageInfo { id, name: id, version: "1.0.0", description: "", kind, dependencies, register: None }
/// }
/// let packages = [
///     package("a.chat", PackageKind::Library, &[]),
///     package("a.talk", PackageKind::Bundle, &["a.chat"]),
///     package("a.all", PackageKind::Bundle, &["a.talk"]),
/// ];
/// let ids: Vec<&str> = bundles_of(&packages, "a.chat").iter().map(|b| b.id).collect();
/// assert_eq!(ids, ["a.talk", "a.all"]);
/// ```
pub fn bundles_of<'p>(packages: &'p [PackageInfo], id: &str) -> Vec<&'p PackageInfo> {
    let bundles: Vec<&PackageInfo> = packages.iter().filter(|p| p.kind == PackageKind::Bundle).collect();
    let mut member_of = vec![false; bundles.len()];
    // A bundle joins once it lists `id` or a bundle already found; repeat until nothing joins.
    // Builds have a few dozen packages, and a lock has no cycles (a cycle would only stop early).
    let mut grew = true;
    while grew {
        grew = false;
        for (i, bundle) in bundles.iter().enumerate() {
            if member_of[i] || bundle.id == id {
                continue;
            }
            let includes = bundle
                .dependencies
                .iter()
                .any(|dep| *dep == id || bundles.iter().zip(&member_of).any(|(b, &m)| m && b.id == *dep));
            if includes {
                member_of[i] = true;
                grew = true;
            }
        }
    }
    bundles.into_iter().zip(member_of).filter_map(|(b, m)| m.then_some(b)).collect()
}

/// Catalog cues, the mix, voice sessions and the microphone. No device, codec or mixer type.
pub mod audio {
    pub use project_watt_cubed::audio::{
        AudioApi, AudioBench, AudioView, BlockSound, CapturedFrame, GameEvent, ModuleFrame, ModuleLink, PeerAudio, Play,
    };
    #[doc(hidden)]
    pub use project_watt_cubed::audio::{ModuleFrame as ModFrame, ModuleLink as ModLink};
}

/// The game's subsystems, as far as modules may use them.
pub use project_watt_cubed::{
    block, derived, engine, gravity, headless, input, inventory, material, math, net, player, render_config, screen,
    session, settings, sim, sky, ui, world,
};

/// The module API version this crate provides (`pwc-api` requirements are checked against it).
pub const API_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The names almost every module uses.
pub mod prelude {
    pub use crate::block::{BlockId, BlockRegistry};
    pub use crate::player::Player;
    pub use crate::settings::{Category, OptionId, OptionSpec, Options};
    pub use crate::ui::{Anchor, HudElement, Line, Panel, Role, Row};
    pub use crate::world::World;
    pub use crate::{FrameContext, GameContext, HudFacts, Module, ModuleContext, ModuleRegistrar};
    #[doc(hidden)]
    pub use crate::{Mod, ModContext, ModRegistrar};
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn package(id: &'static str, kind: PackageKind, dependencies: &'static [&'static str]) -> PackageInfo {
        PackageInfo { id, name: id, version: "1.0.0", description: "", kind, dependencies, register: None }
    }

    /// Registration order: a library, modules, nested bundles, a module outside every bundle.
    const PACKAGES: &[PackageInfo] = &[
        package("t.kit", PackageKind::Library, &[]),
        package("t.chat", PackageKind::Module, &["t.kit"]),
        package("t.commands", PackageKind::Module, &["t.chat"]),
        package("t.chat-commands", PackageKind::Bundle, &["t.chat", "t.commands"]),
        package("t.hotbar", PackageKind::Module, &[]),
        package("t.essentials", PackageKind::Bundle, &["t.chat-commands", "t.hotbar"]),
        package("t.extra", PackageKind::Bundle, &["t.hotbar"]),
        package("t.toolkit", PackageKind::Module, &["t.commands"]),
    ];

    fn ids(id: &str) -> Vec<&'static str> {
        bundles_of(PACKAGES, id).iter().map(|b| b.id).collect()
    }

    #[test]
    fn bundles_include_their_members_and_the_members_of_nested_bundles() {
        assert_eq!(ids("t.chat"), ["t.chat-commands", "t.essentials"]);
        assert_eq!(ids("t.hotbar"), ["t.essentials", "t.extra"], "in list order");
        assert_eq!(ids("t.chat-commands"), ["t.essentials"]);
    }

    #[test]
    fn dependencies_of_members_and_unbundled_packages_belong_to_no_bundle() {
        assert!(ids("t.kit").is_empty(), "a member's dependency is not a member");
        assert!(ids("t.toolkit").is_empty());
        assert!(ids("t.essentials").is_empty(), "a top-level bundle");
        assert!(ids("t.gone").is_empty());
        assert!(bundles_of(&[], "t.chat").is_empty());
    }

    #[test]
    fn a_cycle_ends() {
        let cycle = [
            package("c.a", PackageKind::Bundle, &["c.b", "c.x"]),
            package("c.b", PackageKind::Bundle, &["c.a"]),
        ];
        let found: Vec<&str> = bundles_of(&cycle, "c.x").iter().map(|b| b.id).collect();
        assert_eq!(found, ["c.a", "c.b"]);
        let found: Vec<&str> = bundles_of(&cycle, "c.a").iter().map(|b| b.id).collect();
        assert_eq!(found, ["c.b"], "a bundle is never its own member");
    }
}
