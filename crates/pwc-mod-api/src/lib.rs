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
//! **Rules of the game** (see the package manager's POLICY.md): matter is not a mod — blocks are
//! configurations of elements under one law; names and looks are presentation and never feed
//! back into the law, world generation, saves or the network; anything that affects the world is
//! deterministic; in multiplayer the server is the authority; hooks run at frame/event
//! granularity, never per voxel.

pub use project_watt_cubed::modding::{
    Action, ActionSet, BuildInfo, Command, CommandContext, GameBuild, Mod, ModContext, ModDescriptor, ModRegistrar,
    Mods, PackageInfo, PackageKind, ToolUse, VisualMask,
};

/// Every bundle in `packages` that includes the package `id`, in the order `packages` lists them
/// (for a build: registration order). A bundle includes its own dependencies, and everything a
/// bundle among them includes. A mod's dependencies are not members of the bundles that list the
/// mod, and a bundle is not its own member.
///
/// ```
/// use pwc_mod_api::{bundles_of, PackageInfo, PackageKind};
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
    pub use crate::{Command, CommandContext, Mod, ModContext, ModRegistrar};
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn package(id: &'static str, kind: PackageKind, dependencies: &'static [&'static str]) -> PackageInfo {
        PackageInfo { id, name: id, version: "1.0.0", description: "", kind, dependencies, register: None }
    }

    /// Registration order: a library, mods, nested bundles, a mod outside every bundle.
    const PACKAGES: &[PackageInfo] = &[
        package("t.kit", PackageKind::Library, &[]),
        package("t.chat", PackageKind::Mod, &["t.kit"]),
        package("t.commands", PackageKind::Mod, &["t.chat"]),
        package("t.chat-commands", PackageKind::Bundle, &["t.chat", "t.commands"]),
        package("t.hotbar", PackageKind::Mod, &[]),
        package("t.essentials", PackageKind::Bundle, &["t.chat-commands", "t.hotbar"]),
        package("t.extra", PackageKind::Bundle, &["t.hotbar"]),
        package("t.toolkit", PackageKind::Mod, &["t.commands"]),
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
