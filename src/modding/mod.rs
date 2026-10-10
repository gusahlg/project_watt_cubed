//! The mod host: the game's "minimal core, layers on top" made real. Core
//! gameplay owns the world, the law and physics; everything player-facing that
//! isn't essential — the inventory panel, block looks and names, HUD
//! widgets — is a [`Mod`] that can be toggled at runtime from the mod menu.
//!
//! Mods are **compiled in**. A mod package (`.pwcmod`, see the PWC package
//! manager) is a Rust crate whose `register` function receives a
//! [`ModRegistrar`]; a [`GameBuild`] lists the packages of one exact build, and
//! [`Mods::from_build`] calls each `register` in dependency order. The core
//! itself installs no mod: `GameBuild::vanilla()` is the bare game.
//!
//! **Performance:** mod hooks fire only at frame and event granularity —
//! `update`/`draw` once per frame, `on_block_break` once per broken block. Nothing
//! here is ever called from the voxel hot path (meshing, collision, streaming), and
//! disabled mods are skipped entirely. A mod therefore costs nothing where it would
//! matter and only what it draws where it wouldn't.
mod build;
mod frame;
mod message;
#[cfg(test)]
pub(crate) mod testing;

pub use build::{BuildInfo, GameBuild, ModDescriptor, ModRegistrar, PackageInfo, PackageKind};
pub use frame::{Channel, FrameContext, GameContext, TextFrame};
pub use message::{Message, Notice, NoticeLevel, Notices};

use std::fs;
use std::io;
use std::path::Path;

use crate::block::appearance::{BlockAppearance, FLAT};
use crate::block::naming::MaterialNamer;
use crate::block::BlockId;
use crate::menu::start::{StartFacts, StartScreen};
use crate::menu::theme::MenuTheme;
use crate::player::Player;
use crate::render_config::{RenderConfig, VisualGroup};
use crate::settings::Settings;
use crate::ui::HudElement;
use crate::world::generation::WorldgenKind;
use crate::world::World;

/// Group id of the first-party essentials (menus, inventory, looks, names, worldgen).
pub const ESSENTIALS: &str = "essentials";

/// The well-known essentials group. Mods returning [`ESSENTIALS`] from [`Mod::group`] are shown
/// under it without declaring it themselves.
pub const ESSENTIALS_GROUP: Group = Group {
    id: ESSENTIALS,
    name: "Essentials",
    description: "Menus, inventory, looks, names and worldgen.",
};

/// Named group of related mods. The id is the stable key; the display name
/// can change here without touching every member.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Group {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
}

/// One tunable shown under a mod in the mods menu.
#[derive(Clone, Debug)]
pub struct Knob {
    pub label: &'static str,
    pub value: String,
    /// Allowed range or choice list, shown as the row detail.
    pub hint: String,
}

/// Which fancy visual groups are currently enabled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisualMask {
    pub atmosphere: bool,
    pub post: bool,
    pub lighting: bool,
}

impl Default for VisualMask {
    fn default() -> Self {
        Self::of(VisualGroup::ALL)
    }
}

impl VisualMask {
    pub const NONE: Self = Self {
        atmosphere: false,
        post: false,
        lighting: false,
    };

    /// The mask with exactly `groups` on: a group is on when any enabled mod owns it. The mods
    /// screen and the renderer both build their mask here.
    pub fn of(groups: impl IntoIterator<Item = VisualGroup>) -> Self {
        let mut mask = Self::NONE;
        for group in groups {
            mask.set(group, true);
        }
        mask
    }

    pub fn get(self, group: VisualGroup) -> bool {
        match group {
            VisualGroup::Atmosphere => self.atmosphere,
            VisualGroup::Post => self.post,
            VisualGroup::Lighting => self.lighting,
        }
    }

    pub fn set(&mut self, group: VisualGroup, on: bool) {
        *match group {
            VisualGroup::Atmosphere => &mut self.atmosphere,
            VisualGroup::Post => &mut self.post,
            VisualGroup::Lighting => &mut self.lighting,
        } = on;
    }

    pub fn apply(self, mut cfg: RenderConfig) -> RenderConfig {
        for group in VisualGroup::ALL {
            if !self.get(group) {
                cfg.strip_group(group);
            }
        }
        cfg
    }

    /// Settings lanes with this mask's disabled groups stripped.
    pub fn effective_render(self, settings: &Settings) -> RenderConfig {
        self.apply(settings.render_config())
    }

    /// Name of the visual mod forcing `key` off, if any.
    pub fn forced_off(self, key: &str) -> Option<&'static str> {
        let group = crate::render_config::lane_group(key)?;
        (!self.get(group)).then(|| group.mod_name())
    }
}

/// Marker appended when a visual group has stripped the lane. The settings
/// menu, `/gfx`, and any HUD that prints lanes share this one string.
pub fn forced_off_marker(mod_name: &str) -> String {
    format!("(off: {mod_name} mod)")
}

/// Append [`forced_off_marker`] when a visual group has stripped the lane.
pub fn annotate_setting(value: String, key: &str, mask: VisualMask) -> String {
    match mask.forced_off(key) {
        Some(name) => format!("{value} {}", forced_off_marker(name)),
        None => value,
    }
}

/// One rebindable control a mod declares. The core keeps the chord table and
/// reports which fired; it does not know what the action means. There is no
/// controls screen yet: [`Action::label`] is what that screen will show.
#[derive(Clone, Copy)]
pub struct Action {
    /// Stable id (`"inventory.toggle"`). [`ModContext::action`] matches it.
    pub id: &'static str,
    /// Text a controls screen would show next to the binding.
    pub label: &'static str,
    /// Chords that fire the action, unless a core binding already uses that chord.
    pub default: &'static [crate::input::intent::Chord],
    /// When true, the action autofires with the same timing as breaking and placing.
    /// Ignored when [`held`](Self::held) is set.
    pub repeat: bool,
    /// When true, the action is on for every frame the chord is down, not only the press.
    pub held: bool,
    /// When true, the action is sampled every frame, even with the mod-logic lane off, and
    /// reaches only [`Mod::on_frame`] ([`FrameContext::action`]), never [`Mod::update`]. For keys
    /// that must always work, such as opening the chat.
    pub immediate: bool,
}

/// Which declared actions fired this frame. At most [`ActionSet::CAP`] actions;
/// the bit is the action's index in the router's table. [`Copy`], so a skipped
/// mod tick can queue the set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ActionSet(u128);

impl ActionSet {
    pub const NONE: Self = Self(0);
    pub const CAP: usize = 128;

    pub fn insert(&mut self, index: usize) {
        if index < Self::CAP {
            self.0 |= 1u128 << index;
        }
    }

    pub fn contains(self, index: usize) -> bool {
        index < Self::CAP && self.0 & (1u128 << index) != 0
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// What a primary action did with a tool. `cell` is the targeted configuration
/// before the reaction. The core does not draw this; a mod may.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolUse {
    NoReaction,
    /// The targeted cell became air.
    CellDissolved { cell: BlockId },
    /// The tool was used up.
    ToolDissolved { cell: BlockId },
    /// The tool gained an element from the cell.
    Drew { cell: BlockId },
    /// The tool gave an element to the cell.
    Gave { cell: BlockId },
    /// Elements moved both ways.
    Exchanged { cell: BlockId },
}

/// The coarse, per-frame state a mod may read and mutate. Deliberately holds only
/// whole-game handles (never a voxel), so a mod can't reach into the hot path.
/// `#[non_exhaustive]`: build one with [`ModContext::new`] and set the fields you need.
#[non_exhaustive]
pub struct ModContext<'a> {
    pub player: &'a mut Player,
    pub world: &'a mut World,
    pub screen_w: i32,
    pub screen_h: i32,
    /// Keybind intents; `place` gated on mouse capture separately.
    pub place: bool,
    /// Placement cell resolved from the exact frame that raised `place`.
    /// Fixed-cadence replay may run after the player has moved or looked away.
    pub place_target: Option<(i32, i32, i32)>,
    pub nav_up: bool,
    pub nav_down: bool,
    pub nav_left: bool,
    pub nav_right: bool,
    pub nav_tab: bool,
    pub nav_confirm: bool,
    /// Signed scroll steps this frame (a notch is about 1). Zero when mod logic is off.
    pub wheel: i8,
    /// Whether a mod panel may open. Captured with the edge, so a toggle taken
    /// while the HUD is hidden does not open a panel when it is replayed.
    pub mod_ui: bool,
    /// True when a server owns evaluation (reactions and tool use run there).
    pub networked: bool,
    /// Block placements queued by mods this frame as `(x, y, z, id)`. The game
    /// drains these after `mods.update` and applies each only if the cell is air
    /// and doesn't overlap the player — mods that spend resources on a placement
    /// should pre-check the same so their accounting stays exact.
    pub placements: Vec<(i32, i32, i32, crate::block::registry::BlockId)>,
    pub(crate) fired: ActionSet,
    pub(crate) ids: &'a [&'static str],
    /// Ids from [`ModContext::set_action`]. Empty on the game path, so it allocates nothing.
    pub(crate) extra: Vec<&'static str>,
}

impl<'a> ModContext<'a> {
    /// A context for tests: 800×600, panels allowed, no edges, empty placements.
    pub fn new(player: &'a mut Player, world: &'a mut World) -> Self {
        Self {
            player,
            world,
            screen_w: 800,
            screen_h: 600,
            place: false,
            place_target: None,
            nav_up: false,
            nav_down: false,
            nav_left: false,
            nav_right: false,
            nav_tab: false,
            nav_confirm: false,
            wheel: 0,
            mod_ui: true,
            networked: false,
            placements: Vec::new(),
            fired: ActionSet::NONE,
            ids: &[],
            extra: Vec::new(),
        }
    }

    /// Whether the action `id` fired this frame. True if any enabled mod's
    /// action with that id fired.
    pub fn action(&self, id: &str) -> bool {
        self.ids.iter().enumerate().any(|(i, name)| *name == id && self.fired.contains(i))
            || self.extra.iter().enumerate().any(|(i, name)| *name == id && self.fired.contains(self.ids.len() + i))
    }

    /// Mark `id` fired. For tests; the game path fills [`Self::fired`] instead.
    pub fn set_action(&mut self, id: &'static str) {
        if let Some(i) = self.ids.iter().position(|name| *name == id) {
            self.fired.insert(i);
            return;
        }
        if let Some(i) = self.extra.iter().position(|name| *name == id) {
            self.fired.insert(self.ids.len() + i);
            return;
        }
        let i = self.ids.len() + self.extra.len();
        if i >= ActionSet::CAP {
            return;
        }
        self.extra.push(id);
        self.fired.insert(i);
    }

    /// A machine changed matter at `pos`: wake that cell's contacts in the reaction scheduler.
    /// Chunk load/gen/mesh/save never do this. No-op on a client connected to a server (the
    /// authority runs the scheduler).
    pub fn wake_cell(&mut self, pos: (i32, i32, i32)) {
        self.world.note_cell_changed(pos.0, pos.1, pos.2);
    }
}

/// A unit of layered-on functionality. Every method has a default, so a mod
/// implements only the hooks it cares about. This is the public surface mod authors
/// write against — kept small on purpose.
///
/// Arbitration when more than one enabled mod implements a hook:
/// - **Fan-out**, install order: `update`, `on_frame`, `on_message`, `on_block_break`,
///   `on_break_rejected`, `on_place_rejected`, `on_tool_changed`, `on_tool_used`. `hud` uses the
///   same order as z-order (later draws on top). `actions` are collected, not arbitrated: each
///   enabled mod's list is its own.
/// - **First enabled wins**: `menu_theme`, `start_screen` and `close_overlay` (first `true`),
///   `worldgen`, `worldgen_config`, `appearance`, `namer`, `tool` (first `Some`). The keyboard
///   capture ([`FrameContext::capture_text`]) belongs to the first mod that asks until it gives it
///   back or Escape ends it.
/// - **Compose**: `visual_group` bits OR into the render mask.
///
/// `knobs` / `step_knob` and save hooks are per-mod. `worldgen_config` is an
/// opaque string; the winning worldgen kind parses it.
pub trait Mod {
    /// Short name shown in the mod menu. Not a save key — see [`id`].
    fn name(&self) -> &str;

    /// Stable lowercase code id. Persist, env pins, and lookups use this;
    /// [`name`] is the display label and may change.
    fn id(&self) -> &'static str;

    /// One-line description for the mod menu.
    fn description(&self) -> &str {
        ""
    }

    /// Group id ([`ESSENTIALS`] or one declared with [`ModRegistrar::declare_group`]), or `""`
    /// if ungrouped.
    fn group(&self) -> &'static str {
        ""
    }

    /// Called when the mod is switched on (including at load if enabled).
    fn on_enable(&mut self) {}
    /// Called when the mod is switched off.
    fn on_disable(&mut self) {}

    /// Clear per-world state (crafted blocks, open panels) when entering a
    /// different world. Enable/disable choices are NOT touched — those persist
    /// across worlds. The inventory lives on the player, not here.
    fn reset(&mut self) {}

    /// Cadence-controlled logic while enabled (the game's `mod_hz`). Runs
    /// after movement, before rendering; edge inputs accumulated between
    /// ticks are replayed in order without loss.
    fn update(&mut self, ctx: &mut ModContext) {
        let _ = ctx;
    }

    /// Every in-world frame, after input and before movement, whatever the mod cadence and even
    /// with the mod-logic lane off. `ctx` has this frame's immediate actions, the keyboard capture
    /// (and the typing, while this mod holds it), and the game state; the core follows up on what
    /// the hook changed (see [`GameContext`]). Keep it cheap: it runs at frame rate.
    fn on_frame(&mut self, ctx: &mut FrameContext) {
        let _ = ctx;
    }

    /// A chat line, a join or leave, or a core notice (see [`Message`]). Return `true` when this
    /// mod showed it to the player; a notice no mod showed goes to stderr.
    fn on_message(&mut self, msg: &Message) -> bool {
        let _ = msg;
        false
    }

    /// A block was broken into this configuration. The core has already deposited
    /// it into the player inventory; this is a notification. `overflow` is true
    /// when the inventory dropped it (capacity).
    fn on_block_break(&mut self, id: BlockId, world: &World, overflow: bool) {
        let _ = (id, world, overflow);
    }

    /// The server rejected a break this client predicted (someone else won the
    /// cell). The core has already revoked the loot from the inventory; this is a
    /// notification.
    fn on_break_rejected(&mut self, id: BlockId) {
        let _ = id;
    }

    /// The server rejected a placement this client predicted: refund whatever
    /// was spent on placing a block of `id`.
    fn on_place_rejected(&mut self, id: crate::block::BlockId, world: &World) {
        let _ = (id, world);
    }

    /// Controls this mod wants sampled while it is enabled. The core owns the chord
    /// table; a core binding wins any clash. Default is none.
    fn actions(&self) -> &[Action] {
        &[]
    }

    /// The configuration a primary action applies, if any (first enabled mod that answers
    /// wins). `None` means no tool: the primary action breaks the block into the inventory.
    fn tool(&self, player: &Player) -> Option<BlockId> {
        let _ = player;
        None
    }

    /// A held unit changed configuration through a tool reaction: one unit of `old` became one
    /// unit of `new` (`AIR` when the tool was used up). The core already updated the inventory.
    fn on_tool_changed(&mut self, old: BlockId, new: BlockId) {
        let _ = (old, new);
    }

    /// A primary action finished. `outcome` says what the law did; the core draws nothing for it.
    fn on_tool_used(&mut self, outcome: ToolUse) {
        let _ = outcome;
    }

    /// A game fact the core cannot derive (a block edit, a step, a menu click). `audio` plays cues.
    fn on_game_event(&mut self, ev: &crate::audio::GameEvent, audio: &mut crate::audio::AudioApi) {
        let _ = (ev, audio);
    }

    /// Every frame, menus included. `view` is the listener and the roster; `link` is the mod channel.
    fn on_audio(
        &mut self,
        view: &crate::audio::AudioView,
        audio: &mut crate::audio::AudioApi,
        link: &mut crate::audio::ModLink,
    ) {
        let _ = (view, audio, link);
    }

    /// Optional material namer. First enabled mod that returns `Some` names every
    /// configuration; without one the core describes materials by their readings.
    fn namer(&self) -> Option<&dyn MaterialNamer> {
        None
    }

    /// This mod's HUD contribution while enabled, as data — [`HudElement`]s
    /// pushed into a caller-owned buffer the core renders over the world and
    /// under the console. A mod describes *what* to show and never draws, so
    /// panel chrome and layout live in one place ([`crate::ui::render_hud`]).
    /// `world` gives read access to the registry so names resolve at build time
    /// rather than being cached. `player` is the one path to the core inventory.
    fn hud(&self, world: &World, player: &Player, screen: (i32, i32), out: &mut Vec<HudElement>) {
        let _ = (world, player, screen, out);
    }

    /// Close a modal in-world overlay before the core interprets Escape as
    /// "leave the world". Returns whether this mod consumed the key.
    fn close_overlay(&mut self) -> bool {
        false
    }

    /// Optional theme override; fallback prevents breaking nav.
    fn menu_theme(&self) -> Option<&dyn MenuTheme> {
        None
    }

    /// Optional start screen. First enabled mod that returns `Some` wins;
    /// the core fallback (New world / Load / Settings / Mods / Quit) is used
    /// when every enabled mod returns `None`. Plain-data signatures only.
    fn start_screen(&self, facts: &StartFacts) -> Option<Box<dyn StartScreen>> {
        let _ = facts;
        None
    }

    /// Serialise persistent state, or `None` if the mod has nothing to persist.
    /// The `u16` is the payload version; [`Mods::save_states`] encodes it as a
    /// `v<N>;` prefix so the save codec stays a plain string.
    fn save_state(&self, world: &World) -> Option<(u16, String)> {
        let _ = world;
        None
    }

    /// Restore state produced by [`save_state`](Self::save_state). `version` is
    /// 0 when the on-disk string had no prefix (old saves). `world` is mutable
    /// because restoring may need to re-register blocks. Returns how many
    /// holdings were dropped as unknown specs.
    fn load_state(&mut self, version: u16, data: &str, world: &mut World) -> u32 {
        let _ = (version, data, world);
        0
    }

    /// Which fancy render group this mod owns, if any. The host reads it when the mod is
    /// installed or switched on or off.
    fn visual_group(&self) -> Option<VisualGroup> {
        None
    }

    /// If this mod replaces worldgen, the kind to use when it is enabled.
    fn worldgen(&self) -> Option<WorldgenKind> {
        None
    }

    fn knobs(&self) -> Vec<Knob> {
        Vec::new()
    }

    fn step_knob(&mut self, index: usize, delta: i32) {
        let _ = (index, delta);
    }

    /// Knob/config payload written as `id.state=` in `saves/mods.cfg`.
    /// Per-world [`save_state`] is a different path and is not written here.
    fn save_choice_state(&self) -> Option<String> {
        None
    }

    fn load_choice_state(&mut self, data: &str) {
        let _ = data;
    }

    /// Opaque payload for the winning [`worldgen`] kind. `None` if this mod
    /// does not replace worldgen. InfiniteDiffusion parses it as its knobs.
    fn worldgen_config(&self) -> Option<String> {
        None
    }

    /// Optional block appearance. First enabled mod that returns `Some` wins;
    /// [`FlatAppearance`](crate::block::appearance::FlatAppearance) is used
    /// when every enabled mod returns `None`.
    fn appearance(&self) -> Option<&dyn BlockAppearance> {
        None
    }
}

/// One installed mod and whether it is currently active.
struct Entry {
    module: Box<dyn Mod>,
    enabled: bool,
    /// Id of the package that registered it (`None` for mods installed directly).
    package: Option<&'static str>,
}

/// The set of installed mods and their on/off state. Enable/disable choices persist
/// in `mods.cfg`; per-world state is saved through each mod's `save_state`/`load_state`.
pub struct Mods {
    entries: Vec<Entry>,
    /// Groups declared by packages, after the well-known [`ESSENTIALS_GROUP`].
    declared_groups: Vec<Group>,
    /// Bumped when a mod is installed or enabled or disabled, so the input
    /// table can rebuild once instead of every frame.
    action_gen: u64,
    /// The enabled mods' visual groups, rebuilt with `action_gen`.
    visuals: VisualMask,
    /// Bumped whenever what the mods screen lists may have changed.
    revision: u64,
    /// Package ids the current server refused. Not written to `mods.cfg`.
    server_packages: Vec<String>,
    /// Install index of the mod holding the keyboard (see [`FrameContext::capture_text`]).
    capture: Option<usize>,
    /// Module ids that were on when the server refused their package. Restored
    /// on leave. The session disable itself is not saved.
    server_held: Vec<String>,
}

impl Mods {
    /// Instantiate every package of `build` that has an entry point, in its (dependency) order.
    /// Each package's `register` sees the resources its dependencies provided and the whole
    /// package list.
    pub fn from_build(build: &GameBuild) -> Self {
        let mut mods = Self::empty();
        let mut resources = build::Resources::default();
        let info = build.info();
        for package in info.packages() {
            let Some(register) = package.register else { continue };
            let mut registrar = ModRegistrar::new(package, info, &mut mods, &mut resources);
            register(&mut registrar);
        }
        mods
    }

    /// No mods installed. Appearance is [`FLAT`].
    pub fn empty() -> Self {
        Self {
            entries: Vec::new(),
            declared_groups: Vec::new(),
            action_gen: 0,
            visuals: VisualMask::NONE,
            revision: 0,
            server_packages: Vec::new(),
            server_held: Vec::new(),
            capture: None,
        }
    }

    /// Install a mod, running its enable hook if it starts on.
    pub fn install(&mut self, module: Box<dyn Mod>, enabled: bool) {
        self.install_from(None, module, enabled);
    }

    fn install_from(&mut self, package: Option<&'static str>, module: Box<dyn Mod>, enabled: bool) {
        let mut entry = Entry { module, enabled, package };
        if enabled {
            entry.module.on_enable();
        }
        self.entries.push(entry);
        self.enabled_changed();
    }

    /// The enabled set changed: a new action generation and visual mask.
    fn enabled_changed(&mut self) {
        self.action_gen = self.action_gen.wrapping_add(1);
        let enabled = self.entries.iter().filter(|e| e.enabled);
        self.visuals = VisualMask::of(enabled.filter_map(|e| e.module.visual_group()));
        self.revise();
    }

    fn revise(&mut self) {
        self.revision = self.revision.wrapping_add(1);
    }

    /// Generation of the enabled action lists. Changes when a mod is installed
    /// or switched on or off.
    pub fn action_generation(&self) -> u64 {
        self.action_gen
    }

    /// Changes whenever what the mods screen lists may have changed: a mod installed or switched,
    /// a knob stepped or loaded, a group declared, or a server hold set or released.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Actions of every enabled mod, in install order.
    pub fn enabled_actions(&self) -> impl Iterator<Item = &Action> + '_ {
        self.entries.iter().filter(|e| e.enabled).flat_map(|e| e.module.actions())
    }

    /// Groups shown as sections on the mods screen, in this order: the well-known essentials,
    /// then every group a package declared.
    pub fn groups(&self) -> impl Iterator<Item = &Group> {
        std::iter::once(&ESSENTIALS_GROUP).chain(self.declared_groups.iter())
    }

    fn declare_group(&mut self, group: Group) {
        if group.id != ESSENTIALS && !self.declared_groups.iter().any(|g| g.id == group.id) {
            self.declared_groups.push(group);
            self.revise();
        }
    }

    /// The package that registered the mod at `index`, if any.
    pub fn package(&self, index: usize) -> Option<&'static str> {
        self.entries[index].package
    }

    /// Reset every mod's per-world state (entering a new/loaded/networked
    /// world) while keeping the player's enable/disable choices.
    pub fn reset_state(&mut self) {
        self.capture = None;
        for entry in &mut self.entries {
            entry.module.reset();
        }
    }

    fn each_enabled(&mut self, mut f: impl FnMut(&mut dyn Mod)) {
        for entry in &mut self.entries {
            if entry.enabled {
                f(&mut *entry.module);
            }
        }
    }

    /// Run every enabled mod's per-frame logic.
    pub fn update(&mut self, ctx: &mut ModContext) {
        self.each_enabled(|m| m.update(ctx));
    }

    /// Run every enabled mod's frame hook, in install order. Only the keyboard holder sees the
    /// typing; Escape ends its capture after its hook. A holder that was switched off loses the
    /// keyboard first.
    pub fn on_frame(&mut self, ctx: &mut FrameContext) {
        if let Some(holder) = self.capture
            && !self.entries.get(holder).is_some_and(|e| e.enabled)
        {
            self.capture = None;
        }
        let held = self.capture;
        ctx.set_holder(held);
        for (index, entry) in self.entries.iter_mut().enumerate() {
            if entry.enabled {
                ctx.enter(index);
                entry.module.on_frame(ctx);
            }
        }
        if held.is_some() && ctx.escaped() && ctx.holder() == held {
            ctx.set_holder(None);
        }
        self.capture = ctx.holder();
    }

    /// Whether a mod holds the keyboard. While it does the router reads typing, the world takes
    /// no input, and Escape goes to that mod (and ends the capture) before anything else sees it.
    pub fn text_captured(&self) -> bool {
        self.capture.is_some()
    }

    /// Hand one message to every enabled mod. True when any of them showed it.
    pub fn on_message(&mut self, msg: &Message) -> bool {
        let mut shown = false;
        for entry in &mut self.entries {
            if entry.enabled {
                shown |= entry.module.on_message(msg);
            }
        }
        shown
    }

    /// Fan a block-break event out to every enabled mod.
    pub fn on_block_break(&mut self, id: BlockId, world: &World, overflow: bool) {
        self.each_enabled(|m| m.on_block_break(id, world, overflow));
    }

    /// Fan a rejected-break rollback out to every enabled mod.
    pub fn on_break_rejected(&mut self, id: BlockId) {
        self.each_enabled(|m| m.on_break_rejected(id));
    }

    /// Fan a rejected-placement refund out to every enabled mod.
    pub fn on_place_rejected(&mut self, id: crate::block::BlockId, world: &World) {
        self.each_enabled(|m| m.on_place_rejected(id, world));
    }

    /// Fan a held unit's change of configuration out to every enabled mod.
    pub fn on_tool_changed(&mut self, old: BlockId, new: BlockId) {
        self.each_enabled(|m| m.on_tool_changed(old, new));
    }

    /// Fan a finished primary action out to every enabled mod.
    pub fn on_tool_used(&mut self, outcome: ToolUse) {
        self.each_enabled(|m| m.on_tool_used(outcome));
    }

    /// Fan one game fact out to every enabled mod.
    pub fn on_game_event(&mut self, ev: &crate::audio::GameEvent, audio: &mut crate::audio::AudioApi) {
        self.each_enabled(|m| m.on_game_event(ev, audio));
    }

    /// The per-frame audio hook. Runs even when the frame is otherwise idle.
    pub fn on_audio(
        &mut self,
        view: &crate::audio::AudioView,
        audio: &mut crate::audio::AudioApi,
        link: &mut crate::audio::ModLink,
    ) {
        self.each_enabled(|m| m.on_audio(view, audio, link));
    }

    /// The configuration a primary action applies: the first enabled mod that answers.
    pub fn tool(&self, player: &Player) -> Option<BlockId> {
        self.entries.iter().filter(|e| e.enabled).find_map(|e| e.module.tool(player))
    }

    /// The first enabled namer, if any.
    pub fn namer(&self) -> Option<&dyn MaterialNamer> {
        self.entries.iter().filter(|e| e.enabled).find_map(|e| e.module.namer())
    }

    /// Push every enabled mod's HUD contribution into `out`, in install order
    /// (so a later mod draws over an earlier one). The caller owns `out` and
    /// clears it per frame so capacity is retained.
    pub fn hud(&self, world: &World, player: &Player, screen: (i32, i32), out: &mut Vec<HudElement>) {
        for entry in &self.entries {
            if entry.enabled {
                entry.module.hud(world, player, screen, out);
            }
        }
    }

    /// Give enabled mods first refusal on Escape. The first open overlay closes
    /// and consumes it; otherwise the game can return to its main menu.
    pub fn close_overlay(&mut self) -> bool {
        self.entries
            .iter_mut()
            .filter(|entry| entry.enabled)
            .any(|entry| entry.module.close_overlay())
    }

    /// Fallback theme ensures disabling menu mod never breaks nav.
    pub fn menu_theme(&self) -> Option<&dyn MenuTheme> {
        self.entries
            .iter()
            .filter(|e| e.enabled)
            .find_map(|e| e.module.menu_theme())
    }

    /// First enabled mod that returns a start screen wins. `None` means the
    /// core fallback should be used.
    pub fn start_screen(&self, facts: &StartFacts) -> Option<Box<dyn StartScreen>> {
        self.entries
            .iter()
            .filter(|e| e.enabled)
            .find_map(|e| e.module.start_screen(facts))
    }

    /// Number of installed mods (for the mod menu).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether there are no installed mods.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The name of the mod at `index`.
    pub fn name(&self, index: usize) -> &str {
        self.entries[index].module.name()
    }

    /// The code id of the mod at `index`.
    pub fn id(&self, index: usize) -> &str {
        self.entries[index].module.id()
    }

    /// The description of the mod at `index`.
    pub fn description(&self, index: usize) -> &str {
        self.entries[index].module.description()
    }

    /// The group id of the mod at `index` (`""` if ungrouped).
    pub fn group(&self, index: usize) -> &str {
        self.entries[index].module.group()
    }

    /// The declared group of the mod at `index`, if its group id names one.
    pub fn group_of(&self, index: usize) -> Option<Group> {
        let id = self.entries[index].module.group();
        self.groups().find(|g| g.id == id).copied()
    }

    /// Whether the mod at `index` is enabled.
    pub fn is_enabled(&self, index: usize) -> bool {
        self.entries[index].enabled
    }

    /// Visual group owned by the mod at `index`, if it is a visual mod.
    pub fn visual_group(&self, index: usize) -> Option<VisualGroup> {
        self.entries[index].module.visual_group()
    }

    /// Whether the mod at `index` replaces worldgen when enabled.
    pub fn is_worldgen(&self, index: usize) -> bool {
        self.entries[index].module.worldgen().is_some()
    }

    /// Flip the mod at `index` on or off, running the matching lifecycle hook.
    /// False when the index is out of range or a server hold refuses the turn-on.
    /// A refused turn-on does not change state and must not be written to `mods.cfg`.
    pub fn toggle(&mut self, index: usize) -> bool {
        if index >= self.entries.len() {
            return false;
        }
        let turning_on = !self.entries[index].enabled;
        if turning_on && self.server_off(index) {
            return false;
        }
        let entry = &mut self.entries[index];
        entry.enabled = !entry.enabled;
        if entry.enabled {
            entry.module.on_enable();
        } else {
            entry.module.on_disable();
        }
        self.enabled_changed();
        true
    }

    pub fn knobs(&self, index: usize) -> Vec<Knob> {
        self.entries[index].module.knobs()
    }

    pub fn step_knob(&mut self, index: usize, knob: usize, delta: i32) {
        self.entries[index].module.step_knob(knob, delta);
        self.revise();
    }

    /// Worldgen used for the next world: InfiniteDiffusion if that mod is on, else the flat
    /// core fallback.
    pub fn worldgen_kind(&self) -> WorldgenKind {
        self.first_worldgen()
            .and_then(|m| m.worldgen())
            .unwrap_or(WorldgenKind::Flat)
    }

    /// Opaque payload of the winning worldgen mod. The kind parses it
    /// (`TerrainCfg::from_text` for InfiniteDiffusion).
    pub fn worldgen_config(&self) -> Option<String> {
        self.first_worldgen().and_then(|m| m.worldgen_config())
    }

    fn first_worldgen(&self) -> Option<&dyn Mod> {
        self.entries
            .iter()
            .filter(|e| e.enabled)
            .find(|e| e.module.worldgen().is_some())
            .map(|e| &*e.module)
    }

    /// First enabled appearance mod, or the core flat fallback.
    pub fn appearance(&self) -> &dyn BlockAppearance {
        self.entries
            .iter()
            .filter(|e| e.enabled)
            .find_map(|e| e.module.appearance())
            .unwrap_or(&FLAT)
    }

    /// The visual groups the enabled mods own (see [`VisualMask::of`]).
    pub fn visual_mask(&self) -> VisualMask {
        self.visuals
    }

    /// Enable or disable a mod by [`Mod::id`] (case-insensitive). No-op if
    /// already in that state, the id is unknown, or a server hold refuses the turn-on.
    pub fn set_enabled(&mut self, id: &str, on: bool) {
        if let Some(i) = self
            .entries
            .iter()
            .position(|e| e.module.id().eq_ignore_ascii_case(id))
            && self.entries[i].enabled != on
        {
            let _ = self.toggle(i);
        }
    }

    /// Enable or disable every installed member of `group_id`. Persists as
    /// each member's `id=on|off` line — there is no group-level key. True when
    /// at least one member changed. Members a server hold refuses stay off.
    pub fn set_group_enabled(&mut self, group_id: &str, on: bool) -> bool {
        let mut changed = false;
        for i in 0..self.entries.len() {
            let member = self.entries[i].module.group() == group_id;
            let differs = self.entries[i].enabled != on;
            if member && differs && self.toggle(i) {
                changed = true;
            }
        }
        changed
    }

    /// Turn off every enabled module whose package is in `package_ids`, and
    /// remember those module ids. Already-off siblings of a refused package
    /// also show as server-off and cannot be enabled. This does not write
    /// `mods.cfg`. The list is what the server said; it is not a proof the
    /// client is unmodified.
    pub fn hold_packages(&mut self, package_ids: &[String]) {
        self.server_packages = package_ids.to_vec();
        self.server_held.clear();
        let mut turn_off = Vec::new();
        for i in 0..self.entries.len() {
            let Some(pkg) = self.entries[i].package else { continue };
            if !package_ids.iter().any(|id| id == pkg) || !self.entries[i].enabled {
                continue;
            }
            turn_off.push(self.entries[i].module.id().to_string());
        }
        for id in &turn_off {
            self.set_enabled(id, false);
        }
        self.server_held = turn_off;
        self.revise();
    }

    /// True when the mod at `index` belongs to a package the server refused.
    pub fn server_off(&self, index: usize) -> bool {
        self.entries.get(index).and_then(|e| e.package).is_some_and(|pkg| {
            self.server_packages.iter().any(|id| id == pkg)
        })
    }

    /// Re-enable only the modules [`hold_packages`](Self::hold_packages) turned off.
    pub fn release_server(&mut self) {
        let held = std::mem::take(&mut self.server_held);
        self.server_packages.clear();
        self.revise();
        for id in held {
            self.set_enabled(&id, true);
        }
    }

    /// Enabled mod packages, as `(id, version)`, in build order. A package is
    /// included when it is of kind mod and any of its modules is enabled; libraries
    /// and bundles are never reported. This is what an honest client puts on `Hello`.
    pub fn enabled_package_reports(&self, packages: &[PackageInfo]) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for desc in packages.iter().filter(|p| p.kind == PackageKind::Mod) {
            let on = self.entries.iter().any(|e| e.enabled && e.package == Some(desc.id));
            if on {
                out.push((desc.id.to_string(), desc.version.to_string()));
            }
        }
        out
    }

    /// Apply pins parsed by [`crate::benchmark::Benchmark::mod_pins_from_env`].
    pub fn apply_bench_env(&mut self, worldgen_diffusion: Option<bool>, visuals_core: Option<bool>) {
        match worldgen_diffusion {
            Some(true) => self.set_enabled(WorldgenKind::Diffusion.id(), true),
            Some(false) => self.set_enabled(WorldgenKind::Diffusion.id(), false),
            None => {}
        }
        if visuals_core == Some(true) {
            self.set_enabled("atmosphere", false);
            self.set_enabled("post", false);
            self.set_enabled("lighting", false);
        }
    }

    /// Settings lanes with disabled visual groups stripped. The one
    /// composition world construction, `/gfx` apply, and the engine flags share.
    pub fn effective_render(&self, settings: &Settings) -> RenderConfig {
        self.visual_mask().effective_render(settings)
    }

    /// Persistent state of every mod that has any, as `(id, data)` lines.
    /// Version is encoded inside `data` as a leading `v<N>;` so the save
    /// codec does not change; old unprefixed strings load as version 0.
    pub fn save_states(&self, world: &World) -> Vec<(String, String)> {
        self.entries
            .iter()
            .filter_map(|entry| {
                entry.module.save_state(world).map(|(version, data)| {
                    (entry.module.id().to_string(), format!("v{version};{data}"))
                })
            })
            .collect()
    }

    /// Restore a mod's state by id, or by display name for old saves.
    /// Returns how many holdings that mod dropped as unknown specs.
    pub fn load_state(&mut self, name: &str, data: &str, world: &mut World) -> u32 {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.module.id() == name || e.module.name() == name) {
            let (version, payload) = split_mod_version(data);
            return entry.module.load_state(version, payload, world);
        }
        0
    }

    /// `id=on|off` lines, plus `id.state=<payload>` for mods that persist knobs,
    /// under a `version=` marker.
    pub fn choices_text(&self) -> String {
        let mut text = format!("version={CHOICES_VERSION}\n");
        for entry in &self.entries {
            // A server hold is in-memory only. The saved choice stays what it
            // was, so a flush during the session does not record the hold.
            let held = self.server_held.iter().any(|id| id == entry.module.id());
            let on = entry.enabled || held;
            text.push_str(entry.module.id());
            text.push('=');
            text.push_str(if on { "on" } else { "off" });
            text.push('\n');
            if let Some(payload) = entry.module.save_choice_state() {
                text.push_str(entry.module.id());
                text.push_str(".state=");
                text.push_str(&payload);
                text.push('\n');
            }
        }
        text
    }

    /// Apply `id=on|off` and `id.state=` lines. Unknown ids and malformed lines
    /// are ignored; missing keys keep the current defaults. A file from before
    /// [`CHOICES_VERSION`] 2 recorded `diffusion=off` as the then-default of an
    /// experiment; that mod is the world generator now, so those lines are dropped.
    pub fn apply_choices_text(&mut self, text: &str) {
        let mut version = 1;
        crate::settings::each_kv_line(text, |key, value| {
            if key == "version" {
                version = value.trim().parse().unwrap_or(1);
            }
        });
        crate::settings::each_kv_line(text, |key, value| {
            if key == "version" || (version < 2 && key.starts_with("diffusion")) {
                return;
            }
            if let Some(id) = key.strip_suffix(".state") {
                self.apply_choice_state(id.trim(), value);
                return;
            }
            let Some(on) = crate::settings::parse_toggle(value) else {
                return;
            };
            self.set_enabled(key, on);
        });
    }

    fn apply_choice_state(&mut self, id: &str, data: &str) {
        let Some(entry) = self.entries.iter_mut().find(|e| e.module.id().eq_ignore_ascii_case(id)) else {
            return;
        };
        entry.module.load_choice_state(data);
        self.revise();
    }

    /// Restore enable/disable choices and knob payloads from `mods.cfg`.
    /// Missing or unreadable file leaves the current defaults in place.
    pub fn load_choices(&mut self) {
        self.load_choices_from(&crate::paths::Paths::get().mods_file());
    }

    fn load_choices_from(&mut self, path: &Path) {
        if let Ok(text) = fs::read_to_string(path) {
            self.apply_choices_text(&text);
        }
    }

    /// Write enable/disable choices and knob payloads. Bench-env pins are not
    /// written from startup; only a later toggle or knob step persists.
    pub fn save_choices(&self) -> io::Result<()> {
        self.save_choices_to(&crate::paths::Paths::get().mods_file())
    }

    fn save_choices_to(&self, path: &Path) -> io::Result<()> {
        crate::save::write_atomic_file(path, self.choices_text().as_bytes())
    }
}

/// `mods.cfg` format: 2 since the diffusion mod became the default world generator.
const CHOICES_VERSION: u32 = 2;

/// Debounces file writes so a held Left/Right does not rewrite at key-repeat rate: a write is
/// due once [`IDLE_MS`](Self::IDLE_MS) pass with no further mark.
pub struct Debounce {
    last_ms: Option<u64>,
}

/// The `mods.cfg` debounce, by its mod API name.
pub type ChoicesFlush = Debounce;

impl Debounce {
    pub const IDLE_MS: u64 = 250;

    pub fn new() -> Self {
        Self { last_ms: None }
    }

    pub fn mark(&mut self, now_ms: u64) {
        self.last_ms = Some(now_ms);
    }

    /// True (and clears) when [`IDLE_MS`] has passed with no further marks.
    pub fn poll(&mut self, now_ms: u64) -> bool {
        match self.last_ms {
            Some(t) if now_ms.saturating_sub(t) >= Self::IDLE_MS => {
                self.last_ms = None;
                true
            }
            _ => false,
        }
    }

    /// True (and clears) if a write is pending — leave Mods / quit.
    pub fn take(&mut self) -> bool {
        self.last_ms.take().is_some()
    }
}

impl Default for Debounce {
    fn default() -> Self {
        Self::new()
    }
}

/// Pull a leading `v<N>;` version prefix off a saved mod blob. No prefix
/// (or a prefix that isn't a `u16`) is version 0, the pre-versioning format.
pub(crate) fn split_mod_version(data: &str) -> (u16, &str) {
    let Some(rest) = data.strip_prefix('v') else {
        return (0, data);
    };
    let Some((n, payload)) = rest.split_once(';') else {
        return (0, data);
    };
    match n.parse::<u16>() {
        Ok(version) => (version, payload),
        Err(_) => (0, data),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::split_mod_version;
    use super::testing::Stub;
    use crate::menu::Menu;
    use crate::world::terrain::TerrainCfg;
    use crate::world::World;

    fn payload_cfg(mods: &Mods) -> TerrainCfg {
        mods.worldgen_config()
            .as_deref()
            .map(TerrainCfg::from_text)
            .unwrap_or_default()
    }

    /// Every stand-in, in install order.
    const BUILTINS: [&str; 10] = super::testing::STANDARD_IDS;

    #[test]
    fn worldgen_kind_skips_non_worldgen_mods() {
        let mods = crate::modding::testing::standard();
        assert_eq!(mods.worldgen_kind(), WorldgenKind::Diffusion, "InfiniteDiffusion is on by default");
        let mut off = crate::modding::testing::standard();
        off.set_enabled("diffusion", false);
        assert_eq!(off.worldgen_kind(), WorldgenKind::Flat, "the core fallback is the flat world");
    }

    #[test]
    fn effective_render_strips_disabled_visual_groups() {
        let mut mods = crate::modding::testing::standard();
        mods.set_enabled("Atmosphere", false);
        mods.set_enabled("Post", false);
        mods.set_enabled("Lighting", false);
        let settings = Settings::default();
        let stripped = mods.effective_render(&settings);
        assert!(!stripped.clouds);
        assert!(!stripped.bloom);
        assert!(!stripped.shadows);
        assert!(stripped.sunlight);
        let full = crate::modding::testing::standard().effective_render(&settings);
        assert_eq!(full.clouds, settings.clouds);
        assert_eq!(full.bloom, settings.bloom);
        assert_eq!(full.shadows, settings.shadows);
        let via_mask = mods.visual_mask().effective_render(&settings);
        assert!(!via_mask.clouds && !via_mask.bloom && !via_mask.shadows);
        assert_eq!(via_mask.sunlight, stripped.sunlight);
    }

    #[test]
    fn annotate_setting_names_the_mod_that_forced_the_lane_off() {
        let mut mods = crate::modding::testing::standard();
        mods.set_enabled("Post", false);
        let mask = mods.visual_mask();
        assert_eq!(forced_off_marker("Post"), "(off: Post mod)");
        assert_eq!(
            annotate_setting("On".to_string(), "bloom", mask),
            format!("On {}", forced_off_marker("Post"))
        );
        assert_eq!(annotate_setting("On".to_string(), "shadows", mask), "On");
        mods.set_enabled("Lighting", false);
        let mask = mods.visual_mask();
        assert_eq!(
            annotate_setting("On".to_string(), "shadows", mask),
            format!("On {}", forced_off_marker("Lighting"))
        );
    }

    #[test]
    fn set_enabled_keys_on_id_case_insensitively() {
        let mut mods = crate::modding::testing::standard();
        let i = (0..mods.len())
            .find(|&i| mods.id(i) == WorldgenKind::Diffusion.id())
            .expect("InfiniteDiffusion is installed");
        assert_eq!(mods.name(i), "InfiniteDiffusion");
        assert_ne!(mods.id(i), mods.name(i));
        mods.set_enabled("diffusion", true);
        assert_eq!(mods.worldgen_kind(), WorldgenKind::Diffusion);
        mods.set_enabled("DIFFUSION", false);
        assert_eq!(mods.worldgen_kind(), WorldgenKind::Flat);
        mods.set_enabled("InfiniteDiffusion", true);
        assert_eq!(
            mods.worldgen_kind(),
            WorldgenKind::Flat,
            "display name is not a set_enabled key"
        );
    }

    #[test]
    fn split_mod_version_reads_prefix_and_treats_absent_as_zero() {
        assert_eq!(split_mod_version("Stone,Iron"), (0, "Stone,Iron"));
        assert_eq!(split_mod_version("v1;Stone,Iron"), (1, "Stone,Iron"));
        assert_eq!(split_mod_version("v0;"), (0, ""));
        assert_eq!(split_mod_version("v12;a=b"), (12, "a=b"));
        assert_eq!(split_mod_version("v;nope"), (0, "v;nope"));
        assert_eq!(split_mod_version("vx;nope"), (0, "vx;nope"));
    }

    #[test]
    fn save_states_key_by_id_and_load_accepts_display_name() {
        let mut world = World::new(1);
        let mut mods = crate::modding::testing::standard();
        let rock = world.registry().id_by_label("rock").unwrap();
        let spec = world.registry().spec(rock);
        mods.load_state("Hotbar", &format!("sel=2;2={spec}"), &mut world);
        let saved = mods.save_states(&world);
        assert!(saved.iter().any(|(k, _)| k == "hotbar"), "save keys are stable ids, not display names");
        assert!(!saved.iter().any(|(k, _)| k == "Hotbar"));
        let data = saved.iter().find(|(k, _)| k == "hotbar").map(|(_, d)| d.clone()).expect("hotbar persists");
        for key in ["hotbar", "Hotbar"] {
            let mut fresh = crate::modding::testing::standard();
            fresh.load_state(key, &data, &mut world);
            assert_eq!(
                fresh.save_states(&world).iter().find(|(k, _)| k == "hotbar").map(|(_, d)| d.as_str()),
                Some(data.as_str())
            );
        }
    }

    #[test]
    fn mod_state_round_trips_version_prefix() {
        let mut world = World::new(1);
        let mut mods = crate::modding::testing::standard();
        let rock = world.registry().id_by_label("rock").unwrap();
        let spec = world.registry().spec(rock);
        mods.load_state("hotbar", &format!("v1;sel=1;1={spec}"), &mut world);
        let saved = mods.save_states(&world);
        assert!(saved.iter().all(|(k, _)| k != "inventory"), "the inventory is core state, not an inventory save line");
        let bar = saved.iter().find(|(k, _)| k == "hotbar").map(|(_, d)| d.as_str()).expect("hotbar");
        assert_eq!(bar, format!("v1;sel=1;1={spec}"));
        let mut fresh = crate::modding::testing::standard();
        for (k, v) in &saved {
            fresh.load_state(k, v, &mut world);
        }
        assert_eq!(fresh.save_states(&world), saved);
    }

    fn index_of(mods: &Mods, id: &str) -> usize {
        (0..mods.len())
            .find(|&i| mods.id(i) == id)
            .unwrap_or_else(|| panic!("missing mod {id}"))
    }

    fn temp_choices_path() -> std::path::PathBuf {
        crate::save::store::test_temp_path("mods").with_extension("cfg")
    }

    #[test]
    fn choices_text_round_trips_and_ignores_junk() {
        let mut mods = crate::modding::testing::standard();
        let defaults = mods.choices_text();
        for id in BUILTINS {
            assert!(defaults.contains(&format!("{id}=on")), "{id} is on by default:\n{defaults}");
        }
        assert!(defaults.contains("neural_textures.state=detail=1.0,contrast=1.0"));
        assert!(defaults.contains("material_names.state=style=mineral"));
        assert!(defaults.contains("diffusion.state=relief=100,caves=100,mines=100,space=100"));
        assert!(defaults.starts_with("version=2\n"));

        mods.set_enabled("lighting", false);
        let i = index_of(&mods, "diffusion");
        mods.step_knob(i, 0, 1);
        let cfg = payload_cfg(&mods);
        assert_eq!(cfg.relief, 125);
        let text = mods.choices_text();
        assert!(text.contains("lighting=off"));
        assert!(text.contains(&format!("diffusion.state={}", cfg.to_text())));

        let mut fresh = crate::modding::testing::standard();
        fresh.apply_choices_text(
            "version=2\nlighting=off\nnot-a-mod=on\nmenus=nope\n\ninventory=off\ndiffusion=on\ndiffusion.state=relief=150\nunknown.state=tile=16\n",
        );
        let restored = fresh.choices_text();
        assert!(restored.contains("lighting=off"));
        assert!(restored.contains("inventory=off"));
        assert!(restored.contains("diffusion=on"));
        assert!(restored.contains("menus=on"), "malformed value must not change the default");
        assert_eq!(payload_cfg(&fresh).relief, 150);
    }

    /// A pre-marker file wrote `diffusion=off` (and an unrelated knob payload) for everyone:
    /// it must not switch off the world generator, while its other choices still apply.
    #[test]
    fn version_one_choices_keep_the_world_generator() {
        let mut mods = crate::modding::testing::standard();
        mods.apply_choices_text("lighting=off\ndiffusion=off\ndiffusion.state=tile=16,stride=16,phases=8,relief=1.00\ncrafting=on\n");
        let text = mods.choices_text();
        assert!(text.starts_with("version=2\n"));
        assert!(text.contains("lighting=off"));
        assert!(text.contains("diffusion=on"), "{text}");
        assert_eq!(payload_cfg(&mods).relief, 100);
        mods.apply_choices_text(&text.replace("diffusion=on", "diffusion=off"));
        assert!(mods.choices_text().contains("diffusion=off"), "a current file's choice applies");
    }

    #[test]
    fn choices_file_round_trips_toggles_and_knobs() {
        let path = temp_choices_path();
        let mut mods = crate::modding::testing::standard();
        let i = index_of(&mods, "diffusion");
        mods.set_enabled("lighting", false);
        mods.step_knob(i, 0, 1);
        mods.step_knob(i, 1, -1);
        let cfg = payload_cfg(&mods);
        mods.save_choices_to(&path).unwrap();
        let tmp = {
            let mut name = path.as_os_str().to_os_string();
            name.push(".tmp");
            std::path::PathBuf::from(name)
        };
        assert!(!tmp.exists(), "atomic save must not leave a .tmp");

        let mut fresh = crate::modding::testing::standard();
        fresh.load_choices_from(&path);
        let _ = fs::remove_file(&path);
        assert!(!fresh.is_enabled(index_of(&fresh, "lighting")));
        assert!(fresh.is_enabled(index_of(&fresh, "diffusion")));
        assert_eq!(payload_cfg(&fresh), cfg);
    }

    #[test]
    fn unknown_choice_ids_are_ignored() {
        let mut mods = crate::modding::testing::standard();
        let before = mods.choices_text();
        mods.apply_choices_text("not-a-mod=on\nunknown.state=tile=64\nmenus=nope\n");
        assert_eq!(mods.choices_text(), before);
    }

    #[test]
    fn corrupt_choices_file_falls_back_to_defaults() {
        let defaults = crate::modding::testing::standard().choices_text();

        let bad_utf8 = temp_choices_path();
        fs::write(&bad_utf8, [0xff, 0xfe, 0x00, 0x01]).unwrap();
        let mut mods = crate::modding::testing::standard();
        mods.load_choices_from(&bad_utf8);
        let _ = fs::remove_file(&bad_utf8);
        assert_eq!(mods.choices_text(), defaults);

        let garbage = temp_choices_path();
        fs::write(&garbage, "{{{{ not a config\n!!!\n").unwrap();
        let mut mods = crate::modding::testing::standard();
        mods.load_choices_from(&garbage);
        let _ = fs::remove_file(&garbage);
        assert_eq!(mods.choices_text(), defaults);

        let mut mods = crate::modding::testing::standard();
        mods.load_choices_from(Path::new("/tmp/watt-mods-does-not-exist.cfg"));
        assert_eq!(mods.choices_text(), defaults);
    }

    #[test]
    fn apply_bench_env_pins_worldgen_and_visuals() {
        let mut mods = crate::modding::testing::standard();
        mods.apply_bench_env(Some(true), Some(true));
        assert_eq!(mods.worldgen_kind(), WorldgenKind::Diffusion);
        let mask = mods.visual_mask();
        assert!(!mask.atmosphere && !mask.post && !mask.lighting);
        mods.apply_bench_env(Some(false), Some(false));
        assert_eq!(mods.worldgen_kind(), WorldgenKind::Flat);
        let mask = mods.visual_mask();
        assert!(!mask.atmosphere && !mask.post && !mask.lighting);
    }

    #[test]
    fn apply_bench_env_does_not_write_choices() {
        let path = temp_choices_path();
        let mut mods = crate::modding::testing::standard();
        mods.save_choices_to(&path).unwrap();
        let on_disk = fs::read_to_string(&path).unwrap();
        assert!(on_disk.contains("diffusion=on"));
        mods.apply_bench_env(Some(false), Some(true));
        assert_eq!(mods.worldgen_kind(), WorldgenKind::Flat);
        assert_eq!(fs::read_to_string(&path).unwrap(), on_disk);
        assert!(mods.choices_text().contains("diffusion=off"));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn essentials_lists_every_member_in_install_order() {
        let mods = crate::modding::testing::standard();
        let groups: Vec<&Group> = mods.groups().collect();
        assert_eq!(groups.len(), 1, "only the well-known group when no package declares one");
        let g = groups[0];
        assert_eq!(g.id, ESSENTIALS);
        assert_eq!(g.name, "Essentials");
        assert!(
            g.description.chars().count() <= 60,
            "group description must fit the mods panel: {} chars",
            g.description.chars().count()
        );
        let members: Vec<&str> = (0..mods.len())
            .filter(|&i| mods.group(i) == ESSENTIALS)
            .map(|i| mods.id(i))
            .collect();
        assert_eq!(members, BUILTINS);
        assert!((0..mods.len()).all(|i| mods.group_of(i) == Some(ESSENTIALS_GROUP)));
    }

    #[test]
    fn group_toggle_persists_each_member_line() {
        let path = temp_choices_path();
        let mut mods = crate::modding::testing::standard();
        mods.set_group_enabled(ESSENTIALS, false);
        let text = mods.choices_text();
        for id in BUILTINS {
            assert!(
                text.contains(&format!("{id}=off")),
                "{id} should be off in:\n{text}"
            );
        }
        assert!(
            !text.lines().any(|l| l.starts_with("essentials=")),
            "group toggle must not write a group-level key"
        );
        mods.save_choices_to(&path).unwrap();

        let mut fresh = crate::modding::testing::standard();
        fresh.load_choices_from(&path);
        let _ = fs::remove_file(&path);
        for i in 0..fresh.len() {
            assert!(!fresh.is_enabled(i), "{} still on", fresh.id(i));
        }

        fresh.set_group_enabled(ESSENTIALS, true);
        let on_text = fresh.choices_text();
        for id in BUILTINS {
            assert!(
                on_text.contains(&format!("{id}=on")),
                "{id} should be on in:\n{on_text}"
            );
        }
    }

    #[test]
    fn server_hold_turns_the_package_off_without_changing_saved_choices() {
        let mut mods = Mods::empty();
        mods.install_from(Some("pwc.dev-toolkit"), Box::new(Stub::new("tools")), true);
        let before = mods.choices_text();
        assert!(before.contains("tools=on"));
        mods.hold_packages(&["pwc.dev-toolkit".to_string()]);
        assert!(!mods.is_enabled(0));
        assert!(mods.server_off(0));
        assert!(!mods.toggle(0), "a held mod cannot be turned back on");
        assert!(!mods.is_enabled(0));
        assert_eq!(mods.choices_text(), before, "the hold is not a saved choice");
        mods.release_server();
        assert!(mods.is_enabled(0));
        assert!(!mods.server_off(0));
        assert!(mods.toggle(0));
    }

    fn register_tools(r: &mut ModRegistrar) {
        r.add(Stub::new("tools"));
    }

    fn register_hud(r: &mut ModRegistrar) {
        r.add(Stub::new("hud"));
    }

    fn register_nothing(_: &mut ModRegistrar) {}

    #[test]
    fn hello_reports_only_enabled_mod_packages() {
        const fn package(id: &'static str, kind: PackageKind, register: Option<fn(&mut ModRegistrar)>) -> PackageInfo {
            PackageInfo { id, name: id, version: "1.0.0", description: "", kind, dependencies: &[], register }
        }
        static PACKAGES: &[PackageInfo] = &[
            package("test.names", PackageKind::Library, None),
            package("test.tools", PackageKind::Mod, Some(register_tools)),
            package("test.hud", PackageKind::Mod, Some(register_hud)),
            package("test.empty", PackageKind::Mod, Some(register_nothing)),
            package("test.bundle", PackageKind::Bundle, None),
        ];
        let build = GameBuild::from_static("sha256:02", PACKAGES);
        let mut mods = build.mods();
        let report = |mods: &Mods| mods.enabled_package_reports(build.packages());
        let both = vec![("test.tools".to_string(), "1.0.0".to_string()), ("test.hud".to_string(), "1.0.0".to_string())];
        assert_eq!(report(&mods), both, "libraries, bundles and mod packages with no mod are not reported");
        mods.hold_packages(&["test.tools".to_string()]);
        assert_eq!(report(&mods), both[1..], "a held package is off");
        mods.release_server();
        assert_eq!(report(&mods), both);
    }

    #[test]
    fn worldgen_config_is_the_winning_kind_payload() {
        let mut off = crate::modding::testing::standard();
        off.set_enabled("diffusion", false);
        assert_eq!(off.worldgen_kind(), WorldgenKind::Flat);
        assert_eq!(off.worldgen_config(), None);
        let mut on = crate::modding::testing::standard();
        let text = on.worldgen_config().expect("payload");
        assert_eq!(TerrainCfg::from_text(&text), TerrainCfg::default());
        on.step_knob(index_of(&on, "diffusion"), 3, 1);
        let cfg = TerrainCfg::from_text(&on.worldgen_config().unwrap());
        assert_eq!(cfg.space, 125);
    }

    #[test]
    fn fallback_theme_with_essentials_disabled() {
        let mut mods = crate::modding::testing::standard();
        mods.set_group_enabled(ESSENTIALS, false);
        assert!(mods.menu_theme().is_none());
        let fallback = crate::menu::theme::DefaultTheme;
        let theme: &dyn crate::menu::theme::MenuTheme = mods.menu_theme().unwrap_or(&fallback);
        let snap = crate::menu::ModRow::snapshot(&mods);
        let mut settings = crate::settings::Settings::default();
        let session = crate::session::Session::default();
        let ctx = crate::menu::Ctx {
            settings: &mut settings,
            saves: &[],
            mods: &snap,
            session: &session,
            mods_save_error: None,
        };
        let view = crate::menu::menus::ModsMenu.view(&ctx);
        let pv = crate::menu::present(&view, 1.0);
        let rects = theme.layout(&pv, 1280, 720);
        assert_eq!(rects.len(), view.rows.len());
        assert!(matches!(view.rows[0].kind, crate::menu::RowKind::Heading));
        let mut cursor = crate::menu::Cursor::default();
        cursor.normalize(&view);
        assert!(view.is_selectable(cursor.index));
        assert_ne!(cursor.index, 0, "cursor must skip the group header");
    }

    /// A mod that asks for the keyboard on its action and logs what it types and is told.
    struct Typist {
        id: &'static str,
        log: std::rc::Rc<std::cell::RefCell<Vec<String>>>,
        shows: bool,
    }

    impl Mod for Typist {
        fn name(&self) -> &str {
            self.id
        }
        fn id(&self) -> &'static str {
            self.id
        }
        fn on_frame(&mut self, ctx: &mut FrameContext) {
            if ctx.action("open") {
                let got = ctx.capture_text(true);
                self.log.borrow_mut().push(format!("{} open {got}", self.id));
            }
            if let Some(text) = ctx.text() {
                let typed: String = text.chars.iter().collect();
                self.log.borrow_mut().push(format!("{} typed {typed} esc={}", self.id, text.escape));
            }
        }
        fn on_message(&mut self, msg: &Message) -> bool {
            if let Message::Notice(notice) = msg {
                self.log.borrow_mut().push(format!("{} notice {}", self.id, notice.text));
            }
            self.shows
        }
    }

    fn typists(shows: bool) -> (Mods, std::rc::Rc<std::cell::RefCell<Vec<String>>>) {
        let log = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let mut mods = Mods::empty();
        for id in ["a", "b"] {
            mods.install(Box::new(Typist { id, log: log.clone(), shows }), true);
        }
        (mods, log)
    }

    /// Run one frame of `mods` with `action` fired and `text` typed (as the core would pass it:
    /// only while a capture was held when the input was read).
    fn frame(mods: &mut Mods, action: bool, text: Option<TextFrame<'_>>) {
        let mut world = World::new(1);
        let mut player = Player::new(glam::DVec3::ZERO);
        let (mut settings, mut sky) = (Settings::default(), crate::sky::Sky::new());
        let game = GameContext::new(&mut player, &mut world, &mut settings, &mut sky);
        let mut ctx = FrameContext::frame(game, (800, 600), ActionSet::NONE, &[], text);
        if action {
            ctx.set_action("open");
        }
        mods.on_frame(&mut ctx);
    }

    #[test]
    fn the_keyboard_belongs_to_the_first_mod_that_asks_until_escape() {
        let (mut mods, log) = typists(false);
        assert!(!mods.text_captured());
        frame(&mut mods, true, None);
        assert!(mods.text_captured());
        assert_eq!(log.take(), ["a open true", "b open false"], "the first asker holds it; the second is refused");

        let typed = ['h', 'i'];
        frame(&mut mods, false, Some(TextFrame { chars: &typed, edit: None, escape: false }));
        assert_eq!(log.take(), ["a typed hi esc=false"], "only the holder sees the typing");

        frame(&mut mods, false, Some(TextFrame { chars: &[], edit: None, escape: true }));
        assert_eq!(log.take(), ["a typed  esc=true"], "the holder sees Escape once");
        assert!(!mods.text_captured(), "Escape ends the capture");

        frame(&mut mods, true, None);
        assert!(mods.text_captured());
        mods.reset_state();
        assert!(!mods.text_captured(), "a new world starts with the keyboard free");
        frame(&mut mods, true, None);
        mods.toggle(0);
        frame(&mut mods, false, None);
        assert!(!mods.text_captured(), "a holder that was switched off loses the keyboard");
    }

    #[test]
    fn messages_reach_every_enabled_mod_and_say_whether_one_showed_them() {
        let notice = Notice { level: NoticeLevel::Info, text: "saved".into() };
        let (mut quiet, log) = typists(false);
        assert!(!quiet.on_message(&Message::Notice(&notice)), "nobody showed it: the core logs it");
        assert_eq!(log.take(), ["a notice saved", "b notice saved"]);
        let (mut shown, _) = typists(true);
        shown.toggle(1);
        assert!(shown.on_message(&Message::Notice(&notice)));
        assert!(!Mods::empty().on_message(&Message::Joined { name: "x" }));
    }

    /// The settings count as changed only when a hook changed a value, not when it looked.
    #[test]
    fn a_game_context_reports_changed_settings_and_queues_chat_and_notices() {
        let mut world = World::new(1);
        let mut player = Player::new(glam::DVec3::ZERO);
        let (mut settings, mut sky) = (Settings::default(), crate::sky::Sky::new());
        let mut game = GameContext::new(&mut player, &mut world, &mut settings, &mut sky);
        let fov = game.settings().fov;
        assert!(!game.settings_changed());
        game.settings_mut().fov = fov;
        assert!(!game.settings_changed(), "writing the same value changes nothing");
        game.settings_mut().fov = fov + 5.0;
        assert!(game.settings_changed());
        game.send_chat(Channel::Global, "/op hunter2");
        game.notice(NoticeLevel::Warning, "careful");
        assert_eq!(game.chat_out(), [(Channel::Global, "/op hunter2".to_string())]);
        assert_eq!(game.notices().iter().map(|n| n.text.as_str()).collect::<Vec<_>>(), ["careful"]);
        assert_eq!((Channel::Global.wire(), Channel::from_wire(Channel::Local.wire())), (crate::net::chat::GLOBAL, Channel::Local));
    }

    #[test]
    fn debounce_waits_250ms_then_resets_on_mark() {
        let mut flush = Debounce::new();
        assert!(!flush.poll(0));
        flush.mark(0);
        assert!(!flush.poll(249));
        assert!(flush.poll(250));
        assert!(!flush.poll(500), "already flushed");

        flush.mark(0);
        flush.mark(200);
        assert!(!flush.poll(449));
        assert!(flush.poll(450));

        flush.mark(10);
        assert!(flush.take());
        assert!(!flush.take());
        assert!(!flush.poll(10 + Debounce::IDLE_MS));
    }

    /// Every host change the mods screen can show moves the revision; reading does not.
    #[test]
    fn revision_moves_with_what_the_mods_screen_lists() {
        let mut mods = crate::modding::testing::standard();
        let mut last = mods.revision();
        let _ = (mods.visual_mask(), mods.choices_text(), mods.knobs(index_of(&mods, "diffusion")));
        assert_eq!(mods.revision(), last, "reads");
        let mut moved = |mods: &Mods, what: &str| {
            assert_ne!(mods.revision(), last, "{what}");
            last = mods.revision();
        };
        mods.set_enabled("post", false);
        moved(&mods, "a switch");
        mods.step_knob(index_of(&mods, "diffusion"), 0, 1);
        moved(&mods, "a knob step");
        mods.apply_choices_text("version=2\ndiffusion.state=relief=150\n");
        moved(&mods, "a loaded knob payload");
        mods.hold_packages(&["pwc.visuals".to_string()]);
        moved(&mods, "a server hold");
        mods.release_server();
        moved(&mods, "a release");
        mods.install(Box::new(Stub::new("extra")), false);
        moved(&mods, "an install");
    }

    fn enabled(mods: &Mods, name: &str) -> bool {
        (0..mods.len())
            .find(|&i| mods.name(i) == name)
            .map(|i| mods.is_enabled(i))
            .expect("installed mod")
    }

    #[test]
    fn choices_round_trip_through_the_config_root_and_ignore_unknown() {
        let mut mods = crate::modding::testing::standard();
        mods.set_enabled("diffusion", true);
        mods.set_enabled("atmosphere", false);
        mods.save_choices().unwrap();
        let path = crate::paths::Paths::get().mods_file();
        assert!(path.exists());
        assert!(path.starts_with(&crate::paths::Paths::get().config));
        assert_ne!(path, std::path::PathBuf::from("saves/mods.cfg"));

        let mut loaded = crate::modding::testing::standard();
        loaded.load_choices();
        assert!(enabled(&loaded, "InfiniteDiffusion"));
        assert!(!enabled(&loaded, "Atmosphere"));
        assert!(enabled(&loaded, "Inventory"));

        fs::write(&path, "no-such=on\ninventory=off\nnot-a-pair\natmosphere=true\n").unwrap();
        let mut parsed = crate::modding::testing::standard();
        parsed.load_choices();
        assert!(!enabled(&parsed, "Inventory"));
        assert!(enabled(&parsed, "Atmosphere"));
        assert!(enabled(&parsed, "InfiniteDiffusion"), "an unmentioned mod keeps its default (on)");
        let _ = fs::remove_file(path);
    }
}
