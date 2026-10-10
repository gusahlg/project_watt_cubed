//! The mod host: the game's "minimal core, layers on top" made real. Core
//! gameplay owns the world, the law and physics; everything player-facing that
//! isn't essential — the inventory panel, block looks and names, HUD
//! widgets, menus — is a [`Mod`]. The build decides which mods are in: there is no
//! runtime switch. The core can only *suspend* a package for one session, when the
//! server it joins refuses it (or a benchmark pins it off); suspension has no UI and is
//! never saved.
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
//! suspended mods are skipped entirely. A mod therefore costs nothing where it would
//! matter and only what it draws where it wouldn't.
mod build;
#[cfg(test)]
pub(crate) mod testing;

pub use build::{BuildInfo, GameBuild, ModDescriptor, ModRegistrar, PackageInfo, PackageKind};

use crate::block::appearance::{BlockAppearance, FLAT};
use crate::block::naming::MaterialNamer;
use crate::block::BlockId;
use crate::menu::start::{StartFacts, StartScreen};
use crate::menu::theme::MenuTheme;
use crate::player::Player;
use crate::render_config::{RenderConfig, VisualGroup};
use crate::settings::Settings;
use crate::sky::Sky;
use crate::ui::{HudElement, Line};
use crate::world::generation::WorldgenKind;
use crate::world::World;

/// Which fancy visual groups the installed, unsuspended mods provide.
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

    /// Every group provided.
    pub const ALL: Self = Self {
        atmosphere: true,
        post: true,
        lighting: true,
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

    /// Whether this mask strips the settings lane `key`: no installed, unsuspended mod provides
    /// its visual group. False for a lane outside every group.
    pub fn strips(self, key: &str) -> bool {
        crate::render_config::lane_group(key).is_some_and(|group| !self.get(group))
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

/// One console command a mod handles: what `/help` lists and Tab completes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Command {
    /// The name typed after the `/` (aliases are the handling mod's own business).
    pub name: &'static str,
    /// The arguments as `/help` shows them (`<x y z|name>`), or `""`.
    pub args: &'static str,
    /// What the command does, in a few words.
    pub help: &'static str,
}

/// What a console command may read and change: the whole-game state the core owns. A command only
/// edits it; the core follows up on what changed (applies and saves changed settings, re-mixes the
/// audio, shares a changed clock with the server, streams a moved player's surroundings at once
/// and reports the move as a teleport). Build one with [`CommandContext::new`].
#[non_exhaustive]
pub struct CommandContext<'a> {
    pub player: &'a mut Player,
    pub world: &'a mut World,
    pub settings: &'a mut Settings,
    pub sky: &'a mut Sky,
    /// Which visual groups the active mods provide (`/gfx` marks a lane no mod provides).
    pub visuals: VisualMask,
    /// True when a server owns the session: it sets the day length and may refuse a teleport.
    pub networked: bool,
    /// Every enabled mod's commands, in install order (for `/help`).
    pub commands: &'a [Command],
    /// Set to play the local voice test cue (the core owns the audio).
    pub voice_test: bool,
}

impl<'a> CommandContext<'a> {
    /// A singleplayer context over these handles, with every visual group on and no command list.
    pub fn new(player: &'a mut Player, world: &'a mut World, settings: &'a mut Settings, sky: &'a mut Sky) -> Self {
        Self {
            player,
            world,
            settings,
            sky,
            visuals: VisualMask::default(),
            networked: false,
            commands: &[],
            voice_test: false,
        }
    }
}

/// A unit of layered-on functionality. Every method but [`id`](Self::id) has a default, so a mod
/// implements only the hooks it cares about. This is the public surface mod authors write
/// against — kept small on purpose.
///
/// Arbitration when more than one active mod implements a hook (a mod is active unless the core
/// suspended its package for the session):
/// - **Fan-out**, install order: `update`, `on_block_break`, `on_break_rejected`,
///   `on_place_rejected`, `on_tool_changed`, `on_tool_used`. `hud` uses the same order as z-order
///   (later draws on top). `commands` lists concatenate in the same order. `actions` are collected,
///   not arbitrated: each active mod's list is its own.
/// - **First active wins**: `menu_theme`, `start_screen`, `close_overlay` and `on_toggle_fly`
///   (first `true`),
///   `worldgen`, `worldgen_config`, `appearance`, `namer`, `tool`, `run_command` (first `Some`).
/// - **Compose**: `visual_group` bits OR into the render mask.
///
/// Save hooks are per-mod. `worldgen_config` is an opaque string; the winning worldgen kind
/// parses it. What a package is called and what it does are its `mod.toml` (see
/// [`ModRegistrar::package`]); a mod has no display text of its own.
pub trait Mod {
    /// Name for logs, and the key old saves used before mods had ids. Defaults to
    /// [`id`](Self::id).
    fn name(&self) -> &str {
        self.id()
    }

    /// Stable lowercase code id. Per-world saves key on it; [`name`](Self::name) is only a
    /// fallback for saves older than ids.
    fn id(&self) -> &'static str;

    /// Clear per-world state (crafted blocks, open panels) when entering a
    /// different world. The inventory lives on the player, not here.
    fn reset(&mut self) {}

    /// Cadence-controlled logic while active (the game's `mod_hz`). Runs
    /// after movement, before rendering; edge inputs accumulated between
    /// ticks are replayed in order without loss.
    fn update(&mut self, ctx: &mut ModContext) {
        let _ = ctx;
    }

    /// The flight key (`F`) was pressed. The core has no flight toggle of its own: a mod that
    /// offers flight switches it here and returns `true`; the first active mod that does wins.
    /// Delivered on the frame of the press (whatever the mod cadence), never while a detached
    /// camera holds the player.
    fn on_toggle_fly(&mut self, player: &mut Player, world: &World) -> bool {
        let _ = (player, world);
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

    /// Controls this mod wants sampled while it is active. The core owns the chord
    /// table; a core binding wins any clash. Default is none.
    fn actions(&self) -> &[Action] {
        &[]
    }

    /// The configuration a primary action applies, if any (first active mod that answers
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

    /// Optional material namer. First active mod that returns `Some` names every
    /// configuration; without one the core describes materials by their readings.
    fn namer(&self) -> Option<&dyn MaterialNamer> {
        None
    }

    /// A console command without access to the game state; the default
    /// [`run_command`](Self::run_command) asks this.
    fn command(&mut self, cmd: &str, args: &[&str]) -> Option<Vec<Line>> {
        let _ = (cmd, args);
        None
    }

    /// Handle the console command `cmd` (the leading `/` stripped) with `args`. The first active
    /// mod that returns `Some` handles it; its lines go to the console.
    fn run_command(&mut self, ctx: &mut CommandContext<'_>, cmd: &str, args: &[&str]) -> Option<Vec<Line>> {
        let _ = ctx;
        self.command(cmd, args)
    }

    /// The commands this mod handles, for `/help` and Tab completion.
    fn commands(&self) -> &[Command] {
        &[]
    }

    /// This mod's HUD contribution while active, as data — [`HudElement`]s
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

    /// Optional start screen. First active mod that returns `Some` wins;
    /// the core fallback (New world / Load / Settings / Mods / Quit) is used
    /// when every active mod returns `None`. Plain-data signatures only.
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

    /// Which fancy render group this mod provides, if any. The host reads it when the mod is
    /// installed and when a suspension starts or ends.
    fn visual_group(&self) -> Option<VisualGroup> {
        None
    }

    /// If this mod replaces worldgen, the kind new worlds use while it is active.
    fn worldgen(&self) -> Option<WorldgenKind> {
        None
    }

    /// Opaque payload for the winning [`worldgen`](Self::worldgen) kind. `None` if this mod
    /// does not replace worldgen. InfiniteDiffusion builds it from its options.
    fn worldgen_config(&self) -> Option<String> {
        None
    }

    /// Optional block appearance. First active mod that returns `Some` wins;
    /// [`FlatAppearance`](crate::block::appearance::FlatAppearance) is used
    /// when every active mod returns `None`.
    fn appearance(&self) -> Option<&dyn BlockAppearance> {
        None
    }
}

/// One installed mod and whether it runs this session.
struct Entry {
    module: Box<dyn Mod>,
    /// Id of the package that registered it (`None` for mods installed directly).
    package: Option<&'static str>,
    /// False while the core suspends its package (see [`Mods::suspend_packages`]).
    active: bool,
}

/// The installed mods. Every mod the build installs runs, unless the core suspends its package
/// for the session; per-world state is saved through each mod's `save_state`/`load_state`.
pub struct Mods {
    entries: Vec<Entry>,
    /// Bumped when a mod is installed or a suspension starts or ends, so the input
    /// table can rebuild once instead of every frame.
    action_gen: u64,
    /// The active mods' visual groups, rebuilt with `action_gen`.
    visuals: VisualMask,
    /// Bumped whenever what a mods screen lists may have changed.
    revision: u64,
    /// Package ids suspended for the whole process (`WATT_BENCH_SUSPEND`). Never saved.
    pinned: Vec<String>,
    /// Package ids suspended now: the pinned ones plus what the current server refused. Never
    /// saved.
    suspended: Vec<String>,
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
            action_gen: 0,
            visuals: VisualMask::NONE,
            revision: 0,
            pinned: Vec::new(),
            suspended: Vec::new(),
        }
    }

    /// Install a mod directly (tests, the vanilla harness). Packages install through
    /// [`ModRegistrar::add`].
    pub fn install(&mut self, module: Box<dyn Mod>) {
        self.install_from(None, module);
    }

    fn install_from(&mut self, package: Option<&'static str>, module: Box<dyn Mod>) {
        let active = !package.is_some_and(|pkg| self.suspended.iter().any(|id| id == pkg));
        self.entries.push(Entry { module, package, active });
        self.active_changed();
    }

    /// The active set changed: a new action generation and visual mask.
    fn active_changed(&mut self) {
        self.action_gen = self.action_gen.wrapping_add(1);
        let active = self.entries.iter().filter(|e| e.active);
        self.visuals = VisualMask::of(active.filter_map(|e| e.module.visual_group()));
        self.revise();
    }

    fn revise(&mut self) {
        self.revision = self.revision.wrapping_add(1);
    }

    /// Generation of the active action lists. Changes when a mod is installed or a suspension
    /// starts or ends.
    pub fn action_generation(&self) -> u64 {
        self.action_gen
    }

    /// Changes whenever what a mods screen lists may have changed: a mod installed, or a
    /// suspension started or ended.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Actions of every active mod, in install order.
    pub fn enabled_actions(&self) -> impl Iterator<Item = &Action> + '_ {
        self.entries.iter().filter(|e| e.active).flat_map(|e| e.module.actions())
    }

    /// The package that registered the mod at `index`, if any.
    pub fn package(&self, index: usize) -> Option<&'static str> {
        self.entries[index].package
    }

    /// Reset every mod's per-world state (entering a new/loaded/networked world).
    pub fn reset_state(&mut self) {
        for entry in &mut self.entries {
            entry.module.reset();
        }
    }

    fn each_active(&mut self, mut f: impl FnMut(&mut dyn Mod)) {
        for entry in &mut self.entries {
            if entry.active {
                f(&mut *entry.module);
            }
        }
    }

    /// Run every active mod's per-frame logic.
    pub fn update(&mut self, ctx: &mut ModContext) {
        self.each_active(|m| m.update(ctx));
    }

    /// The flight key: the first active mod that handles it wins. False when none does.
    pub fn on_toggle_fly(&mut self, player: &mut Player, world: &World) -> bool {
        self.entries.iter_mut().filter(|e| e.active).any(|e| e.module.on_toggle_fly(player, world))
    }

    /// Fan a block-break event out to every active mod.
    pub fn on_block_break(&mut self, id: BlockId, world: &World, overflow: bool) {
        self.each_active(|m| m.on_block_break(id, world, overflow));
    }

    /// Fan a rejected-break rollback out to every active mod.
    pub fn on_break_rejected(&mut self, id: BlockId) {
        self.each_active(|m| m.on_break_rejected(id));
    }

    /// Fan a rejected-placement refund out to every active mod.
    pub fn on_place_rejected(&mut self, id: crate::block::BlockId, world: &World) {
        self.each_active(|m| m.on_place_rejected(id, world));
    }

    /// Fan a held unit's change of configuration out to every active mod.
    pub fn on_tool_changed(&mut self, old: BlockId, new: BlockId) {
        self.each_active(|m| m.on_tool_changed(old, new));
    }

    /// Fan a finished primary action out to every active mod.
    pub fn on_tool_used(&mut self, outcome: ToolUse) {
        self.each_active(|m| m.on_tool_used(outcome));
    }

    /// Fan one game fact out to every active mod.
    pub fn on_game_event(&mut self, ev: &crate::audio::GameEvent, audio: &mut crate::audio::AudioApi) {
        self.each_active(|m| m.on_game_event(ev, audio));
    }

    /// The per-frame audio hook. Runs even when the frame is otherwise idle.
    pub fn on_audio(
        &mut self,
        view: &crate::audio::AudioView,
        audio: &mut crate::audio::AudioApi,
        link: &mut crate::audio::ModLink,
    ) {
        self.each_active(|m| m.on_audio(view, audio, link));
    }

    /// The configuration a primary action applies: the first active mod that answers.
    pub fn tool(&self, player: &Player) -> Option<BlockId> {
        self.entries.iter().filter(|e| e.active).find_map(|e| e.module.tool(player))
    }

    /// The first active namer, if any.
    pub fn namer(&self) -> Option<&dyn MaterialNamer> {
        self.entries.iter().filter(|e| e.active).find_map(|e| e.module.namer())
    }

    /// First active mod that handles `cmd` with its context-free [`Mod::command`] wins.
    pub fn command(&mut self, cmd: &str, args: &[&str]) -> Option<Vec<Line>> {
        self.entries.iter_mut().filter(|e| e.active).find_map(|e| e.module.command(cmd, args))
    }

    /// First active mod that handles `cmd` wins.
    pub fn run_command(&mut self, ctx: &mut CommandContext<'_>, cmd: &str, args: &[&str]) -> Option<Vec<Line>> {
        self.entries
            .iter_mut()
            .filter(|e| e.active)
            .find_map(|e| e.module.run_command(ctx, cmd, args))
    }

    /// Every active mod's commands, in install order; a name a mod earlier in that order already
    /// lists is left out (that mod handles it: `run_command` is first-wins).
    pub fn commands(&self) -> impl Iterator<Item = &Command> {
        let all = || self.entries.iter().filter(|e| e.active).flat_map(|e| e.module.commands());
        all().enumerate().filter(move |&(i, c)| !all().take(i).any(|d| d.name == c.name)).map(|(_, c)| c)
    }

    /// Push every active mod's HUD contribution into `out`, in install order
    /// (so a later mod draws over an earlier one). The caller owns `out` and
    /// clears it per frame so capacity is retained.
    pub fn hud(&self, world: &World, player: &Player, screen: (i32, i32), out: &mut Vec<HudElement>) {
        for entry in &self.entries {
            if entry.active {
                entry.module.hud(world, player, screen, out);
            }
        }
    }

    /// Give active mods first refusal on Escape. The first open overlay closes
    /// and consumes it; otherwise the game can return to its main menu.
    pub fn close_overlay(&mut self) -> bool {
        self.entries.iter_mut().filter(|e| e.active).any(|e| e.module.close_overlay())
    }

    /// The first active mod's menu theme, if any.
    pub fn menu_theme(&self) -> Option<&dyn MenuTheme> {
        self.entries.iter().filter(|e| e.active).find_map(|e| e.module.menu_theme())
    }

    /// First active mod that returns a start screen wins. `None` means the
    /// core fallback should be used.
    pub fn start_screen(&self, facts: &StartFacts) -> Option<Box<dyn StartScreen>> {
        self.entries.iter().filter(|e| e.active).find_map(|e| e.module.start_screen(facts))
    }

    /// Number of installed mods.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether there are no installed mods.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The name of the mod at `index` (its id unless it says otherwise).
    pub fn name(&self, index: usize) -> &str {
        self.entries[index].module.name()
    }

    /// The code id of the mod at `index`.
    pub fn id(&self, index: usize) -> &str {
        self.entries[index].module.id()
    }

    /// Whether the mod at `index` runs: false while its package is suspended.
    pub fn is_active(&self, index: usize) -> bool {
        self.entries[index].active
    }

    /// Visual group the mod at `index` provides, if it is a visual mod.
    pub fn visual_group(&self, index: usize) -> Option<VisualGroup> {
        self.entries[index].module.visual_group()
    }

    /// Whether the mod at `index` replaces worldgen.
    pub fn is_worldgen(&self, index: usize) -> bool {
        self.entries[index].module.worldgen().is_some()
    }

    /// Worldgen used for the next world: the first active worldgen mod's kind, else the flat
    /// core fallback.
    pub fn worldgen_kind(&self) -> WorldgenKind {
        self.first_worldgen().and_then(|m| m.worldgen()).unwrap_or(WorldgenKind::Flat)
    }

    /// Opaque payload of the winning worldgen mod. The kind parses it
    /// (`TerrainCfg::from_text` for InfiniteDiffusion).
    pub fn worldgen_config(&self) -> Option<String> {
        self.first_worldgen().and_then(|m| m.worldgen_config())
    }

    fn first_worldgen(&self) -> Option<&dyn Mod> {
        self.entries
            .iter()
            .filter(|e| e.active)
            .find(|e| e.module.worldgen().is_some())
            .map(|e| &*e.module)
    }

    /// First active appearance mod, or the core flat fallback.
    pub fn appearance(&self) -> &dyn BlockAppearance {
        self.entries.iter().filter(|e| e.active).find_map(|e| e.module.appearance()).unwrap_or(&FLAT)
    }

    /// The visual groups the active mods provide (see [`VisualMask::of`]): installed and not
    /// suspended.
    pub fn visual_mask(&self) -> VisualMask {
        self.visuals
    }

    /// Suspend every package in `package_ids` for the session, on top of the pinned ones
    /// ([`pin_suspended`](Self::pin_suspended)): their mods stop running until
    /// [`resume_packages`](Self::resume_packages). A server that refuses packages is the
    /// authority for its session; this is how the client honours it. Nothing is saved, and
    /// there is no way to undo it but leaving. The list is what the server said; it is not a
    /// proof the client is unmodified.
    pub fn suspend_packages(&mut self, package_ids: &[String]) {
        let mut suspended = self.pinned.clone();
        for id in package_ids {
            if !suspended.contains(id) {
                suspended.push(id.clone());
            }
        }
        self.set_suspended(suspended);
    }

    /// End every session suspension. Pinned packages stay suspended.
    pub fn resume_packages(&mut self) {
        self.set_suspended(self.pinned.clone());
    }

    /// Suspend `package_ids` for the whole process (`WATT_BENCH_SUSPEND`): what
    /// [`resume_packages`](Self::resume_packages) returns to.
    pub fn pin_suspended(&mut self, package_ids: &[String]) {
        self.pinned = package_ids.to_vec();
        self.resume_packages();
    }

    fn set_suspended(&mut self, suspended: Vec<String>) {
        self.suspended = suspended;
        for entry in &mut self.entries {
            entry.active = !entry.package.is_some_and(|pkg| self.suspended.iter().any(|id| id == pkg));
        }
        self.active_changed();
    }

    /// The package ids suspended now, pinned ones first.
    pub fn suspended(&self) -> &[String] {
        &self.suspended
    }

    /// Ids of the packages whose mods provide a visual group, in install order, each once.
    pub fn visual_packages(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for entry in &self.entries {
            if let Some(pkg) = entry.package
                && entry.module.visual_group().is_some()
                && !out.iter().any(|id| id == pkg)
            {
                out.push(pkg.to_string());
            }
        }
        out
    }

    /// The mod packages this client runs, as `(id, version)`, in build order: every package of
    /// kind mod that is not suspended. Libraries and bundles are never reported. This is what an
    /// honest client puts on `Hello`.
    pub fn active_package_reports(&self, packages: &[PackageInfo]) -> Vec<(String, String)> {
        packages
            .iter()
            .filter(|p| p.kind == PackageKind::Mod && !self.suspended.iter().any(|id| id == p.id))
            .map(|p| (p.id.to_string(), p.version.to_string()))
            .collect()
    }

    /// Settings lanes with the visual groups no active mod provides stripped. The one
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
}

/// Debounces file writes so a held Left/Right does not rewrite at key-repeat rate: a write is
/// due once [`IDLE_MS`](Self::IDLE_MS) pass with no further mark.
pub struct Debounce {
    last_ms: Option<u64>,
}

impl Debounce {
    pub const IDLE_MS: u64 = 250;

    pub fn new() -> Self {
        Self { last_ms: None }
    }

    pub fn mark(&mut self, now_ms: u64) {
        self.last_ms = Some(now_ms);
    }

    /// True (and clears) when [`IDLE_MS`](Self::IDLE_MS) has passed with no further marks.
    pub fn poll(&mut self, now_ms: u64) -> bool {
        match self.last_ms {
            Some(t) if now_ms.saturating_sub(t) >= Self::IDLE_MS => {
                self.last_ms = None;
                true
            }
            _ => false,
        }
    }

    /// True (and clears) if a write is pending — leaving a world, quitting.
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
    use std::cell::Cell;
    use std::rc::Rc;

    use super::*;
    use super::split_mod_version;
    use super::testing::Stub;
    use crate::world::terrain::TerrainCfg;
    use crate::world::World;

    fn suspend(mods: &mut Mods, ids: &[&str]) {
        mods.suspend_packages(&ids.iter().map(|id| id.to_string()).collect::<Vec<_>>());
    }

    #[test]
    fn worldgen_kind_follows_the_active_worldgen_mod() {
        let mut mods = crate::modding::testing::standard();
        assert_eq!(mods.worldgen_kind(), WorldgenKind::Diffusion, "the worldgen stand-in is installed");
        suspend(&mut mods, &["pwc.infinite-diffusion"]);
        assert_eq!(mods.worldgen_kind(), WorldgenKind::Flat, "the core fallback is the flat world");
        assert_eq!(mods.worldgen_config(), None);
        mods.resume_packages();
        assert_eq!(mods.worldgen_kind(), WorldgenKind::Diffusion);
        let text = mods.worldgen_config().expect("payload");
        assert_eq!(TerrainCfg::from_text(&text), TerrainCfg::default());
        assert_eq!(Mods::empty().worldgen_kind(), WorldgenKind::Flat);
    }

    #[test]
    fn effective_render_strips_the_groups_no_active_mod_provides() {
        let mut mods = crate::modding::testing::standard();
        let settings = Settings::default();
        let full = mods.effective_render(&settings);
        assert_eq!(full.clouds, settings.clouds);
        assert_eq!(full.bloom, settings.bloom);
        assert_eq!(full.shadows, settings.shadows);
        suspend(&mut mods, &["pwc.visuals"]);
        let stripped = mods.effective_render(&settings);
        assert!(!stripped.clouds && !stripped.bloom && !stripped.shadows);
        assert!(stripped.sunlight, "a lane outside every group stays");
        assert_eq!(mods.visual_mask(), VisualMask::NONE);
        mods.resume_packages();
        assert_eq!(mods.visual_mask(), VisualMask::default(), "installed and not suspended");
    }

    /// `strips` marks exactly the lanes the renderer strips, and names no mod.
    #[test]
    fn strips_marks_exactly_the_lanes_the_mask_strips() {
        let mask = VisualMask::of([VisualGroup::Atmosphere, VisualGroup::Lighting]);
        let mut settings = Settings::default();
        settings.bloom = true;
        settings.shadows = true;
        let render = mask.effective_render(&settings);
        assert!(mask.strips("bloom") && !render.bloom);
        assert!(!mask.strips("shadows") && render.shadows);
        assert!(!mask.strips("sunlight") && !mask.strips("not-a-lane"));
        assert!(VisualMask::NONE.strips("clouds"));
        assert!(!VisualMask::default().strips("clouds"));
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

    /// A mod that counts its `update` calls.
    struct Counter {
        id: &'static str,
        calls: Rc<Cell<u32>>,
    }

    impl Mod for Counter {
        fn id(&self) -> &'static str {
            self.id
        }
        fn update(&mut self, _ctx: &mut ModContext) {
            self.calls.set(self.calls.get() + 1);
        }
    }

    /// A refused package stops running for the session and comes back on resume; nothing about
    /// it is written anywhere, and a mod's display name defaults to its id.
    #[test]
    fn a_suspended_package_skips_every_hook_until_resumed() {
        let calls = Rc::new(Cell::new(0));
        let mut mods = Mods::empty();
        mods.install_from(Some("pwc.dev-toolkit"), Box::new(Counter { id: "tools", calls: calls.clone() }));
        mods.install_from(Some("pwc.hotbar"), Box::new(Stub::new("hotbar")));
        assert_eq!(mods.name(0), "tools");
        let mut world = World::new(1);
        let mut player = Player::new(glam::DVec3::ZERO);
        let mut tick = |mods: &mut Mods| mods.update(&mut ModContext::new(&mut player, &mut world));
        tick(&mut mods);
        assert_eq!(calls.get(), 1);
        let generation = mods.action_generation();
        suspend(&mut mods, &["pwc.dev-toolkit"]);
        assert!(!mods.is_active(0) && mods.is_active(1));
        assert_eq!(mods.suspended(), ["pwc.dev-toolkit"]);
        assert_ne!(mods.action_generation(), generation, "the input table rebuilds once");
        tick(&mut mods);
        assert_eq!(calls.get(), 1, "a suspended mod's update does not run");
        mods.install_from(Some("pwc.dev-toolkit"), Box::new(Stub::new("late")));
        assert!(!mods.is_active(2), "a later mod of a suspended package starts suspended");
        mods.resume_packages();
        assert!(mods.is_active(0) && mods.is_active(2));
        assert!(mods.suspended().is_empty());
        tick(&mut mods);
        assert_eq!(calls.get(), 2);
    }

    /// Bench pins stay through a server's session suspension and its end.
    #[test]
    fn pinned_packages_stay_suspended_through_a_session() {
        let mut mods = crate::modding::testing::standard();
        mods.pin_suspended(&["pwc.visuals".to_string()]);
        assert_eq!(mods.visual_mask(), VisualMask::NONE);
        suspend(&mut mods, &["pwc.hotbar", "pwc.visuals"]);
        assert_eq!(mods.suspended(), ["pwc.visuals", "pwc.hotbar"], "pinned first, each once");
        mods.resume_packages();
        assert_eq!(mods.suspended(), ["pwc.visuals"]);
        assert_eq!(mods.visual_mask(), VisualMask::NONE);
        assert_eq!(mods.visual_packages(), ["pwc.visuals"], "the package behind every visual group, once");
    }

    fn register_tools(r: &mut ModRegistrar) {
        r.add(Stub::new("tools"));
    }

    fn register_hud(r: &mut ModRegistrar) {
        r.add(Stub::new("hud"));
    }

    fn register_nothing(_: &mut ModRegistrar) {}

    #[test]
    fn hello_reports_every_mod_package_that_is_not_suspended() {
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
        let report = |mods: &Mods| mods.active_package_reports(build.packages());
        let pair = |id: &str| (id.to_string(), "1.0.0".to_string());
        let all = vec![pair("test.tools"), pair("test.hud"), pair("test.empty")];
        assert_eq!(report(&mods), all, "libraries and bundles are never reported; a mod package always is");
        suspend(&mut mods, &["test.tools"]);
        assert_eq!(report(&mods), all[1..], "a suspended package is not reported");
        mods.resume_packages();
        assert_eq!(report(&mods), all);
    }

    /// A mod written against the context-free hook only.
    struct Legacy;

    impl Mod for Legacy {
        fn id(&self) -> &'static str {
            "legacy"
        }
        fn command(&mut self, cmd: &str, args: &[&str]) -> Option<Vec<Line>> {
            (cmd == "old").then(|| vec![Line::of(crate::ui::Role::Dim, args.join(" "))])
        }
    }

    const A: &[Command] = &[Command { name: "tp", args: "<x y z>", help: "teleport" }];
    const B: &[Command] = &[
        Command { name: "tp", args: "", help: "shadowed" },
        Command { name: "time", args: "", help: "clock" },
    ];

    #[test]
    fn the_first_active_mod_that_knows_a_command_runs_it() {
        let mut mods = Mods::empty();
        mods.install(Box::new(Stub::new("a").commands(A)));
        mods.install(Box::new(Stub::new("b").commands(B)));
        mods.install(Box::new(Legacy));
        mods.install_from(Some("test.off"), Box::new(Stub::new("off").commands(&[Command { name: "hidden", args: "", help: "" }])));
        suspend(&mut mods, &["test.off"]);
        let names: Vec<&str> = mods.commands().map(|c| c.name).collect();
        assert_eq!(names, ["tp", "time"], "active mods only, in install order, a shadowed name once");

        let mut world = World::new(1);
        let mut player = Player::new(glam::DVec3::ZERO);
        let (mut settings, mut sky) = (Settings::default(), crate::sky::Sky::new());
        let mut ctx = CommandContext::new(&mut player, &mut world, &mut settings, &mut sky);
        let first = |out: Option<Vec<Line>>| out.map(|lines| lines[0].text().to_string());
        assert_eq!(first(mods.run_command(&mut ctx, "tp", &[])).as_deref(), Some("a"));
        assert_eq!(first(mods.run_command(&mut ctx, "time", &[])).as_deref(), Some("b"));
        assert_eq!(first(mods.run_command(&mut ctx, "old", &["still", "works"])).as_deref(), Some("still works"));
        assert!(mods.run_command(&mut ctx, "hidden", &[]).is_none(), "a suspended mod runs nothing");
        assert_eq!(ctx.player.position.y, 2.0, "the handlers reached the player through the context");
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

    /// Every host change a mods screen can show moves the revision; reading does not.
    #[test]
    fn revision_moves_with_what_a_mods_screen_lists() {
        let mut mods = crate::modding::testing::standard();
        let mut last = mods.revision();
        let _ = (mods.visual_mask(), mods.suspended(), mods.worldgen_config());
        assert_eq!(mods.revision(), last, "reads");
        let mut moved = |mods: &Mods, what: &str| {
            assert_ne!(mods.revision(), last, "{what}");
            last = mods.revision();
        };
        suspend(&mut mods, &["pwc.visuals"]);
        moved(&mods, "a suspension");
        mods.resume_packages();
        moved(&mods, "a resume");
        mods.install(Box::new(Stub::new("extra")));
        moved(&mods, "an install");
    }
}
