//! The mod system: the game's "minimal core, layers on top" made real. Core
//! gameplay owns the world, the law and physics; everything player-facing that
//! isn't essential — the inventory, the hotbar, block looks and names, HUD
//! widgets — is a [`Mod`] that can be toggled at runtime from the mod menu.
//!
//! **Performance:** mod hooks fire only at frame and event granularity —
//! `update`/`draw` once per frame, `on_block_break` once per broken block. Nothing
//! here is ever called from the voxel hot path (meshing, collision, streaming), and
//! disabled mods are skipped entirely. A mod therefore costs nothing where it would
//! matter and only what it draws where it wouldn't.
pub mod diffusion;
pub mod hotbar;
pub mod inventory;
pub mod menu_default;
pub mod naming;
pub mod start_screen;
pub mod textures;
pub mod visuals;

use std::cell::Cell;
use std::fs;
use std::io;
use std::path::Path;
use std::rc::Rc;

use crate::block::appearance::{BlockAppearance, FLAT};
use crate::block::naming::MaterialNamer;
use crate::block::BlockId;
use crate::menu::start::{StartFacts, StartScreen};
use crate::menu::theme::MenuTheme;
use crate::player::Player;
use crate::render_config::{RenderConfig, VisualGroup};
use crate::settings::Settings;
use crate::ui::{HudElement, Line};
use crate::world::generation::WorldgenKind;
use crate::world::World;

/// Group id of the shipped built-in mods. Display name lives on [`Mods::GROUPS`].
pub const ESSENTIALS: &str = "essentials";

/// Named group of related mods. The id is the stable key; the display name
/// can change here without touching every member.
#[derive(Clone, Copy, Debug)]
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
        Self {
            atmosphere: true,
            post: true,
            lighting: true,
        }
    }
}

impl VisualMask {
    pub fn apply(self, mut cfg: RenderConfig) -> RenderConfig {
        if !self.atmosphere {
            cfg.strip_group(VisualGroup::Atmosphere);
        }
        if !self.post {
            cfg.strip_group(VisualGroup::Post);
        }
        if !self.lighting {
            cfg.strip_group(VisualGroup::Lighting);
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
        let on = match group {
            VisualGroup::Atmosphere => self.atmosphere,
            VisualGroup::Post => self.post,
            VisualGroup::Lighting => self.lighting,
        };
        if on { None } else { Some(group.mod_name()) }
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

/// Shared state of the inventory and the hotbar: whether the inventory panel is open (it then
/// owns the number keys and the wheel, to equip into hotbar slots).
#[derive(Clone, Copy)]
pub(crate) struct ItemUiState {
    pub inventory_visible: bool,
}

impl Default for ItemUiState {
    fn default() -> Self {
        Self { inventory_visible: false }
    }
}

/// The coarse, per-frame state a mod may read and mutate. Deliberately holds only
/// whole-game handles (never a voxel), so a mod can't reach into the hot path.
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
    pub toggle_inventory: bool,
    pub nav_up: bool,
    pub nav_down: bool,
    pub nav_left: bool,
    pub nav_right: bool,
    pub nav_tab: bool,
    pub nav_confirm: bool,
    /// A hotbar key this frame: `Some(0)` = the bare hand (key 0), `Some(1..=9)` = a slot.
    pub hotbar_key: Option<u8>,
    /// Wheel steps this frame: +1 next slot, −1 previous.
    pub hotbar_cycle: i8,
    /// True when a server owns evaluation (reactions and tool use run there).
    pub networked: bool,
    /// Block placements queued by mods this frame as `(x, y, z, id)`. The game
    /// drains these after `mods.update` and applies each only if the cell is air
    /// and doesn't overlap the player — mods that spend resources on a placement
    /// should pre-check the same so their accounting stays exact.
    pub placements: Vec<(i32, i32, i32, crate::block::registry::BlockId)>,
}

impl ModContext<'_> {
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
/// - **Fan-out**, install order: `update`, `on_block_break`, `on_break_rejected`,
///   `on_place_rejected`, `on_tool_changed`. `hud` uses the same order as z-order (later
///   draws on top).
/// - **First enabled wins**: `menu_theme`, `start_screen`, `close_overlay` (first `true`),
///   `worldgen`, `worldgen_config`, `appearance`, `namer`, `held`.
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

    /// Group id from [`Mods::GROUPS`], or `""` if ungrouped.
    fn group(&self) -> &'static str {
        ""
    }

    /// Called when the mod is switched on (including at load if enabled).
    fn on_enable(&mut self) {}
    /// Called when the mod is switched off.
    fn on_disable(&mut self) {}

    /// Clear per-world state (crafted blocks, open panels) when entering a
    /// different world. Enable/disable choices are NOT touched — those persist
    /// across worlds. The element stash lives on the player, not here.
    fn reset(&mut self) {}

    /// Cadence-controlled logic while enabled (the game's `mod_hz`). Runs
    /// after movement, before rendering; edge inputs accumulated between
    /// ticks are replayed in order without loss.
    fn update(&mut self, ctx: &mut ModContext) {
        let _ = ctx;
    }

    /// A block was broken into this configuration. The core has already deposited
    /// it into the player stash; this is a notification. `overflow` is true
    /// when the stash dropped it (capacity).
    fn on_block_break(&mut self, id: BlockId, world: &World, overflow: bool) {
        let _ = (id, world, overflow);
    }

    /// The server rejected a break this client predicted (someone else won the
    /// cell). The core has already revoked the loot from the stash; this is a
    /// notification.
    fn on_break_rejected(&mut self, id: BlockId) {
        let _ = id;
    }

    /// The server rejected a placement this client predicted: refund whatever
    /// was spent on placing a block of `id`.
    fn on_place_rejected(&mut self, id: crate::block::BlockId, world: &World) {
        let _ = (id, world);
    }

    /// The configuration the player holds as a tool, if any (first enabled mod that answers
    /// wins; `None` everywhere = the bare hand, which breaks blocks). The core runs the law
    /// between this configuration and the targeted block on a left click.
    fn held(&self, player: &Player) -> Option<BlockId> {
        let _ = player;
        None
    }

    /// A held unit changed configuration through a tool reaction: one unit of `old` became one
    /// unit of `new` (`AIR` when the tool was used up). The core already updated the stash; a
    /// hotbar follows the unit here.
    fn on_tool_changed(&mut self, old: BlockId, new: BlockId) {
        let _ = (old, new);
    }

    /// Optional material namer. First enabled mod that returns `Some` names every
    /// configuration; without one the core describes materials by their readings.
    fn namer(&self) -> Option<&dyn MaterialNamer> {
        None
    }

    /// First enabled mod that returns `Some` handles the console command.
    fn command(&mut self, cmd: &str, args: &[&str]) -> Option<Vec<Line>> {
        let _ = (cmd, args);
        None
    }

    /// This mod's HUD contribution while enabled, as data — [`HudElement`]s
    /// pushed into a caller-owned buffer the core renders over the world and
    /// under the console. A mod describes *what* to show and never draws, so
    /// panel chrome and layout live in one place ([`crate::ui::render_hud`]).
    /// `world` gives read access to the registry so names resolve at build time
    /// rather than being cached. `player` is the one path to the core stash.
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

    /// Which fancy render group this mod owns, if any.
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
}

/// The set of installed mods and their on/off state. Enable/disable choices persist
/// in `mods.cfg`; per-world state is saved through each mod's `save_state`/`load_state`.
pub struct Mods {
    entries: Vec<Entry>,
}

impl Mods {
    /// Groups shown as sections on the mods screen, in this order.
    pub const GROUPS: &[Group] = &[Group {
        id: ESSENTIALS,
        name: "Essentials",
        description: "Menus, inventory, hotbar, looks, names and worldgen.",
    }];

    /// The default install, all enabled: the menu mod (look/feel of every out-of-game screen)
    /// first so it wins first-handler dispatch, the start-screen mod, the inventory and the
    /// hotbar (sharing [`ItemUiState`]), the fancy visual lanes, neural textures, material names,
    /// and InfiniteDiffusion worldgen.
    pub fn with_defaults() -> Self {
        let mut mods = Self {
            entries: Vec::new(),
        };
        let item_ui = Rc::new(Cell::new(ItemUiState::default()));
        let bar = Rc::new(Cell::new(hotbar::HotbarState::default()));
        mods.install(Box::new(menu_default::MenuDefaultMod::new()), true);
        mods.install(Box::new(start_screen::StartScreenMod::new()), true);
        mods.install(Box::new(inventory::InventoryMod::new(item_ui.clone(), bar.clone())), true);
        mods.install(Box::new(hotbar::HotbarMod::new(item_ui, bar)), true);
        // Fancy lanes live in mods; disable any of these to get the core look.
        mods.install(Box::new(visuals::AtmosphereMod), true);
        mods.install(Box::new(visuals::PostMod), true);
        mods.install(Box::new(visuals::LightingMod), true);
        mods.install(Box::new(textures::neural::NeuralTexturesMod::new()), true);
        mods.install(Box::new(naming::NamingMod::new()), true);
        mods.install(Box::new(diffusion::InfiniteDiffusionMod::new()), true);
        mods
    }

    /// No mods installed. Appearance is [`FLAT`].
    pub fn empty() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Install a mod, running its enable hook if it starts on.
    pub fn install(&mut self, module: Box<dyn Mod>, enabled: bool) {
        let mut entry = Entry { module, enabled };
        if enabled {
            entry.module.on_enable();
        }
        self.entries.push(entry);
    }

    /// Reset every mod's per-world state (entering a new/loaded/networked
    /// world) while keeping the player's enable/disable choices.
    pub fn reset_state(&mut self) {
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

    /// What the player holds as a tool: the first enabled mod that answers.
    pub fn held(&self, player: &Player) -> Option<BlockId> {
        self.entries.iter().filter(|e| e.enabled).find_map(|e| e.module.held(player))
    }

    /// The first enabled namer, if any.
    pub fn namer(&self) -> Option<&dyn MaterialNamer> {
        self.entries.iter().filter(|e| e.enabled).find_map(|e| e.module.namer())
    }

    /// First enabled mod that handles `cmd` wins.
    pub fn command(&mut self, cmd: &str, args: &[&str]) -> Option<Vec<Line>> {
        for entry in &mut self.entries {
            if !entry.enabled {
                continue;
            }
            if let Some(out) = entry.module.command(cmd, args) {
                return Some(out);
            }
        }
        None
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
    pub fn toggle(&mut self, index: usize) {
        let entry = &mut self.entries[index];
        entry.enabled = !entry.enabled;
        if entry.enabled {
            entry.module.on_enable();
        } else {
            entry.module.on_disable();
        }
    }

    pub fn knobs(&self, index: usize) -> Vec<Knob> {
        self.entries[index].module.knobs()
    }

    pub fn step_knob(&mut self, index: usize, knob: usize, delta: i32) {
        self.entries[index].module.step_knob(knob, delta);
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

    pub fn visual_mask(&self) -> VisualMask {
        let mut mask = VisualMask {
            atmosphere: false,
            post: false,
            lighting: false,
        };
        for entry in self.entries.iter().filter(|e| e.enabled) {
            match entry.module.visual_group() {
                Some(VisualGroup::Atmosphere) => mask.atmosphere = true,
                Some(VisualGroup::Post) => mask.post = true,
                Some(VisualGroup::Lighting) => mask.lighting = true,
                None => {}
            }
        }
        mask
    }

    /// Enable or disable a mod by [`Mod::id`] (case-insensitive). No-op if
    /// already in that state or the id is unknown.
    pub fn set_enabled(&mut self, id: &str, on: bool) {
        if let Some(i) = self
            .entries
            .iter()
            .position(|e| e.module.id().eq_ignore_ascii_case(id))
            && self.entries[i].enabled != on
        {
            self.toggle(i);
        }
    }

    /// Enable or disable every installed member of `group_id`. Persists as
    /// each member's `id=on|off` line — there is no group-level key.
    pub fn set_group_enabled(&mut self, group_id: &str, on: bool) {
        for i in 0..self.entries.len() {
            if self.entries[i].module.group() == group_id && self.entries[i].enabled != on {
                self.toggle(i);
            }
        }
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
            text.push_str(entry.module.id());
            text.push('=');
            text.push_str(if entry.enabled { "on" } else { "off" });
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
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|e| e.module.id().eq_ignore_ascii_case(id))
        {
            entry.module.load_choice_state(data);
        }
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

/// Debounces `mods.cfg` writes so a held Left/Right does not rewrite at key-repeat rate.
pub struct ChoicesFlush {
    last_ms: Option<u64>,
}

impl ChoicesFlush {
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

impl Default for ChoicesFlush {
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
    use crate::menu::Menu;
    use crate::world::terrain::TerrainCfg;
    use crate::world::World;

    fn payload_cfg(mods: &Mods) -> TerrainCfg {
        mods.worldgen_config()
            .as_deref()
            .map(TerrainCfg::from_text)
            .unwrap_or_default()
    }

    /// Every built-in, in install order.
    const BUILTINS: [&str; 10] = [
        "menus",
        "start",
        "inventory",
        "hotbar",
        "atmosphere",
        "post",
        "lighting",
        "neural_textures",
        "material_names",
        "diffusion",
    ];

    #[test]
    fn worldgen_kind_skips_non_worldgen_mods() {
        let mods = Mods::with_defaults();
        assert_eq!(mods.worldgen_kind(), WorldgenKind::Diffusion, "InfiniteDiffusion is on by default");
        let mut off = Mods::with_defaults();
        off.set_enabled("diffusion", false);
        assert_eq!(off.worldgen_kind(), WorldgenKind::Flat, "the core fallback is the flat world");
    }

    #[test]
    fn effective_render_strips_disabled_visual_groups() {
        let mut mods = Mods::with_defaults();
        mods.set_enabled("Atmosphere", false);
        mods.set_enabled("Post", false);
        mods.set_enabled("Lighting", false);
        let settings = Settings::default();
        let stripped = mods.effective_render(&settings);
        assert!(!stripped.clouds);
        assert!(!stripped.bloom);
        assert!(!stripped.shadows);
        assert!(stripped.sunlight);
        let full = Mods::with_defaults().effective_render(&settings);
        assert_eq!(full.clouds, settings.clouds);
        assert_eq!(full.bloom, settings.bloom);
        assert_eq!(full.shadows, settings.shadows);
        let via_mask = mods.visual_mask().effective_render(&settings);
        assert!(!via_mask.clouds && !via_mask.bloom && !via_mask.shadows);
        assert_eq!(via_mask.sunlight, stripped.sunlight);
    }

    #[test]
    fn annotate_setting_names_the_mod_that_forced_the_lane_off() {
        let mut mods = Mods::with_defaults();
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
        let mut mods = Mods::with_defaults();
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
        let mut mods = Mods::with_defaults();
        let rock = world.registry().id_by_label("rock").unwrap();
        let spec = world.registry().spec(rock);
        mods.load_state("Hotbar", &format!("sel=2;2={spec}"), &mut world);
        let saved = mods.save_states(&world);
        assert!(saved.iter().any(|(k, _)| k == "hotbar"), "save keys are stable ids, not display names");
        assert!(!saved.iter().any(|(k, _)| k == "Hotbar"));
        let data = saved.iter().find(|(k, _)| k == "hotbar").map(|(_, d)| d.clone()).expect("hotbar persists");
        for key in ["hotbar", "Hotbar"] {
            let mut fresh = Mods::with_defaults();
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
        let mut mods = Mods::with_defaults();
        let rock = world.registry().id_by_label("rock").unwrap();
        let spec = world.registry().spec(rock);
        mods.load_state("hotbar", &format!("v1;sel=1;1={spec}"), &mut world);
        let saved = mods.save_states(&world);
        assert!(saved.iter().all(|(k, _)| k != "inventory"), "the stash is core state, not an inventory save line");
        let bar = saved.iter().find(|(k, _)| k == "hotbar").map(|(_, d)| d.as_str()).expect("hotbar");
        assert_eq!(bar, format!("v1;sel=1;1={spec}"));
        let mut fresh = Mods::with_defaults();
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
        let mut mods = Mods::with_defaults();
        let defaults = mods.choices_text();
        for id in BUILTINS {
            assert!(defaults.contains(&format!("{id}=on")), "{id} is on by default:\n{defaults}");
        }
        assert!(defaults.contains("neural_textures.state=detail=1.0,contrast=1.0"));
        assert!(defaults.contains("material_names.state=style=mineral"));
        assert!(defaults.contains("diffusion.state=relief=100,caves=100,mines=100,space=100"));

        mods.set_enabled("lighting", false);
        let i = index_of(&mods, "diffusion");
        mods.step_knob(i, 0, 1);
        let cfg = payload_cfg(&mods);
        assert_eq!(cfg.relief, 125);
        let text = mods.choices_text();
        assert!(text.contains("lighting=off"));
        assert!(text.contains(&format!("diffusion.state={}", cfg.to_text())));

        let mut fresh = Mods::with_defaults();
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
        let mut mods = Mods::with_defaults();
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
        let mut mods = Mods::with_defaults();
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

        let mut fresh = Mods::with_defaults();
        fresh.load_choices_from(&path);
        let _ = fs::remove_file(&path);
        assert!(!fresh.is_enabled(index_of(&fresh, "lighting")));
        assert!(fresh.is_enabled(index_of(&fresh, "diffusion")));
        assert_eq!(payload_cfg(&fresh), cfg);
    }

    #[test]
    fn unknown_choice_ids_are_ignored() {
        let mut mods = Mods::with_defaults();
        let before = mods.choices_text();
        mods.apply_choices_text("not-a-mod=on\nunknown.state=tile=64\nmenus=nope\n");
        assert_eq!(mods.choices_text(), before);
    }

    #[test]
    fn corrupt_choices_file_falls_back_to_defaults() {
        let defaults = Mods::with_defaults().choices_text();

        let bad_utf8 = temp_choices_path();
        fs::write(&bad_utf8, [0xff, 0xfe, 0x00, 0x01]).unwrap();
        let mut mods = Mods::with_defaults();
        mods.load_choices_from(&bad_utf8);
        let _ = fs::remove_file(&bad_utf8);
        assert_eq!(mods.choices_text(), defaults);

        let garbage = temp_choices_path();
        fs::write(&garbage, "{{{{ not a config\n!!!\n").unwrap();
        let mut mods = Mods::with_defaults();
        mods.load_choices_from(&garbage);
        let _ = fs::remove_file(&garbage);
        assert_eq!(mods.choices_text(), defaults);

        let mut mods = Mods::with_defaults();
        mods.load_choices_from(Path::new("/tmp/watt-mods-does-not-exist.cfg"));
        assert_eq!(mods.choices_text(), defaults);
    }

    #[test]
    fn apply_bench_env_pins_worldgen_and_visuals() {
        let mut mods = Mods::with_defaults();
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
        let mut mods = Mods::with_defaults();
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
    fn essentials_lists_every_builtin_in_install_order() {
        let mods = Mods::with_defaults();
        assert_eq!(Mods::GROUPS.len(), 1);
        let g = &Mods::GROUPS[0];
        assert_eq!(g.id, ESSENTIALS);
        assert_eq!(g.name, "Essentials");
        assert_eq!(
            g.description,
            "Menus, inventory, hotbar, looks, names and worldgen."
        );
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
        assert!((0..mods.len()).all(|i| !mods.group(i).is_empty()), "every built-in is an Essential");
    }

    #[test]
    fn group_toggle_persists_each_member_line() {
        let path = temp_choices_path();
        let mut mods = Mods::with_defaults();
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

        let mut fresh = Mods::with_defaults();
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
    fn worldgen_config_is_the_winning_kind_payload() {
        let mut off = Mods::with_defaults();
        off.set_enabled("diffusion", false);
        assert_eq!(off.worldgen_kind(), WorldgenKind::Flat);
        assert_eq!(off.worldgen_config(), None);
        let mut on = Mods::with_defaults();
        let text = on.worldgen_config().expect("payload");
        assert_eq!(TerrainCfg::from_text(&text), TerrainCfg::default());
        on.step_knob(index_of(&on, "diffusion"), 3, 1);
        let cfg = TerrainCfg::from_text(&on.worldgen_config().unwrap());
        assert_eq!(cfg.space, 125);
    }

    #[test]
    fn fallback_theme_with_essentials_disabled() {
        let mut mods = Mods::with_defaults();
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

    #[test]
    fn choices_flush_waits_250ms_then_resets_on_mark() {
        let mut flush = ChoicesFlush::new();
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
        assert!(!flush.poll(10 + ChoicesFlush::IDLE_MS));
    }

    fn enabled(mods: &Mods, name: &str) -> bool {
        (0..mods.len())
            .find(|&i| mods.name(i) == name)
            .map(|i| mods.is_enabled(i))
            .expect("installed mod")
    }

    #[test]
    fn choices_round_trip_through_the_config_root_and_ignore_unknown() {
        let mut mods = Mods::with_defaults();
        mods.set_enabled("diffusion", true);
        mods.set_enabled("atmosphere", false);
        mods.save_choices().unwrap();
        let path = crate::paths::Paths::get().mods_file();
        assert!(path.exists());
        assert!(path.starts_with(&crate::paths::Paths::get().config));
        assert_ne!(path, std::path::PathBuf::from("saves/mods.cfg"));

        let mut loaded = Mods::with_defaults();
        loaded.load_choices();
        assert!(enabled(&loaded, "InfiniteDiffusion"));
        assert!(!enabled(&loaded, "Atmosphere"));
        assert!(enabled(&loaded, "Inventory"));

        fs::write(&path, "no-such=on\ninventory=off\nnot-a-pair\natmosphere=true\n").unwrap();
        let mut parsed = Mods::with_defaults();
        parsed.load_choices();
        assert!(!enabled(&parsed, "Inventory"));
        assert!(enabled(&parsed, "Atmosphere"));
        assert!(enabled(&parsed, "InfiniteDiffusion"), "an unmentioned mod keeps its default (on)");
        let _ = fs::remove_file(path);
    }
}
