//! The mod system: the game's "minimal core, layers on top" made real. Core
//! gameplay owns the world and physics; everything player-facing that isn't
//! essential — the inventory, crafting UIs, HUD widgets — is a [`Mod`] that can be
//! toggled at runtime from the mod menu.
//!
//! **Performance:** mod hooks fire only at frame and event granularity —
//! `update`/`draw` once per frame, `on_block_break` once per broken block. Nothing
//! here is ever called from the voxel hot path (meshing, collision, streaming), and
//! disabled mods are skipped entirely. A mod therefore costs nothing where it would
//! matter and only what it draws where it wouldn't.
pub mod crafting;
pub mod diffusion;
pub mod inventory;
pub mod menu_default;
pub mod visuals;

use std::cell::{Cell, RefCell};
use std::fs;
use std::path::Path;
use std::rc::Rc;

use crate::block::ElementId;
use crate::menu::theme::MenuTheme;
use crate::player::Player;
use crate::render_config::{RenderConfig, VisualGroup};
use crate::settings::Settings;
use crate::ui::HudElement;
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

/// The element counts the player is carrying — the single source of truth shared
/// by the inventory mod (which fills and displays it) and the crafting mod (which
/// spends it). Shared as `Rc<RefCell<ElementStash>>`: the game is single-threaded
/// and mods run strictly one after another, so `Rc`/`RefCell` is exactly enough —
/// no locking, and any accidental nested borrow would panic loudly in development.
pub struct ElementStash {
    /// Per-element counts in first-seen order, so display rows are stable as
    /// counts change (matching the old inventory's grouped view).
    counts: Vec<(ElementId, u32)>,
    /// Cached sum of `counts`; pickups and crafting read it every frame.
    total: u32,
    /// Soft cap on total held elements — the old inventory capacity, upgradeable.
    capacity: usize,
    /// Bumped on every content change; caches (like the inventory's display rows)
    /// rebuild when they see a rev they haven't.
    rev: u64,
}

impl ElementStash {
    pub fn new(capacity: usize) -> Self {
        Self {
            counts: Vec::new(),
            total: 0,
            capacity,
            rev: 0,
        }
    }

    /// Add elements one by one while there is room, exactly like the old
    /// inventory's per-item add: elements past the capacity are dropped, and the
    /// return value is `false` if any were. Bumps `rev` when anything landed.
    pub fn add(&mut self, elements: &[ElementId]) -> bool {
        let mut all = true;
        let mut added = false;
        for &element in elements {
            if self.total as usize >= self.capacity {
                all = false;
                continue;
            }
            match self.counts.iter_mut().find(|(e, _)| *e == element) {
                Some((_, count)) => *count += 1,
                None => self.counts.push((element, 1)),
            }
            self.total += 1;
            added = true;
        }
        if added {
            self.rev += 1;
        }
        all
    }

    /// How many of one element are held.
    pub fn count(&self, element: ElementId) -> u32 {
        self.counts
            .iter()
            .find(|(e, _)| *e == element)
            .map_or(0, |&(_, c)| c)
    }

    /// Spend elements, all or nothing: `elements` is consumed only if every entry
    /// is covered (listing an element twice requires two of it). Emptied elements
    /// drop out of the display order. Bumps `rev` on success.
    pub fn consume(&mut self, elements: &[ElementId]) -> bool {
        // Check the full multiplicity first so a failure changes nothing.
        for &element in elements {
            let needed = elements.iter().filter(|&&e| e == element).count() as u32;
            if self.count(element) < needed {
                return false;
            }
        }
        for &element in elements {
            if let Some((_, count)) = self.counts.iter_mut().find(|(e, _)| *e == element) {
                *count -= 1;
                self.total -= 1;
            }
        }
        self.counts.retain(|&(_, count)| count > 0);
        self.rev += 1;
        true
    }

    /// Take back elements, best-effort: each entry removes one of that element
    /// if any are held. Unlike [`consume`](Self::consume) this is NOT
    /// all-or-nothing — it is the rollback path for a server-rejected break,
    /// where whatever was already spent elsewhere simply can't be revoked.
    pub fn revoke(&mut self, elements: &[ElementId]) {
        let mut removed = false;
        for &element in elements {
            if let Some((_, count)) = self.counts.iter_mut().find(|(e, _)| *e == element) {
                if *count > 0 {
                    *count -= 1;
                    self.total -= 1;
                    removed = true;
                }
            }
        }
        if removed {
            self.counts.retain(|&(_, count)| count > 0);
            self.rev += 1;
        }
    }

    /// Total elements held, across all kinds.
    pub fn total(&self) -> u32 {
        self.total
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The current content revision (see the field docs).
    pub fn rev(&self) -> u64 {
        self.rev
    }

    /// The held `(element, count)` pairs in stable first-seen order.
    pub fn iter(&self) -> impl Iterator<Item = (ElementId, u32)> + '_ {
        self.counts.iter().copied()
    }

    /// Drop everything (used when loading a save into this stash).
    pub fn clear(&mut self) {
        self.counts.clear();
        self.total = 0;
        self.rev += 1;
    }
}

/// Shared visibility state for the inventory/crafting pair. Crafting replaces
/// the compact inventory panel while open, so two independently toggleable mods
/// never draw over one another.
#[derive(Clone, Copy)]
pub(crate) struct ItemUiState {
    pub inventory_visible: bool,
    pub crafting_open: bool,
}

impl Default for ItemUiState {
    fn default() -> Self {
        Self {
            inventory_visible: true,
            crafting_open: false,
        }
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
    pub toggle_crafting: bool,
    pub nav_up: bool,
    pub nav_down: bool,
    pub nav_confirm: bool,
    /// Block placements queued by mods this frame as `(x, y, z, id)`. The game
    /// drains these after `mods.update` and applies each only if the cell is air
    /// and doesn't overlap the player — mods that spend resources on a placement
    /// should pre-check the same so their accounting stays exact.
    pub placements: Vec<(i32, i32, i32, crate::block::registry::BlockId)>,
}

/// A unit of layered-on functionality. Every method has a default, so a mod
/// implements only the hooks it cares about. This is the public surface mod authors
/// write against — kept small on purpose.
///
/// Arbitration when more than one enabled mod implements a hook:
/// - **Fan-out**, install order: `update`, `on_block_break`, `on_break_rejected`,
///   `on_place_rejected`. `hud` uses the same order as z-order (later draws on top).
/// - **First enabled wins**: `menu_theme`, `close_overlay` (first `true`),
///   `worldgen`, `worldgen_config`.
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

    /// Clear per-world state (inventory contents, crafted blocks, open
    /// panels) when entering a different world. Enable/disable choices are
    /// NOT touched — those persist across worlds.
    fn reset(&mut self) {}

    /// Cadence-controlled logic while enabled (the game's `mod_hz`). Runs
    /// after movement, before rendering; edge inputs accumulated between
    /// ticks are replayed in order without loss.
    fn update(&mut self, ctx: &mut ModContext) {
        let _ = ctx;
    }

    /// A block was broken into these elements. The event the inventory mod listens
    /// to; a crafting or logging mod could too.
    fn on_block_break(&mut self, elements: &[ElementId], world: &World) {
        let _ = (elements, world);
    }

    /// The server rejected a break this client predicted (someone else won the
    /// cell): revoke the loot [`on_block_break`](Self::on_block_break) awarded.
    fn on_break_rejected(&mut self, elements: &[ElementId]) {
        let _ = elements;
    }

    /// The server rejected a placement this client predicted: refund whatever
    /// was spent on placing a block of `id`.
    fn on_place_rejected(&mut self, id: crate::block::BlockId, world: &World) {
        let _ = (id, world);
    }

    /// This mod's HUD contribution while enabled, as data — [`HudElement`]s
    /// pushed into a caller-owned buffer the core renders over the world and
    /// under the console. A mod describes *what* to show and never draws, so
    /// panel chrome and layout live in one place ([`crate::ui::render_hud`]).
    /// `world` gives read access to the registry so names resolve at build time
    /// rather than being cached.
    fn hud(&self, world: &World, screen: (i32, i32), out: &mut Vec<HudElement>) {
        let _ = (world, screen, out);
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

    /// Serialise persistent state, or `None` if the mod has nothing to persist.
    /// The `u16` is the payload version; [`Mods::save_states`] encodes it as a
    /// `v<N>;` prefix so the save codec stays a plain string.
    fn save_state(&self, world: &World) -> Option<(u16, String)> {
        let _ = world;
        None
    }

    /// Restore state produced by [`save_state`](Self::save_state). `version` is
    /// 0 when the on-disk string had no prefix (old saves). `world` is mutable
    /// because restoring may need to re-register blocks.
    fn load_state(&mut self, version: u16, data: &str, world: &mut World) {
        let _ = (version, data, world);
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
}

/// One installed mod and whether it is currently active.
struct Entry {
    module: Box<dyn Mod>,
    enabled: bool,
}

/// The set of installed mods and their on/off state. Persists across worlds so the
/// player's mod choices stick; per-world state (like inventory contents) is saved
/// and restored through each mod's `save_state`/`load_state`.
pub struct Mods {
    entries: Vec<Entry>,
}

const CHOICES_PATH: &str = "saves/mods.cfg";

impl Mods {
    /// Groups shown as sections on the mods screen, in this order.
    pub const GROUPS: &[Group] = &[Group {
        id: ESSENTIALS,
        name: "Essentials",
        description: "The built-in mods that make the game playable as shipped: menus, inventory, crafting, the shipped look, and the alternative worldgen. Disable any of them to see the bare core.",
    }];

    /// The default install: the menu mod (look/feel of every out-of-game
    /// screen) first, then the bare-list inventory mod and the crafting mod,
    /// all enabled. Inventory and crafting share one [`ElementStash`] —
    /// inventory fills it from broken blocks, crafting spends it. Menus goes
    /// first so it wins the first-handler dispatch below by default.
    pub fn with_defaults() -> Self {
        let mut mods = Self {
            entries: Vec::new(),
        };
        let stash = Rc::new(RefCell::new(ElementStash::new(inventory::START_CAPACITY)));
        let item_ui = Rc::new(Cell::new(ItemUiState::default()));
        mods.install(Box::new(menu_default::MenuDefaultMod::new()), true);
        mods.install(
            Box::new(inventory::InventoryMod::new(stash.clone(), item_ui.clone())),
            true,
        );
        mods.install(Box::new(crafting::CraftingMod::new(stash, item_ui)), true);
        // Fancy lanes live in mods; disable any of these to get the core look.
        mods.install(Box::new(visuals::AtmosphereMod), true);
        mods.install(Box::new(visuals::PostMod), true);
        mods.install(Box::new(visuals::LightingMod), true);
        // Worldgen swap: off so classic noise remains the default substrate.
        mods.install(Box::new(diffusion::InfiniteDiffusionMod::new()), false);
        mods
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

    /// Run every enabled mod's per-frame logic.
    pub fn update(&mut self, ctx: &mut ModContext) {
        for entry in &mut self.entries {
            if entry.enabled {
                entry.module.update(ctx);
            }
        }
    }

    /// Fan a block-break event out to every enabled mod.
    pub fn on_block_break(&mut self, elements: &[ElementId], world: &World) {
        for entry in &mut self.entries {
            if entry.enabled {
                entry.module.on_block_break(elements, world);
            }
        }
    }

    /// Fan a rejected-break rollback out to every enabled mod.
    pub fn on_break_rejected(&mut self, elements: &[ElementId]) {
        for entry in &mut self.entries {
            if entry.enabled {
                entry.module.on_break_rejected(elements);
            }
        }
    }

    /// Fan a rejected-placement refund out to every enabled mod.
    pub fn on_place_rejected(&mut self, id: crate::block::BlockId, world: &World) {
        for entry in &mut self.entries {
            if entry.enabled {
                entry.module.on_place_rejected(id, world);
            }
        }
    }

    /// Push every enabled mod's HUD contribution into `out`, in install order
    /// (so a later mod draws over an earlier one). The caller owns `out` and
    /// clears it per frame so capacity is retained.
    pub fn hud(&self, world: &World, screen: (i32, i32), out: &mut Vec<HudElement>) {
        for entry in &self.entries {
            if entry.enabled {
                entry.module.hud(world, screen, out);
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

    /// Worldgen used for the next world: diffusion if that mod is on, else classic.
    pub fn worldgen_kind(&self) -> WorldgenKind {
        self.first_worldgen()
            .and_then(|m| m.worldgen())
            .unwrap_or(WorldgenKind::Classic)
    }

    /// Opaque payload of the winning worldgen mod. The kind parses it
    /// (`DiffusionCfg::from_text` for InfiniteDiffusion).
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
    pub fn load_state(&mut self, name: &str, data: &str, world: &mut World) {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.module.id() == name || e.module.name() == name) {
            let (version, payload) = split_mod_version(data);
            entry.module.load_state(version, payload, world);
        }
    }

    /// `id=on|off` lines, plus `id.state=<payload>` for mods that persist knobs.
    pub fn choices_text(&self) -> String {
        let mut text = String::new();
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
    /// are ignored; missing keys keep the current defaults.
    pub fn apply_choices_text(&mut self, text: &str) {
        for line in text.lines() {
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            let key = key.trim();
            let value = value.trim();
            if let Some(id) = key.strip_suffix(".state") {
                self.apply_choice_state(id.trim(), value);
                continue;
            }
            let on = match value {
                "on" => true,
                "off" => false,
                _ => continue,
            };
            self.set_enabled(key, on);
        }
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

    /// Load enable/disable choices and knob payloads from `saves/mods.cfg`.
    /// Missing or unreadable file leaves the current defaults in place.
    pub fn load_choices(&mut self) {
        self.load_choices_from(Path::new(CHOICES_PATH));
    }

    fn load_choices_from(&mut self, path: &Path) {
        if let Ok(text) = fs::read_to_string(path) {
            self.apply_choices_text(&text);
        }
    }

    /// Best-effort write of enable/disable choices and knob payloads. Bench-env
    /// pins are not written from startup; only a later toggle or knob step persists.
    pub fn save_choices(&self) {
        self.save_choices_to(Path::new(CHOICES_PATH));
    }

    fn save_choices_to(&self, path: &Path) {
        if let Some(dir) = path.parent() {
            let _ = fs::create_dir_all(dir);
        }
        let _ = fs::write(path, self.choices_text());
    }
}

/// Pull a leading `v<N>;` version prefix off a saved mod blob. No prefix
/// (or a prefix that isn't a `u16`) is version 0, the pre-versioning format.
fn split_mod_version(data: &str) -> (u16, &str) {
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
    use crate::block::element::El;
    use crate::menu::Menu;
    use crate::world::diffusion::DiffusionCfg;
    use crate::world::World;

    fn payload_cfg(mods: &Mods) -> DiffusionCfg {
        mods.worldgen_config()
            .as_deref()
            .map(DiffusionCfg::from_text)
            .unwrap_or_default()
    }

    #[test]
    fn stash_add_respects_capacity_per_item() {
        let mut stash = ElementStash::new(3);
        assert!(stash.add(&[El::Stone.id(), El::Soil.id()]));
        // Room for one more: the first lands, the second is dropped.
        assert!(!stash.add(&[El::Stone.id(), El::Clay.id()]));
        assert_eq!(stash.total(), 3);
        assert_eq!(stash.count(El::Stone.id()), 2);
        assert_eq!(stash.count(El::Clay.id()), 0);
    }

    #[test]
    fn stash_consume_is_all_or_nothing() {
        let mut stash = ElementStash::new(10);
        stash.add(&[El::Stone.id(), El::Stone.id(), El::Iron.id()]);
        let rev = stash.rev();
        assert!(!stash.consume(&[El::Stone.id(), El::Copper.id()]));
        assert_eq!(stash.rev(), rev, "a failed consume changes nothing");
        assert_eq!(stash.total(), 3);
        assert!(stash.consume(&[El::Stone.id(), El::Iron.id()]));
        assert_eq!(stash.count(El::Stone.id()), 1);
        assert_eq!(stash.count(El::Iron.id()), 0);
        assert!(stash.rev() > rev);
    }

    #[test]
    fn stash_consume_counts_multiplicity() {
        let mut stash = ElementStash::new(10);
        stash.add(&[El::Stone.id()]);
        // Listing Stone twice needs two Stones; only one is held.
        assert!(!stash.consume(&[El::Stone.id(), El::Stone.id()]));
        assert!(stash.consume(&[El::Stone.id()]));
        assert_eq!(stash.total(), 0);
    }

    #[test]
    fn stash_iteration_keeps_first_seen_order() {
        let mut stash = ElementStash::new(10);
        stash.add(&[El::Iron.id(), El::Stone.id(), El::Iron.id()]);
        let order: Vec<_> = stash.iter().collect();
        assert_eq!(order, vec![(El::Iron.id(), 2), (El::Stone.id(), 1)]);
    }

    #[test]
    fn worldgen_kind_skips_non_worldgen_mods() {
        let mods = Mods::with_defaults();
        assert_eq!(mods.worldgen_kind(), WorldgenKind::Classic);
        let mut on = Mods::with_defaults();
        on.set_enabled("diffusion", true);
        assert_eq!(on.worldgen_kind(), WorldgenKind::Diffusion);
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
        assert_eq!(mods.worldgen_kind(), WorldgenKind::Classic);
        mods.set_enabled("InfiniteDiffusion", true);
        assert_eq!(
            mods.worldgen_kind(),
            WorldgenKind::Classic,
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
        mods.on_block_break(&[El::Stone.id(), El::Iron.id()], &world);
        let saved = mods.save_states(&world);
        assert!(
            saved.iter().any(|(k, _)| k == "inventory"),
            "save keys are stable ids, not display names"
        );
        assert!(!saved.iter().any(|(k, _)| k == "Inventory"));
        let data = saved
            .iter()
            .find(|(k, _)| k == "inventory")
            .map(|(_, d)| d.clone())
            .expect("inventory persists");

        let mut by_id = Mods::with_defaults();
        by_id.load_state("inventory", &data, &mut world);
        assert_eq!(
            by_id
                .save_states(&world)
                .iter()
                .find(|(k, _)| k == "inventory")
                .map(|(_, d)| d.as_str()),
            Some(data.as_str())
        );

        let mut by_name = Mods::with_defaults();
        by_name.load_state("Inventory", &data, &mut world);
        assert_eq!(
            by_name
                .save_states(&world)
                .iter()
                .find(|(k, _)| k == "inventory")
                .map(|(_, d)| d.as_str()),
            Some(data.as_str())
        );
    }

    #[test]
    fn mod_state_round_trips_version_prefix() {
        let mut world = World::new(1);
        let mut mods = Mods::with_defaults();
        mods.on_block_break(&[El::Stone.id(), El::Iron.id()], &world);
        mods.load_state("Crafting", "*IronVein=2", &mut world);
        let saved = mods.save_states(&world);
        let inv = saved
            .iter()
            .find(|(k, _)| k == "inventory")
            .map(|(_, d)| d.as_str())
            .expect("inventory");
        assert!(inv.starts_with("v1;"), "new writes encode a version prefix: {inv}");
        let craft = saved
            .iter()
            .find(|(k, _)| k == "crafting")
            .map(|(_, d)| d.as_str())
            .expect("crafting");
        assert_eq!(craft, "v1;*Stone+Iron=2");

        let mut fresh = Mods::with_defaults();
        for (k, v) in &saved {
            fresh.load_state(k, v, &mut world);
        }
        assert_eq!(fresh.save_states(&world), saved);

        // Unprefixed display-name key is version 0 and still migrates veins.
        let mut legacy = Mods::with_defaults();
        legacy.load_state("Crafting", "*IronVein=1", &mut world);
        let craft = legacy
            .save_states(&world)
            .into_iter()
            .find(|(k, _)| k == "crafting")
            .map(|(_, d)| d)
            .expect("crafting");
        assert_eq!(craft, "v1;*Stone+Iron=1");
    }

    fn index_of(mods: &Mods, id: &str) -> usize {
        (0..mods.len())
            .find(|&i| mods.id(i) == id)
            .unwrap_or_else(|| panic!("missing mod {id}"))
    }

    fn temp_choices_path() -> std::path::PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "watt-mods-{}-{}.cfg",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ))
    }

    #[test]
    fn choices_text_round_trips_and_ignores_junk() {
        let mut mods = Mods::with_defaults();
        let defaults = mods.choices_text();
        assert!(defaults.contains("menus=on"));
        assert!(defaults.contains("inventory=on"));
        assert!(defaults.contains("crafting=on"));
        assert!(defaults.contains("atmosphere=on"));
        assert!(defaults.contains("post=on"));
        assert!(defaults.contains("lighting=on"));
        assert!(defaults.contains("diffusion=off"));
        assert!(defaults.contains("diffusion.state=tile=32,stride=16,phases=2,relief=1.00"));

        mods.set_enabled("lighting", false);
        mods.set_enabled("diffusion", true);
        let i = index_of(&mods, "diffusion");
        mods.step_knob(i, 0, 1);
        let cfg = payload_cfg(&mods);
        assert_ne!(cfg.tile, DiffusionCfg::default().tile);
        let text = mods.choices_text();
        assert!(text.contains("lighting=off"));
        assert!(text.contains("diffusion=on"));
        assert!(text.contains(&format!(
            "diffusion.state=tile={},stride={},phases={},relief={:.2}",
            cfg.tile, cfg.stride, cfg.phases, cfg.relief
        )));

        let mut fresh = Mods::with_defaults();
        fresh.apply_choices_text(
            "lighting=off\nnot-a-mod=on\nmenus=nope\n\ninventory=off\ndiffusion=on\ndiffusion.state=tile=64,stride=16,phases=2,relief=1.00\nunknown.state=tile=16\n",
        );
        let restored = fresh.choices_text();
        assert!(restored.contains("lighting=off"));
        assert!(restored.contains("inventory=off"));
        assert!(restored.contains("diffusion=on"));
        assert!(
            restored.contains("menus=on"),
            "malformed value must not change the default"
        );
        assert!(restored.contains("crafting=on"));
        assert_eq!(payload_cfg(&fresh).tile, 64);
    }

    #[test]
    fn choices_file_round_trips_toggles_and_knobs() {
        let path = temp_choices_path();
        let mut mods = Mods::with_defaults();
        let i = index_of(&mods, "diffusion");
        mods.set_enabled("lighting", false);
        mods.set_enabled("diffusion", true);
        mods.step_knob(i, 0, 1);
        mods.step_knob(i, 1, 1);
        let cfg = payload_cfg(&mods);
        mods.save_choices_to(&path);

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
        assert_eq!(mods.worldgen_kind(), WorldgenKind::Classic);
        let mask = mods.visual_mask();
        assert!(!mask.atmosphere && !mask.post && !mask.lighting);
    }

    #[test]
    fn apply_bench_env_does_not_write_choices() {
        let path = temp_choices_path();
        let mut mods = Mods::with_defaults();
        mods.save_choices_to(&path);
        let on_disk = fs::read_to_string(&path).unwrap();
        assert!(on_disk.contains("diffusion=off"));
        mods.apply_bench_env(Some(true), Some(true));
        assert_eq!(mods.worldgen_kind(), WorldgenKind::Diffusion);
        assert_eq!(fs::read_to_string(&path).unwrap(), on_disk);
        assert!(mods.choices_text().contains("diffusion=on"));
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
            "The built-in mods that make the game playable as shipped: menus, inventory, crafting, the shipped look, and the alternative worldgen. Disable any of them to see the bare core."
        );
        let members: Vec<&str> = (0..mods.len())
            .filter(|&i| mods.group(i) == ESSENTIALS)
            .map(|i| mods.id(i))
            .collect();
        assert_eq!(
            members,
            [
                "menus",
                "inventory",
                "crafting",
                "atmosphere",
                "post",
                "lighting",
                "diffusion"
            ]
        );
        assert_eq!(members.len(), mods.len(), "no ungrouped built-ins");
    }

    #[test]
    fn group_toggle_persists_each_member_line() {
        let path = temp_choices_path();
        let mut mods = Mods::with_defaults();
        mods.set_group_enabled(ESSENTIALS, false);
        let text = mods.choices_text();
        for id in [
            "menus",
            "inventory",
            "crafting",
            "atmosphere",
            "post",
            "lighting",
            "diffusion",
        ] {
            assert!(
                text.contains(&format!("{id}=off")),
                "{id} should be off in:\n{text}"
            );
        }
        assert!(
            !text.lines().any(|l| l.starts_with("essentials=")),
            "group toggle must not write a group-level key"
        );
        mods.save_choices_to(&path);

        let mut fresh = Mods::with_defaults();
        fresh.load_choices_from(&path);
        let _ = fs::remove_file(&path);
        for i in 0..fresh.len() {
            assert!(!fresh.is_enabled(i), "{} still on", fresh.id(i));
        }

        fresh.set_group_enabled(ESSENTIALS, true);
        let on_text = fresh.choices_text();
        for id in [
            "menus",
            "inventory",
            "crafting",
            "atmosphere",
            "post",
            "lighting",
            "diffusion",
        ] {
            assert!(
                on_text.contains(&format!("{id}=on")),
                "{id} should be on in:\n{on_text}"
            );
        }
    }

    #[test]
    fn worldgen_config_is_the_winning_kind_payload() {
        let off = Mods::with_defaults();
        assert_eq!(off.worldgen_kind(), WorldgenKind::Classic);
        assert_eq!(off.worldgen_config(), None);
        let mut on = Mods::with_defaults();
        on.set_enabled("diffusion", true);
        let text = on.worldgen_config().expect("payload");
        assert_eq!(DiffusionCfg::from_text(&text), DiffusionCfg::default());
        on.step_knob(index_of(&on, "diffusion"), 0, 1);
        let cfg = DiffusionCfg::from_text(&on.worldgen_config().unwrap());
        assert_ne!(cfg.tile, DiffusionCfg::default().tile);
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
}
