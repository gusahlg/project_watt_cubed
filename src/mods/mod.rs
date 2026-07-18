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
pub mod inventory;
pub mod menu_default;

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use voxel_engine::Engine;

use crate::block::ElementId;
use crate::menu::theme::MenuTheme;
use crate::player::Player;
use crate::ui::HudElement;
use crate::world::World;

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
pub trait Mod {
    /// Short, stable name shown in the mod menu and used as a save key.
    fn name(&self) -> &str;

    /// One-line description for the mod menu.
    fn description(&self) -> &str {
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
    fn update(&mut self, eng: &Engine, ctx: &mut ModContext) {
        let _ = (eng, ctx);
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

    /// This mod's HUD contribution while enabled, as data — a list of
    /// [`HudElement`]s the core renders over the world and under the console. A
    /// mod describes *what* to show and never draws, so panel chrome and layout
    /// live in one place ([`crate::ui::render_hud`]). `world` gives read access to
    /// the registry so names resolve at build time rather than being cached.
    fn hud(&self, world: &World, screen: (i32, i32)) -> Vec<HudElement> {
        let _ = (world, screen);
        Vec::new()
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

    /// Serialise persistent state to a single line for the save file, or `None` if
    /// the mod has nothing to persist. `world` resolves ids to portable names.
    fn save_state(&self, world: &World) -> Option<String> {
        let _ = world;
        None
    }

    /// Restore state produced by [`save_state`](Self::save_state). `world` is
    /// mutable because restoring may need to re-register blocks (crafted blocks
    /// are saved by name and re-crafted into the palette on load).
    fn load_state(&mut self, data: &str, world: &mut World) {
        let _ = (data, world);
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

impl Mods {
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
    pub fn update(&mut self, eng: &Engine, ctx: &mut ModContext) {
        for entry in &mut self.entries {
            if entry.enabled {
                entry.module.update(eng, ctx);
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

    /// Collect every enabled mod's HUD contribution, in install order (so a
    /// later mod draws over an earlier one).
    pub fn hud(&self, world: &World, screen: (i32, i32)) -> Vec<HudElement> {
        self.entries
            .iter()
            .filter(|e| e.enabled)
            .flat_map(|e| e.module.hud(world, screen))
            .collect()
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

    /// The description of the mod at `index`.
    pub fn description(&self, index: usize) -> &str {
        self.entries[index].module.description()
    }

    /// Whether the mod at `index` is enabled.
    pub fn is_enabled(&self, index: usize) -> bool {
        self.entries[index].enabled
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

    /// Persistent state of every mod that has any, as `(name, data)` lines.
    pub fn save_states(&self, world: &World) -> Vec<(String, String)> {
        self.entries
            .iter()
            .filter_map(|entry| {
                entry
                    .module
                    .save_state(world)
                    .map(|data| (entry.module.name().to_string(), data))
            })
            .collect()
    }

    /// Restore a mod's state by name (ignoring unknown names from other installs).
    pub fn load_state(&mut self, name: &str, data: &str, world: &mut World) {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.module.name() == name) {
            entry.module.load_state(data, world);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::element::El;

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
}
