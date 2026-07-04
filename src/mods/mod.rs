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

use std::cell::RefCell;
use std::rc::Rc;

use voxel_engine::{Engine, Frame};

use crate::block::ElementId;
use crate::player::Player;
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
            if self.total() as usize >= self.capacity {
                all = false;
                continue;
            }
            match self.counts.iter_mut().find(|(e, _)| *e == element) {
                Some((_, count)) => *count += 1,
                None => self.counts.push((element, 1)),
            }
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
            }
        }
        self.counts.retain(|&(_, count)| count > 0);
        self.rev += 1;
        true
    }

    /// Total elements held, across all kinds.
    pub fn total(&self) -> u32 {
        self.counts.iter().map(|&(_, c)| c).sum()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Raise the capacity — the "very upgradeable over time" hook.
    pub fn grow(&mut self, extra: usize) {
        self.capacity += extra;
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
        self.rev += 1;
    }
}

/// The coarse, per-frame state a mod may read and mutate. Deliberately holds only
/// whole-game handles (never a voxel), so a mod can't reach into the hot path.
pub struct ModContext<'a> {
    pub player: &'a mut Player,
    pub world: &'a mut World,
    pub screen_w: i32,
    pub screen_h: i32,
    /// True while the console or a menu is capturing keys, so mods leave input alone.
    pub capturing_text: bool,
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

    /// Per-frame logic while enabled. Runs after movement, before rendering.
    fn update(&mut self, eng: &Engine, ctx: &mut ModContext) {
        let _ = (eng, ctx);
    }

    /// A block was broken into these elements. The event the inventory mod listens
    /// to; a crafting or logging mod could too.
    fn on_block_break(&mut self, elements: &[ElementId], world: &World) {
        let _ = (elements, world);
    }

    /// Draw this mod's HUD while enabled, over the world and under the console.
    fn draw(&mut self, f: &mut Frame, screen_w: i32, screen_h: i32) {
        let _ = (f, screen_w, screen_h);
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
    /// The default install: the bare-list inventory mod and the crafting mod,
    /// both enabled, sharing one [`ElementStash`] — inventory fills it from
    /// broken blocks, crafting spends it.
    pub fn with_defaults() -> Self {
        let mut mods = Self {
            entries: Vec::new(),
        };
        let stash = Rc::new(RefCell::new(ElementStash::new(inventory::START_CAPACITY)));
        mods.install(Box::new(inventory::InventoryMod::new(stash.clone())), true);
        mods.install(Box::new(crafting::CraftingMod::new(stash)), true);
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

    /// Draw every enabled mod's HUD.
    pub fn draw(&mut self, f: &mut Frame, screen_w: i32, screen_h: i32) {
        for entry in &mut self.entries {
            if entry.enabled {
                entry.module.draw(f, screen_w, screen_h);
            }
        }
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
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|e| e.module.name() == name)
        {
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
