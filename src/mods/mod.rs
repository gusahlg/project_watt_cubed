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
pub mod inventory;

use raylib::prelude::*;

use crate::block::ElementId;
use crate::player::Player;
use crate::world::World;

/// The coarse, per-frame state a mod may read and mutate. Deliberately holds only
/// whole-game handles (never a voxel), so a mod can't reach into the hot path.
pub struct ModContext<'a> {
    pub player: &'a mut Player,
    pub world: &'a mut World,
    pub screen_w: i32,
    pub screen_h: i32,
    /// True while the console or a menu is capturing keys, so mods leave input alone.
    pub capturing_text: bool,
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
    fn update(&mut self, rl: &RaylibHandle, ctx: &mut ModContext) {
        let _ = (rl, ctx);
    }

    /// A block was broken into these elements. The event the inventory mod listens
    /// to; a crafting or logging mod could too.
    fn on_block_break(&mut self, elements: &[ElementId], world: &World) {
        let _ = (elements, world);
    }

    /// Draw this mod's HUD while enabled, over the world and under the console.
    fn draw(&self, d: &mut RaylibDrawHandle, screen_w: i32, screen_h: i32) {
        let _ = (d, screen_w, screen_h);
    }

    /// Serialise persistent state to a single line for the save file, or `None` if
    /// the mod has nothing to persist. `world` resolves ids to portable names.
    fn save_state(&self, world: &World) -> Option<String> {
        let _ = world;
        None
    }

    /// Restore state produced by [`save_state`](Self::save_state).
    fn load_state(&mut self, data: &str, world: &World) {
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
    /// The default install: the bare-list inventory mod, enabled — exactly the
    /// "default inventory mod installed and enabled by default" the design calls for.
    pub fn with_defaults() -> Self {
        let mut mods = Self {
            entries: Vec::new(),
        };
        mods.install(Box::new(inventory::InventoryMod::new()), true);
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
    pub fn update(&mut self, rl: &RaylibHandle, ctx: &mut ModContext) {
        for entry in &mut self.entries {
            if entry.enabled {
                entry.module.update(rl, ctx);
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
    pub fn draw(&self, d: &mut RaylibDrawHandle, screen_w: i32, screen_h: i32) {
        for entry in &self.entries {
            if entry.enabled {
                entry.module.draw(d, screen_w, screen_h);
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
    pub fn load_state(&mut self, name: &str, data: &str, world: &World) {
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|e| e.module.name() == name)
        {
            entry.module.load_state(data, world);
        }
    }
}
