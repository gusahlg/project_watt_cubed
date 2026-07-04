//! The default inventory mod: the game's inventory made accessible.
//!
//! The core inventory is, by design, "merely a list containing all of your items" —
//! no grid, no stacking, and unreachable without a mod. This mod *is* that list
//! plus a minimal on-screen view of it. It fills as you break blocks into their
//! elements, holds up to a (soft, upgradeable) capacity, and can be switched off in
//! the mod menu, at which point the inventory is once again inaccessible.
use voxel_engine::{Color, Engine, Frame, Key};

use crate::block::ElementId;
use crate::console::shadowed;
use crate::mods::{Mod, ModContext};
use crate::world::World;

/// Starting capacity. Large-looking, but with no stacking it is modest — and meant
/// to be upgraded over time.
const START_CAPACITY: usize = 100;

/// One held item: an element and its display name (cached at pickup so drawing and
/// saving need no registry lookup).
struct Item {
    element: ElementId,
    name: Box<str>,
}

/// The bare-list inventory and its HUD toggle.
pub struct InventoryMod {
    /// Every item, in pickup order. One element unit per slot; no stacking.
    items: Vec<Item>,
    /// Soft cap on slots. Upgradeable via [`grow`](Self::grow).
    capacity: usize,
    /// Whether the list is currently drawn (toggled with `I`).
    visible: bool,
    /// Cached, pre-formatted "  {count}x {name}" rows for drawing, grouped by
    /// element in first-seen order. Rebuilt lazily instead of every frame.
    rows: Vec<String>,
    /// Set whenever `items` changes, telling `draw` to rebuild `rows`.
    dirty: bool,
}

impl InventoryMod {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            capacity: START_CAPACITY,
            visible: true,
            rows: Vec::new(),
            dirty: false,
        }
    }

    /// Try to add one item; returns `false` (and keeps nothing) when full.
    fn add(&mut self, element: ElementId, name: Box<str>) -> bool {
        if self.items.len() >= self.capacity {
            return false;
        }
        self.items.push(Item { element, name });
        self.dirty = true;
        true
    }

    /// Raise the capacity — the "very upgradeable over time" hook.
    pub fn grow(&mut self, extra: usize) {
        self.capacity += extra;
    }

    /// Rebuild the cached rows: group the flat list into `count x name` lines by
    /// element in first-seen order.
    fn rebuild_rows(&mut self) {
        let mut groups: Vec<(ElementId, &str, usize)> = Vec::new();
        for item in &self.items {
            match groups.iter_mut().find(|(e, _, _)| *e == item.element) {
                Some((_, _, count)) => *count += 1,
                None => groups.push((item.element, item.name.as_ref(), 1)),
            }
        }
        self.rows = groups
            .into_iter()
            .map(|(_, name, count)| format!("  {count}x {name}"))
            .collect();
        self.dirty = false;
    }
}

impl Default for InventoryMod {
    fn default() -> Self {
        Self::new()
    }
}

impl Mod for InventoryMod {
    fn name(&self) -> &str {
        "Inventory"
    }

    fn description(&self) -> &str {
        "The bare-list inventory and a simple view of it (press I to toggle)."
    }

    fn update(&mut self, eng: &Engine, ctx: &mut ModContext) {
        // `I` shows/hides the list, but not while something else is capturing keys.
        if !ctx.capturing_text && eng.is_key_pressed(Key::I) {
            self.visible = !self.visible;
        }
    }

    fn on_block_break(&mut self, elements: &[ElementId], world: &World) {
        // A broken block hands back its elements — each becomes one item.
        let registry = world.registry().elements();
        for &element in elements {
            let name = registry.get(element).name.clone();
            self.add(element, name);
        }
    }

    fn draw(&mut self, f: &mut Frame, screen_w: i32, _screen_h: i32) {
        if !self.visible {
            return;
        }
        if self.dirty {
            self.rebuild_rows();
        }

        let fs = 18;
        let line_h = fs + 4;
        let x = screen_w - 230;
        let mut y = 90;

        let header = format!("Inventory  {}/{}", self.items.len(), self.capacity);
        shadowed(f, &header, x, y, fs, Color::GOLD);
        y += line_h + 2;

        if self.rows.is_empty() {
            shadowed(f, "  (empty) break blocks", x, y, fs, Color::RAYWHITE);
            return;
        }

        // Cap the visible rows so a full inventory doesn't run off-screen.
        for row in self.rows.iter().take(14) {
            shadowed(f, row, x, y, fs, Color::RAYWHITE);
            y += line_h;
        }
    }

    fn save_state(&self, _world: &World) -> Option<String> {
        // Persist by element name so a save survives element-id changes (e.g. a mod
        // that adds elements ahead of these in the registry).
        let names: Vec<&str> = self.items.iter().map(|i| i.name.as_ref()).collect();
        Some(names.join(","))
    }

    fn load_state(&mut self, data: &str, world: &World) {
        let registry = world.registry().elements();
        self.items.clear();
        self.dirty = true;
        for name in data.split(',').filter(|s| !s.is_empty()) {
            if let Some(id) = registry.id_by_name(name) {
                self.add(id, name.into());
            }
        }
    }
}
