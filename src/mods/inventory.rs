//! The default inventory mod: the game's inventory made accessible.
//!
//! The core inventory is, by design, "merely a list containing all of your items" —
//! no grid, no stacking, and unreachable without a mod. This mod *is* that list
//! plus a minimal on-screen view of it. It fills as you break blocks into their
//! elements, holds up to a (soft, upgradeable) capacity, and can be switched off in
//! the mod menu, at which point the inventory is once again inaccessible.
//!
//! The counts themselves live in the shared [`ElementStash`] (see
//! [`mods`](crate::mods)): this mod fills and displays it, the crafting mod spends
//! from it. Its save format is unchanged from when it owned the items outright —
//! one element name per held unit — so old save files load identically.
use std::cell::RefCell;
use std::rc::Rc;
use std::time::{Duration, Instant};

use voxel_engine::{Color, Engine, Frame, Key};

use crate::block::ElementId;
use crate::console::shadowed;
use crate::mods::{ElementStash, Mod, ModContext};
use crate::world::World;

/// Starting capacity. Large-looking, but with no stacking it is modest — and meant
/// to be upgraded over time.
pub(crate) const START_CAPACITY: usize = 100;

/// How long the "elements lost" warning stays on screen after the last
/// overflowing break.
const OVERFLOW_WARNING: Duration = Duration::from_millis(2500);

/// The bare-list inventory view over the shared stash, and its HUD toggle.
pub struct InventoryMod {
    /// The shared element counts (filled here, spent by crafting).
    stash: Rc<RefCell<ElementStash>>,
    /// Whether the list is currently drawn (toggled with `I`).
    visible: bool,
    /// When a break last overflowed the stash (elements were destroyed), if
    /// within the warning window. Drives the HUD's "elements lost" warning.
    overflow_at: Option<Instant>,
}

impl InventoryMod {
    pub fn new(stash: Rc<RefCell<ElementStash>>) -> Self {
        Self {
            stash,
            visible: true,
            overflow_at: None,
        }
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

    fn reset(&mut self) {
        self.stash.borrow_mut().clear();
        self.visible = false;
        self.overflow_at = None;
    }

    fn on_block_break(&mut self, elements: &[ElementId], _world: &World) {
        // A broken block hands back its elements — each becomes one held unit.
        // `add` is per-element best-effort: a full stash drops the overflow on
        // the floor of the void, so arm the HUD warning — silently destroying
        // elements is the one thing this list must never do quietly.
        if !self.stash.borrow_mut().add(elements) {
            self.overflow_at = Some(Instant::now());
        }
    }

    fn draw(&mut self, f: &mut Frame, world: &World, screen_w: i32, _screen_h: i32) {
        let fs = 18;
        let line_h = fs + 4;
        let x = screen_w - 230;
        let mut y = 90;

        // The overflow warning outlives the list toggle: it is drawn for a
        // short window after the last overflowing break EVEN while the list is
        // closed, above where the list's header sits.
        if let Some(at) = self.overflow_at {
            if at.elapsed() <= OVERFLOW_WARNING {
                shadowed(f, "Inventory full - elements lost!", x, y - line_h, fs, Color::RED);
            } else {
                self.overflow_at = None;
            }
        }

        if !self.visible {
            return;
        }

        let stash = self.stash.borrow();
        let elements = world.registry().elements();

        let header = format!("Inventory  {}/{}", stash.total(), stash.capacity());
        shadowed(f, &header, x, y, fs, Color::GOLD);
        y += line_h + 2;

        if stash.total() == 0 {
            shadowed(f, "  (empty) break blocks", x, y, fs, Color::RAYWHITE);
            return;
        }

        // Cap the visible rows so a full inventory doesn't run off-screen.
        for (element, count) in stash.iter().take(14) {
            let row = format!("  {count}x {}", elements.get(element).name);
            shadowed(f, &row, x, y, fs, Color::RAYWHITE);
            y += line_h;
        }
    }

    fn save_state(&self, world: &World) -> Option<String> {
        // Persist by element name so a save survives element-id changes (e.g. a mod
        // that adds elements ahead of these in the registry). One name per held
        // unit, exactly the pre-stash format, so old and new saves are one format.
        let elements = world.registry().elements();
        let stash = self.stash.borrow();
        let mut names: Vec<&str> = Vec::with_capacity(stash.total() as usize);
        for (element, count) in stash.iter() {
            let name = elements.get(element).name.as_ref();
            names.extend(std::iter::repeat_n(name, count as usize));
        }
        Some(names.join(","))
    }

    fn load_state(&mut self, data: &str, world: &mut World) {
        let elements = world.registry().elements();
        {
            let mut stash = self.stash.borrow_mut();
            stash.clear();
            for name in data.split(',').filter(|s| !s.is_empty()) {
                if let Some(id) = elements.id_by_name(name) {
                    stash.add(&[id]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_save_lines_load_and_resave_in_the_same_format() {
        let mut world = World::new(1);
        let stash = Rc::new(RefCell::new(ElementStash::new(10)));
        let mut inventory = InventoryMod::new(stash.clone());
        // A pre-stash save line: one element name per held unit, pickup order,
        // possibly interleaved. Unknown names are skipped, exactly as before.
        inventory.load_state("Stone,Soil,Stone,Bogus", &mut world);
        assert_eq!(stash.borrow().total(), 3);
        assert_eq!(stash.borrow().count(crate::block::element::El::Stone.id()), 2);
        // Re-saving emits the same one-name-per-unit format (grouped by
        // first-seen element, which the old grouped HUD view matched anyway).
        assert_eq!(
            inventory.save_state(&world).as_deref(),
            Some("Stone,Stone,Soil")
        );
    }

    #[test]
    fn breaking_into_a_full_stash_arms_the_overflow_warning() {
        let world = World::new(1);
        let stash = Rc::new(RefCell::new(ElementStash::new(1)));
        let mut inventory = InventoryMod::new(stash.clone());
        let stone = crate::block::element::El::Stone.id();

        // Room left: no warning.
        inventory.on_block_break(&[stone], &world);
        assert!(inventory.overflow_at.is_none(), "no warning while everything fits");

        // Full: the element is destroyed, and the warning must be armed.
        inventory.on_block_break(&[stone], &world);
        assert_eq!(stash.borrow().total(), 1, "the overflow element was dropped");
        let armed = inventory.overflow_at.expect("dropping elements must arm the warning");
        assert!(armed.elapsed() <= OVERFLOW_WARNING, "freshly armed: inside the window");

        // Entering another world clears the warning with the rest of the state.
        inventory.reset();
        assert!(inventory.overflow_at.is_none(), "reset must clear the warning");
    }

    #[test]
    fn load_replaces_previous_contents() {
        let mut world = World::new(1);
        let stash = Rc::new(RefCell::new(ElementStash::new(10)));
        let mut inventory = InventoryMod::new(stash.clone());
        inventory.load_state("Stone,Stone", &mut world);
        inventory.load_state("Iron", &mut world);
        assert_eq!(stash.borrow().total(), 1);
        assert_eq!(stash.borrow().count(crate::block::element::El::Iron.id()), 1);
    }
}
