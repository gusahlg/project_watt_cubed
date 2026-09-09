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
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::{Duration, Instant};

use crate::block::ElementId;
use crate::derived::Memo;
use crate::mods::{ElementStash, ItemUiState, Mod, ModContext};
use crate::ui::{Anchor, HudElement, Panel, Role, Row, PANEL_FONT};
use crate::world::World;

/// Starting capacity. Large-looking, but with no stacking it is modest — and meant
/// to be upgraded over time.
pub(crate) const START_CAPACITY: usize = 100;

/// How long the "elements lost" warning stays on screen after the last
/// overflowing break.
const OVERFLOW_WARNING: Duration = Duration::from_millis(2500);

pub(crate) const PANEL_X: i32 = 12;
pub(crate) const PANEL_Y: i32 = 44;
const PANEL_WIDTH: i32 = 300;
const PANEL_PAD: i32 = 8;
const FONT_SIZE: i32 = 18;
const LINE_HEIGHT: i32 = FONT_SIZE + 4;
/// Console scrollback occupies the bottom 184 pixels at its maximum extent.
const BOTTOM_RESERVE: i32 = 234;

/// The bare-list inventory view over the shared stash, and its HUD toggle.
pub struct InventoryMod {
    /// The shared element counts (filled here, spent by crafting).
    stash: Rc<RefCell<ElementStash>>,
    /// Shared with crafting so its expanded panel replaces this compact list.
    ui: Rc<Cell<ItemUiState>>,
    /// When a break last overflowed the stash (elements were destroyed), if
    /// within the warning window. Drives the HUD's "elements lost" warning.
    overflow_at: Option<Instant>,
    /// Formatted HUD rows, rebuilt only when the stash, screen, or visibility
    /// flags change.
    hud_cache: RefCell<Memo<(u64, i32, i32, bool, bool), Vec<HudElement>>>,
}

impl InventoryMod {
    pub(crate) fn new(stash: Rc<RefCell<ElementStash>>, ui: Rc<Cell<ItemUiState>>) -> Self {
        Self {
            stash,
            ui,
            overflow_at: None,
            hud_cache: RefCell::new(Memo::new()),
        }
    }

    /// Bottom edge of the compact panel for `kinds` rows at this screen height.
    /// Crafting uses the same calculation to place its equipped hint below it.
    pub(crate) fn panel_bottom(screen_h: i32, kinds: usize) -> i32 {
        let rows = visible_rows(screen_h, kinds).max(1);
        PANEL_Y + PANEL_PAD * 2 + LINE_HEIGHT + 2 + rows as i32 * LINE_HEIGHT
    }
}

fn row_capacity(screen_h: i32) -> usize {
    let content_y = PANEL_Y + PANEL_PAD + LINE_HEIGHT + 2;
    ((screen_h - BOTTOM_RESERVE - content_y) / LINE_HEIGHT).max(1) as usize
}

fn visible_rows(screen_h: i32, kinds: usize) -> usize {
    kinds.min(row_capacity(screen_h))
}

fn paint_inventory(
    stash: &RefCell<ElementStash>,
    world: &World,
    screen_w: i32,
    screen_h: i32,
    visible: bool,
    overflow: bool,
) -> Vec<HudElement> {
    if !visible {
        // The overflow warning outlives the list toggle: shown for a short
        // window after the last overflowing break even while the list is
        // closed, centred where the list's header would sit.
        if overflow {
            return vec![HudElement::Label {
                at: Anchor::Top,
                off: (0, PANEL_Y),
                base_fs: PANEL_FONT,
                role: Role::Danger,
                text: "Inventory full - elements lost!".into(),
            }];
        }
        return Vec::new();
    }

    let width = PANEL_WIDTH.min((screen_w - PANEL_X * 2).max(1));
    let stash = stash.borrow();
    let elements = world.registry().elements();
    let total = stash.total();
    let kind_count = stash.iter().count();
    let shown = visible_rows(screen_h, kind_count);

    let header = if overflow {
        Row::new(Role::Danger, "Inventory full - elements lost!")
    } else {
        Row::new(Role::Warning, format!("Inventory  {total}/{}", stash.capacity()))
    };

    let mut rows = Vec::new();
    if total == 0 {
        rows.push(Row::new(Role::Muted, "(empty)"));
    } else {
        // When the kinds overflow the panel, the last row slot becomes the
        // "+N more" summary instead of an element row.
        let listed = if kind_count > shown { shown - 1 } else { shown };
        for (element, count) in stash.iter().take(listed) {
            rows.push(Row::new(
                Role::Muted,
                format!("{count}x {}", elements.get(element).name),
            ));
        }
        if kind_count > listed {
            rows.push(Row::new(Role::Dim, format!("+{} more", kind_count - listed)));
        }
    }

    vec![HudElement::Panel(Panel {
        at: (PANEL_X, PANEL_Y),
        width,
        header: vec![header].into(),
        rows: rows.into(),
    })]
}

impl Mod for InventoryMod {
    fn name(&self) -> &str {
        "Inventory"
    }

    fn id(&self) -> &'static str {
        "inventory"
    }

    fn description(&self) -> &str {
        "The bare-list inventory and a simple view of it (press I to toggle)."
    }

    fn group(&self) -> &'static str {
        crate::mods::ESSENTIALS
    }

    fn update(&mut self, ctx: &mut ModContext) {
        if ctx.toggle_inventory {
            let mut ui = self.ui.get();
            ui.inventory_visible = !ui.inventory_visible;
            self.ui.set(ui);
        }
        // Expire the overflow warning once its window has passed (drawing no
        // longer mutates state, so the clock is advanced here).
        if self.overflow_at.is_some_and(|at| at.elapsed() > OVERFLOW_WARNING) {
            self.overflow_at = None;
        }
    }

    fn reset(&mut self) {
        self.stash.borrow_mut().clear();
        let mut ui = self.ui.get();
        ui.inventory_visible = true;
        self.ui.set(ui);
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

    fn on_break_rejected(&mut self, elements: &[ElementId]) {
        // The server refused the break this loot came from: take it back.
        // Best-effort — anything already spent can't be revoked, which errs
        // in the player's favour on a rare race rather than going negative.
        self.stash.borrow_mut().revoke(elements);
    }

    fn hud(&self, world: &World, (screen_w, screen_h): (i32, i32), out: &mut Vec<HudElement>) {
        let overflow = self
            .overflow_at
            .is_some_and(|at| at.elapsed() <= OVERFLOW_WARNING);
        let ui = self.ui.get();
        let visible = ui.inventory_visible && !ui.crafting_open;
        let rev = self.stash.borrow().rev();
        let key = (rev, screen_w, screen_h, visible, overflow);
        let stash = &self.stash;
        let mut cache = self.hud_cache.borrow_mut();
        let cached = cache.get_or(key, || {
            paint_inventory(stash, world, screen_w, screen_h, visible, overflow)
        });
        out.extend(cached.iter().cloned());
    }

    fn save_state(&self, world: &World) -> Option<(u16, String)> {
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
        Some((1, names.join(",")))
    }

    fn load_state(&mut self, _version: u16, data: &str, world: &mut World) {
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
        let mut inventory =
            InventoryMod::new(stash.clone(), Rc::new(Cell::new(ItemUiState::default())));
        // A pre-stash save line: one element name per held unit, pickup order,
        // possibly interleaved. Unknown names are skipped, exactly as before.
        inventory.load_state(0, "Stone,Soil,Stone,Bogus", &mut world);
        assert_eq!(stash.borrow().total(), 3);
        assert_eq!(
            stash.borrow().count(crate::block::element::El::Stone.id()),
            2
        );
        // Re-saving emits the same one-name-per-unit format (grouped by
        // first-seen element, which the old grouped HUD view matched anyway).
        assert_eq!(
            inventory.save_state(&world),
            Some((1, "Stone,Stone,Soil".into()))
        );
    }

    #[test]
    fn breaking_into_a_full_stash_arms_the_overflow_warning() {
        let world = World::new(1);
        let stash = Rc::new(RefCell::new(ElementStash::new(1)));
        let mut inventory =
            InventoryMod::new(stash.clone(), Rc::new(Cell::new(ItemUiState::default())));
        let stone = crate::block::element::El::Stone.id();

        // Room left: no warning.
        inventory.on_block_break(&[stone], &world);
        assert!(
            inventory.overflow_at.is_none(),
            "no warning while everything fits"
        );

        // Full: the element is destroyed, and the warning must be armed.
        inventory.on_block_break(&[stone], &world);
        assert_eq!(
            stash.borrow().total(),
            1,
            "the overflow element was dropped"
        );
        let armed = inventory
            .overflow_at
            .expect("dropping elements must arm the warning");
        assert!(
            armed.elapsed() <= OVERFLOW_WARNING,
            "freshly armed: inside the window"
        );

        // Entering another world clears the warning with the rest of the state.
        inventory.reset();
        assert!(
            inventory.overflow_at.is_none(),
            "reset must clear the warning"
        );
    }

    #[test]
    fn load_replaces_previous_contents() {
        let mut world = World::new(1);
        let stash = Rc::new(RefCell::new(ElementStash::new(10)));
        let mut inventory =
            InventoryMod::new(stash.clone(), Rc::new(Cell::new(ItemUiState::default())));
        inventory.load_state(0, "Stone,Stone", &mut world);
        inventory.load_state(0, "Iron", &mut world);
        assert_eq!(stash.borrow().total(), 1);
        assert_eq!(
            stash.borrow().count(crate::block::element::El::Iron.id()),
            1
        );
    }
}
