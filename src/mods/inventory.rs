//! The default inventory mod: the game's inventory made accessible.
//!
//! The core inventory is, by design, "merely a list containing all of your items" —
//! no grid, no stacking, and unreachable without a mod. This mod *is* that list
//! plus a minimal on-screen view of it. The counts live on the player
//! ([`crate::stash::ElementStash`]); this mod displays them and can be switched
//! off in the mod menu, at which point the inventory is once again inaccessible
//! — the elements stay on the player.
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::{Duration, Instant};

use crate::block::ElementId;
use crate::derived::Memo;
use crate::mods::{ItemUiState, Mod, ModContext};
use crate::player::Player;
use crate::stash::ElementStash;
use crate::ui::{Anchor, HudElement, Panel, Role, Row, PANEL_FONT};
use crate::world::World;

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

/// The bare-list inventory view over the core stash, and its HUD toggle.
pub struct InventoryMod {
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
    pub(crate) fn new(ui: Rc<Cell<ItemUiState>>) -> Self {
        Self {
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
    stash: &ElementStash,
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
        let mut ui = self.ui.get();
        ui.inventory_visible = true;
        self.ui.set(ui);
        self.overflow_at = None;
    }

    fn on_block_break(&mut self, _elements: &[ElementId], _world: &World, overflow: bool) {
        // The core already deposited. Arm the HUD warning when anything was
        // dropped — silently destroying elements is the one thing this list
        // must never do quietly.
        if overflow {
            self.overflow_at = Some(Instant::now());
        }
    }

    fn hud(
        &self,
        world: &World,
        player: &Player,
        (screen_w, screen_h): (i32, i32),
        out: &mut Vec<HudElement>,
    ) {
        let overflow = self
            .overflow_at
            .is_some_and(|at| at.elapsed() <= OVERFLOW_WARNING);
        let ui = self.ui.get();
        let visible = ui.inventory_visible && !ui.crafting_open;
        let rev = player.stash.rev();
        let key = (rev, screen_w, screen_h, visible, overflow);
        let stash = &player.stash;
        let mut cache = self.hud_cache.borrow_mut();
        let cached = cache.get_or(key, || {
            paint_inventory(stash, world, screen_w, screen_h, visible, overflow)
        });
        out.extend(cached.iter().cloned());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::element::El;
    use crate::mods::Mods;
    use crate::ui::HudElement;
    use voxel_engine::DVec3;

    #[test]
    fn overflowing_break_notification_arms_the_warning() {
        let world = World::new(1);
        let mut inventory = InventoryMod::new(Rc::new(Cell::new(ItemUiState::default())));
        let stone = El::Stone.id();

        inventory.on_block_break(&[stone], &world, false);
        assert!(
            inventory.overflow_at.is_none(),
            "no warning while everything fits"
        );

        inventory.on_block_break(&[stone], &world, true);
        let armed = inventory
            .overflow_at
            .expect("dropping elements must arm the warning");
        assert!(
            armed.elapsed() <= OVERFLOW_WARNING,
            "freshly armed: inside the window"
        );

        inventory.reset();
        assert!(
            inventory.overflow_at.is_none(),
            "reset must clear the warning"
        );
    }

    #[test]
    fn disabling_inventory_does_not_destroy_mined_elements() {
        let world = World::new(1);
        let mut player = Player::new(DVec3::new(0.0, 40.0, 0.0));
        let mut mods = Mods::with_defaults();
        mods.set_enabled("inventory", false);

        let stone = El::Stone.id();
        let elements = [stone, stone, El::Soil.id()];
        let overflow = !player.stash.add(&elements);
        mods.on_block_break(&elements, &world, overflow);

        assert!(!overflow);
        assert_eq!(player.stash.total(), 3, "core keeps the elements");
        assert_eq!(player.stash.count(stone), 2);
        assert_eq!(player.stash.count(El::Soil.id()), 1);

        let mut hidden = Vec::new();
        mods.hud(&world, &player, (800, 600), &mut hidden);
        assert!(
            !hud_text(&hidden).contains("Stone"),
            "disabled inventory must not present the list"
        );

        mods.set_enabled("inventory", true);
        let mut shown = Vec::new();
        mods.hud(&world, &player, (800, 600), &mut shown);
        let text = hud_text(&shown);
        assert!(
            text.contains("2x Stone"),
            "re-enabled HUD lists the held stone: {text}"
        );
        assert!(
            text.contains("1x Soil"),
            "re-enabled HUD lists the held soil: {text}"
        );
    }

    fn hud_text(elements: &[HudElement]) -> String {
        let mut out = String::new();
        for el in elements {
            match el {
                HudElement::Label { text, .. } => {
                    out.push_str(text);
                    out.push('\n');
                }
                HudElement::Panel(panel) => {
                    for row in panel.header.iter().chain(panel.rows.iter()) {
                        out.push_str(&row.text);
                        out.push('\n');
                    }
                }
            }
        }
        out
    }
}
