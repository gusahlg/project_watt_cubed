//! The default inventory mod: the core stash made visible, and the way to equip what you hold.
//!
//! The core inventory is, by design, "merely a list containing all of your items" — no grid, no
//! stacking, and unreachable without a mod. Press I to open this panel: ↑/↓ (or the wheel) choose
//! a material, a number key 1-9 equips it into that hotbar slot, Enter equips it into the slot
//! currently selected (or the first free one when the hand is selected), Esc or I closes. The
//! counts live on the player ([`crate::stash::ElementStash`]); switching the mod off makes the
//! list inaccessible again but keeps every unit on the player.
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::{Duration, Instant};

use crate::block::BlockId;
use crate::derived::Memo;
use crate::mods::hotbar::{HotbarState, SLOTS};
use crate::mods::{ItemUiState, Mod, ModContext};
use crate::player::Player;
use crate::stash::ElementStash;
use crate::ui::{visible_window, Anchor, HudElement, Panel, Role, Row, PANEL_FONT};
use crate::world::World;

/// How long the "elements lost" warning stays on screen after the last overflowing break.
const OVERFLOW_WARNING: Duration = Duration::from_millis(2500);

pub(crate) const PANEL_X: i32 = 12;
pub(crate) const PANEL_Y: i32 = 44;
const PANEL_WIDTH: i32 = 420;
const PANEL_PAD: i32 = 8;
const FONT_SIZE: i32 = 18;
const LINE_HEIGHT: i32 = FONT_SIZE + 4;
/// The console scrollback and the hotbar occupy the bottom of the screen.
const BOTTOM_RESERVE: i32 = 260;

/// The inventory list over the core stash, with a cursor that equips into the hotbar.
pub struct InventoryMod {
    ui: Rc<Cell<ItemUiState>>,
    bar: Rc<Cell<HotbarState>>,
    /// Row under the cursor (index into the stash's first-seen order).
    cursor: Cell<usize>,
    /// When a break last overflowed the stash (units were destroyed), if within the window.
    overflow_at: Option<Instant>,
    /// Formatted HUD, rebuilt only when what it shows changes.
    hud_cache: RefCell<Memo<HudKey, Vec<HudElement>>>,
}

/// What the panel shows: stash revision, screen size, visibility, overflow warning, cursor, slots,
/// registry size.
type HudKey = (u64, i32, i32, bool, bool, usize, HotbarState, usize);

impl InventoryMod {
    pub(crate) fn new(ui: Rc<Cell<ItemUiState>>, bar: Rc<Cell<HotbarState>>) -> Self {
        Self { ui, bar, cursor: Cell::new(0), overflow_at: None, hud_cache: RefCell::new(Memo::new()) }
    }

    fn visible(&self) -> bool {
        self.ui.get().inventory_visible
    }

    fn set_visible(&self, on: bool) {
        let mut ui = self.ui.get();
        ui.inventory_visible = on;
        self.ui.set(ui);
    }

    /// Equip the material under the cursor into hotbar slot `key` (1..=9).
    fn equip(&self, stash: &ElementStash, key: usize) {
        let Some((id, _)) = stash.iter().nth(self.cursor.get()) else { return };
        let mut bar = self.bar.get();
        bar.equip(key, id);
        bar.selected = key;
        self.bar.set(bar);
    }
}

fn row_capacity(screen_h: i32) -> usize {
    let content_y = PANEL_Y + PANEL_PAD + 2 * LINE_HEIGHT + 2;
    ((screen_h - BOTTOM_RESERVE - content_y) / LINE_HEIGHT).max(1) as usize
}

#[allow(clippy::too_many_arguments)]
fn paint_inventory(
    stash: &ElementStash,
    world: &World,
    bar: &HotbarState,
    cursor: usize,
    screen_w: i32,
    screen_h: i32,
    visible: bool,
    overflow: bool,
) -> Vec<HudElement> {
    if !visible {
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
    let total = stash.total();
    let items: Vec<(BlockId, u32)> = stash.iter().collect();
    let header = vec![
        if overflow {
            Row::new(Role::Danger, "Inventory full - elements lost!")
        } else {
            Row::new(Role::Warning, format!("Inventory  {total}/{}", stash.capacity()))
        },
        Row::new(Role::Dim, "↑↓ choose · 1-9 equip · Enter: current slot · I close"),
    ];
    let mut rows = Vec::new();
    if items.is_empty() {
        rows.push(Row::new(Role::Muted, "(empty — break something with the bare hand)"));
    } else {
        let reg = world.registry();
        for i in visible_window(items.len(), cursor, row_capacity(screen_h)) {
            let (id, count) = items[i];
            let slot = bar.key_of(id).map_or(String::new(), |k| format!("[{k}] "));
            let marker = if i == cursor { "▶ " } else { "  " };
            let n = reg.configuration(id).len();
            let role = if i == cursor { Role::Accent } else { Role::Muted };
            rows.push(
                Row::new(role, format!("{marker}{slot}{count}x {}  · {n} el.", reg.display_name(id)))
                    .with_swatch(reg.color(id)),
            );
        }
    }
    vec![HudElement::Panel(Panel { at: (PANEL_X, PANEL_Y), width, header: header.into(), rows: rows.into() })]
}

impl Mod for InventoryMod {
    fn name(&self) -> &str {
        "Inventory"
    }

    fn id(&self) -> &'static str {
        "inventory"
    }

    fn description(&self) -> &str {
        "Your held materials (press I): choose one and press 1-9 to equip it on the hotbar."
    }

    fn group(&self) -> &'static str {
        crate::mods::ESSENTIALS
    }

    fn update(&mut self, ctx: &mut ModContext) {
        if ctx.toggle_inventory {
            self.set_visible(!self.visible());
        }
        if self.overflow_at.is_some_and(|at| at.elapsed() > OVERFLOW_WARNING) {
            self.overflow_at = None;
        }
        if !self.visible() {
            return;
        }
        let kinds = ctx.player.stash.iter().count();
        let mut cursor = self.cursor.get().min(kinds.saturating_sub(1));
        if ctx.nav_up {
            cursor = cursor.saturating_sub(1);
        }
        if ctx.nav_down && cursor + 1 < kinds {
            cursor += 1;
        }
        self.cursor.set(cursor);
        if let Some(k) = ctx.hotbar_key.filter(|&k| (1..=SLOTS as u8).contains(&k)) {
            self.equip(&ctx.player.stash, k as usize);
        }
        if ctx.nav_confirm {
            let bar = self.bar.get();
            let key = if bar.selected != 0 {
                bar.selected
            } else {
                bar.slots.iter().position(|s| s.is_none()).map_or(1, |i| i + 1)
            };
            self.equip(&ctx.player.stash, key);
        }
    }

    fn reset(&mut self) {
        self.set_visible(false);
        self.cursor.set(0);
        self.overflow_at = None;
    }

    fn close_overlay(&mut self) -> bool {
        if self.visible() {
            self.set_visible(false);
            return true;
        }
        false
    }

    fn on_block_break(&mut self, _id: BlockId, _world: &World, overflow: bool) {
        if overflow {
            self.overflow_at = Some(Instant::now());
        }
    }

    fn hud(&self, world: &World, player: &Player, (screen_w, screen_h): (i32, i32), out: &mut Vec<HudElement>) {
        let overflow = self.overflow_at.is_some_and(|at| at.elapsed() <= OVERFLOW_WARNING);
        let visible = self.visible();
        let bar = self.bar.get();
        let cursor = self.cursor.get();
        let key = (
            player.stash.rev(),
            screen_w,
            screen_h,
            visible,
            overflow,
            cursor,
            bar,
            world.registry().block_count(),
        );
        let stash = &player.stash;
        let mut cache = self.hud_cache.borrow_mut();
        let cached = cache.get_or(key, || paint_inventory(stash, world, &bar, cursor, screen_w, screen_h, visible, overflow));
        out.extend(cached.iter().cloned());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mods::Mods;
    use crate::render_config::RenderConfig;
    use crate::world::generation::WorldgenKind;
    use voxel_engine::DVec3;

    fn world() -> World {
        World::with_kind(1, RenderConfig::default(), WorldgenKind::Flat, true)
    }

    fn inventory() -> InventoryMod {
        InventoryMod::new(Rc::new(Cell::new(ItemUiState::default())), Rc::new(Cell::new(HotbarState::default())))
    }

    #[test]
    fn overflowing_break_notification_arms_the_warning() {
        let world = world();
        let mut inventory = inventory();
        let rock = world.registry().id_by_label("rock").unwrap();
        inventory.on_block_break(rock, &world, false);
        assert!(inventory.overflow_at.is_none(), "no warning while everything fits");
        inventory.on_block_break(rock, &world, true);
        let armed = inventory.overflow_at.expect("dropping units must arm the warning");
        assert!(armed.elapsed() <= OVERFLOW_WARNING);
        inventory.reset();
        assert!(inventory.overflow_at.is_none(), "reset must clear the warning");
    }

    #[test]
    fn number_keys_equip_the_row_under_the_cursor() {
        let mut world = world();
        let mut player = Player::new(DVec3::new(0.0, 70.0, 0.0));
        let rock = world.registry().id_by_label("rock").unwrap();
        let soil = world.registry().id_by_label("soil").unwrap();
        player.stash.add(rock, 2);
        player.stash.add(soil, 1);
        let mut inv = inventory();
        inv.set_visible(true);
        let mut ctx = crate::mods::ModContext {
            player: &mut player,
            world: &mut world,
            screen_w: 800,
            screen_h: 600,
            place: false,
            place_target: None,
            toggle_inventory: false,
            nav_up: false,
            nav_down: true,
            nav_left: false,
            nav_right: false,
            nav_tab: false,
            nav_confirm: false,
            hotbar_key: Some(4),
            hotbar_cycle: 0,
            networked: false,
            placements: Vec::new(),
        };
        inv.update(&mut ctx);
        let bar = inv.bar.get();
        assert_eq!(bar.slots[3], Some(soil), "cursor moved to the second row, key 4 equipped it");
        assert_eq!(bar.selected, 4, "equipping selects the slot");
        assert!(inv.close_overlay(), "Esc closes the open panel");
        assert!(!inv.close_overlay());
    }

    #[test]
    fn disabling_inventory_does_not_destroy_mined_blocks() {
        let world = world();
        let mut player = Player::new(DVec3::new(0.0, 70.0, 0.0));
        let mut mods = Mods::with_defaults();
        let rock = world.registry().id_by_label("rock").unwrap();
        assert!(player.stash.add(rock, 2));
        mods.set_enabled("inventory", false);
        mods.on_block_break(rock, &world, false);
        assert_eq!(player.stash.count(rock), 2, "core keeps the configurations");
        mods.set_enabled("inventory", true);
        let inv = inventory();
        inv.set_visible(true);
        let mut shown = Vec::new();
        inv.hud(&world, &player, (800, 600), &mut shown);
        let text = crate::ui::hud_text(&shown);
        assert!(text.contains(&format!("2x {}", world.registry().display_name(rock))), "{text}");
        assert!(!text.contains("rock"), "labels never reach the player: {text}");
    }
}
