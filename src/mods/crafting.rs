//! The default crafting mod: take configurations from the stash into a pouch
//! and place them. Natural mixing is gone; the manipulation workbench is a
//! later task.
use std::cell::{Cell, RefCell};
use std::rc::Rc;

use voxel_engine::DVec3;

use crate::block::registry::BlockId;
use crate::block::AIR;
use crate::derived::Memo;
use crate::math::{Aabb, Bounded};
use crate::mods::inventory::{InventoryMod, PANEL_X, PANEL_Y};
use crate::mods::{ItemUiState, Mod, ModContext};
use crate::player::Player;
use crate::stash::ElementStash;
use crate::ui::{visible_window, HudElement, Panel, Role, Row};
use crate::world::World;

const PANEL_WIDTH: i32 = 360;
const PANEL_PAD: i32 = 8;
const FONT_SIZE: i32 = 18;
const LINE_HEIGHT: i32 = FONT_SIZE + 4;
const BOTTOM_RESERVE: i32 = 190;

/// One pouch entry: its id, its portable spec, and how many are left to place.
struct Crafted {
    id: BlockId,
    spec: Box<str>,
    count: u32,
}

/// The crafting panel, the pouch, and right-click placement.
pub struct CraftingMod {
    /// Shared with inventory so this expanded panel replaces its compact view.
    ui: Rc<Cell<ItemUiState>>,
    /// Cursor over the panel rows: stash kinds, then pouch entries.
    cursor: usize,
    /// Pouch holdings, in first-taken order.
    crafted: Vec<Crafted>,
    /// Index into `crafted` of the block RMB places, if any.
    equipped: Option<usize>,
    /// Held block ids mirrored from the stash only when its revision changes.
    held: Vec<BlockId>,
    seen_stash_rev: u64,
    hud_gen: Cell<u64>,
    hud_cache: RefCell<Memo<(u64, u64, i32, i32, bool), Vec<HudElement>>>,
}

impl CraftingMod {
    pub(crate) fn new(ui: Rc<Cell<ItemUiState>>) -> Self {
        Self {
            ui,
            cursor: 0,
            crafted: Vec::new(),
            equipped: None,
            held: Vec::new(),
            seen_stash_rev: 0,
            hud_gen: Cell::new(0),
            hud_cache: RefCell::new(Memo::new()),
        }
    }

    fn bump_hud(&self) {
        self.hud_gen.set(self.hud_gen.get().wrapping_add(1));
    }

    fn is_open(&self) -> bool {
        self.ui.get().crafting_open
    }

    fn set_open(&self, open: bool) {
        let mut ui = self.ui.get();
        if ui.crafting_open == open {
            return;
        }
        ui.crafting_open = open;
        self.ui.set(ui);
        self.bump_hud();
    }

    fn row_count(&self) -> usize {
        self.held.len() + self.crafted.len()
    }

    fn refresh(&mut self, stash: &ElementStash) -> bool {
        let rev = stash.rev();
        if rev == self.seen_stash_rev {
            if self.row_count() == 0 {
                self.cursor = 0;
                return false;
            }
            let c = self.cursor.min(self.row_count() - 1);
            if c != self.cursor {
                self.cursor = c;
                self.bump_hud();
            }
            return false;
        }
        self.held.clear();
        self.held.extend(stash.iter().map(|(id, _)| id));
        self.seen_stash_rev = rev;
        if self.row_count() == 0 {
            self.cursor = 0;
        } else {
            self.cursor = self.cursor.min(self.row_count() - 1);
        }
        self.bump_hud();
        true
    }

    fn navigate(&mut self, ctx: &mut ModContext) {
        let cursor = self.cursor;
        if ctx.nav_up {
            self.cursor = self.cursor.saturating_sub(1);
        }
        if ctx.nav_down && self.row_count() > 0 {
            self.cursor = (self.cursor + 1).min(self.row_count() - 1);
        }
        if ctx.nav_confirm {
            self.activate(ctx);
        }
        if self.cursor != cursor {
            self.bump_hud();
        }
    }

    /// Confirm on a stash row takes one unit into the pouch; on a pouch row, equips it.
    fn activate(&mut self, ctx: &mut ModContext) {
        let held = self.held.len();
        if self.cursor < held {
            let id = self.held[self.cursor];
            if !ctx.player.stash.consume(id, 1) {
                self.refresh(&ctx.player.stash);
                return;
            }
            self.push_loaded(ctx.world, id, 1, false);
            self.bump_hud();
            self.refresh(&ctx.player.stash);
        } else if !self.crafted.is_empty() {
            self.equipped = Some(self.cursor - held);
            self.bump_hud();
        }
    }

    fn try_place(&mut self, ctx: &mut ModContext) {
        if !ctx.place {
            return;
        }
        let Some(equipped) = self.equipped else {
            return;
        };
        if self.crafted[equipped].count == 0 {
            return;
        }
        let Some((x, y, z)) = ctx.place_target else {
            return;
        };
        if ctx.world.block_at(x, y, z) != AIR || cell_aabb(x, y, z).intersects(&ctx.player.aabb()) {
            return;
        }
        ctx.placements.push((x, y, z, self.crafted[equipped].id));
        self.crafted[equipped].count -= 1;
        self.bump_hud();
    }

    fn push_loaded(&mut self, world: &World, id: BlockId, count: u32, equip: bool) {
        let at = match self.crafted.iter().position(|c| c.id == id) {
            Some(at) => {
                self.crafted[at].count += count;
                at
            }
            None => {
                self.crafted.push(Crafted {
                    id,
                    spec: world.registry().spec(id).into(),
                    count,
                });
                self.crafted.len() - 1
            }
        };
        if equip {
            self.equipped = Some(at);
        }
    }
}

fn cell_aabb(x: i32, y: i32, z: i32) -> Aabb {
    Aabb::new(
        DVec3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5),
        DVec3::splat(0.5),
    )
}

fn paint_crafting(
    stash: &ElementStash,
    ui: ItemUiState,
    cursor: usize,
    crafted: &[Crafted],
    equipped: Option<usize>,
    world: &World,
    screen_w: i32,
    screen_h: i32,
) -> Vec<HudElement> {
    let width = PANEL_WIDTH.min((screen_w - PANEL_X * 2).max(1));

    if !ui.crafting_open {
        let Some(equipped) = equipped else {
            return Vec::new();
        };
        let entry = &crafted[equipped];
        let kinds = stash.iter().count();
        let y = if ui.inventory_visible {
            InventoryMod::panel_bottom(screen_h, kinds) + 6
        } else {
            PANEL_Y
        };
        let name = world.registry().label(entry.id).unwrap_or("unknown material");
        let hint = format!("Equipped: {name} x{}", entry.count);
        let hint_w = (hint.chars().count() as i32 * FONT_SIZE + PANEL_PAD * 2).min(width);
        return vec![HudElement::Panel(Panel {
            at: (PANEL_X, y),
            width: hint_w,
            header: Vec::new().into(),
            rows: vec![Row::new(Role::Muted, hint)].into(),
        })];
    }

    let held: Vec<(BlockId, u32)> = stash.iter().collect();
    let held_n = held.len();
    let total_rows = held_n + crafted.len();
    let header_rows = 1;
    let content_y = PANEL_Y + PANEL_PAD + header_rows as i32 * LINE_HEIGHT + 2;
    let capacity = ((screen_h - BOTTOM_RESERVE - content_y) / LINE_HEIGHT).max(1) as usize;
    let window = visible_window(total_rows, cursor, capacity);

    let header = vec![Row::new(
        Role::Warning,
        format!("Pouch  take from stash / place from pouch"),
    )];

    let rows: Vec<Row> = window
        .map(|row_index| {
            let active = cursor == row_index;
            let cursor_mark = if active { ">" } else { " " };
            if row_index < held_n {
                let (id, count) = held[row_index];
                let name = world.registry().label(id).unwrap_or("unknown material");
                Row::new(
                    if active { Role::Accent } else { Role::Muted },
                    format!("{cursor_mark} {count}x {name}"),
                )
            } else {
                let i = row_index - held_n;
                let entry = &crafted[i];
                let name = world.registry().label(entry.id).unwrap_or("unknown material");
                let equipped_mark = if equipped == Some(i) { "[E]" } else { "   " };
                let role = if equipped == Some(i) {
                    Role::Positive
                } else if active {
                    Role::Accent
                } else {
                    Role::Muted
                };
                Row::new(
                    role,
                    format!("{cursor_mark} {equipped_mark} {}x {name}", entry.count),
                )
            }
        })
        .collect();

    vec![HudElement::Panel(Panel {
        at: (PANEL_X, PANEL_Y),
        width,
        header: header.into(),
        rows: rows.into(),
    })]
}

impl Mod for CraftingMod {
    fn name(&self) -> &str {
        "Crafting"
    }

    fn id(&self) -> &'static str {
        "crafting"
    }

    fn reset(&mut self) {
        self.set_open(false);
        self.cursor = 0;
        self.crafted.clear();
        self.equipped = None;
        self.held.clear();
        self.seen_stash_rev = 0;
        self.bump_hud();
    }

    fn description(&self) -> &str {
        "Take blocks from the stash into a pouch and place them (press C)."
    }

    fn group(&self) -> &'static str {
        crate::mods::ESSENTIALS
    }

    fn update(&mut self, ctx: &mut ModContext) {
        self.refresh(&ctx.player.stash);
        if ctx.toggle_crafting {
            self.set_open(!self.is_open());
        }
        if self.is_open() {
            self.navigate(ctx);
        } else {
            self.try_place(ctx);
        }
    }

    fn on_place_rejected(&mut self, id: BlockId, world: &World) {
        self.push_loaded(world, id, 1, false);
        self.bump_hud();
    }

    fn hud(
        &self,
        world: &World,
        player: &Player,
        (screen_w, screen_h): (i32, i32),
        out: &mut Vec<HudElement>,
    ) {
        let ui = self.ui.get();
        let rev = player.stash.rev();
        let key = (
            rev,
            self.hud_gen.get(),
            screen_w,
            screen_h,
            ui.inventory_visible,
        );
        let stash = &player.stash;
        let cursor = self.cursor;
        let crafted = self.crafted.as_slice();
        let equipped = self.equipped;
        let mut cache = self.hud_cache.borrow_mut();
        let cached = cache.get_or(key, || {
            paint_crafting(stash, ui, cursor, crafted, equipped, world, screen_w, screen_h)
        });
        out.extend(cached.iter().cloned());
    }

    fn close_overlay(&mut self) -> bool {
        if self.is_open() {
            self.set_open(false);
            true
        } else {
            false
        }
    }

    fn save_state(&self, _world: &World) -> Option<(u16, String)> {
        if self.crafted.is_empty() {
            return None;
        }
        let entries: Vec<String> = self
            .crafted
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                let star = if self.equipped == Some(i) { "*" } else { "" };
                format!("{star}{}={}", entry.spec, entry.count)
            })
            .collect();
        Some((1, entries.join(",")))
    }

    fn load_state(&mut self, _version: u16, data: &str, world: &mut World) -> u32 {
        self.crafted.clear();
        self.equipped = None;
        self.cursor = 0;
        let mut skipped = 0u32;
        for raw in data.split(',').filter(|s| !s.is_empty()) {
            let (equip, entry) = match raw.strip_prefix('*') {
                Some(rest) => (true, rest),
                None => (false, raw),
            };
            let Some((spec, count)) = entry.rsplit_once('=') else {
                continue;
            };
            let Ok(count) = count.parse::<u32>() else {
                continue;
            };
            let Some(id) = world.registry_mut().parse_spec(spec) else {
                skipped += 1;
                continue;
            };
            self.push_loaded(world, id, count, equip);
        }
        self.bump_hud();
        skipped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_mod() -> CraftingMod {
        CraftingMod::new(Rc::new(Cell::new(ItemUiState::default())))
    }

    #[test]
    fn save_load_round_trips_pouch_counts_and_equipped() {
        let mut world = World::new(1);
        let rock = world.registry().id_by_label("rock").unwrap();
        let soil = world.registry().id_by_label("soil").unwrap();
        let mut crafting = test_mod();
        let rock_spec = world.registry().spec(rock);
        let soil_spec = world.registry().spec(soil);
        crafting.load_state(1, &format!("{soil_spec}=2,*{rock_spec}=1"), &mut world);
        assert_eq!(crafting.crafted.len(), 2);
        assert_eq!(crafting.crafted[0].count, 2);
        assert_eq!(crafting.equipped, Some(1));
        assert_eq!(
            crafting.save_state(&world),
            Some((1, format!("{soil_spec}=2,*{rock_spec}=1")))
        );
    }

    #[test]
    fn unknown_specs_skip_the_entry() {
        let mut world = World::new(1);
        let rock = world.registry().id_by_label("rock").unwrap();
        let spec = world.registry().spec(rock);
        let mut crafting = test_mod();
        let skipped = crafting.load_state(1, &format!("natural:Stone=5,*{spec}=3"), &mut world);
        assert_eq!(skipped, 1);
        assert_eq!(crafting.crafted.len(), 1);
        assert_eq!(crafting.crafted[0].id, rock);
        assert_eq!(crafting.crafted[0].count, 3);
    }

    #[test]
    fn held_element_rows_rebuild_only_after_stash_mutation() {
        let world = World::new(1);
        let rock = world.registry().id_by_label("rock").unwrap();
        let soil = world.registry().id_by_label("soil").unwrap();
        let mut player = Player::new(DVec3::new(0.0, 40.0, 0.0));
        player.stash.add(rock, 2);
        let mut crafting = test_mod();
        crafting.set_open(true);
        assert!(crafting.refresh(&player.stash));
        let held = crafting.held.clone();
        assert_eq!(held, vec![rock]);

        let mut hud1 = Vec::new();
        crafting.hud(&world, &player, (800, 600), &mut hud1);
        let text1 = hud_text(&hud1);
        assert!(!crafting.refresh(&player.stash), "no stash change: rows stay");
        assert_eq!(crafting.held, held);
        let mut hud2 = Vec::new();
        crafting.hud(&world, &player, (800, 600), &mut hud2);
        assert_eq!(hud_text(&hud2), text1, "HUD cache is reused until a stash/pouch mutation");

        player.stash.add(soil, 1);
        assert!(crafting.refresh(&player.stash));
        assert_eq!(crafting.held, vec![rock, soil]);
        let mut hud3 = Vec::new();
        crafting.hud(&world, &player, (800, 600), &mut hud3);
        assert_ne!(hud_text(&hud3), text1, "a stash mutation rebuilds the held rows");

        crafting.cursor = usize::MAX;
        assert!(player.stash.consume(rock, 2), "spend the rock stack");
        assert!(crafting.refresh(&player.stash));
        assert_eq!(crafting.held, vec![soil], "consumed kinds drop out of held rows");
        assert!(
            crafting.cursor < crafting.row_count(),
            "cursor clamps after rows shrink"
        );
        assert!(!crafting.refresh(&player.stash), "an unchanged stash is a no-op");
    }

    #[test]
    fn escape_close_consumes_only_an_open_panel() {
        let mut crafting = test_mod();
        assert!(!crafting.is_open());
        assert!(
            !crafting.close_overlay(),
            "Escape on a closed panel is not consumed"
        );
        assert!(!crafting.is_open());
        crafting.set_open(true);
        assert!(
            crafting.close_overlay(),
            "Escape closes an open panel and is consumed"
        );
        assert!(!crafting.is_open());
        assert!(
            !crafting.close_overlay(),
            "a second Escape is not consumed"
        );
    }

    #[test]
    fn take_moves_one_unit_stash_to_pouch() {
        let world = World::new(1);
        let rock = world.registry().id_by_label("rock").unwrap();
        let mut player = Player::new(DVec3::new(0.0, 40.0, 0.0));
        player.stash.add(rock, 2);
        let mut crafting = test_mod();
        crafting.refresh(&player.stash);
        let mut ctx = ModContext {
            player: &mut player,
            world: &mut World::new(1),
            screen_w: 800,
            screen_h: 600,
            place: false,
            place_target: None,
            toggle_inventory: false,
            toggle_crafting: false,
            nav_up: false,
            nav_down: false,
            nav_confirm: false,
            placements: Vec::new(),
        };
        crafting.activate(&mut ctx);
        assert_eq!(ctx.player.stash.count(rock), 1);
        assert_eq!(crafting.crafted[0].count, 1);
        assert_eq!(crafting.crafted[0].id, rock);
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
