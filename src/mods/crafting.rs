//! The default crafting mod: spend gathered elements on new natural blocks, then
//! place them in the world.
//!
//! Crafting is a pure registry affair (see [`crate::block::crafting`]); this mod
//! is the player-facing loop around it: pick up to three element kinds from the
//! player's [`ElementStash`](crate::stash::ElementStash), hit Craft to consume
//! one of each and mint (or re-use) the natural block for that set, then equip
//! a crafted block and right-click to place it. Placements are queued on
//! [`ModContext::placements`]; the game applies them after `mods.update` with
//! the same air/no-player-overlap check this mod runs *before* decrementing a
//! count, so the accounting stays exact (see [`try_place`](CraftingMod::try_place)).
use std::cell::{Cell, RefCell};
use std::rc::Rc;

use voxel_engine::DVec3;

use crate::block::crafting::craft_natural;
use crate::block::registry::BlockId;
use crate::block::{AIR, ElementId};
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

/// One crafted block type the player holds: its id, its (stable, portable) name,
/// and how many are left to place.
struct Crafted {
    id: BlockId,
    name: Box<str>,
    count: u32,
}

/// The crafting panel, the crafted-block pouch, and right-click placement.
pub struct CraftingMod {
    /// Shared with inventory so this expanded panel replaces its compact view.
    ui: Rc<Cell<ItemUiState>>,
    /// Cursor over the panel rows: elements, then Craft, then crafted blocks.
    cursor: usize,
    /// Elements marked for the next craft, in pick order. Any number of
    /// distinct elements combines into one natural block (the docs' "takes in
    /// any amount of unique elements"); the list of element types is the bound.
    selected: Vec<ElementId>,
    /// Every crafted block type, in first-crafted order.
    crafted: Vec<Crafted>,
    /// Index into `crafted` of the block RMB places, if any.
    equipped: Option<usize>,
    /// Held element ids mirrored from the stash only when its revision changes.
    /// Stable gameplay frames therefore do no temporary-vector allocation.
    held_elements: Vec<ElementId>,
    seen_stash_rev: u64,
    /// Bumped when cursor, selection, pouch, or open state change so the HUD
    /// memo can stay keyed on Copy values.
    hud_gen: Cell<u64>,
    hud_cache: RefCell<Memo<(u64, u64, i32, i32, bool), Vec<HudElement>>>,
}

impl CraftingMod {
    pub(crate) fn new(ui: Rc<Cell<ItemUiState>>) -> Self {
        Self {
            ui,
            cursor: 0,
            selected: Vec::new(),
            crafted: Vec::new(),
            equipped: None,
            held_elements: Vec::new(),
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

    /// Total rows the cursor can sit on: one per element kind, the Craft row,
    /// one per crafted block type. Always at least 1 (the Craft row).
    fn row_count(&self) -> usize {
        self.held_elements.len() + 1 + self.crafted.len()
    }

    /// Refresh only after an actual stash mutation. The vector retains capacity,
    /// so pickups/crafts rebuild it without turning stable frames into allocator work.
    fn refresh(&mut self, stash: &ElementStash) -> bool {
        let rev = stash.rev();
        if rev == self.seen_stash_rev {
            let c = self.cursor.min(self.row_count() - 1);
            if c != self.cursor {
                self.cursor = c;
                self.bump_hud();
            }
            return false;
        }
        self.held_elements.clear();
        self.held_elements.extend(stash.iter().map(|(element, _)| element));
        self.selected.retain(|element| self.held_elements.contains(element));
        self.seen_stash_rev = rev;
        self.cursor = self.cursor.min(self.row_count() - 1);
        self.bump_hud();
        true
    }

    /// Navigate panel using intent flags.
    fn navigate(&mut self, ctx: &mut ModContext) {
        let cursor = self.cursor;
        if ctx.nav_up {
            self.cursor = self.cursor.saturating_sub(1);
        }
        if ctx.nav_down {
            self.cursor = (self.cursor + 1).min(self.row_count() - 1);
        }
        if ctx.nav_confirm {
            self.activate(ctx);
        }
        if self.cursor != cursor {
            self.bump_hud();
        }
    }

    /// Enter/L on the current row.
    fn activate(&mut self, ctx: &mut ModContext) {
        let element_count = self.held_elements.len();
        if self.cursor < element_count {
            let element = self.held_elements[self.cursor];
            if let Some(at) = self.selected.iter().position(|&e| e == element) {
                self.selected.remove(at);
            } else {
                self.selected.push(element);
            }
            self.bump_hud();
        } else if self.cursor == element_count {
            self.craft(ctx);
        } else {
            self.equipped = Some(self.cursor - element_count - 1);
            self.bump_hud();
        }
    }

    /// Consume one of each selected element and mint the natural block for the
    /// set. The selection is kept so another Enter crafts another, stock allowing.
    fn craft(&mut self, ctx: &mut ModContext) {
        if self.selected.is_empty() {
            return;
        }
        // Avoid growing the block registry if another mod depleted the stash
        // between selection and activation.
        if self
            .selected
            .iter()
            .any(|&element| ctx.player.stash.count(element) == 0)
        {
            self.refresh(&ctx.player.stash);
            return;
        }
        // Resolve the block first: a full palette refuses NEW compositions,
        // and a refused craft must not consume anything.
        let Some(id) = craft_natural(ctx.world.registry_mut(), &self.selected) else {
            return;
        };
        // All-or-nothing: nothing is consumed unless every pick is in stock.
        if !ctx.player.stash.consume(&self.selected) {
            return;
        }
        match self.crafted.iter_mut().find(|c| c.id == id) {
            Some(entry) => entry.count += 1,
            None => self.crafted.push(Crafted {
                id,
                name: ctx.world.registry().block(id).name.clone(),
                count: 1,
            }),
        }
        self.bump_hud();
        self.refresh(&ctx.player.stash);
    }

    /// RMB while the panel is closed: place the equipped block at the cell
    /// resolved from the input edge's aim, even when mod replay is delayed.
    ///
    /// The actual world write happens in the game when it drains
    /// [`ModContext::placements`], guarded by "cell is air, doesn't overlap the
    /// player". We run the *same* check here before queueing and decrementing,
    /// and the queue is applied later in the same frame against the same world
    /// state — so a placement that costs a block always lands, and a rejected
    /// aim costs nothing.
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
}

/// The unit-cube AABB of a voxel cell, in f64 like all position math (an f32
/// centre would sit whole blocks off at far coordinates).
fn cell_aabb(x: i32, y: i32, z: i32) -> Aabb {
    Aabb::new(
        DVec3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5),
        DVec3::splat(0.5),
    )
}

impl CraftingMod {
    /// Register one restored crafted-block entry (merging duplicate ids).
    fn push_loaded(&mut self, world: &World, id: BlockId, count: u32, equip: bool) {
        let at = match self.crafted.iter().position(|c| c.id == id) {
            Some(at) => {
                self.crafted[at].count += count;
                at
            }
            None => {
                self.crafted.push(Crafted {
                    id,
                    name: world.registry().block(id).name.clone(),
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

fn paint_crafting(
    stash: &ElementStash,
    ui: ItemUiState,
    cursor: usize,
    selected: &[ElementId],
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
        let hint = format!("Equipped: {} x{}", entry.name, entry.count);
        let hint_w = (hint.chars().count() as i32 * FONT_SIZE + PANEL_PAD * 2).min(width);
        return vec![HudElement::Panel(Panel {
            at: (PANEL_X, y),
            width: hint_w,
            header: Vec::new().into(),
            rows: vec![Row::new(Role::Muted, hint)].into(),
        })];
    }

    let elements = world.registry().elements();
    let held: Vec<(ElementId, u32)> = stash.iter().collect();
    let element_count = held.len();
    let total_rows = element_count + 1 + crafted.len();
    let selected_names = selected
        .iter()
        .map(|&id| elements.get(id).name.as_ref())
        .collect::<Vec<_>>()
        .join(" + ");
    let header_rows = 1 + usize::from(!selected_names.is_empty());
    let content_y = PANEL_Y + PANEL_PAD + header_rows as i32 * LINE_HEIGHT + 2;
    let capacity = ((screen_h - BOTTOM_RESERVE - content_y) / LINE_HEIGHT).max(1) as usize;
    let window = visible_window(total_rows, cursor, capacity);

    let mut header = vec![Row::new(
        Role::Warning,
        format!("Crafting  {}/{}", selected.len(), element_count),
    )];
    if !selected_names.is_empty() {
        header.push(Row::new(Role::Dim, selected_names));
    }

    let rows: Vec<Row> = window
        .map(|row_index| {
            let active = cursor == row_index;
            let cursor_mark = if active { ">" } else { " " };
            if row_index < element_count {
                let (element, count) = held[row_index];
                let mark = if selected.contains(&element) { "[x]" } else { "[ ]" };
                Row::new(
                    if active { Role::Accent } else { Role::Muted },
                    format!("{cursor_mark} {mark} {count}x {}", elements.get(element).name),
                )
            } else if row_index == element_count {
                let label = if selected.is_empty() { "select elements" } else { "craft selected" };
                Row::new(
                    if selected.is_empty() { Role::Disabled } else { Role::Warning },
                    format!("{cursor_mark} [ {label} ]"),
                )
            } else {
                let i = row_index - element_count - 1;
                let entry = &crafted[i];
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
                    format!("{cursor_mark} {equipped_mark} {}x {}", entry.count, entry.name),
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
        self.selected.clear();
        self.crafted.clear();
        self.equipped = None;
        self.held_elements.clear();
        self.seen_stash_rev = 0;
        self.bump_hud();
    }

    fn description(&self) -> &str {
        "Craft natural blocks from gathered elements and place them (press C)."
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
        // The server refused the placement: the spent block comes back to the
        // pouch (re-listing it if the entry emptied meanwhile).
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
        let selected = self.selected.as_slice();
        let crafted = self.crafted.as_slice();
        let equipped = self.equipped;
        let mut cache = self.hud_cache.borrow_mut();
        let cached = cache.get_or(key, || {
            paint_crafting(
                stash, ui, cursor, selected, crafted, equipped, world, screen_w, screen_h,
            )
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
        // Persist by block *name* ("Stone+Iron"), the same portable choice the
        // core stash makes for elements: names survive id reshuffles across
        // sessions, and the equipped entry carries a `*` prefix.
        if self.crafted.is_empty() {
            return None;
        }
        let entries: Vec<String> = self
            .crafted
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                let star = if self.equipped == Some(i) { "*" } else { "" };
                format!("{star}{}={}", entry.name, entry.count)
            })
            .collect();
        Some((1, entries.join(",")))
    }

    fn load_state(&mut self, version: u16, data: &str, world: &mut World) {
        self.crafted.clear();
        self.equipped = None;
        self.cursor = 0;
        for raw in data.split(',').filter(|s| !s.is_empty()) {
            let (equip, entry) = match raw.strip_prefix('*') {
                Some(rest) => (true, rest),
                None => (false, raw),
            };
            let Some((name, count)) = entry.rsplit_once('=') else {
                continue;
            };
            let Ok(count) = count.parse::<u32>() else {
                continue;
            };
            // Prefer current names so mods remain free to define their own `*Vein`.
            if let Some(id) = world.registry().id_by_name(name) {
                self.push_loaded(world, id, count, equip);
                continue;
            }
            // Older natural Stone+ore blocks persisted aliases such as `IronVein`.
            let migrated;
            let name = if version == 0 {
                if let Some(ore) = name.strip_suffix("Vein") {
                    migrated = format!("Stone+{ore}");
                    &migrated
                } else {
                    name
                }
            } else {
                name
            };
            // A crafted name is its element names joined with '+'. Resolve them
            // all; if any element is unknown (a save from a modded install),
            // skip the whole entry rather than mint a wrong block.
            let ids: Vec<ElementId> = name
                .split('+')
                .filter_map(|n| world.registry().elements().id_by_name(n))
                .collect();
            if ids.is_empty() || ids.len() != name.split('+').count() {
                continue;
            }
            // Re-craft to get this session's id for the same composition.
            let Some(id) = craft_natural(world.registry_mut(), &ids) else {
                continue;
            };
            self.push_loaded(world, id, count, equip);
        }
        self.bump_hud();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_mod() -> CraftingMod {
        CraftingMod::new(Rc::new(Cell::new(ItemUiState::default())))
    }

    #[test]
    fn save_load_round_trips_crafted_counts_and_equipped() {
        let mut world = World::new(1);
        let mut crafting = test_mod();
        // Copper+Glass is reconstructed from its ordinary element names.
        crafting.load_state(1, "Copper+Glass=2,*Stone=1", &mut world);
        assert_eq!(crafting.crafted.len(), 2);
        assert_eq!(crafting.crafted[0].name.as_ref(), "Copper+Glass");
        assert_eq!(crafting.crafted[0].count, 2);
        assert_eq!(crafting.equipped, Some(1));
        assert_eq!(
            crafting.save_state(&world),
            Some((1, "Copper+Glass=2,*Stone=1".into()))
        );
    }

    #[test]
    fn unknown_element_names_skip_the_entry() {
        let mut world = World::new(1);
        let mut crafting = test_mod();
        crafting.load_state(1, "Stone+Unobtainium=5,Iron=3", &mut world);
        assert_eq!(
            crafting.crafted.len(),
            1,
            "unknown-element entry is skipped"
        );
        assert_eq!(crafting.crafted[0].name.as_ref(), "Iron");
        assert_eq!(crafting.crafted[0].count, 3);
    }

    #[test]
    fn loaded_names_recraft_to_registry_ids() {
        let mut world = World::new(1);
        let mut crafting = test_mod();
        crafting.load_state(1, "Copper+Glass=1", &mut world);
        let id = crafting.crafted[0].id;
        assert_eq!(world.registry().id_by_name("Copper+Glass"), Some(id));
    }

    #[test]
    fn legacy_vein_names_migrate_to_compositions() {
        let mut world = World::new(1);
        let mut crafting = test_mod();
        crafting.load_state(0, "*IronVein=2", &mut world);
        assert_eq!(
            crafting.save_state(&world),
            Some((1, "*Stone+Iron=2".into()))
        );
    }

    #[test]
    fn vein_alias_is_not_rewritten_on_versioned_saves() {
        let mut world = World::new(1);
        let mut crafting = test_mod();
        crafting.load_state(1, "*IronVein=2", &mut world);
        assert!(crafting.crafted.is_empty());
    }

    #[test]
    fn held_element_rows_rebuild_only_after_stash_mutation() {
        let mut stash = ElementStash::new(10);
        stash.add(&[ElementId(1), ElementId(2), ElementId(2)]);
        let mut crafting = test_mod();
        assert!(crafting.refresh(&stash));
        assert_eq!(crafting.held_elements, vec![ElementId(1), ElementId(2)]);
        assert!(!crafting.refresh(&stash), "an unchanged stash is a constant-time no-op");

        crafting.selected = vec![ElementId(1), ElementId(2)];
        crafting.cursor = usize::MAX;
        assert!(stash.consume(&[ElementId(1)]));
        assert!(crafting.refresh(&stash));
        assert_eq!(crafting.held_elements, vec![ElementId(2)]);
        assert_eq!(crafting.selected, vec![ElementId(2)]);
        assert!(crafting.cursor < crafting.row_count());
        assert!(!crafting.refresh(&stash));
    }

    #[test]
    fn escape_close_consumes_only_an_open_panel() {
        let mut crafting = test_mod();
        assert!(!crafting.close_overlay());
        crafting.set_open(true);
        assert!(crafting.close_overlay());
        assert!(!crafting.is_open());
        assert!(!crafting.close_overlay());
    }
}
