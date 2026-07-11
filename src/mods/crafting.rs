//! The default crafting mod: spend gathered elements on new natural blocks, then
//! place them in the world.
//!
//! Crafting is a pure registry affair (see [`crate::block::crafting`]); this mod
//! is the player-facing loop around it: pick up to three element kinds from the
//! shared [`ElementStash`], hit Craft to consume one of each and mint (or re-use)
//! the natural block for that set, then equip a crafted block and right-click to
//! place it. Placements are queued on [`ModContext::placements`]; the game applies
//! them after `mods.update` with the same air/no-player-overlap check this mod
//! runs *before* decrementing a count, so the accounting stays exact (see
//! [`try_place`](CraftingMod::try_place)).
use std::cell::{Cell, RefCell};
use std::rc::Rc;

use voxel_engine::{DVec3, Engine};

use crate::block::crafting::craft_natural;
use crate::block::registry::BlockId;
use crate::block::{AIR, ElementId};
use crate::interact;
use crate::math::{Aabb, Bounded};
use crate::mods::inventory::{InventoryMod, PANEL_X, PANEL_Y};
use crate::mods::{ElementStash, ItemUiState, Mod, ModContext};
use crate::ui::{visible_window, HudElement, Panel, Role, Row};
use crate::world::World;

/// How far the player can reach to place a block — matches the break reach.
const PLACE_REACH: f64 = 6.0;
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
    /// The shared element counts (filled by the inventory mod, spent here).
    stash: Rc<RefCell<ElementStash>>,
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
}

impl CraftingMod {
    pub(crate) fn new(stash: Rc<RefCell<ElementStash>>, ui: Rc<Cell<ItemUiState>>) -> Self {
        Self {
            stash,
            ui,
            cursor: 0,
            selected: Vec::new(),
            crafted: Vec::new(),
            equipped: None,
        }
    }

    fn is_open(&self) -> bool {
        self.ui.get().crafting_open
    }

    fn set_open(&self, open: bool) {
        let mut ui = self.ui.get();
        ui.crafting_open = open;
        self.ui.set(ui);
    }

    /// The held element kinds in stash order — the navigable element rows. Read
    /// live from the shared stash (small, changes rarely) so no cache can drift.
    fn elements(&self) -> Vec<ElementId> {
        self.stash.borrow().iter().map(|(e, _)| e).collect()
    }

    /// Total rows the cursor can sit on: one per element kind, the Craft row,
    /// one per crafted block type. Always at least 1 (the Craft row).
    fn row_count(&self) -> usize {
        self.elements().len() + 1 + self.crafted.len()
    }

    /// Drop selections whose element ran out, and keep the cursor on a real row.
    fn refresh(&mut self) {
        let present = self.elements();
        self.selected.retain(|e| present.contains(e));
        self.cursor = self.cursor.min(self.row_count() - 1);
    }

    /// Navigate panel using intent flags.
    fn navigate(&mut self, ctx: &mut ModContext) {
        if ctx.nav_up {
            self.cursor = self.cursor.saturating_sub(1);
        }
        if ctx.nav_down {
            self.cursor = (self.cursor + 1).min(self.row_count() - 1);
        }
        if ctx.nav_confirm {
            self.activate(ctx);
        }
    }

    /// Enter/L on the current row.
    fn activate(&mut self, ctx: &mut ModContext) {
        let elements = self.elements();
        if self.cursor < elements.len() {
            let element = elements[self.cursor];
            if let Some(at) = self.selected.iter().position(|&e| e == element) {
                self.selected.remove(at);
            } else {
                self.selected.push(element);
            }
        } else if self.cursor == elements.len() {
            self.craft(ctx);
        } else {
            self.equipped = Some(self.cursor - elements.len() - 1);
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
            .any(|&element| self.stash.borrow().count(element) == 0)
        {
            self.refresh();
            return;
        }
        // Resolve the block first: a full palette refuses NEW compositions,
        // and a refused craft must not consume anything.
        let Some(id) = craft_natural(ctx.world.registry_mut(), &self.selected) else {
            return;
        };
        // All-or-nothing: nothing is consumed unless every pick is in stock.
        if !self.stash.borrow_mut().consume(&self.selected) {
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
        self.refresh();
    }

    /// RMB while the panel is closed: place the equipped block against whatever
    /// the player is aiming at.
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
        let Some(hit) = interact::raycast(
            ctx.world,
            ctx.player.position,
            ctx.player.forward(),
            PLACE_REACH,
        ) else {
            return;
        };
        let (x, y, z) = hit.previous;
        if ctx.world.block_at(x, y, z) != AIR || cell_aabb(x, y, z).intersects(&ctx.player.aabb()) {
            return;
        }
        ctx.placements.push((x, y, z, self.crafted[equipped].id));
        self.crafted[equipped].count -= 1;
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

impl Mod for CraftingMod {
    fn name(&self) -> &str {
        "Crafting"
    }

    fn reset(&mut self) {
        self.set_open(false);
        self.cursor = 0;
        self.selected.clear();
        self.crafted.clear();
        self.equipped = None;
    }

    fn description(&self) -> &str {
        "Craft natural blocks from gathered elements and place them (press C)."
    }

    fn update(&mut self, _eng: &Engine, ctx: &mut ModContext) {
        self.refresh();
        if ctx.toggle_crafting {
            self.set_open(!self.is_open());
        }
        if self.is_open() {
            self.navigate(ctx);
        } else {
            self.try_place(ctx);
        }
    }

    fn hud(&self, world: &World, (screen_w, screen_h): (i32, i32)) -> Vec<HudElement> {
        let width = PANEL_WIDTH.min((screen_w - PANEL_X * 2).max(1));

        if !self.is_open() {
            // Closed: just a small reminder of what RMB will place.
            let Some(equipped) = self.equipped else {
                return Vec::new();
            };
            let entry = &self.crafted[equipped];
            let kinds = self.stash.borrow().iter().count();
            let ui = self.ui.get();
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
                header: Vec::new(),
                rows: vec![Row::new(Role::Muted, hint)],
            })];
        }

        let stash = self.stash.borrow();
        let elements = world.registry().elements();
        let held: Vec<(ElementId, u32)> = stash.iter().collect();
        let element_count = held.len();
        let total_rows = element_count + 1 + self.crafted.len();
        let selected_names = self
            .selected
            .iter()
            .map(|&id| elements.get(id).name.as_ref())
            .collect::<Vec<_>>()
            .join(" + ");
        let header_rows = 1 + usize::from(!selected_names.is_empty());
        let content_y = PANEL_Y + PANEL_PAD + header_rows as i32 * LINE_HEIGHT + 2;
        let capacity = ((screen_h - BOTTOM_RESERVE - content_y) / LINE_HEIGHT).max(1) as usize;
        let window = visible_window(total_rows, self.cursor, capacity);

        let mut header = vec![Row::new(
            Role::Warning,
            format!("Crafting  {}/{}", self.selected.len(), element_count),
        )];
        if !selected_names.is_empty() {
            header.push(Row::new(Role::Dim, selected_names));
        }

        let rows = window
            .map(|row_index| {
                let active = self.cursor == row_index;
                let cursor = if active { ">" } else { " " };
                if row_index < element_count {
                    let (element, count) = held[row_index];
                    let mark = if self.selected.contains(&element) { "[x]" } else { "[ ]" };
                    Row::new(
                        if active { Role::Accent } else { Role::Muted },
                        format!("{cursor} {mark} {count}x {}", elements.get(element).name),
                    )
                } else if row_index == element_count {
                    let label = if self.selected.is_empty() { "select elements" } else { "craft selected" };
                    Row::new(
                        if self.selected.is_empty() { Role::Disabled } else { Role::Warning },
                        format!("{cursor} [ {label} ]"),
                    )
                } else {
                    let i = row_index - element_count - 1;
                    let entry = &self.crafted[i];
                    let equipped = if self.equipped == Some(i) { "[E]" } else { "   " };
                    let role = if self.equipped == Some(i) {
                        Role::Positive
                    } else if active {
                        Role::Accent
                    } else {
                        Role::Muted
                    };
                    Row::new(role, format!("{cursor} {equipped} {}x {}", entry.count, entry.name))
                }
            })
            .collect();

        vec![HudElement::Panel(Panel {
            at: (PANEL_X, PANEL_Y),
            width,
            header,
            rows,
        })]
    }

    fn close_overlay(&mut self) -> bool {
        if self.is_open() {
            self.set_open(false);
            true
        } else {
            false
        }
    }

    fn save_state(&self, _world: &World) -> Option<String> {
        // Persist by block *name* ("Stone+Iron"), the same portable choice the
        // inventory makes for elements: names survive id reshuffles across
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
        Some(entries.join(","))
    }

    fn load_state(&mut self, data: &str, world: &mut World) {
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
            // A crafted composition can dedup into a BUILTIN block (e.g.
            // Stone+Iron == IronVein), which saves under the builtin's name —
            // resolve block names first, then fall back to the '+'-joined
            // element form.
            if let Some(id) = world.registry().id_by_name(name) {
                self.push_loaded(world, id, count, equip);
                continue;
            }
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mod_with_stash() -> CraftingMod {
        CraftingMod::new(
            Rc::new(RefCell::new(ElementStash::new(10))),
            Rc::new(Cell::new(ItemUiState::default())),
        )
    }

    #[test]
    fn save_load_round_trips_crafted_counts_and_equipped() {
        let mut world = World::new(1);
        let mut crafting = mod_with_stash();
        // Copper+Glass has no builtin block: Stone+Iron would dedup into the
        // builtin IronVein and come back under that name.
        crafting.load_state("Copper+Glass=2,*Stone=1", &mut world);
        assert_eq!(crafting.crafted.len(), 2);
        assert_eq!(crafting.crafted[0].name.as_ref(), "Copper+Glass");
        assert_eq!(crafting.crafted[0].count, 2);
        assert_eq!(crafting.equipped, Some(1));
        assert_eq!(
            crafting.save_state(&world).as_deref(),
            Some("Copper+Glass=2,*Stone=1")
        );
    }

    #[test]
    fn unknown_element_names_skip_the_entry() {
        let mut world = World::new(1);
        let mut crafting = mod_with_stash();
        crafting.load_state("Stone+Unobtainium=5,Iron=3", &mut world);
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
        let mut crafting = mod_with_stash();
        crafting.load_state("Copper+Glass=1", &mut world);
        let id = crafting.crafted[0].id;
        assert_eq!(world.registry().id_by_name("Copper+Glass"), Some(id));
    }

    #[test]
    fn escape_close_consumes_only_an_open_panel() {
        let mut crafting = mod_with_stash();
        assert!(!crafting.close_overlay());
        crafting.set_open(true);
        assert!(crafting.close_overlay());
        assert!(!crafting.is_open());
        assert!(!crafting.close_overlay());
    }
}
