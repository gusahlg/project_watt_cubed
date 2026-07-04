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
use std::cell::RefCell;
use std::rc::Rc;

use voxel_engine::{Color, Engine, Frame, Key, MouseButton, Vec3};

use crate::block::crafting::craft_natural;
use crate::block::registry::BlockId;
use crate::block::{AIR, ElementId};
use crate::console::shadowed;
use crate::interact;
use crate::math::{Aabb, Bounded};
use crate::mods::{ElementStash, Mod, ModContext};
use crate::world::World;

/// At most this many element kinds go into one natural craft.
const MAX_SELECT: usize = 3;
/// How far the player can reach to place a block — matches the break reach.
const PLACE_REACH: f32 = 6.0;
/// Panel geometry: right-aligned like the inventory HUD, starting below the
/// inventory's tallest possible extent (header at y=90 plus 14 capped rows).
const PANEL_X_OFFSET: i32 = 230;
const PANEL_Y: i32 = 440;

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
    /// Whether the panel is on screen (toggled with `C`).
    open: bool,
    /// Cursor over the panel rows: elements, then Craft, then crafted blocks.
    cursor: usize,
    /// Elements marked for the next craft (at most [`MAX_SELECT`]), in pick order.
    selected: Vec<ElementId>,
    /// Every crafted block type, in first-crafted order.
    crafted: Vec<Crafted>,
    /// Index into `crafted` of the block RMB places, if any.
    equipped: Option<usize>,
    /// Cached `(element, name, count)` rows mirroring the stash for drawing
    /// (draw has no world access, so names are resolved when the stash changes).
    element_rows: Vec<(ElementId, Box<str>, u32)>,
    /// The stash revision `element_rows` was built from; `u64::MAX` forces a build.
    seen_rev: u64,
}

impl CraftingMod {
    pub fn new(stash: Rc<RefCell<ElementStash>>) -> Self {
        Self {
            stash,
            open: false,
            cursor: 0,
            selected: Vec::new(),
            crafted: Vec::new(),
            equipped: None,
            element_rows: Vec::new(),
            seen_rev: u64::MAX,
        }
    }

    /// Total rows the cursor can sit on: one per element kind, the Craft row,
    /// one per crafted block type. Always at least 1 (the Craft row).
    fn row_count(&self) -> usize {
        self.element_rows.len() + 1 + self.crafted.len()
    }

    /// Mirror the stash into `element_rows` if it changed, drop selections whose
    /// element ran out, and keep the cursor on a real row.
    fn refresh(&mut self, world: &World) {
        {
            let stash = self.stash.borrow();
            if stash.rev() != self.seen_rev {
                let elements = world.registry().elements();
                self.element_rows = stash
                    .iter()
                    .map(|(e, count)| (e, elements.get(e).name.clone(), count))
                    .collect();
                self.seen_rev = stash.rev();
            }
        }
        let rows = &self.element_rows;
        self.selected.retain(|&e| rows.iter().any(|(re, _, _)| *re == e));
        self.cursor = self.cursor.min(self.row_count() - 1);
    }

    /// Panel-open key handling: move the cursor, toggle selections, craft, equip.
    fn navigate(&mut self, eng: &Engine, ctx: &mut ModContext) {
        // Esc also closes per the design; note the game currently leaves to the
        // menu on Esc before mods run, so in practice C is the close key.
        if eng.is_key_pressed(Key::Escape) {
            self.open = false;
            return;
        }
        if eng.is_key_pressed(Key::Up) || eng.is_key_pressed(Key::K) {
            self.cursor = self.cursor.saturating_sub(1);
        }
        if eng.is_key_pressed(Key::Down) || eng.is_key_pressed(Key::J) {
            self.cursor = (self.cursor + 1).min(self.row_count() - 1);
        }
        if eng.is_key_pressed(Key::Enter) || eng.is_key_pressed(Key::L) {
            self.activate(ctx);
        }
    }

    /// Enter/L on the current row.
    fn activate(&mut self, ctx: &mut ModContext) {
        let elements = self.element_rows.len();
        if self.cursor < elements {
            let element = self.element_rows[self.cursor].0;
            if let Some(at) = self.selected.iter().position(|&e| e == element) {
                self.selected.remove(at);
            } else if self.selected.len() < MAX_SELECT {
                self.selected.push(element);
            }
        } else if self.cursor == elements {
            self.craft(ctx);
        } else {
            self.equipped = Some(self.cursor - elements - 1);
        }
    }

    /// Consume one of each selected element and mint the natural block for the
    /// set. The selection is kept so another Enter crafts another, stock allowing.
    fn craft(&mut self, ctx: &mut ModContext) {
        if self.selected.is_empty() {
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
        self.refresh(ctx.world);
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
    fn try_place(&mut self, eng: &Engine, ctx: &mut ModContext) {
        if !eng.is_mouse_button_pressed(MouseButton::Right) || !ctx.mouse_locked {
            return;
        }
        let Some(equipped) = self.equipped else { return };
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
        // block_at reads out-of-range cells as air but set_block refuses to
        // write them — a placement there would silently eat the block.
        if !(0..crate::world::chunk::CHUNK_HEIGHT as i32).contains(&y) {
            return;
        }
        if ctx.world.block_at(x, y, z) != AIR || cell_aabb(x, y, z).intersects(&ctx.player.aabb()) {
            return;
        }
        ctx.placements.push((x, y, z, self.crafted[equipped].id));
        self.crafted[equipped].count -= 1;
    }
}

/// The unit-cube AABB of a voxel cell.
fn cell_aabb(x: i32, y: i32, z: i32) -> Aabb {
    Aabb::new(
        Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5),
        Vec3::splat(0.5),
    )
}

impl Mod for CraftingMod {
    fn name(&self) -> &str {
        "Crafting"
    }

    fn reset(&mut self) {
        self.open = false;
        self.cursor = 0;
        self.selected.clear();
        self.crafted.clear();
        self.equipped = None;
    }

    fn description(&self) -> &str {
        "Craft natural blocks from gathered elements and place them (press C)."
    }

    fn update(&mut self, eng: &Engine, ctx: &mut ModContext) {
        self.refresh(ctx.world);
        if !ctx.capturing_text && eng.is_key_pressed(Key::C) {
            self.open = !self.open;
        }
        if self.open {
            self.navigate(eng, ctx);
        } else {
            self.try_place(eng, ctx);
        }
    }

    fn draw(&mut self, f: &mut Frame, screen_w: i32, _screen_h: i32) {
        let fs = 18;
        let line_h = fs + 4;
        let x = screen_w - PANEL_X_OFFSET;
        let mut y = PANEL_Y;

        if !self.open {
            // Closed: just a small reminder of what RMB will place.
            if let Some(equipped) = self.equipped {
                let entry = &self.crafted[equipped];
                let hint = format!("RMB place {} ({})", entry.name, entry.count);
                shadowed(f, &hint, x, y, 16, Color::RAYWHITE);
            }
            return;
        }

        shadowed(f, "Crafting", x, y, fs, Color::GOLD);
        y += line_h + 2;

        if self.element_rows.is_empty() {
            shadowed(f, "  (no elements) break blocks", x, y, fs, Color::RAYWHITE);
            y += line_h;
        }
        for (i, (element, name, count)) in self.element_rows.iter().enumerate() {
            let cursor = if self.cursor == i { ">" } else { " " };
            let mark = if self.selected.contains(element) { "[x]" } else { "[ ]" };
            let row = format!("{cursor} {mark} {count}x {name}");
            shadowed(f, &row, x, y, fs, Color::RAYWHITE);
            y += line_h;
        }

        let craft_row = self.element_rows.len();
        let cursor = if self.cursor == craft_row { ">" } else { " " };
        let row = format!("{cursor} [Craft: {}/{MAX_SELECT} picked]", self.selected.len());
        shadowed(f, &row, x, y, fs, Color::GOLD);
        y += line_h;

        for (i, entry) in self.crafted.iter().enumerate() {
            let cursor = if self.cursor == craft_row + 1 + i { ">" } else { " " };
            let equipped = if self.equipped == Some(i) { "*" } else { " " };
            let row = format!("{cursor} {equipped}{}x {}", entry.count, entry.name);
            shadowed(f, &row, x, y, fs, Color::RAYWHITE);
            y += line_h;
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
            let Some((name, count)) = entry.rsplit_once('=') else { continue };
            let Ok(count) = count.parse::<u32>() else { continue };
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mod_with_stash() -> CraftingMod {
        CraftingMod::new(Rc::new(RefCell::new(ElementStash::new(10))))
    }

    #[test]
    fn save_load_round_trips_crafted_counts_and_equipped() {
        let mut world = World::new(1);
        let mut crafting = mod_with_stash();
        crafting.load_state("Stone+Iron=2,*Stone=1", &mut world);
        assert_eq!(crafting.crafted.len(), 2);
        assert_eq!(crafting.crafted[0].name.as_ref(), "Stone+Iron");
        assert_eq!(crafting.crafted[0].count, 2);
        assert_eq!(crafting.equipped, Some(1));
        assert_eq!(
            crafting.save_state(&world).as_deref(),
            Some("Stone+Iron=2,*Stone=1")
        );
    }

    #[test]
    fn unknown_element_names_skip_the_entry() {
        let mut world = World::new(1);
        let mut crafting = mod_with_stash();
        crafting.load_state("Stone+Unobtainium=5,Iron=3", &mut world);
        assert_eq!(crafting.crafted.len(), 1, "unknown-element entry is skipped");
        assert_eq!(crafting.crafted[0].name.as_ref(), "Iron");
        assert_eq!(crafting.crafted[0].count, 3);
    }

    #[test]
    fn loaded_names_recraft_to_registry_ids() {
        let mut world = World::new(1);
        let mut crafting = mod_with_stash();
        crafting.load_state("Stone+Iron=1", &mut world);
        let id = crafting.crafted[0].id;
        assert_eq!(world.registry().id_by_name("Stone+Iron"), Some(id));
    }
}
