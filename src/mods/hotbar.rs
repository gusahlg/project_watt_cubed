//! The hotbar: nine slots of held materials plus the bare hand, along the bottom of the screen.
//!
//! A slot names a stack in the core stash. The selected slot is what the player *holds*: a left
//! click with it runs the law between it and the targeted block (a tool is just a block in hand),
//! a right click places one unit of it. The bare hand (slot 0) breaks blocks. Keys 1-9 select a
//! slot (pressing the selected one again returns to the hand), 0 selects the hand, the wheel
//! cycles. Newly gathered materials drop into the first empty slot; the inventory (press I)
//! equips any held material into any slot. When a tool reaction changes the held unit, the slot
//! follows it to its new configuration.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use voxel_engine::Color;

use crate::block::{BlockId, AIR};
use crate::derived::Memo;
use crate::mods::{ItemUiState, Mod, ModContext};
use crate::player::Player;
use crate::ui::{Anchor, HudElement, Role};
use crate::world::World;

/// Material slots (keys 1-9); slot 0 is the bare hand.
pub const SLOTS: usize = 9;

/// The hotbar's state, shared with the inventory (which equips into it).
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct HotbarState {
    /// `slots[i]` is key `i + 1`.
    pub slots: [Option<BlockId>; SLOTS],
    /// 0 = the bare hand, 1..=9 = a slot.
    pub selected: usize,
}

impl HotbarState {
    /// The material in the selected slot, if any.
    pub fn selected_id(&self) -> Option<BlockId> {
        if self.selected == 0 { None } else { self.slots[self.selected - 1] }
    }

    /// Put `id` into slot `key` (1..=9), clearing it from any other slot.
    pub fn equip(&mut self, key: usize, id: BlockId) {
        if !(1..=SLOTS).contains(&key) {
            return;
        }
        for s in self.slots.iter_mut() {
            if *s == Some(id) {
                *s = None;
            }
        }
        self.slots[key - 1] = Some(id);
    }

    /// The key (1..=9) a material sits in, if any.
    pub fn key_of(&self, id: BlockId) -> Option<usize> {
        self.slots.iter().position(|&s| s == Some(id)).map(|i| i + 1)
    }

    /// Drop `id` into the first empty slot if it is not on the bar yet.
    fn adopt(&mut self, id: BlockId) {
        if id == AIR || self.key_of(id).is_some() {
            return;
        }
        if let Some(i) = self.slots.iter().position(|s| s.is_none()) {
            self.slots[i] = Some(id);
        }
    }

    /// Clear slots whose stack is gone.
    fn prune(&mut self, player: &Player) {
        for s in self.slots.iter_mut() {
            if s.is_some_and(|id| player.stash.count(id) == 0) {
                *s = None;
            }
        }
    }
}

/// Slot edge and pitch on screen, in pixels.
const SLOT: i32 = 46;
const PITCH: i32 = 52;
const BOTTOM: i32 = -14;
const FRAME: Color = Color::new(10, 12, 16, 210);
const SELECTED: Color = Color::new(250, 214, 92, 255);

/// What the hotbar HUD shows: stash revision, slots, screen size, registry names revision.
type HudKey = (u64, HotbarState, i32, i32, u64);

pub struct HotbarMod {
    ui: Rc<Cell<ItemUiState>>,
    bar: Rc<Cell<HotbarState>>,
    hud_cache: RefCell<Memo<HudKey, Vec<HudElement>>>,
}

impl HotbarMod {
    pub(crate) fn new(ui: Rc<Cell<ItemUiState>>, bar: Rc<Cell<HotbarState>>) -> Self {
        Self { ui, bar, hud_cache: RefCell::new(Memo::new()) }
    }

    fn with(&self, f: impl FnOnce(&mut HotbarState)) {
        let mut s = self.bar.get();
        f(&mut s);
        self.bar.set(s);
    }
}

/// Pixel geometry of the bar at one screen size (designed at 720 px tall, scaled up from there).
struct Layout {
    slot: i32,
    pitch: i32,
    bottom: i32,
    k: f32,
}

impl Layout {
    fn of(screen_h: i32) -> Self {
        let k = (screen_h as f32 / 720.0).clamp(1.0, 4.0);
        let px = |v: i32| (v as f32 * k).round() as i32;
        Layout { slot: px(SLOT), pitch: px(PITCH), bottom: px(BOTTOM), k }
    }

    fn px(&self, v: i32) -> i32 {
        (v as f32 * self.k).round() as i32
    }

    /// Horizontal offset of slot `i` (0 = hand … 9) from the screen centre.
    fn slot_x(&self, i: usize) -> i32 {
        (2 * i as i32 - SLOTS as i32) * self.pitch / 2
    }
}

fn paint(state: &HotbarState, player: &Player, world: &World, screen_h: i32) -> Vec<HudElement> {
    let reg = world.registry();
    let l = Layout::of(screen_h);
    let (slot, bottom) = (l.slot, l.bottom);
    let mut out = Vec::with_capacity(5 * (SLOTS + 1) + 2);
    for i in 0..=SLOTS {
        let x = l.slot_x(i);
        if i == state.selected {
            let rim = l.px(3);
            out.push(HudElement::Rect { at: Anchor::Bottom, off: (x, bottom + rim), size: (slot + 2 * rim, slot + 2 * rim), color: SELECTED });
        }
        out.push(HudElement::Rect { at: Anchor::Bottom, off: (x, bottom), size: (slot, slot), color: FRAME });
        let key = i.to_string();
        out.push(HudElement::Label {
            at: Anchor::Bottom,
            off: (x - slot / 2 + l.px(7), bottom - slot + l.px(12)),
            base_fs: 12,
            role: Role::Dim,
            text: key.into(),
        });
        let id = if i == 0 { None } else { state.slots[i - 1] };
        match id {
            Some(id) => {
                // The material's colour with its accent: the HUD cannot draw the texture itself.
                let c = reg.color(id);
                let v = reg.visual(id);
                out.push(HudElement::Rect { at: Anchor::Bottom, off: (x, bottom - l.px(7)), size: (slot - l.px(12), slot - l.px(20)), color: c });
                out.push(HudElement::Rect {
                    at: Anchor::Bottom,
                    off: (x + l.px(6), bottom - l.px(10)),
                    size: (slot - l.px(26), l.px(6)),
                    color: Color::new(v.rgb2[0], v.rgb2[1], v.rgb2[2], 255),
                });
                let n = player.stash.count(id);
                out.push(HudElement::Label {
                    at: Anchor::Bottom,
                    off: (x + slot / 2 - l.px(10), bottom - l.px(3)),
                    base_fs: 14,
                    role: Role::Primary,
                    text: n.to_string().into(),
                });
            }
            None if i == 0 => {
                out.push(HudElement::Label { at: Anchor::Bottom, off: (x, bottom - l.px(14)), base_fs: 12, role: Role::Muted, text: "hand".into() });
            }
            None => {}
        }
    }
    // What is held, by name: the naming mod's tool name, or "Bare hand".
    let title = match state.selected_id() {
        Some(id) => {
            let n = reg.configuration(id).len();
            format!("{}  ({} element{})", reg.tool_name(id), n, if n == 1 { "" } else { "s" })
        }
        None => "Bare hand".to_string(),
    };
    out.push(HudElement::Label { at: Anchor::Bottom, off: (0, bottom - slot - l.px(14)), base_fs: 18, role: Role::Warning, text: title.into() });
    out
}

impl Mod for HotbarMod {
    fn name(&self) -> &str {
        "Hotbar"
    }

    fn id(&self) -> &'static str {
        "hotbar"
    }

    fn description(&self) -> &str {
        "Nine slots of held materials: 1-9 / wheel to pick, left click uses it as a tool, right click places it. 0 = bare hand."
    }

    fn group(&self) -> &'static str {
        crate::mods::ESSENTIALS
    }

    fn reset(&mut self) {
        self.bar.set(HotbarState::default());
    }

    fn update(&mut self, ctx: &mut ModContext) {
        let player = &*ctx.player;
        self.with(|s| s.prune(player));
        // While the inventory is open it owns the number keys and the wheel (to equip).
        if !self.ui.get().inventory_visible {
            if let Some(k) = ctx.hotbar_key {
                self.with(|s| {
                    let k = k as usize;
                    s.selected = if k != 0 && s.selected == k { 0 } else { k.min(SLOTS) };
                });
            }
            if ctx.hotbar_cycle != 0 {
                self.with(|s| {
                    let n = SLOTS as i32 + 1;
                    s.selected = (s.selected as i32 + ctx.hotbar_cycle as i32).rem_euclid(n) as usize;
                });
            }
        }
        if ctx.place {
            let state = self.bar.get();
            let (Some(id), Some((x, y, z))) = (state.selected_id(), ctx.place_target) else { return };
            if ctx.world.block_at(x, y, z) != AIR {
                return;
            }
            // The core re-checks the cell and the player's body, refunding the unit if it refuses.
            if ctx.player.stash.consume(id, 1) {
                ctx.placements.push((x, y, z, id));
            }
        }
    }

    fn on_block_break(&mut self, id: BlockId, _world: &World, overflow: bool) {
        if !overflow {
            self.with(|s| s.adopt(id));
        }
    }

    fn held(&self, player: &Player) -> Option<BlockId> {
        self.bar.get().selected_id().filter(|&id| player.stash.count(id) > 0)
    }

    fn on_tool_changed(&mut self, old: BlockId, new: BlockId) {
        self.with(|s| {
            if s.selected_id() == Some(old) {
                let k = s.selected;
                if new == AIR {
                    s.slots[k - 1] = None;
                } else {
                    s.equip(k, new);
                }
            }
        });
    }

    fn hud(&self, world: &World, player: &Player, screen: (i32, i32), out: &mut Vec<HudElement>) {
        let state = self.bar.get();
        // Names arrive a frame after a configuration is interned: key on the table size too.
        let key = (player.stash.rev(), state, screen.0, screen.1, world.registry().block_count() as u64);
        let mut cache = self.hud_cache.borrow_mut();
        out.extend(cache.get_or(key, || paint(&state, player, world, screen.1)).iter().cloned());
    }

    fn save_state(&self, world: &World) -> Option<(u16, String)> {
        let s = self.bar.get();
        let mut text = format!("sel={}", s.selected);
        for (i, slot) in s.slots.iter().enumerate() {
            if let Some(id) = slot {
                text.push_str(&format!(";{}={}", i + 1, world.registry().spec(*id)));
            }
        }
        Some((1, text))
    }

    fn load_state(&mut self, _version: u16, data: &str, world: &mut World) -> u32 {
        let mut s = HotbarState::default();
        for part in data.split(';') {
            let Some((k, v)) = part.split_once('=') else { continue };
            if k == "sel" {
                s.selected = v.parse::<usize>().unwrap_or(0).min(SLOTS);
            } else if let Ok(key) = k.parse::<usize>()
                && let Some(id) = world.registry_mut().parse_spec(v)
            {
                s.equip(key, id);
            }
        }
        self.bar.set(s);
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mods::ModContext;
    use crate::render_config::RenderConfig;
    use material::{Configuration, Element};
    use voxel_engine::DVec3;

    fn setup() -> (HotbarMod, Rc<Cell<ItemUiState>>, World, Player, BlockId, BlockId) {
        let ui = Rc::new(Cell::new(ItemUiState::default()));
        let bar = Rc::new(Cell::new(HotbarState::default()));
        let m = HotbarMod::new(ui.clone(), bar);
        let mut world = World::with_kind(5, RenderConfig::default(), crate::world::generation::WorldgenKind::Flat, true);
        let a = world.registry_mut().intern(&Configuration::single(Element::new([1, 2, 3, 4]))).unwrap();
        let b = world.registry_mut().intern(&Configuration::single(Element::new([9, 2, 3, 4]))).unwrap();
        let player = Player::new(DVec3::new(0.5, 70.0, 0.5));
        (m, ui, world, player, a, b)
    }

    fn ctx<'a>(player: &'a mut Player, world: &'a mut World) -> ModContext<'a> {
        ModContext {
            player,
            world,
            screen_w: 800,
            screen_h: 600,
            place: false,
            place_target: None,
            toggle_inventory: false,
            nav_up: false,
            nav_down: false,
            nav_left: false,
            nav_right: false,
            nav_tab: false,
            nav_confirm: false,
            hotbar_key: None,
            hotbar_cycle: 0,
            networked: false,
            placements: Vec::new(),
        }
    }

    #[test]
    fn gathered_materials_fill_empty_slots_and_keys_select_them() {
        let (mut m, _ui, mut world, mut player, a, b) = setup();
        player.stash.add(a, 3);
        player.stash.add(b, 1);
        m.on_block_break(a, &world, false);
        m.on_block_break(b, &world, false);
        m.on_block_break(a, &world, false);
        assert_eq!(m.bar.get().slots[..2], [Some(a), Some(b)]);
        assert_eq!(m.held(&player), None, "the hand is selected until a key says otherwise");
        let mut c = ctx(&mut player, &mut world);
        c.hotbar_key = Some(2);
        m.update(&mut c);
        assert_eq!(m.held(&player), Some(b));
        let mut c = ctx(&mut player, &mut world);
        c.hotbar_key = Some(2);
        m.update(&mut c);
        assert_eq!(m.held(&player), None, "pressing the selected slot again returns to the hand");
        let mut c = ctx(&mut player, &mut world);
        c.hotbar_cycle = -1;
        m.update(&mut c);
        assert_eq!(m.bar.get().selected, 9, "the wheel wraps from the hand to slot 9");
    }

    #[test]
    fn a_changed_tool_moves_its_slot_and_a_spent_one_clears_it() {
        let (mut m, _ui, _world, mut player, a, b) = setup();
        player.stash.add(a, 1);
        m.with(|s| {
            s.equip(1, a);
            s.selected = 1;
        });
        m.on_tool_changed(a, b);
        assert_eq!(m.bar.get().selected_id(), Some(b));
        m.on_tool_changed(b, AIR);
        assert_eq!(m.bar.get().selected_id(), None);
    }

    #[test]
    fn placing_spends_one_unit_of_the_selected_slot() {
        let (mut m, _ui, mut world, mut player, a, _) = setup();
        player.stash.add(a, 2);
        m.with(|s| {
            s.equip(3, a);
            s.selected = 3;
        });
        let mut c = ctx(&mut player, &mut world);
        c.place = true;
        c.place_target = Some((10, 64, 10));
        m.update(&mut c);
        assert_eq!(c.placements, vec![(10, 64, 10, a)]);
        assert_eq!(player.stash.count(a), 1);
    }

    #[test]
    fn hotbar_state_round_trips_through_save_state() {
        let (mut m, _ui, mut world, _player, a, b) = setup();
        m.with(|s| {
            s.equip(1, a);
            s.equip(5, b);
            s.selected = 5;
        });
        let (v, text) = m.save_state(&world).unwrap();
        let saved = m.bar.get();
        m.reset();
        m.load_state(v, &text, &mut world);
        assert_eq!(m.bar.get(), saved);
    }
}
