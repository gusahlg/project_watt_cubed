//! The default crafting mod: take configurations from the stash into a pouch,
//! apply physical events at a workbench, and place the result.
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

use material::EventKind;
use voxel_engine::DVec3;

use crate::block::registry::{interact_repeat, BlockId};
use crate::block::AIR;
use crate::derived::Memo;
use crate::math::{Aabb, Bounded};
use crate::mods::inventory::{InventoryMod, PANEL_X, PANEL_Y};
use crate::mods::{CraftRequest, ItemUiState, Mod, ModContext};
use crate::player::Player;
use crate::stash::{ElementStash, START_CAPACITY};
use crate::ui::{visible_window, HudElement, Line, Panel, Role, Row};
use crate::world::World;

const PANEL_WIDTH: i32 = 400;
const PANEL_PAD: i32 = 8;
const FONT_SIZE: i32 = 18;
const LINE_HEIGHT: i32 = FONT_SIZE + 4;
const BOTTOM_RESERVE: i32 = 190;
const REPEAT_MIN: u8 = *crate::net::protocol::WORKBENCH_REPEAT.start();
const REPEAT_MAX: u8 = *crate::net::protocol::WORKBENCH_REPEAT.end();

const EVENTS: [(EventKind, &str); 3] = [
    (EventKind::NewContact, "touch"),
    (EventKind::Collision, "strike"),
    (EventKind::Moved, "shake"),
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Workbench,
    Journal,
}

/// A recorded apply that produced a new configuration id for this player.
/// Never consulted by the simulation.
struct Procedure {
    origin_spec: Box<str>,
    target_spec: Box<str>,
    event: EventKind,
    repeat: u8,
    result_spec: Box<str>,
    name: Option<Box<str>>,
}

/// The crafting panel, the pouch, the workbench, and right-click placement.
pub struct CraftingMod {
    /// Shared with inventory so this expanded panel replaces its compact view.
    ui: Rc<Cell<ItemUiState>>,
    /// Cursor over the panel rows.
    cursor: usize,
    tab: Tab,
    /// Pouch holdings, in first-taken order. Capacity matches the stash.
    pouch: ElementStash,
    /// Index into the pouch of the block RMB places, if any — stored as id.
    equipped: Option<BlockId>,
    origin: Option<BlockId>,
    target: Option<BlockId>,
    event: usize,
    repeat: u8,
    journal: Vec<Procedure>,
    /// Held block ids mirrored from the stash only when its revision changes.
    held: Vec<BlockId>,
    /// Units destroyed because a refund (a rejected placement) found the pouch full.
    lost: u32,
    seen_stash_rev: u64,
    hud_gen: Cell<u64>,
    hud_cache: RefCell<Memo<(u64, u64, i32, i32, bool), Vec<HudElement>>>,
}

impl CraftingMod {
    pub(crate) fn new(ui: Rc<Cell<ItemUiState>>) -> Self {
        Self {
            ui,
            cursor: 0,
            tab: Tab::Workbench,
            pouch: ElementStash::new(START_CAPACITY),
            equipped: None,
            origin: None,
            target: None,
            event: 0,
            repeat: REPEAT_MIN,
            journal: Vec::new(),
            held: Vec::new(),
            lost: 0,
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

    fn event_kind(&self) -> EventKind {
        EVENTS[self.event].0
    }

    fn row_count(&self) -> usize {
        match self.tab {
            Tab::Workbench => 1 + self.held.len() + self.pouch.iter().count() + 5,
            Tab::Journal => 1 + self.journal.len().max(1),
        }
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
        let tab = self.tab;
        let event = self.event;
        let repeat = self.repeat;
        if ctx.nav_up {
            self.cursor = self.cursor.saturating_sub(1);
        }
        if ctx.nav_down && self.row_count() > 0 {
            self.cursor = (self.cursor + 1).min(self.row_count() - 1);
        }
        if ctx.nav_tab {
            self.toggle_tab();
        }
        if ctx.nav_left {
            self.nudge(-1);
        }
        if ctx.nav_right {
            self.nudge(1);
        }
        if ctx.nav_confirm {
            self.activate(ctx);
        }
        if self.cursor != cursor
            || self.tab != tab
            || self.event != event
            || self.repeat != repeat
        {
            self.bump_hud();
        }
    }

    fn toggle_tab(&mut self) {
        self.tab = match self.tab {
            Tab::Workbench => Tab::Journal,
            Tab::Journal => Tab::Workbench,
        };
        self.cursor = 0;
    }

    fn nudge(&mut self, delta: i32) {
        if self.tab != Tab::Workbench {
            if self.cursor == 0 {
                self.toggle_tab();
            }
            return;
        }
        let layout = self.workbench_layout();
        if self.cursor == layout.tab {
            self.toggle_tab();
        } else if self.cursor == layout.event {
            let n = EVENTS.len() as i32;
            self.event = (self.event as i32 + delta).rem_euclid(n) as usize;
        } else if self.cursor == layout.repeat {
            let v = self.repeat as i32 + delta;
            self.repeat = v.clamp(REPEAT_MIN as i32, REPEAT_MAX as i32) as u8;
        }
    }

    fn workbench_layout(&self) -> WorkbenchLayout {
        let held = self.held.len();
        let pouch = self.pouch.iter().count();
        let stash_start = 1;
        let pouch_start = stash_start + held;
        let origin = pouch_start + pouch;
        WorkbenchLayout {
            tab: 0,
            stash_start,
            pouch_start,
            origin,
            target: origin + 1,
            event: origin + 2,
            repeat: origin + 3,
            apply: origin + 4,
        }
    }

    /// Confirm on a stash row takes one unit into the pouch; on a pouch row, equips
    /// it and fills the next workbench slot; on Apply, runs the law.
    fn activate(&mut self, ctx: &mut ModContext) {
        if self.tab == Tab::Journal {
            if self.cursor == 0 {
                self.toggle_tab();
                self.bump_hud();
            }
            return;
        }
        let layout = self.workbench_layout();
        if self.cursor == layout.tab {
            self.toggle_tab();
            self.bump_hud();
            return;
        }
        if self.cursor >= layout.stash_start && self.cursor < layout.pouch_start {
            let id = self.held[self.cursor - layout.stash_start];
            if !ctx.player.stash.consume(id, 1) {
                self.refresh(&ctx.player.stash);
                return;
            }
            if !self.pouch.add(id, 1) {
                ctx.player.stash.add(id, 1);
                self.refresh(&ctx.player.stash);
                return;
            }
            self.bump_hud();
            self.refresh(&ctx.player.stash);
            return;
        }
        if self.cursor >= layout.pouch_start && self.cursor < layout.origin {
            let i = self.cursor - layout.pouch_start;
            if let Some((id, _)) = self.pouch.iter().nth(i) {
                self.equipped = Some(id);
                if self.origin.is_none() {
                    self.origin = Some(id);
                } else {
                    self.target = Some(id);
                }
                self.bump_hud();
            }
            return;
        }
        if self.cursor == layout.origin {
            self.origin = None;
            self.bump_hud();
            return;
        }
        if self.cursor == layout.target {
            self.target = None;
            self.bump_hud();
            return;
        }
        if self.cursor == layout.event {
            self.event = (self.event + 1) % EVENTS.len();
            self.bump_hud();
            return;
        }
        if self.cursor == layout.repeat {
            self.repeat = if self.repeat == REPEAT_MAX {
                REPEAT_MIN
            } else {
                self.repeat + 1
            };
            self.bump_hud();
            return;
        }
        if self.cursor == layout.apply {
            self.apply(ctx);
        }
    }

    fn apply(&mut self, ctx: &mut ModContext) {
        let Some(oid) = self.origin else { return };
        let Some(tid) = self.target else { return };
        // Both slots name held units: the target is consumed, the origin must be present to act.
        if self.pouch.count(oid) < 1 || self.pouch.count(tid) < 1 {
            self.prune_slots();
            return;
        }
        let origin_spec: Arc<str> = ctx.world.registry().spec(oid).into();
        let target_spec: Arc<str> = ctx.world.registry().spec(tid).into();
        let event = self.event_kind() as u8;
        let repeat = self.repeat;
        if ctx.networked {
            ctx.crafts.push(CraftRequest {
                origin_spec,
                target_spec,
                event,
                repeat,
            });
            return;
        }
        self.commit(
            ctx.world,
            &origin_spec,
            &target_spec,
            self.event_kind(),
            repeat,
            None,
        );
    }

    fn commit(
        &mut self,
        world: &mut World,
        origin_spec: &str,
        target_spec: &str,
        event: EventKind,
        repeat: u8,
        result_spec: Option<&str>,
    ) {
        // Held configurations are always known to the registry, so a lookup suffices; an unknown
        // spec cannot be held and is refused without interning it.
        let Some(oid) = world.registry().lookup_spec(origin_spec) else {
            return;
        };
        let Some(tid) = world.registry().lookup_spec(target_spec) else {
            return;
        };
        if self.pouch.count(oid) < 1 || self.pouch.count(tid) < 1 {
            self.prune_slots();
            return;
        }
        let rid = if let Some(spec) = result_spec {
            let Some(id) = world.registry_mut().parse_spec(spec) else {
                return;
            };
            id
        } else {
            let origin = world.registry().configuration(oid).clone();
            let target = world.registry().configuration(tid).clone();
            let law = *world.registry().law();
            let result = interact_repeat(&law, &origin, &target, event, repeat);
            let Some(id) = world.registry_mut().intern(&result) else {
                return;
            };
            id
        };
        // A procedure is knowledge this player discovered: new when this journal has no entry for
        // it and it changed something (a fixed point teaches nothing).
        let was_new = rid != tid
            && !self.journal.iter().any(|p| {
                &*p.origin_spec == origin_spec
                    && &*p.target_spec == target_spec
                    && p.event == event
                    && p.repeat == repeat
            });
        self.finish_commit(
            world,
            origin_spec,
            target_spec,
            event,
            repeat,
            tid,
            rid,
            was_new,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_commit(
        &mut self,
        world: &World,
        origin_spec: &str,
        target_spec: &str,
        event: EventKind,
        repeat: u8,
        tid: BlockId,
        rid: BlockId,
        was_new: bool,
    ) {
        if !self.pouch.consume(tid, 1) {
            return;
        }
        if !self.pouch.add(rid, 1) {
            self.pouch.add(tid, 1);
            return;
        }
        self.prune_slots();
        if was_new {
            self.journal.push(Procedure {
                origin_spec: origin_spec.into(),
                target_spec: target_spec.into(),
                event,
                repeat,
                result_spec: world.registry().spec(rid).into(),
                name: None,
            });
        }
        self.bump_hud();
    }

    fn try_place(&mut self, ctx: &mut ModContext) {
        if !ctx.place {
            return;
        }
        let Some(equipped) = self.equipped else {
            return;
        };
        if self.pouch.count(equipped) == 0 {
            return;
        }
        let Some((x, y, z)) = ctx.place_target else {
            return;
        };
        if ctx.world.block_at(x, y, z) != AIR || cell_aabb(x, y, z).intersects(&ctx.player.aabb()) {
            return;
        }
        ctx.placements.push((x, y, z, equipped));
        self.pouch.consume(equipped, 1);
        self.prune_slots();
        self.bump_hud();
    }

    /// Slots (equipped, origin, target) only ever name held units: clear any whose pouch row ran out.
    fn prune_slots(&mut self) {
        let pouch = &self.pouch;
        for slot in [&mut self.equipped, &mut self.origin, &mut self.target] {
            if slot.is_some_and(|id| pouch.count(id) == 0) {
                *slot = None;
            }
        }
    }

    fn push_loaded(&mut self, _world: &World, id: BlockId, count: u32, equip: bool) {
        self.pouch.add(id, count);
        if equip {
            self.equipped = Some(id);
        }
    }

    fn name_procedure(&mut self, args: &[&str]) -> Vec<Line> {
        let Some(n) = args.first().and_then(|s| s.parse::<usize>().ok()) else {
            return vec![Line::of(Role::Danger, "usage: name <n> <text>")];
        };
        if n == 0 || n > self.journal.len() {
            return vec![Line::of(Role::Danger, "name: no such procedure")];
        }
        let text = args[1..].join(" ");
        if text.is_empty() {
            return vec![Line::of(Role::Danger, "usage: name <n> <text>")];
        }
        self.journal[n - 1].name = Some(text.into());
        self.bump_hud();
        vec![Line::of(Role::Positive, format!("procedure {n} named"))]
    }
}

struct WorkbenchLayout {
    tab: usize,
    stash_start: usize,
    pouch_start: usize,
    origin: usize,
    target: usize,
    event: usize,
    repeat: usize,
    apply: usize,
}

fn cell_aabb(x: i32, y: i32, z: i32) -> Aabb {
    Aabb::new(
        DVec3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5),
        DVec3::splat(0.5),
    )
}

fn event_name(kind: EventKind) -> &'static str {
    EVENTS
        .iter()
        .find(|(k, _)| *k == kind)
        .map(|(_, n)| *n)
        .unwrap_or("?")
}

fn slot_name(world: &World, id: Option<BlockId>) -> String {
    match id {
        Some(id) => world.registry().display_name(id),
        None => "(empty)".into(),
    }
}

fn paint_crafting(
    stash: &ElementStash,
    ui: ItemUiState,
    cursor: usize,
    tab: Tab,
    pouch: &ElementStash,
    equipped: Option<BlockId>,
    origin: Option<BlockId>,
    target: Option<BlockId>,
    event: usize,
    repeat: u8,
    journal: &[Procedure],
    world: &World,
    screen_w: i32,
    screen_h: i32,
) -> Vec<HudElement> {
    let width = PANEL_WIDTH.min((screen_w - PANEL_X * 2).max(1));

    if !ui.crafting_open {
        let Some(equipped) = equipped else {
            return Vec::new();
        };
        if pouch.count(equipped) == 0 {
            return Vec::new();
        }
        let kinds = stash.iter().count();
        let y = if ui.inventory_visible {
            InventoryMod::panel_bottom(screen_h, kinds) + 6
        } else {
            PANEL_Y
        };
        let name = world.registry().display_name(equipped);
        let hint = format!("Equipped: {name} x{}", pouch.count(equipped));
        let hint_w = (hint.chars().count() as i32 * FONT_SIZE + PANEL_PAD * 2).min(width);
        return vec![HudElement::Panel(Panel {
            at: (PANEL_X, y),
            width: hint_w,
            header: Vec::new().into(),
            rows: vec![Row::new(Role::Muted, hint)
                .with_swatch(world.registry().color(equipped))]
            .into(),
        })];
    }

    let header = vec![Row::new(
        Role::Warning,
        match tab {
            Tab::Workbench => "Workbench  take / slot / apply",
            Tab::Journal => "Journal  /name <n> <text>",
        },
    )];

    let rows = match tab {
        Tab::Workbench => paint_workbench(
            stash, cursor, pouch, equipped, origin, target, event, repeat, world, screen_h,
        ),
        Tab::Journal => paint_journal(cursor, journal, world, screen_h),
    };

    vec![HudElement::Panel(Panel {
        at: (PANEL_X, PANEL_Y),
        width,
        header: header.into(),
        rows: rows.into(),
    })]
}

fn paint_workbench(
    stash: &ElementStash,
    cursor: usize,
    pouch: &ElementStash,
    equipped: Option<BlockId>,
    origin: Option<BlockId>,
    target: Option<BlockId>,
    event: usize,
    repeat: u8,
    world: &World,
    screen_h: i32,
) -> Vec<Row> {
    let held: Vec<(BlockId, u32)> = stash.iter().collect();
    let pouch_entries: Vec<(BlockId, u32)> = pouch.iter().collect();
    let held_n = held.len();
    let pouch_n = pouch_entries.len();
    let total = 1 + held_n + pouch_n + 5;
    let content_y = PANEL_Y + PANEL_PAD + LINE_HEIGHT + 2;
    let capacity = ((screen_h - BOTTOM_RESERVE - content_y) / LINE_HEIGHT).max(1) as usize;
    let window = visible_window(total, cursor, capacity);

    window
        .map(|row_index| {
            let active = cursor == row_index;
            let mark = if active { ">" } else { " " };
            let role = if active { Role::Accent } else { Role::Muted };
            if row_index == 0 {
                return Row::new(role, format!("{mark} [Workbench]  Journal"));
            }
            let i = row_index - 1;
            if i < held_n {
                let (id, count) = held[i];
                let name = world.registry().display_name(id);
                return Row::new(role, format!("{mark} {count}x {name}"))
                    .with_swatch(world.registry().color(id));
            }
            let i = i - held_n;
            if i < pouch_n {
                let (id, count) = pouch_entries[i];
                let name = world.registry().display_name(id);
                let eq = if equipped == Some(id) { "[E]" } else { "   " };
                let role = if equipped == Some(id) {
                    Role::Positive
                } else {
                    role
                };
                return Row::new(role, format!("{mark} {eq} {count}x {name}"))
                    .with_swatch(world.registry().color(id));
            }
            let i = i - pouch_n;
            match i {
                0 => {
                    let name = slot_name(world, origin);
                    let mut row = Row::new(role, format!("{mark} Origin: {name}"));
                    if let Some(id) = origin {
                        row = row.with_swatch(world.registry().color(id));
                    }
                    row
                }
                1 => {
                    let name = slot_name(world, target);
                    let mut row = Row::new(role, format!("{mark} Target: {name}"));
                    if let Some(id) = target {
                        row = row.with_swatch(world.registry().color(id));
                    }
                    row
                }
                2 => Row::new(role, format!("{mark} Event: {}", EVENTS[event].1)),
                3 => Row::new(role, format!("{mark} Repeat: {repeat}")),
                _ => Row::new(
                    if active { Role::Positive } else { Role::Muted },
                    format!("{mark} Apply"),
                ),
            }
        })
        .collect()
}

fn paint_journal(cursor: usize, journal: &[Procedure], world: &World, screen_h: i32) -> Vec<Row> {
    let body = journal.len().max(1);
    let total = 1 + body;
    let content_y = PANEL_Y + PANEL_PAD + LINE_HEIGHT + 2;
    let capacity = ((screen_h - BOTTOM_RESERVE - content_y) / LINE_HEIGHT).max(1) as usize;
    let window = visible_window(total, cursor, capacity);
    window
        .map(|row_index| {
            let active = cursor == row_index;
            let mark = if active { ">" } else { " " };
            let role = if active { Role::Accent } else { Role::Muted };
            if row_index == 0 {
                return Row::new(role, format!("{mark} Workbench  [Journal]"));
            }
            if journal.is_empty() {
                return Row::new(Role::Dim, format!("{mark} (empty)"));
            }
            let i = row_index - 1;
            let p = &journal[i];
            let title = p.name.as_deref().unwrap_or("unnamed");
            let origin = spec_name(world, &p.origin_spec);
            let target = spec_name(world, &p.target_spec);
            let result = spec_name(world, &p.result_spec);
            Row::new(
                role,
                format!(
                    "{mark} {}. {title}  {origin} + {target} {} x{} → {result}",
                    i + 1,
                    event_name(p.event),
                    p.repeat
                ),
            )
        })
        .collect()
}

fn spec_name(world: &World, spec: &str) -> String {
    world
        .registry()
        .lookup_spec(spec)
        .map(|id| world.registry().display_name(id))
        .unwrap_or_else(|| "unknown material".into())
}

fn encode_pouch(pouch: &ElementStash, equipped: Option<BlockId>, world: &World) -> String {
    pouch
        .iter()
        .map(|(id, count)| {
            let star = if equipped == Some(id) { "*" } else { "" };
            format!("{star}{}={count}", world.registry().spec(id))
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn decode_pouch(mod_: &mut CraftingMod, data: &str, world: &mut World) -> u32 {
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
        mod_.push_loaded(world, id, count, equip);
    }
    skipped
}

fn encode_journal(journal: &[Procedure]) -> String {
    journal
        .iter()
        .map(|p| {
            let name = p
                .name
                .as_deref()
                .unwrap_or("")
                .replace('\t', " ")
                .replace('\n', " ");
            format!(
                "{}\t{}\t{}\t{}\t{}\t{name}",
                p.origin_spec,
                p.target_spec,
                p.event as u8,
                p.repeat,
                p.result_spec
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn decode_journal(data: &str, world: &mut World) -> Vec<Procedure> {
    let mut out = Vec::new();
    for line in data.lines().filter(|s| !s.is_empty()) {
        let mut parts = line.splitn(6, '\t');
        let Some(origin_spec) = parts.next() else { continue };
        let Some(target_spec) = parts.next() else { continue };
        let Some(event) = parts.next().and_then(|s| s.parse::<u8>().ok()) else {
            continue;
        };
        let Some(repeat) = parts.next().and_then(|s| s.parse::<u8>().ok()) else {
            continue;
        };
        let Some(result_spec) = parts.next() else { continue };
        let name = parts.next().unwrap_or("");
        let Some(event) = crate::net::protocol::workbench_event(event) else {
            continue;
        };
        if !(REPEAT_MIN..=REPEAT_MAX).contains(&repeat) {
            continue;
        }
        if world.registry_mut().parse_spec(origin_spec).is_none() {
            continue;
        }
        if world.registry_mut().parse_spec(target_spec).is_none() {
            continue;
        }
        if world.registry_mut().parse_spec(result_spec).is_none() {
            continue;
        }
        out.push(Procedure {
            origin_spec: origin_spec.into(),
            target_spec: target_spec.into(),
            event,
            repeat,
            result_spec: result_spec.into(),
            name: if name.is_empty() {
                None
            } else {
                Some(name.into())
            },
        });
    }
    out
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
        self.tab = Tab::Workbench;
        self.pouch.clear();
        self.equipped = None;
        self.origin = None;
        self.target = None;
        self.event = 0;
        self.repeat = REPEAT_MIN;
        self.journal.clear();
        self.held.clear();
        self.seen_stash_rev = 0;
        self.bump_hud();
    }

    fn description(&self) -> &str {
        "Take configurations into a pouch, apply events at the workbench, and place them (press C)."
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

    fn on_place_rejected(&mut self, id: BlockId, _world: &World) {
        // The unit left the pouch a moment ago; if its slot was taken meanwhile the refund cannot
        // fit and the unit is lost — counted, so the HUD can say so instead of hiding it.
        if !self.pouch.add(id, 1) {
            self.lost += 1;
        }
        self.bump_hud();
    }

    fn on_craft_result(
        &mut self,
        origin_spec: &str,
        target_spec: &str,
        event: u8,
        repeat: u8,
        result_spec: &str,
        world: &mut World,
    ) {
        let Some(kind) = crate::net::protocol::workbench_event(event) else {
            return;
        };
        if result_spec.is_empty() || !(REPEAT_MIN..=REPEAT_MAX).contains(&repeat) {
            return;
        }
        self.commit(
            world,
            origin_spec,
            target_spec,
            kind,
            repeat,
            Some(result_spec),
        );
    }

    fn command(&mut self, cmd: &str, args: &[&str]) -> Option<Vec<Line>> {
        if cmd != "name" {
            return None;
        }
        Some(self.name_procedure(args))
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
        let tab = self.tab;
        let pouch = &self.pouch;
        let equipped = self.equipped;
        let origin = self.origin;
        let target = self.target;
        let event = self.event;
        let repeat = self.repeat;
        let journal = self.journal.as_slice();
        let mut cache = self.hud_cache.borrow_mut();
        let cached = cache.get_or(key, || {
            paint_crafting(
                stash, ui, cursor, tab, pouch, equipped, origin, target, event, repeat, journal,
                world, screen_w, screen_h,
            )
        });
        out.extend(cached.iter().cloned());
        if self.lost > 0 {
            out.push(HudElement::Label {
                at: crate::ui::Anchor::Top,
                off: (0, PANEL_Y + LINE_HEIGHT),
                base_fs: FONT_SIZE,
                role: Role::Danger,
                text: format!("Pouch full - {} refunded unit(s) lost!", self.lost).into(),
            });
        }
    }

    fn close_overlay(&mut self) -> bool {
        if self.is_open() {
            self.set_open(false);
            true
        } else {
            false
        }
    }

    fn save_state(&self, world: &World) -> Option<(u16, String)> {
        let pouch = encode_pouch(&self.pouch, self.equipped, world);
        if self.journal.is_empty() {
            if pouch.is_empty() {
                return None;
            }
            return Some((1, pouch));
        }
        Some((2, format!("{pouch}\nJ\n{}", encode_journal(&self.journal))))
    }

    fn load_state(&mut self, version: u16, data: &str, world: &mut World) -> u32 {
        self.pouch.clear();
        self.equipped = None;
        self.origin = None;
        self.target = None;
        self.journal.clear();
        self.cursor = 0;
        let skipped = if version >= 2 {
            match data.split_once("\nJ\n") {
                Some((pouch, journal)) => {
                    let skipped = decode_pouch(self, pouch, world);
                    self.journal = decode_journal(journal, world);
                    skipped
                }
                None => decode_pouch(self, data, world),
            }
        } else {
            decode_pouch(self, data, world)
        };
        self.bump_hud();
        skipped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::registry::BlockRegistry;
    use material::{interact, Configuration, Element};

    fn test_mod() -> CraftingMod {
        CraftingMod::new(Rc::new(Cell::new(ItemUiState::default())))
    }

    fn mod_ctx<'a>(
        player: &'a mut Player,
        world: &'a mut World,
        placements: Vec<(i32, i32, i32, BlockId)>,
    ) -> ModContext<'a> {
        ModContext {
            player,
            world,
            screen_w: 800,
            screen_h: 600,
            place: false,
            place_target: None,
            toggle_inventory: false,
            toggle_crafting: false,
            nav_up: false,
            nav_down: false,
            nav_left: false,
            nav_right: false,
            nav_tab: false,
            nav_confirm: false,
            networked: false,
            placements,
            crafts: Vec::new(),
        }
    }

    fn reactive_pair(world: &mut World) -> (BlockId, BlockId, Configuration, Configuration) {
        let (a, b) = crate::sim::reactions::reactive_region_pair(world.registry_mut());
        let law = *world.registry().law();
        let ca = world.registry().configuration(a).clone();
        let cb = world.registry().configuration(b).clone();
        if interact(&law, &ca, &cb, EventKind::Collision).changed {
            (a, b, ca, cb)
        } else {
            (b, a, cb, ca)
        }
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
        assert_eq!(crafting.pouch.total(), 3);
        assert_eq!(crafting.pouch.count(soil), 2);
        assert_eq!(crafting.equipped, Some(rock));
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
        assert_eq!(crafting.pouch.total(), 3);
        assert_eq!(crafting.pouch.count(rock), 3);
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
        let text1 = crate::ui::hud_text(&hud1);
        assert!(!crafting.refresh(&player.stash), "no stash change: rows stay");
        assert_eq!(crafting.held, held);
        let mut hud2 = Vec::new();
        crafting.hud(&world, &player, (800, 600), &mut hud2);
        assert_eq!(crate::ui::hud_text(&hud2), text1, "HUD cache is reused until a stash/pouch mutation");

        player.stash.add(soil, 1);
        assert!(crafting.refresh(&player.stash));
        assert_eq!(crafting.held, vec![rock, soil]);
        let mut hud3 = Vec::new();
        crafting.hud(&world, &player, (800, 600), &mut hud3);
        assert_ne!(crate::ui::hud_text(&hud3), text1, "a stash mutation rebuilds the held rows");

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
        let mut world = World::new(1);
        let rock = world.registry().id_by_label("rock").unwrap();
        let mut player = Player::new(DVec3::new(0.0, 40.0, 0.0));
        player.stash.add(rock, 2);
        let mut crafting = test_mod();
        crafting.refresh(&player.stash);
        crafting.tab = Tab::Workbench;
        crafting.cursor = 1;
        let mut ctx = mod_ctx(&mut player, &mut world, Vec::new());
        crafting.activate(&mut ctx);
        assert_eq!(ctx.player.stash.count(rock), 1);
        assert_eq!(crafting.pouch.count(rock), 1);
    }

    #[test]
    fn apply_is_deterministic_and_equals_interact() {
        let mut world = World::new(1);
        let (origin, target, ca, cb) = reactive_pair(&mut world);
        let law = *world.registry().law();
        let expected = interact(&law, &ca, &cb, EventKind::Collision).target;
        let mut crafting = test_mod();
        crafting.pouch.add(origin, 1);
        crafting.pouch.add(target, 1);
        crafting.origin = Some(origin);
        crafting.target = Some(target);
        crafting.event = EVENTS.iter().position(|(k, _)| *k == EventKind::Collision).unwrap();
        crafting.repeat = 1;
        let mut player = Player::new(DVec3::new(0.0, 40.0, 0.0));
        let mut ctx = mod_ctx(&mut player, &mut world, Vec::new());
        crafting.apply(&mut ctx);
        assert_eq!(crafting.pouch.count(target), 0);
        assert_eq!(crafting.pouch.count(origin), 1, "origin is the tool");
        let rid = ctx.world.registry().lookup(&expected).expect("result interned");
        assert_eq!(crafting.pouch.count(rid), 1);
        assert_eq!(ctx.world.registry().configuration(rid), &expected);
        let origin_spec = ctx.world.registry().spec(origin);
        let target_spec = ctx.world.registry().spec(target);
        let result_spec = ctx.world.registry().spec(rid);

        let mut world2 = World::new(1);
        let o2 = world2.registry_mut().parse_spec(&origin_spec).unwrap();
        let t2 = world2.registry_mut().parse_spec(&target_spec).unwrap();
        let mut r2 = test_mod();
        r2.pouch.add(o2, 1);
        r2.pouch.add(t2, 1);
        r2.origin = Some(o2);
        r2.target = Some(t2);
        r2.event = crafting.event;
        r2.repeat = 1;
        let mut player2 = Player::new(DVec3::new(0.0, 40.0, 0.0));
        let mut ctx2 = mod_ctx(&mut player2, &mut world2, Vec::new());
        r2.apply(&mut ctx2);
        let rid2 = ctx2.world.registry_mut().parse_spec(&result_spec).unwrap();
        assert_eq!(ctx2.world.registry().spec(rid2), result_spec);
        assert_eq!(ctx2.world.registry().configuration(rid2), &expected);
    }

    #[test]
    fn result_is_interned_once() {
        let mut world = World::new(1);
        let (origin, target, ca, cb) = reactive_pair(&mut world);
        let law = *world.registry().law();
        let expected = interact_repeat(&law, &ca, &cb, EventKind::Collision, 2);
        let mut crafting = test_mod();
        crafting.pouch.add(origin, 1);
        crafting.pouch.add(target, 2);
        crafting.origin = Some(origin);
        crafting.target = Some(target);
        crafting.event = EVENTS.iter().position(|(k, _)| *k == EventKind::Collision).unwrap();
        crafting.repeat = 2;
        let mut player = Player::new(DVec3::new(0.0, 40.0, 0.0));
        let mut ctx = mod_ctx(&mut player, &mut world, Vec::new());
        let before = ctx.world.registry().block_count();
        crafting.apply(&mut ctx);
        let after_first = ctx.world.registry().block_count();
        crafting.apply(&mut ctx);
        let after_second = ctx.world.registry().block_count();
        assert_eq!(after_second, after_first, "second apply reuses the interned id");
        assert!(after_first == before || after_first == before + 1);
        let rid = ctx.world.registry().lookup(&expected).unwrap();
        assert_eq!(crafting.pouch.count(rid), 2);
    }

    #[test]
    fn journal_round_trips_and_name_command() {
        let mut world = World::new(1);
        let (origin, target, _, _) = reactive_pair(&mut world);
        let mut crafting = test_mod();
        crafting.pouch.add(origin, 1);
        crafting.pouch.add(target, 1);
        crafting.origin = Some(origin);
        crafting.target = Some(target);
        crafting.event = EVENTS.iter().position(|(k, _)| *k == EventKind::Collision).unwrap();
        crafting.repeat = 1;
        let mut player = Player::new(DVec3::new(0.0, 40.0, 0.0));
        {
            let mut ctx = mod_ctx(&mut player, &mut world, Vec::new());
            crafting.apply(&mut ctx);
        }
        assert_eq!(crafting.journal.len(), 1, "a new configuration is recorded");
        let saved = crafting.save_state(&world).expect("journal persists");
        assert_eq!(saved.0, 2);

        let mut loaded = test_mod();
        loaded.load_state(saved.0, &saved.1, &mut world);
        assert_eq!(loaded.journal.len(), 1);
        assert_eq!(loaded.journal[0].event, EventKind::Collision);
        assert_eq!(loaded.journal[0].repeat, 1);
        assert!(loaded.journal[0].name.is_none());

        let out = loaded.command("name", &["1", "spark", "mix"]).unwrap();
        assert!(out[0].text().contains("named"));
        assert_eq!(loaded.journal[0].name.as_deref(), Some("spark mix"));
        let saved2 = loaded.save_state(&world).unwrap();
        let mut named = test_mod();
        named.load_state(saved2.0, &saved2.1, &mut world);
        assert_eq!(named.journal[0].name.as_deref(), Some("spark mix"));

        let before = named.journal.len();
        named.origin = Some(origin);
        named.target = Some(target);
        named.event = crafting.event;
        named.repeat = 1;
        named.pouch.add(origin, 1);
        named.pouch.add(target, 1);
        {
            let mut ctx = mod_ctx(&mut player, &mut world, Vec::new());
            named.apply(&mut ctx);
        }
        assert_eq!(
            named.journal.len(),
            before,
            "a known result id is not recorded again"
        );
    }

    #[test]
    fn server_evaluation_matches_client_expectation() {
        let mut client = BlockRegistry::with_builtins();
        let mut server = BlockRegistry::with_builtins();
        let origin = Configuration::single(Element::new([40, 80, 120, 160]));
        let target = Configuration::single(Element::new([80, 40, 160, 120]));
        let oid = client.intern(&origin).unwrap();
        let tid = client.intern(&target).unwrap();
        let os = client.spec(oid);
        let ts = client.spec(tid);
        let law = *client.law();
        let expected = interact_repeat(&law, &origin, &target, EventKind::Collision, 4);
        let client_id = client
            .apply_specs(&os, &ts, EventKind::Collision, 4)
            .unwrap();
        let server_id = server
            .apply_specs(&os, &ts, EventKind::Collision, 4)
            .unwrap();
        assert_eq!(client.configuration(client_id), &expected);
        assert_eq!(server.configuration(server_id), &expected);
        assert_eq!(client.spec(client_id), server.spec(server_id));
    }

    #[test]
    fn pouch_capacity_overflow_drops_extra_and_apply_at_cap_keeps_the_result() {
        let mut world = World::new(1);
        let (origin, target, _, _) = reactive_pair(&mut world);
        let mut crafting = test_mod();
        assert!(crafting.pouch.add(target, START_CAPACITY as u32 - 1));
        assert!(crafting.pouch.add(origin, 1));
        assert_eq!(crafting.pouch.total(), START_CAPACITY as u32);
        assert!(!crafting.pouch.add(origin, 1), "pouch uses stash capacity");
        assert_eq!(crafting.pouch.total(), START_CAPACITY as u32);

        let mut player = Player::new(DVec3::new(0.0, 40.0, 0.0));
        player.stash.add(origin, 1);
        crafting.held = vec![origin];
        crafting.tab = Tab::Workbench;
        crafting.cursor = 1;
        let mut ctx = mod_ctx(&mut player, &mut world, Vec::new());
        let stash_before = ctx.player.stash.count(origin);
        crafting.activate(&mut ctx);
        assert_eq!(
            ctx.player.stash.count(origin),
            stash_before,
            "a full pouch restores the taken unit"
        );

        crafting.origin = Some(origin);
        crafting.target = Some(target);
        crafting.event = EVENTS.iter().position(|(k, _)| *k == EventKind::Collision).unwrap();
        crafting.repeat = 1;
        let total_before = crafting.pouch.total();
        crafting.apply(&mut ctx);
        assert_eq!(
            crafting.pouch.total(),
            total_before,
            "consume-then-add at capacity does not drop the result"
        );
        assert!(crafting.pouch.total() > 0);
    }

    #[test]
    fn apply_requires_the_origin_to_be_held() {
        let mut world = World::new(1);
        let (origin, target, _, _) = reactive_pair(&mut world);
        let mut crafting = test_mod();
        assert!(crafting.pouch.add(target, 2));
        crafting.origin = Some(origin); // named, but no unit of it in the pouch
        crafting.target = Some(target);
        crafting.event = EVENTS.iter().position(|(k, _)| *k == EventKind::Collision).unwrap();
        crafting.repeat = 1;
        let mut player = Player::new(DVec3::new(0.0, 40.0, 0.0));
        let mut ctx = mod_ctx(&mut player, &mut world, Vec::new());
        crafting.apply(&mut ctx);
        assert_eq!(crafting.pouch.count(target), 2, "nothing is consumed without the origin");
        assert_eq!(crafting.origin, None, "an unheld origin slot is cleared");
        ctx.networked = true;
        crafting.origin = Some(origin);
        crafting.apply(&mut ctx);
        assert!(ctx.crafts.is_empty(), "no request leaves for an unheld origin");
        // The authoritative reply path is held to the same rule.
        let ospec = ctx.world.registry().spec(origin);
        let tspec = ctx.world.registry().spec(target);
        let rspec = ctx.world.registry().spec(origin);
        crafting.on_craft_result(&ospec, &tspec, EventKind::Collision as u8, 1, &rspec, ctx.world);
        assert_eq!(crafting.pouch.count(target), 2, "a result for an unheld origin does not commit");
    }

    #[test]
    fn a_refund_that_cannot_fit_is_counted_not_hidden() {
        let mut world = World::new(1);
        let (origin, target, _, _) = reactive_pair(&mut world);
        let mut crafting = test_mod();
        assert!(crafting.pouch.add(target, START_CAPACITY as u32));
        crafting.on_place_rejected(origin, &world);
        assert_eq!(crafting.pouch.total(), START_CAPACITY as u32);
        assert_eq!(crafting.lost, 1);
        let mut player = Player::new(DVec3::new(0.0, 40.0, 0.0));
        let mut shown = Vec::new();
        crafting.hud(&world, &player, (800, 600), &mut shown);
        assert!(
            crate::ui::hud_text(&shown).contains("lost"),
            "{}",
            crate::ui::hud_text(&shown)
        );
        player.stash.add(origin, 1);
    }

    #[test]
    fn a_fixed_point_is_not_a_discovery() {
        let mut world = World::new(1);
        let rock = world
            .registry_mut()
            .intern(&Configuration::single(Element::new([120, 130, 140, 150])))
            .unwrap();
        let mut crafting = test_mod();
        assert!(crafting.pouch.add(rock, 2));
        crafting.origin = Some(rock);
        crafting.target = Some(rock);
        crafting.event = EVENTS.iter().position(|(k, _)| *k == EventKind::NewContact).unwrap();
        crafting.repeat = 1;
        let mut player = Player::new(DVec3::new(0.0, 40.0, 0.0));
        let mut ctx = mod_ctx(&mut player, &mut world, Vec::new());
        crafting.apply(&mut ctx);
        assert!(crafting.journal.is_empty(), "rock on rock at rest teaches nothing");
    }

    #[test]
    fn networked_apply_queues_craft_and_result_commits() {
        let mut world = World::new(1);
        let (origin, target, ca, cb) = reactive_pair(&mut world);
        let law = *world.registry().law();
        let expected = interact(&law, &ca, &cb, EventKind::Collision).target;
        let mut crafting = test_mod();
        crafting.pouch.add(origin, 1);
        crafting.pouch.add(target, 1);
        crafting.origin = Some(origin);
        crafting.target = Some(target);
        crafting.event = EVENTS.iter().position(|(k, _)| *k == EventKind::Collision).unwrap();
        crafting.repeat = 1;
        let mut player = Player::new(DVec3::new(0.0, 40.0, 0.0));
        let mut ctx = mod_ctx(&mut player, &mut world, Vec::new());
        ctx.networked = true;
        crafting.apply(&mut ctx);
        assert_eq!(ctx.crafts.len(), 1);
        assert_eq!(crafting.pouch.count(target), 1, "client waits for the server");
        let result_spec = {
            let mut r = BlockRegistry::with_builtins();
            let id = r.intern(&expected).unwrap();
            r.spec(id)
        };
        let req = &ctx.crafts[0];
        crafting.on_craft_result(
            &req.origin_spec,
            &req.target_spec,
            req.event,
            req.repeat,
            &result_spec,
            ctx.world,
        );
        let rid = ctx.world.registry().lookup(&expected).unwrap();
        assert_eq!(crafting.pouch.count(target), 0);
        assert_eq!(crafting.pouch.count(rid), 1);
    }

}
