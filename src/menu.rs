//! The out-of-game screens as pure MODELS: the start menu (new / load / host /
//! join / mods / settings / quit), the mod list, the graphics settings, and the
//! host/join forms.
//!
//! The user's fundamental holds here: a menu is really just a list of
//! alternatives that lead to something. Core therefore owns only the MODEL and
//! the MEANING — a [`MenuModel`] is built per screen by the functions below,
//! and [`crate::app`] interprets the [`MenuEvent`]s that come back. The LOOK
//! and the INTERACTION belong to mods (see
//! [`menu_default`](crate::mods::menu_default), the default "Menus" mod),
//! exactly like the inventory and crafting mods: default-enabled, disableable,
//! replaceable.
//!
//! So that disabling the Menus mod can never brick navigation, this module
//! also keeps a built-in fallback driver and renderer
//! ([`fallback_drive`]/[`fallback_draw`]). The default mod is a thin wrapper
//! around the same free functions ([`drive`]/[`draw_model`]) — one
//! implementation, two entry points.
use voxel_engine::{Color, Engine, Frame, Key};

use crate::console::shadowed;
use crate::mods::Mods;
use crate::net::{DEFAULT_PORT, MAX_NAME};
use crate::settings::Settings;

// ---------------------------------------------------------------------------
// The model (frozen shapes — see ENGINE_DESIGN R6.1).
// ---------------------------------------------------------------------------

/// One whole menu screen: a titled list of entries with a cursor, a key hint
/// line, and an optional error line. Everything a renderer needs — and nothing
/// about what the entries MEAN.
pub struct MenuModel {
    pub title: String,
    /// Present only on the start menu; renderers use it to pick the big
    /// title treatment (48px gold + subtitle) over the compact one.
    pub subtitle: Option<String>,
    pub entries: Vec<MenuEntry>,
    pub cursor: usize,
    /// The key-hint line at the bottom of the screen.
    pub hint: String,
    /// A transient message (bad port, failed connect) drawn in the hint area.
    /// Cleared by the driver on the next editing keystroke.
    pub error: Option<String>,
}

/// One alternative in a menu.
pub struct MenuEntry {
    pub label: String,
    /// A dimmer second line under the entry (mod descriptions).
    pub detail: Option<String>,
    pub kind: EntryKind,
}

/// What kind of alternative an entry is — which decides how the driver
/// interacts with it and which events it can produce.
pub enum EntryKind {
    /// Activating it picks it ([`MenuEvent::Chosen`]).
    Action,
    /// An on/off switch ([`MenuEvent::Toggled`]); the bool is the shown state.
    Toggle(bool),
    /// A value stepped left/right through a list ([`MenuEvent::Cycled`]).
    Cycle { value: String },
    /// An editable text field; the driver types into it. `max` is the byte
    /// cap; `masked` renders as `*`s (passwords).
    Text { value: String, max: usize, masked: bool },
}

/// What the player did to a menu, in meaning-free index terms. The owner of
/// the model (the App) turns these back into meaning.
pub enum MenuEvent {
    /// An [`EntryKind::Action`] entry was activated.
    Chosen(usize),
    /// An [`EntryKind::Toggle`] entry was flipped.
    Toggled(usize),
    /// An [`EntryKind::Cycle`] entry was stepped (`-1` or `+1`).
    Cycled(usize, i32),
    /// Leave this screen.
    Back,
    /// Submit the whole form (Enter while on a text field).
    Submit,
}

impl MenuEntry {
    pub fn action(label: impl Into<String>) -> Self {
        Self { label: label.into(), detail: None, kind: EntryKind::Action }
    }

    pub fn toggle(label: impl Into<String>, on: bool, detail: impl Into<String>) -> Self {
        Self { label: label.into(), detail: Some(detail.into()), kind: EntryKind::Toggle(on) }
    }

    pub fn cycle(label: impl Into<String>, value: impl Into<String>) -> Self {
        Self { label: label.into(), detail: None, kind: EntryKind::Cycle { value: value.into() } }
    }

    pub fn text(label: impl Into<String>, value: impl Into<String>, max: usize, masked: bool) -> Self {
        Self {
            label: label.into(),
            detail: None,
            kind: EntryKind::Text { value: value.into(), max, masked },
        }
    }
}

impl MenuModel {
    /// The value of the text field at `index` (empty for non-text entries) —
    /// how the App reads a submitted form back out of the model.
    pub fn text_value(&self, index: usize) -> &str {
        match &self.entries[index].kind {
            EntryKind::Text { value, .. } => value,
            _ => "",
        }
    }

    /// Keep the cursor on a real entry after the list shrank (save deleted).
    pub fn clamp_cursor(&mut self) {
        self.cursor = self.cursor.min(self.entries.len().saturating_sub(1));
    }
}

// ---------------------------------------------------------------------------
// Model builders — one per screen. The label/detail/value strings here are
// byte-identical to what the old concrete menus rendered, so the default
// renderer reproduces today's screens exactly.
// ---------------------------------------------------------------------------

/// What the player picked on the start menu (the meaning behind
/// [`main_choice_at`]).
pub enum MainChoice {
    NewWorld,
    Load(String),
    Host,
    Join,
    Mods,
    Settings,
    Quit,
}

/// The start menu: New World, one Load row per save, Host, Join, Mods,
/// Settings, Quit.
pub fn main_menu_model(saves: &[String]) -> MenuModel {
    let mut entries = vec![MenuEntry::action("New World")];
    for name in saves {
        entries.push(MenuEntry::action(format!("Load: {name}")));
    }
    entries.push(MenuEntry::action("Host Server"));
    entries.push(MenuEntry::action("Join Server"));
    entries.push(MenuEntry::action("Mods"));
    entries.push(MenuEntry::action("Settings"));
    entries.push(MenuEntry::action("Quit"));
    MenuModel {
        title: "PROJECT WATT CUBED".to_string(),
        subtitle: Some("an infinite voxel world of elements".to_string()),
        entries,
        cursor: 0,
        hint: "Up/Down or j/k select   Enter or l choose".to_string(),
        error: None,
    }
}

/// The index map for [`main_menu_model`]: resolve a [`MenuEvent::Chosen`]
/// index against the same saves list the model was built from.
pub fn main_choice_at(saves: &[String], index: usize) -> MainChoice {
    let count = saves.len();
    if index == 0 {
        MainChoice::NewWorld
    } else if index <= count {
        MainChoice::Load(saves[index - 1].clone())
    } else if index == count + 1 {
        MainChoice::Host
    } else if index == count + 2 {
        MainChoice::Join
    } else if index == count + 3 {
        MainChoice::Mods
    } else if index == count + 4 {
        MainChoice::Settings
    } else {
        MainChoice::Quit
    }
}

/// The mod menu: one Toggle row per installed mod, description as the detail.
pub fn mods_menu_model(mods: &Mods) -> MenuModel {
    let entries = (0..mods.len())
        .map(|i| MenuEntry::toggle(mods.name(i), mods.is_enabled(i), mods.description(i)))
        .collect();
    MenuModel {
        title: "MODS".to_string(),
        subtitle: None,
        entries,
        cursor: 0,
        hint: "Up/Down or j/k select   Enter/l toggle   Esc/h back".to_string(),
        error: None,
    }
}

/// Settings rows, top to bottom: Fullscreen, VSync, MSAA, Max FPS, Render
/// Distance, FOV, Render Scale, Back.
pub const SETTINGS_ROW_BACK: usize = 7;

/// The settings menu: one Cycle row per graphics option plus a Back action.
/// Values are formatted exactly as the old screen printed them.
pub fn settings_menu_model(s: &Settings) -> MenuModel {
    let on_off = |on: bool| if on { "On" } else { "Off" };
    let max_fps = if s.max_fps == 0 {
        "Uncapped".to_string()
    } else {
        s.max_fps.to_string()
    };
    let entries = vec![
        MenuEntry::cycle("Fullscreen", on_off(s.fullscreen)),
        MenuEntry::cycle("VSync", on_off(s.vsync)),
        MenuEntry::cycle("MSAA", format!("{}x", s.msaa)),
        MenuEntry::cycle("Max FPS", max_fps),
        MenuEntry::cycle("Render Distance", s.render_distance.to_string()),
        MenuEntry::cycle("FOV", format!("{}", s.fov)),
        MenuEntry::cycle("Render Scale", format!("{:.0}%", s.render_scale * 100.0)),
        MenuEntry::action("Back"),
    ];
    MenuModel {
        title: "SETTINGS".to_string(),
        subtitle: None,
        entries,
        cursor: 0,
        hint: "Up/Down or j/k select | Left/Right or h/l change | Esc back".to_string(),
        error: None,
    }
}

/// Apply one Left/Right (or Enter) step to a settings row, wrapping — the
/// MEANING of a [`MenuEvent::Cycled`] on the settings screen, kept App-side so
/// no mod ever decides what "MSAA" means.
pub fn apply_settings_cycle(s: &mut Settings, row: usize, dir: i32) {
    match row {
        0 => s.fullscreen = !s.fullscreen,
        1 => s.vsync = !s.vsync,
        2 => s.msaa = cycle_list(&[1, 2, 4, 8], s.msaa, dir),
        3 => s.max_fps = cycle_list(&[0, 30, 60, 120, 144, 240], s.max_fps, dir),
        4 => {
            let v = s.render_distance.clamp(3, 10) + dir;
            s.render_distance = if v > 10 { 3 } else if v < 3 { 10 } else { v };
        }
        5 => {
            let v = s.fov.clamp(50.0, 110.0) + dir as f32 * 5.0;
            s.fov = if v > 110.0 { 50.0 } else if v < 50.0 { 110.0 } else { v };
        }
        6 => {
            // Percent steps; the engine clamps to 25%..200%.
            let pct = cycle_list(
                &[25, 50, 75, 100, 125, 150, 200],
                (s.render_scale * 100.0).round() as u32,
                dir,
            );
            s.render_scale = pct as f32 / 100.0;
        }
        _ => {}
    }
}

/// Step to the adjacent entry in `values`, wrapping at both ends. A current value
/// not in the list (e.g. a hand-edited config) snaps to the first entry first.
pub fn cycle_list(values: &[u32], current: u32, dir: i32) -> u32 {
    match values.iter().position(|&v| v == current) {
        Some(i) => values[(i as i32 + dir).rem_euclid(values.len() as i32) as usize],
        // Off-list (e.g. a /gfx or hand-edited value): snap to the first
        // entry without stepping, so Left can never jump 25% -> 200%.
        None => values[0],
    }
}

/// Details entered to host a server.
pub struct HostInfo {
    pub port: u16,
    pub password: String,
    pub name: String,
}

/// Details entered to join a server.
pub struct JoinInfo {
    pub host: String,
    pub port: u16,
    pub password: String,
    pub name: String,
}

/// The host form: Port, optional Password (masked), Your name. `prior` (a
/// previous host model) carries typed values over so re-opening the screen
/// keeps what the player entered, like the old persistent form did.
pub fn host_menu_model(prior: Option<&MenuModel>) -> MenuModel {
    let mut model = MenuModel {
        title: "HOST SERVER".to_string(),
        subtitle: None,
        entries: vec![
            MenuEntry::text("Port", DEFAULT_PORT.to_string(), 5, false),
            MenuEntry::text("Password (optional)", "", 64, true),
            MenuEntry::text("Your name", "player", MAX_NAME, false),
        ],
        cursor: 0,
        hint: "Up/Down field   type to edit   Enter start   Esc back".to_string(),
        error: None,
    };
    if let Some(prior) = prior {
        carry_text_values(prior, &mut model);
    }
    model
}

/// The join form: Address, Port, Password (masked), Your name. Same `prior`
/// convention as [`host_menu_model`].
pub fn join_menu_model(prior: Option<&MenuModel>) -> MenuModel {
    let mut model = MenuModel {
        title: "JOIN SERVER".to_string(),
        subtitle: None,
        entries: vec![
            MenuEntry::text("Address", "127.0.0.1", 64, false),
            MenuEntry::text("Port", DEFAULT_PORT.to_string(), 5, false),
            MenuEntry::text("Password", "", 64, true),
            MenuEntry::text("Your name", "player", MAX_NAME, false),
        ],
        cursor: 0,
        hint: "Up/Down field   type to edit   Enter connect   Esc back".to_string(),
        error: None,
    };
    if let Some(prior) = prior {
        carry_text_values(prior, &mut model);
    }
    model
}

/// Copy text-field values from a previous incarnation of the same form,
/// positionally, so a rebuilt model keeps what the player typed.
fn carry_text_values(prior: &MenuModel, model: &mut MenuModel) {
    for (old, new) in prior.entries.iter().zip(model.entries.iter_mut()) {
        if let (EntryKind::Text { value: from, .. }, EntryKind::Text { value: to, .. }) =
            (&old.kind, &mut new.kind)
        {
            *to = from.clone();
        }
    }
    model.cursor = prior.cursor.min(model.entries.len().saturating_sub(1));
}

/// The error shown when a submitted port doesn't parse.
pub const PORT_ERROR: &str = "invalid port (1-65535)";

/// Parse a port field. An EMPTY field keeps meaning [`DEFAULT_PORT`] — the
/// form pre-fills the default, and clearing the field is a handy way to say
/// "just use the default". Anything non-empty must be a real port (1-65535):
/// `None` refuses the submit rather than silently falling back (a typo like
/// "99999" used to silently become 5555 and host/join the wrong port).
pub fn parse_port(text: &str) -> Option<u16> {
    let text = text.trim();
    if text.is_empty() {
        return Some(DEFAULT_PORT);
    }
    match text.parse::<u16>() {
        Ok(0) | Err(_) => None,
        Ok(port) => Some(port),
    }
}

// ---------------------------------------------------------------------------
// The driver: keys in, cursor/text mutations + one event out. Pure over a
// per-frame key snapshot so it is unit-testable without an Engine.
// ---------------------------------------------------------------------------

/// One frame of menu-relevant input, snapshotted from the [`Engine`]. `chars`
/// is the frame's drained text queue (layout- and shift-aware), consumed here
/// so a menu frame owns its keystrokes.
#[derive(Default, Clone, Debug)]
pub struct MenuKeys {
    pub up: bool,
    pub down: bool,
    pub j: bool,
    pub k: bool,
    pub enter: bool,
    pub l: bool,
    pub esc: bool,
    pub h: bool,
    pub tab: bool,
    pub space: bool,
    pub left: bool,
    pub right: bool,
    pub chars: Vec<char>,
    pub backspace: bool,
}

impl MenuKeys {
    /// Snapshot this frame's menu input, draining the char queue.
    pub fn capture(eng: &Engine) -> Self {
        let mut chars = Vec::new();
        while let Some(c) = eng.get_char_pressed() {
            chars.push(c);
        }
        Self {
            up: eng.is_key_pressed(Key::Up),
            down: eng.is_key_pressed(Key::Down),
            j: eng.is_key_pressed(Key::J),
            k: eng.is_key_pressed(Key::K),
            enter: eng.is_key_pressed(Key::Enter),
            l: eng.is_key_pressed(Key::L),
            esc: eng.is_key_pressed(Key::Escape),
            h: eng.is_key_pressed(Key::H),
            tab: eng.is_key_pressed(Key::Tab),
            space: eng.is_key_pressed(Key::Space),
            left: eng.is_key_pressed(Key::Left),
            right: eng.is_key_pressed(Key::Right),
            chars,
            backspace: eng.is_key_pressed(Key::Backspace),
        }
    }
}

/// Interpret one frame of keys against a model: move the cursor, edit text
/// fields, flip toggle displays, and emit at most one [`MenuEvent`]. This is
/// the whole input contract of the old concrete menus, generalized per
/// [`EntryKind`]:
///
/// - Up/Down (and j/k off text fields) move with wraparound.
/// - Enter/l activate: Action -> [`MenuEvent::Chosen`], Toggle ->
///   [`MenuEvent::Toggled`] (Space too — mod-menu parity), Cycle ->
///   [`MenuEvent::Cycled`]`(+1)`.
/// - Left/Right (and h/l) step Cycle entries.
/// - Esc -> [`MenuEvent::Back`]; on Toggle rows h and Backspace too
///   (mod-menu parity). On Action rows h does nothing (start-menu parity).
/// - While the CURSOR is on a Text entry the form rules apply instead:
///   Down/Tab and Up move fields, typed chars (including hjkl) go INTO the
///   field (byte cap, control chars rejected), Backspace pops, Enter submits
///   the whole form, Esc cancels. Editing clears `model.error`.
///
/// It never interprets meaning — that stays with the App.
pub fn drive(keys: &MenuKeys, menu: &mut MenuModel) -> Option<MenuEvent> {
    let n = menu.entries.len();
    if n == 0 {
        // Only an emptied mod list can get here; every back alias still works.
        return (keys.esc || keys.h || keys.backspace).then_some(MenuEvent::Back);
    }
    menu.cursor = menu.cursor.min(n - 1);

    if matches!(menu.entries[menu.cursor].kind, EntryKind::Text { .. }) {
        return drive_text(keys, menu, n);
    }

    // List navigation first, so activation reads the post-move row (holding
    // Down and tapping Enter picks what the highlight shows).
    if keys.down || keys.j {
        menu.cursor = (menu.cursor + 1) % n;
    }
    if keys.up || keys.k {
        menu.cursor = (menu.cursor + n - 1) % n;
    }
    let cursor = menu.cursor;
    match &mut menu.entries[cursor].kind {
        EntryKind::Action => {
            if keys.enter || keys.l {
                return Some(MenuEvent::Chosen(cursor));
            }
        }
        EntryKind::Toggle(on) => {
            if keys.enter || keys.l || keys.space {
                // Flip the DISPLAY optimistically; the owner rebuilds the
                // model from the source of truth after acting on the event.
                *on = !*on;
                return Some(MenuEvent::Toggled(cursor));
            }
            if keys.esc || keys.h || keys.backspace {
                return Some(MenuEvent::Back);
            }
        }
        EntryKind::Cycle { .. } => {
            if keys.left || keys.h {
                return Some(MenuEvent::Cycled(cursor, -1));
            }
            if keys.right || keys.l || keys.enter {
                return Some(MenuEvent::Cycled(cursor, 1));
            }
        }
        // Navigation just landed on a text field; editing starts next frame.
        EntryKind::Text { .. } => {}
    }
    if keys.esc {
        return Some(MenuEvent::Back);
    }
    None
}

/// Form semantics while the cursor sits on a text field (see [`drive`]).
fn drive_text(keys: &MenuKeys, menu: &mut MenuModel, n: usize) -> Option<MenuEvent> {
    if keys.down || keys.tab {
        menu.cursor = (menu.cursor + 1) % n;
    }
    if keys.up {
        menu.cursor = (menu.cursor + n - 1) % n;
    }
    if keys.enter {
        return Some(MenuEvent::Submit);
    }
    if keys.esc {
        return Some(MenuEvent::Back);
    }
    // Edits target the (possibly just-moved-to) selected field, like the old
    // form did. On a mixed menu the cursor may have landed on a non-text row,
    // in which case the edits simply have nowhere to go.
    if keys.backspace {
        if let EntryKind::Text { value, .. } = &mut menu.entries[menu.cursor].kind {
            value.pop();
        }
        menu.error = None;
    }
    for &c in &keys.chars {
        if let EntryKind::Text { value, max, .. } = &mut menu.entries[menu.cursor].kind {
            if !c.is_control() && value.len() < *max {
                value.push(c);
            }
        }
        menu.error = None;
    }
    None
}

// ---------------------------------------------------------------------------
// The renderer: the exact current visuals, driven entirely by the model. Three
// families, picked from the model itself: the big-title start-menu style
// (subtitle present), the form style (a text field anywhere), and the compact
// list style (everything else).
// ---------------------------------------------------------------------------

/// Background for every menu screen.
const MENU_BG: Color = Color::new(18, 20, 28, 255);

/// Draw a menu model in the standard style. Free function so the default
/// "Menus" mod and the core fallback share one implementation.
pub fn draw_model(f: &mut Frame, menu: &MenuModel, w: i32, h: i32) {
    f.draw_rect(0, 0, w, h, MENU_BG);
    if menu.subtitle.is_some() {
        draw_main_style(f, menu, w, h);
    } else if menu
        .entries
        .iter()
        .any(|e| matches!(e.kind, EntryKind::Text { .. }))
    {
        draw_form_style(f, menu, w, h);
    } else {
        draw_list_style(f, menu, w, h);
    }
}

/// The start-menu look: 48px gold title at h/6, gray subtitle, centered rows.
fn draw_main_style(f: &mut Frame, menu: &MenuModel, w: i32, h: i32) {
    let title_fs = 48;
    let tx = (w - f.measure_text(&menu.title, title_fs)) / 2;
    shadowed(f, &menu.title, tx, h / 6, title_fs, Color::GOLD);

    let subtitle = menu.subtitle.as_deref().unwrap_or("");
    let sub_fs = 20;
    let sx = (w - f.measure_text(subtitle, sub_fs)) / 2;
    shadowed(f, subtitle, sx, h / 6 + title_fs + 8, sub_fs, Color::GRAY);

    let fs = 28;
    let line_h = fs + 14;
    let start_y = h / 2 - line_h;
    for (i, entry) in menu.entries.iter().enumerate() {
        let selected = i == menu.cursor;
        let text = if selected {
            format!("> {}", entry.label)
        } else {
            format!("  {}", entry.label)
        };
        let color = if selected { Color::RAYWHITE } else { Color::GRAY };
        let x = (w - f.measure_text(&text, fs)) / 2;
        shadowed(f, &text, x, start_y + line_h * i as i32, fs, color);
    }

    draw_hint(f, &menu.hint, w, h);

    // The start menu's status line (failed connect/host), salmon above the hint.
    if let Some(error) = &menu.error {
        let fs = 20;
        let ex = (w - f.measure_text(error, fs)) / 2;
        shadowed(f, error, ex, h - 70, fs, Color::SALMON);
    }
}

/// The compact list look shared by the mod and settings screens: 40px title at
/// h/8, left-aligned rows from h/4 + 20.
fn draw_list_style(f: &mut Frame, menu: &MenuModel, w: i32, h: i32) {
    let title_fs = 40;
    let tx = (w - f.measure_text(&menu.title, title_fs)) / 2;
    shadowed(f, &menu.title, tx, h / 8, title_fs, Color::GOLD);

    let fs = 26;
    let line_h = fs + 20;
    let start_y = h / 4 + 20;
    let x = w / 2 - 260;

    if menu.entries.is_empty() {
        // The only empty menu in the game is a modless mod list.
        shadowed(f, "  (no mods installed)", x, start_y, fs, Color::GRAY);
    }

    for (i, entry) in menu.entries.iter().enumerate() {
        let selected = i == menu.cursor;
        let mark = if selected { ">" } else { " " };
        let row = match &entry.kind {
            EntryKind::Toggle(on) => {
                format!("{} {} {}", mark, if *on { "[x]" } else { "[ ]" }, entry.label)
            }
            EntryKind::Cycle { value } => format!("{} {}: {}", mark, entry.label, value),
            _ => format!("{} {}", mark, entry.label),
        };
        let color = if selected { Color::RAYWHITE } else { Color::GRAY };
        shadowed(f, &row, x, start_y + line_h * i as i32, fs, color);
        if let Some(detail) = &entry.detail {
            shadowed(f, detail, x + 40, start_y + line_h * i as i32 + fs + 2, 16, Color::DARKGRAY);
        }
    }

    draw_error(f, menu, w, h);
    draw_hint(f, &menu.hint, w, h);
}

/// The form look shared by host and join: 40px title at h/6, "label: value"
/// rows from h/2 - line_h with a trailing `_` caret on the selected field and
/// `*`-masked passwords.
fn draw_form_style(f: &mut Frame, menu: &MenuModel, w: i32, h: i32) {
    let title_fs = 40;
    let tx = (w - f.measure_text(&menu.title, title_fs)) / 2;
    shadowed(f, &menu.title, tx, h / 6, title_fs, Color::GOLD);

    let fs = 26;
    let line_h = fs + 22;
    let start_y = h / 2 - line_h;
    let x = w / 2 - 240;
    for (i, entry) in menu.entries.iter().enumerate() {
        let selected = i == menu.cursor;
        let mark = if selected { ">" } else { " " };
        let row = match &entry.kind {
            EntryKind::Text { value, masked, .. } => {
                let shown = if *masked {
                    "*".repeat(value.chars().count())
                } else {
                    value.clone()
                };
                let caret = if selected { "_" } else { "" };
                format!("{} {}: {}{}", mark, entry.label, shown, caret)
            }
            EntryKind::Toggle(on) => {
                format!("{} {} {}", mark, if *on { "[x]" } else { "[ ]" }, entry.label)
            }
            EntryKind::Cycle { value } => format!("{} {}: {}", mark, entry.label, value),
            EntryKind::Action => format!("{} {}", mark, entry.label),
        };
        let color = if selected { Color::RAYWHITE } else { Color::GRAY };
        shadowed(f, &row, x, start_y + line_h * i as i32, fs, color);
    }

    draw_error(f, menu, w, h);
    draw_hint(f, &menu.hint, w, h);
}

/// The key-hint line every screen shows at the bottom.
fn draw_hint(f: &mut Frame, hint: &str, w: i32, h: i32) {
    let hint_fs = 18;
    let hx = (w - f.measure_text(hint, hint_fs)) / 2;
    shadowed(f, hint, hx, h - 40, hint_fs, Color::DARKGRAY);
}

/// A refused submit's error sits just above the hint, in red, until the next
/// keystroke.
fn draw_error(f: &mut Frame, menu: &MenuModel, w: i32, h: i32) {
    if let Some(error) = &menu.error {
        let hint_fs = 18;
        let ex = (w - f.measure_text(error, hint_fs)) / 2;
        shadowed(f, error, ex, h - 40 - (hint_fs + 8), hint_fs, Color::RED);
    }
}

// ---------------------------------------------------------------------------
// The no-brick fallback. The App uses these whenever NO enabled mod handles
// menus, so switching the "Menus" mod off (or replacing it with a broken one
// and disabling that) can never strand the player without navigation.
// ---------------------------------------------------------------------------

/// Built-in menu driver: same logic as the default mod (both are [`drive`]).
pub fn fallback_drive(eng: &Engine, menu: &mut MenuModel) -> Option<MenuEvent> {
    drive(&MenuKeys::capture(eng), menu)
}

/// Built-in menu renderer: same drawing as the default mod ([`draw_model`]).
pub fn fallback_draw(f: &mut Frame, menu: &MenuModel, w: i32, h: i32) {
    draw_model(f, menu, w, h);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn port_parsing_accepts_real_ports_and_refuses_junk() {
        // Valid ports pass through untouched (whitespace tolerated).
        assert_eq!(parse_port("5555"), Some(5555));
        assert_eq!(parse_port(" 8080 "), Some(8080));
        assert_eq!(parse_port("1"), Some(1));
        assert_eq!(parse_port("65535"), Some(65535));

        // An empty field keeps meaning the default — the pre-filled-form
        // convenience must survive validation.
        assert_eq!(parse_port(""), Some(DEFAULT_PORT));
        assert_eq!(parse_port("   "), Some(DEFAULT_PORT));

        // Out-of-range, zero, or non-numeric input refuses the submit instead
        // of silently becoming the default (the "99999 -> 5555" bug).
        assert_eq!(parse_port("99999"), None);
        assert_eq!(parse_port("65536"), None);
        assert_eq!(parse_port("0"), None);
        assert_eq!(parse_port("-1"), None);
        assert_eq!(parse_port("555x"), None);
        assert_eq!(parse_port("port"), None);
    }

    // ---- model builders pin today's exact strings ----

    #[test]
    fn main_menu_model_matches_the_old_screen_exactly() {
        let saves = vec!["alpha".to_string(), "beta".to_string()];
        let m = main_menu_model(&saves);
        assert_eq!(m.title, "PROJECT WATT CUBED");
        assert_eq!(m.subtitle.as_deref(), Some("an infinite voxel world of elements"));
        assert_eq!(m.hint, "Up/Down or j/k select   Enter or l choose");
        let labels: Vec<&str> = m.entries.iter().map(|e| e.label.as_str()).collect();
        assert_eq!(
            labels,
            [
                "New World", "Load: alpha", "Load: beta", "Host Server", "Join Server", "Mods",
                "Settings", "Quit"
            ]
        );
        assert!(m.entries.iter().all(|e| matches!(e.kind, EntryKind::Action)));

        // The index map resolves against the same saves list.
        assert!(matches!(main_choice_at(&saves, 0), MainChoice::NewWorld));
        assert!(matches!(main_choice_at(&saves, 2), MainChoice::Load(n) if n == "beta"));
        assert!(matches!(main_choice_at(&saves, 3), MainChoice::Host));
        assert!(matches!(main_choice_at(&saves, 4), MainChoice::Join));
        assert!(matches!(main_choice_at(&saves, 5), MainChoice::Mods));
        assert!(matches!(main_choice_at(&saves, 6), MainChoice::Settings));
        assert!(matches!(main_choice_at(&saves, 7), MainChoice::Quit));
    }

    #[test]
    fn settings_menu_model_prints_values_like_the_old_screen() {
        let mut s = Settings::default();
        s.msaa = 4;
        s.max_fps = 0;
        s.render_scale = 0.75;
        let m = settings_menu_model(&s);
        assert_eq!(m.title, "SETTINGS");
        assert_eq!(m.hint, "Up/Down or j/k select | Left/Right or h/l change | Esc back");
        let rows: Vec<(String, String)> = m
            .entries
            .iter()
            .filter_map(|e| match &e.kind {
                EntryKind::Cycle { value } => Some((e.label.clone(), value.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(
            rows,
            [
                ("Fullscreen".to_string(), "Off".to_string()),
                ("VSync".to_string(), "Off".to_string()),
                ("MSAA".to_string(), "4x".to_string()),
                ("Max FPS".to_string(), "Uncapped".to_string()),
                ("Render Distance".to_string(), "6".to_string()),
                // f32 Display: whole values print without a decimal point,
                // exactly as `format!("FOV: {}", s.fov)` did.
                ("FOV".to_string(), "70".to_string()),
                ("Render Scale".to_string(), "75%".to_string()),
            ]
        );
        assert!(matches!(m.entries[SETTINGS_ROW_BACK].kind, EntryKind::Action));
        assert_eq!(m.entries[SETTINGS_ROW_BACK].label, "Back");
    }

    #[test]
    fn form_models_keep_the_old_fields_and_carry_values_forward() {
        let host = host_menu_model(None);
        assert_eq!(host.title, "HOST SERVER");
        assert_eq!(host.hint, "Up/Down field   type to edit   Enter start   Esc back");
        assert_eq!(host.text_value(0), DEFAULT_PORT.to_string());
        assert!(matches!(
            host.entries[1].kind,
            EntryKind::Text { masked: true, max: 64, .. }
        ));
        assert_eq!(host.entries[1].label, "Password (optional)");
        assert_eq!(host.text_value(2), "player");

        let join = join_menu_model(None);
        assert_eq!(join.title, "JOIN SERVER");
        assert_eq!(join.hint, "Up/Down field   type to edit   Enter connect   Esc back");
        assert_eq!(join.text_value(0), "127.0.0.1");
        assert_eq!(join.text_value(1), DEFAULT_PORT.to_string());
        assert!(matches!(join.entries[2].kind, EntryKind::Text { masked: true, .. }));
        assert!(matches!(
            join.entries[3].kind,
            EntryKind::Text { max: MAX_NAME, masked: false, .. }
        ));

        // A rebuilt form keeps what the player typed.
        let mut edited = host_menu_model(None);
        if let EntryKind::Text { value, .. } = &mut edited.entries[1].kind {
            *value = "hunter2".to_string();
        }
        edited.cursor = 2;
        let rebuilt = host_menu_model(Some(&edited));
        assert_eq!(rebuilt.text_value(1), "hunter2");
        assert_eq!(rebuilt.cursor, 2);
    }

    // ---- driver: list navigation and events ----

    fn action_menu(n: usize) -> MenuModel {
        MenuModel {
            title: String::new(),
            subtitle: None,
            entries: (0..n).map(|i| MenuEntry::action(format!("e{i}"))).collect(),
            cursor: 0,
            hint: String::new(),
            error: None,
        }
    }

    #[test]
    fn cursor_wraps_both_ways_with_arrows_and_jk() {
        let mut m = action_menu(3);
        assert!(drive(&MenuKeys { down: true, ..Default::default() }, &mut m).is_none());
        assert_eq!(m.cursor, 1);
        drive(&MenuKeys { j: true, ..Default::default() }, &mut m);
        drive(&MenuKeys { j: true, ..Default::default() }, &mut m);
        assert_eq!(m.cursor, 0, "down wraps past the end");
        drive(&MenuKeys { up: true, ..Default::default() }, &mut m);
        assert_eq!(m.cursor, 2, "up wraps past the start");
        drive(&MenuKeys { k: true, ..Default::default() }, &mut m);
        assert_eq!(m.cursor, 1);
    }

    #[test]
    fn enter_and_l_choose_actions_and_esc_backs_out() {
        let mut m = action_menu(3);
        m.cursor = 2;
        assert!(matches!(
            drive(&MenuKeys { enter: true, ..Default::default() }, &mut m),
            Some(MenuEvent::Chosen(2))
        ));
        assert!(matches!(
            drive(&MenuKeys { l: true, ..Default::default() }, &mut m),
            Some(MenuEvent::Chosen(2))
        ));
        // h does nothing on an action list (start-menu parity)…
        assert!(drive(&MenuKeys { h: true, ..Default::default() }, &mut m).is_none());
        // …but Esc always means back.
        assert!(matches!(
            drive(&MenuKeys { esc: true, ..Default::default() }, &mut m),
            Some(MenuEvent::Back)
        ));
        // Activation reads the post-move row.
        m.cursor = 0;
        assert!(matches!(
            drive(&MenuKeys { down: true, enter: true, ..Default::default() }, &mut m),
            Some(MenuEvent::Chosen(1))
        ));
    }

    #[test]
    fn toggle_rows_toggle_on_enter_l_space_and_back_on_h_backspace() {
        let mut m = MenuModel {
            title: String::new(),
            subtitle: None,
            entries: vec![
                MenuEntry::toggle("a", true, ""),
                MenuEntry::toggle("b", false, ""),
            ],
            cursor: 1,
            hint: String::new(),
            error: None,
        };
        for keys in [
            MenuKeys { enter: true, ..Default::default() },
            MenuKeys { l: true, ..Default::default() },
            MenuKeys { space: true, ..Default::default() },
        ] {
            assert!(matches!(drive(&keys, &mut m), Some(MenuEvent::Toggled(1))));
        }
        // The display flipped optimistically each time: false -> true -> false -> true.
        assert!(matches!(m.entries[1].kind, EntryKind::Toggle(true)));
        // Mod-menu parity: h and Backspace also mean back on toggle rows.
        assert!(matches!(
            drive(&MenuKeys { h: true, ..Default::default() }, &mut m),
            Some(MenuEvent::Back)
        ));
        assert!(matches!(
            drive(&MenuKeys { backspace: true, ..Default::default() }, &mut m),
            Some(MenuEvent::Back)
        ));
    }

    #[test]
    fn cycle_rows_step_with_arrows_hl_and_enter() {
        let mut m = MenuModel {
            title: String::new(),
            subtitle: None,
            entries: vec![MenuEntry::cycle("MSAA", "4x")],
            cursor: 0,
            hint: String::new(),
            error: None,
        };
        assert!(matches!(
            drive(&MenuKeys { left: true, ..Default::default() }, &mut m),
            Some(MenuEvent::Cycled(0, -1))
        ));
        assert!(matches!(
            drive(&MenuKeys { h: true, ..Default::default() }, &mut m),
            Some(MenuEvent::Cycled(0, -1))
        ));
        assert!(matches!(
            drive(&MenuKeys { right: true, ..Default::default() }, &mut m),
            Some(MenuEvent::Cycled(0, 1))
        ));
        assert!(matches!(
            drive(&MenuKeys { l: true, ..Default::default() }, &mut m),
            Some(MenuEvent::Cycled(0, 1))
        ));
        assert!(matches!(
            drive(&MenuKeys { enter: true, ..Default::default() }, &mut m),
            Some(MenuEvent::Cycled(0, 1))
        ));
    }

    #[test]
    fn empty_menu_still_backs_out() {
        let mut m = action_menu(0);
        assert!(drive(&MenuKeys { enter: true, ..Default::default() }, &mut m).is_none());
        assert!(matches!(
            drive(&MenuKeys { esc: true, ..Default::default() }, &mut m),
            Some(MenuEvent::Back)
        ));
    }

    // ---- driver: text (form) semantics ----

    #[test]
    fn text_fields_capture_hjkl_as_characters() {
        let mut m = host_menu_model(None);
        m.cursor = 2; // "Your name"
        if let EntryKind::Text { value, .. } = &mut m.entries[2].kind {
            value.clear();
        }
        // j/k/h/l key flags must NOT navigate while typing; the chars land in
        // the field instead.
        let keys = MenuKeys { j: true, chars: vec!['j', 'h'], ..Default::default() };
        assert!(drive(&keys, &mut m).is_none());
        assert_eq!(m.cursor, 2, "j does not move the cursor while on a text field");
        assert_eq!(m.text_value(2), "jh");
    }

    #[test]
    fn text_editing_respects_max_len_control_chars_and_backspace() {
        let mut m = host_menu_model(None); // Port field: max 5, prefilled "5555"
        m.error = Some(PORT_ERROR.to_string());
        drive(&MenuKeys { chars: vec!['9', '9'], ..Default::default() }, &mut m);
        assert_eq!(m.text_value(0), "55559", "the byte cap holds at 5");
        assert_eq!(m.error, None, "any editing keystroke clears the error");

        m.error = Some(PORT_ERROR.to_string());
        drive(&MenuKeys { backspace: true, ..Default::default() }, &mut m);
        assert_eq!(m.text_value(0), "5555");
        assert_eq!(m.error, None, "backspace clears the error too");

        drive(&MenuKeys { chars: vec!['\u{8}', '\t'], ..Default::default() }, &mut m);
        assert_eq!(m.text_value(0), "5555", "control characters are rejected");
    }

    #[test]
    fn form_navigation_uses_tab_and_arrows_and_enter_submits() {
        let mut m = join_menu_model(None);
        drive(&MenuKeys { tab: true, ..Default::default() }, &mut m);
        assert_eq!(m.cursor, 1);
        drive(&MenuKeys { down: true, ..Default::default() }, &mut m);
        assert_eq!(m.cursor, 2);
        drive(&MenuKeys { up: true, ..Default::default() }, &mut m);
        assert_eq!(m.cursor, 1);
        // Tab wraps like Down.
        m.cursor = 3;
        drive(&MenuKeys { tab: true, ..Default::default() }, &mut m);
        assert_eq!(m.cursor, 0);
        assert!(matches!(
            drive(&MenuKeys { enter: true, ..Default::default() }, &mut m),
            Some(MenuEvent::Submit)
        ));
        assert!(matches!(
            drive(&MenuKeys { esc: true, ..Default::default() }, &mut m),
            Some(MenuEvent::Back)
        ));
    }

    // ---- settings meaning stays core-side ----

    #[test]
    fn settings_cycle_wraps_every_row() {
        let mut s = Settings::default();
        apply_settings_cycle(&mut s, 0, 1);
        assert!(s.fullscreen);
        apply_settings_cycle(&mut s, 2, -1);
        assert_eq!(s.msaa, 8, "msaa wraps 1 -> 8 going left");
        s.render_distance = 10;
        apply_settings_cycle(&mut s, 4, 1);
        assert_eq!(s.render_distance, 3, "render distance wraps 10 -> 3");
        s.fov = 50.0;
        apply_settings_cycle(&mut s, 5, -1);
        assert_eq!(s.fov, 110.0, "fov wraps 50 -> 110");
        // The Back row cycles to nothing.
        let before = s.clone();
        apply_settings_cycle(&mut s, SETTINGS_ROW_BACK, 1);
        assert_eq!(s, before);
    }

    #[test]
    fn cycle_list_wraps_and_snaps_off_list_values() {
        assert_eq!(cycle_list(&[1, 2, 4, 8], 4, 1), 8);
        assert_eq!(cycle_list(&[1, 2, 4, 8], 8, 1), 1);
        assert_eq!(cycle_list(&[1, 2, 4, 8], 1, -1), 8);
        // Off-list snaps to the first entry without stepping.
        assert_eq!(cycle_list(&[25, 50, 100], 60, 1), 25);
    }
}
