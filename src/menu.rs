//! The out-of-game screens as plain data models: the start menu (new / load /
//! host / join / mods / settings / quit), the mod list, the graphics settings,
//! and the host/join forms.
//!
//! A menu is just a list of choices. Core owns the data and what each choice
//! means — a [`MenuModel`] is built per screen by the functions below, and
//! [`crate::app`] interprets the [`MenuEvent`]s that come back. The visuals
//! and input handling belong to mods (see
//! [`menu_default`](crate::mods::menu_default), the default "Menus" mod),
//! exactly like the inventory and crafting mods: default-enabled, disableable,
//! replaceable.
//!
//! So that disabling the Menus mod can never brick navigation, this module
//! also keeps a built-in fallback driver and renderer
//! ([`fallback_drive`]/[`fallback_draw`]). The default mod is a thin wrapper
//! around the same free functions ([`drive`]/[`draw_model`]) — one
//! implementation, two entry points.
//!
//! Every invariant lives in a type, not in the driver: [`Field`] owns its byte
//! cap and caret, [`Entries`] owns a cursor that is always on a real row, and
//! [`Layout`] names the render treatment instead of leaving it to be inferred
//! from incidental model shape. An [`EntryKind`] decides which [`MenuEvent`] a
//! row can ever emit, and an Action row carries a typed [`ActionId`] so meaning
//! never rides on a fragile row-index ladder.
use voxel_engine::{Color, Engine, Frame, Key};

use crate::console::shadowed;
use crate::mods::Mods;
use crate::net::{DEFAULT_PORT, MAX_NAME};
use crate::settings::{SETTINGS, Settings};
/// The Back row index on the settings screen, re-exported from the one settings
/// table so `menu::SETTINGS_ROW_BACK` keeps resolving for callers.
pub use crate::settings::SETTINGS_ROW_BACK;

// ---------------------------------------------------------------------------
// Primitives — each owns exactly one value and its invariant.
// ---------------------------------------------------------------------------

/// One step through a cycle/stepper: exactly two directions, so no `+5` or `0`
/// can ever be spelled (the old `i32` admitted both).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Step {
    Prev,
    Next,
}

impl Step {
    /// The `-1`/`+1` the settings stepper still speaks internally.
    pub fn delta(self) -> i32 {
        match self {
            Step::Prev => -1,
            Step::Next => 1,
        }
    }
}

/// A bounded, caret-tracked text field. Owns the two invariants the driver used
/// to re-check at every edit site: `text.len() <= cap` (a byte cap, enforced on
/// whole chars) and `caret <= char_count` (a char index). Every mutation goes
/// through these methods, so no caller can break either.
pub struct Field {
    text: String,
    cap: usize,
    /// Caret position as a CHAR index in `0..=char_count`.
    caret: usize,
}

impl Field {
    fn new(init: &str, cap: usize) -> Self {
        let mut f = Self { text: String::new(), cap, caret: 0 };
        f.set(init);
        f
    }

    pub fn as_str(&self) -> &str {
        &self.text
    }

    /// The byte cap — what a renderer or test reads to know the field's limit.
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// Caret position as a char index (for drawing the cursor mid-string).
    pub fn caret(&self) -> usize {
        self.caret
    }

    fn char_count(&self) -> usize {
        self.text.chars().count()
    }

    /// Byte offset of char index `at` (== `text.len()` at the end).
    fn byte_at(&self, at: usize) -> usize {
        self.text.char_indices().nth(at).map_or(self.text.len(), |(b, _)| b)
    }

    /// Replace the whole value, truncated to the cap on a char boundary; caret
    /// to the end. Empty input is a real clear (callers that want "keep the
    /// default" guard emptiness themselves — see [`MenuModel::set_text`]).
    fn set(&mut self, value: &str) {
        let mut s = value.to_string();
        while s.len() > self.cap {
            s.pop();
        }
        self.caret = s.chars().count();
        self.text = s;
    }

    /// Insert one char at the caret if it is printable and still fits the cap.
    fn insert(&mut self, c: char) -> bool {
        if c.is_control() || self.text.len() + c.len_utf8() > self.cap {
            return false;
        }
        let at = self.byte_at(self.caret);
        self.text.insert(at, c);
        self.caret += 1;
        true
    }

    /// Delete the char before the caret.
    fn backspace(&mut self) -> bool {
        if self.caret == 0 {
            return false;
        }
        let at = self.byte_at(self.caret - 1);
        self.text.remove(at);
        self.caret -= 1;
        true
    }

    fn left(&mut self) {
        self.caret = self.caret.saturating_sub(1);
    }

    fn right(&mut self) {
        self.caret = (self.caret + 1).min(self.char_count());
    }
}

/// A titled list of entries plus a cursor that is always on a real row (when
/// the list is non-empty). Movement is the two wrapping operations `next`/`prev`
/// — the driver never open-codes `% n` again.
pub struct Entries {
    pub items: Vec<MenuEntry>,
    pub cursor: usize,
}

impl Entries {
    fn new(items: Vec<MenuEntry>) -> Self {
        Self { items, cursor: 0 }
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Keep the cursor on a real entry after the list shrank (save deleted).
    fn clamp(&mut self) {
        self.cursor = self.cursor.min(self.items.len().saturating_sub(1));
    }

    fn next(&mut self) {
        let n = self.items.len();
        if n > 0 {
            self.cursor = (self.cursor + 1) % n;
        }
    }

    fn prev(&mut self) {
        let n = self.items.len();
        if n > 0 {
            self.cursor = (self.cursor + n - 1) % n;
        }
    }

    fn iter(&self) -> std::slice::Iter<'_, MenuEntry> {
        self.items.iter()
    }
}

/// Which of two colours a transient message wants.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Level {
    /// A soft status line (a failed connect on the start menu) — salmon.
    Info,
    /// A refused action (a bad port) — red.
    Error,
}

/// A transient one-line message, drawn just above the hint until the next
/// editing keystroke clears it. One slot, one renderer — it replaced the split
/// `error` field and the separate start-menu status line.
pub struct Notice {
    pub text: String,
    pub level: Level,
}

impl Notice {
    pub fn info(text: String) -> Self {
        Self { text, level: Level::Info }
    }

    pub fn error(text: String) -> Self {
        Self { text, level: Level::Error }
    }
}

/// How a screen is laid out — named up front instead of sniffed from the model
/// (subtitle-present ⇒ big title, any-text-field ⇒ form). Making the choice a
/// value means a form *with* a toggle is expressible, and a subtitle can only
/// exist where it is actually shown.
pub enum Layout {
    /// The start-menu look: big gold title, a subtitle, centred rows.
    Title { subtitle: String },
    /// The compact left-aligned list (mods, settings).
    List,
    /// The `label: value` form with caret editing (host, join).
    Form,
}

// ---------------------------------------------------------------------------
// The model (frozen shapes — see ENGINE_DESIGN R6.1).
// ---------------------------------------------------------------------------

/// A stable id an [`EntryKind::Action`] row carries so its meaning never rides
/// on the row's position. The owning screen assigns the ids and maps them back
/// (see [`main_choice_at`]).
pub type ActionId = u32;

/// One whole menu screen: a titled list of entries with a cursor, a chosen
/// [`Layout`], a key-hint line, and an optional [`Notice`]. Everything a
/// renderer needs — and nothing about what the entries mean.
pub struct MenuModel {
    pub title: String,
    pub layout: Layout,
    pub entries: Entries,
    /// The key-hint line at the bottom of the screen.
    pub hint: String,
    /// A transient message (bad port, failed connect). Cleared by the driver on
    /// the next editing keystroke.
    pub notice: Option<Notice>,
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
    /// Activating it picks it ([`MenuEvent::Chosen`]), carrying its [`ActionId`].
    Action(ActionId),
    /// An on/off switch ([`MenuEvent::Toggled`]); the bool is the shown state.
    Toggle(bool),
    /// A value stepped left/right through a list ([`MenuEvent::Cycled`]); the
    /// string is the current display value.
    Cycle { value: String },
    /// An editable text field; the driver types into its [`Field`]. `masked`
    /// renders it as `*`s (passwords).
    Text { field: Field, masked: bool },
}

/// What the player did to a menu, as plain terms. The owner of the model (the
/// App) turns these back into meaning.
pub enum MenuEvent {
    /// An [`EntryKind::Action`] entry was activated (carries its [`ActionId`]).
    Chosen(ActionId),
    /// An [`EntryKind::Toggle`] entry was flipped (carries its row index).
    Toggled(usize),
    /// An [`EntryKind::Cycle`] entry was stepped (row index + direction).
    Cycled(usize, Step),
    /// Leave this screen.
    Back,
    /// Submit the whole form (Enter while on a text field).
    Submit,
}

impl MenuEntry {
    pub fn action(label: impl Into<String>, id: ActionId) -> Self {
        Self { label: label.into(), detail: None, kind: EntryKind::Action(id) }
    }

    pub fn toggle(label: impl Into<String>, on: bool, detail: impl Into<String>) -> Self {
        Self { label: label.into(), detail: Some(detail.into()), kind: EntryKind::Toggle(on) }
    }

    pub fn cycle(label: impl Into<String>, value: impl Into<String>) -> Self {
        Self { label: label.into(), detail: None, kind: EntryKind::Cycle { value: value.into() } }
    }

    pub fn text(label: impl Into<String>, value: &str, max: usize, masked: bool) -> Self {
        Self {
            label: label.into(),
            detail: None,
            kind: EntryKind::Text { field: Field::new(value, max), masked },
        }
    }
}

impl MenuModel {
    /// The value of the text field at `index` (empty for non-text entries) —
    /// how the App reads a submitted form back out of the model.
    pub fn text_value(&self, index: usize) -> &str {
        match self.entries.items.get(index).map(|e| &e.kind) {
            Some(EntryKind::Text { field, .. }) => field.as_str(),
            _ => "",
        }
    }

    /// Pre-fill the text field at `index`. No-op for a non-text or out-of-range
    /// entry, or an empty value (so a blank remembered field leaves the form's
    /// own default in place).
    pub fn set_text(&mut self, index: usize, value: &str) {
        if value.is_empty() {
            return;
        }
        if let Some(entry) = self.entries.items.get_mut(index)
            && let EntryKind::Text { field, .. } = &mut entry.kind
        {
            field.set(value);
        }
    }

    /// Keep the cursor on a real entry after the list shrank (save deleted).
    pub fn clamp_cursor(&mut self) {
        self.entries.clamp();
    }
}

// ---------------------------------------------------------------------------
// Model builders — one per screen. The label/detail/value strings here are
// byte-identical to what the old concrete menus rendered.
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

// Start-menu action ids. Fixed rows get distinct small ids; Load rows get
// `LOAD_BASE + save_index`, a reserved range that can't collide with them.
const ID_NEW_WORLD: ActionId = 1;
const ID_HOST: ActionId = 2;
const ID_JOIN: ActionId = 3;
const ID_MODS: ActionId = 4;
const ID_SETTINGS: ActionId = 5;
const ID_QUIT: ActionId = 6;
const LOAD_BASE: ActionId = 1000;
/// Any Action id whose meaning is just "leave this screen" (the settings Back
/// row). The owning screen treats every `Chosen` as Back, so the value is only
/// a placeholder.
const ID_BACK: ActionId = 0;

/// The fixed choices that trail the dynamic Load rows on the start menu, in row
/// order — the single source of truth for that ordering. [`main_menu_model`]
/// renders each as an Action row tagged with its id, and [`main_choice_at`] maps
/// that id straight back, so a row can never misroute regardless of position.
const MAIN_TRAILING: [(ActionId, &str, fn() -> MainChoice); 5] = [
    (ID_HOST, "Host Server", || MainChoice::Host),
    (ID_JOIN, "Join Server", || MainChoice::Join),
    (ID_MODS, "Mods", || MainChoice::Mods),
    (ID_SETTINGS, "Settings", || MainChoice::Settings),
    (ID_QUIT, "Quit", || MainChoice::Quit),
];

/// The start menu: New World, one Load row per save, Host, Join, Mods,
/// Settings, Quit.
pub fn main_menu_model(saves: &[String]) -> MenuModel {
    let mut entries = vec![MenuEntry::action("New World", ID_NEW_WORLD)];
    for (i, name) in saves.iter().enumerate() {
        entries.push(MenuEntry::action(format!("Load: {name}"), LOAD_BASE + i as ActionId));
    }
    for (id, label, _) in MAIN_TRAILING {
        entries.push(MenuEntry::action(label, id));
    }
    MenuModel {
        title: "PROJECT WATT CUBED".to_string(),
        layout: Layout::Title { subtitle: "an infinite voxel world of elements".to_string() },
        entries: Entries::new(entries),
        hint: "Up/Down or j/k select   Enter or l choose".to_string(),
        notice: None,
    }
}

/// Resolve a start-menu [`MenuEvent::Chosen`] id back into a [`MainChoice`],
/// against the same saves list the model was built from. An unknown id falls
/// through to `Quit`, as before.
pub fn main_choice_at(saves: &[String], id: ActionId) -> MainChoice {
    if id == ID_NEW_WORLD {
        return MainChoice::NewWorld;
    }
    if id >= LOAD_BASE {
        let i = (id - LOAD_BASE) as usize;
        if i < saves.len() {
            return MainChoice::Load(saves[i].clone());
        }
    }
    MAIN_TRAILING
        .iter()
        .find(|(tid, _, _)| *tid == id)
        .map_or(MainChoice::Quit, |(_, _, make)| make())
}

/// The mod menu: one Toggle row per installed mod, description as the detail.
pub fn mods_menu_model(mods: &Mods) -> MenuModel {
    let entries = (0..mods.len())
        .map(|i| MenuEntry::toggle(mods.name(i), mods.is_enabled(i), mods.description(i)))
        .collect();
    MenuModel {
        title: "MODS".to_string(),
        layout: Layout::List,
        entries: Entries::new(entries),
        hint: "Up/Down or j/k select   Enter/l toggle   Esc/h back".to_string(),
        notice: None,
    }
}

/// The settings menu: one Cycle row per graphics option (from the one
/// [`SETTINGS`] table) plus a Back action. Values are formatted by each field's
/// own `show`, so the screen can never drift from the console or persistence.
pub fn settings_menu_model(s: &Settings) -> MenuModel {
    let mut entries: Vec<MenuEntry> =
        SETTINGS.iter().map(|f| MenuEntry::cycle(f.label(), f.show(s))).collect();
    entries.push(MenuEntry::action("Back", ID_BACK));
    MenuModel {
        title: "SETTINGS".to_string(),
        layout: Layout::List,
        entries: Entries::new(entries),
        hint: "Up/Down or j/k select | Left/Right or h/l change | Esc back".to_string(),
        notice: None,
    }
}

/// Apply one Left/Right step to a settings row, wrapping — what a
/// [`MenuEvent::Cycled`] on the settings screen actually does, kept App-side so
/// no mod ever decides what "MSAA" means. The Back row (and any out-of-range
/// index) is a no-op.
pub fn apply_settings_cycle(s: &mut Settings, row: usize, step: Step) {
    if let Some(field) = SETTINGS.get(row) {
        field.step(s, step.delta());
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
        layout: Layout::Form,
        entries: Entries::new(vec![
            MenuEntry::text("Port", &DEFAULT_PORT.to_string(), 5, false),
            MenuEntry::text("Password (optional)", "", 64, true),
            MenuEntry::text("Your name", "player", MAX_NAME, false),
        ]),
        hint: "Up/Down field   type to edit   Enter start   Esc back".to_string(),
        notice: None,
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
        layout: Layout::Form,
        entries: Entries::new(vec![
            MenuEntry::text("Address", "127.0.0.1", 64, false),
            MenuEntry::text("Port", &DEFAULT_PORT.to_string(), 5, false),
            MenuEntry::text("Password", "", 64, true),
            MenuEntry::text("Your name", "player", MAX_NAME, false),
        ]),
        hint: "Up/Down field   type to edit   Enter connect   Esc back".to_string(),
        notice: None,
    };
    if let Some(prior) = prior {
        carry_text_values(prior, &mut model);
    }
    model
}

/// Copy text-field values from a previous incarnation of the same form,
/// positionally, so a rebuilt model keeps what the player typed.
fn carry_text_values(prior: &MenuModel, model: &mut MenuModel) {
    for (old, new) in prior.entries.iter().zip(model.entries.items.iter_mut()) {
        if let (EntryKind::Text { field: from, .. }, EntryKind::Text { field: to, .. }) =
            (&old.kind, &mut new.kind)
        {
            to.set(from.as_str());
        }
    }
    model.entries.cursor = prior.entries.cursor.min(model.entries.len().saturating_sub(1));
}

/// The error shown when a submitted port doesn't parse.
pub const PORT_ERROR: &str = "invalid port (1-65535)";

/// Parse a port field. An empty field keeps meaning [`DEFAULT_PORT`] — the
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
///   [`MenuEvent::Cycled`]`(Next)`.
/// - Left/Right (and h/l) step Cycle entries.
/// - Esc -> [`MenuEvent::Back`]; on Toggle rows h and Backspace too
///   (mod-menu parity). On Action rows h does nothing (start-menu parity).
/// - While the cursor is on a Text entry the form rules apply instead:
///   Down/Tab and Up move fields, Left/Right move the caret, typed chars
///   (including hjkl) go into the field (byte cap, control chars rejected),
///   Backspace deletes, Enter submits the whole form, Esc cancels. Editing
///   clears `model.notice`.
///
/// It never interprets meaning — that stays with the App.
pub fn drive(keys: &MenuKeys, menu: &mut MenuModel) -> Option<MenuEvent> {
    if menu.entries.is_empty() {
        // Only an emptied mod list can get here; every back alias still works.
        return (keys.esc || keys.h || keys.backspace).then_some(MenuEvent::Back);
    }
    menu.entries.clamp();

    if matches!(menu.entries.items[menu.entries.cursor].kind, EntryKind::Text { .. }) {
        return drive_text(keys, menu);
    }

    // List navigation first, so activation reads the post-move row (holding
    // Down and tapping Enter picks what the highlight shows).
    if keys.down || keys.j {
        menu.entries.next();
    }
    if keys.up || keys.k {
        menu.entries.prev();
    }
    let cursor = menu.entries.cursor;
    match &mut menu.entries.items[cursor].kind {
        EntryKind::Action(id) => {
            if keys.enter || keys.l {
                return Some(MenuEvent::Chosen(*id));
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
                return Some(MenuEvent::Cycled(cursor, Step::Prev));
            }
            if keys.right || keys.l || keys.enter {
                return Some(MenuEvent::Cycled(cursor, Step::Next));
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
fn drive_text(keys: &MenuKeys, menu: &mut MenuModel) -> Option<MenuEvent> {
    if keys.down || keys.tab {
        menu.entries.next();
    }
    if keys.up {
        menu.entries.prev();
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
    let edited = keys.backspace || !keys.chars.is_empty();
    if let EntryKind::Text { field, .. } = &mut menu.entries.items[menu.entries.cursor].kind {
        if keys.left {
            field.left();
        }
        if keys.right {
            field.right();
        }
        if keys.backspace {
            field.backspace();
        }
        for &c in &keys.chars {
            field.insert(c);
        }
    }
    if edited {
        menu.notice = None;
    }
    None
}

// ---------------------------------------------------------------------------
// The renderer: the current visuals, driven entirely by the model. The layout
// is read straight off [`MenuModel::layout`]; the big-title screen is its own
// function, and the list/form screens share one body (they differ only in a few
// [`Panel`] metrics).
// ---------------------------------------------------------------------------

/// Background for every menu screen.
const MENU_BG: Color = Color::new(18, 20, 28, 255);

/// Draw a menu model in the standard style. Free function so the default
/// "Menus" mod and the core fallback share one implementation.
pub fn draw_model(f: &mut Frame, menu: &MenuModel, w: i32, h: i32) {
    f.draw_rect(0, 0, w, h, MENU_BG);
    match &menu.layout {
        Layout::Title { subtitle } => draw_title_style(f, menu, subtitle, w, h),
        Layout::List => draw_panel(f, menu, &Panel::list(w, h), w, h),
        Layout::Form => draw_panel(f, menu, &Panel::form(w, h), w, h),
    }
}

/// The start-menu look: 48px gold title at h/6, gray subtitle, centered rows.
fn draw_title_style(f: &mut Frame, menu: &MenuModel, subtitle: &str, w: i32, h: i32) {
    let title_fs = 48;
    let tx = (w - f.measure_text(&menu.title, title_fs)) / 2;
    shadowed(f, &menu.title, tx, h / 6, title_fs, Color::GOLD);

    let sub_fs = 20;
    let sx = (w - f.measure_text(subtitle, sub_fs)) / 2;
    shadowed(f, subtitle, sx, h / 6 + title_fs + 8, sub_fs, Color::GRAY);

    let fs = 28;
    let line_h = fs + 14;
    let start_y = h / 2 - line_h;
    for (i, entry) in menu.entries.iter().enumerate() {
        let selected = i == menu.entries.cursor;
        let text = if selected { format!("> {}", entry.label) } else { format!("  {}", entry.label) };
        let color = if selected { Color::RAYWHITE } else { Color::GRAY };
        let x = (w - f.measure_text(&text, fs)) / 2;
        shadowed(f, &text, x, start_y + line_h * i as i32, fs, color);
    }

    draw_notice(f, menu, w, h);
    draw_hint(f, &menu.hint, w, h);
}

/// The metrics that separate the compact list from the form. Everything else
/// (row formatting, selection, notice, hint) is shared by [`draw_panel`].
struct Panel {
    title_fs: i32,
    title_y: i32,
    fs: i32,
    line_h: i32,
    start_y: i32,
    x: i32,
}

impl Panel {
    fn list(w: i32, h: i32) -> Self {
        let fs = 26;
        Self { title_fs: 40, title_y: h / 8, fs, line_h: fs + 20, start_y: h / 4 + 20, x: w / 2 - 260 }
    }

    fn form(w: i32, h: i32) -> Self {
        let fs = 26;
        let line_h = fs + 22;
        Self { title_fs: 40, title_y: h / 6, fs, line_h, start_y: h / 2 - line_h, x: w / 2 - 240 }
    }
}

/// The list/form body: gold title, one `row_text` per entry (with the mod
/// description detail line), then the shared notice + hint.
fn draw_panel(f: &mut Frame, menu: &MenuModel, p: &Panel, w: i32, h: i32) {
    let tx = (w - f.measure_text(&menu.title, p.title_fs)) / 2;
    shadowed(f, &menu.title, tx, p.title_y, p.title_fs, Color::GOLD);

    if menu.entries.is_empty() {
        // The only empty menu in the game is a modless mod list.
        shadowed(f, "  (no mods installed)", p.x, p.start_y, p.fs, Color::GRAY);
    }

    for (i, entry) in menu.entries.iter().enumerate() {
        let selected = i == menu.entries.cursor;
        let y = p.start_y + p.line_h * i as i32;
        let color = if selected { Color::RAYWHITE } else { Color::GRAY };
        shadowed(f, &row_text(entry, selected), p.x, y, p.fs, color);
        if let Some(detail) = &entry.detail {
            shadowed(f, detail, p.x + 40, y + p.fs + 2, 16, Color::DARKGRAY);
        }
    }

    draw_notice(f, menu, w, h);
    draw_hint(f, &menu.hint, w, h);
}

/// One entry as a single line: the selection mark plus a kind-specific body.
/// Cycles show `< value >` steppers; the selected text field shows a `_` caret
/// at its current position (mid-string, not just at the end).
fn row_text(entry: &MenuEntry, selected: bool) -> String {
    let mark = if selected { ">" } else { " " };
    match &entry.kind {
        EntryKind::Action(_) => format!("{} {}", mark, entry.label),
        EntryKind::Toggle(on) => {
            format!("{} {} {}", mark, if *on { "[x]" } else { "[ ]" }, entry.label)
        }
        EntryKind::Cycle { value } => format!("{} {}: < {} >", mark, entry.label, value),
        EntryKind::Text { field, masked } => {
            let shown: Vec<char> = if *masked {
                std::iter::repeat_n('*', field.as_str().chars().count()).collect()
            } else {
                field.as_str().chars().collect()
            };
            let body: String = if selected {
                let at = field.caret().min(shown.len());
                shown[..at].iter().chain(&['_']).chain(&shown[at..]).collect()
            } else {
                shown.into_iter().collect()
            };
            format!("{} {}: {}", mark, entry.label, body)
        }
    }
}

/// The key-hint line every screen shows at the bottom.
fn draw_hint(f: &mut Frame, hint: &str, w: i32, h: i32) {
    let hint_fs = 18;
    let hx = (w - f.measure_text(hint, hint_fs)) / 2;
    shadowed(f, hint, hx, h - 40, hint_fs, Color::DARKGRAY);
}

/// The one transient-message renderer: just above the hint, coloured by level,
/// until the next keystroke clears it.
fn draw_notice(f: &mut Frame, menu: &MenuModel, w: i32, h: i32) {
    if let Some(notice) = &menu.notice {
        let fs = 18;
        let color = match notice.level {
            Level::Info => Color::SALMON,
            Level::Error => Color::RED,
        };
        let x = (w - f.measure_text(&notice.text, fs)) / 2;
        shadowed(f, &notice.text, x, h - 40 - (fs + 8), fs, color);
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

    // ---- primitives ----

    #[test]
    fn field_enforces_cap_on_char_boundaries_and_tracks_caret() {
        let mut fld = Field::new("", 4);
        assert!(fld.insert('a'));
        assert!(fld.insert('b'));
        assert_eq!(fld.as_str(), "ab");
        assert_eq!(fld.caret(), 2);
        // Control chars are rejected without moving the caret.
        assert!(!fld.insert('\n'));
        assert_eq!(fld.caret(), 2);
        // A 2-byte char that would overflow the byte cap is refused atomically.
        assert!(fld.insert('é')); // "abé" == 4 bytes, fits exactly
        assert!(!fld.insert('x')); // 5 bytes, refused
        assert_eq!(fld.as_str(), "abé");
        // Caret movement + mid-string insert.
        fld.left();
        fld.left();
        assert_eq!(fld.caret(), 1);
        fld.backspace();
        assert_eq!(fld.as_str(), "bé");
    }

    // ---- model builders pin today's exact strings ----

    #[test]
    fn main_menu_model_matches_the_old_screen_exactly() {
        let saves = vec!["alpha".to_string(), "beta".to_string()];
        let m = main_menu_model(&saves);
        assert_eq!(m.title, "PROJECT WATT CUBED");
        assert!(matches!(&m.layout, Layout::Title { subtitle } if subtitle == "an infinite voxel world of elements"));
        assert_eq!(m.hint, "Up/Down or j/k select   Enter or l choose");
        let labels: Vec<&str> = m.entries.iter().map(|e| e.label.as_str()).collect();
        assert_eq!(
            labels,
            [
                "New World", "Load: alpha", "Load: beta", "Host Server", "Join Server", "Mods",
                "Settings", "Quit"
            ]
        );
        assert!(m.entries.iter().all(|e| matches!(e.kind, EntryKind::Action(_))));

        // Each row's own id resolves to the right meaning, regardless of index.
        let choice = |i: usize| {
            let EntryKind::Action(id) = m.entries.items[i].kind else { panic!("not an action") };
            main_choice_at(&saves, id)
        };
        assert!(matches!(choice(0), MainChoice::NewWorld));
        assert!(matches!(choice(2), MainChoice::Load(n) if n == "beta"));
        assert!(matches!(choice(3), MainChoice::Host));
        assert!(matches!(choice(4), MainChoice::Join));
        assert!(matches!(choice(5), MainChoice::Mods));
        assert!(matches!(choice(6), MainChoice::Settings));
        assert!(matches!(choice(7), MainChoice::Quit));
        // An unknown id still falls through to Quit.
        assert!(matches!(main_choice_at(&saves, 9999), MainChoice::Quit));
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
                ("Lighting".to_string(), "On".to_string()),
                ("MSAA".to_string(), "4x".to_string()),
                ("Max FPS".to_string(), "Uncapped".to_string()),
                ("Render Distance".to_string(), "6".to_string()),
                // f32 Display: whole values print without a decimal point,
                // exactly as `format!("FOV: {}", s.fov)` did.
                ("FOV".to_string(), "90".to_string()),
                ("Render Scale".to_string(), "75%".to_string()),
                ("UI Scale".to_string(), "100%".to_string()),
            ]
        );
        assert!(matches!(m.entries.items[SETTINGS_ROW_BACK].kind, EntryKind::Action(_)));
        assert_eq!(m.entries.items[SETTINGS_ROW_BACK].label, "Back");
    }

    #[test]
    fn form_models_keep_the_old_fields_and_carry_values_forward() {
        let host = host_menu_model(None);
        assert_eq!(host.title, "HOST SERVER");
        assert_eq!(host.hint, "Up/Down field   type to edit   Enter start   Esc back");
        assert_eq!(host.text_value(0), DEFAULT_PORT.to_string());
        assert!(matches!(&host.entries.items[1].kind, EntryKind::Text { field, masked: true } if field.cap() == 64));
        assert_eq!(host.entries.items[1].label, "Password (optional)");
        assert_eq!(host.text_value(2), "player");

        let join = join_menu_model(None);
        assert_eq!(join.title, "JOIN SERVER");
        assert_eq!(join.hint, "Up/Down field   type to edit   Enter connect   Esc back");
        assert_eq!(join.text_value(0), "127.0.0.1");
        assert_eq!(join.text_value(1), DEFAULT_PORT.to_string());
        assert!(matches!(join.entries.items[2].kind, EntryKind::Text { masked: true, .. }));
        assert!(matches!(&join.entries.items[3].kind, EntryKind::Text { field, masked: false } if field.cap() == MAX_NAME));

        // A rebuilt form keeps what the player typed.
        let mut edited = host_menu_model(None);
        edited.set_text(1, "hunter2");
        edited.entries.cursor = 2;
        let rebuilt = host_menu_model(Some(&edited));
        assert_eq!(rebuilt.text_value(1), "hunter2");
        assert_eq!(rebuilt.entries.cursor, 2);
    }

    // ---- driver: list navigation and events ----

    fn action_menu(n: usize) -> MenuModel {
        MenuModel {
            title: String::new(),
            layout: Layout::List,
            entries: Entries::new((0..n).map(|i| MenuEntry::action(format!("e{i}"), i as ActionId)).collect()),
            hint: String::new(),
            notice: None,
        }
    }

    #[test]
    fn cursor_wraps_both_ways_with_arrows_and_jk() {
        let mut m = action_menu(3);
        assert!(drive(&MenuKeys { down: true, ..Default::default() }, &mut m).is_none());
        assert_eq!(m.entries.cursor, 1);
        drive(&MenuKeys { j: true, ..Default::default() }, &mut m);
        drive(&MenuKeys { j: true, ..Default::default() }, &mut m);
        assert_eq!(m.entries.cursor, 0, "down wraps past the end");
        drive(&MenuKeys { up: true, ..Default::default() }, &mut m);
        assert_eq!(m.entries.cursor, 2, "up wraps past the start");
        drive(&MenuKeys { k: true, ..Default::default() }, &mut m);
        assert_eq!(m.entries.cursor, 1);
    }

    #[test]
    fn enter_and_l_choose_actions_and_esc_backs_out() {
        let mut m = action_menu(3);
        m.entries.cursor = 2;
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
        m.entries.cursor = 0;
        assert!(matches!(
            drive(&MenuKeys { down: true, enter: true, ..Default::default() }, &mut m),
            Some(MenuEvent::Chosen(1))
        ));
    }

    #[test]
    fn toggle_rows_toggle_on_enter_l_space_and_back_on_h_backspace() {
        let mut m = MenuModel {
            title: String::new(),
            layout: Layout::List,
            entries: Entries::new(vec![
                MenuEntry::toggle("a", true, ""),
                MenuEntry::toggle("b", false, ""),
            ]),
            hint: String::new(),
            notice: None,
        };
        m.entries.cursor = 1;
        for keys in [
            MenuKeys { enter: true, ..Default::default() },
            MenuKeys { l: true, ..Default::default() },
            MenuKeys { space: true, ..Default::default() },
        ] {
            assert!(matches!(drive(&keys, &mut m), Some(MenuEvent::Toggled(1))));
        }
        // The display flipped optimistically each time: false -> true -> false -> true.
        assert!(matches!(m.entries.items[1].kind, EntryKind::Toggle(true)));
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
            layout: Layout::List,
            entries: Entries::new(vec![MenuEntry::cycle("MSAA", "4x")]),
            hint: String::new(),
            notice: None,
        };
        assert!(matches!(
            drive(&MenuKeys { left: true, ..Default::default() }, &mut m),
            Some(MenuEvent::Cycled(0, Step::Prev))
        ));
        assert!(matches!(
            drive(&MenuKeys { h: true, ..Default::default() }, &mut m),
            Some(MenuEvent::Cycled(0, Step::Prev))
        ));
        assert!(matches!(
            drive(&MenuKeys { right: true, ..Default::default() }, &mut m),
            Some(MenuEvent::Cycled(0, Step::Next))
        ));
        assert!(matches!(
            drive(&MenuKeys { l: true, ..Default::default() }, &mut m),
            Some(MenuEvent::Cycled(0, Step::Next))
        ));
        assert!(matches!(
            drive(&MenuKeys { enter: true, ..Default::default() }, &mut m),
            Some(MenuEvent::Cycled(0, Step::Next))
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
        m.entries.cursor = 2; // "Your name"
        m.set_text(2, "player");
        if let EntryKind::Text { field, .. } = &mut m.entries.items[2].kind {
            *field = Field::new("", MAX_NAME);
        }
        // j/k/h/l key flags must NOT navigate while typing; the chars land in
        // the field instead.
        let keys = MenuKeys { j: true, chars: vec!['j', 'h'], ..Default::default() };
        assert!(drive(&keys, &mut m).is_none());
        assert_eq!(m.entries.cursor, 2, "j does not move the cursor while on a text field");
        assert_eq!(m.text_value(2), "jh");
    }

    #[test]
    fn text_editing_respects_max_len_control_chars_and_backspace() {
        let mut m = host_menu_model(None); // Port field: max 5, prefilled "5555"
        m.notice = Some(Notice::error(PORT_ERROR.to_string()));
        drive(&MenuKeys { chars: vec!['9', '9'], ..Default::default() }, &mut m);
        assert_eq!(m.text_value(0), "55559", "the byte cap holds at 5");
        assert!(m.notice.is_none(), "any editing keystroke clears the notice");

        m.notice = Some(Notice::error(PORT_ERROR.to_string()));
        drive(&MenuKeys { backspace: true, ..Default::default() }, &mut m);
        assert_eq!(m.text_value(0), "5555");
        assert!(m.notice.is_none(), "backspace clears the notice too");

        drive(&MenuKeys { chars: vec!['\u{8}', '\t'], ..Default::default() }, &mut m);
        assert_eq!(m.text_value(0), "5555", "control characters are rejected");
    }

    #[test]
    fn caret_moves_and_inserts_mid_string() {
        let mut m = host_menu_model(None); // Port "5555"
        // Move the caret two left, then type: inserts in the middle.
        drive(&MenuKeys { left: true, ..Default::default() }, &mut m);
        drive(&MenuKeys { left: true, ..Default::default() }, &mut m);
        drive(&MenuKeys { chars: vec!['0'], ..Default::default() }, &mut m);
        assert_eq!(m.text_value(0), "55055");
    }

    #[test]
    fn form_navigation_uses_tab_and_arrows_and_enter_submits() {
        let mut m = join_menu_model(None);
        drive(&MenuKeys { tab: true, ..Default::default() }, &mut m);
        assert_eq!(m.entries.cursor, 1);
        drive(&MenuKeys { down: true, ..Default::default() }, &mut m);
        assert_eq!(m.entries.cursor, 2);
        drive(&MenuKeys { up: true, ..Default::default() }, &mut m);
        assert_eq!(m.entries.cursor, 1);
        // Tab wraps like Down.
        m.entries.cursor = 3;
        drive(&MenuKeys { tab: true, ..Default::default() }, &mut m);
        assert_eq!(m.entries.cursor, 0);
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
        // Row indices follow the SETTINGS table order:
        // 0 fullscreen, 1 vsync, 2 cullfaces, 3 msaa, 4 max_fps,
        // 5 render_distance, 6 fov, 7 render_scale, 8 ui_scale.
        let mut s = Settings::default();
        apply_settings_cycle(&mut s, 0, Step::Next);
        assert!(s.fullscreen);
        apply_settings_cycle(&mut s, 3, Step::Prev);
        assert_eq!(s.msaa, 8, "msaa wraps 1 -> 8 going left");
        s.render_distance = 20;
        apply_settings_cycle(&mut s, 5, Step::Next);
        assert_eq!(s.render_distance, 3, "render distance wraps 20 -> 3");
        s.fov = 50.0;
        apply_settings_cycle(&mut s, 6, Step::Prev);
        assert_eq!(s.fov, 120.0, "fov wraps 50 -> 120");
        // The Back row cycles to nothing.
        let before = s.clone();
        apply_settings_cycle(&mut s, SETTINGS_ROW_BACK, Step::Next);
        assert_eq!(s, before);
    }
}
