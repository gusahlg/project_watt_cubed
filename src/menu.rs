//! The out-of-game screens: the start menu (new / load / host / join / mods / quit),
//! the mod menu (toggle installed mods), and the host/join forms. All are simple
//! keyboard-driven — Up/Down to move, Enter to choose — kept deliberately plain so
//! the menus are easy to restyle or replace (a menu is exactly the kind of thing a
//! mod might take over).
use raylib::prelude::*;

use crate::console::shadowed;
use crate::mods::Mods;
use crate::net::{DEFAULT_PORT, MAX_NAME};
use crate::save;

/// What the player picked on the start menu.
pub enum MainChoice {
    NewWorld,
    Load(String),
    Host,
    Join,
    Mods,
    Quit,
}

/// The start menu. Owns its list of existing saves and the current selection.
pub struct MainMenu {
    selected: usize,
    saves: Vec<String>,
}

impl MainMenu {
    pub fn new() -> Self {
        Self {
            selected: 0,
            saves: save::list_saves(),
        }
    }

    /// Re-read the saves on disk (call when returning to the menu).
    pub fn refresh(&mut self) {
        self.saves = save::list_saves();
        let max = self.item_count().saturating_sub(1);
        self.selected = self.selected.min(max);
    }

    /// Total selectable rows: New World, one per save, Host, Join, Mods, Quit.
    fn item_count(&self) -> usize {
        self.saves.len() + 5
    }

    /// Resolve the current selection index into a concrete choice.
    fn choice_at(&self, index: usize) -> MainChoice {
        let saves = self.saves.len();
        if index == 0 {
            MainChoice::NewWorld
        } else if index <= saves {
            MainChoice::Load(self.saves[index - 1].clone())
        } else if index == saves + 1 {
            MainChoice::Host
        } else if index == saves + 2 {
            MainChoice::Join
        } else if index == saves + 3 {
            MainChoice::Mods
        } else {
            MainChoice::Quit
        }
    }

    /// Handle a frame of input, returning a choice when the player presses Enter.
    pub fn update(&mut self, rl: &RaylibHandle) -> Option<MainChoice> {
        let count = self.item_count();
        if rl.is_key_pressed(KeyboardKey::KEY_DOWN) {
            self.selected = (self.selected + 1) % count;
        }
        if rl.is_key_pressed(KeyboardKey::KEY_UP) {
            self.selected = (self.selected + count - 1) % count;
        }
        if rl.is_key_pressed(KeyboardKey::KEY_ENTER) {
            return Some(self.choice_at(self.selected));
        }
        None
    }

    /// Draw the title and menu list.
    pub fn draw(&self, d: &mut RaylibDrawHandle, screen_w: i32, screen_h: i32) {
        d.clear_background(Color::new(18, 20, 28, 255));

        let title = "PROJECT WATT CUBED";
        let title_fs = 48;
        let tx = (screen_w - d.measure_text(title, title_fs)) / 2;
        shadowed(d, title, tx, screen_h / 6, title_fs, Color::GOLD);

        let subtitle = "an infinite voxel world of elements";
        let sub_fs = 20;
        let sx = (screen_w - d.measure_text(subtitle, sub_fs)) / 2;
        shadowed(d, subtitle, sx, screen_h / 6 + title_fs + 8, sub_fs, Color::GRAY);

        // Build the labels in the same order as `choice_at`.
        let mut labels = vec!["New World".to_string()];
        for name in &self.saves {
            labels.push(format!("Load: {name}"));
        }
        labels.push("Host Server".to_string());
        labels.push("Join Server".to_string());
        labels.push("Mods".to_string());
        labels.push("Quit".to_string());

        let fs = 28;
        let line_h = fs + 14;
        let start_y = screen_h / 2 - line_h;
        for (i, label) in labels.iter().enumerate() {
            let selected = i == self.selected;
            let text = if selected {
                format!("> {label}")
            } else {
                format!("  {label}")
            };
            let color = if selected { Color::RAYWHITE } else { Color::GRAY };
            let x = (screen_w - d.measure_text(&text, fs)) / 2;
            shadowed(d, &text, x, start_y + line_h * i as i32, fs, color);
        }

        let hint = "Up/Down select   Enter choose";
        let hint_fs = 18;
        let hx = (screen_w - d.measure_text(hint, hint_fs)) / 2;
        shadowed(d, hint, hx, screen_h - 40, hint_fs, Color::DARKGRAY);
    }
}

impl Default for MainMenu {
    fn default() -> Self {
        Self::new()
    }
}

/// The mod menu: toggle installed mods on and off.
pub struct ModMenu {
    selected: usize,
}

impl ModMenu {
    pub fn new() -> Self {
        Self { selected: 0 }
    }

    /// Handle input; returns `true` when the player wants to go back.
    pub fn update(&mut self, rl: &RaylibHandle, mods: &mut Mods) -> bool {
        let count = mods.len().max(1);
        if rl.is_key_pressed(KeyboardKey::KEY_DOWN) {
            self.selected = (self.selected + 1) % count;
        }
        if rl.is_key_pressed(KeyboardKey::KEY_UP) {
            self.selected = (self.selected + count - 1) % count;
        }
        if (rl.is_key_pressed(KeyboardKey::KEY_ENTER) || rl.is_key_pressed(KeyboardKey::KEY_SPACE))
            && self.selected < mods.len()
        {
            mods.toggle(self.selected);
        }
        rl.is_key_pressed(KeyboardKey::KEY_ESCAPE)
            || rl.is_key_pressed(KeyboardKey::KEY_BACKSPACE)
    }

    /// Draw the list of mods with their on/off state and descriptions.
    pub fn draw(&self, d: &mut RaylibDrawHandle, mods: &Mods, screen_w: i32, screen_h: i32) {
        d.clear_background(Color::new(18, 20, 28, 255));

        let title = "MODS";
        let title_fs = 40;
        let tx = (screen_w - d.measure_text(title, title_fs)) / 2;
        shadowed(d, title, tx, screen_h / 8, title_fs, Color::GOLD);

        let fs = 26;
        let line_h = fs + 20;
        let start_y = screen_h / 4 + 20;
        let x = screen_w / 2 - 260;

        if mods.is_empty() {
            shadowed(d, "  (no mods installed)", x, start_y, fs, Color::GRAY);
        }

        for i in 0..mods.len() {
            let selected = i == self.selected;
            let mark = if mods.is_enabled(i) { "[x]" } else { "[ ]" };
            let row = format!("{} {} {}", if selected { ">" } else { " " }, mark, mods.name(i));
            let color = if selected { Color::RAYWHITE } else { Color::GRAY };
            shadowed(d, &row, x, start_y + line_h * i as i32, fs, color);
            // Description under each row, dimmer.
            shadowed(
                d,
                mods.description(i),
                x + 40,
                start_y + line_h * i as i32 + fs + 2,
                16,
                Color::DARKGRAY,
            );
        }

        let hint = "Up/Down select   Enter toggle   Esc back";
        let hint_fs = 18;
        let hx = (screen_w - d.measure_text(hint, hint_fs)) / 2;
        shadowed(d, hint, hx, screen_h - 40, hint_fs, Color::DARKGRAY);
    }
}

impl Default for ModMenu {
    fn default() -> Self {
        Self::new()
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

/// One editable text field in a [`Form`].
struct Field {
    label: &'static str,
    value: String,
    /// Rendered as dots, for the password.
    masked: bool,
    /// Largest number of characters accepted.
    max: usize,
}

/// A tiny keyboard-driven form: Up/Down (or Tab) to pick a field, type to edit,
/// Enter to submit, Esc to cancel. Shared by the host and join screens so the two
/// stay identical to use.
struct Form {
    fields: Vec<Field>,
    selected: usize,
}

impl Form {
    fn new(fields: Vec<Field>) -> Self {
        Self { fields, selected: 0 }
    }

    /// Process a frame. Returns `Some(true)` on submit, `Some(false)` on cancel.
    fn update(&mut self, rl: &mut RaylibHandle) -> Option<bool> {
        let n = self.fields.len();
        if rl.is_key_pressed(KeyboardKey::KEY_DOWN) || rl.is_key_pressed(KeyboardKey::KEY_TAB) {
            self.selected = (self.selected + 1) % n;
        }
        if rl.is_key_pressed(KeyboardKey::KEY_UP) {
            self.selected = (self.selected + n - 1) % n;
        }
        if rl.is_key_pressed(KeyboardKey::KEY_ENTER) {
            return Some(true);
        }
        if rl.is_key_pressed(KeyboardKey::KEY_ESCAPE) {
            return Some(false);
        }
        if rl.is_key_pressed(KeyboardKey::KEY_BACKSPACE) {
            self.fields[self.selected].value.pop();
        }
        while let Some(c) = rl.get_char_pressed() {
            let field = &mut self.fields[self.selected];
            if !c.is_control() && field.value.len() < field.max {
                field.value.push(c);
            }
        }
        None
    }

    fn value(&self, index: usize) -> &str {
        &self.fields[index].value
    }

    /// Draw the form's title, its fields (the selected one highlighted), and a hint.
    fn draw(&self, d: &mut RaylibDrawHandle, title: &str, hint: &str, screen_w: i32, screen_h: i32) {
        d.clear_background(Color::new(18, 20, 28, 255));

        let title_fs = 40;
        let tx = (screen_w - d.measure_text(title, title_fs)) / 2;
        shadowed(d, title, tx, screen_h / 6, title_fs, Color::GOLD);

        let fs = 26;
        let line_h = fs + 22;
        let start_y = screen_h / 2 - line_h;
        let x = screen_w / 2 - 240;
        for (i, field) in self.fields.iter().enumerate() {
            let selected = i == self.selected;
            let shown = if field.masked {
                "*".repeat(field.value.chars().count())
            } else {
                field.value.clone()
            };
            let caret = if selected { "_" } else { "" };
            let row = format!("{} {}: {}{}", if selected { ">" } else { " " }, field.label, shown, caret);
            let color = if selected { Color::RAYWHITE } else { Color::GRAY };
            shadowed(d, &row, x, start_y + line_h * i as i32, fs, color);
        }

        let hint_fs = 18;
        let hx = (screen_w - d.measure_text(hint, hint_fs)) / 2;
        shadowed(d, hint, hx, screen_h - 40, hint_fs, Color::DARKGRAY);
    }
}

/// Parse a port field, falling back to the default if it's blank or malformed.
fn parse_port(text: &str) -> u16 {
    text.trim().parse().unwrap_or(DEFAULT_PORT)
}

/// The host screen: choose a port, an optional password, and your name.
pub struct HostMenu {
    form: Form,
}

impl HostMenu {
    pub fn new() -> Self {
        Self {
            form: Form::new(vec![
                Field { label: "Port", value: DEFAULT_PORT.to_string(), masked: false, max: 5 },
                Field { label: "Password (optional)", value: String::new(), masked: true, max: 64 },
                Field { label: "Your name", value: "player".to_string(), masked: false, max: MAX_NAME },
            ]),
        }
    }

    /// Returns `Some(info)` to start hosting, `None` while editing. Cancelling (Esc)
    /// is reported through the returned [`Option`] being `None` with `cancelled`.
    pub fn update(&mut self, rl: &mut RaylibHandle) -> FormResult<HostInfo> {
        match self.form.update(rl) {
            Some(true) => FormResult::Submit(HostInfo {
                port: parse_port(self.form.value(0)),
                password: self.form.value(1).to_string(),
                name: self.form.value(2).to_string(),
            }),
            Some(false) => FormResult::Cancel,
            None => FormResult::Editing,
        }
    }

    pub fn draw(&self, d: &mut RaylibDrawHandle, screen_w: i32, screen_h: i32) {
        self.form.draw(
            d,
            "HOST SERVER",
            "Up/Down field   type to edit   Enter start   Esc back",
            screen_w,
            screen_h,
        );
    }
}

impl Default for HostMenu {
    fn default() -> Self {
        Self::new()
    }
}

/// The join screen: enter a server address, port, password, and your name.
pub struct JoinMenu {
    form: Form,
}

impl JoinMenu {
    pub fn new() -> Self {
        Self {
            form: Form::new(vec![
                Field { label: "Address", value: "127.0.0.1".to_string(), masked: false, max: 64 },
                Field { label: "Port", value: DEFAULT_PORT.to_string(), masked: false, max: 5 },
                Field { label: "Password", value: String::new(), masked: true, max: 64 },
                Field { label: "Your name", value: "player".to_string(), masked: false, max: MAX_NAME },
            ]),
        }
    }

    pub fn update(&mut self, rl: &mut RaylibHandle) -> FormResult<JoinInfo> {
        match self.form.update(rl) {
            Some(true) => FormResult::Submit(JoinInfo {
                host: self.form.value(0).trim().to_string(),
                port: parse_port(self.form.value(1)),
                password: self.form.value(2).to_string(),
                name: self.form.value(3).to_string(),
            }),
            Some(false) => FormResult::Cancel,
            None => FormResult::Editing,
        }
    }

    pub fn draw(&self, d: &mut RaylibDrawHandle, screen_w: i32, screen_h: i32) {
        self.form.draw(
            d,
            "JOIN SERVER",
            "Up/Down field   type to edit   Enter connect   Esc back",
            screen_w,
            screen_h,
        );
    }
}

impl Default for JoinMenu {
    fn default() -> Self {
        Self::new()
    }
}

/// The outcome of a form frame: still editing, submitted with a value, or cancelled.
pub enum FormResult<T> {
    Editing,
    Submit(T),
    Cancel,
}
