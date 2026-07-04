//! The out-of-game screens: the start menu (new / load / host / join / mods /
//! settings / quit), the mod menu (toggle installed mods), the settings menu
//! (graphics options), and the host/join forms. All are simple keyboard-driven —
//! Up/Down to move, Enter to choose — kept deliberately plain so the menus are
//! easy to restyle or replace (a menu is exactly the kind of thing a mod might
//! take over).
use voxel_engine::{Color, Engine, Frame, Key};

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
    Settings,
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

    /// Total selectable rows: New World, one per save, Host, Join, Mods, Settings, Quit.
    fn item_count(&self) -> usize {
        self.saves.len() + 6
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
        } else if index == saves + 4 {
            MainChoice::Settings
        } else {
            MainChoice::Quit
        }
    }

    /// Handle a frame of input, returning a choice when the player presses Enter.
    /// Also accepts vim-style j/k/l for down/up/select.
    pub fn update(&mut self, eng: &Engine) -> Option<MainChoice> {
        let count = self.item_count();
        if eng.is_key_pressed(Key::Down) || eng.is_key_pressed(Key::J) {
            self.selected = (self.selected + 1) % count;
        }
        if eng.is_key_pressed(Key::Up) || eng.is_key_pressed(Key::K) {
            self.selected = (self.selected + count - 1) % count;
        }
        if eng.is_key_pressed(Key::Enter) || eng.is_key_pressed(Key::L) {
            return Some(self.choice_at(self.selected));
        }
        None
    }

    /// Draw the title and menu list.
    pub fn draw(&self, f: &mut Frame, screen_w: i32, screen_h: i32) {
        f.draw_rect(0, 0, screen_w, screen_h, Color::new(18, 20, 28, 255));

        let title = "PROJECT WATT CUBED";
        let title_fs = 48;
        let tx = (screen_w - f.measure_text(title, title_fs)) / 2;
        shadowed(f, title, tx, screen_h / 6, title_fs, Color::GOLD);

        let subtitle = "an infinite voxel world of elements";
        let sub_fs = 20;
        let sx = (screen_w - f.measure_text(subtitle, sub_fs)) / 2;
        shadowed(f, subtitle, sx, screen_h / 6 + title_fs + 8, sub_fs, Color::GRAY);

        // Build the labels in the same order as `choice_at`.
        let mut labels = vec!["New World".to_string()];
        for name in &self.saves {
            labels.push(format!("Load: {name}"));
        }
        labels.push("Host Server".to_string());
        labels.push("Join Server".to_string());
        labels.push("Mods".to_string());
        labels.push("Settings".to_string());
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
            let x = (screen_w - f.measure_text(&text, fs)) / 2;
            shadowed(f, &text, x, start_y + line_h * i as i32, fs, color);
        }

        let hint = "Up/Down or j/k select   Enter or l choose";
        let hint_fs = 18;
        let hx = (screen_w - f.measure_text(hint, hint_fs)) / 2;
        shadowed(f, hint, hx, screen_h - 40, hint_fs, Color::DARKGRAY);
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
    /// Also accepts vim-style j/k/l/h for down/up/toggle/back.
    pub fn update(&mut self, eng: &Engine, mods: &mut Mods) -> bool {
        let count = mods.len().max(1);
        if eng.is_key_pressed(Key::Down) || eng.is_key_pressed(Key::J) {
            self.selected = (self.selected + 1) % count;
        }
        if eng.is_key_pressed(Key::Up) || eng.is_key_pressed(Key::K) {
            self.selected = (self.selected + count - 1) % count;
        }
        if (eng.is_key_pressed(Key::Enter)
            || eng.is_key_pressed(Key::Space)
            || eng.is_key_pressed(Key::L))
            && self.selected < mods.len()
        {
            mods.toggle(self.selected);
        }
        eng.is_key_pressed(Key::Escape)
            || eng.is_key_pressed(Key::Backspace)
            || eng.is_key_pressed(Key::H)
    }

    /// Draw the list of mods with their on/off state and descriptions.
    pub fn draw(&self, f: &mut Frame, mods: &Mods, screen_w: i32, screen_h: i32) {
        f.draw_rect(0, 0, screen_w, screen_h, Color::new(18, 20, 28, 255));

        let title = "MODS";
        let title_fs = 40;
        let tx = (screen_w - f.measure_text(title, title_fs)) / 2;
        shadowed(f, title, tx, screen_h / 8, title_fs, Color::GOLD);

        let fs = 26;
        let line_h = fs + 20;
        let start_y = screen_h / 4 + 20;
        let x = screen_w / 2 - 260;

        if mods.is_empty() {
            shadowed(f, "  (no mods installed)", x, start_y, fs, Color::GRAY);
        }

        for i in 0..mods.len() {
            let selected = i == self.selected;
            let mark = if mods.is_enabled(i) { "[x]" } else { "[ ]" };
            let row = format!("{} {} {}", if selected { ">" } else { " " }, mark, mods.name(i));
            let color = if selected { Color::RAYWHITE } else { Color::GRAY };
            shadowed(f, &row, x, start_y + line_h * i as i32, fs, color);
            // Description under each row, dimmer.
            shadowed(
                f,
                mods.description(i),
                x + 40,
                start_y + line_h * i as i32 + fs + 2,
                16,
                Color::DARKGRAY,
            );
        }

        let hint = "Up/Down or j/k select   Enter/l toggle   Esc/h back";
        let hint_fs = 18;
        let hx = (screen_w - f.measure_text(hint, hint_fs)) / 2;
        shadowed(f, hint, hx, screen_h - 40, hint_fs, Color::DARKGRAY);
    }
}

impl Default for ModMenu {
    fn default() -> Self {
        Self::new()
    }
}

/// Rows in the settings menu, top to bottom: Fullscreen, VSync, MSAA, Max FPS,
/// Render Distance, FOV, Render Scale, Back.
const SETTINGS_ROWS: usize = 8;
/// Index of the Back row.
const SETTINGS_ROW_BACK: usize = SETTINGS_ROWS - 1;

/// Step to the adjacent entry in `values`, wrapping at both ends. A current value
/// not in the list (e.g. a hand-edited config) snaps to the first entry first.
fn cycle_list(values: &[u32], current: u32, dir: i32) -> u32 {
    match values.iter().position(|&v| v == current) {
        Some(i) => values[(i as i32 + dir).rem_euclid(values.len() as i32) as usize],
        // Off-list (e.g. a /gfx or hand-edited value): snap to the first
        // entry without stepping, so Left can never jump 25% -> 200%.
        None => values[0],
    }
}

/// The settings menu: graphics options cycled in place. Mutates the passed
/// [`Settings`](crate::settings::Settings) directly; the caller applies and
/// persists them.
pub struct SettingsMenu {
    selected: usize,
}

impl SettingsMenu {
    pub fn new() -> Self {
        Self { selected: 0 }
    }

    /// Handle input; returns `true` when the player wants to go back (Esc
    /// anywhere, or Enter — or l — on the Back row). Left/Right cycle the
    /// selected value down/up; Enter also cycles up. Also accepts vim-style
    /// j/k for down/up and h/l as aliases of Left/Right.
    pub fn update(&mut self, eng: &Engine, s: &mut crate::settings::Settings) -> bool {
        if eng.is_key_pressed(Key::Escape) {
            return true;
        }
        if eng.is_key_pressed(Key::Down) || eng.is_key_pressed(Key::J) {
            self.selected = (self.selected + 1) % SETTINGS_ROWS;
        }
        if eng.is_key_pressed(Key::Up) || eng.is_key_pressed(Key::K) {
            self.selected = (self.selected + SETTINGS_ROWS - 1) % SETTINGS_ROWS;
        }
        let enter = eng.is_key_pressed(Key::Enter);
        let l = eng.is_key_pressed(Key::L);
        if (enter || l) && self.selected == SETTINGS_ROW_BACK {
            return true;
        }
        if eng.is_key_pressed(Key::Left) || eng.is_key_pressed(Key::H) {
            self.cycle(s, -1);
        }
        if eng.is_key_pressed(Key::Right) || l || enter {
            self.cycle(s, 1);
        }
        false
    }

    /// Apply one Left/Right (or Enter) step to the selected row's value, wrapping.
    fn cycle(&self, s: &mut crate::settings::Settings, dir: i32) {
        match self.selected {
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

    /// Draw the settings rows with their current values.
    pub fn draw(&self, f: &mut Frame, s: &crate::settings::Settings, screen_w: i32, screen_h: i32) {
        f.draw_rect(0, 0, screen_w, screen_h, Color::new(18, 20, 28, 255));

        let title = "SETTINGS";
        let title_fs = 40;
        let tx = (screen_w - f.measure_text(title, title_fs)) / 2;
        shadowed(f, title, tx, screen_h / 8, title_fs, Color::GOLD);

        let on_off = |on: bool| if on { "On" } else { "Off" };
        let max_fps = if s.max_fps == 0 {
            "Uncapped".to_string()
        } else {
            s.max_fps.to_string()
        };
        let labels = [
            format!("Fullscreen: {}", on_off(s.fullscreen)),
            format!("VSync: {}", on_off(s.vsync)),
            format!("MSAA: {}x", s.msaa),
            format!("Max FPS: {max_fps}"),
            format!("Render Distance: {}", s.render_distance),
            format!("FOV: {}", s.fov),
            format!("Render Scale: {:.0}%", s.render_scale * 100.0),
            "Back".to_string(),
        ];

        let fs = 26;
        let line_h = fs + 20;
        let start_y = screen_h / 4 + 20;
        let x = screen_w / 2 - 260;
        for (i, label) in labels.iter().enumerate() {
            let selected = i == self.selected;
            let row = format!("{} {label}", if selected { ">" } else { " " });
            let color = if selected { Color::RAYWHITE } else { Color::GRAY };
            shadowed(f, &row, x, start_y + line_h * i as i32, fs, color);
        }

        let hint = "Up/Down or j/k select | Left/Right or h/l change | Esc back";
        let hint_fs = 18;
        let hx = (screen_w - f.measure_text(hint, hint_fs)) / 2;
        shadowed(f, hint, hx, screen_h - 40, hint_fs, Color::DARKGRAY);
    }
}

impl Default for SettingsMenu {
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
    fn update(&mut self, eng: &Engine) -> Option<bool> {
        let n = self.fields.len();
        if eng.is_key_pressed(Key::Down) || eng.is_key_pressed(Key::Tab) {
            self.selected = (self.selected + 1) % n;
        }
        if eng.is_key_pressed(Key::Up) {
            self.selected = (self.selected + n - 1) % n;
        }
        if eng.is_key_pressed(Key::Enter) {
            return Some(true);
        }
        if eng.is_key_pressed(Key::Escape) {
            return Some(false);
        }
        if eng.is_key_pressed(Key::Backspace) {
            self.fields[self.selected].value.pop();
        }
        while let Some(c) = eng.get_char_pressed() {
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
    fn draw(&self, f: &mut Frame, title: &str, hint: &str, screen_w: i32, screen_h: i32) {
        f.draw_rect(0, 0, screen_w, screen_h, Color::new(18, 20, 28, 255));

        let title_fs = 40;
        let tx = (screen_w - f.measure_text(title, title_fs)) / 2;
        shadowed(f, title, tx, screen_h / 6, title_fs, Color::GOLD);

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
            shadowed(f, &row, x, start_y + line_h * i as i32, fs, color);
        }

        let hint_fs = 18;
        let hx = (screen_w - f.measure_text(hint, hint_fs)) / 2;
        shadowed(f, hint, hx, screen_h - 40, hint_fs, Color::DARKGRAY);
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
    pub fn update(&mut self, eng: &Engine) -> FormResult<HostInfo> {
        match self.form.update(eng) {
            Some(true) => FormResult::Submit(HostInfo {
                port: parse_port(self.form.value(0)),
                password: self.form.value(1).to_string(),
                name: self.form.value(2).to_string(),
            }),
            Some(false) => FormResult::Cancel,
            None => FormResult::Editing,
        }
    }

    pub fn draw(&self, f: &mut Frame, screen_w: i32, screen_h: i32) {
        self.form.draw(
            f,
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

    pub fn update(&mut self, eng: &Engine) -> FormResult<JoinInfo> {
        match self.form.update(eng) {
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

    pub fn draw(&self, f: &mut Frame, screen_w: i32, screen_h: i32) {
        self.form.draw(
            f,
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
