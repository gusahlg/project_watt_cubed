//! The out-of-game screens: the start menu (new / load / mods / quit) and the mod
//! menu (toggle installed mods). Both are simple keyboard-driven vertical lists —
//! Up/Down to move, Enter to choose — kept deliberately plain so the menus are easy
//! to restyle or replace (a menu is exactly the kind of thing a mod might take over).
use raylib::prelude::*;

use crate::console::shadowed;
use crate::mods::Mods;
use crate::save;

/// What the player picked on the start menu.
pub enum MainChoice {
    NewWorld,
    Load(String),
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

    /// Total selectable rows: New World, one per save, Mods, Quit.
    fn item_count(&self) -> usize {
        self.saves.len() + 3
    }

    /// Resolve the current selection index into a concrete choice.
    fn choice_at(&self, index: usize) -> MainChoice {
        if index == 0 {
            MainChoice::NewWorld
        } else if index <= self.saves.len() {
            MainChoice::Load(self.saves[index - 1].clone())
        } else if index == self.saves.len() + 1 {
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
