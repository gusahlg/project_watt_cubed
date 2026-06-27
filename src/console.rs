//! A minimal in-game console / chat line.
//!
//! Press `T` (or `/`, which pre-fills a slash) to open it, type a line, and press
//! Enter to submit; Esc closes it. Submitted lines are dispatched as commands by
//! [`command`](crate::command). The console keeps a small scrollback `log`, so it
//! doubles as the seed for a future chat box — swap the command dispatch for a
//! network send and the UI is already here.
use raylib::prelude::*;

/// Longest input line we accept.
const MAX_INPUT: usize = 128;
/// How many recent log lines to show on screen.
const LOG_LINES: usize = 6;

/// The console's state: whether it is capturing text, the current input line, and
/// a bounded scrollback of past lines (command echoes and their output).
pub struct Console {
    active: bool,
    input: String,
    log: Vec<String>,
}

impl Console {
    pub fn new() -> Self {
        Self {
            active: false,
            input: String::new(),
            log: Vec::new(),
        }
    }

    /// Whether the console is open and capturing keystrokes.
    pub fn is_open(&self) -> bool {
        self.active
    }

    /// Open the console. `slash` pre-fills a leading `/` (so pressing `/` starts a
    /// command without the user retyping it).
    pub fn open(&mut self, slash: bool) {
        self.active = true;
        self.input.clear();
        if slash {
            self.input.push('/');
        }
    }

    /// Close the console and discard the in-progress line.
    pub fn close(&mut self) {
        self.active = false;
        self.input.clear();
    }

    /// Append a line to the scrollback log, keeping it bounded.
    pub fn print(&mut self, line: impl Into<String>) {
        self.log.push(line.into());
        let cap = LOG_LINES * 4;
        if self.log.len() > cap {
            let excess = self.log.len() - cap;
            self.log.drain(0..excess);
        }
    }

    /// Process this frame's text input. Returns the submitted line when the user
    /// presses Enter (trimmed and non-empty), otherwise `None`. Esc closes the
    /// console.
    pub fn handle_input(&mut self, rl: &mut RaylibHandle) -> Option<String> {
        // `get_char_pressed` already accounts for keyboard layout and shift state.
        while let Some(c) = rl.get_char_pressed() {
            if self.input.len() < MAX_INPUT && !c.is_control() {
                self.input.push(c);
            }
        }

        if rl.is_key_pressed(KeyboardKey::KEY_BACKSPACE) {
            self.input.pop();
        }
        if rl.is_key_pressed(KeyboardKey::KEY_ESCAPE) {
            self.close();
            return None;
        }
        if rl.is_key_pressed(KeyboardKey::KEY_ENTER) {
            let line = std::mem::take(&mut self.input).trim().to_string();
            self.close();
            if !line.is_empty() {
                return Some(line);
            }
        }
        None
    }

    /// Draw the scrollback log (always, when non-empty) and, while open, the input
    /// line. Kept at the bottom of the screen, chat-style.
    pub fn draw<D: RaylibDraw>(&self, d: &mut D, screen_w: i32, screen_h: i32) {
        let fs = 20;
        let line_h = fs + 4;
        let input_y = screen_h - line_h - 10;

        // Recent log lines stacked upward, just above the input line.
        for (i, line) in self.log.iter().rev().take(LOG_LINES).enumerate() {
            let y = input_y - line_h * (i as i32 + 1) - 6;
            shadowed(d, line, 12, y, fs, Color::RAYWHITE);
        }

        if self.active {
            d.draw_rectangle(8, input_y - 4, screen_w - 16, line_h + 6, Color::new(0, 0, 0, 150));
            // A trailing underscore stands in for a text cursor.
            let text = format!("> {}_", self.input);
            shadowed(d, &text, 12, input_y, fs, Color::YELLOW);
        }
    }
}

impl Default for Console {
    fn default() -> Self {
        Self::new()
    }
}

/// Draw text with a 1px dark drop shadow so it stays readable over bright terrain.
pub fn shadowed<D: RaylibDraw>(d: &mut D, text: &str, x: i32, y: i32, fs: i32, color: Color) {
    d.draw_text(text, x + 1, y + 1, fs, Color::new(0, 0, 0, 180));
    d.draw_text(text, x, y, fs, color);
}
