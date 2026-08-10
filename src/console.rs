//! A minimal in-game console / chat line.
//!
//! Press `T` (or `/`, which pre-fills a slash) to open it, type a line, and press
//! Enter to submit; Esc closes it. Submitted lines are dispatched as commands by
//! [`command`](crate::command). The console keeps a small scrollback `log`, so it
//! doubles as a chat box — command output, chat, and system notices are the same
//! scrollback of [`Line`]s, each span tagged with a [`Role`] that selects its
//! colour when drawn.
//!
//! The editable line itself is a shared [`TextInput`], so cursor movement,
//! history recall, word/line deletion, and Tab-completion all come for free and
//! behave identically here and in any other text field.
use voxel_engine::{Color, Frame};

use crate::command::COMMAND_NAMES;
use crate::input::intent::EditKey;
use crate::ui::{common_prefix, Completion, Line, Ring, Role, TextInput};

/// Longest input line we accept.
const MAX_INPUT: usize = 128;
/// How many recent log lines to show on screen.
const LOG_LINES: usize = 6;

/// The console's state: whether it is capturing text, the editable input line,
/// and a bounded scrollback of past lines (chat, command echoes, and output).
pub struct Console {
    active: bool,
    input: TextInput,
    log: Ring<Line>,
}

impl Console {
    pub fn new() -> Self {
        Self {
            active: false,
            input: TextInput::new(MAX_INPUT).with_completer(complete_command),
            log: Ring::new(LOG_LINES * 4),
        }
    }

    pub fn is_open(&self) -> bool {
        self.active
    }

    pub fn open(&mut self, slash: bool) {
        self.active = true;
        self.input.clear();
        if slash {
            self.input.set("/");
        }
    }

    pub fn close(&mut self) {
        self.active = false;
        self.input.clear();
    }

    pub fn print(&mut self, line: impl Into<String>) {
        self.push(Line::of(Role::Dim, line));
    }

    pub fn echo(&mut self, line: impl Into<String>) {
        self.push(Line::of(Role::Accent, format!("> {}", line.into())));
    }

    pub fn push(&mut self, line: Line) {
        self.log.push(line);
    }

    /// Process input characters and an edit key. Returns submitted line (trimmed,
    /// non-empty), else `None`. Tab shows candidates; closing on Esc is the
    /// caller's job (it owns the event).
    pub fn handle_input(&mut self, chars: &[char], edit: Option<EditKey>) -> Option<String> {
        let submitted = self.input.handle(chars, edit);
        if let Some(candidates) = self.input.take_notice() {
            self.print(candidates.join("   "));
        }
        if let Some(line) = submitted {
            self.close();
            return Some(line);
        }
        None
    }

    pub fn draw(&self, f: &mut Frame, screen_w: i32, screen_h: i32) {
        let fs = 20;
        let line_h = fs + 4;
        let input_y = screen_h - line_h - 10;

        // Recent log lines stacked upward, just above the input line. Each line's
        // spans are drawn left-to-right; the monospace font makes `measure_text`
        // an exact advance, so span placement needs no layout pass.
        for (i, line) in self.log.iter_rev().take(LOG_LINES).enumerate() {
            let y = input_y - line_h * (i as i32 + 1) - 6;
            let mut x = 12;
            for span in line.spans() {
                shadowed(f, &span.text, x, y, fs, span.role.color());
                x += f.measure_text(&span.text, fs);
            }
        }

        if self.active {
            f.draw_rect(8, input_y - 4, screen_w - 16, line_h + 6, Color::new(0, 0, 0, 150));
            // Draw "> text" and a block caret sitting at the cursor column. We
            // measure the text left of the cursor to place it, so mid-line edits
            // show where typing will land.
            let prompt = "> ";
            let text = self.input.text();
            let full = format!("{prompt}{text}");
            shadowed(f, &full, 12, input_y, fs, Color::YELLOW);

            let left = &full[..prompt.len() + self.input.cursor()];
            let caret_x = 12 + f.measure_text(left, fs);
            f.draw_rect(caret_x, input_y, 2, fs, Color::YELLOW);
        }
    }
}

impl Default for Console {
    fn default() -> Self {
        Self::new()
    }
}

fn complete_command(input: &str) -> Completion {
    let body = input.strip_prefix('/').unwrap_or(input);
    if body.is_empty() || body.contains(char::is_whitespace) {
        return Completion::None;
    }
    let lead = if input.starts_with('/') { "/" } else { "" };
    let matches: Vec<&str> = COMMAND_NAMES
        .iter()
        .copied()
        .filter(|n| n.starts_with(body))
        .collect();
    match matches.as_slice() {
        [] => Completion::None,
        [only] => Completion::Full(format!("{lead}{only} ")),
        many => Completion::Ambiguous(
            format!("{lead}{}", common_prefix(many)),
            many.iter().map(|s| s.to_string()).collect(),
        ),
    }
}

pub use crate::ui::shadowed;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::Completion;

    #[test]
    fn completes_unique_command() {
        match complete_command("po") {
            Completion::Full(s) => assert_eq!(s, "pos "),
            _ => panic!("expected a unique completion"),
        }
    }

    #[test]
    fn preserves_leading_slash() {
        match complete_command("/po") {
            Completion::Full(s) => assert_eq!(s, "/pos "),
            _ => panic!("expected a unique completion"),
        }
    }

    #[test]
    fn empty_and_unknown_do_not_complete() {
        assert!(matches!(complete_command(""), Completion::None));
        assert!(matches!(complete_command("zzz"), Completion::None));
    }

    #[test]
    fn no_completion_after_a_space() {
        assert!(matches!(complete_command("tp 1"), Completion::None));
    }
}
