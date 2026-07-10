//! Small reusable UI spine types.
//!
//! The design bet is "more types = less code": a couple of invariant-carrying
//! types here let the console, chat, and (later) menu forms share one editor
//! and one bounded-scrollback container instead of each re-rolling their own.
//!
//! - [`Ring<T>`] — a bounded FIFO. It is the chat/command scrollback, the input
//!   history, and (later) the recent-servers and toast queues.
//! - [`TextInput`] — an editable line with a boundary-safe cursor, history
//!   recall, word/line deletion, and optional Tab-completion. One `handle` per
//!   frame drives every text field in the game the same way.
use std::collections::VecDeque;

use voxel_engine::{Color, Engine, Frame, Key};

/// A screen-space size or offset in pixels, `(x, y)`. Kept as a plain tuple so
/// this module needs no vector-math dependency of its own.
pub type Px = (i32, i32);

/// The nine placement points of a rectangle. A total enum — there is no
/// "some magic offset" corner, so every HUD element names where it lives and the
/// right/bottom/centre arithmetic exists in exactly one place ([`Anchor::origin`]).
#[derive(Clone, Copy)]
pub enum Anchor {
    TopLeft,
    Top,
    TopRight,
    Left,
    Center,
    Right,
    BottomLeft,
    Bottom,
    BottomRight,
}

impl Anchor {
    /// Per-axis alignment factor: 0 = start edge, 1 = centre, 2 = end edge.
    fn factors(self) -> (i32, i32) {
        use Anchor::*;
        match self {
            TopLeft => (0, 0),
            Top => (1, 0),
            TopRight => (2, 0),
            Left => (0, 1),
            Center => (1, 1),
            Right => (2, 1),
            BottomLeft => (0, 2),
            Bottom => (1, 2),
            BottomRight => (2, 2),
        }
    }

    /// Top-left origin at which to draw an item of `size` inside `screen`, plus a
    /// margin `off`. For an end-edge anchor (`Right`/`Bottom`) `off` is naturally
    /// negative — an inset from that edge. This requires the item's measured size,
    /// so an edge-anchored element cannot be misplaced by forgetting to subtract
    /// its width.
    pub fn origin(self, screen: Px, size: Px, off: Px) -> Px {
        let (fx, fy) = self.factors();
        (
            (screen.0 - size.0) * fx / 2 + off.0,
            (screen.1 - size.1) * fy / 2 + off.1,
        )
    }
}

/// Semantic UI colours: call sites ask for a *role*, not a raw RGB, so a palette
/// swap can't miss a site and there are no scattered `Color::` literals.
#[derive(Clone, Copy)]
pub struct Palette {
    pub text: Color,
    pub muted: Color,
    pub accent: Color,
    pub good: Color,
    pub warn: Color,
    pub bad: Color,
}

impl Palette {
    pub const DEFAULT: Self = Self {
        text: Color::WHITE,
        muted: Color::RAYWHITE,
        accent: Color::SKYBLUE,
        good: Color::LIME,
        warn: Color::GOLD,
        bad: Color::SALMON,
    };
}

/// The aiming reticle, as data: swap the value to restyle it.
#[derive(Clone, Copy)]
pub struct Crosshair {
    /// Length of each arm in pixels.
    pub arm: i32,
    /// Gap between the centre and the start of each arm.
    pub gap: i32,
    pub color: Color,
}

impl Crosshair {
    pub const DEFAULT: Self = Self {
        arm: 8,
        gap: 0,
        color: Color::new(255, 255, 255, 180),
    };

    /// Draw the reticle centred on the screen.
    pub fn draw(&self, f: &mut Frame, screen: Px) {
        let (cx, cy) = (screen.0 / 2, screen.1 / 2);
        let (a, g) = (self.arm, self.gap);
        f.draw_line(cx - a - g, cy, cx - g, cy, self.color);
        f.draw_line(cx + g, cy, cx + a + g, cy, self.color);
        f.draw_line(cx, cy - a - g, cx, cy - g, self.color);
        f.draw_line(cx, cy + g, cx, cy + a + g, self.color);
    }
}

/// How much of the HUD is shown. A three-state cycle rather than a bool: `Minimal`
/// keeps the reticle (and world-space name tags) but hides the informational text.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HudMode {
    Full,
    Minimal,
    Off,
}

impl HudMode {
    /// Advance to the next mode, wrapping (drives the F1 toggle).
    pub fn next(self) -> Self {
        match self {
            HudMode::Full => HudMode::Minimal,
            HudMode::Minimal => HudMode::Off,
            HudMode::Off => HudMode::Full,
        }
    }

    /// Whether informational text (coords, help, counts) is shown.
    pub fn shows_info(self) -> bool {
        matches!(self, HudMode::Full)
    }

    /// Whether the reticle and world-space name tags are shown (all but `Off`).
    pub fn shows_world_ui(self) -> bool {
        !matches!(self, HudMode::Off)
    }
}

/// The whole in-world UI look, threaded through drawing. `scale` routes every font
/// size; `hud` is the master visibility cycle.
pub struct Theme {
    pub palette: Palette,
    pub scale: f32,
    pub crosshair: Crosshair,
    pub hud: HudMode,
}

impl Theme {
    pub fn new() -> Self {
        Self {
            palette: Palette::DEFAULT,
            scale: 1.0,
            crosshair: Crosshair::DEFAULT,
            hud: HudMode::Full,
        }
    }

    /// The single sizing path: a base point size scaled by the UI scale.
    pub fn fs(&self, base: i32) -> i32 {
        (base as f32 * self.scale).round() as i32
    }

    /// Cycle the HUD visibility (Full → Minimal → Off → …).
    pub fn cycle_hud(&mut self) {
        self.hud = self.hud.next();
    }
}

impl Default for Theme {
    fn default() -> Self {
        Self::new()
    }
}

/// Measure → resolve → shadowed-draw a line of HUD text in one call. Every HUD
/// text site collapses to this: the alignment math and the drop shadow are hidden.
pub fn label(
    f: &mut Frame,
    theme: &Theme,
    screen: Px,
    at: Anchor,
    off: Px,
    base_fs: i32,
    color: Color,
    text: &str,
) {
    let fs = theme.fs(base_fs);
    let w = f.measure_text(text, fs);
    let (x, y) = at.origin(screen, (w, fs), off);
    crate::console::shadowed(f, text, x, y, fs, color);
}

/// The semantic role of a run of console text — what it *means*, not what colour
/// it is. Rendering resolves a role to a colour through [`Role::color`], so a line
/// can only ever be an on-palette colour and the command layer speaks meaning
/// (`Error` vs `System`) instead of pixels.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    /// A chat message body.
    Chat,
    /// The echo of a command the user submitted.
    Command,
    /// Neutral system/status text or normal command output.
    System,
    /// A rejected input or error.
    Error,
    /// A player name.
    Name,
    /// The `[global]` chat-scope tag.
    Global,
    /// A highlighted value inside otherwise-neutral text.
    Value,
}

impl Role {
    pub fn color(self) -> Color {
        match self {
            Role::Chat => Color::RAYWHITE,
            Role::Command => Color::SKYBLUE,
            Role::System => Color::LIGHTGRAY,
            Role::Error => Color::SALMON,
            Role::Name => Color::SKYBLUE,
            Role::Global => Color::GOLD,
            Role::Value => Color::LIME,
        }
    }
}

/// A run of text in one role. The atom of a [`Line`].
#[derive(Clone)]
pub struct Span {
    pub text: String,
    pub role: Role,
}

/// One scrollback line: a guaranteed first span plus any continuation spans.
/// Non-empty by construction, so it always renders something and drawing needs
/// no empty-line guard. Build with [`Line::of`] and chain [`Line::then`].
#[derive(Clone)]
pub struct Line {
    head: Span,
    tail: Vec<Span>,
}

impl Line {
    /// A whole-line message in one role — the common case.
    pub fn of(role: Role, text: impl Into<String>) -> Self {
        Self {
            head: Span { text: text.into(), role },
            tail: Vec::new(),
        }
    }

    /// Append a run in another role, for multi-colour lines (e.g. a coloured
    /// player name before a white chat body).
    pub fn then(mut self, role: Role, text: impl Into<String>) -> Self {
        self.tail.push(Span { text: text.into(), role });
        self
    }

    /// The spans left-to-right, head first.
    pub fn spans(&self) -> impl Iterator<Item = &Span> {
        std::iter::once(&self.head).chain(&self.tail)
    }

    /// The first span's text — the whole line for single-span lines. Handy for
    /// tests and log inspection.
    pub fn text(&self) -> &str {
        &self.head.text
    }
}

/// A bounded FIFO: pushing past `cap` drops the oldest element. Indexing runs
/// oldest (`0`) to newest (`len-1`).
pub struct Ring<T> {
    buf: VecDeque<T>,
    cap: usize,
}

impl<T> Ring<T> {
    pub fn new(cap: usize) -> Self {
        Self {
            buf: VecDeque::new(),
            cap: cap.max(1),
        }
    }

    pub fn push(&mut self, v: T) {
        if self.buf.len() == self.cap {
            self.buf.pop_front();
        }
        self.buf.push_back(v);
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn get(&self, i: usize) -> Option<&T> {
        self.buf.get(i)
    }

    pub fn last(&self) -> Option<&T> {
        self.buf.back()
    }

    /// Newest first — the order chat/console scrollback is drawn in.
    pub fn iter_rev(&self) -> impl Iterator<Item = &T> {
        self.buf.iter().rev()
    }
}

/// The outcome of asking a [`TextInput`]'s completer to complete the line.
pub enum Completion {
    /// A single match: replace the line with this text.
    Full(String),
    /// Several matches: fill the longest common prefix and offer the candidates
    /// (which the owner typically prints, since the widget cannot).
    Ambiguous(String, Vec<String>),
    /// Nothing to complete.
    None,
}

/// An editable single line of text.
///
/// The cursor is a byte offset kept on a `char` boundary by construction — every
/// mutation goes through a method that steps by whole characters, so UTF-8 text
/// can never panic a `String::insert`/`remove`. History recall stashes the live
/// line as a draft so walking back down restores it.
pub struct TextInput {
    text: String,
    cursor: usize,
    max: usize,
    history: Ring<String>,
    /// `Some(i)` while browsing history at index `i`; `None` when editing live.
    scrub: Option<usize>,
    draft: String,
    completer: Option<fn(&str) -> Completion>,
    /// Candidate list produced by the last ambiguous Tab, for the owner to show.
    notice: Option<Vec<String>>,
}

impl TextInput {
    pub fn new(max: usize) -> Self {
        Self {
            text: String::new(),
            cursor: 0,
            max,
            history: Ring::new(64),
            scrub: None,
            draft: String::new(),
            completer: None,
            notice: None,
        }
    }

    /// Attach a Tab-completion source (e.g. the command table).
    pub fn with_completer(mut self, f: fn(&str) -> Completion) -> Self {
        self.completer = Some(f);
        self
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    /// Byte offset of the cursor within [`text`](Self::text), on a char boundary.
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Clear the line (but keep history), e.g. when the field is opened.
    pub fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
        self.scrub = None;
        self.draft.clear();
    }

    /// Replace the line's contents and park the cursor at the end.
    pub fn set(&mut self, s: impl Into<String>) {
        self.text = s.into();
        self.cursor = self.text.len();
        self.scrub = None;
    }

    /// Candidate list from the most recent ambiguous completion, consumed once.
    pub fn take_notice(&mut self) -> Option<Vec<String>> {
        self.notice.take()
    }

    /// Drive one frame of editing. Returns the submitted line (trimmed,
    /// non-empty) when Enter is pressed, otherwise `None`. Esc is left to the
    /// owner so it can decide what closing a field means.
    pub fn handle(&mut self, eng: &Engine) -> Option<String> {
        let ctrl = eng.is_key_down(Key::LeftControl) || eng.is_key_down(Key::RightControl);

        // Typed characters. While Ctrl is held we skip insertion so chords like
        // Ctrl+U don't also deposit a stray glyph.
        while let Some(c) = eng.get_char_pressed() {
            if !ctrl && !c.is_control() && self.text.len() + c.len_utf8() <= self.max {
                self.text.insert(self.cursor, c);
                self.cursor += c.len_utf8();
                self.scrub = None;
            }
        }

        if eng.is_key_pressed(Key::Left) {
            self.move_left();
        }
        if eng.is_key_pressed(Key::Right) {
            self.move_right();
        }
        if eng.is_key_pressed(Key::Home) {
            self.cursor = 0;
        }
        if eng.is_key_pressed(Key::End) {
            self.cursor = self.text.len();
        }
        if eng.is_key_pressed(Key::Backspace) {
            if ctrl {
                self.delete_word();
            } else {
                self.backspace();
            }
        }
        if eng.is_key_pressed(Key::Delete) {
            self.delete_forward();
        }
        if ctrl && eng.is_key_pressed(Key::U) {
            self.text.clear();
            self.cursor = 0;
            self.scrub = None;
        }
        if eng.is_key_pressed(Key::Up) {
            self.history_prev();
        }
        if eng.is_key_pressed(Key::Down) {
            self.history_next();
        }
        if eng.is_key_pressed(Key::Tab) {
            self.try_complete();
        }
        if eng.is_key_pressed(Key::Enter) {
            let line = std::mem::take(&mut self.text).trim().to_string();
            self.cursor = 0;
            self.scrub = None;
            self.draft.clear();
            if !line.is_empty() {
                self.push_history(line.clone());
                return Some(line);
            }
        }
        None
    }

    fn move_left(&mut self) {
        if let Some((i, _)) = self.text[..self.cursor].char_indices().next_back() {
            self.cursor = i;
        }
    }

    fn move_right(&mut self) {
        if let Some(c) = self.text[self.cursor..].chars().next() {
            self.cursor += c.len_utf8();
        }
    }

    fn backspace(&mut self) {
        if let Some((i, _)) = self.text[..self.cursor].char_indices().next_back() {
            self.text.remove(i);
            self.cursor = i;
            self.scrub = None;
        }
    }

    fn delete_forward(&mut self) {
        if self.cursor < self.text.len() {
            self.text.remove(self.cursor);
            self.scrub = None;
        }
    }

    /// Delete from the cursor back to the start of the previous word: skip any
    /// run of whitespace, then the word before it.
    fn delete_word(&mut self) {
        let left = &self.text[..self.cursor];
        let trimmed = left.trim_end_matches(char::is_whitespace);
        let start = match trimmed.rfind(char::is_whitespace) {
            Some(i) => i + trimmed[i..].chars().next().map_or(1, char::len_utf8),
            None => 0,
        };
        self.text.replace_range(start..self.cursor, "");
        self.cursor = start;
        self.scrub = None;
    }

    fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let next = match self.scrub {
            None => {
                self.draft = self.text.clone();
                self.history.len() - 1
            }
            Some(0) => 0,
            Some(i) => i - 1,
        };
        self.scrub = Some(next);
        if let Some(entry) = self.history.get(next) {
            self.text = entry.clone();
            self.cursor = self.text.len();
        }
    }

    fn history_next(&mut self) {
        let Some(i) = self.scrub else {
            return;
        };
        if i + 1 < self.history.len() {
            self.scrub = Some(i + 1);
            if let Some(entry) = self.history.get(i + 1) {
                self.text = entry.clone();
                self.cursor = self.text.len();
            }
        } else {
            // Past the newest entry: back to the line we were typing.
            self.scrub = None;
            self.text = std::mem::take(&mut self.draft);
            self.cursor = self.text.len();
        }
    }

    fn push_history(&mut self, line: String) {
        // Skip consecutive duplicates so holding a repeat doesn't spam history.
        if self.history.last() != Some(&line) {
            self.history.push(line);
        }
    }

    fn try_complete(&mut self) {
        let Some(f) = self.completer else {
            return;
        };
        match f(&self.text) {
            Completion::Full(s) => self.set(s),
            Completion::Ambiguous(prefix, cands) => {
                self.set(prefix);
                self.notice = Some(cands);
            }
            Completion::None => {}
        }
    }
}

/// Longest common prefix of a set of candidate strings (byte-wise, but only ever
/// called on ASCII command names so it stays on char boundaries).
pub fn common_prefix(items: &[&str]) -> String {
    let Some(first) = items.first() else {
        return String::new();
    };
    let mut end = first.len();
    for s in &items[1..] {
        end = first
            .bytes()
            .zip(s.bytes())
            .take(end)
            .take_while(|(a, b)| a == b)
            .count();
    }
    first[..end].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_drops_oldest_past_cap() {
        let mut r = Ring::new(2);
        r.push(1);
        r.push(2);
        r.push(3);
        assert_eq!(r.len(), 2);
        assert_eq!(r.get(0), Some(&2));
        assert_eq!(r.last(), Some(&3));
    }

    #[test]
    fn anchor_resolves_edges_with_insets() {
        let screen = (800, 600);
        let size = (100, 20);
        // Top-left: just the margin.
        assert_eq!(Anchor::TopLeft.origin(screen, size, (10, 12)), (10, 12));
        // Top-right: inset from the right edge by 12 (negative offset).
        assert_eq!(Anchor::TopRight.origin(screen, size, (-12, 12)), (800 - 100 - 12, 12));
        // Top-centre: horizontally centred, offset ignored horizontally here.
        assert_eq!(Anchor::Top.origin(screen, size, (0, 12)), ((800 - 100) / 2, 12));
    }

    #[test]
    fn hud_mode_cycles_and_gates() {
        assert!(HudMode::Full.shows_info());
        assert!(!HudMode::Minimal.shows_info());
        assert!(HudMode::Minimal.shows_world_ui());
        assert!(!HudMode::Off.shows_world_ui());
        assert_eq!(HudMode::Off.next(), HudMode::Full);
    }

    #[test]
    fn common_prefix_of_candidates() {
        assert_eq!(common_prefix(&["tp", "teleport"]), "t");
        assert_eq!(common_prefix(&["gfx"]), "gfx");
        assert_eq!(common_prefix(&["pos", "gfx"]), "");
    }
}
