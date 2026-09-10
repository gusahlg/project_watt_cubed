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
use std::borrow::Cow;
use std::collections::VecDeque;
use std::sync::Arc;

use voxel_engine::{Color, Frame};

use crate::input::intent::EditKey;

/// A screen-space size or offset in pixels, `(x, y)`. Kept as a plain tuple so
/// this module needs no vector-math dependency of its own.
pub type Px = (i32, i32);

/// Fit a label into a fixed-width monospace row without splitting Unicode.
/// The renderer advances every glyph by one font-size unit, so character count
/// is the exact layout metric here.
pub fn ellipsize(text: &str, max_chars: usize) -> Cow<'_, str> {
    let len = text.chars().count();
    if len <= max_chars {
        return Cow::Borrowed(text);
    }
    if max_chars <= 3 {
        return Cow::Owned(".".repeat(max_chars));
    }
    Cow::Owned(
        text.chars()
            .take(max_chars - 3)
            .chain("...".chars())
            .collect(),
    )
}

/// A cursor-following slice of `0..total` containing at most `capacity` rows.
/// Used by compact HUD lists so the active crafting row never runs off-screen.
pub fn visible_window(total: usize, cursor: usize, capacity: usize) -> std::ops::Range<usize> {
    if total == 0 || capacity == 0 {
        return 0..0;
    }
    let capacity = capacity.min(total);
    let cursor = cursor.min(total - 1);
    let start = cursor.saturating_sub(capacity / 2).min(total - capacity);
    start..start + capacity
}

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
    /// Per-axis alignment: 0=start, 1=center, 2=end.
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

crate::macros::code_enum! {
    /// How much of the HUD is shown. A three-state cycle rather than a bool: `Minimal`
    /// keeps the reticle (and world-space name tags) but hides the informational text.
    pub enum HudMode {
        Full = 2, ["full", "2"], "Full",
        Minimal = 1, ["minimal", "min", "1"], "Minimal",
        Off = 0, ["off", "0"], "Off",
    }
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

    /// Whether the minimap is shown. Informational like coords/FPS: `Full` only.
    pub fn shows_minimap(self) -> bool {
        matches!(self, HudMode::Full)
    }

    /// Whether mod-contributed HUD widgets (hotbar, stash) are shown. Gameplay
    /// UI like the reticle: everything but `Off`.
    pub fn shows_mod_hud(self) -> bool {
        !matches!(self, HudMode::Off)
    }

}

/// The whole in-world UI look, threaded through drawing. `scale` routes every font
/// size; `hud` is the master visibility cycle.
pub struct Theme {
    pub scale: f32,
    pub crosshair: Crosshair,
    pub hud: HudMode,
}

impl Theme {
    pub fn new() -> Self {
        Self {
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
#[allow(clippy::too_many_arguments)] // HUD label needs frame, theme, placement, and text independently
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
    shadowed(f, text, x, y, fs, color);
}

/// Draw text with a 1px dark drop shadow so it stays readable over bright terrain.
/// The base text-draw primitive: every UI string in the game goes through here.
pub fn shadowed(f: &mut Frame, text: &str, x: i32, y: i32, font_size: i32, color: Color) {
    f.draw_text(text, x + 1, y + 1, font_size, Color::new(0, 0, 0, 180));
    f.draw_text(text, x, y, font_size, color);
}

// HUD widget vocabulary: mods describe what to show as data, [`render_hud`] draws it.

/// One panel row's text plus its emphasis. The panel resolves the role to a
/// colour and ellipsizes the text to the panel width. `swatch` is an optional
/// material colour chip drawn before the text.
#[derive(Clone)]
pub struct Row {
    pub text: Arc<str>,
    pub role: Role,
    pub swatch: Option<Color>,
}

impl Row {
    pub fn new(role: Role, text: impl Into<Arc<str>>) -> Self {
        Self {
            text: text.into(),
            role,
            swatch: None,
        }
    }

    /// Colour chip for a configuration row.
    pub fn with_swatch(mut self, color: Color) -> Self {
        self.swatch = Some(color);
        self
    }
}

/// Shared panel rhythm: one padding/font/line-height for every HUD panel, so
/// panels line up and their heights are computed identically.
pub const PANEL_PAD: i32 = 8;
pub const PANEL_FONT: i32 = 18;
pub const PANEL_LINE: i32 = PANEL_FONT + 4;
/// The translucent background behind every HUD panel.
pub const PANEL_BG: Color = Color::new(8, 10, 14, 200);

/// A translucent HUD box at an absolute screen position: a background sized to
/// its content, header line(s), a small gap, then body rows. Every row is
/// ellipsized to fit `width`. The single owner of panel chrome.
///
/// Header/row lists are `Arc` so a cached panel can be pushed into the
/// frame's HUD buffer without allocating on a stable frame.
#[derive(Clone)]
pub struct Panel {
    pub at: Px,
    pub width: i32,
    pub header: Arc<[Row]>,
    pub rows: Arc<[Row]>,
}

impl Panel {
    /// Total pixel height of the drawn box (padding + header + gap + body). The
    /// body reserves at least one line so an empty panel still frames its box.
    pub fn height(&self) -> i32 {
        let body = self.rows.len().max(1) as i32;
        PANEL_PAD * 2 + self.header.len() as i32 * PANEL_LINE + 2 + body * PANEL_LINE
    }

    fn draw(&self, f: &mut Frame) {
        let (x, y) = self.at;
        f.draw_rect(x, y, self.width, self.height(), PANEL_BG);
        let text_x = x + PANEL_PAD;
        let mut cy = y + PANEL_PAD;
        let mut row = |r: &Row, cy: i32| {
            let mut tx = text_x;
            let mut width = self.width - PANEL_PAD * 2;
            if let Some(color) = r.swatch {
                let chip = PANEL_FONT - 4;
                f.draw_rect(tx, cy + 2, chip, chip, color);
                tx += PANEL_FONT;
                width -= PANEL_FONT;
            }
            let max_chars = (width / PANEL_FONT).max(1) as usize;
            shadowed(f, &ellipsize(&r.text, max_chars), tx, cy, PANEL_FONT, r.role.color());
        };
        for r in self.header.iter() {
            row(r, cy);
            cy += PANEL_LINE;
        }
        cy += 2;
        for r in self.rows.iter() {
            row(r, cy);
            cy += PANEL_LINE;
        }
    }
}

/// One thing a mod contributes to the HUD. Closed on purpose (see the module
/// note): a screen-anchored label or a boxed panel — nothing that lets a mod
/// draw arbitrarily.
#[derive(Clone)]
pub enum HudElement {
    /// A screen-anchored line of text, scaled by the theme.
    Label {
        at: Anchor,
        off: Px,
        base_fs: i32,
        role: Role,
        text: Arc<str>,
    },
    /// A translucent content box at an absolute position.
    Panel(Panel),
}

/// Flatten HUD labels and panel rows to text. Test helper: one place for the
/// inventory and crafting mods to assert on what they painted.
#[cfg(test)]
pub(crate) fn hud_text(elements: &[HudElement]) -> String {
    let mut out = String::new();
    for el in elements {
        match el {
            HudElement::Label { text, .. } => {
                out.push_str(text);
                out.push('\n');
            }
            HudElement::Panel(panel) => {
                for row in panel.header.iter().chain(panel.rows.iter()) {
                    out.push_str(&row.text);
                    out.push('\n');
                }
            }
        }
    }
    out
}

/// Draw every mod's contributed HUD. The only place mod HUD reaches the frame.
pub fn render_hud(f: &mut Frame, theme: &Theme, screen: Px, elements: &[HudElement]) {
    for el in elements {
        match el {
            HudElement::Label { at, off, base_fs, role, text } => {
                label(f, theme, screen, *at, *off, *base_fs, role.color(), text)
            }
            HudElement::Panel(p) => p.draw(f),
        }
    }
}

/// The style role of a run of UI text — its *emphasis*, not a raw RGB. Rendering
/// resolves a role to a colour through [`Role::color`], so text can only ever be an
/// on-palette colour and call sites speak emphasis (`Danger` vs `Dim`) instead of
/// pixels. This is the single colour table for the whole UI: HUD, console, menus,
/// and mods all ask for a role, never a raw `Color::`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    /// Primary text: coords, labels, the brightest normal text.
    Primary,
    /// Normal body text (chat, list rows).
    Muted,
    /// De-emphasised text (system output, secondary detail).
    Dim,
    /// Unavailable / inactive text (a disabled action, an unselected row).
    Disabled,
    /// The active/selected item, links, player names.
    Accent,
    /// A positive value or state (counts, equipped, success).
    Positive,
    /// A caution: the `[global]` tag, a header, a soft warning.
    Warning,
    /// An error or destructive outcome.
    Danger,
}

impl Role {
    pub fn color(self) -> Color {
        match self {
            Role::Primary => Color::WHITE,
            Role::Muted => Color::RAYWHITE,
            Role::Dim => Color::LIGHTGRAY,
            Role::Disabled => Color::GRAY,
            Role::Accent => Color::SKYBLUE,
            Role::Positive => Color::LIME,
            Role::Warning => Color::GOLD,
            Role::Danger => Color::SALMON,
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

/// Editable line of text with a boundary-safe caret and byte cap.
///
/// The cursor is a byte offset kept on a `char` boundary by construction — every
/// mutation steps by whole characters, so UTF-8 text can never panic a
/// `String::insert`/`remove`. The text and caret live in one place so there's a
/// single source of truth for editing state.
pub struct EditBuf {
    text: String,
    cursor: usize,
    max: usize,
}

impl EditBuf {
    pub fn new(max: usize) -> Self {
        Self { text: String::new(), cursor: 0, max }
    }

    /// A buffer pre-filled with `init` (truncated to the cap on a char boundary),
    /// caret at the end.
    pub fn with(init: &str, max: usize) -> Self {
        let mut b = Self::new(max);
        b.set(init);
        b
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Caret as a char index, for drawing a cursor mid-string.
    pub fn caret_chars(&self) -> usize {
        self.text[..self.cursor].chars().count()
    }

    pub fn max(&self) -> usize {
        self.max
    }

    /// Replace the whole value (truncated to the cap), caret to the end.
    pub fn set(&mut self, s: &str) {
        let mut s = s.to_string();
        while s.len() > self.max {
            s.pop();
        }
        self.cursor = s.len();
        self.text = s;
    }

    pub fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
    }

    /// Insert one printable char at the caret if it still fits the cap.
    pub fn insert_char(&mut self, c: char) -> bool {
        if c.is_control() || self.text.len() + c.len_utf8() > self.max {
            return false;
        }
        self.text.insert(self.cursor, c);
        self.cursor += c.len_utf8();
        true
    }

    pub fn backspace(&mut self) -> bool {
        if let Some((i, _)) = self.text[..self.cursor].char_indices().next_back() {
            self.text.remove(i);
            self.cursor = i;
            true
        } else {
            false
        }
    }

    pub fn delete_forward(&mut self) {
        if self.cursor < self.text.len() {
            self.text.remove(self.cursor);
        }
    }

    /// Delete back to the start of the previous word.
    pub fn delete_word(&mut self) {
        let left = &self.text[..self.cursor];
        let trimmed = left.trim_end_matches(char::is_whitespace);
        let start = match trimmed.rfind(char::is_whitespace) {
            Some(i) => i + trimmed[i..].chars().next().map_or(1, char::len_utf8),
            None => 0,
        };
        self.text.replace_range(start..self.cursor, "");
        self.cursor = start;
    }

    pub fn left(&mut self) {
        if let Some((i, _)) = self.text[..self.cursor].char_indices().next_back() {
            self.cursor = i;
        }
    }

    pub fn right(&mut self) {
        if let Some(c) = self.text[self.cursor..].chars().next() {
            self.cursor += c.len_utf8();
        }
    }

    pub fn home(&mut self) {
        self.cursor = 0;
    }

    pub fn end(&mut self) {
        self.cursor = self.text.len();
    }
}

/// An editable single line of text.
///
/// Wraps an [`EditBuf`] for the text/caret, and adds history recall (walking back
/// down restores the live draft) and optional Tab-completion.
pub struct TextInput {
    buf: EditBuf,
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
            buf: EditBuf::new(max),
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
        self.buf.text()
    }

    /// Byte offset of the cursor within [`text`](Self::text), on a char boundary.
    pub fn cursor(&self) -> usize {
        self.buf.cursor()
    }

    /// Clear the line (but keep history), e.g. when the field is opened.
    pub fn clear(&mut self) {
        self.buf.clear();
        self.scrub = None;
        self.draft.clear();
    }

    /// Replace the line's contents and park the cursor at the end.
    pub fn set(&mut self, s: impl Into<String>) {
        self.buf.set(&s.into());
        self.scrub = None;
    }

    /// Candidate list from the most recent ambiguous completion, consumed once.
    pub fn take_notice(&mut self) -> Option<Vec<String>> {
        self.notice.take()
    }

    /// Drive one frame of editing with typed chars and at most one [`EditKey`].
    /// Returns the submitted line (trimmed, non-empty) on [`EditKey::Submit`],
    /// otherwise `None`. Esc is left to the owner so it can decide what closing
    /// a field means.
    pub fn handle(&mut self, chars: &[char], edit: Option<EditKey>) -> Option<String> {
        // Control chars are filtered upstream and by insert_char, so chords never deposit a stray glyph.
        for &c in chars {
            if self.buf.insert_char(c) {
                self.scrub = None;
            }
        }

        match edit {
            Some(EditKey::Left) => self.buf.left(),
            Some(EditKey::Right) => self.buf.right(),
            Some(EditKey::Home) => self.buf.home(),
            Some(EditKey::End) => self.buf.end(),
            Some(EditKey::Backspace) => {
                self.buf.backspace();
                self.scrub = None;
            }
            Some(EditKey::DelWord) => {
                self.buf.delete_word();
                self.scrub = None;
            }
            Some(EditKey::Delete) => {
                self.buf.delete_forward();
                self.scrub = None;
            }
            Some(EditKey::ClearLine) => {
                self.buf.clear();
                self.scrub = None;
            }
            Some(EditKey::HistoryUp) => self.history_prev(),
            Some(EditKey::HistoryDown) => self.history_next(),
            Some(EditKey::Complete) => self.try_complete(),
            Some(EditKey::Submit) => {
                let line = self.buf.text().trim().to_string();
                self.buf.clear();
                self.scrub = None;
                self.draft.clear();
                if !line.is_empty() {
                    self.push_history(line.clone());
                    return Some(line);
                }
            }
            None => {}
        }
        None
    }

    fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let next = match self.scrub {
            None => {
                self.draft = self.buf.text().to_string();
                self.history.len() - 1
            }
            Some(0) => 0,
            Some(i) => i - 1,
        };
        self.scrub = Some(next);
        if let Some(entry) = self.history.get(next) {
            self.buf.set(entry);
        }
    }

    fn history_next(&mut self) {
        let Some(i) = self.scrub else {
            return;
        };
        if i + 1 < self.history.len() {
            self.scrub = Some(i + 1);
            if let Some(entry) = self.history.get(i + 1) {
                self.buf.set(entry);
            }
        } else {
            // Past the newest entry: back to the line we were typing.
            self.scrub = None;
            let draft = std::mem::take(&mut self.draft);
            self.buf.set(&draft);
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
        match f(self.buf.text()) {
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
    fn hud_text_joins_labels_and_panel_rows() {
        let elements = [
            HudElement::Label {
                at: Anchor::TopLeft,
                off: (0, 0),
                base_fs: 16,
                role: Role::Primary,
                text: "hello".into(),
            },
            HudElement::Panel(Panel {
                at: (0, 0),
                width: 100,
                header: vec![Row::new(Role::Primary, "head")].into(),
                rows: vec![Row::new(Role::Muted, "body")].into(),
            }),
        ];
        assert_eq!(hud_text(&elements), "hello\nhead\nbody\n");
    }

    #[test]
    fn hud_mode_cycles_and_gates() {
        assert!(HudMode::Full.shows_info());
        assert!(!HudMode::Minimal.shows_info());
        assert!(HudMode::Minimal.shows_world_ui());
        assert!(!HudMode::Off.shows_world_ui());
        assert_eq!(HudMode::Off.next(), HudMode::Full);
        // Off hides EVERY widget: minimap and mod HUD included, not just the reticle/info text.
        assert!(HudMode::Full.shows_minimap());
        assert!(!HudMode::Minimal.shows_minimap());
        assert!(!HudMode::Off.shows_minimap());
        assert!(HudMode::Full.shows_mod_hud());
        assert!(HudMode::Minimal.shows_mod_hud());
        assert!(!HudMode::Off.shows_mod_hud());
    }

    #[test]
    fn common_prefix_of_candidates() {
        assert_eq!(common_prefix(&["tp", "teleport"]), "t");
        assert_eq!(common_prefix(&["gfx"]), "gfx");
        assert_eq!(common_prefix(&["pos", "gfx"]), "");
    }

    #[test]
    fn ellipsize_and_visible_window_respect_hard_bounds() {
        assert_eq!(ellipsize("Copper+Glass", 9), "Copper...");
        assert_eq!(ellipsize("Iron", 9), "Iron");
        assert_eq!(visible_window(20, 10, 5), 8..13);
        assert_eq!(visible_window(20, 19, 5), 15..20);
        assert_eq!(visible_window(2, 1, 5), 0..2);
        assert_eq!(visible_window(2, 1, 0), 0..0);
    }
}
