//! Pluggable rendering for untyped menu screens; mods can provide their own theme.
use voxel_engine::{Color, Frame};

use crate::menu::{Level, Notice, RowKind, Style, ValueView};
use crate::ui::shadowed;

const MENU_BG: Color = Color::new(18, 20, 28, 255);

/// Untyped row for rendering; type tag dropped.
pub struct PresentedRow {
    pub label: String,
    pub detail: Option<String>,
    pub kind: RowKind,
    pub selectable: bool,
}

/// Screen ready to render.
pub struct PresentedView {
    pub title: String,
    pub style: Style,
    pub rows: Vec<PresentedRow>,
    /// User base scale (`Settings::menu_scale`); themes still shrink to fit.
    pub scale: f32,
    pub hint: String,
    pub notice: Option<Notice>,
}

/// A row's screen rectangle — the geometry drawing and hit-testing share.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowRect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

/// Pluggable theme with default layout and drawing.
pub trait MenuTheme {
    fn layout(&self, v: &PresentedView, w: i32, h: i32) -> Vec<RowRect> {
        default_layout(v, w, h)
    }

    fn draw(&self, f: &mut Frame, v: &PresentedView, sel: usize, w: i32, h: i32) {
        default_draw(f, v, sel, w, h);
    }
}

/// Standard theme implementation.
pub struct DefaultTheme;

impl MenuTheme for DefaultTheme {}

// Shared metrics.

/// Bottom strip reserved for the hint and notice lines.
const RESERVE: i32 = 60;

struct Metrics {
    fs: i32,
    line_h: i32,
    start_y: i32,
    /// Left edge for panel rows; centred styles ignore it.
    x: i32,
    centered: bool,
}

/// Base font size scaled by the user's menu scale, then capped so every row
/// fits on screen (the cap wins over the scale; floor keeps text legible).
fn fit_fs(base: i32, scale: f32, cap: i32) -> i32 {
    ((base as f32 * scale) as i32).min(cap).max(8)
}

fn metrics(style: &Style, w: i32, h: i32, rows: usize, scale: f32) -> Metrics {
    let n = rows.max(1) as i32;
    match style {
        Style::Title { .. } => {
            // Rows hang from mid-screen; the cap keeps the last row above the
            // hint strip: (n-2)·line_h + fs ≤ h/2 - RESERVE, line_h = 3·fs/2.
            let cap = (h - 2 * RESERVE) / (3 * n.max(2) - 4);
            let fs = fit_fs(28, scale, cap);
            let line_h = fs * 3 / 2;
            Metrics { fs, line_h, start_y: h / 2 - line_h, x: 0, centered: true }
        }
        Style::Panel => {
            // Rows are centred between the title zone (title_fs = fs + 14 at
            // h/8) and the hint strip; the cap solves n·(7·fs/4) ≤ that region.
            let cap = (7 * h - 8 * (RESERVE + 26)) / (14 * n + 8);
            let fs = fit_fs(26, scale, cap);
            let line_h = fs * 7 / 4;
            let top = h / 8 + (fs + 14) + 12;
            let start_y = top + (h - RESERVE - top - n * line_h).max(0) / 2;
            Metrics { fs, line_h, start_y, x: w / 2 - fs * 10, centered: false }
        }
    }
}

fn default_layout(v: &PresentedView, w: i32, h: i32) -> Vec<RowRect> {
    let m = metrics(&v.style, w, h, v.rows.len(), v.scale);
    v.rows
        .iter()
        .enumerate()
        .map(|(i, _)| RowRect {
            x: if m.centered { 0 } else { m.x },
            y: m.start_y + m.line_h * i as i32,
            w: if m.centered { w } else { m.fs * 20 },
            h: m.fs,
        })
        .collect()
}

// Drawing.

fn default_draw(f: &mut Frame, v: &PresentedView, sel: usize, w: i32, h: i32) {
    f.draw_rect(0, 0, w, h, MENU_BG);
    match &v.style {
        Style::Title { subtitle } => draw_title(f, v, subtitle, sel, w, h),
        Style::Panel => draw_panel(f, v, sel, w, h),
    }
    draw_notice(f, v, w, h);
    draw_hint(f, &v.hint, w, h);
}

fn draw_title(f: &mut Frame, v: &PresentedView, subtitle: &str, sel: usize, w: i32, h: i32) {
    let title_fs = 48;
    let tx = (w - f.measure_text(&v.title, title_fs)) / 2;
    shadowed(f, &v.title, tx, h / 6, title_fs, Color::GOLD);

    let sub_fs = 20;
    let sx = (w - f.measure_text(subtitle, sub_fs)) / 2;
    shadowed(f, subtitle, sx, h / 6 + title_fs + 8, sub_fs, Color::GRAY);

    let m = metrics(&v.style, w, h, v.rows.len(), v.scale);
    for (i, row) in v.rows.iter().enumerate() {
        let selected = i == sel;
        let text = row_body(row, selected);
        let color = row_color(row, selected);
        let x = (w - f.measure_text(&text, m.fs)) / 2;
        shadowed(f, &text, x, m.start_y + m.line_h * i as i32, m.fs, color);
    }
}

fn draw_panel(f: &mut Frame, v: &PresentedView, sel: usize, w: i32, h: i32) {
    let m = metrics(&v.style, w, h, v.rows.len(), v.scale);
    let title_fs = m.fs + 14;
    let tx = (w - f.measure_text(&v.title, title_fs)) / 2;
    shadowed(f, &v.title, tx, h / 8, title_fs, Color::GOLD);

    if v.rows.is_empty() {
        shadowed(f, "  (nothing here)", m.x, m.start_y, m.fs, Color::GRAY);
    }
    let detail_fs = (m.line_h - m.fs - 2).clamp(8, 16);
    for (i, row) in v.rows.iter().enumerate() {
        let selected = i == sel;
        let y = m.start_y + m.line_h * i as i32;
        shadowed(f, &row_body(row, selected), m.x, y, m.fs, row_color(row, selected));
        if let Some(detail) = &row.detail {
            shadowed(f, detail, m.x + m.fs * 3 / 2, y + m.fs + 2, detail_fs, Color::DARKGRAY);
        }
    }
}

fn row_body(row: &PresentedRow, selected: bool) -> String {
    let mark = if selected { ">" } else { " " };
    match &row.kind {
        RowKind::Action | RowKind::Heading => format!("{} {}", mark, row.label),
        RowKind::Value(ValueView::Toggle(on)) => {
            format!("{} {} {}", mark, if *on { "[x]" } else { "[ ]" }, row.label)
        }
        RowKind::Value(ValueView::Choice(value)) => {
            format!("{} {}: < {} >", mark, row.label, value)
        }
        RowKind::Value(ValueView::Bar { t, label }) => {
            format!("{} {}: {} {}", mark, row.label, bar(*t), label)
        }
        RowKind::Text { content, caret, masked } => {
            let shown: Vec<char> = if *masked {
                std::iter::repeat_n('*', content.chars().count()).collect()
            } else {
                content.chars().collect()
            };
            let body: String = if selected {
                let at = (*caret).min(shown.len());
                shown[..at].iter().chain(&['_']).chain(&shown[at..]).collect()
            } else {
                shown.into_iter().collect()
            };
            format!("{} {}: {}", mark, row.label, body)
        }
    }
}

/// Fixed-width bar for consistent UI layout.
fn bar(t: f32) -> String {
    const CELLS: usize = 10;
    let filled = (t.clamp(0.0, 1.0) * CELLS as f32).round() as usize;
    let mut s = String::with_capacity(CELLS + 2);
    s.push('[');
    for i in 0..CELLS {
        s.push(if i < filled { '#' } else { '-' });
    }
    s.push(']');
    s
}

fn row_color(row: &PresentedRow, selected: bool) -> Color {
    if !row.selectable {
        Color::DARKGRAY
    } else if selected {
        Color::RAYWHITE
    } else {
        Color::GRAY
    }
}

fn draw_hint(f: &mut Frame, hint: &str, w: i32, h: i32) {
    if hint.is_empty() {
        return;
    }
    let hint_fs = 18;
    let hx = (w - f.measure_text(hint, hint_fs)) / 2;
    shadowed(f, hint, hx, h - 40, hint_fs, Color::DARKGRAY);
}

fn draw_notice(f: &mut Frame, v: &PresentedView, w: i32, h: i32) {
    if let Some(notice) = &v.notice {
        let fs = 18;
        let color = match notice.level {
            Level::Info => Color::SALMON,
            Level::Error => Color::RED,
        };
        let x = (w - f.measure_text(&notice.text, fs)) / 2;
        shadowed(f, &notice.text, x, h - 40 - (fs + 8), fs, color);
    }
}
