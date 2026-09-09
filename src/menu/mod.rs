//! Menu screens: input flows router events -> Intent -> Msg -> Command.
//! Only the App interprets AppEffect; Presentation folds via MenuTheme.
use voxel_engine::Frame;

use crate::menu::theme::MenuTheme;
use crate::mods::Mods;
use crate::render_config::VisualGroup;
use crate::session::Session;
use crate::settings::Settings;

pub mod input;
pub mod menus;
pub mod start;
pub mod theme;

pub use input::gather;
pub use start::{HostInfo, JoinInfo, MenuModel, StartAction, StartFacts, StartScreen, VERSION};
pub use theme::{DefaultTheme, MenuTheme as _, PresentedRow, PresentedView, RowRect};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dir {
    Prev,
    Next,
}

impl Dir {
    pub fn delta(self) -> i32 {
        match self {
            Dir::Prev => -1,
            Dir::Next => 1,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TextOp {
    Char(char),
    Backspace,
    DelWord,
    Left,
    Right,
    Home,
    End,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Intent {
    Nav(Dir),
    Adjust(Dir),
    Confirm,
    Cancel,
    Edit(TextOp),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Level {
    /// Soft status (failed connect) — salmon.
    Info,
    /// Refused action (bad port) — red.
    Error,
}

#[derive(Clone)]
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

#[derive(Clone)]
pub enum Style {
    Title { subtitle: String },
    Panel,
}

#[derive(Clone)]
pub enum ValueView {
    Toggle(bool),
    Choice(String),
    Bar { t: f32, label: String },
}

/// Decides which Msg a Confirm/Adjust emits.
#[derive(Clone)]
pub enum RowKind {
    Action,
    Value(ValueView),
    Text { content: String, caret: usize, masked: bool },
    Heading,
}

/// Carries action `A` directly (no id ladder).
#[derive(Clone)]
pub struct Row<A: Copy> {
    pub label: String,
    pub detail: Option<String>,
    pub kind: RowKind,
    /// None means not selectable (heading/separator/disabled).
    pub tag: Option<A>,
}

impl<A: Copy> Row<A> {
    pub fn action(label: impl Into<String>, tag: A) -> Self {
        Self { label: label.into(), detail: None, kind: RowKind::Action, tag: Some(tag) }
    }

    pub fn value(label: impl Into<String>, view: ValueView, tag: A) -> Self {
        Self { label: label.into(), detail: None, kind: RowKind::Value(view), tag: Some(tag) }
    }

    pub fn text(label: impl Into<String>, content: String, caret: usize, masked: bool, tag: A) -> Self {
        Self { label: label.into(), detail: None, kind: RowKind::Text { content, caret, masked }, tag: Some(tag) }
    }

    pub fn detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }

    /// Non-selectable section title.
    pub fn heading(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            detail: None,
            kind: RowKind::Heading,
            tag: None,
        }
    }
}

pub struct View<A: Copy> {
    pub title: String,
    pub style: Style,
    pub rows: Vec<Row<A>>,
    /// Confirm while editing a Text row picks this (form submit).
    pub default: Option<A>,
    pub hint: String,
    pub notice: Option<Notice>,
}

impl<A: Copy> View<A> {
    pub fn is_selectable(&self, i: usize) -> bool {
        self.rows.get(i).is_some_and(|r| r.tag.is_some())
    }

    fn tag_at(&self, i: usize) -> Option<A> {
        self.rows.get(i).and_then(|r| r.tag)
    }

    fn kind_at(&self, i: usize) -> Option<&RowKind> {
        self.rows.get(i).map(|r| &r.kind)
    }
}

/// Menu's own action vocabulary.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Msg<A: Copy> {
    Pick(A),
    Step(A, Dir),
    Edited(A, TextOp),
    Back,
}

/// A menu's answer to a message. The stack interprets Pop/Push; only the App interprets Effect.
pub enum Command {
    Stay,
    Pop,
    Push(Box<dyn Screen>),
    Effect(AppEffect),
}

/// A side effect only the App can carry out. Menus emit these instead of
/// touching app state. Start-screen actions ([`StartAction`]) map 1:1 onto
/// the start-related variants; the rest are mods-menu effects.
pub enum AppEffect {
    NewWorld,
    Load(crate::save::SlotId),
    Host(HostInfo),
    Join(JoinInfo),
    /// Push the core Settings hub (start screens emit this instead of pushing).
    Settings,
    /// Push the core Mods screen (start screens emit this instead of pushing).
    Mods,
    ToggleMod(usize),
    StepModKnob { mod_index: usize, knob: usize, delta: i32 },
    /// Enable or disable every member of a group (persists as per-mod lines).
    SetGroup { id: &'static str, on: bool },
    Quit,
}

/// Per-frame snapshot of installed mods.
pub struct ModRow {
    pub name: String,
    pub description: String,
    pub enabled: bool,
    /// `(label, value, hint)` per knob.
    pub knobs: Vec<(String, String, String)>,
    pub visual_group: Option<VisualGroup>,
    pub worldgen: bool,
    /// Group id (`""` if ungrouped).
    pub group: String,
}

impl ModRow {
    pub fn snapshot(mods: &Mods) -> Vec<ModRow> {
        (0..mods.len())
            .map(|i| ModRow {
                name: mods.name(i).to_string(),
                description: mods.description(i).to_string(),
                enabled: mods.is_enabled(i),
                knobs: mods
                    .knobs(i)
                    .into_iter()
                    .map(|k| (k.label.to_string(), k.value, k.hint))
                    .collect(),
                visual_group: mods.visual_group(i),
                worldgen: mods.is_worldgen(i),
                group: mods.group(i).to_string(),
            })
            .collect()
    }
}

/// Everything a menu may read or mutate while running. Settings step in place;
/// everything else is read and turned into AppEffect.
pub struct Ctx<'a> {
    pub settings: &'a mut Settings,
    pub saves: &'a [crate::save::Slot],
    pub mods: &'a [ModRow],
    pub session: &'a Session,
    /// Last `mods.cfg` write error, shown on the Mods screen.
    pub mods_save_error: Option<&'a str>,
}

/// Pure view, effectful update.
pub trait Menu {
    type Action: Copy;
    fn view(&self, ctx: &Ctx) -> View<Self::Action>;
    fn update(&mut self, msg: Msg<Self::Action>, ctx: &mut Ctx) -> Command;
}

/// Type-erased screen on the menu stack.
pub trait Screen {
    fn update(&mut self, intents: &[Intent], ctx: &mut Ctx) -> Command;
    fn draw(&self, ctx: &Ctx, theme: &dyn MenuTheme, f: &mut Frame, w: i32, h: i32);
}

/// Always resolvable to a selectable row.
#[derive(Default)]
pub struct Cursor {
    pub index: usize,
}

impl Cursor {
    pub fn normalize<A: Copy>(&mut self, view: &View<A>) {
        self.index = self.resolved(view);
    }

    /// Immutable resolution (for use in draw).
    pub fn resolved<A: Copy>(&self, view: &View<A>) -> usize {
        let n = view.rows.len();
        if n == 0 {
            return 0;
        }
        let start = self.index.min(n - 1);
        if view.is_selectable(start) {
            return start;
        }
        // Search outward for the nearest selectable row.
        for d in 1..n {
            if start >= d && view.is_selectable(start - d) {
                return start - d;
            }
            if start + d < n && view.is_selectable(start + d) {
                return start + d;
            }
        }
        start
    }

    fn nav<A: Copy>(&mut self, view: &View<A>, dir: Dir) {
        let n = view.rows.len();
        if n == 0 {
            return;
        }
        let step = |i: usize| match dir {
            Dir::Next => (i + 1) % n,
            Dir::Prev => (i + n - 1) % n,
        };
        let mut i = step(self.index.min(n - 1));
        for _ in 0..n {
            if view.is_selectable(i) {
                self.index = i;
                return;
            }
            i = step(i);
        }
    }
}

/// Type erasure point: a concrete Menu plus its cursor, as a Screen.
pub struct Framed<M: Menu> {
    menu: M,
    cursor: Cursor,
}

impl<M: Menu> Framed<M> {
    pub fn new(menu: M) -> Self {
        Self { menu, cursor: Cursor::default() }
    }

    /// Box a menu into a screen for Push.
    pub fn boxed(menu: M) -> Box<dyn Screen>
    where
        M: 'static,
    {
        Box::new(Self::new(menu))
    }

    pub fn view_sel(&self, ctx: &Ctx) -> (View<M::Action>, usize) {
        let view = self.menu.view(ctx);
        let sel = self.cursor.resolved(&view);
        (view, sel)
    }
}

impl<M: Menu> Screen for Framed<M> {
    fn update(&mut self, intents: &[Intent], ctx: &mut Ctx) -> Command {
        let view = self.menu.view(ctx);
        self.cursor.normalize(&view);
        match drive(intents, &view, &mut self.cursor) {
            Some(msg) => self.menu.update(msg, ctx),
            None => Command::Stay,
        }
    }

    fn draw(&self, ctx: &Ctx, theme: &dyn MenuTheme, f: &mut Frame, w: i32, h: i32) {
        let view = self.menu.view(ctx);
        let sel = self.cursor.resolved(&view);
        let pv = present(&view, ctx.settings.menu_scale);
        theme.draw(f, &pv, sel, w, h);
    }
}

/// The pushdown stack of screens. Non-empty by construction; Back at the root is
/// ignored.
pub struct MenuStack {
    frames: Vec<Box<dyn Screen>>,
}

impl MenuStack {
    pub fn new(root: Box<dyn Screen>) -> Self {
        Self { frames: vec![root] }
    }

    pub fn update(&mut self, intents: &[Intent], ctx: &mut Ctx) -> Option<AppEffect> {
        let cmd = self.frames.last_mut().expect("non-empty stack").update(intents, ctx);
        match cmd {
            Command::Stay => None,
            Command::Pop => {
                if self.frames.len() > 1 {
                    self.frames.pop();
                }
                None
            }
            Command::Push(screen) => {
                self.frames.push(screen);
                None
            }
            Command::Effect(effect) => Some(effect),
        }
    }

    pub fn push(&mut self, screen: Box<dyn Screen>) {
        self.frames.push(screen);
    }

    pub fn depth(&self) -> usize {
        self.frames.len()
    }

    pub fn draw(&self, ctx: &Ctx, theme: &dyn MenuTheme, f: &mut Frame, w: i32, h: i32) {
        self.frames.last().expect("non-empty stack").draw(ctx, theme, f, w, h);
    }
}

/// On Text rows, editing has priority so characters don't trigger nav.
pub fn drive<A: Copy>(intents: &[Intent], view: &View<A>, cursor: &mut Cursor) -> Option<Msg<A>> {
    let n = view.rows.len();
    if n == 0 {
        return cancel(intents).then_some(Msg::Back);
    }

    let sel = cursor.index.min(n - 1);
    let on_text = matches!(view.kind_at(sel), Some(RowKind::Text { .. }));

    if on_text {
        let tag = view.tag_at(sel);
        // Edit has priority so chars/backspace don't trigger nav.
        for i in intents {
            if let Intent::Edit(op) = i && let Some(t) = tag {
                return Some(Msg::Edited(t, *op));
            }
        }
        // Adjust moves the caret.
        for i in intents {
            if let Intent::Adjust(d) = i && let Some(t) = tag {
                let op = if *d == Dir::Prev { TextOp::Left } else { TextOp::Right };
                return Some(Msg::Edited(t, op));
            }
        }
        if confirm(intents) && let Some(a) = view.default {
            return Some(Msg::Pick(a));
        }
        if cancel(intents) {
            return Some(Msg::Back);
        }
        // Nav between fields: cursor moves but yields no message.
        for i in intents {
            if let Intent::Nav(d) = i {
                cursor.nav(view, *d);
            }
        }
        return None;
    }

    // Non-text row: navigate first, then act on the row the highlight lands on.
    for i in intents {
        if let Intent::Nav(d) = i {
            cursor.nav(view, *d);
        }
    }
    if cancel(intents) {
        return Some(Msg::Back);
    }
    let sel = cursor.index.min(n - 1);
    let tag = view.tag_at(sel)?;
    match view.kind_at(sel) {
        Some(RowKind::Value(_)) => {
            for i in intents {
                if let Intent::Adjust(d) = i {
                    return Some(Msg::Step(tag, *d));
                }
            }
            if confirm(intents) {
                return Some(Msg::Step(tag, Dir::Next));
            }
        }
        Some(RowKind::Action) if confirm(intents) => {
            return Some(Msg::Pick(tag));
        }
        _ => {}
    }
    None
}

fn confirm(intents: &[Intent]) -> bool {
    intents.iter().any(|i| matches!(i, Intent::Confirm))
}

fn cancel(intents: &[Intent]) -> bool {
    intents.iter().any(|i| matches!(i, Intent::Cancel))
}

pub fn present<A: Copy>(view: &View<A>, scale: f32) -> PresentedView {
    let rows = view
        .rows
        .iter()
        .map(|r| PresentedRow {
            label: r.label.clone(),
            detail: r.detail.clone(),
            kind: r.kind.clone(),
            selectable: r.tag.is_some(),
        })
        .collect();
    PresentedView {
        title: view.title.clone(),
        style: view.style.clone(),
        rows,
        scale,
        hint: view.hint.clone(),
        notice: view.notice.clone(),
    }
}

pub const PORT_ERROR: &str = "invalid port (1-65535)";

/// Parse a port field. Empty means DEFAULT_PORT; otherwise 1-65535.
pub fn parse_port(text: &str) -> Option<u16> {
    let text = text.trim();
    if text.is_empty() {
        return Some(crate::net::DEFAULT_PORT);
    }
    match text.parse::<u16>() {
        Ok(0) | Err(_) => None,
        Ok(port) => Some(port),
    }
}

pub fn apply_text_op(buf: &mut crate::ui::EditBuf, op: TextOp) {
    match op {
        TextOp::Char(c) => {
            buf.insert_char(c);
        }
        TextOp::Backspace => {
            buf.backspace();
        }
        TextOp::DelWord => buf.delete_word(),
        TextOp::Left => buf.left(),
        TextOp::Right => buf.right(),
        TextOp::Home => buf.home(),
        TextOp::End => buf.end(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confirm_on_a_choice_row_steps_forward_like_right() {
        let view = View {
            title: String::new(),
            style: Style::Panel,
            rows: vec![Row::value("Tile", ValueView::Choice("32".into()), 0)],
            default: None,
            hint: String::new(),
            notice: None,
        };
        let mut cursor = Cursor::default();
        assert_eq!(
            drive(&[Intent::Confirm], &view, &mut cursor),
            Some(Msg::Step(0, Dir::Next))
        );
        assert_eq!(
            drive(&[Intent::Adjust(Dir::Next)], &view, &mut cursor),
            Some(Msg::Step(0, Dir::Next))
        );
    }
}
