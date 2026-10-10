//! Menu screens: input flows router events -> Intent -> Msg -> Command.
//! Only the App interprets AppEffect; Presentation folds via MenuTheme.
use std::cell::Cell;

use voxel_engine::Frame;

use crate::menu::theme::MenuTheme;
use crate::modding::{BuildInfo, VisualMask};
use crate::session::Session;
use crate::settings::{Options, OptionsView, Settings};

pub mod input;
pub mod menus;
pub mod start;
pub mod theme;

pub use input::gather;
pub use start::{HostInfo, JoinInfo, StartScreen};
pub use theme::{MenuTheme as _, PresentedRow, PresentedView};

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

#[derive(Clone)]
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
    /// A character typed while an action row is highlighted (a row shortcut, e.g. D to delete).
    Key(A, char),
    /// Navigation moved the highlight onto this row and nothing else happened this frame.
    Hover(A),
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
/// touching app state. Start-screen actions ([`StartAction`](start::StartAction)) map 1:1 onto
/// these.
pub enum AppEffect {
    NewWorld,
    Load(crate::save::SlotId),
    /// Move a saved world to the trash.
    DeleteWorld(crate::save::SlotId),
    Host(HostInfo),
    Join(JoinInfo),
    /// Push the core Settings hub (start screens emit this instead of pushing).
    Settings,
    /// Push the core Mods screen (start screens emit this instead of pushing).
    Mods,
    Quit,
}

/// Everything a menu may read or mutate while running. Settings step in place;
/// everything else is read and turned into AppEffect.
pub struct Ctx<'a> {
    pub settings: &'a mut Settings,
    /// The packages' options; settings pages list them beside the core's settings.
    pub options: &'a mut Options,
    pub saves: &'a [crate::save::Slot],
    pub session: &'a Session,
    /// Every package compiled into this build.
    pub build: &'a BuildInfo,
    /// Package ids the core suspended for this session.
    pub suspended: &'a [String],
    /// The visual groups the installed, unsuspended mods provide.
    pub visuals: VisualMask,
}

impl<'a> Ctx<'a> {
    /// A context over `settings`, `options` and `session` with no saves, an empty build and every
    /// visual group provided.
    pub fn bare(settings: &'a mut Settings, options: &'a mut Options, session: &'a Session) -> Self {
        const EMPTY: &BuildInfo = &BuildInfo::EMPTY;
        Self { settings, options, saves: &[], session, build: EMPTY, suspended: &[], visuals: VisualMask::ALL }
    }

    /// Every tunable, the core's settings and the packages' options, through one interface.
    pub fn view(&mut self) -> OptionsView<'_> {
        OptionsView::new(self.settings, self.options)
    }
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
    /// The view an update built when no message reached the menu, kept for the frame's draw.
    view: Cell<Option<View<M::Action>>>,
}

impl<M: Menu> Framed<M> {
    pub fn new(menu: M) -> Self {
        Self { menu, cursor: Cursor::default(), view: Cell::new(None) }
    }

    /// Box a menu into a screen for Push.
    pub fn boxed(menu: M) -> Box<dyn Screen>
    where
        M: 'static,
    {
        Box::new(Self::new(menu))
    }

    /// The view to show and its selected row: the one the last update built if no message has
    /// reached the menu since, else a fresh one.
    pub fn view_sel(&self, ctx: &Ctx) -> (View<M::Action>, usize) {
        let view = self.view.take().unwrap_or_else(|| self.menu.view(ctx));
        let sel = self.cursor.resolved(&view);
        (view, sel)
    }
}

impl<M: Menu> Screen for Framed<M> {
    fn update(&mut self, intents: &[Intent], ctx: &mut Ctx) -> Command {
        let view = self.menu.view(ctx);
        self.cursor.normalize(&view);
        match drive(intents, &view, &mut self.cursor) {
            Some(msg) => {
                *self.view.get_mut() = None;
                self.menu.update(msg, ctx)
            }
            None => {
                *self.view.get_mut() = Some(view);
                Command::Stay
            }
        }
    }

    fn draw(&self, ctx: &Ctx, theme: &dyn MenuTheme, f: &mut Frame, w: i32, h: i32) {
        let (view, sel) = self.view_sel(ctx);
        theme.draw(f, &present(view, ctx.settings.menu_scale), sel, w, h);
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
    let before = cursor.index;
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
        Some(RowKind::Action) => {
            for i in intents {
                if let Intent::Edit(TextOp::Char(c)) = i {
                    return Some(Msg::Key(tag, *c));
                }
            }
        }
        _ => {}
    }
    (cursor.index != before).then_some(Msg::Hover(tag))
}

fn confirm(intents: &[Intent]) -> bool {
    intents.iter().any(|i| matches!(i, Intent::Confirm))
}

fn cancel(intents: &[Intent]) -> bool {
    intents.iter().any(|i| matches!(i, Intent::Cancel))
}

/// The untyped screen at `scale`. A view passed by value moves its strings; a borrowed one is
/// cloned.
pub fn present(view: impl Into<PresentedView>, scale: f32) -> PresentedView {
    PresentedView { scale, ..view.into() }
}

/// At scale 1.
impl<A: Copy> From<View<A>> for PresentedView {
    fn from(view: View<A>) -> Self {
        let rows = view
            .rows
            .into_iter()
            .map(|r| PresentedRow { label: r.label, detail: r.detail, kind: r.kind, selectable: r.tag.is_some() })
            .collect();
        Self { title: view.title, style: view.style, rows, scale: 1.0, hint: view.hint, notice: view.notice }
    }
}

impl<A: Copy> From<&View<A>> for PresentedView {
    fn from(view: &View<A>) -> Self {
        view.clone().into()
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

    fn actions(n: usize) -> View<usize> {
        View {
            title: String::new(),
            style: Style::Panel,
            rows: (0..n).map(|i| Row::action(format!("row {i}"), i)).collect(),
            default: None,
            hint: String::new(),
            notice: None,
        }
    }

    /// A character on an action row is that row's shortcut; Enter still picks it.
    #[test]
    fn a_typed_character_on_an_action_row_is_a_key_message() {
        let view = actions(2);
        let mut cursor = Cursor::default();
        assert_eq!(drive(&[Intent::Edit(TextOp::Char('d'))], &view, &mut cursor), Some(Msg::Key(0, 'd')));
        assert_eq!(drive(&[Intent::Confirm], &view, &mut cursor), Some(Msg::Pick(0)));
    }

    /// A menu that counts the views it builds.
    struct Counted(Cell<usize>);

    impl Menu for Counted {
        type Action = usize;
        fn view(&self, _ctx: &Ctx) -> View<usize> {
            self.0.set(self.0.get() + 1);
            actions(2)
        }
        fn update(&mut self, _msg: Msg<usize>, _ctx: &mut Ctx) -> Command {
            Command::Stay
        }
    }

    /// A frame whose update reaches the menu with no message draws the view that update built.
    #[test]
    fn a_quiet_frame_builds_the_view_once() {
        let mut settings = Settings::default();
        let mut options = Options::new();
        let session = Session::default();
        let mut ctx = Ctx::bare(&mut settings, &mut options, &session);
        let mut framed = Framed::new(Counted(Cell::new(0)));
        let built = |f: &Framed<Counted>| f.menu.0.get();
        assert!(matches!(framed.update(&[], &mut ctx), Command::Stay));
        assert_eq!(framed.view_sel(&ctx).1, 0);
        assert_eq!(built(&framed), 1, "the draw reuses the update's view");
        let _ = framed.update(&[Intent::Nav(Dir::Next)], &mut ctx);
        assert_eq!(framed.view_sel(&ctx).1, 1);
        assert_eq!(built(&framed), 3, "a message reached the menu: the draw builds afresh");
        let _ = framed.view_sel(&ctx);
        assert_eq!(built(&framed), 4, "a draw with no update since builds its own");
    }

    /// Moving the highlight says where it landed; a frame with no movement says nothing.
    #[test]
    fn navigation_reports_the_row_it_lands_on() {
        let view = actions(3);
        let mut cursor = Cursor::default();
        assert_eq!(drive(&[Intent::Nav(Dir::Next)], &view, &mut cursor), Some(Msg::Hover(1)));
        assert_eq!(drive(&[], &view, &mut cursor), None);
        assert_eq!(drive(&[Intent::Nav(Dir::Prev)], &view, &mut cursor), Some(Msg::Hover(0)));
    }
}
