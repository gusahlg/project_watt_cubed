//! The screen host: every menu is a mod, and the core only hosts them.
//!
//! A [`Screen`] reads the core's raw menu input and the [`ScreenFacts`], and answers with a
//! [`ScreenOutcome`]: stay, go back, push another screen, open a registered entry, or ask the core
//! for something only it can do ([`AppRequest`]). It draws by pushing [`UiElement`]s, which the
//! core renders with a closed primitive renderer: a screen never touches the frame.
//!
//! Screens reach the player through two slots and an entry registry, so no menu names another:
//! - the **root** slot ([`Mod::root_screen`](crate::modding::Mod::root_screen)): the screen out of
//!   a world (a start screen). It also draws the waiting page while the core connects or loads,
//!   from [`ScreenFacts::phase`];
//! - the **pause** slot ([`Mod::pause_screen`](crate::modding::Mod::pause_screen)): opened by Esc in a
//!   world once text capture and every overlay declined it. It draws over the running world, so a
//!   pause screen paints a translucent backdrop of its own (`facts.in_world` says it is one);
//! - **entries** ([`ModRegistrar::screen_entry`](crate::modding::ModRegistrar::screen_entry)): a
//!   label and an `open` function, offered in [`Places::MAIN`] and/or [`Places::PAUSE`]. A root or
//!   pause screen lists `facts.entries_for(place)` without knowing who registered them.
//!
//! The first active mod that answers a slot wins. Without a root screen the core enters the most
//! recent world (or a new one) and Esc saves and quits; without a pause screen Esc leaves the
//! world as it always did.

use std::borrow::Cow;

use voxel_engine::{Color, Frame};

use crate::input::intent::{EditKey, MenuEvent};
use crate::input::router;
use crate::modding::{BuildInfo, VisualMask};
use crate::session::Session;
use crate::settings::{Options, OptionsRef, OptionsView, Settings};

/// The save-slot types a screen sees in [`ScreenFacts::saves`], re-exported for mods.
pub use crate::save::slot::{SaveError, SaveMeta, Slot, SlotId};

/// The game version a screen may show.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Host form result: port, optional password, player name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostInfo {
    pub port: u16,
    pub password: String,
    pub name: String,
}

/// Join form result: address, port, password, player name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JoinInfo {
    pub host: String,
    pub port: u16,
    pub password: String,
    pub name: String,
}

/// What only the core can do, asked by a screen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AppRequest {
    /// Make and enter a new world.
    NewWorld,
    /// Enter a saved world.
    Load(SlotId),
    /// Move a saved world to the trash; the screen stays and sees the shorter list.
    Delete(SlotId),
    /// Host the newest save (or a new world) and join it.
    Host(HostInfo),
    /// Join a server.
    Join(JoinInfo),
    /// Stop connecting or loading (Esc does the same).
    Cancel,
    /// Close the pause screen and play on.
    Resume,
    /// Save and leave the world for the root screen (or quit, without one).
    LeaveWorld,
    /// Save and quit the game.
    Quit,
}

/// What the core is doing besides showing screens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Nothing: the screen is the page.
    Idle,
    /// Connecting to a server. Esc cancels.
    Connecting,
    /// Building or loading a world. Esc cancels.
    Loading,
}

/// Where an entry is offered: on the screen out of a world, on the pause screen, or both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Places(u8);

impl Places {
    /// The screen out of a world (a start screen's main menu).
    pub const MAIN: Self = Self(1);
    /// The pause screen in a world.
    pub const PAUSE: Self = Self(2);
    /// Both.
    pub const BOTH: Self = Self(3);

    /// Whether `self` includes every place in `other`.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Both sets of places.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl std::ops::BitOr for Places {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        self.union(other)
    }
}

/// A screen a package offers: what a root or pause screen lists and opens, without knowing the
/// package. Entries are listed by `order` (ties in registration order).
#[derive(Clone, Copy)]
pub struct ScreenEntry {
    /// Stable id ([`ScreenOutcome::Open`] names it).
    pub id: &'static str,
    /// The row label ("Settings").
    pub label: &'static str,
    pub places: Places,
    /// Lower first.
    pub order: i16,
    /// Build the screen to push.
    pub open: fn(&ScreenFacts) -> Box<dyn Screen>,
}

impl std::fmt::Debug for ScreenEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScreenEntry")
            .field("id", &self.id)
            .field("label", &self.label)
            .field("places", &self.places)
            .field("order", &self.order)
            .finish()
    }
}

/// What a screen may read. Plain data, borrowed from the core for one frame.
#[derive(Clone, Copy)]
pub struct ScreenFacts<'a> {
    /// The saved worlds, most recently played first.
    pub saves: &'a [Slot],
    /// The last connection details used (address, port, name).
    pub session: &'a Session,
    pub version: &'a str,
    /// Whether this process runs an integrated server now.
    pub hosting: bool,
    /// A line the core has for the player (a failed join, a stopped host). Set only in the facts a
    /// root screen is created with; the screen keeps it as long as it likes.
    pub notice: Option<&'a str>,
    pub phase: Phase,
    /// True for the pause screen and what it opens.
    pub in_world: bool,
    /// Every package compiled into this build.
    pub build: &'a BuildInfo,
    /// Package ids the core suspended for this session (a server refused them).
    pub suspended: &'a [String],
    /// The entries of the active packages, by order.
    pub entries: &'a [ScreenEntry],
    /// The visual groups the installed, unsuspended packages provide.
    pub visuals: VisualMask,
}

impl<'a> ScreenFacts<'a> {
    /// The entries offered in `place`, in order.
    pub fn entries_for(&self, place: Places) -> impl Iterator<Item = &'a ScreenEntry> + 'a {
        self.entries.iter().filter(move |e| e.places.contains(place))
    }

    /// The entry with this id, if an active package registered it.
    pub fn entry(&self, id: &str) -> Option<&'a ScreenEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Whether the core suspended the package `id` for this session.
    pub fn is_suspended(&self, id: &str) -> bool {
        self.suspended.iter().any(|s| s == id)
    }
}

/// What a screen gets each frame: the facts, read access to every tunable, and the one way to
/// change them ([`options_mut`](Self::options_mut)), so the core knows when to save.
pub struct ScreenContext<'a> {
    pub facts: ScreenFacts<'a>,
    settings: &'a mut Settings,
    options: &'a mut Options,
}

impl<'a> ScreenContext<'a> {
    pub fn new(facts: ScreenFacts<'a>, settings: &'a mut Settings, options: &'a mut Options) -> Self {
        Self { facts, settings, options }
    }

    /// The core's settings, read-only (`menu_scale`, `vram_notice`, …).
    pub fn settings(&self) -> &Settings {
        self.settings
    }

    /// Every tunable, the core's settings and the packages' options, read-only.
    pub fn options(&self) -> OptionsRef<'_> {
        OptionsRef::new(self.settings, self.options)
    }

    /// Every tunable, to change. Changes move [`Options::revision`], and the core applies and
    /// saves them.
    pub fn options_mut(&mut self) -> OptionsView<'_> {
        OptionsView::new(self.settings, self.options)
    }
}

/// This frame's menu input, as the core's router saw it: which menu events fired (bindings and
/// key repeat already applied), the typed characters, and the one fixed edit key.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MenuInput {
    events: [bool; MenuEvent::COUNT],
    chars: Vec<char>,
    edit: Option<EditKey>,
}

impl MenuInput {
    /// No input.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether `event` fired this frame.
    pub fn event(&self, event: MenuEvent) -> bool {
        self.events[event as usize]
    }

    /// Characters typed this frame (layout- and shift-aware).
    pub fn chars(&self) -> &[char] {
        &self.chars
    }

    /// The fixed edit key pressed this frame (Home, End, Ctrl+Backspace), if any.
    pub fn edit(&self) -> Option<EditKey> {
        self.edit
    }

    /// True when nothing happened.
    pub fn is_quiet(&self) -> bool {
        !self.events.iter().any(|&e| e) && self.chars.is_empty() && self.edit.is_none()
    }

    /// `self` with `event` fired (tests and composed screens).
    pub fn with(mut self, event: MenuEvent) -> Self {
        self.events[event as usize] = true;
        self
    }

    /// `self` with `chars` typed.
    pub fn with_chars(mut self, chars: &[char]) -> Self {
        self.chars.extend_from_slice(chars);
        self
    }

    /// `self` with `edit` pressed.
    pub fn with_edit(mut self, edit: EditKey) -> Self {
        self.edit = Some(edit);
        self
    }

    /// Forget this frame's input, keeping the character buffer's capacity.
    pub fn clear(&mut self) {
        self.events = [false; MenuEvent::COUNT];
        self.chars.clear();
        self.edit = None;
    }

    /// Refill from the router's menu view. Drains the engine's character queue; after the first
    /// frame that typed, a refill allocates nothing.
    pub(crate) fn read(&mut self, menu: &router::Menu) {
        self.clear();
        for event in MenuEvent::ALL {
            self.events[event as usize] = menu.event(event);
        }
        self.chars.extend(menu.chars());
        self.edit = menu.edit();
    }
}

/// A screen's answer to one frame.
pub enum ScreenOutcome {
    Stay,
    /// Close this screen. On the pause screen's root this resumes; on the root screen it does
    /// nothing.
    Back,
    /// Open this screen on top.
    Push(Box<dyn Screen>),
    /// Open the registered entry with this id on top (nothing happens if no active package
    /// registered it).
    Open(&'static str),
    /// Ask the core.
    Request(AppRequest),
}

/// One menu page or form, as a mod writes it.
pub trait Screen {
    /// One frame of input. `ctx` reads the facts and changes tunables.
    fn update(&mut self, input: &MenuInput, ctx: &mut ScreenContext) -> ScreenOutcome;

    /// Push this frame's picture into `out` (cleared by the core, capacity kept) for a screen of
    /// `size` pixels.
    fn draw(&self, ctx: &ScreenContext, out: &mut Vec<UiElement>, size: (i32, i32));
}

/// What a screen draws: a closed set of primitives the core renders. The font is monospace with
/// an advance of exactly `size` pixels per glyph ([`text_width`]).
#[derive(Clone, Debug, PartialEq)]
pub enum UiElement {
    /// A filled rectangle.
    Rect { x: i32, y: i32, w: i32, h: i32, color: Color },
    /// A line of text with its top-left corner at `(x, y)`; `shadow` adds the 1px dark drop
    /// shadow every UI string has.
    Text { x: i32, y: i32, size: i32, color: Color, text: Cow<'static, str>, shadow: bool },
}

/// Width in pixels of `text` at `size`: the widest line's character count times `size`.
pub fn text_width(text: &str, size: i32) -> i32 {
    text.split('\n').map(|line| line.chars().count()).max().unwrap_or(0) as i32 * size
}

/// Draw `elements` in order. The only place a screen's picture reaches the frame.
pub fn render(f: &mut Frame, elements: &[UiElement]) {
    for el in elements {
        match el {
            UiElement::Rect { x, y, w, h, color } => f.draw_rect(*x, *y, *w, *h, *color),
            UiElement::Text { x, y, size, color, text, shadow: true } => {
                crate::ui::shadowed(f, text, *x, *y, *size, *color)
            }
            UiElement::Text { x, y, size, color, text, shadow: false } => f.draw_text(text, *x, *y, *size, *color),
        }
    }
}

/// What a stack update asks of the app.
#[derive(Debug, PartialEq, Eq)]
pub enum StackEvent {
    /// Nothing for the app.
    None,
    /// The top screen asked the core.
    Request(AppRequest),
    /// The root screen went back: the pause stack resumes, the main stack ignores it.
    BackAtRoot,
}

/// The pushdown stack of screens the core hosts. Never empty.
pub struct ScreenStack {
    frames: Vec<Box<dyn Screen>>,
}

impl ScreenStack {
    pub fn new(root: Box<dyn Screen>) -> Self {
        Self { frames: vec![root] }
    }

    /// How many screens are open.
    pub fn depth(&self) -> usize {
        self.frames.len()
    }

    pub fn push(&mut self, screen: Box<dyn Screen>) {
        self.frames.push(screen);
    }

    /// One frame for the top screen.
    pub fn update(&mut self, input: &MenuInput, ctx: &mut ScreenContext) -> StackEvent {
        let top = self.frames.last_mut().expect("a stack is never empty");
        match top.update(input, ctx) {
            ScreenOutcome::Stay => StackEvent::None,
            ScreenOutcome::Back if self.frames.len() > 1 => {
                self.frames.pop();
                StackEvent::None
            }
            ScreenOutcome::Back => StackEvent::BackAtRoot,
            ScreenOutcome::Push(screen) => {
                self.frames.push(screen);
                StackEvent::None
            }
            ScreenOutcome::Open(id) => {
                if let Some(entry) = ctx.facts.entry(id) {
                    let screen = (entry.open)(&ctx.facts);
                    self.frames.push(screen);
                }
                StackEvent::None
            }
            ScreenOutcome::Request(request) => StackEvent::Request(request),
        }
    }

    /// Draw the top screen.
    pub fn draw(&self, ctx: &ScreenContext, out: &mut Vec<UiElement>, size: (i32, i32)) {
        self.frames.last().expect("a stack is never empty").draw(ctx, out, size);
    }

    /// Draw the root screen (the waiting page while the core connects or loads).
    pub fn draw_root(&self, ctx: &ScreenContext, out: &mut Vec<UiElement>, size: (i32, i32)) {
        self.frames[0].draw(ctx, out, size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A screen that answers each frame with what the test queued, and draws a fixed picture.
    struct Scripted {
        name: &'static str,
        next: Vec<ScreenOutcome>,
    }

    impl Scripted {
        fn boxed(name: &'static str, next: Vec<ScreenOutcome>) -> Box<dyn Screen> {
            Box::new(Self { name, next })
        }
    }

    impl Screen for Scripted {
        fn update(&mut self, _input: &MenuInput, _ctx: &mut ScreenContext) -> ScreenOutcome {
            if self.next.is_empty() { ScreenOutcome::Stay } else { self.next.remove(0) }
        }
        fn draw(&self, _ctx: &ScreenContext, out: &mut Vec<UiElement>, size: (i32, i32)) {
            out.push(UiElement::Rect { x: 0, y: 0, w: size.0, h: size.1, color: Color::BLACK });
            out.push(UiElement::Text { x: 1, y: 2, size: 20, color: Color::GOLD, text: Cow::Borrowed(self.name), shadow: true });
        }
    }

    fn open_settings(_facts: &ScreenFacts) -> Box<dyn Screen> {
        Scripted::boxed("settings", Vec::new())
    }

    fn open_mods(_facts: &ScreenFacts) -> Box<dyn Screen> {
        Scripted::boxed("mods", Vec::new())
    }

    const ENTRIES: &[ScreenEntry] = &[
        ScreenEntry { id: "test.mods", label: "Mods", places: Places::BOTH, order: 10, open: open_mods },
        ScreenEntry { id: "test.settings", label: "Settings", places: Places::MAIN, order: 20, open: open_settings },
    ];

    struct Parts {
        settings: Settings,
        options: Options,
        session: Session,
        build: BuildInfo,
    }

    impl Parts {
        fn new() -> Self {
            Self { settings: Settings::default(), options: Options::new(), session: Session::default(), build: BuildInfo::EMPTY }
        }

        fn ctx(&mut self) -> ScreenContext<'_> {
            let facts = ScreenFacts {
                saves: &[],
                session: &self.session,
                version: VERSION,
                hosting: false,
                notice: None,
                phase: Phase::Idle,
                in_world: false,
                build: &self.build,
                suspended: &[],
                entries: ENTRIES,
                visuals: VisualMask::ALL,
            };
            ScreenContext::new(facts, &mut self.settings, &mut self.options)
        }
    }

    fn top_name(stack: &ScreenStack, ctx: &ScreenContext) -> String {
        let mut out = Vec::new();
        stack.draw(ctx, &mut out, (100, 100));
        match &out[1] {
            UiElement::Text { text, .. } => text.to_string(),
            _ => panic!("the name"),
        }
    }

    #[test]
    fn the_stack_pushes_pops_opens_entries_and_passes_requests_up() {
        let mut parts = Parts::new();
        let mut ctx = parts.ctx();
        let quiet = MenuInput::new();
        let root = Scripted::boxed(
            "root",
            vec![
                ScreenOutcome::Open("test.settings"),
                ScreenOutcome::Open("test.nothing"),
                ScreenOutcome::Push(Scripted::boxed("pushed", vec![ScreenOutcome::Back])),
                ScreenOutcome::Request(AppRequest::NewWorld),
                ScreenOutcome::Back,
            ],
        );
        let mut stack = ScreenStack::new(root);
        assert_eq!(stack.update(&quiet, &mut ctx), StackEvent::None);
        assert_eq!((stack.depth(), top_name(&stack, &ctx)), (2, "settings".to_string()), "an entry opened by id");
        // The opened entry stays; pop it with a scripted Back on a fresh stack instead.
        let mut stack = ScreenStack::new(Scripted::boxed(
            "root",
            vec![
                ScreenOutcome::Open("test.nothing"),
                ScreenOutcome::Push(Scripted::boxed("pushed", vec![ScreenOutcome::Back])),
                ScreenOutcome::Request(AppRequest::NewWorld),
                ScreenOutcome::Back,
            ],
        ));
        assert_eq!(stack.update(&quiet, &mut ctx), StackEvent::None);
        assert_eq!(stack.depth(), 1, "an unknown entry opens nothing");
        stack.update(&quiet, &mut ctx);
        assert_eq!(top_name(&stack, &ctx), "pushed");
        stack.update(&quiet, &mut ctx);
        assert_eq!((stack.depth(), top_name(&stack, &ctx)), (1, "root".to_string()), "Back pops");
        assert_eq!(stack.update(&quiet, &mut ctx), StackEvent::Request(AppRequest::NewWorld));
        assert_eq!(stack.update(&quiet, &mut ctx), StackEvent::BackAtRoot, "the root never pops");
        assert_eq!(stack.depth(), 1);
    }

    #[test]
    fn entries_are_offered_by_place_in_order() {
        let mut parts = Parts::new();
        let ctx = parts.ctx();
        let main: Vec<&str> = ctx.facts.entries_for(Places::MAIN).map(|e| e.label).collect();
        let pause: Vec<&str> = ctx.facts.entries_for(Places::PAUSE).map(|e| e.label).collect();
        assert_eq!(main, ["Mods", "Settings"]);
        assert_eq!(pause, ["Mods"]);
        assert!(Places::BOTH.contains(Places::PAUSE) && !Places::MAIN.contains(Places::PAUSE));
        assert_eq!(Places::MAIN | Places::PAUSE, Places::BOTH);
        assert_eq!(ctx.facts.entry("test.settings").map(|e| e.order), Some(20));
    }

    /// A quiet frame through the host (input refill, update, draw into the kept buffer) allocates
    /// nothing once warm.
    #[test]
    fn a_quiet_menu_frame_allocates_nothing() {
        let mut parts = Parts::new();
        let mut stack = ScreenStack::new(Scripted::boxed("root", Vec::new()));
        let mut input = MenuInput::new().with_chars(&['x']);
        let mut out = Vec::new();
        for frame in 0..4 {
            crate::alloc_count::reset();
            input.clear();
            let mut ctx = parts.ctx();
            assert_eq!(stack.update(&input, &mut ctx), StackEvent::None);
            out.clear();
            stack.draw(&ctx, &mut out, (1280, 720));
            if frame > 0 {
                assert_eq!(crate::alloc_count::alloc_count(), 0, "quiet frame {frame}");
            }
        }
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn menu_input_reads_like_the_router_and_the_width_is_monospace() {
        let input = MenuInput::new().with(MenuEvent::Confirm).with_chars(&['a', 'é']).with_edit(EditKey::Home);
        assert!(input.event(MenuEvent::Confirm) && !input.event(MenuEvent::Back));
        assert_eq!((input.chars(), input.edit()), (&['a', 'é'][..], Some(EditKey::Home)));
        assert!(!input.is_quiet() && MenuInput::new().is_quiet());
        assert_eq!(text_width("Loading…", 28), 8 * 28);
        assert_eq!(text_width("ab\nlonger", 10), 60);
    }

    #[test]
    fn writes_through_the_context_move_the_revision_the_host_saves_on() {
        let mut parts = Parts::new();
        let mut ctx = parts.ctx();
        let before = ctx.options().revision();
        let fov = ctx.options().find("fov").expect("a core setting");
        ctx.options_mut().step(fov, 1);
        assert_ne!(ctx.options().revision(), before);
        assert_eq!(ctx.settings().fov, 95.0);
    }
}
