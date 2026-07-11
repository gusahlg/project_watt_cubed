//! The device-side vocabulary the input layer binds against: a [`Source`] (one
//! physical control), a [`Mods`] set, and a [`Chord`] (mods + source). On top of
//! those sit the semantic intent enums — what the game reacts to — split into
//! states (held), events (edges), and axes (continuous). Bindings map
//! intents to sources/chords; nothing here reads the engine except the small
//! `is_down`/`is_pressed` probes on [`Source`].
use std::fmt;

use voxel_engine::{Engine, Key, MouseButton, Vec2};

/// One physical control.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Source {
    Key(Key),
    Mouse(MouseButton),
    WheelUp,
    WheelDown,
}

impl Source {
    /// Whether the control is held this frame.
    pub fn is_down(self, eng: &Engine) -> bool {
        match self {
            Source::Key(k) => eng.is_key_down(k),
            Source::Mouse(b) => eng.is_mouse_button_down(b),
            Source::WheelUp => eng.mouse_wheel() > 0.0,
            Source::WheelDown => eng.mouse_wheel() < 0.0,
        }
    }

    /// Whether the control edged to pressed this frame.
    pub fn is_pressed(self, eng: &Engine) -> bool {
        match self {
            Source::Key(k) => eng.is_key_pressed(k),
            Source::Mouse(b) => eng.is_mouse_button_pressed(b),
            Source::WheelUp => eng.mouse_wheel() > 0.0,
            Source::WheelDown => eng.mouse_wheel() < 0.0,
        }
    }
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Source::Key(k) => write!(f, "{k:?}"),
            Source::Mouse(MouseButton::Left) => write!(f, "Mouse1"),
            Source::Mouse(MouseButton::Right) => write!(f, "Mouse2"),
            Source::Mouse(MouseButton::Middle) => write!(f, "Mouse3"),
            Source::WheelUp => write!(f, "WheelUp"),
            Source::WheelDown => write!(f, "WheelDown"),
        }
    }
}

/// Hand-rolled modifier key set, avoiding bitflags dependency.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct Mods(u8);

impl Mods {
    pub const NONE: Mods = Mods(0);
    pub const CTRL: Mods = Mods(1 << 0);
    pub const SHIFT: Mods = Mods(1 << 1);
    pub const ALT: Mods = Mods(1 << 2);

    /// The modifiers currently held, layout-independent.
    pub fn current(eng: &Engine) -> Mods {
        let mut m = Mods::NONE;
        if eng.is_key_down(Key::LeftControl) || eng.is_key_down(Key::RightControl) {
            m.0 |= Mods::CTRL.0;
        }
        if eng.is_key_down(Key::LeftShift) || eng.is_key_down(Key::RightShift) {
            m.0 |= Mods::SHIFT.0;
        }
        if eng.is_key_down(Key::LeftAlt) {
            m.0 |= Mods::ALT.0;
        }
        m
    }

    pub fn contains(self, other: Mods) -> bool {
        self.0 & other.0 == other.0
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl std::ops::BitOr for Mods {
    type Output = Mods;
    fn bitor(self, rhs: Mods) -> Mods {
        Mods(self.0 | rhs.0)
    }
}

impl fmt::Display for Mods {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for (flag, name) in [(Mods::CTRL, "Ctrl"), (Mods::SHIFT, "Shift"), (Mods::ALT, "Alt")] {
            if self.contains(flag) {
                if !first {
                    write!(f, "+")?;
                }
                write!(f, "{name}")?;
                first = false;
            }
        }
        Ok(())
    }
}

/// Modifier-exact chord: fires only on exact mod match, so Ctrl+U ≠ U.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Chord {
    pub mods: Mods,
    pub source: Source,
}

impl Chord {
    /// A bare (no-modifier) chord over a source.
    pub const fn bare(source: Source) -> Chord {
        Chord { mods: Mods::NONE, source }
    }

    /// A bare key chord.
    pub const fn key(key: Key) -> Chord {
        Chord::bare(Source::Key(key))
    }

    /// A bare mouse-button chord.
    pub const fn mouse(button: MouseButton) -> Chord {
        Chord::bare(Source::Mouse(button))
    }

    /// A chord with an exact modifier set.
    pub const fn with(mods: Mods, source: Source) -> Chord {
        Chord { mods, source }
    }

    /// Chord fired this frame (no subset firing).
    pub fn edged(self, eng: &Engine) -> bool {
        self.source.is_pressed(eng) && Mods::current(eng) == self.mods
    }

    /// Hold condition for auto-repeat (source down, mod match).
    pub fn held(self, eng: &Engine) -> bool {
        self.source.is_down(eng) && Mods::current(eng) == self.mods
    }
}

impl fmt::Display for Chord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.mods.is_empty() {
            write!(f, "{}", self.source)
        } else {
            write!(f, "{}+{}", self.mods, self.source)
        }
    }
}

/// Auto-repeat timing: semantic intent property, not bindable.
#[derive(Clone, Copy)]
pub struct Repeat {
    pub delay: f32,
    pub interval: f32,
}

impl Repeat {
    pub const fn new(delay: f32, interval: f32) -> Repeat {
        Repeat { delay, interval }
    }
}

/// Continuous axis binding.
#[derive(Clone, Copy)]
pub enum AxisSource {
    KeyPair { neg: Source, pos: Source },
    MouseX { sens: f32, invert: bool },
    MouseY { sens: f32, invert: bool },
}

impl AxisSource {
    /// Key-pair axis only; positive wins if both held.
    pub fn sample(self, eng: &Engine) -> f32 {
        match self {
            AxisSource::KeyPair { neg, pos } => {
                if pos.is_down(eng) {
                    1.0
                } else if neg.is_down(eng) {
                    -1.0
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }

    /// Mouse axis only.
    pub fn mouse_component(self, delta: Vec2) -> f32 {
        match self {
            AxisSource::MouseX { sens, invert } => delta.x * sens * if invert { -1.0 } else { 1.0 },
            AxisSource::MouseY { sens, invert } => delta.y * sens * if invert { -1.0 } else { 1.0 },
            AxisSource::KeyPair { .. } => 0.0,
        }
    }
}

// Intent enums: COUNT indexes binding arrays, ALL iterates the total set.

/// A held gameplay state (bound to bare sources).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GameplayState {
    Sprint,
    Sneak,
    Jump,
}

impl GameplayState {
    pub const COUNT: usize = 3;
    pub const ALL: [GameplayState; Self::COUNT] =
        [GameplayState::Sprint, GameplayState::Sneak, GameplayState::Jump];
}

/// A gameplay edge event (bound to modifier-exact chords).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GameplayEvent {
    ToggleFly,
    Break,
    Place,
    OpenConsole,
    OpenChat,
    ToggleInventory,
    ToggleCrafting,
    ToggleCapture,
}

impl GameplayEvent {
    pub const COUNT: usize = 8;
    pub const ALL: [GameplayEvent; Self::COUNT] = [
        GameplayEvent::ToggleFly,
        GameplayEvent::Break,
        GameplayEvent::Place,
        GameplayEvent::OpenConsole,
        GameplayEvent::OpenChat,
        GameplayEvent::ToggleInventory,
        GameplayEvent::ToggleCrafting,
        GameplayEvent::ToggleCapture,
    ];

    /// Break and Place autofire; others don't.
    pub const fn repeat(self) -> Option<Repeat> {
        match self {
            GameplayEvent::Break | GameplayEvent::Place => Some(Repeat::new(0.25, 0.20)),
            _ => None,
        }
    }
}

/// A continuous gameplay movement axis.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GameplayAxis {
    MoveX,
    MoveY,
    MoveZ,
}

impl GameplayAxis {
    pub const COUNT: usize = 3;
    pub const ALL: [GameplayAxis; Self::COUNT] =
        [GameplayAxis::MoveX, GameplayAxis::MoveY, GameplayAxis::MoveZ];
}

/// A menu navigation event (bound to modifier-exact chords, auto-repeat on the
/// directional/delete ones).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MenuEvent {
    Up,
    Down,
    Left,
    Right,
    Confirm,
    Back,
    NextTab,
    Toggle,
    Delete,
}

impl MenuEvent {
    pub const COUNT: usize = 9;
    pub const ALL: [MenuEvent; Self::COUNT] = [
        MenuEvent::Up,
        MenuEvent::Down,
        MenuEvent::Left,
        MenuEvent::Right,
        MenuEvent::Confirm,
        MenuEvent::Back,
        MenuEvent::NextTab,
        MenuEvent::Toggle,
        MenuEvent::Delete,
    ];

    pub const fn repeat(self) -> Option<Repeat> {
        match self {
            MenuEvent::Up | MenuEvent::Down | MenuEvent::Left | MenuEvent::Right | MenuEvent::Delete => {
                Some(Repeat::new(0.35, 0.06))
            }
            _ => None,
        }
    }
}

/// An always-available event, independent of the active context.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GlobalEvent {
    Escape,
    CycleHud,
    Screenshot,
    MinimapMode,
}

impl GlobalEvent {
    pub const COUNT: usize = 4;
    pub const ALL: [GlobalEvent; Self::COUNT] = [
        GlobalEvent::Escape,
        GlobalEvent::CycleHud,
        GlobalEvent::Screenshot,
        GlobalEvent::MinimapMode,
    ];
}

/// Fixed text-edit actions, not bindable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EditKey {
    Left,
    Right,
    Home,
    End,
    Backspace,
    DelWord,
    Delete,
    ClearLine,
    HistoryUp,
    HistoryDown,
    Complete,
    Submit,
}
