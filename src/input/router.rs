//! The router: the once-per-frame transition that turns raw device state into an
//! immutable [`FrameInput`] observation. It advances repeat timers (the only
//! mutable state owned by interpretation), snapshots the fired-event sets for the
//! active context plus the global set, and hands back a view whose queries are
//! all `&self` reads. Context changes via [`Router::set_context`] only after the
//! view has dropped to satisfy borrow checker constraints.
use voxel_engine::{Engine, Vec2};

use crate::input::bindings::Bindings;
use crate::input::intent::{
    Chord, EditKey, GameplayAxis, GameplayEvent, GameplayState, GlobalEvent, MenuEvent, Repeat,
};
use crate::input::intent::Source;
use voxel_engine::Key;

/// The single exclusive interaction context for a frame. Exactly one is active;
/// the global set is always available alongside it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Context {
    Gameplay,
    Menu,
    Text,
}

/// Repeat timers; negative means held before context opened (prevents spurious autofire).
struct Timers {
    gameplay: [f32; GameplayEvent::COUNT],
    menu: [f32; MenuEvent::COUNT],
}

impl Timers {
    fn new() -> Self {
        Self {
            gameplay: [-1.0; GameplayEvent::COUNT],
            menu: [-1.0; MenuEvent::COUNT],
        }
    }

    fn reset(&mut self) {
        self.gameplay = [-1.0; GameplayEvent::COUNT];
        self.menu = [-1.0; MenuEvent::COUNT];
    }
}

/// Advance a repeat timer and report whether the event fires this frame.
fn eval_event(chords: &[Chord], repeat: Option<Repeat>, timer: &mut f32, eng: &Engine, dt: f32) -> bool {
    let edged = chords.iter().any(|c| c.edged(eng));
    let Some(rep) = repeat else {
        return edged;
    };
    let held = chords.iter().any(|c| c.held(eng));
    if !held {
        *timer = -1.0; // unprime
        return edged;
    }
    if edged {
        *timer = rep.delay;
        return true;
    }
    if *timer < 0.0 {
        // Held without ever seeing an edge (key was down when the context
        // opened): don't autofire until a fresh press primes it.
        return false;
    }
    *timer -= dt;
    if *timer <= 0.0 {
        *timer += rep.interval;
        if *timer <= 0.0 {
            *timer = rep.interval; // clamp to one fire per frame on a long dt
        }
        return true;
    }
    false
}

/// Input interpretation: maps device events to high-level intents and manages
/// repeat timers, context, and mouse capture state.
pub struct Router {
    pub bindings: Bindings,
    context: Context,
    captured: bool,
    timers: Timers,
}

impl Router {
    pub fn new() -> Self {
        Self {
            bindings: Bindings::default(),
            context: Context::Gameplay,
            captured: true,
            timers: Timers::new(),
        }
    }

    pub fn context(&self) -> Context {
        self.context
    }

    /// Switch context, resetting stale repeat timers so a key held across the
    /// switch doesn't carry its autofire into the new context.
    pub fn set_context(&mut self, c: Context) {
        if self.context != c {
            self.context = c;
            self.timers.reset();
        }
    }

    /// Whether the mouse is captured for aiming (gameplay's sub-flag). Break,
    /// place, and look read as inert while uncaptured.
    pub fn captured(&self) -> bool {
        self.captured
    }

    pub fn set_captured(&mut self, captured: bool) {
        self.captured = captured;
    }

    /// Per-frame input observation: advances repeat timers and returns an
    /// immutable view of the current input state.
    pub fn frame<'e>(&'e mut self, engine: &'e Engine, dt: f32) -> FrameInput<'e> {
        self.frame_filtered(engine, dt, true, true, true)
    }

    /// Gameplay variant that can structurally skip mod placement, mod UI, and
    /// minimap physical probes independently. Menus use [`frame`](Self::frame).
    pub fn frame_filtered<'e>(
        &'e mut self,
        engine: &'e Engine,
        dt: f32,
        mod_logic: bool,
        mod_ui: bool,
        minimap: bool,
    ) -> FrameInput<'e> {
        let mut global_fired = [false; GlobalEvent::COUNT];
        for e in GlobalEvent::ALL {
            if !minimap && e == GlobalEvent::MinimapMode {
                continue;
            }
            global_fired[e as usize] =
                self.bindings.global_event[e as usize].iter().any(|c| c.edged(engine));
        }

        let mut gameplay_fired = [false; GameplayEvent::COUNT];
        let mut menu_fired = [false; MenuEvent::COUNT];
        match self.context {
            Context::Gameplay => {
                for e in GameplayEvent::ALL {
                    let disabled = match e {
                        GameplayEvent::Place => !mod_logic,
                        GameplayEvent::ToggleInventory | GameplayEvent::ToggleCrafting => !mod_ui,
                        _ => false,
                    };
                    if disabled {
                        self.timers.gameplay[e as usize] = -1.0;
                        continue;
                    }
                    gameplay_fired[e as usize] = eval_event(
                        &self.bindings.gameplay_event[e as usize],
                        e.repeat(),
                        &mut self.timers.gameplay[e as usize],
                        engine,
                        dt,
                    );
                }
            }
            Context::Menu => {
                for e in MenuEvent::ALL {
                    menu_fired[e as usize] = eval_event(
                        &self.bindings.menu_event[e as usize],
                        e.repeat(),
                        &mut self.timers.menu[e as usize],
                        engine,
                        dt,
                    );
                }
            }
            Context::Text => {}
        }

        FrameInput {
            eng: engine,
            bindings: &self.bindings,
            context: self.context,
            captured: self.captured,
            global_fired,
            gameplay_fired,
            menu_fired,
        }
    }
}

impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}

/// Immutable per-frame input observation.
pub struct FrameInput<'e> {
    eng: &'e Engine,
    bindings: &'e Bindings,
    context: Context,
    captured: bool,
    global_fired: [bool; GlobalEvent::COUNT],
    gameplay_fired: [bool; GameplayEvent::COUNT],
    menu_fired: [bool; MenuEvent::COUNT],
}

impl<'e> FrameInput<'e> {
    /// The always-available global event set.
    pub fn global(&self) -> Global<'_> {
        Global { fi: self }
    }

    /// The active exclusive context's view.
    pub fn view(&self) -> View<'_> {
        match self.context {
            Context::Gameplay => View::Gameplay(Gameplay { fi: self }),
            Context::Menu => View::Menu(Menu { fi: self }),
            Context::Text => View::Text(Text { fi: self }),
        }
    }
}

/// The active context's typed query surface.
pub enum View<'a> {
    Gameplay(Gameplay<'a>),
    Menu(Menu<'a>),
    Text(Text<'a>),
}

/// Gameplay queries; break/place/look inert when uncaptured (no gate needed in call sites).
pub struct Gameplay<'a> {
    fi: &'a FrameInput<'a>,
}

impl Gameplay<'_> {
    pub fn state(&self, s: GameplayState) -> bool {
        self.fi.bindings.gameplay_state[s as usize]
            .iter()
            .any(|src| src.is_down(self.fi.eng))
    }

    pub fn event(&self, e: GameplayEvent) -> bool {
        let fired = self.fi.gameplay_fired[e as usize];
        match e {
            GameplayEvent::Break | GameplayEvent::Place if !self.fi.captured => false,
            _ => fired,
        }
    }

    pub fn axis(&self, a: GameplayAxis) -> f32 {
        self.fi.bindings.gameplay_axis[a as usize].sample(self.fi.eng)
    }

    /// Yaw (x) and pitch (y) deltas from this frame's mouse motion, with
    /// sensitivity and invert applied. Zero while uncaptured.
    pub fn look(&self) -> Vec2 {
        if !self.fi.captured {
            return Vec2::ZERO;
        }
        let delta = self.fi.eng.mouse_delta();
        Vec2::new(
            self.fi.bindings.look_x.mouse_component(delta),
            self.fi.bindings.look_y.mouse_component(delta),
        )
    }

    pub fn captured(&self) -> bool {
        self.fi.captured
    }

    /// Menu-style navigation for in-world overlays: reuses menu bindings
    /// without leaving gameplay context, edge-only (no repeat timers).
    pub fn overlay_nav(&self, e: MenuEvent) -> bool {
        self.fi.bindings.menu_event[e as usize].iter().any(|c| c.edged(self.fi.eng))
    }
}

/// Menu queries: navigation events (auto-repeat baked in) and the char stream.
pub struct Menu<'a> {
    fi: &'a FrameInput<'a>,
}

impl Menu<'_> {
    pub fn event(&self, e: MenuEvent) -> bool {
        self.fi.menu_fired[e as usize]
    }

    /// This frame's typed characters (layout- and shift-aware), draining the
    /// engine's char queue.
    pub fn chars(&self) -> impl Iterator<Item = char> + '_ {
        std::iter::from_fn(move || self.fi.eng.get_char_pressed())
    }

    /// Fixed edit keys beyond bindable events (Home/End/Ctrl+Backspace).
    pub fn edit(&self) -> Option<EditKey> {
        let eng = self.fi.eng;
        let ctrl = eng.is_key_down(Key::LeftControl) || eng.is_key_down(Key::RightControl);
        if ctrl && eng.is_key_pressed(Key::Backspace) {
            return Some(EditKey::DelWord);
        }
        if eng.is_key_pressed(Key::Home) {
            return Some(EditKey::Home);
        }
        if eng.is_key_pressed(Key::End) {
            return Some(EditKey::End);
        }
        None
    }
}

/// Text-field queries: the char stream and the fixed, non-bindable edit keys.
pub struct Text<'a> {
    fi: &'a FrameInput<'a>,
}

impl Text<'_> {
    pub fn chars(&self) -> impl Iterator<Item = char> + '_ {
        std::iter::from_fn(move || self.fi.eng.get_char_pressed())
    }

    /// Fixed edit keys pressed this frame, checked in priority order.
    pub fn edit(&self) -> Option<EditKey> {
        let eng = self.fi.eng;
        let ctrl = Source::Key(Key::LeftControl).is_down(eng)
            || Source::Key(Key::RightControl).is_down(eng);
        let pressed = |k: Key| eng.is_key_pressed(k);
        if ctrl && pressed(Key::U) {
            return Some(EditKey::ClearLine);
        }
        if ctrl && pressed(Key::Backspace) {
            return Some(EditKey::DelWord);
        }
        let table = [
            (Key::Left, EditKey::Left),
            (Key::Right, EditKey::Right),
            (Key::Home, EditKey::Home),
            (Key::End, EditKey::End),
            (Key::Backspace, EditKey::Backspace),
            (Key::Delete, EditKey::Delete),
            (Key::Up, EditKey::HistoryUp),
            (Key::Down, EditKey::HistoryDown),
            (Key::Tab, EditKey::Complete),
            (Key::Enter, EditKey::Submit),
        ];
        table.into_iter().find(|(k, _)| pressed(*k)).map(|(_, e)| e)
    }
}

/// The always-available global event set.
pub struct Global<'a> {
    fi: &'a FrameInput<'a>,
}

impl Global<'_> {
    pub fn event(&self, e: GlobalEvent) -> bool {
        self.fi.global_fired[e as usize]
    }
}
