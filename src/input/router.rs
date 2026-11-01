//! The router: the once-per-frame transition that turns raw device state into an
//! immutable [`FrameInput`] observation. It advances repeat timers (the only
//! mutable state owned by interpretation), snapshots the fired-event sets for the
//! active context plus the global set, and hands back a view whose queries are
//! all `&self` reads. Context changes via [`Router::set_context`] only after the
//! view has dropped to satisfy borrow checker constraints.
use voxel_engine::{Engine, Vec2};

use crate::input::bindings::Bindings;
use crate::input::intent::{
    Chord, EditKey, GameplayAxis, GameplayEvent, GameplayState, GlobalEvent, MenuEvent, Mods,
    Repeat,
};
use crate::input::intent::Source;
use crate::modules::ActionSet;
use voxel_engine::Key;

/// Same autofire as breaking and placing, for a module action that asks to repeat.
const MODULE_REPEAT: Repeat = Repeat::new(0.25, 0.20);

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

/// One module action's chords after core bindings have taken any clash.
struct ModuleBinding {
    chords: Vec<Chord>,
    repeat: bool,
    held: bool,
    /// Sampled with the module-logic lane off too, and reported apart (see `Action::immediate`).
    immediate: bool,
}

/// Device edges the router samples. The engine is one; tests inject a bitset.
pub(crate) trait Probe {
    fn edged(&self, chord: Chord) -> bool;
    fn held(&self, chord: Chord) -> bool;
    fn wheel(&self) -> f32;
}

struct EngineProbe<'a> {
    eng: &'a Engine,
    mods: Mods,
}

impl Probe for EngineProbe<'_> {
    fn edged(&self, chord: Chord) -> bool {
        chord.edged(self.eng, self.mods)
    }

    fn held(&self, chord: Chord) -> bool {
        chord.held(self.eng, self.mods)
    }

    fn wheel(&self) -> f32 {
        self.eng.mouse_wheel()
    }
}

/// A frame of edges with no window behind it. `pressed_*` is the edge; `down_*` is the hold.
#[cfg(test)]
#[derive(Clone, Copy)]
pub(crate) struct Press {
    pub pressed_keys: u128,
    pub down_keys: u128,
    pub pressed_mouse: u8,
    pub down_mouse: u8,
    pub wheel: f32,
    pub mods: Mods,
}

#[cfg(test)]
impl Press {
    pub(crate) fn key(key: Key) -> Self {
        let bit = 1u128 << key as u32;
        Self {
            pressed_keys: bit,
            down_keys: bit,
            pressed_mouse: 0,
            down_mouse: 0,
            wheel: 0.0,
            mods: Mods::NONE,
        }
    }

    pub(crate) fn wheel(wheel: f32) -> Self {
        Self {
            pressed_keys: 0,
            down_keys: 0,
            pressed_mouse: 0,
            down_mouse: 0,
            wheel,
            mods: Mods::NONE,
        }
    }

    fn key_bit(bits: u128, key: Key) -> bool {
        bits & (1u128 << key as u32) != 0
    }

    fn source_on(self, source: Source, keys: u128, mouse: u8) -> bool {
        match source {
            Source::Key(k) => Self::key_bit(keys, k),
            Source::Mouse(b) => mouse & (1u8 << b as u8) != 0,
            Source::WheelUp => self.wheel > 0.0,
            Source::WheelDown => self.wheel < 0.0,
        }
    }
}

#[cfg(test)]
impl Probe for Press {
    fn edged(&self, chord: Chord) -> bool {
        self.source_on(chord.source, self.pressed_keys, self.pressed_mouse) && self.mods == chord.mods
    }

    fn held(&self, chord: Chord) -> bool {
        self.source_on(chord.source, self.down_keys, self.down_mouse) && self.mods == chord.mods
    }

    fn wheel(&self) -> f32 {
        self.wheel
    }
}

/// Signed scroll steps. A notch is about 1; a non-zero flick still counts as a chord edge.
fn scroll_steps(wheel: f32) -> i8 {
    if !wheel.is_finite() || wheel == 0.0 {
        0
    } else {
        wheel.round().clamp(i8::MIN as f32, i8::MAX as f32) as i8
    }
}

/// Advance a repeat timer and report whether the event fires this frame.
fn eval_event(chords: &[Chord], repeat: Option<Repeat>, timer: &mut f32, probe: &impl Probe, dt: f32) -> bool {
    let edged = chords.iter().any(|c| probe.edged(*c));
    let Some(rep) = repeat else {
        return edged;
    };
    let held = chords.iter().any(|c| probe.held(*c));
    rep.advance(timer, edged, held, dt)
}

/// True when a core gameplay or global binding already owns `chord`. Menu chords
/// are not claims: the wheel scrolls menus and may also be a module action in play.
fn core_claims(bindings: &Bindings, chord: Chord) -> bool {
    let claimed = bindings
        .gameplay_event
        .iter()
        .chain(bindings.global_event.iter())
        .any(|set| set.iter().any(|c| *c == chord));
    claimed
        || (chord.mods == Mods::NONE
            && bindings.gameplay_state.iter().any(|set| set.iter().any(|src| *src == chord.source)))
}

/// One sample of the router's fired sets. Tests build it without an engine.
pub(crate) struct Sample {
    pub global: [bool; GlobalEvent::COUNT],
    pub gameplay: [bool; GameplayEvent::COUNT],
    pub menu: [bool; MenuEvent::COUNT],
    pub actions: ActionSet,
    /// Immediate actions that fired: sampled whatever the module-logic lane says.
    pub immediate: ActionSet,
    pub wheel: i8,
}

/// Input interpretation: maps device events to high-level intents and manages
/// repeat timers, context, and mouse capture state.
pub struct Router {
    pub bindings: Bindings,
    context: Context,
    captured: bool,
    timers: Timers,
    /// Module actions, rebuilt when modules are enabled or disabled. Empty chords were
    /// dropped because a core binding owns them.
    module_bindings: Vec<ModuleBinding>,
    module_ids: Vec<&'static str>,
    module_timers: Vec<f32>,
    /// `u64::MAX` until the first sync, so the table is built once and a quiet
    /// frame does not allocate.
    module_gen: u64,
}

impl Router {
    pub fn new() -> Self {
        Self {
            bindings: Bindings::default(),
            context: Context::Gameplay,
            captured: true,
            timers: Timers::new(),
            module_bindings: Vec::new(),
            module_ids: Vec::new(),
            module_timers: Vec::new(),
            module_gen: u64::MAX,
        }
    }

    /// Switch context, resetting stale repeat timers so a key held across the
    /// switch doesn't carry its autofire into the new context.
    pub fn set_context(&mut self, c: Context) {
        if self.context != c {
            self.context = c;
            self.timers.reset();
            self.reset_module_timers();
        }
    }

    fn reset_module_timers(&mut self) {
        for t in &mut self.module_timers {
            *t = -1.0;
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

    /// Rebuild the module-action table when the enabled set changed. A matching
    /// generation returns without allocating.
    pub fn sync_actions(&mut self, host: &crate::modules::Modules) {
        let generation = host.action_generation();
        if self.module_gen == generation {
            return;
        }
        self.module_gen = generation;
        self.module_ids.clear();
        self.module_bindings.clear();
        self.module_timers.clear();
        for action in host.enabled_actions() {
            if self.module_ids.len() >= ActionSet::CAP {
                debug_assert!(false, "more than {} module actions", ActionSet::CAP);
                break;
            }
            let chords = action.default.iter().copied().filter(|c| !core_claims(&self.bindings, *c)).collect();
            self.module_ids.push(action.id);
            self.module_bindings.push(ModuleBinding {
                chords,
                repeat: action.repeat,
                held: action.held,
                immediate: action.immediate,
            });
            self.module_timers.push(-1.0);
        }
    }

    /// Ids of the actions in [`Self::sync_actions`], in bit order.
    pub fn action_ids(&self) -> &[&'static str] {
        &self.module_ids
    }

    /// Per-frame input observation: advances repeat timers and returns an
    /// immutable view of the current input state.
    pub fn frame<'e>(&'e mut self, engine: &'e Engine, dt: f32) -> FrameInput<'e> {
        self.frame_filtered(engine, dt, true, true)
    }

    /// Locked-input frame: unprime repeat timers without evaluating any chord.
    /// Engine press-edges and mouse delta still expire at the end of the engine
    /// frame, so they cannot replay when input unlocks.
    pub fn drain_frame(&mut self) {
        self.timers.reset();
        self.reset_module_timers();
    }

    /// Gameplay variant that can structurally skip module logic and the minimap
    /// key. A disabled lane's chords are never evaluated and its repeat timer
    /// is unprimed, so a held key cannot autofire the instant the lane
    /// re-enables. Menus use [`frame`](Self::frame).
    pub fn frame_filtered<'e>(
        &'e mut self,
        engine: &'e Engine,
        dt: f32,
        module_logic: bool,
        minimap: bool,
    ) -> FrameInput<'e> {
        let mods = Mods::current(engine);
        let sample = {
            let probe = EngineProbe { eng: engine, mods };
            self.sample(&probe, dt, module_logic, minimap)
        };
        FrameInput {
            eng: engine,
            bindings: &self.bindings,
            context: self.context,
            captured: self.captured,
            mods,
            global_fired: sample.global,
            gameplay_fired: sample.gameplay,
            menu_fired: sample.menu,
            actions: sample.actions,
            immediate: sample.immediate,
            wheel: sample.wheel,
        }
    }

    /// Evaluate the active context. `module_logic` false unprimes the timers of the
    /// cadence actions and reports none of them and no wheel; immediate actions
    /// are sampled either way.
    pub(crate) fn sample(&mut self, probe: &impl Probe, dt: f32, module_logic: bool, minimap: bool) -> Sample {
        let mut global_fired = [false; GlobalEvent::COUNT];
        for e in GlobalEvent::ALL {
            if !minimap && e == GlobalEvent::MinimapMode {
                continue;
            }
            global_fired[e as usize] =
                self.bindings.global_event[e as usize].iter().any(|c| probe.edged(*c));
        }

        let mut gameplay_fired = [false; GameplayEvent::COUNT];
        let mut menu_fired = [false; MenuEvent::COUNT];
        let mut actions = ActionSet::NONE;
        let mut immediate = ActionSet::NONE;
        let mut wheel = 0i8;
        match self.context {
            Context::Gameplay => {
                for e in GameplayEvent::ALL {
                    if e == GameplayEvent::Place && !module_logic {
                        self.timers.gameplay[e as usize] = -1.0;
                        continue;
                    }
                    gameplay_fired[e as usize] = eval_event(
                        &self.bindings.gameplay_event[e as usize],
                        e.repeat(),
                        &mut self.timers.gameplay[e as usize],
                        probe,
                        dt,
                    );
                }
                if module_logic {
                    wheel = scroll_steps(probe.wheel());
                }
                for i in 0..self.module_bindings.len() {
                    let binding = &self.module_bindings[i];
                    if !binding.immediate && !module_logic {
                        self.module_timers[i] = -1.0;
                        continue;
                    }
                    let hit = if binding.held {
                        binding.chords.iter().any(|chord| probe.held(*chord))
                    } else {
                        let repeat = binding.repeat.then_some(MODULE_REPEAT);
                        eval_event(&binding.chords, repeat, &mut self.module_timers[i], probe, dt)
                    };
                    match (hit, binding.immediate) {
                        (true, true) => immediate.insert(i),
                        (true, false) => actions.insert(i),
                        (false, _) => {}
                    }
                }
            }
            Context::Menu => {
                for e in MenuEvent::ALL {
                    menu_fired[e as usize] = eval_event(
                        &self.bindings.menu_event[e as usize],
                        e.repeat(),
                        &mut self.timers.menu[e as usize],
                        probe,
                        dt,
                    );
                }
            }
            Context::Text => {}
        }

        Sample { global: global_fired, gameplay: gameplay_fired, menu: menu_fired, actions, immediate, wheel }
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
    /// The frame's one modifier sample, shared by every lazy chord query.
    mods: Mods,
    global_fired: [bool; GlobalEvent::COUNT],
    gameplay_fired: [bool; GameplayEvent::COUNT],
    menu_fired: [bool; MenuEvent::COUNT],
    actions: ActionSet,
    immediate: ActionSet,
    wheel: i8,
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

    /// Module actions that fired this frame. Empty when module logic is off.
    pub fn actions(&self) -> ActionSet {
        self.fi.actions
    }

    /// Immediate module actions that fired this frame, module logic on or off.
    pub fn immediate_actions(&self) -> ActionSet {
        self.fi.immediate
    }

    /// Signed scroll steps this frame. Zero when module logic is off.
    pub fn wheel(&self) -> i8 {
        self.fi.wheel
    }

    pub fn axis(&self, a: GameplayAxis) -> f32 {
        self.fi.bindings.gameplay_axis[a as usize].sample(self.fi.eng)
    }

    /// Yaw (x) and pitch (y) deltas from this frame's mouse motion, with
    /// invert applied. Sensitivity is not: callers scale via
    /// `Orientation::look`. Zero while uncaptured.
    pub fn look(&self) -> Vec2 {
        if !self.fi.captured {
            return Vec2::ZERO;
        }
        let delta = self.fi.eng.mouse_delta();
        Vec2::new(
            self.fi.bindings.look_x.sample(delta),
            self.fi.bindings.look_y.sample(delta),
        )
    }

    pub fn captured(&self) -> bool {
        self.fi.captured
    }

    /// Menu-style navigation for in-world overlays: reuses menu bindings
    /// without leaving gameplay context, edge-only (no repeat timers).
    pub fn overlay_nav(&self, e: MenuEvent) -> bool {
        self.fi.bindings.menu_event[e as usize].iter().any(|c| c.edged(self.fi.eng, self.fi.mods))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modules::testing::{action, Stub};
    use crate::modules::{Action, Modules as Host};

    const SLOT3: &[Action] = &[action("bar.slot3", &[Chord::key(Key::Num3)])];
    /// Tab is the core capture key.
    const CLASH: &[Action] = &[action("clash.fire", &[Chord::key(Key::Tab), Chord::key(Key::Num3)])];
    /// The chat key: the physical key left of 1 (§ on a Swedish layout).
    const CHAT: &[Action] = &[Action { immediate: true, ..action("chat.open", &[Chord::key(Key::Backquote)]) }];
    const WHEEL: &[Action] = &[action("wheel.down", &[Chord::bare(Source::WheelDown)])];
    const HOLD: &[Action] = &[Action { held: true, ..action("voice.talk", &[Chord::key(Key::V)]) }];

    /// One enabled module declaring `actions`.
    fn host(actions: &'static [Action]) -> Host {
        let mut modules = Host::empty();
        modules.install(Box::new(Stub::new("stub").actions(actions)));
        modules
    }

    fn fired(router: &Router, id: &str, actions: ActionSet) -> bool {
        router.action_ids().iter().enumerate().any(|(i, name)| *name == id && actions.contains(i))
    }

    #[test]
    fn drain_frame_unprimes_repeat_timers() {
        let mut router = Router::new();
        router.timers.gameplay[0] = 0.12;
        router.timers.menu[0] = 0.08;
        router.module_timers.push(0.4);
        router.drain_frame();
        assert!(router.timers.gameplay.iter().all(|&t| t < 0.0));
        assert!(router.timers.menu.iter().all(|&t| t < 0.0));
        assert!(router.module_timers.iter().all(|&t| t < 0.0));
    }

    #[test]
    fn a_module_action_fires_on_its_default_chord() {
        let mut router = Router::new();
        let modules = host(SLOT3);
        router.sync_actions(&modules);
        let sample = router.sample(&Press::key(Key::Num3), 1.0 / 60.0, true, true);
        assert!(fired(&router, "bar.slot3", sample.actions));
        assert!(sample.immediate.is_empty());
        assert!(!sample.gameplay[GameplayEvent::ToggleCapture as usize]);
    }

    /// An immediate action fires with the module-logic lane off, on its own set, and not while
    /// text is captured; a cadence action does not fire with the lane off.
    #[test]
    fn an_immediate_action_fires_with_module_logic_off() {
        let mut router = Router::new();
        let mut modules = host(CHAT);
        modules.install(Box::new(Stub::new("bar").actions(SLOT3)));
        router.sync_actions(&modules);
        let chat = router.sample(&Press::key(Key::Backquote), 1.0 / 60.0, false, true);
        assert!(fired(&router, "chat.open", chat.immediate), "the chat key works under the Minimum preset");
        assert!(chat.actions.is_empty(), "an immediate action is not a cadence action");
        let slot = router.sample(&Press::key(Key::Num3), 1.0 / 60.0, false, true);
        assert!(slot.actions.is_empty() && slot.immediate.is_empty(), "module logic off silences cadence actions");
        let on = router.sample(&Press::key(Key::Backquote), 1.0 / 60.0, true, true);
        assert!(fired(&router, "chat.open", on.immediate));
        router.set_context(Context::Text);
        let typing = router.sample(&Press::key(Key::Backquote), 1.0 / 60.0, true, true);
        assert!(typing.immediate.is_empty(), "while typing, the key types");
    }

    #[test]
    fn a_core_binding_wins_a_clash() {
        let mut router = Router::new();
        let modules = host(CLASH);
        router.sync_actions(&modules);
        let tab = router.sample(&Press::key(Key::Tab), 1.0 / 60.0, true, true);
        assert!(tab.gameplay[GameplayEvent::ToggleCapture as usize], "Tab stays the capture key");
        assert!(!fired(&router, "clash.fire", tab.actions), "the core chord is not also a module action");
        let slot = router.sample(&Press::key(Key::Num3), 1.0 / 60.0, true, true);
        assert!(fired(&router, "clash.fire", slot.actions), "the other default chord still fires");
        assert!(!slot.gameplay[GameplayEvent::ToggleCapture as usize]);
    }

    #[test]
    fn a_menu_wheel_chord_does_not_block_a_module_action() {
        let mut router = Router::new();
        let modules = host(WHEEL);
        router.sync_actions(&modules);
        let sample = router.sample(&Press::wheel(-1.0), 1.0 / 60.0, true, true);
        assert!(fired(&router, "wheel.down", sample.actions));
        assert_eq!(sample.wheel, -1);
    }

    /// A held action stays on while the key is down, including a frame with no press edge.
    #[test]
    fn a_held_action_stays_on_while_the_chord_is_down() {
        let mut router = Router::new();
        let modules = host(HOLD);
        router.sync_actions(&modules);
        let down = Press {
            pressed_keys: 0,
            down_keys: 1u128 << Key::V as u32,
            pressed_mouse: 0,
            down_mouse: 0,
            wheel: 0.0,
            mods: crate::input::intent::Mods::NONE,
        };
        let held = router.sample(&down, 1.0 / 60.0, true, true);
        assert!(fired(&router, "voice.talk", held.actions));
        let still = router.sample(&down, 1.0 / 60.0, true, true);
        assert!(fired(&router, "voice.talk", still.actions), "the hold fires again with no new edge");
        let up = router.sample(&Press::wheel(0.0), 1.0 / 60.0, true, true);
        assert!(!fired(&router, "voice.talk", up.actions));
        router.set_context(Context::Menu);
        let menu = router.sample(&down, 1.0 / 60.0, true, true);
        assert!(!fired(&router, "voice.talk", menu.actions), "menus do not sample gameplay holds");
    }
}
