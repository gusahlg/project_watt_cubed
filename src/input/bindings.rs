//! Complete binding tables mapping intents to input sources and chords.
//! Every entry is always present (unbound reads as an empty vec), so lookups never need a default case.
use voxel_engine::{Key, MouseButton};

use crate::input::intent::{
    AxisSource, Chord, GameplayAxis, GameplayEvent, GameplayState, GlobalEvent, MenuEvent, Source,
};
use crate::input::look;

pub struct Bindings {
    pub gameplay_state: [Vec<Source>; GameplayState::COUNT],
    pub gameplay_event: [Vec<Chord>; GameplayEvent::COUNT],
    pub gameplay_axis: [AxisSource; GameplayAxis::COUNT],
    pub look_x: AxisSource,
    pub look_y: AxisSource,
    pub menu_event: [Vec<Chord>; MenuEvent::COUNT],
    pub global_event: [Vec<Chord>; GlobalEvent::COUNT],
}

impl Default for Bindings {
    fn default() -> Self {
        use GameplayEvent as GE;
        use GameplayState as GS;
        use MenuEvent as ME;
        use GlobalEvent as GLE;

        let mut gameplay_state: [Vec<Source>; GameplayState::COUNT] = Default::default();
        gameplay_state[GS::Sprint as usize] = vec![Source::Key(Key::LeftControl)];
        gameplay_state[GS::Sneak as usize] = vec![Source::Key(Key::LeftShift)];
        gameplay_state[GS::Jump as usize] = vec![Source::Key(Key::Space)];

        let mut gameplay_event: [Vec<Chord>; GameplayEvent::COUNT] = Default::default();
        gameplay_event[GE::ToggleFly as usize] = vec![Chord::key(Key::F)];
        gameplay_event[GE::Break as usize] = vec![Chord::mouse(MouseButton::Left)];
        gameplay_event[GE::Place as usize] = vec![Chord::mouse(MouseButton::Right)];
        gameplay_event[GE::OpenConsole as usize] = vec![Chord::key(Key::Slash)];
        gameplay_event[GE::OpenChat as usize] = vec![Chord::key(Key::T)];
        gameplay_event[GE::ToggleInventory as usize] = vec![Chord::key(Key::I)];
        gameplay_event[GE::ToggleCrafting as usize] = vec![Chord::key(Key::C)];
        gameplay_event[GE::ToggleCapture as usize] = vec![Chord::key(Key::Tab)];

        // Coordinate system: forward (+Z) = W/S, right (+X) = D/A, up (+Y) = Space/LeftShift.
        let gameplay_axis = [
            AxisSource::KeyPair { neg: Source::Key(Key::A), pos: Source::Key(Key::D) },
            AxisSource::KeyPair { neg: Source::Key(Key::LeftShift), pos: Source::Key(Key::Space) },
            AxisSource::KeyPair { neg: Source::Key(Key::S), pos: Source::Key(Key::W) },
        ];

        // Vim and arrow keys for navigation preferences.
        let mut menu_event: [Vec<Chord>; MenuEvent::COUNT] = Default::default();
        menu_event[ME::Up as usize] =
            vec![Chord::key(Key::Up), Chord::key(Key::K), Chord::bare(Source::WheelUp)];
        menu_event[ME::Down as usize] =
            vec![Chord::key(Key::Down), Chord::key(Key::J), Chord::bare(Source::WheelDown)];
        menu_event[ME::Left as usize] = vec![Chord::key(Key::Left)];
        menu_event[ME::Right as usize] = vec![Chord::key(Key::Right)];
        menu_event[ME::Confirm as usize] = vec![Chord::key(Key::Enter), Chord::key(Key::L)];
        menu_event[ME::Back as usize] = vec![Chord::key(Key::Escape), Chord::key(Key::H)];
        menu_event[ME::NextTab as usize] = vec![Chord::key(Key::Tab)];
        menu_event[ME::Toggle as usize] = vec![Chord::key(Key::Space)];
        menu_event[ME::Delete as usize] = vec![Chord::key(Key::Backspace)];

        let mut global_event: [Vec<Chord>; GlobalEvent::COUNT] = Default::default();
        global_event[GLE::Escape as usize] = vec![Chord::key(Key::Escape)];
        global_event[GLE::CycleHud as usize] = vec![Chord::key(Key::F1)];
        global_event[GLE::Screenshot as usize] = vec![Chord::key(Key::F2)];
        global_event[GLE::MinimapMode as usize] = vec![Chord::key(Key::F3)];

        Self {
            gameplay_state,
            gameplay_event,
            gameplay_axis,
            look_x: AxisSource::MouseX { sens: look::SENSITIVITY, invert: false },
            look_y: AxisSource::MouseY { sens: look::SENSITIVITY, invert: true },
            menu_event,
            global_event,
        }
    }
}
