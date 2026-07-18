//! The cue homomorphism: every game fact the director turns into sound resolves
//! its `(CueId, gain)` here, once, at load. Gameplay never names a cue string —
//! it reports facts and the palette decides the cue. All gains are data
//! (currently 1.0); mode is checked against the `Cue.mode` invariant at build so
//! a Loop role that resolves to a OneShot cue (or vice versa) degrades to silence
//! with a warning instead of leaking a voice at runtime.

use std::collections::BTreeMap;

use crate::block::derive::SoundClass;

use super::content::{Catalog, CueId, CueMode, CueSymbols, Loop, OneShot};

/// In-game UI cues routed through the director (`SoundEvent::Ui`). Menu cues
/// (`menu_click`) stay a direct `SoundSystem::play_ui` on App and are NOT here.
/// Derived solely from game.rs's in-game `play_ui` call sites (only `/voicetest`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UiSound {
    VoiceTest,
}

/// A resolved sound effect: the admitted, mode-typed cue plus its authored gain
/// (event→data). `M` is the cue's mode, so a `Sfx<OneShot>` cannot be routed to an
/// emitter table nor a `Sfx<Loop>` to the occurrence journal.
#[derive(Debug)]
pub struct Sfx<M> {
    pub cue: CueId<M>,
    pub gain: f32,
}

// Copy/Clone by hand (like `CueId`) so `Sfx<M>` is unconditionally `Copy` — the
// generic palette builders copy it out of maps without an `M: Copy` bound.
impl<M> Clone for Sfx<M> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<M> Copy for Sfx<M> {}

/// The closed SoundClass set (block/derive.rs). Enumerated here so per-class cues
/// resolve at load; `registry().sound_class()` only ever returns one of these.
/// A new SoundClass variant must add a line here — the QA doc sweep guards drift.
const CLASSES: [SoundClass; 6] = [
    SoundClass::Stone,
    SoundClass::Soil,
    SoundClass::Wood,
    SoundClass::Glass,
    SoundClass::Foliage,
    SoundClass::Water,
];

const DEFAULT_GAIN: f32 = 1.0;

/// The load-time cue homomorphism. Per-class roles fold their `_default` fallback
/// in at build, so a runtime lookup is a single map hit with no string formatting.
pub struct CuePalette {
    step: BTreeMap<&'static str, Sfx<OneShot>>,
    break_: BTreeMap<&'static str, Sfx<OneShot>>,
    place: BTreeMap<&'static str, Sfx<OneShot>>,
    splash: Option<Sfx<OneShot>>,
    swing: Option<Sfx<OneShot>>,
    underwater_loop: Option<Sfx<Loop>>,
    voicetest: Option<Sfx<OneShot>>, // UI
}

impl CuePalette {
    /// Build once from the loaded catalog. Returns the palette plus one warning per
    /// missing top-level role or mode-mismatched cue (App prints them to the console).
    /// Each role's mode is the turbofish `M`, so the mode check IS the type mint.
    pub fn build(symbols: &CueSymbols, catalog: &Catalog) -> (Self, Vec<String>) {
        let mut warnings = Vec::new();
        let palette = Self {
            step: class_map::<OneShot>("step", symbols, catalog, &mut warnings),
            break_: class_map::<OneShot>("break", symbols, catalog, &mut warnings),
            place: class_map::<OneShot>("place", symbols, catalog, &mut warnings),
            splash: checked::<OneShot>("splash", symbols, catalog, &mut warnings, true),
            swing: checked::<OneShot>("swing", symbols, catalog, &mut warnings, true),
            underwater_loop: checked::<Loop>("underwater_loop", symbols, catalog, &mut warnings, true),
            voicetest: checked::<OneShot>("voicetest", symbols, catalog, &mut warnings, true),
        };
        (palette, warnings)
    }

    pub fn step(&self, class: &str) -> Option<Sfx<OneShot>> {
        self.step.get(class).copied()
    }
    pub fn break_block(&self, class: &str) -> Option<Sfx<OneShot>> {
        self.break_.get(class).copied()
    }
    pub fn place(&self, class: &str) -> Option<Sfx<OneShot>> {
        self.place.get(class).copied()
    }
    pub fn splash(&self) -> Option<Sfx<OneShot>> {
        self.splash
    }
    pub fn swing(&self) -> Option<Sfx<OneShot>> {
        self.swing
    }
    pub fn underwater_loop(&self) -> Option<Sfx<Loop>> {
        self.underwater_loop
    }
    pub fn ui(&self, sound: UiSound) -> Option<Sfx<OneShot>> {
        match sound {
            UiSound::VoiceTest => self.voicetest,
        }
    }
}

/// Resolve `{kind}_{class}` else `{kind}_default` for every SoundClass, folding the
/// default fallback in at build. A class-specific miss is normal (silent); only a
/// missing `_default` or a mode mismatch warns.
fn class_map<M: CueMode>(
    kind: &str,
    symbols: &CueSymbols,
    catalog: &Catalog,
    warnings: &mut Vec<String>,
) -> BTreeMap<&'static str, Sfx<M>> {
    let default = checked::<M>(&format!("{kind}_default"), symbols, catalog, warnings, true);
    let mut map = BTreeMap::new();
    for class in CLASSES {
        let name = class.as_str();
        let specific = checked::<M>(&format!("{kind}_{name}"), symbols, catalog, warnings, false);
        if let Some(sfx) = specific.or(default) {
            map.insert(name, sfx);
        }
    }
    map
}

/// Resolve one cue name into a mode-typed `Sfx<M>`, verifying it exists and its
/// cue-level mode is `M`. `warn_missing` distinguishes a required role (warn on
/// absence) from an optional class-specific override (silent absence, folds to the
/// default). A present-but-wrong-mode cue always warns and is dropped.
fn checked<M: CueMode>(
    name: &str,
    symbols: &CueSymbols,
    catalog: &Catalog,
    warnings: &mut Vec<String>,
    warn_missing: bool,
) -> Option<Sfx<M>> {
    match symbols.raw(name) {
        None => {
            if warn_missing {
                warnings.push(format!("audio: cue `{name}` missing — role silent"));
            }
            None
        }
        Some(raw) => match catalog.typed::<M>(symbols, name) {
            Some(cue) => Some(Sfx { cue, gain: DEFAULT_GAIN }),
            None => {
                warnings.push(format!(
                    "audio: cue `{name}` is {:?} but role needs {:?} — ignored",
                    catalog.mode_of(raw),
                    M::MODE
                ));
                None
            }
        },
    }
}
