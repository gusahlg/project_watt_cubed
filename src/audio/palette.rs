//! The cue homomorphism: every game fact the director turns into sound resolves
//! its `(CueId, gain)` here, once, at load. Gameplay never names a cue string —
//! it reports facts and the palette decides the cue. All gains are data
//! (currently 1.0); mode is checked against the `Cue.mode` invariant at build so
//! a Loop role that resolves to a OneShot cue (or vice versa) degrades to silence
//! with a warning instead of leaking a voice at runtime.

use std::collections::BTreeMap;

use crate::block::derive::SoundClass;

use super::acoustics::Response;
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
    SoundClass::Open,
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
            step: class_map::<OneShot>("step", Response::World, symbols, catalog, &mut warnings),
            break_: class_map::<OneShot>("break", Response::World, symbols, catalog, &mut warnings),
            place: class_map::<OneShot>("place", Response::World, symbols, catalog, &mut warnings),
            splash: checked::<OneShot>(
                "splash",
                Response::World,
                symbols,
                catalog,
                &mut warnings,
                true,
            ),
            swing: checked::<OneShot>(
                "swing",
                Response::World,
                symbols,
                catalog,
                &mut warnings,
                true,
            ),
            underwater_loop: checked::<Loop>(
                "underwater_loop",
                Response::Ambient,
                symbols,
                catalog,
                &mut warnings,
                true,
            ),
            voicetest: checked::<OneShot>(
                "voicetest",
                Response::Ui,
                symbols,
                catalog,
                &mut warnings,
                true,
            ),
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
    response: Response,
    symbols: &CueSymbols,
    catalog: &Catalog,
    warnings: &mut Vec<String>,
) -> BTreeMap<&'static str, Sfx<M>> {
    let default = checked::<M>(
        &format!("{kind}_default"),
        response,
        symbols,
        catalog,
        warnings,
        true,
    );
    let mut map = BTreeMap::new();
    for class in CLASSES {
        let name = class.as_str();
        let specific = checked::<M>(
            &format!("{kind}_{name}"),
            response,
            symbols,
            catalog,
            warnings,
            false,
        );
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
    response: Response,
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
            Some(cue) if catalog.response_of(raw) == response => Some(Sfx {
                cue,
                gain: DEFAULT_GAIN,
            }),
            Some(_) => {
                warnings.push(format!(
                    "audio: cue `{name}` has {:?} response but role needs {response:?} — ignored",
                    catalog.response_of(raw)
                ));
                None
            }
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::audio::backend::{ClipId, ClipStore, StoredClip};

    struct Clips;

    impl ClipStore for Clips {
        fn store(&mut self, _bytes: &[u8]) -> Result<StoredClip, String> {
            Ok(StoredClip {
                id: ClipId(0),
                duration_s: 0.1,
            })
        }
    }

    #[test]
    fn role_rejects_a_mode_correct_but_wrong_response_cue() {
        let manifest = r#"
            [cues.step_default]
            response = "ui"
            [[cues.step_default.layers]]
            variants = ["step.wav"]
            gain = [1.0]
            pitch = [0.0]
            delay = [0.0]
            mode = "one_shot"
        "#;
        let mut clips = Clips;
        let mut resolve = |_name: &str| -> Result<Vec<u8>, PathBuf> { Ok(vec![0]) };
        let (catalog, symbols) =
            Catalog::from_manifest(manifest, &mut resolve, &mut clips).unwrap();
        let (palette, warnings) = CuePalette::build(&symbols, &catalog);

        assert!(palette.step("stone").is_none());
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("response") && warning.contains("step_default"))
        );
    }
}
