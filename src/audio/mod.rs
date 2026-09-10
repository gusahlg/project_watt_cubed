//! Client audio facade: catalog, director, frame, and the [`SoundSystem`] runtime.

pub mod acoustics;
pub mod assets;
pub mod capture;
pub mod content;
pub mod director;
pub mod frame;
pub(crate) mod host;
pub mod palette;
pub mod voice;

pub(crate) mod backend;
pub(crate) mod runtime;

// Re-exports so callers name these through `crate::audio::*` (the seam surface).
pub use acoustics::{Listener, Medium};
pub use assets::SoundConfig;
pub use content::{CueSymbols, OneShot};
pub use director::{AudioCtx, AudioDirector, PeerPose, PlayerPose, SoundEvent};
pub use frame::{AudioFrame, Emitter, EmitterId, Occurrence, OccurrenceId};
pub use palette::{CuePalette, UiSound};
pub use runtime::{Fault, MixChange, SoundSystem};
pub use voice::{Epoch, Seq, SessionKey, VoicePacket};
pub(crate) use runtime::Smoothed;

