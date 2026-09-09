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
pub use acoustics::{Listener, Medium, Response};
pub use assets::{AudioAssets, SoundConfig};
pub use capture::{Capture, CaptureConfig, CaptureError, EncodedFrame};
pub use content::{CueId, CueMode, CueSymbols, Loop, OneShot};
pub use director::{AudioCtx, AudioDirector, PeerPose, PlayerPose, SoundEvent};
pub use frame::{AudioFrame, Emitter, EmitterId, FrameError, Occurrence, OccurrenceId};
pub use palette::{CuePalette, Sfx, UiSound};
pub use runtime::{Fault, MixChange, SoundInitError, SoundSystem};
pub use voice::{Epoch, Seq, SessionKey, VoicePacket};
pub(crate) use runtime::Smoothed;

