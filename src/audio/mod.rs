//! Client audio. The device, mixer, codecs and acoustics stay here. Mods decide
//! which cue plays and who hears voice, through [`service`].

pub(crate) mod acoustics;
mod assets;
mod capture;
mod content;
mod frame;
pub(crate) mod host;
mod service;
mod voice;

pub(crate) mod backend;
pub(crate) mod runtime;

pub(crate) use acoustics::Listener;
pub(crate) use assets::SoundConfig;
pub(crate) use content::CueSymbols;
pub(crate) use runtime::{Fault, MixChange, Smoothed, SoundSystem};

pub(crate) use service::StepPose;
pub use service::{
    AudioApi, AudioBench, AudioService, AudioView, BlockSound, CapturedFrame, GameEvent, ModFrame, ModLink, PeerAudio,
    Play,
};
