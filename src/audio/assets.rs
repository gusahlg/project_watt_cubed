//! Audio asset discovery and runtime tuning.
//!
//! Development runs normally find `<checkout>/assets`, while an installed Nix
//! package finds `<prefix>/share/project_watt_cubed/assets` beside its executable.
//! `WATT_ASSET_DIR` is an explicit override for modding, tests, and deployments
//! with a separate data directory.

use std::path::{Path, PathBuf};

pub const ASSET_DIR_ENV: &str = "WATT_ASSET_DIR";

const PROJECT_DIR: &str = "project_watt_cubed";
const DEFAULT_MAX_VOICES: usize = 32;
const DEFAULT_SMOOTHING_HALFLIFE_S: f32 = 0.05;
// Four 20 ms packets: matches `JITTER_REORDER_WINDOW` instead of advertising
// the old 40 ms value that the jitter buffer silently raised to 80 ms.
const DEFAULT_JITTER_TARGET_MS: u32 = 80;
const MAX_JITTER_TARGET_MS: u32 = 1_000;

/// The common root containing `sounds/` and the reserved `music/` library.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioAssets {
    root: PathBuf,
}

impl AudioAssets {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Resolve assets without depending on the process working directory alone.
    ///
    /// Search order is intentional:
    /// 1. explicit `WATT_ASSET_DIR`;
    /// 2. the installed package beside the current executable;
    /// 3. `assets/` below the current directory;
    /// 4. `assets/` below the compile-time Cargo manifest.
    ///
    /// An explicit override is authoritative even when invalid, so a typo yields
    /// an error naming that path instead of silently loading different content.
    pub fn discover() -> Self {
        if let Some(root) = std::env::var_os(ASSET_DIR_ENV) {
            return Self::new(root);
        }

        let installed = std::env::current_exe().ok().and_then(|exe| {
            exe.parent()?
                .parent()
                .map(|prefix| prefix.join("share").join(PROJECT_DIR).join("assets"))
        });
        let working = std::env::current_dir().ok().map(|cwd| cwd.join("assets"));
        let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets");

        installed
            .into_iter()
            .chain(working)
            .chain(std::iter::once(source.clone()))
            .find(|root| root.join("sounds").join("catalog.toml").is_file())
            .map_or_else(|| Self::new(source), Self::new)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn sounds(&self) -> PathBuf {
        self.root.join("sounds")
    }

    /// Reserved music library. Music is deliberately separate from short,
    /// eagerly-decoded effects so a future streamer can index it lazily.
    pub fn music(&self) -> PathBuf {
        self.root.join("music")
    }
}

impl Default for AudioAssets {
    fn default() -> Self {
        Self::discover()
    }
}

/// Runtime tuning handed to `SoundSystem`.
#[derive(Clone, Debug)]
pub struct SoundConfig {
    pub max_voices: usize,
    pub smoothing_halflife_s: f32,
    pub jitter_target_ms: u32,
    pub content_dir: PathBuf,
}

impl SoundConfig {
    pub fn from_assets(assets: &AudioAssets) -> Self {
        Self {
            max_voices: DEFAULT_MAX_VOICES,
            smoothing_halflife_s: DEFAULT_SMOOTHING_HALFLIFE_S,
            jitter_target_ms: DEFAULT_JITTER_TARGET_MS,
            content_dir: assets.sounds(),
        }
    }

    /// Make hostile or accidentally-corrupt runtime tuning safe before it can
    /// reach allocation or exponentiation paths. A zero voice budget remains a
    /// supported deliberate mute.
    pub(crate) fn normalize(&mut self) {
        if !self.smoothing_halflife_s.is_finite() || self.smoothing_halflife_s <= 0.0 {
            self.smoothing_halflife_s = DEFAULT_SMOOTHING_HALFLIFE_S;
        }
        self.jitter_target_ms = self.jitter_target_ms.min(MAX_JITTER_TARGET_MS);
    }

    pub(crate) fn silent() -> Self {
        Self {
            max_voices: 0,
            smoothing_halflife_s: DEFAULT_SMOOTHING_HALFLIFE_S,
            jitter_target_ms: DEFAULT_JITTER_TARGET_MS,
            content_dir: PathBuf::new(),
        }
    }
}

impl Default for SoundConfig {
    fn default() -> Self {
        Self::from_assets(&AudioAssets::discover())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::backend::null::NullBackend;
    use crate::audio::content::Catalog;
    use crate::audio::palette::CuePalette;

    #[test]
    fn config_from_assets_keeps_sounds_and_music_as_siblings() {
        let assets = AudioAssets::new("/opt/watt/assets");
        let cfg = SoundConfig::from_assets(&assets);
        assert_eq!(cfg.content_dir, PathBuf::from("/opt/watt/assets/sounds"));
        assert_eq!(assets.music(), PathBuf::from("/opt/watt/assets/music"));
    }

    #[test]
    fn normalization_repairs_only_dangerous_tuning() {
        let mut cfg = SoundConfig {
            max_voices: 0,
            smoothing_halflife_s: f32::NAN,
            jitter_target_ms: u32::MAX,
            content_dir: PathBuf::from("custom"),
        };
        cfg.normalize();
        assert_eq!(cfg.max_voices, 0);
        assert_eq!(cfg.smoothing_halflife_s, DEFAULT_SMOOTHING_HALFLIFE_S);
        assert_eq!(cfg.jitter_target_ms, MAX_JITTER_TARGET_MS);
        assert_eq!(cfg.content_dir, PathBuf::from("custom"));
    }

    #[test]
    fn checked_in_catalog_and_every_placeholder_decode_headlessly() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets");
        let mut backend = NullBackend::new();
        let (catalog, symbols) =
            Catalog::load(&root.join("sounds"), &mut backend).expect("packaged sound catalog");
        let (_, warnings) = CuePalette::build(&symbols, &catalog);
        assert!(warnings.is_empty(), "catalog role warnings: {warnings:?}");
        assert!(root.join("music").is_dir());
    }
}
