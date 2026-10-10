//! What a HUD mod may read about the frame: the facts the core owns and a mod cannot derive
//! (frame rate, the network link, loading state, the HUD mode and scale). The core draws no HUD
//! text of its own; the information HUD is a mod built on these facts.

use crate::ui::HudMode;

/// This frame's facts for [`Mod::hud`](super::Mod::hud). Plain `Copy` data built on the stack.
/// Build one for a test with [`HudFacts::new`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct HudFacts {
    /// The window size in pixels.
    pub screen: (i32, i32),
    /// Frames per second as the HUD shows it, sampled at a readable cadence (4 Hz). `None` when
    /// there is no live reading to show (the scripted golden harness pins it).
    pub fps: Option<u32>,
    /// Round-trip time to the server, once measured. `None` in single player.
    pub ping_ms: Option<u32>,
    /// Players on the server, this one included. `None` in single player.
    pub players_online: Option<usize>,
    /// The server's join snapshot has landed (always true in single player).
    pub snapshot_ready: bool,
    /// The terrain around the spawn has loaded.
    pub spawn_ready: bool,
    /// The server has gone quiet and the link may be lost.
    pub link_interrupted: bool,
    /// The master HUD cycle (F1). The core calls `hud` in every mode, Off included; each mod
    /// decides what the mode means for it.
    pub hud_mode: HudMode,
    /// The player's UI scale. A label's `base_fs` is drawn at `round(base_fs * ui_scale)` pixels,
    /// and every glyph advances by exactly the font size.
    pub ui_scale: f32,
    /// The cruise speed in km/s while a cruise is on.
    pub cruise: Option<f64>,
    /// Width and height of the top-right corner the core's minimap takes, `(0, 0)` while it is
    /// hidden. HUD pieces in that corner go below or beside it.
    pub minimap_corner: (i32, i32),
}

impl HudFacts {
    /// Single-player facts for a test: loaded, no reading, Full HUD at scale 1, no minimap.
    pub fn new(screen: (i32, i32)) -> Self {
        Self {
            screen,
            fps: None,
            ping_ms: None,
            players_online: None,
            snapshot_ready: true,
            spawn_ready: true,
            link_interrupted: false,
            hud_mode: HudMode::Full,
            ui_scale: 1.0,
            cruise: None,
            minimap_corner: (0, 0),
        }
    }

    /// The pixel size a label of `base_fs` is drawn at: the one sizing rule the core's renderer
    /// applies.
    pub fn font_px(&self, base_fs: i32) -> i32 {
        (base_fs as f32 * self.ui_scale).round() as i32
    }
}
