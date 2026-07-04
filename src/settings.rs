//! Persistent graphics settings.
//!
//! Stored as plain `key=value` lines in `saves/settings.cfg` (std-only, no
//! dependencies). The settings menu and the `/gfx` console command both edit
//! a [`Settings`] value; [`Settings::apply`] pushes it to the engine, which
//! no-ops for values that didn't change.
use std::fs;
use std::path::Path;

use voxel_engine::Engine;

const SETTINGS_PATH: &str = "saves/settings.cfg";

#[derive(Clone, PartialEq, Debug)]
pub struct Settings {
    pub fullscreen: bool,
    pub vsync: bool,
    /// MSAA sample count: 1 (off), 2, 4 or 8. Clamped to hardware support on apply.
    pub msaa: u32,
    /// Frame cap; 0 = uncapped.
    pub max_fps: u32,
    /// Chunk view radius (world streaming + drawing).
    pub render_distance: i32,
    /// Vertical field of view in degrees.
    pub fov: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            fullscreen: false,
            vsync: false,
            msaa: 1,
            max_fps: 0,
            render_distance: 6,
            fov: 70.0,
        }
    }
}

impl Settings {
    /// Load from disk, falling back to defaults for missing/invalid entries.
    pub fn load() -> Self {
        let mut settings = Self::default();
        if let Ok(text) = fs::read_to_string(SETTINGS_PATH) {
            settings.parse_from(&text);
        }
        settings.clamp();
        settings
    }

    fn parse_from(&mut self, text: &str) {
        for line in text.lines() {
            let line = line.trim();
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            let (key, value) = (key.trim(), value.trim());
            match key {
                "fullscreen" => self.fullscreen = parse_bool(value).unwrap_or(self.fullscreen),
                "vsync" => self.vsync = parse_bool(value).unwrap_or(self.vsync),
                "msaa" => self.msaa = value.parse().unwrap_or(self.msaa),
                "max_fps" => self.max_fps = value.parse().unwrap_or(self.max_fps),
                "render_distance" => {
                    self.render_distance = value.parse().unwrap_or(self.render_distance)
                }
                "fov" => self.fov = value.parse().unwrap_or(self.fov),
                _ => {}
            }
        }
    }

    /// Best-effort save (a failed write shouldn't crash the game).
    pub fn save(&self) {
        if let Some(dir) = Path::new(SETTINGS_PATH).parent() {
            let _ = fs::create_dir_all(dir);
        }
        let text = format!(
            "fullscreen={}\nvsync={}\nmsaa={}\nmax_fps={}\nrender_distance={}\nfov={}\n",
            self.fullscreen, self.vsync, self.msaa, self.max_fps, self.render_distance, self.fov
        );
        let _ = fs::write(SETTINGS_PATH, text);
    }

    /// Force every field into its valid range.
    pub fn clamp(&mut self) {
        self.msaa = match self.msaa {
            0 | 1 => 1,
            2..=3 => 2,
            4..=7 => 4,
            _ => 8,
        };
        if self.max_fps != 0 {
            self.max_fps = self.max_fps.clamp(10, 1000);
        }
        self.render_distance = self.render_distance.clamp(3, 10);
        self.fov = self.fov.clamp(50.0, 110.0);
    }

    /// Push the current values to the engine. Cheap to call every frame: the
    /// engine ignores values that didn't change. MSAA is written back with
    /// the hardware-clamped value so menus and `/gfx` show what actually
    /// applied (e.g. 8x requested, 4x supported).
    pub fn apply(&mut self, eng: &mut Engine) {
        eng.set_fullscreen(self.fullscreen);
        eng.set_vsync(self.vsync);
        self.msaa = eng.set_msaa(self.msaa);
        eng.set_target_fps(self.max_fps);
    }
}

fn parse_bool(value: &str) -> Option<bool> {
    match value {
        "true" | "on" | "1" | "yes" => Some(true),
        "false" | "off" | "0" | "no" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_through_text() {
        let mut s = Settings::default();
        s.fullscreen = true;
        s.msaa = 4;
        s.max_fps = 144;
        s.render_distance = 8;
        s.fov = 90.0;
        let text = format!(
            "fullscreen={}\nvsync={}\nmsaa={}\nmax_fps={}\nrender_distance={}\nfov={}\n",
            s.fullscreen, s.vsync, s.msaa, s.max_fps, s.render_distance, s.fov
        );
        let mut loaded = Settings::default();
        loaded.parse_from(&text);
        loaded.clamp();
        assert_eq!(loaded, s);
    }

    #[test]
    fn invalid_lines_keep_defaults() {
        let mut s = Settings::default();
        s.parse_from("garbage\nmsaa=lots\nfov=\nrender_distance=7");
        s.clamp();
        assert_eq!(s.msaa, 1);
        assert_eq!(s.fov, 70.0);
        assert_eq!(s.render_distance, 7);
    }

    #[test]
    fn clamp_forces_valid_ranges() {
        let mut s = Settings {
            fullscreen: false,
            vsync: false,
            msaa: 6,
            max_fps: 5,
            render_distance: 99,
            fov: 300.0,
        };
        s.clamp();
        assert_eq!(s.msaa, 4);
        assert_eq!(s.max_fps, 10);
        assert_eq!(s.render_distance, 10);
        assert_eq!(s.fov, 110.0);
    }
}
