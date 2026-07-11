//! Persistent graphics settings.
//!
//! Stored as plain `key=value` lines in `saves/settings.cfg` (std-only, no
//! dependencies). The settings menu and the `/gfx` console command both edit
//! a [`Settings`] value; [`Settings::apply`] pushes it to the engine, which
//! no-ops for values that didn't change.
//!
//! Single source of truth: every field is one entry in [`SETTINGS`], a table of
//! [`Setting`] descriptors. Persistence, the `/gfx` console command, the settings
//! menu, and [`Settings::clamp`] all fold over this ONE list — so a new setting is
//! declared in exactly one place and the three surfaces can never drift apart.
//! Each descriptor carries the field's behaviour as a flat set of `fn` pointers
//! (display / human-parse / step / clamp / machine codec); the uniform ones
//! delegate to shared helpers ([`wrap_clamp`], [`cycle_list`], [`snap_down`]),
//! so a field touches its own struct member directly — no common wire type to
//! exclude the float fields.
use std::fs;
use std::ops::RangeInclusive;
use std::path::Path;

use voxel_engine::Engine;

pub use crate::world::VIEW_RADIUS_RANGE;
/// Render-resolution scale clamp range — re-exported from the engine, which owns
/// the single source (it does the real clamp in `set_render_scale`). Re-exporting
/// here mirrors the [`VIEW_RADIUS_RANGE`] re-export so the settings UI and the
/// renderer can never disagree on the bound.
pub use voxel_engine::RENDER_SCALE_RANGE;

const SETTINGS_PATH: &str = "saves/settings.cfg";

/// Field-of-view clamp range, in degrees. Shared with the settings menu stepper.
pub const FOV_RANGE: RangeInclusive<f32> = 60.0..=220.0;

/// HUD/text scale clamp range (multiplier). Shared with the settings menu stepper.
pub const UI_SCALE_RANGE: RangeInclusive<f32> = 0.5..=2.0;

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
    /// Render-resolution scale relative to the window (0.25..=2.0).
    pub render_scale: f32,
    /// HUD/text scale, independent of render resolution (0.5..=2.0). Drives
    /// [`crate::ui::Theme::scale`].
    pub ui_scale: f32,
    /// Cross-chunk lighting. On by default; pushed to [`crate::world::World`] on
    /// world entry and on `/gfx` change (the engine has no say — it is a meshing
    /// input, not a GPU state).
    pub lighting: bool,
    /// Six-way back-face culling of chunk meshes. Not a menu/persisted setting:
    /// sourced once from `WATT_CULL=1` (a GPU-side trade only worth it when
    /// vertex-fetch bound), so it is absent from [`SETTINGS`] and pushed to the
    /// engine by [`apply`](Settings::apply) like the table fields.
    pub cull_faces: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            fullscreen: false,
            vsync: false,
            msaa: 1,
            max_fps: 0,
            render_distance: 6,
            fov: 90.0,
            render_scale: 1.0,
            ui_scale: 1.0,
            lighting: true,
            cull_faces: false,
        }
    }
}

// ---------------------------------------------------------------------------
// One `Setting` per field: a flat set of behaviour `fn`s folded over by every
// surface. No common wire type — each field touches its own struct member.
// ---------------------------------------------------------------------------

/// One setting: its persistence key (+ console aliases), menu label, and the
/// behaviour every surface needs, each as a plain `fn` pointer. Non-capturing
/// closures in [`SETTINGS`] fill these; the uniform fields delegate to the shared
/// helpers below, the float fields touch `f32` directly.
pub struct Setting {
    /// The `key=` name used in `saves/settings.cfg` and the primary console name.
    key: &'static str,
    /// Extra names the `/gfx` console command accepts for this field.
    aliases: &'static [&'static str],
    /// The settings-menu row label.
    label: &'static str,
    /// The exact `/gfx` confirmation line for the current value.
    confirm: fn(&Settings) -> String,
    /// The human-facing value string (menu display and `/gfx` value read-out).
    show: fn(&Settings) -> String,
    /// Parse a `/gfx` value (human form) into the field and clamp; `false` if it
    /// didn't parse (the field is then left untouched).
    parse_human: fn(&mut Settings, &str) -> bool,
    /// Apply one menu Left/Right step (`dir` = -1 or +1), wrapping at the ends.
    step: fn(&mut Settings, i32),
    /// Force the value back into its valid range. Safe to call repeatedly.
    clamp: fn(&mut Settings),
    /// The machine (persistence) text for the current value — save-compatible.
    write: fn(&Settings) -> String,
    /// Read a persisted value into the field (no clamp — [`Settings::clamp`] runs
    /// after the whole file is parsed). `false` if it didn't parse.
    read: fn(&mut Settings, &str) -> bool,
}

impl Setting {
    /// The settings-menu row label.
    pub fn label(&self) -> &'static str {
        self.label
    }

    /// Whether this field answers to `name` (its key or any console alias).
    pub fn matches(&self, name: &str) -> bool {
        self.key == name || self.aliases.contains(&name)
    }

    /// The human-facing value string (menu display and `/gfx` value read-out).
    pub fn show(&self, s: &Settings) -> String {
        (self.show)(s)
    }

    /// Apply one menu Left/Right step (`dir` = -1 or +1), wrapping at the ends.
    pub fn step(&self, s: &mut Settings, dir: i32) {
        (self.step)(s, dir)
    }

    /// Parse a `/gfx` value and clamp. Returns whether the value parsed.
    pub fn parse_human(&self, s: &mut Settings, value: &str) -> bool {
        (self.parse_human)(s, value)
    }

    /// The exact `/gfx` confirmation line for the current value.
    pub fn confirm(&self, s: &Settings) -> String {
        (self.confirm)(s)
    }

    fn clamp(&self, s: &mut Settings) {
        (self.clamp)(s)
    }

    fn write(&self, s: &Settings) -> String {
        (self.write)(s)
    }

    fn read(&self, s: &mut Settings, value: &str) -> bool {
        (self.read)(s, value)
    }
}

/// The MSAA sample counts offered — one list shared by its stepper and its
/// "round down to a supported count" clamp bucket.
const MSAA: &[i32] = &[1, 2, 4, 8];

/// Every setting, in menu/persistence order. The single source of the field set;
/// persistence, `/gfx`, the menu, and [`Settings::clamp`] all fold over it.
pub const SETTINGS: [Setting; 9] = [
    Setting {
        key: "fullscreen",
        aliases: &[],
        label: "Fullscreen",
        confirm: |s| format!("fullscreen {}", on_off(s.fullscreen, false)),
        show: |s| on_off(s.fullscreen, true).to_string(),
        parse_human: |s, v| set_bool(&mut s.fullscreen, v),
        step: |s, _| s.fullscreen = !s.fullscreen,
        clamp: |_| {},
        write: |s| s.fullscreen.to_string(),
        read: |s, v| set_bool(&mut s.fullscreen, v),
    },
    Setting {
        key: "vsync",
        aliases: &[],
        label: "VSync",
        confirm: |s| format!("vsync {}", on_off(s.vsync, false)),
        show: |s| on_off(s.vsync, true).to_string(),
        parse_human: |s, v| set_bool(&mut s.vsync, v),
        step: |s, _| s.vsync = !s.vsync,
        clamp: |_| {},
        write: |s| s.vsync.to_string(),
        read: |s, v| set_bool(&mut s.vsync, v),
    },
    Setting {
        key: "lighting",
        aliases: &["light"],
        label: "Lighting",
        confirm: |s| format!("lighting {}", on_off(s.lighting, false)),
        show: |s| on_off(s.lighting, true).to_string(),
        parse_human: |s, v| set_bool(&mut s.lighting, v),
        step: |s, _| s.lighting = !s.lighting,
        clamp: |_| {},
        write: |s| s.lighting.to_string(),
        read: |s, v| set_bool(&mut s.lighting, v),
    },
    Setting {
        key: "msaa",
        aliases: &[],
        label: "MSAA",
        confirm: |s| format!("msaa {}x", s.msaa),
        show: |s| format!("{}x", s.msaa),
        parse_human: |s, v| {
            let ok = set_parsed(&mut s.msaa, v);
            if ok {
                msaa_clamp(s);
            }
            ok
        },
        step: |s, d| s.msaa = cycle_list(MSAA, s.msaa as i32, d) as u32,
        clamp: msaa_clamp,
        write: |s| s.msaa.to_string(),
        read: |s, v| set_parsed(&mut s.msaa, v),
    },
    Setting {
        key: "max_fps",
        aliases: &["fps"],
        label: "Max FPS",
        confirm: |s| {
            if s.max_fps == 0 {
                "fps cap off".to_string()
            } else {
                format!("fps cap {}", s.max_fps)
            }
        },
        // `0` = uncapped, otherwise the stepper cycles a fixed list but the clamp
        // is a *range* (10..=1000), so it can't collapse to a plain choice list.
        show: |s| {
            if s.max_fps == 0 {
                "Uncapped".to_string()
            } else {
                s.max_fps.to_string()
            }
        },
        parse_human: |s, v| {
            let n = match v {
                "off" | "uncapped" | "0" => 0,
                _ => match v.parse::<u32>() {
                    Ok(n) => n,
                    Err(_) => return false,
                },
            };
            s.max_fps = n;
            fps_clamp(s);
            true
        },
        step: |s, d| s.max_fps = cycle_list(&[0, 30, 60, 120, 144, 240], s.max_fps as i32, d) as u32,
        clamp: fps_clamp,
        write: |s| s.max_fps.to_string(),
        read: |s, v| set_parsed(&mut s.max_fps, v),
    },
    Setting {
        key: "render_distance",
        aliases: &["renderdist", "renderdistance"],
        label: "Render Distance",
        confirm: |s| format!("render distance {}", s.render_distance),
        show: |s| s.render_distance.to_string(),
        parse_human: |s, v| {
            let ok = set_parsed(&mut s.render_distance, v);
            if ok {
                dist_clamp(s);
            }
            ok
        },
        step: |s, d| {
            s.render_distance = wrap_clamp(
                s.render_distance,
                *VIEW_RADIUS_RANGE.start(),
                *VIEW_RADIUS_RANGE.end(),
                d,
            );
        },
        clamp: dist_clamp,
        write: |s| s.render_distance.to_string(),
        read: |s, v| set_parsed(&mut s.render_distance, v),
    },
    Setting {
        key: "fov",
        aliases: &[],
        label: "FOV",
        confirm: |s| format!("fov {:.0}", s.fov),
        // f32 Display prints whole values without a decimal point, exactly as the
        // old `format!("FOV: {}", s.fov)` screen did.
        show: |s| format!("{}", s.fov),
        parse_human: |s, v| {
            let ok = set_parsed(&mut s.fov, v);
            if ok {
                fov_clamp(s);
            }
            ok
        },
        step: |s, d| {
            let (lo, hi) = (*FOV_RANGE.start(), *FOV_RANGE.end());
            let v = clamp_to(&FOV_RANGE, s.fov) + d as f32 * 5.0;
            s.fov = if v > hi {
                lo
            } else if v < lo {
                hi
            } else {
                v
            };
        },
        clamp: fov_clamp,
        write: |s| s.fov.to_string(),
        read: |s, v| set_parsed(&mut s.fov, v),
    },
    Setting {
        key: "render_scale",
        aliases: &["renderscale", "scale"],
        label: "Render Scale",
        confirm: |s| format!("render scale {:.0}%", s.render_scale * 100.0),
        // Percent-encoded for humans (75%), stored raw (0.75) for save-compat.
        show: |s| format!("{:.0}%", s.render_scale * 100.0),
        parse_human: |s, v| match v.parse::<f32>() {
            Ok(pct) => {
                s.render_scale = pct / 100.0;
                scale_clamp(s);
                true
            }
            Err(_) => false,
        },
        step: |s, d| {
            let pct = cycle_list(
                &[25, 50, 75, 100, 125, 150, 200],
                (s.render_scale * 100.0).round() as i32,
                d,
            );
            s.render_scale = pct as f32 / 100.0;
        },
        clamp: scale_clamp,
        write: |s| s.render_scale.to_string(),
        read: |s, v| set_parsed(&mut s.render_scale, v),
    },
    Setting {
        key: "ui_scale",
        aliases: &["uiscale", "hudscale"],
        label: "UI Scale",
        confirm: |s| format!("ui scale {:.0}%", s.ui_scale * 100.0),
        show: |s| format!("{:.0}%", s.ui_scale * 100.0),
        parse_human: |s, v| match v.parse::<f32>() {
            Ok(pct) => {
                s.ui_scale = pct / 100.0;
                ui_scale_clamp(s);
                true
            }
            Err(_) => false,
        },
        step: |s, d| {
            let pct = cycle_list(
                &[50, 75, 100, 125, 150, 200],
                (s.ui_scale * 100.0).round() as i32,
                d,
            );
            s.ui_scale = pct as f32 / 100.0;
        },
        clamp: ui_scale_clamp,
        write: |s| s.ui_scale.to_string(),
        read: |s, v| set_parsed(&mut s.ui_scale, v),
    },
];

/// The Back action sits just past the settings rows — derived, never hand-numbered.
pub const SETTINGS_ROW_BACK: usize = SETTINGS.len();

impl Settings {
    /// Load from disk, falling back to defaults for missing/invalid entries.
    pub fn load() -> Self {
        let mut settings = Self::default();
        if let Ok(text) = fs::read_to_string(SETTINGS_PATH) {
            settings.parse_from(&text);
        }
        // Six-way cull is env-only (not in the persisted table): opt in with
        // `WATT_CULL=1`. Read after the file parse so it can't be overwritten.
        settings.cull_faces = matches!(std::env::var("WATT_CULL").as_deref(), Ok("1"));
        settings.clamp();
        settings
    }

    fn parse_from(&mut self, text: &str) {
        for line in text.lines() {
            let Some((key, value)) = line.trim().split_once('=') else {
                continue;
            };
            let (key, value) = (key.trim(), value.trim());
            if let Some(field) = SETTINGS.iter().find(|f| f.matches(key)) {
                field.read(self, value);
            }
        }
    }

    /// Serialize every field to `key=value` lines — the exact text [`save`] writes.
    ///
    /// [`save`]: Settings::save
    fn to_text(&self) -> String {
        SETTINGS
            .iter()
            .map(|f| format!("{}={}\n", f.key, f.write(self)))
            .collect()
    }

    /// Best-effort save (a failed write shouldn't crash the game).
    pub fn save(&self) {
        if let Some(dir) = Path::new(SETTINGS_PATH).parent() {
            let _ = fs::create_dir_all(dir);
        }
        let _ = fs::write(SETTINGS_PATH, self.to_text());
    }

    /// Force every field into its valid range. Safe to call repeatedly, and
    /// handles NaN/±INF by mapping to an endpoint, so no non-finite value can
    /// reach the renderer.
    pub fn clamp(&mut self) {
        for field in &SETTINGS {
            field.clamp(self);
        }
    }

    /// Push the current values to the engine. Cheap to call every frame: the
    /// engine ignores values that didn't change. MSAA is written back with
    /// the hardware-clamped value so menus and `/gfx` show what actually
    /// applied (e.g. 8x requested, 4x supported).
    pub fn apply(&mut self, eng: &mut Engine) {
        eng.set_fullscreen(self.fullscreen);
        eng.set_vsync(self.vsync);
        self.msaa = eng.set_msaa(self.msaa);
        self.render_scale = eng.set_render_scale(self.render_scale);
        eng.set_target_fps(self.max_fps);
        eng.set_cull_faces(self.cull_faces);
    }
}

// ---------------------------------------------------------------------------
// Shared value helpers — the single definition each surface reuses.
// ---------------------------------------------------------------------------

/// Parse an on/off word. The one toggle parser (persistence AND `/gfx`).
pub fn parse_toggle(value: &str) -> Option<bool> {
    match value {
        "true" | "on" | "1" | "yes" => Some(true),
        "false" | "off" | "0" | "no" => Some(false),
        _ => None,
    }
}

/// Store a parsed on/off word into `dst`, leaving it untouched on a bad value.
/// The one setter shared by a toggle's `parse_human` and `read`.
fn set_bool(dst: &mut bool, value: &str) -> bool {
    match parse_toggle(value) {
        Some(b) => {
            *dst = b;
            true
        }
        None => false,
    }
}

/// Store a parsed numeric value; returns false if parse fails. Clamping is a
/// separate step so persistence can read raw values.
fn set_parsed<T: std::str::FromStr>(dst: &mut T, value: &str) -> bool {
    match value.parse() {
        Ok(n) => {
            *dst = n;
            true
        }
        Err(_) => false,
    }
}

/// Render a boolean as on/off text. `caps` picks the menu style (`On`/`Off`) over
/// the console style (`on`/`off`).
pub fn on_off(v: bool, caps: bool) -> &'static str {
    match (v, caps) {
        (true, true) => "On",
        (false, true) => "Off",
        (true, false) => "on",
        (false, false) => "off",
    }
}

/// Clamp a finite value to a range. NaN should be reset beforehand (see [`reset_nan`]).
fn clamp_to(range: &RangeInclusive<f32>, v: f32) -> f32 {
    v.clamp(*range.start(), *range.end())
}

/// Reset NaN to the default value (f32::clamp would leave NaN untouched).
fn reset_nan(v: f32, default: f32) -> f32 {
    if v.is_nan() { default } else { v }
}

fn fov_clamp(s: &mut Settings) {
    s.fov = clamp_to(&FOV_RANGE, reset_nan(s.fov, Settings::default().fov));
}

fn scale_clamp(s: &mut Settings) {
    s.render_scale = clamp_to(
        &RENDER_SCALE_RANGE,
        reset_nan(s.render_scale, Settings::default().render_scale),
    );
}

fn ui_scale_clamp(s: &mut Settings) {
    s.ui_scale = clamp_to(
        &UI_SCALE_RANGE,
        reset_nan(s.ui_scale, Settings::default().ui_scale),
    );
}

fn fps_clamp(s: &mut Settings) {
    if s.max_fps != 0 {
        s.max_fps = s.max_fps.clamp(10, 1000);
    }
}

fn msaa_clamp(s: &mut Settings) {
    s.msaa = snap_down(MSAA, s.msaa as i32) as u32;
}

fn dist_clamp(s: &mut Settings) {
    s.render_distance =
        s.render_distance.clamp(*VIEW_RADIUS_RANGE.start(), *VIEW_RADIUS_RANGE.end());
}

/// Step an integer within a range, wrapping at ends.
fn wrap_clamp(cur: i32, lo: i32, hi: i32, delta: i32) -> i32 {
    let v = cur.clamp(lo, hi) + delta;
    if v > hi {
        lo
    } else if v < lo {
        hi
    } else {
        v
    }
}

/// Step to adjacent entry in a list, wrapping at ends. Values not in list snap to first.
fn cycle_list(list: &[i32], current: i32, dir: i32) -> i32 {
    match list.iter().position(|&v| v == current) {
        Some(i) => list[(i as i32 + dir).rem_euclid(list.len() as i32) as usize],
        None => list[0],
    }
}

/// Snap to the largest list entry <= v (or first entry if none found).
fn snap_down(list: &[i32], v: i32) -> i32 {
    list.iter().rev().copied().find(|&e| e <= v).unwrap_or(list[0])
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
        s.render_scale = 0.75;
        // Same table-driven serialization as `save`, so this can't drift from
        // what `parse_from` reads.
        let text = s.to_text();
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
        assert_eq!(s.fov, 90.0);
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
            render_scale: 9.0,
            ui_scale: 1.0,
            lighting: true,
            cull_faces: false,
        };
        s.clamp();
        assert_eq!(s.msaa, 4);
        assert_eq!(s.max_fps, 10);
        assert_eq!(s.render_distance, 20);
        assert_eq!(s.fov, 220.0);
        assert_eq!(s.render_scale, 2.0);
    }

    #[test]
    fn nan_resets_to_default_and_infinity_clamps() {
        // `f32::clamp` alone would let NaN sail through to the renderer. NaN is a
        // corrupt/garbage sentinel, so it resets to the field default; ±INF is a
        // genuine over/underflow and clamps to the near endpoint.
        let defaults = Settings::default();
        let mut s = Settings::default();
        s.fov = f32::NAN;
        s.render_scale = f32::NAN;
        s.clamp();
        assert_eq!(s.fov, defaults.fov);
        assert_eq!(s.render_scale, defaults.render_scale);

        let mut s = Settings::default();
        s.fov = f32::INFINITY;
        s.render_scale = f32::NEG_INFINITY;
        s.clamp();
        assert_eq!(s.fov, *FOV_RANGE.end());
        assert_eq!(s.render_scale, *RENDER_SCALE_RANGE.start());
    }

    #[test]
    fn choices_clamp_matches_the_old_msaa_bucket() {
        // The generic "snap down to the list" clamp must reproduce the hand-written
        // MSAA bucket (0|1->1, 2..=3->2, 4..=7->4, _->8) exactly.
        fn old_bucket(v: u32) -> u32 {
            match v {
                0 | 1 => 1,
                2..=3 => 2,
                4..=7 => 4,
                _ => 8,
            }
        }
        for v in 0..=12 {
            assert_eq!(snap_down(&[1, 2, 4, 8], v), old_bucket(v as u32) as i32, "v={v}");
        }
    }

    #[test]
    fn every_menu_reachable_value_is_clamp_stable() {
        // Stepping never leaves a field outside its valid set, and clamping a
        // stepped value is a no-op. Walk far enough to cover every wrap cycle.
        for field in &SETTINGS {
            for &dir in &[1, -1] {
                let mut s = Settings::default();
                for _ in 0..40 {
                    field.step(&mut s, dir);
                    // Clamping a stepped value should change nothing...
                    let mut c = s.clone();
                    c.clamp();
                    assert_eq!(c, s, "{} not clamp-stable after step", field.label());
                    // ...and clamping twice should match clamping once.
                    let mut c2 = c.clone();
                    c2.clamp();
                    assert_eq!(c2, c, "{} clamping twice changed something", field.label());
                }
            }
        }
    }

    #[test]
    fn persist_codec_roundtrips_per_field() {
        // read(write(v)) == v for a spread of values.
        let samples = Settings {
            fullscreen: true,
            vsync: true,
            msaa: 8,
            max_fps: 144,
            render_distance: 9,
            fov: 85.0,
            render_scale: 1.25,
            ui_scale: 1.25,
            lighting: false,
            // Not persisted (env-only); must stay at the default so the composed
            // roundtrip below — which never writes it — still lands `samples`.
            cull_faces: false,
        };
        for field in &SETTINGS {
            let mut back = Settings::default();
            assert!(field.read(&mut back, &field.write(&samples)), "{}", field.label());
        }
        // The composed roundtrip lands the exact struct.
        let mut back = Settings::default();
        back.parse_from(&samples.to_text());
        back.clamp();
        assert_eq!(back, samples);
    }

    #[test]
    fn console_aliases_resolve_and_caps_differ() {
        // Aliases reach the same field...
        let mut a = Settings::default();
        let mut b = Settings::default();
        let dist = SETTINGS.iter().find(|f| f.matches("renderdist")).unwrap();
        let dist2 = SETTINGS.iter().find(|f| f.matches("renderdistance")).unwrap();
        assert!(dist.parse_human(&mut a, "8"));
        assert!(dist2.parse_human(&mut b, "8"));
        assert_eq!(a.render_distance, 8);
        assert_eq!(b.render_distance, 8);
        // ...and the one on/off helper gives menu caps vs console lowercase.
        assert_eq!(on_off(true, true), "On");
        assert_eq!(on_off(true, false), "on");
    }
}
