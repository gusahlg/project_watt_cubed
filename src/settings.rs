//! Persistent graphics settings.
//!
//! Stored as plain `key=value` lines in `settings.cfg` under the config root
//! (std-only, no dependencies). The settings menu and the `/gfx` console command both edit
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
use std::fmt::Write;
use std::fs;
use std::ops::RangeInclusive;
use std::path::PathBuf;

use voxel_engine::Engine;

use crate::render_config::{
    DeviceCaps, LOD_DETAIL_RANGE, LOD_LEVELS_RANGE, RenderConfig, SessionGraphics, VrsChoice,
    engine_applied_differs, engine_applied_notice, fit_render_targets, max_lod_levels,
    vrs_effective,
};
use crate::ui::HudMode;

pub use crate::world::{VERTICAL_RADIUS_RANGE as VERTICAL_DISTANCE_RANGE, VIEW_RADIUS_RANGE};
/// Render-resolution scale clamp range — re-exported from the engine, which owns
/// the single source (it does the real clamp in `set_render_scale`). Re-exporting
/// here mirrors the [`VIEW_RADIUS_RANGE`] re-export so the settings UI and the
/// renderer can never disagree on the bound.
pub use voxel_engine::RENDER_SCALE_RANGE;

fn settings_path() -> PathBuf {
    crate::paths::Paths::get().settings_file()
}

/// Default-preset internal scale when the window is above [`AUTO_RENDER_SCALE_THRESHOLD_PX`].
/// Raise to 0.8 once the engine's temporal upsampler (Catmull-Rom TAAU, engine wt/round4+)
/// is in main and the TAA forcing below is verified in the bench path.
pub const DEFAULT_AUTO_RENDER_SCALE: f32 = 1.0;
/// Window-pixel count above which Default uses [`DEFAULT_AUTO_RENDER_SCALE`] (and TAA).
pub const AUTO_RENDER_SCALE_THRESHOLD_PX: u32 = 1_500_000;

#[cfg(test)]
thread_local! {
    static TEST_AUTO_RENDER_SCALE: std::cell::Cell<Option<f32>> = const { std::cell::Cell::new(None) };
}

/// The Auto scale Default actually applies (the shipped constant, or a test override).
fn live_auto_render_scale() -> f32 {
    #[cfg(test)]
    if let Some(scale) = TEST_AUTO_RENDER_SCALE.with(|slot| slot.get()) {
        return scale;
    }
    DEFAULT_AUTO_RENDER_SCALE
}

/// Run `f` with Default Auto scale temporarily set (proves TAA forcing while the
/// shipped constant is 1.0).
#[cfg(test)]
pub fn with_auto_render_scale<R>(scale: f32, f: impl FnOnce() -> R) -> R {
    TEST_AUTO_RENDER_SCALE.with(|slot| {
        let prev = slot.replace(Some(scale));
        let out = f();
        slot.set(prev);
        out
    })
}

/// Field-of-view clamp range, in degrees. Shared with the settings menu stepper.
pub const FOV_RANGE: RangeInclusive<f32> = 60.0..=220.0;

/// HUD/text scale clamp range (multiplier). Shared with the settings menu stepper.
pub const UI_SCALE_RANGE: RangeInclusive<f32> = 0.5..=2.0;

pub const SHAKE_RANGE: RangeInclusive<f32> = 0.0..=1.0;

crate::macros::code_enum! {
    /// Performance profile marker. Editing any profile-owned setting drops this to
    /// [`Preset::Custom`]; choosing a profile applies it atomically. Personal
    /// controls never touch it. Persisted by its stable [`Preset::code`].
    pub enum Preset {
        Custom = 0, ["custom", "0"], "Custom",
        Minimum = 1, ["minimum", "min", "1"], "Minimum",
        Fast = 2, ["fast", "2"], "Fast",
        Default = 3, ["default", "3"], "Default",
    }
}

/// Declare settings state and shipped defaults together. Descriptor behavior
/// remains in [`SETTINGS`]; runtime-only fields are deliberately declared here
/// without gaining persistence or menu behavior.
macro_rules! settings_fields {
    ($( $(#[$meta:meta])* $field:ident: $ty:ty = $default:expr ),* $(,)?) => {
        #[derive(Clone, PartialEq, Debug)]
        pub struct Settings {
            $( $(#[$meta])* pub $field: $ty, )*
        }

        impl Default for Settings {
            fn default() -> Self {
                Self { $( $field: $default, )* }
            }
        }
    };
}

settings_fields! {
    /// Performance profile marker; interactive owned-field edits make it Custom.
    preset: Preset = Preset::Default,
    fullscreen: bool = false,
    vsync: bool = false,
    /// MSAA samples (1 = off); hardware support clamps this further on apply.
    msaa: u32 = 1,
    /// Frame cap; zero is uncapped.
    max_fps: u32 = 0,
    render_distance: i32 = 6,
    fov: f32 = 90.0,
    render_scale: f32 = 1.0,
    ui_scale: f32 = 1.0,
    menu_scale: f32 = 1.0,
    shake: f32 = 1.0,
    /// Cross-chunk lighting, a world-meshing input rather than an engine flag.
    lighting: bool = true,
    /// Runtime-only `WATT_CULL=1` switch; absent from [`SETTINGS`].
    cull_faces: bool = false,

    vertical_distance: i32 = 3,
    lod_levels: u8 = 7,
    lod_detail: u8 = 2,
    /// Update rates; zero means every frame.
    stream_hz: u32 = 0,
    physics_hz: u32 = 0,
    sky_hz: u32 = 0,
    mod_hz: u32 = 0,
    simulation: bool = true,
    mod_logic: bool = true,
    autosave: bool = true,
    hud_mode: HudMode = HudMode::Full,
    minimap: bool = true,
    mod_hud: bool = true,
    player_models: bool = true,
    name_tags: bool = true,

    // Render lanes. `lod2` ships off; the golden harness opts into its own config.
    occlusion: bool = true,
    lod2: bool = false,
    blocklight: bool = false,
    exposure: bool = false,
    bloom: bool = true,
    godrays: bool = true,
    clouds: bool = true,
    weather: bool = true,
    stars: bool = true,
    day_night: bool = true,
    taa: bool = false,
    fog: bool = false,
    ambient: bool = false,
    sunlight: bool = true,
    shadows: bool = false,
    sky: bool = true,
    vrs: VrsChoice = VrsChoice::Auto,
    water_anim: bool = true,
    /// Baked corner AO, another meshing input.
    ao: bool = true,
    vignette: bool = false,

    master_volume: u8 = 80,
    effects_volume: u8 = 100,
    voice_volume: u8 = 100,
    voice_enabled: bool = true,
    voice_incoming: bool = true,
    /// Runtime-only `/mute` state; absent from [`SETTINGS`].
    muted: bool = false,

    /// Device framebuffer MSAA ceiling from the startup probe; not persisted.
    device_max_msaa: u32 = 8,
    /// Device-local heap size from the startup probe; not persisted.
    device_local_memory_bytes: Option<u64> = None,
    /// Live free device-local bytes (`VK_EXT_memory_budget`); not persisted.
    available_device_bytes: Option<u64> = None,
    /// Session-only VRAM-guard / engine-fallback line for the console and settings menu.
    vram_notice: Option<String> = None,
    /// Engine-applied MSAA after render-target fallback; not persisted.
    session_msaa: Option<u32> = None,
    /// Engine-applied render scale after render-target fallback; not persisted.
    session_render_scale: Option<f32> = None,
    /// True when the engine allocated less MSAA/scale than the session request.
    render_target_fallback: bool = false,
    /// Fitted request the engine fallback was measured against.
    fallback_request_msaa: Option<u32> = None,
    fallback_request_scale: Option<f32> = None,
    /// Largest connected display (fullscreen first allocation).
    startup_display_w: u32 = 1280,
    startup_display_h: u32 = 720,
    /// Last window size used by Default Auto render scale.
    window_w: u32 = 1280,
    window_h: u32 = 720,
    /// Last render extent (window × session scale) used by Auto VRS.
    render_w: u32 = 1280,
    render_h: u32 = 720,
}

// One `Setting` per field: behaviour folded over by every surface (persistence, menu, console).

/// Settings submenu category.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Category {
    Performance,
    Video,
    World,
    Interface,
    Audio,
}

impl Category {
    /// All categories in menu order with their page titles.
    pub const ALL: [(Category, &'static str); 5] = [
        (Category::Performance, "Performance"),
        (Category::Video, "Video"),
        (Category::World, "World"),
        (Category::Interface, "Interface"),
        (Category::Audio, "Audio"),
    ];
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MenuKind {
    Toggle,
    Choice,
    Bar,
}

/// How a setting participates in named performance profiles.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Profile {
    /// Preserve this personal/window control when selecting a profile.
    Personal,
    /// Copy the value from the selected profile.
    Owned,
    /// Copy the value and use this boolean in the stripped profile base.
    Stripped(bool),
}

/// A setting descriptor: key, label, and behavior functions that every surface
/// (persistence, menu, console) uses. Closures in [`SETTINGS`] fill the function
/// pointers; float fields touch `f32` directly via shared helpers.
pub struct Setting {
    category: Category,
    menu_kind: MenuKind,
    profile: Profile,
    fraction: fn(&Settings) -> f32,
    /// The `key=` name used in `settings.cfg` and the primary console name.
    key: &'static str,
    /// Extra names the `/gfx` console command accepts for this field.
    aliases: &'static [&'static str],
    /// The settings-menu row label.
    label: &'static str,
    /// Value syntax shown by `/gfx` help, including the preferred console key.
    usage: &'static str,
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
    /// Append the machine (persistence) text without allocating a value string.
    write: fn(&Settings, &mut String),
    /// Copy this field directly when applying a profile, without a text codec.
    copy: fn(&mut Settings, &Settings),
    /// Read a persisted value into the field (no clamp — [`Settings::clamp`] runs
    /// after the whole file is parsed). `false` if it didn't parse.
    read: fn(&mut Settings, &str) -> bool,
}

impl Setting {
    /// The settings-menu row label.
    pub fn label(&self) -> &'static str {
        self.label
    }

    /// Persistence / `/gfx` key for this field.
    pub fn key(&self) -> &'static str {
        self.key
    }

    pub fn category(&self) -> Category {
        self.category
    }

    pub fn menu_kind(&self) -> MenuKind {
        self.menu_kind
    }

    pub fn fraction(&self, s: &Settings) -> f32 {
        (self.fraction)(s)
    }

    /// Preferred console key and accepted value syntax.
    pub fn usage(&self) -> &'static str {
        self.usage
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
    /// A profile-owned value that actually changes marks the state Custom.
    pub fn step(&self, s: &mut Settings, dir: i32) {
        let before = self.owned_value(s);
        (self.step)(s, dir);
        self.note_custom(s, before);
    }

    /// Parse a `/gfx` value and clamp. Returns whether the value parsed.
    /// A profile-owned value that actually changes marks the state Custom.
    pub fn parse_human(&self, s: &mut Settings, value: &str) -> bool {
        let before = self.owned_value(s);
        let parsed = (self.parse_human)(s, value);
        if parsed {
            self.note_custom(s, before);
        }
        parsed
    }

    /// Snapshot an owned field only while an edit could clear a named profile.
    /// Comparing its persisted value preserves the codec's normalization rules.
    fn owned_value(&self, s: &Settings) -> Option<String> {
        (self.profile != Profile::Personal && s.preset != Preset::Custom).then(|| self.write(s))
    }

    /// The one place the "editing a field marks Custom" rule lives, so no UI
    /// surface has to remember it. The preset row applies its own marker;
    /// personal controls (fullscreen, FOV, UI/menu scale, shake, audio) are
    /// not profile-owned, so this is a no-op for them. Persistence `read`
    /// deliberately bypasses this — loading restores the saved marker.
    fn note_custom(&self, s: &mut Settings, before: Option<String>) {
        if let Some(before) = before
            && self.write(s) != before
        {
            s.preset = Preset::Custom;
        }
    }

    /// The exact `/gfx` confirmation line for the current value.
    pub fn confirm(&self, s: &Settings) -> String {
        (self.confirm)(s)
    }

    fn clamp(&self, s: &mut Settings) {
        (self.clamp)(s)
    }

    fn write(&self, s: &Settings) -> String {
        let mut value = String::new();
        (self.write)(s, &mut value);
        value
    }

    fn read(&self, s: &mut Settings, value: &str) -> bool {
        (self.read)(s, value)
    }
}

/// An on/off row over a single `bool` field in the given category. Every render-lane
/// toggle shares this exact behaviour set, so the field name is the only variable.
macro_rules! toggle_setting {
    ($profile:expr, $cat:expr, $field:ident, $key:literal, $label:literal, $aliases:expr) => {
        Setting {
            category: $cat,
            menu_kind: MenuKind::Toggle,
            profile: $profile,
            fraction: |_| 0.0,
            key: $key,
            aliases: $aliases,
            label: $label,
            usage: concat!($key, " on|off"),
            confirm: |s| format!(concat!($key, " {}"), on_off(s.$field, false)),
            show: |s| on_off(s.$field, true).to_string(),
            parse_human: |s, v| set_bool(&mut s.$field, v),
            step: |s, _| s.$field = !s.$field,
            clamp: |_| {},
            write: |s, text| write_value(s.$field, text),
            copy: |s, source| s.$field = source.$field,
            read: |s, v| set_bool(&mut s.$field, v),
        }
    };
}

macro_rules! video_toggle {
    ($field:ident, $key:literal, $label:literal $(, $aliases:expr)?) => {
        toggle_setting!(Profile::Stripped(false), Category::Video, $field, $key, $label, video_toggle!(@aliases $($aliases)?))
    };
    (strip $value:expr; $field:ident, $key:literal, $label:literal $(, $aliases:expr)?) => {
        toggle_setting!(Profile::Stripped($value), Category::Video, $field, $key, $label, video_toggle!(@aliases $($aliases)?))
    };
    (@aliases) => { &[] };
    (@aliases $aliases:expr) => { $aliases };
}

/// A `Category::Audio` 0..=100 percent volume slider over a `u8` field.
macro_rules! volume_bar {
    ($field:ident, $key:literal, $label:literal $(, $aliases:expr)?) => {
        Setting {
            category: Category::Audio,
            menu_kind: MenuKind::Bar,
            profile: Profile::Personal,
            fraction: |s| s.$field as f32 / 100.0,
            key: $key,
            aliases: volume_bar!(@aliases $($aliases)?),
            label: $label,
            usage: concat!($key, " <0-100>"),
            confirm: |s| format!(concat!($key, " {}%"), s.$field),
            show: |s| format!("{}%", s.$field),
            parse_human: |s, v| {
                let ok = set_parsed(&mut s.$field, v);
                if ok {
                    vol_clamp(&mut s.$field);
                }
                ok
            },
            step: |s, d| s.$field = cycle_list(&[0, 25, 50, 75, 100], s.$field as i32, d) as u8,
            clamp: |s| vol_clamp(&mut s.$field),
            write: |s, text| write_value(s.$field, text),
            copy: |s, source| s.$field = source.$field,
            read: |s, v| set_parsed(&mut s.$field, v),
        }
    };
    (@aliases) => { &[] };
    (@aliases $aliases:expr) => { $aliases };
}

/// A percent-displayed `f32` bar (stored as a raw multiplier, shown ×100):
/// the shape render/UI/menu scale and camera shake all share — fraction from
/// the range, percent show/confirm, parse-as-percent + clamp, and a stepper
/// cycling a fixed percent list. One macro instead of four ~30-line blocks.
macro_rules! percent_bar {
    ($profile:expr, $cat:expr, $field:ident, $key:literal, $label:literal, $range:expr,
     $steps:expr, $usage:literal, $confirm:literal, $aliases:expr) => {
        Setting {
            category: $cat,
            menu_kind: MenuKind::Bar,
            profile: $profile,
            fraction: |s| frac(s.$field, *$range.start(), *$range.end()),
            key: $key,
            aliases: $aliases,
            label: $label,
            usage: $usage,
            confirm: |s| format!(concat!($confirm, " {:.0}%"), s.$field * 100.0),
            show: |s| format!("{:.0}%", s.$field * 100.0),
            parse_human: |s, v| match v.parse::<f32>() {
                Ok(pct) => {
                    s.$field = pct / 100.0;
                    clamp_float(&mut s.$field, &$range, Settings::default().$field);
                    true
                }
                Err(_) => false,
            },
            step: |s, d| {
                let pct = cycle_list($steps, (s.$field * 100.0).round() as i32, d);
                s.$field = pct as f32 / 100.0;
            },
            clamp: |s| clamp_float(&mut s.$field, &$range, Settings::default().$field),
            write: |s, text| write_value(s.$field, text),
            copy: |s, source| s.$field = source.$field,
            read: |s, v| set_parsed(&mut s.$field, v),
        }
    };
}

/// A `Category::Performance` fixed-rate row over a `u32` Hz field: cycles the
/// offered rates, snaps stray persisted values down to a supported rate, and
/// shares the "0 = every frame" convention. One macro instead of four
/// hand-written descriptors, so a new throttled lane is a single row.
macro_rules! rate_setting {
    ($field:ident, $key:literal, $label:literal, $kind:literal, $rates:expr, $usage:literal, $aliases:expr) => {
        Setting {
            category: Category::Performance,
            menu_kind: MenuKind::Choice,
            profile: Profile::Owned,
            fraction: |_| 0.0,
            key: $key,
            aliases: $aliases,
            label: $label,
            usage: $usage,
            confirm: |s| rate_confirm($kind, s.$field),
            show: |s| rate_name(s.$field),
            parse_human: |s, v| parse_rate(&mut s.$field, v, $rates),
            step: |s, d| s.$field = cycle_list($rates, s.$field as i32, d) as u32,
            clamp: |s| s.$field = snap_rate($rates, s.$field.min(i32::MAX as u32) as i32) as u32,
            write: |s, text| write_value(s.$field, text),
            copy: |s, source| s.$field = source.$field,
            read: |s, v| set_parsed(&mut s.$field, v),
        }
    };
}

/// A numeric field whose human parser is its machine parser followed by a
/// clamp. The row supplies only its presentation and stepping policy; the
/// shared persistence/parse skeleton cannot drift between numeric settings.
macro_rules! numeric_setting {
    ($profile:expr, $cat:expr, $kind:expr, $field:ident, $label:literal,
     $usage:literal, $aliases:expr, $fraction:expr, $confirm:expr, $show:expr,
     $step:expr, $clamp:path) => {
        Setting {
            category: $cat,
            menu_kind: $kind,
            profile: $profile,
            fraction: $fraction,
            key: stringify!($field),
            aliases: $aliases,
            label: $label,
            usage: $usage,
            confirm: $confirm,
            show: $show,
            parse_human: |s, v| {
                let parsed = set_parsed(&mut s.$field, v);
                if parsed { $clamp(s); }
                parsed
            },
            step: $step,
            clamp: $clamp,
            write: |s, text| write_value(s.$field, text),
            copy: |s, source| s.$field = source.$field,
            read: |s, v| set_parsed(&mut s.$field, v),
        }
    };
}

/// Stable-code enum choice shared by preset and HUD mode. Interactive parsing
/// and stepping use the same ordered values; persistence restores the marker
/// directly instead of triggering its interactive side effects.
macro_rules! enum_setting {
    ($set:ident, $profile:expr, $cat:expr, $field:ident, $ty:ty, $label:literal,
     $usage:literal, $aliases:expr, $confirm:literal, $order:expr) => {
        Setting {
            category: $cat,
            menu_kind: MenuKind::Choice,
            profile: $profile,
            fraction: |_| 0.0,
            key: stringify!($field), aliases: $aliases, label: $label, usage: $usage,
            confirm: |s| format!(concat!($confirm, " {}"), s.$field.label().to_ascii_lowercase()),
            show: |s| s.$field.label().to_string(),
            parse_human: |s, v| match <$ty>::parse(v) {
                Some(value) => { enum_setting!(@set $set, s, $field, value); true }
                None => false,
            },
            step: |s, d| {
                let order = $order;
                let at = order.iter().position(|&value| value == s.$field).unwrap_or(0);
                let next = (at as i32 + d).rem_euclid(order.len() as i32) as usize;
                enum_setting!(@set $set, s, $field, order[next]);
            },
            clamp: |_| {}, write: |s, text| write_value(s.$field.code(), text),
            copy: |s, source| s.$field = source.$field,
            read: |s, v| match <$ty>::parse(v) {
                Some(value) => { s.$field = value; true }
                None => false,
            },
        }
    };
    (@set apply, $s:ident, $field:ident, $value:expr) => { $s.apply_preset($value) };
    (@set assign, $s:ident, $field:ident, $value:expr) => { $s.$field = $value };
}

/// The MSAA sample counts offered — one list shared by its stepper and its
/// "round down to a supported count" clamp bucket.
const MSAA: &[i32] = &[1, 2, 4, 8];
const UPDATE_RATES: &[i32] = &[0, 15, 30, 60, 120, 240];
const PHYSICS_RATES: &[i32] = &[0, 30, 60, 120, 240, 500, 1000];

/// Every setting, in menu/persistence order. The single source of the field set;
/// persistence, `/gfx`, the menu, and [`Settings::clamp`] all fold over it.
pub const SETTINGS: [Setting; 52] = [
    enum_setting!(
        apply, Profile::Personal, Category::Performance, preset, Preset, "Performance Preset",
        "preset custom|minimum|fast|default", &["profile"], "performance preset",
        &[Preset::Custom, Preset::Minimum, Preset::Fast, Preset::Default]
    ),
    numeric_setting!(
        Profile::Owned, Category::Performance, MenuKind::Bar, vertical_distance,
        "Vertical Distance", "vertical_distance <1-10>", &["vertical", "verticaldist"],
        |s| frac(s.vertical_distance as f32, *VERTICAL_DISTANCE_RANGE.start() as f32,
                  *VERTICAL_DISTANCE_RANGE.end() as f32),
        |s| format!("vertical distance {}", s.vertical_distance),
        |s| s.vertical_distance.to_string(),
        |s, d| s.vertical_distance = wrap_clamp(s.vertical_distance,
            *VERTICAL_DISTANCE_RANGE.start(), *VERTICAL_DISTANCE_RANGE.end(), d),
        vertical_distance_clamp
    ),
    numeric_setting!(
        Profile::Owned, Category::Performance, MenuKind::Bar, lod_levels,
        "LOD Range", "lod_levels <1-8>", &["lodlevels"],
        |s| frac(s.lod_levels as f32, *LOD_LEVELS_RANGE.start() as f32,
                  *LOD_LEVELS_RANGE.end() as f32),
        |s| format!("LOD range {} levels (~{} m)", s.lod_levels, lod_range_metres(s)),
        |s| format!("{} levels (~{} m)", s.lod_levels, lod_range_metres(s)),
        |s, d| s.lod_levels = wrap_clamp(s.lod_levels as i32, 1,
            max_lod_levels(s.lod_detail) as i32, d) as u8,
        lod_clamp
    ),
    numeric_setting!(
        Profile::Owned, Category::Performance, MenuKind::Bar, lod_detail,
        "LOD Quality", "lod_detail <2-6>", &["loddetail"],
        |s| frac(s.lod_detail as f32, *LOD_DETAIL_RANGE.start() as f32,
                  *LOD_DETAIL_RANGE.end() as f32),
        |s| format!("LOD quality {} m cells", lod_cell_metres(s.lod_detail)),
        |s| format!("{} m cells", lod_cell_metres(s.lod_detail)),
        |s, d| {
            s.lod_detail = wrap_clamp(s.lod_detail as i32, *LOD_DETAIL_RANGE.start() as i32,
                *LOD_DETAIL_RANGE.end() as i32, d) as u8;
            lod_clamp(s);
        },
        lod_clamp
    ),
    rate_setting!(
        stream_hz,
        "stream_hz",
        "Streaming Rate",
        "streaming",
        UPDATE_RATES,
        "stream_hz every|15|30|60|120|240",
        &["streamrate"]
    ),
    rate_setting!(
        physics_hz,
        "physics_hz",
        "Physics Rate",
        "physics",
        PHYSICS_RATES,
        "physics_hz every|30|60|120|240|500|1000",
        &["physicsrate"]
    ),
    rate_setting!(
        sky_hz,
        "sky_hz",
        "Sky Clock Rate",
        "sky clock",
        UPDATE_RATES,
        "sky_hz every|15|30|60|120|240",
        &["skyrate"]
    ),
    rate_setting!(
        mod_hz,
        "mod_hz",
        "Mod Update Rate",
        "mod updates",
        UPDATE_RATES,
        "mod_hz every|15|30|60|120|240",
        &["modrate"]
    ),
    enum_setting!(
        assign, Profile::Owned, Category::Performance, hud_mode, HudMode, "HUD Mode",
        "hud_mode off|minimal|full", &["hud"], "HUD",
        &[HudMode::Off, HudMode::Minimal, HudMode::Full]
    ),
    toggle_setting!(
        Profile::Owned,
        Category::Performance,
        simulation,
        "simulation",
        "Simulation",
        &["sim"]
    ),
    toggle_setting!(
        Profile::Owned,
        Category::Performance,
        mod_logic,
        "mod_logic",
        "Mod Updates",
        &["mods"]
    ),
    toggle_setting!(Profile::Owned, Category::Performance, autosave, "autosave", "Autosave", &[]),
    toggle_setting!(
        Profile::Owned,
        Category::Performance,
        minimap,
        "minimap",
        "Minimap",
        &["map"]
    ),
    toggle_setting!(
        Profile::Owned,
        Category::Performance,
        mod_hud,
        "mod_hud",
        "Mod HUD",
        &["modhud"]
    ),
    toggle_setting!(
        Profile::Owned,
        Category::Performance,
        player_models,
        "player_models",
        "Player Models",
        &["models"]
    ),
    toggle_setting!(
        Profile::Owned,
        Category::Performance,
        name_tags,
        "name_tags",
        "Name Tags",
        &["nametags"]
    ),
    toggle_setting!(Profile::Personal, Category::Video, fullscreen, "fullscreen", "Fullscreen", &[]),
    toggle_setting!(Profile::Owned, Category::Video, vsync, "vsync", "VSync", &[]),
    toggle_setting!(
        Profile::Owned,
        Category::World,
        lighting,
        "lighting",
        "Voxel Lighting",
        &["light"]
    ),
    numeric_setting!(
        Profile::Owned, Category::Video, MenuKind::Choice, msaa,
        "MSAA", "msaa 1|2|4|8", &[], |_| 0.0,
        |s| format!("msaa {}x", s.msaa), |s| format!("{}x", s.msaa),
        |s, d| s.msaa = cycle_list(MSAA, s.msaa as i32, d) as u32,
        msaa_clamp
    ),
    Setting {
        category: Category::Video,
        menu_kind: MenuKind::Bar,
        profile: Profile::Owned,
        fraction: |s| (s.max_fps as f32 / 240.0).min(1.0),
        key: "max_fps",
        aliases: &["fps"],
        label: "Max FPS",
        usage: "fps <10-1000>|off",
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
        step: |s, d| {
            s.max_fps = cycle_list(&[0, 30, 60, 120, 144, 240], s.max_fps as i32, d) as u32
        },
        clamp: fps_clamp,
        write: |s, text| write_value(s.max_fps, text),
        copy: |s, source| s.max_fps = source.max_fps,
        read: |s, v| set_parsed(&mut s.max_fps, v),
    },
    numeric_setting!(
        Profile::Owned, Category::World, MenuKind::Bar, render_distance,
        "Render Distance", "renderdist <0-20>", &["renderdist", "renderdistance"],
        |s| frac(s.render_distance as f32, *VIEW_RADIUS_RANGE.start() as f32,
                  *VIEW_RADIUS_RANGE.end() as f32),
        |s| format!("render distance {}", s.render_distance),
        |s| s.render_distance.to_string(),
        |s, d| s.render_distance = wrap_clamp(s.render_distance,
            *VIEW_RADIUS_RANGE.start(), *VIEW_RADIUS_RANGE.end(), d),
        dist_clamp
    ),
    numeric_setting!(
        Profile::Personal, Category::Interface, MenuKind::Bar, fov,
        "FOV", "fov <60-220>", &[],
        |s| frac(s.fov, *FOV_RANGE.start(), *FOV_RANGE.end()),
        |s| format!("fov {:.0}", s.fov), |s| s.fov.to_string(),
        |s, d| {
            let (lo, hi) = (*FOV_RANGE.start(), *FOV_RANGE.end());
            let v = clamp_to(&FOV_RANGE, s.fov) + d as f32 * 5.0;
            s.fov = if v > hi { lo } else if v < lo { hi } else { v };
        },
        fov_clamp
    ),
    Setting {
        category: Category::Video,
        menu_kind: MenuKind::Bar,
        profile: Profile::Owned,
        fraction: |s| frac(s.render_scale, *RENDER_SCALE_RANGE.start(), *RENDER_SCALE_RANGE.end()),
        key: "render_scale",
        aliases: &["renderscale", "scale"],
        label: "Render Scale",
        usage: "renderscale <25-200>",
        confirm: |s| format!("render scale {}", render_scale_show(s)),
        show: render_scale_show,
        parse_human: |s, v| match v.parse::<f32>() {
            Ok(pct) => {
                s.render_scale = pct / 100.0;
                clamp_float(&mut s.render_scale, &RENDER_SCALE_RANGE, Settings::default().render_scale);
                true
            }
            Err(_) => false,
        },
        step: |s, d| {
            let pct = cycle_list(&[25, 50, 75, 100, 125, 150, 200], (s.render_scale * 100.0).round() as i32, d);
            s.render_scale = pct as f32 / 100.0;
        },
        clamp: |s| clamp_float(&mut s.render_scale, &RENDER_SCALE_RANGE, Settings::default().render_scale),
        write: |s, text| write_value(s.render_scale, text),
        copy: |s, source| s.render_scale = source.render_scale,
        read: |s, v| set_parsed(&mut s.render_scale, v),
    },
    percent_bar!(
        Profile::Personal,
        Category::Interface,
        ui_scale,
        "ui_scale",
        "UI Scale",
        UI_SCALE_RANGE,
        &[50, 75, 100, 125, 150, 200],
        "uiscale <50-200>",
        "ui scale",
        &["uiscale", "hudscale"]
    ),
    percent_bar!(
        Profile::Personal,
        Category::Interface,
        menu_scale,
        "menu_scale",
        "Menu Scale",
        UI_SCALE_RANGE,
        &[50, 75, 100, 125, 150, 200],
        "menuscale <50-200>",
        "menu scale",
        &["menuscale"]
    ),
    percent_bar!(
        Profile::Personal,
        Category::Interface,
        shake,
        "shake",
        "Camera Shake",
        SHAKE_RANGE,
        &[0, 25, 50, 75, 100],
        "shake <0-100>",
        "camera shake",
        &["camerashake"]
    ),
    // Render lanes (see [`Settings::render_config`]). Engine lanes apply live via
    // `set_flags`; occlusion/lod2 apply on next world entry; clouds/weather per frame.
    video_toggle!(lod2, "lod2", "Distant LOD", &["lod"]),
    video_toggle!(occlusion, "occlusion", "Occlusion Culling", &["occ"]),
    video_toggle!(sky, "sky", "Procedural Sky"),
    video_toggle!(strip true; sunlight, "sunlight", "Sunlight", &["sun"]),
    video_toggle!(ambient, "ambient", "Ambient Light", &["amb"]),
    video_toggle!(shadows, "shadows", "Shadows", &["shadow"]),
    video_toggle!(blocklight, "blocklight", "Block Light"),
    video_toggle!(fog, "fog", "Distance Fog"),
    video_toggle!(clouds, "clouds", "Clouds"),
    video_toggle!(weather, "weather", "Weather"),
    video_toggle!(stars, "stars", "Night Stars"),
    video_toggle!(day_night, "day_night", "Day/Night Cycle", &["daynight"]),
    video_toggle!(bloom, "bloom", "Bloom"),
    video_toggle!(godrays, "godrays", "Godrays"),
    video_toggle!(exposure, "exposure", "Auto Exposure", &["exp"]),
    video_toggle!(taa, "taa", "Temporal AA", &["aa"]),
    enum_setting!(
        assign, Profile::Stripped(false), Category::Video, vrs, VrsChoice, "Variable-Rate Shading",
        "vrs auto|on|off", &[], "vrs",
        &[VrsChoice::Auto, VrsChoice::On, VrsChoice::Off]
    ),
    video_toggle!(water_anim, "water_anim", "Water Animation", &["water"]),
    video_toggle!(ao, "ao", "Ambient Occlusion", &["vertexao"]),
    video_toggle!(vignette, "vignette", "Vignette"),
    // Audio mix (see [`Settings::mix_change`]).
    volume_bar!(
        master_volume,
        "master_volume",
        "Master Volume",
        &["volume", "master"]
    ),
    volume_bar!(
        effects_volume,
        "effects_volume",
        "Effects Volume",
        &["effects", "sfx"]
    ),
    volume_bar!(voice_volume, "voice_volume", "Voice Volume", &["voice_vol"]),
    toggle_setting!(
        Profile::Personal,
        Category::Audio,
        voice_enabled,
        "voice_enabled",
        "Voice Chat",
        &["voice", "mic"]
    ),
    toggle_setting!(
        Profile::Personal,
        Category::Audio,
        voice_incoming,
        "voice_incoming",
        "Hear Voice",
        &["deafen_inverse", "hearvoice"]
    ),
];

impl Settings {
    /// Copy only settings a profile owns. Personal/window controls deliberately
    /// remain untouched; each descriptor declares that distinction itself.
    fn copy_profile_values(&mut self, profile: &Self) {
        for field in SETTINGS
            .iter()
            .filter(|field| field.profile != Profile::Personal)
        {
            (field.copy)(self, profile);
        }
    }

    /// Record that an individual setting no longer matches a named profile.
    pub fn mark_custom(&mut self) {
        self.preset = Preset::Custom;
    }

    /// Apply a named/numeric performance profile. Shared by `/gfx`, the menu,
    /// and reproducible benchmark startup (`WATT_BENCH_PRESET`).
    pub fn select_preset(&mut self, value: &str) -> bool {
        let Some(preset) = Preset::parse(value) else {
            return false;
        };
        self.apply_preset(preset);
        true
    }

    fn apply_preset(&mut self, preset: Preset) {
        if preset == Preset::Custom {
            self.mark_custom();
            return;
        }

        // Each profile is a diffable override literal over `stripped()` (the
        // default with the costly lanes off); only the perf/gameplay fields and
        // `lod2` differ, so a glance shows exactly what a profile changes. Fast
        // omits `mod_logic`, keeping the default (mods stay live).
        let profile = match preset {
            Preset::Minimum => Settings {
                vsync: false,
                msaa: 1,
                render_distance: 0,
                render_scale: 0.25,
                lighting: false,
                vertical_distance: 1,
                lod_levels: 1,
                lod_detail: 6,
                stream_hz: 15,
                physics_hz: 30,
                sky_hz: 15,
                mod_hz: 15,
                simulation: false,
                mod_logic: false,
                autosave: false,
                hud_mode: HudMode::Off,
                minimap: false,
                mod_hud: false,
                player_models: false,
                name_tags: false,
                lod2: false,
                ..Self::stripped()
            },
            Preset::Fast => Settings {
                vsync: false,
                msaa: 1,
                render_distance: 3,
                render_scale: 0.5,
                lighting: false,
                vertical_distance: 2,
                lod_levels: 3,
                lod_detail: 4,
                stream_hz: 60,
                physics_hz: 60,
                sky_hz: 60,
                mod_hz: 60,
                simulation: true,
                autosave: true,
                hud_mode: HudMode::Minimal,
                minimap: false,
                mod_hud: false,
                player_models: true,
                name_tags: false,
                lod2: true,
                ..Self::stripped()
            },
            Preset::Default => Self::default(),
            Preset::Custom => unreachable!("Custom returned early above"),
        };
        self.copy_profile_values(&profile);
        self.preset = preset;
    }

    /// Default settings with every optional presentation lane stripped — the
    /// shared base the Minimum and Fast profiles override.
    fn stripped() -> Self {
        let mut s = Self::default();
        for field in &SETTINGS {
            if let Profile::Stripped(value) = field.profile {
                field.read(&mut s, if value { "true" } else { "false" });
            }
        }
        s
    }

    /// Load from disk, falling back to defaults for missing/invalid entries.
    pub fn load() -> Self {
        let mut settings = Self::default();
        if let Ok(text) = fs::read_to_string(settings_path()) {
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
        let mut text = String::new();
        for field in &SETTINGS {
            text.push_str(field.key);
            text.push('=');
            (field.write)(self, &mut text);
            text.push('\n');
        }
        text
    }

    /// Best-effort save (a failed write shouldn't crash the game).
    pub fn save(&self) {
        let path = settings_path();
        if let Some(dir) = path.parent() {
            let _ = fs::create_dir_all(dir);
        }
        let _ = crate::save::write_atomic(&path, self.to_text().as_bytes());
    }

    /// Force every field into its valid range. Safe to call repeatedly, and
    /// handles NaN/±INF by mapping to an endpoint, so no non-finite value can
    /// reach the renderer.
    pub fn clamp(&mut self) {
        for field in &SETTINGS {
            field.clamp(self);
        }
    }

    /// Install the one-shot GPU probe. MSAA above [`Self::device_max_msaa`]
    /// is refused by [`Self::clamp`] (and persisted). VRAM over-budget
    /// degrades are session-only and never written back to [`Self::msaa`] /
    /// [`Self::render_scale`].
    pub fn set_device_caps(&mut self, caps: DeviceCaps, display: (u32, u32)) {
        self.device_max_msaa = caps.max_msaa.max(1);
        self.device_local_memory_bytes = caps.device_local_memory_bytes;
        self.available_device_bytes = caps.available_device_bytes;
        self.startup_display_w = display.0.max(1);
        self.startup_display_h = display.1.max(1);
        self.clamp();
    }

    /// Session MSAA/scale after the VRAM guard. Does not mutate persisted fields.
    pub fn session_graphics(&self, width: u32, height: u32) -> SessionGraphics {
        let mut g = self.fitted_session_graphics(width, height);
        if self.render_target_fallback
            && self.fallback_request_msaa == Some(g.msaa)
            && self
                .fallback_request_scale
                .is_some_and(|s| (s - g.render_scale).abs() <= 1e-3)
            && let (Some(msaa), Some(scale)) = (self.session_msaa, self.session_render_scale)
        {
            g.msaa = msaa;
            g.render_scale = scale;
            if let Some(notice) = self.vram_notice.clone() {
                g.notice = Some(notice);
            }
        }
        g
    }

    /// VRAM-fitted request before any engine allocation fallback.
    fn fitted_session_graphics(&self, width: u32, height: u32) -> SessionGraphics {
        let scale = self.effective_render_scale(width, height);
        let mut lanes = self.render_config();
        lanes.taa = self.effective_taa(scale);
        fit_render_targets(
            width,
            height,
            scale,
            self.msaa,
            lanes,
            DeviceCaps {
                device_local_memory_bytes: self.device_local_memory_bytes,
                available_device_bytes: self.available_device_bytes,
                max_msaa: self.device_max_msaa,
            },
        )
    }

    /// Adopt the MSAA / scale the engine actually allocated. Session-only:
    /// persisted [`Self::msaa`] / [`Self::render_scale`] are left alone.
    /// Equal values leave notice, JSON flag, and extent unchanged.
    pub fn adopt_engine_applied(
        &mut self,
        requested: &SessionGraphics,
        applied_msaa: u32,
        applied_scale: f32,
        width: u32,
        height: u32,
    ) {
        if !engine_applied_differs(requested, applied_msaa, applied_scale) {
            return;
        }
        let already = self.render_target_fallback
            && self.session_msaa == Some(applied_msaa)
            && self
                .session_render_scale
                .is_some_and(|s| (s - applied_scale).abs() <= 1e-3);
        if already {
            return;
        }
        self.session_msaa = Some(applied_msaa);
        self.session_render_scale = Some(applied_scale);
        self.render_target_fallback = true;
        self.fallback_request_msaa = Some(requested.msaa);
        self.fallback_request_scale = Some(requested.render_scale);
        let notice = engine_applied_notice(applied_msaa, applied_scale);
        eprintln!("graphics: {notice}");
        self.vram_notice = Some(notice);
        self.note_render_extent(width, height, applied_scale);
    }

    /// Read [`Engine::msaa`] / [`Engine::render_scale`] after create or recreate.
    pub fn sync_engine_applied(&mut self, eng: &Engine) {
        let w = eng.screen_width().max(1) as u32;
        let h = eng.screen_height().max(1) as u32;
        let requested = self.fitted_session_graphics(w, h);
        self.adopt_engine_applied(
            &requested,
            eng.msaa(),
            eng.render_scale(),
            w,
            h,
        );
    }

    /// Whether the Default Auto render-scale rule is live (not Custom/Minimum/Fast).
    pub fn render_scale_auto(&self) -> bool {
        self.preset == Preset::Default
    }

    /// Scale requested for this window. Default picks
    /// [`DEFAULT_AUTO_RENDER_SCALE`] above the pixel threshold and 1.0
    /// otherwise; other profiles keep their stored value. Engine allocation
    /// fallbacks live on [`Self::session_graphics`], not here.
    pub fn effective_render_scale(&self, window_w: u32, window_h: u32) -> f32 {
        if self.render_scale_auto() {
            auto_render_scale(window_w, window_h)
        } else {
            self.render_scale
        }
    }

    fn effective_taa(&self, scale: f32) -> bool {
        self.taa || (self.render_scale_auto() && scale < 1.0)
    }

    /// Push the current values to the engine. Cheap to call every frame: the
    /// engine ignores values that didn't change. Hardware MSAA support is
    /// already snapped in [`Self::clamp`]; VRAM-budget MSAA/scale cuts are
    /// applied here without writing them back (they are this session only).
    pub fn apply(&mut self, eng: &mut Engine) {
        eng.set_fullscreen(self.fullscreen);
        // Vsync and the fps cap are not pushed here: `App::frame` is the
        // single writer, because the effective values also depend on the
        // screen (menus cap the frame rate, vsync off) and the benchmark.
        let w = eng.screen_width().max(1) as u32;
        let h = eng.screen_height().max(1) as u32;
        let fitted = self.fitted_session_graphics(w, h);
        if self.render_target_fallback
            && (self.fallback_request_msaa != Some(fitted.msaa)
                || !self
                    .fallback_request_scale
                    .is_some_and(|s| (s - fitted.render_scale).abs() <= 1e-3))
        {
            self.session_msaa = None;
            self.session_render_scale = None;
            self.render_target_fallback = false;
            self.fallback_request_msaa = None;
            self.fallback_request_scale = None;
        }
        let session = self.session_graphics(w, h);
        if !self.render_target_fallback {
            self.adopt_vram_notice(session.notice.clone());
        }
        let _ = eng.set_msaa(session.msaa);
        let _ = eng.set_render_scale(session.render_scale);
        self.note_render_extent(w, h, session.render_scale);
        eng.set_cull_faces(self.cull_faces);
        // Engine render lanes live-swap on both threads; occlusion/lod2 are world
        // inputs (applied on world entry) and aren't part of `engine_flags`.
        eng.set_flags(self.render_config().engine_flags());
    }

    /// Record the live window and render extent so [`render_config`] can resolve
    /// Auto VRS and Default Auto render scale.
    pub fn note_render_extent(&mut self, window_w: u32, window_h: u32, scale: f32) {
        let scale = scale.max(0.0);
        self.window_w = window_w.max(1);
        self.window_h = window_h.max(1);
        self.render_w = ((window_w as f32 * scale) as u32).max(1);
        self.render_h = ((window_h as f32 * scale) as u32).max(1);
    }

    fn adopt_vram_notice(&mut self, notice: Option<String>) {
        if self.vram_notice != notice {
            if let Some(line) = notice.as_ref() {
                eprintln!("{line}");
            }
            self.vram_notice = notice;
        }
    }

    /// The render lanes this settings state names, before visual-mod masking.
    /// World construction, `/gfx` apply, and engine flags go through
    /// [`Mods::effective_render`](crate::mods::Mods::effective_render). (The
    /// golden harness keeps its own pinned
    /// [`RenderConfig::golden`](crate::render_config::RenderConfig::golden).)
    pub fn render_config(&self) -> RenderConfig {
        let scale = self
            .session_render_scale
            .filter(|_| self.render_target_fallback)
            .unwrap_or_else(|| self.effective_render_scale(self.window_w, self.window_h));
        RenderConfig {
            occlusion: self.occlusion,
            lod2: self.lod2,
            lod_levels: self.lod_levels,
            lod_detail: self.lod_detail,
            blocklight: self.blocklight,
            exposure: self.exposure,
            bloom: self.bloom,
            godrays: self.godrays,
            clouds: self.clouds,
            weather: self.weather,
            stars: self.stars,
            day_night: self.day_night,
            taa: self.effective_taa(scale),
            fog: self.fog,
            ambient: self.ambient,
            sunlight: self.sunlight,
            shadows: self.shadows,
            sky: self.sky,
            vrs: vrs_effective(self.vrs, self.render_w, self.render_h),
            water_anim: self.water_anim,
            vignette: self.vignette,
        }
    }

    /// The audio mix this settings state names — the single source the spine
    /// pushes to `SoundSystem::set_mix`. Percents map to linear [0, 1] gains;
    /// `muted` is the transient runtime `/mute` flag (never persisted);
    /// `deafen` is the inverse of the persisted `voice_incoming` gate.
    pub fn mix_change(&self) -> crate::audio::MixChange {
        crate::audio::MixChange {
            master: self.master_volume as f32 / 100.0,
            effects: self.effects_volume as f32 / 100.0,
            voice: self.voice_volume as f32 / 100.0,
            muted: self.muted,
            deafen: !self.voice_incoming,
        }
    }
}

// Shared value helpers — the single definition each surface reuses.

/// Default Auto scale for a window pixel count. One comparison so the menu,
/// session apply, and tests cannot disagree.
pub fn auto_render_scale(window_w: u32, window_h: u32) -> f32 {
    let px = (window_w as u64).saturating_mul(window_h as u64);
    if px > u64::from(AUTO_RENDER_SCALE_THRESHOLD_PX) {
        live_auto_render_scale()
    } else {
        1.0
    }
}

fn render_scale_show(s: &Settings) -> String {
    if s.render_scale_auto() {
        format!(
            "Auto ({:.1})",
            s.effective_render_scale(s.window_w, s.window_h)
        )
    } else {
        format!("{:.0}%", s.render_scale * 100.0)
    }
}

fn write_value(value: impl std::fmt::Display, text: &mut String) {
    write!(text, "{value}").expect("writing settings to a String cannot fail");
}

fn rate_name(rate: u32) -> String {
    if rate == 0 {
        "Every frame".to_string()
    } else {
        format!("{rate} Hz")
    }
}

fn rate_confirm(kind: &str, rate: u32) -> String {
    if rate == 0 {
        format!("{kind} every frame")
    } else {
        format!("{kind} {rate} Hz")
    }
}

fn parse_rate(dst: &mut u32, value: &str, choices: &[i32]) -> bool {
    let parsed = match value {
        "every" | "frame" | "0" => 0,
        _ => match value.parse::<u32>() {
            Ok(rate) => rate,
            Err(_) => return false,
        },
    };
    *dst = snap_rate(choices, parsed.min(i32::MAX as u32) as i32) as u32;
    true
}

fn vertical_distance_clamp(s: &mut Settings) {
    s.vertical_distance = s.vertical_distance.clamp(
        *VERTICAL_DISTANCE_RANGE.start(),
        *VERTICAL_DISTANCE_RANGE.end(),
    );
}

fn lod_clamp(s: &mut Settings) {
    s.lod_detail = s
        .lod_detail
        .clamp(*LOD_DETAIL_RANGE.start(), *LOD_DETAIL_RANGE.end());
    s.lod_levels = s
        .lod_levels
        .clamp(*LOD_LEVELS_RANGE.start(), max_lod_levels(s.lod_detail));
}

/// The approximate far-field outer range in metres for the confirm/show text.
fn lod_range_metres(s: &Settings) -> u64 {
    let radius = s
        .render_distance
        .clamp(*VIEW_RADIUS_RANGE.start(), *VIEW_RADIUS_RANGE.end())
        .max(1) as u64;
    let levels = s
        .lod_levels
        .clamp(*LOD_LEVELS_RANGE.start(), *LOD_LEVELS_RANGE.end());
    radius * 16 * (1_u64 << levels)
}

fn lod_cell_metres(detail: u8) -> u32 {
    1_u32 << detail.clamp(*LOD_DETAIL_RANGE.start(), *LOD_DETAIL_RANGE.end())
}

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

/// Normalize to 0..=1 for menu bar display.
fn frac(v: f32, lo: f32, hi: f32) -> f32 {
    if hi <= lo {
        0.0
    } else {
        ((v - lo) / (hi - lo)).clamp(0.0, 1.0)
    }
}

/// Clamp a finite value to a range. Callers reset NaN before using it.
fn clamp_to(range: &RangeInclusive<f32>, v: f32) -> f32 {
    v.clamp(*range.start(), *range.end())
}

/// Clamp one float field, resetting NaN because `f32::clamp` preserves it.
fn clamp_float(value: &mut f32, range: &RangeInclusive<f32>, default: f32) {
    *value = clamp_to(range, if value.is_nan() { default } else { *value });
}

fn fov_clamp(s: &mut Settings) {
    clamp_float(&mut s.fov, &FOV_RANGE, Settings::default().fov);
}

fn fps_clamp(s: &mut Settings) {
    if s.max_fps != 0 {
        s.max_fps = s.max_fps.clamp(10, 1000);
    }
}

/// Clamp a volume percent into 0..=100 (u8 already excludes negatives/overflow).
fn vol_clamp(v: &mut u8) {
    *v = (*v).min(100);
}

fn msaa_clamp(s: &mut Settings) {
    s.msaa = snap_down(MSAA, s.msaa as i32) as u32;
    if s.msaa > s.device_max_msaa {
        s.msaa = snap_down(MSAA, s.device_max_msaa as i32) as u32;
    }
}

fn dist_clamp(s: &mut Settings) {
    s.render_distance = s
        .render_distance
        .clamp(*VIEW_RADIUS_RANGE.start(), *VIEW_RADIUS_RANGE.end());
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
    list.iter()
        .rev()
        .copied()
        .find(|&e| e <= v)
        .unwrap_or(list[0])
}

/// Rate zero is the explicit every-frame mode. A malformed positive value must
/// never clamp to zero (which would increase work); it snaps to the nearest
/// supported rate at or below it, with the minimum positive rate as the floor.
fn snap_rate(list: &[i32], v: i32) -> i32 {
    if v == 0 {
        return 0;
    }
    list.iter()
        .rev()
        .copied()
        .find(|&e| e > 0 && e <= v)
        .or_else(|| list.iter().copied().find(|&e| e > 0))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setting(key: &str) -> &'static Setting {
        SETTINGS
            .iter()
            .find(|field| field.matches(key))
            .unwrap_or_else(|| panic!("missing setting descriptor `{key}`"))
    }

    #[test]
    fn roundtrip_through_text() {
        let s = Settings {
            fullscreen: true,
            msaa: 4,
            max_fps: 144,
            render_distance: 8,
            fov: 90.0,
            render_scale: 0.75,
            ..Settings::default()
        };
        // Same table-driven serialization as `save`, so this can't drift from
        // what `parse_from` reads.
        let text = s.to_text();
        let mut loaded = Settings::default();
        loaded.parse_from(&text);
        loaded.clamp();
        assert_eq!(loaded, s);
    }

    #[test]
    fn save_and_load_use_the_config_root() {
        let path = settings_path();
        assert!(path.starts_with(&crate::paths::Paths::get().config));
        assert_ne!(path, PathBuf::from("saves/settings.cfg"));
        let s = Settings { fov: 110.0, ..Default::default() };
        s.save();
        assert!(path.exists());
        let loaded = Settings::load();
        assert_eq!(loaded.fov, 110.0);
        let _ = fs::remove_file(path);
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
            shake: 5.0,
            lighting: true,
            cull_faces: false,
            // Render lanes aren't under test here; take them as-shipped so adding a
            // lane can't break this clamp test.
            ..Settings::default()
        };
        s.clamp();
        assert_eq!(s.msaa, 4);
        assert_eq!(s.max_fps, 10);
        assert_eq!(s.render_distance, 20);
        assert_eq!(s.fov, 220.0);
        assert_eq!(s.render_scale, 2.0);
        assert_eq!(s.shake, 1.0);
    }

    #[test]
    fn clamp_refuses_msaa_above_device_and_scale_above_two() {
        let mut s = Settings {
            msaa: 8,
            device_max_msaa: 4,
            render_scale: 9.0,
            ..Settings::default()
        };
        s.clamp();
        assert_eq!(s.msaa, 4);
        assert_eq!(s.render_scale, 2.0);
    }

    #[test]
    fn nan_resets_to_default_and_infinity_clamps() {
        // `f32::clamp` alone would let NaN sail through to the renderer. NaN is a
        // corrupt/garbage sentinel, so it resets to the field default; ±INF is a
        // genuine over/underflow and clamps to the near endpoint.
        let defaults = Settings::default();
        let mut s = Settings {
            fov: f32::NAN,
            render_scale: f32::NAN,
            ..Settings::default()
        };
        s.clamp();
        assert_eq!(s.fov, defaults.fov);
        assert_eq!(s.render_scale, defaults.render_scale);

        let mut s = Settings {
            fov: f32::INFINITY,
            render_scale: f32::NEG_INFINITY,
            ..Settings::default()
        };
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
            assert_eq!(
                snap_down(&[1, 2, 4, 8], v),
                old_bucket(v as u32) as i32,
                "v={v}"
            );
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
            shake: 0.5,
            lighting: false,
            // Not persisted (env-only); must stay at the default so the composed
            // roundtrip below — which never writes it — still lands `samples`. The
            // render lanes likewise stay at their persisted defaults via the spread.
            cull_faces: false,
            // Fields added since this fixture was written: defaults roundtrip
            // trivially, so the spread above stays the interesting part.
            ..Settings::default()
        };
        for field in &SETTINGS {
            let mut back = Settings::default();
            assert!(
                field.read(&mut back, &field.write(&samples)),
                "{}",
                field.label()
            );
        }
        // The composed roundtrip lands the exact struct.
        let mut back = Settings::default();
        back.parse_from(&samples.to_text());
        back.clamp();
        assert_eq!(back, samples);
    }

    #[test]
    fn audio_fields_roundtrip_and_clamp() {
        // Non-default audio mix survives the text codec, and an out-of-range
        // volume snaps to 100 on clamp (u8 already excludes negatives).
        let s = Settings {
            master_volume: 45,
            effects_volume: 0,
            voice_volume: 75,
            voice_enabled: false,
            voice_incoming: false,
            ..Settings::default()
        };
        let mut back = Settings::default();
        back.parse_from(&s.to_text());
        back.clamp();
        assert_eq!(back, s);

        let mut over = Settings {
            master_volume: 200,
            ..Settings::default()
        };
        over.clamp();
        assert_eq!(over.master_volume, 100);

        // Percent maps to linear gain; deafen is the inverse of voice_incoming.
        let mix = Settings {
            voice_incoming: false,
            master_volume: 50,
            ..Settings::default()
        }
        .mix_change();
        assert_eq!(mix.master, 0.5);
        assert!(mix.deafen);
        assert!(!mix.muted);
    }

    #[test]
    fn positive_rates_never_snap_to_every_frame() {
        assert_eq!(snap_rate(UPDATE_RATES, 0), 0);
        assert_eq!(snap_rate(UPDATE_RATES, 1), 15);
        assert_eq!(snap_rate(UPDATE_RATES, 14), 15);
        assert_eq!(snap_rate(UPDATE_RATES, 29), 15);
        assert_eq!(snap_rate(UPDATE_RATES, 59), 30);
        assert_eq!(snap_rate(PHYSICS_RATES, 999), 500);
    }

    #[test]
    fn presets_apply_owned_fields_and_preserve_personal_controls() {
        let preset = setting("preset");
        let mut s = Settings {
            fullscreen: true,
            max_fps: 777,
            fov: 105.0,
            ui_scale: 1.5,
            menu_scale: 0.75,
            shake: 0.25,
            cull_faces: true,
            ..Settings::default()
        };

        assert!(preset.parse_human(&mut s, "minimum"));
        assert_eq!(s.preset, Preset::Minimum);
        assert_eq!(s.max_fps, 0, "a performance preset must remove an old cap");
        assert_eq!(s.render_scale, 0.25);
        assert_eq!((s.render_distance, s.vertical_distance), (0, 1));
        assert!(!s.lod2);
        assert_eq!((s.lod_levels, s.lod_detail), (1, 6));
        assert_eq!(
            lod_range_metres(&s),
            32,
            "zero near radius keeps one LOD unit"
        );
        assert_eq!(
            (s.stream_hz, s.physics_hz, s.sky_hz, s.mod_hz),
            (15, 30, 15, 15)
        );
        assert_eq!(s.hud_mode, HudMode::Off);
        assert!(!s.simulation && !s.mod_logic && !s.autosave);
        assert!(!s.minimap && !s.mod_hud && !s.player_models && !s.name_tags);
        assert!(!s.lighting && !s.occlusion && !s.ao);
        assert_eq!(s.vrs, VrsChoice::Off);
        assert!(!s.sky && !s.bloom && !s.clouds && !s.water_anim);
        assert!(s.sunlight);

        assert!(preset.parse_human(&mut s, "fast"));
        assert_eq!(s.preset, Preset::Fast);
        assert_eq!(s.max_fps, 0);
        assert_eq!(s.render_scale, 0.5);
        assert_eq!((s.render_distance, s.vertical_distance), (3, 2));
        assert!(s.lod2);
        assert_eq!((s.lod_levels, s.lod_detail), (3, 4));
        assert_eq!(lod_range_metres(&s), 384);
        assert_eq!(
            (s.stream_hz, s.physics_hz, s.sky_hz, s.mod_hz),
            (60, 60, 60, 60)
        );
        assert_eq!(s.hud_mode, HudMode::Minimal);
        assert!(s.simulation && s.mod_logic && s.autosave && s.player_models);
        assert!(!s.minimap && !s.mod_hud && !s.name_tags);
        assert_eq!(s.vrs, VrsChoice::Off);

        assert!(preset.parse_human(&mut s, "default"));
        let expected = Settings {
            fullscreen: true,
            fov: 105.0,
            ui_scale: 1.5,
            menu_scale: 0.75,
            shake: 0.25,
            cull_faces: true,
            ..Settings::default()
        };
        assert_eq!(s, expected);
    }

    #[test]
    fn lod_clamp_enforces_combined_detail_limit() {
        let mut s = Settings {
            lod_detail: 6,
            lod_levels: 8,
            ..Settings::default()
        };
        s.clamp();
        assert_eq!((s.lod_detail, s.lod_levels), (6, 4));
        assert!(
            s.lod_detail + s.lod_levels - 1 <= 9,
            "coarsest level within the ladder cap"
        );

        s.lod_detail = 255;
        s.lod_levels = 0;
        s.clamp();
        assert_eq!((s.lod_detail, s.lod_levels), (6, 1));

        let detail = setting("lod_detail");
        let levels = setting("lod_levels");
        s.lod_detail = 5;
        s.lod_levels = 5;
        detail.step(&mut s, 1);
        assert_eq!((s.lod_detail, s.lod_levels), (6, 4));

        s.render_distance = 3;
        s.lod_levels = 3;
        s.lod_detail = 4;
        assert_eq!(levels.show(&s), "3 levels (~384 m)");
        assert_eq!(detail.show(&s), "16 m cells");
    }

    #[test]
    fn interactive_edits_mark_custom_but_persistence_does_not() {
        let preset = setting("preset");
        let scale = setting("render_scale");

        let mut s = Settings::default();
        assert!(preset.parse_human(&mut s, "fast"));
        assert_eq!(s.preset, Preset::Fast);
        assert!(scale.parse_human(&mut s, "75"));
        assert_eq!(s.preset, Preset::Custom);

        assert!(preset.parse_human(&mut s, "fast"));
        assert!(!scale.parse_human(&mut s, "not-a-number"));
        assert_eq!(s.preset, Preset::Fast);
        scale.step(&mut s, 1);
        assert_eq!(s.preset, Preset::Custom);

        // Personal controls are not profile-owned: editing them keeps the profile.
        let fov = setting("fov");
        assert!(preset.parse_human(&mut s, "fast"));
        fov.step(&mut s, 1);
        assert_eq!(s.preset, Preset::Fast, "FOV is a personal control");

        let mut loaded = Settings::default();
        loaded.parse_from("preset=2\nrender_scale=0.75\nautosave=false\n");
        loaded.clamp();
        assert_eq!(loaded.preset, Preset::Fast);
        assert_eq!(loaded.render_scale, 0.75);
        assert!(!loaded.autosave);

        loaded.mark_custom();
        assert_eq!(loaded.preset, Preset::Custom);
    }

    #[test]
    fn normalized_and_invalid_edits_preserve_named_profiles() {
        let mut s = Settings::default();
        s.apply_preset(Preset::Minimum);

        // Different input text can still clamp to the current field value.
        for (key, value) in [
            ("render_scale", "25.000"),
            ("render_distance", "-1"),
            ("stream_hz", "14"),
            ("msaa", "0"),
        ] {
            assert!(setting(key).parse_human(&mut s, value));
            assert_eq!(s.preset, Preset::Minimum, "{key}");
        }

        let scale = setting("render_scale");
        scale.step(&mut s, 0);
        assert_eq!(s.preset, Preset::Minimum);
        assert!(!scale.parse_human(&mut s, "invalid"));
        assert_eq!(s.preset, Preset::Minimum);

        assert!(scale.parse_human(&mut s, "50"));
        assert_eq!(s.preset, Preset::Custom);
        assert!(scale.parse_human(&mut s, "75"));
        assert_eq!(s.render_scale, 0.75);
        assert_eq!(s.preset, Preset::Custom);
    }

    #[test]
    fn console_aliases_resolve_and_caps_differ() {
        // Aliases reach the same field...
        let mut a = Settings::default();
        let mut b = Settings::default();
        let dist = setting("renderdist");
        let dist2 = setting("renderdistance");
        assert!(dist.parse_human(&mut a, "8"));
        assert!(dist2.parse_human(&mut b, "8"));
        assert_eq!(a.render_distance, 8);
        assert_eq!(b.render_distance, 8);
        // ...and the one on/off helper gives menu caps vs console lowercase.
        assert_eq!(on_off(true, true), "On");
        assert_eq!(on_off(true, false), "on");
    }

    #[test]
    fn vrs_persists_choice_words_and_reads_legacy_bools() {
        let field = setting("vrs");
        let mut s = Settings::default();
        assert_eq!(s.vrs, VrsChoice::Auto);
        assert_eq!(field.write(&s), "auto");
        assert_eq!(field.show(&s), "Auto");
        assert_eq!(field.usage(), "vrs auto|on|off");
        assert_eq!(field.confirm(&s), "vrs auto");

        assert!(field.parse_human(&mut s, "on"));
        assert_eq!(s.vrs, VrsChoice::On);
        assert_eq!(s.preset, Preset::Custom);
        assert_eq!(field.write(&s), "on");
        assert!(field.parse_human(&mut s, "off"));
        assert_eq!(s.vrs, VrsChoice::Off);
        assert!(field.parse_human(&mut s, "auto"));
        assert_eq!(s.vrs, VrsChoice::Auto);

        let mut loaded = Settings::default();
        loaded.parse_from("vrs=true\n");
        assert_eq!(loaded.vrs, VrsChoice::On);
        loaded.parse_from("vrs=false\n");
        assert_eq!(loaded.vrs, VrsChoice::Off);
        loaded.parse_from("vrs=auto\n");
        assert_eq!(loaded.vrs, VrsChoice::Auto);
        assert!(
            loaded.to_text().lines().any(|line| line == "vrs=auto"),
            "new files persist the word form"
        );

        let mut round = Settings::default();
        round.parse_from(&loaded.to_text());
        assert_eq!(round.vrs, VrsChoice::Auto);
    }

    #[test]
    fn render_config_resolves_auto_vrs_from_extent() {
        let mut s = Settings::default();
        assert_eq!(s.vrs, VrsChoice::Auto);
        assert!(!s.render_config().vrs, "1280×720 default is below Auto");
        s.note_render_extent(3840, 2160, 1.0);
        assert!(s.render_config().vrs);
        s.note_render_extent(3440, 1440, 2.0);
        assert!(s.render_config().vrs);
        s.note_render_extent(1920, 1080, 1.0);
        assert!(!s.render_config().vrs);
        s.vrs = VrsChoice::On;
        assert!(s.render_config().vrs);
        s.vrs = VrsChoice::Off;
        s.note_render_extent(3840, 2160, 1.0);
        assert!(!s.render_config().vrs);
    }

    #[test]
    fn default_auto_render_scale_follows_window_pixels() {
        let field = setting("render_scale");
        let mut s = Settings::default();
        assert_eq!(s.preset, Preset::Default);
        assert!(s.render_scale_auto());
        assert_eq!(s.render_scale, 1.0, "stored Default scale stays 1.0");

        s.note_render_extent(1280, 720, 1.0);
        assert_eq!(s.effective_render_scale(1280, 720), 1.0);
        assert_eq!(s.session_graphics(1280, 720).render_scale, 1.0);
        assert!(!s.render_config().taa, "720p leaves TAA at the stored value");
        assert_eq!(field.show(&s), "Auto (1.0)");
        assert_eq!(field.confirm(&s), "render scale Auto (1.0)");

        s.note_render_extent(1920, 1080, 1.0);
        assert_eq!(s.effective_render_scale(1920, 1080), DEFAULT_AUTO_RENDER_SCALE);
        assert_eq!(s.session_graphics(1920, 1080).render_scale, DEFAULT_AUTO_RENDER_SCALE);
        assert_eq!(
            s.render_config().taa,
            DEFAULT_AUTO_RENDER_SCALE < 1.0,
            "TAA is forced only while Auto scale is below 1"
        );
        assert_eq!(
            field.show(&s),
            format!("Auto ({:.1})", DEFAULT_AUTO_RENDER_SCALE)
        );

        s.note_render_extent(3440, 1440, 1.0);
        assert_eq!(s.effective_render_scale(3440, 1440), DEFAULT_AUTO_RENDER_SCALE);
        assert_eq!(s.render_config().taa, DEFAULT_AUTO_RENDER_SCALE < 1.0);

        // Crossing the threshold via the live extent path (resize / fullscreen).
        s.note_render_extent(1280, 720, 1.0);
        assert_eq!(s.effective_render_scale(s.window_w, s.window_h), 1.0);
        assert!(!s.render_config().taa);
        s.note_render_extent(1920, 1080, 1.0);
        assert_eq!(s.effective_render_scale(s.window_w, s.window_h), DEFAULT_AUTO_RENDER_SCALE);
        assert_eq!(s.render_config().taa, DEFAULT_AUTO_RENDER_SCALE < 1.0);

        let mut custom = Settings::default();
        custom.mark_custom();
        custom.render_scale = 1.0;
        custom.note_render_extent(1920, 1080, 1.0);
        assert!(!custom.render_scale_auto());
        assert_eq!(custom.effective_render_scale(1920, 1080), 1.0);
        assert_eq!(custom.session_graphics(1920, 1080).render_scale, 1.0);
        assert!(!custom.render_config().taa);
        assert_eq!(field.show(&custom), "100%");

        let mut fast = Settings::default();
        fast.apply_preset(Preset::Fast);
        fast.note_render_extent(1920, 1080, 0.5);
        assert_eq!(fast.effective_render_scale(1920, 1080), 0.5);
        assert!(!fast.render_config().taa);

        let mut min = Settings::default();
        min.apply_preset(Preset::Minimum);
        min.note_render_extent(1920, 1080, 0.25);
        assert_eq!(min.effective_render_scale(1920, 1080), 0.25);
    }

    #[test]
    fn session_graphics_fits_live_available_budget() {
        use crate::render_config::{VRAM_AVAILABLE_SAFETY_FRACTION, render_target_bytes};
        let mut s = Settings::default();
        s.mark_custom();
        s.msaa = 8;
        s.render_scale = 2.0;
        s.taa = true;
        s.bloom = true;
        s.exposure = true;
        s.set_device_caps(
            DeviceCaps {
                device_local_memory_bytes: Some(8_000_000_000),
                available_device_bytes: Some(2_000_000_000),
                max_msaa: 8,
            },
            (3440, 1440),
        );
        let g = s.session_graphics(3440, 1440);
        let cost = render_target_bytes(3440, 1440, g.render_scale, g.msaa, s.render_config());
        assert!(
            cost <= 2_000_000_000 * VRAM_AVAILABLE_SAFETY_FRACTION / 100,
            "session cost {cost} at {}x / {}",
            g.msaa,
            g.render_scale
        );
        let n = g.notice.expect("over-budget request prints a session notice");
        assert!(
            n.contains("2.0 GB of 8.0 GB is free (other processes hold 6.0 GB)"),
            "{n}"
        );
        assert!(n.contains("running at"), "{n}");
    }

    #[test]
    fn engine_applied_lower_updates_notice_json_flag_and_extent() {
        let mut s = Settings::default();
        s.mark_custom();
        s.msaa = 8;
        s.render_scale = 1.5;
        s.note_render_extent(1920, 1080, 1.5);
        let requested = s.session_graphics(1920, 1080);
        assert_eq!(requested.msaa, 8);
        assert!((requested.render_scale - 1.5).abs() < 1e-4);
        s.adopt_engine_applied(&requested, 2, 1.5, 1920, 1080);
        assert_eq!(
            s.vram_notice.as_deref(),
            Some("the renderer could only allocate 2x MSAA at 150% scale this session")
        );
        assert!(s.render_target_fallback);
        assert_eq!(s.session_graphics(1920, 1080).msaa, 2);
        assert_eq!(s.render_w, ((1920.0 * 1.5) as u32).max(1));
        assert_eq!(s.render_h, ((1080.0 * 1.5) as u32).max(1));
        assert_eq!(s.msaa, 8, "persisted MSAA is unchanged");
        assert!((s.render_scale - 1.5).abs() < 1e-4, "persisted scale is unchanged");
        let g = s.session_graphics(1920, 1080);
        assert_eq!(g.msaa, 2);
        assert!((g.render_scale - 1.5).abs() < 1e-4);
    }

    #[test]
    fn engine_applied_equal_leaves_session_untouched() {
        let mut s = Settings::default();
        s.mark_custom();
        s.msaa = 4;
        s.render_scale = 1.0;
        s.note_render_extent(1280, 720, 1.0);
        let before = s.clone();
        let requested = s.session_graphics(1280, 720);
        s.adopt_engine_applied(&requested, requested.msaa, requested.render_scale, 1280, 720);
        assert_eq!(s, before);
        assert!(!s.render_target_fallback);
        assert!(s.vram_notice.is_none());
    }
}
