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

use crate::render_config::{LOD_DETAIL_RANGE, LOD_LEVELS_RANGE, RenderConfig, max_lod_levels};
use crate::ui::HudMode;

pub use crate::world::{VERTICAL_RADIUS_RANGE as VERTICAL_DISTANCE_RANGE, VIEW_RADIUS_RANGE};
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

pub const SHAKE_RANGE: RangeInclusive<f32> = 0.0..=1.0;

/// Performance profile marker. Editing any profile-owned setting drops this to
/// [`Preset::Custom`]; choosing a profile applies it atomically. Personal
/// controls never touch it. Persisted by its stable [`Preset::code`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Preset {
    Custom,
    Minimum,
    Fast,
    Default,
}

impl Preset {
    /// Stable persistence/console code (`Custom=0, Minimum=1, Fast=2, Default=3`).
    pub fn code(self) -> u8 {
        match self {
            Preset::Custom => 0,
            Preset::Minimum => 1,
            Preset::Fast => 2,
            Preset::Default => 3,
        }
    }

    /// Parse a persisted code or a console word; the single source `/gfx`,
    /// persistence `read`, and benchmark startup fold through.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "custom" | "0" => Some(Preset::Custom),
            "minimum" | "min" | "1" => Some(Preset::Minimum),
            "fast" | "2" => Some(Preset::Fast),
            "default" | "3" => Some(Preset::Default),
            _ => None,
        }
    }

    /// Capitalized display name for the menu row and confirm line.
    pub fn label(self) -> &'static str {
        match self {
            Preset::Custom => "Custom",
            Preset::Minimum => "Minimum",
            Preset::Fast => "Fast",
            Preset::Default => "Default",
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Settings {
    /// Performance profile marker (see [`Preset`] and [`Setting::step`] /
    /// [`Setting::parse_human`]).
    pub preset: Preset,
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
    /// Menu text base scale (0.5..=2.0); the theme still shrinks rows to fit
    /// the window, so this can never push settings off screen.
    pub menu_scale: f32,
    /// Camera shake intensity (0..=1); an accessibility control, not a constant.
    pub shake: f32,
    /// Cross-chunk lighting. On by default; pushed to [`crate::world::World`] on
    /// world entry and on `/gfx` change (the engine has no say — it is a meshing
    /// input, not a GPU state).
    pub lighting: bool,
    /// Six-way back-face culling of chunk meshes. Not a menu/persisted setting:
    /// sourced once from `WATT_CULL=1` (a GPU-side trade only worth it when
    /// vertex-fetch bound), so it is absent from [`SETTINGS`] and pushed to the
    /// engine by [`apply`](Settings::apply) like the table fields.
    pub cull_faces: bool,

    // Performance controls — cost levers independent of the look lanes below.
    /// Number of chunks streamed above and below the camera, decoupled from
    /// the horizontal ring radius so tall invisible columns aren't loaded.
    pub vertical_distance: i32,
    /// Number of far-field LOD levels, including the nearest configured level.
    pub lod_levels: u8,
    /// Detail exponent of the nearest far-field LOD level (`2^detail` metres).
    pub lod_detail: u8,
    /// World streaming update rate; zero updates every frame.
    pub stream_hz: u32,
    /// Fixed physics update rate; zero updates every frame.
    pub physics_hz: u32,
    /// Day/night clock update rate; zero updates every frame.
    pub sky_hz: u32,
    /// Gameplay-mod update rate; zero updates every frame. Input edges are
    /// retained until the next permitted mod tick.
    pub mod_hz: u32,
    /// Run world simulation.
    pub simulation: bool,
    /// Run gameplay-mod update hooks at `mod_hz`. Mining and core movement
    /// remain available when disabled, but inventory/crafting UI logic stays
    /// dormant.
    pub mod_logic: bool,
    /// Periodically persist world edits while playing.
    pub autosave: bool,
    /// HUD visibility. Persisted by its stable [`HudMode::code`].
    pub hud_mode: HudMode,
    pub minimap: bool,
    pub mod_hud: bool,
    pub player_models: bool,
    pub name_tags: bool,

    // Render lanes — the source of truth for [`RenderConfig`] (built by
    // [`render_config`](Settings::render_config)). The engine lanes go live via
    // `set_flags` in [`apply`](Settings::apply); `occlusion`/`lod2` are world
    // construction inputs and take effect on the next world entry; `clouds`/
    // `weather` are per-frame look lanes read through the game's cached config.
    pub occlusion: bool,
    pub lod2: bool,
    pub blocklight: bool,
    pub exposure: bool,
    pub bloom: bool,
    pub godrays: bool,
    pub clouds: bool,
    pub weather: bool,
    /// Night starfield in the sky pass (engine `RenderFlags::stars`).
    pub stars: bool,
    /// Day/night cycle: off freezes the sky clock (permanent current time of
    /// day — a strip-down lever and an accessibility control, not a look lane).
    pub day_night: bool,
    pub taa: bool,
    pub fog: bool,
    pub ambient: bool,
    pub sunlight: bool,
    pub shadows: bool,
    pub sky: bool,
    pub vrs: bool,
    pub water_anim: bool,
    /// Baked corner ambient occlusion in the mesher — a MESHING input like
    /// `lighting` (toggling remeshes the world). Off also merges more quads,
    /// so it doubles as a perf lever.
    pub ao: bool,
    pub vignette: bool,

    // Audio mix (0..=100 percent). The single source for [`mix_change`], which the
    // spine pushes to `SoundSystem::set_mix`. `voice_enabled` is the transmit gate
    // (does PTT capture do anything); `voice_incoming` is the inverse of deafen.
    pub master_volume: u8,
    pub effects_volume: u8,
    pub voice_volume: u8,
    pub voice_enabled: bool,
    pub voice_incoming: bool,
    /// Transient master mute (the `/mute` command). Absent from [`SETTINGS`] so it
    /// is never persisted and resets to `false` each launch — like [`cull_faces`],
    /// a runtime-only field on the same struct so it can ride [`mix_change`] to the
    /// mixer without a separate plumbing path.
    ///
    /// [`cull_faces`]: Settings::cull_faces
    pub muted: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            preset: Preset::Default,
            fullscreen: false,
            vsync: false,
            msaa: 1,
            max_fps: 0,
            render_distance: 6,
            fov: 90.0,
            render_scale: 1.0,
            ui_scale: 1.0,
            menu_scale: 1.0,
            shake: 1.0,
            lighting: true,
            cull_faces: false,
            vertical_distance: 3,
            lod_levels: 7,
            lod_detail: 2,
            stream_hz: 0,
            physics_hz: 0,
            sky_hz: 0,
            mod_hz: 0,
            simulation: true,
            mod_logic: true,
            autosave: true,
            hud_mode: HudMode::Full,
            minimap: true,
            mod_hud: true,
            player_models: true,
            name_tags: true,
            // Render lanes: the shipped defaults. `lod2` (the far field) ships off —
            // near-only by default; the harness keeps it on via `RenderConfig::golden`.
            occlusion: true,
            lod2: false,
            blocklight: false,
            exposure: false,
            bloom: true,
            godrays: true,
            clouds: true,
            weather: true,
            stars: true,
            day_night: true,
            taa: false,
            fog: false,
            ambient: false,
            sunlight: true,
            shadows: false,
            sky: true,
            vrs: true,
            water_anim: true,
            ao: true,
            vignette: false,
            master_volume: 80,
            effects_volume: 100,
            voice_volume: 100,
            voice_enabled: true,
            voice_incoming: true,
            muted: false,
        }
    }
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

/// A setting descriptor: key, label, and behavior functions that every surface
/// (persistence, menu, console) uses. Closures in [`SETTINGS`] fill the function
/// pointers; float fields touch `f32` directly via shared helpers.
pub struct Setting {
    category: Category,
    menu_kind: MenuKind,
    fraction: fn(&Settings) -> f32,
    /// The `key=` name used in `saves/settings.cfg` and the primary console name.
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

    /// This field's serialized value, but only when it is profile-owned —
    /// `None` for personal controls. Scopes the change check to the one field
    /// instead of the whole-`Settings` clone + compare it used to run per edit.
    fn owned_value(&self, s: &Settings) -> Option<String> {
        Settings::PROFILE_OWNED_KEYS
            .contains(&self.key)
            .then(|| (self.write)(s))
    }

    /// The one place the "editing a field marks Custom" rule lives, so no UI
    /// surface has to remember it. The preset row applies its own marker;
    /// personal controls (fullscreen, FOV, UI/menu scale, shake, audio) are
    /// not profile-owned, so this is a no-op for them. Persistence `read`
    /// deliberately bypasses this — loading restores the saved marker.
    fn note_custom(&self, s: &mut Settings, before: Option<String>) {
        if let Some(before) = before
            && (self.write)(s) != before
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
        (self.write)(s)
    }

    fn read(&self, s: &mut Settings, value: &str) -> bool {
        (self.read)(s, value)
    }
}

/// An on/off row over a single `bool` field in the given category. Every render-lane
/// toggle shares this exact behaviour set, so the field name is the only variable.
macro_rules! toggle_setting {
    ($cat:expr, $field:ident, $key:literal, $label:literal, $aliases:expr) => {
        Setting {
            category: $cat,
            menu_kind: MenuKind::Toggle,
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
            write: |s| s.$field.to_string(),
            read: |s, v| set_bool(&mut s.$field, v),
        }
    };
}

macro_rules! video_toggle {
    ($field:ident, $key:literal, $label:literal $(, $aliases:expr)?) => {
        toggle_setting!(Category::Video, $field, $key, $label, video_toggle!(@aliases $($aliases)?))
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
            write: |s| s.$field.to_string(),
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
    ($cat:expr, $field:ident, $key:literal, $label:literal, $range:expr, $clamp:path,
     $steps:expr, $usage:literal, $confirm:literal, $aliases:expr) => {
        Setting {
            category: $cat,
            menu_kind: MenuKind::Bar,
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
                    $clamp(s);
                    true
                }
                Err(_) => false,
            },
            step: |s, d| {
                let pct = cycle_list($steps, (s.$field * 100.0).round() as i32, d);
                s.$field = pct as f32 / 100.0;
            },
            clamp: $clamp,
            write: |s| s.$field.to_string(),
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
            write: |s| s.$field.to_string(),
            read: |s, v| set_parsed(&mut s.$field, v),
        }
    };
}

/// The MSAA sample counts offered — one list shared by its stepper and its
/// "round down to a supported count" clamp bucket.
const MSAA: &[i32] = &[1, 2, 4, 8];
const STREAM_RATES: &[i32] = &[0, 15, 30, 60, 120, 240];
const PHYSICS_RATES: &[i32] = &[0, 30, 60, 120, 240, 500, 1000];
const SKY_RATES: &[i32] = &[0, 15, 30, 60, 120, 240];
const MOD_RATES: &[i32] = &[0, 15, 30, 60, 120, 240];

/// Every setting, in menu/persistence order. The single source of the field set;
/// persistence, `/gfx`, the menu, and [`Settings::clamp`] all fold over it.
pub const SETTINGS: [Setting; 52] = [
    Setting {
        category: Category::Performance,
        menu_kind: MenuKind::Choice,
        fraction: |_| 0.0,
        key: "preset",
        aliases: &["profile"],
        label: "Performance Preset",
        usage: "preset custom|minimum|fast|default",
        confirm: |s| {
            format!(
                "performance preset {}",
                s.preset.label().to_ascii_lowercase()
            )
        },
        show: |s| s.preset.label().to_string(),
        parse_human: Settings::select_preset,
        step: |s, d| {
            // Cycle by persisted code order (Custom→Minimum→Fast→Default).
            let order = [
                Preset::Custom,
                Preset::Minimum,
                Preset::Fast,
                Preset::Default,
            ];
            let code = cycle_list(&[0, 1, 2, 3], s.preset.code() as i32, d);
            s.apply_preset(order[code as usize]);
        },
        clamp: |_| {},
        write: |s| s.preset.code().to_string(),
        // Loading restores the saved marker and every saved field independently;
        // it must not reapply a profile or turn later lines into Custom.
        read: |s, v| match Preset::parse(v) {
            Some(p) => {
                s.preset = p;
                true
            }
            None => false,
        },
    },
    Setting {
        category: Category::Performance,
        menu_kind: MenuKind::Bar,
        fraction: |s| {
            frac(
                s.vertical_distance as f32,
                *VERTICAL_DISTANCE_RANGE.start() as f32,
                *VERTICAL_DISTANCE_RANGE.end() as f32,
            )
        },
        key: "vertical_distance",
        aliases: &["vertical", "verticaldist"],
        label: "Vertical Distance",
        usage: "vertical_distance <1-10>",
        confirm: |s| format!("vertical distance {}", s.vertical_distance),
        show: |s| s.vertical_distance.to_string(),
        parse_human: |s, v| {
            let parsed = set_parsed(&mut s.vertical_distance, v);
            if parsed {
                vertical_distance_clamp(s);
            }
            parsed
        },
        step: |s, d| {
            s.vertical_distance = wrap_clamp(
                s.vertical_distance,
                *VERTICAL_DISTANCE_RANGE.start(),
                *VERTICAL_DISTANCE_RANGE.end(),
                d,
            );
        },
        clamp: vertical_distance_clamp,
        write: |s| s.vertical_distance.to_string(),
        read: |s, v| set_parsed(&mut s.vertical_distance, v),
    },
    Setting {
        category: Category::Performance,
        menu_kind: MenuKind::Bar,
        fraction: |s| {
            frac(
                s.lod_levels as f32,
                *LOD_LEVELS_RANGE.start() as f32,
                *LOD_LEVELS_RANGE.end() as f32,
            )
        },
        key: "lod_levels",
        aliases: &["lodlevels"],
        label: "LOD Range",
        usage: "lod_levels <1-8>",
        confirm: |s| {
            format!(
                "LOD range {} levels (~{} m)",
                s.lod_levels,
                lod_range_metres(s)
            )
        },
        show: |s| format!("{} levels (~{} m)", s.lod_levels, lod_range_metres(s)),
        parse_human: |s, v| {
            let parsed = set_parsed(&mut s.lod_levels, v);
            if parsed {
                lod_clamp(s);
            }
            parsed
        },
        step: |s, d| {
            let max = max_lod_levels(s.lod_detail);
            s.lod_levels = wrap_clamp(s.lod_levels as i32, 1, max as i32, d) as u8;
        },
        clamp: lod_clamp,
        write: |s| s.lod_levels.to_string(),
        read: |s, v| set_parsed(&mut s.lod_levels, v),
    },
    Setting {
        category: Category::Performance,
        menu_kind: MenuKind::Bar,
        fraction: |s| {
            frac(
                s.lod_detail as f32,
                *LOD_DETAIL_RANGE.start() as f32,
                *LOD_DETAIL_RANGE.end() as f32,
            )
        },
        key: "lod_detail",
        aliases: &["loddetail"],
        label: "LOD Quality",
        usage: "lod_detail <2-6>",
        confirm: |s| format!("LOD quality {} m cells", lod_cell_metres(s.lod_detail)),
        show: |s| format!("{} m cells", lod_cell_metres(s.lod_detail)),
        parse_human: |s, v| {
            let parsed = set_parsed(&mut s.lod_detail, v);
            if parsed {
                lod_clamp(s);
            }
            parsed
        },
        step: |s, d| {
            s.lod_detail = wrap_clamp(
                s.lod_detail as i32,
                *LOD_DETAIL_RANGE.start() as i32,
                *LOD_DETAIL_RANGE.end() as i32,
                d,
            ) as u8;
            lod_clamp(s);
        },
        clamp: lod_clamp,
        write: |s| s.lod_detail.to_string(),
        read: |s, v| set_parsed(&mut s.lod_detail, v),
    },
    rate_setting!(
        stream_hz,
        "stream_hz",
        "Streaming Rate",
        "streaming",
        STREAM_RATES,
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
        SKY_RATES,
        "sky_hz every|15|30|60|120|240",
        &["skyrate"]
    ),
    rate_setting!(
        mod_hz,
        "mod_hz",
        "Mod Update Rate",
        "mod updates",
        MOD_RATES,
        "mod_hz every|15|30|60|120|240",
        &["modrate"]
    ),
    Setting {
        category: Category::Performance,
        menu_kind: MenuKind::Choice,
        fraction: |_| 0.0,
        key: "hud_mode",
        aliases: &["hud"],
        label: "HUD Mode",
        usage: "hud_mode off|minimal|full",
        confirm: |s| format!("HUD {}", s.hud_mode.label().to_ascii_lowercase()),
        show: |s| s.hud_mode.label().to_string(),
        parse_human: |s, v| match HudMode::parse(v) {
            Some(mode) => {
                s.hud_mode = mode;
                true
            }
            None => false,
        },
        step: |s, d| {
            // Cycle by the persisted code order (Off→Minimal→Full), unchanged
            // from the pre-enum `[0,1,2]` stepper.
            let modes = [HudMode::Off, HudMode::Minimal, HudMode::Full];
            let code = cycle_list(&[0, 1, 2], s.hud_mode.code() as i32, d);
            s.hud_mode = modes[code as usize];
        },
        clamp: |_| {},
        write: |s| s.hud_mode.code().to_string(),
        read: |s, v| match HudMode::parse(v) {
            Some(mode) => {
                s.hud_mode = mode;
                true
            }
            None => false,
        },
    },
    toggle_setting!(
        Category::Performance,
        simulation,
        "simulation",
        "Simulation",
        &["sim"]
    ),
    toggle_setting!(
        Category::Performance,
        mod_logic,
        "mod_logic",
        "Mod Updates",
        &["mods"]
    ),
    toggle_setting!(Category::Performance, autosave, "autosave", "Autosave", &[]),
    toggle_setting!(
        Category::Performance,
        minimap,
        "minimap",
        "Minimap",
        &["map"]
    ),
    toggle_setting!(
        Category::Performance,
        mod_hud,
        "mod_hud",
        "Mod HUD",
        &["modhud"]
    ),
    toggle_setting!(
        Category::Performance,
        player_models,
        "player_models",
        "Player Models",
        &["models"]
    ),
    toggle_setting!(
        Category::Performance,
        name_tags,
        "name_tags",
        "Name Tags",
        &["nametags"]
    ),
    toggle_setting!(Category::Video, fullscreen, "fullscreen", "Fullscreen", &[]),
    toggle_setting!(Category::Video, vsync, "vsync", "VSync", &[]),
    toggle_setting!(
        Category::World,
        lighting,
        "lighting",
        "Voxel Lighting",
        &["light"]
    ),
    Setting {
        category: Category::Video,
        menu_kind: MenuKind::Choice,
        fraction: |_| 0.0,
        key: "msaa",
        aliases: &[],
        label: "MSAA",
        usage: "msaa 1|2|4|8",
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
        category: Category::Video,
        menu_kind: MenuKind::Bar,
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
        write: |s| s.max_fps.to_string(),
        read: |s, v| set_parsed(&mut s.max_fps, v),
    },
    Setting {
        category: Category::World,
        menu_kind: MenuKind::Bar,
        fraction: |s| {
            frac(
                s.render_distance as f32,
                *VIEW_RADIUS_RANGE.start() as f32,
                *VIEW_RADIUS_RANGE.end() as f32,
            )
        },
        key: "render_distance",
        aliases: &["renderdist", "renderdistance"],
        label: "Render Distance",
        usage: "renderdist <0-20>",
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
        category: Category::Interface,
        menu_kind: MenuKind::Bar,
        fraction: |s| frac(s.fov, *FOV_RANGE.start(), *FOV_RANGE.end()),
        key: "fov",
        aliases: &[],
        label: "FOV",
        usage: "fov <60-220>",
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
    percent_bar!(
        Category::Video,
        render_scale,
        "render_scale",
        "Render Scale",
        RENDER_SCALE_RANGE,
        scale_clamp,
        &[25, 50, 75, 100, 125, 150, 200],
        "renderscale <25-200>",
        "render scale",
        &["renderscale", "scale"]
    ),
    percent_bar!(
        Category::Interface,
        ui_scale,
        "ui_scale",
        "UI Scale",
        UI_SCALE_RANGE,
        ui_scale_clamp,
        &[50, 75, 100, 125, 150, 200],
        "uiscale <50-200>",
        "ui scale",
        &["uiscale", "hudscale"]
    ),
    percent_bar!(
        Category::Interface,
        menu_scale,
        "menu_scale",
        "Menu Scale",
        UI_SCALE_RANGE,
        menu_scale_clamp,
        &[50, 75, 100, 125, 150, 200],
        "menuscale <50-200>",
        "menu scale",
        &["menuscale"]
    ),
    percent_bar!(
        Category::Interface,
        shake,
        "shake",
        "Camera Shake",
        SHAKE_RANGE,
        shake_clamp,
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
    video_toggle!(sunlight, "sunlight", "Sunlight", &["sun"]),
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
    video_toggle!(vrs, "vrs", "Variable-Rate Shading"),
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
        Category::Audio,
        voice_enabled,
        "voice_enabled",
        "Voice Chat",
        &["voice", "mic"]
    ),
    toggle_setting!(
        Category::Audio,
        voice_incoming,
        "voice_incoming",
        "Hear Voice",
        &["deafen_inverse", "hearvoice"]
    ),
];

/// The fields a named performance profile owns, as one declaration: generates
/// both [`Settings::copy_profile_values`] (what selecting a profile writes)
/// and [`Settings::PROFILE_OWNED_KEYS`] (what [`Setting::note_custom`] watches
/// for individual edits). Every listed field's descriptor `key` equals its
/// field name, which the key list relies on. Personal/window controls
/// (fullscreen, FOV, UI/menu scale, shake, cull, audio) are deliberately
/// absent: profiles preserve them.
macro_rules! profile_owned {
    ($($field:ident),* $(,)?) => {
        impl Settings {
            /// The persisted keys whose individual edit makes the state Custom.
            const PROFILE_OWNED_KEYS: &'static [&'static str] = &[$(stringify!($field)),*];

            /// Copy only settings a profile owns. Personal/window controls
            /// deliberately remain untouched when a profile is selected.
            fn copy_profile_values(&mut self, p: &Self) {
                $(self.$field = p.$field;)*
            }
        }
    };
}

profile_owned!(
    vsync,
    msaa,
    max_fps,
    render_distance,
    render_scale,
    lighting,
    vertical_distance,
    lod_levels,
    lod_detail,
    stream_hz,
    physics_hz,
    sky_hz,
    mod_hz,
    simulation,
    mod_logic,
    autosave,
    hud_mode,
    minimap,
    mod_hud,
    player_models,
    name_tags,
    occlusion,
    lod2,
    blocklight,
    exposure,
    bloom,
    godrays,
    clouds,
    weather,
    stars,
    day_night,
    taa,
    fog,
    ambient,
    sunlight,
    shadows,
    sky,
    vrs,
    water_anim,
    ao,
    vignette,
);

impl Settings {
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
        disable_costly_lanes(&mut s);
        s
    }

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
        // Engine render lanes live-swap on both threads; occlusion/lod2 are world
        // inputs (applied on world entry) and aren't part of `engine_flags`.
        eng.set_flags(self.render_config().engine_flags());
    }

    /// The render lanes this settings state names — the single source the game's
    /// world construction and per-frame [`compose`](crate::frame_snapshot::compose)
    /// both derive from. (The golden harness keeps its own pinned
    /// [`RenderConfig::golden`](crate::render_config::RenderConfig::golden).)
    pub fn render_config(&self) -> RenderConfig {
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
            taa: self.taa,
            fog: self.fog,
            ambient: self.ambient,
            sunlight: self.sunlight,
            shadows: self.shadows,
            sky: self.sky,
            vrs: self.vrs,
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

/// The "costly lanes off" set the Minimum and Fast profiles share: every
/// optional presentation lane a stripped profile removes. Sunlight stays on so
/// stripped terrain remains readable.
fn disable_costly_lanes(s: &mut Settings) {
    s.occlusion = false;
    s.lod2 = false;
    s.blocklight = false;
    s.exposure = false;
    s.bloom = false;
    s.godrays = false;
    s.clouds = false;
    s.weather = false;
    s.stars = false;
    s.day_night = false;
    s.taa = false;
    s.fog = false;
    s.ambient = false;
    s.sunlight = true;
    s.shadows = false;
    s.sky = false;
    s.vrs = false;
    s.water_anim = false;
    s.ao = false;
    s.vignette = false;
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

fn menu_scale_clamp(s: &mut Settings) {
    s.menu_scale = clamp_to(
        &UI_SCALE_RANGE,
        reset_nan(s.menu_scale, Settings::default().menu_scale),
    );
}

fn shake_clamp(s: &mut Settings) {
    s.shake = clamp_to(&SHAKE_RANGE, reset_nan(s.shake, Settings::default().shake));
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
        assert_eq!(snap_rate(STREAM_RATES, 0), 0);
        assert_eq!(snap_rate(STREAM_RATES, 1), 15);
        assert_eq!(snap_rate(STREAM_RATES, 14), 15);
        assert_eq!(snap_rate(STREAM_RATES, 29), 15);
        assert_eq!(snap_rate(STREAM_RATES, 59), 30);
        assert_eq!(snap_rate(PHYSICS_RATES, 999), 500);
        assert_eq!(snap_rate(MOD_RATES, 1), 15);
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
        assert!(!s.lighting && !s.occlusion && !s.ao && !s.vrs);
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
}
