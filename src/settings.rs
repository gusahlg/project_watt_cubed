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

use crate::render_config::RenderConfig;
pub use crate::render_config::{LOD_DETAIL_RANGE, LOD_LEVELS_RANGE};

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

const MAX_LOD_DETAIL: u8 = 9;

pub const PRESET_CUSTOM: u8 = 0;
pub const PRESET_MINIMUM: u8 = 1;
pub const PRESET_FAST: u8 = 2;
pub const PRESET_DEFAULT: u8 = 3;

pub const HUD_OFF: u8 = 0;
pub const HUD_MINIMAL: u8 = 1;
pub const HUD_FULL: u8 = 2;

#[derive(Clone, PartialEq, Debug)]
pub struct Settings {
    /// Performance profile marker. Editing any individual setting changes this
    /// to [`PRESET_CUSTOM`]; choosing another profile applies it atomically.
    pub preset: u8,
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

    /// Number of chunks streamed above and below the camera.
    pub vertical_distance: i32,
    /// Number of far-field LOD levels, including the nearest configured level.
    pub lod_levels: u8,
    /// Detail exponent of the nearest far-field LOD level.
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
    /// Run gameplay-mod update hooks at `mod_hz`. Mining and core movement remain
    /// available when disabled, but inventory/crafting UI logic stays dormant.
    pub mod_logic: bool,
    /// Periodically persist world edits while playing.
    pub autosave: bool,
    /// HUD visibility: [`HUD_OFF`], [`HUD_MINIMAL`], or [`HUD_FULL`].
    pub hud_mode: u8,
    pub minimap: bool,
    pub mod_hud: bool,
    pub player_models: bool,
    pub name_tags: bool,

    // Render lanes — the source of truth for [`RenderConfig`] (built by
    // [`render_config`](Settings::render_config)). The engine lanes go live via
    // `set_flags` in [`apply`](Settings::apply); `occlusion`/`lod2` transition
    // live through the world apply path; `clouds`/`weather` are per-frame look
    // lanes read through the game's cached config.
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
    /// Day/night visual cycle: off renders fixed noon while preserving the
    /// authoritative clock for networking and future re-enables.
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
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            preset: PRESET_DEFAULT,
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
            hud_mode: HUD_FULL,
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
        }
    }
}

// ---------------------------------------------------------------------------
// One `Setting` per field: a flat set of behaviour `fn`s folded over by every
// surface. No common wire type — each field touches its own struct member.
// ---------------------------------------------------------------------------

/// Settings submenu category.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Category {
    Performance,
    Video,
    World,
    Interface,
}

impl Category {
    /// All categories in menu order with their page titles.
    pub const ALL: [(Category, &'static str); 4] = [
        (Category::Performance, "Performance"),
        (Category::Video, "Video"),
        (Category::World, "World"),
        (Category::Interface, "Interface"),
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
    pub fn step(&self, s: &mut Settings, dir: i32) {
        (self.step)(s, dir);
        if self.key != "preset" {
            s.mark_custom();
        }
    }

    /// Parse a `/gfx` value and clamp. Returns whether the value parsed.
    pub fn parse_human(&self, s: &mut Settings, value: &str) -> bool {
        let parsed = (self.parse_human)(s, value);
        if parsed && self.key != "preset" {
            s.mark_custom();
        }
        parsed
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

/// A `Category::Video` on/off row over a single `bool` field. Every render-lane
/// toggle shares this exact behaviour set, so the field name is the only variable.
macro_rules! video_toggle {
    ($field:ident, $key:literal, $label:literal $(, $aliases:expr)?) => {
        Setting {
            category: Category::Video,
            menu_kind: MenuKind::Toggle,
            fraction: |_| 0.0,
            key: $key,
            aliases: video_toggle!(@aliases $($aliases)?),
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
    (@aliases) => { &[] };
    (@aliases $aliases:expr) => { $aliases };
}

macro_rules! performance_toggle {
    ($field:ident, $key:literal, $label:literal $(, $aliases:expr)?) => {
        Setting {
            category: Category::Performance,
            menu_kind: MenuKind::Toggle,
            fraction: |_| 0.0,
            key: $key,
            aliases: performance_toggle!(@aliases $($aliases)?),
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
    (@aliases) => { &[] };
    (@aliases $aliases:expr) => { $aliases };
}

/// The MSAA sample counts offered — one list shared by its stepper and its
/// "round down to a supported count" clamp bucket.
const MSAA: &[i32] = &[1, 2, 4, 8];
const PRESETS: &[i32] = &[
    PRESET_CUSTOM as i32,
    PRESET_MINIMUM as i32,
    PRESET_FAST as i32,
    PRESET_DEFAULT as i32,
];
const STREAM_RATES: &[i32] = &[0, 15, 30, 60, 120, 240];
const PHYSICS_RATES: &[i32] = &[0, 30, 60, 120, 240, 500, 1000];
const SKY_RATES: &[i32] = &[0, 15, 30, 60, 120, 240];
const MOD_RATES: &[i32] = &[0, 15, 30, 60, 120, 240];

/// Every setting, in menu/persistence order. The single source of the field set;
/// persistence, `/gfx`, the menu, and [`Settings::clamp`] all fold over it.
pub const SETTINGS: [Setting; 47] = [
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
                preset_name(s.preset).to_ascii_lowercase()
            )
        },
        show: |s| preset_name(s.preset).to_string(),
        parse_human: Settings::select_preset,
        step: |s, d| {
            let preset = cycle_list(PRESETS, s.preset as i32, d) as u8;
            s.apply_preset(preset);
        },
        clamp: preset_clamp,
        write: |s| s.preset.to_string(),
        // Loading restores the saved marker and every saved field independently;
        // it must not reapply a profile or turn later lines into Custom.
        read: |s, v| set_parsed(&mut s.preset, v),
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
    Setting {
        category: Category::Performance,
        menu_kind: MenuKind::Choice,
        fraction: |_| 0.0,
        key: "stream_hz",
        aliases: &["streamrate"],
        label: "Streaming Rate",
        usage: "stream_hz every|15|30|60|120|240",
        confirm: |s| rate_confirm("streaming", s.stream_hz),
        show: |s| rate_name(s.stream_hz),
        parse_human: |s, v| parse_rate(&mut s.stream_hz, v, STREAM_RATES),
        step: |s, d| s.stream_hz = cycle_list(STREAM_RATES, s.stream_hz as i32, d) as u32,
        clamp: stream_rate_clamp,
        write: |s| s.stream_hz.to_string(),
        read: |s, v| set_parsed(&mut s.stream_hz, v),
    },
    Setting {
        category: Category::Performance,
        menu_kind: MenuKind::Choice,
        fraction: |_| 0.0,
        key: "physics_hz",
        aliases: &["physicsrate"],
        label: "Physics Rate",
        usage: "physics_hz every|30|60|120|240|500|1000",
        confirm: |s| rate_confirm("physics", s.physics_hz),
        show: |s| rate_name(s.physics_hz),
        parse_human: |s, v| parse_rate(&mut s.physics_hz, v, PHYSICS_RATES),
        step: |s, d| {
            s.physics_hz = cycle_list(PHYSICS_RATES, s.physics_hz as i32, d) as u32;
        },
        clamp: physics_rate_clamp,
        write: |s| s.physics_hz.to_string(),
        read: |s, v| set_parsed(&mut s.physics_hz, v),
    },
    Setting {
        category: Category::Performance,
        menu_kind: MenuKind::Choice,
        fraction: |_| 0.0,
        key: "sky_hz",
        aliases: &["skyrate"],
        label: "Sky Clock Rate",
        usage: "sky_hz every|15|30|60|120|240",
        confirm: |s| rate_confirm("sky clock", s.sky_hz),
        show: |s| rate_name(s.sky_hz),
        parse_human: |s, v| parse_rate(&mut s.sky_hz, v, SKY_RATES),
        step: |s, d| s.sky_hz = cycle_list(SKY_RATES, s.sky_hz as i32, d) as u32,
        clamp: sky_rate_clamp,
        write: |s| s.sky_hz.to_string(),
        read: |s, v| set_parsed(&mut s.sky_hz, v),
    },
    Setting {
        category: Category::Performance,
        menu_kind: MenuKind::Choice,
        fraction: |_| 0.0,
        key: "mod_hz",
        aliases: &["modrate"],
        label: "Mod Update Rate",
        usage: "mod_hz every|15|30|60|120|240",
        confirm: |s| rate_confirm("mod updates", s.mod_hz),
        show: |s| rate_name(s.mod_hz),
        parse_human: |s, v| parse_rate(&mut s.mod_hz, v, MOD_RATES),
        step: |s, d| s.mod_hz = cycle_list(MOD_RATES, s.mod_hz as i32, d) as u32,
        clamp: mod_rate_clamp,
        write: |s| s.mod_hz.to_string(),
        read: |s, v| set_parsed(&mut s.mod_hz, v),
    },
    Setting {
        category: Category::Performance,
        menu_kind: MenuKind::Choice,
        fraction: |_| 0.0,
        key: "hud_mode",
        aliases: &["hud"],
        label: "HUD Mode",
        usage: "hud_mode off|minimal|full",
        confirm: |s| format!("HUD {}", hud_name(s.hud_mode).to_ascii_lowercase()),
        show: |s| hud_name(s.hud_mode).to_string(),
        parse_human: |s, v| match parse_hud_mode(v) {
            Some(mode) => {
                s.hud_mode = mode;
                true
            }
            None => false,
        },
        step: |s, d| s.hud_mode = cycle_list(&[0, 1, 2], s.hud_mode as i32, d) as u8,
        clamp: hud_mode_clamp,
        write: |s| s.hud_mode.to_string(),
        read: |s, v| set_parsed(&mut s.hud_mode, v),
    },
    performance_toggle!(simulation, "simulation", "Simulation", &["sim"]),
    performance_toggle!(mod_logic, "mod_logic", "Mod Updates", &["mods"]),
    performance_toggle!(autosave, "autosave", "Autosave"),
    performance_toggle!(minimap, "minimap", "Minimap", &["map"]),
    performance_toggle!(mod_hud, "mod_hud", "Mod HUD", &["modhud"]),
    performance_toggle!(player_models, "player_models", "Player Models", &["models"]),
    performance_toggle!(name_tags, "name_tags", "Name Tags", &["nametags"]),
    Setting {
        category: Category::Video,
        menu_kind: MenuKind::Toggle,
        fraction: |_| 0.0,
        key: "fullscreen",
        aliases: &[],
        label: "Fullscreen",
        usage: "fullscreen on|off",
        confirm: |s| format!("fullscreen {}", on_off(s.fullscreen, false)),
        show: |s| on_off(s.fullscreen, true).to_string(),
        parse_human: |s, v| set_bool(&mut s.fullscreen, v),
        step: |s, _| s.fullscreen = !s.fullscreen,
        clamp: |_| {},
        write: |s| s.fullscreen.to_string(),
        read: |s, v| set_bool(&mut s.fullscreen, v),
    },
    Setting {
        category: Category::Video,
        menu_kind: MenuKind::Toggle,
        fraction: |_| 0.0,
        key: "vsync",
        aliases: &[],
        label: "VSync",
        usage: "vsync on|off",
        confirm: |s| format!("vsync {}", on_off(s.vsync, false)),
        show: |s| on_off(s.vsync, true).to_string(),
        parse_human: |s, v| set_bool(&mut s.vsync, v),
        step: |s, _| s.vsync = !s.vsync,
        clamp: |_| {},
        write: |s| s.vsync.to_string(),
        read: |s, v| set_bool(&mut s.vsync, v),
    },
    Setting {
        category: Category::World,
        menu_kind: MenuKind::Toggle,
        fraction: |_| 0.0,
        key: "lighting",
        aliases: &["light"],
        label: "Voxel Lighting",
        usage: "lighting on|off",
        confirm: |s| format!("lighting {}", on_off(s.lighting, false)),
        show: |s| on_off(s.lighting, true).to_string(),
        parse_human: |s, v| set_bool(&mut s.lighting, v),
        step: |s, _| s.lighting = !s.lighting,
        clamp: |_| {},
        write: |s| s.lighting.to_string(),
        read: |s, v| set_bool(&mut s.lighting, v),
    },
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
    Setting {
        category: Category::Video,
        menu_kind: MenuKind::Bar,
        fraction: |s| {
            frac(
                s.render_scale,
                *RENDER_SCALE_RANGE.start(),
                *RENDER_SCALE_RANGE.end(),
            )
        },
        key: "render_scale",
        aliases: &["renderscale", "scale"],
        label: "Render Scale",
        usage: "renderscale <25-200>",
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
        category: Category::Interface,
        menu_kind: MenuKind::Bar,
        fraction: |s| frac(s.ui_scale, *UI_SCALE_RANGE.start(), *UI_SCALE_RANGE.end()),
        key: "ui_scale",
        aliases: &["uiscale", "hudscale"],
        label: "UI Scale",
        usage: "uiscale <50-200>",
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
    Setting {
        category: Category::Interface,
        menu_kind: MenuKind::Bar,
        fraction: |s| frac(s.menu_scale, *UI_SCALE_RANGE.start(), *UI_SCALE_RANGE.end()),
        key: "menu_scale",
        aliases: &["menuscale"],
        label: "Menu Scale",
        usage: "menuscale <50-200>",
        confirm: |s| format!("menu scale {:.0}%", s.menu_scale * 100.0),
        show: |s| format!("{:.0}%", s.menu_scale * 100.0),
        parse_human: |s, v| match v.parse::<f32>() {
            Ok(pct) => {
                s.menu_scale = pct / 100.0;
                menu_scale_clamp(s);
                true
            }
            Err(_) => false,
        },
        step: |s, d| {
            let pct = cycle_list(
                &[50, 75, 100, 125, 150, 200],
                (s.menu_scale * 100.0).round() as i32,
                d,
            );
            s.menu_scale = pct as f32 / 100.0;
        },
        clamp: menu_scale_clamp,
        write: |s| s.menu_scale.to_string(),
        read: |s, v| set_parsed(&mut s.menu_scale, v),
    },
    Setting {
        category: Category::Interface,
        menu_kind: MenuKind::Bar,
        fraction: |s| frac(s.shake, *SHAKE_RANGE.start(), *SHAKE_RANGE.end()),
        key: "shake",
        aliases: &["camerashake"],
        label: "Camera Shake",
        usage: "shake <0-100>",
        confirm: |s| format!("camera shake {:.0}%", s.shake * 100.0),
        show: |s| format!("{:.0}%", s.shake * 100.0),
        parse_human: |s, v| match v.parse::<f32>() {
            Ok(pct) => {
                s.shake = pct / 100.0;
                shake_clamp(s);
                true
            }
            Err(_) => false,
        },
        step: |s, d| {
            let pct = cycle_list(&[0, 25, 50, 75, 100], (s.shake * 100.0).round() as i32, d);
            s.shake = pct as f32 / 100.0;
        },
        clamp: shake_clamp,
        write: |s| s.shake.to_string(),
        read: |s, v| set_parsed(&mut s.shake, v),
    },
    // Render lanes (see [`Settings::render_config`]). Engine flags, derived
    // occlusion/LOD state, and per-frame look lanes all apply live.
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
];

impl Settings {
    /// Record that an individual setting no longer matches a named profile.
    pub fn mark_custom(&mut self) {
        self.preset = PRESET_CUSTOM;
    }

    /// Apply a named/numeric performance profile. Shared by `/gfx`, the menu,
    /// and reproducible benchmark startup.
    pub fn select_preset(&mut self, value: &str) -> bool {
        let Some(preset) = parse_preset(value) else {
            return false;
        };
        self.apply_preset(preset);
        true
    }

    fn apply_preset(&mut self, preset: u8) {
        let preset = preset.min(PRESET_DEFAULT);
        if preset == PRESET_CUSTOM {
            self.mark_custom();
            return;
        }

        let mut profile = Self::default();
        match preset {
            PRESET_MINIMUM => {
                profile.vsync = false;
                profile.msaa = 1;
                profile.render_distance = 0;
                profile.render_scale = 0.25;
                profile.lighting = false;
                profile.vertical_distance = 1;
                profile.lod_levels = 1;
                profile.lod_detail = 6;
                profile.stream_hz = 15;
                profile.physics_hz = 30;
                profile.sky_hz = 15;
                profile.mod_hz = 15;
                profile.simulation = false;
                profile.mod_logic = false;
                profile.autosave = false;
                profile.hud_mode = HUD_OFF;
                profile.minimap = false;
                profile.mod_hud = false;
                profile.player_models = false;
                profile.name_tags = false;
                disable_costly_lanes(&mut profile);
                profile.lod2 = false;
            }
            PRESET_FAST => {
                profile.vsync = false;
                profile.msaa = 1;
                profile.render_distance = 3;
                profile.render_scale = 0.5;
                profile.lighting = false;
                profile.vertical_distance = 2;
                profile.lod_levels = 3;
                profile.lod_detail = 4;
                profile.stream_hz = 60;
                profile.physics_hz = 60;
                profile.sky_hz = 60;
                profile.mod_hz = 60;
                profile.simulation = true;
                profile.autosave = true;
                profile.hud_mode = HUD_MINIMAL;
                profile.minimap = false;
                profile.mod_hud = false;
                profile.player_models = true;
                profile.name_tags = false;
                disable_costly_lanes(&mut profile);
                profile.lod2 = true;
            }
            PRESET_DEFAULT => {}
            _ => unreachable!("preset was clamped above"),
        }
        self.copy_profile_values(&profile);
        self.preset = preset;
    }

    /// Copy only settings a profile owns. Personal/window controls deliberately
    /// remain untouched when a profile is selected.
    fn copy_profile_values(&mut self, p: &Self) {
        self.vsync = p.vsync;
        self.msaa = p.msaa;
        self.max_fps = p.max_fps;
        self.render_distance = p.render_distance;
        self.render_scale = p.render_scale;
        self.lighting = p.lighting;
        self.vertical_distance = p.vertical_distance;
        self.lod_levels = p.lod_levels;
        self.lod_detail = p.lod_detail;
        self.stream_hz = p.stream_hz;
        self.physics_hz = p.physics_hz;
        self.sky_hz = p.sky_hz;
        self.mod_hz = p.mod_hz;
        self.simulation = p.simulation;
        self.mod_logic = p.mod_logic;
        self.autosave = p.autosave;
        self.hud_mode = p.hud_mode;
        self.minimap = p.minimap;
        self.mod_hud = p.mod_hud;
        self.player_models = p.player_models;
        self.name_tags = p.name_tags;
        self.occlusion = p.occlusion;
        self.lod2 = p.lod2;
        self.blocklight = p.blocklight;
        self.exposure = p.exposure;
        self.bloom = p.bloom;
        self.godrays = p.godrays;
        self.clouds = p.clouds;
        self.weather = p.weather;
        self.stars = p.stars;
        self.day_night = p.day_night;
        self.taa = p.taa;
        self.fog = p.fog;
        self.ambient = p.ambient;
        self.sunlight = p.sunlight;
        self.shadows = p.shadows;
        self.sky = p.sky;
        self.vrs = p.vrs;
        self.water_anim = p.water_anim;
        self.ao = p.ao;
        self.vignette = p.vignette;
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
        // Engine render lanes live-swap on both threads. Occlusion/LOD are world
        // inputs handled by `Game::apply_settings`, so they are not engine flags.
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
}

// ---------------------------------------------------------------------------
// Shared value helpers — the single definition each surface reuses.
// ---------------------------------------------------------------------------

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

fn preset_name(preset: u8) -> &'static str {
    match preset {
        PRESET_MINIMUM => "Minimum",
        PRESET_FAST => "Fast",
        PRESET_DEFAULT => "Default",
        _ => "Custom",
    }
}

fn parse_preset(value: &str) -> Option<u8> {
    match value {
        "custom" | "0" => Some(PRESET_CUSTOM),
        "minimum" | "min" | "1" => Some(PRESET_MINIMUM),
        "fast" | "2" => Some(PRESET_FAST),
        "default" | "3" => Some(PRESET_DEFAULT),
        _ => None,
    }
}

fn preset_clamp(s: &mut Settings) {
    s.preset = s.preset.min(PRESET_DEFAULT);
}

fn hud_name(mode: u8) -> &'static str {
    match mode {
        HUD_MINIMAL => "Minimal",
        HUD_FULL => "Full",
        _ => "Off",
    }
}

fn parse_hud_mode(value: &str) -> Option<u8> {
    match value {
        "off" | "0" => Some(HUD_OFF),
        "minimal" | "min" | "1" => Some(HUD_MINIMAL),
        "full" | "2" => Some(HUD_FULL),
        _ => None,
    }
}

fn hud_mode_clamp(s: &mut Settings) {
    s.hud_mode = s.hud_mode.min(HUD_FULL);
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

fn msaa_clamp(s: &mut Settings) {
    s.msaa = snap_down(MSAA, s.msaa as i32) as u32;
}

fn vertical_distance_clamp(s: &mut Settings) {
    s.vertical_distance = s.vertical_distance.clamp(
        *VERTICAL_DISTANCE_RANGE.start(),
        *VERTICAL_DISTANCE_RANGE.end(),
    );
}

fn max_lod_levels(detail: u8) -> u8 {
    (MAX_LOD_DETAIL - detail.clamp(*LOD_DETAIL_RANGE.start(), *LOD_DETAIL_RANGE.end()) + 1)
        .min(*LOD_LEVELS_RANGE.end())
}

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

fn lod_clamp(s: &mut Settings) {
    s.lod_detail = s
        .lod_detail
        .clamp(*LOD_DETAIL_RANGE.start(), *LOD_DETAIL_RANGE.end());
    s.lod_levels = s
        .lod_levels
        .clamp(*LOD_LEVELS_RANGE.start(), max_lod_levels(s.lod_detail));
}

fn stream_rate_clamp(s: &mut Settings) {
    s.stream_hz = snap_rate(STREAM_RATES, s.stream_hz.min(i32::MAX as u32) as i32) as u32;
}

fn physics_rate_clamp(s: &mut Settings) {
    s.physics_hz = snap_rate(PHYSICS_RATES, s.physics_hz.min(i32::MAX as u32) as i32) as u32;
}

fn sky_rate_clamp(s: &mut Settings) {
    s.sky_hz = snap_rate(SKY_RATES, s.sky_hz.min(i32::MAX as u32) as i32) as u32;
}

fn mod_rate_clamp(s: &mut Settings) {
    s.mod_hz = snap_rate(MOD_RATES, s.mod_hz.min(i32::MAX as u32) as i32) as u32;
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

    #[test]
    fn roundtrip_through_text() {
        let mut s = Settings::default();
        s.fullscreen = true;
        s.msaa = 4;
        s.max_fps = 144;
        s.render_distance = 8;
        s.fov = 90.0;
        s.render_scale = 0.75;
        s.preset = PRESET_CUSTOM;
        s.vertical_distance = 7;
        s.lod_levels = 4;
        s.lod_detail = 4;
        s.stream_hz = 60;
        s.physics_hz = 500;
        s.sky_hz = 60;
        s.mod_hz = 120;
        s.simulation = false;
        s.autosave = false;
        s.hud_mode = HUD_MINIMAL;
        s.minimap = false;
        s.mod_hud = false;
        s.player_models = false;
        s.name_tags = false;
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
            assert_eq!(
                snap_down(&[1, 2, 4, 8], v),
                old_bucket(v as u32) as i32,
                "v={v}"
            );
        }
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
            preset: PRESET_CUSTOM,
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
            vertical_distance: 5,
            lod_levels: 5,
            lod_detail: 3,
            stream_hz: 240,
            physics_hz: 1000,
            sky_hz: 240,
            mod_hz: 120,
            simulation: false,
            mod_logic: false,
            autosave: false,
            hud_mode: HUD_MINIMAL,
            minimap: false,
            mod_hud: false,
            player_models: false,
            name_tags: false,
            // Not persisted (env-only); must stay at the default so the composed
            // roundtrip below — which never writes it — still lands `samples`. The
            // render lanes likewise stay at their persisted defaults via the spread.
            cull_faces: false,
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
    fn console_aliases_resolve_and_caps_differ() {
        // Aliases reach the same field...
        let mut a = Settings::default();
        let mut b = Settings::default();
        let dist = SETTINGS.iter().find(|f| f.matches("renderdist")).unwrap();
        let dist2 = SETTINGS
            .iter()
            .find(|f| f.matches("renderdistance"))
            .unwrap();
        assert!(dist.parse_human(&mut a, "8"));
        assert!(dist2.parse_human(&mut b, "8"));
        assert_eq!(a.render_distance, 8);
        assert_eq!(b.render_distance, 8);
        // ...and the one on/off helper gives menu caps vs console lowercase.
        assert_eq!(on_off(true, true), "On");
        assert_eq!(on_off(true, false), "on");
    }

    #[test]
    fn presets_apply_owned_fields_and_preserve_personal_controls() {
        let preset = SETTINGS.iter().find(|f| f.matches("preset")).unwrap();
        let mut s = Settings::default();
        s.fullscreen = true;
        s.max_fps = 777;
        s.fov = 105.0;
        s.ui_scale = 1.5;
        s.menu_scale = 0.75;
        s.shake = 0.25;
        s.cull_faces = true;

        assert!(preset.parse_human(&mut s, "minimum"));
        assert_eq!(s.preset, PRESET_MINIMUM);
        assert_eq!(s.max_fps, 0, "a performance preset must remove an old cap");
        assert_eq!(s.render_scale, 0.25);
        assert_eq!((s.render_distance, s.vertical_distance), (0, 1));
        assert!(!s.lod2);
        assert_eq!((s.lod_levels, s.lod_detail), (1, 6));
        assert_eq!(lod_range_metres(&s), 32, "zero near radius keeps one LOD unit");
        assert_eq!((s.stream_hz, s.physics_hz, s.sky_hz, s.mod_hz), (15, 30, 15, 15));
        assert_eq!(s.hud_mode, HUD_OFF);
        assert!(!s.simulation && !s.mod_logic && !s.autosave);
        assert!(!s.minimap && !s.mod_hud && !s.player_models && !s.name_tags);
        assert!(!s.lighting && !s.occlusion && !s.ao && !s.vrs);
        assert!(!s.sky && !s.bloom && !s.clouds && !s.water_anim);
        assert!(s.sunlight);

        assert!(preset.parse_human(&mut s, "fast"));
        assert_eq!(s.preset, PRESET_FAST);
        assert_eq!(s.max_fps, 0);
        assert_eq!(s.render_scale, 0.5);
        assert_eq!((s.render_distance, s.vertical_distance), (3, 2));
        assert!(s.lod2);
        assert_eq!((s.lod_levels, s.lod_detail), (3, 4));
        assert_eq!(lod_range_metres(&s), 384);
        assert_eq!((s.stream_hz, s.physics_hz, s.sky_hz, s.mod_hz), (60, 60, 60, 60));
        assert_eq!(s.hud_mode, HUD_MINIMAL);
        assert!(s.simulation && s.mod_logic && s.autosave && s.player_models);
        assert!(!s.minimap && !s.mod_hud && !s.name_tags);

        assert!(preset.parse_human(&mut s, "default"));
        let mut expected = Settings::default();
        expected.fullscreen = true;
        expected.fov = 105.0;
        expected.ui_scale = 1.5;
        expected.menu_scale = 0.75;
        expected.shake = 0.25;
        expected.cull_faces = true;
        assert_eq!(s, expected);
    }

    #[test]
    fn lod_clamp_enforces_combined_detail_limit() {
        let mut s = Settings::default();
        s.lod_detail = 6;
        s.lod_levels = 8;
        s.clamp();
        assert_eq!((s.lod_detail, s.lod_levels), (6, 4));
        assert!(s.lod_detail + s.lod_levels - 1 <= MAX_LOD_DETAIL);

        s.lod_detail = 255;
        s.lod_levels = 0;
        s.clamp();
        assert_eq!((s.lod_detail, s.lod_levels), (6, 1));

        let detail = SETTINGS.iter().find(|f| f.matches("lod_detail")).unwrap();
        let levels = SETTINGS.iter().find(|f| f.matches("lod_levels")).unwrap();
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
        let preset = SETTINGS.iter().find(|f| f.matches("preset")).unwrap();
        let scale = SETTINGS.iter().find(|f| f.matches("render_scale")).unwrap();

        let mut s = Settings::default();
        assert!(preset.parse_human(&mut s, "fast"));
        assert_eq!(s.preset, PRESET_FAST);
        assert!(scale.parse_human(&mut s, "75"));
        assert_eq!(s.preset, PRESET_CUSTOM);

        assert!(preset.parse_human(&mut s, "fast"));
        assert!(!scale.parse_human(&mut s, "not-a-number"));
        assert_eq!(s.preset, PRESET_FAST);
        scale.step(&mut s, 1);
        assert_eq!(s.preset, PRESET_CUSTOM);

        let mut loaded = Settings::default();
        loaded.parse_from("preset=2\nrender_scale=0.75\nautosave=false\n");
        loaded.clamp();
        assert_eq!(loaded.preset, PRESET_FAST);
        assert_eq!(loaded.render_scale, 0.75);
        assert!(!loaded.autosave);

        loaded.mark_custom();
        assert_eq!(loaded.preset, PRESET_CUSTOM);
    }
}
