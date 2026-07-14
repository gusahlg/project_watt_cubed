//! Console command parsing and dispatch.
//!
//! [`execute`] takes one submitted line and returns the lines of output to show
//! in the console log. Adding a command is a single `match` arm — the dispatch is
//! deliberately tiny so it can grow into a richer command (or chat) system later.
//!
//! `/gfx` edits the [`Settings`] value only; the caller applies it to the engine
//! (and the world's render distance) after the command returns. That keeps every
//! command testable without a window.
use voxel_engine::DVec3;

use crate::block::Composition;
use crate::math::{WORLD_BORDER, block_coord};
use crate::player::Player;
use crate::settings::{SETTINGS, Settings};
use crate::sky::{DayLength, Sky};
use crate::ui::{Line, Role};
use crate::world::World;

/// Normal command output: each string becomes one neutral [`Role::Dim`] line.
fn shown(lines: Vec<String>) -> Vec<Line> {
    lines.into_iter().map(|l| Line::of(Role::Dim, l)).collect()
}

/// A rejection (bad args, unknown command, usage): [`Role::Danger`] lines. Because
/// the handler that owns the rejection is the only place that names it an error,
/// severity is carried in the type — the caller never guesses it from the text.
fn rejected(lines: Vec<String>) -> Vec<Line> {
    lines.into_iter().map(|l| Line::of(Role::Danger, l)).collect()
}

/// The primary command names, in the order `help` lists them. This is the single
/// source of truth for Tab-completion (see [`crate::console`]); aliases like
/// `teleport` are intentionally omitted so completion offers the canonical name.
pub const COMMAND_NAMES: &[&str] =
    &["tp", "pos", "inspect", "gfx", "time", "walkspeed", "flyspeed", "help"];

/// Run a console line against the game state, returning output lines for the log.
///
/// A leading `/` is optional, so both `tp 1 2 3` and `/tp 1 2 3` work. The world
/// is `&mut` for `tp` alone (it must prepare collision data at the destination);
/// read-only commands like `inspect` reborrow it shared.
pub fn execute(
    line: &str,
    player: &mut Player,
    world: &mut World,
    settings: &mut Settings,
    sky: &mut Sky,
) -> Vec<Line> {
    let line = line.strip_prefix('/').unwrap_or(line);
    let mut parts = line.split_whitespace();
    let Some(cmd) = parts.next() else {
        return Vec::new();
    };
    let args: Vec<&str> = parts.collect();

    match cmd {
        "tp" | "teleport" | "setpos" => teleport(&args, player, world),
        "pos" | "where" => shown(vec![format!("position: {}", fmt_pos(player.position))]),
        "inspect" | "look" => inspect(&args, player, world),
        "gfx" | "graphics" => gfx(&args, settings),
        "time" => time(&args, sky),
        "walkspeed" => walkspeed(&args, player),
        "flyspeed" => flyspeed(&args, player),
        "help" | "?" => help(),
        other => rejected(vec![format!("unknown command '{other}' — type 'help'")]),
    }
}

/// `/time` — show or set the day/night clock, or change the cycle length.
///
///   `time`                 show the current time and cycle length
///   `time set <when>`      `0..1` fraction, `0..24` hour, or a name
///                          (dawn/day/noon/dusk/night/midnight)
///   `time length <secs>`   set how long a full cycle lasts
fn time(args: &[&str], sky: &mut Sky) -> Vec<Line> {
    match args {
        [] => shown(vec![format!(
            "time: {}  ({:.3} of day, cycle {:.0}s)",
            clock_label(sky.clock.day()),
            sky.clock.day(),
            sky.day_length.0,
        )]),
        ["set", when] => match parse_when(when) {
            Some(day) => {
                sky.clock.set_day(day);
                shown(vec![format!("time set to {}", clock_label(day))])
            }
            None => rejected(vec!["time: use 0..1, 0..24, or dawn|day|noon|dusk|night".to_string()]),
        },
        ["length", secs] => match secs.parse::<f64>() {
            Ok(s) if s.is_finite() => {
                sky.day_length = DayLength::clamped(s);
                shown(vec![format!("day length set to {:.0}s", sky.day_length.0)])
            }
            _ => rejected(vec!["time: length must be a number of seconds".to_string()]),
        },
        _ => rejected(vec!["usage: time [set <when> | length <secs>]".to_string()]),
    }
}

/// Parse a `/time set` argument into a day fraction in `[0, 1)`. Accepts named
/// times, a `0..1` fraction, or a `0..24` hour.
fn parse_when(s: &str) -> Option<f64> {
    let named = match s.to_ascii_lowercase().as_str() {
        "midnight" => Some(0.0),
        "dawn" | "sunrise" => Some(0.25),
        "morning" => Some(0.35),
        "day" | "noon" | "midday" => Some(0.5),
        "dusk" | "sunset" => Some(0.75),
        "night" => Some(0.9),
        _ => None,
    };
    if named.is_some() {
        return named;
    }
    let v = s.parse::<f64>().ok().filter(|v| v.is_finite())?;
    // <= 1 reads as a fraction; otherwise as an hour of a 24-hour day.
    Some(if v <= 1.0 { v.rem_euclid(1.0) } else { (v / 24.0).rem_euclid(1.0) })
}

/// A short `HH:MM`-ish label for a day fraction (0.0 = 00:00, 0.5 = 12:00).
fn clock_label(day: f64) -> String {
    let total = (day.rem_euclid(1.0) * 24.0 * 60.0).round() as i32;
    format!("{:02}:{:02}", (total / 60) % 24, total % 60)
}

/// `tp <x> <y> <z>` — move the player to absolute world coordinates, clamped
/// to the ±[`WORLD_BORDER`] cube (the same clamp movement applies, so no code
/// path can carry a position that would overflow i32 block math). The output
/// reports the position actually landed on, clamp included.
///
/// The discontinuity is transactional: collision data around the destination
/// is generated synchronously BEFORE the player lands there, so the next
/// physics step never runs against unloaded not-yet-generated air (falling
/// through or embedding in terrain that streams in a moment later).
fn teleport(args: &[&str], player: &mut Player, world: &mut World) -> Vec<Line> {
    if args.len() != 3 {
        return rejected(vec!["usage: tp <x> <y> <z>".to_string()]);
    }
    let parsed: Result<Vec<f64>, _> = args.iter().map(|a| a.parse::<f64>()).collect();
    match parsed.as_deref() {
        Ok([x, y, z]) if x.is_finite() && y.is_finite() && z.is_finite() => {
            let target = DVec3::new(*x, *y, *z)
                .clamp(DVec3::splat(-WORLD_BORDER), DVec3::splat(WORLD_BORDER));
            world.prepare_around(target);
            player.position = target;
            // Cancel any accumulated fall so the player doesn't rocket down on arrival.
            player.cancel_fall();
            shown(vec![format!("teleported to {}", fmt_pos(player.position))])
        }
        _ => rejected(vec!["tp: x, y and z must be numbers".to_string()]),
    }
}

/// `gfx [setting value]` — show or change graphics settings at runtime.
/// The caller applies the mutated [`Settings`] to the engine and persists it.
fn gfx(args: &[&str], settings: &mut Settings) -> Vec<Line> {
    let usage = || {
        std::iter::once("usage: gfx <setting> <value>".to_string())
            .chain(SETTINGS.iter().map(|field| format!("  gfx {}", field.usage())))
            .collect()
    };

    match args {
        [] => shown(SETTINGS.iter().map(|field| field.confirm(settings)).collect()),
        [key, value] => match gfx_set(settings, key, value) {
            Some(msg) => shown(vec![msg]),
            None => rejected(usage()),
        },
        _ => rejected(usage()),
    }
}

/// `/gfx <key> <value>` dispatches through the one [`SETTINGS`] table: find the
/// field the key (or an alias) names, parse-and-clamp its value, and echo the
/// field's confirm line. `None` (unknown key OR unparseable value) means the
/// caller prints usage — and, because the field is written only after a successful
/// parse, a bad value changes nothing.
fn gfx_set(s: &mut Settings, key: &str, value: &str) -> Option<String> {
    let field = SETTINGS.iter().find(|f| f.matches(key))?;
    field.parse_human(s, value).then(|| field.confirm(s))
}

/// `walkspeed [n]` — show or set the player's ground walk speed, units/second.
fn walkspeed(args: &[&str], player: &mut Player) -> Vec<Line> {
    set_speed(args, "walkspeed", player, |p| &mut p.speed)
}

/// `flyspeed [n]` — show or set the player's flying speed, units/second.
fn flyspeed(args: &[&str], player: &mut Player) -> Vec<Line> {
    set_speed(args, "flyspeed", player, |p| &mut p.fly_speed)
}

/// Shared show/set logic for `walkspeed`/`flyspeed`: both just target a different
/// intrinsic on [`Player`], so the parse-validate-write-confirm shape lives once.
fn set_speed(
    args: &[&str],
    name: &str,
    player: &mut Player,
    field: impl FnOnce(&mut Player) -> &mut f64,
) -> Vec<Line> {
    match args {
        [] => shown(vec![format!("{name}: {:.2}", *field(player))]),
        [value] => match value.parse::<f64>() {
            Ok(v) if v.is_finite() && v > 0.0 => {
                *field(player) = v;
                shown(vec![format!("{name} set to {v:.2}")])
            }
            _ => rejected(vec![format!("{name}: value must be a positive number")]),
        },
        _ => rejected(vec![format!("usage: {name} [<units/second>]")]),
    }
}

/// `inspect [x y z]` — describe the block at a cell (default: the block under the
/// player's feet), showing what it's made of and the properties derived from that.
/// The in-game window onto the element/block system.
fn inspect(args: &[&str], player: &Player, world: &World) -> Vec<Line> {
    let cell = match args {
        [] => {
            // The block supporting the player: directly below the feet. The small
            // bias keeps it stable when standing exactly on a block's top face.
            let p = player.position;
            (
                block_coord(p.x),
                block_coord(player.feet_y() - 0.1),
                block_coord(p.z),
            )
        }
        [x, y, z] => match (x.parse(), y.parse(), z.parse()) {
            (Ok(x), Ok(y), Ok(z)) => (x, y, z),
            _ => return rejected(vec!["inspect: x, y and z must be integers".to_string()]),
        },
        _ => return rejected(vec!["usage: inspect [<x> <y> <z>]".to_string()]),
    };

    let (x, y, z) = cell;
    let id = world.block_at(x, y, z);
    let registry = world.registry();
    let block = registry.block(id);

    let mut out = vec![
        format!("block at {x} {y} {z}: {} (#{}) ", block.name, id.0),
        format!("  made of: {}", describe_composition(world, &block.composition)),
    ];

    let c = &block.core;
    out.push(format!(
        "  durability {}  hardness {}  density {}",
        c.durability, c.hardness, c.density
    ));
    out.push(format!(
        "  conductivity {}  thermal {}  friction {}",
        c.conductivity, c.thermal_conductivity, c.friction
    ));
    out.push(format!(
        "  temp-resist {}  light {}  transparency {}%",
        c.temperature_resistance, c.light_emission, c.transparency
    ));

    if !block.specials.is_empty() {
        let specials: Vec<String> = block
            .specials
            .iter()
            .map(|(kind, strength)| format!("{kind:?} {strength}"))
            .collect();
        out.push(format!("  special: {}", specials.join(", ")));
    }
    for reaction in &block.reactions {
        out.push(format!(
            "  reaction: {} (strength {})",
            reaction.name, reaction.strength
        ));
    }
    shown(out)
}

/// Render a composition as a readable element list, resolving ids to names.
fn describe_composition(world: &World, composition: &Composition) -> String {
    let elements = world.registry().elements();
    match composition {
        Composition::Natural(els) if els.is_empty() => "nothing (air)".to_string(),
        Composition::Natural(els) => els
            .iter()
            .map(|&e| elements.get(e).name.to_string())
            .collect::<Vec<_>>()
            .join(" + "),
        Composition::Mixture(mix) | Composition::Configuration { mix, .. } => mix
            .parts()
            .iter()
            .map(|&(e, p)| format!("{}% {}", p, elements.get(e).name))
            .collect::<Vec<_>>()
            .join(", "),
        Composition::Computational(_) => "logic-gate components".to_string(),
    }
}

fn help() -> Vec<Line> {
    shown(vec![
        "commands (a leading '/' is optional):".to_string(),
        "  tp <x> <y> <z>       teleport to coordinates".to_string(),
        "  pos                  show current coordinates".to_string(),
        "  inspect [x y z]      describe a block's elements & properties".to_string(),
        "  gfx [setting value]  show or change graphics settings".to_string(),
        "  time [set|length]    show or set the day/night clock".to_string(),
        "  walkspeed [n]        show or set ground walk speed".to_string(),
        "  flyspeed [n]         show or set flying speed".to_string(),
        "  help                 show this list".to_string(),
    ])
}

/// Format a position the same way the on-screen coordinate readout does.
fn fmt_pos(p: DVec3) -> String {
    format!("X {:.1}  Y {:.1}  Z {:.1}", p.x, p.y, p.z)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn player() -> Player {
        Player::new(DVec3::new(0.0, 0.0, 0.0))
    }

    /// A real generated world; cheap and GPU-free (meshes are uploaded separately).
    fn world() -> World {
        World::generate()
    }

    fn run(line: &str, p: &mut Player, w: &mut World) -> Vec<Line> {
        let mut s = Settings::default();
        let mut sky = Sky::new();
        execute(line, p, w, &mut s, &mut sky)
    }

    /// All the lines' text joined — for asserting on multi-line output.
    fn joined(lines: &[Line]) -> String {
        lines.iter().map(Line::text).collect::<Vec<_>>().join("\n")
    }

    #[test]
    fn tp_sets_position_and_clears_fall() {
        let (mut p, mut w) = (player(), world());
        p.motion = crate::player::Motion::Walking { velocity: DVec3::new(0.0, -50.0, 0.0), on_ground: false };
        let out = run("tp 1.5 2 3", &mut p, &mut w);
        assert_eq!(p.position, DVec3::new(1.5, 2.0, 3.0));
        assert_eq!(p.velocity().y, 0.0);
        assert!(out[0].text().contains("teleported"));
    }

    #[test]
    fn tp_keeps_f64_precision_and_clamps_to_the_border() {
        let (mut p, mut w) = (player(), world());
        // Far coordinates parse as f64: no f32 quantisation on the way in.
        run("tp 100000000.5 60 -7", &mut p, &mut w);
        assert_eq!(p.position, DVec3::new(100_000_000.5, 60.0, -7.0));

        // Past the border: clamped, and the OUTPUT reports the clamped spot.
        let out = run("tp 99999999999 60 -99999999999", &mut p, &mut w);
        assert_eq!(p.position.x, 1.0e9);
        assert_eq!(p.position.z, -1.0e9);
        assert!(out[0].text().contains("1000000000.0"), "reports the clamped position");

        // Non-finite input is refused outright.
        let before = p.position;
        run("tp inf 0 0", &mut p, &mut w);
        assert_eq!(p.position, before);
    }

    #[test]
    fn tp_prepares_collision_data_at_the_destination() {
        let (mut p, mut w) = (player(), world());
        // Far outside the pre-generated spawn region: without the prepare, the
        // ground under the destination would be unloaded air and the next
        // physics step would fall straight through.
        let (x, z) = (5_000, 5_000);
        let surface = w.surface_y(x, z);
        run(&format!("tp {x} {} {z}", surface + 2), &mut p, &mut w);
        // `is_solid` reads AIR for unloaded chunks, so this proves the ground
        // cell (rock or seabed) was actually generated by the teleport.
        assert!(
            w.is_solid(x, surface, z),
            "the destination's ground must be loaded before physics resumes"
        );
    }

    #[test]
    fn leading_slash_is_optional() {
        let (mut p, mut w) = (player(), world());
        run("/tp 4 5 6", &mut p, &mut w);
        assert_eq!(p.position, DVec3::new(4.0, 5.0, 6.0));
    }

    #[test]
    fn bad_args_do_not_move_the_player() {
        let (mut p, mut w) = (player(), world());
        run("tp 1 two 3", &mut p, &mut w);
        assert_eq!(p.position, DVec3::new(0.0, 0.0, 0.0));
        run("tp 1 2", &mut p, &mut w);
        assert_eq!(p.position, DVec3::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn unknown_command_reports_back() {
        let (mut p, mut w) = (player(), world());
        let out = run("fly-to-moon", &mut p, &mut w);
        assert!(out[0].text().contains("unknown command"));
        assert_eq!(out[0].spans().next().unwrap().role, Role::Danger);
    }

    #[test]
    fn inspect_reports_elements_and_properties() {
        let (mut p, mut w) = (player(), world());
        // Deep underground is stone: a single Stone element with stone's properties.
        let out = run("inspect 8 0 8", &mut p, &mut w);
        let text = joined(&out);
        assert!(text.contains("Stone"), "should name the block: {text}");
        assert!(text.contains("made of: Stone"), "should list elements: {text}");
        assert!(text.contains("density"), "should show core properties: {text}");
    }

    #[test]
    fn inspect_above_world_is_air() {
        let (mut p, mut w) = (player(), world());
        let out = run("inspect 8 60 8", &mut p, &mut w);
        assert!(joined(&out).contains("air"));
    }

    #[test]
    fn gfx_updates_settings_with_clamping() {
        let (mut p, mut w) = (player(), world());
        let mut s = Settings::default();
        let mut sky = Sky::new();
        execute("gfx msaa 4", &mut p, &mut w, &mut s, &mut sky);
        assert_eq!(s.msaa, 4);
        execute("gfx fps 144", &mut p, &mut w, &mut s, &mut sky);
        assert_eq!(s.max_fps, 144);
        execute("gfx fps off", &mut p, &mut w, &mut s, &mut sky);
        assert_eq!(s.max_fps, 0);
        execute("gfx renderdist 99", &mut p, &mut w, &mut s, &mut sky);
        assert_eq!(s.render_distance, 20);
        execute("gfx fullscreen on", &mut p, &mut w, &mut s, &mut sky);
        assert!(s.fullscreen);
        execute("gfx lighting off", &mut p, &mut w, &mut s, &mut sky);
        assert!(!s.lighting);
        let out = execute("gfx", &mut p, &mut w, &mut s, &mut sky);
        let text = joined(&out);
        assert!(text.contains("fullscreen on"));
        assert!(text.contains("lighting off"));
        assert!(text.contains("ui scale"));
    }

    #[test]
    fn gfx_bad_input_prints_usage_and_changes_nothing() {
        let (mut p, mut w) = (player(), world());
        let mut s = Settings::default();
        let mut sky = Sky::new();
        let before = s.clone();
        let out = execute("gfx msaa lots", &mut p, &mut w, &mut s, &mut sky);
        assert!(out[0].text().contains("usage"));
        let text = joined(&out);
        assert!(text.contains("lighting on|off"));
        assert!(text.contains("uiscale <50-200>"));
        assert_eq!(out[0].spans().next().unwrap().role, Role::Danger);
        assert_eq!(s, before);
    }

    #[test]
    fn time_set_accepts_names_fractions_and_hours() {
        let mut sky = Sky::new();
        assert!(time(&["set", "noon"], &mut sky)[0].text().contains("12:00"));
        assert!((sky.clock.day() - 0.5).abs() < 1e-9);
        time(&["set", "0.25"], &mut sky);
        assert!((sky.clock.day() - 0.25).abs() < 1e-9);
        time(&["set", "18"], &mut sky); // 18:00 → 0.75
        assert!((sky.clock.day() - 0.75).abs() < 1e-9);
        // A bad value leaves the clock untouched.
        let before = sky.clock.day();
        assert!(time(&["set", "banana"], &mut sky)[0].text().contains("use"));
        assert_eq!(sky.clock.day(), before);
    }

    #[test]
    fn walkspeed_and_flyspeed_set_independently() {
        let (mut p, mut w) = (player(), world());
        let out = run("walkspeed 10", &mut p, &mut w);
        assert_eq!(p.speed, 10.0);
        assert!(out[0].text().contains("walkspeed set to 10.00"));

        run("flyspeed 25", &mut p, &mut w);
        assert_eq!(p.fly_speed, 25.0);
        // Setting one doesn't disturb the other.
        assert_eq!(p.speed, 10.0);

        let out = run("walkspeed", &mut p, &mut w);
        assert!(out[0].text().contains("walkspeed: 10.00"));
    }

    #[test]
    fn speed_commands_reject_non_positive_and_non_finite() {
        let (mut p, mut w) = (player(), world());
        let before = p.speed;
        for bad in ["0", "-5", "inf", "nan", "banana"] {
            let out = run(&format!("walkspeed {bad}"), &mut p, &mut w);
            assert_eq!(p.speed, before, "{bad} should not change speed");
            assert_eq!(out[0].spans().next().unwrap().role, Role::Danger);
        }
    }

    #[test]
    fn time_length_clamps() {
        let mut sky = Sky::new();
        time(&["length", "1"], &mut sky); // below the 10s floor
        assert_eq!(sky.day_length.0, 10.0);
    }
}
