//! Console command parsing and dispatch.
//!
//! [`execute_with_visuals`] takes one submitted line and returns the lines of
//! output to show in the console log. Adding a command is a single `match`
//! arm — the dispatch is deliberately tiny so it can grow into a richer
//! command (or chat) system later.
//!
//! `/gfx` edits the [`Settings`] value only; the caller applies it to the engine
//! (and the world's render distance) after the command returns. That keeps every
//! command testable without a window.
use voxel_engine::DVec3;

use crate::math::WORLD_BORDER;
use crate::modding::{annotate_setting, VisualMask};
use crate::player::Player;
use crate::settings::{SETTINGS, Settings};
use crate::sky::{DayLength, Sky};
use crate::ui::{Line, Role};
use crate::world::terrain::cosmos::{Body, Cosmos, Kind, Shape};
use crate::world::World;

fn shown(lines: Vec<String>) -> Vec<Line> {
    lines.into_iter().map(|l| Line::of(Role::Dim, l)).collect()
}

fn rejected(lines: Vec<String>) -> Vec<Line> {
    lines.into_iter().map(|l| Line::of(Role::Danger, l)).collect()
}

macro_rules! commands {
    (
        $cmd:ident, $args:ident, $player:ident, $world:ident, $settings:ident, $sky:ident, $visuals:ident;
        $($canon:literal $(| $alias:literal)* , $help:literal => $body:expr);+ $(;)?
    ) => {
        /// The primary command names, in the order `help` lists them.
        pub const COMMAND_NAMES: &[&str] = &[$($canon),+];

        fn dispatch(
            $cmd: &str,
            $args: &[&str],
            $player: &mut Player,
            $world: &mut World,
            $settings: &mut Settings,
            $sky: &mut Sky,
            $visuals: VisualMask,
        ) -> Vec<Line> {
            match $cmd {
                $($canon $(| $alias)* => $body,)+
                other => rejected(vec![format!("unknown command '{other}' — type '/help'")]),
            }
        }

        fn help() -> Vec<Line> {
            shown(vec![
                "commands (a leading '/' is optional):".to_string(),
                $($help.to_string(),)+
            ])
        }
    };
}

commands! {
    cmd, args, player, world, settings, sky, visuals;
    "tp" | "teleport" | "setpos", "  /tp <x y z|name>      teleport to coordinates or a body" => teleport(args, player, world);
    "bodies", "  /bodies               list the worlds, nearest first" => bodies(player, world);
    "noclip", "  /noclip               toggle flight through geometry" => noclip(player);
    "pos" | "where", "  /pos                  show current coordinates" => shown(vec![format!("position: {}", fmt_pos(player.position))]);
    "inspect" | "look", "  /inspect [x y z]      describe a block's elements & properties" => inspect(args, player, world);
    "reactions", "  /reactions            show pending reaction events" => reactions(world);
    "gfx" | "graphics", "  /gfx [setting value]  show or change graphics settings" => gfx(args, settings, visuals);
    "time", "  /time [set|length]    show or set the day/night clock" => time(args, sky);
    "walkspeed", "  /walkspeed [n]        show or set ground walk speed" => walkspeed(args, player);
    "flyspeed", "  /flyspeed [n]         show or set flying speed" => flyspeed(args, player);
    "mute", "  /mute                 toggle master mute (this session)" => mute(settings);
    "deafen", "  /deafen               toggle hearing incoming voice" => deafen(settings);
    "audio" | "volume", "  /audio <chan> <0-100> set master/effects/voice volume" => audio(args, settings);
    "voicetest", "  /voicetest            play a local voice test cue" => voicetest();
    "name", "  /name <n> <text>      name a recorded crafting procedure" => rejected(vec!["/name: no procedure journal (is the crafting mod enabled?)".to_string()]);
    "gravity" | "g", "  /gravity              show the local pull of the matter around you" => gravity(player, world);
    "help" | "?", "  /help                 show this list" => help();
}

/// Run a console line against the game state, returning output lines for the log.
///
/// A leading `/` is optional, so both `tp 1 2 3` and `/tp 1 2 3` work. The world
/// is `&mut` for `tp` alone (it requests the destination collision slab);
/// read-only commands like `inspect` reborrow it shared.
#[cfg(test)]
pub fn execute(
    line: &str,
    player: &mut Player,
    world: &mut World,
    settings: &mut Settings,
    sky: &mut Sky,
) -> Vec<Line> {
    execute_with_visuals(line, player, world, settings, sky, VisualMask::default())
}

/// [`execute`] with the live visual-mod mask so `/gfx` reports effective lanes.
pub fn execute_with_visuals(
    line: &str,
    player: &mut Player,
    world: &mut World,
    settings: &mut Settings,
    sky: &mut Sky,
    visuals: VisualMask,
) -> Vec<Line> {
    let line = line.strip_prefix('/').unwrap_or(line);
    let mut parts = line.split_whitespace();
    let Some(cmd) = parts.next() else {
        return Vec::new();
    };
    let args: Vec<&str> = parts.collect();

    dispatch(cmd, &args, player, world, settings, sky, visuals)
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
            None => rejected(vec!["/time: use 0..1, 0..24, or dawn|day|noon|dusk|night".to_string()]),
        },
        ["length", secs] => match secs.parse::<f64>() {
            Ok(s) if s.is_finite() => {
                sky.day_length = DayLength::clamped(s);
                shown(vec![format!("day length set to {:.0}s", sky.day_length.0)])
            }
            _ => rejected(vec!["/time: length must be a number of seconds".to_string()]),
        },
        _ => rejected(vec!["usage: /time [set <when> | length <secs>]".to_string()]),
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

/// `tp <x> <y> <z>` or `tp <name> [n]` — move the player, clamped to the
/// ±[`WORLD_BORDER`] cube (the same clamp movement applies, so no code path can
/// carry a position that would overflow i32 block math). The output reports the
/// position actually landed on, clamp included.
///
/// The discontinuity is transactional: collision data around the destination
/// is *requested* before the player lands there, and physics stays frozen
/// until [`World::spawn_ready`] is true, so the next physics step never runs
/// against unloaded air.
fn teleport(args: &[&str], player: &mut Player, world: &mut World) -> Vec<Line> {
    let numeric = args.first().is_some_and(|a| a.parse::<f64>().is_ok());
    if numeric || args.len() == 3 {
        return teleport_coords(args, player, world);
    }
    match args {
        [name] => teleport_named(name, 1, player, world),
        [name, n] => match n.parse::<usize>() {
            Ok(n) if n >= 1 => teleport_named(name, n, player, world),
            _ => tp_usage(),
        },
        _ => tp_usage(),
    }
}

fn tp_usage() -> Vec<Line> {
    rejected(vec!["usage: /tp <x> <y> <z>  or  /tp <name> [n]".to_string()])
}

fn teleport_coords(args: &[&str], player: &mut Player, world: &mut World) -> Vec<Line> {
    if args.len() != 3 {
        return rejected(vec!["usage: /tp <x> <y> <z>".to_string()]);
    }
    let parsed: Result<Vec<f64>, _> = args.iter().map(|a| a.parse::<f64>()).collect();
    match parsed.as_deref() {
        Ok([x, y, z]) if x.is_finite() && y.is_finite() && z.is_finite() => {
            place(DVec3::new(*x, *y, *z), player, world)
        }
        _ => rejected(vec!["/tp: x, y and z must be numbers".to_string()]),
    }
}

/// Land on the `n`th body of a kind (1-based, catalog order). A unique prefix of the kind name
/// is enough. Worlds without a cosmos are left untouched.
fn teleport_named(name: &str, n: usize, player: &mut Player, world: &mut World) -> Vec<Line> {
    let Some(cosmos) = world.terrain().cosmos() else {
        return no_cosmos();
    };
    let kind = match resolve_kind(name) {
        Ok(kind) => kind,
        Err(lines) => return lines,
    };
    match cosmos.bodies().iter().filter(|b| b.kind == kind).nth(n - 1) {
        Some(body) => place(landing_vec(body), player, world),
        None => rejected(vec![format!("no {} {n}", kind.name())]),
    }
}

/// Stand the player at `target` the way coordinate `/tp` does.
fn place(target: DVec3, player: &mut Player, world: &mut World) -> Vec<Line> {
    let target = target.clamp(DVec3::splat(-WORLD_BORDER), DVec3::splat(WORLD_BORDER));
    world.prepare_around(target);
    player.position = target;
    // Stand up along the local gravity at once (no slow roll after a jump across the
    // universe); in weightlessness keep the current frame.
    // The body frame follows the last applied pull: make it the destination's at once.
    let pull = world.gravity_at(target);
    player.gravity = pull.accel;
    if let Some(up) = pull.up(0.02 * crate::player::STANDARD_GRAVITY) {
        player.snap_up(up);
    }
    // Cancel any accumulated fall so the player doesn't rocket down on arrival.
    player.cancel_fall();
    shown(vec![format!("teleported to {}", fmt_pos(player.position))])
}

fn no_cosmos() -> Vec<Line> {
    rejected(vec!["this world has no cosmos".to_string()])
}

/// `/noclip` — from walking or ordinary flight into noclip flight, and from noclip back to walking.
fn noclip(player: &mut Player) -> Vec<Line> {
    player.toggle_noclip();
    shown(vec![format!("noclip {}", if player.noclip() { "on" } else { "off" })])
}

/// `/bodies` — every cosmos body, nearest first. The number is 1-based within the kind, in
/// catalog order. The `/tp` lands 2000 blocks above the +Y datum.
fn bodies(player: &Player, world: &World) -> Vec<Line> {
    let Some(cosmos) = world.terrain().cosmos() else {
        return no_cosmos();
    };
    let origin = player.position;
    let mut order: Vec<usize> = (0..cosmos.bodies().len()).collect();
    order.sort_by(|&i, &j| {
        let di = (cosmos.bodies()[i].centre_f() - origin).length_squared();
        let dj = (cosmos.bodies()[j].centre_f() - origin).length_squared();
        di.total_cmp(&dj).then(i.cmp(&j))
    });
    let lines = order
        .into_iter()
        .map(|i| {
            let body = &cosmos.bodies()[i];
            let n = kind_number(cosmos, body);
            let dist = (body.centre_f() - origin).length();
            let at = landing(body);
            format!(
                "{} {n}  {} away  {}  /tp {} {} {}",
                body.kind.name(),
                fmt_dist(dist),
                fmt_size(body),
                at[0],
                at[1],
                at[2],
            )
        })
        .collect();
    shown(lines)
}

/// 1-based index of `body` among bodies of its kind, in catalog order.
fn kind_number(cosmos: &Cosmos, body: &Body) -> usize {
    cosmos.bodies().iter().filter(|b| b.kind == body.kind).position(|b| b.id == body.id).unwrap() + 1
}

fn resolve_kind(prefix: &str) -> Result<Kind, Vec<Line>> {
    let key = prefix.to_ascii_lowercase();
    let mut hit = None;
    for kind in [Kind::Home, Kind::Twin, Kind::Verdant, Kind::Hollow, Kind::Ember, Kind::Moon] {
        if kind.name().starts_with(&key) {
            if hit.is_some() {
                return Err(rejected(vec![format!("'{prefix}' matches more than one kind")]));
            }
            hit = Some(kind);
        }
    }
    hit.ok_or_else(|| rejected(vec![format!("unknown body '{prefix}'")]))
}

/// Cube half-edge, ball radius, or shell outer radius, then 2000 blocks of clearance on +Y.
fn landing(body: &Body) -> [i64; 3] {
    let top = match body.shape {
        Shape::Cube { half } => half,
        Shape::Ball { r } => r,
        Shape::Shell { outer, .. } => outer,
    };
    [body.centre[0], body.centre[1] + top + 2_000, body.centre[2]]
}

fn landing_vec(body: &Body) -> DVec3 {
    let at = landing(body);
    DVec3::new(at[0] as f64, at[1] as f64, at[2] as f64)
}

fn fmt_size(body: &Body) -> String {
    match body.shape {
        Shape::Cube { half } => format!("half {half}"),
        Shape::Ball { r } => format!("radius {r}"),
        Shape::Shell { outer, .. } => format!("radius {outer}"),
    }
}

/// Rounded distance with a `k` or `M` suffix.
fn fmt_dist(d: f64) -> String {
    let d = d.abs();
    if d >= 999_500.0 {
        format!("{}M", (d / 1_000_000.0).round() as i64)
    } else if d >= 999.5 {
        format!("{}k", (d / 1_000.0).round() as i64)
    } else {
        format!("{}", d.round() as i64)
    }
}

/// `gfx [setting value]` — show or change graphics settings at runtime.
/// The caller applies the mutated [`Settings`] to the engine and persists it.
fn gfx(args: &[&str], settings: &mut Settings, visuals: VisualMask) -> Vec<Line> {
    let usage = || {
        std::iter::once("usage: /gfx <setting> <value>".to_string())
            .chain(SETTINGS.iter().map(|field| format!("  /gfx {}", field.usage())))
            .collect()
    };

    match args {
        [] => shown(
            SETTINGS
                .iter()
                .map(|field| {
                    let msg = if field.key() == "vrs" {
                        settings.vrs_gfx_line()
                    } else {
                        field.confirm(settings)
                    };
                    annotate_setting(msg, field.key(), visuals)
                })
                .collect(),
        ),
        [key, value] => match gfx_set(settings, key, value) {
            Some(msg) => {
                let field_key = SETTINGS
                    .iter()
                    .find(|f| f.matches(key))
                    .map(|f| f.key())
                    .unwrap_or(*key);
                let msg = if field_key == "vrs" {
                    settings.vrs_gfx_line()
                } else {
                    msg
                };
                shown(vec![annotate_setting(msg, field_key, visuals)])
            }
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

/// `/mute` — toggle the transient master mute. Not persisted (resets each launch);
/// the caller pushes the mutated [`Settings`] to the mixer via [`Settings::mix_change`].
fn mute(settings: &mut Settings) -> Vec<Line> {
    settings.muted = !settings.muted;
    shown(vec![format!("audio {}", if settings.muted { "muted" } else { "unmuted" })])
}

/// `/deafen` — toggle whether incoming voice is heard. Flips the persisted
/// `voice_incoming` gate (deafen is its inverse), so the caller saves the change.
fn deafen(settings: &mut Settings) -> Vec<Line> {
    settings.voice_incoming = !settings.voice_incoming;
    let msg = if settings.voice_incoming { "undeafened (hearing voice)" } else { "deafened (voice muted)" };
    shown(vec![msg.to_string()])
}

/// `/audio <master|effects|voice> <0-100>` — set one mix volume, clamped to 0..=100.
/// The caller persists the mutated [`Settings`]; a bad channel or value changes nothing.
fn audio(args: &[&str], settings: &mut Settings) -> Vec<Line> {
    let usage = || rejected(vec!["usage: /audio <master|effects|voice> <0-100>".to_string()]);
    let [channel, value] = args else {
        return usage();
    };
    let Ok(pct) = value.parse::<u8>() else {
        return usage();
    };
    let field = match *channel {
        "master" => &mut settings.master_volume,
        "effects" | "sfx" => &mut settings.effects_volume,
        "voice" => &mut settings.voice_volume,
        _ => return usage(),
    };
    *field = pct.min(100);
    shown(vec![format!("{channel} volume {}%", *field)])
}

/// `/voicetest` — play a local test cue so the user can check their voice path.
fn voicetest() -> Vec<Line> {
    // `execute` has no audio access (the `SoundSystem` handle lives in game.rs),
    // so this only reports that the test was requested.
    shown(vec!["queued a voice test cue".to_string()])
}

/// `walkspeed [n]` — show or set the player's ground walk speed, units/second.
fn walkspeed(args: &[&str], player: &mut Player) -> Vec<Line> {
    set_speed(args, "/walkspeed", player, |p| &mut p.speed)
}

/// `flyspeed [n]` — show or set the player's flying speed, units/second.
fn flyspeed(args: &[&str], player: &mut Player) -> Vec<Line> {
    set_speed(args, "/flyspeed", player, |p| &mut p.fly_speed)
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

fn inspect(args: &[&str], player: &Player, world: &World) -> Vec<Line> {
    let cell = match args {
        [] => {
            // The block supporting the player: 0.1 along −up, storage −Y on a chart.
            world.ground_cell(player.feet(), player.up_axis)
        }
        [x, y, z] => match (x.parse(), y.parse(), z.parse()) {
            (Ok(x), Ok(y), Ok(z)) => (x, y, z),
            _ => return rejected(vec!["/inspect: x, y and z must be integers".to_string()]),
        },
        _ => return rejected(vec!["usage: /inspect [<x> <y> <z>]".to_string()]),
    };

    let (x, y, z) = cell;
    let id = world.block_at(x, y, z);
    let registry = world.registry();
    let cfg = registry.configuration(id);
    let obs = registry.observation(id);
    let words = registry.display_name(id);
    // Labels are worldgen roles ("rock:1"), an internal annotation; the console shows them as such.
    let role = registry.label(id).map(|l| format!(", worldgen role {l}")).unwrap_or_default();
    let elems: Vec<String> = cfg
        .elements()
        .iter()
        .map(|e| format!("[{},{},{},{}]", e.0[0], e.0[1], e.0[2], e.0[3]))
        .collect();
    let made = if elems.is_empty() {
        "void".to_string()
    } else {
        elems.join(" + ")
    };
    shown(vec![
        format!("block at {x} {y} {z}: {words} (#{}{role})", id.0),
        format!("  made of: {made}"),
        format!(
            "  solid {}  transparency {}  emission {}",
            obs.solid as u8, obs.transparency, obs.emission
        ),
        format!(
            "  hardness {}  friction {}  cohesion {}",
            obs.hardness, obs.friction, obs.cohesion
        ),
        format!("  descriptor {}", registry.render_layer(id)),
    ])
}

/// `/reactions` — active contacts, turns run, law operations committed.
fn reactions(world: &World) -> Vec<Line> {
    let r = world.reactions();
    shown(vec![format!(
        "reactions: active={} turns={} operations={}",
        r.pending(),
        r.turns,
        r.operations
    )])
}

/// Format a position the same way the on-screen coordinate readout does.
/// `/gravity` — the field at the player: strength, direction, tilt from the ground's grid axis, the
/// potential, the declared error and the source epoch.
fn gravity(player: &Player, world: &World) -> Vec<Line> {
    let s = world.gravity().sample_tidal(player.position);
    let g = s.accel.length();
    let metres = crate::math::BLOCK_METERS;
    let mut out = vec![format!(
        "gravity: {:.3} m/s² ({:.1} % of the spawn pull)",
        g * metres,
        100.0 * g / crate::player::STANDARD_GRAVITY
    )];
    match s.up(0.02 * crate::player::STANDARD_GRAVITY) {
        Some(up) => {
            let n = player.up_axis.dvec();
            let tilt = up.dot(n).clamp(-1.0, 1.0).acos().to_degrees();
            out.push(format!("down: ({:.4}, {:.4}, {:.4}); {tilt:.3}° off the {:?} grid axis", -up.x, -up.y, -up.z, player.up_axis));
        }
        None => out.push("weightless: the body keeps its orientation".to_string()),
    }
    out.push(format!(
        "potential {:.4e} blocks²/s², error ≤ {:.2e} m/s², source epoch {}",
        s.potential,
        s.error * metres,
        s.epoch
    ));
    shown(out)
}

fn fmt_pos(p: DVec3) -> String {
    format!("X {:.1}  Y {:.1}  Z {:.1}", p.x, p.y, p.z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modding::forced_off_marker;
    use crate::render_config::VrsChoice;

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
    fn help_text_is_stable() {
        assert_eq!(
            joined(&help()),
            "commands (a leading '/' is optional):\n  \
             /tp <x y z|name>      teleport to coordinates or a body\n  \
             /bodies               list the worlds, nearest first\n  \
             /noclip               toggle flight through geometry\n  \
             /pos                  show current coordinates\n  \
             /inspect [x y z]      describe a block's elements & properties\n  \
             /reactions            show pending reaction events\n  \
             /gfx [setting value]  show or change graphics settings\n  \
             /time [set|length]    show or set the day/night clock\n  \
             /walkspeed [n]        show or set ground walk speed\n  \
             /flyspeed [n]         show or set flying speed\n  \
             /mute                 toggle master mute (this session)\n  \
             /deafen               toggle hearing incoming voice\n  \
             /audio <chan> <0-100> set master/effects/voice volume\n  \
             /voicetest            play a local voice test cue\n  \
             /name <n> <text>      name a recorded crafting procedure\n  \
             /gravity              show the local pull of the matter around you\n  \
             /help                 show this list"
        );
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
        assert!(
            !w.spawn_ready(),
            "a far teleport must wait on the async spawn slab"
        );
        w.drive_spawn_ready();
        // `is_solid` reads AIR for unloaded chunks, so this proves the ground
        // cell under the surface landed before physics would resume.
        assert!(
            w.is_solid(x, surface - 1, z),
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
        // Deep underground is rock: a labelled configuration with observation readings.
        let out = run("inspect 8 0 8", &mut p, &mut w);
        let text = joined(&out);
        assert!(text.contains("rock"), "should name the block: {text}");
        assert!(text.contains("made of:"), "should list elements: {text}");
        assert!(text.contains("hardness"), "should show observation readings: {text}");
        assert!(text.contains("descriptor"), "should show the render descriptor: {text}");
    }

    #[test]
    fn inspect_above_world_is_air() {
        let (mut p, mut w) = (player(), world());
        let out = run("inspect 8 60 8", &mut p, &mut w);
        assert!(joined(&out).contains("air"));
    }

    #[test]
    fn reactions_prints_active_turns_operations() {
        let (mut p, mut w) = (player(), world());
        let out = run("reactions", &mut p, &mut w);
        assert_eq!(joined(&out), "reactions: active=0 turns=0 operations=0");
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
        execute("gfx vrs on", &mut p, &mut w, &mut s, &mut sky);
        assert_eq!(s.vrs, VrsChoice::On);
        execute("gfx vrs auto", &mut p, &mut w, &mut s, &mut sky);
        assert_eq!(s.vrs, VrsChoice::Auto);
        let out = execute("gfx", &mut p, &mut w, &mut s, &mut sky);
        let text = joined(&out);
        assert!(text.contains("fullscreen on"));
        assert!(text.contains("lighting off"));
        assert!(text.contains("vrs auto"));
        assert!(text.contains("ui scale"));
    }

    #[test]
    fn gfx_lists_default_auto_render_scale() {
        let (mut p, mut w) = (player(), world());
        let mut s = Settings::default();
        let mut sky = Sky::new();
        s.note_render_extent(1920, 1080, 1.0);
        let text = joined(&execute("gfx", &mut p, &mut w, &mut s, &mut sky));
        assert!(
            text.contains(&format!(
                "render scale Auto ({:.1})",
                crate::settings::DEFAULT_AUTO_RENDER_SCALE
            )),
            "Default /gfx prints the effective Auto scale: {text}"
        );
    }

    #[test]
    fn gfx_lists_effective_visual_lanes_when_a_mod_strips_them() {
        let (mut p, mut w) = (player(), world());
        let mut s = Settings::default();
        let mut sky = Sky::new();
        let mask = VisualMask {
            atmosphere: true,
            post: false,
            lighting: true,
        };
        let out = execute_with_visuals("gfx", &mut p, &mut w, &mut s, &mut sky, mask);
        let text = joined(&out);
        assert!(
            text.contains(&format!("bloom on {}", forced_off_marker("Post"))),
            "effective /gfx must name the stripping mod: {text}"
        );
        assert!(
            !text.contains("shadows on (off:"),
            "Lighting is still enabled: {text}"
        );
        let set = execute_with_visuals("gfx bloom off", &mut p, &mut w, &mut s, &mut sky, mask);
        assert!(
            joined(&set).contains(&format!("bloom off {}", forced_off_marker("Post"))),
            "a set confirmation must also show the strip: {}",
            joined(&set)
        );
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

    /// Run a command against an explicit settings value (audio commands mutate it).
    fn run_settings(line: &str, s: &mut Settings) -> Vec<Line> {
        let (mut p, mut w) = (player(), world());
        let mut sky = Sky::new();
        execute(line, &mut p, &mut w, s, &mut sky)
    }

    #[test]
    fn mute_toggles_transient_and_survives_no_save() {
        let mut s = Settings::default();
        assert!(!s.muted);
        assert!(run_settings("mute", &mut s)[0].text().contains("muted"));
        assert!(s.muted);
        assert!(run_settings("mute", &mut s)[0].text().contains("unmuted"));
        assert!(!s.muted);
    }

    #[test]
    fn deafen_flips_the_persisted_incoming_gate() {
        let mut s = Settings::default();
        assert!(s.voice_incoming);
        run_settings("deafen", &mut s);
        assert!(!s.voice_incoming);
        assert!(s.mix_change().deafen, "deafen is the inverse of voice_incoming");
        run_settings("deafen", &mut s);
        assert!(s.voice_incoming);
    }

    #[test]
    fn audio_sets_and_clamps_each_channel() {
        let mut s = Settings::default();
        run_settings("audio master 45", &mut s);
        assert_eq!(s.master_volume, 45);
        run_settings("audio effects 200", &mut s); // over 100 clamps
        assert_eq!(s.effects_volume, 100);
        run_settings("audio voice 0", &mut s);
        assert_eq!(s.voice_volume, 0);

        // Bad channel or value is a Danger rejection that changes nothing.
        let before = s.clone();
        let out = run_settings("audio bass 50", &mut s);
        assert_eq!(out[0].spans().next().unwrap().role, Role::Danger);
        let out = run_settings("audio master loud", &mut s);
        assert_eq!(out[0].spans().next().unwrap().role, Role::Danger);
        assert_eq!(s, before);
    }

    #[test]
    fn voicetest_returns_a_placeholder_line() {
        let mut s = Settings::default();
        let out = run_settings("voicetest", &mut s);
        assert!(out[0].text().contains("voice test"));
        assert_eq!(out[0].spans().next().unwrap().role, Role::Dim);
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

    #[test]
    fn noclip_toggles_walk_to_noclip_and_back() {
        let (mut p, mut w) = (player(), world());
        assert!(!p.flying());
        let on = run("noclip", &mut p, &mut w);
        assert!(p.noclip());
        assert!(on[0].text().contains("on"));
        // From ordinary flight, too.
        p.toggle_noclip();
        assert!(!p.flying());
        p.set_flying(true);
        run("noclip", &mut p, &mut w);
        assert!(p.noclip());
        let off = run("noclip", &mut p, &mut w);
        assert!(!p.flying());
        assert!(!p.noclip());
        assert!(off[0].text().contains("off"));
    }

    #[test]
    fn a_world_without_a_cosmos_has_no_bodies_to_find() {
        let (mut p, mut w) = (player(), world());
        let at = p.position;
        let listed = joined(&run("bodies", &mut p, &mut w));
        assert!(listed.contains("no cosmos"), "{listed}");
        let tp = run("tp verdant", &mut p, &mut w);
        assert!(joined(&tp).contains("no cosmos"), "{}", joined(&tp));
        assert_eq!(tp[0].spans().next().unwrap().role, Role::Danger);
        assert_eq!(p.position, at);
        assert!(w.in_air(DVec3::splat(1.0e8)));
    }

    fn diffusion() -> World {
        use crate::render_config::RenderConfig;
        use crate::world::generation::WorldgenKind;
        World::with_kind(42, RenderConfig::default(), WorldgenKind::Diffusion, false)
    }

    #[test]
    fn bodies_lists_home_first_and_named_tp_lands_on_it() {
        use crate::world::terrain::cosmos::Kind;
        let (mut p, mut w) = (player(), diffusion());
        p.position = DVec3::new(0.5, 80.0, 0.5);
        let cosmos = w.terrain().cosmos().expect("diffusion has a cosmos");
        let home = cosmos.bodies().iter().copied().find(|b| b.kind == Kind::Home).unwrap();
        let verdant = cosmos.bodies().iter().copied().find(|b| b.kind == Kind::Verdant).unwrap();
        let moons: Vec<_> = cosmos.bodies().iter().copied().filter(|b| b.kind == Kind::Moon).collect();
        assert!(moons.len() >= 2, "catalog order has a second moon");

        let lines = run("bodies", &mut p, &mut w);
        let first = lines[0].text();
        assert!(first.starts_with("home 1"), "{first}");
        let home_at = landing(&home);
        assert!(
            first.contains(&format!("/tp {} {} {}", home_at[0], home_at[1], home_at[2])),
            "{first}"
        );
        let text = joined(&lines);
        assert!(text.contains("verdant 1"), "{text}");
        assert!(text.contains("moon 2"), "{text}");

        run("tp verdant", &mut p, &mut w);
        let want = landing_vec(&verdant);
        assert_eq!(p.position, want);
        assert!((p.position - verdant.centre_f()).length() <= verdant.reach());
        let pull = w.gravity_at(p.position).accel;
        let toward = (verdant.centre_f() - p.position).normalize();
        assert!(pull.normalize().dot(toward) > 0.99, "standing in Verdance's pull: {pull:?}");
        assert!(p.up().dot(-pull.normalize()) > 0.99, "up faces away from the pull");

        p.position = DVec3::ZERO;
        run("tp ver", &mut p, &mut w);
        assert_eq!(p.position, want, "a unique prefix selects the same body");

        run("tp moon 2", &mut p, &mut w);
        let moon = landing_vec(&moons[1]);
        assert_eq!(p.position, moon);
        assert_ne!(moon, landing_vec(&moons[0]));

        let at = p.position;
        let bad = run("tp nope", &mut p, &mut w);
        assert_eq!(p.position, at);
        assert_eq!(bad[0].spans().next().unwrap().role, Role::Danger);
        assert!(bad[0].text().contains("nope"), "{}", bad[0].text());
        let ambiguous = run("tp h", &mut p, &mut w);
        assert_eq!(p.position, at);
        assert!(ambiguous[0].text().contains("more than one"), "{}", ambiguous[0].text());
    }
}
