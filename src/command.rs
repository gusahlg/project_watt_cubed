//! Console command parsing and dispatch.
//!
//! [`execute`] takes one submitted line and returns the lines of output to show
//! in the console log. Adding a command is a single `match` arm — the dispatch is
//! deliberately tiny so it can grow into a richer command (or chat) system later.
use raylib::prelude::*;

use crate::block::Composition;
use crate::player::{PLAYER_HALF, Player};
use crate::world::World;

/// Run a console line against the game state, returning output lines for the log.
///
/// A leading `/` is optional, so both `tp 1 2 3` and `/tp 1 2 3` work. Commands
/// that only read the world (like `inspect`) take it by shared reference, so the
/// borrow sits happily alongside the `&mut Player`.
pub fn execute(line: &str, player: &mut Player, world: &World) -> Vec<String> {
    let line = line.strip_prefix('/').unwrap_or(line);
    let mut parts = line.split_whitespace();
    let Some(cmd) = parts.next() else {
        return Vec::new();
    };
    let args: Vec<&str> = parts.collect();

    match cmd {
        "tp" | "teleport" | "setpos" => teleport(&args, player),
        "pos" | "where" => vec![format!("position: {}", fmt_pos(player.position))],
        "inspect" | "look" => inspect(&args, player, world),
        "help" | "?" => help(),
        other => vec![format!("unknown command '{other}' — type 'help'")],
    }
}

/// `tp <x> <y> <z>` — move the player to absolute world coordinates.
fn teleport(args: &[&str], player: &mut Player) -> Vec<String> {
    if args.len() != 3 {
        return vec!["usage: tp <x> <y> <z>".to_string()];
    }
    let parsed: Result<Vec<f32>, _> = args.iter().map(|a| a.parse::<f32>()).collect();
    match parsed.as_deref() {
        Ok([x, y, z]) => {
            player.position = Vector3::new(*x, *y, *z);
            // Cancel any accumulated fall so the player doesn't rocket down on arrival.
            player.velocity_y = 0.0;
            vec![format!("teleported to {}", fmt_pos(player.position))]
        }
        _ => vec!["tp: x, y and z must be numbers".to_string()],
    }
}

/// `inspect [x y z]` — describe the block at a cell (default: the block under the
/// player's feet), showing what it's made of and the properties derived from that.
/// The in-game window onto the element/block system.
fn inspect(args: &[&str], player: &Player, world: &World) -> Vec<String> {
    let cell = match args {
        [] => {
            // The block supporting the player: directly below the feet. The small
            // bias keeps it stable when standing exactly on a block's top face.
            let p = player.position;
            (
                p.x.floor() as i32,
                (p.y - PLAYER_HALF.y - 0.1).floor() as i32,
                p.z.floor() as i32,
            )
        }
        [x, y, z] => match (x.parse(), y.parse(), z.parse()) {
            (Ok(x), Ok(y), Ok(z)) => (x, y, z),
            _ => return vec!["inspect: x, y and z must be integers".to_string()],
        },
        _ => return vec!["usage: inspect [<x> <y> <z>]".to_string()],
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
    out
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
            .0
            .iter()
            .map(|&(e, p)| format!("{}% {}", p, elements.get(e).name))
            .collect::<Vec<_>>()
            .join(", "),
        Composition::Computational(_) => "logic-gate components".to_string(),
    }
}

fn help() -> Vec<String> {
    vec![
        "commands (a leading '/' is optional):".to_string(),
        "  tp <x> <y> <z>       teleport to coordinates".to_string(),
        "  pos                  show current coordinates".to_string(),
        "  inspect [x y z]      describe a block's elements & properties".to_string(),
        "  help                 show this list".to_string(),
    ]
}

/// Format a position the same way the on-screen coordinate readout does.
fn fmt_pos(p: Vector3) -> String {
    format!("X {:.1}  Y {:.1}  Z {:.1}", p.x, p.y, p.z)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn player() -> Player {
        Player::new(Vector3::new(0.0, 0.0, 0.0))
    }

    /// A real generated world; cheap and GPU-free (meshes are uploaded separately).
    fn world() -> World {
        World::generate()
    }

    #[test]
    fn tp_sets_position_and_clears_fall() {
        let (mut p, w) = (player(), world());
        p.velocity_y = -50.0;
        let out = execute("tp 1.5 2 3", &mut p, &w);
        assert_eq!(p.position, Vector3::new(1.5, 2.0, 3.0));
        assert_eq!(p.velocity_y, 0.0);
        assert!(out[0].contains("teleported"));
    }

    #[test]
    fn leading_slash_is_optional() {
        let (mut p, w) = (player(), world());
        execute("/tp 4 5 6", &mut p, &w);
        assert_eq!(p.position, Vector3::new(4.0, 5.0, 6.0));
    }

    #[test]
    fn bad_args_do_not_move_the_player() {
        let (mut p, w) = (player(), world());
        execute("tp 1 two 3", &mut p, &w);
        assert_eq!(p.position, Vector3::new(0.0, 0.0, 0.0));
        execute("tp 1 2", &mut p, &w);
        assert_eq!(p.position, Vector3::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn unknown_command_reports_back() {
        let (mut p, w) = (player(), world());
        let out = execute("fly-to-moon", &mut p, &w);
        assert!(out[0].contains("unknown command"));
    }

    #[test]
    fn inspect_reports_elements_and_properties() {
        let (mut p, w) = (player(), world());
        // Deep underground is stone: a single Stone element with stone's properties.
        let out = execute("inspect 8 0 8", &mut p, &w);
        let text = out.join("\n");
        assert!(text.contains("Stone"), "should name the block: {text}");
        assert!(text.contains("made of: Stone"), "should list elements: {text}");
        assert!(text.contains("density"), "should show core properties: {text}");
    }

    #[test]
    fn inspect_above_world_is_air() {
        let (mut p, w) = (player(), world());
        let out = execute("inspect 8 60 8", &mut p, &w);
        assert!(out.join("\n").contains("air"));
    }
}
