//! Console command parsing and dispatch.
//!
//! [`execute`] takes one submitted line and returns the lines of output to show
//! in the console log. Adding a command is a single `match` arm — the dispatch is
//! deliberately tiny so it can grow into a richer command (or chat) system later.
use raylib::prelude::*;

use crate::player::Player;

/// Run a console line against the game state, returning output lines for the log.
///
/// A leading `/` is optional, so both `tp 1 2 3` and `/tp 1 2 3` work.
pub fn execute(line: &str, player: &mut Player) -> Vec<String> {
    let line = line.strip_prefix('/').unwrap_or(line);
    let mut parts = line.split_whitespace();
    let Some(cmd) = parts.next() else {
        return Vec::new();
    };
    let args: Vec<&str> = parts.collect();

    match cmd {
        "tp" | "teleport" | "setpos" => teleport(&args, player),
        "pos" | "where" => vec![format!("position: {}", fmt_pos(player.position))],
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

fn help() -> Vec<String> {
    vec![
        "commands (a leading '/' is optional):".to_string(),
        "  tp <x> <y> <z>  teleport to coordinates".to_string(),
        "  pos             show current coordinates".to_string(),
        "  help            show this list".to_string(),
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

    #[test]
    fn tp_sets_position_and_clears_fall() {
        let mut p = player();
        p.velocity_y = -50.0;
        let out = execute("tp 1.5 2 3", &mut p);
        assert_eq!(p.position, Vector3::new(1.5, 2.0, 3.0));
        assert_eq!(p.velocity_y, 0.0);
        assert!(out[0].contains("teleported"));
    }

    #[test]
    fn leading_slash_is_optional() {
        let mut p = player();
        execute("/tp 4 5 6", &mut p);
        assert_eq!(p.position, Vector3::new(4.0, 5.0, 6.0));
    }

    #[test]
    fn bad_args_do_not_move_the_player() {
        let mut p = player();
        execute("tp 1 two 3", &mut p);
        assert_eq!(p.position, Vector3::new(0.0, 0.0, 0.0));
        execute("tp 1 2", &mut p);
        assert_eq!(p.position, Vector3::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn unknown_command_reports_back() {
        let mut p = player();
        let out = execute("fly-to-moon", &mut p);
        assert!(out[0].contains("unknown command"));
    }
}
