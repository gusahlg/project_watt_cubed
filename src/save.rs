//! Saving and loading worlds, in a plain-text, dependency-free format.
//!
//! A world is procedural, so a save is tiny: the seed regenerates the terrain, and
//! only the player's state, the blocks they've changed, and each mod's own state
//! are stored. Edits and inventories are written by element/block *name*, not by
//! registry id, so a save stays valid even as mods shift ids around — the multiplayer
//! and modding future both need that portability.
use std::fs;
use std::io;
use std::path::PathBuf;

use raylib::prelude::*;

use crate::block::{AIR, BlockId, Composition};
use crate::mods::Mods;
use crate::player::Player;
use crate::world::World;

const SAVE_VERSION: u32 = 1;

/// Directory holding all saves (relative to the working directory).
fn saves_dir() -> PathBuf {
    PathBuf::from("saves")
}

/// The file backing a named save.
fn save_path(name: &str) -> PathBuf {
    saves_dir().join(format!("{name}.save"))
}

/// The names of all existing saves, newest filesystem entries last.
pub fn list_saves() -> Vec<String> {
    let mut names = Vec::new();
    if let Ok(entries) = fs::read_dir(saves_dir()) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("save") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    names.push(stem.to_string());
                }
            }
        }
    }
    names.sort();
    names
}

/// A fresh, unused save name (`world`, then `world-2`, `world-3`, …).
pub fn next_new_name() -> String {
    if !save_path("world").exists() {
        return "world".to_string();
    }
    (2..)
        .map(|n| format!("world-{n}"))
        .find(|name| !save_path(name).exists())
        .expect("an unused save name exists")
}

/// Write a world, player, and mod state to the named save.
pub fn save(name: &str, world: &World, player: &Player, mods: &Mods) -> io::Result<()> {
    fs::create_dir_all(saves_dir())?;

    let mut out = String::new();
    out.push_str(&format!("watt-cubed save {SAVE_VERSION}\n"));
    out.push_str(&format!("seed {}\n", world.seed()));

    let p = player.position;
    out.push_str(&format!(
        "player {} {} {} {} {} {}\n",
        p.x, p.y, p.z, player.yaw, player.pitch, player.fly as u8
    ));

    for ((x, y, z), id) in world.edits() {
        out.push_str(&format!("edit {x} {y} {z} {}\n", block_spec(world, id)));
    }

    for (mod_name, data) in mods.save_states(world) {
        out.push_str(&format!("mod {mod_name} {data}\n"));
    }

    fs::write(save_path(name), out)
}

/// Load the named save into a ready-to-play world and player, restoring mod state
/// into `mods`.
pub fn load(name: &str, mods: &mut Mods) -> io::Result<(World, Player)> {
    let text = fs::read_to_string(save_path(name))?;
    let mut lines = text.lines();

    // Header + seed drive world construction; everything else layers on top.
    let _header = lines.next();
    let seed = lines
        .clone()
        .find_map(|l| l.strip_prefix("seed ").and_then(|s| s.trim().parse::<i64>().ok()))
        .unwrap_or(crate::world::DEFAULT_SEED);
    let mut world = World::new(seed);
    let mut player = Player::new(Vector3::zero());

    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("player ") {
            parse_player(&mut player, rest);
        } else if let Some(rest) = line.strip_prefix("edit ") {
            parse_edit(&mut world, rest);
        } else if let Some(rest) = line.strip_prefix("mod ") {
            let (mod_name, data) = rest.split_once(' ').unwrap_or((rest, ""));
            mods.load_state(mod_name, data, &world);
        }
    }

    Ok((world, player))
}

/// Restore player fields from a `player` line's arguments.
fn parse_player(player: &mut Player, args: &str) {
    let f: Vec<f32> = args.split_whitespace().filter_map(|v| v.parse().ok()).collect();
    if f.len() == 6 {
        player.position = Vector3::new(f[0], f[1], f[2]);
        player.yaw = f[3];
        player.pitch = f[4];
        player.fly = f[5] != 0.0;
        player.velocity_y = 0.0;
        player.on_ground = false;
    }
}

/// Apply one `edit` line: `x y z <block-spec>`.
fn parse_edit(world: &mut World, args: &str) {
    let mut parts = args.splitn(4, ' ');
    let (Some(x), Some(y), Some(z), Some(spec)) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return;
    };
    let (Ok(x), Ok(y), Ok(z)) = (x.parse::<i32>(), y.parse::<i32>(), z.parse::<i32>()) else {
        return;
    };
    let id = parse_block(world, spec);
    world.set_block(x, y, z, id);
}

/// Describe a block compactly by composition, using portable element names. Shared
/// with the network layer, which sends edits in exactly this portable form so ids
/// never have to agree between machines.
pub(crate) fn block_spec(world: &World, id: BlockId) -> String {
    if id == AIR {
        return "air".to_string();
    }
    let elements = world.registry().elements();
    match &world.registry().block(id).composition {
        Composition::Natural(els) if els.is_empty() => "air".to_string(),
        Composition::Natural(els) => {
            let names: Vec<String> = els.iter().map(|&e| elements.get(e).name.to_string()).collect();
            format!("natural:{}", names.join(","))
        }
        Composition::Mixture(mix) | Composition::Configuration { mix, .. } => {
            let parts: Vec<String> = mix
                .0
                .iter()
                .map(|&(e, pct)| format!("{}={}", elements.get(e).name, pct))
                .collect();
            format!("mixture:{}", parts.join(";"))
        }
        Composition::Computational(_) => "air".to_string(), // not yet reconstructable
    }
}

/// Rebuild a block from a spec, registering it into the world's palette as needed.
/// The inverse of [`block_spec`]; shared with the network layer.
pub(crate) fn parse_block(world: &mut World, spec: &str) -> BlockId {
    if spec == "air" {
        return AIR;
    }
    if let Some(rest) = spec.strip_prefix("natural:") {
        let ids: Vec<_> = rest
            .split(',')
            .filter_map(|n| world.registry().elements().id_by_name(n))
            .collect();
        return if ids.is_empty() {
            AIR
        } else {
            world.registry_mut().natural(&ids)
        };
    }
    if let Some(rest) = spec.strip_prefix("mixture:") {
        let parts: Vec<_> = rest
            .split(';')
            .filter_map(|entry| {
                let (name, pct) = entry.split_once('=')?;
                let id = world.registry().elements().id_by_name(name)?;
                Some((id, pct.parse::<u8>().ok()?))
            })
            .collect();
        return world.registry_mut().mixture(&parts).unwrap_or(AIR);
    }
    AIR
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_preserves_seed_player_and_edits() {
        let name = "__unit_test__";
        let _ = fs::remove_file(save_path(name));

        // Build a world, move the player, and break the surface block at a column.
        let mut world = World::new(4242);
        let (bx, bz) = (8, 8);
        let by = (0..64)
            .rev()
            .find(|&y| world.is_solid(bx, y, bz))
            .unwrap();
        world.set_block(bx, by, bz, AIR);

        let mut player = Player::new(Vector3::new(1.0, 2.0, 3.0));
        player.yaw = 0.5;
        player.pitch = -0.25;
        let mut mods = Mods::with_defaults();

        save(name, &world, &player, &mods).unwrap();
        let (loaded_world, loaded_player) = load(name, &mut mods).unwrap();

        assert_eq!(loaded_world.seed(), 4242);
        assert_eq!(loaded_player.position, Vector3::new(1.0, 2.0, 3.0));
        assert_eq!(loaded_player.yaw, 0.5);
        assert_eq!(loaded_world.block_at(bx, by, bz), AIR, "broken block stays broken");

        let _ = fs::remove_file(save_path(name));
    }
}
