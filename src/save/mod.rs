//! Saving and loading worlds, organized into layers. slot validates names and
//! metadata. format handles the binary codec and salvage of truncated files.
//! store manages atomic writes and the read ladder. autosave drives periodic
//! background saves. bridge is the only game-specific glue to SaveDoc.
//! A world is procedural, so saves are tiny: only player state, edited blocks,
//! and mod state are stored, using portable by-name form shared with the network.

pub mod autosave;
mod bridge;
pub mod format;
pub mod slot;
pub mod store;

pub use autosave::{Autosaver, Tick};
pub use bridge::{LoadReport, encode_current, load, save, unix_now};
pub use slot::{SaveError, SaveMeta, Slot, SlotId};
pub use store::{Source, fresh_id, list};

use crate::block::{AIR, BlockId, BlockRegistry, Composition};
use crate::world::World;

/// Serialize a block as portable element names shared with the network layer.
pub(crate) fn block_spec(world: &World, id: BlockId) -> String {
    registry_block_spec(world.registry(), id)
}

/// [`block_spec`] against a bare registry — the headless server and the
/// content fingerprint have no `World`.
pub(crate) fn registry_block_spec(registry: &BlockRegistry, id: BlockId) -> String {
    if id == AIR {
        return "air".to_string();
    }
    let elements = registry.elements();
    match &registry.block(id).composition {
        Composition::Natural(els) if els.is_empty() => "air".to_string(),
        Composition::Natural(els) => {
            let names: Vec<String> = els.iter().map(|&e| elements.get(e).name.to_string()).collect();
            format!("natural:{}", names.join(","))
        }
        Composition::Mixture(mix) => {
            let parts: Vec<String> = mix
                .parts()
                .iter()
                .map(|&(e, pct)| format!("{}={}", elements.get(e).name, pct))
                .collect();
            format!("mixture:{}", parts.join(";"))
        }
    }
}

/// Deserialize a block spec, registering into palette; inverse of block_spec().
pub(crate) fn parse_block(world: &mut World, spec: &str) -> BlockId {
    registry_parse_block(world.registry_mut(), spec)
}

/// [`parse_block`] against a bare registry. The server uses this to VALIDATE
/// and canonicalize incoming edit specs with the exact rules clients apply,
/// then re-serializes via [`registry_block_spec`] — so an edit overlay never
/// stores two strings for one block, and junk never interns at all.
pub(crate) fn registry_parse_block(registry: &mut BlockRegistry, spec: &str) -> BlockId {
    if spec == "air" {
        return AIR;
    }
    if let Some(rest) = spec.strip_prefix("natural:") {
        // Any unknown element rejects the spec to prevent mismatches with the sender.
        let mut ids = Vec::new();
        for name in rest.split(',') {
            match registry.elements().id_by_name(name) {
                Some(id) => ids.push(id),
                None => return AIR,
            }
        }
        return crate::block::crafting::craft_natural(registry, &ids).unwrap_or(AIR);
    }
    if let Some(rest) = spec.strip_prefix("mixture:") {
        let mut parts = Vec::new();
        for entry in rest.split(';') {
            let Some((name, pct)) = entry.split_once('=') else { return AIR };
            let Some(id) = registry.elements().id_by_name(name) else { return AIR };
            let Ok(pct) = pct.parse::<u8>() else { return AIR };
            parts.push((id, pct));
        }
        let Ok(composition) = crate::block::Composition::mixture(&parts) else {
            return AIR;
        };
        if let Some(existing) = registry.lookup(&composition) {
            return existing;
        }
        if registry.at_capacity() {
            return AIR;
        }
        return registry.mixture(&parts).unwrap_or(AIR);
    }
    AIR
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::element::El;
    use crate::mods::Mods;
    use crate::player::Player;
    use std::fs;
    use voxel_engine::DVec3;

    fn slot(name: &str) -> SlotId {
        let id = SlotId::new(name).unwrap();
        let _ = fs::remove_file(format!("saves/{id}.save"));
        let _ = fs::remove_file(format!("saves/{id}.save.bak"));
        id
    }

    fn cleanup(id: &SlotId) {
        let _ = fs::remove_file(format!("saves/{id}.save"));
        let _ = fs::remove_file(format!("saves/{id}.save.bak"));
    }

    fn meta(name: &str) -> SaveMeta {
        SaveMeta {
            name: name.to_string(),
            seed: 0, // stamped from the world on save
            created: 1_770_000_000,
            last_played: 0,
            playtime_secs: 42,
            edit_count: 0,
        }
    }

    /// Round-trip property: EVERY registered block — the full compiled worldgen
    /// palette plus crafted naturals and mixtures — serializes to a spec that
    /// parses back to the SAME id.
    #[test]
    fn block_specs_round_trip_for_every_supported_composition() {
        let mut world = World::new(11);
        // Add a crafted natural (duplicated listing) and a mixture on top of
        // the compiled palette.
        let dup = crate::block::crafting::craft_natural(
            world.registry_mut(),
            &[El::Copper.id(), El::Copper.id(), El::Glass.id()],
        )
        .unwrap();
        let mix = world.registry_mut().mixture(&[(El::Soil.id(), 70), (El::Clay.id(), 30)]).unwrap();
        let _ = (dup, mix);

        for i in 0..world.registry().block_count() {
            let id = BlockId(i as u16);
            let spec = block_spec(&world, id);
            assert_eq!(
                parse_block(&mut world, &spec),
                id,
                "spec '{spec}' must parse back to block #{i}"
            );
        }
    }

    #[test]
    fn round_trip_preserves_seed_player_edits_and_mod_state() {
        let id = slot("__unit_test_round_trip__");

        let mut world = World::new(4242);
        let (bx, bz) = (8, 8);
        let by = (0..64)
            .rev()
            .find(|&y| world.is_solid(bx, y, bz))
            .unwrap();
        world.set_block(bx, by, bz, AIR);

        let mut player = Player::new(DVec3::new(1.0, 2.0, 3.0));
        player.orientation.yaw = 0.5;
        player.orientation.pitch = -0.25;
        player.set_flying(true);

        // Give the mods some state to persist (elements land in the inventory).
        let mut mods = Mods::with_defaults();
        mods.on_block_break(&[El::Stone.id(), El::Iron.id(), El::Stone.id()], &world);
        let states_before = mods.save_states(&world);

        save(&id, &world, &player, &mods, meta("round trip")).unwrap();

        let mut fresh_mods = Mods::with_defaults();
        let (loaded_world, loaded_player, loaded_meta, report) =
            load(&id, &mut fresh_mods).unwrap();

        assert_eq!(loaded_world.seed(), 4242);
        assert_eq!(loaded_meta.seed, 4242, "seed is stamped into the header");
        assert_eq!(loaded_meta.name, "round trip");
        assert_eq!(loaded_meta.playtime_secs, 42);
        assert_eq!(loaded_player.position, DVec3::new(1.0, 2.0, 3.0));
        assert_eq!(loaded_player.orientation.yaw, 0.5);
        assert_eq!(loaded_player.orientation.pitch, -0.25);
        assert!(loaded_player.flying());
        assert_eq!(loaded_world.block_at(bx, by, bz), AIR, "broken block stays broken");
        assert_eq!(report.source, Source::Live);
        assert!(report.salvage.is_none());
        assert_eq!(
            fresh_mods.save_states(&loaded_world),
            states_before,
            "mod state survives the round trip"
        );

        cleanup(&id);
    }

    #[test]
    fn far_positions_round_trip_bit_exactly() {
        // f64 precision at 1e8 can't be approximated as f32 without losing 4+ blocks.
        let id = slot("__unit_test_far_pos__");

        let world = World::new(77);
        let pos = DVec3::new(1.0e8 + 0.123456789, 61.5, -(1.0e9 - 42.25));
        let mut player = Player::new(pos);
        player.orientation.yaw = 1.25;
        player.orientation.pitch = -0.5;
        let mut mods = Mods::with_defaults();
        save(&id, &world, &player, &mods, meta("far")).unwrap();

        let (_, loaded, _, _) = load(&id, &mut mods).unwrap();
        assert_eq!(loaded.position.x.to_bits(), pos.x.to_bits());
        assert_eq!(loaded.position.y.to_bits(), pos.y.to_bits());
        assert_eq!(loaded.position.z.to_bits(), pos.z.to_bits());
        assert_eq!(loaded.orientation.yaw, 1.25);
        assert_eq!(loaded.orientation.pitch, -0.5);

        cleanup(&id);
    }

    #[test]
    fn empty_world_round_trips() {
        let id = slot("__unit_test_empty__");

        let world = World::new(1234);
        let player = Player::new(DVec3::new(0.0, 40.0, 0.0));
        let mut mods = Mods::with_defaults();
        save(&id, &world, &player, &mods, meta("empty")).unwrap();

        let (loaded_world, loaded_player, _, _) = load(&id, &mut mods).unwrap();
        assert_eq!(loaded_world.seed(), 1234);
        assert_eq!(loaded_player.position, DVec3::new(0.0, 40.0, 0.0));
        assert_eq!(loaded_world.edits().count(), 0);

        cleanup(&id);
    }

    #[test]
    fn corrupt_live_file_falls_back_to_backup() {
        let id = slot("__unit_test_ladder__");

        let mut world = World::new(9);
        world.set_block(1, 200, 1, AIR);
        let player = Player::new(DVec3::new(0.0, 40.0, 0.0));
        let mut mods = Mods::with_defaults();
        save(&id, &world, &player, &mods, meta("v1")).unwrap();
        save(&id, &world, &player, &mods, meta("v2")).unwrap(); // rotates v1 to .bak
        fs::write(format!("saves/{id}.save"), b"NOPE not a save").unwrap();

        let (loaded_world, _, loaded_meta, report) = load(&id, &mut mods).unwrap();
        assert_eq!(report.source, Source::Backup);
        assert_eq!(loaded_meta.name, "v1");
        assert_eq!(loaded_world.block_at(1, 200, 1), AIR);

        cleanup(&id);
    }
}
