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
pub use bridge::{LoadReport, SaveSnapshot, encode_current, load, save, snapshot, unix_now};
pub use slot::{SaveError, SaveMeta, Slot, SlotId};
pub use store::{Source, fresh_id, list, write_atomic, write_atomic_file};

use crate::block::element::ElementId;
use crate::block::{AIR, BlockId, BlockRegistry, Composition};

/// Spec string from a composition and an element-name lookup. Shared by the
/// live registry path and the autosave snapshot so both emit identical bytes.
pub(crate) fn composition_spec<'a>(
    composition: &Composition,
    element_name: impl Fn(ElementId) -> &'a str,
) -> String {
    match composition {
        Composition::Natural(els) if els.is_empty() => "air".to_string(),
        Composition::Natural(els) => {
            let names: Vec<&str> = els.iter().map(|&e| element_name(e)).collect();
            format!("natural:{}", names.join(","))
        }
        Composition::Mixture(mix) => {
            let parts: Vec<String> = mix
                .parts()
                .iter()
                .map(|&(e, pct)| format!("{}={}", element_name(e), pct))
                .collect();
            format!("mixture:{}", parts.join(";"))
        }
    }
}

/// Serialize a block as portable element names shared with the network layer.
pub(crate) fn block_spec(registry: &BlockRegistry, id: BlockId) -> String {
    if id == AIR {
        return "air".to_string();
    }
    let elements = registry.elements();
    composition_spec(&registry.block(id).composition, |e| elements.get(e).name.as_ref())
}

/// Deserialize a block spec, registering into palette; inverse of block_spec().
/// The server uses this to VALIDATE and canonicalize incoming edit specs with
/// the exact rules clients apply, then re-serializes via [`block_spec`] — so an
/// edit overlay never stores two strings for one block, and junk never interns.
pub(crate) fn parse_block(registry: &mut BlockRegistry, spec: &str) -> BlockId {
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
    use super::bridge::from_doc;
    use super::format::{PlayerState, SaveDoc, WorldgenStamp};
    use crate::block::element::El;
    use crate::mods::Mods;
    use crate::player::Player;
    use crate::world::World;
    use crate::world::chunk::CHUNK_SIZE;
    use crate::world::diffusion::DiffusionCfg;
    use crate::world::generation::WorldgenKind;
    use std::fs;
    use voxel_engine::DVec3;

    fn save_file(id: &SlotId) -> std::path::PathBuf {
        crate::paths::Paths::get().data.join(format!("{id}.save"))
    }

    fn bak_file(id: &SlotId) -> std::path::PathBuf {
        crate::paths::Paths::get().data.join(format!("{id}.save.bak"))
    }

    fn make_world(seed: i64, kind: WorldgenKind, cfg: DiffusionCfg) -> World {
        World::with_kind_cfg(
            seed,
            crate::render_config::RenderConfig::default(),
            kind,
            cfg,
            true,
        )
    }

    fn slot(name: &str) -> SlotId {
        let id = SlotId::new(name).unwrap();
        let _ = fs::remove_file(save_file(&id));
        let _ = fs::remove_file(bak_file(&id));
        id
    }

    fn cleanup(id: &SlotId) {
        let _ = fs::remove_file(save_file(id));
        let _ = fs::remove_file(bak_file(id));
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
            let spec = block_spec(world.registry(), id);
            assert_eq!(
                parse_block(world.registry_mut(), &spec),
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

        player.stash.add(&[El::Stone.id(), El::Iron.id(), El::Stone.id()]);
        let mut mods = Mods::with_defaults();
        mods.load_state("Crafting", "*Stone=1", &mut world);
        let states_before = mods.save_states(&world);

        save(&id, &world, &player, &mods, meta("round trip")).unwrap();

        let mut fresh_mods = Mods::with_defaults();
        let (loaded_world, loaded_player, loaded_meta, report) =
            load(&id, &mut fresh_mods, make_world).unwrap();

        assert_eq!(loaded_world.seed(), 4242);
        assert_eq!(loaded_meta.seed, 4242, "seed is stamped into the header");
        assert_eq!(loaded_meta.name, "round trip");
        assert_eq!(loaded_meta.playtime_secs, 42);
        assert_eq!(loaded_player.position, DVec3::new(1.0, 2.0, 3.0));
        assert_eq!(loaded_player.orientation.yaw, 0.5);
        assert_eq!(loaded_player.orientation.pitch, -0.25);
        assert!(loaded_player.flying());
        assert_eq!(loaded_player.stash.total(), 3);
        assert_eq!(loaded_player.stash.count(El::Stone.id()), 2);
        assert_eq!(loaded_player.stash.count(El::Iron.id()), 1);
        let saved_bytes = fs::read(save_file(&id)).unwrap();
        assert_eq!(
            u16::from_le_bytes(saved_bytes[4..6].try_into().unwrap()),
            format::VERSION,
            "new documents write save format v{}",
            format::VERSION
        );
        assert_eq!(loaded_world.block_at(bx, by, bz), AIR, "broken block stays broken");
        assert_eq!(report.source, Source::Live);
        assert!(report.salvage.is_none());
        assert_eq!(
            fresh_mods.save_states(&loaded_world),
            states_before,
            "mod state survives the round trip"
        );
        assert!(
            states_before.iter().all(|(k, _)| k != "inventory"),
            "the stash is core player state, not an inventory save line"
        );

        cleanup(&id);
    }

    fn bare_doc() -> SaveDoc {
        SaveDoc {
            meta: meta("stash"),
            worldgen_version: crate::world::placement::WORLDGEN_VERSION,
            worldgen: WorldgenStamp::default(),
            player: PlayerState {
                pos: [0.0, 40.0, 0.0],
                yaw: 0.0,
                pitch: 0.0,
                flying: false,
                noclip: false,
                stash: None,
            },
            specs: vec![],
            edits: vec![],
            mods: vec![],
        }
    }

    #[test]
    fn old_inventory_mod_line_migrates_into_the_core_stash() {
        let mut doc = bare_doc();
        doc.mods
            .push(("inventory".into(), "v1;Stone,Stone,Soil".into()));
        let mut mods = Mods::with_defaults();
        let (_, player, _) = from_doc(doc, &mut mods, make_world);
        assert_eq!(player.stash.total(), 3);
        assert_eq!(player.stash.count(El::Stone.id()), 2);
        assert_eq!(player.stash.count(El::Soil.id()), 1);
    }

    #[test]
    fn unprefixed_inventory_line_still_migrates() {
        let mut doc = bare_doc();
        doc.mods.push(("Inventory".into(), "Iron,Iron".into()));
        let mut mods = Mods::with_defaults();
        let (_, player, _) = from_doc(doc, &mut mods, make_world);
        assert_eq!(player.stash.total(), 2);
        assert_eq!(player.stash.count(El::Iron.id()), 2);
    }

    #[test]
    fn v6_on_disk_inventory_line_migrates_through_decode() {
        let mut doc = bare_doc();
        doc.mods
            .push(("inventory".into(), "v1;Stone,Stone,Soil".into()));
        let v7 = format::encode(&doc).unwrap();
        // v6 player records end at the flags byte; drop the v7 stash blob.
        let start = format::HEADER_LEN + 33;
        let len = u16::from_le_bytes(v7[start..start + 2].try_into().unwrap()) as usize;
        let mut v6 = Vec::with_capacity(v7.len() - 2 - len);
        v6.extend_from_slice(&v7[..start]);
        v6.extend_from_slice(&v7[start + 2 + len..]);
        v6[4..6].copy_from_slice(&6u16.to_le_bytes());

        let decoded = match format::decode(&v6).unwrap() {
            format::Decoded::Intact(doc) => doc,
            format::Decoded::Salvaged { .. } => panic!("v6 splice must decode intact"),
        };
        assert!(decoded.player.stash.is_none());
        let mut mods = Mods::with_defaults();
        let (_, player, _) = from_doc(decoded, &mut mods, make_world);
        assert_eq!(player.stash.total(), 3);
        assert_eq!(player.stash.count(El::Stone.id()), 2);
        assert_eq!(player.stash.count(El::Soil.id()), 1);
    }

    #[test]
    fn player_stash_field_wins_over_an_old_inventory_line() {
        let mut doc = bare_doc();
        doc.player.stash = Some(vec![("Copper".into(), 1)]);
        doc.mods
            .push(("inventory".into(), "v1;Stone,Stone".into()));
        let mut mods = Mods::with_defaults();
        let (_, player, _) = from_doc(doc, &mut mods, make_world);
        assert_eq!(player.stash.total(), 1);
        assert_eq!(player.stash.count(El::Copper.id()), 1);
        assert_eq!(player.stash.count(El::Stone.id()), 0);
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

        let (_, loaded, _, _) = load(&id, &mut mods, make_world).unwrap();
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

        let (loaded_world, loaded_player, _, _) = load(&id, &mut mods, make_world).unwrap();
        assert_eq!(loaded_world.seed(), 1234);
        assert_eq!(loaded_player.position, DVec3::new(0.0, 40.0, 0.0));
        assert_eq!(loaded_player.stash.total(), 0);
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
        fs::write(save_file(&id), b"NOPE not a save").unwrap();

        let (loaded_world, _, loaded_meta, report) = load(&id, &mut mods, make_world).unwrap();
        assert_eq!(report.source, Source::Backup);
        assert_eq!(loaded_meta.name, "v1");
        assert_eq!(loaded_world.block_at(1, 200, 1), AIR);

        cleanup(&id);
    }

    #[test]
    fn mod_state_unknown_ids_are_ignored_duplicates_last_win() {
        let blank_player = PlayerState {
            pos: [0.0, 40.0, 0.0],
            yaw: 0.0,
            pitch: 0.0,
            flying: false,
            noclip: false,
            stash: Some(vec![]),
        };
        let doc = SaveDoc {
            worldgen_version: crate::world::placement::WORLDGEN_VERSION,
            worldgen: WorldgenStamp::default(),
            meta: meta("mods"),
            player: blank_player.clone(),
            specs: vec![],
            edits: vec![],
            mods: vec![
                ("Crafting".into(), "*IronVein=1".into()),
                ("no-such-mod".into(), "ignored".into()),
                ("Crafting".into(), "*IronVein=2".into()),
            ],
        };
        let mut mods = Mods::with_defaults();
        let (world, _, _) = super::bridge::from_doc(doc, &mut mods, make_world);
        let states = mods.save_states(&world);
        let craft = states.iter().find(|(n, _)| n == "crafting").map(|(_, d)| d.as_str());
        assert_eq!(craft, Some("v1;*Stone+Iron=2"), "duplicate mod lines: last wins");
        assert!(states.iter().all(|(n, _)| n != "no-such-mod"));
    }

    /// Independent of `SaveSnapshot::to_doc`: walk `World::edits` and
    /// `block_spec` the way encode used to on the main thread.
    fn doc_by_walking_overlay(
        world: &World,
        player: &Player,
        mods: &Mods,
        mut meta: SaveMeta,
    ) -> super::format::SaveDoc {
        use super::format::{Edit, PlayerState, SaveDoc};
        let mut specs: Vec<String> = Vec::new();
        let mut index_of: std::collections::HashMap<String, u16> = std::collections::HashMap::new();
        let mut edits: Vec<Edit> = Vec::new();
        for ((x, y, z), id) in world.edits() {
            let spec = block_spec(world.registry(), id);
            let index = match index_of.get(&spec) {
                Some(&index) => index,
                None => {
                    let index = u16::try_from(specs.len()).unwrap();
                    index_of.insert(spec.clone(), index);
                    specs.push(spec);
                    index
                }
            };
            edits.push(Edit { x, y, z, spec: index });
        }
        meta.edit_count = u32::try_from(edits.len()).unwrap();
        SaveDoc {
            meta,
            worldgen_version: crate::world::placement::WORLDGEN_VERSION,
            worldgen: WorldgenStamp {
                kind: world.worldgen().wire(),
                tile: world.diffusion_cfg().tile,
                stride: world.diffusion_cfg().stride,
                phases: world.diffusion_cfg().phases,
                relief: world.diffusion_cfg().relief,
            },
            player: PlayerState {
                pos: [player.position.x, player.position.y, player.position.z],
                yaw: player.orientation.yaw,
                pitch: player.orientation.pitch,
                flying: player.flying(),
                noclip: player.noclip(),
                stash: Some(player.stash.to_portable(|id| {
                    world.registry().elements().get(id).name.as_ref()
                })),
            },
            specs,
            edits,
            mods: mods.save_states(world),
        }
    }

    #[test]
    fn snapshot_encodes_byte_identical_to_walking_the_overlay() {
        let mut world = World::new(4242);
        let (bx, bz) = (8, 8);
        let by = (0..64)
            .rev()
            .find(|&y| world.is_solid(bx, y, bz))
            .unwrap();
        world.set_block(bx, by, bz, AIR);
        let mix = world
            .registry_mut()
            .mixture(&[(El::Soil.id(), 70), (El::Clay.id(), 30)])
            .unwrap();
        world.set_block(bx, by + 1, bz, mix);

        let mut player = Player::new(DVec3::new(1.0, 2.0, 3.0));
        player.orientation.yaw = 0.5;
        player.orientation.pitch = -0.25;
        player.set_flying(true);
        player.stash.add(&[El::Stone.id(), El::Iron.id(), El::Stone.id()]);
        let mut mods = Mods::with_defaults();
        mods.load_state("Crafting", "*Stone=1", &mut world);

        let snap = snapshot(&world, &player, &mods, meta("snap"));
        let snap_doc = snap.to_doc().unwrap();
        let walked = doc_by_walking_overlay(&world, &player, &mods, snap_doc.meta.clone());
        assert_eq!(snap_doc, walked, "snapshot document must match the old overlay walk");
        assert_eq!(
            snap.encode().unwrap(),
            super::format::encode(&walked).unwrap(),
            "snapshot bytes must match the old overlay walk"
        );
    }

    fn dump_chunk(world: &World, cx: i32, cy: i32, cz: i32) -> Vec<crate::block::BlockId> {
        let s = CHUNK_SIZE as i32;
        let mut out = Vec::with_capacity(CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE);
        for ly in 0..s {
            for lz in 0..s {
                for lx in 0..s {
                    out.push(world.block_at(cx * s + lx, cy * s + ly, cz * s + lz));
                }
            }
        }
        out
    }

    #[test]
    fn diffusion_world_round_trips_kind_cfg_and_generated_chunks() {
        let id = slot("__unit_test_diffusion_round_trip__");
        let cfg = DiffusionCfg {
            tile: 64,
            stride: 16,
            phases: 4,
            relief: 1.5,
        };
        let world = make_world(99, WorldgenKind::Diffusion, cfg);
        assert_eq!(world.worldgen(), WorldgenKind::Diffusion);
        let cy = world.surface_y(0, 0).div_euclid(CHUNK_SIZE as i32);
        let chunks = [(0, cy, 0), (1, cy, 0), (0, cy, 1)];
        let before: Vec<_> = chunks
            .iter()
            .map(|&c| dump_chunk(&world, c.0, c.1, c.2))
            .collect();
        assert!(
            before.iter().any(|c| c.iter().any(|&id| id != AIR)),
            "pregenerated origin must contain terrain"
        );

        let player = Player::new(DVec3::new(0.0, 40.0, 0.0));
        // Diffusion mod stays OFF: the save header, not the mod flag, decides
        // the generator on load.
        let mut mods = Mods::with_defaults();
        assert_eq!(mods.worldgen_kind(), WorldgenKind::Classic);
        save(&id, &world, &player, &mods, meta("diffusion")).unwrap();

        let (loaded, _, _, _) = load(&id, &mut mods, make_world).unwrap();
        assert_eq!(loaded.worldgen(), WorldgenKind::Diffusion);
        assert_eq!(loaded.diffusion_cfg(), cfg.clamp());
        assert_eq!(
            mods.worldgen_kind(),
            WorldgenKind::Classic,
            "loading a diffusion world must not flip the mod's enabled flag"
        );
        for (i, &(cx, cy, cz)) in chunks.iter().enumerate() {
            assert_eq!(
                dump_chunk(&loaded, cx, cy, cz),
                before[i],
                "chunk {cx},{cy},{cz} must regenerate identically"
            );
        }

        cleanup(&id);
    }

    #[test]
    fn classic_save_still_loads_as_classic() {
        let id = slot("__unit_test_classic_kind__");
        let world = World::new(7);
        assert_eq!(world.worldgen(), WorldgenKind::Classic);
        let player = Player::new(DVec3::new(0.0, 40.0, 0.0));
        let mut mods = Mods::with_defaults();
        mods.set_enabled("diffusion", true);
        save(&id, &world, &player, &mods, meta("classic")).unwrap();

        let (loaded, _, _, _) = load(&id, &mut mods, make_world).unwrap();
        assert_eq!(loaded.worldgen(), WorldgenKind::Classic);
        assert_eq!(loaded.worldgen_kind(), "classic");

        cleanup(&id);
    }
}
