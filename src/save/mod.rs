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
pub use bridge::{encode_current, load, save, snapshot, unix_now};
pub use slot::{SaveError, SaveMeta, Slot, SlotId};
pub use store::{Source, fresh_id, list, write_atomic_file};
pub(crate) use store::log_fs_err;

use crate::block::{AIR, BlockId, BlockRegistry};

/// Serialize a block as the registry's portable spec, shared with the network.
pub(crate) fn block_spec(registry: &BlockRegistry, id: BlockId) -> String {
    registry.spec(id)
}

/// Deserialize a block spec, interning into the table; inverse of [`block_spec`].
/// Unknown or legacy specs become [`AIR`] — the caller (save load, the wire)
/// decides whether that is a notice or a reject.
pub(crate) fn parse_block(registry: &mut BlockRegistry, spec: &str) -> BlockId {
    registry.parse_spec(spec).unwrap_or(AIR)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::bridge::{from_doc, unknown_material_notice};
    use super::format::{PlayerState, SaveDoc, WorldgenStamp};
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
        let extra = material::Configuration::new(vec![
            material::Element::new([1, 2, 3, 4]),
            material::Element::new([1, 2, 3, 4]),
            material::Element::new([9, 8, 7, 6]),
        ])
        .unwrap();
        world.registry_mut().intern(&extra).unwrap();

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

        let rock = world.registry().id_by_label("rock").unwrap();
        let soil = world.registry().id_by_label("soil").unwrap();
        player.stash.add(rock, 2);
        player.stash.add(soil, 1);
        let mut mods = Mods::with_defaults();
        let rock_spec = world.registry().spec(rock);
        mods.load_state("Crafting", &format!("*{rock_spec}=1"), &mut world);
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
        assert_eq!(loaded_player.stash.count(rock), 2);
        assert_eq!(loaded_player.stash.count(soil), 1);
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
            law_stamp: material::Law::v0().stamp(),
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
    fn legacy_inventory_mod_line_is_ignored() {
        let mut doc = bare_doc();
        doc.mods
            .push(("inventory".into(), "v1;Stone,Stone,Soil".into()));
        let mut mods = Mods::with_defaults();
        let (_, player, _) = from_doc(doc, &mut mods, make_world).unwrap();
        assert_eq!(player.stash.total(), 0);
    }

    #[test]
    fn unknown_stash_specs_are_skipped() {
        let mut doc = bare_doc();
        doc.player.stash = Some(vec![("natural:Stone".into(), 2), ("air".into(), 1)]);
        let mut mods = Mods::with_defaults();
        let (_, player, _) = from_doc(doc, &mut mods, make_world).unwrap();
        assert_eq!(player.stash.total(), 1);
        assert_eq!(player.stash.count(AIR), 1);
    }

    #[test]
    fn unknown_holdings_fold_into_one_load_notice() {
        assert_eq!(unknown_material_notice(0, 0), None);
        assert_eq!(
            unknown_material_notice(3, 0).as_deref(),
            Some("save predates the material model; 3 edits of unknown materials became air")
        );
        assert_eq!(
            unknown_material_notice(0, 2).as_deref(),
            Some("2 holdings of unknown materials were dropped")
        );
        assert_eq!(
            unknown_material_notice(4, 1).as_deref(),
            Some(
                "save predates the material model; 4 edits of unknown materials became air; 1 holdings of unknown materials were dropped"
            )
        );

        let mut doc = bare_doc();
        doc.player.stash = Some(vec![("natural:Stone".into(), 2), ("air".into(), 1)]);
        doc.mods
            .push(("crafting".into(), "v1;natural:Iron=4".into()));
        let mut mods = Mods::with_defaults();
        let (world, player, _) = from_doc(doc, &mut mods, make_world).unwrap();
        assert_eq!(player.stash.total(), 1);
        assert!(
            mods.save_states(&world)
                .iter()
                .all(|(n, d)| n != "crafting" || !d.contains("Iron")),
            "unknown pouch specs are dropped"
        );
    }

    #[test]
    fn stash_specs_round_trip_through_save() {
        let world = World::new(3);
        let rock = world.registry().id_by_label("rock").unwrap();
        let spec = world.registry().spec(rock);
        let mut doc = bare_doc();
        doc.player.stash = Some(vec![(spec, 4)]);
        let mut mods = Mods::with_defaults();
        let (_, player, _) = from_doc(doc, &mut mods, make_world).unwrap();
        assert_eq!(player.stash.count(rock), 4);
    }

    #[test]
    fn unknown_edit_specs_become_air() {
        let mut doc = bare_doc();
        doc.specs = vec!["natural:Stone".into()];
        doc.edits = vec![super::format::Edit { x: 1, y: 40, z: 1, spec: 0 }];
        doc.meta.edit_count = 1;
        let mut mods = Mods::with_defaults();
        let (world, _, _) = from_doc(doc, &mut mods, make_world).unwrap();
        assert_eq!(world.block_at(1, 40, 1), AIR);
    }

    #[test]
    fn law_mismatch_is_refused() {
        let mut doc = bare_doc();
        doc.law_stamp[0] ^= 0xff;
        let mut mods = Mods::with_defaults();
        assert!(matches!(
            from_doc(doc, &mut mods, make_world),
            Err(SaveError::LawMismatch)
        ));
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
            law_stamp: material::Law::v0().stamp(),
            meta: meta("mods"),
            player: blank_player.clone(),
            specs: vec![],
            edits: vec![],
            mods: vec![
                ("Crafting".into(), "*natural:Stone=1".into()),
                ("no-such-mod".into(), "ignored".into()),
                ("Crafting".into(), "air=2".into()),
            ],
        };
        let mut mods = Mods::with_defaults();
        let (world, _, _) = super::bridge::from_doc(doc, &mut mods, make_world).unwrap();
        let states = mods.save_states(&world);
        let craft = states.iter().find(|(n, _)| n == "crafting").map(|(_, d)| d.as_str());
        assert_eq!(craft, Some("v1;air=2"), "duplicate mod lines: last wins");
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
            law_stamp: world.registry().law().stamp(),
            player: PlayerState {
                pos: [player.position.x, player.position.y, player.position.z],
                yaw: player.orientation.yaw,
                pitch: player.orientation.pitch,
                flying: player.flying(),
                noclip: player.noclip(),
                stash: Some(player.stash.to_portable(|id| world.registry().spec(id))),
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
        let soil = world.registry().id_by_label("soil").unwrap();
        world.set_block(bx, by + 1, bz, soil);

        let mut player = Player::new(DVec3::new(1.0, 2.0, 3.0));
        player.orientation.yaw = 0.5;
        player.orientation.pitch = -0.25;
        player.set_flying(true);
        let rock = world.registry().id_by_label("rock").unwrap();
        player.stash.add(rock, 2);
        player.stash.add(soil, 1);
        let mut mods = Mods::with_defaults();
        let rock_spec = world.registry().spec(rock);
        mods.load_state("Crafting", &format!("*{rock_spec}=1"), &mut world);

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
            version: 1,
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
