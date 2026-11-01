// SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;

#[test]
fn audit_mod_extended_inventory_capacity_does_not_survive_save() {
    let make = |seed, kind, cfg| World::with_kind_cfg(seed, crate::render_config::RenderConfig::default(), kind, cfg, false);
    let mut world = make(1, WorldgenKind::Flat, TerrainCfg::default());
    let id = world.registry_mut().intern(&material::Configuration::single(material::Element::new([1, 2, 3, 4]))).unwrap();
    let mut player = Player::new(DVec3::new(0.5, 80.5, 0.5));
    player.inventory = crate::inventory::Inventory::new(250);
    assert!(player.inventory.add(id, 200));
    let meta = SaveMeta { name: "audit".into(), seed: 1, created: 0, last_played: 0, playtime_secs: 0, edit_count: 0 };
    let doc = to_doc(&world, &player, &Mods::empty(), meta).unwrap();
    let restored = rebuild(doc, make).unwrap();
    assert_eq!(restored.player.inventory.capacity(), 100);
    assert_eq!(restored.player.inventory.total(), 100);
    eprintln!("BUG: capacity 250 with 200 units restores as capacity 100 with 100 units");
}
