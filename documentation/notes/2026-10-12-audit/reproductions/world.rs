// SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;
use crate::coord::BlockCoord;
use crate::sim::reactions::CellStore;

#[test]
fn audit_reaction_store_does_not_glue_chart_neighbours() {
    let mut world = World::with_kind_cfg(42, crate::render_config::RenderConfig::default(), WorldgenKind::Diffusion, Default::default(), false);
    let atlas = world.generator.atlases()[0].clone();
    let patch = crate::space::atlas::Patch::Shell { band: 0, face: Face::PosY };
    let (o, size) = atlas.storage_box(patch);
    let x = (o[0] + size[0]) as i32;
    let z = (o[2] + size[2] / 2) as i32;
    let y = world.generator.height(x - 1, z) - 1;
    let ghost = BlockCoord::new(x, y, z);
    let real = world.seams.glue_cell(ghost).expect("edge has a neighbour");
    let cfg = material::Configuration::single(material::Element::new([1, 2, 3, 4]));
    let id = world.registry_mut().intern(&cfg).unwrap();
    world.set_block(real.x, real.y, real.z, id);
    assert_eq!(CellStore::block_at(&world, real.to_tuple()), Some(id));
    assert_ne!(CellStore::block_at(&world, ghost.to_tuple()), Some(id));
    eprintln!("BUG: reaction neighbour {ghost:?} reads a different cell from its glued neighbour {real:?}");
}

#[test]
fn audit_restored_natural_edits_leave_column_indexes_behind() {
    let mut world = World::with_kind_cfg(1, crate::render_config::RenderConfig::default(), WorldgenKind::Flat, Default::default(), false);
    let id = world.registry_mut().intern(&material::Configuration::single(material::Element::new([1, 2, 3, 4]))).unwrap();
    for k in 0..100 {
        let p = (k * 16, 80, 0);
        world.set_block(p.0, p.1, p.2, id);
        world.set_block(p.0, p.1, p.2, crate::block::AIR);
    }
    assert_eq!(world.edits().count(), 0);
    assert_eq!(world.edit_columns.len(), 100);
    assert_eq!(world.edit_column_chunks.len(), 100);
    eprintln!("BUG: zero remaining edits retain 100 column entries in each index");
}
