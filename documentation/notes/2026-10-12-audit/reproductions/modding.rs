// SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;

#[test]
fn audit_duplicate_mod_ids_share_a_persistence_key() {
    struct Saved(&'static str);
    impl Mod for Saved {
        fn id(&self) -> &'static str { "same" }
        fn save_state(&self, _: &World) -> Option<(u16, String)> { Some((1, self.0.into())) }
        fn load_state(&mut self, _: u16, data: &str, _: &mut World) -> u32 {
            self.0 = if data == "first" { "first" } else { "second" };
            0
        }
    }
    let mut world = World::with_kind_cfg(1, crate::render_config::RenderConfig::default(), crate::world::generation::WorldgenKind::Flat, Default::default(), false);
    let mut mods = Mods::empty();
    mods.install_from(Some("a.package"), Box::new(Saved("first")));
    mods.install_from(Some("b.package"), Box::new(Saved("second")));
    let saved = mods.save_states(&world);
    assert_eq!(saved[0].0, saved[1].0);
    for (key, data) in saved { mods.load_state(&key, &data, &mut world); }
    let after = mods.save_states(&world);
    assert_eq!(after[0].1, "v1;second");
    assert_eq!(after[1].1, "v1;second");
    eprintln!("BUG: two different packages' mod ids collide and first state is overwritten");
}

#[test]
fn audit_missing_mod_state_is_not_retained() {
    let mut world = World::with_kind_cfg(1, crate::render_config::RenderConfig::default(), crate::world::generation::WorldgenKind::Flat, Default::default(), false);
    let mut mods = Mods::empty();
    assert_eq!(mods.load_state("absent.machine", "v1;machine=42", &mut world), 0);
    assert!(mods.save_states(&world).is_empty());
    eprintln!("BUG: absent package state disappears from the next snapshot");
}
