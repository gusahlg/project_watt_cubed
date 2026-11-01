// SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;

#[test]
fn audit_provider_switch_with_equal_revision_keeps_old_names() {
    struct Namer(&'static str);
    impl MaterialNamer for Namer {
        fn revision(&self) -> u32 { 0 }
        fn names(&self, _: &NamingSource) -> MaterialNames { MaterialNames { block: self.0.into(), tool: self.0.into() } }
    }
    let mut registry = BlockRegistry::with_builtins();
    let id = registry.intern(&Configuration::single(Element::new([1, 2, 3, 4]))).unwrap();
    registry.refresh_names(Some(&Namer("first")));
    registry.refresh_names(Some(&Namer("second")));
    assert_eq!(registry.display_name(id), "first");
    eprintln!("BUG: switching naming providers with revision 0 leaves first provider's name");
}

#[test]
fn audit_lower_descriptor_cap_does_not_reassign_existing_layers() {
    let mut registry = BlockRegistry::with_builtins();
    for i in 0..5 {
        registry.intern(&Configuration::single(Element::new([i, 2, 3, 4]))).unwrap();
    }
    let id = BlockId(5);
    registry.set_descriptor_cap(2);
    assert_eq!(registry.render_layer(id), 5);
    let mut hot = registry.hot_tables();
    hot.layer_cap = 2;
    let result = std::panic::catch_unwind(|| hot.render_layer(id));
    if cfg!(debug_assertions) { assert!(result.is_err()); }
    else { assert_eq!(result.unwrap(), 1); }
    eprintln!("BUG: pre-existing layer 5 survives cap 2; debug panics, release aliases to last layer");
}
