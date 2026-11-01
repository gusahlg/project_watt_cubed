//! Hello HUD: the smallest useful PWC module. One HUD line, one counter, one event hook.

use pwc_module_api::prelude::*;

/// The package entry point: the PWC builder calls it once at startup.
pub fn register(registrar: &mut ModuleRegistrar) {
    registrar.add(HelloHud::default());
}

/// Counts broken blocks and shows the count.
#[derive(Default)]
pub struct HelloHud {
    broken: u64,
}

impl Module for HelloHud {
    fn name(&self) -> &str {
        "Hello HUD"
    }

    fn id(&self) -> &'static str {
        "hello_hud"
    }

    fn reset(&mut self) {
        self.broken = 0;
    }

    fn on_block_break(&mut self, _id: BlockId, _world: &World, _overflow: bool) {
        self.broken += 1;
    }

    fn hud(&self, facts: &HudFacts, _world: &World, _player: &Player, out: &mut Vec<HudElement>) {
        // The core asks in every HUD mode; this line shows in Full and Minimal.
        if !facts.hud_mode.shows_module_hud() {
            return;
        }
        out.push(HudElement::Label {
            at: Anchor::TopRight,
            off: (-12, 96),
            base_fs: 18,
            role: Role::Muted,
            text: format!("Hello from a module - {} blocks broken", self.broken).into(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pwc_module_api::testing::Harness;
    use pwc_module_api::{GameBuild, ModuleDescriptor};

    #[test]
    fn registers_one_module() {
        let harness = Harness::new(GameBuild::new().with_module(ModuleDescriptor {
            id: "example.hello-hud",
            name: "Hello HUD",
            version: "1.1.1",
            register,
        }));
        assert_eq!(harness.len(), 1);
        assert_eq!(harness.id(0), "hello_hud");
        assert_eq!(harness.package(0), Some("example.hello-hud"));
        assert!(harness.is_active(0));
    }
}
