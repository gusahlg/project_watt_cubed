//! Hello HUD: the smallest useful PWC mod. One HUD line, one counter, one event hook.

use pwc_mod_api::prelude::*;

/// The package entry point: the PWC builder calls it once at startup.
pub fn register(registrar: &mut ModRegistrar) {
    registrar.add(HelloHud::default());
}

/// Counts broken blocks and shows the count.
#[derive(Default)]
pub struct HelloHud {
    broken: u64,
}

impl Mod for HelloHud {
    fn name(&self) -> &str {
        "Hello HUD"
    }

    fn id(&self) -> &'static str {
        "hello_hud"
    }

    fn description(&self) -> &str {
        "Counts the blocks you broke (an example mod)."
    }

    fn reset(&mut self) {
        self.broken = 0;
    }

    fn on_block_break(&mut self, _id: BlockId, _world: &World, _overflow: bool) {
        self.broken += 1;
    }

    fn hud(&self, _world: &World, _player: &Player, _screen: (i32, i32), out: &mut Vec<HudElement>) {
        out.push(HudElement::Label {
            at: Anchor::TopRight,
            off: (-12, 96),
            base_fs: 18,
            role: Role::Muted,
            text: format!("Hello from a mod - {} blocks broken", self.broken).into(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pwc_mod_api::{GameBuild, ModDescriptor};

    #[test]
    fn registers_one_enabled_mod() {
        let build = GameBuild::new().with_mod(ModDescriptor {
            id: "example.hello-hud",
            name: "Hello HUD",
            version: "1.0.0",
            register,
        });
        let mods = build.mods();
        assert_eq!(mods.len(), 1);
        assert_eq!(mods.id(0), "hello_hud");
        assert_eq!(mods.package(0), Some("example.hello-hud"));
        assert!(mods.is_enabled(0));
    }
}
