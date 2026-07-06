//! The default menu mod: the game's menus made visible.
//!
//! The core menus are, by design, just [`MenuModel`]s — lists of alternatives
//! that lead to something, with no look and no keybindings of their own. This
//! mod *is* the standard look and interaction: arrows or j/k to move, Enter or
//! l to pick, Esc (or h, where it isn't a typed character) to go back, plus
//! the host/join text-field editing. Switch it off in the mod menu and the
//! core's plain built-in fallback takes over — navigation can never brick —
//! or install a replacement mod with `handles_menus()` for a whole new skin.
//!
//! Both the driving and the drawing are thin wrappers over the free functions
//! in [`crate::menu`] ([`menu::drive`], [`menu::draw_model`]); the fallback
//! uses the same ones, so there is exactly one implementation of the standard
//! behavior.
use voxel_engine::{Engine, Frame};

use crate::menu::{self, MenuEvent, MenuKeys, MenuModel};
use crate::mods::Mod;

/// The standard menu look and keys, as a disableable, replaceable mod.
pub struct MenuDefaultMod;

impl MenuDefaultMod {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MenuDefaultMod {
    fn default() -> Self {
        Self::new()
    }
}

impl Mod for MenuDefaultMod {
    fn name(&self) -> &str {
        "Menus"
    }

    fn description(&self) -> &str {
        "The standard menu look and keys (arrows/hjkl, Enter, Esc)."
    }

    fn handles_menus(&self) -> bool {
        true
    }

    fn drive_menu(&mut self, eng: &Engine, menu: &mut MenuModel) -> Option<MenuEvent> {
        menu::drive(&MenuKeys::capture(eng), menu)
    }

    fn draw_menu(&mut self, f: &mut Frame, menu: &MenuModel, screen_w: i32, screen_h: i32) {
        menu::draw_model(f, menu, screen_w, screen_h);
    }
}
