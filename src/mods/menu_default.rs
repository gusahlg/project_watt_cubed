//! The default menu mod: the standard out-of-game look.
//!
//! Mods supply themes but never handle menu input/state — that lives in the
//! menu module. This mod provides the built-in theme. Disable it and the App
//! falls back to the same one, so navigation can never brick. Install a
//! replacement mod for a whole new look.
use crate::menu::theme::{DefaultTheme, MenuTheme};
use crate::mods::Mod;

/// The standard menu look, as a disableable, replaceable mod.
pub struct MenuDefaultMod {
    theme: DefaultTheme,
}

impl MenuDefaultMod {
    pub fn new() -> Self {
        Self { theme: DefaultTheme }
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

    fn id(&self) -> &'static str {
        "menus"
    }

    fn description(&self) -> &str {
        "The standard menu look (title/panel screens, bars and toggles)."
    }

    fn group(&self) -> &'static str {
        crate::mods::ESSENTIALS
    }

    fn menu_theme(&self) -> Option<&dyn MenuTheme> {
        Some(&self.theme)
    }
}
