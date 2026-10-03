//! Test stand-ins for the first-party mods. The real ones are packages in the PWC package manager
//! repository (they depend on this crate, so its own tests cannot link them); these mirror the
//! behaviour the core's tests exercise — ids, names, groups, a menu theme, persisted state, the
//! visual groups and the worldgen knobs — so host, menu and save tests keep their meaning.

use super::{Knob, Mod, Mods, ESSENTIALS};
use crate::menu::theme::{DefaultTheme, MenuTheme};
use crate::render_config::VisualGroup;
use crate::world::generation::WorldgenKind;
use crate::world::terrain::TerrainCfg;
use crate::world::World;

/// A stand-in with a fixed identity and optional behaviours.
pub(crate) struct Stand {
    id: &'static str,
    name: &'static str,
    visual: Option<VisualGroup>,
    theme: Option<DefaultTheme>,
    /// Persisted per-world state (echoed back verbatim), like the hotbar's slots.
    state: Option<String>,
    persists: bool,
    /// A fixed `mods.cfg` payload, like the texture and naming knobs.
    choice: Option<&'static str>,
}

impl Stand {
    pub(crate) fn new(id: &'static str, name: &'static str) -> Self {
        Self { id, name, visual: None, theme: None, state: None, persists: false, choice: None }
    }
}

impl Mod for Stand {
    fn name(&self) -> &str {
        self.name
    }
    fn id(&self) -> &'static str {
        self.id
    }
    fn group(&self) -> &'static str {
        ESSENTIALS
    }
    fn visual_group(&self) -> Option<VisualGroup> {
        self.visual
    }
    fn menu_theme(&self) -> Option<&dyn MenuTheme> {
        self.theme.as_ref().map(|t| t as &dyn MenuTheme)
    }
    fn reset(&mut self) {
        if self.persists {
            self.state = None;
        }
    }
    fn save_state(&self, _world: &World) -> Option<(u16, String)> {
        self.state.clone().map(|s| (1, s))
    }
    fn load_state(&mut self, _version: u16, data: &str, _world: &mut World) -> u32 {
        if self.persists {
            self.state = Some(data.to_string());
        }
        0
    }
    fn save_choice_state(&self) -> Option<String> {
        self.choice.map(str::to_string)
    }
}

/// The InfiniteDiffusion stand-in: the worldgen kind and its eight knobs, exactly like the package.
pub(crate) struct Worldgen {
    cfg: TerrainCfg,
}

impl Mod for Worldgen {
    fn name(&self) -> &str {
        "InfiniteDiffusion"
    }
    fn id(&self) -> &'static str {
        WorldgenKind::Diffusion.id()
    }
    fn group(&self) -> &'static str {
        ESSENTIALS
    }
    fn worldgen(&self) -> Option<WorldgenKind> {
        Some(WorldgenKind::Diffusion)
    }
    fn worldgen_config(&self) -> Option<String> {
        Some(self.cfg.to_text())
    }
    fn knobs(&self) -> Vec<Knob> {
        ["Relief", "Caves", "Mines", "Space", "Variety", "Features", "Structures", "Deep"]
            .into_iter()
            .zip(self.cfg.to_wire())
            .map(|(label, v)| Knob { label, value: format!("{v}%"), hint: String::new() })
            .collect()
    }
    fn step_knob(&mut self, index: usize, delta: i32) {
        let mut knobs = self.cfg.to_wire();
        if let Some(slot) = knobs.get_mut(index) {
            *slot = (*slot as i32 + delta * TerrainCfg::STEP as i32).max(0) as u16;
        }
        self.cfg = TerrainCfg::from_wire(knobs);
    }
    fn save_choice_state(&self) -> Option<String> {
        Some(self.cfg.to_text())
    }
    fn load_choice_state(&mut self, data: &str) {
        self.cfg = self.cfg.overlay(data);
    }
}

/// The ids of [`standard`], in install order (the old built-in order).
pub(crate) const STANDARD_IDS: [&str; 10] = [
    "menus",
    "start",
    "inventory",
    "hotbar",
    "atmosphere",
    "post",
    "lighting",
    "neural_textures",
    "material_names",
    "diffusion",
];

/// Stand-ins for the essentials, all enabled, in the old built-in order.
pub(crate) fn standard() -> Mods {
    let mut mods = Mods::empty();
    let mut menus = Stand::new("menus", "Menus");
    menus.theme = Some(DefaultTheme);
    mods.install(Box::new(menus), true);
    mods.install(Box::new(Stand::new("start", "Start")), true);
    mods.install(Box::new(Stand::new("inventory", "Inventory")), true);
    let mut hotbar = Stand::new("hotbar", "Hotbar");
    hotbar.persists = true;
    mods.install(Box::new(hotbar), true);
    for (id, name, group) in [
        ("atmosphere", "Atmosphere", VisualGroup::Atmosphere),
        ("post", "Post", VisualGroup::Post),
        ("lighting", "Lighting", VisualGroup::Lighting),
    ] {
        let mut m = Stand::new(id, name);
        m.visual = Some(group);
        mods.install(Box::new(m), true);
    }
    let mut textures = Stand::new("neural_textures", "Neural textures");
    textures.choice = Some("detail=1.0,contrast=1.0");
    mods.install(Box::new(textures), true);
    let mut names = Stand::new("material_names", "Material names");
    names.choice = Some("style=mineral");
    mods.install(Box::new(names), true);
    mods.install(Box::new(Worldgen { cfg: TerrainCfg::default() }), true);
    mods
}
