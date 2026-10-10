//! Test stand-ins for the first-party mods. The real ones are packages in the PWC package manager
//! repository (they depend on this crate, so its own tests cannot link them); these mirror the
//! behaviour the core's tests exercise — ids, persisted state, the visual groups and the worldgen
//! payload — so host, menu and save tests keep their meaning. Each is installed under the id of
//! the package that ships the real one, so tests can suspend it the way a server does.

use super::{Action, Mod, Mods};
use crate::input::intent::Chord;
use crate::render_config::VisualGroup;
use crate::world::generation::WorldgenKind;
use crate::world::terrain::TerrainCfg;
use crate::world::World;

/// A test mod: an id and whichever hooks a test gives it.
#[derive(Clone, Copy)]
pub(crate) struct Stub {
    id: &'static str,
    actions: &'static [Action],
}

impl Stub {
    pub(crate) const fn new(id: &'static str) -> Self {
        Self { id, actions: &[] }
    }

    pub(crate) const fn actions(self, actions: &'static [Action]) -> Self {
        Self { actions, ..self }
    }
}

impl Mod for Stub {
    fn id(&self) -> &'static str {
        self.id
    }
    fn actions(&self) -> &[Action] {
        self.actions
    }
}

/// An action on `default` that neither repeats, holds nor runs immediately, labelled with its id.
pub(crate) const fn action(id: &'static str, default: &'static [Chord]) -> Action {
    Action { id, label: id, default, repeat: false, held: false, immediate: false }
}

/// A stand-in with a fixed identity and optional behaviours.
pub(crate) struct Stand {
    id: &'static str,
    name: &'static str,
    visual: Option<VisualGroup>,
    /// Persisted per-world state (echoed back verbatim), like a mod's own save line.
    state: Option<String>,
    persists: bool,
}

impl Stand {
    pub(crate) fn new(id: &'static str, name: &'static str) -> Self {
        Self { id, name, visual: None, state: None, persists: false }
    }
}

impl Mod for Stand {
    fn name(&self) -> &str {
        self.name
    }
    fn id(&self) -> &'static str {
        self.id
    }
    fn visual_group(&self) -> Option<VisualGroup> {
        self.visual
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
}

/// The InfiniteDiffusion stand-in: the worldgen kind and its payload, like the package.
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
    fn worldgen(&self) -> Option<WorldgenKind> {
        Some(WorldgenKind::Diffusion)
    }
    fn worldgen_config(&self) -> Option<String> {
        Some(self.cfg.to_text())
    }
}

/// Stand-ins for the essentials, in the old built-in order, each under its package id.
pub(crate) fn standard() -> Mods {
    let mut mods = Mods::empty();
    mods.install_from(Some("pwc.start-screen"), Box::new(Stand::new("start", "Start")));
    mods.install_from(Some("pwc.inventory"), Box::new(Stand::new("inventory", "Inventory")));
    let mut hotbar = Stand::new("hotbar", "Hotbar");
    hotbar.persists = true;
    mods.install_from(Some("pwc.hotbar"), Box::new(hotbar));
    for (id, name, group) in [
        ("atmosphere", "Atmosphere", VisualGroup::Atmosphere),
        ("post", "Post", VisualGroup::Post),
        ("lighting", "Lighting", VisualGroup::Lighting),
    ] {
        let mut m = Stand::new(id, name);
        m.visual = Some(group);
        mods.install_from(Some("pwc.visuals"), Box::new(m));
    }
    mods.install_from(Some("pwc.neural-textures"), Box::new(Stand::new("neural_textures", "Neural textures")));
    mods.install_from(Some("pwc.material-names"), Box::new(Stand::new("material_names", "Material names")));
    mods.install_from(Some("pwc.infinite-diffusion"), Box::new(Worldgen { cfg: TerrainCfg::default() }));
    mods
}
