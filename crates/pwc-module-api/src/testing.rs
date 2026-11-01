//! A test harness for module packages: the packages of a build registered the way the game registers
//! them, with the options they declare and the hooks to drive them. The host type behind it is not
//! part of the API, so a module can neither name nor reach the other modules of a build.
//!
//! ```
//! use pwc_module_api::testing::Harness;
//! use pwc_module_api::{GameBuild, ModuleDescriptor, ModuleRegistrar, Module};
//!
//! struct Hello;
//! impl Module for Hello {
//!     fn id(&self) -> &'static str {
//!         "hello"
//!     }
//! }
//! fn register(registrar: &mut ModuleRegistrar) {
//!     registrar.add(Hello);
//! }
//! let harness = Harness::new(GameBuild::new().with_module(ModuleDescriptor {
//!     id: "example.hello",
//!     name: "Hello",
//!     version: "1.0.0",
//!     register,
//! }));
//! assert_eq!((harness.len(), harness.id(0), harness.package(0)), (1, "hello", Some("example.hello")));
//! ```

use project_watt_cubed::block::appearance::BlockAppearance;
use project_watt_cubed::block::naming::MaterialNamer;
use project_watt_cubed::block::BlockId;
use project_watt_cubed::modules::{
    BuildInfo, FrameContext, GameBuild, HudFacts, Message, ModuleContext, Modules, ToolUse, VisualMask,
};
use project_watt_cubed::player::Player;
use project_watt_cubed::render_config::{RenderConfig, VisualGroup};
use project_watt_cubed::screen::{Screen, ScreenEntry, ScreenFacts};
use project_watt_cubed::settings::{OptionValue, Options, Settings};
use project_watt_cubed::ui::HudElement;
use project_watt_cubed::world::generation::WorldgenKind;
use project_watt_cubed::world::World;

/// The modules a build installs, the options they declare, and the hooks to drive them, for tests.
pub struct Harness {
    build: GameBuild,
    modules: Modules,
    options: Options,
}

impl Harness {
    /// Native authoritative law chosen by this build's currently active providers.
    pub fn reaction_model(&self) -> Option<project_watt_cubed::sim::model::ModelSpec> {
        self.modules.reaction_model()
    }
    /// Register every package of `build` in order, as the game does; every module hears the default
    /// options.
    pub fn new(build: GameBuild) -> Self {
        let mut options = Options::new();
        let modules = Modules::from_build(&build, &mut options);
        Self { build, modules, options }
    }

    /// The build's package list, as [`ModuleRegistrar::build`](crate::ModuleRegistrar::build) gave it.
    pub fn build(&self) -> &BuildInfo {
        self.build.info()
    }

    // The installed modules, by install index.

    /// How many modules the packages installed.
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Whether the packages installed no module.
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// The id of module `i`.
    pub fn id(&self, i: usize) -> &str {
        self.modules.id(i)
    }

    /// The name of module `i` (its id unless it says otherwise).
    pub fn name(&self, i: usize) -> &str {
        self.modules.name(i)
    }

    /// The package that installed module `i`.
    pub fn package(&self, i: usize) -> Option<&'static str> {
        self.modules.package(i)
    }

    /// Whether module `i` runs (false while its package is suspended).
    pub fn is_active(&self, i: usize) -> bool {
        self.modules.is_active(i)
    }

    /// The visual group module `i` provides, if any.
    pub fn visual_group(&self, i: usize) -> Option<VisualGroup> {
        self.modules.visual_group(i)
    }

    /// Whether module `i` replaces worldgen.
    pub fn is_worldgen(&self, i: usize) -> bool {
        self.modules.is_worldgen(i)
    }

    // Options.

    /// The options the packages declared, with their values.
    pub fn options(&self) -> &Options {
        &self.options
    }

    /// The options, to change. Call [`options_changed`](Self::options_changed) to tell the modules.
    pub fn options_mut(&mut self) -> &mut Options {
        &mut self.options
    }

    /// Tell every module the options changed, as the core does after a change.
    pub fn options_changed(&mut self) {
        self.modules.options_changed(&self.options);
    }

    /// Set the option `<package>.<key>` and tell the modules. False when no package declared it.
    pub fn set_option(&mut self, full_key: &str, value: OptionValue) -> bool {
        let Some(id) = self.options.find(full_key) else { return false };
        self.options.set(id, value);
        self.options_changed();
        true
    }

    // Suspension.

    /// Suspend `packages` for the session, as a server that refuses them does.
    pub fn suspend(&mut self, packages: &[&str]) {
        let ids: Vec<String> = packages.iter().map(|id| id.to_string()).collect();
        self.modules.suspend_packages(&ids);
    }

    /// End every suspension.
    pub fn resume(&mut self) {
        self.modules.resume_packages();
    }

    // Hooks, as the core arbitrates them.

    /// Every active module's `update`.
    pub fn update(&mut self, ctx: &mut ModuleContext) {
        self.modules.update(ctx);
    }

    /// Every active module's `on_frame`, with the keyboard capture as the core keeps it.
    pub fn frame(&mut self, ctx: &mut FrameContext) {
        self.modules.on_frame(ctx);
    }

    /// Whether a module holds the keyboard.
    pub fn text_captured(&self) -> bool {
        self.modules.text_captured()
    }

    /// Hand `msg` to every active module; true when one showed it.
    pub fn message(&mut self, msg: &Message) -> bool {
        self.modules.on_message(msg)
    }

    /// Every active module's HUD, in install order.
    pub fn hud(&self, facts: &HudFacts, world: &World, player: &Player, out: &mut Vec<HudElement>) {
        self.modules.hud(facts, world, player, out);
    }

    /// Escape: the first open overlay closes. True when one did.
    pub fn close_overlay(&mut self) -> bool {
        self.modules.close_overlay()
    }

    /// A block broke into this configuration.
    pub fn on_block_break(&mut self, id: BlockId, world: &World, overflow: bool) {
        self.modules.on_block_break(id, world, overflow);
    }

    /// A tool reaction changed a held unit.
    pub fn on_tool_changed(&mut self, old: BlockId, new: BlockId) {
        self.modules.on_tool_changed(old, new);
    }

    /// A primary action finished.
    pub fn on_tool_used(&mut self, outcome: ToolUse) {
        self.modules.on_tool_used(outcome);
    }

    /// The tool a primary action applies: the first active module that answers.
    pub fn tool(&self, player: &Player) -> Option<BlockId> {
        self.modules.tool(player)
    }

    /// The first active appearance, or the core's flat one.
    pub fn appearance(&self) -> &dyn BlockAppearance {
        self.modules.appearance()
    }

    /// The first active namer, if any.
    pub fn namer(&self) -> Option<&dyn MaterialNamer> {
        self.modules.namer()
    }

    /// The generator new worlds use.
    pub fn worldgen_kind(&self) -> WorldgenKind {
        self.modules.worldgen_kind()
    }

    /// The winning worldgen module's payload.
    pub fn worldgen_config(&self) -> Option<String> {
        self.modules.worldgen_config()
    }

    /// The visual groups the active modules provide.
    pub fn visual_mask(&self) -> VisualMask {
        self.modules.visual_mask()
    }

    /// The packages whose modules provide a visual group.
    pub fn visual_packages(&self) -> Vec<String> {
        self.modules.visual_packages()
    }

    /// The settings' render lanes with the unprovided groups stripped.
    pub fn effective_render(&self, settings: &Settings) -> RenderConfig {
        self.modules.effective_render(settings)
    }

    /// Every module's per-world save line, as `(id, data)`.
    pub fn save_states(&self, world: &World) -> Vec<(String, String)> {
        self.modules.save_states(world)
    }

    /// Restore one save line; how many holdings the module dropped.
    pub fn load_state(&mut self, name: &str, data: &str, world: &mut World) -> u32 {
        self.modules.load_state(name, data, world)
    }

    /// Entering a world: every module resets its per-world state.
    pub fn reset(&mut self) {
        self.modules.reset_state();
    }

    /// The screen entries of the active packages, by order.
    pub fn screen_entries(&self) -> &[ScreenEntry] {
        self.modules.screen_entries()
    }

    /// The first active module's root screen.
    pub fn root_screen(&self, facts: &ScreenFacts) -> Option<Box<dyn Screen>> {
        self.modules.root_screen(facts)
    }

    /// The first active module's pause screen.
    pub fn pause_screen(&self, facts: &ScreenFacts) -> Option<Box<dyn Screen>> {
        self.modules.pause_screen(facts)
    }
}
