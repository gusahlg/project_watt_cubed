//! A test harness for mod packages: the packages of a build registered the way the game registers
//! them, with the options they declare and the hooks to drive them. The host type behind it is not
//! part of the API, so a mod can neither name nor reach the other mods of a build.
//!
//! ```
//! use pwc_mod_api::testing::Harness;
//! use pwc_mod_api::{GameBuild, ModDescriptor, ModRegistrar, Mod};
//!
//! struct Hello;
//! impl Mod for Hello {
//!     fn id(&self) -> &'static str {
//!         "hello"
//!     }
//! }
//! fn register(registrar: &mut ModRegistrar) {
//!     registrar.add(Hello);
//! }
//! let harness = Harness::new(GameBuild::new().with_mod(ModDescriptor {
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
use project_watt_cubed::modding::{
    BuildInfo, FrameContext, GameBuild, HudFacts, Message, ModContext, Mods, ToolUse, VisualMask,
};
use project_watt_cubed::player::Player;
use project_watt_cubed::render_config::{RenderConfig, VisualGroup};
use project_watt_cubed::screen::{Screen, ScreenEntry, ScreenFacts};
use project_watt_cubed::settings::{OptionValue, Options, Settings};
use project_watt_cubed::ui::HudElement;
use project_watt_cubed::world::generation::WorldgenKind;
use project_watt_cubed::world::World;

/// The mods a build installs, the options they declare, and the hooks to drive them, for tests.
pub struct Harness {
    build: GameBuild,
    mods: Mods,
    options: Options,
}

impl Harness {
    /// Register every package of `build` in order, as the game does; every mod hears the default
    /// options.
    pub fn new(build: GameBuild) -> Self {
        let mut options = Options::new();
        let mods = Mods::from_build(&build, &mut options);
        Self { build, mods, options }
    }

    /// The build's package list, as [`ModRegistrar::build`](crate::ModRegistrar::build) gave it.
    pub fn build(&self) -> &BuildInfo {
        self.build.info()
    }

    // The installed mods, by install index.

    /// How many mods the packages installed.
    pub fn len(&self) -> usize {
        self.mods.len()
    }

    /// Whether the packages installed no mod.
    pub fn is_empty(&self) -> bool {
        self.mods.is_empty()
    }

    /// The id of mod `i`.
    pub fn id(&self, i: usize) -> &str {
        self.mods.id(i)
    }

    /// The name of mod `i` (its id unless it says otherwise).
    pub fn name(&self, i: usize) -> &str {
        self.mods.name(i)
    }

    /// The package that installed mod `i`.
    pub fn package(&self, i: usize) -> Option<&'static str> {
        self.mods.package(i)
    }

    /// Whether mod `i` runs (false while its package is suspended).
    pub fn is_active(&self, i: usize) -> bool {
        self.mods.is_active(i)
    }

    /// The visual group mod `i` provides, if any.
    pub fn visual_group(&self, i: usize) -> Option<VisualGroup> {
        self.mods.visual_group(i)
    }

    /// Whether mod `i` replaces worldgen.
    pub fn is_worldgen(&self, i: usize) -> bool {
        self.mods.is_worldgen(i)
    }

    // Options.

    /// The options the packages declared, with their values.
    pub fn options(&self) -> &Options {
        &self.options
    }

    /// The options, to change. Call [`options_changed`](Self::options_changed) to tell the mods.
    pub fn options_mut(&mut self) -> &mut Options {
        &mut self.options
    }

    /// Tell every mod the options changed, as the core does after a change.
    pub fn options_changed(&mut self) {
        self.mods.options_changed(&self.options);
    }

    /// Set the option `<package>.<key>` and tell the mods. False when no package declared it.
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
        self.mods.suspend_packages(&ids);
    }

    /// End every suspension.
    pub fn resume(&mut self) {
        self.mods.resume_packages();
    }

    // Hooks, as the core arbitrates them.

    /// Every active mod's `update`.
    pub fn update(&mut self, ctx: &mut ModContext) {
        self.mods.update(ctx);
    }

    /// Every active mod's `on_frame`, with the keyboard capture as the core keeps it.
    pub fn frame(&mut self, ctx: &mut FrameContext) {
        self.mods.on_frame(ctx);
    }

    /// Whether a mod holds the keyboard.
    pub fn text_captured(&self) -> bool {
        self.mods.text_captured()
    }

    /// Hand `msg` to every active mod; true when one showed it.
    pub fn message(&mut self, msg: &Message) -> bool {
        self.mods.on_message(msg)
    }

    /// Every active mod's HUD, in install order.
    pub fn hud(&self, facts: &HudFacts, world: &World, player: &Player, out: &mut Vec<HudElement>) {
        self.mods.hud(facts, world, player, out);
    }

    /// Escape: the first open overlay closes. True when one did.
    pub fn close_overlay(&mut self) -> bool {
        self.mods.close_overlay()
    }

    /// A block broke into this configuration.
    pub fn on_block_break(&mut self, id: BlockId, world: &World, overflow: bool) {
        self.mods.on_block_break(id, world, overflow);
    }

    /// A tool reaction changed a held unit.
    pub fn on_tool_changed(&mut self, old: BlockId, new: BlockId) {
        self.mods.on_tool_changed(old, new);
    }

    /// A primary action finished.
    pub fn on_tool_used(&mut self, outcome: ToolUse) {
        self.mods.on_tool_used(outcome);
    }

    /// The tool a primary action applies: the first active mod that answers.
    pub fn tool(&self, player: &Player) -> Option<BlockId> {
        self.mods.tool(player)
    }

    /// The first active appearance, or the core's flat one.
    pub fn appearance(&self) -> &dyn BlockAppearance {
        self.mods.appearance()
    }

    /// The first active namer, if any.
    pub fn namer(&self) -> Option<&dyn MaterialNamer> {
        self.mods.namer()
    }

    /// The generator new worlds use.
    pub fn worldgen_kind(&self) -> WorldgenKind {
        self.mods.worldgen_kind()
    }

    /// The winning worldgen mod's payload.
    pub fn worldgen_config(&self) -> Option<String> {
        self.mods.worldgen_config()
    }

    /// The visual groups the active mods provide.
    pub fn visual_mask(&self) -> VisualMask {
        self.mods.visual_mask()
    }

    /// The packages whose mods provide a visual group.
    pub fn visual_packages(&self) -> Vec<String> {
        self.mods.visual_packages()
    }

    /// The settings' render lanes with the unprovided groups stripped.
    pub fn effective_render(&self, settings: &Settings) -> RenderConfig {
        self.mods.effective_render(settings)
    }

    /// Every mod's per-world save line, as `(id, data)`.
    pub fn save_states(&self, world: &World) -> Vec<(String, String)> {
        self.mods.save_states(world)
    }

    /// Restore one save line; how many holdings the mod dropped.
    pub fn load_state(&mut self, name: &str, data: &str, world: &mut World) -> u32 {
        self.mods.load_state(name, data, world)
    }

    /// Entering a world: every mod resets its per-world state.
    pub fn reset(&mut self) {
        self.mods.reset_state();
    }

    /// The screen entries of the active packages, by order.
    pub fn screen_entries(&self) -> &[ScreenEntry] {
        self.mods.screen_entries()
    }

    /// The first active mod's root screen.
    pub fn root_screen(&self, facts: &ScreenFacts) -> Option<Box<dyn Screen>> {
        self.mods.root_screen(facts)
    }

    /// The first active mod's pause screen.
    pub fn pause_screen(&self, facts: &ScreenFacts) -> Option<Box<dyn Screen>> {
        self.mods.pause_screen(facts)
    }
}
