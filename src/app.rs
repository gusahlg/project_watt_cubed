//! app.rs owns the top-level state machine: the start menu, an in-world
//! [`Game`], the mod menu, the host/join forms, and the graphics settings
//! screen. It routes each engine frame to the active screen, creates and
//! loads worlds, and autosaves when leaving one.
//!
//! The window itself belongs to the engine: [`App::run`] hands a per-frame
//! closure to [`voxel_engine::run`], which is the moral equivalent of the old
//! raylib `while !window_should_close()` loop.
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use voxel_engine::{Color, DVec3, Engine};

use crate::audio::{AudioDirector, CuePalette, CueSymbols, OneShot, SoundConfig, SoundSystem};
use crate::benchmark::{Benchmark, Step as BenchmarkStep};
use crate::game::{Game, Signal};
use crate::input::router::{Context, Router, View};
use crate::menu::menus::MainMenu;
use crate::menu::theme::{DefaultTheme, MenuTheme};
use crate::menu::{AppEffect, Ctx, Framed, HostInfo, JoinInfo, MenuStack, ModRow};
use crate::mods::Mods;
use crate::net::client::Connection;
use crate::net::server::{self, Config, ServerHandle};
use crate::player::Player;
use crate::save::{self, Autosaver, SaveMeta, Slot, SlotId, Tick};
use crate::session::Session;
use crate::settings::Settings;
use crate::world::World;

const STARTING_WINDOW_WIDTH: u32 = 1280;
const STARTING_WINDOW_HEIGHT: u32 = 720;
/// Background for every non-world screen.
const MENU_CLEAR: Color = Color::new(18, 20, 28, 255);

enum Screen {
    Menus(MenuStack),
    Playing(Box<Game>),
}

/// The whole program: the installed mods (persist across worlds), the graphics
/// settings, and either the menu stack or the current world.
pub struct App {
    /// Available save slots, refreshed on menu return.
    saves: Vec<Slot>,
    /// The slot behind the open singleplayer world; `None` on menus and in
    /// multiplayer (a networked world is a server mirror, never saved locally).
    active: Option<ActiveSlot>,
    /// Shared router for menus and in-game input.
    router: Router,
    /// Installed mods and their on/off state; shared with the game while playing.
    mods: Mods,
    screen: Screen,
    /// The integrated server when hosting, kept alive for the session so friends can
    /// stay connected; stopping it frees the port for a later host.
    host: Option<ServerHandle>,
    /// Graphics settings, persisted in `saves/settings.cfg`.
    settings: Settings,
    /// Last-used connection details, persisted in `saves/session.cfg`.
    session: Session,
    /// Self-describing benchmark mode (`WATT_BENCH=<seconds>`).
    bench: Option<Benchmark>,
    /// Owns all playback continuation; enters/leaves world state as the screen changes.
    sound: SoundSystem,
    /// Cue name → id table resolved once at catalog load; used here to mint the
    /// menu-click UI cue (the director owns every in-world cue).
    cues: CueSymbols,
    /// Gameplay reports facts, this decides sounds. Owns the mic and all
    /// trace-derived state.
    audio: AudioDirector,
}

/// The save slot behind the open singleplayer world: identity, header
/// metadata carried across writes, accumulated playtime, and the autosaver.
struct ActiveSlot {
    id: SlotId,
    /// Persists name/created; seed/playtime/edits restamped on writes.
    meta: SaveMeta,
    /// Total seconds played, fractional to avoid per-frame truncation.
    playtime: f64,
    autosaver: Autosaver,
}

impl ActiveSlot {
    fn new(id: SlotId, meta: SaveMeta) -> Self {
        let playtime = meta.playtime_secs as f64;
        Self {
            id,
            meta,
            playtime,
            autosaver: Autosaver::new(),
        }
    }
}

impl App {
    pub fn new() -> Self {
        let mut mods = Mods::with_defaults();
        mods.apply_bench_env();
        let saves = save::list();
        let mut settings = Settings::load();
        let session = Session::load();
        let bench = Benchmark::from_env();
        // A reproducible benchmark can pin a performance profile without
        // mutating the saved configuration (the run never persists settings).
        if bench.is_some()
            && let Ok(preset) = std::env::var("WATT_BENCH_PRESET")
            && !settings.select_preset(preset.trim())
        {
            eprintln!("WATT_BENCH_PRESET={preset:?} not recognized; using saved settings");
        }
        // Instrumentation is opt-in (`WATT_BENCH_PROFILE=1`): headline
        // measurements stay uninstrumented, attribution runs are explicit and
        // reported separately. Safe here: `new()` runs on the main thread at
        // startup, before the renderer or any worker thread — the only reader
        // of this var — exists. Reads happen later.
        if bench.is_some()
            && matches!(std::env::var("WATT_BENCH_PROFILE").as_deref(), Ok("1"))
            && std::env::var_os("VOXEL_PROFILE").is_none()
        {
            unsafe { std::env::set_var("VOXEL_PROFILE", "1") };
        }
        // A missing device or a missing/corrupt catalog degrades to silence — the
        // client never panics on audio, it just runs muted with a startup warning.
        let (mut sound, cues) = SoundSystem::with_graceful_degradation(SoundConfig::default());
        sound.set_mix(settings.mix_change());
        // A missing or mode-mismatched cue role degrades that cue to silence with
        // a startup warning, rather than failing catalog load.
        let (palette, warnings) = CuePalette::build(&cues, sound.catalog());
        for w in warnings {
            eprintln!("{w}");
        }
        let audio = AudioDirector::new(palette);
        Self {
            saves,
            active: None,
            router: Router::new(),
            mods,
            screen: Screen::Menus(MenuStack::new(Framed::boxed(MainMenu::new()))),
            host: None,
            settings,
            session,
            bench,
            sound,
            cues,
            audio,
        }
    }

    /// Open the window and run until the player quits (menu or close button).
    pub fn run(self) {
        let mut app = self;
        let config = voxel_engine::Config {
            title: "Project Watt Cubed".into(),
            width: STARTING_WINDOW_WIDTH,
            height: STARTING_WINDOW_HEIGHT,
            target_fps: app.settings.max_fps,
            vsync: app.settings.vsync,
            msaa: app.settings.msaa,
            render_scale: app.settings.render_scale,
            resizable: true,
            fullscreen: app.settings.fullscreen,
            // Engine-side render lanes from the persisted settings (the single
            // source; the world's own occlusion/lod2 lanes come from the same
            // `Settings::render_config` at world entry).
            flags: app.settings.render_config().engine_flags(),
        };
        voxel_engine::run(config, move |eng| app.frame(eng));
    }

    /// One engine frame: update and draw the active screen.
    fn frame(&mut self, eng: &mut Engine) -> bool {
        // OS close button: save and go. Settings save too — the player may be
        // mid-edit on the Settings screen.
        if eng.should_close() {
            if self.bench.is_none() {
                self.settings.save();
            }
            self.flush_save();
            return false;
        }

        self.sound.service();

        if self.bench.is_some() && !self.bench_frame(eng) {
            return false;
        }

        let quit = match self.screen {
            Screen::Menus(_) => self.update_menus(eng),
            Screen::Playing(_) => {
                self.update_playing(eng);
                false
            }
        };
        if quit {
            if self.bench.is_none() {
                self.settings.save();
            }
            self.flush_save();
            return false;
        }
        // Force vsync on whenever we're not in a live world (menus, loading):
        // there's nothing to gain from tearing/uncapped frames on a static
        // screen, and it keeps the GPU quiet. In-world we honour the setting.
        // Only send the command on change: a SetVsync every menu frame was
        // waking the render thread even when the mode was already correct.
        let in_world = matches!(self.screen, Screen::Playing(_));
        let want_vsync = !in_world || self.settings.vsync;
        if eng.vsync() != want_vsync {
            eng.set_vsync(want_vsync);
        }
        self.draw(eng);
        true
    }

    /// Drive one benchmark frame: enter a reproducible world, wait for both the
    /// warmup floor and streaming readiness, rotate the camera, and hand every
    /// measured frame to the self-describing recorder.
    fn bench_frame(&mut self, eng: &mut Engine) -> bool {
        let dt = eng.frame_time();

        if !self
            .bench
            .as_ref()
            .expect("bench_frame without bench")
            .has_started()
        {
            let pos = {
                let bench = self.bench.as_mut().expect("bench exists");
                bench.begin();
                bench.position()
            };
            // Uncapped and unsynced, or the bench measures the throttle.
            self.settings.vsync = false;
            self.settings.max_fps = 0;
            self.settings.apply(eng);
            self.start_new_world(eng);
            if let Screen::Playing(game) = &mut self.screen {
                game.set_input_locked(true);
                // Far-coordinate bench: park the player at the requested position
                // with the ground under them made real, and give streaming a
                // little extra warmup to catch up before sampling starts.
                if let Some(pos) = pos {
                    game.player_mut().position = pos;
                    game.world_mut().prepare_around(pos);
                    if let Some(bench) = &mut self.bench {
                        bench.add_warmup(Duration::from_secs(2));
                    }
                }
            }
            return true;
        }
        let Screen::Playing(game) = &mut self.screen else {
            return true;
        };
        // A slow spin sweeps the frustum across the terrain like a player would;
        // an optional flight along +X (`WATT_BENCH_MOVE`) exercises the paths a
        // static camera never touches (shadow-cascade re-render, streaming).
        game.player_mut().orientation.yaw += 0.4 * dt;
        let move_mps = self.bench.as_ref().expect("bench exists").move_mps();
        if move_mps > 0.0 {
            game.player_mut().position.x += move_mps * dt as f64;
        }

        let bench = self.bench.as_mut().expect("bench exists");
        let step = bench.step(
            dt,
            game.world().entry_complete(),
            game.world().stream_gauges(),
        );
        match step {
            BenchmarkStep::ReadyTimeout => {
                eprintln!("{}", game.world().entry_debug());
                return true;
            }
            BenchmarkStep::Warming => {
                if !game.world().entry_complete() && bench.wait_log_due() {
                    eprintln!(
                        "benchmark: waiting for world ({})",
                        game.world().entry_debug()
                    );
                }
                return true;
            }
            BenchmarkStep::Measuring => return true,
            BenchmarkStep::Complete => {}
        }
        let report = bench.finish(&self.settings, eng, game.world(), game.player().position);
        report.emit();
        false
    }

    /// Update the menu stack and apply settings live each frame.
    fn update_menus(&mut self, eng: &mut Engine) -> bool {
        let dt = eng.frame_time();
        self.router.set_context(Context::Menu);
        let intents = match self.router.frame(eng, dt).view() {
            View::Menu(m) => crate::menu::gather(&m),
            _ => Vec::new(),
        };
        // Menus have no per-frame audio cadence, so this is the one cue emission
        // site outside the game.
        if intents.iter().any(|i| {
            matches!(
                i,
                crate::menu::Intent::Confirm | crate::menu::Intent::Nav(_)
            )
        }) && let Some(cue) = self
            .sound
            .catalog()
            .typed::<OneShot>(&self.cues, "menu_click")
        {
            self.sound.play_ui(cue);
        }
        // A per-frame snapshot so a menu never holds a live `&Mods`.
        let mods = ModRow::snapshot(&self.mods);
        let before = self.settings.clone();
        let mut effect = None;
        if let Screen::Menus(stack) = &mut self.screen {
            let mut ctx = Ctx {
                settings: &mut self.settings,
                saves: &self.saves,
                mods: &mods,
                session: &self.session,
            };
            effect = stack.update(&intents, &mut ctx);
        }
        // Apply every frame for immediate feedback and to show hardware clamps.
        self.settings.apply(eng);
        // Persist whenever a step (or a hardware clamp) moved a value.
        if self.settings != before {
            self.settings.save();
            self.sound.set_mix(self.settings.mix_change());
        }
        match effect {
            Some(effect) => self.handle_effect(eng, effect),
            None => false,
        }
    }

    /// Interpret one menu effect. Returns `true` only for Quit.
    fn handle_effect(&mut self, eng: &mut Engine, effect: AppEffect) -> bool {
        match effect {
            AppEffect::NewWorld => self.start_new_world(eng),
            AppEffect::Load(name) => self.load_world(eng, &name),
            AppEffect::Host(info) => {
                self.session.port = info.port.to_string();
                self.session.name = info.name.clone();
                self.session.save();
                self.start_host(eng, info);
            }
            AppEffect::Join(info) => {
                self.session.address = info.host.clone();
                self.session.port = info.port.to_string();
                self.session.name = info.name.clone();
                self.session.save();
                self.start_join(eng, info);
            }
            AppEffect::ToggleMod(index) => self.mods.toggle(index),
            AppEffect::StepModKnob {
                mod_index,
                knob,
                delta,
            } => self.mods.step_knob(mod_index, knob, delta),
            AppEffect::Quit => return true,
        }
        false
    }

    /// Return to the start menu with an optional notice (e.g. a failed connect).
    fn return_to_menu(&mut self, notice: Option<String>) {
        self.sound.leave_world();
        // The director's trace-derived state and mic persist on App across worlds
        // (unlike the old per-Game fields), so they need an explicit reset here.
        self.audio.enter_world();
        self.active = None;
        self.saves = save::list();
        self.screen = Screen::Menus(MenuStack::new(Framed::boxed(MainMenu::with_notice(notice))));
    }

    /// Spin up a fresh integrated server and join it on loopback. Any previous host
    /// is stopped first so its port is free to reuse.
    fn start_host(&mut self, eng: &mut Engine, info: HostInfo) {
        if let Some(previous) = self.host.take() {
            previous.stop();
            thread::sleep(Duration::from_millis(150));
        }
        let seed = fresh_seed();
        let config = Config {
            password: info.password.clone(),
            seed,
            worldgen: self.mods.worldgen_kind(),
            diffusion: self.mods.diffusion_cfg(),
            ..Config::default()
        };
        match server::spawn(info.port, config) {
            Ok(handle) => {
                let port = handle.addr().port();
                self.host = Some(handle);
                match Connection::connect_kind(
                    "127.0.0.1",
                    port,
                    &info.name,
                    &info.password,
                    self.mods.worldgen_kind(),
                    self.mods.diffusion_cfg(),
                ) {
                    Ok(conn) => self.enter_net_game(eng, conn),
                    Err(e) => self.fail_to_menu(format!("hosted, but could not connect: {e}")),
                }
            }
            Err(e) => self.fail_to_menu(format!("could not host on port {}: {e}", info.port)),
        }
    }

    /// Connect to a remote server and enter its world.
    fn start_join(&mut self, eng: &mut Engine, info: JoinInfo) {
        match Connection::connect_kind(
            &info.host,
            info.port,
            &info.name,
            &info.password,
            self.mods.worldgen_kind(),
            self.mods.diffusion_cfg(),
        ) {
            Ok(conn) => self.enter_net_game(eng, conn),
            Err(e) => self.fail_to_menu(format!("could not join: {e}")),
        }
    }

    /// Join a remote world via an existing connection.
    fn enter_net_game(&mut self, eng: &mut Engine, conn: Connection) {
        // Lazy construction: the collision-safe spawn slab is prepared in
        // `enter_game`; streaming fills the remainder asynchronously.
        let world = World::with_kind_cfg(
            conn.seed(),
            self.mods.mask_render(self.settings.render_config()),
            self.mods.worldgen_kind(),
            self.mods.diffusion_cfg(),
            false,
        );
        let player = Player::new(conn.spawn());
        // A networked world is a live mirror, not a save — per-world mod state
        // starts clean, but the player's enable/disable choices persist.
        self.active = None;
        self.mods.reset_state();
        let game = Game::new(world, player, "multiplayer".to_string()).with_net(conn);
        self.enter_game(eng, game);
    }

    /// Report a connection/host failure and return to the menu.
    fn fail_to_menu(&mut self, message: String) {
        self.return_to_menu(Some(message));
    }

    /// Create a fresh world with a time-seeded generator and enter it.
    fn start_new_world(&mut self, eng: &mut Engine) {
        // Benchmarks pin the seed (`WATT_BENCH_SEED`, default when benching) so
        // fps/rss deltas measure the code, not terrain-lottery variance.
        let seed = match (&self.bench, std::env::var("WATT_BENCH_SEED")) {
            (_, Ok(s)) => s.parse().unwrap_or_else(|_| fresh_seed()),
            (Some(_), _) => 42,
            (None, _) => fresh_seed(),
        };
        // Lazy construction: `spawn_player` queries only a few surface columns
        // (generated on demand), and `enter_game` prepares the collision-safe
        // spawn slab — the previous eager default-volume generation is avoided.
        let world = World::with_kind_cfg(
            seed,
            self.mods.mask_render(self.settings.render_config()),
            self.mods.worldgen_kind(),
            self.mods.diffusion_cfg(),
            false,
        );
        let player = spawn_player(&world);
        let id = save::fresh_id();
        let now = save::unix_now();
        self.active = Some(ActiveSlot::new(
            id.clone(),
            SaveMeta {
                name: id.as_str().to_string(),
                seed,
                created: now,
                last_played: now,
                playtime_secs: 0,
                edit_count: 0,
            },
        ));

        // A new world starts from a clean default mod set (empty inventory, etc.);
        // the mod menu's enable/disable choices persist.
        self.mods.reset_state();
        self.enter_game(eng, Game::new(world, player, id.as_str().to_string()));
    }

    /// Load an existing save and enter it. Stays on the menu if loading fails.
    fn load_world(&mut self, eng: &mut Engine, name: &str) {
        let id = match SlotId::new(name) {
            Ok(id) => id,
            Err(e) => return self.fail_to_menu(format!("could not load {name}: {e}")),
        };
        self.mods.reset_state();
        let render = self.mods.mask_render(self.settings.render_config());
        match save::load(&id, &mut self.mods, |seed| {
            World::with_config_lazy(seed, render)
        }) {
            Ok((world, player, meta, report)) => {
                self.active = Some(ActiveSlot::new(id.clone(), meta));
                let mut game = Game::new(world, player, id.as_str().to_string());
                // Degraded loads still enter the world, but say so.
                if report.source == save::Source::Backup {
                    game.notify("* save was unreadable — restored from the backup");
                }
                if let Some((recovered, expected)) = report.salvage {
                    game.notify(format!(
                        "* save was damaged — recovered {recovered} of {expected} edits"
                    ));
                }
                self.enter_game(eng, game)
            }
            Err(e) => self.fail_to_menu(format!("could not load {name}: {e}")),
        }
    }

    /// Install a freshly built game as the active screen.
    fn enter_game(&mut self, eng: &mut Engine, mut game: Game) {
        // World-construction lanes apply on entry only, before streaming spins;
        // everything live-applicable goes through the same path `/gfx` uses.
        let render = self.mods.mask_render(self.settings.render_config());
        game.set_visual_mask(crate::mods::VisualMask::from_mods(&self.mods));
        game.world_mut()
            .set_render_lanes(render.occlusion, render.lod2);
        game.apply_settings(eng, &mut self.settings);
        // Saves and servers can place the player far from the pre-generated
        // origin; make the ground under them real before physics runs.
        let pos = game.player().position;
        game.world_mut().prepare_around(pos);
        game.on_enter(eng, &mut self.router);
        self.sound.enter_world();
        // The director's trace-derived state is world-scoped too, and must reset
        // in lockstep with `sound`; its occurrence clock stays monotone.
        self.audio.enter_world();
        self.screen = Screen::Playing(Box::new(game));
    }

    /// In-world update: run the game and handle autosave.
    fn update_playing(&mut self, eng: &mut Engine) {
        let dt = eng.frame_time() as f64;
        let Screen::Playing(game) = &mut self.screen else {
            return;
        };
        let signal = game.update(
            eng,
            &mut self.router,
            &mut self.mods,
            &mut self.settings,
            &mut self.sound,
            &mut self.audio,
        );
        if let Signal::ExitToMenu = signal {
            self.flush_save();
            if let Screen::Playing(game) = &mut self.screen {
                // Return the world's GPU meshes to the engine before dropping it.
                game.free_gpu(eng);
            }
            eng.enable_cursor();
            self.return_to_menu(None); // drops the Box<Game>
            return;
        }
        // Periodic autosave on edits; bench/multiplayer never save.
        if self.bench.is_some() {
            return;
        }
        let Some(active) = &mut self.active else {
            return;
        };
        active.playtime += dt;
        active.meta.playtime_secs = active.playtime as u64;
        // Disabled autosave performs no polling and no serialization — the
        // explicit save on clean world exit (`flush_save`) remains.
        if !self.settings.autosave {
            return;
        }
        let ActiveSlot {
            id,
            meta,
            autosaver,
            ..
        } = active;
        if let Tick::Finished(Err(e)) = autosaver.poll() {
            game.notify(format!("* autosave failed: {e}"));
        }
        // The interval gate is cleared on the attempt, not on success, so a
        // failing write doesn't retry every frame.
        if autosaver.wants_write(game.world().edit_generation()) && game.autosave_due() {
            game.mark_autosave();
            let started = autosaver.start(id, game.world().edit_generation(), || {
                save::encode_current(game.world(), game.player(), &self.mods, meta.clone())
            });
            if let Tick::Finished(Err(e)) = started {
                game.notify(format!("* autosave failed: {e}"));
            }
        }
    }

    /// Synchronously write the singleplayer world (networked and bench worlds
    /// are never saved).
    fn flush_save(&mut self) {
        let Screen::Playing(game) = &self.screen else {
            return;
        };
        if game.is_multiplayer() || self.bench.is_some() {
            return;
        }
        let Some(active) = &mut self.active else {
            return;
        };
        active.meta.playtime_secs = active.playtime as u64;
        let ActiveSlot {
            id,
            meta,
            autosaver,
            ..
        } = active;
        if let Err(e) = autosaver.flush_now(id, game.world().edit_generation(), || {
            save::encode_current(game.world(), game.player(), &self.mods, meta.clone())
        }) {
            eprintln!("could not save {id}: {e}");
        }
    }

    /// Draw the active screen (game or menu).
    fn draw(&mut self, eng: &mut Engine) {
        let (w, h) = (eng.screen_width(), eng.screen_height());
        if let Screen::Playing(game) = &mut self.screen {
            let fov = self.settings.fov;
            let shake = self.settings.shake;
            game.draw(eng, &mut self.mods, fov, shake);
            return;
        }
        // Mods snapshot avoids borrow conflict between theme and view.
        let mods = ModRow::snapshot(&self.mods);
        let fallback = DefaultTheme;
        let theme: &dyn MenuTheme = self.mods.menu_theme().unwrap_or(&fallback);
        let mut f = eng.begin_frame(MENU_CLEAR.to_linear());
        if let Screen::Menus(stack) = &self.screen {
            let ctx = Ctx {
                settings: &mut self.settings,
                saves: &self.saves,
                mods: &mods,
                session: &self.session,
            };
            stack.draw(&ctx, theme, &mut f, w, h);
        }
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

/// A world seed from the wall clock, so each new world differs.
fn fresh_seed() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(1)
}

/// Spawn the player just above dry land near the world origin, so they drop and
/// land on solid ground instead of sinking into an ocean/lake column that
/// happens to sit at (0, 0). Spirals outward from the origin for the first
/// column above sea level, mirroring `net::server::spawn_point`.
fn spawn_player(world: &World) -> Player {
    let sea = world.sea_level();
    for r in 0..64 {
        for (dx, dz) in [
            (r, 0),
            (0, r),
            (-r, 0),
            (0, -r),
            (r, r),
            (-r, -r),
            (r, -r),
            (-r, r),
        ] {
            let (x, z) = (dx * 8, dz * 8);
            let h = world.surface_y(x, z);
            if h > sea {
                return Player::new(DVec3::new(x as f64 + 0.5, h as f64 + 3.0, z as f64 + 0.5));
            }
        }
    }
    let h = world.surface_y(0, 0).max(sea);
    Player::new(DVec3::new(0.5, h as f64 + 3.0, 0.5))
}
