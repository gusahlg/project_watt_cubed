//! app.rs owns the top-level state machine: the start menu, an in-world
//! [`Game`], the mod menu, the host/join forms, and the graphics settings
//! screen. It routes each engine frame to the active screen, creates and
//! loads worlds, and autosaves when leaving one.
//!
//! The window itself belongs to the engine: [`App::run`] hands a per-frame
//! closure to [`voxel_engine::run`], which is the moral equivalent of the old
//! raylib `while !window_should_close()` loop.
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use voxel_engine::{Color, DVec3, Engine};

use crate::audio::{AudioDirector, CuePalette, CueSymbols, OneShot, SoundConfig, SoundSystem};
use crate::benchmark::{Benchmark, Step as BenchmarkStep};
use crate::game::{Game, Signal};
use crate::input::router::{Context, Router, View};
use crate::menu::menus::{ModsMenu, SettingsHub};
use crate::menu::start::{StartFacts, StartRoot, VERSION};
use crate::menu::theme::{DefaultTheme, MenuTheme};
use crate::menu::{AppEffect, Ctx, Framed, HostInfo, JoinInfo, MenuStack, ModRow};
use crate::mods::{ChoicesFlush, Mods};
use crate::net::client::Connection;
use crate::net::server::{self, Config, ServerHandle};
use crate::player::Player;
use crate::save::{self, Autosaver, SaveMeta, Slot, SlotId, Tick};
use crate::session::Session;
use crate::settings::Settings;
use crate::world::diffusion::DiffusionCfg;
use crate::world::World;

const STARTING_WINDOW_WIDTH: u32 = 1280;
const STARTING_WINDOW_HEIGHT: u32 = 720;
/// Background for every non-world screen.
const MENU_CLEAR: Color = Color::new(18, 20, 28, 255);
/// Menu frame cap. The engine sleeps until the deadline (`WaitUntil`), so
/// 120 Hz is ~8 ms worst-case input-to-photon; a busy-wait cap would want 240.
const MENU_FPS_CAP: u32 = 120;

/// Vsync and fps cap for this screen. Bench is uncapped; menus cap at
/// [`MENU_FPS_CAP`] with vsync off; in-world uses the saved settings.
fn pacing(in_world: bool, bench: bool, settings: &Settings) -> (bool, u32) {
    if bench {
        (false, 0)
    } else if in_world {
        (settings.vsync, settings.max_fps)
    } else {
        (false, MENU_FPS_CAP)
    }
}

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
    /// Graphics settings, persisted as `settings.cfg` under the config root.
    settings: Settings,
    /// Last-used connection details, persisted as `session.cfg` under the config root.
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
    /// Last stall-detector log, so a hung frame names itself once per window.
    last_stall_log: Option<Instant>,
    /// Debounces `mods.cfg` writes (held Left/Right would otherwise rewrite ~22×/s).
    choices_flush: ChoicesFlush,
    clock: Instant,
    /// True while the Mods screen is on the menu stack.
    mods_open: bool,
    mods_save_error: Option<String>,
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
        crate::paths::Paths::init(None);
        let mut mods = Mods::with_defaults();
        mods.load_choices();
        let pins = Benchmark::mod_pins_from_env();
        mods.apply_bench_env(pins.worldgen_diffusion, pins.visuals_core);
        let saves = save::list();
        let mut settings = Settings::load();
        let (caps, display) = crate::benchmark::graphics_caps();
        settings.set_device_caps(caps, display);
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
        let screen = Screen::Menus(Self::start_stack(&mods, &saves, &session, None, false));
        Self {
            saves,
            active: None,
            router: Router::new(),
            mods,
            screen,
            host: None,
            settings,
            session,
            bench,
            sound,
            cues,
            audio,
            last_stall_log: None,
            choices_flush: ChoicesFlush::new(),
            clock: Instant::now(),
            mods_open: false,
            mods_save_error: None,
        }
    }

    fn now_ms(&self) -> u64 {
        self.clock.elapsed().as_millis() as u64
    }

    fn persist_mod_choices(&mut self) {
        match self.mods.save_choices() {
            Ok(()) => self.mods_save_error = None,
            Err(e) => self.mods_save_error = Some(e.to_string()),
        }
    }

    fn flush_mod_choices_if_dirty(&mut self) {
        if self.choices_flush.take() {
            self.persist_mod_choices();
        }
    }

    /// Open the window and run until the player quits (menu or close button).
    pub fn run(self) {
        let mut app = self;
        // Starts on menus (or uncapped if this process is a bench).
        let (vsync, target_fps) = pacing(false, app.bench.is_some(), &app.settings);
        let (extent_w, extent_h) = if app.settings.fullscreen {
            (app.settings.startup_display_w, app.settings.startup_display_h)
        } else {
            (STARTING_WINDOW_WIDTH, STARTING_WINDOW_HEIGHT)
        };
        let session_gfx = app.settings.session_graphics(extent_w, extent_h);
        if let Some(line) = session_gfx.notice.as_ref() {
            eprintln!("{line}");
            app.settings.vram_notice = session_gfx.notice.clone();
        }
        app.settings
            .note_render_extent(extent_w, extent_h, session_gfx.render_scale);
        let config = voxel_engine::Config {
            title: "Project Watt Cubed".into(),
            width: STARTING_WINDOW_WIDTH,
            height: STARTING_WINDOW_HEIGHT,
            target_fps,
            vsync,
            msaa: session_gfx.msaa,
            render_scale: session_gfx.render_scale,
            resizable: true,
            fullscreen: app.settings.fullscreen,
            // Engine-side render lanes from the effective (mod-masked) config.
            flags: app.mods.effective_render(&app.settings).engine_flags(),
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
            self.flush_mod_choices_if_dirty();
            self.flush_save();
            return false;
        }

        let watch = cfg!(debug_assertions) || self.bench.is_some();
        let t0 = watch.then(Instant::now);

        self.sound.service();

        if self.bench.is_some() && !self.bench_frame(eng) {
            self.note_frame_stall(t0, Duration::ZERO);
            return false;
        }

        let t_update = watch.then(Instant::now);
        let quit = match self.screen {
            Screen::Menus(_) => self.update_menus(eng),
            Screen::Playing(_) => {
                self.update_playing(eng);
                false
            }
        };
        let update_dt = t_update.map(|t| t.elapsed()).unwrap_or_default();
        if quit {
            if self.bench.is_none() {
                self.settings.save();
            }
            self.flush_mod_choices_if_dirty();
            self.flush_save();
            self.note_frame_stall(t0, update_dt);
            return false;
        }
        // Applied MSAA/scale from engine create (and later recreates) before
        // we push the session request, so a fallback cannot be overwritten.
        self.settings.sync_engine_applied(eng);
        // VRAM guard + live settings: one push per frame so a resize cannot
        // allocate MSAA/scale the probe already refused.
        self.settings.apply(eng);
        eng.set_flags(self.mods.effective_render(&self.settings).engine_flags());
        // Apply only on change: a SetVsync every menu frame was waking the
        // render thread even when the mode was already correct.
        let in_world = matches!(self.screen, Screen::Playing(_));
        let (want_vsync, want_fps) = pacing(in_world, self.bench.is_some(), &self.settings);
        if eng.vsync() != want_vsync {
            eng.set_vsync(want_vsync);
        }
        if eng.target_fps() != want_fps {
            eng.set_target_fps(want_fps);
        }
        self.draw(eng);
        self.note_frame_stall(t0, update_dt);
        true
    }

    /// If this frame exceeded 250 ms, print `entry_debug` and phase timings
    /// once per 5 s so the next stall names itself. Debug builds and
    /// `WATT_BENCH` only.
    fn note_frame_stall(&mut self, start: Option<Instant>, update_dt: Duration) {
        let Some(start) = start else {
            return;
        };
        let dt = start.elapsed();
        if dt < Duration::from_millis(250) {
            return;
        }
        let now = Instant::now();
        if self
            .last_stall_log
            .is_some_and(|t| now.duration_since(t) < Duration::from_secs(5))
        {
            return;
        }
        self.last_stall_log = Some(now);
        let draw_ms = dt.saturating_sub(update_dt).as_secs_f64() * 1000.0;
        let update_ms = update_dt.as_secs_f64() * 1000.0;
        match &self.screen {
            Screen::Playing(game) => {
                eprintln!(
                    "frame stall {}ms update={update_ms:.1}ms draw={draw_ms:.1}ms\n  {}\n  {}",
                    dt.as_millis(),
                    game.world().entry_debug(),
                    game.phase_debug()
                );
            }
            Screen::Menus(_) => {
                eprintln!(
                    "frame stall {}ms update={update_ms:.1}ms draw={draw_ms:.1}ms (menus)",
                    dt.as_millis()
                );
            }
        }
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
        // One unsampled frame after Complete has presented; capture that image
        // (blocking) before the report so the readback is outside the samples.
        if self
            .bench
            .as_ref()
            .expect("bench exists")
            .measurement_complete()
        {
            return self.finish_bench(eng);
        }
        {
            let Screen::Playing(game) = &mut self.screen else {
                return true;
            };
            // A slow spin (`WATT_BENCH_YAW`, default 0.4 rad/s; 0 = static) sweeps
            // the frustum; an optional flight along +X (`WATT_BENCH_MOVE`)
            // exercises paths a parked camera never touches.
            let yaw_rate = self.bench.as_ref().expect("bench exists").yaw_rate() as f32;
            game.player_mut().orientation.yaw += yaw_rate * dt;
            self.bench
                .as_ref()
                .expect("bench exists")
                .apply_move(game.player_mut(), dt);

            let (ready, gauges) = {
                let bench = self.bench.as_mut().expect("bench exists");
                bench.poll_world(game.world())
            };
            let step = self.bench.as_mut().expect("bench exists").step(dt, ready, gauges);
            match step {
                BenchmarkStep::ReadyTimeout => {
                    eprintln!("{}", game.world().entry_debug());
                    return true;
                }
                BenchmarkStep::Warming => {
                    if !ready && self.bench.as_mut().expect("bench exists").wait_log_due()
                    {
                        eprintln!(
                            "benchmark: waiting for world ({})",
                            game.world().entry_debug()
                        );
                    }
                    return true;
                }
                BenchmarkStep::Measuring => return true,
                BenchmarkStep::Complete => {
                    if self
                        .bench
                        .as_ref()
                        .expect("bench exists")
                        .screenshot_path()
                        .is_some()
                    {
                        return true;
                    }
                }
            }
        }
        self.finish_bench(eng)
    }

    /// Capture the last presented frame if requested, then emit the report.
    fn finish_bench(&mut self, eng: &mut Engine) -> bool {
        let Screen::Playing(game) = &self.screen else {
            return true;
        };
        self.bench
            .as_ref()
            .expect("bench exists")
            .capture_screenshot(eng);
        let report = self.bench.as_mut().expect("bench exists").finish(
            &self.settings,
            eng,
            game.world(),
            game.player().position,
        );
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
        let mods_save_error = self.mods_save_error.clone();
        let before = self.settings.clone();
        let depth_before = match &self.screen {
            Screen::Menus(stack) => stack.depth(),
            _ => 0,
        };
        let now_ms = self.now_ms();
        let mut effect = None;
        if let Screen::Menus(stack) = &mut self.screen {
            let mut ctx = Ctx {
                settings: &mut self.settings,
                saves: &self.saves,
                mods: &mods,
                session: &self.session,
                mods_save_error: mods_save_error.as_deref(),
            };
            effect = stack.update(&intents, &mut ctx);
        }
        // Persist whenever a step (or a hardware clamp) moved a value.
        if self.settings != before {
            self.settings.save();
            self.sound.set_mix(self.settings.mix_change());
        }
        if self.mods_open {
            let depth_after = match &self.screen {
                Screen::Menus(stack) => stack.depth(),
                _ => 0,
            };
            if depth_after < depth_before {
                self.mods_open = false;
                self.flush_mod_choices_if_dirty();
            }
        }
        let quit = match effect {
            Some(effect) => self.handle_effect(eng, effect),
            None => false,
        };
        if self.choices_flush.poll(now_ms) {
            self.persist_mod_choices();
        }
        quit
    }

    /// Interpret one menu effect. Returns `true` only for Quit.
    fn handle_effect(&mut self, eng: &mut Engine, effect: AppEffect) -> bool {
        match effect {
            AppEffect::NewWorld => self.start_new_world(eng),
            AppEffect::Load(id) => self.load_world(eng, &id),
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
            AppEffect::Settings => {
                if let Screen::Menus(stack) = &mut self.screen {
                    stack.push(Framed::boxed(SettingsHub));
                }
            }
            AppEffect::Mods => {
                if let Screen::Menus(stack) = &mut self.screen {
                    stack.push(Framed::boxed(ModsMenu));
                    self.mods_open = true;
                }
            }
            AppEffect::ToggleMod(index) => {
                self.mods.toggle(index);
                self.choices_flush.mark(self.now_ms());
            }
            AppEffect::StepModKnob {
                mod_index,
                knob,
                delta,
            } => {
                self.mods.step_knob(mod_index, knob, delta);
                self.choices_flush.mark(self.now_ms());
            }
            AppEffect::SetGroup { id, on } => {
                self.mods.set_group_enabled(id, on);
                self.choices_flush.mark(self.now_ms());
            }
            AppEffect::Quit => {
                self.flush_mod_choices_if_dirty();
                return true;
            }
        }
        false
    }

    /// Open the start screen: first enabled start-screen mod, else the core fallback.
    fn start_stack(
        mods: &Mods,
        saves: &[Slot],
        session: &Session,
        notice: Option<&str>,
        hosting: bool,
    ) -> MenuStack {
        let facts = StartFacts {
            saves,
            session,
            version: VERSION,
            hosting,
            notice,
        };
        let inner = mods
            .start_screen(&facts)
            .unwrap_or_else(|| crate::menu::start::fallback(&facts));
        MenuStack::new(StartRoot::wrap(inner, hosting))
    }

    /// Return to the start menu with an optional notice (e.g. a failed connect).
    fn return_to_menu(&mut self, notice: Option<String>) {
        self.sound.leave_world();
        // The director's trace-derived state and mic persist on App across worlds
        // (unlike the old per-Game fields), so they need an explicit reset here.
        self.audio.enter_world();
        self.active = None;
        self.saves = save::list();
        self.screen = Screen::Menus(Self::start_stack(
            &self.mods,
            &self.saves,
            &self.session,
            notice.as_deref(),
            self.host.is_some(),
        ));
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
            diffusion: diffusion_from_mods(&self.mods),
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
                    diffusion_from_mods(&self.mods),
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
            diffusion_from_mods(&self.mods),
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
            self.mods.effective_render(&self.settings),
            conn.worldgen(),
            conn.diffusion(),
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
            self.mods.effective_render(&self.settings),
            self.mods.worldgen_kind(),
            diffusion_from_mods(&self.mods),
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
    fn load_world(&mut self, eng: &mut Engine, id: &SlotId) {
        self.mods.reset_state();
        let render = self.mods.effective_render(&self.settings);
        match save::load(id, &mut self.mods, |seed, kind, cfg| {
            // The save header names the generator; the InfiniteDiffusion mod's
            // enabled flag only chooses the next *new* world.
            World::with_kind_cfg(seed, render, kind, cfg, false)
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
            Err(e) => self.fail_to_menu(format!("could not load {id}: {e}")),
        }
    }

    /// Install a freshly built game as the active screen.
    fn enter_game(&mut self, eng: &mut Engine, mut game: Game) {
        // World-construction lanes apply on entry only, before streaming spins;
        // everything live-applicable goes through the same path `/gfx` uses.
        let render = self.mods.effective_render(&self.settings);
        game.set_visual_mask(self.mods.visual_mask());
        game.world_mut()
            .set_render_lanes(render.occlusion, render.lod2);
        game.apply_settings(eng, &mut self.settings);
        // Saves and servers can place the player far from the pre-generated
        // origin; request the collision slab (physics freezes until it lands).
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
            let started = autosaver.start(
                id,
                game.world().edit_generation(),
                save::snapshot(game.world(), game.player(), &self.mods, meta.clone()),
            );
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
                mods_save_error: self.mods_save_error.as_deref(),
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

/// Parse the winning worldgen payload as diffusion knobs (classic has none).
fn diffusion_from_mods(mods: &Mods) -> DiffusionCfg {
    mods.worldgen_config()
        .as_deref()
        .map(DiffusionCfg::from_text)
        .unwrap_or_default()
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
///
/// Classic: up to 512 `height()` probes (~5 ms). Diffusion: 8 rings of 16×16
/// tiles via [`World::heights_16`], so the field is sampled by rectangle.
fn spawn_player(world: &World) -> Player {
    let sea = world.sea_level();
    if world.worldgen_kind() == "diffusion" {
        return spawn_player_diffusion(world, sea);
    }
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

fn spawn_player_diffusion(world: &World, sea: i32) -> Player {
    // Same 8-direction spiral as classic, bounded to 8 rings. Each unique
    // 16×16 chunk column is one `heights_16` sample (one field tile).
    let mut seen = [(i32::MAX, i32::MAX); 32];
    let mut n = 0usize;
    for r in 0i32..8 {
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
            let cx = (dx * 8).div_euclid(16);
            let cz = (dz * 8).div_euclid(16);
            if seen[..n].contains(&(cx, cz)) {
                continue;
            }
            seen[n] = (cx, cz);
            n += 1;
            let heights = world.heights_16(cx, cz);
            for lz in 0..16 {
                for lx in 0..16 {
                    let h = heights[lx + lz * 16];
                    if h > sea {
                        let x = cx * 16 + lx as i32;
                        let z = cz * 16 + lz as i32;
                        return Player::new(DVec3::new(
                            x as f64 + 0.5,
                            h as f64 + 3.0,
                            z as f64 + 0.5,
                        ));
                    }
                }
            }
        }
    }
    let h = world.surface_y(0, 0).max(sea);
    Player::new(DVec3::new(0.5, h as f64 + 3.0, 0.5))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_config::RenderConfig;
    use crate::world::diffusion::DiffusionCfg;
    use crate::world::generation::WorldgenKind;

    #[test]
    fn spawn_player_sits_above_the_surface() {
        let world = World::with_config_lazy(7, RenderConfig::default());
        let p = spawn_player(&world);
        let ground = world.surface_y(
            crate::math::block_coord(p.position.x),
            crate::math::block_coord(p.position.z),
        );
        assert!(p.position.y > ground as f64);
    }

    #[test]
    fn spawn_player_probe_cost() {
        use std::hint::black_box;
        let classic = World::with_kind_cfg(
            1,
            RenderConfig::default(),
            WorldgenKind::Classic,
            DiffusionCfg::default(),
            false,
        );
        let _ = black_box(spawn_player(&classic));
        let t = Instant::now();
        let _ = black_box(spawn_player(&classic));
        let classic_ms = t.elapsed().as_secs_f64() * 1000.0;

        let diffusion = World::with_kind_cfg(
            1,
            RenderConfig::default(),
            WorldgenKind::Diffusion,
            DiffusionCfg::default(),
            false,
        );
        let _ = black_box(spawn_player(&diffusion));
        let t = Instant::now();
        let _ = black_box(spawn_player(&diffusion));
        let diffusion_ms = t.elapsed().as_secs_f64() * 1000.0;

        println!("spawn_player classic={classic_ms:.2}ms diffusion={diffusion_ms:.2}ms");
        assert!(
            classic_ms < 50.0,
            "classic spawn probes should be a few ms, got {classic_ms:.2}"
        );
        assert!(
            diffusion_ms < 250.0,
            "diffusion spawn must stay under a frame, got {diffusion_ms:.2}"
        );
    }

    #[test]
    fn pacing_menus_world_and_bench() {
        let mut settings = Settings::default();
        settings.vsync = true;
        settings.max_fps = 60;
        assert_eq!(
            pacing(false, false, &settings),
            (false, MENU_FPS_CAP),
            "menus: vsync off, cap 120"
        );
        assert_eq!(
            pacing(true, false, &settings),
            (true, 60),
            "in-world: saved vsync and max_fps"
        );
        assert_eq!(pacing(false, true, &settings), (false, 0), "bench: off, 0");
        assert_eq!(pacing(true, true, &settings), (false, 0), "bench wins in-world too");

        settings.vsync = false;
        settings.max_fps = 0;
        assert_eq!(pacing(false, false, &settings), (false, MENU_FPS_CAP));
        assert_eq!(pacing(true, false, &settings), (false, 0));
        settings.max_fps = 144;
        assert_eq!(pacing(true, false, &settings), (false, 144));
    }
}
