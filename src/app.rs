//! app.rs owns the top-level state machine: the start menu, an in-world
//! [`Game`], the mod menu, the host/join forms, and the graphics settings
//! screen. It routes each engine frame to the active screen, creates and
//! loads worlds (off the render thread, in `entry`), and autosaves when
//! leaving one.
//!
//! The window itself belongs to the engine: [`App::run`] hands a per-frame
//! closure to [`voxel_engine::run`], which is the moral equivalent of the old
//! raylib `while !window_should_close()` loop.
mod entry;

use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use voxel_engine::{Color, DVec3, Engine};

use crate::audio::{AudioService, AudioView, CueSymbols, GameEvent, ModLink, PeerAudio, SoundConfig, SoundSystem};
use crate::benchmark::{Benchmark, Step as BenchmarkStep};
use crate::game::{Game, Signal};
use crate::input::router::{Context, Router, View};
use crate::menu::menus::{ModsMenu, SettingsHub};
use crate::menu::start::{StartFacts, StartRoot, VERSION};
use crate::menu::theme::{DefaultTheme, MenuTheme};
use crate::menu::{AppEffect, Ctx, Framed, HostInfo, JoinInfo, MenuStack, ModRow};
use crate::modding::{ActionSet, ChoicesFlush, GameBuild, ModDescriptor, Mods};
#[cfg(test)]
use crate::net::client::ConnectError;
use crate::net::client::{Connection, PendingConnect};
use crate::net::server::{self, Config, NoclipPolicy, ServerHandle, TeleportPolicy};
use crate::ui::{self, Anchor};
use crate::player::Player;
use crate::save::{self, Autosaver, SaveMeta, Slot, SlotId, Tick};
use crate::session::Session;
use crate::settings::Settings;
use crate::world::terrain::TerrainCfg;
use crate::world::World;
use entry::{Loading, Recipe};

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
    /// DNS, handshake, and Welcome, off the render thread.
    Connecting(ConnectJob),
    Playing(Box<Game>),
}

/// One join attempt. A mod refusal starts a second attempt and keeps `retried`.
struct ConnectJob {
    pending: PendingConnect,
    hosted: bool,
    retried: bool,
    host: String,
    port: u16,
    name: String,
    password: String,
    /// Shown once the retry joins.
    notice: Option<String>,
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
    /// Packages compiled into this executable. `Hello` reports the enabled ones.
    packages: Vec<ModDescriptor>,
    screen: Screen,
    /// A world building off the render thread. The menu under it stays as it was; Esc drops it.
    loading: Option<Loading>,
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
    /// Cue name → id table resolved once at catalog load. Mods name cues through it.
    cues: CueSymbols,
    /// Gait, the acoustic window, capture and voice sessions. Mods decide what plays.
    audio: AudioService,
    /// Last stall-detector log, so a hung frame names itself once per window.
    last_stall_log: Option<Instant>,
    /// Debounces `mods.cfg` writes (held Left/Right would otherwise rewrite ~22×/s).
    choices_flush: ChoicesFlush,
    clock: Instant,
    /// True while the Mods screen is on the menu stack.
    mods_open: bool,
    mods_save_error: Option<String>,
    /// Last graphics stamp pushed to the engine; `apply` runs only on change.
    gfx_applied: Option<GfxKey>,
}

/// Values [`Settings::apply`] and `set_flags` actually push. Compared so a
/// quiet/menu frame with unchanged settings does not touch the engine.
#[derive(Clone, Copy, PartialEq, Eq)]
struct GfxKey {
    w: u32,
    h: u32,
    fullscreen: bool,
    msaa: u32,
    scale_bits: u32,
    cull_faces: Option<bool>,
    flags: voxel_engine::RenderFlags,
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
    /// The app for one build: `build` lists the mod packages compiled into this executable.
    pub fn new(build: &GameBuild) -> Self {
        crate::paths::Paths::init(None);
        match build.environment() {
            Some(env) => eprintln!("PWC: {} mod packages, environment {env}", build.packages().len()),
            None if build.packages().is_empty() => eprintln!("PWC: vanilla build (no mod packages)"),
            None => eprintln!("PWC: {} mod packages", build.packages().len()),
        }
        // While the menu is up, so the first world's frame does not pay for it.
        crate::world::terrain::prewarm();
        let packages = build.packages().to_vec();
        let mut mods = Mods::from_build(build);
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
        let audio = AudioService::new();
        let screen = Screen::Menus(Self::start_stack(&mods, &saves, &session, None, false));
        Self {
            saves,
            active: None,
            router: Router::new(),
            mods,
            packages,
            screen,
            loading: None,
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
            gfx_applied: None,
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
            Screen::Menus(_) if self.loading.is_some() => {
                self.update_loading(eng);
                false
            }
            Screen::Menus(_) => self.update_menus(eng),
            Screen::Connecting(_) => {
                self.update_connecting(eng);
                false
            }
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
        // VRAM guard + live settings: push only when the stamp moves so a
        // quiet frame or an idle menu does not wake the render thread. A
        // resize changes the stamp, so a fallback cannot be overwritten.
        self.push_gfx(eng);
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
                    "frame stall {}ms update={update_ms:.1}ms draw={draw_ms:.1}ms ({})",
                    dt.as_millis(),
                    if self.loading.is_some() { "loading" } else { "menus" }
                );
            }
            Screen::Connecting(_) => {
                eprintln!(
                    "frame stall {}ms update={update_ms:.1}ms draw={draw_ms:.1}ms (connecting)",
                    dt.as_millis()
                );
            }
        }
    }

    fn gfx_key(&self, w: u32, h: u32) -> GfxKey {
        let session = self.settings.session_graphics(w, h);
        GfxKey {
            w,
            h,
            fullscreen: self.settings.fullscreen,
            msaa: session.msaa,
            scale_bits: session.render_scale.to_bits(),
            cull_faces: self.settings.cull_faces,
            flags: self.mods.effective_render(&self.settings).engine_flags(),
        }
    }

    fn push_gfx(&mut self, eng: &mut Engine) {
        let w = eng.screen_width().max(1) as u32;
        let h = eng.screen_height().max(1) as u32;
        let key = self.gfx_key(w, h);
        if self.gfx_applied.as_ref() == Some(&key) {
            return;
        }
        // Applied MSAA/scale from engine create (and later recreates) before
        // we push the session request, so a fallback cannot be overwritten.
        self.settings.sync_engine_applied(eng);
        self.settings.apply(eng);
        eng.set_flags(key.flags);
        self.gfx_applied = Some(key);
        #[cfg(test)]
        crate::alloc_count::note_engine(crate::alloc_count::EngineCall::SettingsApply);
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
            let (pos, look, day) = {
                let bench = self.bench.as_mut().expect("bench exists");
                bench.begin();
                (bench.position(), bench.look(), bench.day())
            };
            // Uncapped and unsynced, or the bench measures the throttle.
            self.settings.vsync = false;
            self.settings.max_fps = 0;
            self.settings.apply(eng);
            self.start_new_world(eng);
            if let Screen::Playing(game) = &mut self.screen {
                game.set_input_locked(true);
                if let Some(day) = day {
                    game.set_day(day);
                }
                // Far-coordinate bench: park the player at the requested position
                // with the ground under them made real, and give streaming a
                // little extra warmup to catch up before sampling starts.
                if let Some((yaw, pitch)) = look {
                    game.player_mut().orientation.yaw = yaw;
                    game.player_mut().orientation.pitch = pitch;
                }
                if let Some(pos) = pos {
                    game.player_mut().position = pos;
                    game.player_mut().set_flying(true);
                    game.world_mut().prepare_around(pos);
                    // Stand up along the local pull there, as a teleport does (any face of any body).
                    let weightless = 0.02 * crate::player::STANDARD_GRAVITY;
                    let pull = game.world().gravity_at(pos);
                    game.player_mut().gravity = pull.accel;
                    if let Some(up) = pull.up(weightless) {
                        game.player_mut().snap_up(up);
                    }
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
            let rendered = eng.frames_rendered();
            let coalesced = eng.frames_coalesced();
            let step = self
                .bench
                .as_mut()
                .expect("bench exists")
                .step(dt, ready, gauges, rendered, coalesced);
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
        // One click: confirm wins when both a confirm and a navigation landed together.
        let click = if intents.iter().any(|intent| matches!(intent, crate::menu::Intent::Confirm)) {
            Some(GameEvent::UiConfirm)
        } else if intents.iter().any(|intent| matches!(intent, crate::menu::Intent::Nav(_))) {
            Some(GameEvent::UiNavigate)
        } else {
            None
        };
        self.fan_menu_audio(dt, click);
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
            AppEffect::DeleteWorld(id) => {
                if let Err(e) = save::delete(&id) {
                    eprintln!("could not delete world {id}: {e}");
                }
                self.saves = save::list();
            }
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
                if self.mods.toggle(index) {
                    self.choices_flush.mark(self.now_ms());
                }
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
                if self.mods.set_group_enabled(id, on) {
                    self.choices_flush.mark(self.now_ms());
                }
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

    /// Menus have no world. The hook still runs, so a mod can play a UI cue.
    fn fan_menu_audio(&mut self, dt: f32, event: Option<GameEvent>) {
        const EMPTY: &[PeerAudio] = &[];
        const NO_IDS: &[&str] = &[];
        let view = AudioView {
            dt,
            pos: DVec3::ZERO,
            peers: EMPTY,
            in_world: false,
            voice_enabled: self.settings.voice_enabled,
            hear_voice: self.settings.voice_incoming,
            actions: ActionSet::NONE,
            ids: NO_IDS,
        };
        let mut link = ModLink::idle();
        {
            let mut api = self.audio.api(&mut self.sound, &self.cues, None, None);
            if let Some(event) = event.as_ref() {
                self.mods.on_game_event(event, &mut api);
            }
            self.mods.on_audio(&view, &mut api, &mut link);
        }
        self.audio.settle_menu();
    }

    /// Tell the mods the world is changing, before the service drops its sessions.
    fn fan_world_edge(&mut self, event: GameEvent) {
        const EMPTY: &[PeerAudio] = &[];
        const NO_IDS: &[&str] = &[];
        let in_world = matches!(event, GameEvent::EnterWorld);
        let view = AudioView {
            dt: 0.0,
            pos: DVec3::ZERO,
            peers: EMPTY,
            in_world,
            voice_enabled: self.settings.voice_enabled,
            hear_voice: self.settings.voice_incoming,
            actions: ActionSet::NONE,
            ids: NO_IDS,
        };
        let mut link = ModLink::idle();
        {
            let mut api = self.audio.api(&mut self.sound, &self.cues, None, None);
            self.mods.on_game_event(&event, &mut api);
            self.mods.on_audio(&view, &mut api, &mut link);
        }
        self.audio.settle_menu();
    }

    /// Return to the start menu with an optional notice (e.g. a failed connect).
    fn return_to_menu(&mut self, notice: Option<String>) {
        // Mods a server turned off for the session come back on leave.
        // The hold was never written to mods.cfg.
        self.mods.release_server();
        self.fan_world_edge(GameEvent::LeaveWorld);
        self.sound.leave_world();
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

    /// The most recent readable save, or a new slot file when the list is empty.
    fn host_world_path(&self) -> PathBuf {
        host_save_path(&self.saves)
    }

    /// Spin up the integrated server on the most recent save and join it on loopback.
    /// Any previous host is stopped first so its port is free. The host is an
    /// operator and teleport stays open. A stored seed and generator win.
    fn start_host(&mut self, _eng: &mut Engine, info: HostInfo) {
        if let Some(previous) = self.host.take() {
            previous.stop();
        }
        let config = Config {
            password: info.password.clone(),
            seed: fresh_seed(),
            worldgen: self.mods.worldgen_kind(),
            terrain: terrain_cfg_from_mods(&self.mods),
            teleport: TeleportPolicy::All,
            noclip: NoclipPolicy::All,
            world: Some(self.host_world_path()),
            ops: vec![info.name.clone()],
            warn_world_overrides: false,
            ..Config::default()
        };
        match server::spawn(info.port, config) {
            Ok(handle) => {
                let port = handle.addr().port();
                self.host = Some(handle);
                self.open_connect("127.0.0.1", port, &info.name, &info.password, true);
            }
            Err(e) => self.fail_to_menu(format!("could not host on port {}: {e}", info.port)),
        }
    }

    /// Connect to a remote server. The attempt runs behind the connecting screen.
    fn start_join(&mut self, _eng: &mut Engine, info: JoinInfo) {
        self.open_connect(&info.host, info.port, &info.name, &info.password, false);
    }

    /// Start one attempt. The render thread polls it; Cancel calls [`PendingConnect::cancel`].
    fn open_connect(&mut self, host: &str, port: u16, name: &str, password: &str, hosted: bool) {
        let reports = self.mods.enabled_package_reports(&self.packages);
        let pending = Connection::begin_connect(host, port, name, password, &reports);
        self.screen = Screen::Connecting(ConnectJob {
            pending,
            hosted,
            retried: false,
            host: host.to_string(),
            port,
            name: name.to_string(),
            password: password.to_string(),
            notice: None,
        });
    }

    /// Poll the attempt. Cancel returns to the menu and stops a host we started.
    /// A mod refusal retries once; any other failure leaves a spawned host running.
    fn update_connecting(&mut self, eng: &mut Engine) {
        if self.cancel_pressed(eng) {
            self.cancel_connect();
            return;
        }
        let outcome = match &mut self.screen {
            Screen::Connecting(job) => job.pending.poll(),
            _ => return,
        };
        let Some(result) = outcome else { return };
        let standby = self.standby_menu();
        let Screen::Connecting(job) = std::mem::replace(&mut self.screen, Screen::Menus(standby)) else {
            return;
        };
        match result {
            Ok(conn) => {
                let render = self.mods.effective_render(&self.settings);
                self.begin_loading(eng, Loading::join(conn, render, job.notice, job.hosted));
            }
            Err(err) if !err.mods_denied.is_empty() && !job.retried => {
                let notice = mod_hold_notice(&self.packages, &err.mods_denied);
                self.mods.hold_packages(&err.mods_denied);
                let reports = self.mods.enabled_package_reports(&self.packages);
                let pending = Connection::begin_connect(&job.host, job.port, &job.name, &job.password, &reports);
                self.screen = Screen::Connecting(ConnectJob {
                    pending,
                    hosted: job.hosted,
                    retried: true,
                    host: job.host,
                    port: job.port,
                    name: job.name,
                    password: job.password,
                    notice: Some(notice),
                });
            }
            Err(err) => {
                if job.retried {
                    self.mods.release_server();
                }
                let text = if job.hosted {
                    format!("hosted, but could not connect: {err}")
                } else {
                    format!("could not join: {err}")
                };
                self.fail_to_menu(text);
            }
        }
    }

    fn cancel_connect(&mut self) {
        let hosted = match &self.screen {
            Screen::Connecting(job) => {
                job.pending.cancel();
                job.hosted
            }
            _ => false,
        };
        if hosted && let Some(host) = self.host.take() {
            host.stop();
        }
        self.return_to_menu(None);
    }

    fn standby_menu(&self) -> MenuStack {
        Self::start_stack(&self.mods, &self.saves, &self.session, None, self.host.is_some())
    }

    /// Esc on a waiting screen (connecting, loading).
    fn cancel_pressed(&mut self, eng: &mut Engine) -> bool {
        let dt = eng.frame_time();
        self.router.set_context(Context::Menu);
        match self.router.frame(eng, dt).view() {
            View::Menu(menu) => crate::menu::gather(&menu).iter().any(|intent| matches!(intent, crate::menu::Intent::Cancel)),
            _ => false,
        }
    }

    /// Report a connection/host failure and return to the menu.
    fn fail_to_menu(&mut self, message: String) {
        self.return_to_menu(Some(message));
    }

    /// Start a fresh world with a time-seeded generator.
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
        let recipe = Recipe {
            seed,
            render: self.mods.effective_render(&self.settings),
            kind: self.mods.worldgen_kind(),
            cfg: terrain_cfg_from_mods(&self.mods),
        };
        self.begin_loading(eng, Loading::new_world(recipe));
    }

    /// Start loading a save. A save that fails to load returns to the menu.
    fn load_world(&mut self, eng: &mut Engine, id: &SlotId) {
        // The save header names the generator; the InfiniteDiffusion mod's
        // enabled flag only chooses the next *new* world.
        let render = self.mods.effective_render(&self.settings);
        self.begin_loading(eng, Loading::load(id.clone(), render));
    }

    /// Benchmarks wait for the world here, so they enter on this frame as they always did.
    fn begin_loading(&mut self, eng: &mut Engine, loading: Loading) {
        if self.bench.is_some() {
            self.arrive(eng, loading);
        } else {
            self.loading = Some(loading);
        }
    }

    /// Poll the world job. Esc drops it: nothing was entered or saved. A join being cancelled
    /// also closes the connection and stops a host we started, as cancelling the connect does.
    fn update_loading(&mut self, eng: &mut Engine) {
        if self.cancel_pressed(eng) {
            if let Some(Loading::Join { hosted, .. }) = self.loading.take() {
                if hosted && let Some(host) = self.host.take() {
                    host.stop();
                }
                self.return_to_menu(None);
            }
            return;
        }
        if self.loading.as_ref().is_some_and(Loading::done) {
            let loading = self.loading.take().expect("a landed world");
            self.arrive(eng, loading);
        }
    }

    /// Swap a built world in: fresh mod state, the save's mod state for a load, then the game.
    fn arrive(&mut self, eng: &mut Engine, loading: Loading) {
        // Every world starts from a clean default mod set (empty inventory, etc.);
        // the mod menu's enable/disable choices persist.
        self.mods.reset_state();
        match loading {
            Loading::New(job) => {
                let (world, player, id) = job.wait();
                let now = save::unix_now();
                self.active = Some(ActiveSlot::new(
                    id.clone(),
                    SaveMeta {
                        name: id.as_str().to_string(),
                        seed: world.seed(),
                        created: now,
                        last_played: now,
                        playtime_secs: 0,
                        edit_count: 0,
                    },
                ));
                self.enter_game(eng, Game::new(world, player, id.as_str().to_string()));
            }
            Loading::Load(id, job) => match job.wait() {
                Ok((restored, report)) => {
                    let (world, player, meta) = restored.finish(&mut self.mods);
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
            },
            Loading::Join { job, conn, notice, .. } => {
                let (world, player) = job.wait();
                // A networked world is a live mirror, not a save.
                self.active = None;
                let mut game = Game::new(world, player, "multiplayer".to_string()).with_net(conn);
                if let Some(notice) = notice {
                    game.notify(notice);
                }
                self.enter_game(eng, game);
            }
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
        self.audio.enter_world();
        self.fan_world_edge(GameEvent::EnterWorld);
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
            &self.cues,
        );
        if let Signal::ExitToMenu = signal {
            self.flush_save();
            let notice = if let Screen::Playing(game) = &mut self.screen {
                let notice = game.take_leave_notice();
                // Return the world's GPU meshes to the engine before dropping it.
                game.free_gpu(eng);
                notice
            } else {
                None
            };
            eng.enable_cursor();
            self.return_to_menu(notice); // drops the Box<Game>
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
        let waiting = match &self.screen {
            Screen::Connecting(_) => Some("Connecting…"),
            _ if self.loading.is_some() => Some("Loading…"),
            _ => None,
        };
        if let Some(title) = waiting {
            let mut f = eng.begin_frame(MENU_CLEAR.to_linear());
            let theme = ui::Theme::new();
            let screen = (w as i32, h as i32);
            ui::label(
                &mut f,
                &theme,
                screen,
                Anchor::Center,
                (0, -16),
                28,
                ui::Role::Primary.color(),
                title,
            );
            ui::label(
                &mut f,
                &theme,
                screen,
                Anchor::Center,
                (0, 24),
                20,
                ui::Role::Muted.color(),
                "Cancel",
            );
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


/// Join `host:port` reporting no mods. Test clients use this. The game's own
/// join is [`Connection::begin_connect`], off the render thread.
#[cfg(test)]
pub(crate) fn join_server(host: &str, port: u16, name: &str, password: &str) -> Result<Connection, ConnectError> {
    Connection::connect(host, port, name, password)
}

/// The most recent readable slot, else a fresh id. `saves` is already ordered
/// most-recently played first.
fn host_save_path(saves: &[crate::save::Slot]) -> PathBuf {
    if let Some(slot) = saves.iter().find(|s| s.meta.is_ok()) {
        return save::file_path(&slot.id);
    }
    save::file_path(&save::fresh_id())
}

/// Join, reporting the packages this client has enabled. The mod list is what
/// an honest client says; a modified client can lie. One refusal turns those
/// packages off for this session and retries once. A second failure restores
/// them and returns the error. Nothing here writes `mods.cfg`. The menu uses
/// [`App::update_connecting`]; this stays for the session-hold test.
#[cfg(test)]
fn connect_session(
    mods: &mut Mods,
    packages: &[ModDescriptor],
    host: &str,
    port: u16,
    name: &str,
    password: &str,
) -> Result<(Connection, Option<String>), ConnectError> {
    let reports = mods.enabled_package_reports(packages);
    match Connection::begin_connect(host, port, name, password, &reports).wait() {
        Ok(conn) => Ok((conn, None)),
        Err(err) if err.mods_denied.is_empty() => Err(err),
        Err(err) => {
            let notice = mod_hold_notice(packages, &err.mods_denied);
            mods.hold_packages(&err.mods_denied);
            let reports = mods.enabled_package_reports(packages);
            match Connection::begin_connect(host, port, name, password, &reports).wait() {
                Ok(conn) => Ok((conn, Some(notice))),
                Err(again) => {
                    mods.release_server();
                    Err(again)
                }
            }
        }
    }
}

/// "This server does not allow: Developer Toolkit; it is off while you are connected".
/// Several names use "they are". Display names come from the build; an unknown
/// id is shown as itself.
fn mod_hold_notice(packages: &[ModDescriptor], ids: &[String]) -> String {
    let names: Vec<&str> = ids
        .iter()
        .map(|id| packages.iter().find(|pkg| pkg.id == id).map(|pkg| pkg.name).unwrap_or(id.as_str()))
        .collect();
    let list = names.join("; ");
    if names.len() == 1 {
        format!("This server does not allow: {list}; it is off while you are connected")
    } else {
        format!("This server does not allow: {list}; they are off while you are connected")
    }
}

/// Parse the winning worldgen payload as generator knobs.
fn terrain_cfg_from_mods(mods: &Mods) -> TerrainCfg {
    mods.worldgen_config().as_deref().map(TerrainCfg::from_text).unwrap_or_default()
}

/// A world seed from the wall clock, so each new world differs.
fn fresh_seed() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(1)
}

/// Spawn the player just above level ground near the world origin, so they land on a meadow or a
/// valley floor rather than a cliff edge. Spirals outward over whole 16×16 chunk columns
/// ([`World::heights_16`], one batch each) for the first cell whose 3×3 neighbourhood is flat.
fn spawn_player(world: &World) -> Player {
    if let Some(p) = world.chart_spawn() {
        let mut player = Player::new(p);
        player.stand_in(world.gravity_at(player.position).accel);
        return player;
    }
    let mut seen = [(i32::MAX, i32::MAX); 32];
    let mut n = 0usize;
    for r in 0i32..8 {
        for (dx, dz) in [(r, 0), (0, r), (-r, 0), (0, -r), (r, r), (-r, -r), (r, -r), (-r, r)] {
            let (cx, cz) = ((dx * 8).div_euclid(16), (dz * 8).div_euclid(16));
            if seen[..n].contains(&(cx, cz)) {
                continue;
            }
            seen[n] = (cx, cz);
            n += 1;
            let heights = world.heights_16(cx, cz);
            for lz in 1..15 {
                for lx in 1..15 {
                    let h = heights[lx + lz * 16];
                    let flat = (0..9).all(|k| {
                        let (ox, oz) = (lx + k % 3 - 1, lz + k / 3 - 1);
                        (heights[ox + oz * 16] - h).abs() <= 1
                    });
                    if flat {
                        let (x, z) = (cx * 16 + lx as i32, cz * 16 + lz as i32);
                        let mut player = Player::new(DVec3::new(x as f64 + 0.5, h as f64 + 3.0, z as f64 + 0.5));
                        player.stand_in(world.gravity_at(player.position).accel);
                        return player;
                    }
                }
            }
        }
    }
    let h = world.surface_y(0, 0);
    let mut player = Player::new(DVec3::new(0.5, h as f64 + 3.0, 0.5));
    player.stand_in(world.gravity_at(player.position).accel);
    player
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_config::RenderConfig;
    use crate::world::generation::WorldgenKind;

    #[test]
    fn vanilla_client_joins_a_diffusion_server_through_join_server() {
        assert_eq!(Mods::empty().worldgen_kind(), WorldgenKind::Flat);
        let terrain = TerrainCfg { relief: 150, caves: 25, ..TerrainCfg::default() }.clamp();
        let handle = server::spawn(
            0,
            Config {
                seed: 99,
                worldgen: WorldgenKind::Diffusion,
                terrain,
                ..Config::default()
            },
        )
        .unwrap();
        let conn = join_server("127.0.0.1", handle.addr().port(), "ada", "").expect("join");
        assert_eq!(conn.worldgen(), WorldgenKind::Diffusion);
        assert_eq!(conn.terrain(), terrain);
        assert_eq!(conn.seed(), 99);
        handle.stop();
    }

    struct Named(&'static str, &'static str);

    impl crate::modding::Mod for Named {
        fn id(&self) -> &'static str {
            self.0
        }
        fn name(&self) -> &str {
            self.1
        }
    }

    fn register_toolkit(reg: &mut crate::modding::ModRegistrar) {
        reg.add(Named("dev-toolkit", "Developer Toolkit"));
    }

    fn register_hotbar(reg: &mut crate::modding::ModRegistrar) {
        reg.add(Named("hotbar", "Hotbar"));
    }

    fn sample_packages() -> [ModDescriptor; 2] {
        [
            ModDescriptor {
                id: "pwc.dev-toolkit",
                name: "Developer Toolkit",
                version: "1.0.0",
                register: register_toolkit,
            },
            ModDescriptor {
                id: "pwc.hotbar",
                name: "Hotbar",
                version: "0.1.0",
                register: register_hotbar,
            },
        ]
    }

    #[test]
    fn mod_hold_notice_names_one_and_several() {
        let packages = sample_packages();
        assert_eq!(
            mod_hold_notice(&packages, &["pwc.dev-toolkit".into()]),
            "This server does not allow: Developer Toolkit; it is off while you are connected"
        );
        assert_eq!(
            mod_hold_notice(&packages, &["pwc.dev-toolkit".into(), "pwc.hotbar".into()]),
            "This server does not allow: Developer Toolkit; Hotbar; they are off while you are connected"
        );
    }

    #[test]
    fn host_save_path_is_the_newest_readable_slot() {
        let slots = vec![Slot::for_test("newer", 3, 1), Slot::for_test("older", 1, 0)];
        assert_eq!(host_save_path(&slots), save::file_path(&slots[0].id));
        assert_eq!(
            host_save_path(&[]).extension().and_then(|ext| ext.to_str()),
            Some("save")
        );
    }

    /// The server refuses the toolkit, the client turns it off and joins once,
    /// and the Mods menu will not turn it back on while that hold lasts.
    #[test]
    fn denied_mod_is_disabled_for_the_session_and_the_menu_cannot_reenable_it() {
        use crate::menu::menus::ModsMenu;
        use crate::menu::{Command, Menu, Msg, ValueView};
        let packages = sample_packages();
        let mut mods = GameBuild::new().with_mod(packages[0]).with_mod(packages[1]).mods();
        let handle = server::spawn(
            0,
            Config {
                seed: 1,
                worldgen: WorldgenKind::Flat,
                mods_deny: vec!["pwc.dev-toolkit".into()],
                ..Config::default()
            },
        )
        .unwrap();
        let (conn, notice) = connect_session(&mut mods, &packages, "127.0.0.1", handle.addr().port(), "ada", "")
            .expect("retry joins");
        assert!(conn.is_alive());
        assert_eq!(
            notice.as_deref(),
            Some("This server does not allow: Developer Toolkit; it is off while you are connected")
        );
        let index = (0..mods.len()).find(|&i| mods.id(i) == "dev-toolkit").expect("toolkit");
        assert!(!mods.is_enabled(index));
        assert!(mods.server_off(index));
        assert!(!mods.toggle(index), "the menu's toggle is refused while connected");
        let hotbar = (0..mods.len()).find(|&i| mods.id(i) == "hotbar").expect("hotbar");
        assert!(mods.is_enabled(hotbar));
        let snap = ModRow::snapshot(&mods);
        let mut settings = Settings::default();
        let session = Session::default();
        let mut ctx = crate::menu::Ctx {
            settings: &mut settings,
            saves: &[],
            mods: &snap,
            session: &session,
            mods_save_error: None,
        };
        let view = ModsMenu.view(&ctx);
        let row = view.rows.iter().find(|row| row.label.contains("Developer Toolkit")).expect("row");
        match &row.kind {
            crate::menu::RowKind::Value(ValueView::Choice(value)) => assert_eq!(value, "off (server)"),
            _ => panic!("expected off (server)"),
        }
        assert!(matches!(
            ModsMenu.update(Msg::Pick(crate::menu::menus::ModsAction::ServerOff), &mut ctx),
            Command::Stay
        ));
        drop(conn);
        mods.release_server();
        assert!(mods.is_enabled(index));
        assert!(!mods.server_off(index));
        handle.stop();
    }

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
        let world = World::with_kind_cfg(1, RenderConfig::default(), WorldgenKind::Diffusion, TerrainCfg::default(), false);
        let _ = black_box(spawn_player(&world));
        let t = Instant::now();
        let _ = black_box(spawn_player(&world));
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        println!("spawn_player {ms:.2}ms");
        assert!(ms < 250.0, "spawn must stay under a frame, got {ms:.2}");
    }

    /// Enter one world the way the app does, timing what starting the job costs the clicking
    /// frame, how long the job runs, and what the switching frame still does on the render thread
    /// (all of `arrive` but the engine calls of `enter_game`).
    fn probe_entry(label: &str, mods: &mut Mods, start: impl FnOnce() -> Loading) {
        let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1000.0;
        let t0 = Instant::now();
        let loading = start();
        let t1 = Instant::now();
        while !loading.done() {
            std::thread::sleep(Duration::from_micros(100));
        }
        let t2 = Instant::now();
        mods.reset_state();
        let mut game = match loading {
            Loading::New(job) => {
                let (world, player, id) = job.wait();
                Game::new(world, player, id.as_str().to_string())
            }
            Loading::Load(id, job) => {
                let (world, player, _) = job.wait().unwrap().0.finish(mods);
                Game::new(world, player, id.as_str().to_string())
            }
            Loading::Join { job, conn, .. } => {
                let (world, player) = job.wait();
                Game::new(world, player, "multiplayer".into()).with_net(conn)
            }
        };
        let t3 = Instant::now();
        let pos = game.player().position;
        game.world_mut().prepare_around(pos);
        let t4 = Instant::now();
        println!(
            "{label}: start {:.2}ms job {:.1}ms | switch frame {:.2}ms (land {:.2}ms prepare {:.2}ms)",
            ms(t0, t1), ms(t1, t2), ms(t2, t4), ms(t2, t3), ms(t3, t4)
        );
        drop(std::hint::black_box(game));
    }

    /// Probe for the world-entry stall: each entry path (new, small and ~100k-edit saves, a local
    /// server, empty and serving the 100k-edit world) through the job, cold (first world in the
    /// process) and warm. The 100k join also shows its overlay's worst frame, against applying
    /// each cell as it arrives.
    #[test]
    #[ignore]
    fn entry_cost_probe() {
        use std::hint::black_box;
        let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1000.0;
        let a = Instant::now();
        let _ = black_box(crate::block::BlockRegistry::with_builtins());
        let b = Instant::now();
        let _ = black_box(crate::world::terrain::palette::current());
        let c = Instant::now();
        let mut reg = crate::block::BlockRegistry::with_builtins();
        let g = black_box(crate::world::terrain::generator(&mut reg, 1234, TerrainCfg::default()));
        let d = Instant::now();
        let _ = black_box(crate::gravity::Field::new(g.mass()));
        let e = Instant::now();
        println!("parts: registry {:.1}ms palette {:.1}ms generator {:.1}ms gravity {:.1}ms", ms(a, b), ms(b, c), ms(c, d), ms(d, e));
        let render = RenderConfig::default();
        let recipe = Recipe { seed: 1234, render, kind: WorldgenKind::Diffusion, cfg: TerrainCfg::default() };
        let mut mods = crate::modding::testing::standard();
        for label in ["new cold", "new warm"] {
            probe_entry(label, &mut mods, || Loading::new_world(recipe));
        }
        for (label, edits) in [("load small", 100), ("load 100k", 100_000)] {
            let id = SlotId::new(&format!("__probe_{edits}__")).unwrap();
            let mut world = World::with_kind_cfg(1234, render, WorldgenKind::Diffusion, TerrainCfg::default(), false);
            let player = spawn_player(&world);
            let soil = world.registry().id_by_label("soil").unwrap();
            let spawn = world.chart_spawn().expect("charted start");
            let cell = world.terrain().atlases().iter().find_map(|a| a.storage_of(spawn)).expect("spawn storage");
            let side = (edits as f64).cbrt().ceil() as i32;
            let [x0, y0, z0] = cell.map(|v| v as i32 - side / 2);
            for i in 0..edits as i32 {
                let (x, y, z) = (x0 + i % side, y0 + i / (side * side), z0 + (i / side) % side);
                let id = if world.terrain().voxel_at(x, y, z) == crate::block::AIR { soil } else { crate::block::AIR };
                world.set_block(x, y, z, id);
            }
            assert_eq!(world.edits().count(), edits);
            let meta = SaveMeta { name: label.into(), seed: 0, created: 0, last_played: 0, playtime_secs: 0, edit_count: 0 };
            save::save(&id, &world, &player, &mods, meta).unwrap();
            drop(world);
            let t = Instant::now();
            let _ = black_box(save::store::read(&id).unwrap());
            println!("{label}: read and decode alone {:.1}ms", ms(t, Instant::now()));
            probe_entry(label, &mut mods, || Loading::load(id.clone(), render));
            if edits == 100_000 {
                probe_join_overlay(&save::file_path(&id), render, &mut mods);
            }
            for path in [save::file_path(&id), save::file_path(&id).with_extension("save.bak")] {
                let _ = std::fs::remove_file(path);
            }
        }
        let handle = server::spawn(0, Config { seed: 1234, worldgen: WorldgenKind::Diffusion, ..Config::default() }).unwrap();
        let conn = join_server("127.0.0.1", handle.addr().port(), "probe", "").expect("join");
        probe_entry("join", &mut mods, || Loading::join(conn, render, None, false));
        handle.stop();
    }

    /// Join a server serving the world file at `path` twice: applying each overlay cell as it
    /// arrives (the old path), then through the game's budgeted bulk install. Polls ~8 ms apart.
    fn probe_join_overlay(path: &std::path::Path, render: RenderConfig, mods: &mut Mods) {
        use crate::net::client::Incoming;
        let ms = |t: Instant| t.elapsed().as_secs_f64() * 1000.0;
        let world = Some(path.to_path_buf());
        let config = Config { world, autosave_every: Duration::from_secs(3600), ..Config::default() };
        let handle = server::spawn(0, config).unwrap();
        let port = handle.addr().port();

        let mut conn = join_server("127.0.0.1", port, "each", "").expect("join");
        let mut world = World::with_kind_cfg(conn.seed(), render, conn.worldgen(), conn.terrain(), false);
        let (start, mut frames, mut worst, mut cells) = (Instant::now(), 0, 0.0f64, 0);
        while !conn.snapshot_ready() {
            let t = Instant::now();
            for event in conn.poll() {
                if let Incoming::Mutation { x, y, z, spec } = event {
                    let id = save::parse_block(world.registry_mut(), &spec);
                    world.set_block(x, y, z, id);
                    cells += 1;
                }
            }
            worst = worst.max(ms(t));
            frames += 1;
            std::thread::sleep(Duration::from_millis(8));
        }
        let done = ms(start);
        println!("join 100k overlay, each cell: {cells} cells, {frames} frames, worst frame {worst:.1}ms, done after {done:.0}ms");

        let conn = join_server("127.0.0.1", port, "bulk", "").expect("join");
        let Loading::Join { job, conn, .. } = Loading::join(conn, render, None, false) else { unreachable!() };
        let (world, player) = job.wait();
        mods.reset_state();
        let mut game = Game::new(world, player, "multiplayer".into()).with_net(conn);
        let (start, mut frames, mut worst) = (Instant::now(), 0, 0.0f64);
        loop {
            let t = Instant::now();
            let settled = game.pump_net(mods);
            worst = worst.max(ms(t));
            frames += 1;
            if settled {
                break;
            }
            std::thread::sleep(Duration::from_millis(8));
        }
        let cells = game.world().edits().count();
        let done = ms(start);
        println!("join 100k overlay, bulk: {cells} cells, {frames} frames, worst frame {worst:.1}ms, done after {done:.0}ms");
        handle.stop();
    }

    #[test]
    fn spawn_player_stands_in_local_gravity() {
        let world = World::with_config_lazy(7, RenderConfig::default());
        let p = spawn_player(&world);
        let min = 0.02 * crate::player::STANDARD_GRAVITY;
        match world.gravity_at(p.position).up(min) {
            Some(up) => {
                assert!((p.up() - up).length() < 1e-9, "{} vs {up}", p.up());
                assert_eq!(p.up_axis, crate::coord::Face::from_dominant(up));
            }
            None => {
                assert_eq!(p.orientation.frame, glam::DQuat::IDENTITY);
                assert_eq!(p.up_axis, crate::coord::Face::PosY);
            }
        }
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
