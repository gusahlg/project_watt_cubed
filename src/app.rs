//! app.rs owns the top-level state machine: the start menu, an in-world
//! [`Game`], the mod menu, the host/join forms, and the graphics settings
//! screen. It routes each engine frame to the active screen, creates and
//! loads worlds (off the render thread, in `entry`), and autosaves when
//! leaving one.
//!
//! The window itself belongs to the engine: [`App::run`] hands a per-frame
//! closure to [`voxel_engine::run`], which is the moral equivalent of the old
//! raylib `while !window_should_close()` loop.
mod bench;
mod connect;
mod entry;
mod host;

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use voxel_engine::{Color, DVec3, Engine};

use crate::audio::{AudioService, AudioView, CueSymbols, GameEvent, ModLink, PeerAudio, SoundConfig, SoundSystem};
use crate::benchmark::Benchmark;
use crate::game::{Game, Signal};
use crate::input::router::{Context, Router, View};
use crate::menu::menus::{ModsMenu, SettingsHub};
use crate::menu::start::{StartFacts, StartRoot, VERSION};
use crate::menu::theme::{DefaultTheme, MenuTheme};
use crate::menu::{AppEffect, Ctx, Framed, MenuStack};
use crate::modding::{ActionSet, BuildInfo, Debounce, GameBuild, Mods, VisualMask};
use crate::ui::{self, Anchor};
use crate::player::Player;
use crate::save::{self, Autosaver, SaveMeta, Slot, SlotId, Tick};
use crate::session::Session;
use crate::settings::{GfxEngine, Settings};
use crate::world::generation::WorldgenKind;
use crate::world::terrain::TerrainCfg;
use crate::world::World;
use connect::ConnectJob;
use entry::{Loading, Recipe};
use host::Host;

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
    /// Installed mods and what the core suspended; shared with the game while playing.
    mods: Mods,
    /// Packages compiled into this executable, of every kind. `Hello` reports the unsuspended mods.
    build: BuildInfo,
    screen: Screen,
    /// A world building off the render thread. The menu under it stays as it was; Esc drops it.
    loading: Option<Loading>,
    /// Said in the next world (or on the menu, if it fails or is cancelled): the host stopped for it.
    entry_notice: Option<&'static str>,
    /// The integrated server when hosting, kept alive across menu returns so friends
    /// stay connected. Opening a world, or deleting the hosted one, stops it first.
    host: Host,
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
    /// Settings changes (menu steps, the HUD hotkey, the console) wait here and save once they
    /// go quiet, or on leaving a world and on quit.
    settings_flush: Debounce,
    clock: Instant,
    /// `WATT_BENCH_WORLDGEN`: the generator new worlds use, whatever the mods say.
    worldgen_pin: Option<WorldgenKind>,
    /// The options revision the mods last heard about ([`Mod::on_options`](crate::modding::Mod::on_options)).
    options_seen: u64,
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

impl GfxKey {
    /// The stamp for `settings` with `mask`'s visual groups stripped, in a `w`×`h` window.
    /// Built every frame, so it allocates nothing.
    fn of(settings: &Settings, mask: VisualMask, (w, h): (u32, u32)) -> Self {
        let (msaa, scale) = settings.session_msaa_scale(w, h);
        Self {
            w,
            h,
            fullscreen: settings.fullscreen,
            msaa,
            scale_bits: scale.to_bits(),
            cull_faces: settings.cull_faces,
            flags: mask.effective_render(settings).engine_flags(),
        }
    }
}

/// The one writer of the engine's graphics state: window mode, MSAA, render scale, face culling
/// and the render-lane flags, masked by the visual mods. Pushes only when the stamp moves, so a
/// quiet frame or an idle menu does not wake the render thread; a resize moves it, so a fallback
/// cannot be overwritten. One push writes the flags once, so a lane a mod strips never turns on
/// in between (each turn resets the engine's temporal state). True when it pushed.
fn push_gfx(eng: &mut impl GfxEngine, settings: &mut Settings, mask: VisualMask, applied: &mut Option<GfxKey>) -> bool {
    let extent = eng.window_extent();
    if *applied == Some(GfxKey::of(settings, mask, extent)) {
        return false;
    }
    // Applied MSAA/scale from engine create (and later recreates) before
    // we push the session request, so a fallback cannot be overwritten.
    settings.sync_engine_applied(eng);
    settings.apply(eng);
    // Stamped after the apply: it noted the render extent (Auto VRS and TAA read it) and any
    // fallback, so the next frame's stamp matches and nothing is pushed twice.
    let key = GfxKey::of(settings, mask, extent);
    eng.set_flags(key.flags);
    *applied = Some(key);
    #[cfg(test)]
    crate::alloc_count::note_engine(crate::alloc_count::EngineCall::SettingsApply);
    true
}

/// Settings a frame changed are written once [`Debounce::IDLE_MS`] pass with no further change,
/// so a held Left/Right on a settings row does not rewrite `settings.cfg` at key-repeat rate.
/// True when the write is due now.
fn settings_write_due(flush: &mut Debounce, changed: bool, now_ms: u64) -> bool {
    if changed {
        flush.mark(now_ms);
    }
    flush.poll(now_ms)
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
            Some(env) => eprintln!("PWC: {} packages, environment {env}", build.packages().len()),
            None if build.packages().is_empty() => eprintln!("PWC: vanilla build (no packages)"),
            None => eprintln!("PWC: {} packages", build.packages().len()),
        }
        // While the menu is up, so the first world's frame does not pay for it.
        crate::world::terrain::prewarm();
        let mut mods = Mods::from_build(build);
        let pins = Benchmark::mod_pins_from_env();
        let mut pinned = pins.suspend;
        if pins.visuals_core == Some(true) {
            for id in mods.visual_packages() {
                if !pinned.contains(&id) {
                    pinned.push(id);
                }
            }
        }
        if !pinned.is_empty() {
            eprintln!("PWC: suspended for this run: {}", pinned.join(", "));
            mods.pin_suspended(&pinned);
        }
        let saves = save::list();
        let mut settings = Settings::load(mods.options_mut());
        mods.options_changed();
        let options_seen = mods.options().revision();
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
            build: build.info().clone(),
            screen,
            loading: None,
            entry_notice: None,
            host: Host::default(),
            settings,
            session,
            bench,
            sound,
            cues,
            audio,
            last_stall_log: None,
            settings_flush: Debounce::new(),
            clock: Instant::now(),
            worldgen_pin: pins.worldgen,
            options_seen,
            gfx_applied: None,
        }
    }

    fn now_ms(&self) -> u64 {
        self.clock.elapsed().as_millis() as u64
    }

    /// Tell the mods once the options moved since they last heard.
    fn tell_options(&mut self) {
        let revision = self.mods.options().revision();
        if revision != self.options_seen {
            self.options_seen = revision;
            self.mods.options_changed();
        }
    }

    fn flush_settings_if_dirty(&mut self) {
        if self.settings_flush.take() && self.bench.is_none() {
            self.settings.save(self.mods.options());
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
                self.settings.save(self.mods.options());
            }
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
                self.settings.save(self.mods.options());
            }
            self.flush_save();
            self.note_frame_stall(t0, update_dt);
            return false;
        }
        // VRAM guard + live settings, then the game's half of a console change, which reads
        // the render extent the push noted.
        self.push_gfx(eng);
        if let Screen::Playing(game) = &mut self.screen
            && game.take_settings_changed()
        {
            game.apply_settings(eng, &self.settings);
        }
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
        let head = format!("frame stall {}ms update={update_ms:.1}ms draw={draw_ms:.1}ms", dt.as_millis());
        let screen = match &self.screen {
            Screen::Playing(game) => {
                eprintln!("{head}\n  {}\n  {}", game.world().entry_debug(), game.phase_debug());
                return;
            }
            Screen::Menus(_) if self.loading.is_some() => "loading",
            Screen::Menus(_) => "menus",
            Screen::Connecting(_) => "connecting",
        };
        eprintln!("{head} ({screen})");
    }

    fn push_gfx(&mut self, eng: &mut Engine) {
        push_gfx(eng, &mut self.settings, self.mods.visual_mask(), &mut self.gfx_applied);
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
        self.fan_audio(dt, false, click);
        let before = self.mods.options().revision();
        let now_ms = self.now_ms();
        let mut effect = None;
        if let Screen::Menus(stack) = &mut self.screen {
            let (suspended, visuals, options) = self.mods.menu_parts();
            let mut ctx = Ctx {
                settings: &mut self.settings,
                options,
                saves: &self.saves,
                session: &self.session,
                build: &self.build,
                suspended,
                visuals,
            };
            effect = stack.update(&intents, &mut ctx);
        }
        // Every write goes through the options view, which moves the revision. Persist once the
        // steps go quiet.
        let changed = self.mods.options().revision() != before;
        self.tell_options();
        if changed {
            self.sound.set_mix(self.settings.mix_change());
        }
        if settings_write_due(&mut self.settings_flush, changed, now_ms) {
            self.settings.save(self.mods.options());
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
            AppEffect::Load(id) => self.load_world(eng, &id),
            AppEffect::DeleteWorld(id) => {
                // Stopping saves, so the trashed copy keeps the friends' last edits.
                let stopped = if self.host.serves(&id) { self.host.stop() } else { None };
                if let Err(e) = save::delete(&id) {
                    eprintln!("could not delete world {id}: {e}");
                }
                self.saves = save::list();
                if stopped.is_some() {
                    self.screen = Screen::Menus(Self::start_stack(&self.mods, &self.saves, &self.session, stopped, false));
                }
            }
            AppEffect::Host(info) => {
                self.session.port = info.port.to_string();
                self.session.name = info.name.clone();
                self.session.save();
                self.start_host(info);
            }
            AppEffect::Join(info) => {
                self.session.address = info.host.clone();
                self.session.port = info.port.to_string();
                self.session.name = info.name.clone();
                self.session.save();
                self.start_join(info);
            }
            AppEffect::Settings => {
                if let Screen::Menus(stack) = &mut self.screen {
                    stack.push(Framed::boxed(SettingsHub));
                }
            }
            AppEffect::Mods => {
                if let Screen::Menus(stack) = &mut self.screen {
                    stack.push(Framed::boxed(ModsMenu));
                }
            }
            AppEffect::Quit => return true,
        }
        false
    }

    /// Open the start screen: first active start-screen mod, else the core fallback.
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

    /// Run the mods' audio hooks with no world under them: each menu frame (`dt`, and maybe a UI
    /// click), so a mod can play a UI cue, and the edges into and out of a world, before the
    /// service drops its sessions.
    fn fan_audio(&mut self, dt: f32, in_world: bool, event: Option<GameEvent>) {
        const EMPTY: &[PeerAudio] = &[];
        const NO_IDS: &[&str] = &[];
        let view = AudioView {
            dt,
            pos: DVec3::ZERO,
            peers: EMPTY,
            in_world,
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

    /// Tell the mods the world is changing.
    fn fan_world_edge(&mut self, event: GameEvent) {
        let in_world = matches!(event, GameEvent::EnterWorld);
        self.fan_audio(0.0, in_world, Some(event));
    }

    /// Return to the start menu with an optional notice (e.g. a failed connect).
    fn return_to_menu(&mut self, notice: Option<String>) {
        // Packages a server suspended for the session come back on leave. Nothing was saved.
        self.mods.resume_packages();
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
            self.host.running(),
        ));
    }

    fn standby_menu(&self) -> MenuStack {
        Self::start_stack(&self.mods, &self.saves, &self.session, None, self.host.running())
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
        let stopped = self.host.stop();
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
            kind: self.worldgen_pin.unwrap_or_else(|| self.mods.worldgen_kind()),
            cfg: terrain_cfg_from_mods(&self.mods),
        };
        self.entry_notice = stopped;
        self.begin_loading(eng, Loading::new_world(recipe));
    }

    /// Start loading a save. A save that fails to load returns to the menu.
    fn load_world(&mut self, eng: &mut Engine, id: &SlotId) {
        // Stopping saves first, so the world loads with the friends' last edits. The save header
        // names the generator; the installed worldgen mod only chooses the next *new* world.
        self.entry_notice = self.host.stop();
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
            match self.loading.take() {
                Some(Loading::Join { hosted, .. }) => {
                    if hosted {
                        self.host.stop();
                    }
                    self.return_to_menu(None);
                }
                // The menu under a new world or a load was never replaced. Only a stopped host
                // changes it (its status line), and the notice says why.
                Some(_) => {
                    if let Some(notice) = self.entry_notice.take() {
                        self.return_to_menu(Some(notice.to_string()));
                    }
                }
                None => {}
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
        // Every world starts from a clean default mod state (empty inventory, etc.).
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
                Err(e) => {
                    let message = match self.entry_notice.take() {
                        Some(notice) => format!("{notice}; could not load {id}: {e}"),
                        None => format!("could not load {id}: {e}"),
                    };
                    self.fail_to_menu(message);
                }
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
        if let Some(notice) = self.entry_notice.take() {
            game.notify(notice);
        }
        // World-construction lanes apply on entry only, before streaming spins;
        // everything live-applicable goes through the same path `/gfx` uses.
        let render = self.mods.effective_render(&self.settings);
        game.set_visual_mask(self.mods.visual_mask());
        game.world_mut()
            .set_render_lanes(render.occlusion, render.lod2);
        // Entry re-fits MSAA and scale to the VRAM free now, as it always has; the game then
        // resolves its lanes against the extent that push noted.
        self.gfx_applied = None;
        self.push_gfx(eng);
        game.apply_settings(eng, &self.settings);
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
        let now_ms = self.now_ms();
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
        if settings_write_due(&mut self.settings_flush, game.take_settings_dirty(), now_ms) && self.bench.is_none() {
            self.settings.save(self.mods.options());
        }
        if let Signal::ExitToMenu = signal {
            self.flush_settings_if_dirty();
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
        // The draw only reads; the options step aside for the theme's borrow (a move, no copy).
        let mut options = std::mem::take(self.mods.options_mut());
        let fallback = DefaultTheme;
        let theme: &dyn MenuTheme = self.mods.menu_theme().unwrap_or(&fallback);
        let mut f = eng.begin_frame(MENU_CLEAR.to_linear());
        if let Screen::Menus(stack) = &self.screen {
            let ctx = Ctx {
                settings: &mut self.settings,
                options: &mut options,
                saves: &self.saves,
                session: &self.session,
                build: &self.build,
                suspended: self.mods.suspended(),
                visuals: self.mods.visual_mask(),
            };
            stack.draw(&ctx, theme, &mut f, w, h);
        }
        drop(f);
        *self.mods.options_mut() = options;
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

/// The player on a new world, standing in the local pull at [`spawn_column`](crate::world::generation::spawn_column),
/// the spawn the server gives the same seed.
fn spawn_player(world: &World) -> Player {
    let pos = crate::world::generation::spawn_column(world.terrain());
    Player::standing(pos, world.gravity_at(pos).accel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::client::Connection;
    use crate::net::server::{self, Config};
    use crate::render_config::RenderConfig;
    use crate::world::generation::WorldgenKind;

    #[test]
    fn vanilla_client_joins_a_diffusion_server() {
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
        let conn = Connection::connect("127.0.0.1", handle.addr().port(), "ada", "").expect("join");
        assert_eq!(conn.worldgen(), WorldgenKind::Diffusion);
        assert_eq!(conn.terrain(), terrain);
        assert_eq!(conn.seed(), 99);
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

    /// The server builds its generator from the seed, kind and knobs alone (`server::spawn`), and
    /// single player builds a world; both stand a fresh player on the same point.
    #[test]
    fn single_player_and_the_server_spawn_on_the_same_point() {
        use crate::world::generation::{FlatTerrain, spawn_column};
        let steep = TerrainCfg { relief: 200, space: 200, ..TerrainCfg::default() }.clamp();
        let worlds = [
            (WorldgenKind::Flat, TerrainCfg::default()),
            (WorldgenKind::Diffusion, TerrainCfg::default()),
            (WorldgenKind::Diffusion, steep),
        ];
        for seed in [1, 42, 7] {
            for (kind, cfg) in worlds {
                let mut registry = crate::block::BlockRegistry::with_builtins();
                let server: crate::world::terrain::Generator = match kind {
                    WorldgenKind::Flat => std::sync::Arc::new(FlatTerrain::new(&mut registry, seed)),
                    WorldgenKind::Diffusion => crate::world::terrain::generator(&mut registry, seed, cfg),
                };
                let world = World::with_kind_cfg(seed, RenderConfig::default(), kind, cfg, false);
                let player = spawn_player(&world);
                let spawn = spawn_column(server.as_ref());
                assert_eq!(player.position.to_array().map(f64::to_bits), spawn.to_array().map(f64::to_bits), "seed {seed} {kind:?}");
                // The round start world names its spawn; a flat world spirals to level ground.
                assert_eq!(server.chart_spawn().is_some(), kind == WorldgenKind::Diffusion, "seed {seed} {kind:?}");
                let pull = world.gravity_at(spawn).accel;
                assert_eq!(player.up_axis, crate::player::standing_pose(pull).1);
            }
        }
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
        let conn = Connection::connect("127.0.0.1", handle.addr().port(), "probe", "").expect("join");
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

        let mut conn = Connection::connect("127.0.0.1", port, "each", "").expect("join");
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

        let conn = Connection::connect("127.0.0.1", port, "bulk", "").expect("join");
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

    /// The engine as [`push_gfx`] sees it. It allocates MSAA up to `msaa_cap`, as a device short
    /// of VRAM falls back, and counts every render-lane flag transition.
    struct FakeGfx {
        extent: (u32, u32),
        flags: voxel_engine::RenderFlags,
        transitions: u32,
        bloom_ever_on: bool,
        msaa: u32,
        msaa_cap: u32,
        scale: f32,
    }

    impl FakeGfx {
        /// Created as `App::run` creates the engine: the masked flags of `settings`.
        fn new(settings: &Settings, mask: VisualMask, msaa_cap: u32) -> Self {
            let flags = mask.effective_render(settings).engine_flags();
            Self { extent: (1280, 720), flags, transitions: 0, bloom_ever_on: flags.bloom, msaa: settings.msaa.min(msaa_cap), msaa_cap, scale: 1.0 }
        }
    }

    impl GfxEngine for FakeGfx {
        fn window_extent(&self) -> (u32, u32) {
            self.extent
        }
        fn gpu_caps(&self) -> voxel_engine::GpuCaps {
            voxel_engine::GpuCaps {
                device_name: String::new(),
                device_local_bytes: 0,
                device_local_heap_size: 0,
                device_local_budget: None,
                device_local_usage: None,
                max_texture_array_layers: 2048,
                max_msaa: 8,
                supports_vrs: false,
                supports_pipeline_stats: false,
            }
        }
        fn vrs_useful_above_pixels(&self) -> Option<u32> {
            None
        }
        fn estimate_render_targets(&self, _: u32, _: u32, _: f32, _: u32, _: RenderConfig) -> u64 {
            0
        }
        fn set_fullscreen(&mut self, _: bool) {}
        fn set_msaa(&mut self, samples: u32) -> u32 {
            self.msaa = samples.min(self.msaa_cap);
            self.msaa
        }
        fn set_render_scale(&mut self, scale: f32) -> f32 {
            self.scale = scale;
            scale
        }
        fn set_cull_faces(&mut self, _: bool) {}
        fn set_flags(&mut self, flags: voxel_engine::RenderFlags) {
            if flags != self.flags {
                self.transitions += 1;
                self.bloom_ever_on |= flags.bloom;
                self.flags = flags;
            }
        }
        fn msaa(&self) -> u32 {
            self.msaa
        }
        fn render_scale(&self) -> f32 {
            self.scale
        }
    }

    /// With the Post mod off, a push writes the masked flags once: one transition per change of
    /// a live lane, none for a lane the mod strips, and bloom never turns on in between.
    #[test]
    fn with_a_visual_mod_off_each_apply_is_one_flag_transition() {
        use crate::render_config::VisualGroup;
        let mask = VisualMask::of([VisualGroup::Atmosphere, VisualGroup::Lighting]);
        let mut settings = Settings::default();
        settings.bloom = true;
        let mut eng = FakeGfx::new(&settings, mask, 8);
        let mut applied = None;
        assert!(push_gfx(&mut eng, &mut settings, mask, &mut applied), "the first frame pushes");
        assert_eq!(eng.transitions, 0, "the engine was created with these flags");
        for frame in 0..5 {
            assert!(!push_gfx(&mut eng, &mut settings, mask, &mut applied), "quiet frame {frame} pushes nothing");
        }
        for (lane, flip) in [("shadows", (|s: &mut Settings| s.shadows = !s.shadows) as fn(&mut Settings)), ("fog", |s| s.fog = !s.fog)] {
            let before = eng.transitions;
            flip(&mut settings);
            assert!(push_gfx(&mut eng, &mut settings, mask, &mut applied), "{lane}");
            assert_eq!(eng.transitions, before + 1, "{lane}: one transition");
            assert!(!push_gfx(&mut eng, &mut settings, mask, &mut applied), "{lane}: settled");
        }
        let before = eng.transitions;
        settings.godrays = !settings.godrays;
        push_gfx(&mut eng, &mut settings, mask, &mut applied);
        assert_eq!(eng.transitions, before, "a stripped lane moves nothing");
        eng.extent = (2560, 1440);
        assert!(push_gfx(&mut eng, &mut settings, mask, &mut applied), "a resize pushes");
        assert!(eng.transitions <= before + 1);
        assert!(!push_gfx(&mut eng, &mut settings, mask, &mut applied), "and settles in the same frame");
        assert!(!eng.bloom_ever_on, "the Post mod's bloom never reached the engine");
    }

    /// A device that cannot allocate the MSAA asked for: the fallback is adopted once, and every
    /// later frame compares stamps without a single allocation (the notice is not cloned).
    #[test]
    fn push_gfx_allocates_nothing_in_steady_state() {
        let mask = VisualMask::default();
        let mut settings = Settings::default();
        settings.msaa = 4;
        let mut eng = FakeGfx::new(&settings, mask, 1);
        let mut applied = None;
        assert!(push_gfx(&mut eng, &mut settings, mask, &mut applied));
        assert!(settings.vram_notice.is_some(), "the fallback is noticed");
        assert_eq!(settings.session_msaa_scale(1280, 720).0, 1, "the session runs at what the device gave");
        assert!(!push_gfx(&mut eng, &mut settings, mask, &mut applied), "the fallback is not pushed again");
        for frame in 0..8 {
            crate::alloc_count::reset();
            let pushed = push_gfx(&mut eng, &mut settings, mask, &mut applied);
            assert_eq!((pushed, crate::alloc_count::alloc_count()), (false, 0), "steady frame {frame}");
        }
    }

    /// Left held on a Video row at key-repeat rate for two seconds, frames 8 ms apart, the way
    /// the menu screen runs them: nothing is written while the value moves, and one write lands
    /// once it has been still for the debounce window.
    #[test]
    fn holding_left_writes_settings_at_most_once_per_debounce_window() {
        use crate::menu::menus::SettingsPage;
        use crate::menu::{Dir, Intent};
        let mut settings = Settings::default();
        let mut options = crate::settings::Options::new();
        let session = Session::default();
        let mut stack = MenuStack::new(Framed::boxed(SettingsPage::new(crate::settings::Category::Video)));
        let mut flush = Debounce::new();
        let (mut writes, mut steps) = (Vec::new(), 0);
        for frame in 0..400u64 {
            let now_ms = frame * 8;
            let held = now_ms < 2000 && frame % 4 == 0;
            let intents = if held { vec![Intent::Adjust(Dir::Prev)] } else { Vec::new() };
            let before = settings.clone();
            let mut ctx = Ctx::bare(&mut settings, &mut options, &session);
            stack.update(&intents, &mut ctx);
            let changed = settings != before;
            steps += changed as u32;
            if settings_write_due(&mut flush, changed, now_ms) {
                writes.push(now_ms);
            }
        }
        assert!(steps > 40, "the held key moved the value ({steps} steps)");
        assert_eq!(writes.len(), 1, "one write, after the release: {writes:?}");
        assert!((2000..2000 + Debounce::IDLE_MS + 16).contains(&writes[0]), "written {}ms in", writes[0]);
        assert!(!flush.take(), "nothing left pending");
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
