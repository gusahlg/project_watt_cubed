//! app.rs owns the window and the top-level state machine: the start menu, an
//! in-world [`Game`], and the mod menu. It routes each frame to the active screen,
//! creates and loads worlds, and autosaves when leaving one.
//!
//! Field order is drop order: `game` owns the GPU chunk models, which must be freed
//! while the GL context is still alive, so it is declared before `rl` (whose drop
//! closes the window).
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use raylib::prelude::*;

use crate::console::shadowed;
use crate::game::{Game, Signal};
use crate::menu::{FormResult, HostInfo, HostMenu, JoinInfo, JoinMenu, MainChoice, MainMenu, ModMenu};
use crate::mods::Mods;
use crate::net::client::Connection;
use crate::net::server::{self, Config, ServerHandle};
use crate::player::Player;
use crate::save;
use crate::world::World;

const STARTING_WINDOW_WIDTH: i32 = 1280;
const STARTING_WINDOW_HEIGHT: i32 = 720;
const TARGET_FPS: u32 = 60;

/// Which top-level screen is active.
enum Screen {
    Menu,
    Playing,
    Mods,
    Host,
    Join,
}

/// The whole program: window, the installed mods (persist across worlds), the
/// menus, and the current world if one is open.
pub struct App {
    /// The live world, if the player is in one. Owns GPU models — drops before `rl`.
    game: Option<Game>,
    menu: MainMenu,
    mod_menu: ModMenu,
    host_menu: HostMenu,
    join_menu: JoinMenu,
    /// Installed mods and their on/off state; shared with the game while playing.
    mods: Mods,
    screen: Screen,
    /// The integrated server when hosting, kept alive for the session so friends can
    /// stay connected; stopping it frees the port for a later host.
    host: Option<ServerHandle>,
    /// A one-line status/error shown under the start menu (e.g. a failed connect).
    status: Option<String>,
    rl: RaylibHandle,
    thread: RaylibThread,
}

impl App {
    /// Open the window and start at the menu.
    pub fn new() -> Self {
        let (mut rl, thread) = raylib::init()
            .size(STARTING_WINDOW_WIDTH, STARTING_WINDOW_HEIGHT)
            .title("Project Watt Cubed")
            .build();

        rl.set_target_fps(TARGET_FPS);
        // We manage quitting ourselves so Esc can back out of screens.
        rl.set_exit_key(None);

        Self {
            game: None,
            menu: MainMenu::new(),
            mod_menu: ModMenu::new(),
            host_menu: HostMenu::new(),
            join_menu: JoinMenu::new(),
            mods: Mods::with_defaults(),
            screen: Screen::Menu,
            host: None,
            status: None,
            rl,
            thread,
        }
    }

    /// Run until the window closes or the player quits from the menu.
    pub fn run(mut self) {
        while !self.rl.window_should_close() {
            let quit = match self.screen {
                Screen::Menu => self.update_menu(),
                Screen::Playing => {
                    self.update_playing();
                    false
                }
                Screen::Mods => {
                    self.update_mods();
                    false
                }
                Screen::Host => {
                    self.update_host();
                    false
                }
                Screen::Join => {
                    self.update_join();
                    false
                }
            };
            if quit {
                break;
            }
            self.draw();
        }
        // Persist the open world on the way out.
        self.autosave();
    }

    /// Start-menu logic. Returns `true` to quit the program.
    fn update_menu(&mut self) -> bool {
        if let Some(choice) = self.menu.update(&self.rl) {
            self.status = None;
            match choice {
                MainChoice::NewWorld => self.start_new_world(),
                MainChoice::Load(name) => self.load_world(&name),
                MainChoice::Host => self.screen = Screen::Host,
                MainChoice::Join => self.screen = Screen::Join,
                MainChoice::Mods => self.screen = Screen::Mods,
                MainChoice::Quit => return true,
            }
        }
        false
    }

    /// Host screen: fill in the form, then start an integrated server and connect to
    /// it locally. Esc returns to the menu.
    fn update_host(&mut self) {
        match self.host_menu.update(&mut self.rl) {
            FormResult::Submit(info) => self.start_host(info),
            FormResult::Cancel => self.screen = Screen::Menu,
            FormResult::Editing => {}
        }
    }

    /// Join screen: fill in the address/port/password, then connect. Esc returns.
    fn update_join(&mut self) {
        match self.join_menu.update(&mut self.rl) {
            FormResult::Submit(info) => self.start_join(info),
            FormResult::Cancel => self.screen = Screen::Menu,
            FormResult::Editing => {}
        }
    }

    /// Spin up a fresh integrated server and join it on loopback. Any previous host
    /// is stopped first so its port is free to reuse.
    fn start_host(&mut self, info: HostInfo) {
        if let Some(previous) = self.host.take() {
            previous.stop();
            thread::sleep(Duration::from_millis(150));
        }
        let seed = fresh_seed();
        let config = Config { password: info.password.clone(), seed };
        match server::spawn(info.port, config) {
            Ok(handle) => {
                let port = handle.addr().port();
                self.host = Some(handle);
                // Connect our own client to the server we just started.
                match Connection::connect("127.0.0.1", port, &info.name, &info.password) {
                    Ok(conn) => self.enter_net_game(conn),
                    Err(e) => self.fail_to_menu(format!("hosted, but could not connect: {e}")),
                }
            }
            Err(e) => self.fail_to_menu(format!("could not host on port {}: {e}", info.port)),
        }
    }

    /// Connect to a remote server and enter its world.
    fn start_join(&mut self, info: JoinInfo) {
        match Connection::connect(&info.host, info.port, &info.name, &info.password) {
            Ok(conn) => self.enter_net_game(conn),
            Err(e) => self.fail_to_menu(format!("could not join: {e}")),
        }
    }

    /// Build the local world from the server's seed and spawn, then enter play with
    /// the connection attached.
    fn enter_net_game(&mut self, conn: Connection) {
        let world = World::new(conn.seed());
        let player = Player::new(conn.spawn());
        // A networked world is a live mirror, not a save — start from clean defaults.
        self.mods = Mods::with_defaults();
        let game = Game::new(world, player, "multiplayer".to_string()).with_net(conn);
        self.enter_game(game);
    }

    /// Report a connection/host failure and return to the menu.
    fn fail_to_menu(&mut self, message: String) {
        self.status = Some(message);
        self.menu.refresh();
        self.screen = Screen::Menu;
    }

    /// Create a fresh world with a time-seeded generator and enter it.
    fn start_new_world(&mut self) {
        let seed = fresh_seed();
        let world = World::new(seed);
        let player = spawn_player(&world);
        let name = save::next_new_name();

        // A new world starts from a clean default mod set (empty inventory, etc.).
        self.mods = Mods::with_defaults();
        self.enter_game(Game::new(world, player, name));
    }

    /// Load an existing save and enter it. Stays on the menu if loading fails.
    fn load_world(&mut self, name: &str) {
        self.mods = Mods::with_defaults();
        match save::load(name, &mut self.mods) {
            Ok((world, player)) => self.enter_game(Game::new(world, player, name.to_string())),
            Err(_) => {}
        }
    }

    /// Install a freshly built game as the active screen.
    fn enter_game(&mut self, mut game: Game) {
        game.on_enter(&mut self.rl);
        self.game = Some(game);
        self.screen = Screen::Playing;
    }

    /// In-world logic; leaves to the menu (autosaving) when the game signals it.
    fn update_playing(&mut self) {
        let signal = match &mut self.game {
            Some(game) => game.update(&mut self.rl, &self.thread, &mut self.mods),
            None => Signal::ExitToMenu,
        };
        if let Signal::ExitToMenu = signal {
            self.autosave();
            self.rl.enable_cursor();
            self.game = None;
            self.menu.refresh();
            self.screen = Screen::Menu;
        }
    }

    /// Mod-menu logic; Esc returns to the start menu.
    fn update_mods(&mut self) {
        if self.mod_menu.update(&self.rl, &mut self.mods) {
            self.screen = Screen::Menu;
        }
    }

    /// Save the open world, if any (best-effort — a failed save shouldn't crash).
    /// Networked worlds are server mirrors, not local saves, so they're never written.
    fn autosave(&mut self) {
        if let Some(game) = &self.game {
            if game.is_multiplayer() {
                return;
            }
            let _ = save::save(game.save_name(), game.world(), game.player(), &self.mods);
        }
    }

    /// Draw the active screen.
    fn draw(&mut self) {
        match self.screen {
            Screen::Playing => {
                if let Some(game) = &mut self.game {
                    game.draw(&mut self.rl, &self.thread, &self.mods);
                }
            }
            Screen::Menu => {
                let (w, h) = (self.rl.get_screen_width(), self.rl.get_screen_height());
                let mut d = self.rl.begin_drawing(&self.thread);
                self.menu.draw(&mut d, w, h);
                // A connect/host error from the last attempt, in red under the list.
                if let Some(status) = &self.status {
                    let fs = 20;
                    let sx = (w - d.measure_text(status, fs)) / 2;
                    shadowed(&mut d, status, sx, h - 70, fs, Color::SALMON);
                }
            }
            Screen::Mods => {
                let (w, h) = (self.rl.get_screen_width(), self.rl.get_screen_height());
                let mut d = self.rl.begin_drawing(&self.thread);
                self.mod_menu.draw(&mut d, &self.mods, w, h);
            }
            Screen::Host => {
                let (w, h) = (self.rl.get_screen_width(), self.rl.get_screen_height());
                let mut d = self.rl.begin_drawing(&self.thread);
                self.host_menu.draw(&mut d, w, h);
            }
            Screen::Join => {
                let (w, h) = (self.rl.get_screen_width(), self.rl.get_screen_height());
                let mut d = self.rl.begin_drawing(&self.thread);
                self.join_menu.draw(&mut d, w, h);
            }
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

/// Spawn the player just above the surface at the world origin, so they drop and
/// land on solid ground.
fn spawn_player(world: &World) -> Player {
    let surface = world.surface_y(0, 0);
    Player::new(Vector3::new(0.5, surface as f32 + 3.0, 0.5))
}
