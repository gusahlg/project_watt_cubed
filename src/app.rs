//! app.rs owns the window and the top-level state machine: the start menu, an
//! in-world [`Game`], and the mod menu. It routes each frame to the active screen,
//! creates and loads worlds, and autosaves when leaving one.
//!
//! Field order is drop order: `game` owns the GPU chunk models, which must be freed
//! while the GL context is still alive, so it is declared before `rl` (whose drop
//! closes the window).
use std::time::{SystemTime, UNIX_EPOCH};

use raylib::prelude::*;

use crate::game::{Game, Signal};
use crate::menu::{MainChoice, MainMenu, ModMenu};
use crate::mods::Mods;
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
}

/// The whole program: window, the installed mods (persist across worlds), the
/// menus, and the current world if one is open.
pub struct App {
    /// The live world, if the player is in one. Owns GPU models — drops before `rl`.
    game: Option<Game>,
    menu: MainMenu,
    mod_menu: ModMenu,
    /// Installed mods and their on/off state; shared with the game while playing.
    mods: Mods,
    screen: Screen,
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
            mods: Mods::with_defaults(),
            screen: Screen::Menu,
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
            match choice {
                MainChoice::NewWorld => self.start_new_world(),
                MainChoice::Load(name) => self.load_world(&name),
                MainChoice::Mods => self.screen = Screen::Mods,
                MainChoice::Quit => return true,
            }
        }
        false
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
    fn autosave(&mut self) {
        if let Some(game) = &self.game {
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
            }
            Screen::Mods => {
                let (w, h) = (self.rl.get_screen_width(), self.rl.get_screen_height());
                let mut d = self.rl.begin_drawing(&self.thread);
                self.mod_menu.draw(&mut d, &self.mods, w, h);
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
