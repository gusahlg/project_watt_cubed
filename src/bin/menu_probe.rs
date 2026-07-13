//! menu_probe — manual diagnostic (not part of the acceptance suite).
//! Draws menu screens straight to PNGs so the scale-to-fit layout can be
//! inspected without driving input: the Video settings page (the row-heaviest
//! screen) at menu_scale 1.0 and 2.0, and the main menu.

use std::path::PathBuf;

use project_watt_cubed::menu::menus::{MainMenu, SettingsPage};
use project_watt_cubed::menu::{Ctx, DefaultTheme, Framed, ModRow, Screen};
use project_watt_cubed::mods::Mods;
use project_watt_cubed::session::Session;
use project_watt_cubed::settings::{Category, Settings};

fn main() {
    let mods = Mods::with_defaults();
    let mod_rows = ModRow::snapshot(&mods);
    let session = Session::default();

    let shots: [(&str, f32, Box<dyn Screen>); 3] = [
        ("menu_video_1x", 1.0, Framed::boxed(SettingsPage::new(Category::Video))),
        ("menu_video_2x", 2.0, Framed::boxed(SettingsPage::new(Category::Video))),
        ("menu_main", 1.0, Framed::boxed(MainMenu::new())),
    ];

    let config = voxel_engine::Config {
        title: "menu-probe".into(),
        width: 1280,
        height: 720,
        vsync: false,
        resizable: false,
        ..Default::default()
    };

    let mut shots = shots.into_iter();
    let mut current = shots.next();
    let mut warm = 0u32;
    voxel_engine::run(config, move |eng| {
        let Some((name, scale, screen)) = &current else { return false };
        let mut settings = Settings { menu_scale: *scale, ..Settings::default() };
        let (w, h) = (eng.screen_width(), eng.screen_height());
        let mut f = eng.begin_frame(voxel_engine::Color::BLACK.to_linear());
        let ctx = Ctx { settings: &mut settings, saves: &[], mods: &mod_rows, session: &session };
        screen.draw(&ctx, &DefaultTheme, &mut f, w, h);
        drop(f);
        // A couple of warm frames so the first capture isn't a blank swapchain.
        if warm < 3 {
            warm += 1;
            return true;
        }
        let path = PathBuf::from(format!("/tmp/watt-menu/{name}.png"));
        std::fs::create_dir_all("/tmp/watt-menu").unwrap();
        voxel_engine::skeleton::screenshot_to(eng, &path).expect("capture failed");
        eprintln!("captured {name}");
        current = shots.next();
        true
    });
}
