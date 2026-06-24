//! Binary entry point. All the logic lives in the library crate so it can be
//! unit-tested; `main` just launches the app.
use project_watt_cubed::app::App;

fn main() {
    App::new().run();
}
