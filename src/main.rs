//! The vanilla executable: the bare game, no module packages. A PWC with modules is a different build —
//! `pwc build` (the PWC package manager) generates a crate whose `main` calls
//! [`project_watt_cubed::run`] with the instance's packages.
use project_watt_cubed::modules::GameBuild;

fn main() {
    project_watt_cubed::run(GameBuild::vanilla());
}
