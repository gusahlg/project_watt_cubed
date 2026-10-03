//! The vanilla executable: the bare game, no mod packages. A modded PWC is a different build —
//! `pwc build` (the PWC package manager) generates a crate whose `main` calls
//! [`project_watt_cubed::run`] with the instance's packages.
use project_watt_cubed::modding::GameBuild;

fn main() {
    project_watt_cubed::run(GameBuild::vanilla());
}
