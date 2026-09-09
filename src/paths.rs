//! Application data and config directories, resolved once at startup.
//!
//! Worlds live under [`Paths::data`]; settings, session, and mod choices under
//! [`Paths::config`]. Override with `WATT_DATA_DIR` (or `watt_server --data-dir`),
//! else a launch-directory `saves/` folder is kept as-is, else XDG.
//! `WATT_CHECKOUT_DIR` (`play.sh`) is the source checkout, kept only when it
//! contains `Cargo.toml`.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const APP: &str = "project_watt_cubed";
const ENV_DATA_DIR: &str = "WATT_DATA_DIR";
const ENV_CHECKOUT_DIR: &str = "WATT_CHECKOUT_DIR";

/// Worlds under `data`; `settings.cfg` / `session.cfg` / `mods.cfg` under `config`.
/// Optional source checkout when `WATT_CHECKOUT_DIR` points at a `Cargo.toml`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Paths {
    pub data: PathBuf,
    pub config: PathBuf,
    checkout: Option<PathBuf>,
}

static PATHS: OnceLock<Paths> = OnceLock::new();

impl Paths {
    /// Resolve override → existing launch `saves/` → XDG, without installing globally.
    /// `checkout_dir` is kept only when `<dir>/Cargo.toml` exists.
    pub(crate) fn resolve(override_dir: Option<&Path>, checkout_dir: Option<&Path>) -> Self {
        let mut paths = pick(
            override_dir
                .filter(|p| !p.as_os_str().is_empty())
                .map(|p| absolutize(p.to_path_buf()))
                .or_else(env_override),
            launch_saves(),
            xdg_dir("XDG_DATA_HOME", ".local/share"),
            xdg_dir("XDG_CONFIG_HOME", ".config"),
        );
        paths.checkout = resolve_checkout(checkout_dir);
        paths
    }

    /// Install the process-wide roots and print them once. First caller wins.
    pub fn init(override_dir: Option<&Path>) -> &'static Self {
        PATHS.get_or_init(|| {
            let paths = startup(override_dir);
            #[cfg(not(test))]
            {
                let checkout = paths.checkout_dir().unwrap_or(Path::new("-"));
                println!(
                    "paths: data={} config={} checkout={}",
                    paths.data.display(),
                    paths.config.display(),
                    checkout.display()
                );
            }
            paths
        })
    }

    /// The installed roots, resolving defaults on first use.
    pub(crate) fn get() -> &'static Self {
        Self::init(None)
    }

    pub(crate) fn settings_file(&self) -> PathBuf {
        self.config.join("settings.cfg")
    }

    pub(crate) fn session_file(&self) -> PathBuf {
        self.config.join("session.cfg")
    }

    pub(crate) fn mods_file(&self) -> PathBuf {
        self.config.join("mods.cfg")
    }

    pub(crate) fn checkout_dir(&self) -> Option<&Path> {
        self.checkout.as_deref()
    }

    pub(crate) fn mods_selection_file(&self) -> Option<PathBuf> {
        Some(self.checkout_dir()?.join("mods.toml"))
    }
}

fn startup(override_dir: Option<&Path>) -> Paths {
    #[cfg(test)]
    {
        if override_dir.is_none() {
            return isolated_test_paths();
        }
    }
    Paths::resolve(override_dir, env_checkout().as_deref())
}

fn pick(
    override_dir: Option<PathBuf>,
    launch_saves: Option<PathBuf>,
    xdg_data: PathBuf,
    xdg_config: PathBuf,
) -> Paths {
    if let Some(dir) = override_dir {
        return Paths {
            data: dir.clone(),
            config: dir,
            checkout: None,
        };
    }
    if let Some(saves) = launch_saves {
        return Paths {
            data: saves.clone(),
            config: saves,
            checkout: None,
        };
    }
    Paths {
        data: xdg_data,
        config: xdg_config,
        checkout: None,
    }
}

fn nonempty_path(val: Option<std::ffi::OsString>) -> Option<PathBuf> {
    let val = val.filter(|v| !v.is_empty())?;
    Some(PathBuf::from(val))
}

fn env_override() -> Option<PathBuf> {
    nonempty_path(std::env::var_os(ENV_DATA_DIR)).map(absolutize)
}

fn env_checkout() -> Option<PathBuf> {
    nonempty_path(std::env::var_os(ENV_CHECKOUT_DIR))
}

fn resolve_checkout(dir: Option<&Path>) -> Option<PathBuf> {
    let dir = dir.filter(|p| !p.as_os_str().is_empty())?;
    let dir = absolutize(dir.to_path_buf());
    if dir.join("Cargo.toml").is_file() {
        Some(dir)
    } else {
        println!(
            "paths: WATT_CHECKOUT_DIR={} has no Cargo.toml; ignoring",
            dir.display()
        );
        None
    }
}

fn launch_saves() -> Option<PathBuf> {
    let cwd = std::env::current_dir().ok()?;
    let saves = cwd.join("saves");
    saves.is_dir().then_some(saves)
}

fn xdg_dir(var: &str, under_home: &str) -> PathBuf {
    if let Some(val) = std::env::var_os(var) {
        let p = PathBuf::from(val);
        if p.is_absolute() {
            return p.join(APP);
        }
    }
    match std::env::var_os("HOME") {
        Some(home) => PathBuf::from(home).join(under_home).join(APP),
        None => PathBuf::from(under_home).join(APP),
    }
}

fn absolutize(path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(&path))
            .unwrap_or(path)
    }
}

#[cfg(test)]
fn isolated_test_paths() -> Paths {
    // Per-checkout so parallel worktree `cargo test` runs don't share fixed names.
    let dir = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/target/test-data"));
    let _ = std::fs::create_dir_all(&dir);
    Paths {
        data: dir.clone(),
        config: dir,
        checkout: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonempty_path_drops_empty_and_absent() {
        assert_eq!(nonempty_path(None), None);
        assert_eq!(nonempty_path(Some(std::ffi::OsString::new())), None);
        assert_eq!(
            nonempty_path(Some(std::ffi::OsString::from("/x"))),
            Some(PathBuf::from("/x"))
        );
    }

    #[test]
    fn override_wins_over_saves_and_xdg() {
        let p = pick(
            Some(PathBuf::from("/override")),
            Some(PathBuf::from("/saves")),
            PathBuf::from("/xdg-data"),
            PathBuf::from("/xdg-config"),
        );
        assert_eq!(p.data, PathBuf::from("/override"));
        assert_eq!(p.config, PathBuf::from("/override"));
    }

    #[test]
    fn launch_saves_keeps_one_root() {
        let p = pick(
            None,
            Some(PathBuf::from("/home/dev/game/saves")),
            PathBuf::from("/xdg-data"),
            PathBuf::from("/xdg-config"),
        );
        assert_eq!(p.data, PathBuf::from("/home/dev/game/saves"));
        assert_eq!(p.config, p.data);
    }

    #[test]
    fn xdg_splits_worlds_and_config() {
        let p = pick(
            None,
            None,
            PathBuf::from("/home/u/.local/share/project_watt_cubed"),
            PathBuf::from("/home/u/.config/project_watt_cubed"),
        );
        assert_eq!(
            p.data,
            PathBuf::from("/home/u/.local/share/project_watt_cubed")
        );
        assert_eq!(p.config, PathBuf::from("/home/u/.config/project_watt_cubed"));
        assert_ne!(p.data, p.config);
    }

    #[test]
    fn resolve_override_is_absolute_and_shared() {
        let p = Paths::resolve(Some(Path::new("/tmp/watt-data-dir-test")), None);
        assert_eq!(p.data, PathBuf::from("/tmp/watt-data-dir-test"));
        assert_eq!(p.config, p.data);
        assert_eq!(p.checkout_dir(), None);
    }

    #[test]
    fn test_io_uses_an_injected_root_not_checkout_saves() {
        let paths = Paths::get();
        if let (Ok(data), Ok(checkout)) = (
            paths.data.canonicalize(),
            Path::new("saves").canonicalize(),
        ) {
            assert_ne!(data, checkout);
        } else {
            assert_ne!(paths.data.as_path(), Path::new("saves"));
        }
        assert_eq!(paths.data, paths.config);
        assert_eq!(paths.checkout_dir(), None);
        let checkout_root = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/target/test-data"));
        assert_eq!(paths.data, checkout_root, "each checkout must have its own test data root");
        let marker = paths.data.join("__paths_isolation_marker__");
        std::fs::write(&marker, b"ok").unwrap();
        assert!(marker.exists());
        assert!(!Path::new("saves/__paths_isolation_marker__").exists());
        let _ = std::fs::remove_file(marker);
    }

    #[test]
    fn config_file_helpers_sit_on_the_config_root() {
        let p = Paths {
            data: PathBuf::from("/data"),
            config: PathBuf::from("/config"),
            checkout: None,
        };
        assert_eq!(p.settings_file(), PathBuf::from("/config/settings.cfg"));
        assert_eq!(p.session_file(), PathBuf::from("/config/session.cfg"));
        assert_eq!(p.mods_file(), PathBuf::from("/config/mods.cfg"));
        assert_eq!(p.mods_selection_file(), None);
    }

    #[test]
    fn checkout_dir_accepted_when_cargo_toml_exists() {
        let checkout = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let p = Paths::resolve(Some(Path::new("/tmp/watt-data-dir-test")), Some(&checkout));
        assert_eq!(p.checkout_dir(), Some(checkout.as_path()));
        assert_eq!(p.mods_selection_file(), Some(checkout.join("mods.toml")));
    }

    #[test]
    fn checkout_dir_ignored_without_cargo_toml() {
        let checkout = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/target/test-data"));
        let _ = std::fs::create_dir_all(&checkout);
        assert!(
            !checkout.join("Cargo.toml").is_file(),
            "test-data must not look like a crate root"
        );
        let p = Paths::resolve(Some(Path::new("/tmp/watt-data-dir-test")), Some(&checkout));
        assert_eq!(p.checkout_dir(), None);
        assert_eq!(p.mods_selection_file(), None);
    }

    #[test]
    fn empty_checkout_dir_is_unset() {
        let p = Paths::resolve(
            Some(Path::new("/tmp/watt-data-dir-test")),
            Some(Path::new("")),
        );
        assert_eq!(p.checkout_dir(), None);
        assert_eq!(p.mods_selection_file(), None);
    }
}
