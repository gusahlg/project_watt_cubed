//! Remembered connection details, so the Host/Join forms pre-fill what you used
//! last time instead of the bare defaults.
//!
//! Stored as plain `key=value` lines in `session.cfg` under the config root
//! (std-only, mirroring [`settings`](crate::settings)). Passwords are deliberately
//! never persisted.
use std::fs;
use std::path::PathBuf;

fn session_path() -> PathBuf {
    crate::paths::Paths::get().session_file()
}

/// The last-used join address, port text, and player name. Values are the raw
/// field text (so an empty port keeps its "use the default" meaning on reload).
pub struct Session {
    pub address: String,
    pub port: String,
    pub name: String,
}

impl Default for Session {
    fn default() -> Self {
        Self {
            address: "127.0.0.1".to_string(),
            port: String::new(),
            name: "player".to_string(),
        }
    }
}

impl Session {
    /// Load from disk, falling back to defaults for missing entries.
    pub fn load() -> Self {
        let mut s = Self::default();
        if let Ok(text) = fs::read_to_string(session_path()) {
            crate::settings::each_kv_line(&text, |key, value| match key {
                "address" => s.address = value.to_string(),
                "port" => s.port = value.to_string(),
                "name" => s.name = value.to_string(),
                _ => {}
            });
        }
        s
    }

    /// Best-effort save (a failed write shouldn't crash the game).
    pub fn save(&self) {
        let text = format!(
            "address={}\nport={}\nname={}\n",
            self.address, self.port, self.name
        );
        let path = session_path();
        if let Err(e) = crate::save::write_atomic_file(&path, text.as_bytes()) {
            crate::save::log_fs_err("write", &path, &e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_uses_defaults_for_missing_and_unknown_keys() {
        let path = session_path();
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir).unwrap();
        }
        fs::write(&path, "address = 10.0.0.2\nunknown=x\nnot-a-pair\nname=watt\n").unwrap();
        assert!(path.starts_with(&crate::paths::Paths::get().config));
        let s = Session::load();
        assert_eq!(s.address, "10.0.0.2");
        assert_eq!(s.name, "watt");
        assert_eq!(s.port, "");
        let _ = fs::remove_file(path);
    }
}
