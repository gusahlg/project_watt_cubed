//! Remembered connection details, so the Host/Join forms pre-fill what you used
//! last time instead of the bare defaults.
//!
//! Stored as plain `key=value` lines in `saves/session.cfg` (std-only, mirroring
//! [`settings`](crate::settings)). Passwords are deliberately never persisted.
use std::fs;
use std::path::Path;

const SESSION_PATH: &str = "saves/session.cfg";

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
        if let Ok(text) = fs::read_to_string(SESSION_PATH) {
            for line in text.lines() {
                let Some((key, value)) = line.split_once('=') else {
                    continue;
                };
                let value = value.trim().to_string();
                match key.trim() {
                    "address" => s.address = value,
                    "port" => s.port = value,
                    "name" => s.name = value,
                    _ => {}
                }
            }
        }
        s
    }

    /// Best-effort save (a failed write shouldn't crash the game).
    pub fn save(&self) {
        if let Some(dir) = Path::new(SESSION_PATH).parent() {
            let _ = fs::create_dir_all(dir);
        }
        let text = format!(
            "address={}\nport={}\nname={}\n",
            self.address, self.port, self.name
        );
        let _ = fs::write(SESSION_PATH, text);
    }
}
