//! Save-slot identity and metadata: the types the menu, store, and codec share.
//!
//! `SlotId` is proof-carrying — constructing one is the filename validation,
//! so everything downstream takes `&SlotId` and the question "is this a legal
//! save name" is answered exactly once, at the UI boundary.

use std::fmt;
use std::io;

/// Validated save name; private field ensures it's only constructed via `new()`, making paths always safe.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SlotId(String);

impl SlotId {
    /// Rejects path traversal, filesystem-confusing chars, control chars, and length outside 1..64 bytes.
    pub fn new(raw: &str) -> Result<Self, SaveError> {
        if raw.is_empty() {
            return Err(SaveError::BadName("name is empty"));
        }
        if raw.len() > 64 {
            return Err(SaveError::BadName("name is longer than 64 bytes"));
        }
        if raw.trim() != raw {
            return Err(SaveError::BadName("name starts or ends with whitespace"));
        }
        if raw.chars().any(|c| c.is_control() || matches!(c, '/' | '\\' | '.')) {
            return Err(SaveError::BadName("name contains / \\ . or control characters"));
        }
        Ok(Self(raw.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SlotId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Metadata the menu displays; fixed offsets at file start enable peeking from ~100-byte prefix.
#[derive(Clone, Debug, PartialEq)]
pub struct SaveMeta {
    /// Display name; independent of the `SlotId` so renames don't move files.
    /// Clamped to 64 bytes on encode.
    pub name: String,
    pub seed: i64,
    /// Unix seconds.
    pub created: u64,
    /// Unix seconds.
    pub last_played: u64,
    pub playtime_secs: u64,
    /// Duplicated from the body so peeking never parses edits.
    pub edit_count: u32,
}

/// One slot as the menu sees it. A corrupt-but-present world stays visible
/// (with the error) instead of silently vanishing from the list.
#[derive(Debug)]
pub struct Slot {
    pub id: SlotId,
    pub meta: Result<SaveMeta, SaveError>,
}

#[derive(Debug)]
pub enum SaveError {
    Io(io::Error),
    /// Unparseable or structurally invalid file.
    Corrupt(&'static str),
    BadVersion(u16),
    BadName(&'static str),
}

impl fmt::Display for SaveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SaveError::Io(e) => write!(f, "{e}"),
            SaveError::Corrupt(what) => write!(f, "corrupt save: {what}"),
            SaveError::BadVersion(v) => write!(f, "unsupported save version {v}"),
            SaveError::BadName(why) => write!(f, "bad save name: {why}"),
        }
    }
}

impl std::error::Error for SaveError {}

impl From<io::Error> for SaveError {
    fn from(e: io::Error) -> Self {
        SaveError::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_ordinary_names() {
        for name in ["world", "world-2", "My World_3", "a"] {
            assert!(SlotId::new(name).is_ok(), "{name:?} should be legal");
        }
    }

    #[test]
    fn rejects_path_escapes_and_junk() {
        for name in ["", "../evil", "a/b", "a\\b", "x.save", " padded", "padded ", "nul\0"] {
            assert!(SlotId::new(name).is_err(), "{name:?} should be rejected");
        }
        let long = "x".repeat(65);
        assert!(SlotId::new(&long).is_err());
        for name in ["世界", "åäö", "save_1"] {
            assert!(SlotId::new(name).is_ok(), "{name:?} should be legal");
        }
    }

    #[test]
    fn store_never_writes_outside_saves() {
        use std::fs;
        use std::path::Path;
        let id = SlotId::new("世界").unwrap();
        let path = Path::new("saves").join(format!("{id}.save"));
        assert!(path.starts_with("saves"));
        assert_eq!(path.parent(), Some(Path::new("saves")));
        let _ = fs::remove_file(&path);
        super::super::store::write(&id, b"x").unwrap();
        assert!(path.exists());
        assert!(!Path::new("世界.save").exists());
        let _ = fs::remove_file(&path);
    }
}
