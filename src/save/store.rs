//! Filesystem layer: atomic writes, a one-deep backup, and the read ladder.
//! Writes use .tmp + sync_all + rename so a crash mid-save can't corrupt the
//! only copy; successful writes rotate the old file to .bak. Reads ladder down
//! (live intact → backup intact → salvage) and report which rung succeeded.
//! Deletes move to trash/ under the data root instead of unlinking for cheap undo.

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use super::format::{self, Decoded};
use super::slot::{SaveError, Slot, SlotId};
use crate::paths::Paths;

fn saves_dir() -> &'static Path {
    Paths::get().data.as_path()
}

fn trash_dir() -> PathBuf {
    saves_dir().join("trash")
}

fn live_path(id: &SlotId) -> PathBuf {
    saves_dir().join(format!("{id}.save"))
}

fn bak_path(id: &SlotId) -> PathBuf {
    saves_dir().join(format!("{id}.save.bak"))
}

fn tmp_path(id: &SlotId) -> PathBuf {
    saves_dir().join(format!("{id}.save.tmp"))
}

/// Which rung of the read ladder produced the bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Live,
    Backup,
}

/// All slots on disk, most recently played first; slots whose header can't be
/// read sort last but stay listed. Only headers are touched, never bodies.
pub fn list() -> Vec<Slot> {
    let mut slots = Vec::new();
    if let Ok(entries) = fs::read_dir(saves_dir()) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("save") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else { continue };
            let Ok(id) = SlotId::new(stem) else { continue };
            let meta = peek_file(&path);
            slots.push(Slot { id, meta });
        }
    }
    slots.sort_by(|a, b| match (&a.meta, &b.meta) {
        (Ok(ma), Ok(mb)) => mb.last_played.cmp(&ma.last_played),
        (Ok(_), Err(_)) => std::cmp::Ordering::Less,
        (Err(_), Ok(_)) => std::cmp::Ordering::Greater,
        (Err(_), Err(_)) => a.id.as_str().cmp(b.id.as_str()),
    });
    slots
}

fn peek_file(path: &std::path::Path) -> Result<super::slot::SaveMeta, SaveError> {
    let f = fs::File::open(path)?;
    let mut v = Vec::with_capacity(format::HEADER_LEN);
    f.take(format::HEADER_LEN as u64).read_to_end(&mut v)?;
    match format::peek_meta(&v) {
        Ok(meta) => Ok(meta),
        Err(_) if v.len() < format::HEADER_LEN => Err(SaveError::Corrupt("not a save")),
        Err(e) => Err(e),
    }
}

/// Prefers: intact live > intact backup > salvaged live > salvaged backup.
pub fn read(id: &SlotId) -> Result<(Decoded, Source), SaveError> {
    let live = fs::read(live_path(id))
        .map_err(SaveError::from)
        .and_then(|bytes| format::decode(&bytes));
    if let Ok(d @ Decoded::Intact(_)) = live {
        return Ok((d, Source::Live));
    }
    let backup = fs::read(bak_path(id))
        .ok()
        .and_then(|bytes| format::decode(&bytes).ok());
    match (live, backup) {
        (_, Some(d @ Decoded::Intact(_))) => Ok((d, Source::Backup)),
        (Ok(salvaged), _) => Ok((salvaged, Source::Live)),
        (Err(_), Some(salvaged)) => Ok((salvaged, Source::Backup)),
        (Err(e), None) => Err(e),
    }
}

fn sibling_tmp(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".tmp");
    PathBuf::from(name)
}

/// Write `bytes` to `path` via a sibling `.tmp`, `sync_all`, then rename.
/// A crash mid-write leaves the previous file intact. Success leaves no `.tmp`.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let tmp = sibling_tmp(path);
    let result = (|| {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        fs::rename(&tmp, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

/// Create parent directories, then [`write_atomic`]. Settings, session, and
/// mods.cfg persist through this.
pub fn write_atomic_file(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    write_atomic(path, bytes)
}

/// Log a filesystem error instead of `let _ =`. Callers stay best-effort.
pub(crate) fn log_fs_err(op: &str, path: &Path, err: &io::Error) {
    eprintln!("could not {op} {}: {err}", path.display());
}

#[cfg(test)]
pub(crate) fn test_temp_path(tag: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "watt-{tag}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ))
}

/// Atomically replace a slot's bytes, rotating the previous file to `.bak`.
pub fn write(id: &SlotId, bytes: &[u8]) -> io::Result<()> {
    fs::create_dir_all(saves_dir())?;
    let tmp = tmp_path(id);
    let live = live_path(id);
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    if live.exists() {
        fs::rename(&live, bak_path(id))?;
    }
    fs::rename(&tmp, &live)
}

/// Move a slot to a new id. The display name in the header is patched to
/// match; the backup follows best-effort.
pub fn rename(from: &SlotId, to: &SlotId) -> Result<(), SaveError> {
    if live_path(to).exists() {
        return Err(SaveError::Io(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("a save named {to} already exists"),
        )));
    }
    let mut bytes = fs::read(live_path(from))?;
    // Patch-then-write instead of fs::rename so the operation stays atomic
    // and the old file becomes the new slot's backup.
    format::set_name(&mut bytes, to.as_str())?;
    write(to, &bytes)?;
    let from_bak = bak_path(from);
    if let Err(e) = fs::rename(&from_bak, bak_path(to)) {
        if e.kind() != io::ErrorKind::NotFound {
            log_fs_err("rename", &from_bak, &e);
        }
    }
    fs::remove_file(live_path(from))?;
    Ok(())
}

/// Move a slot (and its backup) into `trash/` under the data root rather than unlinking.
pub fn delete(id: &SlotId) -> io::Result<()> {
    fs::create_dir_all(trash_dir())?;
    let dest = unused_trash_path(id);
    fs::rename(live_path(id), &dest)?;
    let _ = fs::remove_file(bak_path(id)); // backup has no value once trashed
    Ok(())
}

fn unused_trash_path(id: &SlotId) -> PathBuf {
    let first = trash_dir().join(format!("{id}.save"));
    if !first.exists() {
        return first;
    }
    (2..)
        .map(|n| trash_dir().join(format!("{id}-{n}.save")))
        .find(|p| !p.exists())
        .expect("an unused trash name exists")
}

/// Copy a slot to a fresh id, patching the display name for disambiguation.
pub fn duplicate(id: &SlotId) -> Result<SlotId, SaveError> {
    let mut bytes = fs::read(live_path(id))?;
    let new_id = unused_id(&format!("{id}-copy"))?;
    format::set_name(&mut bytes, new_id.as_str())?;
    write(&new_id, &bytes)?;
    Ok(new_id)
}

/// A fresh, unused slot for a new world (`world`, `world-2`, …).
pub fn fresh_id() -> SlotId {
    unused_id("world").expect("default save names are valid")
}

fn unused_id(base: &str) -> Result<SlotId, SaveError> {
    let id = SlotId::new(base)?;
    if !live_path(&id).exists() {
        return Ok(id);
    }
    (2..)
        .map(|n| SlotId::new(&format!("{base}-{n}")))
        .find(|id| id.as_ref().is_ok_and(|id| !live_path(id).exists()))
        .expect("an unused save name exists")
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::format::{Edit, PlayerState, SaveDoc, WorldgenStamp};
    use super::super::slot::SaveMeta;

    fn doc(name: &str, edits: u32) -> SaveDoc {
        SaveDoc {
            worldgen_version: 2,
            worldgen: WorldgenStamp::default(),
            law_stamp: material::Law::v0().stamp(),
            meta: SaveMeta {
                name: name.to_string(),
                seed: 7,
                created: 100,
                last_played: 200,
                playtime_secs: 50,
                edit_count: edits,
            },
            player: PlayerState {
                pos: [0.0, 40.0, 0.0],
                yaw: 0.0,
                pitch: 0.0,
                flying: false,
                noclip: false,
                stash: Some(vec![]),
            },
            specs: vec!["air".to_string()],
            edits: (0..edits as i32).map(|i| Edit { x: i, y: 200, z: -i, spec: 0 }).collect(),
            mods: vec![],
        }
    }

    fn cleanup(id: &SlotId) {
        let _ = fs::remove_file(live_path(id));
        let _ = fs::remove_file(bak_path(id));
        let _ = fs::remove_file(tmp_path(id));
    }

    fn edits_of(d: Decoded) -> Vec<Edit> {
        match d {
            Decoded::Intact(doc) => doc.edits,
            Decoded::Salvaged { doc, .. } => doc.edits,
        }
    }

    #[test]
    fn write_rotates_a_backup_and_read_prefers_live() {
        let id = SlotId::new("__store_rotate__").unwrap();
        cleanup(&id);

        write(&id, &format::encode(&doc("v1", 1)).unwrap()).unwrap();
        write(&id, &format::encode(&doc("v2", 2)).unwrap()).unwrap();
        assert!(bak_path(&id).exists(), "first write became the backup");

        let (decoded, source) = read(&id).unwrap();
        assert_eq!(source, Source::Live);
        assert_eq!(edits_of(decoded).len(), 2);

        cleanup(&id);
    }

    #[test]
    fn read_falls_back_to_intact_backup_when_live_is_corrupt() {
        let id = SlotId::new("__store_ladder__").unwrap();
        cleanup(&id);

        write(&id, &format::encode(&doc("v1", 1)).unwrap()).unwrap();
        write(&id, &format::encode(&doc("v2", 2)).unwrap()).unwrap();
        fs::write(live_path(&id), b"WATT garbage").unwrap();

        let (decoded, source) = read(&id).unwrap();
        assert_eq!(source, Source::Backup);
        assert_eq!(edits_of(decoded).len(), 1);

        cleanup(&id);
    }

    #[test]
    fn read_prefers_intact_backup_over_salvaged_live() {
        let id = SlotId::new("__store_prefer_intact__").unwrap();
        cleanup(&id);

        write(&id, &format::encode(&doc("v1", 1)).unwrap()).unwrap();
        write(&id, &format::encode(&doc("v2", 5)).unwrap()).unwrap();
        let live = fs::read(live_path(&id)).unwrap();
        fs::write(live_path(&id), &live[..live.len() - 20]).unwrap(); // salvageable

        let (decoded, source) = read(&id).unwrap();
        assert_eq!(source, Source::Backup, "a whole backup beats a partial live");
        assert_eq!(edits_of(decoded).len(), 1);

        cleanup(&id);
    }

    #[test]
    fn salvaged_live_is_used_when_no_backup_exists() {
        let id = SlotId::new("__store_salvage__").unwrap();
        cleanup(&id);

        write(&id, &format::encode(&doc("v1", 5)).unwrap()).unwrap();
        let live = fs::read(live_path(&id)).unwrap();
        fs::write(live_path(&id), &live[..live.len() - 20]).unwrap();

        let (decoded, source) = read(&id).unwrap();
        assert_eq!(source, Source::Live);
        match decoded {
            Decoded::Salvaged { recovered, expected, .. } => {
                assert_eq!(expected, 5);
                assert!(recovered < 5);
            }
            Decoded::Intact(_) => panic!("truncated live must salvage"),
        }

        cleanup(&id);
    }

    #[test]
    fn missing_slot_is_an_error() {
        let id = SlotId::new("__store_missing__").unwrap();
        cleanup(&id);
        assert!(read(&id).is_err());
    }

    #[test]
    fn duplicate_gets_a_fresh_id_and_name() {
        let id = SlotId::new("__store_dup__").unwrap();
        cleanup(&id);
        write(&id, &format::encode(&doc("original", 2)).unwrap()).unwrap();

        let copy = duplicate(&id).unwrap();
        assert_ne!(copy, id);
        let (decoded, _) = read(&copy).unwrap();
        match decoded {
            Decoded::Intact(d) => {
                assert_eq!(d.meta.name, copy.as_str());
                assert_eq!(d.edits.len(), 2);
            }
            _ => panic!("copy must be intact"),
        }

        cleanup(&copy);
        cleanup(&id);
    }

    #[test]
    fn rename_moves_id_and_patches_display_name() {
        let from = SlotId::new("__store_ren_a__").unwrap();
        let to = SlotId::new("__store_ren_b__").unwrap();
        cleanup(&from);
        cleanup(&to);
        write(&from, &format::encode(&doc("before", 1)).unwrap()).unwrap();

        rename(&from, &to).unwrap();
        assert!(!live_path(&from).exists());
        let (decoded, _) = read(&to).unwrap();
        match decoded {
            Decoded::Intact(d) => assert_eq!(d.meta.name, to.as_str()),
            _ => panic!("renamed save must be intact"),
        }

        // Renaming onto an existing slot is refused.
        write(&from, &format::encode(&doc("again", 1)).unwrap()).unwrap();
        assert!(rename(&from, &to).is_err());

        cleanup(&from);
        cleanup(&to);
    }

    #[test]
    fn delete_moves_to_trash() {
        let id = SlotId::new("__store_del__").unwrap();
        cleanup(&id);
        write(&id, &format::encode(&doc("doomed", 1)).unwrap()).unwrap();

        delete(&id).unwrap();
        assert!(!live_path(&id).exists());
        let trashed = trash_dir().join(format!("{id}.save"));
        assert!(trashed.exists());

        let _ = fs::remove_file(trashed);
    }

    #[test]
    fn both_rungs_corrupt_errors_and_deletes_nothing() {
        let id = SlotId::new("__store_both_bad__").unwrap();
        cleanup(&id);
        write(&id, &format::encode(&doc("v1", 1)).unwrap()).unwrap();
        write(&id, &format::encode(&doc("v2", 2)).unwrap()).unwrap();
        fs::write(live_path(&id), b"WATT garbage live").unwrap();
        fs::write(bak_path(&id), b"WATT garbage bak").unwrap();
        assert!(read(&id).is_err());
        assert!(live_path(&id).exists(), "a failed ladder must not delete live");
        assert!(bak_path(&id).exists(), "a failed ladder must not delete backup");
        cleanup(&id);
    }

    #[test]
    fn peek_file_reads_a_full_header_and_rejects_a_short_read() {
        let id = SlotId::new("__store_peek__").unwrap();
        cleanup(&id);
        let bytes = format::encode(&doc("peeked", 1)).unwrap();
        write(&id, &bytes).unwrap();
        assert_eq!(peek_file(&live_path(&id)).unwrap().name, "peeked");

        fs::write(live_path(&id), &bytes[..format::HEADER_LEN - 1]).unwrap();
        assert!(
            matches!(
                peek_file(&live_path(&id)),
                Err(SaveError::Corrupt("not a save"))
            ),
            "a v8 prefix shorter than header_len(8) is not a save"
        );
        cleanup(&id);
    }

    /// A v7 document can be 164–205 bytes (header 126 + a tiny body). Peek
    /// must not demand the v8 header length; `peek_meta` enforces `header_len(7)`.
    fn v7_bytes(doc: &SaveDoc) -> Vec<u8> {
        let current = format::encode(doc).unwrap();
        let mut v7 = Vec::with_capacity(current.len() - material::STAMP_LEN);
        v7.extend_from_slice(&current[..format::HEADER_LEN_V7]);
        v7.extend_from_slice(&current[format::HEADER_LEN..]);
        v7[4..6].copy_from_slice(&7u16.to_le_bytes());
        v7
    }

    #[test]
    fn peek_lists_a_minimal_v7_save() {
        let id = SlotId::new("__store_peek_v7__").unwrap();
        cleanup(&id);
        let bytes = v7_bytes(&doc("v7tiny", 0));
        assert!(
            bytes.len() < format::HEADER_LEN,
            "fixture must be shorter than the v8 header ({})",
            bytes.len()
        );
        assert!(bytes.len() >= format::HEADER_LEN_V7);
        fs::write(live_path(&id), &bytes).unwrap();

        assert_eq!(peek_file(&live_path(&id)).unwrap().name, "v7tiny");
        let slots = list();
        let listed = slots.iter().find(|s| s.id == id).expect("v7 slot listed");
        assert_eq!(listed.meta.as_ref().unwrap().name, "v7tiny");

        cleanup(&id);
    }

    #[test]
    fn write_atomic_leaves_no_tmp_on_success() {
        let path = test_temp_path("atomic").with_extension("cfg");
        let tmp = sibling_tmp(&path);
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(&tmp);
        write_atomic(&path, b"ok\n").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"ok\n");
        assert!(!tmp.exists(), "successful write must consume the .tmp");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn write_atomic_into_unwritable_dir_is_err() {
        let parent = test_temp_path("atomic-notdir");
        let _ = fs::remove_file(&parent);
        let _ = fs::remove_dir_all(&parent);
        fs::write(&parent, b"not a directory").unwrap();
        let path = parent.join("mods.cfg");
        assert!(write_atomic(&path, b"nope").is_err());
        assert!(!sibling_tmp(&path).exists(), "failed write must not leave a .tmp");
        let _ = fs::remove_file(&parent);
    }

    #[test]
    fn write_atomic_file_creates_missing_parents() {
        let path = test_temp_path("atomic-nested").join("cfg").join("mods.cfg");
        let _ = fs::remove_dir_all(path.parent().unwrap().parent().unwrap());
        write_atomic_file(&path, b"nested\n").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"nested\n");
        assert!(!sibling_tmp(&path).exists());
        let _ = fs::remove_dir_all(path.parent().unwrap().parent().unwrap());
    }

    #[test]
    fn list_shows_corrupt_slots_last_but_present() {
        let good = SlotId::new("__store_list_good__").unwrap();
        let bad = SlotId::new("__store_list_bad__").unwrap();
        cleanup(&good);
        cleanup(&bad);
        write(&good, &format::encode(&doc("good", 0)).unwrap()).unwrap();
        fs::write(live_path(&bad), b"NOPE").unwrap();

        let slots = list();
        let gi = slots.iter().position(|s| s.id == good).expect("good listed");
        let bi = slots.iter().position(|s| s.id == bad).expect("corrupt still listed");
        assert!(slots[gi].meta.is_ok());
        assert!(slots[bi].meta.is_err());
        assert!(gi < bi, "readable slots sort before corrupt ones");

        cleanup(&good);
        cleanup(&bad);
    }

    #[test]
    fn log_fs_err_does_not_panic() {
        log_fs_err(
            "write",
            Path::new("/watt-audit-no-such"),
            &io::Error::new(io::ErrorKind::PermissionDenied, "denied"),
        );
    }
}
