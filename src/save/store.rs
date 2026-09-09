//! Filesystem layer: atomic writes, a one-deep backup, and the read ladder.
//! Writes use .tmp + sync_all + rename so a crash mid-save can't corrupt the
//! only copy; successful writes rotate the old file to .bak. Reads ladder down
//! (live intact → backup intact → salvage) and report which rung succeeded.
//! Deletes move to saves/trash/ instead of unlinking for cheap undo.

use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;

use super::format::{self, Decoded};
use super::slot::{SaveError, Slot, SlotId};

fn saves_dir() -> PathBuf {
    PathBuf::from("saves")
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
    let mut f = fs::File::open(path)?;
    let mut buf = [0u8; format::HEADER_LEN];
    f.read_exact(&mut buf)?;
    format::peek_meta(&buf)
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

/// Atomically replace a slot's bytes, rotating the previous file to `.bak`.
pub fn write(id: &SlotId, bytes: &[u8]) -> io::Result<()> {
    fs::create_dir_all(saves_dir())?;
    let tmp = tmp_path(id);
    let live = live_path(id);
    {
        let mut f = fs::File::create(&tmp)?;
        io::Write::write_all(&mut f, bytes)?;
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
    let _ = fs::rename(bak_path(from), bak_path(to));
    fs::remove_file(live_path(from))?;
    Ok(())
}

/// Move a slot (and its backup) into `saves/trash/` rather than unlinking.
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
}
