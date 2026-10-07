//! The dedicated server's world file: seed, generator, edit ledger, and clock.
//!
//! The bytes are a [`save::format`](crate::save::format) document. Edits use the
//! spec palette. The clock is the mod record `pwc.clock` (eight hex digits of
//! `f32::to_bits`); singleplayer ignores that name. Pending reactions are the live
//! scheduler's contacts, so a reload continues work in progress. A spec this build
//! cannot parse stays in the file until a later edit replaces that cell.
//!
//! A missing file starts from the caller's flags. A corrupt file, a law mismatch,
//! or a world from before this universe is an error — the server must not replace
//! it with a fresh world. When a file loads, its seed and generator win.

use std::fs;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::save::format::{self, Decoded, Edit, PendingContact, PlayerState, SaveDoc, WorldgenStamp};
use crate::save::slot::SaveMeta;
use crate::save;
use crate::world::generation::WorldgenKind;
use crate::world::terrain::{TerrainCfg, WORLDGEN_VERSION};

/// Mod record that holds the day fraction. Not a gameplay mod.
const CLOCK_MOD: &str = "pwc.clock";
/// Day fraction when a world has no clock record.
const DEFAULT_DAY: f32 = 0.3;

/// What the process asked for. A stored world replaces seed and generator.
pub(crate) struct Flags {
    pub seed: i64,
    pub worldgen: WorldgenKind,
    pub terrain: TerrainCfg,
    /// Log when the file's seed or generator differs from these flags.
    pub warn: bool,
}

/// A world ready to host, plus the skeleton written back on the next save.
pub(crate) struct Loaded {
    pub seed: i64,
    pub worldgen: WorldgenKind,
    pub terrain: TerrainCfg,
    pub day: f32,
    pub edits: Vec<(i32, i32, i32, String)>,
    /// Contacts the file still had waiting. Restored into the scheduler before the first tick.
    pub pending: Vec<PendingContact>,
    pub store: Option<Store>,
}

/// One world file and the player/mod skeleton kept across saves.
pub(crate) struct Store {
    path: PathBuf,
    doc: Mutex<SaveDoc>,
    /// Cells whose spec this build could not parse. Re-emitted on the next save
    /// unless the live ledger has since written that cell.
    kept: Mutex<Vec<(i32, i32, i32, String)>>,
}

/// Why a world file cannot be served. Displays as `{path}: {reason}`.
#[derive(Debug)]
pub(crate) struct LoadError {
    path: PathBuf,
    reason: String,
}

impl LoadError {
    fn new(path: &Path, reason: impl Into<String>) -> Self {
        Self { path: path.to_path_buf(), reason: reason.into() }
    }

    /// The reason alone, without the path.
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.path.display(), self.reason)
    }
}

impl std::error::Error for LoadError {}

/// Live ledger captured for one save. Specs are the server's shared `Arc`s, so capturing under the
/// state lock copies no strings.
pub(crate) struct Snapshot {
    pub seed: i64,
    pub worldgen: WorldgenKind,
    pub terrain: TerrainCfg,
    pub day: f32,
    pub edits: Vec<(i32, i32, i32, Arc<str>)>,
    /// Active reaction contacts, in processing order.
    pub pending: Vec<PendingContact>,
}

/// Operators and mod policy that live beside a world file.
pub(crate) struct SideFiles {
    pub ops: Vec<String>,
    pub allow: Vec<String>,
    pub deny: Vec<String>,
}

impl Store {
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Cells skipped at load because their spec does not parse. Set before the
    /// server threads start.
    pub fn set_kept(&self, cells: Vec<(i32, i32, i32, String)>) {
        *self.kept.lock().unwrap_or_else(|p| p.into_inner()) = cells;
    }

    /// Replace the ledger, clock, generator stamp, and pending reactions.
    /// Player, other mod records, name, created time, and playtime stay.
    /// The previous file is rotated to `.bak` the way a singleplayer slot is.
    pub fn write(&self, snap: &Snapshot) -> io::Result<()> {
        let mut doc = self.doc.lock().unwrap_or_else(|p| p.into_inner());
        let mut kept = self.kept.lock().unwrap_or_else(|p| p.into_inner());
        let bytes = encode_snapshot(&mut doc, snap, &mut kept).map_err(|e| io::Error::other(e))?;
        write_rotating(&self.path, &bytes)
    }
}

/// `{path}.bak` / `{path}.tmp`, matching [`save::store`]'s slot names (`id.save.bak`).
fn suffixed(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(suffix);
    path.with_file_name(name)
}

/// Write `bytes` via a sibling `.tmp`, then rename the live file to `.bak` and
/// the temp file into place. A failed second rename puts the backup back.
fn write_rotating(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            fs::create_dir_all(dir)?;
        }
    }
    let tmp = suffixed(path, ".tmp");
    let bak = suffixed(path, ".bak");
    let wrote = (|| {
        let mut file = fs::File::create(&tmp)?;
        std::io::Write::write_all(&mut file, bytes)?;
        file.sync_all()?;
        Ok(())
    })();
    if let Err(err) = wrote {
        let _ = fs::remove_file(&tmp);
        return Err(err);
    }
    let had_live = path.exists();
    if had_live && let Err(err) = fs::rename(path, &bak) {
        let _ = fs::remove_file(&tmp);
        return Err(err);
    }
    if let Err(err) = fs::rename(&tmp, path) {
        if had_live {
            let _ = fs::rename(&bak, path);
        }
        let _ = fs::remove_file(&tmp);
        return Err(err);
    }
    Ok(())
}

/// No file: the flags are the world, and nothing is saved.
pub(crate) fn fresh(flags: &Flags) -> Loaded {
    Loaded {
        seed: flags.seed,
        worldgen: flags.worldgen,
        terrain: flags.terrain,
        day: DEFAULT_DAY,
        edits: Vec::new(),
        pending: Vec::new(),
        store: None,
    }
}

/// Load `path`, or prepare a new file there when it does not exist yet.
pub(crate) fn load(path: &Path, flags: &Flags) -> Result<Loaded, LoadError> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == ErrorKind::NotFound => {
            return Ok(Loaded {
                seed: flags.seed,
                worldgen: flags.worldgen,
                terrain: flags.terrain,
                day: DEFAULT_DAY,
                edits: Vec::new(),
                pending: Vec::new(),
                store: Some(Store {
                    path: path.to_path_buf(),
                    doc: Mutex::new(blank_doc(flags)),
                    kept: Mutex::new(Vec::new()),
                }),
            });
        }
        Err(e) => return Err(LoadError::new(path, e.to_string())),
    };
    let mut doc = match format::decode(&bytes).map_err(|e| LoadError::new(path, e.to_string()))? {
        Decoded::Intact(doc) => doc,
        Decoded::Salvaged { doc, recovered, expected } => {
            eprintln!(
                "warning: {} was truncated; loaded {recovered} of {expected} edits",
                path.display()
            );
            doc
        }
    };
    if doc.law_stamp != material::Law::current().stamp() {
        return Err(LoadError::new(path, "save belongs to a different universe (law stamp mismatch)"));
    }
    if doc.worldgen_version != WORLDGEN_VERSION {
        eprintln!(
            "warning: {} was written by worldgen v{} (current v{WORLDGEN_VERSION})",
            path.display(),
            doc.worldgen_version
        );
    }
    let worldgen = WorldgenKind::from_wire(doc.worldgen.kind)
        .ok_or_else(|| LoadError::new(path, format!("unknown worldgen kind {}", doc.worldgen.kind)))?;
    let terrain = TerrainCfg::from_wire(doc.worldgen.knobs);
    if flags.warn && overrides(flags, doc.meta.seed, worldgen, terrain) {
        eprintln!(
            "warning: {} stores seed {} worldgen {} — those win over the flags (seed {} worldgen {})",
            path.display(),
            doc.meta.seed,
            worldgen.id(),
            flags.seed,
            flags.worldgen.id()
        );
    }
    let day = clock_of(&doc.mods);
    let spec_table = std::mem::take(&mut doc.specs);
    let stored = std::mem::take(&mut doc.edits);
    let pending = std::mem::take(&mut doc.pending);
    let mut edits = Vec::with_capacity(stored.len());
    for edit in stored {
        let Some(spec) = spec_table.get(usize::from(edit.spec)) else { continue };
        edits.push((edit.x, edit.y, edit.z, spec.clone()));
    }
    Ok(Loaded {
        seed: doc.meta.seed,
        worldgen,
        terrain,
        day,
        edits,
        pending,
        store: Some(Store {
            path: path.to_path_buf(),
            doc: Mutex::new(doc),
            kept: Mutex::new(Vec::new()),
        }),
    })
}

fn overrides(flags: &Flags, seed: i64, kind: WorldgenKind, terrain: TerrainCfg) -> bool {
    if seed != flags.seed || kind != flags.worldgen {
        return true;
    }
    kind == WorldgenKind::Diffusion && terrain != flags.terrain
}

fn blank_doc(flags: &Flags) -> SaveDoc {
    let now = save::unix_now();
    SaveDoc {
        meta: SaveMeta {
            name: "world".to_string(),
            seed: flags.seed,
            created: now,
            last_played: now,
            playtime_secs: 0,
            edit_count: 0,
        },
        worldgen_version: WORLDGEN_VERSION,
        worldgen: WorldgenStamp { kind: flags.worldgen.wire(), knobs: flags.terrain.to_wire() },
        law_stamp: material::Law::current().stamp(),
        player: PlayerState {
            pos: [0.0, 64.0, 0.0],
            yaw: 0.0,
            pitch: 0.0,
            frame: glam::DQuat::IDENTITY,
            velocity: [0.0; 3],
            up: 5,
            legacy_pose: false,
            flying: false,
            noclip: false,
            inventory: Some(Vec::new()),
        },
        specs: Vec::new(),
        edits: Vec::new(),
        mods: Vec::new(),
        pending: Vec::new(),
    }
}

fn spec_index(specs: &mut Vec<String>, spec: &str) -> Result<u16, String> {
    if let Some(index) = specs.iter().position(|have| have == spec) {
        return u16::try_from(index).map_err(|_| "too many distinct block specs to save".to_string());
    }
    let index = u16::try_from(specs.len()).map_err(|_| "too many distinct block specs to save".to_string())?;
    specs.push(spec.to_string());
    Ok(index)
}

fn encode_snapshot(doc: &mut SaveDoc, snap: &Snapshot, kept: &mut Vec<(i32, i32, i32, String)>) -> Result<Vec<u8>, String> {
    let mut cells: Vec<(i32, i32, i32, &str)> = snap.edits.iter().map(|(x, y, z, spec)| (*x, *y, *z, spec.as_ref())).collect();
    // A live edit replaces an unparsed one at the same cell. The unparsed text stays only while
    // this build still has no opinion about that cell.
    kept.retain(|(x, y, z, _)| !snap.edits.iter().any(|edit| edit.0 == *x && edit.1 == *y && edit.2 == *z));
    for (x, y, z, spec) in kept.iter() {
        cells.push((*x, *y, *z, spec.as_str()));
    }
    cells.sort_unstable_by_key(|cell| (cell.0, cell.1, cell.2));
    let mut specs: Vec<String> = Vec::new();
    let mut records = Vec::with_capacity(cells.len());
    for (x, y, z, spec) in cells {
        records.push(Edit { x, y, z, spec: spec_index(&mut specs, spec)? });
    }
    doc.meta.seed = snap.seed;
    doc.meta.last_played = save::unix_now();
    doc.meta.edit_count = u32::try_from(records.len()).unwrap_or(u32::MAX);
    doc.worldgen = WorldgenStamp { kind: snap.worldgen.wire(), knobs: snap.terrain.to_wire() };
    doc.law_stamp = material::Law::current().stamp();
    doc.specs = specs;
    doc.edits = records;
    doc.pending = snap.pending.clone();
    doc.mods.retain(|(name, _)| name != CLOCK_MOD);
    let day = if snap.day.is_finite() { snap.day.rem_euclid(1.0) } else { DEFAULT_DAY };
    doc.mods.push((CLOCK_MOD.to_string(), format!("{:08x}", day.to_bits())));
    format::encode(doc).map_err(|e| e.to_string())
}

fn clock_of(mods: &[(String, String)]) -> f32 {
    let Some((_, text)) = mods.iter().find(|(name, _)| name == CLOCK_MOD) else {
        return DEFAULT_DAY;
    };
    let Ok(bits) = u32::from_str_radix(text, 16) else { return DEFAULT_DAY };
    let day = f32::from_bits(bits);
    if day.is_finite() { day.rem_euclid(1.0) } else { DEFAULT_DAY }
}

/// `ops.txt` (one name a line, `#` comments) and `mods.toml` (`allow` / `deny`
/// string arrays) in the world's directory. A missing file is empty. A
/// `mods.toml` that is not that shape is an error.
pub(crate) fn read_side_files(world: &Path) -> Result<SideFiles, String> {
    let Some(dir) = world.parent() else {
        return Ok(SideFiles { ops: Vec::new(), allow: Vec::new(), deny: Vec::new() });
    };
    let ops = read_ops(&dir.join("ops.txt"))?;
    let (allow, deny) = read_mods_toml(&dir.join("mods.toml"))?;
    Ok(SideFiles { ops, allow, deny })
}

fn read_ops(path: &Path) -> Result<Vec<String>, String> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("could not read {}: {e}", path.display())),
    };
    Ok(parse_ops(&text))
}

pub(crate) fn parse_ops(text: &str) -> Vec<String> {
    let mut ops = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let name: String = line.chars().filter(|c| !c.is_control()).take(super::MAX_NAME).collect();
        let name = name.trim().to_ascii_lowercase();
        if name.is_empty() || ops.iter().any(|op: &String| op == &name) {
            continue;
        }
        ops.push(name);
    }
    ops
}

fn read_mods_toml(path: &Path) -> Result<(Vec<String>, Vec<String>), String> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok((Vec::new(), Vec::new())),
        Err(e) => return Err(format!("could not read {}: {e}", path.display())),
    };
    parse_mods_toml(&text).map_err(|e| format!("{}: {e}", path.display()))
}

pub(crate) fn parse_mods_toml(text: &str) -> Result<(Vec<String>, Vec<String>), String> {
    let value: toml::Value = toml::from_str(text).map_err(|e| e.to_string())?;
    let Some(table) = value.as_table() else {
        return Err("mods.toml must be a table".into());
    };
    Ok((mod_ids(table, "allow")?, mod_ids(table, "deny")?))
}

fn mod_ids(table: &toml::map::Map<String, toml::Value>, key: &str) -> Result<Vec<String>, String> {
    let Some(value) = table.get(key) else { return Ok(Vec::new()) };
    let Some(items) = value.as_array() else {
        return Err(format!("{key} must be an array of mod ids"));
    };
    let mut ids = Vec::with_capacity(items.len());
    for item in items {
        let Some(id) = item.as_str() else {
            return Err(format!("{key} must be an array of mod ids"));
        };
        let id = id.trim();
        if !id.is_empty() && !ids.iter().any(|have: &String| have == id) {
            ids.push(id.to_string());
        }
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::save::format::{Edit, PendingContact, SaveDoc};
    use crate::save::write_atomic_file;

    fn flags(seed: i64) -> Flags {
        Flags {
            seed,
            worldgen: WorldgenKind::Flat,
            terrain: TerrainCfg::default(),
            warn: true,
        }
    }

    fn doc_with(seed: i64, day_bits: u32, edits: Vec<Edit>, specs: Vec<String>) -> SaveDoc {
        let mut doc = blank_doc(&flags(seed));
        doc.meta.edit_count = edits.len() as u32;
        doc.specs = specs;
        doc.edits = edits;
        doc.mods = vec![
            ("inventory".into(), "Stone".into()),
            (CLOCK_MOD.into(), format!("{day_bits:08x}")),
        ];
        doc.pending = vec![PendingContact { x: 1, y: 2, z: 3, axis: 0, age: 4 }];
        doc
    }

    #[test]
    fn world_round_trip_keeps_edits_and_clock() {
        let path = crate::save::store::test_temp_path("round");
        let day = 0.75f32;
        let doc = doc_with(
            42,
            day.to_bits(),
            vec![Edit { x: 3, y: 1, z: 2, spec: 0 }, Edit { x: 1, y: 2, z: 3, spec: 1 }],
            vec!["air".into(), "natural:Stone".into()],
        );
        write_atomic_file(&path, &format::encode(&doc).unwrap()).unwrap();
        let loaded = load(&path, &flags(1)).unwrap();
        assert_eq!(loaded.seed, 42);
        assert_eq!(loaded.worldgen, WorldgenKind::Flat);
        assert_eq!(loaded.day.to_bits(), day.to_bits());
        assert_eq!(
            loaded.edits,
            vec![(3, 1, 2, "air".into()), (1, 2, 3, "natural:Stone".into())]
        );
        let store = loaded.store.unwrap();
        let pending = loaded.pending.clone();
        assert_eq!(pending, vec![PendingContact { x: 1, y: 2, z: 3, axis: 0, age: 4 }]);
        store
            .write(&Snapshot {
                seed: 42,
                worldgen: WorldgenKind::Flat,
                terrain: TerrainCfg::default(),
                day,
                edits: vec![(1, 2, 3, "natural:Stone".into()), (3, 1, 2, "air".into())],
                pending: pending.clone(),
            })
            .unwrap();
        let again = load(&path, &flags(99)).unwrap();
        assert_eq!(again.seed, 42);
        assert_eq!(again.day.to_bits(), day.to_bits());
        assert_eq!(again.edits, vec![(1, 2, 3, "natural:Stone".into()), (3, 1, 2, "air".into())]);
        assert_eq!(again.pending, pending, "a server save keeps the scheduler's contacts");
        let saved = match format::decode(&fs::read(&path).unwrap()).unwrap() {
            Decoded::Intact(doc) => doc,
            Decoded::Salvaged { .. } => panic!("resave must be intact"),
        };
        assert_eq!(saved.pending, pending);
        assert!(saved.mods.iter().any(|(n, d)| n == "inventory" && d == "Stone"));
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(suffixed(&path, ".bak"));
    }

    #[test]
    fn v9_flat_world_loads_its_edit() {
        let path = crate::save::store::test_temp_path("v9");
        write_atomic_file(&path, &v9_with_edit()).unwrap();
        let loaded = load(&path, &flags(7)).unwrap();
        assert_eq!(loaded.seed, 5);
        assert_eq!(loaded.worldgen, WorldgenKind::Flat);
        assert_eq!(loaded.day, DEFAULT_DAY, "a v9 file has no clock");
        assert_eq!(loaded.edits, vec![(4, 5, 6, "air".into())]);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn corrupt_world_is_refused() {
        let path = crate::save::store::test_temp_path("bad");
        write_atomic_file(&path, b"not a save").unwrap();
        assert!(load(&path, &flags(1)).is_err());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn missing_world_uses_the_flags() {
        let path = crate::save::store::test_temp_path("missing");
        let loaded = load(&path, &flags(11)).unwrap();
        assert_eq!(loaded.seed, 11);
        assert!(loaded.edits.is_empty());
        assert!(!path.exists());
    }

    #[test]
    fn side_files_union_ops_and_mod_lists() {
        let dir = crate::save::store::test_temp_path("side");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("ops.txt"), "# staff\nAda\n\n bob \nAda\n").unwrap();
        fs::write(
            dir.join("mods.toml"),
            "allow = [\"pwc.hotbar\", \"pwc.hotbar\"]\ndeny = [\"pwc.dev-toolkit\"]\n",
        )
        .unwrap();
        let side = read_side_files(&dir.join("world.save")).unwrap();
        assert_eq!(side.ops, vec!["ada".to_string(), "bob".to_string()]);
        assert_eq!(side.allow, vec!["pwc.hotbar".to_string()]);
        assert_eq!(side.deny, vec!["pwc.dev-toolkit".to_string()]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn host_save_rotates_a_slot_backup() {
        let id = crate::save::SlotId::new("__pwc_g25_rotate__").unwrap();
        let path = crate::save::store::file_path(&id);
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(suffixed(&path, ".bak"));
        let loaded = load(&path, &flags(1)).unwrap();
        let store = loaded.store.unwrap();
        let pending = vec![PendingContact { x: 4, y: 5, z: 6, axis: 1, age: 2 }];
        store
            .write(&Snapshot {
                seed: 1,
                worldgen: WorldgenKind::Flat,
                terrain: TerrainCfg::default(),
                day: 0.25,
                edits: vec![(1, 2, 3, "air".into())],
                pending: pending.clone(),
            })
            .unwrap();
        store
            .write(&Snapshot {
                seed: 2,
                worldgen: WorldgenKind::Flat,
                terrain: TerrainCfg::default(),
                day: 0.5,
                edits: vec![(9, 9, 9, "air".into())],
                pending: Vec::new(),
            })
            .unwrap();
        fs::write(&path, b"not a save").unwrap();
        let (decoded, source) = crate::save::store::read(&id).unwrap();
        assert_eq!(source, crate::save::store::Source::Backup);
        let doc = match decoded {
            Decoded::Intact(doc) => doc,
            Decoded::Salvaged { .. } => panic!("the backup must be intact"),
        };
        assert_eq!(doc.meta.seed, 1);
        assert_eq!(doc.pending, pending);
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(suffixed(&path, ".bak"));
    }

    #[test]
    fn unparsed_specs_survive_the_next_save() {
        let path = crate::save::store::test_temp_path("kept");
        let doc = doc_with(
            3,
            0.3f32.to_bits(),
            vec![Edit { x: 1, y: 2, z: 3, spec: 0 }, Edit { x: 4, y: 5, z: 6, spec: 1 }],
            vec!["natural:Stone".into(), "air".into()],
        );
        write_atomic_file(&path, &format::encode(&doc).unwrap()).unwrap();
        let loaded = load(&path, &flags(1)).unwrap();
        let mut registry = crate::block::BlockRegistry::with_builtins();
        let mut kept = Vec::new();
        let mut parsed = Vec::new();
        for (x, y, z, spec) in &loaded.edits {
            if registry.parse_spec(spec).is_none() {
                kept.push((*x, *y, *z, spec.clone()));
            } else {
                parsed.push((*x, *y, *z, Arc::from(spec.clone())));
            }
        }
        assert_eq!(kept, vec![(1, 2, 3, "natural:Stone".to_string())]);
        let store = loaded.store.unwrap();
        store.set_kept(kept);
        store
            .write(&Snapshot {
                seed: 3,
                worldgen: WorldgenKind::Flat,
                terrain: TerrainCfg::default(),
                day: 0.3,
                edits: parsed,
                pending: Vec::new(),
            })
            .unwrap();
        let again = load(&path, &flags(1)).unwrap();
        assert!(again.edits.iter().any(|edit| edit.3 == "natural:Stone"));
        assert!(again.edits.iter().any(|edit| edit.3 == "air"));
        let store = again.store.unwrap();
        store.set_kept(vec![(1, 2, 3, "natural:Stone".into())]);
        store
            .write(&Snapshot {
                seed: 3,
                worldgen: WorldgenKind::Flat,
                terrain: TerrainCfg::default(),
                day: 0.3,
                edits: vec![(1, 2, 3, "air".into()), (4, 5, 6, "air".into())],
                pending: Vec::new(),
            })
            .unwrap();
        let replaced = load(&path, &flags(1)).unwrap();
        assert!(replaced.edits.iter().all(|edit| edit.3 == "air"));
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(suffixed(&path, ".bak"));
    }

    /// A minimal v9 flat file: four knobs, the old 33-byte player, one `air` edit.
    fn v9_with_edit() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(format::MAGIC);
        out.extend_from_slice(&9u16.to_le_bytes());
        out.push(1);
        let mut field = [0u8; 64];
        field[0] = b'v';
        out.extend_from_slice(&field);
        out.extend_from_slice(&5i64.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&6u16.to_le_bytes());
        out.push(0);
        for k in [100u16, 100, 100, 100] {
            out.extend_from_slice(&k.to_le_bytes());
        }
        out.extend_from_slice(&material::Law::current().stamp());
        out.extend_from_slice(&[0u8; 33]);
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&3u16.to_le_bytes());
        out.extend_from_slice(b"air");
        out.extend_from_slice(&4i32.to_le_bytes());
        out.extend_from_slice(&5i32.to_le_bytes());
        out.extend_from_slice(&6i32.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.push(0);
        out.extend_from_slice(&0u32.to_le_bytes());
        out
    }
}
