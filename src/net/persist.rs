//! The dedicated server's world file: seed, generator, edit ledger, and clock.
//!
//! The bytes are a [`save::format`](crate::save::format) document. Edits use the
//! spec palette. The clock is the mod record `pwc.clock` (eight hex digits of
//! `f32::to_bits`); singleplayer ignores that name. Pending reactions are written
//! empty: the server has no way to export a live scheduler, and replaying a stale
//! list would apply those contacts twice.
//!
//! A missing file starts from the caller's flags. A corrupt file, a law mismatch,
//! or a world from before this universe is an error — the server must not replace
//! it with a fresh world. When a file loads, its seed and generator win.

use std::fs;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::save::format::{self, Decoded, Edit, PlayerState, SaveDoc, WorldgenStamp};
use crate::save::slot::SaveMeta;
use crate::save::{self, write_atomic_file};
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
    pub store: Option<Store>,
}

/// One world file and the player/mod skeleton kept across saves.
pub(crate) struct Store {
    path: PathBuf,
    doc: Mutex<SaveDoc>,
}

/// Live ledger captured for one save.
pub(crate) struct Snapshot {
    pub seed: i64,
    pub worldgen: WorldgenKind,
    pub terrain: TerrainCfg,
    pub day: f32,
    pub edits: Vec<(i32, i32, i32, String)>,
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

    /// Replace the ledger, clock, and generator stamp. Player, other mod
    /// records, name, created time, and playtime stay.
    pub fn write(&self, snap: &Snapshot) -> io::Result<()> {
        let mut doc = self.doc.lock().unwrap_or_else(|p| p.into_inner());
        let bytes = encode_snapshot(&mut doc, snap).map_err(|e| io::Error::other(e))?;
        write_atomic_file(&self.path, &bytes)
    }
}

/// No file: the flags are the world, and nothing is saved.
pub(crate) fn fresh(flags: &Flags) -> Loaded {
    Loaded {
        seed: flags.seed,
        worldgen: flags.worldgen,
        terrain: flags.terrain,
        day: DEFAULT_DAY,
        edits: Vec::new(),
        store: None,
    }
}

/// Load `path`, or prepare a new file there when it does not exist yet.
pub(crate) fn load(path: &Path, flags: &Flags) -> Result<Loaded, String> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == ErrorKind::NotFound => {
            return Ok(Loaded {
                seed: flags.seed,
                worldgen: flags.worldgen,
                terrain: flags.terrain,
                day: DEFAULT_DAY,
                edits: Vec::new(),
                store: Some(Store {
                    path: path.to_path_buf(),
                    doc: Mutex::new(blank_doc(flags)),
                }),
            });
        }
        Err(e) => return Err(format!("could not read {}: {e}", path.display())),
    };
    let mut doc = match format::decode(&bytes).map_err(|e| format!("{}: {e}", path.display()))? {
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
        return Err(format!(
            "{}: save belongs to a different universe (law stamp mismatch)",
            path.display()
        ));
    }
    if doc.worldgen_version != WORLDGEN_VERSION {
        eprintln!(
            "warning: {} was written by worldgen v{} (current v{WORLDGEN_VERSION})",
            path.display(),
            doc.worldgen_version
        );
    }
    let worldgen = WorldgenKind::from_wire(doc.worldgen.kind)
        .ok_or_else(|| format!("{}: unknown worldgen kind {}", path.display(), doc.worldgen.kind))?;
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
    doc.pending.clear();
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
        store: Some(Store { path: path.to_path_buf(), doc: Mutex::new(doc) }),
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

fn encode_snapshot(doc: &mut SaveDoc, snap: &Snapshot) -> Result<Vec<u8>, String> {
    let mut edits: Vec<&(i32, i32, i32, String)> = snap.edits.iter().collect();
    edits.sort_by_key(|e| (e.0, e.1, e.2));
    let mut specs: Vec<String> = Vec::new();
    let mut records = Vec::with_capacity(edits.len());
    for (x, y, z, spec) in edits {
        let index = match specs.iter().position(|s| s == spec) {
            Some(i) => i,
            None => {
                specs.push(spec.clone());
                specs.len() - 1
            }
        };
        let Ok(spec_index) = u16::try_from(index) else {
            return Err("too many distinct block specs to save".into());
        };
        records.push(Edit { x: *x, y: *y, z: *z, spec: spec_index });
    }
    doc.meta.seed = snap.seed;
    doc.meta.last_played = save::unix_now();
    doc.meta.edit_count = u32::try_from(records.len()).unwrap_or(u32::MAX);
    doc.worldgen = WorldgenStamp { kind: snap.worldgen.wire(), knobs: snap.terrain.to_wire() };
    doc.law_stamp = material::Law::current().stamp();
    doc.specs = specs;
    doc.edits = records;
    doc.pending.clear();
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
        store
            .write(&Snapshot {
                seed: 42,
                worldgen: WorldgenKind::Flat,
                terrain: TerrainCfg::default(),
                day,
                edits: vec![(1, 2, 3, "natural:Stone".into()), (3, 1, 2, "air".into())],
            })
            .unwrap();
        let again = load(&path, &flags(99)).unwrap();
        assert_eq!(again.seed, 42);
        assert_eq!(again.day.to_bits(), day.to_bits());
        assert_eq!(again.edits, vec![(1, 2, 3, "natural:Stone".into()), (3, 1, 2, "air".into())]);
        let saved = match format::decode(&fs::read(&path).unwrap()).unwrap() {
            Decoded::Intact(doc) => doc,
            Decoded::Salvaged { .. } => panic!("resave must be intact"),
        };
        assert!(saved.pending.is_empty(), "a server save does not keep stale reactions");
        assert!(saved.mods.iter().any(|(n, d)| n == "inventory" && d == "Stone"));
        let _ = fs::remove_file(&path);
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
