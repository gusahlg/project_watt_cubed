//! Saving and loading worlds, in a compact binary, dependency-free format.
//!
//! A world is procedural, so a save is tiny: the seed regenerates the terrain, and
//! only the player's state, the blocks they've changed, and each mod's own state
//! are stored. Block specs and mod state stay *strings* — the same portable
//! by-name form the network protocol uses — but the container is binary (version
//! 2, little-endian) because the old text format repeated the full spec on every
//! edit line: thousands of mined blocks each spelled out "air" (or a long natural
//! spec). Version 2 stores each distinct spec once in a table and each edit as a
//! fixed 14 bytes referencing it.
//!
//! Layout (all integers little-endian):
//!
//! ```text
//! magic      b"WATT"                                          4 bytes
//! version    u16 = 2
//! seed       i64
//! player     pos f32 x3, yaw f32, pitch f32, flags u8 (bit 0 = fly)
//! spec table u16 count, then per spec: u16 byte-len + utf8 bytes
//! edits      u32 count, then per edit: i32 x, i32 y, i32 z, u16 spec index
//! mods       u8 count, then per mod: u8 name-len + utf8 name,
//!                                    u32 state-len + utf8 state
//! ```
//!
//! Mod state strings are each mod's own `save_state` line, unchanged — mods keep
//! their own formats. Loading never panics on truncated or corrupt input: every
//! read is bounds-checked and counts are sanity-capped, so a bad file surfaces as
//! an `io::Error` the menu can shrug off.
use std::collections::HashMap;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;

use voxel_engine::Vec3;

use crate::block::{AIR, BlockId, Composition};
use crate::mods::Mods;
use crate::player::Player;
use crate::world::World;

const MAGIC: &[u8; 4] = b"WATT";
const SAVE_VERSION: u16 = 2;

/// Sanity caps while reading, so a corrupt length prefix can't balloon memory.
const MAX_SPECS: usize = 4096;
const MAX_EDITS: u32 = 50_000_000;
const MAX_MOD_STATE: u32 = 16 * 1024 * 1024;

/// Directory holding all saves (relative to the working directory).
fn saves_dir() -> PathBuf {
    PathBuf::from("saves")
}

/// The file backing a named save.
fn save_path(name: &str) -> PathBuf {
    saves_dir().join(format!("{name}.save"))
}

/// The names of all existing saves, newest filesystem entries last.
pub fn list_saves() -> Vec<String> {
    let mut names = Vec::new();
    if let Ok(entries) = fs::read_dir(saves_dir()) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("save") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    names.push(stem.to_string());
                }
            }
        }
    }
    names.sort();
    names
}

/// A fresh, unused save name (`world`, then `world-2`, `world-3`, …).
pub fn next_new_name() -> String {
    if !save_path("world").exists() {
        return "world".to_string();
    }
    (2..)
        .map(|n| format!("world-{n}"))
        .find(|name| !save_path(name).exists())
        .expect("an unused save name exists")
}

/// An `InvalidData` error for a malformed save file.
fn corrupt(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// Write a world, player, and mod state to the named save.
pub fn save(name: &str, world: &World, player: &Player, mods: &Mods) -> io::Result<()> {
    fs::create_dir_all(saves_dir())?;
    let mut w = BufWriter::new(fs::File::create(save_path(name))?);

    w.write_all(MAGIC)?;
    w.write_all(&SAVE_VERSION.to_le_bytes())?;
    w.write_all(&world.seed().to_le_bytes())?;

    let p = player.position;
    for v in [p.x, p.y, p.z, player.yaw, player.pitch] {
        w.write_all(&v.to_le_bytes())?;
    }
    w.write_all(&[player.fly as u8])?;

    // Deduplicate specs into a first-seen-order table; edits reference it by index.
    let mut table: Vec<String> = Vec::new();
    let mut index_of: HashMap<String, u16> = HashMap::new();
    let mut edits: Vec<(i32, i32, i32, u16)> = Vec::new();
    for ((x, y, z), id) in world.edits() {
        let spec = block_spec(world, id);
        let index = match index_of.get(&spec) {
            Some(&index) => index,
            None => {
                let index = u16::try_from(table.len())
                    .map_err(|_| corrupt("too many distinct block specs to save"))?;
                index_of.insert(spec.clone(), index);
                table.push(spec);
                index
            }
        };
        edits.push((x, y, z, index));
    }

    w.write_all(&(table.len() as u16).to_le_bytes())?;
    for spec in &table {
        let len = u16::try_from(spec.len()).map_err(|_| corrupt("block spec too long to save"))?;
        w.write_all(&len.to_le_bytes())?;
        w.write_all(spec.as_bytes())?;
    }

    let count = u32::try_from(edits.len()).map_err(|_| corrupt("too many edits to save"))?;
    w.write_all(&count.to_le_bytes())?;
    for (x, y, z, index) in &edits {
        w.write_all(&x.to_le_bytes())?;
        w.write_all(&y.to_le_bytes())?;
        w.write_all(&z.to_le_bytes())?;
        w.write_all(&index.to_le_bytes())?;
    }

    let states = mods.save_states(world);
    let count = u8::try_from(states.len()).map_err(|_| corrupt("too many mod states to save"))?;
    w.write_all(&[count])?;
    for (mod_name, data) in &states {
        let name_len =
            u8::try_from(mod_name.len()).map_err(|_| corrupt("mod name too long to save"))?;
        w.write_all(&[name_len])?;
        w.write_all(mod_name.as_bytes())?;
        let data_len =
            u32::try_from(data.len()).map_err(|_| corrupt("mod state too long to save"))?;
        w.write_all(&data_len.to_le_bytes())?;
        w.write_all(data.as_bytes())?;
    }

    w.flush()
}

/// A bounds-checked cursor over a save's bytes. Every read returns `io::Result`,
/// so truncated or corrupt files fail cleanly instead of panicking.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        if self.bytes.len() - self.pos < n {
            return Err(corrupt("save file is truncated"));
        }
        let slice = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> io::Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn i32(&mut self) -> io::Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn i64(&mut self) -> io::Result<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> io::Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn string(&mut self, len: usize) -> io::Result<String> {
        String::from_utf8(self.take(len)?.to_vec())
            .map_err(|_| corrupt("invalid UTF-8 in save file"))
    }
}

/// Load the named save into a ready-to-play world and player, restoring mod state
/// into `mods`.
pub fn load(name: &str, mods: &mut Mods) -> io::Result<(World, Player)> {
    let bytes = fs::read(save_path(name))?;
    let mut r = Reader::new(&bytes);

    if r.take(MAGIC.len())? != MAGIC {
        return Err(corrupt("not a watt-cubed save (bad magic)"));
    }
    let version = r.u16()?;
    if version != SAVE_VERSION {
        return Err(corrupt(format!(
            "unsupported save version {version} (expected {SAVE_VERSION})"
        )));
    }

    let seed = r.i64()?;
    let mut world = World::new(seed);

    let mut player = Player::new(Vec3::new(r.f32()?, r.f32()?, r.f32()?));
    player.yaw = r.f32()?;
    player.pitch = r.f32()?;
    player.fly = r.u8()? & 1 != 0;

    // Resolve each table spec to a block id once; edits then reuse the ids.
    let spec_count = r.u16()? as usize;
    if spec_count > MAX_SPECS {
        return Err(corrupt("spec table too large"));
    }
    let mut block_ids = Vec::with_capacity(spec_count);
    for _ in 0..spec_count {
        let len = r.u16()? as usize;
        let spec = r.string(len)?;
        block_ids.push(parse_block(&mut world, &spec));
    }

    let edit_count = r.u32()?;
    if edit_count > MAX_EDITS {
        return Err(corrupt("edit count too large"));
    }
    for _ in 0..edit_count {
        let (x, y, z) = (r.i32()?, r.i32()?, r.i32()?);
        let index = r.u16()? as usize;
        let &id = block_ids
            .get(index)
            .ok_or_else(|| corrupt("edit references a spec outside the table"))?;
        world.set_block(x, y, z, id);
    }

    let mod_count = r.u8()?;
    for _ in 0..mod_count {
        let name_len = r.u8()? as usize;
        let mod_name = r.string(name_len)?;
        let data_len = r.u32()?;
        if data_len > MAX_MOD_STATE {
            return Err(corrupt("mod state too large"));
        }
        let data = r.string(data_len as usize)?;
        // Mutable: restoring crafted blocks re-registers them by name.
        mods.load_state(&mod_name, &data, &mut world);
    }

    Ok((world, player))
}

/// Describe a block compactly by composition, using portable element names. Shared
/// with the network layer, which sends edits in exactly this portable form so ids
/// never have to agree between machines.
pub(crate) fn block_spec(world: &World, id: BlockId) -> String {
    if id == AIR {
        return "air".to_string();
    }
    let elements = world.registry().elements();
    match &world.registry().block(id).composition {
        Composition::Natural(els) if els.is_empty() => "air".to_string(),
        Composition::Natural(els) => {
            let names: Vec<String> = els.iter().map(|&e| elements.get(e).name.to_string()).collect();
            format!("natural:{}", names.join(","))
        }
        Composition::Mixture(mix) | Composition::Configuration { mix, .. } => {
            let parts: Vec<String> = mix
                .0
                .iter()
                .map(|&(e, pct)| format!("{}={}", elements.get(e).name, pct))
                .collect();
            format!("mixture:{}", parts.join(";"))
        }
        Composition::Computational(_) => "air".to_string(), // not yet reconstructable
    }
}

/// Rebuild a block from a spec, registering it into the world's palette as needed.
/// The inverse of [`block_spec`]; shared with the network layer.
pub(crate) fn parse_block(world: &mut World, spec: &str) -> BlockId {
    if spec == "air" {
        return AIR;
    }
    if let Some(rest) = spec.strip_prefix("natural:") {
        // Strict: ANY unknown element rejects the whole spec — registering a
        // subset would mint a different block than the sender meant. Palette
        // growth from remote specs is capped inside craft_natural.
        let mut ids = Vec::new();
        for name in rest.split(',') {
            match world.registry().elements().id_by_name(name) {
                Some(id) => ids.push(id),
                None => return AIR,
            }
        }
        return crate::block::crafting::craft_natural(world.registry_mut(), &ids)
            .unwrap_or(AIR);
    }
    if let Some(rest) = spec.strip_prefix("mixture:") {
        let mut parts = Vec::new();
        for entry in rest.split(';') {
            let Some((name, pct)) = entry.split_once('=') else { return AIR };
            let Some(id) = world.registry().elements().id_by_name(name) else { return AIR };
            let Ok(pct) = pct.parse::<u8>() else { return AIR };
            parts.push((id, pct));
        }
        let Ok(composition) = crate::block::Composition::mixture(&parts) else {
            return AIR;
        };
        if let Some(existing) = world.registry().lookup(&composition) {
            return existing;
        }
        if world.registry().at_capacity() {
            return AIR;
        }
        return world.registry_mut().mixture(&parts).unwrap_or(AIR);
    }
    AIR
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::element::El;

    #[test]
    fn round_trip_preserves_seed_player_edits_and_mod_state() {
        let name = "__unit_test_round_trip__";
        let _ = fs::remove_file(save_path(name));

        // Build a world, move the player, and break the surface block at a column.
        let mut world = World::new(4242);
        let (bx, bz) = (8, 8);
        let by = (0..64)
            .rev()
            .find(|&y| world.is_solid(bx, y, bz))
            .unwrap();
        world.set_block(bx, by, bz, AIR);

        let mut player = Player::new(Vec3::new(1.0, 2.0, 3.0));
        player.yaw = 0.5;
        player.pitch = -0.25;
        player.fly = true;

        // Give the mods some state to persist (elements land in the inventory).
        let mut mods = Mods::with_defaults();
        mods.on_block_break(&[El::Stone.id(), El::Iron.id(), El::Stone.id()], &world);
        let states_before = mods.save_states(&world);

        save(name, &world, &player, &mods).unwrap();

        let mut fresh_mods = Mods::with_defaults();
        let (loaded_world, loaded_player) = load(name, &mut fresh_mods).unwrap();

        assert_eq!(loaded_world.seed(), 4242);
        assert_eq!(loaded_player.position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(loaded_player.yaw, 0.5);
        assert_eq!(loaded_player.pitch, -0.25);
        assert!(loaded_player.fly);
        assert_eq!(loaded_world.block_at(bx, by, bz), AIR, "broken block stays broken");
        assert_eq!(
            fresh_mods.save_states(&loaded_world),
            states_before,
            "mod state survives the round trip"
        );

        let _ = fs::remove_file(save_path(name));
    }

    #[test]
    fn spec_table_stores_a_repeated_spec_once() {
        let name = "__unit_test_dedup__";
        let _ = fs::remove_file(save_path(name));

        // 500 edits, all the same spec ("air"): the old text format repeated the
        // spec per line; v2 stores it once and each edit is a fixed 14 bytes.
        let mut world = World::new(7);
        for i in 0..500 {
            world.set_block(i, 200, -i, AIR);
        }
        let player = Player::new(Vec3::ZERO);
        let mods = Mods::with_defaults();
        save(name, &world, &player, &mods).unwrap();

        let size = fs::metadata(save_path(name)).unwrap().len();
        assert!(
            size < 500 * 15 + 256,
            "spec must be stored once, not per edit (file is {size} bytes)"
        );

        let _ = fs::remove_file(save_path(name));
    }

    #[test]
    fn corrupt_magic_is_an_error() {
        let name = "__unit_test_bad_magic__";
        fs::create_dir_all(saves_dir()).unwrap();
        fs::write(save_path(name), b"NOPE this is not a watt-cubed save").unwrap();

        let mut mods = Mods::with_defaults();
        assert!(load(name, &mut mods).is_err());

        let _ = fs::remove_file(save_path(name));
    }

    #[test]
    fn truncated_file_is_an_error_not_a_panic() {
        let name = "__unit_test_truncated__";
        let _ = fs::remove_file(save_path(name));

        let mut world = World::new(99);
        for i in 0..20 {
            world.set_block(i, 200, i, AIR);
        }
        let player = Player::new(Vec3::ZERO);
        let mut mods = Mods::with_defaults();
        save(name, &world, &player, &mods).unwrap();

        let bytes = fs::read(save_path(name)).unwrap();
        fs::write(save_path(name), &bytes[..bytes.len() / 2]).unwrap();

        assert!(load(name, &mut mods).is_err());

        let _ = fs::remove_file(save_path(name));
    }

    #[test]
    fn empty_world_round_trips() {
        let name = "__unit_test_empty__";
        let _ = fs::remove_file(save_path(name));

        let world = World::new(1234);
        let player = Player::new(Vec3::new(0.0, 40.0, 0.0));
        let mut mods = Mods::with_defaults();
        save(name, &world, &player, &mods).unwrap();

        let (loaded_world, loaded_player) = load(name, &mut mods).unwrap();
        assert_eq!(loaded_world.seed(), 1234);
        assert_eq!(loaded_player.position, Vec3::new(0.0, 40.0, 0.0));
        assert_eq!(loaded_world.edits().count(), 0);

        let _ = fs::remove_file(save_path(name));
    }
}
