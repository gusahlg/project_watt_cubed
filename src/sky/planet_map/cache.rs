//! The bake's disk cache: one file per face, keyed on everything that changes the image, with a
//! payload checksum in the header. Only the newest keys of this bake version are kept.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use voxel_engine::Color;

use super::BAKE_VERSION;
use crate::hash::Fnv64;
use crate::ident::codec::{CodecError, Reader, Writer};

/// How many distinct bake keys the disk cache keeps.
const KEPT_KEYS: usize = 8;
/// An orphan `.tmp` older than this is deleted.
const TMP_MAX_AGE: Duration = Duration::from_secs(60);

const MAGIC: &[u8; 4] = b"PWCM";
/// magic, version, seed, body, face, size, key hash, payload checksum.
const HEADER: usize = 40;

static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// The cache key's hash: FNV-1a 64 over every channel of the colour snapshot, then the worldgen
/// fingerprint, chart `n`, and the FNV-1a 64 of the datum offsets' little-endian bits.
pub(super) fn key_hash(colors: &[Color], fingerprint: u64, n: i64, datum: &[f32]) -> u64 {
    let mut offsets = Fnv64::new();
    for o in datum {
        offsets.bytes(&o.to_le_bytes());
    }
    let mut h = Fnv64::new();
    for c in colors {
        h.bytes(&[c.r, c.g, c.b, c.a]);
    }
    h.bytes(&fingerprint.to_le_bytes()).bytes(&n.to_le_bytes()).bytes(&offsets.finish().to_le_bytes()).finish()
}

#[derive(Clone, Copy)]
pub(super) struct Key {
    pub(super) seed: i64,
    pub(super) body: u16,
    pub(super) hash: u64,
    pub(super) version: u32,
}

fn cache_path(dir: &Path, key: &Key, size: u32, face: usize) -> PathBuf {
    dir.join(format!(
        "pm-{}-{}-{:016x}-v{}-s{size}-f{face}.bin",
        key.seed, key.body, key.hash, key.version
    ))
}

fn tmp_path(path: &Path) -> PathBuf {
    let n = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let mut name = path.file_name().map(|s| s.to_os_string()).unwrap_or_default();
    name.push(format!(".{}.{n}.tmp", std::process::id()));
    path.with_file_name(name)
}

fn touch(path: &Path) {
    let Ok(file) = std::fs::File::options().write(true).open(path) else { return };
    let _ = file.set_modified(SystemTime::now());
}

/// `pm-{seed}-{body}-{hash}-v{version}-s{size}-f{face}.bin` → (group, version).
fn cache_name_parts(name: &str) -> Option<(&str, u32)> {
    let stem = name.strip_suffix(".bin")?;
    let (head, face) = stem.rsplit_once("-f")?;
    face.parse::<u16>().ok()?;
    let (head, size) = head.rsplit_once("-s")?;
    size.parse::<u32>().ok()?;
    let (group, ver) = head.rsplit_once("-v")?;
    Some((group, ver.parse().ok()?))
}

/// Keep the [`KEPT_KEYS`] newest keys of this [`BAKE_VERSION`]. Other versions go, and so does
/// an orphan `.tmp` older than a minute.
pub(super) fn prune_cache(dir: &Path) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let now = SystemTime::now();
    struct Item {
        path: PathBuf,
        group: String,
        mtime: SystemTime,
    }
    let mut items = Vec::new();
    for ent in rd.flatten() {
        let path = ent.path();
        let name = ent.file_name();
        let name = name.to_string_lossy();
        let mtime = ent.metadata().ok().and_then(|m| m.modified().ok()).unwrap_or(SystemTime::UNIX_EPOCH);
        let tmp = path.extension().is_some_and(|e| e == "tmp");
        if tmp {
            if now.duration_since(mtime).unwrap_or_default() > TMP_MAX_AGE {
                let _ = std::fs::remove_file(&path);
            }
            continue;
        }
        let Some((group, version)) = cache_name_parts(&name) else { continue };
        if version != BAKE_VERSION {
            let _ = std::fs::remove_file(&path);
            continue;
        }
        items.push(Item { path, group: group.to_string(), mtime });
    }
    let mut groups: Vec<(String, SystemTime)> = Vec::new();
    for item in &items {
        if let Some(found) = groups.iter_mut().find(|g| g.0 == item.group) {
            if item.mtime > found.1 {
                found.1 = item.mtime;
            }
        } else {
            groups.push((item.group.clone(), item.mtime));
        }
    }
    if groups.len() <= KEPT_KEYS {
        return;
    }
    groups.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    for item in &items {
        if groups[KEPT_KEYS..].iter().any(|g| g.0 == item.group) {
            let _ = std::fs::remove_file(&item.path);
        }
    }
}

/// Write one face. A short or mismatched buffer is not written. The payload's FNV checksum
/// sits in the header; a later load rejects a body that does not match it.
pub(super) fn cache_save(dir: &Path, key: &Key, size: u32, face: usize, rgba: &[u8]) {
    let expect = (size as usize).saturating_mul(size as usize).saturating_mul(4);
    if rgba.len() != expect || face > u16::MAX as usize {
        return;
    }
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let path = cache_path(dir, key, size, face);
    let tmp = tmp_path(&path);
    let mut head = Writer::new();
    head.raw(MAGIC);
    head.u32(key.version);
    head.i64(key.seed);
    head.u16(key.body);
    head.u16(face as u16);
    head.u32(size);
    head.u64(key.hash);
    head.u64(Fnv64::new().bytes(rgba).finish());
    let written = std::fs::File::create(&tmp).and_then(|mut file| {
        file.write_all(&head.into_inner())?;
        file.write_all(rgba)
    });
    if written.is_err() || std::fs::rename(&tmp, &path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

/// The face bytes, or `None` when the file is missing, short, keyed differently, or its
/// payload checksum does not match. A hit touches the file so the cache keeps recent keys.
pub(super) fn cache_load(dir: &Path, key: &Key, size: u32, face: usize) -> Option<Vec<u8>> {
    let path = cache_path(dir, key, size, face);
    let bytes = std::fs::read(&path).ok()?;
    let expect = (size as usize).saturating_mul(size as usize).saturating_mul(4);
    if bytes.len() != HEADER + expect || !header_matches(&bytes, key, size, face).unwrap_or(false) {
        return None;
    }
    touch(&path);
    let mut rgba = bytes;
    rgba.drain(..HEADER);
    Some(rgba)
}

/// The header names this key, size and face, and its checksum matches the payload behind it.
fn header_matches(bytes: &[u8], key: &Key, size: u32, face: usize) -> Result<bool, CodecError> {
    let mut r = Reader::new(bytes);
    Ok(r.take(4)? == MAGIC
        && r.u32()? == key.version
        && r.i64()? == key.seed
        && r.u16()? == key.body
        && r.u16()? == face as u16
        && r.u32()? == size
        && r.u64()? == key.hash
        && r.u64()? == Fnv64::new().bytes(r.take(r.remaining())?).finish())
}

#[cfg(test)]
mod tests {
    use super::super::tests::scratch;
    use super::*;

    #[test]
    fn cache_round_trips_and_rejects_a_stale_key() {
        let dir = scratch("cache");
        let key = Key { seed: 42, body: 3, hash: 0xabc, version: BAKE_VERSION };
        let rgba = vec![9u8, 8, 7, 255, 1, 2, 3, 255, 4, 5, 6, 255, 7, 8, 9, 255];
        cache_save(&dir, &key, 2, 4, &rgba);
        assert_eq!(cache_load(&dir, &key, 2, 4).as_deref(), Some(rgba.as_slice()));
        let stale = Key { hash: 0xabd, ..key };
        assert!(cache_load(&dir, &stale, 2, 4).is_none());
        let old = Key { version: BAKE_VERSION + 1, ..key };
        assert!(cache_load(&dir, &old, 2, 4).is_none());
        let path = cache_path(&dir, &key, 2, 4);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[HEADER] ^= 0xff;
        std::fs::write(&path, &bytes).unwrap();
        assert!(cache_load(&dir, &key, 2, 4).is_none(), "a bad payload checksum is rejected");
        bytes[HEADER] ^= 0xff;
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(cache_load(&dir, &key, 2, 4).as_deref(), Some(rgba.as_slice()));
        bytes[0] = b'X';
        std::fs::write(&path, &bytes).unwrap();
        assert!(cache_load(&dir, &key, 2, 4).is_none());
        std::fs::write(&path, &bytes[..10]).unwrap();
        assert!(cache_load(&dir, &key, 2, 4).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_tmp_name_is_unique_to_the_process_and_the_write() {
        let path = PathBuf::from("pm-1-1-0000000000000001-v3-s4-f0.bin");
        let a = tmp_path(&path);
        let b = tmp_path(&path);
        assert_ne!(a, b);
        let pid = std::process::id().to_string();
        for name in [a, b].map(|p| p.file_name().unwrap().to_string_lossy().into_owned()) {
            assert!(name.contains(&pid), "{name}");
            assert!(name.ends_with(".tmp"), "{name}");
        }
        let (group, ver) = cache_name_parts("pm--7-2-0000000000000001-v3-s1024-f0.bin").unwrap();
        assert_eq!(group, "pm--7-2-0000000000000001");
        assert_eq!(ver, BAKE_VERSION);
    }

    #[test]
    fn the_cache_keeps_the_eight_newest_keys_and_drops_stale_files() {
        let dir = scratch("prune");
        std::fs::create_dir_all(&dir).unwrap();
        let rgba = vec![1u8, 2, 3, 255];
        let now = SystemTime::now();
        for i in 0..KEPT_KEYS + 1 {
            let key = Key { seed: i as i64, body: 1, hash: i as u64, version: BAKE_VERSION };
            cache_save(&dir, &key, 1, 0, &rgba);
            let path = cache_path(&dir, &key, 1, 0);
            let age = Duration::from_secs(10 * (KEPT_KEYS as u64 + 1 - i as u64));
            let file = std::fs::File::options().write(true).open(&path).unwrap();
            file.set_modified(now.checked_sub(age).unwrap()).unwrap();
        }
        let old = Key { seed: 99, body: 1, hash: 99, version: BAKE_VERSION + 1 };
        cache_save(&dir, &old, 1, 0, &rgba);
        let old_tmp = dir.join(format!("orphan.{}.tmp", std::process::id()));
        std::fs::write(&old_tmp, b"x").unwrap();
        let file = std::fs::File::options().write(true).open(&old_tmp).unwrap();
        file.set_modified(now.checked_sub(Duration::from_secs(120)).unwrap()).unwrap();
        drop(file);
        let fresh_tmp = dir.join("fresh.tmp");
        std::fs::write(&fresh_tmp, b"y").unwrap();

        prune_cache(&dir);

        assert!(!old_tmp.exists(), "a tmp older than a minute is deleted");
        assert!(fresh_tmp.exists(), "a fresh tmp stays");
        assert!(!cache_path(&dir, &old, 1, 0).exists(), "another bake version is deleted");
        for i in 0..KEPT_KEYS + 1 {
            let key = Key { seed: i as i64, body: 1, hash: i as u64, version: BAKE_VERSION };
            let exists = cache_path(&dir, &key, 1, 0).exists();
            assert_eq!(exists, i != 0, "key {i} kept={exists}");
        }

        let key = Key { seed: 1, body: 1, hash: 1, version: BAKE_VERSION };
        let path = cache_path(&dir, &key, 1, 0);
        let old_m = now.checked_sub(Duration::from_secs(10_000)).unwrap();
        std::fs::File::options().write(true).open(&path).unwrap().set_modified(old_m).unwrap();
        assert_eq!(cache_load(&dir, &key, 1, 0).as_deref(), Some(rgba.as_slice()));
        let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
        assert!(mtime > old_m + Duration::from_secs(1_000), "a hit touches the file");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A face cached by an older build still loads, and a save writes the same bytes.
    #[test]
    fn a_cached_face_keeps_its_bytes() {
        let dir = scratch("golden");
        let key = Key { seed: -7, body: 3, hash: 0x0123_4567_89ab_cdef, version: BAKE_VERSION };
        let rgba: Vec<u8> = (0..16u8).map(|i| i.wrapping_mul(37)).collect();
        cache_save(&dir, &key, 2, 4, &rgba);
        let path = cache_path(&dir, &key, 2, 4);
        let bytes = std::fs::read(&path).unwrap();
        const GOLDEN: [u8; HEADER + 16] = [
            80, 87, 67, 77, 3, 0, 0, 0, 249, 255, 255, 255, 255, 255, 255, 255, 3, 0, 4, 0, 2, 0, 0, 0, 239, 205, 171,
            137, 103, 69, 35, 1, 53, 71, 62, 183, 21, 152, 109, 55, 0, 37, 74, 111, 148, 185, 222, 3, 40, 77, 114, 151,
            188, 225, 6, 43,
        ];
        assert_eq!(bytes, GOLDEN);
        std::fs::write(&path, GOLDEN).unwrap();
        assert_eq!(cache_load(&dir, &key, 2, 4), Some(rgba));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
