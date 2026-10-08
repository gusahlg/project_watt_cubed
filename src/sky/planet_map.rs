//! Bake a home body's albedo cube map, and hand that map to the sky impostor.
//!
//! Texel directions are the engine's [`far_cube_texel_dir`](voxel_engine::far_cube_texel_dir),
//! so a baked face is the cube the sky samples. The draw keeps the 256² previews until all six
//! 1024² faces are ready, then installs that cube once.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime};

use glam::DVec3;
use voxel_engine::Color;

use crate::block::registry::{AIR, BlockId};
use crate::coord::Face;
use crate::sky::palette::Rgb;
use crate::space::atlas::FACES;
use crate::space::chart::{basis, Map};
use crate::space::datum::DatumField;
use crate::world::generation::TerrainGenerator;
use crate::world::terrain::cosmos::{Body, Cosmos, Kind};
use crate::world::terrain::Generator;

/// Bump when the bake bytes change. Files from any other version are deleted.
pub(crate) const BAKE_VERSION: u32 = 3;
/// First pass, every face, so a far body has a colour before the full map lands.
pub(crate) const PREVIEW: u32 = 256;
/// Full face. Six of these are the cached map.
pub(crate) const FULL: u32 = 1024;
/// How many distinct bake keys the disk cache keeps.
const KEPT_KEYS: usize = 8;
/// An orphan `.tmp` older than this is deleted.
const TMP_MAX_AGE: Duration = Duration::from_secs(60);

const MAGIC: &[u8; 4] = b"PWCM";
/// magic, version, seed, body, face, size, key hash, payload checksum.
const HEADER: usize = 40;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x100_0000_01b3;

static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Body-space direction of a cube texel centre. The engine owns the cube table;
/// this promotes it to f64 for the chart lookup.
pub(crate) fn cube_texel_dir(face: usize, size: u32, x: u32, y: u32) -> DVec3 {
    let d = voxel_engine::far_cube_texel_dir(face, size, x, y);
    DVec3::new(d.x as f64, d.y as f64, d.z as f64)
}

/// Chart column under `dir`: dominant face, then equiangular `(ξ, η)` rounded to a column.
/// Corners are shared by three faces; [`Face::from_dominant`] picks one. Not clamped.
pub(crate) fn column_of(n: i64, dir: DVec3) -> (usize, i32, i32) {
    let face = Face::from_dominant(dir);
    let f = FACES.iter().position(|&g| g == face).unwrap_or(2);
    let (tu, nn, tv) = basis(face);
    let local = DVec3::new(dir.dot(tu), dir.dot(nn), dir.dot(tv));
    let (xi, eta) = Map::Equiangular.inverse(local);
    let half = n as f64 * 0.5;
    (f, round_i32(xi * half), round_i32(eta * half))
}

/// Unit direction of chart column `(u, v)` on `face` ([`FACES`] order), equiangular.
#[cfg(test)]
fn column_direction(face: usize, n: i64, u: i32, v: i32) -> DVec3 {
    let n = n.max(1) as f64;
    let d = Map::Equiangular.dir(2.0 * u as f64 / n, 2.0 * v as f64 / n);
    let (tu, nn, tv) = basis(FACES[face.min(5)]);
    (tu * d.x + nn * d.y + tv * d.z).normalize()
}

fn round_i32(x: f64) -> i32 {
    if !x.is_finite() {
        return 0;
    }
    if x >= i32::MAX as f64 {
        i32::MAX
    } else if x <= i32::MIN as f64 {
        i32::MIN
    } else {
        x.round() as i32
    }
}

/// Keep a column on the chart. `u = +n/2` is the seam past the last cell.
pub(crate) fn clamp_column(n: i64, u: i32, v: i32) -> (i32, i32) {
    let half = (n / 2).clamp(1, i32::MAX as i64) as i32;
    let hi = half - 1;
    (u.clamp(-half, hi), v.clamp(-half, hi))
}

fn linear_of(colors: &[Color], id: BlockId) -> Rgb {
    match colors.get(id.0 as usize) {
        Some(c) => Rgb::from_srgb8(c.r, c.g, c.b),
        None => Rgb::linear(0.0, 0.0, 0.0),
    }
}

/// Linear colour the far field shows: surface snapshot, blended toward the leaf by `trees`.
pub(crate) fn column_linear(colors: &[Color], surface: BlockId, trees: f32, leaf: BlockId) -> [f32; 3] {
    let base = linear_of(colors, surface);
    let blended = if leaf == AIR || trees == 0.0 { base } else { base.lerp(linear_of(colors, leaf), trees) };
    [blended.r(), blended.g(), blended.b()]
}

/// One face, RGBA8 sRGB, row-major. Each texel is the mean of a 2×2 of texel-centre directions.
/// `None` when `cancel` is set, checked once per row.
pub(crate) fn bake_face(
    size: u32,
    face: usize,
    n: i64,
    cancel: &AtomicBool,
    mut sample: impl FnMut(usize, i32, i32) -> [f32; 3],
) -> Option<Vec<u8>> {
    let side = size.max(1);
    let texels = (side as usize).saturating_mul(side as usize);
    let mut out = vec![0u8; texels.saturating_mul(4)];
    let ss = side.saturating_mul(2);
    for y in 0..side {
        if cancel.load(Ordering::Relaxed) {
            return None;
        }
        for x in 0..side {
            let mut acc = [0.0f32; 3];
            for sy in 0..2u32 {
                for sx in 0..2u32 {
                    let d = cube_texel_dir(face, ss, x * 2 + sx, y * 2 + sy);
                    let (f, u, v) = column_of(n, d);
                    let (u, v) = clamp_column(n, u, v);
                    let c = sample(f, u, v);
                    acc[0] += c[0];
                    acc[1] += c[1];
                    acc[2] += c[2];
                }
            }
            let q = Rgb::linear(acc[0] * 0.25, acc[1] * 0.25, acc[2] * 0.25).to_srgb8();
            let i = (y as usize * side as usize + x as usize) * 4;
            out[i] = q.r;
            out[i + 1] = q.g;
            out[i + 2] = q.b;
            out[i + 3] = 255;
        }
    }
    Some(out)
}

/// `(datum_res, offsets)` in [`DatumField`] order, relative to the impostor's radius:
/// `datum[i] = field[i] − lo`, so `R_imp + datum[i] = R + field[i] − sink`.
pub(crate) fn impostor_datum(cosmos: &Cosmos, body: &Body, field: &DatumField) -> Option<(u32, Vec<f32>)> {
    if body.kind != Kind::Home || field.g < 2 {
        return None;
    }
    let lo = cosmos.relief_range(body).0.min(cosmos.face_offset(body));
    let lo_f = lo as f32;
    let datum = field.offsets.iter().map(|o| o - lo_f).collect();
    Some((field.g as u32, datum))
}

/// Sine the shader compares with `dot(ray, -dir)`.
///
/// `hi` is the reference radius plus the highest datum offset. Every surface point and every
/// air point lies inside the ball `hi + air`, so a ray that could hit is never culled.
/// `1` disables the cull: the eye is inside that ball, and `dot > 1` is impossible.
pub(crate) fn horizon_sine(dist: f64, hi: f64, air: f64) -> f32 {
    let reach = hi + air;
    if !(dist > reach) || !dist.is_finite() || !reach.is_finite() || reach < 0.0 {
        return 1.0;
    }
    let ratio = reach / dist;
    let inside = 1.0 - ratio * ratio;
    if !(inside > 0.0) {
        return 1.0;
    }
    let h = -inside.sqrt();
    if h.is_finite() { h as f32 } else { 1.0 }
}

fn fnv_byte(mut h: u64, b: u8) -> u64 {
    h ^= b as u64;
    h.wrapping_mul(FNV_PRIME)
}

fn fnv_bytes(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h = fnv_byte(h, b);
    }
    h
}

/// FNV-1a 64 over every channel of the colour snapshot.
pub(crate) fn color_hash(colors: &[Color]) -> u64 {
    let mut h = FNV_OFFSET;
    for c in colors {
        for b in [c.r, c.g, c.b, c.a] {
            h = fnv_byte(h, b);
        }
    }
    h
}

/// FNV-1a 64 over the datum offsets' little-endian bits.
fn datum_hash(offsets: &[f32]) -> u64 {
    let mut h = FNV_OFFSET;
    for o in offsets {
        h = fnv_bytes(h, &o.to_le_bytes());
    }
    h
}

/// Colour hash, worldgen fingerprint, chart `n`, and the datum hash, folded into the cache key.
fn fold_key(color: u64, fingerprint: u64, n: i64, datum: u64) -> u64 {
    let mut h = fnv_bytes(color, &fingerprint.to_le_bytes());
    h = fnv_bytes(h, &n.to_le_bytes());
    fnv_bytes(h, &datum.to_le_bytes())
}

#[derive(Clone, Copy)]
struct Key {
    seed: i64,
    body: u16,
    hash: u64,
    version: u32,
}

fn cache_path(dir: &Path, key: &Key, size: u32, face: usize) -> PathBuf {
    dir.join(format!(
        "pm-{}-{}-{:016x}-v{}-s{size}-f{face}.bin",
        key.seed, key.body, key.hash, key.version
    ))
}

fn push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn read_exact(bytes: &[u8], at: usize, n: usize) -> Option<&[u8]> {
    bytes.get(at..at + n)
}

fn u32_at(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(read_exact(bytes, at, 4)?.try_into().ok()?))
}

fn u64_at(bytes: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(read_exact(bytes, at, 8)?.try_into().ok()?))
}

fn i64_at(bytes: &[u8], at: usize) -> Option<i64> {
    Some(i64::from_le_bytes(read_exact(bytes, at, 8)?.try_into().ok()?))
}

fn u16_at(bytes: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(read_exact(bytes, at, 2)?.try_into().ok()?))
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
fn prune_cache(dir: &Path) {
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
fn cache_save(dir: &Path, key: &Key, size: u32, face: usize, rgba: &[u8]) {
    let expect = (size as usize).saturating_mul(size as usize).saturating_mul(4);
    if rgba.len() != expect || face > u16::MAX as usize {
        return;
    }
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let path = cache_path(dir, key, size, face);
    let tmp = tmp_path(&path);
    let sum = fnv_bytes(FNV_OFFSET, rgba);
    let mut bytes = Vec::with_capacity(HEADER + rgba.len());
    bytes.extend_from_slice(MAGIC);
    push_u32(&mut bytes, key.version);
    bytes.extend_from_slice(&key.seed.to_le_bytes());
    bytes.extend_from_slice(&key.body.to_le_bytes());
    bytes.extend_from_slice(&(face as u16).to_le_bytes());
    push_u32(&mut bytes, size);
    bytes.extend_from_slice(&key.hash.to_le_bytes());
    bytes.extend_from_slice(&sum.to_le_bytes());
    bytes.extend_from_slice(rgba);
    if std::fs::write(&tmp, &bytes).is_err() {
        let _ = std::fs::remove_file(&tmp);
        return;
    }
    if std::fs::rename(&tmp, &path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

/// The face bytes, or `None` when the file is missing, short, keyed differently, or its
/// payload checksum does not match. A hit touches the file so the cache keeps recent keys.
fn cache_load(dir: &Path, key: &Key, size: u32, face: usize) -> Option<Vec<u8>> {
    let path = cache_path(dir, key, size, face);
    let bytes = std::fs::read(&path).ok()?;
    let expect = (size as usize).saturating_mul(size as usize).saturating_mul(4);
    if bytes.len() != HEADER + expect || bytes.get(..4) != Some(MAGIC) {
        return None;
    }
    if u32_at(&bytes, 4)? != key.version
        || i64_at(&bytes, 8)? != key.seed
        || u16_at(&bytes, 16)? != key.body
        || u16_at(&bytes, 18)? != face as u16
        || u32_at(&bytes, 20)? != size
        || u64_at(&bytes, 24)? != key.hash
        || u64_at(&bytes, 32)? != fnv_bytes(FNV_OFFSET, &bytes[HEADER..])
    {
        return None;
    }
    touch(&path);
    Some(bytes[HEADER..].to_vec())
}

struct Job {
    generator: Generator,
    colors: Box<[Color]>,
    key: Key,
    n: i64,
    cache: PathBuf,
    preview: u32,
    full: u32,
    cancel: Arc<AtomicBool>,
}

/// `(body, seed, chart n)` for the start world, when it has a datum chart.
fn home_chart_n(generator: &dyn TerrainGenerator) -> Option<(u16, i64, i64)> {
    let cosmos = generator.cosmos()?;
    let body = cosmos.home();
    if body.kind != Kind::Home {
        return None;
    }
    let centre = body.centre_f();
    let n = generator.atlases().iter().find_map(|a| {
        if a.datum.is_none() || (a.centre - centre).length_squared() >= 1.0 {
            return None;
        }
        a.bands.first().map(|b| b.n)
    })?;
    (n >= 2).then_some((body.id, generator.seed(), n))
}

/// Raw datum offsets of the start world's chart, the bytes the cache key hashes.
fn home_datum(generator: &dyn TerrainGenerator) -> Option<&[f32]> {
    let cosmos = generator.cosmos()?;
    let body = cosmos.home();
    if body.kind != Kind::Home {
        return None;
    }
    let centre = body.centre_f();
    generator.atlases().iter().find_map(|a| {
        let same = (a.centre - centre).length_squared() < 1.0;
        a.datum.as_ref().filter(|_| same).map(|d| d.offsets.as_slice())
    })
}

/// Cache key for this world's bake. Relief, variety and the other worldgen knobs are inside
/// the fingerprint, so a knob change does not reuse another world's faces.
fn map_key(
    generator: &dyn TerrainGenerator,
    registry: &crate::block::registry::BlockRegistry,
    kind: crate::world::generation::WorldgenKind,
    cfg: crate::world::terrain::TerrainCfg,
) -> Option<Key> {
    let (body, seed, n) = home_chart_n(generator)?;
    let datum = home_datum(generator)?;
    let colors = registry.color_snapshot();
    let fp = world_fingerprint(registry, kind, cfg);
    let hash = fold_key(color_hash(&colors), fp, n, datum_hash(datum));
    Some(Key { seed, body, hash, version: BAKE_VERSION })
}

/// The world's content id (generator version, gravity, law, palette) folded with the worldgen kind
/// and its knobs: the inputs that change the baked image.
fn world_fingerprint(
    registry: &crate::block::registry::BlockRegistry,
    kind: crate::world::generation::WorldgenKind,
    cfg: crate::world::terrain::TerrainCfg,
) -> u64 {
    let id = crate::net::content_id(registry);
    let mut h = fnv_bytes(0xcbf2_9ce4_8422_2325, &id.worldgen.to_le_bytes());
    for word in [id.gravity, id.law, id.palette] {
        h = fnv_bytes(h, &word.to_le_bytes());
    }
    h = fnv_bytes(h, kind.id().as_bytes());
    for v in cfg.clamp().to_wire() {
        h = fnv_bytes(h, &v.to_le_bytes());
    }
    h
}

/// A benchmark or a scripted game does not bake. `WATT_BENCH_PLANET_MAP=1` opts a benchmark back in.
/// A scripted game never bakes, opt-in or not.
pub(crate) fn bake_enabled(scripted: bool, bench_set: bool, map_opt_in: bool) -> bool {
    if scripted {
        false
    } else if bench_set {
        map_opt_in
    } else {
        true
    }
}

pub(crate) fn bake_wanted(scripted: bool) -> bool {
    let bench = std::env::var_os("WATT_BENCH").is_some_and(|v| !v.is_empty());
    let opt_in = std::env::var_os("WATT_BENCH_PLANET_MAP").is_some_and(|v| v == "1");
    bake_enabled(scripted, bench, opt_in)
}

fn sample_linear(job: &Job, face: usize, u: i32, v: i32) -> [f32; 3] {
    let terra: &dyn TerrainGenerator = job.generator.as_ref();
    match terra.home_far_column(face, u, v) {
        Some((surface, trees, leaf)) => column_linear(&job.colors, surface, trees, leaf),
        None => [0.0, 0.0, 0.0],
    }
}

struct Msg {
    full: bool,
    face: usize,
    rgba: Vec<u8>,
}

/// Lowest scheduling priority for this thread. Linux nice 19; other platforms leave it.
fn low_priority() {
    // Nice 19: the bake must not compete with the simulation. Failure is ignored.
    #[cfg(target_os = "linux")]
    unsafe {
        let _ = libc::setpriority(libc::PRIO_PROCESS, libc::gettid() as libc::id_t, 19);
    }
}

fn worker(job: Job, tx: Sender<Msg>) {
    low_priority();
    prune_cache(&job.cache);
    if job.cancel.load(Ordering::Relaxed) {
        return;
    }
    let mut cached: [Option<Vec<u8>>; 6] = [None, None, None, None, None, None];
    for face in 0..6 {
        if job.cancel.load(Ordering::Relaxed) {
            return;
        }
        if let Some(rgba) = cache_load(&job.cache, &job.key, job.full, face) {
            cached[face] = Some(rgba);
        }
    }
    if cached.iter().all(|c| c.is_some()) {
        for face in 0..6 {
            let Some(rgba) = cached[face].take() else { return };
            if tx.send(Msg { full: true, face, rgba }).is_err() {
                return;
            }
        }
        return;
    }
    // A partial cache still draws previews. Baking the cached faces at preview size is what
    // fills the faces the disk already had, so the cube is never half flat colours.
    if job.preview > 0 && job.preview != job.full {
        for face in 0..6 {
            let Some(rgba) = bake_face(job.preview, face, job.n, &job.cancel, |f, u, v| sample_linear(&job, f, u, v)) else {
                return;
            };
            if tx.send(Msg { full: false, face, rgba }).is_err() {
                return;
            }
        }
    }
    for face in 0..6 {
        if let Some(rgba) = cached[face].take() {
            if tx.send(Msg { full: true, face, rgba }).is_err() {
                return;
            }
            continue;
        }
        let Some(rgba) = bake_face(job.full, face, job.n, &job.cancel, |f, u, v| sample_linear(&job, f, u, v)) else {
            return;
        };
        cache_save(&job.cache, &job.key, job.full, face, &rgba);
        if tx.send(Msg { full: true, face, rgba }).is_err() {
            return;
        }
    }
}

/// Faces the mapped impostor will upload, plus the datum and horizon already on [`super::bodies::FarBodies`].
#[cfg(test)]
pub(crate) struct PlanetHandle<'a> {
    pub datum_res: u32,
    pub datum: &'a [f32],
    pub radius: f64,
    pub horizon: f32,
    pub preview: [Option<&'a [u8]>; 6],
    pub full: [Option<&'a [u8]>; 6],
}

/// Background bake. A quiet [`poll`](Self::poll) receives nothing and allocates nothing.
pub(crate) struct PlanetBake {
    started: bool,
    worker: Option<JoinHandle<()>>,
    rx: Option<Receiver<Msg>>,
    cancel: Option<Arc<AtomicBool>>,
    preview: [Option<Vec<u8>>; 6],
    full: [Option<Vec<u8>>; 6],
}

impl Default for PlanetBake {
    fn default() -> Self {
        Self {
            started: false,
            worker: None,
            rx: None,
            cancel: None,
            preview: [None, None, None, None, None, None],
            full: [None, None, None, None, None, None],
        }
    }
}

impl Drop for PlanetBake {
    fn drop(&mut self) {
        if let Some(flag) = &self.cancel {
            flag.store(true, Ordering::Relaxed);
        }
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
        self.rx = None;
    }
}

impl PlanetBake {
    pub(crate) fn started(&self) -> bool {
        self.started
    }

    /// Preview (`full == false`) or full faces currently in hand. Missing faces are `None`.
    #[cfg(test)]
    pub(crate) fn faces(&self, full: bool) -> [Option<&[u8]>; 6] {
        let src = if full { &self.full } else { &self.preview };
        std::array::from_fn(|i| src[i].as_deref())
    }

    #[cfg(test)]
    pub(super) fn deliver(&mut self, full: bool, face: usize, rgba: Vec<u8>) {
        if face >= 6 {
            return;
        }
        let slot = if full { &mut self.full } else { &mut self.preview };
        slot[face] = Some(rgba);
    }

    /// Start the production bake (256² then 1024²) under the game's data dir. Once only.
    /// A failed spawn is not retried: `started` is set before the thread exists.
    pub(crate) fn ensure(
        &mut self,
        generator: Generator,
        registry: &crate::block::registry::BlockRegistry,
        kind: crate::world::generation::WorldgenKind,
        cfg: crate::world::terrain::TerrainCfg,
    ) {
        if self.started {
            return;
        }
        self.started = true;
        let Some(key) = map_key(generator.as_ref(), registry, kind, cfg) else { return };
        let Some((_, _, n)) = home_chart_n(generator.as_ref()) else { return };
        let colors = registry.color_snapshot();
        let cache = crate::paths::Paths::get().data.join("planet-maps");
        self.spawn(Job {
            generator,
            colors,
            key,
            n,
            cache,
            preview: PREVIEW,
            full: FULL,
            cancel: Arc::new(AtomicBool::new(false)),
        });
    }

    fn spawn(&mut self, job: Job) {
        let cancel = Arc::clone(&job.cancel);
        let (tx, rx) = std::sync::mpsc::channel();
        let handle = match std::thread::Builder::new().name("planet-map".into()).spawn(move || worker(job, tx)) {
            Ok(h) => h,
            Err(_) => return,
        };
        self.cancel = Some(cancel);
        self.rx = Some(rx);
        self.worker = Some(handle);
    }

    /// Take any faces the worker has finished. Empty when nothing arrived.
    pub(crate) fn poll(&mut self) {
        loop {
            let msg = {
                let Some(rx) = self.rx.as_ref() else { break };
                match rx.try_recv() {
                    Ok(msg) => msg,
                    Err(_) => break,
                }
            };
            if msg.face >= 6 {
                continue;
            }
            let slot = if msg.full { &mut self.full } else { &mut self.preview };
            slot[msg.face] = Some(msg.rgba);
        }
        if self.worker.as_ref().is_some_and(|h| h.is_finished()) {
            if let Some(handle) = self.worker.take() {
                let _ = handle.join();
            }
            self.rx = None;
            self.cancel = None;
        }
    }

    #[cfg(test)]
    fn start_with(&mut self, generator: Generator, colors: Box<[Color]>, n: i64, key: Key, preview: u32, full: u32, cache: PathBuf) {
        if self.started {
            return;
        }
        self.started = true;
        self.spawn(Job {
            generator,
            colors,
            key,
            n,
            cache,
            preview,
            full,
            cancel: Arc::new(AtomicBool::new(false)),
        });
    }

    #[cfg(test)]
    fn finish(&mut self) {
        let start = std::time::Instant::now();
        while self.worker.is_some() {
            self.poll();
            if self.worker.is_some() {
                assert!(start.elapsed().as_secs() < 30, "planet-map bake did not finish");
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        self.poll();
    }
}

/// The start world's map slot. One world, one map.
pub(super) const HOME_MAP: voxel_engine::FarMapId = voxel_engine::FarMapId(0);

/// Where the home datum and its faces are installed. The engine is one; tests record another.
pub(crate) trait FarSink {
    fn install(&mut self, datum_res: u32, datum: &[f32], albedo_size: u32) -> bool;
    fn face(&mut self, face: usize, rgba: &[u8]) -> bool;
    fn clear(&mut self);
}

/// Which cube is installed, and which faces have already been sent. A quiet frame only reads these.
#[derive(Default)]
pub(super) struct MapFeed {
    size: Option<u32>,
    sent_preview: [bool; 6],
    sent_full: [bool; 6],
    /// The full cube replaced a complete preview cube: the engine keeps drawing that one until every
    /// full face has landed (its atomic swap), so home stays mapped meanwhile.
    swapping: bool,
}

fn six_none() -> [Option<Vec<u8>>; 6] {
    [None, None, None, None, None, None]
}

fn all_present(faces: &[Option<Vec<u8>>; 6]) -> bool {
    faces.iter().all(|f| f.is_some())
}

fn all_sent(sent: &[bool; 6]) -> bool {
    sent.iter().all(|s| *s)
}

/// Send the next unsent face and drop its bytes. One face, so a frame copies at most one.
fn send_one(sent: &mut [bool; 6], faces: &mut [Option<Vec<u8>>; 6], sink: &mut impl FarSink) {
    for i in 0..6 {
        if sent[i] {
            continue;
        }
        let Some(bytes) = faces[i].as_deref() else { continue };
        if sink.face(i, bytes) {
            sent[i] = true;
            faces[i] = None;
        }
        return;
    }
}

impl MapFeed {
    /// Install once we have six previews, or six full faces. Stay on the preview cube until every
    /// full face is in hand, then install the full cube once. At most one face leaves per call,
    /// and that face's bytes are dropped. `true` once all six faces of a cube have been sent, so
    /// the impostor never draws a flat fallback face: across the preview-to-full swap the engine
    /// keeps drawing the complete preview cube.
    pub(super) fn flush(
        &mut self,
        datum_res: u32,
        datum: &[f32],
        bake: &mut PlanetBake,
        sink: &mut impl FarSink,
    ) -> bool {
        let all_full = all_present(&bake.full);
        let all_preview = all_present(&bake.preview);
        if self.size.is_none() {
            if all_full {
                if !sink.install(datum_res, datum, FULL) {
                    return false;
                }
                self.size = Some(FULL);
                bake.preview = six_none();
            } else if all_preview {
                if !sink.install(datum_res, datum, PREVIEW) {
                    return false;
                }
                self.size = Some(PREVIEW);
            } else {
                return false;
            }
        } else if self.size == Some(PREVIEW) && all_sent(&self.sent_preview) && all_full {
            if sink.install(datum_res, datum, FULL) {
                self.size = Some(FULL);
                self.sent_full = [false; 6];
                self.swapping = true;
                bake.preview = six_none();
            }
        }
        match self.size {
            Some(PREVIEW) if !all_sent(&self.sent_preview) => {
                send_one(&mut self.sent_preview, &mut bake.preview, sink);
            }
            Some(FULL) if !all_sent(&self.sent_full) => {
                send_one(&mut self.sent_full, &mut bake.full, sink);
            }
            _ => {}
        }
        match self.size {
            Some(PREVIEW) => all_sent(&self.sent_preview),
            Some(FULL) => self.swapping || all_sent(&self.sent_full),
            _ => false,
        }
    }

    /// Drop the installed map. The next [`flush`](Self::flush) installs it again.
    pub(super) fn clear(&mut self, sink: &mut impl FarSink) {
        sink.clear();
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::{
        bake_face, bake_enabled, cache_load, cache_name_parts, cache_path, cache_save, clamp_column, color_hash,
        column_direction, column_linear, column_of, cube_texel_dir, home_chart_n, horizon_sine, impostor_datum,
        map_key, prune_cache, round_i32, tmp_path, world_fingerprint, FarSink, Key, MapFeed, PlanetBake, BAKE_VERSION,
        FULL, HEADER, HOME_MAP, KEPT_KEYS, PREVIEW,
    };
    use crate::alloc_count;
    use crate::block::registry::{AIR, BlockId, BlockRegistry};
    use crate::space::atlas::{surface_n, FACES};
    use crate::space::chart::{basis, Map};
    use crate::space::datum::DatumField;
    use crate::world::generation::TerrainGenerator;
    use crate::world::generation::WorldgenKind;
    use crate::world::terrain::cosmos::{Kind, AIR_TOP, HOME_RADIUS};
    use crate::world::terrain::TerrainCfg;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::{Duration, SystemTime};
    use crate::sky::palette::{Anchor, Role, Rgb, NEW_SHOKA};
    use crate::world::terrain::Terrain;
    use glam::DVec3;
    use std::path::PathBuf;
    use voxel_engine::{far_cube_texel_dir, far_map_basis, Color, FarShape};

    fn params_of(face: usize, dir: DVec3) -> (f64, f64) {
        let (tu, nn, tv) = basis(FACES[face]);
        Map::Equiangular.inverse(DVec3::new(dir.dot(tu), dir.dot(nn), dir.dot(tv)))
    }

    #[test]
    fn the_preview_is_256_and_the_full_face_is_1024() {
        assert_eq!(PREVIEW, 256);
        assert_eq!(FULL, 1024);
    }

    fn promote(v: voxel_engine::Vec3) -> DVec3 {
        DVec3::new(v.x as f64, v.y as f64, v.z as f64)
    }

    /// The engine chart is the game chart, and a baked texel lands on the column it sampled.
    #[test]
    fn engine_basis_matches_the_chart_and_texels_round_trip() {
        let n = surface_n(HOME_RADIUS);
        for face in 0..6 {
            let (tu, nn, tv) = basis(FACES[face]);
            let (eu, en, ev) = far_map_basis(face);
            assert_eq!((promote(eu), promote(en), promote(ev)), (tu, nn, tv), "face {face}");
            let centre = far_cube_texel_dir(face, 1, 0, 0);
            let dir = cube_texel_dir(face, 1, 0, 0);
            assert_eq!(dir, promote(centre));
            assert_eq!(column_of(n, dir), (face, 0, 0), "face centre texel");
            for (x, y) in [(4u32, 4), (8, 4), (4, 12), (12, 8)] {
                let d = cube_texel_dir(face, 16, x, y);
                assert_eq!(d, promote(far_cube_texel_dir(face, 16, x, y)));
                let (f, u, v) = column_of(n, d);
                assert_eq!(f, face, "texel {x},{y} on face {face} -> {f}");
                let back = column_direction(f, n, u, v);
                assert!(d.dot(back) > 1.0 - 1e-6, "face {face} texel {x},{y} dot {}", d.dot(back));
                assert_eq!(column_of(n, back), (f, u, v));
            }
        }
    }

    #[derive(Default)]
    struct Rec {
        events: Vec<Ev>,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Ev {
        Install(u32),
        Face(usize, usize),
        Clear,
    }

    impl FarSink for Rec {
        fn install(&mut self, _datum_res: u32, datum: &[f32], albedo_size: u32) -> bool {
            assert!(!datum.is_empty());
            self.events.push(Ev::Install(albedo_size));
            true
        }
        fn face(&mut self, face: usize, rgba: &[u8]) -> bool {
            self.events.push(Ev::Face(face, rgba.len()));
            true
        }
        fn clear(&mut self) {
            self.events.push(Ev::Clear);
        }
    }

    fn six(byte: u8) -> [Option<Vec<u8>>; 6] {
        std::array::from_fn(|i| Some(vec![byte, byte, byte, i as u8]))
    }

    fn hold_cancel() -> &'static AtomicBool {
        static C: AtomicBool = AtomicBool::new(false);
        &C
    }

    fn faces_since_install(events: &[Ev]) -> usize {
        let Some(last) = events.iter().rposition(|e| matches!(e, Ev::Install(_))) else { return 0 };
        events[last + 1..].iter().filter(|e| matches!(e, Ev::Face(_, _))).count()
    }

    /// Faces sent since the install before the last one: the cube the engine keeps drawing
    /// while a swap is pending.
    fn faces_of_previous_cube(events: &[Ev]) -> usize {
        let installs: Vec<usize> = events.iter().enumerate().filter(|(_, e)| matches!(e, Ev::Install(_))).map(|(i, _)| i).collect();
        let [.., prev, last] = installs[..] else { return 0 };
        events[prev + 1..last].iter().filter(|e| matches!(e, Ev::Face(_, _))).count()
    }

    /// One flush sends at most one face. A live cube has all six of its faces already sent, or it
    /// replaced a complete cube the engine keeps drawing until the new one has landed.
    fn step(feed: &mut MapFeed, datum: &[f32], bake: &mut PlanetBake, rec: &mut Rec) -> bool {
        let before = rec.events.len();
        let live = feed.flush(2, datum, bake, rec);
        let added = rec.events[before..].iter().filter(|e| matches!(e, Ev::Face(_, _))).count();
        assert!(added <= 1, "more than one face in a frame: {:?}", &rec.events[before..]);
        if live {
            let complete = faces_since_install(&rec.events) == 6 || faces_of_previous_cube(&rec.events) == 6;
            assert!(complete, "a live cube is missing a face with no complete cube behind it: {:?}", rec.events);
        }
        live
    }

    #[test]
    fn full_faces_arrive_one_at_a_time_and_nothing_falls_back_to_flat() {
        let datum = vec![0.0f32; 24];
        let mut bake = PlanetBake::default();
        let mut feed = MapFeed::default();
        let mut rec = Rec::default();
        assert!(!step(&mut feed, &datum, &mut bake, &mut rec));
        assert!(rec.events.is_empty(), "no faces yet, so home stays a sphere");

        for (i, bytes) in six(1).into_iter().enumerate() {
            bake.deliver(false, i, bytes.unwrap());
        }
        for _ in 0..5 {
            assert!(!step(&mut feed, &datum, &mut bake, &mut rec), "previews are not all sent");
        }
        assert!(step(&mut feed, &datum, &mut bake, &mut rec));
        assert_eq!(rec.events[0], Ev::Install(PREVIEW));
        assert_eq!(faces_since_install(&rec.events), 6);
        assert!(bake.faces(false).iter().all(|f| f.is_none()), "a sent preview is dropped");

        for face in 0..5 {
            bake.deliver(true, face, vec![2, 2, 2, face as u8]);
            let n = rec.events.len();
            assert!(step(&mut feed, &datum, &mut bake, &mut rec));
            assert_eq!(rec.events.len(), n, "a partial full set keeps the previews");
        }
        bake.deliver(true, 5, vec![2, 2, 2, 5]);
        assert!(step(&mut feed, &datum, &mut bake, &mut rec), "the engine keeps the preview cube through the swap");
        assert_eq!(rec.events.iter().filter(|e| matches!(e, Ev::Install(FULL))).count(), 1);
        let mut live = false;
        for _ in 0..5 {
            live = step(&mut feed, &datum, &mut bake, &mut rec);
        }
        assert!(live, "all six full faces have been sent");
        assert_eq!(faces_since_install(&rec.events), 6);
        assert_eq!(rec.events.iter().filter(|e| matches!(e, Ev::Install(_))).count(), 2);

        let n = rec.events.len();
        alloc_count::reset();
        assert!(step(&mut feed, &datum, &mut bake, &mut rec));
        assert_eq!(alloc_count::alloc_count(), 0, "a quiet upload allocated");
        assert_eq!(rec.events.len(), n);
    }

    #[test]
    fn a_full_cache_skips_the_preview_and_a_partial_cache_keeps_it() {
        let datum = vec![0.0f32; 24];
        let mut bake = PlanetBake::default();
        for (i, bytes) in six(3).into_iter().enumerate() {
            bake.deliver(true, i, bytes.unwrap());
        }
        let mut feed = MapFeed::default();
        let mut rec = Rec::default();
        let mut live = false;
        for _ in 0..6 {
            live = step(&mut feed, &datum, &mut bake, &mut rec);
        }
        assert!(live);
        assert!(rec.events.iter().all(|e| !matches!(e, Ev::Install(PREVIEW))));
        assert_eq!(rec.events.iter().filter(|e| matches!(e, Ev::Install(FULL))).count(), 1);
        feed.clear(&mut rec);
        assert_eq!(rec.events.last(), Some(&Ev::Clear));

        let mut again = PlanetBake::default();
        for (i, bytes) in six(3).into_iter().enumerate() {
            again.deliver(true, i, bytes.unwrap());
        }
        let mut feed = MapFeed::default();
        for _ in 0..6 {
            feed.flush(2, &datum, &mut again, &mut rec);
        }
        assert_eq!(rec.events.iter().filter(|e| matches!(e, Ev::Install(FULL))).count(), 2);

        let mut partial = PlanetBake::default();
        for (i, bytes) in six(1).into_iter().enumerate() {
            partial.deliver(false, i, bytes.unwrap());
        }
        for i in 0..3 {
            partial.deliver(true, i, vec![9, 9, 9, i as u8]);
        }
        let mut feed = MapFeed::default();
        let mut rec = Rec::default();
        for _ in 0..8 {
            step(&mut feed, &datum, &mut partial, &mut rec);
        }
        assert!(rec.events.contains(&Ev::Install(PREVIEW)));
        assert!(rec.events.iter().all(|e| !matches!(e, Ev::Install(FULL))), "three cached faces stay previews");
    }

    #[test]
    fn direction_round_trips_every_face() {
        let n = surface_n(HOME_RADIUS);
        assert!(n > 2 && n % 2 == 0);
        let half = (n / 2) as i32;
        for face in 0..6 {
            let centre = column_direction(face, n, 0, 0);
            assert_eq!(column_of(n, centre), (face, 0, 0));
            assert_eq!(column_of(n, cube_texel_dir(face, 1, 0, 0)), (face, 0, 0));
            let (xi, eta) = params_of(face, centre);
            assert!(xi.abs() < 1e-12 && eta.abs() < 1e-12, "face {face} centre params {xi} {eta}");
            for (u, v) in [(half, half), (half, -half), (-half, half), (-half, -half), (0, 0)] {
                let dir = column_direction(face, n, u, v);
                let (xi, eta) = params_of(face, dir);
                let want_u = 2.0 * u as f64 / n as f64;
                let want_v = 2.0 * v as f64 / n as f64;
                assert!((xi - want_u).abs() < 1e-9 && (eta - want_v).abs() < 1e-9, "face {face} ({u},{v}) -> {xi},{eta}");
                assert_eq!(round_i32(xi * n as f64 * 0.5), u);
                assert_eq!(round_i32(eta * n as f64 * 0.5), v);
                let (f2, u2, v2) = column_of(n, dir);
                let back = column_direction(f2, n, u2, v2);
                assert!(dir.dot(back) > 1.0 - 1e-8, "face {face} corner dir {}", dir.dot(back));
                assert_eq!(column_of(n, back), (f2, u2, v2));
            }
            // A datum sample's direction uses the same equiangular parameters.
            let g = 5usize;
            let step = 2.0 / (g - 1) as f64;
            for j in [0, g / 2, g - 1] {
                for i in [0, g / 2, g - 1] {
                    let d = DatumField::direction(face, g, i, j);
                    let (xi, eta) = params_of(face, d);
                    let want_u = -1.0 + i as f64 * step;
                    let want_v = -1.0 + j as f64 * step;
                    assert!((xi - want_u).abs() < 1e-9 && (eta - want_v).abs() < 1e-9);
                }
            }
        }
    }

    #[test]
    fn horizon_is_one_inside_the_ball_and_matches_the_limb() {
        let hi = 11_000.0;
        let air = AIR_TOP;
        let reach = hi + air;
        assert_eq!(horizon_sine(0.0, hi, air), 1.0);
        assert_eq!(horizon_sine(reach, hi, air), 1.0);
        assert_eq!(horizon_sine(reach - 1.0, hi, air), 1.0);
        let near = reach + 1_000.0;
        let far = reach + 50_000.0;
        let a = horizon_sine(near, hi, air);
        let b = horizon_sine(far, hi, air);
        let expect = |dist: f64| {
            let ratio = reach / dist;
            (-(1.0 - ratio * ratio).sqrt()) as f32
        };
        assert_eq!(a.to_bits(), expect(near).to_bits());
        assert_eq!(b.to_bits(), expect(far).to_bits());
        // The shader culls `dot(ray, -dir) > horizon`. Toward the centre that dot is -1
        // (kept); away from it the dot is +1 (culled). Farther out the limb sine is more negative.
        assert!(a < 0.0 && b < a, "near {a} far {b}");
        assert!(!(-1.0 > a));
        assert!(1.0 > a);
    }

    /// No datum sample, and no point one air-thickness above it, is culled from an eye outside the ball.
    #[test]
    fn horizon_keeps_every_datum_sample_and_its_air() {
        let g = 5usize;
        let mut offsets = vec![0.0f32; 6 * g * g];
        offsets[0] = 1_000.0;
        offsets[g * g + 3] = 400.0;
        offsets[3 * g * g + 7] = -50.0;
        let mut dirs = Vec::with_capacity(offsets.len());
        for f in 0..6 {
            for j in 0..g {
                for i in 0..g {
                    dirs.push(DatumField::direction(f, g, i, j));
                }
            }
        }
        let centre = DVec3::new(10.0, -20.0, 5.0);
        let radius = 10_000.0;
        let hi_off = offsets.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let hi = radius + hi_off as f64;
        let air = 2_000.0;
        let reach = hi + air;
        let check = |eye: DVec3| {
            let to_body = centre - eye;
            let dist = to_body.length();
            assert!(dist > reach, "eye must be outside the ball");
            let body_dir = to_body / dist;
            let horizon = horizon_sine(dist, hi, air) as f64;
            for (d, o) in dirs.iter().zip(&offsets) {
                for extra in [0.0, air] {
                    let p = centre + *d * (radius + *o as f64 + extra);
                    let v = p - eye;
                    let ray = v / v.length();
                    let dot = ray.dot(-body_dir);
                    assert!(
                        dot <= horizon + 1e-5,
                        "culled dot {dot} horizon {horizon} extra {extra} eye {eye}"
                    );
                }
            }
        };
        for axis in 0..3 {
            for sign in [-1.0, 1.0] {
                for spread in [0.0, 0.35, -0.5] {
                    let mut d = DVec3::ZERO;
                    d[axis] = sign;
                    d[(axis + 1) % 3] = spread;
                    let dir = d.normalize();
                    for scale in [1.01, 1.25, 4.0, 30.0] {
                        check(centre + dir * (reach * scale));
                    }
                }
            }
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("pwc-planet-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

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

    #[test]
    fn a_worldgen_knob_changes_the_cache_key() {
        let mut registry = BlockRegistry::with_builtins();
        let terrain = Terrain::new(&mut registry, 42);
        let (body, seed, n) = home_chart_n(&terrain).unwrap();
        assert!(n >= 2);
        let base = map_key(&terrain, &registry, WorldgenKind::Diffusion, TerrainCfg::default()).unwrap();
        assert_eq!(base.body, body);
        assert_eq!(base.seed, seed);
        assert_eq!(base.version, BAKE_VERSION);
        let relief = map_key(&terrain, &registry, WorldgenKind::Diffusion, TerrainCfg { relief: 150, ..TerrainCfg::default() }).unwrap();
        let variety = map_key(&terrain, &registry, WorldgenKind::Diffusion, TerrainCfg { variety: 50, ..TerrainCfg::default() }).unwrap();
        assert_ne!(base.hash, relief.hash, "relief");
        assert_ne!(base.hash, variety.hash, "variety");
        assert_ne!(relief.hash, variety.hash);
    }

    /// The bake cache is keyed on these: a change re-bakes every cached map.
    #[test]
    fn the_world_fingerprint_and_the_cache_key_are_pinned() {
        let registry = BlockRegistry::with_builtins();
        let knobs = TerrainCfg { relief: 150, caves: 50, deep: 25, ..TerrainCfg::default() };
        let fp = [
            world_fingerprint(&registry, WorldgenKind::Flat, TerrainCfg::default()),
            world_fingerprint(&registry, WorldgenKind::Flat, knobs),
            world_fingerprint(&registry, WorldgenKind::Diffusion, TerrainCfg::default()),
            world_fingerprint(&registry, WorldgenKind::Diffusion, knobs),
        ];
        assert_eq!(fp, [0x2ae6_d587_2d8b_db9a, 0x6ce5_8e44_4cb6_e65b, 0x7cdc_238a_9080_c3e4, 0x725c_01ff_99cd_ef75]);
        let mut registry = BlockRegistry::with_builtins();
        let terrain = Terrain::new(&mut registry, 42);
        let key = map_key(&terrain, &registry, WorldgenKind::Diffusion, knobs).unwrap();
        assert_eq!((key.seed, key.body, key.hash), (42, 0, 0xc0d8_998d_8559_c507));
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

    #[test]
    fn a_benchmark_bakes_only_when_opted_in() {
        assert!(!bake_enabled(false, true, false), "a benchmark stays a sphere");
        assert!(bake_enabled(false, true, true), "WATT_BENCH_PLANET_MAP=1 bakes");
        assert!(bake_enabled(false, false, false), "a normal game bakes");
        assert!(!bake_enabled(true, false, true), "a scripted game never bakes");
    }

    fn home_sample(terrain: &Terrain, colors: &[Color], face: usize, u: i32, v: i32) -> [f32; 3] {
        let (surface, trees, leaf) = terrain.home_far_column(face, u, v).unwrap();
        column_linear(colors, surface, trees, leaf)
    }

    #[test]
    fn spawn_face_bake_matches_the_far_field() {
        let mut registry = BlockRegistry::with_builtins();
        let terrain = Terrain::new(&mut registry, 42);
        let colors = registry.color_snapshot();
        let cosmos = terrain.cosmos().unwrap();
        let body = cosmos.home();
        let atlas = terrain.atlases().iter().find(|a| a.datum.is_some()).unwrap();
        let field = atlas.datum.as_ref().unwrap();
        let n = atlas.bands[0].n;
        let (res, datum) = impostor_datum(cosmos, body, field).unwrap();
        assert_eq!(res as usize, field.g);
        assert_eq!(datum.len(), 6 * field.g * field.g);
        let lo = cosmos.relief_range(body).0.min(cosmos.face_offset(body));
        let g = field.g;
        let mid = g / 2;
        let idx = 2 * g * g + mid * g + mid;
        assert_eq!(datum[idx], field.offsets[idx] - lo as f32);
        let radius = super::super::bodies::home_impostor(cosmos, body).1;
        let sample_r = radius + datum[idx] as f64;
        let physical = HOME_RADIUS as f64 + field.offsets[idx] as f64 - 150.0;
        assert!((sample_r - physical).abs() < 1.0, "{sample_r} vs {physical}");

        let (surface, trees, leaf) = terrain.home_far_column(2, 0, 0).unwrap();
        assert_eq!(column_of(n, DVec3::Y), (2, 0, 0));
        let raw = colors[surface.0 as usize];
        assert!(raw.g > raw.r && raw.g > raw.b, "spawn surface {:?}", raw);
        let blended = column_linear(&colors, surface, trees, leaf);
        let q = Rgb::linear(blended[0], blended[1], blended[2]).to_srgb8();
        assert!(q.g > q.r && q.g > q.b, "blended spawn {:?}", q);

        let sample = |f, u, v| home_sample(&terrain, &colors, f, u, v);
        let face = 2usize;
        let a = bake_face(16, face, n, hold_cancel(), sample).unwrap();
        let b = bake_face(16, face, n, hold_cancel(), sample).unwrap();
        assert_eq!(a, b);
        let ss = 32u32;
        for y in 0..16u32 {
            for x in 0..16u32 {
                let mut acc = [0.0f32; 3];
                for sy in 0..2u32 {
                    for sx in 0..2u32 {
                        let d = cube_texel_dir(face, ss, x * 2 + sx, y * 2 + sy);
                        let (f, u, v) = column_of(n, d);
                        let (u, v) = clamp_column(n, u, v);
                        let c = home_sample(&terrain, &colors, f, u, v);
                        acc[0] += c[0];
                        acc[1] += c[1];
                        acc[2] += c[2];
                    }
                }
                let pix = Rgb::linear(acc[0] * 0.25, acc[1] * 0.25, acc[2] * 0.25).to_srgb8();
                let i = (y as usize * 16 + x as usize) * 4;
                assert_eq!(&a[i..i + 4], &[pix.r, pix.g, pix.b, 255], "texel {x},{y}");
            }
        }

        let centre = body.centre_f();
        let mut sky = super::super::Sky::new();
        sky.warm_far(&terrain, centre);
        let handle = sky.planet().unwrap();
        assert_eq!(handle.datum_res, res);
        assert_eq!(handle.datum.len(), datum.len());
        assert_eq!(handle.datum[idx], datum[idx]);
        assert!((handle.radius - radius).abs() < 1e-6);
        assert_eq!(handle.horizon, 1.0);
        assert!(handle.preview.iter().all(|f| f.is_none()), "nothing is baked yet");
        assert!(handle.full.iter().all(|f| f.is_none()));

        let mut far = super::super::bodies::FarBodies::default();
        far.update(&terrain, centre);
        assert_eq!(far.horizon(), 1.0);
        let hi = radius + datum.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let eye_near = centre + DVec3::Y * (hi + AIR_TOP + 1_000.0);
        let eye_far = centre + DVec3::Y * (hi + AIR_TOP + 50_000.0);
        far.update(&terrain, eye_near);
        let h_near = far.horizon();
        assert_eq!(h_near.to_bits(), horizon_sine((centre - eye_near).length(), hi, AIR_TOP).to_bits());
        assert!(h_near < 0.0);
        alloc_count::reset();
        far.update(&terrain, eye_near);
        let shifted = eye_near + DVec3::X * 1_000.0;
        far.update(&terrain, shifted);
        assert_eq!(alloc_count::alloc_count(), 0, "horizon refresh allocated");
        assert_eq!(far.horizon().to_bits(), horizon_sine((centre - shifted).length(), hi, AIR_TOP).to_bits());
        far.update(&terrain, eye_far);
        let h_far = far.horizon();
        assert_eq!(h_far.to_bits(), horizon_sine((centre - eye_far).length(), hi, AIR_TOP).to_bits());
        assert!(h_far < h_near, "near {h_near} far {h_far}");
        alloc_count::reset();
        let further = eye_far + DVec3::X * 1001.0;
        far.update(&terrain, further);
        assert_eq!(alloc_count::alloc_count(), 0);
        assert_eq!(far.horizon().to_bits(), horizon_sine((centre - further).length(), hi, AIR_TOP).to_bits());
    }

    struct Tint(u16);

    struct Slow {
        hits: std::sync::Arc<AtomicUsize>,
    }

    impl TerrainGenerator for Slow {
        fn height(&self, _: i32, _: i32) -> i32 {
            0
        }
        fn surface_at(&self, _: i32, _: i32) -> BlockId {
            BlockId(1)
        }
        fn deep(&self) -> BlockId {
            AIR
        }
        fn seed(&self) -> i64 {
            1
        }
        fn home_far_column(&self, _: usize, _: i32, _: i32) -> Option<(BlockId, f32, BlockId)> {
            self.hits.fetch_add(1, Ordering::Relaxed);
            std::thread::sleep(Duration::from_millis(5));
            Some((BlockId(1), 0.0, AIR))
        }
    }

    impl TerrainGenerator for Tint {
        fn height(&self, _: i32, _: i32) -> i32 {
            0
        }
        fn surface_at(&self, _: i32, _: i32) -> BlockId {
            BlockId(self.0)
        }
        fn deep(&self) -> BlockId {
            AIR
        }
        fn seed(&self) -> i64 {
            7
        }
        fn home_far_column(&self, _: usize, _: i32, _: i32) -> Option<(BlockId, f32, BlockId)> {
            Some((BlockId(self.0), 0.0, AIR))
        }
    }

    #[test]
    fn the_baker_loads_the_cache_and_a_quiet_poll_allocates_nothing() {
        let dir = scratch("bake");
        let mut colors = vec![Color::rgb(0, 0, 0); 3].into_boxed_slice();
        colors[1] = Color::rgb(20, 160, 40);
        colors[2] = Color::rgb(180, 20, 20);
        let key = Key { seed: 7, body: 1, hash: color_hash(&colors), version: BAKE_VERSION };
        let mut bake = PlanetBake::default();
        alloc_count::reset();
        bake.poll();
        assert_eq!(alloc_count::alloc_count(), 0);
        bake.start_with(std::sync::Arc::new(Tint(1)), colors.clone(), 64, key, 2, 4, dir.clone());
        bake.finish();
        assert!(bake.faces(false).iter().all(|f| f.is_some_and(|b| b.len() == 2 * 2 * 4)));
        let saved: Vec<Vec<u8>> = (0..6).map(|i| bake.faces(true)[i].unwrap().to_vec()).collect();
        assert!(saved.iter().all(|b| b.len() == 4 * 4 * 4 && b[1] == 160));
        alloc_count::reset();
        bake.poll();
        assert_eq!(alloc_count::alloc_count(), 0, "quiet poll allocated");

        let mut again = PlanetBake::default();
        again.start_with(std::sync::Arc::new(Tint(2)), colors, 64, key, 2, 4, dir.clone());
        again.finish();
        assert!(again.faces(false).iter().all(|f| f.is_none()), "a full cache skips the preview");
        for face in 0..6 {
            assert_eq!(again.faces(true)[face], Some(saved[face].as_slice()), "face {face} loaded");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ensure_without_a_chart_is_started_and_not_retried() {
        let registry = BlockRegistry::with_builtins();
        let mut bake = PlanetBake::default();
        let chart = std::sync::Arc::new(Tint(1));
        bake.ensure(chart.clone(), &registry, WorldgenKind::Flat, TerrainCfg::default());
        assert!(bake.started());
        bake.ensure(chart, &registry, WorldgenKind::Flat, TerrainCfg::default());
        assert!(bake.started());
        bake.poll();
        assert!(bake.faces(false).iter().all(|f| f.is_none()));
        assert!(bake.faces(true).iter().all(|f| f.is_none()));
    }

    #[test]
    fn dropping_a_bake_mid_face_returns_and_leaves_no_tmp() {
        let dir = scratch("drop");
        let colors = vec![Color::rgb(0, 0, 0); 2].into_boxed_slice();
        let key = Key { seed: 1, body: 1, hash: 1, version: BAKE_VERSION };
        let hits = std::sync::Arc::new(AtomicUsize::new(0));
        let mut bake = PlanetBake::default();
        bake.start_with(std::sync::Arc::new(Slow { hits: hits.clone() }), colors, 8, key, 0, 8, dir.clone());
        let started = std::time::Instant::now();
        while hits.load(Ordering::Relaxed) == 0 {
            assert!(started.elapsed() < Duration::from_secs(5), "the bake never sampled");
            std::thread::sleep(Duration::from_millis(1));
        }
        let dropping = std::time::Instant::now();
        drop(bake);
        assert!(dropping.elapsed() < Duration::from_secs(2), "drop took {:?}", dropping.elapsed());
        assert!(hits.load(Ordering::Relaxed) < 8 * 8 * 4, "the face ran to completion");
        if dir.exists() {
            for ent in std::fs::read_dir(&dir).unwrap().flatten() {
                let name = ent.file_name();
                let name = name.to_string_lossy();
                assert!(!name.ends_with(".tmp"), "tmp left behind: {name}");
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn home_is_mapped_once_its_datum_is_uploaded_and_leaving_clears_it() {
        let mut registry = BlockRegistry::with_builtins();
        let terrain = Terrain::new(&mut registry, 42);
        let cosmos = terrain.cosmos().unwrap();
        let home = cosmos.home();
        let eye = DVec3::new(1.0e8, 0.0, 0.0);
        let mut sky = super::super::Sky::new();
        let mut rec = Rec::default();

        let before = sky.far_at(&terrain, eye);
        let before_home = before.iter().find(|b| b.seed == home.seed).expect("home from afar");
        assert!(
            matches!(before_home.shape, FarShape::Sphere),
            "home is a sphere until the datum is uploaded"
        );

        sky.sync_far_map(&mut rec, &terrain);
        assert!(rec.events.is_empty(), "no face has arrived, so nothing is installed");
        assert!(
            matches!(sky.far_at(&terrain, eye).iter().find(|b| b.seed == home.seed).unwrap().shape, FarShape::Sphere),
            "home stays a sphere until every preview has been sent"
        );

        let px = [9u8, 8, 7, 255];
        for face in 0..6 {
            sky.deliver_face(false, face, px.to_vec());
        }
        let mut mapped_now = false;
        for _ in 0..6 {
            sky.sync_far_map(&mut rec, &terrain);
            mapped_now = matches!(
                sky.far_at(&terrain, eye).iter().find(|b| b.seed == home.seed).unwrap().shape,
                FarShape::Mapped { .. }
            );
        }
        assert!(mapped_now, "six previews sent: home draws mapped");
        assert_eq!(rec.events.iter().filter(|e| matches!(e, Ev::Face(_, _))).count(), 6);
        assert_eq!(rec.events.iter().filter(|e| matches!(e, Ev::Install(PREVIEW))).count(), 1);

        let listed = sky.far_at(&terrain, eye);
        let mapped = listed.iter().find(|b| b.seed == home.seed).expect("home stays visible");
        let radius = super::super::bodies::home_impostor(cosmos, home).1;
        match mapped.shape {
            FarShape::Mapped { map, horizon, air } => {
                assert_eq!(map, HOME_MAP);
                assert!(horizon.is_finite() && horizon < 0.0, "horizon {horizon}");
                // Engine FarShape::Mapped: air is the air-shell thickness in the same unit as radius.
                assert_eq!(air.to_bits(), (AIR_TOP as f32).to_bits(), "air is AIR_TOP blocks, like radius");
                assert!((radius as f32) > air, "the reference radius is the same unit, and much larger");
            }
            other => panic!("home draws mapped once the previews are sent, got {other:?}"),
        }
        assert_eq!(mapped.radius.to_bits(), (radius as f32).to_bits(), "the reference radius stays under the ground");
        let want = NEW_SHOKA.get(Role::Horizon, Anchor::Day).to_linear();
        assert_eq!(mapped.atmosphere.0, want.0, "daylight air; moons stay black");
        assert!(listed.iter().filter(|b| b.seed != home.seed).all(|b| !matches!(b.shape, FarShape::Mapped { .. })));
        let mut moons = 0;
        for body in cosmos.bodies().iter().filter(|b| b.kind == Kind::Moon) {
            let Some(moon) = listed.iter().find(|b| b.seed == body.seed) else { continue };
            assert_eq!(moon.shape, FarShape::Sphere);
            assert_eq!(moon.atmosphere.0, [0.0, 0.0, 0.0]);
            moons += 1;
        }
        assert!(moons > 0, "a moon is still drawn");
        assert!(
            sky.far_at(&terrain, home.centre_f()).iter().all(|b| b.seed != home.seed),
            "an eye inside the reference sphere drops the body"
        );

        let full = [1u8, 2, 3, 255];
        for face in 0..5 {
            sky.deliver_face(true, face, full.to_vec());
            sky.sync_far_map(&mut rec, &terrain);
            assert!(
                matches!(sky.far_at(&terrain, eye).iter().find(|b| b.seed == home.seed).unwrap().shape, FarShape::Mapped { .. }),
                "partial full faces keep the preview"
            );
        }
        assert_eq!(rec.events.iter().filter(|e| matches!(e, Ev::Install(FULL))).count(), 0);
        sky.deliver_face(true, 5, full.to_vec());
        // The full cube installs and its faces go out one per frame. Home is a sphere until all six are sent.
        let mut back = false;
        for _ in 0..6 {
            sky.sync_far_map(&mut rec, &terrain);
            back = matches!(
                sky.far_at(&terrain, eye).iter().find(|b| b.seed == home.seed).unwrap().shape,
                FarShape::Mapped { .. }
            );
        }
        assert!(back, "six full faces sent");
        assert_eq!(rec.events.iter().filter(|e| matches!(e, Ev::Install(FULL))).count(), 1);
        assert_eq!(rec.events.iter().filter(|e| matches!(e, Ev::Face(_, _))).count(), 12);

        let n = rec.events.len();
        let _ = voxel_engine::profile::is_enabled();
        alloc_count::reset();
        sky.sync_far_map(&mut rec, &terrain);
        assert_eq!(alloc_count::alloc_count(), 0, "a quiet frame allocated");
        assert_eq!(rec.events.len(), n);

        sky.release_far_map(&mut rec);
        assert_eq!(rec.events.last(), Some(&Ev::Clear));
        let after = sky.far_at(&terrain, eye);
        let after_home = after.iter().find(|b| b.seed == home.seed).expect("home");
        assert!(matches!(after_home.shape, FarShape::Sphere), "a cleared map draws the sphere");
    }

    /// `cargo test --release --lib bake_timing -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bake_timing() {
        let mut registry = BlockRegistry::with_builtins();
        let terrain = Terrain::new(&mut registry, 42);
        let colors = registry.color_snapshot();
        let n = terrain.atlases().iter().find(|a| a.datum.is_some()).unwrap().bands[0].n;
        let sample = |f, u, v| home_sample(&terrain, &colors, f, u, v);
        let reps = 20_000u32;
        let mut acc = 0u32;
        let t0 = std::time::Instant::now();
        for i in 0..reps {
            let c = sample(2, (i.wrapping_mul(1024)) as i32, (i.wrapping_mul(17)) as i32);
            acc = acc.wrapping_add(c[1].to_bits());
        }
        let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
        println!("planet-map sample: {us:.3} µs  acc {acc}");
        for size in [256u32, 1024] {
            let t = std::time::Instant::now();
            let face = bake_face(size, 2, n, hold_cancel(), sample).unwrap();
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let samples = size as u64 * size as u64 * 4;
            println!("planet-map face {size}²: {ms:.1} ms  ({samples} samples, {} bytes)", face.len());
            std::hint::black_box(face);
        }
    }
}
