//! Bake a home body's albedo cube map, and hand that map to the sky impostor.
//!
//! Texel directions are the engine's [`far_cube_texel_dir`](voxel_engine::far_cube_texel_dir),
//! so a baked face is the cube the sky samples. The draw uploads the rebased datum once, then
//! each face as it arrives: a 256² preview, replaced by the 1024² face.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};
use std::thread::JoinHandle;

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

/// Bump when the bake bytes change. Stale files are ignored.
pub(crate) const BAKE_VERSION: u32 = 2;
/// First pass, every face, so a far body has a colour before the full map lands.
pub(crate) const PREVIEW: u32 = 256;
/// Full face. Six of these are the cached map.
pub(crate) const FULL: u32 = 1024;
/// Squared eye movement (blocks) that refreshes the horizon. Exactly 1 km does not.
pub(crate) const HORIZON_STEP_SQ: f64 = 1000.0 * 1000.0;

const MAGIC: &[u8; 4] = b"PWCM";
const HEADER: usize = 32;

/// `(face, size, x, y) -> body-space direction of that texel centre`.
pub(crate) type TexelDir = fn(usize, u32, u32, u32) -> DVec3;

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
pub(crate) fn bake_face(
    size: u32,
    face: usize,
    n: i64,
    dir: TexelDir,
    mut sample: impl FnMut(usize, i32, i32) -> [f32; 3],
) -> Vec<u8> {
    let side = size.max(1);
    let texels = (side as usize).saturating_mul(side as usize);
    let mut out = vec![0u8; texels.saturating_mul(4)];
    let ss = side.saturating_mul(2);
    for y in 0..side {
        for x in 0..side {
            let mut acc = [0.0f32; 3];
            for sy in 0..2u32 {
                for sx in 0..2u32 {
                    let d = dir(face, ss, x * 2 + sx, y * 2 + sy);
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
    out
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

/// Sine of the highest elevation of any datum sample above the plane through the eye
/// normal to eye→body. 1 when the eye is inside the outermost sample.
pub(crate) fn horizon_sin(centre: DVec3, radius: f64, dirs: &[DVec3], offsets: &[f32], eye: DVec3) -> f32 {
    let to_body = centre - eye;
    let dist = to_body.length();
    if !(dist > 0.0) || !dist.is_finite() {
        return 1.0;
    }
    let hi_off = offsets.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let hi = radius + hi_off as f64;
    if dist <= hi {
        return 1.0;
    }
    let n_up = (eye - centre) / dist;
    let mut best = f64::NEG_INFINITY;
    for (dir, off) in dirs.iter().zip(offsets) {
        let s = centre + *dir * (radius + *off as f64);
        let v = s - eye;
        let len = v.length();
        if !(len > 0.0) || !len.is_finite() {
            continue;
        }
        let elev = v.dot(n_up) / len;
        if elev > best {
            best = elev;
        }
    }
    if best.is_finite() { best as f32 } else { 1.0 }
}

/// FNV-1a 64 over every channel of the colour snapshot.
pub(crate) fn color_hash(colors: &[Color]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for c in colors {
        for b in [c.r, c.g, c.b, c.a] {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    h
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

/// Write one face. A short or mismatched buffer is not written. Corrupt files are replaced
/// only when a later bake calls this again.
fn cache_save(dir: &Path, key: &Key, size: u32, face: usize, rgba: &[u8]) {
    let expect = (size as usize).saturating_mul(size as usize).saturating_mul(4);
    if rgba.len() != expect || face > u16::MAX as usize {
        return;
    }
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let path = cache_path(dir, key, size, face);
    let tmp = path.with_extension("tmp");
    let mut bytes = Vec::with_capacity(HEADER + rgba.len());
    bytes.extend_from_slice(MAGIC);
    push_u32(&mut bytes, key.version);
    bytes.extend_from_slice(&key.seed.to_le_bytes());
    bytes.extend_from_slice(&key.body.to_le_bytes());
    bytes.extend_from_slice(&(face as u16).to_le_bytes());
    push_u32(&mut bytes, size);
    bytes.extend_from_slice(&key.hash.to_le_bytes());
    bytes.extend_from_slice(rgba);
    if std::fs::write(&tmp, &bytes).is_err() {
        return;
    }
    if std::fs::rename(&tmp, &path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

/// The face bytes, or `None` when the file is missing, short, or keyed differently.
fn cache_load(dir: &Path, key: &Key, size: u32, face: usize) -> Option<Vec<u8>> {
    let bytes = std::fs::read(cache_path(dir, key, size, face)).ok()?;
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
    {
        return None;
    }
    Some(bytes[HEADER..].to_vec())
}

struct Job {
    generator: Generator,
    colors: Box<[Color]>,
    key: Key,
    n: i64,
    dir: TexelDir,
    cache: PathBuf,
    preview: u32,
    full: u32,
}

#[cfg(not(test))]
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

fn worker(job: Job, tx: Sender<Msg>) {
    let mut have = [false; 6];
    for face in 0..6 {
        if let Some(rgba) = cache_load(&job.cache, &job.key, job.full, face) {
            have[face] = true;
            if tx.send(Msg { full: true, face, rgba }).is_err() {
                return;
            }
        }
    }
    if have.iter().all(|h| *h) {
        return;
    }
    if job.preview > 0 && job.preview != job.full {
        for face in 0..6 {
            if have[face] {
                continue;
            }
            let rgba = bake_face(job.preview, face, job.n, job.dir, |f, u, v| sample_linear(&job, f, u, v));
            if tx.send(Msg { full: false, face, rgba }).is_err() {
                return;
            }
        }
    }
    for face in 0..6 {
        if have[face] {
            continue;
        }
        let rgba = bake_face(job.full, face, job.n, job.dir, |f, u, v| sample_linear(&job, f, u, v));
        cache_save(&job.cache, &job.key, job.full, face, &rgba);
        if tx.send(Msg { full: true, face, rgba }).is_err() {
            return;
        }
    }
}

/// Faces the mapped impostor will upload, plus the datum and horizon already on [`super::bodies::FarBodies`].
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
    preview: [Option<Vec<u8>>; 6],
    full: [Option<Vec<u8>>; 6],
}

impl Default for PlanetBake {
    fn default() -> Self {
        Self {
            started: false,
            worker: None,
            rx: None,
            preview: [None, None, None, None, None, None],
            full: [None, None, None, None, None, None],
        }
    }
}

impl PlanetBake {
    pub(crate) fn started(&self) -> bool {
        self.started
    }

    /// Preview (`full == false`) or full faces currently in hand. Missing faces are `None`.
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
    #[cfg(not(test))]
    pub(crate) fn ensure(&mut self, generator: Generator, registry: &crate::block::registry::BlockRegistry) {
        if self.started {
            return;
        }
        let Some((body, seed, n)) = home_chart_n(generator.as_ref()) else {
            self.started = true;
            return;
        };
        let colors = registry.color_snapshot();
        let key = Key { seed, body, hash: color_hash(&colors), version: BAKE_VERSION };
        let cache = crate::paths::Paths::get().data.join("planet-maps");
        self.launch(Job { generator, colors, key, n, dir: cube_texel_dir, cache, preview: PREVIEW, full: FULL });
    }

    fn launch(&mut self, job: Job) {
        if self.started {
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let handle = match std::thread::Builder::new().name("planet-map".into()).spawn(move || worker(job, tx)) {
            Ok(h) => h,
            Err(_) => return,
        };
        self.started = true;
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
        }
    }

    #[cfg(test)]
    fn start_with(&mut self, generator: Generator, colors: Box<[Color]>, n: i64, key: Key, preview: u32, full: u32, cache: PathBuf) {
        self.launch(Job { generator, colors, key, n, dir: cube_texel_dir, cache, preview, full });
    }

    #[cfg(test)]
    fn finish(&mut self) {
        let start = std::time::Instant::now();
        while self.worker.is_some() {
            self.poll();
            if self.worker.is_some() {
                assert!(start.elapsed().as_secs() < 30, "planet-map bake did not finish");
                std::thread::sleep(std::time::Duration::from_millis(1));
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
}

impl MapFeed {
    /// Install the datum once, then each preview face once, then replace the cube and send each
    /// full face once. `true` when the impostor should draw [`voxel_engine::FarShape::Mapped`].
    pub(super) fn flush(
        &mut self,
        datum_res: u32,
        datum: &[f32],
        preview: &[Option<&[u8]>; 6],
        full: &[Option<&[u8]>; 6],
        sink: &mut impl FarSink,
    ) -> bool {
        let any_preview = preview.iter().any(|f| f.is_some());
        let any_full = full.iter().any(|f| f.is_some());
        let all_full = full.iter().all(|f| f.is_some());
        if self.size.is_none() {
            // A cache that already holds every full face skips the preview cube.
            let size = if all_full && !any_preview { FULL } else { PREVIEW };
            if !sink.install(datum_res, datum, size) {
                return false;
            }
            self.size = Some(size);
        }
        if self.size == Some(PREVIEW) {
            self.send(false, preview, sink);
            if any_full && sink.install(datum_res, datum, FULL) {
                self.size = Some(FULL);
            }
        }
        if self.size == Some(FULL) {
            self.send(true, full, sink);
        }
        true
    }

    fn send(&mut self, full: bool, faces: &[Option<&[u8]>; 6], sink: &mut impl FarSink) {
        let sent = if full { &mut self.sent_full } else { &mut self.sent_preview };
        for i in 0..6 {
            if sent[i] {
                continue;
            }
            let Some(bytes) = faces[i] else { continue };
            if sink.face(i, bytes) {
                sent[i] = true;
            }
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
        bake_face, cache_load, cache_path, cache_save, clamp_column, color_hash, column_direction, column_linear,
        column_of, cube_texel_dir, horizon_sin, impostor_datum, round_i32, FarSink, Key, MapFeed, PlanetBake,
        BAKE_VERSION, FULL, HOME_MAP, PREVIEW,
    };
    use crate::alloc_count;
    use crate::block::registry::{AIR, BlockId, BlockRegistry};
    use crate::space::atlas::{surface_n, FACES};
    use crate::space::chart::{basis, Map};
    use crate::space::datum::DatumField;
    use crate::world::generation::TerrainGenerator;
    use crate::world::terrain::cosmos::{Kind, AIR_TOP, HOME_RADIUS};
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

    fn view(faces: &[Option<Vec<u8>>; 6]) -> [Option<&[u8]>; 6] {
        std::array::from_fn(|i| faces[i].as_deref())
    }

    #[test]
    fn preview_faces_then_full_faces_upload_once() {
        let datum = vec![0.0f32; 24];
        let mut feed = MapFeed::default();
        let mut rec = Rec::default();
        let none = [None, None, None, None, None, None];
        assert!(feed.flush(2, &datum, &none, &none, &mut rec));
        assert_eq!(rec.events, vec![Ev::Install(PREVIEW)]);

        let preview = six(1);
        let preview_ref = view(&preview);
        feed.flush(2, &datum, &preview_ref, &none, &mut rec);
        let mut expect = vec![Ev::Install(PREVIEW)];
        for i in 0..6 {
            expect.push(Ev::Face(i, 4));
        }
        assert_eq!(rec.events, expect);

        let full = six(2);
        let full_ref = view(&full);
        feed.flush(2, &datum, &preview_ref, &full_ref, &mut rec);
        expect.push(Ev::Install(FULL));
        for i in 0..6 {
            expect.push(Ev::Face(i, 4));
        }
        assert_eq!(rec.events, expect);

        alloc_count::reset();
        feed.flush(2, &datum, &preview_ref, &full_ref, &mut rec);
        assert_eq!(alloc_count::alloc_count(), 0, "a quiet upload allocated");
        assert_eq!(rec.events, expect, "each face is sent once");
    }

    #[test]
    fn a_full_cache_skips_the_preview_and_a_clear_installs_again() {
        let datum = vec![0.0f32; 24];
        let full = six(3);
        let full_ref = view(&full);
        let none = [None, None, None, None, None, None];
        let mut feed = MapFeed::default();
        let mut rec = Rec::default();
        // Both resolutions in one step: previews go out, then the cube is replaced.
        let preview = six(1);
        let preview_ref = view(&preview);
        feed.flush(2, &datum, &preview_ref, &full_ref, &mut rec);
        let mut once = vec![Ev::Install(PREVIEW)];
        for i in 0..6 {
            once.push(Ev::Face(i, 4));
        }
        once.push(Ev::Install(FULL));
        for i in 0..6 {
            once.push(Ev::Face(i, 4));
        }
        assert_eq!(rec.events, once);

        let mut cached = MapFeed::default();
        let mut rec = Rec::default();
        cached.flush(2, &datum, &none, &full_ref, &mut rec);
        let mut only = vec![Ev::Install(FULL)];
        for i in 0..6 {
            only.push(Ev::Face(i, 4));
        }
        assert_eq!(rec.events, only);
        cached.clear(&mut rec);
        assert_eq!(rec.events.last(), Some(&Ev::Clear));
        cached.flush(2, &datum, &none, &full_ref, &mut rec);
        assert_eq!(rec.events.iter().filter(|e| matches!(e, Ev::Install(FULL))).count(), 2);
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
    fn horizon_is_one_inside_and_falls_with_height() {
        let g = 5usize;
        let mut offsets = vec![0.0f32; 6 * g * g];
        offsets[0] = 1000.0;
        let mut dirs = Vec::with_capacity(offsets.len());
        for f in 0..6 {
            for j in 0..g {
                for i in 0..g {
                    dirs.push(DatumField::direction(f, g, i, j));
                }
            }
        }
        let centre = DVec3::ZERO;
        let radius = 10_000.0;
        let hi = radius + 1000.0;
        assert_eq!(horizon_sin(centre, radius, &dirs, &offsets, centre), 1.0);
        assert_eq!(horizon_sin(centre, radius, &dirs, &offsets, centre + DVec3::Y * (hi - 1.0)), 1.0);
        let near = centre + DVec3::Y * (hi + 1_000.0);
        let far = centre + DVec3::Y * (hi + 50_000.0);
        let a = horizon_sin(centre, radius, &dirs, &offsets, near);
        let b = horizon_sin(centre, radius, &dirs, &offsets, far);
        assert!(a < 1.0 && b < a, "near {a} far {b}");
        let n_up = (near - centre).normalize();
        let mut best = f64::NEG_INFINITY;
        for (d, o) in dirs.iter().zip(&offsets) {
            let s = centre + *d * (radius + *o as f64);
            let v = s - near;
            best = best.max(v.dot(n_up) / v.length());
        }
        assert_eq!(a, best as f32);
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
        bytes[0] = b'X';
        std::fs::write(&path, &bytes).unwrap();
        assert!(cache_load(&dir, &key, 2, 4).is_none());
        std::fs::write(&path, &bytes[..10]).unwrap();
        assert!(cache_load(&dir, &key, 2, 4).is_none());
        let _ = std::fs::remove_dir_all(&dir);
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
        let a = bake_face(16, face, n, cube_texel_dir, sample);
        let b = bake_face(16, face, n, cube_texel_dir, sample);
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
        let mut dirs = Vec::with_capacity(datum.len());
        for f in 0..6 {
            for j in 0..g {
                for i in 0..g {
                    dirs.push(DatumField::direction(f, g, i, j));
                }
            }
        }
        let eye_near = centre + DVec3::Y * (hi + 1_000.0);
        let eye_far = centre + DVec3::Y * (hi + 50_000.0);
        far.update(&terrain, eye_near);
        let h_near = far.horizon();
        assert_eq!(h_near, horizon_sin(centre, radius, &dirs, &datum, eye_near));
        assert!(h_near < 1.0);
        alloc_count::reset();
        far.update(&terrain, eye_near);
        far.update(&terrain, eye_near + DVec3::X * 1_000.0);
        assert_eq!(alloc_count::alloc_count(), 0, "horizon refresh allocated");
        assert_eq!(far.horizon(), h_near);
        far.update(&terrain, eye_far);
        let h_far = far.horizon();
        assert_eq!(h_far, horizon_sin(centre, radius, &dirs, &datum, eye_far));
        assert!(h_far < h_near, "near {h_near} far {h_far}");
        alloc_count::reset();
        far.update(&terrain, eye_far + DVec3::new(1001.0, 0.0, 0.0));
        assert_eq!(alloc_count::alloc_count(), 0);
        assert_eq!(far.horizon(), horizon_sin(centre, radius, &dirs, &datum, eye_far + DVec3::X * 1001.0));
    }

    struct Tint(u16);

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
        assert_eq!(rec.events, vec![Ev::Install(PREVIEW)], "the datum is installed once, at preview size");

        let listed = sky.far_at(&terrain, eye);
        let mapped = listed.iter().find(|b| b.seed == home.seed).expect("home stays visible");
        let radius = super::super::bodies::home_impostor(cosmos, home).1;
        match mapped.shape {
            FarShape::Mapped { map, horizon, air } => {
                assert_eq!(map, HOME_MAP);
                assert!(horizon.is_finite() && horizon < 1.0, "horizon {horizon}");
                assert_eq!(air.to_bits(), ((AIR_TOP / radius) as f32).to_bits(), "air is AIR_TOP / R");
                assert!(air > 0.0 && air.is_finite());
            }
            other => panic!("home draws mapped once the datum is uploaded, got {other:?}"),
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

        let px = [9u8, 8, 7, 255];
        for face in 0..6 {
            sky.deliver_face(false, face, px.to_vec());
        }
        sky.sync_far_map(&mut rec, &terrain);
        assert_eq!(
            rec.events.iter().filter(|e| matches!(e, Ev::Face(_, _))).count(),
            6,
            "each preview face once"
        );

        let full = [1u8, 2, 3, 255];
        for face in 0..6 {
            sky.deliver_face(true, face, full.to_vec());
        }
        sky.sync_far_map(&mut rec, &terrain);
        assert_eq!(
            rec.events.iter().filter(|e| matches!(e, Ev::Install(_))).copied().collect::<Vec<_>>(),
            vec![Ev::Install(PREVIEW), Ev::Install(FULL)]
        );
        let faces: Vec<_> = rec.events.iter().copied().filter(|e| matches!(e, Ev::Face(_, _))).collect();
        assert_eq!(faces.len(), 12, "preview then full, each face once");
        assert_eq!(&faces[..6], &(0..6).map(|i| Ev::Face(i, 4)).collect::<Vec<_>>()[..]);
        assert_eq!(&faces[6..], &(0..6).map(|i| Ev::Face(i, 4)).collect::<Vec<_>>()[..]);

        let n = rec.events.len();
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
            let face = bake_face(size, 2, n, cube_texel_dir, sample);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let samples = size as u64 * size as u64 * 4;
            println!("planet-map face {size}²: {ms:.1} ms  ({samples} samples, {} bytes)", face.len());
            std::hint::black_box(face);
        }
    }
}
