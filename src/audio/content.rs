//! Catalog loader. Every runtime cue question is a load-time fold and no cue
//! evaluation touches mutable state. `CueId` is a private mint — only this
//! module constructs one, so possession proves the id was admitted at load.

use std::collections::BTreeMap;
use std::fmt;
use std::marker::PhantomData;
use std::ops::RangeInclusive;
use std::path::{Component, Path, PathBuf};

use super::backend::StoredClip;
use super::backend::{ClipId, ClipStore};
use super::frame::OccurrenceId;

// Response is owned by acoustics.rs (its fixed curves live there); re-exported
// so callers referencing `content::Response` still resolve. Authored cues never
// name Voice — the runtime supplies it for voice sessions (rejected at load).
pub use super::acoustics::Response;

/// Private mint: only the loader (or [`Catalog::typed`]) constructs a `CueId`, so
/// a value in hand proves the cue was admitted at load. The phantom `M` records
/// the cue-level mode (`OneShot`/`Loop`), so a Loop cue can never reach a one-shot
/// sink (or vice versa): the seam is mode-correct by construction. Index into
/// `Catalog::cues`.
#[derive(Debug)]
pub struct CueId<M>(u16, PhantomData<M>);

// By-hand so the phantom never forces `M: Clone`/`Eq`/… bounds on holders.
impl<M> Clone for CueId<M> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<M> Copy for CueId<M> {}
impl<M> PartialEq for CueId<M> {
    fn eq(&self, o: &Self) -> bool {
        self.0 == o.0
    }
}
impl<M> Eq for CueId<M> {}
impl<M> std::hash::Hash for CueId<M> {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        self.0.hash(h);
    }
}

impl<M> CueId<M> {
    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

/// Cue-level mode markers (the phantom parameter of [`CueId`]). A cue's mode is a
/// load-time invariant — every layer agrees, checked in `parse_cue` — lifted into
/// the type so the game↔audio seam carries it.
#[derive(Debug)]
pub struct OneShot;
#[derive(Debug)]
pub struct Loop;

pub trait CueMode {
    const MODE: ClipMode;
}
impl CueMode for OneShot {
    const MODE: ClipMode = ClipMode::OneShot;
}
impl CueMode for Loop {
    const MODE: ClipMode = ClipMode::Loop;
}

// Test-only escape hatch for sibling ctor tests (frame.rs) that must build an
// Occurrence; cfg(test) keeps the private-mint invariant intact in real code.
#[cfg(test)]
impl<M> CueId<M> {
    pub(crate) const TEST: Self = CueId(0, PhantomData);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClipMode {
    OneShot,
    Loop,
}

// Catalog safety bounds. They are deliberately generous for effects while
// excluding values that underflow playback rate, overflow wall-clock expiry, or
// retain an inaudible one-shot for effectively forever.
const MIN_PITCH_SEMITONES: f32 = -48.0;
const MAX_PITCH_SEMITONES: f32 = 48.0;
const MAX_DELAY_SECONDS: f32 = 60.0;
const MAX_ONE_SHOT_SECONDS: f32 = 3_600.0;

// Catalog internals: only consumed within the crate via `Catalog::cue` (pub(crate)),
// so they carry crate-private fields (ClipId) without over-exposing them.
pub(crate) struct Layer {
    pub(crate) variants: Box<[ClipId]>, // non-empty (checked at load); ClipId is crate-private
    pub gain: RangeInclusive<f32>,
    pub pitch: RangeInclusive<f32>, // semitones, compiled to rate at start
    pub delay: RangeInclusive<f32>, // seconds
    pub mode: ClipMode,
}

pub(crate) struct Cue {
    pub layers: Box<[Layer]>, // non-empty; homogeneous mode per cue (v1)
    pub response: Response,
    pub mode: ClipMode, // cue-level invariant: all layers agree (checked at load)
    pub max_duration: f32, // computed at load (f32::INFINITY for Loop)
}

/// Immutable value at construction; the index of a cue IS its `CueId`.
pub struct Catalog {
    cues: Box<[Cue]>,
}

impl Catalog {
    /// Load + validate the whole catalog. Reads `dir/catalog.toml`, resolves
    /// each variant file relative to `dir`, and hands the decoded bytes to
    /// `clips`. Rejects every malformed shape as a `CatalogError`.
    pub(crate) fn load(
        dir: &Path,
        clips: &mut dyn ClipStore,
    ) -> Result<(Self, CueSymbols), CatalogError> {
        let manifest_path = dir.join("catalog.toml");
        let manifest =
            std::fs::read_to_string(&manifest_path).map_err(|source| CatalogError::Io {
                path: manifest_path,
                source,
            })?;
        // A read failure names the path it tried so the error can point at the missing asset.
        let mut resolve = |name: &str| -> Result<Vec<u8>, PathBuf> {
            let p = dir.join(name);
            std::fs::read(&p).map_err(|_| p)
        };
        Self::from_manifest(&manifest, &mut resolve, clips)
    }

    /// A catalog with no cues, for the muted `SoundSystem` (no disk, no device,
    /// no `ClipStore`). Every `typed` lookup returns None, so the palette resolves
    /// every role to silence.
    pub(crate) fn empty() -> (Self, CueSymbols) {
        (Catalog { cues: Box::new([]) }, CueSymbols(BTreeMap::new()))
    }

    pub(crate) fn cue<M>(&self, id: CueId<M>) -> &Cue {
        &self.cues[id.index()]
    }

    /// Mint a mode-typed id for `name` iff it exists AND its cue-level mode is
    /// `M`. The sole typed-id constructor outside the loader: possession of a
    /// `CueId<M>` therefore proves both admission (loaded) and mode (matches the
    /// sink). Used by the palette and by App's direct menu-cue lookup.
    pub fn typed<M: CueMode>(&self, symbols: &CueSymbols, name: &str) -> Option<CueId<M>> {
        let raw = symbols.raw(name)?;
        (self.cues[raw as usize].mode == M::MODE).then_some(CueId(raw, PhantomData))
    }

    /// The cue-level mode behind a raw symbol index (for the palette's
    /// mode-mismatch diagnostic).
    pub(crate) fn mode_of(&self, raw: u16) -> ClipMode {
        self.cues[raw as usize].mode
    }

    pub(crate) fn response_of(&self, raw: u16) -> Response {
        self.cues[raw as usize].response
    }

    /// Core loader, split from `load` so tests drive it with an in-memory
    /// manifest and a mock file resolver (no filesystem, no real assets).
    pub(crate) fn from_manifest(
        manifest: &str,
        resolve: &mut dyn FnMut(&str) -> Result<Vec<u8>, PathBuf>,
        clips: &mut dyn ClipStore,
    ) -> Result<(Self, CueSymbols), CatalogError> {
        let root: toml::Value = toml::from_str(manifest)
            .map_err(|e| CatalogError::Manifest(format!("toml parse: {e}")))?;
        let cues_tbl = root
            .get("cues")
            .and_then(toml::Value::as_table)
            .ok_or_else(|| CatalogError::Manifest("missing top-level [cues] table".into()))?;

        if cues_tbl.len() > u16::MAX as usize {
            return Err(CatalogError::TooManyCues);
        }

        // Sorted name order gives a stable, process-independent CueId assignment.
        let names: BTreeMap<&String, &toml::Value> = cues_tbl.iter().collect();

        let mut cues = Vec::with_capacity(names.len());
        let mut symbols = BTreeMap::new();
        // A clip referenced by several cues/layers is decoded and retained once.
        // The normalized relative name is a stable key because unsafe aliases
        // (`.`, `..`, absolute paths) are rejected before resolution.
        let mut clip_cache: BTreeMap<String, StoredClip> = BTreeMap::new();
        for (index, (name, cue_val)) in names.into_iter().enumerate() {
            let cue = parse_cue(name, cue_val, resolve, clips, &mut clip_cache)?;
            cues.push(cue);
            symbols.insert(name.clone(), index as u16);
        }

        Ok((
            Catalog {
                cues: cues.into_boxed_slice(),
            },
            CueSymbols(symbols),
        ))
    }
}

/// name → CueId map returned once at load; Game resolves its trigger constants
/// from it.
pub struct CueSymbols(BTreeMap<String, u16>);

impl CueSymbols {
    /// The raw catalog index for `name` (untyped). Only [`Catalog::typed`] turns
    /// it into a mode-typed [`CueId`]; nothing else resolves cues by name.
    pub(crate) fn raw(&self, name: &str) -> Option<u16> {
        self.0.get(name).copied()
    }
}

/// Deterministic variant/parameter selection (no mutable RNG). The exact
/// splitmix64 spec below is a contract: any implementer must reproduce these
/// draws bit-for-bit (regression-pinned in the tests). Returns a bucket in
/// `[0, n)`; range parameters map the bucket linearly at realization (mod.rs).
pub(crate) fn draw(id: OccurrenceId, layer: u16, kind: DrawKind, n: u32) -> u32 {
    debug_assert!(
        n >= 1,
        "draw n must be >= 1 (variant/param counts are load-checked)"
    );
    draw_raw(id, layer, kind) % n
}

/// The pre-modulo 32-bit draw, uniform over `[0, 2^32)`. Callers wanting a bucket
/// use `draw`; callers wanting a unit-interval sample scale this directly (range
/// realization), which avoids the `% (2^32 - 1)` bias of folding through `draw`.
pub(crate) fn draw_raw(id: OccurrenceId, layer: u16, kind: DrawKind) -> u32 {
    let x = id.0 ^ ((layer as u64) << 48) ^ ((kind as u64) << 56);
    (splitmix64(x) >> 32) as u32
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[derive(Clone, Copy)]
pub(crate) enum DrawKind {
    Variant = 0,
    Gain = 1,
    Pitch = 2,
    Delay = 3,
}

#[derive(Debug)]
pub enum CatalogError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Manifest(String),
    UnsafePath(PathBuf),
    UnknownFile(PathBuf),
    EmptyCue(String),
    EmptyLayer(String),
    BadRange(String),
    MixedMode(String),
    TooManyCues,
    TooManyLayers(String),
    TooManyVariants(String),
    Decode(String),
    InvalidDuration(String),
}

impl fmt::Display for CatalogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(f, "could not read `{}`: {source}", path.display())
            }
            Self::Manifest(reason) => write!(f, "invalid catalog: {reason}"),
            Self::UnsafePath(path) => {
                write!(
                    f,
                    "asset path `{}` must be a safe relative path",
                    path.display()
                )
            }
            Self::UnknownFile(path) => write!(f, "catalog asset `{}` is missing", path.display()),
            Self::EmptyCue(cue) => write!(f, "cue `{cue}` has no layers"),
            Self::EmptyLayer(cue) => write!(f, "cue `{cue}` has a layer with no variants"),
            Self::BadRange(reason) => write!(f, "invalid catalog range: {reason}"),
            Self::MixedMode(cue) => write!(f, "cue `{cue}` mixes one-shot and loop layers"),
            Self::TooManyCues => write!(f, "catalog contains more than {} cues", u16::MAX),
            Self::TooManyLayers(cue) => write!(f, "cue `{cue}` contains more than 255 layers"),
            Self::TooManyVariants(cue) => {
                write!(f, "cue `{cue}` has a layer with more than 256 variants")
            }
            Self::Decode(reason) => write!(f, "audio clip decode failed: {reason}"),
            Self::InvalidDuration(reason) => write!(f, "invalid audio clip duration: {reason}"),
        }
    }
}

impl std::error::Error for CatalogError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

// ---- loader interior (pure over the resolver + ClipStore) ----

fn parse_cue(
    name: &str,
    val: &toml::Value,
    resolve: &mut dyn FnMut(&str) -> Result<Vec<u8>, PathBuf>,
    clips: &mut dyn ClipStore,
    clip_cache: &mut BTreeMap<String, StoredClip>,
) -> Result<Cue, CatalogError> {
    let tbl = val
        .as_table()
        .ok_or_else(|| CatalogError::Manifest(format!("cue `{name}` is not a table")))?;

    let response = parse_response(name, tbl.get("response"))?;

    let layers_val = tbl
        .get("layers")
        .and_then(toml::Value::as_array)
        .filter(|a| !a.is_empty())
        .ok_or_else(|| CatalogError::EmptyCue(name.to_string()))?;

    // Layer index rides bits 48..=55 of the draw key (`layer << 48`); bounding it to
    // 255 keeps it clear of `kind << 56` and the emitter salt. More than 255
    // layers on one cue is a manifest error, not a runtime clamp.
    if layers_val.len() > 255 {
        return Err(CatalogError::TooManyLayers(name.to_string()));
    }

    let mut layers = Vec::with_capacity(layers_val.len());
    let mut clip_secs = Vec::with_capacity(layers_val.len()); // longest variant per layer
    for layer_val in layers_val {
        let (layer, secs) = parse_layer(name, layer_val, resolve, clips, clip_cache)?;
        layers.push(layer);
        clip_secs.push(secs);
    }

    // Homogeneous mode per cue (v1): reject any layer disagreeing with the first.
    let mode = layers[0].mode;
    if layers.iter().any(|l| l.mode != mode) {
        return Err(CatalogError::MixedMode(name.to_string()));
    }

    // A Loop cue never ends. A OneShot cue ends when its slowest layer
    // finishes: layer end = max_delay + audible_duration, where the lowest
    // authored pitch stretches playout (min_rate = 2^(pitch_lo/12)).
    let max_duration = match mode {
        ClipMode::Loop => f32::INFINITY,
        ClipMode::OneShot => layers
            .iter()
            .zip(&clip_secs)
            .map(|(l, secs)| {
                let min_rate = 2.0_f32.powf(*l.pitch.start() / 12.0);
                *l.delay.end() + secs / min_rate
            })
            .fold(0.0_f32, f32::max),
    };
    if matches!(mode, ClipMode::OneShot)
        && (!max_duration.is_finite() || max_duration > MAX_ONE_SHOT_SECONDS)
    {
        return Err(CatalogError::InvalidDuration(format!(
            "cue `{name}` lasts {max_duration:?} s; maximum is {MAX_ONE_SHOT_SECONDS} s"
        )));
    }
    if matches!(response, Response::Ui)
        && layers
            .iter()
            .any(|layer| *layer.delay.start() != 0.0 || *layer.delay.end() != 0.0)
    {
        return Err(CatalogError::BadRange(format!(
            "cue `{name}`: UI layers require `delay = [0]`"
        )));
    }

    Ok(Cue {
        layers: layers.into_boxed_slice(),
        response,
        mode,
        max_duration,
    })
}

fn parse_response(cue: &str, val: Option<&toml::Value>) -> Result<Response, CatalogError> {
    match val.and_then(toml::Value::as_str) {
        Some("world") => Ok(Response::World),
        Some("ui") => Ok(Response::Ui),
        Some("ambient") => Ok(Response::Ambient),
        // Voice cues are synthesized from live sessions, never authored.
        Some(other) => Err(CatalogError::Manifest(format!(
            "cue `{cue}`: unknown response `{other}` (expected world|ui|ambient)"
        ))),
        None => Err(CatalogError::Manifest(format!(
            "cue `{cue}`: missing string `response`"
        ))),
    }
}

fn parse_layer(
    cue: &str,
    val: &toml::Value,
    resolve: &mut dyn FnMut(&str) -> Result<Vec<u8>, PathBuf>,
    clips: &mut dyn ClipStore,
    clip_cache: &mut BTreeMap<String, StoredClip>,
) -> Result<(Layer, f32), CatalogError> {
    let tbl = val
        .as_table()
        .ok_or_else(|| CatalogError::Manifest(format!("cue `{cue}`: layer is not a table")))?;

    let variant_files = tbl
        .get("variants")
        .and_then(toml::Value::as_array)
        .filter(|a| !a.is_empty())
        .ok_or_else(|| CatalogError::EmptyLayer(cue.to_string()))?;
    if variant_files.len() > 256 {
        return Err(CatalogError::TooManyVariants(cue.to_owned()));
    }

    let mut variants = Vec::with_capacity(variant_files.len());
    let mut max_variant_s = 0.0_f32; // longest decoded clip in this layer
    for file_val in variant_files {
        let file = file_val.as_str().ok_or_else(|| {
            CatalogError::Manifest(format!("cue `{cue}`: variant entry is not a string"))
        })?;
        let relative = Path::new(file);
        if file.is_empty()
            || !relative
                .components()
                .all(|component| matches!(component, Component::Normal(_)))
        {
            return Err(CatalogError::UnsafePath(relative.to_owned()));
        }
        let StoredClip { id, duration_s } = match clip_cache.get(file).copied() {
            Some(stored) => stored,
            None => {
                let bytes = resolve(file).map_err(CatalogError::UnknownFile)?;
                let stored = clips.store(&bytes).map_err(CatalogError::Decode)?;
                if !stored.duration_s.is_finite() || stored.duration_s <= 0.0 {
                    return Err(CatalogError::InvalidDuration(format!(
                        "`{file}` decoded to {:?} s",
                        stored.duration_s
                    )));
                }
                clip_cache.insert(file.to_owned(), stored);
                stored
            }
        };
        max_variant_s = max_variant_s.max(duration_s);
        variants.push(id);
    }

    let gain = parse_range(cue, tbl.get("gain"), "gain")?;
    // Authored gain must be in (0, 4]. A non-positive gain is silence authored as
    // a voice (waste a slot); > 4 breaks the audibility upper-bound proof.
    if *gain.start() <= 0.0 || *gain.end() > 4.0 {
        return Err(CatalogError::BadRange(format!(
            "cue `{cue}`: `gain` must be in (0, 4]"
        )));
    }
    let pitch = parse_range(cue, tbl.get("pitch"), "pitch")?;
    let delay = parse_range(cue, tbl.get("delay"), "delay")?;
    if *pitch.start() < MIN_PITCH_SEMITONES || *pitch.end() > MAX_PITCH_SEMITONES {
        return Err(CatalogError::BadRange(format!(
            "cue `{cue}`: `pitch` must be within [{MIN_PITCH_SEMITONES}, {MAX_PITCH_SEMITONES}] semitones"
        )));
    }
    if *delay.start() < 0.0 || *delay.end() > MAX_DELAY_SECONDS {
        return Err(CatalogError::BadRange(format!(
            "cue `{cue}`: `delay` must be within [0, {MAX_DELAY_SECONDS}] seconds"
        )));
    }
    let mode = parse_mode(cue, tbl.get("mode"))?;

    Ok((
        Layer {
            variants: variants.into_boxed_slice(),
            gain,
            pitch,
            delay,
            mode,
        },
        max_variant_s,
    ))
}

fn parse_mode(cue: &str, val: Option<&toml::Value>) -> Result<ClipMode, CatalogError> {
    match val.and_then(toml::Value::as_str) {
        Some("one_shot") => Ok(ClipMode::OneShot),
        Some("loop") => Ok(ClipMode::Loop),
        Some(other) => Err(CatalogError::Manifest(format!(
            "cue `{cue}`: unknown mode `{other}` (expected one_shot|loop)"
        ))),
        None => Err(CatalogError::Manifest(format!(
            "cue `{cue}`: missing string `mode`"
        ))),
    }
}

/// A single-element array is a fixed value; two elements are `[lo, hi]`. Any
/// other length, a non-number, a non-finite bound, or `lo > hi` is rejected.
fn parse_range(
    cue: &str,
    val: Option<&toml::Value>,
    field: &str,
) -> Result<RangeInclusive<f32>, CatalogError> {
    let arr = val
        .and_then(toml::Value::as_array)
        .ok_or_else(|| CatalogError::Manifest(format!("cue `{cue}`: missing array `{field}`")))?;
    let nums: Vec<f32> = arr
        .iter()
        .map(as_f32)
        .collect::<Option<Vec<f32>>>()
        .ok_or_else(|| {
            CatalogError::Manifest(format!("cue `{cue}`: `{field}` has non-number entries"))
        })?;
    let (lo, hi) = match nums.as_slice() {
        [v] => (*v, *v),
        [lo, hi] => (*lo, *hi),
        _ => {
            return Err(CatalogError::Manifest(format!(
                "cue `{cue}`: `{field}` must have 1 or 2 elements"
            )));
        }
    };
    if !lo.is_finite() || !hi.is_finite() {
        return Err(CatalogError::BadRange(format!(
            "cue `{cue}`: `{field}` non-finite"
        )));
    }
    if lo > hi {
        return Err(CatalogError::BadRange(format!(
            "cue `{cue}`: `{field}` reversed"
        )));
    }
    Ok(lo..=hi)
}

fn as_f32(v: &toml::Value) -> Option<f32> {
    match v {
        toml::Value::Float(f) => Some(*f as f32),
        toml::Value::Integer(i) => Some(*i as f32),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reports each clip's duration as its byte length, so tests set a variant's
    // duration by the number of bytes their resolver returns.
    struct MockClips {
        next: u32,
    }
    impl ClipStore for MockClips {
        fn store(&mut self, bytes: &[u8]) -> Result<StoredClip, String> {
            let id = ClipId(self.next);
            self.next += 1;
            Ok(StoredClip {
                id,
                duration_s: bytes.len() as f32,
            })
        }
    }

    struct FailClips;
    impl ClipStore for FailClips {
        fn store(&mut self, _bytes: &[u8]) -> Result<StoredClip, String> {
            Err("decode boom".into())
        }
    }

    // Resolver that pretends every named file exists as a 1-byte (1 s) clip.
    fn ok_resolver() -> impl FnMut(&str) -> Result<Vec<u8>, PathBuf> {
        |_name: &str| Ok(vec![0u8])
    }

    const HAPPY: &str = r#"
        [cues.break_default]
        response = "world"
        [[cues.break_default.layers]]
        variants = ["a.wav", "b.wav"]
        gain = [0.8, 1.0]
        pitch = [-2.0, 2.0]
        delay = [0.0, 0.1]
        mode = "one_shot"

        [cues.underwater_loop]
        response = "ambient"
        [[cues.underwater_loop.layers]]
        variants = ["u.wav"]
        gain = [0.5]
        pitch = [0.0]
        delay = [0.0]
        mode = "loop"
    "#;

    #[test]
    fn happy_path_parse() {
        let mut clips = MockClips { next: 0 };
        let mut r = ok_resolver();
        let (cat, syms) = Catalog::from_manifest(HAPPY, &mut r, &mut clips).unwrap();

        let brk = cat.typed::<OneShot>(&syms, "break_default").unwrap();
        let uw = cat.typed::<Loop>(&syms, "underwater_loop").unwrap();
        // Sorted-name assignment: break_default < underwater_loop.
        assert_eq!(brk.index(), 0);
        assert_eq!(uw.index(), 1);

        let c = cat.cue(brk);
        assert_eq!(c.response, Response::World);
        assert_eq!(c.layers.len(), 1);
        assert_eq!(c.layers[0].variants.len(), 2);
        assert_eq!(c.layers[0].gain, 0.8..=1.0);
        // delay.end (0.1) + 1 s clip stretched by the lowest pitch (-2 semitones).
        let expect = 0.1 + 1.0 / 2.0_f32.powf(-2.0 / 12.0);
        assert!((c.max_duration - expect).abs() < 1e-4);

        let l = cat.cue(uw);
        assert_eq!(l.response, Response::Ambient);
        assert!(l.max_duration.is_infinite());
        assert!(syms.raw("missing").is_none());
    }

    // Regression pins: computed once from the frozen splitmix64 spec. If these
    // change, the deterministic-draw contract has been broken.
    #[test]
    fn draw_fixed_vectors() {
        assert_eq!(draw(OccurrenceId(1), 0, DrawKind::Variant, 2), 0);
        assert_eq!(draw(OccurrenceId(1), 0, DrawKind::Gain, 1000), 131);
        assert_eq!(draw(OccurrenceId(42), 3, DrawKind::Pitch, 256), 2);
        assert_eq!(draw(OccurrenceId(0), 0, DrawKind::Delay, 7), 2);
    }

    #[test]
    fn max_duration_picks_slowest_layer() {
        // Two one_shot layers; the fold must take the later-finishing one.
        //   A: 1 s clip, pitch 0 (rate 1), delay 0        -> end 1.0
        //   B: 2 s clip, pitch -12 (rate 0.5), delay 0.5  -> end 0.5 + 2/0.5 = 4.5
        let m = r#"[cues.x]
            response = "world"
            [[cues.x.layers]]
            variants = ["a.wav"]
            gain = [1.0]
            pitch = [0.0]
            delay = [0.0]
            mode = "one_shot"
            [[cues.x.layers]]
            variants = ["b.wav"]
            gain = [1.0]
            pitch = [-12.0]
            delay = [0.5]
            mode = "one_shot""#;
        let mut clips = MockClips { next: 0 };
        // Byte length = seconds: a.wav -> 1 s, b.wav -> 2 s.
        let mut r = |name: &str| -> Result<Vec<u8>, PathBuf> {
            Ok(vec![0u8; if name == "b.wav" { 2 } else { 1 }])
        };
        let (cat, syms) = Catalog::from_manifest(m, &mut r, &mut clips).unwrap();
        let c = cat.cue(cat.typed::<OneShot>(&syms, "x").unwrap());
        assert!((c.max_duration - 4.5).abs() < 1e-4);
    }

    fn expect_err(res: Result<(Catalog, CueSymbols), CatalogError>) -> CatalogError {
        match res {
            Ok(_) => panic!("expected a CatalogError"),
            Err(e) => e,
        }
    }

    fn load_err(manifest: &str) -> CatalogError {
        let mut clips = MockClips { next: 0 };
        let mut r = ok_resolver();
        expect_err(Catalog::from_manifest(manifest, &mut r, &mut clips))
    }

    #[test]
    fn rejects_missing_cues_table() {
        assert!(matches!(load_err("foo = 1\n"), CatalogError::Manifest(_)));
    }

    #[test]
    fn rejects_bad_toml() {
        assert!(matches!(
            load_err("this is not = = toml"),
            CatalogError::Manifest(_)
        ));
    }

    #[test]
    fn rejects_empty_cue() {
        let m = r#"[cues.x]
            response = "world""#;
        assert!(matches!(load_err(m), CatalogError::EmptyCue(_)));
    }

    #[test]
    fn rejects_empty_layer() {
        let m = r#"[cues.x]
            response = "world"
            [[cues.x.layers]]
            variants = []
            gain = [1.0]
            pitch = [0.0]
            delay = [0.0]
            mode = "one_shot""#;
        assert!(matches!(load_err(m), CatalogError::EmptyLayer(_)));
    }

    #[test]
    fn rejects_reversed_range() {
        let m = r#"[cues.x]
            response = "world"
            [[cues.x.layers]]
            variants = ["a.wav"]
            gain = [1.0, 0.2]
            pitch = [0.0]
            delay = [0.0]
            mode = "one_shot""#;
        assert!(matches!(load_err(m), CatalogError::BadRange(_)));
    }

    #[test]
    fn rejects_negative_delay_and_extreme_pitch() {
        let negative_delay = r#"[cues.x]
            response = "world"
            [[cues.x.layers]]
            variants = ["a.wav"]
            gain = [1.0]
            pitch = [0.0]
            delay = [-0.1]
            mode = "one_shot""#;
        assert!(matches!(
            load_err(negative_delay),
            CatalogError::BadRange(_)
        ));

        let extreme_pitch = r#"[cues.x]
            response = "world"
            [[cues.x.layers]]
            variants = ["a.wav"]
            gain = [1.0]
            pitch = [-1000.0]
            delay = [0.0]
            mode = "one_shot""#;
        assert!(matches!(load_err(extreme_pitch), CatalogError::BadRange(_)));
    }

    #[test]
    fn ui_delay_is_rejected_instead_of_silently_ignored() {
        let m = r#"[cues.x]
            response = "ui"
            [[cues.x.layers]]
            variants = ["a.wav"]
            gain = [1.0]
            pitch = [0.0]
            delay = [0.1]
            mode = "one_shot""#;
        assert!(matches!(load_err(m), CatalogError::BadRange(_)));
    }

    #[test]
    fn rejects_paths_that_escape_the_catalog_root() {
        for path in ["../secret.wav", "/tmp/secret.wav", "./alias.wav", ""] {
            let m = format!(
                r#"[cues.x]
                    response = "world"
                    [[cues.x.layers]]
                    variants = ["{path}"]
                    gain = [1.0]
                    pitch = [0.0]
                    delay = [0.0]
                    mode = "one_shot""#
            );
            assert!(matches!(load_err(&m), CatalogError::UnsafePath(_)));
        }
    }

    #[test]
    fn repeated_variant_file_is_decoded_once() {
        let m = r#"[cues.x]
            response = "world"
            [[cues.x.layers]]
            variants = ["shared.wav"]
            gain = [1.0]
            pitch = [0.0]
            delay = [0.0]
            mode = "one_shot"
            [[cues.x.layers]]
            variants = ["shared.wav"]
            gain = [0.5]
            pitch = [0.0]
            delay = [0.0]
            mode = "one_shot""#;
        let mut clips = MockClips { next: 0 };
        let mut resolve = ok_resolver();
        let (catalog, symbols) = Catalog::from_manifest(m, &mut resolve, &mut clips).unwrap();
        let cue = catalog.cue(catalog.typed::<OneShot>(&symbols, "x").unwrap());
        assert_eq!(clips.next, 1);
        assert_eq!(cue.layers[0].variants[0], cue.layers[1].variants[0]);
    }

    #[test]
    fn rejects_non_finite_range() {
        let m = r#"[cues.x]
            response = "world"
            [[cues.x.layers]]
            variants = ["a.wav"]
            gain = [1.0]
            pitch = [nan]
            delay = [0.0]
            mode = "one_shot""#;
        assert!(matches!(load_err(m), CatalogError::BadRange(_)));
    }

    #[test]
    fn rejects_mixed_mode() {
        let m = r#"[cues.x]
            response = "world"
            [[cues.x.layers]]
            variants = ["a.wav"]
            gain = [1.0]
            pitch = [0.0]
            delay = [0.0]
            mode = "one_shot"
            [[cues.x.layers]]
            variants = ["b.wav"]
            gain = [1.0]
            pitch = [0.0]
            delay = [0.0]
            mode = "loop""#;
        assert!(matches!(load_err(m), CatalogError::MixedMode(_)));
    }

    #[test]
    fn rejects_unknown_response() {
        let m = r#"[cues.x]
            response = "wobble"
            [[cues.x.layers]]
            variants = ["a.wav"]
            gain = [1.0]
            pitch = [0.0]
            delay = [0.0]
            mode = "one_shot""#;
        assert!(matches!(load_err(m), CatalogError::Manifest(_)));
    }

    #[test]
    fn rejects_voice_response() {
        // Voice cues are runtime-only; authoring one must fail.
        let m = r#"[cues.x]
            response = "voice"
            [[cues.x.layers]]
            variants = ["a.wav"]
            gain = [1.0]
            pitch = [0.0]
            delay = [0.0]
            mode = "one_shot""#;
        assert!(matches!(load_err(m), CatalogError::Manifest(_)));
    }

    #[test]
    fn rejects_unknown_file() {
        let m = r#"[cues.x]
            response = "world"
            [[cues.x.layers]]
            variants = ["ghost.wav"]
            gain = [1.0]
            pitch = [0.0]
            delay = [0.0]
            mode = "one_shot""#;
        let mut clips = MockClips { next: 0 };
        let mut r = |name: &str| -> Result<Vec<u8>, PathBuf> { Err(PathBuf::from(name)) };
        let err = expect_err(Catalog::from_manifest(m, &mut r, &mut clips));
        assert!(matches!(err, CatalogError::UnknownFile(_)));
    }

    #[test]
    fn rejects_decode_failure() {
        let m = r#"[cues.x]
            response = "world"
            [[cues.x.layers]]
            variants = ["a.wav"]
            gain = [1.0]
            pitch = [0.0]
            delay = [0.0]
            mode = "one_shot""#;
        let mut clips = FailClips;
        let mut r = ok_resolver();
        let err = expect_err(Catalog::from_manifest(m, &mut r, &mut clips));
        assert!(matches!(err, CatalogError::Decode(_)));
    }
}
