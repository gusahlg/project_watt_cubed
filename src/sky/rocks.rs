//! Distant asteroids as lit boxes. Class 2 and 3 out to 60 000 blocks, class 1
//! out to 8 000, never class 0 (pebbles). Rocks inside the chunk view are left
//! to the voxels. Each class is scanned again only after the eye moves an
//! eighth of its range; other frames reuse the buffers. A rescan (a few ms in a
//! dense swarm) runs on a background thread while the previous list keeps
//! drawing: the scan margin keeps that list valid until the new one lands.

use std::sync::Arc;
use std::thread::JoinHandle;

use voxel_engine::{Color, DVec3, Frame3D, Mat3, Vec3};

use crate::world::generation::TerrainGenerator;
use crate::world::terrain::cosmos::{Cosmos, RockKind};
use crate::world::terrain::noise::{hash3, unit};

/// Smallest drawn class (radii 22..170).
const NEAR: f64 = 8_000.0;
/// Class 2 and class 3.
const FAR: f64 = 60_000.0;
/// Rebuild a class after the eye moves this fraction of its range.
const REBUILD_FRAC: f64 = 0.125;
/// The fade occupies this tail of the range.
const FADE_TAIL: f64 = 0.15;
/// Above this altitude over a body (or with no body) the camera is in space.
const SPACE_ALTITUDE: f64 = 20_000.0;

/// Half-extent, in blocks, of a chunk view of `view_chunks` rings. One extra
/// chunk covers the eye's place inside its own chunk and the far block of the ring.
pub(super) fn chunk_view_blocks(view_chunks: i32) -> f64 {
    (view_chunks.max(0) as f64 + 1.0) * crate::world::chunk::CHUNK_SIZE as f64
}

/// One size class of rocks, reused across frames.
struct Band {
    class: usize,
    /// Draw a rock closer than this.
    range: f64,
    /// Scan past `range` by the rebuild margin, so a rock already fading in is
    /// in the buffer when the eye crosses the range between rebuilds.
    scan: f64,
    margin_sq: f64,
    built_at: Option<DVec3>,
    rocks: Vec<Cached>,
    /// A background rescan around this eye.
    pending: Option<(DVec3, JoinHandle<Vec<Cached>>)>,
}

#[derive(Clone, Copy)]
struct Cached {
    world: DVec3,
    half: Vec3,
    rot: Mat3,
    albedo: Color,
    #[cfg(test)]
    centre: [i32; 3],
    #[cfg(test)]
    seed: u32,
    #[cfg(test)]
    r: f32,
}

/// One box to draw this frame, camera-relative.
struct RockBox {
    center: Vec3,
    half: Vec3,
    rot: Mat3,
    color: Color,
    #[cfg(test)]
    centre: [i32; 3],
    #[cfg(test)]
    seed: u32,
    #[cfg(test)]
    r: f32,
}

/// Reused distant-asteroid list. Capacity stays after the first build.
pub(super) struct DistantRocks {
    bands: [Band; 3],
    shown: Vec<RockBox>,
    /// Eye and chunk-view reach the `shown` buffer was built for.
    posed: Option<(DVec3, f64)>,
    #[cfg(test)]
    scans: u32,
}

impl Default for DistantRocks {
    fn default() -> Self {
        Self {
            bands: [Band::new(1, NEAR), Band::new(2, FAR), Band::new(3, FAR)],
            shown: Vec::new(),
            posed: None,
            #[cfg(test)]
            scans: 0,
        }
    }
}

impl Band {
    fn new(class: usize, range: f64) -> Self {
        let margin = range * REBUILD_FRAC;
        Self {
            class,
            range,
            scan: range + margin,
            margin_sq: margin * margin,
            built_at: None,
            rocks: Vec::new(),
            pending: None,
        }
    }

    fn needs_rebuild(&self, eye: DVec3) -> bool {
        match self.built_at {
            None => true,
            Some(at) => (at - eye).length_squared() > self.margin_sq,
        }
    }

    fn rebuild(&mut self, cosmos: &Cosmos, eye: DVec3) {
        scan(cosmos, self.class, self.scan, eye, &mut self.rocks);
    }

    /// Adopt a finished background rescan. True when the list changed.
    fn adopt(&mut self) -> bool {
        if !self.pending.as_ref().is_some_and(|(_, h)| h.is_finished()) {
            return false;
        }
        let (at, handle) = self.pending.take().expect("checked above");
        match handle.join() {
            Ok(rocks) => {
                self.rocks = rocks;
                self.built_at = Some(at);
                true
            }
            // A panicked scan leaves the old list; the next frame asks again.
            Err(_) => false,
        }
    }
}

/// Every rock of `class` within `radius` of `eye`, into `out` (cleared first).
fn scan(cosmos: &Cosmos, class: usize, radius: f64, eye: DVec3, out: &mut Vec<Cached>) {
    out.clear();
    let (lo, hi) = cell_box(eye, radius);
    let scan_sq = radius * radius;
    cosmos.for_class_rocks(class, lo, hi, |rock| {
        let world = DVec3::new(rock.centre[0] as f64, rock.centre[1] as f64, rock.centre[2] as f64);
        if (world - eye).length_squared() > scan_sq {
            return;
        }
        out.push(Cached {
            world,
            half: Vec3::new(rock.r * rock.axes[0], rock.r * rock.axes[1], rock.r * rock.axes[2]),
            rot: rotation(rock.seed),
            albedo: kind_color(rock.kind),
            #[cfg(test)]
            centre: rock.centre,
            #[cfg(test)]
            seed: rock.seed,
            #[cfg(test)]
            r: rock.r,
        });
    });
    if out.capacity() < out.len() + 32 {
        out.reserve(32);
    }
}

impl DistantRocks {
    /// Rocks to draw around `eye`. `view_blocks` is the chunk view's reach
    /// ([`chunk_view_blocks`]): a rock inside it is a voxel, so it is not listed.
    /// Empty inside a body's atmosphere. With a `shared` catalog a rescan runs in the
    /// background (the old list draws meanwhile); without one it runs here.
    fn update(
        &mut self,
        cosmos: &Cosmos,
        shared: &dyn Fn() -> Option<Arc<Cosmos>>,
        eye: DVec3,
        view_blocks: f64,
    ) -> &[RockBox] {
        if !in_space(cosmos, eye) {
            self.shown.clear();
            self.posed = None;
            return &self.shown;
        }
        let mut changed = false;
        for band in &mut self.bands {
            changed |= band.adopt();
        }
        let wants = self.bands.iter().any(|b| b.pending.is_none() && b.needs_rebuild(eye));
        if !changed && !wants && self.posed == Some((eye, view_blocks)) {
            return &self.shown;
        }
        let mut handle = None;
        let mut rebuilt = 0u32;
        for band in &mut self.bands {
            if band.pending.is_some() || !band.needs_rebuild(eye) {
                continue;
            }
            if handle.is_none() {
                handle = Some(shared());
            }
            match handle.as_ref().expect("set above") {
                Some(arc) => {
                    let (arc, class, radius) = (arc.clone(), band.class, band.scan);
                    let spawned = std::thread::Builder::new().name("rock-scan".into()).spawn(move || {
                        let mut out = Vec::new();
                        scan(&arc, class, radius, eye, &mut out);
                        out
                    });
                    match spawned {
                        Ok(h) => band.pending = Some((eye, h)),
                        Err(_) => {
                            band.rebuild(cosmos, eye);
                            band.built_at = Some(eye);
                        }
                    }
                }
                None => {
                    band.rebuild(cosmos, eye);
                    band.built_at = Some(eye);
                }
            }
            rebuilt += 1;
        }
        #[cfg(test)]
        {
            self.scans += rebuilt;
        }
        #[cfg(not(test))]
        let _ = rebuilt;
        self.present(eye, view_blocks);
        self.posed = Some((eye, view_blocks));
        &self.shown
    }

    /// Draw the list. Immediate boxes are camera-relative and lit by the frame's
    /// key light, so each rock is its crust colour, not a pre-shaded mid-tone.
    pub(super) fn draw(
        &mut self,
        f: &mut Frame3D,
        generator: &dyn TerrainGenerator,
        eye: DVec3,
        view_blocks: f64,
    ) {
        let Some(cosmos) = generator.cosmos() else {
            self.shown.clear();
            self.posed = None;
            return;
        };
        self.update(cosmos, &|| generator.cosmos_arc(), eye, view_blocks);
        for rock in &self.shown {
            f.draw_box(rock.center, rock.half, rock.rot, rock.color);
        }
    }

    fn present(&mut self, eye: DVec3, view_blocks: f64) {
        self.shown.clear();
        for b in 0..self.bands.len() {
            let range = self.bands[b].range;
            for i in 0..self.bands[b].rocks.len() {
                let rock = self.bands[b].rocks[i];
                let delta = rock.world - eye;
                let fade = fade(delta.length(), range);
                // The debug-box pipeline does not blend, so the fade also scales
                // the albedo toward black. Open space is black; the rock dissolves.
                if fade == 0.0 || in_view(delta, view_blocks) {
                    continue;
                }
                self.shown.push(RockBox {
                    center: delta.as_vec3(),
                    half: rock.half,
                    rot: rock.rot,
                    color: faded(rock.albedo, fade),
                    #[cfg(test)]
                    centre: rock.centre,
                    #[cfg(test)]
                    seed: rock.seed,
                    #[cfg(test)]
                    r: rock.r,
                });
            }
        }
        if self.shown.capacity() < self.shown.len() + 32 {
            self.shown.reserve(32);
        }
    }
}

fn in_space(cosmos: &Cosmos, eye: DVec3) -> bool {
    match cosmos.body_at(eye) {
        Some(body) => body.altitude(eye) > SPACE_ALTITUDE,
        None => true,
    }
}

/// Chebyshev: the chunk view is a cube about the eye.
fn in_view(delta: DVec3, reach: f64) -> bool {
    delta.abs().max_element() < reach
}

fn fade(dist: f64, range: f64) -> f32 {
    let start = fade_start(range);
    if dist <= start {
        1.0
    } else if dist >= range {
        0.0
    } else {
        ((range - dist) / (range - start)) as f32
    }
}

fn fade_start(range: f64) -> f64 {
    range * (1.0 - FADE_TAIL)
}

/// Crust colours, matched to the voxel painter: regolith, near-black carbon,
/// rust, frost, crystal, and a warm grey for a derelict's stone and timber.
fn kind_color(kind: RockKind) -> Color {
    match kind {
        RockKind::Rocky => Color::rgb(148, 144, 140),
        RockKind::Carbon => Color::rgb(42, 36, 40),
        RockKind::Metallic => Color::rgb(138, 72, 42),
        RockKind::Icy => Color::rgb(204, 226, 246),
        RockKind::Geode => Color::rgb(176, 124, 255),
        RockKind::Derelict => Color::rgb(164, 148, 128),
    }
}

fn faded(albedo: Color, fade: f32) -> Color {
    let c = |v: u8| (v as f32 * fade).round().clamp(0.0, 255.0) as u8;
    Color::new(
        c(albedo.r),
        c(albedo.g),
        c(albedo.b),
        (255.0 * fade).round().clamp(0.0, 255.0) as u8,
    )
}

fn cell_box(eye: DVec3, radius: f64) -> ([i64; 3], [i64; 3]) {
    (
        std::array::from_fn(|a| (eye[a] - radius).floor() as i64),
        std::array::from_fn(|a| (eye[a] + radius).ceil() as i64),
    )
}

/// Unit vector from two hashes. Rejection sampling, no trigonometry.
fn hashed_dir(seed: u32, n: i32) -> Vec3 {
    for t in 0..64 {
        let h = |a: i32| unit(hash3(seed, n, t, a)) * 2.0 - 1.0;
        let v = Vec3::new(h(0), h(1), h(2));
        let l2 = v.length_squared();
        if l2 > 0.01 && l2 <= 1.0 {
            return v / l2.sqrt();
        }
    }
    Vec3::Y
}

/// Fixed orientation of a rock: Gram–Schmidt on two hashed directions.
fn rotation(seed: u32) -> Mat3 {
    let e1 = hashed_dir(seed, 1);
    let b = hashed_dir(seed, 2);
    let mut v = b - e1 * e1.dot(b);
    if v.length_squared() < 1.0e-8 {
        let axis = if e1.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
        v = axis - e1 * e1.dot(axis);
    }
    let e2 = v.normalize();
    let e3 = e1.cross(e2);
    Mat3::from_cols(e1, e2, e3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc_count;
    use crate::world::terrain::cosmos::{Cosmos, Rock};
    use std::time::Instant;

    fn pos(rock: &Rock) -> DVec3 {
        DVec3::new(rock.centre[0] as f64, rock.centre[1] as f64, rock.centre[2] as f64)
    }

    fn keys(list: &[RockBox]) -> Vec<([i32; 3], u32)> {
        let mut v: Vec<_> = list.iter().map(|s| (s.centre, s.seed)).collect();
        v.sort_unstable();
        v
    }

    fn expected(cosmos: &Cosmos, eye: DVec3, reach: f64) -> Vec<([i32; 3], u32)> {
        let mut v = Vec::new();
        for (class, range) in [(1usize, NEAR), (2, FAR), (3, FAR)] {
            let (lo, hi) = cell_box(eye, range);
            cosmos.for_class_rocks(class, lo, hi, |r| {
                let delta = pos(r) - eye;
                if delta.length() < range && !in_view(delta, reach) {
                    v.push((r.centre, r.seed));
                }
            });
        }
        v.sort_unstable();
        v
    }

    /// An eye half a view-radius from a class-2 rock, with other big rocks still in range.
    fn beside(cosmos: &Cosmos) -> (DVec3, Rock) {
        let reach = chunk_view_blocks(20);
        for c in cosmos.clusters().iter().take(80) {
            let (lo, hi) = cell_box(c.centre, FAR);
            let mut anchor = None;
            cosmos.for_class_rocks(2, lo, hi, |r| {
                if anchor.is_none() {
                    anchor = Some(*r);
                }
            });
            let Some(anchor) = anchor else { continue };
            let eye = pos(&anchor) + DVec3::new(reach * 0.5, 0.0, 0.0);
            if !in_space(cosmos, eye) {
                continue;
            }
            let mut big = 0;
            for class in [2usize, 3] {
                let (lo, hi) = cell_box(eye, FAR);
                cosmos.for_class_rocks(class, lo, hi, |r| {
                    let delta = pos(r) - eye;
                    if delta.length() < FAR && !in_view(delta, reach) {
                        big += 1;
                    }
                });
            }
            if big > 0 {
                return (eye, anchor);
            }
        }
        panic!("seed 42 has no cluster that shows a big rock past the chunk view");
    }

    #[test]
    fn chunk_view_blocks_covers_the_render_cap() {
        let cap = *crate::world::VIEW_RADIUS_RANGE.end();
        let edge = crate::world::chunk::CHUNK_SIZE as f64;
        assert_eq!(chunk_view_blocks(cap), (cap as f64 + 1.0) * edge);
        assert_eq!(chunk_view_blocks(0), edge);
        assert_eq!(chunk_view_blocks(6), 7.0 * edge);
    }

    #[test]
    fn kinds_have_distinct_crusts() {
        use RockKind::*;
        let colors = [Rocky, Carbon, Metallic, Icy, Geode, Derelict].map(kind_color);
        for i in 0..colors.len() {
            for j in 0..i {
                assert_ne!(colors[i], colors[j]);
            }
        }
        let carbon = kind_color(Carbon);
        assert!(carbon.r < 60 && carbon.g < 60 && carbon.b < 60, "{carbon:?}");
        let icy = kind_color(Icy);
        assert!(icy.b > icy.g && icy.g > icy.r);
        let metal = kind_color(Metallic);
        assert!(metal.r > metal.g && metal.g > metal.b);
        let geode = kind_color(Geode);
        assert!(geode.b > geode.r && geode.r > geode.g);
        let rocky = kind_color(Rocky);
        let derelict = kind_color(Derelict);
        assert!((rocky.r as i16 - rocky.g as i16).abs() < 12);
        assert!(derelict.r > derelict.g && derelict.g > derelict.b);
    }

    #[test]
    fn fade_covers_the_last_fifteen_percent() {
        let start = fade_start(FAR);
        assert_eq!(fade(0.0, FAR), 1.0);
        assert_eq!(fade(start, FAR), 1.0);
        assert_eq!(fade(FAR, FAR), 0.0);
        assert_eq!(fade(FAR + 10.0, FAR), 0.0);
        let mid = (start + FAR) * 0.5;
        assert_eq!(fade(mid, FAR), 0.5);
    }

    #[test]
    fn a_rock_rotation_is_a_fixed_right_handed_frame() {
        let mut prev = Mat3::IDENTITY;
        for seed in [0u32, 1, 7, 42, 99, 1_000_003] {
            let a = rotation(seed);
            assert_eq!(a, rotation(seed));
            for axis in [a.x_axis, a.y_axis, a.z_axis] {
                assert!((axis.length() - 1.0).abs() < 1e-4, "{seed}");
            }
            assert!(a.x_axis.dot(a.y_axis).abs() < 1e-4);
            assert!(a.y_axis.dot(a.z_axis).abs() < 1e-4);
            assert!((a.determinant() - 1.0).abs() < 1e-4);
            if seed != 0 {
                assert_ne!(a, prev);
            }
            prev = a;
        }
    }

    #[test]
    fn beside_a_cluster_lists_its_big_rocks_and_skips_the_view() {
        let cosmos = Cosmos::new(42, 1.0);
        let reach = chunk_view_blocks(20);
        let (eye, anchor) = beside(&cosmos);
        assert!(in_view(pos(&anchor) - eye, reach), "the anchor rock is inside the chunk view");
        let mut rocks = DistantRocks::default();
        let list = rocks.update(&cosmos, &|| None, eye, reach);
        assert_eq!(keys(list), expected(&cosmos, eye, reach));
        assert!(list.iter().any(|s| s.r >= 170.0), "a class-2 or class-3 rock is listed");
        assert!(list.iter().all(|s| s.r >= 22.0), "class 0 pebbles stay voxels-only");
        assert!(!list.iter().any(|s| s.centre == anchor.centre && s.seed == anchor.seed));
        for s in list {
            let delta = DVec3::new(s.centre[0] as f64, s.centre[1] as f64, s.centre[2] as f64) - eye;
            assert!(!in_view(delta, reach), "listed rock inside the view: {:?}", s.centre);
        }
        // A pebble outside the view is still not a box.
        let (lo, hi) = cell_box(eye, reach + 20_000.0);
        let mut pebble = None;
        cosmos.for_class_rocks(0, lo, hi, |r| {
            let delta = pos(r) - eye;
            if pebble.is_none() && !in_view(delta, reach) {
                pebble = Some(r.centre);
            }
        });
        let pebble = pebble.expect("a cluster has pebbles outside the view");
        assert!(!rocks.shown.iter().any(|s| s.centre == pebble));
    }

    #[test]
    fn open_space_lists_nothing() {
        let cosmos = Cosmos::new(42, 1.0);
        let cell = (1i64 << 24) as f64;
        let mut eye = None;
        for n in 0..48 {
            let at = DVec3::new((8 + n) as f64 + 0.5, 3.5, -6.5) * cell;
            let clear = cosmos.clusters().iter().all(|c| (c.centre - at).length() > c.radius + FAR * 2.0);
            if clear && in_space(&cosmos, at) {
                eye = Some(at);
                break;
            }
        }
        let eye = eye.expect("an empty cell in space");
        let mut rocks = DistantRocks::default();
        assert!(rocks.update(&cosmos, &|| None, eye, chunk_view_blocks(20)).is_empty());
        assert_eq!(rocks.scans, 3, "empty space is scanned, not skipped as atmosphere");
    }

    #[test]
    fn atmosphere_lists_nothing_and_does_not_scan() {
        let cosmos = Cosmos::new(42, 1.0);
        let spawn = DVec3::new(0.5, 80.0, 0.5);
        assert!(cosmos.home().altitude(spawn) < SPACE_ALTITUDE);
        let mut rocks = DistantRocks::default();
        assert!(rocks.update(&cosmos, &|| None, spawn, chunk_view_blocks(6)).is_empty());
        assert_eq!(rocks.scans, 0);
        let high = DVec3::new(0.5, 30_000.0, 0.5);
        assert!(cosmos.body_at(high).is_some());
        assert!(cosmos.home().altitude(high) > SPACE_ALTITUDE);
        let _ = rocks.update(&cosmos, &|| None, high, chunk_view_blocks(6));
        assert_eq!(rocks.scans, 3, "above the atmosphere the field is drawn");
    }

    #[test]
    fn a_short_move_does_not_rebuild() {
        let cosmos = Cosmos::new(42, 1.0);
        let (eye, _) = beside(&cosmos);
        let reach = chunk_view_blocks(20);
        let mut rocks = DistantRocks::default();
        rocks.update(&cosmos, &|| None, eye, reach);
        let scans = rocks.scans;
        let built: Vec<_> = rocks.bands.iter().map(|b| b.built_at).collect();
        let nudged = eye + DVec3::new(10.0, -4.0, 6.0);
        rocks.update(&cosmos, &|| None, nudged, reach);
        assert_eq!(rocks.scans, scans, "a few blocks must not rescan");
        assert_eq!(rocks.bands.iter().map(|b| b.built_at).collect::<Vec<_>>(), built);
        assert_eq!(rocks.posed, Some((nudged, reach)));
        // Class 1's margin is 1 000; class 2 and 3 keep theirs until 7 500.
        let step = eye + DVec3::new(2_000.0, 0.0, 0.0);
        rocks.update(&cosmos, &|| None, step, reach);
        assert_eq!(rocks.scans, scans + 1);
        assert_eq!(rocks.bands[0].built_at, Some(step));
        assert_eq!(rocks.bands[1].built_at, built[1]);
        assert_eq!(rocks.bands[2].built_at, built[2]);
    }

    #[test]
    fn the_last_band_of_the_range_fades() {
        let cosmos = Cosmos::new(42, 1.0);
        let mut rock = None;
        for c in cosmos.clusters().iter().take(80) {
            let (lo, hi) = cell_box(c.centre, FAR);
            cosmos.for_class_rocks(2, lo, hi, |r| {
                if rock.is_none() {
                    rock = Some(*r);
                }
            });
            if rock.is_some() {
                break;
            }
        }
        let rock = rock.expect("a class-2 rock");
        let reach = chunk_view_blocks(20);
        let dist = (fade_start(FAR) + FAR) * 0.5;
        let eye = pos(&rock) + DVec3::new(dist, 0.0, 0.0);
        assert!(in_space(&cosmos, eye));
        assert!(!in_view(pos(&rock) - eye, reach));
        let mut rocks = DistantRocks::default();
        let shown = rocks
            .update(&cosmos, &|| None, eye, reach)
            .iter()
            .find(|s| s.centre == rock.centre && s.seed == rock.seed)
            .expect("the rock is inside its range");
        let f = fade(dist, FAR);
        assert!((f - 0.5).abs() < 1e-5);
        assert_eq!(shown.color, faded(kind_color(rock.kind), f));
        assert_ne!(shown.color, kind_color(rock.kind));
        assert_eq!(shown.center, (pos(&rock) - eye).as_vec3());

        let near = pos(&rock) + DVec3::new(10_000.0, 0.0, 0.0);
        let shown = rocks
            .update(&cosmos, &|| None, near, reach)
            .iter()
            .find(|s| s.centre == rock.centre && s.seed == rock.seed)
            .expect("10 000 blocks is inside the class-2 range");
        assert_eq!(shown.color, kind_color(rock.kind));
        assert_eq!(
            shown.half,
            Vec3::new(rock.r * rock.axes[0], rock.r * rock.axes[1], rock.r * rock.axes[2])
        );
        assert_eq!(shown.rot, rotation(rock.seed));

        let past = pos(&rock) + DVec3::new(FAR + 100.0, 0.0, 0.0);
        assert!(rocks
            .update(&cosmos, &|| None, past, reach)
            .iter()
            .all(|s| s.centre != rock.centre || s.seed != rock.seed));
    }

    #[test]
    fn warm_lists_allocate_nothing() {
        let cosmos = Cosmos::new(42, 1.0);
        let (eye, _) = beside(&cosmos);
        let reach = chunk_view_blocks(20);
        let mut rocks = DistantRocks::default();
        let n = rocks.update(&cosmos, &|| None, eye, reach).len();
        assert!(n > 0);
        alloc_count::reset();
        assert_eq!(rocks.update(&cosmos, &|| None, eye, reach).len(), n);
        let nudged = rocks.update(&cosmos, &|| None, eye + DVec3::new(3.0, -1.0, 2.0), reach).len();
        assert_eq!(alloc_count::alloc_count(), 0, "warm distant-rock update allocated");
        assert!(nudged > 0);
    }

    /// `cargo test --release --lib distant_rock_rebuild_cost -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn distant_rock_rebuild_cost() {
        let cosmos = Cosmos::new(42, 1.0);
        let home = cosmos.home().centre_f();
        let near = 2.0e8;
        let densest = cosmos
            .clusters()
            .iter()
            .filter(|c| (c.centre - home).length() < near)
            .max_by(|a, b| a.count.total_cmp(&b.count))
            .expect("a cluster within 2e8 of home");
        let reach = chunk_view_blocks(20);
        let eye = densest.centre;
        let mut rocks = DistantRocks::default();
        let t0 = Instant::now();
        let n = rocks.update(&cosmos, &|| None, eye, reach).len();
        let cold = t0.elapsed();
        let cached: usize = rocks.bands.iter().map(|b| b.rocks.len()).sum();
        let per: Vec<_> = rocks.bands.iter().map(|b| b.rocks.len()).collect();
        let eye2 = eye + DVec3::new(8_000.0, 0.0, 0.0);
        let t1 = Instant::now();
        let n2 = rocks.update(&cosmos, &|| None, eye2, reach).len();
        let warm = t1.elapsed();
        let t2 = Instant::now();
        let _ = rocks.update(&cosmos, &|| None, eye2, reach);
        let still = t2.elapsed();
        println!(
            "densest within 2e8 of home: count {:.0} radius {:.0} dist {:.3e} centre {:?}",
            densest.count,
            densest.radius,
            (densest.centre - home).length(),
            densest.centre
        );
        println!("cold rebuild {cold:?} shown {n} cached {cached} per class {per:?}");
        println!("warm rebuild (eye + 8000) {warm:?} shown {n2}");
        println!("same eye again {still:?}");
    }

    /// With a shared catalog the rescan runs in the background and lands as the same list.
    #[test]
    fn a_background_rescan_lands_the_same_list() {
        let cosmos = Arc::new(Cosmos::new(42, 1.0));
        let (eye, _) = beside(&cosmos);
        let reach = chunk_view_blocks(20);
        let mut sync = DistantRocks::default();
        let want = keys(sync.update(&cosmos, &|| None, eye, reach));
        assert!(!want.is_empty());
        let mut rocks = DistantRocks::default();
        let shared = cosmos.clone();
        let first = rocks.update(&cosmos, &|| Some(shared.clone()), eye, reach).len();
        assert_eq!(first, 0, "nothing drawn before the first scan lands");
        let start = Instant::now();
        while rocks.bands.iter().any(|b| b.pending.is_some()) {
            assert!(start.elapsed().as_secs() < 30, "the background scan never landed");
            std::thread::yield_now();
            let _ = rocks.update(&cosmos, &|| Some(shared.clone()), eye, reach);
        }
        assert_eq!(keys(rocks.update(&cosmos, &|| Some(shared.clone()), eye, reach)), want);
    }
}
