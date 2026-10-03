//! Far worlds and moons as sky impostors, taken from the cosmos catalog.
//! The list is rebuilt every frame into a reused buffer.

use voxel_engine::{DVec3, FarBody, FarShape, LinearRgb, Quat, Vec3, MAX_FAR_BODIES};

use crate::sky::palette::Rgb;
use crate::world::generation::TerrainGenerator;
use crate::world::terrain::cosmos::{Body, Kind, Shape};

/// Above this altitude the voxel terrain is not drawn, so the impostor takes over.
const STREAM_ALTITUDE: f64 = 20_000.0;

/// Reused far-body list. Capacity stays at [`MAX_FAR_BODIES`] after the first frame.
#[derive(Debug)]
pub struct FarBodies {
    list: Vec<FarBody>,
}

impl Default for FarBodies {
    fn default() -> Self {
        Self {
            list: Vec::with_capacity(MAX_FAR_BODIES),
        }
    }
}

impl FarBodies {
    /// Bodies to draw this frame, relative to `eye`. Empty when the generator has no cosmos.
    pub fn update(&mut self, generator: &dyn TerrainGenerator, eye: DVec3) -> &[FarBody] {
        self.list.clear();
        let Some(cosmos) = generator.cosmos() else {
            return &self.list;
        };
        let mut twin = 0u32;
        for body in cosmos.bodies() {
            let ordinal = if body.kind == Kind::Twin {
                let i = twin;
                twin += 1;
                i
            } else {
                0
            };
            if self.list.len() == MAX_FAR_BODIES {
                break;
            }
            if let Some(far) = impostor(body, eye, ordinal) {
                self.list.push(far);
            }
        }
        &self.list
    }
}

fn black() -> LinearRgb {
    LinearRgb([0.0, 0.0, 0.0])
}

fn srgb(r: u8, g: u8, b: u8) -> LinearRgb {
    Rgb::from_srgb8(r, g, b).to_linear()
}

fn hdr(r: f32, g: f32, b: f32) -> LinearRgb {
    Rgb::linear(r, g, b).to_linear()
}

/// Sphere tones: every face is land, and `[1]` is the second tone the shader mixes in.
fn tones(land: LinearRgb, second: LinearRgb) -> [LinearRgb; 6] {
    let mut albedo = [land; 6];
    albedo[1] = second;
    albedo
}

/// Home cube faces: +Y basin, −Y ash, +X dune, −X shattered, +Z glass, −Z fungal.
fn home_faces() -> [LinearRgb; 6] {
    [
        srgb(214, 176, 112),
        srgb(128, 128, 132),
        srgb(72, 140, 58),
        srgb(58, 50, 46),
        srgb(214, 228, 236),
        srgb(132, 72, 158),
    ]
}

/// The first catalog twin (side −1) is lush; the next is crystalline.
fn twin_faces(ordinal: u32) -> [LinearRgb; 6] {
    if ordinal == 0 {
        [
            srgb(64, 130, 52),
            srgb(48, 112, 58),
            srgb(86, 158, 64),
            srgb(36, 84, 40),
            srgb(70, 140, 90),
            srgb(54, 120, 48),
        ]
    } else {
        [
            srgb(198, 220, 238),
            srgb(176, 206, 230),
            srgb(220, 234, 246),
            srgb(150, 184, 214),
            srgb(190, 216, 240),
            srgb(180, 208, 232),
        ]
    }
}

fn paint(body: &Body, twin_ordinal: u32) -> (FarShape, [LinearRgb; 6], LinearRgb) {
    match body.kind {
        Kind::Home => (FarShape::Cube, home_faces(), black()),
        Kind::Twin => (FarShape::Cube, twin_faces(twin_ordinal), black()),
        // Meadow and forest green: Verdance has no seas.
        Kind::Verdant => (
            FarShape::Sphere,
            tones(srgb(70, 140, 56), srgb(32, 92, 44)),
            srgb(186, 216, 232),
        ),
        Kind::Hollow => (
            FarShape::Sphere,
            tones(srgb(220, 232, 238), srgb(160, 190, 210)),
            black(),
        ),
        Kind::Ember => (
            FarShape::Sphere,
            tones(srgb(255, 120, 36), srgb(170, 48, 16)),
            hdr(1.8, 0.42, 0.06),
        ),
        // The same three characters the moon painter gives them (`round::Style::Moon { tone }`).
        Kind::Moon => {
            let albedo = match body.seed % 3 {
                0 => tones(srgb(150, 148, 152), srgb(60, 66, 72)),
                1 => tones(srgb(200, 222, 240), srgb(150, 176, 220)),
                _ => tones(srgb(186, 104, 90), srgb(170, 124, 110)),
            };
            (FarShape::Sphere, albedo, black())
        }
    }
}

fn radius_of(body: &Body) -> f64 {
    match body.shape {
        Shape::Cube { half } => half as f64,
        Shape::Ball { r } => r as f64,
        // A shell reads as its outer sphere. The cavity is not an impostor.
        Shape::Shell { outer, .. } => outer as f64,
    }
}

/// How far a round body's impostor sinks below its datum (under its valleys; moons' big craters
/// go deeper): standing on the body, the sphere fills the horizon beyond the streamed chunks
/// without covering them (the sky pass draws only where no terrain was drawn). Round bodies have no
/// far LOD of their own yet; cubes do, so theirs hide while streamed.
fn sink(body: &Body) -> Option<f64> {
    match (body.kind, body.shape) {
        (_, Shape::Cube { .. }) => None,
        (Kind::Moon, _) => Some(650.0),
        _ => Some(150.0),
    }
}

/// One catalog body as seen from `eye`, or nothing while voxels cover it.
fn impostor(body: &Body, eye: DVec3, twin_ordinal: u32) -> Option<FarBody> {
    let delta = body.centre_f() - eye;
    let dist = delta.length();
    let radius = radius_of(body) - sink(body).unwrap_or(0.0);
    // A cube's own mesh is the body while it streams; a round body's sphere is drawn unless the eye
    // is inside it.
    let streamed = sink(body).is_none() && body.altitude(eye) < STREAM_ALTITUDE;
    if streamed || !(dist > radius) || !dist.is_finite() {
        return None;
    }
    let n = delta / dist;
    let dir = Vec3::new(n.x as f32, n.y as f32, n.z as f32);
    let distance = dist as f32;
    let radius_f = radius as f32;
    if !dir.is_finite()
        || dir.length_squared() == 0.0
        || !distance.is_finite()
        || !radius_f.is_finite()
        || !(distance > radius_f)
    {
        return None;
    }
    let (shape, albedo, atmosphere) = paint(body, twin_ordinal);
    Some(FarBody {
        dir,
        distance,
        radius: radius_f,
        shape,
        rotation: Quat::IDENTITY,
        albedo,
        atmosphere,
        seed: body.seed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc_count;
    use crate::block::registry::BlockRegistry;
    use crate::world::generation::FlatTerrain;
    use crate::world::terrain::cosmos::{Kind, HOME_HALF};
    use crate::world::terrain::Terrain;

    fn radius_f(body: &Body) -> f32 {
        (radius_of(body) - sink(body).unwrap_or(0.0)) as f32
    }

    fn find<'a>(list: &'a [FarBody], body: &Body) -> Option<&'a FarBody> {
        list.iter()
            .find(|far| far.seed == body.seed && (far.radius - radius_f(body)).abs() < 4.0)
    }

    /// Independent of `impostor`: cubes by altitude and outside-the-solid, round bodies outside
    /// their sunk sphere, in f64.
    fn expect_visible(body: &Body, eye: DVec3) -> bool {
        let dist = (body.centre_f() - eye).length();
        match body.shape {
            Shape::Cube { .. } => body.altitude(eye) >= STREAM_ALTITUDE && dist > radius_of(body),
            _ => dist > radius_of(body) - sink(body).unwrap(),
        }
    }

    fn finite_unit(far: &FarBody) {
        assert!(far.dir.is_finite(), "dir {:?}", far.dir);
        assert!((far.dir.length() - 1.0).abs() < 1e-4, "{}", far.dir.length());
        assert!(far.distance.is_finite() && far.distance > far.radius);
        assert!(far.radius.is_finite() && far.radius > 0.0);
        assert_eq!(far.rotation, Quat::IDENTITY);
    }

    #[test]
    fn flat_world_has_no_far_bodies() {
        let mut registry = BlockRegistry::with_builtins();
        let flat = FlatTerrain::new(&mut registry, 1);
        assert!(flat.cosmos().is_none());
        let mut far = FarBodies::default();
        assert!(far.update(&flat, DVec3::ZERO).is_empty());
    }

    #[test]
    fn spawn_lists_the_far_worlds_and_a_distant_eye_lists_home() {
        let mut registry = BlockRegistry::with_builtins();
        let terrain = Terrain::new(&mut registry, 1);
        let cosmos = terrain.cosmos().expect("the diffusion generator has a cosmos");
        for kind in [Kind::Twin, Kind::Verdant, Kind::Hollow, Kind::Ember, Kind::Moon] {
            assert!(
                cosmos.bodies().iter().any(|b| b.kind == kind),
                "catalog missing {kind:?}"
            );
        }

        let spawn = DVec3::new(0.5, 8.0, 0.5);
        let mut far = FarBodies::default();
        let listed = far.update(&terrain, spawn);
        assert!(find(listed, cosmos.home()).is_none());
        let mut expect = 0usize;
        for body in cosmos.bodies() {
            let visible = expect_visible(body, spawn);
            let got = find(listed, body);
            if body.kind == Kind::Home {
                assert!(!visible);
                assert!(got.is_none());
                continue;
            }
            assert!(visible, "{:?} should be a sky body from spawn", body.kind);
            let got = got.expect("missing far body");
            finite_unit(got);
            let shape = match body.kind {
                Kind::Home | Kind::Twin => FarShape::Cube,
                _ => FarShape::Sphere,
            };
            assert_eq!(got.shape, shape);
            expect += 1;
        }
        assert_eq!(listed.len(), expect);

        let twins: Vec<_> = cosmos.bodies().iter().filter(|b| b.kind == Kind::Twin).collect();
        assert!(twins.len() >= 2);
        let lush = find(listed, twins[0]).unwrap();
        let crystal = find(listed, twins[1]).unwrap();
        assert!(lush.albedo[2].0[1] > lush.albedo[2].0[0] && lush.albedo[2].0[1] > lush.albedo[2].0[2]);
        assert!(crystal.albedo[2].0[2] > crystal.albedo[2].0[0]);

        let verdant = cosmos.bodies().iter().find(|b| b.kind == Kind::Verdant).unwrap();
        let verdant = find(listed, verdant).unwrap();
        assert!(verdant.albedo[0].0[1] > verdant.albedo[0].0[0]);
        assert!(verdant.albedo[1].0[1] > verdant.albedo[1].0[2], "forest green, no seas");
        assert!(verdant.atmosphere.0[2] > verdant.atmosphere.0[0]);
        assert!(verdant.atmosphere.0[0] > 0.0);

        let hollow = cosmos.bodies().iter().find(|b| b.kind == Kind::Hollow).unwrap();
        let hollow = find(listed, hollow).unwrap();
        assert_eq!(hollow.atmosphere.0, [0.0, 0.0, 0.0]);
        assert!(hollow.albedo[0].0[2] > 0.5);

        let ember = cosmos.bodies().iter().find(|b| b.kind == Kind::Ember).unwrap();
        let ember = find(listed, ember).unwrap();
        assert!(ember.atmosphere.0[0] > 1.0);
        assert!(ember.atmosphere.0[0] > ember.atmosphere.0[1]);
        assert!(ember.atmosphere.0[1] > ember.atmosphere.0[2]);
        assert!(ember.albedo[0].0[0] > ember.albedo[0].0[1]);

        for moon in cosmos.bodies().iter().filter(|b| b.kind == Kind::Moon) {
            let moon = find(listed, moon).unwrap();
            assert_eq!(moon.shape, FarShape::Sphere);
            assert_eq!(moon.atmosphere.0, [0.0, 0.0, 0.0]);
            let a = moon.albedo[0].0;
            assert!(a.iter().all(|&c| c > 0.0 && c <= 1.0), "a lit, plain surface tone: {a:?}");
        }

        // Warm the buffer, then two more updates must not allocate.
        alloc_count::reset();
        let n = far.update(&terrain, spawn).len();
        assert!(n > 0);
        let away = far.update(&terrain, DVec3::new(1.0e8, 0.0, 0.0));
        assert_eq!(alloc_count::alloc_count(), 0, "far-body update allocated");
        let home = find(away, cosmos.home()).expect("home is a sky body from 1e8");
        assert_eq!(home.shape, FarShape::Cube);
        assert!((home.radius - HOME_HALF as f32).abs() < 4.0);
        finite_unit(home);
        assert!(home.dir.x < -0.9, "home should sit toward −X, dir {:?}", home.dir);
        // +Y green basin, −Y ash, +X dune, −X grey, +Z glass, −Z fungal.
        let [px, nx, py, ny, pz, nz] = home.albedo;
        assert!(py.0[1] > py.0[0] && py.0[1] > py.0[2]);
        assert!(ny.0[0] < 0.08 && ny.0[0] > ny.0[2]);
        assert!(px.0[0] > px.0[2]);
        assert!((nx.0[0] - nx.0[1]).abs() < 0.02 && (nx.0[1] - nx.0[2]).abs() < 0.05);
        assert!(pz.0[2] > pz.0[0] && pz.0[0] > 0.5);
        assert!(nz.0[0] > nz.0[1] && nz.0[2] > nz.0[1]);
        assert_eq!(home.atmosphere.0, [0.0, 0.0, 0.0]);
        for body in away {
            finite_unit(body);
        }
    }
}
