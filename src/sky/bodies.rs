//! Far worlds and moons as sky impostors, taken from the cosmos catalog.
//! The list is rebuilt every frame into a reused buffer.

use voxel_engine::{
    DVec3, FarBody, FarShape, LinearRgb, Quat, SunOverride, Vec3, MAX_FAR_BODIES,
};

use crate::sky::palette::Rgb;
use crate::world::generation::TerrainGenerator;
use crate::world::terrain::cosmos::{Body, Cosmos, Kind, Shape};

/// Above this altitude the voxel terrain is not drawn, so the impostor takes over.
const STREAM_ALTITUDE: f64 = 20_000.0;
/// The inner wall sits this far into the shell, behind the inward-hanging terrain.
const WALL_BEHIND: f32 = 150.0;
/// Warm orange of the core light, before the distance scale (`inner / distance`).
const CORE_ORANGE: [f32; 3] = [1.35, 0.86, 0.46];
/// The Ember's rim inside the cavity, so the core reads as the light.
const CORE_GLOW: f32 = 5.0;
/// Brightest the Ember's light gets, relative to its light on the inner wall.
const EMBER_CAP: f64 = 2.0;
/// The inner wall's glow on the Ember's own surface, relative to the Ember's light on the wall.
const WALL_GLOW: f64 = 0.35;

/// Reused far-body list. Capacity stays at [`MAX_FAR_BODIES`] after the first frame.
#[derive(Debug)]
pub struct FarBodies {
    list: Vec<FarBody>,
    /// Core light while the eye is in a Hollow's cavity. `None` outside it.
    sun: Option<SunOverride>,
}

impl Default for FarBodies {
    fn default() -> Self {
        Self {
            list: Vec::with_capacity(MAX_FAR_BODIES),
            sun: None,
        }
    }
}

impl FarBodies {
    /// Point light for this frame. `None` outside a Hollow's cavity.
    pub fn sun_override(&self) -> Option<SunOverride> {
        self.sun
    }

    /// Bodies to draw this frame, relative to `eye`. Empty when the generator has no cosmos.
    pub fn update(&mut self, generator: &dyn TerrainGenerator, eye: DVec3) -> &[FarBody] {
        self.list.clear();
        self.sun = None;
        let Some(cosmos) = generator.cosmos() else {
            return &self.list;
        };
        if let Some((hollow, ember)) = cosmos.hollow_cavity(eye) {
            self.fill_cavity(cosmos, hollow, ember, eye);
            return &self.list;
        }
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
            if let Some(far) = impostor(cosmos, body, eye, ordinal) {
                self.list.push(far);
            }
        }
        &self.list
    }

    /// The shell's far wall and the Ember. Nothing outside the shell is visible.
    fn fill_cavity(&mut self, cosmos: &Cosmos, hollow: &Body, ember: &Body, eye: DVec3) {
        let Shape::Shell { inner, .. } = hollow.shape else {
            return;
        };
        let delta = hollow.centre_f() - eye;
        let dist = delta.length();
        if !(dist > 0.0) || !dist.is_finite() {
            return;
        }
        let n = delta / dist;
        let dir = Vec3::new(n.x as f32, n.y as f32, n.z as f32);
        let distance = dist as f32;
        let radius = inner as f32 + WALL_BEHIND;
        if dir.is_finite()
            && dir.length_squared() > 0.0
            && distance.is_finite()
            && distance > 0.0
            && radius.is_finite()
            && distance < radius
        {
            self.list.push(FarBody {
                dir,
                distance,
                radius,
                shape: FarShape::InnerSphere,
                rotation: Quat::IDENTITY,
                albedo: tones(srgb(132, 88, 172), srgb(176, 124, 255)),
                atmosphere: black(),
                seed: hollow.seed,
            });
        }
        if let Some(mut far) = impostor(cosmos, ember, eye, 0) {
            far.atmosphere = LinearRgb([
                far.atmosphere.0[0] * CORE_GLOW,
                far.atmosphere.0[1] * CORE_GLOW,
                far.atmosphere.0[2] * CORE_GLOW,
            ]);
            self.list.push(far);
        }
        if !dir.is_finite() || dir.length_squared() == 0.0 {
            return;
        }
        let (dir, scale) = ember_light(dir, dist, inner as f64, ember_radius(ember));
        self.sun = Some(SunOverride {
            dir,
            color: LinearRgb([
                CORE_ORANGE[0] * scale,
                CORE_ORANGE[1] * scale,
                CORE_ORANGE[2] * scale,
            ]),
            show_disc: false,
        });
    }
}

/// The Ember's radius (a ball at the Hollow's centre); 0 for any other shape.
fn ember_radius(ember: &Body) -> f64 {
    match ember.shape {
        Shape::Ball { r } => r as f64,
        _ => 0.0,
    }
}

/// The cavity's key light at `dist` from the centre, `toward` the centre: `(direction to the light,
/// brightness)`. Out in the cavity the Ember is the sun, brighter as it nears (relative to the inner
/// wall, capped at [`EMBER_CAP`]). Close to its surface the ground itself is the source, so what
/// lights a face from outside is the far wall's glow overhead: dimmer ([`WALL_GLOW`]) and from
/// above. The brightness passes through zero where the two meet, one Ember radius up, so the
/// direction flip never shows.
fn ember_light(toward: Vec3, dist: f64, inner: f64, ember_r: f64) -> (Vec3, f32) {
    let smooth = |e0: f64, e1: f64, x: f64| {
        let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    let up = if ember_r > 0.0 { (dist - ember_r) / ember_r } else { f64::INFINITY };
    if up >= 1.0 {
        let scale = (inner / dist).min(EMBER_CAP) * smooth(1.0, 2.0, up);
        (toward, scale as f32)
    } else {
        (-toward, (WALL_GLOW * smooth(1.0, 0.5, up)) as f32)
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
        Kind::Home => (FarShape::Sphere, home_faces(), black()),
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

/// [`sink`] below a relaxed body's lowest datum offset, so the sphere stays under its lowlands.
fn sink_in(cosmos: &Cosmos, body: &Body) -> Option<f64> {
    sink(body).map(|s| s - cosmos.relief_range(body).0.min(0.0))
}

/// One catalog body as seen from `eye`, or nothing while voxels cover it.
fn impostor(cosmos: &Cosmos, body: &Body, eye: DVec3, twin_ordinal: u32) -> Option<FarBody> {
    let delta = body.centre_f() - eye;
    let dist = delta.length();
    let radius = radius_of(body) - sink_in(cosmos, body).unwrap_or(0.0);
    // A cube's own mesh is the body while it streams; a round body's sphere is drawn unless the eye
    // is inside it.
    let streamed = sink(body).is_none() && cosmos.altitude(body, eye) < STREAM_ALTITUDE;
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
    use crate::world::terrain::cosmos::Kind;
    use crate::world::terrain::Terrain;

    fn radius_f(cosmos: &Cosmos, body: &Body) -> f32 {
        (radius_of(body) - sink_in(cosmos, body).unwrap_or(0.0)) as f32
    }

    fn find<'a>(cosmos: &Cosmos, list: &'a [FarBody], body: &Body) -> Option<&'a FarBody> {
        list.iter()
            .find(|far| far.seed == body.seed && (far.radius - radius_f(cosmos, body)).abs() < 4.0)
    }

    /// Independent of `impostor`: cubes by altitude and outside-the-solid, round bodies outside
    /// their sunk sphere, in f64.
    fn expect_visible(cosmos: &Cosmos, body: &Body, eye: DVec3) -> bool {
        let dist = (body.centre_f() - eye).length();
        match body.shape {
            Shape::Cube { .. } => cosmos.altitude(body, eye) >= STREAM_ALTITUDE && dist > radius_of(body),
            _ => dist > radius_of(body) - sink_in(cosmos, body).unwrap(),
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
        assert!(listed.iter().all(|b| b.shape != FarShape::InnerSphere));
        let mut expect = 0usize;
        for body in cosmos.bodies() {
            let visible = expect_visible(cosmos, body, spawn);
            let got = find(cosmos, listed, body);
            assert!(visible, "{:?} should be a sky body from spawn", body.kind);
            let got = got.expect("missing far body");
            finite_unit(got);
            let shape = match body.kind {
                Kind::Twin => FarShape::Cube,
                _ => FarShape::Sphere,
            };
            assert_eq!(got.shape, shape, "{:?}", body.kind);
            expect += 1;
        }
        assert_eq!(listed.len(), expect);

        let twins: Vec<_> = cosmos.bodies().iter().filter(|b| b.kind == Kind::Twin).collect();
        assert!(twins.len() >= 2);
        let lush = find(cosmos, listed, twins[0]).unwrap();
        let crystal = find(cosmos, listed, twins[1]).unwrap();
        assert!(lush.albedo[2].0[1] > lush.albedo[2].0[0] && lush.albedo[2].0[1] > lush.albedo[2].0[2]);
        assert!(crystal.albedo[2].0[2] > crystal.albedo[2].0[0]);

        let verdant = cosmos.bodies().iter().find(|b| b.kind == Kind::Verdant).unwrap();
        let verdant = find(cosmos, listed, verdant).unwrap();
        assert!(verdant.albedo[0].0[1] > verdant.albedo[0].0[0]);
        assert!(verdant.albedo[1].0[1] > verdant.albedo[1].0[2], "forest green, no seas");
        assert!(verdant.atmosphere.0[2] > verdant.atmosphere.0[0]);
        assert!(verdant.atmosphere.0[0] > 0.0);

        let hollow = cosmos.bodies().iter().find(|b| b.kind == Kind::Hollow).unwrap();
        let hollow = find(cosmos, listed, hollow).unwrap();
        assert_eq!(hollow.atmosphere.0, [0.0, 0.0, 0.0]);
        assert!(hollow.albedo[0].0[2] > 0.5);

        let ember = cosmos.bodies().iter().find(|b| b.kind == Kind::Ember).unwrap();
        let ember = find(cosmos, listed, ember).unwrap();
        assert!(ember.atmosphere.0[0] > 1.0);
        assert!(ember.atmosphere.0[0] > ember.atmosphere.0[1]);
        assert!(ember.atmosphere.0[1] > ember.atmosphere.0[2]);
        assert!(ember.albedo[0].0[0] > ember.albedo[0].0[1]);

        for moon in cosmos.bodies().iter().filter(|b| b.kind == Kind::Moon) {
            let moon = find(cosmos, listed, moon).unwrap();
            assert_eq!(moon.shape, FarShape::Sphere);
            assert_eq!(moon.atmosphere.0, [0.0, 0.0, 0.0]);
            let a = moon.albedo[0].0;
            assert!(a.iter().all(|&c| c > 0.0 && c <= 1.0), "a lit, plain surface tone: {a:?}");
        }
        assert!(far.sun_override().is_none());

        // Warm the buffer, then two more updates must not allocate.
        alloc_count::reset();
        let n = far.update(&terrain, spawn).len();
        assert!(n > 0);
        let away = far.update(&terrain, DVec3::new(1.0e8, 0.0, 0.0));
        assert_eq!(alloc_count::alloc_count(), 0, "far-body update allocated");
        let home = find(cosmos, away, cosmos.home()).expect("home is a sky body from 1e8");
        assert_eq!(home.shape, FarShape::Sphere);
        let Shape::Ball { r } = cosmos.home().shape else { panic!("home is a ball") };
        // Sunk under the relaxed lowlands.
        let lowest = cosmos.relief_range(cosmos.home()).0.min(0.0);
        assert!((home.radius - (r as f64 - 150.0 + lowest) as f32).abs() < 4.0);
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
        assert!(far.sun_override().is_none());
    }

    #[test]
    fn inside_the_hollow_the_wall_and_the_ember_are_the_sky() {
        let mut registry = BlockRegistry::with_builtins();
        let terrain = Terrain::new(&mut registry, 1);
        let cosmos = terrain.cosmos().expect("cosmos");
        let hollow = cosmos.bodies().iter().find(|b| b.kind == Kind::Hollow).unwrap();
        let ember = cosmos.bodies().iter().find(|b| b.kind == Kind::Ember).unwrap();
        let Shape::Shell { outer, inner } = hollow.shape else {
            panic!("the Hollow is a shell");
        };
        let Shape::Ball { r } = ember.shape else {
            panic!("the Ember is a ball");
        };
        assert_eq!(hollow.centre, ember.centre);

        let mut far = FarBodies::default();
        let outside = hollow.centre_f() + DVec3::new(0.0, outer as f64 + 10_000.0, 0.0);
        let outside_air = {
            let listed = far.update(&terrain, outside);
            assert!(listed.iter().all(|b| b.shape != FarShape::InnerSphere));
            let shell = listed.iter().find(|b| b.seed == hollow.seed).expect("outer shell");
            assert_eq!(shell.shape, FarShape::Sphere);
            listed
                .iter()
                .find(|b| b.seed == ember.seed)
                .expect("ember")
                .atmosphere
        };
        assert!(far.sun_override().is_none());

        let halfway = hollow.centre_f() + DVec3::new(0.0, inner as f64 * 0.5, 0.0);
        let listed = far.update(&terrain, halfway);
        assert_eq!(listed.len(), 2);
        let wall = &listed[0];
        let core = &listed[1];
        assert_eq!(wall.shape, FarShape::InnerSphere);
        assert_eq!(wall.seed, hollow.seed);
        assert_eq!(wall.radius.to_bits(), (inner as f32 + WALL_BEHIND).to_bits());
        assert_eq!(wall.distance.to_bits(), (inner as f32 / 2.0).to_bits());
        assert_eq!(wall.dir.x.to_bits(), 0.0f32.to_bits());
        assert_eq!(wall.dir.y.to_bits(), (-1.0f32).to_bits());
        assert_eq!(wall.dir.z.to_bits(), 0.0f32.to_bits());
        assert_eq!(wall.atmosphere.0, [0.0, 0.0, 0.0]);
        assert_eq!(wall.rotation, Quat::IDENTITY);
        let want = tones(srgb(132, 88, 172), srgb(176, 124, 255));
        for face in 0..6 {
            assert_eq!(wall.albedo[face].0[0].to_bits(), want[face].0[0].to_bits());
            assert_eq!(wall.albedo[face].0[1].to_bits(), want[face].0[1].to_bits());
            assert_eq!(wall.albedo[face].0[2].to_bits(), want[face].0[2].to_bits());
        }
        assert_eq!(core.shape, FarShape::Sphere);
        assert_eq!(core.seed, ember.seed);
        for channel in 0..3 {
            assert_eq!(
                core.atmosphere.0[channel].to_bits(),
                (outside_air.0[channel] * CORE_GLOW).to_bits()
            );
        }
        assert!(listed.iter().all(|b| b.seed == hollow.seed || b.seed == ember.seed));

        let sun = far.sun_override().expect("the Ember lights the cavity");
        assert!(!sun.show_disc);
        assert_eq!(sun.dir.x.to_bits(), 0.0f32.to_bits());
        assert_eq!(sun.dir.y.to_bits(), (-1.0f32).to_bits());
        assert_eq!(sun.dir.z.to_bits(), 0.0f32.to_bits());
        assert_eq!(sun.color.0[0].to_bits(), (CORE_ORANGE[0] * 2.0).to_bits());
        assert_eq!(sun.color.0[1].to_bits(), (CORE_ORANGE[1] * 2.0).to_bits());
        assert_eq!(sun.color.0[2].to_bits(), (CORE_ORANGE[2] * 2.0).to_bits());
        assert!(sun.color.0[0] > sun.color.0[1] && sun.color.0[1] > sun.color.0[2]);
        let halfway_r = sun.color.0[0];

        // Two Ember radii up the Ember is still the sun below, a little brighter than at halfway
        // would be without the cap; on its surface the light is the far wall's glow, from above.
        let near = hollow.centre_f() + DVec3::new(0.0, r as f64 * 3.0, 0.0);
        let listed = far.update(&terrain, near);
        assert_eq!(listed.len(), 2);
        let nearer = far.sun_override().expect("near the Ember");
        assert!(nearer.color.0[0] >= halfway_r);
        assert!(nearer.color.0[0] <= CORE_ORANGE[0] * EMBER_CAP as f32);
        assert_eq!(nearer.dir.y.to_bits(), (-1.0f32).to_bits());
        let close = hollow.centre_f() + DVec3::new(0.0, r as f64 + 50_000.0, 0.0);
        let listed = far.update(&terrain, close);
        assert_eq!(listed.len(), 2);
        let ground = far.sun_override().expect("on the Ember");
        assert!(!ground.show_disc);
        assert_eq!(ground.dir.y.to_bits(), 1.0f32.to_bits(), "lit from the wall overhead");
        assert!(ground.color.0[0] < halfway_r && ground.color.0[0] > 0.0);
        let flip = hollow.centre_f() + DVec3::new(0.0, r as f64 * 2.0, 0.0);
        let _ = far.update(&terrain, flip);
        assert!(far.sun_override().expect("at the flip").color.0[0].abs() < 1.0e-6, "no pop where the light turns");

        let lip = hollow.centre_f() + DVec3::new(0.0, inner as f64 - 1.0, 0.0);
        let listed = far.update(&terrain, lip);
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].shape, FarShape::InnerSphere);
        assert!(far.sun_override().is_some());

        let rock = hollow.centre_f() + DVec3::new(0.0, inner as f64 + 1.0, 0.0);
        let listed = far.update(&terrain, rock);
        assert!(listed.iter().all(|b| b.shape != FarShape::InnerSphere));
        assert!(far.sun_override().is_none());

        let centre = far.update(&terrain, hollow.centre_f());
        assert!(centre.iter().all(|b| b.shape != FarShape::InnerSphere));
        assert!(far.sun_override().is_none());

        let listed = far.update(&terrain, outside);
        assert!(listed.len() > 2);
        assert!(listed.iter().any(|b| b.seed == hollow.seed && b.shape == FarShape::Sphere));
        assert!(far.sun_override().is_none());
    }
}
