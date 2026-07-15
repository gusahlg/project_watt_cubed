//! The camera as its own system: mode machine (Person/Free) plus effects layer.
//! Modes are exclusive — input routing and pose generation both discriminate on
//! the same arm. Renderer sees one [`ViewPose`] per frame, downstream doesn't know why.
use voxel_engine::{Camera3D, DVec3, Lens, Vec2, Vec3, WarpStrength};

use crate::input::look::PITCH_LIMIT;
use crate::interact;
use crate::player::Player;
use crate::world::World;

/// Default third-person boom length in blocks.
pub const THIRD_PERSON_DISTANCE: f64 = 4.0;

/// How far short of a wall the boom stops, so the near plane never pokes
/// through the surface the ray hit.
const BOOM_MARGIN: f64 = 0.2;

/// Camera rebase happens in `camera3d()` — f64 eye becomes f32 origin so far
/// terrain doesn't jitter.
#[derive(Clone, Copy)]
pub struct ViewPose {
    pub eye: DVec3,
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    /// Vertical FOV in degrees, pre-lens: values above 120° are resolved into
    /// the wide lens by [`camera3d`](Self::camera3d), not here.
    pub fovy: f32,
}

impl ViewPose {
    /// Full view direction from the angles — same construction as
    /// [`Player::forward`], `f64` trig of the `f32` angles.
    pub fn forward(&self) -> DVec3 {
        let (yaw, pitch) = (self.yaw as f64, self.pitch as f64);
        let (sin_yaw, cos_yaw) = yaw.sin_cos();
        let (sin_pitch, cos_pitch) = pitch.sin_cos();
        DVec3::new(cos_yaw * cos_pitch, sin_pitch, sin_yaw * cos_pitch)
    }

    /// Build the engine camera. Rebases the f64 eye to the origin so the engine
    /// can render all positions relative to it at f32. Above 120° fovy, the wide
    /// lens takes over to gain horizontal reach without vertical stretch.
    pub fn camera3d(&self) -> Camera3D {
        let (fovy, lens) = if self.fovy > 120.0 {
            let strength = WarpStrength::new((self.fovy - 120.0) / 50.0).unwrap();
            (120.0, Lens::WideFov { strength })
        } else {
            (self.fovy, Lens::Rectilinear)
        };
        let forward = self.forward().as_vec3();
        // Roll rotates up around forward. Zero roll keeps the Y-up the pipeline expects.
        let up = if self.roll != 0.0 {
            let y = Vec3::new(0.0, 1.0, 0.0);
            let (sin_r, cos_r) = self.roll.sin_cos();
            y * cos_r + forward.cross(y) * sin_r + forward * forward.dot(y) * (1.0 - cos_r)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        Camera3D {
            position: Vec3::ZERO,
            target: forward,
            up,
            fovy,
            lens,
        }
    }
}

/// The player-anchored views. Third person is one variant with a `front` flag
/// rather than two variants: the flag only mirrors the boom and angles, and the
/// F5-style cycle (first → back → front → first) stays a three-line match.
#[derive(Clone, Copy)]
pub enum PersonView {
    First,
    Third { distance: f64, front: bool },
}

impl PersonView {
    pub fn cycle(self) -> Self {
        match self {
            PersonView::First => PersonView::Third { distance: THIRD_PERSON_DISTANCE, front: false },
            PersonView::Third { distance, front: false } => PersonView::Third { distance, front: true },
            PersonView::Third { .. } => PersonView::First,
        }
    }

    /// Whether the player's own body should be drawn (any view that can see it).
    pub fn shows_body(self) -> bool {
        !matches!(self, PersonView::First)
    }
}

/// Continuous flight input for the detached rig, already resolved from
/// bindings at the call site so this module never reads devices.
#[derive(Clone, Copy, Default)]
pub struct FlyAxes {
    pub forward: f64,
    pub right: f64,
    pub up: f64,
    /// Held: multiply speed (sprint-equivalent).
    pub boost: bool,
}

/// A detached camera: its own `f64` position and angles, no collision, no
/// gameplay effect. Exists only inside [`CameraMode::Free`].
pub struct FreeRig {
    pub pos: DVec3,
    pub yaw: f32,
    pub pitch: f32,
    /// Flight speed in units/second.
    pub speed: f64,
}

impl FreeRig {
    /// Seed the rig from the pose it detaches from, so entering freecam is
    /// seamless (the first detached frame renders the identical view).
    fn from_pose(pose: ViewPose, speed: f64) -> Self {
        Self { pos: pose.eye, yaw: pose.yaw, pitch: pose.pitch, speed }
    }

    /// Apply an already-scaled look delta, same clamp as the player look path.
    pub fn look(&mut self, look: Vec2) {
        self.yaw += look.x;
        self.pitch = (self.pitch + look.y).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    /// No inertia — camera wants crisp stops, not player feel.
    pub fn fly(&mut self, axes: FlyAxes, dt: f32) {
        let (yaw, pitch) = (self.yaw as f64, self.pitch as f64);
        let forward = DVec3::new(yaw.cos() * pitch.cos(), pitch.sin(), yaw.sin() * pitch.cos());
        let right = DVec3::new(-yaw.sin(), 0.0, yaw.cos());
        let wish = forward * axes.forward + right * axes.right + DVec3::Y * axes.up;
        if wish != DVec3::ZERO {
            let speed = self.speed * if axes.boost { 3.0 } else { 1.0 };
            self.pos += wish.normalize() * speed * dt as f64;
        }
    }
}

/// Which camera is active. Input routing and pose generation both match on this
/// discriminant so player and rig never consume input in the same frame.
pub enum CameraMode {
    Person(PersonView),
    Free { rig: FreeRig, resume: PersonView },
}

/// Additive pose perturbations: currently shake. Effects are read-only on gameplay
/// state and decay autonomously.
pub struct CamFx {
    /// Amplitude follows `trauma²` so small hits whisper and big ones slam; linear
    /// decay reads as natural ring-down.
    trauma: f32,
    /// Advanced by wall dt while trauma is live.
    time: f32,
}

/// Roll sells impact but nauseates fastest, so gets less than yaw/pitch.
const SHAKE_ANGLE: f32 = 0.045;
const SHAKE_ROLL: f32 = 0.02;

impl Default for CamFx {
    fn default() -> Self {
        Self::new()
    }
}

impl CamFx {
    pub fn new() -> Self {
        Self { trauma: 0.0, time: 0.0 }
    }

    /// Inject shake energy (damage ~0.4, nearby explosion ~0.7, capped at 1).
    pub fn add_trauma(&mut self, amount: f32) {
        self.trauma = (self.trauma + amount).min(1.0);
    }

    /// Trauma drains in ~0.7s from full.
    pub fn update(&mut self, dt: f32) {
        if self.trauma > 0.0 {
            self.trauma = (self.trauma - dt * 1.4).max(0.0);
            self.time += dt;
        }
    }

    /// Pure in the pose so callers can't accumulate effects into state.
    pub fn apply(&self, mut pose: ViewPose, intensity: f32) -> ViewPose {
        let amp = self.trauma * self.trauma * intensity;
        if amp > 0.0 {
            // Three incommensurate-frequency sine pairs stand in for smooth
            // noise: cheap, deterministic, and no visible repetition inside a
            // ring-down's lifetime.
            let t = self.time;
            let n = |f1: f32, f2: f32| ((t * f1).sin() + (t * f2).sin() * 0.5) / 1.5;
            pose.yaw += amp * SHAKE_ANGLE * n(31.0, 17.3);
            pose.pitch += amp * SHAKE_ANGLE * n(27.7, 19.1);
            pose.roll += amp * SHAKE_ROLL * n(23.3, 13.7);
        }
        pose
    }
}

/// `Game::update` routes input by mode; `Game::draw` calls `pose()` and uses
/// nothing else.
pub struct GameCamera {
    pub mode: CameraMode,
    pub fx: CamFx,
}

impl Default for GameCamera {
    fn default() -> Self {
        Self::new()
    }
}

impl GameCamera {
    pub fn new() -> Self {
        Self { mode: CameraMode::Person(PersonView::First), fx: CamFx::new() }
    }

    /// The rig, when detached — `Game::update` routes look/fly input here
    /// instead of at the player.
    pub fn free_rig(&mut self) -> Option<&mut FreeRig> {
        match &mut self.mode {
            CameraMode::Free { rig, .. } => Some(rig),
            CameraMode::Person(_) => None,
        }
    }

    /// F5-style cycle. In freecam this cycles the view that will be resumed,
    /// which is harmless and avoids a modal error case.
    pub fn cycle_person(&mut self) {
        match &mut self.mode {
            CameraMode::Person(view) => *view = view.cycle(),
            CameraMode::Free { resume, .. } => *resume = resume.cycle(),
        }
    }

    /// Detach into freecam from the current view, or reattach to the view the
    /// rig was seeded from. Seeding from the live pose (not the player) means
    /// detaching from third person starts the rig at the boom position — the
    /// frame doesn't jump.
    pub fn toggle_freecam(&mut self, player: &Player, world: &World, base_fov: f32) {
        self.mode = match std::mem::replace(&mut self.mode, CameraMode::Person(PersonView::First)) {
            CameraMode::Person(view) => {
                let pose = person_pose(view, player, world, base_fov);
                CameraMode::Free {
                    rig: FreeRig::from_pose(pose, player.fly_speed),
                    resume: view,
                }
            }
            CameraMode::Free { resume, .. } => CameraMode::Person(resume),
        };
    }

    /// Applies effects (shake) on top of the mode's base pose.
    pub fn pose(&self, player: &Player, world: &World, base_fov: f32, shake: f32) -> ViewPose {
        let pose = match &self.mode {
            CameraMode::Person(view) => person_pose(*view, player, world, base_fov),
            CameraMode::Free { rig, .. } => ViewPose {
                eye: rig.pos,
                yaw: rig.yaw,
                pitch: rig.pitch,
                roll: 0.0,
                fovy: base_fov,
            },
        };
        self.fx.apply(pose, shake)
    }

    pub fn shows_body(&self) -> bool {
        match &self.mode {
            CameraMode::Person(view) => view.shows_body(),
            // Detached: the player is in the scene like any peer.
            CameraMode::Free { .. } => true,
        }
    }
}

/// Clamped to avoid terrain occlusion.
fn person_pose(view: PersonView, player: &Player, world: &World, base_fov: f32) -> ViewPose {
    match view {
        PersonView::First => ViewPose {
            eye: player.position,
            yaw: player.yaw,
            pitch: player.pitch,
            roll: 0.0,
            fovy: base_fov,
        },
        PersonView::Third { distance, front } => {
            let dir = if front { player.forward() } else { -player.forward() };
            let len = boom_clamp(world, player.position, dir, distance);
            let (yaw, pitch) = if front {
                (player.yaw + std::f32::consts::PI, -player.pitch)
            } else {
                (player.yaw, player.pitch)
            };
            ViewPose { eye: player.position + dir * len, yaw, pitch, roll: 0.0, fovy: base_fov }
        }
    }
}

/// Liquids passable so the camera doesn't snap into water.
fn boom_clamp(world: &World, eye: DVec3, dir: DVec3, max: f64) -> f64 {
    let Some(hit) = interact::raycast(world, eye, dir, max) else {
        return max;
    };
    let (bx, by, bz) = hit.block;
    let mut t_enter: f64 = 0.0;
    for (o, d, lo) in [
        (eye.x, dir.x, bx as f64),
        (eye.y, dir.y, by as f64),
        (eye.z, dir.z, bz as f64),
    ] {
        if d != 0.0 {
            let (t0, t1) = ((lo - o) / d, (lo + 1.0 - o) / d);
            t_enter = t_enter.max(t0.min(t1));
        }
    }
    (t_enter - BOOM_MARGIN).clamp(0.0, max)
}
