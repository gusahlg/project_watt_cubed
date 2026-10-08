//! Frame composition, world rendering, and HUD presentation for the live game.
use voxel_engine::{Camera3D, Color, DVec3, Engine, Vec2};

use super::Game;
use crate::avatar::Pose;
use crate::camera::ViewPose;
use crate::console;
use crate::derived::Memo;
use super::{DebugView, SKY_KEY, TERRAIN_KEY};
use crate::interact;
use crate::modding::Mods;
use crate::presence::{self, Eye, Feet, Gait, RenderPose, Stance, TagVisibility};
use crate::sched::RateGate;
use crate::sky::SkyFrame;
use crate::ui::{self, Anchor, HudMode};

/// Retained presentation state; gameplay only initializes it.
pub(super) struct DrawState {
    /// Camera orientation by frame/yaw/pitch/roll/FOV bits; translation stays separate.
    camera_cache: Memo<[u64; 6], Camera3D>,
    sky_frame_cache: Memo<(u64, [u32; 3]), SkyFrame>,
    /// Frozen lighting by day, content revision, space factor, and body up.
    static_frame_cache: Memo<(u64, u64, u32, [u32; 3]), StaticFrame>,
    anim_uv_cache: Memo<[u64; 2], [f32; 2]>,
    coord_cache: Memo<[i64; 4], String>,
    fps_cache: Memo<i32, String>,
    fps_refresh: RateGate,
    online_cache: Memo<(usize, Option<u32>), String>,
    peer_scratch: Vec<PeerDraw>,
    /// Last composed frame uniforms handed to `begin_3d`.
    last_uniforms: Option<voxel_engine::skeleton::FrameUniformsGpu>,
}

impl DrawState {
    pub(super) fn new() -> Self {
        Self {
            camera_cache: Memo::new(),
            sky_frame_cache: Memo::new(),
            static_frame_cache: Memo::new(),
            anim_uv_cache: Memo::new(),
            coord_cache: Memo::new(),
            fps_cache: Memo::new(),
            fps_refresh: RateGate::from_hz(4),
            online_cache: Memo::new(),
            peer_scratch: Vec::new(),
            last_uniforms: None,
        }
    }
}

/// Everything [`Game::compose_phase`] decides before frame recording starts:
/// the camera pose, the composed per-frame lighting truth, and peer render
/// poses — handed read-only to the scene and HUD phases. HUD strings stay in
/// their `Game`-side caches (no per-frame `String` clones into the scene).
struct Scene {
    pose: ViewPose,
    camera: Camera3D,
    lighting: Lighting,
    peers: Vec<PeerDraw>,
    screen: (i32, i32),
    dt: f32,
}

/// The frame's composed lighting truth: the source of the engine's per-frame
/// UBO for sky/fog and avatar key lighting, and the clear.
struct Lighting {
    /// The one clock sample every sun consumer shares this frame.
    sky_frame: SkyFrame,
    /// Body up and altitude above the local surface datum.
    sky_ctx: crate::frame_snapshot::SkyContext,
    uniforms: voxel_engine::skeleton::FrameUniformsGpu,
    clear: voxel_engine::LinearRgb,
    debug_flat: Option<Color>,
}

/// Lighting/clear state for a profile whose sky and animation inputs are
/// frozen (weather, clouds, and exposure all disabled).
/// Wrapped tangent-plane animation coordinates are cached independently so camera
/// motion does not force the palette and lighting packet to be recomposed.
#[derive(Clone, Copy)]
struct StaticFrame {
    uniforms: voxel_engine::skeleton::FrameUniformsGpu,
    clear: voxel_engine::LinearRgb,
}

/// Quantised body up, and the eye projected onto the local tangent plane.
/// The plane coords key the anim-UV cache: a still eye skips `rem_euclid`.
fn sky_keys(eye: DVec3, up: DVec3) -> ([u32; 3], [f64; 2]) {
    let u = up.as_vec3();
    let up_q = [u.x.to_bits(), u.y.to_bits(), u.z.to_bits()];
    let (t, _, b) = voxel_engine::local_sky_basis(u);
    let plane = [
        eye.dot(DVec3::new(t.x as f64, t.y as f64, t.z as f64)),
        eye.dot(DVec3::new(b.x as f64, b.y as f64, b.z as f64)),
    ];
    (up_q, plane)
}

fn hud_label(
    f: &mut voxel_engine::Frame,
    theme: &crate::ui::Theme,
    screen: (i32, i32),
    at: Anchor,
    off: (i32, i32),
    base_fs: i32,
    color: Color,
    text: &str,
) {
    ui::label(f, theme, screen, at, off, base_fs, color, text);
}

impl Game {
    /// Render the world and HUD.
    ///
    /// The camera is at the origin looking along the view direction, and all
    /// 3D draws are camera-relative. Differences are computed at f64 precision
    /// before narrowing to f32 for the GPU, keeping far terrain stable.
    pub fn draw(&mut self, eng: &mut Engine, mods: &mut Mods, fov: f32, shake: f32) {
        // The home map is installed on the engine, so it has to land before the frame opens.
        if matches!(self.debug_view, DebugView::Normal) {
            // Benchmarks and the scripted harness stay on the sphere unless a bench opts in.
            if crate::sky::planet_map::bake_wanted(self.scripted) {
                self.sky.drive_planet(
                    || self.world.terrain_arc(),
                    self.world.registry(),
                    self.world.worldgen(),
                    self.world.terrain_cfg(),
                );
            }
            self.sky.sync_far_map(eng, self.world.terrain());
        }
        let mut scene = self.compose_phase(eng, fov, shake);
        let mut f = eng.begin_frame(scene.lighting.clear);
        self.scene_phase(&mut f, &scene);
        self.hud_phase(&mut f, mods, &scene);
        // Reclaim peer capacity after both consumers finish with the immutable
        // scene. Stable multiplayer frames allocate no new draw-record vector.
        self.drawing.peer_scratch = std::mem::take(&mut scene.peers);
        self.drawing.peer_scratch.clear();
    }

    /// Everything a frame needs decided BEFORE recording starts: the camera
    /// pose, the per-frame lighting truth (the engine UBO's single source),
    /// peer render poses, and the refreshed HUD string caches.
    fn compose_phase(&mut self, eng: &mut Engine, fov: f32, shake: f32) -> Scene {
        let dt = eng.frame_time();
        // The one pose this frame renders from: mode observation plus effects.
        // Orientation is cached by exact angle/lens bit patterns — the f64 eye
        // stays outside the key, so translation with unchanged orientation
        // reuses the basis and repeats none of its trigonometry.
        let pose = self.camera.pose(&self.player, &self.world, fov, shake);
        let f = pose.frame;
        let camera_key = [
            f.x.to_bits(),
            f.y.to_bits(),
            f.z.to_bits(),
            f.w.to_bits(),
            (pose.yaw.to_bits() as u64) << 32 | pose.pitch.to_bits() as u64,
            (pose.roll.to_bits() as u64) << 32 | pose.fovy.to_bits() as u64,
        ];
        let camera = *self
            .drawing
            .camera_cache
            .get_or(camera_key, || pose.camera3d());

        self.refresh_hud_text(eng.fps(), dt);
        let screen = (eng.screen_width(), eng.screen_height());

        // `dt` steps each peer's animator (body-yaw follow, stance blend, swing).
        let want_tags = self.name_tags && self.theme.hud.shows_world_ui();
        let peers = self.peer_draws(screen, &camera, &pose, dt, self.player_models, want_tags);

        // Exposure is the render thread's latest metered+smoothed value,
        // sourced through `Engine::exposure_for_compose`; temporal smoothing
        // already happened render-side, so frame delta is passed only for
        // signature symmetry (unused there). With the exposure lane off, the
        // metered read is skipped entirely.
        // Allow pinning exposure to a fixed default for stable bless/debug output.
        static EXPOSURE_ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            !matches!(std::env::var("WATT_EXPOSURE").as_deref(), Ok("0"))
        });
        let exposure = if self.render.exposure && *EXPOSURE_ON {
            eng.exposure_for_compose(dt)
        } else {
            voxel_engine::skeleton::Exposure::DEFAULT
        };
        let lighting = self.compose_lighting(&pose, exposure);

        Scene {
            pose,
            camera,
            lighting,
            peers,
            screen,
            dt,
        }
    }

    /// Compose the single per-frame lighting truth for `pose`: the source for
    /// the engine's per-frame UBO for sky/fog and avatar key lighting. The UBO
    /// is the only path; legacy push lanes have been retired.
    fn compose_lighting(&mut self, pose: &ViewPose, exposure: voxel_engine::skeleton::Exposure) -> Lighting {
        // ONE clock sample for lighting, clear colour, and sky geometry,
        // cached by the quantised day and body up. Day/night off renders fixed
        // noon (cheap, readable stripped-profile lighting) while the
        // authoritative clock keeps its stored time for networking and re-enables.
        let sky_day = if self.render.day_night {
            // 1/4096 of a day (~0.15 s of a 600 s day) — imperceptible, and
            // the scripted/golden path pins the clock so goldens stay bit-stable.
            (self.sky.clock.day() * 4096.0).round() / 4096.0
        } else {
            0.5
        };
        let sky_ctx = crate::frame_snapshot::SkyContext {
            up: pose.up(),
            altitude: self.sky_altitude(pose.eye),
            fade: self.space_fade(),
        };
        let (up_q, plane) = sky_keys(pose.eye, sky_ctx.up);
        let up = sky_ctx.up.as_vec3();
        let sky = &self.sky;
        let sky_frame = *self
            .drawing
            .sky_frame_cache
            .get_or((sky_day.to_bits(), up_q), || sky.frame_at_day(sky_day, up));

        // Tangent-plane animation coordinates, recomputed only when that projection changes.
        let uv_key = [plane[0].to_bits(), plane[1].to_bits()];
        let anim_uv = *self.drawing.anim_uv_cache.get_or(uv_key, || {
            crate::frame_snapshot::wrap_plane(plane[0], plane[1])
        });

        // With weather, clouds, and exposure all disabled the composed packet is
        // a pure function of the day, the body up, and the altitude's space fade
        // (clouds off also freezes the engine's animation clock): cache it and
        // patch only the camera-anchored UV lanes.
        // Minimum/Fast ride this path.
        let cacheable_frame = !self.render.weather && !self.render.clouds && !self.render.exposure;
        let (mut uniforms, cached_clear) = if cacheable_frame {
            let render = &self.render;
            // (day, content_rev, altitude's space fade, body up): any render/palette
            // change bumps the stamp, so the freeze predicate's own inputs invalidate
            // the entry structurally.
            let space = crate::frame_snapshot::space_factor(sky_ctx.altitude, sky_ctx.fade).to_bits();
            let key = (sky_day.to_bits(), self.content_rev.0, space, up_q);
            let cached = self.drawing.static_frame_cache.get_or(key, || {
                let snapshot = crate::frame_snapshot::compose_at(
                    sky, sky_frame, sky_ctx, anim_uv, exposure, render,
                );
                StaticFrame {
                    uniforms: voxel_engine::skeleton::FrameUniformsGpu::from(&snapshot),
                    clear: sky.clear_at(sky_frame, up),
                }
            });
            (cached.uniforms, Some(cached.clear))
        } else {
            let snapshot = crate::frame_snapshot::compose_at(
                sky,
                sky_frame,
                sky_ctx,
                anim_uv,
                exposure,
                &self.render,
            );
            (
                voxel_engine::skeleton::FrameUniformsGpu::from(&snapshot),
                None,
            )
        };
        if cacheable_frame {
            // The camera-anchored UV lanes are the only inputs that can differ
            // while lighting is frozen; exposure and jitter are fixed by the
            // cache predicate.
            uniforms.anim[1] = anim_uv[0];
            uniforms.anim[2] = anim_uv[1];
        }
        if self.drawing.last_uniforms != Some(uniforms) {
            self.drawing.last_uniforms = Some(uniforms);
            #[cfg(test)]
            crate::alloc_count::note_engine(crate::alloc_count::EngineCall::FrameUniforms);
        }

        // TerrainKey: flat terrain, sky/fog disabled, magenta clear for the
        // sky-hole detector. Normal: real clear, no debug flat.
        let (clear, debug_flat) = match self.debug_view {
            DebugView::Normal => (
                cached_clear.unwrap_or_else(|| self.sky.clear_at(sky_frame, up)),
                None,
            ),
            // Pure-magenta endpoints (255/0) decode identically under sRGB and raw
            // normalize, so the sky-hole detector's HDR key value is unchanged.
            DebugView::TerrainKey => (
                SKY_KEY.to_linear(),
                Some(TERRAIN_KEY),
            ),
        };
        Lighting {
            sky_frame,
            sky_ctx,
            uniforms,
            clear,
            debug_flat,
        }
    }

    /// Refresh the cached HUD strings (coordinates, FPS, players-online) only
    /// when their displayed value changes — and not at all below Full HUD.
    fn refresh_hud_text(&mut self, fps: i32, dt: f32) {
        if !self.theme.hud.shows_info() {
            return;
        }
        let p = self.player.position;
        // 0.1-block display resolution: only re-format when a shown digit moves.
        let cruise = self.player.cruise.map(|c| c.speed * crate::math::BLOCK_METERS / 1000.0);
        let key = [
            (p.x * 10.0) as i64,
            (p.y * 10.0) as i64,
            (p.z * 10.0) as i64,
            cruise.map_or(-1, |km_s| km_s as i64),
        ];
        self.drawing.coord_cache.get_or(key, || match cruise {
            Some(km_s) => format!("X: {:.1}    Y: {:.1}    Z: {:.1}    CRUISE {km_s:.0} km/s", p.x, p.y, p.z),
            None => format!("X: {:.1}    Y: {:.1}    Z: {:.1}", p.x, p.y, p.z),
        });
        // Scripted (harness) frames pin the readout: a live FPS number is the
        // one nondeterministic pixel region in an otherwise reproducible shot,
        // and golden diffs must only ever see real rendering drift. Live FPS is
        // sampled at human display cadence and re-formatted only when the
        // displayed integer changes.
        if self.scripted {
            self.drawing.fps_cache.get_or(-1, || "-- FPS".to_string());
        } else if self.drawing.fps_refresh.steps(dt) != 0 || self.drawing.fps_cache.get().is_none()
        {
            self.drawing
                .fps_cache
                .get_or(fps, || format!("{fps:2} FPS"));
        }
        if let Some(net) = &self.net {
            let count = net.peers().count() + 1;
            let ping = net.ping_ms();
            self.drawing
                .online_cache
                .get_or((count, ping), || match ping {
                    Some(ms) => format!("players online: {count}   {ms} ms"),
                    None => format!("players online: {count}"),
                });
        }
    }

    /// The 3D scope: sky, world, and every humanoid, all camera-relative.
    fn scene_phase(&mut self, f: &mut voxel_engine::Frame, scene: &Scene) {
        let Scene {
            pose,
            camera,
            peers,
            dt,
            ..
        } = scene;
        {
            // The pose's f64 eye is the render-space origin for camera rebase:
            // TAA's translation reprojection depends on this.
            // Lighting is decided when the 3D scope opens (no post-hoc setter):
            // the composed per-frame UBO carries the lighting truth in every mode
            // (the renderer overlays the debug-flat reserved key for TerrainKey).
            let lit = &scene.lighting;
            let mut f3 = f.begin_3d(
                camera,
                pose.eye,
                voxel_engine::Lighting::Composed(lit.uniforms),
            );
            f3.set_debug_flat(lit.debug_flat);
            // Fog and water read the same basis as the sky, including debug-flat frames.
            f3.set_local_frame(lit.sky_ctx.up.as_vec3(), lit.sky_ctx.altitude as f32);
            if matches!(self.debug_view, DebugView::Normal) {
                let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::ListSky);
                let view_blocks = crate::sky::chunk_view_blocks(self.world.view_radius());
                self.sky.draw(
                    &mut f3,
                    lit.sky_frame,
                    pose.eye,
                    self.world.terrain(),
                    view_blocks,
                );
            }
            let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::ListWorld);
            self.world.render(&mut f3, pose.eye);
            // Other players: a six-box humanoid, head tracking their look and
            // body lazily following, limbs swinging with their gait. Poses are
            // already camera-relative (see peer_draws). A tags-only record
            // carries no model at all — nothing composed, nothing to skip here.
            for peer in peers {
                if let Some((rp, rig)) = &peer.model {
                    Pose::resolve(rp, rig).draw(&mut f3, peer.color, true);
                }
            }
            // The player's own body — third person and freecam only. First
            // person draws NOTHING of it (no pose, no animator step, no
            // shadow): seeing your own torso from inside is noise, and the
            // common first-person frame skips the whole block.
            if self.camera.shows_body() {
                let feet = Feet(self.player.feet());
                let mut v = self.player.velocity();
                v[self.player.up_axis.axis()] = 0.0;
                let speed = v.length() as f32;
                let rp = RenderPose::new(
                    feet,
                    Eye(pose.eye),
                    self.player.orientation.yaw,
                    self.player.orientation.pitch,
                    self.player.orientation.frame,
                    Stance::of_player(&self.player),
                    Gait::new(self.local_gait as f32, speed),
                );
                let rig = self.local_anim.step(&rp, *dt);
                Pose::resolve(&rp, &rig).draw(&mut f3, self.local_color, true);
            }
        }
    }

    /// Everything over the world: minimap, reticle, name tags, info text, the
    /// mods' HUD data (rendered by the core — mods never touch the frame), and
    /// the console on top.
    fn hud_phase(&mut self, f: &mut voxel_engine::Frame, mods: &mut Mods, scene: &Scene) {
        let map_sample = self.map_sample.take();
        // HUD Off records nothing at all — unless the console is open, which
        // must stay reachable in every mode.
        if matches!(self.theme.hud, HudMode::Off) && !self.console.is_open() {
            return;
        }
        let screen = scene.screen;
        let _hud = voxel_engine::profile::scope(voxel_engine::profile::Meter::ListHud);
        let theme = &self.theme;

        // Minimap: informational, so Full mode only (HUD Off must blank it too).
        if theme.hud.shows_minimap()
            && let Some(minimap) = &self.minimap
        {
            let sample = map_sample.unwrap_or_else(|| crate::minimap::MapSample::of(&self.world, &self.player));
            minimap.draw(f, screen, sample);
        }

        // Reticle and world-space name tags: shown in every mode but fully-off.
        if theme.hud.shows_world_ui() {
            theme.crosshair.draw(f, screen);

            // Floating name tags over each visible player, in the peer's own
            // tint, fading with distance and dimming when terrain occludes the
            // head (instead of drawing full-strength through walls).
            for peer in &scene.peers {
                if let Some(tag) = &peer.tag {
                    let fs = theme.fs(18);
                    let tw = f.measure_text(&tag.name, fs);
                    let c = peer.color;
                    console::shadowed(
                        f,
                        &tag.name,
                        tag.screen.x as i32 - tw / 2,
                        tag.screen.y as i32,
                        fs,
                        Color::new(c.r, c.g, c.b, (tag.alpha * 255.0) as u8),
                    );
                }
            }
        }

        // Informational HUD text: coords, help, FPS, player count. Full mode
        // only — read from the `Game`-side caches `refresh_hud_text` maintains.
        // Loading covers Full and Minimal (not Off) until the spawn slab lands.
        // A network join holds "Loading world…" until the edit overlay arrives.
        let overlay_hold = self.net.as_ref().is_some_and(|net| !net.snapshot_ready());
        if overlay_hold && theme.hud.shows_world_ui() {
            ui::label(
                f,
                theme,
                screen,
                Anchor::Top,
                (0, 12),
                26,
                ui::Role::Primary.color(),
                "Loading world…",
            );
        } else if !self.world.spawn_ready() && theme.hud.shows_world_ui() {
            ui::label(
                f,
                theme,
                screen,
                Anchor::Top,
                (0, 12),
                26,
                ui::Role::Primary.color(),
                "Loading terrain…",
            );
        } else if theme.hud.shows_info() {
            // The centred coordinates shrink to fit between the FPS readout and the minimap.
            let fps_w = self.drawing.fps_cache.get().map_or(0, |t| 10 + f.measure_text(t, theme.fs(20)));
            let map_w = self.minimap.as_ref().filter(|_| theme.hud.shows_minimap()).map_or(0, |m| m.reserved_width());
            let side = fps_w.max(map_w) + 12;
            if let Some(coord_text) = self.drawing.coord_cache.get() {
                let color = ui::Role::Primary.color();
                ui::label_fit(f, theme, screen, Anchor::Top, (0, 12), 26, screen.0 - 2 * side, color, coord_text);
            }
            if let Some(fps_text) = self.drawing.fps_cache.get() {
                hud_label(f, theme, screen, Anchor::TopLeft, (10, 12), 20, ui::Role::Positive.color(), fps_text);
            }
            if self.net.is_some()
                && let Some(online_text) = self.drawing.online_cache.get()
            {
                hud_label(f, theme, screen, Anchor::TopRight, (-12, 180), 20, ui::Role::Positive.color(), online_text);
            }
        }

        if self.net.as_ref().is_some_and(|net| net.link_interrupted()) && theme.hud.shows_world_ui() {
            ui::label(
                f,
                theme,
                screen,
                Anchor::Top,
                (0, 44),
                22,
                ui::Role::Warning.color(),
                crate::net::client::INTERRUPTED,
            );
        }

        // Enabled mods contribute their HUD as data; the core renders it over the
        // world, under the console. Mods never touch the frame themselves.
        // Gameplay UI, so it follows the reticle: hidden only when HUD is Off
        // or the mod-HUD lane itself is disabled.
        if self.mod_hud && theme.hud.shows_mod_hud() {
            self.hud_scratch.clear();
            mods.hud(&self.world, &self.player, screen, &mut self.hud_scratch);
            ui::render_hud(f, theme, screen, &self.hud_scratch);
        }
        // Minimal keeps the world readable: no closed-console scrollback.
        if matches!(theme.hud, HudMode::Full) || self.console.is_open() {
            self.console.draw(f, screen.0, screen.1);
        }
    }

    /// Lighting + HUD string caches for a headless quiet frame (no Engine).
    #[cfg(test)]
    pub(super) fn compose_quiet(&mut self, mods: &mut Mods) {
        const DT: f32 = 1.0 / 60.0;
        self.refresh_hud_text(0, DT);
        let pose = self.camera.pose(&self.player, &self.world, 90.0, 0.0);
        self.compose_lighting(&pose, voxel_engine::skeleton::Exposure::DEFAULT);
        if self.mod_hud && self.theme.hud.shows_mod_hud() {
            self.hud_scratch.clear();
            mods.hud(
                &self.world,
                &self.player,
                (1280, 720),
                &mut self.hud_scratch,
            );
        }
    }

    /// Build the per-frame draw data for other players. `&mut self` because
    /// animator state advances here. Disabled models skip animator and rig
    /// composition entirely; disabled tags skip projection, occlusion
    /// raycasts, and name cloning — each stops at its owning boundary.
    fn peer_draws(
        &mut self,
        screen: (i32, i32),
        camera: &Camera3D,
        pose: &ViewPose,
        dt: f32,
        want_models: bool,
        want_tags: bool,
    ) -> Vec<PeerDraw> {
        let mut draws = std::mem::take(&mut self.drawing.peer_scratch);
        draws.clear();
        if !want_models && !want_tags {
            return draws;
        }
        let Some(net) = &mut self.net else {
            return draws;
        };
        let world = &self.world;
        let eye = pose.eye;
        let forward = pose.forward();
        let (screen_w, screen_h) = (screen.0 as f32, screen.1 as f32);
        // The frame's poses were sampled in this same order. Outside interest
        // range there is no live pose: drawing the last heard one would freeze
        // a ghost in place.
        for (peer, frame) in net.peers_mut().zip(&self.peer_frames) {
            debug_assert_eq!(peer.id(), frame.id);
            if !frame.visible {
                continue;
            }
            let r = &frame.rendered;
            let feet = r.pos.feet(r.stance, r.up);
            let color = peer_color(&peer.name);
            let model = want_models.then(|| {
                let rp = RenderPose::new(
                    feet,
                    Eye(eye),
                    r.yaw,
                    r.pitch,
                    r.frame,
                    r.stance,
                    Gait::new(r.phase, r.speed),
                );
                let rig = peer.anim.step(&rp, dt);
                (rp, rig)
            });
            let tag = if want_tags {
                let head = feet.0
                    + crate::camera::rotate(r.frame, DVec3::Y) * (Pose::HEAD_TOP as f64 + 0.2);
                let to_head = head - eye;
                tag_draw(world, eye, forward, to_head, camera, screen_w, screen_h, dt, peer)
            } else {
                None
            };
            // A tags-only peer that is entirely hidden needs no record at all.
            if model.is_some() || tag.is_some() {
                draws.push(PeerDraw { model, color, tag });
            }
        }
        draws
    }
}

/// Everything needed to draw one other player this frame.
struct PeerDraw {
    /// Camera-relative render pose (world minus eye, subtracted in f64, then
    /// narrowed) — safe to hand to the f32 immediate draws — plus this frame's
    /// animator output. Absent in name-tags-only mode, so animator and
    /// humanoid composition are skipped rather than merely hidden at draw time.
    model: Option<(RenderPose, presence::RigParams)>,
    color: Color,
    /// The visible name tag, or `None` when off-screen, behind us, occluded
    /// past range, or tags are disabled. Screen position, fade, and the shared
    /// name travel together — they exist together by construction.
    tag: Option<PeerTag>,
}

/// One on-screen name tag: everything the HUD pass draws for it.
struct PeerTag {
    screen: Vec2,
    /// Distance/occlusion fade in `[0, 1]`.
    alpha: f32,
    /// Refcount bump of the connection's interned name, never a fresh String.
    name: std::sync::Arc<str>,
}

/// Distance if the tag is in front and inside range; `None` is a cheap reject.
fn tag_candidate(to_head: DVec3, forward: DVec3) -> Option<f64> {
    let distance = to_head.length();
    if !(to_head.dot(forward) > 0.0 && distance > 1e-6)
        || matches!(TagVisibility::of(distance, false), TagVisibility::Hidden)
    {
        return None;
    }
    Some(distance)
}

/// Screen-space reject before the occlusion raycast.
fn tag_on_screen(screen: Vec2, w: f32, h: f32) -> bool {
    screen.x >= 0.0 && screen.y >= 0.0 && screen.x <= w && screen.y <= h
}

fn tag_draw(
    world: &crate::world::World,
    eye: DVec3,
    forward: DVec3,
    to_head: DVec3,
    camera: &Camera3D,
    screen_w: f32,
    screen_h: f32,
    dt: f32,
    peer: &mut crate::net::client::RemotePlayer,
) -> Option<PeerTag> {
    let distance = tag_candidate(to_head, forward)?;
    let screen = voxel_engine::world_to_screen(to_head.as_vec3(), camera, screen_w, screen_h);
    if !tag_on_screen(screen, screen_w, screen_h) {
        return None;
    }
    let occluded = peer.cached_tag_occlusion(dt, || {
        interact::raycast(world, eye, to_head / distance, (distance - 0.5).max(0.0)).is_some()
    });
    match TagVisibility::of(distance, occluded) {
        TagVisibility::Hidden => None,
        TagVisibility::Visible { alpha } => Some(PeerTag {
            screen,
            alpha,
            name: peer.name.clone(),
        }),
    }
}

/// Reject tags by direction and distance before querying terrain occlusion.
#[cfg(test)]
fn tag_visibility(
    to_head: DVec3,
    forward: DVec3,
    occluded: impl FnOnce(f64) -> bool,
) -> TagVisibility {
    let Some(distance) = tag_candidate(to_head, forward) else {
        return TagVisibility::Hidden;
    };
    TagVisibility::of(distance, occluded(distance))
}

/// A stable, cheerful colour for a player, hashed from their name so the same player
/// keeps the same tint across clients.
pub(super) fn peer_color(name: &str) -> Color {
    const PALETTE: [Color; 6] = [
        Color::new(230, 90, 90, 255),
        Color::new(90, 170, 230, 255),
        Color::new(110, 210, 120, 255),
        Color::new(230, 190, 90, 255),
        Color::new(200, 120, 220, 255),
        Color::new(240, 150, 90, 255),
    ];
    // FNV-1a over the name, then index the palette.
    let h = crate::hash::fnv1a_32(name.as_bytes());
    PALETTE[h as usize % PALETTE.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_tags_never_query_terrain() {
        for head in [
            DVec3::ZERO,
            -DVec3::Z,
            DVec3::X,
            DVec3::Z * 1e-7,
            DVec3::Z * TagVisibility::RANGE,
            DVec3::Z * (TagVisibility::RANGE + 1.0),
        ] {
            assert!(matches!(
                tag_visibility(head, DVec3::Z, |_| panic!("hidden tag queried terrain")),
                TagVisibility::Hidden
            ));
        }
    }

    #[test]
    fn visible_tags_keep_distance_and_occlusion_fades() {
        for distance in [1.0, 75.0, 80.0, 89.0] {
            for blocked in [false, true] {
                let mut queries = 0;
                let actual = tag_visibility(DVec3::Z * distance, DVec3::Z, |d| {
                    queries += 1;
                    assert_eq!(d, distance);
                    blocked
                });
                let TagVisibility::Visible { alpha: expected } =
                    TagVisibility::of(distance, blocked)
                else {
                    panic!("test distance should be visible");
                };
                let TagVisibility::Visible { alpha } = actual else {
                    panic!("visible tag was hidden");
                };
                assert_eq!(alpha, expected);
                assert_eq!(queries, 1);
            }
        }
    }

    #[test]
    fn off_screen_tags_are_rejected_by_bounds() {
        assert!(!tag_on_screen(Vec2::new(-1.0, 10.0), 100.0, 100.0));
        assert!(!tag_on_screen(Vec2::new(10.0, -1.0), 100.0, 100.0));
        assert!(!tag_on_screen(Vec2::new(101.0, 10.0), 100.0, 100.0));
        assert!(!tag_on_screen(Vec2::new(10.0, 101.0), 100.0, 100.0));
        assert!(tag_on_screen(Vec2::new(0.0, 0.0), 100.0, 100.0));
        assert!(tag_on_screen(Vec2::new(100.0, 100.0), 100.0, 100.0));
        assert!(tag_on_screen(Vec2::new(50.0, 50.0), 100.0, 100.0));
    }
}
