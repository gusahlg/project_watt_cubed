//! A top-down minimap: a throttled RGBA raster of the terrain around the player,
//! uploaded to a dedicated engine texture and drawn as one rotatable/zoomable 2D
//! quad in the HUD corner.
//!
//! The raster is the plane perpendicular to the player's up axis (face-local
//! `(u, v)`). On a round body it is the storage chart around the stream eye,
//! where storage +Y is up. PosY on a flat world is world XZ, unchanged.
//!
//! The CPU rebuild ([`Minimap::refresh`]) scans the loaded columns' top solid
//! blocks, folds in slope shading, and hands the pixels to the engine on a
//! throttle. The per-frame [`Minimap::draw`] is just one textured quad, so it
//! costs two triangles regardless of the raster resolution.

use std::time::Duration;

use glam::DQuat;
use voxel_engine::{Color, DVec3, Engine, Frame, IVec2, Vec2};

use crate::camera::rotate;
use crate::coord::Face;
use crate::space::FaceFrame;
use crate::world::World;

/// How the map is oriented relative to the world.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rotation {
    /// North (the face frame's −t_v; world −Z on +Y) always points up; the map
    /// never rotates and the player marker spins instead.
    NorthUp,
    /// The map rotates so the player's facing is always up.
    Heading,
}

#[derive(Clone, Copy)]
pub struct MinimapConfig {
    /// Raster edge in texels (must match the engine's `MINIMAP_SIZE`).
    pub size: u16,
    /// On-screen quad edge in pixels.
    pub screen_px: u16,
    /// HUD inset from the corner, in pixels `(x, y)`.
    pub margin: (i32, i32),
    /// Minimum wall-clock between CPU rebuilds.
    pub refresh_every: Duration,
    /// Re-center once the player has moved this many blocks from the raster's
    /// current center. A move of `d < size` shifts and repaints the exposed
    /// strip; `d ≥ size` (or the interval) does a full rescan.
    pub recenter_after: u16,
    /// Colour painted where a column has no loaded solid block.
    pub void: Color,
    pub orient: Rotation,
}

impl MinimapConfig {
    pub const DEFAULT: Self = Self {
        size: 256,
        screen_px: 160,
        margin: (12, 12),
        refresh_every: Duration::from_millis(500),
        recenter_after: 16,
        void: Color::rgb(18, 18, 24),
        orient: Rotation::NorthUp,
    };
}

/// Where the map is looking: the face whose columns it rasters, the face-local
/// `(u, v)` column under the eye, and the marker heading in that plane.
/// A round-body chart uses storage PosY around [`World::stream_eye`].
#[derive(Clone, Copy)]
pub struct MapSample {
    pub face: Face,
    pub col: IVec2,
    pub heading: f32,
}

impl MapSample {
    /// [`from_player`](Self::from_player) for `player`'s pose.
    pub fn of(world: &World, player: &crate::player::Player) -> Self {
        let o = &player.orientation;
        Self::from_player(world, player.position, player.up_axis, o.frame, o.yaw)
    }

    pub fn from_player(world: &World, eye: DVec3, up: Face, frame: DQuat, yaw: f32) -> Self {
        if let Some(storage) = world.chart_eye(eye) {
            let heading = chart_heading(world, eye, frame, yaw)
                .unwrap_or_else(|| face_heading(frame, yaw, Face::PosY));
            return Self {
                face: Face::PosY,
                col: IVec2::new(storage.x.floor() as i32, storage.z.floor() as i32),
                heading,
            };
        }
        let local = FaceFrame::new(up).point_to_local(eye);
        Self {
            face: up,
            col: IVec2::new(local.x.floor() as i32, local.z.floor() as i32),
            heading: face_heading(frame, yaw, up),
        }
    }
}

/// Face-local heading `atan2(forward·t_v, forward·t_u)`. On PosY with an identity
/// frame this is the world yaw, so the +Y marker keeps its old angle bits.
fn face_heading(frame: DQuat, yaw: f32, face: Face) -> f32 {
    if face == Face::PosY && frame == DQuat::IDENTITY {
        return yaw;
    }
    let (sin_yaw, cos_yaw) = (yaw as f64).sin_cos();
    let forward = rotate(frame, DVec3::new(cos_yaw, 0.0, sin_yaw));
    let basis = FaceFrame::new(face);
    let tu = basis.point_to_world(DVec3::X);
    let tv = basis.point_to_world(DVec3::Z);
    (forward.dot(tv) as f32).atan2(forward.dot(tu) as f32)
}

/// Heading in the chart's storage XZ, from the patch Jacobian. `None` above the band.
fn chart_heading(world: &World, eye: DVec3, frame: DQuat, yaw: f32) -> Option<f32> {
    let local = world.chart_local(eye)?;
    let (sin_yaw, cos_yaw) = (yaw as f64).sin_cos();
    let forward = rotate(frame, DVec3::new(cos_yaw, 0.0, sin_yaw));
    let rot = local.rotation();
    Some((forward.dot(rot.z_axis) as f32).atan2(forward.dot(rot.x_axis) as f32))
}

pub struct Minimap {
    cfg: MinimapConfig,
    /// Latest RGBA raster (`size² * 4`), reused across refreshes.
    rgba: Vec<u8>,
    /// Per-texel top-solid height, scratch for slope shading (`size²`).
    top_y: Vec<i32>,
    /// The block column the current raster is centered on (`None` = never built).
    /// The recenter half of the refresh gate compares the player against this
    /// directly; the throttle half rides the scheduler's interval gate, so no
    /// `Instant` lives here.
    center: Option<IVec2>,
    /// Face of `center`. A column from another face is not a shift of this one.
    face: Face,
    /// [`World::surface_stamp`] of the area the raster last painted.
    stamp: u64,
}

impl Minimap {
    pub fn new(cfg: MinimapConfig) -> Self {
        assert_eq!(cfg.size, 256, "minimap size must match engine MINIMAP_SIZE");
        let texels = cfg.size as usize * cfg.size as usize;
        Self {
            cfg,
            rgba: vec![0u8; texels * 4],
            top_y: vec![i32::MIN; texels],
            center: None,
            face: Face::PosY,
            stamp: 0,
        }
    }

    /// Toggle between north-up (fixed map, spinning marker) and heading-up
    /// (rotating map, marker locked pointing up).
    pub fn toggle_rotation(&mut self) {
        self.cfg.orient = match self.cfg.orient {
            Rotation::NorthUp => Rotation::Heading,
            Rotation::Heading => Rotation::NorthUp,
        };
    }

    /// Whether a rebuild is due: never built, OR the throttle elapsed, OR the
    /// player moved past the recenter distance.
    pub fn due(&self, face: Face, player_col: IVec2, interval_elapsed: bool) -> bool {
        match self.center {
            None => true,
            Some(_) if self.face != face => true,
            Some(c) => {
                let moved = (player_col.x - c.x).abs().max((player_col.y - c.y).abs());
                interval_elapsed || moved >= self.cfg.recenter_after as i32
            }
        }
    }

    /// Throttled + recenter-gated rescan: when [`Self::due`], rebuilds `rgba`
    /// from the world's top-solid columns (colour × slope-shade) and uploads it
    /// via [`Engine::update_minimap`]. `interval_elapsed` is the scheduler's
    /// throttle decision. Returns `true` when the gate was due (painted or
    /// found unchanged), so the caller resets the scheduler's interval gate on
    /// the attempt.
    pub fn refresh(
        &mut self,
        eng: &mut Engine,
        world: &World,
        sample: MapSample,
        interval_elapsed: bool,
    ) -> bool {
        let Some(painted) = self.rebuild(world, sample.face, sample.col, interval_elapsed) else {
            return false;
        };
        if painted {
            eng.update_minimap(&self.rgba);
        }
        true
    }

    /// CPU half of [`Self::refresh`]: full rebuild on the interval / first
    /// build / a face change / `d ≥ size`, otherwise shift the raster and
    /// repaint exposed strips. An interval rebuild over an unchanged surface
    /// under a still player is skipped. `None` when not due, else whether it
    /// painted.
    fn rebuild(&mut self, world: &World, face: Face, player_col: IVec2, interval_elapsed: bool) -> Option<bool> {
        if !self.due(face, player_col, interval_elapsed) {
            return None;
        }
        let size = self.cfg.size as i32;
        let face_changed = self.center.is_some() && self.face != face;
        self.face = face;
        let stamp = self.surface_stamp(world, player_col);
        match self.center {
            Some(prev) if !interval_elapsed && !face_changed => {
                let dx = player_col.x - prev.x;
                let dz = player_col.y - prev.y;
                let d = dx.abs().max(dz.abs());
                if d > 0 && d < size {
                    self.rebuild_shift(world, player_col, dx, dz);
                } else {
                    self.rebuild_full(world, player_col);
                }
            }
            Some(prev) if prev == player_col && !face_changed && stamp == self.stamp => return Some(false),
            _ => self.rebuild_full(world, player_col),
        }
        self.center = Some(player_col);
        self.stamp = stamp;
        Some(true)
    }

    /// [`World::surface_stamp`] over the chunk columns a raster centred on `player_col` covers.
    fn surface_stamp(&self, world: &World, player_col: IVec2) -> u64 {
        let size = self.cfg.size as i32;
        let s = crate::world::chunk::CHUNK_SIZE as i32;
        let (u0, v0) = (player_col.x - size / 2, player_col.y - size / 2);
        world.surface_stamp(
            self.face,
            u0.div_euclid(s)..=(u0 + size - 1).div_euclid(s),
            v0.div_euclid(s)..=(v0 + size - 1).div_euclid(s),
        )
    }

    fn rebuild_full(&mut self, world: &World, player_col: IVec2) {
        let sz = self.cfg.size as usize;
        self.paint_rect(world, player_col, 0, 0, sz, sz);
        self.shade_rect(0, 0, sz, sz);
    }

    fn rebuild_shift(&mut self, world: &World, player_col: IVec2, dx: i32, dz: i32) {
        let sz = self.cfg.size as usize;
        shift(&mut self.top_y, sz, 1, dx, dz);
        shift(&mut self.rgba, sz, 4, dx, dz);

        if dx > 0 {
            self.paint_rect(world, player_col, sz - dx as usize, 0, sz, sz);
        } else if dx < 0 {
            self.paint_rect(world, player_col, 0, 0, (-dx) as usize, sz);
        }
        if dz > 0 {
            self.paint_rect(world, player_col, 0, sz - dz as usize, sz, sz);
        } else if dz < 0 {
            self.paint_rect(world, player_col, 0, 0, sz, (-dz) as usize);
        }

        // Shade the L without visiting the corner twice (in-place factor).
        let (v_lo, v_hi) = kept_range(sz, dz);
        if dx > 0 {
            self.shade_rect(sz - dx as usize, v_lo, sz, v_hi);
        } else if dx < 0 {
            self.shade_rect(0, v_lo, (-dx) as usize, v_hi);
        }
        if dz > 0 {
            self.shade_rect(0, sz - dz as usize, sz, sz);
        } else if dz < 0 {
            self.shade_rect(0, 0, sz, (-dz) as usize);
        }

        // Kept west/north edges whose neighbour set changed: restore unshaded colour then re-shade.
        let (u_lo, u_hi) = kept_range(sz, dx);
        let (v_lo, v_hi) = kept_range(sz, dz);
        if dx != 0 {
            let u = if dx > 0 { 0 } else { (-dx) as usize };
            self.restore_unshaded(world, player_col, u, v_lo, u + 1, v_hi);
            self.shade_rect(u, v_lo, u + 1, v_hi);
        }
        if dz != 0 {
            let v = if dz > 0 { 0 } else { (-dz) as usize };
            self.restore_unshaded(world, player_col, u_lo, v, u_hi, v + 1);
            self.shade_rect(u_lo, v, u_hi, v + 1);
        }
    }

    /// Paint texels `[u0, u1) × [v0, v1)` from loaded columns. Writes void only
    /// for columns the scan does not overwrite — no full-buffer memset.
    fn paint_rect(
        &mut self,
        world: &World,
        player_col: IVec2,
        u0: usize,
        v0: usize,
        u1: usize,
        v1: usize,
    ) {
        if u0 >= u1 || v0 >= v1 {
            return;
        }
        let size = self.cfg.size as i32;
        let sz = size as usize;
        let origin_u = player_col.x - size / 2;
        let origin_v = player_col.y - size / 2;
        let u0w = origin_u + u0 as i32;
        let v0w = origin_v + v0 as i32;
        let u1w = origin_u + u1 as i32 - 1;
        let v1w = origin_v + v1 as i32 - 1;
        let s = crate::world::chunk::CHUNK_SIZE as i32;
        let void = self.cfg.void;
        let mut tops = [[None; crate::world::chunk::CHUNK_SIZE]; crate::world::chunk::CHUNK_SIZE];

        for cu in u0w.div_euclid(s)..=u1w.div_euclid(s) {
            for cv in v0w.div_euclid(s)..=v1w.div_euclid(s) {
                let us0 = u0w.max(cu * s);
                let us1 = u1w.min((cu + 1) * s - 1);
                let vs0 = v0w.max(cv * s);
                let vs1 = v1w.min((cv + 1) * s - 1);
                let lus = us0.rem_euclid(s) as usize..us1.rem_euclid(s) as usize + 1;
                let lvs = vs0.rem_euclid(s) as usize..vs1.rem_euclid(s) as usize + 1;
                world.top_solids_on_face(self.face, (cu, cv), lus, lvs, &mut tops);
                for u in us0..=us1 {
                    let lu = u.rem_euclid(s) as usize;
                    let tu = (u - origin_u) as usize;
                    for v in vs0..=vs1 {
                        let lv = v.rem_euclid(s) as usize;
                        let idx = (v - origin_v) as usize * sz + tu;
                        let (ty, color) = tops[lv][lu].unwrap_or((i32::MIN, void));
                        self.top_y[idx] = ty;
                        self.rgba[idx * 4..idx * 4 + 4].copy_from_slice(&[color.r, color.g, color.b, color.a]);
                    }
                }
            }
        }
    }

    fn restore_unshaded(
        &mut self,
        world: &World,
        player_col: IVec2,
        u0: usize,
        v0: usize,
        u1: usize,
        v1: usize,
    ) {
        let size = self.cfg.size as i32;
        let sz = size as usize;
        let origin_u = player_col.x - size / 2;
        let origin_v = player_col.y - size / 2;
        let frame = FaceFrame::new(self.face);
        let void = self.cfg.void;
        for v in v0..v1 {
            for u in u0..u1 {
                let idx = v * sz + u;
                let ty = self.top_y[idx];
                let color = if ty == i32::MIN {
                    void
                } else {
                    let (x, y, z) = frame.cell_to_world((origin_u + u as i32, ty, origin_v + v as i32));
                    world.registry().color(world.block_at(x, y, z))
                };
                self.rgba[idx * 4..idx * 4 + 4]
                    .copy_from_slice(&[color.r, color.g, color.b, color.a]);
            }
        }
    }

    fn shade_rect(&mut self, u0: usize, v0: usize, u1: usize, v1: usize) {
        let sz = self.cfg.size as usize;
        for v in v0..v1 {
            for u in u0..u1 {
                shade_texel(&mut self.rgba, &self.top_y, sz, u, v);
            }
        }
    }

    /// Draw the map, border, and player marker. Between raster refreshes the
    /// marker tracks the player's offset from the cached center instead of
    /// falsely remaining centered over stale terrain.
    /// Pixels the map takes from the right screen edge, its margin included.
    pub fn reserved_width(&self) -> i32 {
        i32::from(self.cfg.screen_px) + self.cfg.margin.0
    }

    pub fn draw(&self, f: &mut Frame, screen: (i32, i32), sample: MapSample) {
        let player_col = sample.col;
        let half = self.cfg.screen_px as f32 / 2.0;
        let cx = screen.0 as f32 - self.cfg.margin.0 as f32 - half;
        let cy = self.cfg.margin.1 as f32 + half;
        let rotation = map_rotation(self.cfg.orient, sample.heading);
        f.draw_minimap([cx, cy], half, rotation, Color::WHITE);

        let edge = half as i32;
        let (left, top) = (cx as i32 - edge, cy as i32 - edge);
        let (right, bottom) = (cx as i32 + edge, cy as i32 + edge);
        let border = Color::new(235, 238, 244, 210);
        f.draw_line(left, top, right, top, border);
        f.draw_line(right, top, right, bottom, border);
        f.draw_line(right, bottom, left, bottom, border);
        f.draw_line(left, bottom, left, top, border);

        let center = self.center.unwrap_or(player_col);
        let texel_scale = self.cfg.screen_px as f32 / self.cfg.size as f32;
        let map_offset = Vec2::new(
            (player_col.x - center.x) as f32 * texel_scale,
            (player_col.y - center.y) as f32 * texel_scale,
        );
        let (sin, cos) = rotation.sin_cos();
        let marker = Vec2::new(
            cx + map_offset.x * cos - map_offset.y * sin,
            cy + map_offset.x * sin + map_offset.y * cos,
        );
        let marker_angle = match self.cfg.orient {
            Rotation::NorthUp => sample.heading,
            Rotation::Heading => -std::f32::consts::FRAC_PI_2,
        };
        draw_player_marker(f, marker, marker_angle);
    }
}

fn kept_range(sz: usize, delta: i32) -> (usize, usize) {
    if delta > 0 {
        (0, sz - delta as usize)
    } else if delta < 0 {
        ((-delta) as usize, sz)
    } else {
        (0, sz)
    }
}

/// Move a `sz × sz` raster of `per` values a texel by `(dx, dz)`: texel `(u, v)` takes
/// `(u + dx, v + dz)` wherever that lies inside, one row copy at a time; the rest keep their
/// old values for the caller to repaint.
fn shift<T: Copy>(buf: &mut [T], sz: usize, per: usize, dx: i32, dz: i32) {
    debug_assert!(dx.unsigned_abs() < sz as u32 && dz.unsigned_abs() < sz as u32);
    let len = (sz - dx.unsigned_abs() as usize) * per;
    let (dst_u, src_u) = if dx < 0 { (dx.unsigned_abs() as usize, 0) } else { (0, dx as usize) };
    let mut copy = |v: usize| {
        let sv = (v as i32 + dz) as usize;
        let src = (sv * sz + src_u) * per;
        buf.copy_within(src..src + len, (v * sz + dst_u) * per);
    };
    // Read each source row before it is overwritten.
    if dz > 0 {
        (0..sz - dz as usize).for_each(&mut copy);
    } else {
        (dz.unsigned_abs() as usize..sz).rev().for_each(&mut copy);
    }
}

fn shade_texel(rgba: &mut [u8], top_y: &[i32], sz: usize, u: usize, v: usize) {
    let idx = v * sz + u;
    let h = top_y[idx];
    if h == i32::MIN {
        return;
    }
    let nx = if u > 0 { top_y[idx - 1] } else { i32::MIN };
    let nz = if v > 0 { top_y[idx - sz] } else { i32::MIN };
    let neighbour = nx.max(nz);
    if neighbour == i32::MIN {
        return;
    }
    let factor = match h.cmp(&neighbour) {
        std::cmp::Ordering::Greater => 1.15,
        std::cmp::Ordering::Less => 0.85,
        std::cmp::Ordering::Equal => return,
    };
    let px = &mut rgba[idx * 4..idx * 4 + 3];
    for c in px {
        *c = (*c as f32 * factor).round().clamp(0.0, 255.0) as u8;
    }
}

/// In heading-up mode yaw zero points along world +X (texture-right), so the
/// map needs an additional quarter-turn to place that direction at screen-up.
fn map_rotation(orientation: Rotation, yaw: f32) -> f32 {
    match orientation {
        Rotation::NorthUp => 0.0,
        Rotation::Heading => -yaw - std::f32::consts::FRAC_PI_2,
    }
}

fn draw_player_marker(f: &mut Frame, center: Vec2, angle: f32) {
    let dir = Vec2::new(angle.cos(), angle.sin());
    let side = Vec2::new(-dir.y, dir.x);
    let tip = center + dir * 10.0;
    let tail = center - dir * 7.0;
    let left = tail + side * 5.0;
    let right = tail - side * 5.0;
    let line = |f: &mut Frame, a: Vec2, b: Vec2| {
        f.draw_line(
            a.x.round() as i32,
            a.y.round() as i32,
            b.x.round() as i32,
            b.y.round() as i32,
            Color::WHITE,
        );
    };
    line(f, tip, left);
    line(f, left, right);
    line(f, right, tip);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The refresh gate is (never-built OR throttle-elapsed OR moved ≥ recenter).
    #[test]
    fn refresh_gate_is_the_dual_or_of_throttle_and_recenter() {
        let mut map = Minimap::new(MinimapConfig::DEFAULT);
        let recenter = MinimapConfig::DEFAULT.recenter_after as i32;

        // Never built: always due, regardless of the throttle.
        assert!(map.due(Face::PosY, IVec2::new(0, 0), false), "never-built is always due");

        // Pretend a rebuild happened centered at the origin.
        map.center = Some(IVec2::new(0, 0));

        // Built, throttle not elapsed, still close: skip.
        assert!(!map.due(Face::PosY, IVec2::new(recenter - 1, 0), false), "recent + close ⇒ skip");
        // Built, throttle elapsed, still close: the interval half fires.
        assert!(map.due(Face::PosY, IVec2::new(recenter - 1, 0), true), "throttle elapsed ⇒ due");
        // Built, throttle not elapsed, moved past recenter: the recenter half fires.
        assert!(map.due(Face::PosY, IVec2::new(recenter, 0), false), "moved ≥ recenter ⇒ due");
        // Distance is the Chebyshev max of the two axes.
        assert!(map.due(Face::PosY, IVec2::new(0, recenter), false), "recenter checks either axis");
        // A different face is a different plane, even on the same column index.
        assert!(map.due(Face::PosX, IVec2::new(0, 0), false), "face change ⇒ due");
    }

    #[test]
    fn heading_up_rotates_world_forward_to_screen_up() {
        for yaw in [-2.0, 0.0, 1.25] {
            let rotation = map_rotation(Rotation::Heading, yaw);
            let screen_angle = yaw + rotation;
            assert!((screen_angle + std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        }
        assert_eq!(map_rotation(Rotation::NorthUp, 2.0), 0.0);
    }

    #[test]
    fn shift_strip_matches_full_rebuild() {
        let world = crate::world::World::new(73);
        let offsets = [
            IVec2::new(16, 0),
            IVec2::new(0, 16),
            IVec2::new(16, 8),
            IVec2::new(-16, -8),
            IVec2::new(-20, 24),
        ];
        for delta in offsets {
            let origin = IVec2::new(0, 0);
            let dest = IVec2::new(origin.x + delta.x, origin.y + delta.y);

            let mut shifted = Minimap::new(MinimapConfig::DEFAULT);
            assert_eq!(shifted.rebuild(&world, Face::PosY, origin, true), Some(true));
            let void = MinimapConfig::DEFAULT.void;
            let mid = (128 * 256 + 128) * 4;
            assert_ne!(
                &shifted.rgba[mid..mid + 4],
                &[void.r, void.g, void.b, void.a],
                "PosY origin still paints loaded ground"
            );
            assert_eq!(shifted.rebuild(&world, Face::PosY, dest, false), Some(true));

            let mut full = Minimap::new(MinimapConfig::DEFAULT);
            assert_eq!(full.rebuild(&world, Face::PosY, dest, true), Some(true));

            assert_eq!(
                shifted.rgba, full.rgba,
                "rgba mismatch for delta {delta:?}"
            );
            assert_eq!(
                shifted.top_y, full.top_y,
                "height mismatch for delta {delta:?}"
            );
        }
    }

    /// An interval refresh under a still player over an unchanged surface paints nothing; an edit,
    /// a newly loaded chunk or a step repaints.
    #[test]
    fn an_idle_interval_refresh_is_skipped() {
        use crate::world::chunk::Chunk;
        let mut world = crate::world::World::new(73);
        let origin = IVec2::new(0, 0);
        let mut map = Minimap::new(MinimapConfig::DEFAULT);
        assert_eq!(map.rebuild(&world, Face::PosY, origin, true), Some(true));
        assert_eq!(map.rebuild(&world, Face::PosY, origin, false), None, "not due");
        assert_eq!(map.rebuild(&world, Face::PosY, origin, true), Some(false), "idle");

        let stone = world.registry().id_by_label("rock").unwrap();
        world.set_block(3, 60, 3, stone);
        assert_eq!(map.rebuild(&world, Face::PosY, origin, true), Some(true), "an edit repaints");
        assert_eq!(map.rebuild(&world, Face::PosY, origin, true), Some(false));

        let coord = crate::coord::ChunkCoord::new(-8, 9, 7);
        world.store_column_chunk(Face::PosY, coord, Chunk::from_uniform(-8, 9, 7, stone));
        assert_eq!(map.rebuild(&world, Face::PosY, origin, true), Some(true), "a loaded chunk repaints");
        let mut full = Minimap::new(MinimapConfig::DEFAULT);
        assert_eq!(full.rebuild(&world, Face::PosY, origin, true), Some(true));
        assert!(map.rgba == full.rgba && map.top_y == full.top_y);

        assert_eq!(map.rebuild(&world, Face::PosY, IVec2::new(1, 0), true), Some(true), "a step repaints");
    }

    /// Cost probe: one full rebuild against the idle interval check that replaces it.
    #[test]
    #[ignore]
    fn minimap_rebuild_cost() {
        use std::time::Instant;
        let world = crate::world::World::new(73);
        let origin = IVec2::new(0, 0);
        let mut map = Minimap::new(MinimapConfig::DEFAULT);
        const N: u32 = 50;
        let t = Instant::now();
        for _ in 0..N {
            map.center = None;
            assert_eq!(map.rebuild(&world, Face::PosY, origin, true), Some(true));
        }
        let full = t.elapsed().as_secs_f64() * 1e6 / f64::from(N);
        let t = Instant::now();
        for _ in 0..N {
            assert_eq!(map.rebuild(&world, Face::PosY, origin, true), Some(false), "the idle rebuild is skipped");
        }
        let idle = t.elapsed().as_secs_f64() * 1e6 / f64::from(N);
        println!("minimap_rebuild_cost ({N} iters): full rebuild {full:.1} us, idle check {idle:.1} us");
    }

    /// A player standing on the +X face rasters that face's (u, v), and the height
    /// stored is the face altitude, not world Y.
    #[test]
    fn plus_x_face_rasters_the_face_plane() {
        use crate::camera::Orientation;
        use crate::world::chunk::Chunk;

        let mut world = crate::world::World::with_config_lazy(1, crate::render_config::RenderConfig::default());
        let stone = world.registry().id_by_label("rock").unwrap();
        let coord = crate::coord::ChunkCoord::new(2, 0, 0);
        world.store_column_chunk(Face::PosX, coord, Chunk::from_uniform(2, 0, 0, stone));

        let eye = FaceFrame::new(Face::PosX).point_to_world(DVec3::new(-8.0, 50.0, 8.0));
        let mut orientation = Orientation::new(0.0, 0.0);
        orientation.snap(DVec3::X);
        let sample = MapSample::from_player(&world, eye, Face::PosX, orientation.frame, 0.0);
        assert_eq!(sample.face, Face::PosX);
        assert_eq!(sample.col, IVec2::new(-8, 8));
        assert_ne!(sample.col.x, eye.x.floor() as i32, "the plane is not world XZ");

        let mut map = Minimap::new(MinimapConfig::DEFAULT);
        assert_eq!(map.rebuild(&world, sample.face, sample.col, true), Some(true));
        let idx = 128 * 256 + 128;
        let color = world.registry().color(stone);
        assert_eq!(
            &map.rgba[idx * 4..idx * 4 + 4],
            &[color.r, color.g, color.b, color.a],
            "center texel is the +X column's stone"
        );
        // Chunk (2, 0, 0) on +X has alt0 = 32; a uniform solid tops out at 47.
        // World Y of that column's top cell is 8, so a Y-up walk would store 8.
        assert_eq!(map.top_y[idx], 47);
    }

    /// On a round body the map rasters storage columns around the stream eye.
    #[test]
    fn round_world_map_uses_storage_columns() {
        use crate::space::atlas::{Atlas, Patch};
        let mut world = crate::world::World::with_config_lazy(3, crate::render_config::RenderConfig::default());
        let centre = DVec3::new(2.0e7, 3.0e7, -1.0e7);
        let r = 3_000i64;
        let atlas = std::sync::Arc::new(Atlas::new(centre, r, r + 64, false, crate::space::atlas::STORAGE_X0));
        world.set_atlases(vec![atlas.clone()]);
        let top = Patch::Shell { band: 0, face: Face::PosY };
        let b = atlas.bands[0];
        let (k, mid) = (r - b.r_lo - 1, b.n / 2);
        let stone = world.registry().id_by_label("rock").unwrap();
        let eye = atlas.embed(top, DVec3::new(mid as f64 + 0.5, k as f64 + 3.0, mid as f64 + 0.5));
        world.ensure_around(eye);
        let s = atlas.storage(top, [mid, k, mid]);
        world.set_block(s[0] as i32, s[1] as i32, s[2] as i32, stone);

        let sample = MapSample::from_player(&world, eye, Face::PosX, DQuat::IDENTITY, 0.0);
        assert_eq!(sample.face, Face::PosY, "chart columns run along storage +Y");
        assert_eq!(sample.col, IVec2::new(s[0] as i32, s[2] as i32));

        let mut map = Minimap::new(MinimapConfig::DEFAULT);
        assert_eq!(map.rebuild(&world, sample.face, sample.col, true), Some(true));
        let idx = 128 * 256 + 128;
        let color = world.registry().color(stone);
        assert_eq!(&map.rgba[idx * 4..idx * 4 + 4], &[color.r, color.g, color.b, color.a]);
        assert_eq!(map.top_y[idx], s[1] as i32);
    }
}
