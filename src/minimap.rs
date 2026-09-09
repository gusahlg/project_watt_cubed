//! A Xaero-style top-down minimap: a throttled RGBA raster of the terrain
//! around the player, uploaded to a dedicated engine texture and drawn as one
//! rotatable/zoomable 2D quad in the HUD corner.
//!
//! The CPU rebuild ([`Minimap::refresh`]) scans the loaded chunks' top solid
//! blocks, folds in slope shading, and hands the pixels to the engine on a
//! throttle. The per-frame [`Minimap::draw`] is just one textured quad, so it
//! costs two triangles regardless of the raster resolution.

use std::time::Duration;

use voxel_engine::{Color, Engine, Frame, IVec2, Vec2};

use crate::world::World;

/// How the map is oriented relative to the world.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orientation {
    /// North (−Z? +Z — the world's fixed axis) always points up; the map never
    /// rotates and the player marker spins instead.
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
    pub orient: Orientation,
}

impl MinimapConfig {
    pub const DEFAULT: Self = Self {
        size: 256,
        screen_px: 160,
        margin: (12, 12),
        refresh_every: Duration::from_millis(500),
        recenter_after: 16,
        void: Color::rgb(18, 18, 24),
        orient: Orientation::NorthUp,
    };
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
}

impl Minimap {
    pub fn new(cfg: MinimapConfig) -> Self {
        assert_eq!(cfg.size, 256, "minimap size must match engine MINIMAP_SIZE");
        let texels = cfg.size as usize * cfg.size as usize;
        Self { cfg, rgba: vec![0u8; texels * 4], top_y: vec![i32::MIN; texels], center: None }
    }

    /// Toggle between north-up (fixed map, spinning marker) and heading-up
    /// (rotating map, marker locked pointing up).
    pub fn toggle_orientation(&mut self) {
        self.cfg.orient = match self.cfg.orient {
            Orientation::NorthUp => Orientation::Heading,
            Orientation::Heading => Orientation::NorthUp,
        };
    }

    /// Whether a rebuild is due: never built, OR the throttle elapsed, OR the
    /// player moved past the recenter distance.
    pub fn due(&self, player_col: IVec2, interval_elapsed: bool) -> bool {
        match self.center {
            None => true,
            Some(c) => {
                let moved = (player_col.x - c.x).abs().max((player_col.y - c.y).abs());
                interval_elapsed || moved >= self.cfg.recenter_after as i32
            }
        }
    }

    /// Throttled + recenter-gated rescan: when [`Self::due`], rebuilds `rgba`
    /// from the world's top-solid columns (colour × slope-shade) and uploads it
    /// via [`Engine::update_minimap`]. `interval_elapsed` is the scheduler's
    /// throttle decision. Returns `true` when it rebuilt, so the caller resets
    /// the scheduler's interval gate on the attempt.
    pub fn refresh(
        &mut self,
        eng: &mut Engine,
        world: &World,
        player_col: IVec2,
        interval_elapsed: bool,
    ) -> bool {
        if !self.rebuild(world, player_col, interval_elapsed) {
            return false;
        }
        eng.update_minimap(&self.rgba);
        true
    }

    /// CPU half of [`Self::refresh`]: full rebuild on the interval / first
    /// build / `d ≥ size`, otherwise shift the raster and repaint exposed strips.
    fn rebuild(&mut self, world: &World, player_col: IVec2, interval_elapsed: bool) -> bool {
        if !self.due(player_col, interval_elapsed) {
            return false;
        }
        let size = self.cfg.size as i32;
        match self.center {
            Some(prev) if !interval_elapsed => {
                let dx = player_col.x - prev.x;
                let dz = player_col.y - prev.y;
                let d = dx.abs().max(dz.abs());
                if d > 0 && d < size {
                    self.rebuild_shift(world, player_col, dx, dz);
                } else {
                    self.rebuild_full(world, player_col);
                }
            }
            _ => self.rebuild_full(world, player_col),
        }
        self.center = Some(player_col);
        true
    }

    fn rebuild_full(&mut self, world: &World, player_col: IVec2) {
        let sz = self.cfg.size as usize;
        self.paint_rect(world, player_col, 0, 0, sz, sz);
        self.shade_rect(0, 0, sz, sz);
    }

    fn rebuild_shift(&mut self, world: &World, player_col: IVec2, dx: i32, dz: i32) {
        let sz = self.cfg.size as usize;
        shift_heights(&mut self.top_y, sz, dx, dz);
        shift_rgba(&mut self.rgba, sz, dx, dz);

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
        let origin_x = player_col.x - size / 2;
        let origin_z = player_col.y - size / 2;
        let x0 = origin_x + u0 as i32;
        let z0 = origin_z + v0 as i32;
        let x1 = origin_x + u1 as i32 - 1;
        let z1 = origin_z + v1 as i32 - 1;
        let s = crate::world::chunk::CHUNK_SIZE as i32;
        let void = self.cfg.void;

        for cx in x0.div_euclid(s)..=x1.div_euclid(s) {
            for cz in z0.div_euclid(s)..=z1.div_euclid(s) {
                let xs0 = x0.max(cx * s);
                let xs1 = x1.min((cx + 1) * s - 1);
                let zs0 = z0.max(cz * s);
                let zs1 = z1.min((cz + 1) * s - 1);
                let ys = world.column_chunks(cx, cz);
                for x in xs0..=xs1 {
                    let lx = x.rem_euclid(s) as usize;
                    let u = (x - origin_x) as usize;
                    for z in zs0..=zs1 {
                        let lz = z.rem_euclid(s) as usize;
                        let v = (z - origin_z) as usize;
                        let idx = v * sz + u;
                        match world.top_solid_in_column(cx, cz, ys, lx, lz) {
                            Some((ty, color)) => {
                                self.top_y[idx] = ty;
                                self.rgba[idx * 4..idx * 4 + 4]
                                    .copy_from_slice(&[color.r, color.g, color.b, color.a]);
                            }
                            None => {
                                self.top_y[idx] = i32::MIN;
                                self.rgba[idx * 4..idx * 4 + 4]
                                    .copy_from_slice(&[void.r, void.g, void.b, void.a]);
                            }
                        }
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
        let origin_x = player_col.x - size / 2;
        let origin_z = player_col.y - size / 2;
        let void = self.cfg.void;
        for v in v0..v1 {
            for u in u0..u1 {
                let idx = v * sz + u;
                let ty = self.top_y[idx];
                let color = if ty == i32::MIN {
                    void
                } else {
                    world
                        .registry()
                        .color(world.block_at(origin_x + u as i32, ty, origin_z + v as i32))
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
    pub fn draw(&self, f: &mut Frame, screen: (i32, i32), player_col: IVec2, yaw: f32) {
        let half = self.cfg.screen_px as f32 / 2.0;
        let cx = screen.0 as f32 - self.cfg.margin.0 as f32 - half;
        let cy = self.cfg.margin.1 as f32 + half;
        let rotation = map_rotation(self.cfg.orient, yaw);
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
            Orientation::NorthUp => yaw,
            Orientation::Heading => -std::f32::consts::FRAC_PI_2,
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

fn shift_heights(buf: &mut [i32], sz: usize, dx: i32, dz: i32) {
    shift2d(sz, dx, dz, |u, v, su, sv| {
        buf[v * sz + u] = buf[sv * sz + su];
    });
}

fn shift_rgba(buf: &mut [u8], sz: usize, dx: i32, dz: i32) {
    shift2d(sz, dx, dz, |u, v, su, sv| {
        let dst = (v * sz + u) * 4;
        let src = (sv * sz + su) * 4;
        buf.copy_within(src..src + 4, dst);
    });
}

fn shift2d(sz: usize, dx: i32, dz: i32, mut copy: impl FnMut(usize, usize, usize, usize)) {
    if dz < 0 {
        for v in (0..sz).rev() {
            shift_row(sz, v, dx, dz, &mut copy);
        }
    } else {
        for v in 0..sz {
            shift_row(sz, v, dx, dz, &mut copy);
        }
    }
}

fn shift_row(
    sz: usize,
    v: usize,
    dx: i32,
    dz: i32,
    copy: &mut impl FnMut(usize, usize, usize, usize),
) {
    if dx < 0 {
        for u in (0..sz).rev() {
            try_shift(sz, u, v, dx, dz, copy);
        }
    } else {
        for u in 0..sz {
            try_shift(sz, u, v, dx, dz, copy);
        }
    }
}

fn try_shift(
    sz: usize,
    u: usize,
    v: usize,
    dx: i32,
    dz: i32,
    copy: &mut impl FnMut(usize, usize, usize, usize),
) {
    let su = u as i32 + dx;
    let sv = v as i32 + dz;
    if su >= 0 && su < sz as i32 && sv >= 0 && sv < sz as i32 {
        copy(u, v, su as usize, sv as usize);
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
fn map_rotation(orientation: Orientation, yaw: f32) -> f32 {
    match orientation {
        Orientation::NorthUp => 0.0,
        Orientation::Heading => -yaw - std::f32::consts::FRAC_PI_2,
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
        assert!(map.due(IVec2::new(0, 0), false), "never-built is always due");

        // Pretend a rebuild happened centered at the origin.
        map.center = Some(IVec2::new(0, 0));

        // Built, throttle not elapsed, still close: skip.
        assert!(!map.due(IVec2::new(recenter - 1, 0), false), "recent + close ⇒ skip");
        // Built, throttle elapsed, still close: the interval half fires.
        assert!(map.due(IVec2::new(recenter - 1, 0), true), "throttle elapsed ⇒ due");
        // Built, throttle not elapsed, moved past recenter: the recenter half fires.
        assert!(map.due(IVec2::new(recenter, 0), false), "moved ≥ recenter ⇒ due");
        // Distance is the Chebyshev max of the two axes.
        assert!(map.due(IVec2::new(0, recenter), false), "recenter checks either axis");
    }

    #[test]
    fn heading_up_rotates_world_forward_to_screen_up() {
        for yaw in [-2.0, 0.0, 1.25] {
            let rotation = map_rotation(Orientation::Heading, yaw);
            let screen_angle = yaw + rotation;
            assert!((screen_angle + std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        }
        assert_eq!(map_rotation(Orientation::NorthUp, 2.0), 0.0);
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
            assert!(shifted.rebuild(&world, origin, true));
            assert!(shifted.rebuild(&world, dest, false));

            let mut full = Minimap::new(MinimapConfig::DEFAULT);
            assert!(full.rebuild(&world, dest, true));

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
}
