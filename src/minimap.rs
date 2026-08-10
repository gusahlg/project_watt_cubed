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
    /// Re-center (full rescan) once the player has moved this many blocks from
    /// the raster's current center.
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
        if !self.due(player_col, interval_elapsed) {
            return false;
        }

        let size = self.cfg.size as i32;
        let x0 = player_col.x - size / 2;
        let z0 = player_col.y - size / 2;
        let x1 = x0 + size - 1;
        let z1 = z0 + size - 1;

        let void = self.cfg.void;
        for px in self.rgba.chunks_exact_mut(4) {
            px.copy_from_slice(&[void.r, void.g, void.b, void.a]);
        }
        self.top_y.fill(i32::MIN);

        let (rgba, top_y) = (&mut self.rgba, &mut self.top_y);
        world.for_surface_columns(x0, z0, x1, z1, |bx, bz, ty, color| {
            let u = (bx - x0) as usize;
            let v = (bz - z0) as usize;
            let idx = v * size as usize + u;
            top_y[idx] = ty;
            rgba[idx * 4..idx * 4 + 4].copy_from_slice(&[color.r, color.g, color.b, color.a]);
        });

        // Slope shading: brighten uphill / darken downhill vs the north-west
        // neighbour, reading heights (never shaded rgb) so the gradient stays clean.
        let sz = size as usize;
        for v in 0..sz {
            for u in 0..sz {
                let idx = v * sz + u;
                let h = self.top_y[idx];
                if h == i32::MIN {
                    continue;
                }
                let nx = if u > 0 { self.top_y[idx - 1] } else { i32::MIN };
                let nz = if v > 0 {
                    self.top_y[idx - sz]
                } else {
                    i32::MIN
                };
                let neighbour = nx.max(nz);
                if neighbour == i32::MIN {
                    continue;
                }
                let factor = match h.cmp(&neighbour) {
                    std::cmp::Ordering::Greater => 1.15,
                    std::cmp::Ordering::Less => 0.85,
                    std::cmp::Ordering::Equal => continue,
                };
                let px = &mut self.rgba[idx * 4..idx * 4 + 3];
                for c in px {
                    *c = (*c as f32 * factor).round().clamp(0.0, 255.0) as u8;
                }
            }
        }

        eng.update_minimap(&self.rgba);
        self.center = Some(player_col);
        true
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
}
