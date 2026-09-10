//! Geometry helpers shared across the game.
//!
//! Positions are `f64` ([`DVec3`]) throughout the game logic: at |pos| ~100k
//! an `f32` ULP is already 0.0156 blocks, which rounds high-fps movement
//! deltas to zero and makes collision epsilons vanish. `f64` keeps sub-
//! millimetre precision out to the [`WORLD_BORDER`].
use voxel_engine::DVec3;

/// How far the world extends from the origin on every axis, in blocks.
/// Movement and `/tp` clamp positions to ±this, so it doubles as the far
/// bound every float→block conversion can rely on.
pub const WORLD_BORDER: f64 = 1.0e9;

/// How large one voxel is, in metres. The voxel lattice itself never moves
/// (one block = one world unit — saves, worldgen, meshes, coordinates all
/// stay put); instead, everything HUMAN-scale is authored in metres and
/// multiplied by [`PER_METER`], so shrinking this makes the whole grid read
/// finer relative to the player while timings (jump arcs, walk feel) stay
/// identical — lengths and velocities scale, seconds don't.
pub const BLOCK_METERS: f64 = 0.85;

/// Metres → world units (blocks): the multiplier for every authored
/// human-scale length and velocity. Exponential response RATES (1/s) never
/// take it — they live in the time domain.
pub const PER_METER: f64 = 1.0 / BLOCK_METERS;

/// Slack past ±[`WORLD_BORDER`] so [`block_coord`] still resolves cells of an
/// AABB whose half-extents stick past a position clamped to the border.
/// 16 covers any in-game AABB; `(1e9 + 16) / 16` chunks still fit `i32`.
const BLOCK_COORD_SLACK: f64 = 16.0;

/// Hermite smoothstep on a unit interval: `t²(3−2t)`.
#[inline]
pub fn smooth(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Clamped Hermite smoothstep from `edge0` to `edge1`.
#[inline]
pub fn smooth_between(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    smooth(t)
}

/// Linear interpolate `a` toward `b` by `t`.
#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// `f64` world coordinate → block: clamp to ±([`WORLD_BORDER`] + [`BLOCK_COORD_SLACK`]),
/// then floor. Downstream i32 math (chunk scale, ±1 neighbours, squared diffs)
/// is overflow-free only while the input stays well inside `i32`. Unbounded
/// `as i32` saturates at `i32::MAX` and wraps later. `NaN` clamps to `NaN` and
/// casts to 0 (origin), not a poisoned coordinate.
#[inline]
pub fn block_coord(v: f64) -> i32 {
    // Floor in f64 (exact for |v| <= 1e9 + 16, far below 2^53), then narrow
    // via i64 so the intermediate can provably never truncate.
    v.clamp(-(WORLD_BORDER + BLOCK_COORD_SLACK), WORLD_BORDER + BLOCK_COORD_SLACK).floor() as i64
        as i32
}

/// Exclusive-upper partner of [`block_coord`]: last cell an interval ending at
/// `v` still overlaps. A voxel is `[x, x+1)`, so an integer max touches but
/// does not overlap the next cell — `ceil(v) - 1`, matching [`Aabb::intersects`].
/// `NaN` → cell 0 like [`block_coord`].
#[inline]
pub fn block_coord_end(v: f64) -> i32 {
    (v.clamp(-(WORLD_BORDER + BLOCK_COORD_SLACK), WORLD_BORDER + BLOCK_COORD_SLACK).ceil() - 1.0)
        as i64 as i32
}

/// An axis-aligned bounding box defined by a centre point and half-extents.
#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub center: DVec3,
    pub half: DVec3,
}

impl Aabb {
    pub fn new(center: DVec3, half: DVec3) -> Self {
        Self { center, half }
    }

    /// The lower corner (centre minus half-extents).
    pub fn min(&self) -> DVec3 {
        self.center - self.half
    }

    /// The upper corner (centre plus half-extents).
    pub fn max(&self) -> DVec3 {
        self.center + self.half
    }

    /// Whether this box overlaps another (strictly — merely touching faces do
    /// not count, so a block placed flush against the player is fine).
    pub fn intersects(&self, other: &Aabb) -> bool {
        let d = self.center - other.center;
        d.x.abs() < self.half.x + other.half.x
            && d.y.abs() < self.half.y + other.half.y
            && d.z.abs() < self.half.z + other.half.z
    }

    /// Every integer voxel cell this box strictly overlaps. A voxel `(x, y, z)`
    /// occupies the unit cube `[x, x+1)` on each axis, so cells run from the
    /// floor of the box minimum ([`block_coord`]) to the last cell before its
    /// maximum ([`block_coord_end`]) — a box ending exactly on an integer
    /// boundary does not visit the touching-only next voxel, matching
    /// [`Aabb::intersects`]. Both bounds clamp, so a box at the border can't
    /// overflow block math.
    pub fn voxel_cells(&self) -> impl Iterator<Item = (i32, i32, i32)> {
        let min = self.min();
        let max = self.max();
        let (x0, x1) = (block_coord(min.x), block_coord_end(max.x));
        let (y0, y1) = (block_coord(min.y), block_coord_end(max.y));
        let (z0, z1) = (block_coord(min.z), block_coord_end(max.z));

        (x0..=x1)
            .flat_map(move |x| (y0..=y1).flat_map(move |y| (z0..=z1).map(move |z| (x, y, z))))
    }
}

/// Something that occupies an axis-aligned box in the world. Implementors get
/// uniform collision handling via [`Aabb`].
pub trait Bounded {
    fn aabb(&self) -> Aabb;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_coord_floors_and_clamps() {
        assert_eq!(block_coord(0.0), 0);
        assert_eq!(block_coord(1.9), 1);
        assert_eq!(block_coord(-0.1), -1);
        // Far coordinates stay exact in f64.
        assert_eq!(block_coord(100_000_000.75), 100_000_000);
        assert_eq!(block_coord(-100_000_000.25), -100_000_001);
        // An AABB corner just past the border (player clamped AT the border,
        // half-extent hanging over) must resolve its true cell, not collapse
        // onto the border column — the negative-border interpenetration bug.
        assert_eq!(block_coord(-1.0e9 - 0.3), -1_000_000_001);
        assert_eq!(block_coord(1.0e9 + 0.3), 1_000_000_000);
        // Far past the border + slack: clamped, never overflowing i32 math
        // downstream.
        assert_eq!(block_coord(WORLD_BORDER * 3.0), 1_000_000_016);
        assert_eq!(block_coord(f64::INFINITY), 1_000_000_016);
        assert_eq!(block_coord(f64::NEG_INFINITY), -1_000_000_016);
        assert_eq!(block_coord(f64::NAN), 0);
    }

    #[test]
    fn exact_face_contact_excludes_the_next_voxel() {
        // A box spanning exactly [0, 1] on each axis overlaps only voxel 0:
        // voxel 1 begins at the non-overlapping boundary.
        let unit = Aabb::new(DVec3::new(0.5, 0.5, 0.5), DVec3::new(0.5, 0.5, 0.5));
        assert_eq!(unit.voxel_cells().collect::<Vec<_>>(), vec![(0, 0, 0)]);

        // The same at a negative integer boundary: [-1, 0] is voxel -1 only.
        let neg = Aabb::new(DVec3::new(-0.5, -0.5, -0.5), DVec3::new(0.5, 0.5, 0.5));
        assert_eq!(neg.voxel_cells().collect::<Vec<_>>(), vec![(-1, -1, -1)]);

        // Any real protrusion past the boundary includes the next voxel again.
        let over = Aabb::new(DVec3::new(0.5, 0.5, 0.5), DVec3::new(0.501, 0.5, 0.5));
        let xs: Vec<i32> = over.voxel_cells().map(|c| c.0).collect();
        assert!(xs.contains(&-1) && xs.contains(&0) && xs.contains(&1));

        assert_eq!(block_coord_end(1.0), 0);
        assert_eq!(block_coord_end(-1.0), -2);
        assert_eq!(block_coord_end(1.5), 1);
        assert_eq!(block_coord_end(f64::NAN), 0);
        assert_eq!(block_coord_end(f64::INFINITY), 1_000_000_015);
    }

    #[test]
    fn voxel_cells_are_exact_at_far_coordinates() {
        // A player-sized box at x = 1e8: f32 could not even represent the
        // corners distinctly; the f64 path must return exactly the four
        // straddled columns' cells.
        let x = 1.0e8;
        let aabb = Aabb::new(DVec3::new(x + 0.5, 40.9, 0.5), DVec3::new(0.3, 0.9, 0.3));
        let cells: Vec<_> = aabb.voxel_cells().collect();
        let xs: Vec<i32> = {
            let mut v: Vec<i32> = cells.iter().map(|c| c.0).collect();
            v.dedup();
            v
        };
        assert_eq!(xs, vec![100_000_000], "0.2..0.8 stays inside one column");
        let ys: Vec<i32> = {
            let mut v: Vec<i32> = cells.iter().map(|c| c.1).collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        assert_eq!(ys, vec![40, 41], "feet at 40.0 .. head at 41.8");

        // Straddling a column boundary at the same distance is still exact.
        let aabb = Aabb::new(DVec3::new(x, 40.5, 0.5), DVec3::new(0.3, 0.3, 0.3));
        let mut xs: Vec<i32> = aabb.voxel_cells().map(|c| c.0).collect();
        xs.sort_unstable();
        xs.dedup();
        assert_eq!(xs, vec![99_999_999, 100_000_000]);
    }
}
