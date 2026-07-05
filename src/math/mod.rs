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

/// The one conversion from an `f64` world coordinate to an integer block
/// coordinate: clamp to ±[`WORLD_BORDER`], then floor.
///
/// Everything that turns a position into a cell goes through here (collision
/// cell ranges, chunk lookup, the interact raycast's start cell, placement,
/// interest buckets) because i32 block math must never overflow: downstream
/// code multiplies block coords by [`CHUNK_SIZE`](crate::world::chunk::CHUNK_SIZE)
/// scale factors, offsets them by ±1 for neighbours, and squares differences —
/// all safe only while the input is bounded well inside `i32` range. A raw
/// `as i32` cast of an unbounded float would saturate at `i32::MAX` and make
/// that arithmetic wrap. `NaN` clamps to `NaN` and casts to 0 — a harmless
/// origin cell rather than a poisoned coordinate.
#[inline]
pub fn block_coord(v: f64) -> i32 {
    // Floor in f64 (exact for |v| <= 1e9, far below 2^53), then narrow via
    // i64 so the intermediate can provably never truncate.
    v.clamp(-WORLD_BORDER, WORLD_BORDER).floor() as i64 as i32
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

    /// Every integer voxel cell this box overlaps. A voxel `(x, y, z)` occupies
    /// the unit cube `[x, x+1)` on each axis, so the overlapped cells run from the
    /// floor of the box minimum to the floor of its maximum (both through
    /// [`block_coord`], so a box at the border can't overflow block math).
    pub fn voxel_cells(&self) -> impl Iterator<Item = (i32, i32, i32)> {
        let min = self.min();
        let max = self.max();
        let (x0, x1) = (block_coord(min.x), block_coord(max.x));
        let (y0, y1) = (block_coord(min.y), block_coord(max.y));
        let (z0, z1) = (block_coord(min.z), block_coord(max.z));

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
        // Past the border: clamped, never overflowing i32 math downstream.
        assert_eq!(block_coord(WORLD_BORDER * 3.0), 1_000_000_000);
        assert_eq!(block_coord(f64::INFINITY), 1_000_000_000);
        assert_eq!(block_coord(f64::NEG_INFINITY), -1_000_000_000);
        assert_eq!(block_coord(f64::NAN), 0);
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
