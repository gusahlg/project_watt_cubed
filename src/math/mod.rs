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

/// Slack past ±[`WORLD_BORDER`] within which [`block_coord`] still resolves a
/// true cell instead of clamping. Player *positions* are clamped to exactly
/// ±`WORLD_BORDER` (movement and `/tp`, the only continuous writers), but the
/// AABBs built *around* a position extend up to their half-extents beyond it —
/// clamping the conversion at the border itself collapsed those outer corners
/// onto the border column (e.g. an AABB min of `-1e9 - 0.3` skipped its true
/// column `-1_000_000_001`, so the outermost cells never collided and the
/// player interpenetrated terrain at the negative border). 16 blocks covers
/// any in-game AABB by a wide margin while staying light-years inside `i32`
/// block math: `(1e9 + 16) / 16` chunks of 16 blocks fits `i32` fine.
const BLOCK_COORD_SLACK: f64 = 16.0;

/// The one conversion from an `f64` world coordinate to an integer block
/// coordinate: clamp to ±([`WORLD_BORDER`] + [`BLOCK_COORD_SLACK`]), then
/// floor. Together with movement/`/tp` clamping positions to exactly
/// ±`WORLD_BORDER`, the slack makes every AABB reachable in play — including
/// one straddling the border — resolve its true cells.
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
    // Floor in f64 (exact for |v| <= 1e9 + 16, far below 2^53), then narrow
    // via i64 so the intermediate can provably never truncate.
    v.clamp(-(WORLD_BORDER + BLOCK_COORD_SLACK), WORLD_BORDER + BLOCK_COORD_SLACK).floor() as i64
        as i32
}

/// The exclusive-upper-edge partner of [`block_coord`]: the last cell an
/// interval ending at `v` still overlaps. A voxel spans `[x, x+1)`, so a box
/// whose maximum lands exactly on an integer boundary touches — but does not
/// overlap — the next cell: `ceil(v) - 1`, not `floor(v)`. Matches the strict
/// overlap rule of [`Aabb::intersects`]. `NaN` resolves to cell 0 like
/// [`block_coord`] (subtraction happens in f64, so the NaN survives to the
/// saturating cast).
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
