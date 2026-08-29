//! interact.rs turns where the player looks into which block they act on: a voxel
//! ray-march from the eye along the view direction. Ordinary clearance/placement
//! rays return the first obstacle and pass through liquids; the mining variant
//! returns the first solid block so Water can be broken consistently with every
//! other material block.
//!
//! The march runs in `f64`: at far coordinates an `f32` origin can't even
//! represent which cell the eye is in (ULP > 1 block past ~2^24), while `f64`
//! boundary distances stay exact out to the world border.
use voxel_engine::DVec3;

use crate::math::block_coord;
use crate::world::World;

/// Player interaction radius: six metres expressed in world units. Breaking
/// and placement share this value so changes in the unit scale cannot make one
/// action reach farther than the other.
pub const REACH: f64 = 6.0 * crate::math::PER_METER;

/// A block the aim ray struck.
pub struct RayHit {
    /// The block selected by the ray's targeting policy.
    pub block: (i32, i32, i32),
    /// The last cell the ray passed through *before* the hit block — where a
    /// placed block would go. If the ray starts inside a block selected by the
    /// active targeting policy, this is the start cell itself.
    pub previous: (i32, i32, i32),
}

/// March a ray from `origin` along `dir` up to `reach` world units and return the
/// first obstacle using Amanatides–Woo grid traversal (each iteration crosses
/// exactly one voxel face, so nothing is skipped or double-visited). Passable
/// liquids are transparent to this general-purpose ray.
pub fn raycast(world: &World, origin: DVec3, dir: DVec3, reach: f64) -> Option<RayHit> {
    raycast_where(origin, dir, reach, |x, y, z| world.is_obstacle(x, y, z))
}

/// Target every material block, including physically passable liquids.
pub(crate) fn raycast_solid(
    world: &World,
    origin: DVec3,
    dir: DVec3,
    reach: f64,
) -> Option<RayHit> {
    raycast_where(origin, dir, reach, |x, y, z| world.is_solid(x, y, z))
}

fn raycast_where(
    origin: DVec3,
    dir: DVec3,
    reach: f64,
    is_hit: impl Fn(i32, i32, i32) -> bool,
) -> Option<RayHit> {
    let len = dir.length();
    if len == 0.0 {
        return None;
    }
    let dir = dir * (1.0 / len);

    // The start cell goes through the shared clamped conversion; every further
    // cell is one ±1 step from it, so the i32 march can't overflow either.
    let (mut x, mut y, mut z) = (
        block_coord(origin.x),
        block_coord(origin.y),
        block_coord(origin.z),
    );
    if is_hit(x, y, z) {
        return Some(RayHit {
            block: (x, y, z),
            previous: (x, y, z),
        });
    }

    let step = |d: f64| if d > 0.0 { 1 } else if d < 0.0 { -1 } else { 0 };
    let (step_x, step_y, step_z) = (step(dir.x), step(dir.y), step(dir.z));

    // Distance (in ray length) to the first voxel boundary on each axis, and the
    // distance between successive boundaries. A zero component never crosses, so its
    // boundaries sit at infinity. (`cell as f64` is exact: cells are bounded by
    // the world border, far below 2^53.)
    let boundary = |o: f64, cell: i32, d: f64| -> f64 {
        if d == 0.0 {
            return f64::INFINITY;
        }
        let next = if d > 0.0 {
            (cell as f64 + 1.0) - o
        } else {
            o - cell as f64
        };
        next / d.abs()
    };
    let (mut t_max_x, mut t_max_y, mut t_max_z) = (
        boundary(origin.x, x, dir.x),
        boundary(origin.y, y, dir.y),
        boundary(origin.z, z, dir.z),
    );
    let t_delta = |d: f64| if d == 0.0 { f64::INFINITY } else { (1.0 / d).abs() };
    let (t_delta_x, t_delta_y, t_delta_z) = (t_delta(dir.x), t_delta(dir.y), t_delta(dir.z));

    let mut t = 0.0;
    while t <= reach {
        let previous = (x, y, z);
        if t_max_x <= t_max_y && t_max_x <= t_max_z {
            x += step_x;
            t = t_max_x;
            t_max_x += t_delta_x;
        } else if t_max_y <= t_max_z {
            y += step_y;
            t = t_max_y;
            t_max_y += t_delta_y;
        } else {
            z += step_z;
            t = t_max_z;
            t_max_z += t_delta_z;
        }
        if t > reach {
            break;
        }
        if is_hit(x, y, z) {
            return Some(RayHit {
                block: (x, y, z),
                previous,
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn previous_is_the_cell_above_when_looking_down() {
        let world = World::generate();
        let origin = DVec3::new(8.5, 40.0, 8.5);
        let hit = raycast(&world, origin, DVec3::new(0.0, -1.0, 0.0), 60.0)
            .expect("a downward ray should hit the terrain");
        // Straight down: `previous` is exactly the cell above the hit block, and
        // it is empty (a placed block would fit there).
        let (bx, by, bz) = hit.block;
        assert_eq!(hit.previous, (bx, by + 1, bz));
        assert!(!world.is_obstacle(bx, by + 1, bz));
    }

    #[test]
    fn ray_into_open_sky_misses() {
        let world = World::generate();
        let origin = DVec3::new(8.5, 40.0, 8.5);
        assert!(raycast(&world, origin, DVec3::new(0.0, 1.0, 0.0), 20.0).is_none());
    }

    #[test]
    fn liquid_is_mineable_but_remains_passable() {
        let mut world = World::generate();
        let (x, y, z) = (8, 40, 8);
        world.prepare_around(DVec3::new(x as f64, y as f64, z as f64));
        let water = world.registry().id_by_name("Water").unwrap();
        world.set_block(x, y, z, water);
        world.set_block(x, y + 1, z, crate::block::AIR);
        let origin = DVec3::new(x as f64 + 0.5, y as f64 + 1.5, z as f64 + 0.5);

        let down = DVec3::new(0.0, -1.0, 0.0);
        assert_eq!(
            raycast_solid(&world, origin, down, 1.0).unwrap().block,
            (x, y, z)
        );
        assert!(raycast(&world, origin, down, 1.0).is_none());
    }

    #[test]
    fn raycast_hits_correctly_at_1e8() {
        // Far out, the ray must still land on the surface column under the eye
        // and report the empty cell above it — the f32 version couldn't even
        // resolve which column the origin was in.
        let mut world = World::generate();
        // Start above whatever the generator built here (the surface reaches
        // ~49 at this column), so the ray enters from open air rather than
        // starting inside the ground and tripping the inside-a-block guard.
        let probe = DVec3::new(1.0e8 + 8.5, 40.0, 8.5);
        world.prepare_around(probe);
        let top = world.surface_y(100_000_008, 8) as f64 + 5.0;
        let origin = DVec3::new(1.0e8 + 8.5, top, 8.5);
        let hit = raycast(&world, origin, DVec3::new(0.0, -1.0, 0.0), 60.0)
            .expect("a downward ray should hit the terrain at 1e8");
        let (bx, by, bz) = hit.block;
        assert_eq!((bx, bz), (100_000_008, 8), "hits the column under the eye");
        // The topmost *obstacle* of the column: the terrain surface. Where the column
        // is below sea level the ray passes straight through the water column above it
        // (a passable liquid) and lands on the seabed, so `previous` — the placement
        // cell — is water, not air, but still placeable (not an obstacle).
        assert!(world.is_obstacle(bx, by, bz), "hit is an obstacle");
        assert!(world.is_obstacle(bx, by - 1, bz), "and it is a real surface, not a floater");
        assert_eq!(hit.previous, (bx, by + 1, bz));
        assert!(!world.is_obstacle(bx, by + 1, bz), "placeable cell above the surface");

        // A slanted ray from the same eye still steps cell-exactly.
        let hit = raycast(&world, origin, DVec3::new(0.4, -1.0, 0.2), 60.0)
            .expect("slanted far ray hits");
        assert!(world.is_obstacle(hit.block.0, hit.block.1, hit.block.2));
        assert!(!world.is_obstacle(hit.previous.0, hit.previous.1, hit.previous.2));
    }
}
