//! interact.rs turns where the player looks into which block they act on: a voxel
//! ray-march from the eye along the view direction, returning the first solid block
//! within reach. Breaking and (later) placing are built on this one query.
use raylib::prelude::*;

use crate::world::World;

/// A block the aim ray struck.
pub struct RayHit {
    /// The solid block that was hit.
    pub block: (i32, i32, i32),
    /// The empty cell just before it along the ray — where a placed block would go.
    pub before: (i32, i32, i32),
}

/// March a ray from `origin` along `dir` up to `reach` world units and return the
/// first solid block, using Amanatides–Woo grid traversal (each iteration crosses
/// exactly one voxel face, so nothing is skipped or double-visited).
pub fn raycast(world: &World, origin: Vector3, dir: Vector3, reach: f32) -> Option<RayHit> {
    let len = dir.length();
    if len == 0.0 {
        return None;
    }
    let dir = dir.scale(1.0 / len);

    let (mut x, mut y, mut z) = (
        origin.x.floor() as i32,
        origin.y.floor() as i32,
        origin.z.floor() as i32,
    );
    if world.is_solid(x, y, z) {
        return Some(RayHit {
            block: (x, y, z),
            before: (x, y, z),
        });
    }

    let step = |d: f32| if d > 0.0 { 1 } else if d < 0.0 { -1 } else { 0 };
    let (step_x, step_y, step_z) = (step(dir.x), step(dir.y), step(dir.z));

    // Distance (in ray length) to the first voxel boundary on each axis, and the
    // distance between successive boundaries. A zero component never crosses, so its
    // boundaries sit at infinity.
    let boundary = |o: f32, cell: i32, d: f32| -> f32 {
        if d == 0.0 {
            return f32::INFINITY;
        }
        let next = if d > 0.0 {
            (cell as f32 + 1.0) - o
        } else {
            o - cell as f32
        };
        next / d.abs()
    };
    let (mut t_max_x, mut t_max_y, mut t_max_z) = (
        boundary(origin.x, x, dir.x),
        boundary(origin.y, y, dir.y),
        boundary(origin.z, z, dir.z),
    );
    let t_delta = |d: f32| if d == 0.0 { f32::INFINITY } else { (1.0 / d).abs() };
    let (t_delta_x, t_delta_y, t_delta_z) = (t_delta(dir.x), t_delta(dir.y), t_delta(dir.z));

    let mut t = 0.0;
    while t <= reach {
        let before = (x, y, z);
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
        if world.is_solid(x, y, z) {
            return Some(RayHit {
                block: (x, y, z),
                before,
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn looking_down_hits_the_ground() {
        let world = World::generate();
        // Start high above a known column and look straight down.
        let origin = Vector3::new(8.5, 40.0, 8.5);
        let hit = raycast(&world, origin, Vector3::new(0.0, -1.0, 0.0), 60.0)
            .expect("a downward ray should hit the terrain");
        assert!(world.is_solid(hit.block.0, hit.block.1, hit.block.2));
        // The cell just above the hit block is the empty one the ray last passed.
        assert_eq!(hit.before.1, hit.block.1 + 1);
    }

    #[test]
    fn ray_into_open_sky_misses() {
        let world = World::generate();
        let origin = Vector3::new(8.5, 40.0, 8.5);
        assert!(raycast(&world, origin, Vector3::new(0.0, 1.0, 0.0), 20.0).is_none());
    }
}
