//! interact.rs turns where the player looks into which block they act on: a voxel
//! ray-march from the eye along the view direction that returns the first solid
//! block. Mining, placement, camera clearance, and name-tag occlusion all share it.
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
    /// The first solid block along the ray.
    pub block: (i32, i32, i32),
    /// The last cell the ray passed through *before* the hit block — where a
    /// placed block would go. If the ray starts inside a solid block, this is
    /// the start cell itself.
    pub previous: (i32, i32, i32),
}

/// March a ray from `origin` along `dir` up to `reach` world units and return the
/// first solid block using Amanatides–Woo grid traversal (each iteration crosses
/// exactly one voxel face, so nothing is skipped or double-visited). On a round
/// world (inside an atlas) the ray runs in the storage cells of the patch under it.
pub fn raycast(world: &World, origin: DVec3, dir: DVec3, reach: f64) -> Option<RayHit> {
    let len = dir.length();
    if len == 0.0 {
        return None;
    }
    let dir = dir * (1.0 / len);
    if let Some(atlas) = world.atlas_at(origin) {
        return raycast_charted(world, atlas, origin, dir, reach);
    }
    match march(origin, dir, reach, |x, y, z| world.is_solid(x, y, z), |_, _, _| false) {
        March::Hit(hit) => Some(hit),
        March::Left { .. } | March::Miss => None,
    }
}

/// How far a charted march may run outside its patch's box before it re-enters the neighbouring
/// patch: cells one outside still read through the glue.
const CHART_SLACK: i64 = 1;

/// The ray through a round world: in the storage frame of the patch under the current point (the
/// local Jacobian carries the direction; cells there are axis aligned), re-entering the next patch
/// when it leaves a box. Hits report the cells that really hold them (glued across seams).
fn raycast_charted(world: &World, atlas: &crate::space::atlas::Atlas, origin: DVec3, dir: DVec3, reach: f64) -> Option<RayHit> {
    let (mut p, mut left) = (origin, reach);
    // A ray of a few blocks crosses at most a seam or two; the bound only guards degenerate input.
    for _ in 0..6 {
        let Some(here) = atlas.local(p) else {
            // Out of the atlas (above the relief band): physical cells for the rest.
            return match march(p, dir, left, |x, y, z| world.is_solid(x, y, z), |_, _, _| false) {
                March::Hit(hit) => Some(hit),
                _ => None,
            };
        };
        let ds = here.jacobian.inverse() * dir;
        let scale = ds.length();
        if !(scale > 0.0 && scale.is_finite()) {
            return None;
        }
        let ds = ds / scale;
        let (o, size) = atlas.storage_box(here.patch);
        let outside = |x: i32, y: i32, z: i32| {
            let c = [x as i64, y as i64, z as i64];
            (0..3).any(|a| c[a] < o[a] - CHART_SLACK || c[a] >= o[a] + size[a] + CHART_SLACK)
        };
        match march(here.storage, ds, left * scale, |x, y, z| world.is_solid(x, y, z), outside) {
            March::Hit(hit) => {
                return Some(RayHit { block: world.glued(hit.block), previous: world.glued(hit.previous) });
            }
            March::Left { t } => {
                p = atlas.embed_storage(here.patch, here.storage + ds * t);
                left -= t / scale;
                if left <= 0.0 {
                    return None;
                }
            }
            March::Miss => return None,
        }
    }
    None
}

/// How a [`march`] ended.
enum March {
    Hit(RayHit),
    /// The ray entered a cell `leave` rejected, at ray length `t`.
    Left { t: f64 },
    Miss,
}

/// Amanatides–Woo over unit cells from `origin` along unit `dir`, up to `reach`: the first cell
/// `solid` accepts, or the first cell `leave` rejects (the march's frame ends there).
fn march(
    origin: DVec3,
    dir: DVec3,
    reach: f64,
    solid: impl Fn(i32, i32, i32) -> bool,
    leave: impl Fn(i32, i32, i32) -> bool,
) -> March {
    // The start cell goes through the shared clamped conversion; every further
    // cell is one ±1 step from it, so the i32 march can't overflow either.
    let (mut x, mut y, mut z) = (
        block_coord(origin.x),
        block_coord(origin.y),
        block_coord(origin.z),
    );
    if solid(x, y, z) {
        return March::Hit(RayHit {
            block: (x, y, z),
            previous: (x, y, z),
        });
    }

    let step = |d: f64| if d > 0.0 { 1 } else if d < 0.0 { -1 } else { 0 };
    let (step_x, step_y, step_z) = (step(dir.x), step(dir.y), step(dir.z));

    // Distance (in ray length) to the first voxel boundary on each axis, and the
    // distance between successive boundaries. A zero component never crosses, so its
    // boundaries sit at infinity. (`cell as f64` is exact: cells are bounded by
    // the storage limit, far below 2^53.)
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
        if leave(x, y, z) {
            return March::Left { t };
        }
        if solid(x, y, z) {
            return March::Hit(RayHit {
                block: (x, y, z),
                previous,
            });
        }
    }
    March::Miss
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
        assert!(!world.is_solid(bx, by + 1, bz));
    }

    #[test]
    fn ray_into_open_sky_misses() {
        let world = World::generate();
        let origin = DVec3::new(8.5, 40.0, 8.5);
        assert!(raycast(&world, origin, DVec3::new(0.0, 1.0, 0.0), 20.0).is_none());
    }

    #[test]
    fn raycast_hits_correctly_far_out() {
        // Far out (5e7: an f32 step is 4 blocks there; still on the flat world's finite slab), the
        // ray must land on the surface column under the eye and report the empty cell above it —
        // the f32 version couldn't even resolve which column the origin was in.
        let mut world = World::generate();
        // Start above whatever the generator built here (the surface reaches
        // ~49 at this column), so the ray enters from open air rather than
        // starting inside the ground and tripping the inside-a-block guard.
        let probe = DVec3::new(5.0e7 + 8.5, 40.0, 8.5);
        world.ensure_around(probe);
        let top = world.surface_y(50_000_008, 8) as f64 + 5.0;
        let origin = DVec3::new(5.0e7 + 8.5, top, 8.5);
        let hit = raycast(&world, origin, DVec3::new(0.0, -1.0, 0.0), 60.0)
            .expect("a downward ray should hit the terrain far out");
        let (bx, by, bz) = hit.block;
        assert_eq!((bx, bz), (50_000_008, 8), "hits the column under the eye");
        // The topmost solid block of the column: the terrain surface, with the
        // empty placement cell directly above it.
        assert!(world.is_solid(bx, by, bz), "hit is solid");
        assert!(world.is_solid(bx, by - 1, bz), "and it is a real surface, not a floater");
        assert_eq!(hit.previous, (bx, by + 1, bz));
        assert!(!world.is_solid(bx, by + 1, bz), "placeable cell above the surface");

        // A slanted ray from the same eye still steps cell-exactly.
        let hit = raycast(&world, origin, DVec3::new(0.4, -1.0, 0.2), 60.0)
            .expect("slanted far ray hits");
        assert!(world.is_solid(hit.block.0, hit.block.1, hit.block.2));
        assert!(!world.is_solid(hit.previous.0, hit.previous.1, hit.previous.2));
    }

    #[test]
    fn breaking_a_block_yields_its_configuration() {
        let mut world = World::generate();
        world.ensure_around(DVec3::new(8.5, 20.0, 8.5));
        let (x, z) = (8, 8);
        let y = (0..96)
            .rev()
            .find(|&y| world.is_solid(x, y, z))
            .expect("a solid cell near spawn");
        let id = world.block_at(x, y, z);
        assert_ne!(id, crate::block::AIR);
        world.set_block(x, y, z, crate::block::AIR);
        let mut inventory = crate::inventory::Inventory::new(10);
        assert!(inventory.add(id, 1));
        assert_eq!(inventory.count(id), 1);
        assert_eq!(world.block_at(x, y, z), crate::block::AIR);
    }

    /// A round world: hand-placed storage cells (a floor on the +Y chart running across its +u seam
    /// onto the +X chart).
    fn curved_floor() -> (World, std::sync::Arc<crate::space::atlas::Atlas>, i64, i64, i64) {
        use crate::space::atlas::{Atlas, Patch};
        let mut world = World::generate();
        let centre = DVec3::new(2.0e7, 3.0e7, -1.0e7);
        let r = 3_000i64;
        let atlas = std::sync::Arc::new(Atlas::new(centre, r, r + 64, false, crate::space::atlas::STORAGE_X0));
        world.set_atlases(vec![atlas.clone()]);
        let top = Patch::Shell { band: 0, face: crate::coord::Face::PosY };
        let b = atlas.bands[0];
        let (k, mid) = (r - b.r_lo - 1, b.n / 2);
        let stone = world.registry().id_by_label("rock").unwrap();
        // The floor: chart cells up to the seam, and beyond it the cells holding the extended map's
        // points (the +X chart's cells).
        for i in b.n - 12..b.n + 8 {
            for dj in -3..=3 {
                let p = atlas.embed(top, DVec3::new(i as f64 + 0.5, k as f64 + 0.5, (mid + dj) as f64 + 0.5));
                let s = atlas.storage_of(p).expect("covered");
                world.ensure_around(DVec3::new(s[0] as f64, s[1] as f64, s[2] as f64));
                world.set_block(s[0] as i32, s[1] as i32, s[2] as i32, stone);
            }
        }
        (world, atlas, k, mid, b.n)
    }

    #[test]
    fn a_ray_on_a_round_world_hits_the_storage_cell_below() {
        use crate::space::atlas::Patch;
        let (world, atlas, k, mid, n) = curved_floor();
        let top = Patch::Shell { band: 0, face: crate::coord::Face::PosY };
        let i = n - 8;
        let eye = atlas.embed(top, DVec3::new(i as f64 + 0.5, k as f64 + 2.6, mid as f64 + 0.5));
        let down = (atlas.centre - eye).normalize();
        let hit = raycast(&world, eye, down, REACH).expect("the floor below");
        let cell = atlas.storage(top, [i, k, mid]);
        assert_eq!(hit.block, (cell[0] as i32, cell[1] as i32, cell[2] as i32));
        assert_eq!(hit.previous, (cell[0] as i32, cell[1] as i32 + 1, cell[2] as i32), "placement goes on top");
    }

    #[test]
    fn a_ray_crosses_a_chart_seam_and_reports_the_real_cell() {
        use crate::space::atlas::Patch;
        let (world, atlas, k, mid, n) = curved_floor();
        let top = Patch::Shell { band: 0, face: crate::coord::Face::PosY };
        let eye = atlas.embed(top, DVec3::new(n as f64 - 2.5, k as f64 + 2.6, mid as f64 + 0.5));
        let aim = atlas.embed(top, DVec3::new(n as f64 + 3.5, k as f64 + 0.5, mid as f64 + 0.5));
        let hit = raycast(&world, eye, aim - eye, REACH * 2.0).expect("the floor across the seam");
        let s = [hit.block.0 as i64, hit.block.1 as i64, hit.block.2 as i64];
        let (patch, _) = atlas.locate(s).expect("a real storage cell, not one outside a box");
        assert_eq!(patch, Patch::Shell { band: 0, face: crate::coord::Face::PosX }, "on the far chart");
        assert!(world.is_solid(hit.block.0, hit.block.1, hit.block.2));
        // Where the straight physical ray meets the floor's top (y = k + 1 in the extended map).
        let f = (2.6 - 1.0) / (2.6 - 0.5);
        let expect = atlas.embed(top, DVec3::new(n as f64 - 2.5 + f * 6.0, k as f64 + 0.5, mid as f64 + 0.5));
        let got = crate::space::atlas::embed_cell(world.atlases(), hit.block).unwrap();
        assert!((got - expect).length() < 1.5, "hit {got:?} vs the crossing {expect:?}");
        let prev = [hit.previous.0 as i64, hit.previous.1 as i64, hit.previous.2 as i64];
        assert!(atlas.locate(prev).is_some(), "placement lands in a real cell too");
        assert!(!world.is_solid(hit.previous.0, hit.previous.1, hit.previous.2));
    }
}
