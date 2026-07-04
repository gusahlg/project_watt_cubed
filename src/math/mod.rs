//! Geometry helpers shared across the game.
use voxel_engine::Vec3;

/// An axis-aligned bounding box defined by a centre point and half-extents.
#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub center: Vec3,
    pub half: Vec3,
}

impl Aabb {
    pub fn new(center: Vec3, half: Vec3) -> Self {
        Self { center, half }
    }

    /// The lower corner (centre minus half-extents).
    pub fn min(&self) -> Vec3 {
        self.center - self.half
    }

    /// The upper corner (centre plus half-extents).
    pub fn max(&self) -> Vec3 {
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
    /// floor of the box minimum to the floor of its maximum.
    pub fn voxel_cells(&self) -> impl Iterator<Item = (i32, i32, i32)> {
        let min = self.min();
        let max = self.max();
        let (x0, x1) = (min.x.floor() as i32, max.x.floor() as i32);
        let (y0, y1) = (min.y.floor() as i32, max.y.floor() as i32);
        let (z0, z1) = (min.z.floor() as i32, max.z.floor() as i32);

        (x0..=x1)
            .flat_map(move |x| (y0..=y1).flat_map(move |y| (z0..=z1).map(move |z| (x, y, z))))
    }
}

/// Something that occupies an axis-aligned box in the world. Implementors get
/// uniform collision handling via [`Aabb`].
pub trait Bounded {
    fn aabb(&self) -> Aabb;
}
