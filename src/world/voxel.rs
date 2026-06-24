//! Block types and their properties.
//!
//! The [`Voxel`] enum and its `is_solid` / `color` lookups are generated from the
//! table below by the [`voxels!`](crate::macros::voxels) macro. To add a new block
//! type, add one row here.
use crate::macros::voxels;

voxels! {
    Air   => { solid: false, color: (0, 0, 0, 0) },
    Grass => { solid: true,  color: (86, 176, 0, 255) },
    Dirt  => { solid: true,  color: (121, 85, 58, 255) },
    Stone => { solid: true,  color: (128, 128, 128, 255) },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn air_is_empty_solids_are_solid() {
        assert!(!Voxel::Air.is_solid());
        assert!(Voxel::Grass.is_solid());
        assert!(Voxel::Stone.is_solid());
    }
}
