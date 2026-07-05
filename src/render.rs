//! Rendering abstractions shared by anything that draws itself in the 3D scene.
use voxel_engine::{DVec3, Frame3D};

/// Something that can draw itself into an active 3D drawing context.
///
/// The engine exposes a single concrete 3D recording type, [`Frame3D`], so this
/// takes it directly — no generic indirection needed. `cam` is the camera's
/// world-space eye in `f64`: the camera itself is rebased to the origin (the
/// engine is `f32`), so implementors draw everything *relative* to `cam`,
/// which keeps far-coordinate geometry precise on the GPU.
pub trait Render {
    fn render(&self, f: &mut Frame3D, cam: DVec3);
}
