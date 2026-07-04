//! Rendering abstractions shared by anything that draws itself in the 3D scene.
use voxel_engine::Frame3D;

/// Something that can draw itself into an active 3D drawing context.
///
/// The engine exposes a single concrete 3D recording type, [`Frame3D`], so this
/// takes it directly — no generic indirection needed.
pub trait Render {
    fn render(&self, f: &mut Frame3D);
}
