//! Rendering abstractions shared by anything that draws itself in the 3D scene.
use raylib::prelude::*;

/// Something that can draw itself into an active 3D drawing context.
///
/// Generic over the concrete raylib draw handle so it works with `RaylibMode3D`
/// and any other [`RaylibDraw3D`] implementor without naming its lifetimes.
pub trait Render {
    fn render<D: RaylibDraw3D>(&self, d: &mut D);
}
