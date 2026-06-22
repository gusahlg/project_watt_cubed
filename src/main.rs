/// This is some prototype code for the game, the api it uses for interacting with Vulkan is under
/// development so things will change on this end as well to reflect those changes.
use raylib::prelude::*;
fn main() {
    let (mut rl, thread) = raylib::init()
        .size(1280, 720)
        .title("voxel prototype")
        .build();

    rl.set_target_fps(100);

    let camera = Camera3D::perspective(
        Vector3::new(10.0, 10.0, 10.0),
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        60.0,
    );

    while !rl.window_should_close() {
        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::SKYBLUE);

        let mut d3 = d.begin_mode3D(camera);
        d3.draw_cube(Vector3::new(0.0, 0.0, 0.0), 1.0, 1.0, 1.0, Color::RED);
        d3.draw_cube(Vector3::new(1.0, -1.0, 0.0), 1.0, 1.0, 1.0, Color::BLUE);
        d3.draw_cube(Vector3::new(5.0, 0.0, 0.0), 1.0, 1.0, 1.0, Color::GREEN);
    }
}
