use std::f64::consts::PI;
use crate::canvas::Canvas;
use crate::types::{point_light, scaling};
use crate::world::render;
use crate::shapes::sphere;

mod types;
mod canvas;
mod world;
mod shapes;

fn main() {
    let mut world = world::default_world();
    //Create floor
    let mut floor = sphere();
    floor.transform = types::scaling(10.0, 0.01, 10.0);
    floor.material = types::material();
    floor.material.color = types::color(1.0, 0.9, 0.9);
    floor.material.specular = 0.0;
    //create left wall
    let mut left_wall = sphere();
    left_wall.transform = types::translation(0.0, 0.0, 5.0) *
        types::rotation_y(PI / -4.0) * types::rotation_x(PI / 2.0) *
        scaling(10.0, 0.01, 10.0);
    left_wall.material = floor.material.clone();
    //create right wall
    let mut right_wall = sphere();
    right_wall.transform = types::translation(0.0, 0.0, 5.0) *
        types::rotation_y(PI / 4.0) * types::rotation_x(PI / 2.0) *
        scaling(10.0, 0.01, 10.0);
    right_wall.material = floor.material.clone();
    //create middle where, unit sphere which is translated slightly upward and green
    let mut middle = sphere();
    middle.transform = types::translation(-0.5, 1.0, 0.5);
    middle.material = types::material();
    middle.material.color = types::color(0.1, 1.0, 0.5);
    middle.material.diffuse = 0.7;
    middle.material.specular = 0.3;
    //create small sphere to the right, scaled in half
    let mut right_sphere = sphere();
    right_sphere.transform = types::translation(1.5, 0.5, -0.5) * scaling(0.5, 0.5, 0.1);
    right_sphere.material = types::material();
    right_sphere.material.color = types::color(1.0, 0.1, 1.0);
    right_sphere.material.diffuse = 0.7;
    right_sphere.material.specular = 0.3;
    //create smallest sphere, scaled by a this, before translating
    let mut left_sphere = sphere();
    left_sphere.transform = types::translation(-1.5, 0.33, -0.75) * scaling(0.33, 0.33, 0.33);
    left_sphere.material = types::material();
    left_sphere.material.color = types::color(1.0, 0.8, 0.1);
    left_sphere.material.diffuse = 0.7;
    left_sphere.material.specular = 0.3;
    //now light source of the world
    world.light = point_light(types::point(-10.0, 10.0, -10.0), types::color(1.0, 1.0, 1.0));
    //Create camera
    let mut camera = world::camera(500, 250, PI/3.0);
    camera.transform = world::view_transform(types::point(0.0, 1.5, -5.0), types::point(0.0, 1.0, 0.0), types::vector(0.0, 1.0, 0.0));
    world.objects = vec![floor, left_wall, right_wall, middle, right_sphere, left_sphere];
    let render_canvas: Canvas = render(&camera, &world);


    //post-processing, save to ppm file and all that.
    let ppm_string = canvas::canvas_to_ppm_header(&render_canvas) + &canvas::canvas_to_ppm_body(&render_canvas);
    let _ = std::fs::write("test.ppm", ppm_string);
}
