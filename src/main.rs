use crate::canvas::Canvas;
use crate::types::{intersect};

mod types;
mod canvas;

fn main() {

    //For each width and height in picture, create a ray from an origin point
    //to each point on the canvas (all coordinates) and then check for hit
    //if hit we have red pixel on that canvas coordinate, else black
    //let width = 900;
    //let height = 550;
    let light_origin = types::point(0.0, 0.0, -5.0);
    let wall_z = 10.0;
    let wall_size = 7.0;
    let canvas_pixels = 300;
    let pixel_size = wall_size / canvas_pixels as f64;
    let half = wall_size / 2.0;
    let mut render_canvas: Canvas = Canvas::new(canvas_pixels, canvas_pixels);
    let mut sphere = types::sphere();

    //add material to sphere
    sphere.material = types::material();
    sphere.material.color = types::color(1.0, 0.2, 1.0);

    //add lightsource
    let light_position = types::point(-10.0, 10.0, -10.0);
    let light_color = types::color(1.0, 1.0, 1.0);
    let light = types::point_light(light_position, light_color);

    //sphere.transform = rotation_z((PI/4.0) as f64) * scaling(0.5, 1.0, 1.0);
    for y in 0..canvas_pixels {
        let world_y = half - pixel_size * y as f64;
        for x in 0..canvas_pixels {
            let world_x = -half + pixel_size * x as f64;
            let point_vector = types::point(world_x, world_y, wall_z) - light_origin;
            let ray = types::Ray::new(light_origin, point_vector.normalize());
            let intersections = intersect(&sphere, &ray);
            if intersections.len() > 0 {
                let hit = intersections[0];
                let point = types::position(&ray, hit.t);
                let normal = types::normal_at(&hit.object, point);
                let eye_vector = -ray.direction;
                let color = types::lighting(hit.object.material, light.clone(), point, eye_vector, normal);
                canvas::write_pixel(&mut render_canvas, x, y, color);
            }
            else {
                canvas::write_pixel(&mut render_canvas, x, y, types::color(0.0,0.0,0.0));
            }

        }
    }
    //post-processing, save to ppm file and all that.
    let ppm_string = canvas::canvas_to_ppm_header(&render_canvas) + &canvas::canvas_to_ppm_body(&render_canvas);
    let _ = std::fs::write("test.ppm", ppm_string);
}
