use num_traits::ToPrimitive;
use crate::canvas::Canvas;
use crate::types::{scaling, translation};

mod types;
mod canvas;

fn main() {
    let width = 900;
    let height = 550;
    let mut render_canvas: Canvas = Canvas::new(width, height);
    let vector = types::point(0.0, 1.0, 0.0);
    let translation_matrix = translation(usize::to_f64(&width).unwrap()/2.0,usize::to_f64(&height).unwrap()/2.0, 0.0);
    let  scaling_matrix = scaling(150.0,150.0,1.0);
    let const_transform = translation_matrix * scaling_matrix;
    for i in 0..12
    {
        let write_vector = const_transform * types::rotation_z(std::f64::consts::PI/6.0*i.to_f64().unwrap()) * vector;
        canvas::write_pixel(&mut render_canvas ,write_vector.x as usize ,height - (write_vector.y.round() as usize),types::color(1.0,1.0,1.0));
    }
    //post-processing, save to ppm file and all that.
    let ppm_string = canvas::canvas_to_ppm_header(&render_canvas) + &canvas::canvas_to_ppm_body(&render_canvas);
    let _ = std::fs::write("test.ppm", ppm_string);
}
