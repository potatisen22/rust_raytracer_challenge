use crate::canvas::Canvas;

mod types;
mod canvas;

struct Projectile {
    position: types::Tuple4D,
    velocity: types::Tuple4D
}

struct Environment {
    gravity: types::Tuple4D,
    wind: types::Tuple4D
}

fn tick(env: &Environment, proj: &mut Projectile) {
    proj.position = proj.position + proj.velocity;
    proj.velocity = proj.velocity + env.gravity + env.wind;
}

fn main() {
    let width = 900;
    let height = 550;
    let env = Environment {gravity: types::vector(0.0, -0.1, 0.0), wind: types::vector(-0.04, 0.0, 0.0) };
    let mut proj = Projectile {position: types::point(0.0, 1.0, 0.0), velocity: types::vector(1.0, 1.8, 0.0).normalize() * 11.25 };
    let mut iterator = 1;
    let mut render_canvas: Canvas = Canvas::new(width, height);
    canvas::write_pixel(&mut render_canvas ,proj.position.x as usize ,proj.position.y as usize,types::color(1.0,0.0,0.0));
    while proj.position.y > 0.0 {
        tick(&env, &mut proj);
        if (proj.position.y as usize) < render_canvas.height && (proj.position.x as usize) < render_canvas.width {
            canvas::write_pixel(&mut render_canvas, proj.position.x as usize, height - (proj.position.y as usize), types::color(1.0, 0.0, 0.0));
        }
        println!("Tick Number: {}", iterator);
        println!("Current Position [{:?}, {:?}, {:?}]", proj.position.x, proj.position.y, proj.position.z);
        iterator+=1;
    }
    //post-processing, save to ppm file and all that.
    let ppm_string = canvas::canvas_to_ppm_header(&render_canvas) + &canvas::canvas_to_ppm_body(&render_canvas);
    let _ = std::fs::write("test.ppm", ppm_string);
}
