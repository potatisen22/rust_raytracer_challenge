use crate::types::Tuple4D;

// Explicitly specify the path to the `types` module file if Rust can't find it automatically.
mod types;

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
    let env = Environment {gravity: Tuple4D::vector(0.0, -0.1, 0.0), wind: Tuple4D::vector(-0.01, 0.0, 0.0) };
    let mut proj = Projectile {position: Tuple4D::point(0.0, 1.0, 0.0), velocity: Tuple4D::vector(1.0, 1.0, 0.0) };
    let mut iterator = 1;
    while proj.position.y > 0.0 {
        tick(&env, &mut proj);
        println!("Tick Number: {}", iterator);
        println!("Current Position [{:?}, {:?}, {:?}]", proj.position.x, proj.position.y, proj.position.z);
        iterator+=1;
    }
}
