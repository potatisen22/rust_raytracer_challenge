#![allow(dead_code)]
#![allow(unused_imports)]

use std::thread::yield_now;
use crate::types;
use crate::types::{identity_matrix, material, position, scaling, translation, vector, Color, Intersection, Light, Material, Matrix, Ray, Sphere, Tuple4D};
use crate::types::{color, point, point_light, sphere};
use crate::canvas;
use crate::canvas::write_pixel;

pub struct World {
    pub light: Light,
    pub objects: Vec<Sphere>
}

pub fn default_world() -> World {
    let mut sphere_1 = sphere();
    sphere_1.material.color = color(0.8, 1.0, 0.6);
    sphere_1.material.diffuse = 0.7;
    sphere_1.material.specular = 0.2;
    let mut sphere_2 = sphere();
    sphere_2.transform = scaling(0.5, 0.5, 0.5);
    let light = point_light(point(-10.0, 10.0, -10.0), color(1.0, 1.0, 1.0));
    World {
        light,
        objects: vec![sphere_1, sphere_2]
    }
}

pub fn intersect_world(world: &World, ray: &Ray) -> Vec<Intersection> {
    let mut xs = vec![];
    for object in world.objects.iter() {
        let intersection = types::intersect(object, ray);
        for x in intersection {
            xs.push(x);
        }
        xs.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());
    }
    xs
}

pub struct Computations {
    t: f64,
    object: Sphere,
    point: Tuple4D,
    eye_vector: Tuple4D,
    normal_vector: Tuple4D,
    inside: bool
}

impl Computations {
    fn new() -> Self {
        Computations{ t: 0.0, object: sphere(), point: point(0.0, 0.0, 0.0), eye_vector: vector(0.0, 0.0, 0.0), normal_vector: vector(0.0, 0.0, 0.0), inside: false}
    }
}

pub struct Camera {
    h_size: usize,
    v_size: usize,
    field_of_view: f64,
    half_width: f64,
    half_height: f64,
    pixel_size: f64,
    pub transform: Matrix<4,4>
}

impl Camera {
    fn new(h_size: usize, v_size: usize, field_of_view: f64) -> Self {
        Camera{ h_size, v_size, field_of_view,
            half_width: 0.0, half_height: 0.0, pixel_size: 0.0 ,
            transform: identity_matrix::<4>() }
    }
}

pub fn prepare_computations(intersection: &Intersection, ray: Ray) -> Computations {
    let mut computations = Computations::new();
    computations.t = intersection.t;
    computations.object = intersection.object;
    computations.point = position(&ray, computations.t);
    computations.eye_vector = - ray.direction;
    computations.normal_vector = types::normal_at(&computations.object, computations.point);
    if types::dot(&computations.normal_vector, &computations.eye_vector) < 0.0
    {
        computations.inside = true;
        computations.normal_vector = -computations.normal_vector;
    }
    else { computations.inside = false; }
    computations
}

pub fn shade_hit(world: &World, computations: &Computations) -> Color
{
    types::lighting(computations.object.material, world.light, computations.point, computations.eye_vector, computations.normal_vector)
}

pub fn color_at(world: &World, ray: Ray) -> Color {
    let intersections = intersect_world(world, &ray);
    if intersections.len() == 0 {
        color(0.0, 0.0, 0.0)
    }
    else {
        let hit = types::hit(&intersections).unwrap();
        let computations = prepare_computations(&hit, ray);
        shade_hit(world, &computations)
    }
}

pub fn view_transform(from: Tuple4D, to: Tuple4D, up: Tuple4D) -> Matrix::<4,4> {
    let forward = (to - from).normalize();
    let left = types::cross(&forward, &up.normalize());
    let true_up = types::cross(&left, &forward);
    let mut orientation = Matrix::<4,4>::new();
    orientation.insert_row(0, vec![left.x, left.y, left.z, 0.0]);
    orientation.insert_row(1, vec![true_up.x, true_up.y, true_up.z, 0.0]);
    orientation.insert_row(2, vec![-forward.x, -forward.y, -forward.z, 0.0]);
    orientation.insert_row(3, vec![0.0, 0.0, 0.0, 1.0]);
    orientation * translation(-from.x, -from.y, -from.z)
}

pub fn camera(h_size: usize, v_size: usize, field_of_view: f64) -> Camera {
    let mut return_camera = Camera::new(h_size, v_size, field_of_view);
    let half_view = f64::tan(return_camera.field_of_view / 2.0);
    let aspect_ratio = return_camera.h_size as f64 / return_camera.v_size as f64;
    if aspect_ratio >= 1.0 {
        return_camera.half_width = half_view;
        return_camera.half_height = half_view / aspect_ratio;
    }
    else {
        return_camera.half_width = half_view * aspect_ratio;
        return_camera.half_height = half_view;

    }
    return_camera.pixel_size = (return_camera.half_width * 2.0) / return_camera.h_size as f64;
    return_camera
}

pub fn ray_for_pixel(camera: &Camera, px: usize, py: usize) -> Ray {
    //the offset from the edge of the canvas to pixels center
    let x_offset = (px as f64 + 0.5) * camera.pixel_size;
    let y_offset = (py as f64 + 0.5) * camera.pixel_size;
    //the untransformed coordinates of the pixel in world space.
    //reminding that the camera looks towards -z so +x is to the "left"
    let world_x = camera.half_width - x_offset;
    let world_y = camera.half_height - y_offset;
    //using the camera matrix, transforms the canvas point and the origin
    //then computes the rays direction vector. Canvas is at z=-1
    let pixel = types::inverse(&camera.transform) * point(world_x, world_y, -1.0);
    let origin = types::inverse(&camera.transform) * point(0.0, 0.0, 0.0);
    let direction = (pixel - origin).normalize();
    Ray::new(origin, direction)
}

pub fn render(camera: &Camera, world: &World) -> canvas::Canvas {
    let mut image = canvas::Canvas::new(camera.h_size, camera.v_size);
    for y in 0..camera.v_size {
        for x in 0..camera.h_size {
            let ray = ray_for_pixel(camera, x, y);
            let color = color_at(world, ray);
            write_pixel(&mut image, x, y, color);
        }
    }
    image
}
#[cfg(test)]
mod tests {
    use std::f64::consts::PI;
    use num_traits::FloatConst;
    use crate::types::{intersection, rotation_y, scaling, translation, vector};
    use super::*;

    #[test]
    fn test_default_world() {
        let light = point_light(point(-10.0, 10.0, -10.0), color(1.0, 1.0, 1.0));
        let mut sphere = sphere();
        sphere.material.color = color(0.8, 1.0, 0.6);
        sphere.material.diffuse = 0.7;
        sphere.material.specular = 0.2;
        let mut sphere_2 = types::sphere();
        sphere_2.transform = scaling(0.5, 0.5, 0.5);
        let my_world = default_world();
        assert_eq!(my_world.light, light);
        assert!(my_world.objects[0].transform == sphere.transform && my_world.objects[0].material == sphere.material);
        assert!(my_world.objects[1].transform == sphere_2.transform && my_world.objects[1].material == sphere_2.material);
    }
    #[test]
    fn test_intersect_world_with_ray() {
        let my_world = default_world();
        let my_ray = Ray::new(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let xs = intersect_world(&my_world, &my_ray);
        assert_eq!(xs.len(), 4);
        assert_eq!(xs[0].t, 4.0);
        assert_eq!(xs[1].t, 4.5);
        assert_eq!(xs[2].t, 5.5);
        assert_eq!(xs[3].t, 6.0);
    }
    #[test]
    fn test_precompute_state_of_intersections() {
        let my_ray = Ray::new(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let my_shape = sphere();
        let i = intersection(4.0, &my_shape);
        let computations = prepare_computations(&i, my_ray);
        assert_eq!(computations.t, i.t);
        assert_eq!(computations.object,i.object);
        assert_eq!(computations.point, point(0.0, 0.0, -1.0));
        assert_eq!(computations.eye_vector, vector(0.0, 0.0, -1.0));
        assert_eq!(computations.normal_vector, vector(0.0, 0.0, -1.0));
    }
    #[test]
    fn test_hit_when_intersection_on_outside() {
        let ray = Ray::new(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let shape = sphere();
        let i = intersection(4.0, &shape);
        let computations = prepare_computations(&i, ray);
        assert!(!computations.inside);
    }
    #[test]
    fn test_hit_when_intersection_on_inside() {
        let ray = Ray::new(point(0.0, 0.0, 0.0), vector(0.0, 0.0, 1.0));
        let shape = sphere();
        let i = intersection(1.0, &shape);
        let computations = prepare_computations(&i, ray);
        assert_eq!(computations.point, point(0.0, 0.0, 1.0));
        assert_eq!(computations.eye_vector, vector(0.0, 0.0, -1.0));
        assert!(computations.inside);
        assert_eq!(computations.normal_vector, vector(0.0, 0.0, -1.0));
    }
    #[test]
    fn test_shading_an_intersection() {
        let world = default_world();
        let ray = Ray::new(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let shape = world.objects[0];
        let i = intersection(4.0, &shape);
        let computations = prepare_computations(&i, ray);
        let c = shade_hit(&world, &computations);
        assert_eq!(c, color(0.38066, 0.47583, 0.2855));
    }
    #[test]
    fn test_shading_an_intersection_from_the_inside() {
        let mut world = default_world();
        world.light = point_light(point(0.0, 0.25, 0.0), color(1.0, 1.0, 1.0));
        let ray = Ray::new(point(0.0, 0.0, 0.0), vector(0.0, 0.0, 1.0));
        let shape = world.objects[1];
        let i = intersection(0.5, &shape);
        let computations = prepare_computations(&i, ray);
        let c = shade_hit(&world, &computations);
        assert_eq!(c, color(0.90498, 0.90498, 0.90498));
    }
    #[test]
    fn test_color_when_ray_misses() {
        let world = default_world();
        let ray = Ray::new(point(0.0, 0.0, -5.0), vector(0.0, 1.0, 0.0));
        let c = color_at(&world, ray);
        assert_eq!(c, color(0.0, 0.0, 0.0));
    }
    #[test]
    fn test_color_when_ray_hits() {
        let world = default_world();
        let ray = Ray::new(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let c = color_at(&world, ray);
        assert_eq!(c, color(0.38066, 0.47583, 0.2855));
    }
    #[test]
    fn test_color_with_an_intersection_behind_the_ray() {
        let mut world = default_world();
        world.objects[0].material.ambient = 1.0;
        world.objects[1].material.ambient = 1.0;
        let ray = Ray::new(point(0.0, 0.0, 0.75), vector(0.0, 0.0, -1.0));
        let color = color_at(&world, ray);
        assert_eq!(color, world.objects[1].material.color);
    }

    #[test]
    fn test_transformation_matrix_for_default_orientation() {
        let from = point(0.0, 0.0, 0.0);
        let to = point(0.0, 0.0, -1.0);
        let up = vector(0.0, 1.0, 0.0);
        let transform = view_transform(from, to, up);
        assert_eq!(transform, types::identity_matrix::<4>())
    }
    #[test]
    fn test_view_transformation_matrix_looking_in_positive_z_direction() {
        let from = point(0.0, 0.0, 0.0);
        let to = point(0.0, 0.0, 1.0);
        let up = vector(0.0, 1.0, 0.0);
        let transform = view_transform(from, to, up);
        assert_eq!(transform, scaling(-1.0, 1.0, -1.0))
    }
    #[test]
    fn test_view_transformation_moves_world() {
        let from = point(0.0, 0.0, 8.0);
        let to = point(0.0, 0.0, 0.0);
        let up = vector(0.0, 1.0, 0.0);
        let transform = view_transform(from, to, up);
        assert_eq!(transform, translation(0.0, 0.0, -8.0))
    }
    #[test]
    fn test_arbitrary_view_transform() {
        let from = point(1.0, 3.0, 2.0);
        let to = point(4.0, -2.0, 8.0);
        let up = vector(1.0, 1.0, 0.0);
        let transform = view_transform(from, to, up);
        let mut expected_result = types::Matrix::<4,4>::new();
        expected_result.insert_column(0,vec![-0.50709, 0.76772, -0.35857, 0.00000]);
        expected_result.insert_column(1,vec![0.50709, 0.60609, 0.59761, 0.00000]);
        expected_result.insert_column(2,vec![0.67612, 0.12122, -0.71714, 0.00000]);
        expected_result.insert_column(3,vec![-2.36643, -2.82843, 0.00000, 1.00000]);
        assert_eq!(transform, expected_result);
    }
    #[test]
    fn test_constructing_camera() {
        let h_size = 160;
        let v_size = 120;
        let field_of_view = PI/2.0;
        let c = camera(h_size,v_size,field_of_view);
        assert_eq!(c.h_size, h_size);
        assert_eq!(c.v_size, v_size);
        assert_eq!(c.field_of_view, field_of_view);
        assert_eq!(c.transform, types::identity_matrix::<4>());
    }
    #[test]
    fn test_pixel_size_for_horizontal_canvas() {
        let c = camera(200,125,PI/2.0);
        assert!((c.pixel_size - 0.01).abs() < types::EPSILON);
    }
    #[test]
    fn test_pixel_size_for_vertical_canvas() {
        let c = camera(125,200,PI/2.0);
        assert!((c.pixel_size - 0.01).abs() < types::EPSILON);
    }
    #[test]
    fn test_construct_ray_through_center_of_canvas() {
        let c = camera(201, 101, PI/2.0);
        let my_ray = ray_for_pixel(&c, 100, 50);
        assert_eq!(my_ray.origin, point(0.0, 0.0, 0.0));
        assert_eq!(my_ray.direction, vector(0.0, 0.0, -1.0));
    }
    #[test]
    fn test_construct_ray_through_corner_of_canvas() {
        let c = camera(201, 101, PI/2.0);
        let my_ray = ray_for_pixel(&c, 0, 0);
        assert_eq!(my_ray.origin, point(0.0, 0.0, 0.0));
        assert_eq!(my_ray.direction, vector(0.66519, 0.33259, -0.66851));
    }
    #[test]
    fn test_construct_ray_when_camera_is_transformed() {
        let mut c = camera(201, 101, PI/2.0);
        c.transform = rotation_y(PI/4.0) * translation(0.0, -2.0, 5.0);
        let my_ray = ray_for_pixel(&c, 100, 50);
        assert_eq!(my_ray.origin, point(0.0, 2.0, -5.0));
        assert_eq!(my_ray.direction, vector(f64::sqrt(2.0)/2.0, 0.0, -f64::sqrt(2.0)/2.0));
    }
    #[test]
    fn test_rendering_world_with_camera() {
        let my_world = default_world();
        let mut my_camera = camera(11,11, PI/2.0);
        let from = point(0.0, 0.0, -5.0);
        let to = point(0.0, 0.0, 0.0);
        let up = vector(0.0, 1.0, 0.0);
        my_camera.transform = view_transform(from, to, up);
        let image = render(&my_camera, &my_world);
        assert_eq!(canvas::pixel_at(&image, 5, 5).unwrap(), color(0.38066, 0.47583, 0.2855));
    }
}