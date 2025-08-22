#![allow(dead_code)]
#![allow(unused_imports)]

use std::sync::atomic::{AtomicU64, Ordering};
use crate::types::{point, dot, inverse, material, transform, transpose, Material, Matrix, Ray, Tuple4D, identity_matrix};

#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    pub id: u64,
    pub transform: Matrix::<4,4>,
    pub material: Material
}

impl PartialEq for Sphere {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub fn sphere() -> Sphere {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    Sphere { id, transform: identity_matrix::<4>(), material: material() }
}

pub fn set_transform(sphere: &mut Sphere, m: &Matrix<4,4>) {
    sphere.transform = *m;
}


#[derive(Debug, Clone, Copy)]
pub struct Intersection {
    pub t: f64,
    pub object: Sphere
}

impl PartialEq for Intersection {
    fn eq(&self, other: &Self) -> bool {
        self.object == other.object && self.t == other.t
    }
}
pub fn intersection(t: f64, sphere: &Sphere ) -> Intersection {
    Intersection { t, object: *sphere }
}

pub fn intersect(sphere: &Sphere, ray: &Ray) -> Vec<Intersection> {
    let ray2 = transform(ray, &inverse(&sphere.transform));
    let sphere_to_ray = ray2.origin - point(0.0, 0.0, 0.0);
    let a = dot(&ray2.direction, &ray2.direction);
    let b = 2.0 * dot(&ray2.direction, &sphere_to_ray);
    let c = dot(&sphere_to_ray, &sphere_to_ray) - 1.0;
    let discriminant: f64 = (b * b) - (4.0 * a * c);
    if discriminant < 0.0 {
        return vec![];
    };
    let t1 = (-b - f64::sqrt(discriminant)) / (2.0 * a);
    let t2 = (-b + f64::sqrt(discriminant)) / (2.0 * a);
    vec![intersection(t1, sphere),intersection(t2, sphere)]
}

pub type Intersections = Vec<Intersection>;

pub fn intersections<I>(iter: I) -> Intersections
where I: IntoIterator<Item = Intersection>,
{
    let xs: Vec<Intersection> = iter.into_iter().collect();
    xs
}

pub fn hit(intersections: &Intersections) -> Option<Intersection> {
    let mut hit_list = intersections.clone();
    hit_list.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());
    hit_list.into_iter().find(|i| i.t >= 0.0)
}

pub fn normal_at(sphere: &Sphere, in_point: Tuple4D) -> Tuple4D {
    let object_point = inverse(&sphere.transform) * in_point;
    let object_normal = object_point - point(0.0, 0.0, 0.0);
    let mut world_normal = transpose(inverse(&sphere.transform)) * object_normal;
    world_normal.w = 0.0;
    world_normal.normalize()
}


#[cfg(test)]
mod tests {
    use std::f64::consts::PI;
    use num_traits::FloatConst;
    use crate::types::{rotation_y, rotation_z, scaling, translation, vector, EPSILON};
    use super::*;

    #[test]
    fn test_sphere_default_transformation() {
        let s = sphere();
        assert_eq!(s.transform, identity_matrix::<4>());
    }

    #[test]
    fn test_changing_sphere_transformation() {
        let mut sphere = sphere();
        let translation_matrix = translation(2.0, 3.0, 4.0);
        set_transform(&mut sphere, &translation_matrix);
        sphere.transform = translation_matrix;
        assert_eq!(sphere.transform, translation_matrix);
    }
    #[test]
    fn test_intersect_scaled_sphere_with_ray() {
        let ray = Ray::new(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let mut sphere = sphere();
        set_transform(&mut sphere, &scaling(2.0, 2.0, 2.0));
        let xs = intersect(&sphere, &ray);
        assert_eq!(xs.len(), 2);
        assert_eq!(xs[0].t, 3.0);
        assert_eq!(xs[1].t, 7.0);
    }
    #[test]
    fn test_intersect_translated_sphere_with_ray() {
        let ray = Ray::new(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let mut sphere = sphere();
        set_transform(&mut sphere, &translation(5.0, 0.0, 0.0));
        let xs = intersect(&sphere, &ray);
        assert_eq!(xs.len(), 0);
    }



    #[test]
    fn test_normal_on_sphere_at_point_on_x_axis() {
        let my_sphere = sphere();
        let my_normal = normal_at(&my_sphere,point(1.0, 0.0, 0.0));
        assert_eq!(my_normal, vector(1.0, 0.0, 0.0));
    }
    #[test]
    fn test_normal_on_sphere_at_point_on_y_axis() {
        let my_sphere = sphere();
        let my_normal = normal_at(&my_sphere,point(0.0, 1.0, 0.0));
        assert_eq!(my_normal, vector(0.0, 1.0, 0.0));
    }
    #[test]
    fn test_normal_on_sphere_at_point_on_z_axis() {
        let my_sphere = sphere();
        let my_normal = normal_at(&my_sphere,point(0.0, 0.0, 1.0));
        assert_eq!(my_normal, vector(0.0, 0.0, 1.0));
    }
    #[test]
    fn test_normal_on_sphere_at_nonaxial_point() {
        let my_sphere = sphere();
        let my_normal = normal_at(&my_sphere,point(f64::sqrt(3.0)/3.0, f64::sqrt(3.0)/3.0, f64::sqrt(3.0)/3.0));
        assert_eq!(my_normal, vector(f64::sqrt(3.0)/3.0, f64::sqrt(3.0)/3.0, f64::sqrt(3.0)/3.0));
    }
    #[test]
    fn test_normal_is_normalized_vector() {
        let my_sphere = sphere();
        let my_normal = normal_at(&my_sphere,point(f64::sqrt(3.0)/3.0, f64::sqrt(3.0)/3.0, f64::sqrt(3.0)/3.0));
        assert_eq!(my_normal.normalize(), my_normal);
    }
    #[test]
    fn test_normal_on_translated_sphere() {
        let mut my_sphere = sphere();
        set_transform(&mut my_sphere, &translation(0.0, 1.0, 0.0));
        let my_normal = normal_at(&my_sphere,point(0.0, 1.70711, -std::f64::consts::FRAC_1_SQRT_2));
        assert_eq!(my_normal, vector(0.0, std::f64::consts::FRAC_1_SQRT_2, -std::f64::consts::FRAC_1_SQRT_2));
    }
    #[test]
    fn test_normal_on_transformed_sphere() {
        let mut my_sphere = sphere();
        let transform_matrix = scaling(1.0, 0.5, 1.0)* rotation_z(PI/5.0);
        set_transform(&mut my_sphere, &transform_matrix);
        let my_normal = normal_at(&my_sphere,point(0.0, f64::sqrt(2.0)/2.0, -f64::sqrt(2.0)/2.0));
        assert_eq!(my_normal, vector(0.0, 0.97014, -0.24254));
    }
    #[test]
    fn test_intersection_function() {
        let sphere = sphere();
        let i = intersection(3.5, &sphere);
        assert_eq!(i.t, 3.5);
        assert_eq!(i.object, sphere);
    }
    #[test]
    fn test_aggregating_intersections() {
        let sphere = sphere();
        let inter1 = intersection(1.0, &sphere);
        let inter2 = intersection(2.0, &sphere);
        let xs = intersections([inter1, inter2]);
        assert_eq!(xs.len(), 2);
        assert_eq!(xs[0].t, 1.0);
        assert_eq!(xs[1].t, 2.0);
    }
    #[test]
    fn test_intersect_sets_object_of_intersection() {
        let ray = Ray::new(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let sphere = sphere();
        let xs = intersect(&sphere, &ray);
        assert_eq!(xs.len(), 2);
        assert_eq!(xs[0].object, sphere);
        assert_eq!(xs[1].object, sphere);
    }
    #[test]
    fn test_hit_all_intersections_positive() {
        let s = sphere();
        let i1 = intersection(1.0, &s);
        let i2 = intersection(2.0, &s);
        let xs = intersections([i1, i2]);
        let i = hit(&xs);
        assert_eq!(i.unwrap(), i1);
    }
    #[test]
    fn test_hit_some_intersections_negative() {
        let s = sphere();
        let i1 = intersection(-1.0, &s);
        let i2 = intersection(1.0, &s);
        let xs = intersections([i2, i1]);
        let i = hit(&xs);
        assert_eq!(i.unwrap(), i2);
    }
    #[test]
    fn test_hit_all_intersections_negative() {
        let s = sphere();
        let i1 = intersection(-2.0, &s);
        let i2 = intersection(-1.0, &s);
        let xs = intersections([i2, i1]);
        let i = hit(&xs);
        assert_eq!(i, None);
    }
    #[test]
    fn test_hit_always_lowest_nonnegative_intersection() {
        let s = sphere();
        let i1 = intersection(5.0, &s);
        let i2 = intersection(7.0, &s);
        let i3 = intersection(-3.0, &s);
        let i4 = intersection(2.0, &s);
        let xs = intersections([i1, i2, i3, i4]);
        let i = hit(&xs);
        assert_eq!(i.unwrap(), i4);
    }

}
