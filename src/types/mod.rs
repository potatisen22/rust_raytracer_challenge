#![allow(dead_code)]
#![allow(unused_imports)]
use std::sync::atomic::{AtomicU64, Ordering};

const EPSILON: f64 = 0.00001;

#[derive(Debug, Copy, Clone)]
pub struct Tuple4D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64
}

impl Tuple4D {

    fn point(x: f64, y: f64, z: f64) -> Tuple4D {
        Self {x, y, z, w: 1.0}
    }
    fn vector(x: f64, y: f64, z: f64) -> Tuple4D {
        Self {x, y, z, w: 0.0}
    }
    pub fn is_vector(&self) -> bool {
        if self.w == 0.0 {
            return true;
        }
        false
    }
    pub fn is_point(&self) -> bool {
        if self.w == 1.0 {
            return true;
        }
        false
    }

    pub fn magnitude(&self) -> f64 {
        // If this ever triggers, you're calling magnitude on a "point".
        debug_assert!(self.is_vector(), "magnitude is only defined for vectors");
        self.x.hypot(self.y).hypot(self.z).hypot(self.w)
    }

    pub fn normalize(&self) -> Tuple4D {
        let mag = self.magnitude();
        Self {
            x: self.x / mag,
            y: self.y / mag,
            z: self.z / mag,
            w: self.w / mag

        }
    }
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    pub fn cross(&self, other: &Self) -> Tuple4D {
        Tuple4D::vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    }
}
impl PartialEq for Tuple4D {
    fn eq(&self, other: &Self) -> bool {
        (self.x - other.x).abs() < EPSILON &&
            (self.y - other.y).abs() < EPSILON &&
            (self.z - other.z).abs() < EPSILON &&
            (self.w - other.w).abs() < EPSILON
    }
}
impl std::ops::Add for Tuple4D {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w
        }
    }
}

impl ::std::ops::Neg for Tuple4D {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: -self.w
        }
    }
}
impl std::ops::Sub for Tuple4D {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w - other.w
        }
    }
}

impl<T: num_traits::ToPrimitive> std::ops::Mul<T> for Tuple4D {
    type Output = Tuple4D;
    fn mul(self, other: T) -> Self::Output {
        let _scalar = other.to_f64().expect("Input scalar cannot be represented as f64");
        Self {
            x: self.x * _scalar,
            y: self.y * _scalar,
            z: self.z * _scalar,
            w: self.w * _scalar
        }
    }
}

impl<T: num_traits::ToPrimitive> std::ops::Div<T> for Tuple4D {
    type Output = Tuple4D;

    fn div(self, other: T) -> Self::Output {
        let _scalar = other.to_f64().expect("Input scalar cannot be represented as f64");
        Self {
            x: self.x / _scalar,
            y: self.y / _scalar,
            z: self.z / _scalar,
            w: self.w / _scalar
        }
    }
}
//Free functions to create different types of Tuples
pub fn point(x: f64, y: f64, z: f64) -> Tuple4D {
    Tuple4D::point(x, y, z)
}

pub fn vector(x: f64, y: f64, z: f64) -> Tuple4D {
    Tuple4D::vector(x, y, z)
}

pub fn dot(vec1: &Tuple4D, vec2: &Tuple4D) -> f64 {
    Tuple4D::dot(vec1, vec2)
}

pub fn cross(vec1: &Tuple4D, vec2: &Tuple4D) -> Tuple4D {
    Tuple4D::cross(vec1, vec2)
}

#[derive(Debug, Copy, Clone)]
pub struct Color {
    pub red: f64,
    pub green: f64,
    pub blue: f64
}

impl std::ops::Add for Color {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let t = Tuple4D::from(self) + Tuple4D::from(rhs);
        Self::from(t)
    }
}

impl std::ops::Sub for Color {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let t = Tuple4D::from(self) - Tuple4D::from(rhs);
        Self::from(t)
    }
}

impl<T: num_traits::ToPrimitive> std::ops::Mul<T> for Color {
    type Output = Color;
    fn mul(self, other: T) -> Self::Output {
        let t = Tuple4D::from(self) * other;
        Self::from(t)
    }
}

impl<T: num_traits::ToPrimitive> std::ops::Div<T> for Color {
    type Output = Color;
    fn div(self, other: T) -> Self::Output {
        let t = Tuple4D::from(self) / other;
        Self::from(t)
    }
}

impl std::ops::Mul<Color> for Color {
    type Output = Color;
    fn mul(self, other: Color) -> Self::Output {
        color(self.red * other.red, self.green * other.green, self.blue * other.blue)
    }
}

impl std::ops::Div<Color> for Color {
    type Output = Color;
    fn div(self, other: Color) -> Self::Output {
        color(self.red / other.red, self.green / other.green, self.blue / other.blue)
    }
}

impl PartialEq for Color {
    fn eq(&self, other: &Self) -> bool {
        Tuple4D::from(*self) == Tuple4D::from(*other)
    }
}


//Color is in its essence just a vector, this is for clarity of code usage
pub fn color(r: f64, g: f64, b: f64) -> Color {
    Color{ red: r, green: g, blue: b }
}

//now to map Color <-> Tuple4d (x -> red, y -> green, z -> blue)
impl From<Color> for Tuple4D {
    fn from(c: Color) -> Self {
        Tuple4D {x: c.red, y: c.green, z: c.blue, w: 0.0}
    }
}

impl From<Tuple4D> for Color {
    fn from(t: Tuple4D) -> Self {
        Color {red: t.x, green: t.y, blue: t.z}
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Matrix<const M: usize, const N: usize> {
    pub data: [[f64; M]; N],
    pub size: usize

}

impl<const M: usize, const N: usize> Matrix<M, N> {
    pub fn new() -> Matrix<M, N> {
        Matrix {data: [[0.0; M]; N], size: M}
    }
    pub fn insert_row(&mut self, row: usize, vector: Vec<f64> ) {
        for i in 0..N {
            self.data[row][i] = vector[i];
        }
    }

    pub fn insert_column(&mut self, column: usize, vector: Vec<f64>) {
        for i in 0..M {
            self.data[i][column] = vector[i];
        }
    }

}

impl<const M: usize> Matrix<M, M> {
    fn identity() -> Matrix<M, M> {
        let mut output_vector = Matrix::<M,M>::new();
        for i in 0..M {
            output_vector[(i,i)] = 1.0;
        }
        output_vector
    }

}

impl<const M: usize, const N: usize> std::ops::Index<(usize, usize)> for Matrix<M, N> {
    type Output = f64;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.data[row][col]
    }
}

// Implement IndexMut for write access: matrix[(row, col)] = value
impl<const M: usize, const N: usize> std::ops::IndexMut<(usize, usize)> for Matrix<M, N> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        &mut self.data[row][col]
    }
}
impl<const M: usize, const N: usize> PartialEq for Matrix<M, N> {
    fn eq(&self, other: &Self) -> bool {
        for i in 0..M {
            for j in 0..N {
                if (self.data[i][j] - other.data[i][j]).abs() > EPSILON {
                    return false;
                }
            }
        }
        true
    }
}

impl<const M: usize, const N: usize, const P: usize> std::ops::Mul<Matrix<N,P>> for Matrix<M,N> {
    type Output = Matrix<M,P>;

    fn mul(self, other: Matrix<N,P>) -> Self::Output {
        let mut output_vector = Matrix::<M,P>::new();
        for row in 0..M {
            for col in 0..P {
                for i in 0..N {
                    output_vector[(row,col)] += self[(row,i)] * other[(i,col)];
                }
            }
        }
        output_vector
    }
}

impl std::ops::Mul<Tuple4D> for Matrix<4,4> {  // Must be 4×4 for Tuple4D output
    type Output = Tuple4D;

    fn mul(self, other: Tuple4D) -> Self::Output {
        Tuple4D {
            x: self[(0,0)] * other.x + self[(0,1)] * other.y + self[(0,2)] * other.z + self[(0,3)] * other.w,
            y: self[(1,0)] * other.x + self[(1,1)] * other.y + self[(1,2)] * other.z + self[(1,3)] * other.w,
            z: self[(2,0)] * other.x + self[(2,1)] * other.y + self[(2,2)] * other.z + self[(2,3)] * other.w,
            w: self[(3,0)] * other.x + self[(3,1)] * other.y + self[(3,2)] * other.z + self[(3,3)] * other.w,
        }
    }
}

pub fn transpose<const M: usize, const N: usize>(m: Matrix<M, N>) -> Matrix<N, M> {
    let mut output_matrix = Matrix::<N, M>::new();
    for i in 0..M {
        for j in 0..N {
            output_matrix[(j,i)] = m[(i,j)];
        }
    }
    output_matrix
}

pub fn determinant<const M: usize, const N: usize>(m: &Matrix<M,N>) -> f64 {
    let mut det = 0.0;
    if m.size == 2 {
        det = m[(0,0)] * m[(1,1)] - m[(0,1)] * m[(1,0)];
    }
    else {
        for column in 0..m.size {
            det += m[(0,column)] * cofactor(m, 0, column)
        }
    }
    det
}

//TODO: M1 and N1 Should be replaced/removed as soon as rust decides to implement generic_const_exprs...any year now...
fn submatrix<const M: usize, const N: usize, const M1: usize, const N1: usize> (
    m: &Matrix<M, N>,
    skip_row: usize,
    skip_col: usize
) -> Matrix<M1, N1>
where
    [(); M1]:,
    [(); N1]:,
{
    assert_eq!(M1, M.saturating_sub(1));
    assert_eq!(N1, N.saturating_sub(1));


    let mut output_matrix = Matrix::<M1, N1>::new();

    let mut output_i = 0;
    for i in 0..M {
        if i == skip_row {
            continue;
        }

        let mut output_j = 0;
        for j in 0..N {
            if j == skip_col {
                continue;
            }

            output_matrix[(output_i, output_j)] = m[(i, j)];
            output_j += 1;
        }
        output_i += 1;
    }

    output_matrix
}

//Because of the generic_const_exprs not working/being implemented in rust the only workaround here is to have special cases depending on the size of the matrix.
//add more cases for more sizes I guess?
pub fn minor<const M: usize, const N: usize>(m: &Matrix<M,N>, row: usize, col: usize) -> f64
{
    let mut det = 0.0;
    if m.size == 3 {
        det = determinant::<2,2>(&submatrix(&m, row, col))
    }
    else if m.size == 4
    {
        det = determinant::<3,3>(&submatrix(&m, row, col))
    }
    det
}

pub fn cofactor<const M: usize, const N: usize>(m: &Matrix<M,N>, row: usize, col: usize) -> f64 {
    let minor_value = minor(m, row, col);
    if (row + col) % 2 == 0 {
        minor_value
    }
    else {
        -minor_value
    }
}

pub fn inverse<const M: usize, const N: usize>(m: &Matrix<M,N>) -> Matrix<M,N> {
    let mut output_matrix = Matrix::<M,N>::new();
    let det = determinant::<M,N>(m);
    assert!(!det.is_nan());
    assert!(!det.is_infinite());
    assert!(det != 0.0);
    for row in 0..M {
        for col in 0..N {
            let cofactor_value = cofactor(m, row, col);
            output_matrix[(col,row)] = cofactor_value / det;
        }
    }
    output_matrix
}

pub fn translation(x: f64, y: f64, z: f64) -> Matrix<4,4> {
    let mut identity_matrix = Matrix::<4,4>::identity();
    identity_matrix.insert_column(3, vec![x, y, z, 1.0]);
    identity_matrix
}

pub fn scaling(x: f64, y: f64, z: f64) -> Matrix<4,4> {
    let mut identity_matrix = Matrix::<4,4>::identity();
    identity_matrix[(0,0)] = x;
    identity_matrix[(1,1)] = y;
    identity_matrix[(2,2)] = z;
    identity_matrix
}

pub fn rotation_x(radians: f64) -> Matrix<4,4> {
    let mut identity_matrix = Matrix::<4,4>::identity();
    identity_matrix[(1,1)] = radians.cos();
    identity_matrix[(1,2)] = -radians.sin();
    identity_matrix[(2,1)] = radians.sin();
    identity_matrix[(2,2)] = radians.cos();
    identity_matrix
}

pub fn rotation_y(radians: f64) -> Matrix<4,4> {
    let mut identity_matrix = Matrix::<4,4>::identity();
    identity_matrix[(0,0)] = radians.cos();
    identity_matrix[(0,2)] = radians.sin();
    identity_matrix[(2,0)] = -radians.sin();
    identity_matrix[(2,2)] = radians.cos();
    identity_matrix
}

pub fn rotation_z(radians: f64) -> Matrix<4,4> {
    let mut identity_matrix = Matrix::<4,4>::identity();
    identity_matrix[(0,0)] = radians.cos();
    identity_matrix[(0,1)] = -radians.sin();
    identity_matrix[(1,0)] = radians.sin();
    identity_matrix[(1,1)] = radians.cos();
    identity_matrix
}

pub fn shearing(xy: f64, xz: f64, yx: f64, yz: f64, zx: f64, zy: f64) -> Matrix<4,4> {
    let mut identity_matrix = Matrix::<4,4>::identity();
    identity_matrix[(0,1)] = xy;
    identity_matrix[(0,2)] = xz;
    identity_matrix[(1,0)] = yx;
    identity_matrix[(1,2)] = yz;
    identity_matrix[(2,0)] = zx;
    identity_matrix[(2,1)] = zy;
    identity_matrix
}

pub struct Ray {
    pub origin: Tuple4D,
    pub direction: Tuple4D
}
impl Ray {
    pub fn new(origin: Tuple4D, direction: Tuple4D) -> Ray {
        Ray { origin, direction }
    }
}

pub fn position(ray: &Ray, t: f64) -> Tuple4D {
    ray.origin + ray.direction * t
}

pub fn transform(ray: Ray, m: &Matrix<4,4>) -> Ray {
    let mut output_ray = Ray::new(Tuple4D::point(0.0, 0.0, 0.0), Tuple4D::vector(0.0, 0.0, 0.0));
    output_ray.origin = *m * ray.origin;
    output_ray.direction = *m * ray.direction;
    output_ray
}

#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    pub id: u64,
    pub transform: Matrix::<4,4>,
}

impl PartialEq for Sphere {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub fn sphere() -> Sphere {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    Sphere { id, transform: Matrix::<4,4>::identity() }
}

pub fn set_transform(sphere: &mut Sphere, m: Matrix<4,4>) {
    sphere.transform = m;
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



pub fn intersect(sphere: &Sphere, ray: Ray) -> Vec<Intersection> {
    let ray2 = transform(ray, &inverse(&sphere.transform));
    let sphere_to_ray = ray2.origin - Tuple4D::point(0.0, 0.0, 0.0);
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

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;
    use std::ops::Index;
    use num_traits::ToPrimitive;
    use super::*;
    #[test]
    fn test_point() {
        let point = Tuple4D { x: 4.3, y: -4.2, z: 3.1, w: 1.0 };
        assert_eq!(point.x, 4.3);
        assert_eq!(point.y, -4.2);
        assert_eq!(point.z, 3.1);
        assert!(point.is_point());
        assert!(!point.is_vector());
    }

    #[test]
    fn test_vector() {
        let vector = Tuple4D { x: 4.3, y: -4.2, z: 3.1, w: 0.0};
        assert_eq!(vector.x, 4.3);
        assert_eq!(vector.y, -4.2);
        assert_eq!(vector.z, 3.1);
        assert!(vector.is_vector());
        assert!(!vector.is_point());
    }
    #[test]
    fn test_point_factory() {
        let _point = point(4.0,-4.0,3.0);
        assert!(_point.is_point());
        assert_eq!(_point, Tuple4D { x: 4.0, y: -4.0, z: 3.0, w: 1.0 });
    }

    #[test]
    fn test_vector_factory() {
        let _vector = vector(4.0,-4.0,4.0);
        assert!(_vector.is_vector());
    }

    #[test]
    fn test_equality_floats_logic() {
        let _vector = vector(4.0,-4.0,4.0);
        let _vector2 = vector(4.0 + 0.0000100001,-4.0,4.0);
        assert_ne!(_vector, _vector2);
    }

    #[test]
    fn test_adding_tuples() {
        let _t1 = Tuple4D{x:3.0, y:-2.0, z:5.0, w:1.0};
        let _t2 = Tuple4D{x:-2.0, y:3.0, z:1.0, w:0.0};
        assert_eq!(_t1 + _t2, Tuple4D { x: 1.0, y: 1.0, z: 6.0, w: 1.0 });
    }
    #[test]
    fn test_subtracting_two_points() {
        let _p1 = point(3.0, 2.0, 1.0);
        let _p2 = point(5.0, 6.0, 7.0);
        assert_eq!(_p1 - _p2, vector(-2.0, -4.0, -6.0))
    }
    #[test]
    fn test_subtracting_vector_from_point() {
        let _p1 = point(3.0, 2.0, 1.0);
        let _v1 = vector(5.0, 6.0, 7.0);
        assert_eq!(_p1 - _v1, point(-2.0, -4.0, -6.0))
    }

    #[test]
    fn test_subtracting_two_vectors () {
        let _v1 = vector(3.0, 2.0, 1.0);
        let _v2 = vector(5.0, 6.0, 7.0);
        assert_eq!(_v1 - _v2, vector(-2.0, -4.0, -6.0))
    }
    #[test]
    fn negating_tuple() {
        let _vector = Tuple4D{x: 1.0, y: -2.0, z: 3.0, w: -4.0};
        assert_eq!(-_vector, Tuple4D { x: -1.0, y: 2.0, z: -3.0, w: 4.0 })
    }
    #[test]
    fn multiplying_tuple_by_scalar() {
        let _tuple = Tuple4D{x: 1.0, y: -2.0, z: 3.0, w: -4.0};
        assert_eq!(_tuple * 3.5, Tuple4D { x: 3.5, y: -7.0, z: 10.5, w: -14.0 })
    }

    #[test]
    fn multiplying_tuple_by_fraction() {
        let _tuple = Tuple4D{x:1.0, y:-2.0, z:3.0, w:-4.0};
        assert_eq!(_tuple * 0.5, Tuple4D { x: 0.5, y: -1.0, z: 1.5, w: -2.0 })
    }

    #[test]
    fn dividing_tuple_by_scalar() {
        let _tuple = Tuple4D{x:1.0, y:-2.0, z:3.0, w:-4.0};
        assert_eq!(_tuple / 2, Tuple4D { x: 0.5, y: -1.0, z: 1.5, w: -2.0 })
    }
    #[test]
    fn magnitude_of_vector_x() {
        let _vector = vector(1.0,0.0, 0.0);
        assert_eq!(_vector.magnitude(), 1.0);
    }
    #[test]
    fn magnitude_of_vector_y() {
        let _vector = vector(0.0,1.0, 0.0);
        assert_eq!(_vector.magnitude(), 1.0);
    }
    #[test]
    fn magnitude_of_vector_z() {
        let _vector = vector(0.0,0.0, 1.0);
        assert_eq!(_vector.magnitude(), 1.0);
    }
    #[test]
    fn magnitude_of_positive_vector() {
        let _vector = vector(1.0,2.0, 3.0);
        assert_eq!(_vector.magnitude(), 14.0_f64.sqrt());
    }
    #[test]
    fn magnitude_of_negative_vector() {
        let _vector = vector(-1.0,-2.0, -3.0);
        assert_eq!(_vector.magnitude(), 14.0_f64.sqrt());
    }
    #[test]
    fn normalize_vector_x() {
        let _vector = vector(4.0,0.0, 0.0);
        assert_eq!(_vector.normalize(), vector(1.0, 0.0, 0.0));
    }
    #[test]
    fn normalize_vector() {
        let _vector = vector(1.0,2.0, 3.0);
        assert_eq!(_vector.normalize(), vector(1.0 / 14.0_f64.sqrt(), 2.0 / 14.0_f64.sqrt(), 3.0 / 14.0_f64.sqrt()));
    }
    #[test]
    fn magnitude_of_normalized_vector() {
        let _vector = vector(1.0,2.0, 3.0);
        assert_eq!(_vector.normalize().magnitude(), 1.0);
    }

    #[test]
    fn dot_product() {
        let _v1 = vector(1.0, 2.0, 3.0);
        let _v2 = vector(2.0, 3.0, 4.0);
        assert_eq!(dot(&_v1, &_v2), 20.0)
    }
    #[test]
    fn cross_product() {
        let _v1 = vector(1.0, 2.0, 3.0);
        let _v2 = vector(2.0, 3.0, 4.0);
        assert!(cross(&_v1,&_v2) == vector(-1.0, 2.0, -1.0)
            && cross(&_v2,&_v1) == vector(1.0, -2.0, 1.0)
        )
    }
    #[test]
    fn color_tuples(){
        let _color = color(-0.5, 0.4, 1.7);
        assert!(_color.red == -0.5 && _color.green == 0.4 && _color.blue == 1.7);
    }

    #[test]
    fn adding_colors() {
        let _color1 = color(0.9, 0.6, 0.75);
        let _color2 = color(0.7, 0.1, 0.25);
        assert_eq!(_color1 + _color2, color(1.6, 0.7, 1.0));
    }
    #[test]
    fn subtracting_colors() {
        let _color1 = color(0.9, 0.6, 0.75);
        let _color2 = color(0.7, 0.1, 0.25);
        assert_eq!(_color1 - _color2, color(0.2, 0.5, 0.5));
    }
    #[test]
    fn multiplying_color_by_scalar() {
        let _color = color(0.2, 0.3, 0.4);
        assert_eq!(_color * 2, color(0.4, 0.6, 0.8));
    }
    #[test]
    fn multiplying_colors() {
        let _color1 = color(1.0, 0.2, 0.4);
        let _color2 = color(0.9, 1.0, 0.1);
        assert_eq!(_color1 * _color2, color(0.9, 0.2, 0.04));
    }

    #[test]
    fn create_2x2_matrix() {
        let mut _matrix = Matrix::<2,2>::new();
        _matrix.insert_row(0, vec![-3.0, 5.0]);
        _matrix.insert_row(1, vec![1.0, -2.0]);
        assert_eq!(_matrix[(0,0)], -3.0);
        assert_eq!(_matrix[(0,1)], 5.0);
        assert_eq!(_matrix[(1,0)], 1.0);
        assert_eq!(_matrix[(1,1)], -2.0)
    }

    #[test]
    fn create_3x3_matrix() {
        let mut _matrix = Matrix::<3,3>::new();
        _matrix.insert_row(0, vec![-3.0, 5.0, 0.0]);
        _matrix.insert_row(1, vec![1.0, -2.0, -7.0]);
        _matrix.insert_row(2, vec![0.0, 1.0, 1.0]);
        assert_eq!(_matrix[(0,0)], -3.0);
        assert_eq!(_matrix[(1,1)], -2.0);
        assert_eq!(_matrix[(2,2)], 1.0)
    }
    #[test]
    fn matrix_equality_test() {
        let mut matrix_a = Matrix::<4,4>::new();
        let mut matrix_b = Matrix::<4,4>::new();
        matrix_a.insert_row(0, vec![1.0, 2.0, 3.0, 4.0]);
        matrix_a.insert_row(1, vec![5.0, 6.0, 7.0, 8.0]);
        matrix_a.insert_row(2, vec![9.0, 8.0, 7.0, 6.0]);
        matrix_a.insert_row(3, vec![5.0, 4.0, 3.0, 2.0]);
        matrix_b.insert_row(0, vec![1.0, 2.0, 3.0, 4.0]);
        matrix_b.insert_row(1, vec![5.0, 6.0, 7.0, 8.0]);
        matrix_b.insert_row(2, vec![9.0, 8.0, 7.0, 6.0]);
        matrix_b.insert_row(3, vec![5.0, 4.0, 3.0, 2.0]);
        assert_eq!(matrix_a, matrix_b);
    }
    #[test]
    fn matrix_inequality_test() {
        let mut matrix_a = Matrix::<4,4>::new();
        let mut matrix_b = Matrix::<4,4>::new();
        matrix_a.insert_row(0, vec![1.0, 2.0, 3.0, 4.0]);
        matrix_a.insert_row(1, vec![5.0, 6.0, 7.0, 8.0]);
        matrix_a.insert_row(2, vec![9.0, 8.0, 7.0, 6.0]);
        matrix_a.insert_row(3, vec![5.0, 4.0, 3.0, 2.0]);
        matrix_b.insert_row(0, vec![2.0, 3.0, 4.0, 5.0]);
        matrix_b.insert_row(1, vec![6.0, 7.0, 8.0, 9.0]);
        matrix_b.insert_row(2, vec![8.0, 7.0, 6.0, 5.0]);
        matrix_b.insert_row(3, vec![4.0, 3.0, 2.0, 1.0]);
        matrix_b.data[0][0] = 0.0;
        assert_ne!(matrix_a, matrix_b);
    }

    #[test]
    fn matrix_multiply_test() {
        let mut matrix_a = Matrix::<4,4>::new();
        let mut matrix_b = Matrix::<4,4>::new();
        let mut matrix_expected_result = Matrix::<4,4>::new();
        matrix_a.insert_row(0, vec![1.0, 2.0, 3.0, 4.0]);
        matrix_a.insert_row(1, vec![5.0, 6.0, 7.0, 8.0]);
        matrix_a.insert_row(2, vec![9.0, 8.0, 7.0, 6.0]);
        matrix_a.insert_row(3, vec![5.0, 4.0, 3.0, 2.0]);
        matrix_b.insert_row(0, vec![-2.0, 1.0, 2.0, 3.0]);
        matrix_b.insert_row(1, vec![3.0, 2.0, 1.0, -1.0]);
        matrix_b.insert_row(2, vec![4.0, 3.0, 6.0, 5.0]);
        matrix_b.insert_row(3, vec![1.0, 2.0, 7.0, 8.0]);
        matrix_expected_result.insert_row(0, vec![20.0, 22.0, 50.0, 48.0]);
        matrix_expected_result.insert_row(1, vec![44.0, 54.0, 114.0, 108.0]);
        matrix_expected_result.insert_row(2, vec![40.0, 58.0, 110.0, 102.0]);
        matrix_expected_result.insert_row(3, vec![16.0, 26.0, 46.0, 42.0]);
        assert_eq!(matrix_a * matrix_b, matrix_expected_result);
    }

    #[test]
    fn matrix_multiply_by_tuple_test() {
        let mut matrix = Matrix::<4,4>::new();
        matrix.insert_row(0, vec![1.0, 2.0, 3.0, 4.0]);
        matrix.insert_row(1, vec![2.0, 4.0, 4.0, 2.0]);
        matrix.insert_row(2, vec![8.0, 6.0, 4.0, 1.0]);
        matrix.insert_row(3, vec![0.0, 0.0, 0.0, 1.0]);
        let tuple = Tuple4D{x:1.0, y:2.0, z:3.0, w:1.0};
        let expected_tuple = Tuple4D{x:18.0, y:24.0, z:33.0, w:1.0};
        assert_eq!(matrix * tuple, expected_tuple);
    }

    #[test]
    fn matrix_multiply_by_identity_matrix_test() {
        let mut matrix = Matrix::<4,4>::new();
        matrix.insert_row(0, vec![0.0, 1.0, 2.0, 4.0]);
        matrix.insert_row(1, vec![1.0, 2.0, 4.0, 8.0]);
        matrix.insert_row(2, vec![2.0, 4.0, 8.0, 16.0]);
        matrix.insert_row(3, vec![4.0, 8.0, 16.0, 32.0]);
        let identity_matrix = Matrix::<4,4>::identity();
        assert_eq!(matrix * identity_matrix, matrix);
    }
    #[test]
    fn multiplying_identity_matrix_by_tuple_test() {
        let tuple = Tuple4D{x:1.0, y:2.0, z:3.0, w:4.0};
        let identity_matrix = Matrix::<4,4>::identity();
        assert_eq!(identity_matrix * tuple, tuple);
    }
    #[test]
    fn transposing_matrix() {
        let mut matrix_a = Matrix::<4,4>::new();
        let mut matrix_transposed = Matrix::<4,4>::new();
        matrix_a.insert_row(0, vec![0.0, 9.0, 3.0, 0.0]);
        matrix_a.insert_row(1, vec![9.0, 8.0, 0.0, 8.0]);
        matrix_a.insert_row(2, vec![1.0, 8.0, 5.0, 3.0]);
        matrix_a.insert_row(3, vec![0.0, 0.0, 5.0, 8.0]);
        matrix_transposed.insert_row(0, vec![0.0, 9.0, 1.0, 0.0]);
        matrix_transposed.insert_row(1, vec![9.0, 8.0, 8.0, 0.0]);
        matrix_transposed.insert_row(2, vec![3.0, 0.0, 5.0, 5.0]);
        matrix_transposed.insert_row(3, vec![0.0, 8.0, 3.0, 8.0]);
        assert_eq!(transpose(matrix_a), matrix_transposed);
    }
    #[test]
    fn transposing_identity_matrix() {
        let identity_matrix = Matrix::<4,4>::identity();
        assert_eq!(transpose(identity_matrix), identity_matrix);
    }
    #[test]
    fn determinant_2x2_matrix() {
        let mut matrix = Matrix::<2,2>::new();
        matrix.insert_row(0, vec![1.0, 5.0]);
        matrix.insert_row(1, vec![-3.0, 2.0]);
        assert_eq!(determinant(&matrix), 17.0);
    }
    #[test]
    fn submatrix_3x3_matrix() {
        let mut matrix = Matrix::<3,3>::new();
        matrix.insert_row(0, vec![1.0, 5.0, 0.0]);
        matrix.insert_row(1, vec![-3.0, 2.0, 7.0]);
        matrix.insert_row(2, vec![0.0, 6.0, -3.0]);
        let mut expected_matrix = Matrix::<2,2>::new();
        expected_matrix.insert_row(0, vec![-3.0, 2.0]);
        expected_matrix.insert_row(1, vec![0.0, 6.0]);
        assert_eq!(submatrix(&matrix, 0, 2), expected_matrix);
    }
    #[test]
    fn submatrix_4x4_matrix() {
        let mut matrix = Matrix::<4,4>::new();
        matrix.insert_row(0, vec![-6.0, 1.0, 1.0, 6.0]);
        matrix.insert_row(1, vec![-8.0, 5.0, 8.0, 6.0]);
        matrix.insert_row(2, vec![-1.0, 0.0, 8.0, 2.0]);
        matrix.insert_row(3, vec![-7.0, 1.0, -1.0, 1.0]);
        let mut expected_matrix = Matrix::<3,3>::new();
        expected_matrix.insert_row(0, vec![-6.0, 1.0, 6.0]);
        expected_matrix.insert_row(1, vec![-8.0, 8.0, 6.0]);
        expected_matrix.insert_row(2, vec![-7.0, -1.0, 1.0]);
        assert_eq!(submatrix(&matrix, 2, 1), expected_matrix)
    }

    #[test]
    fn calculate_minor_3x3_matrix() {
        let mut matrix_a = Matrix::<3,3>::new();
        matrix_a.insert_row(0, vec![3.0, 5.0, 0.0]);
        matrix_a.insert_row(1, vec![2.0, -1.0, -7.0]);
        matrix_a.insert_row(2, vec![6.0, -1.0, 5.0]);
        let matrix_b: Matrix::<2,2> = submatrix(&matrix_a, 1, 0);
        assert_eq!(determinant(&matrix_b), 25.0);
        assert_eq!(determinant(&matrix_b), minor(&matrix_a, 1, 0))
    }

    #[test]
    fn calculate_cofactor_3x3_matrix() {
        let mut matrix = Matrix::<3,3>::new();
        matrix.insert_row(0, vec![3.0, 5.0, 0.0]);
        matrix.insert_row(1, vec![2.0, -1.0, -7.0]);
        matrix.insert_row(2, vec![6.0, -1.0, 5.0]);
        assert_eq!(minor(&matrix, 0, 0), -12.0);
        assert_eq!(cofactor(&matrix, 0, 0), -12.0);
        assert_eq!(minor(&matrix, 1, 0), 25.0);
        assert_eq!(cofactor(&matrix, 1, 0), -25.0);
    }
    #[test]
    fn calculate_determinant_3x3_matrix() {
        let mut matrix = Matrix::<3,3>::new();
        matrix.insert_row(0, vec![1.0, 2.0, 6.0]);
        matrix.insert_row(1, vec![-5.0, 8.0, -4.0]);
        matrix.insert_row(2, vec![2.0, 6.0, 4.0]);
        assert_eq!(cofactor(&matrix, 0, 0), 56.0);
        assert_eq!(cofactor(&matrix, 0, 1), 12.0);
        assert_eq!(cofactor(&matrix, 0, 2), -46.0);
        assert_eq!(determinant(&matrix), -196.0)
    }
    #[test]
    fn calculate_determinant_4x4_matrix() {
        let mut matrix = Matrix::<4,4>::new();
        matrix.insert_row(0, vec![-2.0, -8.0, 3.0, 5.0]);
        matrix.insert_row(1, vec![-3.0, 1.0, 7.0, 3.0]);
        matrix.insert_row(2, vec![1.0, 2.0, -9.0, 6.0]);
        matrix.insert_row(3, vec![-6.0, 7.0, 7.0, -9.0]);
        assert_eq!(cofactor(&matrix, 0, 0), 690.0);
        assert_eq!(cofactor(&matrix, 0, 1), 447.0);
        assert_eq!(cofactor(&matrix, 0, 2), 210.0);
        assert_eq!(cofactor(&matrix, 0, 3), 51.0);
        assert_eq!(determinant(&matrix), -4071.0);
    }
    #[test]
    fn test_invertability_of_invertable_matrix()
    {
        let mut matrix = Matrix::<4,4>::new();
        matrix.insert_row(0, vec![6.0, 4.0, 4.0, 4.0]);
        matrix.insert_row(1, vec![5.0, 5.0, 7.0, 6.0]);
        matrix.insert_row(2, vec![4.0, -9.0, 3.0, -7.0]);
        matrix.insert_row(3, vec![9.0, 1.0, 7.0, -6.0]);
        assert_eq!(determinant(&matrix), -2120.0);
    }

    #[test]
    fn test_invertability_of_non_invertable_matrix()
    {
        let mut matrix = Matrix::<4,4>::new();
        matrix.insert_row(0, vec![-4.0, 2.0, -2.0, -3.0]);
        matrix.insert_row(1, vec![9.0, 6.0, 2.0, 6.0]);
        matrix.insert_row(2, vec![0.0, -5.0, 1.0, -5.0]);
        matrix.insert_row(3, vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(determinant(&matrix), 0.0); //non invertable since det(matrix) is 0
    }
    #[test]
    fn test_inverting_matrix()
    {
        let mut matrix = Matrix::<4,4>::new();
        matrix.insert_row(0, vec![-5.0, 2.0, 6.0, -8.0]);
        matrix.insert_row(1, vec![1.0, -5.0, 1.0, 8.0]);
        matrix.insert_row(2, vec![7.0, 7.0, -6.0, -7.0]);
        matrix.insert_row(3, vec![1.0, -3.0, 7.0, 4.0]);
        let matrix_inverse = inverse(&matrix);
        assert_eq!(determinant(&matrix), 532.0);
        assert_eq!(cofactor(&matrix, 2, 3), -160.0);
        assert_eq!(matrix_inverse[(3,2)], -160.0/532.0);
        assert_eq!(cofactor(&matrix, 3, 2), 105.0);
        assert_eq!(matrix_inverse[(2,3)], 105.0/532.0);
        let mut expected_matrix = Matrix::<4,4>::new();
        expected_matrix.insert_row(0, vec![0.21805, 0.45113, 0.24060, -0.04511]);
        expected_matrix.insert_row(1, vec![-0.80827, -1.45677, -0.44361, 0.52068]);
        expected_matrix.insert_row(2, vec![-0.07895, -0.22368, -0.05263, 0.19737]);
        expected_matrix.insert_row(3, vec![-0.52256, -0.81391, -0.30075, 0.30639]);
        assert_eq!(matrix_inverse, expected_matrix);
    }

    #[test]
    fn test_maxtrix_inversion_1() {
        let mut matrix = Matrix::<4,4>::new();
        matrix.insert_row(0, vec![8.0, -5.0, 9.0, 2.0]);
        matrix.insert_row(1, vec![7.0, 5.0, 6.0, 1.0]);
        matrix.insert_row(2, vec![-6.0, 0.0, 9.0, 6.0]);
        matrix.insert_row(3, vec![-3.0, 0.0, -9.0, -4.0]);
        let matrix_inverse = inverse(&matrix);
        let mut expected_matrix = Matrix::<4,4>::new();
        expected_matrix.insert_row(0, vec![-0.15385, -0.15385, -0.28205, -0.53846]);
        expected_matrix.insert_row(1, vec![-0.07692, 0.12308, 0.02564, 0.03077]);
        expected_matrix.insert_row(2, vec![0.35897, 0.35897, 0.43590, 0.92308]);
        expected_matrix.insert_row(3, vec![-0.69231, -0.69231, -0.76923, -1.92308]);
        assert_eq!(matrix_inverse, expected_matrix);
    }

    #[test]
    fn test_matrix_inversion_2() {
        let mut matrix = Matrix::<4,4>::new();
        matrix.insert_row(0, vec![9.0, 3.0, 0.0, 9.0]);
        matrix.insert_row(1, vec![-5.0, -2.0, -6.0, -3.0]);
        matrix.insert_row(2, vec![-4.0, 9.0, 6.0, 4.0]);
        matrix.insert_row(3, vec![-7.0, 6.0, 6.0, 2.0]);
        let matrix_inverse = inverse(&matrix);
        let mut expected_matrix = Matrix::<4,4>::new();
        expected_matrix.insert_row(0, vec![-0.04074, -0.07778, 0.14444, -0.22222]);
        expected_matrix.insert_row(1, vec![-0.07778, 0.03333, 0.36667, -0.33333]);
        expected_matrix.insert_row(2, vec![-0.02901, -0.14630, -0.10926, 0.12963]);
        expected_matrix.insert_row(3, vec![0.17778, 0.06667, -0.26667, 0.33333]);
        assert_eq!(matrix_inverse, expected_matrix);
    }

    #[test]
    fn test_multiply_product_by_inverse()
    {
        let mut matrix_a = Matrix::<4,4>::new();
        matrix_a.insert_row(0, vec![3.0, -9.0, 7.0, 3.0]);
        matrix_a.insert_row(1, vec![3.0, -8.0, 2.0, -9.0]);
        matrix_a.insert_row(2, vec![-4.0, 4.0, 4.0, 1.0]);
        matrix_a.insert_row(3, vec![-6.0, 5.0, -1.0, 1.0]);
        let mut matrix_b = Matrix::<4,4>::new();
        matrix_b.insert_row(0, vec![8.0, 2.0, 2.0, 2.0]);
        matrix_b.insert_row(1, vec![3.0, -1.0, 7.0, 0.0]);
        matrix_b.insert_row(2, vec![7.0, 0.0, 5.0, 4.0]);
        matrix_b.insert_row(3, vec![6.0, -2.0, 0.0, 5.0]);
        let matrix_c = matrix_a * matrix_b;
        assert_eq!(matrix_c * inverse(&matrix_b), matrix_a);
    }

    #[test]
    fn test_multiply_by_tanslation_matrix() {
        let transform_matrix = translation(5.0, -3.0, 2.0);
        let point = Tuple4D::point(-3.0, 4.0, 5.0);
        assert_eq!(transform_matrix * point, Tuple4D::point(2.0, 1.0, 7.0));
    }
    #[test]
    fn test_multiply_by_inv_translation_matrix() {
        let transform_matrix = translation(5.0, -3.0, 2.0);
        let inv = inverse(&transform_matrix);
        let point = Tuple4D::point(-3.0, 4.0, 5.0);
        assert_eq!(inv * point, Tuple4D::point(-8.0, 7.0, 3.0));
    }
    #[test]
    fn test_translation_not_affect_vectors() {
        let transform_matrix = translation(5.0, -3.0, 2.0);
        let vector = Tuple4D::vector(-3.0, 4.0, 5.0);
        assert_eq!(transform_matrix * vector, vector);
    }
    #[test]
    fn test_scaling_matrix_on_point() {
        let transform_matrix = scaling(2.0, 3.0, 4.0);
        let point = Tuple4D::point(-4.0, 6.0, 8.0);
        assert_eq!(transform_matrix * point, Tuple4D::point(-8.0, 18.0, 32.0));
    }
    #[test]
    fn test_scaling_matrix_on_vector() {
        let transform_matrix = scaling(2.0, 3.0, 4.0);
        let vector = Tuple4D::vector(-4.0, 6.0, 8.0);
        assert_eq!(transform_matrix * vector, Tuple4D::vector(-8.0, 18.0, 32.0));
    }
    #[test]
    fn test_inverse_scaling_matrix() {
        let transform_matrix = scaling(2.0, 3.0, 4.0);
        let inv_transform_matrix = inverse(&transform_matrix);
        let vector = Tuple4D::vector(-4.0, 6.0, 8.0);
        assert_eq!(inv_transform_matrix * vector, Tuple4D::vector(-2.0, 2.0, 2.0));
    }
    #[test]
    fn test_reflection_across_x_by_scaling_matrix() {
        let transform_matrix = scaling(-1.0, 1.0, 1.0);
        let point = Tuple4D::point(2.0, 3.0, 4.0);
        assert_eq!(transform_matrix * point, Tuple4D::point(-2.0, 3.0, 4.0));
    }
    #[test]
    fn test_rotation_around_x() {
        let point = Tuple4D::point(0.0, 1.0, 0.0);
        let half_quarter_rot_matrix = rotation_x(PI/ 4.0);
        let quarter_rot_matrix = rotation_x(PI / 2.0);
        assert_eq!(half_quarter_rot_matrix * point, Tuple4D::point(0.0, f64::sqrt(2.0)/2.0, f64::sqrt(2.0)/2.0));
        assert_eq!(quarter_rot_matrix * point, Tuple4D::point(0.0, 0.0, 1.0));

    }
    #[test]
    fn test_inverse_rotation_around_x() {
        let point = Tuple4D::point(0.0, 1.0, 0.0);
        let half_quarter_rot_matrix = rotation_x(PI/ 4.0);
        let inv_half_quarter_rot_matrix = inverse(&half_quarter_rot_matrix);
        assert_eq!(inv_half_quarter_rot_matrix * point, Tuple4D::point(0.0, f64::sqrt(2.0)/2.0, -f64::sqrt(2.0)/2.0));
    }
    #[test]
    fn test_rotation_around_y() {
        let point = Tuple4D::point(0.0, 0.0, 1.0);
        let half_quarter_rot_matrix = rotation_y(PI/4.0);
        let quarter_rot_matrix = rotation_y(PI/2.0);
        assert_eq!(half_quarter_rot_matrix * point, Tuple4D::point(f64::sqrt(2.0)/2.0, 0.0, f64::sqrt(2.0)/2.0));
        assert_eq!(quarter_rot_matrix * point, Tuple4D::point(1.0, 0.0, 0.0));
    }
    #[test]
    fn test_rotation_around_z() {
        let point = Tuple4D::point(0.0, 1.0, 0.0);
        let half_quarter_rot_matrix = rotation_z(PI/4.0);
        let quarter_rot_matrix = rotation_z(PI/2.0);
        assert_eq!(half_quarter_rot_matrix * point, Tuple4D::point(- f64::sqrt(2.0)/2.0, f64::sqrt(2.0)/2.0, 0.0));
        assert_eq!(quarter_rot_matrix * point, Tuple4D::point(-1.0, 0.0, 0.0));
    }
    #[test]
    fn test_shearing_matrix_x_in_prop_y() {
        let transform_matrix = shearing(0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let point = Tuple4D::point(2.0, 3.0, 4.0);
        assert_eq!(transform_matrix * point, Tuple4D::point(6.0, 3.0, 4.0));
    }
    #[test]
    fn test_shearing_matrix_y_in_prop_x() {
        let transform_matrix = shearing(0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
        let point = Tuple4D::point(2.0, 3.0, 4.0);
        assert_eq!(transform_matrix * point, Tuple4D::point(2.0, 5.0, 4.0));
    }

    #[test]
    fn test_shearing_matrix_y_in_prop_() {
        let transform_matrix = shearing(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        let point = Tuple4D::point(2.0, 3.0, 4.0);
        assert_eq!(transform_matrix * point, Tuple4D::point(2.0, 7.0, 4.0));
    }

    #[test]
    fn test_shearing_matrix_z_in_prop_x() {
        let transform_matrix = shearing(0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        let point = Tuple4D::point(2.0, 3.0, 4.0);
        assert_eq!(transform_matrix * point, Tuple4D::point(2.0, 3.0, 6.0));
    }
    #[test]
    fn test_shearing_matrix_z_in_prop_y() {
        let transform_matrix = shearing(0.0, 0.0, 0.0, 0.0, 0.0, 1.0 );
        let point = Tuple4D::point(2.0, 3.0, 4.0);
        assert_eq!(transform_matrix * point, Tuple4D::point(2.0, 3.0, 7.0));
    }
    #[test]
    fn test_transformations_in_sequence() {
        let point = Tuple4D::point(1.0, 0.0, 1.0);
        let a = rotation_x(PI/2.0);
        let b = scaling(5.0, 5.0, 5.0);
        let c = translation(10.0, 5.0, 7.0);
        let point_2 = a * point;
        assert_eq!(point_2, Tuple4D::point(1.0, -1.0, 0.0));
        let point_3 = b * point_2;
        assert_eq!(point_3, Tuple4D::point(5.0, -5.0, 0.0));
        let point_4 = c * point_3;
        assert_eq!(point_4, Tuple4D::point(15.0, 0.0, 7.0));
    }
    #[test]
    fn test_transformations_chained() {
        let point = Tuple4D::point(1.0, 0.0, 1.0);
        let a = rotation_x(PI/2.0);
        let b = scaling(5.0, 5.0, 5.0);
        let c = translation(10.0, 5.0, 7.0);
        let transformation_matrix = c * b * a;
        assert_eq!(transformation_matrix * point, Tuple4D::point(15.0, 0.0, 7.0));
    }

    #[test]
    fn test_create_and_query_ray() {
        let origin = Tuple4D::point(1.0, 2.0, 3.0);
        let direction = Tuple4D::vector(4.0, 5.0, 6.0);
        let r = Ray::new(origin, direction);
        assert_eq!(r.origin, origin);
        assert_eq!(r.direction, direction);
    }

    #[test]
    fn test_compute_point_along_ray() {
        let r = Ray::new(Tuple4D::point(2.0, 3.0, 4.0), Tuple4D::vector(1.0, 0.0, 0.0));
        assert_eq!(position(&r, 0.0), Tuple4D::point(2.0, 3.0, 4.0));
        assert_eq!(position(&r, 1.0), Tuple4D::point(3.0, 3.0, 4.0));
        assert_eq!(position(&r, -1.0), Tuple4D::point(1.0, 3.0, 4.0));
        assert_eq!(position(&r, 2.5), Tuple4D::point(4.5, 3.0, 4.0));
    }
    #[test]
    fn test_ray_intersecting_sphere() {
        let ray = Ray::new(Tuple4D::point(0.0, 0.0, -5.0),Tuple4D::vector(0.0, 0.0, 1.0));
        let sphere = sphere();
        let sphere_intersection = intersect(&sphere, ray);
        assert_eq!(sphere_intersection.len(), 2);
        assert_eq!(sphere_intersection[0].t, 4.0);
        assert_eq!(sphere_intersection[1].t, 6.0);
    }
    #[test]
    fn test_ray_intersect_tangent() {
        let ray = Ray::new(Tuple4D::point(0.0, 1.0, -5.0),Tuple4D::vector(0.0, 0.0, 1.0));
        let sphere = sphere();
        let sphere_intersection = intersect(&sphere, ray);
        assert_eq!(sphere_intersection.len(), 2);
        assert_eq!(sphere_intersection[0].t, 5.0);
        assert_eq!(sphere_intersection[1].t, 5.0);
    }
    #[test]
    fn test_ray_misses_sphere() {
        let ray = Ray::new(Tuple4D::point(0.0, 2.0, -5.0), Tuple4D::vector(0.0, 0.0, 1.0));
        let sphere = sphere();
        let sphere_intersection = intersect(&sphere, ray);
        assert_eq!(sphere_intersection.len(), 0);
    }
    #[test]
    fn test_ray_originates_inside_sphere() {
        let ray = Ray::new(Tuple4D::point(0.0, 0.0, 0.0), Tuple4D::vector(0.0, 0.0, 1.0));
        let sphere = sphere();
        let sphere_intersection = intersect(&sphere, ray);
        assert_eq!(sphere_intersection.len(), 2);
        assert_eq!(sphere_intersection[0].t, -1.0);
        assert_eq!(sphere_intersection[1].t, 1.0);
    }
    #[test]
    fn test_sphere_is_behind_ray() {
        let ray = Ray::new(Tuple4D::point(0.0, 0.0, 5.0), Tuple4D::vector(0.0, 0.0, 1.0));
        let sphere = sphere();
        let sphere_intersection = intersect(&sphere, ray);
        assert_eq!(sphere_intersection.len(), 2);
        assert_eq!(sphere_intersection[0].t, -6.0);
        assert_eq!(sphere_intersection[1].t, -4.0);
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
        let ray = Ray::new(Tuple4D::point(0.0, 0.0, -5.0), Tuple4D::vector(0.0, 0.0, 1.0));
        let sphere = sphere();
        let xs = intersect(&sphere, ray);
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
    #[test]
    fn test_translating_a_ray() {
        let original_ray = Ray::new(Tuple4D::point(1.0, 2.0, 3.0), Tuple4D::vector(0.0, 1.0, 0.0));
        let translation_matrix = translation(3.0, 4.0, 5.0);
        let translated_ray = transform(original_ray, &translation_matrix);
        assert_eq!(translated_ray.origin, Tuple4D::point(4.0, 6.0, 8.0));
        assert_eq!(translated_ray.direction, Tuple4D::vector(0.0, 1.0, 0.0));
    }
    #[test]
    fn test_scaling_ray() {
        let original_ray = Ray::new(Tuple4D::point(1.0, 2.0, 3.0), Tuple4D::vector(0.0, 1.0, 0.0));
        let scaling_matrix = scaling(2.0, 3.0, 4.0);
        let scaled_ray = transform(original_ray, &scaling_matrix);
        assert_eq!(scaled_ray.origin, Tuple4D::point(2.0, 6.0, 12.0));
        assert_eq!(scaled_ray.direction, Tuple4D::vector(0.0, 3.0, 0.0));
    }
    #[test]
    fn test_sphere_default_transformation() {
        let s = sphere();
        assert_eq!(s.transform, Matrix::identity());
    }

    #[test]
    fn test_changing_sphere_transformation() {
        let mut sphere = sphere();
        let translation_matrix = translation(2.0, 3.0, 4.0);
        set_transform(&mut sphere, translation_matrix);
        sphere.transform = translation_matrix;
        assert_eq!(sphere.transform, translation_matrix);
    }
    #[test]
    fn test_intersect_scaled_sphere_with_ray() {
        let ray = Ray::new(Tuple4D::point(0.0, 0.0, -5.0), Tuple4D::vector(0.0, 0.0, 1.0));
        let mut sphere = sphere();
        set_transform(&mut sphere, scaling(2.0, 2.0, 2.0));
        let xs = intersect(&sphere, ray);
        assert_eq!(xs.len(), 2);
        assert_eq!(xs[0].t, 3.0);
        assert_eq!(xs[1].t, 7.0);
    }
    #[test]
    fn test_intersect_translated_sphere_with_ray() {
        let ray = Ray::new(Tuple4D::point(0.0, 0.0, -5.0), Tuple4D::vector(0.0, 0.0, 1.0));
        let mut sphere = sphere();
        set_transform(&mut sphere, translation(5.0, 0.0, 0.0));
        let xs = intersect(&sphere, ray);
        assert_eq!(xs.len(), 0);
    }
}