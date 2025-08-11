
const EPSILON: f64 = 0.00001;
#[derive(Copy, Clone)]
pub struct Tuple4D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64
}

impl Tuple4D {

    pub fn point(x: f64, y: f64, z: f64) -> Tuple4D {
        Self {x, y, z, w: 1.0}
    }
    pub fn vector(x: f64, y: f64, z: f64) -> Tuple4D {
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

#[cfg(test)]
mod tests {
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
        let _point = Tuple4D::point(4.0,-4.0,3.0);
        assert!(_point.is_point());
        assert!(_point == Tuple4D{x:4.0,y:-4.0,z:3.0,w:1.0});
    }

    #[test]
    fn test_vector_factory() {
        let _vector = Tuple4D::vector(4.0,-4.0,4.0);
        assert!(_vector.is_vector());
    }

    #[test]
    fn test_equality_floats_logic() {
        let _vector = Tuple4D::vector(4.0,-4.0,4.0);
        let _vector2 = Tuple4D::vector(4.0 + 0.0000100001,-4.0,4.0);
        assert!(_vector != _vector2);
    }

    #[test]
    fn test_adding_tuples() {
        let _t1 = Tuple4D{x:3.0, y:-2.0, z:5.0, w:1.0};
        let _t2 = Tuple4D{x:-2.0, y:3.0, z:1.0, w:0.0};
        assert!(_t1 + _t2 == Tuple4D{x:1.0, y:1.0, z:6.0, w:1.0});
    }
    #[test]
    fn test_subtracting_two_points() {
        let _p1 = Tuple4D::point(3.0, 2.0, 1.0);
        let _p2 = Tuple4D::point(5.0, 6.0, 7.0);
        assert!(_p1-_p2 == Tuple4D::vector(-2.0, -4.0, -6.0))
    }
    #[test]
    fn test_subtracting_vector_from_point() {
        let _p1 = Tuple4D::point(3.0, 2.0, 1.0);
        let _v1 = Tuple4D::vector(5.0, 6.0, 7.0);
        assert!(_p1-_v1 == Tuple4D::point(-2.0, -4.0, -6.0))
    }

    #[test]
    fn test_subtracting_two_vectors () {
        let _v1 = Tuple4D::vector(3.0, 2.0, 1.0);
        let _v2 = Tuple4D::vector(5.0, 6.0, 7.0);
        assert!(_v1-_v2 == Tuple4D::vector(-2.0, -4.0, -6.0))
    }
    #[test]
    fn negating_tuple() {
        let _vector = Tuple4D{x: 1.0, y: -2.0, z: 3.0, w: -4.0};
        assert!(-_vector == Tuple4D{x: -1.0, y: 2.0, z: -3.0, w: 4.0})
    }
    #[test]
    fn multiplying_tuple_by_scalar() {
        let _tuple = Tuple4D{x: 1.0, y: -2.0, z: 3.0, w: -4.0};
        assert!(_tuple * 3.5 == Tuple4D{x: 3.5, y:-7.0, z:10.5, w:-14.0})
    }

    #[test]
    fn multiplying_tuple_by_fraction() {
        let _tuple = Tuple4D{x:1.0, y:-2.0, z:3.0, w:-4.0};
        assert!(_tuple * 0.5 == Tuple4D{x:0.5, y:-1.0, z:1.5, w:-2.0})
    }

    #[test]
    fn dividing_tuple_by_scalar() {
        let _tuple = Tuple4D{x:1.0, y:-2.0, z:3.0, w:-4.0};
        assert!(_tuple / 2 == Tuple4D{x:0.5, y:-1.0, z:1.5, w:-2.0})
    }
    #[test]
    fn magnitude_of_vector_x() {
        let _vector = Tuple4D::vector(1.0,0.0, 0.0);
        assert_eq!(_vector.magnitude(), 1.0);
    }
    #[test]
    fn magnitude_of_vector_y() {
        let _vector = Tuple4D::vector(0.0,1.0, 0.0);
        assert_eq!(_vector.magnitude(), 1.0);
    }
    #[test]
    fn magnitude_of_vector_z() {
        let _vector = Tuple4D::vector(0.0,0.0, 1.0);
        assert_eq!(_vector.magnitude(), 1.0);
    }
    #[test]
    fn magnitude_of_positive_vector() {
        let _vector = Tuple4D::vector(1.0,2.0, 3.0);
        assert_eq!(_vector.magnitude(), 14.0_f64.sqrt());
    }
    #[test]
    fn magnitude_of_negative_vector() {
        let _vector = Tuple4D::vector(-1.0,-2.0, -3.0);
        assert_eq!(_vector.magnitude(), 14.0_f64.sqrt());
    }
    #[test]
    fn normalize_vector_x() {
        let _vector = Tuple4D::vector(4.0,0.0, 0.0);
        assert!(_vector.normalize() == Tuple4D::vector(1.0, 0.0, 0.0));
    }
    #[test]
    fn normalize_vector() {
        let _vector = Tuple4D::vector(1.0,2.0, 3.0);
        assert!(_vector.normalize() == Tuple4D::vector(1.0/14.0_f64.sqrt(),2.0/14.0_f64.sqrt(),3.0/14.0_f64.sqrt()));
    }
    #[test]
    fn magnitude_of_normalized_vector() {
        let _vector = Tuple4D::vector(1.0,2.0, 3.0);
        assert_eq!(_vector.normalize().magnitude(), 1.0);
    }

    #[test]
    fn dot_product() {
        let _v1 = Tuple4D::vector(1.0, 2.0, 3.0);
        let _v2 = Tuple4D::vector(2.0, 3.0, 4.0);
        assert_eq!(Tuple4D::dot(&_v1, &_v2), 20.0)
    }
    fn cross_product() {
        let _v1 = Tuple4D::vector(1.0, 2.0, 3.0);
        let _v2 = Tuple4D::vector(2.0, 3.0, 4.0);
        assert!(Tuple4D::cross(&_v1,&_v2) == Tuple4D::vector(-1.0, 2.0, -1.0)
            && Tuple4D::cross(&_v2,&_v1) == Tuple4D::vector(1.0, -2.0, 1.0)
        )
    }


}


