//! Small vector helpers shared by all models.

pub type Vec3 = nalgebra::Vector3<f64>;

/// Component of `v` perpendicular to the unit vector `b`.
#[inline]
pub fn perp(v: &Vec3, b: &Vec3) -> Vec3 {
    v - b * v.dot(b)
}

/// `v` scaled down so that `|v| <= n` (unchanged if already inside).
#[inline]
pub fn clip_norm(v: &Vec3, n: f64) -> Vec3 {
    let vn = v.norm();
    if vn <= n { *v } else { v * (n / vn) }
}

/// Rotate the unit vector `b` by angular velocity `w` over `dt` (Rodrigues), re-normalised.
pub fn rotate(b: &Vec3, w: &Vec3, dt: f64) -> Vec3 {
    let wn = w.norm();
    let ang = wn * dt;
    if ang < 1e-12 {
        return *b;
    }
    let k = w / wn;
    let (s, c) = ang.sin_cos();
    let out = b * c + k.cross(b) * s + k * k.dot(b) * (1.0 - c);
    out / out.norm()
}

/// Piecewise-linear interpolation with end clamping (same semantics as `numpy.interp`).
/// `xp` must be increasing.
pub fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    debug_assert_eq!(xp.len(), fp.len());
    let n = xp.len();
    if n == 0 {
        return 0.0;
    }
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[n - 1] {
        return fp[n - 1];
    }
    // index of the last xp <= x
    let j = xp.partition_point(|&v| v <= x) - 1;
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    slope * (x - xp[j]) + fp[j]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rotate_quarter_turn() {
        let b = Vec3::new(1.0, 0.0, 0.0);
        let w = Vec3::new(0.0, 0.0, std::f64::consts::FRAC_PI_2);
        let out = rotate(&b, &w, 1.0);
        assert_relative_eq!(out, Vec3::new(0.0, 1.0, 0.0), epsilon = 1e-12);
    }

    #[test]
    fn clip_and_perp() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert_relative_eq!(clip_norm(&v, 1.0).norm(), 1.0, epsilon = 1e-12);
        assert_eq!(clip_norm(&v, 10.0), v);
        assert_relative_eq!(perp(&v, &Vec3::x()), Vec3::new(0.0, 4.0, 0.0));
    }

    #[test]
    fn interp_clamps_and_interpolates() {
        let xp = [0.0, 1.0, 2.0];
        let fp = [0.0, 10.0, 0.0];
        assert_eq!(interp(-1.0, &xp, &fp), 0.0);
        assert_eq!(interp(0.5, &xp, &fp), 5.0);
        assert_eq!(interp(1.5, &xp, &fp), 5.0);
        assert_eq!(interp(3.0, &xp, &fp), 0.0);
    }
}
