//! Event detection helpers.

/// Fraction `s in [0, 1]` of a step at which a scalar guard `g` crosses zero going from
/// `g0 > 0` to `g1 <= 0`, by linear interpolation. Returns `None` if there is no crossing.
pub fn zero_crossing(g0: f64, g1: f64) -> Option<f64> {
    if g1 > 0.0 {
        return None;
    }
    Some(if g0 > 0.0 { g0 / (g0 - g1) } else { 0.0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crossing() {
        assert_eq!(zero_crossing(1.0, 0.5), None);
        assert_eq!(zero_crossing(1.0, -1.0), Some(0.5));
        assert_eq!(zero_crossing(-1.0, -2.0), Some(0.0));
    }
}
