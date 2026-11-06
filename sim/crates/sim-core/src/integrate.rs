//! Fixed-step ODE integrators.

use crate::model::Model;

/// Minimal vector-space interface a state needs to be integrated.
pub trait StateVector: Clone {
    /// `self + a * dx`
    fn axpy(&self, a: f64, dx: &Self) -> Self;
}

impl<const N: usize> StateVector for [f64; N] {
    fn axpy(&self, a: f64, dx: &Self) -> Self {
        std::array::from_fn(|i| self[i] + a * dx[i])
    }
}

pub trait Integrator {
    fn step<M>(&self, model: &M, x: &M::State, u: &M::Input, dt: f64) -> M::State
    where
        M: Model,
        M::State: StateVector;
}

/// Classical 4th-order Runge-Kutta with the input held over the step.
#[derive(Debug, Clone, Copy, Default)]
pub struct Rk4;

impl Integrator for Rk4 {
    fn step<M>(&self, model: &M, x: &M::State, u: &M::Input, dt: f64) -> M::State
    where
        M: Model,
        M::State: StateVector,
    {
        let k1 = model.derivative(x, u);
        let k2 = model.derivative(&x.axpy(0.5 * dt, &k1), u);
        let k3 = model.derivative(&x.axpy(0.5 * dt, &k2), u);
        let k4 = model.derivative(&x.axpy(dt, &k3), u);
        // x + dt/6 (k1 + 2 k2 + 2 k3 + k4)
        let sum = k1.axpy(2.0, &k2).axpy(2.0, &k3).axpy(1.0, &k4);
        x.axpy(dt / 6.0, &sum)
    }
}

/// Explicit Euler (mainly for tests and cheap sub-models).
#[derive(Debug, Clone, Copy, Default)]
pub struct Euler;

impl Integrator for Euler {
    fn step<M>(&self, model: &M, x: &M::State, u: &M::Input, dt: f64) -> M::State
    where
        M: Model,
        M::State: StateVector,
    {
        x.axpy(dt, &model.derivative(x, u))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    struct Decay;
    impl Model for Decay {
        type State = [f64; 1];
        type Input = f64;
        fn derivative(&self, x: &[f64; 1], k: &f64) -> [f64; 1] {
            [-k * x[0]]
        }
    }

    #[test]
    fn rk4_matches_exponential() {
        let mut x = [1.0];
        for _ in 0..100 {
            x = Rk4.step(&Decay, &x, &1.0, 0.01);
        }
        assert_relative_eq!(x[0], (-1.0f64).exp(), epsilon = 1e-10);
    }
}
