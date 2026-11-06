//! Continuous-time model abstraction.

/// A continuous-time system `x' = f(x, u)`. Inputs are held constant over an integration step
/// (zero-order hold), which is how flight software commands reach the plant.
pub trait Model {
    type State;
    type Input;

    fn derivative(&self, x: &Self::State, u: &Self::Input) -> Self::State;
}
