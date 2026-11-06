//! Python bindings. Thin by design: argument conversion and error mapping only.

use std::path::PathBuf;

use pyo3::{exceptions::PyRuntimeError, prelude::*};
use sim_runner::SolveOpts;

/// Plan and simulate a mission's cases; returns the viewer data as a JSON string
/// `{scenario, rocket, plans, sims}`. With `only`, `sims` holds just those cases.
/// `progress(stage, done, total, label)` is called before each plan / sim and once with "done".
#[pyfunction]
#[pyo3(signature = (scenario, only=None, verbose=true, artifacts=None, progress=None))]
fn solve(
    py: Python<'_>,
    scenario: PathBuf,
    only: Option<Vec<String>>,
    verbose: bool,
    artifacts: Option<PathBuf>,
    progress: Option<Py<PyAny>>,
) -> PyResult<String> {
    // a Python error in the callback (or Ctrl-C) cancels the run and is re-raised
    let mut failed: Option<PyErr> = None;
    let out = {
        let cb = |stage: &str, done: usize, total: usize, label: &str| -> bool {
            let r = py.check_signals().and_then(|_| match &progress {
                Some(f) => f.call1(py, (stage, done, total, label)).map(|_| ()),
                None => Ok(()),
            });
            r.map_err(|e| failed = Some(e)).is_ok()
        };
        let opts = SolveOpts { only, verbose, artifacts, progress: Some(Box::new(cb)) };
        sim_runner::solve(&scenario, opts)
    };
    if let Some(e) = failed {
        return Err(e);
    }
    out.map(|v| v.to_string()).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pymodule]
fn boosted_sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
