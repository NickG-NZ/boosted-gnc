//! Python bindings. Thin by design: conversion and error mapping only, no physics.

use std::path::PathBuf;

use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use sim_core::Vec3;
use sim_vehicle::{Command, Plant as RsPlant, PlantError};

fn err(e: PlantError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn v3(v: &Vec3) -> [f64; 3] {
    [v[0], v[1], v[2]]
}

/// Truth plant. Built from a JSON `PlantConfig` (see sim-vehicle/src/config.rs).
#[pyclass(module = "boosted_sim", unsendable)]
struct Plant {
    inner: RsPlant,
}

#[pymethods]
impl Plant {
    #[new]
    fn new(config_json: &str) -> PyResult<Self> {
        Ok(Self { inner: RsPlant::from_json(config_json).map_err(err)? })
    }

    /// Advance one step with the commands held. Returns a dict of step signals.
    #[pyo3(signature = (thrust, dt, gimbal=None, fins=None))]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        thrust: f64,
        dt: f64,
        gimbal: Option<[f64; 3]>,
        fins: Option<[f64; 3]>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let cmd = Command { thrust, gimbal: gimbal.map(Vec3::from), fins: fins.map(Vec3::from) };
        let info = self.inner.step(&cmd, dt).map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("dt", info.dt)?;
        d.set_item("contact", info.contact)?;
        d.set_item("thrust", info.thrust)?;
        d.set_item("thrust_vec", v3(&info.thrust_vec))?;
        d.set_item("g_cmd", v3(&info.g_cmd))?;
        d.set_item("torque", v3(&info.torque))?;
        d.set_item("alpha", v3(&info.alpha))?;
        Ok(d)
    }

    /// Append the final telemetry row (call once after the last step).
    fn finish(&mut self) {
        self.inner.finish();
    }

    #[getter]
    fn t(&self) -> f64 {
        self.inner.state().t
    }
    #[getter]
    fn r(&self) -> [f64; 3] {
        v3(&self.inner.state().r)
    }
    #[getter]
    fn v(&self) -> [f64; 3] {
        v3(&self.inner.state().v)
    }
    #[getter]
    fn m(&self) -> f64 {
        self.inner.state().m
    }
    #[getter]
    fn b(&self) -> [f64; 3] {
        v3(&self.inner.state().b)
    }
    #[getter]
    fn w(&self) -> [f64; 3] {
        v3(&self.inner.state().w)
    }
    #[getter]
    fn g(&self) -> [f64; 3] {
        v3(&self.inner.state().g)
    }
    #[getter]
    fn thrust_dir(&self) -> [f64; 3] {
        v3(&self.inner.state().thrust_dir())
    }
    #[getter]
    fn gimbal_deg(&self) -> f64 {
        self.inner.state().gimbal_deg()
    }
    #[getter]
    fn landed(&self) -> bool {
        self.inner.landed()
    }

    /// Truth wind (m/s) at position `r` at the current time.
    fn wind_at(&self, r: [f64; 3]) -> [f64; 3] {
        v3(&self.inner.wind_at(&Vec3::from(r)))
    }

    /// Recorded channels: {name: (width, unit, flat row-major data)}.
    fn telemetry<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        if let Some(rec) = self.inner.recorder() {
            for c in rec.channels() {
                d.set_item(&c.name, (c.width, &c.unit, &c.data))?;
            }
        }
        Ok(d)
    }

    /// Write the telemetry artifact (JSON) to `path`; `meta_json` is stored alongside.
    #[pyo3(signature = (path, meta_json="{}"))]
    fn write_telemetry(&self, path: PathBuf, meta_json: &str) -> PyResult<()> {
        let rec = self.inner.recorder().ok_or_else(|| PyValueError::new_err("recording disabled"))?;
        let meta = serde_json::from_str(meta_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        sim_telemetry::artifact::write_json(&path, rec, meta).map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pymodule]
fn boosted_sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Plant>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
