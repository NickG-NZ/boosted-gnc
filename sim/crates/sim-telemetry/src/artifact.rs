//! Run artifacts: self-describing telemetry files written next to other run outputs.
//!
//! Format (JSON): `{"format": "boosted-sim/telemetry", "version": 1, "meta": {...},
//! "channels": {"<name>": {"width": w, "unit": "...", "data": [[...], ...]}}}`

use std::{fs, io, path::Path};

use serde_json::{Map, Value, json};

use crate::Recorder;

pub const FORMAT: &str = "boosted-sim/telemetry";
pub const VERSION: u32 = 1;

pub fn to_json(rec: &Recorder, meta: Value) -> Value {
    let mut channels = Map::new();
    for c in rec.channels() {
        let rows: Vec<&[f64]> = (0..c.rows()).map(|i| c.row(i)).collect();
        channels.insert(c.name.clone(), json!({"width": c.width, "unit": c.unit, "data": rows}));
    }
    json!({"format": FORMAT, "version": VERSION, "meta": meta, "channels": channels})
}

/// Write atomically (temp file + rename), creating parent directories.
pub fn write_json(path: &Path, rec: &Recorder, meta: Value) -> io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, serde_json::to_vec(&to_json(rec, meta))?)?;
    fs::rename(tmp, path)
}
