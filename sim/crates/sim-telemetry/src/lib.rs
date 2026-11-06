//! Telemetry recording and artifact output.
//!
//! `Recorder` holds fixed-width, append-only channels (one row per sample). Models register the
//! channels they publish; consumers (Python bindings, artifact writers, a future live stream)
//! read them back without the models knowing who listens.

pub mod artifact;
mod recorder;

pub use recorder::{Channel, ChannelId, Recorder};
