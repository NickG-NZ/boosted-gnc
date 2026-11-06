use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelId(usize);

/// A named channel of fixed-width rows, stored row-major in one flat buffer.
#[derive(Debug, Clone, Serialize)]
pub struct Channel {
    pub name: String,
    pub width: usize,
    pub unit: String,
    pub data: Vec<f64>,
}

impl Channel {
    pub fn rows(&self) -> usize {
        self.data.len() / self.width
    }

    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.width..(i + 1) * self.width]
    }
}

#[derive(Debug, Clone, Default)]
pub struct Recorder {
    channels: Vec<Channel>,
}

impl Recorder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a channel (or return the existing one with the same name).
    pub fn channel(&mut self, name: &str, width: usize, unit: &str) -> ChannelId {
        assert!(width > 0, "channel width must be > 0");
        if let Some(i) = self.channels.iter().position(|c| c.name == name) {
            assert_eq!(self.channels[i].width, width, "channel {name} re-registered with another width");
            return ChannelId(i);
        }
        self.channels.push(Channel { name: name.into(), width, unit: unit.into(), data: Vec::new() });
        ChannelId(self.channels.len() - 1)
    }

    pub fn push(&mut self, id: ChannelId, row: &[f64]) {
        let c = &mut self.channels[id.0];
        debug_assert_eq!(row.len(), c.width, "row width mismatch on {}", c.name);
        c.data.extend_from_slice(row);
    }

    pub fn get(&self, name: &str) -> Option<&Channel> {
        self.channels.iter().find(|c| c.name == name)
    }

    pub fn channels(&self) -> &[Channel] {
        &self.channels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_and_read_back() {
        let mut rec = Recorder::new();
        let r = rec.channel("r", 3, "m");
        rec.push(r, &[1.0, 2.0, 3.0]);
        rec.push(r, &[4.0, 5.0, 6.0]);
        let c = rec.get("r").unwrap();
        assert_eq!(c.rows(), 2);
        assert_eq!(c.row(1), &[4.0, 5.0, 6.0]);
        assert_eq!(rec.channel("r", 3, "m"), r);
    }
}
