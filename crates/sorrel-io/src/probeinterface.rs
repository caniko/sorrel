//! Minimal reader for the `probeinterface` JSON format
//! (https://github.com/SpikeInterface/probeinterface).
//!
//! The format is the de-facto standard probe-geometry exchange used by
//! SpikeInterface, NEO, MEArec and others. We extract just what
//! [`HasGeometry`] needs: 2-D contact positions, per-contact shank id (if
//! present), and the device-channel ↔ logical-channel map.
//!
//! Spec we target (subset):
//! ```json
//! {
//!   "specification": "probeinterface",
//!   "version": "0.2.21",
//!   "probes": [{
//!     "ndim": 2,
//!     "si_units": "um",
//!     "contact_positions": [[x0, y0], [x1, y1], ...],
//!     "device_channel_indices": [3, 0, 1, 2, ...],
//!     "shank_ids": ["0", "0", "1", ...]   // optional
//!   }]
//! }
//! ```
//! 3-D probes collapse the third axis (z is rarely used by viewers).

use crate::provider::ChannelId;
use anyhow::{anyhow, bail, Context, Result};
use std::path::Path;

/// Parsed probe geometry, in the shape [`HasGeometry`](crate::extras::HasGeometry)
/// expects: positions indexed by raw-dat row, optional shank ids, optional
/// channel-map.
#[derive(Debug, Clone, Default)]
pub struct ProbeGeometry {
    pub channel_positions: Vec<[f32; 2]>,
    pub channel_shanks: Vec<u32>,
    pub channel_map: Vec<ChannelId>,
}

impl ProbeGeometry {
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        Self::from_json(&text)
    }

    pub fn from_json(text: &str) -> Result<Self> {
        let v: serde_json::Value =
            serde_json::from_str(text).context("parse probeinterface json")?;
        let probes = v
            .get("probes")
            .and_then(|p| p.as_array())
            .ok_or_else(|| anyhow!("probeinterface: missing 'probes' array"))?;
        if probes.is_empty() {
            bail!("probeinterface: empty 'probes' array");
        }

        let mut all_positions = Vec::new();
        let mut all_shanks = Vec::new();
        let mut all_device_idx = Vec::new();
        let mut shank_offset: u32 = 0;

        for (probe_idx, probe) in probes.iter().enumerate() {
            let positions = probe
                .get("contact_positions")
                .and_then(|p| p.as_array())
                .ok_or_else(|| anyhow!("probe {probe_idx}: missing 'contact_positions'"))?;
            for (i, row) in positions.iter().enumerate() {
                let pair = row.as_array().ok_or_else(|| {
                    anyhow!("probe {probe_idx} contact {i}: position is not an array")
                })?;
                let x = as_f32(&pair[0])?;
                // y is present for 2-D and 3-D probes; we ignore z.
                let y = pair.get(1).map(as_f32).transpose()?.unwrap_or(0.0);
                all_positions.push([x, y]);
            }

            let n_contacts = positions.len();
            let mut shanks_local = vec![0u32; n_contacts];
            if let Some(shank_arr) = probe.get("shank_ids").and_then(|s| s.as_array()) {
                let mut max_id = 0u32;
                for (i, s) in shank_arr.iter().enumerate().take(n_contacts) {
                    let id = match s {
                        serde_json::Value::String(s) => s.parse::<u32>().unwrap_or(0),
                        serde_json::Value::Number(n) => n.as_u64().unwrap_or(0) as u32,
                        _ => 0,
                    };
                    shanks_local[i] = id;
                    max_id = max_id.max(id);
                }
                all_shanks.extend(shanks_local.iter().map(|s| s + shank_offset));
                shank_offset += max_id + 1;
            } else {
                all_shanks.resize(all_shanks.len() + n_contacts, shank_offset);
                shank_offset += 1;
            }

            if let Some(idx_arr) = probe
                .get("device_channel_indices")
                .and_then(|d| d.as_array())
            {
                for v in idx_arr.iter().take(n_contacts) {
                    let i = v.as_i64().ok_or_else(|| {
                        anyhow!("probe {probe_idx}: device_channel_indices not integer")
                    })?;
                    // -1 means "disabled"; map onto u32::MAX as a sentinel.
                    all_device_idx.push(if i < 0 {
                        ChannelId(u32::MAX)
                    } else {
                        ChannelId(i as u32)
                    });
                }
            }
        }

        Ok(Self {
            channel_positions: all_positions,
            channel_shanks: all_shanks,
            channel_map: all_device_idx,
        })
    }
}

fn as_f32(v: &serde_json::Value) -> Result<f32> {
    v.as_f64()
        .map(|x| x as f32)
        .ok_or_else(|| anyhow!("expected number, got {v}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_2d_probe() {
        let json = r#"{
            "specification": "probeinterface",
            "version": "0.2.21",
            "probes": [{
                "ndim": 2,
                "si_units": "um",
                "contact_positions": [[0.0, 0.0], [10.0, 20.0], [20.0, 40.0]],
                "device_channel_indices": [2, 0, 1]
            }]
        }"#;
        let p = ProbeGeometry::from_json(json).unwrap();
        assert_eq!(p.channel_positions.len(), 3);
        assert_eq!(p.channel_positions[1], [10.0, 20.0]);
        assert_eq!(p.channel_shanks, vec![0, 0, 0]);
        assert_eq!(
            p.channel_map,
            vec![ChannelId(2), ChannelId(0), ChannelId(1)]
        );
    }

    #[test]
    fn parses_multi_shank_with_string_shank_ids() {
        let json = r#"{
            "specification": "probeinterface",
            "probes": [{
                "contact_positions": [[0,0],[0,10],[100,0],[100,10]],
                "shank_ids": ["0","0","1","1"]
            }]
        }"#;
        let p = ProbeGeometry::from_json(json).unwrap();
        assert_eq!(p.channel_shanks, vec![0, 0, 1, 1]);
        assert!(p.channel_map.is_empty());
    }

    #[test]
    fn rejects_empty_probes() {
        let json = r#"{"probes": []}"#;
        assert!(ProbeGeometry::from_json(json).is_err());
    }

    #[test]
    fn handles_disabled_channel_with_negative_one() {
        let json = r#"{
            "probes": [{
                "contact_positions": [[0,0],[10,0]],
                "device_channel_indices": [0, -1]
            }]
        }"#;
        let p = ProbeGeometry::from_json(json).unwrap();
        assert_eq!(p.channel_map, vec![ChannelId(0), ChannelId(u32::MAX)]);
    }
}
