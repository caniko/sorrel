//! Open Ephys binary-format recording metadata reader.
//!
//! The "Binary" Open Ephys writer produces a directory tree with a JSON
//! sidecar `structure.oebin` describing every continuous stream and
//! event/spike file. We read just enough of it to populate
//! [`KilosortOpenParams`](crate::kilosort::KilosortOpenParams) for the chosen
//! continuous stream — sample rate, channel count, dtype, and the bin path.
//!
//! Schema reference (subset):
//! ```json
//! {
//!   "continuous": [{
//!     "folder_name": "Neuropix-PXI-100.0/",
//!     "sample_rate": 30000.0,
//!     "num_channels": 384,
//!     "channels": [...]
//!   }]
//! }
//! ```

use crate::provider::TraceDtype;
use anyhow::{anyhow, bail, Context, Result};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct OebinStream {
    /// Subdirectory under the experiment root that contains `continuous.dat`.
    pub folder: PathBuf,
    pub sample_rate: f32,
    pub n_channels: u32,
    /// Per-channel calibration in µV/bit. Non-empty when present in the JSON.
    pub bit_volts: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct OebinMeta {
    pub continuous: Vec<OebinStream>,
}

impl OebinMeta {
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        Self::from_json(&text)
    }

    pub fn from_json(text: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(text).context("parse structure.oebin")?;
        let arr = v
            .get("continuous")
            .and_then(|c| c.as_array())
            .ok_or_else(|| anyhow!("oebin: missing 'continuous' array"))?;
        if arr.is_empty() {
            bail!("oebin: empty 'continuous' array");
        }
        let mut out = Vec::with_capacity(arr.len());
        for s in arr {
            let folder: String = s
                .get("folder_name")
                .and_then(|f| f.as_str())
                .ok_or_else(|| anyhow!("oebin stream missing folder_name"))?
                .to_string();
            let sample_rate = s
                .get("sample_rate")
                .and_then(|x| x.as_f64())
                .ok_or_else(|| anyhow!("oebin stream missing sample_rate"))?
                as f32;
            let n_channels = s
                .get("num_channels")
                .and_then(|x| x.as_u64())
                .ok_or_else(|| anyhow!("oebin stream missing num_channels"))?
                as u32;
            let mut bit_volts = Vec::new();
            if let Some(channels) = s.get("channels").and_then(|c| c.as_array()) {
                for ch in channels {
                    if let Some(b) = ch.get("bit_volts").and_then(|b| b.as_f64()) {
                        bit_volts.push(b as f32);
                    }
                }
            }
            out.push(OebinStream {
                folder: PathBuf::from(folder),
                sample_rate,
                n_channels,
                bit_volts,
            });
        }
        Ok(Self { continuous: out })
    }

    /// First continuous stream — what most single-probe recordings have.
    pub fn primary(&self) -> Option<&OebinStream> {
        self.continuous.first()
    }
}

impl OebinStream {
    /// Open Ephys binary always writes int16 in `continuous.dat`.
    pub const DTYPE: TraceDtype = TraceDtype::I16;

    /// Resolve `<oebin_dir>/<folder>/continuous.dat`.
    pub fn dat_path(&self, oebin_dir: &Path) -> PathBuf {
        oebin_dir.join(&self.folder).join("continuous.dat")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_oebin() {
        let json = r#"{
            "continuous": [{
                "folder_name": "Neuropix-PXI-100.0/",
                "sample_rate": 30000.0,
                "num_channels": 384,
                "channels": [{"bit_volts": 0.195}, {"bit_volts": 0.195}]
            }]
        }"#;
        let m = OebinMeta::from_json(json).unwrap();
        let s = m.primary().unwrap();
        assert_eq!(s.sample_rate as u32, 30000);
        assert_eq!(s.n_channels, 384);
        assert_eq!(s.bit_volts.len(), 2);
        assert_eq!(s.folder, PathBuf::from("Neuropix-PXI-100.0/"));
    }

    #[test]
    fn rejects_empty_continuous() {
        assert!(OebinMeta::from_json(r#"{"continuous": []}"#).is_err());
    }

    #[test]
    fn rejects_missing_required_fields() {
        assert!(OebinMeta::from_json(r#"{"continuous": [{}]}"#).is_err());
    }
}
