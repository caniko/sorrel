//! SpikeGLX recording metadata reader.
//!
//! SpikeGLX writes `<run>_g<g>_t<t>.<stream>.bin` plus a sibling `.meta` file
//! in INI-like `key=value` format. The bin layout is identical to phy's raw
//! `.dat` (interleaved int16, channel-major-within-sample), so we feed the
//! bin straight into [`KilosortProvider`](crate::kilosort::KilosortProvider)
//! via [`KilosortOpenParams`](crate::kilosort::KilosortOpenParams).
//!
//! This module does *not* itself implement `DataProvider` — there's no spike
//! sorting in a raw SpikeGLX run. The intended flow is "use SpikeGLX bin as
//! the dat for an existing kilosort/phy directory."

use crate::provider::TraceDtype;
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::Path;

/// Subset of SpikeGLX `.meta` fields we care about.
#[derive(Debug, Clone)]
pub struct SpikeGlxMeta {
    /// Saved sample rate (Hz). `imSampRate` for IMEC, `niSampRate` for NI-DAQ.
    pub sample_rate: f32,
    /// Total channels saved per timepoint (the `nSavedChans` field).
    pub n_channels: u32,
    /// Always int16 in SpikeGLX, kept for parity with [`TraceDtype`].
    pub dtype: TraceDtype,
    /// All raw key/value pairs, for callers that need extra fields.
    pub raw: HashMap<String, String>,
}

impl SpikeGlxMeta {
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read {}", path.display()))?;
        Self::parse(&text)
    }

    pub fn parse(text: &str) -> Result<Self> {
        let mut raw = HashMap::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with(';') || line.starts_with('#') {
                continue;
            }
            if let Some((k, v)) = line.split_once('=') {
                raw.insert(k.trim().to_string(), v.trim().to_string());
            }
        }

        let sample_rate = raw
            .get("imSampRate")
            .or_else(|| raw.get("niSampRate"))
            .and_then(|s| s.parse::<f32>().ok())
            .or_else(|| raw.get("sRateHz").and_then(|s| s.parse().ok()))
            .ok_or_else(|| {
                anyhow::anyhow!("spikeglx meta: no imSampRate/niSampRate field")
            })?;

        let n_channels = raw
            .get("nSavedChans")
            .and_then(|s| s.parse::<u32>().ok())
            .or_else(|| raw.get("nChans").and_then(|s| s.parse().ok()))
            .ok_or_else(|| anyhow::anyhow!("spikeglx meta: no nSavedChans field"))?;

        if let Some(dt) = raw.get("dataType") {
            if dt.trim() != "I" && !dt.trim().eq_ignore_ascii_case("int16") {
                bail!("spikeglx dataType={dt} not supported; expected int16");
            }
        }

        Ok(Self {
            sample_rate,
            n_channels,
            dtype: TraceDtype::I16,
            raw,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_imec_meta() {
        let text = "
imSampRate=30000.123
nSavedChans=385
typeImEnabled=1
fileSizeBytes=12345
";
        let m = SpikeGlxMeta::parse(text).unwrap();
        assert!((m.sample_rate - 30000.123).abs() < 1e-3);
        assert_eq!(m.n_channels, 385);
        assert_eq!(m.dtype, TraceDtype::I16);
    }

    #[test]
    fn parses_nidaq_meta_and_falls_back_to_ni_samp_rate() {
        let text = "niSampRate=25000\nnSavedChans=8\n";
        let m = SpikeGlxMeta::parse(text).unwrap();
        assert_eq!(m.sample_rate as u32, 25000);
        assert_eq!(m.n_channels, 8);
    }

    #[test]
    fn rejects_missing_sample_rate() {
        assert!(SpikeGlxMeta::parse("nSavedChans=2").is_err());
    }
}
