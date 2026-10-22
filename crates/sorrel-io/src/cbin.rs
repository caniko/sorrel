//! mtscomp (`.cbin` + `.ch`) compressed recordings — the IBL standard for
//! archived Neuropixels data (<https://github.com/int-brain-lab/mtscomp>).
//!
//! Layout: the `.ch` sidecar is JSON with `chunk_bounds` (sample index of
//! each chunk start, `n_chunks + 1` long), `chunk_offsets` (byte offset of
//! each chunk in the `.cbin`, same length), `n_channels`, `sample_rate`,
//! `dtype`, `do_time_diff`, `do_spatial_diff` and `chunk_order`. Each
//! chunk is zlib over an `int16` `(n_samples, n_channels)` block whose
//! rows after the first are first differences along time (wrapping
//! arithmetic, as numpy's `diff`/`cumsum` on `int16` behave). Decoding a
//! chunk is one inflate plus a cumulative sum.
//!
//! Only the shapes mtscomp actually writes are accepted: `int16`, no
//! spatial diff. A two-entry chunk cache absorbs the GUI asking for the
//! same window every frame.
//!
//! The hot path is the window read, so it is a plain `Mutex` around the
//! tiny cache rather than anything cleverer:
//! `// ponytail: 2-entry cache; grow or make it LRU if multi-window views arrive`

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use serde::Deserialize;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Deserialize)]
struct ChMeta {
    chunk_bounds: Vec<u64>,
    chunk_offsets: Vec<u64>,
    n_channels: usize,
    sample_rate: f64,
    #[serde(default = "default_dtype")]
    dtype: String,
    #[serde(default = "default_true")]
    do_time_diff: bool,
    #[serde(default)]
    do_spatial_diff: bool,
    #[serde(default = "default_order")]
    chunk_order: String,
    #[serde(default = "default_algorithm")]
    algorithm: String,
}

fn default_dtype() -> String {
    "int16".into()
}
fn default_true() -> bool {
    true
}
fn default_order() -> String {
    "C".into()
}
fn default_algorithm() -> String {
    "zlib".into()
}

pub struct CbinReader {
    mmap: Mmap,
    meta: ChMeta,
    cache: Mutex<Vec<(usize, Arc<Vec<i16>>)>>,
    path: PathBuf,
}

impl std::fmt::Debug for CbinReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CbinReader")
            .field("path", &self.path)
            .field("n_chunks", &self.n_chunks())
            .finish()
    }
}

impl CbinReader {
    /// Open `<name>.cbin`; the sidecar is `<name>.ch` next to it.
    pub fn open(cbin: &Path) -> Result<Self> {
        let ch = cbin.with_extension("ch");
        let text = std::fs::read_to_string(&ch)
            .with_context(|| format!("read mtscomp sidecar {}", ch.display()))?;
        let meta: ChMeta =
            serde_json::from_str(&text).with_context(|| format!("parse {}", ch.display()))?;
        if meta.dtype != "int16" {
            bail!("{}: only int16 mtscomp files are supported", ch.display());
        }
        if meta.do_spatial_diff {
            bail!("{}: do_spatial_diff is not supported", ch.display());
        }
        if meta.algorithm != "zlib" {
            bail!(
                "{}: unsupported algorithm {:?}",
                ch.display(),
                meta.algorithm
            );
        }
        if meta.chunk_bounds.len() != meta.chunk_offsets.len() || meta.chunk_bounds.len() < 2 {
            bail!("{}: chunk_bounds/chunk_offsets malformed", ch.display());
        }
        let f = File::open(cbin).with_context(|| format!("open {}", cbin.display()))?;
        // SAFETY: read-only map of an archive file sorrel never writes.
        let mmap = unsafe { Mmap::map(&f) }?;
        Ok(Self {
            mmap,
            meta,
            cache: Mutex::new(Vec::with_capacity(2)),
            path: cbin.to_path_buf(),
        })
    }

    pub fn n_channels(&self) -> usize {
        self.meta.n_channels
    }
    pub fn sample_rate(&self) -> f32 {
        self.meta.sample_rate as f32
    }
    pub fn n_samples(&self) -> u64 {
        *self.meta.chunk_bounds.last().unwrap_or(&0)
    }
    fn n_chunks(&self) -> usize {
        self.meta.chunk_bounds.len() - 1
    }

    /// Interleaved `(len, n_channels)` samples for `[start, start + len)`,
    /// clamped to the recording.
    pub fn window(&self, start: u64, len: usize) -> Vec<i16> {
        let nc = self.meta.n_channels;
        let end = (start + len as u64).min(self.n_samples());
        if start >= end {
            return Vec::new();
        }
        let mut out = Vec::with_capacity((end - start) as usize * nc);
        let bounds = &self.meta.chunk_bounds;
        let mut k = bounds.partition_point(|&b| b <= start).saturating_sub(1);
        let mut t = start;
        while t < end && k < self.n_chunks() {
            let chunk = match self.chunk(k) {
                Ok(c) => c,
                Err(e) => {
                    log::warn!("{}: chunk {k}: {e:#}", self.path.display());
                    break;
                }
            };
            let lo = (t - bounds[k]) as usize;
            let hi = (end.min(bounds[k + 1]) - bounds[k]) as usize;
            out.extend_from_slice(&chunk[lo * nc..hi * nc]);
            t = bounds[k + 1];
            k += 1;
        }
        out
    }

    fn chunk(&self, k: usize) -> Result<Arc<Vec<i16>>> {
        if let Some((_, c)) = self.cache.lock().unwrap().iter().find(|(i, _)| *i == k) {
            return Ok(Arc::clone(c));
        }
        let decoded = Arc::new(self.decode_chunk(k)?);
        let mut cache = self.cache.lock().unwrap();
        if cache.len() == 2 {
            cache.remove(0);
        }
        cache.push((k, Arc::clone(&decoded)));
        Ok(decoded)
    }

    fn decode_chunk(&self, k: usize) -> Result<Vec<i16>> {
        let (b0, b1) = (self.meta.chunk_offsets[k], self.meta.chunk_offsets[k + 1]);
        let raw = self
            .mmap
            .get(b0 as usize..b1 as usize)
            .context("chunk offsets exceed .cbin size")?;
        let n_rows = (self.meta.chunk_bounds[k + 1] - self.meta.chunk_bounds[k]) as usize;
        let nc = self.meta.n_channels;
        let mut bytes = Vec::with_capacity(n_rows * nc * 2);
        flate2::read::ZlibDecoder::new(raw).read_to_end(&mut bytes)?;
        if bytes.len() != n_rows * nc * 2 {
            bail!(
                "decoded {} bytes, expected {} ({n_rows} rows × {nc} channels)",
                bytes.len(),
                n_rows * nc * 2
            );
        }
        let flat: Vec<i16> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| i16::from_le_bytes(*c))
            .collect();
        // `F` order stores channels contiguously; bring it to (rows, channels).
        let mut data = if self.meta.chunk_order.eq_ignore_ascii_case("F") {
            let mut d = vec![0i16; flat.len()];
            for c in 0..nc {
                for r in 0..n_rows {
                    d[r * nc + c] = flat[c * n_rows + r];
                }
            }
            d
        } else {
            flat
        };
        if self.meta.do_time_diff {
            for r in 1..n_rows {
                for c in 0..nc {
                    data[r * nc + c] = data[(r - 1) * nc + c].wrapping_add(data[r * nc + c]);
                }
            }
        }
        Ok(data)
    }
}

/// Write a `.cbin`/`.ch` pair from interleaved samples (tests and tooling;
/// mtscomp itself is Python). Chunks are `chunk_rows` samples long.
pub fn write_cbin(
    cbin: &Path,
    samples: &[i16],
    n_channels: usize,
    sample_rate: f64,
    chunk_rows: usize,
) -> Result<()> {
    use std::io::Write;
    let n_rows = samples.len() / n_channels.max(1);
    let mut bounds = vec![0u64];
    let mut offsets = vec![0u64];
    let mut out = Vec::new();
    for r0 in (0..n_rows).step_by(chunk_rows.max(1)) {
        let r1 = (r0 + chunk_rows).min(n_rows);
        let mut diffed: Vec<u8> = Vec::with_capacity((r1 - r0) * n_channels * 2);
        for r in r0..r1 {
            for c in 0..n_channels {
                let v = samples[r * n_channels + c];
                let d = if r == r0 {
                    v
                } else {
                    v.wrapping_sub(samples[(r - 1) * n_channels + c])
                };
                diffed.extend_from_slice(&d.to_le_bytes());
            }
        }
        let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(&diffed)?;
        out.extend_from_slice(&enc.finish()?);
        bounds.push(r1 as u64);
        offsets.push(out.len() as u64);
    }
    std::fs::write(cbin, &out)?;
    let meta = serde_json::json!({
        "version": "1.0", "algorithm": "zlib", "comp_level": 6,
        "do_time_diff": true, "do_spatial_diff": false, "dtype": "int16",
        "n_channels": n_channels, "sample_rate": sample_rate,
        "chunk_bounds": bounds, "chunk_offsets": offsets, "chunk_order": "C",
        "shape": [n_rows, n_channels],
    });
    std::fs::write(cbin.with_extension("ch"), serde_json::to_vec_pretty(&meta)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_across_chunk_boundaries_round_trip_including_wrapping_jumps() {
        let nc = 3;
        let n_rows = 25;
        // Includes a jump larger than i16::MAX so the diff wraps.
        let samples: Vec<i16> = (0..n_rows * nc)
            .map(|i| match i / nc {
                7 => -30000,
                8 => 30000,
                r => (r as i16) * 100 + (i % nc) as i16,
            })
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let cbin = dir.path().join("rec.cbin");
        write_cbin(&cbin, &samples, nc, 30000.0, 10).unwrap();

        let r = CbinReader::open(&cbin).unwrap();
        assert_eq!(r.n_samples(), 25);
        assert_eq!(r.n_channels(), 3);
        assert_eq!(r.window(0, 25), samples);
        // Spans chunks 0..2 and clamps at the end.
        assert_eq!(r.window(8, 100), &samples[8 * nc..]);
        assert_eq!(r.window(9, 2), &samples[9 * nc..11 * nc]);
        assert!(r.window(25, 5).is_empty());
        assert_eq!(r.window(0, 1), &samples[..nc]);
    }
}
