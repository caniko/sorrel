//! Kilosort 4 `rez.mat` backend.
//!
//! KS4 writes a MATLAB v7.3 file (= HDF5 under the hood) containing the
//! preprocessed sorting state. Fields we read:
//!
//! - `rez/st3` — `(n_spikes, ≥3)` matrix; column 0 is sample index
//!   (1-based, MATLAB convention), column 1 is template id, column 2 is
//!   amplitude. We use columns 0 and 1.
//! - `rez/ops/fs` — sample rate.
//! - `rez/ops/Nchan` — channel count of the *preprocessed* recording.
//! - `rez/ops/fbinary` (or `fproc`) — path to the raw / preprocessed dat.
//! - `rez/xc`, `rez/yc` — channel positions (µm), 1-D each.
//!
//! The dat file is opened separately as a regular int16 mmap (KS4 writes
//! the same flat layout phy expects); we mmap it via memmap2 like the
//! Kilosort backend.
//!
//! MATLAB stores arrays in column-major / Fortran order, but the HDF5
//! library returns the on-disk shape as (cols, rows) — we transpose at
//! load time so callers see row-major.
//!
//! Compiled out by default — enable with `--features hdf5`.

#[cfg(not(feature = "hdf5"))]
mod stub {
    use anyhow::{bail, Result};
    use std::path::Path;

    pub fn open_stub(_path: impl AsRef<Path>) -> Result<()> {
        bail!(
            "KS4 rez.mat backend not built; rebuild sorrel-io with `--features hdf5` \
             to enable the reader, or open the phy/ subdirectory that Kilosort \
             writes alongside rez.mat"
        )
    }
}

#[cfg(not(feature = "hdf5"))]
pub use stub::open_stub;

#[cfg(feature = "hdf5")]
mod imp {
    use crate::extras::{HasGeometry, HasSpikeTemplates};
    use crate::kilosort::PhyLabel;
    use crate::provider::{
        ChannelId, ClusterId, DataProvider, SampleIndex, TraceDtype, TraceSamples, TraceSlice,
    };
    use anyhow::{anyhow, bail, Context, Result};
    use hdf5_metno as hdf5;
    use hdf5_metno::File as H5File;
    use memmap2::Mmap;
    use std::fs::File;
    use std::path::{Path, PathBuf};

    pub struct Ks4RezProvider {
        sample_rate: f32,
        n_channels: u32,
        n_samples: SampleIndex,
        spikes_per_cluster: Vec<Vec<SampleIndex>>,
        templates_per_cluster: Vec<Vec<u32>>,
        initial_labels: Vec<PhyLabel>,
        channel_positions: Vec<[f32; 2]>,
        _trace_mmap: Option<Mmap>,
        trace_ptr: *const u8,
        trace_byte_len: usize,
    }

    unsafe impl Send for Ks4RezProvider {}
    unsafe impl Sync for Ks4RezProvider {}

    /// Optional override: explicit dat path. Useful when `ops/fbinary` was
    /// recorded as an absolute path that no longer exists (common when
    /// moving runs between machines).
    #[derive(Debug, Clone, Default)]
    pub struct Ks4RezOpenParams {
        pub dat_path: Option<PathBuf>,
    }

    impl Ks4RezProvider {
        pub fn open(path: impl AsRef<Path>, overrides: Ks4RezOpenParams) -> Result<Self> {
            let path = path.as_ref();
            let f = H5File::open(path).with_context(|| format!("open {}", path.display()))?;

            let rez = f.group("rez").context("missing /rez group")?;
            let ops = rez.group("ops").context("missing /rez/ops")?;

            let sample_rate: f32 = read_scalar_f32(&ops, "fs")?;
            let n_channels: u32 = read_scalar_f32(&ops, "Nchan")? as u32;

            // st3: shape stored as (cols, rows) by MATLAB. We want a contiguous
            // (n_spikes, n_cols) flat array.
            let st3 = rez.dataset("st3").context("missing /rez/st3")?;
            let shape = st3.shape();
            if shape.len() != 2 {
                bail!("rez/st3 expected 2-D, got shape {:?}", shape);
            }
            // HDF5 reports MATLAB shape verbatim; rows along the first axis
            // in MATLAB are the inner stride on disk, so the actual semantic
            // shape is (shape[1], shape[0]).
            let (n_spikes, n_cols) = (shape[1], shape[0]);
            if n_cols < 2 {
                bail!("rez/st3 has only {n_cols} columns; expected ≥2");
            }

            // read_raw gives a flat Vec following the MATLAB layout. We
            // transpose to row-major for direct (spike, col) indexing.
            let raw: Vec<f64> = st3.read_raw()?;
            if raw.len() != n_spikes * n_cols {
                bail!("rez/st3: payload size doesn't match declared shape");
            }
            let mut row_major = vec![0.0f64; raw.len()];
            for r in 0..n_spikes {
                for c in 0..n_cols {
                    // MATLAB column-major: raw[c * n_spikes + r] is (r, c).
                    row_major[r * n_cols + c] = raw[c * n_spikes + r];
                }
            }

            // Bucket spikes by template (column 1, 1-based template id).
            let mut max_template: u32 = 0;
            let mut tuples: Vec<(SampleIndex, u32)> = Vec::with_capacity(n_spikes);
            for r in 0..n_spikes {
                let sample = row_major[r * n_cols] as i64;
                let tmpl = row_major[r * n_cols + 1] as i64;
                if sample <= 0 || tmpl <= 0 {
                    continue;
                }
                // Convert to 0-based.
                let sample = SampleIndex::new((sample - 1) as u64);
                let tmpl = (tmpl - 1) as u32;
                max_template = max_template.max(tmpl);
                tuples.push((sample, tmpl));
            }
            let n_clusters = (max_template as usize) + 1;
            let mut spikes_per_cluster: Vec<Vec<SampleIndex>> = vec![Vec::new(); n_clusters];
            let mut templates_per_cluster: Vec<Vec<u32>> = vec![Vec::new(); n_clusters];
            for (sample, tmpl) in tuples {
                spikes_per_cluster[tmpl as usize].push(sample);
                templates_per_cluster[tmpl as usize].push(tmpl);
            }
            for b in spikes_per_cluster.iter_mut() {
                b.sort_unstable();
            }
            let initial_labels = vec![PhyLabel::Unsorted; n_clusters];

            // Channel positions from xc/yc.
            let xs: Vec<f64> = rez
                .dataset("xc")
                .ok()
                .and_then(|d| d.read_raw().ok())
                .unwrap_or_default();
            let ys: Vec<f64> = rez
                .dataset("yc")
                .ok()
                .and_then(|d| d.read_raw().ok())
                .unwrap_or_default();
            let np = xs.len().min(ys.len());
            let channel_positions: Vec<[f32; 2]> =
                (0..np).map(|i| [xs[i] as f32, ys[i] as f32]).collect();

            // Locate the dat file. KS4 writes it as a string dataset under
            // ops; HDF5-rs returns those as `VarLenAscii` / `FixedAscii`.
            let dat_path = overrides
                .dat_path
                .or_else(|| read_string_dataset(&ops, "fbinary").map(PathBuf::from))
                .or_else(|| read_string_dataset(&ops, "fproc").map(PathBuf::from));
            let (trace_mmap, trace_ptr, trace_byte_len, n_samples) = match dat_path {
                Some(p) => map_dat(&p, n_channels)?,
                None => (None, std::ptr::null(), 0usize, SampleIndex::new(0)),
            };

            Ok(Self {
                sample_rate,
                n_channels,
                n_samples,
                spikes_per_cluster,
                templates_per_cluster,
                initial_labels,
                channel_positions,
                _trace_mmap: trace_mmap,
                trace_ptr,
                trace_byte_len,
            })
        }

        unsafe fn typed_slice<T: Copy>(&self) -> &[T] {
            let len = self.trace_byte_len / std::mem::size_of::<T>();
            std::slice::from_raw_parts(self.trace_ptr as *const T, len)
        }

        fn samples_window(&self, start: SampleIndex, len: u32) -> TraceSamples<'_> {
            if self.trace_byte_len == 0 {
                return TraceSamples::I16(&[]);
            }
            let nc = self.n_channels as usize;
            let s = start.idx().saturating_mul(nc);
            let e = s + (len as usize) * nc;
            // KS4 fbinary is always int16.
            let buf = unsafe { self.typed_slice::<i16>() };
            let e = e.min(buf.len());
            TraceSamples::I16(&buf[s.min(e)..e])
        }
    }

    impl DataProvider for Ks4RezProvider {
        type Label = PhyLabel;

        fn sample_rate(&self) -> f32 {
            self.sample_rate
        }
        fn n_channels(&self) -> u32 {
            self.n_channels
        }
        fn n_samples(&self) -> SampleIndex {
            self.n_samples
        }
        fn n_clusters(&self) -> u32 {
            self.spikes_per_cluster.len() as u32
        }

        fn spike_times(&self, cluster: ClusterId) -> &[SampleIndex] {
            self.spikes_per_cluster
                .get(cluster.idx())
                .map(Vec::as_slice)
                .unwrap_or(&[])
        }

        fn trace(&self, start: SampleIndex, len: u32) -> TraceSlice<'_> {
            TraceSlice {
                start,
                n_channels: self.n_channels,
                samples: self.samples_window(start, len),
            }
        }

        fn initial_labels(&self) -> Vec<Self::Label> {
            self.initial_labels.clone()
        }

        fn amplitude_full_scale(&self) -> f32 {
            TraceDtype::I16.nominal_full_scale()
        }
    }

    impl HasGeometry for Ks4RezProvider {
        fn channel_positions(&self) -> &[[f32; 2]] {
            &self.channel_positions
        }
        fn channel_map(&self) -> &[ChannelId] {
            &[]
        }
    }

    impl HasSpikeTemplates for Ks4RezProvider {
        fn spike_templates(&self, cluster: ClusterId) -> &[u32] {
            self.templates_per_cluster
                .get(cluster.idx())
                .map(Vec::as_slice)
                .unwrap_or(&[])
        }
    }

    fn read_scalar_f32(g: &hdf5::Group, name: &str) -> Result<f32> {
        let ds = g
            .dataset(name)
            .with_context(|| format!("missing /rez/ops/{name}"))?;
        if let Ok(v) = ds.read_scalar::<f32>() {
            return Ok(v);
        }
        if let Ok(v) = ds.read_scalar::<f64>() {
            return Ok(v as f32);
        }
        // KS4 stores most numeric scalars as 1-element arrays.
        if let Ok(v) = ds.read_raw::<f64>() {
            if let Some(x) = v.first() {
                return Ok(*x as f32);
            }
        }
        if let Ok(v) = ds.read_raw::<f32>() {
            if let Some(x) = v.first() {
                return Ok(*x);
            }
        }
        Err(anyhow!("could not read /rez/ops/{name} as float"))
    }

    fn read_string_dataset(g: &hdf5::Group, name: &str) -> Option<String> {
        let ds = g.dataset(name).ok()?;
        // MATLAB stores char arrays as uint16 codepoints (UTF-16). Try that
        // form first, then fall back to a real HDF5 string.
        if let Ok(codes) = ds.read_raw::<u16>() {
            // Strip trailing nulls.
            let s: String = char::decode_utf16(codes.into_iter().take_while(|&c| c != 0))
                .filter_map(|r| r.ok())
                .collect();
            if !s.is_empty() {
                return Some(s);
            }
        }
        if let Ok(s) = ds.read_scalar::<hdf5::types::VarLenUnicode>() {
            return Some(s.to_string());
        }
        None
    }

    fn map_dat(p: &Path, n_channels: u32) -> Result<(Option<Mmap>, *const u8, usize, SampleIndex)> {
        if !p.exists() {
            log::warn!(
                "KS4 dat path {} does not exist; trace view disabled",
                p.display()
            );
            return Ok((None, std::ptr::null(), 0, SampleIndex::new(0)));
        }
        let f = File::open(p).with_context(|| format!("open {}", p.display()))?;
        let mmap = unsafe { Mmap::map(&f)? };
        let total = mmap.len();
        let bps = n_channels as usize * 2; // i16
        if bps == 0 || total.checked_rem(bps) != Some(0) {
            bail!("KS4 dat size {total} not divisible by frame size {bps}");
        }
        let n_samples = (total / bps) as u64;
        let ptr = mmap.as_ptr();
        Ok((Some(mmap), ptr, total, SampleIndex::new(n_samples)))
    }
}

#[cfg(feature = "hdf5")]
pub use imp::{Ks4RezOpenParams, Ks4RezProvider};
