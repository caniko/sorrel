//! Legacy MATLAB `rez.mat` backend for Kilosort 2, 2.5, and 3.
//!
//! This reader opens MATLAB v7.3 (HDF5) files with the fields below.
//! Kilosort 4 instead writes native Phy-format `.npy` arrays; open those
//! with Sorrel's [Phy reader](crate::KilosortProvider). The `ks4_rez` module
//! and `Ks4Rez*` type names are retained for compatibility.
//!
//! - `rez/st3` — MATLAB `(n_spikes, n_columns)` matrix. The first two columns
//!   hold 1-based sample indices and template IDs. The reader converts both
//!   to zero-based indices and uses template IDs as cluster IDs; it ignores
//!   all remaining columns.
//! - `rez/ops/fs` — sample rate.
//! - `rez/ops/Nchan` — channel count used to open the recording.
//! - `rez/ops/fbinary` (or `fproc`) — path to the raw / preprocessed dat.
//! - `rez/xc`, `rez/yc` — channel positions (µm).
//!
//! If available, the binary is opened separately as headerless, interleaved
//! `int16` samples.
//!
//! The `hdf5` feature is optional in `sorrel-io` and enabled by default in
//! the `sorrel` application.

#[cfg(not(feature = "hdf5"))]
mod stub {
    use anyhow::{bail, Result};
    use std::path::Path;

    pub fn open_stub(_path: impl AsRef<Path>) -> Result<()> {
        bail!(
            "legacy MATLAB rez.mat requires HDF5; enable the `hdf5` feature or \
             open a Phy export of the sorting instead"
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
        ChannelId, ClusterId, DataProvider, RawFile, SampleIndex, TraceDtype, TraceSamples,
        TraceSlice,
    };
    use anyhow::{anyhow, bail, Context, Result};
    use hdf5_metno as hdf5;
    use hdf5_metno::File as H5File;
    use memmap2::Mmap;
    use std::fs::File;
    use std::path::{Path, PathBuf};

    /// Reader for legacy MATLAB v7.3 `rez.mat` files, not Kilosort 4 output.
    pub struct Ks4RezProvider {
        sample_rate: f64,
        n_channels: u32,
        n_samples: SampleIndex,
        spikes_per_cluster: Vec<Vec<SampleIndex>>,
        templates_per_cluster: Vec<Vec<u32>>,
        initial_labels: Vec<PhyLabel>,
        channel_positions: Vec<[f32; 2]>,
        identity_bytes: Vec<u8>,
        raw: Option<RawFile>,
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
            let f = H5File::open(path)
                .with_context(|| format!("open legacy MATLAB v7.3 file {}", path.display()))?;

            let rez = f.group("rez").context("missing /rez group")?;
            let ops = rez.group("ops").context("missing /rez/ops")?;

            let sample_rate = read_number(&ops, "fs")?;
            let n_channels = read_number(&ops, "Nchan")? as u32;

            // MATLAB's column-major layout appears as (columns, spikes) in HDF5.
            let st3 = rez.dataset("st3").context("missing /rez/st3")?;
            let shape = st3.shape();
            if shape.len() != 2 {
                bail!("rez/st3 expected 2-D, got shape {:?}", shape);
            }
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
            let mut n_clusters = 0;
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
                n_clusters = n_clusters.max(tmpl as usize + 1);
                tuples.push((sample, tmpl));
            }
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

            let dat_path = overrides
                .dat_path
                .or_else(|| read_string_dataset(&ops, "fbinary").map(PathBuf::from))
                .or_else(|| read_string_dataset(&ops, "fproc").map(PathBuf::from));
            let identity_bytes = provider_identity_bytes(path, dat_path.as_deref());
            let (trace_mmap, trace_ptr, trace_byte_len, n_samples) = match &dat_path {
                Some(p) => map_dat(p, n_channels)?,
                None => (None, std::ptr::null(), 0usize, SampleIndex::new(0)),
            };
            let raw = trace_mmap.as_ref().and(dat_path).map(|path| RawFile {
                path,
                dtype: TraceDtype::I16,
                offset: 0,
                hp_filtered: false,
            });

            Ok(Self {
                sample_rate,
                n_channels,
                n_samples,
                spikes_per_cluster,
                templates_per_cluster,
                initial_labels,
                channel_positions,
                identity_bytes,
                raw,
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
            let buf = unsafe { self.typed_slice::<i16>() };
            let e = e.min(buf.len());
            TraceSamples::I16(&buf[s.min(e)..e])
        }
    }

    impl DataProvider for Ks4RezProvider {
        type Label = PhyLabel;

        fn sample_rate(&self) -> f32 {
            self.sample_rate as f32
        }
        fn exact_sample_rate(&self) -> Result<f64> {
            Ok(self.sample_rate)
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

        fn identity_bytes(&self) -> Vec<u8> {
            self.identity_bytes.clone()
        }

        fn amplitude_full_scale(&self) -> f32 {
            TraceDtype::I16.nominal_full_scale()
        }

        fn raw_file(&self) -> Option<RawFile> {
            self.raw.clone()
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

    fn read_number(g: &hdf5::Group, name: &str) -> Result<f64> {
        let ds = g
            .dataset(name)
            .with_context(|| format!("missing /rez/ops/{name}"))?;
        if let Ok(v) = ds.read_scalar::<f64>() {
            return Ok(v);
        }
        // MATLAB can store scalars as 1-element arrays.
        if let Ok(v) = ds.read_raw::<f64>() {
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
                "rez.mat recording path {} does not exist; trace view disabled",
                p.display()
            );
            return Ok((None, std::ptr::null(), 0, SampleIndex::new(0)));
        }
        let f = File::open(p).with_context(|| format!("open {}", p.display()))?;
        let mmap = unsafe { Mmap::map(&f)? };
        let total = mmap.len();
        let bps = n_channels as usize * 2; // i16
        if bps == 0 || total.checked_rem(bps) != Some(0) {
            bail!("rez.mat recording size {total} not divisible by frame size {bps}");
        }
        let n_samples = (total / bps) as u64;
        let ptr = mmap.as_ptr();
        Ok((Some(mmap), ptr, total, SampleIndex::new(n_samples)))
    }

    fn provider_identity_bytes(rez_path: &Path, dat_path: Option<&Path>) -> Vec<u8> {
        let mut hasher = blake3::Hasher::new();
        // Preserve identities used by existing journals.
        hasher.update(b"sorrel-provider-identity-v1/ks4-rez");
        for path in [Some(rez_path), dat_path].into_iter().flatten() {
            hasher.update(
                path.file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("")
                    .as_bytes(),
            );
            match std::fs::metadata(path) {
                Ok(meta) => {
                    hasher.update(&meta.len().to_le_bytes());
                    if let Ok(modified) = meta.modified() {
                        if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
                            hasher.update(&duration.as_secs().to_le_bytes());
                            hasher.update(&duration.subsec_nanos().to_le_bytes());
                        }
                    }
                }
                Err(_) => {
                    hasher.update(b"missing");
                }
            }
        }
        hasher.finalize().as_bytes().to_vec()
    }
}

#[cfg(feature = "hdf5")]
pub use imp::{Ks4RezOpenParams, Ks4RezProvider};
