//! NWB / DANDI backend.
//!
//! Reads the canonical NWB extracellular layout:
//!
//! - `/units/spike_times` + `/units/spike_times_index` — ragged vlen array
//!   collapsed into one `Vec<u64>` per cluster (samples, not seconds).
//! - `/units/id` — sorted unit ids (the column order in `spike_times_index`).
//! - `/general/extracellular_ephys/electrodes` — column-store table; we read
//!   `x`, `y`, `rel_x`, `rel_y` (any of those that exist) for channel positions.
//! - `/acquisition/<name>/data` — first `ElectricalSeries` we find; loaded
//!   into RAM (NWB datasets are usually chunked + compressed, so memory
//!   mapping isn't an option).
//!
//! NWB stores spike times in **seconds**. We multiply by the recording's
//! sample rate (read from the same `ElectricalSeries`'s `starting_time`
//! attribute or from `/general/extracellular_ephys/.../sampling_frequency`)
//! to get sample indices.
//!
//! Compiled out by default — enable with `--features hdf5`.
//!
//! Limitations:
//! - DANDI streaming is local-only (we open the file with the `hdf5` crate).
//! - Multi-segment recordings collapse to the first segment's sample rate.
//! - Trace data is loaded eagerly; a large recording will use that much RAM.

#[cfg(not(feature = "hdf5"))]
mod stub {
    use anyhow::{bail, Result};
    use std::path::Path;

    pub fn open_stub(_path: impl AsRef<Path>) -> Result<()> {
        bail!(
            "NWB backend not built; rebuild sorrel-io with `--features hdf5` \
             to enable the HDF5-based reader, or export the units table to phy \
             via SpikeInterface and open that directory instead"
        )
    }
}

#[cfg(not(feature = "hdf5"))]
pub use stub::open_stub;

#[cfg(feature = "hdf5")]
mod imp {
    use crate::extras::HasGeometry;
    use crate::kilosort::PhyLabel;
    use crate::provider::{
        ChannelId, ClusterId, DataProvider, SampleIndex, TraceDtype, TraceSamples, TraceSlice,
    };
    use anyhow::{anyhow, bail, Context, Result};
    use hdf5_metno as hdf5;
    use hdf5_metno::File as H5File;
    use std::path::Path;

    pub struct NwbProvider {
        sample_rate: f32,
        n_channels: u32,
        n_samples: SampleIndex,
        dtype: TraceDtype,
        spikes_per_cluster: Vec<Vec<SampleIndex>>,
        initial_labels: Vec<PhyLabel>,
        channel_positions: Vec<[f32; 2]>,
        identity_bytes: Vec<u8>,
        // Eagerly-loaded trace buffer; HDF5 datasets are typically chunked
        // and compressed so we can't mmap. Stored as raw bytes and reinterpreted
        // at access time, just like the other providers.
        trace_buf: Vec<u8>,
    }

    impl NwbProvider {
        pub fn open(path: impl AsRef<Path>) -> Result<Self> {
            let path = path.as_ref();
            let f = H5File::open(path).with_context(|| format!("open NWB {}", path.display()))?;
            let identity_bytes = provider_identity_bytes(path);

            // 1. ElectricalSeries: first dataset under /acquisition that has
            // both `data` and a sampling rate (either explicit field or
            // derivable from `starting_time`'s `rate` attribute).
            let (sample_rate, n_channels, n_samples, dtype, trace_buf) =
                load_electrical_series(&f)?;

            // 2. Units table.
            let (spikes_per_cluster, n_clusters) =
                load_units(&f, sample_rate).context("load /units")?;
            let initial_labels = vec![PhyLabel::Unsorted; n_clusters];

            // 3. Channel positions from /general/extracellular_ephys/electrodes.
            let channel_positions =
                load_electrode_positions(&f, n_channels as usize).unwrap_or_default();

            Ok(Self {
                sample_rate,
                n_channels,
                n_samples,
                dtype,
                spikes_per_cluster,
                initial_labels,
                channel_positions,
                identity_bytes,
                trace_buf,
            })
        }

        fn samples_window(&self, start: SampleIndex, len: u32) -> TraceSamples<'_> {
            if self.trace_buf.is_empty() {
                return TraceSamples::I16(&[]);
            }
            let nc = self.n_channels as usize;
            let s = start.idx().saturating_mul(nc);
            let e = s + (len as usize) * nc;
            TraceSamples::from_bytes_clamped(&self.trace_buf, self.dtype, s, e)
        }
    }

    impl DataProvider for NwbProvider {
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

        fn identity_bytes(&self) -> Vec<u8> {
            self.identity_bytes.clone()
        }

        fn amplitude_full_scale(&self) -> f32 {
            self.dtype.nominal_full_scale()
        }
    }

    fn provider_identity_bytes(path: &Path) -> Vec<u8> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"sorrel-provider-identity-v1/nwb");
        hasher.update(
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("")
                .as_bytes(),
        );
        if let Ok(meta) = std::fs::metadata(path) {
            hasher.update(&meta.len().to_le_bytes());
            if let Ok(modified) = meta.modified() {
                if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
                    hasher.update(&duration.as_secs().to_le_bytes());
                    hasher.update(&duration.subsec_nanos().to_le_bytes());
                }
            }
        }
        hasher.finalize().as_bytes().to_vec()
    }

    impl HasGeometry for NwbProvider {
        fn channel_positions(&self) -> &[[f32; 2]] {
            &self.channel_positions
        }
        fn channel_map(&self) -> &[ChannelId] {
            &[]
        }
    }

    /// Walk `/acquisition/*` looking for the first dataset that smells like
    /// an `ElectricalSeries` (has `data` + a rate). Returns
    /// `(sample_rate, n_channels, n_samples, dtype, raw_bytes)`.
    fn load_electrical_series(f: &H5File) -> Result<(f32, u32, SampleIndex, TraceDtype, Vec<u8>)> {
        let acq = f
            .group("/acquisition")
            .context("missing /acquisition group")?;
        let names = acq.member_names().unwrap_or_default();
        for name in names {
            let g = match acq.group(&name) {
                Ok(g) => g,
                Err(_) => continue,
            };
            let Ok(data) = g.dataset("data") else {
                continue;
            };
            let shape = data.shape();
            // ElectricalSeries data is (n_samples, n_channels).
            if shape.len() != 2 {
                continue;
            }
            let n_samples = shape[0] as u64;
            let n_channels = shape[1] as u32;

            // Sample rate: prefer `starting_time/rate` attribute, fall back
            // to a sibling `sampling_rate` dataset / attribute.
            let sample_rate =
                read_rate(&g).ok_or_else(|| anyhow!("ElectricalSeries {name} has no rate"))?;

            // Read the dataset into a typed Vec then cast to bytes. This
            // lets the `hdf5` crate do dtype coercion without us juggling
            // every numpy dtype.
            let dt = data.dtype()?;
            let (dtype, bytes) = if dt.size() == 2 {
                let v: Vec<i16> = data.read_raw()?;
                (TraceDtype::I16, bytemuck::cast_slice(&v).to_vec())
            } else if dt.size() == 4 {
                // Pick float vs int by trying float first.
                if let Ok(v) = data.read_raw::<f32>() {
                    (TraceDtype::F32, bytemuck::cast_slice(&v).to_vec())
                } else {
                    let v: Vec<i32> = data.read_raw()?;
                    (TraceDtype::I32, bytemuck::cast_slice(&v).to_vec())
                }
            } else {
                bail!(
                    "ElectricalSeries {name}: unsupported element size {}",
                    dt.size()
                );
            };
            return Ok((
                sample_rate,
                n_channels,
                SampleIndex::new(n_samples),
                dtype,
                bytes,
            ));
        }
        bail!("no ElectricalSeries (with shape (n_samples, n_channels)) under /acquisition")
    }

    fn read_rate(g: &hdf5::Group) -> Option<f32> {
        // NWB convention: `starting_time` dataset has a `rate` attribute (Hz).
        if let Ok(st) = g.dataset("starting_time") {
            if let Ok(attr) = st.attr("rate") {
                if let Ok(v) = attr.read_scalar::<f32>() {
                    return Some(v);
                }
                if let Ok(v) = attr.read_scalar::<f64>() {
                    return Some(v as f32);
                }
            }
        }
        // Some writers stash a top-level `sampling_rate` attribute instead.
        for name in ["sampling_rate", "rate"] {
            if let Ok(attr) = g.attr(name) {
                if let Ok(v) = attr.read_scalar::<f32>() {
                    return Some(v);
                }
                if let Ok(v) = attr.read_scalar::<f64>() {
                    return Some(v as f32);
                }
            }
        }
        None
    }

    /// Load `/units/spike_times` (vlen) bucketed by unit. NWB stores the
    /// ragged offsets in `/units/spike_times_index`.
    fn load_units(f: &H5File, sample_rate: f32) -> Result<(Vec<Vec<SampleIndex>>, usize)> {
        let units = f.group("/units").context("missing /units")?;
        let times = units
            .dataset("spike_times")
            .context("missing /units/spike_times")?;
        let index = units
            .dataset("spike_times_index")
            .context("missing /units/spike_times_index")?;

        // spike_times is 1-D (concatenated), spike_times_index is 1-D ends.
        let all: Vec<f64> = times.read_raw()?;
        let ends: Vec<u64> = match index.read_raw::<u64>() {
            Ok(v) => v,
            Err(_) => index
                .read_raw::<i64>()?
                .into_iter()
                .map(|x| x as u64)
                .collect(),
        };
        let n_clusters = ends.len();
        let mut out: Vec<Vec<SampleIndex>> = Vec::with_capacity(n_clusters);
        let mut start = 0u64;
        for &end in &ends {
            let slice = &all[start as usize..end as usize];
            let mut bucket: Vec<SampleIndex> = slice
                .iter()
                .map(|&t| SampleIndex::new((t * sample_rate as f64).round() as u64))
                .collect();
            bucket.sort_unstable();
            out.push(bucket);
            start = end;
        }
        Ok((out, n_clusters))
    }

    fn load_electrode_positions(f: &H5File, n_channels: usize) -> Result<Vec<[f32; 2]>> {
        let g = f
            .group("/general/extracellular_ephys/electrodes")
            .context("no electrodes table")?;
        let xs = read_optional_f32_column(&g, &["rel_x", "x"]).unwrap_or_default();
        let ys = read_optional_f32_column(&g, &["rel_y", "y"]).unwrap_or_default();
        let n = xs.len().max(ys.len()).min(n_channels);
        if n == 0 {
            bail!("electrodes table has no x/y columns");
        }
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push([
                xs.get(i).copied().unwrap_or(0.0),
                ys.get(i).copied().unwrap_or(0.0),
            ]);
        }
        Ok(out)
    }

    fn read_optional_f32_column(g: &hdf5::Group, candidates: &[&str]) -> Option<Vec<f32>> {
        for name in candidates {
            if let Ok(ds) = g.dataset(name) {
                if let Ok(v) = ds.read_raw::<f32>() {
                    return Some(v);
                }
                if let Ok(v) = ds.read_raw::<f64>() {
                    return Some(v.into_iter().map(|x| x as f32).collect());
                }
            }
        }
        None
    }
}

#[cfg(feature = "hdf5")]
pub use imp::NwbProvider;
