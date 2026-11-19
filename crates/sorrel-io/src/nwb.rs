//! NWB backend (local files).
//!
//! Reads the canonical NWB extracellular layout:
//!
//! - `/acquisition/<name>` — the first group (by name) whose
//!   `neurodata_type` is `ElectricalSeries` or a subtype declared in the
//!   file's cached `/specifications`, with 2-D `data` (time × channels).
//!   Other acquisitions are ignored; a file without one is rejected.
//!   - Time base: `starting_time` (seconds) and its `rate` attribute. A
//!     series with per-sample `timestamps` instead is rejected.
//!   - Samples: int8, uint8, int16, uint16, int32, uint32, int64, float32 or
//!     float64 `data`, scaled to
//!     microvolts as `data × conversion × channel_conversion[ch] + offset`
//!     (volts × 10⁶). Any other dtype is rejected.
//!   - Channels: the series' `electrodes` region maps each data column to a
//!     row of `/general/extracellular_ephys/electrodes`; that row's `id` is
//!     the channel id and its `rel_x`/`rel_y` the probe position. The `x`/`y`
//!     columns are brain coordinates and are not used as probe geometry.
//! - `/units` — `id`, `spike_times` and `spike_times_index`. The table's own
//!   ids are the unit ids Sorrel reports (`curation.json`, MCP evidence).
//!   Spike times (seconds on the session clock) become sample indices
//!   relative to the series' `starting_time`.
//!
//! Compiled out by default — enable with `--features hdf5`.
//!
//! Limitations:
//! - Local files only; no DANDI streaming.
//! - One series: multiple segments or probes collapse to the first
//!   `ElectricalSeries`.
//! - The trace is read on demand with a bounded cache of float32 µV windows.

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
        appended_unit_id, ChannelId, ClusterId, DataProvider, SampleIndex, TraceSamples, TraceSlice,
    };
    use anyhow::{anyhow, bail, ensure, Context, Result};
    use hdf5_metno as hdf5;
    use hdf5_metno::types::{FloatSize, IntSize, TypeDescriptor, VarLenAscii, VarLenUnicode};
    use hdf5_metno::File as H5File;
    use std::collections::{HashMap, VecDeque};
    use std::path::Path;
    use std::sync::Mutex;

    const ELECTRODES: &str = "/general/extracellular_ephys/electrodes";

    pub struct NwbProvider {
        sample_rate: f32,
        n_channels: u32,
        n_samples: SampleIndex,
        spikes_per_cluster: Vec<Vec<SampleIndex>>,
        unit_ids: Vec<u32>,
        initial_labels: Vec<PhyLabel>,
        channel_positions: Vec<[f32; 2]>,
        channel_map: Vec<ChannelId>,
        identity_bytes: Vec<u8>,
        trace: NwbTrace,
    }

    impl NwbProvider {
        pub fn open(path: impl AsRef<Path>) -> Result<Self> {
            let path = path.as_ref();
            let f = H5File::open(path).with_context(|| format!("open NWB {}", path.display()))?;
            let (name, series) = find_electrical_series(&f)?;
            let data = series.dataset("data")?;
            let shape = data.shape();
            let (n_samples, n_channels) = (shape[0], shape[1]);
            let (rate, t0) = read_time_base(&series, &name)?;
            let (channel_map, channel_positions) = read_channels(&f, &series, &name, n_channels)?;
            let trace = NwbTrace::new(&series, data, &name, n_samples, n_channels)?;
            let (spikes_per_cluster, unit_ids) = load_units(&f, rate, t0).context("load /units")?;
            Ok(Self {
                sample_rate: rate as f32,
                n_channels: n_channels as u32,
                n_samples: SampleIndex::new(n_samples as u64),
                initial_labels: vec![PhyLabel::Unsorted; unit_ids.len()],
                spikes_per_cluster,
                unit_ids,
                channel_positions,
                channel_map,
                identity_bytes: provider_identity_bytes(path),
                trace,
            })
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
            let samples = self.trace.window(start, len).unwrap_or_else(|err| {
                // The trace interface cannot propagate I/O errors. Returning
                // no samples avoids mistaking a partial read for real data.
                log::error!("read NWB trace at sample {start}: {err:#}");
                Vec::new()
            });
            TraceSlice {
                start,
                n_channels: self.n_channels,
                samples: TraceSamples::OwnedF32(samples),
            }
        }

        fn unit_id(&self, cluster: ClusterId) -> u32 {
            appended_unit_id(&self.unit_ids, cluster)
        }

        fn initial_labels(&self) -> Vec<Self::Label> {
            self.initial_labels.clone()
        }

        fn identity_bytes(&self) -> Vec<u8> {
            self.identity_bytes.clone()
        }
    }

    impl HasGeometry for NwbProvider {
        fn channel_positions(&self) -> &[[f32; 2]] {
            &self.channel_positions
        }
        fn channel_map(&self) -> &[ChannelId] {
            &self.channel_map
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

    fn read_str(attr: hdf5::Attribute) -> Option<String> {
        attr.read_scalar::<VarLenUnicode>()
            .map(|s| s.to_string())
            .or_else(|_| attr.read_scalar::<VarLenAscii>().map(|s| s.to_string()))
            .ok()
    }

    /// `neurodata_type_def → neurodata_type_inc` for every type in the
    /// file's cached specifications (core and extensions).
    fn type_parents(f: &H5File) -> HashMap<String, String> {
        fn collect(v: &serde_json::Value, out: &mut HashMap<String, String>) {
            match v {
                serde_json::Value::Object(m) => {
                    if let (Some(def), Some(inc)) = (
                        m.get("neurodata_type_def").and_then(|v| v.as_str()),
                        m.get("neurodata_type_inc").and_then(|v| v.as_str()),
                    ) {
                        out.insert(def.to_owned(), inc.to_owned());
                    }
                    m.values().for_each(|v| collect(v, out));
                }
                serde_json::Value::Array(a) => a.iter().for_each(|v| collect(v, out)),
                _ => {}
            }
        }
        let mut out = HashMap::new();
        let specs = f.group("/specifications").and_then(|g| g.groups());
        for version in specs
            .into_iter()
            .flatten()
            .flat_map(|ns| ns.groups())
            .flatten()
        {
            for ds in version.datasets().into_iter().flatten() {
                let text = ds
                    .read_scalar::<VarLenUnicode>()
                    .map(|s| s.to_string())
                    .or_else(|_| ds.read_scalar::<VarLenAscii>().map(|s| s.to_string()));
                if let Some(v) = text.ok().and_then(|t| serde_json::from_str(&t).ok()) {
                    collect(&v, &mut out);
                }
            }
        }
        out
    }

    /// The first `/acquisition` group typed `ElectricalSeries` (or a
    /// subtype) with 2-D data.
    fn find_electrical_series(f: &H5File) -> Result<(String, hdf5::Group)> {
        let acq = f
            .group("/acquisition")
            .context("missing /acquisition group")?;
        let parents = type_parents(f);
        let is_series = |t: &str| {
            let mut t = t;
            // Bounded walk: a malformed spec could loop.
            for _ in 0..32 {
                if t == "ElectricalSeries" {
                    return true;
                }
                match parents.get(t) {
                    Some(p) => t = p,
                    None => return false,
                }
            }
            false
        };
        let mut seen = Vec::new();
        let mut found = Vec::new();
        for name in acq.member_names()? {
            let Ok(g) = acq.group(&name) else { continue };
            let ty = g.attr("neurodata_type").ok().and_then(read_str);
            let ty = ty.unwrap_or_else(|| "untyped".into());
            let two_d = g.dataset("data").is_ok_and(|d| d.ndim() == 2);
            if is_series(&ty) && two_d {
                found.push((name, g));
            } else {
                seen.push(format!("{name} ({ty})"));
            }
        }
        let mut found = found.into_iter();
        let Some(first) = found.next() else {
            bail!(
                "no ElectricalSeries with 2-D data under /acquisition (found: {})",
                if seen.is_empty() {
                    "nothing".to_owned()
                } else {
                    seen.join(", ")
                }
            );
        };
        let rest: Vec<String> = found.map(|(n, _)| n).collect();
        if !rest.is_empty() {
            log::warn!(
                "NWB: reading ElectricalSeries {}; ignoring {}",
                first.0,
                rest.join(", ")
            );
        }
        Ok(first)
    }

    /// `(rate Hz, starting_time s)`. Only a constant rate is supported.
    fn read_time_base(g: &hdf5::Group, name: &str) -> Result<(f64, f64)> {
        let Ok(st) = g.dataset("starting_time") else {
            if g.link_exists("timestamps") {
                bail!(
                    "ElectricalSeries {name} has per-sample timestamps instead of \
                     starting_time + rate; Sorrel needs a constant sample rate"
                );
            }
            bail!("ElectricalSeries {name} has neither starting_time nor timestamps");
        };
        let rate: f64 = st
            .attr("rate")
            .and_then(|a| a.read_scalar())
            .with_context(|| format!("ElectricalSeries {name}: starting_time has no rate"))?;
        ensure!(
            rate.is_finite() && rate > 0.0,
            "ElectricalSeries {name}: invalid rate {rate}"
        );
        Ok((rate, st.read_scalar()?))
    }

    /// Channel ids and probe positions for each data column, through the
    /// series' `electrodes` region.
    fn read_channels(
        f: &H5File,
        series: &hdf5::Group,
        name: &str,
        n_channels: usize,
    ) -> Result<(Vec<ChannelId>, Vec<[f32; 2]>)> {
        // ponytail: assumes the region points at the canonical electrodes
        // table rather than dereferencing its `table` object reference.
        let rows: Vec<i64> = series
            .dataset("electrodes")
            .and_then(|d| d.read_raw())
            .with_context(|| format!("ElectricalSeries {name}: no electrodes region"))?;
        ensure!(
            rows.len() == n_channels,
            "ElectricalSeries {name}: {} electrodes for {n_channels} data columns",
            rows.len()
        );
        let table = f.group(ELECTRODES).context("missing electrodes table")?;
        let ids: Vec<i64> = table.dataset("id")?.read_raw()?;
        let rows = rows
            .into_iter()
            .map(|r| {
                usize::try_from(r)
                    .ok()
                    .filter(|&r| r < ids.len())
                    .ok_or_else(|| anyhow!("electrodes region row {r} is outside the table"))
            })
            .collect::<Result<Vec<usize>>>()?;
        let map = rows
            .iter()
            .map(|&r| {
                u32::try_from(ids[r])
                    .map(ChannelId)
                    .map_err(|_| anyhow!("electrode id {} does not fit u32", ids[r]))
            })
            .collect::<Result<_>>()?;
        let column = |c: &str| -> Option<Vec<f32>> {
            let v: Vec<f32> = table.dataset(c).and_then(|d| d.read_raw()).ok()?;
            (v.len() == ids.len()).then_some(v)
        };
        let positions = match (column("rel_x"), column("rel_y")) {
            (Some(x), Some(y)) => rows.iter().map(|&r| [x[r], y[r]]).collect(),
            _ => Vec::new(),
        };
        Ok((map, positions))
    }

    /// Each of eight cached blocks targets at most 1 MiB of scaled samples
    /// (or one frame if wider). A caller owns only its requested window.
    const TRACE_CACHE_BLOCKS: usize = 8;
    const TRACE_BLOCK_BYTES: usize = 1024 * 1024;

    struct NwbTrace {
        data: hdf5::Dataset,
        dtype: TypeDescriptor,
        gain: Vec<f64>,
        offset_uv: f64,
        n_samples: usize,
        block_frames: usize,
        // ponytail: one lock through the read to avoid duplicate decompression;
        // split it only if parallel readers become a bottleneck.
        cache: Mutex<VecDeque<(usize, Vec<f32>)>>,
    }

    impl NwbTrace {
        fn new(
            series: &hdf5::Group,
            data: hdf5::Dataset,
            name: &str,
            n_samples: usize,
            n_channels: usize,
        ) -> Result<Self> {
            ensure!(n_channels > 0, "ElectricalSeries {name}: no channels");
            let dtype = data.dtype()?.to_descriptor()?;
            ensure!(
                matches!(
                    dtype,
                    TypeDescriptor::Integer(_)
                        | TypeDescriptor::Unsigned(IntSize::U1 | IntSize::U2 | IntSize::U4)
                        | TypeDescriptor::Float(_)
                ),
                "ElectricalSeries {name}: {dtype} samples are not supported \
                 (expected int8, uint8, int16, uint16, int32, uint32, int64, float32 or float64)"
            );
            let (gain, offset_uv) = trace_scale(series, &data, name, n_channels)?;
            let max_frames = (TRACE_BLOCK_BYTES / 4 / n_channels).max(1);
            let block_frames = data
                .chunk()
                .map_or((max_frames / 4).max(1), |chunk| chunk[0].min(max_frames));
            Ok(Self {
                data,
                dtype,
                gain,
                offset_uv,
                n_samples,
                block_frames,
                cache: Mutex::new(VecDeque::new()),
            })
        }

        fn window(&self, start: SampleIndex, len: u32) -> Result<Vec<f32>> {
            let start = usize::try_from(start.0)
                .unwrap_or(usize::MAX)
                .min(self.n_samples);
            let end = start.saturating_add(len as usize).min(self.n_samples);
            let nc = self.gain.len();
            let mut out = Vec::with_capacity((end - start) * nc);
            let mut t = start;
            while t < end {
                let base = t / self.block_frames * self.block_frames;
                let mut cache = self.cache.lock().expect("NWB trace cache poisoned");
                if let Some(i) = cache.iter().position(|(key, _)| *key == base) {
                    let entry = cache.remove(i).expect("position in cache");
                    cache.push_front(entry);
                } else {
                    if cache.len() == TRACE_CACHE_BLOCKS {
                        cache.pop_back();
                    }
                    let stop = base.saturating_add(self.block_frames).min(self.n_samples);
                    cache.push_front((base, self.read_block(base, stop)?));
                }
                let (_, block) = &cache[0];
                let stop = end.min(base + block.len() / nc);
                out.extend_from_slice(&block[(t - base) * nc..(stop - base) * nc]);
                t = stop;
            }
            Ok(out)
        }

        fn read_block(&self, start: usize, end: usize) -> Result<Vec<f32>> {
            // Read in the stored type: HDF5 conversion can clip (e.g. uint16
            // to int16). Scale in f64 before narrowing, including f64 input.
            macro_rules! read_scaled {
                ($ty:ty) => {{
                    let raw = self.data.read_slice_2d::<$ty, _>((start..end, ..))?;
                    raw.iter()
                        .zip(self.gain.iter().cycle())
                        .map(|(&v, g)| (v as f64 * g + self.offset_uv) as f32)
                        .collect()
                }};
            }
            Ok(match self.dtype {
                TypeDescriptor::Integer(IntSize::U1) => read_scaled!(i8),
                TypeDescriptor::Unsigned(IntSize::U1) => read_scaled!(u8),
                TypeDescriptor::Integer(IntSize::U2) => read_scaled!(i16),
                TypeDescriptor::Unsigned(IntSize::U2) => read_scaled!(u16),
                TypeDescriptor::Integer(IntSize::U4) => read_scaled!(i32),
                TypeDescriptor::Unsigned(IntSize::U4) => read_scaled!(u32),
                TypeDescriptor::Integer(IntSize::U8) => read_scaled!(i64),
                TypeDescriptor::Float(FloatSize::U4) => read_scaled!(f32),
                TypeDescriptor::Float(FloatSize::U8) => read_scaled!(f64),
                _ => unreachable!("dtype checked on open"),
            })
        }
    }

    /// `data × conversion × channel_conversion + offset` volts, times 10⁶.
    fn trace_scale(
        series: &hdf5::Group,
        data: &hdf5::Dataset,
        name: &str,
        n_channels: usize,
    ) -> Result<(Vec<f64>, f64)> {
        if let Some(unit) = data.attr("unit").ok().and_then(read_str) {
            ensure!(
                unit.eq_ignore_ascii_case("volts"),
                "ElectricalSeries {name}: data unit is {unit:?}, expected volts"
            );
        }
        let scalar = |attr: &str, default: f64| -> Result<f64> {
            match data.attr(attr) {
                Ok(a) => Ok(a.read_scalar()?),
                Err(_) => Ok(default),
            }
        };
        let conversion = scalar("conversion", 1.0)?;
        let offset_uv = scalar("offset", 0.0)? * 1e6;
        let channel_conversion: Vec<f64> = match series.dataset("channel_conversion") {
            Ok(d) => d.read_raw()?,
            Err(_) => vec![1.0; n_channels],
        };
        ensure!(
            channel_conversion.len() == n_channels,
            "ElectricalSeries {name}: channel_conversion has {} entries for {n_channels} channels",
            channel_conversion.len()
        );
        let gain = channel_conversion
            .iter()
            .map(|c| c * conversion * 1e6)
            .collect();
        Ok((gain, offset_uv))
    }

    /// Spike sample indices per unit (relative to `t0`) and the units
    /// table's ids.
    fn load_units(f: &H5File, rate: f64, t0: f64) -> Result<(Vec<Vec<SampleIndex>>, Vec<u32>)> {
        let units = f.group("/units").context("missing /units")?;
        let ids: Vec<i64> = units
            .dataset("id")
            .context("missing /units/id")?
            .read_raw()?;
        let all: Vec<f64> = units
            .dataset("spike_times")
            .context("missing /units/spike_times")?
            .read_raw()?;
        let ends: Vec<i64> = units
            .dataset("spike_times_index")
            .context("missing /units/spike_times_index")?
            .read_raw()?;
        ensure!(
            ids.len() == ends.len(),
            "/units/id has {} rows but spike_times_index has {}",
            ids.len(),
            ends.len()
        );
        let unit_ids = ids
            .iter()
            .map(|&id| u32::try_from(id).map_err(|_| anyhow!("unit id {id} does not fit u32")))
            .collect::<Result<Vec<u32>>>()?;
        let mut sorted = unit_ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        ensure!(sorted.len() == unit_ids.len(), "/units/id has duplicates");

        let mut out = Vec::with_capacity(ends.len());
        let mut start = 0usize;
        for (&end, &id) in ends.iter().zip(&unit_ids) {
            let end = usize::try_from(end)
                .ok()
                .filter(|&e| e >= start && e <= all.len())
                .ok_or_else(|| anyhow!("/units/spike_times_index is not a valid ragged index"))?;
            let mut bucket = all[start..end]
                .iter()
                .map(|&t| {
                    let s = ((t - t0) * rate).round();
                    ensure!(
                        s.is_finite() && s >= 0.0,
                        "unit {id}: spike at {t} s precedes the recording's starting_time {t0} s"
                    );
                    Ok(SampleIndex::new(s as u64))
                })
                .collect::<Result<Vec<_>>>()?;
            bucket.sort_unstable();
            out.push(bucket);
            start = end;
        }
        Ok((out, unit_ids))
    }

    #[cfg(test)]
    #[test]
    fn lazy_window_matches_eager_scaling_across_chunks_and_end() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../release/fixtures/nwb/good.nwb");
        let p = NwbProvider::open(path).unwrap();
        let trace = &p.trace;
        assert_eq!(trace.block_frames, 32);
        assert!(trace.cache.lock().unwrap().is_empty());
        let eager: Vec<f32> = trace
            .data
            .read_raw::<i16>()
            .unwrap()
            .into_iter()
            .zip(trace.gain.iter().cycle())
            .map(|(v, gain)| (v as f64 * gain + trace.offset_uv) as f32)
            .collect();
        let slice = p.trace(SampleIndex(29), 300);
        assert_eq!(slice.start, SampleIndex(29));
        assert_eq!(slice.n_channels, 4);
        assert_eq!(slice.samples.to_f32_centred(), eager[29 * 4..]);
        assert_eq!(trace.cache.lock().unwrap().len(), TRACE_CACHE_BLOCKS);
        assert_eq!(
            p.trace(SampleIndex(290), 10).samples.to_f32_centred(),
            eager[290 * 4..]
        );
        assert!(p.trace(SampleIndex(u64::MAX), 1).samples.is_empty());
    }
}

#[cfg(feature = "hdf5")]
pub use imp::NwbProvider;
