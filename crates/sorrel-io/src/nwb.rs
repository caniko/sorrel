//! NWB backend (local files).
//!
//! Reads `/units` and, optionally, one ElectricalSeries from `/acquisition`,
//! `/processing` or a second file. Uses a regular time base and reads trace
//! windows on demand as float32 µV. Enable with `--features hdf5` or `--features zarr`.

/// Recognize an NWB-Zarr directory by its name or root NWB version attribute.
pub fn is_zarr_path(path: &std::path::Path) -> bool {
    path.is_dir()
        && (path.file_name().is_some_and(|name| {
            name.to_string_lossy()
                .to_ascii_lowercase()
                .ends_with(".nwb.zarr")
        }) || std::fs::read(path.join(".zattrs"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
            .is_some_and(|attrs| attrs.get("nwb_version").is_some()))
}

#[cfg(not(any(feature = "hdf5", feature = "zarr")))]
mod stub {
    use anyhow::{bail, Result};
    use std::path::Path;

    pub fn open_stub(_path: impl AsRef<Path>) -> Result<()> {
        bail!(
            "NWB backend not built; rebuild sorrel-io with `--features hdf5` or `--features zarr`, \
             or export the units table to phy \
             via SpikeInterface and open that directory instead"
        )
    }
}

#[cfg(not(any(feature = "hdf5", feature = "zarr")))]
pub use stub::open_stub;

#[cfg(any(feature = "hdf5", feature = "zarr"))]
mod imp {
    mod storage;
    use crate::extras::{HasGeometry, HasQualityMetrics, QualityMetrics};
    use crate::kilosort::{pinned_sorter, valid_tag, PhyLabel};
    use crate::provider::{
        appended_unit_id, ChannelId, ClusterId, DataProvider, SampleIndex, TraceSamples,
        TraceSlice, UnitAnnotation,
    };
    use anyhow::{anyhow, ensure, Context, Result};
    use std::cmp::Reverse;
    use std::collections::{BinaryHeap, HashMap, VecDeque};
    use std::ops::Range;
    use std::path::Path;
    use std::sync::Mutex;
    use storage::{Dataset, Group, SampleType};

    const ELECTRODES: &str = "/general/extracellular_ephys/electrodes";

    #[derive(Clone, Copy, Debug, Default)]
    pub struct NwbOpenParams<'a> {
        /// Acquisition name or absolute NWB path, also applied to a paired file.
        pub series: Option<&'a str>,
        /// Read the ElectricalSeries from this file instead of the units file.
        pub raw_path: Option<&'a Path>,
        /// Exact declared clock (Hz), or a supplied rate for units without a time base.
        pub sample_rate: Option<f64>,
    }

    pub struct NwbProvider {
        rate: f64,
        t0: f64,
        n_channels: u32,
        n_samples: SampleIndex,
        spikes_per_cluster: Vec<Vec<SampleIndex>>,
        unit_ids: Vec<u32>,
        metadata: UnitMetadata,
        channel_positions: Vec<[f32; 2]>,
        channel_map: Vec<ChannelId>,
        identity_bytes: Vec<u8>,
        trace: Option<NwbTrace>,
    }

    impl NwbProvider {
        pub fn open(path: impl AsRef<Path>) -> Result<Self> {
            Self::open_with(path, NwbOpenParams::default())
        }

        pub fn open_with(path: impl AsRef<Path>, params: NwbOpenParams<'_>) -> Result<Self> {
            let path = path.as_ref();
            let f = Group::open(path)?;
            let spike_times = f.stored_dataset("/units/spike_times")?;
            let spike_times_index = f.stored_dataset("/units/spike_times_index")?;
            let ids = f.stored_dataset("/units/id")?;
            if let Some(target) = spike_times_index.reference_path("target")? {
                ensure!(
                    target == "/units/spike_times",
                    "invalid /units/spike_times_index target {target:?}"
                );
            }
            let units = f
                .group("/units")
                .context("missing /units in the sorting NWB file")?;
            let raw = params.raw_path.map(Group::open).transpose()?;
            let recording = raw.as_ref().unwrap_or(&f);
            let series = find_electrical_series(recording, params.series, raw.is_some())?;
            // In a single file with a series, an invalid resolution value is
            // not a clock statement; its storage type must still be exact.
            let units_rate = read_units_rate(&spike_times, raw.is_some() || series.is_none())?;

            let mut legacy_identity = false;
            let mut comparison_rate = None;
            let (trace, rate, t0, (channel_map, channel_positions), timestamps) = if let Some((
                name,
                series,
            )) = &series
            {
                let data = series.dataset("data")?;
                let shape = data.shape();
                let (n_samples, n_channels) = (shape[0], shape.get(1).copied().unwrap_or(1));
                let (stored_rate, t0, timestamps) =
                    read_time_base(recording, name, n_samples)?;
                let units_clock = if let Some(declared) = units_rate {
                    // The one-ppm gate decides whether resolution declares a clock,
                    // not whether an explicit value matches that clock.
                    let close = (declared - stored_rate).abs() <= 1e-6 * stored_rate;
                    ensure!(close || raw.is_none(),
                        "NWB rate mismatch: sorting sample rate is {declared} Hz but {name} is {stored_rate} Hz");
                    close.then_some(declared)
                } else {
                    None
                };
                let rate = params.sample_rate.unwrap_or(stored_rate);
                if rate != stored_rate && Some(rate) != units_clock {
                    let source = if timestamps.is_some() {
                        "timestamps"
                    } else {
                        "starting_time.rate"
                    };
                    let units = units_clock.map_or_else(String::new, |units| {
                        format!(", {units} Hz (/units/spike_times resolution)")
                    });
                    anyhow::bail!("--sample-rate {rate} Hz is not a clock this NWB declares; pass one of: {stored_rate} Hz (series {name} {source}){units}");
                }
                legacy_identity = raw.is_none()
                    && timestamps.is_none()
                    && f.supports_legacy_seal()
                    && is_trunk_series(&f, name);
                comparison_rate = if params.sample_rate.is_none() {
                    units_clock.filter(|&units| units != stored_rate)
                } else {
                    (legacy_identity && rate != stored_rate).then_some(stored_rate)
                };
                let channels = read_channels(recording, series, name, n_channels)?;
                let trace = NwbTrace::new(series, data, name, n_samples, n_channels)?;
                (Some(trace), rate, t0, channels, timestamps)
            } else {
                if let (Some(explicit), Some(declared)) = (params.sample_rate, units_rate) {
                    ensure!(explicit == declared,
                        "--sample-rate {explicit} Hz is not a clock this NWB declares; pass one of: {declared} Hz (/units/spike_times resolution)");
                }
                (
                        None,
                        units_rate.or(params.sample_rate).map(checked_rate).transpose()?.context(
                            "no ElectricalSeries time base in the sorting NWB and no /units/spike_times resolution; pass --sample-rate HZ",
                        )?,
                        0.0,
                        read_units_only_channels(&f)?,
                        None,
                    )
            };
            if let Some(raw) = &raw {
                let units_epoch =
                    timestamp_epoch(&f).context("sorting NWB timestamps_reference_time")?;
                let raw_epoch =
                    timestamp_epoch(raw).context("paired raw NWB timestamps_reference_time")?;
                ensure!(units_epoch == raw_epoch,
                    "NWB timestamps_reference_time mismatch: sorting NWB uses {units_epoch}, paired raw NWB uses {raw_epoch}");
            }
            let n_samples = trace.as_ref().map_or(0, |t| t.n_samples);
            let require_agreement = params.sample_rate.is_none();
            let (spikes_per_cluster, unit_ids, same_mapping) = load_units(
                &ids,
                &spike_times,
                &spike_times_index,
                rate,
                t0,
                timestamps.as_ref().map(|ds| (ds, n_samples)),
                comparison_rate.map(|other_rate| (other_rate, require_agreement)),
            )
            .context("load /units")?;
            if raw.is_some() {
                ensure!(spikes_per_cluster.iter().filter_map(|s| s.last()).all(|s| s.0 < n_samples as u64),
                    "spike times extend past the paired ElectricalSeries (starting time {t0} s, {n_samples} samples at {rate} Hz)");
            }
            let mut identity_bytes = f.identity(path)?;
            if !(legacy_identity && same_mapping) {
                if let Some((name, _)) = &series {
                    identity_bytes.extend_from_slice(name.as_bytes());
                }
                identity_bytes.push(0); // Terminate the path (empty for units-only).
                identity_bytes.extend(rate.to_le_bytes());
                if let Some((raw, raw_path)) = raw.as_ref().zip(params.raw_path) {
                    identity_bytes.extend(raw.identity(raw_path)?);
                }
            }
            let metadata = read_unit_metadata(&units, &unit_ids)?;
            Ok(Self {
                rate,
                t0,
                n_channels: u32::try_from(channel_map.len()).context("too many NWB channels")?,
                n_samples: SampleIndex(n_samples as u64),
                metadata,
                spikes_per_cluster,
                unit_ids,
                channel_positions,
                channel_map,
                identity_bytes,
                trace,
            })
        }

        /// First sample time in seconds on the NWB session clock.
        pub fn starting_time(&self) -> f64 {
            self.t0
        }
    }

    impl DataProvider for NwbProvider {
        type Label = PhyLabel;

        fn sample_rate(&self) -> f32 {
            self.rate as f32
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
            let samples = self
                .trace
                .as_ref()
                .map(|trace| trace.window(start, len))
                .transpose()
                .unwrap_or_else(|err| {
                    // The trace interface cannot propagate I/O errors. Returning
                    // no samples avoids mistaking a partial read for real data.
                    log::error!("read NWB trace at sample {start}: {err:#}");
                    None
                });
            TraceSlice {
                start,
                n_channels: self.n_channels,
                samples: samples.map_or(TraceSamples::F32(&[]), TraceSamples::OwnedF32),
            }
        }

        fn unit_id(&self, cluster: ClusterId) -> u32 {
            appended_unit_id(&self.unit_ids, cluster)
        }

        fn unit_id_json(&self, cluster: ClusterId) -> serde_json::Value {
            let Some(names) = &self.metadata.names else {
                return self.unit_id(cluster).into();
            };
            if let Some(name) = names.get(cluster.idx()) {
                return name.as_str().into();
            }
            // Like SortingAnalyzer, keep appended string IDs clear of any
            // upstream integer-like names as well as the table's integer IDs.
            let offset = cluster.idx() - names.len();
            let max = self
                .unit_ids
                .iter()
                .copied()
                .chain(names.iter().filter_map(|n| n.parse().ok()))
                .max()
                .unwrap_or(0);
            (u64::from(max) + 1 + offset as u64).to_string().into()
        }

        fn initial_labels(&self) -> Vec<Self::Label> {
            self.metadata.labels.clone()
        }

        fn initial_tags(&self) -> Vec<Vec<String>> {
            self.metadata.tags.clone()
        }

        fn initial_annotations(&self) -> Vec<UnitAnnotation> {
            self.metadata.annotations.clone()
        }

        fn sorter(&self) -> Option<String> {
            pinned_sorter(&self.metadata.annotations)
        }

        fn identity_bytes(&self) -> Vec<u8> {
            self.identity_bytes.clone()
        }
    }

    fn checked_rate(rate: f64) -> Result<f64> {
        ensure!(
            (rate as f32).is_finite() && (rate as f32) > 0.0,
            "invalid sample rate {rate} Hz"
        );
        Ok(rate)
    }

    fn read_units_rate(spike_times: &Dataset, strict: bool) -> Result<Option<f64>> {
        let Some(attr) = spike_times.optional_attr("resolution")? else {
            return Ok(None);
        };
        match attr.exact_number("/units/spike_times resolution")? {
            -1.0 => Ok(None), // Unspecified in the NWB schema.
            resolution => match checked_rate(resolution.recip()) {
                Err(_) if !strict => Ok(None),
                rate => rate
                    .map(Some)
                    .context("invalid /units/spike_times resolution"),
            },
        }
    }

    fn timestamp_epoch(file: &Group) -> Result<jiff::Timestamp> {
        // NWB defines time zero separately from the experiment's start time.
        let ds = file.dataset("/timestamps_reference_time")?;
        let value = ds.read_string()?;
        value
            .parse()
            .map_err(|err| anyhow!("invalid NWB epoch {value:?}: {err}"))
    }

    impl HasGeometry for NwbProvider {
        fn channel_positions(&self) -> &[[f32; 2]] {
            &self.channel_positions
        }
        fn channel_map(&self) -> &[ChannelId] {
            &self.channel_map
        }
    }

    impl HasQualityMetrics for NwbProvider {
        fn quality_metrics(&self) -> &QualityMetrics {
            &self.metadata.metrics
        }
    }

    pub(crate) fn read_trials(path: &Path) -> Result<crate::Events> {
        let f = Group::open(path)?;
        let g = f
            .group("/intervals/trials")
            .with_context(|| format!("{}: no /intervals/trials table", path.display()))?;
        let starts: Vec<f64> = g.dataset("start_time")?.read_raw()?;
        let mut names = None;
        for column in ["trial_type", "stimulus", "condition"] {
            if let Some(ds) = g.optional_dataset(column)? {
                if let Some(values) = ds.strings().ok().filter(|v| v.len() == starts.len()) {
                    names = Some(values);
                    break;
                }
            }
        }
        let mut events = crate::Events::default();
        for (i, t) in starts.into_iter().enumerate() {
            events.push(t, names.as_ref().map_or("trial", |n| n[i].as_str()));
        }
        Ok(events)
    }

    /// `neurodata_type_def → neurodata_type_inc` for every type in the
    /// file's cached specifications (core and extensions).
    fn type_parents(f: &Group, legacy: bool) -> HashMap<String, String> {
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
                let text = ds.spec_text(legacy);
                if let Some(v) = text.ok().and_then(|t| serde_json::from_str(&t).ok()) {
                    collect(&v, &mut out);
                }
            }
        }
        out
    }

    fn is_electrical_series(ty: &str, parents: &HashMap<String, String>) -> bool {
        let mut ty = ty;
        // Bounded walk: a malformed spec could loop.
        for _ in 0..32 {
            if ty == "ElectricalSeries" {
                return true;
            }
            match parents.get(ty) {
                Some(parent) => ty = parent,
                None => return false,
            }
        }
        false
    }

    /// Mirror trunk's discovery for the journal seal, including the old
    /// variable-length-only decoder for both type tags and cached specs.
    fn is_trunk_series(f: &Group, selected: &str) -> bool {
        let Some(name) = selected.strip_prefix("/acquisition/") else {
            return false;
        };
        let Ok(acq) = f.group("/acquisition") else {
            return false;
        };
        let Ok(names) = acq.member_names() else {
            return false;
        };
        let parents = type_parents(f, true);
        names
            .into_iter()
            .find(|name| {
                let Ok(g) = acq.group(name) else { return false };
                g.attr("neurodata_type")
                    .ok()
                    .and_then(|a| a.legacy_string().ok())
                    .is_some_and(|t| is_electrical_series(&t, &parents))
                    && g.dataset("data").is_ok_and(|d| d.ndim() == 2)
            })
            .is_some_and(|first| first == name)
    }

    fn find_electrical_series(
        f: &Group,
        requested: Option<&str>,
        required: bool,
    ) -> Result<Option<(String, Group)>> {
        fn collect(
            group: &Group,
            path: &str,
            depth: usize,
            is_series: &impl Fn(&str) -> bool,
            found: &mut Vec<(String, Group)>,
        ) -> Result<()> {
            for name in group.member_names()? {
                let Some(child) = group.child_group(&name)? else {
                    continue;
                };
                let path = format!("{path}/{name}");
                if child
                    .optional_attr("neurodata_type")?
                    .map(|a| a.string())
                    .transpose()?
                    .is_some_and(|t| is_series(&t))
                {
                    found.push((path, child));
                } else if depth > 0 {
                    // Bound the recursion in case of cyclic HDF5 links.
                    collect(&child, &path, depth - 1, is_series, found)?;
                }
            }
            Ok(())
        }

        let parents = type_parents(f, false);
        let is_series = |t: &str| is_electrical_series(t, &parents);
        let mut found = Vec::new();
        for (root, depth) in [("/acquisition", 0), ("/processing", 32)] {
            if let Some(group) = f.optional_group(root)? {
                collect(&group, root, depth, &is_series, &mut found)?;
            }
        }
        found.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let data_rank = |g: &Group| -> Result<usize> {
            Ok(
                match g.optional_dataset("data")?.map(|d| d.shape()).as_deref() {
                    Some([_]) => 1,
                    Some([_, channels]) if *channels > 0 => 2,
                    _ => 0,
                },
            )
        };
        let available = || {
            if found.is_empty() {
                "none".to_string()
            } else {
                found
                    .iter()
                    .map(|(path, _)| path.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        };
        let index = if let Some(requested) = requested {
            let path = if requested.starts_with('/') {
                requested.to_owned()
            } else {
                format!("/acquisition/{requested}")
            };
            if let Some(index) = found.iter().position(|(name, _)| *name == path) {
                index
            } else {
                // A selected path outside the discovery walk can still contain a link.
                f.optional_group(&path)?;
                anyhow::bail!(
                    "no ElectricalSeries matching {requested:?}; available series: {}",
                    available()
                );
            }
        } else if found.is_empty() {
            ensure!(
                !required,
                "no ElectricalSeries in the raw NWB; available series: none"
            );
            return Ok(None);
        } else {
            let mut best = (0, 0);
            for (i, (path, group)) in found.iter().enumerate() {
                let rank = data_rank(group)
                    .with_context(|| format!("ElectricalSeries {path}: invalid data"))?;
                if rank > best.0 {
                    best = (rank, i);
                }
            }
            best.1
        };
        let (path, group) = &found[index];
        ensure!(
            data_rank(group).with_context(|| format!("ElectricalSeries {path}: invalid data"))? != 0,
            "ElectricalSeries {path} requires 1-D data (one channel) or 2-D data (time × channels); available series: {}",
            available()
        );
        if requested.is_none() && found.len() > 1 {
            log::warn!(
                "NWB: reading ElectricalSeries {path}; available series: {} (choose with --nwb-series)",
                available()
            );
        }
        Ok(Some(found.swap_remove(index)))
    }

    /// `(rate Hz, starting_time s, recorded timestamps)`.
    fn read_time_base(
        file: &Group,
        name: &str,
        n_samples: usize,
    ) -> Result<(f64, f64, Option<Dataset>)> {
        let read = || {
            if let Some(ds) = file.optional_stored_dataset(&format!("{name}/timestamps"))? {
                ds.exact_float("timestamps")?;
                let (rate, t0) = regular_timestamps(&ds, n_samples)?;
                return Ok((rate, t0, Some(ds)));
            }
            let st = file
                .optional_stored_dataset(&format!("{name}/starting_time"))?
                .context("neither starting_time nor timestamps")?;
            let rate = st.attr("rate").context("starting_time has no rate")?;
            st.exact_float("starting_time")?;
            let rate = checked_rate(rate.exact_number("starting_time rate")?)?;
            let t0: f64 = st.read_scalar()?;
            ensure!(t0.is_finite(), "invalid starting_time {t0}");
            Ok((rate, t0, None))
        };
        read().with_context(|| format!("ElectricalSeries {name}: invalid time base"))
    }

    const TIMESTAMP_BLOCK: usize = 65_536;

    fn read_timestamp_block(ds: &Dataset, range: Range<usize>) -> Result<Vec<f64>> {
        ds.read_rows(range)
    }

    fn regular_timestamps(ds: &Dataset, n: usize) -> Result<(f64, f64)> {
        ensure!(
            ds.shape() == [n] && n >= 2,
            "timestamps must contain one time per sample and at least two times to infer a rate"
        );
        let t0 = read_timestamp_block(ds, 0..1)?[0];
        let last = read_timestamp_block(ds, n - 1..n)?[0];
        // Use the end-to-end mean period, so a tiny bias from one rounded
        // interval cannot accumulate over a long recording.
        let period = (last - t0) / (n - 1) as f64;
        let rate =
            checked_rate(period.recip()).context("irregular timestamps: invalid mean interval")?;
        let scale = t0.abs().max(last.abs());
        let ulp = if scale == f64::MAX {
            scale - scale.next_down()
        } else {
            scale.next_up() - scale
        };
        // A 0.01-sample ULP caps the tolerance at 0.041 samples, well below
        // half a sample, and allows 30 kHz Unix-epoch timestamps (~0.0072).
        ensure!(ulp * rate <= 0.01,
            "timestamps are too coarse at this magnitude to verify regular sampling; if sampling was regular, rewrite the series with starting_time + rate");
        // The grid interpolates two rounded endpoints. Allow a few ulps
        // for their error, each rounded timestamp, and the arithmetic.
        let limit = 1e-3 + 4.0 * ulp * rate;
        let (mut max_drift, mut worst_sample) = (0.0, 0);
        let mut prev = t0;
        for start in (1..n).step_by(TIMESTAMP_BLOCK) {
            let times =
                read_timestamp_block(ds, start..start.saturating_add(TIMESTAMP_BLOCK).min(n))?;
            for (i, &t) in times.iter().enumerate() {
                let sample = start + i;
                let step = t - prev;
                ensure!(step.is_finite() && step > 0.0,
                    "irregular timestamps at sample {sample}: interval {step} s; timestamps must strictly increase");
                let drift = ((t - t0) / period - sample as f64).abs();
                if drift > max_drift {
                    (max_drift, worst_sample) = (drift, sample);
                }
                prev = t;
            }
        }
        ensure!(max_drift <= limit,
            "irregular timestamps at sample {worst_sample}: max deviation {max_drift:.6} samples from the inferred grid (limit {limit:.6})");
        Ok((rate, t0))
    }

    /// Channel ids and probe positions for each data column, through the
    /// series' `electrodes` region.
    type Channels = (Vec<ChannelId>, Vec<[f32; 2]>);

    fn read_channels(f: &Group, series: &Group, name: &str, n_channels: usize) -> Result<Channels> {
        let region = series
            .dataset("electrodes")
            .with_context(|| format!("ElectricalSeries {name}: no electrodes region"))?;
        let rows: Vec<i64> = region.read_raw()?;
        ensure!(
            rows.len() == n_channels,
            "ElectricalSeries {name}: {} electrodes for {n_channels} data columns",
            rows.len()
        );
        let path = region.reference_path("table")?;
        let table = f
            .group(path.as_deref().unwrap_or(ELECTRODES))
            .context("missing electrodes table")?;
        read_electrodes(&table, Some(rows))
    }

    fn read_units_only_channels(f: &Group) -> Result<Channels> {
        match f.optional_group(ELECTRODES)? {
            Some(table) => read_electrodes(&table, None),
            None => Ok(Channels::default()),
        }
    }

    fn read_electrodes(table: &Group, rows: Option<Vec<i64>>) -> Result<Channels> {
        let ids: Vec<i64> = table.dataset("id")?.read_raw()?;
        let rows = if let Some(rows) = rows {
            rows.into_iter()
                .map(|r| {
                    usize::try_from(r)
                        .ok()
                        .filter(|&r| r < ids.len())
                        .ok_or_else(|| anyhow!("electrodes region row {r} is outside the table"))
                })
                .collect::<Result<Vec<usize>>>()?
        } else {
            (0..ids.len()).collect()
        };
        let map = rows
            .iter()
            .map(|&r| {
                u32::try_from(ids[r])
                    .map(ChannelId)
                    .map_err(|_| anyhow!("electrode id {} does not fit u32", ids[r]))
            })
            .collect::<Result<_>>()?;
        let column = |c: &str| -> Result<Option<Vec<f32>>> {
            let Some(ds) = table.optional_dataset(c)? else {
                return Ok(None);
            };
            let values: Vec<f32> = ds
                .read_raw()
                .with_context(|| format!("read electrodes {c}"))?;
            ensure!(
                values.len() == ids.len(),
                "electrodes {c} length does not match id"
            );
            Ok(Some(values))
        };
        let positions = match (column("rel_x")?, column("rel_y")?) {
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
        data: Dataset,
        dtype: SampleType,
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
            series: &Group,
            data: Dataset,
            name: &str,
            n_samples: usize,
            n_channels: usize,
        ) -> Result<Self> {
            ensure!(n_channels > 0, "ElectricalSeries {name}: no channels");
            let dtype = data
                .sample_type()
                .with_context(|| format!("ElectricalSeries {name}: unsupported sample datatype"))?;
            ensure!(
                matches!(
                    dtype,
                    SampleType::I8
                        | SampleType::U8
                        | SampleType::I16
                        | SampleType::U16
                        | SampleType::I32
                        | SampleType::U32
                        | SampleType::I64
                        | SampleType::F32
                        | SampleType::F64
                ),
                "ElectricalSeries {name}: {} samples are not supported \
                 (expected int8, uint8, int16, uint16, int32, uint32, int64, float32 or float64)",
                dtype.name()
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
                    let raw = self.data.read_rows::<$ty>(start..end)?;
                    raw.into_iter()
                        .zip(self.gain.iter().cycle())
                        .map(|(v, g)| (v as f64 * g + self.offset_uv) as f32)
                        .collect()
                }};
            }
            Ok(match self.dtype {
                SampleType::I8 => read_scaled!(i8),
                SampleType::U8 => read_scaled!(u8),
                SampleType::I16 => read_scaled!(i16),
                SampleType::U16 => read_scaled!(u16),
                SampleType::I32 => read_scaled!(i32),
                SampleType::U32 => read_scaled!(u32),
                SampleType::I64 => read_scaled!(i64),
                SampleType::F32 => read_scaled!(f32),
                SampleType::F64 => read_scaled!(f64),
                _ => unreachable!("dtype checked on open"),
            })
        }
    }

    /// `data × conversion × channel_conversion + offset` volts, times 10⁶.
    fn trace_scale(
        series: &Group,
        data: &Dataset,
        name: &str,
        n_channels: usize,
    ) -> Result<(Vec<f64>, f64)> {
        if let Some(unit) = data.optional_attr("unit")? {
            let unit = unit.string()?;
            ensure!(
                unit.eq_ignore_ascii_case("volts"),
                "ElectricalSeries {name}: data unit is {unit:?}, expected volts"
            );
        }
        let scalar = |attr: &str, default: f64| -> Result<f64> {
            match data.optional_attr(attr)? {
                Some(a) => a
                    .number()
                    .with_context(|| format!("ElectricalSeries {name}: invalid {attr}")),
                None => Ok(default),
            }
        };
        let conversion = scalar("conversion", 1.0)?;
        let offset_uv = scalar("offset", 0.0)? * 1e6;
        let channel_conversion: Vec<f64> = match series.optional_dataset("channel_conversion")? {
            Some(d) => d.read_raw().context("read channel_conversion")?,
            None => vec![1.0; n_channels],
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

    /// Spike sample indices per unit (relative to `t0`), unit ids, and
    /// whether the mapping agrees with the optional comparison rate. Any
    /// disagreement is an error when there is no explicit clock choice.
    fn load_units(
        ids: &Dataset,
        spike_times: &Dataset,
        spike_times_index: &Dataset,
        rate: f64,
        t0: f64,
        timestamps: Option<(&Dataset, usize)>,
        mut comparison: Option<(f64, bool)>,
    ) -> Result<(Vec<Vec<SampleIndex>>, Vec<u32>, bool)> {
        let ids: Vec<i64> = ids.read_raw()?;
        spike_times.exact_float("/units/spike_times")?;
        let mut all: Vec<f64> = spike_times.read_raw()?;
        let ends: Vec<i64> = spike_times_index.read_raw()?;
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
        let mut n_differing = 0;
        let mut first_difference = None;
        for (&end, &id) in ends.iter().zip(&unit_ids) {
            let end = usize::try_from(end)
                .ok()
                .filter(|&e| e >= start && e <= all.len())
                .ok_or_else(|| anyhow!("/units/spike_times_index is not a valid ragged index"))?;
            let times = &mut all[start..end];
            if timestamps.is_some() && !times.is_sorted() {
                times.sort_unstable_by(f64::total_cmp);
            }
            let mut bucket = times
                .iter()
                .enumerate()
                .map(|(spike, &t)| {
                    let s = exact_sample(t, t0, rate);
                    ensure!(s.is_finite() && (0.0..u64::MAX as f64).contains(&s)
                        && timestamps.is_none_or(|(_, n_samples)| (s as u64) < n_samples as u64),
                        "unit {id}: spike at {t} s is outside the sample index range or cannot be placed on a sample exactly (starting time {t0} s, {rate} Hz)");
                    if let Some((other_rate, require_agreement)) = comparison {
                        let other = exact_sample(t, t0, other_rate);
                        if other != s {
                            n_differing += 1;
                            if require_agreement {
                                first_difference.get_or_insert((other_rate, id, spike, t, s, other));
                            } else {
                                // One changed sample already rules out the legacy seal.
                                comparison = None;
                            }
                        }
                    }
                    Ok(SampleIndex::new(s as u64))
                })
                .collect::<Result<Vec<_>>>()?;
            if timestamps.is_none() {
                bucket.sort_unstable();
            }
            out.push(bucket);
            start = end;
        }
        if let Some((ds, n_samples)) = timestamps {
            check_timestamp_spikes(ds, n_samples, &all, &unit_ids, &out)?;
        }
        if let Some((units_rate, id, spike, t, stored, precise)) = first_difference {
            let noun = if n_differing == 1 { "spike" } else { "spikes" };
            anyhow::bail!(
                "Ambiguous NWB sample rate: stored series rate {rate} Hz and Units resolution rate {units_rate} Hz map {n_differing} {noun} to different samples (first: unit {id} spike {spike} at {t} s, sample indices {stored} vs {precise}, respectively); pass --sample-rate <Hz> to choose which clock applies"
            );
        }
        Ok((out, unit_ids, n_differing == 0))
    }

    fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let bb = s - a;
        (s, (a - (s - bb)) + (b - bb))
    }

    /// Sign of the exact sum of `terms` (Shewchuk's grow-expansion: the
    /// components stay non-overlapping in increasing magnitude, so the last
    /// nonzero one carries the sign).
    fn exact_sign(terms: &[f64]) -> std::cmp::Ordering {
        let mut expansion: Vec<f64> = Vec::with_capacity(terms.len());
        for &term in terms {
            let mut q = term;
            let mut next = Vec::with_capacity(expansion.len() + 1);
            for &e in &expansion {
                let (s, err) = two_sum(q, e);
                if err != 0.0 {
                    next.push(err);
                }
                q = s;
            }
            next.push(q);
            expansion = next;
        }
        let top = expansion.iter().rev().find(|&&x| x != 0.0).copied();
        top.map_or(std::cmp::Ordering::Equal, |x| x.total_cmp(&0.0))
    }

    /// `floor((t - t0) × rate + 0.5)` of the exact real product, not of its
    /// f64 rounding: near a half-sample boundary the rounded product can land
    /// on the wrong side. The difference is split exactly (TwoSum) and each
    /// part multiplied exactly (FMA), so the position is a sum of four f64s
    /// whose position relative to each half-sample boundary is decided by an
    /// exact expansion sign. NaN (refused by the caller) where that is not
    /// provably exact: a nonzero product below 2^-960 (its FMA error could
    /// underflow), a position of -0.5 samples or less, or 2^51 samples or
    /// more (where n ± 0.5 is no longer an f64).
    fn exact_sample(t: f64, t0: f64, rate: f64) -> f64 {
        const LIMIT: f64 = (1u64 << 51) as f64;
        let tiny = f64::MIN_POSITIVE * 2f64.powi(62); // 2^-960
        let (d_hi, d_lo) = two_sum(t, -t0);
        let parts = [(d_hi, d_hi * rate), (d_lo, d_lo * rate)];
        // |d_lo × rate| <= 2^-52 |d_hi × rate|: bounding the leading product
        // keeps every term, and the expansion sums, far from overflow.
        let leading_in_range = parts[0].1.abs() < 2.0 * LIMIT; // false for NaN
        if !leading_in_range || parts.iter().any(|&(d, p)| d != 0.0 && p.abs() < tiny) {
            return f64::NAN;
        }
        let terms = [
            parts[0].1,
            d_hi.mul_add(rate, -parts[0].1),
            parts[1].1,
            d_lo.mul_add(rate, -parts[1].1),
        ];
        let cmp = |c: f64| exact_sign(&[terms[0], terms[1], terms[2], terms[3], -c]);
        if cmp(-0.5).is_le() || cmp(LIMIT).is_ge() {
            return f64::NAN;
        }
        // In range, n ± 0.5 are exact; the f64 estimate is within a few
        // samples, so step until n - 0.5 <= position < n + 0.5.
        let mut n = terms.iter().sum::<f64>().round().clamp(0.0, LIMIT);
        while cmp(n - 0.5).is_lt() {
            n -= 1.0;
        }
        while cmp(n + 0.5).is_ge() {
            n += 1.0;
        }
        n
    }

    fn check_timestamp_spikes(
        ds: &Dataset,
        n_samples: usize,
        mut times: &[f64],
        unit_ids: &[u32],
        indices: &[Vec<SampleIndex>],
    ) -> Result<()> {
        let mut rows: Vec<_> = indices
            .iter()
            .map(|row| {
                let (spikes, rest) = times.split_at(row.len());
                times = rest;
                spikes.iter().zip(row).peekable()
            })
            .collect();
        // Merge sorted unit rows by assigned index so each needed block is read
        // once, even when many units have spikes throughout the recording.
        let mut queue: BinaryHeap<_> = rows
            .iter_mut()
            .enumerate()
            .filter_map(|(row, spikes)| spikes.peek().map(|&(_, &sample)| Reverse((sample, row))))
            .collect();
        let (mut block_start, mut next_block) = (0, 0);
        let mut block = Vec::new();
        while let Some(Reverse((sample, row))) = queue.pop() {
            let i = sample.0 as usize; // load_units checked the recording bounds.
            let (&t, _) = rows[row].next().expect("queued spike");
            if i >= next_block {
                let base = i / TIMESTAMP_BLOCK * TIMESTAMP_BLOCK;
                next_block = base.saturating_add(TIMESTAMP_BLOCK);
                // Include both neighbors even at block boundaries.
                block_start = base.saturating_sub(1);
                block = read_timestamp_block(
                    ds,
                    block_start..next_block.saturating_add(1).min(n_samples),
                )?;
            }
            let distance = |j: usize| exact_distance(t, block[j - block_start]);
            let mut nearest = i;
            // The timestamps strictly increase and were verified to differ by
            // far less than half a sample from the grid, so only neighbors can win.
            for j in i.saturating_sub(1)..=(i + 1).min(n_samples - 1) {
                if distance(j) < distance(nearest) {
                    nearest = j;
                }
            }
            ensure!(nearest == i,
                "unit {}: spike at {t} s was assigned sample {i}, but its nearest recorded sample is {nearest}",
                unit_ids[row]);
            if let Some(&(_, &next)) = rows[row].peek() {
                queue.push(Reverse((next, row)));
            }
        }
        Ok(())
    }

    /// `|a - b|` as an unevaluated sum `(hi, lo)` (TwoSum), compared
    /// lexicographically. Rounding is monotonic, so unequal `hi` already order
    /// the exact distances, and equal `hi` leave the exact difference in `lo`.
    /// A plain `(a - b).abs()` can round two different distances to a tie.
    fn exact_distance(a: f64, b: f64) -> (f64, f64) {
        let hi = a - b;
        let bb = hi - a;
        let lo = (a - (hi - bb)) + (-b - bb);
        if hi < 0.0 || (hi == 0.0 && lo < 0.0) {
            (-hi, -lo)
        } else {
            (hi, lo)
        }
    }

    struct UnitMetadata {
        labels: Vec<PhyLabel>,
        tags: Vec<Vec<String>>,
        annotations: Vec<UnitAnnotation>,
        names: Option<Vec<String>>,
        metrics: QualityMetrics,
    }

    fn read_unit_metadata(units: &Group, ids: &[u32]) -> Result<UnitMetadata> {
        let n = ids.len();
        let string_column = |name| -> Result<Option<Vec<String>>> {
            let Some(ds) = units.optional_dataset(name)? else {
                return Ok(None);
            };
            Ok((ds.shape() == [n]).then(|| ds.strings().ok()).flatten())
        };
        let mut label_columns = Vec::new();
        for name in ["label", "quality", "KSLabel"] {
            label_columns.extend(string_column(name)?);
        }
        let labels = (0..n)
            .map(|i| {
                label_columns
                    .iter()
                    .find_map(|col| {
                        let value = &col[i];
                        let label = PhyLabel::parse_tsv_value(value);
                        (label != PhyLabel::Unsorted
                            || value.trim().eq_ignore_ascii_case("unsorted"))
                        .then_some(label)
                    })
                    .unwrap_or_default()
            })
            .collect();

        let index = |column: &str| -> Result<Option<Vec<usize>>> {
            let Some(ds) = units.optional_dataset(&format!("{column}_index"))? else {
                return Ok(None);
            };
            if ds.shape() != [n] {
                return Ok(None);
            }
            Ok(ds
                .read_raw::<u64>()
                .ok()
                .and_then(|ends| ends.into_iter().map(|v| usize::try_from(v).ok()).collect()))
        };
        let tag_index = index("tags")?;
        let tag_values = units.optional_dataset("tags")?;
        let tags = tag_index
            .zip(tag_values)
            .and_then(|(ends, ds)| {
                let values = ds.strings().ok()?;
                let mut rows = ragged_rows(&values, &ends)?;
                for row in &mut rows {
                    row.retain(|tag| valid_tag(tag));
                }
                Some(rows)
            })
            .unwrap_or_default();
        let mut annotations = vec![UnitAnnotation::default(); n];
        if let Some(origins) = string_column("origin")? {
            for (annotation, origin) in annotations.iter_mut().zip(origins) {
                annotation.origin = origin;
            }
        }
        let parent_index = index("parent_unit_ids")?;
        let parent_values = units.optional_dataset("parent_unit_ids")?;
        if let Some(parents) = parent_index.zip(parent_values).and_then(|(ends, ds)| {
            let values = ds.read_raw::<i64>().ok()?;
            ragged_rows(&values, &ends)
        }) {
            let clusters: HashMap<_, _> = ids
                .iter()
                .enumerate()
                .map(|(i, &id)| (i64::from(id), ClusterId(i as u32)))
                .collect();
            for (annotation, row) in annotations.iter_mut().zip(parents) {
                for id in row {
                    if let Some(&parent) = clusters.get(&id) {
                        annotation.parents.push(parent);
                    } else {
                        // The exporter omits emptied units. Keep their IDs
                        // without confusing them with local cluster slots.
                        annotation.missing_parent_unit_ids.push(id);
                    }
                }
            }
        }
        let names = string_column("unit_name")?.filter(|names| {
            let mut unique = std::collections::HashSet::new();
            names
                .iter()
                .all(|name| !name.is_empty() && unique.insert(name))
        });

        let mut metrics = QualityMetrics::new();
        for name in units.member_names()? {
            // An unresolved link may hide a Sorrel metric. Unrecognized local
            // datatypes may belong to other writers and remain outside our schema.
            if !units.local_member_exists(&name)? {
                continue;
            }
            let Ok(ds) = units.dataset(&name) else {
                continue;
            };
            // The Python exporter marks its QC columns explicitly; a numeric
            // foreign Units column alone does not make it a quality metric.
            if ds.shape() == [n]
                && ds
                    .attr("description")
                    .ok()
                    .and_then(|a| a.string().ok())
                    .is_some_and(|d| d.starts_with("Sorrel QC metric. "))
            {
                let values = ds
                    .read_raw::<f32>()
                    .with_context(|| format!("read Units metric {name}"))?;
                metrics.insert(name, values);
            }
        }
        metrics.sort_by_name();
        Ok(UnitMetadata {
            labels,
            tags,
            annotations,
            names,
            metrics,
        })
    }

    fn ragged_rows<T: Clone>(values: &[T], ends: &[usize]) -> Option<Vec<Vec<T>>> {
        if ends.last().copied().unwrap_or(0) != values.len() {
            return None;
        }
        let mut start = 0;
        ends.iter()
            .map(|&end| {
                let row = values.get(start..end)?.to_vec();
                start = end;
                Some(row)
            })
            .collect()
    }

    #[cfg(test)]
    #[test]
    fn lazy_window_matches_eager_scaling_across_chunks_and_end() {
        const FIXTURES: &[&str] = &[
            #[cfg(feature = "hdf5")]
            "good.nwb",
            #[cfg(feature = "zarr")]
            "good.nwb.zarr",
        ];
        for name in FIXTURES {
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../release/fixtures/nwb")
                .join(name);
            let p = NwbProvider::open(path).unwrap();
            let trace = p.trace.as_ref().unwrap();
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

    #[cfg(all(test, feature = "hdf5", feature = "zarr"))]
    #[test]
    fn zarr_and_hdf5_windows_match() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../release/fixtures/nwb");
        for file in ["good.nwb", "timestamps.nwb"] {
            // Blosc/LZ4 and zlib, respectively.
            let hdf5 = NwbProvider::open(dir.join(file)).unwrap();
            let zarr = NwbProvider::open(dir.join(format!("{file}.zarr"))).unwrap();
            let window = |p: &NwbProvider| p.trace(SampleIndex(29), 176).samples.to_f32_centred();
            assert_eq!(window(&hdf5), window(&zarr), "{file}");
        }
    }

    #[cfg(all(test, feature = "hdf5"))]
    #[test]
    fn timestamp_validation_rejects_subsample_jitter() {
        let dir = tempfile::tempdir().unwrap();
        let file = hdf5_metno::File::create(dir.path().join("timestamps.h5")).unwrap();
        // Alternating 0.8/1.2 ms intervals on a nominal 1 kHz grid.
        let times = [0.5, 0.5008, 0.502, 0.5028, 0.504];
        let ds = file
            .new_dataset_builder()
            .with_data(&times)
            .create("timestamps")
            .unwrap();
        let err = regular_timestamps(&Dataset::Hdf5(ds.clone()), times.len()).unwrap_err();
        assert!(err.to_string().contains("max deviation 0.200000 samples"));
    }

    #[cfg(all(test, feature = "hdf5"))]
    #[test]
    fn rejects_external_links_for_spike_mapping() {
        let dir = tempfile::tempdir().unwrap();
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../release/fixtures/nwb");
        for (fixture, dataset) in [
            ("good.nwb", "/acquisition/ElectricalSeriesRaw/starting_time"),
            (
                "timestamps.nwb",
                "/acquisition/ElectricalSeriesRaw/timestamps",
            ),
            ("good.nwb", "/units/spike_times"),
            ("good.nwb", "/units/spike_times_index"),
            ("good.nwb", "/units/id"),
        ] {
            let source = fixtures.join(fixture);
            let path = dir.path().join(fixture);
            std::fs::write(&path, std::fs::read(&source).unwrap()).unwrap();
            {
                let file = hdf5_metno::File::open_rw(&path).unwrap();
                file.unlink(dataset).unwrap();
                file.link_external(source.to_str().unwrap(), dataset, dataset)
                    .unwrap();
            }
            let error = NwbProvider::open(&path).err().expect(dataset);
            let error = format!("{error:#}");
            assert!(
                error.contains(&format!("{dataset} must be stored in this file"))
                    && error.contains(&format!("{dataset} is an HDF5 external link")),
                "{error}"
            );
        }
    }

    #[cfg(test)]
    #[test]
    fn exact_sample_rounds_the_real_product() {
        // 8.333333333333333e-05 × 30000 is just below 2.5 but rounds to 2.5 in f64.
        let t = 8.333333333333333e-05;
        assert_eq!(t * 30000.0, 2.5);
        assert_eq!(exact_sample(t, 0.0, 30000.0), 2.0);
        // Exact ties and plain cases keep f64::round's answer for positions ≥ 0.
        assert_eq!(exact_sample(2.5, 0.0, 1.0), 3.0);
        assert_eq!(exact_sample(1.25, 0.25, 1000.0), 1000.0);
        assert_eq!(exact_sample(11000.0, 0.0, 30000.0), 330_000_000.0);
        // The subtraction is exact too: 2.5 - 2^-60 rounds to 2.5 in f64.
        assert_eq!(2.5 - 2f64.powi(-60), 2.5);
        assert_eq!(exact_sample(2.5, 2f64.powi(-60), 1.0), 2.0);
        // Far positions stay exact up to the 2^51-sample limit.
        assert_eq!(exact_sample(2f64.powi(50), 0.0, 1.0), 2f64.powi(50));
        assert_eq!(
            exact_sample(2f64.powi(50) + 0.5, 0.0, 1.0),
            2f64.powi(50) + 1.0
        );
        // Outside what can be proved exact: refused (NaN), never guessed.
        assert!(exact_sample(2f64.powi(52), 0.0, 1.0).is_nan());
        assert!(exact_sample(2f64.powi(44) + 2f64.powi(-8), 0.0, 30000.0).is_nan());
        assert!(exact_sample(1.0, 2f64.powi(-1074), 0.5).is_nan());
        assert!(exact_sample(0.0, 0.5, 1.0).is_nan());
        assert_eq!(exact_sample(0.0, 0.25, 1.0), 0.0);
        assert!(exact_sample(f64::NAN, 0.0, 1.0).is_nan());
        assert!(exact_sample(f64::INFINITY, 0.0, 1.0).is_nan());
    }

    #[cfg(test)]
    #[test]
    fn exact_distance_breaks_rounded_ties() {
        // |2^-11 - 2^-70| rounds to 2^-11 = |2^-11 - 2^-10|, but the first is nearer.
        let t = 2f64.powi(-11);
        assert_eq!((t - 2f64.powi(-70)).abs(), (t - 2f64.powi(-10)).abs());
        assert!(exact_distance(t, 2f64.powi(-70)) < exact_distance(t, 2f64.powi(-10)));
        assert!(exact_distance(t, 2f64.powi(-10)) > exact_distance(t, 2f64.powi(-70)));
        // A true tie stays a tie, from either side.
        assert_eq!(exact_distance(1.5, 1.0), exact_distance(1.5, 2.0));
        assert_eq!(exact_distance(1.0, 1.5), exact_distance(2.0, 1.5));
    }

    #[cfg(all(test, feature = "hdf5"))]
    #[test]
    fn timestamp_validation_rejects_coarse_epoch_clock() {
        let dir = tempfile::tempdir().unwrap();
        let file = hdf5_metno::File::create(dir.path().join("timestamps.h5")).unwrap();
        let origin = 1_700_000_000.0_f64;
        let ulp = origin.next_up() - origin;
        let mut times: Vec<_> = (0..128)
            .map(|i| origin + f64::from(i) * 4.0 * ulp)
            .collect();
        times[50] += 3.0 * ulp; // 0.75 samples of jitter, exactly representable.
        let ds = file
            .new_dataset_builder()
            .with_data(&times)
            .create("timestamps")
            .unwrap();
        let error = regular_timestamps(&Dataset::Hdf5(ds.clone()), times.len()).unwrap_err();
        assert!(error.to_string().contains("timestamps are too coarse"));
    }

    #[cfg(all(test, feature = "hdf5"))]
    #[test]
    fn timestamp_validation_checks_the_interval_between_read_blocks() {
        let dir = tempfile::tempdir().unwrap();
        let file = hdf5_metno::File::create(dir.path().join("timestamps.h5")).unwrap();
        let origin = 1_700_000_000.0;
        let mut times: Vec<f64> = (0..65_540)
            .map(|i| origin + f64::from(i) / 30_000.0)
            .collect();
        let ds = file
            .new_dataset_builder()
            .with_data(&times)
            .create("timestamps")
            .unwrap();
        let (rate, t0) = regular_timestamps(&Dataset::Hdf5(ds.clone()), times.len()).unwrap();
        assert!((rate - 30_000.0).abs() < 0.01);
        assert_eq!(t0, origin);
        // Duplicate the first timestamp in the second block at the boundary.
        times[65_537] = times[65_536];
        ds.write(&times).unwrap();
        let error = regular_timestamps(&Dataset::Hdf5(ds.clone()), times.len()).unwrap_err();
        assert!(error
            .to_string()
            .contains("irregular timestamps at sample 65537"));
    }
}

#[cfg(any(feature = "hdf5", feature = "zarr"))]
pub(crate) use imp::read_trials;
#[cfg(any(feature = "hdf5", feature = "zarr"))]
pub use imp::{NwbOpenParams, NwbProvider};
