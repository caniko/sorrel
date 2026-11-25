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

/// The NWB file has no trials table (as opposed to an unreadable table).
#[derive(Debug, thiserror::Error)]
#[error("no /intervals/trials table")]
pub struct NoTrials;

/// The `--nwb-series` value that skips the trace; use an absolute path to select a series named `none`.
#[cfg(any(feature = "hdf5", feature = "zarr"))]
pub const UNITS_ONLY_SERIES: &str = "none";

#[cfg(any(feature = "hdf5", feature = "zarr"))]
mod imp {
    mod specs;
    pub(super) mod storage;
    use crate::extras::{HasGeometry, HasQualityMetrics, QualityMetrics};
    use crate::kilosort::{pinned_sorter, valid_tag, PhyLabel};
    use crate::nwb_units::exact_sample;
    use crate::provider::{
        appended_unit_id, ChannelId, ClusterId, DataProvider, SampleIndex, SessionSpan, SpanSource,
        TraceSamples, TraceSlice, UnitAnnotation,
    };
    use anyhow::{anyhow, ensure, Context, Result};
    use serde_json::{json, Value};
    use specs::{legacy_is_series, legacy_type_parents, TypeSpecs};
    use std::cmp::Reverse;
    use std::collections::{BinaryHeap, HashMap, VecDeque};
    use std::ops::Range;
    use std::path::Path;
    use std::sync::Mutex;
    use storage::{Dataset, Group, SampleType};

    const ELECTRODES: &str = "/general/extracellular_ephys/electrodes";

    #[derive(Clone, Copy, Debug, Default)]
    pub struct NwbOpenParams<'a> {
        /// Acquisition name or absolute NWB path, also applied to a paired file;
        /// `none` skips the trace and uses the Units clock or a supplied rate.
        pub series: Option<&'a str>,
        /// Read the ElectricalSeries from this file instead of the units file.
        pub raw_path: Option<&'a Path>,
        /// Exact declared clock (Hz), or a supplied rate for units without a time base.
        pub sample_rate: Option<f64>,
    }

    impl NwbOpenParams<'_> {
        fn units_only(self) -> bool {
            self.series == Some(super::UNITS_ONLY_SERIES)
        }

        /// Reject incompatible options before opening a local or remote source.
        pub fn validate(self) -> Result<()> {
            ensure!(
                !self.units_only() || self.raw_path.is_none(),
                "--nwb-series none (units-only) cannot be combined with --dat"
            );
            Ok(())
        }
    }

    pub struct NwbProvider {
        rate: f64,
        t0: f64,
        n_channels: u32,
        n_samples: SampleIndex,
        span: SessionSpan,
        spikes_per_cluster: Vec<Vec<SampleIndex>>,
        unit_ids: Vec<u32>,
        metadata: UnitMetadata,
        channel_positions: Vec<[f32; 2]>,
        channel_map: Vec<ChannelId>,
        identity_bytes: Vec<u8>,
        trace: Option<NwbTrace>,
        skipped_series: Option<Value>,
        #[cfg(any(feature = "hdf5", feature = "dandi"))]
        source: Group,
        #[cfg(feature = "dandi")]
        remote: bool,
    }

    impl NwbProvider {
        pub fn open(path: impl AsRef<Path>) -> Result<Self> {
            Self::open_with(path, NwbOpenParams::default())
        }

        pub fn open_with(path: impl AsRef<Path>, params: NwbOpenParams<'_>) -> Result<Self> {
            params.validate()?;
            let path = path.as_ref();
            #[cfg(feature = "dandi")]
            return Self::open_pinned(
                path,
                params,
                None,
                Self::remote_raw_locator(params)
                    .map(crate::dandi::resolve)
                    .transpose()?,
            );
            #[cfg(not(feature = "dandi"))]
            Self::from_group(Group::open(path)?, path, params, None)
        }

        /// Open a DANDI asset or HTTPS NWB stream and return its relative local state directory.
        #[cfg(feature = "dandi")]
        pub fn open_remote(
            locator: &str,
            params: NwbOpenParams<'_>,
        ) -> Result<(Self, std::path::PathBuf)> {
            params.validate()?;
            let [root, raw] =
                crate::dandi::resolve_pair(Some(locator), Self::remote_raw_locator(params))?;
            let state = root
                .as_ref()
                .context("missing resolved NWB dataset")?
                .state
                .clone();
            Ok((
                Self::open_pinned(Path::new(locator), params, root, raw)?,
                state,
            ))
        }

        #[cfg(feature = "dandi")]
        fn remote_raw_locator(params: NwbOpenParams<'_>) -> Option<&str> {
            params
                .raw_path
                .filter(|path| crate::dandi::is_remote(path))
                .and_then(Path::to_str)
        }

        /// Open NWB using the supplied remote revisions, never resolving their locators again.
        /// Local inputs are opened from `path` and `params.raw_path` as usual.
        #[cfg(feature = "dandi")]
        pub fn open_pinned(
            path: &Path,
            params: NwbOpenParams<'_>,
            root: Option<crate::dandi::Resolved>,
            raw: Option<crate::dandi::Resolved>,
        ) -> Result<Self> {
            params.validate()?;
            ensure!(
                root.is_some() || !crate::dandi::is_remote(path),
                "missing resolved NWB dataset"
            );
            ensure!(
                raw.is_some() == params.raw_path.is_some_and(crate::dandi::is_remote),
                "remote NWB recording must be resolved before opening"
            );
            let (f, identity) = match root {
                Some(root) => (Group::open_remote(root.store)?, Some(root.identity)),
                None => (Group::open(path)?, None),
            };
            Self::from_group(f, path, params, identity, raw)
        }

        fn from_group(
            f: Group,
            path: &Path,
            params: NwbOpenParams<'_>,
            remote_identity: Option<Vec<u8>>,
            #[cfg(feature = "dandi")] resolved_raw: Option<crate::dandi::Resolved>,
        ) -> Result<Self> {
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
            let raw = params
                .raw_path
                .map(|path| {
                    #[cfg(feature = "dandi")]
                    if let Some(resolved) = resolved_raw {
                        return Ok::<_, anyhow::Error>((
                            Group::open_remote(resolved.store)?,
                            Some(resolved.identity),
                        ));
                    }
                    Ok::<_, anyhow::Error>((Group::open(path)?, None))
                })
                .transpose()?;
            let recording = raw.as_ref().map_or(&f, |(group, _)| group);
            let (mut series, processed) = if params.units_only() {
                (None, None)
            } else {
                find_electrical_series(recording, params.series, raw.is_some(), false)?
            };
            let mut skipped_for_electrodes = None;
            if params.series.is_none() && raw.is_none() {
                if let Some((name, candidate)) = &series {
                    if misses_units_electrodes(&f, &units, &ids, candidate).unwrap_or(false) {
                        log::warn!("NWB: skipping ElectricalSeries {name} because it does not record the Units' electrodes");
                        skipped_for_electrodes = series.take().map(|(name, _)| name);
                    }
                }
            }
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
                let data = series.trace_data()?;
                let shape = data.shape();
                let (n_samples, n_channels) = (shape[0], shape.get(1).copied().unwrap_or(1));
                let hint = match (params.series, raw.is_some()) {
                    (None, false) => "; to open the Units without this trace, pass --nwb-series none (with --sample-rate HZ if /units/spike_times declares no resolution)",
                    (None, true) => "; to open the Units without this trace, omit --dat and pass --nwb-series none (with --sample-rate HZ if /units/spike_times declares no resolution)",
                    (Some(_), _) => "",
                };
                let (stored_rate, t0, timestamps) =
                    read_time_base_with(recording, name, n_samples, regular_timestamps, hint)?;
                let units_clock = if let Some(declared) = units_rate {
                    // The one-ppm gate decides whether resolution declares a clock,
                    // not whether an explicit value matches that clock.
                    let close = is_units_clock(declared, stored_rate);
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
                    && remote_identity.is_none()
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
                let rate = units_rate.or(params.sample_rate).with_context(|| {
                    if params.units_only() {
                        return "units-only NWB has no /units/spike_times resolution; pass --sample-rate HZ as the analysis grid".into();
                    }
                    if let Some(name) = &skipped_for_electrodes {
                        return format!("skipped acquisition ElectricalSeries {name} because it does not record the Units' electrodes, and /units/spike_times has no resolution; pass --sample-rate HZ as the analysis grid, or --nwb-series PATH to force a series");
                    }
                    let missing = "no /acquisition ElectricalSeries and no /units/spike_times resolution";
                    match processed {
                        Some(paths) => format!("{missing}; /processing series ({paths}) are not a spike clock; pass --sample-rate HZ as the analysis grid, or --nwb-series PATH to use a processed series' clock"),
                        None => format!("{missing}; pass --sample-rate HZ as the analysis grid"),
                    }
                })?;
                (
                    None,
                    checked_rate(rate)?,
                    0.0,
                    read_units_only_channels(&f)?,
                    None,
                )
            };
            if let Some((raw, _)) = &raw {
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
                trace.as_ref().map(|t| (t.n_samples, timestamps.as_ref())),
                comparison_rate.map(|other_rate| (other_rate, require_agreement)),
            )
            .context("load /units")?;
            let span = if trace.is_some() {
                (n_samples as u64).into()
            } else {
                units_span(&f, &spikes_per_cluster, rate, t0)
            };
            if raw.is_some() {
                ensure!(spikes_per_cluster.iter().filter_map(|s| s.last()).all(|s| s.0 < n_samples as u64),
                    "spike times extend past the paired ElectricalSeries (starting time {t0} s, {n_samples} samples at {rate} Hz)");
            }
            #[cfg(feature = "dandi")]
            let remote = remote_identity.is_some();
            let mut identity_bytes = match remote_identity {
                Some(identity) => identity,
                None => storage::local_identity(path)?,
            };
            if !(legacy_identity && same_mapping) {
                if let Some((name, _)) = &series {
                    identity_bytes.extend_from_slice(name.as_bytes());
                }
                // An empty path seals units-only mode, explicit or by absence.
                identity_bytes.push(0);
                identity_bytes.extend(rate.to_le_bytes());
                if let Some(((_, raw_identity), raw_path)) = raw.zip(params.raw_path) {
                    identity_bytes.extend(match raw_identity {
                        Some(identity) => identity,
                        None => storage::local_identity(raw_path)?,
                    });
                }
            }
            let metadata = read_unit_metadata(&units, &unit_ids)?;
            Ok(Self {
                rate,
                t0,
                n_channels: u32::try_from(channel_map.len()).context("too many NWB channels")?,
                n_samples: SampleIndex(n_samples as u64),
                span,
                metadata,
                spikes_per_cluster,
                unit_ids,
                channel_positions,
                channel_map,
                identity_bytes,
                trace,
                skipped_series: params.units_only().then(|| inspect_skipped_series(&f)),
                #[cfg(any(feature = "hdf5", feature = "dandi"))]
                source: f,
                #[cfg(feature = "dandi")]
                remote,
            })
        }

        /// Read trials from the same remote sorting file and revision already opened.
        #[cfg(feature = "dandi")]
        pub fn read_remote_trials(&self) -> Result<crate::Events> {
            ensure!(self.remote, "not a remote NWB");
            trials_from_group(&self.source)
        }

        /// Read export metadata from the same sorting file or pinned store as the units.
        #[cfg(feature = "hdf5")]
        pub fn export_metadata(&self) -> Result<crate::nwb_write::Metadata> {
            let read = |name| self.source.stored_dataset(name)?.read_string();
            let mut metadata = crate::nwb_write::Metadata::new(
                read("/identifier")?,
                read("/session_description")?,
                read("/session_start_time")?,
            )?;
            if let Some(time) = self.export_reference_time()? {
                metadata.timestamps_reference_time = time;
            }
            Ok(metadata)
        }

        #[cfg(feature = "hdf5")]
        fn export_reference_time(&self) -> Result<Option<String>> {
            self.source
                .optional_stored_dataset("/timestamps_reference_time")?
                .map(|ds| ds.read_string())
                .transpose()
        }

        /// Time zero for appending exported units (legacy files use session start).
        #[cfg(feature = "hdf5")]
        pub fn export_timestamp_reference(&self) -> Result<String> {
            match self.export_reference_time()? {
                Some(time) => Ok(time),
                None => self
                    .source
                    .stored_dataset("/session_start_time")?
                    .read_string(),
            }
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
        fn exact_sample_rate(&self) -> Result<f64> {
            Ok(self.rate)
        }
        fn n_channels(&self) -> u32 {
            self.n_channels
        }
        fn n_samples(&self) -> SampleIndex {
            self.n_samples
        }

        fn session_span(&self) -> SessionSpan {
            self.span
        }

        fn trace_provenance(&self) -> Value {
            if let Some(skipped) = &self.skipped_series {
                return json!({ "mode": "units-only", "skipped_series": skipped });
            }
            json!({ "mode": if self.trace.is_some() { "trace" } else { "units-only" } })
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
            self.read_trace(start, len).unwrap_or_else(|err| {
                // Interactive callers using the infallible trait get no samples, never zeros.
                log::error!("read NWB trace at sample {start}: {err:#}");
                TraceSlice {
                    start,
                    n_channels: self.n_channels,
                    samples: TraceSamples::F32(&[]),
                }
            })
        }

        fn read_trace(&self, start: SampleIndex, len: u32) -> Result<TraceSlice<'_>> {
            let samples = self
                .trace
                .as_ref()
                .map(|trace| trace.window(start, len))
                .transpose()?;
            Ok(TraceSlice {
                start,
                n_channels: self.n_channels,
                samples: samples.map_or(TraceSamples::F32(&[]), TraceSamples::OwnedF32),
            })
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

    // At 2^53 an adjacent integer can round to the same f64.
    const CLOCK_LIMIT: f64 = 9_007_199_254_740_992.0;

    fn checked_rate(rate: f64) -> Result<f64> {
        ensure!(
            rate < CLOCK_LIMIT && (rate as f32) > 0.0,
            "invalid sample rate {rate} Hz (must be positive and below 2^53 Hz)"
        );
        Ok(rate)
    }

    fn check_time(t: f64, what: &str) -> Result<()> {
        ensure!(
            t.abs() < CLOCK_LIMIT,
            "invalid {what} {t} s (must be finite and have magnitude below 2^53 s)"
        );
        Ok(())
    }

    fn is_units_clock(units_rate: f64, series_rate: f64) -> bool {
        (units_rate - series_rate).abs() <= 1e-6 * series_rate
    }

    #[cfg(feature = "hdf5")]
    pub(crate) fn check_append_rate(file: &hdf5_metno::File, units_rate: f64) -> Result<()> {
        let units_rate = checked_rate(units_rate)?;
        let file = Group::Hdf5(file.group("/")?);
        if let (Some((name, series)), _) = find_electrical_series(&file, None, false, false)? {
            let samples = series.dataset("data")?.shape()[0];
            let (series_rate, _, _) =
                read_time_base_with(&file, &name, samples, regular_timestamps, "")?;
            // A single-file open ignores out-of-gate resolutions. Refuse to
            // append here so reopening cannot silently use the wrong clock.
            ensure!(is_units_clock(units_rate, series_rate),
                "NWB rate mismatch: exported Units clock is {units_rate} Hz but destination {name} is {series_rate} Hz; use --out for a separate Units file");
        }
        Ok(())
    }

    fn read_units_rate(spike_times: &Dataset, strict: bool) -> Result<Option<f64>> {
        let Some(attr) = spike_times.optional_attr("resolution")? else {
            return Ok(None);
        };
        let resolution = attr.exact_number("/units/spike_times resolution")?;
        ensure!(
            !(resolution.abs() >= CLOCK_LIMIT || (resolution > 0.0 && resolution <= CLOCK_LIMIT.recip())),
            "invalid /units/spike_times resolution {resolution} s (resolution and sample rate must be below 2^53)"
        );
        match resolution {
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

    fn units_span(f: &Group, spikes: &[Vec<SampleIndex>], rate: f64, t0: f64) -> SessionSpan {
        let first = spikes.iter().filter_map(|row| row.first().copied()).min();
        let last = spikes.iter().filter_map(|row| row.last().copied()).max();
        let mut rejected = None;
        for path in [
            "/intervals/epochs",
            "/units/obs_intervals",
            "/intervals/trials",
        ] {
            let (start, end) = match interval_extent(f, path, spikes.len()) {
                Ok(Some((start, end))) => {
                    (exact_sample(start, t0, rate), exact_sample(end, t0, rate))
                }
                Ok(None) => continue,
                Err(err) => {
                    log::warn!("ignoring NWB session extent at {path}: {err:#}");
                    continue;
                }
            };
            if !(0.0..u64::MAX as f64).contains(&start)
                || !(start..u64::MAX as f64).contains(&end)
                || start == end
            {
                log::warn!(
                    "ignoring NWB session extent at {path}: not a positive span on the sample grid"
                );
                continue;
            }
            let (start, end) = (SampleIndex(start as u64), SampleIndex(end as u64));
            if first.is_some_and(|t| t < start) || last.is_some_and(|t| t >= end) {
                rejected = Some(path);
                break;
            }
            return SessionSpan {
                start,
                end,
                source: SpanSource::Declared(path),
            };
        }
        match first.zip(last) {
            Some((start, last)) => SessionSpan {
                start,
                end: SampleIndex(last.0 + 1),
                source: SpanSource::SpikeExtent { rejected },
            },
            None => SessionSpan {
                source: SpanSource::Unavailable,
                ..SessionSpan::from(0)
            },
        }
    }

    fn interval_extent(f: &Group, path: &str, n_units: usize) -> Result<Option<(f64, f64)>> {
        // Only read time columns and the Units ragged index, never the rest of
        // the DynamicTable. Bound temporary buffers for remote and local files.
        const BATCH: usize = 8192;
        let (mut first, mut last) = (f64::INFINITY, f64::NEG_INFINITY);
        let mut add = |start: f64, stop: f64| -> Result<()> {
            check_time(start, "interval start")?;
            check_time(stop, "interval stop")?;
            ensure!(start <= stop, "interval stop precedes start");
            first = first.min(start);
            last = last.max(stop);
            Ok(())
        };
        if path == "/units/obs_intervals" {
            let Some(ds) = f.optional_dataset(path)? else {
                return Ok(None);
            };
            let shape = ds.shape();
            let &[n, 2] = shape.as_slice() else {
                anyhow::bail!("expected an N×2 observation interval table");
            };
            if n == 0 {
                return Ok(None);
            }
            let index = f.dataset("/units/obs_intervals_index")?;
            ensure!(
                index.shape() == [n_units],
                "expected one observation index per unit"
            );
            let signed = match index.sample_type()? {
                SampleType::I8 | SampleType::I16 | SampleType::I32 | SampleType::I64 => true,
                SampleType::U8 | SampleType::U16 | SampleType::U32 | SampleType::U64 => false,
                _ => anyhow::bail!("expected an integer observation interval index"),
            };
            let mut prev = 0;
            for i in (0..n_units).step_by(BATCH) {
                let range = i..n_units.min(i + BATCH);
                let ends = if signed {
                    index
                        .read_rows::<i64>(range)?
                        .into_iter()
                        .map(u64::try_from)
                        .collect::<Result<Vec<_>, _>>()
                        .context("negative observation interval index")?
                } else {
                    index.read_rows::<u64>(range)?
                };
                for end in ends {
                    ensure!(end >= prev, "observation interval index is not monotonic");
                    ensure!(
                        end <= n as u64,
                        "observation interval index is out of bounds"
                    );
                    prev = end;
                }
            }
            ensure!(
                prev == n as u64,
                "observation interval index does not account for every interval"
            );
            for i in (0..n).step_by(BATCH) {
                let rows = ds.read_rows::<f64>(i..n.min(i + BATCH))?;
                for &[start, stop] in rows.as_chunks::<2>().0 {
                    add(start, stop)?;
                }
            }
        } else {
            let Some(g) = f.optional_group(path)? else {
                return Ok(None);
            };
            let (starts, stops) = (g.dataset("start_time")?, g.dataset("stop_time")?);
            let shape = starts.shape();
            let &[n] = shape.as_slice() else {
                anyhow::bail!("expected one-dimensional start_time");
            };
            ensure!(
                stops.shape() == shape,
                "start_time and stop_time shapes differ"
            );
            for i in (0..n).step_by(BATCH) {
                let range = i..n.min(i + BATCH);
                for (start, stop) in starts
                    .read_rows::<f64>(range.clone())?
                    .into_iter()
                    .zip(stops.read_rows::<f64>(range)?)
                {
                    add(start, stop)?;
                }
            }
        }
        Ok(first.is_finite().then_some((first, last)))
    }

    pub(crate) fn read_trials(path: &Path) -> Result<crate::Events> {
        let f = Group::open(path)?;
        trials_from_group(&f).with_context(|| format!("{}: read NWB trials", path.display()))
    }

    fn trials_from_group(f: &Group) -> Result<crate::Events> {
        let g = f
            .optional_group("/intervals/trials")?
            .ok_or(super::NoTrials)?;
        let starts: Vec<f64> = g
            .dataset("start_time")?
            .read_raw()
            .context("read NWB trials/start_time")?;
        let mut names = None;
        for column in ["trial_type", "stimulus", "condition"] {
            if let Some(ds) = g.optional_dataset(column)? {
                if let Some(values) = ds
                    .optional_strings()
                    .with_context(|| format!("read NWB trials/{column}"))?
                {
                    ensure!(
                        values.len() == starts.len(),
                        "NWB trials/{column} length differs from start_time"
                    );
                    names = Some(values);
                    break;
                }
            }
        }
        let mut events = crate::Events::default();
        for (i, t) in starts.into_iter().enumerate() {
            events.push(t, names.as_ref().map_or("trial", |n| n[i].as_str()));
        }
        events.sort_by_time();
        Ok(events)
    }

    /// Mirror trunk's discovery for the journal seal, including the old
    /// variable-length-only decoder for both type tags and cached specs.
    fn is_trunk_series(f: &Group, selected: &str) -> bool {
        let Some(name) = selected.strip_prefix("/acquisition/") else {
            return false;
        };
        let Ok(Some(acq)) = f.legacy_group("/acquisition") else {
            return false;
        };
        let Ok(names) = acq.member_names() else {
            return false;
        };
        let parents = legacy_type_parents(f).unwrap_or_default();
        let mut first = None;
        for candidate in names {
            let g = match acq.legacy_group(&candidate) {
                Ok(Some(g)) => g,
                Ok(None) => continue,
                Err(_) => return false, // A linked member cannot inherit the legacy seal.
            };
            if first.is_some() {
                continue;
            }
            let ty = match g.optional_attr("neurodata_type") {
                Ok(attr) => attr.and_then(|a| a.legacy_string().ok()),
                Err(_) => return false,
            };
            if ty.is_some_and(|t| legacy_is_series(&t, &parents)) {
                match g.legacy_dataset_rank("data") {
                    Ok(Some(2)) => first = Some(candidate == name),
                    Ok(_) => (),
                    Err(_) => return false, // Might have been trunk's series.
                }
            }
        }
        first.unwrap_or(false)
    }

    type ElectricalSeries = (String, Group);

    #[derive(Debug, thiserror::Error)]
    #[error("{message}")]
    struct SeriesDataError {
        series: String,
        message: String,
    }

    fn find_electrical_series(
        f: &Group,
        requested: Option<&str>,
        required: bool,
        skipping: bool,
    ) -> Result<(Option<ElectricalSeries>, Option<String>)> {
        fn collect(
            group: &Group,
            path: &str,
            depth: usize,
            is_series: &impl Fn(&str, &str) -> Result<bool>,
            found: &mut Vec<(String, Group)>,
        ) -> Result<()> {
            for name in group.member_names()? {
                let Some(child) = group.child_group(&name)? else {
                    continue;
                };
                let path = format!("{path}/{name}");
                let ty = child
                    .optional_attr("neurodata_type")?
                    .map(|a| a.string())
                    .transpose()?;
                let is_series_node = if let Some(ty) = ty {
                    is_series(&ty, &path)?
                } else {
                    false
                };
                if is_series_node {
                    found.push((path, child));
                } else if depth > 0 && !child.is_alias(group, &name) {
                    // Never walk into soft aliases; also bound recursion for hard links.
                    collect(&child, &path, depth - 1, is_series, found)?;
                }
            }
            Ok(())
        }

        let requested_path = requested.map(|path| {
            if path.starts_with('/') {
                path.to_owned()
            } else {
                format!("/acquisition/{path}")
            }
        });
        let specs = std::cell::OnceCell::new();
        let is_series = |ty: &str, path: &str| {
            let unavailable = || {
                format!("cannot determine ElectricalSeries ancestry of {path} ({ty:?}); readable same-file cached NWB specifications are required")
            };
            let specs = specs.get_or_init(|| TypeSpecs::read(f));
            let specs = specs
                .as_ref()
                .map_err(|err| anyhow!("{}: {err:#}", unavailable()))?;
            // Refuse type conflicts only where they can change the selection.
            let refuse_conflicts = requested_path
                .as_deref()
                .map_or_else(|| path.starts_with("/acquisition/"), |p| p == path);
            specs.is_series(ty, refuse_conflicts)
        };
        let mut found = Vec::new();
        for (root, depth) in [("/acquisition", 0), ("/processing", 32)] {
            if skipping && root == "/processing" {
                continue;
            }
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
        let checked_rank = |path: &str, g: &Group| {
            data_rank(g).with_context(|| SeriesDataError {
                series: path.to_owned(),
                message: format!("ElectricalSeries {path}: invalid data"),
            })
        };
        let available = |found: &[(String, Group)]| {
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
        let index = if let Some(path) = &requested_path {
            let mut index = found.iter().position(|(name, _)| name == path);
            if index.is_none() {
                // Discovery does not walk soft-linked groups, but an explicit
                // path can select one once the storage layer resolves it.
                if let Some(group) = f.optional_group(path)? {
                    if let Some(ty) = group.optional_attr("neurodata_type")? {
                        if is_series(&ty.string()?, path)? {
                            index = Some(found.len());
                            found.push((path.clone(), group));
                        }
                    }
                }
            }
            index.with_context(|| {
                let requested = requested.unwrap_or(path);
                format!(
                    "no ElectricalSeries matching {requested:?}; available series: {}",
                    available(&found)
                )
            })?
        } else {
            let mut best = None;
            for (i, (path, group)) in found.iter().enumerate() {
                if !path.starts_with("/acquisition/") {
                    continue;
                }
                let rank = checked_rank(path, group)?;
                if best.is_none_or(|(previous, _)| rank > previous) {
                    best = Some((rank, i));
                }
            }
            let Some((_, index)) = best else {
                if required {
                    ensure!(
                        !found.is_empty(),
                        "no ElectricalSeries in the raw NWB; available series: none"
                    );
                    anyhow::bail!(
                        "no /acquisition ElectricalSeries in the raw NWB; available series: {}; pass --nwb-series PATH to use a processed series",
                        available(&found)
                    );
                }
                return Ok((None, (!found.is_empty()).then(|| available(&found))));
            };
            index
        };
        let (path, group) = &found[index];
        ensure!(
            checked_rank(path, group)? != 0,
            SeriesDataError {
                series: path.clone(),
                message: format!(
                    "ElectricalSeries {path} requires 1-D data (one channel) or 2-D data (time × channels); available series: {}",
                    available(&found)
                ),
            }
        );
        if !skipping && requested.is_none() && found.len() > 1 {
            log::warn!(
                "NWB: reading ElectricalSeries {path}; available series: {} (choose with --nwb-series)",
                available(&found)
            );
        }
        Ok((Some(found.swap_remove(index)), None))
    }

    fn misses_units_electrodes(
        f: &Group,
        units: &Group,
        ids: &Dataset,
        series: &Group,
    ) -> Result<bool> {
        let Some(unit_region) = units.optional_dataset("electrodes")? else {
            return Ok(false);
        };
        let series_region = series.dataset("electrodes")?;
        let Some(table) = unit_region.shared_table(&series_region, f)? else {
            return Ok(false);
        };
        let rows = |ds: &Dataset| -> Result<Vec<i64>> {
            ensure!(
                ds.shape().len() == 1
                    && !matches!(ds.sample_type()?, SampleType::F32 | SampleType::F64)
            );
            ds.read_raw()
        };
        let referenced = rows(&unit_region)?;
        let ends = rows(&units.dataset("electrodes_index")?)?;
        ensure!(
            ids.shape() == [ends.len()]
                && ends.iter().is_sorted()
                && ends.first().is_none_or(|&end| end >= 0)
                && ends.last().copied().unwrap_or(0) == referenced.len() as i64
        );
        let recorded = rows(&series_region)?;
        let shape = table.dataset("id")?.shape();
        ensure!(
            shape.len() == 1
                && referenced
                    .iter()
                    .chain(&recorded)
                    .all(|&r| { usize::try_from(r).is_ok_and(|r| r < shape[0]) })
        );
        let recorded: std::collections::HashSet<_> = recorded.into_iter().collect();
        Ok(referenced.iter().any(|row| !recorded.contains(row)))
    }

    fn inspect_skipped_series(file: &Group) -> Value {
        let (name, series) = match find_electrical_series(file, None, false, true) {
            Ok((Some(series), _)) => series,
            Ok((None, _)) => {
                return json!({
                    "status": "no acquisition series",
                    "message": "No /acquisition ElectricalSeries was found.",
                })
            }
            Err(err) => {
                let mut skipped = json!({
                    "status": "refused", "check": "series selection", "message": format!("{err:#}"),
                });
                if let Some(data) = err.downcast_ref::<SeriesDataError>() {
                    skipped["series"] = json!(data.series);
                    skipped["check"] = json!("data shape/type");
                }
                return skipped;
            }
        };
        let data = series.dataset("data").and_then(|data| {
            trace_sample_type(&data, &name)?;
            Ok(data)
        });
        let data = match data {
            Ok(data) => data,
            Err(err) => {
                return json!({
                    "series": name, "status": "refused", "check": "data shape/type", "message": format!("{err:#}"),
                })
            }
        };
        let mut timestamps_fit = None;
        let result = read_time_base_with(
            file,
            &name,
            data.shape()[0],
            |ds, n| {
                let grid = probe_timestamps(ds, n, true)?;
                timestamps_fit = Some(grid.probe_regular);
                Ok((grid.rate, grid.t0))
            },
            "",
        );
        if let Err(err) = result {
            let mut skipped = json!({
                "series": name, "status": "refused", "check": "time base", "message": format!("{err:#}"),
            });
            if let Some(&TimestampDiscontinuity {
                check, left, right, ..
            }) = err.downcast_ref()
            {
                skipped["check"] = json!(check);
                skipped["sample"] = json!(left.0);
                skipped["next_sample"] = json!(right.0);
                skipped["jump_s"] = json!(right.1 - left.1);
                skipped["t_s"] = json!([left.1, right.1]);
            }
            return skipped;
        }
        let (status, message) = match timestamps_fit {
            Some(true) => ("regular (probe)", "Sampled timestamps fit the inferred grid; the full scan was skipped."),
            Some(false) => ("not fully checked", "Sampled timestamps deviate from the inferred grid; the probe did not locate a discontinuity and the full scan was skipped."),
            None => ("not fully checked", "No per-sample timestamps; checked series metadata and the declared clock only."),
        };
        json!({ "series": name, "status": status, "message": message })
    }

    /// `(rate Hz, starting_time s, recorded timestamps)`.
    fn read_time_base_with(
        file: &Group,
        name: &str,
        n_samples: usize,
        check_timestamps: impl FnOnce(&Dataset, usize) -> Result<(f64, f64)>,
        hint: &str,
    ) -> Result<(f64, f64, Option<Dataset>)> {
        let read = || -> Result<_> {
            if let Some(ds) = file.optional_stored_dataset(&format!("{name}/timestamps"))? {
                ds.exact_float("timestamps")?;
                let (rate, t0) = check_timestamps(&ds, n_samples)?;
                return Ok((rate, t0, Some(ds)));
            }
            let st = file
                .optional_stored_dataset(&format!("{name}/starting_time"))?
                .context("neither starting_time nor timestamps")?;
            let rate = st.attr("rate").context("starting_time has no rate")?;
            st.exact_float("starting_time")?;
            let rate = checked_rate(rate.exact_number("starting_time rate")?)?;
            let t0: f64 = st.read_scalar()?;
            check_time(t0, "starting_time")?;
            Ok((rate, t0, None))
        };
        read().with_context(|| format!("ElectricalSeries {name}: invalid time base{hint}"))
    }

    const TIMESTAMP_BLOCK: usize = 65_536;

    fn read_timestamp_block(ds: &Dataset, range: Range<usize>) -> Result<Vec<f64>> {
        let times = ds.read_rows(range)?;
        for &t in &times {
            check_time(t, "timestamps value")?;
        }
        Ok(times)
    }

    #[derive(Debug, thiserror::Error)]
    #[error("{message}")]
    struct TimestampDiscontinuity {
        check: &'static str,
        left: (usize, f64),
        right: (usize, f64),
        message: String,
    }

    fn checked_timestamp_step(left: (usize, f64), right: (usize, f64)) -> Result<f64> {
        let step = right.1 - left.1;
        let back = -step;
        ensure!(
            step.is_finite() && step > 0.0,
            TimestampDiscontinuity {
                check: "timestamp order",
                left,
                right,
                message: format!(
                    "timestamps must strictly increase: they step back by {back:.6} s between samples {} and {} \
                     (t={} s to {} s); choose another series with --nwb-series",
                    left.0, right.0, left.1, right.1
                ),
            }
        );
        Ok(step)
    }

    struct TimestampGrid {
        rate: f64,
        t0: f64,
        period: f64,
        limit: f64,
        probe_regular: bool,
    }

    impl TimestampGrid {
        fn error(&self, sample: usize, t: f64) -> f64 {
            (t - self.t0) / self.period - sample as f64
        }
    }

    fn probe_timestamps(ds: &Dataset, n: usize, probe_small: bool) -> Result<TimestampGrid> {
        ensure!(
            ds.shape() == [n] && n >= 2,
            "timestamps must contain one time per sample and at least two times to infer a rate"
        );
        let time_at =
            |sample| -> Result<_> { Ok(read_timestamp_block(ds, sample..sample + 1)?[0]) };
        let t0 = time_at(0)?;
        let last = time_at(n - 1)?;
        if last <= t0 {
            // At least one half must retain the non-increasing endpoints.
            let (mut left, mut right) = ((0, t0), (n - 1, last));
            while right.0 - left.0 > 1 {
                let mid = left.0.midpoint(right.0);
                let t = time_at(mid)?;
                if t <= left.1 {
                    right = (mid, t);
                } else {
                    left = (mid, t);
                }
            }
            checked_timestamp_step(left, right)?;
        }
        // Use the end-to-end mean period, so a tiny bias from one rounded
        // interval cannot accumulate over a long recording.
        let period = (last - t0) / (n - 1) as f64;
        let rate =
            checked_rate(period.recip()).context("irregular timestamps: invalid mean interval")?;
        let scale = t0.abs().max(last.abs());
        let ulp = scale.next_up() - scale;
        // A 0.01-sample ULP caps the tolerance at 0.041 samples, well below
        // half a sample, and allows 30 kHz Unix-epoch timestamps (~0.0072).
        ensure!(ulp * rate <= 0.01,
            "timestamps are too coarse at this magnitude to verify regular sampling; if sampling was regular, rewrite the series with starting_time + rate");
        // The grid interpolates two rounded endpoints. Allow a few ulps
        // for their error, each rounded timestamp, and the arithmetic.
        let limit = 1e-3 + 4.0 * ulp * rate;
        let mut grid = TimestampGrid {
            rate,
            t0,
            period,
            limit,
            probe_regular: true,
        };
        // Don't probe if a full scan is planned and fits in one block.
        let stride = if !probe_small && n <= TIMESTAMP_BLOCK {
            n - 1
        } else {
            (n - 1).div_ceil(63)
        };
        // ponytail: fast refusal is best-effort and only localizes the first
        // failed probe. The full scan still refuses discontinuities that return
        // to the grid between probes or that the probe can't explain.
        let mut last_in_grid = (0, t0);
        for sample in (stride..n - 1).step_by(stride) {
            let time = time_at(sample)?;
            let drift = grid.error(sample, time);
            if drift.abs() <= limit {
                last_in_grid = (sample, time);
                continue;
            }
            let probe = (sample, time);
            let before = ((0, t0), probe);
            let after = (probe, (n - 1, last));
            let (longer, shorter) = if drift > 0.0 {
                (before, after)
            } else {
                (after, before)
            };
            // First bracket an in-grid to out-of-grid transition. A lasting
            // offset skews the mean, so this crossing may be on an ordinary
            // interval. Also look for a short step and a long gap on the
            // sides where each could account for the signed probe error.
            let mut largest_error = 2.0 * limit;
            let mut gap = None;
            for ((mut left, mut right), direction) in [
                ((last_in_grid, probe), None),
                (shorter, Some(-1.0)),
                (longer, Some(1.0)),
            ] {
                while right.0 - left.0 > 1 {
                    let mid = left.0.midpoint(right.0);
                    let t = time_at(mid)?;
                    let take_left = if let Some(sign) = direction {
                        let left_mean = (t - left.1) / (mid - left.0) as f64;
                        let right_mean = (right.1 - t) / (right.0 - mid) as f64;
                        sign * left_mean >= sign * right_mean
                    } else {
                        grid.error(mid, t).abs() > limit
                    };
                    if take_left {
                        right = (mid, t);
                    } else {
                        left = (mid, t);
                    }
                }
                let step = checked_timestamp_step(left, right)?;
                // Adjacent grid errors differ by the interval error in samples.
                let error = (step * rate - 1.0).abs();
                if error > largest_error {
                    largest_error = error;
                    gap = Some((left, right, step));
                }
            }
            if let Some((left, right, jump)) = gap {
                let jump_samples = jump * rate;
                return Err(TimestampDiscontinuity {
                    check: "timestamp grid",
                    left,
                    right,
                    message: format!(
                        "timestamps are not a regular grid: they jump by {jump:.6} s \
                         ({jump_samples:.3} samples at the mean rate {rate} Hz) between samples {} and {} \
                         (t={} s to {} s); Sorrel needs a regular sample grid for the trace; \
                         choose another series with --nwb-series",
                        left.0, right.0, left.1, right.1
                    ),
                }.into());
            }
            grid.probe_regular = false;
            break;
        }
        Ok(grid)
    }

    fn regular_timestamps(ds: &Dataset, n: usize) -> Result<(f64, f64)> {
        let grid = probe_timestamps(ds, n, false)?;
        let TimestampGrid {
            t0, rate, limit, ..
        } = grid;
        let (mut max_drift, mut worst_sample) = (0.0, 0);
        let (mut max_step, mut step_end) = (0.0, 1);
        let mut prev = t0;
        for start in (1..n).step_by(TIMESTAMP_BLOCK) {
            let times =
                read_timestamp_block(ds, start..start.saturating_add(TIMESTAMP_BLOCK).min(n))?;
            for (i, &t) in times.iter().enumerate() {
                let sample = start + i;
                let step = t - prev;
                ensure!(step.is_finite() && step > 0.0,
                    "irregular timestamps at sample {sample}: interval {step} s; timestamps must strictly increase");
                if step > max_step {
                    (max_step, step_end) = (step, sample);
                }
                let drift = grid.error(sample, t).abs();
                if drift > max_drift {
                    (max_drift, worst_sample) = (drift, sample);
                }
                prev = t;
            }
        }
        ensure!(max_drift <= limit,
            "irregular timestamps at sample {worst_sample}: max deviation {max_drift:.6} samples from the inferred grid (limit {limit:.6}); largest interval {max_step:.6} s between samples {} and {step_end}",
            step_end - 1);
        Ok((rate, t0))
    }

    /// Channel ids and probe positions for each data column, through the
    /// series' `electrodes` region.
    type Channels = (Vec<ChannelId>, Vec<[f32; 2]>);

    fn read_channels(f: &Group, series: &Group, name: &str, n_channels: usize) -> Result<Channels> {
        let region = series
            .dataset("electrodes")
            .with_context(|| format!("ElectricalSeries {name}: cannot open electrodes region"))?;
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

    fn trace_sample_type(data: &Dataset, name: &str) -> Result<SampleType> {
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
        Ok(dtype)
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
            let dtype = trace_sample_type(&data, name)?;
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
        series_clock: Option<(usize, Option<&Dataset>)>,
        mut comparison: Option<(f64, bool)>,
    ) -> Result<(Vec<Vec<SampleIndex>>, Vec<u32>, bool)> {
        let timestamps = series_clock.and_then(|(n_samples, ds)| ds.map(|ds| (ds, n_samples)));
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
                    check_time(t, "/units/spike_times value")?;
                    let s = exact_sample(t, t0, rate);
                    ensure!(s.is_finite() && (0.0..u64::MAX as f64).contains(&s)
                        && timestamps.is_none_or(|(_, n_samples)| (s as u64) < n_samples as u64),
                        "unit {id}: spike at {t} s is outside the sample index range or cannot be placed on a sample exactly (starting time {t0} s, {rate} Hz){}",
                        if series_clock.is_none() && t < 0.0 {
                            "; pair the raw NWB file with --dat to restore its time base"
                        } else {
                            ""
                        });
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
        // A short/long pair on a nominal 1 kHz grid.
        let mut times: Vec<_> = (0..127).map(|i| 0.5 + f64::from(i) / 1000.0).collect();
        times[3] -= 0.0002;
        let ds = file
            .new_dataset_builder()
            .with_data(&times)
            .create("timestamps")
            .unwrap();
        let err = regular_timestamps(&Dataset::Hdf5(ds.clone()), times.len()).unwrap_err();
        let err = err.to_string();
        assert!(err.contains("max deviation 0.200000 samples"), "{err}");
        assert!(
            err.contains("largest interval 0.001200 s between samples 3 and 4"),
            "{err}"
        );
    }

    #[cfg(all(test, feature = "hdf5"))]
    #[test]
    fn timestamp_validation_scans_one_block_without_probing() {
        let dir = tempfile::tempdir().unwrap();
        let file = hdf5_metno::File::create(dir.path().join("timestamps.h5")).unwrap();
        let mut times: Vec<_> = (0..TIMESTAMP_BLOCK).map(|i| i as f64 / 1000.0).collect();
        times[1041] -= 0.0002; // The first sparse probe would hit this sample.
        let ds = file
            .new_dataset_builder()
            .with_data(&times)
            .create("timestamps")
            .unwrap();
        let err = regular_timestamps(&Dataset::Hdf5(ds), times.len()).unwrap_err();
        assert!(
            err.to_string()
                .contains("at sample 1041: max deviation 0.200000 samples"),
            "{err}"
        );
    }

    #[cfg(all(test, feature = "hdf5"))]
    #[test]
    fn timestamp_validation_locates_one_dropped_sample() {
        let dir = tempfile::tempdir().unwrap();
        let file = hdf5_metno::File::create(dir.path().join("timestamps.h5")).unwrap();
        let times: Vec<_> = (0..=100_001)
            .filter(|&i| i != 73_259)
            .map(|i| f64::from(i) / 1000.0)
            .collect();
        let ds = file
            .new_dataset_builder()
            .with_data(&times)
            .create("timestamps")
            .unwrap();
        let err = regular_timestamps(&Dataset::Hdf5(ds), times.len())
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("timestamps are not a regular grid: they jump by 0.002000 s"),
            "{err}"
        );
        assert!(err.contains("between samples 73258 and 73259"), "{err}");
    }

    #[cfg(all(test, feature = "hdf5"))]
    #[test]
    fn timestamp_validation_reports_large_gap_at_the_adjacent_samples() {
        let dir = tempfile::tempdir().unwrap();
        let file = hdf5_metno::File::create(dir.path().join("gapped.h5")).unwrap();
        let name = "/acquisition/e-series";
        let series = file.create_group(name).unwrap();
        let root = Group::Hdf5(file.group("/").unwrap());
        let n = 100_001;
        let ds = series
            .new_dataset::<f64>()
            .shape(n)
            .create("timestamps")
            .unwrap();
        for gap in [1, 73_259, n - 1] {
            let times: Vec<_> = (0..n)
                .map(|i| i as f64 / 1000.0 + if i >= gap { 25.0 } else { 0.0 })
                .collect();
            ds.write(&times).unwrap();
            let err = read_time_base_with(&root, name, n, regular_timestamps, "")
                .err()
                .unwrap();
            assert_eq!(
                err.to_string(),
                format!("ElectricalSeries {name}: invalid time base")
            );
            let err = format!("{err:#}");
            assert!(err.contains("jump by 25.001000 s"), "{err}");
            assert!(
                err.contains("20000.800 samples at the mean rate 800"),
                "{err}"
            );
            assert!(
                err.contains(&format!("between samples {} and {gap}", gap - 1)),
                "{err}"
            );
            assert!(err.contains("--nwb-series"), "{err}");
        }
    }

    #[cfg(all(test, feature = "hdf5"))]
    #[test]
    fn timestamp_validation_reports_backward_step_at_the_adjacent_samples() {
        let dir = tempfile::tempdir().unwrap();
        let file = hdf5_metno::File::create(dir.path().join("backward.h5")).unwrap();
        let name = "/acquisition/e-series";
        let series = file.create_group(name).unwrap();
        let root = Group::Hdf5(file.group("/").unwrap());
        let n = 100_001;
        let ds = series
            .new_dataset::<f64>()
            .shape(n)
            .create("timestamps")
            .unwrap();
        // Test a temporary offset detected by a later probe and a permanent
        // offset that skews the endpoint mean before the step.
        for end in [74_847, n] {
            let times: Vec<_> = (0..n)
                .map(|i| {
                    i as f64 / 1000.0
                        - if (73_259..end).contains(&i) {
                            0.029
                        } else {
                            0.0
                        }
                })
                .collect();
            ds.write(&times).unwrap();
            let err = format!(
                "{:#}",
                read_time_base_with(&root, name, n, regular_timestamps, "")
                    .err()
                    .unwrap()
            );
            assert!(
                err.contains("invalid time base: timestamps must strictly increase: they step back by 0.028000 s between samples 73258 and 73259"),
                "{err}"
            );
            assert!(err.contains("--nwb-series"), "{err}");
        }
    }

    #[cfg(all(test, feature = "hdf5"))]
    #[test]
    fn timestamp_validation_locates_backward_step_with_nonincreasing_endpoints() {
        let dir = tempfile::tempdir().unwrap();
        let file = hdf5_metno::File::create(dir.path().join("timestamps.h5")).unwrap();
        let times: Vec<_> = (0..100_001)
            .map(|i| f64::from(i) / 1000.0 - if i >= 73_259 { 150.0 } else { 0.0 })
            .collect();
        let ds = file
            .new_dataset_builder()
            .with_data(&times)
            .create("timestamps")
            .unwrap();
        let err = regular_timestamps(&Dataset::Hdf5(ds), times.len()).unwrap_err();
        assert!(
            err.to_string().contains("timestamps must strictly increase: they step back by 149.999000 s between samples 73258 and 73259"),
            "{err}"
        );
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

#[cfg(feature = "hdf5")]
pub(crate) use imp::check_append_rate;
#[cfg(any(feature = "hdf5", feature = "zarr"))]
pub(crate) use imp::read_trials;
#[cfg(feature = "hdf5")]
pub(crate) use imp::storage::{read_str, strings};
#[cfg(any(feature = "hdf5", feature = "zarr"))]
pub use imp::{NwbOpenParams, NwbProvider};
