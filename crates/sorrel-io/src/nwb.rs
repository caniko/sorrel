//! NWB backend (local files).
//!
//! Reads `/units` and, optionally, one ElectricalSeries from `/acquisition`,
//! `/processing` or a second file. Uses a regular time base and reads trace
//! windows on demand as float32 µV. Enable with `--features hdf5`.

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
    use crate::extras::{HasGeometry, HasQualityMetrics, QualityMetrics};
    use crate::kilosort::{pinned_sorter, valid_tag, PhyLabel};
    use crate::provider::{
        appended_unit_id, ChannelId, ClusterId, DataProvider, SampleIndex, TraceSamples,
        TraceSlice, UnitAnnotation,
    };
    use anyhow::{anyhow, ensure, Context, Result};
    use hdf5_metno as hdf5;
    use hdf5_metno::types::{FloatSize, IntSize, TypeDescriptor, VarLenAscii, VarLenUnicode};
    use hdf5_metno::File as H5File;
    use std::collections::{HashMap, VecDeque};
    use std::path::Path;
    use std::sync::Mutex;

    const ELECTRODES: &str = "/general/extracellular_ephys/electrodes";

    #[derive(Clone, Copy, Debug, Default)]
    pub struct NwbOpenParams<'a> {
        /// Acquisition name or absolute HDF5 path, also applied to a paired file.
        pub series: Option<&'a str>,
        /// Read the ElectricalSeries from this file instead of the units file.
        pub raw_path: Option<&'a Path>,
        /// Sampling rate (Hz) when the units file lacks its own time base.
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
            let f = open_hdf5(path)?;
            let units = f
                .group("/units")
                .context("missing /units in the sorting NWB file")?;
            reject_external_links(
                &f,
                &units,
                &["spike_times", "spike_times_index", "id"],
                "open the NWB file that holds the units directly",
            )?;
            let spike_times = units
                .dataset("spike_times")
                .context("missing /units/spike_times")?;
            let raw = params.raw_path.map(open_hdf5).transpose()?;
            let recording = raw.as_ref().unwrap_or(&f);
            let series = find_electrical_series(recording, params.series, raw.is_some())?;
            let units_rate = if raw.is_some() || series.is_none() {
                let rate = match spike_times.attr("resolution") {
                    // -1 means unspecified in the NWB schema.
                    Ok(attr) => match attr.read_scalar::<f64>()? {
                        -1.0 => None,
                        resolution => Some(
                            checked_rate(resolution.recip())
                                .context("invalid /units/spike_times resolution")?,
                        ),
                    },
                    Err(_) => None,
                };
                if let (Some(declared), Some(explicit)) = (rate, params.sample_rate) {
                    check_matching_rate(
                        "--sample-rate",
                        explicit,
                        "/units/spike_times resolution",
                        declared,
                        false,
                    )?;
                }
                rate.or(params.sample_rate).map(checked_rate).transpose()?
            } else {
                None
            };

            let mut legacy_identity = false;
            let (trace, rate, t0, (channel_map, channel_positions)) = if let Some((name, series)) =
                &series
            {
                reject_external_links(
                    recording,
                    series,
                    &["starting_time", "timestamps"],
                    "open the NWB file that holds the series directly, or pass it via --dat",
                )?;
                let data = series.dataset("data")?;
                let shape = data.shape();
                let (n_samples, n_channels) = (shape[0], shape.get(1).copied().unwrap_or(1));
                let (stored_rate, t0, float32_rate) = read_time_base(series, name, n_samples)?;
                let mut rate = stored_rate;
                if let Some(expected) = units_rate.or(params.sample_rate) {
                    check_matching_rate("sorting sample rate", expected, name, rate, float32_rate)?;
                    if float32_rate {
                        rate = expected;
                    }
                }
                let channels = read_channels(recording, series, name, n_channels)?;
                let trace = NwbTrace::new(series, data, name, n_samples, n_channels)?;
                legacy_identity = raw.is_none()
                    && rate == stored_rate
                    && !series.link_exists("timestamps")
                    && is_trunk_series(&f, name);
                (Some(trace), rate, t0, channels)
            } else {
                (
                        None,
                        units_rate.context(
                            "no ElectricalSeries time base in the sorting NWB and no /units/spike_times resolution; pass --sample-rate HZ",
                        )?,
                        0.0,
                        read_units_only_channels(&f)?,
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
            let n_samples = trace.as_ref().map_or(0, |t| t.n_samples as u64);
            let (spikes_per_cluster, unit_ids) =
                load_units(&units, &spike_times, rate, t0).context("load /units")?;
            if raw.is_some() {
                ensure!(spikes_per_cluster.iter().filter_map(|s| s.last()).all(|s| s.0 < n_samples),
                    "spike times extend past the paired ElectricalSeries (starting time {t0} s, {n_samples} samples at {rate} Hz)");
            }
            let mut identity_bytes = provider_identity_bytes(path);
            if !legacy_identity {
                if let Some((name, _)) = &series {
                    identity_bytes.extend_from_slice(name.as_bytes());
                }
                identity_bytes.push(0); // Terminate the path (empty for units-only).
                identity_bytes.extend(rate.to_le_bytes());
                if let Some(raw_path) = params.raw_path {
                    identity_bytes.extend(provider_identity_bytes(raw_path));
                }
            }
            let metadata = read_unit_metadata(&units, &unit_ids)?;
            Ok(Self {
                rate,
                t0,
                n_channels: u32::try_from(channel_map.len()).context("too many NWB channels")?,
                n_samples: SampleIndex(n_samples),
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

    fn open_hdf5(path: &Path) -> Result<H5File> {
        H5File::open(path).with_context(|| format!("open NWB {}", path.display()))
    }

    fn checked_rate(rate: f64) -> Result<f64> {
        ensure!(
            (rate as f32).is_finite() && (rate as f32) > 0.0,
            "invalid sample rate {rate} Hz"
        );
        Ok(rate)
    }

    fn timestamp_epoch(file: &H5File) -> Result<jiff::Timestamp> {
        // NWB defines time zero separately from the experiment's start time.
        let ds = file.dataset("/timestamps_reference_time")?;
        let value = read_str(&ds)?;
        value
            .parse()
            .map_err(|err| anyhow!("invalid NWB epoch {value:?}: {err}"))
    }

    fn check_matching_rate(
        a_name: &str,
        a: f64,
        b_name: &str,
        b: f64,
        b_float32: bool,
    ) -> Result<()> {
        let a = checked_rate(a)?;
        // Only allow round-off from reciprocals or timestamp-derived rates,
        // not a nominally similar clock that could drift by many samples.
        ensure!(
            if b_float32 {
                a as f32 == b as f32
            } else {
                (a - b).abs() <= 1e-9 * b
            },
            "NWB rate mismatch: {a_name} is {a} Hz but {b_name} is {b} Hz"
        );
        Ok(())
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

    fn read_str(data: &hdf5::Container) -> Result<String> {
        ensure!(data.is_scalar(), "expected a scalar HDF5 string");
        strings(data)?.pop().context("missing HDF5 string")
    }

    fn strings(data: &hdf5::Container) -> Result<Vec<String>> {
        let dtype = data.dtype()?;
        match dtype.to_descriptor()? {
            TypeDescriptor::VarLenUnicode => Ok(data
                .read_raw::<VarLenUnicode>()?
                .iter()
                .map(ToString::to_string)
                .collect()),
            TypeDescriptor::VarLenAscii => Ok(data
                .read_raw::<VarLenAscii>()?
                .iter()
                .map(ToString::to_string)
                .collect()),
            TypeDescriptor::FixedAscii(width) | TypeDescriptor::FixedUnicode(width) => {
                fixed_strings(data, &dtype, width)
            }
            _ => Err(anyhow!("expected an HDF5 string datatype")),
        }
    }

    fn fixed_strings(
        data: &hdf5::Container,
        dtype: &hdf5::Datatype,
        width: usize,
    ) -> Result<Vec<String>> {
        use hdf5_metno_sys::{
            h5a::H5Aread,
            h5d::H5Dread,
            h5i::{H5Iget_type, H5I_ATTR},
            h5p::H5P_DEFAULT,
            h5s::H5Sget_simple_extent_npoints,
            h5t::{H5T_str_t::H5T_STR_SPACEPAD, H5Tget_strpad},
        };

        // hdf5-metno's fixed strings require a compile-time width. Read with
        // the file datatype as the memory datatype instead (no conversion).
        let space = data.space()?;
        // SAFETY: space is a live HDF5 dataspace, accessed under the HDF5 lock.
        let count = hdf5::sync::sync(|| unsafe {
            hdf5::h5check(H5Sget_simple_extent_npoints(space.id()))
        })?;
        let count = usize::try_from(count)?;
        if count == 0 {
            return Ok(Vec::new());
        }
        let len = count
            .checked_mul(width)
            .filter(|&n| n != 0)
            .context("invalid HDF5 string size")?;
        let mut bytes = Vec::<u8>::new();
        bytes.try_reserve_exact(len)?;
        bytes.resize(len, 0);
        let space_pad = hdf5::sync::sync(|| {
            // SAFETY: data and dtype are live; bytes holds count * width bytes
            // for the whole dataspace, and the HDF5 lock serializes the calls.
            unsafe {
                let buf = bytes.as_mut_ptr().cast();
                hdf5::h5check(if H5Iget_type(data.id()) == H5I_ATTR {
                    H5Aread(data.id(), dtype.id(), buf)
                } else {
                    H5Dread(
                        data.id(),
                        dtype.id(),
                        space.id(),
                        space.id(),
                        H5P_DEFAULT,
                        buf,
                    )
                })?;
                Ok::<_, hdf5::Error>(H5Tget_strpad(dtype.id()) == H5T_STR_SPACEPAD)
            }
        })?;
        bytes
            .chunks_exact(width)
            .map(|s| {
                let s = std::str::from_utf8(s.split(|&b| b == 0).next().unwrap_or_default())?;
                Ok(if space_pad {
                    s.trim_end_matches(' ')
                } else {
                    s
                }
                .to_owned())
            })
            .collect()
    }

    pub(crate) fn read_trials(path: &Path) -> Result<crate::Events> {
        let f = open_hdf5(path)?;
        let g = f
            .group("intervals/trials")
            .with_context(|| format!("{}: no /intervals/trials table", path.display()))?;
        let starts: Vec<f64> = g.dataset("start_time")?.read_raw()?;
        let names = ["trial_type", "stimulus", "condition"]
            .into_iter()
            .filter_map(|c| g.dataset(c).ok())
            .find_map(|ds| {
                strings(&ds)
                    .ok()
                    .filter(|names| names.len() == starts.len())
            });
        let mut events = crate::Events::default();
        for (i, t) in starts.into_iter().enumerate() {
            events.push(t, names.as_ref().map_or("trial", |n| n[i].as_str()));
        }
        Ok(events)
    }

    /// `neurodata_type_def → neurodata_type_inc` for every type in the
    /// file's cached specifications (core and extensions).
    fn type_parents(
        f: &H5File,
        read: impl Fn(&hdf5::Container) -> Result<String>,
    ) -> HashMap<String, String> {
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
                let text = read(&ds);
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
    fn is_trunk_series(f: &H5File, selected: &str) -> bool {
        fn read_legacy_str(data: &hdf5::Container) -> Result<String> {
            Ok(data
                .read_scalar::<VarLenUnicode>()
                .map(|s| s.to_string())
                .or_else(|_| data.read_scalar::<VarLenAscii>().map(|s| s.to_string()))?)
        }
        let Some(name) = selected.strip_prefix("/acquisition/") else {
            return false;
        };
        let Ok(acq) = f.group("/acquisition") else {
            return false;
        };
        let Ok(names) = acq.member_names() else {
            return false;
        };
        let parents = type_parents(f, read_legacy_str);
        names
            .into_iter()
            .find(|name| {
                let Ok(g) = acq.group(name) else { return false };
                g.attr("neurodata_type")
                    .ok()
                    .and_then(|a| read_legacy_str(&a).ok())
                    .is_some_and(|t| is_electrical_series(&t, &parents))
                    && g.dataset("data").is_ok_and(|d| d.ndim() == 2)
            })
            .is_some_and(|first| first == name)
    }

    fn find_electrical_series(
        f: &H5File,
        requested: Option<&str>,
        required: bool,
    ) -> Result<Option<(String, hdf5::Group)>> {
        fn collect(
            group: &hdf5::Group,
            path: &str,
            depth: usize,
            is_series: &impl Fn(&str) -> bool,
            found: &mut Vec<(String, hdf5::Group)>,
        ) -> Result<()> {
            for name in group.member_names()? {
                if group.loc_type_by_name(&name).ok() != Some(hdf5::LocationType::Group) {
                    continue;
                }
                let child = group.group(&name)?;
                let path = format!("{path}/{name}");
                if child
                    .attr("neurodata_type")
                    .ok()
                    .and_then(|a| read_str(&a).ok())
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

        let parents = type_parents(f, read_str);
        let is_series = |t: &str| is_electrical_series(t, &parents);
        let mut found = Vec::new();
        for (root, depth) in [("/acquisition", 0), ("/processing", 32)] {
            if f.link_exists(root) {
                collect(&f.group(root)?, root, depth, &is_series, &mut found)?;
            }
        }
        found.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let has_2d_data = |g: &hdf5::Group| g.dataset("data").is_ok_and(|d| d.ndim() == 2);
        let supported_data = |g: &hdf5::Group| {
            g.dataset("data").is_ok_and(|d| match d.shape().as_slice() {
                [_] => true,
                [_, channels] => *channels > 0,
                _ => false,
            })
        };
        // Keep the first 2-D acquisition as the historical default, but list
        // unsupported shapes too so a selection error can name them.
        let default_index = found
            .iter()
            .position(|(_, g)| has_2d_data(g) && supported_data(g))
            .or_else(|| found.iter().position(|(_, g)| supported_data(g)))
            .unwrap_or(0);
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
            found
                .iter()
                .position(|(name, _)| *name == path)
                .with_context(|| {
                    format!(
                        "no ElectricalSeries matching {requested:?}; available series: {}",
                        available()
                    )
                })?
        } else if found.is_empty() {
            ensure!(
                !required,
                "no ElectricalSeries in the raw NWB; available series: none"
            );
            return Ok(None);
        } else {
            default_index
        };
        let (path, group) = &found[index];
        ensure!(
            supported_data(group),
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

    // The journal only seals the explicitly opened files. Check link metadata
    // before reading time-base or unit data, even when the external target is missing.
    fn reject_external_links(
        file: &H5File,
        group: &hdf5::Group,
        names: &[&str],
        advice: &str,
    ) -> Result<()> {
        let fileno = file.loc_info()?.fileno;
        ensure!(
            group.loc_info()?.fileno == fileno,
            "{} resolves through an HDF5 external link; {advice}",
            group.name()
        );
        let links = group.iter_visit_default(Vec::new(), |_, name, info, links| {
            if names.contains(&name) && info.link_type != hdf5::LinkType::Hard {
                links.push((name.to_owned(), info.link_type));
            }
            true
        })?;
        for (name, kind) in links {
            // A soft link is local only if its resolved object is in this file.
            ensure!(
                kind == hdf5::LinkType::Soft && group.loc_info_by_name(&name)?.fileno == fileno,
                "{}/{name} resolves through an HDF5 external link; {advice}",
                group.name()
            );
        }
        Ok(())
    }

    /// `(rate Hz, starting_time s, rate stored as float32)`.
    fn read_time_base(g: &hdf5::Group, name: &str, n_samples: usize) -> Result<(f64, f64, bool)> {
        let read = || {
            if g.link_exists("timestamps") {
                let (rate, t0) = regular_timestamps(&g.dataset("timestamps")?, n_samples)?;
                return Ok((rate, t0, false));
            }
            let st = g
                .dataset("starting_time")
                .context("neither starting_time nor timestamps")?;
            let rate = st.attr("rate").context("starting_time has no rate")?;
            let float32_rate =
                rate.dtype()?.to_descriptor()? == TypeDescriptor::Float(FloatSize::U4);
            let rate = checked_rate(rate.read_scalar()?)?;
            let t0: f64 = st.read_scalar()?;
            ensure!(t0.is_finite(), "invalid starting_time {t0}");
            Ok((rate, t0, float32_rate))
        };
        read().with_context(|| format!("ElectricalSeries {name}: invalid time base"))
    }

    fn regular_timestamps(ds: &hdf5::Dataset, n: usize) -> Result<(f64, f64)> {
        const BLOCK: usize = 65_536;
        ensure!(
            ds.shape() == [n] && n >= 2,
            "timestamps must contain one time per sample and at least two times to infer a rate"
        );
        let t0 = ds.read_slice_1d::<f64, _>(0..1)?[0];
        let last = ds.read_slice_1d::<f64, _>(n - 1..n)?[0];
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
        for start in (1..n).step_by(BLOCK) {
            let times = ds.read_slice_1d::<f64, _>(start..start.saturating_add(BLOCK).min(n))?;
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

    fn read_channels(
        f: &H5File,
        series: &hdf5::Group,
        name: &str,
        n_channels: usize,
    ) -> Result<Channels> {
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
        read_electrodes(&table, Some(rows))
    }

    fn read_units_only_channels(f: &H5File) -> Result<Channels> {
        // The parent group may also be absent; avoid a nested link_exists
        // call on an absent parent (an HDF5 error, not just a missing link).
        match f.group(ELECTRODES) {
            Ok(table) => read_electrodes(&table, None),
            Err(_) => Ok(Channels::default()),
        }
    }

    fn read_electrodes(table: &hdf5::Group, rows: Option<Vec<i64>>) -> Result<Channels> {
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
            let dtype = data.dtype()?.to_descriptor().with_context(|| {
                format!("ElectricalSeries {name}: unsupported HDF5 sample datatype")
            })?;
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
                    let raw = if self.data.ndim() == 1 {
                        self.data
                            .read_slice_1d::<$ty, _>(start..end)?
                            .into_raw_vec_and_offset()
                            .0
                    } else {
                        self.data
                            .read_slice_2d::<$ty, _>((start..end, ..))?
                            .into_raw_vec_and_offset()
                            .0
                    };
                    raw.into_iter()
                        .zip(self.gain.iter().cycle())
                        .map(|(v, g)| (v as f64 * g + self.offset_uv) as f32)
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
        if let Some(unit) = data.attr("unit").ok().and_then(|a| read_str(&a).ok()) {
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
    fn load_units(
        units: &hdf5::Group,
        spike_times: &hdf5::Dataset,
        rate: f64,
        t0: f64,
    ) -> Result<(Vec<Vec<SampleIndex>>, Vec<u32>)> {
        let ids: Vec<i64> = units
            .dataset("id")
            .context("missing /units/id")?
            .read_raw()?;
        let all: Vec<f64> = spike_times.read_raw()?;
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
                    ensure!(s.is_finite() && (0.0..u64::MAX as f64).contains(&s),
                        "unit {id}: spike at {t} s is outside the sample index range (starting time {t0} s, {rate} Hz)");
                    Ok(SampleIndex::new(s as u64))
                })
                .collect::<Result<Vec<_>>>()?;
            bucket.sort_unstable();
            out.push(bucket);
            start = end;
        }
        Ok((out, unit_ids))
    }

    struct UnitMetadata {
        labels: Vec<PhyLabel>,
        tags: Vec<Vec<String>>,
        annotations: Vec<UnitAnnotation>,
        names: Option<Vec<String>>,
        metrics: QualityMetrics,
    }

    fn read_unit_metadata(units: &hdf5::Group, ids: &[u32]) -> Result<UnitMetadata> {
        let n = ids.len();
        let string_column = |name| {
            let ds = units.dataset(name).ok()?;
            (ds.shape() == [n]).then(|| strings(&ds).ok()).flatten()
        };
        let label_columns: Vec<_> = ["label", "quality", "KSLabel"]
            .into_iter()
            .filter_map(string_column)
            .collect();
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

        let index = |column: &str| -> Option<Vec<usize>> {
            let ds = units.dataset(&format!("{column}_index")).ok()?;
            if ds.shape() != [n] {
                return None;
            }
            ds.read_raw::<u64>()
                .ok()?
                .into_iter()
                .map(|v| usize::try_from(v).ok())
                .collect()
        };
        let tags = (|| {
            let ends = index("tags")?;
            let ds = units.dataset("tags").ok()?;
            let values = strings(&ds).ok()?;
            let mut rows = ragged_rows(&values, &ends)?;
            for row in &mut rows {
                row.retain(|tag| valid_tag(tag));
            }
            Some(rows)
        })()
        .unwrap_or_default();
        let mut annotations = vec![UnitAnnotation::default(); n];
        if let Some(origins) = string_column("origin") {
            for (annotation, origin) in annotations.iter_mut().zip(origins) {
                annotation.origin = origin;
            }
        }
        if let Some(parents) = (|| {
            let ends = index("parent_unit_ids")?;
            let ds = units.dataset("parent_unit_ids").ok()?;
            let values = ds.read_raw::<i64>().ok()?;
            ragged_rows(&values, &ends)
        })() {
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
        let names = string_column("unit_name").filter(|names| {
            let mut unique = std::collections::HashSet::new();
            names
                .iter()
                .all(|name| !name.is_empty() && unique.insert(name))
        });

        let mut metrics = QualityMetrics::new();
        for name in units.member_names()? {
            let Ok(ds) = units.dataset(&name) else {
                continue;
            };
            // The Python exporter marks its QC columns explicitly; a numeric
            // foreign Units column alone does not make it a quality metric.
            if ds.shape() == [n]
                && ds
                    .attr("description")
                    .ok()
                    .and_then(|a| read_str(&a).ok())
                    .is_some_and(|d| d.starts_with("Sorrel QC metric. "))
            {
                if let Ok(values) = ds.read_raw::<f32>() {
                    metrics.insert(name, values);
                }
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
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../release/fixtures/nwb/good.nwb");
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

    #[cfg(test)]
    #[test]
    fn timestamp_validation_rejects_subsample_jitter() {
        let dir = tempfile::tempdir().unwrap();
        let file = H5File::create(dir.path().join("timestamps.h5")).unwrap();
        // Alternating 0.8/1.2 ms intervals on a nominal 1 kHz grid.
        let times = [0.5, 0.5008, 0.502, 0.5028, 0.504];
        let ds = file
            .new_dataset_builder()
            .with_data(&times)
            .create("timestamps")
            .unwrap();
        let err = regular_timestamps(&ds, times.len()).unwrap_err();
        assert!(err.to_string().contains("max deviation 0.200000 samples"));
    }

    #[cfg(test)]
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
                let file = H5File::open_rw(&path).unwrap();
                file.unlink(dataset).unwrap();
                file.link_external(source.to_str().unwrap(), dataset, dataset)
                    .unwrap();
            }
            let error = NwbProvider::open(&path).err().expect(dataset);
            assert!(
                format!("{error:#}")
                    .contains(&format!("{dataset} resolves through an HDF5 external link")),
                "{error:#}"
            );
        }
    }

    #[cfg(test)]
    #[test]
    fn timestamp_validation_rejects_coarse_epoch_clock() {
        let dir = tempfile::tempdir().unwrap();
        let file = H5File::create(dir.path().join("timestamps.h5")).unwrap();
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
        let error = regular_timestamps(&ds, times.len()).unwrap_err();
        assert!(error.to_string().contains("timestamps are too coarse"));
    }

    #[cfg(test)]
    #[test]
    fn timestamp_validation_checks_the_interval_between_read_blocks() {
        let dir = tempfile::tempdir().unwrap();
        let file = H5File::create(dir.path().join("timestamps.h5")).unwrap();
        let origin = 1_700_000_000.0;
        let mut times: Vec<f64> = (0..65_540)
            .map(|i| origin + f64::from(i) / 30_000.0)
            .collect();
        let ds = file
            .new_dataset_builder()
            .with_data(&times)
            .create("timestamps")
            .unwrap();
        let (rate, t0) = regular_timestamps(&ds, times.len()).unwrap();
        assert!((rate - 30_000.0).abs() < 0.01);
        assert_eq!(t0, origin);
        // Duplicate the first timestamp in the second block at the boundary.
        times[65_537] = times[65_536];
        ds.write(&times).unwrap();
        let error = regular_timestamps(&ds, times.len()).unwrap_err();
        assert!(error
            .to_string()
            .contains("irregular timestamps at sample 65537"));
    }
}

#[cfg(feature = "hdf5")]
pub(crate) use imp::read_trials;
#[cfg(feature = "hdf5")]
pub use imp::{NwbOpenParams, NwbProvider};
