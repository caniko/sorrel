//! SpikeInterface `SortingAnalyzer` (native binary folder format) backend.
//!
//! Layout we support — what `SortingAnalyzer.save_as(format='binary_folder')`
//! emits:
//! ```text
//! analyzer/
//!   sorting/
//!     spikes.npy            structured: (sample_index, unit_index, ...)
//!     unit_ids.npy
//!     properties/...
//!   recording.json          BinaryRecordingExtractor or BinaryFolderRecording
//!                           description (or binary.json) — rate, dtype, bin path —
//!                           optionally under channel- and sample-preserving
//!                           preprocessors (bandpass, CAR, whiten, ...)
//!   extensions/
//!     waveforms/
//!     templates/templates_average.npy
//!     quality_metrics/metrics.csv
//!     ...
//! ```
//!
//! Zarr ("zarr_folder") format is **not** supported here; users save to
//! `binary_folder` first.
//!
//! This backend is more conservative than [`KilosortProvider`]: it requires
//! a sorting and a recording, but everything else (templates, waveforms, PCs)
//! is optional and surfaced through the existing capability traits.

use crate::extras::{HasGeometry, HasQualityMetrics};
use crate::kilosort::PhyLabel;
use crate::npy::{read_1d_u32, read_header};
use crate::provider::{
    ChannelId, ClusterId, DataProvider, RawFile, SampleIndex, TraceDtype, TraceSamples, TraceSlice,
};
use anyhow::{anyhow, bail, Context, Result};
use memmap2::Mmap;
use std::fs::File;
use std::path::{Path, PathBuf};

pub struct SortingAnalyzerProvider {
    sample_rate: f32,
    n_channels: u32,
    n_samples: SampleIndex,
    dtype: TraceDtype,

    spikes_per_cluster: Vec<Vec<SampleIndex>>,
    initial_labels: Vec<PhyLabel>,
    /// Numeric unit ids, parallel to `spikes_per_cluster`. Can be sparse.
    /// Dense indices when SI's ids are not all integers.
    unit_ids: Vec<u32>,
    /// SI's own ids when it stored strings (`"0"`, `"007"`, `"unit_a"`), as
    /// SpikeInterface 0.105's generators and many extractors do; written
    /// back verbatim because SpikeInterface compares ids by value and type.
    unit_names: Option<Vec<String>>,
    /// First id for clusters sorrel creates: past every numeric id and every
    /// SI string that reads as one, so a new id never names an old unit.
    next_unit_id: u32,

    channel_positions: Vec<[f32; 2]>,
    channel_map: Vec<ChannelId>,

    metrics: crate::extras::QualityMetrics,
    identity_bytes: Vec<u8>,

    /// The mapped recording bin, for exporters.
    raw: Option<RawFile>,

    _trace_mmap: Option<Mmap>,
    trace_ptr: *const u8,
    trace_byte_len: usize,
}

unsafe impl Send for SortingAnalyzerProvider {}
unsafe impl Sync for SortingAnalyzerProvider {}

impl SortingAnalyzerProvider {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_labels(root, None)
    }

    /// [`Self::open`] with the initial labels read from `labels` instead of
    /// `<root>/curation.json`: the copy taken when the session's journal was
    /// created (see `sorrel_data::open::open_sorting_analyzer`), so a save
    /// that rewrote `curation.json` cannot change what the journal replays
    /// onto. `None` reads `<root>/curation.json`, as before.
    pub fn open_with_labels(root: impl AsRef<Path>, labels: Option<&Path>) -> Result<Self> {
        let root = root.as_ref();
        let recording_meta = load_recording_json(root)?;

        // Sorting: spikes.npy is structured numpy, but binary_folder also
        // emits separate spike_times and unit_index arrays under
        // sorting/numpy_folder when `save_as('numpy_folder', ...)` was used.
        // We accept either layout.
        let (times, unit_indices) = load_sorting(root)?;

        let unit_ids_path = root.join("sorting").join("unit_ids.npy");
        let (unit_ids, unit_names) = if unit_ids_path.exists() {
            (read_1d_u32(&unit_ids_path)?, None)
        } else if let Some(ids) = numpysorting_unit_ids(root)? {
            ids
        } else {
            // Fall back to dense unit numbering [0..max+1).
            let max = unit_indices.iter().copied().max().unwrap_or(0);
            ((0..=max).collect(), None)
        };
        let n_clusters = unit_ids.len();
        let next_unit_id = unit_ids
            .iter()
            .copied()
            .chain(unit_names.iter().flatten().filter_map(|s| s.parse().ok()))
            .max()
            .map_or(0, |m: u32| m + 1);

        let mut spikes_per_cluster: Vec<Vec<SampleIndex>> = vec![Vec::new(); n_clusters];
        for (i, &uidx) in unit_indices.iter().enumerate() {
            if let Some(b) = spikes_per_cluster.get_mut(uidx as usize) {
                b.push(times[i]);
            }
        }
        for b in spikes_per_cluster.iter_mut() {
            b.sort_unstable();
        }
        // SI has no phy-style label sidecar; a previous sorrel session (or
        // `spikeinterface.curation`) may have left a `curation.json`.
        let default_labels = root.join("curation.json");
        let initial_labels = crate::curation_json::read_curation_labels(
            labels.unwrap_or(&default_labels),
            n_clusters,
            |uid| {
                match &unit_names {
                    // SI's own string first, then the number it reads as ("7" vs 7).
                    Some(names) => names.iter().position(|n| n == uid),
                    None => None,
                }
                .or_else(|| {
                    let u: u32 = uid.parse().ok()?;
                    unit_ids.iter().position(|&x| x == u)
                })
            },
        )
        .unwrap_or_else(|| vec![PhyLabel::Unsorted; n_clusters]);

        // Probe geometry via probeinterface JSON if SI exported one.
        let mut channel_positions = Vec::new();
        let mut channel_map = Vec::new();
        for candidate in ["probegroup.json", "probe.json", "extensions/probe.json"] {
            let p = root.join(candidate);
            if p.exists() {
                if let Ok(geom) = crate::probeinterface::ProbeGeometry::read(&p) {
                    channel_positions = geom.channel_positions;
                    channel_map = geom.channel_map;
                    break;
                }
            }
        }

        let metrics = load_quality_metrics_csv(
            &root
                .join("extensions")
                .join("quality_metrics")
                .join("metrics.csv"),
            n_clusters,
        )
        .unwrap_or_default();

        // Map raw recording. SI stores either a relative path (against
        // `recording_meta.base`) or an absolute path.
        let (trace_mmap, trace_ptr, trace_byte_len, n_samples) = map_recording(&recording_meta)?;
        let identity_bytes = provider_identity_bytes(root, &recording_meta);
        let raw = trace_mmap
            .as_ref()
            .and(recording_meta.file_paths.first())
            .map(|p| RawFile {
                path: resolve_path(&recording_meta.base, p),
                dtype: recording_meta.dtype,
                offset: recording_meta.offset,
                hp_filtered: false,
            });

        Ok(Self {
            sample_rate: recording_meta.sample_rate,
            n_channels: recording_meta.n_channels,
            n_samples,
            dtype: recording_meta.dtype,
            spikes_per_cluster,
            initial_labels,
            unit_ids,
            unit_names,
            next_unit_id,
            channel_positions,
            channel_map,
            metrics,
            identity_bytes,
            raw,
            _trace_mmap: trace_mmap,
            trace_ptr,
            trace_byte_len,
        })
    }

    fn samples_window(&self, start: SampleIndex, len: u32) -> TraceSamples<'_> {
        if self.trace_byte_len == 0 {
            return TraceSamples::I16(&[]);
        }
        let nc = self.n_channels as usize;
        let s = start.idx().saturating_mul(nc);
        let e = s + (len as usize) * nc;
        let bytes = unsafe { std::slice::from_raw_parts(self.trace_ptr, self.trace_byte_len) };
        TraceSamples::from_bytes_clamped(bytes, self.dtype, s, e)
    }
}

#[derive(Debug, Clone)]
struct RecordingMeta {
    sample_rate: f32,
    n_channels: u32,
    dtype: TraceDtype,
    /// Paths as written in the JSON; relative ones resolve against `base`.
    file_paths: Vec<PathBuf>,
    /// Folder the JSON they came from resolves its relative paths against.
    base: PathBuf,
    /// Bytes to skip at the start of the bin file (rare; SI usually 0).
    offset: u64,
}

/// SI writes relative paths against the folder holding the JSON
/// (`to_dict(relative_to=...)`); absolute ones are already final.
fn resolve_path(base: &Path, p: &Path) -> PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        base.join(p)
    }
}

/// Unit ids from `sorting/numpysorting_info.json` (SpikeInterface's
/// `NumpyFolderSorting`, the sorting of every binary_folder analyzer since
/// 0.101): the numeric ids, plus SI's strings when it stored strings. String
/// ids that are not distinct integers (`"unit_a"`, `"7"` beside `"007"`)
/// number the units densely. `None` if the file is absent or a numeric id
/// does not fit sorrel's `u32`.
type UnitIds = (Vec<u32>, Option<Vec<String>>);
fn numpysorting_unit_ids(root: &Path) -> Result<Option<UnitIds>> {
    let p = root.join("sorting").join("numpysorting_info.json");
    if !p.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
    let v: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("parse {}", p.display()))?;
    let Some(ids) = v.get("unit_ids").and_then(|x| x.as_array()) else {
        return Ok(None);
    };
    if !ids.iter().any(|x| x.is_string()) {
        let parsed: Option<Vec<u32>> = ids
            .iter()
            .map(|x| x.as_u64().and_then(|u| u32::try_from(u).ok()))
            .collect();
        return Ok(parsed.map(|ids| (ids, None)));
    }
    let names: Vec<String> = ids
        .iter()
        .map(|x| x.as_str().map_or_else(|| x.to_string(), str::to_owned))
        .collect();
    let parsed: Option<Vec<u32>> = names.iter().map(|s| s.parse().ok()).collect();
    let numeric = parsed
        .filter(|p| p.iter().collect::<std::collections::HashSet<_>>().len() == p.len())
        .unwrap_or_else(|| (0..names.len() as u32).collect());
    Ok(Some((numeric, Some(names))))
}

fn load_recording_json(root: &Path) -> Result<RecordingMeta> {
    for name in ["recording.json", "binary.json", "recording/recording.json"] {
        let p = root.join(name);
        if p.exists() {
            let text =
                std::fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
            // SI dumps these relative to the analyzer folder.
            return parse_recording_json_at(&text, &p, root, false);
        }
    }
    bail!(
        "SortingAnalyzer: no recording.json/binary.json found in {}",
        root.display()
    );
}

#[cfg(test)]
fn parse_recording_json(text: &str, json_path: &Path) -> Result<RecordingMeta> {
    let base = json_path.parent().unwrap_or(Path::new("."));
    parse_recording_json_at(text, json_path, base, false)
}

/// Preprocessors that keep every channel and every sample. Sorrel reads the
/// binary source beneath them and does **not** re-apply the chain: the trace
/// is the source samples through sorrel's own highpass, and the provider
/// reports `hp_filtered: false`.
const CHANNEL_AND_SAMPLE_PRESERVING: &[&str] = &[
    "BandpassFilterRecording",
    "HighpassFilterRecording",
    "FilterRecording",
    "NotchFilterRecording",
    "CommonReferenceRecording",
    "WhitenRecording",
    "PhaseShiftRecording",
    "ScaleRecording",
    "CenterRecording",
    "ZScoreRecording",
    "NormalizeByQuantileRecording",
    "HighpassSpatialFilterRecording",
];

/// Parse a recording description whose relative paths resolve against `base`.
/// `from_binary` is set once a BinaryFolderRecording was followed into its
/// `binary.json`, which must itself describe a binary extractor.
fn parse_recording_json_at(
    text: &str,
    json_path: &Path,
    base: &Path,
    from_binary: bool,
) -> Result<RecordingMeta> {
    let v: serde_json::Value =
        serde_json::from_str(text).with_context(|| format!("parse {}", json_path.display()))?;
    parse_recording_value(&v, json_path, base, from_binary)
}

fn parse_recording_value(
    v: &serde_json::Value,
    json_path: &Path,
    base: &Path,
    from_binary: bool,
) -> Result<RecordingMeta> {
    let kwargs = v.get("kwargs").unwrap_or(v);
    let class = v.get("class").and_then(|x| x.as_str()).unwrap_or("unknown");
    // SI writes the Python path, e.g.
    // `spikeinterface.preprocessing.filter.BandpassFilterRecording`.
    let class_name = class.rsplit('.').next().unwrap_or(class);
    if class_name == "ZarrRecordingExtractor" {
        bail!(
            "{}: zarr recording ({class}): sorrel cannot read zarr; re-save the recording \
             as binary with recording.save(format=\"binary\", folder=...) and recreate the \
             analyzer on it (analyzer.save_as keeps the zarr recording)",
            json_path.display()
        );
    }
    let Some(sample_rate) = kwargs.get("sampling_frequency").and_then(|x| x.as_f64()) else {
        // A preprocessing chain only names its inner recording, with the
        // chain's own kwargs; the rate/dtype/bin live at the binary source.
        if CHANNEL_AND_SAMPLE_PRESERVING.contains(&class_name) {
            let inner = kwargs.get("recording").ok_or_else(|| {
                anyhow!(
                    "{}: {class_name} without a nested recording",
                    json_path.display()
                )
            })?;
            return parse_recording_value(inner, json_path, base, from_binary);
        }
        // `rec.save(folder=...)` / `si.load(folder)` produce a
        // BinaryFolderRecording, whose own kwargs are only `folder_path`;
        // the extractor's description lives in that folder's `binary.json`
        // (SI itself asserts it is a BinaryRecordingExtractor there).
        if class_name != "BinaryFolderRecording" {
            bail!(
                "{}: unsupported recording class {class} (missing sampling_frequency); sorrel \
                 reads only channel- and sample-preserving preprocessing chains over a binary \
                 recording — re-save it with recording.save(format=\"binary\", folder=...)",
                json_path.display()
            );
        }
        if from_binary {
            bail!("{}: nested BinaryFolderRecording", json_path.display());
        }
        let folder = kwargs
            .get("folder_path")
            .and_then(|x| x.as_str())
            .ok_or_else(|| {
                anyhow!(
                    "{}: BinaryFolderRecording without folder_path",
                    json_path.display()
                )
            })?;
        let folder = resolve_path(base, Path::new(folder));
        let binary_json = folder.join("binary.json");
        let text = std::fs::read_to_string(&binary_json)
            .with_context(|| format!("read {}", binary_json.display()))?;
        return parse_recording_json_at(&text, &binary_json, &folder, true);
    };
    let sample_rate = sample_rate as f32;
    let n_channels = kwargs
        .get("num_channels")
        .or_else(|| kwargs.get("num_chan"))
        .and_then(|x| x.as_u64())
        .ok_or_else(|| anyhow!("recording.json: missing num_channels"))?
        as u32;
    let dtype_str = kwargs
        .get("dtype")
        .and_then(|x| x.as_str())
        .unwrap_or("int16");
    let dtype = TraceDtype::from_phy_name(dtype_str)
        .ok_or_else(|| anyhow!("recording.json: unsupported dtype {dtype_str}"))?;
    let offset = kwargs
        .get("file_offset")
        .and_then(|x| x.as_u64())
        .unwrap_or(0);

    let mut file_paths = Vec::new();
    if let Some(arr) = kwargs.get("file_paths").and_then(|x| x.as_array()) {
        for v in arr {
            if let Some(s) = v.as_str() {
                file_paths.push(PathBuf::from(s));
            }
        }
    }
    Ok(RecordingMeta {
        sample_rate,
        n_channels,
        dtype,
        file_paths,
        base: base.to_path_buf(),
        offset,
    })
}

/// Try to map the first recording bin into memory. Falls back to
/// `(None, null, 0, 0)` when the path can't be found — `Session` callers
/// can still browse spike-time-only data even if the trace is missing.
fn map_recording(rec: &RecordingMeta) -> Result<(Option<Mmap>, *const u8, usize, SampleIndex)> {
    let Some(p) = rec.file_paths.first() else {
        return Ok((None, std::ptr::null(), 0, SampleIndex(0)));
    };
    let p = resolve_path(&rec.base, p);
    if !p.exists() {
        return Ok((None, std::ptr::null(), 0, SampleIndex(0)));
    }
    let f = File::open(&p).with_context(|| format!("open {}", p.display()))?;
    let mmap = unsafe { Mmap::map(&f)? };
    let total = mmap.len();
    if (rec.offset as usize) > total {
        bail!("file_offset exceeds bin size for {}", p.display());
    }
    let payload = total - rec.offset as usize;
    let bps = rec.n_channels as usize * rec.dtype.size_bytes();
    if bps == 0 || payload.checked_rem(bps) != Some(0) {
        bail!("recording payload not divisible by frame size");
    }
    let n_samples = (payload / bps) as u64;
    let trace_ptr = unsafe { mmap.as_ptr().add(rec.offset as usize) };
    Ok((Some(mmap), trace_ptr, payload, SampleIndex(n_samples)))
}

fn provider_identity_bytes(root: &Path, rec: &RecordingMeta) -> Vec<u8> {
    let mut paths = vec![
        root.join("recording.json"),
        root.join("binary.json"),
        root.join("sorting").join("spikes.npy"),
        root.join("sorting").join("spike_times.npy"),
        root.join("sorting").join("unit_indices.npy"),
        root.join("sorting").join("unit_ids.npy"),
    ];
    if let Some(p) = rec.file_paths.first() {
        paths.push(resolve_path(&rec.base, p));
    }

    let mut hasher = blake3::Hasher::new();
    hasher.update(b"sorrel-provider-identity-v1/sorting-analyzer");
    hasher.update(&rec.sample_rate.to_le_bytes());
    hasher.update(&rec.n_channels.to_le_bytes());
    hasher.update(&rec.offset.to_le_bytes());
    for path in paths {
        hasher.update(
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("")
                .as_bytes(),
        );
        match std::fs::metadata(&path) {
            Ok(meta) => {
                hasher.update(&meta.len().to_le_bytes());
                if let Ok(modified) = meta.modified() {
                    if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
                        hasher.update(&duration.as_secs().to_le_bytes());
                        hasher.update(&duration.subsec_nanos().to_le_bytes());
                    }
                }
                if path.extension().and_then(|ext| ext.to_str()) == Some("npy") {
                    if let Ok(header) = read_header(&path) {
                        hasher.update(header.dtype.as_bytes());
                        hasher.update(&[header.fortran_order as u8]);
                        hasher.update(&header.data_offset.to_le_bytes());
                        for dim in header.shape {
                            hasher.update(&(dim as u64).to_le_bytes());
                        }
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

/// Loads either:
/// * `sorting/spikes.npy` (structured array with `sample_index`+`unit_index`)
/// * `sorting/spike_times.npy` + `sorting/unit_indices.npy` (flat npys)
fn load_sorting(root: &Path) -> Result<(Vec<SampleIndex>, Vec<u32>)> {
    let st_flat = root.join("sorting").join("spike_times.npy");
    let ui_flat = root.join("sorting").join("unit_indices.npy");
    if st_flat.exists() && ui_flat.exists() {
        let times = read_u64_any(&st_flat)?;
        let units = read_1d_u32(&ui_flat)?;
        if times.len() != units.len() {
            bail!("spike_times/unit_indices length mismatch");
        }
        return Ok((times, units));
    }

    let structured = root.join("sorting").join("spikes.npy");
    if structured.exists() {
        return load_structured_spikes(&structured);
    }

    bail!(
        "SortingAnalyzer: no sorting arrays found under {}/sorting/",
        root.display()
    )
}

/// Parse SI's `sorting/spikes.npy` structured array. The dtype looks like
/// `[('sample_index','<i8'),('unit_index','<i8'),('segment_index','<i8')]` —
/// we accept any field order and any record width as long as the two fields
/// we care about are present and `<i8` / `<u8` / `<i4` / `<u4`.
fn load_structured_spikes(path: &Path) -> Result<(Vec<SampleIndex>, Vec<u32>)> {
    let h = read_header(path)?;
    if h.shape.len() != 1 {
        bail!(
            "{}: expected 1-D structured array, got {:?}",
            path.display(),
            h.shape
        );
    }
    let fields = parse_record_dtype(&h.dtype)
        .with_context(|| format!("parse record dtype {} in {}", h.dtype, path.display()))?;

    let record_size: usize = fields.iter().map(|f| f.width).sum();
    let sample_field = fields
        .iter()
        .find(|f| f.name == "sample_index")
        .ok_or_else(|| {
            anyhow!(
                "spikes.npy missing 'sample_index' field (dtype {})",
                h.dtype
            )
        })?;
    let unit_field = fields
        .iter()
        .find(|f| f.name == "unit_index")
        .ok_or_else(|| anyhow!("spikes.npy missing 'unit_index' field (dtype {})", h.dtype))?;

    let bytes = std::fs::read(path)?;
    let payload = &bytes[h.data_offset as usize..];
    let n = h.elem_count();
    if payload.len() < n * record_size {
        bail!("{}: structured payload truncated", path.display());
    }

    let mut times = Vec::with_capacity(n);
    let mut units = Vec::with_capacity(n);
    for rec in payload[..n * record_size].chunks_exact(record_size) {
        let s = sample_field.offset;
        let u = unit_field.offset;
        let sample = read_int_le(&rec[s..s + sample_field.width], sample_field.width)?;
        let unit = read_int_le(&rec[u..u + unit_field.width], unit_field.width)?;
        times.push(SampleIndex(sample as u64));
        units.push(unit as u32);
    }
    Ok((times, units))
}

#[derive(Debug)]
struct RecordField {
    name: String,
    offset: usize,
    width: usize,
}

/// Parse a numpy record-dtype string like
/// `[('sample_index', '<i8'), ('unit_index', '<i8'), ('segment_index', '<i8')]`.
/// Returns the fields with byte offsets computed in declaration order.
fn parse_record_dtype(s: &str) -> Result<Vec<RecordField>> {
    let s = s.trim();
    let inner = s
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .ok_or_else(|| anyhow!("expected list-of-tuples dtype, got {s:?}"))?;
    let mut fields = Vec::new();
    let mut offset = 0;
    let mut depth = 0i32;
    let mut start = 0usize;
    let bytes = inner.as_bytes();
    for (i, &c) in bytes.iter().enumerate() {
        match c {
            b'(' => {
                if depth == 0 {
                    start = i + 1;
                }
                depth += 1;
            }
            b')' => {
                depth -= 1;
                if depth == 0 {
                    let tup = &inner[start..i];
                    let parts: Vec<&str> = tup.splitn(2, ',').collect();
                    if parts.len() != 2 {
                        bail!("malformed field tuple {tup:?}");
                    }
                    let name = parts[0]
                        .trim()
                        .trim_matches('\'')
                        .trim_matches('"')
                        .to_string();
                    let dtype = parts[1].trim().trim_matches('\'').trim_matches('"');
                    let width = int_dtype_width(dtype)
                        .ok_or_else(|| anyhow!("unsupported field dtype {dtype}"))?;
                    fields.push(RecordField {
                        name,
                        offset,
                        width,
                    });
                    offset += width;
                }
            }
            _ => {}
        }
    }
    if fields.is_empty() {
        bail!("no fields parsed from {s:?}");
    }
    Ok(fields)
}

fn int_dtype_width(d: &str) -> Option<usize> {
    match d {
        "<i8" | "<u8" => Some(8),
        "<i4" | "<u4" => Some(4),
        "<i2" | "<u2" => Some(2),
        _ => None,
    }
}

fn read_int_le(bytes: &[u8], width: usize) -> Result<i64> {
    Ok(match width {
        8 => i64::from_le_bytes(bytes.try_into().unwrap()),
        4 => i32::from_le_bytes(bytes.try_into().unwrap()) as i64,
        2 => i16::from_le_bytes(bytes.try_into().unwrap()) as i64,
        w => bail!("unexpected int width {w}"),
    })
}

fn read_u64_any(path: &Path) -> Result<Vec<SampleIndex>> {
    use crate::npy::read_npy_u32_flat;
    let h = read_header(path)?;
    if !h.is_vector() {
        bail!("{}: expected 1-D, got {:?}", path.display(), h.shape);
    }
    if h.dtype.contains("8") {
        // 64-bit ints (signed or unsigned) — read raw and reinterpret.
        let bytes = std::fs::read(path)?;
        let payload = &bytes[h.data_offset as usize..];
        let n = h.elem_count();
        if payload.len() < n * 8 {
            bail!("truncated 64-bit array");
        }
        Ok(payload[..n * 8]
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| SampleIndex(u64::from_le_bytes(*c)))
            .collect())
    } else {
        let (v, _shape) = read_npy_u32_flat(path)?;
        Ok(v.into_iter().map(|x| SampleIndex(x as u64)).collect())
    }
}

fn load_quality_metrics_csv(
    path: &Path,
    n_clusters: usize,
) -> Result<crate::extras::QualityMetrics> {
    let mut out = crate::extras::QualityMetrics::new();
    if !path.exists() {
        return Ok(out);
    }
    let text = std::fs::read_to_string(path)?;
    let mut lines = text.lines();
    let Some(header) = lines.next() else {
        return Ok(out);
    };
    let cols: Vec<&str> = header.split(',').collect();
    if cols.is_empty() {
        return Ok(out);
    }
    // SI writes the unit-id column (often unnamed) first.
    let metric_names: Vec<String> = cols[1..].iter().map(|s| s.trim().to_string()).collect();
    for name in &metric_names {
        out.insert(name.clone(), vec![f32::NAN; n_clusters]);
    }

    for line in lines {
        let cells: Vec<&str> = line.split(',').collect();
        if cells.is_empty() {
            continue;
        }
        let Ok(unit) = cells[0].trim().parse::<usize>() else {
            continue;
        };
        if unit >= n_clusters {
            continue;
        }
        for (i, name) in metric_names.iter().enumerate() {
            let cell = cells.get(i + 1).copied().unwrap_or("");
            if let Ok(v) = cell.trim().parse::<f32>() {
                if let Some(col) = out.get_mut(name) {
                    col[unit] = v;
                }
            }
        }
    }
    Ok(out)
}

impl DataProvider for SortingAnalyzerProvider {
    type Label = PhyLabel;

    #[inline]
    fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
    #[inline]
    fn n_channels(&self) -> u32 {
        self.n_channels
    }
    #[inline]
    fn n_samples(&self) -> SampleIndex {
        self.n_samples
    }
    #[inline]
    fn n_clusters(&self) -> u32 {
        self.spikes_per_cluster.len() as u32
    }

    #[inline]
    fn spike_times(&self, cluster: ClusterId) -> &[SampleIndex] {
        self.spikes_per_cluster
            .get(cluster.idx())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    #[inline]
    fn trace(&self, start: SampleIndex, len: u32) -> TraceSlice<'_> {
        TraceSlice {
            start,
            n_channels: self.n_channels,
            samples: self.samples_window(start, len),
        }
    }

    /// Like [`crate::provider::appended_unit_id`], but new ids also clear
    /// every SI string id that reads as a number.
    fn unit_id(&self, cluster: ClusterId) -> u32 {
        match self.unit_ids.get(cluster.idx()) {
            Some(&u) => u,
            None => self.next_unit_id + (cluster.0 - self.unit_ids.len() as u32),
        }
    }

    fn unit_id_json(&self, cluster: ClusterId) -> serde_json::Value {
        match (
            &self.unit_names,
            self.unit_names.as_ref().and_then(|n| n.get(cluster.idx())),
        ) {
            (_, Some(name)) => name.as_str().into(),
            (Some(_), None) => self.unit_id(cluster).to_string().into(),
            (None, None) => self.unit_id(cluster).into(),
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

    fn raw_file(&self) -> Option<RawFile> {
        self.raw.clone()
    }
}

impl HasGeometry for SortingAnalyzerProvider {
    #[inline]
    fn channel_positions(&self) -> &[[f32; 2]] {
        &self.channel_positions
    }
    #[inline]
    fn channel_map(&self) -> &[ChannelId] {
        &self.channel_map
    }
}

impl HasQualityMetrics for SortingAnalyzerProvider {
    fn quality_metrics(&self) -> &crate::extras::QualityMetrics {
        &self.metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_recording_json_with_kwargs_block() {
        let json = r#"{
            "kwargs": {
                "sampling_frequency": 30000.0,
                "num_channels": 384,
                "dtype": "int16",
                "file_paths": ["recording.dat"]
            }
        }"#;
        let m = parse_recording_json(json, Path::new("/tmp/recording.json")).unwrap();
        assert_eq!(m.sample_rate as u32, 30000);
        assert_eq!(m.n_channels, 384);
        assert_eq!(m.dtype, TraceDtype::I16);
        assert_eq!(m.file_paths, vec![PathBuf::from("recording.dat")]);
    }

    #[test]
    fn parses_record_dtype_three_field_layout() {
        let f = parse_record_dtype(
            "[('sample_index', '<i8'), ('unit_index', '<i8'), ('segment_index', '<i8')]",
        )
        .unwrap();
        assert_eq!(f.len(), 3);
        assert_eq!(f[0].name, "sample_index");
        assert_eq!(f[0].offset, 0);
        assert_eq!(f[0].width, 8);
        assert_eq!(f[1].name, "unit_index");
        assert_eq!(f[1].offset, 8);
        assert_eq!(f[2].offset, 16);
    }

    #[test]
    fn parses_record_dtype_mixed_widths() {
        let f = parse_record_dtype("[('sample_index', '<i8'), ('unit_index', '<u4')]").unwrap();
        assert_eq!(f[0].width, 8);
        assert_eq!(f[1].offset, 8);
        assert_eq!(f[1].width, 4);
    }

    #[test]
    fn parses_recording_json_top_level_fields() {
        let json = r#"{
            "sampling_frequency": 25000.0,
            "num_channels": 32,
            "dtype": "float32",
            "file_paths": []
        }"#;
        let m = parse_recording_json(json, Path::new("rec.json")).unwrap();
        assert_eq!(m.sample_rate as u32, 25000);
        assert_eq!(m.dtype, TraceDtype::F32);
        assert!(m.file_paths.is_empty());
    }

    #[test]
    fn rejects_record_dtype_missing_required_field() {
        // No 'unit_index' → load_structured_spikes should error.
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("spikes.npy");
        // Single 'sample_index' i8 field, 1 record.
        write_record_npy(&p, "[('sample_index', '<i8')]", 1, &0u64.to_le_bytes());
        let err = load_structured_spikes(&p).unwrap_err().to_string();
        assert!(err.contains("unit_index"), "got: {err}");
    }

    #[test]
    fn structured_spikes_roundtrip_parses_sample_and_unit_indices() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("spikes.npy");
        // Three records: (sample_index<i8>, unit_index<i8>, segment_index<i8>).
        let records: &[(u64, u64, u64)] = &[(100, 0, 0), (250, 2, 0), (999, 1, 0)];
        let mut payload = Vec::new();
        for (s, u, seg) in records {
            payload.extend_from_slice(&s.to_le_bytes());
            payload.extend_from_slice(&u.to_le_bytes());
            payload.extend_from_slice(&seg.to_le_bytes());
        }
        write_record_npy(
            &p,
            "[('sample_index', '<i8'), ('unit_index', '<i8'), ('segment_index', '<i8')]",
            records.len(),
            &payload,
        );

        let (times, units) = load_structured_spikes(&p).unwrap();
        assert_eq!(
            times,
            vec![SampleIndex(100), SampleIndex(250), SampleIndex(999)]
        );
        assert_eq!(units, vec![0, 2, 1]);
    }

    #[test]
    fn structured_spikes_handles_reordered_fields_and_mixed_widths() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("spikes.npy");
        // Reordered: unit_index first (u4), sample_index second (i8).
        let records: &[(u32, u64)] = &[(7, 42), (3, 10_000_000), (0, 1)];
        let mut payload = Vec::new();
        for (u, s) in records {
            payload.extend_from_slice(&u.to_le_bytes());
            payload.extend_from_slice(&s.to_le_bytes());
        }
        write_record_npy(
            &p,
            "[('unit_index', '<u4'), ('sample_index', '<i8')]",
            records.len(),
            &payload,
        );
        let (times, units) = load_structured_spikes(&p).unwrap();
        assert_eq!(
            times,
            vec![SampleIndex(42), SampleIndex(10_000_000), SampleIndex(1)]
        );
        assert_eq!(units, vec![7, 3, 0]);
    }

    #[test]
    fn opens_synthetic_binary_folder_end_to_end() {
        // Build a minimal `binary_folder` layout matching what
        // `SortingAnalyzer.save_as("binary_folder")` emits, then open it.
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        // 1) recording.json — points at a sibling .dat file.
        let recording_json = r#"{
            "kwargs": {
                "sampling_frequency": 30000.0,
                "num_channels": 4,
                "dtype": "int16",
                "file_paths": ["recording.dat"]
            }
        }"#;
        std::fs::write(root.join("recording.json"), recording_json).unwrap();

        // 2) recording.dat — 100 samples × 4 channels × i16.
        let n_samples_target: usize = 100;
        let n_channels: usize = 4;
        let mut bin = Vec::with_capacity(n_samples_target * n_channels * 2);
        for s in 0..n_samples_target {
            for c in 0..n_channels {
                let v = (s as i16) * 10 + c as i16;
                bin.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(root.join("recording.dat"), &bin).unwrap();

        // 3) sorting/spikes.npy + unit_ids.npy.
        let sorting_dir = root.join("sorting");
        std::fs::create_dir_all(&sorting_dir).unwrap();
        let records: &[(u64, u64, u64)] =
            &[(10, 0, 0), (20, 1, 0), (30, 0, 0), (40, 2, 0), (50, 1, 0)];
        let mut payload = Vec::new();
        for (s, u, seg) in records {
            payload.extend_from_slice(&s.to_le_bytes());
            payload.extend_from_slice(&u.to_le_bytes());
            payload.extend_from_slice(&seg.to_le_bytes());
        }
        write_record_npy(
            &sorting_dir.join("spikes.npy"),
            "[('sample_index', '<i8'), ('unit_index', '<i8'), ('segment_index', '<i8')]",
            records.len(),
            &payload,
        );
        // unit_ids: dense 0..3.
        write_u32_npy(&sorting_dir.join("unit_ids.npy"), &[0, 1, 2]);

        let p = SortingAnalyzerProvider::open(root).unwrap();
        assert_eq!(p.sample_rate(), 30000.0);
        assert_eq!(p.n_channels(), 4);
        assert_eq!(p.n_clusters(), 3);
        assert_eq!(p.n_samples(), SampleIndex(n_samples_target as u64));

        // Per-cluster spike times: cluster 0 → [10, 30], 1 → [20, 50], 2 → [40].
        assert_eq!(
            p.spike_times(ClusterId(0)),
            &[SampleIndex(10), SampleIndex(30)]
        );
        assert_eq!(
            p.spike_times(ClusterId(1)),
            &[SampleIndex(20), SampleIndex(50)]
        );
        assert_eq!(p.spike_times(ClusterId(2)), &[SampleIndex(40)]);
        // Out-of-range cluster id returns empty rather than panicking.
        assert!(p.spike_times(ClusterId(99)).is_empty());

        // Trace samples are mmapped — read a 2-sample window starting at
        // sample 5 and verify it matches the synthetic generator.
        let slice = p.trace(SampleIndex(5), 2);
        match slice.samples {
            TraceSamples::I16(buf) => {
                assert_eq!(buf.len(), 2 * n_channels);
                assert_eq!(buf[0], 50); // sample=5, channel=0 → 5*10+0
                assert_eq!(buf[3], 53); // sample=5, channel=3 → 5*10+3
                assert_eq!(buf[4], 60); // sample=6, channel=0
            }
            other => panic!("expected I16 trace, got {other:?}"),
        }
        assert_eq!(p.unit_id_json(ClusterId(1)), serde_json::json!(1));

        // SpikeInterface 0.105 keeps the ids in numpysorting_info.json, as
        // strings for its generators; sparse ones must survive.
        std::fs::remove_file(sorting_dir.join("unit_ids.npy")).unwrap();
        std::fs::write(
            sorting_dir.join("numpysorting_info.json"),
            r#"{"sampling_frequency": 30000.0, "unit_ids": ["3", "7", "12"], "num_segments": 1}"#,
        )
        .unwrap();
        let p = SortingAnalyzerProvider::open(root).unwrap();
        assert_eq!(p.unit_id_json(ClusterId(1)), serde_json::json!("7"));
        assert_eq!(p.unit_id(ClusterId(1)), 7);
        assert_eq!(p.unit_id(ClusterId(3)), 13); // a sorrel split appends
        assert_eq!(p.unit_id_json(ClusterId(3)), serde_json::json!("13"));

        // Ids that are not distinct integers keep SI's strings verbatim and
        // number the units densely; new ids clear every numeric-looking one.
        std::fs::write(
            sorting_dir.join("numpysorting_info.json"),
            r#"{"sampling_frequency": 30000.0, "unit_ids": ["unit_a", "007", "5"], "num_segments": 1}"#,
        )
        .unwrap();
        let p = SortingAnalyzerProvider::open(root).unwrap();
        assert_eq!(p.unit_id_json(ClusterId(0)), serde_json::json!("unit_a"));
        assert_eq!(p.unit_id_json(ClusterId(1)), serde_json::json!("007"));
        assert_eq!(p.unit_id(ClusterId(1)), 1);
        assert_eq!(p.unit_id_json(ClusterId(3)), serde_json::json!("8"));
    }

    #[test]
    fn binary_folder_without_recording_dat_still_opens_with_zero_samples() {
        // SortingAnalyzers without a bound recording binary should still
        // open — n_samples falls back to 0 and the trace window is empty.
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        let recording_json = r#"{
            "kwargs": {
                "sampling_frequency": 20000.0,
                "num_channels": 2,
                "dtype": "int16",
                "file_paths": []
            }
        }"#;
        std::fs::write(root.join("recording.json"), recording_json).unwrap();
        let sorting_dir = root.join("sorting");
        std::fs::create_dir_all(&sorting_dir).unwrap();
        // Empty sorting via flat spike_times/unit_indices.
        write_u32_npy(&sorting_dir.join("spike_times.npy"), &[]);
        write_u32_npy(&sorting_dir.join("unit_indices.npy"), &[]);
        write_u32_npy(&sorting_dir.join("unit_ids.npy"), &[0]);

        let p = SortingAnalyzerProvider::open(root).unwrap();
        assert_eq!(p.n_samples(), SampleIndex(0));
        assert_eq!(p.n_clusters(), 1);
        let slice = p.trace(SampleIndex(0), 4);
        match slice.samples {
            TraceSamples::I16(buf) => assert!(buf.is_empty()),
            other => panic!("expected empty I16 trace, got {other:?}"),
        }
    }

    /// `rec.save(folder=...)` / `si.load(folder)` yield a
    /// BinaryFolderRecording: `recording.json` holds only `folder_path`, and
    /// the extractor kwargs live in that folder's `binary.json`. Match SI's
    /// relative-path handling: resolve `file_paths` against the folder that
    /// holds the JSON.
    #[test]
    fn opens_analyzer_on_binary_folder_recording_relative_folder_path() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let rec = root.join("rec");
        write_binary_folder_rec(&rec, 25000.0, 2, 100);
        std::fs::write(
            root.join("recording.json"),
            serde_json::json!({
                "class": "spikeinterface.core.binaryfolder.BinaryFolderRecording",
                "module": "spikeinterface",
                "relative_paths": true,
                "kwargs": {"folder_path": "rec"},
            })
            .to_string(),
        )
        .unwrap();
        write_tiny_sorting(root);

        let p = SortingAnalyzerProvider::open(root).unwrap();
        assert_eq!(p.sample_rate(), 25000.0);
        assert_eq!(p.n_channels(), 2);
        assert_eq!(p.n_samples(), SampleIndex(100));
        assert_eq!(
            p.raw_file().unwrap().path,
            rec.join("traces_cached_seg0.raw")
        );
        assert_eq!(p.raw_file().unwrap().dtype, TraceDtype::F32);
        assert_eq!(
            p.spike_times(ClusterId(0)),
            &[SampleIndex(10), SampleIndex(20)]
        );
        match p.trace(SampleIndex(5), 1).samples {
            TraceSamples::F32(buf) => assert_eq!(buf, &[50.0, 51.0]),
            other => panic!("expected F32 trace, got {other:?}"),
        }
    }

    /// A recording folder outside the analyzer's tree is dumped with an
    /// absolute `folder_path` and `relative_paths: false`.
    #[test]
    fn opens_analyzer_on_binary_folder_recording_absolute_folder_path() {
        let analyzer = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        let rec = other.path().join("rec");
        write_binary_folder_rec(&rec, 30000.0, 2, 50);
        std::fs::write(
            analyzer.path().join("recording.json"),
            serde_json::json!({
                "class": "spikeinterface.core.binaryfolder.BinaryFolderRecording",
                "module": "spikeinterface",
                "relative_paths": false,
                "kwargs": {"folder_path": rec.to_str().unwrap()},
            })
            .to_string(),
        )
        .unwrap();
        write_tiny_sorting(analyzer.path());

        let p = SortingAnalyzerProvider::open(analyzer.path()).unwrap();
        assert_eq!(p.sample_rate(), 30000.0);
        assert_eq!(p.n_samples(), SampleIndex(50));
        assert_eq!(
            p.raw_file().unwrap().path,
            rec.join("traces_cached_seg0.raw")
        );
    }

    /// A chain of channel- and sample-preserving preprocessors names its
    /// source through nested `kwargs.recording` dicts; the source's rate,
    /// dtype and bin are what sorrel opens.
    #[test]
    fn follows_two_level_preprocessing_chain_to_binary_source() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let rec = root.join("rec");
        write_binary_folder_rec(&rec, 30000.0, 3, 40);
        std::fs::write(
            root.join("recording.json"),
            serde_json::json!({
                "class": "spikeinterface.preprocessing.filter.BandpassFilterRecording",
                "module": "spikeinterface",
                "relative_paths": true,
                "kwargs": {
                    "freq_min": 300.0,
                    "recording": {
                        "class": "spikeinterface.preprocessing.common_reference.CommonReferenceRecording",
                        "relative_paths": false,
                        "kwargs": {
                            "reference": "global",
                            "recording": {
                                "class": "spikeinterface.core.binaryfolder.BinaryFolderRecording",
                                "relative_paths": false,
                                "kwargs": {"folder_path": "rec"}
                            }
                        }
                    }
                }
            })
            .to_string(),
        )
        .unwrap();
        write_tiny_sorting(root);

        let p = SortingAnalyzerProvider::open(root).unwrap();
        assert_eq!(p.sample_rate(), 30000.0);
        assert_eq!(p.n_channels(), 3);
        assert_eq!(p.n_samples(), SampleIndex(40));
        assert_eq!(
            p.raw_file().unwrap().path,
            rec.join("traces_cached_seg0.raw")
        );
    }

    /// Slicing changes which channels or samples exist, so the source binary
    /// is not the analyzer's recording; refuse instead of showing it.
    #[test]
    fn refuses_channel_slice_recording_naming_its_class() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("recording.json"),
            r#"{
                "class": "spikeinterface.core.channelslice.ChannelSliceRecording",
                "module": "spikeinterface",
                "relative_paths": true,
                "kwargs": {"channel_ids": ["0"], "recording": {"class": "x.BinaryRecordingExtractor", "kwargs": {}}}
            }"#,
        )
        .unwrap();
        let err = load_recording_json(dir.path()).unwrap_err().to_string();
        assert!(err.contains("ChannelSliceRecording"), "got: {err}");
        assert!(err.contains("channel- and sample-preserving"), "got: {err}");
    }

    #[test]
    fn refuses_zarr_recording_with_conversion_instructions() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("recording.json"),
            r#"{
                "class": "spikeinterface.core.zarrextractors.ZarrRecordingExtractor",
                "module": "spikeinterface",
                "relative_paths": true,
                "kwargs": {"folder_path": "rec.zarr", "storage_options": null}
            }"#,
        )
        .unwrap();
        let err = load_recording_json(dir.path()).unwrap_err().to_string();
        assert!(err.contains("ZarrRecordingExtractor"), "got: {err}");
        assert!(
            err.contains("recording.save(format=\"binary\""),
            "got: {err}"
        );
    }

    fn write_binary_folder_rec(folder: &Path, sample_rate: f64, n_channels: u32, n_samples: usize) {
        std::fs::create_dir_all(folder).unwrap();
        let mut bin = Vec::with_capacity(n_samples * n_channels as usize * 4);
        for s in 0..n_samples {
            for c in 0..n_channels as usize {
                bin.extend_from_slice(&((s * 10 + c) as f32).to_le_bytes());
            }
        }
        std::fs::write(folder.join("traces_cached_seg0.raw"), bin).unwrap();
        std::fs::write(
            folder.join("binary.json"),
            serde_json::json!({
                "class": "spikeinterface.core.binaryrecordingextractor.BinaryRecordingExtractor",
                "module": "spikeinterface",
                "relative_paths": true,
                "kwargs": {
                    "file_paths": ["traces_cached_seg0.raw"],
                    "sampling_frequency": sample_rate,
                    "num_channels": n_channels,
                    "dtype": "<f4",
                    "file_offset": 0,
                },
            })
            .to_string(),
        )
        .unwrap();
    }

    fn write_tiny_sorting(root: &Path) {
        let sorting_dir = root.join("sorting");
        std::fs::create_dir_all(&sorting_dir).unwrap();
        let mut payload = Vec::new();
        for (s, u) in [(10u64, 0u64), (20, 0)] {
            payload.extend_from_slice(&s.to_le_bytes());
            payload.extend_from_slice(&u.to_le_bytes());
            payload.extend_from_slice(&0u64.to_le_bytes());
        }
        write_record_npy(
            &sorting_dir.join("spikes.npy"),
            "[('sample_index', '<i8'), ('unit_index', '<i8'), ('segment_index', '<i8')]",
            2,
            &payload,
        );
        write_u32_npy(&sorting_dir.join("unit_ids.npy"), &[0]);
    }

    /// Test helper: write an `.npy` v1 file with the given dtype string
    /// (record-dtype OK), shape `(n,)`, and raw payload bytes.
    pub(super) fn write_record_npy(path: &Path, descr: &str, n: usize, payload: &[u8]) {
        use std::io::Write;
        let dict = format!("{{'descr': {descr}, 'fortran_order': False, 'shape': ({n},), }}");
        let prelude_len = 6 + 2 + 2 + dict.len() + 1;
        let pad = (64 - (prelude_len % 64)) % 64;
        let mut header = dict.into_bytes();
        header.resize(header.len() + pad, b' ');
        header.push(b'\n');
        let header_len = header.len() as u16;
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(b"\x93NUMPY").unwrap();
        f.write_all(&[1u8, 0u8]).unwrap();
        f.write_all(&header_len.to_le_bytes()).unwrap();
        f.write_all(&header).unwrap();
        f.write_all(payload).unwrap();
    }

    pub(super) fn write_u32_npy(path: &Path, data: &[u32]) {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        write_record_npy(path, "'<u4'", data.len(), &bytes);
    }
}

#[cfg(test)]
mod curation_tests {
    use super::*;

    /// `curation.json` seeds labels through SI unit ids, and clusters that
    /// sorrel creates by splitting get ids appended after the largest one.
    #[test]
    fn curation_json_seeds_labels_by_unit_id_and_split_ids_append() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(
            root.join("recording.json"),
            r#"{"kwargs":{"sampling_frequency":30000.0,"num_channels":1,"dtype":"int16","file_paths":["recording.dat"]}}"#,
        )
        .unwrap();
        std::fs::write(root.join("recording.dat"), [0u8; 20]).unwrap();
        let sorting_dir = root.join("sorting");
        std::fs::create_dir_all(&sorting_dir).unwrap();
        let mut payload = Vec::new();
        for (s, u) in [(10u64, 0u64), (20, 1), (30, 2)] {
            payload.extend_from_slice(&s.to_le_bytes());
            payload.extend_from_slice(&u.to_le_bytes());
            payload.extend_from_slice(&0u64.to_le_bytes());
        }
        tests::write_record_npy(
            &sorting_dir.join("spikes.npy"),
            "[('sample_index', '<i8'), ('unit_index', '<i8'), ('segment_index', '<i8')]",
            3,
            &payload,
        );
        // Sparse SI ids: index 0 → 5, 1 → 7, 2 → 9.
        tests::write_u32_npy(&sorting_dir.join("unit_ids.npy"), &[5, 7, 9]);
        std::fs::write(
            root.join("curation.json"),
            r#"{"format_version":"2","unit_ids":[5,7,9],
                "manual_labels":[{"unit_id":7,"labels":{"quality":["noise"]}}],
                "merges":[],"splits":[],"removed":[]}"#,
        )
        .unwrap();

        let p = SortingAnalyzerProvider::open(root).unwrap();
        assert_eq!(
            p.initial_labels(),
            vec![PhyLabel::Unsorted, PhyLabel::Noise, PhyLabel::Unsorted]
        );
        assert_eq!(p.unit_id(ClusterId(1)), 7);
        assert_eq!(p.unit_id(ClusterId(3)), 10, "first split-created cluster");
        assert_eq!(p.unit_id(ClusterId(4)), 11);
    }
}
