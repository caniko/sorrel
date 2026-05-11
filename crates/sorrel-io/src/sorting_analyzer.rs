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
//!   recording.json          (or binary.json) — sample rate + dtype + bin path
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
    ChannelId, ClusterId, DataProvider, SampleIndex, TraceDtype, TraceSamples, TraceSlice,
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

    channel_positions: Vec<[f32; 2]>,
    channel_map: Vec<ChannelId>,

    metrics: crate::extras::QualityMetrics,

    _trace_mmap: Option<Mmap>,
    trace_ptr: *const u8,
    trace_byte_len: usize,
}

unsafe impl Send for SortingAnalyzerProvider {}
unsafe impl Sync for SortingAnalyzerProvider {}

impl SortingAnalyzerProvider {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        let recording_meta = load_recording_json(root)?;

        // Sorting: spikes.npy is structured numpy, but binary_folder also
        // emits separate spike_times and unit_index arrays under
        // sorting/numpy_folder when `save_as('numpy_folder', ...)` was used.
        // We accept either layout.
        let (times, unit_indices) = load_sorting(root)?;

        let unit_ids_path = root.join("sorting").join("unit_ids.npy");
        let unit_ids = if unit_ids_path.exists() {
            read_1d_u32(&unit_ids_path)?
        } else {
            // Fall back to dense unit numbering [0..max+1).
            let max = unit_indices.iter().copied().max().unwrap_or(0);
            (0..=max).collect()
        };
        let n_clusters = unit_ids.len();

        let mut spikes_per_cluster: Vec<Vec<SampleIndex>> = vec![Vec::new(); n_clusters];
        for (i, &uidx) in unit_indices.iter().enumerate() {
            if let Some(b) = spikes_per_cluster.get_mut(uidx as usize) {
                b.push(times[i]);
            }
        }
        for b in spikes_per_cluster.iter_mut() {
            b.sort_unstable();
        }
        let initial_labels = vec![PhyLabel::Unsorted; n_clusters];

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

        // Map raw recording. SI stores either a relative path or an absolute
        // path inside recording.json's `kwargs.file_paths`.
        let (trace_mmap, trace_ptr, trace_byte_len, n_samples) =
            map_recording(root, &recording_meta)?;

        Ok(Self {
            sample_rate: recording_meta.sample_rate,
            n_channels: recording_meta.n_channels,
            n_samples,
            dtype: recording_meta.dtype,
            spikes_per_cluster,
            initial_labels,
            channel_positions,
            channel_map,
            metrics,
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
    file_paths: Vec<PathBuf>,
    /// Bytes to skip at the start of the bin file (rare; SI usually 0).
    offset: u64,
}

fn load_recording_json(root: &Path) -> Result<RecordingMeta> {
    for name in ["recording.json", "binary.json", "recording/recording.json"] {
        let p = root.join(name);
        if p.exists() {
            let text =
                std::fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
            return parse_recording_json(&text, &p);
        }
    }
    bail!(
        "SortingAnalyzer: no recording.json/binary.json found in {}",
        root.display()
    );
}

fn parse_recording_json(text: &str, json_path: &Path) -> Result<RecordingMeta> {
    let v: serde_json::Value =
        serde_json::from_str(text).with_context(|| format!("parse {}", json_path.display()))?;
    let kwargs = v.get("kwargs").unwrap_or(&v);
    let sample_rate = kwargs
        .get("sampling_frequency")
        .and_then(|x| x.as_f64())
        .ok_or_else(|| anyhow!("recording.json: missing sampling_frequency"))?
        as f32;
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
        offset,
    })
}

/// Try to map the first recording bin into memory. Falls back to
/// `(None, null, 0, 0)` when the path can't be found — `Session` callers
/// can still browse spike-time-only data even if the trace is missing.
fn map_recording(
    root: &Path,
    rec: &RecordingMeta,
) -> Result<(Option<Mmap>, *const u8, usize, SampleIndex)> {
    let Some(p) = rec.file_paths.first() else {
        return Ok((None, std::ptr::null(), 0, SampleIndex(0)));
    };
    let p = if p.is_absolute() {
        p.clone()
    } else {
        root.join(p)
    };
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
    if h.shape.len() != 1 {
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
        let mut out = Vec::with_capacity(n);
        for c in payload[..n * 8].chunks_exact(8) {
            out.push(SampleIndex(u64::from_le_bytes(c.try_into().unwrap())));
        }
        Ok(out)
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

    fn initial_labels(&self) -> Vec<Self::Label> {
        self.initial_labels.clone()
    }

    fn amplitude_full_scale(&self) -> f32 {
        self.dtype.nominal_full_scale()
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

    /// Test helper: write an `.npy` v1 file with the given dtype string
    /// (record-dtype OK), shape `(n,)`, and raw payload bytes.
    fn write_record_npy(path: &Path, descr: &str, n: usize, payload: &[u8]) {
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

    fn write_u32_npy(path: &Path, data: &[u32]) {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        write_record_npy(path, "'<u4'", data.len(), &bytes);
    }
}
