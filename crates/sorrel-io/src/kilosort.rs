//! Concrete Kilosort/phy2 backend. Memory-maps `spike_times.npy`,
//! `spike_clusters.npy`, the raw recording, and `cluster_group.tsv`.
//!
//! Recording layout, dtype, sample rate, and channel count come from
//! `params.py` (phy's canonical source of truth). CLI overrides may force
//! values when `params.py` is missing or wrong.

use crate::extras::{
    HasAmplitudes, HasGeometry, HasPcFeatures, HasQualityMetrics, HasSpikeTemplates,
};
use crate::npy::{
    read_1d_f32, read_1d_u32, read_2d_f32, read_header, read_npy_f32_flat, read_npy_u32_flat,
    NpyHeader,
};
use crate::params::PhyParams;
use crate::provider::{
    ChannelId, ClusterId, DataProvider, RawFile, SampleIndex, TraceDtype, TraceSamples, TraceSlice,
};
use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

/// V1 phy2 schema as a `#[repr(u8)]` enum so the label vector packs to
/// 1 byte per cluster.
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum PhyLabel {
    #[default]
    Unsorted = 0,
    Good = 1,
    Mua = 2,
    Noise = 3,
}

impl PhyLabel {
    pub fn parse_tsv_value(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "good" => Self::Good,
            "mua" => Self::Mua,
            "noise" => Self::Noise,
            _ => Self::Unsorted,
        }
    }
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unsorted => "unsorted",
            Self::Good => "good",
            Self::Mua => "mua",
            Self::Noise => "noise",
        }
    }
}

pub struct KilosortProvider {
    root: PathBuf,
    sample_rate: f32,
    n_channels: u32,
    n_samples: SampleIndex,
    dtype: TraceDtype,

    // mmaps kept alive for the lifetime of the provider; the slices below
    // borrow into them via raw pointers cast back inside accessor methods.
    _spike_times_mmap: Mmap,
    _spike_clusters_mmap: Mmap,
    _trace_mmap: Mmap,

    // Per-cluster spike-time vectors (we re-bucket once at load time so the
    // hot `spike_times(cluster)` call is a single slice borrow). Amplitudes
    // and templates are bucketed in the same order so accessors are constant
    // time and align by index.
    spikes_per_cluster: Vec<Vec<SampleIndex>>,
    amps_per_cluster: Vec<Vec<f32>>,
    templates_per_cluster: Vec<Vec<u32>>,
    initial_labels: Vec<PhyLabel>,
    initial_tags: Vec<Vec<String>>,

    // Probe geometry. Empty when the corresponding `.npy` file is absent.
    channel_positions: Vec<[f32; 2]>,
    channel_shanks: Vec<u32>,
    channel_map: Vec<ChannelId>,

    // PC features. Flat `(n_spikes, n_pcs, n_channels_per_template)` row-major.
    // Empty when the file is absent.
    pc_features: std::sync::Arc<[f32]>,
    pc_shape: (usize, usize), // (n_pcs, n_channels_per_template)
    pc_feature_ind: Vec<u32>, // (n_templates * n_channels_per_template) flat
    /// Per-cluster, time-sorted, original NPY row indices into pc_features.
    /// Doubles as the row index into `amplitudes.npy`/`spike_times.npy`/etc.
    pc_indices_per_cluster: Vec<Vec<u32>>,

    // Template waveforms `(n_templates, n_samples, n_channels)` flat. Empty
    // when `templates.npy` is absent. `similar_templates.npy` is the square
    // template-similarity matrix (`n_templates × n_templates`).
    template_waveforms: std::sync::Arc<[f32]>,
    template_shape: (usize, usize, usize),
    similar_templates: Vec<f32>,

    // Optional sidecar quality metrics. Loaded from
    // `cluster_*.tsv` (phy convention) or `quality_metrics.csv`
    // (SpikeInterface). One column per metric, length == n_clusters,
    // NaN where the upstream tool reports nothing.
    metrics: crate::extras::QualityMetrics,
    identity_bytes: Vec<u8>,

    // Raw byte view of the trace mmap *after* `offset` has been applied.
    // We cast at access time based on `dtype`.
    trace_ptr: *const u8,
    trace_byte_len: usize,
    /// Set when `dat_path` is an mtscomp `.cbin`; windows are decoded on
    /// demand and `trace_ptr`/`trace_byte_len` stay empty.
    cbin: Option<crate::cbin::CbinReader>,
    dat_path: PathBuf,
    dat_offset: u64,
    hp_filtered: bool,
    trace_fetch: Option<std::sync::Arc<dyn crate::provider::TraceFetch>>,
}

// SAFETY: the raw pointer aliases data inside `_trace_mmap` which lives as long
// as the struct. The mmap is read-only and Rust's borrow checker enforces
// non-aliasing on the produced slices.
unsafe impl Send for KilosortProvider {}
unsafe impl Sync for KilosortProvider {}

/// Optional overrides that take precedence over `params.py`. All fields are
/// optional so callers can supply only what they want to override.
#[derive(Clone, Default)]
pub struct KilosortOpenParams {
    pub sample_rate: Option<f32>,
    pub n_channels: Option<u32>,
    pub dtype: Option<TraceDtype>,
    pub offset: Option<u64>,
    /// Path to the raw binary; resolved relative to the kilosort directory if
    /// it isn't absolute. Falls back to `params.py:dat_path`, then
    /// `<root>/recording.dat`.
    pub dat_path: Option<PathBuf>,
    /// Remote-mirror hook: consulted before every trace read. `None` for
    /// ordinary local datasets.
    pub trace_fetch: Option<std::sync::Arc<dyn crate::provider::TraceFetch>>,
    /// Replaces the file-metadata identity (a sparse mirror's mtime changes on
    /// every fetch, which would churn derived caches).
    pub identity_override: Option<Vec<u8>>,
}

impl std::fmt::Debug for KilosortOpenParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KilosortOpenParams")
            .field("sample_rate", &self.sample_rate)
            .field("n_channels", &self.n_channels)
            .field("dtype", &self.dtype)
            .field("offset", &self.offset)
            .field("dat_path", &self.dat_path)
            .field("trace_fetch", &self.trace_fetch.is_some())
            .field("identity_override", &self.identity_override.is_some())
            .finish()
    }
}

impl KilosortProvider {
    pub fn open(root: impl AsRef<Path>, overrides: KilosortOpenParams) -> Result<Self> {
        let root = root.as_ref().to_path_buf();

        // 1. params.py — optional but strongly preferred. If present, it owns
        // sample_rate / n_channels / dtype / offset / dat_path defaults.
        let params_path = root.join("params.py");
        let params: PhyParams = if params_path.exists() {
            PhyParams::read(&params_path)?
        } else {
            PhyParams::default()
        };

        let sample_rate = overrides
            .sample_rate
            .or(params.sample_rate)
            .context("sample_rate not in params.py and no override given")?;
        let n_channels = overrides
            .n_channels
            .or(params.n_channels_dat)
            .context("n_channels_dat not in params.py and no override given")?;
        let dtype = overrides.dtype.or(params.dtype).unwrap_or(TraceDtype::I16);
        let offset = overrides.offset.or(params.offset).unwrap_or(0);

        // 2. Spike index files.
        let spike_times_path = root.join("spike_times.npy");
        let spike_clusters_path = root.join("spike_clusters.npy");
        let cluster_group_path = root.join("cluster_group.tsv");

        let st_hdr = read_header(&spike_times_path)?;
        let sc_hdr = read_header(&spike_clusters_path)?;

        if st_hdr.elem_count() != sc_hdr.elem_count() {
            bail!("spike_times and spike_clusters length mismatch");
        }
        let n_spikes = st_hdr.elem_count();

        let st_mmap = unsafe { Mmap::map(&File::open(&spike_times_path)?)? };
        let sc_mmap = unsafe { Mmap::map(&File::open(&spike_clusters_path)?)? };

        let times = read_u64_array(&st_mmap, &st_hdr, n_spikes)?;
        let clusters = read_u32_array(&sc_mmap, &sc_hdr, n_spikes)?;

        let max_cluster = clusters.iter().copied().max().unwrap_or(0);
        let n_clusters = (max_cluster as usize) + 1;

        // Optional per-spike side data. When present, length must match the
        // number of spikes; we keep `None` when the file is missing so the
        // trait impls can return empty slices instead of fabricating zeros.
        let amps_full = read_optional_f32_1d(&root.join("amplitudes.npy"), n_spikes)?;
        let templates_full = read_optional_u32_1d(&root.join("spike_templates.npy"), n_spikes)?;

        // Single bucketing pass that keeps (time, amp?, tpl?) aligned: collect
        // (time, idx) tuples per cluster, sort by time, then gather aligned
        // values from the per-spike arrays. Two passes over `clusters` total.
        let mut indices_per_cluster: Vec<Vec<u32>> = vec![Vec::new(); n_clusters];
        for (i, &c) in clusters.iter().enumerate() {
            indices_per_cluster[c as usize].push(i as u32);
        }

        let mut spikes_per_cluster: Vec<Vec<SampleIndex>> = Vec::with_capacity(n_clusters);
        let mut amps_per_cluster: Vec<Vec<f32>> = Vec::with_capacity(n_clusters);
        let mut templates_per_cluster: Vec<Vec<u32>> = Vec::with_capacity(n_clusters);
        for idx_bucket in indices_per_cluster.iter_mut() {
            // Sort the indices into the global spike arrays by time, so
            // downstream slices share the same per-spike order.
            idx_bucket.sort_unstable_by_key(|&i| times[i as usize]);
            let bucket_times: Vec<SampleIndex> =
                idx_bucket.iter().map(|&i| times[i as usize]).collect();
            let bucket_amps: Vec<f32> = match amps_full.as_ref() {
                Some(a) => idx_bucket.iter().map(|&i| a[i as usize]).collect(),
                None => Vec::new(),
            };
            let bucket_templates: Vec<u32> = match templates_full.as_ref() {
                Some(t) => idx_bucket.iter().map(|&i| t[i as usize]).collect(),
                None => Vec::new(),
            };
            spikes_per_cluster.push(bucket_times);
            amps_per_cluster.push(bucket_amps);
            templates_per_cluster.push(bucket_templates);
        }
        // After sorting, `indices_per_cluster[c]` holds the per-cluster
        // time-sorted *original NPY row* indices — exactly what
        // `HasPcFeatures::spike_pc_indices` returns. Move it onto the
        // provider rather than dropping it on the floor.
        let pc_indices_per_cluster = indices_per_cluster;

        // phy semantics: curated `cluster_group.tsv` wins, then a
        // SpikeInterface `curation.json`, then the sorter's own
        // `cluster_KSLabel.tsv` from a fresh Kilosort output.
        let initial_labels = if cluster_group_path.exists() {
            load_cluster_groups(&cluster_group_path, n_clusters)
        } else {
            crate::curation_json::read_curation_labels(
                &root.join("curation.json"),
                n_clusters,
                |uid| usize::try_from(uid).ok(),
            )
            .unwrap_or_else(|| load_cluster_groups(&root.join("cluster_KSLabel.tsv"), n_clusters))
        };

        let initial_tags = read_tags_tsv(&root.join(TAGS_TSV));

        // 2.5. Probe geometry. All optional; absence is fine.
        let channel_positions = read_optional_positions(&root.join("channel_positions.npy"))?;
        let channel_shanks =
            read_optional_u32_1d(&root.join("channel_shanks.npy"), 0)?.unwrap_or_default();
        let channel_map: Vec<ChannelId> = read_optional_u32_1d(&root.join("channel_map.npy"), 0)?
            .unwrap_or_default()
            .into_iter()
            .map(ChannelId)
            .collect();

        // 2.6. PC features. Both files are optional but we need both or
        // neither — having pc_features without pc_feature_ind makes the
        // shape unusable.
        let (pc_features, pc_shape, pc_feature_ind) = read_optional_pc_features(&root, n_spikes)?;

        // 2.7. Template waveforms + similarity matrix. Both optional; the
        // TemplateView and SimilarityView gracefully degrade when absent.
        let (template_waveforms, template_shape) =
            read_optional_templates(&root.join("templates.npy"))?;
        let similar_templates =
            read_optional_similar_templates(&root.join("similar_templates.npy"), template_shape.0)?;

        // 2.8. Quality metrics. Two sidecar conventions:
        //   - phy:  one `cluster_<metric>.tsv` per metric.
        //   - SI:   a single `quality_metrics.csv` with one row per cluster.
        // We accept both; SI's CSV wins when both are present.
        let metrics = load_quality_metrics(&root, n_clusters);

        // 3. Raw recording.
        let dat_rel = overrides
            .dat_path
            .or(params.dat_path)
            .unwrap_or_else(|| PathBuf::from("recording.dat"));
        let dat_path = if dat_rel.is_absolute() {
            dat_rel
        } else {
            root.join(&dat_rel)
        };
        let is_cbin = dat_path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("cbin"));
        let cbin = if is_cbin {
            let r = crate::cbin::CbinReader::open(&dat_path)?;
            if r.n_channels() != n_channels as usize {
                bail!(
                    "{}: n_channels {} does not match params ({n_channels})",
                    dat_path.display(),
                    r.n_channels()
                );
            }
            if dtype != TraceDtype::I16 {
                bail!("{}: mtscomp recordings are int16", dat_path.display());
            }
            Some(r)
        } else {
            None
        };
        // A compressed recording has no raw bytes to map: keep an empty map
        // so every downstream byte-range computation clamps to zero.
        let trace_file = if is_cbin {
            File::open(dat_path.with_extension("ch"))
        } else {
            File::open(&dat_path)
        }
        .with_context(|| format!("open {}", dat_path.display()))?;
        let trace_mmap = unsafe { Mmap::map(&trace_file)? };
        let total_bytes = if is_cbin { 0 } else { trace_mmap.len() };
        if (offset as usize) > total_bytes {
            bail!(
                "params.offset ({}) exceeds dat size ({})",
                offset,
                total_bytes
            );
        }
        let payload_bytes = total_bytes - offset as usize;
        let bytes_per_sample = (n_channels as usize) * dtype.size_bytes();
        if bytes_per_sample == 0 || payload_bytes.checked_rem(bytes_per_sample) != Some(0) {
            bail!(
                "dat payload size ({}) not divisible by n_channels({}) * dtype_bytes({})",
                payload_bytes,
                n_channels,
                dtype.size_bytes()
            );
        }
        let n_samples = match &cbin {
            Some(r) => SampleIndex(r.n_samples()),
            None => SampleIndex((payload_bytes / bytes_per_sample) as u64),
        };
        // SAFETY: pointer arithmetic stays inside the mmap; we just verified
        // `offset <= total_bytes`.
        let trace_ptr = unsafe { trace_mmap.as_ptr().add(offset as usize) };
        let trace_byte_len = payload_bytes;
        let identity_bytes = provider_identity_bytes(&[
            &spike_times_path,
            &spike_clusters_path,
            &root.join("amplitudes.npy"),
            &dat_path,
        ]);

        Ok(Self {
            root,
            sample_rate,
            n_channels,
            n_samples,
            dtype,
            _spike_times_mmap: st_mmap,
            _spike_clusters_mmap: sc_mmap,
            _trace_mmap: trace_mmap,
            spikes_per_cluster,
            amps_per_cluster,
            templates_per_cluster,
            initial_labels,
            initial_tags,
            channel_positions,
            channel_shanks,
            channel_map,
            pc_features: pc_features.into(),
            pc_shape,
            pc_feature_ind,
            pc_indices_per_cluster,
            template_waveforms: template_waveforms.into(),
            template_shape,
            similar_templates,
            metrics,
            identity_bytes: overrides.identity_override.unwrap_or(identity_bytes),
            trace_ptr,
            trace_byte_len,
            cbin,
            dat_path,
            dat_offset: offset,
            hp_filtered: params.hp_filtered.unwrap_or(false),
            trace_fetch: overrides.trace_fetch,
        })
    }

    /// Byte range of the trace payload (offset already removed) that
    /// `[start, start + len)` samples occupy, clamped to the file.
    fn window_bytes(&self, start: SampleIndex, len: u32) -> std::ops::Range<u64> {
        let frame = (self.n_channels as u64) * self.dtype.size_bytes() as u64;
        let s = start
            .0
            .saturating_mul(frame)
            .min(self.trace_byte_len as u64);
        let e = (start.0 + len as u64)
            .saturating_mul(frame)
            .min(self.trace_byte_len as u64);
        s..e
    }

    fn samples_window(&self, start: SampleIndex, len: u32) -> TraceSamples<'_> {
        if let Some(fetch) = &self.trace_fetch {
            if !fetch.ensure(self.window_bytes(start, len)) {
                // Not resident yet: same contract as a missing recording.
                return TraceSamples::empty(self.dtype);
            }
        }
        if let Some(r) = &self.cbin {
            return TraceSamples::OwnedI16(r.window(start.0, len as usize));
        }
        let nc = self.n_channels as usize;
        let s = start.idx().saturating_mul(nc);
        let e = s + (len as usize) * nc;
        let bytes = unsafe { std::slice::from_raw_parts(self.trace_ptr, self.trace_byte_len) };
        TraceSamples::from_bytes_clamped(bytes, self.dtype, s, e)
    }

    /// Resolved raw-recording path and the header bytes skipped before the
    /// first sample.
    pub fn dat_path(&self) -> &Path {
        &self.dat_path
    }

    pub fn dat_offset(&self) -> u64 {
        self.dat_offset
    }

    pub fn dtype(&self) -> TraceDtype {
        self.dtype
    }
}

impl DataProvider for KilosortProvider {
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

    fn initial_tags(&self) -> Vec<Vec<String>> {
        self.initial_tags.clone()
    }

    fn identity_bytes(&self) -> Vec<u8> {
        self.identity_bytes.clone()
    }

    fn prefetch(&self, windows: &[(SampleIndex, u32)]) {
        if let Some(fetch) = &self.trace_fetch {
            for &(start, len) in windows {
                fetch.ensure(self.window_bytes(start, len));
            }
        }
    }

    fn pending_fetches(&self) -> usize {
        self.trace_fetch.as_ref().map_or(0, |f| f.pending())
    }

    fn amplitude_full_scale(&self) -> f32 {
        self.dtype.nominal_full_scale()
    }

    fn raw_file(&self) -> Option<RawFile> {
        Some(RawFile {
            path: self.dat_path.clone(),
            dtype: self.dtype,
            offset: self.dat_offset,
            hp_filtered: self.hp_filtered,
        })
    }

    fn phy_dir(&self) -> Option<&Path> {
        Some(&self.root)
    }
}

impl HasGeometry for KilosortProvider {
    #[inline]
    fn channel_positions(&self) -> &[[f32; 2]] {
        &self.channel_positions
    }
    #[inline]
    fn channel_shanks(&self) -> &[u32] {
        &self.channel_shanks
    }
    #[inline]
    fn channel_map(&self) -> &[ChannelId] {
        &self.channel_map
    }
}

impl HasAmplitudes for KilosortProvider {
    #[inline]
    fn spike_amplitudes(&self, cluster: ClusterId) -> &[f32] {
        self.amps_per_cluster
            .get(cluster.idx())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}

impl HasSpikeTemplates for KilosortProvider {
    #[inline]
    fn spike_templates(&self, cluster: ClusterId) -> &[u32] {
        self.templates_per_cluster
            .get(cluster.idx())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}

impl HasPcFeatures for KilosortProvider {
    #[inline]
    fn pc_features(&self) -> &[f32] {
        &self.pc_features
    }
    fn pc_features_shared(&self) -> std::sync::Arc<[f32]> {
        self.pc_features.clone()
    }
    #[inline]
    fn pc_shape(&self) -> (usize, usize) {
        self.pc_shape
    }
    #[inline]
    fn pc_feature_ind(&self) -> &[u32] {
        &self.pc_feature_ind
    }
    #[inline]
    fn spike_pc_indices(&self, cluster: ClusterId) -> &[u32] {
        self.pc_indices_per_cluster
            .get(cluster.idx())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}

impl crate::extras::HasTemplateWaveforms for KilosortProvider {
    #[inline]
    fn template_waveforms(&self) -> &[f32] {
        &self.template_waveforms
    }
    fn template_waveforms_shared(&self) -> std::sync::Arc<[f32]> {
        self.template_waveforms.clone()
    }
    #[inline]
    fn template_shape(&self) -> (usize, usize, usize) {
        self.template_shape
    }
    #[inline]
    fn similar_templates(&self) -> &[f32] {
        &self.similar_templates
    }
}

impl HasQualityMetrics for KilosortProvider {
    #[inline]
    fn quality_metrics(&self) -> &crate::extras::QualityMetrics {
        &self.metrics
    }
}

/// Discover sidecar quality metrics. Tries SpikeInterface's
/// `quality_metrics.csv` first, then falls back to phy's per-metric
/// `cluster_<metric>.tsv` files. Always returns the metric list sorted by
/// name so the cluster-table column order is deterministic.
fn load_quality_metrics(root: &Path, n_clusters: usize) -> crate::extras::QualityMetrics {
    let mut metrics = crate::extras::QualityMetrics::new();

    // SI: single CSV with header row, first column is unit id.
    let si_path = root.join("quality_metrics.csv");
    if let Ok(text) = std::fs::read_to_string(&si_path) {
        ingest_si_metrics_csv(&text, n_clusters, &mut metrics);
    }

    // phy: one TSV per metric. Skip phy's well-known label files since
    // those aren't metrics (they're labels we already loaded).
    if let Ok(rd) = std::fs::read_dir(root) {
        for entry in rd.flatten() {
            let name = entry.file_name();
            let name_s = name.to_string_lossy();
            let Some(stripped) = name_s
                .strip_prefix("cluster_")
                .and_then(|s| s.strip_suffix(".tsv"))
            else {
                continue;
            };
            // Label/annotation sidecars are not numeric metrics.
            if matches!(
                stripped,
                "group" | "groups" | "KSLabel" | "purity" | "info" | "sorrel_tags"
            ) {
                continue;
            }
            if metrics.contains(stripped) {
                continue; // SI CSV took precedence.
            }
            if let Ok(text) = std::fs::read_to_string(entry.path()) {
                if let Some(col) = parse_phy_metric_tsv(&text, stripped, n_clusters) {
                    metrics.insert(stripped.to_string(), col);
                }
            }
        }
    }

    metrics.sort_by_name();
    metrics
}

fn ingest_si_metrics_csv(text: &str, n_clusters: usize, out: &mut crate::extras::QualityMetrics) {
    let mut lines = text.lines();
    let Some(header) = lines.next() else {
        return;
    };
    // SpikeInterface writes commas, but exporters and hand edits also
    // produce tab- or semicolon-separated files; sniff from the header.
    let sep = ['\t', ';', ',']
        .into_iter()
        .find(|&c| header.contains(c))
        .unwrap_or(',');
    let cols: Vec<&str> = header.split(sep).collect();
    if cols.len() < 2 {
        return;
    }
    let metric_cols: Vec<String> = cols[1..].iter().map(|s| s.trim().to_string()).collect();
    for name in &metric_cols {
        if !out.contains(name) {
            out.insert(name.clone(), vec![f32::NAN; n_clusters]);
        }
    }
    for line in lines {
        let cells: Vec<&str> = line.split(sep).collect();
        if cells.is_empty() {
            continue;
        }
        let Ok(unit) = cells[0].trim().parse::<usize>() else {
            continue;
        };
        if unit >= n_clusters {
            continue;
        }
        for (i, name) in metric_cols.iter().enumerate() {
            let cell = cells.get(i + 1).copied().unwrap_or("");
            if let Ok(v) = cell.trim().parse::<f32>() {
                if let Some(col) = out.get_mut(name) {
                    col[unit] = v;
                }
            }
        }
    }
}

fn parse_phy_metric_tsv(text: &str, name: &str, n_clusters: usize) -> Option<Vec<f32>> {
    let mut lines = text.lines();
    let header = lines.next()?;
    // Header is "cluster_id\t<name>"; we accept any tab-separated 2-col TSV.
    let cols: Vec<&str> = header.split('\t').collect();
    if cols.len() < 2 {
        return None;
    }
    let mut out = vec![f32::NAN; n_clusters];
    let mut any = false;
    for line in lines {
        let mut it = line.split('\t');
        let (Some(id), Some(val)) = (it.next(), it.next()) else {
            continue;
        };
        let Ok(id) = id.trim().parse::<usize>() else {
            continue;
        };
        let Ok(v) = val.trim().parse::<f32>() else {
            continue;
        };
        if id < n_clusters {
            out[id] = v;
            any = true;
        }
    }
    if any {
        log::debug!("loaded phy metric '{name}' for {} clusters", n_clusters);
        Some(out)
    } else {
        None
    }
}

/// Skip-on-missing wrapper: returns `default` when `path` doesn't exist,
/// otherwise runs `loader(path)`. Centralises the optional-file pattern used
/// across the per-array readers below.
fn load_optional<T>(path: &Path, default: T, loader: impl FnOnce(&Path) -> Result<T>) -> Result<T> {
    if !path.exists() {
        return Ok(default);
    }
    loader(path)
}

fn check_len(path: &Path, got: usize, expected: usize) -> Result<()> {
    if got != expected {
        bail!(
            "{}: expected {} elements, got {}",
            path.display(),
            expected,
            got
        );
    }
    Ok(())
}

/// Read a 1-D `f32` array if the file exists, validating its length matches
/// `expected_len`. Returns `Ok(None)` when the file is absent.
fn read_optional_f32_1d(path: &Path, expected_len: usize) -> Result<Option<Vec<f32>>> {
    load_optional(path, None, |p| {
        let v = read_1d_f32(p)?;
        check_len(p, v.len(), expected_len)?;
        Ok(Some(v))
    })
}

/// Read a 1-D `u32` array if the file exists. When `expected_len` is non-zero
/// the loaded length must match; pass 0 for arrays whose length is determined
/// by the file itself (channel_map, channel_shanks).
fn read_optional_u32_1d(path: &Path, expected_len: usize) -> Result<Option<Vec<u32>>> {
    load_optional(path, None, |p| {
        let v = read_1d_u32(p)?;
        if expected_len != 0 {
            check_len(p, v.len(), expected_len)?;
        }
        Ok(Some(v))
    })
}

/// Read `pc_features.npy` and `pc_feature_ind.npy` together when both are
/// present. Returns `(flat, (n_pcs, n_chans_per_template), feature_ind_flat)`
/// with empty defaults when either file is absent.
type PcFeatures = (Vec<f32>, (usize, usize), Vec<u32>);

fn read_optional_pc_features(root: &Path, n_spikes: usize) -> Result<PcFeatures> {
    let feat_path = root.join("pc_features.npy");
    let ind_path = root.join("pc_feature_ind.npy");
    if !feat_path.exists() || !ind_path.exists() {
        return Ok((Vec::new(), (0, 0), Vec::new()));
    }
    let (flat, shape) = read_npy_f32_flat(&feat_path)?;
    if shape.len() != 3 {
        bail!(
            "pc_features.npy: expected 3-D (n_spikes, n_pcs, n_channels) shape, got {:?}",
            shape
        );
    }
    if shape[0] != n_spikes {
        bail!(
            "pc_features.npy: shape[0]={} doesn't match n_spikes={}",
            shape[0],
            n_spikes
        );
    }
    let n_pcs = shape[1];
    let n_chans_per_template = shape[2];

    let (ind_flat, ind_shape) = read_npy_u32_flat(&ind_path)?;
    let ind_chans = match ind_shape.as_slice() {
        [_, c] => *c,
        [c] => *c,
        other => bail!("pc_feature_ind.npy: unexpected shape {:?}", other),
    };
    if ind_chans != n_chans_per_template {
        bail!(
            "pc_feature_ind.npy: channels-per-template ({}) doesn't match pc_features.npy ({})",
            ind_chans,
            n_chans_per_template
        );
    }

    Ok((flat, (n_pcs, n_chans_per_template), ind_flat))
}

/// Read a `(N, 2)` channel_positions array if the file exists.
fn read_optional_positions(path: &Path) -> Result<Vec<[f32; 2]>> {
    load_optional(path, Vec::new(), |p| {
        let (flat, (rows, cols)) = read_2d_f32(p)?;
        if cols != 2 {
            bail!("{}: expected 2 columns (x, y), got {}", p.display(), cols);
        }
        let mut out = Vec::with_capacity(rows);
        for r in 0..rows {
            out.push([flat[r * 2], flat[r * 2 + 1]]);
        }
        Ok(out)
    })
}

/// Read `templates.npy` if present. Returns `(flat, (n_templates, n_samples,
/// n_channels))`. Empty buffer + `(0, 0, 0)` shape when absent.
fn read_optional_templates(path: &Path) -> Result<(Vec<f32>, (usize, usize, usize))> {
    load_optional(path, (Vec::new(), (0, 0, 0)), |path| {
        let (flat, shape) = read_npy_f32_flat(path)?;
        if shape.len() != 3 {
            bail!(
                "{}: expected 3-D (n_templates, n_samples, n_channels), got shape {:?}",
                path.display(),
                shape
            );
        }
        Ok((flat, (shape[0], shape[1], shape[2])))
    })
}

/// Read `similar_templates.npy` if present. Validates the matrix is square
/// and matches the template count we already loaded.
fn read_optional_similar_templates(path: &Path, n_templates: usize) -> Result<Vec<f32>> {
    load_optional(path, Vec::new(), |path| {
        let (flat, (rows, cols)) = read_2d_f32(path)?;
        if rows != cols {
            bail!(
                "{}: expected square matrix, got {rows}×{cols}",
                path.display()
            );
        }
        if n_templates != 0 && rows != n_templates {
            bail!(
                "{}: similar_templates is {rows}×{rows} but templates.npy has {n_templates} templates",
                path.display(),
            );
        }
        Ok(flat)
    })
}

fn provider_identity_bytes(paths: &[&Path]) -> Vec<u8> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"sorrel-provider-identity-v1/kilosort");
    for path in paths {
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
                if path.extension().and_then(|ext| ext.to_str()) == Some("npy") {
                    if let Ok(header) = read_header(path) {
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

/// Read `n` little-endian integers from a `.npy` mmap. Accepts both the numpy
/// byte-width dtype shorthand (`<i4`, `<i8`) and the bit-width form some older
/// sorrel test fixtures used (`<i32`, `<i64`). The `from32`/`from64` callbacks
/// handle the (un)widening into the caller's target type.
fn read_int_array<T>(
    mmap: &Mmap,
    hdr: &NpyHeader,
    n: usize,
    label: &str,
    from32: impl Fn(u32) -> T,
    from64: impl Fn(u64) -> T,
) -> Result<Vec<T>> {
    let off = hdr.data_offset as usize;
    let bytes = &mmap[off..];
    let dtype =
        crate::npy::NpyDtype::parse(&hdr.dtype).with_context(|| format!("{label} dtype"))?;
    let elem = dtype.size_bytes();
    let need = n * elem;
    if bytes.len() < need {
        bail!("{label} truncated");
    }
    let chunks = bytes[..need].chunks_exact(elem);
    use crate::npy::NpyDtype as D;
    match dtype {
        D::I32 | D::U32 => Ok(chunks
            .map(|c| from32(u32::from_le_bytes(c.try_into().unwrap())))
            .collect()),
        D::I64 | D::U64 => Ok(chunks
            .map(|c| from64(u64::from_le_bytes(c.try_into().unwrap())))
            .collect()),
        d => bail!("unsupported {label} dtype {d:?}"),
    }
}

fn read_u64_array(mmap: &Mmap, hdr: &NpyHeader, n: usize) -> Result<Vec<SampleIndex>> {
    // `<i8`/`<u8` spike_times are the common case: one memcpy instead of
    // a per-element decode (SampleIndex is a transparent u64).
    if let Ok(d) = crate::npy::NpyDtype::parse(&hdr.dtype) {
        let bytes = &mmap[hdr.data_offset as usize..];
        let same = matches!(d, crate::npy::NpyDtype::I64 | crate::npy::NpyDtype::U64);
        if let Some(v) = crate::npy::cast_fast::<SampleIndex>(bytes, n, same) {
            return Ok(v);
        }
    }
    read_int_array(
        mmap,
        hdr,
        n,
        "spike_times",
        |v| SampleIndex(v as u64),
        SampleIndex,
    )
}

fn read_u32_array(mmap: &Mmap, hdr: &NpyHeader, n: usize) -> Result<Vec<u32>> {
    read_int_array(mmap, hdr, n, "spike_clusters", |v| v, |v| v as u32)
}

fn load_cluster_groups(path: &Path, n_clusters: usize) -> Vec<PhyLabel> {
    let mut out = vec![PhyLabel::Unsorted; n_clusters];
    for (id, label) in read_cluster_group_tsv(path).unwrap_or_default() {
        if (id as usize) < out.len() {
            out[id as usize] = label;
        }
    }
    out
}

/// Parse a phy `cluster_group.tsv` (or any `cluster_id\t<label>` file such
/// as `cluster_KSLabel.tsv`) into `(cluster_id, label)` rows, header
/// skipped, unparsable rows dropped. `None` when the file is unreadable.
pub fn read_cluster_group_tsv(path: &Path) -> Option<Vec<(u32, PhyLabel)>> {
    let text = std::fs::read_to_string(path).ok()?;
    Some(
        text.lines()
            .skip(1)
            .filter_map(|line| {
                let mut cols = line.split('\t');
                let (id, group) = (cols.next()?, cols.next()?);
                Some((id.trim().parse().ok()?, PhyLabel::parse_tsv_value(group)))
            })
            .collect(),
    )
}

/// Sorrel's tag sidecar in a phy directory: `cluster_id\tsorrel_tags` with
/// comma-joined tags, which phy shows as an ordinary column.
pub const TAGS_TSV: &str = "cluster_sorrel_tags.tsv";

/// A tag must survive the comma-joined TSV column and curation.json.
pub fn valid_tag(tag: &str) -> bool {
    !tag.is_empty() && !tag.contains([',', '\t', '\n', '\r'])
}

/// Parse [`TAGS_TSV`] into a per-cluster tag vector indexed by cluster id
/// (empty when the file is absent or unreadable).
pub fn read_tags_tsv(path: &Path) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = Vec::new();
    let Ok(text) = std::fs::read_to_string(path) else {
        return out;
    };
    for line in text.lines().skip(1) {
        let mut cols = line.split('\t');
        let (Some(id), Some(tags)) = (cols.next(), cols.next()) else {
            continue;
        };
        let Ok(id) = id.trim().parse::<usize>() else {
            continue;
        };
        if out.len() <= id {
            out.resize_with(id + 1, Vec::new);
        }
        out[id] = tags
            .split(',')
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(str::to_owned)
            .collect();
    }
    out
}

/// Convenience: count spikes per cluster (mostly for tests / status bars).
pub fn spike_counts<P: DataProvider>(p: &P) -> HashMap<ClusterId, usize> {
    (0..p.n_clusters())
        .map(ClusterId)
        .map(|c| (c, p.spike_times(c).len()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_tsv_value_recognises_known_groups() {
        assert_eq!(PhyLabel::parse_tsv_value("good"), PhyLabel::Good);
        assert_eq!(PhyLabel::parse_tsv_value("MUA"), PhyLabel::Mua);
        assert_eq!(PhyLabel::parse_tsv_value(" Noise "), PhyLabel::Noise);
        assert_eq!(PhyLabel::parse_tsv_value(""), PhyLabel::Unsorted);
        assert_eq!(PhyLabel::parse_tsv_value("???"), PhyLabel::Unsorted);
    }

    #[test]
    fn as_str_round_trips() {
        for lbl in [
            PhyLabel::Unsorted,
            PhyLabel::Good,
            PhyLabel::Mua,
            PhyLabel::Noise,
        ] {
            assert_eq!(PhyLabel::parse_tsv_value(lbl.as_str()), lbl);
        }
    }

    #[test]
    fn label_packs_to_one_byte() {
        assert_eq!(std::mem::size_of::<PhyLabel>(), 1);
    }

    #[test]
    fn load_cluster_groups_from_tsv() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("cluster_group.tsv");
        std::fs::write(
            &p,
            "cluster_id\tgroup\n0\tgood\n1\tmua\n2\tnoise\n3\tgarbage\n",
        )
        .unwrap();
        let labels = load_cluster_groups(&p, 5);
        assert_eq!(labels.len(), 5);
        assert_eq!(labels[0], PhyLabel::Good);
        assert_eq!(labels[1], PhyLabel::Mua);
        assert_eq!(labels[2], PhyLabel::Noise);
        assert_eq!(labels[3], PhyLabel::Unsorted);
        assert_eq!(labels[4], PhyLabel::Unsorted);
    }

    #[test]
    fn load_cluster_groups_missing_file_returns_unsorted() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("absent.tsv");
        let labels = load_cluster_groups(&p, 3);
        assert_eq!(labels, vec![PhyLabel::Unsorted; 3]);
    }
}
