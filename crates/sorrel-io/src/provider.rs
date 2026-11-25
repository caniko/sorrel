use std::fmt::Debug;
use std::path::{Path, PathBuf};

/// One saved row of the annotations sidecar, as text: the origin label is
/// parsed by the session (`sorrel_data::UnitOrigin::parse`), which owns the
/// type. Empty `origin` means `Auto`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UnitAnnotation {
    pub origin: String,
    pub parents: Vec<ClusterId>,
    /// External parent unit IDs without a corresponding cluster in this dataset.
    pub missing_parent_unit_ids: Vec<i64>,
    pub note: String,
}

#[repr(transparent)]
#[derive(
    Copy,
    Clone,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    bytemuck::Pod,
    bytemuck::Zeroable,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
pub struct SampleIndex(pub u64);

impl SampleIndex {
    #[inline]
    pub const fn new(v: u64) -> Self {
        Self(v)
    }
    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }
    #[inline]
    pub const fn idx(self) -> usize {
        self.0 as usize
    }
    #[inline]
    pub const fn as_i64(self) -> i64 {
        self.0 as i64
    }
    #[inline]
    pub fn as_f32(self) -> f32 {
        self.0 as f32
    }
    #[inline]
    pub fn as_f64(self) -> f64 {
        self.0 as f64
    }
}
impl From<u64> for SampleIndex {
    fn from(v: u64) -> Self {
        Self(v)
    }
}
impl From<SampleIndex> for u64 {
    fn from(s: SampleIndex) -> Self {
        s.0
    }
}
impl std::fmt::Display for SampleIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

/// Provenance of the exposure used for session-wide spike statistics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpanSource {
    Trace,
    Declared(&'static str),
    SpikeExtent { rejected: Option<&'static str> },
    Unavailable,
}

impl std::fmt::Display for SpanSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Trace => f.write_str("trace"),
            Self::Declared(path) => write!(f, "declared: {path}"),
            Self::SpikeExtent { rejected: None } => f.write_str("estimate: spike extent"),
            Self::SpikeExtent {
                rejected: Some(path),
            } => {
                write!(
                    f,
                    "estimate: spike extent ({path} does not cover all spikes)"
                )
            }
            Self::Unavailable => f.write_str("unavailable"),
        }
    }
}

/// Half-open session exposure `[start, end)` on the spike sample grid.
/// It does not imply that any raw samples can be read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionSpan {
    pub start: SampleIndex,
    pub end: SampleIndex,
    pub source: SpanSource,
}

impl SessionSpan {
    pub fn duration_samples(self) -> u64 {
        self.end.0.saturating_sub(self.start.0)
    }

    /// Spikes in the exposure. Trace-backed statistics retain their legacy
    /// handling of sorter-supplied spikes outside the raw recording.
    pub fn metric_spikes(self, times: &[SampleIndex]) -> &[SampleIndex] {
        if self.source == SpanSource::Trace {
            return times;
        }
        let start = times.partition_point(|&t| t < self.start);
        let len = times[start..].partition_point(|&t| t < self.end);
        &times[start..start + len]
    }
}

impl From<u64> for SessionSpan {
    fn from(trace_samples: u64) -> Self {
        Self {
            start: SampleIndex(0),
            end: SampleIndex(trace_samples),
            source: SpanSource::Trace,
        }
    }
}

#[repr(transparent)]
#[derive(
    Copy,
    Clone,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    bytemuck::Pod,
    bytemuck::Zeroable,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
pub struct ChannelId(pub u32);

impl ChannelId {
    #[inline]
    pub const fn new(v: u32) -> Self {
        Self(v)
    }
    #[inline]
    pub const fn get(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn idx(self) -> usize {
        self.0 as usize
    }
    #[inline]
    pub const fn as_i64(self) -> i64 {
        self.0 as i64
    }
    #[inline]
    pub fn as_f32(self) -> f32 {
        self.0 as f32
    }
    #[inline]
    pub fn as_f64(self) -> f64 {
        self.0 as f64
    }
}
impl From<u32> for ChannelId {
    fn from(v: u32) -> Self {
        Self(v)
    }
}
impl From<ChannelId> for u32 {
    fn from(c: ChannelId) -> Self {
        c.0
    }
}
impl std::fmt::Display for ChannelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

#[repr(transparent)]
#[derive(
    Copy,
    Clone,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    bytemuck::Pod,
    bytemuck::Zeroable,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
pub struct ClusterId(pub u32);

impl ClusterId {
    #[inline]
    pub const fn new(v: u32) -> Self {
        Self(v)
    }
    #[inline]
    pub const fn get(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn idx(self) -> usize {
        self.0 as usize
    }
    #[inline]
    pub const fn as_i64(self) -> i64 {
        self.0 as i64
    }
    #[inline]
    pub fn as_f32(self) -> f32 {
        self.0 as f32
    }
    #[inline]
    pub fn as_f64(self) -> f64 {
        self.0 as f64
    }
}
impl From<u32> for ClusterId {
    fn from(v: u32) -> Self {
        Self(v)
    }
}
impl From<ClusterId> for u32 {
    fn from(c: ClusterId) -> Self {
        c.0
    }
}
impl std::fmt::Display for ClusterId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

/// Optional row index packed into a `u32`. `u32::MAX` encodes "absent" so a
/// flat `Vec<OptionalRow>` is the same size as a `Vec<u32>`; `Option<u32>`
/// would double the footprint, which matters for >10M-spike recordings.
///
/// Use [`OptionalRow::get`] to convert to a real `Option<u32>` at the use
/// site so callers can't accidentally treat the sentinel as a valid row.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OptionalRow(u32);

impl OptionalRow {
    /// The "absent" sentinel.
    pub const NONE: Self = Self(u32::MAX);

    /// Wrap a raw row index. `u32::MAX` becomes `NONE`.
    pub const fn new(row: u32) -> Self {
        Self(row)
    }

    /// Construct from `Option<u32>`. `Some(u32::MAX)` collapses to `NONE`.
    pub const fn from_option(row: Option<u32>) -> Self {
        match row {
            Some(r) => Self(r),
            None => Self::NONE,
        }
    }

    /// Resolve to an `Option<u32>` for use at the call site.
    #[inline]
    pub const fn get(self) -> Option<u32> {
        if self.0 == u32::MAX {
            None
        } else {
            Some(self.0)
        }
    }

    pub const fn is_some(self) -> bool {
        self.0 != u32::MAX
    }
}

impl Default for OptionalRow {
    fn default() -> Self {
        Self::NONE
    }
}

/// On-disk dtype for the raw trace file. phy supports int16/uint16/int32/float32;
/// Kilosort's default is int16 but other recording rigs (e.g. SpikeGLX float
/// exports, OpenEphys uint16) require the rest.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TraceDtype {
    I16,
    U16,
    I32,
    F32,
}

impl TraceDtype {
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::F32 => 4,
        }
    }

    /// Map a phy/numpy dtype string to our enum. Accepts both numpy-style
    /// (`'<i2'`, `'int16'`) and the bare names that phy writes.
    pub fn from_phy_name(s: &str) -> Option<Self> {
        match s.trim() {
            "int16" | "i2" | "<i2" | ">i2" => Some(Self::I16),
            "uint16" | "u2" | "<u2" | ">u2" => Some(Self::U16),
            "int32" | "i4" | "<i4" | ">i4" => Some(Self::I32),
            "float32" | "single" | "f4" | "<f4" | ">f4" => Some(Self::F32),
            _ => None,
        }
    }

    /// The numpy name phy writes in `params.py` (inverse of [`Self::from_phy_name`]).
    pub const fn phy_name(self) -> &'static str {
        match self {
            Self::I16 => "int16",
            Self::U16 => "uint16",
            Self::I32 => "int32",
            Self::F32 => "float32",
        }
    }

    /// Approximate full-scale magnitude used by render code for amplitude
    /// normalisation. Float buffers are assumed already in physical units of
    /// magnitude ~1 — callers should override this when they know better.
    pub const fn nominal_full_scale(self) -> f32 {
        match self {
            Self::I16 => i16::MAX as f32,
            Self::U16 => i16::MAX as f32, // rendered as zero-centred int16
            Self::I32 => i32::MAX as f32,
            Self::F32 => 1.0,
        }
    }
}

/// Typed trace samples for dtype-specialised inner loops. Windowed sources
/// such as mtscomp and NWB return owned samples instead of borrowing a mmap.
/// Consumers match on [`TraceSamples::borrow`] and never see the difference.
#[derive(Debug)]
pub enum TraceSamples<'a> {
    I16(&'a [i16]),
    U16(&'a [u16]),
    I32(&'a [i32]),
    F32(&'a [f32]),
    OwnedI16(Vec<i16>),
    OwnedF32(Vec<f32>),
}

impl<'a> TraceSamples<'a> {
    /// A borrowing view; owned variants become their borrowed equivalents.
    #[inline]
    pub fn borrow(&self) -> TraceSamples<'_> {
        match self {
            Self::I16(s) => TraceSamples::I16(s),
            Self::U16(s) => TraceSamples::U16(s),
            Self::I32(s) => TraceSamples::I32(s),
            Self::F32(s) => TraceSamples::F32(s),
            Self::OwnedI16(v) => TraceSamples::I16(v),
            Self::OwnedF32(v) => TraceSamples::F32(v),
        }
    }
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::I16(s) => s.len(),
            Self::U16(s) => s.len(),
            Self::I32(s) => s.len(),
            Self::F32(s) => s.len(),
            Self::OwnedI16(v) => v.len(),
            Self::OwnedF32(v) => v.len(),
        }
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn dtype(&self) -> TraceDtype {
        match self {
            Self::I16(_) | Self::OwnedI16(_) => TraceDtype::I16,
            Self::U16(_) => TraceDtype::U16,
            Self::I32(_) => TraceDtype::I32,
            Self::F32(_) | Self::OwnedF32(_) => TraceDtype::F32,
        }
    }

    /// The samples as native-endian bytes, as they sit in a raw `.dat`.
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::I16(s) => bytemuck::cast_slice(s),
            Self::U16(s) => bytemuck::cast_slice(s),
            Self::I32(s) => bytemuck::cast_slice(s),
            Self::F32(s) => bytemuck::cast_slice(s),
            Self::OwnedI16(v) => bytemuck::cast_slice(v),
            Self::OwnedF32(v) => bytemuck::cast_slice(v),
        }
    }

    /// Zero samples of `dtype`. Use this instead of casting an empty byte
    /// slice: `bytemuck` rejects a 1-byte-aligned pointer for wider types.
    pub fn empty(dtype: TraceDtype) -> Self {
        match dtype {
            TraceDtype::I16 => Self::I16(&[]),
            TraceDtype::U16 => Self::U16(&[]),
            TraceDtype::I32 => Self::I32(&[]),
            TraceDtype::F32 => Self::F32(&[]),
        }
    }

    /// Build a typed view from a raw byte buffer of the given dtype, clamping
    /// `[start_elem, end_elem)` to the buffer bounds. `bytes` must be aligned
    /// for `dtype` — mmap pages and `Vec<u8>` allocations satisfy this.
    pub fn from_bytes_clamped(
        bytes: &'a [u8],
        dtype: TraceDtype,
        start_elem: usize,
        end_elem: usize,
    ) -> Self {
        fn clamp<T>(buf: &[T], s: usize, e: usize) -> &[T] {
            let e = e.min(buf.len());
            &buf[s.min(e)..e]
        }
        match dtype {
            TraceDtype::I16 => Self::I16(clamp(bytemuck::cast_slice(bytes), start_elem, end_elem)),
            TraceDtype::U16 => Self::U16(clamp(bytemuck::cast_slice(bytes), start_elem, end_elem)),
            TraceDtype::I32 => Self::I32(clamp(bytemuck::cast_slice(bytes), start_elem, end_elem)),
            TraceDtype::F32 => Self::F32(clamp(bytemuck::cast_slice(bytes), start_elem, end_elem)),
        }
    }

    /// Convert samples to `f32`, centring `U16` around zero by subtracting
    /// `i16::MAX`. This matches the convention used by render and UI which
    /// treat U16 as zero-centred int16.
    pub fn to_f32_centred(&self) -> Vec<f32> {
        match self.borrow() {
            TraceSamples::I16(s) => s.iter().map(|&v| v as f32).collect(),
            TraceSamples::U16(s) => s.iter().map(|&v| v as f32 - i16::MAX as f32).collect(),
            TraceSamples::I32(s) => s.iter().map(|&v| v as f32).collect(),
            TraceSamples::F32(s) => s.to_vec(),
            TraceSamples::OwnedI16(v) => v.iter().map(|&v| v as f32).collect(),
            TraceSamples::OwnedF32(v) => v,
        }
    }
}

/// A contiguous window of trace samples, borrowed or owned.
///
/// `samples` is interleaved `[t0_ch0, t0_ch1, ..., t0_chN-1, t1_ch0, ...]`
/// in the backend's native dtype.
pub struct TraceSlice<'a> {
    pub start: SampleIndex,
    pub n_channels: u32,
    pub samples: TraceSamples<'a>,
}

/// A raw recording on disk, described the way phy's `params.py` does.
#[derive(Clone, Debug, PartialEq)]
pub struct RawFile {
    pub path: PathBuf,
    pub dtype: TraceDtype,
    /// Header bytes before the first sample.
    pub offset: u64,
    pub hp_filtered: bool,
}

/// Static contract every backend implements. The `Label` associated type lets
/// the compiler bake the schema (size, transitions) into the monomorphised
/// session — no dynamic schema branching at runtime.
pub trait DataProvider: Send + Sync + 'static {
    /// Concrete label type; `Copy + 'static` so it packs into a flat
    /// `Vec<Self::Label>` indexed by [`ClusterId`].
    type Label: Copy + Debug + Default + Send + Sync + 'static;

    fn sample_rate(&self) -> f32;
    /// The recording's sample rate before rounding to the internal f32 time base.
    fn exact_sample_rate(&self) -> anyhow::Result<f64> {
        Ok(f64::from(self.sample_rate()))
    }
    fn n_channels(&self) -> u32;
    /// Length of the readable raw trace, not the session exposure.
    fn n_samples(&self) -> SampleIndex;
    /// Exposure for session-wide spike statistics, independent of raw reads.
    fn session_span(&self) -> SessionSpan {
        self.n_samples().0.into()
    }
    /// Trace mode and, for an explicitly skipped NWB trace, its inspection result.
    fn trace_provenance(&self) -> serde_json::Value {
        let mode = if self.session_span().source == SpanSource::Trace {
            "trace"
        } else {
            "units-only"
        };
        serde_json::json!({ "mode": mode })
    }
    fn n_clusters(&self) -> u32;

    /// Spike sample indices for a single cluster, sorted ascending.
    fn spike_times(&self, cluster: ClusterId) -> &[SampleIndex];

    /// Read a window of the raw trace; memory-mapped backends lend a slice.
    fn trace(&self, start: SampleIndex, len: u32) -> TraceSlice<'_>;

    /// Fallible trace read for callers that must report a failed window.
    fn read_trace(&self, start: SampleIndex, len: u32) -> anyhow::Result<TraceSlice<'_>> {
        Ok(self.trace(start, len))
    }

    /// Initial label vector loaded from the backend's on-disk schema.
    fn initial_labels(&self) -> Vec<Self::Label>;

    /// Saved provenance/review tags per cluster (phy: `cluster_sorrel_tags.tsv`).
    /// Missing slots mean no tags.
    fn initial_tags(&self) -> Vec<Vec<String>> {
        Vec::new()
    }

    /// Saved unit origin/lineage/notes per cluster (phy:
    /// `cluster_sorrel_annotations.tsv`), indexed by cluster id. Missing slots
    /// mean `Auto`, no parents, no note.
    fn initial_annotations(&self) -> Vec<UnitAnnotation> {
        Vec::new()
    }

    /// The sorter that produced this sorting, when the backend pinned it (an
    /// `Auto[<sorter>]` row in the annotations sidecar). `None` leaves the
    /// session to fall back to detection or `"unknown"`.
    fn sorter(&self) -> Option<String> {
        None
    }

    /// The backend's own identifier for a cluster, as written to interchange
    /// files such as SpikeInterface's `curation.json`. Phy directories number
    /// clusters densely so the default is the cluster index itself; backends
    /// with sparse unit ids override this.
    fn unit_id(&self, cluster: ClusterId) -> u32 {
        cluster.0
    }

    /// The unit id exactly as interchange files must write it: `7`, or the
    /// backend's own string (`"7"`, `"unit_a"`) when it stores strings,
    /// because SpikeInterface compares ids by value and type.
    fn unit_id_json(&self, cluster: ClusterId) -> serde_json::Value {
        self.unit_id(cluster).into()
    }

    /// Stable bytes identifying this provider's input data, not its path.
    ///
    /// Concrete file-backed providers should prefer content headers for
    /// spike arrays and file metadata for huge raw traces. The default is
    /// intentionally path-free and covers in-memory/test providers.
    fn identity_bytes(&self) -> Vec<u8> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"sorrel-provider-identity-v1/default");
        hasher.update(&self.sample_rate().to_le_bytes());
        hasher.update(&self.n_channels().to_le_bytes());
        hasher.update(&self.n_samples().0.to_le_bytes());
        hasher.update(&self.n_clusters().to_le_bytes());
        for cluster in 0..self.n_clusters() {
            let spikes = self.spike_times(ClusterId(cluster));
            hasher.update(&(spikes.len() as u64).to_le_bytes());
            for spike in spikes {
                hasher.update(&spike.0.to_le_bytes());
            }
        }
        hasher.finalize().as_bytes().to_vec()
    }

    /// Hint for amplitude scaling; defaults to the nominal full-scale of the
    /// dtype but backends with calibration data should override.
    fn amplitude_full_scale(&self) -> f32 {
        TraceDtype::I16.nominal_full_scale()
    }

    /// Ask the backend to make these trace windows resident before a burst of
    /// [`DataProvider::trace`] calls (e.g. one snippet per spike). Local
    /// backends already have everything; remote-backed providers batch the
    /// network round trip here.
    fn prefetch(&self, _windows: &[(SampleIndex, u32)]) {}

    /// Number of trace fetches still in flight. Non-zero means a recent
    /// [`DataProvider::trace`] returned an incomplete window and the caller
    /// should repaint soon.
    fn pending_fetches(&self) -> usize {
        0
    }

    /// The file behind [`DataProvider::trace`], for exporters that point at it
    /// instead of copying it. `None` when the samples live only in memory.
    fn raw_file(&self) -> Option<RawFile> {
        None
    }

    /// The phy/Kilosort directory this provider reads, if it reads one.
    fn phy_dir(&self) -> Option<&Path> {
        None
    }
}

/// [`DataProvider::unit_id`] for a backend with its own unit ids: clusters
/// sorrel creates by splitting (indices past `ids`) get ids appended after
/// the largest one, matching SpikeInterface's
/// `apply_curation(new_id_strategy="append")`.
pub fn appended_unit_id(ids: &[u32], cluster: ClusterId) -> u32 {
    match ids.get(cluster.idx()) {
        Some(&u) => u,
        None => {
            let max = ids.iter().copied().max().unwrap_or(0);
            max + 1 + (cluster.0 - ids.len() as u32)
        }
    }
}

/// Fills a trace mirror on demand. Implemented by remote backends; a provider
/// holding one consults it before reading a window and returns an empty
/// window while the bytes are still in flight.
pub trait TraceFetch: Send + Sync + 'static {
    /// Make `bytes` (payload-relative, i.e. after any header offset) resident.
    /// Returns `true` when the whole range is already readable; `false`
    /// schedules the fetch and returns immediately.
    fn ensure(&self, bytes: std::ops::Range<u64>) -> bool;

    /// Fetches scheduled but not yet written.
    fn pending(&self) -> usize;
}
