use std::fmt::Debug;

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

/// Typed view into the raw mmap. Each variant carries a slice in the native
/// dtype so consumers can run dtype-specialised inner loops without copying.
#[derive(Debug)]
pub enum TraceSamples<'a> {
    I16(&'a [i16]),
    U16(&'a [u16]),
    I32(&'a [i32]),
    F32(&'a [f32]),
}

impl<'a> TraceSamples<'a> {
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::I16(s) => s.len(),
            Self::U16(s) => s.len(),
            Self::I32(s) => s.len(),
            Self::F32(s) => s.len(),
        }
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn dtype(&self) -> TraceDtype {
        match self {
            Self::I16(_) => TraceDtype::I16,
            Self::U16(_) => TraceDtype::U16,
            Self::I32(_) => TraceDtype::I32,
            Self::F32(_) => TraceDtype::F32,
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
        match *self {
            Self::I16(s) => s.iter().map(|&v| v as f32).collect(),
            Self::U16(s) => s.iter().map(|&v| v as f32 - i16::MAX as f32).collect(),
            Self::I32(s) => s.iter().map(|&v| v as f32).collect(),
            Self::F32(s) => s.to_vec(),
        }
    }
}

/// A contiguous, zero-copy slice into a memory-mapped trace buffer.
///
/// `samples` is interleaved `[t0_ch0, t0_ch1, ..., t0_chN-1, t1_ch0, ...]`
/// in the backend's native dtype.
pub struct TraceSlice<'a> {
    pub start: SampleIndex,
    pub n_channels: u32,
    pub samples: TraceSamples<'a>,
}

/// Static contract every backend implements. The `Label` associated type lets
/// the compiler bake the schema (size, transitions) into the monomorphised
/// session — no dynamic schema branching at runtime.
pub trait DataProvider: Send + Sync + 'static {
    /// Concrete label type; `Copy + 'static` so it packs into a flat
    /// `Vec<Self::Label>` indexed by [`ClusterId`].
    type Label: Copy + Debug + Default + Send + Sync + 'static;

    fn sample_rate(&self) -> f32;
    fn n_channels(&self) -> u32;
    fn n_samples(&self) -> SampleIndex;
    fn n_clusters(&self) -> u32;

    /// Spike sample indices for a single cluster, sorted ascending.
    fn spike_times(&self, cluster: ClusterId) -> &[SampleIndex];

    /// Borrow a window of the raw trace as a zero-copy slice.
    fn trace(&self, start: SampleIndex, len: u32) -> TraceSlice<'_>;

    /// Initial label vector loaded from the backend's on-disk schema.
    fn initial_labels(&self) -> Vec<Self::Label>;

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
}
