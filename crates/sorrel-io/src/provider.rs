use std::fmt::Debug;

pub type SampleIndex = u64;
pub type ChannelId = u32;
pub type ClusterId = u32;

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

    /// Hint for amplitude scaling; defaults to the nominal full-scale of the
    /// dtype but backends with calibration data should override.
    fn amplitude_full_scale(&self) -> f32 {
        TraceDtype::I16.nominal_full_scale()
    }
}
