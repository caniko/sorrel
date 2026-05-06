use std::fmt::Debug;

pub type SampleIndex = u64;
pub type ChannelId = u32;
pub type ClusterId = u32;

/// A contiguous, zero-copy slice into a memory-mapped trace buffer.
///
/// `samples` is interleaved `[t0_ch0, t0_ch1, ..., t0_chN-1, t1_ch0, ...]`
/// in the backend's native dtype, exposed as `i16` for V1.
pub struct TraceSlice<'a> {
    pub start: SampleIndex,
    pub n_channels: u32,
    pub samples: &'a [i16],
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
}
