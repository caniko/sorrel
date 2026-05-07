//! Optional capability traits — backends opt in when the underlying data is
//! available. Views and metrics depend on these as separate bounds, so a
//! backend that lacks (e.g.) channel positions can still drive the cluster
//! table and trace view without faking the missing data.

use crate::provider::{ChannelId, ClusterId, DataProvider};

/// Probe geometry: 2-D xy positions per channel, optional shank assignment,
/// optional channel-map (active row in the raw dat → logical channel id).
pub trait HasGeometry: DataProvider {
    /// `(x, y)` micrometre positions for every channel; one row per channel
    /// in the raw dat (so this length matches `n_channels`).
    fn channel_positions(&self) -> &[[f32; 2]];

    /// Per-channel shank index (zero-based). Default returns an empty slice
    /// for single-shank probes.
    fn channel_shanks(&self) -> &[u32] {
        &[]
    }

    /// Logical channel id per raw-dat row. Default is the identity mapping.
    fn channel_map(&self) -> &[ChannelId] {
        &[]
    }
}

/// Per-spike scalar amplitudes. Backends that store template-relative
/// amplitudes return them in their native units; consumers normalise.
pub trait HasAmplitudes: DataProvider {
    /// Amplitudes for the spikes returned by `spike_times(cluster)`,
    /// in matching order.
    fn spike_amplitudes(&self, cluster: ClusterId) -> &[f32];
}

/// Per-spike template assignment. Distinct from `spike_clusters` after
/// merges/splits — phy uses templates as the immutable seed and clusters as
/// the mutable curation surface.
pub trait HasSpikeTemplates: DataProvider {
    /// Template id per spike in the same order as `spike_times(cluster)`.
    fn spike_templates(&self, cluster: ClusterId) -> &[u32];
}

/// PC-feature backing for the FeatureView and lasso split.
///
/// phy stores `pc_features.npy` as a `(n_spikes, n_pcs, n_channels_per_template)`
/// float array, and `pc_feature_ind.npy` as `(n_templates, n_channels_per_template)`
/// channel id remap. We expose the flat buffer + shape and let view code do
/// the indexing, which keeps providers from having to materialise per-cluster
/// duplicates of an otherwise large table (up to hundreds of MB).
pub trait HasPcFeatures: DataProvider {
    /// Flat `f32` buffer in `(n_spikes, n_pcs, n_channels_per_template)`
    /// row-major order. Length = `n_spikes * n_pcs * n_channels_per_template`.
    fn pc_features(&self) -> &[f32];

    /// `(n_pcs, n_channels_per_template)` shape. Multiply with `n_spikes`
    /// (from spike_times) to get the flat length.
    fn pc_shape(&self) -> (usize, usize);

    /// `(n_templates, n_channels_per_template)` channel id remap.
    fn pc_feature_ind(&self) -> &[u32];

    /// Per-cluster, time-sorted, *original NPY row indices* into
    /// `pc_features` — equivalently, indices into `spike_times.npy` and
    /// `amplitudes.npy`. Returns an empty slice for clusters out of range.
    fn spike_pc_indices(&self, cluster: ClusterId) -> &[u32];
}

/// Backing for the TemplateView and SimilarityView.
///
/// phy stores template waveforms in `templates.npy` as a `(n_templates,
/// n_samples_per_template, n_channels)` float array, and pairwise template
/// similarity in `similar_templates.npy` as `(n_templates, n_templates)`.
/// Both are flat by design — view code does its own indexing.
pub trait HasTemplateWaveforms: DataProvider {
    /// Flat row-major buffer; length = `n_templates * n_samples * n_channels`.
    fn template_waveforms(&self) -> &[f32];

    /// `(n_templates, n_samples, n_channels)`. Returns `(0, 0, 0)` when the
    /// backend has no templates loaded.
    fn template_shape(&self) -> (usize, usize, usize);

    /// `(n_templates, n_templates)` similarity matrix. Returns an empty
    /// slice when `similar_templates.npy` wasn't found.
    fn similar_templates(&self) -> &[f32];
}

/// Per-cluster scalar metrics provided by an upstream tool (SpikeInterface's
/// `quality_metrics.csv`, phy's `cluster_*.tsv`, etc.). Each metric is a
/// named column; values are aligned by `ClusterId`.
///
/// Backends opt in by reading sidecar files at load time and storing one
/// flat `Vec<f32>` per metric (NaN where the upstream tool reports nothing).
pub trait HasQualityMetrics: DataProvider {
    /// Sorted list of metric names exposed by this provider.
    fn metric_names(&self) -> &[String];

    /// Per-cluster values for `name`, length = `n_clusters()`. `None` when
    /// the metric isn't present. NaN means "missing for this cluster".
    fn metric_values(&self, name: &str) -> Option<&[f32]>;
}
