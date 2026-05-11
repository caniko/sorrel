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

/// Owning table of per-cluster scalar metrics. Names and per-metric value
/// columns are kept in lockstep — there's no way to add one without the
/// other, which is the bug the previous parallel `Vec<String>` +
/// `HashMap<String, Vec<f32>>` shape made easy to introduce.
///
/// Iteration order matches insertion order, so callers (e.g. the cluster
/// table) get a deterministic column layout. Look-up is linear in the number
/// of metrics; in practice this is <20 columns and lives off the hot path.
#[derive(Debug, Default, Clone)]
pub struct QualityMetrics {
    names: Vec<String>,
    columns: Vec<Vec<f32>>,
}

impl QualityMetrics {
    pub const fn new() -> Self {
        Self {
            names: Vec::new(),
            columns: Vec::new(),
        }
    }

    /// Insert a metric column. Replaces any existing column with the same
    /// name. The caller is responsible for the column length matching
    /// `n_clusters` — typically `vec![NAN; n_clusters]` initialised once and
    /// filled in as rows arrive.
    pub fn insert(&mut self, name: String, values: Vec<f32>) {
        if let Some(i) = self.names.iter().position(|n| *n == name) {
            self.columns[i] = values;
        } else {
            self.names.push(name);
            self.columns.push(values);
        }
    }

    /// `true` if a metric named `name` is present.
    pub fn contains(&self, name: &str) -> bool {
        self.names.iter().any(|n| n == name)
    }

    /// Look up the per-cluster column for `name`. `None` when absent.
    pub fn get(&self, name: &str) -> Option<&[f32]> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| self.columns[i].as_slice())
    }

    /// Mutable access for in-place fills during CSV / TSV ingest.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut [f32]> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| self.columns[i].as_mut_slice())
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Sort columns alphabetically by name. The cluster-table UI relies on
    /// this for stable column ordering across loads.
    pub fn sort_by_name(&mut self) {
        let mut idx: Vec<usize> = (0..self.names.len()).collect();
        idx.sort_by(|&a, &b| self.names[a].cmp(&self.names[b]));
        let names = idx.iter().map(|&i| self.names[i].clone()).collect();
        let columns = idx
            .iter()
            .map(|&i| std::mem::take(&mut self.columns[i]))
            .collect();
        self.names = names;
        self.columns = columns;
    }
}

/// Per-cluster scalar metrics provided by an upstream tool (SpikeInterface's
/// `quality_metrics.csv`, phy's `cluster_*.tsv`, etc.). Each metric is a
/// named column; values are aligned by `ClusterId`.
///
/// Backends opt in by reading sidecar files at load time and storing them in
/// a [`QualityMetrics`] table (NaN where the upstream tool reports nothing).
pub trait HasQualityMetrics: DataProvider {
    /// Borrow the full metric table.
    fn quality_metrics(&self) -> &QualityMetrics;

    /// Sorted list of metric names exposed by this provider.
    fn metric_names(&self) -> &[String] {
        self.quality_metrics().names()
    }

    /// Per-cluster values for `name`, length = `n_clusters()`. `None` when
    /// the metric isn't present. NaN means "missing for this cluster".
    fn metric_values(&self, name: &str) -> Option<&[f32]> {
        self.quality_metrics().get(name)
    }
}
