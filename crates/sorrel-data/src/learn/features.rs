//! Session feature/target extraction for the learned curator.
//!
//! [`extract`] runs on the *input only*: it opens no truth file, sees no
//! human label and builds every feature from the sorter's clusters, the
//! seeded amplitudes/PC features/templates and the recording's duration.
//! [`label`] then attaches supervised targets from an aligned truth, after
//! all input-only work is done. Keeping the two phases in separate functions
//! is deliberate: a caller cannot accidentally let a human decision steer a
//! feature, and the ordering is visible at the call site.
//!
//! Five tables come out of one session (see [`super::table`] for the file
//! format):
//!
//! * `roots`: one row per non-empty Kilosort root, with the brief's
//!   level-2 card metrics plus ACG/amplitude-shape/mixture/proposer summaries
//!   and within-session percentiles. Targets: the [`root_truths`] verdict and
//!   the best whole-root Jaccard against any human good unit.
//! * `candidates`: up to [`MAX_CANDIDATES_PER_ROOT`] deduplicated membership
//!   hypotheses per root with at least [`MIN_ROOT_SPIKES`] spikes, each scored
//!   by part/complement statistics and a CCG split test. Targets: Jaccard with
//!   the best human good unit (and with any non-noise unit).
//! * `pairs`: template-similar root pairs (top 5 per root), with template,
//!   amplitude, CCG and merged-refractory evidence. Targets: whether a human
//!   good unit spans both roots.
//! * `units`: one row per human final cluster (all labels), with its own
//!   spike-train statistics and its spread inside its parent roots' feature
//!   spaces. Target: the phy label.
//! * `spikes`: a deterministic time-stratified subsample per root with
//!   root-relative feature-space, amplitude and timing features. Targets: the
//!   spike's label and its root's majority good unit.
//!
//! Every membership-shaped feature is computed from a `Vec<u32>` of root-local
//! spike indices, so the same code scores sorrel's own candidate memberships
//! at inference time (root, core, kmeans, outlier, cleanup, density and axis
//! proposals all arrive in that form).
//!
//! [`root_truths`]: crate::diff::root_truths

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{ensure, Result};
use sorrel_compute::{
    auto_correlogram, bimodality_coefficient, fit_diag_gmm, ks_two_sample, mean_amplitude,
    sliding_refractory_argmin,
};
use sorrel_io::kilosort::PhyLabel;
use sorrel_io::{ClusterId, DataProvider, HasGeometry, SampleIndex};

use crate::axis_split::axis_candidates;
use crate::density_tree::density_candidates;
use crate::diff::{root_truths, Curation, SpikeDiffOpts};
use crate::evaluate::{ccg_z, eval_trains, unit_stats};
use crate::probe::compare_templates;
use crate::proposers::{
    all_cluster_globals, cluster_feature_space, core_split, kmeans_split, outlier_split,
};
use crate::qc_export::{collect_qc_rows_with, QcRow};
use crate::robust_cleanup::robust_cleanup as run_robust_cleanup;
use crate::session::Session;

use super::table::Table;
use super::EXTRACTOR_VERSION;

/// Roots below this many spikes get root/unit features but no candidate,
/// pair-reference or spike rows: proposer statistics are meaningless there.
pub const MIN_ROOT_SPIKES: usize = 100;
/// Hard cap on one root's candidate rows after deduplication.
pub const MAX_CANDIDATES_PER_ROOT: usize = 60;
/// Roots per pair's top-N similar list.
pub const PAIR_NEIGHBOURS: usize = 5;
/// ACG log bins: 0.5–100 ms, normalised by the 50–100 ms rate.
pub const ACG_BINS: usize = 25;
/// Amplitude-over-time bins, each normalised by the root's median amplitude.
pub const AMP_TIME_BINS: usize = 10;
/// k of the per-spike kNN density ratio.
pub const KNN_K: usize = 10;
/// Refractory window of the `f_refractory` flag: 1.5 ms.
pub const REFRACTORY_MS: f32 = 1.5;
/// Amplitude drift correction is computed in this many equal time bins.
const AMP_DRIFT_BINS: usize = 20;
/// Stride-sampled rows for mixture/`hartigan_dip` fits.
const MAX_SHAPE_FIT: usize = 5000;

/// Proposer family of a candidate membership, stored in `f_method`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// The whole root.
    Whole = 0,
    /// `core_split_proposal`'s kept or minor part.
    Core = 1,
    /// `kmeans_split_proposal`'s largest part or the moved-out part.
    Kmeans = 2,
    /// `outlier_proposal`'s outlier set.
    Outlier = 3,
    /// `robust_cleanup`'s kept majority or dropped minority.
    Robust = 4,
    /// A `density_tree` node.
    Density = 5,
    /// An `axis_split` node.
    Axis = 6,
}

impl Method {
    fn code(self) -> f32 {
        self as u32 as f32
    }
}

/// Extraction knobs.
#[derive(Clone, Copy, Debug)]
pub struct ExtractOpts {
    /// Upper bound on `spikes` rows per root (a time-stratified subsample).
    pub max_spikes_per_root: usize,
    /// Reserved for tie-breaking; the current extractor is fully
    /// deterministic without randomness.
    pub seed: u64,
}

impl Default for ExtractOpts {
    fn default() -> Self {
        Self {
            max_spikes_per_root: 1500,
            seed: 0,
        }
    }
}

/// The five tables of one session.
pub struct SessionTables {
    /// One row per non-empty root.
    pub roots: Table,
    /// Candidate memberships of roots with at least [`MIN_ROOT_SPIKES`].
    pub candidates: Table,
    /// Similar root pairs.
    pub pairs: Table,
    /// One row per human final cluster (filled by [`label`]).
    pub units: Table,
    /// Per-spike rows for roots with at least [`MIN_ROOT_SPIKES`].
    pub spikes: Table,
}

/// Everything [`extract`] computed, plus the side data [`label`] needs.
///
/// Owns plain vectors (no borrow of the session), so a worker thread can move
/// it to the main thread before the truth is opened.
pub struct ExtractData {
    /// The five tables.
    pub tables: SessionTables,
    /// Root spike id per session-global spike.
    pub root_of_spike: Vec<u32>,
    /// Root-local index per session-global spike.
    pub local_index: Vec<u32>,
    /// Time per session-global spike.
    pub time_of_spike: Vec<SampleIndex>,
    /// Amplitude per session-global spike, `NaN` where unavailable.
    pub amp_of_spike: Vec<f32>,
    /// Whitened feature space per root (`None` without PC features).
    pub root_feats: Vec<Option<(Vec<f32>, usize)>>,
    /// Robust per-dimension centre and scale per root, computed once.
    pub root_center: Vec<Option<(Vec<f32>, Vec<f32>)>>,
    /// Spike count per root (empty roots are zero).
    pub root_sizes: Vec<usize>,
    /// Sum and count of finite amplitudes per root.
    pub root_amp_sum: Vec<f64>,
    /// Finite amplitudes per root.
    pub root_amp_count: Vec<usize>,
    /// Global spikes of each candidate row, aligned with `tables.candidates`.
    pub candidate_members: Vec<Vec<u32>>,
    /// Global spike of each `tables.spikes` row.
    pub spike_rows: Vec<u32>,
    /// `(root_a, root_b)` of each `tables.pairs` row, `root_a < root_b`.
    pub pair_roots: Vec<(u32, u32)>,
    /// Session sample rate.
    pub sr: f32,
    /// Recording length in samples.
    pub total_samples: u64,
}

/// Aligned human truth for one session.
pub struct Truth {
    /// Truth cluster id per session-global spike, in session spike order.
    /// The caller must have aligned the NPY rows through the validated
    /// permutation (see `bench::coverage::aligned_truth`).
    pub cluster: Vec<u32>,
    /// `(cluster, label)` rows of the truth's `cluster_group.tsv`.
    pub labels: Vec<(u32, PhyLabel)>,
}

/// Small name-keyed row builder: reads by name, writes by column order.
struct Row<'a> {
    names: &'a [String],
    values: Vec<f32>,
}

impl<'a> Row<'a> {
    fn new(names: &'a [String]) -> Self {
        Self {
            names,
            values: vec![f32::NAN; names.len()],
        }
    }

    fn set(&mut self, name: &str, value: f32) {
        let i = self
            .names
            .iter()
            .position(|c| c == name)
            .unwrap_or_else(|| panic!("feature column {name:?} is not declared"));
        self.values[i] = value;
    }

    fn set_bool(&mut self, name: &str, value: bool) {
        self.set(name, u8::from(value) as f32);
    }
}

fn cols(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

fn numbered(prefix: &str, n: usize) -> Vec<String> {
    (0..n).map(|i| format!("{prefix}{i:02}")).collect()
}

/// All feature/target column names of the `roots` table.
pub fn root_columns() -> Vec<String> {
    let mut c = cols(&["sid_idx", "root"]);
    c.extend(cols(&[
        "f_n_spikes",
        "f_log_n_spikes",
        "f_firing_rate",
        "f_presence",
        "f_isi_violations",
        "f_isi_violation_ratio",
        "f_rp_contamination",
        "f_rp_estimate_ms",
        "f_amplitude_cutoff",
        "f_amplitude_snr",
        "f_amplitude_dip",
        "f_amp_bimodality",
        "f_mean_amplitude",
        "f_drift_correlation",
        "f_drift_slope",
        "f_longest_silent_gap",
        "f_isolation_distance",
        "f_l_ratio",
        "f_nn_isolation",
        "f_d_prime",
        "f_silhouette",
        "f_trough_to_peak_ms",
        "f_half_width_ms",
        "f_peak_trough_ratio",
        "f_repolarisation_slope",
        "f_n_extrema",
        "f_x_um",
        "f_y_um",
        "f_n_active_channels",
        "f_footprint_decay_um",
        "f_footprint_r80_um",
        "f_propagation_ms_per_100um",
        "f_burst_index",
        "f_feature_dim",
        "f_maha_frac3",
        "f_maha_frac5",
        "f_bic_gain_k2",
        "f_bic_gain_k3",
        "f_bic_gain_k4",
        "f_core_found",
        "f_core_kind",
        "f_core_k",
        "f_core_minor_frac",
        "f_core_minor_rpc",
        "f_core_kept_rpc",
        "f_core_dprime",
        "f_core_score",
        "f_kmeans_found",
        "f_kmeans_k",
        "f_kmeans_minor_frac",
        "f_kmeans_silhouette",
        "f_kmeans_dprime",
        "f_robust_found",
        "f_robust_minor_frac",
        "f_robust_bic_gain",
        "f_robust_whole_rpc",
        "f_robust_kept_rpc",
        "f_outlier_found",
        "f_outlier_frac",
        "f_density_nodes",
        "f_axis_nodes",
        "f_n_roots",
        "f_root_size_rank",
        "f_pct_n_spikes",
        "f_pct_firing_rate",
        "f_pct_rp_contamination",
        "f_pct_amplitude_snr",
        "f_pct_amplitude_cutoff",
        "f_pct_presence",
        "f_pct_drift_correlation",
        "f_pct_isolation_distance",
        "f_pct_l_ratio",
        "f_pct_d_prime",
        "f_pct_silhouette",
        "f_pct_burst_index",
    ]));
    c.extend(numbered("f_acg_", ACG_BINS));
    c.extend(numbered("f_amp_time_", AMP_TIME_BINS));
    c.extend(cols(&[
        "y_verdict",
        "y_n_good",
        "y_good_share",
        "y_whole_jaccard",
    ]));
    c
}

/// All feature/target column names of the `candidates` table.
pub fn candidate_columns() -> Vec<String> {
    let mut c = cols(&["sid_idx", "root", "cand_idx"]);
    c.extend(cols(&[
        "f_method",
        "f_complement",
        "f_depth",
        "f_size_frac",
        "f_log_n",
        "f_part_rpc",
        "f_part_isi",
        "f_part_presence",
        "f_part_amp_cutoff",
        "f_part_drift_corr",
        "f_comp_rpc",
        "f_comp_isi",
        "f_comp_presence",
        "f_comp_amp_cutoff",
        "f_comp_drift_corr",
        "f_dip_z",
        "f_sync_z",
        "f_amp_ratio",
        "f_bic_gain",
        "f_supported",
        "f_rpc_delta",
        "f_root_rpc",
        "y_jaccard",
        "y_jaccard_any",
        "y_hit",
        "y_best_unit",
    ]));
    c
}

/// All feature/target column names of the `pairs` table.
pub fn pair_columns() -> Vec<String> {
    let mut c = cols(&["sid_idx", "root_a", "root_b"]);
    c.extend(cols(&[
        "f_template_cosine",
        "f_peak_ch_abs_diff",
        "f_peak_ch_dist_um",
        "f_shared_ch",
        "f_amp_ratio",
        "f_amp_ks",
        "f_size_ratio",
        "f_rpc_a",
        "f_rpc_b",
        "f_ccg_dip_z",
        "f_ccg_sync_z",
        "f_merged_rpc",
        "f_merged_isi",
        "f_max_part_rpc",
        "f_rpc_delta",
        "y_merge",
        "y_shared",
    ]));
    c
}

/// All feature/target column names of the `units` table.
pub fn unit_columns() -> Vec<String> {
    let mut c = cols(&["sid_idx", "unit"]);
    c.extend(cols(&[
        "f_n_spikes",
        "f_log_n_spikes",
        "f_n_roots",
        "f_root_share",
        "f_rpc",
        "f_isi_violation_ratio",
        "f_presence",
        "f_amp_cutoff",
        "f_drift_corr",
        "f_mean_amplitude",
        "f_amp_ratio",
        "f_feature_dim",
        "f_maha_frac3",
        "f_maha_frac5",
        "f_maha_median",
    ]));
    c.extend(numbered("f_acg_", ACG_BINS));
    c.extend(numbered("f_amp_time_", AMP_TIME_BINS));
    c.push("y_label".to_owned());
    c
}

/// All feature/target column names of the `spikes` table.
pub fn spike_columns() -> Vec<String> {
    cols(&[
        "sid_idx",
        "root",
        "local_idx",
        "f_maha",
        "f_maha_pct",
        "f_z_pc0",
        "f_z_pc1",
        "f_z_pc2",
        "f_z_pc3",
        "f_z_pc4",
        "f_z_pc5",
        "f_amp_z",
        "f_amp_z_drift",
        "f_amp_pct",
        "f_knn_density_ratio",
        "f_time_frac",
        "f_log_isi_prev",
        "f_log_isi_next",
        "f_refractory",
        "y_good",
        "y_major",
        "y_unit",
    ])
}

/// Proposer summary of one root, computed once and shared by the root and
/// candidate rows.
struct RootProposals {
    core: Option<crate::proposers::CoreSplit>,
    kmeans: Option<crate::proposers::KmeansSplit>,
    outlier: Option<Vec<u32>>,
    robust: Option<crate::robust_cleanup::RobustCleanup>,
    density: Vec<crate::density_tree::DensityNode>,
    axis: Vec<crate::axis_split::AxisCandidate>,
}

struct RootData {
    root: u32,
    n: usize,
    rpc: f32,
    proposal: RootProposals,
}

/// A metric selector over one root's QC row.
type MetricGet = fn(&QcRow) -> f32;

/// One candidate hypothesis before it becomes a row.
type Proposal = (Method, bool, Option<f64>, bool, u32, Vec<u32>);

/// How a candidate membership was proposed.
struct Spec {
    method: Method,
    bic: Option<f64>,
    supported: bool,
    depth: u32,
}

/// Compute the input-only tables for one session.
pub fn extract<P: DataProvider + HasGeometry>(
    session: &Session<P>,
    sid: &str,
    opts: &ExtractOpts,
) -> Result<ExtractData> {
    let sr = session.provider.sample_rate().max(1.0);
    let total = session.provider.n_samples().0;
    let phase_start = std::time::Instant::now();
    let phase = |name: &str| {
        if std::env::var_os("SORREL_LEARN_TRACE").is_some() {
            eprintln!("  {name} {:.1}s", phase_start.elapsed().as_secs_f64());
        }
    };
    let globals = all_cluster_globals(session);
    let n_spikes = session.spike_clusters_global().len();
    ensure!(
        globals.iter().map(Vec::len).sum::<usize>() == n_spikes,
        "cluster buckets do not cover every spike"
    );

    // Global per-spike views, built once. Root-local order equals global order
    // within a root, so `local_index` is the bridge back to feature rows.
    let mut root_of_spike = vec![0u32; n_spikes];
    let mut local_index = vec![0u32; n_spikes];
    let mut time_of_spike = vec![SampleIndex(0); n_spikes];
    let mut amp_of_spike = vec![f32::NAN; n_spikes];
    let mut root_amp_sum = vec![0.0f64; globals.len()];
    let mut root_amp_count = vec![0usize; globals.len()];
    for (r, g) in globals.iter().enumerate() {
        let times = session.spike_times(ClusterId(r as u32));
        let amps = session.spike_amplitudes(ClusterId(r as u32));
        ensure!(
            times.len() == g.len(),
            "root {r} has {} times for {} spikes",
            times.len(),
            g.len()
        );
        for (i, &global) in g.iter().enumerate() {
            root_of_spike[global as usize] = r as u32;
            local_index[global as usize] = i as u32;
            time_of_spike[global as usize] = times[i];
            if amps.len() == times.len() {
                amp_of_spike[global as usize] = amps[i];
                if amps[i].is_finite() {
                    root_amp_sum[r] += amps[i] as f64;
                    root_amp_count[r] += 1;
                }
            }
        }
    }

    let positions = session.provider.channel_positions().to_vec();
    let qc = collect_qc_rows_with(session, Some(&positions));
    phase("root cards");
    let non_empty: Vec<usize> = (0..globals.len())
        .filter(|&r| !globals[r].is_empty())
        .collect();

    // Every feature below is input-only. Proposers run before any truth is
    // even loaded by the caller.
    let mut root_data = Vec::with_capacity(non_empty.len());
    let mut root_feats: Vec<Option<(Vec<f32>, usize)>> = vec![None; globals.len()];
    let mut root_center: Vec<Option<(Vec<f32>, Vec<f32>)>> = vec![None; globals.len()];
    for &r in &non_empty {
        let c = ClusterId(r as u32);
        let g = &globals[r];
        let feats = cluster_feature_space(session, c, g);
        root_center[r] = feats.as_ref().map(|(f, d)| robust_center_scale_dims(f, *d));
        let proposal = RootProposals {
            core: core_split(session, c, g),
            kmeans: kmeans_split(session, c, g),
            outlier: outlier_split(session, c, g),
            robust: run_robust_cleanup(session, c, g),
            density: density_candidates(session, c, g),
            axis: axis_candidates(session, c, g),
        };
        root_data.push(RootData {
            root: r as u32,
            n: g.len(),
            rpc: qc[r].rp_contamination,
            proposal,
        });
        root_feats[r] = feats;
    }
    phase("proposers");

    let mut roots = Table::new(EXTRACTOR_VERSION, root_columns());
    roots.sids = vec![sid.to_owned()];
    roots.codes.insert(
        "y_verdict".into(),
        [("noise", 0), ("kept", 1), ("cleanup", 2), ("cores", 3)]
            .into_iter()
            .map(|(name, code)| (name.to_owned(), code))
            .collect(),
    );
    roots.notes.insert(
        "y_verdict".into(),
        "the dev DB's root_truths verdict; `other` is folded into cores (3)".into(),
    );
    roots.notes.insert(
        "f_root_size_rank".into(),
        "fraction of the session's roots with at most this root's spike count".into(),
    );
    let mut candidates = Table::new(EXTRACTOR_VERSION, candidate_columns());
    candidates.sids = vec![sid.to_owned()];
    candidates.codes.insert(
        "f_method".into(),
        [
            (Method::Whole, "whole"),
            (Method::Core, "core"),
            (Method::Kmeans, "kmeans"),
            (Method::Outlier, "outlier"),
            (Method::Robust, "robust"),
            (Method::Density, "density"),
            (Method::Axis, "axis"),
        ]
        .into_iter()
        .map(|(m, name)| (name.to_owned(), m.code() as i64))
        .collect(),
    );
    candidates.codes.insert(
        "f_complement".into(),
        [("part", 0), ("complement", 1)]
            .into_iter()
            .map(|(name, code)| (name.to_owned(), code))
            .collect(),
    );
    candidates.notes.insert(
        "y_jaccard_any".into(),
        "max Jaccard against any human unit whose label is not noise (mua and unsorted count)"
            .into(),
    );
    candidates.notes.insert(
        "y_best_unit".into(),
        "truth cluster id of the good unit that achieves y_jaccard; NaN when no good unit is \
         present in the candidate"
            .into(),
    );
    candidates.notes.insert(
        "f_amp_ratio".into(),
        "mean amplitude of the candidate part divided by the complement's".into(),
    );
    candidates.notes.insert(
        "f_rpc_delta".into(),
        "candidate part's sliding-refractory contamination minus the whole root's".into(),
    );
    let mut pairs = Table::new(EXTRACTOR_VERSION, pair_columns());
    pairs.sids = vec![sid.to_owned()];
    pairs.notes.insert(
        "y_merge".into(),
        "1 when some human good unit takes at least 20% of each root's spikes and at least 20% \
         of its own spikes come from each root"
            .into(),
    );
    pairs.notes.insert(
        "y_shared".into(),
        "max over human good units of Jaccard(root_a ∪ root_b, unit) with global unit totals"
            .into(),
    );
    let mut units = Table::new(EXTRACTOR_VERSION, unit_columns());
    units.codes.insert(
        "y_label".into(),
        [("noise", 0), ("mua", 1), ("good", 2)]
            .into_iter()
            .map(|(name, code)| (name.to_owned(), code))
            .collect(),
    );
    units.notes.insert(
        "y_label".into(),
        "phy label; unsorted is folded into noise (0)".into(),
    );
    units.notes.insert(
        "f_root_share".into(),
        "unit spikes over the spike counts of the roots it spans".into(),
    );
    let mut spikes = Table::new(EXTRACTOR_VERSION, spike_columns());
    spikes.sids = vec![sid.to_owned()];
    spikes.notes.insert(
        "y_unit".into(),
        "dense local index of the spike's truth cluster within this session (sorted truth ids); \
         for core grouping, not a global unit id"
            .into(),
    );
    spikes.notes.insert(
        "f_refractory".into(),
        "1 when the previous or next spike of the same root is within 1.5 ms".into(),
    );
    spikes.notes.insert(
        "f_knn_density_ratio".into(),
        "mean distance to the 10 nearest spikes of the same root's subsample divided by the \
         root's median of that distance"
            .into(),
    );

    // Within-session context: percentiles are over non-empty roots.
    let metrics: [(&str, MetricGet); 12] = [
        ("f_pct_n_spikes", |q| q.n_spikes as f32),
        ("f_pct_firing_rate", |q| q.firing_rate_hz),
        ("f_pct_rp_contamination", |q| q.rp_contamination),
        ("f_pct_amplitude_snr", |q| q.amplitude_snr),
        ("f_pct_amplitude_cutoff", |q| q.amplitude_cutoff),
        ("f_pct_presence", |q| q.presence_ratio),
        ("f_pct_drift_correlation", |q| q.drift_correlation),
        ("f_pct_isolation_distance", |q| q.isolation_distance_sq),
        ("f_pct_l_ratio", |q| q.l_ratio),
        ("f_pct_d_prime", |q| q.d_prime),
        ("f_pct_silhouette", |q| q.silhouette),
        ("f_pct_burst_index", |q| q.burst_index),
    ];
    let n_roots = non_empty.len();

    for rd in &root_data {
        let r = rd.root as usize;
        let q = &qc[r];
        let g = &globals[r];
        let times = session.spike_times(ClusterId(r as u32));
        let amps = session.spike_amplitudes(ClusterId(r as u32));
        let amps = if amps.len() == times.len() { amps } else { &[] };
        let mut row = Row::new(&roots.columns);
        row.set("sid_idx", 0.0);
        row.set("root", r as f32);
        row.set("f_n_spikes", q.n_spikes as f32);
        row.set("f_log_n_spikes", (1.0 + q.n_spikes as f32).log10());
        row.set("f_firing_rate", q.firing_rate_hz);
        row.set("f_presence", q.presence_ratio);
        row.set("f_isi_violations", q.isi_violations as f32);
        row.set("f_isi_violation_ratio", q.isi_violation_ratio);
        row.set("f_rp_contamination", q.rp_contamination);
        row.set("f_rp_estimate_ms", q.rp_estimate_ms);
        row.set("f_amplitude_cutoff", q.amplitude_cutoff);
        row.set("f_amplitude_snr", q.amplitude_snr);
        row.set("f_amplitude_dip", q.amplitude_dip);
        row.set("f_mean_amplitude", q.mean_amplitude);
        row.set("f_drift_correlation", q.drift_correlation);
        row.set("f_drift_slope", q.drift_slope);
        row.set("f_longest_silent_gap", q.longest_silent_gap);
        row.set("f_isolation_distance", q.isolation_distance_sq);
        row.set("f_l_ratio", q.l_ratio);
        row.set("f_nn_isolation", q.nn_isolation);
        row.set("f_d_prime", q.d_prime);
        row.set("f_silhouette", q.silhouette);
        row.set("f_trough_to_peak_ms", q.trough_to_peak_ms);
        row.set("f_half_width_ms", q.half_width_ms);
        row.set("f_peak_trough_ratio", q.peak_trough_ratio);
        row.set("f_repolarisation_slope", q.repolarisation_slope);
        row.set("f_n_extrema", q.n_extrema);
        row.set("f_x_um", q.x_um);
        row.set("f_y_um", q.y_um);
        row.set("f_n_active_channels", q.n_active_channels);
        row.set("f_footprint_decay_um", q.footprint_decay_um);
        row.set("f_footprint_r80_um", q.footprint_r80_um);
        row.set("f_propagation_ms_per_100um", q.propagation_ms_per_100um);
        row.set("f_burst_index", q.burst_index);

        let strided: Vec<f32> = amps
            .iter()
            .copied()
            .step_by(amps.len().div_ceil(MAX_SHAPE_FIT).max(1))
            .collect();
        if strided.len() >= 4 {
            row.set("f_amp_bimodality", bimodality_coefficient(&strided));
        }
        for (i, v) in acg_log_bins(times, sr).into_iter().enumerate() {
            row.set(&format!("f_acg_{i:02}"), v);
        }
        for (i, v) in amp_time_bins(times, amps, total, AMP_TIME_BINS)
            .into_iter()
            .enumerate()
        {
            row.set(&format!("f_amp_time_{i:02}"), v);
        }

        if let Some((feats, d)) = &root_feats[r] {
            row.set("f_feature_dim", *d as f32);
            let maha = mahalanobis(feats, *d);
            row.set("f_maha_frac3", fraction_above(&maha, 3.0));
            row.set("f_maha_frac5", fraction_above(&maha, 5.0));
            let sub = stride_rows(feats, *d, MAX_SHAPE_FIT);
            let m1 = fit_diag_gmm(&sub, *d, 1, 60);
            if let Some(m1) = m1 {
                for k in 2..=4 {
                    if let Some(mk) = fit_diag_gmm(&sub, *d, k, 60) {
                        row.set(&format!("f_bic_gain_k{k}"), (m1.bic() - mk.bic()) as f32);
                    }
                }
            }
        }

        let p = &rd.proposal;
        row.set_bool("f_core_found", p.core.is_some());
        if let Some(core) = &p.core {
            row.set("f_core_kind", if core.method == "core" { 1.0 } else { 2.0 });
            row.set("f_core_k", core.k as f32);
            row.set(
                "f_core_minor_frac",
                core.minor_idx.len() as f32 / g.len() as f32,
            );
            row.set(
                "f_core_minor_rpc",
                measure(times, &core.minor_idx, sr, total),
            );
            row.set("f_core_kept_rpc", core.core_contamination);
            row.set("f_core_dprime", core.d_prime);
            row.set("f_core_score", core.score);
        }
        row.set_bool("f_kmeans_found", p.kmeans.is_some());
        if let Some(k) = &p.kmeans {
            row.set("f_kmeans_k", k.k as f32);
            row.set(
                "f_kmeans_minor_frac",
                k.minor_idx.len() as f32 / g.len() as f32,
            );
            row.set("f_kmeans_silhouette", k.silhouette);
            row.set("f_kmeans_dprime", k.d_prime);
        }
        row.set_bool("f_robust_found", p.robust.is_some());
        if let Some(b) = &p.robust {
            row.set(
                "f_robust_minor_frac",
                b.minor_idx.len() as f32 / g.len() as f32,
            );
            row.set("f_robust_bic_gain", b.bic_gain as f32);
            row.set("f_robust_whole_rpc", b.whole_contamination);
            row.set("f_robust_kept_rpc", b.kept_contamination);
        }
        row.set_bool("f_outlier_found", p.outlier.is_some());
        if let Some(o) = &p.outlier {
            row.set("f_outlier_frac", o.len() as f32 / g.len() as f32);
        }
        row.set("f_density_nodes", p.density.len() as f32);
        row.set("f_axis_nodes", p.axis.len() as f32);
        row.set("f_n_roots", n_roots as f32);
        row.set(
            "f_root_size_rank",
            (non_empty
                .iter()
                .filter(|&&o| globals[o].len() <= g.len())
                .count()) as f32
                / n_roots.max(1) as f32,
        );
        for (name, get) in metrics {
            row.set(
                name,
                within_session_percentile(&qc, &non_empty, get, get(q)),
            );
        }
        roots.push_row(&row.values)?;
    }

    // Candidate memberships: generated in a fixed priority order, complements
    // appended, duplicates dropped by a hash of the membership.
    let mut candidate_members: Vec<Vec<u32>> = Vec::new();
    for rd in &root_data {
        if rd.n < MIN_ROOT_SPIKES {
            continue;
        }
        let r = rd.root as usize;
        let g = &globals[r];
        let times = session.spike_times(ClusterId(rd.root));
        let amps = session.spike_amplitudes(ClusterId(rd.root));
        let amps = if amps.len() == times.len() { amps } else { &[] };
        let mut proposals: Vec<Proposal> = Vec::new();
        let mut seen: BTreeSet<[u8; 32]> = BTreeSet::new();
        let all: Vec<u32> = (0..rd.n as u32).collect();
        seen.insert(membership_hash(&all));
        proposals.push((Method::Whole, false, None, true, 0, all));
        let p = &rd.proposal;
        let spec = |method: Method, bic: Option<f64>, supported: bool, depth: u32| Spec {
            method,
            bic,
            supported,
            depth,
        };
        if let Some(core) = &p.core {
            let kept: Vec<u32> = (0..rd.n as u32)
                .filter(|i| core.minor_idx.binary_search(i).is_err())
                .collect();
            let s = spec(Method::Core, None, core.score >= 0.5, 0);
            push_candidate(&mut proposals, &mut seen, rd.n, kept, &s);
            push_candidate(&mut proposals, &mut seen, rd.n, core.minor_idx.clone(), &s);
        }
        if let Some(k) = &p.kmeans {
            let kept: Vec<u32> = (0..rd.n as u32)
                .filter(|i| k.minor_idx.binary_search(i).is_err())
                .collect();
            let s = spec(Method::Kmeans, None, k.silhouette >= 0.25, 0);
            push_candidate(&mut proposals, &mut seen, rd.n, kept, &s);
            push_candidate(&mut proposals, &mut seen, rd.n, k.minor_idx.clone(), &s);
        }
        if let Some(o) = &p.outlier {
            let s = spec(Method::Outlier, None, true, 0);
            push_candidate(&mut proposals, &mut seen, rd.n, o.clone(), &s);
        }
        if let Some(b) = &p.robust {
            let kept: Vec<u32> = (0..rd.n as u32)
                .filter(|i| b.minor_idx.binary_search(i).is_err())
                .collect();
            let s = spec(Method::Robust, Some(b.bic_gain), b.supported, 0);
            push_candidate(&mut proposals, &mut seen, rd.n, kept, &s);
            push_candidate(&mut proposals, &mut seen, rd.n, b.minor_idx.clone(), &s);
        }
        for node in &p.density {
            let s = spec(
                Method::Density,
                Some(node.bic_gain),
                node.supported,
                node.depth as u32,
            );
            push_candidate(&mut proposals, &mut seen, rd.n, node.members.clone(), &s);
        }
        for node in &p.axis {
            let s = spec(Method::Axis, Some(node.bic_gain), node.supported, 0);
            push_candidate(&mut proposals, &mut seen, rd.n, node.members.clone(), &s);
        }
        proposals.truncate(MAX_CANDIDATES_PER_ROOT);
        let whole_rpc = rd.rpc;
        for (cand_idx, (method, complement, bic, supported, depth, members)) in
            proposals.into_iter().enumerate()
        {
            candidate_members.push(members.iter().map(|&i| g[i as usize]).collect());
            let mut row = Row::new(&candidates.columns);
            row.set("sid_idx", 0.0);
            row.set("root", rd.root as f32);
            row.set("cand_idx", cand_idx as f32);
            row.set("f_method", method.code());
            row.set_bool("f_complement", complement);
            row.set("f_depth", depth as f32);
            row.set("f_size_frac", members.len() as f32 / rd.n as f32);
            row.set("f_log_n", (1.0 + members.len() as f32).log10());
            let ev = crate::evaluate::eval_split_trains(times, amps, &members, sr, total);
            // `eval_split_trains` puts the representatives of `members` last:
            // parts[1] is the candidate, parts[0] its complement.
            let part = ev.parts[1];
            let comp = ev.parts[0];
            row.set("f_part_rpc", part.rp_contamination);
            row.set("f_part_isi", part.isi_violation_ratio);
            row.set("f_part_presence", part.presence_ratio);
            row.set("f_part_amp_cutoff", part.amplitude_cutoff);
            row.set("f_part_drift_corr", part.drift_correlation);
            row.set("f_comp_rpc", comp.rp_contamination);
            row.set("f_comp_isi", comp.isi_violation_ratio);
            row.set("f_comp_presence", comp.presence_ratio);
            row.set("f_comp_amp_cutoff", comp.amplitude_cutoff);
            row.set("f_comp_drift_corr", comp.drift_correlation);
            row.set("f_dip_z", ev.dip_z);
            row.set("f_sync_z", ev.sync_z);
            row.set(
                "f_amp_ratio",
                if ev.mean_amp[0].is_finite() && ev.mean_amp[0].abs() > 0.0 {
                    ev.mean_amp[1] / ev.mean_amp[0]
                } else {
                    f32::NAN
                },
            );
            if let Some(b) = bic {
                row.set("f_bic_gain", b as f32);
            }
            if method != Method::Whole {
                row.set_bool("f_supported", supported);
            }
            row.set(
                "f_rpc_delta",
                if part.rp_contamination.is_finite() && whole_rpc.is_finite() {
                    part.rp_contamination - whole_rpc
                } else {
                    f32::NAN
                },
            );
            row.set("f_root_rpc", whole_rpc);
            candidates.push_row(&row.values)?;
        }
    }

    phase("candidates");
    // Merge pairs: the five most template-similar roots of every non-empty
    // root with enough spikes, as unordered deduplicated pairs.
    let mut pair_roots: Vec<(u32, u32)> = Vec::new();
    let mut pair_rows: Vec<[f32; 15]> = Vec::new();
    let mut seen_pairs: BTreeSet<(u32, u32)> = BTreeSet::new();
    for rd in &root_data {
        if rd.n < MIN_ROOT_SPIKES {
            continue;
        }
        let a = ClusterId(rd.root);
        let neighbours = session.similar_clusters(a, &[]);
        for &(b, similarity) in neighbours.iter().take(PAIR_NEIGHBOURS) {
            let (lo, hi) = (rd.root.min(b.0), rd.root.max(b.0));
            if lo == hi || !seen_pairs.insert((lo, hi)) {
                continue;
            }
            let times_a = session.spike_times(ClusterId(lo));
            let times_b = session.spike_times(ClusterId(hi));
            let amps_a = session.spike_amplitudes(ClusterId(lo));
            let amps_b = session.spike_amplitudes(ClusterId(hi));
            let (amps_a, amps_b) = (
                if amps_a.len() == times_a.len() {
                    amps_a
                } else {
                    &[]
                },
                if amps_b.len() == times_b.len() {
                    amps_b
                } else {
                    &[]
                },
            );
            let (cosine, peak_diff, dist_um, shared) = template_compare(session, lo, hi);
            let merged = eval_trains(&[(times_a, amps_a), (times_b, amps_b)], sr, total);
            let max_part_rpc =
                merged
                    .parts
                    .iter()
                    .map(|p| p.rp_contamination)
                    .fold(f32::NAN, |acc, v| {
                        if v.is_nan() {
                            acc
                        } else if acc.is_nan() {
                            v
                        } else {
                            acc.max(v)
                        }
                    });
            let (dip, sync) = ccg_z(times_a, times_b, sr);
            let f = |v: f32| if v.is_finite() { v } else { f32::NAN };
            pair_rows.push([
                cosine.unwrap_or_else(|| {
                    if similarity.is_finite() {
                        similarity
                    } else {
                        f32::NAN
                    }
                }),
                peak_diff,
                dist_um,
                shared as f32,
                ratio_of(mean_amplitude(amps_a), mean_amplitude(amps_b)),
                f(ks_two_sample(amps_a, amps_b)),
                ratio_of(times_a.len() as f32, times_b.len() as f32),
                qc[lo as usize].rp_contamination,
                qc[hi as usize].rp_contamination,
                dip,
                sync,
                merged.merged.rp_contamination,
                merged.merged.isi_violation_ratio,
                max_part_rpc,
                if merged.merged.rp_contamination.is_finite() && max_part_rpc.is_finite() {
                    merged.merged.rp_contamination - max_part_rpc
                } else {
                    f32::NAN
                },
            ]);
            pair_roots.push((lo, hi));
        }
    }
    for (i, (lo, hi)) in pair_roots.iter().enumerate() {
        let mut row = Row::new(&pairs.columns);
        row.set("sid_idx", 0.0);
        row.set("root_a", *lo as f32);
        row.set("root_b", *hi as f32);
        for (j, name) in [
            "f_template_cosine",
            "f_peak_ch_abs_diff",
            "f_peak_ch_dist_um",
            "f_shared_ch",
            "f_amp_ratio",
            "f_amp_ks",
            "f_size_ratio",
            "f_rpc_a",
            "f_rpc_b",
            "f_ccg_dip_z",
            "f_ccg_sync_z",
            "f_merged_rpc",
            "f_merged_isi",
            "f_max_part_rpc",
            "f_rpc_delta",
        ]
        .into_iter()
        .enumerate()
        {
            row.set(name, pair_rows[i][j]);
        }
        pairs.push_row(&row.values)?;
    }

    phase("pairs");
    // Per-spike rows from a deterministic time-stratified subsample.
    let mut spike_rows: Vec<u32> = Vec::new();
    for rd in &root_data {
        if rd.n < MIN_ROOT_SPIKES || opts.max_spikes_per_root == 0 {
            continue;
        }
        let r = rd.root as usize;
        let g = &globals[r];
        let times = session.spike_times(ClusterId(rd.root));
        let amps = session.spike_amplitudes(ClusterId(rd.root));
        let amps = if amps.len() == times.len() { amps } else { &[] };
        let n = rd.n;
        let m = n.min(opts.max_spikes_per_root);
        let pick: Vec<usize> = (0..m)
            .map(|i| {
                let lo = i * n / m;
                let hi = (i + 1) * n / m;
                lo + (hi - lo) / 2
            })
            .collect();
        let feats = root_feats[r].as_ref();
        let maha = feats.map(|(f, d)| mahalanobis(f, *d));
        let centre_scale: Option<(Vec<f32>, Vec<f32>)> =
            feats.as_ref().map(|(f, d)| robust_center_scale_dims(f, *d));
        let maha_sorted = maha.as_ref().map(|m| {
            let mut s = m.clone();
            s.sort_by(f32::total_cmp);
            s
        });
        let (med_amp, mad_amp) = robust_center_scale(amps);
        let amp_bins = amp_drift_bins(times, amps, total, AMP_DRIFT_BINS);
        let mut sorted_amps: Vec<f32> = amps.iter().copied().filter(|v| v.is_finite()).collect();
        sorted_amps.sort_by(f32::total_cmp);
        let knn = feats.map(|(f, d)| {
            let sub = stride_rows_pick(f, *d, &pick);
            knn_relative_density(&sub, *d)
        });
        let refractory = (sr * REFRACTORY_MS / 1000.0).round() as u64;
        for (j, &i) in pick.iter().enumerate() {
            let mut row = Row::new(&spikes.columns);
            row.set("sid_idx", 0.0);
            row.set("root", rd.root as f32);
            row.set("local_idx", i as f32);
            if let (Some(maha), Some((f, d))) = (&maha, feats) {
                row.set("f_maha", maha[i]);
                if let Some(sorted) = &maha_sorted {
                    row.set("f_maha_pct", percentile_rank(sorted, maha[i]));
                }
                if let Some((centre, scale)) = &centre_scale {
                    for k in 0..6.min(*d) {
                        let z = (f[i * d + k] - centre[k]) / scale[k];
                        row.set(&format!("f_z_pc{k}"), z.abs());
                    }
                }
            }
            if !amps.is_empty() {
                let a = amps[i];
                row.set(
                    "f_amp_z",
                    if mad_amp.is_finite() && mad_amp > 0.0 {
                        (a - med_amp) / mad_amp
                    } else {
                        f32::NAN
                    },
                );
                row.set("f_amp_pct", percentile_rank(&sorted_amps, a));
                let bin = amp_bins.get(i).copied().unwrap_or(None);
                row.set(
                    "f_amp_z_drift",
                    match bin {
                        Some((bmed, bmad)) if bmad.is_finite() && bmad > 0.0 => (a - bmed) / bmad,
                        _ => f32::NAN,
                    },
                );
            }
            if let Some(k) = &knn {
                row.set("f_knn_density_ratio", k[j]);
            }
            row.set("f_time_frac", times[i].0 as f32 / total.max(1) as f32);
            let prev = if i > 0 {
                Some(times[i].0.saturating_sub(times[i - 1].0))
            } else {
                None
            };
            let next = if i + 1 < n {
                Some(times[i + 1].0.saturating_sub(times[i].0))
            } else {
                None
            };
            let ms = |s: u64| (s as f32) * 1000.0 / sr;
            row.set(
                "f_log_isi_prev",
                prev.map_or(f32::NAN, |s| (1.0 + ms(s)).log10()),
            );
            row.set(
                "f_log_isi_next",
                next.map_or(f32::NAN, |s| (1.0 + ms(s)).log10()),
            );
            let close = |s: Option<u64>| s.is_some_and(|s| s <= refractory);
            row.set_bool("f_refractory", close(prev) || close(next));
            spikes.push_row(&row.values)?;
            spike_rows.push(g[i]);
        }
    }

    phase("spikes");
    Ok(ExtractData {
        tables: SessionTables {
            roots,
            candidates,
            pairs,
            units,
            spikes,
        },
        root_of_spike,
        local_index,
        time_of_spike,
        amp_of_spike,
        root_feats,
        root_center,
        root_sizes: globals.iter().map(Vec::len).collect(),
        root_amp_sum,
        root_amp_count,
        candidate_members,
        spike_rows,
        pair_roots,
        sr,
        total_samples: total,
    })
}

/// Attach supervised targets from `truth`. Call only after [`extract`].
pub fn label(data: &mut ExtractData, truth: &Truth) -> Result<()> {
    let n = data.root_of_spike.len();
    ensure!(
        truth.cluster.len() == n,
        "truth has {} spikes, session {n}",
        truth.cluster.len()
    );
    ensure!(data.tables.roots.n_rows() > 0, "no roots to label");
    let label_of: BTreeMap<u32, PhyLabel> = truth
        .labels
        .iter()
        .map(|&(id, label)| (id, label))
        .collect();
    let label = |u: u32| label_of.get(&u).copied().unwrap_or(PhyLabel::Unsorted);

    // Global unit sizes and a dense local index per truth unit.
    let mut unit_sizes: BTreeMap<u32, usize> = BTreeMap::new();
    for &u in &truth.cluster {
        *unit_sizes.entry(u).or_default() += 1;
    }
    let unit_ids: Vec<u32> = unit_sizes.keys().copied().collect();
    let unit_index: BTreeMap<u32, u32> = unit_ids
        .iter()
        .enumerate()
        .map(|(i, &u)| (u, i as u32))
        .collect();

    // Per-root unit histograms (one pass), reused by roots, pairs and spikes.
    let mut root_units: Vec<BTreeMap<u32, usize>> =
        vec![BTreeMap::new(); data.tables.roots.n_rows()];
    let root_ids: Vec<u32> = data
        .tables
        .roots
        .column_values("root")?
        .into_iter()
        .map(|v| v as u32)
        .collect();
    let mut root_row: BTreeMap<u32, usize> = BTreeMap::new();
    for (i, &r) in root_ids.iter().enumerate() {
        root_row.insert(r, i);
    }
    for g in 0..n {
        let r = data.root_of_spike[g];
        if let Some(&row) = root_row.get(&r) {
            *root_units[row].entry(truth.cluster[g]).or_default() += 1;
        }
    }

    // Root verdicts come from the one classifier the dev DB uses, with the
    // `min_spikes` floor disabled so even tiny roots get a verdict.
    let seq_roots: Vec<u32> = data.root_of_spike.clone();
    let truths = root_truths(
        &seq_roots,
        &Curation {
            clusters: &truth.cluster,
            labels: &truth.labels,
        },
        &SpikeDiffOpts {
            min_spikes: 0,
            ..Default::default()
        },
    )?;
    let by_root: BTreeMap<u32, _> = truths.into_iter().map(|t| (t.root, t)).collect();
    {
        let mut roots = std::mem::replace(
            &mut data.tables.roots,
            Table::new(EXTRACTOR_VERSION, root_columns()),
        );
        for row in 0..roots.n_rows() {
            let r = roots.get(row, "root") as u32;
            let Some(t) = by_root.get(&r) else {
                continue;
            };
            let hits = &root_units[root_row[&r]];
            let best = hits
                .iter()
                .filter(|(u, _)| label(**u) == PhyLabel::Good)
                .map(|(&u, &c)| jaccard(c, t.n_spikes as usize, unit_sizes[&u]))
                .fold(f64::NAN, |acc, v| {
                    if v.is_nan() {
                        acc
                    } else if acc.is_nan() {
                        v
                    } else {
                        acc.max(v)
                    }
                });
            let verdict = match t.verdict {
                "noise" => 0.0,
                "kept" => 1.0,
                "cleanup" => 2.0,
                _ => 3.0,
            };
            set(&mut roots, row, "y_verdict", verdict);
            set(&mut roots, row, "y_n_good", t.n_good as f32);
            set(&mut roots, row, "y_good_share", t.good_share as f32);
            set(&mut roots, row, "y_whole_jaccard", best as f32);
        }
        data.tables.roots = roots;
    }

    // Candidate targets: best Jaccard against a good unit, and against any
    // unit that is not noise.
    {
        let mut table = std::mem::replace(
            &mut data.tables.candidates,
            Table::new(EXTRACTOR_VERSION, candidate_columns()),
        );
        ensure!(
            table.n_rows() == data.candidate_members.len(),
            "candidate side data is out of sync"
        );
        for (row, members) in data.candidate_members.iter().enumerate() {
            let mut hits: BTreeMap<u32, usize> = BTreeMap::new();
            for &g in members {
                *hits.entry(truth.cluster[g as usize]).or_default() += 1;
            }
            let mut best_good = f64::NAN;
            let mut best_any = f64::NAN;
            let mut best_unit = f32::NAN;
            for (&u, &count) in &hits {
                let j = jaccard(count, members.len(), unit_sizes[&u]);
                match label(u) {
                    PhyLabel::Good => {
                        if j.is_finite() && (best_good.is_nan() || j > best_good) {
                            best_good = j;
                            best_unit = u as f32;
                        }
                    }
                    PhyLabel::Noise => {}
                    _ => best_any = max_f64(best_any, j),
                }
            }
            set(&mut table, row, "y_jaccard", best_good as f32);
            set(&mut table, row, "y_jaccard_any", best_any as f32);
            set(
                &mut table,
                row,
                "y_hit",
                if best_good.is_finite() && best_good >= 0.8 {
                    1.0
                } else {
                    0.0
                },
            );
            set(&mut table, row, "y_best_unit", best_unit);
        }
        data.tables.candidates = table;
    }

    // Pair targets.
    {
        let mut table = std::mem::replace(
            &mut data.tables.pairs,
            Table::new(EXTRACTOR_VERSION, pair_columns()),
        );
        for (row, &(a, b)) in data.pair_roots.iter().enumerate() {
            let (Some(&ra), Some(&rb)) = (root_row.get(&a), root_row.get(&b)) else {
                continue;
            };
            let (ha, hb) = (&root_units[ra], &root_units[rb]);
            let (na, nb) = (ha.values().sum::<usize>(), hb.values().sum::<usize>());
            let mut merge = 0.0;
            let mut shared = f64::NAN;
            for (&u, &ca) in ha {
                if label(u) != PhyLabel::Good {
                    continue;
                }
                let cb = hb.get(&u).copied().unwrap_or(0);
                shared = max_f64(shared, jaccard(ca + cb, na + nb, unit_sizes[&u]));
                let size = unit_sizes[&u] as f64;
                if ca as f64 >= 0.2 * na as f64
                    && cb as f64 >= 0.2 * nb as f64
                    && ca as f64 >= 0.2 * size
                    && cb as f64 >= 0.2 * size
                {
                    merge = 1.0;
                }
            }
            set(&mut table, row, "y_merge", merge);
            set(&mut table, row, "y_shared", shared as f32);
        }
        data.tables.pairs = table;
    }

    // Spike targets: label, root-majority good unit, and a dense unit index.
    {
        let mut table = std::mem::replace(
            &mut data.tables.spikes,
            Table::new(EXTRACTOR_VERSION, spike_columns()),
        );
        ensure!(
            table.n_rows() == data.spike_rows.len(),
            "spike side data is out of sync"
        );
        let mut major: BTreeMap<u32, u32> = BTreeMap::new();
        for (&r, units) in &root_row {
            let picked = root_units[*units]
                .iter()
                .filter(|(u, _)| label(**u) == PhyLabel::Good)
                .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)));
            if let Some((&u, _)) = picked {
                major.insert(r, u);
            }
        }
        for row in 0..table.n_rows() {
            let g = data.spike_rows[row] as usize;
            let u = truth.cluster[g];
            let r = data.root_of_spike[g];
            let is_good = label(u) == PhyLabel::Good;
            set(&mut table, row, "y_good", if is_good { 1.0 } else { 0.0 });
            set(
                &mut table,
                row,
                "y_major",
                if major.get(&r) == Some(&u) { 1.0 } else { 0.0 },
            );
            set(&mut table, row, "y_unit", unit_index[&u] as f32);
        }
        data.tables.spikes = table;
    }

    // Unit rows: one per human final cluster, features from its membership.
    {
        let mut members: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for (g, &u) in truth.cluster.iter().enumerate() {
            members.entry(u).or_default().push(g as u32);
        }
        let mut table = Table::new(EXTRACTOR_VERSION, unit_columns());
        table.sids = data.tables.spikes.sids.clone();
        for (&u, globals) in &members {
            let mut row = Row::new(&table.columns);
            row.set("sid_idx", 0.0);
            row.set("unit", u as f32);
            let times: Vec<SampleIndex> = globals
                .iter()
                .map(|&g| data.time_of_spike[g as usize])
                .collect();
            let amps: Vec<f32> = globals
                .iter()
                .map(|&g| data.amp_of_spike[g as usize])
                .collect();
            let amps = if amps.is_empty() || amps.iter().any(|v| !v.is_finite()) {
                Vec::new()
            } else {
                amps
            };
            let stats = unit_stats(&times, &amps, data.sr, data.total_samples);
            row.set("f_n_spikes", stats.n as f32);
            row.set("f_log_n_spikes", (1.0 + stats.n as f32).log10());
            row.set("f_rpc", stats.rp_contamination);
            row.set("f_isi_violation_ratio", stats.isi_violation_ratio);
            row.set("f_presence", stats.presence_ratio);
            row.set("f_amp_cutoff", stats.amplitude_cutoff);
            row.set("f_drift_corr", stats.drift_correlation);
            if !amps.is_empty() {
                row.set("f_mean_amplitude", mean_amplitude(&amps));
            }
            for (i, v) in acg_log_bins(&times, data.sr).into_iter().enumerate() {
                row.set(&format!("f_acg_{i:02}"), v);
            }
            for (i, v) in amp_time_bins(&times, &amps, data.total_samples, AMP_TIME_BINS)
                .into_iter()
                .enumerate()
            {
                row.set(&format!("f_amp_time_{i:02}"), v);
            }
            let roots: BTreeSet<u32> = globals
                .iter()
                .map(|&g| data.root_of_spike[g as usize])
                .collect();
            row.set("f_n_roots", roots.len() as f32);
            let parent_n: usize = roots.iter().map(|&r| data.root_sizes[r as usize]).sum();
            if parent_n > 0 {
                row.set("f_root_share", globals.len() as f32 / parent_n as f32);
            }
            if !amps.is_empty() {
                let (sum, count) = roots.iter().fold((0.0f64, 0usize), |(s, n), &r| {
                    (
                        s + data.root_amp_sum[r as usize],
                        n + data.root_amp_count[r as usize],
                    )
                });
                if count > 0 {
                    row.set(
                        "f_amp_ratio",
                        ratio_of(mean_amplitude(&amps), (sum / count as f64) as f32),
                    );
                }
            }
            let mut d_max = 0usize;
            let mut maha: Vec<f32> = Vec::new();
            for &g in globals {
                let r = data.root_of_spike[g as usize] as usize;
                if let (Some((f, d)), Some((centre, scale))) =
                    (&data.root_feats[r], &data.root_center[r])
                {
                    d_max = d_max.max(*d);
                    let local = data.local_index[g as usize] as usize;
                    if local * d >= f.len() {
                        continue;
                    }
                    let sum: f32 = (0..*d)
                        .map(|j| {
                            let z = (f[local * d + j] - centre[j]) / scale[j];
                            z * z
                        })
                        .sum();
                    maha.push(sum.sqrt());
                }
            }
            if d_max > 0 {
                row.set("f_feature_dim", d_max as f32);
                if !maha.is_empty() {
                    row.set("f_maha_frac3", fraction_above(&maha, 3.0));
                    row.set("f_maha_frac5", fraction_above(&maha, 5.0));
                    row.set("f_maha_median", median(&mut maha.clone()));
                }
            }
            let target = match label(u) {
                PhyLabel::Good => 2.0,
                PhyLabel::Mua => 1.0,
                PhyLabel::Noise | PhyLabel::Unsorted => 0.0,
            };
            row.set("y_label", target);
            table.push_row(&row.values)?;
        }
        data.tables.units = table;
    }

    Ok(())
}

fn set(table: &mut Table, row: usize, name: &str, value: f32) {
    let c = table
        .column(name)
        .unwrap_or_else(|| panic!("table column {name:?} is missing"));
    let stride = table.columns.len();
    table.data[row * stride + c] = value;
}

/// Stable hash of a sorted membership, used to drop duplicate hypotheses.
fn membership_hash(sorted: &[u32]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for i in sorted {
        hasher.update(&i.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

/// Complement of a sorted membership over `0..n`.
fn complement_of(sorted: &[u32], n: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(n - sorted.len());
    let mut it = sorted.iter().peekable();
    for i in 0..n as u32 {
        match it.peek() {
            Some(&&next) if next == i => {
                it.next();
            }
            _ => out.push(i),
        }
    }
    out
}

/// Add a membership and its complement, skipping the empty set, the whole
/// root and anything already seen.
fn push_candidate(
    proposals: &mut Vec<Proposal>,
    seen: &mut BTreeSet<[u8; 32]>,
    n: usize,
    membership: Vec<u32>,
    spec: &Spec,
) {
    let mut sorted = membership;
    sorted.sort_unstable();
    sorted.dedup();
    if sorted.is_empty() || sorted.len() == n {
        return;
    }
    if !seen.insert(membership_hash(&sorted)) {
        return;
    }
    let complement = complement_of(&sorted, n);
    proposals.push((
        spec.method,
        false,
        spec.bic,
        spec.supported,
        spec.depth,
        sorted,
    ));
    if !complement.is_empty() && seen.insert(membership_hash(&complement)) {
        proposals.push((
            spec.method,
            true,
            spec.bic,
            spec.supported,
            spec.depth,
            complement,
        ));
    }
}

fn max_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        b
    } else if b.is_nan() {
        a
    } else {
        a.max(b)
    }
}

fn jaccard(intersection: usize, predicted: usize, truth: usize) -> f64 {
    let union = predicted + truth - intersection;
    if union > 0 {
        intersection as f64 / union as f64
    } else {
        0.0
    }
}

fn ratio_of(a: f32, b: f32) -> f32 {
    if a.is_finite() && b.is_finite() && b.abs() > 0.0 {
        a / b
    } else {
        f32::NAN
    }
}

/// `(cosine, |peak_ch_a - peak_ch_b|, peak-channel distance µm, shared active
/// channels)` of two roots' primary templates; `NaN`/`0` without templates.
fn template_compare<P: DataProvider + HasGeometry>(
    session: &Session<P>,
    a: u32,
    b: u32,
) -> (Option<f32>, f32, f32, usize) {
    let (_, ns, nc) = session.template_shape();
    let wa = session
        .primary_template(ClusterId(a))
        .and_then(|t| session.template_waveform(t));
    let wb = session
        .primary_template(ClusterId(b))
        .and_then(|t| session.template_waveform(t));
    let (Some(wa), Some(wb)) = (wa, wb) else {
        return (None, f32::NAN, f32::NAN, 0);
    };
    let positions = session.provider.channel_positions();
    let Some(cmp) = compare_templates(
        wa,
        wb,
        ns,
        nc,
        (!positions.is_empty()).then_some(positions),
        session.provider.sample_rate().max(1.0),
    ) else {
        return (None, f32::NAN, f32::NAN, 0);
    };
    (
        cmp.cosine.is_finite().then_some(cmp.cosine),
        (cmp.peak_ch.0 as f32 - cmp.peak_ch.1 as f32).abs(),
        cmp.dist_um.unwrap_or(f32::NAN),
        cmp.shared_ch,
    )
}

/// Sliding-refractory contamination (the brief's `rpc`) of the selected
/// root-local indices; `NaN` when no refractory window fits.
fn measure(times: &[SampleIndex], idx: &[u32], sr: f32, total: u64) -> f32 {
    let pick: Vec<SampleIndex> = idx.iter().map(|&i| times[i as usize]).collect();
    let (c, rp) = sliding_refractory_argmin(
        &pick,
        (sr * 0.0005) as u64,
        (sr * 0.010) as u64,
        20,
        total,
        sr,
    );
    if rp > 0 {
        c
    } else {
        f32::NAN
    }
}

/// 25 log-spaced ACG rates (0.5–100 ms) divided by the 50–100 ms band's rate.
///
/// The raw auto-correlogram is two-sided; it is folded into one-sided bins
/// (`|lag|` in 1 ms steps) before integrating the log-spaced bands, so the
/// 50–100 ms reference band really is 50–100 ms and not the near-zero lags.
fn acg_log_bins(times: &[SampleIndex], sr: f32) -> Vec<f32> {
    let mut out = vec![f32::NAN; ACG_BINS];
    if times.len() < 2 || !sr.is_finite() || sr <= 0.0 {
        return out;
    }
    let max_lag = (sr * 0.100).round() as u64;
    let bins = 200usize;
    let h = auto_correlogram(times, max_lag, bins);
    if max_lag == 0 {
        return out;
    }
    // `h[half + j]` counts lag [j, j+1) ms; `h[half - 1 - j]` counts
    // (-(j+1), -j] ms, the same |lag| band by symmetry.
    let half = bins / 2;
    let folded: Vec<f64> = (0..half)
        .map(|j| h[half + j] as f64 + h[half - 1 - j] as f64)
        .collect();
    let bin_ms = 100.0 / half as f64;
    let edges: Vec<f64> = (0..=ACG_BINS)
        .map(|i| 0.5 * 200.0f64.powf(i as f64 / ACG_BINS as f64))
        .collect();
    let rate = |band: usize| -> f64 {
        let (lo, hi) = (edges[band], edges[band + 1]);
        let mut sum = 0.0;
        for (j, &count) in folded.iter().enumerate() {
            let b0 = j as f64 * bin_ms;
            let b1 = b0 + bin_ms;
            let overlap = b1.min(hi) - b0.max(lo);
            if overlap > 0.0 {
                sum += count * overlap / bin_ms;
            }
        }
        // Pairs per second per spike, then per ms of band.
        sum / (hi - lo) * 1000.0 / (times.len() as f64 - 1.0).max(1.0)
    };
    let reference = rate(ACG_BINS - 1);
    if reference <= 0.0 || !reference.is_finite() {
        return out;
    }
    for (band, slot) in out.iter_mut().enumerate() {
        *slot = (rate(band) / reference) as f32;
    }
    out
}
/// Median amplitude in `bins` equal time bins, each divided by the overall
/// median amplitude.
fn amp_time_bins(times: &[SampleIndex], amps: &[f32], total: u64, bins: usize) -> Vec<f32> {
    let mut out = vec![f32::NAN; bins];
    if amps.len() != times.len() || amps.is_empty() || total == 0 || bins == 0 {
        return out;
    }
    let overall = median(&mut amps.to_vec());
    if !overall.is_finite() || overall.abs() <= 0.0 {
        return out;
    }
    let mut buckets: Vec<Vec<f32>> = vec![Vec::new(); bins];
    for (&t, &a) in times.iter().zip(amps) {
        if !a.is_finite() {
            continue;
        }
        let b = ((t.0 as u128 * bins as u128) / total as u128) as usize;
        buckets[b.min(bins - 1)].push(a);
    }
    for (i, b) in buckets.iter_mut().enumerate() {
        if !b.is_empty() {
            out[i] = median(b) / overall;
        }
    }
    out
}

/// `(bin, (median, MAD))` per spike for the amplitude-drift correction.
fn amp_drift_bins(
    times: &[SampleIndex],
    amps: &[f32],
    total: u64,
    bins: usize,
) -> Vec<Option<(f32, f32)>> {
    let mut out = vec![None; times.len()];
    if amps.len() != times.len() || amps.is_empty() || total == 0 || bins == 0 {
        return out;
    }
    let mut buckets: Vec<Vec<f32>> = vec![Vec::new(); bins];
    let mut bin_of = vec![0usize; times.len()];
    for (i, &t) in times.iter().enumerate() {
        if !amps[i].is_finite() {
            continue;
        }
        let b = ((t.0 as u128 * bins as u128) / total as u128) as usize;
        let b = b.min(bins - 1);
        bin_of[i] = b;
        buckets[b].push(amps[i]);
    }
    let stats: Vec<Option<(f32, f32)>> = buckets
        .iter_mut()
        .map(|b| {
            if b.len() < 4 {
                None
            } else {
                Some(robust_center_scale(b))
            }
        })
        .collect();
    for i in 0..times.len() {
        if amps[i].is_finite() {
            out[i] = stats[bin_of[i]];
        }
    }
    out
}

fn stride_rows(feats: &[f32], d: usize, max: usize) -> Vec<f32> {
    let n = feats.len() / d;
    let stride = n.div_ceil(max).max(1);
    let mut out = Vec::with_capacity(n.div_ceil(stride) * d);
    for i in (0..n).step_by(stride) {
        out.extend_from_slice(&feats[i * d..(i + 1) * d]);
    }
    out
}

fn stride_rows_pick(feats: &[f32], d: usize, pick: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(pick.len() * d);
    for &i in pick {
        out.extend_from_slice(&feats[i * d..(i + 1) * d]);
    }
    out
}

/// Per-dimension median and `1.4826 × MAD` (floored at 1e-6).
fn robust_center_scale_dims(feats: &[f32], d: usize) -> (Vec<f32>, Vec<f32>) {
    let n = feats.len() / d;
    let mut centre = vec![0.0f32; d];
    let mut scale = vec![1.0f32; d];
    for j in 0..d {
        let mut col: Vec<f32> = (0..n).map(|i| feats[i * d + j]).collect();
        centre[j] = median(&mut col);
        let dev: Vec<f32> = col.iter().map(|v| (v - centre[j]).abs()).collect();
        let mad = median(&mut dev.clone());
        scale[j] = if mad.is_finite() && mad > 1e-6 {
            1.4826 * mad
        } else {
            1.0
        };
    }
    (centre, scale)
}

/// Robust Mahalanobis distances (diagonal, median/MAD).
fn mahalanobis(feats: &[f32], d: usize) -> Vec<f32> {
    let n = feats.len() / d;
    let (centre, scale) = robust_center_scale_dims(feats, d);
    (0..n)
        .map(|i| {
            let sum: f32 = (0..d)
                .map(|j| {
                    let z = (feats[i * d + j] - centre[j]) / scale[j];
                    z * z
                })
                .sum();
            sum.sqrt()
        })
        .collect()
}

fn robust_center_scale(values: &[f32]) -> (f32, f32) {
    let finite: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return (f32::NAN, f32::NAN);
    }
    let med = median(&mut finite.clone());
    let dev: Vec<f32> = finite.iter().map(|v| (v - med).abs()).collect();
    let mad = median(&mut dev.clone());
    (med, 1.4826 * mad)
}

fn median(values: &mut [f32]) -> f32 {
    let mut x: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if x.is_empty() {
        return f32::NAN;
    }
    x.sort_by(f32::total_cmp);
    x[x.len() / 2]
}

fn fraction_above(values: &[f32], threshold: f32) -> f32 {
    let mut n = 0usize;
    let mut hit = 0usize;
    for &v in values {
        if v.is_finite() {
            n += 1;
            if v > threshold {
                hit += 1;
            }
        }
    }
    if n == 0 {
        f32::NAN
    } else {
        hit as f32 / n as f32
    }
}

fn within_session_percentile(
    qc: &[QcRow],
    roots: &[usize],
    get: impl Fn(&QcRow) -> f32,
    value: f32,
) -> f32 {
    if !value.is_finite() {
        return f32::NAN;
    }
    let mut below = 0usize;
    let mut n = 0usize;
    for &r in roots {
        let v = get(&qc[r]);
        if v.is_finite() {
            n += 1;
            if v <= value {
                below += 1;
            }
        }
    }
    if n == 0 {
        f32::NAN
    } else {
        below as f32 / n as f32
    }
}

fn percentile_rank(sorted: &[f32], value: f32) -> f32 {
    if !value.is_finite() || sorted.is_empty() {
        return f32::NAN;
    }
    let below = sorted.partition_point(|&v| v < value);
    let equal = sorted[below..].partition_point(|&v| v <= value);
    (below as f32 + 0.5 * equal as f32) / sorted.len() as f32
}

/// Mean distance to the `KNN_K` nearest other rows, divided by the median of
/// that distance over the same rows.
fn knn_relative_density(points: &[f32], d: usize) -> Vec<f32> {
    let m = points.len() / d;
    let mut mean_dist = vec![f32::NAN; m];
    if m <= KNN_K || d == 0 {
        return mean_dist;
    }
    let mut scratch = vec![0.0f32; m];
    for i in 0..m {
        let mut len = 0usize;
        for j in 0..m {
            if i == j {
                continue;
            }
            let dist: f32 = (0..d)
                .map(|k| {
                    let a = points[i * d + k] - points[j * d + k];
                    a * a
                })
                .sum::<f32>()
                .sqrt();
            scratch[len] = dist;
            len += 1;
        }
        let (_, _, _) = scratch[..len].select_nth_unstable_by(KNN_K - 1, f32::total_cmp);
        let sum: f32 = scratch[..KNN_K].iter().sum();
        mean_dist[i] = sum / KNN_K as f32;
    }
    let reference = median(&mut mean_dist.clone());
    if !reference.is_finite() || reference <= 0.0 {
        return mean_dist.iter().map(|_| f32::NAN).collect();
    }
    mean_dist.iter().map(|v| v / reference).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acg_and_amplitude_shape_are_defined_and_normalised() {
        // A 5 Hz Poisson-ish train with an artificial 3 ms refractory hole.
        let sr = 30_000.0;
        let mut t = 0u64;
        let mut seed = 7u64;
        let mut times = Vec::new();
        for _ in 0..20_000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            t += 100 + (seed >> 40) % 12_000;
            times.push(SampleIndex(t));
        }
        let acg = acg_log_bins(&times, sr);
        assert_eq!(acg.len(), ACG_BINS);
        assert!(acg.iter().all(|v| v.is_finite() && *v >= 0.0));
        // The last band normalises to 1 (its own rate).
        assert!((acg[ACG_BINS - 1] - 1.0).abs() < 1e-5);
        let amps: Vec<f32> = (0..times.len())
            .map(|i| 1.0 + (i % 10) as f32 * 0.1)
            .collect();
        let bins = amp_time_bins(&times, &amps, t + 1, AMP_TIME_BINS);
        assert_eq!(bins.len(), AMP_TIME_BINS);
        assert!(bins.iter().all(|v| v.is_finite() && *v > 0.0));
        // No amplitudes: everything missing, never a panic.
        assert!(amp_time_bins(&times, &[], t + 1, AMP_TIME_BINS)
            .iter()
            .all(|v| v.is_nan()));
        assert!(acg_log_bins(&[], sr).iter().all(|v| v.is_nan()));
    }

    #[test]
    fn robust_distances_find_a_far_cloud() {
        let mut feats = Vec::new();
        for i in 0..1000 {
            let x = if i % 100 == 0 {
                40.0
            } else {
                (i % 7) as f32 * 0.1 + i as f32 * 0.001
            };
            feats.push(x);
            feats.push((i % 5) as f32 * 0.1 + i as f32 * 0.0007);
        }
        let maha = mahalanobis(&feats, 2);
        assert_eq!(maha.len(), 1000);
        assert!(fraction_above(&maha, 3.0) > 0.0);
        assert!(fraction_above(&maha, 3.0) < 0.05);
        let density = knn_relative_density(&feats, 2);
        assert_eq!(density.len(), 1000);
        assert_eq!(density.iter().filter(|v| v.is_finite()).count(), 1000);
        let rank = percentile_rank(&[1.0, 2.0, 2.0, 4.0], 2.0);
        assert!((rank - 0.5).abs() < 1e-6);
    }
}
