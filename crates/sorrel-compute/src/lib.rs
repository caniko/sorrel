//! sorrel-compute: generic, vectorisable kernels.
//!
//! Functions are parameterised over numeric types so the compiler can emit
//! SIMD-friendly inner loops per call site.

pub mod binning;
pub mod ccg_analysis;
pub mod cmr;
pub mod distribution;
pub mod drift;
pub mod filter;
pub mod gmm;
pub mod linalg;
pub mod lttb;
pub mod metrics;
pub mod metrics_iso;
pub mod quality;
pub mod snippets;

pub use binning::{histogram_u64, isi_histogram};
pub use ccg_analysis::{analyse_refractory_dip, refractory_dip_score, CcgRefractoryAnalysis};
pub use cmr::subtract_channel_median;
pub use distribution::{
    amplitude_cutoff, bimodality_coefficient, excess_kurtosis, ks_pvalue, ks_two_sample,
    mean_f64, mean_std, percentile, skewness, Moments,
};
pub use drift::{
    amplitude_drift_correlation, amplitude_drift_slope, longest_silent_gap_frac, presence_cv,
    sliding_refractory_contamination,
};
pub use gmm::{gmm_split_proposal, GmmSplitProposal};
pub use filter::{Biquad, BiquadState};
pub use lttb::lttb_downsample;
pub use metrics::{
    amplitude_snr, auto_correlogram, cross_correlogram, fraction_below, isi_violation_rate,
    isi_violations, mean_amplitude, presence_ratio, refractory_contamination, std_amplitude,
};
pub use metrics_iso::{
    isolation_metrics, nn_isolation_metric, IsolationMetrics, MIN_CLUSTER_SPIKES,
};
pub use quality::{quality_breakdown, QualityBreakdown};
pub use snippets::{extract_snippets_single_channel, mean_snippet};
