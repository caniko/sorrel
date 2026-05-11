//! `Session`-level quality computation that adds PC-feature isolation
//! metrics on top of the [`sorrel_compute::quality_breakdown`] baseline.
//! Lives here (not in `sorrel-compute`) because pulling the per-spike PC
//! subspace requires the live `Session` state.

use crate::feature_subspace::{
    collect_pc_subspace, DEFAULT_CHANNEL_IDX, DEFAULT_D, DEFAULT_MAX_BACKGROUND,
};
use crate::session::Session;
use sorrel_compute::{isolation_metrics, quality_breakdown, IsolationMetrics, QualityBreakdown};
use sorrel_io::{ClusterId, DataProvider};

/// Refractory window used by the quality computation: 1.5 ms. Matches the
/// suggester default.
const REFRACTORY_SECONDS: f32 = 0.0015;

/// Build a full per-cluster quality breakdown including PC-feature
/// isolation when the session has features seeded. Cheap O(n_spikes) plus
/// a brute-force k-NN bounded by [`DEFAULT_MAX_BACKGROUND`].
pub fn cluster_quality<P: DataProvider>(
    session: &Session<P>,
    cluster: ClusterId,
) -> QualityBreakdown {
    let sr = session.provider.sample_rate().max(1.0);
    let refractory_samples = (sr * REFRACTORY_SECONDS).round() as u64;
    let total_duration = session.provider.n_samples();
    let mut q = quality_breakdown(
        session.spike_times(cluster),
        session.spike_amplitudes(cluster),
        refractory_samples,
        total_duration.0,
        sr,
        50,
    );
    q.isolation = compute_isolation(session, cluster);
    q
}

/// Compute the PC-feature isolation metrics for one cluster, if the
/// session has features seeded. Returns `None` otherwise.
pub fn compute_isolation<P: DataProvider>(
    session: &Session<P>,
    cluster: ClusterId,
) -> Option<IsolationMetrics> {
    let sub = collect_pc_subspace(
        session,
        cluster,
        DEFAULT_D,
        DEFAULT_CHANNEL_IDX,
        DEFAULT_MAX_BACKGROUND,
    )?;
    let m = isolation_metrics(&sub.features, &sub.is_in_cluster, sub.d, 8);
    Some(m)
}
