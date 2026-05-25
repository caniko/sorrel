//! `Session`-level quality computation that adds PC-feature isolation
//! metrics on top of the [`sorrel_compute::quality_breakdown`] baseline.
//! Lives here (not in `sorrel-compute`) because pulling the per-spike PC
//! subspace requires the live `Session` state.

use crate::cache::{
    isolation::{self, archived_to_metrics, CachedIsolationMetrics, KIND as ISOLATION_KIND},
    pc_subspace,
};
use crate::feature_subspace::{
    collect_pc_subspace, DEFAULT_CHANNEL_IDX, DEFAULT_D, DEFAULT_MAX_BACKGROUND,
};
use crate::session::Session;
use sorrel_cache::{CacheKey, CacheStore};
use sorrel_compute::{isolation_metrics, quality_breakdown, IsolationMetrics, QualityBreakdown};
use sorrel_io::{ClusterId, DataProvider};

/// Refractory window used by the quality computation: 1.5 ms. Matches the
/// suggester default.
const REFRACTORY_SECONDS: f32 = 0.0015;
pub const DEFAULT_K_NN: usize = 8;

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
    let cache_key = isolation_cache_key(
        session,
        cluster,
        DEFAULT_D,
        DEFAULT_CHANNEL_IDX,
        DEFAULT_MAX_BACKGROUND,
        DEFAULT_K_NN,
    );
    if let (Some(dataset_dir), Some(cache_key)) = (session.cache_dataset_dir(), cache_key.as_ref())
    {
        match CacheStore::open(dataset_dir) {
            Ok(store) => {
                match store.get::<CachedIsolationMetrics>(cache_key) {
                    Ok(Some(cached)) => {
                        if let Some(metrics) = archived_to_metrics(&cached) {
                            log::debug!(
                                "cache hit {}/{}",
                                ISOLATION_KIND,
                                cache_key.fingerprint.hex()
                            );
                            return Some(metrics);
                        }
                        log::debug!(
                            "cache miss {}/{} after algo version mismatch",
                            ISOLATION_KIND,
                            cache_key.fingerprint.hex()
                        );
                    }
                    Ok(None) => {
                        log::debug!(
                            "cache miss {}/{}",
                            ISOLATION_KIND,
                            cache_key.fingerprint.hex()
                        );
                    }
                    Err(err) => {
                        log::debug!(
                            "cache miss {}/{} after read error: {err}",
                            ISOLATION_KIND,
                            cache_key.fingerprint.hex()
                        );
                    }
                }

                let metrics = compute_isolation_uncached(session, cluster)?;
                let value = CachedIsolationMetrics::from_metrics(metrics);
                if let Err(err) = store.put(cache_key, &value) {
                    log::debug!(
                        "failed to write cache {}/{}: {err}",
                        ISOLATION_KIND,
                        cache_key.fingerprint.hex()
                    );
                }
                return Some(metrics);
            }
            Err(err) => {
                log::debug!(
                    "cache miss {}/{} after open error: {err}",
                    ISOLATION_KIND,
                    cache_key.fingerprint.hex()
                );
            }
        }
    }

    compute_isolation_uncached(session, cluster)
}

fn compute_isolation_uncached<P: DataProvider>(
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
    let m = isolation_metrics(&sub.features, &sub.is_in_cluster, sub.d, DEFAULT_K_NN);
    Some(m)
}

fn isolation_cache_key<P: DataProvider>(
    session: &Session<P>,
    cluster: ClusterId,
    d_pcs: usize,
    channel_idx: usize,
    max_background: usize,
    k_nn: usize,
) -> Option<CacheKey> {
    let (n_pcs, n_chans) = session.pc_shape();
    if n_pcs == 0 || n_chans == 0 || d_pcs == 0 || d_pcs > n_pcs || channel_idx >= n_chans {
        return None;
    }
    let (cluster_spike_count, cluster_spike_fingerprint) =
        cluster_spike_fingerprint(session, cluster);
    let subspace_key = pc_subspace::key(
        &session.provider.identity_bytes(),
        session.journal_head(),
        cluster,
        cluster_spike_count,
        cluster_spike_fingerprint,
        d_pcs,
        channel_idx,
        max_background,
    );

    Some(isolation::key(&subspace_key, k_nn))
}

fn cluster_spike_fingerprint<P: DataProvider>(
    session: &Session<P>,
    cluster: ClusterId,
) -> (usize, [u8; 32]) {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"sorrel-pc-subspace-cluster-spikes-v1");
    let mut count = 0usize;
    for (global_idx, &assigned) in session.spike_clusters_global().iter().enumerate() {
        if assigned == cluster {
            count += 1;
            hasher.update(&(global_idx as u64).to_le_bytes());
        }
    }
    (count, *hasher.finalize().as_bytes())
}
