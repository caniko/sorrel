//! Helpers that pull per-spike PC features into a dense `(n, d)` buffer
//! plus an `is_in_cluster` mask so the isolation metrics in
//! `sorrel-compute::metrics_iso` can run.
//!
//! Why here: extracting the subspace touches `Session` internals (the
//! global per-spike cluster assignment, the PC feature mapping). Keeping it
//! in `sorrel-data` lets `sorrel-compute` stay metrics-only and
//! Session-agnostic.

use crate::cache::pc_subspace::{
    self, CachedPcSubspace, ALGO_VERSION as PC_SUBSPACE_ALGO_VERSION, KIND as PC_SUBSPACE_KIND,
};
use crate::session::Session;
use sorrel_cache::CacheStore;
use sorrel_io::{ClusterId, DataProvider};

/// Default channel index used for the isolation subspace. PC features are
/// stored per *template-relative* channel — Kilosort orders channels by
/// proximity to the template peak, so index 0 is usually the most
/// informative single channel.
pub const DEFAULT_CHANNEL_IDX: usize = 0;

/// Default number of PCs to use for isolation metrics.
pub const DEFAULT_D: usize = 3;

/// Default cap on the number of background spikes considered. Brute-force
/// k-NN cost is O(n_in × n_total × d), so we sub-sample the background to
/// keep it bounded for huge recordings.
pub const DEFAULT_MAX_BACKGROUND: usize = 5_000;

/// Result of collecting a cluster's PC subspace plus its background.
pub struct PcSubspace {
    /// Row-major `(n_total, d)` flat features. Cluster spikes precede
    /// background spikes — but the [`Self::is_in_cluster`] mask is the
    /// authoritative indicator either way.
    pub features: Vec<f32>,
    /// One bool per row of `features`.
    pub is_in_cluster: Vec<bool>,
    /// Dimensionality of the feature subspace.
    pub d: usize,
}

/// Two row-major feature matrices whose columns share the same physical
/// channel and PC, even when the spikes belong to different templates.
pub struct PairwisePhysicalSubspace {
    pub feats_a: Vec<f32>,
    pub feats_b: Vec<f32>,
    pub d: usize,
    pub channels: Vec<u32>,
}

pub fn collect_pairwise_physical_subspace<P: DataProvider>(
    session: &Session<P>,
    a: ClusterId,
    b: ClusterId,
    max_channels: usize,
    d_pcs: usize,
) -> Option<PairwisePhysicalSubspace> {
    if !session.has_pc_features() || max_channels == 0 {
        return None;
    }
    let use_pcs = d_pcs.min(session.pc_shape().0);
    if use_pcs == 0 {
        return None;
    }
    // ponytail: deterministic stride caps work at 5,000 samples per part;
    // rare template memberships may be missed; stratify if that matters.
    let sample = |c| {
        let indices = session.cluster_index().global_indices(c);
        (0..indices.len().min(5_000)).map(move |i| {
            indices[(i as u64 * indices.len() as u64 / indices.len().min(5_000) as u64) as usize]
        })
    };
    let has_channel = |c, ch| {
        sample(c)
            .filter(|&g| session.pc_feature(g, ch, 0).is_some())
            .nth(1)
            .is_some()
    };
    let mut channels = Vec::new();
    for ch in session.best_channels(&[a, b]) {
        if !channels.contains(&ch) && has_channel(a, ch) && has_channel(b, ch) {
            channels.push(ch);
            if channels.len() == max_channels {
                break;
            }
        }
    }
    let d = channels.len() * use_pcs;
    if d == 0 {
        return None;
    }
    let collect = |c| {
        let mut features = Vec::new();
        for g in sample(c) {
            let start = features.len();
            let values = channels
                .iter()
                .flat_map(|&ch| (0..use_pcs).map(move |pc| session.pc_feature(g, ch, pc)));
            for value in values {
                let Some(v) = value.filter(|v| v.is_finite()) else {
                    features.truncate(start);
                    break;
                };
                features.push(v);
            }
        }
        features
    };
    let (feats_a, feats_b) = (collect(a), collect(b));
    (feats_a.len() / d >= 2 && feats_b.len() / d >= 2).then_some(PairwisePhysicalSubspace {
        feats_a,
        feats_b,
        d,
        channels,
    })
}

pub fn pairwise_cluster_separation<P: DataProvider>(
    session: &Session<P>,
    a: ClusterId,
    b: ClusterId,
) -> Option<sorrel_compute::PairwiseSeparation> {
    let sub = collect_pairwise_physical_subspace(session, a, b, 4, 3)?;
    Some(sorrel_compute::pairwise_separation(
        &sub.feats_a,
        &sub.feats_b,
        sub.d,
    ))
}

/// Pull a `(d_pcs, channel_idx)` subspace of every spike's PC features
/// into a dense buffer, marking which rows belong to `cluster`. Background
/// spikes are taken from all *other* clusters and sub-sampled at
/// `max_background`.
///
/// Returns `None` when the session has no PC features seeded, when
/// `channel_idx` is out of range, or when there are fewer than 2 spikes.
pub fn collect_pc_subspace<P: DataProvider>(
    session: &Session<P>,
    cluster: ClusterId,
    d_pcs: usize,
    channel_idx: usize,
    max_background: usize,
) -> Option<PcSubspace> {
    let (n_pcs, n_chans) = session.pc_shape();
    if n_pcs == 0 || n_chans == 0 || d_pcs == 0 || d_pcs > n_pcs || channel_idx >= n_chans {
        return None;
    }
    let stride = n_pcs * n_chans;
    let spike_clusters = session.spike_clusters_global();
    if spike_clusters.is_empty() {
        return None;
    }
    let (cluster_spike_count, cluster_spike_fingerprint) =
        cluster_spike_fingerprint(session, cluster);
    let cache_key = pc_subspace::key(pc_subspace::KeyParts {
        session_identity: &session.provider.identity_bytes(),
        n_spikes_total: session.spike_clusters_global().len() as u64,
        cluster,
        cluster_spike_count,
        cluster_spike_fingerprint,
        d_pcs,
        channel_idx,
        max_background,
    });
    if let Some(dataset_dir) = session.cache_dataset_dir() {
        if let Ok(store) = CacheStore::open(dataset_dir) {
            match store.get::<CachedPcSubspace>(&cache_key) {
                Ok(Some(cached)) if cached.algo_version.to_native() == PC_SUBSPACE_ALGO_VERSION => {
                    log::debug!(
                        "cache hit {}/{}",
                        PC_SUBSPACE_KIND,
                        cache_key.fingerprint.hex()
                    );
                    return Some(PcSubspace {
                        features: cached
                            .features
                            .as_slice()
                            .iter()
                            .map(|value| value.to_native())
                            .collect(),
                        is_in_cluster: cached.is_in_cluster.as_slice().to_vec(),
                        d: cached.d.to_native() as usize,
                    });
                }
                Ok(Some(_)) | Ok(None) => {
                    log::debug!(
                        "cache miss {}/{}",
                        PC_SUBSPACE_KIND,
                        cache_key.fingerprint.hex()
                    );
                }
                Err(err) => {
                    log::debug!(
                        "cache miss {}/{} after read error: {err}",
                        PC_SUBSPACE_KIND,
                        cache_key.fingerprint.hex()
                    );
                }
            }
        } else {
            log::debug!(
                "cache miss {}/{} after open error",
                PC_SUBSPACE_KIND,
                cache_key.fingerprint.hex()
            );
        }
    }

    // First pass: count cluster + non-cluster sizes so we can allocate once.
    let mut n_in = 0usize;
    let mut n_out = 0usize;
    for &c in spike_clusters {
        if c == cluster {
            n_in += 1;
        } else {
            n_out += 1;
        }
    }
    if n_in == 0 || n_out == 0 {
        return None;
    }
    let bg_keep = n_out.min(max_background);
    let bg_stride = (n_out / bg_keep).max(1);

    let mut features: Vec<f32> = Vec::with_capacity((n_in + bg_keep) * d_pcs);
    let mut is_in: Vec<bool> = Vec::with_capacity(n_in + bg_keep);

    // Cluster rows first.
    for (g, &c) in spike_clusters.iter().enumerate() {
        if c != cluster {
            continue;
        }
        let Some(feats) = session.pc_feature_for(g as u32) else {
            continue;
        };
        if feats.len() < stride {
            continue;
        }
        for pc in 0..d_pcs {
            features.push(feats[pc * n_chans + channel_idx]);
        }
        is_in.push(true);
    }
    // Background rows, sub-sampled.
    let mut bg_seen = 0usize;
    for (g, &c) in spike_clusters.iter().enumerate() {
        if c == cluster {
            continue;
        }
        let take = bg_seen.checked_rem(bg_stride) == Some(0);
        bg_seen += 1;
        if !take {
            continue;
        }
        let Some(feats) = session.pc_feature_for(g as u32) else {
            continue;
        };
        if feats.len() < stride {
            continue;
        }
        for pc in 0..d_pcs {
            features.push(feats[pc * n_chans + channel_idx]);
        }
        is_in.push(false);
    }
    if features.is_empty() {
        return None;
    }
    let subspace = PcSubspace {
        features,
        is_in_cluster: is_in,
        d: d_pcs,
    };
    if let Some(dataset_dir) = session.cache_dataset_dir() {
        match CacheStore::open(dataset_dir) {
            Ok(store) => {
                let value = CachedPcSubspace {
                    features: subspace.features.clone(),
                    is_in_cluster: subspace.is_in_cluster.clone(),
                    d: subspace.d as u32,
                    algo_version: PC_SUBSPACE_ALGO_VERSION,
                };
                if let Err(err) = store.put(&cache_key, &value) {
                    log::debug!(
                        "failed to write cache {}/{}: {err}",
                        PC_SUBSPACE_KIND,
                        cache_key.fingerprint.hex()
                    );
                }
            }
            Err(err) => {
                log::debug!(
                    "failed to open cache {}/{} for write: {err}",
                    PC_SUBSPACE_KIND,
                    cache_key.fingerprint.hex()
                );
            }
        }
    }
    Some(subspace)
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{journal::Journal, qc_export::tests::MockProvider};
    use sorrel_io::{HasPcFeatures, HasSpikeTemplates, HasTemplateWaveforms, SampleIndex};

    impl HasSpikeTemplates for MockProvider {
        fn spike_templates(&self, c: ClusterId) -> &[u32] {
            [&[0, 0, 0][..], &[1, 1, 1][..]]
                .get(c.idx())
                .copied()
                .unwrap_or(&[])
        }
    }

    impl HasPcFeatures for MockProvider {
        fn pc_features(&self) -> &[f32] {
            &[
                90.0, 1.0, 91.0, 2.0, 92.0, 3.0, 7.0, 50.0, 8.0, 51.0, 9.0, 52.0,
            ]
        }
        fn pc_shape(&self) -> (usize, usize) {
            (1, 2)
        }
        fn pc_feature_ind(&self) -> &[u32] {
            &[0, 1, 1, 2]
        }
        fn spike_pc_indices(&self, c: ClusterId) -> &[u32] {
            [&[0, 1, 2][..], &[3, 4, 5][..]]
                .get(c.idx())
                .copied()
                .unwrap_or(&[])
        }
    }

    impl HasTemplateWaveforms for MockProvider {
        fn template_waveforms(&self) -> &[f32] {
            &[0.0, 0.0, 0.0, -1.0, -2.0, -3.0, 1.0, 2.0, 3.0]
        }
        fn template_shape(&self) -> (usize, usize, usize) {
            (1, 3, 3)
        }
        fn similar_templates(&self) -> &[f32] {
            &[]
        }
    }

    pub(crate) fn physical_test_session(dir: &std::path::Path) -> Session<MockProvider> {
        let provider = MockProvider {
            spikes: vec![
                vec![SampleIndex(10), SampleIndex(12), SampleIndex(150)],
                vec![SampleIndex(20), SampleIndex(60), SampleIndex(200)],
            ],
            amps: vec![vec![1.0, 2.0, 3.0], vec![f32::NAN; 3]],
            n_samples: SampleIndex(1000),
        };
        let mut session = Session::new(
            provider,
            Journal::open(&dir.join("physical.journal")).unwrap(),
        );
        session.seed_templates();
        session.seed_amplitudes();
        session
    }

    #[test]
    fn pairwise_aligns_the_same_physical_channel_across_templates() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = physical_test_session(dir.path());
        s.seed_pc_features();
        let sub = collect_pairwise_physical_subspace(&s, ClusterId(0), ClusterId(1), 4, 3).unwrap();
        assert_eq!(sub.channels, vec![1]);
        assert_eq!(sub.d, 1);
        assert_eq!(sub.feats_a, vec![1.0, 2.0, 3.0]);
        assert_eq!(sub.feats_b, vec![7.0, 8.0, 9.0]);
    }
}
