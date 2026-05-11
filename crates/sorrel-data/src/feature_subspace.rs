//! Helpers that pull per-spike PC features into a dense `(n, d)` buffer
//! plus an `is_in_cluster` mask so the isolation metrics in
//! `sorrel-compute::metrics_iso` can run.
//!
//! Why here: extracting the subspace touches `Session` internals (the
//! global per-spike cluster assignment, the PC feature mapping). Keeping it
//! in `sorrel-data` lets `sorrel-compute` stay metrics-only and
//! Session-agnostic.

use crate::session::Session;
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
    Some(PcSubspace {
        features,
        is_in_cluster: is_in,
        d: d_pcs,
    })
}
