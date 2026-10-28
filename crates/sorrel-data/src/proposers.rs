//! Multi-dimensional statistical proposers: a k-means split, a Mahalanobis
//! outlier cut, and the "needs attention" score that ranks the review queue.
//!
//! The proposers are pure functions on a whitened feature matrix
//! (`(n, d)` row-major `f32`, one row per spike of a single cluster, in the
//! cluster's local time order), so tests need no `Session`. The session
//! wiring (`cluster_feature_space`, `kmeans_split`, `outlier_split`) only
//! gathers PC features.

use crate::session::Session;
use crate::suggest::{MergeCandidate, SplitCandidate, SuggestConfig};
use sorrel_compute::linalg::{chi2_sf, covariance, invert, mahalanobis_sq, row_mean};
use sorrel_compute::{
    kmeans, kmeans_assign, lda_d_prime, quality_breakdown, refractory_contamination,
};
use sorrel_io::{ClusterId, DataProvider};

/// Channels (best first) and PCs per channel that make up the feature space.
const FEATURE_CHANNELS: usize = 3;
const FEATURE_PCS: usize = 3;
/// A dimension present for fewer than this fraction of spikes is dropped.
const MIN_PRESENT: f32 = 0.5;
/// k-means is fitted on at most this many spikes; all spikes are then assigned.
const MAX_FIT: usize = 5000;
const MIN_SILHOUETTE: f32 = 0.25;
const MIN_PART_FRAC: f32 = 0.05;
const MIN_PART_SPIKES: usize = 30;
const MIN_D_PRIME: f32 = 2.0;
/// An outlier has `chi2_sf(d², d) < OUTLIER_P`; it is only proposed when
/// the outliers are between these fractions of the cluster.
const OUTLIER_P: f64 = 1e-3;
const OUTLIER_MIN_FRAC: f32 = 0.001;
const OUTLIER_MAX_FRAC: f32 = 0.10;

/// Whiten a raw `(n, d0)` matrix where `NaN` marks a missing value (the
/// spike's template has no feature on that channel). Dimensions present for
/// fewer than half the spikes, or constant, are dropped; remaining missing
/// values become 0, i.e. the dimension's mean after z-scoring. Returns the
/// `(n, d)` matrix and `d`, or `None` when nothing is left.
pub fn whiten(raw: &[f32], d0: usize) -> Option<(Vec<f32>, usize)> {
    if d0 == 0 || raw.len() < 2 * d0 {
        return None;
    }
    let n = raw.len() / d0;
    let mut cols: Vec<(usize, f64, f64)> = Vec::new(); // (dim, mean, std)
    for j in 0..d0 {
        let vals: Vec<f64> = (0..n)
            .map(|i| raw[i * d0 + j] as f64)
            .filter(|v| v.is_finite())
            .collect();
        if (vals.len() as f32) < MIN_PRESENT * n as f32 || vals.len() < 2 {
            continue;
        }
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (vals.len() - 1) as f64;
        if var > 1e-12 {
            cols.push((j, mean, var.sqrt()));
        }
    }
    if cols.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(n * cols.len());
    for i in 0..n {
        for &(j, mean, std) in &cols {
            let v = raw[i * d0 + j] as f64;
            out.push(if v.is_finite() {
                ((v - mean) / std) as f32
            } else {
                0.0
            });
        }
    }
    Some((out, cols.len()))
}

/// Global spike indices of `cluster`, in local order. One pass over every
/// spike; batch callers should build all buckets at once instead.
pub fn cluster_globals<P: DataProvider>(session: &Session<P>, cluster: ClusterId) -> Vec<u32> {
    (session.spike_clusters_global().iter().enumerate())
        .filter(|&(_, &c)| c == cluster)
        .map(|(g, _)| g as u32)
        .collect()
}

/// Whitened PC feature space of one cluster: its 3 best channels × 3 PCs
/// (fewer when the data has fewer). `None` without PC features.
pub fn cluster_feature_space<P: DataProvider>(
    session: &Session<P>,
    cluster: ClusterId,
    globals: &[u32],
) -> Option<(Vec<f32>, usize)> {
    let n_pcs = session.pc_shape().0.min(FEATURE_PCS);
    let chans: Vec<u32> = session
        .best_channels(&[cluster])
        .into_iter()
        .take(FEATURE_CHANNELS)
        .collect();
    if n_pcs == 0 || chans.is_empty() {
        return None;
    }
    let mut raw = Vec::with_capacity(globals.len() * chans.len() * n_pcs);
    for &g in globals {
        for &ch in &chans {
            for pc in 0..n_pcs {
                raw.push(session.pc_feature(g, ch, pc).unwrap_or(f32::NAN));
            }
        }
    }
    whiten(&raw, chans.len() * n_pcs)
}

/// A k-means partition of one cluster.
#[derive(Clone, Debug)]
pub struct KmeansSplit {
    pub k: usize,
    /// Mean simplified silhouette of the chosen partition.
    pub silhouette: f32,
    /// Smallest pairwise LDA d′ between the parts.
    pub d_prime: f32,
    /// Local spike indices (ascending) to move out: every part but the
    /// largest. Feeds `CurationCommand::Split` like the GMM proposal.
    pub minor_idx: Vec<u32>,
}

/// Propose a split of the rows of `feats` into 2 or 3 parts by k-means.
///
/// Accepted only when the best silhouette is at least 0.25, every part holds
/// at least 5% of the spikes and at least 30, every pair of parts has
/// d′ ≥ 2, and the largest part's contamination (`contamination(local
/// indices)`, Hill) is no worse than the whole cluster's.
pub fn kmeans_split_proposal(
    feats: &[f32],
    d: usize,
    contamination: impl Fn(&[u32]) -> f32,
) -> Option<KmeansSplit> {
    if d == 0 {
        return None;
    }
    let n = feats.len() / d;
    if n < 2 * MIN_PART_SPIKES {
        return None;
    }
    let stride = n.div_ceil(MAX_FIT);
    let sub: Vec<f32> = (0..n)
        .step_by(stride)
        .flat_map(|i| feats[i * d..(i + 1) * d].iter().copied())
        .collect();
    let (sil, km) = [2, 3]
        .into_iter()
        .map(|k| {
            let km = kmeans(&sub, d, k, 50);
            (km.silhouette(&sub, d), km)
        })
        .filter(|(s, _)| s.is_finite())
        .max_by(|a, b| a.0.total_cmp(&b.0))?;
    if sil < MIN_SILHOUETTE {
        return None;
    }
    let k = km.centroids.len() / d;

    let mut parts: Vec<Vec<u32>> = vec![Vec::new(); k];
    for (i, l) in kmeans_assign(feats, d, &km.centroids)
        .into_iter()
        .enumerate()
    {
        parts[l as usize].push(i as u32);
    }
    let min_part = MIN_PART_SPIKES.max((MIN_PART_FRAC * n as f32).ceil() as usize);
    if parts.iter().any(|p| p.len() < min_part) {
        return None;
    }

    // Pairwise d′ on the fitting sample.
    let mut d_prime = f32::INFINITY;
    for a in 0..k as u32 {
        for b in (a + 1)..k as u32 {
            let (mut rows, mut mask) = (Vec::new(), Vec::new());
            for (row, &l) in sub.chunks_exact(d).zip(&km.labels) {
                if l == a || l == b {
                    rows.extend_from_slice(row);
                    mask.push(l == a);
                }
            }
            d_prime = d_prime.min(lda_d_prime(&rows, &mask, d));
        }
    }
    if !d_prime.is_finite() || d_prime < MIN_D_PRIME {
        return None;
    }

    let largest = (0..k).max_by_key(|&j| parts[j].len())?;
    let all: Vec<u32> = (0..n as u32).collect();
    if contamination(&parts[largest]) > contamination(&all) {
        return None;
    }
    let mut minor_idx: Vec<u32> = parts
        .iter()
        .enumerate()
        .filter(|&(j, _)| j != largest)
        .flat_map(|(_, p)| p.iter().copied())
        .collect();
    minor_idx.sort_unstable();
    Some(KmeansSplit {
        k,
        silhouette: sil,
        d_prime,
        minor_idx,
    })
}

/// Local indices of rows whose Mahalanobis² distance from the cluster mean
/// has `chi2_sf(d², d) < 1e-3`; `None` unless they are 0.1%–10% of the rows.
///
/// The mean/covariance are refitted once on the non-outliers so a few far
/// points do not inflate the covariance and hide themselves.
pub fn outlier_proposal(feats: &[f32], d: usize) -> Option<Vec<u32>> {
    if d == 0 {
        return None;
    }
    let n = feats.len() / d;
    if n < 50 {
        return None;
    }
    let rows: Vec<f64> = feats.iter().map(|&v| v as f64).collect();
    let fit = |keep: &[bool]| {
        let sel: Vec<f64> = rows
            .chunks_exact(d)
            .zip(keep)
            .filter(|&(_, &k)| k)
            .flat_map(|(r, _)| r.iter().copied())
            .collect();
        let m = sel.len() / d;
        let mu = row_mean(&sel, m, d);
        let mut cov = covariance(&sel, m, d);
        cov.ridge(1e-6);
        invert(&mut cov, 1e-12).map(|()| (mu, cov))
    };
    let outliers = |mu: &[f64], inv: &_| -> Vec<bool> {
        rows.chunks_exact(d)
            .map(|r| chi2_sf(mahalanobis_sq(r, mu, inv), d as f64) < OUTLIER_P)
            .collect()
    };
    let (mu, inv) = fit(&vec![true; n])?;
    let first = outliers(&mu, &inv);
    let keep: Vec<bool> = first.iter().map(|o| !o).collect();
    let flags = match fit(&keep) {
        Some((mu, inv)) => outliers(&mu, &inv),
        None => first,
    };
    let idx: Vec<u32> = (0..n as u32).filter(|&i| flags[i as usize]).collect();
    let frac = idx.len() as f32 / n as f32;
    (OUTLIER_MIN_FRAC..=OUTLIER_MAX_FRAC)
        .contains(&frac)
        .then_some(idx)
}

/// Session wiring: [`kmeans_split_proposal`] on `cluster`'s PC feature space
/// (`globals` = its global spike indices). `None` without PC features.
pub fn kmeans_split<P: DataProvider>(
    session: &Session<P>,
    cluster: ClusterId,
    globals: &[u32],
) -> Option<KmeansSplit> {
    let (feats, d) = cluster_feature_space(session, cluster, globals)?;
    let times = session.spike_times(cluster);
    let sr = session.provider.sample_rate().max(1.0);
    let refractory = (sr * SuggestConfig::default().refractory_seconds).round() as u64;
    let duration = session.provider.n_samples().0;
    kmeans_split_proposal(&feats, d, |idx| {
        let t: Vec<_> = idx.iter().map(|&i| times[i as usize]).collect();
        refractory_contamination(&t, refractory, duration, sr)
    })
}

/// Session wiring: [`outlier_proposal`] on `cluster`'s PC feature space.
pub fn outlier_split<P: DataProvider>(
    session: &Session<P>,
    cluster: ClusterId,
    globals: &[u32],
) -> Option<Vec<u32>> {
    let (feats, d) = cluster_feature_space(session, cluster, globals)?;
    outlier_proposal(&feats, d)
}

/// Global spike indices of every cluster in one pass (batch counterpart of
/// [`cluster_globals`]).
pub fn all_cluster_globals<P: DataProvider>(session: &Session<P>) -> Vec<Vec<u32>> {
    let mut out = vec![Vec::new(); session.n_clusters() as usize];
    for (g, c) in session.spike_clusters_global().iter().enumerate() {
        out[c.idx()].push(g as u32);
    }
    out
}

/// Components at or above this are listed as reasons.
// ponytail: hand-set weights/threshold; learn them from accept/reject
// outcomes once the proposal log has enough of them.
const REASON_MIN: f32 = 0.3;

/// Attention score of `cluster` from precomputed suggestions: the maximum
/// of `1 − quality` (cheap composite, no isolation), the split score, the
/// best merge score, the outlier fraction × 10, and 1.0 for a `review:*`
/// tag; plus the reasons whose component reaches 0.3. `globals` is the
/// cluster's global spike indices.
pub fn attention_with<P: DataProvider>(
    session: &Session<P>,
    cluster: ClusterId,
    globals: &[u32],
    merges: &[MergeCandidate],
    splits: &[SplitCandidate],
) -> (f32, Vec<&'static str>) {
    let sr = session.provider.sample_rate().max(1.0);
    let refractory = (sr * SuggestConfig::default().refractory_seconds).round() as u64;
    let quality = quality_breakdown(
        session.spike_times(cluster),
        session.spike_amplitudes(cluster),
        refractory,
        session.provider.n_samples().0,
        sr,
        50,
    )
    .composite();
    let outlier_frac = outlier_split(session, cluster, globals)
        .map_or(0.0, |o| o.len() as f32 / globals.len().max(1) as f32);
    let comps = [
        ("low_quality", 1.0 - quality),
        (
            "split",
            splits
                .iter()
                .filter(|s| s.cluster == cluster)
                .map(|s| s.score)
                .fold(0.0, f32::max),
        ),
        (
            "merge",
            merges
                .iter()
                .filter(|m| m.a == cluster || m.b == cluster)
                .map(|m| m.score)
                .fold(0.0, f32::max),
        ),
        ("outliers", (outlier_frac * 10.0).min(1.0)),
        (
            "review_tag",
            if session
                .tags(cluster)
                .iter()
                .any(|t| t.starts_with("review:"))
            {
                1.0
            } else {
                0.0
            },
        ),
    ];
    let score = comps
        .iter()
        .map(|c| c.1)
        .filter(|v| v.is_finite())
        .fold(0.0, f32::max)
        .clamp(0.0, 1.0);
    let reasons = comps
        .iter()
        .filter(|c| c.1 >= REASON_MIN)
        .map(|c| c.0)
        .collect();
    (score, reasons)
}

/// Attention for every non-empty cluster, highest first. Pass the
/// suggestions in (compute them once with unbounded `top_*`, see
/// [`SuggestConfig`]) so a caller that already ranked merges/splits does
/// not pay for them again.
pub fn attention_all<P: DataProvider>(
    session: &Session<P>,
    merges: &[MergeCandidate],
    splits: &[SplitCandidate],
) -> Vec<(ClusterId, f32, Vec<&'static str>)> {
    use rayon::prelude::*;
    let buckets = all_cluster_globals(session);
    let mut out: Vec<_> = (0..session.n_clusters())
        .into_par_iter()
        .map(ClusterId)
        .filter(|&c| !session.spike_times(c).is_empty())
        .map(|c| {
            let (s, r) = attention_with(session, c, &buckets[c.idx()], merges, splits);
            (c, s, r)
        })
        .collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaussian(s: &mut u32) -> f32 {
        let mut u = || {
            *s ^= *s << 13;
            *s ^= *s >> 17;
            *s ^= *s << 5;
            (*s as f64 / u32::MAX as f64).clamp(1e-9, 1.0)
        };
        let (u1, u2) = (u(), u());
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }

    /// `n` rows of 9-D unit Gaussian around `centre` (offset on dim 0 and 1).
    pub fn blob(n: usize, centre: f32, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n * 9)
            .map(|i| gaussian(&mut s) + if i % 9 < 2 { centre } else { 0.0 })
            .collect()
    }

    /// Adjusted Rand index of two labelings.
    pub fn ari(a: &[u8], b: &[u8]) -> f64 {
        let c2 = |x: f64| x * (x - 1.0) / 2.0;
        let mut t = [[0.0_f64; 2]; 2];
        for (&x, &y) in a.iter().zip(b) {
            t[x as usize][y as usize] += 1.0;
        }
        let n = a.len() as f64;
        let sum: f64 = t.iter().flatten().map(|&v| c2(v)).sum();
        let ra: f64 = t.iter().map(|r| c2(r[0] + r[1])).sum();
        let rb: f64 = (0..2).map(|j| c2(t[0][j] + t[1][j])).sum();
        let exp = ra * rb / c2(n);
        (sum - exp) / (0.5 * (ra + rb) - exp)
    }

    fn two_units() -> (Vec<f32>, usize, Vec<u8>) {
        let (mut raw, mut truth) = (blob(300, 0.0, 11), vec![0u8; 300]);
        raw.extend(blob(300, 5.0, 29));
        truth.extend(vec![1u8; 300]);
        let (feats, d) = whiten(&raw, 9).unwrap();
        (feats, d, truth)
    }

    #[test]
    fn kmeans_split_recovers_two_units() {
        let (feats, d, truth) = two_units();
        let p = kmeans_split_proposal(&feats, d, |_| 0.0).expect("two units must split");
        let mut pred = vec![0u8; truth.len()];
        p.minor_idx.iter().for_each(|&i| pred[i as usize] = 1);
        let score = ari(&truth, &pred);
        assert!(score > 0.9, "ARI {score}");
    }

    #[test]
    fn kmeans_split_ignores_single_unit() {
        let (feats, d) = whiten(&blob(600, 0.0, 7), 9).unwrap();
        assert!(kmeans_split_proposal(&feats, d, |_| 0.0).is_none());
    }

    #[test]
    fn kmeans_split_vetoed_by_contamination() {
        let (feats, d, _) = two_units();
        // Parent clean, every part dirty.
        let c = |idx: &[u32]| if idx.len() == 600 { 0.0 } else { 0.5 };
        assert!(kmeans_split_proposal(&feats, d, c).is_none());
    }

    #[test]
    fn outliers_recovered() {
        let mut raw = blob(1000, 0.0, 3);
        let mut s = 99_u32;
        // 2% far points, random direction, ~15 sigma out.
        let far: Vec<usize> = (0..1000).step_by(50).collect();
        for &i in &far {
            let dir: Vec<f32> = (0..9).map(|_| gaussian(&mut s)).collect();
            let norm = dir.iter().map(|v| v * v).sum::<f32>().sqrt();
            for j in 0..9 {
                raw[i * 9 + j] = dir[j] / norm * 15.0;
            }
        }
        let (feats, d) = whiten(&raw, 9).unwrap();
        let found = outlier_proposal(&feats, d).expect("outliers expected");
        let hit = far.iter().filter(|&&i| found.contains(&(i as u32))).count();
        assert!(hit == far.len(), "recovered {hit}/{}", far.len());
        assert!(
            found.len() <= far.len() + 5,
            "too many false hits: {}",
            found.len()
        );
    }

    #[test]
    fn whiten_drops_mostly_missing_and_imputes_rest() {
        let mut raw = Vec::new();
        for i in 0..10 {
            // dim 0 varies, dim 1 missing for 8/10, dim 2 missing for 1/10.
            raw.extend([
                i as f32,
                if i < 2 { 1.0 + i as f32 } else { f32::NAN },
                if i == 0 { f32::NAN } else { i as f32 * 2.0 },
            ]);
        }
        let (out, d) = whiten(&raw, 3).unwrap();
        assert_eq!(d, 2);
        assert_eq!(out[1], 0.0, "missing value imputed to the mean");
    }
}
