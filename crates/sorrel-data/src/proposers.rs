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
    fit_diag_gmm, isi_violations, kmeans, kmeans_assign, lda_d_prime, quality_breakdown,
    refractory_contamination, sliding_refractory_argmin,
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
/// Core proposer: BIC picks among k = 1..=CORE_MAX_K diagonal GMM components
/// fitted on at most CORE_MAX_FIT rows; a core holds at least
/// max(CORE_MIN_SPIKES, CORE_MIN_FRAC) of the spikes. CORE_RPV is the brief's
/// RPV flag threshold: the whole must exceed it, the core must not.
const CORE_MAX_K: usize = 8;
const CORE_MAX_FIT: usize = 20_000;
const CORE_MIN_SPIKES: usize = 100;
const CORE_MIN_FRAC: f32 = 0.01;
const CORE_RPV: f32 = 0.1;
/// The core's feature space spans this many best channels (a tetrode fully).
const CORE_CHANNELS: usize = 4;
/// Interior candidates are bounded, model-defined ellipsoids, never a scan of
/// arbitrary spike subsets. Keep the largest passing interior per component.
const CORE_POSTERIOR: f64 = 0.95;
const CORE_DENSITY_MASS: [f64; 7] = [1.0, 0.99, 0.95, 0.90, 0.75, 0.50, 0.25];

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
    feature_space(session, cluster, globals, FEATURE_CHANNELS)
}

fn feature_space<P: DataProvider>(
    session: &Session<P>,
    cluster: ClusterId,
    globals: &[u32],
    n_chans: usize,
) -> Option<(Vec<f32>, usize)> {
    let n_pcs = session.pc_shape().0.min(FEATURE_PCS);
    let chans: Vec<u32> = session
        .best_channels(&[cluster])
        .into_iter()
        .take(n_chans)
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

/// A clean part of a contaminated cluster: a small `core` carved out of it, or
/// the majority left once a `cleanup` of contaminated components is dropped.
#[derive(Clone, Debug)]
pub struct CoreSplit {
    /// `"core"` (minor part = the clean core) or `"cleanup"` (minor part = the
    /// dropped contaminated/odd components; the kept majority is the clean one).
    pub method: &'static str,
    /// Components of the BIC-chosen mixture.
    pub k: usize,
    /// LDA d′ of the minor part against the rest.
    pub d_prime: f32,
    /// Sliding-refractory contamination of the clean (kept) part and of the
    /// whole cluster.
    pub core_contamination: f32,
    pub whole_contamination: f32,
    /// Suggestion score in [0.5, 1]: separation `min(d′/4, 1)` and cleanliness gain, a quarter each.
    pub score: f32,
    /// Local spike indices (ascending) of the minor part.
    pub minor_idx: Vec<u32>,
}

/// The cheap gates of [`core_split_proposal`]: at least `2 * CORE_MIN_SPIKES`
/// spikes and a whole-cluster contamination (`whole`, evaluated only after the
/// size gate) above `CORE_RPV`. Returns that contamination.
fn core_gate(n: usize, whole: impl FnOnce() -> f32) -> Option<f32> {
    if n < 2 * CORE_MIN_SPIKES {
        return None;
    }
    let whole = whole();
    (!whole.is_nan() && whole > CORE_RPV).then_some(whole)
}

/// Propose a clean part of a contaminated cluster, as a core or a cleanup.
///
/// Only runs when the whole cluster fails the RPV flag (`contamination(all) >
/// 0.1`). Fits diagonal GMMs with k = 1..=8 on (a ≤ 20k-row subsample of)
/// `feats` and keeps the BIC winner (k = 1 means no structure). Two readings
/// of that mixture are tried; both need the minor part to hold at least
/// max(100, 1%) of the spikes and, for a core, an LDA d′ against the rest of at least 2.
/// - `core`: one component (or a high-posterior dense interior when its full
///   component fails) of at most half the spikes with contamination at
///   most 0.1 and significantly fewer ISI violations than a random subset of
///   that size would have (`violations(local indices)` below the chance count
///   minus 2.33 Poisson sd, none at all when that is below zero). The kept
///   part is the core, the minor part too.
/// - `cleanup`: the human kept the majority and dropped a contaminated or odd
///   minority (10-30% in practice). Components holding ISI violations are
///   dropped, most violations per spike first, until the remainder (at least
///   two thirds of the spikes) has
///   contamination at most 0.1 and, as for a core, significantly fewer ISI
///   violations than chance. The minor part is the dropped components, at
///   least max(100, 1%) of the spikes. It has no d′ floor: its dropped tail
///   overlaps the kept unit (that is why it contaminates it), so the ISI gain
///   is the evidence and d′ only weighs the rank and score.
///
/// Ranking: the reading that keeps more clean spikes wins, weighted by
/// separation and cleanliness gain: `kept/n × min(d′/4, 1) × gain`, with
/// `gain = (whole − kept contamination) / whole`. A cleanup keeps far more
/// spikes than a core, so it wins whenever both exist unless its cut is very
/// leaky (d′ near 0). Ties go to the core. A second core comes from
/// re-running on the remainder after the first split.
pub fn core_split_proposal(
    feats: &[f32],
    d: usize,
    contamination: impl Fn(&[u32]) -> f32,
    violations: impl Fn(&[u32]) -> usize,
) -> Option<CoreSplit> {
    if d == 0 {
        return None;
    }
    let n = feats.len() / d;
    let whole = core_gate(n, || contamination(&(0..n as u32).collect::<Vec<_>>()))?;
    core_from_mixture(feats, d, whole, contamination, violations)
}

/// [`core_split_proposal`] past its gates, with the whole-cluster
/// contamination `whole` already computed (it is a full refractory sweep).
fn core_from_mixture(
    feats: &[f32],
    d: usize,
    whole: f32,
    contamination: impl Fn(&[u32]) -> f32,
    violations: impl Fn(&[u32]) -> usize,
) -> Option<CoreSplit> {
    let n = feats.len() / d;
    let stride = n.div_ceil(CORE_MAX_FIT);
    let sub: Vec<f32> = (0..n)
        .step_by(stride)
        .flat_map(|i| feats[i * d..(i + 1) * d].iter().copied())
        .collect();
    let gmm = (1..=CORE_MAX_K)
        .filter_map(|k| fit_diag_gmm(&sub, d, k, 60))
        .min_by(|a, b| a.bic().total_cmp(&b.bic()))?;
    core_from_fitted(feats, d, whole, &gmm, contamination, violations)
}

fn core_from_fitted(
    feats: &[f32],
    d: usize,
    whole: f32,
    gmm: &sorrel_compute::DiagGmm,
    contamination: impl Fn(&[u32]) -> f32,
    violations: impl Fn(&[u32]) -> usize,
) -> Option<CoreSplit> {
    if gmm.k < 2 {
        return None;
    }
    let n = feats.len() / d;
    let all: Vec<u32> = (0..n as u32).collect();
    let stride = n.div_ceil(CORE_MAX_FIT);
    let sub: Vec<f32> = (0..n)
        .step_by(stride)
        .flat_map(|i| feats[i * d..(i + 1) * d].iter().copied())
        .collect();
    let details = gmm.assign_with_confidence(feats);
    let labels: Vec<_> = details.iter().map(|x| x.0).collect();
    let mut parts: Vec<Vec<u32>> = vec![Vec::new(); gmm.k];
    for (i, &l) in labels.iter().enumerate() {
        parts[l as usize].push(i as u32);
    }
    let min_size = CORE_MIN_SPIKES.max((CORE_MIN_FRAC * n as f32).ceil() as usize);
    let viol_all = violations(&all) as f64;
    // LDA d′ of the rows labelled by `minor` (component set) against the rest.
    let d_prime = |minor: &[usize]| {
        let (mut rows, mut mask) = (Vec::new(), Vec::new());
        for (row, &l) in sub.chunks_exact(d).zip(labels.iter().step_by(stride)) {
            rows.extend_from_slice(row);
            mask.push(minor.contains(&(l as usize)));
        }
        lda_d_prime(&rows, &mask, d)
    };
    // (rank, candidate)
    let mut best: Option<(f32, CoreSplit)> = None;
    let mut offer = |method: &'static str, minor: Vec<u32>, kept_n: usize, c: f32, dp: f32| {
        let gain = ((whole - c) / whole).clamp(0.0, 1.0);
        let sep = (dp / 4.0).clamp(0.0, 1.0);
        let rank = kept_n as f32 / n as f32 * (dp / 4.0).clamp(0.0, 1.0) * gain;
        if best.as_ref().is_none_or(|(r, _)| rank > *r) {
            best = Some((
                rank,
                CoreSplit {
                    method,
                    k: gmm.k,
                    d_prime: dp,
                    core_contamination: c,
                    whole_contamination: whole,
                    score: 0.5 + 0.25 * sep + 0.25 * gain,
                    minor_idx: minor,
                },
            ));
        }
    };
    // Significantly fewer violations than a random subset would hold: below
    // the chance count minus 2.33 Poisson sd (none at all when that is < 0).
    let fewer_than_chance =
        |v: usize, expected: f64| v as f64 <= (expected - 2.33 * expected.sqrt()).max(0.0);
    let chance = |len: usize| viol_all * (len as f64 / n as f64).powi(2);
    for (j, p) in parts.iter().enumerate() {
        if p.len() < min_size {
            continue;
        }
        if p.len() * 2 <= n {
            let c = contamination(p);
            let dp = d_prime(&[j]);
            if c.is_finite()
                && c <= CORE_RPV
                && fewer_than_chance(violations(p), chance(p.len()))
                && dp.is_finite()
                && dp >= MIN_D_PRIME
            {
                offer("core", p.clone(), p.len(), c, dp);
                // Preserve a supported full component; don't trim a clean unit
                // merely to improve its apparent refractory score.
                continue;
            }
        }
        let mut previous_size = p.len();
        for mass in CORE_DENSITY_MASS {
            let interior: Vec<u32> = p
                .iter()
                .copied()
                .filter(|&i| {
                    let (_, posterior, radius_sq) = details[i as usize];
                    posterior >= CORE_POSTERIOR
                        && radius_sq.is_finite()
                        && chi2_sf(radius_sq, d as f64) >= 1.0 - mass
                })
                .collect();
            if interior.len() < min_size {
                break;
            }
            if interior.len() == previous_size {
                continue;
            }
            previous_size = interior.len();
            if interior.len() * 2 > n {
                continue;
            }
            let c = contamination(&interior);
            if !c.is_finite()
                || c > CORE_RPV
                || !fewer_than_chance(violations(&interior), chance(interior.len()))
            {
                continue;
            }
            let mask: Vec<_> = (0..n as u32)
                .step_by(stride)
                .map(|i| interior.binary_search(&i).is_ok())
                .collect();
            let dp = lda_d_prime(&sub, &mask, d);
            if dp.is_finite() && dp >= MIN_D_PRIME {
                offer("core", interior.clone(), interior.len(), c, dp);
                break;
            }
        }
    }
    // Cleanup: drop components with ISI violations, most violations per spike
    // first, until the remainder passes. The BIC mixture often fragments a
    // contaminated tail over many components, so this is a prefix, not ≤ 2.
    let mut order: Vec<(usize, f64)> = (0..gmm.k)
        .filter(|&j| !parts[j].is_empty())
        .map(|j| (j, violations(&parts[j]) as f64 / parts[j].len() as f64))
        .filter(|&(_, v)| v > 0.0)
        .collect();
    order.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut set = Vec::new();
    for (j, _) in order {
        set.push(j);
        let dropped: usize = set.iter().map(|&j| parts[j].len()).sum();
        let kept = n - dropped;
        if kept * 3 < n * 2 {
            break;
        }
        if dropped < min_size {
            continue;
        }
        let rest: Vec<u32> = (0..n as u32)
            .filter(|&i| !set.contains(&(labels[i as usize] as usize)))
            .collect();
        let c = contamination(&rest);
        if !c.is_nan() && c <= CORE_RPV && fewer_than_chance(violations(&rest), chance(kept)) {
            let dp = d_prime(&set);
            if dp > 0.0 {
                let minor = (0..n as u32)
                    .filter(|&i| set.contains(&(labels[i as usize] as usize)))
                    .collect();
                offer("cleanup", minor, kept, c, dp);
            }
            break;
        }
    }
    best.map(|b| b.1)
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

/// Session wiring: [`core_split_proposal`] on `cluster`'s PC feature space
/// (best 4 channels × 3 PCs). Contamination is the sliding-refractory value
/// the probe's `rpc` shows. `None` without PC features.
pub fn core_split<P: DataProvider>(
    session: &Session<P>,
    cluster: ClusterId,
    globals: &[u32],
) -> Option<CoreSplit> {
    let times = session.spike_times(cluster);
    let sr = session.provider.sample_rate().max(1.0);
    let duration = session.provider.n_samples().0;
    let pick = |idx: &[u32]| -> Vec<_> { idx.iter().map(|&i| times[i as usize]).collect() };
    core_split_lazy(
        globals.len(),
        || feature_space(session, cluster, globals, CORE_CHANNELS),
        |idx| {
            let (c, rp) = sliding_refractory_argmin(
                &pick(idx),
                (sr * 0.0005) as u64,
                (sr * 0.010) as u64,
                20,
                duration,
                sr,
            );
            if rp > 0 {
                c
            } else {
                f32::NAN
            }
        },
        |idx| isi_violations(&pick(idx), (sr * 0.0015).round() as u64),
    )
}

/// [`core_split_proposal`] on the `n` spikes of a cluster, with the PC
/// features (`features`, the costly part) extracted only once the cheap gates
/// (size, whole-cluster contamination) pass.
pub fn core_split_lazy(
    n: usize,
    features: impl FnOnce() -> Option<(Vec<f32>, usize)>,
    contamination: impl Fn(&[u32]) -> f32,
    violations: impl Fn(&[u32]) -> usize,
) -> Option<CoreSplit> {
    let whole = core_gate(n, || contamination(&(0..n as u32).collect::<Vec<_>>()))?;
    let (feats, d) = features()?;
    (d > 0 && feats.len() / d == n).then_some(())?;
    core_from_mixture(&feats, d, whole, contamination, violations)
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

    fn nested_core(n_core: usize) -> (Vec<f32>, sorrel_compute::DiagGmm, Vec<bool>) {
        let mut seed = 43;
        let rows: Vec<f32> = (0..(1000 + n_core) * 2)
            .map(|i| {
                gaussian(&mut seed)
                    + if i % 2 == 0 {
                        if i / 2 < 1000 {
                            -5.0
                        } else {
                            5.0
                        }
                    } else {
                        0.0
                    }
            })
            .collect();
        let clean: Vec<_> = rows
            .chunks_exact(2)
            .map(|r| (r[0] - 5.0).powi(2) + r[1].powi(2) <= 2.0)
            .collect();
        let gmm = sorrel_compute::DiagGmm {
            k: 2,
            d: 2,
            weights: vec![
                1000.0 / (1000 + n_core) as f64,
                n_core as f64 / (1000 + n_core) as f64,
            ],
            means: vec![-5.0, 0.0, 5.0, 0.0],
            vars: vec![1.0; 4],
            log_lik: 0.0,
            n: 1000 + n_core,
        };
        (rows, gmm, clean)
    }

    #[test]
    fn dense_interior_is_available_when_the_full_component_is_contaminated() {
        let (rows, gmm, clean) = nested_core(400);
        let c = |idx: &[u32]| {
            if idx.iter().all(|&i| clean[i as usize]) {
                0.0
            } else {
                0.6
            }
        };
        let v = |idx: &[u32]| {
            if idx.iter().all(|&i| clean[i as usize]) {
                0
            } else {
                idx.len() / 2
            }
        };
        let p = core_from_fitted(&rows, 2, 0.6, &gmm, c, v).expect("dense clean interior");
        assert_eq!(p.method, "core");
        assert!(p.minor_idx.len() >= CORE_MIN_SPIKES && p.minor_idx.len() < 400);
        assert!(p.minor_idx.iter().all(|&i| clean[i as usize]));
        assert!(p.minor_idx.windows(2).all(|w| w[0] < w[1]));
        assert!(p.d_prime >= MIN_D_PRIME && p.core_contamination <= CORE_RPV);
        // The same capability is reached through actual BIC fitting, not only
        // the controlled fitted-model test seam.
        assert!(core_split_proposal(&rows, 2, c, v).is_some());
    }

    #[test]
    fn density_trimming_keeps_the_original_size_and_refractory_vetoes() {
        let (rows, gmm, _) = nested_core(400);
        assert!(core_from_fitted(&rows, 2, 0.6, &gmm, |_| 0.6, |_| 0).is_none());
        assert!(core_from_fitted(&rows, 2, 0.6, &gmm, |_| f32::NAN, |_| 0).is_none());
        assert!(core_from_fitted(&rows, 2, 0.6, &gmm, |_| 0.0, |idx| idx.len()).is_none());
        let (rows, gmm, clean) = nested_core(120);
        let c = |idx: &[u32]| {
            if idx.iter().all(|&i| clean[i as usize]) {
                0.0
            } else {
                0.6
            }
        };
        assert!(
            core_from_fitted(&rows, 2, 0.6, &gmm, c, |_| 0).is_none(),
            "a tiny interior must not bypass the 100-spike minimum"
        );
    }

    #[test]
    fn supported_whole_components_are_not_trimmed_for_a_better_score() {
        let (rows, gmm, _) = nested_core(400);
        let c = |idx: &[u32]| {
            if idx.iter().all(|&i| i >= 1000) {
                0.0
            } else {
                0.6
            }
        };
        let v = |idx: &[u32]| {
            if idx.iter().all(|&i| i >= 1000) {
                0
            } else {
                idx.len() / 2
            }
        };
        let p = core_from_fitted(&rows, 2, 0.6, &gmm, c, v).unwrap();
        assert_eq!(p.minor_idx, (1000..1400).collect::<Vec<_>>());
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
