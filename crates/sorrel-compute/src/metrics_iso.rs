//! PC-feature-based cluster-isolation metrics.
//!
//! These quantify *how separable* a cluster is from everything else in
//! feature space — the dimension manual curators usually rely on most when
//! they're staring at the FeatureView lasso. The classics:
//!
//! * **Isolation distance** (Schmitzer-Torbert et al. 2005): the
//!   Mahalanobis distance from the cluster centroid to the `N`-th closest
//!   non-cluster point, where `N = |cluster|`. Bigger = better-isolated.
//! * **L-ratio**: chi-squared survival sum over the same non-cluster
//!   points' Mahalanobis distances, normalised by N. Smaller = better.
//! * **Nearest-neighbour isolation** (Chung 2017 style): for a sample of
//!   cluster points, the fraction of their k nearest neighbours (in
//!   Euclidean PC-space) that *do* belong to the same cluster. Robust to
//!   non-Gaussian shapes.
//!
//! All three operate on a dense `(n_spikes, d)` PC feature buffer plus a
//! per-spike cluster id. They share a small linear-algebra layer in
//! [`crate::linalg`] so we don't drag in a matrix crate.
//!
//! Caller responsibilities:
//! - Sub-select a `d`-dim subspace (e.g. 3 PCs on the peak channel) before
//!   calling. d > ~6 makes the covariance ill-conditioned and the metrics
//!   pointless on typical cluster sizes.
//! - Skip clusters with fewer than [`MIN_CLUSTER_SPIKES`] spikes — the
//!   metrics return `f32::NAN` rather than misleading values.

use crate::linalg::{chi2_sf, covariance, invert, mahalanobis_sq, row_mean, Sym};

/// Minimum cluster size for which the Mahalanobis-based metrics are
/// considered meaningful. Below this the covariance is too noisy to invert
/// even with regularisation. Matches `spikeinterface`'s default of 50.
pub const MIN_CLUSTER_SPIKES: usize = 50;

/// Tikhonov ridge added to the covariance diagonal before inversion. Tiny —
/// only kicks in for nearly-singular matrices; real clusters with even a
/// few hundred spikes are far above this floor.
const RIDGE_FRAC: f64 = 1e-4;

/// Bundle of PC-feature isolation metrics for one cluster vs. its
/// background. Any field is `NaN` if the inputs were degenerate (too few
/// spikes, all-zero features, singular covariance even after ridging).
#[derive(Clone, Copy, Debug, Default)]
pub struct IsolationMetrics {
    /// Mahalanobis distance² to the N-th closest non-cluster point.
    /// Stored as the squared distance — that's the chi-squared scale and
    /// it avoids a sqrt the caller usually doesn't need. NaN if not enough
    /// non-cluster points to find an N-th neighbour.
    pub isolation_distance_sq: f32,
    /// L-ratio. Lower = better. Bounded only above 0; for a perfectly
    /// isolated cluster it asymptotes near 0.
    pub l_ratio: f32,
    /// Nearest-neighbour isolation in `[0, 1]`. 1.0 means every neighbour
    /// of every cluster spike is also in the cluster.
    pub nn_isolation: f32,
    /// Number of cluster spikes used in the calculation (after any
    /// sub-sampling).
    pub n_in: u32,
    /// Number of non-cluster spikes considered in the background.
    pub n_out: u32,
}

impl IsolationMetrics {
    /// True if any field carries usable evidence (i.e. not NaN).
    pub fn has_evidence(&self) -> bool {
        !(self.isolation_distance_sq.is_nan()
            && self.l_ratio.is_nan()
            && self.nn_isolation.is_nan())
    }

    /// Convert to a single `[0, 1]` isolation score for use in a weighted
    /// quality composite. Uses a saturating transform on isolation_distance
    /// (a typical "well isolated" Mahalanobis² is ≥ 20) and a complement
    /// transform on L-ratio (well isolated ≪ 1).
    pub fn score(&self) -> f32 {
        let mut parts: Vec<f32> = Vec::new();
        if self.isolation_distance_sq.is_finite() {
            // Mahalanobis² of 20 ≈ 99.9% chi² for d=3 → "well isolated".
            // Map [0, 30] → [0, 1] via x/(x+10) → x=20 gives 0.67, x=30 gives 0.75.
            let v = self.isolation_distance_sq;
            parts.push((v / (v + 10.0)).clamp(0.0, 1.0));
        }
        if self.l_ratio.is_finite() {
            // L-ratio of 0.1 is the rule-of-thumb good threshold.
            // Map via 1 / (1 + L/0.1) — L=0 → 1.0, L=0.1 → 0.5, L=1 → 0.09.
            parts.push((1.0 / (1.0 + self.l_ratio * 10.0)).clamp(0.0, 1.0));
        }
        if self.nn_isolation.is_finite() {
            parts.push(self.nn_isolation.clamp(0.0, 1.0));
        }
        if parts.is_empty() {
            return f32::NAN;
        }
        parts.iter().copied().sum::<f32>() / parts.len() as f32
    }
}

/// Compute the full isolation bundle for one cluster. `features` is a
/// row-major `(n_spikes, d)` buffer of *all* spikes' PC features (cluster +
/// background combined); `is_in_cluster[i]` is true iff row `i` belongs to
/// the cluster under test.
///
/// `k_nn` is the number of nearest neighbours to consider for the NN
/// metric (8 is the standard).
pub fn isolation_metrics(
    features: &[f32],
    is_in_cluster: &[bool],
    d: usize,
    k_nn: usize,
) -> IsolationMetrics {
    let mut out = IsolationMetrics {
        isolation_distance_sq: f32::NAN,
        l_ratio: f32::NAN,
        nn_isolation: f32::NAN,
        n_in: 0,
        n_out: 0,
    };
    if d == 0 || features.is_empty() || is_in_cluster.is_empty() {
        return out;
    }
    let n_total = is_in_cluster.len();
    if features.len() != n_total * d {
        return out;
    }
    // Partition rows into in / out (as f64 so the linalg routines run at
    // higher precision; PC features themselves are float32).
    let mut in_rows: Vec<f64> = Vec::new();
    let mut out_rows: Vec<f64> = Vec::new();
    for (i, &is_in) in is_in_cluster.iter().enumerate() {
        let row = &features[i * d..(i + 1) * d];
        if !row.iter().all(|v| v.is_finite()) {
            continue;
        }
        if is_in {
            in_rows.extend(row.iter().map(|&v| v as f64));
        } else {
            out_rows.extend(row.iter().map(|&v| v as f64));
        }
    }
    let n_in = in_rows.len() / d;
    let n_out = out_rows.len() / d;
    out.n_in = n_in as u32;
    out.n_out = n_out as u32;

    if n_in < MIN_CLUSTER_SPIKES || n_out == 0 {
        return out;
    }

    // Cluster centroid + covariance (Bessel-corrected).
    let mu = row_mean(&in_rows, n_in, d);
    let mut cov = covariance(&in_rows, n_in, d);
    // Tikhonov regularisation: ridge proportional to mean diagonal so the
    // scale is sensible regardless of feature units.
    let mean_diag = (0..d).map(|i| cov.get(i, i)).sum::<f64>() / d as f64;
    cov.ridge(mean_diag * RIDGE_FRAC);
    if invert(&mut cov, 1e-12).is_none() {
        // Even with the ridge, the covariance is singular — typical only
        // when every cluster spike has identical features. Leave the
        // metrics as NaN.
        out.nn_isolation = nn_isolation_metric(features, is_in_cluster, d, k_nn);
        return out;
    }
    let inv_cov: Sym = cov; // it's been overwritten in place

    // Mahalanobis² of every non-cluster point against the cluster mean.
    let mut d2: Vec<f64> = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let row = &out_rows[i * d..(i + 1) * d];
        d2.push(mahalanobis_sq(row, &mu, &inv_cov));
    }
    d2.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Isolation distance: Mahalanobis² of the n_in-th closest non-cluster
    // point. If there are fewer than n_in non-cluster points, the metric is
    // undefined (per Schmitzer-Torbert).
    if n_out >= n_in {
        out.isolation_distance_sq = d2[n_in - 1] as f32;
    }
    // L-ratio: Σ_i χ²-survival(d²_i, df=d) / n_in.
    let l_sum: f64 = d2.iter().map(|&v| chi2_sf(v, d as f64)).sum();
    out.l_ratio = (l_sum / n_in as f64) as f32;

    // NN-isolation: walks the full feature buffer, so we feed it the
    // original (unsplit) inputs and get to share the partition pass above
    // for free in callers that need it later.
    out.nn_isolation = nn_isolation_metric(features, is_in_cluster, d, k_nn);

    out
}

/// Nearest-neighbour isolation only. Cheap and works even when the
/// covariance is singular; useful as a fallback signal when the
/// Mahalanobis-based metrics return NaN.
pub fn nn_isolation_metric(features: &[f32], is_in_cluster: &[bool], d: usize, k: usize) -> f32 {
    let n = is_in_cluster.len();
    if n == 0 || d == 0 || k == 0 || features.len() != n * d {
        return f32::NAN;
    }
    let n_in: usize = is_in_cluster.iter().filter(|&&b| b).count();
    if n_in < 2 {
        return f32::NAN;
    }
    // Sub-sample for cost: brute-force k-NN is O(n_in × n × d).
    // 200 cluster points × n=20k × d=3 = 12M ops — fine. Above ~500 we
    // stride.
    const TARGET: usize = 200;
    let stride = (n_in / TARGET).max(1);
    let k_eff = k.min(n - 1);

    // Pick out the cluster-spike indices we'll query, applying the stride.
    let queries: Vec<usize> = is_in_cluster
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| b.then_some(i))
        .step_by(stride)
        .collect();
    if queries.is_empty() {
        return f32::NAN;
    }
    use rayon::prelude::*;
    let (own, total) = queries
        .par_iter()
        .map(|&i| {
            let xi = &features[i * d..(i + 1) * d];
            let mut topk: Vec<(f32, bool)> = Vec::with_capacity(k_eff + 1);
            for j in 0..n {
                if j == i {
                    continue;
                }
                let xj = &features[j * d..(j + 1) * d];
                let mut d2 = 0.0_f32;
                for r in 0..d {
                    let diff = xi[r] - xj[r];
                    d2 += diff * diff;
                }
                if topk.len() < k_eff {
                    topk.push((d2, is_in_cluster[j]));
                    topk.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                } else if d2 < topk[k_eff - 1].0 {
                    topk[k_eff - 1] = (d2, is_in_cluster[j]);
                    topk.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                }
            }
            let mut own = 0u32;
            let total = topk.len() as u32;
            for &(_, is_in_other) in &topk {
                if is_in_other {
                    own += 1;
                }
            }
            (own, total)
        })
        .reduce(|| (0u32, 0u32), |(o1, t1), (o2, t2)| (o1 + o2, t1 + t2));
    if total == 0 {
        return f32::NAN;
    }
    own as f32 / total as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a 2-D Gaussian cloud at `mu` with isotropic spread `sigma`.
    fn cloud(n: usize, mu: [f32; 2], sigma: f32, seed: u32) -> Vec<f32> {
        let mut v = Vec::with_capacity(n * 2);
        let mut s = seed | 1;
        for _ in 0..n {
            // Box–Muller from xorshift32.
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let u1 = (s as f64 / u32::MAX as f64).clamp(1e-9, 1.0);
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let u2 = (s as f64 / u32::MAX as f64).clamp(1e-9, 1.0);
            let r = (-2.0 * u1.ln()).sqrt();
            let g1 = (r * (2.0 * std::f64::consts::PI * u2).cos()) as f32;
            let g2 = (r * (2.0 * std::f64::consts::PI * u2).sin()) as f32;
            v.push(mu[0] + g1 * sigma);
            v.push(mu[1] + g2 * sigma);
        }
        v
    }

    #[test]
    fn well_separated_clusters_yield_high_isolation() {
        let n_in = 200;
        let n_out = 1000;
        let mut feats = cloud(n_in, [0.0, 0.0], 1.0, 0xCAFEBABE);
        feats.extend(cloud(n_out, [10.0, 10.0], 1.0, 0xDEADBEEF));
        let mut is_in = vec![true; n_in];
        is_in.extend(vec![false; n_out]);
        let m = isolation_metrics(&feats, &is_in, 2, 5);
        assert!(
            m.isolation_distance_sq > 20.0,
            "expected large isolation, got {}",
            m.isolation_distance_sq
        );
        assert!(m.l_ratio < 0.01, "expected tiny L-ratio, got {}", m.l_ratio);
        assert!(
            m.nn_isolation > 0.95,
            "expected high NN, got {}",
            m.nn_isolation
        );
        assert!(m.score() > 0.7);
    }

    #[test]
    fn overlapping_clusters_yield_low_isolation() {
        let n_in = 200;
        let n_out = 200;
        // Same centroid → no isolation possible.
        let mut feats = cloud(n_in, [0.0, 0.0], 1.0, 0x12345678);
        feats.extend(cloud(n_out, [0.0, 0.0], 1.0, 0x87654321));
        let mut is_in = vec![true; n_in];
        is_in.extend(vec![false; n_out]);
        let m = isolation_metrics(&feats, &is_in, 2, 5);
        // For overlapping unit-variance 2-D Gaussians the n_in-th non-cluster
        // Mahalanobis² is roughly χ² at percentile n_in/n_out — order-10 in
        // 2-D when n_in ≈ n_out. We just want the contrast against the
        // well-separated case (which scores >20).
        assert!(
            m.isolation_distance_sq < 15.0,
            "expected small isolation, got {}",
            m.isolation_distance_sq
        );
        assert!(m.l_ratio > 0.1, "expected high L-ratio, got {}", m.l_ratio);
        assert!(
            m.nn_isolation < 0.7,
            "expected low NN, got {}",
            m.nn_isolation
        );
    }

    #[test]
    fn small_cluster_returns_nan() {
        let n_in = 10;
        let n_out = 100;
        let mut feats = cloud(n_in, [0.0, 0.0], 1.0, 1);
        feats.extend(cloud(n_out, [5.0, 5.0], 1.0, 2));
        let mut is_in = vec![true; n_in];
        is_in.extend(vec![false; n_out]);
        let m = isolation_metrics(&feats, &is_in, 2, 5);
        assert!(m.isolation_distance_sq.is_nan());
        assert!(m.l_ratio.is_nan());
    }

    #[test]
    fn singular_cov_falls_back_to_nn_only() {
        // All cluster spikes share identical features → cov is zero.
        let n_in = MIN_CLUSTER_SPIKES + 10;
        let mut feats: Vec<f32> = Vec::new();
        for _ in 0..n_in {
            feats.push(1.0);
            feats.push(1.0);
        }
        // Background spread out.
        feats.extend(cloud(200, [5.0, 5.0], 1.0, 7));
        let mut is_in = vec![true; n_in];
        is_in.extend(vec![false; 200]);
        let m = isolation_metrics(&feats, &is_in, 2, 5);
        // After the ridge the covariance becomes invertible, so we may
        // still get a finite Mahalanobis-based answer; what we *insist*
        // on is that NN-isolation is finite and large (every neighbour of
        // a cluster point will be another cluster point because they're
        // colocated).
        assert!(m.nn_isolation.is_finite());
        assert!(m.nn_isolation > 0.95);
    }

    #[test]
    fn empty_inputs_safe() {
        let m = isolation_metrics(&[], &[], 2, 5);
        assert_eq!(m.n_in, 0);
        assert_eq!(m.n_out, 0);
        assert!(m.isolation_distance_sq.is_nan());
    }

    #[test]
    fn nn_isolation_only_works_without_cov() {
        let n_in = 5;
        let mut feats = cloud(n_in, [0.0, 0.0], 1.0, 11);
        feats.extend(cloud(50, [10.0, 10.0], 1.0, 13));
        let mut is_in = vec![true; n_in];
        is_in.extend(vec![false; 50]);
        // n_in < MIN_CLUSTER_SPIKES — full metrics return NaN, but the
        // standalone NN function still works.
        let nn = nn_isolation_metric(&feats, &is_in, 2, 4);
        assert!(nn.is_finite());
    }
}
