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
    /// LDA d-prime: standardized mean separation between cluster and
    /// background along the Fisher discriminant axis. Larger = more
    /// discriminable; `≳ 4` is "well isolated". NaN if degenerate.
    pub d_prime: f32,
    /// Simplified silhouette in `[-1, 1]`: cluster tightness vs. distance to
    /// the background centroid. Near 1 = well separated. NaN if degenerate.
    pub silhouette: f32,
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
            && self.nn_isolation.is_nan()
            && self.d_prime.is_nan()
            && self.silhouette.is_nan())
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
        if self.d_prime.is_finite() {
            // d' of 4 is the rule-of-thumb "well isolated" threshold.
            // Map via d'/(d'+2): d'=2 → 0.5, d'=4 → 0.67, d'=8 → 0.8.
            let v = self.d_prime.max(0.0);
            parts.push((v / (v + 2.0)).clamp(0.0, 1.0));
        }
        if self.silhouette.is_finite() {
            // Silhouette ∈ [-1, 1] → [0, 1]; s=0 → 0.5, s=1 → 1.0.
            parts.push(((self.silhouette + 1.0) * 0.5).clamp(0.0, 1.0));
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
        d_prime: f32::NAN,
        silhouette: f32::NAN,
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
    // d-prime and silhouette use their own (pooled / centroid) geometry and
    // don't depend on the cluster-covariance inverse, so compute them up front
    // — they stay valid even on the singular-covariance fallback path below.
    out.d_prime = lda_d_prime(features, is_in_cluster, d);
    out.silhouette = simplified_silhouette(features, is_in_cluster, d);

    if invert(&mut cov, 1e-12).is_none() {
        // Even with the ridge, the covariance is singular — typical only
        // when every cluster spike has identical features. Leave the
        // Mahalanobis-based metrics as NaN.
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

/// Nearest-neighbour isolation only (Chung et al. 2017 `nn_hit_rate`-style).
/// Cheap and works even when the covariance is singular; useful as a fallback
/// signal when the Mahalanobis-based metrics return NaN.
///
/// The in-cluster and background populations are **balanced** before counting
/// neighbours: the background is sub-sampled down to the cluster size so the
/// hit rate measures genuine feature-space separation rather than the
/// in/out population ratio. Without balancing, a well-isolated cluster that is
/// only a small fraction of all spikes reads artificially low, because near
/// its boundary the far more numerous background spikes dominate the
/// neighbour list (this matches SpikeInterface's balanced `nn_hit_rate`).
pub fn nn_isolation_metric(features: &[f32], is_in_cluster: &[bool], d: usize, k: usize) -> f32 {
    let n = is_in_cluster.len();
    if n == 0 || d == 0 || k == 0 || features.len() != n * d {
        return f32::NAN;
    }
    let n_in: usize = is_in_cluster.iter().filter(|&&b| b).count();
    if n_in < 2 {
        return f32::NAN;
    }
    let n_out = n - n_in;
    if n_out == 0 {
        return f32::NAN;
    }

    // Balanced neighbour pool: every in-cluster spike plus a deterministically
    // strided subsample of the background, sized to ~n_in so neither
    // population dominates the k-NN counts.
    let out_stride = (n_out / n_in).max(1);
    let mut pool: Vec<usize> = Vec::with_capacity(2 * n_in);
    let mut out_seen = 0usize;
    for (i, &b) in is_in_cluster.iter().enumerate() {
        if b {
            pool.push(i);
        } else {
            if out_seen % out_stride == 0 {
                pool.push(i);
            }
            out_seen += 1;
        }
    }

    // Sub-sample the query points for cost: brute-force k-NN is
    // O(queries × pool × d), e.g. 200 × ~2·n_in × 3.
    const TARGET: usize = 200;
    let stride = (n_in / TARGET).max(1);
    let k_eff = k.min(pool.len() - 1);
    if k_eff == 0 {
        return f32::NAN;
    }

    // Query points: strided in-cluster spikes.
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
            for &j in &pool {
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

/// Partition a row-major `(n, d)` feature buffer into in-cluster and
/// background rows as `f64` (rows with any non-finite value are dropped).
fn partition_rows(features: &[f32], is_in_cluster: &[bool], d: usize) -> (Vec<f64>, Vec<f64>) {
    let mut in_rows = Vec::new();
    let mut out_rows = Vec::new();
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
    (in_rows, out_rows)
}

/// Linear-discriminant-analysis d-prime between the cluster and its
/// background (SpikeInterface `lda_metric` style, after Hill et al. 2011).
///
/// Projects every spike onto the Fisher LDA axis `w = Σ_w⁻¹ (μ_in − μ_out)`
/// (with `Σ_w` the pooled within-class covariance) — the 1-D direction that
/// best separates in- from out-of-cluster points — then reports the
/// standardized mean separation along that axis:
/// `d' = |μ_in − μ_out| / sqrt((σ²_in + σ²_out) / 2)`. Larger = more
/// discriminable; `d' ≳ 4` is the rule-of-thumb "well isolated" threshold.
///
/// Returns `NaN` when either class has fewer than 2 finite rows or the pooled
/// covariance is singular even after ridging.
pub fn lda_d_prime(features: &[f32], is_in_cluster: &[bool], d: usize) -> f32 {
    let n = is_in_cluster.len();
    if d == 0 || n == 0 || features.len() != n * d {
        return f32::NAN;
    }
    let (in_rows, out_rows) = partition_rows(features, is_in_cluster, d);
    let n_in = in_rows.len() / d;
    let n_out = out_rows.len() / d;
    if n_in < 2 || n_out < 2 {
        return f32::NAN;
    }
    let mu_in = row_mean(&in_rows, n_in, d);
    let mu_out = row_mean(&out_rows, n_out, d);
    let cov_in = covariance(&in_rows, n_in, d);
    let cov_out = covariance(&out_rows, n_out, d);

    // Pooled within-class covariance: ((n_in-1)·Σ_in + (n_out-1)·Σ_out) / df.
    let df = (n_in + n_out - 2) as f64;
    let mut sw = Sym::zeros(d);
    for r in 0..d {
        for c in 0..d {
            let v = ((n_in - 1) as f64 * cov_in.get(r, c) + (n_out - 1) as f64 * cov_out.get(r, c))
                / df;
            sw.set(r, c, v);
        }
    }
    let mean_diag = (0..d).map(|i| sw.get(i, i)).sum::<f64>() / d as f64;
    sw.ridge(mean_diag * RIDGE_FRAC);
    if invert(&mut sw, 1e-12).is_none() {
        return f32::NAN;
    }
    // Fisher axis w = Σ_w⁻¹ (μ_in − μ_out).
    let diff: Vec<f64> = (0..d).map(|j| mu_in[j] - mu_out[j]).collect();
    let w: Vec<f64> = (0..d)
        .map(|r| (0..d).map(|c| sw.get(r, c) * diff[c]).sum::<f64>())
        .collect();

    // Project each class onto w and take its mean / population variance.
    let project = |rows: &[f64], count: usize| -> (f64, f64) {
        let projs: Vec<f64> = (0..count)
            .map(|i| (0..d).map(|j| w[j] * rows[i * d + j]).sum::<f64>())
            .collect();
        let mean = projs.iter().sum::<f64>() / count as f64;
        let var = projs.iter().map(|&p| (p - mean).powi(2)).sum::<f64>() / count as f64;
        (mean, var)
    };
    let (m_in, v_in) = project(&in_rows, n_in);
    let (m_out, v_out) = project(&out_rows, n_out);
    let pooled = 0.5 * (v_in + v_out);
    if pooled <= 0.0 || pooled.is_nan() {
        return f32::NAN;
    }
    ((m_in - m_out).abs() / pooled.sqrt()) as f32
}

/// Simplified silhouette (Hruschka et al. 2004) of the cluster against its
/// pooled background. For each cluster spike, `a` is its Euclidean distance
/// to the cluster centroid and `b` its distance to the background centroid;
/// the score is the mean of `(b − a) / max(a, b)` over cluster spikes.
///
/// Ranges `[-1, 1]`: near 1 means a tight, well-separated cluster; near 0
/// means it overlaps the background; negative means cluster spikes sit closer
/// to the background centroid than their own. This is the binary (cluster vs
/// pooled background) form — it treats all non-cluster spikes as one group
/// rather than scoring against each neighbouring unit separately.
///
/// Returns `NaN` when either side is empty.
pub fn simplified_silhouette(features: &[f32], is_in_cluster: &[bool], d: usize) -> f32 {
    let n = is_in_cluster.len();
    if d == 0 || n == 0 || features.len() != n * d {
        return f32::NAN;
    }
    let (in_rows, out_rows) = partition_rows(features, is_in_cluster, d);
    let n_in = in_rows.len() / d;
    let n_out = out_rows.len() / d;
    if n_in == 0 || n_out == 0 {
        return f32::NAN;
    }
    let mu_in = row_mean(&in_rows, n_in, d);
    let mu_out = row_mean(&out_rows, n_out, d);
    let dist_to = |row: &[f64], centroid: &[f64]| -> f64 {
        (0..d)
            .map(|j| (row[j] - centroid[j]).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let mut acc = 0.0_f64;
    for i in 0..n_in {
        let row = &in_rows[i * d..(i + 1) * d];
        let a = dist_to(row, &mu_in);
        let b = dist_to(row, &mu_out);
        let m = a.max(b);
        if m > 0.0 {
            acc += (b - a) / m;
        }
    }
    (acc / n_in as f64) as f32
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

    /// The balancing property: a moderately-separated cluster's NN isolation
    /// must stay roughly stable as the background population grows. Without
    /// the in/out balancing, a 10×-larger background pulls more out-cluster
    /// spikes into every neighbour list and the score collapses; with
    /// balancing the background is sub-sampled to the cluster size first, so
    /// the score reflects separation, not the population ratio.
    fn nn_with_background(n_out: usize) -> f32 {
        let n_in = 200;
        // Partial overlap (centroids ~3σ apart) so the metric is sensitive
        // to neighbour composition rather than saturating at 1.0.
        let mut feats = cloud(n_in, [0.0, 0.0], 1.0, 0xA11CE);
        feats.extend(cloud(n_out, [3.0, 0.0], 1.0, 0xB0B));
        let mut is_in = vec![true; n_in];
        is_in.extend(vec![false; n_out]);
        nn_isolation_metric(&feats, &is_in, 2, 5)
    }

    #[test]
    fn nn_isolation_is_stable_under_background_growth() {
        let small_bg = nn_with_background(200);
        let large_bg = nn_with_background(2000);
        assert!(small_bg.is_finite() && large_bg.is_finite());
        // A 10× background change should barely move the balanced metric.
        assert!(
            (small_bg - large_bg).abs() < 0.1,
            "balanced NN isolation drifted with background size: {small_bg} vs {large_bg}",
        );
    }

    #[test]
    fn d_prime_larger_for_better_separated() {
        let n = 300;
        let mut is_in = vec![true; n];
        is_in.extend(vec![false; n]);
        // Far-apart clusters → large d'.
        let mut far = cloud(n, [0.0, 0.0], 1.0, 0x1111);
        far.extend(cloud(n, [8.0, 0.0], 1.0, 0x2222));
        // Near clusters → small d'.
        let mut near = cloud(n, [0.0, 0.0], 1.0, 0x3333);
        near.extend(cloud(n, [1.0, 0.0], 1.0, 0x4444));
        let d_far = lda_d_prime(&far, &is_in, 2);
        let d_near = lda_d_prime(&near, &is_in, 2);
        assert!(d_far.is_finite() && d_near.is_finite());
        assert!(
            d_far > d_near,
            "d' should grow with separation: far {d_far} vs near {d_near}",
        );
        // 8σ separation is very well isolated — comfortably past the d'≈4 rule.
        assert!(d_far > 4.0, "well-separated d' {d_far} unexpectedly small");
    }

    #[test]
    fn silhouette_high_when_separated_low_when_overlapping() {
        let n = 300;
        let mut is_in = vec![true; n];
        is_in.extend(vec![false; n]);
        let mut sep = cloud(n, [0.0, 0.0], 1.0, 0x5555);
        sep.extend(cloud(n, [12.0, 0.0], 1.0, 0x6666));
        let mut overlap = cloud(n, [0.0, 0.0], 1.0, 0x7777);
        overlap.extend(cloud(n, [0.0, 0.0], 1.0, 0x8888));
        let s_sep = simplified_silhouette(&sep, &is_in, 2);
        let s_overlap = simplified_silhouette(&overlap, &is_in, 2);
        assert!((-1.0..=1.0).contains(&s_sep) && (-1.0..=1.0).contains(&s_overlap));
        assert!(s_sep > 0.5, "separated silhouette {s_sep} should be high",);
        assert!(
            s_overlap < s_sep,
            "overlapping silhouette {s_overlap} should be below separated {s_sep}",
        );
    }

    #[test]
    fn d_prime_and_silhouette_nan_on_degenerate_inputs() {
        // No background.
        let feats = cloud(60, [0.0, 0.0], 1.0, 9);
        let is_in = vec![true; 60];
        assert!(lda_d_prime(&feats, &is_in, 2).is_nan());
        assert!(simplified_silhouette(&feats, &is_in, 2).is_nan());
    }
}
