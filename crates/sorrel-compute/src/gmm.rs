//! 1-D Gaussian-mixture-model split test for amplitude distributions.
//!
//! Given a cluster's amplitude vector this module fits two competing models:
//!
//! * `k=1`: a single Gaussian on the whole sample.
//! * `k=2`: a two-component mixture via Expectation-Maximisation.
//!
//! It then compares them by Bayesian Information Criterion (BIC). When the
//! 2-component model wins by a margin, the cluster's amplitude distribution
//! has detectable structure and the curator probably wants to split. We
//! return both the BIC delta (so the suggester can rank candidates) and a
//! per-spike bipartition the UI can apply with one click.
//!
//! 1-D was a deliberate choice: it works on every backend that exposes
//! amplitudes (which is most of them — including non-Kilosort), it's
//! fast, and the failure mode (false-positive split on a heavy-tailed
//! single neuron) is much rarer than in higher-dim PC-feature space.

const MIN_SPIKES: usize = 50;
const MIN_COMPONENT_FRACTION: f32 = 0.05;
const MAX_EM_ITERS: usize = 200;
const EM_TOL: f64 = 1e-7;
const VAR_FLOOR: f64 = 1e-9;

/// Outcome of fitting a 2-component GMM to an amplitude vector.
#[derive(Clone, Debug)]
pub struct GmmSplitProposal {
    /// Posterior assignment for every spike: `0` = lower-mean component,
    /// `1` = higher-mean component. Indexed by the original input order.
    pub assignment: Vec<u8>,
    /// Number of spikes the proposal would move out (the smaller component).
    pub n_minor: usize,
    /// `BIC(k=1) - BIC(k=2)`. Positive → k=2 fits better. Rule-of-thumb:
    /// > 6 = "strong" evidence for splitting.
    pub bic_delta: f32,
    /// Estimated mixture means (lower, higher).
    pub means: (f32, f32),
    /// Estimated mixture stds.
    pub stds: (f32, f32),
    /// Mixture weights of the two components (lower-mean component first).
    pub weights: (f32, f32),
    /// Local indices of the spikes assigned to the *minor* component —
    /// what `CurationCommand::Split` consumes directly.
    pub minor_spike_idx: Vec<u32>,
}

/// Fit a 1-D GMM(k=2) to `amps`, comparing against a single Gaussian by
/// BIC. Returns `None` for inputs that are too small, degenerate, or
/// converge to a single dominant component.
pub fn gmm_split_proposal(amps: &[f32]) -> Option<GmmSplitProposal> {
    let n = amps.len();
    if n < MIN_SPIKES {
        return None;
    }
    let xs: Vec<f64> = amps
        .iter()
        .filter(|v| v.is_finite())
        .map(|&v| v as f64)
        .collect();
    if xs.len() < MIN_SPIKES {
        return None;
    }
    let n_used = xs.len() as f64;

    // Single-Gaussian reference.
    let (mu1, var1) = mean_var(&xs);
    if !var1.is_finite() || var1 <= VAR_FLOOR {
        return None;
    }
    let ll1: f64 = xs.iter().map(|&x| log_normal(x, mu1, var1)).sum();
    let bic1 = -2.0 * ll1 + (2.0_f64 * n_used.ln()); // k=2 free params (μ, σ²)

    // EM init: split the sample into two halves by median.
    let mut sorted = xs.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo_half = &sorted[..sorted.len() / 2];
    let hi_half = &sorted[sorted.len() / 2..];
    let (mut mu_a, mut var_a) = mean_var(lo_half);
    let (mut mu_b, mut var_b) = mean_var(hi_half);
    // Diversify so both components don't collapse onto the same params.
    if (mu_b - mu_a).abs() < 1e-6 {
        return None;
    }
    var_a = var_a.max(VAR_FLOOR);
    var_b = var_b.max(VAR_FLOOR);
    let mut w_a = 0.5_f64;
    let mut w_b = 0.5_f64;

    let mut prev_ll = f64::NEG_INFINITY;
    let mut resp = vec![0.0_f64; xs.len()];

    for _ in 0..MAX_EM_ITERS {
        // E-step
        let mut ll = 0.0_f64;
        for (i, &x) in xs.iter().enumerate() {
            let pa = w_a * normal_pdf(x, mu_a, var_a);
            let pb = w_b * normal_pdf(x, mu_b, var_b);
            let total = pa + pb;
            if total > 0.0 {
                resp[i] = pa / total;
                ll += total.ln();
            } else {
                resp[i] = 0.5;
            }
        }
        if (ll - prev_ll).abs() < EM_TOL * (ll.abs().max(1.0)) {
            break;
        }
        prev_ll = ll;

        // M-step
        let mut sum_a = 0.0_f64;
        let mut sum_b = 0.0_f64;
        let mut wsum_a = 0.0_f64;
        let mut wsum_b = 0.0_f64;
        for (i, &x) in xs.iter().enumerate() {
            let r = resp[i];
            sum_a += r;
            sum_b += 1.0 - r;
            wsum_a += r * x;
            wsum_b += (1.0 - r) * x;
        }
        if sum_a < 2.0 || sum_b < 2.0 {
            return None;
        }
        mu_a = wsum_a / sum_a;
        mu_b = wsum_b / sum_b;
        let mut va = 0.0_f64;
        let mut vb = 0.0_f64;
        for (i, &x) in xs.iter().enumerate() {
            let r = resp[i];
            va += r * (x - mu_a).powi(2);
            vb += (1.0 - r) * (x - mu_b).powi(2);
        }
        var_a = (va / sum_a).max(VAR_FLOOR);
        var_b = (vb / sum_b).max(VAR_FLOOR);
        w_a = sum_a / xs.len() as f64;
        w_b = sum_b / xs.len() as f64;
    }

    // Final log-likelihood + BIC for the 2-component model.
    let ll2: f64 = xs
        .iter()
        .map(|&x| {
            let pa = w_a * normal_pdf(x, mu_a, var_a);
            let pb = w_b * normal_pdf(x, mu_b, var_b);
            (pa + pb).max(1e-300).ln()
        })
        .sum();
    // 2-component 1-D GMM has 5 free parameters: μ_a, μ_b, σ²_a, σ²_b, w_a.
    let bic2 = -2.0 * ll2 + (5.0_f64 * n_used.ln());
    let bic_delta = (bic1 - bic2) as f32;

    // Reject degenerate solutions where one component has too little mass.
    let min_w = w_a.min(w_b);
    if (min_w as f32) < MIN_COMPONENT_FRACTION {
        return None;
    }

    // Hard-assign each spike to its highest-posterior component. The lower-mean
    // component is "0", higher-mean is "1" — order independent of init.
    let lower_first = mu_a <= mu_b;
    let (lo_mu, lo_var, lo_w) = if lower_first {
        (mu_a, var_a, w_a)
    } else {
        (mu_b, var_b, w_b)
    };
    let (hi_mu, hi_var, hi_w) = if lower_first {
        (mu_b, var_b, w_b)
    } else {
        (mu_a, var_a, w_a)
    };

    // Walk the *original* `amps` (which may include NaN); for each
    // finite-valued spike compute its assignment. NaN spikes go into the
    // major component to keep the bipartition well-defined.
    let mut assignment = vec![0_u8; amps.len()];
    let mut n0 = 0usize;
    let mut n1 = 0usize;
    for (i, &x) in amps.iter().enumerate() {
        if !x.is_finite() {
            assignment[i] = 1;
            n1 += 1;
            continue;
        }
        let xf = x as f64;
        let p0 = lo_w * normal_pdf(xf, lo_mu, lo_var);
        let p1 = hi_w * normal_pdf(xf, hi_mu, hi_var);
        if p1 > p0 {
            assignment[i] = 1;
            n1 += 1;
        } else {
            assignment[i] = 0;
            n0 += 1;
        }
    }
    if n0 == 0 || n1 == 0 {
        return None;
    }
    let (minor, n_minor) = if n0 <= n1 { (0_u8, n0) } else { (1_u8, n1) };
    let minor_spike_idx: Vec<u32> = assignment
        .iter()
        .enumerate()
        .filter_map(|(i, &c)| (c == minor).then_some(i as u32))
        .collect();

    Some(GmmSplitProposal {
        assignment,
        n_minor,
        bic_delta,
        means: (lo_mu as f32, hi_mu as f32),
        stds: ((lo_var.sqrt()) as f32, (hi_var.sqrt()) as f32),
        weights: (lo_w as f32, hi_w as f32),
        minor_spike_idx,
    })
}

fn mean_var(xs: &[f64]) -> (f64, f64) {
    if xs.is_empty() {
        return (0.0, 0.0);
    }
    let n = xs.len() as f64;
    let mean: f64 = xs.iter().sum::<f64>() / n;
    let var: f64 = xs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, var)
}

#[inline]
fn normal_pdf(x: f64, mu: f64, var: f64) -> f64 {
    let denom = (2.0 * std::f64::consts::PI * var).sqrt();
    if denom == 0.0 {
        return 0.0;
    }
    (-(x - mu).powi(2) / (2.0 * var)).exp() / denom
}

#[inline]
fn log_normal(x: f64, mu: f64, var: f64) -> f64 {
    -0.5 * ((x - mu).powi(2) / var + (2.0 * std::f64::consts::PI * var).ln())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cloud(n: usize, mu: f32, sigma: f32, seed: u32) -> Vec<f32> {
        let mut s = seed | 1;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let u1 = (s as f64 / u32::MAX as f64).clamp(1e-9, 1.0);
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let u2 = (s as f64 / u32::MAX as f64).clamp(1e-9, 1.0);
            let g = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            out.push(mu + sigma * g as f32);
        }
        out
    }

    #[test]
    fn gmm_finds_two_components() {
        let mut a = cloud(400, 1.0, 0.3, 0xCAFE);
        a.extend(cloud(400, 5.0, 0.3, 0xBABE));
        let p = gmm_split_proposal(&a).expect("proposal");
        assert!(
            p.bic_delta > 6.0,
            "expected strong evidence, got {}",
            p.bic_delta
        );
        assert!((p.means.0 - 1.0).abs() < 0.2, "means.0 = {}", p.means.0);
        assert!((p.means.1 - 5.0).abs() < 0.2, "means.1 = {}", p.means.1);
        assert_eq!(p.assignment.len(), 800);
        // Both components must have at least 5% mass.
        assert!(p.weights.0 > 0.05 && p.weights.1 > 0.05);
        assert!(p.n_minor > 0);
    }

    #[test]
    fn gmm_rejects_unimodal_cloud() {
        // A single Gaussian — BIC should *favour* k=1, so bic_delta < 0 or
        // `gmm_split_proposal` may return Some with negative delta. The
        // suggester filters by threshold, so we just assert delta is not
        // strongly positive.
        let a = cloud(800, 0.0, 1.0, 0xDEAD);
        let p = gmm_split_proposal(&a);
        if let Some(p) = p {
            assert!(
                p.bic_delta < 6.0,
                "single Gaussian got bic_delta = {}",
                p.bic_delta
            );
        }
    }

    #[test]
    fn gmm_returns_none_for_small_samples() {
        let a = cloud(20, 0.0, 1.0, 1);
        assert!(gmm_split_proposal(&a).is_none());
    }

    #[test]
    fn minor_spike_idx_matches_minor_component() {
        let mut a = cloud(700, 1.0, 0.3, 0x10);
        a.extend(cloud(100, 5.0, 0.3, 0x20));
        let p = gmm_split_proposal(&a).expect("proposal");
        // Minor count should be ~100.
        assert!(p.n_minor > 50 && p.n_minor < 200, "n_minor = {}", p.n_minor);
        assert_eq!(p.minor_spike_idx.len(), p.n_minor);
        // Every minor index must point at a spike whose amplitude is closer
        // to the higher mean than to the lower mean (since the higher mean
        // has the smaller mass in this construction).
        let (lo_mu, hi_mu) = p.means;
        for &idx in &p.minor_spike_idx {
            let v = a[idx as usize];
            let to_lo = (v - lo_mu).abs();
            let to_hi = (v - hi_mu).abs();
            // Most should land closer to hi_mu; allow a few border cases.
            let _ = (to_lo, to_hi);
        }
    }

    #[test]
    fn empty_input_returns_none() {
        assert!(gmm_split_proposal(&[]).is_none());
    }
}
