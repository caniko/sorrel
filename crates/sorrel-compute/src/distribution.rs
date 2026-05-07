//! Distribution-shape statistics used by the curation suggestion engine.
//!
//! These are deliberately self-contained (no external linear-algebra crate)
//! and operate on small `&[f32]` slices the rest of the workspace already
//! produces — amplitudes, ISIs, single PC-feature axes. They feed the
//! split/merge suggesters in `sorrel-data` and the per-cluster quality
//! score, and stay in `sorrel-compute` so any consumer can pull a metric
//! without dragging in the GUI or session code.

/// Arithmetic mean of an `f32` slice in `f64` precision. Returns 0 on empty.
pub fn mean_f64(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().map(|&v| v as f64).sum::<f64>() / values.len() as f64
}

/// Central moments (m2, m3, m4) of a sample, computed in a single pass after
/// the mean. `m_k = E[(x - mean)^k]`. Bundling these so callers that need
/// several distribution-shape statistics (skewness, excess kurtosis, BC) walk
/// the input once instead of three times.
#[derive(Clone, Copy, Debug)]
pub struct Moments {
    pub n: usize,
    pub mean: f64,
    pub m2: f64,
    pub m3: f64,
    pub m4: f64,
}

impl Moments {
    pub fn from_f32(values: &[f32]) -> Self {
        let n = values.len();
        let mean = mean_f64(values);
        let mut m2 = 0.0_f64;
        let mut m3 = 0.0_f64;
        let mut m4 = 0.0_f64;
        for &v in values {
            let d = v as f64 - mean;
            let d2 = d * d;
            m2 += d2;
            m3 += d2 * d;
            m4 += d2 * d2;
        }
        let nf = n.max(1) as f64;
        Self { n, mean, m2: m2 / nf, m3: m3 / nf, m4: m4 / nf }
    }

    /// Bessel-corrected sample standard deviation. 0 for n < 2.
    pub fn sample_std(&self) -> f64 {
        if self.n < 2 {
            return 0.0;
        }
        // m2 stored as biased (÷n); rescale to (÷(n-1)).
        let nf = self.n as f64;
        (self.m2 * nf / (nf - 1.0)).sqrt()
    }

    /// Fisher-Pearson sample skewness (bias-uncorrected). 0 if n < 3 or m2 <= 0.
    pub fn skewness(&self) -> f32 {
        if self.n < 3 || self.m2 <= 0.0 {
            return 0.0;
        }
        (self.m3 / self.m2.powf(1.5)) as f32
    }

    /// Sample excess kurtosis (≈ 0 for a normal distribution). 0 if n < 4 or m2 <= 0.
    pub fn excess_kurtosis(&self) -> f32 {
        if self.n < 4 || self.m2 <= 0.0 {
            return 0.0;
        }
        (self.m4 / (self.m2 * self.m2) - 3.0) as f32
    }
}

/// Mean and (Bessel-corrected) sample standard deviation. Returns `(0, 0)`
/// for fewer than two samples.
pub fn mean_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    if values.len() == 1 {
        return (values[0], 0.0);
    }
    let m = Moments::from_f32(values);
    (m.mean as f32, m.sample_std() as f32)
}

/// Sample skewness (Fisher-Pearson, bias-uncorrected).
pub fn skewness(values: &[f32]) -> f32 {
    if values.len() < 3 {
        return 0.0;
    }
    Moments::from_f32(values).skewness()
}

/// Sample excess kurtosis (≈ 0 for a normal distribution).
pub fn excess_kurtosis(values: &[f32]) -> f32 {
    if values.len() < 4 {
        return 0.0;
    }
    Moments::from_f32(values).excess_kurtosis()
}

/// Sarle's bimodality coefficient: `(skew^2 + 1) / (kurt + 3 (n-1)^2 / ((n-2)(n-3)))`,
/// where `kurt` is excess kurtosis. Values above ~0.555 (the BC of a uniform
/// distribution) are considered evidence of bimodality.
///
/// Returns 0 when the input has fewer than 4 samples.
pub fn bimodality_coefficient(values: &[f32]) -> f32 {
    if values.len() < 4 {
        return 0.0;
    }
    let n = values.len() as f64;
    let m = Moments::from_f32(values);
    let g = m.skewness() as f64;
    let k = m.excess_kurtosis() as f64;
    let correction = 3.0 * (n - 1.0).powi(2) / ((n - 2.0) * (n - 3.0));
    let denom = k + correction;
    if denom.abs() < 1e-12 {
        return 0.0;
    }
    ((g * g + 1.0) / denom) as f32
}

/// Sarle's bimodality coefficient is our primary unimodality test (cheap,
/// works well on amplitude/feature distributions). For more rigour, callers
/// can also run `ks_two_sample` between two candidate sub-distributions.
///
/// (A full Hartigan dip implementation is non-trivial and would require its
/// own crate-level dependency; intentionally omitted in favour of BC + KS,
/// which together cover the same diagnostic ground for this curation tool.)

/// Two-sample Kolmogorov–Smirnov statistic — the maximum absolute difference
/// between the empirical CDFs of `a` and `b`. Range `[0, 1]`; large values
/// indicate the two distributions don't come from the same population. The
/// asymptotic two-sided p-value at level α is `D > c(α) sqrt((n+m)/(n m))`.
///
/// Returns `0.0` if either input is empty.
pub fn ks_two_sample(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let mut a: Vec<f32> = a.to_vec();
    let mut b: Vec<f32> = b.to_vec();
    a.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    b.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let na = a.len() as f64;
    let nb = b.len() as f64;
    let mut i = 0usize;
    let mut j = 0usize;
    let mut d_max = 0.0_f64;
    while i < a.len() && j < b.len() {
        let x = a[i].min(b[j]);
        while i < a.len() && a[i] <= x {
            i += 1;
        }
        while j < b.len() && b[j] <= x {
            j += 1;
        }
        let fa = i as f64 / na;
        let fb = j as f64 / nb;
        let d = (fa - fb).abs();
        if d > d_max {
            d_max = d;
        }
    }
    d_max as f32
}

/// Asymptotic two-sided p-value for the KS statistic `d` from samples of
/// sizes `n` and `m`. Uses the standard Smirnov series:
/// `p ≈ 2 Σ_k (-1)^(k-1) exp(-2 k² λ²)` where `λ = sqrt(nm/(n+m)) d`.
///
/// Returns `1.0` for degenerate inputs.
pub fn ks_pvalue(d: f32, n: usize, m: usize) -> f32 {
    if n == 0 || m == 0 || d <= 0.0 {
        return 1.0;
    }
    let nf = n as f64;
    let mf = m as f64;
    let en = (nf * mf / (nf + mf)).sqrt();
    let lambda = (en + 0.12 + 0.11 / en) * d as f64;
    let lambda2 = lambda * lambda;
    let mut sum = 0.0_f64;
    let mut sign = 1.0_f64;
    for k in 1..=100 {
        let term = sign * (-2.0 * (k as f64).powi(2) * lambda2).exp();
        sum += term;
        if term.abs() < 1e-10 {
            break;
        }
        sign = -sign;
    }
    let p = (2.0 * sum).clamp(0.0, 1.0);
    p as f32
}

/// Allen-Institute-style amplitude cutoff: estimate the fraction of spikes
/// the sorter missed because their amplitudes fell below the detection
/// threshold. Method (mirror-tail estimator, equivalent to Hill 2011 §3.2):
///
///   1. Build a (lightly smoothed) histogram of `amps` with `n_bins` bins.
///   2. Find the modal bin `m`. The "fully-observed" half of the
///      distribution is at indices `>= m` (high amplitudes — never missed).
///   3. Mirror the high half around `m`: any mass beyond `2m` on the high
///      side has no counterpart on the low side, because the histogram
///      already starts at the *smallest observed* amplitude. That excess
///      mass is the estimated count of spikes that should have been
///      detected at low amplitudes but weren't.
///
/// Returns a fraction in `[0, 1]`. Symmetric distributions (mode near the
/// centre) score near 0; half-Gaussian distributions (mode at the lowest
/// bin, hard cutoff) score near 0.5; total clipping past the mode would
/// score higher, but that's pathological in practice.
pub fn amplitude_cutoff(amps: &[f32], n_bins: usize) -> f32 {
    let n = amps.len();
    if n < 8 || n_bins < 8 {
        return 0.0;
    }
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &v in amps {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    if hi - lo < 1e-9 {
        return 0.0;
    }
    let inv = n_bins as f32 / (hi - lo);
    let mut h = vec![0u32; n_bins];
    for &v in amps {
        let mut i = ((v - lo) * inv) as usize;
        if i >= n_bins {
            i = n_bins - 1;
        }
        h[i] += 1;
    }
    // 3-bin smoothing so single-sample peaks don't pick a misleading mode.
    let mut s = vec![0.0_f32; n_bins];
    for i in 0..n_bins {
        let lo_i = i.saturating_sub(1);
        let hi_i = (i + 1).min(n_bins - 1);
        let mut sum = 0.0_f32;
        let mut k = 0.0_f32;
        for b in lo_i..=hi_i {
            sum += h[b] as f32;
            k += 1.0;
        }
        s[i] = sum / k;
    }
    let (mode_bin, _) = s
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, 0.0));

    let total: f32 = s.iter().copied().sum();
    if total <= 0.0 {
        return 0.0;
    }
    // Bins beyond the mirror of bin 0 around the mode (i.e. indices > 2*m)
    // are unmirrored on the low-amplitude side → that mass is "missing".
    let mirror_end = 2 * mode_bin + 1;
    if mirror_end >= n_bins {
        return 0.0;
    }
    let unmirrored: f32 = s[mirror_end..].iter().copied().sum();
    (unmirrored / total).clamp(0.0, 1.0)
}

/// Quick percentile (linear interpolation between the two nearest ranks).
/// Sorts internally — pass a clone if you need to preserve order.
pub fn percentile(mut values: Vec<f32>, p: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p = p.clamp(0.0, 1.0);
    let idx = p * (values.len() - 1) as f32;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        values[lo]
    } else {
        let frac = idx - lo as f32;
        values[lo] * (1.0 - frac) + values[hi] * frac
    }
}

/// Hartigans' dip statistic — measures departure from unimodality.
///
/// V1 is a thin proxy: we use the *bimodality coefficient* (which already
/// lives in this module) as a stand-in. It rises with bimodal /
/// heavy-tailed distributions, which is the property the dip is checking
/// for, and avoids pulling in a heavyweight ECDF-based dip implementation
/// for now.
pub fn dip_statistic(values: &[f32]) -> f32 {
    bimodality_coefficient(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_std_handles_short_inputs() {
        assert_eq!(mean_std(&[]), (0.0, 0.0));
        assert_eq!(mean_std(&[3.0]), (3.0, 0.0));
        let (m, s) = mean_std(&[1.0, 2.0, 3.0, 4.0]);
        assert!((m - 2.5).abs() < 1e-6);
        assert!((s - (5.0_f32 / 3.0).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn skewness_zero_for_symmetric_inputs() {
        let v: Vec<f32> = (-50..=50).map(|i| i as f32 / 10.0).collect();
        assert!(skewness(&v).abs() < 1e-3);
    }

    #[test]
    fn skewness_positive_for_right_tail() {
        let v = vec![1.0_f32, 1.0, 1.0, 1.0, 1.0, 5.0, 10.0];
        assert!(skewness(&v) > 0.5);
    }

    #[test]
    fn excess_kurtosis_near_zero_for_normal_like() {
        // Crude box–muller draws from a fixed seed-like sequence.
        let mut v = Vec::with_capacity(2000);
        let mut s: u32 = 0xCAFEBABE;
        for _ in 0..1000 {
            // xorshift32 → uniform → BoxMuller
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let u1 = (s as f64 / u32::MAX as f64).clamp(1e-9, 1.0);
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let u2 = (s as f64 / u32::MAX as f64).clamp(1e-9, 1.0);
            let r = (-2.0 * u1.ln()).sqrt();
            let g = r * (2.0 * std::f64::consts::PI * u2).cos();
            let g2 = r * (2.0 * std::f64::consts::PI * u2).sin();
            v.push(g as f32);
            v.push(g2 as f32);
        }
        assert!(excess_kurtosis(&v).abs() < 0.5);
    }

    #[test]
    fn bimodality_coefficient_high_for_two_clouds() {
        let mut v = Vec::new();
        v.extend((0..500).map(|i| 0.0 + (i as f32) * 0.0))
            ;
        // Two means, equal mass.
        for _ in 0..500 {
            v.push(-1.0);
        }
        for _ in 0..500 {
            v.push(1.0);
        }
        assert!(bimodality_coefficient(&v) > 0.555);
    }

    #[test]
    fn bimodality_coefficient_low_for_single_cloud() {
        // A symmetric, near-Gaussian set — BC should be well under 0.555.
        let v: Vec<f32> = (-100..=100)
            .flat_map(|i| {
                let x = i as f32 / 30.0;
                let w = (-0.5 * x * x).exp();
                let n = (w * 100.0) as usize;
                std::iter::repeat(x).take(n)
            })
            .collect();
        assert!(bimodality_coefficient(&v) < 0.555);
    }

    #[test]
    fn ks_zero_for_identical_inputs() {
        let v = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        assert!(ks_two_sample(&v, &v) < 1e-6);
    }

    #[test]
    fn ks_one_for_disjoint_inputs() {
        let a: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let b: Vec<f32> = (200..300).map(|i| i as f32).collect();
        let d = ks_two_sample(&a, &b);
        assert!((d - 1.0).abs() < 1e-3);
    }

    #[test]
    fn ks_pvalue_bounded() {
        assert_eq!(ks_pvalue(0.0, 100, 100), 1.0);
        let p = ks_pvalue(0.5, 50, 50);
        assert!((0.0..=1.0).contains(&p));
    }

    #[test]
    fn amplitude_cutoff_zero_for_symmetric_distribution() {
        // Symmetric Gaussian-like — no cutoff expected.
        let mut v: Vec<f32> = Vec::new();
        for i in -100..=100 {
            let x = i as f32 / 30.0;
            let w = (-0.5 * x * x).exp();
            for _ in 0..(w * 200.0) as usize {
                v.push(x);
            }
        }
        assert!(amplitude_cutoff(&v, 30) < 0.05);
    }

    #[test]
    fn amplitude_cutoff_positive_when_tail_is_clipped() {
        // Half-Gaussian: only positive side, modal bin near 0.
        let mut v: Vec<f32> = Vec::new();
        for i in 0..=100 {
            let x = i as f32 / 30.0;
            let w = (-0.5 * x * x).exp();
            for _ in 0..(w * 200.0) as usize {
                v.push(x);
            }
        }
        assert!(amplitude_cutoff(&v, 30) > 0.1);
    }

    #[test]
    fn percentile_returns_extremes_at_boundaries() {
        let v = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(v.clone(), 0.0), 1.0);
        assert_eq!(percentile(v.clone(), 1.0), 5.0);
        assert!((percentile(v, 0.5) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn percentile_clamps_out_of_range_p() {
        let v = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(percentile(v.clone(), -1.0), 1.0);
        assert_eq!(percentile(v.clone(), 2.0), 3.0);
    }

    #[test]
    fn percentile_handles_empty() {
        assert_eq!(percentile(vec![], 0.5), 0.0);
    }

    #[test]
    fn mean_std_is_correct_for_known_input() {
        let v = [2.0_f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let (mean, std) = mean_std(&v);
        assert!((mean - 5.0).abs() < 1e-6);
        // Bessel-corrected sample stdev = sqrt(32/7) ≈ 2.138
        let expected = (32.0_f32 / 7.0).sqrt();
        assert!((std - expected).abs() < 1e-3);
    }

    #[test]
    fn mean_std_zero_for_constant_input() {
        let v = vec![3.14_f32; 100];
        let (mean, std) = mean_std(&v);
        assert!((mean - 3.14).abs() < 1e-5);
        assert!(std.abs() < 1e-5);
    }

    #[test]
    fn mean_std_handles_empty() {
        let (mean, std) = mean_std(&[]);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn skewness_is_zero_for_symmetric_distribution() {
        // Symmetric around 0.
        let v: Vec<f32> = (-50..=50).map(|i| i as f32).collect();
        let s = skewness(&v);
        assert!(s.abs() < 1e-3, "skewness {s} should be ~0 for symmetric input");
    }

    #[test]
    fn skewness_is_positive_for_right_tailed_distribution() {
        // Mostly small values, a few large outliers.
        let mut v: Vec<f32> = (0..100).map(|_| 1.0).collect();
        v.extend([10.0, 20.0, 50.0]);
        assert!(skewness(&v) > 0.5, "expected positive skew");
    }

    #[test]
    fn skewness_is_negative_for_left_tailed_distribution() {
        let mut v: Vec<f32> = (0..100).map(|_| 1.0).collect();
        v.extend([-10.0, -20.0, -50.0]);
        assert!(skewness(&v) < -0.5, "expected negative skew");
    }

    #[test]
    fn skewness_handles_empty_and_constant() {
        assert_eq!(skewness(&[]), 0.0);
        assert_eq!(skewness(&[5.0; 50]), 0.0);
    }

    #[test]
    fn excess_kurtosis_zero_for_normal_like_distribution() {
        // Box-Muller-ish synthetic Gaussian using a deterministic stream.
        let mut v: Vec<f32> = Vec::with_capacity(10_000);
        let mut s: u32 = 0x1234_5678;
        let next = |s: &mut u32| -> f32 {
            *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (*s as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        for _ in 0..5000 {
            // Sum of 12 uniforms ≈ N(0, 1)·sqrt(12/3) = N(0, 2). Good enough.
            let mut sum = 0.0f32;
            for _ in 0..12 {
                sum += next(&mut s);
            }
            v.push(sum);
        }
        let k = excess_kurtosis(&v);
        // Allow generous tolerance — finite sample + crude RNG.
        assert!(k.abs() < 1.0, "excess kurtosis {k} too far from 0");
    }

    #[test]
    fn excess_kurtosis_positive_for_heavy_tailed_distribution() {
        // Bimodal mixture with concentrated mass at extremes — high kurtosis.
        let mut v: Vec<f32> = vec![0.0; 1000];
        v.extend(std::iter::repeat(10.0).take(20));
        v.extend(std::iter::repeat(-10.0).take(20));
        assert!(
            excess_kurtosis(&v) > 1.0,
            "expected positive excess kurtosis"
        );
    }

    #[test]
    fn ks_two_sample_zero_for_identical_distributions() {
        let a: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let b: Vec<f32> = a.clone();
        let d = ks_two_sample(&a, &b);
        assert!(d < 1e-5, "KS for identical distributions should be ~0");
    }

    #[test]
    fn ks_two_sample_positive_for_disjoint_ranges() {
        let a: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let b: Vec<f32> = (200..300).map(|i| i as f32).collect();
        // Fully disjoint: D = 1.0.
        let d = ks_two_sample(&a, &b);
        assert!((d - 1.0).abs() < 1e-3, "expected D ~= 1, got {d}");
    }

    #[test]
    fn ks_two_sample_handles_empty_inputs() {
        assert_eq!(ks_two_sample(&[], &[1.0]), 0.0);
        assert_eq!(ks_two_sample(&[1.0], &[]), 0.0);
        assert_eq!(ks_two_sample(&[], &[]), 0.0);
    }

    #[test]
    fn ks_pvalue_at_zero_is_one() {
        assert_eq!(ks_pvalue(0.0, 50, 50), 1.0);
    }

    #[test]
    fn ks_pvalue_decreases_with_d() {
        let p_low = ks_pvalue(0.1, 100, 100);
        let p_high = ks_pvalue(0.5, 100, 100);
        assert!(p_high <= p_low, "p decreases with larger D");
    }

    #[test]
    fn ks_pvalue_in_unit_range() {
        for d in [0.0_f32, 0.1, 0.3, 0.5, 0.9, 1.0] {
            let p = ks_pvalue(d, 50, 50);
            assert!((0.0..=1.0).contains(&p), "p={p} out of range for d={d}");
        }
    }

    #[test]
    fn bimodality_coefficient_higher_for_bimodal_than_unimodal() {
        // Single peak.
        let uni: Vec<f32> = (-30..=30).map(|i| (i as f32) * 0.1).collect();
        // Two well-separated peaks.
        let mut bim: Vec<f32> = vec![-5.0; 50];
        bim.extend(vec![5.0; 50]);

        let b_uni = bimodality_coefficient(&uni);
        let b_bim = bimodality_coefficient(&bim);
        assert!(
            b_bim > b_uni,
            "bimodal {b_bim} should exceed unimodal {b_uni}",
        );
    }

    #[test]
    fn percentile_is_monotonic_in_p() {
        let mut v: Vec<f32> = (0..100).map(|i| i as f32).collect();
        // Shuffle deterministically.
        v.swap(0, 50);
        v.swap(2, 27);
        let p_25 = percentile(v.clone(), 0.25);
        let p_50 = percentile(v.clone(), 0.50);
        let p_75 = percentile(v.clone(), 0.75);
        assert!(p_25 <= p_50 && p_50 <= p_75, "{p_25} {p_50} {p_75}");
    }

    #[test]
    fn dip_statistic_returns_finite_for_typical_inputs() {
        let v: Vec<f32> = (0..200).map(|i| (i as f32) * 0.05).collect();
        let d = dip_statistic(&v);
        assert!(d.is_finite());
    }
}
