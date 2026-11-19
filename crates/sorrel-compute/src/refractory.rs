//! Statistical evidence from refractory-period violations.

use sorrel_io::SampleIndex;

use crate::{linalg::ln_gamma, RefractoryParams};

#[derive(Clone, Copy, Debug)]
pub struct SlidingRp {
    /// Smallest confirmable contamination fraction, or `NaN`.
    pub min_contamination: f32,
    /// Highest-confidence window at that fraction (first on ties), as the
    /// right sample edge in milliseconds, or `NaN`.
    pub rp_ms: f32,
    /// Whether any sweep window was usable; `false` makes `NaN` "not tested"
    /// rather than "nothing passed".
    pub tested: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct ContaminationPosterior {
    pub mean: f32,
    pub lo95: f32,
    pub hi95: f32,
    /// Observed pairs, capped at `u32::MAX` for reporting; zero if unavailable.
    pub violations: u32,
}

const NO_SRP: SlidingRp = SlidingRp {
    min_contamination: f32::NAN,
    rp_ms: f32::NAN,
    tested: false,
};
const NO_POSTERIOR: ContaminationPosterior = ContaminationPosterior {
    mean: f32::NAN,
    lo95: f32::NAN,
    hi95: f32::NAN,
    violations: 0,
};

/// IBL sliding-RP test (Roth et al., 2026 preprint, doi:10.64898/2026.03.06.710130).
/// Tests 0.5% through 35% in 0.5% steps, requiring nominal Poisson confidence
/// strictly above 90% (no multiple-comparisons correction). Unlike IBL, which
/// tests every sample bin above 0.5 ms, this uses the session sweep (by default
/// 0.75–10 ms by 0.25 ms), rounded to sample edges, and charges a window of
/// `e` samples `e - 1` samples of exposure. At the smallest passing fraction
/// it reports the window of highest confidence, IBL's refractory-period
/// estimate. Assumes diffuse (Poisson) contamination. Both are `NaN` if
/// nothing passes (too few spikes, or contamination above 35%), there are
/// fewer than two spikes, or the recording duration or rate is invalid.
/// `times` must be sorted in ascending sample order.
pub fn sliding_rp(
    times: &[SampleIndex],
    total_samples: u64,
    sr: f32,
    params: &RefractoryParams,
) -> SlidingRp {
    compute_evidence(times, total_samples, sr, sweep_edges(params, sr), 0).0
}

/// Fixed-window contamination posterior with a uniform Beta(1,1) prior on
/// [0, 1], a Poisson likelihood, and the shortest 95% credible interval.
/// Assumes the tested window is no longer than the neuron's true refractory
/// period, stationary rates and diffuse (Poisson) contamination. An interval
/// from near 0 to high values means too little data to tell; one bounded well
/// above 0 is wide because Ve flattens as C nears 1. Fractions are `NaN` with
/// fewer than two spikes or invalid
/// recording duration, rate, or fixed window. `times` must be sorted.
pub fn contamination_posterior(
    times: &[SampleIndex],
    total_samples: u64,
    sr: f32,
    params: &RefractoryParams,
) -> ContaminationPosterior {
    compute_evidence(times, total_samples, sr, Vec::new(), fixed_edge(params, sr)).1
}

/// Compute [`sliding_rp`] and [`contamination_posterior`] together with a
/// single pass over the pairs in the sorted spike train.
pub fn refractory_evidence(
    times: &[SampleIndex],
    total_samples: u64,
    sr: f32,
    params: &RefractoryParams,
) -> (SlidingRp, ContaminationPosterior) {
    compute_evidence(
        times,
        total_samples,
        sr,
        sweep_edges(params, sr),
        fixed_edge(params, sr),
    )
}

fn sweep_edges(params: &RefractoryParams, sr: f32) -> Vec<u64> {
    if params.validate().is_err() {
        return Vec::new();
    }
    // A 1-sample window holds no positive lag, so its zero count is no evidence.
    let mut edges: Vec<_> = params.windows(sr).filter(|&edge| edge > 1).collect();
    edges.dedup();
    edges
}

fn fixed_edge(params: &RefractoryParams, sr: f32) -> u64 {
    match params.fixed_ms.is_finite() {
        // Below two samples no positive lag fits: the posterior is unavailable.
        true => Some(params.fixed_samples(sr))
            .filter(|&e| e > 1)
            .unwrap_or(0),
        false => 0,
    }
}

fn compute_evidence(
    times: &[SampleIndex],
    total_samples: u64,
    sr: f32,
    edges: Vec<u64>,
    fixed: u64,
) -> (SlidingRp, ContaminationPosterior) {
    if times.len() < 2 || total_samples == 0 || !sr.is_finite() || sr <= 0.0 {
        return (NO_SRP, NO_POSTERIOR);
    }
    let n = times.len() as f64;
    // A window as long as the recording holds every pair, so more width
    // cannot add counts while Ve keeps growing: such windows are no evidence.
    // Ve's 2·τ/D pair probability also exceeds 1 past half the recording.
    let fits = |e: u64| e.saturating_mul(2) < total_samples;
    let edges: Vec<u64> = edges.into_iter().filter(|&e| fits(e)).collect();
    let fixed = if fits(fixed) { fixed } else { 0 };
    // Exposure / duration. Integer lags 1..edge-1 see edge - 1 samples of
    // continuous lag (uniform sub-sample phase), not edge: charging the full
    // edge, as IBL does, understates contamination on coarse timestamps.
    // The sample rate cancels.
    let ratio = |edge: u64| (edge - 1) as f64 / total_samples as f64;
    let (counts, fixed_count) = violation_counts(times, &edges, fixed);
    let srp = if edges.is_empty() {
        NO_SRP
    } else {
        let grid: [f64; 70] = std::array::from_fn(|i| (i + 1) as f64 / 200.0);
        // Among windows passing at `c`, the one of highest confidence, as in
        // IBL `pass_slidingRP_confmat` (argmax, first on ties). Compare log
        // P(X <= V) so confidences that round to 1.0 still order.
        let best_edge = |c| {
            edges
                .iter()
                .zip(&counts)
                .map(|(&edge, &v)| {
                    (
                        edge,
                        poisson_log_cdf(v, expected_violations(n, ratio(edge), c)),
                    )
                })
                .filter(|&(_, log_cdf)| -log_cdf.exp_m1() > 0.9)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(edge, _)| edge)
        };
        // Expected violations increase with C throughout the tested grid.
        let i = grid.partition_point(|&c| best_edge(c).is_none());
        grid.get(i)
            .and_then(|&c| {
                best_edge(c).map(|edge| SlidingRp {
                    min_contamination: c as f32,
                    rp_ms: (edge as f64 * 1000.0 / f64::from(sr)) as f32,
                    tested: true,
                })
            })
            .unwrap_or(SlidingRp {
                tested: true,
                ..NO_SRP
            })
    };
    let posterior = if fixed > 0 {
        posterior_from_count(fixed_count, n, ratio(fixed))
    } else {
        NO_POSTERIOR
    };
    (srp, posterior)
}

/// Count positive lags in `[0, edge)` as in IBL `computeACG`: each pair
/// contributes once, not just adjacent spikes, and zero lags are excluded.
/// Bucket by the first right edge above the lag, then prefix-sum. This is
/// O(N + pairs) for the bounded sweep (at most 1000 edges, ~10 comparisons
/// per pair), with no allocation proportional to the largest lag in samples.
fn violation_counts(times: &[SampleIndex], edges: &[u64], fixed: u64) -> (Vec<u64>, u64) {
    let mut bins = vec![0_u64; edges.len()];
    let mut fixed_count = 0_u64;
    let sweep_max = edges.last().copied().unwrap_or(0);
    let max_lag = sweep_max.max(fixed);
    if max_lag <= 1 {
        return (bins, 0);
    }
    let mut i = 0;
    while i < times.len() {
        let t = times[i].0;
        let mut j = i + 1;
        while j < times.len() && times[j].0 == t {
            j += 1;
        }
        let multiplicity = (j - i) as u64;
        for later in &times[j..] {
            let lag = later.0 - t;
            if lag >= max_lag {
                break;
            }
            if lag < fixed {
                fixed_count = fixed_count.saturating_add(multiplicity);
            }
            if lag < sweep_max {
                let k = edges.partition_point(|&edge| edge <= lag);
                bins[k] = bins[k].saturating_add(multiplicity);
            }
        }
        i = j;
    }
    let mut cumulative = 0_u64;
    for count in &mut bins {
        cumulative = cumulative.saturating_add(*count);
        *count = cumulative;
    }
    (bins, fixed_count)
}

/// Llobet et al. (2022, doi:10.1101/2022.02.08.479192), with both
/// contaminant/base and contaminant/contaminant pairs.
fn expected_violations(n: f64, tau_over_duration: f64, c: f64) -> f64 {
    let nc = c * n;
    let nb = (1.0 - c) * n;
    2.0 * tau_over_duration * nc * (nb + (nc - 1.0) / 2.0)
}

fn poisson_log_pmf(k: f64, mean: f64) -> f64 {
    if k == 0.0 {
        -mean
    } else {
        k * mean.ln() - mean - ln_gamma(k + 1.0)
    }
}

/// P(X > v).
#[cfg(test)]
fn poisson_confidence(v: u64, mean: f64) -> f64 {
    -poisson_log_cdf(v, mean).exp_m1()
}

/// ln P(X <= v). Sum the shorter tail relative to its largest mass, then
/// restore scale in log space; no exp(-mean) seed that could underflow.
fn poisson_log_cdf(v: u64, mean: f64) -> f64 {
    if v == 0 {
        return -mean;
    }
    let mut term = 1.0;
    let mut sum = 1.0;
    let k = v as f64;
    if mean > k {
        for j in (1..=v).rev() {
            term *= j as f64 / mean;
            sum += term;
            if term <= f64::EPSILON * sum {
                break;
            }
        }
        (poisson_log_pmf(k, mean) + sum.ln()).min(0.0)
    } else {
        for j in v + 2.. {
            term *= mean / j as f64;
            sum += term;
            if term <= f64::EPSILON * sum {
                break;
            }
        }
        (-(poisson_log_pmf(k + 1.0, mean) + sum.ln()).min(0.0).exp()).ln_1p()
    }
}

fn posterior_from_count(violations: u64, n: f64, tau_over_duration: f64) -> ContaminationPosterior {
    const STEPS: usize = 2000;
    // Integration stops where the density falls to e^-40 of its peak; with a
    // unimodal density the mass beyond is negligible but not formally bounded.
    const SPAN: f64 = 40.0;
    let k = violations as f64;
    // Log-likelihood without the log-factorial, constant in C. With a uniform
    // prior this is the log posterior. Ve grows with C up to 1 - 1/(2N) and
    // dips slightly above it; treating it as unimodal there is an approximation.
    let ll = |c: f64| {
        let mean = expected_violations(n, tau_over_duration, c);
        if k == 0.0 {
            -mean
        } else {
            k * mean.ln() - mean
        }
    };
    // 100 halvings of [0, 1] reach f64 resolution.
    let bisect = |mut lo: f64, mut hi: f64, below: &dyn Fn(f64) -> bool| {
        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            if below(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    };
    // Integrate only where the posterior is within SPAN nats of its peak
    // (Ve(C) = k), so a narrow peak near 0 on a long recording is resolved.
    let c_peak = match k == 0.0 {
        true => 0.0,
        // Ve rises until 1 - 1/(2N), so bisect only where it is monotone.
        false => bisect(0.0, 1.0 - 0.5 / n, &|c| {
            expected_violations(n, tau_over_duration, c) < k
        }),
    };
    let top = ll(c_peak);
    let c_lo = match c_peak == 0.0 {
        true => 0.0,
        false => bisect(0.0, c_peak, &|c| ll(c) < top - SPAN),
    };
    let c_hi = bisect(c_peak, 1.0, &|c| ll(c) >= top - SPAN);
    if c_hi <= c_lo {
        return ContaminationPosterior {
            mean: c_peak as f32,
            lo95: c_peak as f32,
            hi95: c_peak as f32,
            violations: violations.min(u64::from(u32::MAX)) as u32,
        };
    }
    let fraction = |i: usize| c_lo + (c_hi - c_lo) * i as f64 / STEPS as f64;
    // Subtract the peak before exponentiating so a tiny likelihood normalises.
    let weights: [f64; STEPS + 1] = std::array::from_fn(|i| (ll(fraction(i)) - top).exp());

    // Trapezoidal integration (common factor 0.5 / STEPS cancels).
    let mut cumulative = [0.0; STEPS + 1];
    let mut moment = 0.0;
    for i in 1..=STEPS {
        cumulative[i] = cumulative[i - 1] + weights[i - 1] + weights[i];
        moment += fraction(i - 1) * weights[i - 1] + fraction(i) * weights[i];
    }
    let mass = cumulative[STEPS];
    let quantile = |p| {
        let target = p * mass;
        let i = cumulative.partition_point(|&x| x < target).max(1);
        let t = (target - cumulative[i - 1]) / (cumulative[i] - cumulative[i - 1]);
        c_lo + (c_hi - c_lo) * (i as f64 - 1.0 + t) / STEPS as f64
    };
    // Shortest contiguous 95% interval, [q(a), q(a + 0.95)] over a in 0..0.05
    // (the highest-density one wherever the density is unimodal). With no
    // violations the density falls from C = 0, so it is [0, q(0.95)].
    let (lo95, hi95) = match k == 0.0 {
        true => (0.0, quantile(0.95)),
        false => (0..=100)
            .map(|j| {
                let a = 0.05 * j as f64 / 100.0;
                (quantile(a), quantile(a + 0.95))
            })
            .min_by(|x, y| (x.1 - x.0).total_cmp(&(y.1 - y.0)))
            .unwrap_or((f64::NAN, f64::NAN)),
    };
    ContaminationPosterior {
        mean: (moment / mass) as f32,
        lo95: lo95 as f32,
        hi95: hi95 as f32,
        violations: violations.min(u64::from(u32::MAX)) as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 32768.0;

    fn train(duration: u64, rate: f64, inject: bool) -> Vec<SampleIndex> {
        let sr = f64::from(SR);
        let end = duration * SR as u64;
        let mut state = 42_u32;
        let mut uniform = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (f64::from(state) + 0.5) / 4294967296.0
        };
        let mut times = Vec::new();
        let mut t = 0;
        loop {
            t += (0.002 * sr).ceil() as u64 + (-uniform().ln() * sr / rate) as u64;
            if t >= end {
                break;
            }
            times.push(SampleIndex(t));
        }
        if inject {
            times
                .extend((0..times.len() / 9).map(|_| SampleIndex((uniform() * end as f64) as u64)));
            times.sort_unstable();
        }
        times
    }

    #[test]
    fn ibl_nominal_matrix_parity() {
        // Seeded dead-time Poisson trains (2 ms dead time). Reference values
        // from IBL slidingRefractory python/slidingRP/metrics.py (main, 2026-11-19).
        // Pair counts and each `cell` (Ve at the full edge, Poisson
        // confidence) match IBL computeACG / computeViol exactly. The
        // end-to-end `expected` results charge edge - 1 samples of exposure
        // instead of IBL's full edge; the fractions and the argmax window
        // bin (66, whose right edge Sorrel reports as 2.01 ms; IBL reports
        // its centre) still equal IBL's on these trains.
        let params = RefractoryParams::default();
        for (duration, rate, inject, n, expected, cell) in [
            (
                1800,
                25.0,
                false,
                43092,
                Some((0.005, 66)),
                (0.005, 57, 0, 0.9999999831686436),
            ),
            (
                1800,
                25.0,
                true,
                47880,
                Some((0.105, 66)),
                (0.105, 57, 405, 0.9551630055080321),
            ),
            (
                100,
                0.5,
                false,
                51,
                None,
                (0.35, 328, 0, 0.13805834316210686),
            ),
        ] {
            let times = train(duration, rate, inject);
            assert_eq!(times.len(), n);
            let total = duration * SR as u64;
            let srp = sliding_rp(&times, total, SR, &params);
            if let Some((contamination, edge)) = expected {
                assert_eq!(srp.min_contamination, contamination);
                assert_eq!(srp.rp_ms, edge as f32 * 1000.0 / SR);
            } else {
                assert!(srp.min_contamination.is_nan() && srp.rp_ms.is_nan());
            }
            let (c, edge, v, confidence) = cell;
            let (counts, _) = violation_counts(&times, &[edge], 0);
            assert_eq!(counts, [v]);
            let lambda = expected_violations(n as f64, edge as f64 / total as f64, c);
            assert!((poisson_confidence(v, lambda) - confidence).abs() < 1e-10);
        }
    }

    #[test]
    fn count_all_positive_lags_strictly_inside_window() {
        let times = [0, 0, 1, 2, 4].map(SampleIndex);
        assert_eq!(
            violation_counts(&times, &[1, 2, 3, 4, 5], 3),
            (vec![0, 3, 6, 7, 9], 6)
        );
    }

    #[test]
    fn posterior_exposes_low_power_and_recovers_injected_fraction() {
        let params = RefractoryParams::default();
        let posterior = |duration, rate, inject| {
            contamination_posterior(
                &train(duration, rate, inject),
                duration * SR as u64,
                SR,
                &params,
            )
        };
        let sparse = posterior(100, 0.5, false);
        assert_eq!(sparse.violations, 0);
        assert!(sparse.hi95 > 0.1 && sparse.mean > 0.1, "{sparse:?}");
        let clean = posterior(1800, 25.0, false);
        assert_eq!(clean.violations, 0);
        assert!(clean.hi95 < 0.02, "{clean:?}");
        let contaminated = posterior(1800, 25.0, true);
        assert_eq!(contaminated.violations, 352);
        assert!(
            contaminated.lo95 < 0.1 && 0.1 < contaminated.hi95,
            "{contaminated:?}"
        );
        // True contamination is exactly 10% (N/9 injected); the e - 1
        // exposure puts the mean within 0.1 point of it.
        assert!((contaminated.mean - 0.0996154).abs() < 1e-5);
        assert!((contaminated.lo95 - 0.0887504).abs() < 1e-5);
        assert!((contaminated.hi95 - 0.1106647).abs() < 1e-5);
    }

    #[test]
    fn insufficient_or_invalid_input_is_not_clean() {
        let params = RefractoryParams::default();
        let times = [0, 100].map(SampleIndex);
        for (times, total, sr) in [
            (&times[..1], 1000, SR),
            (&times[..], 0, SR),
            (&times[..], 1000, f32::NAN),
            (&times[..], 1000, 0.0),
        ] {
            let srp = sliding_rp(times, total, sr, &params);
            assert!(srp.min_contamination.is_nan() && srp.rp_ms.is_nan() && !srp.tested);
            let p = contamination_posterior(times, total, sr, &params);
            assert!(p.mean.is_nan() && p.lo95.is_nan() && p.hi95.is_nan());
        }
    }

    #[test]
    fn narrow_posterior_near_zero_is_resolved() {
        // 50 Hz for an hour, no violations at 1.5 ms: Ve(C) ~ a*C for tiny C,
        // so the posterior is exponential with rate a.
        let (n, r) = (180_000.0, 0.0015 / 3600.0);
        let a = 2.0 * r * n * (n - 0.5);
        let p = posterior_from_count(0, n, r);
        let near = |got: f32, want: f64| ((f64::from(got) - want) / want).abs() < 0.01;
        assert!(near(p.mean, 1.0 / a), "{p:?}");
        // Highest-density interval of a decreasing density starts at 0.
        assert_eq!(p.lo95, 0.0, "{p:?}");
        // Sparse, zero violations: still starts at 0 (f32 rounding once
        // made a shifted interval look shorter).
        let three = [0, 100, 200].map(SampleIndex);
        let params = RefractoryParams::default();
        let (srp, sparse) = refractory_evidence(&three, 3600 * 30_000, 30_000.0, &params);
        assert_eq!(sparse.lo95, 0.0, "{sparse:?}");
        assert!(srp.tested && srp.min_contamination.is_nan(), "{srp:?}");
        assert!(near(p.hi95, -(0.05_f64).ln() / a), "{p:?}");
    }

    #[test]
    fn one_sample_windows_are_not_evidence() {
        // Doublets one sample apart at 1 kHz: a 1 ms window rounds to one
        // sample and cannot see them, so it must not read as clean.
        let times: Vec<_> = (0..3000_u64)
            .flat_map(|i| [i * 1000, i * 1000 + 1])
            .map(SampleIndex)
            .collect();
        let params = RefractoryParams {
            fixed_ms: 1.0,
            ..Default::default()
        };
        let (srp, post) = refractory_evidence(&times, 3_000_000, 1000.0, &params);
        assert!(srp.min_contamination.is_nan(), "{srp:?}");
        assert!(post.mean.is_nan(), "{post:?}");
    }
}
