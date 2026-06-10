//! Per-cluster quality metrics — phylib parity for the basic ones.
//!
//! All metrics here are *per-cluster scalars*: the inputs are a cluster's
//! spike times (and optionally amplitudes), the outputs are a single number
//! the cluster table can show or a quality scorer can rank by.

use sorrel_io::SampleIndex;

/// Cross-correlogram between two spike trains: for every spike in `a`, count
/// spikes in `b` whose Δt falls within ±`max_lag_samples`, binned uniformly.
///
/// Both inputs must be sorted ascending. The algorithm is two-pointer over
/// the sorted `b` train so total cost is `O(|a| + sum_of_window_counts)`,
/// which is what phy uses internally.
pub fn cross_correlogram(
    a: &[SampleIndex],
    b: &[SampleIndex],
    max_lag_samples: u64,
    bins: usize,
) -> Vec<u32> {
    let mut h = vec![0u32; bins];
    if max_lag_samples == 0 || bins == 0 || a.is_empty() || b.is_empty() {
        return h;
    }
    let span = 2.0_f64 * max_lag_samples as f64;
    let inv_bin = bins as f64 / span;
    let max_lag = max_lag_samples;

    let mut lo = 0usize;
    let mut hi = 0usize;
    for &ta in a {
        // Advance the window's lower edge past spikes that are more than
        // `max_lag` *before* `ta`. We rephrase to avoid u64 underflow.
        while lo < b.len() && b[lo].0 + max_lag < ta.0 {
            lo += 1;
        }
        if hi < lo {
            hi = lo;
        }
        while hi < b.len() && b[hi].0 <= ta.0.saturating_add(max_lag) {
            hi += 1;
        }
        for &tb in &b[lo..hi] {
            let dt = tb.as_i64() - ta.as_i64();
            let centred = dt as f64 + max_lag as f64;
            let mut idx = (centred * inv_bin) as usize;
            if idx >= bins {
                idx = bins - 1;
            }
            h[idx] += 1;
        }
    }
    h
}

/// Auto-correlogram: like `cross_correlogram(times, times, ...)` but
/// excludes the diagonal (a spike compared with itself).
pub fn auto_correlogram(times: &[SampleIndex], max_lag_samples: u64, bins: usize) -> Vec<u32> {
    let mut h = vec![0u32; bins];
    if max_lag_samples == 0 || bins == 0 || times.is_empty() {
        return h;
    }
    let span = 2.0_f64 * max_lag_samples as f64;
    let inv_bin = bins as f64 / span;
    let max_lag = max_lag_samples;

    let mut lo = 0usize;
    let mut hi = 0usize;
    for (i, &ta) in times.iter().enumerate() {
        while lo < times.len() && times[lo].0 + max_lag < ta.0 {
            lo += 1;
        }
        if hi < lo {
            hi = lo;
        }
        while hi < times.len() && times[hi].0 <= ta.0.saturating_add(max_lag) {
            hi += 1;
        }
        for (j, &tb) in times[lo..hi].iter().enumerate() {
            if lo + j == i {
                continue; // exclude self
            }
            let dt = tb.as_i64() - ta.as_i64();
            let centred = dt as f64 + max_lag as f64;
            let mut idx = (centred * inv_bin) as usize;
            if idx >= bins {
                idx = bins - 1;
            }
            h[idx] += 1;
        }
    }
    h
}

/// Number of inter-spike intervals shorter than `refractory_samples`.
///
/// Both inputs are in *samples*, not seconds, so the caller doesn't need to
/// thread a sample rate through. `spike_times` must be sorted ascending.
///
/// # Examples
///
/// ```
/// use sorrel_compute::isi_violations;
/// use sorrel_io::SampleIndex;
///
/// // Spikes at samples 0, 5, 30, 35, 100 with refractory window 10:
/// //   intervals 5, 25, 5, 65 → two intervals are < 10.
/// let times = [0, 5, 30, 35, 100].map(SampleIndex);
/// assert_eq!(isi_violations(&times, 10), 2);
/// ```
pub fn isi_violations(spike_times: &[SampleIndex], refractory_samples: u64) -> usize {
    if spike_times.len() < 2 || refractory_samples == 0 {
        return 0;
    }
    spike_times
        .windows(2)
        .filter(|w| w[1].0.saturating_sub(w[0].0) < refractory_samples)
        .count()
}

/// ISI violations per second, useful as a noise indicator (legitimate
/// neurons fire well below this rate after the refractory period).
pub fn isi_violation_rate(
    spike_times: &[SampleIndex],
    refractory_samples: u64,
    sample_rate: f32,
) -> f32 {
    if spike_times.len() < 2 || sample_rate <= 0.0 {
        return 0.0;
    }
    let n = isi_violations(spike_times, refractory_samples);
    let duration_s = (spike_times[spike_times.len() - 1]
        .0
        .saturating_sub(spike_times[0].0)) as f32
        / sample_rate;
    if duration_s <= 0.0 {
        return 0.0;
    }
    n as f32 / duration_s
}

/// Hill et al. (2011) estimate of the refractory-period contamination
/// fraction `f_p` — the fraction of spikes in the cluster that are false
/// positives (from other units), inferred from how many spike pairs fall
/// inside the refractory window. `0.0` means no detectable contamination;
/// values approach (and are clamped at) `1.0` for heavily contaminated units.
/// Returns 0 when too few spikes to estimate.
///
/// The estimator inverts the expected-violation relation
/// `N_viol = 2·t_ref·N²·f_p·(1−f_p) / T` (Hill et al. 2011, J. Neurosci.
/// 31:8699). We use the small-contamination linearisation
/// `f_p ≈ N_viol·T / (2·t_ref·N²)`; the full quadratic has no real root once
/// the observed violation count implies `f_p > 0.5`, in which case the unit
/// is at least half contaminated and we saturate to `1.0`.
///
/// Note the `N²` denominator: `N_viol` grows with the number of spike *pairs*,
/// so the contamination *fraction* normalises by `N²`, not `N`. The `N²` term
/// is accumulated in `f64` because it overflows `f32` for large clusters.
///
/// `refractory_samples` is the length of the refractory window in samples;
/// `total_duration_samples` is the recording's total length.
pub fn refractory_contamination(
    spike_times: &[SampleIndex],
    refractory_samples: u64,
    total_duration_samples: u64,
    sample_rate: f32,
) -> f32 {
    if spike_times.len() < 2 || refractory_samples == 0 || total_duration_samples == 0 {
        return 0.0;
    }
    let n = spike_times.len() as f64;
    let viol = isi_violations(spike_times, refractory_samples) as f64;
    let total_s = total_duration_samples as f64 / sample_rate as f64;
    let rp_s = refractory_samples as f64 / sample_rate as f64;
    if total_s <= 0.0 || rp_s <= 0.0 {
        return 0.0;
    }
    // Hill 2011 linearised false-positive fraction: N_viol·T / (2·t_ref·N²).
    let f_p = viol * total_s / (2.0 * rp_s * n * n);
    // f_p > 0.5 means the quadratic has no real root → at least half
    // contaminated; clamp to the valid [0, 1] fraction range.
    (f_p as f32).clamp(0.0, 1.0)
}

/// Llobet et al. (2022) sliding-refractory minimum contamination.
///
/// The true biological refractory period of a unit is unknown and varies, so
/// estimating contamination at a single fixed window is fragile. This sweeps
/// a range of candidate refractory periods `[min, max]` in `n_steps` and
/// returns the **smallest** Hill false-positive fraction across them — the
/// most likely true contamination, since the window that best matches the
/// unit's real refractory period yields the cleanest (lowest) estimate.
///
/// Returns 0 when there are too few spikes or the window range is degenerate.
/// The result is the same `[0, 1]` fraction as [`refractory_contamination`].
pub fn min_contamination_sliding_refractory(
    spike_times: &[SampleIndex],
    min_refractory_samples: u64,
    max_refractory_samples: u64,
    n_steps: usize,
    total_duration_samples: u64,
    sample_rate: f32,
) -> f32 {
    if spike_times.len() < 2
        || n_steps == 0
        || total_duration_samples == 0
        || max_refractory_samples < min_refractory_samples
        || max_refractory_samples == 0
    {
        return 0.0;
    }
    let lo = min_refractory_samples.max(1);
    let hi = max_refractory_samples.max(lo);
    let mut min_contam = f32::INFINITY;
    for step in 0..n_steps {
        // Linearly spaced refractory windows from lo to hi (inclusive).
        let rp = if n_steps == 1 {
            hi
        } else {
            lo + ((hi - lo) * step as u64) / (n_steps as u64 - 1)
        };
        let c = refractory_contamination(spike_times, rp, total_duration_samples, sample_rate);
        if c < min_contam {
            min_contam = c;
        }
    }
    if min_contam.is_finite() {
        min_contam
    } else {
        0.0
    }
}

/// Fraction of `n_bins` evenly-sized time bins that contain at least one
/// spike. Closer to 1.0 indicates a unit that fires across the whole
/// recording (good); closer to 0.0 indicates a unit only present in part of
/// the recording (often drift / electrode contact loss).
pub fn presence_ratio(
    spike_times: &[SampleIndex],
    total_duration_samples: u64,
    n_bins: usize,
) -> f32 {
    if spike_times.is_empty() || n_bins == 0 || total_duration_samples == 0 {
        return 0.0;
    }
    let bin_width = (total_duration_samples as f64 / n_bins as f64).max(1.0);
    let mut filled = vec![false; n_bins];
    for &t in spike_times {
        // Clamp the final bin so a spike landing exactly at the recording end
        // (idx == n_bins) is counted in the last bin rather than dropped.
        let idx = ((t.as_f64() / bin_width) as usize).min(n_bins - 1);
        filled[idx] = true;
    }
    let count = filled.iter().filter(|&&b| b).count();
    count as f32 / n_bins as f32
}

/// Mean amplitude across the cluster's spikes. Returns 0 for an empty input.
pub fn mean_amplitude(amps: &[f32]) -> f32 {
    if amps.is_empty() {
        return 0.0;
    }
    amps.iter().sum::<f32>() / amps.len() as f32
}

/// Sample standard deviation of amplitudes (Bessel-corrected). Returns 0
/// for fewer than 2 samples.
pub fn std_amplitude(amps: &[f32]) -> f32 {
    if amps.len() < 2 {
        return 0.0;
    }
    let mean = mean_amplitude(amps);
    let var = amps.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / (amps.len() - 1) as f32;
    var.sqrt()
}

/// Crude amplitude SNR: |mean| / std. Returns 0 when std is 0 or input is
/// empty. This is *not* the formal isolation SNR phylib computes from
/// templates — that one needs full waveform extraction.
pub fn amplitude_snr(amps: &[f32]) -> f32 {
    if amps.len() < 2 {
        return 0.0;
    }
    let m = crate::distribution::Moments::from_f32(amps);
    let s = m.sample_std();
    if s <= 0.0 {
        return 0.0;
    }
    (m.mean.abs() / s) as f32
}

/// Fraction of amplitudes below `threshold`. phylib's amplitude-cutoff
/// metric estimates the fraction of *missed* spikes by looking at the
/// shape of the amplitude distribution; this is a much cheaper proxy that's
/// useful as a column in the cluster table.
pub fn fraction_below(amps: &[f32], threshold: f32) -> f32 {
    if amps.is_empty() {
        return 0.0;
    }
    let n = amps.iter().filter(|&&a| a < threshold).count();
    n as f32 / amps.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn si<const N: usize>(xs: [u64; N]) -> [SampleIndex; N] {
        xs.map(SampleIndex)
    }

    fn siv(xs: impl IntoIterator<Item = u64>) -> Vec<SampleIndex> {
        xs.into_iter().map(SampleIndex).collect()
    }

    #[test]
    fn isi_violations_counts_short_intervals_only() {
        // Spikes at t = 0, 5, 30, 35, 100. Refractory = 10.
        // Intervals: 5 (viol), 25 (ok), 5 (viol), 65 (ok). -> 2.
        let times = si([0, 5, 30, 35, 100]);
        assert_eq!(isi_violations(&times, 10), 2);
    }

    #[test]
    fn isi_violations_handles_short_inputs() {
        assert_eq!(isi_violations(&[], 10), 0);
        assert_eq!(isi_violations(&si([5]), 10), 0);
    }

    #[test]
    fn isi_violations_zero_window_is_zero() {
        assert_eq!(isi_violations(&si([0, 5, 10]), 0), 0);
    }

    #[test]
    fn isi_violation_rate_returns_zero_for_pathological_input() {
        assert_eq!(isi_violation_rate(&[], 10, 1000.0), 0.0);
        assert_eq!(isi_violation_rate(&si([5]), 10, 1000.0), 0.0);
        assert_eq!(isi_violation_rate(&si([0, 5]), 10, 0.0), 0.0);
    }

    #[test]
    fn isi_violation_rate_normalises_by_recording_duration() {
        // Spike train spanning 1 second at 1000 Hz, with 1 violation.
        // Times in samples: [0, 50, 1000]. RP = 100 samples.
        // viol = 1 (the 0→50 gap). duration = 1 s. rate = 1.0 /s.
        let times = si([0u64, 50, 1000]);
        assert!((isi_violation_rate(&times, 100, 1000.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn presence_ratio_empty_returns_zero() {
        assert_eq!(presence_ratio(&[], 1000, 10), 0.0);
        assert_eq!(presence_ratio(&si([1, 2, 3]), 0, 10), 0.0);
        assert_eq!(presence_ratio(&si([1, 2, 3]), 100, 0), 0.0);
    }

    #[test]
    fn presence_ratio_full_coverage_is_one() {
        // 10 bins of width 100. Place a spike in each bin.
        let times = siv((0..10).map(|i| i as u64 * 100 + 50));
        assert_eq!(presence_ratio(&times, 1000, 10), 1.0);
    }

    #[test]
    fn presence_ratio_half_coverage_is_half() {
        // 4 bins of width 100. Put spikes only in the first 2.
        let times = si([10u64, 50, 110, 150]);
        assert!((presence_ratio(&times, 400, 4) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn mean_amplitude_handles_empty() {
        assert_eq!(mean_amplitude(&[]), 0.0);
    }

    #[test]
    fn std_amplitude_returns_zero_for_short_inputs() {
        assert_eq!(std_amplitude(&[]), 0.0);
        assert_eq!(std_amplitude(&[5.0]), 0.0);
    }

    #[test]
    fn std_amplitude_matches_known_value() {
        // Variance of [1, 2, 3, 4] (Bessel) = sum((x-2.5)^2)/3 = (2.25+0.25+0.25+2.25)/3 = 5/3
        let s = std_amplitude(&[1.0, 2.0, 3.0, 4.0]);
        assert!((s - (5.0_f32 / 3.0).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn amplitude_snr_zero_when_std_is_zero() {
        assert_eq!(amplitude_snr(&[]), 0.0);
        assert_eq!(amplitude_snr(&[5.0]), 0.0);
        assert_eq!(amplitude_snr(&[5.0, 5.0, 5.0]), 0.0);
    }

    #[test]
    fn fraction_below_counts_strict_less_than() {
        assert_eq!(fraction_below(&[], 1.0), 0.0);
        assert_eq!(fraction_below(&[0.5, 1.0, 2.0, 3.0], 1.0), 0.25);
        assert_eq!(fraction_below(&[0.5, 0.5, 0.5], 1.0), 1.0);
        assert_eq!(fraction_below(&[2.0, 3.0], 1.0), 0.0);
    }

    #[test]
    fn refractory_contamination_zero_for_clean_train() {
        // Spikes spread out beyond the refractory window — contamination = 0.
        let times = si([0u64, 10_000, 20_000, 30_000]);
        let rp = 100;
        let total = 30_000;
        assert_eq!(refractory_contamination(&times, rp, total, 1000.0), 0.0);
    }

    #[test]
    fn refractory_contamination_positive_when_violations_present() {
        // Two close-together spikes inside RP window.
        let times = si([0u64, 5, 1_000, 5_000]);
        let rp = 100;
        let total = 5_000;
        let c = refractory_contamination(&times, rp, total, 1000.0);
        assert!(c > 0.0);
    }

    /// Golden test for the Hill (2011) estimator: build a large train with a
    /// *known* number of refractory violations and confirm the recovered
    /// false-positive fraction matches the analytic value
    /// `f_p = N_viol·T / (2·t_ref·N²)`.
    ///
    /// This is the regression guard for the `N²` denominator. The previous
    /// implementation divided by `N` instead of `N²`, which for `N = 10_000`
    /// inflated the result by ~10_000× (the value saturated far above 1.0,
    /// killing the contamination score). With the fix the recovered fraction
    /// lands near the injected ~4.7% and stays well inside `[0, 1]`.
    #[test]
    fn refractory_contamination_recovers_known_fraction() {
        let n: u64 = 10_000;
        let k: u64 = 19; // injected consecutive violations
        let t_ref: u64 = 60; // 2 ms at 30 kHz
        let normal_gap: u64 = 3_000; // well outside the refractory window

        // First `k` inter-spike gaps are inside the refractory window; the
        // rest are not → exactly `k` ISI violations.
        let mut times = Vec::with_capacity(n as usize);
        let mut cursor = 0u64;
        for j in 0..n {
            times.push(SampleIndex(cursor));
            cursor += if j < k { 10 } else { normal_gap };
        }
        let total = cursor; // recording length = last spike position
        assert_eq!(isi_violations(&times, t_ref) as u64, k);

        let recovered = refractory_contamination(&times, t_ref, total, 30_000.0);
        let expected =
            (k as f64 * total as f64 / (2.0 * t_ref as f64 * n as f64 * n as f64)) as f32;
        assert!(
            (recovered - expected).abs() < 1e-3,
            "recovered {recovered} != analytic {expected}",
        );
        // The old (÷N instead of ÷N²) formula returned ~N× this, far above 1.
        assert!(
            recovered < 0.2,
            "recovered fraction {recovered} implausibly large — N² denominator regressed?",
        );
    }

    #[test]
    fn sliding_refractory_min_is_clean_for_spread_train() {
        // Well-spread train: no violations at any swept refractory period.
        let times = siv((0..200).map(|i| (i as u64 + 1) * 10_000));
        let total = 2_000_000;
        let c = min_contamination_sliding_refractory(&times, 30, 300, 8, total, 30_000.0);
        assert_eq!(c, 0.0);
    }

    #[test]
    fn sliding_refractory_min_never_exceeds_single_window() {
        // Some close pairs → nonzero contamination. The swept minimum must be
        // ≤ the estimate at any single refractory window in the swept range.
        let mut v = vec![0u64];
        for i in 0..500u64 {
            // Mostly spaced, but every 25th spike sits inside the refractory
            // window of its predecessor.
            let gap = if i % 25 == 0 { 20 } else { 5_000 };
            v.push(v.last().unwrap() + gap);
        }
        let times = siv(v);
        let total = *times.last().unwrap();
        let min_c = min_contamination_sliding_refractory(&times, 30, 300, 10, total.0, 30_000.0);
        let at_max = refractory_contamination(&times, 300, total.0, 30_000.0);
        assert!((0.0..=1.0).contains(&min_c));
        assert!(
            min_c <= at_max + 1e-6,
            "swept min {min_c} should not exceed single-window {at_max}",
        );
    }

    #[test]
    fn cross_correlogram_total_count_matches_pair_count_within_window() {
        // Two trains with all pairs within ±max_lag: expect total = |a|*|b|.
        let a = si([0u64, 5, 10]);
        let b = si([1u64, 4, 9]);
        let h = cross_correlogram(&a, &b, 100, 20);
        assert_eq!(h.iter().copied().sum::<u32>(), 9);
    }

    #[test]
    fn cross_correlogram_outside_window_is_dropped() {
        let a = si([0u64]);
        let b = si([50u64, 200]);
        let h = cross_correlogram(&a, &b, 60, 12);
        // Only the first b is within ±60.
        assert_eq!(h.iter().copied().sum::<u32>(), 1);
    }

    #[test]
    fn auto_correlogram_excludes_zero_lag_diagonal() {
        // Single spike — nothing to correlate with itself.
        let times = si([42u64]);
        let h = auto_correlogram(&times, 50, 10);
        assert_eq!(h.iter().copied().sum::<u32>(), 0);
    }

    #[test]
    fn auto_correlogram_counts_each_off_diagonal_pair_twice() {
        // Two spikes 5 samples apart -> contributes a (+5) and a (-5).
        let times = si([10u64, 15]);
        let h = auto_correlogram(&times, 50, 10);
        assert_eq!(h.iter().copied().sum::<u32>(), 2);
    }

    #[test]
    fn empty_inputs_return_zero_histograms() {
        let h = cross_correlogram(&[], &si([1, 2]), 10, 5);
        assert_eq!(h, vec![0; 5]);
        let h = auto_correlogram(&[], 10, 5);
        assert_eq!(h, vec![0; 5]);
    }

    /// CCG(a,b) and CCG(b,a) count exactly the same set of pairs, just
    /// mirrored around the zero-lag bin. The strongest invariant we can
    /// state without arguing about half-open bin boundaries is that the
    /// two have equal totals.
    #[test]
    fn cross_correlogram_total_count_is_symmetric_under_swap() {
        let a = siv([10, 50, 90, 130]);
        let b = siv([20, 60, 100]);
        let bins = 20usize;
        let max_lag = 100u64;

        let ab = cross_correlogram(&a, &b, max_lag, bins);
        let ba = cross_correlogram(&b, &a, max_lag, bins);
        let total_ab: u32 = ab.iter().sum();
        let total_ba: u32 = ba.iter().sum();
        assert_eq!(total_ab, total_ba, "CCG totals must match under swap");
    }

    /// ACG is symmetric around its centre (every off-diagonal pair is
    /// counted twice — once for each direction).
    #[test]
    fn auto_correlogram_is_symmetric_around_centre() {
        let times = siv((0..20).map(|i| (i * 37) as u64 + 5));
        let bins = 30usize;
        let h = auto_correlogram(&times, 1000, bins);
        for i in 0..bins / 2 {
            let mirror = bins - 1 - i;
            // Allow ±1 slop from rounding when 2*max_lag/bins isn't integer.
            let diff = (h[i] as i32 - h[mirror] as i32).abs();
            assert!(
                diff <= 1,
                "ACG asymmetry at bin {i}↔{mirror}: {} vs {} (diff {diff})",
                h[i],
                h[mirror],
            );
        }
    }

    /// ISI violations + presence ratio combined: a perfectly clean,
    /// well-distributed train spanning the full recording has 0 violations
    /// and full presence.
    #[test]
    fn well_distributed_clean_train_has_perfect_metrics() {
        let times = siv((0..100).map(|i| (i as u64 + 1) * 1000));
        // Total = max spike time, so every bin has at least one spike.
        let total = 100_000u64;
        assert_eq!(isi_violations(&times, 100), 0);
        assert!(
            (presence_ratio(&times, total, 10) - 1.0).abs() < 1e-6,
            "presence ratio not 1.0: got {}",
            presence_ratio(&times, total, 10)
        );
    }

    /// SNR scales correctly: doubling all amplitudes leaves SNR unchanged.
    #[test]
    fn amplitude_snr_is_scale_invariant() {
        let amps = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let scaled: Vec<f32> = amps.iter().map(|v| v * 7.0).collect();
        let s1 = amplitude_snr(&amps);
        let s2 = amplitude_snr(&scaled);
        assert!(
            (s1 - s2).abs() < 1e-5,
            "snr changed under uniform scaling: {s1} vs {s2}",
        );
    }
}
