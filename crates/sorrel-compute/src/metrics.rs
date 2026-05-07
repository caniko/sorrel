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
    let duration_s = (spike_times[spike_times.len() - 1].0.saturating_sub(spike_times[0].0))
        as f32
        / sample_rate;
    if duration_s <= 0.0 {
        return 0.0;
    }
    n as f32 / duration_s
}

/// Hill et al. (2011) refractory-period contamination ratio: the rate of
/// double-counted spikes inside the refractory window relative to the total
/// firing rate. Returns 0 when too few spikes to estimate.
///
/// `refractory_samples` is the length of the refractory window in samples;
/// `total_duration_samples` is the recording's total length used to compute
/// the firing rate denominator.
pub fn refractory_contamination(
    spike_times: &[SampleIndex],
    refractory_samples: u64,
    total_duration_samples: u64,
    sample_rate: f32,
) -> f32 {
    if spike_times.len() < 2 || refractory_samples == 0 || total_duration_samples == 0 {
        return 0.0;
    }
    let n = spike_times.len() as f32;
    let viol = isi_violations(spike_times, refractory_samples) as f32;
    let total_s = total_duration_samples as f32 / sample_rate;
    let rp_s = refractory_samples as f32 / sample_rate;
    if total_s <= 0.0 || rp_s <= 0.0 {
        return 0.0;
    }
    // Hill formula: viol_rate / (n * 2 * rp / total_duration)
    let denom = n * 2.0 * rp_s / total_s;
    if denom <= 0.0 {
        return 0.0;
    }
    viol / denom
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
        let idx = ((t.as_f64()) / bin_width) as usize;
        if idx < n_bins {
            filled[idx] = true;
        }
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
                h[i], h[mirror],
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
