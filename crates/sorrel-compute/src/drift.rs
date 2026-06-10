//! Drift detection on per-cluster spike streams.
//!
//! "Drift" in extracellular recording shows up as a slow change in the
//! per-spike amplitude (electrode walks toward or away from the soma) and as
//! gaps in firing (electrode-contact loss). These metrics quantify both so
//! the curator can flag clusters that are likely *one neuron seen partially*
//! rather than one neuron seen cleanly.
//!
//! These are per-cluster *quality-control indicators* (amplitude-vs-time
//! correlation/slope, presence CV, silent-gap fraction), not a spatial
//! motion estimate. They do not recover a µm-vs-time motion trace and are not
//! the cross-correlation-of-depth-histogram registration that Kilosort2.5 /
//! dredge perform; do not treat their output as a drift-correction signal.

use sorrel_io::SampleIndex;

/// Sums needed to compute Pearson correlation or least-squares slope of a
/// (time, amplitude) cloud. Returns `None` for fewer than 3 paired samples.
/// `time_to_f64` lets the caller pick units (raw samples vs seconds).
fn time_amp_covariance(
    times: &[SampleIndex],
    amps: &[f32],
    time_to_f64: impl Fn(SampleIndex) -> f64,
) -> Option<(f64, f64, f64)> {
    let n = times.len().min(amps.len());
    if n < 3 {
        return None;
    }
    let nf = n as f64;
    let mean_t = times[..n].iter().map(|&t| time_to_f64(t)).sum::<f64>() / nf;
    let mean_a = amps[..n].iter().map(|&a| a as f64).sum::<f64>() / nf;
    let mut num = 0.0_f64;
    let mut denom_t = 0.0_f64;
    let mut denom_a = 0.0_f64;
    for i in 0..n {
        let dt = time_to_f64(times[i]) - mean_t;
        let da = amps[i] as f64 - mean_a;
        num += dt * da;
        denom_t += dt * dt;
        denom_a += da * da;
    }
    Some((num, denom_t, denom_a))
}

/// Pearson correlation between spike time and amplitude — signed, bounded
/// to `[-1, 1]`. Magnitude near 0 means amplitudes are stationary; large
/// magnitude means a monotonic drift.
///
/// Returns 0 for fewer than 3 spikes.
pub fn amplitude_drift_correlation(times: &[SampleIndex], amps: &[f32]) -> f32 {
    let Some((num, denom_t, denom_a)) = time_amp_covariance(times, amps, |t| t.as_f64()) else {
        return 0.0;
    };
    if denom_t <= 0.0 || denom_a <= 0.0 {
        return 0.0;
    }
    (num / (denom_t * denom_a).sqrt()) as f32
}

/// Slope (Δamp / Δt, in amplitude-units per second) of a least-squares line
/// fit through the (time, amplitude) cloud. Useful as an absolute drift
/// magnitude rather than the unitless correlation. Sample rate is needed to
/// convert from sample-indexed times to seconds.
pub fn amplitude_drift_slope(times: &[SampleIndex], amps: &[f32], sample_rate: f32) -> f32 {
    if sample_rate <= 0.0 {
        return 0.0;
    }
    let inv_sr = 1.0_f64 / sample_rate as f64;
    let Some((num, denom_t, _)) = time_amp_covariance(times, amps, |t| t.as_f64() * inv_sr) else {
        return 0.0;
    };
    if denom_t <= 0.0 {
        return 0.0;
    }
    (num / denom_t) as f32
}

/// Coefficient of variation (std / |mean|) of bin counts in a presence-bin
/// histogram. Higher = more bursty / patchy presence. Drops to 0 for fewer
/// than 2 non-empty bins.
pub fn presence_cv(times: &[SampleIndex], total_duration_samples: u64, n_bins: usize) -> f32 {
    if times.is_empty() || n_bins == 0 || total_duration_samples == 0 {
        return 0.0;
    }
    let bin_width = (total_duration_samples as f64 / n_bins as f64).max(1.0);
    let mut counts = vec![0u32; n_bins];
    for &t in times {
        let idx = ((t.as_f64()) / bin_width) as usize;
        if idx < n_bins {
            counts[idx] += 1;
        }
    }
    let counts_f32: Vec<f32> = counts.iter().map(|&c| c as f32).collect();
    let m = crate::distribution::Moments::from_f32(&counts_f32);
    if m.mean <= 0.0 {
        return 0.0;
    }
    (m.m2.sqrt() / m.mean) as f32
}

/// Walk the spike train in `n_windows` evenly-sized time slices and compute
/// the refractory-contamination ratio (Hill 2011) inside each window.
/// Returns `(window_centre_seconds, contamination)` pairs — useful for
/// the sliding-contamination plot in the UI.
///
/// Windows shorter than `min_spikes_per_window` spikes return contamination
/// `NaN` so the renderer can interrupt the line at that bin.
pub fn sliding_refractory_contamination(
    times: &[SampleIndex],
    refractory_samples: u64,
    total_duration_samples: u64,
    sample_rate: f32,
    n_windows: usize,
    min_spikes_per_window: usize,
) -> Vec<(f32, f32)> {
    use crate::metrics::refractory_contamination;
    if times.is_empty() || n_windows == 0 || total_duration_samples == 0 || sample_rate <= 0.0 {
        return Vec::new();
    }
    let win_samples = (total_duration_samples / n_windows as u64).max(1);
    let inv_sr = 1.0_f32 / sample_rate;
    let mut out = Vec::with_capacity(n_windows);
    let mut idx = 0usize;
    for w in 0..n_windows {
        let lo = w as u64 * win_samples;
        let hi = lo + win_samples;
        // Advance to the first spike inside this window.
        while idx < times.len() && times[idx].0 < lo {
            idx += 1;
        }
        let start = idx;
        let mut end = idx;
        while end < times.len() && times[end].0 < hi {
            end += 1;
        }
        let centre_s = (lo as f32 + win_samples as f32 * 0.5) * inv_sr;
        let contam = if end - start >= min_spikes_per_window {
            refractory_contamination(
                &times[start..end],
                refractory_samples,
                win_samples,
                sample_rate,
            )
        } else {
            f32::NAN
        };
        out.push((centre_s, contam));
    }
    out
}

/// Largest "gap" in the recording where this cluster is silent, expressed as
/// a fraction of the total recording length. Useful to flag electrode drop:
/// 0.05 = a 5%-of-recording silent stretch.
pub fn longest_silent_gap_frac(times: &[SampleIndex], total_duration_samples: u64) -> f32 {
    if times.is_empty() || total_duration_samples == 0 {
        return 1.0;
    }
    let total = total_duration_samples;
    let first = times[0].0;
    let last = times.last().unwrap().0;
    let mut max_gap = first.max(total.saturating_sub(last));
    for w in times.windows(2) {
        let g = w[1].0.saturating_sub(w[0].0);
        if g > max_gap {
            max_gap = g;
        }
    }
    (max_gap as f64 / total as f64).clamp(0.0, 1.0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn si(xs: impl IntoIterator<Item = u64>) -> Vec<SampleIndex> {
        xs.into_iter().map(SampleIndex).collect()
    }

    #[test]
    fn drift_correlation_zero_for_constant_amplitude() {
        let times = si((0..100).map(|i| i as u64 * 10));
        let amps = vec![5.0_f32; 100];
        assert_eq!(amplitude_drift_correlation(&times, &amps), 0.0);
    }

    #[test]
    fn drift_correlation_positive_for_rising_amplitude() {
        let times = si((0..100).map(|i| i as u64 * 10));
        let amps: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        assert!(amplitude_drift_correlation(&times, &amps) > 0.99);
    }

    #[test]
    fn drift_correlation_handles_short_inputs() {
        assert_eq!(amplitude_drift_correlation(&[], &[]), 0.0);
        assert_eq!(amplitude_drift_correlation(&si([1, 2]), &[3.0, 4.0]), 0.0);
    }

    #[test]
    fn drift_slope_zero_for_flat_amplitude() {
        let times = si((0..50).map(|i| i as u64 * 1000));
        let amps = vec![2.0_f32; 50];
        assert_eq!(amplitude_drift_slope(&times, &amps, 1000.0), 0.0);
    }

    #[test]
    fn drift_slope_positive_for_rising_amplitude() {
        // 1 spike per second, amplitude = 0.5 * t.
        let times = si((0..50).map(|i| i as u64 * 1000));
        let amps: Vec<f32> = (0..50).map(|i| i as f32 * 0.5).collect();
        let slope = amplitude_drift_slope(&times, &amps, 1000.0);
        assert!((slope - 0.5).abs() < 1e-3);
    }

    #[test]
    fn presence_cv_zero_for_uniform_spike_train() {
        // 10 bins of width 100; place exactly 5 spikes inside each bin so all
        // counts are identical regardless of binning rounding.
        let mut times: Vec<SampleIndex> = Vec::new();
        for bin in 0..10 {
            for k in 0..5 {
                times.push(SampleIndex((bin * 100 + 10 + k * 15) as u64));
            }
        }
        assert!(presence_cv(&times, 1000, 10) < 1e-6);
    }

    #[test]
    fn presence_cv_high_for_bursty_train() {
        // All spikes packed into 1 bin out of 10.
        let times = si(0..100);
        assert!(presence_cv(&times, 1000, 10) > 1.0);
    }

    #[test]
    fn longest_silent_gap_full_for_empty_train() {
        assert_eq!(longest_silent_gap_frac(&[], 1000), 1.0);
    }

    #[test]
    fn longest_silent_gap_picks_largest_inter_spike_gap() {
        let times = si([0u64, 100, 900]);
        // Gaps: 0 (lead), 100, 800, 100 (trail). max=800/1000=0.8
        let g = longest_silent_gap_frac(&times, 1000);
        assert!((g - 0.8).abs() < 1e-6);
    }

    #[test]
    fn sliding_contamination_returns_one_row_per_window() {
        // 100 spikes evenly spaced, no contamination — every window should
        // be finite and zero.
        let times = si((0..100).map(|i| i as u64 * 100));
        let s = sliding_refractory_contamination(&times, 10, 10_000, 1000.0, 5, 5);
        assert_eq!(s.len(), 5);
        for (_, c) in &s {
            assert!(c.is_finite());
            assert!(*c < 1e-3);
        }
    }

    #[test]
    fn sliding_contamination_marks_low_count_windows_nan() {
        // Single spike. Most windows have 0 spikes — they should report NaN.
        let times = si([500_u64]);
        let s = sliding_refractory_contamination(&times, 10, 10_000, 1000.0, 5, 2);
        assert_eq!(s.len(), 5);
        let nan_count = s.iter().filter(|(_, c)| c.is_nan()).count();
        assert!(nan_count >= 4);
    }

    #[test]
    fn sliding_contamination_returns_empty_for_empty_input() {
        let s = sliding_refractory_contamination(&[], 10, 1000, 1000.0, 5, 1);
        assert!(s.is_empty());
    }

    #[test]
    fn longest_silent_gap_includes_pre_first_and_post_last() {
        let times = si([400u64, 500]);
        // Pre-first = 400, post-last = 500, internal = 100. max = 500.
        let g = longest_silent_gap_frac(&times, 1000);
        assert!((g - 0.5).abs() < 1e-6);
    }

    #[test]
    fn drift_correlation_handles_constant_times() {
        let times = vec![SampleIndex(100); 10];
        let amps = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0];
        let r = amplitude_drift_correlation(&times, &amps);
        assert!(r.is_finite() && r.abs() < 1e-3);
    }

    #[test]
    fn drift_correlation_negative_for_falling_amplitude() {
        let times = si((0..50).map(|i| i as u64 * 1000));
        let amps: Vec<f32> = (0..50).map(|i| 100.0 - i as f32 * 0.5).collect();
        let r = amplitude_drift_correlation(&times, &amps);
        assert!(r < -0.9, "expected strong negative correlation, got {r}");
    }

    #[test]
    fn drift_correlation_in_unit_range() {
        // Random-looking input — r must remain in [-1, 1].
        let times = si((0..100).map(|i| i as u64 * 1000));
        let amps: Vec<f32> = (0..100).map(|i| ((i * 17) % 50) as f32).collect();
        let r = amplitude_drift_correlation(&times, &amps);
        assert!((-1.0..=1.0).contains(&r), "r={r} out of [-1, 1]");
    }

    #[test]
    fn drift_slope_handles_short_inputs() {
        assert_eq!(amplitude_drift_slope(&[], &[], 1000.0), 0.0);
        assert_eq!(amplitude_drift_slope(&si([100]), &[1.0], 1000.0), 0.0);
    }

    #[test]
    fn drift_slope_handles_zero_or_negative_sample_rate() {
        let times = si(0..10);
        let amps: Vec<f32> = (0..10).map(|i| i as f32).collect();
        assert_eq!(amplitude_drift_slope(&times, &amps, 0.0), 0.0);
        assert_eq!(amplitude_drift_slope(&times, &amps, -1.0), 0.0);
    }

    #[test]
    fn drift_slope_negative_for_falling_amplitude() {
        let times = si((0..50).map(|i| i as u64 * 1000));
        let amps: Vec<f32> = (0..50).map(|i| -0.5 * i as f32).collect();
        let slope = amplitude_drift_slope(&times, &amps, 1000.0);
        assert!((slope + 0.5).abs() < 1e-3);
    }

    #[test]
    fn presence_cv_zero_inputs_return_finite() {
        assert!(presence_cv(&[], 1000, 10).is_finite());
        assert!(presence_cv(&si([1, 2, 3]), 0, 10).is_finite());
        assert!(presence_cv(&si([1, 2, 3]), 1000, 0).is_finite());
    }

    #[test]
    fn longest_silent_gap_zero_total_returns_finite() {
        let g = longest_silent_gap_frac(&si([100]), 0);
        assert!(g.is_finite());
    }

    /// Property: longest gap is in [0, 1] for any valid input.
    #[test]
    fn longest_silent_gap_in_unit_range() {
        for (times, total) in [
            (si([0u64, 100, 200]), 1000u64),
            (si([500u64]), 1000u64),
            (si([0u64]), 1000u64),
            (si([999u64]), 1000u64),
        ] {
            let g = longest_silent_gap_frac(&times, total);
            assert!(
                (0.0..=1.0).contains(&g),
                "g={g} out of [0, 1] for total={total}",
            );
        }
    }

    /// Property: presence_cv is non-negative for any input.
    #[test]
    fn presence_cv_is_non_negative() {
        for n_bins in [1usize, 5, 10, 50] {
            let times = si((0..50).map(|i| i as u64 * 20));
            let cv = presence_cv(&times, 1000, n_bins);
            assert!(cv >= 0.0, "cv={cv} is negative");
        }
    }
}
