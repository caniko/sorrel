//! Stable firing-rate and amplitude epochs.

use sorrel_io::SampleIndex;

/// One half-open interval in recording time, with levels reported from the raw spikes.
#[derive(Clone, Copy, Debug)]
pub struct Epoch {
    pub start_s: f64,
    pub end_s: f64,
    pub n: u32,
    pub rate_hz: f32,
    /// Spike-weighted mean of the stored amplitudes, or `NaN` if unavailable.
    pub amp_mean: f32,
    /// Whether amplitude contributed to the fit of this epoch, not just its description.
    pub amp_fitted: bool,
}

const BINS: usize = 100;
const MIN_BINS: usize = 5;
const MIN_SPIKES: usize = 2 * MIN_BINS;
// Twenty residual degrees of freedom give a noise estimate with ~32% relative
// standard error under Gaussian noise, rather than relying on a few pairs.
const MIN_AMP_DF: u32 = 20;

#[derive(Default, Clone, Copy)]
struct Stats {
    n: u32,
    mean: f64,
    ss: f64,
}

impl Stats {
    // Weighted Welford update, used for individual amplitudes (n=1) and for
    // bin means (weighted by spike count). Avoids subtracting large squares.
    fn add(&mut self, mean: f64, n: u32) {
        if n > 0 {
            let delta = mean - self.mean;
            let weight = f64::from(n) / f64::from(self.n + n);
            self.ss += delta * delta * weight * f64::from(self.n);
            self.mean += delta * weight;
            self.n += n;
        }
    }
}

fn poisson_cap(mean: u32) -> u32 {
    // Sum mass relative to the mode so a large mean cannot underflow the seed.
    let mode = mean;
    let mean = f64::from(mean);
    let mut cdf = 1.0;
    let mut mass = 1.0;
    for k in (1..=mode).rev() {
        mass *= f64::from(k) / mean;
        cdf += mass;
        if mass <= f64::EPSILON * cdf {
            break;
        }
    }
    let mut total = cdf;
    mass = 1.0;
    for k in mode + 1.. {
        mass *= mean / f64::from(k);
        total += mass;
        if mass <= f64::EPSILON * total {
            break;
        }
    }
    let mut k = mode;
    mass = 1.0;
    while cdf < 0.999 * total {
        k += 1;
        mass *= mean / f64::from(k);
        cdf += mass;
    }
    k.max(3) // Allow a few spikes even if the local median is zero.
}

/// Estimate each bin's baseline from the median (rounded down) of six nearby
/// bins, three on each side outside the bin and its immediate neighbours.
/// Take more from the other side near recording ends. A run of at least three
/// bins above their individual Poisson caps is left alone. Otherwise apply the
/// individual cap and, for adjacent bins both above baseline, cap their sum
/// using the sum of their baselines as the Poisson mean. Share that budget in
/// proportion to each bin's excess, rounding down and taking the tighter cap
/// if pairs overlap. All caps use the 99.9th percentile with a floor of three.
fn capped_counts(bins: &[Stats; BINS]) -> [u32; BINS] {
    const FLANKS: usize = 6;
    let raw = bins.map(|bin| bin.n);
    let baseline: [_; BINS] = std::array::from_fn(|b| {
        let gap_start = b.saturating_sub(1);
        let gap_len = (b + 2).min(BINS) - gap_start;
        // Center and clamp the window in the index space with the gap removed.
        let start = gap_start
            .saturating_sub(FLANKS / 2)
            .min(BINS - gap_len - FLANKS);
        let mut window: [_; FLANKS] = std::array::from_fn(|i| {
            let i = start + i;
            raw[if i < gap_start { i } else { i + gap_len }]
        });
        window.sort_unstable();
        window[FLANKS / 2 - 1].midpoint(window[FLANKS / 2])
    });
    let caps = baseline.map(poisson_cap);
    let outliers: [_; BINS] = std::array::from_fn(|b| raw[b] > caps[b]);
    let mut sustained = [false; BINS];
    for (b, triple) in outliers.windows(3).enumerate() {
        if triple.iter().all(|&outlier| outlier) {
            sustained[b..b + 3].fill(true);
        }
    }

    let mut counts = std::array::from_fn(|b| {
        if sustained[b] {
            raw[b]
        } else {
            raw[b].min(caps[b])
        }
    });
    for a in 0..BINS - 1 {
        let b = a + 1;
        if sustained[a] || sustained[b] || raw[a] <= baseline[a] || raw[b] <= baseline[b] {
            continue;
        }
        let mean = baseline[a] + baseline[b];
        let cap = poisson_cap(mean);
        let sum = raw[a] + raw[b];
        if sum > cap {
            for i in [a, b] {
                let share =
                    u64::from(raw[i] - baseline[i]) * u64::from(cap - mean) / u64::from(sum - mean);
                counts[i] = counts[i].min(baseline[i] + share as u32);
            }
        }
    }
    counts
}

fn dispersion(counts: &[u32; BINS]) -> f64 {
    // Pool squared differences at lags 1..=5 bins to capture short runs of
    // elevated firing, not just isolated bursts. Winsorize at the 90th
    // percentile: a single abrupt step crosses at most 15 pairs, less than
    // the top decile. For stationary Poisson counts, E[Δ²] = 2 × mean count;
    // winsorizing chi-square(1) at its 90th percentile retains ~0.831 of that.
    let mut differences = Vec::new();
    for lag in 1..=MIN_BINS {
        differences.extend(
            counts
                .iter()
                .zip(&counts[lag..])
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)),
        );
    }
    differences.sort_unstable_by(f64::total_cmp);
    let cap = differences[9 * (differences.len() - 1) / 10];
    let mean_diff = differences.iter().map(|&d| d.min(cap)).sum::<f64>() / differences.len() as f64;
    let mean_count = counts.iter().map(|&n| f64::from(n)).sum::<f64>() / BINS as f64;
    (mean_diff / (0.831 * 2.0 * mean_count)).max(1.0)
}

fn amp_variance(bins: &[Stats]) -> Option<f64> {
    // Pool spike-level within-bin sample variances by degrees of freedom,
    // winsorizing at their 90th percentile so isolated noisy bins don't set
    // the scale. Singleton and empty bins cannot estimate within-bin noise.
    let mut variances: Vec<_> = bins
        .iter()
        .filter(|b| b.n > 1)
        .map(|b| (b.ss / f64::from(b.n - 1), f64::from(b.n - 1)))
        .collect();
    if variances.iter().map(|&(_, df)| df).sum::<f64>() < f64::from(MIN_AMP_DF) {
        return None;
    }
    variances.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let cap = variances[(9 * (variances.len() - 1)).div_ceil(10)].0;
    let (ss, df) = variances.iter().fold((0.0, 0.0), |(ss, df), &(v, w)| {
        (ss + v.min(cap) * w, df + w)
    });
    // A digitized signal can have zero within-bin noise. Use one f32 epsilon
    // at the mean-amplitude scale as a floor rather than dividing by zero.
    let scale = bins
        .iter()
        .map(|b| b.mean.abs())
        .fold(f64::from(f32::MIN_POSITIVE), f64::max);
    Some((ss / df).max((scale * f64::from(f32::EPSILON)).powi(2)))
}

/// Fit constant rate and (if available) amplitude over 100 equal recording
/// bins, with at least five bins per segment. Each fitted level costs
/// ln(spikes) (one rate per segment, one amplitude per nonempty segment if
/// enabled). Searching over boundary locations also incurs a penalty: charge
/// 2 ln(91) per boundary, the cost of coding a location uniformly among the
/// 91 possible interior edges on the deviance scale. This avoids treating the
/// best of many candidate edges as a prespecified edge and is additive for DP.
///
/// For rate fitting only, cap isolated high counts and jointly cap adjacent
/// elevated bins using Poisson limits from the surrounding counts (see
/// `capped_counts`). This limits brief bursts without flattening sustained
/// rate changes. Scale the rate deviance by the winsorized short-lag
/// dispersion of the capped counts (at least one). Report raw counts and rates;
/// neither the amplitude fit nor its reported means use capped counts.
///
/// Returns empty with fewer than ten spikes (twice the minimum bin count),
/// invalid duration/sample rate, or times outside `[0, total_samples)`.
/// Missing/invalid amplitudes or fewer than 20 pooled within-bin degrees of
/// freedom reduce the fit to rates only. Observed amplitude means are still
/// reported if the amplitude data are valid.
pub fn stable_epochs(
    times: &[SampleIndex],
    amps: &[f32],
    total_samples: u64,
    sr: f32,
) -> Vec<Epoch> {
    let duration = total_samples as f64 / f64::from(sr);
    if times.len() < MIN_SPIKES
        || u32::try_from(times.len()).is_err()
        || !sr.is_finite()
        || sr <= 0.0
        || total_samples == 0
        || !(duration as f32).is_finite()
    {
        return Vec::new();
    }
    let has_amp = amps.len() == times.len() && amps.iter().all(|a| a.is_finite());
    let mut bins = [Stats::default(); BINS];
    for (i, &t) in times.iter().enumerate() {
        if t.0 >= total_samples {
            return Vec::new();
        }
        // Exact edges, including for recordings not divisible by 100 samples.
        let b = &mut bins[(u128::from(t.0) * BINS as u128 / u128::from(total_samples)) as usize];
        if has_amp {
            b.add(f64::from(amps[i]), 1);
        } else {
            b.n += 1;
        }
    }

    let rate_counts = capped_counts(&bins);
    let phi = dispersion(&rate_counts);
    let sigma2 = if has_amp { amp_variance(&bins) } else { None };
    let level_penalty = (times.len() as f64).ln();
    let boundary_penalty = 2.0 * ((BINS - 2 * MIN_BINS + 1) as f64).ln();
    let mut best = [f64::INFINITY; BINS + 1];
    let mut previous = [0; BINS + 1];
    let mut fits = [Stats::default(); BINS + 1];
    best[0] = 0.0;
    // Exact optimal partitioning. ponytail: use PELT pruning if B grows;
    // a quadratic search over 100 bins is small and has no pruning assumptions.
    for start in 0..=BINS - MIN_BINS {
        if !best[start].is_finite() {
            continue;
        }
        let base = best[start] + level_penalty + if start == 0 { 0.0 } else { boundary_penalty };
        let mut segment = Stats::default();
        let mut rate_n = 0;
        for (i, (bin, &count)) in bins[start..].iter().zip(&rate_counts[start..]).enumerate() {
            segment.add(bin.mean, bin.n);
            rate_n += count;
            let width = i + 1;
            if width < MIN_BINS {
                continue;
            }
            // Profiled Poisson deviance / phi, dropping terms that sum to a
            // constant across partitions; the n=0 limit is zero.
            let n = f64::from(rate_n);
            let rate = if n > 0.0 {
                -2.0 * n * (n / width as f64).ln() / phi
            } else {
                0.0
            };
            let amp = sigma2
                .filter(|_| segment.n > 0)
                .map_or(0.0, |v| segment.ss / v + level_penalty);
            let cost = base + rate + amp;
            let end = start + width;
            if cost < best[end] {
                best[end] = cost;
                previous[end] = start;
                fits[end] = segment;
            }
        }
    }

    let mut epochs = Vec::new();
    let mut end = BINS;
    while end > 0 {
        let start = previous[end];
        let fit = fits[end];
        // Report the first sample on the right, so a time split at this
        // boundary can reproduce the exact half-open integer partition.
        let edge_s = |b| {
            (u128::from(total_samples) * b as u128).div_ceil(BINS as u128) as f64 / f64::from(sr)
        };
        let rate_hz = (f64::from(fit.n) / (duration * (end - start) as f64 / BINS as f64)) as f32;
        if !rate_hz.is_finite() {
            return Vec::new();
        }
        epochs.push(Epoch {
            start_s: edge_s(start),
            end_s: edge_s(end),
            n: fit.n,
            rate_hz,
            amp_mean: if has_amp && fit.n > 0 {
                fit.mean as f32
            } else {
                f32::NAN
            },
            amp_fitted: sigma2.is_some() && fit.n > 0,
        });
        end = start;
    }
    epochs.reverse();
    epochs
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 1_000.0;
    const TOTAL: u64 = BINS as u64 * 1_000;

    fn train(
        seed: u64,
        rate: impl Fn(usize) -> f64,
        amp: impl Fn(usize) -> f32,
    ) -> (Vec<SampleIndex>, Vec<f32>) {
        let mut state = seed;
        let mut uniform = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (1 + (state >> 33)) as f64 / ((1_u64 << 31) + 1) as f64
        };
        let mut times = Vec::new();
        let mut amps = Vec::new();
        for bin in 0..BINS {
            // Exponential inter-arrival times, with independent Gaussian amplitudes.
            let mut t = 0.0;
            loop {
                t -= uniform().ln() / rate(bin);
                if t >= 1.0 {
                    break;
                }
                times.push(SampleIndex(bin as u64 * 1000 + (1000.0 * t) as u64));
                let z = (-2.0 * uniform().ln()).sqrt() * (std::f64::consts::TAU * uniform()).cos();
                amps.push(amp(bin) + 5.0 * z as f32);
            }
        }
        (times, amps)
    }

    #[test]
    fn amplitude_step() {
        let (times, amps) = train(0, |_| 30.0, |b| if b < 60 { 100.0 } else { 50.0 });
        let e = stable_epochs(&times, &amps, TOTAL, SR);
        assert_eq!(e.len(), 2, "{e:?}");
        assert!((e[1].start_s - 60.0).abs() <= 1.0, "{e:?}");
        assert!((e[1].amp_mean / e[0].amp_mean - 0.5).abs() < 0.01);
        assert_eq!(
            (e[0].start_s, e[0].end_s, e[1].end_s),
            (0.0, e[1].start_s, 100.0)
        );
        assert_eq!((e[0].n + e[1].n) as usize, times.len());
    }

    #[test]
    fn stationary_and_bursty_trains_rarely_split() {
        for seed in 0..3 {
            let (times, _) = train(seed, |_| 1.0, |_| 100.0);
            assert_eq!(stable_epochs(&times, &[], TOTAL, SR).len(), 1);
            // Three isolated one-bin bursts, none sustained for five bins.
            let (times, _) = train(
                seed,
                |b| if [20, 36, 70].contains(&b) { 75.0 } else { 5.0 },
                |_| 100.0,
            );
            assert_eq!(stable_epochs(&times, &[], TOTAL, SR).len(), 1);
        }
    }

    #[test]
    fn burst_straddling_a_bin_edge_does_not_split() {
        let mut times: Vec<_> = (0..100)
            .flat_map(|b| {
                let n = [5, 5, 8, 7, 6].get(b as usize).copied().unwrap_or(5);
                (1..=n).map(move |i| SampleIndex(b * 1000 + i * 1000 / (n + 1)))
            })
            .collect();
        // A 0.8 s burst centered on the edge between the first two bins.
        times.extend((0..75).map(|i| SampleIndex(600 + i * 800 / 75)));
        times.sort_unstable_by_key(|t| t.0);
        let e = stable_epochs(&times, &[], TOTAL, SR);
        assert_eq!(e.len(), 1, "{e:?}");
        assert_eq!(e[0].n as usize, times.len());
    }

    #[test]
    fn isolated_burst_does_not_hide_silence() {
        let times: Vec<_> = (0..100)
            .flat_map(|b| {
                let n = if (30..70).contains(&b) {
                    0
                } else if b == 80 {
                    75
                } else {
                    5
                };
                (0..n).map(move |i| SampleIndex(b * 1000 + i * 1000 / n))
            })
            .collect();
        let e = stable_epochs(&times, &[], TOTAL, SR);
        assert_eq!(e.len(), 3, "{e:?}");
        assert_eq!((e[1].start_s, e[1].end_s, e[1].n), (30.0, 70.0, 0));
        assert_eq!(e[2].n, 220);
        assert_eq!(e[2].rate_hz, 220.0 / 30.0);
    }

    #[test]
    fn rate_cap_preserves_three_bin_changes() {
        let mut bins = [Stats {
            n: 5,
            ..Stats::default()
        }; BINS];
        for b in [0, 1, 30, 31, 32, 98, 99] {
            bins[b].n = 75;
        }
        // Include all six flanking bins in the silent interval.
        bins[58..67].fill(Stats::default());
        bins[62].n = 75;
        let counts = capped_counts(&bins);
        let share = poisson_cap(10) / 2;
        assert_eq!(counts[..2], [share; 2]);
        assert_eq!(counts[98..], [share; 2]);
        assert_eq!(counts[30..33], [75; 3]);
        assert_eq!(counts[62], 3);
    }

    #[test]
    fn one_matching_pair_cannot_set_the_amplitude_noise() {
        let mut times: Vec<_> = (0..100).map(|b| SampleIndex(b * 1000)).collect();
        let mut amps: Vec<_> = (0..100).map(|b| 95.0 + (b % 11) as f32).collect();
        times.insert(1, SampleIndex(1));
        amps.insert(1, amps[0]);
        let e = stable_epochs(&times, &amps, TOTAL, SR);
        assert_eq!(e.len(), 1, "{e:?}");
        assert!(e[0].amp_mean.is_finite());
    }

    #[test]
    fn silence_has_no_amplitude_parameter() {
        // The rate gain (~17.5) pays for a boundary and rate (~14.2), but
        // not a spurious amplitude parameter for the five silent bins (~19.3).
        let times: Vec<_> = (5..100)
            .flat_map(|b| {
                (0..if b % 5 == 0 { 1 } else { 2 }).map(move |i| SampleIndex(b * 1000 + i * 500))
            })
            .collect();
        for amps in [&[][..], &vec![100.0; times.len()]] {
            let e = stable_epochs(&times, amps, TOTAL, SR);
            assert_eq!(e.len(), 2, "{e:?}");
            assert_eq!((e[0].end_s, e[0].n), (5.0, 0));
        }
    }

    #[test]
    fn rate_step_with_and_without_amplitudes() {
        let (times, _) = train(42, |b| if b < 60 { 60.0 } else { 30.0 }, |_| 100.0);
        let amps = vec![100.0; times.len()];
        let e = stable_epochs(&times, &amps, TOTAL, SR);
        assert_eq!(e.len(), 2, "{e:?}");
        assert!((e[1].start_s - 60.0).abs() <= 1.0, "{e:?}");
        assert_eq!((e[0].amp_mean, e[1].amp_mean), (100.0, 100.0));
        for amps in [&[][..], &amps[..1], &vec![f32::NAN; times.len()]] {
            let e = stable_epochs(&times, amps, TOTAL, SR);
            assert_eq!(e.len(), 2, "{e:?}");
            assert!((e[1].rate_hz / e[0].rate_hz - 0.5).abs() < 0.05);
            assert!(e.iter().all(|epoch| epoch.amp_mean.is_nan()));
        }
    }

    #[test]
    fn insufficient_or_invalid_input() {
        let times = vec![SampleIndex(500); MIN_SPIKES];
        assert!(stable_epochs(&times[..MIN_SPIKES - 1], &[], TOTAL, SR).is_empty());
        for (total, sr) in [
            (0, SR),
            (500, SR),
            (TOTAL, 0.0),
            (TOTAL, -SR),
            (TOTAL, f32::NAN),
            (TOTAL, f32::INFINITY),
        ] {
            assert!(stable_epochs(&times, &[], total, sr).is_empty());
        }
    }
}
