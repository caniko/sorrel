//! Stable firing-rate and amplitude epochs.
//!
//! Exact optimal partitioning of 100 recording bins with bounded per-bin
//! losses, after the biweight loss of Fearnhead & Rigaill (2019),
//! "Changepoint detection in the presence of outliers", JASA
//! 114(525):169–183, doi:10.1080/01621459.2017.1385466. Each series is scaled
//! by a noise level measured from robust differences between bins, so
//! overdispersion from bursts, quantized amplitudes and correlated spikes is
//! taken from the data rather than assumed away.

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
/// Consecutive nonempty bin pairs needed to estimate the amplitude noise.
const MIN_AMP_PAIRS: usize = 20;
/// Squared biweight threshold, K = 3 noise standard deviations as in
/// Fearnhead & Rigaill's simulations.
const K2: f64 = 9.0;
/// Standard deviation per median absolute value of a centred Gaussian.
const MAD_TO_SD: f64 = 1.482_602_218_505_602;

/// One standardized bin: its loss is `min(w (y - θ)², K²)`.
#[derive(Clone, Copy)]
struct Point {
    y: f64,
    w: f64,
}

/// Where a point enters (`w > 0`) or leaves (`w < 0`) the inlier set as θ increases.
#[derive(Clone, Copy)]
struct Event {
    at: f64,
    y: f64,
    w: f64,
}

/// Add a point's two events to a list kept sorted by position.
fn insert(events: &mut Vec<Event>, p: Point) {
    let h = (K2 / p.w).sqrt();
    for e in [
        Event {
            at: p.y - h,
            y: p.y,
            w: p.w,
        },
        Event {
            at: p.y + h,
            y: p.y,
            w: -p.w,
        },
    ] {
        let i = events.partition_point(|x| x.at <= e.at);
        events.insert(i, e);
    }
}

/// Exact `min over θ of Σ min(w (y - θ)², K²)` for the points behind `events`.
/// Between consecutive events the inlier set is fixed, so the loss is a
/// quadratic minimized at the inlier weighted mean clamped to the interval;
/// every other point costs K². The loss is continuous in θ, so ties between
/// events need no particular order. Inlier sums are kept relative to the
/// current event, where every inlier lies within 2K/√w, to avoid cancellation.
fn biweight_min(events: &[Event], points: usize) -> f64 {
    let mut best = K2 * points as f64;
    let (mut a, mut b, mut c, mut inliers) = (0.0, 0.0, 0.0, 0_usize);
    let mut at = 0.0;
    for (k, e) in events.iter().enumerate() {
        if inliers > 0 {
            let d = e.at - at;
            c += d * (a * d - 2.0 * b);
            b -= a * d;
        }
        at = e.at;
        let r = e.y - at;
        a += e.w;
        b += e.w * r;
        c += e.w * r * r;
        if e.w > 0.0 {
            inliers += 1;
        } else {
            inliers -= 1;
            if inliers == 0 {
                (a, b, c) = (0.0, 0.0, 0.0);
            }
        }
        if inliers > 0 {
            let len = events.get(k + 1).map_or(0.0, |next| next.at - at);
            let t = (b / a).clamp(0.0, len);
            let q = (c - t * (2.0 * b - t * a)).max(0.0);
            best = best.min(q + K2 * (points - inliers) as f64);
        }
    }
    best
}

/// Where a bin's count enters (`enter`) or leaves the inlier set of the
/// bounded Poisson deviance as the segment mean count μ increases, with
/// `ln μ` and `c ln c` precomputed because the sweep runs per segment.
#[derive(Clone, Copy)]
struct CountEvent {
    at: f64,
    ln_at: f64,
    c: f64,
    clc: f64,
    enter: bool,
}

/// A bin's count and the `(μ, ln μ)` ends of the interval on which its
/// Poisson deviance `2 [c ln(c/μ) - c + μ]` is below `cap`. The deviance is
/// convex in ln μ with its minimum 0 at μ = c, so each side has one root,
/// found by bisection in ln μ.
#[derive(Clone, Copy)]
struct RateBin {
    c: f64,
    lo: (f64, f64),
    hi: (f64, f64),
}

fn rate_bin(c: f64, cap: f64) -> RateBin {
    let t = cap / 2.0;
    if c == 0.0 {
        return RateBin {
            c,
            lo: (0.0, f64::NEG_INFINITY),
            hi: (t, t.ln()),
        };
    }
    let f = |x: f64| c * c.ln() - c * x - c + x.exp() - t;
    let root = |mut a: f64, mut b: f64| {
        let positive_at_a = f(a) > 0.0;
        loop {
            let mid = a.midpoint(b);
            if mid == a || mid == b {
                return mid;
            }
            if (f(mid) > 0.0) == positive_at_a {
                a = mid;
            } else {
                b = mid;
            }
        }
    };
    let x0 = c.ln();
    let bracket = |sign: f64| {
        let mut s = 1.0;
        while f(x0 + sign * s) <= 0.0 {
            s *= 2.0;
        }
        x0 + sign * s
    };
    let (lo, hi) = (root(bracket(-1.0), x0), root(x0, bracket(1.0)));
    RateBin {
        c,
        lo: (lo.exp(), lo),
        hi: (hi.exp(), hi),
    }
}

fn insert_count(events: &mut Vec<CountEvent>, bin: &RateBin) {
    let clc = if bin.c > 0.0 { bin.c * bin.c.ln() } else { 0.0 };
    for ((at, ln_at), enter) in [(bin.lo, true), (bin.hi, false)] {
        let i = events.partition_point(|x| x.at <= at);
        events.insert(
            i,
            CountEvent {
                at,
                ln_at,
                c: bin.c,
                clc,
                enter,
            },
        );
    }
}

/// Exact `min over μ of Σ min(D(c, μ) / φ, K²)` with the Poisson deviance D.
/// Between events the inlier set is fixed and its deviance is convex in μ,
/// minimized at the inlier mean count clamped to the interval.
fn poisson_min(events: &[CountEvent], points: usize, phi: f64) -> f64 {
    let mut best = K2 * points as f64;
    let (mut m, mut s, mut slc) = (0_usize, 0.0, 0.0);
    for (k, e) in events.iter().enumerate() {
        if e.enter {
            (m, s, slc) = (m + 1, s + e.c, slc + e.clc);
        } else {
            (m, s, slc) = (m - 1, s - e.c, slc - e.clc);
            if m == 0 {
                (s, slc) = (0.0, 0.0);
            }
        }
        if m > 0 {
            let (end, ln_end) = events
                .get(k + 1)
                .map_or((e.at, e.ln_at), |n| (n.at, n.ln_at));
            let mean = s / m as f64;
            let half = if s == 0.0 {
                m as f64 * e.at
            } else {
                let (mu, ln_mu) = if mean < e.at {
                    (e.at, e.ln_at)
                } else if mean > end {
                    (end, ln_end)
                } else {
                    (mean, mean.ln())
                };
                slc - s * ln_mu - s + m as f64 * mu
            };
            best = best.min(2.0 * half.max(0.0) / phi + K2 * (points - m) as f64);
        }
    }
    best
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_unstable_by(f64::total_cmp);
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        v[m - 1].midpoint(v[m])
    }
}

/// Robust noise scale of a series of `(value, weight)` points. Each
/// difference scaled by `sqrt(w_i w_j / (w_i + w_j))` has the unit-weight
/// scale σ for independent points. `v(k)` is the robust variance of those
/// differences at lag k: the squared Gaussian-consistent MAD of the absolute
/// differences (not centred, so periodic or quantized series do not give
/// zero), or of their mean absolute value if the median is zero. Isolated
/// outliers and a few steps barely move either.
///
/// For independent bins v(1) = σ². With `long_run`, allow for a lag-one
/// correlation r between bins (spike amplitudes correlated across a bin
/// edge), which shrinks v(1) to σ²(1 − r) while v(3) ≈ σ² once the
/// correlation has died out: return the larger of v(1) and the lag-one
/// long-run variance σ²(1 + 2r) = 3 v(3) − 2 v(1), the spread that segment
/// means actually have.
fn noise_scale(points: &[(f64, f64)], long_run: bool) -> f64 {
    let v = |lag: usize| {
        let d: Vec<f64> = points
            .iter()
            .zip(&points[lag..])
            .map(|(p, q)| (q.0 - p.0).abs() * (p.1 * q.1 / (p.1 + q.1)).sqrt())
            .collect();
        let mean_abs = d.iter().sum::<f64>() / d.len() as f64;
        let mad = MAD_TO_SD * median(d);
        let s = if mad > 0.0 {
            mad
        } else {
            (std::f64::consts::PI / 2.0).sqrt() * mean_abs
        };
        s * s
    };
    let v1 = v(1);
    if long_run {
        v1.max(3.0 * v(3) - 2.0 * v1).sqrt()
    } else {
        v1.sqrt()
    }
}

/// Fit a constant firing rate and (if available) amplitude level over 100
/// equal recording bins, with at least five bins per segment.
///
/// Rate: each bin costs `min(D(c, μ) / φ, 9)`, its Poisson deviance from the
/// segment's mean count μ divided by the dispersion φ, the squared
/// first-difference noise scale of the Anscombe-transformed counts
/// `2 sqrt(c + 3/8)` (at least one, the Poisson value). Amplitude: the bin
/// mean amplitudes of nonempty bins, weighted by spike count and divided by
/// the per-spike noise scale from first differences, widened to the
/// long-run scale when neighbouring bins are correlated (see `noise_scale`);
/// each bin costs `min(w (y - θ)², 9)` at the segment's level θ. Either way a
/// bin more than about three noise units from its segment's level (a burst,
/// a stray bin) costs a bounded amount instead of buying a boundary.
///
/// Each fitted level costs ln(100) (one rate per segment, one amplitude per
/// nonempty segment when fitted), and each boundary costs 2 ln(91) for the
/// search over the 91 possible interior edges. The optimum is exact.
///
/// Returns empty with fewer than ten spikes (twice the minimum bin count),
/// invalid duration/sample rate, or times outside `[0, total_samples)`.
/// Missing/invalid amplitudes, or fewer than 20 consecutive pairs of
/// nonempty bins, reduce the fit to rates only. Observed amplitude means are
/// still reported if the amplitude data are valid.
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
    let mut counts = [0_u32; BINS];
    let mut sums = [0.0_f64; BINS];
    for (i, &t) in times.iter().enumerate() {
        if t.0 >= total_samples {
            return Vec::new();
        }
        // Exact edges, including for recordings not divisible by 100 samples.
        let b = (u128::from(t.0) * BINS as u128 / u128::from(total_samples)) as usize;
        counts[b] += 1;
        if has_amp {
            sums[b] += f64::from(amps[i]);
        }
    }

    let anscombe = counts.map(|c| 2.0 * (f64::from(c) + 0.375).sqrt());
    // Poisson counts have unit variance on this scale, and discrete low
    // counts often have a zero median difference. Bursts last milliseconds,
    // so they do not correlate counts across bins. The deviance (not the
    // transformed count) is what is fitted: near silence a single spike is
    // strong evidence of firing, which a Gaussian on the transform misses.
    let sigma_rate = noise_scale(&anscombe.map(|y| (y, 1.0)), false).max(1.0);
    // Estimate count dispersion from first differences winsorized past the
    // top ~8% of jumps so a few step boundaries or isolated bursts do not
    // inflate φ, while recurrent bursts (whether sparse or dense) still
    // reflect their true compound-Poisson overdispersion.
    let count_phi = {
        let mean_c = counts.iter().map(|&c| f64::from(c)).sum::<f64>() / BINS as f64;
        if mean_c > 0.0 {
            let mut diffs: Vec<f64> = counts
                .windows(2)
                .map(|w| (f64::from(w[1]) - f64::from(w[0])).abs())
                .collect();
            diffs.sort_unstable_by(f64::total_cmp);
            let cap = diffs[diffs.len().saturating_sub(9)];
            let var = diffs
                .iter()
                .map(|&d| {
                    let w = d.min(cap);
                    0.5 * w * w
                })
                .sum::<f64>()
                / diffs.len() as f64;
            var / mean_c
        } else {
            1.0
        }
    };
    let phi = (sigma_rate * sigma_rate).max(count_phi);
    let rate = counts.map(|c| rate_bin(f64::from(c), K2 * phi));
    let nonempty: Vec<_> = (0..BINS)
        .filter(|&b| counts[b] > 0)
        .map(|b| (sums[b] / f64::from(counts[b]), f64::from(counts[b])))
        .collect();
    // Never below one f32 epsilon at the amplitude scale (identical bins).
    let floor = f64::from(f32::EPSILON)
        * nonempty
            .iter()
            .map(|&(m, _)| m.abs())
            .fold(f64::from(f32::MIN_POSITIVE), f64::max);
    let sigma_amp = (has_amp && nonempty.len() > MIN_AMP_PAIRS)
        .then(|| noise_scale(&nonempty, true).max(floor));
    let amp: [Option<Point>; BINS] = match sigma_amp {
        Some(sigma) => {
            // Centre before scaling so a tiny noise scale keeps precision.
            let centre = median(nonempty.iter().map(|&(m, _)| m).collect());
            std::array::from_fn(|b| {
                (counts[b] > 0).then(|| Point {
                    y: (sums[b] / f64::from(counts[b]) - centre) / sigma,
                    w: f64::from(counts[b]),
                })
            })
        }
        None => [None; BINS],
    };

    let level_penalty = (BINS as f64).ln();
    let boundary_penalty = 2.0 * ((BINS - 2 * MIN_BINS + 1) as f64).ln();
    let mut best = [f64::INFINITY; BINS + 1];
    let mut previous = [0; BINS + 1];
    best[0] = 0.0;
    // Exact optimal partitioning. ponytail: use PELT pruning if B grows;
    // a quadratic search over 100 bins is small and has no pruning assumptions.
    for start in 0..=BINS - MIN_BINS {
        if !best[start].is_finite() {
            continue;
        }
        let base = best[start] + level_penalty + if start == 0 { 0.0 } else { boundary_penalty };
        let (mut rate_events, mut amp_events) = (Vec::new(), Vec::new());
        let mut amp_bins = 0;
        for end in start + 1..=BINS {
            insert_count(&mut rate_events, &rate[end - 1]);
            if let Some(p) = amp[end - 1] {
                insert(&mut amp_events, p);
                amp_bins += 1;
            }
            if end - start < MIN_BINS {
                continue;
            }
            let mut cost = base + poisson_min(&rate_events, end - start, phi);
            if amp_bins > 0 {
                cost += biweight_min(&amp_events, amp_bins) + level_penalty;
            }
            if cost < best[end] {
                best[end] = cost;
                previous[end] = start;
            }
        }
    }

    // The first sample of bin b, so a time split at a boundary reproduces
    // the exact half-open integer partition.
    let edge = |b: usize| (u128::from(total_samples) * b as u128).div_ceil(BINS as u128) as u64;
    let mut epochs = Vec::new();
    let mut end = BINS;
    while end > 0 {
        let start = previous[end];
        let n: u32 = counts[start..end].iter().sum();
        let exposure = (edge(end) - edge(start)) as f64 / f64::from(sr);
        let rate_hz = (f64::from(n) / exposure) as f32;
        if !rate_hz.is_finite() {
            return Vec::new();
        }
        epochs.push(Epoch {
            start_s: edge(start) as f64 / f64::from(sr),
            end_s: edge(end) as f64 / f64::from(sr),
            n,
            rate_hz,
            amp_mean: if has_amp && n > 0 {
                (sums[start..end].iter().sum::<f64>() / f64::from(n)) as f32
            } else {
                f32::NAN
            },
            amp_fitted: sigma_amp.is_some() && n > 0,
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

    /// SplitMix64, uniform in (0, 1). A plain LCG from small seeds gave
    /// visibly overdispersed spike trains.
    fn rng(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1_u64 << 53) as f64 + 0.5 / (1_u64 << 53) as f64
        }
    }

    fn train(
        seed: u64,
        rate: impl Fn(usize) -> f64,
        amp: impl Fn(usize) -> f32,
    ) -> (Vec<SampleIndex>, Vec<f32>) {
        let mut uniform = rng(seed);
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
    fn biweight_minimum_matches_a_grid() {
        let mut uniform = rng(7);
        for _ in 0..20 {
            let points: Vec<_> = (0..12)
                .map(|i| Point {
                    y: 10.0 * uniform() + if i % 4 == 0 { 20.0 } else { 0.0 },
                    w: 1.0 + 4.0 * uniform(),
                })
                .collect();
            let mut events = Vec::new();
            points.iter().for_each(|&p| insert(&mut events, p));
            let loss = |theta: f64| -> f64 {
                points
                    .iter()
                    .map(|p| (p.w * (p.y - theta).powi(2)).min(K2))
                    .sum()
            };
            let grid = (0..=400_000)
                .map(|i| loss(-5.0 + i as f64 * 1e-4))
                .fold(f64::INFINITY, f64::min);
            let exact = biweight_min(&events, points.len());
            assert!(
                exact <= grid + 1e-9 && grid - exact < 1e-6,
                "{exact} {grid}"
            );
        }
    }

    #[test]
    fn bounded_deviance_minimum_matches_a_grid() {
        let mut uniform = rng(11);
        for phi in [1.0, 2.7] {
            for _ in 0..20 {
                let counts: Vec<f64> = (0..12)
                    .map(|i| (12.0 * uniform()).floor() + if i % 5 == 0 { 40.0 } else { 0.0 })
                    .collect();
                let mut events = Vec::new();
                for &c in &counts {
                    insert_count(&mut events, &rate_bin(c, K2 * phi));
                }
                let deviance = |c: f64, mu: f64| {
                    2.0 * (if c > 0.0 { c * (c / mu).ln() } else { 0.0 } - c + mu)
                };
                let loss = |mu: f64| -> f64 {
                    counts
                        .iter()
                        .map(|&c| (deviance(c, mu) / phi).min(K2))
                        .sum()
                };
                let grid = (1..=400_000)
                    .map(|i| loss(i as f64 * 1.5e-4))
                    .fold(f64::INFINITY, f64::min);
                let exact = poisson_min(&events, counts.len(), phi);
                assert!(
                    exact <= grid + 1e-9 && grid - exact < 1e-6,
                    "{exact} {grid}"
                );
            }
        }
    }

    #[test]
    fn single_spike_bins_stay_with_firing() {
        // A one-spike bin next to silence is far more likely under firing
        // than under silence; a Gaussian on transformed counts said otherwise.
        let times: Vec<_> = (0..100)
            .flat_map(|b| {
                let n = match b {
                    29 => 1,
                    30..70 => 0,
                    _ => 5,
                };
                (0..n).map(move |i| SampleIndex(b * 1000 + i * 1000 / n))
            })
            .collect();
        let e = stable_epochs(&times, &[], TOTAL, SR);
        assert_eq!(e.len(), 3, "{e:?}");
        assert_eq!((e[1].start_s, e[1].end_s, e[1].n), (30.0, 70.0, 0));
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
            let (times, amps) = train(
                seed,
                |b| if [20, 36, 70].contains(&b) { 75.0 } else { 5.0 },
                |_| 100.0,
            );
            assert_eq!(stable_epochs(&times, &[], TOTAL, SR).len(), 1);
            assert_eq!(stable_epochs(&times, &amps, TOTAL, SR).len(), 1);
        }
    }

    #[test]
    fn recurrent_bursts_rounded_and_correlated_amplitudes_do_not_split() {
        for seed in 0..5 {
            let mut uniform = rng(seed);
            let mut gauss =
                move || (-2.0 * uniform().ln()).sqrt() * (std::f64::consts::TAU * uniform()).cos();
            let (times, _) = train(seed + 100, |_| 5.0, |_| 0.0);
            // Integer-rounded amplitudes with SD 0.2 leave almost every bin identical.
            let rounded: Vec<_> = times
                .iter()
                .map(|_| (100.0 + 0.2 * gauss()).round() as f32)
                .collect();
            assert_eq!(stable_epochs(&times, &rounded, TOTAL, SR).len(), 1);
            // AR(1) amplitudes, correlation 0.75 between consecutive spikes.
            let mut x = 0.0;
            let correlated: Vec<_> = times
                .iter()
                .map(|_| {
                    x = 0.75 * x + (1.0 - 0.75_f64.powi(2)).sqrt() * gauss();
                    (100.0 + 10.0 * x) as f32
                })
                .collect();
            assert_eq!(stable_epochs(&times, &correlated, TOTAL, SR).len(), 1);
            // Eight-spike bursts, 4 ms apart, starting as a 0.2 Hz Poisson process.
            let mut bursty = times.clone();
            let mut onset = rng(seed + 200);
            let mut t = 0.0;
            loop {
                t -= onset().ln() / 0.2;
                if t >= 99.9 {
                    break;
                }
                bursty
                    .extend((0..8).map(|i| SampleIndex(((t + 0.004 * i as f64) * 1000.0) as u64)));
            }
            bursty.sort_unstable_by_key(|s| s.0);
            assert_eq!(stable_epochs(&bursty, &[], TOTAL, SR).len(), 1);
        }
    }

    #[test]
    fn burst_only_units_rarely_split_while_silence_stays_detectable() {
        // Stationary 100 s recordings with 0.2 Hz (mostly-zero bin diffs) and
        // 0.4 Hz (nonzero median bin diff) Poisson burst onsets (8 spikes,
        // 4 ms apart) and no tonic background.
        for (rate_hz, seed_base) in [(0.2, 10_000_u64), (0.4, 30_000_u64)] {
            let mut splits_rate = 0;
            let mut splits_amp = 0;
            for seed in 0..200_u64 {
                let mut onset = rng(seed + seed_base);
                let mut times = Vec::new();
                let mut t = 0.0;
                loop {
                    t -= onset().ln() / rate_hz;
                    if t >= 99.9 {
                        break;
                    }
                    times.extend(
                        (0..8).map(|i| SampleIndex(((t + 0.004 * i as f64) * 1000.0) as u64)),
                    );
                }
                if times.len() < MIN_SPIKES {
                    continue;
                }
                if stable_epochs(&times, &[], TOTAL, SR).len() > 1 {
                    splits_rate += 1;
                }
                let amps = vec![100.0; times.len()];
                if stable_epochs(&times, &amps, TOTAL, SR).len() > 1 {
                    splits_amp += 1;
                }
            }
            assert!(
                splits_rate <= 12,
                "burst-only ({rate_hz} Hz) rate false splits: {splits_rate}/200"
            );
            assert!(
                splits_amp <= 12,
                "burst-only ({rate_hz} Hz) amp false splits: {splits_amp}/200"
            );
        }

        // Sustained silence on a burst-only train remains detectable:
        // 0.35 Hz bursts for 60 s, then 40 s of silence. Check that the
        // silent tail is found and starts near the last burst.
        let mut detected = 0;
        for seed in 0..50_u64 {
            let mut onset = rng(seed + 20_000);
            let mut times = Vec::new();
            let mut t = 0.0;
            loop {
                t -= onset().ln() / 0.35;
                if t >= 60.0 {
                    break;
                }
                times.extend((0..8).map(|i| SampleIndex(((t + 0.004 * i as f64) * 1000.0) as u64)));
            }
            let last_s = times.last().unwrap().0 as f64 / f64::from(SR);
            let e = stable_epochs(&times, &[], TOTAL, SR);
            if e.len() >= 2
                && e.last()
                    .is_some_and(|last| last.n == 0 && (last.start_s - last_s).abs() <= 2.0)
            {
                detected += 1;
            }
        }
        assert!(
            detected >= 45,
            "burst-only silence detection: {detected}/50"
        );
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
    fn rates_use_integer_edge_exposure() {
        // 1003 samples: bins hold 10 or 11 samples, so nominal widths would be wrong.
        let times: Vec<_> = (0..1003).step_by(3).map(SampleIndex).collect();
        let e = stable_epochs(&times, &[], 1003, SR);
        assert_eq!(e.len(), 1, "{e:?}");
        assert_eq!(e[0].rate_hz, (times.len() as f64 / 1.003) as f32);
    }

    #[test]
    fn one_matching_pair_cannot_set_the_amplitude_noise() {
        let mut times: Vec<_> = (0..100).map(|b| SampleIndex(b * 1000)).collect();
        let mut amps: Vec<_> = (0..100).map(|b| 95.0 + (b * 7 % 11) as f32).collect();
        times.insert(1, SampleIndex(1));
        amps.insert(1, amps[0]);
        let e = stable_epochs(&times, &amps, TOTAL, SR);
        assert_eq!(e.len(), 1, "{e:?}");
        assert!(e[0].amp_fitted && e[0].amp_mean.is_finite());
    }

    #[test]
    fn silence_has_no_amplitude_parameter() {
        // Splitting off the five silent bins gains ~17.5 in the rate deviance,
        // which pays for a boundary and a rate (~13.6) but would not also pay
        // for an amplitude level (~4.6) on the silent segment.
        let times: Vec<_> = (5..100)
            .flat_map(|b| {
                (0..if b % 5 == 0 { 1 } else { 2 }).map(move |i| SampleIndex(b * 1000 + i * 500))
            })
            .collect();
        for amps in [&[][..], &vec![100.0; times.len()]] {
            let e = stable_epochs(&times, amps, TOTAL, SR);
            assert_eq!(e.len(), 2, "{e:?}");
            assert_eq!((e[0].end_s, e[0].n, e[0].amp_fitted), (5.0, 0, false));
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
