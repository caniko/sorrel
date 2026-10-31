//! Waveform and footprint descriptors: the physical quantities a curator
//! reads off a template (spike width, repolarisation, spatial spread,
//! propagation) plus SAX strings for compact numeric series.
//!
//! Pure functions over plain slices; templates are row-major
//! `[sample * n_channels + channel]`.

/// A positive-going unit is flipped when `|max| > INVERT_RATIO * |min|`.
const INVERT_RATIO: f32 = 1.5;
/// Extrema below this fraction of the peak-to-peak are noise wiggles.
const EXTREMUM_FRAC: f32 = 0.2;

/// Shape of one channel's template trace. All times in ms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaveformLandmarks {
    /// Deepest value of the (possibly negated) trace; always the minimum.
    pub trough: f32,
    /// True when the trace was negated because the unit is positive-going.
    pub inverted: bool,
    /// Max value after the trough over `|trough|`.
    pub peak_trough_ratio: f32,
    pub trough_to_peak_ms: f32,
    /// Width of the trough at half its depth, linearly interpolated.
    pub half_width_ms: f32,
    /// `(peak - trough) / trough_to_peak_ms`; 0 when the trace never recovers.
    pub repolarisation_slope: f32,
    /// Local extrema with `|value| >= 20%` of peak-to-peak. >= 3 hints at noise.
    pub n_extrema: u32,
}

/// Landmarks of one channel's trace. A unit whose positive deflection
/// exceeds 1.5x its negative one is analysed on the negated trace
/// (`inverted`). `None` for a trace under 3 samples, a flat trace or a
/// non-positive sample rate.
pub fn waveform_landmarks(trace: &[f32], sample_rate: f32) -> Option<WaveformLandmarks> {
    if trace.len() < 3 || sample_rate <= 0.0 {
        return None;
    }
    let (lo, hi) = trace
        .iter()
        .fold((f32::MAX, f32::MIN), |(l, h), &v| (l.min(v), h.max(v)));
    let ptp = hi - lo;
    if !ptp.is_finite() || ptp <= f32::EPSILON {
        return None;
    }
    let inverted = hi.abs() > INVERT_RATIO * lo.abs();
    let v: Vec<f32> = if inverted {
        trace.iter().map(|x| -x).collect()
    } else {
        trace.to_vec()
    };
    let ms = 1000.0 / sample_rate;
    let t = (0..v.len()).fold(0, |b, i| if v[i] < v[b] { i } else { b });
    let trough = v[t];
    let p = (t..v.len()).fold(t, |b, i| if v[i] > v[b] { i } else { b });
    let peak = v[p];
    let t2p = (p - t) as f32 * ms;

    // Half-depth crossings, walking out from the trough; an edge is used
    // when the trace never climbs back above half depth.
    let half = trough / 2.0;
    let mut l = t as f32;
    for i in (0..t).rev() {
        if v[i] > half {
            l = i as f32 + (v[i] - half) / (v[i] - v[i + 1]);
            break;
        }
        l = i as f32;
    }
    let mut r = t as f32;
    for i in t + 1..v.len() {
        if v[i] > half {
            r = (i - 1) as f32 + (half - v[i - 1]) / (v[i] - v[i - 1]);
            break;
        }
        r = i as f32;
    }

    let floor = EXTREMUM_FRAC * ptp;
    let n_extrema = (1..v.len() - 1)
        .filter(|&i| {
            let (a, b, c) = (v[i - 1], v[i], v[i + 1]);
            b.abs() >= floor && ((b > a && b >= c) || (b < a && b <= c))
        })
        .count() as u32;

    Some(WaveformLandmarks {
        trough,
        inverted,
        peak_trough_ratio: peak / trough.abs(),
        trough_to_peak_ms: t2p,
        half_width_ms: (r - l) * ms,
        repolarisation_slope: if p > t { (peak - trough) / t2p } else { 0.0 },
        n_extrema,
    })
}

/// Spatial summary of one template. Distances in micrometres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Footprint {
    pub peak_channel: usize,
    /// Centroid weighted by per-channel ptp².
    pub x_um: f32,
    pub y_um: f32,
    /// Exponential decay length λ of ptp with distance from the peak channel
    /// (`ptp ~ exp(-d/λ)`); NaN when not decaying or under 3 channels fit.
    pub decay_um: f32,
    /// Smallest radius around the centroid holding 80% of Σptp².
    pub r80_um: f32,
    /// Channels with ptp >= 20% of the maximum.
    pub n_active: u32,
    /// Slope of trough time vs distance from the peak channel over active
    /// channels, ms per 100 µm. ~0 for somatic units, large for axonal
    /// ones; NaN under 3 active channels.
    pub propagation_ms_per_100um: f32,
}

/// Peak-to-peak of every channel of a row-major template.
fn channel_ptp(template: &[f32], n_samples: usize, n_channels: usize) -> Vec<f32> {
    (0..n_channels)
        .map(|c| {
            let (lo, hi) = (0..n_samples)
                .map(|s| template[s * n_channels + c])
                .fold((f32::MAX, f32::MIN), |(l, h), v| (l.min(v), h.max(v)));
            hi - lo
        })
        .collect()
}

/// Channel with the largest peak-to-peak (lowest index on ties); `None` for
/// an empty or mis-sized template or one that is flat everywhere.
pub fn peak_channel(template: &[f32], n_samples: usize, n_channels: usize) -> Option<usize> {
    if n_samples == 0 || n_channels == 0 || template.len() != n_samples * n_channels {
        return None;
    }
    let ptp = channel_ptp(template, n_samples, n_channels);
    let best = (0..n_channels).fold(0, |b, c| if ptp[c] > ptp[b] { c } else { b });
    (ptp[best] > 0.0 && ptp[best].is_finite()).then_some(best)
}

/// Least-squares slope of y on x; `None` under 3 points or constant x.
fn lsq_slope(pts: &[(f64, f64)]) -> Option<f64> {
    let n = pts.len() as f64;
    if pts.len() < 3 {
        return None;
    }
    let (mx, my) = (
        pts.iter().map(|p| p.0).sum::<f64>() / n,
        pts.iter().map(|p| p.1).sum::<f64>() / n,
    );
    let sxx: f64 = pts.iter().map(|p| (p.0 - mx).powi(2)).sum();
    let sxy: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    (sxx > 0.0).then(|| sxy / sxx)
}

fn dist(a: [f32; 2], b: [f32; 2]) -> f64 {
    ((a[0] - b[0]) as f64).hypot((a[1] - b[1]) as f64)
}

/// Footprint of a row-major template on a probe. `None` when the template
/// or `positions` (one per channel) is mis-sized, or the template is flat.
pub fn footprint(
    template: &[f32],
    n_samples: usize,
    n_channels: usize,
    positions: &[[f32; 2]],
    sample_rate: f32,
) -> Option<Footprint> {
    if positions.len() != n_channels || sample_rate <= 0.0 {
        return None;
    }
    let pk = peak_channel(template, n_samples, n_channels)?;
    let ptp = channel_ptp(template, n_samples, n_channels);
    let max = ptp[pk];
    let w: Vec<f64> = ptp.iter().map(|&p| (p as f64).powi(2)).collect();
    let wsum: f64 = w.iter().sum();
    let cx = w
        .iter()
        .zip(positions)
        .map(|(w, p)| w * p[0] as f64)
        .sum::<f64>()
        / wsum;
    let cy = w
        .iter()
        .zip(positions)
        .map(|(w, p)| w * p[1] as f64)
        .sum::<f64>()
        / wsum;
    let centroid = [cx as f32, cy as f32];

    // decay: ln(ptp) vs distance over channels above 5% of max.
    let fit: Vec<(f64, f64)> = (0..n_channels)
        .filter(|&c| ptp[c] >= 0.05 * max)
        .map(|c| (dist(positions[c], positions[pk]), (ptp[c] as f64).ln()))
        .collect();
    let decay_um = match lsq_slope(&fit) {
        Some(s) if s < 0.0 => (-1.0 / s) as f32,
        _ => f32::NAN,
    };

    // r80: grow a disc around the centroid until it holds 80% of Σptp².
    let mut by_r: Vec<(f64, f64)> = (0..n_channels)
        .map(|c| (dist(positions[c], centroid), w[c]))
        .collect();
    by_r.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut acc = 0.0;
    let r80_um = by_r
        .iter()
        .find(|(_, w)| {
            acc += w;
            acc >= 0.8 * wsum
        })
        .map_or(f32::NAN, |p| p.0 as f32);

    // propagation: trough time (ms) vs distance over active channels.
    let ms = 1000.0 / sample_rate as f64;
    let active: Vec<usize> = (0..n_channels).filter(|&c| ptp[c] >= 0.2 * max).collect();
    let prop: Vec<(f64, f64)> = active
        .iter()
        .map(|&c| {
            // Positive-going channels (same rule as `waveform_landmarks`) are
            // timed at their maximum.
            let v = |s: usize| template[s * n_channels + c];
            let (lo, hi) =
                (0..n_samples).fold((f32::MAX, f32::MIN), |(l, h), s| (l.min(v(s)), h.max(v(s))));
            let sign = if hi.abs() > INVERT_RATIO * lo.abs() {
                -1.0
            } else {
                1.0
            };
            let t = (0..n_samples).fold(0, |b, s| if sign * v(s) < sign * v(b) { s } else { b });
            (dist(positions[c], positions[pk]), t as f64 * ms)
        })
        .collect();
    let propagation_ms_per_100um = lsq_slope(&prop).map_or(f32::NAN, |s| (s * 100.0) as f32);

    Some(Footprint {
        peak_channel: pk,
        x_um: cx as f32,
        y_um: cy as f32,
        decay_um,
        r80_um,
        n_active: active.len() as u32,
        propagation_ms_per_100um,
    })
}

/// Mean count over bins covering lags `[lo_ms, hi_ms)` of the positive half
/// of a centred ACG; NaN when the ACG does not reach `hi_ms`.
fn acg_mean(acg: &[u32], bin_ms: f32, lo_ms: f32, hi_ms: f32) -> f32 {
    let h = acg.len() / 2;
    let (a, b) = (
        h + (lo_ms / bin_ms).round() as usize,
        h + (hi_ms / bin_ms).round() as usize,
    );
    if bin_ms <= 0.0 || b > acg.len() || a >= b {
        return f32::NAN;
    }
    acg[a..b].iter().map(|&c| c as f32).sum::<f32>() / (b - a) as f32
}

/// Burstiness: mean ACG count at 2-10 ms lags over the mean at 30-50 ms.
/// `acg` is the centred two-sided histogram `auto_correlogram` produces
/// (`2 * window / bin` bins; bin `k` covers lags `[(k - len/2) * bin,
/// (k - len/2 + 1) * bin)`), so only the positive half `acg[len/2..]` is
/// read. NaN when the window is shorter than 50 ms or the baseline is 0.
pub fn burst_index(acg: &[u32], bin_ms: f32) -> f32 {
    let base = acg_mean(acg, bin_ms, 30.0, 50.0);
    if base > 0.0 {
        acg_mean(acg, bin_ms, 2.0, 10.0) / base
    } else {
        f32::NAN
    }
}

/// Sum the positive half of a centred ACG (see [`burst_index`]) into lag
/// bins `[edges_ms[k], edges_ms[k+1])`, as rate per ms of lag so unequal
/// bins compare. Linear bins are split fractionally across edges.
pub fn log_bins(counts: &[u32], bin_ms: f32, edges_ms: &[f32]) -> Vec<f32> {
    let pos = &counts[counts.len() / 2..];
    edges_ms
        .windows(2)
        .map(|e| {
            let sum: f32 = pos
                .iter()
                .enumerate()
                .map(|(i, &c)| {
                    let lo = i as f32 * bin_ms;
                    let ov = (lo + bin_ms).min(e[1]) - lo.max(e[0]);
                    c as f32 * (ov / bin_ms).max(0.0)
                })
                .sum();
            sum / (e[1] - e[0])
        })
        .collect()
}

/// Gaussian equiprobable breakpoints (Lin et al. 2003), alphabet 3..=7.
const SAX_BREAKS: [&[f32]; 5] = [
    &[-0.43, 0.43],
    &[-0.67, 0.0, 0.67],
    &[-0.84, -0.25, 0.25, 0.84],
    &[-0.97, -0.43, 0.0, 0.43, 0.97],
    &[-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
];

/// Symbolic Aggregate approXimation: z-normalise (population std), average
/// into `segments` equal runs, letter each by Gaussian breakpoints
/// (`'a'` lowest). `alphabet` is clamped to 3..=7. A constant series maps to
/// the middle letter throughout.
pub fn sax(values: &[f32], segments: usize, alphabet: u8) -> String {
    if values.is_empty() || segments == 0 {
        return String::new();
    }
    let alphabet = alphabet.clamp(3, 7);
    let breaks = SAX_BREAKS[alphabet as usize - 3];
    let n = values.len();
    let mean = values.iter().sum::<f32>() / n as f32;
    let sd = (values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32).sqrt();
    (0..segments)
        .map(|i| {
            let (a, b) = (i * n / segments, (i + 1) * n / segments);
            let seg = &values[a.min(n - 1)..b.max(a + 1).min(n)];
            let z = if sd > 1e-9 {
                (seg.iter().sum::<f32>() / seg.len() as f32 - mean) / sd
            } else {
                0.0
            };
            let k = if sd > 1e-9 {
                breaks.iter().filter(|&&b| z > b).count()
            } else {
                (alphabet as usize - 1) / 2
            };
            (b'a' + k as u8) as char
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gauss(i: usize, mu: f32, sigma: f32) -> f32 {
        (-((i as f32 - mu).powi(2)) / (2.0 * sigma * sigma)).exp()
    }

    #[test]
    fn biphasic_landmarks_match_construction() {
        let sr = 30_000.0;
        let tr: Vec<f32> = (0..80)
            .map(|i| -gauss(i, 30.0, 3.0) + 0.4 * gauss(i, 50.0, 6.0))
            .collect();
        let l = waveform_landmarks(&tr, sr).unwrap();
        assert!(!l.inverted);
        assert!((l.trough + 1.0).abs() < 5e-3);
        assert!((l.trough_to_peak_ms - 20.0 / 30.0).abs() < 1e-4);
        assert!((l.peak_trough_ratio - 0.4).abs() < 0.01);
        // FWHM of a sigma = 3 samples gaussian is 7.065 samples.
        assert!((l.half_width_ms - 7.065 / 30.0).abs() < 0.01, "{l:?}");
        assert!((l.repolarisation_slope - 1.4 / (20.0 / 30.0)).abs() < 0.05);
        assert_eq!(l.n_extrema, 2);
    }

    #[test]
    fn positive_going_unit_is_analysed_inverted() {
        let tr: Vec<f32> = (0..80).map(|i| gauss(i, 30.0, 3.0)).collect();
        let l = waveform_landmarks(&tr, 30_000.0).unwrap();
        assert!(l.inverted && l.trough < 0.0);
        assert!((l.half_width_ms - 7.065 / 30.0).abs() < 0.01);
        assert!(waveform_landmarks(&[0.0; 10], 30_000.0).is_none());
    }

    #[test]
    fn multi_peak_trace_counts_extrema() {
        let tr: Vec<f32> = (0..90)
            .map(|i| gauss(i, 20.0, 2.0) - gauss(i, 40.0, 2.0) + gauss(i, 60.0, 2.0))
            .collect();
        assert!(waveform_landmarks(&tr, 30_000.0).unwrap().n_extrema >= 3);
    }

    /// 16-channel linear probe at 20 µm; sample 5 holds each channel's
    /// trough of `amp(d)` at sample `t(d)`, d = distance from channel 4.
    fn probe_template(
        amp: impl Fn(f32) -> f32,
        t: impl Fn(f32) -> usize,
    ) -> (Vec<f32>, Vec<[f32; 2]>) {
        let (ns, nc) = (40, 16);
        let pos: Vec<[f32; 2]> = (0..nc).map(|c| [0.0, 20.0 * c as f32]).collect();
        let mut tpl = vec![0.0; ns * nc];
        for (c, p) in pos.iter().enumerate() {
            let d = (p[1] - pos[4][1]).abs();
            tpl[t(d) * nc + c] = -amp(d);
        }
        (tpl, pos)
    }

    #[test]
    fn exponential_decay_recovers_lambda() {
        let (tpl, pos) = probe_template(|d| (-d / 40.0).exp(), |_| 5);
        let f = footprint(&tpl, 40, 16, &pos, 30_000.0).unwrap();
        assert_eq!(f.peak_channel, 4);
        assert!((f.decay_um - 40.0).abs() < 4.0, "{f:?}");
        // Single-sample troughs at one time: no propagation.
        assert!(f.propagation_ms_per_100um.abs() < 1e-3);
        assert!(f.y_um > 70.0 && f.y_um < 95.0 && f.r80_um < 60.0);
    }

    #[test]
    fn trough_time_gradient_recovers_propagation() {
        // Flat amplitude over 11 channels around channel 4 would clip at the
        // probe edge, so use a slow decay; 2 samples of delay per 20 µm.
        let (tpl, pos) = probe_template(|d| (-d / 400.0).exp(), |d| 5 + (d / 10.0) as usize);
        let f = footprint(&tpl, 40, 16, &pos, 30_000.0).unwrap();
        // 2 samples = 0.0667 ms per 20 µm = 0.333 ms per 100 µm.
        assert!((f.propagation_ms_per_100um - 0.333).abs() < 0.02, "{f:?}");
        assert_eq!(f.n_active, 16);
        assert!(footprint(&tpl, 40, 16, &pos[..8], 30_000.0).is_none());
    }

    #[test]
    fn positive_going_channels_are_timed_at_their_maximum() {
        let (mut tpl, pos) = probe_template(|d| (-d / 400.0).exp(), |d| 5 + (d / 10.0) as usize);
        tpl.iter_mut().for_each(|x| *x = -*x);
        let f = footprint(&tpl, 40, 16, &pos, 30_000.0).unwrap();
        assert!((f.propagation_ms_per_100um - 0.333).abs() < 0.02, "{f:?}");
    }

    #[test]
    fn burst_index_reads_positive_half() {
        // 100 ms window at 1 ms: 200 bins, lag 0 between bins 99 and 100.
        let mut acg = vec![10u32; 200];
        for c in &mut acg[102..110] {
            *c = 40;
        }
        assert!((burst_index(&acg, 1.0) - 4.0).abs() < 1e-6);
        assert!(burst_index(&acg[..60], 1.0).is_nan());
        assert!(burst_index(&[0; 200], 1.0).is_nan());
    }

    #[test]
    fn log_bins_conserves_counts() {
        let acg = vec![2u32; 100];
        let out = log_bins(&acg, 1.0, &[0.0, 1.0, 2.0, 4.0, 8.0, 50.0]);
        assert_eq!(out.len(), 5);
        assert!(out.iter().all(|&r| (r - 2.0).abs() < 1e-6));
    }

    #[test]
    fn sax_ramp_is_monotone_and_constant_is_middle() {
        let ramp: Vec<f32> = (0..100).map(|i| i as f32).collect();
        assert_eq!(sax(&ramp, 5, 5), "abcde");
        assert_eq!(sax(&[3.0; 20], 4, 5), "cccc");
        assert_eq!(sax(&[3.0; 20], 4, 4), "bbbb");
        assert_eq!(sax(&[], 4, 5), "");
    }
}
