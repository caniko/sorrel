//! Matched-filter spike rescue: detector math.
//!
//! Model: a chunk of (high-passed, CMR'd) data around a missed spike of unit
//! `u` is `x = a·M + noise`, with `M` the unit's template (`C × nt`, channel
//! major) and noise spatially correlated with covariance `Σ`. The
//! generalised matched filter is `F = Σ⁻¹M` (spatial whitening folded in),
//! its output `g(t) = Σ_{c,k} F[c][k]·x[c][t+k]` has mean `a·h` and variance
//! `h` with `h = Σ_{c,k} F[c][k]·M[c][k]`, so `a = g/h` and the energy
//! explained by the spike is `ΔE = g²/h`.
//!
//! `t` of a score / [`Peak`] is the *window start*: the sample aligned with
//! template column 0. The caller adds the template's trough offset.
//!
//! ponytail: noise is treated as white in time (only spatial covariance is
//! modelled). Temporal noise correlation after the 300 Hz high-pass is
//! small; whiten in time too if the null FPs look off.

use crate::linalg::{covariance, invert};

/// One detection candidate for a unit. `t` is a window start (chunk-local
/// out of [`pick_peaks`]; callers shift it to absolute samples).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Peak {
    pub t: u64,
    /// Explained energy `g²/h`.
    pub delta_e: f32,
    /// Fitted template amplitude `g/h` (1.0 = the template itself).
    pub amp: f32,
}

/// Noise-precision for [`matched_filter_weights`]: the inverse of the
/// ridge-regularised covariance of `n` spike-free samples (`n × c`
/// row-major). The ridge is `ridge_frac` × the mean variance, so it is
/// scale-free (0.01 is a sane start). `None` if `n < 2` or singular.
pub fn estimate_noise_cov_inv(
    samples: &[f32],
    n: usize,
    c: usize,
    ridge_frac: f32,
) -> Option<Vec<f32>> {
    if n < 2 || c == 0 || samples.len() != n * c {
        return None;
    }
    let data: Vec<f64> = samples.iter().map(|&v| f64::from(v)).collect();
    let mut cov = covariance(&data, n, c);
    let mean_var = (0..c).map(|i| cov.get(i, i)).sum::<f64>() / c as f64;
    cov.ridge(f64::from(ridge_frac) * mean_var);
    invert(&mut cov, 1e-12 * mean_var.max(f64::MIN_POSITIVE))?;
    Some(cov.data.iter().map(|&v| v as f32).collect())
}

/// Matched filter for a template (`c × nt`, channel major) and the noise
/// precision matrix `Σ⁻¹` (`c × c`). Returns `(filter, h)` with `filter`
/// laid out like the template (`F = Σ⁻¹M`, the layout the GPU kernel reads)
/// and `h = Σ F·M`.
pub fn matched_filter_weights(template: &[f32], noise_cov_inv: &[f32]) -> (Vec<f32>, f32) {
    let c = (noise_cov_inv.len() as f64).sqrt().round() as usize;
    assert!(c > 0 && c * c == noise_cov_inv.len(), "Σ⁻¹ must be square");
    assert!(
        template.len().is_multiple_of(c),
        "template length not a multiple of C"
    );
    let nt = template.len() / c;
    let mut filter = vec![0.0_f32; template.len()];
    let mut h = 0.0_f64;
    for k in 0..nt {
        for i in 0..c {
            let f: f64 = (0..c)
                .map(|j| f64::from(noise_cov_inv[i * c + j]) * f64::from(template[j * nt + k]))
                .sum();
            filter[i * nt + k] = f as f32;
            h += f * f64::from(template[i * nt + k]);
        }
    }
    (filter, h as f32)
}

/// Single-unit CPU reference of the GPU kernel (`GpuMatchedFilter`):
/// `out[u][t] = Σ_c Σ_k filters[u][c][k] · data[(t+k+shifts[ch]) mod n_time][ch]`
/// with `ch = chans[u][c]`. `data` is `n_time × n_chan` row-major, `filters`
/// `U × c_per_unit × nt`, `chans` `U × c_per_unit`, `shifts` one per channel.
pub fn matched_filter_reference(
    data: &[f32],
    n_chan: usize,
    filters: &[f32],
    chans: &[u32],
    nt: usize,
    c_per_unit: usize,
    shifts: &[i32],
) -> Vec<f32> {
    let n_time = data.len() / n_chan;
    let n_units = chans.len() / c_per_unit;
    let mut out = vec![0.0_f32; n_units * n_time];
    for u in 0..n_units {
        for t in 0..n_time {
            let mut acc = 0.0_f64;
            for c in 0..c_per_unit {
                let ch = chans[u * c_per_unit + c] as usize;
                let sh = i64::from(shifts[ch]);
                let f = &filters[(u * c_per_unit + c) * nt..][..nt];
                for (k, &w) in f.iter().enumerate() {
                    let idx = (t as i64 + k as i64 + sh).rem_euclid(n_time as i64) as usize;
                    acc += f64::from(w) * f64::from(data[idx * n_chan + ch]);
                }
            }
            out[u * n_time + t] = acc as f32;
        }
    }
    out
}

/// Candidate spikes in a matched-filter `score` trace (`g`): positive local
/// maxima within ±`nt/2` samples (ties go to the earliest), with fitted
/// amplitude `g/h` inside `amp_range` (inclusive). `h <= 0` yields nothing.
pub fn pick_peaks(score: &[f32], h: f32, nt: usize, amp_range: (f32, f32)) -> Vec<Peak> {
    if h <= 0.0 || h.is_nan() {
        return Vec::new();
    }
    let half = nt / 2;
    let mut peaks = Vec::new();
    for (t, &g) in score.iter().enumerate() {
        let amp = g / h;
        // Cheap amplitude gate first so the window scan only runs on the few
        // samples that could matter (`!(..)` also rejects NaN).
        if !(g > 0.0 && amp >= amp_range.0 && amp <= amp_range.1) {
            continue;
        }
        let lo = t.saturating_sub(half);
        let hi = (t + half + 1).min(score.len());
        let is_max =
            score[lo..t].iter().all(|&s| s < g) && score[t + 1..hi].iter().all(|&s| s <= g);
        if is_max {
            peaks.push(Peak {
                t: t as u64,
                delta_e: g * g / h,
                amp,
            });
        }
    }
    peaks
}

/// Second-best margin: keep a peak of unit `u` only if its `ΔE` beats the
/// best competitor `ΔE` within ±`nt/2` samples by at least `margin·ΔE`
/// (0.2 in the plan). `competitor_delta_e_at(t)` is the largest `ΔE` over
/// the *other* units at sample `t` (0 when none is positive); peaks keep
/// their order.
pub fn apply_margin(
    peaks: &[Peak],
    nt: usize,
    competitor_delta_e_at: impl Fn(u64) -> f32,
    margin: f32,
) -> Vec<Peak> {
    let half = (nt / 2) as u64;
    peaks
        .iter()
        .filter(|p| {
            let best = (p.t.saturating_sub(half)..=p.t + half)
                .map(&competitor_delta_e_at)
                .fold(0.0_f32, f32::max);
            p.delta_e - best >= margin * p.delta_e
        })
        .copied()
        .collect()
}

/// Drop peaks within `refractory_samples` of an existing spike
/// (`existing_sorted`, ascending), then resolve the survivors among
/// themselves greedily by descending `ΔE`. Result is sorted by `t`.
pub fn apply_refractory(
    peaks: &[Peak],
    existing_sorted: &[u64],
    refractory_samples: u64,
) -> Vec<Peak> {
    let near = |sorted: &[u64], t: u64| {
        let i = sorted.partition_point(|&e| e < t);
        sorted[i..]
            .first()
            .is_some_and(|&e| e - t < refractory_samples)
            || sorted[..i]
                .last()
                .is_some_and(|&e| t - e < refractory_samples)
    };
    let mut cand: Vec<Peak> = peaks
        .iter()
        .filter(|p| !near(existing_sorted, p.t))
        .copied()
        .collect();
    cand.sort_by(|a, b| b.delta_e.total_cmp(&a.delta_e));
    let mut kept_t: Vec<u64> = Vec::new();
    let mut kept = Vec::new();
    for p in cand {
        if !near(&kept_t, p.t) {
            kept_t.insert(kept_t.partition_point(|&e| e < p.t), p.t);
            kept.push(p);
        }
    }
    kept.sort_by_key(|p| p.t);
    kept
}

/// Result of [`fdr_threshold`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fdr {
    /// Smallest score threshold meeting the FDR target.
    pub theta: f32,
    /// Real detections with score ≥ `theta`.
    pub n_accepted: usize,
    /// Expected false positives among them: `count(null ≥ theta) / R`.
    pub false_positives: f32,
}

/// Empirical-null FDR. `real` are peak scores (e.g. `ΔE`) from the real
/// pass, `null` the pooled peak scores of `replicates` shifted-null passes.
/// Returns the smallest `theta` (a real score) with
/// `(count(null ≥ θ)/R) / count(real ≥ θ) ≤ q`, or `None` if no threshold
/// qualifies. NaNs are ignored.
///
/// ponytail: the FDR curve is not monotone in θ; "smallest qualifying" can
/// sit on a lucky dip with few null counts. Add a running-min (q-value)
/// pass if rescue yields look noisy.
pub fn fdr_threshold(real: &[f32], null: &[f32], replicates: usize, q: f32) -> Option<Fdr> {
    if replicates == 0 {
        return None;
    }
    let mut r: Vec<f32> = real.iter().copied().filter(|v| !v.is_nan()).collect();
    let mut n: Vec<f32> = null.iter().copied().filter(|v| !v.is_nan()).collect();
    r.sort_by(f32::total_cmp);
    n.sort_by(f32::total_cmp);
    // Candidate thresholds ascending: more accepted spikes first.
    r.iter().enumerate().find_map(|(i, &theta)| {
        if i > 0 && r[i - 1] == theta {
            return None;
        }
        let n_real = r.len() - i;
        let fp = (n.len() - n.partition_point(|&s| s < theta)) as f32 / replicates as f32;
        (fp / n_real as f32 <= q).then_some(Fdr {
            theta,
            n_accepted: n_real,
            false_positives: fp,
        })
    })
}

/// Linear map `y ≈ x·w + intercept` fitted by [`ridge_fit`]. `w` is `p × m`
/// row-major.
#[derive(Clone, Debug)]
pub struct RidgeModel {
    pub p: usize,
    pub m: usize,
    pub w: Vec<f32>,
    pub intercept: Vec<f32>,
}

/// Ridge regression of `y` (`n × m`) on `x` (`n × p`), both row-major, with
/// an unpenalised intercept. `lambda` is relative: the penalty added to the
/// diagonal of `XᵀX` is `lambda × mean(diag)`, so it needs no unit
/// juggling. `None` for empty input or a singular system (`lambda = 0`).
pub fn ridge_fit(
    x: &[f32],
    y: &[f32],
    n: usize,
    p: usize,
    m: usize,
    lambda: f32,
) -> Option<RidgeModel> {
    if n == 0 || p == 0 || m == 0 || x.len() != n * p || y.len() != n * m {
        return None;
    }
    let mean = |a: &[f32], d: usize| -> Vec<f64> {
        let mut mu = vec![0.0_f64; d];
        for row in a.chunks_exact(d) {
            for (s, &v) in mu.iter_mut().zip(row) {
                *s += f64::from(v);
            }
        }
        mu.iter().map(|s| s / n as f64).collect()
    };
    let (mx, my) = (mean(x, p), mean(y, m));
    let mut a = crate::linalg::Sym::zeros(p);
    let mut b = vec![0.0_f64; p * m];
    for (xr, yr) in x.chunks_exact(p).zip(y.chunks_exact(m)) {
        for i in 0..p {
            let xi = f64::from(xr[i]) - mx[i];
            for j in i..p {
                a.data[i * p + j] += xi * (f64::from(xr[j]) - mx[j]);
            }
            for k in 0..m {
                b[i * m + k] += xi * (f64::from(yr[k]) - my[k]);
            }
        }
    }
    for i in 0..p {
        for j in 0..i {
            a.data[i * p + j] = a.data[j * p + i];
        }
    }
    let mean_diag = (0..p).map(|i| a.get(i, i)).sum::<f64>() / p as f64;
    if mean_diag <= 0.0 {
        // Constant features: predict the mean.
        return Some(RidgeModel {
            p,
            m,
            w: vec![0.0; p * m],
            intercept: my.iter().map(|&v| v as f32).collect(),
        });
    }
    a.ridge(f64::from(lambda) * mean_diag);
    invert(&mut a, 1e-12 * mean_diag)?;
    let mut w = vec![0.0_f32; p * m];
    let mut intercept: Vec<f64> = my.clone();
    for k in 0..m {
        for i in 0..p {
            let wik: f64 = (0..p).map(|j| a.get(i, j) * b[j * m + k]).sum();
            w[i * m + k] = wik as f32;
            intercept[k] -= mx[i] * wik;
        }
    }
    Some(RidgeModel {
        p,
        m,
        w,
        intercept: intercept.iter().map(|&v| v as f32).collect(),
    })
}

/// Predict `n × m` outputs for `x` (`n × p`).
pub fn ridge_predict(model: &RidgeModel, x: &[f32]) -> Vec<f32> {
    let (p, m) = (model.p, model.m);
    let mut out = Vec::with_capacity(x.len() / p * m);
    for xr in x.chunks_exact(p) {
        for k in 0..m {
            let s: f64 = (0..p)
                .map(|i| f64::from(xr[i]) * f64::from(model.w[i * m + k]))
                .sum();
            out.push((s + f64::from(model.intercept[k])) as f32);
        }
    }
    out
}

/// In-sample R² per output column of `model` on `(x, y)`; 0 for a constant
/// column. ponytail: in-sample, optimistic when `n` is close to `p`; split
/// held-out rows if that matters.
pub fn ridge_r2(model: &RidgeModel, x: &[f32], y: &[f32]) -> Vec<f32> {
    let m = model.m;
    let pred = ridge_predict(model, x);
    let n = y.len() / m;
    (0..m)
        .map(|k| {
            let mu = (0..n).map(|r| f64::from(y[r * m + k])).sum::<f64>() / n as f64;
            let (mut res, mut tot) = (0.0_f64, 0.0_f64);
            for r in 0..n {
                let yv = f64::from(y[r * m + k]);
                res += (yv - f64::from(pred[r * m + k])).powi(2);
                tot += (yv - mu).powi(2);
            }
            if tot > 0.0 {
                (1.0 - res / tot) as f32
            } else {
                0.0
            }
        })
        .collect()
}

/// Fraction of inter-spike intervals shorter than `refractory` samples
/// (`times_sorted` ascending). 0 for fewer than two spikes. The count form
/// is [`crate::isi_violations`] (which wants `SampleIndex`); this is the
/// rate used for "did the additions more than double the violations".
pub fn isi_violation_fraction(times_sorted: &[u64], refractory: u64) -> f32 {
    if times_sorted.len() < 2 {
        return 0.0;
    }
    let v = times_sorted
        .windows(2)
        .filter(|w| w[1] - w[0] < refractory)
        .count();
    v as f32 / (times_sorted.len() - 1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift in [-1, 1).
    fn rng(seed: &mut u64) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }

    #[test]
    fn weights_whiten_and_h_matches_definition() {
        // Σ⁻¹ = diag(4, 1): channel 0 is quieter, so it gets 4x the weight.
        let template = [1.0, 2.0, 3.0, /* ch1 */ 1.0, 1.0, 1.0];
        let (f, h) = matched_filter_weights(&template, &[4.0, 0.0, 0.0, 1.0]);
        assert_eq!(f, vec![4.0, 8.0, 12.0, 1.0, 1.0, 1.0]);
        assert_eq!(h, 4.0 + 16.0 + 36.0 + 3.0);
    }

    #[test]
    fn noise_cov_inv_recovers_precision() {
        let mut s = 7;
        // ch1 = 2*ch0 noise + small own noise -> strongly correlated pair.
        let n = 4000;
        let mut x = Vec::new();
        for _ in 0..n {
            let a = rng(&mut s);
            x.extend([a, 2.0 * a + 0.1 * rng(&mut s)]);
        }
        let p = estimate_noise_cov_inv(&x, n, 2, 1e-4).unwrap();
        assert_eq!(p.len(), 4);
        // Whitened residual direction (2,-1) is cheap, (1,2) is expensive.
        let q =
            |v: [f32; 2]| v[0] * (p[0] * v[0] + p[1] * v[1]) + v[1] * (p[2] * v[0] + p[3] * v[1]);
        assert!(q([2.0, -1.0]) > 20.0 * q([1.0, 2.0]));
        assert!(estimate_noise_cov_inv(&x, 1, 2, 0.0).is_none());
    }

    #[test]
    fn filter_recovers_planted_spikes() {
        let (c, nt, n_time) = (3usize, 11usize, 400usize);
        let mut s = 99;
        let template: Vec<f32> = (0..c * nt)
            .map(|i| -((i % nt) as f32 - 5.0).powi(2) / 5.0 + 5.0)
            .collect();
        let mut x: Vec<f32> = (0..n_time * c).map(|_| 0.3 * rng(&mut s)).collect();
        let planted = [(50usize, 1.0f32), (200, 0.6), (333, 1.5)];
        for &(t0, a) in &planted {
            for k in 0..nt {
                for ch in 0..c {
                    x[(t0 + k) * c + ch] += a * template[ch * nt + k];
                }
            }
        }
        let eye = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let (f, h) = matched_filter_weights(&template, &eye);
        let chans: Vec<u32> = (0..c as u32).collect();
        let score = matched_filter_reference(&x, c, &f, &chans, nt, c, &[0; 3]);
        let peaks = pick_peaks(&score, h, nt, (0.3, 3.0));
        assert_eq!(peaks.len(), planted.len(), "{peaks:?}");
        for (p, &(t0, a)) in peaks.iter().zip(&planted) {
            assert_eq!(p.t, t0 as u64);
            assert!((p.amp - a).abs() < 0.05, "amp {} vs {a}", p.amp);
            assert!((p.delta_e - p.amp * p.amp * h).abs() < 1e-3 * p.delta_e);
        }
        // Amplitude gate: only the 0.6 spike survives (0.5, 0.8).
        let only = pick_peaks(&score, h, nt, (0.5, 0.8));
        assert_eq!(only.len(), 1);
        assert_eq!(only[0].t, 200);
    }

    #[test]
    fn reference_wraps_and_shifts() {
        // 1 channel, 4 samples, filter [1,0] -> out[t] = data[t + shift].
        let data = [10.0, 20.0, 30.0, 40.0];
        let out = matched_filter_reference(&data, 1, &[1.0, 0.0], &[0], 2, 1, &[3]);
        assert_eq!(out, vec![40.0, 10.0, 20.0, 30.0]);
        let out = matched_filter_reference(&data, 1, &[0.0, 1.0], &[0], 2, 1, &[0]);
        assert_eq!(out, vec![20.0, 30.0, 40.0, 10.0]);
    }

    #[test]
    fn peaks_are_local_maxima_with_tie_to_earliest() {
        let mut score = vec![0.0; 40];
        score[10] = 5.0;
        score[12] = 8.0; // within nt/2=3 of 10 -> 10 is dominated
        score[30] = 4.0;
        score[31] = 4.0; // tie -> earliest wins
        let p = pick_peaks(&score, 1.0, 6, (0.0, 100.0));
        assert_eq!(p.iter().map(|p| p.t).collect::<Vec<_>>(), vec![12, 30]);
        assert!(pick_peaks(&score, 0.0, 6, (0.0, 100.0)).is_empty());
    }

    #[test]
    fn margin_rejects_peak_dominated_by_competitor() {
        let peaks = [
            Peak {
                t: 100,
                delta_e: 10.0,
                amp: 1.0,
            },
            Peak {
                t: 300,
                delta_e: 10.0,
                amp: 1.0,
            },
            Peak {
                t: 500,
                delta_e: 10.0,
                amp: 1.0,
            },
        ];
        // Competitor 9.0 near t=102 (inside ±nt/2=5): 10-9 < 0.2*10 -> drop.
        // Competitor 7.0 at t=300: 10-7 = 3 >= 2 -> keep. Competitor 9.0 at
        // t=520 is outside the window -> keep.
        let comp = |t: u64| match t {
            102 => 9.0,
            300 => 7.0,
            520 => 9.0,
            _ => 0.0,
        };
        let kept = apply_margin(&peaks, 10, comp, 0.2);
        assert_eq!(kept.iter().map(|p| p.t).collect::<Vec<_>>(), vec![300, 500]);
    }

    #[test]
    fn refractory_drops_near_duplicates() {
        let pk = |t, e| Peak {
            t,
            delta_e: e,
            amp: 1.0,
        };
        let peaks = [pk(100, 5.0), pk(110, 9.0), pk(200, 3.0), pk(1000, 4.0)];
        // Existing spike at 995 kills 1000; 100/110 collapse to the stronger
        // (110); 200 is near existing 190? no existing at 190 -> dropped.
        let kept = apply_refractory(&peaks, &[190, 995], 30);
        assert_eq!(kept.iter().map(|p| p.t).collect::<Vec<_>>(), vec![110]);
        // Boundary: exactly `refractory` apart is allowed.
        let kept = apply_refractory(&[pk(130, 1.0)], &[100], 30);
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn fdr_threshold_matches_hand_count() {
        // real: 10 scores 1..=10. null (R=2): 3 pooled hits >= 1, one >= 5.
        let real: Vec<f32> = (1..=10).map(|v| v as f32).collect();
        let null = [1.5, 2.5, 6.0];
        // theta=1: fp=3/2=1.5, 10 real -> 0.15; theta=3: null>=3 is 1 -> 0.5/8
        // = 0.0625; theta=4: 0.5/7=0.071 ... all >0.05 until theta=7 where
        // null>=7 is 0 -> 0.
        let f = fdr_threshold(&real, &null, 2, 0.05).unwrap();
        assert_eq!(f.theta, 7.0);
        assert_eq!(f.n_accepted, 4);
        assert_eq!(f.false_positives, 0.0);
        let f = fdr_threshold(&real, &null, 2, 0.1).unwrap();
        assert_eq!((f.theta, f.n_accepted), (3.0, 8));
        assert_eq!(f.false_positives, 0.5);
        // Null as big as real: nothing qualifies.
        assert!(fdr_threshold(&real, &real, 1, 0.05).is_none());
        assert!(fdr_threshold(&real, &null, 0, 0.05).is_none());
    }

    #[test]
    fn ridge_recovers_known_linear_map() {
        let (n, p, m) = (200usize, 3usize, 2usize);
        let mut s = 5;
        let x: Vec<f32> = (0..n * p).map(|_| rng(&mut s)).collect();
        let w = [1.0, -2.0, 0.5, 3.0, -1.0, 0.25]; // p x m
        let b = [0.7, -1.3];
        let mut y = Vec::new();
        for r in 0..n {
            for k in 0..m {
                y.push((0..p).map(|i| x[r * p + i] * w[i * m + k]).sum::<f32>() + b[k]);
            }
        }
        let model = ridge_fit(&x, &y, n, p, m, 1e-6).unwrap();
        for (a, e) in model.w.iter().zip(&w) {
            assert!((a - e).abs() < 1e-3, "{a} vs {e}");
        }
        for (a, e) in model.intercept.iter().zip(&b) {
            assert!((a - e).abs() < 1e-3);
        }
        let r2 = ridge_r2(&model, &x, &y);
        assert!(r2.iter().all(|&v| v > 0.9999), "{r2:?}");
        // Pure-noise target: R² well below 1.
        let noise: Vec<f32> = (0..n * m).map(|_| rng(&mut s)).collect();
        let nm = ridge_fit(&x, &noise, n, p, m, 1e-6).unwrap();
        assert!(ridge_r2(&nm, &x, &noise).iter().all(|&v| v < 0.2));
        assert!(ridge_fit(&[], &[], 0, 3, 2, 0.1).is_none());
    }

    #[test]
    fn isi_fraction() {
        assert_eq!(isi_violation_fraction(&[], 30), 0.0);
        assert_eq!(isi_violation_fraction(&[0, 10, 100, 129, 300], 30), 0.5);
    }
}
