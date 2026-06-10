//! CCG-shape analysis used by the merge-candidate suggester.
//!
//! Two spike trains from the *same* neuron should never co-fire inside the
//! refractory period — so a cross-correlogram with a clear dip around zero
//! lag is strong evidence the two clusters should be merged. These helpers
//! quantify that dip without depending on any external stats library.

/// Result of analysing a CCG centred on zero lag with bins symmetric around
/// the centre. All counts are bin counts, not rates.
#[derive(Clone, Copy, Debug)]
pub struct CcgRefractoryAnalysis {
    /// Total spike-pair count inside the central refractory window.
    pub center_count: u32,
    /// Mean bin count in the "shoulder" region (bins outside the refractory
    /// window but inside the analysis window — the neuron's baseline rate).
    pub shoulder_mean: f32,
    /// Standard deviation of shoulder bin counts, used to z-score the centre.
    pub shoulder_std: f32,
    /// `(shoulder_mean - center_mean) / shoulder_std` — large positive values
    /// indicate the centre is suppressed below baseline (refractory dip).
    /// Negative values mean the centre is *above* baseline (synchronous
    /// firing — likely electrical artifact rather than the same unit).
    pub z: f32,
    /// Number of bins counted as the centre (refractory window total).
    pub center_bins: usize,
    /// Number of bins counted as the shoulder.
    pub shoulder_bins: usize,
}

/// Analyse a centred CCG histogram for a refractory-period dip.
///
/// `histogram` must be the output of `cross_correlogram(a, b, max_lag, bins)`
/// (so `bins` is even and the histogram is centred). `refractory_bins` is the
/// number of bins on *each side* of zero lag that count as the refractory
/// window — typical: round(refractory_period / bin_width). `shoulder_bins` is
/// the per-side width of the baseline window, also measured outward from the
/// centre.
pub fn analyse_refractory_dip(
    histogram: &[u32],
    refractory_bins: usize,
    shoulder_bins: usize,
) -> CcgRefractoryAnalysis {
    let bins = histogram.len();
    // Zero lag is the boundary between bins `half-1` and `half`, which only
    // centers the refractory window correctly when the count is even.
    debug_assert!(
        bins % 2 == 0 || bins == 0,
        "analyse_refractory_dip expects an even bin count (got {bins}); \
         zero lag would be mis-centered otherwise",
    );
    if bins == 0 {
        return CcgRefractoryAnalysis {
            center_count: 0,
            shoulder_mean: 0.0,
            shoulder_std: 0.0,
            z: 0.0,
            center_bins: 0,
            shoulder_bins: 0,
        };
    }
    let half = bins / 2;
    let r = refractory_bins.min(half);
    let s = shoulder_bins.min(half.saturating_sub(r));
    let center_lo = half.saturating_sub(r);
    let center_hi = (half + r).min(bins);
    let center: u32 = histogram[center_lo..center_hi].iter().copied().sum();
    let center_n = (center_hi - center_lo).max(1);

    // Shoulder = the band outside the refractory window but inside [half-r-s,
    // half+r+s], on both sides.
    let left_lo = center_lo.saturating_sub(s);
    let right_hi = (center_hi + s).min(bins);
    let mut shoulder = Vec::with_capacity(2 * s);
    if center_lo > left_lo {
        shoulder.extend_from_slice(&histogram[left_lo..center_lo]);
    }
    if right_hi > center_hi {
        shoulder.extend_from_slice(&histogram[center_hi..right_hi]);
    }
    let shoulder_n = shoulder.len();
    if shoulder_n == 0 {
        return CcgRefractoryAnalysis {
            center_count: center,
            shoulder_mean: 0.0,
            shoulder_std: 0.0,
            z: 0.0,
            center_bins: center_n,
            shoulder_bins: 0,
        };
    }
    let mean = shoulder.iter().map(|&v| v as f64).sum::<f64>() / shoulder_n as f64;
    let var = shoulder
        .iter()
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>()
        / shoulder_n as f64;
    let std = var.sqrt().max(1e-6);
    let center_per_bin = center as f64 / center_n as f64;
    // Positive z = centre is *below* shoulder = refractory dip.
    let z = ((mean - center_per_bin) / std) as f32;

    CcgRefractoryAnalysis {
        center_count: center,
        shoulder_mean: mean as f32,
        shoulder_std: std as f32,
        z,
        center_bins: center_n,
        shoulder_bins: shoulder_n,
    }
}

/// Convenience scoring: `[0, 1]` where 1.0 = perfect refractory dip
/// (suggests same unit), 0 = no dip or synchronous excess. Saturates the
/// z-score from `analyse_refractory_dip` through a soft sigmoid so big
/// outliers don't dominate the merge ranking.
pub fn refractory_dip_score(analysis: &CcgRefractoryAnalysis) -> f32 {
    if analysis.shoulder_bins == 0 {
        return 0.0;
    }
    let z = analysis.z.max(0.0);
    // Smooth, saturating curve. z=1 → 0.27, z=2 → 0.50, z=3 → 0.69, z=5 → 0.88.
    1.0 - (-z * 0.5).exp().clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(n: usize, v: u32) -> Vec<u32> {
        vec![v; n]
    }

    #[test]
    fn flat_ccg_has_zero_dip() {
        let h = flat(20, 10);
        let a = analyse_refractory_dip(&h, 2, 5);
        assert!(a.z.abs() < 1e-3);
        assert!(refractory_dip_score(&a) < 0.05);
    }

    #[test]
    fn deep_central_dip_yields_high_score() {
        let mut h = flat(40, 50);
        // Carve out a 4-bin refractory zone at the centre (bins 18..22).
        for bin in h.iter_mut().take(22).skip(18) {
            *bin = 0;
        }
        let a = analyse_refractory_dip(&h, 2, 6);
        assert!(a.z > 3.0, "expected large z, got {}", a.z);
        assert!(refractory_dip_score(&a) > 0.7);
    }

    #[test]
    fn central_excess_yields_zero_score() {
        let mut h = flat(40, 10);
        for bin in h.iter_mut().take(22).skip(18) {
            *bin = 200; // synchronous firing
        }
        let a = analyse_refractory_dip(&h, 2, 6);
        assert!(a.z < 0.0);
        assert_eq!(refractory_dip_score(&a), 0.0);
    }

    #[test]
    fn empty_histogram_safe() {
        let a = analyse_refractory_dip(&[], 2, 5);
        assert_eq!(a.center_count, 0);
        assert_eq!(a.center_bins, 0);
        assert_eq!(refractory_dip_score(&a), 0.0);
    }

    #[test]
    fn refractory_dip_score_in_unit_range() {
        for shape in [
            flat(20, 10),
            {
                let mut h = flat(20, 10);
                h[10] = 0;
                h
            },
            {
                let mut h = flat(20, 10);
                h[10] = 50;
                h
            },
        ] {
            let a = analyse_refractory_dip(&shape, 1, 5);
            let s = refractory_dip_score(&a);
            assert!((0.0..=1.0).contains(&s), "score {s} out of range");
        }
    }

    #[test]
    fn refractory_bins_clamped_to_half() {
        // Asking for refractory_bins > half should clamp gracefully.
        let h = flat(10, 5);
        let a = analyse_refractory_dip(&h, 100, 100);
        assert!(a.center_bins <= h.len());
        assert!(a.shoulder_bins <= h.len());
    }

    #[test]
    fn zero_shoulder_bins_returns_zero_z() {
        let h = flat(10, 5);
        let a = analyse_refractory_dip(&h, 5, 0);
        assert_eq!(a.z, 0.0);
        assert_eq!(refractory_dip_score(&a), 0.0);
    }

    #[test]
    fn analysis_center_count_matches_sum_of_central_bins() {
        let mut h = flat(20, 0);
        h[8] = 3;
        h[9] = 5;
        h[10] = 7;
        h[11] = 11;
        let a = analyse_refractory_dip(&h, 2, 4);
        // half = 10, refractory = 2 -> bins 8..12 = 3+5+7+11 = 26.
        assert_eq!(a.center_count, 26);
        assert_eq!(a.center_bins, 4);
    }

    #[test]
    fn shallow_dip_yields_lower_score_than_deep_dip() {
        // Use noisy shoulders so std isn't clamped to the floor — that's
        // what makes the z-score sensitive to the size of the dip.
        let make_shoulder = |base: u32| {
            let mut h = flat(40, base);
            // Add a small alternation in the shoulders (bins 4..18, 22..36).
            for (i, bin) in h.iter_mut().enumerate().take(40) {
                if (4..18).contains(&i) || (22..36).contains(&i) {
                    if i % 2 == 0 {
                        *bin = base + 3;
                    } else {
                        *bin = base.saturating_sub(3);
                    }
                }
            }
            h
        };
        let mut shallow = make_shoulder(50);
        for bin in shallow.iter_mut().take(22).skip(18) {
            *bin = 35; // mild reduction
        }
        let mut deep = make_shoulder(50);
        for bin in deep.iter_mut().take(22).skip(18) {
            *bin = 1;
        }
        let s_shallow = refractory_dip_score(&analyse_refractory_dip(&shallow, 2, 6));
        let s_deep = refractory_dip_score(&analyse_refractory_dip(&deep, 2, 6));
        assert!(
            s_deep >= s_shallow,
            "deep {s_deep} should be ≥ shallow {s_shallow}",
        );
    }
}
