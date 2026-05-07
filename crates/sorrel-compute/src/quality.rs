//! Composite, single-number cluster quality.
//!
//! The metrics elsewhere in this crate are deliberately one-aspect each
//! (contamination, drift, presence, amplitude SNR, ...). For the cluster
//! table sort and the suggestion engine we want one number per cluster the
//! curator can sort by. This module combines them — but exposes the
//! intermediate evidence so the UI can show a tooltip explaining *why* a
//! given cluster scored where it did.

use crate::distribution::amplitude_cutoff;
use crate::drift::{amplitude_drift_correlation, longest_silent_gap_frac, presence_cv};
use crate::metrics::{
    amplitude_snr, isi_violations, presence_ratio, refractory_contamination,
};
use crate::metrics_iso::IsolationMetrics;
use sorrel_io::SampleIndex;

/// Per-cluster quality breakdown. All fields are bounded to `[0, 1]`
/// where 1.0 = best and 0.0 = worst, except the raw inputs noted below.
#[derive(Clone, Copy, Debug, Default)]
pub struct QualityBreakdown {
    /// `1 - clamp(refractory_contamination, 0, 1)`. 1.0 means a clean
    /// refractory period, 0.0 means heavy contamination.
    pub contamination_score: f32,
    /// `presence_ratio` (already in `[0, 1]`).
    pub presence_score: f32,
    /// `1 - longest_silent_gap_fraction`. 1.0 means gapless coverage.
    pub coverage_score: f32,
    /// Saturating SNR score: `amp_snr / (amp_snr + 1)`. 1.0 is unreachable
    /// but values above 0.7 indicate a confident detection.
    pub snr_score: f32,
    /// `1 - amplitude_cutoff`. 1.0 means the amp distribution looks fully
    /// sampled; lower means likely missed spikes below threshold.
    pub completeness_score: f32,
    /// `1 - |drift_correlation|`. 1.0 means stationary amplitudes; lower
    /// means the cluster drifted across the recording.
    pub stability_score: f32,

    /// Raw evidence retained for tooltips:
    pub raw_contamination: f32,
    pub raw_presence_ratio: f32,
    pub raw_silent_gap: f32,
    pub raw_snr: f32,
    pub raw_amp_cutoff: f32,
    pub raw_drift_corr: f32,
    pub raw_isi_violations: u32,
    pub raw_presence_cv: f32,
    pub n_spikes: usize,

    /// Optional PC-feature isolation evidence. `None` when the backend
    /// has no PC features, the cluster has too few spikes, or the
    /// covariance is too ill-conditioned to invert. When present it adds
    /// a 7th term (`isolation_score`) to the composite.
    pub isolation: Option<IsolationMetrics>,
}

impl QualityBreakdown {
    /// Composite score in `[0, 1]`. Geometric mean of all available
    /// sub-scores so no single dimension can boost a cluster past a clear
    /// weakness in another — this matches what curators do when eyeballing
    /// the views (any one ugly view sinks the cluster).
    pub fn composite(&self) -> f32 {
        let mut parts: Vec<f32> = vec![
            self.contamination_score.max(1e-3),
            self.presence_score.max(1e-3),
            self.coverage_score.max(1e-3),
            self.snr_score.max(1e-3),
            self.completeness_score.max(1e-3),
            self.stability_score.max(1e-3),
        ];
        if let Some(iso) = self.isolation {
            let s = iso.score();
            if s.is_finite() {
                parts.push(s.max(1e-3));
            }
        }
        let log_mean: f32 =
            parts.iter().map(|p| p.ln()).sum::<f32>() / parts.len() as f32;
        log_mean.exp().clamp(0.0, 1.0)
    }

    /// True when this breakdown carries PC-feature isolation evidence.
    pub fn has_isolation(&self) -> bool {
        self.isolation.is_some_and(|i| i.has_evidence())
    }
}

/// Compute the full quality breakdown for one cluster from its raw spike
/// arrays. Cheap O(n_spikes); safe to call once per cluster per UI build.
pub fn quality_breakdown(
    spike_times: &[SampleIndex],
    spike_amplitudes: &[f32],
    refractory_samples: u64,
    total_duration_samples: u64,
    sample_rate: f32,
    presence_bins: usize,
) -> QualityBreakdown {
    let n = spike_times.len();
    let raw_contamination =
        refractory_contamination(spike_times, refractory_samples, total_duration_samples, sample_rate);
    let raw_presence = presence_ratio(spike_times, total_duration_samples, presence_bins);
    let raw_gap = longest_silent_gap_frac(spike_times, total_duration_samples);
    let raw_snr = amplitude_snr(spike_amplitudes);
    let raw_cutoff = amplitude_cutoff(spike_amplitudes, 30);
    let raw_drift = amplitude_drift_correlation(spike_times, spike_amplitudes);
    let raw_isi_violations = isi_violations(spike_times, refractory_samples) as u32;
    let raw_presence_cv = presence_cv(spike_times, total_duration_samples, presence_bins);

    QualityBreakdown {
        contamination_score: (1.0 - raw_contamination).clamp(0.0, 1.0),
        presence_score: raw_presence.clamp(0.0, 1.0),
        coverage_score: (1.0 - raw_gap).clamp(0.0, 1.0),
        snr_score: if raw_snr <= 0.0 {
            0.0
        } else {
            (raw_snr / (raw_snr + 1.0)).clamp(0.0, 1.0)
        },
        completeness_score: (1.0 - raw_cutoff * 2.0).clamp(0.0, 1.0),
        stability_score: (1.0 - raw_drift.abs()).clamp(0.0, 1.0),

        raw_contamination,
        raw_presence_ratio: raw_presence,
        raw_silent_gap: raw_gap,
        raw_snr,
        raw_amp_cutoff: raw_cutoff,
        raw_drift_corr: raw_drift,
        raw_isi_violations,
        raw_presence_cv,
        n_spikes: n,
        isolation: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composite_is_geometric_mean() {
        let q = QualityBreakdown {
            contamination_score: 1.0,
            presence_score: 1.0,
            coverage_score: 1.0,
            snr_score: 1.0,
            completeness_score: 1.0,
            stability_score: 1.0,
            ..QualityBreakdown::default()
        };
        assert!((q.composite() - 1.0).abs() < 1e-3);

        let q = QualityBreakdown {
            contamination_score: 0.0,
            presence_score: 1.0,
            coverage_score: 1.0,
            snr_score: 1.0,
            completeness_score: 1.0,
            stability_score: 1.0,
            ..QualityBreakdown::default()
        };
        // Geometric mean clamps at 1e-3 floor → composite is small, not 0.
        assert!(q.composite() < 0.5);
    }

    #[test]
    fn breakdown_clean_train_scores_high() {
        // Evenly-spaced spike train, constant amplitude, no contamination.
        let times: Vec<SampleIndex> = (0..100).map(|i| i as u64 * 1000).map(SampleIndex).collect();
        let amps: Vec<f32> = (0..100).map(|_| 3.0).collect();
        let q = quality_breakdown(&times, &amps, 100, 100_000, 1000.0, 50);
        assert!(q.contamination_score > 0.95);
        assert!(q.presence_score > 0.5);
        assert!(q.stability_score > 0.95);
    }

    #[test]
    fn breakdown_drifting_train_scores_lower_stability() {
        let times: Vec<SampleIndex> = (0..100).map(|i| i as u64 * 1000).map(SampleIndex).collect();
        // Amplitude grows linearly with time → drift_corr ≈ 1.
        let amps: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let q = quality_breakdown(&times, &amps, 100, 100_000, 1000.0, 50);
        assert!(q.stability_score < 0.1);
    }

    #[test]
    fn composite_score_in_unit_range() {
        let q = QualityBreakdown {
            contamination_score: 0.5,
            presence_score: 0.7,
            coverage_score: 0.6,
            snr_score: 0.8,
            completeness_score: 0.4,
            stability_score: 0.9,
            ..QualityBreakdown::default()
        };
        let c = q.composite();
        assert!(
            (0.0..=1.0).contains(&c),
            "composite {c} out of [0, 1]"
        );
    }

    #[test]
    fn empty_inputs_produce_zero_scores() {
        let q = quality_breakdown(&[], &[], 100, 1000, 1000.0, 10);
        assert_eq!(q.n_spikes, 0);
        assert_eq!(q.raw_isi_violations, 0);
        // SNR over no amplitudes is 0 → snr_score should be 0.
        assert_eq!(q.snr_score, 0.0);
    }

    #[test]
    fn breakdown_records_n_spikes() {
        let times: Vec<SampleIndex> = (0..42).map(|i| i as u64 * 100).map(SampleIndex).collect();
        let amps: Vec<f32> = (0..42).map(|i| i as f32).collect();
        let q = quality_breakdown(&times, &amps, 50, 4200, 1000.0, 10);
        assert_eq!(q.n_spikes, 42);
    }

    #[test]
    fn breakdown_drift_corr_negative_still_penalises_stability() {
        let times: Vec<SampleIndex> = (0..100).map(|i| i as u64 * 1000).map(SampleIndex).collect();
        // Amplitude falling with time → drift_corr ≈ -1.
        let amps: Vec<f32> = (0..100).map(|i| -(i as f32) * 0.1).collect();
        let q = quality_breakdown(&times, &amps, 100, 100_000, 1000.0, 50);
        assert!(q.stability_score < 0.1);
        assert!(q.raw_drift_corr < -0.9);
    }

    #[test]
    fn contamination_score_clamped_when_input_exceeds_one() {
        // Bursty input — heavy ISI violations.
        let times: Vec<SampleIndex> = (0..50).map(|i| i as u64).map(SampleIndex).collect();
        let amps = vec![1.0; 50];
        let q = quality_breakdown(&times, &amps, 100, 50, 1000.0, 5);
        assert!((0.0..=1.0).contains(&q.contamination_score));
    }

    #[test]
    fn has_isolation_returns_false_when_isolation_is_none() {
        let q = QualityBreakdown::default();
        assert!(!q.has_isolation());
    }

    #[test]
    fn composite_with_zero_in_one_dimension_reflects_geometric_punishment() {
        // Geometric mean clamps weak scores at 1e-3, so even one zero limits
        // the composite well below the arithmetic mean.
        let q = QualityBreakdown {
            contamination_score: 1.0,
            presence_score: 1.0,
            coverage_score: 1.0,
            snr_score: 1.0,
            completeness_score: 1.0,
            stability_score: 0.0,
            ..QualityBreakdown::default()
        };
        let arithmetic = (1.0 + 1.0 + 1.0 + 1.0 + 1.0 + 0.0) / 6.0; // = 0.833
        let geometric = q.composite();
        assert!(geometric < arithmetic);
    }
}
