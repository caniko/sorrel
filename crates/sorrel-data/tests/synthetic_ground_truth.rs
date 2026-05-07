//! Ground-truth integration test for the suggestion engine.
//!
//! We synthesise a known scenario and check the suggesters' output:
//!
//! - Two clusters that are *secretly the same neuron* — split into A/B by
//!   alternating spike emission. The merge suggester must rank (A, B) at
//!   the top: their cross-correlogram has a perfect refractory dip.
//! - One cluster with two clearly different amplitude populations (jitter
//!   on one neuron then a contaminating noise spike). The split suggester
//!   must flag it; the GMM should pick a clean bipartition.
//! - One genuinely separate, well-isolated cluster — must NOT show up in
//!   either ranked list (or only at the very bottom).
//!
//! These are the failure modes that keep curators awake at night.

use sorrel_data::{
    journal::SqliteJournal,
    rank_merge_candidates, rank_split_candidates, Session, SuggestConfig,
};
use sorrel_io::{
    ClusterId, DataProvider, HasAmplitudes, SampleIndex, TraceSamples, TraceSlice,
};

struct GroundTruth {
    /// Per-cluster spike times, time-sorted.
    spikes: Vec<Vec<SampleIndex>>,
    /// Per-cluster amplitudes, in the same order.
    amps: Vec<Vec<f32>>,
    n_samples: SampleIndex,
}

impl DataProvider for GroundTruth {
    type Label = u8;
    fn sample_rate(&self) -> f32 { 30_000.0 }
    fn n_channels(&self) -> u32 { 1 }
    fn n_samples(&self) -> SampleIndex { self.n_samples }
    fn n_clusters(&self) -> u32 { self.spikes.len() as u32 }
    fn spike_times(&self, c: ClusterId) -> &[SampleIndex] {
        self.spikes.get(c.idx()).map(Vec::as_slice).unwrap_or(&[])
    }
    fn trace(&self, _: SampleIndex, _: u32) -> TraceSlice<'_> {
        TraceSlice { start: SampleIndex(0), n_channels: 1, samples: TraceSamples::I16(&[]) }
    }
    fn initial_labels(&self) -> Vec<u8> { vec![0; self.spikes.len()] }
}

impl HasAmplitudes for GroundTruth {
    fn spike_amplitudes(&self, c: ClusterId) -> &[f32] {
        self.amps.get(c.idx()).map(Vec::as_slice).unwrap_or(&[])
    }
}

/// Box-Muller Gaussian from xorshift32 seed.
fn gaussian(seed: &mut u32) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    let u1 = (*seed as f64 / u32::MAX as f64).clamp(1e-9, 1.0);
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    let u2 = (*seed as f64 / u32::MAX as f64).clamp(1e-9, 1.0);
    let g = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    g as f32
}

fn build_ground_truth() -> GroundTruth {
    // 30 kHz, 30 s of data → 900 000 samples. Long enough that the
    // suggester thresholds are sane.
    let sample_rate = 30_000_u64;
    let total_s = 30_u64;
    let n_samples = sample_rate * total_s;

    let mut seed = 0xCAFEBABE_u32;

    // ---- Cluster 0+1: secretly the same neuron, Poisson-ish ~30 Hz.
    //      Generate one neuron's spike train respecting a 2 ms absolute
    //      refractory period, then alternate-assign each spike to either
    //      cluster A or B. Result: A and B individually have no
    //      refractory violations; their cross-correlogram has a deep
    //      refractory dip near zero lag versus a Poisson-like baseline.
    let mut times_a = Vec::new();
    let mut times_b = Vec::new();
    let mut amps_a = Vec::new();
    let mut amps_b = Vec::new();
    {
        let lambda_per_sample = 30.0_f64 / 30_000.0; // 30 Hz at 30 kHz
        let refractory = 60_u64; // 2 ms
        let mut t = 0_u64;
        let mut last = 0_u64;
        let mut emitted = 0_usize;
        while t < n_samples {
            // Sample inter-arrival from exponential (-ln(U) / lambda).
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let u = (seed as f64 / u32::MAX as f64).clamp(1e-9, 1.0);
            let dt = (-u.ln() / lambda_per_sample) as u64;
            t = t.saturating_add(dt.max(1));
            if t >= n_samples {
                break;
            }
            if t < last + refractory {
                continue;
            }
            last = t;
            let amp = 5.0 + 0.3 * gaussian(&mut seed);
            if emitted % 2 == 0 {
                times_a.push(SampleIndex(t));
                amps_a.push(amp);
            } else {
                times_b.push(SampleIndex(t));
                amps_b.push(amp);
            }
            emitted += 1;
        }
    }

    // ---- Cluster 2: bimodal-amplitude over-merge candidate.
    //      Two underlying neurons firing independently at ~100 Hz each, with
    //      distinct amplitude means. Together this looks like one cluster
    //      with a bimodal amp distribution.
    let mut times_c = Vec::new();
    let mut amps_c = Vec::new();
    let mut t = 50_u64;
    let mut which = 0;
    let isi_c = 300_u64; // 10 ms at 30 kHz
    while t < n_samples {
        times_c.push(SampleIndex(t));
        let amp = if which % 2 == 0 {
            3.0 + 0.4 * gaussian(&mut seed)
        } else {
            8.0 + 0.4 * gaussian(&mut seed)
        };
        amps_c.push(amp);
        t += isi_c;
        which += 1;
    }

    // ---- Cluster 3: well-behaved isolated neuron, ~30 Hz.
    let isi_d = 1000_u64; // ~33 Hz at 30 kHz
    let mut times_d = Vec::new();
    let mut amps_d = Vec::new();
    t = 200;
    while t < n_samples {
        times_d.push(SampleIndex(t));
        amps_d.push(6.0 + 0.3 * gaussian(&mut seed));
        t += isi_d;
    }

    GroundTruth {
        spikes: vec![times_a, times_b, times_c, times_d],
        amps: vec![amps_a, amps_b, amps_c, amps_d],
        n_samples: SampleIndex(n_samples),
    }
}

fn fresh_session(prov: GroundTruth) -> (Session<GroundTruth>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let journal = SqliteJournal::open(&dir.path().join("j.sqlite")).unwrap();
    let mut s = Session::new(prov, journal);
    s.seed_amplitudes();
    (s, dir)
}

#[test]
fn merge_suggester_picks_secretly_same_neuron_pair() {
    let (sess, _d) = fresh_session(build_ground_truth());
    let cfg = SuggestConfig::default();
    let merges = rank_merge_candidates(&sess, &cfg);
    assert!(!merges.is_empty(), "expected at least one merge candidate");
    let top = &merges[0];
    let pair = (top.a.min(top.b), top.a.max(top.b));
    assert_eq!(pair, (ClusterId(0), ClusterId(1)), "expected (0,1) to be the top pair, got {pair:?}");
    assert!(top.score > 0.5, "top merge score {} is too low", top.score);
    // The CCG dip z-score should be unambiguously positive.
    assert!(top.ccg_dip_z > 2.0, "weak dip z = {}", top.ccg_dip_z);
}

#[test]
fn merge_suggester_does_not_propose_isolated_cluster() {
    let (sess, _d) = fresh_session(build_ground_truth());
    let cfg = SuggestConfig::default();
    let merges = rank_merge_candidates(&sess, &cfg);
    // Cluster 3 is well-isolated; if it appears it must be at the bottom
    // and below 0.5.
    for m in &merges {
        if m.a == ClusterId(3) || m.b == ClusterId(3) {
            assert!(
                m.score < 0.5,
                "cluster 3 (isolated) shouldn't rank with score {}",
                m.score
            );
        }
    }
}

#[test]
fn split_suggester_flags_bimodal_overmerge() {
    let (sess, _d) = fresh_session(build_ground_truth());
    let cfg = SuggestConfig::default();
    let splits = rank_split_candidates(&sess, &cfg);
    assert!(!splits.is_empty(), "expected the bimodal cluster to be flagged");
    // Cluster 2 must rank first.
    assert_eq!(
        splits[0].cluster, ClusterId(2),
        "expected c2 to top, got c{}",
        splits[0].cluster
    );
    assert!(
        splits[0].amp_bimodality > 0.55,
        "BC = {} below 0.555 threshold",
        splits[0].amp_bimodality
    );
    // GMM should detect the structure and offer a bipartition.
    assert!(
        splits[0].gmm_bic_delta > 6.0,
        "BIC delta = {} below strong-evidence threshold",
        splits[0].gmm_bic_delta
    );
    assert!(
        !splits[0].bipartition_minor_idx.is_empty(),
        "GMM should have produced a bipartition"
    );
}

#[test]
fn split_suggester_does_not_flag_isolated_cluster() {
    let (sess, _d) = fresh_session(build_ground_truth());
    let cfg = SuggestConfig::default();
    let splits = rank_split_candidates(&sess, &cfg);
    for s in &splits {
        assert_ne!(
            s.cluster, ClusterId(3),
            "cluster 3 (isolated) should not be in split candidates"
        );
    }
}
