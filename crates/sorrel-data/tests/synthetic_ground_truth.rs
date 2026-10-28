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
    journal::Journal, rank_merge_candidates, rank_split_candidates, Session, SuggestConfig,
};
use sorrel_io::{ClusterId, DataProvider, HasAmplitudes, SampleIndex, TraceSamples, TraceSlice};

struct GroundTruth {
    /// Per-cluster spike times, time-sorted.
    spikes: Vec<Vec<SampleIndex>>,
    /// Per-cluster amplitudes, in the same order.
    amps: Vec<Vec<f32>>,
    n_samples: SampleIndex,
}

impl DataProvider for GroundTruth {
    type Label = u8;
    fn sample_rate(&self) -> f32 {
        30_000.0
    }
    fn n_channels(&self) -> u32 {
        1
    }
    fn n_samples(&self) -> SampleIndex {
        self.n_samples
    }
    fn n_clusters(&self) -> u32 {
        self.spikes.len() as u32
    }
    fn spike_times(&self, c: ClusterId) -> &[SampleIndex] {
        self.spikes.get(c.idx()).map(Vec::as_slice).unwrap_or(&[])
    }
    fn trace(&self, _: SampleIndex, _: u32) -> TraceSlice<'_> {
        TraceSlice {
            start: SampleIndex(0),
            n_channels: 1,
            samples: TraceSamples::I16(&[]),
        }
    }
    fn initial_labels(&self) -> Vec<u8> {
        vec![0; self.spikes.len()]
    }
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
            if emitted & 1 == 0 {
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
        let amp = if which & 1 == 0 {
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
    let journal = Journal::open(&dir.path().join("j.sqlite")).unwrap();
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
    assert_eq!(
        pair,
        (ClusterId(0), ClusterId(1)),
        "expected (0,1) to be the top pair, got {pair:?}"
    );
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
    assert!(
        !splits.is_empty(),
        "expected the bimodal cluster to be flagged"
    );
    // Cluster 2 must rank first.
    assert_eq!(
        splits[0].cluster,
        ClusterId(2),
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
            s.cluster,
            ClusterId(3),
            "cluster 3 (isolated) should not be in split candidates"
        );
    }
}

// ---------------------------------------------------------------------
// Incremental suggestions, drift fragments and burst-aware splits.
// ---------------------------------------------------------------------

use sorrel_data::suggest::{rank_merge_candidates_memo, rank_split_candidates_memo, SuggestMemo};
use sorrel_data::CurationCommand;
use sorrel_io::{HasSpikeTemplates, HasTemplateWaveforms};

/// Exponential-ISI Poisson train with a 2 ms refractory floor on `[from, to)`.
fn poisson(seed: &mut u32, rate_hz: f64, from: u64, to: u64) -> Vec<SampleIndex> {
    let mut out = Vec::new();
    let mut t = from;
    loop {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 17;
        *seed ^= *seed << 5;
        let u = (*seed as f64 / u32::MAX as f64).clamp(1e-9, 1.0);
        t += ((-u.ln() / (rate_hz / 30_000.0)) as u64).max(60);
        if t >= to {
            return out;
        }
        out.push(SampleIndex(t));
    }
}

fn amps_for(seed: &mut u32, n: usize) -> Vec<f32> {
    (0..n).map(|_| 5.0 + 0.3 * gaussian(seed)).collect()
}

/// Ground truth plus templates (one per cluster, peak channel `peaks[c]`,
/// identity similarity so only the drift source can propose a pair).
struct Templated {
    gt: GroundTruth,
    tpl: Vec<Vec<u32>>,
    waves: Vec<f32>,
    sim: Vec<f32>,
}

impl DataProvider for Templated {
    type Label = u8;
    fn sample_rate(&self) -> f32 {
        self.gt.sample_rate()
    }
    fn n_channels(&self) -> u32 {
        4
    }
    fn n_samples(&self) -> SampleIndex {
        self.gt.n_samples
    }
    fn n_clusters(&self) -> u32 {
        self.gt.n_clusters()
    }
    fn spike_times(&self, c: ClusterId) -> &[SampleIndex] {
        self.gt.spike_times(c)
    }
    fn trace(&self, s: SampleIndex, l: u32) -> TraceSlice<'_> {
        self.gt.trace(s, l)
    }
    fn initial_labels(&self) -> Vec<u8> {
        self.gt.initial_labels()
    }
}
impl HasAmplitudes for Templated {
    fn spike_amplitudes(&self, c: ClusterId) -> &[f32] {
        self.gt.spike_amplitudes(c)
    }
}
impl HasSpikeTemplates for Templated {
    fn spike_templates(&self, c: ClusterId) -> &[u32] {
        self.tpl.get(c.idx()).map(Vec::as_slice).unwrap_or(&[])
    }
}
impl HasTemplateWaveforms for Templated {
    fn template_waveforms(&self) -> &[f32] {
        &self.waves
    }
    fn template_shape(&self) -> (usize, usize, usize) {
        (4, 4, 4)
    }
    fn similar_templates(&self) -> &[f32] {
        &self.sim
    }
}

/// Clusters: 0 = unit U before 15 s (channel 0), 1 = U after 15 s (channel 1,
/// 20 um away), 2 = an independent unit active only after 15 s on channel 2
/// (400 um away), 3 = a stationary unit on channel 0.
fn drift_session() -> (Session<Templated>, tempfile::TempDir) {
    let mut seed = 0x1234_5678_u32;
    let end = 900_000;
    let spikes = vec![
        poisson(&mut seed, 20.0, 0, 450_000),
        poisson(&mut seed, 20.0, 450_000, end),
        poisson(&mut seed, 20.0, 450_000, end),
        poisson(&mut seed, 20.0, 0, end),
    ];
    let amps: Vec<Vec<f32>> = spikes
        .iter()
        .map(|s| amps_for(&mut seed, s.len()))
        .collect();
    let tpl = spikes
        .iter()
        .enumerate()
        .map(|(c, s)| vec![c as u32; s.len()])
        .collect();
    let mut waves = vec![0.0; 4 * 4 * 4];
    for (t, peak) in [0, 1, 2, 0].into_iter().enumerate() {
        waves[(t * 4 + 1) * 4 + peak] = -1.0;
    }
    let sim = (0..16)
        .map(|i| if i / 4 == i % 4 { 1.0 } else { 0.0 })
        .collect();
    let prov = Templated {
        gt: GroundTruth {
            spikes,
            amps,
            n_samples: SampleIndex(end),
        },
        tpl,
        waves,
        sim,
    };
    let dir = tempfile::tempdir().unwrap();
    let journal = Journal::open(&dir.path().join("j.sqlite")).unwrap();
    let mut s = Session::new(prov, journal);
    s.seed_amplitudes();
    s.seed_templates();
    s.seed_template_waveforms();
    (s, dir)
}

#[test]
fn drift_fragment_pair_ranks_top3_and_distant_pair_does_not() {
    let (sess, _d) = drift_session();
    let cfg = SuggestConfig::default();
    // No positions: the drift source is off and templates gate the pair out.
    assert!(!rank_merge_candidates(&sess, &cfg)
        .iter()
        .any(|m| (m.a, m.b) == (ClusterId(0), ClusterId(1))));
    let mut memo = SuggestMemo::default();
    memo.positions = vec![[0.0, 0.0], [0.0, 20.0], [0.0, 400.0], [0.0, 420.0]];
    let merges = rank_merge_candidates_memo(&sess, &cfg, &mut memo);
    let rank = merges
        .iter()
        .position(|m| (m.a, m.b) == (ClusterId(0), ClusterId(1)))
        .unwrap_or_else(|| panic!("fragments not proposed: {merges:?}"));
    assert!(rank < 3, "fragment pair ranked {rank}: {merges:?}");
    // Cluster 2 is complementary to cluster 0 but 400 um away.
    assert!(!merges
        .iter()
        .any(|m| m.a.0.max(m.b.0) == 2 && m.a.0.min(m.b.0) == 0));
    // Cluster 3 fires throughout: not complementary to anything.
    assert!(!merges
        .iter()
        .any(|m| m.a == ClusterId(3) || m.b == ClusterId(3)));
}

fn same_merges(a: &[sorrel_data::MergeCandidate], b: &[sorrel_data::MergeCandidate]) -> bool {
    format!("{a:?}") == format!("{b:?}")
}

#[test]
fn memoised_ranking_matches_fresh_after_a_merge() {
    let (mut sess, _d) = fresh_session(build_ground_truth());
    let cfg = SuggestConfig::default();
    let mut memo = SuggestMemo::default();
    let cold = rank_merge_candidates_memo(&sess, &cfg, &mut memo);
    assert!(same_merges(&cold, &rank_merge_candidates(&sess, &cfg)));
    sess.dispatch(CurationCommand::Merge {
        sources: vec![ClusterId(0)],
        target: ClusterId(1),
    })
    .unwrap();
    let warm = rank_merge_candidates_memo(&sess, &cfg, &mut memo);
    assert!(same_merges(&warm, &rank_merge_candidates(&sess, &cfg)));
    let splits = rank_split_candidates_memo(&sess, &cfg, &mut memo);
    let fresh = rank_split_candidates(&sess, &cfg);
    assert_eq!(format!("{splits:?}"), format!("{fresh:?}"));
}

/// Ground truth plus cluster 4: a bursting cell whose amplitude falls with
/// shorter preceding ISI (raw amplitudes bimodal: first-in-burst vs rest).
fn with_burst_cell() -> GroundTruth {
    let mut gt = build_ground_truth();
    let mut seed = 0x0BAD_F00D_u32;
    let (mut times, mut amps) = (Vec::new(), Vec::new());
    let mut t = 0_u64;
    let mut last: Option<u64> = None;
    let rnd = |seed: &mut u32| {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 17;
        *seed ^= *seed << 5;
        *seed as f64 / u32::MAX as f64
    };
    while t < 850_000 {
        t += 3_000 + (rnd(&mut seed) * 3_000.0) as u64; // 100-200 ms between bursts
        for k in 0..4 {
            if k > 0 {
                t += 75 + (rnd(&mut seed) * 45.0) as u64; // 2.5-4 ms within
            }
            if let Some(l) = last {
                let isi_ms = (t - l) as f64 / 30.0;
                amps.push((2.9 + isi_ms.ln() + 0.15 * gaussian(&mut seed) as f64) as f32);
                times.push(SampleIndex(t));
            } else {
                amps.push(8.0);
                times.push(SampleIndex(t));
            }
            last = Some(t);
        }
    }
    gt.spikes.push(times);
    gt.amps.push(amps);
    gt
}

#[test]
fn burst_modulated_amplitude_is_not_a_top_split() {
    let (sess, _d) = fresh_session(with_burst_cell());
    let mut cfg = SuggestConfig::default();
    let splits = rank_split_candidates(&sess, &cfg);
    assert_eq!(
        splits[0].cluster,
        ClusterId(2),
        "true two-amplitude mix must stay on top"
    );
    assert!(
        !splits.iter().any(|s| s.cluster == ClusterId(4)),
        "bursting cell proposed: {splits:?}"
    );
    cfg.min_score = 0.0;
    let all = rank_split_candidates(&sess, &cfg);
    let burst = all.iter().find(|s| s.cluster == ClusterId(4)).unwrap();
    assert!(
        burst.amp_bimodality > 0.555,
        "premise: raw amplitudes bimodal"
    );
    assert_eq!(burst.reasons, ["burst_explained"]);
    assert!(all
        .iter()
        .find(|s| s.cluster == ClusterId(2))
        .unwrap()
        .reasons
        .is_empty());
}

impl sorrel_data::session::ApplyPhyLabel for GroundTruth {
    fn label_from_op(op: sorrel_data::PhyLabelOp) -> u8 {
        op as u8
    }
}
