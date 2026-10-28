//! Re-ranking merge/split suggestions after one merge: cold vs memoised.
//! ~1k clusters, ~2M spikes; the memoised run must return exactly the
//! candidates a fresh memo does. `SORREL_BENCH_CLUSTERS` overrides K.

use sorrel_data::suggest::{rank_merge_candidates_memo, rank_split_candidates_memo, SuggestMemo};
use sorrel_data::SuggestConfig;
use sorrel_data::{session::ApplyPhyLabel, CurationCommand, Journal, PhyLabelOp, Session};
use sorrel_io::{ClusterId, DataProvider, HasAmplitudes, SampleIndex, TraceSamples, TraceSlice};
use std::time::Instant;

const TOTAL_SPIKES: usize = 2_000_000;

struct Synthetic {
    spikes: Vec<Vec<SampleIndex>>,
    amps: Vec<Vec<f32>>,
    n_samples: u64,
}

impl Synthetic {
    fn new(k: usize) -> Self {
        let per = TOTAL_SPIKES / k;
        let (mut spikes, mut amps) = (Vec::with_capacity(k), Vec::with_capacity(k));
        for c in 0..k {
            // Interleaved trains; the amplitude level spreads over 50 steps so
            // the mean-amplitude pre-filter prunes most distant pairs.
            let level = 3.0 + 0.2 * (c % 50) as f32;
            spikes.push((0..per).map(|i| SampleIndex((i * k + c) as u64)).collect());
            amps.push(
                (0..per)
                    .map(|i| level + ((i * 7919 + c) % 257) as f32 / 257.0)
                    .collect(),
            );
        }
        Self {
            spikes,
            amps,
            n_samples: (per * k) as u64,
        }
    }
}

impl DataProvider for Synthetic {
    type Label = u8;
    fn sample_rate(&self) -> f32 {
        30_000.0
    }
    fn n_channels(&self) -> u32 {
        1
    }
    fn n_samples(&self) -> SampleIndex {
        SampleIndex(self.n_samples)
    }
    fn n_clusters(&self) -> u32 {
        self.spikes.len() as u32
    }
    fn spike_times(&self, c: ClusterId) -> &[SampleIndex] {
        self.spikes.get(c.idx()).map_or(&[], Vec::as_slice)
    }
    fn trace(&self, start: SampleIndex, _len: u32) -> TraceSlice<'_> {
        TraceSlice {
            start,
            n_channels: 1,
            samples: TraceSamples::I16(&[]),
        }
    }
    fn initial_labels(&self) -> Vec<u8> {
        vec![0; self.spikes.len()]
    }
}

impl HasAmplitudes for Synthetic {
    fn spike_amplitudes(&self, c: ClusterId) -> &[f32] {
        self.amps.get(c.idx()).map_or(&[], Vec::as_slice)
    }
}

impl ApplyPhyLabel for Synthetic {
    fn label_from_op(op: PhyLabelOp) -> u8 {
        op as u8
    }
}

fn main() {
    let k: usize = std::env::var("SORREL_BENCH_CLUSTERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);
    let dir = tempfile::tempdir().unwrap();
    let journal = Journal::open(&dir.path().join("sorrel.journal")).unwrap();
    let mut session = Session::new(Synthetic::new(k), journal);
    session.seed_amplitudes();
    let cfg = SuggestConfig {
        top_merges: usize::MAX,
        top_splits: usize::MAX,
        min_score: 0.0,
        ..Default::default()
    };
    let rank = |memo: &mut SuggestMemo, s: &Session<Synthetic>| {
        let t = Instant::now();
        let m = rank_merge_candidates_memo(s, &cfg, memo);
        let sp = rank_split_candidates_memo(s, &cfg, memo);
        (format!("{m:?}{sp:?}"), m.len(), sp.len(), t.elapsed())
    };

    let mut memo = SuggestMemo::default();
    let (cold_out, nm, ns, cold) = rank(&mut memo, &session);
    println!("cold rank ({k} clusters, {TOTAL_SPIKES} spikes, {nm} merges, {ns} splits): {cold:?}");

    session
        .dispatch(CurationCommand::Merge {
            sources: vec![ClusterId(0)],
            target: ClusterId(1),
        })
        .unwrap();
    let (warm_out, _, _, warm) = rank(&mut memo, &session);
    let (fresh_out, _, _, fresh) = rank(&mut SuggestMemo::default(), &session);
    assert_eq!(warm_out, fresh_out, "memoised rank must equal a fresh rank");
    assert_ne!(cold_out, warm_out, "the merge must change the ranking");
    let (again_out, _, _, again) = rank(&mut memo, &session);
    assert_eq!(again_out, fresh_out);
    println!("re-rank after one merge, fresh memo:     {fresh:?}");
    println!("re-rank after one merge, memoised:       {warm:?}");
    println!("re-rank, nothing changed, memoised:      {again:?}");
    println!(
        "speedup after merge: {:.1}x",
        fresh.as_secs_f64() / warm.as_secs_f64().max(f64::EPSILON)
    );
}
