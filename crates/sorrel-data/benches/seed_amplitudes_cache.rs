use sorrel_data::{Journal, Session};
use sorrel_io::{ClusterId, DataProvider, HasAmplitudes, SampleIndex, TraceSamples, TraceSlice};
use std::time::{Duration, Instant};

const N_CLUSTERS: usize = 100;
const SPIKES_PER_CLUSTER: usize = 10_000;

struct SyntheticProvider {
    spikes: Vec<Vec<SampleIndex>>,
    amps: Vec<Vec<f32>>,
}

impl SyntheticProvider {
    fn new() -> Self {
        let mut spikes = Vec::with_capacity(N_CLUSTERS);
        let mut amps = Vec::with_capacity(N_CLUSTERS);
        for cluster in 0..N_CLUSTERS {
            let mut cluster_spikes = Vec::with_capacity(SPIKES_PER_CLUSTER);
            let mut cluster_amps = Vec::with_capacity(SPIKES_PER_CLUSTER);
            for i in 0..SPIKES_PER_CLUSTER {
                cluster_spikes.push(SampleIndex::new((i * N_CLUSTERS + cluster) as u64));
                cluster_amps.push(((i % 257) as f32) * 0.25 + cluster as f32);
            }
            spikes.push(cluster_spikes);
            amps.push(cluster_amps);
        }
        Self { spikes, amps }
    }
}

impl DataProvider for SyntheticProvider {
    type Label = u8;

    fn sample_rate(&self) -> f32 {
        30_000.0
    }

    fn n_channels(&self) -> u32 {
        1
    }

    fn n_samples(&self) -> SampleIndex {
        SampleIndex::new((N_CLUSTERS * SPIKES_PER_CLUSTER) as u64)
    }

    fn n_clusters(&self) -> u32 {
        self.spikes.len() as u32
    }

    fn spike_times(&self, cluster: ClusterId) -> &[SampleIndex] {
        self.spikes
            .get(cluster.idx())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    fn trace(&self, start: SampleIndex, _len: u32) -> TraceSlice<'_> {
        TraceSlice {
            start,
            n_channels: 1,
            samples: TraceSamples::I16(&[]),
        }
    }

    fn initial_labels(&self) -> Vec<Self::Label> {
        vec![0; self.spikes.len()]
    }
}

impl HasAmplitudes for SyntheticProvider {
    fn spike_amplitudes(&self, cluster: ClusterId) -> &[f32] {
        std::thread::sleep(Duration::from_micros(500));
        self.amps
            .get(cluster.idx())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}

fn timed_seed(root: &std::path::Path) -> Duration {
    let provider = SyntheticProvider::new();
    let journal = Journal::open(&root.join("sorrel.journal")).unwrap();
    let mut session = Session::new(provider, journal);
    let start = Instant::now();
    session.seed_amplitudes();
    start.elapsed()
}

fn main() {
    let dir = tempfile::tempdir().unwrap();
    let cold = timed_seed(dir.path());
    let warm = timed_seed(dir.path());
    let ratio = cold.as_secs_f64() / warm.as_secs_f64().max(f64::EPSILON);

    println!("seed_amplitudes cache cold: {cold:?}");
    println!("seed_amplitudes cache warm: {warm:?}");
    println!("seed_amplitudes cache warmup ratio: {ratio:.2}x");

    let (k, t) = timed_index_build();
    println!(
        "index build ({k} clusters, {} spikes): {t:?}",
        N_CLUSTERS * SPIKES_PER_CLUSTER
    );
}

/// Time `ClusterIndex::from_provider` (via `Session::new`) for the k-way
/// merge of per-cluster buckets. `SORREL_BENCH_CLUSTERS` overrides the
/// cluster count so the K dependence is visible.
fn timed_index_build() -> (usize, Duration) {
    let n_clusters: usize = std::env::var("SORREL_BENCH_CLUSTERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(N_CLUSTERS);
    let per = (N_CLUSTERS * SPIKES_PER_CLUSTER) / n_clusters.max(1);
    let spikes: Vec<Vec<SampleIndex>> = (0..n_clusters)
        .map(|c| {
            (0..per)
                .map(|i| SampleIndex::new((i * n_clusters + c) as u64))
                .collect()
        })
        .collect();
    let provider = SyntheticProvider {
        amps: vec![Vec::new(); n_clusters],
        spikes,
    };
    let start = Instant::now();
    let index = sorrel_data::ClusterIndex::from_provider(&provider);
    (index.n_clusters() as usize, start.elapsed())
}
