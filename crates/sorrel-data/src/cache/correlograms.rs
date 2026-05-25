use crate::session::Session;
use rkyv::{Archive, Deserialize, Serialize};
use sorrel_cache::{CacheKey, CacheStore, Fingerprint};
use sorrel_compute::{auto_correlogram, cross_correlogram};
use sorrel_io::{ClusterId, DataProvider, SampleIndex};

pub const ACG_KIND: &str = "correlogram_acg";
pub const CCG_KIND: &str = "correlogram_ccg";
pub const ALGO_VERSION: u32 = 1;

#[derive(Archive, Serialize, Deserialize, Debug, PartialEq)]
pub struct CachedCorrelogram {
    pub bins: Vec<u32>,
    pub bin_size_samples: u32,
    pub window_samples: u32,
    pub algo_version: u32,
}

pub fn acg_key(
    session_identity: &[u8],
    journal_head: u64,
    cluster_fp: [u8; 32],
    bin_size_samples: u32,
    window_samples: u32,
) -> CacheKey {
    let fingerprint = Fingerprint::builder()
        .add_provider_identity(session_identity)
        .add_journal_head(journal_head)
        .add_algo_version(ALGO_VERSION)
        .add_param_bytes("cluster_spike_fingerprint", &cluster_fp)
        .add_param_bytes("bin_size_samples", &bin_size_samples.to_le_bytes())
        .add_param_bytes("window_samples", &window_samples.to_le_bytes())
        .finish();
    CacheKey::new(ACG_KIND, ALGO_VERSION, fingerprint)
}

/// Builds the cache key for an ordered cross-correlogram pair.
///
/// The pair is intentionally directional: `cross_correlogram(a, b, ...)`
/// and `cross_correlogram(b, a, ...)` differ because the lag sign flips, so
/// callers must not sort the fingerprints before hashing.
pub fn ccg_key(
    session_identity: &[u8],
    journal_head: u64,
    a_fp: [u8; 32],
    b_fp: [u8; 32],
    bin_size_samples: u32,
    window_samples: u32,
) -> CacheKey {
    let fingerprint = Fingerprint::builder()
        .add_provider_identity(session_identity)
        .add_journal_head(journal_head)
        .add_algo_version(ALGO_VERSION)
        .add_param_bytes("cluster_a_spike_fingerprint", &a_fp)
        .add_param_bytes("cluster_b_spike_fingerprint", &b_fp)
        .add_param_bytes("bin_size_samples", &bin_size_samples.to_le_bytes())
        .add_param_bytes("window_samples", &window_samples.to_le_bytes())
        .finish();
    CacheKey::new(CCG_KIND, ALGO_VERSION, fingerprint)
}

pub fn acg_or_compute<P: DataProvider>(
    session: &Session<P>,
    cluster: ClusterId,
    bin_size_samples: u32,
    window_samples: u32,
) -> Vec<u32> {
    acg_or_compute_with(
        session,
        cluster,
        bin_size_samples,
        window_samples,
        auto_correlogram,
    )
}

pub fn ccg_or_compute<P: DataProvider>(
    session: &Session<P>,
    a: ClusterId,
    b: ClusterId,
    bin_size_samples: u32,
    window_samples: u32,
) -> Vec<u32> {
    ccg_or_compute_with(
        session,
        a,
        b,
        bin_size_samples,
        window_samples,
        cross_correlogram,
    )
}

fn acg_or_compute_with<P, F>(
    session: &Session<P>,
    cluster: ClusterId,
    bin_size_samples: u32,
    window_samples: u32,
    compute: F,
) -> Vec<u32>
where
    P: DataProvider,
    F: Fn(&[SampleIndex], u64, usize) -> Vec<u32>,
{
    let times = session.spike_times(cluster);
    if times.is_empty() || bin_size_samples == 0 || window_samples == 0 {
        return Vec::new();
    }
    let bins = histogram_bins(window_samples, bin_size_samples);
    if bins == 0 {
        return Vec::new();
    }

    let cluster_fp = spike_time_fingerprint(times);
    let key = acg_key(
        &session.provider.identity_bytes(),
        session.journal_head(),
        cluster_fp,
        bin_size_samples,
        window_samples,
    );
    if let Some(cached) = read_cached(session, &key, ACG_KIND, bin_size_samples, window_samples) {
        return cached;
    }

    let bins_vec = compute(times, window_samples as u64, bins);
    write_cached(
        session,
        &key,
        ACG_KIND,
        bin_size_samples,
        window_samples,
        bins_vec.clone(),
    );
    bins_vec
}

fn ccg_or_compute_with<P, F>(
    session: &Session<P>,
    a: ClusterId,
    b: ClusterId,
    bin_size_samples: u32,
    window_samples: u32,
    compute: F,
) -> Vec<u32>
where
    P: DataProvider,
    F: Fn(&[SampleIndex], &[SampleIndex], u64, usize) -> Vec<u32>,
{
    let times_a = session.spike_times(a);
    let times_b = session.spike_times(b);
    if times_a.is_empty() || times_b.is_empty() || bin_size_samples == 0 || window_samples == 0 {
        return Vec::new();
    }
    let bins = histogram_bins(window_samples, bin_size_samples);
    if bins == 0 {
        return Vec::new();
    }

    let key = ccg_key(
        &session.provider.identity_bytes(),
        session.journal_head(),
        spike_time_fingerprint(times_a),
        spike_time_fingerprint(times_b),
        bin_size_samples,
        window_samples,
    );
    if let Some(cached) = read_cached(session, &key, CCG_KIND, bin_size_samples, window_samples) {
        return cached;
    }

    let bins_vec = compute(times_a, times_b, window_samples as u64, bins);
    write_cached(
        session,
        &key,
        CCG_KIND,
        bin_size_samples,
        window_samples,
        bins_vec.clone(),
    );
    bins_vec
}

fn read_cached<P: DataProvider>(
    session: &Session<P>,
    key: &CacheKey,
    kind: &'static str,
    bin_size_samples: u32,
    window_samples: u32,
) -> Option<Vec<u32>> {
    let dataset_dir = session.cache_dataset_dir()?;
    let store = match CacheStore::open(dataset_dir) {
        Ok(store) => store,
        Err(err) => {
            log::debug!(
                "cache miss {}/{} after open error: {err}",
                kind,
                key.fingerprint.hex()
            );
            return None;
        }
    };

    match store.get::<CachedCorrelogram>(key) {
        Ok(Some(cached))
            if cached.algo_version.to_native() == ALGO_VERSION
                && cached.bin_size_samples.to_native() == bin_size_samples
                && cached.window_samples.to_native() == window_samples =>
        {
            log::debug!("cache hit {}/{}", kind, key.fingerprint.hex());
            Some(
                cached
                    .bins
                    .as_slice()
                    .iter()
                    .map(|value| value.to_native())
                    .collect(),
            )
        }
        Ok(Some(_)) | Ok(None) => {
            log::debug!("cache miss {}/{}", kind, key.fingerprint.hex());
            None
        }
        Err(err) => {
            log::debug!(
                "cache miss {}/{} after read error: {err}",
                kind,
                key.fingerprint.hex()
            );
            None
        }
    }
}

fn write_cached<P: DataProvider>(
    session: &Session<P>,
    key: &CacheKey,
    kind: &'static str,
    bin_size_samples: u32,
    window_samples: u32,
    bins: Vec<u32>,
) {
    let Some(dataset_dir) = session.cache_dataset_dir() else {
        return;
    };
    match CacheStore::open(dataset_dir) {
        Ok(store) => {
            let value = CachedCorrelogram {
                bins,
                bin_size_samples,
                window_samples,
                algo_version: ALGO_VERSION,
            };
            if let Err(err) = store.put(key, &value) {
                log::debug!(
                    "failed to write cache {}/{}: {err}",
                    kind,
                    key.fingerprint.hex()
                );
            }
        }
        Err(err) => {
            log::debug!(
                "failed to open cache {}/{} for write: {err}",
                kind,
                key.fingerprint.hex()
            );
        }
    }
}

fn histogram_bins(window_samples: u32, bin_size_samples: u32) -> usize {
    if window_samples == 0 || bin_size_samples == 0 {
        return 0;
    }
    let span = u64::from(window_samples) * 2;
    span.div_ceil(u64::from(bin_size_samples)) as usize
}

fn spike_time_fingerprint(times: &[SampleIndex]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"sorrel-correlogram-spike-times-v1");
    hasher.update(&(times.len() as u64).to_le_bytes());
    for spike in times {
        hasher.update(&spike.0.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sorrel_cache::CacheStore;
    use sorrel_io::{DataProvider, TraceSamples, TraceSlice};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    #[derive(Clone)]
    struct MockProvider {
        spikes: Vec<Vec<SampleIndex>>,
    }

    impl DataProvider for MockProvider {
        type Label = u8;

        fn sample_rate(&self) -> f32 {
            30_000.0
        }

        fn n_channels(&self) -> u32 {
            1
        }

        fn n_samples(&self) -> SampleIndex {
            self.spikes
                .iter()
                .flat_map(|cluster| cluster.iter().copied())
                .max()
                .unwrap_or(SampleIndex(0))
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

    fn fresh_session() -> (Session<MockProvider>, tempfile::TempDir) {
        let provider = MockProvider {
            spikes: vec![
                vec![
                    SampleIndex(10),
                    SampleIndex(30),
                    SampleIndex(50),
                    SampleIndex(80),
                ],
                vec![
                    SampleIndex(15),
                    SampleIndex(45),
                    SampleIndex(55),
                    SampleIndex(90),
                ],
                vec![],
            ],
        };
        let dir = tempfile::tempdir().unwrap();
        let journal = crate::SqliteJournal::open(&dir.path().join("j.sqlite")).unwrap();
        (Session::new(provider, journal), dir)
    }

    #[test]
    fn cached_correlogram_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let store = CacheStore::open(dir.path()).unwrap();
        let key = acg_key(b"provider", 7, [0x11; 32], 30, 1_500);
        let value = CachedCorrelogram {
            bins: vec![1, 2, 3, 5, 8],
            bin_size_samples: 30,
            window_samples: 1_500,
            algo_version: ALGO_VERSION,
        };

        store.put(&key, &value).unwrap();
        let cached = store.get::<CachedCorrelogram>(&key).unwrap().unwrap();

        let bins: Vec<u32> = cached
            .bins
            .as_slice()
            .iter()
            .map(|value| value.to_native())
            .collect();

        assert_eq!(bins, value.bins);
        assert_eq!(cached.bin_size_samples.to_native(), value.bin_size_samples);
        assert_eq!(cached.window_samples.to_native(), value.window_samples);
        assert_eq!(cached.algo_version.to_native(), value.algo_version);
    }

    #[test]
    fn acg_or_compute_hits_cache_on_second_call() {
        let (session, _dir) = fresh_session();
        let calls = Arc::new(AtomicUsize::new(0));
        let compute_calls = calls.clone();
        let compute = move |times: &[SampleIndex], max_lag: u64, bins: usize| {
            compute_calls.fetch_add(1, Ordering::SeqCst);
            auto_correlogram(times, max_lag, bins)
        };

        let first = acg_or_compute_with(&session, ClusterId(0), 30, 1_500, compute);
        let calls_after_first = calls.load(Ordering::SeqCst);
        let second =
            acg_or_compute_with(&session, ClusterId(0), 30, 1_500, |times, max_lag, bins| {
                calls.fetch_add(1, Ordering::SeqCst);
                auto_correlogram(times, max_lag, bins)
            });

        assert_eq!(first, second);
        assert_eq!(calls_after_first, 1);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn ccg_or_compute_keeps_ordered_pairs_distinct() {
        let (session, _dir) = fresh_session();
        let times_a = session.spike_times(ClusterId(0));
        let times_b = session.spike_times(ClusterId(1));
        let key_ab = ccg_key(
            &session.provider.identity_bytes(),
            session.journal_head(),
            spike_time_fingerprint(times_a),
            spike_time_fingerprint(times_b),
            30,
            1_500,
        );
        let key_ba = ccg_key(
            &session.provider.identity_bytes(),
            session.journal_head(),
            spike_time_fingerprint(times_b),
            spike_time_fingerprint(times_a),
            30,
            1_500,
        );
        assert_ne!(key_ab.fingerprint, key_ba.fingerprint);

        let calls = Arc::new(AtomicUsize::new(0));
        let first = ccg_or_compute_with(&session, ClusterId(0), ClusterId(1), 30, 1_500, {
            let calls = calls.clone();
            move |a, b, max_lag, bins| {
                calls.fetch_add(1, Ordering::SeqCst);
                cross_correlogram(a, b, max_lag, bins)
            }
        });
        let second = ccg_or_compute_with(&session, ClusterId(0), ClusterId(1), 30, 1_500, {
            let calls = calls.clone();
            move |a, b, max_lag, bins| {
                calls.fetch_add(1, Ordering::SeqCst);
                cross_correlogram(a, b, max_lag, bins)
            }
        });
        let reverse = ccg_or_compute_with(&session, ClusterId(1), ClusterId(0), 30, 1_500, {
            let calls = calls.clone();
            move |a, b, max_lag, bins| {
                calls.fetch_add(1, Ordering::SeqCst);
                cross_correlogram(a, b, max_lag, bins)
            }
        });

        assert_eq!(first, second);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_ne!(first, reverse);
    }
}
