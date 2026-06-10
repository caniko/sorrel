use rkyv::{Archive, Deserialize, Serialize};
use sorrel_cache::{CacheKey, Fingerprint};
use sorrel_compute::IsolationMetrics;

pub const KIND: &str = "isolation";
// v2: added LDA d-prime and simplified silhouette fields.
pub const ALGO_VERSION: u32 = 2;

#[derive(Archive, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct CachedIsolationMetrics {
    pub isolation_distance_sq: f32,
    pub l_ratio: f32,
    pub nn_isolation: f32,
    pub d_prime: f32,
    pub silhouette: f32,
    pub n_in: u32,
    pub n_out: u32,
    pub algo_version: u32,
}

impl CachedIsolationMetrics {
    pub fn from_metrics(metrics: IsolationMetrics) -> Self {
        Self {
            isolation_distance_sq: metrics.isolation_distance_sq,
            l_ratio: metrics.l_ratio,
            nn_isolation: metrics.nn_isolation,
            d_prime: metrics.d_prime,
            silhouette: metrics.silhouette,
            n_in: metrics.n_in,
            n_out: metrics.n_out,
            algo_version: ALGO_VERSION,
        }
    }
}

pub fn key(subspace_key: &CacheKey, k_nn: usize) -> CacheKey {
    key_with_algo_version(ALGO_VERSION, subspace_key, k_nn)
}

fn key_with_algo_version(algo_version: u32, subspace_key: &CacheKey, k_nn: usize) -> CacheKey {
    let fingerprint = Fingerprint::builder()
        .add_algo_version(algo_version)
        .add_param_bytes("subspace_kind", subspace_key.kind.as_bytes())
        .add_param_bytes(
            "subspace_algo_version",
            &subspace_key.algo_version.to_le_bytes(),
        )
        .add_param_bytes("subspace_fingerprint", subspace_key.fingerprint.as_bytes())
        .add_param_bytes("k_nn", &(k_nn as u64).to_le_bytes())
        .finish();
    CacheKey::new(KIND, algo_version, fingerprint)
}

pub fn archived_to_metrics(
    cached: &<CachedIsolationMetrics as Archive>::Archived,
) -> Option<IsolationMetrics> {
    if cached.algo_version.to_native() != ALGO_VERSION {
        return None;
    }
    Some(IsolationMetrics {
        isolation_distance_sq: cached.isolation_distance_sq.to_native(),
        l_ratio: cached.l_ratio.to_native(),
        nn_isolation: cached.nn_isolation.to_native(),
        d_prime: cached.d_prime.to_native(),
        silhouette: cached.silhouette.to_native(),
        n_in: cached.n_in.to_native(),
        n_out: cached.n_out.to_native(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::pc_subspace;
    use sorrel_cache::CacheStore;
    use sorrel_io::ClusterId;

    fn subspace_key() -> CacheKey {
        pc_subspace::key(pc_subspace::KeyParts {
            session_identity: b"provider-identity",
            journal_head: 17,
            cluster: ClusterId(3),
            cluster_spike_count: 2,
            cluster_spike_fingerprint: [0x5a; 32],
            d_pcs: 3,
            channel_idx: 0,
            max_background: 5_000,
        })
    }

    #[test]
    fn put_get_roundtrip_for_cached_isolation_metrics() {
        let dir = tempfile::tempdir().unwrap();
        let store = CacheStore::open(dir.path()).unwrap();
        let key = key(&subspace_key(), 8);
        let metrics = IsolationMetrics {
            isolation_distance_sq: 12.5,
            l_ratio: 0.25,
            nn_isolation: 0.875,
            d_prime: 5.5,
            silhouette: 0.42,
            n_in: 120,
            n_out: 500,
        };

        store
            .put(&key, &CachedIsolationMetrics::from_metrics(metrics))
            .unwrap();
        let cached = store.get::<CachedIsolationMetrics>(&key).unwrap().unwrap();
        let roundtrip = archived_to_metrics(&cached).unwrap();

        assert_eq!(
            roundtrip.isolation_distance_sq,
            metrics.isolation_distance_sq
        );
        assert_eq!(roundtrip.l_ratio, metrics.l_ratio);
        assert_eq!(roundtrip.nn_isolation, metrics.nn_isolation);
        assert_eq!(roundtrip.d_prime, metrics.d_prime);
        assert_eq!(roundtrip.silhouette, metrics.silhouette);
        assert_eq!(roundtrip.n_in, metrics.n_in);
        assert_eq!(roundtrip.n_out, metrics.n_out);
        assert!(!cached.bytes().is_empty());
    }

    #[test]
    fn changing_algo_version_busts_cache_key() {
        let dir = tempfile::tempdir().unwrap();
        let store = CacheStore::open(dir.path()).unwrap();
        let subspace = subspace_key();
        let key_v1 = key_with_algo_version(ALGO_VERSION, &subspace, 8);
        let key_v2 = key_with_algo_version(ALGO_VERSION + 1, &subspace, 8);

        store
            .put(
                &key_v1,
                &CachedIsolationMetrics {
                    isolation_distance_sq: 1.0,
                    l_ratio: 2.0,
                    nn_isolation: 3.0,
                    d_prime: 4.5,
                    silhouette: 0.1,
                    n_in: 4,
                    n_out: 5,
                    algo_version: ALGO_VERSION,
                },
            )
            .unwrap();

        assert_ne!(key_v1.fingerprint, key_v2.fingerprint);
        assert!(store
            .get::<CachedIsolationMetrics>(&key_v2)
            .unwrap()
            .is_none());
    }

    #[test]
    fn changing_k_nn_busts_cache_key() {
        let subspace = subspace_key();

        assert_ne!(
            key(&subspace, 8).fingerprint,
            key(&subspace, 16).fingerprint
        );
    }
}
