use rkyv::{Archive, Deserialize, Serialize};
use sorrel_cache::{CacheKey, Fingerprint};
use sorrel_io::ClusterId;

pub const KIND: &str = "pc_subspace";
pub const ALGO_VERSION: u32 = 1;

#[derive(Archive, Serialize, Deserialize, Debug, PartialEq)]
pub struct CachedPcSubspace {
    pub features: Vec<f32>,
    pub is_in_cluster: Vec<bool>,
    pub d: u32,
    pub algo_version: u32,
}

pub struct KeyParts<'a> {
    pub session_identity: &'a [u8],
    pub journal_head: u64,
    pub cluster: ClusterId,
    pub cluster_spike_count: usize,
    pub cluster_spike_fingerprint: [u8; 32],
    pub d_pcs: usize,
    pub channel_idx: usize,
    pub max_background: usize,
}

pub fn key(parts: KeyParts<'_>) -> CacheKey {
    key_with_algo_version(ALGO_VERSION, parts)
}

fn key_with_algo_version(algo_version: u32, parts: KeyParts<'_>) -> CacheKey {
    let fingerprint = Fingerprint::builder()
        .add_provider_identity(parts.session_identity)
        .add_journal_head(parts.journal_head)
        .add_algo_version(algo_version)
        .add_param_bytes("cluster_id", &parts.cluster.0.to_le_bytes())
        .add_param_bytes(
            "cluster_spike_count",
            &(parts.cluster_spike_count as u64).to_le_bytes(),
        )
        .add_param_bytes(
            "cluster_spike_fingerprint",
            &parts.cluster_spike_fingerprint,
        )
        .add_param_bytes("d_pcs", &(parts.d_pcs as u64).to_le_bytes())
        .add_param_bytes("channel_idx", &(parts.channel_idx as u64).to_le_bytes())
        .add_param_bytes(
            "max_background",
            &(parts.max_background as u64).to_le_bytes(),
        )
        .finish();
    CacheKey::new(KIND, algo_version, fingerprint)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sorrel_cache::CacheStore;

    fn fixture_key(algo_version: u32, d_pcs: usize) -> CacheKey {
        key_with_algo_version(
            algo_version,
            KeyParts {
                session_identity: b"provider-identity",
                journal_head: 17,
                cluster: ClusterId(3),
                cluster_spike_count: 2,
                cluster_spike_fingerprint: [0x5a; 32],
                d_pcs,
                channel_idx: 0,
                max_background: 5_000,
            },
        )
    }

    #[test]
    fn put_get_roundtrip_for_cached_pc_subspace() {
        let dir = tempfile::tempdir().unwrap();
        let store = CacheStore::open(dir.path()).unwrap();
        let key = fixture_key(ALGO_VERSION, 3);
        let value = CachedPcSubspace {
            features: vec![1.0, 2.0, 3.0, 4.0],
            is_in_cluster: vec![true, false],
            d: 2,
            algo_version: ALGO_VERSION,
        };

        store.put(&key, &value).unwrap();
        let cached = store.get::<CachedPcSubspace>(&key).unwrap().unwrap();

        let features: Vec<f32> = cached
            .features
            .as_slice()
            .iter()
            .map(|value| value.to_native())
            .collect();
        let is_in_cluster: Vec<bool> = cached.is_in_cluster.as_slice().to_vec();

        assert_eq!(features, value.features);
        assert_eq!(is_in_cluster, value.is_in_cluster);
        assert_eq!(cached.d.to_native(), value.d);
        assert_eq!(cached.algo_version.to_native(), value.algo_version);
        assert!(!cached.bytes().is_empty());
    }

    #[test]
    fn changing_algo_version_busts_cache_key() {
        assert_ne!(
            fixture_key(ALGO_VERSION, 3).fingerprint,
            fixture_key(ALGO_VERSION + 1, 3).fingerprint
        );
    }

    #[test]
    fn changing_d_pcs_busts_cache_key() {
        assert_ne!(
            fixture_key(ALGO_VERSION, 3).fingerprint,
            fixture_key(ALGO_VERSION, 4).fingerprint
        );
    }
}
