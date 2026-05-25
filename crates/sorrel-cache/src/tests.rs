use crate::{CacheKey, CacheStore, Fingerprint};
use rkyv::{Archive, Deserialize, Serialize};

#[derive(Archive, Serialize, Deserialize, Debug, PartialEq)]
struct ToyArchive {
    id: u32,
    values: Vec<u32>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit)]
struct ToyParams {
    window: u32,
    scale: u32,
}

fn key(kind: &'static str, algo_version: u32, provider: &[u8], journal_head: u64) -> CacheKey {
    let params = ToyParams {
        window: 32,
        scale: 7,
    };
    let fingerprint = Fingerprint::builder()
        .add_provider_identity(provider)
        .add_algo_version(algo_version)
        .add_journal_head(journal_head)
        .add_params(&params)
        .finish();
    CacheKey::new(kind, algo_version, fingerprint)
}

#[test]
fn put_get_roundtrip_for_toy_archive() {
    let dir = tempfile::tempdir().unwrap();
    let store = CacheStore::open(dir.path()).unwrap();
    let key = key("amplitudes", 1, b"clusters-content-hash", 4);
    let value = ToyArchive {
        id: 42,
        values: vec![1, 3, 5, 8],
    };

    store.put(&key, &value).unwrap();
    let cached = store.get::<ToyArchive>(&key).unwrap().unwrap();

    assert_eq!(cached.id, 42);
    assert_eq!(cached.values.as_slice(), [1, 3, 5, 8]);
    assert!(!cached.bytes().is_empty());
}

#[test]
fn corrupted_file_returns_miss_and_evicts_file() {
    let dir = tempfile::tempdir().unwrap();
    let store = CacheStore::open(dir.path()).unwrap();
    let key = key("amplitudes", 1, b"clusters-content-hash", 4);
    let value = ToyArchive {
        id: 7,
        values: vec![9, 9],
    };

    store.put(&key, &value).unwrap();
    let path = store.file_path(&key).unwrap();
    std::fs::write(&path, b"not a valid rkyv archive").unwrap();

    assert!(store.get::<ToyArchive>(&key).unwrap().is_none());
    assert!(!path.exists());
}

#[test]
fn put_with_same_key_overwrites_atomically() {
    let dir = tempfile::tempdir().unwrap();
    let store = CacheStore::open(dir.path()).unwrap();
    let key = key("amplitudes", 1, b"clusters-content-hash", 4);

    store
        .put(
            &key,
            &ToyArchive {
                id: 1,
                values: vec![1],
            },
        )
        .unwrap();
    store
        .put(
            &key,
            &ToyArchive {
                id: 2,
                values: vec![2, 3],
            },
        )
        .unwrap();

    let cached = store.get::<ToyArchive>(&key).unwrap().unwrap();
    assert_eq!(cached.id, 2);
    assert_eq!(cached.values.as_slice(), [2, 3]);
}

#[test]
fn fingerprints_differ_when_key_inputs_change() {
    let base = key("amplitudes", 1, b"provider-a", 10).fingerprint;
    let provider = key("amplitudes", 1, b"provider-b", 10).fingerprint;
    let journal = key("amplitudes", 1, b"provider-a", 11).fingerprint;
    let algo = key("amplitudes", 2, b"provider-a", 10).fingerprint;

    let params = ToyParams {
        window: 64,
        scale: 7,
    };
    let param_change = CacheKey::new(
        "amplitudes",
        1,
        Fingerprint::builder()
            .add_provider_identity(b"provider-a")
            .add_algo_version(1)
            .add_journal_head(10)
            .add_params(&params)
            .finish(),
    )
    .fingerprint;

    assert_ne!(base, provider);
    assert_ne!(base, journal);
    assert_ne!(base, algo);
    assert_ne!(base, param_change);
}
