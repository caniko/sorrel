use crate::{store::sync_dir, CacheError, CacheStore};
use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    time::{Duration, SystemTime},
};

const DEFAULT_MAX_AGE_DAYS: u64 = 30;
const DEFAULT_MAX_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const MAX_SCAN_ENTRIES: usize = 1_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GcConfig {
    pub max_age: Duration,
    pub max_bytes: u64,
    pub scan_limit: usize,
}

impl GcConfig {
    pub fn from_env() -> Self {
        let max_age_days = std::env::var("SORREL_CACHE_MAX_AGE_DAYS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(DEFAULT_MAX_AGE_DAYS);
        let max_bytes = std::env::var("SORREL_CACHE_MAX_BYTES")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(DEFAULT_MAX_BYTES);

        Self {
            max_age: Duration::from_secs(max_age_days.saturating_mul(24 * 60 * 60)),
            max_bytes,
            scan_limit: MAX_SCAN_ENTRIES,
        }
    }
}

impl Default for GcConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GcSummary {
    pub evicted_files: usize,
    pub evicted_bytes: u64,
    pub scanned_files: usize,
}

#[derive(Clone, Debug)]
struct CacheFile {
    path: PathBuf,
    index_key: Option<(String, String)>,
    len: u64,
    modified: SystemTime,
}

impl CacheStore {
    pub fn gc(&self) -> Result<GcSummary, CacheError> {
        self.gc_with_config(GcConfig::from_env())
    }

    pub fn gc_with_config(&self, config: GcConfig) -> Result<GcSummary, CacheError> {
        let indexed = read_index_keys(self.root())?;
        let mut files = scan_cache_files(self.root(), config.scan_limit)?;
        let mut summary = GcSummary {
            scanned_files: files.len(),
            ..GcSummary::default()
        };
        let now = SystemTime::now();

        let mut retained = Vec::with_capacity(files.len());
        for file in files.drain(..) {
            let referenced = file
                .index_key
                .as_ref()
                .is_some_and(|key| indexed.contains(key));
            let too_old = now
                .duration_since(file.modified)
                .map(|age| age > config.max_age)
                .unwrap_or(false);
            if !referenced || too_old {
                evict_file(&file.path, file.len, &mut summary)?;
            } else {
                retained.push(file);
            }
        }

        let mut total_bytes: u64 = retained.iter().map(|file| file.len).sum();
        if total_bytes > config.max_bytes {
            retained.sort_by_key(|file| file.modified);
            for file in retained {
                if total_bytes <= config.max_bytes {
                    break;
                }
                total_bytes = total_bytes.saturating_sub(file.len);
                evict_file(&file.path, file.len, &mut summary)?;
            }
        }

        let evicted_mib = summary.evicted_bytes as f64 / (1024.0 * 1024.0);
        if summary.evicted_files == 0 {
            log::debug!("cache gc: evicted 0 files, 0.0 MiB");
        } else {
            log::info!(
                "cache gc: evicted {} files, {:.1} MiB",
                summary.evicted_files,
                evicted_mib
            );
        }
        Ok(summary)
    }
}

fn read_index_keys(root: &Path) -> Result<HashSet<(String, String)>, CacheError> {
    let path = root.join("INDEX");
    let Ok(text) = fs::read_to_string(&path) else {
        return Ok(HashSet::new());
    };
    let mut keys = HashSet::new();
    for line in text.lines() {
        let mut parts = line.split('|');
        let Some(kind) = parts.next() else {
            continue;
        };
        let Some(fingerprint) = parts.next() else {
            continue;
        };
        if valid_index_key(kind, fingerprint) {
            keys.insert((kind.to_string(), fingerprint.to_string()));
        }
    }
    Ok(keys)
}

fn scan_cache_files(root: &Path, scan_limit: usize) -> Result<Vec<CacheFile>, CacheError> {
    let mut out = Vec::new();
    if !root.exists() {
        return Ok(out);
    }
    for entry in fs::read_dir(root)? {
        if out.len() >= scan_limit {
            break;
        }
        let entry = entry?;
        let path = entry.path();
        if path.file_name().and_then(|name| name.to_str()) == Some("INDEX") {
            continue;
        }
        if !path.is_dir() {
            continue;
        }
        let kind = entry.file_name().to_string_lossy().into_owned();
        for child in fs::read_dir(&path)? {
            if out.len() >= scan_limit {
                break;
            }
            let child = child?;
            let child_path = child.path();
            if !child_path.is_file()
                || child_path.extension().and_then(|ext| ext.to_str()) != Some("rkyv")
            {
                continue;
            }
            let metadata = child.metadata()?;
            let fingerprint = child_path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .map(str::to_string);
            let index_key = fingerprint
                .filter(|fingerprint| valid_index_key(&kind, fingerprint))
                .map(|fingerprint| (kind.clone(), fingerprint));
            out.push(CacheFile {
                path: child_path,
                index_key,
                len: metadata.len(),
                modified: metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            });
        }
    }
    Ok(out)
}

fn valid_index_key(kind: &str, fingerprint: &str) -> bool {
    let valid_kind = !kind.is_empty()
        && kind
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-');
    let valid_fingerprint = fingerprint.len() == 64
        && fingerprint
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase());
    valid_kind && valid_fingerprint
}

fn evict_file(path: &Path, len: u64, summary: &mut GcSummary) -> Result<(), CacheError> {
    match fs::remove_file(path) {
        Ok(()) => {
            summary.evicted_files += 1;
            summary.evicted_bytes = summary.evicted_bytes.saturating_add(len);
            if let Some(parent) = path.parent() {
                sync_dir(parent)?;
            }
            Ok(())
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CacheKey, Fingerprint};
    use rkyv::{Archive, Deserialize, Serialize};

    #[derive(Archive, Serialize, Deserialize)]
    struct Blob {
        bytes: Vec<u8>,
    }

    fn key(kind: &'static str, seed: u8) -> CacheKey {
        let fingerprint = Fingerprint::builder()
            .add_provider_identity(&[seed])
            .add_journal_head(seed as u64)
            .add_algo_version(1)
            .finish();
        CacheKey::new(kind, 1, fingerprint)
    }

    #[test]
    fn gc_evicts_orphan_files() {
        let dir = tempfile::tempdir().unwrap();
        let store = CacheStore::open_with_gc_config(
            dir.path(),
            GcConfig {
                max_age: Duration::from_secs(60),
                max_bytes: u64::MAX,
                scan_limit: 1_000,
            },
        )
        .unwrap();
        let root = dir.path().join(".sorrel/cache/orphans");
        fs::create_dir_all(&root).unwrap();
        let orphan = root.join(format!("{}.rkyv", "a".repeat(64)));
        fs::write(&orphan, b"orphan").unwrap();

        let summary = store
            .gc_with_config(GcConfig {
                max_age: Duration::from_secs(60),
                max_bytes: u64::MAX,
                scan_limit: 1_000,
            })
            .unwrap();
        assert!(summary.evicted_files >= 1);
        assert!(!orphan.exists());
    }

    #[test]
    fn gc_respects_max_age() {
        let dir = tempfile::tempdir().unwrap();
        let store = CacheStore::open_with_gc_config(
            dir.path(),
            GcConfig {
                max_age: Duration::from_secs(60),
                max_bytes: u64::MAX,
                scan_limit: 1_000,
            },
        )
        .unwrap();
        let key = key("aged", 1);
        store.put(&key, &Blob { bytes: vec![1; 16] }).unwrap();
        let path = store.file_path(&key).unwrap();

        store
            .gc_with_config(GcConfig {
                max_age: Duration::from_secs(0),
                max_bytes: u64::MAX,
                scan_limit: 1_000,
            })
            .unwrap();

        assert!(!path.exists());
    }

    #[test]
    fn gc_respects_max_bytes_with_lru_eviction() {
        let dir = tempfile::tempdir().unwrap();
        let store = CacheStore::open_with_gc_config(
            dir.path(),
            GcConfig {
                max_age: Duration::from_secs(60),
                max_bytes: u64::MAX,
                scan_limit: 1_000,
            },
        )
        .unwrap();
        let old = key("sized", 1);
        let new = key("sized", 2);

        store
            .put(
                &old,
                &Blob {
                    bytes: vec![1; 512],
                },
            )
            .unwrap();
        std::thread::sleep(Duration::from_millis(20));
        store
            .put(
                &new,
                &Blob {
                    bytes: vec![2; 512],
                },
            )
            .unwrap();
        let old_path = store.file_path(&old).unwrap();
        let new_path = store.file_path(&new).unwrap();
        let new_len = fs::metadata(&new_path).unwrap().len();

        store
            .gc_with_config(GcConfig {
                max_age: Duration::from_secs(60),
                max_bytes: new_len,
                scan_limit: 1_000,
            })
            .unwrap();

        assert!(!old_path.exists(), "oldest entry should be evicted first");
        assert!(new_path.exists(), "newest entry should remain under cap");
    }
}
