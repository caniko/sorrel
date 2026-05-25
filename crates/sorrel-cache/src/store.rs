use crate::gc::GcConfig;
use crate::CacheKey;
use memmap2::Mmap;
use rkyv::{
    api::high::{HighSerializer, HighValidator},
    bytecheck::CheckBytes,
    rancor::Error as RkyvError,
    ser::allocator::ArenaHandle,
    util::AlignedVec,
    Archive, Portable, Serialize,
};
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    marker::PhantomData,
    ops::Deref,
    path::{Path, PathBuf},
    ptr::NonNull,
    time::{SystemTime, UNIX_EPOCH},
};

/// On-disk cache rooted at `<dataset_dir>/.sorrel/cache`.
#[derive(Clone, Debug)]
pub struct CacheStore {
    root: PathBuf,
}

impl CacheStore {
    pub fn open(dataset_dir: &Path) -> Result<Self, CacheError> {
        Self::open_with_gc_config(dataset_dir, GcConfig::from_env())
    }

    pub fn open_with_gc_config(
        dataset_dir: &Path,
        gc_config: GcConfig,
    ) -> Result<Self, CacheError> {
        let root = dataset_dir.join(".sorrel").join("cache");
        fs::create_dir_all(&root)?;
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(root.join("INDEX"))?
            .sync_all()?;
        let store = Self { root };
        store.gc_with_config(gc_config)?;
        Ok(store)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn cache_root_for(dataset_dir: &Path) -> PathBuf {
        dataset_dir.join(".sorrel").join("cache")
    }

    pub fn clear_dataset_cache(dataset_dir: &Path) -> Result<bool, CacheError> {
        let root = Self::cache_root_for(dataset_dir);
        if !root.exists() {
            return Ok(false);
        }
        let canonical_dataset = dataset_dir.canonicalize()?;
        let canonical_root = root.canonicalize()?;
        assert_safe_cache_root(&canonical_root)?;
        if !canonical_root.starts_with(&canonical_dataset) {
            return Err(CacheError::UnsafeClearPath(canonical_root));
        }
        fs::remove_dir_all(&canonical_root)?;
        if let Some(parent) = canonical_root.parent() {
            sync_dir(parent)?;
        }
        Ok(true)
    }

    pub fn get<A>(&self, key: &CacheKey) -> Result<Option<CacheRead<A>>, CacheError>
    where
        A: Archive,
        A::Archived: Portable + for<'a> CheckBytes<HighValidator<'a, RkyvError>>,
    {
        let path = self.file_path(key)?;
        let file = match File::open(&path) {
            Ok(file) => file,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(err) => return Err(err.into()),
        };
        let mmap = unsafe { Mmap::map(&file)? };
        let archived = match rkyv::access::<A::Archived, RkyvError>(&mmap) {
            Ok(archived) => NonNull::from(archived),
            Err(_) => {
                self.evict_file(&path)?;
                return Ok(None);
            }
        };
        Ok(Some(CacheRead {
            mmap,
            archived,
            _marker: PhantomData,
        }))
    }

    pub fn put<A>(&self, key: &CacheKey, value: &A) -> Result<(), CacheError>
    where
        A: Archive + for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, RkyvError>>,
    {
        let path = self.file_path(key)?;
        let dir = path
            .parent()
            .ok_or_else(|| CacheError::InvalidPath(path.clone()))?;
        fs::create_dir_all(dir)?;

        let bytes = rkyv::to_bytes::<RkyvError>(value).map_err(|err| {
            CacheError::Archive(format!(
                "serialize cache value for {}: {err}",
                key.fingerprint
            ))
        })?;
        let tmp_path = self.tmp_path(&path);
        let write_result = (|| -> Result<(), CacheError> {
            let mut tmp = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&tmp_path)?;
            tmp.write_all(&bytes)?;
            tmp.sync_all()?;
            fs::rename(&tmp_path, &path)?;
            sync_dir(dir)?;
            self.append_index(key, bytes.len() as u64)?;
            Ok(())
        })();

        if write_result.is_err() {
            let _ = fs::remove_file(&tmp_path);
        }
        write_result
    }

    pub fn evict(&self, key: &CacheKey) -> Result<bool, CacheError> {
        let path = self.file_path(key)?;
        self.evict_file(&path)
    }

    pub(crate) fn file_path(&self, key: &CacheKey) -> Result<PathBuf, CacheError> {
        validate_kind(key.kind)?;
        Ok(self
            .root
            .join(key.kind)
            .join(format!("{}.rkyv", key.fingerprint.hex())))
    }

    fn tmp_path(&self, path: &Path) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or_default();
        let filename = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("cache-entry.rkyv");
        path.with_file_name(format!("{filename}.tmp.{}.{}", std::process::id(), nonce))
    }

    fn append_index(&self, key: &CacheKey, len: u64) -> Result<(), CacheError> {
        let mtime = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| CacheError::ClockBeforeEpoch)?
            .as_secs();
        let mut index = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.root.join("INDEX"))?;
        writeln!(index, "{}|{}|{}|{}", key.kind, key.fingerprint, len, mtime)?;
        index.sync_all()?;
        Ok(())
    }

    fn evict_file(&self, path: &Path) -> Result<bool, CacheError> {
        match fs::remove_file(path) {
            Ok(()) => {
                if let Some(parent) = path.parent() {
                    sync_dir(parent)?;
                }
                Ok(true)
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(err) => Err(err.into()),
        }
    }
}

/// Guard for a bytechecked archived value borrowed from an owned mmap.
///
/// The archived reference is valid only while this guard is alive. Dropping the
/// guard releases the mmap, so callers must not store references derived from
/// it beyond the guard's lifetime.
pub struct CacheRead<A: Archive> {
    mmap: Mmap,
    archived: NonNull<A::Archived>,
    _marker: PhantomData<A>,
}

impl<A: Archive> CacheRead<A> {
    pub fn bytes(&self) -> &[u8] {
        &self.mmap
    }
}

impl<A: Archive> Deref for CacheRead<A> {
    type Target = A::Archived;

    fn deref(&self) -> &Self::Target {
        unsafe { self.archived.as_ref() }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    #[error("cache I/O error")]
    Io(#[from] std::io::Error),
    #[error("invalid cache kind {0:?}; use ASCII alnum, underscore, or dash")]
    InvalidKind(String),
    #[error("invalid cache path {0}")]
    InvalidPath(PathBuf),
    #[error("refusing to clear path that is not a .sorrel/cache directory: {0}")]
    UnsafeClearPath(PathBuf),
    #[error("{0}")]
    Archive(String),
    #[error("system clock is before UNIX_EPOCH")]
    ClockBeforeEpoch,
}

fn validate_kind(kind: &str) -> Result<(), CacheError> {
    let valid = !kind.is_empty()
        && kind
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-');
    if valid {
        Ok(())
    } else {
        Err(CacheError::InvalidKind(kind.to_string()))
    }
}

pub(crate) fn sync_dir(path: &Path) -> Result<(), CacheError> {
    File::open(path)?.sync_all()?;
    Ok(())
}

fn assert_safe_cache_root(path: &Path) -> Result<(), CacheError> {
    let is_cache_root = path.file_name().and_then(|name| name.to_str()) == Some("cache")
        && path
            .parent()
            .and_then(|parent| parent.file_name())
            .and_then(|name| name.to_str())
            == Some(".sorrel");
    if is_cache_root {
        Ok(())
    } else {
        Err(CacheError::UnsafeClearPath(path.to_path_buf()))
    }
}
