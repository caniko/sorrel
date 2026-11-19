use super::{hash_field, Api};
use crate::http_range::{strong_etag, Cache, Object};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    sync::Arc,
};
use ureq::http::StatusCode;
use zarrs::storage::{
    byte_range::{ByteRange, ByteRangeIterator},
    ListableStorageTraits, MaybeBytesIterator, ReadableStorageTraits, StorageError, StoreKey,
    StoreKeys, StoreKeysPrefixes, StorePrefix,
};

pub(crate) struct ZarrStore(BTreeMap<StoreKey, Arc<Object>>);

#[derive(Deserialize)]
struct Entry {
    #[serde(rename = "Key")]
    key: String,
    #[serde(rename = "ETag")]
    etag: String,
    #[serde(rename = "Size")]
    size: u64,
}

#[derive(Deserialize)]
struct Record {
    checksum: Option<String>,
}

// Match zarr_checksum 0.4.7's ZarrChecksumManifest and tree processing:
// https://github.com/dandi/zarr_checksum/blob/v0.4.7/zarr_checksum/checksum.py
// Field order, name sorting and Python's compact ASCII JSON are part of the digest.
#[derive(Serialize, Default)]
struct Manifest<'a> {
    directories: Vec<Child<'a>>,
    files: Vec<Child<'a>>,
    #[serde(skip)]
    count: usize,
    #[serde(skip)]
    size: u64,
}

#[derive(Serialize)]
struct Child<'a> {
    digest: String,
    name: &'a str,
    size: u64,
}

fn listing_checksum(objects: &BTreeMap<StoreKey, Arc<Object>>) -> Result<String> {
    let split = |path: &str| path.rfind('/').map_or(0, |i| i + 1);
    let mut dirs = BTreeMap::from([("", Manifest::default())]);
    let add = |manifest: &mut Manifest<'_>, count, size| -> Result<()> {
        manifest.count += count;
        manifest.size = manifest
            .size
            .checked_add(size)
            .context("Zarr listing size overflow")?;
        Ok(())
    };
    for (key, object) in objects {
        let (parent, name) = key.as_str().split_at(split(key.as_str()));
        let dir = dirs.entry(parent).or_default();
        add(dir, 1, object.size)?;
        dir.files.push(Child {
            digest: object.etag[1..object.etag.len() - 1].to_owned(),
            name,
            size: object.size,
        });
    }
    // Every directory prefix sorts before its descendants, so pop each child first.
    while let Some((path, mut dir)) = dirs.pop_last() {
        dir.directories.sort_unstable_by_key(|c| c.name);
        dir.files.sort_unstable_by_key(|c| c.name);
        let json = serde_json::to_string(&dir)?;
        let mut ascii = String::with_capacity(json.len());
        for c in json.chars() {
            if c < '\u{7f}' {
                ascii.push(c);
            } else {
                for unit in c.encode_utf16(&mut [0; 2]) {
                    write!(ascii, "\\u{unit:04x}").unwrap();
                }
            }
        }
        let digest = format!("{:x}-{}--{}", md5::compute(ascii), dir.count, dir.size);
        if path.is_empty() {
            return Ok(digest);
        }
        let path = path.strip_suffix('/').unwrap();
        let (parent, name) = path.split_at(split(path));
        let parent = dirs.entry(parent).or_default();
        add(parent, dir.count, dir.size)?;
        parent.directories.push(Child {
            digest,
            name,
            size: dir.size,
        });
    }
    unreachable!("checksum tree always has a root")
}

impl ZarrStore {
    pub(super) fn open(
        api: &Api,
        asset: &str,
        zarr: &str,
        asset_checksum: Option<&str>,
    ) -> Result<(Self, Vec<u8>)> {
        let record_url = api.endpoint(&format!("zarr/{zarr}/"));
        let response = api.get(&record_url)?;
        let record_checksum = if response.status() == StatusCode::NOT_FOUND {
            None
        } else {
            Api::decode_json::<Record>(&record_url, response)?.checksum
        };
        let record_checksum = record_checksum.as_deref().filter(|s| !s.is_empty());
        let asset_checksum = asset_checksum.filter(|s| !s.is_empty());
        if let (Some(record), Some(asset)) = (record_checksum, asset_checksum) {
            ensure!(
                record == asset,
                "DANDI Zarr record and asset checksums disagree"
            );
        }
        let expected = record_checksum
            .or(asset_checksum)
            .context("no DANDI Zarr checksum available")?;
        let endpoint = api.endpoint(&format!("zarr/{zarr}/files/"));
        let cache = Cache::new();
        let mut objects = BTreeMap::new();
        api.pages(endpoint.clone(), |entry: Entry| {
            let key = StoreKey::new(&entry.key)?;
            ensure!(
                entry.key.split('/').all(|p| !matches!(p, "" | "." | ".."))
                    && !entry.key.contains('\\')
                    && !entry.key.chars().any(char::is_control),
                "invalid key in DANDI Zarr listing"
            );
            let etag = if entry.etag.starts_with('"') {
                entry.etag
            } else {
                format!("\"{}\"", entry.etag)
            };
            strong_etag(&etag)?;
            let mut download = endpoint.clone();
            download
                .query_pairs_mut()
                .append_pair("prefix", &entry.key)
                .append_pair("download", "true");
            let source = api.source(format!("DANDI asset {asset}, key {}", entry.key), download);
            let object = Object::new(objects.len(), cache.clone(), source, entry.size, etag);
            ensure!(
                objects.insert(key, Arc::new(object)).is_none(),
                "duplicate key in DANDI Zarr listing"
            );
            Ok(())
        })?;
        let actual = listing_checksum(&objects)?;
        ensure!(
            actual == expected,
            "Zarr listing checksum mismatch: expected {expected}, computed {actual}"
        );
        let mut hash =
            blake3::Hasher::new_derive_key("sorrel-provider-identity-v1/remote-nwb-zarr");
        hash_field(&mut hash, asset);
        for (key, object) in &objects {
            hash_field(&mut hash, key.as_str());
            hash_field(&mut hash, &object.etag);
            hash.update(&object.size.to_le_bytes());
        }
        Ok((Self(objects), hash.finalize().as_bytes().to_vec()))
    }

    pub(crate) fn has_dir(&self, path: &str) -> bool {
        let Ok(prefix) = StorePrefix::new(format!("{path}/")) else {
            return false;
        };
        let found = self.keys(&prefix).next().is_some();
        found
    }

    fn keys<'a>(&'a self, prefix: &'a StorePrefix) -> impl Iterator<Item = &'a StoreKey> {
        let lower =
            StoreKey::new(prefix.as_str().trim_end_matches('/')).expect("valid store prefix");
        self.0
            .range(lower..)
            .map(|(key, _)| key)
            .skip_while(move |key| key.as_str() < prefix.as_str())
            .take_while(move |key| key.as_str().starts_with(prefix.as_str()))
    }
}

impl ReadableStorageTraits for ZarrStore {
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        let Some(object) = self.0.get(key) else {
            return Ok(None);
        };
        Ok(Some(Box::new(ranges.map(move |range| {
            let err = || {
                StorageError::Other(format!("{}: invalid Zarr byte range {range}", object.label))
            };
            let (start, end) = match range {
                ByteRange::FromStart(start, len) => (
                    start,
                    match len {
                        Some(len) => start.checked_add(len).ok_or_else(err)?,
                        None => object.size,
                    },
                ),
                ByteRange::Suffix(len) => {
                    (object.size.checked_sub(len).ok_or_else(err)?, object.size)
                }
            };
            if start > end || end > object.size {
                return Err(err());
            }
            let len = usize::try_from(end - start).map_err(|_| err())?;
            let mut bytes = Vec::new();
            bytes.try_reserve_exact(len).map_err(|_| {
                StorageError::Other(format!(
                    "{}: cannot allocate bytes {start}..{end}",
                    object.label
                ))
            })?;
            object
                .read_uninit(start, &mut bytes.spare_capacity_mut()[..len], false)
                .map_err(|err| StorageError::Other(format!("{err:#}")))?;
            // The range reader initializes every byte on success.
            unsafe { bytes.set_len(len) };
            Ok(bytes.into())
        }))))
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        Ok(self.0.get(key).map(|object| object.size))
    }

    fn supports_get_partial(&self) -> bool {
        true
    }
}

impl ListableStorageTraits for ZarrStore {
    fn list(&self) -> Result<StoreKeys, StorageError> {
        Ok(self.0.keys().cloned().collect())
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        Ok(self.keys(prefix).cloned().collect())
    }

    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        self.keys(prefix)
            .try_fold(0u64, |sum, key| sum.checked_add(self.0[key].size))
            .ok_or_else(|| StorageError::Other("Zarr store size overflow".into()))
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        let (mut keys, mut prefixes) = (Vec::new(), BTreeSet::new());
        for key in self.keys(prefix) {
            let start = prefix.as_str().len();
            if let Some(slash) = key.as_str()[start..].find('/') {
                prefixes.insert(StorePrefix::new(&key.as_str()[..start + slash + 1])?);
            } else {
                keys.push(key.clone());
            }
        }
        Ok(StoreKeysPrefixes::new(keys, prefixes.into_iter().collect()))
    }
}
