use super::SampleType;
use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;
use std::{fs, ops::Range, path::Path, sync::Arc};
use zarrs::array::{Array, Element, ElementOwned};
use zarrs::array_subset::ArraySubset;
use zarrs::config::MetadataRetrieveVersion;
use zarrs::filesystem::FilesystemStore;
use zarrs::group::Group as ZarrGroup;
use zarrs::metadata::{v2::DataTypeMetadataV2, ArrayMetadata};
use zarrs::storage::{ReadableListableStorage, ReadableListableStorageTraits, StoreKey};

#[path = "zarr/blosc.rs"]
mod blosc;

type Store = dyn ReadableListableStorageTraits;

pub(super) fn exact_number(value: &Value, what: &str) -> Result<f64> {
    // Don't round stored integers during conversion (use wider types to
    // avoid saturating at i64/u64). optional_attr checks for parser overflow.
    value
        .as_f64()
        .filter(|&f| {
            f.is_finite()
                && value.as_u64().is_none_or(|n| f as u128 == u128::from(n))
                && value.as_i64().is_none_or(|n| f as i128 == i128::from(n))
        })
        .with_context(|| format!("{what} must be a finite JSON number representable as float64"))
}

pub(super) fn open(path: &Path) -> Result<Group> {
    verify_consolidated_metadata(path)?;
    let store: ReadableListableStorage = Arc::new(FilesystemStore::new(path)?);
    Group::open(store, path.into(), "/")
}

fn verify_consolidated_metadata(root: &Path) -> Result<()> {
    let path = root.join(".zmetadata");
    match fs::symlink_metadata(&path) {
        Ok(meta) => ensure!(meta.is_file(), ".zmetadata is not a regular file"),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(err).context("inspect .zmetadata"),
    }
    let mut consolidated: Value =
        serde_json::from_slice(&fs::read(path).context("read .zmetadata")?)
            .context("parse .zmetadata")?;
    ensure!(
        consolidated["zarr_consolidated_format"] == 1,
        "unsupported .zmetadata zarr_consolidated_format"
    );
    let metadata = consolidated["metadata"]
        .as_object_mut()
        .context("expected a metadata object in .zmetadata")?;

    // Walk lexical paths rather than resolving keys supplied by .zmetadata.
    // Compare only metadata contents, never read data chunks here.
    fn walk(root: &Path, dir: &Path, metadata: &mut serde_json::Map<String, Value>) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let kind = entry.file_type()?;
            if matches!(
                entry.file_name().to_str(),
                Some(".zattrs" | ".zarray" | ".zgroup")
            ) {
                let key = path
                    .strip_prefix(root)?
                    .to_str()
                    .context("non-UTF-8 Zarr metadata key")?
                    .replace(std::path::MAIN_SEPARATOR, "/");
                ensure!(kind.is_file(), "Zarr metadata {key} is not a regular file");
                let expected = metadata.remove(&key).with_context(|| {
                    format!("Zarr metadata key {key} is missing from .zmetadata")
                })?;
                let actual: Value = serde_json::from_slice(
                    &fs::read(&path).with_context(|| format!("read Zarr metadata {key}"))?,
                )
                .with_context(|| format!("parse Zarr metadata {key}"))?;
                ensure!(
                    actual == expected,
                    "Zarr metadata key {key} disagrees with .zmetadata"
                );
            } else if kind.is_dir() {
                walk(root, &path, metadata)?;
            }
        }
        Ok(())
    }
    walk(root, root, metadata).context("verify .zmetadata")?;
    if let Some(key) = metadata.keys().next() {
        bail!("Zarr metadata key {key} in .zmetadata has no matching per-node metadata file");
    }
    Ok(())
}

pub(in crate::nwb::imp) struct Group(ZarrGroup<Store>, Arc<Path>);

impl Group {
    fn open(store: ReadableListableStorage, root: Arc<Path>, path: &str) -> Result<Self> {
        Ok(Self(
            ZarrGroup::open_opt(store, path, &MetadataRetrieveVersion::V2)?,
            root,
        ))
    }

    pub fn optional_attr(&self, name: &str) -> Option<&Value> {
        self.0.attributes().get(name)
    }

    fn links(&self) -> Result<&[Value]> {
        match self.0.attributes().get("zarr_link") {
            Some(links) => Ok(links.as_array().context("expected a zarr_link list")?),
            None => Ok(&[]),
        }
    }

    fn link(&self, name: &str) -> Result<Option<&Value>> {
        let mut found = None;
        for link in self.links()? {
            if link["name"].as_str().context("zarr_link has no name")? == name {
                ensure!(found.is_none(), "duplicate zarr_link {name}");
                found = Some(link);
            }
        }
        Ok(found)
    }

    pub fn member_names(&self) -> Result<Vec<String>> {
        let dir = self.1.join(&self.0.path().as_str()[1..]);
        let mut names = Vec::new();
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let kind = entry.file_type()?;
            if kind.is_dir() || kind.is_symlink() {
                names.push(
                    entry
                        .file_name()
                        .into_string()
                        .map_err(|_| anyhow::anyhow!("non-UTF-8 Zarr member name"))?,
                );
            }
        }
        for link in self.links()? {
            names.push(
                link["name"]
                    .as_str()
                    .context("zarr_link has no name")?
                    .to_owned(),
            );
        }
        names.sort_unstable();
        names.dedup();
        Ok(names)
    }

    /// Only a missing lexical member is absent. Do not follow aliases.
    pub fn local_path(&self, path: &str) -> Result<Option<String>> {
        let base = self.0.path().as_str().trim_end_matches('/');
        let path = if path.starts_with('/') {
            path.to_owned()
        } else {
            format!("{base}/{path}")
        };
        let mut prefix = String::new();
        for name in components(&path)? {
            let parent_path = if prefix.is_empty() { "/" } else { &prefix };
            let parent;
            let group = if self.0.path().as_str() == parent_path {
                self
            } else {
                parent = self.open_group(parent_path)?;
                &parent
            };
            prefix.push('/');
            prefix.push_str(name);
            if let Some(link) = group.link(name)? {
                ensure!(
                    link["source"].as_str() == Some("."),
                    "{prefix} is an external zarr_link"
                );
                bail!("{prefix} is a same-store zarr_link (soft link)");
            }
            match fs::symlink_metadata(self.1.join(&prefix[1..])) {
                Ok(meta) => ensure!(
                    meta.is_dir(),
                    "{prefix} is not a Zarr directory (files and symbolic links are unsupported)"
                ),
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(err) => {
                    return Err(err).with_context(|| format!("inspect Zarr member {prefix}"))
                }
            }
        }
        Ok(Some(path))
    }

    pub fn open_group(&self, path: &str) -> Result<Self> {
        Self::open(self.0.storage(), self.1.clone(), path)
            .with_context(|| format!("open Zarr group {path}"))
    }

    pub fn child_group(&self, name: &str) -> Result<Option<Self>> {
        let Some(path) = self.local_path(name)? else {
            return Ok(None);
        };
        let store = self.0.storage();
        let key = |name| StoreKey::new(format!("{}/{name}", &path[1..]));
        // Classify arrays from metadata without requiring a supported codec.
        if let Some(meta) = store.get(&key(".zarray")?)? {
            ensure!(
                store.get(&key(".zgroup")?)?.is_none(),
                "ambiguous Zarr node {path}: both .zarray and .zgroup"
            );
            ensure!(
                serde_json::from_slice::<Value>(&meta)?["zarr_format"] == 2,
                "invalid Zarr array metadata for {path}"
            );
            return Ok(None);
        }
        self.open_group(&path).map(Some)
    }

    pub fn open_dataset(&self, path: &str, spec: bool) -> Result<Dataset> {
        let store = self.0.storage();
        let key = |name| StoreKey::new(format!("{}/{name}", &path[1..]));
        let meta = store
            .get(&key(".zarray")?)?
            .with_context(|| format!("missing Zarr v2 array metadata for {path}"))?;
        let meta: Value =
            serde_json::from_slice(&meta).with_context(|| format!("parse .zarray for {path}"))?;
        ensure!(meta["zarr_format"] == 2, "expected Zarr v2 array at {path}");
        ensure!(
            meta["order"] == "C",
            "unsupported Zarr order {} for array {path}; expected C",
            meta["order"]
        );
        if let Some(separator) = meta.get("dimension_separator") {
            ensure!(
                matches!(separator.as_str(), Some("." | "/")),
                "unsupported Zarr dimension_separator {separator} for array {path}"
            );
        }
        // zarrs would invent a default for an undefined v2 fill value.
        ensure!(
            !meta["fill_value"].is_null(),
            "undefined Zarr fill_value for array {path}"
        );
        let compressor = &meta["compressor"];
        if !compressor.is_null() {
            ensure!(
                matches!(compressor["id"].as_str(), Some("blosc" | "zlib" | "gzip")),
                "unsupported Zarr compressor {} for array {path}",
                compressor["id"]
            );
        }
        let filters = match &meta["filters"] {
            Value::Null => &[][..],
            Value::Array(filters) => filters.as_slice(),
            filters => bail!("invalid Zarr filters {filters} for array {path}"),
        };
        // hdmf-zarr writes cached specs as a single uncompressed object
        // chunk using numcodecs JSON. Decode that specific layout ourselves.
        let json_spec = spec
            && matches!(filters, [filter] if filter["id"] == "json2"
                && filter.get("encoding").is_none_or(|v| v == "utf-8"))
            && meta["dtype"] == "|O"
            && meta["shape"] == serde_json::json!([1])
            && meta["chunks"] == serde_json::json!([1])
            && compressor.is_null();
        let data = if json_spec {
            Data::JsonString {
                store,
                key: key("0")?,
            }
        } else {
            // These object encodings are needed for strings (e.g. the NWB
            // epoch), not numeric filters. Never deserialize pickle.
            if let [filter, ..] = filters {
                ensure!(
                    filters.len() == 1
                        && meta["dtype"] == "|O"
                        && (filter == &serde_json::json!({"id":"vlen-utf8"})
                            || filter == &serde_json::json!({"id":"vlen-bytes"})),
                    "unsupported Zarr filters {} for array {path}",
                    meta["filters"]
                );
            }
            Data::Array(Box::new(Array::open_opt(
                store,
                path,
                &MetadataRetrieveVersion::V2,
            )?))
        };
        let shape = match &data {
            Data::Array(array) => array
                .shape()
                .iter()
                .map(|&n| usize::try_from(n))
                .collect::<Result<_, _>>()?,
            Data::JsonString { .. } => vec![1],
        };
        Ok(Dataset {
            data,
            shape,
            attrs_key: key(".zattrs")?,
        })
    }
}

fn components(path: &str) -> Result<Vec<&str>> {
    ensure!(
        path.starts_with('/'),
        "expected an absolute Zarr object path"
    );
    if path == "/" {
        return Ok(Vec::new());
    }
    path[1..]
        .split('/')
        .map(|name| {
            ensure!(
                !name.is_empty() && !matches!(name, "." | "..") && !name.contains(['\\', ':']),
                "invalid Zarr object path {path:?}"
            );
            Ok(name)
        })
        .collect()
}

enum Data {
    Array(Box<Array<Store>>),
    JsonString {
        store: ReadableListableStorage,
        key: StoreKey,
    },
}

pub(in crate::nwb::imp) struct Dataset {
    data: Data,
    shape: Vec<usize>,
    attrs_key: StoreKey,
}

pub(in crate::nwb::imp) trait Number:
    ElementOwned + num_traits::NumCast
{
}
impl<T: ElementOwned + num_traits::NumCast> Number for T {}

macro_rules! numeric_types {
    ($body:ident) => {
        $body!(I8: i8, U8: u8, I16: i16, U16: u16, I32: i32, U32: u32, I64: i64, U64: u64, F32: f32, F64: f64)
    };
}

impl Dataset {
    pub fn exact_float(&self, what: &str) -> Result<()> {
        ensure!(
            matches!(self.array()?.metadata(), ArrayMetadata::V2(meta)
                if matches!(&meta.dtype, DataTypeMetadataV2::Simple(dtype)
                    if matches!(dtype.as_str(), "<f4" | "<f8"))),
            "{what} must be stored as Zarr v2 little-endian IEEE float32 or float64"
        );
        Ok(())
    }

    fn array(&self) -> Result<&Array<Store>> {
        match &self.data {
            Data::Array(a) => Ok(a),
            Data::JsonString { .. } => bail!("expected a Zarr array"),
        }
    }

    pub fn optional_attr(&self, name: &str) -> Result<Option<&Value>> {
        let array = self.array()?;
        let value = array.attributes().get(name);
        if value.is_some_and(Value::is_f64) {
            // serde_json also exposes overflowing integer tokens as f64.
            // Check the original token without changing JSON parsing across
            // the workspace (arbitrary_precision affects other decoders).
            let raw = array
                .storage()
                .get(&self.attrs_key)?
                .context("missing Zarr .zattrs")?;
            let attrs: std::collections::HashMap<String, &serde_json::value::RawValue> =
                serde_json::from_slice(&raw)?;
            let token = attrs.get(name).context("missing raw Zarr attribute")?.get();
            ensure!(
                token.contains(['.', 'e', 'E']) || token == "-0",
                "Zarr attribute {name}: JSON integer is outside the supported i64/u64 range"
            );
        }
        Ok(value)
    }

    pub fn reference_path(&self, name: &str) -> Result<Option<String>> {
        let Some(attr) = self.array()?.attributes().get(name) else {
            return Ok(None);
        };
        let reference = &attr["value"];
        ensure!(
            attr["zarr_dtype"] == "object",
            "invalid Zarr object reference {name}"
        );
        ensure!(
            reference["source"].as_str() == Some("."),
            "external Zarr object reference {name}"
        );
        let path = reference["path"]
            .as_str()
            .context("Zarr object reference has no path")?;
        components(path)?;
        Ok(Some(path.to_owned()))
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn chunk(&self) -> Option<Vec<usize>> {
        self.array()
            .ok()?
            .chunk_shape_usize(&vec![0; self.shape.len()])
            .ok()
    }

    pub fn sample_type(&self) -> Result<SampleType> {
        let a = self.array()?;
        macro_rules! find_type {
            ($($kind:ident: $ty:ty),*) => { $(
                if <$ty>::validate_data_type(a.data_type()).is_ok() { return Ok(SampleType::$kind); }
            )* };
        }
        numeric_types!(find_type);
        bail!("unsupported Zarr sample datatype {}", a.data_type())
    }

    fn read<T: Number>(&self, subset: &ArraySubset) -> Result<Vec<T>> {
        let a = self.array()?;
        if T::validate_data_type(a.data_type()).is_ok() {
            return Ok(a.retrieve_array_subset_elements::<T>(subset)?);
        }
        macro_rules! convert {
            ($($kind:ident: $ty:ty),*) => { match self.sample_type()? { $(
                SampleType::$kind => a.retrieve_array_subset_elements::<$ty>(subset)?
                    .into_iter().map(|n| T::from(n).context("Zarr numeric conversion out of range")).collect(),
            )* } };
        }
        numeric_types!(convert)
    }

    pub fn read_rows<T: Number>(&self, range: Range<usize>) -> Result<Vec<T>> {
        let rows = range.start as u64..range.end as u64;
        let subset = match *self.shape {
            [_] => ArraySubset::new_with_ranges(&[rows]),
            [_, n] => ArraySubset::new_with_ranges(&[rows, 0..n as u64]),
            _ => bail!("expected a rank 1 or 2 Zarr dataset"),
        };
        self.read(&subset)
    }

    pub fn read_raw<T: Number>(&self) -> Result<Vec<T>> {
        self.read(&self.array()?.subset_all())
    }

    fn check_scalar(&self) -> Result<()> {
        ensure!(
            self.shape.is_empty()
                || (self.shape == [1]
                    && self
                        .optional_attr("zarr_dtype")?
                        .is_some_and(|v| v == "scalar")),
            "expected a scalar Zarr dataset"
        );
        Ok(())
    }

    pub fn read_number(&self) -> Result<f64> {
        self.check_scalar()?;
        self.read_raw()?.pop().context("missing scalar Zarr number")
    }

    pub fn read_string(&self) -> Result<String> {
        if matches!(&self.data, Data::Array(_)) {
            self.check_scalar()?;
        }
        self.strings()?.pop().context("missing scalar Zarr string")
    }

    pub fn strings(&self) -> Result<Vec<String>> {
        match &self.data {
            Data::Array(a) if Vec::<u8>::validate_data_type(a.data_type()).is_ok() => a
                .retrieve_array_subset_elements::<Vec<u8>>(&a.subset_all())?
                .into_iter()
                .map(|s| Ok(String::from_utf8(s)?))
                .collect(),
            Data::Array(a) => Ok(a.retrieve_array_subset_elements::<String>(&a.subset_all())?),
            Data::JsonString { store, key } => {
                let bytes = store.get(key)?.context("missing Zarr JSON chunk")?;
                // Numcodecs appends dtype and shape to the flattened values.
                let (text, dtype, shape): (String, String, [usize; 1]) =
                    serde_json::from_slice(&bytes)?;
                ensure!(
                    dtype == "|O" && shape == [1],
                    "invalid Zarr JSON string chunk"
                );
                Ok(vec![text])
            }
        }
    }
}

/// Seal metadata contents and stat chunks without reading the recording. Like
/// the HDF5 seal this assumes a static input and does not detect edits that
/// deliberately preserve chunk length and mtime.
pub(super) fn identity(root: &Path) -> Result<Vec<u8>> {
    fn walk(root: &Path, dir: &Path, hash: &mut blake3::Hasher) -> Result<()> {
        let mut entries = fs::read_dir(dir)?.collect::<Result<Vec<_>, _>>()?;
        entries.sort_unstable_by_key(|entry| entry.file_name());
        for entry in entries {
            let kind = entry.file_type()?;
            let path = entry.path();
            if kind.is_dir() {
                walk(root, &path, hash)?;
                continue;
            }
            ensure!(
                kind.is_file(),
                "non-file entry (or symbolic link) in Zarr store: {}",
                path.display()
            );
            let key = path.strip_prefix(root)?.to_string_lossy();
            hash.update(&(key.len() as u64).to_le_bytes());
            hash.update(key.as_bytes());
            if matches!(
                entry.file_name().to_str(),
                Some(".zattrs" | ".zarray" | ".zgroup" | ".zmetadata" | "zarr.json")
            ) {
                hash.update(blake3::hash(&fs::read(path)?).as_bytes());
            } else {
                let meta = entry.metadata()?;
                hash.update(&meta.len().to_le_bytes());
                let nanos = meta
                    .modified()?
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_nanos();
                hash.update(&nanos.to_le_bytes());
            }
        }
        Ok(())
    }
    let mut hash = blake3::Hasher::new();
    hash.update(b"sorrel-provider-identity-v1/nwb-zarr\0");
    hash.update(root.file_name().unwrap_or_default().as_encoded_bytes());
    hash.update(b"\0");
    walk(root, root, &mut hash).context("seal NWB-Zarr directory store")?;
    Ok(hash.finalize().as_bytes().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn json_clock_attributes_roundtrip_without_rounding_stored_integers() {
        let parse = |token| exact_number(&serde_json::from_str(token).unwrap(), "rate");
        assert_eq!(parse("12711.729583157607").unwrap(), 12711.729583157607);
        assert_eq!(parse("9007199254740992").unwrap(), 9007199254740992.0);
        assert!(parse("9007199254740993").is_err());
        assert!(parse("18446744073709551615").is_err());
        assert!(parse(r#""1000""#).is_err());
    }

    #[test]
    fn consolidated_metadata_must_match_in_both_directions() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let group = json!({"zarr_format": 2});
        let key = "acquisition/Raw/starting_time/.zattrs";
        let attrs = root.join(key);
        fs::create_dir_all(attrs.parent().unwrap()).unwrap();
        fs::write(root.join(".zgroup"), group.to_string()).unwrap();
        fs::write(&attrs, r#"{"rate":1000.0}"#).unwrap();
        let consolidate = |metadata| {
            fs::write(
                root.join(".zmetadata"),
                json!({"zarr_consolidated_format": 1, "metadata": metadata}).to_string(),
            )
            .unwrap();
        };
        let refuses_key = || {
            let err = open(root).err().expect("opened inconsistent metadata");
            assert!(format!("{err:#}").contains(key), "{err:#}");
        };

        assert!(open(root).is_ok());
        consolidate(json!({".zgroup": group, key: {"rate": 1000.0}}));
        assert!(open(root).is_ok());
        fs::write(&attrs, r#"{"rate":2000.0}"#).unwrap();
        refuses_key();
        fs::remove_file(&attrs).unwrap();
        refuses_key();
        fs::write(&attrs, r#"{"rate":1000.0}"#).unwrap();
        consolidate(json!({".zgroup": group}));
        refuses_key();
    }

    #[test]
    fn rejects_fixedscaleoffset_but_keeps_verified_string_encodings() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join(".zgroup"), r#"{"zarr_format":2}"#).unwrap();
        let path = "/acquisition/Raw/data";
        let data = root.join(&path[1..]);
        fs::create_dir_all(&data).unwrap();
        let meta = json!({
            "zarr_format": 2, "shape": [256, 1], "chunks": [256, 1], "dtype": "<i2",
            "compressor": null, "fill_value": 0, "order": "C",
            "filters": [{"id": "fixedscaleoffset", "offset": 0, "scale": 1024.0,
                         "dtype": "<i2", "astype": "<i4"}]
        });
        fs::write(data.join(".zarray"), meta.to_string()).unwrap();
        let err = open(root)
            .unwrap()
            .open_dataset(path, false)
            .err()
            .expect("opened an array with fixedscaleoffset");
        let message = format!("{err:#}");
        assert!(
            message.contains(path) && message.contains("fixedscaleoffset"),
            "{message}"
        );

        // Reference values decoded with numcodecs 0.15.1 from the hdmf-zarr
        // fixture: VLenBytes for the epoch and VLenUTF8 for locations.
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../release/fixtures/nwb/good.nwb.zarr");
        let group = open(&fixture).unwrap();
        let read = |path| group.open_dataset(path, false).unwrap().strings().unwrap();
        assert_eq!(
            read("/timestamps_reference_time"),
            ["2026-01-01T00:00:00+00:00"]
        );
        assert_eq!(
            read("/general/extracellular_ephys/electrodes/location"),
            ["CA1"; 6]
        );
    }
}
