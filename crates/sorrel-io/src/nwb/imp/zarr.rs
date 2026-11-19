use super::SampleType;
use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;
use std::{fs, ops::Range, path::Path, sync::Arc};
use zarrs::array::{Array, ArraySubset, Element, ElementOwned};
use zarrs::filesystem::FilesystemStore;
use zarrs::group::Group as ZarrGroup;
use zarrs::metadata::{v2::DataTypeMetadataV2, ArrayMetadata};
use zarrs::storage::{ReadableListableStorage, ReadableListableStorageTraits, StoreKey};

type Store = dyn ReadableListableStorageTraits;

pub(super) fn exact_number(value: &Value, what: &str) -> Result<f64> {
    // Decimal floating-point tokens are parsed with float_roundtrip. JSON
    // integers may have been stored as integers by serde_json; don't round
    // those during conversion (use wider types to avoid saturating at i64/u64).
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
    let store: ReadableListableStorage = Arc::new(FilesystemStore::new(path)?);
    Group::open(store, "/")
}

pub(in crate::nwb::imp) struct Group(ZarrGroup<Store>);

impl Group {
    fn open(store: ReadableListableStorage, path: &str) -> Result<Self> {
        Ok(Self(ZarrGroup::open(store, path)?))
    }

    pub fn attr(&self, name: &str) -> Result<&Value> {
        self.0
            .attributes()
            .get(name)
            .with_context(|| format!("missing {name} attribute"))
    }

    fn links(&self) -> Result<&[Value]> {
        match self.0.attributes().get("zarr_link") {
            Some(links) => Ok(links.as_array().context("expected a zarr_link list")?),
            None => Ok(&[]),
        }
    }

    fn link(&self, name: &str) -> Result<Option<&Value>> {
        Ok(self
            .links()?
            .iter()
            .find(|link| link["name"].as_str() == Some(name)))
    }

    pub fn member_names(&self) -> Result<Vec<String>> {
        let mut names: Vec<_> = self
            .0
            .children(false)?
            .iter()
            .map(|n| n.name().to_string())
            .collect();
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

    /// Walk the lexical path, checking each component before opening it. A
    /// zarr_link is a soft alias even when its target is in the same store.
    fn resolve(&self, path: &str, stored: bool) -> Result<String> {
        let base = self.0.path().as_str().trim_end_matches('/');
        let mut path = if path.starts_with('/') {
            path.to_owned()
        } else {
            format!("{base}/{path}")
        };
        let store = self.0.storage();
        'resolve: for _ in 0..32 {
            let components = components(&path)?;
            let mut prefix = String::new();
            for (i, name) in components.iter().enumerate() {
                let parent =
                    Self::open(store.clone(), if prefix.is_empty() { "/" } else { &prefix })?;
                prefix.push('/');
                prefix.push_str(name);
                if let Some(link) = parent.link(name)? {
                    ensure!(
                        link["source"].as_str() == Some("."),
                        "{prefix} is an external zarr_link"
                    );
                    ensure!(!stored, "{prefix} is a same-store zarr_link (soft link)");
                    let target = link["path"].as_str().context("zarr_link has no path")?;
                    path = std::iter::once(target.trim_end_matches('/'))
                        .chain(components[i + 1..].iter().copied())
                        .collect::<Vec<_>>()
                        .join("/");
                    if path.is_empty() {
                        path.push('/');
                    }
                    continue 'resolve;
                }
            }
            return Ok(path);
        }
        bail!("too many nested zarr_link entries")
    }

    pub fn group(&self, name: &str) -> Result<Self> {
        Self::open(self.0.storage(), &self.resolve(name, false)?)
    }

    pub fn child_group(&self, name: &str) -> Result<Option<Self>> {
        // An unrelated external link need not prevent listing local series.
        Ok(self.group(name).ok())
    }

    pub fn link_exists(&self, name: &str) -> Result<bool> {
        Ok(self.link(name)?.is_some()
            || self
                .member_names()?
                .iter()
                .any(|n| n == name.trim_start_matches('/')))
    }

    pub fn dataset(&self, name: &str, stored: bool) -> Result<Dataset> {
        let path = self.resolve(name, stored)?;
        let store = self.0.storage();
        let data = match Array::open(store.clone(), &path) {
            Ok(array) => Data::Array(Box::new(array)),
            Err(error) => {
                // hdmf-zarr writes cached specs as a single uncompressed
                // object chunk using numcodecs JSON. Never deserialize pickle.
                let Some(meta) = store.get(&StoreKey::new(format!("{}/.zarray", &path[1..]))?)?
                else {
                    return Err(error.into());
                };
                let meta: Value = serde_json::from_slice(&meta)?;
                if !(meta["dtype"] == "|O"
                    && meta["shape"] == serde_json::json!([1])
                    && meta["chunks"] == serde_json::json!([1])
                    && meta["compressor"].is_null()
                    && meta["filters"]
                        .as_array()
                        .is_some_and(|f| f.len() == 1 && f[0]["id"] == "json2"))
                {
                    return Err(error.into());
                }
                Data::JsonString {
                    store,
                    key: StoreKey::new(format!("{}/0", &path[1..]))?,
                }
            }
        };
        let shape = match &data {
            Data::Array(array) => array
                .shape()
                .iter()
                .map(|&n| usize::try_from(n))
                .collect::<Result<_, _>>()?,
            Data::JsonString { .. } => vec![1],
        };
        Ok(Dataset { data, shape })
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

    pub fn attr(&self, name: &str) -> Result<&Value> {
        self.array()?
            .attributes()
            .get(name)
            .with_context(|| format!("missing {name} attribute"))
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
            return Ok(a.retrieve_array_subset::<Vec<T>>(subset)?);
        }
        macro_rules! convert {
            ($($kind:ident: $ty:ty),*) => { match self.sample_type()? { $(
                SampleType::$kind => a.retrieve_array_subset::<Vec<$ty>>(subset)?
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
            self.shape.is_empty() || (self.shape == [1] && self.attr("zarr_dtype")? == "scalar"),
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
                .retrieve_array_subset::<Vec<Vec<u8>>>(&a.subset_all())?
                .into_iter()
                .map(|s| Ok(String::from_utf8(s)?))
                .collect(),
            Data::Array(a) => Ok(a.retrieve_array_subset::<Vec<String>>(&a.subset_all())?),
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
