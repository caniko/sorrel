//! Internal storage vocabulary. Nothing here chooses an NWB series or
//! interprets its clock, units, or channel metadata.

#[cfg(feature = "hdf5")]
#[path = "hdf.rs"]
mod hdf;
#[cfg(feature = "hdf5")]
pub(crate) use hdf::{read_str, strings};
#[cfg(all(feature = "dandi", feature = "hdf5"))]
#[path = "http_vfd.rs"]
mod http_vfd;
#[cfg(feature = "zarr")]
#[path = "zarr.rs"]
mod zarr;

use anyhow::{bail, Context, Result};
#[cfg(feature = "hdf5")]
use hdf5_metno as h5;
use std::{ops::Range, path::Path};

pub(super) enum Group {
    #[cfg(feature = "hdf5")]
    Hdf5(h5::Group),
    #[cfg(feature = "zarr")]
    Zarr(zarr::Group),
}

pub(super) enum Dataset {
    #[cfg(feature = "hdf5")]
    Hdf5(h5::Dataset),
    #[cfg(feature = "zarr")]
    Zarr(Box<zarr::Dataset>),
}

pub(super) enum Attr {
    #[cfg(feature = "hdf5")]
    Hdf5(h5::Attribute),
    #[cfg(feature = "zarr")]
    Zarr(serde_json::Value),
}

macro_rules! dispatch {
    ($value:expr, $kind:ident, $d:ident => $hdf:expr, $zarr:expr) => {
        match $value {
            #[cfg(feature = "hdf5")]
            $kind::Hdf5($d) => $hdf,
            #[cfg(feature = "zarr")]
            $kind::Zarr($d) => $zarr,
        }
    };
}

impl Group {
    #[cfg(feature = "dandi")]
    pub fn open_remote(store: crate::dandi::Store) -> Result<Self> {
        match store {
            #[cfg(feature = "hdf5")]
            crate::dandi::Store::Blob(object) => {
                Ok(Self::Hdf5(http_vfd::open(&object)?.group("/")?))
            }
            crate::dandi::Store::Zarr(store) => Ok(Self::Zarr(zarr::open_remote(store)?)),
        }
    }

    pub fn open(path: &Path) -> Result<Self> {
        let open = || -> Result<Self> {
            if path.is_dir() {
                #[cfg(feature = "zarr")]
                return Ok(Self::Zarr(zarr::open(path)?));
                #[cfg(not(feature = "zarr"))]
                bail!("NWB directory stores require building with --features zarr");
            }
            #[cfg(feature = "hdf5")]
            return Ok(Self::Hdf5(h5::File::open(path)?.group("/")?));
            #[cfg(not(feature = "hdf5"))]
            bail!("HDF5 NWB files require building with --features hdf5");
        };
        open().with_context(|| format!("open NWB {}", path.display()))
    }

    pub fn group(&self, path: &str) -> Result<Self> {
        self.optional_group(path)?
            .with_context(|| format!("missing NWB group {path}"))
    }

    pub fn optional_group(&self, path: &str) -> Result<Option<Self>> {
        dispatch!(self, Self, g => {
            if !hdf::hard_link_exists(g, path)? { return Ok(None); }
            Ok(Some(Self::Hdf5(g.group(path).with_context(|| format!("open HDF5 group {path}"))?)))
        }, g.local_path(path)?.map(|path| g.open_group(&path).map(Self::Zarr)).transpose())
    }

    pub fn child_group(&self, name: &str) -> Result<Option<Self>> {
        dispatch!(self, Self, g => {
            if hdf::hard_link_exists(g, name)? && g.loc_type_by_name(name)? == h5::LocationType::Group {
                Ok(Some(Self::Hdf5(g.group(name)?)))
            } else {
                Ok(None)
            }
        }, Ok(g.child_group(name)?.map(Self::Zarr)))
    }

    pub fn spec_group(&self, legacy: bool) -> Result<Option<Self>> {
        dispatch!(self, Self, g => {
            if let Some(Attr::Hdf5(attr)) = if legacy { None } else { hdf_attr(g, ".specloc")? } {
                return Ok(Some(Self::Hdf5(hdf::spec_reference(g, &attr)?)));
            }
            if !g.link_exists("/specifications") { return Ok(None); }
            Ok(Some(Self::Hdf5(hdf::spec_object(g, "/specifications")?.as_group()?)))
        }, {
            let _ = legacy;
            let path = match g.optional_attr(".specloc") {
                Some(attr) => attr.as_str().filter(|s| !s.is_empty() && *s != "/")
                    .context("Zarr .specloc must be a same-store group path")?,
                None if g.local_path("/specifications")?.is_none() => return Ok(None),
                None => "/specifications",
            };
            let group = g.child_group(path)?.context("missing Zarr specifications group")?;
            Ok(Some(Self::Zarr(group)))
        })
    }

    pub fn spec_groups(&self) -> Result<Vec<(String, Self)>> {
        self.member_names()?
            .into_iter()
            .filter_map(|name| {
                let group = dispatch!(self, Self, g => match hdf::spec_object(g, &name) {
                    Ok(obj) => obj.as_group().ok().map(|group| Ok(Self::Hdf5(group))),
                    Err(err) => Some(Err(err)),
                }, g.child_group(&name).map(|group| group.map(Self::Zarr)).transpose());
                group.map(|g| g.map(|g| (name, g)))
            })
            .collect()
    }

    pub fn spec_datasets(&self) -> Result<Vec<(String, Dataset)>> {
        self.member_names()?
            .into_iter()
            .map(|name| {
                let ds = dispatch!(self, Self, g => {
                    let ds = hdf::spec_object(g, &name)?.as_dataset()?;
                    hdf::check_dataset_storage(&ds)?;
                    Ok::<_, anyhow::Error>(Dataset::Hdf5(ds))
                }, {
                    let path = g.local_path(&name)?.context("missing Zarr spec array")?;
                    g.open_dataset(&path, true).map(|d| Dataset::Zarr(Box::new(d)))
                })?;
                Ok((name, ds))
            })
            .collect()
    }

    pub fn member_names(&self) -> Result<Vec<String>> {
        dispatch!(self, Self, g => Ok(g.member_names()?), g.member_names())
    }

    pub fn local_member_exists(&self, path: &str) -> Result<bool> {
        dispatch!(self, Self, g => hdf::hard_link_exists(g, path), Ok(g.local_path(path)?.is_some()))
    }

    pub fn dataset(&self, path: &str) -> Result<Dataset> {
        self.optional_dataset(path)?
            .with_context(|| format!("missing NWB dataset {path}"))
    }

    /// `None` means absent, not a broken link, wrong node type, or open error.
    pub fn optional_dataset(&self, path: &str) -> Result<Option<Dataset>> {
        dispatch!(self, Self, g => {
            if !hdf::hard_link_exists(g, path)? { return Ok(None); }
            hdf::local_dataset(g, path).map(Dataset::Hdf5).map(Some)
        }, g.local_path(path)?.map(|path| g.open_dataset(&path, false).map(|d| Dataset::Zarr(Box::new(d)))
            .with_context(|| format!("open Zarr dataset {path}"))).transpose())
    }

    pub fn optional_stored_dataset(&self, path: &str) -> Result<Option<Dataset>> {
        self.optional_dataset(path)
        .with_context(|| format!("NWB dataset {path} must be stored in this file; open the file that holds the data directly, or pass the recording file via --dat"))
    }

    pub fn stored_dataset(&self, path: &str) -> Result<Dataset> {
        self.optional_stored_dataset(path)?
            .with_context(|| format!("missing NWB dataset {path}"))
    }

    pub fn optional_attr(&self, name: &str) -> Result<Option<Attr>> {
        dispatch!(self, Self, g => hdf_attr(g, name),
            Ok(g.optional_attr(name).cloned().map(Attr::Zarr)))
    }

    pub fn supports_legacy_seal(&self) -> bool {
        dispatch!(self, Self, _g => true, false)
    }

    // Historical lookups only: never use these to choose or load current data.
    pub fn legacy_group(&self, name: &str) -> Result<Option<Self>> {
        dispatch!(self, Self, _g => {
            if !_g.link_exists(name) || _g.loc_type_by_name(name)? != h5::LocationType::Group {
                return Ok(None);
            }
            Ok(Some(Self::Hdf5(_g.group(name)?)))
        }, { let _ = name; Ok(None) })
    }

    pub fn legacy_dataset_rank(&self, name: &str) -> Result<Option<usize>> {
        dispatch!(self, Self, _g => {
            if !_g.link_exists(name) || _g.loc_type_by_name(name)? != h5::LocationType::Dataset {
                return Ok(None);
            }
            Ok(Some(_g.dataset(name)?.ndim()))
        }, { let _ = name; Ok(None) })
    }
}

pub(super) fn local_identity(path: &Path) -> Result<Vec<u8>> {
    let meta = path.metadata()?;
    if meta.is_dir() {
        #[cfg(feature = "zarr")]
        return zarr::identity(path);
        #[cfg(not(feature = "zarr"))]
        bail!("NWB directory stores require building with --features zarr");
    }
    anyhow::ensure!(meta.is_file(), "not a regular NWB file");
    #[cfg(feature = "hdf5")]
    return hdf::identity(path, &meta);
    #[cfg(not(feature = "hdf5"))]
    bail!("HDF5 NWB files require building with --features hdf5");
}

#[cfg(feature = "hdf5")]
fn hdf_attr(location: &h5::Location, name: &str) -> Result<Option<Attr>> {
    if location.attr_names()?.iter().any(|attr| attr == name) {
        Ok(Some(Attr::Hdf5(location.attr(name)?)))
    } else {
        Ok(None)
    }
}

impl Attr {
    pub fn string(&self) -> Result<String> {
        match self {
            #[cfg(feature = "hdf5")]
            Self::Hdf5(a) => hdf::read_str(a),
            #[cfg(feature = "zarr")]
            Self::Zarr(a) => Ok(a
                .as_str()
                .context("expected a string attribute")?
                .to_owned()),
        }
    }

    pub fn legacy_string(&self) -> Result<String> {
        match self {
            #[cfg(feature = "hdf5")]
            Self::Hdf5(a) => hdf::legacy_string(a),
            #[cfg(feature = "zarr")]
            Self::Zarr(_) => bail!("no legacy Zarr seal"),
        }
    }

    pub fn number(&self) -> Result<f64> {
        match self {
            #[cfg(feature = "hdf5")]
            Self::Hdf5(a) => Ok(a.read_scalar()?),
            #[cfg(feature = "zarr")]
            Self::Zarr(a) => zarr::exact_number(a, "Zarr numeric attribute"),
        }
    }

    pub fn exact_number(&self, what: &str) -> Result<f64> {
        match self {
            #[cfg(feature = "hdf5")]
            Self::Hdf5(a) => {
                hdf::exact_float(a.dtype(), what)?;
                self.number()
            }
            #[cfg(feature = "zarr")]
            Self::Zarr(a) => zarr::exact_number(a, what),
        }
    }
}

// The numeric types NWB needs. The backend bounds stay private.
#[cfg(all(feature = "hdf5", feature = "zarr"))]
pub(super) trait Number: h5::H5Type + zarr::Number {}
#[cfg(all(feature = "hdf5", feature = "zarr"))]
impl<T: h5::H5Type + zarr::Number> Number for T {}
#[cfg(all(feature = "hdf5", not(feature = "zarr")))]
use h5::H5Type as Number;
#[cfg(all(feature = "zarr", not(feature = "hdf5")))]
use zarr::Number;

impl Dataset {
    pub fn shape(&self) -> Vec<usize> {
        dispatch!(self, Self, d => d.shape(), d.shape().to_vec())
    }

    pub fn chunk(&self) -> Option<Vec<usize>> {
        dispatch!(self, Self, d => d.chunk(), d.chunk())
    }

    pub fn sample_type(&self) -> Result<SampleType> {
        dispatch!(self, Self, d => hdf::sample_type(d), d.sample_type())
    }

    pub fn exact_float(&self, what: &str) -> Result<()> {
        dispatch!(self, Self, d => hdf::exact_float(d.dtype(), what), d.exact_float(what))
    }

    pub fn attr(&self, name: &str) -> Result<Attr> {
        self.optional_attr(name)?
            .with_context(|| format!("missing {name} attribute"))
    }

    pub fn optional_attr(&self, name: &str) -> Result<Option<Attr>> {
        dispatch!(self, Self, d => hdf_attr(d, name),
            Ok(d.optional_attr(name)?.cloned().map(Attr::Zarr)))
    }

    pub fn read_rows<T: Number>(&self, range: Range<usize>) -> Result<Vec<T>> {
        dispatch!(self, Self, d => match d.ndim() {
            1 => Ok(d.read_slice_1d::<T, _>(range)?.into_raw_vec_and_offset().0),
            2 => Ok(d.read_slice_2d::<T, _>((range, ..))?.into_raw_vec_and_offset().0),
            _ => bail!("expected a rank 1 or 2 NWB dataset"),
        }, d.read_rows(range))
    }

    pub fn read_raw<T: Number>(&self) -> Result<Vec<T>> {
        dispatch!(self, Self, d => Ok(d.read_raw::<T>()?), d.read_raw())
    }

    pub fn read_scalar(&self) -> Result<f64> {
        dispatch!(self, Self, d => Ok(d.read_scalar()?), d.read_number())
    }

    pub fn read_string(&self) -> Result<String> {
        dispatch!(self, Self, d => hdf::read_str(d), d.read_string())
    }

    pub fn strings(&self) -> Result<Vec<String>> {
        dispatch!(self, Self, d => hdf::strings(d), d.strings())
    }

    pub fn optional_strings(&self) -> Result<Option<Vec<String>>> {
        dispatch!(self, Self, d => hdf::optional_strings(d), d.optional_strings())
    }

    pub fn spec_text(&self, legacy: bool) -> Result<String> {
        if legacy {
            dispatch!(self, Self, _d => hdf::legacy_string(_d), Err(anyhow::anyhow!("no legacy Zarr seal")))
        } else {
            self.read_string()
        }
    }

    pub fn reference_path(&self, attribute: &str) -> Result<Option<String>> {
        dispatch!(self, Self, _d => {
            // Keep HDF5's historical canonical table lookup and journal
            // mapping unchanged. Zarr provides the actual object path.
            let _ = attribute;
            Ok(None)
        }, _d.reference_path(attribute))
    }
}

pub(super) enum SampleType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
}

impl SampleType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::I8 => "int8",
            Self::U8 => "uint8",
            Self::I16 => "int16",
            Self::U16 => "uint16",
            Self::I32 => "int32",
            Self::U32 => "uint32",
            Self::I64 => "int64",
            Self::U64 => "uint64",
            Self::F32 => "float32",
            Self::F64 => "float64",
        }
    }
}
