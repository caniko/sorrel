use super::SampleType;
use anyhow::{anyhow, ensure, Context, Result};
use hdf5::types::{TypeDescriptor, VarLenAscii, VarLenUnicode};
use hdf5_metno as hdf5;
use std::path::Path;

/// H5Tequal checks the complete layout, including byte order. Do not let
/// HDF5 silently convert an integer, extended-precision or custom float.
pub(super) fn exact_float(dtype: hdf5::Result<hdf5::Datatype>, what: &str) -> Result<()> {
    let dtype = dtype?;
    ensure!(
        dtype == hdf5::Datatype::from_type::<f64>()?
            || dtype == hdf5::Datatype::from_type::<f32>()?,
        "{what} must be stored as native IEEE float32 or float64"
    );
    Ok(())
}

pub(super) fn legacy_string(data: &hdf5::Container) -> Result<String> {
    Ok(data
        .read_scalar::<VarLenUnicode>()
        .map(|s| s.to_string())
        .or_else(|_| data.read_scalar::<VarLenAscii>().map(|s| s.to_string()))?)
}

pub(super) fn sample_type(data: &hdf5::Dataset) -> Result<SampleType> {
    use hdf5::types::{FloatSize, IntSize, TypeDescriptor::*};
    Ok(match data.dtype()?.to_descriptor()? {
        Integer(IntSize::U1) => SampleType::I8,
        Unsigned(IntSize::U1) => SampleType::U8,
        Integer(IntSize::U2) => SampleType::I16,
        Unsigned(IntSize::U2) => SampleType::U16,
        Integer(IntSize::U4) => SampleType::I32,
        Unsigned(IntSize::U4) => SampleType::U32,
        Integer(IntSize::U8) => SampleType::I64,
        Unsigned(IntSize::U8) => SampleType::U64,
        Float(FloatSize::U4) => SampleType::F32,
        Float(FloatSize::U8) => SampleType::F64,
        other => anyhow::bail!("{other} samples are not supported"),
    })
}

pub(super) fn identity(path: &Path, meta: &std::fs::Metadata) -> Result<Vec<u8>> {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("missing UTF-8 NWB filename")?;
    let modified = meta.modified()?.duration_since(std::time::UNIX_EPOCH)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"sorrel-provider-identity-v1/nwb");
    hasher.update(name.as_bytes());
    hasher.update(&meta.len().to_le_bytes());
    hasher.update(&modified.as_secs().to_le_bytes());
    hasher.update(&modified.subsec_nanos().to_le_bytes());
    Ok(hasher.finalize().as_bytes().to_vec())
}

pub(crate) fn read_str(data: &hdf5::Container) -> Result<String> {
    ensure!(data.is_scalar(), "expected a scalar HDF5 string");
    strings(data)?.pop().context("missing HDF5 string")
}

pub(crate) fn strings(data: &hdf5::Container) -> Result<Vec<String>> {
    let dtype = data.dtype()?;
    match dtype.to_descriptor()? {
        TypeDescriptor::VarLenUnicode => data
            .read_raw::<VarLenUnicode>()?
            .iter()
            .map(|s| Ok(std::str::from_utf8(s.as_bytes())?.to_owned()))
            .collect(),
        TypeDescriptor::VarLenAscii => data
            .read_raw::<VarLenAscii>()?
            .iter()
            .map(|s| Ok(std::str::from_utf8(s.as_bytes())?.to_owned()))
            .collect(),
        TypeDescriptor::FixedAscii(width) | TypeDescriptor::FixedUnicode(width) => {
            fixed_strings(data, &dtype, width)
        }
        _ => Err(anyhow!("expected an HDF5 string datatype")),
    }
}

pub(super) fn optional_strings(data: &hdf5::Container) -> Result<Option<Vec<String>>> {
    use hdf5_metno_sys::h5t::{H5Tget_class, H5T_NO_CLASS, H5T_STRING};

    let dtype = data.dtype()?;
    // Query the class, not a full descriptor (some non-string classes have none).
    let class = hdf5::sync::sync(|| unsafe { H5Tget_class(dtype.id()) });
    ensure!(
        class != H5T_NO_CLASS,
        "cannot determine HDF5 datatype class"
    );
    (class == H5T_STRING).then(|| strings(data)).transpose()
}

fn fixed_strings(
    data: &hdf5::Container,
    dtype: &hdf5::Datatype,
    width: usize,
) -> Result<Vec<String>> {
    use hdf5_metno_sys::{
        h5a::H5Aread,
        h5d::H5Dread,
        h5i::{H5Iget_type, H5I_ATTR},
        h5p::H5P_DEFAULT,
        h5s::H5Sget_simple_extent_npoints,
        h5t::{H5T_str_t::H5T_STR_SPACEPAD, H5Tget_strpad},
    };

    // hdf5-metno's fixed strings require a compile-time width. Read with
    // the file datatype as the memory datatype instead (no conversion).
    let space = data.space()?;
    // SAFETY: space is a live HDF5 dataspace, accessed under the HDF5 lock.
    let count =
        hdf5::sync::sync(|| unsafe { hdf5::h5check(H5Sget_simple_extent_npoints(space.id())) })?;
    let count = usize::try_from(count)?;
    if count == 0 {
        return Ok(Vec::new());
    }
    let len = count
        .checked_mul(width)
        .filter(|&n| n != 0)
        .context("invalid HDF5 string size")?;
    let mut bytes = Vec::<u8>::new();
    bytes.try_reserve_exact(len)?;
    bytes.resize(len, 0);
    let space_pad = hdf5::sync::sync(|| {
        // SAFETY: data and dtype are live; bytes holds count * width bytes
        // for the whole dataspace, and the HDF5 lock serializes the calls.
        unsafe {
            let buf = bytes.as_mut_ptr().cast();
            hdf5::h5check(if H5Iget_type(data.id()) == H5I_ATTR {
                H5Aread(data.id(), dtype.id(), buf)
            } else {
                H5Dread(
                    data.id(),
                    dtype.id(),
                    space.id(),
                    space.id(),
                    H5P_DEFAULT,
                    buf,
                )
            })?;
            Ok::<_, hdf5::Error>(H5Tget_strpad(dtype.id()) == H5T_STR_SPACEPAD)
        }
    })?;
    bytes
        .chunks_exact(width)
        .map(|s| {
            let s = std::str::from_utf8(s.split(|&b| b == 0).next().unwrap_or_default())?;
            Ok(if space_pad {
                s.trim_end_matches(' ')
            } else {
                s
            }
            .to_owned())
        })
        .collect()
}

// The journal only seals explicitly opened files. Check each link before
// following it: even a link that resolves back into this file can go via a sidecar.
pub(super) fn resolve_local_path(
    group: &hdf5::Group,
    path: &str,
    include_soft_links: bool,
) -> Result<Option<String>> {
    fn walk(
        group: &hdf5::Group,
        path: &str,
        mut prefix: String,
        include_soft_links: bool,
        active: &mut Vec<String>,
        hops: &mut usize,
    ) -> Result<Option<String>> {
        for component in path.split('/').filter(|c| !matches!(*c, "" | ".")) {
            ensure!(component != "..", "invalid HDF5 object path {path:?}");
            let parent_len = prefix.len();
            prefix.push('/');
            prefix.push_str(component);
            match link(group, &prefix)? {
                None => return Ok(None),
                Some(Link::Hard) => {}
                Some(Link::Soft(target)) => {
                    ensure!(include_soft_links, "{prefix} is an HDF5 soft link");
                    ensure!(
                        !active.contains(&prefix),
                        "HDF5 soft link cycle at {prefix}"
                    );
                    // ponytail: track paths, not object tokens; the cap also
                    // covers cycles reached through different hard aliases.
                    ensure!(*hops < 32, "too many HDF5 soft link hops at {prefix}");
                    *hops += 1;
                    let parent = if target.starts_with('/') {
                        String::new()
                    } else {
                        prefix[..parent_len].to_owned()
                    };
                    // Resolve the target before checking the remaining caller
                    // path: an optional child of a valid alias may be absent.
                    active.push(prefix);
                    let resolved = walk(group, &target, parent, include_soft_links, active, hops);
                    let link = active.pop().expect("active soft link");
                    prefix = resolved?
                        .with_context(|| format!("dangling HDF5 soft link {link} -> {target}"))?;
                }
            }
        }
        Ok(Some(prefix))
    }

    ensure!(
        path.trim_start_matches('/')
            .split('/')
            .all(|c| !matches!(c, "" | "." | "..")),
        "invalid HDF5 object path {path:?}"
    );
    let prefix = if path.starts_with('/') {
        String::new()
    } else {
        // Regular groups are only opened from resolved hard paths.
        group.name().trim_end_matches('/').to_owned()
    };
    let mut hops = 0;
    let resolved = walk(
        group,
        path,
        prefix,
        include_soft_links,
        &mut Vec::new(),
        &mut hops,
    )?;
    Ok(resolved.map(|p| if p.is_empty() { "/".into() } else { p }))
}

enum Link {
    Hard,
    Soft(String),
}

fn link(group: &hdf5::Group, path: &str) -> Result<Option<Link>> {
    use hdf5_metno_sys::{
        h5l::{self, H5L_info2_t, H5L_TYPE_EXTERNAL, H5L_TYPE_HARD, H5L_TYPE_SOFT},
        h5p::H5P_DEFAULT,
    };
    use std::ffi::{c_int, CStr, CString};
    use std::mem::MaybeUninit;

    let name = CString::new(path)?;
    let mut info = MaybeUninit::<H5L_info2_t>::uninit();
    hdf5::sync::sync(|| {
        // SAFETY: IDs and C strings are live under the lock. Read the type
        // as a C int because user-defined links can use other values. For
        // soft links, the active union field gives the link value buffer size.
        unsafe {
            if hdf5::h5check(h5l::H5Lexists(group.id(), name.as_ptr(), H5P_DEFAULT))? == 0 {
                return Ok(None);
            }
            // HDF5 2.x no longer exports H5Lget_info1.
            hdf5::h5check(h5l::H5Lget_info2(
                group.id(),
                name.as_ptr(),
                info.as_mut_ptr(),
                H5P_DEFAULT,
            ))?;
            match info.as_ptr().cast::<c_int>().read() {
                k if k == H5L_TYPE_HARD as c_int => Ok(Some(Link::Hard)),
                k if k == H5L_TYPE_SOFT as c_int => {
                    let size = *info.assume_init().u.val_size();
                    let mut bytes = Vec::new();
                    bytes.try_reserve_exact(size)?;
                    bytes.resize(size, 0u8);
                    hdf5::h5check(h5l::H5Lget_val(
                        group.id(),
                        name.as_ptr(),
                        bytes.as_mut_ptr().cast(),
                        size,
                        H5P_DEFAULT,
                    ))?;
                    let target = CStr::from_bytes_with_nul(&bytes)?.to_str()?;
                    ensure!(!target.is_empty(), "empty HDF5 soft link target");
                    Ok(Some(Link::Soft(target.to_owned())))
                }
                k if k == H5L_TYPE_EXTERNAL as c_int => {
                    anyhow::bail!("{path} is an HDF5 external link")
                }
                _ => anyhow::bail!("{path} is an HDF5 non-hard link"),
            }
        }
    })
    .with_context(|| format!("cannot inspect HDF5 link {path}"))
}

pub(super) fn local_dataset(file: &hdf5::Group, path: &str) -> Result<hdf5::Dataset> {
    let ds = file
        .dataset(path)
        .with_context(|| format!("cannot open HDF5 dataset {path}"))?;
    check_dataset_storage(&ds)?;
    Ok(ds)
}

pub(super) fn local_trace_dataset(group: &hdf5::Group, path: &str) -> Result<hdf5::Dataset> {
    use hdf5::plist::DatasetAccess;
    use hdf5_metno_sys::h5d::H5Dopen2;
    use std::ffi::CString;

    let ds = local_dataset(group, path)?;
    let Some(chunk) = ds.dcpl()?.get_chunk()? else {
        return Ok(ds);
    };
    let channel_chunks = match (chunk.as_slice(), ds.space()?.shape().as_slice()) {
        ([_], [_]) => 1,
        ([_, chunk_channels], [_, channels]) => channels.div_ceil(*chunk_channels),
        _ => return Ok(ds),
    };
    // ponytail: cap at 256 MiB per open trace; use channel-subset reads if a
    // complete time tile no longer fits.
    let tile_bytes = chunk
        .iter()
        .fold(ds.dtype()?.size(), |bytes, &dim| bytes.saturating_mul(dim))
        .saturating_mul(channel_chunks)
        .min(256 * 1024 * 1024);
    let default = ds.dapl()?.get_chunk_cache()?;
    // HDF5 recommends at least ten hash slots per cached chunk. Keep prime
    // sizes to limit collisions and bound the table for very wide traces.
    let target_slots = channel_chunks.saturating_mul(10).min(262_147);
    if tile_bytes <= default.nbytes && target_slots <= default.nslots {
        return Ok(ds);
    }
    let slots = if target_slots <= 12_421 {
        12_421
    } else {
        262_147
    };
    let dapl = DatasetAccess::build()
        .chunk_cache(
            slots.max(default.nslots),
            tile_bytes.max(default.nbytes),
            default.w0,
        )
        .finish()?;
    let name = CString::new(path)?;
    hdf5::sync::sync(|| {
        // HDF5 shares the cache across open handles. Close the probe first.
        drop(ds);
        // SAFETY: the group and DAPL are live under the HDF5 lock; the name
        // is the same checked hard path, and from_id owns the returned ID.
        unsafe { hdf5::from_id(H5Dopen2(group.id(), name.as_ptr(), dapl.id())) }
    })
    .with_context(|| format!("cannot reopen HDF5 trace dataset {path}"))
}

pub(super) fn check_dataset_storage(ds: &hdf5::Dataset) -> Result<()> {
    use hdf5::plist::dataset_create::Layout;
    use hdf5_metno_sys::h5p::H5Pget_external_count;

    let dcpl = ds
        .dcpl()
        .context("cannot inspect dataset creation properties")?;
    ensure!(
        [Layout::Compact, Layout::Contiguous, Layout::Chunked].contains(
            &dcpl
                .get_layout()
                .context("cannot inspect HDF5 dataset layout")?
        ),
        "HDF5 virtual dataset layout"
    );
    // SAFETY: dcpl is a live dataset creation plist and HDF5 is locked.
    let external = hdf5::sync::sync(|| unsafe { hdf5::h5check(H5Pget_external_count(dcpl.id())) })
        .context("cannot inspect HDF5 external raw storage")?;
    ensure!(external == 0, "HDF5 external raw storage");
    Ok(())
}

/// Cached specs may use same-file soft links. Block external links even
/// when they are reached indirectly through a soft link.
pub(super) fn spec_object(group: &hdf5::Group, path: &str) -> Result<hdf5::Location> {
    use hdf5_metno_sys::{
        h5::herr_t,
        h5i::hid_t,
        h5o::H5Oopen,
        h5p::{H5Pcreate, H5Pset_elink_cb, H5P_CLS_LINK_ACCESS},
    };
    use std::ffi::{c_char, c_uint, c_void, CString};

    extern "C" fn deny_external(
        _: *const c_char,
        _: *const c_char,
        _: *const c_char,
        _: *const c_char,
        _: *mut c_uint,
        _: hid_t,
        _: *mut c_void,
    ) -> herr_t {
        -1
    }
    let name = CString::new(path)?;
    hdf5::sync::sync(|| {
        // SAFETY: IDs and the C string are live under the HDF5 lock. The
        // callback ignores its arguments; owned IDs are wrapped immediately.
        unsafe {
            let lapl: hdf5::PropertyList = hdf5::from_id(H5Pcreate(*H5P_CLS_LINK_ACCESS))?;
            hdf5::h5check(H5Pset_elink_cb(
                lapl.id(),
                Some(deny_external),
                std::ptr::null_mut(),
            ))?;
            hdf5::from_id(H5Oopen(group.id(), name.as_ptr(), lapl.id()))
        }
    })
    .with_context(|| format!("cannot open same-file HDF5 spec object {path:?}"))
}

pub(super) fn spec_reference(root: &hdf5::Group, attr: &hdf5::Attribute) -> Result<hdf5::Group> {
    use hdf5_metno_sys::{
        h5a::H5Aread,
        h5p::H5P_DEFAULT,
        h5r::{hobj_ref_t, H5R_type_t::H5R_OBJECT1, H5Rdereference2},
        h5t::{H5Tequal, H5T_STD_REF_OBJ},
    };

    ensure!(attr.ndim() == 0, "HDF5 .specloc must be scalar");
    let dtype = attr.dtype()?;
    hdf5::sync::sync(|| {
        // SAFETY: the scalar old-style object reference has exactly the
        // hobj_ref_t layout. It can only reference objects in the same file.
        unsafe {
            ensure!(
                H5Tequal(dtype.id(), *H5T_STD_REF_OBJ) > 0,
                "HDF5 .specloc must be an object reference"
            );
            let mut reference: hobj_ref_t = 0;
            hdf5::h5check(H5Aread(
                attr.id(),
                *H5T_STD_REF_OBJ,
                (&mut reference as *mut hobj_ref_t).cast(),
            ))?;
            Ok(hdf5::from_id(H5Rdereference2(
                root.id(),
                H5P_DEFAULT,
                H5R_OBJECT1,
                (&reference as *const hobj_ref_t).cast(),
            ))?)
        }
    })
}
