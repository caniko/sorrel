//! Minimal `.npy` v1.0/v2.0 parser sufficient for Kilosort outputs.
//!
//! Supports the dtypes Kilosort emits: `<i64`, `<u32`, `<i32`, `<f32`, `<f64`.
//! Returns the byte offset to the raw data so the caller can mmap-and-cast.

use anyhow::{anyhow, bail, Context, Result};
use std::fs::File;
use std::io::{Read, Seek, Write};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct NpyHeader {
    pub dtype: String,
    pub fortran_order: bool,
    pub shape: Vec<usize>,
    pub data_offset: u64,
}

impl NpyHeader {
    pub fn elem_count(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Numpy dtype tags we accept for the arrays Kilosort and SpikeInterface
/// emit. Backends typically downcast to a single in-memory type (e.g. spike
/// times always land as `u64`), so the decoder takes both `<f4`/`<f8`
/// equivalents and lets the caller resolve the conversion.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NpyDtype {
    F32,
    F64,
    I32,
    U32,
    I64,
    U64,
}

impl NpyDtype {
    /// Parse a numpy `descr` string. Accepts both numpy-canonical forms
    /// (`'<f4'`, `'<i8'`) and the bare aliases that occasionally appear in
    /// Kilosort outputs (`'i4'`, `'<i32'`).
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim() {
            "<f4" | "f4" | "<float32" | "float32" => Self::F32,
            "<f8" | "f8" | "<float64" | "float64" => Self::F64,
            "<i4" | "i4" | "<i32" | "i32" => Self::I32,
            "<u4" | "u4" | "<u32" | "u32" => Self::U32,
            "<i8" | "i8" | "<i64" | "i64" => Self::I64,
            "<u8" | "u8" | "<u64" | "u64" => Self::U64,
            other => bail!("unsupported numpy dtype {other}"),
        })
    }

    pub const fn size_bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::I64 | Self::U64 => 8,
        }
    }

    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }

    pub const fn is_integer(self) -> bool {
        !self.is_float()
    }
}

/// Decode `n` elements of `dtype` from `bytes`, applying `f` to each chunk.
/// Returns `Err` if the buffer is short.
fn decode_chunks<T>(
    bytes: &[u8],
    n: usize,
    dtype: NpyDtype,
    f: impl FnMut(&[u8]) -> T,
) -> Result<Vec<T>> {
    let elem = dtype.size_bytes();
    let need = n * elem;
    if bytes.len() < need {
        bail!("array truncated: have {} bytes, need {need}", bytes.len());
    }
    Ok(bytes[..need].chunks_exact(elem).map(f).collect())
}

pub fn read_header(path: &Path) -> Result<NpyHeader> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut magic = [0u8; 6];
    f.read_exact(&mut magic)?;
    if &magic != b"\x93NUMPY" {
        bail!("not a .npy file: {}", path.display());
    }
    let mut ver = [0u8; 2];
    f.read_exact(&mut ver)?;

    let header_len = match ver[0] {
        1 => {
            let mut buf = [0u8; 2];
            f.read_exact(&mut buf)?;
            u16::from_le_bytes(buf) as u64
        }
        2 | 3 => {
            let mut buf = [0u8; 4];
            f.read_exact(&mut buf)?;
            u32::from_le_bytes(buf) as u64
        }
        v => bail!("unsupported .npy version {v}"),
    };

    let mut header = vec![0u8; header_len as usize];
    f.read_exact(&mut header)?;
    let header = std::str::from_utf8(&header)?.trim().trim_end_matches('\0').trim();

    let dtype = extract(header, "'descr':")?;
    let fortran_order = extract(header, "'fortran_order':")?;
    let shape_str = extract(header, "'shape':")?;

    let dtype = dtype.trim().trim_matches('\'').to_string();
    let fortran_order = fortran_order.trim() == "True";
    let shape = parse_shape(&shape_str)?;

    let data_offset = f.stream_position()?;
    Ok(NpyHeader {
        dtype,
        fortran_order,
        shape,
        data_offset,
    })
}

fn extract(header: &str, key: &str) -> Result<String> {
    let i = header
        .find(key)
        .ok_or_else(|| anyhow!("missing key {key} in header"))?;
    let rest = &header[i + key.len()..];
    // Value ends at the next top-level comma.
    let mut depth = 0i32;
    let mut end = rest.len();
    for (idx, c) in rest.char_indices() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                end = idx;
                break;
            }
            _ => {}
        }
    }
    Ok(rest[..end].trim().to_string())
}

fn parse_shape(s: &str) -> Result<Vec<usize>> {
    let s = s.trim().trim_start_matches('(').trim_end_matches(')');
    s.split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(|p| p.parse::<usize>().map_err(Into::into))
        .collect()
}

/// Open a `.npy` file, parse its header, and return the (header, raw bytes
/// after the header) pair. The byte slice owns its data so callers don't have
/// to keep an mmap alive — use the lower-level [`read_header`] + [`memmap2`]
/// when zero-copy access matters.
fn read_payload(path: &Path) -> Result<(NpyHeader, Vec<u8>)> {
    let header = read_header(path)?;
    if header.fortran_order {
        bail!(
            "fortran-order arrays unsupported (file: {})",
            path.display()
        );
    }
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    f.seek(std::io::SeekFrom::Start(header.data_offset))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    Ok((header, buf))
}

/// Read a 1-D array of `f32`. Accepts `<f4` and `<f8` (downcast).
pub fn read_1d_f32(path: &Path) -> Result<Vec<f32>> {
    let (h, bytes) = read_payload(path)?;
    if h.shape.len() != 1 {
        bail!("expected 1-D array, got shape {:?}", h.shape);
    }
    decode_f32(&h, &bytes, h.elem_count())
}

/// Read an N-D array of `f32` as a flat row-major buffer plus the original
/// shape. Phy's `pc_features.npy` is `(n_spikes, n_pcs, n_channels)` so
/// the existing 1-D / 2-D readers don't fit; this returns the shape verbatim
/// and lets the caller decide how to index.
pub fn read_npy_f32_flat(path: &Path) -> Result<(Vec<f32>, Vec<usize>)> {
    let (h, bytes) = read_payload(path)?;
    let n: usize = h.shape.iter().product();
    let v = decode_f32(&h, &bytes, n)?;
    Ok((v, h.shape))
}

/// Same idea for `u32`-shaped integer arrays of any rank.
pub fn read_npy_u32_flat(path: &Path) -> Result<(Vec<u32>, Vec<usize>)> {
    let (h, bytes) = read_payload(path)?;
    let n: usize = h.shape.iter().product();
    let v = decode_u32(&h, &bytes, n)?;
    Ok((v, h.shape))
}

/// Read a 2-D array of `f32` as a flat row-major buffer plus the (rows, cols)
/// shape. Accepts `<f4` and `<f8`.
pub fn read_2d_f32(path: &Path) -> Result<(Vec<f32>, (usize, usize))> {
    let (h, bytes) = read_payload(path)?;
    if h.shape.len() != 2 {
        bail!("expected 2-D array, got shape {:?}", h.shape);
    }
    let rows = h.shape[0];
    let cols = h.shape[1];
    let v = decode_f32(&h, &bytes, rows * cols)?;
    Ok((v, (rows, cols)))
}

/// Read a 1-D array of unsigned 32-bit ints. Accepts `<i4`/`<u4`/`<i8`/`<u8`
/// (downcast where the value fits).
pub fn read_1d_u32(path: &Path) -> Result<Vec<u32>> {
    let (h, bytes) = read_payload(path)?;
    if h.shape.len() != 1 {
        bail!("expected 1-D array, got shape {:?}", h.shape);
    }
    decode_u32(&h, &bytes, h.elem_count())
}

fn decode_f32(h: &NpyHeader, bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    let dtype = NpyDtype::parse(&h.dtype)?;
    match dtype {
        NpyDtype::F32 => decode_chunks(bytes, n, dtype, |c| {
            f32::from_le_bytes(c.try_into().unwrap())
        }),
        NpyDtype::F64 => decode_chunks(bytes, n, dtype, |c| {
            f64::from_le_bytes(c.try_into().unwrap()) as f32
        }),
        d => bail!("unsupported float dtype {d:?}"),
    }
}

/// Write a 1-D `u32` array as an NPY 1.0 file, with `descr='<u4'`. Atomic
/// when `path`'s directory allows rename: writes to `<path>.tmp` first and
/// renames on success.
pub fn write_1d_u32_atomic(path: &Path, data: &[u32]) -> Result<()> {
    let tmp = with_tmp_suffix(path);
    write_npy_1d(&tmp, "<u4", data.len(), |f| {
        for &v in data {
            f.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    })?;
    std::fs::rename(&tmp, path)
        .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

/// Write a 1-D `i32` array as an NPY 1.0 file, with `descr='<i4'`.
pub fn write_1d_i32_atomic(path: &Path, data: &[i32]) -> Result<()> {
    let tmp = with_tmp_suffix(path);
    write_npy_1d(&tmp, "<i4", data.len(), |f| {
        for &v in data {
            f.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    })?;
    std::fs::rename(&tmp, path)
        .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

fn with_tmp_suffix(path: &Path) -> std::path::PathBuf {
    let mut p = path.as_os_str().to_owned();
    p.push(".tmp");
    p.into()
}

/// Shared driver for the 1-D writers above. Writes header + payload then
/// fsync's so the rename in the caller is durable.
fn write_npy_1d<F: FnOnce(&mut File) -> std::io::Result<()>>(
    path: &Path,
    descr: &str,
    n: usize,
    write_payload: F,
) -> Result<()> {
    let dict = format!(
        "{{'descr': '{descr}', 'fortran_order': False, 'shape': ({n},), }}"
    );
    // NPY 1.0 prelude: \x93NUMPY (6) + version (2) + header_len (2) + dict + \n.
    // Pad the dict so the prelude is a multiple of 64 bytes, matching the
    // numpy convention (and what our reader's tests expect for parity).
    let prelude_len = 6 + 2 + 2 + dict.len() + 1;
    let pad = (64 - (prelude_len % 64)) % 64;
    let mut header = dict.into_bytes();
    header.extend(std::iter::repeat(b' ').take(pad));
    header.push(b'\n');
    let header_len = u16::try_from(header.len())
        .context("npy v1 header longer than u16::MAX — switch to v2")?;

    let mut f = File::create(path).with_context(|| format!("create {}", path.display()))?;
    f.write_all(b"\x93NUMPY")?;
    f.write_all(&[1u8, 0u8])?;
    f.write_all(&header_len.to_le_bytes())?;
    f.write_all(&header)?;
    write_payload(&mut f)?;
    f.sync_all()
        .with_context(|| format!("fsync {}", path.display()))?;
    Ok(())
}

fn decode_u32(h: &NpyHeader, bytes: &[u8], n: usize) -> Result<Vec<u32>> {
    let dtype = NpyDtype::parse(&h.dtype)?;
    match dtype {
        NpyDtype::I32 | NpyDtype::U32 => decode_chunks(bytes, n, dtype, |c| {
            u32::from_le_bytes(c.try_into().unwrap())
        }),
        NpyDtype::I64 | NpyDtype::U64 => decode_chunks(bytes, n, dtype, |c| {
            u64::from_le_bytes(c.try_into().unwrap()) as u32
        }),
        d => bail!("unsupported integer dtype {d:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Write a minimal NPY 1.0 file with the supplied header dict and raw data.
    fn write_npy_v1(path: &Path, descr: &str, shape: &[usize], data: &[u8]) {
        let shape_str = if shape.len() == 1 {
            format!("({},)", shape[0])
        } else {
            let inner: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            format!("({})", inner.join(", "))
        };
        let dict = format!(
            "{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_str}, }}"
        );
        // Pad header so total prelude is a multiple of 64 (NPY convention,
        // not strictly required by our parser but matches real files).
        let prelude_len = 6 + 2 + 2 + dict.len() + 1;
        let pad = (64 - (prelude_len % 64)) % 64;
        let mut header = dict.into_bytes();
        header.extend(std::iter::repeat(b' ').take(pad));
        header.push(b'\n');
        let header_len = header.len() as u16;

        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(b"\x93NUMPY").unwrap();
        f.write_all(&[1u8, 0u8]).unwrap();
        f.write_all(&header_len.to_le_bytes()).unwrap();
        f.write_all(&header).unwrap();
        f.write_all(data).unwrap();
    }

    #[test]
    fn parse_1d_i64_header() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.npy");
        let data: Vec<u8> = (0u64..5).flat_map(|v| v.to_le_bytes()).collect();
        write_npy_v1(&p, "<i64", &[5], &data);

        let h = read_header(&p).unwrap();
        assert_eq!(h.dtype, "<i64");
        assert!(!h.fortran_order);
        assert_eq!(h.shape, vec![5]);
        assert_eq!(h.elem_count(), 5);
    }

    #[test]
    fn parse_2d_shape() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("b.npy");
        write_npy_v1(&p, "<u32", &[2, 3], &[0u8; 24]);
        let h = read_header(&p).unwrap();
        assert_eq!(h.shape, vec![2, 3]);
        assert_eq!(h.elem_count(), 6);
    }

    #[test]
    fn rejects_non_npy_files() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("c.npy");
        std::fs::write(&p, b"not a numpy file at all").unwrap();
        assert!(read_header(&p).is_err());
    }

    #[test]
    fn data_offset_points_past_header() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("d.npy");
        let data: Vec<u8> = (0u64..3).flat_map(|v| v.to_le_bytes()).collect();
        write_npy_v1(&p, "<i64", &[3], &data);

        let h = read_header(&p).unwrap();
        let raw = std::fs::read(&p).unwrap();
        let payload = &raw[h.data_offset as usize..];
        assert_eq!(payload, data.as_slice());
    }

    #[test]
    fn write_then_read_u32_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("u32.npy");
        let data: Vec<u32> = vec![0, 1, 2, 1_000_000, u32::MAX, 17];
        write_1d_u32_atomic(&p, &data).unwrap();

        let read_back = read_1d_u32(&p).unwrap();
        assert_eq!(read_back, data);
    }

    #[test]
    fn write_then_read_i32_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("i32.npy");
        let data: Vec<i32> = vec![-1_000_000, -1, 0, 1, i32::MIN, i32::MAX];
        let as_u32: Vec<u32> = data.iter().map(|&v| v as u32).collect();
        write_1d_i32_atomic(&p, &data).unwrap();

        // i32 round-trips through u32 read since both are 32-bit LE.
        let h = read_header(&p).unwrap();
        assert_eq!(h.dtype, "<i4");
        assert_eq!(h.shape, vec![data.len()]);
        let read_back = read_1d_u32(&p).unwrap();
        assert_eq!(read_back, as_u32);
    }

    #[test]
    fn writer_atomicity_no_tmp_after_success() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("atomic.npy");
        write_1d_u32_atomic(&p, &[1, 2, 3]).unwrap();
        let names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| Some(e.ok()?.file_name().to_string_lossy().into_owned()))
            .collect();
        assert!(names.iter().any(|n| n == "atomic.npy"));
        assert!(!names.iter().any(|n| n.ends_with(".tmp")));
    }

    #[test]
    fn writer_overwrites_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("over.npy");
        write_1d_u32_atomic(&p, &[1, 2, 3]).unwrap();
        write_1d_u32_atomic(&p, &[100, 200]).unwrap();
        let read_back = read_1d_u32(&p).unwrap();
        assert_eq!(read_back, vec![100, 200]);
    }

    #[test]
    fn rejects_truncated_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("trunc.npy");
        // Header claims 100 elements but only 3 bytes of data follow.
        write_npy_v1(&p, "<f4", &[100], &[0u8, 0, 0]);
        assert!(read_1d_f32(&p).is_err());
    }

    #[test]
    fn rejects_unsupported_dtype() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("c8.npy");
        write_npy_v1(&p, "<c8", &[1], &[0u8; 8]);
        assert!(read_1d_f32(&p).is_err());
    }

    #[test]
    fn rejects_wrong_rank_for_1d_reader() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("rank.npy");
        write_npy_v1(&p, "<f4", &[2, 3], &[0u8; 24]);
        assert!(read_1d_f32(&p).is_err());
    }

    #[test]
    fn read_2d_f32_returns_correct_shape_and_values() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a2d.npy");
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        write_npy_v1(&p, "<f4", &[2, 3], &bytes);

        let (flat, (rows, cols)) = read_2d_f32(&p).unwrap();
        assert_eq!((rows, cols), (2, 3));
        assert_eq!(flat, values);
    }

    #[test]
    fn read_npy_f32_flat_handles_3d_array() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a3d.npy");
        let values: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        write_npy_v1(&p, "<f4", &[2, 3, 4], &bytes);

        let (flat, shape) = read_npy_f32_flat(&p).unwrap();
        assert_eq!(shape, vec![2, 3, 4]);
        assert_eq!(flat, values);
    }

    #[test]
    fn f64_input_downcasts_to_f32_on_read() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f64.npy");
        let values_f64: Vec<f64> = vec![1.5, -2.25, 3.125];
        let bytes: Vec<u8> = values_f64.iter().flat_map(|v| v.to_le_bytes()).collect();
        write_npy_v1(&p, "<f8", &[3], &bytes);

        let read_back = read_1d_f32(&p).unwrap();
        for (got, want) in read_back.iter().zip(values_f64.iter()) {
            assert!((got - *want as f32).abs() < 1e-6);
        }
    }

    #[test]
    fn rejects_unsupported_npy_version() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("v9.npy");
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(b"\x93NUMPY").unwrap();
        f.write_all(&[9u8, 9u8]).unwrap();
        f.write_all(&[0u8; 64]).unwrap();
        assert!(read_header(&p).is_err());
    }

    #[test]
    fn rejects_nonexistent_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("missing.npy");
        assert!(read_1d_f32(&p).is_err());
    }

    #[test]
    fn parses_header_with_extra_whitespace() {
        // Real-world numpy headers vary; ours should tolerate leading/
        // trailing whitespace inside the dict.
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("ws.npy");
        // Hand-craft a dict with extra spaces.
        let dict =
            "{   'descr':   '<f4'  ,  'fortran_order':  False  ,  'shape':  (3,)  ,  }";
        let prelude_len = 6 + 2 + 2 + dict.len() + 1;
        let pad = (64 - (prelude_len % 64)) % 64;
        let mut header = dict.as_bytes().to_vec();
        header.extend(std::iter::repeat(b' ').take(pad));
        header.push(b'\n');
        let header_len = header.len() as u16;
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(b"\x93NUMPY").unwrap();
        f.write_all(&[1u8, 0u8]).unwrap();
        f.write_all(&header_len.to_le_bytes()).unwrap();
        f.write_all(&header).unwrap();
        let data: Vec<u8> = vec![1.0_f32, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        f.write_all(&data).unwrap();
        drop(f);

        let v = read_1d_f32(&p).unwrap();
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn rejects_fortran_order_array() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f.npy");
        // Manually build a fortran-order header.
        let dict = "{'descr': '<f4', 'fortran_order': True, 'shape': (4,), }";
        let prelude_len = 6 + 2 + 2 + dict.len() + 1;
        let pad = (64 - (prelude_len % 64)) % 64;
        let mut header = dict.as_bytes().to_vec();
        header.extend(std::iter::repeat(b' ').take(pad));
        header.push(b'\n');
        let header_len = header.len() as u16;
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(b"\x93NUMPY").unwrap();
        f.write_all(&[1u8, 0u8]).unwrap();
        f.write_all(&header_len.to_le_bytes()).unwrap();
        f.write_all(&header).unwrap();
        f.write_all(&[0u8; 16]).unwrap();
        drop(f);

        // The 1-D reader uses the payload helper which rejects fortran order.
        assert!(read_1d_f32(&p).is_err());
    }

    #[test]
    fn read_1d_f32_handles_zero_length_array() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("empty.npy");
        write_npy_v1(&p, "<f4", &[0], &[]);
        let v = read_1d_f32(&p).unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn write_then_read_zero_length_array() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("empty.npy");
        write_1d_u32_atomic(&p, &[]).unwrap();
        let v = read_1d_u32(&p).unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn writer_produces_file_we_can_re_parse_header_for() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.npy");
        write_1d_u32_atomic(&p, &[7, 8, 9]).unwrap();
        let h = read_header(&p).unwrap();
        assert_eq!(h.dtype, "<u4");
        assert_eq!(h.shape, vec![3]);
        assert!(!h.fortran_order);
    }
}
