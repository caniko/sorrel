//! Mountainsort `.mda` (multi-dimensional array) reader.
//!
//! MDA is a tiny dense-array container Mountainsort/Mountainview use for
//! both raw recordings and `firings.mda` spike tables. Header is six 32-bit
//! little-endian fields: `[dtype, dtype_bytes, ndim, dim_0, dim_1, ...]`.
//!
//! dtype codes (from `mountainlab-js/mountainsort/sortprocesses/mdaio`):
//! ```text
//! -2 = uint8
//! -3 = float32
//! -4 = int16
//! -5 = int32
//! -6 = uint16
//! -7 = float64
//! -8 = uint32
//! ```
//! We surface the on-disk dtype + shape and the byte offset to the payload.
//! Memory mapping is left to the caller.

use crate::provider::TraceDtype;
use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Parsed MDA header. `data_offset` is the byte where the row-major payload
/// begins.
#[derive(Debug, Clone)]
pub struct MdaHeader {
    pub dtype_code: i32,
    pub dtype_bytes: u32,
    pub shape: Vec<u32>,
    pub data_offset: u64,
}

impl MdaHeader {
    pub fn elem_count(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    pub fn trace_dtype(&self) -> Result<TraceDtype> {
        match self.dtype_code {
            -3 => Ok(TraceDtype::F32),
            -4 => Ok(TraceDtype::I16),
            -5 => Ok(TraceDtype::I32),
            -6 => Ok(TraceDtype::U16),
            other => bail!("mda dtype code {other} has no TraceDtype mapping"),
        }
    }

    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mut buf = [0u8; 4];

        f.read_exact(&mut buf)?;
        let dtype_code = i32::from_le_bytes(buf);
        f.read_exact(&mut buf)?;
        let dtype_bytes = u32::from_le_bytes(buf);
        f.read_exact(&mut buf)?;
        let ndim = i32::from_le_bytes(buf);

        // MDA "version 2" headers signal `ndim < 0`: |ndim| dimensions follow,
        // each as i64. The "version 1" headers we target use i32 dimensions.
        if ndim < 0 {
            // V2: dims as i64.
            let nd = (-ndim) as usize;
            let mut shape = Vec::with_capacity(nd);
            let mut buf8 = [0u8; 8];
            for _ in 0..nd {
                f.read_exact(&mut buf8)?;
                let d = i64::from_le_bytes(buf8);
                if d < 0 {
                    bail!("mda v2 negative dimension");
                }
                shape.push(d as u32);
            }
            let data_offset = 12 + 8 * nd as u64;
            Ok(Self {
                dtype_code,
                dtype_bytes,
                shape,
                data_offset,
            })
        } else {
            let nd = ndim as usize;
            if nd == 0 || nd > 16 {
                bail!("mda: implausible ndim {nd}");
            }
            let mut shape = Vec::with_capacity(nd);
            for _ in 0..nd {
                f.read_exact(&mut buf)?;
                let d = i32::from_le_bytes(buf);
                if d < 0 {
                    bail!("mda v1 negative dimension");
                }
                shape.push(d as u32);
            }
            let data_offset = 12 + 4 * nd as u64;
            Ok(Self {
                dtype_code,
                dtype_bytes,
                shape,
                data_offset,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_v1_header(dtype_code: i32, dtype_bytes: u32, dims: &[i32]) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&dtype_code.to_le_bytes()).unwrap();
        f.write_all(&dtype_bytes.to_le_bytes()).unwrap();
        f.write_all(&(dims.len() as i32).to_le_bytes()).unwrap();
        for &d in dims {
            f.write_all(&d.to_le_bytes()).unwrap();
        }
        f
    }

    #[test]
    fn parses_v1_int16_2d_header() {
        let f = write_v1_header(-4, 2, &[384, 30_000]);
        let h = MdaHeader::read(f.path()).unwrap();
        assert_eq!(h.dtype_code, -4);
        assert_eq!(h.dtype_bytes, 2);
        assert_eq!(h.shape, vec![384, 30_000]);
        assert_eq!(h.trace_dtype().unwrap(), TraceDtype::I16);
        assert_eq!(h.data_offset, 12 + 4 * 2);
    }

    #[test]
    fn parses_v1_float32_3d_header() {
        let f = write_v1_header(-3, 4, &[64, 100, 10]);
        let h = MdaHeader::read(f.path()).unwrap();
        assert_eq!(h.shape, vec![64, 100, 10]);
        assert_eq!(h.trace_dtype().unwrap(), TraceDtype::F32);
    }

    #[test]
    fn elem_count_multiplies_dims() {
        let f = write_v1_header(-4, 2, &[3, 4, 5]);
        let h = MdaHeader::read(f.path()).unwrap();
        assert_eq!(h.elem_count(), 60);
    }
}
