//! Minimal `.npy` v1.0/v2.0 parser sufficient for Kilosort outputs.
//!
//! Supports the dtypes Kilosort emits: `<i64`, `<u32`, `<i32`, `<f32`, `<f64`.
//! Returns the byte offset to the raw data so the caller can mmap-and-cast.

use anyhow::{anyhow, bail, Context, Result};
use std::fs::File;
use std::io::{Read, Seek};
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
