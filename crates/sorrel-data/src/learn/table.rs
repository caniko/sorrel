//! Columnar `f32` learn tables: a small, boring format a Python or Julia
//! script can read without any library.
//!
//! A table is a pair of files, `<name>.json` and `<name>.f32`. The JSON header
//! carries the extractor version, the column names, the session ids and the
//! row count; the data file is little-endian `f32`, row-major, exactly
//! `n_rows × columns.len()` values:
//!
//! ```text
//! {"extractor_version":"learn-v1","columns":["sid_idx","root"],
//!  "n_rows":2,"sids":["abc"]}   +   2 × 2 × 4 little-endian bytes
//! ```
//!
//! Column names are the schema: `sid_idx` indexes [`Table::sids`], `root` and
//! other id columns are integers stored as `f32`, features are prefixed `f_`,
//! supervised targets `y_` and sample weights `w_`. A missing measurement is
//! `NaN`.
//!
//! Writes are atomic in the same directory (temporary file, sync, rename), so
//! an interrupted extraction leaves either the previous shard or the new one,
//! never a half file. The header is written last: a directory with a readable
//! header necessarily has its data.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};

/// Suffix of the JSON header.
pub const HEADER_SUFFIX: &str = ".json";
/// Suffix of the raw little-endian `f32` data.
pub const DATA_SUFFIX: &str = ".f32";

/// The JSON header of a saved table. `n_rows` is stored twice: here, and
/// implicitly by the data file's length.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TableHeader {
    /// Identifies the feature extractor that produced the columns.
    pub extractor_version: String,
    /// Target definition version, when the table carries (or joins) `y_*`
    /// columns. `None` for an input-only feature shard.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_version: Option<String>,
    /// Column names in data order.
    pub columns: Vec<String>,
    /// Number of rows in the data file.
    pub n_rows: usize,
    /// Session ids, indexed by the `sid_idx` column.
    pub sids: Vec<String>,
    /// Optional integer encoding of a column, e.g. `{"f_method": {"whole": 0}}`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub codes: BTreeMap<String, BTreeMap<String, i64>>,
    /// Optional human-readable notes, e.g. an overlap measure's definition.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub notes: BTreeMap<String, String>,
}

/// One in-memory table: a shared header plus row-major data.
///
/// Equality compares the data bitwise, so a `NaN` cell equals itself: two
/// tables saved and reloaded from the same extraction must compare equal.
#[derive(Clone, Debug)]
pub struct Table {
    /// Identifies the extractor that produced the columns.
    pub extractor_version: String,
    /// Target definition version when the table carries `y_*` columns.
    pub target_version: Option<String>,
    /// Column names in data order; unique and non-empty.
    pub columns: Vec<String>,
    /// Session ids, indexed by `sid_idx` values.
    pub sids: Vec<String>,
    /// `n_rows × columns.len()` values, row-major.
    pub data: Vec<f32>,
    /// Integer encodings of any categorical columns.
    pub codes: BTreeMap<String, BTreeMap<String, i64>>,
    /// Free-form notes written to the header.
    pub notes: BTreeMap<String, String>,
}

impl PartialEq for Table {
    fn eq(&self, other: &Self) -> bool {
        self.extractor_version == other.extractor_version
            && self.target_version == other.target_version
            && self.columns == other.columns
            && self.sids == other.sids
            && self.codes == other.codes
            && self.notes == other.notes
            && self.data.len() == other.data.len()
            && self
                .data
                .iter()
                .zip(&other.data)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}

impl Table {
    /// An empty table with `columns`. Sids are added later, once the session
    /// order of the concatenated shards is known.
    pub fn new(extractor_version: impl Into<String>, columns: Vec<String>) -> Self {
        Self {
            extractor_version: extractor_version.into(),
            target_version: None,
            columns,
            sids: Vec::new(),
            data: Vec::new(),
            codes: BTreeMap::new(),
            notes: BTreeMap::new(),
        }
    }

    /// Number of rows, derived from the data length.
    pub fn n_rows(&self) -> usize {
        match self.columns.len() {
            0 => 0,
            n => self.data.len() / n,
        }
    }

    /// Index of `name` in [`Table::columns`].
    pub fn column(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c == name)
    }

    /// One value, or `NaN` when the row or column is out of range.
    pub fn value(&self, row: usize, column: usize) -> f32 {
        let stride = self.columns.len();
        if row >= self.n_rows() || column >= stride {
            return f32::NAN;
        }
        self.data[row * stride + column]
    }

    /// One value by column name, or `NaN` when absent.
    pub fn get(&self, row: usize, name: &str) -> f32 {
        self.column(name).map_or(f32::NAN, |c| self.value(row, c))
    }

    /// One row, or an empty slice when `row` is out of range.
    pub fn row(&self, row: usize) -> &[f32] {
        let stride = self.columns.len();
        if row >= self.n_rows() || stride == 0 {
            return &[];
        }
        &self.data[row * stride..(row + 1) * stride]
    }

    /// Append one row. The length must match the columns exactly.
    pub fn push_row(&mut self, row: &[f32]) -> Result<()> {
        ensure!(
            row.len() == self.columns.len(),
            "row has {} values for {} columns",
            row.len(),
            self.columns.len()
        );
        self.data.extend_from_slice(row);
        Ok(())
    }

    /// Append a row from `f64` values, converting (non-finite becomes `NaN`).
    pub fn push_row_f64(&mut self, row: &[f64]) -> Result<()> {
        let row: Vec<f32> = row
            .iter()
            .map(|&v| if v.is_finite() { v as f32 } else { f32::NAN })
            .collect();
        self.push_row(&row)
    }

    /// Rows where `keep[i]` is true, with every column copied.
    pub fn filter_rows(&self, keep: &[bool]) -> Result<Table> {
        let n = self.n_rows();
        ensure!(
            keep.len() == n,
            "filter mask has {} flags for {n} rows",
            keep.len()
        );
        let stride = self.columns.len();
        let mut out = self.clone();
        out.data.clear();
        out.data
            .reserve(keep.iter().filter(|&&k| k).count() * stride);
        for (i, &k) in keep.iter().enumerate() {
            if k {
                out.data
                    .extend_from_slice(&self.data[i * stride..(i + 1) * stride]);
            }
        }
        Ok(out)
    }

    /// Project onto `names`, in the given order.
    pub fn select_columns(&self, names: &[String]) -> Result<Table> {
        let mut indices = Vec::with_capacity(names.len());
        for name in names {
            indices.push(
                self.column(name)
                    .with_context(|| format!("table has no column {name:?}"))?,
            );
        }
        let n = self.n_rows();
        let mut out = Table::new(self.extractor_version.clone(), names.to_vec());
        out.target_version = self.target_version.clone();
        out.sids = self.sids.clone();
        out.codes = self.codes.clone();
        out.notes = self.notes.clone();
        out.data.reserve(n * names.len());
        for i in 0..n {
            for &c in &indices {
                out.data.push(self.data[i * self.columns.len() + c]);
            }
        }
        Ok(out)
    }

    /// The column values as a fresh vector (an inherent `values` cannot return
    /// a slice because the data is row-major).
    pub fn column_values(&self, name: &str) -> Result<Vec<f32>> {
        let c = self
            .column(name)
            .with_context(|| format!("table has no column {name:?}"))?;
        let stride = self.columns.len();
        Ok((0..self.n_rows())
            .map(|i| self.data[i * stride + c])
            .collect())
    }

    /// Header for this table, validating the shape first.
    pub fn header(&self) -> Result<TableHeader> {
        ensure!(!self.columns.is_empty(), "table has no columns");
        let n = self.n_rows();
        ensure!(
            self.data.len() == n * self.columns.len(),
            "table data has {} values, not {n} × {}",
            self.data.len(),
            self.columns.len()
        );
        Ok(TableHeader {
            extractor_version: self.extractor_version.clone(),
            target_version: self.target_version.clone(),
            columns: self.columns.clone(),
            n_rows: n,
            sids: self.sids.clone(),
            codes: self.codes.clone(),
            notes: self.notes.clone(),
        })
    }

    /// Write `<dir>/<name>.json` and `<dir>/<name>.f32` atomically (the
    /// header last).
    pub fn save(&self, dir: &Path, name: &str) -> Result<()> {
        let header = self.header()?;
        fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        let (header_path, data_path) = paths(dir, name);
        let mut bytes = Vec::with_capacity(self.data.len() * 4);
        for v in &self.data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        write_atomic(&data_path, &bytes)?;
        let json = serde_json::to_vec_pretty(&header)?;
        write_atomic(&header_path, &json)
    }

    /// Read `<dir>/<name>.json` and `<dir>/<name>.f32`, validating that they
    /// agree.
    pub fn load(dir: &Path, name: &str) -> Result<Table> {
        let (header_path, data_path) = paths(dir, name);
        let text = fs::read(&header_path)
            .with_context(|| format!("read table header {}", header_path.display()))?;
        let header: TableHeader = serde_json::from_slice(&text)
            .with_context(|| format!("parse table header {}", header_path.display()))?;
        let bytes = fs::read(&data_path)
            .with_context(|| format!("read table data {}", data_path.display()))?;
        Self::from_parts(header, &bytes)
    }

    /// Read a table if it exists; `None` when the header is absent.
    pub fn load_optional(dir: &Path, name: &str) -> Result<Option<Table>> {
        let (header_path, _) = paths(dir, name);
        if !header_path.is_file() {
            return Ok(None);
        }
        Self::load(dir, name).map(Some)
    }

    /// Whether a saved shard was produced by the same extractor.
    pub fn version_matches(dir: &Path, name: &str, version: &str) -> bool {
        let (header_path, _) = paths(dir, name);
        let Ok(text) = fs::read(&header_path) else {
            return false;
        };
        serde_json::from_slice::<TableHeader>(&text).is_ok_and(|h| h.extractor_version == version)
    }

    /// Whether a shard is complete and current: its header names the expected
    /// extractor and target versions, and its data file is exactly the size
    /// the header promises. A truncated or missing `.f32` is not "ready".
    pub fn integrity(
        dir: &Path,
        name: &str,
        extractor_version: &str,
        target_version: Option<&str>,
    ) -> bool {
        let (header_path, data_path) = paths(dir, name);
        let Ok(text) = fs::read(&header_path) else {
            return false;
        };
        let Ok(header) = serde_json::from_slice::<TableHeader>(&text) else {
            return false;
        };
        if header.extractor_version != extractor_version
            || header.target_version.as_deref() != target_version
        {
            return false;
        }
        let Some(columns) = header.columns.len().checked_mul(header.n_rows) else {
            return false;
        };
        let Some(expected) = columns.checked_mul(4) else {
            return false;
        };
        fs::metadata(&data_path).is_ok_and(|m| m.len() == expected as u64)
    }

    fn from_parts(header: TableHeader, bytes: &[u8]) -> Result<Table> {
        let stride = header.columns.len();
        ensure!(!header.columns.is_empty(), "table header has no columns");
        ensure!(
            bytes.len() == header.n_rows * stride * 4,
            "table data is {} bytes, expected {} × {stride} × 4",
            bytes.len(),
            header.n_rows
        );
        let mut data = Vec::with_capacity(header.n_rows * stride);
        for i in 0..header.n_rows * stride {
            let chunk = &bytes[i * 4..i * 4 + 4];
            data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(Table {
            extractor_version: header.extractor_version,
            target_version: header.target_version,
            columns: header.columns,
            sids: header.sids,
            data,
            codes: header.codes,
            notes: header.notes,
        })
    }
}

/// `(<dir>/<name>.json, <dir>/<name>.f32)`.
pub fn paths(dir: &Path, name: &str) -> (PathBuf, PathBuf) {
    (
        dir.join(format!("{name}{HEADER_SUFFIX}")),
        dir.join(format!("{name}{DATA_SUFFIX}")),
    )
}

/// Write `bytes` to `path` through a temporary file in the same directory, so
/// an interrupted write never leaves a truncated file at the destination.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(dir)?;
    let name = path
        .file_name()
        .context("table file has no name")?
        .to_string_lossy();
    let tmp = dir.join(format!(".{name}.{}.tmp", std::process::id()));
    if let Err(err) = (|| -> Result<()> {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        Ok(())
    })() {
        let _ = fs::remove_file(&tmp);
        return Err(err);
    }
    fs::rename(&tmp, path).inspect_err(|_| {
        let _ = fs::remove_file(&tmp);
    })?;
    Ok(())
}

/// Append `from`'s rows onto `to` after remapping `from`'s `sid_idx` values
/// through `sid_map`. Columns must match exactly.
pub fn append_remapped(to: &mut Table, from: &Table, sid_map: &[f32]) -> Result<()> {
    ensure!(
        to.columns == from.columns,
        "cannot append tables with different columns"
    );
    let stride = from.columns.len();
    let sid = from
        .column("sid_idx")
        .context("shard table has no sid_idx column")?;
    let mut row = vec![0.0; stride];
    for i in 0..from.n_rows() {
        row.copy_from_slice(from.row(i));
        let old = row[sid];
        ensure!(
            old.is_finite() && old >= 0.0 && (old as usize) < sid_map.len(),
            "shard sid_idx {old} is outside the session list"
        );
        row[sid] = sid_map[old as usize];
        to.push_row(&row)?;
    }
    Ok(())
}

/// Parse a column name as an integer id, refusing non-finite or fractional
/// values.
pub fn id_value(v: f32) -> Result<u32> {
    if !v.is_finite() || v < 0.0 || v.fract() != 0.0 || v > u32::MAX as f32 {
        bail!("{v} is not a non-negative integer id");
    }
    Ok(v as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> Table {
        let mut t = Table::new(
            "learn-v1",
            ["sid_idx", "root", "f_a", "y_b"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
        );
        t.sids = vec!["s0".into(), "s1".into()];
        t.push_row(&[0.0, 3.0, 1.5, f32::NAN]).unwrap();
        t.push_row(&[1.0, 4.0, -2.0, 1.0]).unwrap();
        t
    }

    #[test]
    fn roundtrip_preserves_every_byte_and_nan() {
        let dir = tempfile::tempdir().unwrap();
        let t = table();
        t.save(dir.path(), "roots").unwrap();
        let got = Table::load(dir.path(), "roots").unwrap();
        assert_eq!(got, t);
        assert_eq!(got.n_rows(), 2);
        assert!(got.value(0, 3).is_nan());
        assert_eq!(got.get(1, "f_a"), -2.0);
        assert!(got.get(9, "f_a").is_nan());
        assert!(Table::version_matches(dir.path(), "roots", "learn-v1"));
        assert!(!Table::version_matches(dir.path(), "roots", "learn-v2"));
        assert!(Table::load_optional(dir.path(), "absent")
            .unwrap()
            .is_none());
    }

    #[test]
    fn filtering_and_projection_keep_the_schema() {
        let t = table();
        let kept = t.filter_rows(&[false, true]).unwrap();
        assert_eq!(kept.n_rows(), 1);
        assert_eq!(kept.sids, t.sids);
        assert_eq!(kept.row(0), &[1.0, 4.0, -2.0, 1.0]);
        let projected = kept
            .select_columns(&["root".to_owned(), "f_a".to_owned()])
            .unwrap();
        assert_eq!(projected.columns, vec!["root", "f_a"]);
        assert_eq!(projected.data, vec![4.0, -2.0]);
        assert_eq!(kept.column_values("f_a").unwrap(), vec![-2.0]);
        assert!(kept.select_columns(&["nope".to_owned()]).is_err());
    }

    #[test]
    fn malformed_files_are_refused_not_panicked_on() {
        let dir = tempfile::tempdir().unwrap();
        let t = table();
        t.save(dir.path(), "t").unwrap();
        let (header, data) = paths(dir.path(), "t");
        let mut bytes = fs::read(&data).unwrap();
        bytes.truncate(7);
        fs::write(&data, &bytes).unwrap();
        assert!(Table::load(dir.path(), "t").is_err());
        fs::write(&data, vec![0u8; 32]).unwrap();
        fs::write(&header, b"{\"nonsense\":true}").unwrap();
        assert!(Table::load(dir.path(), "t").is_err());
        assert!(!Table::version_matches(dir.path(), "t", "learn-v1"));
    }

    #[test]
    fn append_remaps_session_indices() {
        let mut shard = Table::new("learn-v1", vec!["sid_idx".into(), "root".into()]);
        shard.sids = vec!["s7".into()];
        shard.push_row(&[0.0, 5.0]).unwrap();
        let mut all = Table::new("learn-v1", vec!["sid_idx".into(), "root".into()]);
        append_remapped(&mut all, &shard, &[3.0]).unwrap();
        assert_eq!(all.row(0), &[3.0, 5.0]);
        assert_eq!(id_value(3.0).unwrap(), 3);
        assert!(id_value(1.5).is_err());
        assert!(id_value(f32::NAN).is_err());
    }
}
