//! Two-part session shards: cheap-to-reuse input features and retargetable
//! truth labels.
//!
//! Input-only feature extraction (proposers, PC features, per-spike distances)
//! costs minutes per session; attaching truth costs seconds. Targets will keep
//! changing, so a session directory stores two independently keyed parts:
//!
//! * `features/`: every `f_*` column of `roots`, `candidates`, `pairs` and
//!   `spikes`, plus a sidecar with everything target computation needs and no
//!   proposer does — per-spike root/local index, time, amplitude, PC space,
//!   candidate memberships as deflate-compressed bitsets over root-local
//!   indices, spike-row globals and the validated NPY row map. Keyed by
//!   `(EXTRACTOR_VERSION, ExtractOpts, input manifest)`.
//! * `targets/`: every `y_*` column of those tables plus the complete `units`
//!   and `cand_units` tables (their features are membership-shaped and depend
//!   on the truth). Keyed by `(TARGET_VERSION, truth hash, features key)`.
//!
//! [`save_features`] runs before any truth file is opened. [`retarget`] loads
//! a features shard, attaches a truth and writes a targets shard without
//! opening the recording, the sorter output or a proposer.

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use flate2::read::DeflateDecoder;
use flate2::write::DeflateEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};

use super::features::{
    self, cand_unit_columns, candidate_columns, pair_columns, root_columns, spike_columns,
    unit_columns, ExtractData, SessionTables, Truth,
};
use super::table::{paths as table_paths, Table};
use super::{EXTRACTOR_VERSION, TARGET_VERSION};
use sorrel_io::SampleIndex;

/// Directory under one session's shard holding input-only features.
pub const FEATURES_DIR: &str = "features";
/// Directory under one session's shard holding truth-derived targets.
pub const TARGETS_DIR: &str = "targets";
/// Sidecar inside [`FEATURES_DIR`].
pub const SIDE_FILE: &str = "side.bin";
/// Header inside either part.
pub const META_FILE: &str = "meta.json";

const FEATURES_FORMAT: u32 = 1;
const TARGETS_FORMAT: u32 = 1;
const SIDE_MAGIC: &[u8; 8] = b"SRLFEAT1";

/// Input-only tables stored as `<name>.json`/`.f32` in [`FEATURES_DIR`].
pub const INPUT_TABLES: [&str; 4] = ["roots", "candidates", "pairs", "spikes"];
/// Truth-derived tables stored in [`TARGETS_DIR`].
pub const TARGET_TABLES: [&str; 6] = [
    "roots",
    "candidates",
    "pairs",
    "units",
    "cand_units",
    "spikes",
];

/// Everything the feature half is keyed by.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureKey {
    /// [`EXTRACTOR_VERSION`] at extraction time.
    pub extractor_version: String,
    /// `ExtractOpts::max_spikes_per_root`.
    pub max_spikes_per_root: usize,
    /// `ExtractOpts::seed`.
    pub seed: u64,
    /// Blake3 of the session's immutable input NPYs (`bench::input_hash`).
    ///
    /// Kept byte-for-byte identical to the key used before the manifest
    /// existed, so a shard from that build still matches it.
    pub input_fingerprint: String,
    /// Content identity of every file in the session input, when known.
    ///
    /// `None` only for feature shards written before this field existed.
    /// Compatibility rule: such a shard is accepted when all legacy fields
    /// above still match (the caller logs a warning), because those three
    /// NPYs were the whole key then. Every new shard records the manifest,
    /// and any manifest difference forces re-extraction. [`FeatureKey::id`]
    /// appends the manifest only when present, so legacy target shards keep
    /// their old `features_id`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest: Option<InputManifest>,
}

/// Content identity of every input file a feature extraction consumed.
///
/// Files are sorted by name, so the manifest and its id do not depend on
/// directory order. Large files are identified by size plus a digest of
/// their first and last MiB: the bench's raw recordings and full PC arrays
/// are large, hashing them in full on every resume would dominate the check,
/// and a re-sorted or truncated recording always changes its size or an
/// edge. Small files (params.py, spike/template NPYs, positions, ...) are
/// hashed in full. Modification times are deliberately excluded: a bench
/// copy legitimately receives a fresh mtime while keeping the same bytes.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputManifest {
    /// Every input file the extraction consumes, sorted by name.
    pub files: Vec<InputFile>,
}

/// One file's entry in an [`InputManifest`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputFile {
    /// File name only (the input directory is flat).
    pub name: String,
    /// Size in bytes; also covers a truncated recording.
    pub size: u64,
    /// `true` when only the first and last MiB were hashed.
    pub sampled: bool,
    /// Blake3 hex digest of the content (or of the sampled edges).
    pub digest: String,
}

impl InputManifest {
    /// Stable content id, folded into [`FeatureKey::id`].
    pub fn id(&self) -> String {
        let mut h = blake3::Hasher::new();
        for file in &self.files {
            h.update(file.name.as_bytes());
            h.update(&[0]);
            h.update(&file.size.to_le_bytes());
            h.update(&[u8::from(file.sampled)]);
            h.update(file.digest.as_bytes());
            h.update(&[0]);
        }
        h.finalize().to_hex().to_string()
    }
}

impl FeatureKey {
    /// Stable content id used by the target key.
    pub fn id(&self) -> String {
        let mut text = format!(
            "{}\n{}\n{}\n{}",
            self.extractor_version, self.max_spikes_per_root, self.seed, self.input_fingerprint
        );
        if let Some(manifest) = &self.manifest {
            text.push('\n');
            text.push_str(&manifest.id());
        }
        blake3::hash(text.as_bytes()).to_hex().to_string()
    }

    fn current(&self) -> bool {
        self.extractor_version == EXTRACTOR_VERSION
    }

    /// Whether the stored key found on disk describes the same extraction as
    /// `self`. A stored key without a manifest is a pre-manifest shard: it is
    /// compatible when the legacy fields match.
    fn compatible(&self, stored: &Self) -> bool {
        self.extractor_version == stored.extractor_version
            && self.max_spikes_per_root == stored.max_spikes_per_root
            && self.seed == stored.seed
            && self.input_fingerprint == stored.input_fingerprint
            && match (&self.manifest, &stored.manifest) {
                (Some(requested), Some(stored)) => requested == stored,
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (None, None) => true,
            }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct FeaturesMeta {
    format: u32,
    key: FeatureKey,
    features_id: String,
    n_spikes: usize,
    n_roots: usize,
    n_candidates: usize,
    n_pair_roots: usize,
    n_spike_rows: usize,
    sr_bits: u32,
    total_samples: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct TargetsMeta {
    format: u32,
    target_version: String,
    features_id: String,
    truth_hash: String,
}

/// Column names of a full table, in canonical order. `None` for a name that is
/// not a learn table.
pub fn full_columns(name: &str) -> Option<Vec<String>> {
    Some(match name {
        "roots" => root_columns(),
        "candidates" => candidate_columns(),
        "pairs" => pair_columns(),
        "units" => unit_columns(),
        "cand_units" => cand_unit_columns(),
        "spikes" => spike_columns(),
        _ => return None,
    })
}

fn is_id_column(name: &str) -> bool {
    !name.starts_with("f_") && !name.starts_with("y_") && !name.starts_with("w_")
}

/// Project to id + `f_*` columns (the input-only half).
fn feature_view(table: &Table) -> Table {
    let names: Vec<String> = table
        .columns
        .iter()
        .filter(|c| is_id_column(c) || c.starts_with("f_"))
        .cloned()
        .collect();
    let mut out = table
        .select_columns(&names)
        .expect("projection columns exist");
    retain_owned_metadata(&mut out);
    out
}

/// Project to id + `y_*` columns (the truth-derived half).
fn target_view(table: &Table) -> Table {
    let names: Vec<String> = table
        .columns
        .iter()
        .filter(|c| is_id_column(c) || c.starts_with("y_"))
        .cloned()
        .collect();
    let mut out = table
        .select_columns(&names)
        .expect("projection columns exist");
    retain_owned_metadata(&mut out);
    out
}

/// Header metadata belongs to the columns it documents: a feature shard keeps
/// only metadata for its id/`f_*` columns, a target shard only for its
/// id/`y_*` columns. Target metadata is then regenerated at labeling, so a
/// target-version change can never be frozen into an old feature shard.
fn retain_owned_metadata(table: &mut Table) {
    let columns = table.columns.clone();
    table.codes.retain(|key, _| columns.contains(key));
    table.notes.retain(|key, _| columns.contains(key));
}

/// Expand a projected table back to the canonical full schema, filling absent
/// cells with `NaN`.
fn expand(name: &str, projected: &Table) -> Result<Table> {
    let columns = full_columns(name).with_context(|| format!("unknown table {name:?}"))?;
    let mut out = Table::new(EXTRACTOR_VERSION, columns.clone());
    out.target_version = projected.target_version.clone();
    out.sids = projected.sids.clone();
    out.codes = projected.codes.clone();
    out.notes = projected.notes.clone();
    let mut row = vec![f32::NAN; columns.len()];
    for r in 0..projected.n_rows() {
        row.fill(f32::NAN);
        for (i, column) in projected.columns.iter().enumerate() {
            if let Some(j) = out.column(column) {
                row[j] = projected.value(r, i);
            }
        }
        out.push_row(&row)?;
    }
    Ok(out)
}

fn empty_units() -> (Table, Table) {
    (
        Table::new(EXTRACTOR_VERSION, unit_columns()),
        Table::new(EXTRACTOR_VERSION, cand_unit_columns()),
    )
}

/// Persist the input-only half. Must be called before any truth is loaded.
pub fn save_features(dir: &Path, key: &FeatureKey, data: &ExtractData) -> Result<()> {
    ensure!(key.current(), "feature key has a stale extractor version");
    let t = &data.tables;
    for (name, table) in [
        ("roots", &t.roots),
        ("candidates", &t.candidates),
        ("pairs", &t.pairs),
        ("spikes", &t.spikes),
    ] {
        feature_view(table).save(dir, name)?;
    }
    let side = encode_side(data)?;
    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&side)?;
    let compressed = encoder.finish()?;
    fs::create_dir_all(dir)?;
    // Temp file + rename, never truncate-in-place: an existing hard link to
    // side.bin must keep its bytes when this name is rewritten.
    super::table::write_atomic(&dir.join(SIDE_FILE), &compressed)
        .with_context(|| format!("write {}", dir.join(SIDE_FILE).display()))?;
    let meta = FeaturesMeta {
        format: FEATURES_FORMAT,
        key: key.clone(),
        features_id: key.id(),
        n_spikes: data.root_of_spike.len(),
        n_roots: data.root_sizes.len(),
        n_candidates: data.candidate_members.len(),
        n_pair_roots: data.pair_roots.len(),
        n_spike_rows: data.spike_rows.len(),
        sr_bits: data.sr.to_bits(),
        total_samples: data.total_samples,
    };
    write_json(&dir.join(META_FILE), &meta)
}

/// The key stored in a features shard (used by `--retarget`, which must not
/// touch the input to reconstruct it).
pub fn read_feature_key(dir: &Path) -> Result<FeatureKey> {
    let meta: FeaturesMeta = read_json(&dir.join(META_FILE))?;
    ensure!(
        meta.format == FEATURES_FORMAT,
        "features shard has format {}, expected {FEATURES_FORMAT}",
        meta.format
    );
    Ok(meta.key)
}

/// Whether a features shard is complete, current, and keyed like `key`.
///
/// A shard written before [`FeatureKey::manifest`] existed is accepted only
/// when its legacy key fields still match; callers log that fallback. New
/// shards must match the full key, manifest included.
pub fn features_match(dir: &Path, key: &FeatureKey) -> bool {
    if !key.current() {
        return false;
    }
    let Ok(meta) = read_json::<FeaturesMeta>(&dir.join(META_FILE)) else {
        return false;
    };
    if meta.format != FEATURES_FORMAT
        || !key.compatible(&meta.key)
        || meta.features_id != meta.key.id()
        || !dir.join(SIDE_FILE).is_file()
    {
        return false;
    }
    INPUT_TABLES
        .iter()
        .all(|name| Table::integrity(dir, name, EXTRACTOR_VERSION, None))
}

/// Load the input-only half and rebuild everything [`features::label`] needs.
pub fn load_features(dir: &Path, key: &FeatureKey) -> Result<ExtractData> {
    let meta: FeaturesMeta = read_json(&dir.join(META_FILE))
        .with_context(|| format!("read {}", dir.join(META_FILE).display()))?;
    ensure!(
        meta.format == FEATURES_FORMAT,
        "features shard has format {}, expected {FEATURES_FORMAT}",
        meta.format
    );
    ensure!(
        meta.key.compatible(key),
        "features shard was extracted with different options or input; re-extract"
    );
    ensure!(
        features_match(dir, key),
        "features shard is incomplete or stale; re-extract"
    );
    let compressed = fs::read(dir.join(SIDE_FILE))
        .with_context(|| format!("read {}", dir.join(SIDE_FILE).display()))?;
    let mut decoder = DeflateDecoder::new(&compressed[..]);
    let mut bytes = Vec::new();
    decoder
        .read_to_end(&mut bytes)
        .context("decompress features sidecar")?;
    let side = decode_side(&bytes, &meta)?;

    let roots = expand("roots", &Table::load(dir, "roots")?)?;
    let candidates = expand("candidates", &Table::load(dir, "candidates")?)?;
    let pairs = expand("pairs", &Table::load(dir, "pairs")?)?;
    let spikes = expand("spikes", &Table::load(dir, "spikes")?)?;
    let (units, cand_units) = empty_units();
    let mut centers = vec![None; meta.n_roots];
    // Recomputed exactly as `features::extract` does.
    for (r, feats) in side.root_feats.iter().enumerate() {
        centers[r] = feats
            .as_ref()
            .map(|(f, d)| features::robust_center_scale_dims(f, *d));
    }
    Ok(ExtractData {
        tables: SessionTables {
            roots,
            candidates,
            pairs,
            units,
            cand_units,
            spikes,
        },
        root_of_spike: side.root_of_spike,
        local_index: side.local_index,
        time_of_spike: side.time_of_spike,
        amp_of_spike: side.amp_of_spike,
        root_feats: side.root_feats,
        root_center: centers,
        root_sizes: side.root_sizes,
        root_amp_sum: side.root_amp_sum,
        root_amp_count: side.root_amp_count,
        candidate_members: side.candidate_members,
        spike_rows: side.spike_rows,
        pair_roots: side.pair_roots,
        row_map: side.row_map,
        session_roots: side.session_roots,
        sr: f32::from_bits(meta.sr_bits),
        total_samples: meta.total_samples,
    })
}

/// Attach `truth` to loaded features and persist the target half.
pub fn retarget(
    dir: &Path,
    features_id: &str,
    truth_hash: &str,
    data: &mut ExtractData,
    truth: &Truth,
) -> Result<()> {
    features::label(data, truth)?;
    save_targets(dir, features_id, truth_hash, data)
}

/// Persist the truth-derived half.
pub fn save_targets(
    dir: &Path,
    features_id: &str,
    truth_hash: &str,
    data: &ExtractData,
) -> Result<()> {
    let t = &data.tables;
    for (name, table) in [
        ("roots", &t.roots),
        ("candidates", &t.candidates),
        ("pairs", &t.pairs),
        ("units", &t.units),
        ("cand_units", &t.cand_units),
        ("spikes", &t.spikes),
    ] {
        let mut out = match name {
            "units" | "cand_units" => table.clone(),
            _ => target_view(table),
        };
        out.extractor_version = EXTRACTOR_VERSION.to_owned();
        out.target_version = Some(TARGET_VERSION.to_owned());
        out.save(dir, name)?;
    }
    let meta = TargetsMeta {
        format: TARGETS_FORMAT,
        target_version: TARGET_VERSION.to_owned(),
        features_id: features_id.to_owned(),
        truth_hash: truth_hash.to_owned(),
    };
    write_json(&dir.join(META_FILE), &meta)
}

/// Whether a targets shard is complete and keyed to `features_id` + truth.
pub fn targets_match(dir: &Path, features_id: &str, truth_hash: &str) -> bool {
    let Ok(meta) = read_json::<TargetsMeta>(&dir.join(META_FILE)) else {
        return false;
    };
    if meta.format != TARGETS_FORMAT
        || meta.target_version != TARGET_VERSION
        || meta.features_id != features_id
        || meta.truth_hash != truth_hash
    {
        return false;
    }
    TARGET_TABLES
        .iter()
        .all(|name| Table::integrity(dir, name, EXTRACTOR_VERSION, Some(TARGET_VERSION)))
}

/// Load the truth-derived half, fully expanded to the output schema.
pub fn load_targets(dir: &Path, features_id: &str, truth_hash: &str) -> Result<SessionTables> {
    ensure!(
        targets_match(dir, features_id, truth_hash),
        "targets shard is missing, stale or keyed to different features/truth"
    );
    let roots = expand("roots", &Table::load(dir, "roots")?)?;
    let candidates = expand("candidates", &Table::load(dir, "candidates")?)?;
    let pairs = expand("pairs", &Table::load(dir, "pairs")?)?;
    let units = Table::load(dir, "units")?;
    let cand_units = Table::load(dir, "cand_units")?;
    let spikes = expand("spikes", &Table::load(dir, "spikes")?)?;
    Ok(SessionTables {
        roots,
        candidates,
        pairs,
        units,
        cand_units,
        spikes,
    })
}

/// Join the two halves of one table. Rows must already be aligned; ids are
/// cross-checked so a mismatch is an error, never a silent shift.
pub fn join(name: &str, features: &Table, targets: &Table) -> Result<Table> {
    let columns = full_columns(name).with_context(|| format!("unknown table {name:?}"))?;
    ensure!(
        features.n_rows() == targets.n_rows(),
        "{name}: {} feature rows but {} target rows",
        features.n_rows(),
        targets.n_rows()
    );
    for id in features.columns.iter().filter(|c| is_id_column(c)) {
        let Some(f) = features.column(id) else {
            continue;
        };
        let Some(t) = targets.column(id) else {
            continue;
        };
        for row in 0..features.n_rows() {
            let (a, b) = (features.value(row, f), targets.value(row, t));
            ensure!(
                (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
                "{name}: id {id:?} differs at row {row} ({a} vs {b}); shards are not aligned"
            );
        }
    }
    let mut out = Table::new(EXTRACTOR_VERSION, columns.clone());
    out.target_version = Some(TARGET_VERSION.to_owned());
    out.sids = features.sids.clone();
    // Metadata is partitioned by column ownership: a feature shard owns only
    // its id/f_* entries (old shards may still carry stale y_* copies) and
    // the target shard owns the y_* entries regenerated at labeling. Targets
    // are authoritative, so a retarget can never advertise new values under
    // old encodings.
    out.codes = features
        .codes
        .iter()
        .filter(|(key, _)| !key.starts_with("y_"))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    out.notes = features
        .notes
        .iter()
        .filter(|(key, _)| !key.starts_with("y_"))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    for (key, value) in &targets.codes {
        out.codes.insert(key.clone(), value.clone());
    }
    for (key, value) in &targets.notes {
        out.notes.insert(key.clone(), value.clone());
    }
    let mut row = vec![f32::NAN; columns.len()];
    for r in 0..features.n_rows() {
        row.fill(f32::NAN);
        for table in [features, targets] {
            for (i, column) in table.columns.iter().enumerate() {
                if let Some(j) = out.column(column) {
                    row[j] = table.value(r, i);
                }
            }
        }
        out.push_row(&row)?;
    }
    Ok(out)
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    super::table::write_atomic(path, &bytes)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

struct Side {
    root_of_spike: Vec<u32>,
    local_index: Vec<u32>,
    session_roots: Vec<u32>,
    row_map: Vec<Option<u32>>,
    time_of_spike: Vec<SampleIndex>,
    amp_of_spike: Vec<f32>,
    root_sizes: Vec<usize>,
    root_amp_sum: Vec<f64>,
    root_amp_count: Vec<usize>,
    root_feats: Vec<Option<(Vec<f32>, usize)>>,
    candidate_members: Vec<Vec<u32>>,
    spike_rows: Vec<u32>,
    pair_roots: Vec<(u32, u32)>,
}

fn encode_side(data: &ExtractData) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(SIDE_MAGIC);
    put_u32(&mut out, data.root_of_spike.len() as u32);
    put_u32(&mut out, data.root_sizes.len() as u32);
    put_u32(&mut out, data.root_feats.len() as u32);
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); data.root_sizes.len()];
    for (g, &r) in data.root_of_spike.iter().enumerate() {
        ensure!(
            (r as usize) < buckets.len(),
            "root_of_spike[{g}] = {r} is outside {} roots",
            buckets.len()
        );
        buckets[r as usize].push(g as u32);
    }
    for &v in &data.root_of_spike {
        put_u32(&mut out, v);
    }
    for &v in &data.local_index {
        put_u32(&mut out, v);
    }
    for &v in &data.session_roots {
        put_u32(&mut out, v);
    }
    for v in &data.row_map {
        put_u32(&mut out, v.unwrap_or(u32::MAX));
    }
    for v in &data.time_of_spike {
        out.extend_from_slice(&v.0.to_le_bytes());
    }
    for &v in &data.amp_of_spike {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for &v in &data.root_sizes {
        put_u64(&mut out, v as u64);
    }
    for &v in &data.root_amp_sum {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for &v in &data.root_amp_count {
        put_u64(&mut out, v as u64);
    }
    for feats in &data.root_feats {
        match feats {
            None => put_u32(&mut out, 0),
            Some((values, d)) => {
                put_u32(&mut out, *d as u32);
                for &v in values {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
    // Candidate memberships: a bitset over root-local indices, per candidate.
    ensure!(
        data.candidate_members.len() == data.tables.candidates.n_rows(),
        "candidate side data is out of sync while saving features"
    );
    for (row, members) in data.candidate_members.iter().enumerate() {
        let root = data.tables.candidates.get(row, "root");
        ensure!(
            root.is_finite() && root >= 0.0,
            "candidate {row} has no root"
        );
        let root = root as usize;
        ensure!(root < buckets.len(), "candidate {row} root is out of range");
        let n = buckets[root].len();
        let nbytes = n.div_ceil(8);
        let mut bits = vec![0u8; nbytes];
        let root_of = &data.root_of_spike;
        for &g in members {
            ensure!(
                root_of[g as usize] as usize == root,
                "candidate {row} mixes roots"
            );
            let local = data.local_index[g as usize] as usize;
            ensure!(local < n, "candidate {row} local index is out of range");
            bits[local / 8] |= 1 << (local % 8);
        }
        put_u32(&mut out, root as u32);
        put_u32(&mut out, nbytes as u32);
        out.extend_from_slice(&bits);
    }
    for &v in &data.spike_rows {
        put_u32(&mut out, v);
    }
    for &(a, b) in &data.pair_roots {
        put_u32(&mut out, a);
        put_u32(&mut out, b);
    }
    Ok(out)
}

fn decode_side(bytes: &[u8], meta: &FeaturesMeta) -> Result<Side> {
    let mut c = Cursor { buf: bytes, pos: 0 };
    ensure!(
        c.take(SIDE_MAGIC.len())? == SIDE_MAGIC,
        "features sidecar has bad magic"
    );
    let n_spikes = c.u32()? as usize;
    let n_roots = c.u32()? as usize;
    let n_feat_roots = c.u32()? as usize;
    ensure!(
        n_spikes == meta.n_spikes && n_roots == meta.n_roots && n_feat_roots == n_roots,
        "features sidecar counts disagree with its header"
    );
    let take_u32 = |n: usize, c: &mut Cursor<'_>| -> Result<Vec<u32>> {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(c.u32()?);
        }
        Ok(v)
    };
    let root_of_spike = take_u32(n_spikes, &mut c)?;
    let local_index = take_u32(n_spikes, &mut c)?;
    let session_roots = take_u32(n_spikes, &mut c)?;
    let row_map = take_u32(n_spikes, &mut c)?;
    let mut time_of_spike = Vec::with_capacity(n_spikes);
    for _ in 0..n_spikes {
        time_of_spike.push(SampleIndex(c.u64()?));
    }
    let mut amp_of_spike = Vec::with_capacity(n_spikes);
    for _ in 0..n_spikes {
        amp_of_spike.push(c.f32()?);
    }
    let mut root_sizes = Vec::with_capacity(n_roots);
    for _ in 0..n_roots {
        root_sizes.push(c.u64()? as usize);
    }
    let mut root_amp_sum = Vec::with_capacity(n_roots);
    for _ in 0..n_roots {
        root_amp_sum.push(c.f64()?);
    }
    let mut root_amp_count = Vec::with_capacity(n_roots);
    for _ in 0..n_roots {
        root_amp_count.push(c.u64()? as usize);
    }
    let mut root_feats = Vec::with_capacity(n_roots);
    for &size in &root_sizes {
        let d = c.u32()? as usize;
        if d == 0 {
            root_feats.push(None);
        } else {
            let n = size
                .checked_mul(d)
                .context("root feature count overflows")?;
            let mut values = Vec::with_capacity(n);
            for _ in 0..n {
                values.push(c.f32()?);
            }
            root_feats.push(Some((values, d)));
        }
    }
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); n_roots];
    for (g, &r) in root_of_spike.iter().enumerate() {
        ensure!((r as usize) < n_roots, "sidecar root id out of range");
        buckets[r as usize].push(g as u32);
    }
    let mut candidate_members = Vec::with_capacity(meta.n_candidates);
    for _ in 0..meta.n_candidates {
        let root = c.u32()? as usize;
        ensure!(root < n_roots, "sidecar candidate root out of range");
        let nbytes = c.u32()? as usize;
        ensure!(
            nbytes == buckets[root].len().div_ceil(8),
            "sidecar candidate bitset has the wrong width"
        );
        let bits = c.take(nbytes)?;
        let mut members = Vec::new();
        for (byte_index, &byte) in bits.iter().enumerate() {
            for bit in 0..8 {
                if byte & (1 << bit) != 0 {
                    let local = byte_index * 8 + bit;
                    ensure!(
                        local < buckets[root].len(),
                        "sidecar candidate bit is out of range"
                    );
                    members.push(buckets[root][local]);
                }
            }
        }
        ensure!(!members.is_empty(), "sidecar has an empty candidate");
        candidate_members.push(members);
    }
    let spike_rows = take_u32(meta.n_spike_rows, &mut c)?;
    let mut pair_roots = Vec::with_capacity(meta.n_pair_roots);
    for _ in 0..meta.n_pair_roots {
        pair_roots.push((c.u32()?, c.u32()?));
    }
    ensure!(c.pos == bytes.len(), "features sidecar has trailing bytes");
    Ok(Side {
        root_of_spike,
        local_index,
        session_roots,
        row_map: row_map
            .into_iter()
            .map(|v| (v != u32::MAX).then_some(v))
            .collect(),
        time_of_spike,
        amp_of_spike,
        root_sizes,
        root_amp_sum,
        root_amp_count,
        root_feats,
        candidate_members,
        spike_rows,
        pair_roots,
    })
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(
            self.pos
                .checked_add(n)
                .is_some_and(|end| end <= self.buf.len()),
            "features sidecar is truncated"
        );
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }
    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }
}

/// One table file path pair for a shard part, exposed for the CLI's size logs.
pub fn table_path(dir: &Path, name: &str) -> (PathBuf, PathBuf) {
    table_paths(dir, name)
}

/// Merge-domain helper: collect the tables of one shard part in output order.
pub fn tables_of(tables: &SessionTables, names: &[&str]) -> Result<BTreeMap<String, Table>> {
    let mut out = BTreeMap::new();
    for &name in names {
        let table = match name {
            "roots" => tables.roots.clone(),
            "candidates" => tables.candidates.clone(),
            "pairs" => tables.pairs.clone(),
            "units" => tables.units.clone(),
            "cand_units" => tables.cand_units.clone(),
            "spikes" => tables.spikes.clone(),
            _ => bail!("unknown table {name:?}"),
        };
        out.insert(name.to_owned(), table);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learn::features::{
        cand_unit_columns, candidate_columns, pair_columns, root_columns, spike_columns,
        unit_columns, SessionTables,
    };

    fn empty_data() -> ExtractData {
        ExtractData {
            tables: SessionTables {
                roots: Table::new(EXTRACTOR_VERSION, root_columns()),
                candidates: Table::new(EXTRACTOR_VERSION, candidate_columns()),
                pairs: Table::new(EXTRACTOR_VERSION, pair_columns()),
                units: Table::new(EXTRACTOR_VERSION, unit_columns()),
                cand_units: Table::new(EXTRACTOR_VERSION, cand_unit_columns()),
                spikes: Table::new(EXTRACTOR_VERSION, spike_columns()),
            },
            root_of_spike: Vec::new(),
            local_index: Vec::new(),
            time_of_spike: Vec::new(),
            amp_of_spike: Vec::new(),
            root_feats: Vec::new(),
            root_center: Vec::new(),
            root_sizes: Vec::new(),
            root_amp_sum: Vec::new(),
            root_amp_count: Vec::new(),
            candidate_members: Vec::new(),
            spike_rows: Vec::new(),
            pair_roots: Vec::new(),
            row_map: Vec::new(),
            session_roots: Vec::new(),
            sr: 30_000.0,
            total_samples: 0,
        }
    }

    fn key() -> FeatureKey {
        FeatureKey {
            extractor_version: EXTRACTOR_VERSION.into(),
            max_spikes_per_root: 10,
            seed: 0,
            input_fingerprint: "fp".into(),
            manifest: None,
        }
    }

    #[test]
    fn saving_features_replaces_a_hard_linked_sidecar_without_touching_the_other_link() {
        let dir = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        let elsewhere = other.path().join(SIDE_FILE);
        fs::write(&elsewhere, b"previous sidecar").unwrap();
        if fs::hard_link(&elsewhere, dir.path().join(SIDE_FILE)).is_err() {
            eprintln!("skipping: the filesystem cannot create hard links here");
            return;
        }
        save_features(dir.path(), &key(), &empty_data()).unwrap();
        assert_eq!(
            fs::read(&elsewhere).unwrap(),
            b"previous sidecar".as_slice()
        );
        assert_ne!(
            fs::read(dir.path().join(SIDE_FILE)).unwrap(),
            b"previous sidecar".as_slice()
        );
    }

    #[test]
    fn projections_partition_header_metadata_by_column_ownership() {
        let mut table = Table::new(
            EXTRACTOR_VERSION,
            vec!["sid_idx".into(), "f_a".into(), "y_b".into()],
        );
        table
            .codes
            .insert("f_a".into(), [("x".to_owned(), 1)].into_iter().collect());
        table
            .codes
            .insert("y_b".into(), [("old".to_owned(), 0)].into_iter().collect());
        table.notes.insert("f_a".into(), "feature note".into());
        table.notes.insert("y_b".into(), "old target note".into());
        let features = feature_view(&table);
        assert!(features.codes.contains_key("f_a"));
        assert!(!features.codes.contains_key("y_b"));
        assert!(features.notes.contains_key("f_a"));
        assert!(!features.notes.contains_key("y_b"));
        let targets = target_view(&table);
        assert!(!targets.codes.contains_key("f_a"));
        assert!(targets.codes.contains_key("y_b"));
        assert!(!targets.notes.contains_key("f_a"));
        assert!(targets.notes.contains_key("y_b"));
    }

    #[test]
    fn legacy_keys_match_only_when_the_old_fields_match() {
        let stored = key();
        let requested = FeatureKey {
            manifest: Some(InputManifest {
                files: vec![InputFile {
                    name: "amplitudes.npy".into(),
                    size: 4,
                    sampled: false,
                    digest: "abc".into(),
                }],
            }),
            ..stored.clone()
        };
        assert!(requested.compatible(&stored));
        let legacy = blake3::hash(
            format!(
                "{}\n{}\n{}\n{}",
                stored.extractor_version,
                stored.max_spikes_per_root,
                stored.seed,
                stored.input_fingerprint
            )
            .as_bytes(),
        )
        .to_hex()
        .to_string();
        assert_eq!(stored.id(), legacy, "legacy id has no manifest");
        assert_ne!(requested.id(), stored.id(), "a manifest changes the id");
        for field in ["extractor_version", "input_fingerprint"] {
            let mut changed = stored.clone();
            match field {
                "extractor_version" => changed.extractor_version = "other".into(),
                _ => changed.input_fingerprint = "other".into(),
            }
            assert!(!requested.compatible(&changed), "{field}");
        }
        let mut other_manifest = requested.clone();
        other_manifest.manifest.as_mut().unwrap().files[0].digest = "different".into();
        assert!(!other_manifest.compatible(&requested));
        assert_ne!(other_manifest.id(), requested.id());
    }
}
