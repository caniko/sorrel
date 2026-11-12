//! Portable `.sorrel-model` curator bundles.
//!
//! A bundle is the unit a user copies between machines: one file holding
//! every learned "head" (a merge-accept model, a cut model, …), the feature
//! schema each head was trained against, and provenance. The encoding is
//! deliberately boring and self-describing:
//!
//! ```text
//! b"SRLMODEL" | u32-LE format version | deflate(MessagePack bundle) | blake3(payload)
//! ```
//!
//! The checksum covers the *uncompressed* MessagePack payload, so it also
//! detects a decoder that silently stopped halfway. `save` writes to a
//! temporary file in the destination directory, syncs it and renames it over
//! the target, so an interrupted write never leaves a half bundle behind.
//!
//! Feature schemas are keyed by name and carry a hash of their column names
//! plus the extractor version that produced them. Heads reference a schema by
//! key, and [`Head::predict`] refuses to run when the caller's hash differs:
//! a model trained on features from a different extractor must not silently
//! score garbage.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::io::{Read, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, ensure, Context, Result};
use flate2::read::DeflateDecoder;
use flate2::write::DeflateEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};

use super::gbdt::{self, Objective};

/// File magic, eight bytes so the header is easy to spot in a hex dump.
pub const MAGIC: &[u8; 8] = b"SRLMODEL";
/// Value of [`CuratorBundle::format`].
pub const FORMAT_NAME: &str = "sorrel-curator-model";
/// Current on-disk format version. Bumped when the payload layout changes.
pub const FORMAT_VERSION: u32 = 1;
/// Length of the trailing blake3 checksum.
const CHECKSUM_LEN: usize = 32;
/// Refuse absurd payloads (decompression bombs) rather than exhausting memory.
const MAX_PAYLOAD_BYTES: u64 = 1 << 30;
/// Number of hex characters kept from a schema hash.
const HASH_CHARS: usize = 16;

/// Feature columns and the extractor version they were produced by.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureSchema {
    /// Column names, in model order.
    pub names: Vec<String>,
    /// First 16 hex characters of blake3 over the names and extractor version.
    pub hash: String,
}

impl FeatureSchema {
    /// Build a schema and compute its hash for `extractor_version`.
    pub fn new(names: Vec<String>, extractor_version: &str) -> Self {
        let hash = Self::hash_names(&names, extractor_version);
        Self { names, hash }
    }

    /// The hash a caller must present to score a head trained on `names`.
    ///
    /// The exact construction is `blake3(names.join("\n") + "\n" +
    /// extractor_version)` truncated to 16 hex characters. It is intentionally
    /// simple so a Python/Julia extractor can reproduce it.
    pub fn hash_names(names: &[String], extractor_version: &str) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(names.join("\n").as_bytes());
        hasher.update(b"\n");
        hasher.update(extractor_version.as_bytes());
        hasher
            .finalize()
            .to_hex()
            .chars()
            .take(HASH_CHARS)
            .collect()
    }

    /// Whether a caller-computed hash matches this schema.
    pub fn matches(&self, caller_hash: &str) -> bool {
        self.hash == caller_hash
    }

    /// Bundle key for this schema: derived from its hash, so two heads trained
    /// on the same columns and extractor version share one entry and a
    /// different schema can never overwrite it.
    pub fn key(&self) -> String {
        format!("schema:{}", self.hash)
    }
}

/// One learned model plus the schema and metadata needed to interpret it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Head {
    /// Key into [`CuratorBundle::feature_schemas`].
    pub schema: String,
    /// Fitted trees and base score.
    pub model: gbdt::Model,
    /// Output names. Empty for regression and softmax (where class order is
    /// implicit), or `["neg", "pos"]` for a binary head.
    pub classes: Vec<String>,
    /// Target column the head was trained on, so scoring raw curation rows
    /// uses the same target rather than assuming one.
    pub target: String,
    /// `OLD=NEW` class-code remap applied at training time; scoring applies it
    /// again, so a model never silently skips a class it was trained to merge.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub class_map: BTreeMap<i64, i64>,
    /// Optional decision threshold for a binary or score head.
    pub threshold: Option<f64>,
    /// Evaluation metrics, e.g. `auc` or `jaccard`, for reporting.
    pub metrics: BTreeMap<String, f64>,
}

impl Head {
    /// A head with no threshold or metrics. Binary heads default to
    /// `["neg", "pos"]`; regression and softmax leave `classes` empty.
    pub fn new(schema: impl Into<String>, model: gbdt::Model) -> Self {
        let classes = if matches!(model.objective, Objective::Logistic) {
            vec!["neg".to_string(), "pos".to_string()]
        } else {
            Vec::new()
        };
        Self {
            schema: schema.into(),
            model,
            classes,
            target: String::new(),
            class_map: BTreeMap::new(),
            threshold: None,
            metrics: BTreeMap::new(),
        }
    }

    /// Set the output names.
    pub fn with_classes(mut self, classes: Vec<String>) -> Self {
        self.classes = classes;
        self
    }

    /// Set the decision threshold.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Add one metric.
    pub fn with_metric(mut self, key: impl Into<String>, value: f64) -> Self {
        self.metrics.insert(key.into(), value);
        self
    }

    fn check_schema<'a>(
        &self,
        bundle: &'a CuratorBundle,
        caller_hash: &str,
    ) -> Result<&'a FeatureSchema> {
        let schema = bundle.feature_schemas.get(&self.schema).with_context(|| {
            format!(
                "head references feature schema {:?}, which the bundle does not contain",
                self.schema
            )
        })?;
        if !schema.matches(caller_hash) {
            bail!(
                "feature schema mismatch for head using {:?}: bundle hash is {} but the caller computed {}; \
                 re-extract features with the same extractor version or retrain the model",
                self.schema,
                schema.hash,
                caller_hash
            );
        }
        Ok(schema)
    }

    /// Score one row, refusing caller features from a different schema.
    ///
    /// The return value follows [`gbdt::Model::predict`]: a class index for
    /// classification (binary uses 0/1) or the predicted value for regression.
    /// For a binary head with a stored [`Head::threshold`], the positive class
    /// is decided against that threshold instead of the model's default 0.5.
    pub fn predict(
        &self,
        bundle: &CuratorBundle,
        features: &[f32],
        caller_schema_hash: &str,
    ) -> Result<f64> {
        self.check_schema(bundle, caller_schema_hash)?;
        ensure!(
            features.len() == self.model.n_features,
            "head expects {} features, caller supplied {}",
            self.model.n_features,
            features.len()
        );
        match (self.threshold, self.model.objective) {
            (Some(threshold), Objective::Logistic) => Ok(f64::from(
                self.model.predict_proba(features)[0] >= threshold,
            )),
            _ => Ok(self.model.predict(features)),
        }
    }

    /// Probabilities for one row, aligned to [`Head::classes`] when this is a
    /// binary head with two class names. See [`gbdt::Model::predict_proba`]
    /// for the objective-native shapes.
    pub fn predict_proba(
        &self,
        bundle: &CuratorBundle,
        features: &[f32],
        caller_schema_hash: &str,
    ) -> Result<Vec<f64>> {
        self.check_schema(bundle, caller_schema_hash)?;
        ensure!(
            features.len() == self.model.n_features,
            "head expects {} features, caller supplied {}",
            self.model.n_features,
            features.len()
        );
        let mut p = self.model.predict_proba(features);
        if matches!(self.model.objective, Objective::Logistic) && self.classes.len() == 2 {
            let positive = p[0];
            p = vec![1.0 - positive, positive];
        }
        Ok(p)
    }
}

/// Where a bundle came from and what built it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// Unix timestamp at training time.
    pub created_unix: u64,
    /// Sorrel version that produced the bundle.
    pub sorrel_version: String,
    /// Optional source revision (e.g. `git describe`).
    pub sorrel_rev: Option<String>,
    /// Feature extractor version the schemas' hashes were computed with.
    pub extractor_version: String,
    /// Opaque session identifiers, never paths or raw data.
    pub training_sessions: Vec<String>,
    /// Training row count per head or session.
    pub n_rows: BTreeMap<String, u64>,
    /// Free-form notes.
    pub notes: String,
}

impl Provenance {
    /// Provenance with an explicit timestamp.
    pub fn new(
        created_unix: u64,
        sorrel_version: impl Into<String>,
        sorrel_rev: Option<String>,
        extractor_version: impl Into<String>,
    ) -> Self {
        Self {
            created_unix,
            sorrel_version: sorrel_version.into(),
            sorrel_rev,
            extractor_version: extractor_version.into(),
            training_sessions: Vec::new(),
            n_rows: BTreeMap::new(),
            notes: String::new(),
        }
    }

    /// Provenance stamped with the current time.
    pub fn now(
        sorrel_version: impl Into<String>,
        sorrel_rev: Option<String>,
        extractor_version: impl Into<String>,
    ) -> Self {
        let created_unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self::new(created_unix, sorrel_version, sorrel_rev, extractor_version)
    }
}

/// A complete curator model bundle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CuratorBundle {
    /// Always [`FORMAT_NAME`].
    pub format: String,
    /// On-disk format version, also stored in the file header.
    pub format_version: u32,
    /// Feature schemas keyed by a caller-chosen name.
    pub feature_schemas: BTreeMap<String, FeatureSchema>,
    /// Learned heads keyed by a caller-chosen name.
    pub heads: BTreeMap<String, Head>,
    /// Where and how the bundle was produced.
    pub provenance: Provenance,
}

impl CuratorBundle {
    /// An empty bundle with no schemas or heads.
    pub fn new(provenance: Provenance) -> Self {
        Self {
            format: FORMAT_NAME.to_string(),
            format_version: FORMAT_VERSION,
            feature_schemas: BTreeMap::new(),
            heads: BTreeMap::new(),
            provenance,
        }
    }

    /// Add a feature schema.
    pub fn add_schema(&mut self, key: impl Into<String>, schema: FeatureSchema) -> &mut Self {
        self.feature_schemas.insert(key.into(), schema);
        self
    }

    /// Add a feature schema without ever replacing one that a different hash
    /// already owns. Two heads on the same table and target share a schema;
    /// retraining a head with different columns must add a new entry instead of
    /// silently changing the schema another head still depends on.
    pub fn add_schema_checked(
        &mut self,
        key: impl Into<String>,
        schema: FeatureSchema,
    ) -> Result<&mut Self> {
        let key = key.into();
        if let Some(existing) = self.feature_schemas.get(&key) {
            ensure!(
                existing.hash == schema.hash,
                "schema {key:?} already exists with hash {} and would be overwritten with {}; \
                 a head may still depend on the old schema",
                existing.hash,
                schema.hash
            );
        }
        Ok(self.add_schema(key, schema))
    }

    /// Add a learned head.
    pub fn add_head(&mut self, key: impl Into<String>, head: Head) -> &mut Self {
        self.heads.insert(key.into(), head);
        self
    }

    /// Structural and cross-reference validation.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.format == FORMAT_NAME,
            "unexpected bundle format {:?} (expected {:?})",
            self.format,
            FORMAT_NAME
        );
        ensure!(
            self.format_version == FORMAT_VERSION,
            "bundle payload is format v{}, expected v{FORMAT_VERSION}",
            self.format_version
        );
        ensure!(
            !self.provenance.extractor_version.is_empty(),
            "provenance has an empty extractor version; schema hashes would be meaningless"
        );
        for (key, schema) in &self.feature_schemas {
            ensure!(!key.is_empty(), "feature schema key is empty");
            ensure!(
                !schema.names.is_empty(),
                "feature schema {key:?} has no feature names"
            );
            let mut seen = BTreeSet::new();
            for name in &schema.names {
                ensure!(
                    !name.is_empty(),
                    "feature schema {key:?} contains an empty feature name"
                );
                ensure!(
                    seen.insert(name.as_str()),
                    "feature schema {key:?} repeats feature name {name:?}"
                );
            }
            let expected =
                FeatureSchema::hash_names(&schema.names, &self.provenance.extractor_version);
            ensure!(
                schema.hash == expected,
                "feature schema {key:?} hash is {} but its names and extractor version hash to {expected}",
                schema.hash
            );
        }
        for (key, head) in &self.heads {
            ensure!(!key.is_empty(), "head key is empty");
            let schema = self.feature_schemas.get(&head.schema).with_context(|| {
                format!("head {key:?} references missing schema {:?}", head.schema)
            })?;
            ensure!(
                schema.names.len() == head.model.n_features,
                "head {key:?} expects {} features but schema {:?} has {}",
                head.model.n_features,
                head.schema,
                schema.names.len()
            );
            head.model
                .validate()
                .with_context(|| format!("head {key:?} model is invalid"))?;
            match head.model.objective {
                Objective::Logistic => ensure!(
                    head.classes.is_empty() || head.classes.len() == 2,
                    "binary head {key:?} must have 0 or 2 class names, has {}",
                    head.classes.len()
                ),
                Objective::Softmax { n_classes } => ensure!(
                    head.classes.is_empty() || head.classes.len() == n_classes,
                    "softmax head {key:?} has {} classes but {} class names",
                    n_classes,
                    head.classes.len()
                ),
                Objective::SquaredError => ensure!(
                    head.classes.is_empty(),
                    "regression head {key:?} must not name classes"
                ),
            }
            if let Some(threshold) = head.threshold {
                ensure!(
                    threshold.is_finite(),
                    "head {key:?} has a non-finite threshold"
                );
            }
            for (metric, value) in &head.metrics {
                ensure!(!metric.is_empty(), "head {key:?} has an empty metric name");
                ensure!(
                    value.is_finite(),
                    "head {key:?} metric {metric:?} is not finite"
                );
            }
        }
        Ok(())
    }

    /// Encode the bundle as `.sorrel-model` bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let payload = rmp_serde::to_vec_named(self).context("encode curator bundle payload")?;
        let checksum = blake3::hash(&payload);
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder
            .write_all(&payload)
            .context("compress curator bundle payload")?;
        let compressed = encoder.finish().context("finish curator bundle")?;
        let mut out = Vec::with_capacity(MAGIC.len() + 4 + compressed.len() + CHECKSUM_LEN);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&compressed);
        out.extend_from_slice(checksum.as_bytes());
        Ok(out)
    }

    /// Decode and validate `.sorrel-model` bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let header = MAGIC.len() + 4 + CHECKSUM_LEN;
        ensure!(
            bytes.len() >= header,
            "curator model is too short ({} bytes)",
            bytes.len()
        );
        ensure!(
            &bytes[..MAGIC.len()] == MAGIC,
            "not a sorrel curator model: bad magic bytes (expected {:?})",
            String::from_utf8_lossy(MAGIC)
        );
        let version = u32::from_le_bytes([
            bytes[MAGIC.len()],
            bytes[MAGIC.len() + 1],
            bytes[MAGIC.len() + 2],
            bytes[MAGIC.len() + 3],
        ]);
        if version > FORMAT_VERSION {
            bail!(
                "curator model format v{version} is newer than this sorrel build supports (v{FORMAT_VERSION}); \
                 upgrade sorrel to load it"
            );
        }
        ensure!(
            version == FORMAT_VERSION,
            "curator model uses unsupported format v{version} (this build reads v{FORMAT_VERSION})"
        );
        let payload_end = bytes.len() - CHECKSUM_LEN;
        let compressed = &bytes[MAGIC.len() + 4..payload_end];
        let mut decoder = DeflateDecoder::new(compressed).take(MAX_PAYLOAD_BYTES + 1);
        let mut payload = Vec::new();
        decoder
            .read_to_end(&mut payload)
            .context("decompress curator model payload")?;
        ensure!(
            payload.len() as u64 <= MAX_PAYLOAD_BYTES,
            "curator model payload exceeds {MAX_PAYLOAD_BYTES} bytes"
        );
        let expected = &bytes[payload_end..];
        let actual = blake3::hash(&payload);
        ensure!(
            actual.as_bytes() == expected,
            "curator model checksum mismatch ({} vs {}); the file is corrupted or truncated",
            actual.to_hex(),
            hex_bytes(expected)
        );
        let bundle: Self =
            rmp_serde::from_slice(&payload).context("decode curator bundle payload")?;
        bundle.validate()?;
        Ok(bundle)
    }

    /// Write the bundle atomically: temp file in the same directory, fsync,
    /// then rename over `path`.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let bytes = self.to_bytes()?;
        let dir = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(dir).with_context(|| format!("create directory {}", dir.display()))?;
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("model");
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let tmp = dir.join(format!(".{name}.tmp.{}.{stamp}", std::process::id()));
        let result = (|| -> Result<()> {
            let mut file = fs::File::create(&tmp)
                .with_context(|| format!("create temporary file {}", tmp.display()))?;
            file.write_all(&bytes)
                .with_context(|| format!("write temporary file {}", tmp.display()))?;
            file.sync_all()
                .with_context(|| format!("sync temporary file {}", tmp.display()))?;
            fs::rename(&tmp, path)
                .with_context(|| format!("rename {} to {}", tmp.display(), path.display()))?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&tmp);
        }
        result
    }

    /// Read and validate a `.sorrel-model` file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes =
            fs::read(path).with_context(|| format!("read curator model {}", path.display()))?;
        Self::from_bytes(&bytes).with_context(|| format!("load curator model {}", path.display()))
    }

    /// Human-readable description used by `sorrel model show` and reports.
    pub fn summary(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "sorrel curator model v{} ({})",
            self.format_version, self.format
        );
        let _ = writeln!(
            out,
            "created: {} (unix seconds)",
            self.provenance.created_unix
        );
        match &self.provenance.sorrel_rev {
            Some(rev) => {
                let _ = writeln!(
                    out,
                    "sorrel: {} (rev {rev})",
                    self.provenance.sorrel_version
                );
            }
            None => {
                let _ = writeln!(out, "sorrel: {}", self.provenance.sorrel_version);
            }
        }
        let _ = writeln!(out, "extractor: {}", self.provenance.extractor_version);
        let _ = writeln!(
            out,
            "training sessions ({}): {}",
            self.provenance.training_sessions.len(),
            if self.provenance.training_sessions.is_empty() {
                "none".to_string()
            } else {
                self.provenance.training_sessions.join(", ")
            }
        );
        if !self.provenance.n_rows.is_empty() {
            let rows: Vec<String> = self
                .provenance
                .n_rows
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect();
            let _ = writeln!(out, "rows: {}", rows.join(", "));
        }
        if !self.provenance.notes.is_empty() {
            let _ = writeln!(out, "notes: {}", self.provenance.notes);
        }
        let _ = writeln!(out, "schemas:");
        for (key, schema) in &self.feature_schemas {
            let _ = writeln!(
                out,
                "  {key}: {} features, hash {}",
                schema.names.len(),
                schema.hash
            );
        }
        let _ = writeln!(out, "heads:");
        for (key, head) in &self.heads {
            let per_round = head.model.trees_per_round;
            let nodes: usize = head.model.trees.iter().map(|tree| tree.value.len()).sum();
            let size = rmp_serde::to_vec_named(&head.model)
                .map(|b| b.len())
                .unwrap_or(0);
            let _ = writeln!(
                out,
                "  {key}: schema {} ({} features), objective {}",
                head.schema,
                self.feature_schemas
                    .get(&head.schema)
                    .map_or(0, |s| s.names.len()),
                objective_name(head.model.objective)
            );
            let _ = writeln!(
                out,
                "    classes: {}",
                if head.classes.is_empty() {
                    "none".to_string()
                } else {
                    format!("[{}]", head.classes.join(", "))
                }
            );
            if let Some(threshold) = head.threshold {
                let _ = writeln!(out, "    threshold: {threshold:.4}");
            }
            if !head.metrics.is_empty() {
                let metrics: Vec<String> = head
                    .metrics
                    .iter()
                    .map(|(k, v)| format!("{k}={v:.4}"))
                    .collect();
                let _ = writeln!(out, "    metrics: {}", metrics.join(", "));
            }
            let _ = writeln!(
                out,
                "    trees: {} ({} per round), nodes: {nodes}, msgpack: {size} bytes",
                head.model.trees.len(),
                per_round
            );
        }
        match self.to_bytes() {
            Ok(bytes) => {
                let _ = writeln!(out, "file size: {} bytes", bytes.len());
            }
            Err(e) => {
                let _ = writeln!(out, "file size: unavailable ({e})");
            }
        }
        out
    }
}

fn objective_name(objective: Objective) -> String {
    match objective {
        Objective::Logistic => "logistic".to_string(),
        Objective::Softmax { n_classes } => format!("softmax/{n_classes}"),
        Objective::SquaredError => "squared_error".to_string(),
    }
}

fn hex_bytes(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learn::gbdt::{train, Params};
    use crate::learn::test_support::Rng;

    fn tiny_model() -> gbdt::Model {
        let mut rng = Rng::new(5);
        let n = 400;
        let mut x = Vec::with_capacity(n * 2);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            let a = rng.normal();
            let b = rng.normal();
            x.push(a as f32);
            x.push(b as f32);
            y.push(f32::from(a + b > 0.0));
        }
        let params = Params {
            n_rounds: 20,
            ..Params::default()
        };
        train(&x, 2, &y, None, Objective::Logistic, &params).unwrap()
    }

    fn sample_bundle() -> CuratorBundle {
        let extractor = "spikes-v3";
        let names = vec!["amp_iqr".to_string(), "isolation".to_string()];
        let mut provenance = Provenance::new(
            1_700_000_000,
            "0.4.0",
            Some("abc1234".to_string()),
            extractor,
        );
        provenance.training_sessions = vec!["ses_alpha".to_string(), "ses_beta".to_string()];
        provenance.n_rows.insert("merge_accept".to_string(), 900);
        provenance.notes = "unit-test bundle".to_string();
        let mut bundle = CuratorBundle::new(provenance);
        bundle.add_schema("cluster", FeatureSchema::new(names, extractor));
        let head = Head::new("cluster", tiny_model())
            .with_classes(vec!["neg".to_string(), "pos".to_string()])
            .with_threshold(0.5)
            .with_metric("auc", 0.93);
        bundle.add_head("merge_accept", head);
        bundle
    }

    fn rows() -> Vec<[f32; 2]> {
        let mut rng = Rng::new(9);
        (0..40)
            .map(|_| [rng.normal() as f32, rng.normal() as f32])
            .collect()
    }

    #[test]
    fn schema_hash_is_stable_and_version_sensitive() {
        let names = vec!["a".to_string(), "b".to_string()];
        let a = FeatureSchema::new(names.clone(), "ex1");
        assert_eq!(a.hash.len(), HASH_CHARS);
        assert_eq!(a.hash, FeatureSchema::hash_names(&names, "ex1"));
        assert_ne!(a.hash, FeatureSchema::hash_names(&names, "ex2"));
        assert_ne!(
            a.hash,
            FeatureSchema::hash_names(&["a".to_string(), "c".to_string()], "ex1")
        );
        assert!(a.matches(&a.hash));
    }

    #[test]
    fn file_round_trip_preserves_predictions() {
        let bundle = sample_bundle();
        let before: Vec<Vec<f64>> = rows()
            .iter()
            .map(|row| {
                bundle.heads["merge_accept"]
                    .predict_proba(&bundle, row, &bundle.feature_schemas["cluster"].hash)
                    .unwrap()
            })
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("curator.sorrel-model");
        bundle.save(&path).unwrap();
        let loaded = CuratorBundle::load(&path).unwrap();
        assert_eq!(loaded, bundle);
        for (row, expected) in rows().iter().zip(&before) {
            let got = loaded.heads["merge_accept"]
                .predict_proba(&loaded, row, &loaded.feature_schemas["cluster"].hash)
                .unwrap();
            assert_eq!(&got, expected);
        }
        let bytes = bundle.to_bytes().unwrap();
        assert_eq!(CuratorBundle::from_bytes(&bytes).unwrap(), bundle);
        // Overwriting an existing bundle must also be atomic and valid.
        bundle.save(&path).unwrap();
        assert!(CuratorBundle::load(&path).is_ok());
        let summary = loaded.summary();
        assert!(summary.contains("merge_accept"), "{summary}");
        assert!(summary.contains("ses_alpha"), "{summary}");
        assert!(
            summary.contains(&bundle.feature_schemas["cluster"].hash),
            "{summary}"
        );
    }

    #[test]
    fn corrupted_file_is_rejected_by_checksum() {
        let mut bytes = sample_bundle().to_bytes().unwrap();
        *bytes.last_mut().unwrap() ^= 0xFF;
        let err = CuratorBundle::from_bytes(&bytes).unwrap_err().to_string();
        assert!(err.contains("checksum"), "{err}");
    }

    #[test]
    fn corrupted_payload_is_rejected() {
        let mut bytes = sample_bundle().to_bytes().unwrap();
        // A byte inside the compressed payload must never decode silently.
        bytes[MAGIC.len() + 4 + 3] ^= 0xFF;
        assert!(CuratorBundle::from_bytes(&bytes).is_err());
    }

    #[test]
    fn wrong_magic_is_rejected() {
        let mut bytes = sample_bundle().to_bytes().unwrap();
        bytes[0] ^= 0xFF;
        let err = CuratorBundle::from_bytes(&bytes).unwrap_err().to_string();
        assert!(err.contains("magic"), "{err}");
    }

    #[test]
    fn newer_format_is_refused_with_a_clear_message() {
        let mut bytes = sample_bundle().to_bytes().unwrap();
        bytes[MAGIC.len()..MAGIC.len() + 4].copy_from_slice(&(FORMAT_VERSION + 1).to_le_bytes());
        let err = CuratorBundle::from_bytes(&bytes).unwrap_err().to_string();
        assert!(err.contains("newer") && err.contains("upgrade"), "{err}");
    }

    #[test]
    fn a_foreign_schema_hash_cannot_score_a_head() {
        let bundle = sample_bundle();
        let err = bundle.heads["merge_accept"]
            .predict(&bundle, &[0.0, 0.0], "0000000000000000")
            .unwrap_err()
            .to_string();
        assert!(err.contains("schema mismatch"), "{err}");
        assert!(bundle.heads["merge_accept"]
            .predict(
                &bundle,
                &[0.0, 0.0],
                &bundle.feature_schemas["cluster"].hash
            )
            .is_ok());
    }

    #[test]
    fn mismatched_widths_and_classes_fail_validation() {
        let mut bundle = sample_bundle();
        bundle.add_schema(
            "wrong_width",
            FeatureSchema::new(vec!["only".to_string()], "spikes-v3"),
        );
        bundle.heads.get_mut("merge_accept").unwrap().schema = "wrong_width".to_string();
        let err = bundle.validate().unwrap_err().to_string();
        assert!(err.contains("expects 2 features"), "{err}");

        let mut bundle = sample_bundle();
        bundle.heads.get_mut("merge_accept").unwrap().classes = vec!["one".to_string()];
        assert!(bundle.validate().is_err());

        let mut bundle = sample_bundle();
        bundle.feature_schemas.get_mut("cluster").unwrap().hash = "deadbeefdeadbeef".to_string();
        let err = bundle.validate().unwrap_err().to_string();
        assert!(err.contains("hash"), "{err}");
    }

    #[test]
    fn schema_keys_are_hashes_and_never_overwrite_a_different_schema() {
        let mut bundle = CuratorBundle::new(Provenance::new(0, "0.4.0", None, "ex"));
        let a = FeatureSchema::new(vec!["a".to_string()], "ex");
        let b = FeatureSchema::new(vec!["b".to_string()], "ex");
        assert_eq!(a.key(), format!("schema:{}", a.hash));
        assert_ne!(a.key(), b.key());
        bundle.add_schema_checked(a.key(), a.clone()).unwrap();
        // Same content is a no-op, not a conflict.
        bundle.add_schema_checked(a.key(), a.clone()).unwrap();
        bundle.add_schema_checked(b.key(), b.clone()).unwrap();
        assert_eq!(bundle.feature_schemas.len(), 2);
        // A fixed key that already holds a different hash is refused.
        let mut other = CuratorBundle::new(Provenance::new(0, "0.4.0", None, "ex"));
        other.add_schema_checked("fixed", a.clone()).unwrap();
        let err = other
            .add_schema_checked("fixed", b.clone())
            .unwrap_err()
            .to_string();
        assert!(err.contains("would be overwritten"), "{err}");
    }

    #[test]
    fn binary_threshold_is_applied_to_predict() {
        let extractor = "ex-threshold";
        let mut bundle = CuratorBundle::new(Provenance::new(0, "0.4.0", None, extractor));
        let schema = FeatureSchema::new(vec!["a".to_string()], extractor);
        let key = schema.key();
        bundle.add_schema(key.clone(), schema);
        // A base-only model with positive probability 0.8.
        let model = gbdt::Model {
            n_features: 1,
            objective: Objective::Logistic,
            base_score: vec![(0.8f64 / 0.2).ln()],
            trees_per_round: 1,
            trees: Vec::new(),
            importance: vec![0.0],
        };
        let mut head = Head::new(key.clone(), model);
        let hash = bundle.feature_schemas[&key].hash.clone();
        assert_eq!(head.predict(&bundle, &[0.0], &hash).unwrap(), 1.0);
        head = head.with_threshold(0.9);
        assert_eq!(head.predict(&bundle, &[0.0], &hash).unwrap(), 0.0);
        head = head.with_threshold(0.5);
        assert_eq!(head.predict(&bundle, &[0.0], &hash).unwrap(), 1.0);
        let p = head.predict_proba(&bundle, &[0.0], &hash).unwrap();
        assert!((p[1] - 0.8).abs() < 1e-12, "{p:?}");
    }
}
