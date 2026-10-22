//! Threshold rules that turn QC metrics into labels (`sorrel apply`).
//!
//! A rule file is TOML:
//!
//! ```toml
//! # First matching rule wins; clusters matching none keep their label
//! # unless `default` is set.
//! default = "good"
//!
//! [[rule]]
//! metric = "n_spikes"
//! op = "<"
//! value = 300
//! label = "noise"
//!
//! [[rule]]
//! metric = "isi_violation_ratio"
//! op = ">"
//! value = 0.1
//! label = "mua"
//! ```
//!
//! Metric names are the `cluster_qc.tsv` columns ([`QcRow::METRICS`]).
//! A rule whose metric is `NaN` for a cluster never matches. Thresholds
//! are deliberately not built in: Bombcell, IBL and every lab differ, and
//! the example file in the docs is the place for opinions.

use crate::command::{CurationCommand, PhyLabelOp};
use crate::qc_export::QcRow;
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use sorrel_io::ClusterId;
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
pub struct Rule {
    pub metric: String,
    pub op: String,
    pub value: f32,
    pub label: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct RuleSet {
    /// Label for clusters that match no rule. Absent = leave unchanged.
    pub default: Option<String>,
    #[serde(default, rename = "rule")]
    pub rules: Vec<Rule>,
}

impl RuleSet {
    pub fn from_path(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let set: RuleSet =
            toml::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
        set.validate()?;
        Ok(set)
    }

    fn validate(&self) -> Result<()> {
        for r in &self.rules {
            if !QcRow::METRICS.contains(&r.metric.as_str()) {
                bail!(
                    "unknown metric {:?}; expected one of {}",
                    r.metric,
                    QcRow::METRICS.join(", ")
                );
            }
            compare(&r.op, 0.0, 0.0).with_context(|| format!("rule on {:?}", r.metric))?;
            label_op(&r.label)?;
        }
        if let Some(d) = &self.default {
            label_op(d)?;
        }
        Ok(())
    }

    /// The label the first matching rule assigns, or the default.
    pub fn label_for(&self, row: &QcRow) -> Result<Option<PhyLabelOp>> {
        for r in &self.rules {
            let Some(v) = row.get(&r.metric) else {
                continue;
            };
            if v.is_nan() {
                continue;
            }
            if compare(&r.op, v, r.value)? {
                return Ok(Some(label_op(&r.label)?));
            }
        }
        self.default.as_deref().map(label_op).transpose()
    }
}

fn compare(op: &str, v: f32, threshold: f32) -> Result<bool> {
    Ok(match op {
        "<" => v < threshold,
        "<=" => v <= threshold,
        ">" => v > threshold,
        ">=" => v >= threshold,
        "==" | "=" => v == threshold,
        "!=" => v != threshold,
        other => bail!("unknown op {other:?}; expected <, <=, >, >=, ==, !="),
    })
}

fn label_op(s: &str) -> Result<PhyLabelOp> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "good" => PhyLabelOp::SetGood,
        "mua" => PhyLabelOp::SetMua,
        "noise" => PhyLabelOp::SetNoise,
        "unsorted" => PhyLabelOp::SetUnsorted,
        other => bail!("unknown label {other:?}; expected good, mua, noise or unsorted"),
    })
}

/// Evaluate `rules` over `rows` and return one `Relabel` per cluster whose
/// label changes. `current` maps a cluster to its present label string
/// (`"good"`, ...), so already-correct clusters are skipped and the batch
/// is a pure delta. Empty clusters are never relabelled.
pub fn apply_rules(
    rules: &RuleSet,
    rows: &[QcRow],
    current: impl Fn(ClusterId) -> &'static str,
) -> Result<Vec<CurationCommand>> {
    let mut out = Vec::new();
    for row in rows {
        if row.n_spikes == 0 {
            continue;
        }
        let cluster = ClusterId(row.cluster_id);
        let Some(op) = rules.label_for(row)? else {
            continue;
        };
        if label_op(current(cluster))? == op {
            continue;
        }
        out.push(CurationCommand::Relabel { cluster, op });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: u32, n: u32, isi: f32) -> QcRow {
        QcRow {
            cluster_id: id,
            n_spikes: n,
            mean_amplitude: 1.0,
            firing_rate_hz: 1.0,
            presence_ratio: 1.0,
            isi_violations: 0,
            isi_violation_ratio: isi,
            amplitude_cutoff: 0.0,
            amplitude_snr: 5.0,
            drift_correlation: 0.0,
            longest_silent_gap: 0.0,
            composite_quality: 0.5,
            isolation_distance_sq: f32::NAN,
            l_ratio: f32::NAN,
            nn_isolation: f32::NAN,
            d_prime: f32::NAN,
            silhouette: f32::NAN,
        }
    }

    #[test]
    fn first_match_wins_nan_never_matches_and_unchanged_labels_are_skipped() {
        let set: RuleSet = toml::from_str(
            r#"
            default = "good"
            [[rule]]
            metric = "n_spikes"
            op = "<"
            value = 100
            label = "noise"
            [[rule]]
            metric = "isi_violation_ratio"
            op = ">"
            value = 0.1
            label = "mua"
            [[rule]]
            metric = "l_ratio"
            op = ">"
            value = 0.0
            label = "noise"
            "#,
        )
        .unwrap();
        set.validate().unwrap();
        // 0: tiny → noise (rule 1, even though isi also matches rule 2)
        // 1: isi high → mua; 2: nothing matches (l_ratio NaN) → default good
        // 3: already good → skipped; 4: empty → skipped.
        let rows = [
            row(0, 10, 0.5),
            row(1, 500, 0.5),
            row(2, 500, 0.0),
            row(3, 500, 0.0),
            row(4, 0, 0.0),
        ];
        let current = |c: ClusterId| if c.0 == 3 { "good" } else { "unsorted" };
        let cmds = apply_rules(&set, &rows, current).unwrap();
        assert_eq!(cmds.len(), 3);
        assert!(matches!(
            cmds[0],
            CurationCommand::Relabel {
                cluster: ClusterId(0),
                op: PhyLabelOp::SetNoise
            }
        ));
        assert!(matches!(
            cmds[1],
            CurationCommand::Relabel {
                cluster: ClusterId(1),
                op: PhyLabelOp::SetMua
            }
        ));
        assert!(matches!(
            cmds[2],
            CurationCommand::Relabel {
                cluster: ClusterId(2),
                op: PhyLabelOp::SetGood
            }
        ));

        let bad: RuleSet =
            toml::from_str("[[rule]]\nmetric = \"snr\"\nop = \"<\"\nvalue = 1\nlabel = \"good\"\n")
                .unwrap();
        assert!(bad
            .validate()
            .unwrap_err()
            .to_string()
            .contains("unknown metric"));
    }
}
