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
//! `name = "..."` gives a rule a reason code (default `r<index>`);
//! `all = [{ metric = "...", op = "|>", value = 0.5 }]` adds conditions
//! ANDed with the rule's own, and an op prefixed with `|` compares the
//! absolute value.
//!
//! Metric names are the `cluster_qc.tsv` columns ([`QcRow::METRICS`]).
//! A rule whose metric is `NaN` for a cluster never matches. Thresholds
//! are deliberately not built in: Bombcell, IBL and every lab differ, and
//! the example file in the docs is the place for opinions.

use crate::command::{CurationCommand, PhyLabelOp};
use crate::qc_export::QcRow;
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::Value;
use sorrel_io::ClusterId;
use std::path::Path;

/// One comparison of a metric with a threshold. An `op` starting with `|`
/// compares the absolute value (`"|>"` is `|metric| > value`).
#[derive(Clone, Debug, Deserialize)]
pub struct Cond {
    pub metric: String,
    pub op: String,
    pub value: f32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Rule {
    pub metric: String,
    pub op: String,
    pub value: f32,
    pub label: String,
    /// Reason code shown in reports and proposals; default `r<index>`.
    #[serde(default)]
    pub name: Option<String>,
    /// Further conditions, ANDed with the rule's own.
    #[serde(default)]
    pub all: Vec<Cond>,
}

impl Rule {
    /// The rule's own condition followed by its `all` conditions.
    pub fn conds(&self) -> impl Iterator<Item = (&str, &str, f32)> {
        std::iter::once((self.metric.as_str(), self.op.as_str(), self.value)).chain(
            self.all
                .iter()
                .map(|c| (c.metric.as_str(), c.op.as_str(), c.value)),
        )
    }
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
        Self::parse(&text).with_context(|| path.display().to_string())
    }

    /// Parse and validate TOML rule text.
    pub fn parse(text: &str) -> Result<Self> {
        let set: RuleSet = toml::from_str(text).context("parse rules")?;
        set.validate()?;
        Ok(set)
    }

    fn validate(&self) -> Result<()> {
        for r in &self.rules {
            for (metric, op, t) in r.conds() {
                if !QcRow::METRICS.contains(&metric) {
                    bail!(
                        "unknown metric {metric:?}; expected one of {}",
                        QcRow::METRICS.join(", ")
                    );
                }
                compare(op.strip_prefix('|').unwrap_or(op), 0.0, 0.0)
                    .with_context(|| format!("rule on {metric:?}"))?;
                if !t.is_finite() {
                    bail!("rule on {metric:?}: value {t} is not finite");
                }
            }
            label_op(&r.label)?;
        }
        if let Some(d) = &self.default {
            label_op(d)?;
        }
        Ok(())
    }

    /// Name of rule `i`: its `name`, else `r<i>`.
    pub fn name(&self, i: usize) -> String {
        self.rules[i]
            .name
            .clone()
            .unwrap_or_else(|| format!("r{i}"))
    }

    /// Index of the first rule whose conditions all hold for `row`.
    pub fn decide(&self, row: &QcRow) -> Option<usize> {
        self.rules.iter().position(|r| {
            r.conds()
                .all(|(m, op, t)| cond_holds(op, row.get(m).unwrap_or(f32::NAN), t))
        })
    }

    /// The label the first matching rule assigns, or the default.
    pub fn label_for(&self, row: &QcRow) -> Result<Option<PhyLabelOp>> {
        match self.decide(row) {
            Some(i) => label_op(&self.rules[i].label).map(Some),
            None => self.default.as_deref().map(label_op).transpose(),
        }
    }

    /// Every `(metric, threshold)` the rules compare against (plus the
    /// negative for `|` ops), so a quantiser can keep their decisions.
    pub fn thresholds(&self) -> Vec<(String, f32)> {
        self.rules
            .iter()
            .flat_map(|r| r.conds())
            .flat_map(|(m, op, t)| {
                let neg = op.starts_with('|').then_some(-t);
                std::iter::once(t)
                    .chain(neg)
                    .map(move |t| (m.to_owned(), t))
            })
            .collect()
    }
}

/// `v op threshold`, where an `op` starting with `|` compares `|v|`.
/// `NaN` and unknown ops never hold. Shared by brief flags and rules.
pub fn cond_holds(op: &str, v: f32, threshold: f32) -> bool {
    let (op, v) = match op.strip_prefix('|') {
        Some(op) => (op, v.abs()),
        None => (op, v),
    };
    !v.is_nan() && compare(op, v, threshold).unwrap_or(false)
}

pub(crate) fn compare(op: &str, v: f32, threshold: f32) -> Result<bool> {
    Ok(match op {
        "<" => v < threshold,
        "<=" => v <= threshold,
        ">" => v > threshold,
        ">=" => v >= threshold,
        "==" | "=" => v == threshold,
        "!=" => v != threshold,
        other => {
            bail!("unknown op {other:?}; expected <, <=, >, >=, ==, !=, optionally prefixed by |")
        }
    })
}

pub fn op_name(op: PhyLabelOp) -> &'static str {
    match op {
        PhyLabelOp::SetGood => "good",
        PhyLabelOp::SetMua => "mua",
        PhyLabelOp::SetNoise => "noise",
        PhyLabelOp::SetUnsorted => "unsorted",
    }
}

pub(crate) fn label_op(s: &str) -> Result<PhyLabelOp> {
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

/// A label word as `sorrel_auto.py`'s `norm_label` reads it.
fn norm_label(s: &str) -> Option<&'static str> {
    let s = s.to_lowercase();
    let has = |k: &[&str]| k.iter().any(|k| s.contains(k));
    match () {
        _ if has(&["noise"]) => Some("noise"),
        _ if has(&["mua", "multi"]) => Some("mua"),
        _ if has(&["sua", "good", "single", "neural"]) => Some("good"),
        _ => None,
    }
}

/// What a rule set does to a sort, for a reviewer who judges the policy
/// rather than the units: per-rule counts, label transitions, margin units
/// (a condition within `margin` x IQR of its threshold, so a small nudge
/// would change the decision) and disagreements with `sorrel auto` evidence.
/// `current` is a unit's present label; `evidence` yields its evidence record.
pub fn eval_report(
    set: &RuleSet,
    rows: &[QcRow],
    margin: f32,
    current: impl Fn(u32) -> &'static str,
    evidence: Option<&dyn Fn(u32) -> Option<Value>>,
) -> String {
    use crate::brief::{quantise, rle};
    use std::collections::BTreeMap;
    use std::fmt::Write;
    let live: Vec<&QcRow> = rows.iter().filter(|r| r.n_spikes > 0).collect();
    let n = set.rules.len();
    let iqr: BTreeMap<&str, f32> = set
        .rules
        .iter()
        .flat_map(|r| r.conds())
        .map(|(m, ..)| {
            let mut v: Vec<f32> = live
                .iter()
                .filter_map(|r| r.get(m))
                .filter(|v| v.is_finite())
                .collect();
            v.sort_by(f32::total_cmp);
            let q = |p: f32| v[((v.len() - 1) as f32 * p).round() as usize];
            (m, if v.is_empty() { 0.0 } else { q(0.75) - q(0.25) })
        })
        .collect();
    let all_th = set.thresholds();
    let cond_text = |r: &Rule| {
        r.conds()
            .map(|(m, op, t)| format!("{m}{op}{t}"))
            .collect::<Vec<_>>()
            .join(" & ")
    };
    let th = |m: &str| -> Vec<f32> { all_th.iter().filter(|(k, _)| k == m).map(|e| e.1).collect() };
    let mut decided = vec![0usize; n + 1];
    let mut moves: BTreeMap<(&str, &str), Vec<u32>> = BTreeMap::new();
    let mut unchanged = 0;
    let mut near: Vec<(f32, String)> = Vec::new();
    let mut effective: Vec<(u32, &str, String)> = Vec::new(); // id, label, deciding rule
    for r in &live {
        let id = r.cluster_id;
        let d = set.decide(r);
        decided[d.unwrap_or(n)] += 1;
        let label = d.map(|i| &set.rules[i].label).or(set.default.as_ref());
        let label = label
            .and_then(|l| label_op(l).ok())
            .map_or_else(|| current(id), op_name);
        let by = d.map_or("default".to_owned(), |i| set.name(i));
        match label == current(id) {
            true => unchanged += 1,
            false => moves.entry((current(id), label)).or_default().push(id),
        }
        // A condition is in the margin if it is close to its threshold and
        // flipping it would change the outcome: a held condition of the
        // deciding rule, or a failing one of an earlier rule.
        let mut best: Option<(f32, String)> = None;
        for (i, rule) in set.rules.iter().enumerate().take(d.map_or(n, |d| d + 1)) {
            for (m, op, t) in rule.conds() {
                let v = r.get(m).unwrap_or(f32::NAN);
                let holds = cond_holds(op, v, t);
                let x = if op.starts_with('|') { v.abs() } else { v };
                let ratio = (x - t).abs() / iqr[m].max(f32::MIN_POSITIVE);
                if x.is_finite()
                    && ratio <= margin
                    && (holds == (Some(i) == d))
                    && best.as_ref().is_none_or(|b| ratio < b.0)
                {
                    best = Some((
                        ratio,
                        format!("{id}\t{op}{m}\t{}\t{t}\t{by}->{label}", quantise(v, &th(m))),
                    ));
                }
            }
        }
        near.extend(best);
        effective.push((id, label, by));
    }
    let mut out = format!(
        "# rules_eval: {} live units, margin {margin} x IQR\n",
        live.len()
    );
    for (i, rule) in set.rules.iter().enumerate() {
        let _ = writeln!(
            out,
            "{}: {} -> {}: {}",
            set.name(i),
            cond_text(rule),
            rule.label,
            decided[i]
        );
    }
    let _ = writeln!(
        out,
        "default{}: {}",
        set.default
            .as_deref()
            .map_or(" (keep label)".to_owned(), |d| format!(" -> {d}")),
        decided[n]
    );
    let _ = writeln!(out, "transitions (unchanged {unchanged}):");
    for ((from, to), ids) in &moves {
        let _ = writeln!(out, "{from}->{to} {}: {}", ids.len(), rle(ids));
    }
    near.sort_by(|a, b| a.0.total_cmp(&b.0));
    let _ = writeln!(
        out,
        "margin ({} units; id, metric, value, threshold, decided):",
        near.len()
    );
    near.iter()
        .take(40)
        .for_each(|(_, l)| out.push_str(&format!("{l}\n")));
    if near.len() > 40 {
        let _ = writeln!(out, "(+{} more)", near.len() - 40);
    }
    for (src, ev) in ["bombcell", "unitrefine"]
        .into_iter()
        .filter_map(|s| Some((s, evidence?)))
    {
        let off: Vec<String> = effective
            .iter()
            .filter_map(|(id, label, by)| {
                let theirs = ev(*id)?.get(src)?.as_str().and_then(norm_label)?;
                (theirs != *label).then(|| format!("{id} {by}:{label} vs {theirs}"))
            })
            .collect();
        let _ = writeln!(out, "disagree {src} ({}):", off.len());
        off.iter()
            .take(40)
            .for_each(|l| out.push_str(&format!("{l}\n")));
        if off.len() > 40 {
            let _ = writeln!(out, "(+{} more)", off.len() - 40);
        }
    }
    out
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
            ..QcRow::default()
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

    #[test]
    fn name_all_and_abs_ops_parse_and_gate() {
        let set = RuleSet::parse(
            r#"
            [[rule]]
            name = "drifty"
            metric = "n_spikes"
            op = ">"
            value = 10
            label = "mua"
            all = [{ metric = "drift_correlation", op = "|>", value = 0.5 }]
            [[rule]]
            metric = "n_spikes"
            op = ">"
            value = 0
            label = "good"
            "#,
        )
        .unwrap();
        assert_eq!((set.name(0), set.name(1)), ("drifty".into(), "r1".into()));
        let mut r = row(0, 50, 0.0);
        r.drift_correlation = -0.7; // |-0.7| > 0.5
        assert_eq!(set.decide(&r), Some(0));
        r.drift_correlation = -0.2;
        assert_eq!(set.decide(&r), Some(1));
        r.drift_correlation = f32::NAN;
        assert_eq!(set.decide(&r), Some(1), "NaN fails the extra condition");
        assert!(set
            .thresholds()
            .contains(&("drift_correlation".into(), -0.5)));
        assert!(RuleSet::parse("[[rule]]\nmetric = \"n_spikes\"\nop = \"<\"\nvalue = 1\nlabel = \"good\"\nall = [{ metric = \"x\", op = \"<\", value = 1 }]\n").is_err());
        assert!(cond_holds("|<", -0.2, 0.5) && !cond_holds("|>", f32::NAN, 0.0));
        let one = |op: &str, v: &str| {
            RuleSet::parse(&format!(
                "[[rule]]\nmetric = \"firing_rate\"\nop = \"{op}\"\nvalue = {v}\nlabel = \"good\"\n"
            ))
        };
        assert!(one("|>", "1").is_ok());
        assert!(one("||>", "1").is_err(), "only one leading bar");
        assert!(one(">", "nan").is_err() && one(">", "inf").is_err());
    }
}
