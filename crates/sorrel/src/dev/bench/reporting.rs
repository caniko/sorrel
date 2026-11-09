//! Dev-only miss reports and paired, aggregate-only holdout comparisons.

use super::*;
use rand::{Rng, SeedableRng};
use std::io::Write;

fn metrics(score: &Value) -> BTreeMap<String, f64> {
    fn walk(value: &Value, prefix: &str, out: &mut BTreeMap<String, f64>) {
        if let Some(object) = value.as_object() {
            for (key, value) in object {
                let key = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                if value.is_object() {
                    walk(value, &key, out);
                } else if let Some(number) = value.as_f64() {
                    out.insert(key, number);
                }
            }
        }
    }
    let mut filtered = score.clone();
    if let Some(object) = filtered.as_object_mut() {
        for key in [
            "misses",
            "n_spikes",
            "n_root_clusters",
            "n_disputed",
            "match_jaccard",
            "min_spikes",
            "split_share",
            "binary",
        ] {
            object.remove(key);
        }
    }
    let mut out = BTreeMap::new();
    walk(&filtered, "", &mut out);
    if let Some(exact) = score["exact_match"].as_bool() {
        out.insert("exact_match".into(), if exact { 1.0 } else { 0.0 });
    }
    for miss in score["misses"].as_array().into_iter().flatten() {
        let category = miss["category"].as_str().unwrap_or("unknown");
        *out.entry(format!("misses.{category}")).or_default() += 1.0;
    }
    out
}

fn cfg(record: &Value, score: &Value) -> String {
    json!([
        record["model"],
        record["prompt_rev"],
        record["sorrel_rev"],
        record["learned_rev"],
        score["match_jaccard"],
        score["min_spikes"],
        score["split_share"],
        score["binary"],
        record["scorer_blake3"]
    ])
    .to_string()
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn summary(values: &[f64]) -> String {
    let m = mean(values);
    let sd = if values.len() > 1 {
        (values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64).sqrt()
    } else {
        0.0
    };
    format!("{m:.4} ± {sd:.4}")
}

#[derive(Default)]
struct GoodUnitPool {
    truth: u64,
    predicted: u64,
    hit_truth: u64,
    hit_test: u64,
    no_truth_runs: usize,
    no_prediction_runs: usize,
}

impl GoodUnitPool {
    fn add(&mut self, score: &Value) -> Result<()> {
        let count = |key: &str| {
            score["good_units"][key]
                .as_u64()
                .with_context(|| format!("missing good-unit counter {key}"))
        };
        let truth = count("n_truth")?;
        let predicted = count("n_test")?;
        self.truth = self
            .truth
            .checked_add(truth)
            .context("pooled truth overflow")?;
        self.predicted = self
            .predicted
            .checked_add(predicted)
            .context("pooled prediction overflow")?;
        self.hit_truth = self
            .hit_truth
            .checked_add(count("hit_truth")?)
            .context("pooled hit overflow")?;
        self.hit_test = self
            .hit_test
            .checked_add(count("hit_test")?)
            .context("pooled hit overflow")?;
        self.no_truth_runs += usize::from(truth == 0);
        self.no_prediction_runs += usize::from(predicted == 0);
        Ok(())
    }

    fn recall(&self) -> Option<f64> {
        (self.truth > 0).then(|| self.hit_truth as f64 / self.truth as f64)
    }
    fn precision(&self) -> Option<f64> {
        (self.predicted > 0).then(|| self.hit_test as f64 / self.predicted as f64)
    }
    fn render(&self) -> String {
        let ratio = |value: Option<f64>| {
            value.map_or_else(
                || "undefined (empty denominator)".into(),
                |v| format!("{v:.4}"),
            )
        };
        format!("Good units pooled over repeated runs: recall {} ({}/{}), precision {} ({}/{}). Zero-good truth on {} runs; no predicted good units on {} runs. These counts repeat each session once per repeat; they are not counts of unique human units.\n",
            ratio(self.recall()), self.hit_truth, self.truth,
            ratio(self.precision()), self.hit_test, self.predicted,
            self.no_truth_runs, self.no_prediction_runs)
    }
}

impl Bench {
    fn scored(&self, split: Split) -> Result<Vec<(Value, Value)>> {
        let mut out = Vec::new();
        for dir in self.run_dirs(split)? {
            let record = read_json(&dir.join("run.json"))?;
            if record["status"] == "ok" && dir.join("score.json").is_file() {
                out.push((record, read_json(&dir.join("score.json"))?));
            }
        }
        Ok(out)
    }

    pub(super) fn report(
        &self,
        split: Split,
        baseline: Option<&str>,
        candidate: Option<&str>,
        dev_limit: Option<usize>,
    ) -> Result<()> {
        if split == Split::Dev {
            return self.report_dev(dev_limit);
        }
        let baseline = baseline.context("holdout report needs --baseline")?;
        let candidate = candidate.context("holdout report needs --candidate")?;
        ensure!(baseline != candidate, "baseline and candidate must differ");
        self.report_holdout(baseline, candidate, dev_limit)
    }

    fn report_dev(&self, dev_limit: Option<usize>) -> Result<()> {
        let mut scored = self.scored(Split::Dev)?;
        if let Some(limit) = dev_limit {
            let selected: BTreeSet<_> = super::coverage::dev_sessions(&self.map, limit)?
                .into_iter()
                .collect();
            scored.retain(|(r, _)| {
                r["session"]
                    .as_str()
                    .is_some_and(|sid| selected.contains(sid))
            });
        }
        ensure!(!scored.is_empty(), "no scored dev runs");
        let mut groups: BTreeMap<String, Vec<(&Value, &Value)>> = BTreeMap::new();
        for (record, score) in &scored {
            let key = format!(
                "{}/{} [{}]",
                record["arm"].as_str().unwrap_or("?"),
                record["variant"].as_str().unwrap_or("?"),
                cfg(record, score)
            );
            groups.entry(key).or_default().push((record, score));
        }
        let mut md =
            String::from("# Dev report\n\nTarget: human good-unit recall and precision = 1.0.\n");
        if let Some(limit) = dev_limit {
            md+=&format!("\nScope: deterministic first {limit} dev sessions by opaque ID, matching run --dev-limit.\n");
        }
        for (label, runs) in groups {
            let mut pooled = GoodUnitPool::default();
            for (_, score) in &runs {
                pooled.add(score)?;
            }
            md += &format!(
                "\n## {label}\n\n{} scored runs\n\n{}\n| metric | mean ± SD |\n|---|---|\n",
                runs.len(),
                pooled.render()
            );
            let rows: Vec<_> = runs.iter().map(|(_, score)| metrics(score)).collect();
            let keys: BTreeSet<_> = rows.iter().flat_map(|row| row.keys().cloned()).collect();
            for key in keys {
                let values: Vec<_> = rows
                    .iter()
                    .filter_map(|r| {
                        r.get(&key)
                            .copied()
                            .or_else(|| key.starts_with("misses.").then_some(0.0))
                    })
                    .collect();
                md += &format!("| {key} | {} |\n", summary(&values));
            }
            let mut misses: BTreeMap<(String, String, String), usize> = BTreeMap::new();
            for (record, score) in &runs {
                for miss in score["misses"].as_array().into_iter().flatten() {
                    *misses
                        .entry((
                            miss["category"].as_str().unwrap_or("?").to_owned(),
                            record["session"].as_str().unwrap_or("?").to_owned(),
                            miss["roots"].to_string(),
                        ))
                        .or_default() += 1;
                }
            }
            md += "\n### Misses and the input cards\n";
            for ((category, sid, roots), count) in misses {
                md += &format!("\n- {category}: session {sid}, roots {roots} ({count} runs)\n");
                let record = runs
                    .iter()
                    .find(|(r, _)| r["session"] == sid)
                    .context("missing run for cards")?
                    .0;
                let rid = record["id"].as_str().context("run id")?;
                let initial = fs::read_to_string(self.root.join("runs").join(rid).join("input-brief.txt"))
                    .context("missing revision-pinned input cards; rerun rather than regenerate with another binary")?;
                let ids: Vec<u32> = serde_json::from_str(&roots)?;
                let cards = initial
                    .lines()
                    .filter(|line| {
                        let id = line.split('\t').next().unwrap_or("");
                        id == "id" || id.parse::<u32>().is_ok_and(|id| ids.contains(&id))
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                md += &format!("\n```text\n{cards}\n```\n");
            }
        }
        let out = self
            .root
            .join("reports")
            .join(format!("dev-{}-{}.md", now(), &opaque()[..8]));
        fs::create_dir_all(out.parent().unwrap())?;
        fs::write(&out, md)?;
        println!("{}", out.display());
        Ok(())
    }

    fn report_holdout(
        &self,
        baseline: &str,
        candidate: &str,
        dev_limit: Option<usize>,
    ) -> Result<()> {
        // Log every query, including refusals, before looking at results.
        let query = opaque();
        let mut ledger = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.root.join(".sealed/holdout-ledger.jsonl"))?;
        writeln!(
            ledger,
            "{}",
            json!({"query":query,"time":now(),"baseline":baseline,"candidate":candidate})
        )?;
        ledger.sync_all()?;
        ensure!(
            dev_limit.is_none(),
            "--dev-limit is dev-only; this refused holdout query was ledgered"
        );
        let scored = self.scored(Split::Holdout)?;
        let arms: BTreeSet<_> = scored
            .iter()
            .filter_map(|(r, _)| r["arm"].as_str().map(str::to_owned))
            .collect();
        let mut lines = Vec::new();
        let mut aggregates = Vec::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        for arm in arms {
            let runs: Vec<_> = scored
                .iter()
                .filter(|(r, _)| {
                    r["arm"] == arm && (r["variant"] == baseline || r["variant"] == candidate)
                })
                .collect();
            let categories: BTreeSet<_> = runs
                .iter()
                .flat_map(|(_, s)| metrics(s).into_keys().filter(|k| k.starts_with("misses.")))
                .collect();
            let mut configs = BTreeMap::<String, BTreeSet<String>>::new();
            let mut options = BTreeSet::new();
            let mut by_session: BTreeMap<(String, String), BTreeMap<String, Vec<f64>>> =
                BTreeMap::new();
            for (r, s) in runs {
                let variant = r["variant"].as_str().context("variant")?.to_owned();
                configs
                    .entry(variant.clone())
                    .or_default()
                    .insert(cfg(r, s));
                options.insert(
                    json!([
                        s["match_jaccard"],
                        s["min_spikes"],
                        s["split_share"],
                        s["binary"],
                        r["scorer_blake3"]
                    ])
                    .to_string(),
                );
                let mut row = metrics(s);
                for cat in &categories {
                    row.entry(cat.clone()).or_default();
                }
                let acc = by_session
                    .entry((
                        variant,
                        r["session"].as_str().context("session")?.to_owned(),
                    ))
                    .or_default();
                for (key, value) in row {
                    acc.entry(key).or_default().push(value);
                }
            }
            ensure!(
                configs.values().all(|set| set.len() == 1),
                "mixed revisions in a holdout variant"
            );
            ensure!(
                options.len() <= 1,
                "holdout comparisons need identical scorer and options"
            );
            let base_sessions: BTreeSet<_> = by_session
                .keys()
                .filter(|(v, _)| v == baseline)
                .map(|(_, sid)| sid.clone())
                .collect();
            let candidate_sessions: BTreeSet<_> = by_session
                .keys()
                .filter(|(v, _)| v == candidate)
                .map(|(_, sid)| sid.clone())
                .collect();
            ensure!(
                base_sessions == candidate_sessions,
                "incomplete holdout pairing; score both variants on every selected session"
            );
            let common: Vec<_> = by_session
                .keys()
                .filter(|(v, s)| {
                    v == baseline && by_session.contains_key(&(candidate.to_owned(), s.clone()))
                })
                .map(|(_, s)| s.clone())
                .collect();
            if common.is_empty() {
                continue;
            }
            for sid in &common {
                let base = &by_session[&(baseline.to_owned(), sid.clone())];
                let cand = &by_session[&(candidate.to_owned(), sid.clone())];
                ensure!(
                    base.keys().eq(cand.keys())
                        && base
                            .iter()
                            .all(|(key, values)| values.len() == cand[key].len()),
                    "holdout variants have different metrics or repeat counts"
                );
            }
            lines.push(format!("{arm}: {} paired holdout sessions", common.len()));
            let keys: BTreeSet<_> = common
                .iter()
                .flat_map(|s| {
                    by_session[&(baseline.to_owned(), s.clone())]
                        .keys()
                        .cloned()
                })
                .collect();
            for key in keys {
                let pairs: Vec<_> = common
                    .iter()
                    .filter_map(|s| {
                        Some((
                            mean(
                                by_session
                                    .get(&(baseline.to_owned(), s.clone()))?
                                    .get(&key)?,
                            ),
                            mean(
                                by_session
                                    .get(&(candidate.to_owned(), s.clone()))?
                                    .get(&key)?,
                            ),
                        ))
                    })
                    .collect();
                if pairs.is_empty() {
                    continue;
                }
                let deltas: Vec<_> = pairs.iter().map(|(b, c)| c - b).collect();
                let base = mean(&pairs.iter().map(|(b, _)| *b).collect::<Vec<_>>());
                let cand = mean(&pairs.iter().map(|(_, c)| *c).collect::<Vec<_>>());
                let mut boot: Vec<_> = (0..2000)
                    .map(|_| {
                        mean(
                            &(0..deltas.len())
                                .map(|_| deltas[rng.random_range(0..deltas.len())])
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect();
                boot.sort_by(f64::total_cmp);
                let (lo, hi) = if pairs.len() >= 3 {
                    (Some(boot[50]), Some(boot[1949]))
                } else {
                    (None, None)
                };
                lines.push(format!(
                    "  {key}: base {base:.4}, candidate {cand:.4}, delta {:+.4}, CI {lo:?}..{hi:?}",
                    cand - base
                ));
                aggregates.push(json!({"query":query,"baseline":baseline,"candidate":candidate,"arm":arm,"metric":key,"n_sessions":pairs.len(),"base":base,"cand":cand,"delta":cand-base,"ci_lo":lo,"ci_hi":hi}));
            }
        }
        ensure!(!aggregates.is_empty(), "no paired scored holdout runs");
        if self.db.is_some() {
            let path = self
                .root
                .join(".sealed")
                .join(format!("ledger-{query}.json"));
            write_json(&path, &aggregates)?;
            let mut command = self.db_command()?;
            command.arg("ledger").arg("--json").arg(path);
            process::checked(&mut command, Duration::from_secs(600))?;
        }
        println!("{}", lines.join("\n"));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limited_dev_report_uses_the_same_cohort_for_every_variant() {
        let t = tempfile::tempdir().unwrap();
        let args = BenchArgs {
            bench: Some(t.path().to_owned()),
            sorrel: None,
            scorer: None,
            cmd: BenchCmd::Status,
        };
        let mut b = Bench::open(&args, None).unwrap();
        for sid in ["b", "c"] {
            b.map.sessions.insert(
                sid.into(),
                SessionEntry {
                    path: "unused".into(),
                    split: Split::Dev,
                    forced_dev: true,
                    pure_manual: false,
                    keep: vec![],
                    fingerprint: String::new(),
                    recuration_of: None,
                },
            );
        }
        for variant in ["baseline", "candidate"] {
            for (sid, n, hit) in [("b", 1, 1), ("c", 100, 0)] {
                let p = b.root.join("runs").join(format!("{variant}-{sid}"));
                write_json(
                    &p.join("run.json"),
                    &json!({"status":"ok","session":sid,"arm":"mcp-review","variant":variant}),
                )
                .unwrap();
                write_json(&p.join("score.json"),&json!({"good_units":{"n_truth":n,"n_test":hit,"hit_truth":hit,"hit_test":hit},"misses":[],"exact_match":false})).unwrap();
            }
        }
        b.report_dev(Some(1)).unwrap();
        let report = fs::read_dir(b.root.join("reports"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let text = fs::read_to_string(report).unwrap();
        assert!(text.contains("deterministic first 1 dev sessions"));
        assert_eq!(text.matches("1 scored runs").count(), 2);
        assert_eq!(text.matches("recall 1.0000 (1/1)").count(), 2);
        assert!(!text.contains("(1/101)"));
    }

    #[test]
    fn pooled_recall_is_not_inflated_by_empty_truth_sessions() {
        let mut pool = GoodUnitPool::default();
        pool.add(&json!({"good_units":{"n_truth":0,"n_test":0,"hit_truth":0,"hit_test":0}}))
            .unwrap();
        pool.add(&json!({"good_units":{"n_truth":10,"n_test":2,"hit_truth":2,"hit_test":2}}))
            .unwrap();
        assert_eq!(pool.recall(), Some(0.2));
        assert_eq!(pool.precision(), Some(1.0));
        assert_eq!(pool.no_truth_runs, 1);
        assert_eq!(pool.no_prediction_runs, 1);
        assert!(pool.render().contains("not counts of unique human units"));
        let mut empty = GoodUnitPool::default();
        empty
            .add(&json!({"good_units":{"n_truth":3,"n_test":0,"hit_truth":0,"hit_test":0}}))
            .unwrap();
        assert_eq!(
            empty.precision(),
            None,
            "no predictions must not be displayed as perfect precision"
        );
        assert_eq!(empty.recall(), Some(0.0));
        assert!(GoodUnitPool::default().recall().is_none());
    }

    #[test]
    fn metrics_exclude_options_and_count_misses() {
        let row = metrics(&json!({"n_spikes":10,"match_jaccard":0.8,"binary":true,
            "good_units":{"recall":0.5,"precision":1.0},
            "misses":[{"category":"missed_split"},{"category":"missed_split"}]}));
        assert_eq!(row.len(), 3);
        assert_eq!(row["misses.missed_split"], 2.0);
        assert_eq!(row["good_units.recall"], 0.5);
        assert!(!row.contains_key("match_jaccard"));
    }
}
