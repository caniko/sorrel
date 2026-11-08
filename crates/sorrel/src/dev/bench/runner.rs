//! Fresh-copy execution and fixed-binary scoring for the native bench.

use super::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

fn variant_env(variant: &Path) -> Vec<(String, Option<String>)> {
    [
        ("SORREL_RUBRIC", "rubric.md"),
        ("SORREL_REVIEW", "review.md"),
    ]
    .into_iter()
    .map(|(key, name)| {
        let path = variant.join(name);
        (
            key.to_owned(),
            path.is_file().then(|| path.to_string_lossy().into_owned()),
        )
    })
    .collect()
}

fn environment(command: &mut Command, env: &[(String, Option<String>)]) {
    for (key, value) in env {
        if let Some(value) = value {
            command.env(key, value);
        } else {
            command.env_remove(key);
        }
    }
}

pub(super) fn file_hash(path: &Path) -> Result<String> {
    let mut h = blake3::Hasher::new();
    let mut f = fs::File::open(path)?;
    let mut buf = vec![0; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().to_hex().to_string())
}

fn same_completed_run(record: &Value, identity: &Value) -> bool {
    record["status"] == "ok"
        && identity
            .as_object()
            .is_some_and(|fields| fields.iter().all(|(key, value)| &record[key] == value))
}

fn review_turns_ok(turns: u64, plumbing_check: bool) -> bool {
    turns >= if plumbing_check { 4 } else { 1 }
}

/// Bounded session-level parallelism. On an error, finish/account for work
/// already in flight but start no further sessions or repeats. Per-run caps
/// and timeouts still apply independently to each active subprocess.
fn run_sessions<T: Sync>(
    sessions: &[T],
    jobs: usize,
    work: impl Fn(&T, &AtomicBool) -> Result<()> + Sync,
) -> Result<()> {
    ensure!((1..=8).contains(&jobs), "--jobs must be in 1..=8");
    let next = AtomicUsize::new(0);
    let stopped = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..jobs.min(sessions.len()))
            .map(|_| {
                scope.spawn(|| -> Result<()> {
                    loop {
                        if stopped.load(Ordering::Acquire) {
                            return Ok(());
                        }
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(session) = sessions.get(index) else {
                            return Ok(());
                        };
                        if stopped.load(Ordering::Acquire) {
                            return Ok(());
                        }
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            work(session, &stopped)
                        }))
                        .unwrap_or_else(|_| Err(anyhow::anyhow!("bench session worker panicked")));
                        if result.is_err() {
                            // Preserve the originating failure, not a sibling's
                            // later "collection stopped" cancellation message.
                            return if stopped.swap(true, Ordering::AcqRel) {
                                Ok(())
                            } else {
                                result
                            };
                        }
                    }
                })
            })
            .collect();
        let mut first_error = None;
        for worker in workers {
            let result = worker
                .join()
                .unwrap_or_else(|_| Err(anyhow::anyhow!("bench worker panicked")));
            if let Err(error) = result {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    })
}

impl Bench {
    fn input_hashes(&self, sid: &str) -> Result<BTreeMap<String, String>> {
        self.map.sessions[sid]
            .keep
            .iter()
            .map(|name| {
                Ok((
                    name.clone(),
                    file_hash(&self.session(sid).join("input").join(name))?,
                ))
            })
            .collect()
    }

    pub(super) fn run_arm(&self, args: &BenchCmd) -> Result<()> {
        let BenchCmd::Run {
            arm,
            variant,
            split,
            repeats,
            jobs,
            backend,
            model,
            dev_limit,
            plumbing_check,
            sorrel_rev,
            max_budget_usd,
            timeout,
        } = args
        else {
            unreachable!()
        };
        ensure!(
            *repeats > 0 && *timeout > 0,
            "repeats and timeout must be positive"
        );
        ensure!((1..=8).contains(jobs), "--jobs must be in 1..=8");
        ensure!(
            max_budget_usd.is_finite() && *max_budget_usd > 0.0,
            "per-run cap must be finite and positive"
        );
        ensure!(
            sorrel_rev.len() >= 7 && sorrel_rev.bytes().all(|b| b.is_ascii_hexdigit()),
            "--sorrel-rev must name a commit"
        );
        ensure!(
            *split == Split::Dev || dev_limit.is_none(),
            "--dev-limit is dev-only"
        );
        ensure!(
            !*plumbing_check || (*split == Split::Dev && matches!(arm, Arm::McpReview)),
            "--plumbing-check requires the dev MCP arm"
        );
        ensure!(
            !matches!(backend, process::Backend::Opencode) || model.is_some(),
            "OpenCode requires --model provider/model"
        );
        let variant = variant.canonicalize().context("variant directory")?;
        let name = variant
            .file_name()
            .and_then(|n| n.to_str())
            .context("variant name")?;
        ensure!(
            name.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_".contains(&b)),
            "unsafe variant name"
        );
        let mut sessions: Vec<_> = self
            .map
            .sessions
            .iter()
            .filter(|(_, s)| s.split == *split)
            .map(|(id, _)| id.clone())
            .collect();
        if let Some(limit) = dev_limit {
            sessions.truncate(*limit);
        }
        ensure!(!sessions.is_empty(), "no sessions selected");
        let binary_hash = file_hash(&self.sorrel)?;
        let env = variant_env(&variant);
        let variant_text: Vec<_> = env
            .iter()
            .map(|(key, path)| Ok((key, path.as_ref().map(fs::read_to_string).transpose()?)))
            .collect::<Result<_>>()?;
        let variant_hash = blake3::hash(&serde_json::to_vec(&variant_text)?)
            .to_hex()
            .to_string();
        let previous: Vec<Value> = self
            .run_dirs(*split)?
            .iter()
            .map(|dir| read_json(&dir.join("run.json")))
            .collect::<Result<_>>()?;
        let duration = Duration::from_secs(*timeout);
        let mut prompt_command = self.command(false);
        environment(&mut prompt_command, &env);
        prompt_command.args(["agent-guide", "--prompt", "review"]);
        let prompt = process::checked(&mut prompt_command, duration)?;
        let prompt_rev = blake3::hash(prompt.as_bytes()).to_hex().to_string();
        if self.db.is_some() {
            let _ = self.db_command()?;
        }
        run_sessions(&sessions, *jobs, |sid, stopped| {
            if *split == Split::Dev {
                self.record_session(sid)?;
            }
            for repeat in 0..*repeats {
                ensure!(
                    !stopped.load(Ordering::Acquire),
                    "parallel collection stopped after another run failed"
                );
                let identity = json!({"session":sid,"split":split.name(),"arm":arm.name(),
                    "variant":name,"repeat":repeat,"model":model,"sorrel_rev":sorrel_rev,
                    "binary_blake3":binary_hash,"backend":backend.name(),"variant_blake3":variant_hash});
                if previous.iter().any(|record| {
                    same_completed_run(record, &identity)
                        && (!*plumbing_check
                            || record["num_turns"]
                                .as_u64()
                                .is_some_and(|n| review_turns_ok(n, true)))
                }) {
                    continue;
                }
                ensure!(
                    file_hash(&self.sorrel)? == binary_hash,
                    "arm binary changed during the round"
                );
                let rid = format!("{}-{name}-{}", arm.name(), opaque());
                let root = self.home(*split).join("runs").join(&rid);
                fs::create_dir_all(&root)?;
                let data = root.join("data");
                self.copy_input(sid, &data)?;
                let before = self.input_hashes(sid)?;
                let mut record = identity;
                record["id"] = json!(rid);
                record["prompt_rev"] = json!(prompt_rev);
                record["status"] = json!("running");
                record["cost_usd"] = Value::Null;
                record["plumbing_check"] = json!(plumbing_check);
                record["runner_jobs"] = json!(jobs);
                write_json(&root.join("run.json"), &record)?;
                let start = Instant::now();
                let outcome = (|| -> Result<()> {
                    let mut cards = self.command(false);
                    environment(&mut cards, &env);
                    cards
                        .arg("call")
                        .arg(&data)
                        .arg("brief")
                        .arg(r#"{"level":2,"budget":100000}"#);
                    fs::write(
                        root.join("input-brief.txt"),
                        process::checked(&mut cards, duration)?,
                    )?;
                    match arm {
                        Arm::McpReview => {
                            let review = process::review(
                                *backend,
                                &root,
                                &self.sorrel,
                                &data,
                                &prompt,
                                model.as_deref(),
                                *max_budget_usd,
                                duration,
                                &env,
                            )?;
                            record["cost_usd"] = json!(review.cost);
                            record["num_turns"] = json!(review.turns);
                            record["tool_calls"] = json!(review.tool_calls);
                            record["usage"] = review.usage;
                            if let Some(error) = review.error {
                                bail!("{error}");
                            }
                            ensure!(
                                review_turns_ok(review.turns, *plumbing_check),
                                "review has no completed model turn or failed the explicit four-turn plumbing check"
                            );
                            if matches!(backend, process::Backend::Opencode) {
                                ensure!(
                                    review.tool_calls > 0,
                                    "plumbing check failed: no sorrel tool calls"
                                );
                            } else {
                                ensure!(
                                    data.join("sorrel.journal").is_file(),
                                    "plumbing check failed: no session journal"
                                );
                            }
                            let mut save = self.command(false);
                            save.arg("call").arg(&data).arg("save");
                            environment(&mut save, &env);
                            process::checked(&mut save, duration)?;
                        }
                        Arm::AutoLlm => {
                            let mut command = self.command(false);
                            command
                                .current_dir(&root)
                                .env("PWD", &root)
                                .env_remove("OPENCODE_SESSION_ID");
                            if matches!(backend, process::Backend::Opencode) {
                                write_json(&root.join("opencode.json"), &process::auto_config())?;
                            }
                            environment(&mut command, &env);
                            command
                                .arg("auto")
                                .arg(&data)
                                .arg(format!("--llm={}", backend.name()))
                                .arg("--save");
                            if let Some(model) = model {
                                command.arg("--llm-model").arg(model);
                            }
                            process::checked_logged(&mut command, duration, &root)?;
                            let evidence = read_json(&data.join("sorrel_auto/evidence.json"))?;
                            for key in ["prompt_rev", "learned_rev"] {
                                if !evidence[key].is_null() {
                                    record[key] = evidence[key].clone();
                                }
                            }
                            // The sidecar does not currently expose billing. Unknown is
                            // deliberately not recorded as a zero-cost run.
                        }
                    }
                    Ok(())
                })();
                record["wall_s"] = json!(start.elapsed().as_secs_f64());
                record["status"] = json!(if outcome.is_ok() { "ok" } else { "failed" });
                if let Err(error) = outcome {
                    record["error"] = json!(format!("{error:#}"));
                }
                if self.input_hashes(sid)? != before {
                    record["status"] = json!("input-mutated");
                    record["error"] = json!("a run mutated the sealed input");
                }
                write_json(&root.join("run.json"), &record)?;
                if *split == Split::Dev && self.db.is_some() {
                    let mut command = self.db_command()?;
                    command
                        .arg("record-run")
                        .arg("--run-dir")
                        .arg(&root)
                        .arg("--variant-dir")
                        .arg(&variant);
                    process::checked(&mut command, duration)?;
                }
                if *split == Split::Dev {
                    println!(
                        "{rid}: {} {:.1}s cost {}",
                        record["status"],
                        start.elapsed().as_secs_f64(),
                        record["cost_usd"]
                    );
                } else {
                    println!("holdout run: {}", record["status"]);
                }
                ensure!(
                    record["status"] == "ok",
                    "run failed; stopped before collecting further data"
                );
            }
            Ok(())
        })
    }

    pub(super) fn score(&self, split: Split, match_jaccard: f64) -> Result<()> {
        ensure!(
            match_jaccard > 0.0 && match_jaccard <= 1.0,
            "match must be in (0, 1]"
        );
        let scorer_hash = file_hash(&self.scorer)?;
        let mut count = 0;
        for root in self.run_dirs(split)? {
            let mut record = read_json(&root.join("run.json"))?;
            if record["status"] != "ok" {
                continue;
            }
            let sid = record["session"].as_str().context("run session")?;
            ensure!(
                self.map.sessions.get(sid).is_some_and(|s| s.split == split),
                "run session has wrong split"
            );
            let session = self.session(sid);
            let cache = root.join("score.json");
            if cache.exists() {
                fs::remove_file(&cache)?;
            }
            ensure!(
                file_hash(&self.scorer)? == scorer_hash,
                "scorer changed during comparison"
            );
            let mut command = self.command(true);
            command
                .arg("diff")
                .arg(session.join("truth"))
                .arg(root.join("data"))
                .arg("--root")
                .arg(session.join("input"))
                .arg("--json")
                .arg("--match")
                .arg(match_jaccard.to_string());
            let disputed = self.home(split).join("disputed").join(format!("{sid}.tsv"));
            if disputed.is_file() {
                command.arg("--disputed").arg(disputed);
            }
            if split == Split::Dev {
                if let Some(db) = self.db.as_ref() {
                    // Build the deny file before the scorer records anything.
                    let _ = self.db_command()?;
                    command
                        .env("SORREL_DEV_DENY", self.root.join(".sealed/dev-deny.txt"))
                        .arg("--record")
                        .arg(record["id"].as_str().context("run id")?)
                        .arg("--db")
                        .arg(db);
                }
            }
            let stdout = process::checked(&mut command, Duration::from_secs(1800))?;
            let score: Value =
                serde_json::from_str(&stdout).context("scorer returned invalid JSON")?;
            record["scorer_blake3"] = json!(scorer_hash);
            write_json(&root.join("run.json"), &record)?;
            write_json(&cache, &score)?;
            count += 1;
        }
        println!("scored {count} {} runs; scorer {scorer_hash}", split.name());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_override_unsets_inherited_prompt() {
        let t = tempfile::tempdir().unwrap();
        let env = variant_env(t.path());
        assert!(env.iter().all(|(_, value)| value.is_none()));
        fs::write(t.path().join("review.md"), "review").unwrap();
        assert!(variant_env(t.path())[1].1.is_some());
    }

    #[test]
    fn resume_requires_success_and_identical_configuration() {
        let identity = json!({"repeat":0,"variant_blake3":"a","binary_blake3":"b"});
        let mut record = identity.clone();
        record["status"] = json!("failed");
        assert!(!same_completed_run(&record, &identity));
        record["status"] = json!("ok");
        assert!(same_completed_run(&record, &identity));
        record["variant_blake3"] = json!("changed");
        assert!(!same_completed_run(&record, &identity));
    }

    #[test]
    fn short_reviews_are_valid_after_the_first_machine_check() {
        assert!(!review_turns_ok(0, false));
        assert!(review_turns_ok(2, false));
        assert!(!review_turns_ok(2, true));
        assert!(review_turns_ok(4, true));
    }

    #[test]
    fn parallel_sessions_visit_each_once_and_obey_the_worker_bound() {
        let items: Vec<_> = (0..6).map(|_| AtomicUsize::new(0)).collect();
        let active = AtomicUsize::new(0);
        let maximum = AtomicUsize::new(0);
        let paired = std::sync::Barrier::new(2);
        run_sessions(&items, 2, |item, _| {
            let count = active.fetch_add(1, Ordering::SeqCst) + 1;
            maximum.fetch_max(count, Ordering::SeqCst);
            paired.wait();
            item.fetch_add(1, Ordering::Relaxed);
            active.fetch_sub(1, Ordering::SeqCst);
            paired.wait();
            Ok(())
        })
        .unwrap();
        assert_eq!(maximum.load(Ordering::Relaxed), 2);
        assert!(items.iter().all(|n| n.load(Ordering::Relaxed) == 1));
    }

    #[test]
    fn worker_errors_and_panics_stop_new_sessions() {
        let calls = AtomicUsize::new(0);
        for panic in [false, true] {
            calls.store(0, Ordering::Relaxed);
            assert!(run_sessions(&[0, 1, 2], 1, |_, _| {
                calls.fetch_add(1, Ordering::Relaxed);
                if panic {
                    panic!("synthetic worker failure");
                }
                bail!("synthetic run failure")
            })
            .is_err());
            assert_eq!(calls.load(Ordering::Relaxed), 1);
        }
        assert!(run_sessions(&[0], 0, |_, _| Ok(())).is_err());
        assert!(run_sessions(&[0], 9, |_, _| Ok(())).is_err());
    }
}
