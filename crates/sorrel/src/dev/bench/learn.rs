//! `sorrel dev bench --bench BENCH learn-dataset`: extract the learned
//! curator's training tables from dev sessions.
//!
//! Dev sessions come from the native sealed map only (exactly as `coverage`),
//! so a holdout id is unreachable. Each session is copied to a disposable
//! directory under BENCH, opened and seeded, every *input-only* feature and
//! candidate is computed, and only then is the truth read and aligned to
//! session spike order. Sessions run in parallel up to `--jobs`; each writes
//! one shard per table, and the final tables are concatenated in session
//! order and row-sorted, so the output bytes do not depend on `--jobs`.
//!
//! A shard is skipped when all five of its tables exist with the current
//! [`EXTRACTOR_VERSION`], which makes an interrupted 182-session run resumable.
//!
//! Bounds, so one session's cost stays predictable: roots with no spikes are
//! omitted; only roots with at least 100 spikes get candidates, spike rows and
//! pair references; candidates are deduplicated and capped at 60 per root;
//! pair similarity is evaluated from roots with at least 100 spikes; and the
//! spike rows are a deterministic time-stratified subsample.

use super::coverage::{aligned_truth, dev_sessions};
use super::{Bench, LearnDatasetArgs};
use anyhow::{ensure, Context, Result};
use sorrel_data::learn::features::{self, ExtractOpts, Truth};
use sorrel_data::learn::table::{append_remapped, Table};
use sorrel_data::learn::EXTRACTOR_VERSION;
use sorrel_data::{Journal, Session};
use sorrel_io::kilosort::{read_cluster_group_tsv, KilosortProvider};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// The five tables, in output order.
const TABLES: [&str; 5] = ["roots", "candidates", "pairs", "units", "spikes"];

struct SessionStat {
    sid: String,
    roots: usize,
    candidates: usize,
    pairs: usize,
    units: usize,
    spikes: usize,
    elapsed: Duration,
}

impl Bench {
    pub(super) fn learn_dataset(&self, a: &LearnDatasetArgs) -> Result<()> {
        ensure!(a.jobs >= 1, "--jobs must be at least 1");
        ensure!(
            a.max_spikes_per_root > 0,
            "--max-spikes-per-root must be positive"
        );
        let limit = a.dev_limit.unwrap_or(usize::MAX);
        let sids: Vec<String> = dev_sessions(&self.map, limit)?
            .into_iter()
            .map(str::to_owned)
            .collect();
        let out = &a.out;
        fs::create_dir_all(out.join("shards"))?;
        let opts = ExtractOpts {
            max_spikes_per_root: a.max_spikes_per_root,
            seed: a.seed,
        };

        let pending: Vec<usize> = (0..sids.len())
            .filter(|&i| !shard_ready(&shard_dir(out, &sids[i])))
            .collect();
        println!(
            "learn-dataset: {} dev session(s), {} already extracted, {} to do, {} job(s), extractor {EXTRACTOR_VERSION}",
            sids.len(),
            sids.len() - pending.len(),
            pending.len(),
            a.jobs
        );

        let results: Mutex<Vec<(usize, Result<SessionStat>)>> = Mutex::new(Vec::new());
        let next = AtomicUsize::new(0);
        if !pending.is_empty() {
            std::thread::scope(|scope| {
                for _ in 0..a.jobs.min(pending.len()) {
                    scope.spawn(|| loop {
                        let slot = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&index) = pending.get(slot) else {
                            break;
                        };
                        let sid = &sids[index];
                        let started = Instant::now();
                        let result =
                            process_session(self, sid, out, &opts).map(|stat| SessionStat {
                                sid: sid.clone(),
                                elapsed: started.elapsed(),
                                ..stat
                            });
                        results.lock().unwrap().push((index, result));
                    });
                }
            });
        }
        let mut results = results.into_inner().unwrap();
        results.sort_by_key(|(i, _)| *i);
        for (_, result) in &results {
            match result {
                Ok(stat) => println!(
                    "dev {}: roots {} candidates {} pairs {} units {} spikes {} in {:.1}s",
                    stat.sid,
                    stat.roots,
                    stat.candidates,
                    stat.pairs,
                    stat.units,
                    stat.spikes,
                    stat.elapsed.as_secs_f64()
                ),
                Err(err) => println!("dev ?: FAILED: {err:#}"),
            }
        }
        if let Some((_, Err(err))) = results.into_iter().find(|(_, r)| r.is_err()) {
            return Err(err);
        }

        // Concatenate in session order and sort rows, so jobs cannot reorder
        // the output. A shard from another extractor version is refused rather
        // than silently mixed in.
        for name in TABLES {
            let mut all: Option<Table> = None;
            for (idx, sid) in sids.iter().enumerate() {
                let shard = Table::load(&shard_dir(out, sid), name)
                    .with_context(|| format!("load {sid} {name} shard"))?;
                ensure!(
                    shard.extractor_version == EXTRACTOR_VERSION,
                    "{sid} {name} shard was extracted with {:?}, expected {EXTRACTOR_VERSION}",
                    shard.extractor_version
                );
                let table = all.get_or_insert_with(|| {
                    let mut t = Table::new(EXTRACTOR_VERSION, shard.columns.clone());
                    t.codes = shard.codes.clone();
                    t.notes = shard.notes.clone();
                    t.sids = sids.clone();
                    t
                });
                append_remapped(table, &shard, &[idx as f32])?;
            }
            let Some(mut table) = all else {
                continue;
            };
            sort_rows(&mut table, row_keys(name));
            table.save(out, name)?;
            let (_, data) = sorrel_data::learn::table::paths(out, name);
            println!(
                "{name}: {} rows × {} columns, {} bytes",
                table.n_rows(),
                table.columns.len(),
                fs::metadata(&data).map(|m| m.len()).unwrap_or(0)
            );
        }
        println!("{}", out.display());
        Ok(())
    }
}

fn row_keys(table: &str) -> &'static [&'static str] {
    match table {
        "roots" => &["sid_idx", "root"],
        "candidates" => &["sid_idx", "root", "cand_idx"],
        "pairs" => &["sid_idx", "root_a", "root_b"],
        "units" => &["sid_idx", "unit"],
        "spikes" => &["sid_idx", "root", "local_idx"],
        _ => &["sid_idx"],
    }
}

/// Stable row sort by integer id columns (`NaN` sorts last).
fn sort_rows(table: &mut Table, keys: &[&str]) {
    let columns: Vec<usize> = keys.iter().filter_map(|k| table.column(k)).collect();
    let stride = table.columns.len();
    let n = table.n_rows();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        for &c in &columns {
            let (x, y) = (table.data[a * stride + c], table.data[b * stride + c]);
            let ord = match (x.is_finite(), y.is_finite()) {
                (true, true) => x.total_cmp(&y),
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                (false, false) => std::cmp::Ordering::Equal,
            };
            if ord != std::cmp::Ordering::Equal {
                return ord;
            }
        }
        a.cmp(&b)
    });
    let sorted: Vec<f32> = order
        .iter()
        .flat_map(|&i| table.data[i * stride..(i + 1) * stride].iter().copied())
        .collect();
    table.data = sorted;
}

fn shard_dir(out: &Path, sid: &str) -> PathBuf {
    out.join("shards").join(sid)
}

fn shard_ready(dir: &Path) -> bool {
    TABLES
        .iter()
        .all(|name| Table::version_matches(dir, name, EXTRACTOR_VERSION))
}

/// One session: input-only extraction first, truth afterwards.
fn process_session(
    bench: &Bench,
    sid: &str,
    out: &Path,
    opts: &ExtractOpts,
) -> Result<SessionStat> {
    let started = Instant::now();
    let temp = tempfile::Builder::new()
        .prefix("learn-")
        .tempdir_in(&bench.root)?;
    let data = temp.path().join("input");
    bench.copy_input(sid, &data)?;
    let provider = KilosortProvider::open(&data, Default::default())?;
    let mut session = Session::new(provider, Journal::open(&temp.path().join("learn.journal"))?);
    session.seed_pc_features();
    session.seed_templates();
    session.seed_template_waveforms();
    session.seed_amplitudes();
    let seeded = Instant::now();

    // Every feature, candidate and proposer output below is input-only. No
    // truth file is opened until this call returns.
    let mut extracted = features::extract(&session, sid, opts)?;
    let extracted_at = Instant::now();
    eprintln!(
        "dev {sid}: seeded {:.1}s, input-only extract {:.1}s",
        seeded.duration_since(started).as_secs_f64(),
        extracted_at.duration_since(seeded).as_secs_f64()
    );

    // Truth now: align NPY rows to session spike order through the validated
    // permutation and the input-root crosscheck.
    let truth_dir = bench.session(sid).join("truth");
    let truth_rows = sorrel_io::npy::read_1d_u32(&truth_dir.join("spike_clusters.npy"))?;
    let input_roots = sorrel_io::npy::read_1d_u32(&data.join("spike_clusters.npy"))?;
    let rows: Vec<_> = session
        .cluster_index()
        .spike_pc_indices()
        .iter()
        .map(|row| row.get())
        .collect();
    let session_roots: Vec<u32> = session
        .spike_clusters_global()
        .iter()
        .map(|c| c.0)
        .collect();
    let cluster = aligned_truth(&rows, &truth_rows, &input_roots, &session_roots)?;
    let labels = read_cluster_group_tsv(&truth_dir.join("cluster_group.tsv"))
        .context("dev truth labels missing or unreadable")?;
    features::label(&mut extracted, &Truth { cluster, labels })?;
    eprintln!(
        "dev {sid}: truth label {:.1}s",
        Instant::now().duration_since(extracted_at).as_secs_f64()
    );

    let shard = shard_dir(out, sid);
    fs::create_dir_all(&shard)?;
    let t = &extracted.tables;
    t.roots.save(&shard, "roots")?;
    t.candidates.save(&shard, "candidates")?;
    t.pairs.save(&shard, "pairs")?;
    t.units.save(&shard, "units")?;
    t.spikes.save(&shard, "spikes")?;
    Ok(SessionStat {
        sid: sid.to_owned(),
        roots: t.roots.n_rows(),
        candidates: t.candidates.n_rows(),
        pairs: t.pairs.n_rows(),
        units: t.units.n_rows(),
        spikes: t.spikes.n_rows(),
        elapsed: Duration::ZERO,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::bench::{SessionEntry, SessionMap, Split};

    fn entry(split: Split) -> SessionEntry {
        SessionEntry {
            path: "unreadable".into(),
            split,
            forced_dev: false,
            pure_manual: false,
            keep: Vec::new(),
            fingerprint: String::new(),
        }
    }

    #[test]
    fn learn_dataset_session_selection_cannot_read_holdout() {
        let mut map = SessionMap::default();
        for (sid, split) in [
            ("h-sealed-a", Split::Holdout),
            ("dev-b", Split::Dev),
            ("h-sealed-c", Split::Holdout),
            ("dev-d", Split::Dev),
        ] {
            map.sessions.insert(sid.into(), entry(split));
        }
        let picked = dev_sessions(&map, usize::MAX).unwrap();
        assert_eq!(picked, vec!["dev-b", "dev-d"]);
        assert!(picked.iter().all(|s| !s.starts_with("h-sealed")));
        assert!(dev_sessions(&map, 1).unwrap() == vec!["dev-b"]);
        // The dataset only ever asks for dev ids, so a holdout directory is
        // never opened even if it exists.
        assert!(dev_sessions(&map, 0).is_err());
    }

    #[test]
    fn a_stale_or_partial_shard_is_not_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let shard = dir.path();
        let mut t = Table::new(EXTRACTOR_VERSION, vec!["sid_idx".into(), "root".into()]);
        t.sids = vec!["s".into()];
        t.push_row(&[0.0, 1.0]).unwrap();
        assert!(!shard_ready(shard));
        t.save(shard, "roots").unwrap();
        assert!(!shard_ready(shard), "only one of five tables exists");
        for name in TABLES {
            t.save(shard, name).unwrap();
        }
        assert!(shard_ready(shard));
        assert_eq!(
            Table::load(shard, "roots").unwrap().extractor_version,
            EXTRACTOR_VERSION
        );
    }
}
