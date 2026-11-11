//! `sorrel dev bench --bench BENCH learn-dataset`: extract the learned
//! curator's training tables from dev sessions.
//!
//! Dev sessions come from the native sealed map only (exactly as `coverage`),
//! so a holdout id is unreachable. Each session is copied to a disposable
//! directory under BENCH, opened and seeded, every *input-only* feature and
//! candidate is computed, and only then is the truth read and aligned to
//! session spike order. Sessions run in parallel up to `--jobs`; each writes
//! one two-part shard (input-only [`FEATURES_DIR`] and truth-derived
//! [`TARGETS_DIR`]), and the final tables are joined, concatenated in session
//! order and row-sorted, so the output bytes do not depend on `--jobs`.
//!
//! A features shard is reused when it matches the current extractor version,
//! extraction options and input identity: a manifest of every input file the
//! extraction read (with the legacy three-NPY fingerprint kept alongside for
//! pre-manifest shards). A targets shard is reused when it additionally
//! matches [`TARGET_VERSION`], the features key and the session's truth hash.
//! `--retarget` skips extraction entirely: it loads the stored features,
//! attaches the truth and rewrites only the targets, so changing a target
//! costs seconds rather than the proposer suite.
//!
//! Bounds, so one session's cost stays predictable: roots with no spikes are
//! omitted; only roots with at least 100 spikes get candidates, spike rows and
//! pair references; candidates are deduplicated and capped at 60 per root;
//! pair similarity is evaluated from roots with at least 100 spikes; and the
//! spike rows are a deterministic time-stratified subsample.

use super::coverage::dev_sessions;
use super::{Bench, LearnDatasetArgs};
use anyhow::{bail, ensure, Context, Result};
use sorrel_data::learn::features::{self, ExtractOpts};
use sorrel_data::learn::shard::{self, FeatureKey, FEATURES_DIR, TARGETS_DIR};
use sorrel_data::learn::table::{append_remapped, Table};
use sorrel_data::learn::{EXTRACTOR_VERSION, TARGET_VERSION};
use sorrel_data::{Journal, Session};
use sorrel_io::kilosort::KilosortProvider;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// The six output tables, in output order.
const TABLES: [&str; 6] = [
    "roots",
    "candidates",
    "pairs",
    "units",
    "cand_units",
    "spikes",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Action {
    Extract,
    Reuse,
    Retarget,
}

struct Job {
    sid: String,
    action: Action,
    key: FeatureKey,
}

struct SessionStat {
    sid: String,
    action: &'static str,
    counts: Option<[usize; 6]>,
    elapsed: Duration,
}

impl Bench {
    pub(super) fn learn_dataset(&self, a: &LearnDatasetArgs) -> Result<()> {
        ensure!(a.jobs >= 1, "--jobs must be at least 1");
        ensure!(
            a.max_spikes_per_root > 0,
            "--max-spikes-per-root must be positive"
        );
        let out = &a.out;
        // Refuse a writable destination inside the bench's immutable session
        // originals or the sealed holdout home BEFORE the first write, also
        // through junctions/symlinks (the check compares canonical paths).
        // Every destination directory is re-checked before a write, because a
        // junction can appear below the --out root after this check.
        let guard = ensure_out_safe(self, out)?;
        let limit = a.dev_limit.unwrap_or(usize::MAX);
        let sids: Vec<String> = dev_sessions(&self.map, limit)?
            .into_iter()
            .map(str::to_owned)
            .collect();
        guard.prepare_dir(&guard.root().join("shards"))?;
        let opts = ExtractOpts {
            max_spikes_per_root: a.max_spikes_per_root,
            seed: a.seed,
        };

        let mut jobs = Vec::with_capacity(sids.len());
        let mut reused = 0usize;
        for sid in &sids {
            let fdir = guard.shard_dir(sid).join(FEATURES_DIR);
            if a.retarget {
                let key = shard::read_feature_key(&fdir).with_context(|| {
                    format!(
                        "--retarget needs a complete features shard for {sid}; \
                         run without --retarget first"
                    )
                })?;
                ensure!(
                    key.extractor_version == EXTRACTOR_VERSION
                        && key.max_spikes_per_root == opts.max_spikes_per_root
                        && key.seed == opts.seed,
                    "--retarget: {sid} features were extracted with extractor {:?}, \
                     --max-spikes-per-root {} and --seed {}; this run asks for {EXTRACTOR_VERSION}, {} and {}",
                    key.extractor_version,
                    key.max_spikes_per_root,
                    key.seed,
                    opts.max_spikes_per_root,
                    opts.seed
                );
                ensure!(
                    shard::features_match(&fdir, &key),
                    "--retarget: {sid} features shard is incomplete; run without --retarget"
                );
                warn_legacy_key(sid, &key);
                jobs.push(Job {
                    sid: sid.clone(),
                    action: Action::Retarget,
                    key,
                });
            } else {
                let key = feature_key(self, sid, &opts)?;
                let action = if shard::features_match(&fdir, &key) {
                    // Use the key actually stored on disk: a pre-manifest
                    // shard is accepted on its legacy fields, and its targets
                    // are keyed to that legacy features id.
                    let stored = shard::read_feature_key(&fdir)?;
                    warn_legacy_key(sid, &stored);
                    reused += 1;
                    jobs.push(Job {
                        sid: sid.clone(),
                        action: Action::Reuse,
                        key: stored,
                    });
                    continue;
                } else {
                    Action::Extract
                };
                jobs.push(Job {
                    sid: sid.clone(),
                    action,
                    key,
                });
            }
        }
        println!(
            "learn-dataset: {} dev session(s), {} already extracted features, {} session(s) to do, {} job(s), extractor {EXTRACTOR_VERSION}, targets {TARGET_VERSION}",
            sids.len(),
            reused,
            jobs.len(),
            a.jobs
        );

        let results: Mutex<Vec<(usize, Result<SessionStat>)>> = Mutex::new(Vec::new());
        let next = AtomicUsize::new(0);
        if !jobs.is_empty() {
            std::thread::scope(|scope| {
                for _ in 0..a.jobs.min(jobs.len()) {
                    scope.spawn(|| loop {
                        let slot = next.fetch_add(1, Ordering::Relaxed);
                        let Some(job) = jobs.get(slot) else {
                            break;
                        };
                        let started = Instant::now();
                        let result = match job.action {
                            Action::Retarget => retarget_session(self, &job.sid, &guard, &job.key),
                            Action::Extract | Action::Reuse => {
                                process_session(self, &job.sid, &guard, &job.key, &opts)
                            }
                        }
                        .map(|(action, counts)| SessionStat {
                            sid: job.sid.clone(),
                            action,
                            counts,
                            elapsed: started.elapsed(),
                        });
                        results.lock().unwrap().push((slot, result));
                    });
                }
            });
        }
        let mut results = results.into_inner().unwrap();
        results.sort_by_key(|(i, _)| *i);
        for (_, result) in &results {
            match result {
                Ok(stat) => match stat.counts {
                    Some([roots, candidates, pairs, units, cand_units, spikes]) => println!(
                        "dev {}: roots {roots} candidates {candidates} pairs {pairs} units {units} cand_units {cand_units} spikes {spikes} in {:.1}s ({})",
                        stat.sid,
                        stat.elapsed.as_secs_f64(),
                        stat.action
                    ),
                    None => println!(
                        "dev {}: reused both shards in {:.1}s",
                        stat.sid,
                        stat.elapsed.as_secs_f64()
                    ),
                },
                Err(err) => println!("dev ?: FAILED: {err:#}"),
            }
        }
        if let Some((_, Err(err))) = results.into_iter().find(|(_, r)| r.is_err()) {
            return Err(err);
        }

        // Join each session's halves, append in session order and sort rows so
        // jobs cannot reorder the output. Every part is version-checked.
        for name in TABLES {
            let mut all: Option<Table> = None;
            for (idx, sid) in sids.iter().enumerate() {
                let shard = guard.shard_dir(sid);
                let table = load_output_table(&shard, name)
                    .with_context(|| format!("load {sid} {name} shard"))?;
                ensure!(
                    table.extractor_version == EXTRACTOR_VERSION,
                    "{sid} {name} shard was extracted with {:?}, expected {EXTRACTOR_VERSION}",
                    table.extractor_version
                );
                ensure!(
                    table.target_version.as_deref() == Some(TARGET_VERSION),
                    "{sid} {name} shard has targets {:?}, expected {TARGET_VERSION}",
                    table.target_version
                );
                if all.is_none() {
                    let mut t = Table::new(EXTRACTOR_VERSION, table.columns.clone());
                    t.target_version = Some(TARGET_VERSION.to_owned());
                    t.codes = table.codes.clone();
                    t.notes = table.notes.clone();
                    t.sids = sids.clone();
                    all = Some(t);
                }
                append_remapped(
                    all.as_mut().expect("table was initialised"),
                    &table,
                    &[idx as f32],
                )?;
            }
            let Some(mut table) = all else {
                continue;
            };
            table.recordings = sids
                .iter()
                .map(|sid| (sid.clone(), self.map.sessions[sid].fingerprint.clone()))
                .collect();
            sort_rows(&mut table, row_keys(name));
            guard.prepare_dir(guard.root())?;
            table.save(guard.root(), name)?;
            let (_, data) = sorrel_data::learn::table::paths(guard.root(), name);
            println!(
                "{name}: {} rows × {} columns, {} bytes",
                table.n_rows(),
                table.columns.len(),
                fs::metadata(&data).map(|m| m.len()).unwrap_or(0)
            );
        }
        println!("{}", guard.root().display());
        Ok(())
    }
}

fn load_output_table(shard: &Path, name: &str) -> Result<Table> {
    if name == "units" || name == "cand_units" {
        return Table::load(&shard.join(TARGETS_DIR), name);
    }
    let features = Table::load(&shard.join(FEATURES_DIR), name)?;
    let targets = Table::load(&shard.join(TARGETS_DIR), name)
        .with_context(|| format!("{name} targets missing; run without --retarget"))?;
    shard::join(name, &features, &targets)
}

fn row_keys(table: &str) -> &'static [&'static str] {
    match table {
        "roots" => &["sid_idx", "root"],
        "candidates" => &["sid_idx", "root", "cand_idx"],
        "pairs" => &["sid_idx", "root_a", "root_b"],
        "units" => &["sid_idx", "unit"],
        "cand_units" => &["sid_idx", "root", "cand_idx"],
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

/// A checked `--out` root and the paths it may never reach.
///
/// [`ensure_out_safe`] resolves the root once. Because a junction or symlink
/// can appear below it later (or already be there), every destination
/// directory is re-checked immediately before a write: its path must stay
/// under the root, none of its components under the root may be a reparse
/// point, and its canonical form must not land in a protected directory.
#[derive(Debug)]
struct OutGuard {
    root: PathBuf,
    protected: Vec<PathBuf>,
}

impl OutGuard {
    fn root(&self) -> &Path {
        &self.root
    }

    fn shard_dir(&self, sid: &str) -> PathBuf {
        self.root.join("shards").join(sid)
    }

    /// Check `dir`, create it if needed, then check it again.
    fn prepare_dir(&self, dir: &Path) -> Result<()> {
        self.check_dir(dir)?;
        fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        self.check_dir(dir)
    }

    fn check_dir(&self, dir: &Path) -> Result<()> {
        ensure!(
            dir.starts_with(&self.root),
            "{} is outside --out {}",
            dir.display(),
            self.root.display()
        );
        let mut current = self.root.clone();
        for component in dir
            .strip_prefix(&self.root)
            .expect("checked above")
            .components()
        {
            current.push(component.as_os_str());
            match fs::symlink_metadata(&current) {
                Ok(meta) if super::is_link(&meta) => bail!(
                    "{} is a junction/symlink under --out; refusing to write through it",
                    current.display()
                ),
                Ok(_) => {}
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => break,
                Err(err) => {
                    return Err(err).with_context(|| format!("inspect {}", current.display()))
                }
            }
        }
        let resolved = super::canonical_target(dir)?;
        ensure!(
            resolved.starts_with(&self.root),
            "{} resolves outside --out {} (a junction/symlink?)",
            dir.display(),
            self.root.display()
        );
        for p in &self.protected {
            ensure!(
                !resolved.starts_with(p),
                "{} resolves inside protected {}; refusing to write",
                dir.display(),
                p.display()
            );
        }
        Ok(())
    }
}

/// Refuse a writable destination inside the bench's immutable session copies,
/// its sealed home or any recorded session source. Canonical paths are
/// compared, so a junction or symlink cannot smuggle `--out` back inside.
fn ensure_out_safe(bench: &Bench, out: &Path) -> Result<OutGuard> {
    // Resolve the nearest existing ancestor so a not-yet-created --out still
    // compares canonically (and through a junction/symlink) without writing.
    let root = super::canonical_target(out)?;
    let mut protected = Vec::new();
    for p in [bench.root.join("sessions"), bench.root.join(".sealed")] {
        if let Ok(p) = super::canonical_target(&p) {
            protected.push(p);
        }
    }
    for entry in bench.map.sessions.values() {
        if let Ok(p) = entry.path.canonicalize() {
            protected.push(p);
        }
    }
    for p in &protected {
        ensure!(
            !root.starts_with(p),
            "--out {} is inside protected {}; choose a directory outside BENCH/sessions, \
             BENCH/.sealed and every session source",
            root.display(),
            p.display()
        );
    }
    fs::create_dir_all(&root).with_context(|| format!("create --out {}", root.display()))?;
    Ok(OutGuard { root, protected })
}

fn feature_key(bench: &Bench, sid: &str, opts: &ExtractOpts) -> Result<FeatureKey> {
    let input = bench.session(sid).join("input");
    Ok(FeatureKey {
        extractor_version: EXTRACTOR_VERSION.to_owned(),
        max_spikes_per_root: opts.max_spikes_per_root,
        seed: opts.seed,
        input_fingerprint: super::input_hash(&input)?,
        manifest: Some(super::input_manifest(
            &input,
            &bench.map.sessions[sid].keep,
        )?),
    })
}

/// A shard from before the manifest existed is still usable when its legacy
/// fingerprint matches; say so, because it does not cover every consumed file.
fn warn_legacy_key(sid: &str, key: &FeatureKey) {
    if key.manifest.is_none() {
        eprintln!(
            "dev {sid}: features shard predates the input manifest; reusing it because the \
             legacy fingerprint still matches (re-extract to cover amplitudes, PC features, \
             params.py and the recording length)"
        );
    }
}

/// One session: input-only extraction first, truth afterwards.
fn process_session(
    bench: &Bench,
    sid: &str,
    out: &OutGuard,
    key: &FeatureKey,
    opts: &ExtractOpts,
) -> Result<(&'static str, Option<[usize; 6]>)> {
    let started = Instant::now();
    let shard = out.shard_dir(sid);
    let fdir = shard.join(FEATURES_DIR);
    let tdir = shard.join(TARGETS_DIR);
    let (mut data, action) = if shard::features_match(&fdir, key) {
        let data = shard::load_features(&fdir, key)?;
        eprintln!(
            "dev {sid}: reused features in {:.1}s",
            started.elapsed().as_secs_f64()
        );
        (data, Action::Reuse)
    } else {
        let temp = tempfile::Builder::new()
            .prefix("learn-")
            .tempdir_in(&bench.root)?;
        let input = temp.path().join("input");
        bench.copy_input(sid, &input)?;
        ensure!(
            super::input_hash(&input)? == key.input_fingerprint,
            "{sid}: copied input fingerprint differs from the sealed session input"
        );
        if let Some(manifest) = &key.manifest {
            ensure!(
                super::input_manifest(&input, &bench.map.sessions[sid].keep)? == *manifest,
                "{sid}: copied input manifest differs from the sealed session input"
            );
        }
        let provider = KilosortProvider::open(&input, Default::default())?;
        let mut session =
            Session::new(provider, Journal::open(&temp.path().join("learn.journal"))?);
        session.seed_pc_features();
        session.seed_templates();
        session.seed_template_waveforms();
        session.seed_amplitudes();
        let seeded = Instant::now();

        // Every feature, candidate and proposer output below is input-only.
        // The features shard is written before any truth file is opened.
        let data = features::extract(&session, sid, opts)?;
        let extracted_at = Instant::now();
        eprintln!(
            "dev {sid}: seeded {:.1}s, input-only extract {:.1}s",
            seeded.duration_since(started).as_secs_f64(),
            extracted_at.duration_since(seeded).as_secs_f64()
        );
        out.prepare_dir(&fdir)?;
        shard::save_features(&fdir, key, &data)?;
        (data, Action::Extract)
    };

    // Truth now: only after every input-only feature exists.
    let truth_dir = bench.session(sid).join("truth");
    let (truth, truth_hash, truth_secs) = super::load_truth(&truth_dir, &data)?;
    if shard::targets_match(&tdir, &key.id(), &truth_hash) {
        eprintln!("dev {sid}: reused targets (truth hash read in {truth_secs:.1}s)");
        return Ok(("reused", None));
    }
    let label_started = Instant::now();
    features::label(&mut data, &truth)?;
    out.prepare_dir(&tdir)?;
    shard::save_targets(&tdir, &key.id(), &truth_hash, &data)?;
    eprintln!(
        "dev {sid}: truth read {truth_secs:.1}s, label {:.1}s",
        label_started.elapsed().as_secs_f64()
    );
    let t = &data.tables;
    Ok((
        if action == Action::Extract {
            "extracted"
        } else {
            "retargeted"
        },
        Some([
            t.roots.n_rows(),
            t.candidates.n_rows(),
            t.pairs.n_rows(),
            t.units.n_rows(),
            t.cand_units.n_rows(),
            t.spikes.n_rows(),
        ]),
    ))
}

/// Recompute only the targets of a stored features shard. Never opens the
/// recording, the sorter output, a session or a proposer.
fn retarget_session(
    bench: &Bench,
    sid: &str,
    out: &OutGuard,
    key: &FeatureKey,
) -> Result<(&'static str, Option<[usize; 6]>)> {
    let started = Instant::now();
    let shard = out.shard_dir(sid);
    let mut data = shard::load_features(&shard.join(FEATURES_DIR), key)?;
    let loaded = Instant::now();
    let truth_dir = bench.session(sid).join("truth");
    let (truth, truth_hash, truth_secs) = super::load_truth(&truth_dir, &data)?;
    let label_started = Instant::now();
    features::label(&mut data, &truth)?;
    out.prepare_dir(&shard.join(TARGETS_DIR))?;
    shard::save_targets(&shard.join(TARGETS_DIR), &key.id(), &truth_hash, &data)?;
    eprintln!(
        "dev {sid}: retarget features {:.1}s, truth read {truth_secs:.1}s, label {:.1}s",
        loaded.duration_since(started).as_secs_f64(),
        label_started.elapsed().as_secs_f64()
    );
    let t = &data.tables;
    Ok((
        "retargeted",
        Some([
            t.roots.n_rows(),
            t.candidates.n_rows(),
            t.pairs.n_rows(),
            t.units.n_rows(),
            t.cand_units.n_rows(),
            t.spikes.n_rows(),
        ]),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::bench::{SessionEntry, SessionMap, Split};
    use sorrel_data::learn::features::ExtractData;
    use sorrel_io::SampleIndex;

    fn entry(split: Split, path: &str) -> SessionEntry {
        SessionEntry {
            path: path.into(),
            split,
            forced_dev: false,
            pure_manual: false,
            keep: Vec::new(),
            fingerprint: String::new(),
            recuration_of: None,
        }
    }

    fn bench_at(root: &Path, sessions: SessionMap) -> Bench {
        fs::create_dir_all(root.join(".sealed")).unwrap();
        let lock = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(root.join(".sealed/bench.lock"))
            .unwrap();
        Bench {
            root: root.canonicalize().unwrap(),
            sorrel: std::env::current_exe().unwrap(),
            scorer: std::env::current_exe().unwrap(),
            db: None,
            map: sessions,
            _lock: lock,
        }
    }

    /// Create a directory symlink/junction, or report that this machine
    /// refuses to. A Windows junction needs no special privilege, so it is
    /// the fallback when Developer Mode is off.
    fn link_dir(target: &Path, link: &Path) -> bool {
        #[cfg(windows)]
        {
            if std::os::windows::fs::symlink_dir(target, link).is_ok() {
                return true;
            }
            std::process::Command::new("cmd")
                .args(["/C", "mklink", "/J"])
                .arg(link)
                .arg(target)
                .status()
                .is_ok_and(|s| s.success())
        }
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(target, link).is_ok()
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
            map.sessions.insert(sid.into(), entry(split, "unreadable"));
        }
        let picked = dev_sessions(&map, usize::MAX).unwrap();
        assert_eq!(picked, vec!["dev-b", "dev-d"]);
        assert!(picked.iter().all(|s| !s.starts_with("h-sealed")));
        assert!(dev_sessions(&map, 1).unwrap() == vec!["dev-b"]);
        assert!(dev_sessions(&map, 0).is_err());
    }

    #[test]
    fn out_inside_sessions_sealed_or_a_source_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("bench");
        let source = tmp.path().join("source");
        fs::create_dir_all(&source).unwrap();
        let mut map = SessionMap::default();
        map.sessions
            .insert("s".into(), entry(Split::Dev, source.to_str().unwrap()));
        let bench = bench_at(&root, map);
        fs::create_dir_all(root.join("sessions")).unwrap();
        for bad in [
            root.join("sessions/out"),
            root.join(".sealed/out"),
            source.join("out"),
        ] {
            let err = ensure_out_safe(&bench, &bad).unwrap_err().to_string();
            assert!(err.contains("protected"), "{bad:?}: {err}");
        }
        assert!(ensure_out_safe(&bench, &root.join("learn/out")).is_ok());
    }

    #[cfg(windows)]
    #[test]
    fn out_through_a_junction_or_symlink_into_sessions_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("bench");
        fs::create_dir_all(&root).unwrap();
        let sessions = root.join("sessions");
        fs::create_dir_all(&sessions).unwrap();
        let bench = bench_at(&root, SessionMap::default());
        let link = root.join("outlink");
        if !link_dir(&sessions, &link) {
            eprintln!("skipping: cannot create a junction or symlink here");
            return;
        }
        let err = ensure_out_safe(&bench, &link.join("out"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("protected"), "{err}");
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn a_junction_or_symlink_below_out_is_refused_before_writing() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("bench");
        fs::create_dir_all(&root).unwrap();
        let sessions = root.join("sessions");
        fs::create_dir_all(&sessions).unwrap();
        let bench = bench_at(&root, SessionMap::default());
        let out = root.join("learn");
        let guard = ensure_out_safe(&bench, &out).unwrap();
        let shards = guard.root().join("shards");
        if !link_dir(&sessions, &shards) {
            eprintln!("skipping: cannot create a junction or symlink here");
            return;
        }
        for dest in [
            guard.shard_dir("sid").join(FEATURES_DIR),
            guard.shard_dir("sid").join(TARGETS_DIR),
        ] {
            let err = guard.prepare_dir(&dest).unwrap_err().to_string();
            assert!(
                err.contains("junction/symlink") || err.contains("protected"),
                "{err}"
            );
        }
        // Even a link that points somewhere benign is refused: no reparse
        // point below --out is a writable destination.
        let elsewhere = tmp.path().join("elsewhere");
        fs::create_dir_all(&elsewhere).unwrap();
        let nested = guard.root().join("nested");
        if link_dir(&elsewhere, &nested) {
            let err = guard.prepare_dir(&nested).unwrap_err().to_string();
            assert!(err.contains("junction/symlink"), "{err}");
        }
    }

    #[test]
    fn a_stale_or_partial_features_shard_is_not_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let key = FeatureKey {
            extractor_version: EXTRACTOR_VERSION.into(),
            max_spikes_per_root: 1500,
            seed: 0,
            input_fingerprint: "abc".into(),
            manifest: None,
        };
        assert!(!shard::features_match(dir.path(), &key));
        let mut t = Table::new(EXTRACTOR_VERSION, vec!["sid_idx".into(), "root".into()]);
        t.sids = vec!["s".into()];
        t.push_row(&[0.0, 1.0]).unwrap();
        t.save(dir.path(), "roots").unwrap();
        assert!(!shard::features_match(dir.path(), &key), "only one table");
        // Different extraction options are a different identity even with
        // the same tables sitting in the directory.
        for name in shard::INPUT_TABLES {
            t.save(dir.path(), name).unwrap();
        }
        let meta = serde_json::json!({
            "format": 1u32,
            "key": &key,
            "features_id": key.id(),
            "n_spikes": 1,
            "n_roots": 1,
            "n_candidates": 0,
            "n_pair_roots": 0,
            "n_spike_rows": 0,
            "sr_bits": 30000.0f32.to_bits(),
            "total_samples": 1,
        });
        fs::write(
            dir.path().join("meta.json"),
            serde_json::to_vec(&meta).unwrap(),
        )
        .unwrap();
        fs::write(dir.path().join("side.bin"), b"not really deflated").unwrap();
        assert!(shard::features_match(dir.path(), &key));
        let mut other = key.clone();
        other.max_spikes_per_root = 10;
        assert!(!shard::features_match(dir.path(), &other));
        let mut other = key.clone();
        other.seed = 7;
        assert!(!shard::features_match(dir.path(), &other));
        let mut other = key.clone();
        other.input_fingerprint = "different".into();
        assert!(!shard::features_match(dir.path(), &other));

        // A pre-manifest shard is accepted on the legacy fields...
        let mut with_manifest = key.clone();
        with_manifest.manifest = Some(sorrel_data::learn::shard::InputManifest {
            files: vec![sorrel_data::learn::shard::InputFile {
                name: "amplitudes.npy".into(),
                size: 4,
                sampled: false,
                digest: "d".into(),
            }],
        });
        assert!(shard::features_match(dir.path(), &with_manifest));
        assert_ne!(with_manifest.id(), key.id());
        // ...but once the shard itself records a manifest it must match.
        let meta = serde_json::json!({
            "format": 1u32,
            "key": &with_manifest,
            "features_id": with_manifest.id(),
            "n_spikes": 1,
            "n_roots": 1,
            "n_candidates": 0,
            "n_pair_roots": 0,
            "n_spike_rows": 0,
            "sr_bits": 30000.0f32.to_bits(),
            "total_samples": 1,
        });
        fs::write(
            dir.path().join("meta.json"),
            serde_json::to_vec(&meta).unwrap(),
        )
        .unwrap();
        assert!(shard::features_match(dir.path(), &with_manifest));
        let mut changed = with_manifest.clone();
        changed.manifest.as_mut().unwrap().files[0].digest = "e".into();
        assert!(!shard::features_match(dir.path(), &changed));
    }

    #[test]
    fn a_truncated_data_file_is_not_ready() {
        let dir = tempfile::tempdir().unwrap();
        let mut t = Table::new(EXTRACTOR_VERSION, vec!["sid_idx".into(), "root".into()]);
        t.sids = vec!["s".into()];
        t.push_row(&[0.0, 1.0]).unwrap();
        t.save(dir.path(), "roots").unwrap();
        assert!(Table::integrity(
            dir.path(),
            "roots",
            EXTRACTOR_VERSION,
            None
        ));
        let (_, data) = sorrel_data::learn::table::paths(dir.path(), "roots");
        let bytes = fs::read(&data).unwrap();
        fs::write(&data, &bytes[..bytes.len() - 4]).unwrap();
        assert!(!Table::integrity(
            dir.path(),
            "roots",
            EXTRACTOR_VERSION,
            None
        ));
    }

    #[test]
    fn target_tables_are_versions_not_stale_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let mut t = Table::new(EXTRACTOR_VERSION, vec!["sid_idx".into(), "root".into()]);
        t.target_version = Some(TARGET_VERSION.into());
        t.sids = vec!["s".into()];
        t.push_row(&[0.0, 1.0]).unwrap();
        for name in shard::TARGET_TABLES {
            t.save(dir.path(), name).unwrap();
        }
        let meta = serde_json::json!({
            "format": 1u32,
            "target_version": TARGET_VERSION,
            "features_id": "f",
            "truth_hash": "t",
        });
        fs::write(
            dir.path().join(shard::META_FILE),
            serde_json::to_vec(&meta).unwrap(),
        )
        .unwrap();
        assert!(shard::targets_match(dir.path(), "f", "t"));
        assert!(!shard::targets_match(dir.path(), "f", "other truth"));
        assert!(!shard::targets_match(dir.path(), "other features", "t"));
    }

    /// `--retarget` must work with no input copy at all: features carry every
    /// per-spike input array and the validated row map.
    #[test]
    fn retarget_needs_no_input_recording_or_proposers() {
        use sorrel_io::npy::write_1d_u32_atomic;
        use sorrel_io::SampleIndex;

        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("bench");
        let sid = "s-retarget";
        fs::create_dir_all(root.join(".sealed")).unwrap();
        let truth_dir = root.join("sessions").join(sid).join("truth");
        fs::create_dir_all(&truth_dir).unwrap();
        let cluster = [10u32, 10, 11, 11];
        write_1d_u32_atomic(&truth_dir.join("spike_clusters.npy"), &cluster).unwrap();
        fs::write(
            truth_dir.join("cluster_group.tsv"),
            "cluster_id\tgroup\n10\tgood\n11\tnoise\n",
        )
        .unwrap();
        let mut map = SessionMap::default();
        map.sessions.insert(
            sid.into(),
            entry(
                Split::Dev,
                tmp.path().join("absent-source").to_str().unwrap(),
            ),
        );
        let bench = bench_at(&root, map);

        // Features: two roots, each with two spikes; a candidate that takes
        // root 0's first spike and one that takes none of the good unit.
        let mut data = synthetic_data(sid, &cluster);
        data.root_of_spike = vec![0, 0, 1, 1];
        data.local_index = vec![0, 1, 0, 1];
        data.session_roots = vec![0, 0, 1, 1];
        data.row_map = vec![Some(0), Some(1), Some(2), Some(3)];
        data.time_of_spike = vec![
            SampleIndex(0),
            SampleIndex(10),
            SampleIndex(20),
            SampleIndex(30),
        ];
        data.root_sizes = vec![2, 2];
        data.root_feats = vec![None, None];
        data.root_center = vec![None, None];
        data.candidate_members = vec![vec![0, 1], vec![2]];
        let out = tmp.path().join("learn");
        let guard = ensure_out_safe(&bench, &out).unwrap();
        let shard = guard.shard_dir(sid);
        let key = FeatureKey {
            extractor_version: EXTRACTOR_VERSION.into(),
            max_spikes_per_root: 1500,
            seed: 0,
            input_fingerprint: "fp".into(),
            manifest: None,
        };
        shard::save_features(&shard.join(FEATURES_DIR), &key, &data).unwrap();
        // No `sessions/<sid>/input` exists at all.
        assert!(!root.join("sessions").join(sid).join("input").exists());
        let (action, _) = retarget_session(&bench, sid, &guard, &key).unwrap();
        assert_eq!(action, "retargeted");
        let targets = shard::load_targets(&shard.join(TARGETS_DIR), &key.id(), &{
            sorrel_data::devdb::truth_hash(&truth_dir).unwrap()
        })
        .unwrap();
        assert_eq!(targets.roots.n_rows(), 2);
        // Candidate 0 is exactly the human good unit; 1 shares nothing with
        // good, so its Jaccard is 0, not NaN.
        assert_eq!(targets.candidates.get(0, "y_jaccard"), 1.0);
        assert_eq!(targets.candidates.get(1, "y_jaccard"), 0.0);
        assert_eq!(targets.cand_units.n_rows(), 2);
        assert_eq!(targets.cand_units.get(0, "y_label"), 2.0);
        assert_eq!(targets.cand_units.get(1, "y_label"), 0.0);
    }

    fn synthetic_data(sid: &str, truth: &[u32]) -> ExtractData {
        let mut roots = Table::new(EXTRACTOR_VERSION, features::root_columns());
        roots.sids = vec![sid.into()];
        for root in 0..2 {
            let mut values = vec![f32::NAN; roots.columns.len()];
            let r = roots.column("root").unwrap();
            values[r] = root as f32;
            let s = roots.column("sid_idx").unwrap();
            values[s] = 0.0;
            roots.push_row(&values).unwrap();
        }
        let mut candidates = Table::new(EXTRACTOR_VERSION, features::candidate_columns());
        candidates.sids = vec![sid.into()];
        for (row, root) in [(0, 0), (1, 1)] {
            let mut values = vec![f32::NAN; candidates.columns.len()];
            values[candidates.column("sid_idx").unwrap()] = 0.0;
            values[candidates.column("root").unwrap()] = root as f32;
            values[candidates.column("cand_idx").unwrap()] = row as f32;
            candidates.push_row(&values).unwrap();
        }
        let mut pairs = Table::new(EXTRACTOR_VERSION, features::pair_columns());
        pairs.sids = vec![sid.into()];
        {
            let mut values = vec![f32::NAN; pairs.columns.len()];
            values[pairs.column("sid_idx").unwrap()] = 0.0;
            values[pairs.column("root_a").unwrap()] = 0.0;
            values[pairs.column("root_b").unwrap()] = 1.0;
            pairs.push_row(&values).unwrap();
        }
        let mut spikes = Table::new(EXTRACTOR_VERSION, features::spike_columns());
        spikes.sids = vec![sid.into()];
        for (root, local) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            let mut values = vec![f32::NAN; spikes.columns.len()];
            values[spikes.column("sid_idx").unwrap()] = 0.0;
            values[spikes.column("root").unwrap()] = root as f32;
            values[spikes.column("local_idx").unwrap()] = local as f32;
            spikes.push_row(&values).unwrap();
        }
        ExtractData {
            tables: features::SessionTables {
                roots,
                candidates,
                pairs,
                units: Table::new(EXTRACTOR_VERSION, features::unit_columns()),
                cand_units: Table::new(EXTRACTOR_VERSION, features::cand_unit_columns()),
                spikes,
            },
            root_of_spike: vec![0; truth.len()],
            local_index: vec![0; truth.len()],
            time_of_spike: vec![SampleIndex(0); truth.len()],
            amp_of_spike: vec![f32::NAN; truth.len()],
            root_feats: vec![None, None],
            root_center: vec![None, None],
            root_sizes: vec![truth.len(), truth.len()],
            root_amp_sum: vec![0.0, 0.0],
            root_amp_count: vec![0, 0],
            candidate_members: Vec::new(),
            spike_rows: (0..4).collect(),
            pair_roots: vec![(0, 1)],
            row_map: (0..truth.len() as u32).map(Some).collect(),
            session_roots: vec![0; truth.len()],
            sr: 30_000.0,
            total_samples: 30_000,
        }
    }
}
