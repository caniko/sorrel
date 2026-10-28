//! Export the live curation state into a phy-compatible directory.
//!
//! Save semantics: the `.sorrel` journal is the running source of truth.
//! `save_to_phy` is an *export* — it writes:
//! - `spike_clusters.npy` — `uint32` array of length `n_spikes`, the
//!   per-spike cluster assignment, in `spike_times.npy` row order.
//! - `cluster_group.tsv` — one row per cluster with a non-default label.
//!
//! After a successful export, the journal is **resealed**: its baseline
//! hash is updated to match the freshly-written `spike_clusters.npy`, and
//! its append history is cleared. This means save = commit, the same
//! mental model as Word / Photoshop / Git: prior undo state is no longer
//! reachable, but the user's data is durable.
//!
//! Both files are written atomically (write-then-rename) so a crash
//! mid-save never leaves the directory half-written.

use crate::cluster_index::ClusterIndex;
use crate::curation_json::write_curation_json;
use crate::journal::{baseline_hash, Journal};
use crate::session::Session;
use anyhow::{ensure, Context, Result};
use sorrel_io::npy::{read_1d_u32, read_header, write_1d_u32_atomic, write_rows_tmp, NpyRows};
use sorrel_io::DataProvider;
use std::borrow::Cow;
use std::path::Path;

/// Write `spike_clusters.npy`, `cluster_group.tsv` and `curation.json` into
/// `root`.
///
/// `label_str` maps each backend label to the string phy expects in the TSV
/// (`"good"`, `"mua"`, `"noise"`, `"unsorted"`). Unsorted entries are
/// elided so a phy directory that has never been curated round-trips with
/// no TSV at all.
///
/// Does **not** touch the journal. Most callers want
/// [`save_and_reseal_journal`] instead, which writes the artifacts and
/// then commits the journal so future opens treat the saved state as
/// canonical.
pub fn save_to_phy<P, F>(session: &Session<P>, root: &Path, label_str: F) -> Result<()>
where
    P: DataProvider,
    F: Fn(&P::Label) -> &'static str,
{
    ensure!(
        !has_added_spikes(session.cluster_index()),
        "this session added spikes: save with reseal (save_and_reseal_journal); a plain save \
         would leave them only in the journal and the next replay would add them twice"
    );
    write_curation(session, root, label_str)
}

fn write_curation<P, F>(session: &Session<P>, root: &Path, label_str: F) -> Result<()>
where
    P: DataProvider,
    F: Fn(&P::Label) -> &'static str,
{
    write_phy_assignments(session, root, &label_str)?;
    write_curation_json(session, &root.join("curation.json"), label_str)
}

/// Write `spike_clusters.npy` and `cluster_group.tsv` into `root`: the two
/// files phy itself reads curation from.
pub(crate) fn write_phy_assignments<P, F>(
    session: &Session<P>,
    root: &Path,
    label_str: F,
) -> Result<()>
where
    P: DataProvider,
    F: Fn(&P::Label) -> &'static str,
{
    let path = root.join("spike_clusters.npy");
    write_1d_u32_atomic(&path, &clusters_in_row_order(session.cluster_index()))
        .with_context(|| format!("write {}", path.display()))?;

    let tsv_path = root.join("cluster_group.tsv");
    let mut text = String::from("cluster_id\tgroup\n");
    for (i, lbl) in session.labels().iter().enumerate() {
        let s = label_str(lbl);
        if !s.is_empty() && s != "unsorted" {
            text.push_str(&format!("{i}\t{s}\n"));
        }
    }
    write_atomic(&tsv_path, &text)?;

    // Tags: written whenever a cluster has any, or to clear a stale sidecar.
    let tags_path = root.join(sorrel_io::kilosort::TAGS_TSV);
    let mut tags_text = String::from("cluster_id\tsorrel_tags\n");
    let mut any = false;
    for c in 0..session.n_clusters() {
        let tags = session.tags(sorrel_io::ClusterId(c));
        if !tags.is_empty() {
            any = true;
            tags_text.push_str(&format!("{c}\t{}\n", tags.join(",")));
        }
    }
    if any || tags_path.exists() {
        write_atomic(&tags_path, &tags_text)?;
    }

    write_annotations_tsv(session, root)?;

    // phy custom labels: one `cluster_<key>.tsv` per key, as phy writes them.
    for key in session.custom_label_keys() {
        let mut text = format!("cluster_id\t{key}\n");
        for c in (0..session.n_clusters()).map(sorrel_io::ClusterId) {
            if let Some(v) = session.custom_label(c, &key) {
                text.push_str(&format!("{c}\t{v}\n"));
            }
        }
        write_atomic(&root.join(format!("cluster_{key}.tsv")), &text)?;
    }
    write_cluster_info(session, root, &label_str)
}

/// `cluster_sorrel_annotations.tsv`: origin, direct parents and note for every
/// non-empty unit, so a resealed (emptied) journal loses nothing. An
/// `Auto[<sorter>]` row also pins the sorter name for the next open.
fn write_annotations_tsv<P: DataProvider>(session: &Session<P>, root: &Path) -> Result<()> {
    let mut text = String::from("cluster_id\tsorrel_origin\tsorrel_parents\tsorrel_note\n");
    for c in (0..session.n_clusters()).map(sorrel_io::ClusterId) {
        if session.spike_times(c).is_empty() {
            continue;
        }
        let parents: Vec<String> = session.parents(c).iter().map(|p| p.to_string()).collect();
        // Origins carry untrusted names (MCP clientInfo, sorrel-py `by=`),
        // so they are escaped like notes.
        text.push_str(&format!(
            "{c}\t{}\t{}\t{}\n",
            sorrel_io::kilosort::escape_note(&session.origin_label(c)),
            parents.join(","),
            sorrel_io::kilosort::escape_note(session.note(c))
        ));
    }
    write_atomic(&root.join(sorrel_io::kilosort::ANNOTATIONS_TSV), &text)
}

/// phy's `cluster_info.tsv`: one row per non-empty cluster with the columns
/// phy's cluster view shows (`cluster_id`, `ch`, `fr`, `amp`, `n_spikes`,
/// `group`, custom labels), rewritten on every save. `depth` needs probe
/// geometry, which the session does not hold, so it is left out.
fn write_cluster_info<P, F>(session: &Session<P>, root: &Path, label_str: &F) -> Result<()>
where
    P: DataProvider,
    F: Fn(&P::Label) -> &'static str,
{
    let keys = session.custom_label_keys();
    let duration =
        session.provider.n_samples().0 as f64 / session.provider.sample_rate().max(1.0) as f64;
    let mut text = String::from("cluster_id\tch\tfr\tamp\tn_spikes\tgroup");
    for k in &keys {
        text.push('\t');
        text.push_str(k);
    }
    text.push('\n');
    for c in (0..session.n_clusters()).map(sorrel_io::ClusterId) {
        let n = session.spike_times(c).len();
        if n == 0 {
            continue;
        }
        // Peak channel of the primary template.
        let ch = session
            .primary_template(c)
            .and_then(|t| session.template_waveform(t))
            .map(|w| {
                let (_, ns, nc) = session.template_shape();
                let ptp = |ch: usize| {
                    let col = (0..ns).filter_map(|s| w.get(s * nc + ch));
                    let (lo, hi) =
                        col.fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
                    hi - lo
                };
                let peak = (0..nc)
                    .max_by(|&a, &b| ptp(a).total_cmp(&ptp(b)))
                    .unwrap_or(0);
                peak.to_string()
            })
            .unwrap_or_default();
        let amps = session.spike_amplitudes(c);
        let amp = if amps.is_empty() {
            String::new()
        } else {
            format!(
                "{:.3}",
                amps.iter().map(|&a| a as f64).sum::<f64>() / amps.len() as f64
            )
        };
        let group = session.label(c).map_or("", |l| label_str(&l));
        text.push_str(&format!(
            "{c}\t{ch}\t{:.3}\t{amp}\t{n}\t{group}",
            n as f64 / duration.max(f64::MIN_POSITIVE)
        ));
        for k in &keys {
            text.push('\t');
            text.push_str(session.custom_label(c, k).unwrap_or(""));
        }
        text.push('\n');
    }
    write_atomic(&root.join("cluster_info.tsv"), &text)
}

fn write_atomic(path: &Path, text: &str) -> Result<()> {
    let tmp = with_tmp_suffix(path);
    std::fs::write(&tmp, text).with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))
}

/// The per-spike cluster ids in the backend's original row order (the order
/// of `spike_times.npy`, `amplitudes.npy`, `pc_features.npy`, ...), so the
/// written file stays aligned with every other per-spike array. The index
/// itself is time-sorted with ties broken by cluster id, which differs from
/// the file order for unsorted inputs and for simultaneous spikes.
///
/// Falls back to time order when the row map is absent or not a
/// permutation; only phy sessions carry one, and only phy directories have
/// sibling arrays to stay aligned with.
fn clusters_in_row_order(index: &ClusterIndex) -> Vec<u32> {
    let clusters = index.spike_clusters();
    let rows = index.spike_pc_indices();
    let time_order = || clusters.iter().map(|c| c.0).collect::<Vec<u32>>();
    // After `rewrite_spike_arrays` every file is in global (time) order.
    if rows.len() != clusters.len() || has_added_spikes(index) {
        return time_order();
    }
    // Cluster ids never reach u32::MAX (split placeholders are resolved
    // before they land in the index), so it marks an unfilled row.
    let mut out = vec![u32::MAX; clusters.len()];
    for (c, r) in clusters.iter().zip(rows) {
        match r.get().and_then(|r| out.get_mut(r as usize)) {
            Some(slot) if *slot == u32::MAX => *slot = c.0,
            _ => return time_order(),
        }
    }
    out
}

/// Save the curation state to phy artifacts AND reseal the journal at
/// `journal_path`. The previous journal (with all its undo history) is
/// truncated; the new journal's baseline hash matches the freshly-written
/// `spike_clusters.npy`. Subsequent reopens load the new state cleanly
/// with empty undo history.
pub fn save_and_reseal_journal<P, F>(
    session: &Session<P>,
    root: &Path,
    journal_path: &Path,
    label_str: F,
) -> Result<Journal>
where
    P: DataProvider,
    F: Fn(&P::Label) -> &'static str,
{
    if has_added_spikes(session.cluster_index()) {
        rewrite_spike_arrays(session, root)?;
    }
    write_curation(session, root, &label_str)?;

    // Hash the bytes we just wrote (post-rename, post-fsync).
    let bytes = std::fs::read(root.join("spike_clusters.npy"))
        .context("hash freshly-written spike_clusters.npy")?;
    let new_baseline = baseline_hash(&bytes);
    let n_clusters = session.n_clusters();
    let journal = Journal::truncate(journal_path, new_baseline, n_clusters)?;
    // The artifacts now carry every merge and split, so the interchange
    // document must not repeat them: rewrite it as labels only. The session
    // still holds the old history until the caller swaps in `journal`.
    let labels_only = crate::curation_json::curation_document_sealed(session, label_str);
    let json_path = root.join("curation.json");
    let tmp = json_path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(&labels_only)?)
        .with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, &json_path)
        .with_context(|| format!("rename {} -> {}", tmp.display(), json_path.display()))?;
    Ok(journal)
}

fn with_tmp_suffix(path: &Path) -> std::path::PathBuf {
    let mut p = path.as_os_str().to_owned();
    p.push(".tmp");
    p.into()
}

/// Original files of a rewrite are hard-linked (or copied) here, once.
const BACKUP_DIR: &str = "sorrel_original";
/// Attribute marking spikes inserted by `AddSpikes` (u32 0/1).
const RESCUED: &str = "sorrel_rescued";

/// True when the session holds spikes no file has yet: feature rows past the
/// provider's (see `ClusterIndex::add_spikes`).
pub(crate) fn has_added_spikes(index: &ClusterIndex) -> bool {
    let n = index.n_orig_rows();
    index
        .spike_pc_indices()
        .iter()
        .any(|r| r.get().is_some_and(|r| r as usize >= n))
}

/// Rewrite every per-spike array of a phy directory in global (time) order so
/// they include the spikes added by `AddSpikes`. Crash-safe: tmp files are
/// fsynced, a manifest naming them is fsynced, then the renames are rolled
/// out by [`sorrel_io::kilosort::roll_forward_pending`], which
/// `KilosortProvider::open` also runs after an interruption. Originals go to
/// `sorrel_original/` first (never overwritten), and phy's `.phy/` cache,
/// now stale, is moved to `.phy.sorrel-stale`.
///
/// The session's own index keeps the old row numbering: reopen the directory
/// afterwards (a second rewrite from the same session is refused).
pub(crate) fn rewrite_spike_arrays<P: DataProvider>(
    session: &Session<P>,
    root: &Path,
) -> Result<()> {
    use sorrel_io::kilosort::{roll_forward_pending, REWRITE_PENDING};
    use std::io::Write;
    let names = write_spike_array_tmps(session, root, root)?;
    let tmp_of = |n: &String| root.join(format!("{n}.tmp"));
    let prepare = || -> Result<()> {
        let bak = root.join(BACKUP_DIR);
        std::fs::create_dir_all(&bak).with_context(|| format!("create {}", bak.display()))?;
        for n in &names {
            let (from, to) = (root.join(n), bak.join(n));
            if from.exists() && !to.exists() && std::fs::hard_link(&from, &to).is_err() {
                std::fs::copy(&from, &to).with_context(|| format!("back up {n}"))?;
            }
        }
        let phy = root.join(".phy");
        if phy.exists() {
            let stale = root.join(".phy.sorrel-stale");
            let _ = std::fs::remove_dir_all(&stale);
            std::fs::rename(&phy, &stale).context("move aside stale .phy cache")?;
        }
        // Manifest via tmp + rename so a torn one is never seen.
        let manifest = root.join(REWRITE_PENDING);
        let tmp = with_tmp_suffix(&manifest);
        let text: String = names.iter().map(|n| format!("{n}.tmp\t{n}\n")).collect();
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(text.as_bytes())?;
        f.sync_all()?;
        std::fs::rename(&tmp, &manifest).context("publish rewrite manifest")
    };
    if let Err(e) = prepare() {
        names
            .iter()
            .for_each(|n| drop(std::fs::remove_file(tmp_of(n))));
        return Err(e);
    }
    roll_forward_pending(root)?;
    // ponytail: best effort (a directory cannot be opened on Windows).
    let _ = std::fs::File::open(root).and_then(|d| d.sync_all());
    Ok(())
}

/// Write every per-spike array of `session` as `<dst>/<name>.tmp` (fsynced,
/// not renamed), reading the originals from `src`, and return the names.
/// Rows follow the global order; original rows are copied byte for byte
/// through the row map, rows of added spikes are encoded in the file's dtype.
/// Refuses, writing nothing, when `src` holds a per-spike `.npy` this does not
/// know (it would fall out of alignment) or no longer matches the session.
pub(crate) fn write_spike_array_tmps<P: DataProvider>(
    session: &Session<P>,
    src: &Path,
    dst: &Path,
) -> Result<Vec<String>> {
    let ci = session.cluster_index();
    let (n, n_orig) = (ci.n_spikes(), ci.n_orig_rows());
    let rows: Vec<u32> = ci
        .spike_pc_indices()
        .iter()
        .map_while(|r| r.get())
        .collect();
    ensure!(
        rows.len() == n,
        "the PC row map is incomplete; cannot rewrite the per-spike arrays"
    );
    let times_hdr = read_header(&src.join("spike_times.npy"))?;
    ensure!(
        times_hdr.shape.first() == Some(&n_orig),
        "spike_times.npy no longer has the {n_orig} rows this session opened with (already \
         rewritten?); reopen the directory"
    );

    let (pc_w, tf) = (
        ci.pc_shape().0 * ci.pc_shape().1,
        ci.has_template_features(),
    );
    let attrs: Vec<String> = ci
        .spike_attribute_names()
        .into_iter()
        .map(|(a, _)| a.to_owned())
        .collect();
    let mut known = vec![
        "spike_times.npy".to_owned(),
        "spike_clusters.npy".to_owned(),
    ];
    let flags = [
        (
            !ci.spike_templates_global().is_empty(),
            "spike_templates.npy",
        ),
        (!ci.spike_amplitudes_global().is_empty(), "amplitudes.npy"),
        (pc_w > 0, "pc_features.npy"),
        (tf, "template_features.npy"),
        (ci.has_spike_reorder(), "spike_reorder.npy"),
    ];
    known.extend(flags.iter().filter(|f| f.0).map(|f| f.1.to_owned()));
    known.extend(attrs.iter().map(|a| format!("spike_{a}.npy")));
    if !attrs.iter().any(|a| a == RESCUED) {
        known.push(format!("spike_{RESCUED}.npy"));
    }
    let mut unknown: Vec<String> = std::fs::read_dir(src)
        .with_context(|| format!("read {}", src.display()))?
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            let hdr = name
                .ends_with(".npy")
                .then(|| read_header(&e.path()).ok())??;
            (hdr.shape.first() == Some(&n_orig) && !known.contains(&name)).then_some(name)
        })
        .collect();
    unknown.sort();
    ensure!(
        unknown.is_empty(),
        "refusing to save: {unknown:?} hold one row per spike but sorrel does not know how to \
         insert the added spikes into them (or never loaded them); nothing was written"
    );

    let mut done: Vec<String> = Vec::new();
    let result = (|| -> Result<()> {
        let mut put = |name: &str| {
            done.push(name.to_owned());
            dst.join(name)
        };
        let u4 = "<u4";
        write_rows_tmp(
            &put("spike_times.npy"),
            "<u8",
            &[],
            ci.all_spike_times().iter().map(|t| t.0.to_le_bytes()),
        )?;
        write_rows_tmp(
            &put("spike_clusters.npy"),
            u4,
            &[],
            ci.spike_clusters().iter().map(|c| c.0.to_le_bytes()),
        )?;
        if flags[0].0 {
            let t = ci.spike_templates_global();
            write_rows_tmp(&put(flags[0].1), u4, &[], t.iter().map(|v| v.to_le_bytes()))?;
        }
        if flags[1].0 {
            let a = ci.spike_amplitudes_global();
            write_rows_tmp(
                &put(flags[1].1),
                "<f4",
                &[],
                a.iter().map(|v| v.to_le_bytes()),
            )?;
        }
        if pc_w > 0 {
            put("pc_features.npy");
            rewrite_rows(src, dst, "pc_features.npy", &rows, n_orig, |g| {
                ci.pc_feature_for(g as u32)
            })?;
        }
        if tf {
            put("template_features.npy");
            rewrite_rows(src, dst, "template_features.npy", &rows, n_orig, |g| {
                ci.template_feature_row(g as u32)
            })?;
        }
        const ONE: [f32; 1] = [1.0];
        for a in &attrs {
            let name = format!("spike_{a}.npy");
            put(&name);
            rewrite_rows(src, dst, &name, &rows, n_orig, |g| {
                if a == RESCUED {
                    Some(&ONE[..])
                } else {
                    ci.spike_attribute(a, g as u32)
                }
            })?;
        }
        if !attrs.iter().any(|a| a == RESCUED) {
            let marks = rows
                .iter()
                .map(|&r| u32::from(r as usize >= n_orig).to_le_bytes());
            write_rows_tmp(&put(&format!("spike_{RESCUED}.npy")), u4, &[], marks)?;
        }
        if ci.has_spike_reorder() {
            // Old value = old row whose time is shown; point at that spike's new index.
            let old = read_1d_u32(&src.join("spike_reorder.npy"))?;
            let mut new_of_row = vec![u32::MAX; n_orig];
            for (g, &r) in rows.iter().enumerate() {
                if let Some(slot) = new_of_row.get_mut(r as usize) {
                    *slot = g as u32;
                }
            }
            let reorder = rows.iter().enumerate().map(|(g, &r)| {
                let to = old
                    .get(r as usize)
                    .and_then(|&j| new_of_row.get(j as usize));
                to.copied()
                    .filter(|&v| v != u32::MAX)
                    .unwrap_or(g as u32)
                    .to_le_bytes()
            });
            write_rows_tmp(&put("spike_reorder.npy"), u4, &[], reorder)?;
        }
        Ok(())
    })();
    if let Err(e) = result {
        done.iter()
            .for_each(|n| drop(std::fs::remove_file(with_tmp_suffix(&dst.join(n)))));
        return Err(e);
    }
    Ok(done)
}

/// One row-addressed array: original rows are copied byte for byte from
/// `<src>/<name>`, rows of added spikes come from `added(g)`, encoded in the
/// file's dtype. Writes `<dst>/<name>.tmp`.
fn rewrite_rows<'a>(
    src: &Path,
    dst: &Path,
    name: &str,
    rows: &[u32],
    n_orig: usize,
    added: impl Fn(usize) -> Option<&'a [f32]>,
) -> Result<()> {
    let f = NpyRows::open(&src.join(name))?;
    ensure!(
        f.n_rows == n_orig,
        "{name}: {} rows, expected {n_orig}",
        f.n_rows
    );
    // ponytail: rows are gathered first (a Cow per spike, ~32 B) because
    // encoding can fail; stream them if 100M-spike sessions need it.
    let out = rows
        .iter()
        .enumerate()
        .map(|(g, &r)| match (r as usize) < n_orig {
            true => Ok(Cow::Borrowed(f.row(r as usize))),
            false => added(g)
                .with_context(|| format!("{name}: no values for added spike {g}"))
                .and_then(|v| f.encode(v))
                .map(Cow::Owned),
        })
        .collect::<Result<Vec<_>>>()?;
    write_rows_tmp(&dst.join(name), &f.descr, &f.tail, out.into_iter()).map(drop)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::PhyLabelOp;
    use crate::session::ApplyPhyLabel;
    use crate::{CurationCommand, Journal, Session};
    use sorrel_io::npy::read_1d_u32;
    use sorrel_io::{ClusterId, DataProvider, SampleIndex, TraceSamples, TraceSlice};

    /// Tiny stand-in provider for the writer tests.
    struct MockProvider {
        spikes: Vec<Vec<SampleIndex>>,
        labels: Vec<MockLabel>,
    }

    #[repr(u8)]
    #[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
    enum MockLabel {
        #[default]
        Unsorted = 0,
        Good = 1,
        Mua = 2,
        Noise = 3,
    }

    impl DataProvider for MockProvider {
        type Label = MockLabel;
        fn sample_rate(&self) -> f32 {
            1000.0
        }
        fn n_channels(&self) -> u32 {
            0
        }
        fn n_samples(&self) -> SampleIndex {
            SampleIndex(0)
        }
        fn n_clusters(&self) -> u32 {
            self.spikes.len() as u32
        }
        fn spike_times(&self, c: ClusterId) -> &[SampleIndex] {
            self.spikes.get(c.idx()).map(Vec::as_slice).unwrap_or(&[])
        }
        fn trace(&self, _: SampleIndex, _: u32) -> TraceSlice<'_> {
            TraceSlice {
                start: SampleIndex(0),
                n_channels: 0,
                samples: TraceSamples::I16(&[]),
            }
        }
        fn initial_labels(&self) -> Vec<Self::Label> {
            self.labels.clone()
        }
    }

    impl ApplyPhyLabel for MockProvider {
        fn label_from_op(op: PhyLabelOp) -> Self::Label {
            match op {
                PhyLabelOp::SetUnsorted => MockLabel::Unsorted,
                PhyLabelOp::SetGood => MockLabel::Good,
                PhyLabelOp::SetMua => MockLabel::Mua,
                PhyLabelOp::SetNoise => MockLabel::Noise,
            }
        }
    }

    fn label_str(l: &MockLabel) -> &'static str {
        match l {
            MockLabel::Unsorted => "unsorted",
            MockLabel::Good => "good",
            MockLabel::Mua => "mua",
            MockLabel::Noise => "noise",
        }
    }

    fn fresh_session() -> (Session<MockProvider>, tempfile::TempDir) {
        let provider = MockProvider {
            spikes: vec![
                vec![SampleIndex(10), SampleIndex(30), SampleIndex(150)],
                vec![SampleIndex(20), SampleIndex(200)],
                vec![SampleIndex(100)],
            ],
            labels: vec![MockLabel::Unsorted; 3],
        };
        let dir = tempfile::tempdir().unwrap();
        let journal = Journal::open(&dir.path().join("j.sqlite")).unwrap();
        (Session::new(provider, journal), dir)
    }

    #[test]
    fn save_writes_spikeinterface_curation_json_and_reseal_drops_structure() {
        use sorrel_io::read_curation_labels;
        let (mut s, dir) = fresh_session();
        s.dispatch(CurationCommand::Relabel {
            cluster: ClusterId(0),
            op: PhyLabelOp::SetGood,
        })
        .unwrap();
        s.dispatch(CurationCommand::Merge {
            sources: vec![ClusterId(1)],
            target: ClusterId(2),
        })
        .unwrap();
        // Split spike t=30 (local index 1) out of cluster 0 → new cluster 3.
        s.dispatch(CurationCommand::Split {
            cluster: ClusterId(0),
            spike_idx: vec![1],
            new_cluster: ClusterId(u32::MAX),
            by: crate::command::Method::Manual,
        })
        .unwrap();

        save_to_phy(&s, dir.path(), label_str).unwrap();
        let path = dir.path().join("curation.json");
        let doc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();

        assert_eq!(doc["format_version"], "2");
        assert_eq!(doc["unit_ids"], serde_json::json!([0, 1, 2]));
        assert_eq!(doc["label_definitions"]["quality"]["exclusive"], true);
        // Every unit carries its exclusive origin label.
        assert_eq!(doc["label_definitions"]["origin"]["exclusive"], true);
        let first = &doc["manual_labels"][0]["labels"];
        assert_eq!(first["quality"], serde_json::json!(["good"]));
        assert_eq!(first["origin"], serde_json::json!(["Auto[unknown]"]));
        assert_eq!(doc["merges"][0]["unit_ids"], serde_json::json!([1, 2]));
        assert_eq!(doc["merges"][0]["new_unit_id"], 2);
        assert_eq!(doc["splits"][0]["unit_id"], 0);
        assert_eq!(doc["splits"][0]["indices"], serde_json::json!([[1]]));
        // The journal placeholder (u32::MAX) was resolved to the allocated id.
        assert_eq!(doc["splits"][0]["new_unit_ids"], serde_json::json!([3]));
        assert_eq!(doc["removed"], serde_json::json!([]));

        let labels = read_curation_labels(&path, 4, |u| Some(u as usize)).unwrap();
        assert_eq!(labels[0].as_str(), "good");
        assert_eq!(labels[1].as_str(), "unsorted");

        // Reseal: the artifacts now carry the merge and split, so the JSON
        // becomes labels-only over every cluster.
        save_and_reseal_journal(&s, dir.path(), &dir.path().join("j.sqlite"), label_str).unwrap();
        let doc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(doc["unit_ids"], serde_json::json!([0, 1, 2, 3]));
        assert_eq!(doc["merges"], serde_json::json!([]));
        assert_eq!(doc["splits"], serde_json::json!([]));
        assert!(!dir.path().join("curation.json.tmp").exists());
    }

    #[test]
    fn save_writes_tags_to_tsv_and_curation_json() {
        let (mut s, dir) = fresh_session();
        s.dispatch(CurationCommand::Tag {
            cluster: ClusterId(1),
            tags: vec!["src:unitrefine".into(), "review:low-conf".into()],
        })
        .unwrap();
        save_to_phy(&s, dir.path(), label_str).unwrap();

        let tsv = dir.path().join(sorrel_io::kilosort::TAGS_TSV);
        let read = sorrel_io::kilosort::read_tags_tsv(&tsv);
        assert_eq!(read[1], ["src:unitrefine", "review:low-conf"]);
        assert!(read[0].is_empty());

        let doc: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.path().join("curation.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            doc["manual_labels"][1],
            serde_json::json!({ "unit_id": 1, "labels": {
                "origin": ["Auto[unknown]"],
                "tags": ["src:unitrefine", "review:low-conf"]
            } })
        );
        assert_eq!(doc["label_definitions"]["tags"]["exclusive"], false);
        let imported = sorrel_io::read_curation(&dir.path().join("curation.json")).unwrap();
        assert_eq!(imported.labels[1].2, ["src:unitrefine", "review:low-conf"]);

        // Clearing every tag rewrites the sidecar rather than leaving it stale.
        s.dispatch(CurationCommand::Tag {
            cluster: ClusterId(1),
            tags: vec![],
        })
        .unwrap();
        save_to_phy(&s, dir.path(), label_str).unwrap();
        assert!(sorrel_io::kilosort::read_tags_tsv(&tsv)
            .iter()
            .all(Vec::is_empty));
    }

    #[test]
    fn custom_labels_write_phy_tsvs_and_undo() {
        let (mut s, dir) = fresh_session();
        let cmd = s
            .set_label_command(ClusterId(1), "quality", "great")
            .unwrap();
        s.dispatch(cmd).unwrap();
        let cmd = s.set_label_command(ClusterId(1), "quality", "ok").unwrap();
        s.dispatch(cmd).unwrap();
        assert_eq!(
            s.custom_label(ClusterId(1), "quality"),
            Some("ok"),
            "one value per key"
        );
        assert!(s.set_label_command(ClusterId(1), "group", "x").is_err());
        assert!(s.set_label_command(ClusterId(1), "q", "a,b").is_err());
        save_to_phy(&s, dir.path(), label_str).unwrap();

        let tsv = std::fs::read_to_string(dir.path().join("cluster_quality.tsv")).unwrap();
        assert_eq!(tsv, "cluster_id\tquality\n1\tok\n");
        let info = std::fs::read_to_string(dir.path().join("cluster_info.tsv")).unwrap();
        let mut lines = info.lines();
        assert_eq!(
            lines.next(),
            Some("cluster_id\tch\tfr\tamp\tn_spikes\tgroup\tquality")
        );
        assert!(lines.any(|l| l.starts_with("1\t") && l.ends_with("\tok")));

        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(s.custom_label(ClusterId(1), "quality"), Some("great"));
    }

    #[test]
    fn import_is_a_delta_batch_that_one_undo_reverts() {
        use crate::curation_json::import_commands;
        use sorrel_io::kilosort::PhyLabel;
        use sorrel_io::CurationImport;
        let (mut s, _d) = fresh_session();
        let doc = CurationImport {
            labels: vec![
                (0, Some(PhyLabel::Good), vec!["agree".into()]),
                (1, Some(PhyLabel::Noise), vec!["merged:consensus".into()]),
                (2, Some(PhyLabel::Noise), vec!["merged:consensus".into()]),
            ],
            merges: vec![vec![2, 1]],
        };
        let current = |c: ClusterId| label_str(&s.label(c).unwrap_or_default());
        let cmds = import_commands(&s, &doc, current).unwrap();
        assert_eq!(cmds.len(), 7, "3 relabels + 3 retags + 1 merge: {cmds:?}");
        assert!(matches!(
            cmds.last(),
            Some(CurationCommand::Merge { sources, target })
                if *target == ClusterId(1) && sources == &[ClusterId(2)]
        ));
        s.dispatch(CurationCommand::batch(cmds).unwrap()).unwrap();
        // The Auto merge target also gains the session's `tuned` marker.
        assert_eq!(s.tags(ClusterId(1)), ["merged:consensus", "tuned"]);
        assert!(s.spike_times(ClusterId(2)).is_empty());
        // Re-importing the same document is a no-op, merge included.
        let current = |c: ClusterId| label_str(&s.label(c).unwrap_or_default());
        let again = import_commands(&s, &doc, current).unwrap();
        assert!(again.is_empty(), "{again:?}");

        s.dispatch(CurationCommand::Undo).unwrap();
        assert!(s.tags(ClusterId(1)).is_empty());
        assert_eq!(s.spike_times(ClusterId(2)).len(), 1);

        let current = |_| "unsorted";
        let unknown = CurationImport {
            labels: vec![(9, None, vec![])],
            merges: vec![],
        };
        assert!(import_commands(&s, &unknown, current).is_err());
        let bad_tag = CurationImport {
            labels: vec![(0, None, vec!["a,b".into()])],
            merges: vec![],
        };
        assert!(import_commands(&s, &bad_tag, current).is_err());
        let twice = CurationImport {
            labels: vec![],
            merges: vec![vec![0, 1], vec![1, 2]],
        };
        assert!(import_commands(&s, &twice, current).is_err());
    }

    #[test]
    fn import_never_grants_session_markers() {
        use crate::curation_json::import_commands;
        use sorrel_io::CurationImport;
        let (mut s, _d) = fresh_session();
        let doc = CurationImport {
            labels: vec![
                (
                    0,
                    None,
                    vec!["agree".into(), "tuned".into(), "external".into()],
                ),
                (1, None, vec!["tuned".into()]),
            ],
            merges: vec![],
        };
        let current = |c: ClusterId| label_str(&s.label(c).unwrap_or_default());
        let cmds = import_commands(&s, &doc, current).unwrap();
        s.dispatch(CurationCommand::batch(cmds).unwrap()).unwrap();
        assert_eq!(s.tags(ClusterId(0)), ["agree"]);
        assert!(s.tags(ClusterId(1)).is_empty());
        // Re-importing the same document is a no-op.
        let current = |c: ClusterId| label_str(&s.label(c).unwrap_or_default());
        assert!(import_commands(&s, &doc, current).unwrap().is_empty());
    }

    #[test]
    fn save_writes_spike_clusters_matching_current_state() {
        let (mut s, dir) = fresh_session();
        // Merge cluster 0 into 2: every spike in the global arrays that was
        // in 0 should now read as 2 in the file we write.
        s.dispatch(CurationCommand::Merge {
            sources: vec![ClusterId(0)],
            target: ClusterId(2),
        })
        .unwrap();

        save_to_phy(&s, dir.path(), label_str).unwrap();

        let written = read_1d_u32(&dir.path().join("spike_clusters.npy")).unwrap();
        // Time order is [10(c0), 20(c1), 30(c0), 100(c2), 150(c0), 200(c1)].
        // After merging 0 into 2: [2, 1, 2, 2, 2, 1].
        assert_eq!(written, vec![2, 1, 2, 2, 2, 1]);
    }

    #[test]
    fn save_writes_cluster_group_tsv_excluding_unsorted_rows() {
        let (mut s, dir) = fresh_session();
        s.dispatch(CurationCommand::Relabel {
            cluster: ClusterId(0),
            op: PhyLabelOp::SetGood,
        })
        .unwrap();
        s.dispatch(CurationCommand::Relabel {
            cluster: ClusterId(2),
            op: PhyLabelOp::SetNoise,
        })
        .unwrap();

        save_to_phy(&s, dir.path(), label_str).unwrap();

        let tsv = std::fs::read_to_string(dir.path().join("cluster_group.tsv")).unwrap();
        assert_eq!(
            tsv, "cluster_id\tgroup\n0\tgood\n2\tnoise\n",
            "unsorted cluster 1 should be elided"
        );
    }

    #[test]
    fn save_is_atomic_no_tmp_files_left_behind() {
        let (s, dir) = fresh_session();
        save_to_phy(&s, dir.path(), label_str).unwrap();

        // Confirm only the expected files exist (no `.tmp` leftovers).
        let names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| Some(e.ok()?.file_name().to_string_lossy().into_owned()))
            .collect();
        assert!(names.iter().any(|n| n == "spike_clusters.npy"));
        assert!(names.iter().any(|n| n == "cluster_group.tsv"));
        assert!(
            !names.iter().any(|n| n.ends_with(".tmp")),
            "no tmp files should remain after a successful save (got {names:?})"
        );
    }

    #[test]
    fn save_after_split_writes_new_cluster_assignments() {
        let (mut s, dir) = fresh_session();
        // Split spike at local index 1 of cluster 0 -> new cluster id.
        s.dispatch(CurationCommand::Split {
            cluster: ClusterId(0),
            spike_idx: vec![1],
            new_cluster: ClusterId(0), // ignored: auto-allocated
            by: crate::command::Method::Manual,
        })
        .unwrap();

        save_to_phy(&s, dir.path(), label_str).unwrap();
        let written = read_1d_u32(&dir.path().join("spike_clusters.npy")).unwrap();

        // Time order: [10(c0), 20(c1), 30(c0), 100(c2), 150(c0), 200(c1)].
        // Cluster 0 had local indices [0, 1, 2] for times [10, 30, 150];
        // splitting local idx 1 means t=30 moves to the new cluster.
        // Initial n_clusters was 3, so new id is 3.
        assert_eq!(written, vec![0, 1, 3, 2, 0, 1]);
    }

    #[test]
    fn save_after_undo_writes_pre_change_state() {
        let (mut s, dir) = fresh_session();
        s.dispatch(CurationCommand::Merge {
            sources: vec![ClusterId(0), ClusterId(1)],
            target: ClusterId(2),
        })
        .unwrap();
        s.dispatch(CurationCommand::Undo).unwrap();

        save_to_phy(&s, dir.path(), label_str).unwrap();
        let written = read_1d_u32(&dir.path().join("spike_clusters.npy")).unwrap();
        // Original time order: [10(c0), 20(c1), 30(c0), 100(c2), 150(c0), 200(c1)].
        assert_eq!(written, vec![0, 1, 0, 2, 0, 1]);
    }

    #[test]
    fn save_round_trip_after_many_relabels() {
        let (mut s, dir) = fresh_session();
        s.dispatch(CurationCommand::Relabel {
            cluster: ClusterId(0),
            op: PhyLabelOp::SetGood,
        })
        .unwrap();
        s.dispatch(CurationCommand::Relabel {
            cluster: ClusterId(1),
            op: PhyLabelOp::SetMua,
        })
        .unwrap();
        s.dispatch(CurationCommand::Relabel {
            cluster: ClusterId(2),
            op: PhyLabelOp::SetNoise,
        })
        .unwrap();

        save_to_phy(&s, dir.path(), label_str).unwrap();
        let tsv = std::fs::read_to_string(dir.path().join("cluster_group.tsv")).unwrap();
        let lines: Vec<&str> = tsv.lines().collect();
        assert_eq!(lines[0], "cluster_id\tgroup");
        // Three rows, in cluster id order.
        assert_eq!(lines[1], "0\tgood");
        assert_eq!(lines[2], "1\tmua");
        assert_eq!(lines[3], "2\tnoise");
        assert_eq!(lines.len(), 4);
    }

    #[test]
    fn save_writes_no_tsv_rows_when_no_curation_happened() {
        let (s, dir) = fresh_session();
        save_to_phy(&s, dir.path(), label_str).unwrap();
        let tsv = std::fs::read_to_string(dir.path().join("cluster_group.tsv")).unwrap();
        // Header only — every cluster is "unsorted" by default.
        assert_eq!(tsv, "cluster_id\tgroup\n");
    }

    #[test]
    fn double_save_is_idempotent() {
        let (mut s, dir) = fresh_session();
        s.dispatch(CurationCommand::Relabel {
            cluster: ClusterId(1),
            op: PhyLabelOp::SetGood,
        })
        .unwrap();
        save_to_phy(&s, dir.path(), label_str).unwrap();
        let first = std::fs::read(dir.path().join("spike_clusters.npy")).unwrap();
        save_to_phy(&s, dir.path(), label_str).unwrap();
        let second = std::fs::read(dir.path().join("spike_clusters.npy")).unwrap();
        assert_eq!(first, second);
    }
}
