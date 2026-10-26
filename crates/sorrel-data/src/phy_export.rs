//! Write the curated session as a complete phy template-gui directory.
//!
//! Two paths, picked by [`DataProvider::phy_dir`]:
//! - **phy/Kilosort source**: its `*.npy`/`*.tsv`/`*.csv` files are copied
//!   byte for byte; only the curation (`spike_clusters.npy`,
//!   `cluster_group.tsv`) and `params.py` are rewritten.
//! - **any other backend** (SortingAnalyzer, NWB, rez): the arrays phy
//!   requires are written from memory, with one mean template per cluster
//!   computed from the raw trace, the way SpikeInterface's `export_to_phy`
//!   does it.
//!
//! `params.py` points at the source's raw file. With `copy_raw`, or when the
//! samples live only in memory (NWB), the trace is streamed into
//! `recording.dat` instead. It is written last, so an interrupted export
//! never looks complete to phy.

use crate::save::write_phy_assignments;
use crate::session::Session;
use anyhow::{bail, ensure, Context, Result};
use rayon::prelude::*;
use sorrel_io::npy::{write_1d_u32_atomic, write_1d_u64_atomic, write_f32_atomic};
use sorrel_io::{ClusterId, DataProvider, HasGeometry, SampleIndex, TraceDtype};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// What a phy directory keeps per spike and per cluster. Everything else in
/// a Kilosort output (`temp_wh.dat`, `rez.mat`, raw recordings, the sorrel
/// journal, `curation.json`, `.phy/`) stays behind.
const COPIED_EXTENSIONS: [&str; 3] = ["npy", "tsv", "csv"];
/// Rewritten from the session rather than copied.
const REWRITTEN: [&str; 3] = [
    "spike_clusters.npy",
    "cluster_group.tsv",
    sorrel_io::kilosort::TAGS_TSV,
];

/// Template window around each spike (SpikeInterface's defaults).
const MS_BEFORE: f32 = 1.0;
const MS_AFTER: f32 = 2.0;
/// Spikes averaged per template (SpikeInterface's `max_spikes_per_unit`).
const MAX_SPIKES_PER_TEMPLATE: usize = 500;
/// Frames per `trace()` read when streaming `recording.dat`.
const STREAM_FRAMES: u64 = 1 << 16;

/// Write `session` as a phy directory at `out`, created if missing; files
/// of the same names are replaced. `label_str` maps labels to the strings
/// phy expects in `cluster_group.tsv`.
pub fn export_phy<P, F>(
    session: &Session<P>,
    out: &Path,
    copy_raw: bool,
    label_str: F,
) -> Result<()>
where
    P: DataProvider + HasGeometry,
    F: Fn(&P::Label) -> &'static str,
{
    let provider = &session.provider;
    std::fs::create_dir_all(out).with_context(|| format!("create {}", out.display()))?;
    let raw = provider.raw_file();
    let stream = provider.n_samples().0 > 0 && (copy_raw || raw.is_none());
    let dat = out.join("recording.dat");
    // Copying a file onto itself truncates it.
    if provider.phy_dir().is_some_and(|src| same_file(src, out)) {
        bail!(
            "{} is the source directory; export into another one",
            out.display()
        );
    }
    if stream && raw.as_ref().is_some_and(|r| same_file(&r.path, &dat)) {
        bail!(
            "{} is the source recording; export into another directory",
            dat.display()
        );
    }

    match provider.phy_dir() {
        Some(src) => copy_phy_files(src, out)?,
        None => write_arrays(session, out)?,
    }
    write_phy_assignments(session, out, label_str)?;

    let (dat_path, dtype, offset) = if stream {
        ("'recording.dat'".to_owned(), stream_raw(provider, &dat)?, 0)
    } else if let Some(r) = &raw {
        let abs = std::path::absolute(&r.path)
            .with_context(|| format!("resolve {}", r.path.display()))?;
        (python_raw_str(&abs)?, r.dtype, r.offset)
    } else {
        log::warn!("no raw samples to reference; phy will open without traces");
        ("[]".to_owned(), TraceDtype::I16, 0)
    };
    let hp_filtered = if raw.as_ref().is_some_and(|r| r.hp_filtered) {
        "True"
    } else {
        "False"
    };
    let params = format!(
        "dat_path = {dat_path}\nn_channels_dat = {}\ndtype = '{}'\noffset = {offset}\n\
         sample_rate = {:?}\nhp_filtered = {hp_filtered}\n",
        provider.n_channels(),
        dtype.phy_name(),
        provider.sample_rate(),
    );
    atomically(&out.join("params.py"), |tmp| std::fs::write(tmp, params))
}

/// Copy the phy files of `src` (top level only) into `out`.
fn copy_phy_files(src: &Path, out: &Path) -> Result<()> {
    for entry in std::fs::read_dir(src).with_context(|| format!("read {}", src.display()))? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let ext = Path::new(name).extension().and_then(|e| e.to_str());
        let wanted = ext.is_some_and(|e| COPIED_EXTENSIONS.contains(&e))
            && !name.starts_with('.')
            && !REWRITTEN.contains(&name);
        if wanted && entry.file_type()?.is_file() {
            let from = entry.path();
            atomically(&out.join(name), |tmp| std::fs::copy(&from, tmp).map(drop))?;
        }
    }
    Ok(())
}

/// The arrays phy requires, for backends without a phy directory. Everything
/// per spike is in the index's time order; `write_phy_assignments` writes
/// `spike_clusters.npy` in that same order for sessions without a row map.
fn write_arrays<P: DataProvider + HasGeometry>(session: &Session<P>, out: &Path) -> Result<()> {
    let index = session.cluster_index();
    ensure!(
        index.spike_pc_indices().is_empty(),
        "session carries phy row order but its backend has no phy directory"
    );
    let times: Vec<u64> = index.all_spike_times().iter().map(|t| t.0).collect();
    write_1d_u64_atomic(&out.join("spike_times.npy"), &times)?;
    // One template per cluster, so a spike's template is its cluster.
    let clusters: Vec<u32> = index.spike_clusters().iter().map(|c| c.0).collect();
    write_1d_u32_atomic(&out.join("spike_templates.npy"), &clusters)?;
    let (templates, shape) = mean_templates(session);
    write_f32_atomic(&out.join("templates.npy"), &shape, &templates)?;

    // Templates span every raw-dat row in order, and `HasGeometry` indexes
    // positions and shanks by raw-dat row, so phy's channel map is identity.
    let p = &session.provider;
    let n_ch = p.n_channels() as usize;
    let map: Vec<u32> = (0..n_ch as u32).collect();
    write_1d_u32_atomic(&out.join("channel_map.npy"), &map)?;
    let positions: Vec<f32> = match p.channel_positions() {
        pos if pos.len() == n_ch => pos.iter().flatten().copied().collect(),
        // phy's own fallback: a vertical line of contacts.
        _ => (0..n_ch).flat_map(|i| [0.0, i as f32]).collect(),
    };
    write_f32_atomic(&out.join("channel_positions.npy"), &[n_ch, 2], &positions)?;
    if p.channel_shanks().len() == n_ch {
        write_1d_u32_atomic(&out.join("channel_shanks.npy"), p.channel_shanks())?;
    }
    Ok(())
}

/// One mean waveform per cluster over every channel, `(n_clusters, n_t,
/// n_ch)` in C order. Clusters with no usable spikes (merged away, all at
/// the recording edges) stay zero.
fn mean_templates<P: DataProvider>(session: &Session<P>) -> (Vec<f32>, [usize; 3]) {
    let p = &session.provider;
    let sr = p.sample_rate();
    let before = (sr * MS_BEFORE / 1000.0).round() as u64;
    let n_t = before as usize + (sr * MS_AFTER / 1000.0).round() as usize;
    let n_ch = p.n_channels() as usize;
    let n_samples = p.n_samples().0;
    let n_clusters = session.n_clusters() as usize;
    let mut out = vec![0f32; n_clusters * n_t * n_ch];
    let shape = [n_clusters, n_t, n_ch];
    if out.is_empty() {
        return (out, shape);
    }
    if n_samples == 0 {
        log::warn!("no raw samples: templates.npy is all zeros");
        return (out, shape);
    }
    out.par_chunks_mut(n_t * n_ch)
        .enumerate()
        .for_each(|(c, template)| {
            let times = session.spike_times(ClusterId(c as u32));
            // ponytail: a strided subsample is deterministic but can alias
            // with periodic firing; a seeded random draw is the upgrade.
            let step = times.len().div_ceil(MAX_SPIKES_PER_TEMPLATE).max(1);
            let mut sum = vec![0f64; template.len()];
            let mut n = 0u32;
            for t in times.iter().step_by(step) {
                let Some(start) = t.0.checked_sub(before) else {
                    continue;
                };
                if start + n_t as u64 > n_samples {
                    continue;
                }
                let window = p.trace(SampleIndex(start), n_t as u32);
                let window = window.samples.to_f32_centred();
                // A remote mirror returns an empty window while fetching.
                if window.len() != sum.len() {
                    continue;
                }
                sum.iter_mut().zip(window).for_each(|(s, v)| *s += v as f64);
                n += 1;
            }
            if n == 0 {
                return;
            }
            for (dst, s) in template.iter_mut().zip(&sum) {
                *dst = (*s / n as f64) as f32;
            }
            // ponytail: DC removal only, no high-pass, so templates of
            // unfiltered recordings keep slow drift; filter the snippets
            // with sorrel-compute's filter if that shows in phy.
            for ch in 0..n_ch {
                let mean = (0..n_t).map(|i| template[i * n_ch + ch]).sum::<f32>() / n_t as f32;
                (0..n_t).for_each(|i| template[i * n_ch + ch] -= mean);
            }
        });
    (out, shape)
}

/// Stream the whole trace into `path` in the backend's native dtype, which
/// is returned. Fails rather than writing gaps when a window comes back
/// short, which a remote mirror does for bytes still in flight.
fn stream_raw<P: DataProvider>(p: &P, path: &Path) -> Result<TraceDtype> {
    let n_samples = p.n_samples().0;
    let n_ch = p.n_channels() as usize;
    let mut dtype = TraceDtype::I16;
    atomically(path, |tmp| {
        let mut f = BufWriter::new(File::create(tmp)?);
        let mut start = 0;
        while start < n_samples {
            let len = (n_samples - start).min(STREAM_FRAMES);
            let window = p.trace(SampleIndex(start), len as u32);
            if window.samples.len() != len as usize * n_ch {
                return Err(std::io::Error::other(format!(
                    "trace window at sample {start} came back incomplete"
                )));
            }
            dtype = window.samples.dtype();
            f.write_all(window.samples.as_bytes())?;
            start += len;
        }
        f.into_inner().map_err(|e| e.into_error())?.sync_all()
    })?;
    Ok(dtype)
}

/// `path` as a Python raw string literal for `params.py`.
fn python_raw_str(path: &Path) -> Result<String> {
    let s = path
        .to_str()
        .with_context(|| format!("{} is not UTF-8", path.display()))?;
    ensure!(
        !s.contains(['\'', '\n']),
        "{s}: a quote or newline cannot go in params.py; export with copy_raw instead"
    );
    Ok(format!("r'{s}'"))
}

fn same_file(a: &Path, b: &Path) -> bool {
    matches!((a.canonicalize(), b.canonicalize()), (Ok(a), Ok(b)) if a == b)
}

/// Run `write` against `<path>.tmp`, then rename it over `path`; the tmp
/// file is removed on failure so a failed raw copy leaves no stray bytes.
fn atomically(path: &Path, write: impl FnOnce(&Path) -> std::io::Result<()>) -> Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    if let Err(e) = write(&tmp) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e).with_context(|| format!("write {}", tmp.display()));
    }
    std::fs::rename(&tmp, path)
        .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))
}
