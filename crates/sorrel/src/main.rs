//! Sorrel binary. Detects the backend at startup and instantiates the
//! correctly-typed, monomorphised `SorrelApp<P>`.
//!
//! Backend dispatch is a single match in [`main`] — it expands to one
//! fully-monomorphised arm per supported provider type. Adding a backend
//! means adding one arm here plus the provider crate; the trait is *never*
//! used as `dyn DataProvider` on the hot path.

use anyhow::{bail, Context, Result};
use sorrel_data::{export_qc, Session, SqliteJournal};
use sorrel_io::kilosort::{KilosortOpenParams, PhyLabel};
use sorrel_io::open_ephys::OebinMeta;
use sorrel_io::spikeglx::SpikeGlxMeta;
use sorrel_io::{
    DataProvider, HasGeometry, KilosortProvider, SortingAnalyzerProvider, TraceDtype,
};
use sorrel_ui::{RasterPipeline, SorrelApp, TracePipeline};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum Backend {
    Kilosort,
    SortingAnalyzer,
    #[cfg(feature = "hdf5")]
    Nwb,
    #[cfg(feature = "hdf5")]
    Ks4Rez,
}

#[derive(Debug, Clone, Default)]
struct Args {
    root: Option<PathBuf>,
    backend: Option<Backend>,
    dat: Option<PathBuf>,
    sample_rate: Option<f32>,
    n_channels: Option<u32>,
    dtype: Option<TraceDtype>,
    offset: Option<u64>,
    journal: Option<PathBuf>,
    /// Path to a SpikeGLX `.meta` file. When set, `sample_rate`, `n_channels`,
    /// and `dtype` default from it (CLI flags still override).
    spikeglx_meta: Option<PathBuf>,
    /// Path to an Open Ephys `structure.oebin`. Picks the first continuous
    /// stream and feeds its parameters as Kilosort overrides.
    oebin: Option<PathBuf>,
    /// Headless QC export: run every metric on the loaded session and
    /// write `cluster_qc.tsv` + `cluster_qc.json` into the given dir,
    /// then exit. No GUI is spawned.
    export_qc: Option<PathBuf>,
}

fn parse_args() -> Result<Args> {
    let mut args = Args::default();

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--dat" => args.dat = Some(it.next().context("--dat needs a path")?.into()),
            "--sample-rate" => {
                args.sample_rate =
                    Some(it.next().context("--sample-rate needs a value")?.parse()?)
            }
            "--channels" => {
                args.n_channels = Some(it.next().context("--channels needs a value")?.parse()?)
            }
            "--dtype" => {
                let v = it.next().context("--dtype needs a value")?;
                args.dtype = Some(
                    TraceDtype::from_phy_name(&v)
                        .with_context(|| format!("unsupported --dtype {v:?}"))?,
                );
            }
            "--offset" => {
                args.offset = Some(it.next().context("--offset needs a value")?.parse()?)
            }
            "--journal" => args.journal = Some(it.next().context("--journal needs a path")?.into()),
            "--backend" => {
                let v = it.next().context("--backend needs a value")?;
                args.backend = Some(match v.as_str() {
                    "kilosort" | "phy" => Backend::Kilosort,
                    "sorting-analyzer" | "spikeinterface" | "si" => Backend::SortingAnalyzer,
                    #[cfg(feature = "hdf5")]
                    "nwb" => Backend::Nwb,
                    #[cfg(feature = "hdf5")]
                    "ks4-rez" | "rez" => Backend::Ks4Rez,
                    #[cfg(not(feature = "hdf5"))]
                    "nwb" | "ks4-rez" | "rez" => bail!(
                        "backend {v} requires building with --features hdf5"
                    ),
                    other => bail!("unknown backend {other}"),
                });
            }
            "--spikeglx-meta" => {
                args.spikeglx_meta =
                    Some(it.next().context("--spikeglx-meta needs a path")?.into())
            }
            "--oebin" => args.oebin = Some(it.next().context("--oebin needs a path")?.into()),
            "--export-qc" => {
                args.export_qc =
                    Some(it.next().context("--export-qc needs a path")?.into())
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            other if args.root.is_none() => args.root = Some(other.into()),
            other => bail!("unexpected argument {other}"),
        }
    }

    if args.root.is_none() {
        bail!("missing data directory (positional arg)");
    }
    Ok(args)
}

fn print_help() {
    println!(
        "sorrel — spike-sorting curation GUI\n\n\
         USAGE:\n  sorrel <DATA_DIR> [OPTIONS]\n\n\
         OPTIONS:\n\
         \x20 --backend NAME       kilosort | sorting-analyzer (auto-detected by default)\n\
         \x20 --dat PATH           raw recording (overrides params.py:dat_path)\n\
         \x20 --sample-rate Hz     overrides params.py:sample_rate\n\
         \x20 --channels N         overrides params.py:n_channels_dat\n\
         \x20 --dtype DTYPE        int16|uint16|int32|float32 (overrides params.py:dtype)\n\
         \x20 --offset BYTES       header bytes to skip in the dat (overrides params.py:offset)\n\
         \x20 --journal PATH       SQLite curation log (default: <root>/sorrel.sqlite)\n\
         \x20 --spikeglx-meta P    populate sample-rate/channels/dtype from a SpikeGLX .meta\n\
         \x20 --oebin PATH         populate sample-rate/channels from an Open Ephys structure.oebin\n\
         \x20 --export-qc DIR      headless: write cluster_qc.tsv + cluster_qc.json into DIR and exit\n\
         \n\
         When params.py is present, sample-rate / channels / dtype / offset / dat-path\n\
         all default from it; CLI flags override on a per-field basis."
    );
}

/// Detect which backend a path looks like. Phy markers take precedence
/// because they're the most specific; HDF5 backends are leaf-file based
/// (`.nwb` / `rez.mat`) so they trigger only when the path is a file.
fn detect_backend(root: &Path) -> Result<Backend> {
    if root.is_file() {
        let name = root.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let ext = root
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        #[cfg(feature = "hdf5")]
        {
            if ext == "nwb" {
                return Ok(Backend::Nwb);
            }
            if name == "rez.mat" || name == "rez2.mat" {
                return Ok(Backend::Ks4Rez);
            }
        }
        #[cfg(not(feature = "hdf5"))]
        {
            if ext == "nwb" || name == "rez.mat" || name == "rez2.mat" {
                bail!(
                    "{} requires building with --features hdf5",
                    root.display()
                );
            }
        }
        bail!("unrecognised file {}; expected a directory", root.display());
    }

    if root.join("spike_times.npy").exists() && root.join("spike_clusters.npy").exists() {
        return Ok(Backend::Kilosort);
    }
    if (root.join("recording.json").exists() || root.join("binary.json").exists())
        && root.join("sorting").is_dir()
    {
        return Ok(Backend::SortingAnalyzer);
    }
    bail!(
        "could not detect a supported backend in {} \
         (looked for kilosort outputs and a SortingAnalyzer folder)",
        root.display()
    );
}

/// Apply SpikeGLX/Open Ephys metadata overrides to a `KilosortOpenParams`,
/// preserving any explicit CLI overrides the user supplied.
fn apply_recording_overrides(args: &Args, p: &mut KilosortOpenParams) -> Result<()> {
    if let Some(meta_path) = &args.spikeglx_meta {
        let meta = SpikeGlxMeta::read(meta_path)
            .with_context(|| format!("read spikeglx meta {}", meta_path.display()))?;
        p.sample_rate.get_or_insert(meta.sample_rate);
        p.n_channels.get_or_insert(meta.n_channels);
        p.dtype.get_or_insert(meta.dtype);
    }
    if let Some(oebin_path) = &args.oebin {
        let oebin = OebinMeta::read(oebin_path)
            .with_context(|| format!("read oebin {}", oebin_path.display()))?;
        let stream = oebin
            .primary()
            .ok_or_else(|| anyhow::anyhow!("oebin had no continuous streams"))?;
        p.sample_rate.get_or_insert(stream.sample_rate);
        p.n_channels.get_or_insert(stream.n_channels);
        p.dtype.get_or_insert(sorrel_io::OebinStream::DTYPE);
        if p.dat_path.is_none() {
            p.dat_path = Some(stream.dat_path(oebin_path.parent().unwrap_or(Path::new("."))));
        }
    }
    Ok(())
}

fn run_kilosort(args: Args) -> Result<()> {
    let root = args.root.clone().expect("validated in parse_args");
    let mut params = KilosortOpenParams {
        sample_rate: args.sample_rate,
        n_channels: args.n_channels,
        dtype: args.dtype,
        offset: args.offset,
        dat_path: args.dat.clone(),
    };
    apply_recording_overrides(&args, &mut params)?;

    let provider = KilosortProvider::open(&root, params)?;
    let journal_path = args
        .journal
        .clone()
        .unwrap_or_else(|| root.join("sorrel.sqlite"));
    let journal = SqliteJournal::open(&journal_path)?;
    let mut session = Session::new(provider, journal);
    session.seed_amplitudes();
    session.seed_templates();
    session.seed_pc_features();
    session.seed_template_waveforms();
    session.replay_journal()?;

    if let Some(out_dir) = args.export_qc.clone() {
        return run_export_qc(&session, &out_dir);
    }

    let positions = session.provider.channel_positions().to_vec();
    run_app(session, root, positions, |l: &PhyLabel| l.as_str())
}

fn run_sorting_analyzer(args: Args) -> Result<()> {
    let root = args.root.clone().expect("validated in parse_args");
    let provider = SortingAnalyzerProvider::open(&root)?;
    let journal_path = args
        .journal
        .clone()
        .unwrap_or_else(|| root.join("sorrel.sqlite"));
    let journal = SqliteJournal::open(&journal_path)?;
    let mut session = Session::new(provider, journal);
    session.replay_journal()?;

    if let Some(out_dir) = args.export_qc.clone() {
        return run_export_qc(&session, &out_dir);
    }

    let positions = session.provider.channel_positions().to_vec();
    run_app(session, root, positions, |l: &PhyLabel| l.as_str())
}

/// Headless QC export shared across backends. Runs the full metrics
/// pipeline on the loaded `session` and writes both TSV + JSON outputs.
fn run_export_qc<P>(session: &Session<P>, out_dir: &Path) -> Result<()>
where
    P: DataProvider,
{
    let n = export_qc(session, out_dir)
        .with_context(|| format!("exporting QC to {}", out_dir.display()))?;
    log::info!(
        "exported QC for {n} clusters → {} ({}/cluster_qc.tsv, {}/cluster_qc.json)",
        out_dir.display(),
        out_dir.display(),
        out_dir.display(),
    );
    println!(
        "exported QC for {n} clusters into {}",
        out_dir.display()
    );
    Ok(())
}

fn run_app<P>(
    session: Session<P>,
    save_dir: PathBuf,
    positions: Vec<[f32; 2]>,
    label_str: fn(&P::Label) -> &'static str,
) -> Result<()>
where
    P: DataProvider + sorrel_data::session::ApplyPhyLabel,
{
    let mut app = SorrelApp::new(session, label_str);
    app.set_save_dir(save_dir);
    if !positions.is_empty() {
        app.set_channel_positions(positions);
    }

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 800.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Sorrel",
        native_options,
        Box::new(move |cc| {
            let mut app = app;
            if let Some(rs) = cc.wgpu_render_state.as_ref() {
                TracePipeline::install(rs);
                RasterPipeline::install(rs);
                app.install_gpu_compute(rs.device.clone(), rs.queue.clone());
            } else {
                log::warn!(
                    "no wgpu render state available; trace view will not render"
                );
            }
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow::anyhow!("eframe: {e}"))
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = parse_args()?;
    let root = args.root.as_deref().expect("validated in parse_args");
    let backend = match args.backend {
        Some(b) => b,
        None => detect_backend(root)?,
    };
    match backend {
        Backend::Kilosort => run_kilosort(args),
        Backend::SortingAnalyzer => run_sorting_analyzer(args),
        #[cfg(feature = "hdf5")]
        Backend::Nwb => run_nwb(args),
        #[cfg(feature = "hdf5")]
        Backend::Ks4Rez => run_ks4(args),
    }
}

#[cfg(feature = "hdf5")]
fn run_nwb(args: Args) -> Result<()> {
    use sorrel_io::NwbProvider;
    let path = args.root.clone().expect("validated in parse_args");
    let provider = NwbProvider::open(&path)?;
    let parent = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let journal_path = args.journal.clone().unwrap_or_else(|| {
        parent.join(format!(
            "{}.sorrel.sqlite",
            path.file_name().and_then(|s| s.to_str()).unwrap_or("nwb")
        ))
    });
    let journal = SqliteJournal::open(&journal_path)?;
    let mut session = Session::new(provider, journal);
    session.replay_journal()?;
    let positions = session.provider.channel_positions().to_vec();
    run_app(session, parent, positions, |l: &PhyLabel| l.as_str())
}

#[cfg(feature = "hdf5")]
fn run_ks4(args: Args) -> Result<()> {
    use sorrel_io::{Ks4RezOpenParams, Ks4RezProvider};
    let path = args.root.clone().expect("validated in parse_args");
    let provider = Ks4RezProvider::open(
        &path,
        Ks4RezOpenParams {
            dat_path: args.dat.clone(),
        },
    )?;
    let parent = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let journal_path = args.journal.clone().unwrap_or_else(|| {
        parent.join(format!(
            "{}.sorrel.sqlite",
            path.file_name().and_then(|s| s.to_str()).unwrap_or("rez")
        ))
    });
    let journal = SqliteJournal::open(&journal_path)?;
    let mut session = Session::new(provider, journal);
    session.seed_templates();
    session.replay_journal()?;
    let positions = session.provider.channel_positions().to_vec();
    run_app(session, parent, positions, |l: &PhyLabel| l.as_str())
}
