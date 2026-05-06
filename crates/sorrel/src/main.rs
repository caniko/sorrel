//! Sorrel binary. Detects the backend at startup and instantiates the
//! correctly-typed, monomorphised `SorrelApp<P>`.

use anyhow::{bail, Context, Result};
use sorrel_data::{Session, SqliteJournal};
use sorrel_io::kilosort::{KilosortOpenParams, PhyLabel};
use sorrel_io::KilosortProvider;
use sorrel_ui::SorrelApp;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
struct Args {
    root: PathBuf,
    dat: PathBuf,
    sample_rate: f32,
    n_channels: u32,
    journal: PathBuf,
}

fn parse_args() -> Result<Args> {
    let mut root: Option<PathBuf> = None;
    let mut dat: Option<PathBuf> = None;
    let mut sample_rate: f32 = 30_000.0;
    let mut n_channels: u32 = 32;
    let mut journal: Option<PathBuf> = None;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--dat" => dat = Some(it.next().context("--dat needs a path")?.into()),
            "--sample-rate" => {
                sample_rate = it.next().context("--sample-rate needs a value")?.parse()?
            }
            "--channels" => n_channels = it.next().context("--channels needs a value")?.parse()?,
            "--journal" => journal = Some(it.next().context("--journal needs a path")?.into()),
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            other if root.is_none() => root = Some(other.into()),
            other => bail!("unexpected argument {other}"),
        }
    }

    let root = root.context("missing kilosort directory (positional arg)")?;
    let dat = dat.unwrap_or_else(|| root.join("recording.dat"));
    let journal = journal.unwrap_or_else(|| root.join("sorrel.sqlite"));

    Ok(Args {
        root,
        dat,
        sample_rate,
        n_channels,
        journal,
    })
}

fn print_help() {
    println!(
        "sorrel — spike-sorting curation GUI\n\n\
         USAGE:\n  sorrel <KILOSORT_DIR> [--dat PATH] [--sample-rate Hz] [--channels N] [--journal PATH]\n"
    );
}

fn detect_backend(root: &Path) -> Result<&'static str> {
    if root.join("spike_times.npy").exists() && root.join("spike_clusters.npy").exists() {
        return Ok("kilosort");
    }
    bail!(
        "could not detect a supported backend in {} (expected kilosort outputs)",
        root.display()
    );
}

fn run_kilosort(args: Args) -> Result<()> {
    let provider = KilosortProvider::open(
        &args.root,
        KilosortOpenParams {
            sample_rate: args.sample_rate,
            n_channels: args.n_channels,
            dat_path: args.dat.clone(),
        },
    )?;
    let journal = SqliteJournal::open(&args.journal)?;
    let mut session = Session::new(provider, journal);
    session.replay_journal()?;

    let app = SorrelApp::new(session, |l: &PhyLabel| l.as_str());

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 800.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Sorrel",
        native_options,
        Box::new(|_cc| Ok(Box::new(app))),
    )
    .map_err(|e| anyhow::anyhow!("eframe: {e}"))
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = parse_args()?;
    match detect_backend(&args.root)? {
        "kilosort" => run_kilosort(args),
        other => bail!("backend {other} not yet implemented"),
    }
}
