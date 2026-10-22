//! Sorrel binary. Detects the backend at startup and instantiates the
//! correctly-typed, monomorphised `SorrelApp<P>`.
//!
//! Backend dispatch is a single match in [`with_session`] — it expands to
//! one fully-monomorphised arm per supported provider type. Adding a
//! backend means adding one arm there plus the provider crate; the trait
//! is *never* used as `dyn DataProvider` on the hot path. Every subcommand
//! is a [`SessionVisitor`] so it gets the same open/seal/replay path.

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use sorrel_data::session::ApplyPhyLabel;
use sorrel_data::{
    apply_rules, baseline_hash, collect_qc_rows, diff_labels, export_qc, rank_merge_candidates,
    rank_split_candidates, save_and_reseal_journal, save_to_phy, serve, write_report, Journal,
    ReportFilter, RuleSet, Session, SuggestConfig,
};
use sorrel_io::kilosort::{read_cluster_group_tsv, KilosortOpenParams, PhyLabel};
use sorrel_io::open_ephys::OebinMeta;
use sorrel_io::spikeglx::SpikeGlxMeta;
use sorrel_io::{DataProvider, HasGeometry, KilosortProvider, SortingAnalyzerProvider, TraceDtype};
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

/// sorrel — spike-sorting curation GUI
///
/// `sorrel <DATA_DIR>` opens the GUI; the subcommands run headlessly on the
/// same data and the same curation journal.
#[derive(Parser, Debug)]
#[command(
    name = "sorrel",
    version,
    args_conflicts_with_subcommands = true,
    subcommand_negates_reqs = true,
    after_help = "When params.py is present, sample-rate / channels / dtype / offset / \
                  dat-path all default from it; flags override per field."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
    /// `sorrel <DATA_DIR> [flags]` with no subcommand means `open`.
    #[command(flatten)]
    open: OpenArgs,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Open the curation GUI (the default when no subcommand is given)
    Open(OpenArgs),
    /// Headless: run every metric and write cluster_qc.tsv + cluster_qc.json
    Qc {
        #[command(flatten)]
        open: OpenArgs,
        /// Output directory (created if missing)
        #[arg(long, value_name = "DIR")]
        out: PathBuf,
    },
    /// Headless: rank merge and split suggestions for the current curation
    Suggest {
        #[command(flatten)]
        open: OpenArgs,
        /// Emit JSON (`{"merges": [...], "splits": [...]}`) instead of a table
        #[arg(long)]
        json: bool,
    },
    /// Headless: label clusters from QC-metric threshold rules (journaled, undoable)
    Apply {
        #[command(flatten)]
        open: OpenArgs,
        /// TOML rule file; see docs/src/usage/rules-example.toml
        #[arg(long, value_name = "FILE")]
        rules: PathBuf,
        /// Print the relabels without journaling them
        #[arg(long)]
        dry_run: bool,
        /// Also write the phy artifacts (same as Ctrl-S in the GUI)
        #[arg(long)]
        save: bool,
    },
    /// Headless: write a self-contained HTML report with one card per cluster
    Report {
        #[command(flatten)]
        open: OpenArgs,
        /// Output file
        #[arg(long, value_name = "FILE", default_value = "sorrel_report.html")]
        out: PathBuf,
        /// Only these cluster ids (comma-separated)
        #[arg(long, value_name = "IDS", value_delimiter = ',')]
        clusters: Option<Vec<u32>>,
        /// Only clusters with this label (good | mua | noise | unsorted)
        #[arg(long, value_name = "LABEL")]
        label: Option<String>,
    },
    /// Serve JSON-RPC 2.0 over stdio (MCP shape) so agents and notebooks can
    /// inspect metrics, request plots and curate through the journal
    Serve {
        #[command(flatten)]
        open: OpenArgs,
    },
    /// Compare two cluster_group.tsv files: confusion table, kappa, disagreements
    Diff {
        a: PathBuf,
        b: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Inspect or replay a curation journal
    Journal {
        #[command(subcommand)]
        action: JournalCmd,
    },
    /// Share a phy dataset with a remote guest: headless host that prints a
    /// pairing code and applies the guest's edits to the local journal
    Share {
        #[command(flatten)]
        open: OpenArgs,
        /// Listen for a direct connection on this port (QUIC + TCP)
        #[arg(
            long,
            value_name = "PORT",
            default_value_t = 7373,
            conflicts_with = "relay"
        )]
        port: u16,
        /// Reach the guest through a sorrel-relay instead:
        /// /dns4/relay.example.org/tcp/443/wss/p2p/<RelayId>
        #[arg(long, value_name = "MULTIADDR")]
        relay: Option<sorrel_remote::Multiaddr>,
    },
    /// Join a shared dataset: sync it into a local directory, stream the raw
    /// trace on demand, and open the GUI
    Join {
        /// The six words printed by `sorrel share`
        #[arg(value_name = "CODE")]
        code: String,
        /// Host address printed by `sorrel share` (omit on the same LAN: mDNS)
        #[arg(long, value_name = "MULTIADDR", conflicts_with = "relay")]
        direct: Option<sorrel_remote::Multiaddr>,
        /// The relay address the host was started with
        #[arg(long, value_name = "MULTIADDR")]
        relay: Option<sorrel_remote::Multiaddr>,
        /// Local directory for the synced dataset
        /// (default: ./sorrel-remote-<first word of the code>)
        #[arg(long, value_name = "DIR")]
        into: Option<PathBuf>,
    },
    /// Remove <DATA_DIR>/.sorrel/cache/
    ClearCache {
        /// Dataset directory (or an .nwb / rez.mat file inside it)
        #[arg(value_name = "DATA_DIR")]
        root: PathBuf,
    },
}

#[derive(Subcommand, Debug)]
enum JournalCmd {
    /// Print every committed record, oldest first
    Dump {
        /// Journal file (default name: <DATA_DIR>/sorrel.journal)
        #[arg(value_name = "JOURNAL")]
        path: PathBuf,
        /// Emit a JSON array instead of one record per line
        #[arg(long)]
        json: bool,
    },
    /// Open the data, replay its journal, and write the phy artifacts
    /// (spike_clusters.npy, cluster_group.tsv, curation.json) into --out
    Replay {
        #[command(flatten)]
        open: OpenArgs,
        /// Output directory (created if missing)
        #[arg(long, value_name = "DIR")]
        out: PathBuf,
    },
}

/// Everything needed to open a dataset and its journal. Shared by the GUI
/// and every headless subcommand.
#[derive(Args, Debug, Clone, Default)]
struct OpenArgs {
    /// phy / Kilosort or SortingAnalyzer directory, or a *.nwb / rez.mat file
    #[arg(value_name = "DATA_DIR", required = true)]
    root: Option<PathBuf>,
    /// kilosort | sorting-analyzer | nwb | ks4-rez (auto-detected by default)
    #[arg(long, value_name = "NAME", value_parser = parse_backend)]
    backend: Option<Backend>,
    /// Raw recording (overrides params.py:dat_path)
    #[arg(long, value_name = "PATH")]
    dat: Option<PathBuf>,
    /// Overrides params.py:sample_rate
    #[arg(long, value_name = "Hz")]
    sample_rate: Option<f32>,
    /// Overrides params.py:n_channels_dat
    #[arg(long = "channels", value_name = "N")]
    n_channels: Option<u32>,
    /// int16 | uint16 | int32 | float32 (overrides params.py:dtype)
    #[arg(long, value_name = "DTYPE", value_parser = parse_dtype)]
    dtype: Option<TraceDtype>,
    /// Header bytes to skip in the dat (overrides params.py:offset)
    #[arg(long, value_name = "BYTES")]
    offset: Option<u64>,
    /// Curation journal (default: <DATA_DIR>/sorrel.journal)
    #[arg(long, value_name = "PATH")]
    journal: Option<PathBuf>,
    /// Delete the existing journal before opening. The escape hatch for a
    /// journal sealed against a spike_clusters.npy that has since changed.
    #[arg(long)]
    discard_journal: bool,
    /// Populate sample-rate / channels / dtype from a SpikeGLX .meta
    #[arg(long, value_name = "PATH")]
    spikeglx_meta: Option<PathBuf>,
    /// Populate sample-rate / channels / dat path from an Open Ephys structure.oebin
    #[arg(long, value_name = "PATH")]
    oebin: Option<PathBuf>,
    /// Deprecated alias for `sorrel qc <DATA_DIR> --out DIR`
    #[arg(long, value_name = "DIR", hide = true)]
    export_qc: Option<PathBuf>,
    /// Deprecated alias for `sorrel clear-cache <DATA_DIR>`
    #[arg(long, hide = true)]
    clear_cache: bool,
}

impl OpenArgs {
    fn root(&self) -> &Path {
        self.root.as_deref().expect("clap marks DATA_DIR required")
    }
}

fn parse_backend(s: &str) -> std::result::Result<Backend, String> {
    Ok(match s {
        "kilosort" | "phy" => Backend::Kilosort,
        "sorting-analyzer" | "spikeinterface" | "si" => Backend::SortingAnalyzer,
        #[cfg(feature = "hdf5")]
        "nwb" => Backend::Nwb,
        #[cfg(feature = "hdf5")]
        "ks4-rez" | "rez" => Backend::Ks4Rez,
        #[cfg(not(feature = "hdf5"))]
        "nwb" | "ks4-rez" | "rez" => {
            return Err(format!(
                "backend {s} requires building with --features hdf5"
            ));
        }
        other => return Err(format!("unknown backend {other}")),
    })
}

fn parse_dtype(s: &str) -> std::result::Result<TraceDtype, String> {
    TraceDtype::from_phy_name(s).ok_or_else(|| format!("unsupported dtype {s:?}"))
}

fn phy_label_str(l: &PhyLabel) -> &'static str {
    l.as_str()
}

/// Resolve the journal path: explicit `--journal`, else `<dir>/<stem>.journal`.
/// Falls back to the legacy `<stem>.sqlite` name when only that file exists
/// (the format was never SQLite; the name was historical).
fn journal_path(open: &OpenArgs, dir: &Path, stem: &str) -> PathBuf {
    if let Some(p) = &open.journal {
        return p.clone();
    }
    let current = dir.join(format!("{stem}.journal"));
    let legacy = dir.join(format!("{stem}.sqlite"));
    if !current.exists() && legacy.exists() {
        log::warn!(
            "using legacy journal {}; rename it to {} at your convenience",
            legacy.display(),
            current.display()
        );
        return legacy;
    }
    current
}

/// Open the journal sealed against `baseline`, honouring `--discard-journal`.
/// A baseline mismatch means the journal's edits would replay onto data they
/// were not recorded against, so this fails closed and names both ways out.
fn open_journal(open: &OpenArgs, path: &Path, baseline: u64, n_clusters: u32) -> Result<Journal> {
    if open.discard_journal && path.exists() {
        std::fs::remove_file(path)
            .with_context(|| format!("discard journal {}", path.display()))?;
        log::warn!("discarded journal {}", path.display());
    }
    Journal::open_or_create(path, baseline, n_clusters).with_context(|| {
        format!(
            "journal {} was sealed against different data (or by an older sorrel without a \
             seal). If that curation is already saved, rerun with --discard-journal; to keep \
             unsaved edits, restore the spike_clusters.npy the journal was recorded against.",
            path.display()
        )
    })
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
                bail!("{} requires building with --features hdf5", root.display());
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
fn apply_recording_overrides(open: &OpenArgs, p: &mut KilosortOpenParams) -> Result<()> {
    if let Some(meta_path) = &open.spikeglx_meta {
        let meta = SpikeGlxMeta::read(meta_path)
            .with_context(|| format!("read spikeglx meta {}", meta_path.display()))?;
        p.sample_rate.get_or_insert(meta.sample_rate);
        p.n_channels.get_or_insert(meta.n_channels);
        p.dtype.get_or_insert(meta.dtype);
    }
    if let Some(oebin_path) = &open.oebin {
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

/// Where a session's artifacts go, how its journal is treated on save, and
/// which backend-specific extras the GUI may use.
struct OpenCtx<P: DataProvider> {
    /// Directory `save_to_phy` writes into.
    save_dir: PathBuf,
    /// True only for backends that read the saved `spike_clusters.npy`
    /// back on open (phy/Kilosort); see `SorrelApp::set_save_dir`.
    reseal_on_save: bool,
    positions: Vec<[f32; 2]>,
    /// Shank id per raw channel (empty = single shank); drives per-shank CMR.
    shanks: Vec<u32>,
    /// Per-spike depth for the drift map; needs PC features + geometry, so
    /// only the phy backend provides it (`sorrel_data::spike_depths`).
    depth_source: Option<sorrel_ui::app::DepthFn<P>>,
}

/// Seeds optional per-spike data and replays the journal. Headless commands
/// run it at once; the GUI runs it on a worker behind a loading screen.
type Seed<P> = Box<dyn FnOnce(&mut Session<P>) -> Result<()> + Send + 'static>;

/// A subcommand body, monomorphised once per backend by [`with_session`].
trait SessionVisitor {
    fn visit<P>(self, session: Session<P>, seed: Seed<P>, ctx: OpenCtx<P>) -> Result<()>
    where
        P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry;
}

/// Run `seed` now and hand back the ready session.
fn seeded<P: DataProvider>(mut session: Session<P>, seed: Seed<P>) -> Result<Session<P>> {
    seed(&mut session)?;
    Ok(session)
}

/// Open the backend named (or detected) by `open`, seal and replay its
/// journal, then hand the typed session to `v`.
fn with_session<V: SessionVisitor>(open: &OpenArgs, v: V) -> Result<()> {
    let backend = match open.backend {
        Some(b) => b,
        None => detect_backend(open.root())?,
    };
    match backend {
        Backend::Kilosort => {
            let mut params = KilosortOpenParams {
                sample_rate: open.sample_rate,
                n_channels: open.n_channels,
                dtype: open.dtype,
                offset: open.offset,
                dat_path: open.dat.clone(),
                ..Default::default()
            };
            apply_recording_overrides(open, &mut params)?;
            let (session, ctx) = open_kilosort(open, params)?;
            v.visit(session, Box::new(seed_kilosort), ctx)
        }
        Backend::SortingAnalyzer => {
            let root = open.root().to_path_buf();
            let provider = SortingAnalyzerProvider::open(&root)?;
            // The analyzer's own sorting is reloaded on every open, so the
            // journal stays the source of truth and is never resealed.
            let baseline = baseline_hash(&provider.identity_bytes());
            let journal_path = journal_path(open, &root, "sorrel");
            let journal = open_journal(open, &journal_path, baseline, provider.n_clusters())?;
            let session = Session::new(provider, journal);
            let positions = session.provider.channel_positions().to_vec();
            let shanks = session.provider.channel_shanks().to_vec();
            v.visit(
                session,
                Box::new(|s| s.replay_journal()),
                OpenCtx {
                    save_dir: root,
                    reseal_on_save: false,
                    positions,
                    shanks,
                    depth_source: None,
                },
            )
        }
        #[cfg(feature = "hdf5")]
        Backend::Nwb => {
            let path = open.root().to_path_buf();
            let provider = sorrel_io::NwbProvider::open(&path)?;
            let (session, ctx) = open_file_backed(open, provider, &path, "nwb")?;
            v.visit(session, Box::new(|s| s.replay_journal()), ctx)
        }
        #[cfg(feature = "hdf5")]
        Backend::Ks4Rez => {
            let path = open.root().to_path_buf();
            let provider = sorrel_io::Ks4RezProvider::open(
                &path,
                sorrel_io::Ks4RezOpenParams {
                    dat_path: open.dat.clone(),
                },
            )?;
            let (session, ctx) = open_file_backed(open, provider, &path, "rez")?;
            v.visit(
                session,
                Box::new(|s| {
                    s.seed_templates();
                    s.replay_journal()
                }),
                ctx,
            )
        }
    }
}

/// Open a phy directory with explicit `params`, sealed against the saved
/// `spike_clusters.npy` (phy dirs read it back, so saves reseal). The
/// session comes back **unseeded**: run [`seed_kilosort`] (or hand it to
/// the GUI loader) before using it.
fn open_kilosort(
    open: &OpenArgs,
    params: KilosortOpenParams,
) -> Result<(Session<KilosortProvider>, OpenCtx<KilosortProvider>)> {
    let root = open.root().to_path_buf();
    let provider = KilosortProvider::open(&root, params)?;
    let clusters_path = root.join("spike_clusters.npy");
    let baseline = baseline_hash(
        &std::fs::read(&clusters_path)
            .with_context(|| format!("read {}", clusters_path.display()))?,
    );
    let journal_path = journal_path(open, &root, "sorrel");
    let journal = open_journal(open, &journal_path, baseline, provider.n_clusters())?;
    let session = Session::new(provider, journal);
    let positions = session.provider.channel_positions().to_vec();
    let shanks = session.provider.channel_shanks().to_vec();
    Ok((
        session,
        OpenCtx {
            save_dir: root,
            reseal_on_save: true,
            positions,
            shanks,
            depth_source: Some(sorrel_data::spike_depths),
        },
    ))
}

/// Pull the optional per-spike arrays into the index and replay the journal.
fn seed_kilosort(session: &mut Session<KilosortProvider>) -> Result<()> {
    session.seed_amplitudes();
    session.seed_templates();
    session.seed_pc_features();
    session.seed_template_waveforms();
    session.replay_journal()
}

/// `sorrel share <DIR> [--port N | --relay ADDR]`: headless host.
fn run_share(open: &OpenArgs, port: u16, relay: Option<sorrel_remote::Multiaddr>) -> Result<()> {
    let backend = match open.backend {
        Some(b) => b,
        None => detect_backend(open.root())?,
    };
    if backend != Backend::Kilosort {
        // ponytail: other backends map the recording differently (or load it
        // into RAM); add them when someone needs to share one.
        bail!("remote share supports kilosort/phy datasets");
    }
    let mut params = KilosortOpenParams {
        sample_rate: open.sample_rate,
        n_channels: open.n_channels,
        dtype: open.dtype,
        offset: open.offset,
        dat_path: open.dat.clone(),
        ..Default::default()
    };
    apply_recording_overrides(open, &mut params)?;
    let (mut session, ctx) = open_kilosort(open, params)?;
    seed_kilosort(&mut session)?;
    let p = &session.provider;
    let dat = sorrel_remote::DatSource {
        path: p.dat_path().to_path_buf(),
        offset: p.dat_offset(),
        dtype: p.dtype(),
        n_channels: p.n_channels(),
        sample_rate: p.sample_rate(),
    };
    let mode = match relay {
        Some(relay) => sorrel_remote::Mode::Relay(relay),
        None => sorrel_remote::Mode::Direct { port, dial: None },
    };
    sorrel_remote::share(
        Box::new(sorrel_remote::SessionServed(session)),
        ctx.save_dir,
        dat,
        mode,
    )
}

/// `sorrel join <CODE> [--direct ADDR | --relay ADDR] [--into DIR]`: guest GUI.
fn run_join(
    code: &str,
    direct: Option<sorrel_remote::Multiaddr>,
    relay: Option<sorrel_remote::Multiaddr>,
    into: Option<PathBuf>,
) -> Result<()> {
    let code = sorrel_remote::PairingCode::parse(code)?;
    let mode = match relay {
        Some(relay) => sorrel_remote::Mode::Relay(relay),
        None => sorrel_remote::Mode::Direct {
            port: 0,
            dial: direct,
        },
    };
    let into = into.unwrap_or_else(|| {
        let first = code.as_str().split(' ').next().unwrap_or("guest");
        PathBuf::from(format!("sorrel-remote-{first}"))
    });
    let joined = sorrel_remote::join(&code, &mode, &into)?;
    // The host tracks our journal cursor, so the guest never reseals: a
    // reseal would reset the record count and every later record would be
    // refused as diverged. Saving still writes the phy files locally.
    let ctx = OpenCtx {
        save_dir: joined.root,
        reseal_on_save: false,
        positions: joined.positions,
        shanks: Vec::new(),
        depth_source: Some(sorrel_data::spike_depths),
    };
    run_app(joined.session, Box::new(|_| Ok(())), ctx)
}

/// Shared open path for single-file backends (`.nwb`, `rez.mat`): the
/// journal and save artifacts live next to the file, named after it.
#[cfg(feature = "hdf5")]
fn open_file_backed<P>(
    open: &OpenArgs,
    provider: P,
    path: &Path,
    fallback_stem: &str,
) -> Result<(Session<P>, OpenCtx<P>)>
where
    P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
{
    let parent = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let stem = format!(
        "{}.sorrel",
        path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(fallback_stem)
    );
    let journal_path = journal_path(open, &parent, &stem);
    let baseline = baseline_hash(&provider.identity_bytes());
    let journal = open_journal(open, &journal_path, baseline, provider.n_clusters())?;
    let session = Session::new(provider, journal);
    let positions = session.provider.channel_positions().to_vec();
    let shanks = session.provider.channel_shanks().to_vec();
    Ok((
        session,
        OpenCtx {
            save_dir: parent,
            reseal_on_save: false,
            positions,
            shanks,
            depth_source: None,
        },
    ))
}

// ---------------------------------------------------------------- commands

struct Gui;
impl SessionVisitor for Gui {
    fn visit<P>(self, session: Session<P>, seed: Seed<P>, ctx: OpenCtx<P>) -> Result<()>
    where
        P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
    {
        run_app(session, seed, ctx)
    }
}

struct Qc(PathBuf);
impl SessionVisitor for Qc {
    fn visit<P>(self, session: Session<P>, seed: Seed<P>, _: OpenCtx<P>) -> Result<()>
    where
        P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
    {
        let session = seeded(session, seed)?;
        let out_dir = &self.0;
        let n = export_qc(&session, out_dir)
            .with_context(|| format!("exporting QC to {}", out_dir.display()))?;
        println!("exported QC for {n} clusters into {}", out_dir.display());
        Ok(())
    }
}

struct Suggest {
    json: bool,
}
impl SessionVisitor for Suggest {
    fn visit<P>(self, session: Session<P>, seed: Seed<P>, _: OpenCtx<P>) -> Result<()>
    where
        P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
    {
        let session = seeded(session, seed)?;
        let cfg = SuggestConfig::default();
        let merges = rank_merge_candidates(&session, &cfg);
        let splits = rank_split_candidates(&session, &cfg);
        if self.json {
            let doc = serde_json::json!({ "merges": merges, "splits": splits });
            serde_json::to_writer_pretty(std::io::stdout().lock(), &doc)?;
            println!();
            return Ok(());
        }
        println!("merges (score, a, b, ccg_dip_z, amp_ks)");
        for m in &merges {
            println!(
                "  {:.3}\t{}\t{}\t{:.2}\t{:.3}",
                m.score, m.a.0, m.b.0, m.ccg_dip_z, m.amp_ks
            );
        }
        println!("splits (score, cluster, bimodality, dip, contamination, gmm_bic_delta)");
        for s in &splits {
            println!(
                "  {:.3}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.1}",
                s.score, s.cluster.0, s.amp_bimodality, s.amp_dip, s.contamination, s.gmm_bic_delta
            );
        }
        Ok(())
    }
}

struct Replay(PathBuf);
impl SessionVisitor for Replay {
    fn visit<P>(self, session: Session<P>, seed: Seed<P>, _: OpenCtx<P>) -> Result<()>
    where
        P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
    {
        let session = seeded(session, seed)?;
        std::fs::create_dir_all(&self.0).with_context(|| format!("create {}", self.0.display()))?;
        save_to_phy(&session, &self.0, phy_label_str)?;
        println!(
            "replayed {} journal records into {}",
            session.history_len(),
            self.0.display()
        );
        Ok(())
    }
}

struct Apply {
    rules: PathBuf,
    dry_run: bool,
    save: bool,
}
impl SessionVisitor for Apply {
    fn visit<P>(self, session: Session<P>, seed: Seed<P>, ctx: OpenCtx<P>) -> Result<()>
    where
        P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
    {
        let mut session = seeded(session, seed)?;
        let rules = RuleSet::from_path(&self.rules)?;
        let rows = collect_qc_rows(&session);
        let current = |c: sorrel_io::ClusterId| session.label(c).unwrap_or_default().as_str();
        let cmds = apply_rules(&rules, &rows, current)?;
        for cmd in &cmds {
            if let sorrel_data::CurationCommand::Relabel { cluster, op } = cmd {
                println!(
                    "{}\t{} -> {}",
                    cluster.0,
                    current(*cluster),
                    P::label_from_op(*op).as_str()
                );
            }
        }
        println!(
            "{} relabels{}",
            cmds.len(),
            if self.dry_run { " (dry run)" } else { "" }
        );
        if self.dry_run || cmds.is_empty() {
            return Ok(());
        }
        // One batch: a single Ctrl-Z in the GUI reverts the whole rule pass.
        if let Some(batch) = sorrel_data::CurationCommand::batch(cmds) {
            session.dispatch(batch)?;
        }
        if self.save {
            if ctx.reseal_on_save {
                let journal_path = session.journal_path().to_path_buf();
                let journal =
                    save_and_reseal_journal(&session, &ctx.save_dir, &journal_path, phy_label_str)?;
                session.reseal(journal);
            } else {
                save_to_phy(&session, &ctx.save_dir, phy_label_str)?;
            }
            println!("saved → {}", ctx.save_dir.display());
        }
        Ok(())
    }
}

struct Report {
    out: PathBuf,
    filter: ReportFilter,
}
impl SessionVisitor for Report {
    fn visit<P>(self, session: Session<P>, seed: Seed<P>, _: OpenCtx<P>) -> Result<()>
    where
        P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
    {
        let session = seeded(session, seed)?;
        let n = write_report(&session, &self.out, phy_label_str, &self.filter)?;
        println!("wrote {n} cluster cards → {}", self.out.display());
        Ok(())
    }
}

struct Serve;
impl SessionVisitor for Serve {
    fn visit<P>(self, session: Session<P>, seed: Seed<P>, ctx: OpenCtx<P>) -> Result<()>
    where
        P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
    {
        let mut session = seeded(session, seed)?;
        let stdin = std::io::stdin();
        let stdout = std::io::stdout();
        serve(
            &mut session,
            &ctx.save_dir,
            ctx.reseal_on_save,
            phy_label_str,
            stdin.lock(),
            stdout.lock(),
        )
    }
}

fn diff_tsvs(a: &Path, b: &Path, json: bool) -> Result<()> {
    let read =
        |p: &Path| read_cluster_group_tsv(p).with_context(|| format!("read {}", p.display()));
    let d = diff_labels(&read(a)?, &read(b)?);
    if json {
        serde_json::to_writer_pretty(std::io::stdout().lock(), &d)?;
        println!();
    } else {
        print!("{}", d.render());
    }
    Ok(())
}

fn journal_dump(path: &Path, json: bool) -> Result<()> {
    let records = Journal::replay_at(path)?;
    if json {
        serde_json::to_writer_pretty(std::io::stdout().lock(), &records)?;
        println!();
    } else {
        for (seq, cmd) in records.iter().enumerate() {
            println!("{seq}\t{cmd:?}");
        }
    }
    Ok(())
}

fn clear_cache(root: &Path) -> Result<()> {
    let dataset_dir = if root.is_file() {
        root.parent().unwrap_or_else(|| Path::new("."))
    } else {
        root
    };
    sorrel_cache::CacheStore::clear_dataset_cache(dataset_dir)
        .with_context(|| format!("clear cache under {}", dataset_dir.display()))?;
    println!(
        "cleared cache at {}",
        sorrel_cache::CacheStore::cache_root_for(dataset_dir).display()
    );
    Ok(())
}

fn run_app<P>(session: Session<P>, seed: Seed<P>, ctx: OpenCtx<P>) -> Result<()>
where
    P: DataProvider<Label = PhyLabel> + ApplyPhyLabel,
{
    // Seeding and replay run on a worker behind the loading screen.
    let mut app = SorrelApp::with_loader(session, phy_label_str, seed);
    app.set_save_dir(ctx.save_dir, ctx.reseal_on_save);
    if !ctx.positions.is_empty() {
        app.set_channel_positions(ctx.positions);
    }
    if !ctx.shanks.is_empty() {
        app.set_channel_shanks(ctx.shanks);
    }
    if let Some(f) = ctx.depth_source {
        app.set_depth_source(f);
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
                app.install_gpu_compute(rs.device.clone().into(), rs.queue.clone().into());
            } else {
                log::warn!("no wgpu render state available; trace view will not render");
            }
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow::anyhow!("eframe: {e}"))
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli = Cli::parse();
    match cli.cmd {
        None => run_open(cli.open),
        Some(Cmd::Open(open)) => run_open(open),
        Some(Cmd::Qc { open, out }) => with_session(&open, Qc(out)),
        Some(Cmd::Suggest { open, json }) => with_session(&open, Suggest { json }),
        Some(Cmd::Apply {
            open,
            rules,
            dry_run,
            save,
        }) => with_session(
            &open,
            Apply {
                rules,
                dry_run,
                save,
            },
        ),
        Some(Cmd::Diff { a, b, json }) => diff_tsvs(&a, &b, json),
        Some(Cmd::Serve { open }) => with_session(&open, Serve),
        Some(Cmd::Report {
            open,
            out,
            clusters,
            label,
        }) => with_session(
            &open,
            Report {
                out,
                filter: ReportFilter { clusters, label },
            },
        ),
        Some(Cmd::Journal {
            action: JournalCmd::Dump { path, json },
        }) => journal_dump(&path, json),
        Some(Cmd::Journal {
            action: JournalCmd::Replay { open, out },
        }) => with_session(&open, Replay(out)),
        Some(Cmd::Share { open, port, relay }) => run_share(&open, port, relay),
        Some(Cmd::Join {
            code,
            direct,
            relay,
            into,
        }) => run_join(&code, direct, relay, into),
        Some(Cmd::ClearCache { root }) => clear_cache(&root),
    }
}

/// `sorrel <DATA_DIR>` plus the two deprecated flag spellings of `qc` and
/// `clear-cache`, kept so existing scripts (and `sorrel-py`) keep working.
fn run_open(open: OpenArgs) -> Result<()> {
    if open.clear_cache {
        return clear_cache(open.root());
    }
    if let Some(out) = open.export_qc.clone() {
        log::warn!("--export-qc is deprecated; use `sorrel qc <DATA_DIR> --out DIR`");
        return with_session(&open, Qc(out));
    }
    with_session(&open, Gui)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_directory_with_flags_means_open() {
        let cli = Cli::try_parse_from(["sorrel", "/x", "--sample-rate", "3e4", "--channels", "8"])
            .unwrap();
        assert!(cli.cmd.is_none());
        assert_eq!(cli.open.root(), Path::new("/x"));
        assert_eq!(cli.open.sample_rate, Some(30000.0));
        assert_eq!(cli.open.n_channels, Some(8));
        assert_eq!(cli.open.backend, None);
    }

    #[test]
    fn subcommands_parse_and_deprecated_aliases_survive() {
        let cli = Cli::try_parse_from(["sorrel", "suggest", "/x", "--json"]).unwrap();
        assert!(matches!(cli.cmd, Some(Cmd::Suggest { json: true, .. })));

        let cli =
            Cli::try_parse_from(["sorrel", "qc", "/x", "--out", "o", "--backend", "si"]).unwrap();
        match cli.cmd {
            Some(Cmd::Qc { open, out }) => {
                assert_eq!(out, PathBuf::from("o"));
                assert_eq!(open.backend, Some(Backend::SortingAnalyzer));
            }
            other => panic!("unexpected {other:?}"),
        }

        let cli =
            Cli::try_parse_from(["sorrel", "journal", "dump", "j.journal", "--json"]).unwrap();
        assert!(matches!(
            cli.cmd,
            Some(Cmd::Journal {
                action: JournalCmd::Dump { json: true, .. }
            })
        ));

        let cli = Cli::try_parse_from(["sorrel", "/x", "--export-qc", "out"]).unwrap();
        assert_eq!(cli.open.export_qc, Some(PathBuf::from("out")));
        let cli = Cli::try_parse_from(["sorrel", "--clear-cache", "/x"]).unwrap();
        assert!(cli.open.clear_cache);

        assert!(
            Cli::try_parse_from(["sorrel"]).is_err(),
            "DATA_DIR is required"
        );
        assert!(Cli::try_parse_from(["sorrel", "/x", "--dtype", "int7"]).is_err());
    }
}
