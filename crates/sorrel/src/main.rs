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
    rank_split_candidates, save_session as save_dir_session, save_to_phy, serve, write_report,
    Journal, ReportFilter, RuleSet, Session, SuggestConfig,
};
use sorrel_io::kilosort::{read_cluster_group_tsv, KilosortOpenParams, PhyLabel};
use sorrel_io::open_ephys::OebinMeta;
use sorrel_io::spikeglx::SpikeGlxMeta;
use sorrel_io::{
    detect_sorter, DataProvider, Events, HasGeometry, HasQualityMetrics, KilosortProvider,
    SortingAnalyzerProvider, TraceDtype,
};
use sorrel_ui::{RasterPipeline, ScatterPipeline, SorrelApp, TracePipeline};
use std::path::{Path, PathBuf};

mod rescue;

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
    /// Run the session host without the GUI: MCP clients connect with
    /// `sorrel mcp <DATA_DIR>`. Stops on Ctrl-C.
    #[arg(long)]
    headless: bool,
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
        #[arg(long, value_name = "FILE", required_unless_present = "curation")]
        rules: Option<PathBuf>,
        /// SpikeInterface curation.json (labels, `tags`, merges) to import,
        /// e.g. the one `sorrel auto` writes
        #[arg(long, value_name = "FILE", conflicts_with = "rules")]
        curation: Option<PathBuf>,
        /// Print the relabels without journaling them
        #[arg(long)]
        dry_run: bool,
        /// Also write the phy artifacts (same as Ctrl-S in the GUI)
        #[arg(long)]
        save: bool,
    },
    /// Headless: automatic curation with SpikeInterface (cleanup, merge
    /// presets, Bombcell + UnitRefine labels, optional Claude tiebreak),
    /// applied as one undoable batch; every decision is tagged with its
    /// source, and anything uncertain gets a `review:*` tag
    Auto {
        #[command(flatten)]
        open: OpenArgs,
        /// Ask Claude to settle units the classifiers disagree on and merges
        /// only one preset found (API key or `ant auth login` required)
        #[arg(long)]
        llm: bool,
        /// Claude model for --llm
        #[arg(long, value_name = "MODEL", default_value = "claude-opus-5")]
        llm_model: String,
        /// Microvolts per ADC bit (e.g. 2.34375 for Neuropixels 1.0 at gain
        /// 500). Bombcell's amplitude threshold is in uV; without this (or
        /// --spikeglx-meta) amplitudes stay in ADC units
        #[arg(long, value_name = "UV")]
        gain_to_uv: Option<f32>,
        /// Skip the merge presets
        #[arg(long)]
        no_merge: bool,
        /// Skip the UnitRefine models (offline): Bombcell labels only
        #[arg(long)]
        no_unitrefine: bool,
        /// Import the existing <DATA_DIR>/sorrel_auto/curation.json instead
        /// of rerunning SpikeInterface
        #[arg(long)]
        reuse: bool,
        /// Print the changes without journaling them
        #[arg(long)]
        dry_run: bool,
        /// Also write the phy artifacts (same as Ctrl-S in the GUI)
        #[arg(long)]
        save: bool,
    },
    /// Print a dataset summary (phy `template-describe`)
    Describe {
        #[command(flatten)]
        open: OpenArgs,
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
        /// Only clusters with a tag starting with PREFIX (`review:` = the
        /// review queue `sorrel auto` leaves)
        #[arg(long, value_name = "PREFIX")]
        tag: Option<String>,
    },
    /// MCP over stdio, the command an agent launches: attaches to the host
    /// running on DATA_DIR (the GUI or `--headless`), else opens the dataset
    /// in-process. Agents inspect metrics, request plots and curate through
    /// the journal.
    #[command(alias = "serve")]
    Mcp {
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
    /// Write the curated data as a complete phy directory into --out: a
    /// phy source is copied with its curation applied; other backends get
    /// templates computed from the raw trace
    #[cfg(feature = "phy-export")]
    ExportPhy {
        #[command(flatten)]
        open: OpenArgs,
        /// Output directory (created if missing; must not be the source)
        #[arg(long, value_name = "DIR")]
        out: PathBuf,
        /// Copy the raw recording into DIR/recording.dat instead of pointing
        /// params.py at the original
        #[arg(long)]
        copy_raw: bool,
    },
    /// Remove <DATA_DIR>/.sorrel/cache/
    ClearCache {
        /// Dataset directory (or an .nwb / rez.mat file inside it)
        #[arg(value_name = "DATA_DIR")]
        root: PathBuf,
    },
    /// List GPU adapters and check the compute kernels against the CPU on each
    ///
    /// Exits non-zero when a kernel disagrees with the CPU on any adapter, or
    /// when no adapter can build the kernels at all. An adapter whose driver
    /// rejects them ("kernels unavailable") is reported and skipped: the GUI
    /// then preprocesses traces on the CPU.
    GpuCheck,
    /// Headless: matched-filter rescue of spikes a sorter missed from known
    /// units (FDR-controlled, dry run unless --apply), plus a virtual
    /// hybrid-injection recall test (--hybrid-frac)
    Rescue {
        #[command(flatten)]
        open: OpenArgs,
        #[command(flatten)]
        args: rescue::RescueArgs,
    },
}

#[derive(Subcommand, Debug)]
// Parsed once at startup; boxing OpenArgs would buy nothing.
#[allow(clippy::large_enum_variant)]
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
    /// Sorter name shown as Auto[NAME]; overrides detection
    #[arg(long, value_name = "NAME")]
    sorter: Option<String>,
    /// Curation journal (default: <DATA_DIR>/sorrel.journal)
    #[arg(long, value_name = "PATH")]
    journal: Option<PathBuf>,
    /// Set the existing journal aside before opening (renamed to
    /// `<journal>.<unix seconds>`, not deleted; a phy root's sorrel_original/
    /// is set aside too). The escape hatch for a journal sealed against a
    /// spike_clusters.npy that has since changed.
    #[arg(long)]
    discard_journal: bool,
    /// Populate sample-rate / channels / dtype from a SpikeGLX .meta
    #[arg(long, value_name = "PATH")]
    spikeglx_meta: Option<PathBuf>,
    /// Populate sample-rate / channels / dat path from an Open Ephys structure.oebin
    #[arg(long, value_name = "PATH")]
    oebin: Option<PathBuf>,
    /// Events for the PSTH pane: events.csv (time,label) or events.npy
    /// (seconds), or an .nwb file for its trials table. Default:
    /// <DATA_DIR>/events.csv|npy, else the opened NWB file's trials
    #[arg(long, value_name = "FILE")]
    events: Option<PathBuf>,
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

fn journal_opts(open: &OpenArgs) -> sorrel_data::JournalOpts {
    sorrel_data::JournalOpts {
        path: open.journal.clone(),
        discard: open.discard_journal,
    }
}

/// `--journal`, else `<dir>/<stem>.journal` (see [`sorrel_data::journal_path`]).
fn journal_path(open: &OpenArgs, dir: &Path, stem: &str) -> PathBuf {
    sorrel_data::journal_path(&journal_opts(open), dir, stem)
}

/// Open the journal sealed against `baseline`, honouring `--discard-journal`.
fn open_journal(open: &OpenArgs, path: &Path, baseline: u64, n_clusters: u32) -> Result<Journal> {
    sorrel_data::open_journal(&journal_opts(open), path, baseline, n_clusters)
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
    /// Per-cluster quality metrics (KS `cluster_*.tsv`, SI `quality_metrics.csv`);
    /// empty when the backend has none.
    metrics: sorrel_io::QualityMetrics,
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
            // journal stays the source of truth and is never sealed.
            let baseline = baseline_hash(&provider.identity_bytes());
            let journal_path = journal_path(open, &root, "sorrel");
            let journal = open_journal(open, &journal_path, baseline, provider.n_clusters())?;
            let session = new_session(provider, journal, open, sorrel_io::Backend::SortingAnalyzer);
            let metrics = session.provider.quality_metrics().clone();
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
                    metrics,
                    depth_source: None,
                },
            )
        }
        #[cfg(feature = "hdf5")]
        Backend::Nwb => {
            let path = open.root().to_path_buf();
            let provider = sorrel_io::NwbProvider::open(&path)?;
            let (session, ctx) =
                open_file_backed(open, provider, &path, "nwb", sorrel_io::Backend::Nwb)?;
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
            let (session, ctx) =
                open_file_backed(open, provider, &path, "rez", sorrel_io::Backend::Ks4Rez)?;
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

/// Wrap `Session::new` and name the sorter: `--sorter`, else the saved pin, else detected from the data.
fn new_session<P: DataProvider>(
    provider: P,
    journal: Journal,
    open: &OpenArgs,
    backend: sorrel_io::Backend,
) -> Session<P> {
    let mut session = Session::new(provider, journal);
    match &open.sorter {
        Some(flag) => session.set_sorter(flag.clone()),
        None => session.fill_detected_sorter(detect_sorter(open.root(), backend)),
    }
    session
}

/// Open a phy directory with explicit `params` (see [`sorrel_data::open_phy`]:
/// sealed against the root, refusing external edits). The session comes
/// back **unseeded**: run [`seed_kilosort`] (or hand it to the GUI loader)
/// before using it.
fn open_kilosort(
    open: &OpenArgs,
    params: KilosortOpenParams,
) -> Result<(Session<KilosortProvider>, OpenCtx<KilosortProvider>)> {
    let root = open.root().to_path_buf();
    let mut session = sorrel_data::open_phy(&root, params, &journal_opts(open))?;
    if let Some(flag) = &open.sorter {
        session.set_sorter(flag.clone());
    }
    let metrics = session.provider.quality_metrics().clone();
    let positions = session.provider.channel_positions().to_vec();
    let shanks = session.provider.channel_shanks().to_vec();
    Ok((
        session,
        OpenCtx {
            save_dir: root,
            reseal_on_save: true,
            positions,
            shanks,
            metrics,
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
        Box::new(sorrel_remote::SessionServed(std::sync::Arc::new(
            sorrel_data::SessionHost::new(session),
        ))),
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
    // The host tracks our journal cursor, so the guest never seals: a seal
    // record would advance only our count and every later record would be
    // refused as diverged. Saving still writes the phy files locally.
    let ctx = OpenCtx {
        save_dir: joined.root,
        reseal_on_save: false,
        positions: joined.positions,
        shanks: Vec::new(),
        metrics: Default::default(),
        depth_source: Some(sorrel_data::spike_depths),
    };
    run_app(joined.session, Box::new(|_| Ok(())), ctx, None)
}

/// Shared open path for single-file backends (`.nwb`, `rez.mat`): the
/// journal and save artifacts live next to the file, named after it.
#[cfg(feature = "hdf5")]
fn open_file_backed<P>(
    open: &OpenArgs,
    provider: P,
    path: &Path,
    fallback_stem: &str,
    backend: sorrel_io::Backend,
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
    let session = new_session(provider, journal, open, backend);
    let positions = session.provider.channel_positions().to_vec();
    let shanks = session.provider.channel_shanks().to_vec();
    Ok((
        session,
        OpenCtx {
            save_dir: parent,
            reseal_on_save: false,
            positions,
            shanks,
            metrics: Default::default(),
            depth_source: None,
        },
    ))
}

// ---------------------------------------------------------------- commands

struct Gui(Option<Events>);
impl SessionVisitor for Gui {
    fn visit<P>(self, session: Session<P>, seed: Seed<P>, ctx: OpenCtx<P>) -> Result<()>
    where
        P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
    {
        run_app(session, seed, ctx, self.0)
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
        println!("splits (score, cluster, method, bimodality, dip, contamination, gmm_bic_delta)");
        for s in &splits {
            println!(
                "  {:.3}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.1}",
                s.score,
                s.cluster.0,
                s.method,
                s.amp_bimodality,
                s.amp_dip,
                s.contamination,
                s.gmm_bic_delta
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

#[cfg(feature = "phy-export")]
struct ExportPhy {
    out: PathBuf,
    copy_raw: bool,
}
#[cfg(feature = "phy-export")]
impl SessionVisitor for ExportPhy {
    fn visit<P>(self, session: Session<P>, seed: Seed<P>, _: OpenCtx<P>) -> Result<()>
    where
        P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
    {
        let session = seeded(session, seed)?;
        sorrel_data::export_phy(&session, &self.out, self.copy_raw, phy_label_str)
            .with_context(|| format!("exporting phy to {}", self.out.display()))?;
        println!(
            "exported {} clusters as phy into {}",
            session.n_clusters(),
            self.out.display()
        );
        Ok(())
    }
}

enum ApplySource {
    Rules(PathBuf),
    /// A curation.json import; `tag_splits` adds `review:split` tags from
    /// sorrel's own split ranking (the auto pipeline has no split detector).
    Curation {
        path: PathBuf,
        tag_splits: bool,
    },
}

struct Apply {
    source: ApplySource,
    dry_run: bool,
    save: bool,
}
impl SessionVisitor for Apply {
    fn visit<P>(self, session: Session<P>, seed: Seed<P>, ctx: OpenCtx<P>) -> Result<()>
    where
        P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
    {
        use sorrel_data::CurationCommand;
        let mut session = seeded(session, seed)?;
        let current = |c: sorrel_io::ClusterId| session.label(c).unwrap_or_default().as_str();
        let cmds = match &self.source {
            ApplySource::Rules(path) => {
                let rules = RuleSet::from_path(path)?;
                apply_rules(&rules, &collect_qc_rows(&session), current)?
            }
            ApplySource::Curation { path, tag_splits } => {
                let doc = sorrel_io::read_curation(path)?;
                let mut cmds =
                    sorrel_data::curation_json::import_commands(&session, &doc, current)?;
                if *tag_splits {
                    tag_split_candidates(&session, &mut cmds);
                }
                cmds
            }
        };
        let (mut retags, mut merges, mut review) = (0, 0, 0);
        for cmd in &cmds {
            match cmd {
                CurationCommand::Relabel { cluster, op } => println!(
                    "{}\t{} -> {}",
                    cluster.0,
                    current(*cluster),
                    P::label_from_op(*op).as_str()
                ),
                CurationCommand::Merge { sources, target } => {
                    merges += 1;
                    let ids: Vec<String> = sources.iter().map(|s| s.0.to_string()).collect();
                    println!("{}\t<- merge {}", target.0, ids.join(","));
                }
                CurationCommand::Tag { tags, .. } => {
                    retags += 1;
                    review += usize::from(tags.iter().any(|t| t.starts_with("review:")));
                }
                _ => {}
            }
        }
        let relabels = cmds.len() - retags - merges;
        println!(
            "{relabels} relabels, {merges} merges, {retags} retags ({review} need review){}",
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
            save_session(&mut session, &ctx)?;
        }
        Ok(())
    }
}

/// Write the phy artifacts like Ctrl-S in the GUI (sealing the journal on
/// backends that read the saved clusters back).
fn save_session<P>(session: &mut Session<P>, ctx: &OpenCtx<P>) -> Result<()>
where
    P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
{
    save_dir_session(session, &ctx.save_dir, ctx.reseal_on_save, phy_label_str)?;
    println!("saved → {}", ctx.save_dir.display());
    Ok(())
}

/// Add `review:split` + `src:sorrel` to every cluster for which sorrel's
/// split ranking has a concrete (one-click) bipartition, on top of the tags
/// `cmds` already gives it. Merge-group members are skipped: the ranking saw
/// them before the merge. Retags that would not change anything are dropped,
/// so re-importing the same document is a no-op.
fn tag_split_candidates<P: DataProvider>(
    session: &Session<P>,
    cmds: &mut Vec<sorrel_data::CurationCommand>,
) {
    use sorrel_data::CurationCommand;
    let merged: Vec<sorrel_io::ClusterId> = cmds
        .iter()
        .filter_map(|c| match c {
            CurationCommand::Merge { sources, target } => Some(sources.iter().chain([target])),
            _ => None,
        })
        .flatten()
        .copied()
        .collect();
    for cand in rank_split_candidates(session, &SuggestConfig::default()) {
        if cand.bipartition_minor_idx.is_empty() || merged.contains(&cand.cluster) {
            continue;
        }
        let pending = cmds.iter_mut().find_map(|c| match c {
            CurationCommand::Tag { cluster, tags } if *cluster == cand.cluster => Some(tags),
            _ => None,
        });
        let tags = match pending {
            Some(tags) => tags,
            None => {
                cmds.push(CurationCommand::Tag {
                    cluster: cand.cluster,
                    tags: session.tags(cand.cluster).to_vec(),
                });
                match cmds.last_mut() {
                    Some(CurationCommand::Tag { tags, .. }) => tags,
                    _ => unreachable!("just pushed a Tag"),
                }
            }
        };
        for t in ["review:split", "src:sorrel"] {
            if !tags.iter().any(|x| x == t) {
                tags.push(t.into());
            }
        }
    }
    cmds.retain(|c| {
        !matches!(c, CurationCommand::Tag { cluster, tags } if session.tags(*cluster) == tags.as_slice())
    });
}

/// The SpikeInterface pipeline, embedded so an installed binary carries it.
const AUTO_SCRIPT: &str = include_str!("sorrel_auto.py");

/// Whether `uv` (`uv.exe` on Windows) is in a directory of the PATH value.
fn uv_on_path(path: &std::ffi::OsStr) -> bool {
    let uv = format!("uv{}", std::env::consts::EXE_SUFFIX);
    std::env::split_paths(path).any(|d| d.join(&uv).is_file())
}

/// Run the embedded `sorrel_auto.py` into `out`: with `uv run --script` when uv is
/// on PATH (its PEP 723 header pins the dependencies), else with
/// `$SORREL_PYTHON` / `python` (Windows) or `python3`, which must already have
/// them installed.
fn run_auto_sidecar(open: &OpenArgs, out: &Path, flags: &[String]) -> Result<()> {
    if !matches!(open.backend, None | Some(Backend::Kilosort)) || !open.root().is_dir() {
        bail!("sorrel auto supports Kilosort/phy directories");
    }
    if open.oebin.is_some() {
        bail!("sorrel auto: pass --dat/--sample-rate/--channels instead of --oebin");
    }
    std::fs::create_dir_all(out).with_context(|| format!("create {}", out.display()))?;
    let script = out.join("sorrel_auto.py");
    std::fs::write(&script, AUTO_SCRIPT).with_context(|| format!("write {}", script.display()))?;

    let mut args: Vec<std::ffi::OsString> = vec![
        script.into(),
        "--data-dir".into(),
        open.root().into(),
        "--out".into(),
        out.into(),
    ];
    let mut opt = |flag: &str, v: Option<String>| {
        if let Some(v) = v {
            args.extend([flag.into(), v.into()]);
        }
    };
    opt("--dat", open.dat.as_ref().map(|p| p.display().to_string()));
    opt("--sample-rate", open.sample_rate.map(|v| v.to_string()));
    opt("--channels", open.n_channels.map(|v| v.to_string()));
    opt("--dtype", open.dtype.map(|d| d.phy_name().to_string()));
    opt("--offset", open.offset.map(|v| v.to_string()));
    opt(
        "--spikeglx-meta",
        open.spikeglx_meta.as_ref().map(|p| p.display().to_string()),
    );
    args.extend(flags.iter().map(Into::into));

    let uv = std::env::var_os("PATH").is_some_and(|path| uv_on_path(&path));
    let mut cmd = if uv && std::env::var_os("SORREL_PYTHON").is_none() {
        let mut c = std::process::Command::new("uv");
        c.args(["run", "--script"]);
        c
    } else {
        let python = if cfg!(windows) { "python" } else { "python3" };
        std::process::Command::new(
            std::env::var_os("SORREL_PYTHON").unwrap_or_else(|| python.into()),
        )
    };
    log::info!("running the SpikeInterface pipeline (first run computes the analyzer; slow)");
    let status = cmd
        .args(&args)
        .status()
        .context("spawn the sorrel_auto.py sidecar (install uv, or set SORREL_PYTHON)")?;
    if !status.success() {
        bail!("sorrel_auto.py failed ({status})");
    }
    Ok(())
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

struct Describe;
impl SessionVisitor for Describe {
    fn visit<P>(self, session: Session<P>, seed: Seed<P>, _: OpenCtx<P>) -> Result<()>
    where
        P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
    {
        let s = seeded(session, seed)?;
        print!("{}", describe(&s));
        Ok(())
    }
}

/// phy `template-describe`: one `name  value` line per dataset fact.
fn describe<P: DataProvider<Label = PhyLabel>>(s: &Session<P>) -> String {
    let p = &s.provider;
    let sr = p.sample_rate();
    let n_spikes: usize = (0..s.n_clusters())
        .map(|c| s.spike_times(sorrel_io::ClusterId(c)).len())
        .sum();
    let alive = (0..s.n_clusters())
        .filter(|&c| !s.spike_times(sorrel_io::ClusterId(c)).is_empty())
        .count();
    let (n_pcs, n_pc_ch) = s.pc_shape();
    let (n_templates, n_t, _) = s.template_shape();
    let mut groups = std::collections::BTreeMap::<&str, usize>::new();
    for c in (0..s.n_clusters()).filter(|&c| !s.spike_times(sorrel_io::ClusterId(c)).is_empty()) {
        *groups
            .entry(
                s.label(sorrel_io::ClusterId(c))
                    .map_or("unsorted", |l| l.as_str()),
            )
            .or_default() += 1;
    }
    let rows = [
        (
            "data files",
            p.raw_file()
                .map_or("—".into(), |r| r.path.display().to_string()),
        ),
        (
            "dtype",
            p.raw_file()
                .map_or("—".into(), |r| format!("{:?}", r.dtype)),
        ),
        ("sample rate", format!("{:.1} kHz", sr / 1000.0)),
        (
            "duration",
            format!("{:.1} s", p.n_samples().0 as f64 / sr.max(1.0) as f64),
        ),
        ("# channels", p.n_channels().to_string()),
        ("# spikes", n_spikes.to_string()),
        (
            "# clusters",
            format!("{alive} (ids up to {})", s.n_clusters()),
        ),
        (
            "groups",
            groups
                .iter()
                .map(|(g, n)| format!("{g} {n}"))
                .collect::<Vec<_>>()
                .join(", "),
        ),
        (
            "# templates",
            if n_templates > 0 {
                format!("{n_templates} × {n_t} samples")
            } else {
                "—".into()
            },
        ),
        (
            "pc features",
            if n_pcs > 0 {
                format!("{n_pcs} PCs × {n_pc_ch} channels")
            } else {
                "— (FeatureView fits PCs on raw snippets)".into()
            },
        ),
        ("labels", {
            let k = s.custom_label_keys();
            if k.is_empty() {
                "—".into()
            } else {
                k.join(", ")
            }
        }),
        ("undo history", s.history_len().to_string()),
    ];
    rows.iter().map(|(k, v)| format!("{k:<14}{v}\n")).collect()
}

/// Where a host for `open` writes `.sorrel/mcp.json`: the save directory,
/// i.e. the dataset directory or a single file's parent.
fn host_dir(open: &OpenArgs) -> PathBuf {
    let root = open.root();
    match root.is_dir() {
        true => root.to_path_buf(),
        false => root.parent().unwrap_or(Path::new(".")).to_path_buf(),
    }
}

/// `sorrel mcp`: bridge stdio to a running host, else serve in-process.
fn run_mcp(open: &OpenArgs) -> Result<()> {
    if sorrel_data::attach(&host_dir(open), std::io::stdin(), std::io::stdout())? {
        return Ok(());
    }
    // attach found no listener. If another process still has the journal
    // open, serving here would be a second writer: refuse and say so.
    with_session(open, Serve).map_err(busy_hint)
}

/// Name the process holding the journal when `e` is a [`JournalBusy`].
fn busy_hint(e: anyhow::Error) -> anyhow::Error {
    match e.downcast_ref::<sorrel_data::journal::JournalBusy>() {
        Some(b) => anyhow::anyhow!(
            "{b}: another sorrel owns this dataset but its MCP listener is not \
             reachable (no usable MCP token file). Close it, or run `sorrel mcp` \
             once its host (the GUI or `sorrel --headless`) is up"
        ),
        None => e,
    }
}

/// `sorrel --headless`: the host and its MCP listener, no window.
struct Headless;
impl SessionVisitor for Headless {
    fn visit<P>(self, session: Session<P>, seed: Seed<P>, ctx: OpenCtx<P>) -> Result<()>
    where
        P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
    {
        let host = std::sync::Arc::new(sorrel_data::SessionHost::new(seeded(session, seed)?));
        let listener = sorrel_data::spawn_listener(
            host,
            &ctx.save_dir,
            ctx.reseal_on_save,
            phy_label_str,
            None,
        )?;
        println!(
            "sorrel host on 127.0.0.1:{}; connect an agent with `sorrel mcp {}`; Ctrl-C stops",
            listener.port(),
            ctx.save_dir.display()
        );
        // Ctrl-C returns, dropping the listener so it removes its mcp.json.
        sorrel_remote::wait_for_ctrl_c();
        drop(listener);
        Ok(())
    }
}

struct Serve;
impl SessionVisitor for Serve {
    fn visit<P>(self, session: Session<P>, seed: Seed<P>, ctx: OpenCtx<P>) -> Result<()>
    where
        P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
    {
        let host = sorrel_data::SessionHost::new(seeded(session, seed)?);
        let stdin = std::io::stdin();
        let stdout = std::io::stdout();
        serve(
            &host,
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
    let records = Journal::read_only_records(path)?;
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

fn gpu_check() -> Result<()> {
    let (report, ok) = sorrel_gpu::self_check();
    print!("{report}");
    anyhow::ensure!(ok, "GPU check failed");
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

fn run_app<P>(
    session: Session<P>,
    seed: Seed<P>,
    ctx: OpenCtx<P>,
    events: Option<Events>,
) -> Result<()>
where
    P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
{
    // Seeding and replay run on a worker behind the loading screen.
    let mut app = SorrelApp::with_loader(session, phy_label_str, seed);
    #[cfg(feature = "phy-export")]
    {
        // A sibling of the dataset, so `share` (which syncs everything under
        // it) never ships the export to guests.
        let dir = ctx
            .save_dir
            .canonicalize()
            .unwrap_or_else(|_| ctx.save_dir.clone());
        let name = dir
            .file_name()
            .map_or("sorrel".into(), |n| n.to_string_lossy());
        let out = dir.with_file_name(format!("{name}_phy"));
        app.set_phy_export(std::sync::Arc::new(move |s: &Session<P>| {
            sorrel_data::export_phy(s, &out, false, phy_label_str)?;
            Ok(out.clone())
        }));
    }
    app.set_save_dir(ctx.save_dir, ctx.reseal_on_save);
    if let Some(events) = events {
        app.set_events(events);
    }
    if !ctx.positions.is_empty() {
        app.set_channel_positions(ctx.positions);
    }
    if !ctx.shanks.is_empty() {
        app.set_channel_shanks(ctx.shanks);
    }
    if !ctx.metrics.is_empty() {
        app.set_cluster_metrics(ctx.metrics);
    }
    if let Some(f) = ctx.depth_source {
        app.set_depth_source(f);
    }

    let mut wgpu_options = eframe::egui_wgpu::WgpuConfiguration::default();
    if let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut wgpu_options.wgpu_setup {
        setup.instance_descriptor.flags = sorrel_gpu::instance_flags();
    }
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 800.0]),
        wgpu_options,
        ..Default::default()
    };
    eframe::run_native(
        "Sorrel",
        native_options,
        Box::new(move |cc| {
            let mut app = app;
            if let Err(e) = app.serve_mcp() {
                log::warn!("MCP listener not started: {e:#}");
            }
            if let Some(rs) = cc.wgpu_render_state.as_ref() {
                TracePipeline::install(rs);
                RasterPipeline::install(rs);
                ScatterPipeline::install(rs);
                app.install_gpu_compute(
                    rs.device.clone().into(),
                    rs.queue.clone().into(),
                    &rs.adapter.get_info(),
                );
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
        None if cli.headless => with_session(&cli.open, Headless),
        None => run_open(cli.open),
        Some(Cmd::Open(open)) => run_open(open),
        Some(Cmd::Qc { open, out }) => with_session(&open, Qc(out)),
        Some(Cmd::Suggest { open, json }) => with_session(&open, Suggest { json }),
        Some(Cmd::Apply {
            open,
            rules,
            curation,
            dry_run,
            save,
        }) => {
            let source = match (rules, curation) {
                (Some(rules), _) => ApplySource::Rules(rules),
                (None, Some(path)) => ApplySource::Curation {
                    path,
                    tag_splits: false,
                },
                (None, None) => unreachable!("clap requires --rules or --curation"),
            };
            with_session(
                &open,
                Apply {
                    source,
                    dry_run,
                    save,
                },
            )
        }
        Some(Cmd::Auto {
            open,
            llm,
            llm_model,
            gain_to_uv,
            no_merge,
            no_unitrefine,
            reuse,
            dry_run,
            save,
        }) => {
            let out = open.root().join("sorrel_auto");
            if !reuse {
                let mut flags = vec![];
                if llm {
                    flags.extend(["--llm".to_string(), "--llm-model".into(), llm_model]);
                }
                if let Some(g) = gain_to_uv {
                    flags.extend(["--gain-to-uv".to_string(), g.to_string()]);
                }
                if no_merge {
                    flags.push("--no-merge".into());
                }
                if no_unitrefine {
                    flags.push("--no-unitrefine".into());
                }
                run_auto_sidecar(&open, &out, &flags)?;
            }
            with_session(
                &open,
                Apply {
                    source: ApplySource::Curation {
                        path: out.join("curation.json"),
                        tag_splits: true,
                    },
                    dry_run,
                    save,
                },
            )
        }
        Some(Cmd::Diff { a, b, json }) => diff_tsvs(&a, &b, json),
        Some(Cmd::Mcp { open }) => run_mcp(&open),
        Some(Cmd::Describe { open }) => with_session(&open, Describe),
        Some(Cmd::Report {
            open,
            out,
            clusters,
            label,
            tag,
        }) => with_session(
            &open,
            Report {
                out,
                filter: ReportFilter {
                    clusters,
                    label,
                    tag,
                },
            },
        ),
        Some(Cmd::Journal {
            action: JournalCmd::Dump { path, json },
        }) => journal_dump(&path, json),
        Some(Cmd::Journal {
            action: JournalCmd::Replay { open, out },
        }) => with_session(&open, Replay(out)),
        #[cfg(feature = "phy-export")]
        Some(Cmd::ExportPhy {
            open,
            out,
            copy_raw,
        }) => with_session(&open, ExportPhy { out, copy_raw }),
        Some(Cmd::Share { open, port, relay }) => run_share(&open, port, relay),
        Some(Cmd::Join {
            code,
            direct,
            relay,
            into,
        }) => run_join(&code, direct, relay, into),
        Some(Cmd::ClearCache { root }) => clear_cache(&root),
        Some(Cmd::GpuCheck) => gpu_check(),
        Some(Cmd::Rescue { open, args }) => with_session(&open, rescue::Rescue(args)),
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
    let events = match &open.events {
        Some(p) => {
            Some(Events::read(p).with_context(|| format!("reading events {}", p.display()))?)
        }
        // Sidecar events next to an NWB file win (the app detects them);
        // otherwise its trials table, whose absence is normal.
        None if open
            .root()
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("nwb"))
            && matches!(
                Events::detect(open.root().parent().unwrap_or(Path::new("."))),
                Ok(None)
            ) =>
        {
            Events::read(open.root()).ok()
        }
        None => None,
    };
    with_session(&open, Gui(events))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `sorrel mcp` on a dataset another process has open names the owner
    /// instead of becoming a second writer.
    #[test]
    fn busy_journal_is_reported_with_its_owner() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sorrel.journal");
        let _owner = Journal::open_or_create(&path, 0, 1).unwrap();
        let busy = Journal::open_or_create(&path, 0, 1).unwrap_err();
        let msg = busy_hint(busy.context("open journal")).to_string();
        assert!(
            msg.contains("already open") && msg.contains("MCP token file"),
            "{msg}"
        );
        let other = busy_hint(anyhow::anyhow!("disk full"));
        assert_eq!(other.to_string(), "disk full");
    }

    #[test]
    fn uv_is_found_by_its_platform_file_name() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path()
                .join(format!("uv{}", std::env::consts::EXE_SUFFIX)),
            b"",
        )
        .unwrap();
        assert!(uv_on_path(dir.path().as_os_str()));
        assert!(!uv_on_path(std::ffi::OsStr::new("")));
    }

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

        #[cfg(feature = "phy-export")]
        {
            let cli =
                Cli::try_parse_from(["sorrel", "export-phy", "/x", "--out", "o", "--copy-raw"])
                    .unwrap();
            assert!(matches!(
                cli.cmd,
                Some(Cmd::ExportPhy { copy_raw: true, .. })
            ));
        }

        for name in ["mcp", "serve"] {
            let cli = Cli::try_parse_from(["sorrel", name, "/x"]).unwrap();
            assert!(matches!(cli.cmd, Some(Cmd::Mcp { .. })), "{name}");
        }
        let cli = Cli::try_parse_from(["sorrel", "--headless", "/x"]).unwrap();
        assert!(cli.headless && cli.cmd.is_none());
        assert_eq!(
            host_dir(&cli.open),
            PathBuf::from("/"),
            "a missing path is a file"
        );

        let cli = Cli::try_parse_from(["sorrel", "/x", "--events", "e.csv"]).unwrap();
        assert_eq!(cli.open.events, Some(PathBuf::from("e.csv")));

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
