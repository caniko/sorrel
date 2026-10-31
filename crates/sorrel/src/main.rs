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
use eframe::egui_wgpu::RenderState;
use sorrel_data::session::ApplyPhyLabel;
use sorrel_data::InstanceRequest;
use sorrel_data::{
    apply_rules, baseline_hash, collect_qc_rows_with, diff_labels, export_qc,
    rank_merge_candidates, rank_split_candidates, save_session as save_dir_session, save_to_phy,
    serve, write_report, Journal, ReportFilter, RuleSet, Session, SuggestConfig,
};
use sorrel_io::kilosort::{read_cluster_group_tsv, KilosortOpenParams, PhyLabel};
use sorrel_io::open_ephys::OebinMeta;
use sorrel_io::spikeglx::SpikeGlxMeta;
use sorrel_io::{
    detect_sorter, DataProvider, Events, HasGeometry, HasQualityMetrics, KilosortProvider,
    SortingAnalyzerProvider, TraceDtype,
};
use sorrel_ui::{OpenRequest, Opened, Opener, Shell, SorrelApp, Tab};
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
/// Bare `sorrel` starts the GUI (or focuses the running one); `open` adds a
/// dataset to it as a tab. The other subcommands run headlessly on the same
/// data and the same curation journal.
#[derive(Parser, Debug)]
#[command(
    name = "sorrel",
    version,
    after_help = "When params.py is present, sample-rate / channels / dtype / offset / \
                  dat-path all default from it; flags override per field."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Open DATA_DIR as a tab in the running GUI, starting one (detached) if
    /// none is running
    Open(OpenArgs),
    /// The GUI in the foreground (logs, Ctrl-C). Hands DATA_DIR to a running
    /// GUI unless --standalone
    #[command(mut_arg("root", |a| a.required(false)))]
    Run {
        #[command(flatten)]
        open: OpenArgs,
        /// Always start a separate window: neither forward to a running GUI
        /// nor register as the instance `open` talks to
        #[arg(long)]
        standalone: bool,
        /// Run the session host without a window: MCP clients connect with
        /// `sorrel mcp <DATA_DIR>`. Stops on Ctrl-C
        #[arg(long, requires = "root", conflicts_with = "standalone")]
        headless: bool,
    },
    /// Show the running GUI (tabs), session hosts and the detached GUI's log
    Status {
        /// Only the hosts of this dataset
        #[arg(value_name = "DATA_DIR")]
        dir: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Run one MCP tool: through the running host, else on the dataset
    /// opened in-process
    Call {
        #[command(flatten)]
        open: OpenArgs,
        /// Tool name (see `sorrel mcp tools`)
        tool: String,
        /// Tool arguments as a JSON object
        args: Option<String>,
        /// Print the whole JSON-RPC response
        #[arg(long)]
        raw: bool,
    },
    /// Print a primer for LLM agents, with the live command list, MCP tools
    /// and running hosts
    AgentGuide,
    /// Print a shell completion script
    Completions { shell: clap_complete::Shell },
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
    /// presets, Bombcell + UnitRefine labels, optional LLM tiebreak),
    /// applied as one undoable batch; every decision is tagged with its
    /// source, and anything uncertain gets a `review:*` tag
    Auto {
        #[command(flatten)]
        open: OpenArgs,
        /// Ask an LLM to settle units the classifiers disagree on and merges
        /// only one preset found. Bare `--llm` uses the logged-in `claude`
        /// CLI; name another backend with `--llm=codex`. (`=` is required so
        /// `auto --llm DIR` keeps DIR a positional.)
        #[arg(
            long,
            value_enum,
            num_args = 0..=1,
            require_equals = true,
            default_missing_value = "claude",
            value_name = "BACKEND",
            conflicts_with = "reuse"
        )]
        llm: Option<LlmBackend>,
        /// Model for --llm (default: the backend's own, `claude-opus-5` for api)
        #[arg(long, value_name = "MODEL", requires = "llm")]
        llm_model: Option<String>,
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
        /// Emit a JSON object instead of the table
        #[arg(long)]
        json: bool,
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
    /// running on DATA_DIR (the GUI or `run --headless`), else opens the
    /// dataset in-process. Without DATA_DIR: the GUI's active tab, else the
    /// only running host. Agents inspect metrics, request plots and curate
    /// through the journal.
    #[command(
        args_conflicts_with_subcommands = true,
        mut_arg("root", |a| a.required(false))
    )]
    Mcp {
        #[command(flatten)]
        open: OpenArgs,
        #[command(subcommand)]
        sub: Option<McpCmd>,
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
enum McpCmd {
    /// List the MCP tools (offline)
    Tools {
        /// Print the full `tools/list` JSON
        #[arg(long)]
        json: bool,
    },
    /// Print how to register sorrel with an agent client
    Config {
        #[arg(long, value_enum, default_value = "claude")]
        client: Client,
        /// Pin the server to this dataset (default: the active one)
        #[arg(value_name = "DATA_DIR")]
        dir: Option<PathBuf>,
    },
}

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum Client {
    Claude,
    Codex,
    Json,
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

/// Where `sorrel auto --llm` sends its cases.
#[derive(Clone, Copy, Debug, PartialEq, clap::ValueEnum)]
enum LlmBackend {
    /// The logged-in `claude` CLI (Claude Code)
    Claude,
    /// The logged-in `codex` CLI
    Codex,
    /// The `opencode` CLI
    Opencode,
    /// The Anthropic API (ANTHROPIC_API_KEY or `ant auth login`)
    Api,
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
}

impl OpenArgs {
    fn root(&self) -> &Path {
        self.root.as_deref().expect("clap marks DATA_DIR required")
    }

    /// Make the path fields absolute against `cwd`: the args came from a
    /// `sorrel open` run somewhere else.
    fn rebase(&mut self, cwd: &Path) {
        for p in [
            &mut self.root,
            &mut self.dat,
            &mut self.journal,
            &mut self.spikeglx_meta,
            &mut self.oebin,
            &mut self.events,
        ]
        .into_iter()
        .flatten()
        {
            *p = cwd.join(&*p);
        }
    }
}

/// A forwarded `sorrel open`'s arguments, parsed again inside the GUI.
#[derive(Parser)]
struct TabCli {
    #[command(flatten)]
    open: OpenArgs,
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
    let root = std::path::absolute(&joined.root)?;
    let ctx = OpenCtx {
        save_dir: joined.root.clone(),
        reseal_on_save: false,
        positions: joined.positions.clone(),
        shanks: Vec::new(),
        metrics: Default::default(),
        depth_source: Some(sorrel_data::spike_depths),
    };
    // The network state lives in this process, so the window is standalone
    // and its one tab is the session just joined (`joined` stays alive until
    // the window closes).
    let session = joined.session;
    let mut once = Some(move |rs: Option<&RenderState>| {
        let app = build_tab(session, Box::new(|_| Ok(())), ctx, None, rs);
        Opened::Tab(Box::new(app) as Box<dyn Tab>, root)
    });
    let opener: Opener = Box::new(move |_, _, rs| {
        once.take()
            .map(|f| f(rs))
            .ok_or_else(|| anyhow::anyhow!("a joined session is a single window"))
    });
    run_shell(
        opener,
        Some(OpenRequest {
            cwd: PathBuf::new(),
            args: vec![],
        }),
        false,
    )
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

/// Builds the GUI tab for a session (the typed `SorrelApp<P>` is boxed into
/// a [`Tab`], so the visitor trait stays unchanged).
struct TabBuilder<'a> {
    out: &'a mut Option<Box<dyn Tab>>,
    events: Option<Events>,
    rs: Option<&'a RenderState>,
}
impl SessionVisitor for TabBuilder<'_> {
    fn visit<P>(self, session: Session<P>, seed: Seed<P>, ctx: OpenCtx<P>) -> Result<()>
    where
        P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
    {
        *self.out = Some(Box::new(build_tab(
            session,
            seed,
            ctx,
            self.events,
            self.rs,
        )));
        Ok(())
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
        let n = export_qc(
            &session,
            out_dir,
            Some(session.provider.channel_positions()),
        )
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
                apply_rules(
                    &rules,
                    &collect_qc_rows_with(&session, Some(session.provider.channel_positions())),
                    current,
                )?
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

/// `--discard-journal` would set the journal aside again on every `sorrel call`
/// the `--llm` stage makes, so it must happen beforehand.
fn auto_llm_journal_ok(llm: bool, discard_journal: bool) -> Result<()> {
    if llm && discard_journal {
        bail!("sorrel auto --llm: set the journal aside first (`sorrel open --discard-journal`) or drop --llm");
    }
    Ok(())
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

    // Open overrides shared by the sidecar and the `sorrel call` it makes.
    let mut shared: Vec<String> = vec![];
    let mut opt = |flag: &str, v: Option<String>| {
        if let Some(v) = v {
            shared.extend([flag.into(), v]);
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
    // Only `sorrel call` takes this: the journal `sorrel auto` will open.
    // Never --discard-journal: each call would set the journal aside again.
    let mut brief_args = shared.clone();
    if let Some(j) = &open.journal {
        brief_args.extend(["--journal".into(), j.display().to_string()]);
    }

    let mut args: Vec<std::ffi::OsString> = vec![
        script.into(),
        "--data-dir".into(),
        open.root().into(),
        "--out".into(),
        out.into(),
    ];
    args.extend(shared.iter().map(Into::into));
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
        .env("SORREL_BIN", std::env::current_exe()?)
        .env("SORREL_OPEN_ARGS", serde_json::to_string(&brief_args)?)
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

struct Describe {
    json: bool,
}
impl SessionVisitor for Describe {
    fn visit<P>(self, session: Session<P>, seed: Seed<P>, _: OpenCtx<P>) -> Result<()>
    where
        P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
    {
        let s = seeded(session, seed)?;
        let rows = describe(&s);
        if self.json {
            let obj: serde_json::Map<String, serde_json::Value> = rows
                .into_iter()
                .map(|(k, v)| (k.to_owned(), v.into()))
                .collect();
            println!("{}", serde_json::to_string_pretty(&obj)?);
        } else {
            print!(
                "{}",
                rows.iter()
                    .map(|(k, v)| format!("{k:<14}{v}\n"))
                    .collect::<String>()
            );
        }
        Ok(())
    }
}

/// phy `template-describe`: one `(name, value)` row per dataset fact.
fn describe<P: DataProvider<Label = PhyLabel>>(s: &Session<P>) -> Vec<(&'static str, String)> {
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
    rows.to_vec()
}

/// The save directory a host for `open` registers under: the dataset
/// directory or a single file's parent.
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
             once its host (the GUI or `sorrel run --headless`) is up"
        ),
        None => e,
    }
}

/// `sorrel run --headless`: the host and its MCP listener, no window.
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
        // Ctrl-C returns, dropping the listener so it removes its registry file.
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

/// One curation tab for a session: export hook, save dir, optional extras,
/// the MCP listener and (with a render state) the GPU compute path.
fn build_tab<P>(
    session: Session<P>,
    seed: Seed<P>,
    ctx: OpenCtx<P>,
    events: Option<Events>,
    rs: Option<&RenderState>,
) -> SorrelApp<P>
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
    if let Err(e) = app.serve_mcp() {
        log::warn!("MCP listener not started: {e:#}");
    }
    match rs {
        Some(rs) => app.install_gpu_compute(
            rs.device.clone().into(),
            rs.queue.clone().into(),
            &rs.adapter.get_info(),
        ),
        None => log::warn!("no wgpu render state available; trace view will not render"),
    }
    app
}

/// `--events`, else (for an NWB file without sidecar events) its trials table.
fn load_events(open: &OpenArgs) -> Result<Option<Events>> {
    Ok(match &open.events {
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
    })
}

/// The shell's opener: a forwarded `sorrel open`'s args become a tab, or the
/// index of the tab that already shows the dataset.
fn tab_opener() -> Opener {
    Box::new(|req, open_dirs, rs| {
        let args = std::iter::once("sorrel").chain(req.args.iter().map(String::as_str));
        let mut open = TabCli::try_parse_from(args)?.open;
        open.rebase(&req.cwd);
        let canon = |p: &Path| p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
        let dir = canon(&host_dir(&open));
        if let Some(i) = open_dirs.iter().position(|d| canon(d) == dir) {
            return Ok(Opened::Existing(i));
        }
        let events = load_events(&open)?;
        let mut tab = None;
        with_session(
            &open,
            TabBuilder {
                out: &mut tab,
                events,
                rs,
            },
        )?;
        let tab = tab.context("backend produced no tab")?;
        Ok(Opened::Tab(tab, open.root().to_path_buf()))
    })
}

/// The window. `register` makes it the instance `sorrel open` hands datasets
/// to; `first` is opened at startup.
fn run_shell(opener: Opener, first: Option<OpenRequest>, register: bool) -> Result<()> {
    let mut wgpu_options = eframe::egui_wgpu::WgpuConfiguration::default();
    if let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut wgpu_options.wgpu_setup {
        setup.instance_descriptor.flags = sorrel_gpu::instance_flags();
    }
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_app_id("sorrel"),
        wgpu_options,
        ..Default::default()
    };
    // Held here, not in the app, so it drops (removing gui.json) when
    // `run_native` returns.
    let held = std::rc::Rc::new(std::cell::RefCell::new(None));
    let slot = held.clone();
    eframe::run_native(
        "Sorrel",
        native_options,
        Box::new(move |cc| {
            let mut rx = None;
            if register {
                let ctx = cc.egui_ctx.clone();
                match sorrel_data::serve_instance(std::sync::Arc::new(move || {
                    ctx.request_repaint()
                })) {
                    Ok((handle, r)) => {
                        *slot.borrow_mut() = Some(handle);
                        rx = Some(r);
                    }
                    Err(e) => log::warn!("single-instance listener not started: {e}"),
                }
            }
            let mut shell = Shell::new(cc, opener, rx);
            if let Some(req) = first {
                shell.open(req);
            }
            Ok(Box::new(shell))
        }),
    )
    .map_err(|e| anyhow::anyhow!("eframe: {e}"))?;
    drop(held);
    Ok(())
}

/// This invocation's arguments after the subcommand, as a forwarded request.
/// ponytail: assumes no global flags precede the subcommand (there are none);
/// parse them out of `env::args` if one is ever added.
fn open_request() -> Result<OpenRequest> {
    Ok(OpenRequest {
        cwd: std::env::current_dir()?,
        // `run`'s own flag; the tab parser only knows the open options.
        args: std::env::args()
            .skip(2)
            .filter(|a| a != "--standalone")
            .collect(),
    })
}

/// Hand `req` to the running GUI. `false`: none is running.
fn forward(req: &InstanceRequest) -> Result<bool> {
    match sorrel_data::send_instance(req)? {
        None => Ok(false),
        Some(v) => match v["error"].as_str() {
            Some(e) => bail!("{e}"),
            None => Ok(true),
        },
    }
}

/// Start the GUI with `run_args`: detached when a terminal would otherwise
/// hold it, else in-process (a desktop launch, or an AppImage, which unmounts
/// when its parent exits).
fn start_gui(run_args: &[String], first: Option<OpenRequest>) -> Result<()> {
    use std::io::IsTerminal;
    if !std::io::stderr().is_terminal() || std::env::var_os("APPIMAGE").is_some() {
        return run_shell(tab_opener(), first, true);
    }
    let log_path = sorrel_data::user_dir()?.join("gui.log");
    let log = std::fs::File::create(&log_path)
        .with_context(|| format!("create {}", log_path.display()))?;
    let mut cmd = std::process::Command::new(std::env::current_exe()?);
    cmd.args(run_args)
        .stdin(std::process::Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        cmd.creation_flags(0x0000_0008 | 0x0000_0200);
    }
    cmd.spawn().context("start the sorrel window")?;
    println!("sorrel started; log: {}", log_path.display());
    Ok(())
}

/// `sorrel`: focus the running GUI, else start one with the launcher.
fn run_bare() -> Result<()> {
    if forward(&InstanceRequest::Focus)? {
        return Ok(());
    }
    start_gui(&["run".into()], None)
}

fn require_exists(open: &OpenArgs) -> Result<()> {
    anyhow::ensure!(
        open.root().exists(),
        "{} does not exist",
        open.root().display()
    );
    Ok(())
}

/// `sorrel open`: a tab in the running GUI, else a new detached GUI.
fn run_open(open: &OpenArgs) -> Result<()> {
    require_exists(open)?;
    let req = open_request()?;
    let msg = "opened in the running sorrel window";
    if forward(&InstanceRequest::Open {
        cwd: req.cwd.clone(),
        args: req.args.clone(),
    })? {
        println!("{msg}");
        return Ok(());
    }
    let run_args = ["run".to_owned()].into_iter().chain(req.args.clone());
    start_gui(&run_args.collect::<Vec<_>>(), Some(req))
}

/// `sorrel run`: the GUI in the foreground.
fn run_gui(open: &OpenArgs, standalone: bool) -> Result<()> {
    let first = match open.root {
        Some(_) => {
            require_exists(open)?;
            Some(open_request()?)
        }
        None => None,
    };
    if !standalone {
        let req = match &first {
            Some(r) => InstanceRequest::Open {
                cwd: r.cwd.clone(),
                args: r.args.clone(),
            },
            None => InstanceRequest::Focus,
        };
        if forward(&req)? {
            println!("opened in the running sorrel window");
            return Ok(());
        }
    }
    run_shell(tab_opener(), first, !standalone)
}

/// The dataset `sorrel mcp` (no DATA_DIR) serves: the GUI's active tab, else
/// the only live host.
fn mcp_target() -> Result<PathBuf> {
    // A hung GUI is skipped: the host registry may still name its datasets.
    let gui = sorrel_data::send_instance(&InstanceRequest::List)
        .ok()
        .flatten();
    let active = gui
        .as_ref()
        .and_then(|v| v["tabs"].as_array())
        .and_then(|t| t.iter().find(|t| t["active"] == true))
        .and_then(|t| t["dataset"].as_str());
    if let Some(d) = active {
        return Ok(d.into());
    }
    let mut alive: Vec<_> = sorrel_data::list_hosts()
        .into_iter()
        .filter(|h| h.alive)
        .collect();
    match alive.len() {
        0 => bail!("no running sorrel host; pass DATA_DIR"),
        1 => Ok(alive.remove(0).dataset),
        _ => bail!(
            "several sorrel hosts are running; pass DATA_DIR:\n{}",
            alive
                .iter()
                .map(|h| format!("  {} ({}, pid {})", h.dataset.display(), h.kind, h.pid))
                .collect::<Vec<_>>()
                .join("\n")
        ),
    }
}

/// `sorrel mcp` without DATA_DIR: bridge stdio to the target's host.
fn run_mcp_active() -> Result<()> {
    let dir = mcp_target()?;
    if !sorrel_data::attach(&dir, std::io::stdin(), std::io::stdout())? {
        bail!("the sorrel host for {} is not reachable", dir.display());
    }
    Ok(())
}

fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or_default()
}

/// `sorrel mcp tools`.
fn mcp_tools(json: bool) -> Result<()> {
    let list = sorrel_data::tool_list();
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({"tools": list}))?
        );
        return Ok(());
    }
    for t in list.as_array().into_iter().flatten() {
        let name = t["name"].as_str().unwrap_or_default();
        println!(
            "{name:<20}{}",
            first_line(t["description"].as_str().unwrap_or_default())
        );
    }
    Ok(())
}

/// `sorrel mcp config`: how to register this binary with an agent client.
fn mcp_config(client: Client, dir: Option<PathBuf>) -> Result<()> {
    let exe = std::env::current_exe()?.display().to_string();
    let mut args = vec!["mcp".to_owned()];
    if let Some(d) = dir {
        args.push(std::path::absolute(d)?.display().to_string());
    }
    let sh = |s: &str| {
        if s.chars()
            .all(|c| c.is_alphanumeric() || "/._-+:@".contains(c))
        {
            s.to_owned()
        } else {
            format!("'{}'", s.replace('\'', r"'\''"))
        }
    };
    let json = |s: &[String]| serde_json::to_string(s).expect("strings serialise");
    match client {
        Client::Claude => println!(
            "claude mcp add sorrel -- {} {}",
            sh(&exe),
            args.iter().map(|a| sh(a)).collect::<Vec<_>>().join(" ")
        ),
        // JSON string escapes are valid TOML basic strings.
        Client::Codex => println!(
            "[mcp_servers.sorrel]\ncommand = {}\nargs = {}",
            serde_json::to_string(&exe)?,
            json(&args)
        ),
        Client::Json => println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "mcpServers": {"sorrel": {"command": exe, "args": args}}
            }))?
        ),
    }
    Ok(())
}

/// One-shot tool call on a dataset opened in-process.
struct CallOnce<'a> {
    request: &'a str,
    reply: &'a mut Option<serde_json::Value>,
}
impl SessionVisitor for CallOnce<'_> {
    fn visit<P>(self, session: Session<P>, seed: Seed<P>, ctx: OpenCtx<P>) -> Result<()>
    where
        P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
    {
        let host = sorrel_data::SessionHost::new(seeded(session, seed)?);
        let mut env = sorrel_data::Env::new(&ctx.save_dir, ctx.reseal_on_save, phy_label_str);
        *self.reply = sorrel_data::handle_line(&host, &mut env, self.request);
        Ok(())
    }
}

/// `sorrel call`: through the running host, else in-process.
fn run_call(open: &OpenArgs, tool: &str, args: Option<&str>, raw: bool) -> Result<()> {
    let args: serde_json::Value = match args {
        Some(s) => serde_json::from_str(s).context("ARGS_JSON is not valid JSON")?,
        None => serde_json::json!({}),
    };
    anyhow::ensure!(args.is_object(), "ARGS_JSON must be a JSON object");
    let request = serde_json::json!({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": tool, "arguments": args},
    });
    let reply = match sorrel_data::call_once(&host_dir(open), &request)? {
        Some(r) => r,
        None => {
            let mut reply = None;
            let line = request.to_string();
            with_session(
                open,
                CallOnce {
                    request: &line,
                    reply: &mut reply,
                },
            )
            .map_err(busy_hint)?;
            reply.context("no reply")?
        }
    };
    print_reply(&reply, raw)
}

/// Print a `tools/call` reply; an RPC error or `isError` result is an `Err`.
fn print_reply(reply: &serde_json::Value, raw: bool) -> Result<()> {
    if raw {
        println!("{}", serde_json::to_string_pretty(reply)?);
    }
    if let Some(e) = reply.get("error") {
        bail!("{}", e["message"].as_str().unwrap_or("JSON-RPC error"));
    }
    let result = &reply["result"];
    let texts: Vec<&str> = result["content"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|c| c["text"].as_str())
        .collect();
    if result["isError"] == true {
        bail!("{}", texts.join("\n"));
    }
    if !raw {
        for t in texts {
            match serde_json::from_str::<serde_json::Value>(t) {
                Ok(v) => println!("{}", serde_json::to_string_pretty(&v)?),
                Err(_) => println!("{t}"),
            }
        }
    }
    Ok(())
}

/// `sorrel status`: the GUI, the hosts and the detached log.
fn run_status(dir: Option<&Path>, json: bool) -> Result<()> {
    let gui = sorrel_data::send_instance(&InstanceRequest::List).unwrap_or_else(|e| {
        log::warn!("GUI did not answer: {e:#}");
        None
    });
    let canon = |p: &Path| p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
    let want = dir.map(|d| {
        canon(&host_dir(&OpenArgs {
            root: Some(d.into()),
            ..Default::default()
        }))
    });
    let hosts: Vec<_> = sorrel_data::list_hosts()
        .into_iter()
        .filter(|h| want.as_ref().is_none_or(|w| canon(&h.dataset) == *w))
        .collect();
    let log = sorrel_data::user_dir()
        .map(|d| d.join("gui.log"))
        .ok()
        .filter(|p| p.exists());
    if json {
        let doc = serde_json::json!({"gui": gui, "hosts": hosts, "log": log});
        println!("{}", serde_json::to_string_pretty(&doc)?);
        return Ok(());
    }
    match &gui {
        None => println!("gui: not running"),
        Some(g) => {
            let tabs = g["tabs"].as_array().map_or(&[][..], Vec::as_slice);
            println!("gui: pid {}, {} tab(s)", g["pid"], tabs.len());
            for t in tabs {
                println!(
                    "  {}{} {}",
                    if t["active"] == true { "*" } else { " " },
                    if t["dirty"] == true { " (unsaved)" } else { "" },
                    t["dataset"].as_str().unwrap_or_default()
                );
            }
        }
    }
    println!("hosts:{}", if hosts.is_empty() { " none" } else { "" });
    for h in &hosts {
        println!(
            "  {:<9} pid {:<7} port {:<6} {:<6} {}",
            h.kind,
            h.pid,
            h.port,
            if h.alive { "alive" } else { "stale" },
            h.dataset.display()
        );
    }
    if let Some(l) = log {
        println!("log: {}", l.display());
    }
    Ok(())
}

/// `(name, about)` of every visible subcommand, nested ones as `parent child`.
fn command_rows() -> Vec<(String, String)> {
    fn walk(cmd: &clap::Command, prefix: &str, depth: usize, out: &mut Vec<(String, String)>) {
        for c in cmd.get_subcommands().filter(|c| !c.is_hide_set()) {
            let name = format!("{prefix}{}", c.get_name());
            let about = c.get_about().map(|a| a.to_string()).unwrap_or_default();
            out.push((name.clone(), about));
            if depth == 0 {
                walk(c, &format!("{name} "), 1, out);
            }
        }
    }
    let mut out = Vec::new();
    walk(&<Cli as clap::CommandFactory>::command(), "", 0, &mut out);
    out
}

/// The primer plus sections generated from the live program, so they cannot
/// drift from it.
fn agent_guide() -> String {
    use std::fmt::Write;
    let mut s = include_str!("agent_guide.md").trim_end().to_owned();
    s.push_str("\n\n## Commands\n\n");
    for (name, about) in command_rows() {
        let _ = writeln!(s, "- `sorrel {name}`: {}", first_line(&about));
    }
    s.push_str("\n## MCP tools\n\n");
    for t in sorrel_data::tool_list().as_array().into_iter().flatten() {
        let desc = t["description"].as_str().unwrap_or_default();
        let sentence = desc.split_once(". ").map_or(desc, |(a, _)| a);
        let _ = writeln!(
            s,
            "- `{}`: {}",
            t["name"].as_str().unwrap_or_default(),
            first_line(sentence)
        );
    }
    s.push_str("\n## Running now\n\n");
    let alive: Vec<_> = sorrel_data::list_hosts()
        .into_iter()
        .filter(|h| h.alive)
        .collect();
    if alive.is_empty() {
        s.push_str("none\n");
    }
    for h in alive {
        let _ = writeln!(s, "- {} ({}, pid {})", h.dataset.display(), h.kind, h.pid);
    }
    s
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli = Cli::parse();
    match cli.cmd {
        None => run_bare(),
        Some(Cmd::Open(open)) => run_open(&open),
        Some(Cmd::Run {
            open,
            standalone,
            headless,
        }) => match headless {
            true => with_session(&open, Headless),
            false => run_gui(&open, standalone),
        },
        Some(Cmd::Status { dir, json }) => run_status(dir.as_deref(), json),
        Some(Cmd::Call {
            open,
            tool,
            args,
            raw,
        }) => run_call(&open, &tool, args.as_deref(), raw),
        Some(Cmd::AgentGuide) => {
            print!("{}", agent_guide());
            Ok(())
        }
        Some(Cmd::Completions { shell }) => {
            clap_complete::generate(
                shell,
                &mut <Cli as clap::CommandFactory>::command(),
                "sorrel",
                &mut std::io::stdout(),
            );
            Ok(())
        }
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
                auto_llm_journal_ok(llm.is_some(), open.discard_journal)?;
                let mut flags = vec![];
                if let Some(backend) = llm {
                    let name = clap::ValueEnum::to_possible_value(&backend)
                        .expect("LlmBackend variants are not skipped")
                        .get_name()
                        .to_string();
                    flags.push(format!("--llm={name}"));
                    if let Some(model) = llm_model {
                        flags.extend(["--llm-model".to_string(), model]);
                    }
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
        Some(Cmd::Mcp { open, sub }) => match (sub, open.root.is_some()) {
            (Some(McpCmd::Tools { json }), _) => mcp_tools(json),
            (Some(McpCmd::Config { client, dir }), _) => mcp_config(client, dir),
            (None, true) => run_mcp(&open),
            (None, false) => run_mcp_active(),
        },
        Some(Cmd::Describe { open, json }) => with_session(&open, Describe { json }),
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

    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        Cli::try_parse_from(std::iter::once("sorrel").chain(args.iter().copied()))
    }

    #[test]
    fn auto_llm_backend_parses() {
        let llm = |args: &[&str]| match parse(args).unwrap().cmd {
            Some(Cmd::Auto {
                llm,
                llm_model,
                open,
                ..
            }) => (llm, llm_model, open.root),
            other => panic!("unexpected {other:?}"),
        };
        let dir = Some(PathBuf::from("DIR"));
        assert_eq!(llm(&["auto", "DIR"]), (None, None, dir.clone()));
        assert_eq!(
            llm(&["auto", "DIR", "--llm"]),
            (Some(LlmBackend::Claude), None, dir.clone())
        );
        // `--llm` before the positional must not swallow it
        assert_eq!(
            llm(&["auto", "--llm", "DIR"]),
            (Some(LlmBackend::Claude), None, dir.clone())
        );
        assert_eq!(
            llm(&["auto", "DIR", "--llm=codex"]).0,
            Some(LlmBackend::Codex)
        );
        assert_eq!(
            llm(&["auto", "--llm=opencode", "DIR"]).0,
            Some(LlmBackend::Opencode)
        );
        assert_eq!(
            llm(&["auto", "DIR", "--llm=api", "--llm-model", "m"]),
            (Some(LlmBackend::Api), Some("m".into()), dir)
        );
        assert!(parse(&["auto", "DIR", "--llm=gpt"]).is_err());
        assert!(parse(&["auto", "DIR", "--llm-model", "m"]).is_err());
        // a reused curation.json makes no LLM calls
        assert!(parse(&["auto", "DIR", "--llm", "--reuse"]).is_err());
        assert!(parse(&["auto", "DIR", "--reuse"]).is_ok());
        assert!(auto_llm_journal_ok(true, true).is_err());
        assert!(auto_llm_journal_ok(true, false).is_ok());
        assert!(auto_llm_journal_ok(false, true).is_ok());
    }

    #[test]
    fn gui_entry_points_parse() {
        assert!(parse(&[]).unwrap().cmd.is_none());

        match parse(&["run"]).unwrap().cmd {
            Some(Cmd::Run {
                open,
                standalone: false,
                headless: false,
            }) => {
                assert!(open.root.is_none());
            }
            other => panic!("unexpected {other:?}"),
        }
        match parse(&["run", "/x", "--standalone", "--channels", "8"])
            .unwrap()
            .cmd
        {
            Some(Cmd::Run {
                open,
                standalone: true,
                ..
            }) => {
                assert_eq!(open.root(), Path::new("/x"));
                assert_eq!(open.n_channels, Some(8));
            }
            other => panic!("unexpected {other:?}"),
        }
        let cli = parse(&["run", "--headless", "/x"]).unwrap();
        assert!(matches!(cli.cmd, Some(Cmd::Run { headless: true, .. })));
        assert!(
            parse(&["run", "--headless"]).is_err(),
            "headless needs DATA_DIR"
        );

        match parse(&["open", "/x", "--sample-rate", "3e4"]).unwrap().cmd {
            Some(Cmd::Open(open)) => {
                assert_eq!(open.root(), Path::new("/x"));
                assert_eq!(open.sample_rate, Some(30000.0));
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(parse(&["open"]).is_err(), "open needs DATA_DIR");
        assert!(parse(&["/x"]).is_err(), "`sorrel <DIR>` is gone");
        assert!(parse(&["--headless", "/x"]).is_err());
        assert!(parse(&["serve", "/x"]).is_err());
    }

    #[test]
    fn agent_commands_parse() {
        match parse(&["mcp"]).unwrap().cmd {
            Some(Cmd::Mcp { open, sub: None }) => assert!(open.root.is_none()),
            other => panic!("unexpected {other:?}"),
        }
        match parse(&["mcp", "/x"]).unwrap().cmd {
            Some(Cmd::Mcp { open, sub: None }) => assert_eq!(open.root(), Path::new("/x")),
            other => panic!("unexpected {other:?}"),
        }
        assert!(matches!(
            parse(&["mcp", "tools", "--json"]).unwrap().cmd,
            Some(Cmd::Mcp {
                sub: Some(McpCmd::Tools { json: true }),
                ..
            })
        ));
        assert!(matches!(
            parse(&["mcp", "config", "--client", "codex"]).unwrap().cmd,
            Some(Cmd::Mcp {
                sub: Some(McpCmd::Config {
                    client: Client::Codex,
                    dir: None
                }),
                ..
            })
        ));
        match parse(&["call", "/x", "metrics", r#"{"cluster":1}"#])
            .unwrap()
            .cmd
        {
            Some(Cmd::Call {
                open,
                tool,
                args,
                raw: false,
            }) => {
                assert_eq!(open.root(), Path::new("/x"));
                assert_eq!(tool, "metrics");
                assert_eq!(args.as_deref(), Some(r#"{"cluster":1}"#));
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(matches!(
            parse(&["status", "--json"]).unwrap().cmd,
            Some(Cmd::Status {
                dir: None,
                json: true
            })
        ));
        assert!(matches!(
            parse(&["completions", "bash"]).unwrap().cmd,
            Some(Cmd::Completions { .. })
        ));
        assert!(matches!(
            parse(&["agent-guide"]).unwrap().cmd,
            Some(Cmd::AgentGuide)
        ));
    }

    #[test]
    fn subcommands_still_parse() {
        let cli = parse(&["suggest", "/x", "--json"]).unwrap();
        assert!(matches!(cli.cmd, Some(Cmd::Suggest { json: true, .. })));

        match parse(&["qc", "/x", "--out", "o", "--backend", "si"])
            .unwrap()
            .cmd
        {
            Some(Cmd::Qc { open, out }) => {
                assert_eq!(out, PathBuf::from("o"));
                assert_eq!(open.backend, Some(Backend::SortingAnalyzer));
            }
            other => panic!("unexpected {other:?}"),
        }
        let cli = parse(&["journal", "dump", "j.journal", "--json"]).unwrap();
        assert!(matches!(
            cli.cmd,
            Some(Cmd::Journal {
                action: JournalCmd::Dump { json: true, .. }
            })
        ));
        assert!(matches!(
            parse(&["describe", "/x", "--json"]).unwrap().cmd,
            Some(Cmd::Describe { json: true, .. })
        ));
        #[cfg(feature = "phy-export")]
        assert!(matches!(
            parse(&["export-phy", "/x", "--out", "o", "--copy-raw"])
                .unwrap()
                .cmd,
            Some(Cmd::ExportPhy { copy_raw: true, .. })
        ));
        match parse(&["open", "/x", "--events", "e.csv"]).unwrap().cmd {
            Some(Cmd::Open(open)) => assert_eq!(open.events, Some(PathBuf::from("e.csv"))),
            other => panic!("unexpected {other:?}"),
        }
        assert!(parse(&["open", "/x", "--dtype", "int7"]).is_err());
        assert!(
            parse(&["open", "/x", "--export-qc", "o"]).is_err(),
            "removed flag"
        );
        assert!(parse(&["clear-cache", "/x"]).is_ok());
    }

    #[test]
    fn host_dir_of_a_missing_path_is_its_parent() {
        let open = OpenArgs {
            root: Some("/x".into()),
            ..Default::default()
        };
        assert_eq!(host_dir(&open), PathBuf::from("/"));
    }

    #[test]
    fn rebase_makes_every_path_absolute() {
        let rel = |s: &str| Some(PathBuf::from(s));
        let mut open = OpenArgs {
            root: rel("ds"),
            dat: rel("rec.dat"),
            journal: rel("j/x.journal"),
            spikeglx_meta: rel("a.meta"),
            oebin: rel("structure.oebin"),
            events: rel("e.csv"),
            ..Default::default()
        };
        open.rebase(Path::new("/cwd"));
        assert_eq!(open.root, rel("/cwd/ds"));
        assert_eq!(open.dat, rel("/cwd/rec.dat"));
        assert_eq!(open.journal, rel("/cwd/j/x.journal"));
        assert_eq!(open.spikeglx_meta, rel("/cwd/a.meta"));
        assert_eq!(open.oebin, rel("/cwd/structure.oebin"));
        assert_eq!(open.events, rel("/cwd/e.csv"));
        // Absolute paths and unset fields stay as they are.
        let mut open = OpenArgs {
            root: rel("/abs/ds"),
            ..Default::default()
        };
        open.rebase(Path::new("/cwd"));
        assert_eq!((open.root, open.dat), (rel("/abs/ds"), None));
    }

    #[test]
    fn agent_guide_lists_commands_and_tools() {
        let g = agent_guide();
        assert!(g.contains("## Commands") && g.contains("`sorrel call`"));
        assert!(
            g.contains("`sorrel mcp tools`"),
            "nested subcommands are listed"
        );
        assert!(g.contains("## MCP tools") && g.contains("`relabel`"));
        assert!(g.contains("## Running now"));
    }
}
