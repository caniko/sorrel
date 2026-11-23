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
#[cfg(any(feature = "hdf5", feature = "zarr"))]
use sorrel_data::baseline_hash;
use sorrel_data::session::ApplyPhyLabel;
use sorrel_data::InstanceRequest;
use sorrel_data::{
    apply_rules, collect_qc_rows_with, diff_labels, export_qc, rank_merge_candidates,
    rank_split_candidates, save_session as save_dir_session, save_to_phy, serve, write_report,
    Journal, ReportFilter, RuleSet, Session, SuggestConfig,
};
#[cfg(any(feature = "hdf5", feature = "zarr"))]
use sorrel_io::detect_sorter;
use sorrel_io::kilosort::{read_cluster_group_tsv, KilosortOpenParams, PhyLabel};
use sorrel_io::open_ephys::OebinMeta;
use sorrel_io::spikeglx::SpikeGlxMeta;
use sorrel_io::{
    DataProvider, Events, HasGeometry, HasQualityMetrics, KilosortProvider, TraceDtype,
};
use sorrel_ui::{OpenRequest, Opened, Opener, Shell, SorrelApp, Tab};
use std::path::{Path, PathBuf};

#[cfg(feature = "dev")]
mod dev;
#[cfg(feature = "hdf5")]
mod export_nwb;
mod rescue;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum Backend {
    Kilosort,
    SortingAnalyzer,
    #[cfg(any(feature = "hdf5", feature = "zarr"))]
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
        /// `sorrel mcp` and the same dataset path and open options. Stops on Ctrl-C
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
    AgentGuide {
        /// Print only the effective text of this MCP prompt (`review`, alias
        /// `triage`), as `prompts/get` serves it, instead of the primer
        #[arg(long, value_name = "NAME", value_parser = ["review", "triage"])]
        prompt: Option<String>,
    },
    /// Print a shell completion script
    Completions { shell: clap_complete::Shell },
    /// Windows: add a Start-menu shortcut to `sorrelw.exe` (`--remove` deletes it)
    InstallShortcut {
        #[arg(long)]
        remove: bool,
    },
    /// Headless: run every metric and write cluster_qc.tsv + cluster_qc.json
    Qc {
        #[command(flatten)]
        open: OpenArgs,
        /// Output directory (required for local datasets)
        #[arg(long, value_name = "DIR")]
        out: Option<PathBuf>,
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
        #[arg(long, value_name = "FILE")]
        out: Option<PathBuf>,
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
    /// Compare two cluster_group.tsv files (confusion table, kappa,
    /// disagreements), or with `--root`, two curated directories spike by
    /// spike (the bench score)
    Diff {
        /// cluster_group.tsv, or the truth directory
        a: PathBuf,
        /// cluster_group.tsv, or the directory under test
        b: PathBuf,
        #[arg(long)]
        json: bool,
        /// Directory mode: the sort both started from (a `sorrel blind` input)
        #[arg(long, value_name = "DIR")]
        root: Option<PathBuf>,
        /// Directory mode: minimum Jaccard overlap to count a unit as recovered
        #[arg(long = "match", value_name = "JACCARD", default_value_t = 0.8)]
        match_jaccard: f64,
        /// Directory mode: root clusters with fewer spikes are left out of the
        /// merge and split scores
        #[arg(long, value_name = "N", default_value_t = 10)]
        min_spikes: usize,
        /// Directory mode: a split piece must hold this share of its root cluster's
        /// kept (non-noise) spikes
        #[arg(long, value_name = "F", default_value_t = 0.1)]
        split_share: f64,
        /// Directory mode: score good against not-good (mua and unsorted count as
        /// noise); the default is on when the truth never uses mua
        #[arg(long, conflicts_with = "no_binary")]
        binary: bool,
        /// Directory mode: keep mua and unsorted distinct even if the truth never uses mua
        #[arg(long)]
        no_binary: bool,
        /// Directory mode: root cluster ids (one per line) to leave out of scoring
        #[arg(long, value_name = "FILE")]
        disputed: Option<PathBuf>,
        /// Dev builds: write the score and misses of the recorded run RID into the results DB
        #[cfg(feature = "dev")]
        #[arg(long, value_name = "RID")]
        record: Option<String>,
        /// Dev builds: the results DB for --record (default as for `sorrel dev`)
        #[cfg(feature = "dev")]
        #[arg(long, value_name = "FILE", requires = "record")]
        db: Option<PathBuf>,
    },
    /// Write a double-blind bench pair from a curated session: --input is the
    /// sorter's output with no curation, --truth the curated head plus
    /// provenance.json
    Blind {
        #[command(flatten)]
        open: OpenArgs,
        /// Blind input directory (must be empty or absent)
        #[arg(long, value_name = "DIR")]
        input: PathBuf,
        /// Truth directory: spike_clusters.npy, cluster_group.tsv, curation.json
        /// and provenance.json of the head
        #[arg(long, value_name = "DIR")]
        truth: PathBuf,
        /// The sort was curated in phy, not sorrel: spike_templates.npy is the
        /// sorter's clustering, spike_clusters.npy and cluster_group.tsv the
        /// human's. DATA_DIR is only read.
        #[arg(long)]
        phy_curated: bool,
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
        /// Output directory (required for local datasets; must not be the source)
        #[arg(long, value_name = "DIR")]
        out: Option<PathBuf>,
        /// Copy the raw recording into DIR/recording.dat instead of pointing
        /// params.py at the original
        #[arg(long)]
        copy_raw: bool,
    },
    /// Write curated units to a new or existing NWB file
    #[cfg(feature = "hdf5")]
    ExportNwb(export_nwb::ExportNwb),
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
    /// The bench results DB (record sessions, runs and scores; distill what
    /// they teach). Built with `--features dev` only
    #[cfg(feature = "dev")]
    Dev(dev::DevArgs),
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
    #[command(flatten)]
    refractory: RefractoryArgs,
    /// phy / Kilosort, SortingAnalyzer, NWB (local, DANDI or HTTPS), or legacy MATLAB rez.mat
    #[arg(value_name = "DATA_DIR", required = true)]
    root: Option<PathBuf>,
    /// kilosort | sorting-analyzer | nwb | rez (legacy alias: ks4-rez)
    #[arg(long, value_name = "NAME", value_parser = parse_backend)]
    backend: Option<Backend>,
    /// Raw recording (overrides params.py:dat_path, or pairs NWB units with a raw NWB)
    #[arg(long, value_name = "PATH")]
    dat: Option<PathBuf>,
    /// Overrides params.py:sample_rate. For NWB, select an exact declared clock (copy a value from the error), or supply a rate if no clock is declared
    #[arg(long, value_name = "Hz")]
    sample_rate: Option<f64>,
    /// NWB ElectricalSeries name under /acquisition or absolute NWB object path
    #[cfg(any(feature = "hdf5", feature = "zarr"))]
    #[arg(long, value_name = "PATH|NAME")]
    nwb_series: Option<String>,
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

#[derive(clap::Args, Clone, Copy, Debug, Default)]
struct RefractoryArgs {
    /// Fixed refractory window (ms).
    #[arg(long)]
    refractory_ms: Option<f32>,
    /// Exclusive lower bound of the refractory sweep (ms).
    #[arg(long)]
    rp_min_ms: Option<f32>,
    /// Inclusive upper bound of the refractory sweep (ms).
    #[arg(long)]
    rp_max_ms: Option<f32>,
    /// Refractory sweep step (ms).
    #[arg(long)]
    rp_step_ms: Option<f32>,
}

impl RefractoryArgs {
    fn resolve(
        self,
        base: sorrel_compute::RefractoryParams,
    ) -> Result<sorrel_compute::RefractoryParams> {
        let params = sorrel_compute::RefractoryParams {
            fixed_ms: self.refractory_ms.unwrap_or(base.fixed_ms),
            sweep_min_ms: self.rp_min_ms.unwrap_or(base.sweep_min_ms),
            sweep_max_ms: self.rp_max_ms.unwrap_or(base.sweep_max_ms),
            sweep_step_ms: self.rp_step_ms.unwrap_or(base.sweep_step_ms),
        };
        params
            .validate()
            .map_err(|e| anyhow::anyhow!("Invalid refractory settings: {e}"))?;
        Ok(params)
    }

    fn overrides(self) -> [bool; 4] {
        [
            self.refractory_ms.is_some(),
            self.rp_min_ms.is_some(),
            self.rp_max_ms.is_some(),
            self.rp_step_ms.is_some(),
        ]
    }

    /// When attaching, patch only flags that were supplied, not the host's defaults.
    fn configure_host(
        self,
        call: impl FnOnce(&serde_json::Value) -> Result<Option<serde_json::Value>>,
    ) -> Result<bool> {
        let mut arguments = serde_json::Map::new();
        for (key, value) in [
            ("fixed_ms", self.refractory_ms),
            ("sweep_min_ms", self.rp_min_ms),
            ("sweep_max_ms", self.rp_max_ms),
            ("sweep_step_ms", self.rp_step_ms),
        ] {
            if let Some(value) = value {
                anyhow::ensure!(
                    value.is_finite() && value > 0.0,
                    "Invalid refractory settings: {key} must be finite and greater than zero"
                );
                arguments.insert(key.into(), serde_json::json!(value));
            }
        }
        if arguments.is_empty() {
            return Ok(false);
        }
        let request = serde_json::json!({"jsonrpc":"2.0", "id":0, "method":"tools/call", "params":{"name":"refractory", "arguments":arguments}});
        if let Some(reply) = call(&request)? {
            if reply.get("error").is_some() || reply["result"]["isError"] != false {
                let message = reply["error"]["message"]
                    .as_str()
                    .or_else(|| reply["result"]["content"][0]["text"].as_str())
                    .unwrap_or("host rejected the request");
                bail!("Invalid refractory settings: {message}");
            }
            return Ok(true);
        }
        Ok(false)
    }
}

impl OpenArgs {
    fn root(&self) -> &Path {
        self.root.as_deref().expect("clap marks DATA_DIR required")
    }

    fn has_remote_input(&self) -> bool {
        is_remote(self.root()) || self.dat.as_deref().is_some_and(is_remote)
    }

    /// Make the path fields absolute against `cwd`: the args came from a
    /// `sorrel open` run somewhere else.
    fn rebase(&mut self, cwd: &Path) {
        for p in [&mut self.root, &mut self.dat].into_iter().flatten() {
            if !is_remote(p) {
                *p = cwd.join(&*p);
            }
        }
        for p in [
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
        #[cfg(any(feature = "hdf5", feature = "zarr"))]
        "nwb" => Backend::Nwb,
        #[cfg(not(any(feature = "hdf5", feature = "zarr")))]
        "nwb" => {
            return Err(
                "backend nwb is unavailable in this build; rebuild with default features".into(),
            )
        }
        #[cfg(feature = "hdf5")]
        "ks4-rez" | "rez" => Backend::Ks4Rez,
        #[cfg(not(feature = "hdf5"))]
        "ks4-rez" | "rez" => {
            return Err(format!(
                "backend {s} (legacy MATLAB rez.mat) requires HDF5; rebuild with default features"
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
#[cfg(any(feature = "hdf5", feature = "zarr"))]
fn journal_path(open: &OpenArgs, dir: &Path, stem: &str) -> PathBuf {
    sorrel_data::journal_path(&journal_opts(open), dir, stem)
}

/// Open the journal sealed against `baseline`, honouring `--discard-journal`.
#[cfg(any(feature = "hdf5", feature = "zarr"))]
fn open_journal(open: &OpenArgs, path: &Path, baseline: u64, n_clusters: u32) -> Result<Journal> {
    sorrel_data::open_journal(&journal_opts(open), path, baseline, n_clusters).map_err(|e| {
        if open.has_remote_input() && e.is::<sorrel_data::journal::JournalBusy>() {
            anyhow::anyhow!("{e}: stop the running session or pass a different --journal")
        } else {
            e
        }
    })
}

fn detect_backend(root: &Path) -> Result<Backend> {
    if is_remote(root) {
        #[cfg(feature = "dandi")]
        return Ok(Backend::Nwb);
        #[cfg(not(feature = "dandi"))]
        bail!("remote NWB datasets require building with --features dandi");
    }
    Ok(match sorrel_io::detect_backend(root)? {
        sorrel_io::Backend::Kilosort => Backend::Kilosort,
        sorrel_io::Backend::SortingAnalyzer => Backend::SortingAnalyzer,
        sorrel_io::Backend::Nwb => {
            let (enabled, feature) = if root.is_dir() {
                (cfg!(feature = "zarr"), "zarr")
            } else {
                (cfg!(feature = "hdf5"), "hdf5")
            };
            anyhow::ensure!(
                enabled,
                "{} needs the {feature} feature; rebuild with default features",
                root.display()
            );
            #[cfg(any(feature = "hdf5", feature = "zarr"))]
            {
                Backend::Nwb
            }
            #[cfg(not(any(feature = "hdf5", feature = "zarr")))]
            unreachable!()
        }
        #[cfg(feature = "hdf5")]
        sorrel_io::Backend::Ks4Rez => Backend::Ks4Rez,
        #[cfg(not(feature = "hdf5"))]
        sorrel_io::Backend::Ks4Rez => {
            bail!(
                "{} needs the hdf5 feature; rebuild with default features",
                root.display()
            )
        }
    })
}

/// Merge explicit CLI overrides with SpikeGLX/Open Ephys recording metadata.
/// The provider fills any remaining defaults from `params.py`.
fn kilosort_open_params(open: &OpenArgs) -> Result<KilosortOpenParams> {
    let mut p = KilosortOpenParams {
        sample_rate: open.sample_rate,
        n_channels: open.n_channels,
        dtype: open.dtype,
        offset: open.offset,
        dat_path: open.dat.clone(),
        ..Default::default()
    };
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
    Ok(p)
}

/// Where a session's artifacts go, how its journal is treated on save, and
/// which backend-specific extras the GUI may use.
struct OpenCtx<P: DataProvider> {
    /// Directory `save_to_phy` writes into.
    save_dir: PathBuf,
    /// Remote source: default exports stay inside the local state directory.
    remote: bool,
    /// True only for backends that read the saved `spike_clusters.npy`
    /// back on open (phy/Kilosort); see `SorrelApp::set_save_dir`.
    reseal_on_save: bool,
    positions: Vec<[f32; 2]>,
    /// Shank id per raw channel (empty = single shank); drives per-shank CMR.
    shanks: Vec<u32>,
    /// Per-cluster quality metrics (KS `cluster_*.tsv`, SI `quality_metrics.csv`);
    /// empty when the backend has none.
    metrics: sorrel_io::QualityMetrics,
    /// NWB session-clock offset for the PSTH's trials; CSV/NPY are already relative.
    nwb_start_time: f64,
    /// Per-spike depth for the drift map; needs PC features + geometry, so
    /// only the phy backend provides it (`sorrel_data::spike_depths`).
    depth_source: Option<sorrel_ui::app::DepthFn<P>>,
}

/// Seeds optional per-spike data and replays the journal. Headless commands
/// run it at once; the GUI runs it on a worker behind a loading screen.
type Seed<P> = Box<dyn FnOnce(&mut Session<P>) -> Result<()> + Send + 'static>;

/// A subcommand body, monomorphised once per backend by [`with_session`].
trait SessionVisitor {
    #[cfg(feature = "dandi")]
    fn read_remote_trials(&mut self, _: &sorrel_io::NwbProvider) {}

    #[cfg(feature = "hdf5")]
    fn prepare_nwb(&mut self, _: &sorrel_io::NwbProvider) -> Result<()> {
        Ok(())
    }

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
    with_session_refractory(open, Default::default(), v)
}

fn with_session_refractory<V: SessionVisitor>(
    open: &OpenArgs,
    base: sorrel_compute::RefractoryParams,
    v: V,
) -> Result<()> {
    let refractory = open.refractory.resolve(base)?;
    let inputs = RemoteInputs::resolve(open)?;
    #[cfg(any(feature = "hdf5", feature = "dandi"))]
    let mut v = v;
    #[cfg(not(feature = "dandi"))]
    let _ = inputs;
    let backend = match (open.backend, is_remote(open.root())) {
        (backend, true) => {
            let detected = detect_backend(open.root())?;
            anyhow::ensure!(
                backend.is_none_or(|b| b == detected),
                "a remote dataset requires the NWB backend"
            );
            detected
        }
        (Some(b), false) => b,
        (None, false) => detect_backend(open.root())?,
    };
    #[cfg(feature = "dandi")]
    anyhow::ensure!(
        inputs.dat.is_none() || backend == Backend::Nwb,
        "a remote --dat recording requires the NWB backend"
    );
    match backend {
        Backend::Kilosort => {
            let (mut session, ctx) = open_kilosort(open, kilosort_open_params(open)?)?;
            session
                .set_refractory(refractory)
                .map_err(anyhow::Error::msg)?;
            v.visit(session, Box::new(seed_kilosort), ctx)
        }
        Backend::SortingAnalyzer => {
            let root = open.root().to_path_buf();
            // The analyzer's own sorting is reloaded on every open, so the
            // journal stays the source of truth and is never sealed.
            let mut session = sorrel_data::open_sorting_analyzer(&root, &journal_opts(open))?;
            session
                .set_refractory(refractory)
                .map_err(anyhow::Error::msg)?;
            if let Some(flag) = &open.sorter {
                session.set_sorter(flag.clone());
            }
            let metrics = session.provider.quality_metrics().clone();
            let positions = session.provider.channel_positions().to_vec();
            let shanks = session.provider.channel_shanks().to_vec();
            v.visit(
                session,
                Box::new(|s| s.replay_journal()),
                OpenCtx {
                    remote: false,
                    save_dir: root,
                    reseal_on_save: false,
                    positions,
                    shanks,
                    metrics,
                    nwb_start_time: 0.0,
                    depth_source: None,
                },
            )
        }
        #[cfg(any(feature = "hdf5", feature = "zarr"))]
        Backend::Nwb => {
            let path = open.root().to_path_buf();
            let params = sorrel_io::NwbOpenParams {
                series: open.nwb_series.as_deref(),
                raw_path: open.dat.as_deref(),
                sample_rate: open.sample_rate,
            };
            #[cfg(feature = "dandi")]
            let (provider, path) = {
                let dir = inputs
                    .root
                    .as_ref()
                    .map(|asset| Ok::<_, anyhow::Error>(dandi_data_dir()?.join(asset.state_path())))
                    .transpose()?;
                let provider =
                    sorrel_io::NwbProvider::open_pinned(&path, params, inputs.root, inputs.dat)?;
                let path = if let Some(dir) = dir {
                    v.read_remote_trials(&provider);
                    let mut builder = std::fs::DirBuilder::new();
                    builder.recursive(true);
                    #[cfg(unix)]
                    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
                    builder
                        .create(&dir)
                        .context("create local DANDI state directory")?;
                    dir.join("nwb")
                } else {
                    path
                };
                (provider, path)
            };
            #[cfg(not(feature = "dandi"))]
            let provider = sorrel_io::NwbProvider::open_with(&path, params)?;
            #[cfg(feature = "hdf5")]
            v.prepare_nwb(&provider)?;
            let starting_time = provider.starting_time();
            let metrics = provider.quality_metrics().clone();
            let (mut session, mut ctx) =
                open_file_backed(open, provider, &path, "nwb", sorrel_io::Backend::Nwb)?;
            ctx.nwb_start_time = starting_time;
            ctx.remote = is_remote(open.root());
            ctx.metrics = metrics;
            session
                .set_refractory(refractory)
                .map_err(anyhow::Error::msg)?;
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
            let (mut session, ctx) =
                open_file_backed(open, provider, &path, "rez", sorrel_io::Backend::Ks4Rez)?;
            session
                .set_refractory(refractory)
                .map_err(anyhow::Error::msg)?;
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
#[cfg(any(feature = "hdf5", feature = "zarr"))]
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
            remote: false,
            save_dir: root,
            reseal_on_save: true,
            positions,
            shanks,
            metrics,
            nwb_start_time: 0.0,
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
    let refractory = open.refractory.resolve(Default::default())?;
    let backend = match open.backend {
        Some(b) => b,
        None => detect_backend(open.root())?,
    };
    if backend != Backend::Kilosort {
        // ponytail: other backends map the recording differently (or load it
        // into RAM); add them when someone needs to share one.
        bail!("remote share supports kilosort/phy datasets");
    }
    let (mut session, ctx) = open_kilosort(open, kilosort_open_params(open)?)?;
    session
        .set_refractory(refractory)
        .map_err(anyhow::Error::msg)?;
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
        remote: false,
        save_dir: joined.root.clone(),
        reseal_on_save: false,
        positions: joined.positions.clone(),
        shanks: Vec::new(),
        metrics: Default::default(),
        nwb_start_time: 0.0,
        depth_source: Some(sorrel_data::spike_depths),
    };
    // The network state lives in this process, so the window is standalone
    // and its one tab is the session just joined (`joined` stays alive until
    // the window closes).
    let mut session = joined.session;
    let mut once = Some(
        move |rs: Option<&RenderState>, refractory| -> Result<Opened> {
            session
                .set_refractory(refractory)
                .map_err(anyhow::Error::msg)?;
            let app = build_tab(session, Box::new(|_| Ok(())), ctx, None, rs, None, None);
            Ok(Opened::Tab(Box::new(app), root))
        },
    );
    let opener: Opener = Box::new(move |_, _, rs, refractory| {
        once.take().context("a joined session is a single window")?(rs, refractory)
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

/// NWB and rez.mat journals and save artifacts live beside the file or store.
#[cfg(any(feature = "hdf5", feature = "zarr"))]
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
            remote: false,
            save_dir: parent,
            reseal_on_save: false,
            positions,
            shanks,
            metrics: Default::default(),
            nwb_start_time: 0.0,
            depth_source: None,
        },
    ))
}

// ---------------------------------------------------------------- commands

/// Builds the GUI tab for a session (the typed `SorrelApp<P>` is boxed into
/// a [`Tab`], so the visitor trait stays unchanged).
struct TabBuilder<'a> {
    out: &'a mut Option<Box<dyn Tab>>,
    open: &'a OpenArgs,
    rs: Option<&'a RenderState>,
    session_key: Option<String>,
    remote_trials: Option<Events>,
}
impl SessionVisitor for TabBuilder<'_> {
    #[cfg(feature = "dandi")]
    fn read_remote_trials(&mut self, provider: &sorrel_io::NwbProvider) {
        if self.open.events.is_none() {
            self.remote_trials = implicit_nwb_trials(provider.read_remote_trials());
        }
    }

    fn visit<P>(self, session: Session<P>, seed: Seed<P>, ctx: OpenCtx<P>) -> Result<()>
    where
        P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
    {
        let events = load_events(self.open, ctx.nwb_start_time, self.remote_trials)?;
        *self.out = Some(Box::new(build_tab(
            session,
            seed,
            ctx,
            events,
            self.rs,
            self.session_key,
            remote_registry_key(self.open)?,
        )));
        Ok(())
    }
}

struct Qc(Option<PathBuf>);
impl SessionVisitor for Qc {
    fn visit<P>(self, session: Session<P>, seed: Seed<P>, ctx: OpenCtx<P>) -> Result<()>
    where
        P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
    {
        let out = dataset_output(&ctx, self.0, "qc", None)?;
        let session = seeded(session, seed)?;
        let out_dir = &out;
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

/// `sorrel blind`: the double-blind bench pair of a curated session.
struct Blind {
    dat: Option<PathBuf>,
    input: PathBuf,
    truth: PathBuf,
}
impl SessionVisitor for Blind {
    fn visit<P>(self, session: Session<P>, seed: Seed<P>, ctx: OpenCtx<P>) -> Result<()>
    where
        P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
    {
        if !ctx.reseal_on_save {
            bail!("sorrel blind supports phy/Kilosort directories");
        }
        let session = seeded(session, seed)?;
        if self.truth.exists() && std::fs::read_dir(&self.truth)?.next().is_some() {
            bail!("{} is not empty", self.truth.display());
        }
        let files = sorrel_data::write_blind_input(
            &ctx.save_dir,
            self.dat.as_deref(),
            &self.input,
            "spike_clusters.npy",
            session.journal().baseline(),
        )?;
        std::fs::create_dir_all(&self.truth)
            .with_context(|| format!("create {}", self.truth.display()))?;
        save_to_phy(&session, &self.truth, phy_label_str)?;
        let prov = sorrel_data::provenance(&session)?;
        std::fs::write(
            self.truth.join("provenance.json"),
            serde_json::to_vec_pretty(&prov)?,
        )?;
        println!(
            "blind input {} ({} files), truth {} (pure_manual {}, anchored {})",
            self.input.display(),
            files.len(),
            self.truth.display(),
            prov.pure_manual,
            prov.anchored
        );
        Ok(())
    }
}

#[cfg(feature = "phy-export")]
struct ExportPhy {
    out: Option<PathBuf>,
    copy_raw: bool,
}
#[cfg(feature = "phy-export")]
impl SessionVisitor for ExportPhy {
    fn visit<P>(self, session: Session<P>, seed: Seed<P>, ctx: OpenCtx<P>) -> Result<()>
    where
        P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
    {
        let out = dataset_output(&ctx, self.out, "phy", None)?;
        let session = seeded(session, seed)?;
        sorrel_data::export_phy(&session, &out, self.copy_raw, phy_label_str)
            .with_context(|| format!("exporting phy to {}", out.display()))?;
        println!(
            "exported {} clusters as phy into {}",
            session.n_clusters(),
            out.display()
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
    println!("saved → {}", sorrel_io::fsutil::display_path(&ctx.save_dir));
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
    let cfg = SuggestConfig::default();
    for cand in rank_split_candidates(session, &cfg) {
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

    // One prompt source with the MCP `review` prompt; a path override is resolved to text.
    let (rubric, _) = serve::effective_prompts()?;
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
        .env("SORREL_RUBRIC", rubric)
        .env("SORREL_PROMPT_REV", serve::prompt_rev()?)
        .status()
        .context("spawn the sorrel_auto.py sidecar (install uv, or set SORREL_PYTHON)")?;
    if !status.success() {
        bail!("sorrel_auto.py failed ({status})");
    }
    Ok(())
}

struct Report {
    out: Option<PathBuf>,
    filter: ReportFilter,
}
impl SessionVisitor for Report {
    fn visit<P>(self, session: Session<P>, seed: Seed<P>, ctx: OpenCtx<P>) -> Result<()>
    where
        P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
    {
        let out = dataset_output(
            &ctx,
            self.out,
            "sorrel_report.html",
            Some("sorrel_report.html"),
        )?;
        let session = seeded(session, seed)?;
        let n = write_report(&session, &out, phy_label_str, &self.filter)?;
        println!("wrote {n} cluster cards → {}", out.display());
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
    match is_dataset_directory(root) {
        true => root.to_path_buf(),
        false => root.parent().unwrap_or(Path::new(".")).to_path_buf(),
    }
}

fn session_host_dir(open: &OpenArgs) -> Result<PathBuf> {
    if is_remote(open.root()) {
        #[cfg(feature = "dandi")]
        return Ok(
            dandi_data_dir()?.join(sorrel_io::dandi::state_path(open.root().to_str().unwrap())?)
        );
        #[cfg(not(feature = "dandi"))]
        bail!("remote NWB datasets require building with --features dandi");
    }
    Ok(host_dir(open))
}

fn dataset_output<P: DataProvider>(
    ctx: &OpenCtx<P>,
    out: Option<PathBuf>,
    remote_name: &str,
    local_default: Option<&str>,
) -> Result<PathBuf> {
    if let Some(out) = out {
        Ok(out)
    } else if ctx.remote {
        Ok(ctx.save_dir.join(remote_name))
    } else {
        Ok(PathBuf::from(
            local_default.context("--out is required for local datasets")?,
        ))
    }
}

fn is_dataset_directory(path: &Path) -> bool {
    !is_remote(path) && path.is_dir() && !sorrel_io::nwb::is_zarr_path(path)
}

fn is_remote(path: &Path) -> bool {
    #[cfg(feature = "dandi")]
    return sorrel_io::dandi::is_remote(path);
    #[cfg(not(feature = "dandi"))]
    path.to_str()
        .is_some_and(|s| s.starts_with("dandi://") || s.starts_with("https://"))
}

/// Keep both verified stores with the request so the session opens exactly those revisions.
struct RemoteInputs {
    #[cfg(feature = "dandi")]
    root: Option<sorrel_io::dandi::Resolved>,
    #[cfg(feature = "dandi")]
    dat: Option<sorrel_io::dandi::Resolved>,
}

impl RemoteInputs {
    fn resolve(open: &OpenArgs) -> Result<Self> {
        #[cfg(feature = "dandi")]
        {
            let [root, dat] = [Some(open.root()), open.dat.as_deref()]
                .map(|path| path.filter(|path| is_remote(path)).and_then(Path::to_str));
            let [root, dat] = sorrel_io::dandi::resolve_pair(root, dat)?;
            Ok(Self { root, dat })
        }
        #[cfg(not(feature = "dandi"))]
        {
            anyhow::ensure!(
                !open.has_remote_input(),
                "remote NWB datasets require building with --features dandi"
            );
            Ok(Self {})
        }
    }
}

#[cfg(feature = "dandi")]
fn dandi_data_dir() -> Result<PathBuf> {
    let var = |key| {
        std::env::var_os(key)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    let root = if let Some(dir) = var("SORREL_USER_DIR") {
        dir
    } else if cfg!(windows) {
        var("LOCALAPPDATA")
            .context("LOCALAPPDATA is not set")?
            .join("sorrel")
    } else if let Some(dir) = var("XDG_DATA_HOME").filter(|path| path.is_absolute()) {
        dir.join("sorrel")
    } else if cfg!(target_os = "macos") {
        var("HOME")
            .context("HOME is not set")?
            .join("Library/Application Support/sorrel")
    } else {
        var("HOME")
            .context("HOME is not set")?
            .join(".local/share/sorrel")
    };
    Ok(root.join("dandi"))
}

fn session_path(path: &Path) -> PathBuf {
    let path = if path.as_os_str().is_empty() {
        Path::new(".")
    } else {
        path
    };
    if let Ok(path) = path.canonicalize() {
        return path;
    }
    // A file the host creates later (a journal) must key the same before and
    // after it exists: canonicalize its nearest existing ancestor instead.
    match (path.parent(), path.file_name()) {
        (Some(parent), Some(name)) => session_path(parent).join(name),
        _ => path.to_path_buf(),
    }
}

/// Only local requests may attach to a running session. Remote and mixed
/// requests always open fresh and let the journal lock arbitrate ownership.
fn session_key(open: &OpenArgs) -> Result<Option<String>> {
    if open.has_remote_input() {
        return Ok(None);
    }
    let mut args = open.clone();
    args.rebase(&std::env::current_dir()?);
    // Discard is a one-shot action, not a property of the live session.
    args.discard_journal = false;
    // Refractory flags patch a matching live session; they don't identify it.
    args.refractory = Default::default();
    // A file's name matters even through a symlink: it names its journal.
    args.root = Some(if is_dataset_directory(args.root()) {
        session_path(args.root())
    } else {
        session_path(&host_dir(&args)).join(
            args.root()
                .file_name()
                .context("missing dataset filename")?,
        )
    });
    if let Some(path) = args.dat.as_mut() {
        *path = session_path(path);
    }
    for path in [
        &mut args.journal,
        &mut args.spikeglx_meta,
        &mut args.oebin,
        &mut args.events,
    ]
    .into_iter()
    .flatten()
    {
        *path = session_path(path);
    }
    #[cfg(any(feature = "hdf5", feature = "zarr"))]
    if let Some(name) = args.nwb_series.as_mut().filter(|s| !s.starts_with('/')) {
        *name = format!("/acquisition/{name}");
    }
    // ponytail: compare normalized flags without opening a provider to resolve implicit defaults.
    Ok(Some(format!("open-args-v1:{args:?}")))
}

/// A remote host can share its save directory with other sessions. Hash the
/// full request (including meaningful URL queries) without recording it.
#[cfg(feature = "dandi")]
fn remote_registry_key(open: &OpenArgs) -> Result<Option<String>> {
    if !open.has_remote_input() {
        return Ok(None);
    }
    let mut args = open.clone();
    args.rebase(&std::env::current_dir()?);
    args.discard_journal = false;
    args.refractory = Default::default();
    Ok(Some(
        blake3::hash(format!("{args:?}").as_bytes())
            .to_hex()
            .to_string(),
    ))
}

#[cfg(not(feature = "dandi"))]
fn remote_registry_key(_: &OpenArgs) -> Result<Option<String>> {
    Ok(None)
}

/// `sorrel mcp`: bridge stdio to a running host, else serve in-process.
fn run_mcp(open: &OpenArgs) -> Result<()> {
    if let Some(key) = session_key(open)? {
        let dir = host_dir(open);
        open.refractory.configure_host(|request| {
            sorrel_data::serve::call_once_for_session(
                &dir,
                Some(&key),
                open.discard_journal,
                request,
            )
        })?;
        if sorrel_data::serve::attach_for_session(
            &dir,
            Some(&key),
            open.discard_journal,
            std::io::stdin(),
            std::io::stdout(),
        )? {
            return Ok(());
        }
    }
    // If another process holds the journal, the fresh open refuses a second writer.
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
struct Headless<'a>(&'a OpenArgs, Option<String>);
impl SessionVisitor for Headless<'_> {
    fn visit<P>(self, session: Session<P>, seed: Seed<P>, ctx: OpenCtx<P>) -> Result<()>
    where
        P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
    {
        let command = if self.1.is_some() {
            format!("sorrel mcp {}", mcp_open_args(self.0)?)
        } else {
            "sorrel mcp".to_owned()
        };
        let host = std::sync::Arc::new(sorrel_data::SessionHost::new(seeded(session, seed)?));
        let listener = sorrel_data::serve::spawn_listener_for_session(
            host,
            &ctx.save_dir,
            ctx.reseal_on_save,
            phy_label_str,
            None,
            self.1.as_deref(),
            remote_registry_key(self.0)?.as_deref(),
        )?;
        println!(
            "sorrel host on 127.0.0.1:{}; connect an agent with `{command}`; Ctrl-C stops",
            listener.port()
        );
        // Ctrl-C returns, dropping the listener so it removes its registry file.
        sorrel_remote::wait_for_ctrl_c();
        drop(listener);
        Ok(())
    }
}

fn mcp_open_args(open: &OpenArgs) -> Result<String> {
    let mut open = open.clone();
    open.rebase(&std::env::current_dir()?);
    let mut args = if is_remote(open.root()) {
        "'<remote NWB locator>'".into()
    } else {
        shell_quote(&sorrel_io::fsutil::display_path(open.root()).to_string())
    };
    let mut flag = |name, value: Option<String>| {
        if let Some(value) = value {
            let sep = if value.starts_with('-') { "=" } else { " " };
            args.push_str(&format!(" --{name}{sep}{}", shell_quote(&value)));
        }
    };
    for (name, path) in [
        ("dat", open.dat),
        ("journal", open.journal),
        ("spikeglx-meta", open.spikeglx_meta),
        ("oebin", open.oebin),
        ("events", open.events),
    ] {
        flag(
            name,
            path.map(|p| {
                if is_remote(&p) {
                    "<remote NWB locator>".into()
                } else {
                    sorrel_io::fsutil::display_path(&p).to_string()
                }
            }),
        );
    }
    #[cfg(any(feature = "hdf5", feature = "zarr"))]
    flag("nwb-series", open.nwb_series);
    flag("sample-rate", open.sample_rate.map(|r| r.to_string()));
    flag("channels", open.n_channels.map(|n| n.to_string()));
    flag("dtype", open.dtype.map(|d| d.phy_name().into()));
    flag("offset", open.offset.map(|n| n.to_string()));
    flag("sorter", open.sorter);
    flag(
        "backend",
        open.backend.map(|b| {
            match b {
                Backend::Kilosort => "kilosort",
                Backend::SortingAnalyzer => "sorting-analyzer",
                #[cfg(any(feature = "hdf5", feature = "zarr"))]
                Backend::Nwb => "nwb",
                #[cfg(feature = "hdf5")]
                Backend::Ks4Rez => "ks4-rez",
            }
            .into()
        }),
    );
    Ok(args)
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

fn diff_dirs(
    truth: &Path,
    test: &Path,
    root: &Path,
    opts: &sorrel_data::diff::SpikeDiffOpts,
    json: bool,
) -> Result<sorrel_data::diff::SpikeDiff> {
    let d = sorrel_data::diff::diff_dirs(truth, test, root, opts)?;
    if json {
        serde_json::to_writer_pretty(std::io::stdout().lock(), &d)?;
        println!();
    } else {
        print!("{}", d.render());
    }
    Ok(d)
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
    let dataset_dir = if root.is_file() || sorrel_io::nwb::is_zarr_path(root) {
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
    session_key: Option<String>,
    registry_key: Option<String>,
) -> SorrelApp<P>
where
    P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
{
    // Seeding and replay run on a worker behind the loading screen.
    let mut app = SorrelApp::with_loader(session, phy_label_str, seed);
    if let Some(key) = session_key {
        app.set_session_key(key);
    }
    if let Some(key) = registry_key {
        app.set_mcp_registry_key(key);
    }
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
        let out = if ctx.remote {
            dir.join("phy")
        } else {
            dir.with_file_name(format!("{name}_phy"))
        };
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

fn implicit_nwb_trials(trials: Result<Events>) -> Option<Events> {
    match trials {
        Ok(events) => Some(events),
        Err(e) if e.is::<sorrel_io::nwb::NoTrials>() => None,
        Err(e) => {
            log::warn!("ignoring NWB trials: {e:#}");
            None
        }
    }
}

/// `--events`, else (for an NWB file without sidecar events) its trials table.
fn load_events(
    open: &OpenArgs,
    nwb_start_time: f64,
    remote_trials: Option<Events>,
) -> Result<Option<Events>> {
    let is_nwb = |p: &Path| {
        p.extension().is_some_and(|e| e.eq_ignore_ascii_case("nwb"))
            || sorrel_io::nwb::is_zarr_path(p)
    };
    let mut events = match &open.events {
        Some(p) => {
            Some(Events::read(p).with_context(|| format!("reading events {}", p.display()))?)
        }
        None if is_remote(open.root()) => remote_trials,
        // Sidecar events next to an NWB file win (the app detects them);
        // otherwise its trials table, whose absence is normal.
        None if is_nwb(open.root())
            && matches!(
                Events::detect(open.root().parent().unwrap_or(Path::new("."))),
                Ok(None)
            ) =>
        {
            implicit_nwb_trials(Events::read(open.root()))
        }
        None => None,
    };
    if let Some(ev) = events.as_mut().filter(|_| {
        open.events
            .as_deref()
            .map_or_else(|| is_remote(open.root()) || is_nwb(open.root()), is_nwb)
    }) {
        // A trial before the trace starts must not be clamped to sample zero.
        let first = ev.times_s.partition_point(|&t| t < nwb_start_time);
        ev.times_s.drain(..first);
        ev.label.drain(..first);
        ev.times_s.iter_mut().for_each(|t| *t -= nwb_start_time);
    }
    Ok(events)
}

fn tab_target<'a>(
    open: &OpenArgs,
    key: Option<&str>,
    tabs: impl Iterator<Item = (&'a Path, Option<&'a str>)>,
) -> Result<Option<usize>> {
    let Some(key) = key else { return Ok(None) };
    let dir = session_path(&host_dir(open));
    for (i, (dataset, tab_key)) in tabs.enumerate() {
        if session_path(dataset) == dir {
            sorrel_data::serve::check_session_reuse(
                "open tab",
                tab_key,
                key,
                open.discard_journal,
            )?;
            return Ok(Some(i));
        }
    }
    Ok(None)
}

/// The shell's opener: a forwarded `sorrel open`'s args become a tab, or the
/// index of the tab that already shows the dataset.
fn tab_opener() -> Opener {
    Box::new(|req, open_tabs, rs, refractory| {
        let args = std::iter::once("sorrel").chain(req.args.iter().map(String::as_str));
        let mut open = TabCli::try_parse_from(args)?.open;
        open.rebase(&req.cwd);
        // Recents keep the user's locator, without any signed query.
        let display = if is_remote(open.root()) {
            PathBuf::from(open.root().to_str().unwrap().split('?').next().unwrap())
        } else {
            open.root().to_path_buf()
        };
        let tabs = open_tabs
            .iter()
            .map(|tab| (tab.dataset(), tab.session_key()));
        let key = session_key(&open)?;
        if let Some(i) = tab_target(&open, key.as_deref(), tabs)? {
            if open.refractory.overrides().into_iter().any(|flag| flag) {
                let configured = open.refractory.configure_host(|request| {
                    sorrel_data::serve::call_once_for_session(
                        &host_dir(&open),
                        key.as_deref(),
                        open.discard_journal,
                        request,
                    )
                })?;
                anyhow::ensure!(configured, "The session is already open and its MCP listener is not ready; retry once it has loaded");
            }
            return Ok(Opened::Existing(i));
        }
        let mut tab = None;
        with_session_refractory(
            &open,
            refractory,
            TabBuilder {
                remote_trials: None,
                out: &mut tab,
                open: &open,
                rs,
                session_key: key,
            },
        )?;
        let tab = tab.context("backend produced no tab")?;
        Ok(Opened::Tab(tab, display))
    })
}

/// Whether winit will pick its Wayland backend. winit 0.30 decides from the
/// environment alone (no `WINIT_UNIX_BACKEND` override since 0.29): Wayland
/// whenever a non-empty `WAYLAND_DISPLAY` or `WAYLAND_SOCKET` is set.
fn wayland_session() -> bool {
    let set = |k| std::env::var_os(k).is_some_and(|v| !v.is_empty());
    cfg!(target_os = "linux") && (set("WAYLAND_DISPLAY") || set("WAYLAND_SOCKET"))
}

/// The 256 px app icon (copy of `assets/icon/sorrel-256.png`, kept in the crate
/// so `cargo package` is self-contained).
fn window_icon() -> egui::IconData {
    eframe::icon_data::from_png_bytes(include_bytes!("../assets/sorrel-256.png"))
        .expect("embedded icon is a valid PNG")
}

/// The window. `register` makes it the instance `sorrel open` hands datasets
/// to; `first` is opened at startup.
fn run_shell(opener: Opener, first: Option<OpenRequest>, register: bool) -> Result<()> {
    let mut wgpu_options = eframe::egui_wgpu::WgpuConfiguration::default();
    if let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut wgpu_options.wgpu_setup {
        setup.instance_descriptor.flags = sorrel_gpu::instance_flags();
    }
    // Wayland sends no frame callbacks to a covered surface, so a vsync
    // (FIFO) present on the main window blocks the event loop while e.g. a
    // full-screen pop-out covers it, freezing every window. Mailbox (or
    // immediate where there is none) never waits; egui still only repaints
    // on input, so this costs no idle frames.
    // ponytail: AutoNoVsync falls back to FIFO on a surface that offers
    // nothing else. Mesa's Wayland WSI (RADV, ANV, radeonsi, lavapipe…)
    // offers immediate and mailbox, so only a FIFO-only driver still stalls;
    // skip painting a covered main window if one turns up.
    if wayland_session() {
        wgpu_options.present_mode = eframe::egui_wgpu::wgpu::PresentMode::AutoNoVsync;
    }
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_app_id("sorrel")
            .with_icon(window_icon()),
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
        is_remote(open.root()) || open.root().exists(),
        "{} does not exist",
        sorrel_io::fsutil::display_path(open.root())
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
        None => {
            anyhow::ensure!(
                !open.refractory.overrides().into_iter().any(|flag| flag),
                "Refractory launch flags require a dataset"
            );
            None
        }
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

/// The host `sorrel mcp` (no DATA_DIR) serves: the GUI's active tab, else
/// the only live host. A save directory alone may name several remote hosts.
fn mcp_target(
    gui: Option<&serde_json::Value>,
    hosts: impl FnOnce() -> Vec<sorrel_data::serve::HostRecord>,
) -> Result<(PathBuf, Option<String>)> {
    let active = gui
        .and_then(|v| v["tabs"].as_array())
        .and_then(|t| t.iter().find(|t| t["active"] == true));
    if let Some((tab, dir)) = active.and_then(|t| Some((t, t["dataset"].as_str()?))) {
        return Ok((
            dir.into(),
            tab["mcp_registry_key"].as_str().map(str::to_owned),
        ));
    }
    let alive: Vec<_> = hosts().into_iter().filter(|h| h.alive).collect();
    match alive.as_slice() {
        [] => bail!("no running sorrel host; pass DATA_DIR"),
        [host] => Ok((host.dataset.clone(), host.registry_key.clone())),
        _ => bail!(
            "several sorrel hosts are running; select a GUI tab, stop the other hosts, \
             or pass DATA_DIR and its open options for a local session. Hosts:\n{}",
            alive
                .iter()
                .map(|h| format!(
                    "  {} ({}, pid {}, port {})",
                    h.dataset.display(),
                    h.kind,
                    h.pid,
                    h.port
                ))
                .collect::<Vec<_>>()
                .join("\n")
        ),
    }
}

/// `sorrel mcp` without DATA_DIR: bridge stdio to the target's host.
fn run_mcp_active(refractory: RefractoryArgs) -> Result<()> {
    // A hung GUI is skipped: its hosts may still appear in the registry.
    let gui = sorrel_data::send_instance(&InstanceRequest::List)
        .ok()
        .flatten();
    let (dir, key) = mcp_target(gui.as_ref(), sorrel_data::list_hosts)?;
    refractory.configure_host(|request| {
        sorrel_data::serve::call_once_registered(&dir, key.as_deref(), request)
    })?;
    if !sorrel_data::serve::attach_registered(
        &dir,
        key.as_deref(),
        std::io::stdin(),
        std::io::stdout(),
    )? {
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

fn shell_quote(s: &str) -> String {
    if !s.is_empty()
        && s.chars()
            .all(|c| c.is_alphanumeric() || "/._-+:@".contains(c))
    {
        s.to_owned()
    } else {
        format!("'{}'", s.replace('\'', r"'\''"))
    }
}

/// `sorrel mcp config`: how to register this binary with an agent client.
fn mcp_config(client: Client, dir: Option<PathBuf>) -> Result<()> {
    let exe = std::env::current_exe()?.display().to_string();
    let mut args = vec!["mcp".to_owned()];
    if let Some(d) = dir {
        args.push(std::path::absolute(d)?.display().to_string());
    }
    let json = |s: &[String]| serde_json::to_string(s).expect("strings serialise");
    match client {
        Client::Claude => println!(
            "claude mcp add sorrel -- {} {}",
            shell_quote(&exe),
            args.iter()
                .map(|a| shell_quote(a))
                .collect::<Vec<_>>()
                .join(" ")
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
    let running_reply = if let Some(key) = session_key(open)? {
        let dir = host_dir(open);
        let call = |request: &serde_json::Value| {
            sorrel_data::serve::call_once_for_session(
                &dir,
                Some(&key),
                open.discard_journal,
                request,
            )
        };
        open.refractory.configure_host(call)?;
        call(&request)?
    } else {
        None
    };
    let reply = match running_reply {
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
    let want = dir
        .map(|d| {
            session_host_dir(&OpenArgs {
                root: Some(d.into()),
                ..Default::default()
            })
            .map(|dir| canon(&dir))
        })
        .transpose()?;
    // `list_hosts` paths carry no `\\?\` prefix, so compare without it.
    let canon = |p: &Path| sorrel_io::fsutil::simplified(&canon(p)).to_path_buf();
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

/// `sorrel install-shortcut`: `%APPDATA%\...\Start Menu\Programs\Sorrel.lnk`
/// -> the sibling `sorrelw.exe`, icon included. For winget and zip installs;
/// the Chocolatey package makes its own.
fn install_shortcut(remove: bool) -> Result<()> {
    anyhow::ensure!(cfg!(windows), "install-shortcut is Windows only");
    let appdata = std::env::var_os("APPDATA").context("APPDATA is not set")?;
    let lnk = PathBuf::from(appdata).join(r"Microsoft\Windows\Start Menu\Programs\Sorrel.lnk");
    if remove {
        match std::fs::remove_file(&lnk) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                return Err(e).with_context(|| format!("remove {}", lnk.display()))
            }
            _ => println!("removed {}", lnk.display()),
        }
        return Ok(());
    }
    let target = std::env::current_exe()?.with_file_name("sorrelw.exe");
    anyhow::ensure!(target.exists(), "{} does not exist", target.display());
    // Paths go through the environment, never into the script text.
    let script = "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:SORREL_LNK); \
        $s.TargetPath = $env:SORREL_TARGET; $s.IconLocation = $env:SORREL_TARGET + ',0'; \
        $s.Description = 'Sorrel spike-sorting curation'; $s.Save()";
    std::fs::create_dir_all(lnk.parent().context("shortcut path has no parent")?)?;
    let status = std::process::Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", script])
        .env("SORREL_LNK", &lnk)
        .env("SORREL_TARGET", &target)
        .status()
        .context("run powershell")?;
    anyhow::ensure!(
        status.success() && lnk.exists(),
        "powershell could not create {}",
        lnk.display()
    );
    println!("created {} -> {}", lnk.display(), target.display());
    Ok(())
}

fn main() -> Result<()> {
    let mut logger =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"));
    // Signed URLs are credentials. A more specific RUST_LOG target would
    // override a crate-level filter, so disable those directives too.
    let transport = ["ureq", "ureq_proto", "rustls"];
    let filter = std::env::var("RUST_LOG").unwrap_or_default();
    let targets = filter
        .split('/')
        .next()
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().split('=').next().unwrap_or_default())
        .filter(|target| transport.iter().any(|prefix| target.starts_with(prefix)));
    for target in transport.into_iter().chain(targets) {
        logger.filter_module(target, log::LevelFilter::Off);
    }
    logger.init();
    let cli = Cli::parse();
    match cli.cmd {
        None => run_bare(),
        Some(Cmd::Open(open)) => run_open(&open),
        Some(Cmd::Run {
            open,
            standalone,
            headless,
        }) => match headless {
            true => with_session(&open, Headless(&open, session_key(&open)?)),
            false => run_gui(&open, standalone),
        },
        Some(Cmd::Status { dir, json }) => run_status(dir.as_deref(), json),
        Some(Cmd::Call {
            open,
            tool,
            args,
            raw,
        }) => run_call(&open, &tool, args.as_deref(), raw),
        Some(Cmd::AgentGuide { prompt }) => {
            match prompt {
                Some(_) => print!("{}", serve::review_prompt()?),
                None => print!("{}", agent_guide()),
            }
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
        Some(Cmd::InstallShortcut { remove }) => install_shortcut(remove),
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
            let out = if is_remote(open.root()) {
                anyhow::ensure!(reuse, "sorrel auto supports Kilosort/phy directories unless --reuse is used");
                session_host_dir(&open)?.join("sorrel_auto")
            } else {
                open.root().join("sorrel_auto")
            };
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
        Some(Cmd::Diff {
            a,
            b,
            json,
            root,
            match_jaccard,
            min_spikes,
            split_share,
            binary,
            no_binary,
            disputed,
            #[cfg(feature = "dev")]
            record,
            #[cfg(feature = "dev")]
            db,
        }) => match (root, a.is_dir() && b.is_dir()) {
            (Some(root), true) => {
                let opts = sorrel_data::diff::SpikeDiffOpts {
                    match_jaccard,
                    min_spikes,
                    split_share,
                    binary: match (binary, no_binary) {
                        (true, _) => Some(true),
                        (_, true) => Some(false),
                        _ => None,
                    },
                    disputed: disputed
                        .map(|p| sorrel_data::diff::read_disputed(&p))
                        .transpose()?
                        .unwrap_or_default(),
                };
                let _score = diff_dirs(&a, &b, &root, &opts, json)?;
                #[cfg(feature = "dev")]
                if let Some(rid) = record {
                    dev::record_diff(db.as_deref(), &rid, &_score, (&b, &a, &root), &opts.disputed)?;
                }
                Ok(())
            }
            #[cfg(feature = "dev")]
            (None, false) if record.is_some() => {
                bail!("sorrel diff: --record scores a curation; it needs two directories and --root DIR")
            }
            (None, false) if disputed.is_none() => diff_tsvs(&a, &b, json),
            _ => bail!("sorrel diff: two directories need --root DIR; two files take no --root or --disputed"),
        },
        Some(Cmd::Blind {
            open,
            input,
            truth,
            phy_curated: true,
        }) => {
            let dir = open
                .root
                .as_deref()
                .context("sorrel blind --phy-curated needs DATA_DIR")?;
            let files =
                sorrel_data::write_phy_curated_pair(dir, open.dat.as_deref(), &input, &truth)?;
            println!(
                "blind input {} ({} files), truth {} (phy-curated)",
                input.display(),
                files.len(),
                truth.display()
            );
            Ok(())
        }
        Some(Cmd::Blind {
            open, input, truth, ..
        }) => with_session(
            &open,
            Blind {
                dat: open.dat.clone(),
                input,
                truth,
            },
        ),
        Some(Cmd::Mcp { open, sub }) => match (sub, open.root.is_some()) {
            (Some(McpCmd::Tools { json }), _) => mcp_tools(json),
            (Some(McpCmd::Config { client, dir }), _) => mcp_config(client, dir),
            (None, true) => run_mcp(&open),
            (None, false) => run_mcp_active(open.refractory),
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
        #[cfg(feature = "hdf5")]
        Some(Cmd::ExportNwb(args)) => args.run(),
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
        #[cfg(feature = "dev")]
        Some(Cmd::Dev(args)) => dev::run(args),
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    /// The crate-local icon decodes, and still equals the canonical asset
    /// (absent in a packaged crate).
    #[test]
    fn window_icon_is_the_app_icon() {
        let icon = window_icon();
        assert_eq!((icon.width, icon.height), (256, 256));
        let canonical =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/icon/sorrel-256.png");
        if let Ok(bytes) = std::fs::read(canonical) {
            assert_eq!(&bytes[..], include_bytes!("../assets/sorrel-256.png"));
        }
    }

    #[test]
    #[cfg(not(windows))]
    fn install_shortcut_is_windows_only() {
        let e = install_shortcut(false).unwrap_err();
        assert!(e.to_string().contains("Windows only"), "{e}");
    }

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
            Some(Cmd::AgentGuide { prompt: None })
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
                assert_eq!(out, Some(PathBuf::from("o")));
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

    /// A host keys its journal before creating it; `call` keys it after.
    /// (On Windows every canonical path differs: it gains a `\\?\` prefix.)
    #[cfg(unix)]
    #[test]
    fn session_path_is_stable_when_the_file_appears() {
        let dir = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(dir.path(), dir.path().join("link")).unwrap();
        let journal = dir.path().join("link/first.journal");
        let before = session_path(&journal);
        std::fs::write(&journal, b"").unwrap();
        assert_eq!(before, session_path(&journal));
    }

    #[test]
    fn refractory_flags_override_only_supplied_values() {
        let Some(Cmd::Qc { open, .. }) = parse(&[
            "qc",
            "/x",
            "--out",
            "o",
            "--refractory-ms",
            "2",
            "--rp-min-ms",
            "0.75",
            "--rp-step-ms",
            "0.25",
        ])
        .unwrap()
        .cmd
        else {
            panic!("expected qc");
        };
        let mut base = sorrel_compute::RefractoryParams {
            sweep_max_ms: 5.0,
            ..Default::default()
        };
        let params = open.refractory.resolve(base).unwrap();
        assert_eq!(
            (
                params.fixed_ms,
                params.sweep_min_ms,
                params.sweep_max_ms,
                params.sweep_step_ms
            ),
            (2.0, 0.75, 5.0, 0.25)
        );
        base.sweep_max_ms = 0.75;
        assert!(open
            .refractory
            .resolve(base)
            .unwrap_err()
            .to_string()
            .contains("sweep_min_ms"));
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
    #[cfg(feature = "dandi")]
    fn remote_session_paths_survive_rebasing() -> Result<()> {
        let locator = "dandi://dandi/000123@draft/test.nwb";
        let url = "https://example.org/raw.nwb?signature=not-for-the-registry";
        let mut open = OpenArgs {
            root: Some(locator.into()),
            dat: Some(url.into()),
            journal: Some("override.journal".into()),
            ..Default::default()
        };
        let cwd = std::env::current_dir()?;
        open.rebase(&cwd);
        assert_eq!(open.root(), Path::new(locator));
        assert_eq!(open.dat.as_deref(), Some(Path::new(url)));
        assert_eq!(open.journal, Some(cwd.join("override.journal")));
        assert_eq!(session_key(&open)?, None);
        assert!(!mcp_open_args(&open)?.contains("signature="));
        let hash = blake3::hash(b"test.nwb").to_hex();
        assert_eq!(
            session_host_dir(&open)?,
            dandi_data_dir()?.join("000123/draft").join(hash.as_str())
        );
        Ok(())
    }

    #[test]
    #[cfg(feature = "dandi")]
    fn bare_mcp_distinguishes_remote_sessions_with_the_same_sorting() -> Result<()> {
        let mut open = OpenArgs {
            root: Some("dandi://dandi/000123@draft/sorting.nwb".into()),
            dat: Some("https://example.org/first.nwb?signature=secret".into()),
            ..Default::default()
        };
        let first = remote_registry_key(&open)?.unwrap();
        assert!(!first.contains("signature="));
        open.dat = Some("https://example.org/second.nwb?signature=secret".into());
        let second = remote_registry_key(&open)?.unwrap();
        assert_ne!(first, second);
        open.events = Some("events.csv".into());
        assert_ne!(
            remote_registry_key(&open)?.as_deref(),
            Some(second.as_str())
        );

        let dataset = PathBuf::from("save-dir");
        let host = |port, key: &str| sorrel_data::serve::HostRecord {
            dataset: dataset.clone(),
            registry_key: Some(key.to_owned()),
            port,
            pid: 123,
            kind: "headless".into(),
            alive: true,
        };
        assert_eq!(
            mcp_target(None, || vec![host(101, &first)])?,
            (dataset.clone(), Some(first.clone()))
        );
        let both = || vec![host(101, &first), host(102, &second)];
        let err = mcp_target(None, both).unwrap_err().to_string();
        assert!(
            err.contains("port 101") && err.contains("port 102"),
            "{err}"
        );
        let gui = serde_json::json!({"tabs": [
            {"active": false, "dataset": dataset, "mcp_registry_key": first},
            {"active": true, "dataset": dataset, "mcp_registry_key": second},
        ]});
        assert_eq!(mcp_target(Some(&gui), both)?, (dataset, Some(second)));
        Ok(())
    }

    #[test]
    fn only_local_requests_reuse_a_tab_or_host() -> Result<()> {
        let local = tempfile::tempdir()?;
        let mut open = OpenArgs {
            root: Some(local.path().join("sorting.nwb")),
            dat: Some(local.path().join("recording.nwb")),
            ..Default::default()
        };
        let key = session_key(&open)?.unwrap();
        let tabs = || [(local.path(), Some(key.as_str()))].into_iter();
        assert_eq!(tab_target(&open, Some(&key), tabs())?, Some(0));
        open.refractory.refractory_ms = Some(2.0);
        assert_eq!(session_key(&open)?.as_deref(), Some(key.as_str()));
        assert_eq!(
            tab_target(&open, session_key(&open)?.as_deref(), tabs())?,
            Some(0)
        );
        open.dat = Some("https://example.org/raw.nwb".into());
        assert_eq!(session_key(&open)?, None);
        assert_eq!(
            tab_target(&open, session_key(&open)?.as_deref(), tabs())?,
            None
        );
        open.root = Some("dandi://dandi/000123/test/sorting.nwb".into());
        open.dat = None;
        assert_eq!(session_key(&open)?, None);
        assert_eq!(
            tab_target(&open, session_key(&open)?.as_deref(), tabs())?,
            None
        );
        Ok(())
    }

    #[test]
    #[cfg(feature = "dandi")]
    fn dandi_reopen_refuses_a_journal_held_by_another_session() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let mut open = OpenArgs {
            root: Some("dandi://dandi/000123/test/sorting.nwb".into()),
            ..Default::default()
        };
        let path = journal_path(&open, dir.path(), "nwb.sorrel");
        let owner = open_journal(&open, &path, 23, 1)?;
        open.discard_journal = true;
        let err = open_journal(&open, &path, 23, 1).err().unwrap().to_string();
        assert!(err.contains("already open"), "{err}");
        assert!(
            err.contains("stop the running session or pass a different --journal"),
            "{err}"
        );
        open.journal = Some(dir.path().join("another.journal"));
        let _other = open_journal(&open, &journal_path(&open, dir.path(), "nwb.sorrel"), 23, 1)?;
        drop(owner);
        Ok(())
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

    #[cfg(feature = "hdf5")]
    #[test]
    fn nwb_flags_and_trial_times() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../release/fixtures/nwb/units_only.nwb"
        );
        let Some(Cmd::Open(open)) = Cli::try_parse_from([
            "sorrel",
            "open",
            path,
            "--dat",
            "raw.nwb",
            "--nwb-series",
            "ElectricalSeriesRaw",
            "--sample-rate",
            "1000",
            "--events",
            path,
        ])
        .unwrap()
        .cmd
        else {
            panic!("expected open")
        };
        assert_eq!(open.nwb_series.as_deref(), Some("ElectricalSeriesRaw"));
        assert_eq!(open.dat.as_deref(), Some(Path::new("raw.nwb")));
        assert_eq!(open.sample_rate, Some(1000.0));
        let events = load_events(&open, 0.5, None).unwrap().unwrap();
        assert_eq!(events.samples(1000.0, None), [sorrel_io::SampleIndex(100)]);
        assert!(load_events(&open, 0.61, None).unwrap().unwrap().is_empty());
    }

    #[cfg(feature = "zarr")]
    #[test]
    fn nwb_zarr_detects_suffix_and_nwb_version() {
        let dir = tempfile::tempdir().unwrap();
        let named = dir.path().join("recording.nwb.zarr");
        let unnamed = dir.path().join("download");
        std::fs::create_dir(&named).unwrap();
        std::fs::create_dir(&unnamed).unwrap();
        assert!(detect_backend(&unnamed).is_err());
        std::fs::write(unnamed.join(".zattrs"), br#"{"nwb_version":"2.10.0"}"#).unwrap();
        for root in [named, unnamed] {
            assert_eq!(detect_backend(&root).unwrap(), Backend::Nwb);
            let open = OpenArgs {
                root: Some(root),
                ..Default::default()
            };
            assert_eq!(host_dir(&open), dir.path());
        }
    }

    #[cfg(feature = "zarr")]
    #[test]
    fn nwb_zarr_trial_times_are_relative_to_the_trace() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../release/fixtures/nwb/units_only.nwb.zarr");
        if !path.is_dir() {
            return;
        } // Cargo package builds omit workspace fixtures.
        let open = OpenArgs {
            root: Some(path),
            ..Default::default()
        };
        let events = load_events(&open, 0.5, None).unwrap().unwrap();
        assert_eq!(events.samples(1000.0, None), [sorrel_io::SampleIndex(100)]);
    }

    #[cfg(feature = "dandi")]
    #[test]
    fn remote_nwb_trials_align_and_explicit_events_win() -> Result<()> {
        let mut open = OpenArgs {
            root: Some("https://example.test/asset/download/".into()),
            ..Default::default()
        };
        let trials = Events {
            times_s: vec![0.4, 0.6],
            label: vec![0, 1],
            names: vec!["pre-trace".into(), "remote trial".into()],
        };
        let events = load_events(&open, 0.5, Some(trials.clone()))?.unwrap();
        assert_eq!(events.samples(1000.0, None), [sorrel_io::SampleIndex(100)]);
        assert_eq!(events.names[events.label[0] as usize], "remote trial");

        let dir = tempfile::tempdir()?;
        let path = dir.path().join("events.csv");
        std::fs::write(&path, "time,label\n0.2,explicit\n")?;
        open.events = Some(path);
        let events = load_events(&open, 0.5, Some(trials))?.unwrap();
        assert_eq!(events.samples(1000.0, None), [sorrel_io::SampleIndex(200)]);
        assert_eq!(events.names[events.label[0] as usize], "explicit");
        Ok(())
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
