use super::{
    detect_backend, seeded, with_session, Backend, OpenArgs, OpenCtx, Seed, SessionVisitor,
};
use anyhow::{ensure, Context, Result};
use clap::Args;
use sorrel_data::{nwb_export::collect_units, session::ApplyPhyLabel, Session};
use sorrel_io::{kilosort::PhyLabel, nwb_write, DataProvider, HasGeometry};
use std::path::PathBuf;

#[derive(Args, Debug)]
#[command(group(clap::ArgGroup::new("destination").required(true).args(["out", "into", "in_place"])))]
pub struct ExportNwb {
    #[command(flatten)]
    open: OpenArgs,
    /// Create a new NWB file (will not overwrite)
    #[arg(long, value_name = "FILE")]
    out: Option<PathBuf>,
    /// Add /units to an existing NWB file (must not already have /units)
    #[arg(long, value_name = "FILE")]
    into: Option<PathBuf>,
    /// Use the input NWB file as the append destination
    #[arg(long)]
    in_place: bool,
    /// NWB identifier; required for new files from non-NWB sources
    #[arg(long, requires = "out")]
    identifier: Option<String>,
    /// Session description; required for new files from non-NWB sources
    #[arg(long, requires = "out")]
    session_description: Option<String>,
    /// ISO 8601 datetime with timezone; required for new files from non-NWB sources
    #[arg(long, requires = "out")]
    session_start_time: Option<String>,
    /// First sample's time on the NWB time axis, in seconds (default: source NWB offset, or 0)
    #[arg(long, allow_negative_numbers = true)]
    starting_time: Option<f64>,
}

impl ExportNwb {
    pub fn run(self) -> Result<()> {
        let backend = match self.open.backend {
            Some(b) => b,
            None => detect_backend(self.open.root())?,
        };
        let source = (backend == Backend::Nwb).then(|| self.open.root().to_path_buf());
        let destination = if let Some(out) = self.out {
            let metadata = if let Some(source) = &source {
                ensure!(self.identifier.is_none() && self.session_description.is_none() && self.session_start_time.is_none(),
                    "metadata flags are only for non-NWB inputs; these fields are copied from the source NWB");
                nwb_write::Metadata::read(source).context("reading source NWB session metadata")?
            } else {
                nwb_write::Metadata::new(
                    self.identifier
                        .context("--out requires --identifier for a non-NWB source")?,
                    self.session_description
                        .context("--out requires --session-description for a non-NWB source")?,
                    self.session_start_time
                        .context("--out requires --session-start-time for a non-NWB source")?,
                )?
            };
            Destination::New(out, metadata)
        } else {
            Destination::Append(match self.into {
                Some(path) => path,
                None => source
                    .clone()
                    .context("--in-place requires an NWB source")?,
            })
        };
        with_session(
            &self.open,
            Export {
                source,
                destination,
                starting_time: self.starting_time,
            },
        )
    }
}

enum Destination {
    New(PathBuf, nwb_write::Metadata),
    Append(PathBuf),
}

struct Export {
    source: Option<PathBuf>,
    destination: Destination,
    starting_time: Option<f64>,
}

impl SessionVisitor for Export {
    fn visit<P>(self, session: Session<P>, seed: Seed<P>, _: OpenCtx<P>) -> Result<()>
    where
        P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
    {
        let session = seeded(session, seed)?;
        let p = &session.provider;
        let (rate, start) = if let Some(source) = self.source {
            nwb_write::source_time_base(
                &source,
                p.n_samples().idx(),
                p.n_channels() as usize,
                p.sample_rate(),
            )?
        } else {
            (p.exact_sample_rate()?, 0.0)
        };
        let units = collect_units(&session, rate, self.starting_time.unwrap_or(start))?;
        let (path, result) = match self.destination {
            Destination::New(path, metadata) => {
                let r = nwb_write::write_new(
                    &path,
                    &metadata,
                    p.channel_positions(),
                    p.channel_map(),
                    &units,
                );
                (path, r)
            }
            Destination::Append(path) => {
                let r = nwb_write::append(&path, &units);
                (path, r)
            }
        };
        result.with_context(|| format!("exporting NWB to {}", path.display()))?;
        println!("exported {} units into {}", units.id.len(), path.display());
        Ok(())
    }
}
