use super::{
    detect_backend, is_remote, seeded, with_session, Backend, OpenArgs, OpenCtx, Seed,
    SessionVisitor,
};
use anyhow::{ensure, Context, Result};
use clap::Args;
use sorrel_data::{nwb_export::collect_units, session::ApplyPhyLabel, Session};
use sorrel_io::{kilosort::PhyLabel, nwb_write, DataProvider, HasGeometry};
use std::path::{Path, PathBuf};

#[derive(Args, Debug)]
#[command(group(clap::ArgGroup::new("destination").required(true).args(["out", "into"])))]
pub struct ExportNwb {
    #[command(flatten)]
    open: OpenArgs,
    /// Create a new NWB file (will not overwrite)
    #[arg(long, value_name = "FILE")]
    out: Option<PathBuf>,
    /// Add /units to an existing NWB file (must not already have /units)
    #[arg(long, value_name = "FILE")]
    into: Option<PathBuf>,
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
        let source = (backend == Backend::Nwb).then(|| self.open.root());
        if let Some(journal) = &self.open.journal {
            let raw = self.open.dat.as_deref().filter(|_| source.is_some());
            for path in source
                .into_iter()
                .chain(raw)
                .filter(|p| !is_remote(p))
                .chain(self.out.as_deref())
                .chain(self.into.as_deref())
            {
                reject_journal_alias(journal, path)?;
            }
        }
        let destination = if let Some(out) = self.out {
            let metadata = if source.is_some() {
                ensure!(self.identifier.is_none() && self.session_description.is_none() && self.session_start_time.is_none(),
                    "metadata flags are only for non-NWB inputs; these fields are copied from the source NWB");
                None
            } else {
                Some(nwb_write::Metadata::new(
                    self.identifier
                        .context("--out requires --identifier for a non-NWB source")?,
                    self.session_description
                        .context("--out requires --session-description for a non-NWB source")?,
                    self.session_start_time
                        .context("--out requires --session-start-time for a non-NWB source")?,
                )?)
            };
            Destination::New(out, metadata)
        } else {
            let into = self.into.context("expected --out or --into")?;
            Destination::Append(into)
        };
        with_session(
            &self.open,
            Export {
                destination,
                starting_time: self.starting_time,
                preserve_metrics: backend == Backend::Nwb,
            },
        )
    }
}

fn reject_journal_alias(journal: &Path, nwb: &Path) -> Result<()> {
    fn canonical(path: &Path) -> Result<(PathBuf, bool)> {
        match path.canonicalize() {
            Ok(path) => Ok((path, true)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // A dangling symlink isn't a new filename: opening it may create its target.
                ensure!(!path.is_symlink(), "dangling symlink {}", path.display());
                let name = path.file_name().context("missing file name")?;
                let parent = path.parent().filter(|p| !p.as_os_str().is_empty());
                Ok((
                    parent.unwrap_or(Path::new(".")).canonicalize()?.join(name),
                    false,
                ))
            }
            Err(e) => Err(e.into()),
        }
    }
    let check = || -> Result<bool> {
        let (journal, j_exists) = canonical(journal)?;
        let (nwb, n_exists) = canonical(nwb)?;
        Ok(journal == nwb || (j_exists && n_exists && same_file::is_same_file(journal, nwb)?))
    };
    ensure!(
        !check()
            .with_context(|| format!("checking whether --journal aliases {}", nwb.display()))?,
        "--journal {} aliases NWB file {}",
        journal.display(),
        nwb.display()
    );
    Ok(())
}

enum Destination {
    New(PathBuf, Option<nwb_write::Metadata>),
    Append(PathBuf),
}

struct Export {
    destination: Destination,
    starting_time: Option<f64>,
    preserve_metrics: bool,
}

impl SessionVisitor for Export {
    fn prepare_nwb(&mut self, provider: &sorrel_io::NwbProvider) -> Result<()> {
        match &mut self.destination {
            Destination::New(_, metadata) => {
                *metadata = Some(
                    provider
                        .export_metadata()
                        .context("reading source NWB session metadata")?,
                );
            }
            Destination::Append(path) if self.starting_time.is_none() => {
                let reference = provider
                    .export_timestamp_reference()
                    .context("reading source NWB timestamps_reference_time")?;
                nwb_write::check_timestamp_reference(&reference, path)?;
            }
            Destination::Append(_) => {}
        }
        Ok(())
    }

    fn visit<P>(self, session: Session<P>, seed: Seed<P>, ctx: OpenCtx<P>) -> Result<()>
    where
        P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
    {
        let session = seeded(session, seed)?;
        let p = &session.provider;
        let rate = p.exact_sample_rate()?;
        let units = collect_units(
            &session,
            rate,
            self.starting_time.unwrap_or(ctx.nwb_start_time),
            self.preserve_metrics.then_some(&ctx.metrics),
        )?;
        let (path, result) = match self.destination {
            Destination::New(path, metadata) => {
                let r = nwb_write::write_new(
                    &path,
                    &metadata.context("missing source NWB session metadata")?,
                    p.channel_positions(),
                    p.channel_map(),
                    &units,
                );
                (path, r)
            }
            Destination::Append(path) => {
                // The destination may also be --dat. Close that read handle
                // before replacing the file, which Windows otherwise forbids.
                drop(session);
                let r = nwb_write::append(&path, &units);
                (path, r)
            }
        };
        result.with_context(|| format!("exporting NWB to {}", path.display()))?;
        println!("exported {} units into {}", units.id.len(), path.display());
        Ok(())
    }
}
