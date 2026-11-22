//! Native HDF5 serialization of curated units. Layout mirrors the cached
//! schemas in `nwb_schema` and https://nwb-schema.readthedocs.io/en/stable/format.html.

use crate::nwb::{check_append_rate, read_str, strings};
use crate::nwb_units::{Units, Values};
use crate::ChannelId;
use anyhow::{ensure, Context, Result};
use hdf5_metno::types::VarLenUnicode;
use hdf5_metno::{Dataset, File, Group, H5Type, Location, ObjectReference1};
use std::fs;
use std::path::Path;
use tempfile::NamedTempFile;

const NWB_VERSION: &str = "2.11.0";
const HDMF: &str = "hdmf-common";
const ELECTROPHYS: &str = "/general/extracellular_ephys";

pub struct Metadata {
    pub identifier: String,
    pub session_description: String,
    pub session_start_time: String,
    pub timestamps_reference_time: String,
}

impl Metadata {
    pub fn new(
        identifier: String,
        session_description: String,
        session_start_time: String,
    ) -> Result<Self> {
        let _ = session_start_time
            .parse::<jiff::Timestamp>()
            .context("--session-start-time must be an ISO 8601 datetime with timezone")?;
        ensure!(
            !identifier.trim().is_empty() && !session_description.trim().is_empty(),
            "identifier and session description must not be empty"
        );
        Ok(Self {
            identifier,
            session_description,
            timestamps_reference_time: session_start_time.clone(),
            session_start_time,
        })
    }
}

/// Require the two NWB time axes to have the same origin for implicit alignment.
pub fn check_timestamp_reference(src: &str, destination: &Path) -> Result<()> {
    fn read(path: &Path) -> Result<String> {
        let f = File::open(path)?;
        let name = if f.link_exists("timestamps_reference_time") {
            "timestamps_reference_time"
        } else {
            "session_start_time"
        };
        read_str(&*f.dataset(name)?)
    }
    let dst = read(destination).with_context(|| {
        format!(
            "reading timestamps_reference_time in {}",
            destination.display()
        )
    })?;
    let parse = |time: &str| {
        time.parse::<jiff::Timestamp>()
            .with_context(|| format!("invalid timestamps_reference_time {time:?}"))
    };
    ensure!(
        parse(src)? == parse(&dst)?,
        "source timestamps_reference_time ({src}) differs from destination ({dst}); pass \
         --starting-time in seconds on the destination's time axis"
    );
    Ok(())
}

/// Write a new NWB file. Never overwrite an existing path.
pub fn write_new(
    path: &Path,
    metadata: &Metadata,
    positions: &[[f32; 2]],
    channels: &[ChannelId],
    units: &Units,
) -> Result<()> {
    ensure!(
        path.symlink_metadata().is_err(),
        "{} already exists",
        path.display()
    );
    let tmp = sibling_temp(path)?;
    let f = File::create(tmp.path())?;
    typed(&f, "core", "NWBFile")?;
    text_attr(&f, "nwb_version", NWB_VERSION)?;
    for name in [
        "acquisition",
        "analysis",
        "processing",
        "stimulus/presentation",
        "stimulus/templates",
        "general",
    ] {
        f.create_group(name)?;
    }
    for (name, text) in [
        ("identifier", &metadata.identifier),
        ("session_description", &metadata.session_description),
        ("session_start_time", &metadata.session_start_time),
        (
            "timestamps_reference_time",
            &metadata.timestamps_reference_time,
        ),
    ] {
        text_dataset(&f, name, text)?;
    }
    dataset(
        &f,
        "file_create_date",
        &[text(&jiff::Timestamp::now().to_string())?],
    )?;
    write_specs(&f)?;
    if !positions.is_empty() && positions.iter().flatten().any(|x| x.is_finite()) {
        write_electrodes(&f, positions, channels)?;
    }
    write_units(&f, units)?;
    add_generated_by(&f.group("general")?, units.generated_by)?;
    f.close()?;
    tmp.as_file().sync_all()?;
    tmp.persist_noclobber(path)
        .with_context(|| format!("create {}", path.display()))?;
    Ok(())
}

/// Append root `/units` on a sibling copy; the original is untouched on error.
/// The caller must not let another process edit the destination concurrently.
pub fn append(path: &Path, units: &Units) -> Result<()> {
    let path = fs::canonicalize(path)?;
    let mut original = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .with_context(|| format!("open {} for writing", path.display()))?;
    let before = original.metadata()?;
    ensure!(before.is_file(), "{} is not a regular file", path.display());
    let source = File::open(&path)?;
    ensure!(
        read_str(&*source.attr("nwb_version")?)?.starts_with("2.")
            && read_str(&*source.attr("neurodata_type")?)? == "NWBFile",
        "{} is not an NWB 2.x file",
        path.display()
    );
    ensure!(
        !source.link_exists("/units"),
        "{} already has /units; refusing to replace it",
        path.display()
    );
    check_append_rate(&source, units.resolution.recip())?;
    let mut tmp = sibling_temp(&path)?;
    std::io::copy(&mut original, tmp.as_file_mut())?;
    let f = File::open_rw(tmp.path())?;
    let general = if f.link_exists("general") {
        f.group("general")?
    } else {
        f.create_group("general")?
    };
    ensure!(
        general.loc_info()?.fileno == f.loc_info()?.fileno,
        "cannot update an externally linked /general"
    );
    write_units(&f, units)?;
    add_generated_by(&general, units.generated_by)?;
    let dates = f.dataset("file_create_date")?;
    ensure!(dates.ndim() == 1, "invalid /file_create_date shape");
    let mut history = strings(&dates)?
        .iter()
        .map(|s| text(s))
        .collect::<Result<Vec<_>>>()?;
    history.push(text(&jiff::Timestamp::now().to_string())?);
    f.unlink("file_create_date")?;
    dataset(&f, "file_create_date", &history)?;
    drop(dates);
    drop(general);
    f.close()?;
    tmp.as_file().set_permissions(before.permissions())?;
    tmp.as_file().sync_all()?;
    let after = fs::metadata(&path)?;
    ensure!(
        after.len() == before.len() && after.modified()? == before.modified()?,
        "{} changed during export; not replacing it",
        path.display()
    );
    source.close()?;
    drop(original);
    tmp.persist(&path)
        .with_context(|| format!("replace {}", path.display()))?;
    Ok(())
}

fn sibling_temp(path: &Path) -> Result<NamedTempFile> {
    let dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    Ok(tempfile::Builder::new()
        .prefix(".sorrel-nwb-")
        .tempfile_in(dir)?)
}

fn text(s: &str) -> Result<VarLenUnicode> {
    Ok(s.parse()?)
}

fn text_attr(loc: &Location, name: &str, value: &str) -> Result<()> {
    loc.new_attr::<VarLenUnicode>()
        .shape(())
        .create(name)?
        .write_scalar(&text(value)?)?;
    Ok(())
}

fn text_dataset(group: &Group, name: &str, value: &str) -> Result<()> {
    group
        .new_dataset::<VarLenUnicode>()
        .shape(())
        .create(name)?
        .write_scalar(&text(value)?)?;
    Ok(())
}

fn typed(loc: &Location, namespace: &str, neurodata_type: &str) -> Result<()> {
    text_attr(loc, "namespace", namespace)?;
    text_attr(loc, "neurodata_type", neurodata_type)?;
    text_attr(loc, "object_id", &uuid::Uuid::new_v4().to_string())
}

fn dataset<T: H5Type>(group: &Group, name: &str, values: &[T]) -> Result<Dataset> {
    Ok(group.new_dataset_builder().with_data(values).create(name)?)
}

fn vector<T: H5Type>(group: &Group, name: &str, doc: &str, values: &[T]) -> Result<Dataset> {
    let d = dataset(group, name, values)?;
    typed(&d, HDMF, "VectorData")?;
    text_attr(&d, "description", doc)?;
    Ok(d)
}

fn colnames(table: &Group, names: impl Iterator<Item = impl AsRef<str>>) -> Result<()> {
    let names = names
        .map(|s| text(s.as_ref()))
        .collect::<Result<Vec<_>>>()?;
    table
        .new_attr_builder()
        .with_data(&names)
        .create("colnames")?;
    Ok(())
}

fn reference_attr(from: &Location, attr: &str, group: &Group, to: &str) -> Result<()> {
    let reference = group.reference::<ObjectReference1>(to)?;
    from.new_attr::<ObjectReference1>()
        .shape(())
        .create(attr)?
        .write_scalar(&reference)?;
    Ok(())
}

fn write_units(f: &File, units: &Units) -> Result<()> {
    // Always create local datasets, including on append. The reader rejects
    // non-local storage for the datasets it uses to map spikes to samples.
    let g = f.create_group("units")?;
    typed(&g, "core", "Units")?;
    text_attr(&g, "description", &units.description)?;
    colnames(&g, units.columns.iter().map(|c| c.name.as_str()))?;
    let id = dataset(&g, "id", &units.id)?;
    typed(&id, HDMF, "ElementIdentifiers")?;
    for col in &units.columns {
        let (name, doc) = (col.name.as_str(), col.description.as_str());
        // Spike times and resolution are new native f64 objects, never
        // converted to the destination series' (possibly f32) clock type.
        ensure!(
            name != "spike_times" || matches!(&col.data, Values::Float(_)),
            "spike_times must be stored as native IEEE float64"
        );
        let ds = match &col.data {
            Values::Float(v) => vector(&g, name, doc, v)?,
            Values::Int(v) => vector(&g, name, doc, v)?,
            Values::Text(v) => vector(
                &g,
                name,
                doc,
                &v.iter().map(|s| text(s)).collect::<Result<Vec<_>>>()?,
            )?,
        };
        if name == "spike_times" {
            ds.new_attr::<f64>()
                .shape(())
                .create("resolution")?
                .write_scalar(&units.resolution)?;
        }
        if let Some(ends) = &col.index {
            let index = dataset(&g, &format!("{name}_index"), ends)?;
            typed(&index, HDMF, "VectorIndex")?;
            text_attr(
                &index,
                "description",
                &format!("Index for VectorData '{name}'"),
            )?;
            reference_attr(&index, "target", &g, name)?;
        }
    }
    Ok(())
}

fn write_electrodes(f: &File, positions: &[[f32; 2]], channels: &[ChannelId]) -> Result<()> {
    let ephys = f.create_group(ELECTROPHYS)?;
    let device = f.create_group("/general/devices/probe")?;
    typed(&device, "core", "Device")?;
    let group = ephys.create_group("probe_shank")?;
    typed(&group, "core", "ElectrodeGroup")?;
    text_attr(
        &group,
        "description",
        "Electrodes with geometry from the Sorrel session",
    )?;
    text_attr(&group, "location", "unknown")?;
    group.link_soft("/general/devices/probe", "device")?;

    let table = ephys.create_group("electrodes")?;
    typed(&table, "core", "ElectrodesTable")?;
    text_attr(
        &table,
        "description",
        "Extracellular electrodes with probe-relative geometry",
    )?;
    colnames(
        &table,
        ["location", "group", "group_name", "rel_x", "rel_y"].into_iter(),
    )?;
    let n = positions.len();
    let ids: Vec<_> = if channels.is_empty() {
        (0..n as i64).collect()
    } else {
        ensure!(
            channels.len() == n,
            "channel map and geometry lengths differ"
        );
        channels.iter().map(|c| i64::from(c.0)).collect()
    };
    let id = dataset(&table, "id", &ids)?;
    typed(&id, HDMF, "ElementIdentifiers")?;
    vector(
        &table,
        "location",
        "Anatomical location of the electrode, if known.",
        &vec![text("unknown")?; n],
    )?;
    let group_ref = ephys.reference::<ObjectReference1>("probe_shank")?;
    vector(
        &table,
        "group",
        "Reference to the ElectrodeGroup.",
        &vec![group_ref; n],
    )?;
    vector(
        &table,
        "group_name",
        "Name of the ElectrodeGroup.",
        &vec![text("probe_shank")?; n],
    )?;
    for (i, name) in ["rel_x", "rel_y"].into_iter().enumerate() {
        let axis: Vec<_> = positions.iter().map(|p| p[i]).collect();
        vector(
            &table,
            name,
            "Position relative to the electrode group, in micrometers.",
            &axis,
        )?;
    }
    Ok(())
}

fn add_generated_by(g: &Group, entry: [&str; 2]) -> Result<()> {
    let name = "was_generated_by";
    let mut values = if g.link_exists(name) {
        let ds = g.dataset(name)?;
        ensure!(
            ds.ndim() == 2 && ds.shape()[1] == 2,
            "invalid /general/{name} shape"
        );
        strings(&ds)?
    } else {
        Vec::new()
    };
    if values
        .as_chunks::<2>()
        .0
        .iter()
        .any(|[name, version]| [name.as_str(), version.as_str()] == entry)
    {
        return Ok(());
    }
    values.extend(entry.map(str::to_owned));
    let values = values.iter().map(|s| text(s)).collect::<Result<Vec<_>>>()?;
    if g.link_exists(name) {
        g.unlink(name)?;
    }
    g.new_dataset::<VarLenUnicode>()
        .shape((values.len() / 2, 2))
        .create(name)?
        .write_raw(&values)?;
    Ok(())
}

fn write_specs(f: &File) -> Result<()> {
    macro_rules! namespace {
        ($space:literal, $version:literal, $($name:literal),+ $(,)?) => {{
            let g = f.create_group(concat!("specifications/", $space, "/", $version))?;
            $(text_dataset(&g, $name, include_str!(concat!("nwb_schema/", $space, "/", $version, "/", $name, ".json")))?;)+
        }};
    }
    namespace!(
        "core",
        "2.11.0",
        "namespace",
        "nwb.base",
        "nwb.behavior",
        "nwb.device",
        "nwb.ecephys",
        "nwb.epoch",
        "nwb.event",
        "nwb.file",
        "nwb.icephys",
        "nwb.image",
        "nwb.misc",
        "nwb.ogen",
        "nwb.ophys",
        "nwb.retinotopy"
    );
    namespace!(
        "hdmf-common",
        "1.10.0",
        "namespace",
        "base",
        "table",
        "resources",
        "sparse"
    );
    namespace!("hdmf-experimental", "0.6.0", "namespace", "experimental");
    reference_attr(f, ".specloc", f, "/specifications")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nwb_units::Column;
    use crate::{ClusterId, DataProvider, NwbProvider, SampleIndex};

    #[test]
    fn appended_units_round_trip_with_local_storage() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let recording = tmp.path().join("recording.nwb");
        fs::copy(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../release/fixtures/nwb/good.nwb"),
            &recording,
        )?;
        let f = File::open_rw(&recording)?;
        f.unlink("/units")?;
        f.close()?;

        let units = Units {
            description: "Curated units".into(),
            resolution: 0.001,
            id: vec![5, 9],
            columns: vec![Column {
                name: "spike_times".into(),
                description: "Spike times in seconds".into(),
                data: Values::Float(vec![0.55, 0.701, 0.79]),
                index: Some(vec![2, 3]),
            }],
            generated_by: ["sorrel-ephys", env!("CARGO_PKG_VERSION")],
        };
        // Fail after /units has already been written to the staged copy.
        let f = File::open_rw(&recording)?;
        text_dataset(&f, "/general/was_generated_by", "not a two-column dataset")?;
        f.close()?;
        let before = fs::read(&recording)?;
        assert!(append(&recording, &units).is_err());
        assert_eq!(fs::read(&recording)?, before);
        let f = File::open_rw(&recording)?;
        f.unlink("/general/was_generated_by")?;
        f.close()?;

        append(&recording, &units)?;
        let r = NwbProvider::open(&recording)?;
        assert_eq!(r.n_clusters(), 2);
        assert_eq!(r.unit_id(ClusterId(0)), 5);
        assert_eq!(r.unit_id(ClusterId(1)), 9);
        assert_eq!(
            r.spike_times(ClusterId(0)),
            &[SampleIndex(50), SampleIndex(201)]
        );
        assert_eq!(r.spike_times(ClusterId(1)), &[SampleIndex(290)]);
        assert_eq!((r.exact_sample_rate()?, r.starting_time()), (1000., 0.5));

        let before = fs::read(&recording)?;
        assert!(append(&recording, &units)
            .unwrap_err()
            .to_string()
            .contains("already has /units"));
        assert_eq!(fs::read(&recording)?, before);
        Ok(())
    }
}
