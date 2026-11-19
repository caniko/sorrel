//! NWB reader invariants against files written by real PyNWB
//! (`release/fixtures/nwb/generate.py`).
#![cfg(any(feature = "hdf5", feature = "zarr"))]

use sorrel_io::kilosort::PhyLabel;
use sorrel_io::{
    ChannelId, ClusterId, DataProvider, HasGeometry, HasQualityMetrics, NwbOpenParams, NwbProvider,
    SampleIndex,
};
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../release/fixtures/nwb")
        .join(name)
}

#[derive(Clone, Copy)]
enum Format {
    #[cfg(feature = "hdf5")]
    Hdf5,
    #[cfg(feature = "zarr")]
    Zarr,
}

const FORMATS: &[Format] = &[
    #[cfg(feature = "hdf5")]
    Format::Hdf5,
    #[cfg(feature = "zarr")]
    Format::Zarr,
];

impl Format {
    fn fixture(self, name: &str) -> PathBuf {
        match self {
            #[cfg(feature = "hdf5")]
            Self::Hdf5 => fixture(name),
            #[cfg(feature = "zarr")]
            Self::Zarr => fixture(&format!("{name}.zarr")),
        }
    }

    fn open(self, name: &str) -> anyhow::Result<NwbProvider> {
        NwbProvider::open(self.fixture(name))
    }

    fn open_err(self, name: &str) -> String {
        match self.open(name) {
            Ok(_) => panic!("{name} must be rejected"),
            Err(e) => format!("{e:#}"),
        }
    }
}

/// The first frame of the trace, as f32.
fn frame(p: &NwbProvider, t: u64) -> Vec<f32> {
    p.trace(SampleIndex(t), 1).samples.to_f32_centred()
}

#[test]
fn unit_ids_and_spike_times_come_from_the_units_table() {
    for format in FORMATS {
        let p = format.open("good.nwb").unwrap();
        assert_eq!(p.n_clusters(), 3);
        let ids: Vec<u32> = (0..3).map(|c| p.unit_id(ClusterId(c))).collect();
        assert_eq!(ids, [5, 7, 9]);
        assert_eq!(
            p.unit_id(ClusterId(3)),
            10,
            "split-created clusters append after the largest id"
        );
        // Seconds on the session clock, relative to the series' starting_time
        // (0.5 s) at 1 kHz.
        let s = |c| -> Vec<u64> { p.spike_times(ClusterId(c)).iter().map(|s| s.0).collect() };
        assert_eq!(s(0), [100, 200, 250]);
        assert!(s(1).is_empty(), "unit 7 has no spikes");
        assert_eq!(s(2), [50, 290]);
    }
}

#[test]
fn traces_are_microvolts_after_conversion_offset_and_channel_conversion() {
    for format in FORMATS {
        let p = format.open("good.nwb").unwrap();
        assert_eq!(p.sample_rate(), 1000.0);
        assert_eq!(p.n_channels(), 4);
        assert_eq!(p.n_samples(), SampleIndex(300));
        let channel_conversion = [1.0, 2.0, 1.0, 0.5];
        let got = frame(&p, 7);
        for (c, cc) in channel_conversion.iter().enumerate() {
            let raw = (7 * (c as i32 + 1) - 100) as f64;
            let uv = (raw * 0.195e-6 * cc - 2e-6) * 1e6;
            assert!(
                (got[c] as f64 - uv).abs() < 1e-3,
                "channel {c}: {} µV, want {uv}",
                got[c]
            );
        }
    }
}

#[test]
fn channels_follow_the_series_electrodes_region() {
    for format in FORMATS {
        let p = format.open("good.nwb").unwrap();
        // Region [3, 0, 5, 1] into an electrodes table with ids 10..=15.
        assert_eq!(
            p.channel_map(),
            [ChannelId(13), ChannelId(10), ChannelId(15), ChannelId(11)]
        );
        // Probe-relative rel_x/rel_y, not the x/y brain coordinates.
        assert_eq!(
            p.channel_positions(),
            [[16.0, 20.0], [0.0, 0.0], [32.0, 20.0], [0.0, 20.0]]
        );
    }
}

#[test]
fn uint16_samples_are_read_unsigned() {
    for format in FORMATS {
        let p = format.open("uint16.nwb").unwrap();
        // data[t, c] = 40000 + t + c, conversion 1e-6 V → the raw value in µV.
        assert_eq!(frame(&p, 3), [40003.0, 40004.0, 40005.0, 40006.0]);
    }
}

#[test]
fn samples_are_scaled_from_their_stored_dtype() {
    for format in FORMATS {
        for (dtype, value) in [
            ("int8", -120.0),
            ("uint8", 250.0),
            ("int32", -2e9),
            ("uint32", 4e9),
            ("int64", -(2.0f32).powi(48)),
            ("float32", 0.125),
            ("float64", 1.0),
        ] {
            let p = format.open(&format!("{dtype}.nwb")).unwrap();
            assert_eq!(frame(&p, 0), [value; 4], "{dtype}");
        }
    }
}

#[test]
fn electrical_series_subtypes_are_accepted() {
    for format in FORMATS {
        let p = format.open("subtype.nwb").unwrap();
        assert_eq!(p.n_channels(), 4);
        assert_eq!(frame(&p, 0), [-100.0, -100.0, -100.0, -100.0]);
    }
}

#[test]
fn rejects_what_it_cannot_read_faithfully() {
    for format in FORMATS {
        let e = format.open_err("not_electrical.nwb");
        assert!(e.contains("ElectricalSeries"), "{e}");
        let e = format.open_err("uint64.nwb");
        assert!(e.contains("uint64"), "{e}");
        let e = format.open_err("irregular_timestamps.nwb");
        assert!(
            e.contains("irregular timestamps") && e.contains("197"),
            "{e}"
        );
    }
}

#[test]
fn timestamps_supply_the_same_time_base_as_starting_time() {
    for format in FORMATS {
        let p = format.open("timestamps.nwb").unwrap();
        let good = format.open("good.nwb").unwrap();
        assert_eq!(p.sample_rate(), 1000.0);
        assert_eq!(p.spike_times(ClusterId(0)), good.spike_times(ClusterId(0)));
        assert_eq!(frame(&p, 7), frame(&good, 7));
    }
}

#[cfg(feature = "hdf5")]
#[test]
fn timestamps_and_close_units_resolution_are_both_selectable_clocks() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("timestamps.nwb");
    std::fs::copy(fixture("timestamps.nwb"), &path).unwrap();
    let stored = {
        let file = hdf5_metno::File::open_rw(&path).unwrap();
        let timestamps = file
            .dataset("/acquisition/ElectricalSeriesRaw/timestamps")
            .unwrap()
            .read_raw::<f64>()
            .unwrap();
        file.dataset("/units/spike_times")
            .unwrap()
            .new_attr::<f64>()
            .create("resolution")
            .unwrap()
            .write_scalar(&0.0009999995_f64)
            .unwrap();
        ((timestamps[timestamps.len() - 1] - timestamps[0]) / (timestamps.len() - 1) as f64).recip()
    };
    let units = 0.0009999995_f64.recip();
    let open = |raw_path, sample_rate| {
        NwbProvider::open_with(
            &path,
            NwbOpenParams {
                raw_path,
                sample_rate,
                ..Default::default()
            },
        )
    };
    for raw_path in [None, Some(path.as_path())] {
        let expected = open(raw_path, None).unwrap();
        for rate in [stored, units] {
            let p = open(raw_path, Some(rate)).unwrap();
            assert_eq!(
                p.spike_times(ClusterId(0)),
                expected.spike_times(ClusterId(0))
            );
            assert!(p
                .identity_bytes()
                .windows(8)
                .any(|bytes| bytes == rate.to_le_bytes()));
        }
        let near = 1000.00000001;
        let err = open(raw_path, Some(near)).err().unwrap();
        assert_eq!(err.to_string(), format!("--sample-rate {near} Hz is not a clock this NWB declares; pass one of: {stored} Hz (series /acquisition/ElectricalSeriesRaw timestamps), {units} Hz (/units/spike_times resolution)"));
    }
}

#[cfg(feature = "hdf5")]
#[test]
fn refuses_spikes_that_disagree_with_recorded_timestamps() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("midpoint.nwb");
    std::fs::copy(fixture("good.nwb"), &path).unwrap();
    let origin = 1_700_000_000.0_f64;
    let ulp = origin.next_up() - origin;
    let period = 128.0 * ulp;
    // The grid assigns the spike to the first sample of a new read block.
    let assigned = 65_536;
    let last = assigned + 3;
    let mut times: Vec<_> = (0..=last).map(|i| origin + i as f64 * period).collect();
    let midpoint = times[assigned - 1] + period / 2.0;
    let spike = midpoint + ulp;
    let (timestamps, spikes) = {
        let file = hdf5_metno::File::open_rw(&path).unwrap();
        let series = file.group("/acquisition/ElectricalSeriesRaw").unwrap();
        series.unlink("starting_time").unwrap();
        series.unlink("data").unwrap();
        series
            .new_dataset::<i16>()
            .shape((times.len(), 4))
            .create("data")
            .unwrap();
        let timestamps = series
            .new_dataset_builder()
            .with_data(&times)
            .create("timestamps")
            .unwrap();
        (timestamps, file.dataset("/units/spike_times").unwrap())
    };
    let write_spikes = |t| {
        spikes
            .write(&[times[0], t, times[last], times[0], times[last]])
            .unwrap()
    };
    for t in [midpoint, spike] {
        write_spikes(t);
        let p = NwbProvider::open(&path).unwrap();
        assert_eq!(
            p.spike_times(ClusterId(0)),
            [
                SampleIndex(0),
                SampleIndex(assigned as u64),
                SampleIndex(last as u64)
            ]
        );
    }
    // Still a valid inferred grid, but the closest recorded sample is before the block boundary.
    write_spikes(spike);
    times[assigned - 1] += 3.0 * ulp;
    timestamps.write(&times).unwrap();
    let err = NwbProvider::open(&path).err().expect("mismapped midpoint");
    let err = format!("{err:#}");
    for text in [
        format!("unit 5: spike at {spike} s"),
        format!("assigned sample {assigned}"),
        format!("nearest recorded sample is {}", assigned - 1),
    ] {
        assert!(err.contains(&text), "{err}");
    }

    for t in [origin - period, times[last] + period] {
        spikes.write(&[t, t, t, t, t]).unwrap();
        let err = NwbProvider::open(&path).err().expect("out-of-range spike");
        assert!(
            format!("{err:#}").contains("outside the sample index range"),
            "{err:#}"
        );
    }
}

#[cfg(feature = "hdf5")]
#[test]
fn spike_mapping_datasets_must_be_stored_in_the_opened_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("recording.nwb");
    let clock = "/acquisition/ElectricalSeriesRaw/starting_time";
    for (kind, component) in [
        ("soft link", clock),
        ("soft link", "/acquisition"),
        ("external raw storage", clock),
    ] {
        std::fs::copy(fixture("good.nwb"), &path).unwrap();
        {
            let file = hdf5_metno::File::open_rw(&path).unwrap();
            if kind == "soft link" {
                file.relink(component, "/local_target").unwrap();
                file.link_soft("/local_target", component).unwrap();
            } else {
                let external = dir.path().join("clock.bin");
                file.unlink(clock).unwrap();
                let st = file
                    .new_dataset::<f64>()
                    .external(external.to_str().unwrap(), 0, 8)
                    .shape(())
                    .create(clock)
                    .unwrap();
                assert_eq!(st.dcpl().unwrap().get_external().unwrap().len(), 1);
                st.write_scalar(&0.5).unwrap();
                st.new_attr::<f64>()
                    .create("rate")
                    .unwrap()
                    .write_scalar(&1000.0)
                    .unwrap();
            }
        }
        // Exercise the same check for the units file and the --dat file.
        for paired in [false, true] {
            let (units, raw_path) = if paired {
                (fixture("units_only.nwb"), Some(path.as_path()))
            } else {
                (path.clone(), None)
            };
            let err = NwbProvider::open_with(
                units,
                NwbOpenParams {
                    raw_path,
                    ..Default::default()
                },
            )
            .err()
            .unwrap_or_else(|| panic!("must reject {kind} at {component} (paired: {paired})"));
            let err = format!("{err:#}");
            assert!(
                err.contains(clock) && err.contains(kind) && err.contains("--dat"),
                "{err}"
            );
            if kind == "soft link" {
                assert!(
                    err.contains(&format!("{component} is an HDF5 soft link")),
                    "{err}"
                );
            }
        }
    }
}

#[test]
fn series_selection_prefers_acquisition_and_lists_available_paths() {
    for format in FORMATS {
        let path = format.fixture("multiple_series.nwb");
        let default = NwbProvider::open(&path).unwrap();
        assert_eq!(frame(&default, 0), [1.0; 4]);
        for (series, value) in [
            ("A_first", 1.0),
            ("Z_second", 2.0),
            ("/acquisition/Z_second", 2.0),
            ("/processing/ecephys/FilteredEphys/Processed", 3.0),
        ] {
            let p = NwbProvider::open_with(
                &path,
                NwbOpenParams {
                    series: Some(series),
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(frame(&p, 0), [value; 4]);
            assert_eq!(p.identity_bytes() == default.identity_bytes(), value == 1.0);
        }
        let err = NwbProvider::open_with(
            &path,
            NwbOpenParams {
                series: Some("Processed"),
                ..Default::default()
            },
        )
        .err()
        .unwrap()
        .to_string();
        for path in [
            "/acquisition/A_first",
            "/acquisition/Z_second",
            "/processing/ecephys/FilteredEphys/Processed",
        ] {
            assert!(err.contains(path), "{err}");
        }
    }
}

#[test]
fn units_only_have_no_trace_and_use_resolution_or_sample_rate() {
    for format in FORMATS {
        let p = format.open("units_only.nwb").unwrap();
        assert_eq!(p.sample_rate(), 1000.0);
        assert_eq!(p.n_samples(), SampleIndex(0));
        assert_eq!(p.channel_map(), (10..16).map(ChannelId).collect::<Vec<_>>());
        assert_eq!(p.channel_positions()[3], [16.0, 20.0]);
        assert_eq!(p.spike_times(ClusterId(0))[0], SampleIndex(600));
        let window = p.trace(SampleIndex(12), 10);
        assert_eq!(window.n_channels, 6);
        assert_eq!(window.start, SampleIndex(12));
        assert!(window.samples.is_empty());
        let err = format.open_err("not_electrical.nwb");
        assert!(err.contains("--sample-rate"), "{err}");
        let p = NwbProvider::open_with(
            format.fixture("not_electrical.nwb"),
            NwbOpenParams {
                sample_rate: Some(2000.0),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(p.spike_times(ClusterId(0))[0], SampleIndex(1200));
        let with_resolution = |sample_rate| {
            NwbProvider::open_with(
                format.fixture("units_only.nwb"),
                NwbOpenParams {
                    sample_rate: Some(sample_rate),
                    ..Default::default()
                },
            )
        };
        let exact = with_resolution(1000.0).unwrap();
        assert_eq!(exact.spike_times(ClusterId(0))[0], SampleIndex(600));
        let err = with_resolution(1000.00000001).err().unwrap();
        assert_eq!(err.to_string(), "--sample-rate 1000.00000001 Hz is not a clock this NWB declares; pass one of: 1000 Hz (/units/spike_times resolution)");
    }
}

#[test]
fn paired_raw_uses_its_series_channels_and_time_base() {
    for format in FORMATS {
        for raw_format in FORMATS {
            let raw = raw_format.fixture("raw_only.nwb");
            let p = NwbProvider::open_with(
                format.fixture("units_only.nwb"),
                NwbOpenParams {
                    raw_path: Some(&raw),
                    series: Some("ElectricalSeriesRaw"),
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(p.sample_rate(), 1000.0);
            assert_eq!(p.n_samples(), SampleIndex(300));
            assert_eq!(
                p.channel_map(),
                [ChannelId(13), ChannelId(10), ChannelId(15), ChannelId(11)]
            );
            assert_eq!(p.spike_times(ClusterId(0))[0], SampleIndex(100));
            assert_eq!(frame(&p, 0), [-100.0; 4]);
            assert!(format.open_err("raw_only.nwb").contains("/units"));

            let err = NwbProvider::open_with(
                format.fixture("not_electrical.nwb"),
                NwbOpenParams {
                    raw_path: Some(&raw),
                    sample_rate: Some(2000.0),
                    ..Default::default()
                },
            )
            .err()
            .unwrap()
            .to_string();
            assert_eq!(err, "--sample-rate 2000 Hz is not a clock this NWB declares; pass one of: 1000 Hz (series /acquisition/ElectricalSeriesRaw starting_time.rate)");
            let err = NwbProvider::open_with(
                format.fixture("units_only.nwb"),
                NwbOpenParams {
                    raw_path: Some(&format.fixture("not_electrical.nwb")),
                    ..Default::default()
                },
            )
            .err()
            .unwrap()
            .to_string();
            assert!(err.contains("available series: none"), "{err}");
        }
    }
}

#[cfg(feature = "zarr")]
mod zarr_tests {
    use super::*;
    use std::{fs, path::Path, time::Duration};

    fn copy_store(src: &Path, dst: &Path) {
        fs::create_dir(dst).unwrap();
        for entry in fs::read_dir(src).unwrap() {
            let entry = entry.unwrap();
            let target = dst.join(entry.file_name());
            if entry.file_type().unwrap().is_dir() {
                copy_store(&entry.path(), &target);
            } else {
                fs::copy(entry.path(), target).unwrap();
            }
        }
    }

    #[test]
    fn refuses_linked_clock_even_when_target_is_in_the_store() {
        let err = Format::Zarr.open_err("linked_clock.nwb");
        assert!(
            err.contains("timestamps must be stored in this file")
                && err.contains("external zarr_link"),
            "{err}"
        );

        let temp = tempfile::tempdir().unwrap();
        let store = temp.path().join("local.nwb.zarr");
        copy_store(&Format::Zarr.fixture("linked_clock.nwb"), &store);
        let src = Format::Zarr
            .fixture("timestamps.nwb")
            .join("acquisition/ElectricalSeriesRaw/timestamps");
        copy_store(&src, &store.join("clock"));
        let path = store.join("acquisition/ElectricalSeriesRaw/.zattrs");
        let mut attrs: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        attrs["zarr_link"][0]["source"] = ".".into();
        attrs["zarr_link"][0]["path"] = "/clock".into();
        fs::write(path, serde_json::to_vec(&attrs).unwrap()).unwrap();
        let err = NwbProvider::open(&store).err().expect("linked clock");
        let err = format!("{err:#}");
        assert!(
            err.contains("timestamps must be stored in this file") && err.contains("soft link"),
            "{err}"
        );
    }

    #[test]
    fn electrode_region_resolves_its_local_table_reference() {
        let temp = tempfile::tempdir().unwrap();
        let store = temp.path().join("region.nwb.zarr");
        copy_store(&Format::Zarr.fixture("good.nwb"), &store);
        fs::rename(
            store.join("general/extracellular_ephys/electrodes"),
            store.join("alternate_table"),
        )
        .unwrap();
        let path = store.join("acquisition/ElectricalSeriesRaw/electrodes/.zattrs");
        let mut attrs: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        attrs["table"]["value"]["path"] = "/alternate_table".into();
        fs::write(&path, serde_json::to_vec(&attrs).unwrap()).unwrap();
        let p = NwbProvider::open(&store).unwrap();
        assert_eq!(
            p.channel_map(),
            [ChannelId(13), ChannelId(10), ChannelId(15), ChannelId(11)]
        );
        attrs["table"]["value"]["source"] = "../external.nwb.zarr".into();
        fs::write(path, serde_json::to_vec(&attrs).unwrap()).unwrap();
        let err = NwbProvider::open(&store).err().expect("external table");
        assert!(
            format!("{err:#}").contains("external Zarr object reference table"),
            "{err:#}"
        );
    }

    #[test]
    fn chunk_and_metadata_edits_invalidate_identity() {
        let temp = tempfile::tempdir().unwrap();
        let store = temp.path().join("units.nwb.zarr");
        copy_store(&Format::Zarr.fixture("units_only.nwb"), &store);
        let seal = || NwbProvider::open(&store).unwrap().identity_bytes();
        let original = seal();
        let array = store.join("units/spike_times");
        let chunk = array.join("0");
        let original_chunk = fs::read(&chunk).unwrap();
        let stamp = fs::metadata(&chunk).unwrap().modified().unwrap();
        // Sorrel's Blosc plugin only decodes; replace the chunk with a
        // numcodecs-encoded version containing different spike times.
        let edited_chunk = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../release/fixtures/zarr-blosc/edited-spikes.bin");
        fs::copy(edited_chunk, &chunk).unwrap();
        assert_ne!(original_chunk, fs::read(&chunk).unwrap());
        let file = fs::OpenOptions::new().write(true).open(&chunk).unwrap();
        file.set_modified(stamp - Duration::from_secs(2)).unwrap();
        let edited = seal();
        assert_ne!(original, edited);

        let attrs = array.join(".zattrs");
        let timestamp = fs::metadata(&attrs).unwrap().modified().unwrap();
        let text = fs::read_to_string(&attrs).unwrap().replacen('\n', "\t", 1);
        fs::write(&attrs, text).unwrap();
        fs::File::options()
            .write(true)
            .open(&attrs)
            .unwrap()
            .set_modified(timestamp)
            .unwrap();
        assert_ne!(edited, seal());
    }
}

#[test]
fn reads_python_export_metadata_by_unit_row() {
    for format in FORMATS {
        let p = format.open("curated_units.nwb").unwrap();
        assert_eq!(p.n_channels(), 0);
        assert_eq!(p.initial_labels(), [PhyLabel::Mua, PhyLabel::Good]);
        assert_eq!(
            p.initial_tags(),
            [
                vec![],
                vec!["split".to_string(), "review:waveform".to_string()]
            ]
        );
        assert_eq!(p.sorter().as_deref(), Some("kilosort4"));
        let annotations = p.initial_annotations();
        assert_eq!(annotations[0].origin, "Auto[kilosort4]");
        assert_eq!(annotations[1].origin, "Sorrel[Manual]");
        assert_eq!(
            annotations[1].parents,
            [ClusterId(0)],
            "parent 77 has no row"
        );
        assert_eq!(p.unit_id(ClusterId(1)), 5);
        assert_eq!(p.unit_id_json(ClusterId(0)), serde_json::json!("007"));
        assert_eq!(p.unit_id_json(ClusterId(1)), serde_json::json!("unit_a"));
        assert_eq!(p.unit_id_json(ClusterId(2)), serde_json::json!("10"));
        assert_eq!(p.metric_names(), ["l_ratio", "n_spikes"]);
        assert_eq!(p.metric_values("n_spikes").unwrap(), [2.0, 3.0]);
        let l_ratio = p.metric_values("l_ratio").unwrap();
        assert!(l_ratio[0].is_nan());
        assert_eq!(l_ratio[1], 0.25);
    }
}

#[test]
fn paired_series_supplies_rate_without_units_resolution() {
    for format in FORMATS {
        let raw = format.fixture("raw_only.nwb");
        let p = NwbProvider::open_with(
            format.fixture("not_electrical.nwb"),
            NwbOpenParams {
                raw_path: Some(&raw),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(p.spike_times(ClusterId(0))[0], SampleIndex(100));
    }
}

#[cfg(feature = "hdf5")]
#[test]
fn single_file_float32_maps_long_recording_at_either_exact_clock() {
    let err = Format::Hdf5.open_err("precise_rate_long.nwb");
    for part in [
        "30000.123046875 Hz",
        "30000.123 Hz",
        "1 spike",
        "unit 0 spike 0",
        "330000001 vs 330000000",
        "--sample-rate <Hz>",
    ] {
        assert!(err.contains(part), "{err}");
    }
    for (rate, sample) in [(30000.123, 330_000_000), (30000.123046875, 330_000_001)] {
        let p = NwbProvider::open_with(
            fixture("precise_rate_long.nwb"),
            NwbOpenParams {
                sample_rate: Some(rate),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(p.n_samples(), SampleIndex(330_000_002));
        assert_eq!(p.spike_times(ClusterId(0)), [SampleIndex(sample)]);
    }
}

#[cfg(feature = "hdf5")]
#[test]
fn single_file_float32_rate_does_not_assume_resampled_units_use_acquisition_clock() {
    let path = fixture("resampled_units_clock.nwb");
    let open = |sample_rate| {
        NwbProvider::open_with(
            &path,
            NwbOpenParams {
                series: Some("ElectricalSeriesRaw"),
                sample_rate,
                ..Default::default()
            },
        )
    };
    let err = format!(
        "{:#}",
        open(None).err().expect("ambiguous clocks must be refused")
    );
    // Taking the reciprocal of the stored resolution can move the last bit.
    for part in ["30000 Hz", "29999.99909", "101 vs 100"] {
        assert!(err.contains(part), "{err}");
    }
    let p = open(Some(30000.0)).unwrap();
    assert_eq!(p.spike_times(ClusterId(0)), [SampleIndex(101)]);
}

#[cfg(feature = "hdf5")]
#[test]
fn single_file_float64_maps_long_recording_at_either_exact_clock() {
    let name = "close_f64_rate_long.nwb";
    let err = Format::Hdf5.open_err(name);
    assert!(err.contains("660000001 vs 660000000"), "{err}");
    assert!(err.contains("--sample-rate <Hz>"), "{err}");

    for (rate, sample) in [
        (30000.123, 660_000_000),
        (30_000.123_026_999_998, 660_000_001),
    ] {
        let p = NwbProvider::open_with(
            fixture(name),
            NwbOpenParams {
                sample_rate: Some(rate),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(p.spike_times(ClusterId(0)), [SampleIndex(sample)]);
    }
}

#[cfg(feature = "hdf5")]
#[test]
fn reviewer_short_files_require_exact_clock_values() {
    let dir = tempfile::tempdir().unwrap();
    // Keep the attribute types of the committed source files. All three
    // spikes are close enough to a half-sample to expose a changed clock.
    for (name, source, stored, resolution, spike, near, stored_sample) in [
        (
            "near-f64.nwb",
            "close_f64_rate_short.nwb",
            30000.0_f64,
            3.3333333305555553e-5_f64,
            0.503_349_999_997_487_5_f64,
            30000.00002,
            100,
        ),
        (
            "near-f32.nwb",
            "precise_rate_short.nwb",
            30000.0,
            3.333333332222222e-5,
            0.5033499999991067,
            30000.000006,
            100,
        ),
        (
            "approx-short.nwb",
            "close_f64_rate_short.nwb",
            30000.000029,
            3.333333322222222e-5,
            0.5033499999983333,
            30000.0,
            101,
        ),
    ] {
        let path = dir.path().join(name);
        std::fs::copy(fixture(source), &path).unwrap();
        {
            let f = hdf5_metno::File::open_rw(&path).unwrap();
            let start = f
                .dataset("/acquisition/ElectricalSeriesRaw/starting_time")
                .unwrap();
            start.write_scalar(&0.5_f64).unwrap();
            start.attr("rate").unwrap().write_scalar(&stored).unwrap();
            let spikes = f.dataset("/units/spike_times").unwrap();
            spikes
                .attr("resolution")
                .unwrap()
                .write_scalar(&resolution)
                .unwrap();
            spikes.write_raw(&[spike]).unwrap();
        }
        let open = |sample_rate| {
            NwbProvider::open_with(
                &path,
                NwbOpenParams {
                    sample_rate,
                    ..Default::default()
                },
            )
        };
        let units = resolution.recip();
        for (rate, sample) in [(stored, stored_sample), (units, 101)] {
            let p = open(Some(rate)).unwrap();
            assert_eq!(
                p.spike_times(ClusterId(0)),
                [SampleIndex(sample)],
                "{name} at {rate}"
            );
        }
        let err = open(Some(near)).err().expect(name);
        assert_eq!(err.to_string(), format!("--sample-rate {near} Hz is not a clock this NWB declares; pass one of: {stored} Hz (series /acquisition/ElectricalSeriesRaw starting_time.rate), {units} Hz (/units/spike_times resolution)"));
        if stored_sample == 101 {
            let default = open(None).unwrap();
            assert_eq!(default.spike_times(ClusterId(0)), [SampleIndex(101)]);
            assert_eq!(
                default.identity_bytes(),
                open(Some(units)).unwrap().identity_bytes()
            );
        } else {
            let err = open(None).err().expect(name);
            assert!(format!("{err:#}").contains("100 vs 101"), "{name}: {err:#}");
        }
    }
}

#[cfg(feature = "hdf5")]
#[test]
fn clock_values_must_be_stored_as_float32_or_float64() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("int-rate.nwb");
    std::fs::copy(fixture("close_f64_rate_short.nwb"), &path).unwrap();
    {
        let f = hdf5_metno::File::open_rw(&path).unwrap();
        let st = f
            .dataset("/acquisition/ElectricalSeriesRaw/starting_time")
            .unwrap();
        st.delete_attr("rate").unwrap();
        st.new_attr::<i32>()
            .create("rate")
            .unwrap()
            .write_scalar(&30000)
            .unwrap();
    }
    let err = format!("{:#}", NwbProvider::open(&path).err().expect("refused"));
    assert!(
        err.contains("starting_time rate must be stored as native IEEE float32 or float64"),
        "{err}"
    );
}

#[cfg(feature = "hdf5")]
#[test]
fn single_file_refuses_a_resolution_it_cannot_read_exactly() {
    // A swallowed storage error would erase the competing Units clock.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("int-resolution.nwb");
    std::fs::copy(fixture("close_f64_rate_short.nwb"), &path).unwrap();
    {
        let f = hdf5_metno::File::open_rw(&path).unwrap();
        let spikes = f.dataset("/units/spike_times").unwrap();
        spikes.delete_attr("resolution").unwrap();
        spikes
            .new_attr::<i32>()
            .create("resolution")
            .unwrap()
            .write_scalar(&1)
            .unwrap();
    }
    let err = format!("{:#}", NwbProvider::open(&path).err().expect("refused"));
    assert!(
        err.contains("resolution must be stored as native IEEE"),
        "{err}"
    );
}

#[cfg(feature = "hdf5")]
#[test]
fn coarse_units_resolution_is_ignored_only_in_a_single_file() {
    let dir = tempfile::tempdir().unwrap();
    let sorting = dir.path().join("coarse.nwb");
    let raw = fixture("close_f64_rate_short.nwb");
    std::fs::copy(&raw, &sorting).unwrap();
    hdf5_metno::File::open_rw(&sorting)
        .unwrap()
        .dataset("/units/spike_times")
        .unwrap()
        .attr("resolution")
        .unwrap()
        .write_scalar(&1e-4_f64)
        .unwrap();

    let p = NwbProvider::open(&sorting).unwrap();
    assert_eq!(p.sample_rate(), 30000.123_f32);
    assert_eq!(p.spike_times(ClusterId(0)), [SampleIndex(100)]);
    let err = NwbProvider::open_with(
        &sorting,
        NwbOpenParams {
            sample_rate: Some(10000.0),
            ..Default::default()
        },
    )
    .err()
    .unwrap();
    assert_eq!(err.to_string(), "--sample-rate 10000 Hz is not a clock this NWB declares; pass one of: 30000.123026999998 Hz (series /acquisition/ElectricalSeriesRaw starting_time.rate)");

    let err = NwbProvider::open_with(
        &sorting,
        NwbOpenParams {
            raw_path: Some(&raw),
            ..Default::default()
        },
    )
    .err()
    .expect("a paired sorting with an incompatible rate must be refused");
    assert!(err.to_string().contains("NWB rate mismatch"), "{err:#}");
}

#[cfg(feature = "hdf5")]
#[test]
fn paired_rate_within_one_ppm_requires_mapping_agreement() {
    let dir = tempfile::tempdir().unwrap();
    let raw = dir.path().join("raw.nwb");
    let sorting = fixture("close_f64_rate_long.nwb");
    std::fs::copy(&sorting, &raw).unwrap();
    hdf5_metno::File::open_rw(&raw)
        .unwrap()
        .dataset("/acquisition/ElectricalSeriesRaw/starting_time")
        .unwrap()
        .attr("rate")
        .unwrap()
        .write_scalar(&30000.125_f64) // ~0.067 ppm away, beyond f32 equality and the old f64 check.
        .unwrap();

    let open = |sample_rate| {
        NwbProvider::open_with(
            &sorting,
            NwbOpenParams {
                raw_path: Some(&raw),
                sample_rate,
                ..Default::default()
            },
        )
    };
    let err = open(None).err().expect("ambiguous clocks must be refused");
    assert!(format!("{err:#}").contains("--sample-rate <Hz>"), "{err:#}");
    let p = open(Some(30000.123)).unwrap();
    assert_eq!(p.spike_times(ClusterId(0)), [SampleIndex(660_000_000)]);
}

#[cfg(feature = "hdf5")]
#[test]
fn paired_float32_requires_and_seals_an_exact_clock_choice() {
    let raw = fixture("rounded_raw.nwb");
    let open = |sample_rate| {
        NwbProvider::open_with(
            fixture("precise_units.nwb"),
            NwbOpenParams {
                raw_path: Some(&raw),
                sample_rate,
                ..Default::default()
            },
        )
    };
    let err = format!(
        "{:#}",
        open(None).err().expect("ambiguous clocks must be refused")
    );
    assert!(
        err.contains("31 vs 30") && err.contains("--sample-rate <Hz>"),
        "{err}"
    );
    let precise = open(Some(30000.123)).unwrap();
    let rounded = open(Some(f64::from(30000.123_f32))).unwrap();
    assert_eq!(precise.sample_rate(), rounded.sample_rate());
    assert_eq!(precise.spike_times(ClusterId(0)), [SampleIndex(30)]);
    assert_eq!(rounded.spike_times(ClusterId(0)), [SampleIndex(31)]);
    assert_ne!(precise.identity_bytes(), rounded.identity_bytes());
    let err = open(Some(30000.12301)).err().unwrap();
    assert_eq!(err.to_string(), "--sample-rate 30000.12301 Hz is not a clock this NWB declares; pass one of: 30000.123046875 Hz (series /acquisition/ElectricalSeriesRaw starting_time.rate), 30000.123 Hz (/units/spike_times resolution)");
}

#[test]
fn paired_series_rejects_different_timestamp_epoch() {
    for format in FORMATS {
        let raw = format.fixture("shifted_raw.nwb");
        let err = NwbProvider::open_with(
            format.fixture("units_only.nwb"),
            NwbOpenParams {
                raw_path: Some(&raw),
                ..Default::default()
            },
        )
        .err()
        .unwrap();
        assert!(
            err.to_string().contains("timestamps_reference_time"),
            "{err:#}"
        );
    }
}

#[cfg(feature = "hdf5")]
#[test]
fn paired_series_reads_fixed_and_variable_length_epochs() {
    use hdf5_metno::types::{FixedAscii, FixedUnicode, VarLenAscii, VarLenUnicode};

    fn write_epoch<T: hdf5_metno::H5Type>(f: &hdf5_metno::File, epoch: &T) {
        f.unlink("timestamps_reference_time").unwrap();
        f.new_dataset::<T>()
            .shape(())
            .create("timestamps_reference_time")
            .unwrap()
            .write_scalar(epoch)
            .unwrap();
    }

    let dir = tempfile::tempdir().unwrap();
    let raw = dir.path().join("raw.nwb");
    const EPOCH: &str = "2026-01-01T00:00:00+00:00";
    for write in [
        (|f| write_epoch(f, &FixedAscii::<25>::from_ascii(EPOCH).unwrap()))
            as fn(&hdf5_metno::File),
        |f| write_epoch(f, &EPOCH.parse::<FixedUnicode<64>>().unwrap()),
        |f| write_epoch(f, &VarLenAscii::from_ascii(EPOCH).unwrap()),
        |f| write_epoch(f, &EPOCH.parse::<VarLenUnicode>().unwrap()),
    ] {
        std::fs::copy(fixture("raw_only.nwb"), &raw).unwrap();
        write(&hdf5_metno::File::open_rw(&raw).unwrap());
        NwbProvider::open_with(
            fixture("units_only.nwb"),
            NwbOpenParams {
                raw_path: Some(&raw),
                ..Default::default()
            },
        )
        .unwrap();
    }
}

#[test]
fn rank_one_series_reads_scaled_single_channel_windows() {
    for format in FORMATS {
        let p = format.open("one_channel.nwb").unwrap();
        assert_eq!(p.n_channels(), 1);
        assert_eq!(p.channel_map(), [ChannelId(10)]);
        assert_eq!(
            p.trace(SampleIndex(297), 5).samples.to_f32_centred(),
            [297.0, 298.0, 299.0]
        );
    }
}

#[test]
fn accepts_recognized_foreign_labels_without_guessing_other_columns() {
    for format in FORMATS {
        let p = NwbProvider::open_with(
            format.fixture("foreign_quality.nwb"),
            NwbOpenParams {
                sample_rate: Some(1000.0),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            p.initial_labels(),
            [PhyLabel::Good, PhyLabel::Noise, PhyLabel::Unsorted]
        );
        assert!(p.metric_names().is_empty());
        assert_eq!(p.unit_id_json(ClusterId(0)), serde_json::json!(5));
    }
}
