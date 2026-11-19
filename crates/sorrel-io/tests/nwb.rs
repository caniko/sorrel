//! NWB reader invariants against files written by real PyNWB
//! (`release/fixtures/nwb/generate.py`).
#![cfg(feature = "hdf5")]

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

fn open(name: &str) -> anyhow::Result<NwbProvider> {
    NwbProvider::open(fixture(name))
}

fn open_err(name: &str) -> String {
    match open(name) {
        Ok(_) => panic!("{name} must be rejected"),
        Err(e) => format!("{e:#}"),
    }
}

/// The first frame of the trace, as f32.
fn frame(p: &NwbProvider, t: u64) -> Vec<f32> {
    p.trace(SampleIndex(t), 1).samples.to_f32_centred()
}

#[test]
fn unit_ids_and_spike_times_come_from_the_units_table() {
    let p = open("good.nwb").unwrap();
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

#[test]
fn traces_are_microvolts_after_conversion_offset_and_channel_conversion() {
    let p = open("good.nwb").unwrap();
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

#[test]
fn channels_follow_the_series_electrodes_region() {
    let p = open("good.nwb").unwrap();
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

#[test]
fn uint16_samples_are_read_unsigned() {
    let p = open("uint16.nwb").unwrap();
    // data[t, c] = 40000 + t + c, conversion 1e-6 V → the raw value in µV.
    assert_eq!(frame(&p, 3), [40003.0, 40004.0, 40005.0, 40006.0]);
}

#[test]
fn samples_are_scaled_from_their_stored_dtype() {
    for (dtype, value) in [
        ("int8", -120.0),
        ("uint8", 250.0),
        ("int32", -2e9),
        ("uint32", 4e9),
        ("int64", -(2.0f32).powi(48)),
        ("float32", 0.125),
        ("float64", 1.0),
    ] {
        let p = open(&format!("{dtype}.nwb")).unwrap();
        assert_eq!(frame(&p, 0), [value; 4], "{dtype}");
    }
}

#[test]
fn electrical_series_subtypes_are_accepted() {
    let p = open("subtype.nwb").unwrap();
    assert_eq!(p.n_channels(), 4);
    assert_eq!(frame(&p, 0), [-100.0, -100.0, -100.0, -100.0]);
}

#[test]
fn rejects_what_it_cannot_read_faithfully() {
    let e = open_err("not_electrical.nwb");
    assert!(e.contains("ElectricalSeries"), "{e}");
    let e = open_err("uint64.nwb");
    assert!(e.contains("uint64"), "{e}");
    let e = open_err("irregular_timestamps.nwb");
    assert!(
        e.contains("irregular timestamps") && e.contains("197"),
        "{e}"
    );
}

#[test]
fn timestamps_supply_the_same_time_base_as_starting_time() {
    let p = open("timestamps.nwb").unwrap();
    let good = open("good.nwb").unwrap();
    assert_eq!(p.sample_rate(), 1000.0);
    assert_eq!(p.spike_times(ClusterId(0)), good.spike_times(ClusterId(0)));
    assert_eq!(frame(&p, 7), frame(&good, 7));
}

#[test]
fn series_selection_prefers_acquisition_and_lists_available_paths() {
    let path = fixture("multiple_series.nwb");
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

#[test]
fn units_only_have_no_trace_and_use_resolution_or_sample_rate() {
    let p = open("units_only.nwb").unwrap();
    assert_eq!(p.sample_rate(), 1000.0);
    assert_eq!(p.n_samples(), SampleIndex(0));
    assert_eq!(p.channel_map(), (10..16).map(ChannelId).collect::<Vec<_>>());
    assert_eq!(p.channel_positions()[3], [16.0, 20.0]);
    assert_eq!(p.spike_times(ClusterId(0))[0], SampleIndex(600));
    let window = p.trace(SampleIndex(12), 10);
    assert_eq!(window.n_channels, 6);
    assert_eq!(window.start, SampleIndex(12));
    assert!(window.samples.is_empty());
    let err = open_err("not_electrical.nwb");
    assert!(err.contains("--sample-rate"), "{err}");
    let p = NwbProvider::open_with(
        fixture("not_electrical.nwb"),
        NwbOpenParams {
            sample_rate: Some(2000.0),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(p.spike_times(ClusterId(0))[0], SampleIndex(1200));
}

#[test]
fn paired_raw_uses_its_series_channels_and_time_base() {
    let raw = fixture("raw_only.nwb");
    let p = NwbProvider::open_with(
        fixture("units_only.nwb"),
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
    assert!(open_err("raw_only.nwb").contains("/units"));

    let err = NwbProvider::open_with(
        fixture("not_electrical.nwb"),
        NwbOpenParams {
            raw_path: Some(&raw),
            sample_rate: Some(2000.0),
            ..Default::default()
        },
    )
    .err()
    .unwrap()
    .to_string();
    assert!(
        err.contains("rate mismatch") && err.contains("2000") && err.contains("1000"),
        "{err}"
    );
    let err = NwbProvider::open_with(
        fixture("units_only.nwb"),
        NwbOpenParams {
            raw_path: Some(&fixture("not_electrical.nwb")),
            ..Default::default()
        },
    )
    .err()
    .unwrap()
    .to_string();
    assert!(err.contains("available series: none"), "{err}");
}

#[test]
fn reads_python_export_metadata_by_unit_row() {
    let p = open("curated_units.nwb").unwrap();
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

#[test]
fn paired_series_supplies_rate_without_units_resolution() {
    let raw = fixture("raw_only.nwb");
    let p = NwbProvider::open_with(
        fixture("not_electrical.nwb"),
        NwbOpenParams {
            raw_path: Some(&raw),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(p.spike_times(ClusterId(0))[0], SampleIndex(100));
}

#[test]
fn paired_float32_rate_uses_precise_units_rate_for_spikes() {
    let raw = fixture("rounded_raw.nwb");
    let p = NwbProvider::open_with(
        fixture("precise_units.nwb"),
        NwbOpenParams {
            raw_path: Some(&raw),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(p.spike_times(ClusterId(0)), [SampleIndex(30)]);
}

#[test]
fn paired_float32_rate_seals_the_effective_rate() {
    let dir = tempfile::tempdir().unwrap();
    let units = dir.path().join("units.nwb");
    std::fs::copy(fixture("precise_units.nwb"), &units).unwrap();
    hdf5_metno::File::open_rw(&units)
        .unwrap()
        .dataset("/units/spike_times")
        .unwrap()
        .attr("resolution")
        .unwrap()
        .write_scalar(&-1.0_f64)
        .unwrap();
    let raw = fixture("rounded_raw.nwb");
    let open = |rate| {
        NwbProvider::open_with(
            &units,
            NwbOpenParams {
                raw_path: Some(&raw),
                sample_rate: Some(rate),
                ..Default::default()
            },
        )
        .unwrap()
    };
    let precise = open(30000.123);
    let rounded = open(f64::from(30000.123_f32));
    assert_eq!(precise.sample_rate(), rounded.sample_rate());
    assert_eq!(precise.spike_times(ClusterId(0)), [SampleIndex(30)]);
    assert_eq!(rounded.spike_times(ClusterId(0)), [SampleIndex(31)]);
    assert_ne!(precise.identity_bytes(), rounded.identity_bytes());
}

#[test]
fn paired_series_rejects_different_timestamp_epoch() {
    let raw = fixture("shifted_raw.nwb");
    let err = NwbProvider::open_with(
        fixture("units_only.nwb"),
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
    let p = open("one_channel.nwb").unwrap();
    assert_eq!(p.n_channels(), 1);
    assert_eq!(p.channel_map(), [ChannelId(10)]);
    assert_eq!(
        p.trace(SampleIndex(297), 5).samples.to_f32_centred(),
        [297.0, 298.0, 299.0]
    );
}

#[test]
fn accepts_recognized_foreign_labels_without_guessing_other_columns() {
    let p = NwbProvider::open_with(
        fixture("foreign_quality.nwb"),
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
