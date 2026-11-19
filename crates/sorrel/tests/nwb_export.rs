#![cfg(feature = "hdf5")]

use sorrel_io::{
    ClusterId, DataProvider, HasQualityMetrics, NwbOpenParams, NwbProvider, SampleIndex,
};
use std::{fs, path::Path, process::Command};

fn fixtures() -> &'static Path {
    Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../release/fixtures/nwb"
    ))
}

fn export(source: &Path, dest: &Path, options: &[&str]) {
    let output = Command::new(env!("CARGO_BIN_EXE_sorrel"))
        .arg("export-nwb")
        .arg(source)
        .arg("--out")
        .arg(dest)
        .args(options)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn spike_seconds(provider: &NwbProvider, unit: ClusterId) -> Vec<f64> {
    let rate = provider.exact_sample_rate().unwrap();
    provider
        .spike_times(unit)
        .iter()
        .map(|s| provider.starting_time() + s.0 as f64 / rate)
        .collect()
}

#[test]
fn native_export_round_trips_curated_units_without_electrodes() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input.nwb");
    let output = temp.path().join("output.nwb");
    fs::copy(fixtures().join("curated_units.nwb"), &input).unwrap();
    let source = NwbProvider::open(&input).unwrap();
    export(&input, &output, &[]);
    let reopened = NwbProvider::open(&output).unwrap();
    assert_eq!(reopened.starting_time(), 0.0);
    assert_eq!(reopened.n_channels(), 0);
    assert_eq!(reopened.n_clusters(), source.n_clusters());
    assert_eq!(reopened.initial_labels(), source.initial_labels());
    assert_eq!(reopened.initial_tags(), source.initial_tags());
    let annotations = source.initial_annotations();
    assert_eq!(annotations[1].parents, [ClusterId(0)]);
    assert_eq!(annotations[1].missing_parent_unit_ids, [77]);
    assert_eq!(reopened.initial_annotations(), annotations);
    for c in (0..source.n_clusters()).map(ClusterId) {
        assert_eq!(reopened.unit_id_json(c), source.unit_id_json(c));
        assert_eq!(reopened.unit_id(c), source.unit_id(c));
        assert_eq!(spike_seconds(&reopened, c), spike_seconds(&source, c));
    }
    let metrics = source.quality_metrics();
    let actual = reopened.quality_metrics();
    assert_eq!(actual.names(), metrics.names());
    for name in metrics.names() {
        let expected = metrics.get(name).unwrap();
        let actual = actual.get(name).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (&a, &b) in actual.iter().zip(expected) {
            assert!(a == b || (a.is_nan() && b.is_nan()), "{name}: {a} != {b}");
        }
    }
}

#[test]
fn native_export_round_trips_reader_time_bases() {
    let temp = tempfile::tempdir().unwrap();
    let raw = fixtures().join("rounded_raw.nwb");
    let precise = 30000.123;
    for (fixture, params, options) in [
        ("timestamps.nwb", NwbOpenParams::default(), vec![]),
        ("precise_rate_short.nwb", NwbOpenParams::default(), vec![]),
        (
            "legacy_series.nwb",
            NwbOpenParams {
                series: Some("Z_second"),
                ..Default::default()
            },
            vec!["--nwb-series", "Z_second"],
        ),
        (
            "precise_units.nwb",
            NwbOpenParams {
                raw_path: Some(&raw),
                sample_rate: Some(precise),
                ..Default::default()
            },
            vec!["--dat", raw.to_str().unwrap(), "--sample-rate", "30000.123"],
        ),
    ] {
        let input = temp.path().join(fixture);
        let output = input.with_extension("export.nwb");
        fs::copy(fixtures().join(fixture), &input).unwrap();
        let source = NwbProvider::open_with(&input, params).unwrap();
        export(&input, &output, &options);
        let reopened = NwbProvider::open(&output).unwrap();
        assert_eq!(reopened.starting_time(), 0.0, "{fixture}");
        let rate = source.exact_sample_rate().unwrap();
        assert_eq!(
            reopened.exact_sample_rate().unwrap(),
            rate.recip().recip(),
            "{fixture}"
        );
        if fixture == "precise_rate_short.nwb" {
            assert_eq!(rate, f64::from(precise as f32));
            assert_eq!(reopened.spike_times(ClusterId(0)), [SampleIndex(100)]);
        }
        let paired = NwbProvider::open_with(
            &output,
            NwbOpenParams {
                raw_path: params.raw_path.or(Some(&input)),
                ..params
            },
        )
        .unwrap();
        assert_eq!(paired.starting_time(), source.starting_time(), "{fixture}");
        assert_eq!(paired.exact_sample_rate().unwrap(), rate, "{fixture}");
        let active: Vec<_> = (0..source.n_clusters())
            .map(ClusterId)
            .filter(|&c| !source.spike_times(c).is_empty())
            .collect();
        assert_eq!(reopened.n_clusters() as usize, active.len(), "{fixture}");
        for (row, c) in active.into_iter().enumerate() {
            let r = ClusterId(row as u32);
            assert_eq!(reopened.unit_id(r), source.unit_id(c), "{fixture}");
            assert_eq!(paired.spike_times(r), source.spike_times(c), "{fixture}");
            let expected = spike_seconds(&source, c);
            let relative = spike_seconds(&reopened, r);
            assert_eq!(relative.len(), expected.len(), "{fixture}");
            // The units-only sample grid is anchored at zero, not at the raw offset.
            for (relative, expected) in relative.into_iter().zip(expected) {
                assert!((relative - expected).abs() <= 0.5 / rate, "{fixture}");
            }
        }
    }
}

#[test]
fn native_export_long_recording_requires_explicit_rate_when_paired() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("long.nwb");
    let output = temp.path().join("export.nwb");
    fs::copy(fixtures().join("precise_rate_long.nwb"), &input).unwrap();
    export(&input, &output, &["--sample-rate", "30000.123"]);
    let expected = [SampleIndex(330_000_000)];
    let params = NwbOpenParams {
        raw_path: Some(&input),
        ..Default::default()
    };
    let err = NwbProvider::open_with(&output, params).err().unwrap();
    assert!(format!("{err:#}").contains("Ambiguous NWB sample rate"));
    let paired = NwbProvider::open_with(
        &output,
        NwbOpenParams {
            sample_rate: Some(30000.123),
            ..params
        },
    )
    .unwrap();
    assert_eq!(paired.exact_sample_rate().unwrap(), 30000.123);
    assert_eq!(paired.spike_times(ClusterId(0)), expected);
    let standalone = NwbProvider::open(&output).unwrap();
    assert_eq!(standalone.spike_times(ClusterId(0)), expected);
}

#[test]
fn native_export_refuses_lossy_spike_times() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("short.nwb");
    let output = temp.path().join("export.nwb");
    fs::copy(fixtures().join("precise_rate_short.nwb"), &input).unwrap();
    // At this offset the spacing between f64 timestamps exceeds one sample.
    let result = Command::new(env!("CARGO_BIN_EXE_sorrel"))
        .arg("export-nwb")
        .arg(&input)
        .arg("--out")
        .arg(&output)
        .args(["--starting-time", "1099511627776"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("cannot represent sample 100"));
    assert!(!output.exists());
}
