#![cfg(feature = "hdf5")]

use sorrel_io::{ClusterId, DataProvider, HasQualityMetrics, NwbOpenParams, NwbProvider};
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

#[test]
fn native_export_round_trips_curated_units_without_electrodes() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input.nwb");
    let output = temp.path().join("output.nwb");
    fs::copy(fixtures().join("curated_units.nwb"), &input).unwrap();
    let source = NwbProvider::open(&input).unwrap();
    export(&input, &output, &[]);
    let reopened = NwbProvider::open(&output).unwrap();
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
        assert_eq!(reopened.spike_times(c), source.spike_times(c));
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
    for (fixture, params, options) in [
        ("timestamps.nwb", NwbOpenParams::default(), vec![]),
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
                ..Default::default()
            },
            vec!["--dat", raw.to_str().unwrap()],
        ),
    ] {
        let input = temp.path().join(fixture);
        let output = input.with_extension("export.nwb");
        fs::copy(fixtures().join(fixture), &input).unwrap();
        let source = NwbProvider::open_with(&input, params).unwrap();
        export(&input, &output, &options);
        let reopened = NwbProvider::open(&output).unwrap();
        assert_eq!(
            reopened.starting_time(),
            source.starting_time(),
            "{fixture}"
        );
        assert_eq!(
            reopened.exact_sample_rate().unwrap(),
            source.exact_sample_rate().unwrap(),
            "{fixture}"
        );
        let active: Vec<_> = (0..source.n_clusters())
            .map(ClusterId)
            .filter(|&c| !source.spike_times(c).is_empty())
            .collect();
        assert_eq!(reopened.n_clusters() as usize, active.len(), "{fixture}");
        for (row, c) in active.into_iter().enumerate() {
            let r = ClusterId(row as u32);
            assert_eq!(reopened.unit_id(r), source.unit_id(c), "{fixture}");
            assert_eq!(reopened.spike_times(r), source.spike_times(c), "{fixture}");
        }
    }
}
