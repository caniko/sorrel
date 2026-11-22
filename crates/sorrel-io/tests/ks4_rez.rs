//! Legacy MATLAB `rez.mat` reader tests with on-disk HDF5 fixtures.
#![cfg(feature = "hdf5")]

use hdf5_metno as hdf5;
use sorrel_io::{
    ClusterId, DataProvider, HasSpikeTemplates, Ks4RezOpenParams, Ks4RezProvider, SampleIndex,
};
use std::path::Path;

fn write_rez(path: &Path, write_spikes: impl FnOnce(&hdf5::Group)) {
    let file = hdf5::File::create(path).unwrap();
    let rez = file.create_group("rez").unwrap();
    let ops = rez.create_group("ops").unwrap();
    for (name, value) in [("fs", 30_000.0_f64), ("Nchan", 2.0)] {
        ops.new_dataset::<f64>()
            .shape((1, 1))
            .create(name)
            .unwrap()
            .write_raw(&[value])
            .unwrap();
    }
    write_spikes(&rez);
}

#[test]
fn reads_matlab_columns_as_sorted_zero_based_spike_trains() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rez.mat");
    write_rez(&path, |rez| {
        // MATLAB stores the four columns of this six-spike table consecutively.
        let st3 = [
            31.0_f64, 1.0, 11.0, 51.0, 0.0, 61.0, // samples (1-based)
            3.0, 1.0, 3.0, 1.0, 4.0, 0.0, // template IDs (1-based)
            0.8, 1.1, 1.0, 1.3, 0.9, 1.2, // amplitudes (ignored)
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // ignored column
        ];
        rez.new_dataset::<f64>()
            .shape((4, 6))
            .create("st3")
            .unwrap()
            .write_raw(&st3)
            .unwrap();
    });

    let p = Ks4RezProvider::open(&path, Ks4RezOpenParams::default()).unwrap();
    assert_eq!(p.sample_rate(), 30_000.0);
    assert_eq!(p.n_channels(), 2);
    assert_eq!(p.n_clusters(), 3);
    assert_eq!(
        p.spike_times(ClusterId(0)),
        [SampleIndex(0), SampleIndex(50)]
    );
    assert!(p.spike_times(ClusterId(1)).is_empty());
    assert_eq!(
        p.spike_times(ClusterId(2)),
        [SampleIndex(10), SampleIndex(30)]
    );
    assert_eq!(p.spike_templates(ClusterId(0)), [0, 0]);
    assert_eq!(p.spike_templates(ClusterId(2)), [2, 2]);
}

#[test]
fn no_usable_spikes_means_no_clusters() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rez.mat");
    write_rez(&path, |rez| {
        rez.new_dataset::<f64>()
            .shape((2, 1))
            .create("st3")
            .unwrap()
            .write_raw(&[0.0, 1.0])
            .unwrap();
    });

    let p = Ks4RezProvider::open(&path, Ks4RezOpenParams::default()).unwrap();
    assert_eq!(p.n_clusters(), 0);
    assert!(p.initial_labels().is_empty());
}

#[test]
fn rejects_missing_or_malformed_spike_tables() {
    let dir = tempfile::tempdir().unwrap();
    for (name, shape, expected) in [
        ("missing", &[][..], "missing /rez/st3"),
        ("rank", &[4][..], "rez/st3 expected 2-D"),
        ("columns", &[1, 4][..], "rez/st3 has only 1 columns"),
    ] {
        let path = dir.path().join(format!("{name}.mat"));
        write_rez(&path, |rez| {
            if !shape.is_empty() {
                rez.new_dataset::<f64>().shape(shape).create("st3").unwrap();
            }
        });
        let err = Ks4RezProvider::open(&path, Ks4RezOpenParams::default())
            .err()
            .expect("malformed rez.mat must be rejected");
        assert!(err.to_string().contains(expected), "{name}: {err:#}");
    }
}
