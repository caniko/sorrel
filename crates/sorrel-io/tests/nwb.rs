//! NWB reader invariants against files written by real PyNWB
//! (`release/fixtures/nwb/generate.py`).
#![cfg(feature = "hdf5")]

use sorrel_io::{ChannelId, ClusterId, DataProvider, HasGeometry, NwbProvider, SampleIndex};
use std::path::PathBuf;

fn open(name: &str) -> anyhow::Result<NwbProvider> {
    NwbProvider::open(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../release/fixtures/nwb")
            .join(name),
    )
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
fn electrical_series_subtypes_are_accepted() {
    let p = open("subtype.nwb").unwrap();
    assert_eq!(p.n_channels(), 4);
    assert_eq!(frame(&p, 0), [-100.0, -100.0, -100.0, -100.0]);
}

#[test]
fn rejects_what_it_cannot_read_faithfully() {
    let e = open_err("not_electrical.nwb");
    assert!(e.contains("ElectricalSeries"), "{e}");
    let e = open_err("float64.nwb");
    assert!(e.contains("float64"), "{e}");
    let e = open_err("timestamps.nwb");
    assert!(e.contains("timestamps"), "{e}");
}
