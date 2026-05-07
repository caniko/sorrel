//! End-to-end test for the Kilosort backend: writes a minimal phy2 fixture
//! to a tempdir and exercises `DataProvider` through `KilosortProvider`.

use sorrel_io::kilosort::{KilosortOpenParams, KilosortProvider, PhyLabel};
use sorrel_io::{
    ChannelId, ClusterId, DataProvider, HasAmplitudes, HasGeometry, HasSpikeTemplates,
    SampleIndex, TraceDtype, TraceSamples,
};
use std::io::Write;
use std::path::Path;

fn write_npy_v1(path: &Path, descr: &str, shape_dim: usize, data: &[u8]) {
    let shape = format!("({shape_dim},)");
    write_npy_with_shape(path, descr, &shape, data);
}

fn write_npy_v1_2d(path: &Path, descr: &str, rows: usize, cols: usize, data: &[u8]) {
    let shape = format!("({rows}, {cols})");
    write_npy_with_shape(path, descr, &shape, data);
}

fn write_npy_with_shape(path: &Path, descr: &str, shape_str: &str, data: &[u8]) {
    let dict = format!(
        "{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_str}, }}"
    );
    let prelude_len = 6 + 2 + 2 + dict.len() + 1;
    let pad = (64 - (prelude_len % 64)) % 64;
    let mut header = dict.into_bytes();
    header.extend(std::iter::repeat(b' ').take(pad));
    header.push(b'\n');
    let header_len = header.len() as u16;

    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(b"\x93NUMPY").unwrap();
    f.write_all(&[1u8, 0u8]).unwrap();
    f.write_all(&header_len.to_le_bytes()).unwrap();
    f.write_all(&header).unwrap();
    f.write_all(data).unwrap();
}

#[test]
fn opens_minimal_kilosort_fixture() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    // 6 spikes across 3 clusters (max id = 2 -> n_clusters = 3).
    let times: [u64; 6] = [10, 50, 30, 100, 200, 150];
    let clusters: [u32; 6] = [0, 1, 0, 2, 1, 0];

    let times_bytes: Vec<u8> = times.iter().flat_map(|t| t.to_le_bytes()).collect();
    let clusters_bytes: Vec<u8> = clusters.iter().flat_map(|c| c.to_le_bytes()).collect();
    write_npy_v1(&root.join("spike_times.npy"), "<i64", times.len(), &times_bytes);
    write_npy_v1(
        &root.join("spike_clusters.npy"),
        "<u32",
        clusters.len(),
        &clusters_bytes,
    );

    std::fs::write(
        root.join("cluster_group.tsv"),
        "cluster_id\tgroup\n0\tgood\n2\tnoise\n",
    )
    .unwrap();

    // Raw .dat: 4 channels * 8 samples of i16, interleaved, monotonic.
    let n_channels = 4u32;
    let n_samples = 8usize;
    let mut dat: Vec<u8> = Vec::new();
    for s in 0..n_samples {
        for ch in 0..n_channels as usize {
            let v: i16 = (s * 10 + ch) as i16;
            dat.extend_from_slice(&v.to_le_bytes());
        }
    }
    let dat_path = root.join("recording.dat");
    std::fs::write(&dat_path, &dat).unwrap();

    let p = KilosortProvider::open(
        root,
        KilosortOpenParams {
            sample_rate: Some(30_000.0),
            n_channels: Some(n_channels),
            dat_path: Some(dat_path.clone()),
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(p.sample_rate(), 30_000.0);
    assert_eq!(p.n_channels(), n_channels);
    assert_eq!(p.n_samples(), SampleIndex(n_samples as u64));
    assert_eq!(p.n_clusters(), 3);
    assert_eq!(p.dtype(), TraceDtype::I16);

    // spike_times per cluster, sorted ascending.
    assert_eq!(
        p.spike_times(ClusterId(0)),
        &[SampleIndex(10), SampleIndex(30), SampleIndex(150)]
    );
    assert_eq!(p.spike_times(ClusterId(1)), &[SampleIndex(50), SampleIndex(200)]);
    assert_eq!(p.spike_times(ClusterId(2)), &[SampleIndex(100)]);
    assert!(p.spike_times(ClusterId(99)).is_empty());

    // initial labels from the TSV.
    let labels = p.initial_labels();
    assert_eq!(labels, vec![PhyLabel::Good, PhyLabel::Unsorted, PhyLabel::Noise]);

    // trace slice contents and clamping.
    let slice = p.trace(SampleIndex(0), 2);
    assert_eq!(slice.start, SampleIndex(0));
    assert_eq!(slice.n_channels, n_channels);
    let TraceSamples::I16(s) = slice.samples else {
        panic!("expected i16 samples, got {:?}", slice.samples.dtype());
    };
    assert_eq!(s.len(), 2 * n_channels as usize);
    assert_eq!(s[0], 0);
    assert_eq!(s[1], 1);
    assert_eq!(s[4], 10);

    // out-of-range request clamps to end without panic.
    let clamped = p.trace(SampleIndex(n_samples as u64 - 1), 100);
    assert_eq!(clamped.samples.len(), n_channels as usize);
}

#[test]
fn rejects_dat_with_wrong_size_for_channel_count() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    let times: [u64; 1] = [0];
    let clusters: [u32; 1] = [0];
    let tb: Vec<u8> = times.iter().flat_map(|t| t.to_le_bytes()).collect();
    let cb: Vec<u8> = clusters.iter().flat_map(|c| c.to_le_bytes()).collect();
    write_npy_v1(&root.join("spike_times.npy"), "<i64", times.len(), &tb);
    write_npy_v1(&root.join("spike_clusters.npy"), "<u32", clusters.len(), &cb);

    let dat_path = root.join("recording.dat");
    // 7 bytes — not divisible by 4 channels * 2 bytes = 8.
    std::fs::write(&dat_path, [0u8; 7]).unwrap();

    let res = KilosortProvider::open(
        root,
        KilosortOpenParams {
            sample_rate: Some(30_000.0),
            n_channels: Some(4),
            dat_path: Some(dat_path),
            ..Default::default()
        },
    );
    assert!(res.is_err());
}

#[test]
fn reads_params_py_when_present() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    let times: [u64; 2] = [0, 100];
    let clusters: [u32; 2] = [0, 1];
    let tb: Vec<u8> = times.iter().flat_map(|t| t.to_le_bytes()).collect();
    let cb: Vec<u8> = clusters.iter().flat_map(|c| c.to_le_bytes()).collect();
    write_npy_v1(&root.join("spike_times.npy"), "<i64", times.len(), &tb);
    write_npy_v1(&root.join("spike_clusters.npy"), "<u32", clusters.len(), &cb);

    // 4 ch × 4 samples × 2 bytes = 32 bytes
    std::fs::write(root.join("recording.dat"), [0u8; 32]).unwrap();

    std::fs::write(
        root.join("params.py"),
        "dat_path = r'recording.dat'\n\
         n_channels_dat = 4\n\
         dtype = 'int16'\n\
         offset = 0\n\
         sample_rate = 25000.\n\
         hp_filtered = False\n",
    )
    .unwrap();

    // No overrides at all — everything should come from params.py.
    let p = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
    assert_eq!(p.sample_rate(), 25_000.0);
    assert_eq!(p.n_channels(), 4);
    assert_eq!(p.n_samples(), SampleIndex(4));
    assert_eq!(p.dtype(), TraceDtype::I16);
}

#[test]
fn float32_dtype_round_trips_through_provider() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    let times: [u64; 1] = [0];
    let clusters: [u32; 1] = [0];
    let tb: Vec<u8> = times.iter().flat_map(|t| t.to_le_bytes()).collect();
    let cb: Vec<u8> = clusters.iter().flat_map(|c| c.to_le_bytes()).collect();
    write_npy_v1(&root.join("spike_times.npy"), "<i64", times.len(), &tb);
    write_npy_v1(&root.join("spike_clusters.npy"), "<u32", clusters.len(), &cb);

    // 2 ch × 3 samples × 4 bytes = 24 bytes; values increase per-sample.
    let nc = 2u32;
    let n_samples = 3usize;
    let mut bytes: Vec<u8> = Vec::with_capacity(nc as usize * n_samples * 4);
    for s in 0..n_samples {
        for ch in 0..nc as usize {
            let v: f32 = (s as f32) + (ch as f32) * 0.1;
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(root.join("recording.dat"), &bytes).unwrap();

    std::fs::write(
        root.join("params.py"),
        "dat_path = r'recording.dat'\n\
         n_channels_dat = 2\n\
         dtype = 'float32'\n\
         sample_rate = 1000.\n",
    )
    .unwrap();

    let p = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
    assert_eq!(p.dtype(), TraceDtype::F32);
    let slice = p.trace(SampleIndex(0), 3);
    let TraceSamples::F32(s) = slice.samples else {
        panic!("expected f32 samples");
    };
    assert_eq!(s.len(), 6);
    assert!((s[0] - 0.0).abs() < 1e-6);
    assert!((s[1] - 0.1).abs() < 1e-6);
    assert!((s[2] - 1.0).abs() < 1e-6);
}

#[test]
fn dat_offset_is_skipped() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    let times: [u64; 1] = [0];
    let clusters: [u32; 1] = [0];
    let tb: Vec<u8> = times.iter().flat_map(|t| t.to_le_bytes()).collect();
    let cb: Vec<u8> = clusters.iter().flat_map(|c| c.to_le_bytes()).collect();
    write_npy_v1(&root.join("spike_times.npy"), "<i64", times.len(), &tb);
    write_npy_v1(&root.join("spike_clusters.npy"), "<u32", clusters.len(), &cb);

    // 16 bytes header + 2 ch × 4 samples × 2 bytes = 32 bytes payload.
    let mut bytes = vec![0xFFu8; 16]; // pretend header
    let nc = 2u32;
    let n_samples = 4usize;
    for s in 0..n_samples {
        for ch in 0..nc as usize {
            let v: i16 = (s * 10 + ch) as i16;
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(root.join("recording.dat"), &bytes).unwrap();

    std::fs::write(
        root.join("params.py"),
        "n_channels_dat = 2\nsample_rate = 1000.\noffset = 16\n",
    )
    .unwrap();

    let p = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
    assert_eq!(p.n_samples(), SampleIndex(n_samples as u64));
    let slice = p.trace(SampleIndex(0), 1);
    let TraceSamples::I16(s) = slice.samples else {
        panic!("expected i16 samples");
    };
    // First sample after the 16-byte preamble: ch0=0, ch1=1.
    assert_eq!(s[0], 0);
    assert_eq!(s[1], 1);
}

/// Build a fixture with 6 spikes split across 3 clusters, with amplitudes and
/// per-spike templates also written so the geometry/amps/templates traits can
/// all be exercised in one test.
fn write_full_phy_fixture(root: &Path) {
    // Spike order in the global arrays — deliberately *not* sorted by time so
    // the test catches incorrect bucketing.
    let times: [u64; 6] = [10, 50, 30, 100, 200, 150];
    let clusters: [u32; 6] = [0, 1, 0, 2, 1, 0];
    let amps: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let templates: [u32; 6] = [10, 11, 10, 12, 11, 10];

    let tb: Vec<u8> = times.iter().flat_map(|t| t.to_le_bytes()).collect();
    let cb: Vec<u8> = clusters.iter().flat_map(|c| c.to_le_bytes()).collect();
    let ab: Vec<u8> = amps.iter().flat_map(|a| a.to_le_bytes()).collect();
    let pb: Vec<u8> = templates.iter().flat_map(|p| p.to_le_bytes()).collect();
    write_npy_v1(&root.join("spike_times.npy"), "<i64", times.len(), &tb);
    write_npy_v1(&root.join("spike_clusters.npy"), "<u32", clusters.len(), &cb);
    write_npy_v1(&root.join("amplitudes.npy"), "<f4", amps.len(), &ab);
    write_npy_v1(&root.join("spike_templates.npy"), "<i4", templates.len(), &pb);

    // 4 channels, monotonic dat, params.py drives shape.
    let nc = 4u32;
    let n_samples = 8usize;
    let mut dat: Vec<u8> = Vec::new();
    for s in 0..n_samples {
        for ch in 0..nc as usize {
            let v: i16 = (s * 10 + ch) as i16;
            dat.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(root.join("recording.dat"), &dat).unwrap();
    std::fs::write(
        root.join("params.py"),
        "n_channels_dat = 4\nsample_rate = 1000.\ndtype = 'int16'\n",
    )
    .unwrap();

    // Probe geometry: 4 channels in a vertical line; one shank.
    let positions: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 20.0], [0.0, 40.0], [0.0, 60.0]];
    let pos_bytes: Vec<u8> = positions
        .iter()
        .flat_map(|p| p.iter().flat_map(|v| v.to_le_bytes()))
        .collect();
    write_npy_v1_2d(&root.join("channel_positions.npy"), "<f4", 4, 2, &pos_bytes);
    let shanks: [u32; 4] = [0, 0, 0, 0];
    let shanks_bytes: Vec<u8> = shanks.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_npy_v1(&root.join("channel_shanks.npy"), "<i4", 4, &shanks_bytes);
    let cmap: [u32; 4] = [0, 1, 2, 3];
    let cmap_bytes: Vec<u8> = cmap.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_npy_v1(&root.join("channel_map.npy"), "<i4", 4, &cmap_bytes);
}

#[test]
fn amplitudes_and_templates_align_with_time_sorted_spikes() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_full_phy_fixture(root);

    let p = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();

    // Cluster 0 has spikes at global indices [0,2,5] with times [10,30,150]
    // and amps [1.0,3.0,6.0]. After time-sort the order is [10,30,150].
    assert_eq!(
        p.spike_times(ClusterId(0)),
        &[SampleIndex(10), SampleIndex(30), SampleIndex(150)]
    );
    assert_eq!(p.spike_amplitudes(ClusterId(0)), &[1.0, 3.0, 6.0]);
    assert_eq!(p.spike_templates(ClusterId(0)), &[10, 10, 10]);

    // Cluster 1 has [1,4] -> times [50,200], amps [2.0,5.0], templates
    // [11,11] (already in time order).
    assert_eq!(p.spike_times(ClusterId(1)), &[SampleIndex(50), SampleIndex(200)]);
    assert_eq!(p.spike_amplitudes(ClusterId(1)), &[2.0, 5.0]);
    assert_eq!(p.spike_templates(ClusterId(1)), &[11, 11]);

    // Cluster 2: single spike.
    assert_eq!(p.spike_times(ClusterId(2)), &[SampleIndex(100)]);
    assert_eq!(p.spike_amplitudes(ClusterId(2)), &[4.0]);
    assert_eq!(p.spike_templates(ClusterId(2)), &[12]);

    // Out-of-range clusters return empty slices, never panic.
    assert!(p.spike_amplitudes(ClusterId(99)).is_empty());
    assert!(p.spike_templates(ClusterId(99)).is_empty());
}

#[test]
fn channel_geometry_loads_when_files_present() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_full_phy_fixture(root);

    let p = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
    assert_eq!(p.channel_positions().len(), 4);
    assert_eq!(p.channel_positions()[0], [0.0, 0.0]);
    assert_eq!(p.channel_positions()[3], [0.0, 60.0]);
    assert_eq!(p.channel_shanks(), &[0, 0, 0, 0]);
    assert_eq!(
        p.channel_map(),
        &[ChannelId(0), ChannelId(1), ChannelId(2), ChannelId(3)]
    );
}

#[test]
fn missing_optional_files_yield_empty_slices() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    // Minimal phy directory with no amplitudes / templates / geometry.
    let times: [u64; 1] = [0];
    let clusters: [u32; 1] = [0];
    let tb: Vec<u8> = times.iter().flat_map(|t| t.to_le_bytes()).collect();
    let cb: Vec<u8> = clusters.iter().flat_map(|c| c.to_le_bytes()).collect();
    write_npy_v1(&root.join("spike_times.npy"), "<i64", times.len(), &tb);
    write_npy_v1(&root.join("spike_clusters.npy"), "<u32", clusters.len(), &cb);
    std::fs::write(root.join("recording.dat"), [0u8; 4]).unwrap(); // 1 ch × 2 samples
    std::fs::write(
        root.join("params.py"),
        "n_channels_dat = 1\nsample_rate = 1000.\n",
    )
    .unwrap();

    let p = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
    assert!(p.spike_amplitudes(ClusterId(0)).is_empty());
    assert!(p.spike_templates(ClusterId(0)).is_empty());
    assert!(p.channel_positions().is_empty());
    assert!(p.channel_shanks().is_empty());
    assert!(p.channel_map().is_empty());
}

#[test]
fn rejects_amplitudes_with_wrong_length() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    // 2 spikes but amplitudes file has 3 entries.
    let times: [u64; 2] = [0, 100];
    let clusters: [u32; 2] = [0, 0];
    let amps: [f32; 3] = [1.0, 2.0, 3.0];
    let tb: Vec<u8> = times.iter().flat_map(|t| t.to_le_bytes()).collect();
    let cb: Vec<u8> = clusters.iter().flat_map(|c| c.to_le_bytes()).collect();
    let ab: Vec<u8> = amps.iter().flat_map(|a| a.to_le_bytes()).collect();
    write_npy_v1(&root.join("spike_times.npy"), "<i64", times.len(), &tb);
    write_npy_v1(&root.join("spike_clusters.npy"), "<u32", clusters.len(), &cb);
    write_npy_v1(&root.join("amplitudes.npy"), "<f4", amps.len(), &ab);

    std::fs::write(root.join("recording.dat"), [0u8; 4]).unwrap();
    std::fs::write(
        root.join("params.py"),
        "n_channels_dat = 1\nsample_rate = 1000.\n",
    )
    .unwrap();

    let res = KilosortProvider::open(root, KilosortOpenParams::default());
    assert!(res.is_err(), "expected length-mismatch error");
}

#[test]
fn templates_npy_round_trips_through_provider() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    // Minimal phy directory.
    let times: [u64; 2] = [0, 100];
    let clusters: [u32; 2] = [0, 0];
    let tb: Vec<u8> = times.iter().flat_map(|t| t.to_le_bytes()).collect();
    let cb: Vec<u8> = clusters.iter().flat_map(|c| c.to_le_bytes()).collect();
    write_npy_v1(&root.join("spike_times.npy"), "<i64", times.len(), &tb);
    write_npy_v1(&root.join("spike_clusters.npy"), "<u32", clusters.len(), &cb);
    std::fs::write(root.join("recording.dat"), [0u8; 8]).unwrap();
    std::fs::write(
        root.join("params.py"),
        "n_channels_dat = 2\nsample_rate = 1000.\n",
    )
    .unwrap();

    // Templates: 3 templates × 5 samples × 2 channels.
    let n_t = 3usize;
    let n_s = 5usize;
    let n_c = 2usize;
    let mut tpl_values: Vec<f32> = Vec::with_capacity(n_t * n_s * n_c);
    for t in 0..n_t {
        for s in 0..n_s {
            for c in 0..n_c {
                tpl_values.push((t * 100 + s * 10 + c) as f32);
            }
        }
    }
    let tpl_bytes: Vec<u8> = tpl_values.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_npy_v1_3d(
        &root.join("templates.npy"),
        "<f4",
        n_t,
        n_s,
        n_c,
        &tpl_bytes,
    );

    // Similar templates: identity matrix as a placeholder.
    let mut sim: Vec<f32> = vec![0.0; n_t * n_t];
    for i in 0..n_t {
        sim[i * n_t + i] = 1.0;
    }
    let sim_bytes: Vec<u8> = sim.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_npy_v1_2d(&root.join("similar_templates.npy"), "<f4", n_t, n_t, &sim_bytes);

    let p = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();

    use sorrel_io::HasTemplateWaveforms;
    assert_eq!(p.template_shape(), (n_t, n_s, n_c));
    assert_eq!(p.template_waveforms().len(), n_t * n_s * n_c);
    assert_eq!(p.template_waveforms()[0], 0.0);
    assert_eq!(p.template_waveforms()[1], 1.0); // first row, second channel
    assert_eq!(p.similar_templates().len(), n_t * n_t);
    assert_eq!(p.similar_templates()[0], 1.0);
    assert_eq!(p.similar_templates()[1], 0.0);
}

#[test]
fn templates_with_mismatched_similar_count_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_npy_v1(&root.join("spike_times.npy"), "<i64", 0, &[]);
    write_npy_v1(&root.join("spike_clusters.npy"), "<u32", 0, &[]);
    std::fs::write(root.join("recording.dat"), [0u8; 4]).unwrap();
    std::fs::write(
        root.join("params.py"),
        "n_channels_dat = 1\nsample_rate = 1000.\n",
    )
    .unwrap();

    // 2 templates of 3 samples × 1 channel.
    let n_t = 2usize;
    let n_s = 3usize;
    let n_c = 1usize;
    let zeros = vec![0u8; n_t * n_s * n_c * 4];
    write_npy_v1_3d(&root.join("templates.npy"), "<f4", n_t, n_s, n_c, &zeros);
    // 3×3 similar matrix instead of the expected 2×2.
    let sim_bytes = vec![0u8; 3 * 3 * 4];
    write_npy_v1_2d(&root.join("similar_templates.npy"), "<f4", 3, 3, &sim_bytes);

    let res = KilosortProvider::open(root, KilosortOpenParams::default());
    assert!(res.is_err(), "should reject mismatched template/sim shapes");
}

fn write_npy_v1_3d(
    path: &Path,
    descr: &str,
    d0: usize,
    d1: usize,
    d2: usize,
    data: &[u8],
) {
    let shape = format!("({d0}, {d1}, {d2})");
    write_npy_with_shape(path, descr, &shape, data);
}

/// Helper: minimal phy fixture with `n_clusters` clusters and one spike each.
fn write_minimal_fixture(root: &Path, n_clusters: u32, n_channels: u32) {
    let times: Vec<u64> = (0..n_clusters as u64).map(|i| i * 10).collect();
    let clusters: Vec<u32> = (0..n_clusters).collect();
    let tb: Vec<u8> = times.iter().flat_map(|t| t.to_le_bytes()).collect();
    let cb: Vec<u8> = clusters.iter().flat_map(|c| c.to_le_bytes()).collect();
    write_npy_v1(&root.join("spike_times.npy"), "<i64", times.len(), &tb);
    write_npy_v1(&root.join("spike_clusters.npy"), "<u32", clusters.len(), &cb);

    let dat_bytes = vec![0u8; n_channels as usize * 4 * 2];
    std::fs::write(root.join("recording.dat"), &dat_bytes).unwrap();
    std::fs::write(
        root.join("params.py"),
        format!("n_channels_dat = {n_channels}\nsample_rate = 1000.\n"),
    )
    .unwrap();
}

#[test]
fn quality_metrics_si_csv_round_trips() {
    use sorrel_io::HasQualityMetrics;
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_minimal_fixture(root, 3, 2);
    std::fs::write(
        root.join("quality_metrics.csv"),
        "cluster_id,snr,firing_rate,isi_viol_ratio\n0,5.0,12.3,0.01\n1,8.5,7.7,0.0\n2,2.1,1.5,0.5\n",
    )
    .unwrap();

    let p = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
    let names = p.metric_names();
    assert!(names.contains(&"snr".to_string()));
    assert!(names.contains(&"firing_rate".to_string()));
    assert!(names.contains(&"isi_viol_ratio".to_string()));

    let snr = p.metric_values("snr").unwrap();
    assert_eq!(snr.len(), 3);
    assert!((snr[0] - 5.0).abs() < 1e-6);
    assert!((snr[1] - 8.5).abs() < 1e-6);
    assert!((snr[2] - 2.1).abs() < 1e-6);

    assert!(p.metric_values("nope_not_a_metric").is_none());
}

#[test]
fn quality_metrics_phy_per_metric_tsv_loads() {
    use sorrel_io::HasQualityMetrics;
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_minimal_fixture(root, 4, 2);
    // phy convention: cluster_<metric>.tsv with cluster_id<TAB><metric>
    std::fs::write(
        root.join("cluster_amp.tsv"),
        "cluster_id\tamp\n0\t10.5\n2\t7.0\n",
    )
    .unwrap();

    let p = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
    let amp = p.metric_values("amp").unwrap();
    assert_eq!(amp.len(), 4);
    assert!((amp[0] - 10.5).abs() < 1e-6);
    assert!(amp[1].is_nan(), "missing cluster should be NaN, got {}", amp[1]);
    assert!((amp[2] - 7.0).abs() < 1e-6);
    assert!(amp[3].is_nan());
}

#[test]
fn quality_metrics_si_csv_takes_precedence_over_phy_tsv() {
    use sorrel_io::HasQualityMetrics;
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_minimal_fixture(root, 2, 2);
    std::fs::write(
        root.join("quality_metrics.csv"),
        "cluster_id,snr\n0,99.0\n1,88.0\n",
    )
    .unwrap();
    // Conflicting phy file with different values.
    std::fs::write(
        root.join("cluster_snr.tsv"),
        "cluster_id\tsnr\n0\t1.0\n1\t1.0\n",
    )
    .unwrap();

    let p = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
    let snr = p.metric_values("snr").unwrap();
    assert!((snr[0] - 99.0).abs() < 1e-6);
    assert!((snr[1] - 88.0).abs() < 1e-6);
}

#[test]
fn no_quality_metrics_files_yields_empty_metric_list() {
    use sorrel_io::HasQualityMetrics;
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_minimal_fixture(root, 2, 1);

    let p = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
    assert!(p.metric_names().is_empty());
    assert!(p.metric_values("anything").is_none());
}

#[test]
fn quality_metric_with_missing_value_records_nan() {
    use sorrel_io::HasQualityMetrics;
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_minimal_fixture(root, 3, 1);
    std::fs::write(
        root.join("quality_metrics.csv"),
        "cluster_id,snr\n0,5.0\n2,3.0\n", // skip cluster 1
    )
    .unwrap();

    let p = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
    let snr = p.metric_values("snr").unwrap();
    assert!((snr[0] - 5.0).abs() < 1e-6);
    assert!(snr[1].is_nan(), "missing entry should be NaN");
    assert!((snr[2] - 3.0).abs() < 1e-6);
}

#[test]
fn quality_metrics_skips_out_of_range_cluster_ids() {
    use sorrel_io::HasQualityMetrics;
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_minimal_fixture(root, 2, 1);
    std::fs::write(
        root.join("quality_metrics.csv"),
        "cluster_id,foo\n0,1.0\n1,2.0\n99,9.9\n", // 99 doesn't exist
    )
    .unwrap();

    let p = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
    let foo = p.metric_values("foo").unwrap();
    assert_eq!(foo.len(), 2); // length matches n_clusters, not the CSV
    assert!((foo[0] - 1.0).abs() < 1e-6);
    assert!((foo[1] - 2.0).abs() < 1e-6);
}

#[test]
fn metric_names_are_sorted_alphabetically() {
    use sorrel_io::HasQualityMetrics;
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_minimal_fixture(root, 2, 1);
    std::fs::write(
        root.join("quality_metrics.csv"),
        "cluster_id,zeta,alpha,delta\n0,0.0,0.0,0.0\n1,0.0,0.0,0.0\n",
    )
    .unwrap();

    let p = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
    let names = p.metric_names();
    assert_eq!(names, &["alpha", "delta", "zeta"]);
}
