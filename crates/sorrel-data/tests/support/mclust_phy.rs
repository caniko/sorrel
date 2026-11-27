use sorrel_io::npy::{write_1d_u32_atomic, write_1d_u64_atomic, write_f32_atomic, write_rows_tmp};
use std::{fs, path::Path};

pub const SPIKES_PER_UNIT: usize = 48;
pub const DUPLICATE_TIMES: [u64; 2] = [899, 1_206];

/// Two templates see the same physical channels in reverse order. The channels
/// have very different PC baselines so using local PC slots instead of physical
/// channels cannot pass the overlapping-cloud test.
pub fn write_phy(root: &Path, separation: f32) {
    fs::create_dir(root).unwrap();
    fs::write(
        root.join("params.py"),
        "dat_path = 'recording.dat'\nn_channels_dat = 2\nsample_rate = 30000.\ndtype = 'int16'\n",
    )
    .unwrap();
    fs::write(root.join("recording.dat"), vec![0; 16_000 * 2 * 2]).unwrap();

    let mut times = Vec::new();
    let mut clusters = Vec::new();
    let mut amps = Vec::new();
    let mut pcs = Vec::new();
    for i in 0..SPIKES_PER_UNIT {
        let t = 900 + i as u64 * 300;
        let x = (i % 8) as f32 - 3.5;
        let y = 50.0 + (i / 8) as f32 - 2.5;
        // One weaker spike before its partner (one sample), another
        // after its partner (exactly 0.2 ms). Neither is first by amplitude.
        for (time, cluster, amp, pc) in [
            (t - 1, 0, 0.25, [x, y]),
            (t, 0, 5.0 + i as f32, [x, y]),
            (t + 6, 0, 0.5, [x, y]),
            (t + 100, 1, 10.0 + i as f32, [y, x + separation]),
        ]
        .into_iter()
        .enumerate()
        .filter_map(|(slot, spike)| match slot {
            0 if i != 0 => None,
            2 if i != 1 => None,
            _ => Some(spike),
        }) {
            times.push(time);
            clusters.push(cluster);
            amps.push(amp);
            pcs.extend(pc);
        }
    }
    let n = times.len();
    write_1d_u64_atomic(&root.join("spike_times.npy"), &times).unwrap();
    for name in ["spike_clusters.npy", "spike_templates.npy"] {
        write_1d_u32_atomic(&root.join(name), &clusters).unwrap();
    }
    write_f32_atomic(&root.join("amplitudes.npy"), &[n], &amps).unwrap();
    write_f32_atomic(&root.join("pc_features.npy"), &[n, 1, 2], &pcs).unwrap();
    let ind = root.join("pc_feature_ind.npy");
    let tmp = write_rows_tmp(
        &ind,
        "<u4",
        &[2],
        [[0u32, 1], [1, 0]].into_iter().map(|row| {
            row.into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>()
        }),
    )
    .unwrap();
    fs::rename(tmp, ind).unwrap();
    write_f32_atomic(
        &root.join("templates.npy"),
        &[2, 5, 2],
        &[
            0., 0., -2., -1., -10., -4., 5., 2., 0., 0., 0., 0., -1., -2., -4., -10., 2., 5., 0.,
            0.,
        ],
    )
    .unwrap();
    write_f32_atomic(
        &root.join("channel_positions.npy"),
        &[2, 2],
        &[0., 0., 20., 30.],
    )
    .unwrap();
}
