//! Headless UI states: drive the real `SorrelApp` with egui_kittest, render
//! it through wgpu, and compare against committed snapshots.
//!
//! `UPDATE_SNAPSHOTS=1 cargo test -p sorrel-ui --test ui_states` accepts new
//! images; diffs land next to the snapshots as `*.diff.png`/`*.new.png`.

use egui::accesskit::Role;
use egui_kittest::kittest::{By, NodeT, Queryable};
use egui_kittest::{Harness, SnapshotResults};
use sorrel_data::{Journal, Session};
use sorrel_io::kilosort::{KilosortOpenParams, PhyLabel};
use sorrel_io::npy;
use sorrel_io::{DataProvider, HasGeometry, KilosortProvider};
use sorrel_ui::{RasterPipeline, SorrelApp, TracePipeline};
use std::path::{Path, PathBuf};
use std::time::Duration;

const SIZE: [f32; 2] = [1280.0, 800.0];
const N_CHANNELS: usize = 16;
const SAMPLE_RATE: f32 = 30_000.0;
const N_SAMPLES: usize = 4 * SAMPLE_RATE as usize;
const N_T: usize = 61;

type App = SorrelApp<KilosortProvider>;

/// Snapshots render on the dev shell's pinned lavapipe (`SORREL_LAVAPIPE_ICD`)
/// so atlas and GPU-less CI runners rasterise identically. The Vulkan loader
/// reads the driver list once, so every test starts here. Outside the dev
/// shell with no GPU the tests skip, like sorrel-gpu's.
fn no_gpu() -> bool {
    static ERR: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let err = ERR.get_or_init(|| {
        if let Some(icd) = std::env::var_os("SORREL_LAVAPIPE_ICD") {
            std::env::set_var("VK_DRIVER_FILES", &icd);
            std::env::set_var("VK_ICD_FILENAMES", &icd);
        }
        sorrel_gpu::GpuContext::headless()
            .err()
            .map(|e| e.to_string())
    });
    if let Some(e) = err {
        eprintln!("skipping UI snapshot test: {e}");
    }
    err.is_some()
}

/// Deterministic xorshift so the fixture never changes between runs.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// A two-column, 20 µm pitch probe with `n_clusters` units: biphasic
/// templates centred on a peak channel, refractory spike trains, PCs that
/// separate by unit, and a raw recording with the spikes added to noise.
fn write_fixture(root: &Path, n_clusters: usize) {
    let mut rng = Rng(0x5eed_0f50_11e1);
    let positions: Vec<f32> = (0..N_CHANNELS)
        .flat_map(|c| [(c % 2) as f32 * 32.0, (c / 2) as f32 * 20.0])
        .collect();
    let peak = |u: usize| (u * 5 + 3) % N_CHANNELS;
    let falloff = |u: usize, ch: usize| {
        let (pu, pc) = (peak(u), ch);
        let dx = positions[2 * pu] - positions[2 * pc];
        let dy = positions[2 * pu + 1] - positions[2 * pc + 1];
        (-(dx * dx + dy * dy) / (2.0 * 30.0f32.powi(2))).exp()
    };

    let mut templates = vec![0f32; n_clusters * N_T * N_CHANNELS];
    for u in 0..n_clusters {
        let width = 3.0 + (u % 4) as f32;
        for t in 0..N_T {
            let x = (t as f32 - 20.0) / width;
            let shape = -(-x * x).exp() + 0.35 * (-(x - 2.5).powi(2) / 2.0).exp();
            for ch in 0..N_CHANNELS {
                templates[(u * N_T + t) * N_CHANNELS + ch] = shape * falloff(u, ch);
            }
        }
    }

    let mut spikes: Vec<(u64, u32, f32)> = Vec::new();
    for u in 0..n_clusters {
        let rate = 4.0 + (u * 7 % 23) as f32;
        let mut t = 100.0;
        loop {
            t += 60.0 + (-(1.0 - rng.next()).ln()) * SAMPLE_RATE / rate;
            if t as usize >= N_SAMPLES - N_T {
                break;
            }
            spikes.push((
                t as u64,
                u as u32,
                40.0 + 15.0 * (u % 5) as f32 + 6.0 * rng.next(),
            ));
        }
    }
    spikes.sort_by_key(|s| s.0);
    let n = spikes.len();

    let mut raw = vec![0f32; N_SAMPLES * N_CHANNELS];
    for v in raw.iter_mut() {
        *v = (rng.next() - 0.5) * 24.0;
    }
    let mut pcs = vec![0f32; n * 3 * 4];
    for (i, &(t, u, amp)) in spikes.iter().enumerate() {
        let u = u as usize;
        for dt in 0..N_T {
            for ch in 0..N_CHANNELS {
                raw[(t as usize + dt - 20) * N_CHANNELS + ch] +=
                    amp * 3.0 * templates[(u * N_T + dt) * N_CHANNELS + ch];
            }
        }
        let (cx, cy) = ((u as f32 * 2.4).cos(), (u as f32 * 2.4).sin());
        for k in 0..4 {
            pcs[i * 12 + k] = cx * (4.0 - k as f32) + rng.next() - 0.5;
            pcs[i * 12 + 4 + k] = cy * (4.0 - k as f32) + rng.next() - 0.5;
            pcs[i * 12 + 8 + k] = rng.next() - 0.5;
        }
    }
    let raw: Vec<u8> = raw
        .iter()
        .flat_map(|v| (v.round().clamp(-32768.0, 32767.0) as i16).to_le_bytes())
        .collect();
    std::fs::write(root.join("recording.dat"), raw).unwrap();

    let times: Vec<u64> = spikes.iter().map(|s| s.0).collect();
    let clusters: Vec<u32> = spikes.iter().map(|s| s.1).collect();
    let amps: Vec<f32> = spikes.iter().map(|s| s.2).collect();
    npy::write_1d_u64_atomic(&root.join("spike_times.npy"), &times).unwrap();
    npy::write_1d_u32_atomic(&root.join("spike_clusters.npy"), &clusters).unwrap();
    npy::write_1d_u32_atomic(&root.join("spike_templates.npy"), &clusters).unwrap();
    npy::write_f32_atomic(&root.join("amplitudes.npy"), &[n], &amps).unwrap();
    npy::write_f32_atomic(&root.join("pc_features.npy"), &[n, 3, 4], &pcs).unwrap();
    let pc_ind: Vec<u32> = (0..n_clusters)
        .flat_map(|u| (0..4).map(move |k| ((peak(u) + k) % N_CHANNELS) as u32))
        .collect();
    npy::write_1d_u32_atomic(
        &root.join("channel_map.npy"),
        &(0..N_CHANNELS as u32).collect::<Vec<_>>(),
    )
    .unwrap();
    npy::write_f32_atomic(
        &root.join("channel_positions.npy"),
        &[N_CHANNELS, 2],
        &positions,
    )
    .unwrap();
    npy::write_f32_atomic(
        &root.join("templates.npy"),
        &[n_clusters, N_T, N_CHANNELS],
        &templates,
    )
    .unwrap();
    let sim: Vec<f32> = (0..n_clusters)
        .flat_map(|a| (0..n_clusters).map(move |b| (a, b)))
        .map(|(a, b)| falloff(a, peak(b)))
        .collect();
    npy::write_f32_atomic(
        &root.join("similar_templates.npy"),
        &[n_clusters, n_clusters],
        &sim,
    )
    .unwrap();
    write_u32_2d(&root.join("pc_feature_ind.npy"), n_clusters, 4, &pc_ind);

    std::fs::write(
        root.join("params.py"),
        format!(
            "dat_path = 'recording.dat'\nn_channels_dat = {N_CHANNELS}\ndtype = 'int16'\n\
             offset = 0\nsample_rate = {SAMPLE_RATE:.1}\nhp_filtered = False\n"
        ),
    )
    .unwrap();
    let groups = ["good", "mua", "noise", "good", "unsorted"];
    let tsv: String = (0..n_clusters)
        .map(|u| format!("{u}\t{}\n", groups[u % groups.len()]))
        .collect();
    std::fs::write(
        root.join("cluster_group.tsv"),
        format!("cluster_id\tgroup\n{tsv}"),
    )
    .unwrap();
}

/// `npy` only writes 1-d u32; pc_feature_ind is (n_templates, 4).
fn write_u32_2d(path: &Path, rows: usize, cols: usize, data: &[u32]) {
    let dict = format!("{{'descr': '<u4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    let mut header = dict.into_bytes();
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(b' ');
    }
    header.push(b'\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend((header.len() as u16).to_le_bytes());
    out.extend(header);
    out.extend(data.iter().flat_map(|v| v.to_le_bytes()));
    std::fs::write(path, out).unwrap();
}

fn label_str(l: &PhyLabel) -> &'static str {
    l.as_str()
}

/// Mirrors `sorrel`'s `open_kilosort` + `seed_kilosort` + `run_app`.
fn open(root: &Path) -> (Session<KilosortProvider>, Vec<[f32; 2]>) {
    let provider = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
    let journal =
        Journal::open_or_create(&root.join("sorrel.journal"), 0, provider.n_clusters()).unwrap();
    let positions = provider.channel_positions().to_vec();
    let mut session = Session::new(provider, journal);
    session.seed_amplitudes();
    session.seed_templates();
    session.seed_pc_features();
    session.seed_template_waveforms();
    session.replay_journal().unwrap();
    (session, positions)
}

fn configure(app: &mut App, root: &Path, positions: Vec<[f32; 2]>) {
    app.set_save_dir(root.to_path_buf(), true);
    app.set_channel_positions(positions);
    app.set_depth_source(sorrel_data::spike_depths);
}

/// kittest's software-first adapter choice, minus GL: the host's GL
/// llvmpipe would otherwise compete with the pinned lavapipe.
fn vulkan_only() -> egui_wgpu::WgpuSetup {
    let mut setup = egui_kittest::wgpu::default_wgpu_setup();
    if let egui_wgpu::WgpuSetup::CreateNew(new) = &mut setup {
        new.instance_descriptor.backends = egui_wgpu::wgpu::Backends::VULKAN;
    }
    setup
}

fn harness_with(
    theme: egui::Theme,
    build: impl FnOnce() -> App + 'static,
) -> Harness<'static, App> {
    Harness::builder()
        .with_size(SIZE)
        .with_theme(theme)
        .wgpu_setup(vulkan_only())
        .build_eframe(move |cc| {
            let mut app = build();
            let rs = cc
                .wgpu_render_state
                .as_ref()
                .expect("kittest wgpu render state");
            TracePipeline::install(rs);
            RasterPipeline::install(rs);
            app.install_gpu_compute(rs.device.clone().into(), rs.queue.clone().into());
            app
        })
}

/// A fresh dataset at a fixed path per `slot`: the top bar prints the
/// save directory, so a random tempdir name would break every snapshot.
fn dataset(slot: &str, n_clusters: usize) -> PathBuf {
    let slot = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("ui_states")
        .join(slot);
    let _ = std::fs::remove_dir_all(&slot);
    let root = slot.join("fixture/synthetic-probe");
    std::fs::create_dir_all(&root).unwrap();
    write_fixture(&root, n_clusters);
    root
}

fn fixture(slot: &str, n_clusters: usize, theme: egui::Theme) -> Harness<'static, App> {
    let root = dataset(slot, n_clusters);
    let mut harness = harness_with(theme, move || {
        let (session, positions) = open(&root);
        let mut app = SorrelApp::new(session, label_str);
        configure(&mut app, &root, positions);
        app
    });
    settle(&mut harness);
    harness
}

/// Views compute on worker threads; step in real time until they land.
// ponytail: fixed 1.5 s settle; replace with a "no jobs pending" hook if it flakes.
fn settle(h: &mut Harness<'static, App>) {
    for _ in 0..30 {
        h.step();
        std::thread::sleep(Duration::from_millis(50));
    }
    h.run_ok();
}

/// Build one harness per theme, let `drive` take its snapshots (`snap`
/// suffixes the theme), and report every theme's diffs together.
fn each_theme(mut drive: impl FnMut(egui::Theme, &str) -> Harness<'static, App>) {
    let mut results = SnapshotResults::new();
    for (theme, name) in [(egui::Theme::Dark, "dark"), (egui::Theme::Light, "light")] {
        let mut h = drive(theme, name);
        results.extend_harness(&mut h);
    }
    results.unwrap();
}

fn snap(h: &mut Harness<'static, App>, state: &str, theme: &str) {
    settle(h);
    h.remove_cursor();
    h.run_ok();
    h.snapshot(format!("{state}_{theme}"));
}

fn click_tab(h: &mut Harness<'static, App>, title: &str) {
    h.get_by_label(title).click();
    settle(h);
}

#[test]
fn dump_accesskit_tree() {
    if no_gpu() {
        return;
    }
    // Debug aid for writing queries:
    // `SORREL_DUMP_TREE=1 cargo test -p sorrel-ui --test ui_states dump -- --nocapture`.
    if std::env::var_os("SORREL_DUMP_TREE").is_some() {
        let mut h = fixture("dump", 3, egui::Theme::Dark);
        h.key_press(egui::Key::J);
        settle(&mut h);
        println!("{:#?}", h.root());
    }
}

#[test]
fn startup() {
    if no_gpu() {
        return;
    }
    each_theme(|theme, name| {
        let mut h = fixture(&format!("startup-{name}"), 3, theme);
        snap(&mut h, "startup", name);
        h
    });
}

#[test]
fn one_selected() {
    if no_gpu() {
        return;
    }
    each_theme(|theme, name| {
        let mut h = fixture(&format!("one-selected-{name}"), 3, theme);
        h.key_press(egui::Key::J);
        snap(&mut h, "one_selected", name);
        h
    });
}

#[test]
fn help_overlay() {
    if no_gpu() {
        return;
    }
    each_theme(|theme, name| {
        let mut h = fixture(&format!("help-{name}"), 3, theme);
        h.key_press(egui::Key::F1);
        snap(&mut h, "help", name);
        h
    });
}

#[test]
fn quit_with_unsaved_edits() {
    if no_gpu() {
        return;
    }
    each_theme(|theme, name| {
        let mut h = fixture(&format!("quit-{name}"), 3, theme);
        h.key_press(egui::Key::J);
        h.key_press(egui::Key::M);
        settle(&mut h);
        h.input_mut()
            .viewports
            .entry(egui::ViewportId::ROOT)
            .or_default()
            .events
            .push(egui::ViewportEvent::Close);
        snap(&mut h, "quit_confirm", name);
        h
    });
}

#[test]
fn dense_tabs() {
    if no_gpu() {
        return;
    }
    // 40 units: table scroll, palette reuse, crowded legends.
    each_theme(|theme, name| {
        let mut h = fixture(&format!("dense-{name}"), 40, theme);
        h.key_press(egui::Key::J);
        snap(&mut h, "dense_waveforms", name);
        for tab in [
            "Templates",
            "Probe",
            "Features",
            "Amplitudes",
            "Similar",
            "Stats",
            "Raster",
        ] {
            click_tab(&mut h, tab);
            snap(&mut h, &format!("dense_{}", tab.to_lowercase()), name);
        }
        h
    });
}

#[test]
fn load_error() {
    if no_gpu() {
        return;
    }
    each_theme(|theme, name| {
        let root = dataset(&format!("load-error-{name}"), 3);
        let mut h = harness_with(theme, move || {
            let (session, positions) = open(&root);
            let mut app = SorrelApp::with_loader(session, label_str, |_| {
                anyhow::bail!("spike_clusters.npy has 12 rows but spike_times.npy has 10")
            });
            configure(&mut app, &root, positions);
            app
        });
        snap(&mut h, "load_error", name);
        h
    });
}

#[test]
fn interactive_nodes_are_labelled() {
    if no_gpu() {
        return;
    }
    // A screen reader (and kittest queries) need a name on every control.
    let h = fixture("labelled", 3, egui::Theme::Dark);
    let interactive = [
        Role::Button,
        Role::CheckBox,
        Role::RadioButton,
        Role::ComboBox,
        Role::TextInput,
        Role::Slider,
        Role::Tab,
    ];
    let mut missing = Vec::new();
    for node in h
        .root()
        .query_all(By::new().predicate(move |n| interactive.contains(&n.role())))
    {
        let ak = node.accesskit_node();
        if ak.label().unwrap_or_default().trim().is_empty()
            && ak.value().unwrap_or_default().trim().is_empty()
        {
            missing.push(format!("{:?} at {:?}", ak.role(), node.rect()));
        }
    }
    assert!(
        missing.is_empty(),
        "unlabelled controls:\n{}",
        missing.join("\n")
    );
}
