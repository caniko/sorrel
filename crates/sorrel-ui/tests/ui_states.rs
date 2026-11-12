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
use sorrel_io::ClusterId;
use sorrel_io::{DataProvider, HasGeometry, KilosortProvider};
use sorrel_ui::brush::{BrushSource, Ghost};
use sorrel_ui::{RasterPipeline, ScatterPipeline, Shell, SorrelApp, TracePipeline};
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
/// shell with no GPU the tests skip, like sorrel-gpu's. The Windows native-test
/// job points `SORREL_LAVAPIPE_ICD` at `release/fetch-lavapipe-windows.ps1`'s
/// Mesa build and sets `SORREL_REQUIRE_GPU`, so there they cannot skip.
fn no_gpu() -> bool {
    static ERR: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let err = ERR.get_or_init(|| {
        // Every History row reads 15:06:40 UTC (read when a session opens;
        // set here, before any test builds one).
        std::env::set_var("SORREL_FIXED_CLOCK_MS", "1793200000000");
        if let Some(icd) = std::env::var_os("SORREL_LAVAPIPE_ICD") {
            std::env::set_var("VK_DRIVER_FILES", &icd);
            std::env::set_var("VK_ICD_FILENAMES", &icd);
            // A host-wide driver filter (e.g. `*radeon*`) would hide lavapipe.
            std::env::remove_var("VK_LOADER_DRIVERS_SELECT");
            std::env::remove_var("VK_LOADER_DRIVERS_DISABLE");
        }
        sorrel_gpu::GpuContext::headless()
            .err()
            .map(|e| e.to_string())
    });
    if let Some(e) = err {
        // The Nix check sets this, so a broken lavapipe fails instead of
        // passing every snapshot test as a skip.
        assert!(
            std::env::var_os("SORREL_REQUIRE_GPU").is_none(),
            "SORREL_REQUIRE_GPU is set but no GPU is available: {e}"
        );
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
fn write_fixture(root: &Path, n_clusters: usize, n_channels: usize) {
    let mut rng = Rng(0x5eed_0f50_11e1);
    let positions: Vec<f32> = (0..n_channels)
        .flat_map(|c| [(c % 2) as f32 * 32.0, (c / 2) as f32 * 20.0])
        .collect();
    let peak = |u: usize| (u * 5 + 3) % n_channels;
    let falloff = |u: usize, ch: usize| {
        let (pu, pc) = (peak(u), ch);
        let dx = positions[2 * pu] - positions[2 * pc];
        let dy = positions[2 * pu + 1] - positions[2 * pc + 1];
        (-(dx * dx + dy * dy) / (2.0 * 30.0f32.powi(2))).exp()
    };

    let mut templates = vec![0f32; n_clusters * N_T * n_channels];
    for u in 0..n_clusters {
        let width = 3.0 + (u % 4) as f32;
        for t in 0..N_T {
            let x = (t as f32 - 20.0) / width;
            let shape = -(-x * x).exp() + 0.35 * (-(x - 2.5).powi(2) / 2.0).exp();
            for ch in 0..n_channels {
                templates[(u * N_T + t) * n_channels + ch] = shape * falloff(u, ch);
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

    let mut raw = vec![0f32; N_SAMPLES * n_channels];
    for v in raw.iter_mut() {
        *v = (rng.next() - 0.5) * 24.0;
    }
    let mut pcs = vec![0f32; n * 3 * 4];
    for (i, &(t, u, amp)) in spikes.iter().enumerate() {
        let u = u as usize;
        for dt in 0..N_T {
            for ch in 0..n_channels {
                raw[(t as usize + dt - 20) * n_channels + ch] +=
                    amp * 3.0 * templates[(u * N_T + dt) * n_channels + ch];
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
        .flat_map(|u| (0..4).map(move |k| ((peak(u) + k) % n_channels) as u32))
        .collect();
    npy::write_1d_u32_atomic(
        &root.join("channel_map.npy"),
        &(0..n_channels as u32).collect::<Vec<_>>(),
    )
    .unwrap();
    npy::write_f32_atomic(
        &root.join("channel_positions.npy"),
        &[n_channels, 2],
        &positions,
    )
    .unwrap();
    npy::write_f32_atomic(
        &root.join("templates.npy"),
        &[n_clusters, N_T, n_channels],
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
    // TemplateFeatureView: each spike projected on its own template and
    // the next one.
    let tf: Vec<f32> = spikes
        .iter()
        .flat_map(|&(_, u, amp)| [amp, 0.3 * amp + 4.0 * (u % 3) as f32])
        .collect();
    npy::write_f32_atomic(&root.join("template_features.npy"), &[n, 2], &tf).unwrap();
    let tf_ind: Vec<u32> = (0..n_clusters as u32)
        .flat_map(|u| [u, (u + 1) % n_clusters as u32])
        .collect();
    write_u32_2d(
        &root.join("template_feature_ind.npy"),
        n_clusters,
        2,
        &tf_ind,
    );
    // A phylib spike attribute for the Spike data pane.
    let snr: Vec<f32> = spikes.iter().map(|s| s.2 / 10.0).collect();
    npy::write_f32_atomic(&root.join("spike_snr.npy"), &[n], &snr).unwrap();

    std::fs::write(
        root.join("params.py"),
        format!(
            "dat_path = 'recording.dat'\nn_channels_dat = {n_channels}\ndtype = 'int16'\n\
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
    // Two alternating stimuli every 150 ms for the PSTH pane.
    let events: String = (0..24)
        .map(|i| {
            format!(
                "{:.3},{}\n",
                0.2 + 0.15 * i as f32,
                ["tone", "light"][i % 2]
            )
        })
        .collect();
    std::fs::write(root.join("events.csv"), format!("time,label\n{events}")).unwrap();
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
    harness_sized(theme, SIZE, build)
}

fn harness_sized(
    theme: egui::Theme,
    size: [f32; 2],
    build: impl FnOnce() -> App + 'static,
) -> Harness<'static, App> {
    Harness::builder()
        .with_size(size)
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
            ScatterPipeline::install(rs);
            sorrel_ui::wgpu_scene3d::Scene3dPipeline::install(rs);
            app.install_gpu_compute(
                rs.device.clone().into(),
                rs.queue.clone().into(),
                &rs.adapter.get_info(),
            );
            app
        })
}

/// A fresh dataset at a fixed path per `slot`: the top bar prints the
/// save directory, so a random tempdir name would break every snapshot.
fn dataset(slot: &str, n_clusters: usize) -> PathBuf {
    dataset_channels(slot, n_clusters, N_CHANNELS)
}

fn dataset_channels(slot: &str, n_clusters: usize, n_channels: usize) -> PathBuf {
    let slot = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("ui_states")
        .join(slot);
    let _ = std::fs::remove_dir_all(&slot);
    let root = slot.join("fixture/synthetic-probe");
    std::fs::create_dir_all(&root).unwrap();
    write_fixture(&root, n_clusters, n_channels);
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
    if let Some(dir) = std::env::var_os("SORREL_DESIGN_DIR") {
        design::dump(h, Path::new(&dir), state, theme);
    }
    h.snapshot(format!("{state}_{theme}"));
}

fn click_tab(h: &mut Harness<'static, App>, title: &str) {
    // Tabs scrolled out of the strip have no on-screen point to click;
    // the AccessKit action reaches them like a screen reader would.
    h.get_by_label(title).click_accesskit();
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
        h.key_press(egui::Key::ArrowDown);
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
        h.key_press(egui::Key::ArrowDown);
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

/// The real settings modal also exports editable OpenPencil frames through
/// `snap`, alongside its dark/light snapshot references.
#[test]
fn keyboard_settings() {
    if no_gpu() {
        return;
    }
    each_theme(|theme, name| {
        let mut h = fixture(&format!("settings-{name}"), 3, theme);
        sorrel_ui::settings::open(&h.ctx);
        snap(&mut h, "keyboard_settings", name);
        h
    });
}

#[test]
fn keyboard_settings_capture_and_conflict() {
    if no_gpu() {
        return;
    }
    each_theme(|theme, name| {
        let mut h = fixture(&format!("settings-capture-{name}"), 3, theme);
        sorrel_ui::settings::open(&h.ctx);
        settle(&mut h);
        h.get_by_label("Search shortcuts").focus();
        settle(&mut h);
        h.get_by_label("Search shortcuts")
            .type_text("inside the lasso");
        settle(&mut h);
        h.get_by_label("Add shortcut").click();
        snap(&mut h, "keyboard_settings_capture", name);
        h.key_press(egui::Key::G);
        snap(&mut h, "keyboard_settings_conflict", name);
        assert!(sorrel_ui::settings::shared(&h.ctx).lock().unwrap().open);
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
        h.key_press(egui::Key::ArrowDown);
        h.key_press_modifiers(egui::Modifiers::ALT, egui::Key::M);
        settle(&mut h);
        // The shell asks every tab to close on a window close.
        h.state_mut().request_close();
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
        h.key_press(egui::Key::ArrowDown);
        snap(&mut h, "dense_waveforms", name);
        for tab in [
            "Templates",
            "Probe",
            "Features",
            "Amplitudes",
            "Similar",
            "Stats",
            "Raster",
            "Summary",
            "ISI",
            "Rate",
            "PSTH",
            "CCG",
            "Suggest",
            "Drift",
            "Contam·time",
            "Quality",
        ] {
            click_tab(&mut h, tab);
            let state = tab.to_lowercase().replace('·', "_");
            snap(&mut h, &format!("dense_{state}"), name);
        }
        h
    });
}

/// Ctrl+click the table row of cluster `id` (the table is the leftmost
/// node with that label; trace channels and other columns repeat numbers).
fn ctrl_click_row(h: &mut Harness<'static, App>, id: &str) {
    h.get_all_by_label(id)
        .min_by(|a, b| a.rect().min.x.total_cmp(&b.rect().min.x))
        .unwrap_or_else(|| panic!("cluster {id} row"))
        .click_modifiers(egui::Modifiers::COMMAND);
    settle(h);
}

fn click_cluster_row(h: &mut Harness<'static, App>, id: u32) {
    h.get_all_by_label(id.to_string().as_str())
        .min_by(|a, b| a.rect().min.x.total_cmp(&b.rect().min.x))
        .unwrap_or_else(|| panic!("cluster {id} row"))
        .click();
    settle(h);
}

/// Native Ctrl shortcuts set both flags on Windows/Linux; COMMAND alone is
/// a synthetic macOS modifier and does not describe a Windows key event.
fn primary_modifiers() -> egui::Modifiers {
    egui::Modifiers {
        ctrl: !cfg!(target_os = "macos"),
        command: true,
        ..Default::default()
    }
}

#[test]
fn curation_keys_alt_g_and_alt_n_label_selected_clusters_with_one_undo_step() {
    if no_gpu() {
        return;
    }
    for (theme, name) in [(egui::Theme::Dark, "dark"), (egui::Theme::Light, "light")] {
        let mut h = fixture(&format!("label-key-events-{name}"), 3, theme);
        let before = {
            let s = sorrel_ui::jobs::read(h.state().session());
            (
                (0..3).map(|c| s.label(ClusterId(c))).collect::<Vec<_>>(),
                s.spike_clusters_global().to_vec(),
            )
        };
        for (key, expected) in [
            (egui::Key::G, PhyLabel::Good),
            (egui::Key::N, PhyLabel::Noise),
        ] {
            click_cluster_row(&mut h, 1);
            ctrl_click_row(&mut h, "2");
            assert_eq!(h.state().selected_clusters(), [ClusterId(1), ClusterId(2)]);
            // Leave a table widget focused: no click in a plot is required.
            h.key_press_modifiers(egui::Modifiers::ALT, key);
            settle(&mut h);
            {
                let s = sorrel_ui::jobs::read(h.state().session());
                assert_eq!(s.label(ClusterId(1)), Some(expected));
                assert_eq!(s.label(ClusterId(2)), Some(expected));
                assert_eq!(s.label(ClusterId(0)), before.0[0]);
                assert_eq!(s.spike_clusters_global(), before.1);
                assert_eq!(s.history_len(), 1);
            }
            h.key_press_modifiers(primary_modifiers(), egui::Key::Z);
            settle(&mut h);
            let s = sorrel_ui::jobs::read(h.state().session());
            assert_eq!(
                (0..3).map(|c| s.label(ClusterId(c))).collect::<Vec<_>>(),
                before.0
            );
            assert_eq!(s.spike_clusters_global(), before.1);
        }
    }
}

#[test]
fn curation_keys_g_merges_membership_into_first_cluster_and_one_undo_restores_it() {
    if no_gpu() {
        return;
    }
    for (theme, name) in [(egui::Theme::Dark, "dark"), (egui::Theme::Light, "light")] {
        let mut h = fixture(&format!("merge-key-events-{name}"), 3, theme);
        click_cluster_row(&mut h, 2);
        ctrl_click_row(&mut h, "0");
        ctrl_click_row(&mut h, "1");
        assert_eq!(
            h.state().selected_clusters(),
            [ClusterId(2), ClusterId(0), ClusterId(1)]
        );
        click_tab(&mut h, "Features");
        lasso_points(&mut h, 4);
        let (before, mut times) = {
            let s = sorrel_ui::jobs::read(h.state().session());
            (
                s.spike_clusters_global().to_vec(),
                (0..3)
                    .flat_map(|c| s.spike_times(ClusterId(c)).to_vec())
                    .collect::<Vec<_>>(),
            )
        };
        times.sort_by_key(|t| t.0);
        h.key_press(egui::Key::G);
        settle(&mut h);
        {
            let s = sorrel_ui::jobs::read(h.state().session());
            assert_eq!(s.spike_times(ClusterId(2)), times);
            assert!(s.spike_times(ClusterId(0)).is_empty());
            assert!(s.spike_times(ClusterId(1)).is_empty());
            assert_eq!(s.history_len(), 1);
        }
        assert_eq!(h.state().selected_clusters(), [ClusterId(2)]);
        assert!(
            h.state().brushes().spikes.is_none(),
            "old lasso cannot follow renumbered buckets"
        );
        assert_full_feature_grid(&h);
        // A lone selected cluster is not a merge and cannot add history.
        h.key_press(egui::Key::G);
        settle(&mut h);
        assert_eq!(sorrel_ui::jobs::read(h.state().session()).history_len(), 1);
        h.key_press_modifiers(primary_modifiers(), egui::Key::Z);
        settle(&mut h);
        assert_eq!(
            sorrel_ui::jobs::read(h.state().session()).spike_clusters_global(),
            before
        );
    }
}

#[test]
fn curation_keys_feature_selection_changes_clear_old_lasso_before_new_features_arrive() {
    if no_gpu() {
        return;
    }
    let mut h = fixture("feature-selection-key-events", 3, egui::Theme::Dark);
    select_two_and_open_features(&mut h);
    lasso_points(&mut h, 4);
    assert!(h.state().brushes().spikes.is_some());
    click_cluster_row(&mut h, 2);
    assert!(h.state().brushes().spikes.is_none());
    assert_full_feature_grid(&h);
    h.key_press_modifiers(egui::Modifiers::ALT, egui::Key::G);
    settle(&mut h);
    assert_eq!(
        sorrel_ui::jobs::read(h.state().session()).label(ClusterId(2)),
        Some(PhyLabel::Good)
    );
    assert!(h.state().brushes().spikes.is_none());
    assert_full_feature_grid(&h);
}

#[test]
fn waveform_m_repeatedly_toggles_on_hover_without_plot_focus_and_respects_text_and_modals() {
    if no_gpu() {
        return;
    }
    use sorrel_ui::prefs::WaveformMode::{Mean, Waveforms};
    for (theme, name) in [(egui::Theme::Dark, "dark"), (egui::Theme::Light, "light")] {
        let root = dataset_channels(&format!("waveform-m-key-events-{name}"), 3, 4);
        let mut h = harness_with(theme, move || {
            let (session, positions) = open(&root);
            let mut app = SorrelApp::new(session, label_str);
            configure(&mut app, &root, positions);
            app
        });
        settle(&mut h);
        click_cluster_row(&mut h, 2);
        ctrl_click_row(&mut h, "0");
        ctrl_click_row(&mut h, "1");
        h.key_press(egui::Key::G);
        settle(&mut h);
        let expected_spikes = sorrel_ui::jobs::read(h.state().session())
            .spike_times(ClusterId(2))
            .len();
        assert!(expected_spikes > 100);
        let loaded = h.state().waveform_all_summary();
        assert!(loaded.ready && !loaded.loading && !loaded.incomplete);
        assert_eq!(loaded.valid, expected_spikes);
        assert_eq!(loaded.total, expected_spikes);
        assert_eq!(loaded.drawn, expected_spikes);
        assert_eq!(
            h.state().view_params.waveform.n_spikes,
            100,
            "legacy preview cap does not affect all mode"
        );
        let at = h.get_by_label("waveform channel 0").rect().center();
        // A real focused button must not be mistaken for text editing. No
        // click or focus request is made on the waveform canvas itself.
        h.get_by_label("Help").focus();
        h.run_ok();
        assert!(h.ctx.egui_wants_keyboard_input());
        assert!(!h.ctx.text_edit_focused());
        assert_eq!(h.state().view_params.waveform.mode, Waveforms);
        for expected in [Mean, Waveforms, Mean, Waveforms, Mean, Waveforms] {
            h.hover_at(at);
            h.key_press(egui::Key::M);
            settle(&mut h);
            assert_eq!(h.state().view_params.waveform.mode, expected);
            let summary = h.state().waveform_all_summary();
            assert!(summary.ready && !summary.loading && !summary.incomplete);
            assert_eq!(
                (summary.valid, summary.total, summary.drawn),
                (expected_spikes, expected_spikes, expected_spikes)
            );
            assert_eq!(
                summary.generation, loaded.generation,
                "M reuses the same full spike set and mean"
            );
            assert_eq!(
                h.state().view_params.waveform.source,
                sorrel_ui::prefs::WaveformSource::AllSpikes
            );
            match expected {
                Mean => {
                    h.get_by_label_contains("mean of all");
                }
                Waveforms => {
                    h.get_by_label_contains("drawing all");
                }
                _ => unreachable!(),
            }
            review_image(&mut h, &format!("all-spikes-m-{expected:?}-{name}"));
        }
        h.hover_at(egui::pos2(10.0, 10.0));
        h.key_press(egui::Key::M);
        settle(&mut h);
        assert_eq!(
            h.state().view_params.waveform.mode,
            Waveforms,
            "M is pane-local"
        );
        click_tab(&mut h, "Probe");
        click_tab(&mut h, "Waveforms");
        assert_eq!(
            h.state().waveform_all_summary().generation,
            loaded.generation,
            "Probe's sampled peak cache cannot replace the all-spike result"
        );

        h.get_all_by_role(Role::TextInput)
            .min_by(|a, b| a.rect().min.x.total_cmp(&b.rect().min.x))
            .expect("cluster filter")
            .focus();
        h.run_ok();
        h.hover_at(at);
        h.key_press(egui::Key::M);
        h.key_press_modifiers(egui::Modifiers::ALT, egui::Key::G);
        settle(&mut h);
        assert_eq!(
            h.state().view_params.waveform.mode,
            Waveforms,
            "text owns M even while hovering a plot"
        );
        assert_eq!(sorrel_ui::jobs::read(h.state().session()).history_len(), 1);

        sorrel_ui::settings::open(&h.ctx);
        settle(&mut h);
        h.hover_at(at);
        h.key_press(egui::Key::M);
        settle(&mut h);
        assert_eq!(
            h.state().view_params.waveform.mode,
            Waveforms,
            "Settings owns the keyboard"
        );
        h.key_press(egui::Key::Escape);
        settle(&mut h);
        ctrl_click_row(&mut h, "2");
        assert!(h.state().selected_clusters().is_empty());
        let cleared = h.state().waveform_all_summary();
        assert!(!cleared.ready && !cleared.loading);
        assert_eq!((cleared.total, cleared.valid, cleared.drawn), (0, 0, 0));
    }
}

#[test]
fn parity_views() {
    if no_gpu() {
        return;
    }
    // The phy-parity views with three units selected on the dense
    // dataset, a filtered table, and the shell at 1920×1080.
    each_theme(|theme, name| {
        let mut h = fixture(&format!("parity-{name}"), 40, theme);
        h.key_press(egui::Key::ArrowDown);
        settle(&mut h);
        ctrl_click_row(&mut h, "5");
        ctrl_click_row(&mut h, "9");
        snap(&mut h, "p3_waveforms", name);
        for tab in ["CCG", "Features", "ISI", "Similar", "Scatter"] {
            click_tab(&mut h, tab);
            snap(&mut h, &format!("p3_{}", tab.to_lowercase()), name);
        }
        // The cluster table's filter is the leftmost text input.
        h.get_all_by_role(Role::TextInput)
            .min_by(|a, b| a.rect().min.x.total_cmp(&b.rect().min.x))
            .expect("cluster filter")
            .type_text("n_spikes > 50");
        h.key_press(egui::Key::Enter);
        snap(&mut h, "p3_table_filter", name);
        h
    });
}

#[test]
fn wide_shell() {
    if no_gpu() {
        return;
    }
    each_theme(|theme, name| {
        let root = dataset(&format!("wide-{name}"), 40);
        let mut h = harness_sized(theme, [1920.0, 1080.0], move || {
            let (session, positions) = open(&root);
            let mut app = SorrelApp::new(session, label_str);
            configure(&mut app, &root, positions);
            app
        });
        settle(&mut h);
        h.key_press(egui::Key::ArrowDown);
        settle(&mut h);
        ctrl_click_row(&mut h, "5");
        snap(&mut h, "wide_shell", name);
        h
    });
}

#[test]
fn multi_select() {
    if no_gpu() {
        return;
    }
    // Two units: CCG grid, overlaid waveforms, comparison summary.
    each_theme(|theme, name| {
        let mut h = fixture(&format!("multi-{name}"), 3, theme);
        h.key_press(egui::Key::ArrowDown);
        settle(&mut h);
        // The trace also labels channel 2; the table is the leftmost "2".
        h.get_all_by_label("2")
            .min_by(|a, b| a.rect().min.x.total_cmp(&b.rect().min.x))
            .expect("cluster 2 row")
            .click_modifiers(egui::Modifiers::COMMAND);
        snap(&mut h, "multi_ccg", name);
        click_tab(&mut h, "Summary");
        snap(&mut h, "multi_summary", name);
        h
    });
}

#[test]
fn saved_status() {
    if no_gpu() {
        return;
    }
    each_theme(|theme, name| {
        let mut h = fixture(&format!("saved-{name}"), 3, theme);
        h.key_press(egui::Key::ArrowDown);
        h.key_press_modifiers(egui::Modifiers::ALT, egui::Key::G);
        settle(&mut h);
        h.key_press_modifiers(egui::Modifiers::COMMAND, egui::Key::S);
        snap(&mut h, "saved", name);
        h
    });
}

#[test]
fn launcher() {
    if no_gpu() {
        return;
    }
    let mut results = SnapshotResults::new();
    for (theme, name) in [(egui::Theme::Dark, "dark"), (egui::Theme::Light, "light")] {
        // No storage in the harness, so the recent list is empty.
        let mut h = Harness::builder()
            .with_size(SIZE)
            .with_theme(theme)
            .wgpu_setup(vulkan_only())
            .build_eframe(|cc| {
                Shell::new_in_memory(cc, Box::new(|_, _, _| anyhow::bail!("no opener")), None)
            });
        h.run_ok();
        h.remove_cursor();
        h.run_ok();
        h.snapshot(format!("launcher_{name}"));
        results.extend_harness(&mut h);
    }
    results.unwrap();
}

#[test]
fn loading() {
    if no_gpu() {
        return;
    }
    each_theme(|theme, name| {
        let root = dataset(&format!("loading-{name}"), 3);
        // The loader blocks until `release` drops, after the snapshot.
        let (release, wait) = std::sync::mpsc::channel::<()>();
        let mut h = harness_with(theme, move || {
            let (session, positions) = open(&root);
            let mut app = SorrelApp::with_loader(session, label_str, move |_| {
                let _ = wait.recv();
                Ok(())
            });
            configure(&mut app, &root, positions);
            app
        });
        snap(&mut h, "loading", name);
        drop(release);
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
            // The panel prints the path: pin it so the snapshot is the same on
            // every machine (the stub loader never reads it).
            app.set_save_dir(PathBuf::from("/data/synthetic-probe"), true);
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

/// `SORREL_DESIGN_DIR=… cargo test … ui_states` also writes every snapshot
/// state as `<state>_<theme>.js`, an OpenPencil (Figma plugin API) script
/// that rebuilds the frame from egui's paint list: rects, text and paths
/// become editable nodes, grouped into one clipping frame per clip rect and
/// named after the AccessKit widget they belong to. GPU callbacks and meshes
/// are cropped from the rendered frame. Load them into the design file with
/// `for f in $DIR/*.js; openpencil eval design/sorrel.fig --stdin -w < $f`;
/// each run refreshes its `Implemented · <state>` page in place. A brand-new
/// file must not start from `openpencil import`'s placeholder nodes: the
/// first eval then reuses their ids and mis-parents the frame, so clear the
/// placeholder page's children with one eval before loading.
mod design {
    use super::App;
    use egui::epaint::{ClippedShape, ColorMode, Shape, StrokeKind};
    use egui::{Color32, Pos2, Rect};
    use egui_kittest::kittest::{By, NodeT, Queryable};
    use egui_kittest::Harness;
    use serde_json::{json, Value};
    use std::path::Path;

    /// Rebuilds one state's frame on the page named after the state; dark
    /// sits at x = 0 and light to its right. Re-running replaces the frame, so
    /// re-ingesting refreshes the `Implemented · …` pages in place.
    const LOADER: &str = r#"
const FONTS = {};
async function font(mono) {
  // OpenPencil's renderer lacks Hack, so prefer DejaVu Sans Mono for monospace.
  const want = mono ? ["DejaVu Sans Mono", "Book"] : ["Ubuntu", "Light"];
  const key = want.join(" ");
  if (!FONTS[key]) {
    for (const [family, style] of [want, mono ? ["DejaVu Sans Mono", "Regular"] : ["Ubuntu", "Regular"], ...(mono ? [["Hack", "Regular"]] : []), ["Inter", "Regular"]]) {
      try { await figma.loadFontAsync({ family, style }); FONTS[key] = { family, style }; break; } catch (e) {}
    }
  }
  return FONTS[key];
}
const paint = (c) => ({ type: "SOLID", color: { r: c[0] / 255, g: c[1] / 255, b: c[2] / 255 }, opacity: c[3] / 255 });
let page = figma.root.children.find((p) => p.name === D.page);
if (!page) { page = figma.createPage(); page.name = D.page; }
for (const old of page.children.filter((n) => n.name === D.name)) old.remove();
const root = figma.createFrame();
page.appendChild(root);
root.name = D.name; root.x = D.theme === "dark" ? 0 : D.w + 120; root.y = 0;
root.resize(D.w, D.h); root.clipsContent = true; root.fills = [paint(D.bg)];
for (const g of D.groups) {
  const f = figma.createFrame();
  root.appendChild(f);
  f.name = g.name; f.x = g.clip[0]; f.y = g.clip[1]; f.resize(Math.max(g.clip[2], 1), Math.max(g.clip[3], 1));
  f.clipsContent = true; f.fills = [];
  for (const it of g.items) {
    let n;
    if (it.k === "rect") {
      n = figma.createRectangle();
      n.resize(Math.max(it.w, 0.01), Math.max(it.h, 0.01));
      n.fills = it.fill ? [paint(it.fill)] : [];
      [n.topLeftRadius, n.topRightRadius, n.bottomRightRadius, n.bottomLeftRadius] = it.r;
      if (it.stroke) { n.strokes = [paint(it.stroke)]; n.strokeWeight = it.sw; n.strokeAlign = it.align; }
    } else if (it.k === "text") {
      n = figma.createText();
      n.fontName = await font(it.mono);
      n.characters = it.text; n.fontSize = it.size; n.fills = [paint(it.fill)];
      // Headless eval has no text metrics: take egui's measured size, then
      // let the editor re-fit once the text is edited.
      n.resize(Math.max(it.w, 1), Math.max(it.h, 1)); n.textAutoResize = "WIDTH_AND_HEIGHT";
      if (it.rot) n.rotation = it.rot;
    } else if (it.k === "path") {
      n = figma.createVector();
      n.vectorPaths = [{ windingRule: it.fill ? "NONZERO" : "NONE", data: it.d }];
      n.fills = it.fill ? [paint(it.fill)] : [];
      if (it.stroke) { n.strokes = [paint(it.stroke)]; n.strokeWeight = it.sw; } else n.strokes = [];
    } else if (it.k === "img") {
      n = figma.createRectangle();
      n.resize(it.w, it.h);
      const img = figma.createImage(new Uint8Array(it.png));
      n.fills = [{ type: "IMAGE", imageHash: img.hash, scaleMode: "FILL" }];
    }
    f.appendChild(n);
    n.name = it.name; n.x = it.x - g.clip[0]; n.y = it.y - g.clip[1];
  }
}
return { page: D.page, frame: D.name, groups: D.groups.length };
"#;

    fn rgba(c: Color32) -> Option<[u8; 4]> {
        (c.a() > 0).then(|| c.to_srgba_unmultiplied())
    }

    fn ir(r: Rect) -> [i32; 4] {
        [
            r.min.x.round() as i32,
            r.min.y.round() as i32,
            r.width().round() as i32,
            r.height().round() as i32,
        ]
    }

    /// The AccessKit widgets of the frame, smallest first, so the first
    /// container hit is the most specific name.
    fn widgets(h: &Harness<'static, App>) -> Vec<(Rect, String)> {
        let mut out: Vec<(Rect, String)> = h
            .root()
            .query_all(By::new().predicate(|_| true))
            .filter_map(|n| {
                let ak = n.accesskit_node();
                ak.bounding_box()?;
                let label = ak.label().or(ak.value()).unwrap_or_default();
                let label = label.trim();
                (!label.is_empty()).then(|| (n.rect(), format!("{:?}: {label}", ak.role())))
            })
            .collect();
        out.sort_by(|a, b| a.0.area().total_cmp(&b.0.area()));
        out
    }

    fn name_for(widgets: &[(Rect, String)], r: Rect, fallback: &str) -> String {
        widgets
            .iter()
            .find(|(w, _)| w.expand(1.0).contains_rect(r))
            .map_or_else(|| fallback.to_owned(), |(_, n)| n.clone())
    }

    fn flatten(shape: &Shape, out: &mut Vec<Shape>) {
        match shape {
            Shape::Vec(v) => v.iter().for_each(|s| flatten(s, out)),
            Shape::Noop => {}
            s => out.push(s.clone()),
        }
    }

    fn path_d(points: &[Pos2], closed: bool) -> String {
        let mut d = String::new();
        for (i, p) in points.iter().enumerate() {
            d += &format!("{}{:.2} {:.2} ", if i == 0 { "M" } else { "L" }, p.x, p.y);
        }
        if closed {
            d += "Z";
        }
        d
    }

    /// A circle as four cubic arcs (vectorPaths has no arc command).
    fn circle_d(c: Pos2, r: f32) -> String {
        let k = 0.5523 * r;
        let (x, y) = (c.x, c.y);
        format!(
            "M{:.2} {:.2} C{:.2} {:.2} {:.2} {:.2} {:.2} {:.2} C{:.2} {:.2} {:.2} {:.2} {:.2} {:.2} \
             C{:.2} {:.2} {:.2} {:.2} {:.2} {:.2} C{:.2} {:.2} {:.2} {:.2} {:.2} {:.2} Z ",
            x + r, y, x + r, y + k, x + k, y + r, x, y + r,
            x - k, y + r, x - r, y + k, x - r, y,
            x - r, y - k, x - k, y - r, x, y - r,
            x + k, y - r, x + r, y - k, x + r, y,
        )
    }

    /// Paths are written in frame coordinates; the loader positions the
    /// vector, so shift the data to the bounding box origin.
    fn shift_d(d: &str, by: Pos2) -> String {
        let mut x_next = true;
        d.split(' ')
            .map(|tok| {
                let (cmd, num) = tok.split_at(
                    tok.find(|c: char| c == '-' || c.is_ascii_digit())
                        .unwrap_or(tok.len()),
                );
                if !cmd.is_empty() {
                    x_next = true;
                }
                match num.parse::<f32>() {
                    Ok(v) => {
                        let v = if x_next { v - by.x } else { v - by.y };
                        x_next = !x_next;
                        format!("{cmd}{v:.2}")
                    }
                    Err(_) => tok.to_owned(),
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// A vector run: consecutive segments/circles with one style merge
    /// into one node, so tick marks and scatter dots stay manageable.
    struct Run {
        d: String,
        bbox: Rect,
        fill: Option<[u8; 4]>,
        stroke: Option<[u8; 4]>,
        sw: f32,
    }

    fn flush(run: &mut Option<Run>, items: &mut Vec<Value>, widgets: &[(Rect, String)]) {
        if let Some(r) = run.take() {
            items.push(json!({
                "k": "path", "name": name_for(widgets, r.bbox, "vector"),
                "x": r.bbox.min.x, "y": r.bbox.min.y, "d": shift_d(r.d.trim(), r.bbox.min),
                "fill": r.fill, "stroke": r.stroke, "sw": r.sw,
            }));
        }
    }

    fn push_vector(
        run: &mut Option<Run>,
        items: &mut Vec<Value>,
        widgets: &[(Rect, String)],
        new: Run,
    ) {
        if let Some(r) = run {
            if r.fill == new.fill && r.stroke == new.stroke && r.sw == new.sw {
                r.d += &new.d;
                r.bbox = r.bbox.union(new.bbox);
                return;
            }
        }
        flush(run, items, widgets);
        *run = Some(new);
    }

    fn crop_png(img: &image::RgbaImage, r: Rect, ppp: f32) -> Option<(Rect, Vec<u8>)> {
        let full = Rect::from_min_size(
            Pos2::ZERO,
            egui::vec2(img.width() as f32, img.height() as f32) / ppp,
        );
        let r = r.intersect(full);
        let r = Rect::from_min_max(r.min.floor(), r.max.ceil());
        (r.width() >= 1.0 && r.height() >= 1.0).then(|| {
            let sub = image::imageops::crop_imm(
                img,
                (r.min.x * ppp) as u32,
                (r.min.y * ppp) as u32,
                (r.width() * ppp) as u32,
                (r.height() * ppp) as u32,
            )
            .to_image();
            let mut png = Vec::new();
            sub.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
                .unwrap();
            (r, png)
        })
    }

    fn text_items(
        t: &egui::epaint::TextShape,
        items: &mut Vec<Value>,
        widgets: &[(Rect, String)],
        img: &image::RgbaImage,
        ppp: f32,
    ) {
        let job = &t.galley.job;
        let starts: Vec<usize> = job.text.char_indices().map(|(b, _)| b).collect();
        let mut char_at = 0usize;
        for placed in &t.galley.rows {
            let text = placed.row.text();
            // ponytail: one colour per row (the section the row starts in);
            // per-glyph ranges when mixed-colour labels need them.
            let byte = starts.get(char_at).copied().unwrap_or(job.text.len());
            char_at +=
                placed.row.char_count_excluding_newline() + usize::from(placed.ends_with_newline);
            if text.trim().is_empty() {
                continue;
            }
            let section = job
                .sections
                .iter()
                .find(|s| s.byte_range.contains(&byte))
                .or(job.sections.last());
            let Some(section) = section else { continue };
            let mut colour = t.override_text_color.unwrap_or(section.format.color);
            if colour == Color32::PLACEHOLDER {
                colour = t.fallback_color;
            }
            colour = colour.gamma_multiply(t.opacity_factor);
            // Aligned labels offset their glyphs inside the row.
            let lead = placed.row.glyphs.first().map_or(0.0, |g| g.pos.x);
            let pos = t.pos + placed.pos.to_vec2() + egui::vec2(lead, 0.0);
            let rect = Rect::from_min_size(pos, placed.row.size);
            // egui's icon font has no design-tool equivalent: keep pixels.
            if text
                .chars()
                .any(|c| matches!(c as u32, 0x2190..=0x2BFF | 0xE000..=0xF8FF | 0x1F000..))
            {
                let bounds = placed
                    .row
                    .visuals
                    .mesh_bounds
                    .translate(t.pos.to_vec2() + placed.pos.to_vec2());
                if let Some((r, png)) = crop_png(img, bounds.expand(1.0), ppp) {
                    items.push(json!({
                        "k": "img", "name": name_for(widgets, r, &text),
                        "x": r.min.x, "y": r.min.y, "w": r.width(), "h": r.height(), "png": png,
                    }));
                }
                continue;
            }
            items.push(json!({
                "k": "text", "name": name_for(widgets, rect, &text),
                "x": pos.x, "y": pos.y, "w": placed.row.size.x, "h": placed.row.size.y, "text": text,
                "size": section.format.font_id.size,
                "mono": section.format.font_id.family == egui::FontFamily::Monospace,
                "fill": rgba(colour).unwrap_or([0, 0, 0, 0]),
                "rot": -t.angle.to_degrees(),
            }));
        }
    }

    pub fn dump(h: &mut Harness<'static, App>, dir: &Path, state: &str, theme: &str) {
        std::fs::create_dir_all(dir).unwrap();
        let ppp = h.ctx.pixels_per_point();
        let img = h.render().expect("render for design dump");
        let widgets = widgets(h);
        let mut shapes: Vec<(Rect, Shape)> = Vec::new();
        for ClippedShape { clip_rect, shape } in &h.output().shapes {
            let mut flat = Vec::new();
            flatten(shape, &mut flat);
            shapes.extend(flat.into_iter().map(|s| (*clip_rect, s)));
        }
        let screen = h.ctx.content_rect();

        let mut groups: Vec<Value> = Vec::new();
        let mut i = 0;
        while i < shapes.len() {
            let group_clip = shapes[i].0;
            let clip = group_clip.intersect(screen);
            let mut items = Vec::new();
            let mut run: Option<Run> = None;
            while i < shapes.len() && shapes[i].0 == group_clip {
                let (_, shape) = &shapes[i];
                i += 1;
                if !shape.visual_bounding_rect().intersects(clip) {
                    continue;
                }
                match shape {
                    Shape::Rect(r) if r.brush.is_none() => {
                        flush(&mut run, &mut items, &widgets);
                        let fill = rgba(r.fill);
                        let stroke = (r.stroke.width > 0.0)
                            .then(|| rgba(r.stroke.color))
                            .flatten();
                        if fill.is_none() && stroke.is_none() {
                            continue;
                        }
                        let cr = r.corner_radius;
                        items.push(json!({
                            "k": "rect", "name": name_for(&widgets, r.rect, "rect"),
                            "x": r.rect.min.x, "y": r.rect.min.y, "w": r.rect.width(), "h": r.rect.height(),
                            "fill": fill, "stroke": stroke, "sw": r.stroke.width,
                            "align": match r.stroke_kind {
                                StrokeKind::Inside => "INSIDE",
                                StrokeKind::Middle => "CENTER",
                                StrokeKind::Outside => "OUTSIDE",
                            },
                            "r": [cr.nw, cr.ne, cr.se, cr.sw],
                        }));
                    }
                    Shape::Text(t) => {
                        flush(&mut run, &mut items, &widgets);
                        text_items(t, &mut items, &widgets, &img, ppp);
                    }
                    Shape::LineSegment { points, stroke } => {
                        let bbox = Rect::from_two_pos(points[0], points[1]);
                        push_vector(
                            &mut run,
                            &mut items,
                            &widgets,
                            Run {
                                d: path_d(points, false),
                                bbox,
                                fill: None,
                                stroke: rgba(stroke.color),
                                sw: stroke.width,
                            },
                        );
                    }
                    Shape::Path(p) if p.points.len() > 1 => {
                        let stroke = match p.stroke.color {
                            ColorMode::Solid(c) if p.stroke.width > 0.0 => rgba(c),
                            _ => None,
                        };
                        let bbox = Rect::from_points(&p.points);
                        push_vector(
                            &mut run,
                            &mut items,
                            &widgets,
                            Run {
                                d: path_d(&p.points, p.closed),
                                bbox,
                                fill: rgba(p.fill),
                                stroke,
                                sw: p.stroke.width,
                            },
                        );
                    }
                    Shape::Circle(c) => {
                        let stroke = (c.stroke.width > 0.0)
                            .then(|| rgba(c.stroke.color))
                            .flatten();
                        let bbox =
                            Rect::from_center_size(c.center, egui::Vec2::splat(2.0 * c.radius));
                        push_vector(
                            &mut run,
                            &mut items,
                            &widgets,
                            Run {
                                d: circle_d(c.center, c.radius),
                                bbox,
                                fill: rgba(c.fill),
                                stroke,
                                sw: c.stroke.width,
                            },
                        );
                    }
                    // Meshes, GPU callbacks, textured rects, beziers: pixels.
                    other => {
                        flush(&mut run, &mut items, &widgets);
                        let r = other.visual_bounding_rect().intersect(clip);
                        if let Some((r, png)) = crop_png(&img, r, ppp) {
                            items.push(json!({
                                "k": "img", "name": name_for(&widgets, r, "raster"),
                                "x": r.min.x, "y": r.min.y, "w": r.width(), "h": r.height(), "png": png,
                            }));
                        }
                    }
                }
            }
            flush(&mut run, &mut items, &widgets);
            if !items.is_empty() {
                let clip_name = widgets
                    .iter()
                    .rev()
                    .find(|(w, _)| {
                        (w.intersect(clip).area() / w.union(clip).area().max(1.0)) > 0.85
                    })
                    .map_or_else(|| format!("clip {:?}", ir(clip)), |(_, n)| n.clone());
                groups.push(json!({ "name": clip_name, "clip": ir(clip), "items": items }));
            }
        }

        let bg = h
            .ctx
            .global_style()
            .visuals
            .panel_fill
            .to_srgba_unmultiplied();
        let data = json!({
            "page": format!("Implemented · {state}"), "name": format!("{state} · {theme}"), "theme": theme,
            "w": screen.width(), "h": screen.height(), "bg": bg, "groups": groups,
        });
        std::fs::write(
            dir.join(format!("{state}_{theme}.js")),
            format!("const D = {data};\n{LOADER}"),
        )
        .unwrap();
    }
}

/// Ctrl+click four points around the middle of Features cell 1 (x = 1A,
/// y = 0A) and press `k`.
fn lasso_and_split(h: &mut Harness<'static, App>) {
    lasso_points(h, 4);
}

/// Ctrl+click the first `n` corners of a square around cell 1's middle.
fn lasso_points(h: &mut Harness<'static, App>, n: usize) {
    let cell = h.get_by_label("feature cell 1").rect();
    let (c, r) = (cell.center(), cell.size() * 0.35);
    for d in [
        egui::vec2(-1.0, -1.0),
        egui::vec2(1.0, -1.0),
        egui::vec2(1.0, 1.0),
        egui::vec2(-1.0, 1.0),
    ]
    .into_iter()
    .take(n)
    {
        let pos = c + d * r;
        h.hover_at(pos);
        // Ctrl stays down across press and release, as a real click.
        h.input_mut().modifiers = egui::Modifiers::COMMAND;
        for pressed in [true, false] {
            h.event(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::COMMAND,
            });
            // One frame per edge: run_ok may advance past egui's long-press
            // deadline while the pointer is held and turn a click into a drag.
            h.step();
        }
        // Past egui's double-click window so each press is its own click.
        settle(h);
    }
    h.input_mut().modifiers = egui::Modifiers::NONE;
    settle(h);
}

fn select_two_and_open_features(h: &mut Harness<'static, App>) {
    h.key_press(egui::Key::ArrowDown);
    settle(h);
    h.get_all_by_label("2")
        .min_by(|a, b| a.rect().min.x.total_cmp(&b.rect().min.x))
        .expect("cluster 2 row")
        .click_modifiers(egui::Modifiers::COMMAND);
    click_tab(h, "Features");
}

fn shape_draws_rect(shape: &egui::Shape, rect: egui::Rect) -> bool {
    match shape {
        egui::Shape::Rect(r) => {
            (r.rect.min - rect.min).length() < 0.5 && (r.rect.size() - rect.size()).length() < 0.5
        }
        egui::Shape::Vec(shapes) => shapes.iter().any(|s| shape_draws_rect(s, rect)),
        _ => false,
    }
}

/// The visible geometry of a Features grid cell (its plot, clipped to the
/// canvas viewport by the view).
fn feature_cell(h: &Harness<'static, App>, cell: usize) -> egui::Rect {
    h.get_by_label(format!("feature cell {cell}").as_str())
        .rect()
}

/// Where a painted text label sits, if it is in this frame's shapes.
fn text_position(h: &Harness<'static, App>, text: &str) -> Option<egui::Pos2> {
    fn walk(shape: &egui::Shape, text: &str, out: &mut Option<egui::Pos2>) {
        match shape {
            egui::Shape::Text(t) if t.galley.text().trim() == text => *out = Some(t.pos),
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| walk(s, text, out)),
            _ => {}
        }
    }
    let mut out = None;
    for clipped in &h.output().shapes {
        walk(&clipped.shape, text, &mut out);
    }
    out
}

fn assert_full_feature_grid(h: &Harness<'static, App>) {
    let cells: Vec<_> = (0..16)
        .map(|cell| {
            h.get_by_label(format!("feature cell {cell}").as_str())
                .rect()
        })
        .collect();
    assert_eq!(h.query_all_by_label_contains("feature cell ").count(), 16);
    assert!(h.query_by_label("Navigator").is_none());
    assert!(h.query_by_label("Navigator + collision").is_none());
    for (cell, rect) in cells.iter().enumerate() {
        assert!(
            rect.width() > 10.0 && rect.height() > 10.0,
            "cell {cell} is visible"
        );
        assert!((rect.size() - cells[0].size()).length() < 0.5);
        assert!(
            (rect.left() - cells[cell % 4].left()).abs() < 0.5,
            "four fixed columns"
        );
        assert!(
            (rect.top() - cells[(cell / 4) * 4].top()).abs() < 0.5,
            "four fixed rows"
        );
        assert!(
            h.output().shapes.iter().any(|s| {
                s.clip_rect.contains_rect(rect.shrink(0.5)) && shape_draws_rect(&s.shape, *rect)
            }),
            "cell {cell} must really paint in full, not just exist in accessibility or below a scroll clip"
        );
    }
}

/// Optional visual-review output without accepting or changing baselines.
fn review_image(h: &mut Harness<'static, App>, name: &str) {
    if let Some(dir) = std::env::var_os("SORREL_UI_REVIEW_DIR") {
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        h.render()
            .expect("render for review")
            .save(dir.join(format!("{name}.png")))
            .unwrap();
    }
}

#[test]
fn tetrode_feature_grid_is_complete_at_narrow_default_wide_and_maximized_sizes() {
    if no_gpu() {
        return;
    }
    each_theme(|theme, name| {
        let root = dataset_channels(&format!("tetrode-grid-{name}"), 3, 4);
        let mut h = harness_with(theme, move || {
            let (session, positions) = open(&root);
            let mut app = SorrelApp::new(session, label_str);
            configure(&mut app, &root, positions);
            app
        });
        settle(&mut h);
        select_two_and_open_features(&mut h);
        for size in [[1000.0, 700.0], SIZE, [2200.0, 1100.0]] {
            h.set_size(egui::vec2(size[0], size[1]));
            settle(&mut h);
            assert_full_feature_grid(&h);
            review_image(&mut h, &format!("tetrode-grid-{}-{name}", size[0]));
        }
        group_button(&h, "maximize pane", 1).click();
        settle(&mut h);
        assert_full_feature_grid(&h);
        review_image(&mut h, &format!("tetrode-grid-maximized-{name}"));
        h
    });
}

#[test]
fn tetrode_waveforms_keep_one_four_channel_set_across_clusters_and_modes() {
    if no_gpu() {
        return;
    }
    let mut results = SnapshotResults::new();
    for n_channels in [4, 16] {
        for geometry in [true, false] {
            for (theme, name) in [(egui::Theme::Dark, "dark"), (egui::Theme::Light, "light")] {
                let root = dataset_channels(
                    &format!("tetrode-waveforms-{n_channels}-{geometry}-{name}"),
                    3,
                    n_channels,
                );
                if !geometry {
                    std::fs::remove_file(root.join("channel_positions.npy")).unwrap();
                }
                let mut h = harness_with(theme, move || {
                    let (session, positions) = open(&root);
                    let mut app = SorrelApp::new(session, label_str);
                    configure(&mut app, &root, positions);
                    app
                });
                settle(&mut h);
                h.get_all_by_label("0")
                    .min_by(|a, b| a.rect().min.x.total_cmp(&b.rect().min.x))
                    .expect("primary cluster 0 row")
                    .click();
                settle(&mut h);
                click_tab(&mut h, "Waveforms");
                let channel_ids = |h: &Harness<'static, App>| -> Vec<usize> {
                    (0..n_channels)
                        .filter(|ch| {
                            h.query_by_label(format!("waveform channel {ch}").as_str())
                                .is_some()
                        })
                        .collect()
                };
                let initial = channel_ids(&h);
                assert_eq!(
                    initial.len(),
                    4,
                    "four channels total, not a peak-only fallback"
                );
                if n_channels == 4 {
                    assert_eq!(initial, [0, 1, 2, 3]);
                    h.set_size(egui::vec2(1000.0, 700.0));
                    settle(&mut h);
                    for ch in &initial {
                        let rect = h
                            .get_by_label(format!("waveform channel {ch}").as_str())
                            .rect();
                        assert!(
                            h.output().shapes.iter().any(|s| {
                                s.clip_rect.contains_rect(rect.shrink(0.5))
                                    && shape_draws_rect(&s.shape, rect)
                            }),
                            "all four waveform boxes fit even in a narrow pane"
                        );
                    }
                    if geometry {
                        review_image(&mut h, &format!("tetrode-waveforms-narrow-{name}"));
                    }
                    h.set_size(egui::vec2(SIZE[0], SIZE[1]));
                    settle(&mut h);
                }
                for cluster in [1, 2] {
                    h.get_all_by_label(cluster.to_string().as_str())
                        .min_by(|a, b| a.rect().min.x.total_cmp(&b.rect().min.x))
                        .expect("cluster row")
                        .click_modifiers(egui::Modifiers::COMMAND);
                    settle(&mut h);
                    assert_eq!(
                        channel_ids(&h),
                        initial,
                        "comparison clusters cannot add channels"
                    );
                }
                if n_channels == 4 && geometry {
                    review_image(&mut h, &format!("tetrode-waveforms-{name}"));
                }
                let point = h
                    .get_by_label(format!("waveform channel {}", initial[0]).as_str())
                    .rect()
                    .center();
                for _ in 0..3 {
                    h.hover_at(point);
                    h.key_press(egui::Key::W);
                    settle(&mut h);
                    assert_eq!(
                        channel_ids(&h),
                        initial,
                        "spikes, means and templates share the same channels"
                    );
                }
                results.extend_harness(&mut h);
            }
        }
    }
    results.unwrap();
}

#[test]
fn features_grid_lasso_split() {
    if no_gpu() {
        return;
    }
    each_theme(|theme, name| {
        let mut h = fixture(&format!("features-{name}"), 3, theme);
        select_two_and_open_features(&mut h);
        snap(&mut h, "features_grid", name);
        lasso_and_split(&mut h);
        snap(&mut h, "features_lasso", name);
        h.key_press(egui::Key::K);
        settle(&mut h);
        let s = sorrel_ui::jobs::read(h.state().session()).history_len();
        assert_eq!(s, 1, "k splits the lassoed spikes as one journal entry");
        h
    });
}

#[test]
fn polygon_draft_with_tracking_off_does_not_publish_spikes_before_ctrl_release() {
    if no_gpu() {
        return;
    }
    let mut h = fixture("polygon-draft-no-live", 3, egui::Theme::Dark);
    h.key_press(egui::Key::ArrowDown);
    settle(&mut h);
    click_tab(&mut h, "Features");
    let preview_generation = h.state().feature_interaction_summary().preview_generation;
    let ctrl = primary_modifiers();
    h.input_mut().modifiers = ctrl;
    for point in [
        egui::vec2(0.1, 0.1),
        egui::vec2(0.9, 0.1),
        egui::vec2(0.9, 0.9),
        egui::vec2(0.1, 0.9),
    ] {
        let pos = feature_cell(&h, 1).lerp_inside(point);
        h.hover_at(pos);
        for pressed in [true, false] {
            h.event(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: ctrl,
            });
            h.step();
        }
        h.step();
        assert!(
            h.state().brushes().spikes.is_none(),
            "tracking is off: adding a polygon vertex must not select spikes"
        );
        let summary = h.state().feature_interaction_summary();
        assert!(summary.drafting && !summary.live_tracking);
        assert_eq!(
            summary.preview_generation, preview_generation,
            "no preview vertex rebuild during the draft"
        );
        assert!(
            summary.hit_generation.is_none(),
            "no full spike hit-test during the draft"
        );
    }
    h.input_mut().modifiers = egui::Modifiers::NONE;
    h.key_press(egui::Key::K);
    settle(&mut h);
    assert_eq!(sorrel_ui::jobs::read(h.state().session()).history_len(), 1);
}

#[test]
fn packet_letters_without_key_events_still_merge_and_split() {
    if no_gpu() {
        return;
    }
    // Remote-desktop clients inject letters as Windows VK_PACKET: winit
    // reports only the text, never a key.
    let packet = |h: &mut Harness<'static, App>, text: &str| {
        h.event(egui::Event::Text(text.into()));
        h.step();
        settle(h);
    };
    let mut h = fixture("packet-keys", 3, egui::Theme::Dark);
    select_two_and_open_features(&mut h);
    settle(&mut h);
    packet(&mut h, "g");
    let history =
        |h: &Harness<'static, App>| sorrel_ui::jobs::read(h.state().session()).history_len();
    assert_eq!(history(&h), 1, "text-only g merges the selection");
    lasso_points(&mut h, 4);
    packet(&mut h, "k");
    settle(&mut h);
    assert_eq!(history(&h), 2, "text-only k splits the lasso");
}

#[test]
fn computed_pcs_new_vertex_on_settled_lasso_keeps_the_frozen_full_set() {
    if no_gpu() {
        return;
    }
    let root = dataset("features-raw-draft-freeze", 3);
    for f in ["pc_features.npy", "pc_feature_ind.npy"] {
        let _ = std::fs::remove_file(root.join(f));
    }
    let mut h = harness_with(egui::Theme::Dark, move || {
        let (session, positions) = open(&root);
        let mut app = SorrelApp::new(session, label_str);
        configure(&mut app, &root, positions);
        app
    });
    settle(&mut h);
    select_two_and_open_features(&mut h);
    settle(&mut h);
    lasso_points(&mut h, 4);
    h.input_mut().modifiers = egui::Modifiers::NONE;
    h.step();
    settle(&mut h);
    assert!(
        h.state().brushes().spikes.is_some(),
        "the settled computed lasso brushes from the full set"
    );
    let preview_generation = h.state().feature_interaction_summary().preview_generation;
    // Another Ctrl-click starts a draft; tracking is off, so nothing reloads.
    let pos = feature_cell(&h, 1).lerp_inside(egui::vec2(0.5, 0.05));
    h.hover_at(pos);
    h.input_mut().modifiers = egui::Modifiers::COMMAND;
    for pressed in [true, false] {
        h.event(egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::COMMAND,
        });
        h.step();
    }
    for _ in 0..4 {
        h.step();
    }
    let summary = h.state().feature_interaction_summary();
    assert!(summary.drafting && !summary.live_tracking);
    assert!(
        h.state().brushes().spikes.is_some(),
        "the frozen brush survives the draft"
    );
    assert_eq!(
        summary.preview_generation, preview_generation,
        "no reload of the display sample mid-draft"
    );
}

#[test]
fn computed_pcs_live_tracking_brushes_a_clicked_polygon_while_ctrl_is_held() {
    if no_gpu() {
        return;
    }
    let root = dataset("features-raw-live-draft", 3);
    for f in ["pc_features.npy", "pc_feature_ind.npy"] {
        let _ = std::fs::remove_file(root.join(f));
    }
    let mut h = harness_with(egui::Theme::Dark, move || {
        let (session, positions) = open(&root);
        let mut app = SorrelApp::new(session, label_str);
        configure(&mut app, &root, positions);
        app
    });
    settle(&mut h);
    select_two_and_open_features(&mut h);
    settle(&mut h);
    // The switch shows while a lasso is in play; turn it on, then start over.
    lasso_points(&mut h, 4);
    h.get_by_label("live tracking").click();
    settle(&mut h);
    h.key_press(egui::Key::Escape);
    settle(&mut h);
    let ctrl = egui::Modifiers::COMMAND;
    h.input_mut().modifiers = ctrl;
    for point in [
        egui::vec2(0.15, 0.15),
        egui::vec2(0.85, 0.15),
        egui::vec2(0.85, 0.85),
        egui::vec2(0.15, 0.85),
    ] {
        let pos = feature_cell(&h, 1).lerp_inside(point);
        h.hover_at(pos);
        for pressed in [true, false] {
            h.event(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: ctrl,
            });
            h.step();
        }
        settle(&mut h);
    }
    let summary = h.state().feature_interaction_summary();
    assert!(summary.drafting && summary.live_tracking && summary.nodes == 4);
    assert!(
        h.state().brushes().spikes.is_some(),
        "live tracking previews the clicked polygon from the full set before Ctrl release"
    );
}

#[test]
fn polygon_ctrl_release_before_last_mouse_up_and_k_still_cuts_once() {
    if no_gpu() {
        return;
    }
    let mut h = fixture("polygon-native-release-order", 3, egui::Theme::Dark);
    h.key_press(egui::Key::ArrowDown);
    settle(&mut h);
    click_tab(&mut h, "Features");
    let before = sorrel_ui::jobs::read(h.state().session())
        .spike_clusters_global()
        .to_vec();
    let ctrl = primary_modifiers();
    h.input_mut().modifiers = ctrl;
    for point in [egui::vec2(0.1, 0.1), egui::vec2(0.9, 0.1)] {
        let pos = feature_cell(&h, 1).lerp_inside(point);
        h.hover_at(pos);
        for pressed in [true, false] {
            h.event(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: ctrl,
            });
            h.step();
        }
    }
    let pos = feature_cell(&h, 1).lerp_inside(egui::vec2(0.9, 0.9));
    h.hover_at(pos);
    h.event(egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed: true,
        modifiers: ctrl,
    });
    h.step();
    h.input_mut().modifiers = egui::Modifiers::NONE;
    h.input_mut().events.extend([
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        },
        egui::Event::Key {
            key: egui::Key::K,
            physical_key: Some(egui::Key::K),
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        },
        egui::Event::Text("k".into()),
    ]);
    h.run_ok();
    settle(&mut h);
    assert_eq!(sorrel_ui::jobs::read(h.state().session()).history_len(), 1);
    assert_ne!(
        sorrel_ui::jobs::read(h.state().session()).spike_clusters_global(),
        before
    );
    h.key_up(egui::Key::K);
    h.key_press_modifiers(primary_modifiers(), egui::Key::Z);
    settle(&mut h);
    assert_eq!(
        sorrel_ui::jobs::read(h.state().session()).spike_clusters_global(),
        before
    );
}

#[test]
fn typing_k_then_enter_in_filter_never_commits_an_existing_polygon() {
    if no_gpu() {
        return;
    }
    let mut h = fixture("polygon-k-text-protection", 3, egui::Theme::Dark);
    h.key_press(egui::Key::ArrowDown);
    settle(&mut h);
    click_tab(&mut h, "Features");
    lasso_points(&mut h, 4);
    h.get_all_by_role(Role::TextInput)
        .min_by(|a, b| a.rect().min.x.total_cmp(&b.rect().min.x))
        .expect("cluster filter")
        .focus();
    h.run_ok();
    assert!(h.ctx.text_edit_focused());
    h.input_mut().events.extend([
        egui::Event::Key {
            key: egui::Key::K,
            physical_key: Some(egui::Key::K),
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        },
        egui::Event::Text("k".into()),
        egui::Event::Key {
            key: egui::Key::Enter,
            physical_key: Some(egui::Key::Enter),
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        },
    ]);
    h.run_ok();
    settle(&mut h);
    assert_eq!(
        sorrel_ui::jobs::read(h.state().session()).history_len(),
        0,
        "ending text focus in this frame must not turn typed K into a split"
    );
    assert!(!h.state().feature_interaction_summary().split_pending);
}

#[test]
fn features_last_ctrl_click_and_plain_k_in_one_frame_splits() {
    if no_gpu() {
        return;
    }
    let mut h = fixture("features-last-node-k", 3, egui::Theme::Dark);
    select_two_and_open_features(&mut h);
    lasso_points(&mut h, 2);
    let cell = feature_cell(&h, 1);
    let pos = cell.lerp_inside(egui::vec2(0.85, 0.85));
    h.hover_at(pos);
    h.input_mut().modifiers = egui::Modifiers::COMMAND;
    h.event(egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed: true,
        modifiers: egui::Modifiers::COMMAND,
    });
    h.step();
    h.input_mut().modifiers = egui::Modifiers::NONE;
    h.input_mut().events.extend([
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::COMMAND,
        },
        egui::Event::Key {
            key: egui::Key::K,
            physical_key: Some(egui::Key::K),
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        },
        egui::Event::Text("k".into()),
    ]);
    h.run_ok();
    assert_eq!(sorrel_ui::jobs::read(h.state().session()).history_len(), 1);
}

#[test]
fn features_freehand_release_and_k_in_one_frame_splits_and_undoes() {
    if no_gpu() {
        return;
    }
    each_theme(|theme, name| {
        let mut h = fixture(&format!("features-fast-k-{name}"), 3, theme);
        select_two_and_open_features(&mut h);
        let before = sorrel_ui::jobs::read(h.state().session())
            .spike_clusters_global()
            .to_vec();
        // A real workflow starts by filtering the table, then drawing in
        // Features. A drag is not an egui click, so it must explicitly take
        // focus back from the text editor before the user's K arrives.
        h.get_all_by_role(Role::TextInput)
            .min_by(|a, b| a.rect().min.x.total_cmp(&b.rect().min.x))
            .expect("cluster filter")
            .focus();
        h.run_ok();
        assert!(h.ctx.text_edit_focused());
        let cell = feature_cell(&h, 1);
        let start = cell.lerp_inside(egui::vec2(0.15, 0.15));
        h.hover_at(start);
        h.input_mut().modifiers = egui::Modifiers::COMMAND;
        h.event(egui::Event::PointerButton {
            pos: start,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::COMMAND,
        });
        h.run_ok();
        for point in [
            egui::vec2(0.85, 0.15),
            egui::vec2(0.85, 0.85),
            egui::vec2(0.15, 0.85),
        ] {
            h.hover_at(cell.lerp_inside(point));
            h.run_ok();
        }
        assert!(
            !h.ctx.text_edit_focused(),
            "drawing takes focus from the filter"
        );
        // Native event queues can deliver release and the next key together.
        // Do not settle between them: the pane has not closed the stroke yet.
        h.input_mut().modifiers = egui::Modifiers::NONE;
        h.input_mut().events.push(egui::Event::PointerButton {
            pos: cell.lerp_inside(egui::vec2(0.15, 0.85)),
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        });
        h.input_mut().events.push(egui::Event::Key {
            key: egui::Key::K,
            physical_key: Some(egui::Key::K),
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        });
        h.input_mut().events.push(egui::Event::Text("k".into()));
        h.run_ok();
        settle(&mut h);
        {
            let session = sorrel_ui::jobs::read(h.state().session());
            assert_eq!(session.history_len(), 1, "the first K must not be lost");
            assert_ne!(session.spike_clusters_global(), before);
        }
        h.key_up(egui::Key::K);
        h.key_press_modifiers(primary_modifiers(), egui::Key::Z);
        settle(&mut h);
        assert_eq!(
            sorrel_ui::jobs::read(h.state().session()).spike_clusters_global(),
            before,
            "one Undo restores every spike"
        );
        h
    });
}

/// Opt-in real-scale regression. All mutable files are copied into a fresh
/// directory; the original raw binary is only memory-mapped read-only.
/// SORREL_REAL_SORT=<sort> cargo test -p sorrel-ui --test ui_states \
///     real_sort_fast_k -- --ignored --nocapture
#[test]
#[ignore = "requires SORREL_REAL_SORT and several million spikes"]
fn real_sort_fast_k_changes_membership_and_one_undo_restores_it() {
    real_sort_keyboard_cut(false);
}

#[test]
#[ignore = "requires SORREL_REAL_SORT and several million spikes"]
fn real_sort_polygon_k_without_live_tracking_splits_and_undoes() {
    real_sort_keyboard_cut(true);
}

fn real_sort_keyboard_cut(polygon: bool) {
    use std::time::Instant;
    assert!(!no_gpu(), "real-scale UI regression requires a GPU");
    let original = PathBuf::from(std::env::var_os("SORREL_REAL_SORT").expect("SORREL_REAL_SORT"));
    let original_clusters = std::fs::read(original.join("spike_clusters.npy")).unwrap();
    let original_labels = std::fs::read(original.join("cluster_group.tsv")).unwrap();
    let temp_root = std::env::var_os("SORREL_TEST_TEMP")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let copy = tempfile::Builder::new()
        .prefix("sorrel-real-fast-k-")
        .tempdir_in(temp_root)
        .unwrap();
    for entry in std::fs::read_dir(&original).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file()
            && entry.file_name() != "recording.dat"
            && matches!(
                entry.path().extension().and_then(|s| s.to_str()),
                Some("npy" | "tsv" | "py" | "journal" | "json")
            )
        {
            std::fs::copy(entry.path(), copy.path().join(entry.file_name())).unwrap();
        }
    }
    fn copy_snapshot(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            let target = to.join(entry.file_name());
            if entry.file_type().unwrap().is_dir() {
                copy_snapshot(&entry.path(), &target);
            } else {
                assert_ne!(entry.file_name(), "recording.dat");
                std::fs::copy(entry.path(), target).unwrap();
            }
        }
    }
    if original.join("sorrel_original").exists() {
        copy_snapshot(
            &original.join("sorrel_original"),
            &copy.path().join("sorrel_original"),
        );
    }
    let opened = Instant::now();
    let provider = KilosortProvider::open(
        copy.path(),
        KilosortOpenParams {
            dat_path: Some(original.join("recording.dat")),
            ..Default::default()
        },
    )
    .unwrap();
    eprintln!("real sort: provider open {:?}", opened.elapsed());
    let positions = provider.channel_positions().to_vec();
    let baseline = sorrel_data::baseline_hash(
        &std::fs::read(
            sorrel_io::RootView::open(copy.path())
                .unwrap()
                .path("spike_clusters.npy"),
        )
        .unwrap(),
    );
    let journal = Journal::open_or_create(
        &copy.path().join("sorrel.journal"),
        baseline,
        provider.n_clusters(),
    )
    .unwrap();
    let indexed = Instant::now();
    let mut session = Session::new(provider, journal);
    eprintln!("real sort: session index {:?}", indexed.elapsed());
    let seeded = Instant::now();
    session.seed_amplitudes();
    session.seed_templates();
    session.seed_pc_features();
    session.seed_template_waveforms();
    session.replay_journal().unwrap();
    eprintln!("real sort: seeding/replay {:?}", seeded.elapsed());
    let before = session.spike_clusters_global().to_vec();
    let before_labels = session.labels().to_vec();
    let history = session.history_len();
    let root = copy.path().to_path_buf();
    let mut h = harness_with(egui::Theme::Dark, move || {
        let mut app = SorrelApp::new(session, label_str);
        configure(&mut app, &root, positions);
        app
    });
    h.key_press(egui::Key::ArrowDown);
    h.run_ok();
    h.get_by_label("Features").click_accesskit();
    let features = Instant::now();
    while h.query_by_label("feature cell 1").is_none()
        || h.state().feature_interaction_summary().preview_generation == 0
    {
        assert!(
            features.elapsed() < Duration::from_secs(180),
            "features timed out"
        );
        h.step();
        std::thread::sleep(Duration::from_millis(10));
    }
    eprintln!(
        "real sort: features ready {:?}, selected {:?}, total open {:?}",
        features.elapsed(),
        h.state().selected_clusters(),
        opened.elapsed()
    );
    if std::env::var("SORREL_TEST_WAIT_WAVEFORMS").is_ok_and(|value| value == "1") {
        let wait = Instant::now();
        loop {
            let summary = h.state().waveform_all_summary();
            if summary.ready && !summary.loading {
                assert!(summary.total >= 100_000, "must exercise a real large unit");
                assert_eq!(summary.drawn, summary.valid, "no all-spike drawing cap");
                assert_eq!(summary.raster_failed, 0);
                eprintln!(
                    "real sort: all waveforms ready, {} valid/{} total/{} drawn, total open {:?}",
                    summary.valid,
                    summary.total,
                    summary.drawn,
                    opened.elapsed()
                );
                break;
            }
            assert!(
                wait.elapsed() < Duration::from_secs(180),
                "all waveforms timed out: {summary:?}"
            );
            h.step();
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    h.get_all_by_role(Role::TextInput)
        .min_by(|a, b| a.rect().min.x.total_cmp(&b.rect().min.x))
        .expect("real sort cluster filter")
        .focus();
    h.run_ok();
    let cell = feature_cell(&h, 1);
    let ctrl = primary_modifiers();
    h.input_mut().modifiers = ctrl;
    if polygon {
        let initial_preview = h.state().feature_interaction_summary().preview_generation;
        let mut max_frame = Duration::ZERO;
        for (node, point) in [
            egui::vec2(0.15, 0.15),
            egui::vec2(0.85, 0.15),
            egui::vec2(0.85, 0.85),
            egui::vec2(0.15, 0.85),
        ]
        .into_iter()
        .enumerate()
        {
            let pos = feature_cell(&h, 1).lerp_inside(point);
            h.hover_at(pos);
            for pressed in [true, false] {
                h.event(egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: ctrl,
                });
                let frame = Instant::now();
                h.step();
                max_frame = max_frame.max(frame.elapsed());
            }
            h.step();
            assert!(
                h.state().brushes().spikes.is_none(),
                "no selection updates during the polygon draft"
            );
            let summary = h.state().feature_interaction_summary();
            assert_eq!(summary.nodes, node + 1);
            assert!(summary.drafting && !summary.live_tracking);
            assert!(summary.hit_generation.is_none());
            assert_eq!(summary.preview_generation, initial_preview);
        }
        eprintln!("real polygon: no selection or preview rebuilds while drawing; slowest input frame {max_frame:?}");
        assert!(
            max_frame < Duration::from_millis(250),
            "polygon input blocked: {max_frame:?}"
        );
    } else {
        let start = cell.lerp_inside(egui::vec2(0.15, 0.15));
        h.hover_at(start);
        h.event(egui::Event::PointerButton {
            pos: start,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: ctrl,
        });
        h.step();
        for point in [
            egui::vec2(0.85, 0.15),
            egui::vec2(0.85, 0.85),
            egui::vec2(0.15, 0.85),
        ] {
            h.hover_at(cell.lerp_inside(point));
            h.step();
        }
    }
    h.input_mut().modifiers = egui::Modifiers::NONE;
    if polygon {
        // The user's actual sequence: release Ctrl, let that native frame
        // finish, then press plain K while preview work may still be pending.
        let release = Instant::now();
        h.step();
        eprintln!("real polygon: Ctrl-release frame {:?}", release.elapsed());
        assert!(
            release.elapsed() < Duration::from_millis(250),
            "Ctrl release blocked the input frame"
        );
    } else {
        h.input_mut().events.push(egui::Event::PointerButton {
            pos: cell.lerp_inside(egui::vec2(0.15, 0.85)),
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        });
    }
    h.input_mut().events.extend([
        egui::Event::Key {
            key: egui::Key::K,
            physical_key: Some(egui::Key::K),
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        },
        egui::Event::Text("k".into()),
    ]);
    let cut = Instant::now();
    h.step();
    eprintln!("real sort: release/K input frame {:?}", cut.elapsed());
    while cut.elapsed() < Duration::from_secs(10) {
        if sorrel_ui::jobs::read(h.state().session()).history_len() > history {
            break;
        }
        h.step();
        std::thread::sleep(Duration::from_millis(1));
    }
    {
        let session = sorrel_ui::jobs::read(h.state().session());
        assert_eq!(session.history_len(), history + 1);
        let moved = session
            .spike_clusters_global()
            .iter()
            .zip(&before)
            .filter(|(a, b)| a != b)
            .count();
        assert!(moved > 0, "K must move actual spikes");
        eprintln!("real sort: K moved {moved} spikes in {:?}", cut.elapsed());
    }
    h.key_up(egui::Key::K);
    h.key_press_modifiers(primary_modifiers(), egui::Key::Z);
    h.run_ok();
    assert_eq!(
        sorrel_ui::jobs::read(h.state().session()).spike_clusters_global(),
        before,
        "one keyboard Undo restores the entire real sort"
    );
    assert_eq!(
        sorrel_ui::jobs::read(h.state().session()).labels(),
        before_labels,
        "one keyboard Undo also restores the original cluster labels"
    );
    drop(h);
    assert_eq!(
        std::fs::read(original.join("spike_clusters.npy")).unwrap(),
        original_clusters
    );
    assert_eq!(
        std::fs::read(original.join("cluster_group.tsv")).unwrap(),
        original_labels
    );
}

#[test]
fn features_cancel_clears_the_lasso_without_changing_cluster_membership() {
    if no_gpu() {
        return;
    }
    each_theme(|theme, name| {
        let mut h = fixture(&format!("features-cancel-{name}"), 3, theme);
        select_two_and_open_features(&mut h);
        let before = {
            let session = sorrel_ui::jobs::read(h.state().session());
            (session.version(), session.spike_clusters_global().to_vec())
        };
        lasso_points(&mut h, 4);
        assert!(
            h.state().brushes().spikes.is_some(),
            "the lasso selected spikes"
        );
        h.key_press(egui::Key::Escape);
        settle(&mut h);
        assert!(h.state().brushes().spikes.is_none());
        // An abandoned lasso must not become a pending cut again later.
        h.key_press(egui::Key::K);
        settle(&mut h);
        let session = sorrel_ui::jobs::read(h.state().session());
        assert_eq!(session.version(), before.0);
        assert_eq!(session.spike_clusters_global(), before.1);
        assert_eq!(session.history_len(), 0);
        drop(session);
        h
    });
}

#[test]
fn features_cancel_during_freehand_drawing_does_not_restore_it_on_release() {
    if no_gpu() {
        return;
    }
    each_theme(|theme, name| {
        let mut h = fixture(&format!("features-cancel-stroke-{name}"), 3, theme);
        select_two_and_open_features(&mut h);
        let cell = h.get_by_label("feature cell 1").rect();
        let start = cell.lerp_inside(egui::vec2(0.2, 0.2));
        h.hover_at(start);
        h.input_mut().modifiers = egui::Modifiers::COMMAND;
        h.event(egui::Event::PointerButton {
            pos: start,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::COMMAND,
        });
        h.run_ok();
        for point in [egui::vec2(0.8, 0.2), egui::vec2(0.8, 0.8)] {
            h.hover_at(cell.lerp_inside(point));
            h.run_ok();
        }
        h.input_mut().modifiers = egui::Modifiers::NONE;
        h.key_press(egui::Key::Escape);
        let end = cell.lerp_inside(egui::vec2(0.2, 0.8));
        h.hover_at(end);
        h.event(egui::Event::PointerButton {
            pos: end,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        });
        h.run_ok();
        settle(&mut h);
        assert!(h.state().brushes().spikes.is_none());
        h.key_press(egui::Key::K);
        settle(&mut h);
        assert_eq!(sorrel_ui::jobs::read(h.state().session()).history_len(), 0);
        h
    });
}

#[test]
fn features_navigation_keeps_layout_and_moves_projection_windows_as_a_group() {
    if no_gpu() {
        return;
    }
    each_theme(|theme, name| {
        let mut h = wide_fixture(&format!("features-navigation-{name}"), theme);
        select_two_and_open_features(&mut h);
        // The fitted default shows every projection before navigation.
        assert_full_feature_grid(&h);
        // The expanded alternatives strip reduces the canvas before any
        // navigation; settle it and capture that layout as the baseline.
        h.get_by_label("ALTERNATIVES").click();
        settle(&mut h);
        let before: Vec<_> = (0..16).map(|cell| feature_cell(&h, cell)).collect();
        let controls = h.get_by_label("k-means").rect();
        let alternative = h
            .get_all_by_label_contains("alternative ")
            .next()
            .expect("the original alternatives strip")
            .rect();
        let caption_before = [
            text_position(&h, "t × 0A").expect("cell 0 caption"),
            text_position(&h, "0B × 0A").expect("cell 1 caption"),
        ];
        let data_before = {
            let session = sorrel_ui::jobs::read(h.state().session());
            (
                session.version(),
                session.spike_clusters_global().to_vec(),
                session.history_len(),
            )
        };
        lasso_points(&mut h, 4);
        let brush = h.state().brushes().spikes.as_ref().unwrap().key();

        // A modest pointer-anchored zoom near the top of cell 1 leaves it
        // fully on screen while the first cell is cropped at the canvas edge.
        let anchor = egui::pos2(
            before[1].center().x,
            before[1].top() + before[1].height() * 0.1,
        );
        wheel_at(&mut h, anchor, 100.0, egui::Modifiers::NONE);
        let zoomed: Vec<_> = [0, 1, 2]
            .iter()
            .map(|&cell| feature_cell(&h, cell))
            .collect();
        let scale = zoomed[1].width() / before[1].width();
        assert!(scale > 1.05, "plain wheel enlarges the projection squares");
        assert!(
            (zoomed[1].height() / before[1].height() - scale).abs() < 0.01,
            "squares grow by one ratio on both axes"
        );
        let expected_min = anchor + (before[1].min - anchor) * scale;
        let expected_max = anchor + (before[1].max - anchor) * scale;
        assert!(
            (zoomed[1].min - expected_min).length() < 0.75
                && (zoomed[1].max - expected_max).length() < 0.75,
            "the image point under the pointer stays put: {zoomed:?} vs anchor {anchor:?}"
        );
        assert!(
            zoomed[0].width() < zoomed[1].width() - 1.0
                && (zoomed[0].left() - before[0].left()).abs() < 0.75,
            "a cell cropped by the canvas reports its visible geometry, not its old full rect"
        );
        // Exercise the GPU path with callback viewports clipped to the
        // canvas, not just AccessKit/paint metadata.
        h.render().expect("render the pointer-anchored canvas zoom");

        // Pan inward: the enlarged first cell becomes fully visible again,
        // and the whole matrix moves equally in screen pixels.
        let start = zoomed[2].center();
        let end = start + egui::vec2(80.0, 60.0);
        h.hover_at(start);
        h.event(egui::Event::PointerButton {
            pos: start,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        });
        h.run_ok();
        h.hover_at(end);
        h.run_ok();
        h.event(egui::Event::PointerButton {
            pos: end,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        });
        settle(&mut h);
        let panned: Vec<_> = [0, 1, 2, 4]
            .iter()
            .map(|&cell| feature_cell(&h, cell))
            .collect();
        let movement = panned[1].center() - zoomed[1].center();
        assert!(movement.length() > 1.0, "dragging must move the grid");
        assert!(
            (panned[1].size() - zoomed[1].size()).length() < 0.5,
            "dragging after zoom does not rescale the squares"
        );
        assert!(
            (panned[0].width() - panned[1].width()).abs() < 0.75
                && panned[0].left() > zoomed[0].left() + 1.0,
            "panning brings the enlarged, cropped cell fully into view"
        );
        for rect in [panned[0], panned[1]] {
            assert!(
                h.output().shapes.iter().any(|s| {
                    s.clip_rect.contains_rect(rect.shrink(0.5)) && shape_draws_rect(&s.shape, rect)
                }),
                "a panned cell is painted in full inside the shared viewport"
            );
        }
        h.render()
            .expect("render the enlarged, panned canvas and its clipped cells");

        // Size and both centre-to-centre pitches grow by the same ratio.
        let scale_now = panned[1].width() / before[1].width();
        assert!((scale_now - scale).abs() < 0.01);
        let pitch_x_before = before[1].center().x - before[0].center().x;
        let pitch_x_after = panned[1].center().x - panned[0].center().x;
        assert!(
            (pitch_x_after / pitch_x_before - scale_now).abs() < 0.01,
            "horizontal spacing scales with the squares: {pitch_x_after} / {pitch_x_before} vs {scale_now}"
        );
        let pitch_y_before = before[4].center().y - before[0].center().y;
        let pitch_y_after = panned[3].center().y - panned[0].center().y;
        assert!(
            (pitch_y_after / pitch_y_before - scale_now).abs() < 0.01,
            "vertical spacing scales with the squares"
        );
        let caption_after = [
            text_position(&h, "t × 0A").expect("cell 0 caption after zoom"),
            text_position(&h, "0B × 0A").expect("cell 1 caption after zoom"),
        ];
        let caption_pitch_before = caption_before[1].x - caption_before[0].x;
        let caption_pitch_after = caption_after[1].x - caption_after[0].x;
        assert!(
            (caption_pitch_after / caption_pitch_before - scale_now).abs() < 0.01,
            "captions travel and scale with their squares"
        );
        assert_eq!(h.state().brushes().spikes.as_ref().unwrap().key(), brush);
        assert_eq!(h.get_by_label("k-means").rect(), controls);
        assert_eq!(
            h.get_all_by_label_contains("alternative ")
                .next()
                .unwrap()
                .rect(),
            alternative
        );

        // Plain wheel-down shrinks the canvas again; double-click resets zoom
        // and pan to the fitted all-16 default.
        let zoomed_in = panned[1];
        wheel_at(&mut h, zoomed_in.center(), -60.0, egui::Modifiers::NONE);
        let zoomed_out = feature_cell(&h, 1);
        assert!(
            zoomed_out.width() < zoomed_in.width() - 0.5,
            "wheel-down zooms the canvas out"
        );
        let reset_at = feature_cell(&h, 0).center();
        double_click_at(&mut h, reset_at);
        for (cell, rect) in before.iter().enumerate() {
            let restored = feature_cell(&h, cell);
            assert!(
                (restored.min - rect.min).length() < 0.75
                    && (restored.size() - rect.size()).length() < 0.75,
                "double-click restores cell {cell} to the fitted layout"
            );
        }
        assert_full_feature_grid(&h);
        assert_eq!(h.state().brushes().spikes.as_ref().unwrap().key(), brush);

        // Ctrl-wheel and Alt-wheel keep their old meanings and never scale
        // the canvas.
        for modifiers in [egui::Modifiers::COMMAND, egui::Modifiers::ALT] {
            let restored = feature_cell(&h, 1);
            wheel_at(&mut h, restored.center(), 100.0, modifiers);
            let after = feature_cell(&h, 1);
            assert!(
                (after.min - restored.min).length() < 0.75
                    && (after.size() - restored.size()).length() < 0.75,
                "{modifiers:?}+wheel must not zoom the projection canvas"
            );
        }
        let data_after = {
            let session = sorrel_ui::jobs::read(h.state().session());
            (
                session.version(),
                session.spike_clusters_global().to_vec(),
                session.history_len(),
            )
        };
        assert_eq!(
            data_before, data_after,
            "navigation never changes cluster data or history"
        );
        h
    });
}

#[test]
fn features_computed_pcs() {
    if no_gpu() {
        return;
    }
    // No pc_features.npy: the grid falls back to PCs fitted on raw snippets.
    each_theme(|theme, name| {
        let root = dataset(&format!("features-raw-{name}"), 3);
        for f in ["pc_features.npy", "pc_feature_ind.npy"] {
            let _ = std::fs::remove_file(root.join(f));
        }
        let mut h = harness_with(theme, move || {
            let (session, positions) = open(&root);
            let mut app = SorrelApp::new(session, label_str);
            configure(&mut app, &root, positions);
            app
        });
        settle(&mut h);
        select_two_and_open_features(&mut h);
        snap(&mut h, "features_computed", name);
        lasso_and_split(&mut h);
        h.key_press(egui::Key::K);
        settle(&mut h);
        settle(&mut h);
        let s = sorrel_ui::jobs::read(h.state().session()).history_len();
        assert_eq!(s, 1, "k splits once every lassoed spike is projected");
        h
    });
}

#[test]
fn amplitudes_threshold_split() {
    if no_gpu() {
        return;
    }
    // phy v2.1: Alt+right-drag a threshold on one cluster, `k` splits the
    // spikes below it.
    each_theme(|theme, name| {
        let mut h = fixture(&format!("amplitudes-{name}"), 3, theme);
        h.key_press(egui::Key::ArrowDown);
        settle(&mut h);
        click_tab(&mut h, "Amplitudes");
        let plot = h.get_by_label("amplitude plot").rect();
        let (a, b) = (
            egui::pos2(plot.center().x - 40.0, plot.top() + 14.0),
            egui::pos2(plot.center().x + 40.0, plot.top() + 14.0),
        );
        h.input_mut().modifiers = egui::Modifiers::ALT;
        h.hover_at(a);
        for (pos, pressed) in [(a, true), (b, true), (b, false)] {
            h.hover_at(pos);
            if pressed && pos == b {
                h.run_ok();
                continue;
            }
            h.event(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Secondary,
                pressed,
                modifiers: egui::Modifiers::ALT,
            });
            h.run_ok();
        }
        h.input_mut().modifiers = egui::Modifiers::NONE;
        snap(&mut h, "amplitudes_threshold", name);
        h.key_press(egui::Key::K);
        settle(&mut h);
        let s = sorrel_ui::jobs::read(h.state().session()).history_len();
        assert_eq!(s, 1, "k splits the spikes under the threshold");
        h
    });
}

#[test]
fn cluster_scatter_dense() {
    if no_gpu() {
        return;
    }
    // phy v2.1 ClusterScatterView: one point per cluster (amp × depth,
    // size = log firing rate), the selection ringed.
    each_theme(|theme, name| {
        let mut h = fixture(&format!("cluster-scatter-{name}"), 40, theme);
        h.key_press(egui::Key::ArrowDown);
        settle(&mut h);
        click_tab(&mut h, "Scatter");
        snap(&mut h, "cluster_scatter", name);
        h
    });
}

#[test]
fn template_features_two_clusters() {
    if no_gpu() {
        return;
    }
    // phy TemplateFeatureView: each spike projected on the two selected
    // clusters' templates.
    each_theme(|theme, name| {
        let mut h = fixture(&format!("template-features-{name}"), 3, theme);
        select_two_and_open_features(&mut h);
        click_tab(&mut h, "Tmpl features");
        snap(&mut h, "template_features", name);
        h
    });
}

#[test]
fn spike_data_and_trace_image() {
    if no_gpu() {
        return;
    }
    // phy Spike*View (spike_snr.npy against time) and TraceImageView.
    each_theme(|theme, name| {
        let mut h = fixture(&format!("spike-data-{name}"), 3, theme);
        select_two_and_open_features(&mut h);
        click_tab(&mut h, "Spike data");
        click_tab(&mut h, "Trace image");
        snap(&mut h, "spike_data_trace_image", name);
        h
    });
}

#[test]
fn features_lasso_drawing_and_kmeans() {
    if no_gpu() {
        return;
    }
    // The lasso tool's states: open polygon, then the k-means button's
    // auto-lasso (with the exact part highlighted), `k` splitting it, and
    // auto-run refitting on what is left.
    each_theme(|theme, name| {
        let mut h = fixture(&format!("features-km-{name}"), 3, theme);
        select_two_and_open_features(&mut h);
        lasso_points(&mut h, 2);
        snap(&mut h, "features_lasso_drawing", name);
        // The button replaces the open lasso; Esc would leave a timed toast.
        h.get_by_label("k-means").click();
        settle(&mut h);
        snap(&mut h, "features_kmeans", name);
        h.key_press(egui::Key::K);
        settle(&mut h);
        let s = sorrel_ui::jobs::read(h.state().session()).history_len();
        assert_eq!(s, 1, "k splits the k-means lasso as one journal entry");
        // The Features pane is narrow (Compact): auto-run lives in its ⚙ popover.
        h.get_all_by_label("⚙")
            .find(|n| (630.0..955.0).contains(&n.rect().center().x))
            .expect("Features pane ⚙")
            .click();
        settle(&mut h);
        h.get_by_label("auto-run").click();
        settle(&mut h);
        h.key_press(egui::Key::Escape);
        snap(&mut h, "features_kmeans_auto", name);
        h
    });
}

/// Hover `at` and scroll the wheel by `delta` points with `modifiers`.
fn wheel_at(h: &mut Harness<'static, App>, at: egui::Pos2, delta: f32, modifiers: egui::Modifiers) {
    h.hover_at(at);
    h.input_mut().modifiers = modifiers;
    h.event(egui::Event::MouseWheel {
        unit: egui::MouseWheelUnit::Point,
        delta: egui::vec2(0.0, delta),
        phase: egui::TouchPhase::Move,
        modifiers,
    });
    settle(h);
    h.input_mut().modifiers = egui::Modifiers::NONE;
}

/// Hover the middle of the plot labelled `label` and scroll the wheel up.
fn wheel_zoom(h: &mut Harness<'static, App>, label: &str) {
    let at = h.get_by_label(label).rect().center();
    wheel_at(h, at, 300.0, egui::Modifiers::NONE);
}

/// Two quick primary clicks at `at`, close enough to count as a double-click.
/// kittest advances 250 ms per queued event, so put both clicks in one raw
/// frame instead of spreading them past egui's double-click window.
fn double_click_at(h: &mut Harness<'static, App>, at: egui::Pos2) {
    h.hover_at(at);
    h.run_ok();
    for _ in 0..2 {
        for pressed in [true, false] {
            h.input_mut().events.push(egui::Event::PointerButton {
                pos: at,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::NONE,
            });
        }
    }
    h.run_ok();
    settle(h);
}

#[test]
fn raster_and_rate_zoomed() {
    if no_gpu() {
        return;
    }
    // phy's PanZoom on Raster (time and rows, GPU dots included) and Rate
    // (time, re-binned over the visible span).
    each_theme(|theme, name| {
        let mut h = fixture(&format!("panzoom-{name}"), 40, theme);
        h.key_press(egui::Key::ArrowDown);
        settle(&mut h);
        click_tab(&mut h, "Raster");
        wheel_zoom(&mut h, "raster plot");
        snap(&mut h, "raster_zoomed", name);
        click_tab(&mut h, "Rate");
        wheel_zoom(&mut h, "rate plot");
        snap(&mut h, "rate_zoomed", name);
        h
    });
}

/// An MCP agent ("codex") connected to the window merges a cluster: the
/// top bar names it with a lit activity dot, the toast says what it did,
/// and the touched cluster is selected and highlighted in the table.
#[test]
fn agent_edit_toast_and_highlight() {
    if no_gpu() {
        return;
    }
    use std::io::{BufRead, Write};
    each_theme(|theme, name| {
        let root = dataset(&format!("agent-edit-{name}"), 3);
        let dir = root.clone();
        let mut h = harness_with(theme, move || {
            let (session, positions) = open(&dir);
            let mut app = SorrelApp::new(session, label_str);
            configure(&mut app, &dir, positions);
            app.serve_mcp().unwrap();
            // The toast and tint must outlast however long the renderer takes.
            app.set_feedback_duration(Duration::from_secs(24 * 3600));
            app
        });
        settle(&mut h);
        let (port, token) = sorrel_data::serve::read_mcp_json(&root).unwrap();
        let mut agent = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        let mut replies = std::io::BufReader::new(agent.try_clone().unwrap());
        let mut line = String::new();
        for msg in [
            format!(r#"{{"token":"{token}"}}"#),
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"clientInfo":{"name":"codex"}}}"#.into(),
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"merge","arguments":{"sources":[1],"target":2}}}"#.into(),
        ] {
            writeln!(agent, "{msg}").unwrap();
            replies.read_line(&mut line).unwrap();
        }
        assert!(line.contains(r#"\"ok\":true"#), "{line}");
        snap(&mut h, "agent_edit", name);
        drop(agent); // after the snapshot: the indicator lists connected agents
        h
    });
}

/// Two agent proposals waiting in the Proposals pane.
#[test]
fn proposals_pane() {
    if no_gpu() {
        return;
    }
    use sorrel_data::proposals::{Action, Proposal, Source};
    use sorrel_io::ClusterId;
    each_theme(|theme, name| {
        let mut h = fixture(&format!("proposals-{name}"), 3, theme);
        for (action, confidence, reasons) in [
            (
                Action::Merge {
                    clusters: vec![ClusterId(2), ClusterId(1)],
                },
                0.9,
                vec!["CCG dip at 0 ms", "templates nearly identical"],
            ),
            (
                Action::Relabel {
                    cluster: ClusterId(0),
                    label: "noise".into(),
                },
                0.62,
                vec!["footprint spread over every channel"],
            ),
        ] {
            let reasons = reasons.into_iter().map(String::from).collect();
            let mut p = Proposal::new(
                Source::Agent,
                action,
                confidence,
                reasons,
                Default::default(),
            );
            p.proposer = Some("codex".into());
            h.state().host().propose(p);
        }
        click_tab(&mut h, "Proposals");
        snap(&mut h, "proposals_pane", name);
        h
    });
}

/// The History pane over a journal with a branch: you relabelled two
/// clusters, undid the second, then codex merged instead (a second branch
/// off the first edit) and you saved. Times are fixed in the records.
#[test]
fn history_with_a_branch() {
    if no_gpu() {
        return;
    }
    use sorrel_data::{Actor, CurationCommand, PhyLabelOp, Record};
    use sorrel_io::ClusterId;
    each_theme(|theme, name| {
        let root = dataset(&format!("history-{name}"), 3);
        let mut j = Journal::open_or_create(&root.join("sorrel.journal"), 0, 3).unwrap();
        let t0 = 1_793_200_000_000; // 15:06:40 UTC
        let node = |id, parent, actor, min: u64, cmd| Record::Node {
            id,
            parent,
            actor,
            ts_ms: t0 + min * 60_000,
            cmd,
        };
        let codex = Actor::Agent {
            client: "codex".into(),
        };
        for rec in [
            node(
                0,
                None,
                Actor::Human,
                0,
                CurationCommand::Relabel {
                    cluster: ClusterId(0),
                    op: PhyLabelOp::SetGood,
                },
            ),
            node(
                1,
                Some(0),
                Actor::Human,
                1,
                CurationCommand::Relabel {
                    cluster: ClusterId(1),
                    op: PhyLabelOp::SetMua,
                },
            ),
            Record::Head { node: Some(0) },
            node(
                2,
                Some(0),
                codex,
                3,
                CurationCommand::Merge {
                    sources: vec![ClusterId(1)],
                    target: ClusterId(2),
                },
            ),
            Record::Seal {
                node: Some(2),
                artifacts: 0,
            },
        ] {
            j.append(&rec).unwrap();
        }
        drop(j);
        let dir = root.clone();
        let mut h = harness_with(theme, move || {
            let (session, positions) = open(&dir);
            let mut app = SorrelApp::new(session, label_str);
            configure(&mut app, &dir, positions);
            app
        });
        settle(&mut h);
        click_tab(&mut h, "History");
        snap(&mut h, "history_branch", name);
        h
    });
}

/// Drag the primary button through `path` (one frame per point) with `m`
/// held.
fn drag_path(h: &mut Harness<'static, App>, path: &[egui::Pos2], m: egui::Modifiers) {
    h.hover_at(path[0]);
    h.input_mut().modifiers = m;
    let btn = |pos, pressed| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: m,
    };
    h.event(btn(path[0], true));
    h.run_ok();
    for &p in &path[1..] {
        h.event(egui::Event::PointerMoved(p));
        h.run_ok();
    }
    h.event(btn(*path.last().unwrap(), false));
    h.run_ok();
    h.input_mut().modifiers = egui::Modifiers::NONE;
    settle(h);
}

#[test]
fn features_lasso_freehand_and_edit() {
    if no_gpu() {
        return;
    }
    // Ctrl+drag draws a closed lasso freehand; a plain drag then moves one
    // of its nodes without panning the plot.
    each_theme(|theme, name| {
        let mut h = fixture(&format!("features-freehand-{name}"), 3, theme);
        select_two_and_open_features(&mut h);
        let cell = h.get_by_label("feature cell 1").rect();
        let (c, r) = (cell.center(), cell.size() * 0.35);
        // A square stroke, eight points per side, ending short of the start.
        let corners = [
            (-1.0, -1.0),
            (1.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, -0.95),
        ];
        let at = |(x, y): (f32, f32)| c + egui::vec2(x, y) * r;
        let stroke: Vec<egui::Pos2> = corners
            .windows(2)
            .flat_map(|w| (0..8).map(move |k| (w[0], w[1], k as f32 / 8.0)))
            .map(|(a, b, t)| at((a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t)))
            .chain([at(corners[4])])
            .collect();
        drag_path(&mut h, &stroke, egui::Modifiers::COMMAND);
        // The lasso's live-tracking switch joins the top toolbar row; it
        // starts off, so dragging a node settles the hits on release.
        h.get_by_label("live tracking");
        snap(&mut h, "features_lasso_freehand", name);
        // Pull the bottom-right node towards the middle.
        drag_path(
            &mut h,
            &[at((1.0, 1.0)), at((0.6, 0.6)), at((0.2, 0.2))],
            egui::Modifiers::NONE,
        );
        snap(&mut h, "features_lasso_edit", name);
        h.key_press(egui::Key::K);
        settle(&mut h);
        let s = sorrel_ui::jobs::read(h.state().session()).history_len();
        assert_eq!(s, 1, "the edited lasso still splits as one entry");
        h
    });
}

#[test]
fn pane_maximized() {
    if no_gpu() {
        return;
    }
    each_theme(|theme, name| {
        let root = dataset(&format!("maximized-{name}"), 3);
        let mut h = harness_sized(theme, [1920.0, 1080.0], move || {
            let (session, positions) = open(&root);
            let mut app = SorrelApp::new(session, label_str);
            configure(&mut app, &root, positions);
            app
        });
        settle(&mut h);
        select_two_and_open_features(&mut h);
        // Every tab group has the button; Features sits in the second group
        // (shape, compare, assess), so take the second.
        h.get_all_by_label("maximize pane")
            .nth(1)
            .expect("the compare group's maximize button")
            .click();
        settle(&mut h);
        h.get_by_label("Features · maximized");
        assert!(
            h.query_by_label("Waveforms").is_none(),
            "the tree is not drawn while a pane is maximized"
        );
        snap(&mut h, "features_maximized", name);
        h.key_press(egui::Key::Escape);
        settle(&mut h);
        h.get_by_label("Waveforms");
        assert!(h.query_by_label("Features · maximized").is_none());
        h
    });
}

#[test]
fn features_3d() {
    if no_gpu() {
        return;
    }
    each_theme(|theme, name| {
        let mut h = fixture(&format!("features-3d-{name}"), 3, theme);
        select_two_and_open_features(&mut h);
        h.get_by_label("3D").click();
        settle(&mut h);
        h.get_by_label("feature 3d canvas");
        snap(&mut h, "features_3d", name);
        h
    });
}

#[test]
fn features_3d_kmeans_prism() {
    if no_gpu() {
        return;
    }
    // The k-means button's auto-lasso, then the 3-D view shows its prism and
    // the exact part's hull.
    each_theme(|theme, name| {
        let mut h = fixture(&format!("features-3d-km-{name}"), 3, theme);
        select_two_and_open_features(&mut h);
        h.get_by_label("k-means").click();
        settle(&mut h);
        h.get_by_label("3D").click();
        settle(&mut h);
        h.get_by_label("feature 3d canvas");
        snap(&mut h, "features_3d_kmeans_prism", name);
        h
    });
}

#[test]
fn features_kmeans_collision() {
    if no_gpu() {
        return;
    }
    // k-means on two clusters in a wide window: collision evidence and the
    // part's hull are drawn on the actual sixteen projections, not a navigator.
    each_theme(|theme, name| {
        let root = dataset(&format!("features-collision-{name}"), 3);
        let mut h = harness_sized(theme, [2200.0, 1100.0], move || {
            let (session, positions) = open(&root);
            let mut app = SorrelApp::new(session, label_str);
            configure(&mut app, &root, positions);
            app
        });
        settle(&mut h);
        select_two_and_open_features(&mut h);
        h.get_by_label("k-means").click();
        settle(&mut h);
        assert_full_feature_grid(&h);
        snap(&mut h, "features_kmeans_collision", name);
        h
    });
}

#[test]
fn features_compact() {
    if no_gpu() {
        return;
    }
    // The default narrow pane still shows the complete matrix. Focusing a
    // projection never replaces its fifteen siblings with a navigator.
    each_theme(|theme, name| {
        let mut h = fixture(&format!("features-compact-{name}"), 3, theme);
        select_two_and_open_features(&mut h);
        assert_full_feature_grid(&h);
        h.get_by_label("k-means").click();
        settle(&mut h);
        snap(&mut h, "features_compact", name);
        h.get_by_label("feature cell 6").click();
        settle(&mut h);
        h.get_by_label("feature cell 6");
        assert_full_feature_grid(&h);
        h
    });
}

#[test]
fn features_3d_maximized() {
    if no_gpu() {
        return;
    }
    // Wide pane: the axes sidebar and legend instead of the combo box.
    each_theme(|theme, name| {
        let root = dataset(&format!("maximized-3d-{name}"), 3);
        let mut h = harness_sized(theme, [1920.0, 1080.0], move || {
            let (session, positions) = open(&root);
            let mut app = SorrelApp::new(session, label_str);
            configure(&mut app, &root, positions);
            app
        });
        settle(&mut h);
        select_two_and_open_features(&mut h);
        h.get_by_label("3D").click();
        settle(&mut h);
        h.get_all_by_label("maximize pane")
            .nth(1)
            .expect("the compare group's maximize button")
            .click();
        settle(&mut h);
        h.get_by_label("feature 3d canvas");
        snap(&mut h, "features_3d_maximized", name);
        h
    });
}

#[test]
fn waveforms_feature_link() {
    if no_gpu() {
        return;
    }
    // Cross-talk: a Shift+clicked Waveforms channel joins the Features set,
    // a lasso in Features draws its spikes (white/black traces) on the
    // focused channels, and the others dim.
    each_theme(|theme, name| {
        let root = dataset(&format!("waveforms-link-{name}"), 3);
        let mut h = harness_sized(theme, [2200.0, 1100.0], move || {
            let (session, positions) = open(&root);
            let mut app = SorrelApp::new(session, label_str);
            configure(&mut app, &root, positions);
            app
        });
        settle(&mut h);
        select_two_and_open_features(&mut h);
        click_tab(&mut h, "Waveforms");
        let pane = h
            .get_all_by_label_contains("waveform channel ")
            .next()
            .expect("a real waveform channel box")
            .rect()
            .center();
        h.hover_at(pane);
        h.input_mut().modifiers = egui::Modifiers::SHIFT;
        for pressed in [true, false] {
            h.event(egui::Event::PointerButton {
                pos: pane,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::SHIFT,
            });
            h.run_ok();
        }
        h.input_mut().modifiers = egui::Modifiers::NONE;
        settle(&mut h);
        click_tab(&mut h, "Features");
        lasso_points(&mut h, 4);
        click_tab(&mut h, "Waveforms");
        snap(&mut h, "waveforms_feature_link", name);
        h
    });
}

#[test]
fn features_alternatives() {
    if no_gpu() {
        return;
    }
    // With a lasso drawn, the first click on an alternative
    // only arms its promotion (a warning row names what is dropped), the
    // second promotes it to A/B and the lasso is gone.
    each_theme(|theme, name| {
        let root = dataset(&format!("features-alternatives-{name}"), 3);
        let mut h = harness_sized(theme, [2200.0, 1100.0], move || {
            let (session, positions) = open(&root);
            let mut app = SorrelApp::new(session, label_str);
            configure(&mut app, &root, positions);
            app
        });
        settle(&mut h);
        select_two_and_open_features(&mut h);
        h.get_by_label("ALTERNATIVES").click();
        settle(&mut h);
        assert!(
            h.query_all_by_label_contains("alternative ").count() > 0,
            "the strip lists alternative views"
        );
        lasso_points(&mut h, 4);
        let brush = h
            .state()
            .brushes()
            .spikes
            .as_ref()
            .expect("lasso brush")
            .key();
        h.get_all_by_label_contains("alternative ")
            .next()
            .expect("first alternative")
            .click();
        snap(&mut h, "features_alternatives", name);
        assert_eq!(
            h.state()
                .brushes()
                .spikes
                .as_ref()
                .expect("first click preserves the lasso")
                .key(),
            brush
        );
        h.get_all_by_label_contains("alternative ")
            .next()
            .expect("first alternative")
            .click();
        settle(&mut h);
        assert!(
            h.state().brushes().spikes.is_none(),
            "promoting drops the lasso"
        );
        h
    });
}

#[test]
fn features_3d_pinned() {
    if no_gpu() {
        return;
    }
    // Focus another actual 2-D projection, then switch to 3-D: the
    // focused plot's pair is pinned (the chip), so only its third axis is
    // chosen. The chip's × lists every triple again.
    each_theme(|theme, name| {
        let mut h = fixture(&format!("features-3d-pinned-{name}"), 3, theme);
        select_two_and_open_features(&mut h);
        h.get_by_label("feature cell 2").click();
        settle(&mut h);
        h.get_by_label("feature cell 2");
        h.get_by_label("3D").click();
        settle(&mut h);
        h.get_by_label("feature 3d canvas");
        h.get_by_label("pin focused plot");
        snap(&mut h, "features_3d_pinned", name);
        h.get_by_label("pin focused plot").click();
        settle(&mut h);
        assert!(
            h.query_by_label("pin focused plot").is_none(),
            "× unpins and lists every triple"
        );
        h
    });
}

#[test]
fn xtalk_probe_set() {
    if no_gpu() {
        return;
    }
    // Cross-talk: Shift+click on a Probe site adds the channel to the
    // Features set (one more toolbar chip); A / B badges and set rings
    // mark the members on the probe.
    each_theme(|theme, name| {
        let root = dataset(&format!("xtalk-probe-{name}"), 3);
        let mut h = harness_sized(theme, [2200.0, 1100.0], move || {
            let (session, positions) = open(&root);
            let mut app = SorrelApp::new(session, label_str);
            configure(&mut app, &root, positions);
            app
        });
        settle(&mut h);
        select_two_and_open_features(&mut h);
        click_tab(&mut h, "Probe");
        // The set chips are buttons labelled "5", "5 A", "7 B".
        let chips = |h: &Harness<'static, App>| {
            h.get_all_by_role(Role::Button)
                .filter(|n| {
                    let l = n.accesskit_node().label().unwrap_or_default();
                    let l = l.strip_suffix(" A").or(l.strip_suffix(" B")).unwrap_or(&l);
                    !l.is_empty() && l.bytes().all(|b| b.is_ascii_digit())
                })
                .count()
        };
        let before = chips(&h);
        assert!(before >= 2, "A and B chips");
        // The last site (the one most likely outside the set).
        let sites: Vec<_> = h.get_all_by_label_contains("probe site ch").collect();
        assert!(!sites.is_empty(), "probe sites are labelled");
        sites
            .last()
            .unwrap()
            .click_modifiers(egui::Modifiers::SHIFT);
        settle(&mut h);
        assert_eq!(chips(&h), before + 1, "Shift+click adds the channel");
        snap(&mut h, "xtalk_probe_set", name);
        h
    });
}

#[test]
fn xtalk_time_brush() {
    if no_gpu() {
        return;
    }
    // Cross-talk: Shift+drag along Amplitudes sets the app-wide time range;
    // the top bar shows its chip and the band covers the plot.
    each_theme(|theme, name| {
        let root = dataset(&format!("xtalk-time-{name}"), 3);
        let mut h = harness_sized(theme, [2200.0, 1100.0], move || {
            let (session, positions) = open(&root);
            let mut app = SorrelApp::new(session, label_str);
            configure(&mut app, &root, positions);
            app
        });
        settle(&mut h);
        select_two_and_open_features(&mut h);
        click_tab(&mut h, "Amplitudes");
        let plot = h.get_by_label("amplitude plot").rect();
        let y = plot.center().y;
        let (a, b) = (
            egui::pos2(plot.left() + plot.width() * 0.3, y),
            egui::pos2(plot.left() + plot.width() * 0.6, y),
        );
        let shift = egui::Modifiers::SHIFT;
        h.input_mut().modifiers = shift;
        h.hover_at(a);
        h.event(egui::Event::PointerButton {
            pos: a,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: shift,
        });
        h.run_ok();
        h.hover_at(b);
        h.run_ok();
        h.event(egui::Event::PointerButton {
            pos: b,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: shift,
        });
        h.run_ok();
        h.input_mut().modifiers = egui::Modifiers::NONE;
        settle(&mut h);
        h.get_all_by_label_contains("time ")
            .find(|n| {
                n.accesskit_node()
                    .label()
                    .is_some_and(|l| l.starts_with("time ") && l.contains('–'))
            })
            .expect("the time-range chip in the top bar");
        snap(&mut h, "xtalk_time_brush", name);
        h
    });
}

#[test]
fn xtalk_features_brush() {
    if no_gpu() {
        return;
    }
    // Cross-talk: a Features lasso becomes the app-wide spike brush (the top
    // bar chip, brushed spikes in ink on top), and a Shift+drag along the
    // time cell sets the time brush: the band shows on the time axis and the
    // spikes outside it fade.
    each_theme(|theme, name| {
        let root = dataset(&format!("xtalk-features-brush-{name}"), 3);
        let mut h = harness_sized(theme, [2200.0, 1100.0], move || {
            let (session, positions) = open(&root);
            let mut app = SorrelApp::new(session, label_str);
            configure(&mut app, &root, positions);
            app
        });
        settle(&mut h);
        select_two_and_open_features(&mut h);
        assert!(
            h.query_by_label_contains("brush:").is_none(),
            "no brush before the lasso"
        );
        lasso_points(&mut h, 4);
        h.get_by_label_contains("brush:");
        h.get_by_label_contains("Features lasso");
        // Shift+drag across the middle of the time cell (x = time).
        let cell = h.get_by_label("feature cell 0").rect();
        let y = cell.center().y;
        let at = |f: f32| egui::pos2(cell.left() + f * cell.width(), y);
        let path: Vec<egui::Pos2> = (0..=6).map(|k| at(0.3 + 0.05 * k as f32)).collect();
        drag_path(&mut h, &path, egui::Modifiers::SHIFT);
        h.get_by_label_contains("time ");
        h.get_by_label_contains("brush:");
        snap(&mut h, "xtalk_features_brush", name);
        h
    });
}

#[test]
fn xtalk_3d_space() {
    if no_gpu() {
        return;
    }
    // Cross-talk: Features 3-D gains the space family (spike depth) as an
    // axis, ranked with the PCs; a 2-D lasso made before switching shows as
    // brushed spikes in ink in the scene and as a legend entry.
    each_theme(|theme, name| {
        let root = dataset(&format!("xtalk-3d-space-{name}"), 3);
        let mut h = harness_sized(theme, [2200.0, 1100.0], move || {
            let (session, positions) = open(&root);
            let mut app = SorrelApp::new(session, label_str);
            configure(&mut app, &root, positions);
            app
        });
        settle(&mut h);
        select_two_and_open_features(&mut h);
        lasso_points(&mut h, 4);
        h.get_by_label_contains("brush:");
        h.get_by_label("3D").click();
        settle(&mut h);
        h.get_by_label("feature 3d canvas");
        h.get_by_label_contains("brush · ");
        // The fixture has depths: a triple with the depth axis is listed.
        let row = h
            .get_all_by_label_contains("axes ")
            .find(|n| {
                n.accesskit_node()
                    .label()
                    .is_some_and(|l| l.contains("depth"))
            })
            .expect("a triple with depth");
        row.click_accesskit();
        settle(&mut h);
        snap(&mut h, "xtalk_3d_space", name);
        h
    });
}

/// Give every cluster some sub-millisecond doublets: a twin spike 12 samples
/// (0.4 ms) after every eighth spike, so ISI violators exist. Rows are
/// duplicated in every per-spike array.
fn add_isi_doublets(root: &Path) {
    let raw = std::fs::read(root.join("spike_times.npy")).unwrap();
    let hlen = u16::from_le_bytes([raw[8], raw[9]]) as usize;
    let times: Vec<u64> = raw[10 + hlen..]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| u64::from_le_bytes(*b))
        .collect();
    let clusters = npy::read_1d_u32(&root.join("spike_clusters.npy")).unwrap();
    let amps = npy::read_1d_f32(&root.join("amplitudes.npy")).unwrap();
    let (pcs, _) = npy::read_npy_f32_flat(&root.join("pc_features.npy")).unwrap();
    let (tf, _) = npy::read_npy_f32_flat(&root.join("template_features.npy")).unwrap();
    let snr = npy::read_1d_f32(&root.join("spike_snr.npy")).unwrap();
    let (mut t2, mut c2, mut a2, mut p2) = (vec![], vec![], vec![], vec![]);
    let (mut f2, mut s2) = (vec![], vec![]);
    for i in 0..times.len() {
        let mut push = |t: u64| {
            t2.push(t);
            c2.push(clusters[i]);
            a2.push(amps[i]);
            p2.extend_from_slice(&pcs[i * 12..i * 12 + 12]);
            f2.extend_from_slice(&tf[i * 2..i * 2 + 2]);
            s2.push(snr[i]);
        };
        push(times[i]);
        if i % 8 == 0 && times.get(i + 1).is_none_or(|&n| n > times[i] + 13) {
            push(times[i] + 12);
        }
    }
    let n = t2.len();
    npy::write_1d_u64_atomic(&root.join("spike_times.npy"), &t2).unwrap();
    npy::write_1d_u32_atomic(&root.join("spike_clusters.npy"), &c2).unwrap();
    npy::write_1d_u32_atomic(&root.join("spike_templates.npy"), &c2).unwrap();
    npy::write_f32_atomic(&root.join("amplitudes.npy"), &[n], &a2).unwrap();
    npy::write_f32_atomic(&root.join("pc_features.npy"), &[n, 3, 4], &p2).unwrap();
    npy::write_f32_atomic(&root.join("template_features.npy"), &[n, 2], &f2).unwrap();
    npy::write_f32_atomic(&root.join("spike_snr.npy"), &[n], &s2).unwrap();
}

#[test]
fn xtalk_isi_whatif() {
    if no_gpu() {
        return;
    }
    // Clicking the ISI refractory region brushes both spikes of every pair
    // under 1 ms (the top-bar chip names it); with a lasso brush up instead,
    // the ISI pane overlays the split what-if (inside dashed, outside dotted).
    each_theme(|theme, name| {
        let root = dataset(&format!("xtalk-isi-{name}"), 3);
        add_isi_doublets(&root);
        let mut h = harness_sized(theme, [2200.0, 1100.0], move || {
            let (session, positions) = open(&root);
            let mut app = SorrelApp::new(session, label_str);
            configure(&mut app, &root, positions);
            app
        });
        settle(&mut h);
        select_two_and_open_features(&mut h);
        lasso_points(&mut h, 4);
        settle(&mut h);
        h.get_by_label_contains("brush:");
        click_tab(&mut h, "ISI");
        settle(&mut h);
        h.get_by_label_contains("if split: inside");
        snap(&mut h, "xtalk_isi_whatif_split", name);

        // The refractory band then brushes the violators instead.
        h.get_all_by_label_contains("isi violators c")
            .next()
            .expect("per-cluster violators chip")
            .click();
        settle(&mut h);
        h.get_by_label_contains("ISI violators");
        assert!(h.query_by_label_contains("if split").is_none());
        snap(&mut h, "xtalk_isi_whatif", name);
        h
    });
}

#[test]
fn xtalk_merge_ghost() {
    if no_gpu() {
        return;
    }
    // Cross-talk: hovering a Similar row ghosts that cluster in Features
    // (grey points, "cN (ghost)") and shows the merge's quality change.
    each_theme(|theme, name| {
        let root = dataset(&format!("xtalk-ghost-{name}"), 3);
        let mut h = harness_sized(theme, [2200.0, 1100.0], move || {
            let (session, positions) = open(&root);
            let mut app = SorrelApp::new(session, label_str);
            configure(&mut app, &root, positions);
            app
        });
        settle(&mut h);
        h.key_press(egui::Key::ArrowDown);
        settle(&mut h);
        click_tab(&mut h, "Similar");
        // The first Similar row: its "mua" label sits right of the cluster
        // table (which has its own "mua" cell).
        let row = h
            .get_all_by_label("mua")
            .map(|n| n.rect())
            .find(|r| r.min.x > 900.0)
            .expect("a Similar row");
        h.hover_at(row.center());
        settle(&mut h);
        h.hover_at(row.center());
        settle(&mut h);
        // Waveforms (visible beside it) draws the ghost's dashed mean, and
        // the row's tooltip names the merge.
        h.get_by_label_contains("merge c");
        snap(&mut h, "xtalk_merge_ghost", name);
        h
    });
}

/// The wide (2200 x 1100) harness: Features, Waveforms and the history /
/// similar group all fit side by side.
fn wide_fixture(slot: &str, theme: egui::Theme) -> Harness<'static, App> {
    let root = dataset(slot, 3);
    let mut h = harness_sized(theme, [2200.0, 1100.0], move || {
        let (session, positions) = open(&root);
        let mut app = SorrelApp::new(session, label_str);
        configure(&mut app, &root, positions);
        app
    });
    settle(&mut h);
    h
}

/// The tab-bar button `label` of the `n`-th tab group, left to right.
fn group_button<'a>(
    h: &'a Harness<'static, App>,
    label: &'a str,
    n: usize,
) -> egui_kittest::Node<'a> {
    let mut all: Vec<_> = h.get_all_by_label(label).collect();
    all.sort_by(|a, b| a.rect().min.x.total_cmp(&b.rect().min.x));
    all.into_iter().nth(n).expect("the group's tab-bar button")
}

/// Pop `Similar` out of the compare group (the second), then show Features
/// there: the popped tab stays active until another is clicked.
fn pop_out_similar(h: &mut Harness<'static, App>) {
    click_tab(h, "Similar");
    group_button(h, "pop out pane", 1).click();
    settle(h);
    click_tab(h, "Features");
}

/// The Similar row whose label is `label`, in the popped-out Similar window.
/// The embedded window floats over the cluster table, so its label overlaps
/// the table's own; the window's header puts it higher.
fn similar_row(h: &Harness<'static, App>, label: &str) -> egui::Rect {
    h.get_all_by_label(label)
        .map(|n| n.rect())
        .min_by(|a, b| a.min.y.total_cmp(&b.min.y))
        .expect("a Similar row")
}

#[test]
fn pane_popout() {
    if no_gpu() {
        return;
    }
    // Pop-out: the tab-bar button moves the pane to its own window (an
    // embedded egui window under kittest) and leaves a placeholder with
    // "Bring back" in the tree; Bring back undoes it.
    each_theme(|theme, name| {
        let mut h = wide_fixture(&format!("popout-{name}"), theme);
        select_two_and_open_features(&mut h);
        assert!(h.query_by_label("In its own window").is_none());
        // Features sits in the second group (shape, compare, assess).
        group_button(&h, "pop out pane", 1).click();
        settle(&mut h);
        h.get_by_label("In its own window");
        h.get_by_label("Bring back");
        h.get_by_label("Features — Sorrel");
        assert!(
            h.get_all_by_label_contains("feature cell").count() > 0,
            "the pane is drawn in its window"
        );
        snap(&mut h, "pane_popout", name);
        h.get_by_label("Bring back").click();
        settle(&mut h);
        assert!(h.query_by_label("In its own window").is_none());
        assert!(h.query_by_label("Features — Sorrel").is_none());
        h.get_by_label("feature cell 1");
        // Similar popped out leaves Features in the main window.
        pop_out_similar(&mut h);
        h.get_by_label("Similar — Sorrel");
        h.get_by_label("feature cell 1");
        snap(&mut h, "pane_popout_similar", name);
        h
    });
}

#[test]
fn xtalk_features_ghost() {
    if no_gpu() {
        return;
    }
    // Cross-talk: hovering a Similar row ghosts that cluster in Features even
    // though the two share a tab group (Similar is popped out): grey points
    // and "cN (ghost)".
    each_theme(|theme, name| {
        let mut h = wide_fixture(&format!("xtalk-fghost-{name}"), theme);
        // c0 + c1 selected, c2 the ghost: the fixture's units are zero off
        // their own channels, so c2 shows as a grey blob at the origin of the
        // cells that pair c0's channel with c1's.
        h.key_press(egui::Key::ArrowDown);
        settle(&mut h);
        h.get_all_by_label("1")
            .min_by(|a, b| a.rect().min.x.total_cmp(&b.rect().min.x))
            .expect("cluster 1 row")
            .click_modifiers(egui::Modifiers::COMMAND);
        click_tab(&mut h, "Features");
        pop_out_similar(&mut h);
        h.get_by_label("feature cell 1");
        assert!(h.state().brushes().ghost.is_none(), "nothing hovered yet");
        let at = similar_row(&h, "noise").center();
        // Hover is promoted at the end of the frame: one frame to write it,
        // one to read it.
        h.hover_at(at);
        settle(&mut h);
        h.hover_at(at);
        settle(&mut h);
        assert!(
            matches!(h.state().brushes().ghost, Some(Ghost::Merge(c)) if c == ClusterId(2)),
            "a Similar row hover is a merge ghost of c2, got {:?}",
            h.state().brushes().ghost
        );
        // Snap removes the cursor, which would drop the hover: keep the
        // ghost by snapshotting before the pointer leaves.
        snap_hovered(&mut h, "xtalk_features_ghost", name, at, 0);
        h
    });
}

/// `snap` with the pointer kept at `at` (the state under test is a hover).
/// `slack` pixels may differ.
fn snap_hovered(
    h: &mut Harness<'static, App>,
    state: &str,
    theme: &str,
    at: egui::Pos2,
    slack: usize,
) {
    settle(h);
    h.hover_at(at);
    h.run_ok();
    h.hover_at(at);
    h.run_ok();
    if let Some(dir) = std::env::var_os("SORREL_DESIGN_DIR") {
        design::dump(h, Path::new(&dir), state, theme);
    }
    h.snapshot_options(
        format!("{state}_{theme}"),
        &egui_kittest::SnapshotOptions::new().failed_pixel_count_threshold(slack),
    );
}

#[test]
fn xtalk_history_ghost() {
    if no_gpu() {
        return;
    }
    // Cross-talk: hovering a History row draws that edit's "before #n"
    // clusters as hulls in the Features 3-D view, with a legend entry.
    each_theme(|theme, name| {
        let mut h = wide_fixture(&format!("xtalk-hghost-{name}"), theme);
        select_two_and_open_features(&mut h);
        lasso_and_split(&mut h);
        h.key_press(egui::Key::K);
        settle(&mut h);
        h.get_by_label("3D").click();
        settle(&mut h);
        h.get_by_label("feature 3d canvas");
        // History sits in the third tab group, beside Features' group.
        click_tab(&mut h, "History");
        assert!(h.query_by_label_contains("before #").is_none());
        let row = h
            .get_all_by_label_contains("split cluster")
            .find(|n| {
                n.accesskit_node()
                    .label()
                    .is_some_and(|l| l.starts_with("split cluster"))
            })
            .expect("the split's History row");
        let at = row.rect().center();
        h.hover_at(at);
        settle(&mut h);
        h.hover_at(at);
        settle(&mut h);
        assert!(
            matches!(h.state().brushes().ghost, Some(Ghost::History(_))),
            "a History row hover is a history ghost, got {:?}",
            h.state().brushes().ghost
        );
        h.get_by_label_contains("before #");
        snap_hovered(&mut h, "xtalk_history_ghost", name, at, 0);
        h
    });
}

#[test]
fn xtalk_hover_spike() {
    if no_gpu() {
        return;
    }
    // Cross-talk: the spike under the pointer in a Features cell is written
    // to the shared hover, and Waveforms (visible beside it) draws that spike
    // as a bold accent waveform.
    each_theme(|theme, name| {
        let mut h = wide_fixture(&format!("xtalk-hover-{name}"), theme);
        select_two_and_open_features(&mut h);
        assert!(h.state().brushes().hover.is_none());
        let cell = h.get_by_label("feature cell 1").rect();
        // Scan the cell on a coarse grid until the 6 px pick lands on a
        // spike; hover is promoted at the end of the frame, so two frames.
        let step = 4.0;
        let mut found = None;
        let mut tried = 0;
        'scan: for iy in 0..(cell.height() / step) as usize {
            for ix in 0..(cell.width() / step) as usize {
                tried += 1;
                let p = cell.min + egui::vec2(ix as f32, iy as f32) * step;
                h.hover_at(p);
                h.run_ok();
                h.run_ok();
                if h.state().brushes().hover.is_some() {
                    found = Some(p);
                    break 'scan;
                }
            }
        }
        let at = found.unwrap_or_else(|| {
            panic!("no spike under any of {tried} pointer positions in feature cell 1 ({cell:?})")
        });
        let hover = h.state().brushes().hover.unwrap();
        assert!(hover.sample > 0, "a real spike: {hover:?}");
        snap_hovered(&mut h, "xtalk_hover_spike", name, at, 0);
        assert!(h.state().brushes().hover.is_some(), "still hovered");
        h
    });
}

#[test]
fn xtalk_amplitudes_ink() {
    if no_gpu() {
        return;
    }
    // Cross-talk: a Features lasso is the app-wide spike brush; Amplitudes
    // (same tab group, opened afterwards) draws those spikes in ink.
    each_theme(|theme, name| {
        let mut h = wide_fixture(&format!("xtalk-amp-ink-{name}"), theme);
        select_two_and_open_features(&mut h);
        lasso_points(&mut h, 4);
        let brush = h.state().brushes().spikes.clone().expect("a spike brush");
        assert_eq!(brush.source, BrushSource::FeatureLasso);
        assert!(!brush.is_empty());
        click_tab(&mut h, "Amplitudes");
        h.get_by_label("amplitude plot");
        h.get_by_label_contains("brush:");
        snap(&mut h, "xtalk_amplitudes_ink", name);
        assert!(
            h.state().brushes().spikes.is_some(),
            "the brush survives the tab switch"
        );
        h
    });
}

#[test]
fn xtalk_isi_tooltip() {
    if no_gpu() {
        return;
    }
    // Cross-talk: hovering the ISI refractory band says how many spikes a
    // click would brush.
    each_theme(|theme, name| {
        let root = dataset(&format!("xtalk-isi-tip-{name}"), 3);
        add_isi_doublets(&root);
        let mut h = harness_sized(theme, [2200.0, 1100.0], move || {
            let (session, positions) = open(&root);
            let mut app = SorrelApp::new(session, label_str);
            configure(&mut app, &root, positions);
            app
        });
        settle(&mut h);
        select_two_and_open_features(&mut h);
        click_tab(&mut h, "ISI");
        let band = h.get_by_label("isi violators").rect();
        let at = band.center();
        h.hover_at(at);
        settle(&mut h);
        h.hover_at(at);
        settle(&mut h);
        h.get_by_label_contains("violating pairs");
        snap_hovered(&mut h, "xtalk_isi_tooltip", name, at, 0);
        h
    });
}

#[test]
fn xtalk_brush_cleared_with_the_selection() {
    if no_gpu() {
        return;
    }
    // Deselecting every cluster takes back the Features lasso's brush, and
    // Amplitudes (whose kept set still has the old slots) must not panic.
    each_theme(|theme, name| {
        let mut h = wide_fixture(&format!("xtalk-desel-{name}"), theme);
        select_two_and_open_features(&mut h);
        lasso_points(&mut h, 4);
        settle(&mut h);
        assert!(h.state().brushes().spikes.is_some(), "the lasso is a brush");
        click_tab(&mut h, "Amplitudes");
        for id in ["2", "0"] {
            h.get_all_by_label(id)
                .min_by(|a, b| a.rect().min.x.total_cmp(&b.rect().min.x))
                .expect("cluster row")
                .click_modifiers(egui::Modifiers::COMMAND);
            settle(&mut h);
        }
        click_tab(&mut h, "Features");
        settle(&mut h);
        assert!(
            h.state().brushes().spikes.is_none(),
            "no selection, no Features brush"
        );
        h
    });
}

#[test]
fn xtalk_merge_hulls() {
    if no_gpu() {
        return;
    }
    // Cross-talk: a merged unit shows its sources as dashed "before merge"
    // hulls in the Features 3-D view without hovering its History row.
    each_theme(|theme, name| {
        let mut h = wide_fixture(&format!("xtalk-mhulls-{name}"), theme);
        h.key_press(egui::Key::ArrowDown);
        settle(&mut h);
        h.get_all_by_label("1")
            .min_by(|a, b| a.rect().min.x.total_cmp(&b.rect().min.x))
            .expect("cluster 1 row")
            .click_modifiers(egui::Modifiers::COMMAND);
        settle(&mut h);
        // G merges c0 and c1; the merged unit stays selected.
        h.key_press(egui::Key::G);
        settle(&mut h);
        click_tab(&mut h, "Features");
        h.get_by_label("3D").click();
        settle(&mut h);
        h.get_by_label("feature 3d canvas");
        assert!(
            h.state().brushes().ghost.is_none(),
            "no hover drives the hulls"
        );
        h.get_by_label_contains("before merge (c");
        snap(&mut h, "xtalk_merge_hulls", name);
        h
    });
}

#[test]
fn xtalk_3d_positions() {
    if no_gpu() {
        return;
    }
    // A Kilosort 4 `spike_positions.npy` gives Features 3-D an x axis next to
    // depth: pick a ranked triple holding both and snapshot it.
    each_theme(|theme, name| {
        let root = dataset(&format!("xtalk-3d-positions-{name}"), 3);
        let clusters = npy::read_1d_u32(&root.join("spike_clusters.npy")).unwrap();
        // Units sit on a diagonal of the probe plane, with a deterministic jitter.
        let xy: Vec<f32> = clusters
            .iter()
            .enumerate()
            .flat_map(|(i, &u)| {
                let jitter = |k: usize| ((i * k) % 11) as f32 - 5.0;
                [30.0 * u as f32 + jitter(7), 60.0 * u as f32 + jitter(13)]
            })
            .collect();
        npy::write_f32_atomic(&root.join("spike_positions.npy"), &[clusters.len(), 2], &xy)
            .unwrap();
        let mut h = harness_sized(theme, [2200.0, 1100.0], move || {
            let (session, positions) = open(&root);
            let mut app = SorrelApp::new(session, label_str);
            configure(&mut app, &root, positions);
            app
        });
        settle(&mut h);
        select_two_and_open_features(&mut h);
        h.get_by_label("3D").click();
        settle(&mut h);
        h.get_by_label("feature 3d canvas");
        // Entering 3-D pins the focused plot's pair; unpin to list every triple.
        h.get_by_label("pin focused plot").click();
        settle(&mut h);
        let row = h
            .get_all_by_label_contains("axes ")
            .find(|n| {
                n.accesskit_node().label().is_some_and(|l| {
                    let dims: Vec<_> = l.trim_start_matches("axes ").split(" · ").collect();
                    dims.contains(&"depth") && dims.contains(&"x")
                })
            })
            .expect("a triple with depth and x");
        row.click_accesskit();
        settle(&mut h);
        snap(&mut h, "xtalk_3d_positions", name);
        h
    });
}
