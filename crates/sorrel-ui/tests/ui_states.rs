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
use sorrel_ui::{RasterPipeline, ScatterPipeline, SorrelApp, TracePipeline};
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
            // A host-wide driver filter (e.g. `*radeon*`) would hide lavapipe.
            std::env::remove_var("VK_LOADER_DRIVERS_SELECT");
            std::env::remove_var("VK_LOADER_DRIVERS_DISABLE");
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
            h.run_ok();
        }
        h.input_mut().modifiers = egui::Modifiers::NONE;
        // Past egui's double-click window so each press is its own click.
        settle(h);
    }
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
        h.key_press(egui::Key::Escape);
        settle(&mut h);
        h.get_by_label("k-means").click();
        settle(&mut h);
        snap(&mut h, "features_kmeans", name);
        h.key_press(egui::Key::K);
        settle(&mut h);
        let s = sorrel_ui::jobs::read(h.state().session()).history_len();
        assert_eq!(s, 1, "k splits the k-means lasso as one journal entry");
        h.get_by_label("auto-run").click();
        settle(&mut h);
        snap(&mut h, "features_kmeans_auto", name);
        h
    });
}
