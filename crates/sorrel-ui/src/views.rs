//! Per-cluster summary views for the central panel.
//!
//! Each view function consumes a `&Session<P>` and the active selection,
//! draws into the supplied `egui::Ui`, and stays free of side effects so
//! the caller decides when to repaint.

use egui::{Color32, Pos2, Sense, Stroke, Ui};
use sorrel_compute::{
    auto_correlogram, cross_correlogram, isi_histogram, sliding_refractory_contamination,
    QualityBreakdown,
};
use sorrel_data::{
    cluster_quality, preview_merge, rank_merge_candidates, rank_split_candidates, MergeCandidate,
    Session, SplitCandidate, SuggestConfig,
};
use sorrel_io::{ClusterId, DataProvider, SampleIndex};

/// Categorical palette used to colour clusters across views. Index modulo
/// length so we never run out — phy uses essentially the same trick.
const PALETTE: [Color32; 8] = [
    Color32::from_rgb(78, 121, 167),
    Color32::from_rgb(242, 142, 43),
    Color32::from_rgb(225, 87, 89),
    Color32::from_rgb(118, 183, 178),
    Color32::from_rgb(89, 161, 79),
    Color32::from_rgb(237, 201, 72),
    Color32::from_rgb(176, 122, 161),
    Color32::from_rgb(255, 157, 167),
];

#[inline]
fn cluster_colour(c: ClusterId) -> Color32 {
    PALETTE[c.idx() % PALETTE.len()]
}

/// Amplitude-vs-time scatter for every cluster in the selection. Only
/// renders something when the session has seeded amplitudes; otherwise
/// shows a hint that the backend / open path didn't expose them.
pub fn amplitude_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
) {
    let avail = ui.available_size_before_wrap();
    let height = avail.y.max(120.0).min(360.0);
    let (rect, _resp) =
        ui.allocate_exact_size(egui::vec2(avail.x.max(120.0), height), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, Color32::from_gray(18));

    if selection.is_empty() {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "no selection",
            egui::FontId::proportional(13.0),
            Color32::GRAY,
        );
        return;
    }

    // Determine global x (time in seconds) and y (amplitude) ranges from
    // the union of the selected clusters.
    let inv_sr = 1.0 / session.provider.sample_rate().max(1.0);
    let mut t_min = f32::INFINITY;
    let mut t_max = f32::NEG_INFINITY;
    let mut a_min = f32::INFINITY;
    let mut a_max = f32::NEG_INFINITY;
    let mut total_points = 0usize;
    for &c in selection {
        let times = session.spike_times(c);
        let amps = session.spike_amplitudes(c);
        if times.is_empty() || amps.is_empty() {
            continue;
        }
        total_points += times.len();
        if let (Some(&first), Some(&last)) = (times.first(), times.last()) {
            t_min = t_min.min(first.as_f32() * inv_sr);
            t_max = t_max.max(last.as_f32() * inv_sr);
        }
        for &a in amps {
            if a < a_min {
                a_min = a;
            }
            if a > a_max {
                a_max = a;
            }
        }
    }

    if total_points == 0 || !t_min.is_finite() || !a_min.is_finite() {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "amplitudes not seeded for this backend",
            egui::FontId::proportional(13.0),
            Color32::GRAY,
        );
        return;
    }
    let t_span = (t_max - t_min).max(1e-6);
    let a_span = (a_max - a_min).max(1e-6);

    // Sub-sample very large clusters so we don't push 100k+ painter ops.
    const POINT_BUDGET: usize = 20_000;
    let stride = ((total_points as f32 / POINT_BUDGET as f32).ceil() as usize).max(1);

    for &c in selection {
        let colour = cluster_colour(c);
        let times = session.spike_times(c);
        let amps = session.spike_amplitudes(c);
        if times.is_empty() || amps.is_empty() {
            continue;
        }
        let n = times.len().min(amps.len());
        for i in (0..n).step_by(stride) {
            let t = times[i].as_f32() * inv_sr;
            let a = amps[i];
            let x = rect.left() + (t - t_min) / t_span * rect.width();
            let y = rect.bottom() - (a - a_min) / a_span * rect.height();
            painter.circle_filled(egui::pos2(x, y), 1.5, colour);
        }
    }

    // Axis hints — minimal, just so the user can orient.
    painter.text(
        rect.left_top() + egui::vec2(4.0, 2.0),
        egui::Align2::LEFT_TOP,
        format!("amp [{a_min:.1}, {a_max:.1}]"),
        egui::FontId::proportional(11.0),
        Color32::GRAY,
    );
    painter.text(
        rect.right_bottom() + egui::vec2(-4.0, -2.0),
        egui::Align2::RIGHT_BOTTOM,
        format!("t [{t_min:.2}, {t_max:.2}] s"),
        egui::FontId::proportional(11.0),
        Color32::GRAY,
    );
}

/// Inter-spike-interval histogram for every cluster in the selection,
/// stacked vertically so individual cluster shapes stay legible.
pub fn isi_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
) {
    if selection.is_empty() {
        ui.label("no selection");
        return;
    }
    // Histogram window: 50 ms with 1 ms bins. Refractory zone = 1 ms.
    let sr = session.provider.sample_rate().max(1.0);
    let max_samples = (sr * 0.050).round() as u64;
    let bins = 50usize;
    let refractory_samples = (sr * 0.001).round() as u64;
    let refractory_bin =
        ((refractory_samples as f64) / (max_samples as f64) * bins as f64).round() as usize;

    for &c in selection {
        let times = session.spike_times(c);
        let h = isi_histogram(times, max_samples, bins);
        let max_count = h.iter().copied().max().unwrap_or(1).max(1) as f32;

        ui.horizontal(|ui| {
            let colour = cluster_colour(c);
            ui.colored_label(colour, format!("c{c}"));
            ui.label(format!("({} spikes)", times.len()));
        });

        let avail = ui.available_size_before_wrap();
        let (rect, _resp) =
            ui.allocate_exact_size(egui::vec2(avail.x.max(120.0), 56.0), Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::from_gray(18));

        // Shade the refractory zone so violations jump out.
        if refractory_bin > 0 && refractory_bin < bins {
            let x0 = rect.left();
            let x1 = rect.left() + rect.width() * refractory_bin as f32 / bins as f32;
            painter.rect_filled(
                egui::Rect::from_min_max(egui::pos2(x0, rect.top()), egui::pos2(x1, rect.bottom())),
                0.0,
                Color32::from_rgba_unmultiplied(220, 70, 70, 40),
            );
        }

        let bw = rect.width() / bins as f32;
        let colour = cluster_colour(c);
        for (i, &count) in h.iter().enumerate() {
            let h_norm = count as f32 / max_count;
            let bx0 = rect.left() + bw * i as f32;
            let bx1 = bx0 + bw * 0.95;
            let by0 = rect.bottom() - h_norm * rect.height();
            painter.rect_filled(
                egui::Rect::from_min_max(egui::pos2(bx0, by0), egui::pos2(bx1, rect.bottom())),
                0.0,
                colour,
            );
        }
        // Bottom axis line.
        painter.line_segment(
            [rect.left_bottom(), rect.right_bottom()],
            Stroke::new(0.5_f32, Color32::from_gray(60)),
        );
        ui.label(format!(
            "ISI 0–{:.0} ms · 1 ms bins · refractory shaded",
            (max_samples as f32 / sr) * 1000.0,
        ));
    }
}

/// Short summary block — title, spike counts, label string, palette swatch.
pub fn selection_summary<P, F>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
    label_str: F,
) where
    P: DataProvider,
    F: Fn(&P::Label) -> &'static str,
{
    match selection.len() {
        0 => {
            ui.heading("No selection");
        }
        1 => {
            let c = selection[0];
            ui.horizontal(|ui| {
                ui.colored_label(cluster_colour(c), "■");
                ui.heading(format!("Cluster {c}"));
            });
            let n_spikes = session.spike_times(c).len();
            ui.label(format!("{n_spikes} spikes"));
            ui.label(format!(
                "label: {}",
                label_str(&session.label(c).unwrap_or_default())
            ));
        }
        n => {
            ui.heading(format!("{n} clusters selected"));
            let total: usize = selection
                .iter()
                .map(|&c| session.spike_times(c).len())
                .sum();
            ui.label(format!("{total} spikes (sum)"));
            ui.horizontal_wrapped(|ui| {
                for &c in selection {
                    ui.colored_label(cluster_colour(c), format!("c{c}"));
                }
            });
        }
    }
}

// ---- Waveform extraction & cache ---------------------------------------

/// Window-around-spike used by `WaveformCache`. Picks ±1 ms by default.
fn waveform_window_samples(sr: f32) -> u32 {
    (sr.max(1.0) * 0.001).round() as u32
}

/// Cached snippet extraction keyed on `(selection, pre, post)`. Spike
/// extraction walks the trace mmap N times per spike, which is fast in the
/// steady state thanks to the page cache, but recomputing it every paint
/// stalls the egui thread on cluster switches.
#[derive(Debug, Default)]
pub struct WaveformCache {
    key: WaveformKey,
    /// `[cluster][spike][sample_in_snippet * n_channels + channel]`
    snippets: Vec<Vec<Vec<f32>>>,
    /// `[cluster][channel][sample_in_snippet]`
    means: Vec<Vec<Vec<f32>>>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct WaveformKey {
    clusters: Vec<ClusterId>,
    pre: u32,
    post: u32,
    n_channels: u32,
}

impl WaveformCache {
    pub fn new() -> Self {
        Self::default()
    }

    fn ensure<P: DataProvider>(
        &mut self,
        session: &Session<P>,
        selection: &[ClusterId],
        pre: u32,
        post: u32,
    ) {
        let key = WaveformKey {
            clusters: selection.to_vec(),
            pre,
            post,
            n_channels: session.provider.n_channels(),
        };
        if key == self.key {
            return;
        }
        let snippet_len = (pre + post + 1) as usize;
        let nc = key.n_channels as usize;

        const MAX_SPIKES_PER_CLUSTER: usize = 100;

        let mut snippets: Vec<Vec<Vec<f32>>> = Vec::with_capacity(selection.len());
        for &c in selection {
            let times = session.spike_times(c);
            if times.is_empty() || nc == 0 || snippet_len < 3 {
                snippets.push(Vec::new());
                continue;
            }
            let stride = (times.len() / MAX_SPIKES_PER_CLUSTER).max(1);
            let mut bucket = Vec::new();
            for &t in times.iter().step_by(stride).take(MAX_SPIKES_PER_CLUSTER) {
                let lo = t.0.saturating_sub(pre as u64);
                let slice = session.provider.trace(SampleIndex(lo), snippet_len as u32);
                if slice.samples.len() != snippet_len * nc {
                    continue;
                }
                let buf: Vec<f32> = slice.samples.to_f32_centred();
                bucket.push(buf);
            }
            snippets.push(bucket);
        }

        // Per-cluster mean waveform per channel.
        let mut means: Vec<Vec<Vec<f32>>> = Vec::with_capacity(selection.len());
        for cluster_snips in &snippets {
            if cluster_snips.is_empty() || nc == 0 {
                means.push(vec![vec![0.0; snippet_len]; nc]);
                continue;
            }
            let mut sum = vec![vec![0.0f64; snippet_len]; nc];
            let count = cluster_snips.len() as f64;
            for snip in cluster_snips {
                for t in 0..snippet_len {
                    let base = t * nc;
                    for ch in 0..nc {
                        sum[ch][t] += snip[base + ch] as f64;
                    }
                }
            }
            let cluster_means: Vec<Vec<f32>> = sum
                .into_iter()
                .map(|row| row.into_iter().map(|v| (v / count) as f32).collect())
                .collect();
            means.push(cluster_means);
        }

        self.key = key;
        self.snippets = snippets;
        self.means = means;
    }
}

/// Per-channel peak amplitude across all clusters' means. Used to rank
/// channels for the multi-channel layout.
fn channel_peak_abs(means: &[Vec<Vec<f32>>], n_channels: usize) -> Vec<f32> {
    let mut peaks = vec![0.0f32; n_channels];
    for cluster_means in means {
        for (ch, mean) in cluster_means.iter().enumerate() {
            if ch >= peaks.len() {
                continue;
            }
            for &v in mean {
                let a = v.abs();
                if a > peaks[ch] {
                    peaks[ch] = a;
                }
            }
        }
    }
    peaks
}

/// Multi-cluster waveform overlay. Auto-picks the *peak channel* (single-
/// channel layout) when no probe geometry is available; renders the top
/// channels by amplitude in their `(x, y)` probe positions when geometry
/// is supplied. Caches the extracted snippets so cluster-switching stalls
/// happen at most once per change.
pub fn waveform_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
    geometry: &[[f32; 2]],
    cache: &mut WaveformCache,
) {
    if selection.is_empty() {
        ui.label("no selection");
        return;
    }

    let sr = session.provider.sample_rate().max(1.0);
    let pre = waveform_window_samples(sr);
    let post = pre;
    let snippet_len = (pre + post + 1) as usize;
    let nc = session.provider.n_channels() as usize;
    if nc == 0 || snippet_len < 3 {
        ui.label("no channels");
        return;
    }

    cache.ensure(session, selection, pre, post);
    let peaks = channel_peak_abs(&cache.means, nc);

    // Decide layout.
    let geometry_available = geometry.len() == nc;
    if geometry_available {
        draw_probe_layout(ui, selection, cache, geometry, &peaks, snippet_len, nc, sr);
    } else {
        draw_single_channel(ui, selection, cache, &peaks, snippet_len, nc, sr, pre);
    }
}

fn draw_single_channel(
    ui: &mut Ui,
    selection: &[ClusterId],
    cache: &WaveformCache,
    peaks: &[f32],
    snippet_len: usize,
    nc: usize,
    sr: f32,
    pre: u32,
) {
    // Pick the global peak channel.
    let (peak_ch, peak_val) = peaks
        .iter()
        .enumerate()
        .fold((0usize, 0.0f32), |(bi, bv), (i, &v)| {
            if v > bv {
                (i, v)
            } else {
                (bi, bv)
            }
        });
    if peak_val <= 0.0 {
        ui.label("no waveforms in range — try expanding the selection");
        return;
    }

    let mut y_max = 1e-6f32;
    for cluster_means in &cache.means {
        for &v in &cluster_means[peak_ch] {
            y_max = y_max.max(v.abs());
        }
    }
    for cluster_snips in &cache.snippets {
        for snip in cluster_snips {
            for t in 0..snippet_len {
                let v = snip[t * nc + peak_ch].abs();
                if v > y_max {
                    y_max = v;
                }
            }
        }
    }

    let avail = ui.available_size_before_wrap();
    let height = avail.y.max(160.0).min(420.0);
    let (rect, _resp) =
        ui.allocate_exact_size(egui::vec2(avail.x.max(120.0), height), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, Color32::from_gray(18));
    let cy = rect.center().y;
    painter.line_segment(
        [Pos2::new(rect.left(), cy), Pos2::new(rect.right(), cy)],
        Stroke::new(0.5_f32, Color32::from_gray(50)),
    );

    let x_step = rect.width() / (snippet_len as f32 - 1.0).max(1.0);
    let y_scale = (rect.height() * 0.45) / y_max;

    for (ci, &c) in selection.iter().enumerate() {
        let colour = cluster_colour(c);
        let faint = Color32::from_rgba_unmultiplied(colour.r(), colour.g(), colour.b(), 30);
        let stroke_individual = Stroke::new(0.5_f32, faint);
        let stroke_mean = Stroke::new(2.0_f32, colour);

        for snip in &cache.snippets[ci] {
            let mut prev: Option<Pos2> = None;
            for t in 0..snippet_len {
                let v = snip[t * nc + peak_ch];
                let p = Pos2::new(rect.left() + t as f32 * x_step, cy - v * y_scale);
                if let Some(pp) = prev {
                    painter.line_segment([pp, p], stroke_individual);
                }
                prev = Some(p);
            }
        }
        let mean = &cache.means[ci][peak_ch];
        let mut prev: Option<Pos2> = None;
        for (t, &v) in mean.iter().enumerate() {
            let p = Pos2::new(rect.left() + t as f32 * x_step, cy - v * y_scale);
            if let Some(pp) = prev {
                painter.line_segment([pp, p], stroke_mean);
            }
            prev = Some(p);
        }
    }

    painter.text(
        rect.left_top() + egui::vec2(6.0, 4.0),
        egui::Align2::LEFT_TOP,
        format!(
            "channel {peak_ch} · ±{} samples (~±{:.1} ms) · no probe geometry",
            pre,
            (pre as f32) / sr * 1000.0,
        ),
        egui::FontId::proportional(11.0),
        Color32::GRAY,
    );
}

fn draw_probe_layout(
    ui: &mut Ui,
    selection: &[ClusterId],
    cache: &WaveformCache,
    geometry: &[[f32; 2]],
    peaks: &[f32],
    snippet_len: usize,
    nc: usize,
    sr: f32,
) {
    // Pick top N channels by peak amplitude.
    const N_BEST: usize = 16;
    let mut idx: Vec<usize> = (0..nc).collect();
    idx.sort_unstable_by(|&a, &b| peaks[b].partial_cmp(&peaks[a]).unwrap_or(std::cmp::Ordering::Equal));
    let best: Vec<usize> = idx.into_iter().take(N_BEST).collect();
    if best.is_empty() || peaks[best[0]] <= 0.0 {
        ui.label("no waveforms in range — try expanding the selection");
        return;
    }

    // Bounding box of the selected channel positions.
    let mut x_min = f32::INFINITY;
    let mut x_max = f32::NEG_INFINITY;
    let mut y_min = f32::INFINITY;
    let mut y_max_geom = f32::NEG_INFINITY;
    for &ch in &best {
        let p = geometry[ch];
        x_min = x_min.min(p[0]);
        x_max = x_max.max(p[0]);
        y_min = y_min.min(p[1]);
        y_max_geom = y_max_geom.max(p[1]);
    }
    let x_span = (x_max - x_min).max(1.0);
    let y_span = (y_max_geom - y_min).max(1.0);

    // Symmetric amplitude range over the picked channels.
    let mut y_amp = 1e-6f32;
    for &ch in &best {
        for cluster_means in &cache.means {
            for &v in &cluster_means[ch] {
                y_amp = y_amp.max(v.abs());
            }
        }
    }

    let avail = ui.available_size_before_wrap();
    let height = avail.y.max(220.0).min(640.0);
    let (rect, _resp) =
        ui.allocate_exact_size(egui::vec2(avail.x.max(120.0), height), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, Color32::from_gray(18));

    // Per-subplot box size: divide the panel into a grid sized to the
    // probe's aspect ratio while keeping each subplot at least 60×40px.
    let panel_w = rect.width() - 24.0;
    let panel_h = rect.height() - 24.0;
    // Sub-rect width/height proportional to the spacing between picked
    // channels — roughly half the smallest x-gap and y-gap.
    let mut min_dx = x_span;
    let mut min_dy = y_span;
    for i in 0..best.len() {
        for j in (i + 1)..best.len() {
            let pi = geometry[best[i]];
            let pj = geometry[best[j]];
            let dx = (pi[0] - pj[0]).abs();
            let dy = (pi[1] - pj[1]).abs();
            if dx > 0.5 && dx < min_dx {
                min_dx = dx;
            }
            if dy > 0.5 && dy < min_dy {
                min_dy = dy;
            }
        }
    }
    let sub_w = ((min_dx / x_span) * panel_w).clamp(60.0, panel_w * 0.45);
    let sub_h = ((min_dy / y_span) * panel_h).clamp(40.0, panel_h * 0.45);

    let scale_x = (panel_w - sub_w).max(1.0) / x_span;
    let scale_y = (panel_h - sub_h).max(1.0) / y_span;

    for &ch in &best {
        let p = geometry[ch];
        // Map probe coords to panel coords. y is flipped (probes have y
        // increasing upward; egui has y increasing downward).
        let cx = rect.left() + 12.0 + (p[0] - x_min) * scale_x + sub_w * 0.5;
        let cy = rect.bottom() - 12.0 - (p[1] - y_min) * scale_y - sub_h * 0.5;
        let sub = egui::Rect::from_center_size(Pos2::new(cx, cy), egui::vec2(sub_w, sub_h));
        painter.rect_filled(sub, 0.0, Color32::from_gray(24));

        let baseline_y = sub.center().y;
        let x_step = sub.width() / (snippet_len as f32 - 1.0).max(1.0);
        let y_scale = (sub.height() * 0.45) / y_amp;

        for (ci, &c) in selection.iter().enumerate() {
            let colour = cluster_colour(c);
            let faint =
                Color32::from_rgba_unmultiplied(colour.r(), colour.g(), colour.b(), 30);
            let stroke_individual = Stroke::new(0.4_f32, faint);
            let stroke_mean = Stroke::new(1.5_f32, colour);

            for snip in &cache.snippets[ci] {
                let mut prev: Option<Pos2> = None;
                for t in 0..snippet_len {
                    let v = snip[t * nc + ch];
                    let pt = Pos2::new(sub.left() + t as f32 * x_step, baseline_y - v * y_scale);
                    if let Some(pp) = prev {
                        painter.line_segment([pp, pt], stroke_individual);
                    }
                    prev = Some(pt);
                }
            }
            let mean = &cache.means[ci][ch];
            let mut prev: Option<Pos2> = None;
            for (t, &v) in mean.iter().enumerate() {
                let pt = Pos2::new(sub.left() + t as f32 * x_step, baseline_y - v * y_scale);
                if let Some(pp) = prev {
                    painter.line_segment([pp, pt], stroke_mean);
                }
                prev = Some(pt);
            }
        }

        // Channel label, top-left of the subplot.
        painter.text(
            sub.left_top() + egui::vec2(2.0, 1.0),
            egui::Align2::LEFT_TOP,
            format!("ch{ch}"),
            egui::FontId::proportional(10.0),
            Color32::from_gray(140),
        );
    }

    painter.text(
        rect.left_top() + egui::vec2(6.0, 4.0),
        egui::Align2::LEFT_TOP,
        format!(
            "{} best channels · ~±{:.1} ms · probe layout",
            best.len(),
            (snippet_len as f32 / 2.0) / sr * 1000.0,
        ),
        egui::FontId::proportional(11.0),
        Color32::GRAY,
    );
}

/// Pairwise correlogram grid for the selected clusters.
///
/// The diagonal renders auto-correlograms (one per cluster); off-diagonal
/// cells render cross-correlograms between row-cluster and column-cluster.
/// Both spike trains are taken from `Session::spike_times` so the histogram
/// reflects the live (post-curation) bucket assignments.
pub fn correlogram_grid_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
) {
    if selection.is_empty() {
        ui.label("no selection");
        return;
    }
    let sr = session.provider.sample_rate().max(1.0);
    // Window: ±50 ms. 1 ms bins -> 100 bins centred on 0.
    let max_lag_samples = (sr * 0.050).round() as u64;
    let bins = 100usize;
    if max_lag_samples == 0 {
        ui.label("sample rate not set — can't compute correlogram");
        return;
    }
    let refractory_bin = ((sr * 0.001) / max_lag_samples as f32 * bins as f32 * 0.5).round()
        .max(0.0) as usize;

    let n = selection.len();
    let avail = ui.available_size_before_wrap();
    let header_h = 14.0;
    let panel_h = (avail.y - 8.0).max(160.0);
    let cell_w = (avail.x - 4.0) / n as f32;
    let cell_h = ((panel_h - 4.0) / n as f32).clamp(40.0, 96.0);

    let total_h = cell_h * n as f32 + 8.0 + header_h;
    let (rect, _resp) = ui.allocate_exact_size(
        egui::vec2(avail.x.max(120.0), total_h),
        Sense::hover(),
    );
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, Color32::from_gray(18));

    // Caption.
    painter.text(
        rect.left_top() + egui::vec2(6.0, 2.0),
        egui::Align2::LEFT_TOP,
        format!(
            "ACG diag · CCG off-diag · ±{:.0} ms · 1 ms bins · refractory shaded",
            max_lag_samples as f32 / sr * 1000.0,
        ),
        egui::FontId::proportional(11.0),
        Color32::GRAY,
    );

    let grid_top = rect.top() + header_h + 4.0;
    for (ri, &c_row) in selection.iter().enumerate() {
        let ta = session.spike_times(c_row);
        for (ci, &c_col) in selection.iter().enumerate() {
            let tb = session.spike_times(c_col);
            let cell = egui::Rect::from_min_size(
                egui::pos2(rect.left() + 2.0 + ci as f32 * cell_w, grid_top + ri as f32 * cell_h),
                egui::vec2(cell_w - 2.0, cell_h - 2.0),
            );
            painter.rect_filled(cell, 0.0, Color32::from_gray(24));

            let h = if ri == ci {
                auto_correlogram(ta, max_lag_samples, bins)
            } else {
                cross_correlogram(ta, tb, max_lag_samples, bins)
            };
            let max_count = h.iter().copied().max().unwrap_or(1).max(1) as f32;

            // Centre line.
            let cx = cell.center().x;
            painter.line_segment(
                [egui::pos2(cx, cell.top()), egui::pos2(cx, cell.bottom())],
                Stroke::new(0.5_f32, Color32::from_gray(50)),
            );
            // Refractory shading: ±refractory_bin around the centre.
            if refractory_bin > 0 && refractory_bin * 2 < bins {
                let half = bins / 2;
                let lo_bin = half.saturating_sub(refractory_bin);
                let hi_bin = (half + refractory_bin).min(bins);
                let xl = cell.left() + cell.width() * lo_bin as f32 / bins as f32;
                let xr = cell.left() + cell.width() * hi_bin as f32 / bins as f32;
                painter.rect_filled(
                    egui::Rect::from_min_max(
                        egui::pos2(xl, cell.top()),
                        egui::pos2(xr, cell.bottom()),
                    ),
                    0.0,
                    Color32::from_rgba_unmultiplied(220, 70, 70, 30),
                );
            }

            let bw = cell.width() / bins as f32;
            let colour = if ri == ci {
                cluster_colour(c_row)
            } else {
                // Average the two row/col colours for clarity in cross cells.
                let a = cluster_colour(c_row);
                let b = cluster_colour(c_col);
                Color32::from_rgb(
                    ((a.r() as u16 + b.r() as u16) / 2) as u8,
                    ((a.g() as u16 + b.g() as u16) / 2) as u8,
                    ((a.b() as u16 + b.b() as u16) / 2) as u8,
                )
            };
            for (b_idx, &count) in h.iter().enumerate() {
                let h_norm = count as f32 / max_count;
                let bx0 = cell.left() + bw * b_idx as f32;
                let bx1 = bx0 + bw * 0.95;
                let by0 = cell.bottom() - h_norm * (cell.height() - 4.0);
                painter.rect_filled(
                    egui::Rect::from_min_max(
                        egui::pos2(bx0, by0),
                        egui::pos2(bx1, cell.bottom()),
                    ),
                    0.0,
                    colour,
                );
            }

            // Cell label, top-left.
            painter.text(
                cell.left_top() + egui::vec2(2.0, 1.0),
                egui::Align2::LEFT_TOP,
                format!("c{c_row}×c{c_col}"),
                egui::FontId::proportional(10.0),
                Color32::from_gray(140),
            );
        }
    }
}

// ---- FeatureView (PC scatter + lasso split) ----------------------------

/// Persistent state for the FeatureView. Lives on `SorrelApp` so the lasso
/// rectangle and axis selectors survive re-paints.
#[derive(Clone, Debug)]
pub struct FeatureViewState {
    pub pc_x: u8,
    pub pc_y: u8,
    pub channel_idx: u8,
    /// While the user is dragging a lasso rectangle, this is the start
    /// position in screen coords. `None` when not actively lassoing.
    drag_start: Option<Pos2>,
}

impl Default for FeatureViewState {
    fn default() -> Self {
        Self {
            pc_x: 0,
            pc_y: 1,
            channel_idx: 0,
            drag_start: None,
        }
    }
}

/// PC-feature scatter for the selected clusters with a rectangular lasso
/// for splitting. Returns `Some(splits)` when the user finishes a lasso —
/// each entry is `(source_cluster, local_spike_indices_to_move)` ready for
/// `CurationCommand::Split`.
pub fn feature_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
    state: &mut FeatureViewState,
) -> Option<Vec<(ClusterId, Vec<u32>)>> {
    let (n_pcs, n_chans) = session.pc_shape();
    if n_pcs == 0 || n_chans == 0 {
        ui.label(
            "PC features not available — open a directory containing \
             pc_features.npy and pc_feature_ind.npy",
        );
        return None;
    }
    if selection.is_empty() {
        ui.label("no selection");
        return None;
    }

    // Axis pickers. Bounds: PC < n_pcs, channel < n_chans.
    ui.horizontal(|ui| {
        ui.label("x:");
        ui.add(
            egui::DragValue::new(&mut state.pc_x)
                .range(0..=(n_pcs as u8 - 1))
                .prefix("PC")
                .speed(0.1),
        );
        ui.label("y:");
        ui.add(
            egui::DragValue::new(&mut state.pc_y)
                .range(0..=(n_pcs as u8 - 1))
                .prefix("PC")
                .speed(0.1),
        );
        ui.label("ch:");
        ui.add(
            egui::DragValue::new(&mut state.channel_idx)
                .range(0..=(n_chans as u8 - 1))
                .speed(0.1),
        );
        ui.weak("(template-relative channel index)");
    });

    let avail = ui.available_size_before_wrap();
    let height = avail.y.max(220.0).min(540.0);
    let (rect, resp) = ui.allocate_exact_size(
        egui::vec2(avail.x.max(160.0), height),
        Sense::click_and_drag(),
    );
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, Color32::from_gray(18));

    // First pass: gather every selected spike's (x, y, cluster, global_idx).
    // Walk spike_clusters_global once (O(n_spikes)) and pick membership.
    let pc_x = state.pc_x as usize;
    let pc_y = state.pc_y as usize;
    let ch = state.channel_idx as usize;
    let stride = n_pcs * n_chans;
    let offset_x = pc_x * n_chans + ch;
    let offset_y = pc_y * n_chans + ch;

    let in_sel: std::collections::HashSet<ClusterId> = selection.iter().copied().collect();
    let spike_clusters = session.spike_clusters_global();

    let mut points: Vec<(f32, f32, ClusterId, u32)> = Vec::new();
    for (g, &c) in spike_clusters.iter().enumerate() {
        if !in_sel.contains(&c) {
            continue;
        }
        let Some(feats) = session.pc_feature_for(g as u32) else {
            continue;
        };
        if feats.len() < stride {
            continue;
        }
        let x = feats[offset_x];
        let y = feats[offset_y];
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        points.push((x, y, c, g as u32));
    }

    if points.is_empty() {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "no PC features in range",
            egui::FontId::proportional(13.0),
            Color32::GRAY,
        );
        return None;
    }

    // Auto-scale: 1st/99th percentile per axis to clip outliers without
    // hiding the bulk of the cloud.
    let (x_lo, x_hi) = percentile_range(&points, |p| p.0);
    let (y_lo, y_hi) = percentile_range(&points, |p| p.1);
    let x_span = (x_hi - x_lo).max(1e-6);
    let y_span = (y_hi - y_lo).max(1e-6);

    // Sub-sample for painter cost — same budget as AmplitudeView.
    const POINT_BUDGET: usize = 20_000;
    let stride_pt = ((points.len() as f32 / POINT_BUDGET as f32).ceil() as usize).max(1);

    let to_screen = |x: f32, y: f32| {
        let sx = rect.left() + ((x - x_lo) / x_span).clamp(0.0, 1.0) * rect.width();
        // Flip y so larger PC values go up.
        let sy =
            rect.bottom() - ((y - y_lo) / y_span).clamp(0.0, 1.0) * rect.height();
        Pos2::new(sx, sy)
    };

    for (i, (x, y, c, _g)) in points.iter().enumerate() {
        if i % stride_pt != 0 {
            continue;
        }
        let p = to_screen(*x, *y);
        painter.circle_filled(p, 1.5, cluster_colour(*c));
    }

    // Lasso interaction. Drag begin/continue/end is fully covered by
    // egui's dragged()/drag_started()/drag_released() — but we also need to
    // know the *start* pointer position, which egui doesn't keep, so we
    // stash it in `state.drag_start`.
    let mut splits: Option<Vec<(ClusterId, Vec<u32>)>> = None;
    if resp.drag_started() {
        if let Some(p) = resp.interact_pointer_pos() {
            state.drag_start = Some(p);
        }
    }
    if let Some(start) = state.drag_start {
        if let Some(now) = resp.interact_pointer_pos() {
            let lasso = egui::Rect::from_two_pos(start, now);
            painter.rect_stroke(
                lasso,
                0.0,
                Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(255, 255, 255, 200)),
            );
        }
    }
    if resp.drag_stopped() {
        if let (Some(start), Some(end)) = (state.drag_start.take(), resp.interact_pointer_pos()) {
            let lasso = egui::Rect::from_two_pos(start, end);
            // Collect hits = (cluster, global_idx). Skip degenerate lassos.
            if lasso.width() >= 4.0 && lasso.height() >= 4.0 {
                let mut hits: Vec<(ClusterId, u32)> = Vec::new();
                for (x, y, c, g) in &points {
                    let p = to_screen(*x, *y);
                    if lasso.contains(p) {
                        hits.push((*c, *g));
                    }
                }
                if !hits.is_empty() {
                    splits = Some(hits_to_local_splits(spike_clusters, &hits));
                }
            }
        }
    }

    // Caption.
    painter.text(
        rect.left_top() + egui::vec2(6.0, 4.0),
        egui::Align2::LEFT_TOP,
        format!(
            "PC{}×PC{} on ch-idx {} · drag to lasso → split · {} spikes",
            pc_x, pc_y, ch, points.len()
        ),
        egui::FontId::proportional(11.0),
        Color32::GRAY,
    );

    splits
}

/// Convert lasso hits (`Vec<(cluster, global_idx)>`) into the
/// `Vec<(cluster, Vec<local_idx>)>` shape `CurationCommand::Split` expects.
///
/// Walks `spike_clusters` once and assigns each spike its local-within-cluster
/// position; picks out the hits along the way. O(n_spikes + n_hits).
fn hits_to_local_splits(
    spike_clusters: &[ClusterId],
    hits: &[(ClusterId, u32)],
) -> Vec<(ClusterId, Vec<u32>)> {
    use std::collections::{HashMap, HashSet};
    let mut hit_globals: HashMap<ClusterId, HashSet<u32>> = HashMap::new();
    for (c, g) in hits {
        hit_globals.entry(*c).or_default().insert(*g);
    }
    let mut local_idx_for: HashMap<ClusterId, Vec<u32>> = HashMap::new();
    let mut counters: HashMap<ClusterId, u32> = HashMap::new();
    for (g, &c) in spike_clusters.iter().enumerate() {
        let local = *counters.entry(c).or_insert(0);
        if let Some(gset) = hit_globals.get(&c) {
            if gset.contains(&(g as u32)) {
                local_idx_for.entry(c).or_default().push(local);
            }
        }
        *counters.get_mut(&c).unwrap() += 1;
    }
    local_idx_for.into_iter().collect()
}

fn percentile_range<T, F>(points: &[T], extract: F) -> (f32, f32)
where
    F: Fn(&T) -> f32,
{
    if points.is_empty() {
        return (0.0, 1.0);
    }
    let mut vals: Vec<f32> = points.iter().map(&extract).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = vals.len();
    let lo_idx = (n as f32 * 0.01) as usize;
    let hi_idx = ((n as f32 * 0.99) as usize).min(n - 1);
    (vals[lo_idx], vals[hi_idx])
}

// ---- RasterView (global spike scatter) ---------------------------------

use crate::wgpu_raster::{pack_rgba, RasterCallback, RasterVertex};

/// Cached raster vertex buffer. Invalidated by curation history changes
/// (merge/split/relabel — anything that mutates the cluster_index).
#[derive(Debug, Default)]
pub struct RasterCache {
    /// Packed `(time_seconds, cluster_row, rgba)` vertices.
    vertices: Vec<RasterVertex>,
    /// Map `cluster_id → row_index_in_raster`. Empty when not built.
    rows: Vec<i32>,
    /// Number of populated rows (clusters with at least one spike).
    n_rows: u32,
    /// Earliest spike time in seconds.
    t0: f32,
    /// Latest spike time in seconds.
    t1: f32,
    /// `(history_len, redo_len, n_clusters)` snapshot at last build —
    /// changes invalidate the cache.
    last_key: (usize, usize, u32),
}

impl RasterCache {
    pub fn new() -> Self {
        Self::default()
    }

    fn ensure<P: DataProvider>(&mut self, session: &Session<P>) {
        let key = (
            session.history_len(),
            session.redo_len(),
            session.n_clusters(),
        );
        if key == self.last_key && !self.vertices.is_empty() {
            return;
        }

        let n_clusters = session.n_clusters();
        let inv_sr = 1.0_f32 / session.provider.sample_rate().max(1.0);

        // Assign row indices only to non-empty clusters so the y-axis isn't
        // dominated by holes left behind after merges.
        let mut rows = vec![-1i32; n_clusters as usize];
        let mut n_rows = 0u32;
        for c in (0..n_clusters).map(ClusterId) {
            if !session.spike_times(c).is_empty() {
                rows[c.idx()] = n_rows as i32;
                n_rows += 1;
            }
        }

        let total: usize = (0..n_clusters)
            .map(ClusterId)
            .map(|c| session.spike_times(c).len())
            .sum();

        let mut vertices = Vec::with_capacity(total);
        let mut t_min = f32::INFINITY;
        let mut t_max = f32::NEG_INFINITY;
        for c in (0..n_clusters).map(ClusterId) {
            let row = rows[c.idx()];
            if row < 0 {
                continue;
            }
            let colour = cluster_colour(c);
            let packed = pack_rgba(colour.r(), colour.g(), colour.b(), 220);
            for &t in session.spike_times(c) {
                let ts = t.as_f32() * inv_sr;
                if ts < t_min {
                    t_min = ts;
                }
                if ts > t_max {
                    t_max = ts;
                }
                vertices.push(RasterVertex {
                    pos: [ts, row as f32],
                    color: packed,
                });
            }
        }

        if !t_min.is_finite() {
            t_min = 0.0;
            t_max = 1.0;
        }

        self.vertices = vertices;
        self.rows = rows;
        self.n_rows = n_rows.max(1);
        self.t0 = t_min;
        self.t1 = t_max;
        self.last_key = key;
    }

    pub fn invalidate(&mut self) {
        self.last_key = (usize::MAX, usize::MAX, u32::MAX);
        self.vertices.clear();
    }
}

/// Global raster: every spike of every non-empty cluster, time on x, cluster
/// row on y. Click to seek the trace window to that time.
///
/// Returns the time (in samples) the user clicked, if any.
pub fn raster_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    cache: &mut RasterCache,
) -> Option<u64> {
    cache.ensure(session);

    if cache.vertices.is_empty() {
        ui.label("no spikes");
        return None;
    }

    let avail = ui.available_size_before_wrap();
    let height = avail.y.max(160.0).min(640.0);
    let (rect, resp) = ui.allocate_exact_size(
        egui::vec2(avail.x.max(120.0), height),
        Sense::click(),
    );
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, Color32::from_gray(16));

    let t_span = (cache.t1 - cache.t0).max(1e-6);
    let n_vertices = cache.vertices.len() as u32;

    let callback = RasterCallback {
        vertices: cache.vertices.clone(),
        n_vertices,
        t0: cache.t0,
        t_span,
        n_rows: cache.n_rows,
    };
    ui.painter()
        .add(eframe::egui_wgpu::Callback::new_paint_callback(rect, callback));

    // Caption.
    painter.text(
        rect.left_top() + egui::vec2(6.0, 4.0),
        egui::Align2::LEFT_TOP,
        format!(
            "{} spikes · {} clusters · {:.1}–{:.1} s · click to seek trace",
            n_vertices, cache.n_rows, cache.t0, cache.t1
        ),
        egui::FontId::proportional(11.0),
        Color32::from_gray(180),
    );

    // Click → seek. Map x in rect to time, return as samples.
    let mut seek_to = None;
    if resp.clicked() {
        if let Some(p) = resp.interact_pointer_pos() {
            let frac = ((p.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
            let t_sec = cache.t0 + frac * t_span;
            let sample = (t_sec * session.provider.sample_rate()).max(0.0) as u64;
            seek_to = Some(sample);
        }
    }
    seek_to
}

// ---- FiringRateView -----------------------------------------------------

/// Per-cluster firing rate over the recording, plotted as overlaid line
/// graphs colour-coded by cluster. Useful for spotting drift, electrode
/// failure, and post-stimulus modulation — all phy parity.
pub fn firing_rate_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
) {
    if selection.is_empty() {
        ui.label("no selection");
        return;
    }
    let sr = session.provider.sample_rate().max(1.0);
    let total_samples = session.provider.n_samples().0;
    if total_samples == 0 {
        ui.label("recording has zero length");
        return;
    }
    let total_s = total_samples as f32 / sr;

    // Bin width: target ~100 bins, but at least 1 s per bin so very short
    // recordings still produce stable rates.
    const TARGET_BINS: usize = 100;
    let mut bins = TARGET_BINS;
    let mut bin_s = total_s / bins as f32;
    if bin_s < 1.0 {
        bin_s = 1.0;
        bins = (total_s.ceil() as usize).max(1);
    }
    let bin_samples = (bin_s * sr) as u64;
    if bin_samples == 0 {
        ui.label("bin width collapsed to 0 — recording too short");
        return;
    }

    // Bin every selected cluster's spike train into the same axis.
    let mut histograms: Vec<Vec<u32>> = Vec::with_capacity(selection.len());
    let mut max_count = 1u32;
    for &c in selection {
        let times = session.spike_times(c);
        let mut h = vec![0u32; bins];
        for &t in times {
            let idx = (t.0 / bin_samples) as usize;
            if idx < bins {
                h[idx] += 1;
            }
        }
        if let Some(&m) = h.iter().max() {
            if m > max_count {
                max_count = m;
            }
        }
        histograms.push(h);
    }

    let avail = ui.available_size_before_wrap();
    let height = avail.y.max(160.0).min(360.0);
    let (rect, _resp) =
        ui.allocate_exact_size(egui::vec2(avail.x.max(160.0), height), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, Color32::from_gray(18));

    // Convert max_count to Hz scale for the y-axis label.
    let max_rate_hz = max_count as f32 / bin_s;

    // Time gridlines every ~1/4 of the panel (4 segments).
    for k in 1..4 {
        let x = rect.left() + rect.width() * k as f32 / 4.0;
        painter.line_segment(
            [
                egui::pos2(x, rect.top()),
                egui::pos2(x, rect.bottom()),
            ],
            Stroke::new(0.3_f32, Color32::from_gray(34)),
        );
    }

    // Plot each cluster's rate as a line.
    let bin_w = rect.width() / bins as f32;
    let y_scale = (rect.height() - 14.0) / max_count.max(1) as f32;
    for (ci, &c) in selection.iter().enumerate() {
        let colour = cluster_colour(c);
        let stroke = Stroke::new(1.5_f32, colour);
        let mut prev: Option<egui::Pos2> = None;
        for (b, &count) in histograms[ci].iter().enumerate() {
            let x = rect.left() + bin_w * (b as f32 + 0.5);
            let y = rect.bottom() - count as f32 * y_scale - 2.0;
            let p = egui::pos2(x, y);
            if let Some(pp) = prev {
                painter.line_segment([pp, p], stroke);
            }
            prev = Some(p);
        }
    }

    // Captions.
    painter.text(
        rect.left_top() + egui::vec2(6.0, 2.0),
        egui::Align2::LEFT_TOP,
        format!(
            "rate · {:.1} Hz peak · {} bins × {:.1} s",
            max_rate_hz, bins, bin_s,
        ),
        egui::FontId::proportional(11.0),
        Color32::GRAY,
    );
    painter.text(
        rect.right_bottom() + egui::vec2(-6.0, -2.0),
        egui::Align2::RIGHT_BOTTOM,
        format!("0 — {:.1} s", total_s),
        egui::FontId::proportional(11.0),
        Color32::GRAY,
    );
}

// ---- ProbeView ----------------------------------------------------------

/// 2-D probe layout with channels drawn as small dots. For each selected
/// cluster, we use the [`WaveformCache`] (already populated by the
/// WaveformView) to find the peak channel and draw a colour-matching ring
/// at that position. Without populated waveforms we just show the layout.
pub fn probe_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
    channel_positions: &[[f32; 2]],
    waveforms: &WaveformCache,
) {
    let avail = ui.available_size_before_wrap();
    let height = avail.y.max(220.0);
    let (rect, _resp) =
        ui.allocate_exact_size(egui::vec2(avail.x.max(160.0), height), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, Color32::from_gray(18));

    if channel_positions.is_empty() {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "channel_positions.npy not loaded — probe layout unavailable",
            egui::FontId::proportional(13.0),
            Color32::GRAY,
        );
        return;
    }

    // Probe extents → screen rect with a small margin.
    let mut x_min = f32::INFINITY;
    let mut x_max = f32::NEG_INFINITY;
    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for &[x, y] in channel_positions {
        x_min = x_min.min(x);
        x_max = x_max.max(x);
        y_min = y_min.min(y);
        y_max = y_max.max(y);
    }
    let x_span = (x_max - x_min).max(1e-6);
    let y_span = (y_max - y_min).max(1e-6);
    let margin = 28.0;
    let inner = rect.shrink(margin);

    // Map (x, y) probe coords to inner-rect screen coords.
    // y is increasing upward conventionally on probes — phy flips so deeper
    // channels are at the bottom, which matches what users expect.
    let project = |x: f32, y: f32| -> egui::Pos2 {
        let nx = (x - x_min) / x_span;
        let ny = (y - y_min) / y_span;
        egui::pos2(
            inner.left() + nx * inner.width(),
            inner.bottom() - ny * inner.height(),
        )
    };

    // Channel dots.
    for &[x, y] in channel_positions {
        painter.circle_filled(project(x, y), 2.0, Color32::from_gray(80));
    }

    // For each selected cluster, find peak channel from the waveform cache
    // and draw a ring at that channel's position.
    let n_channels = session.provider.n_channels() as usize;
    for (ci, &c) in selection.iter().enumerate() {
        let Some(cluster_means) = waveforms.means.get(ci) else {
            continue;
        };
        if cluster_means.is_empty() {
            continue;
        }
        // Peak channel = argmax of max|mean|.
        let mut peak_ch = 0usize;
        let mut peak_val = -1.0f32;
        for (ch, mean) in cluster_means.iter().enumerate() {
            if ch >= n_channels.min(channel_positions.len()) {
                break;
            }
            for &v in mean {
                let a = v.abs();
                if a > peak_val {
                    peak_val = a;
                    peak_ch = ch;
                }
            }
        }
        if peak_val < 0.0 || peak_ch >= channel_positions.len() {
            continue;
        }
        let [px, py] = channel_positions[peak_ch];
        let centre = project(px, py);
        let colour = cluster_colour(c);
        // Outer ring sized by relative cluster position in selection so
        // multiple selected clusters at the same peak channel don't perfectly
        // occlude each other.
        let radius = 5.0 + (ci as f32) * 0.5;
        painter.circle_stroke(centre, radius, Stroke::new(1.6_f32, colour));
        painter.circle_filled(centre, 1.5, colour);
    }

    // Caption.
    painter.text(
        rect.left_top() + egui::vec2(6.0, 4.0),
        egui::Align2::LEFT_TOP,
        format!(
            "{} channels · x [{:.0}, {:.0}] · y [{:.0}, {:.0}] (µm)",
            channel_positions.len(),
            x_min,
            x_max,
            y_min,
            y_max,
        ),
        egui::FontId::proportional(11.0),
        Color32::GRAY,
    );
    if waveforms.means.is_empty() {
        painter.text(
            rect.right_top() + egui::vec2(-6.0, 4.0),
            egui::Align2::RIGHT_TOP,
            "open Waveforms tab to populate peak rings",
            egui::FontId::proportional(11.0),
            Color32::GRAY,
        );
    }
}

// ---- TemplateView -------------------------------------------------------

/// Primary (most common) template id for a cluster. Returns `None` for
/// empty / never-seeded clusters.
fn primary_template_id<P: DataProvider>(
    session: &Session<P>,
    cluster: ClusterId,
) -> Option<u32> {
    let templates = session.spike_templates(cluster);
    if templates.is_empty() {
        return None;
    }
    // Bucket counts by template id. n_templates is typically O(1k), so a
    // small HashMap is fine; we don't know the upper bound here.
    use std::collections::HashMap;
    let mut counts: HashMap<u32, usize> = HashMap::new();
    for &t in templates {
        *counts.entry(t).or_insert(0) += 1;
    }
    counts.into_iter().max_by_key(|&(_, c)| c).map(|(t, _)| t)
}

/// Render the per-cluster template waveform on its own peak channel,
/// colour-coded by cluster. Falls back to a hint when templates aren't
/// seeded.
pub fn template_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
) {
    if !session.has_template_waveforms() {
        ui.label("templates.npy not loaded — TemplateView unavailable");
        return;
    }
    if selection.is_empty() {
        ui.label("no selection");
        return;
    }
    let (n_templates, n_samples, n_channels) = session.template_shape();
    if n_templates == 0 || n_samples < 2 || n_channels == 0 {
        ui.label("template buffer is empty");
        return;
    }

    // Resolve primary template + peak channel per cluster.
    let mut entries: Vec<(ClusterId, u32, usize, Vec<f32>)> = Vec::new(); // (cluster, tpl_id, peak_ch, waveform on peak)
    let mut y_max = 1e-6f32;
    for &c in selection {
        let Some(tid) = primary_template_id(session, c) else {
            continue;
        };
        let Some(flat) = session.template_waveform(tid) else {
            continue;
        };
        // Peak channel = argmax of max|template[*, ch]|.
        let mut peak_ch = 0usize;
        let mut peak_val = -1.0f32;
        for ch in 0..n_channels {
            for s in 0..n_samples {
                let v = flat[s * n_channels + ch].abs();
                if v > peak_val {
                    peak_val = v;
                    peak_ch = ch;
                }
            }
        }
        let waveform: Vec<f32> = (0..n_samples)
            .map(|s| flat[s * n_channels + peak_ch])
            .collect();
        for &v in &waveform {
            y_max = y_max.max(v.abs());
        }
        entries.push((c, tid, peak_ch, waveform));
    }
    if entries.is_empty() {
        ui.label("no spike-templates seeded for this selection");
        return;
    }

    let avail = ui.available_size_before_wrap();
    let height = avail.y.max(160.0).min(420.0);
    let (rect, _resp) =
        ui.allocate_exact_size(egui::vec2(avail.x.max(120.0), height), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, Color32::from_gray(18));
    let cy = rect.center().y;
    painter.line_segment(
        [Pos2::new(rect.left(), cy), Pos2::new(rect.right(), cy)],
        Stroke::new(0.5_f32, Color32::from_gray(50)),
    );

    let x_step = rect.width() / (n_samples as f32 - 1.0).max(1.0);
    let y_scale = (rect.height() * 0.45) / y_max;

    for (c, tid, peak_ch, waveform) in &entries {
        let colour = cluster_colour(*c);
        let stroke = Stroke::new(2.0_f32, colour);
        let mut prev: Option<Pos2> = None;
        for (s, &v) in waveform.iter().enumerate() {
            let p = Pos2::new(rect.left() + s as f32 * x_step, cy - v * y_scale);
            if let Some(pp) = prev {
                painter.line_segment([pp, p], stroke);
            }
            prev = Some(p);
        }
        let _ = (tid, peak_ch); // captured in caption below if we add hover
    }

    // Caption: list (cluster, template id, peak channel) tuples.
    let summary: Vec<String> = entries
        .iter()
        .map(|(c, tid, ch, _)| format!("c{c}→t{tid}@ch{ch}"))
        .collect();
    painter.text(
        rect.left_top() + egui::vec2(6.0, 4.0),
        egui::Align2::LEFT_TOP,
        format!("{n_samples} samples · {}", summary.join(", ")),
        egui::FontId::proportional(11.0),
        Color32::GRAY,
    );
}

// ---- SimilarityView -----------------------------------------------------

/// For the first selected cluster, list the top-N most similar templates'
/// clusters with their similarity scores. Lets the curator quickly hop to
/// candidate-merge partners.
pub fn similarity_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
) {
    if !session.has_template_waveforms() {
        ui.label("templates.npy not loaded — SimilarityView unavailable");
        return;
    }
    let sim = session.similar_templates();
    if sim.is_empty() {
        ui.label("similar_templates.npy not loaded — no template-similarity matrix");
        return;
    }
    let Some(&primary) = selection.first() else {
        ui.label("no selection");
        return;
    };
    let Some(tid) = primary_template_id(session, primary) else {
        ui.label(format!("cluster {primary} has no spike templates"));
        return;
    };
    let (n_templates, _n_samples, _n_channels) = session.template_shape();
    if (tid as usize) >= n_templates {
        ui.label("primary template id out of range");
        return;
    }

    // Row of similarity scores for `tid`.
    let row_start = (tid as usize) * n_templates;
    let row = &sim[row_start..row_start + n_templates];

    // Find top-N similar templates (excluding self).
    const TOP_N: usize = 12;
    let mut ranked: Vec<(u32, f32)> = row
        .iter()
        .enumerate()
        .filter(|(j, _)| *j as u32 != tid)
        .map(|(j, &v)| (j as u32, v))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(TOP_N);

    // Map template id → first cluster whose primary template is that id.
    // Cheap O(n_clusters) walk; small N.
    let n_clusters = session.n_clusters();
    let mut tpl_to_cluster: std::collections::HashMap<u32, ClusterId> =
        std::collections::HashMap::new();
    for c in (0..n_clusters).map(ClusterId) {
        if let Some(t) = primary_template_id(session, c) {
            tpl_to_cluster.entry(t).or_insert(c);
        }
    }

    ui.heading(format!(
        "Most similar to cluster {primary} (template {tid})",
    ));
    ui.separator();
    use egui_extras::{Column, TableBuilder};
    TableBuilder::new(ui)
        .striped(true)
        .resizable(false)
        .column(Column::auto().at_least(48.0))
        .column(Column::auto().at_least(64.0))
        .column(Column::auto().at_least(72.0))
        .column(Column::auto().at_least(64.0))
        .header(20.0, |mut h| {
            h.col(|ui| {
                ui.strong("template");
            });
            h.col(|ui| {
                ui.strong("similarity");
            });
            h.col(|ui| {
                ui.strong("first cluster");
            });
            h.col(|ui| {
                ui.strong("# spikes");
            });
        })
        .body(|mut body| {
            for &(tpl, score) in &ranked {
                body.row(18.0, |mut row| {
                    row.col(|ui| {
                        ui.label(format!("t{tpl}"));
                    });
                    row.col(|ui| {
                        ui.label(format!("{score:.3}"));
                    });
                    row.col(|ui| match tpl_to_cluster.get(&tpl) {
                        Some(&c) => {
                            ui.colored_label(cluster_colour(c), format!("c{c}"));
                        }
                        None => {
                            ui.weak("—");
                        }
                    });
                    row.col(|ui| {
                        let label = match tpl_to_cluster.get(&tpl) {
                            Some(&c) => session.spike_times(c).len().to_string(),
                            None => "—".to_string(),
                        };
                        ui.label(label);
                    });
                });
            }
        });
}

// ---- ClusterStatistics --------------------------------------------------

/// Histograms of three scalar metrics across *every* cluster: spike count,
/// mean amplitude, ISI<1ms count. Bars are coloured by their relative bin
/// rank so eye-balling outliers is easy.
pub fn cluster_statistics_view<P: DataProvider>(ui: &mut Ui, session: &Session<P>) {
    use sorrel_compute::{isi_violations, mean_amplitude};
    let n = session.n_clusters() as usize;
    if n == 0 {
        ui.label("no clusters");
        return;
    }

    // Collect three series.
    let refractory_samples = (session.provider.sample_rate() * 0.001).round() as u64;
    let mut spike_counts: Vec<f32> = Vec::with_capacity(n);
    let mut mean_amps: Vec<f32> = Vec::new();
    let mut isi_viols: Vec<f32> = Vec::with_capacity(n);
    for c in (0..n as u32).map(ClusterId) {
        spike_counts.push(session.spike_times(c).len() as f32);
        let amps = session.spike_amplitudes(c);
        if !amps.is_empty() {
            mean_amps.push(mean_amplitude(amps));
        }
        isi_viols.push(isi_violations(session.spike_times(c), refractory_samples) as f32);
    }

    histogram_panel(ui, "spike count", &spike_counts, 30, Color32::from_rgb(78, 121, 167));
    if !mean_amps.is_empty() {
        histogram_panel(
            ui,
            "mean amplitude",
            &mean_amps,
            30,
            Color32::from_rgb(89, 161, 79),
        );
    }
    histogram_panel(
        ui,
        "ISI<1ms count",
        &isi_viols,
        30,
        Color32::from_rgb(225, 87, 89),
    );
}

fn histogram_panel(ui: &mut Ui, title: &str, values: &[f32], bins: usize, colour: Color32) {
    if values.is_empty() {
        return;
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in values {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    if min == f32::INFINITY {
        return;
    }
    if (max - min).abs() < 1e-6 {
        // All values identical — pad so the bin range isn't zero-width.
        max = min + 1.0;
    }
    let span = max - min;
    let inv_bin = bins as f32 / span;
    let mut h = vec![0u32; bins];
    for &v in values {
        let mut i = ((v - min) * inv_bin) as usize;
        if i >= bins {
            i = bins - 1;
        }
        h[i] += 1;
    }
    let max_count = h.iter().copied().max().unwrap_or(1).max(1) as f32;

    ui.label(format!(
        "{title} · n={} · range [{min:.2}, {max:.2}]",
        values.len()
    ));
    let avail = ui.available_size_before_wrap();
    let (rect, _resp) =
        ui.allocate_exact_size(egui::vec2(avail.x.max(120.0), 80.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, Color32::from_gray(18));

    let bw = rect.width() / bins as f32;
    for (i, &count) in h.iter().enumerate() {
        let h_norm = count as f32 / max_count;
        let bx0 = rect.left() + bw * i as f32;
        let bx1 = bx0 + bw * 0.95;
        let by0 = rect.bottom() - h_norm * rect.height();
        painter.rect_filled(
            egui::Rect::from_min_max(egui::pos2(bx0, by0), egui::pos2(bx1, rect.bottom())),
            0.0,
            colour,
        );
    }
    painter.line_segment(
        [rect.left_bottom(), rect.right_bottom()],
        Stroke::new(0.5_f32, Color32::from_gray(60)),
    );
    ui.add_space(4.0);
}

// ---- SuggestView (statistical merge/split candidates) -------------------

/// Action emitted when the curator clicks a row in the suggest panel.
#[derive(Clone, Debug)]
pub enum SuggestAction {
    /// Replace the active selection with these clusters (typically a merge
    /// pair the user is reviewing) — the curator inspects, then triggers
    /// the merge themselves with the existing keybinding.
    Select(Vec<ClusterId>),
    /// Apply the merge directly: reassign all spikes from `sources` into
    /// `target`. The UI dispatches a `CurationCommand::Merge` for this.
    ApplyMerge {
        sources: Vec<ClusterId>,
        target: ClusterId,
    },
    /// Apply every merge in the cache that exceeds `threshold` and whose
    /// preview does not warn, as a single atomic batch. Reversible with
    /// one Undo.
    ApplyHighConfidenceMerges { threshold: f32 },
    /// Apply a GMM-derived split: move the listed local spike indices of
    /// `cluster` to a fresh cluster id. The UI dispatches a
    /// `CurationCommand::Split` for this.
    ApplyGmmSplit {
        cluster: ClusterId,
        spike_idx: Vec<u32>,
    },
}

/// Lightweight summary of a merge preview — only the fields the UI needs.
/// We don't keep the full `MergePreview` because it would force the cache
/// to allocate a `QualityBreakdown` per row.
#[derive(Clone, Copy, Debug, Default)]
pub struct MergePreviewSummary {
    pub composite_delta: f32,
    pub contamination_delta: f32,
    pub warns: bool,
}

/// Cache of the last-computed suggestion lists. Recomputed when the
/// curation history changes (any merge/split/relabel invalidates everything
/// — the suggesters score against the live state).
#[derive(Default)]
pub struct SuggestCache {
    merges: Vec<MergeCandidate>,
    splits: Vec<SplitCandidate>,
    /// Indexed parallel to `merges` — predicted post-merge deltas.
    merge_previews: Vec<MergePreviewSummary>,
    last_key: (usize, usize, u32),
    pub config: SuggestConfig,
}

impl SuggestCache {
    pub fn new() -> Self {
        Self::default()
    }

    fn ensure<P: DataProvider>(&mut self, session: &Session<P>) {
        let key = (
            session.history_len(),
            session.redo_len(),
            session.n_clusters(),
        );
        if key == self.last_key && !(self.merges.is_empty() && self.splits.is_empty()) {
            return;
        }
        self.merges = rank_merge_candidates(session, &self.config);
        self.splits = rank_split_candidates(session, &self.config);
        // Parallel preview computation. Session<P> is Send + Sync now that
        // the journal is a plain `BufWriter<File>` (the rusqlite RefCell
        // internals are gone), so rayon can capture `&Session` directly.
        use rayon::prelude::*;
        self.merge_previews = self
            .merges
            .par_iter()
            .map(|m| {
                let p = preview_merge(session, &[m.a], m.b);
                MergePreviewSummary {
                    composite_delta: p.composite_delta(),
                    contamination_delta: p.contamination_delta(),
                    warns: p.warns(),
                }
            })
            .collect();
        self.last_key = key;
    }

    pub fn invalidate(&mut self) {
        self.last_key = (usize::MAX, usize::MAX, u32::MAX);
        self.merges.clear();
        self.splits.clear();
        self.merge_previews.clear();
    }

    /// Pick out the merge pairs whose suggester score is at least
    /// `threshold` and whose preview does not warn (no contamination
    /// regression, no composite drop). Returns `(source, target)` pairs
    /// in ranked order. We deduplicate so a cluster only appears once on
    /// either side: applying overlapping merges in one batch would change
    /// what the second merge meant.
    pub fn high_confidence_merges(&self, threshold: f32) -> Vec<(ClusterId, ClusterId)> {
        use std::collections::HashSet;
        let mut taken: HashSet<ClusterId> = HashSet::new();
        let mut out = Vec::new();
        for (m, p) in self.merges.iter().zip(self.merge_previews.iter()) {
            if m.score < threshold || p.warns {
                continue;
            }
            if taken.contains(&m.a) || taken.contains(&m.b) {
                continue;
            }
            taken.insert(m.a);
            taken.insert(m.b);
            out.push((m.a, m.b));
        }
        out
    }
}

/// Render the suggest panel — two tables (merges, splits) with per-row
/// evidence and click-to-act buttons. Returns the action the curator
/// triggered, if any.
pub fn suggest_view<P, F>(
    ui: &mut Ui,
    session: &Session<P>,
    cache: &mut SuggestCache,
    label_str: F,
) -> Option<SuggestAction>
where
    P: DataProvider,
    F: Fn(&P::Label) -> &'static str,
{
    cache.ensure(session);
    let mut action: Option<SuggestAction> = None;

    ui.horizontal(|ui| {
        ui.heading("Statistical suggestions");
        ui.weak(format!(
            "{} merges · {} splits · refractory {:.1} ms · CCG ±{:.0} ms",
            cache.merges.len(),
            cache.splits.len(),
            cache.config.refractory_seconds * 1000.0,
            cache.config.ccg_max_lag_seconds * 1000.0,
        ));
        if ui.small_button("recompute").clicked() {
            cache.invalidate();
        }
        let n_safe = cache
            .merges
            .iter()
            .zip(cache.merge_previews.iter())
            .filter(|(m, p)| m.score >= 0.7 && !p.warns)
            .count();
        let label = format!("apply {n_safe} high-conf merges");
        let response = ui.add_enabled(
            n_safe > 0,
            egui::Button::new(label).small(),
        );
        if response
            .on_hover_text(
                "Applies every score≥0.70 merge whose predicted Δquality is not negative. \
                 Atomic — one Undo reverts the whole batch.",
            )
            .clicked()
        {
            action = Some(SuggestAction::ApplyHighConfidenceMerges { threshold: 0.7 });
        }
    });
    ui.separator();

    egui::CollapsingHeader::new("Merge candidates")
        .default_open(true)
        .show(ui, |ui| {
            if cache.merges.is_empty() {
                ui.weak(
                    "No merge candidates above threshold — increase recording length, \
                     or seed amplitudes for stronger evidence.",
                );
            } else {
                let merge_action =
                    render_merge_table(ui, &cache.merges, &cache.merge_previews);
                if action.is_none() {
                    action = merge_action;
                }
            }
        });

    ui.add_space(6.0);
    egui::CollapsingHeader::new("Split candidates")
        .default_open(true)
        .show(ui, |ui| {
            if cache.splits.is_empty() {
                ui.weak("No split candidates above threshold.");
            } else {
                let split_action = render_split_table(ui, session, &cache.splits, &label_str);
                if action.is_none() {
                    action = split_action;
                }
            }
        });

    action
}

fn render_merge_table(
    ui: &mut Ui,
    merges: &[MergeCandidate],
    previews: &[MergePreviewSummary],
) -> Option<SuggestAction> {
    use egui_extras::{Column, TableBuilder};
    let mut action: Option<SuggestAction> = None;
    TableBuilder::new(ui)
        .striped(true)
        .resizable(true)
        .column(Column::auto().at_least(56.0))
        .column(Column::auto().at_least(56.0))
        .column(Column::auto().at_least(54.0))
        .column(Column::auto().at_least(54.0))
        .column(Column::auto().at_least(54.0))
        .column(Column::auto().at_least(64.0))
        .column(Column::auto().at_least(64.0))
        .column(Column::auto().at_least(64.0))
        .column(Column::auto().at_least(140.0))
        .header(22.0, |mut h| {
            h.col(|ui| { ui.strong("a"); });
            h.col(|ui| { ui.strong("b"); });
            h.col(|ui| { ui.strong("score"); });
            h.col(|ui| { ui.strong("dip z"); });
            h.col(|ui| { ui.strong("amp KS"); });
            h.col(|ui| { ui.strong("Δamp"); });
            h.col(|ui| { ui.strong("Δqual").on_hover_text("predicted change in composite quality after merge"); });
            h.col(|ui| { ui.strong("Δcontam").on_hover_text("predicted change in refractory contamination"); });
            h.col(|ui| { ui.strong("action"); });
        })
        .body(|mut body| {
            for (idx, m) in merges.iter().enumerate() {
                let preview = previews.get(idx).copied().unwrap_or_default();
                body.row(22.0, |mut row| {
                    row.col(|ui| {
                        ui.colored_label(cluster_colour(m.a), format!("c{}", m.a));
                    });
                    row.col(|ui| {
                        ui.colored_label(cluster_colour(m.b), format!("c{}", m.b));
                    });
                    row.col(|ui| {
                        ui.label(format!("{:.2}", m.score));
                    });
                    row.col(|ui| {
                        ui.label(format!("{:+.1}", m.ccg_dip_z));
                    });
                    row.col(|ui| {
                        ui.label(format!("{:.2}", m.amp_ks));
                    });
                    row.col(|ui| {
                        ui.label(format!("{:.2}", m.amp_mean_delta));
                    });
                    row.col(|ui| {
                        let txt = format!("{:+.2}", preview.composite_delta);
                        let colour = if preview.composite_delta < -0.05 {
                            Color32::from_rgb(225, 87, 89)
                        } else if preview.composite_delta > 0.02 {
                            Color32::from_rgb(89, 161, 79)
                        } else {
                            Color32::GRAY
                        };
                        ui.colored_label(colour, txt);
                    });
                    row.col(|ui| {
                        let txt = format!("{:+.3}", preview.contamination_delta);
                        let colour = if preview.contamination_delta > 0.05 {
                            Color32::from_rgb(225, 87, 89)
                        } else {
                            Color32::GRAY
                        };
                        ui.colored_label(colour, txt);
                    });
                    row.col(|ui| {
                        if ui.small_button("review").clicked() {
                            action = Some(SuggestAction::Select(vec![m.a, m.b]));
                        }
                        let merge_label = if preview.warns { "merge ⚠" } else { "merge →" };
                        let tt = if preview.warns {
                            format!(
                                "Predicted Δquality {:+.2}, Δcontam {:+.3}. \
                                 The merge looks risky — review before applying.",
                                preview.composite_delta, preview.contamination_delta,
                            )
                        } else {
                            "Reassigns all spikes from a into b. Reversible via undo.".into()
                        };
                        if ui.small_button(merge_label).on_hover_text(tt).clicked() {
                            action = Some(SuggestAction::ApplyMerge {
                                sources: vec![m.a],
                                target: m.b,
                            });
                        }
                    });
                });
            }
        });
    action
}

fn render_split_table<P, F>(
    ui: &mut Ui,
    session: &Session<P>,
    splits: &[SplitCandidate],
    label_str: &F,
) -> Option<SuggestAction>
where
    P: DataProvider,
    F: Fn(&P::Label) -> &'static str,
{
    use egui_extras::{Column, TableBuilder};
    let mut action: Option<SuggestAction> = None;
    TableBuilder::new(ui)
        .striped(true)
        .resizable(true)
        .column(Column::auto().at_least(56.0))
        .column(Column::auto().at_least(54.0))
        .column(Column::auto().at_least(56.0))
        .column(Column::auto().at_least(64.0))
        .column(Column::auto().at_least(56.0))
        .column(Column::auto().at_least(58.0))
        .column(Column::auto().at_least(56.0))
        .column(Column::auto().at_least(140.0))
        .header(22.0, |mut h| {
            h.col(|ui| { ui.strong("cluster"); });
            h.col(|ui| { ui.strong("score"); });
            h.col(|ui| { ui.strong("BC"); });
            h.col(|ui| { ui.strong("contam"); });
            h.col(|ui| { ui.strong("|drift|"); });
            h.col(|ui| { ui.strong("ΔBIC").on_hover_text("BIC(k=1) − BIC(k=2) on amplitudes; >6 = strong evidence for splitting"); });
            h.col(|ui| { ui.strong("label"); });
            h.col(|ui| { ui.strong("action"); });
        })
        .body(|mut body| {
            for s in splits {
                body.row(22.0, |mut row| {
                    row.col(|ui| {
                        ui.colored_label(cluster_colour(s.cluster), format!("c{}", s.cluster));
                    });
                    row.col(|ui| {
                        ui.label(format!("{:.2}", s.score));
                    });
                    row.col(|ui| {
                        ui.label(format!("{:.2}", s.amp_bimodality));
                    });
                    row.col(|ui| {
                        ui.label(format!("{:.3}", s.contamination));
                    });
                    row.col(|ui| {
                        ui.label(format!("{:.2}", s.drift_corr));
                    });
                    row.col(|ui| {
                        if s.gmm_bic_delta.is_finite() {
                            let colour = if s.gmm_bic_delta >= 6.0 {
                                Color32::from_rgb(89, 161, 79)
                            } else {
                                Color32::GRAY
                            };
                            ui.colored_label(colour, format!("{:+.1}", s.gmm_bic_delta));
                        } else {
                            ui.label("—");
                        }
                    });
                    row.col(|ui| {
                        ui.label(label_str(&session.label(s.cluster).unwrap_or_default()));
                    });
                    row.col(|ui| {
                        if ui
                            .small_button("inspect")
                            .on_hover_text(
                                "Selects this cluster so you can lasso-split it \
                                 in the Features tab.",
                            )
                            .clicked()
                        {
                            action = Some(SuggestAction::Select(vec![s.cluster]));
                        }
                        if !s.bipartition_minor_idx.is_empty()
                            && ui
                                .small_button("split →")
                                .on_hover_text(format!(
                                    "Move {} amplitude-outlier spikes to a fresh cluster. \
                                     Reversible via undo.",
                                    s.bipartition_minor_idx.len(),
                                ))
                                .clicked()
                        {
                            action = Some(SuggestAction::ApplyGmmSplit {
                                cluster: s.cluster,
                                spike_idx: s.bipartition_minor_idx.clone(),
                            });
                        }
                    });
                });
            }
        });
    action
}

// ---- QualityView (per-cluster composite score breakdown) ---------------

/// Render a quality-score panel for the selected clusters: the composite
/// score plus a breakdown into the six contributing factors. Hovering each
/// score reveals the underlying raw metric.
pub fn quality_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
) {
    if selection.is_empty() {
        ui.label("no selection");
        return;
    }
    let has_features = session.has_pc_features();

    use egui_extras::{Column, TableBuilder};
    let mut tb = TableBuilder::new(ui)
        .striped(true)
        .resizable(true)
        .column(Column::auto().at_least(54.0))
        .column(Column::auto().at_least(60.0))
        .column(Column::auto().at_least(60.0))
        .column(Column::auto().at_least(60.0))
        .column(Column::auto().at_least(60.0))
        .column(Column::auto().at_least(60.0))
        .column(Column::auto().at_least(60.0))
        .column(Column::auto().at_least(60.0));
    if has_features {
        tb = tb.column(Column::auto().at_least(60.0));
    }
    tb.header(22.0, |mut h| {
            h.col(|ui| { ui.strong("cluster"); });
            h.col(|ui| { ui.strong("composite"); });
            h.col(|ui| { ui.strong("clean"); });
            h.col(|ui| { ui.strong("present"); });
            h.col(|ui| { ui.strong("cover"); });
            h.col(|ui| { ui.strong("SNR"); });
            h.col(|ui| { ui.strong("complete"); });
            h.col(|ui| { ui.strong("stable"); });
            if has_features {
                h.col(|ui| { ui.strong("isolated"); });
            }
        })
        .body(|mut body| {
            for &c in selection {
                let q: QualityBreakdown = cluster_quality(session, c);
                let composite = q.composite();
                body.row(20.0, |mut row| {
                    row.col(|ui| {
                        ui.colored_label(cluster_colour(c), format!("c{c}"));
                    });
                    row.col(|ui| {
                        ui.label(format!("{:.2}", composite));
                    });
                    row.col(|ui| {
                        ui.label(format!("{:.2}", q.contamination_score)).on_hover_text(format!(
                            "refractory contamination = {:.3}\nISI<refrac = {}",
                            q.raw_contamination, q.raw_isi_violations,
                        ));
                    });
                    row.col(|ui| {
                        ui.label(format!("{:.2}", q.presence_score)).on_hover_text(format!(
                            "presence ratio = {:.2}\npresence CV = {:.2}",
                            q.raw_presence_ratio, q.raw_presence_cv,
                        ));
                    });
                    row.col(|ui| {
                        ui.label(format!("{:.2}", q.coverage_score)).on_hover_text(format!(
                            "largest silent gap = {:.2}",
                            q.raw_silent_gap
                        ));
                    });
                    row.col(|ui| {
                        ui.label(format!("{:.2}", q.snr_score))
                            .on_hover_text(format!("amplitude SNR = {:.2}", q.raw_snr));
                    });
                    row.col(|ui| {
                        ui.label(format!("{:.2}", q.completeness_score))
                            .on_hover_text(format!("amplitude cutoff = {:.3}", q.raw_amp_cutoff));
                    });
                    row.col(|ui| {
                        ui.label(format!("{:.2}", q.stability_score))
                            .on_hover_text(format!("drift correlation = {:+.2}", q.raw_drift_corr));
                    });
                    if has_features {
                        row.col(|ui| match q.isolation {
                            Some(iso) if iso.has_evidence() => {
                                let s = iso.score();
                                let tt = format!(
                                    "isolation² = {:.1}\nL-ratio = {:.3}\nNN-isolation = {:.2}\nN_in = {}, N_out = {}",
                                    iso.isolation_distance_sq,
                                    iso.l_ratio,
                                    iso.nn_isolation,
                                    iso.n_in,
                                    iso.n_out,
                                );
                                let label = if s.is_finite() {
                                    format!("{:.2}", s)
                                } else {
                                    "—".to_string()
                                };
                                ui.label(label).on_hover_text(tt);
                            }
                            _ => {
                                ui.label("—")
                                    .on_hover_text("not enough spikes / no PC evidence");
                            }
                        });
                    }
                });
            }
        });
    ui.add_space(4.0);
    ui.weak(
        "Higher is better. Composite = geometric mean of all available sub-scores — \
         one weak dimension drags it down. Hover any score for the raw metric.",
    );
}

// ---- DriftMapView (amp vs time scatter colour-coded by density) ---------

/// Per-spike amplitude (y) vs spike time (x) scatter, with point density
/// reflected in alpha. The single most useful drift diagnostic in phylib —
/// a healthy cluster paints a flat horizontal cloud; a drifting cluster
/// shows a sloped or fragmented one.
///
/// Renders in pure egui — fast enough for tens of thousands of points
/// because we sub-sample at a fixed point budget. Selected clusters are
/// overlaid with the standard palette.
pub fn drift_map_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
) {
    if selection.is_empty() {
        ui.label("no selection");
        return;
    }
    let avail = ui.available_size_before_wrap();
    let height = avail.y.max(220.0).min(560.0);
    let (rect, _resp) =
        ui.allocate_exact_size(egui::vec2(avail.x.max(160.0), height), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, Color32::from_gray(18));

    let inv_sr = 1.0_f32 / session.provider.sample_rate().max(1.0);

    // Determine global bounds across the selection.
    let mut t_min = f32::INFINITY;
    let mut t_max = f32::NEG_INFINITY;
    let mut a_min = f32::INFINITY;
    let mut a_max = f32::NEG_INFINITY;
    let mut total_spikes = 0usize;
    for &c in selection {
        let times = session.spike_times(c);
        let amps = session.spike_amplitudes(c);
        if times.is_empty() || amps.is_empty() {
            continue;
        }
        total_spikes += times.len();
        if let (Some(&first), Some(&last)) = (times.first(), times.last()) {
            t_min = t_min.min(first.as_f32() * inv_sr);
            t_max = t_max.max(last.as_f32() * inv_sr);
        }
        for &a in amps {
            if a < a_min { a_min = a; }
            if a > a_max { a_max = a; }
        }
    }
    if total_spikes == 0 || !t_min.is_finite() || !a_min.is_finite() {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "amplitudes not seeded for this backend",
            egui::FontId::proportional(13.0),
            Color32::GRAY,
        );
        return;
    }
    // Pad amplitude range slightly so points don't touch the borders.
    let a_pad = (a_max - a_min).max(1e-3) * 0.05;
    a_min -= a_pad;
    a_max += a_pad;
    let t_span = (t_max - t_min).max(1e-6);
    let a_span = (a_max - a_min).max(1e-6);

    const POINT_BUDGET: usize = 30_000;
    let stride = ((total_spikes as f32 / POINT_BUDGET as f32).ceil() as usize).max(1);

    // Linear regression line per cluster: shows the drift direction.
    for &c in selection {
        let times = session.spike_times(c);
        let amps = session.spike_amplitudes(c);
        if times.len() < 3 || amps.len() < 3 {
            continue;
        }
        let colour = cluster_colour(c);
        let faint = Color32::from_rgba_unmultiplied(colour.r(), colour.g(), colour.b(), 60);
        let n = times.len().min(amps.len());
        for i in (0..n).step_by(stride) {
            let t = times[i].as_f32() * inv_sr;
            let a = amps[i];
            let x = rect.left() + (t - t_min) / t_span * rect.width();
            let y = rect.bottom() - (a - a_min) / a_span * rect.height();
            painter.circle_filled(egui::pos2(x, y), 1.4, faint);
        }
        // OLS regression line on (t, a). cheap.
        let nf = n as f64;
        let mean_t: f64 = times[..n].iter().map(|&t| t.as_f64() * inv_sr as f64).sum::<f64>() / nf;
        let mean_a: f64 = amps[..n].iter().map(|&a| a as f64).sum::<f64>() / nf;
        let mut num = 0.0_f64;
        let mut denom = 0.0_f64;
        for i in 0..n {
            let dt = times[i].as_f64() * inv_sr as f64 - mean_t;
            let da = amps[i] as f64 - mean_a;
            num += dt * da;
            denom += dt * dt;
        }
        if denom <= 0.0 {
            continue;
        }
        let slope = num / denom;
        let intercept = mean_a - slope * mean_t;
        let predict = |t: f64| (slope * t + intercept) as f32;
        let xy = |t: f32, a: f32| {
            egui::pos2(
                rect.left() + (t - t_min) / t_span * rect.width(),
                rect.bottom() - (a - a_min) / a_span * rect.height(),
            )
        };
        let p0 = xy(t_min, predict(t_min as f64));
        let p1 = xy(t_max, predict(t_max as f64));
        painter.line_segment([p0, p1], Stroke::new(1.6_f32, colour));
    }

    painter.text(
        rect.left_top() + egui::vec2(6.0, 4.0),
        egui::Align2::LEFT_TOP,
        format!(
            "drift map · {total_spikes} spikes · t [{t_min:.1}, {t_max:.1}] s · \
             amp [{a_min:.1}, {a_max:.1}] · regression line per cluster",
        ),
        egui::FontId::proportional(11.0),
        Color32::GRAY,
    );
}

// ---- ContaminationOverTimeView (sliding refractory contamination) ------

/// Plot refractory contamination across `n_windows` time slices, one line
/// per selected cluster. Spikes in clusters not in the selection are
/// ignored. Useful to spot cases where contamination is concentrated in a
/// short stretch (drift artifact, noise burst) rather than uniform across
/// the recording.
pub fn contamination_over_time_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
) {
    if selection.is_empty() {
        ui.label("no selection");
        return;
    }
    let sr = session.provider.sample_rate().max(1.0);
    let total_samples = session.provider.n_samples().0;
    if total_samples == 0 {
        ui.label("recording has zero length");
        return;
    }
    let total_s = total_samples as f32 / sr;
    let refractory_samples = (sr * 0.0015).round() as u64;
    const N_WINDOWS: usize = 60;
    const MIN_SPIKES_PER_WINDOW: usize = 20;

    // Compute per-cluster series.
    let series: Vec<(ClusterId, Vec<(f32, f32)>)> = selection
        .iter()
        .map(|&c| {
            let times = session.spike_times(c);
            let s = sliding_refractory_contamination(
                times,
                refractory_samples,
                total_samples,
                sr,
                N_WINDOWS,
                MIN_SPIKES_PER_WINDOW,
            );
            (c, s)
        })
        .collect();

    let mut y_max = 0.05_f32;
    for (_, s) in &series {
        for &(_, y) in s {
            if y.is_finite() && y > y_max {
                y_max = y;
            }
        }
    }
    y_max = y_max.max(0.05).min(2.0);

    let avail = ui.available_size_before_wrap();
    let height = avail.y.max(180.0).min(420.0);
    let (rect, _resp) =
        ui.allocate_exact_size(egui::vec2(avail.x.max(160.0), height), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, Color32::from_gray(18));

    // Reference line at the conventional 0.1 contamination threshold.
    let ref_y = 0.1_f32;
    if ref_y < y_max {
        let y = rect.bottom() - (ref_y / y_max) * rect.height();
        painter.line_segment(
            [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
            Stroke::new(0.5_f32, Color32::from_rgba_unmultiplied(225, 87, 89, 90)),
        );
        painter.text(
            egui::pos2(rect.right() - 4.0, y - 2.0),
            egui::Align2::RIGHT_BOTTOM,
            "0.1",
            egui::FontId::proportional(10.0),
            Color32::from_rgb(225, 87, 89),
        );
    }

    // Plot each cluster's polyline.
    for (c, s) in &series {
        let colour = cluster_colour(*c);
        let stroke = Stroke::new(1.5_f32, colour);
        let mut prev: Option<egui::Pos2> = None;
        for &(t, y) in s {
            if !y.is_finite() {
                prev = None;
                continue;
            }
            let x = rect.left() + (t / total_s).clamp(0.0, 1.0) * rect.width();
            let py = rect.bottom() - (y / y_max).clamp(0.0, 1.0) * rect.height();
            let p = egui::pos2(x, py);
            if let Some(pp) = prev {
                painter.line_segment([pp, p], stroke);
            }
            prev = Some(p);
        }
    }

    painter.text(
        rect.left_top() + egui::vec2(6.0, 4.0),
        egui::Align2::LEFT_TOP,
        format!(
            "contamination · {N_WINDOWS} windows × {:.1} s · y-max {y_max:.2}",
            total_s / N_WINDOWS as f32,
        ),
        egui::FontId::proportional(11.0),
        Color32::GRAY,
    );
    painter.text(
        rect.right_bottom() + egui::vec2(-6.0, -2.0),
        egui::Align2::RIGHT_BOTTOM,
        format!("0 — {total_s:.0} s"),
        egui::FontId::proportional(11.0),
        Color32::GRAY,
    );
}
