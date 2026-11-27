//! Self-contained HTML report (`sorrel report`): an overview table, then one
//! card per cluster with its headline QC numbers, the template on its
//! strongest channels, ISI (linear and log), ACG and amplitude over time as
//! inline SVG, and every metric underneath. The page is an askama template
//! (`templates/report.html`) checked and compiled into the binary. No
//! external assets, so the file can be mailed, dropped into a lab wiki or
//! printed to PDF (cards never split across pages).
//!
//! Snippets from the raw trace are deliberately not extracted here: that
//! is a per-spike disk walk the GUI does lazily per selection, and a
//! report over a thousand clusters would take minutes for a picture the
//! template already gives.

use crate::cache::correlograms::acg_or_compute;
use crate::qc_export::{collect_qc_rows, QcRow};
use crate::session::Session;
use anyhow::{Context, Result};
use askama::Template;
use rayon::prelude::*;
use sorrel_compute::{isi_histogram, isi_histogram_log};
use sorrel_io::{ClusterId, DataProvider, SampleIndex, SessionSpan};
use std::fmt::Write;
use std::path::Path;

/// Which clusters to include.
#[derive(Clone, Debug, Default)]
pub struct ReportFilter {
    /// Explicit cluster ids; `None` = all non-empty clusters.
    pub clusters: Option<Vec<u32>>,
    /// Only clusters whose label string equals this (`"good"`, ...).
    pub label: Option<String>,
    /// Only clusters with a tag starting with this (`"review:"` = the review queue).
    pub tag: Option<String>,
}

/// The plots of one cluster card, as SVG fragments.
pub struct ClusterSvgs {
    pub waveform: Option<String>,
    pub isi: String,
    pub log_isi: String,
    pub acg: String,
    pub amplitude: Option<String>,
    pub amplitude_iqr: Option<String>,
}

const W: f32 = 240.0;
const H: f32 = 80.0;
/// ISI and ACG windows: 50 ms at 1 ms bins, matching the GUI defaults.
const WINDOW_S: f32 = 0.050;
const BIN_S: f32 = 0.001;
/// Channels in the waveform panel: the strongest few, a tetrode's worth.
const WAVE_CHANNELS: usize = 4;
/// Log-ISI axis: 0.2 ms to 10 s.
const LOG_ISI_MIN_MS: f64 = 0.2;
const LOG_ISI_MAX_MS: f64 = 10_000.0;
const LOG_ISI_BINS: usize = 80;
/// Amplitude strip size and its dot budget (evenly strided beyond it).
const STRIP_W: f32 = 960.0;
const STRIP_H: f32 = 64.0;
const MAX_DOTS: usize = 2000;
const AMP_TIME_BINS: usize = 24;

/// Headline numbers: (heading, metric column, unit). The card's tiles and
/// the overview table's columns.
const KPIS: [(&str, &str, &str); 8] = [
    ("spikes", "n_spikes", ""),
    ("rate", "firing_rate", "Hz"),
    ("SNR", "amplitude_snr", ""),
    ("presence", "presence_ratio", ""),
    ("ISI viol.", "isi_violation_ratio", ""),
    ("RP contam.", "rp_contamination", ""),
    ("iso. dist.", "isolation_distance_sq", ""),
    ("L-ratio", "l_ratio", ""),
];

/// Render the plots for one cluster.
pub fn render_cluster_svgs<P: DataProvider>(
    session: &Session<P>,
    cluster: ClusterId,
) -> ClusterSvgs {
    let sr = session.provider.sample_rate().max(1.0);
    let window = (WINDOW_S * sr) as u32;
    let bin = ((BIN_S * sr) as u32).max(1);

    let waveform = session.primary_template(cluster).and_then(|t| {
        let wf = session.template_waveform(t)?;
        let (_, n_samples, n_channels) = session.template_shape();
        if n_samples == 0 || n_channels == 0 || wf.len() / n_channels < n_samples {
            return None;
        }
        let trace =
            |c: usize| -> Vec<f32> { (0..n_samples).map(|s| wf[s * n_channels + c]).collect() };
        let ptp = |c: usize| {
            let (lo, hi) = range(&trace(c));
            hi - lo
        };
        let mut chans: Vec<usize> = (0..n_channels).collect();
        chans.sort_by(|&a, &b| ptp(b).total_cmp(&ptp(a)));
        chans.truncate(WAVE_CHANNELS);
        let peak = chans[0];
        // Back to channel order, so neighbours sit next to each other.
        chans.sort_unstable();
        let traces: Vec<(usize, Vec<f32>)> = chans.into_iter().map(|c| (c, trace(c))).collect();
        Some(svg_waveforms(&traces, peak))
    });

    let spikes = session.spike_times(cluster);
    let isi = isi_histogram(spikes, window as u64, (window / bin) as usize);
    let acg = acg_or_compute(session, cluster, bin, window);
    let amps = session.spike_amplitudes(cluster);
    let span = session.unit_span(cluster);
    ClusterSvgs {
        waveform,
        isi: svg_bars(&isi, W, H),
        log_isi: svg_log_isi(spikes, sr, session.refractory().fixed_ms),
        acg: svg_bars(&acg, W, H),
        amplitude: (!amps.is_empty()).then(|| svg_dots(spikes, amps, span.clone())),
        amplitude_iqr: svg_amplitude_iqr(spikes, amps, &span),
    }
}

fn range(values: &[f32]) -> (f32, f32) {
    values
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)))
}

fn svg_log_isi(spikes: &[SampleIndex], sample_rate: f32, fixed_ms: f32) -> String {
    let samples_per_ms = f64::from(sample_rate) / 1000.0;
    let counts = isi_histogram_log(
        spikes,
        LOG_ISI_MIN_MS * samples_per_ms,
        LOG_ISI_MAX_MS * samples_per_ms,
        LOG_ISI_BINS,
    );
    let x = |ms: f64| {
        f64::from(W) * (ms / LOG_ISI_MIN_MS).ln() / (LOG_ISI_MAX_MS / LOG_ISI_MIN_MS).ln()
    };
    let mut out = format!(
        r#"<svg viewBox="0 0 {W} {}" role="img" aria-label="Log ISI in milliseconds and seconds">{}"#,
        H + 13.0,
        bar_elements(&counts, W, H),
    );
    let fixed_ms = f64::from(fixed_ms);
    if fixed_ms.is_finite() && fixed_ms > 0.0 {
        let x = x(fixed_ms.clamp(LOG_ISI_MIN_MS, LOG_ISI_MAX_MS));
        out += &format!(
            r#"<line x1="{x:.1}" x2="{x:.1}" y1="0" y2="{H}" stroke="red" stroke-dasharray="2 2"><title>Fixed refractory window: {fixed_ms:.3} ms</title></line>"#
        );
    }
    for (ms, label, anchor) in [
        (0.2, "0.2", "start"),
        (1.0, "1", "middle"),
        (10.0, "10", "middle"),
        (100.0, "100", "middle"),
        (1000.0, "1s", "middle"),
        (10_000.0, "10s", "end"),
    ] {
        out += &format!(
            r#"<text class="tick" x="{:.1}" y="{}" text-anchor="{anchor}">{label}</text>"#,
            x(ms),
            H + 11.0,
        );
    }
    out + "</svg>"
}

/// Template traces side by side on one shared amplitude scale, each with
/// a zero line and its channel number; `peak` is drawn in the accent.
pub fn svg_waveforms(traces: &[(usize, Vec<f32>)], peak: usize) -> String {
    let (lo, hi) = traces
        .iter()
        .map(|(_, t)| range(t))
        .fold((f32::MAX, f32::MIN), |(a, b), (lo, hi)| {
            (a.min(lo), b.max(hi))
        });
    let span = (hi - lo).max(f32::EPSILON);
    let slot = W / traces.len().max(1) as f32;
    let pad = slot * 0.08;
    let y = |v: f32| H - (v - lo) / span * H;
    let mut out = format!(
        r#"<svg viewBox="0 -4 {W} {}" role="img" aria-label="template waveform">"#,
        H + 18.0
    );
    for (i, (ch, t)) in traces.iter().enumerate() {
        let x0 = i as f32 * slot + pad;
        let n = t.len().max(2) as f32;
        let pts: Vec<String> = t
            .iter()
            .enumerate()
            .map(|(k, &v)| {
                format!(
                    "{:.1},{:.1}",
                    x0 + k as f32 / (n - 1.0) * (slot - 2.0 * pad),
                    y(v)
                )
            })
            .collect();
        let class = if *ch == peak { "wf peak" } else { "wf" };
        out += &format!(
            r#"<line class="zero" x1="{x0:.1}" x2="{:.1}" y1="{z:.1}" y2="{z:.1}"/><polyline class="{class}" points="{}"/><text x="{:.1}" y="{}" class="{class}">ch {ch}</text>"#,
            x0 + slot - 2.0 * pad,
            pts.join(" "),
            x0 + (slot - 2.0 * pad) / 2.0,
            H + 12.0,
            z = y(0.0),
        );
    }
    out + "</svg>"
}

/// Bars scaled to the tallest bin, left to right, on a baseline.
pub fn svg_bars(counts: &[u32], w: f32, h: f32) -> String {
    format!(
        r#"<svg viewBox="0 0 {w} {h}" role="img">{}</svg>"#,
        bar_elements(counts, w, h)
    )
}

fn bar_elements(counts: &[u32], w: f32, h: f32) -> String {
    let max = counts.iter().copied().max().unwrap_or(0).max(1) as f32;
    let n = counts.len().max(1) as f32;
    let bw = w / n;
    let bars: String = counts
        .iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(i, &c)| {
            let bh = c as f32 / max * h;
            format!(
                r#"<rect x="{:.1}" y="{:.1}" width="{:.2}" height="{:.1}"/>"#,
                i as f32 * bw,
                h - bh,
                (bw - 0.4).max(0.5),
                bh
            )
        })
        .collect();
    format!(r#"<g class="bars">{bars}</g><line class="axis" x1="0" x2="{w}" y1="{h}" y2="{h}"/>"#)
}

/// Amplitude against recording time as one dotted path (round caps on
/// zero-length segments), at most `MAX_DOTS` dots. y runs from zero to a
/// little above the drawn dots' 99th percentile, so a few outliers pin to
/// the top edge instead of flattening everything else.
pub fn svg_dots(
    times: &[SampleIndex],
    amps: &[f32],
    span: impl Into<sorrel_io::SessionSpan>,
) -> String {
    let span = span.into();
    let width = span.extent_samples().max(1) as f32;
    let n = times.len().min(amps.len());
    let stride = n.div_ceil(MAX_DOTS).max(1);
    let drawn: Vec<usize> = (0..n)
        .step_by(stride)
        .filter(|&i| span.source == sorrel_io::SpanSource::Trace || span.contains(times[i]))
        .collect();
    let mut sorted: Vec<f32> = drawn.iter().map(|&i| amps[i].abs()).collect();
    sorted.sort_unstable_by(f32::total_cmp);
    let top = sorted
        .get(sorted.len() * 99 / 100)
        .map_or(0.0, |&p| p * 1.15)
        .max(f32::EPSILON);
    let d: String = drawn
        .iter()
        .map(|&i| {
            format!(
                "M{:.1} {:.1}h0",
                (times[i].0 - span.start.0) as f32 / width * STRIP_W,
                STRIP_H - (amps[i].abs() / top).min(1.0) * (STRIP_H - 2.0) - 1.0
            )
        })
        .collect();
    format!(
        r#"<svg viewBox="0 0 {STRIP_W} {STRIP_H}" preserveAspectRatio="none" role="img" aria-label="amplitude over time"><line class="axis" x1="0" x2="{STRIP_W}" y1="{STRIP_H}" y2="{STRIP_H}"/><path class="dots" d="{d}"/></svg>"#
    )
}

/// Equal time bins on the unit's own clock, with a median trace and IQR ribbon.
fn svg_amplitude_iqr(times: &[SampleIndex], amps: &[f32], span: &SessionSpan) -> Option<String> {
    if times.len() != amps.len() {
        return None;
    }
    let width = span.extent_samples().max(1) as u128;
    let offset = |time: SampleIndex| u128::from(time.0.saturating_sub(span.start.0));
    let bin_coord = |time| (offset(time) * AMP_TIME_BINS as u128) as f64 / width as f64;
    let mut segments = Vec::new();
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    // Bin each recorded interval separately: even one bin may straddle a gap.
    for (start, end) in span.intervals() {
        let range = if span.source == sorrel_io::SpanSource::Trace && span.n_intervals() == 1 {
            // Preserve the edge bins for spikes outside a contiguous trace.
            0..times.len()
        } else {
            times.partition_point(|&t| t < start)..times.partition_point(|&t| t < end)
        };
        let mut bins: [Vec<f32>; AMP_TIME_BINS] = std::array::from_fn(|_| Vec::new());
        for (&time, &amp) in times[range.clone()].iter().zip(&amps[range]) {
            if amp.is_finite() {
                let bin =
                    (offset(time) * AMP_TIME_BINS as u128 / width).min((AMP_TIME_BINS - 1) as u128);
                bins[bin as usize].push(amp);
            }
        }
        let mut stats = Vec::new();
        for (i, b) in bins.iter_mut().enumerate().filter(|(_, b)| !b.is_empty()) {
            b.sort_unstable_by(f32::total_cmp);
            let quantile = |q: f64| {
                let pos = q * (b.len() - 1) as f64;
                let (lo, hi) = (
                    f64::from(b[pos as usize]),
                    f64::from(b[pos.ceil() as usize]),
                );
                lo + (hi - lo) * pos.fract()
            };
            let (q25, q50, q75) = (quantile(0.25), quantile(0.5), quantile(0.75));
            lo = lo.min(q25);
            hi = hi.max(q75);
            stats.push((i, q25, q50, q75));
        }
        if !stats.is_empty() {
            segments.push((bin_coord(start), bin_coord(end), stats));
        }
    }
    segments.first()?;
    let point = |bin: f64, a| {
        let level = if hi == lo { 0.5 } else { (a - lo) / (hi - lo) };
        format!(
            "{:.2},{:.1}",
            bin / AMP_TIME_BINS as f64 * f64::from(STRIP_W),
            4.0 + (1.0 - level) * f64::from(STRIP_H - 8.0)
        )
    };
    let mut svg = format!(
        r#"<svg viewBox="0 0 {STRIP_W} {STRIP_H}" preserveAspectRatio="none" role="img" aria-label="Binned amplitude median and interquartile range">"#
    );
    for (start, end, stats) in segments {
        for run in stats.chunk_by(|a, b| a.0 + 1 == b.0) {
            let (mut upper, mut median, mut lower) = (Vec::new(), Vec::new(), Vec::new());
            let mut push = |bin, q25, q50, q75| {
                upper.push(point(bin, q75));
                median.push(point(bin, q50));
                lower.push(point(bin, q25));
            };
            if let &[(i, q25, q50, q75)] = run {
                push((i as f64).max(start), q25, q50, q75);
                push((i as f64 + 1.0).min(end), q25, q50, q75);
            } else {
                for &(i, q25, q50, q75) in run {
                    let from = (i as f64).max(start);
                    let to = (i as f64 + 1.0).min(end);
                    push(f64::midpoint(from, to), q25, q50, q75);
                }
            }
            upper.extend(lower.into_iter().rev());
            let _ = write!(
                svg,
                r#"<polygon class="iqr" points="{}"/><polyline class="median" points="{}"/>"#,
                upper.join(" "),
                median.join(" "),
            );
        }
    }
    svg.push_str("</svg>");
    Some(svg)
}

/// A metric for display: `–` for missing, integers bare, otherwise about
/// three significant digits.
fn num(v: f32) -> String {
    if !v.is_finite() {
        return "–".into();
    }
    let a = v.abs();
    if v.fract() == 0.0 && a < 1e7 {
        return format!("{v:.0}");
    }
    let decimals = match a {
        a if a >= 100.0 => 0,
        a if a >= 10.0 => 1,
        a if a >= 1.0 => 2,
        a => ((-a.log10().floor()) as usize + 2).min(6),
    };
    format!("{v:.decimals$}")
}

struct Tag {
    text: String,
    review: bool,
}

struct Card {
    id: u32,
    label: &'static str,
    origin: String,
    tags: Vec<Tag>,
    /// `KPIS` values, in order.
    kpis: Vec<String>,
    /// Waveform landmarks under the waveform panel.
    shape: String,
    svgs: ClusterSvgs,
    /// Every `QcRow::METRICS` entry.
    metrics: Vec<(&'static str, String)>,
}

#[derive(Template)]
#[template(path = "report.html")]
struct Page {
    kpis: &'static [(&'static str, &'static str, &'static str)],
    /// Clusters per label, in first-seen order.
    counts: Vec<(&'static str, usize)>,
    sample_rate_khz: String,
    duration_min: String,
    cards: Vec<Card>,
    version: &'static str,
}

/// Build the whole document.
pub fn render_report<P, F>(
    session: &Session<P>,
    label_str: F,
    filter: &ReportFilter,
) -> Result<String>
where
    P: DataProvider,
    F: Fn(&P::Label) -> &'static str + Sync,
{
    let rows: Vec<QcRow> = collect_qc_rows(session)
        .into_iter()
        .filter(|r| r.n_spikes > 0)
        .filter(|r| {
            filter
                .clusters
                .as_ref()
                .is_none_or(|ids| ids.contains(&r.cluster_id))
        })
        .filter(|r| {
            filter.label.as_deref().is_none_or(|want| {
                session
                    .label(ClusterId(r.cluster_id))
                    .map(|l| label_str(&l) == want)
                    .unwrap_or(false)
            })
        })
        .filter(|r| {
            filter.tag.as_deref().is_none_or(|t| {
                session
                    .tags(ClusterId(r.cluster_id))
                    .iter()
                    .any(|x| x.starts_with(t))
            })
        })
        .collect();

    let cards: Vec<Card> = rows
        .par_iter()
        .map(|row| {
            let c = ClusterId(row.cluster_id);
            let get = |name: &str| row.get(name).unwrap_or(f32::NAN);
            Card {
                id: row.cluster_id,
                label: session
                    .label(c)
                    .map(|l| label_str(&l))
                    .unwrap_or("unsorted"),
                origin: session.origin_label(c),
                tags: session
                    .tags(c)
                    .iter()
                    .map(|t| Tag {
                        review: t.starts_with("review:"),
                        text: t.clone(),
                    })
                    .collect(),
                kpis: KPIS.iter().map(|(_, m, _)| num(get(m))).collect(),
                shape: [
                    ("trough→peak ", "trough_to_peak_ms", " ms"),
                    ("half-width ", "half_width_ms", " ms"),
                    ("", "n_active_channels", " active ch"),
                    ("y ", "y_um", " µm"),
                ]
                .iter()
                .filter(|(_, m, _)| get(m).is_finite())
                .map(|(pre, m, post)| format!("{pre}{}{post}", num(get(m))))
                .collect::<Vec<_>>()
                .join(" · "),
                svgs: render_cluster_svgs(session, c),
                metrics: QcRow::METRICS.iter().map(|&m| (m, num(get(m)))).collect(),
            }
        })
        .collect();

    let mut counts: Vec<(&'static str, usize)> = Vec::new();
    for card in &cards {
        match counts.iter_mut().find(|(l, _)| *l == card.label) {
            Some((_, n)) => *n += 1,
            None => counts.push((card.label, 1)),
        }
    }
    let sr = session.provider.sample_rate().max(1.0);
    Page {
        kpis: &KPIS,
        counts,
        sample_rate_khz: num(sr / 1000.0),
        duration_min: num(session.provider.session_span().duration_samples() as f32 / sr / 60.0),
        cards,
        version: env!("CARGO_PKG_VERSION"),
    }
    .render()
    .context("render report template")
}

/// Write the report to `path`; returns the number of cluster cards.
pub fn write_report<P, F>(
    session: &Session<P>,
    path: &Path,
    label_str: F,
    filter: &ReportFilter,
) -> Result<usize>
where
    P: DataProvider,
    F: Fn(&P::Label) -> &'static str + Sync,
{
    let html = render_report(session, label_str, filter)?;
    std::fs::write(path, &html).with_context(|| format!("write {}", path.display()))?;
    Ok(html.matches("<section id=\"cluster-").count())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::Journal;
    use crate::qc_export::tests::MockProvider;

    #[test]
    fn askama_report_renders_waveforms_log_isi_and_amplitude_iqr() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = crate::feature_subspace::tests::physical_test_session(dir.path());
        s.seed_template_waveforms();
        s.set_refractory(sorrel_compute::RefractoryParams {
            fixed_ms: 2.25,
            ..Default::default()
        })
        .unwrap();
        let plots = render_cluster_svgs(&s, ClusterId(0));
        let waveform = plots.waveform.unwrap();
        for ch in 0..3 {
            assert!(waveform.contains(&format!(">ch {ch}</text>")));
        }
        assert_eq!(waveform.matches("<polyline").count(), 3);
        assert!(plots.log_isi.contains("stroke=\"red\""));
        assert!(plots.log_isi.contains("Fixed refractory window: 2.250 ms"));
        for label in ["0.2", "1", "10", "100", "1s", "10s"] {
            assert!(plots.log_isi.contains(&format!(">{label}</text>")));
        }
        let amplitudes = plots.amplitude_iqr.unwrap();
        assert!(amplitudes.contains("<polygon class=\"iqr\""));
        assert!(amplitudes.contains("<polyline class=\"median\""));
        for id in [1, 2] {
            assert!(render_cluster_svgs(&s, ClusterId(id))
                .amplitude_iqr
                .is_none());
        }
        let html = render_report(&s, |_| "unsorted", &ReportFilter::default()).unwrap();
        assert!(html.contains("<b>Log-ISI</b>"));
        assert!(html.contains(&amplitudes));
        assert!(html.contains("<path class=\"dots\""));
    }

    #[test]
    fn amplitude_iqr_uses_the_unit_clock_and_excludes_recording_gaps() {
        let span = SessionSpan::from_intervals(
            vec![
                (SampleIndex(100), SampleIndex(104)),
                (SampleIndex(120), SampleIndex(124)),
            ],
            sorrel_io::SpanSource::Declared("test"),
        );
        let svg = svg_amplitude_iqr(
            &[SampleIndex(100), SampleIndex(110), SampleIndex(123)],
            &[1.0, 1000.0, 3.0],
            &span,
        )
        .unwrap();
        assert_eq!(svg.matches(r#"<polygon class="iqr""#).count(), 2);
        assert_eq!(svg.matches(r#"<polyline class="median""#).count(), 2);
        assert!(svg.contains(r#"class="median" points="0.00,60.0 40.00,60.0""#));
        assert!(svg.contains(r#"class="median" points="920.00,4.0 960.00,4.0""#));
    }

    #[test]
    fn amplitude_iqr_does_not_bridge_a_gap_within_a_single_bin() {
        let span = SessionSpan::from_intervals(
            vec![
                (SampleIndex(0), SampleIndex(1)),
                (SampleIndex(2), SampleIndex(240)),
            ],
            sorrel_io::SpanSource::Declared("test"),
        );
        let svg = svg_amplitude_iqr(
            &[
                SampleIndex(0),
                SampleIndex(1),
                SampleIndex(2),
                SampleIndex(30),
            ],
            &[1.0, 1000.0, 3.0, 2.0],
            &span,
        )
        .unwrap();
        assert_eq!(svg.matches(r#"<polygon class="iqr""#).count(), 3);
        assert_eq!(svg.matches(r#"<polyline class="median""#).count(), 3);
        assert!(svg.contains(r#"class="median" points="0.00,60.0 4.00,60.0""#));
        assert!(svg.contains(r#"class="median" points="8.00,4.0 40.00,4.0""#));

        // Sub-tenth-pixel gap on a 30,000-sample span still serializes with distinct x edges.
        let fine = SessionSpan::from_intervals(
            vec![
                (SampleIndex(0), SampleIndex(300)),
                (SampleIndex(301), SampleIndex(30_000)),
            ],
            sorrel_io::SpanSource::Declared("test"),
        );
        let fine_svg =
            svg_amplitude_iqr(&[SampleIndex(100), SampleIndex(500)], &[5.0, 5.0], &fine).unwrap();
        assert!(fine_svg.contains(r#"class="median" points="0.00,32.0 9.60,32.0""#));
        assert!(fine_svg.contains(r#"class="median" points="9.63,32.0 40.00,32.0""#));
    }

    #[test]
    fn report_distinguishes_confirmed_srp_and_unavailable_metrics() {
        let dir = tempfile::tempdir().unwrap();
        let s = Session::new(
            MockProvider {
                spikes: vec![
                    (0..100).map(|i| SampleIndex(i * 10)).collect(),
                    vec![SampleIndex(1)],
                ],
                amps: vec![vec![], vec![]],
                n_samples: SampleIndex(1000),
            },
            Journal::open(&dir.path().join("j")).unwrap(),
        );
        let html = render_report(&s, |_| "unsorted", &ReportFilter::default()).unwrap();
        let card = |id| {
            html.split_once(&format!("<section id=\"cluster-{id}\""))
                .unwrap()
                .1
                .split_once("</section>")
                .unwrap()
                .0
        };
        let confirmed = card(0);
        assert!(
            confirmed.contains("<th>srp_contamination</th><td>0.0150</td>"),
            "{confirmed}"
        );
        assert!(confirmed.contains("<th>srp_rp_ms</th><td>10</td>"));
        assert!(confirmed.contains("<th>rpv_post_lo95</th><td>0</td>"));
        let sparse = card(1);
        for name in [
            "srp_contamination",
            "srp_rp_ms",
            "rpv_post_mean",
            "rpv_post_lo95",
            "rpv_post_hi95",
        ] {
            assert!(
                sparse.contains(&format!("<th>{name}</th><td>–</td>")),
                "{name}"
            );
        }
    }

    #[test]
    fn report_has_one_section_per_cluster_with_svg_plots_and_honours_filters() {
        let prov = MockProvider {
            spikes: vec![
                vec![SampleIndex(10), SampleIndex(30), SampleIndex(150)],
                vec![SampleIndex(20), SampleIndex(200)],
                vec![],
            ],
            amps: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0], vec![]],
            n_samples: SampleIndex(1000),
        };
        let dir = tempfile::tempdir().unwrap();
        let journal = Journal::open(&dir.path().join("j.journal")).unwrap();
        let mut s = Session::new(prov, journal);
        s.seed_amplitudes();
        s.dispatch(crate::CurationCommand::Tag {
            cluster: ClusterId(0),
            tags: vec!["review:<odd>".into()],
        })
        .unwrap();
        let label = |l: &u8| if *l == 1 { "good" } else { "unsorted" };

        let html = render_report(&s, label, &ReportFilter::default()).unwrap();
        assert_eq!(
            html.matches("<section id=\"cluster-").count(),
            2,
            "empty cluster skipped"
        );
        assert!(html.contains("<svg"));
        assert!(html.contains("class=\"dots\""), "amplitude strip drawn");
        assert!(html.contains("no templates"));
        assert!(html.contains("<th>isi_violation_ratio</th>"));
        assert!(html.contains("href=\"#cluster-1\""), "overview links cards");
        assert!(
            html.contains("<b>review:&#60;odd&#62;</b>")
                || html.contains("<b>review:&lt;odd&gt;</b>"),
            "review tags highlighted and escaped"
        );
        assert!(html.contains(&format!(
            "<span class=\"origin\">{}</span>",
            s.origin_label(ClusterId(0))
        )));

        let only_one = ReportFilter {
            clusters: Some(vec![1]),
            label: None,
            tag: None,
        };
        let html = render_report(&s, label, &only_one).unwrap();
        assert!(html.contains("id=\"cluster-1\"") && !html.contains("id=\"cluster-0\""));

        let none_good = ReportFilter {
            clusters: None,
            label: Some("good".into()),
            tag: None,
        };
        assert_eq!(
            render_report(&s, label, &none_good)
                .unwrap()
                .matches("<section")
                .count(),
            0
        );
    }

    #[test]
    fn plot_helpers() {
        assert!(svg_bars(&[0, 3, 1], 30.0, 10.0).contains("<rect"));
        let wf = svg_waveforms(&[(2, vec![-1.0, 0.0, 1.0]), (5, vec![0.0, 0.5, 0.0])], 2);
        assert_eq!(wf.matches("<polyline").count(), 2);
        assert!(wf.contains(r#"class="wf peak">ch 2<"#) && wf.contains(r#"class="wf">ch 5<"#));
        // At 1 kHz, 1 ms and 1 s hit their geometric bins; 100 s is out of range.
        let t = [0, 1, 1001, 101_001].map(SampleIndex);
        let log = svg_log_isi(&t, 1000.0, 1.5);
        assert_eq!(log.matches("<rect").count(), 2);
        assert!(log.contains(r#"<rect x="33.0""#) && log.contains(r#"<rect x="186.0""#));
        let many: Vec<SampleIndex> = (0..10_000).map(SampleIndex).collect();
        let dots = svg_dots(&many, &vec![1.0; 10_000], 10_000);
        assert_eq!(dots.matches('M').count(), MAX_DOTS);
        assert_eq!(
            [
                num(f32::NAN),
                num(2523.0),
                num(0.4567),
                num(17.83),
                num(0.0336)
            ],
            ["–", "2523", "0.457", "17.8", "0.0336"]
        );
    }
}
