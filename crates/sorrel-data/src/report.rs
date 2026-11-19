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
use sorrel_compute::isi_histogram;
use sorrel_io::{ClusterId, DataProvider, SampleIndex};
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
}

const W: f32 = 240.0;
const H: f32 = 80.0;
/// ISI and ACG windows: 50 ms at 1 ms bins, matching the GUI defaults.
const WINDOW_S: f32 = 0.050;
const BIN_S: f32 = 0.001;
/// Channels in the waveform panel: the strongest few, a tetrode's worth.
const WAVE_CHANNELS: usize = 4;
/// Log-ISI axis: 1 ms to 1000 s, ten bins per decade.
const LOG_ISI_MIN: f64 = -3.0;
const LOG_ISI_MAX: f64 = 3.0;
const LOG_ISI_BINS: usize = 60;
/// Amplitude strip size and its dot budget (evenly strided beyond it).
const STRIP_W: f32 = 960.0;
const STRIP_H: f32 = 64.0;
const MAX_DOTS: usize = 2000;

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
        if n_samples == 0 || n_channels == 0 {
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
    ClusterSvgs {
        waveform,
        isi: svg_bars(&isi, W, H),
        log_isi: svg_bars(&log_isi_histogram(spikes, sr), W, H),
        acg: svg_bars(&acg, W, H),
        amplitude: (!amps.is_empty())
            .then(|| svg_dots(spikes, amps, session.provider.n_samples().0.max(1))),
    }
}

fn range(values: &[f32]) -> (f32, f32) {
    values
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)))
}

/// ISI counts over log10(seconds) in `[LOG_ISI_MIN, LOG_ISI_MAX)`; the
/// bimodal shape (bursts vs tonic firing) the linear 50 ms window hides.
fn log_isi_histogram(spikes: &[SampleIndex], sample_rate: f32) -> Vec<u32> {
    let mut h = vec![0u32; LOG_ISI_BINS];
    let per_bin = (LOG_ISI_MAX - LOG_ISI_MIN) / LOG_ISI_BINS as f64;
    for w in spikes.windows(2) {
        let dt = w[1].0.saturating_sub(w[0].0);
        if dt == 0 {
            continue;
        }
        let x = (dt as f64 / sample_rate as f64).log10();
        if (LOG_ISI_MIN..LOG_ISI_MAX).contains(&x) {
            h[(((x - LOG_ISI_MIN) / per_bin) as usize).min(LOG_ISI_BINS - 1)] += 1;
        }
    }
    h
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
    format!(
        r#"<svg viewBox="0 0 {w} {h}" role="img"><g class="bars">{bars}</g><line class="axis" x1="0" x2="{w}" y1="{h}" y2="{h}"/></svg>"#
    )
}

/// Amplitude against recording time as one dotted path (round caps on
/// zero-length segments), at most `MAX_DOTS` dots. y runs from zero to a
/// little above the drawn dots' 99th percentile, so a few outliers pin to
/// the top edge instead of flattening everything else.
pub fn svg_dots(times: &[SampleIndex], amps: &[f32], n_samples: u64) -> String {
    let n = times.len().min(amps.len());
    let stride = n.div_ceil(MAX_DOTS).max(1);
    let drawn: Vec<usize> = (0..n).step_by(stride).collect();
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
                times[i].0 as f32 / n_samples as f32 * STRIP_W,
                STRIP_H - (amps[i].abs() / top).min(1.0) * (STRIP_H - 2.0) - 1.0
            )
        })
        .collect();
    format!(
        r#"<svg viewBox="0 0 {STRIP_W} {STRIP_H}" preserveAspectRatio="none" role="img" aria-label="amplitude over time"><line class="axis" x1="0" x2="{STRIP_W}" y1="{STRIP_H}" y2="{STRIP_H}"/><path class="dots" d="{d}"/></svg>"#
    )
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
        duration_min: num(session.provider.n_samples().0 as f32 / sr / 60.0),
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
        // 1 ms, 1 s and 100 s at 1 kHz land in the 1 ms, 1 s and 100 s bins.
        let t = [0, 1, 1001, 101_001].map(SampleIndex);
        let h = log_isi_histogram(&t, 1000.0);
        assert_eq!((h[0], h[30], h[50], h.iter().sum::<u32>()), (1, 1, 1, 3));
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
