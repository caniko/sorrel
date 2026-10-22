//! Self-contained HTML report (`sorrel report`): one card per cluster with
//! its QC metrics, template waveform on the peak channel (when templates
//! are loaded), ISI histogram and autocorrelogram as inline SVG. No
//! external assets, so the file can be mailed or dropped into a lab wiki.
//!
//! Snippets from the raw trace are deliberately not extracted here: that
//! is a per-spike disk walk the GUI does lazily per selection, and a
//! report over a thousand clusters would take minutes for a picture the
//! template already gives.

use crate::cache::correlograms::acg_or_compute;
use crate::qc_export::{collect_qc_rows, QcRow};
use crate::session::Session;
use anyhow::{Context, Result};
use rayon::prelude::*;
use sorrel_compute::isi_histogram;
use sorrel_io::{ClusterId, DataProvider};
use std::path::Path;

/// Which clusters to include.
#[derive(Clone, Debug, Default)]
pub struct ReportFilter {
    /// Explicit cluster ids; `None` = all non-empty clusters.
    pub clusters: Option<Vec<u32>>,
    /// Only clusters whose label string equals this (`"good"`, ...).
    pub label: Option<String>,
}

/// The three plots of one cluster card, as SVG fragments.
pub struct ClusterSvgs {
    pub waveform: Option<String>,
    pub isi: String,
    pub acg: String,
}

const W: f32 = 240.0;
const H: f32 = 90.0;
/// ISI and ACG windows: 50 ms at 1 ms bins, matching the GUI defaults.
const WINDOW_S: f32 = 0.050;
const BIN_S: f32 = 0.001;

/// Render the plots for one cluster. Also used by `serve` for `render`.
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
        // Peak channel = largest peak-to-peak.
        let peak = (0..n_channels)
            .max_by(|&a, &b| {
                let ptp = |c: usize| {
                    let col = (0..n_samples).map(|s| wf[s * n_channels + c]);
                    let (lo, hi) =
                        col.fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));
                    hi - lo
                };
                ptp(a).total_cmp(&ptp(b))
            })
            .unwrap_or(0);
        let trace: Vec<f32> = (0..n_samples).map(|s| wf[s * n_channels + peak]).collect();
        Some(svg_polyline(&trace, W, H))
    });

    let spikes = session.spike_times(cluster);
    let isi = svg_bars(
        &isi_histogram(spikes, window as u64, (window / bin) as usize),
        W,
        H,
    );
    let acg = svg_bars(&acg_or_compute(session, cluster, bin, window), W, H);
    ClusterSvgs { waveform, isi, acg }
}

/// A polyline scaled to fit `w × h`, with a zero line.
pub fn svg_polyline(values: &[f32], w: f32, h: f32) -> String {
    let (lo, hi) = values
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let span = (hi - lo).max(f32::EPSILON);
    let n = values.len().max(2) as f32;
    let pts: Vec<String> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let x = i as f32 / (n - 1.0) * w;
            let y = h - (v - lo) / span * h;
            format!("{x:.1},{y:.1}")
        })
        .collect();
    let zero_y = h - (0.0 - lo) / span * h;
    format!(
        r#"<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}"><line x1="0" y1="{zero_y:.1}" x2="{w}" y2="{zero_y:.1}" stroke="{grey}" stroke-dasharray="2 2"/><polyline fill="none" stroke="{green}" stroke-width="1.5" points="{}"/></svg>"#,
        pts.join(" "),
        grey = "#999",
        green = "#2a7",
    )
}

/// Bars scaled to the tallest bin, left to right.
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
                bw.max(0.5),
                bh
            )
        })
        .collect();
    format!(
        r#"<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" fill="{blue}">{bars}</svg>"#,
        blue = "#468"
    )
}

/// Build the whole document.
pub fn render_report<P, F>(session: &Session<P>, label_str: F, filter: &ReportFilter) -> String
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
        .collect();

    let cards: Vec<String> = rows
        .par_iter()
        .map(|row| {
            let c = ClusterId(row.cluster_id);
            let label = session.label(c).map(|l| label_str(&l)).unwrap_or("unsorted");
            let svgs = render_cluster_svgs(session, c);
            let metrics: String = QcRow::METRICS
                .iter()
                .map(|name| {
                    let v = row.get(name).unwrap_or(f32::NAN);
                    let cell = if v.is_nan() {
                        "–".to_string()
                    } else {
                        format!("{v:.4}")
                    };
                    format!("<tr><th>{name}</th><td>{cell}</td></tr>")
                })
                .collect();
            format!(
                r#"<section id="cluster-{id}" class="{label}"><h2>cluster {id} <span class="label">{label}</span></h2><div class="row"><table>{metrics}</table><div class="plots"><figure>{wf}<figcaption>template, peak channel</figcaption></figure><figure>{isi}<figcaption>ISI 0–50 ms</figcaption></figure><figure>{acg}<figcaption>ACG ±50 ms</figcaption></figure></div></div></section>"#,
                id = row.cluster_id,
                wf = svgs.waveform.unwrap_or_else(|| "<p class=\"none\">no templates</p>".into()),
                isi = svgs.isi,
                acg = svgs.acg,
            )
        })
        .collect();

    format!(
        r#"<!doctype html><html lang="en"><head><meta charset="utf-8"><title>sorrel report</title><style>
body{{font:14px system-ui,sans-serif;margin:1.5rem;color:#222;background:#fff}}
section{{border:1px solid #ddd;border-radius:6px;padding:.75rem 1rem;margin-bottom:1rem}}
h2{{margin:0 0 .5rem;font-size:1.1rem}}.label{{font-weight:normal;color:#666}}
section.good h2 .label{{color:#2a7}}section.mua h2 .label{{color:#c80}}section.noise h2 .label{{color:#c33}}
.row{{display:flex;gap:1.5rem;flex-wrap:wrap}}table{{border-collapse:collapse;font-size:12px}}
th{{text-align:left;padding:1px 8px 1px 0;font-weight:normal;color:#555}}td{{font-variant-numeric:tabular-nums}}
.plots{{display:flex;gap:1rem;flex-wrap:wrap}}figure{{margin:0}}figcaption{{font-size:11px;color:#666;text-align:center}}
.none{{color:#999;font-size:12px}}
</style></head><body><h1>sorrel report</h1><p>{n} clusters</p>{cards}</body></html>"#,
        n = rows.len(),
        cards = cards.join("\n"),
    )
}

/// Write the report to `path`.
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
    let html = render_report(session, label_str, filter);
    std::fs::write(path, &html).with_context(|| format!("write {}", path.display()))?;
    Ok(html.matches("<section id=\"cluster-").count())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::Journal;
    use crate::qc_export::tests::MockProvider;
    use sorrel_io::SampleIndex;

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
        let label = |l: &u8| if *l == 1 { "good" } else { "unsorted" };

        let html = render_report(&s, label, &ReportFilter::default());
        assert_eq!(
            html.matches("<section id=\"cluster-").count(),
            2,
            "empty cluster skipped"
        );
        assert!(html.contains("<svg"));
        assert!(html.contains("no templates"));
        assert!(html.contains("<th>isi_violation_ratio</th>"));

        let only_one = ReportFilter {
            clusters: Some(vec![1]),
            label: None,
        };
        let html = render_report(&s, label, &only_one);
        assert!(html.contains("id=\"cluster-1\"") && !html.contains("id=\"cluster-0\""));

        let none_good = ReportFilter {
            clusters: None,
            label: Some("good".into()),
        };
        assert_eq!(
            render_report(&s, label, &none_good)
                .matches("<section")
                .count(),
            0
        );

        assert!(svg_bars(&[0, 3, 1], 30.0, 10.0).contains("<rect"));
        assert!(svg_polyline(&[-1.0, 0.0, 1.0], 30.0, 10.0).contains("points=\"0.0,10.0"));
    }
}
