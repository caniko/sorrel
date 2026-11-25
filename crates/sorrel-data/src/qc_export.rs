//! QC export — write every cluster's quality metrics to a tab-separated
//! file (SpikeInterface-compatible columns) and/or to a JSON snapshot.
//!
//! Two output formats:
//! * `cluster_qc.tsv`: one row per cluster, headed by phy-style column
//!   names. Compatible with `pandas.read_csv(..., sep='\t')` and
//!   `spikeinterface.curation.read_qm_csv`.
//! * `cluster_qc.json`: a versioned object with session-span and trace provenance metadata and an
//!   array of rows with the same fields as the TSV.
//!
//! Both run the rayon-parallel quality + isolation pipeline so even 1k+
//! cluster sessions export in a few seconds.

use crate::cache::correlograms::acg_or_compute;
use crate::quality_ext::cluster_quality_with;
use crate::refractory::RefractoryWindows;
use crate::session::Session;
use anyhow::{Context, Result};
use sorrel_compute::epochs::{stable_epochs, Epoch};
use sorrel_io::{ClusterId, DataProvider, SpanSource};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// One cluster's exported row. Field names mirror SpikeInterface where
/// possible so downstream pipelines can read this without remapping.
#[derive(Clone, Debug)]
pub struct QcRow {
    pub cluster_id: u32,
    pub n_spikes: u32,
    pub mean_amplitude: f32,
    pub firing_rate_hz: f32,
    pub presence_ratio: f32,
    pub isi_violations: u32,
    /// Violations per second of the unit's own first-to-last spike span.
    pub isi_violation_ratio: f32,
    pub amplitude_cutoff: f32,
    pub amplitude_snr: f32,
    pub drift_correlation: f32,
    pub longest_silent_gap: f32,
    pub composite_quality: f32,
    /// `NaN` if no PC features / too few spikes.
    pub isolation_distance_sq: f32,
    pub l_ratio: f32,
    pub nn_isolation: f32,
    pub d_prime: f32,
    pub silhouette: f32,
    // Waveform landmarks of the primary template on its peak channel
    // (`NaN` without seeded templates). See `sorrel_compute::shape`.
    pub trough_to_peak_ms: f32,
    pub half_width_ms: f32,
    pub peak_trough_ratio: f32,
    pub repolarisation_slope: f32,
    pub n_extrema: f32,
    // Footprint on the probe; `NaN` unless channel positions were passed.
    pub x_um: f32,
    pub y_um: f32,
    pub footprint_decay_um: f32,
    pub footprint_r80_um: f32,
    pub n_active_channels: f32,
    pub propagation_ms_per_100um: f32,
    /// Mean ACG count at 2-10 ms over 30-50 ms lags.
    pub burst_index: f32,
    /// Longest refractory window (ms) at the minimum sliding contamination.
    pub rp_estimate_ms: f32,
    /// That minimum contamination fraction.
    pub rp_contamination: f32,
    /// Hartigan dip of the amplitudes (0 unimodal, up to 0.25).
    pub amplitude_dip: f32,
    /// Amplitude change per second (least squares).
    pub drift_slope: f32,
    /// Minimum contamination fraction confirmed by sliding-RP, or `NaN`.
    pub srp_contamination: f32,
    /// Highest-confidence window at that fraction (ms, first on ties), or `NaN`.
    pub srp_rp_ms: f32,
    /// Fixed-window contamination posterior mean and shortest 95% interval.
    pub rpv_post_mean: f32,
    pub rpv_post_lo95: f32,
    pub rpv_post_hi95: f32,
    /// Number of fitted stable epochs; `NaN` if not computed.
    pub n_epochs: f32,
    /// Lowest / highest magnitude of the epoch mean amplitudes; `NaN` unless
    /// every epoch has a fitted amplitude.
    pub epoch_amp_ratio: f32,
    /// Lowest / highest fitted firing rate.
    pub epoch_rate_ratio: f32,
}

impl Default for QcRow {
    /// Counts zero, every measurement `NaN` (nothing computed).
    fn default() -> Self {
        let n = f32::NAN;
        Self {
            cluster_id: 0,
            n_spikes: 0,
            mean_amplitude: n,
            firing_rate_hz: n,
            presence_ratio: n,
            isi_violations: 0,
            isi_violation_ratio: n,
            amplitude_cutoff: n,
            amplitude_snr: n,
            drift_correlation: n,
            longest_silent_gap: n,
            composite_quality: n,
            isolation_distance_sq: n,
            l_ratio: n,
            nn_isolation: n,
            d_prime: n,
            silhouette: n,
            trough_to_peak_ms: n,
            half_width_ms: n,
            peak_trough_ratio: n,
            repolarisation_slope: n,
            n_extrema: n,
            x_um: n,
            y_um: n,
            footprint_decay_um: n,
            footprint_r80_um: n,
            n_active_channels: n,
            propagation_ms_per_100um: n,
            burst_index: n,
            rp_estimate_ms: n,
            rp_contamination: n,
            amplitude_dip: n,
            drift_slope: n,
            srp_contamination: n,
            srp_rp_ms: n,
            rpv_post_mean: n,
            rpv_post_lo95: n,
            rpv_post_hi95: n,
            n_epochs: n,
            epoch_amp_ratio: n,
            epoch_rate_ratio: n,
        }
    }
}

impl QcRow {
    /// Every metric column name, in TSV order. The single source of truth
    /// for names used by `get`, the rule engine, and the exports.
    pub const METRICS: [&'static str; 40] = [
        "n_spikes",
        "firing_rate",
        "mean_amplitude",
        "presence_ratio",
        "isi_violations",
        "isi_violation_ratio",
        "amplitude_cutoff",
        "amplitude_snr",
        "drift_correlation",
        "longest_silent_gap",
        "composite_quality",
        "isolation_distance_sq",
        "l_ratio",
        "nn_isolation",
        "d_prime",
        "silhouette",
        "trough_to_peak_ms",
        "half_width_ms",
        "peak_trough_ratio",
        "repolarisation_slope",
        "n_extrema",
        "x_um",
        "y_um",
        "footprint_decay_um",
        "footprint_r80_um",
        "n_active_channels",
        "propagation_ms_per_100um",
        "burst_index",
        "rp_estimate_ms",
        "rp_contamination",
        "amplitude_dip",
        "drift_slope",
        "srp_contamination",
        "srp_rp_ms",
        "rpv_post_mean",
        "rpv_post_lo95",
        "rpv_post_hi95",
        "n_epochs",
        "epoch_amp_ratio",
        "epoch_rate_ratio",
    ];

    /// `{"cluster_id": .., "<metric>": number | null, ...}` — NaN becomes
    /// `null` so consumers never meet a non-JSON number.
    pub fn to_json(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        m.insert("cluster_id".into(), self.cluster_id.into());
        for name in Self::METRICS {
            let v = self.get(name).unwrap_or(f32::NAN);
            m.insert(
                name.to_string(),
                if v.is_nan() {
                    serde_json::Value::Null
                } else {
                    v.into()
                },
            );
        }
        serde_json::Value::Object(m)
    }

    /// Look a metric up by its column name; `None` for unknown names.
    pub fn get(&self, name: &str) -> Option<f32> {
        Some(match name {
            "n_spikes" => self.n_spikes as f32,
            "firing_rate" => self.firing_rate_hz,
            "mean_amplitude" => self.mean_amplitude,
            "presence_ratio" => self.presence_ratio,
            "isi_violations" => self.isi_violations as f32,
            "isi_violation_ratio" => self.isi_violation_ratio,
            "amplitude_cutoff" => self.amplitude_cutoff,
            "amplitude_snr" => self.amplitude_snr,
            "drift_correlation" => self.drift_correlation,
            "longest_silent_gap" => self.longest_silent_gap,
            "composite_quality" => self.composite_quality,
            "isolation_distance_sq" => self.isolation_distance_sq,
            "l_ratio" => self.l_ratio,
            "nn_isolation" => self.nn_isolation,
            "d_prime" => self.d_prime,
            "silhouette" => self.silhouette,
            "trough_to_peak_ms" => self.trough_to_peak_ms,
            "half_width_ms" => self.half_width_ms,
            "peak_trough_ratio" => self.peak_trough_ratio,
            "repolarisation_slope" => self.repolarisation_slope,
            "n_extrema" => self.n_extrema,
            "x_um" => self.x_um,
            "y_um" => self.y_um,
            "footprint_decay_um" => self.footprint_decay_um,
            "footprint_r80_um" => self.footprint_r80_um,
            "n_active_channels" => self.n_active_channels,
            "propagation_ms_per_100um" => self.propagation_ms_per_100um,
            "burst_index" => self.burst_index,
            "rp_estimate_ms" => self.rp_estimate_ms,
            "rp_contamination" => self.rp_contamination,
            "amplitude_dip" => self.amplitude_dip,
            "drift_slope" => self.drift_slope,
            "srp_contamination" => self.srp_contamination,
            "srp_rp_ms" => self.srp_rp_ms,
            "rpv_post_mean" => self.rpv_post_mean,
            "rpv_post_lo95" => self.rpv_post_lo95,
            "rpv_post_hi95" => self.rpv_post_hi95,
            "n_epochs" => self.n_epochs,
            "epoch_amp_ratio" => self.epoch_amp_ratio,
            "epoch_rate_ratio" => self.epoch_rate_ratio,
            _ => return None,
        })
    }

    fn tsv_header() -> String {
        format!("cluster_id\t{}\n", Self::METRICS.join("\t"))
    }

    fn export_value(&self, name: &str, missing: &str) -> String {
        match name {
            // Do not round integer counts through the f32 metric lookup.
            "n_spikes" => self.n_spikes.to_string(),
            "isi_violations" => self.isi_violations.to_string(),
            _ => match self.get(name).filter(|v| v.is_finite()) {
                Some(value) => num(value),
                None => missing.to_owned(),
            },
        }
    }

    fn write_tsv(&self, w: &mut impl Write) -> std::io::Result<()> {
        write!(w, "{}", self.cluster_id)?;
        for name in Self::METRICS {
            write!(w, "\t{}", self.export_value(name, ""))?;
        }
        writeln!(w)
    }

    fn write_json(&self, w: &mut impl Write, leading_comma: bool) -> std::io::Result<()> {
        let prefix = if leading_comma { ",\n" } else { "\n" };
        write!(w, r#"{prefix}  {{"cluster_id": {}"#, self.cluster_id)?;
        for name in Self::METRICS {
            write!(w, r#", "{name}": {}"#, self.export_value(name, "null"))?;
        }
        write!(w, "}}")
    }
}

/// Six decimals, except tiny non-zero values (posterior bounds of long clean
/// units) print in full so they do not round to 0.
fn num(v: f32) -> String {
    match v != 0.0 && v.abs() < 1e-3 {
        true => format!("{v}"),
        false => format!("{v:.6}"),
    }
}

/// [`collect_qc_rows_with`] without probe geometry: footprint columns are `NaN`.
pub fn collect_qc_rows<P: DataProvider>(session: &Session<P>) -> Vec<QcRow> {
    collect_qc_rows_with(session, None)
}

/// Compute QC rows for every cluster in the session. `positions` (one
/// `(x, y)` µm row per channel, e.g. `HasGeometry::channel_positions`)
/// enables the footprint columns.
pub fn collect_qc_rows_with<P: DataProvider>(
    session: &Session<P>,
    positions: Option<&[[f32; 2]]>,
) -> Vec<QcRow> {
    collect_qc_rows_with_windows(
        session,
        positions,
        RefractoryWindows::Session(session.refractory()),
    )
}

pub(crate) fn collect_qc_rows_with_windows<P: DataProvider>(
    session: &Session<P>,
    positions: Option<&[[f32; 2]]>,
    windows: RefractoryWindows<'_>,
) -> Vec<QcRow> {
    use sorrel_compute::{
        amplitude_drift_slope, burst_index, footprint, hartigan_dip, isi_violation_rate,
        isi_violations, mean_amplitude, peak_channel, refractory_evidence, waveform_landmarks,
    };
    let sample_rate = session.provider.sample_rate();
    let sr = sample_rate.max(1.0);
    let span = session.provider.session_span();
    let total_samples = span.duration_samples();
    let duration_rate = if span.source == SpanSource::Trace {
        sr
    } else {
        sample_rate
    };
    let total_seconds = total_samples as f32 / duration_rate;
    let refractory_samples = windows.fixed_samples(sr);
    let (_, tpl_samples, tpl_channels) = session.template_shape();
    // ACG at 1 ms bins / 50 ms, the report's (and cache's) window.
    let acg_window = (0.050 * sr) as u32;
    let acg_bin = ((0.001 * sr) as u32).max(1);

    let n = session.n_clusters();
    let mut rows = Vec::with_capacity(n as usize);
    for c in (0..n).map(ClusterId) {
        let times = session.spike_times(c);
        let exposed = span.metric_spikes(times);
        let amps = session.spike_amplitudes(c);
        let q = cluster_quality_with(session, c, refractory_samples);
        let mean_amp = if amps.is_empty() {
            f32::NAN
        } else {
            mean_amplitude(amps)
        };
        let firing_rate = if total_seconds > 0.0 {
            exposed.len() as f32 / total_seconds
        } else {
            f32::NAN
        };
        let isi_v = isi_violations(times, refractory_samples) as u32;
        let isi_ratio = if !times.is_empty() {
            isi_violation_rate(times, refractory_samples, sr)
        } else {
            0.0
        };
        let (iso2, l_ratio, nn, d_prime, silhouette) = match q.isolation {
            Some(iso) => (
                iso.isolation_distance_sq,
                iso.l_ratio,
                iso.nn_isolation,
                iso.d_prime,
                iso.silhouette,
            ),
            None => (f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN),
        };
        let tpl = session
            .primary_template(c)
            .and_then(|t| session.template_waveform(t));
        let lm = tpl.and_then(|w| {
            let pk = peak_channel(w, tpl_samples, tpl_channels)?;
            let trace: Vec<f32> = (0..tpl_samples).map(|s| w[s * tpl_channels + pk]).collect();
            waveform_landmarks(&trace, sr)
        });
        let fp = tpl
            .zip(positions)
            .and_then(|(w, p)| footprint(w, tpl_samples, tpl_channels, p, sr));
        let nan = f32::NAN;
        // (0, 0) means there is no estimate.
        let (rp_contam, rp) = windows.sweep(exposed, total_samples, sr);
        let (srp_contamination, srp_rp_ms, rpv_post_mean, rpv_post_lo95, rpv_post_hi95) =
            match windows {
                RefractoryWindows::Session(params) => {
                    let (srp, post) =
                        refractory_evidence(exposed, total_samples, sample_rate, params);
                    (
                        srp.min_contamination,
                        srp.rp_ms,
                        post.mean,
                        post.lo95,
                        post.hi95,
                    )
                }
                // These are public QC metrics, not versioned model features.
                RefractoryWindows::Learn => (nan, nan, nan, nan, nan),
            };
        let epochs = match windows {
            RefractoryWindows::Session(_) => stable_epochs(times, amps, span, sample_rate),
            RefractoryWindows::Learn => Vec::new(), // not a versioned model feature
        };
        let acg = acg_or_compute(session, c, acg_bin, acg_window);
        // ponytail: evenly strided to <= 5000 points to bound the O(n log n)
        // sort per cluster; a rare second mode in a huge cluster can thin out.
        let dip_amps: Vec<f32> = amps
            .iter()
            .step_by(amps.len().div_ceil(5000).max(1))
            .copied()
            .collect();
        rows.push(QcRow {
            cluster_id: c.0,
            n_spikes: times.len() as u32,
            mean_amplitude: mean_amp,
            firing_rate_hz: firing_rate,
            presence_ratio: q.raw_presence_ratio,
            isi_violations: isi_v,
            isi_violation_ratio: isi_ratio,
            amplitude_cutoff: q.raw_amp_cutoff,
            amplitude_snr: q.raw_snr,
            drift_correlation: q.raw_drift_corr,
            longest_silent_gap: q.raw_silent_gap,
            composite_quality: q.composite(),
            isolation_distance_sq: iso2,
            l_ratio,
            nn_isolation: nn,
            d_prime,
            silhouette,
            trough_to_peak_ms: lm.map_or(nan, |l| l.trough_to_peak_ms),
            half_width_ms: lm.map_or(nan, |l| l.half_width_ms),
            peak_trough_ratio: lm.map_or(nan, |l| l.peak_trough_ratio),
            repolarisation_slope: lm.map_or(nan, |l| l.repolarisation_slope),
            n_extrema: lm.map_or(nan, |l| l.n_extrema as f32),
            x_um: fp.map_or(nan, |f| f.x_um),
            y_um: fp.map_or(nan, |f| f.y_um),
            footprint_decay_um: fp.map_or(nan, |f| f.decay_um),
            footprint_r80_um: fp.map_or(nan, |f| f.r80_um),
            n_active_channels: fp.map_or(nan, |f| f.n_active as f32),
            propagation_ms_per_100um: fp.map_or(nan, |f| f.propagation_ms_per_100um),
            burst_index: burst_index(&acg, acg_bin as f32 * 1000.0 / sr),
            rp_estimate_ms: if rp > 0 { rp as f32 * 1000.0 / sr } else { nan },
            rp_contamination: if rp > 0 { rp_contam } else { nan },
            amplitude_dip: if dip_amps.len() >= 4 {
                hartigan_dip(&dip_amps)
            } else {
                nan
            },
            drift_slope: if amps.len() >= 2 && amps.len() == times.len() {
                amplitude_drift_slope(times, amps, sr)
            } else {
                nan
            },
            srp_contamination,
            srp_rp_ms,
            rpv_post_mean,
            rpv_post_lo95,
            rpv_post_hi95,
            n_epochs: if epochs.is_empty() {
                nan
            } else {
                epochs.len() as f32
            },
            epoch_amp_ratio: epoch_amp_ratio(&epochs),
            epoch_rate_ratio: epoch_ratio(&epochs, |e| e.rate_hz),
        });
    }
    rows
}

fn epoch_amp_ratio(epochs: &[Epoch]) -> f32 {
    if epochs.iter().all(|e| e.amp_fitted) {
        epoch_ratio(epochs, |e| e.amp_mean.abs())
    } else {
        f32::NAN
    }
}

fn epoch_ratio(epochs: &[Epoch], level: impl Fn(&Epoch) -> f32) -> f32 {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for x in epochs.iter().map(level) {
        if !x.is_finite() {
            return f32::NAN;
        }
        min = min.min(x);
        max = max.max(x);
    }
    if max > 0.0 {
        min / max
    } else {
        f32::NAN
    }
}

/// Write QC rows to a tab-separated file.
pub fn write_qc_tsv<P: AsRef<Path>>(rows: &[QcRow], path: P) -> Result<()> {
    let f = File::create(path.as_ref()).context("creating QC TSV file")?;
    let mut w = BufWriter::new(f);
    w.write_all(QcRow::tsv_header().as_bytes())?;
    for row in rows {
        row.write_tsv(&mut w)?;
    }
    w.flush()?;
    Ok(())
}

/// Write QC rows to a JSON file (`[{...}, {...}]`). Streaming so we don't
/// build a giant in-memory string for large recordings.
pub fn write_qc_json<P: AsRef<Path>>(rows: &[QcRow], path: P) -> Result<()> {
    write_qc_json_document(rows, path.as_ref(), None)
}

fn write_qc_json_document(
    rows: &[QcRow],
    path: &Path,
    metadata: Option<(serde_json::Value, serde_json::Value)>,
) -> Result<()> {
    let f = File::create(path).context("creating QC JSON file")?;
    let mut w = BufWriter::new(f);
    if let Some((span, provenance)) = &metadata {
        write!(
            w,
            "{{\"schema_version\":2,\"session_span\":{span},\"provenance\":{provenance},\"clusters\":"
        )?;
    }
    w.write_all(b"[")?;
    for (i, row) in rows.iter().enumerate() {
        row.write_json(&mut w, i > 0)?;
    }
    w.write_all(if metadata.is_some() {
        b"\n]}\n"
    } else {
        b"\n]\n"
    })?;
    w.flush()?;
    Ok(())
}

/// Convenience: compute all rows and write both formats into `out_dir`
/// under `cluster_qc.tsv` / `cluster_qc.json`. `positions` as for
/// [`collect_qc_rows_with`].
pub fn export_qc<P: DataProvider>(
    session: &Session<P>,
    out_dir: impl AsRef<Path>,
    positions: Option<&[[f32; 2]]>,
) -> Result<usize> {
    let dir = out_dir.as_ref();
    std::fs::create_dir_all(dir).context("creating QC output dir")?;
    let rows = collect_qc_rows_with(session, positions);
    write_qc_tsv(&rows, dir.join("cluster_qc.tsv"))?;
    let p = &session.provider;
    let span = p.session_span();
    let rate = p
        .exact_sample_rate()
        .unwrap_or_else(|_| f64::from(p.sample_rate()));
    let known = span.duration_samples() > 0 && rate.is_finite() && rate > 0.0;
    let seconds = |sample| known.then(|| sample as f64 / rate);
    write_qc_json_document(
        &rows,
        &dir.join("cluster_qc.json"),
        Some((
            serde_json::json!({
                "start_s": seconds(span.start.0),
                "end_s": seconds(span.end.0),
                "duration_s": seconds(span.duration_samples()),
                "source": span.source.to_string(),
            }),
            p.trace_provenance(),
        )),
    )?;
    Ok(rows.len())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::journal::Journal;
    use sorrel_io::{
        ClusterId, DataProvider, HasAmplitudes, SampleIndex, SessionSpan, TraceSamples, TraceSlice,
    };

    /// Shared with `report::tests`.
    pub(crate) struct MockProvider {
        pub(crate) spikes: Vec<Vec<SampleIndex>>,
        pub(crate) amps: Vec<Vec<f32>>,
        pub(crate) n_samples: SampleIndex,
    }
    impl DataProvider for MockProvider {
        type Label = u8;
        fn sample_rate(&self) -> f32 {
            1000.0
        }
        fn n_channels(&self) -> u32 {
            1
        }
        fn n_samples(&self) -> SampleIndex {
            self.n_samples
        }
        fn n_clusters(&self) -> u32 {
            self.spikes.len() as u32
        }
        fn spike_times(&self, c: ClusterId) -> &[SampleIndex] {
            self.spikes.get(c.idx()).map(Vec::as_slice).unwrap_or(&[])
        }
        fn trace(&self, _: SampleIndex, _: u32) -> TraceSlice<'_> {
            TraceSlice {
                start: SampleIndex(0),
                n_channels: 1,
                samples: TraceSamples::I16(&[]),
            }
        }
        fn initial_labels(&self) -> Vec<u8> {
            vec![0; self.spikes.len()]
        }
    }
    impl crate::session::ApplyPhyLabel for MockProvider {
        fn label_from_op(op: crate::command::PhyLabelOp) -> u8 {
            match op {
                crate::command::PhyLabelOp::SetUnsorted => 0,
                crate::command::PhyLabelOp::SetGood => 1,
                crate::command::PhyLabelOp::SetMua => 2,
                crate::command::PhyLabelOp::SetNoise => 3,
            }
        }
    }
    impl HasAmplitudes for MockProvider {
        fn spike_amplitudes(&self, c: ClusterId) -> &[f32] {
            self.amps.get(c.idx()).map(Vec::as_slice).unwrap_or(&[])
        }
    }

    struct SpanProvider(MockProvider, SessionSpan);

    impl DataProvider for SpanProvider {
        type Label = u8;
        fn sample_rate(&self) -> f32 {
            self.0.sample_rate()
        }
        fn n_channels(&self) -> u32 {
            self.0.n_channels()
        }
        fn n_samples(&self) -> SampleIndex {
            self.0.n_samples()
        }
        fn session_span(&self) -> SessionSpan {
            self.1
        }
        fn n_clusters(&self) -> u32 {
            self.0.n_clusters()
        }
        fn spike_times(&self, c: ClusterId) -> &[SampleIndex] {
            self.0.spike_times(c)
        }
        fn trace(&self, start: SampleIndex, len: u32) -> TraceSlice<'_> {
            self.0.trace(start, len)
        }
        fn initial_labels(&self) -> Vec<u8> {
            self.0.initial_labels()
        }
    }

    #[test]
    fn qc_uses_offset_spike_span_without_a_trace() {
        let dir = tempfile::tempdir().unwrap();
        let provider = |start: u64, trace_len| MockProvider {
            spikes: vec![(0..100).map(|i| SampleIndex(start + i * 10)).collect()],
            amps: vec![],
            n_samples: SampleIndex(trace_len),
        };
        let journal = |name| Journal::open(&dir.path().join(name)).unwrap();
        let trace = Session::new(provider(0, 1000), journal("trace"));
        let units = Session::new(
            SpanProvider(
                provider(5000, 0),
                SessionSpan {
                    start: SampleIndex(5000),
                    end: SampleIndex(6000),
                    source: SpanSource::SpikeExtent { rejected: None },
                },
            ),
            journal("units"),
        );
        let row = collect_qc_rows(&units)[0].to_json();
        assert_eq!(row, collect_qc_rows(&trace)[0].to_json());
        assert_eq!(row["firing_rate"], 100.0);
        assert_eq!(row["presence_ratio"], 1.0);
        for key in ["rp_estimate_ms", "srp_contamination", "rpv_post_mean"] {
            assert!(!row[key].is_null(), "{key}");
        }
        export_qc(&units, dir.path(), None).unwrap();
        let qc: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("cluster_qc.json")).unwrap())
                .unwrap();
        assert_eq!(
            qc["session_span"],
            serde_json::json!({
                "start_s": 5.0, "end_s": 6.0, "duration_s": 1.0, "source": "estimate: spike extent",
            })
        );
    }

    #[test]
    fn qc_missing_span_leaves_rate_and_presence_null() {
        let dir = tempfile::tempdir().unwrap();
        let session = Session::new(
            SpanProvider(
                MockProvider {
                    spikes: vec![vec![]],
                    amps: vec![],
                    n_samples: SampleIndex(0),
                },
                SessionSpan {
                    source: SpanSource::Unavailable,
                    ..0.into()
                },
            ),
            Journal::open(&dir.path().join("j")).unwrap(),
        );
        let row = collect_qc_rows(&session)[0].to_json();
        assert!(row["firing_rate"].is_null() && row["presence_ratio"].is_null());
    }

    #[test]
    fn session_refractory_changes_qc_and_evaluation() {
        use crate::learn::features::{extract, ExtractOpts};

        let dir = tempfile::tempdir().unwrap();
        let prov = MockProvider {
            spikes: vec![vec![SampleIndex(10), SampleIndex(11), SampleIndex(150)]],
            amps: vec![vec![1.0; 3]],
            n_samples: SampleIndex(1000),
        };
        let mut s = Session::new(prov, Journal::open(&dir.path().join("j")).unwrap());
        let edit_version = s.version();
        let metrics_version = s.metrics_version();
        let before = &collect_qc_rows(&s)[0];
        let learned_before = extract(&s, "fixture", &ExtractOpts::default()).unwrap();
        let mut params = *s.refractory();
        params.fixed_ms = 0.5;
        params.sweep_min_ms = 1.5;
        s.set_refractory(params).unwrap();
        let after = &collect_qc_rows(&s)[0];
        assert_eq!((before.isi_violations, after.isi_violations), (1, 0));
        assert_eq!(before.rp_contamination, 0.0);
        assert!(after.rp_contamination > 0.0);
        let ev = crate::evaluate::eval_merge(&s, &[ClusterId(0)]);
        assert_eq!(ev.merged.rp_contamination, after.rp_contamination);
        assert_eq!(s.version(), edit_version);
        assert_ne!(s.metrics_version(), metrics_version);
        let learned_after = extract(&s, "fixture", &ExtractOpts::default()).unwrap();
        assert_eq!(
            learned_before.tables.roots.get(0, "f_rp_contamination"),
            0.0
        );
        assert_eq!(learned_before.tables.roots, learned_after.tables.roots);
    }

    #[test]
    fn custom_refractory_survives_edits_but_does_not_change_journal_replay() {
        use crate::command::{CurationCommand as Cmd, Method};
        use crate::{evaluate, preview, quality_ext};

        let dir = tempfile::tempdir().unwrap();
        let open = || {
            let prov = MockProvider {
                spikes: vec![
                    [0, 3, 30, 60, 90].map(SampleIndex).to_vec(),
                    [5, 50].map(SampleIndex).to_vec(),
                ],
                amps: vec![vec![1.0; 5], vec![1.0; 2]],
                n_samples: SampleIndex(100),
            };
            Session::new(prov, Journal::open(&dir.path().join("j")).unwrap())
        };
        let mut s = open();
        let c = ClusterId(0);
        let d = ClusterId(1);
        let e = ClusterId(2);
        let version = s.version();
        let params: sorrel_compute::RefractoryParams =
            serde_json::from_str(r#"{"fixed_ms":4.0,"sweep_min_ms":3.5}"#).unwrap();
        assert_eq!(params.sweep_max_ms, 10.0);
        assert_eq!(params.sweep_step_ms, 0.25);
        assert_eq!(
            serde_json::from_value::<sorrel_compute::RefractoryParams>(
                serde_json::to_value(params).unwrap()
            )
            .unwrap(),
            params
        );
        let before = preview::preview_merge(&s, &[d], c);
        s.set_refractory(params).unwrap();
        assert_eq!(s.version(), version);
        assert_eq!(s.journal().n_records(), 0);

        let after = preview::preview_merge(&s, &[d], c);
        assert!(after.post.contamination_score < before.post.contamination_score);
        let expected = evaluate::eval_merge(&s, &[c, d]).merged;
        s.dispatch(Cmd::Merge {
            sources: vec![d],
            target: c,
        })
        .unwrap();
        let merged_version = s.version();
        assert_eq!(collect_qc_rows(&s)[0].isi_violations, 2);
        assert_eq!(
            collect_qc_rows(&s)[0].rp_contamination,
            expected.rp_contamination
        );
        assert_eq!(
            quality_ext::cluster_quality(&s, c).contamination_score,
            after.post.contamination_score
        );

        let preview = preview::preview_split(&s, c, &[0, 1]);
        let eval = evaluate::eval_split(&s, c, &[0, 1]);
        assert!(preview.inside.contamination_score < preview.outside.contamination_score);
        assert_eq!(eval.parts[1].isi_violation_ratio, 1_000.0 / 3.0);
        s.dispatch(Cmd::Split {
            cluster: c,
            spike_idx: vec![0, 1],
            new_cluster: c,
            by: Method::Manual,
        })
        .unwrap();
        let split_version = s.version();
        let split_times = s.spike_times(e).to_vec();
        assert_eq!(split_times, preview.inside_times);
        let counts = |session: &Session<_>| {
            collect_qc_rows(session)
                .into_iter()
                .filter(|row| row.n_spikes != 0)
                .map(|row| (row.cluster_id, row.isi_violations))
                .collect::<Vec<_>>()
        };
        let split = counts(&s);
        assert_eq!(split, [(0, 0), (2, 1)]);
        s.dispatch(Cmd::Undo).unwrap();
        assert_eq!(s.version(), merged_version);
        assert_eq!(counts(&s), [(0, 2)]);
        s.dispatch(Cmd::Redo).unwrap();
        assert_eq!(s.version(), split_version);
        assert_eq!(counts(&s), split);
        drop(s);

        let mut reopened = open();
        reopened.replay_journal().unwrap();
        assert_eq!(
            *reopened.refractory(),
            sorrel_compute::RefractoryParams::default()
        );
        assert_eq!(reopened.version(), split_version);
        assert_eq!(reopened.spike_times(e), split_times);
        assert_eq!(counts(&reopened), [(0, 0), (2, 0)]);
        reopened.set_refractory(params).unwrap();
        assert_eq!(counts(&reopened), split);
    }

    #[test]
    fn export_writes_both_formats_with_one_row_per_cluster() {
        let prov = MockProvider {
            spikes: vec![
                vec![SampleIndex(10), SampleIndex(30), SampleIndex(150)],
                vec![SampleIndex(20), SampleIndex(200)],
                vec![SampleIndex(100)],
                vec![],
            ],
            amps: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0], vec![6.0], vec![]],
            n_samples: SampleIndex(1000),
        };
        let dir = tempfile::tempdir().unwrap();
        let journal = Journal::open(&dir.path().join("j.sqlite")).unwrap();
        let mut s = Session::new(prov, journal);
        s.seed_amplitudes();

        let outdir = dir.path().join("export");
        let n = export_qc(&s, &outdir, None).unwrap();
        assert_eq!(n, 4);

        let tsv = std::fs::read_to_string(outdir.join("cluster_qc.tsv")).unwrap();
        let lines: Vec<&str> = tsv.lines().collect();
        // 1 header + 4 rows.
        assert_eq!(lines.len(), 5);
        assert!(lines[0].starts_with("cluster_id\t"));
        assert!(lines[1].starts_with("0\t"));

        let json = std::fs::read_to_string(outdir.join("cluster_qc.json")).unwrap();
        let document: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(document["schema_version"], 2);
        assert_eq!(document["session_span"]["source"], "trace");
        assert_eq!(document["provenance"]["mode"], "trace");
        // Each row contains its cluster id.
        assert!(json.contains("\"cluster_id\": 0"));
        assert!(json.contains("\"cluster_id\": 2"));

        // Both formats expose the same column names, in the same order, so
        // downstream code can switch between them without a rename table.
        let tsv_keys: Vec<&str> = lines[0].split('\t').collect();
        let first_row = json
            .lines()
            .find(|l| l.contains("\"cluster_id\": 0"))
            .unwrap();
        let json_keys: Vec<&str> = first_row.split('"').skip(1).step_by(2).collect();
        assert_eq!(tsv_keys, json_keys);
        assert!(tsv_keys.ends_with(&["n_epochs", "epoch_amp_ratio", "epoch_rate_ratio"]));
        let rows = &document["clusters"];
        for name in [
            "srp_contamination",
            "srp_rp_ms",
            "rpv_post_mean",
            "rpv_post_lo95",
            "rpv_post_hi95",
            "n_epochs",
            "epoch_amp_ratio",
            "epoch_rate_ratio",
        ] {
            let i = tsv_keys.iter().position(|&k| k == name).unwrap();
            for row in [2, 3] {
                assert!(rows[row].get(name).unwrap().is_null(), "{name}");
                assert_eq!(lines[row + 1].split('\t').nth(i), Some(""), "{name}");
            }
        }
        assert!(rows[0]["rpv_post_hi95"].as_f64().unwrap() > 0.1);
    }

    #[test]
    fn refractory_evidence_tsv_and_json_agree_on_schema_values_and_missingness() {
        let dir = tempfile::tempdir().unwrap();
        let s = Session::new(
            MockProvider {
                spikes: vec![
                    (0..100).map(|i| SampleIndex(i * 10)).collect(),
                    vec![SampleIndex(10)],
                ],
                amps: vec![vec![], vec![]],
                n_samples: SampleIndex(1000),
            },
            Journal::open(&dir.path().join("j")).unwrap(),
        );
        let mut rows = collect_qc_rows(&s);
        assert_eq!(rows[0].srp_contamination, 0.015);
        assert_eq!(rows[0].srp_rp_ms, 10.0);
        let tiny = 2.5e-10;
        rows[0].rpv_post_mean = tiny;
        rows[0].n_spikes = (1 << 24) + 1;
        rows[0].drift_correlation = f32::INFINITY;
        let tsv = dir.path().join("qc.tsv");
        let json = dir.path().join("qc.json");
        write_qc_tsv(&rows, &tsv).unwrap();
        write_qc_json(&rows, &json).unwrap();
        let tsv = std::fs::read_to_string(tsv).unwrap();
        let json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(json).unwrap()).unwrap();
        let mut lines = tsv.lines();
        let header: Vec<_> = lines.next().unwrap().split('\t').collect();
        assert_eq!(header.len(), 41);
        let evidence = [
            "srp_contamination",
            "srp_rp_ms",
            "rpv_post_mean",
            "rpv_post_lo95",
            "rpv_post_hi95",
        ];
        assert_eq!(&header[33..38], evidence);
        let records: Vec<_> = lines
            .map(|line| line.split('\t').collect::<Vec<_>>())
            .collect();
        assert_eq!(records.len(), 2);
        assert!(records.iter().all(|row| row.len() == header.len()));
        assert_eq!(records[0][1], "16777217");
        assert_eq!(json[0]["n_spikes"], 16777217);
        let drift = header
            .iter()
            .position(|&h| h == "drift_correlation")
            .unwrap();
        assert_eq!(records[0][drift], "");
        assert!(json[0]["drift_correlation"].is_null());
        for (i, name) in evidence.into_iter().enumerate() {
            let col = i + 33;
            let tsv_value = records[0][col].parse::<f64>().unwrap();
            assert_eq!(json[0][name].as_f64(), Some(tsv_value), "{name}");
            assert_eq!(records[1][col], "");
            assert!(json[1][name].is_null(), "{name}");
        }
        assert_eq!(json[0]["rpv_post_mean"].as_f64().unwrap() as f32, tiny);
    }

    #[test]
    fn isolation_columns_are_blank_in_tsv_when_no_pc_features() {
        let prov = MockProvider {
            spikes: vec![vec![SampleIndex(10), SampleIndex(20)]],
            amps: vec![vec![1.0, 2.0]],
            n_samples: SampleIndex(100),
        };
        let dir = tempfile::tempdir().unwrap();
        let journal = Journal::open(&dir.path().join("j.sqlite")).unwrap();
        let mut s = Session::new(prov, journal);
        s.seed_amplitudes();
        let outdir = dir.path().join("export");
        export_qc(&s, &outdir, None).unwrap();
        let tsv = std::fs::read_to_string(outdir.join("cluster_qc.tsv")).unwrap();
        let row = tsv.lines().nth(1).unwrap();
        // Isolation and footprint columns are empty without PC features /
        // templates / geometry.
        let header = tsv.lines().next().unwrap().split('\t');
        let cells: Vec<&str> = row.split('\t').collect();
        assert_eq!(header.clone().count(), cells.len());
        for (name, cell) in header.zip(&cells) {
            if [
                "l_ratio",
                "silhouette",
                "half_width_ms",
                "x_um",
                "footprint_decay_um",
            ]
            .contains(&name)
            {
                assert_eq!(*cell, "", "{name}");
            }
        }
    }
}
