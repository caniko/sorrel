//! QC export — write every cluster's quality metrics to a tab-separated
//! file (SpikeInterface-compatible columns) and/or to a JSON snapshot.
//!
//! Two output formats:
//! * `cluster_qc.tsv`: one row per cluster, headed by phy-style column
//!   names. Compatible with `pandas.read_csv(..., sep='\t')` and
//!   `spikeinterface.curation.read_qm_csv`.
//! * `cluster_qc.json`: a list of objects with the same fields plus the
//!   isolation breakdown when available. Easier to consume from notebooks
//!   when the column count grows.
//!
//! Both run the rayon-parallel quality + isolation pipeline so even 1k+
//! cluster sessions export in a few seconds.

use crate::cache::correlograms::acg_or_compute;
use crate::quality_ext::cluster_quality;
use crate::session::Session;
use anyhow::{Context, Result};
use sorrel_io::{ClusterId, DataProvider};
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
        }
    }
}

impl QcRow {
    /// Columns before the shape/footprint additions; the TSV and JSON
    /// writers emit those by name from `METRICS[N_BASE..]`.
    const N_BASE: usize = 16;

    /// Every metric column name, in TSV order. The single source of truth
    /// for names used by `get`, the rule engine, and the exports.
    pub const METRICS: [&'static str; 32] = [
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
            _ => return None,
        })
    }

    fn tsv_header() -> String {
        let mut h = "cluster_id\tn_spikes\tfiring_rate\tmean_amplitude\tpresence_ratio\t\
         isi_violations\tisi_violation_ratio\tamplitude_cutoff\tamplitude_snr\t\
         drift_correlation\tlongest_silent_gap\tcomposite_quality\t\
         isolation_distance_sq\tl_ratio\tnn_isolation\td_prime\tsilhouette"
            .to_string();
        for name in &Self::METRICS[Self::N_BASE..] {
            h.push('\t');
            h.push_str(name);
        }
        h.push('\n');
        h
    }

    fn write_tsv(&self, w: &mut impl Write) -> std::io::Result<()> {
        write!(
            w,
            "{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{}\t{}\t{}",
            self.cluster_id,
            self.n_spikes,
            self.firing_rate_hz,
            self.mean_amplitude,
            self.presence_ratio,
            self.isi_violations,
            self.isi_violation_ratio,
            self.amplitude_cutoff,
            self.amplitude_snr,
            self.drift_correlation,
            self.longest_silent_gap,
            self.composite_quality,
            fmt_or_empty(self.isolation_distance_sq),
            fmt_or_empty(self.l_ratio),
            fmt_or_empty(self.nn_isolation),
            fmt_or_empty(self.d_prime),
            fmt_or_empty(self.silhouette),
        )?;
        for name in &Self::METRICS[Self::N_BASE..] {
            write!(w, "\t{}", fmt_or_empty(self.get(name).unwrap_or(f32::NAN)))?;
        }
        writeln!(w)
    }

    fn write_json(&self, w: &mut impl Write, leading_comma: bool) -> std::io::Result<()> {
        let prefix = if leading_comma { ",\n" } else { "\n" };
        write!(
            w,
            r#"{prefix}  {{"cluster_id": {}, "n_spikes": {}, "firing_rate": {:.6}, "mean_amplitude": {:.6}, "presence_ratio": {:.6}, "isi_violations": {}, "isi_violation_ratio": {:.6}, "amplitude_cutoff": {:.6}, "amplitude_snr": {:.6}, "drift_correlation": {:.6}, "longest_silent_gap": {:.6}, "composite_quality": {:.6}, "isolation_distance_sq": {}, "l_ratio": {}, "nn_isolation": {}, "d_prime": {}, "silhouette": {}"#,
            self.cluster_id,
            self.n_spikes,
            self.firing_rate_hz,
            self.mean_amplitude,
            self.presence_ratio,
            self.isi_violations,
            self.isi_violation_ratio,
            self.amplitude_cutoff,
            self.amplitude_snr,
            self.drift_correlation,
            self.longest_silent_gap,
            self.composite_quality,
            json_num(self.isolation_distance_sq),
            json_num(self.l_ratio),
            json_num(self.nn_isolation),
            json_num(self.d_prime),
            json_num(self.silhouette),
        )?;
        for name in &Self::METRICS[Self::N_BASE..] {
            write!(
                w,
                r#", "{name}": {}"#,
                json_num(self.get(name).unwrap_or(f32::NAN))
            )?;
        }
        write!(w, "}}")
    }
}

fn fmt_or_empty(v: f32) -> String {
    if v.is_finite() {
        format!("{v:.6}")
    } else {
        String::new()
    }
}

fn json_num(v: f32) -> String {
    if v.is_finite() {
        format!("{v:.6}")
    } else {
        "null".to_string()
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
    use sorrel_compute::{
        amplitude_drift_correlation, amplitude_drift_slope, amplitude_snr, burst_index, footprint,
        hartigan_dip, isi_violation_rate, isi_violations, longest_silent_gap_frac, mean_amplitude,
        peak_channel, presence_ratio, refractory_contamination, sliding_refractory_argmin,
        waveform_landmarks,
    };
    let sr = session.provider.sample_rate().max(1.0);
    let total_samples = session.provider.n_samples();
    let total_seconds = total_samples.as_f32() / sr;
    let refractory_samples = (sr * 0.0015).round() as u64;
    let (_, tpl_samples, tpl_channels) = session.template_shape();
    // ACG at 1 ms bins / 50 ms, the report's (and cache's) window.
    let acg_window = (0.050 * sr) as u32;
    let acg_bin = ((0.001 * sr) as u32).max(1);

    let n = session.n_clusters();
    let mut rows = Vec::with_capacity(n as usize);
    for c in (0..n).map(ClusterId) {
        let times = session.spike_times(c);
        let amps = session.spike_amplitudes(c);
        let q = cluster_quality(session, c);
        let mean_amp = if amps.is_empty() {
            f32::NAN
        } else {
            mean_amplitude(amps)
        };
        let firing_rate = if total_seconds > 0.0 {
            times.len() as f32 / total_seconds
        } else {
            0.0
        };
        let isi_v = isi_violations(times, refractory_samples) as u32;
        let isi_ratio = if !times.is_empty() {
            isi_violation_rate(times, refractory_samples, sr)
        } else {
            0.0
        };
        let _contam = refractory_contamination(times, refractory_samples, total_samples.0, sr);
        let _ = (
            amplitude_snr,
            amplitude_drift_correlation,
            presence_ratio,
            longest_silent_gap_frac,
        );
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
        // 0.5-10 ms in 20 steps; (0, 0) means too few spikes to say.
        let (rp_contam, rp) = sliding_refractory_argmin(
            times,
            (sr * 0.0005) as u64,
            (sr * 0.010) as u64,
            20,
            total_samples.0,
            sr,
        );
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
        });
    }
    rows
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
    let f = File::create(path.as_ref()).context("creating QC JSON file")?;
    let mut w = BufWriter::new(f);
    w.write_all(b"[")?;
    for (i, row) in rows.iter().enumerate() {
        row.write_json(&mut w, i > 0)?;
    }
    w.write_all(b"\n]\n")?;
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
    write_qc_json(&rows, dir.join("cluster_qc.json"))?;
    Ok(rows.len())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::journal::Journal;
    use sorrel_io::{
        ClusterId, DataProvider, HasAmplitudes, SampleIndex, TraceSamples, TraceSlice,
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

    #[test]
    fn export_writes_both_formats_with_one_row_per_cluster() {
        let prov = MockProvider {
            spikes: vec![
                vec![SampleIndex(10), SampleIndex(30), SampleIndex(150)],
                vec![SampleIndex(20), SampleIndex(200)],
                vec![SampleIndex(100)],
            ],
            amps: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0], vec![6.0]],
            n_samples: SampleIndex(1000),
        };
        let dir = tempfile::tempdir().unwrap();
        let journal = Journal::open(&dir.path().join("j.sqlite")).unwrap();
        let mut s = Session::new(prov, journal);
        s.seed_amplitudes();

        let outdir = dir.path().join("export");
        let n = export_qc(&s, &outdir, None).unwrap();
        assert_eq!(n, 3);

        let tsv = std::fs::read_to_string(outdir.join("cluster_qc.tsv")).unwrap();
        let lines: Vec<&str> = tsv.lines().collect();
        // 1 header + 3 rows.
        assert_eq!(lines.len(), 4);
        assert!(lines[0].starts_with("cluster_id\t"));
        assert!(lines[1].starts_with("0\t"));

        let json = std::fs::read_to_string(outdir.join("cluster_qc.json")).unwrap();
        assert!(json.starts_with("["));
        assert!(json.trim_end().ends_with("]"));
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
