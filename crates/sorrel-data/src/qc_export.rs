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
}

impl QcRow {
    fn tsv_header() -> &'static str {
        "cluster_id\tn_spikes\tfiring_rate\tmean_amplitude\tpresence_ratio\t\
         isi_violations\tisi_violation_ratio\tamplitude_cutoff\tamplitude_snr\t\
         drift_correlation\tlongest_silent_gap\tcomposite_quality\t\
         isolation_distance_sq\tl_ratio\tnn_isolation\td_prime\tsilhouette\n"
    }

    fn write_tsv(&self, w: &mut impl Write) -> std::io::Result<()> {
        writeln!(
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
        )
    }

    fn write_json(&self, w: &mut impl Write, leading_comma: bool) -> std::io::Result<()> {
        let prefix = if leading_comma { ",\n" } else { "\n" };
        write!(
            w,
            r#"{prefix}  {{"cluster_id": {}, "n_spikes": {}, "firing_rate_hz": {:.6}, "mean_amplitude": {:.6}, "presence_ratio": {:.6}, "isi_violations": {}, "isi_violation_ratio": {:.6}, "amplitude_cutoff": {:.6}, "amplitude_snr": {:.6}, "drift_correlation": {:.6}, "longest_silent_gap": {:.6}, "composite_quality": {:.6}, "isolation_distance_sq": {}, "l_ratio": {}, "nn_isolation": {}, "d_prime": {}, "silhouette": {}}}"#,
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
        )
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

/// Compute QC rows for every cluster in the session.
pub fn collect_qc_rows<P: DataProvider>(session: &Session<P>) -> Vec<QcRow> {
    use sorrel_compute::{
        amplitude_drift_correlation, amplitude_snr, isi_violation_rate, isi_violations,
        longest_silent_gap_frac, mean_amplitude, presence_ratio, refractory_contamination,
    };
    let sr = session.provider.sample_rate().max(1.0);
    let total_samples = session.provider.n_samples();
    let total_seconds = total_samples.as_f32() / sr;
    let refractory_samples = (sr * 0.0015).round() as u64;

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
/// under `cluster_qc.tsv` / `cluster_qc.json`.
pub fn export_qc<P: DataProvider>(
    session: &Session<P>,
    out_dir: impl AsRef<Path>,
) -> Result<usize> {
    let dir = out_dir.as_ref();
    std::fs::create_dir_all(dir).context("creating QC output dir")?;
    let rows = collect_qc_rows(session);
    write_qc_tsv(&rows, dir.join("cluster_qc.tsv"))?;
    write_qc_json(&rows, dir.join("cluster_qc.json"))?;
    Ok(rows.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::SqliteJournal;
    use sorrel_io::{
        ClusterId, DataProvider, HasAmplitudes, SampleIndex, TraceSamples, TraceSlice,
    };

    struct MockProvider {
        spikes: Vec<Vec<SampleIndex>>,
        amps: Vec<Vec<f32>>,
        n_samples: SampleIndex,
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
        let journal = SqliteJournal::open(&dir.path().join("j.sqlite")).unwrap();
        let mut s = Session::new(prov, journal);
        s.seed_amplitudes();

        let outdir = dir.path().join("export");
        let n = export_qc(&s, &outdir).unwrap();
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
    }

    #[test]
    fn isolation_columns_are_blank_in_tsv_when_no_pc_features() {
        let prov = MockProvider {
            spikes: vec![vec![SampleIndex(10), SampleIndex(20)]],
            amps: vec![vec![1.0, 2.0]],
            n_samples: SampleIndex(100),
        };
        let dir = tempfile::tempdir().unwrap();
        let journal = SqliteJournal::open(&dir.path().join("j.sqlite")).unwrap();
        let mut s = Session::new(prov, journal);
        s.seed_amplitudes();
        let outdir = dir.path().join("export");
        export_qc(&s, &outdir).unwrap();
        let tsv = std::fs::read_to_string(outdir.join("cluster_qc.tsv")).unwrap();
        let row = tsv.lines().nth(1).unwrap();
        // Last 3 columns should be empty for a session without PC features.
        let trailing: Vec<&str> = row.split('\t').collect();
        assert_eq!(trailing[trailing.len() - 1], "");
        assert_eq!(trailing[trailing.len() - 2], "");
        assert_eq!(trailing[trailing.len() - 3], "");
    }
}
