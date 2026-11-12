//! Bounded waveform evidence from the current spike membership, not the
//! primary sorter template (which split children can share).

use crate::session::Session;
use sorrel_compute::{peak_channel, waveform_landmarks, WaveformLandmarks};
use sorrel_io::{ClusterId, DataProvider, SampleIndex};

const MAX_SPIKES: usize = 128;
const MAX_CHANNELS: usize = 8;

pub struct RawWaveform {
    pub sampled: usize,
    pub used: usize,
    pub channels: usize,
    pub peak_channel: u32,
    pub snr: f32,
    pub early_late_cos: f32,
    pub shape: Option<WaveformLandmarks>,
    pub input_high_passed: Option<bool>,
}

fn median(values: &mut [f32]) -> f32 {
    values.sort_unstable_by(f32::total_cmp);
    let m = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[m - 1] + values[m]) * 0.5
    } else {
        values[m]
    }
}

/// One deterministic jittered draw per time-ordered stratum: bounded and
/// without replacement, avoiding the regular stride's periodic aliasing.
fn sample_indices(n: usize) -> Vec<usize> {
    let m = n.min(MAX_SPIKES);
    let mut seed = 0x4d59_5df4_d0f3_3173_u64;
    (0..m)
        .map(|j| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let (lo, hi) = (j * n / m, (j + 1) * n / m);
            lo + (seed as usize % (hi - lo))
        })
        .collect()
}

pub fn measure<P: DataProvider>(session: &Session<P>, c: ClusterId) -> Result<RawWaveform, String> {
    let p = &session.provider;
    let sr = p.sample_rate();
    if !sr.is_finite() || !(1000.0..=1_000_000.0).contains(&sr) {
        return Err("raw waveform requires a finite sample rate in 1kHz..1MHz".into());
    }
    let nc = p.n_channels() as usize;
    if nc == 0 || nc > 4096 {
        return Err("raw waveform requires 1..4096 trace channels".into());
    }
    let mut channels = Vec::new();
    for ch in session.best_channels(&[c]) {
        if (ch as usize) < nc && !channels.contains(&ch) {
            channels.push(ch);
            if channels.len() == MAX_CHANNELS {
                break;
            }
        }
    }
    if channels.is_empty() {
        channels = (0..nc.min(MAX_CHANNELS) as u32).collect();
    }
    let times = session.spike_times(c);
    if times.len() < 2 {
        return Err("raw waveform needs at least two spikes".into());
    }
    let indices = sample_indices(times.len());
    let before = (0.002 * sr).round() as usize;
    let after = (0.003 * sr).round() as usize;
    let len = before + after + 1;
    if len * nc > 1_000_000 {
        return Err("raw waveform window exceeds the bounded trace-element budget".into());
    }
    let baseline = (before / 2).max(1);
    let windows: Vec<_> = indices
        .iter()
        .filter_map(|&i| {
            let start = times[i].0.checked_sub(before as u64)?;
            (start.checked_add(len as u64)? <= p.n_samples().0)
                .then_some((SampleIndex(start), len as u32))
        })
        .collect();
    p.prefetch(&windows);
    let d = channels.len();
    let mut sums = [vec![0.0_f64; len * d], vec![0.0_f64; len * d]];
    let mut counts = [0_usize; 2];
    let mut noise = vec![Vec::new(); d];
    // Preserve chronological halves even when edge/incomplete windows are
    // skipped, rather than mixing early and late data through parity sampling.
    let split_time = times[times.len() / 2];
    for (start, _) in windows {
        let trace = p.trace(start, len as u32);
        if trace.start != start
            || trace.n_channels as usize != nc
            || trace.samples.len() != len * nc
        {
            continue;
        }
        let values = trace.samples.to_f32_centred();
        if !values.iter().all(|v| v.is_finite()) {
            continue;
        }
        let half = usize::from(start.0 + before as u64 >= split_time.0);
        for (j, &ch) in channels.iter().enumerate() {
            let mut base: Vec<_> = (0..baseline)
                .map(|t| values[t * nc + ch as usize])
                .collect();
            let offset = median(&mut base);
            noise[j].extend(base.into_iter().map(|v| (v - offset).abs()));
            for t in 0..len {
                sums[half][t * d + j] += (values[t * nc + ch as usize] - offset) as f64;
            }
        }
        counts[half] += 1;
    }
    let used = counts[0] + counts[1];
    if used < 2 {
        return Err("fewer than two complete raw waveform windows; no template fallback".into());
    }
    // Landmarks use -0.5..+1.5 ms; the longer window supplies a pre-spike
    // baseline. DC subtraction only: never pretend unfiltered raw was filtered.
    let lo = before - (0.0005 * sr).round() as usize;
    let hi = before + (0.0015 * sr).round() as usize + 1;
    let mean: Vec<f32> = (lo * d..hi * d)
        .map(|i| ((sums[0][i] + sums[1][i]) / used as f64) as f32)
        .collect();
    let pk = peak_channel(&mean, hi - lo, d).ok_or("no finite raw peak channel")?;
    let waveform: Vec<_> = mean.chunks_exact(d).map(|row| row[pk]).collect();
    let sigma = 1.4826 * median(&mut noise[pk]);
    let ptp = waveform.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - waveform.iter().copied().fold(f32::INFINITY, f32::min);
    let snr = if sigma > 0.0 { ptp / sigma } else { f32::NAN };
    let mut dot = 0.0;
    let mut norm = [0.0_f64; 2];
    for t in lo..hi {
        let a = sums[0][t * d + pk];
        let b = sums[1][t * d + pk];
        dot += a * b;
        norm[0] += a * a;
        norm[1] += b * b;
    }
    let early_late_cos = if counts.iter().all(|&n| n > 0) && norm.iter().all(|&n| n > 0.0) {
        (dot / (norm[0] * norm[1]).sqrt()).clamp(-1.0, 1.0) as f32
    } else {
        f32::NAN
    };
    Ok(RawWaveform {
        sampled: indices.len(),
        used,
        channels: d,
        peak_channel: channels[pk],
        snr,
        early_late_cos,
        shape: waveform_landmarks(&waveform, sr),
        input_high_passed: p.raw_file().map(|r| r.hp_filtered),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CurationCommand, Journal, Method};
    use sorrel_io::{TraceSamples, TraceSlice};

    struct Raw {
        times: Vec<SampleIndex>,
        samples: Vec<f32>,
        missing: bool,
    }
    impl crate::session::ApplyPhyLabel for Raw {
        fn label_from_op(op: crate::command::PhyLabelOp) -> u8 {
            match op {
                crate::command::PhyLabelOp::SetUnsorted => 0,
                crate::command::PhyLabelOp::SetGood => 1,
                crate::command::PhyLabelOp::SetMua => 2,
                crate::command::PhyLabelOp::SetNoise => 3,
            }
        }
    }
    impl DataProvider for Raw {
        type Label = u8;
        fn sample_rate(&self) -> f32 {
            10_000.0
        }
        fn n_channels(&self) -> u32 {
            2
        }
        fn n_samples(&self) -> SampleIndex {
            SampleIndex((self.samples.len() / 2) as u64)
        }
        fn n_clusters(&self) -> u32 {
            1
        }
        fn spike_times(&self, _: ClusterId) -> &[SampleIndex] {
            &self.times
        }
        fn initial_labels(&self) -> Vec<u8> {
            vec![0]
        }
        fn trace(&self, start: SampleIndex, len: u32) -> TraceSlice<'_> {
            let lo = start.0 as usize * 2;
            let hi = lo + len as usize * 2;
            TraceSlice {
                start,
                n_channels: 2,
                samples: TraceSamples::F32(if self.missing {
                    &[]
                } else {
                    &self.samples[lo..hi]
                }),
            }
        }
    }

    fn raw() -> Raw {
        let times: Vec<_> = (0..20).map(|i| SampleIndex(100 + i * 200)).collect();
        let mut samples: Vec<f32> = (0..8200)
            .map(|i| 0.2 * ((i as f32 * 0.31).sin() + (i as f32 * 0.17).cos()))
            .collect();
        for (i, time) in times.iter().enumerate() {
            for dt in -15_i64..=20 {
                let t = (time.0 as i64 + dt) as usize;
                let width = if i % 2 == 0 { 2.0 } else { 4.0 };
                let pulse = -10.0 * (-(dt as f32 / width).powi(2)).exp()
                    + 4.0 * (-((dt as f32 - 7.0) / 3.0).powi(2)).exp();
                samples[t * 2 + i % 2] += pulse;
            }
        }
        Raw {
            times,
            samples,
            missing: false,
        }
    }

    #[test]
    fn waveform_follows_membership_after_split_not_the_parent_template() {
        let t = tempfile::tempdir().unwrap();
        let mut s = Session::new(raw(), Journal::open(&t.path().join("journal")).unwrap());
        s.dispatch(CurationCommand::Split {
            cluster: ClusterId(0),
            spike_idx: (1..20).step_by(2).collect(),
            new_cluster: ClusterId(99),
            by: Method::Manual,
        })
        .unwrap();
        let a = measure(&s, ClusterId(0)).unwrap();
        let b = measure(&s, ClusterId(1)).unwrap();
        assert_eq!((a.used, b.used), (10, 10));
        assert_eq!((a.peak_channel, b.peak_channel), (0, 1));
        assert!(a.snr > 5.0 && b.snr > 5.0);
        assert!(a.early_late_cos > 0.95 && b.early_late_cos > 0.95);
        assert!(a.shape.unwrap().half_width_ms < b.shape.unwrap().half_width_ms);
        let history = s.history_len();
        let _ = measure(&s, ClusterId(0)).unwrap();
        assert_eq!(s.history_len(), history, "probe must remain read-only");
    }

    #[test]
    fn sampling_is_bounded_deterministic_without_duplicates() {
        for n in [0, 1, 2, 127, 128, 129, 100_000] {
            let indices = sample_indices(n);
            assert_eq!(indices.len(), n.min(MAX_SPIKES));
            assert_eq!(indices, sample_indices(n));
            assert!(indices.windows(2).all(|w| w[0] < w[1]));
            assert!(indices.iter().all(|&i| i < n));
        }
    }

    #[test]
    fn unavailable_raw_never_silently_becomes_template_evidence() {
        let t = tempfile::tempdir().unwrap();
        let mut p = raw();
        p.missing = true;
        let s = Session::new(p, Journal::open(&t.path().join("journal")).unwrap());
        assert!(measure(&s, ClusterId(0))
            .err()
            .unwrap()
            .contains("no template fallback"));
    }
}
