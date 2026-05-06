use sorrel_io::SampleIndex;

/// Linear histogram over an `i64`/`u64`-shaped axis. Generic so the compiler
/// inlines bound checks.
pub fn histogram_u64(values: &[u64], min: u64, max: u64, bins: usize) -> Vec<u32> {
    let mut h = vec![0u32; bins];
    if max <= min || bins == 0 {
        return h;
    }
    let span = (max - min) as f64;
    let inv = bins as f64 / span;
    for &v in values {
        if v < min || v >= max {
            continue;
        }
        let idx = (((v - min) as f64) * inv) as usize;
        let idx = idx.min(bins - 1);
        h[idx] += 1;
    }
    h
}

/// Inter-spike-interval histogram in samples, capped at `max_samples`.
pub fn isi_histogram(spikes: &[SampleIndex], max_samples: u64, bins: usize) -> Vec<u32> {
    if spikes.len() < 2 || bins == 0 || max_samples == 0 {
        return vec![0; bins];
    }
    let mut h = vec![0u32; bins];
    let inv = bins as f64 / max_samples as f64;
    for w in spikes.windows(2) {
        let dt = w[1].saturating_sub(w[0]);
        if dt >= max_samples {
            continue;
        }
        let idx = ((dt as f64) * inv) as usize;
        h[idx.min(bins - 1)] += 1;
    }
    h
}
