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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_counts_in_range_values() {
        let values = [0u64, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let h = histogram_u64(&values, 0, 10, 5);
        assert_eq!(h, vec![2, 2, 2, 2, 2]);
    }

    #[test]
    fn histogram_skips_out_of_range() {
        let values = [0u64, 5, 10, 15, 20];
        let h = histogram_u64(&values, 5, 15, 2);
        // 5 -> bin 0, 10 -> bin 1, 15 excluded (>= max), 0 and 20 excluded.
        assert_eq!(h, vec![1, 1]);
    }

    #[test]
    fn histogram_degenerate_returns_zeros() {
        assert_eq!(histogram_u64(&[1, 2, 3], 5, 5, 4), vec![0, 0, 0, 0]);
        assert_eq!(histogram_u64(&[1, 2, 3], 0, 10, 0), Vec::<u32>::new());
    }

    #[test]
    fn histogram_clamps_max_value_to_last_bin() {
        // max-min = 100, 1 bin -> all in-range values land in bin 0.
        let values = [0u64, 50, 99];
        let h = histogram_u64(&values, 0, 100, 1);
        assert_eq!(h, vec![3]);
    }

    #[test]
    fn isi_handles_too_few_spikes() {
        assert_eq!(isi_histogram(&[], 100, 4), vec![0, 0, 0, 0]);
        assert_eq!(isi_histogram(&[10], 100, 4), vec![0, 0, 0, 0]);
    }

    #[test]
    fn isi_skips_intervals_at_or_above_max() {
        let spikes = vec![0u64, 10, 200, 210];
        let h = isi_histogram(&spikes, 100, 4);
        // Intervals: 10 -> bin 0, 190 -> skipped (>= 100), 10 -> bin 0
        assert_eq!(h.iter().sum::<u32>(), 2);
        assert_eq!(h[0], 2);
    }

    #[test]
    fn isi_distributes_across_bins() {
        let spikes = vec![0u64, 25, 75];
        // intervals 25 (bin 1 of 4 over [0,100)) and 50 (bin 2)
        let h = isi_histogram(&spikes, 100, 4);
        assert_eq!(h, vec![0, 1, 1, 0]);
    }

    #[test]
    fn isi_degenerate_returns_zeros() {
        assert_eq!(isi_histogram(&[0, 1, 2], 0, 4), vec![0, 0, 0, 0]);
        assert_eq!(isi_histogram(&[0, 1, 2], 100, 0), Vec::<u32>::new());
    }
}
