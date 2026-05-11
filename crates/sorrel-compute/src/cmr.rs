//! Common median referencing.
//!
//! At each time step, subtract the median across the supplied channels from
//! every channel. Phy applies CMR before display when the recording isn't
//! pre-filtered; for multi-shank probes the caller should run this per-shank.

/// Subtract the per-time-step median from each sample. `samples` is laid out
/// time-major: `[t0_ch0, t0_ch1, ..., t0_chN-1, t1_ch0, ...]`. Multi-shank
/// callers should slice their input per-shank and call this once per shank.
///
/// # Examples
///
/// ```
/// use sorrel_compute::subtract_channel_median;
///
/// // 3 channels × 2 time steps. Median of [1,5,3] = 3, of [10,100,50] = 50.
/// let mut s = vec![1.0_f32, 5.0, 3.0, 10.0, 100.0, 50.0];
/// subtract_channel_median(&mut s, 3);
/// assert_eq!(s, vec![-2.0, 2.0, 0.0, -40.0, 50.0, 0.0]);
/// ```
pub fn subtract_channel_median(samples: &mut [f32], n_channels: usize) {
    if n_channels == 0 || samples.is_empty() {
        return;
    }
    debug_assert_eq!(samples.len() % n_channels, 0);
    let mut row: Vec<f32> = vec![0.0; n_channels];
    for chunk in samples.chunks_exact_mut(n_channels) {
        row.copy_from_slice(chunk);
        let m = median_in_place(&mut row);
        for v in chunk.iter_mut() {
            *v -= m;
        }
    }
}

/// Exact median of a small slice via partial sort. We accept that the input
/// gets shuffled — callers pass a scratch buffer.
fn median_in_place(buf: &mut [f32]) -> f32 {
    debug_assert!(!buf.is_empty());
    let n = buf.len();
    // `select_nth_unstable_by(k)` lands the kth smallest at index k and
    // splits the slice into `(left = strictly_less, m, right = strictly_greater_or_equal)`.
    // For even n we average the two middle values: lower = max of the left
    // partition (after picking the upper middle at k = n/2).
    let mid = n / 2;
    let (left, m, _right) = buf.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
    if n % 2 == 1 {
        *m
    } else {
        let lower = left.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        0.5 * (lower + *m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_is_a_noop() {
        let mut s: Vec<f32> = vec![];
        subtract_channel_median(&mut s, 4);
        assert!(s.is_empty());
    }

    #[test]
    fn zero_channels_is_a_noop() {
        let mut s = vec![1.0_f32, 2.0, 3.0];
        subtract_channel_median(&mut s, 0);
        assert_eq!(s, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn single_time_step_subtracts_median_from_each_channel() {
        let mut s = vec![1.0_f32, 5.0, 3.0]; // median = 3.0
        subtract_channel_median(&mut s, 3);
        assert_eq!(s, vec![-2.0, 2.0, 0.0]);
    }

    #[test]
    fn even_channel_count_uses_average_of_two_middles() {
        // [1, 2, 3, 4] -> median = 2.5
        let mut s = vec![1.0_f32, 2.0, 3.0, 4.0];
        subtract_channel_median(&mut s, 4);
        assert_eq!(s, vec![-1.5, -0.5, 0.5, 1.5]);
    }

    #[test]
    fn multi_time_step_works_per_row_independently() {
        // Two time steps, 3 channels:
        //   t=0: [1, 5, 3] -> median 3 -> [-2, 2, 0]
        //   t=1: [10, 100, 50] -> median 50 -> [-40, 50, 0]
        let mut s = vec![1.0_f32, 5.0, 3.0, 10.0, 100.0, 50.0];
        subtract_channel_median(&mut s, 3);
        assert_eq!(s, vec![-2.0, 2.0, 0.0, -40.0, 50.0, 0.0]);
    }

    #[test]
    fn constant_dc_signal_is_zeroed_per_channel() {
        // If every channel reads the same value at each time step, CMR
        // should fully zero them.
        let mut s = vec![7.0_f32; 3 * 16];
        subtract_channel_median(&mut s, 3);
        for v in s {
            assert_eq!(v, 0.0);
        }
    }

    /// Property: after CMR, each row's median is 0 (within rounding).
    #[test]
    fn after_cmr_each_row_has_zero_median() {
        let n_channels = 7;
        let n_samples = 100;
        let mut s: Vec<f32> = (0..n_channels * n_samples)
            .map(|i| ((i * 17) % 213) as f32 - 100.0)
            .collect();
        subtract_channel_median(&mut s, n_channels);
        for chunk in s.chunks_exact(n_channels) {
            let mut row = chunk.to_vec();
            row.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let m = row[row.len() / 2];
            // Odd n_channels -> exact zero. Use small epsilon for rounding.
            assert!(m.abs() < 1e-5, "row median = {m} after CMR, expected ~0",);
        }
    }

    /// Verify against an oracle that uses full sort.
    #[test]
    fn matches_oracle_via_full_sort() {
        let n_channels = 5;
        let _n_samples = 4;
        let mut s = vec![
            3.0_f32, 1.0, 4.0, 1.0, 5.0, // row 0; sorted: 1,1,3,4,5; median 3
            9.0, 2.0, 6.0, 5.0, 3.0, // row 1; sorted: 2,3,5,6,9; median 5
            5.0, 8.0, 9.0, 7.0, 9.0, // row 2; sorted: 5,7,8,9,9; median 8
            3.0, 2.0, 3.0, 8.0, 4.0,
        ]; // row 3; sorted: 2,3,3,4,8; median 3
        let expected = [
            0.0, -2.0, 1.0, -2.0, 2.0, 4.0, -3.0, 1.0, 0.0, -2.0, -3.0, 0.0, 1.0, -1.0, 1.0, 0.0,
            -1.0, 0.0, 5.0, 1.0,
        ];
        subtract_channel_median(&mut s, n_channels);
        for (i, (got, want)) in s.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-6, "i={i}: got {got}, want {want}",);
        }
    }

    /// CMR doesn't reorder samples within a row (median is centring only).
    #[test]
    fn preserves_row_signature_under_constant_offset() {
        let n_channels = 4;
        let mut s = vec![1.0_f32, 2.0, 3.0, 4.0]; // baseline
        let mut s2 = vec![101.0_f32, 102.0, 103.0, 104.0]; // shift by 100
        subtract_channel_median(&mut s, n_channels);
        subtract_channel_median(&mut s2, n_channels);
        for (a, b) in s.iter().zip(s2.iter()) {
            assert!((a - b).abs() < 1e-6, "row signature changed under DC shift");
        }
    }
}
