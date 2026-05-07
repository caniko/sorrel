/// Largest-Triangle-Three-Buckets downsampling. Operates on densely sampled
/// `i16` traces and emits `(x, y)` `f32` pairs ready for upload as vertex
/// data. Generic over `Sample: Into<f32> + Copy` so the same function compiles
/// for `i16`, `f32`, etc., with no dispatch.
///
/// # Examples
///
/// ```
/// use sorrel_compute::lttb_downsample;
///
/// // 1000 samples, downsample to 50 representative points.
/// let samples: Vec<i16> = (0..1000).map(|i| (i as i16) % 200 - 100).collect();
/// let pts = lttb_downsample::<i16>(&samples, 50, 0.0, 1.0);
/// assert_eq!(pts.len(), 50);
///
/// // First and last points are always preserved.
/// assert_eq!(pts[0], [0.0, samples[0] as f32]);
/// assert_eq!(pts.last().unwrap()[0], 999.0);
/// ```
pub fn lttb_downsample<S>(samples: &[S], threshold: usize, x_origin: f32, x_step: f32) -> Vec<[f32; 2]>
where
    S: Copy + Into<f32>,
{
    let n = samples.len();
    if threshold >= n || threshold < 3 {
        return samples
            .iter()
            .enumerate()
            .map(|(i, s)| [x_origin + x_step * i as f32, (*s).into()])
            .collect();
    }

    let mut out = Vec::with_capacity(threshold);
    let bucket_size = (n - 2) as f32 / (threshold - 2) as f32;

    let mut a = 0usize;
    out.push([x_origin, samples[0].into()]);

    for i in 0..(threshold - 2) {
        // Next bucket average (point B).
        let avg_start = ((i as f32 + 1.0) * bucket_size).floor() as usize + 1;
        let avg_end = (((i as f32 + 2.0) * bucket_size).floor() as usize + 1).min(n);
        let avg_count = (avg_end - avg_start).max(1);
        let mut avg_x = 0.0f32;
        let mut avg_y = 0.0f32;
        for (j, sample) in samples.iter().enumerate().take(avg_end).skip(avg_start) {
            avg_x += x_origin + x_step * j as f32;
            avg_y += (*sample).into();
        }
        avg_x /= avg_count as f32;
        avg_y /= avg_count as f32;

        // Current bucket range (point A's bucket).
        let range_start = ((i as f32) * bucket_size).floor() as usize + 1;
        let range_end = (((i as f32 + 1.0) * bucket_size).floor() as usize + 1).min(n);

        let pa_x = x_origin + x_step * a as f32;
        let pa_y: f32 = samples[a].into();

        let mut max_area = -1.0f32;
        let mut max_idx = range_start;
        for (j, sample) in samples.iter().enumerate().take(range_end).skip(range_start) {
            let x = x_origin + x_step * j as f32;
            let y: f32 = (*sample).into();
            let area = ((pa_x - avg_x) * (y - pa_y) - (pa_x - x) * (avg_y - pa_y)).abs() * 0.5;
            if area > max_area {
                max_area = area;
                max_idx = j;
            }
        }
        out.push([x_origin + x_step * max_idx as f32, samples[max_idx].into()]);
        a = max_idx;
    }

    out.push([x_origin + x_step * (n - 1) as f32, samples[n - 1].into()]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_all_points_when_threshold_too_large() {
        let samples: Vec<i16> = vec![1, 2, 3, 4, 5];
        let out = lttb_downsample::<i16>(&samples, 100, 0.0, 1.0);
        assert_eq!(out.len(), 5);
        assert_eq!(out[0], [0.0, 1.0]);
        assert_eq!(out[4], [4.0, 5.0]);
    }

    #[test]
    fn returns_all_points_when_threshold_below_three() {
        let samples: Vec<i16> = vec![1, 2, 3, 4, 5];
        let out = lttb_downsample::<i16>(&samples, 2, 0.0, 1.0);
        assert_eq!(out.len(), 5);
    }

    #[test]
    fn produces_threshold_points_and_preserves_endpoints() {
        let samples: Vec<i16> = (0..1000).map(|i| ((i * 13) % 200 - 100) as i16).collect();
        let out = lttb_downsample::<i16>(&samples, 50, 0.0, 1.0);
        assert_eq!(out.len(), 50);
        assert_eq!(out[0], [0.0, samples[0] as f32]);
        let last = *out.last().unwrap();
        assert_eq!(last[0], (samples.len() - 1) as f32);
        assert_eq!(last[1], samples[samples.len() - 1] as f32);
    }

    #[test]
    fn x_origin_and_step_applied() {
        let samples: Vec<i16> = vec![0, 10, 20, 30, 40, 50, 60, 70, 80, 90];
        let out = lttb_downsample::<i16>(&samples, 5, 100.0, 0.5);
        assert_eq!(out[0][0], 100.0);
        let last = *out.last().unwrap();
        assert_eq!(last[0], 100.0 + 0.5 * 9.0);
    }

    #[test]
    fn empty_input_with_invalid_threshold_returns_empty() {
        let samples: Vec<i16> = vec![];
        let out = lttb_downsample::<i16>(&samples, 0, 0.0, 1.0);
        assert!(out.is_empty());
    }

    /// Property: for any input with n > threshold ≥ 3, the output length
    /// equals threshold exactly.
    #[test]
    fn output_length_equals_threshold_when_threshold_in_range() {
        let samples: Vec<i16> = (0..500).map(|i| (i as i16).wrapping_mul(7)).collect();
        for threshold in [3, 5, 50, 100, 250, 499] {
            let out = lttb_downsample::<i16>(&samples, threshold, 0.0, 1.0);
            assert_eq!(out.len(), threshold, "threshold {threshold} produced {} points", out.len());
        }
    }

    /// Property: the x-coordinates of the output are strictly monotonic
    /// non-decreasing — LTTB never reorders samples.
    #[test]
    fn output_x_is_monotonic_non_decreasing() {
        let samples: Vec<i16> = (0..1000).map(|i| ((i * 17) % 256 - 128) as i16).collect();
        let out = lttb_downsample::<i16>(&samples, 73, 0.0, 1.0);
        for w in out.windows(2) {
            assert!(w[1][0] >= w[0][0], "x went backward: {} < {}", w[1][0], w[0][0]);
        }
    }

    /// Property: every output y-value is one of the input y-values
    /// (LTTB picks samples — it doesn't synthesise new values).
    #[test]
    fn output_y_values_come_from_input() {
        let samples: Vec<i16> = (0..400).map(|i| ((i * 31) % 1000) as i16).collect();
        let input_set: std::collections::HashSet<i32> =
            samples.iter().map(|&v| v as i32).collect();
        let out = lttb_downsample::<i16>(&samples, 40, 0.0, 1.0);
        for [_, y] in &out {
            assert!(
                input_set.contains(&(*y as i32)),
                "y={y} not in input set",
            );
        }
    }

    /// Constant signal: every chosen point should have the same y value.
    #[test]
    fn constant_signal_produces_constant_output() {
        let samples: Vec<i16> = vec![42; 200];
        let out = lttb_downsample::<i16>(&samples, 30, 0.0, 1.0);
        for [_, y] in &out {
            assert_eq!(*y, 42.0);
        }
    }

    /// Strictly monotone input: the output triangle area metric should
    /// pick the endpoints of each bucket — but at minimum, the middle
    /// pick must lie within [bucket_min, bucket_max] (which is always
    /// true here since input is monotone).
    #[test]
    fn monotone_input_output_is_monotone() {
        let samples: Vec<i16> = (0..256).map(|i| i as i16).collect();
        let out = lttb_downsample::<i16>(&samples, 32, 0.0, 1.0);
        for w in out.windows(2) {
            assert!(w[1][1] >= w[0][1], "non-monotone: {} -> {}", w[0][1], w[1][1]);
        }
    }

    /// Stress: 1M-sample input downsampled to 1024. Smoke-tests that the
    /// algorithm doesn't degenerate on realistic Neuropixels-grade buffers.
    /// `#[ignore]` so it doesn't run by default; invoke with
    /// `cargo test --release -- --ignored`.
    #[test]
    #[ignore]
    fn stress_one_million_samples_to_one_thousand_points() {
        let samples: Vec<i16> = (0..1_000_000)
            .map(|i| ((i as i32 * 31) % 30_000 - 15_000) as i16)
            .collect();
        let out = lttb_downsample::<i16>(&samples, 1024, 0.0, 1.0);
        assert_eq!(out.len(), 1024);
    }
}
