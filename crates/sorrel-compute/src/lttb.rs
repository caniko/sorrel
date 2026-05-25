/// Largest-Triangle-Three-Buckets downsampling. Operates on densely sampled
/// `i16` traces and emits `(x, y)` `f32` pairs ready for upload as vertex
/// data. Generic over `Sample: Into<f32> + Copy` so the same function compiles
/// for `i16`, `f32`, etc., with no dispatch.
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
