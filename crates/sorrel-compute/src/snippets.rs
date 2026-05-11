//! Per-spike snippet extraction.
//!
//! Given a window-around-spike size and a list of spike indices, gather a
//! `(n_spikes, snippet_len)` view from a contiguous channel-major trace.
//! The waveform view, mean-template builder, and PC-feature compute all
//! start here.

use sorrel_io::SampleIndex;

/// Extract `pre + post + 1` samples around each spike from a single
/// channel's already-flattened trace `channel_samples` whose first sample
/// corresponds to global time `start_sample`.
///
/// Out-of-range spikes (those whose snippet would extend past either edge
/// of `channel_samples`) are silently skipped — callers use `out.len()` to
/// detect truncation against the input.
pub fn extract_snippets_single_channel(
    channel_samples: &[f32],
    start_sample: SampleIndex,
    spike_times: &[SampleIndex],
    pre: u32,
    post: u32,
) -> Vec<Vec<f32>> {
    let snippet_len = (pre + post + 1) as usize;
    let mut out = Vec::new();
    if channel_samples.is_empty() || snippet_len == 0 {
        return out;
    }
    let pre = pre as i64;
    let post = post as i64;
    let len_i = channel_samples.len() as i64;
    let start_i = start_sample.as_i64();
    for &t in spike_times {
        let idx = t.as_i64() - start_i;
        let lo = idx - pre;
        let hi = idx + post + 1;
        if lo < 0 || hi > len_i {
            continue;
        }
        out.push(channel_samples[lo as usize..hi as usize].to_vec());
    }
    out
}

/// Average each sample position across `snippets` to produce a single
/// "mean waveform" of length `snippet_len`. Returns an empty `Vec` when no
/// snippets are supplied.
pub fn mean_snippet(snippets: &[Vec<f32>]) -> Vec<f32> {
    if snippets.is_empty() {
        return Vec::new();
    }
    let len = snippets[0].len();
    let mut sum = vec![0.0f64; len];
    let mut n = 0usize;
    for s in snippets {
        if s.len() != len {
            continue;
        }
        for (i, &v) in s.iter().enumerate() {
            sum[i] += v as f64;
        }
        n += 1;
    }
    if n == 0 {
        return Vec::new();
    }
    sum.into_iter().map(|x| (x / n as f64) as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn si(xs: impl IntoIterator<Item = u64>) -> Vec<SampleIndex> {
        xs.into_iter().map(SampleIndex).collect()
    }

    #[test]
    fn extracts_centred_window_for_each_in_range_spike() {
        // Trace samples at global times [0..10], values = time * 10.
        let trace: Vec<f32> = (0..10).map(|t| (t as f32) * 10.0).collect();
        let spikes = si([3u64, 5, 7]);
        let snips = extract_snippets_single_channel(&trace, SampleIndex(0), &spikes, 1, 1);
        assert_eq!(snips.len(), 3);
        assert_eq!(snips[0], vec![20.0, 30.0, 40.0]);
        assert_eq!(snips[1], vec![40.0, 50.0, 60.0]);
        assert_eq!(snips[2], vec![60.0, 70.0, 80.0]);
    }

    #[test]
    fn skips_spikes_whose_window_falls_off_the_edges() {
        let trace: Vec<f32> = (0..10).map(|t| t as f32).collect();
        // Spikes at 0 (lo would be -2) and at 9 (hi would be 12).
        let spikes = si([0u64, 9]);
        let snips = extract_snippets_single_channel(&trace, SampleIndex(0), &spikes, 2, 2);
        assert!(snips.is_empty(), "both edges should be rejected");
    }

    #[test]
    fn honours_start_sample_offset() {
        // `start_sample = 100` means the slice represents samples 100..110.
        let trace: Vec<f32> = (0..10).map(|t| t as f32).collect();
        let spikes = si([102u64]);
        let snips = extract_snippets_single_channel(&trace, SampleIndex(100), &spikes, 1, 1);
        assert_eq!(snips, vec![vec![1.0, 2.0, 3.0]]);
    }

    #[test]
    fn empty_trace_returns_empty() {
        let snips = extract_snippets_single_channel(&[], SampleIndex(0), &si([5u64]), 1, 1);
        assert!(snips.is_empty());
    }

    #[test]
    fn mean_snippet_averages_aligned_samples() {
        let snips = vec![
            vec![1.0_f32, 2.0, 3.0],
            vec![3.0_f32, 4.0, 5.0],
            vec![5.0_f32, 6.0, 7.0],
        ];
        let m = mean_snippet(&snips);
        assert_eq!(m, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn mean_snippet_skips_mismatched_lengths() {
        let snips = vec![vec![1.0_f32, 2.0, 3.0], vec![4.0_f32, 5.0]];
        let m = mean_snippet(&snips);
        assert_eq!(m, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn mean_snippet_empty_returns_empty() {
        let m = mean_snippet(&[]);
        assert!(m.is_empty());
    }

    /// Property: every extracted snippet has the same length, equal to
    /// `pre + post + 1`.
    #[test]
    fn extracted_snippet_length_is_pre_plus_post_plus_one() {
        let trace: Vec<f32> = (0..1000).map(|t| t as f32).collect();
        let spikes = si((50..950).step_by(50).map(|t| t as u64));
        for &(pre, post) in &[(0u32, 0u32), (5, 5), (10, 30), (30, 10)] {
            let snips = extract_snippets_single_channel(&trace, SampleIndex(0), &spikes, pre, post);
            for s in &snips {
                assert_eq!(
                    s.len(),
                    (pre + post + 1) as usize,
                    "snippet length wrong for pre={pre} post={post}",
                );
            }
        }
    }

    /// Property: the centre sample of a snippet equals the trace value at
    /// the spike time.
    #[test]
    fn centre_sample_equals_trace_at_spike_time() {
        let trace: Vec<f32> = (0..100).map(|t| (t as f32) * 7.0).collect();
        let spikes = si([20u64, 50, 80]);
        let pre = 5u32;
        let post = 5u32;
        let snips = extract_snippets_single_channel(&trace, SampleIndex(0), &spikes, pre, post);
        for (i, snip) in snips.iter().enumerate() {
            let centre = snip[pre as usize];
            let expected = trace[spikes[i].idx()];
            assert_eq!(centre, expected);
        }
    }

    /// Property: mean of N identical snippets equals the snippet itself.
    #[test]
    fn mean_of_n_identical_snippets_equals_the_snippet() {
        let snippet = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let snips = vec![snippet.clone(); 17];
        let m = mean_snippet(&snips);
        for (got, want) in m.iter().zip(snippet.iter()) {
            assert!((got - want).abs() < 1e-6);
        }
    }

    /// Property: mean linearity — mean(snip + offset) == mean(snip) + offset.
    #[test]
    fn mean_snippet_is_linear_under_constant_offset() {
        let base = [1.0_f32, 2.0, 3.0];
        let mut shifted = Vec::new();
        for k in 0..5 {
            shifted.push(base.iter().map(|v| v + k as f32 * 10.0).collect());
        }
        let m = mean_snippet(&shifted);
        // Mean of [0, 10, 20, 30, 40] = 20. So m[i] = base[i] + 20.
        for (got, want) in m.iter().zip(base.iter()) {
            assert!((got - (want + 20.0)).abs() < 1e-5);
        }
    }
}
