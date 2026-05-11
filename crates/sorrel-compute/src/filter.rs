//! Causal IIR filtering for trace display when `params.py:hp_filtered=False`.
//!
//! A single 2nd-order Butterworth biquad gets us to ~12 dB/octave roll-off,
//! which is enough to flatten the LFP background under the ~300 Hz spike
//! cutoff phy uses by default. Higher-order filters can be built by chaining
//! sections — each `Biquad` keeps its own state, so a cascade is just a `Vec`.

use std::f32::consts::PI;

/// 2nd-order direct-form-II-transposed IIR section. `a0` is normalised to 1.
#[derive(Copy, Clone, Debug)]
pub struct Biquad {
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub a1: f32,
    pub a2: f32,
}

/// Per-stream filter state. Keep one per signal (per channel) — a single
/// `Biquad` defines the *shape*, the state carries the recursion across calls.
#[derive(Copy, Clone, Debug, Default)]
pub struct BiquadState {
    pub s1: f32,
    pub s2: f32,
}

impl Biquad {
    /// 2nd-order Butterworth high-pass, designed via the bilinear transform.
    /// Equivalent to phy's `scipy.signal.butter(2, fc/(fs/2), 'highpass')`
    /// when serialised as a single biquad.
    ///
    /// # Examples
    ///
    /// ```
    /// use sorrel_compute::{Biquad, BiquadState};
    ///
    /// let f = Biquad::butterworth_hp(300.0, 30_000.0);
    /// let mut state = BiquadState::default();
    ///
    /// // A constant DC signal should settle close to zero through a HP filter.
    /// for _ in 0..2048 {
    ///     let _ = f.step(1000.0, &mut state);
    /// }
    /// let settled = f.step(1000.0, &mut state);
    /// assert!(settled.abs() < 0.01);
    /// ```
    pub fn butterworth_hp(cutoff_hz: f32, sample_rate: f32) -> Self {
        debug_assert!(sample_rate > 0.0);
        debug_assert!(cutoff_hz > 0.0 && cutoff_hz < sample_rate * 0.5);

        // RBJ cookbook coefficients with Q = 1/sqrt(2) (Butterworth).
        let w0 = 2.0 * PI * cutoff_hz / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let q = std::f32::consts::FRAC_1_SQRT_2;
        let alpha = sin_w0 / (2.0 * q);

        let b0 = (1.0 + cos_w0) * 0.5;
        let b1 = -(1.0 + cos_w0);
        let b2 = (1.0 + cos_w0) * 0.5;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        let inv_a0 = 1.0 / a0;
        Self {
            b0: b0 * inv_a0,
            b1: b1 * inv_a0,
            b2: b2 * inv_a0,
            a1: a1 * inv_a0,
            a2: a2 * inv_a0,
        }
    }

    /// Step a single sample through the filter, returning the filtered value
    /// and threading the state forward.
    #[inline]
    pub fn step(&self, x: f32, state: &mut BiquadState) -> f32 {
        let y = self.b0 * x + state.s1;
        state.s1 = self.b1 * x - self.a1 * y + state.s2;
        state.s2 = self.b2 * x - self.a2 * y;
        y
    }

    /// Apply the filter to `input`, writing into `output`. Uses fresh state;
    /// for stable streaming filtering across windows the caller should keep
    /// a `BiquadState` alive between calls and use [`Self::step`] directly.
    pub fn apply<S>(&self, input: &[S], output: &mut Vec<f32>)
    where
        S: Copy + Into<f32>,
    {
        output.clear();
        output.reserve(input.len());
        let mut state = BiquadState::default();
        for &x in input {
            output.push(self.step(x.into(), &mut state));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DC component (constant signal) should be killed off in steady-state.
    #[test]
    fn high_pass_kills_dc_in_steady_state() {
        let f = Biquad::butterworth_hp(300.0, 30_000.0);
        let mut out = Vec::new();
        let input = vec![1000.0_f32; 4096];
        f.apply(&input, &mut out);
        // After enough samples to settle (~50 ms here), the output should be
        // close to zero. Tolerate the unsettled tail conservatively.
        let tail = &out[2048..];
        let max_abs = tail.iter().copied().fold(0.0_f32, |m, v| m.max(v.abs()));
        assert!(
            max_abs < 1e-2,
            "DC residual {max_abs} should be ~0 after settle"
        );
    }

    /// A 50 Hz tone should be heavily attenuated by a 300 Hz HP filter.
    #[test]
    fn high_pass_attenuates_below_cutoff() {
        let fs = 30_000.0;
        let f = Biquad::butterworth_hp(300.0, fs);
        let n = 8192;
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 50.0 * i as f32 / fs).sin())
            .collect();
        let mut out = Vec::new();
        f.apply(&input, &mut out);
        // Skip the unsettled head.
        let tail = &out[1024..];
        let rms_in: f32 =
            (input[1024..].iter().map(|v| v * v).sum::<f32>() / tail.len() as f32).sqrt();
        let rms_out: f32 = (tail.iter().map(|v| v * v).sum::<f32>() / tail.len() as f32).sqrt();
        // 50 Hz is ~1.5 octaves below the cutoff for a 2nd-order HP — expect
        // strong attenuation, well below 10% of input RMS.
        assert!(
            rms_out < 0.1 * rms_in,
            "rms_out={rms_out} not attenuated below 10% of rms_in={rms_in}"
        );
    }

    /// A frequency well above cutoff should pass through nearly unchanged
    /// after settling.
    #[test]
    fn high_pass_passes_well_above_cutoff() {
        let fs = 30_000.0;
        let f = Biquad::butterworth_hp(300.0, fs);
        let n = 8192;
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 3000.0 * i as f32 / fs).sin())
            .collect();
        let mut out = Vec::new();
        f.apply(&input, &mut out);
        let tail = &out[1024..];
        let rms_in: f32 =
            (input[1024..].iter().map(|v| v * v).sum::<f32>() / tail.len() as f32).sqrt();
        let rms_out: f32 = (tail.iter().map(|v| v * v).sum::<f32>() / tail.len() as f32).sqrt();
        assert!(
            rms_out > 0.85 * rms_in,
            "rms_out={rms_out} not preserved above 85% of rms_in={rms_in}"
        );
    }

    /// Stepping samples one-by-one should match `apply` exactly.
    #[test]
    fn step_matches_apply() {
        let f = Biquad::butterworth_hp(300.0, 30_000.0);
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin() * 1000.0).collect();
        let mut a = Vec::new();
        f.apply(&input, &mut a);

        let mut state = BiquadState::default();
        let b: Vec<f32> = input.iter().map(|&x| f.step(x, &mut state)).collect();
        for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
            assert!((av - bv).abs() < 1e-6, "mismatch at {i}: {av} vs {bv}");
        }
    }

    /// Linearity: filter(a*x + b*y) == a*filter(x) + b*filter(y).
    #[test]
    fn filter_is_linear() {
        let f = Biquad::butterworth_hp(300.0, 30_000.0);
        let n = 1024;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin() * 1000.0).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.13).cos() * 500.0 + 200.0)
            .collect();
        let a = 0.7_f32;
        let b = -1.3_f32;

        let mixed: Vec<f32> = x.iter().zip(&y).map(|(xi, yi)| a * xi + b * yi).collect();

        let mut fx = Vec::new();
        let mut fy = Vec::new();
        let mut fmixed = Vec::new();
        f.apply(&x, &mut fx);
        f.apply(&y, &mut fy);
        f.apply(&mixed, &mut fmixed);

        for i in 0..n {
            let combined = a * fx[i] + b * fy[i];
            // Filter operates on f32; ~ULP differences accumulate over the
            // 1024-sample run. 1% relative tolerance is fine for a linearity
            // sanity check.
            let scale = combined.abs().max(fmixed[i].abs()).max(1.0);
            let diff = (combined - fmixed[i]).abs();
            assert!(
                diff / scale < 1e-2,
                "linearity violated at {i}: {combined} vs {} (rel {})",
                fmixed[i],
                diff / scale,
            );
        }
    }

    /// Time-invariance: shifting input forward shifts output forward.
    /// Approx-only because the filter starts from zero state both times,
    /// so the unshifted leading window is unsettled. We compare the tails.
    #[test]
    fn filter_is_approximately_time_invariant() {
        let f = Biquad::butterworth_hp(300.0, 30_000.0);
        let n = 2048;
        let pad = 64;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).sin() * 1000.0).collect();
        let mut x_shifted = vec![0.0_f32; n + pad];
        x_shifted[pad..].copy_from_slice(&x);

        let mut fx = Vec::new();
        let mut fxs = Vec::new();
        f.apply(&x, &mut fx);
        f.apply(&x_shifted, &mut fxs);

        // Compare fx[1024..] with fxs[1024+pad..]; both should be settled.
        for i in 1024..n {
            let a = fx[i];
            let b = fxs[i + pad];
            assert!(
                (a - b).abs() < 1e-3,
                "time-shift mismatch at {i}: {a} vs {b}",
            );
        }
    }

    /// Coefficients sum: Butterworth HP has DC gain 0, so the sum of the
    /// numerator coefficients (b0 + b1 + b2) must be zero.
    #[test]
    fn high_pass_dc_gain_is_zero_via_coefficients() {
        let f = Biquad::butterworth_hp(300.0, 30_000.0);
        let dc_gain_numerator = f.b0 + f.b1 + f.b2;
        assert!(
            dc_gain_numerator.abs() < 1e-5,
            "b0+b1+b2={dc_gain_numerator} should be ~0 for HP",
        );
    }

    /// Different cutoffs produce *different* filters (sanity).
    #[test]
    fn different_cutoffs_yield_different_coefficients() {
        let a = Biquad::butterworth_hp(300.0, 30_000.0);
        let b = Biquad::butterworth_hp(600.0, 30_000.0);
        assert!((a.a1 - b.a1).abs() > 1e-3);
        assert!((a.b0 - b.b0).abs() > 1e-3);
    }
}
