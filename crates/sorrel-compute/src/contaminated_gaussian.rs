//! One full-covariance Gaussian cloud plus a broader, concentric contaminant.
//! Unlike a diagonal mixture's whole-component split, the posterior boundary
//! can pass through a correlated cloud's tails. This is a boundary proposal,
//! not evidence that the central cloud is a neuron.

use crate::linalg::{covariance, invert, mahalanobis_sq, row_mean, Sym};

const MAX_ITER: usize = 80;
const MAX_D: usize = 12;

#[derive(Clone, Debug)]
pub struct ContaminatedGaussian {
    d: usize,
    mean: Vec<f64>,
    inverse: Sym,
    log_det: f64,
    /// Fitted narrow-component probability (constrained to a majority).
    pub clean_weight: f64,
    /// Broad covariance = this factor times the narrow covariance.
    pub inflation: f64,
    pub log_lik: f64,
    /// BIC(single Gaussian) - BIC(contaminated Gaussian); positive favors tails.
    pub bic_gain: f64,
}

fn prepare(mut cov: Sym, ridge: f64) -> Option<(Sym, f64)> {
    cov.ridge(ridge);
    // Cholesky supplies the determinant and rejects non-positive covariance.
    let d = cov.d;
    let mut lower = vec![0.0; d * d];
    let mut log_det = 0.0;
    for i in 0..d {
        for j in 0..=i {
            let x = cov.get(i, j)
                - (0..j)
                    .map(|k| lower[i * d + k] * lower[j * d + k])
                    .sum::<f64>();
            if i == j {
                if !x.is_finite() || x <= 0.0 {
                    return None;
                }
                lower[i * d + j] = x.sqrt();
                log_det += x.ln();
            } else {
                lower[i * d + j] = x / lower[j * d + j];
            }
        }
    }
    invert(&mut cov, 1e-15)?;
    cov.data
        .iter()
        .all(|x| x.is_finite())
        .then_some((cov, log_det))
}

impl ContaminatedGaussian {
    fn terms(&self, row: &[f64]) -> (f64, f64) {
        let q = mahalanobis_sq(row, &self.mean, &self.inverse);
        let base = self.log_det + self.d as f64 * (2.0 * std::f64::consts::PI).ln();
        (
            self.clean_weight.ln() - 0.5 * (base + q),
            (-self.clean_weight).ln_1p()
                - 0.5 * (base + self.d as f64 * self.inflation.ln() + q / self.inflation),
        )
    }

    /// Narrow-cloud posterior. Input shape/finite values are checked; no
    /// arbitrary spike subset is optimized against refractory violations.
    pub fn clean_posteriors(&self, data: &[f32]) -> Option<Vec<f64>> {
        if !data.len().is_multiple_of(self.d) || data.iter().any(|x| !x.is_finite()) {
            return None;
        }
        Some(
            data.chunks_exact(self.d)
                .map(|r| {
                    let row: Vec<_> = r.iter().map(|&v| v as f64).collect();
                    let (a, b) = self.terms(&row);
                    let delta = b - a;
                    if delta > 0.0 {
                        let z = (-delta).exp();
                        z / (1.0 + z)
                    } else {
                        1.0 / (1.0 + delta.exp())
                    }
                })
                .collect(),
        )
    }
}

/// Fit a majority cloud with a broad concentric contaminant. Callers bound
/// the fitting sample; runtime is O(rows * d² * iterations), d <= 12. Two
/// fixed covariance-inflation starts (4 and 25) avoid a single local start;
/// the likelihood, not human labels or ISIs, selects between them.
pub fn fit_contaminated_gaussian(data: &[f32], d: usize) -> Option<ContaminatedGaussian> {
    if d == 0
        || d > MAX_D
        || !data.len().is_multiple_of(d)
        || data.len() / d < (4 * d).max(100)
        || data.iter().any(|x| !x.is_finite())
    {
        return None;
    }
    let n = data.len() / d;
    let rows: Vec<_> = data.iter().map(|&x| x as f64).collect();
    let mean = row_mean(&rows, n, d);
    let mut cov = covariance(&rows, n, d);
    // Convert sample covariance to the ML covariance used in likelihood/BIC.
    cov.data
        .iter_mut()
        .for_each(|x| *x *= (n - 1) as f64 / n as f64);
    let variance = (0..d).map(|j| cov.get(j, j)).sum::<f64>() / d as f64;
    if !variance.is_finite() || variance <= 1e-12 {
        return None;
    }
    let ridge = variance * 1e-6;
    let (inverse, log_det) = prepare(cov.clone(), ridge)?;
    let single_ll: f64 = rows
        .chunks_exact(d)
        .map(|r| {
            -0.5 * (d as f64 * (2.0 * std::f64::consts::PI).ln()
                + log_det
                + mahalanobis_sq(r, &mean, &inverse))
        })
        .sum();
    [4.0, 25.0]
        .into_iter()
        .filter_map(|inflation| {
            let mut narrow = cov.clone();
            narrow
                .data
                .iter_mut()
                .for_each(|x| *x /= 0.8 + 0.2 * inflation);
            let (inverse, log_det) = prepare(narrow, ridge)?;
            let mut model = ContaminatedGaussian {
                d,
                mean: mean.clone(),
                inverse,
                log_det,
                clean_weight: 0.8,
                inflation,
                log_lik: f64::NEG_INFINITY,
                bic_gain: 0.0,
            };
            let mut previous = f64::NEG_INFINITY;
            for iteration in 0..=MAX_ITER {
                let mut likelihood = 0.0;
                let mut posteriors = Vec::with_capacity(n);
                for row in rows.chunks_exact(d) {
                    let (a, b) = model.terms(row);
                    let m = a.max(b);
                    let sum = (a - m).exp() + (b - m).exp();
                    likelihood += m + sum.ln();
                    posteriors.push((a - m).exp() / sum);
                }
                if !likelihood.is_finite() {
                    return None;
                }
                model.log_lik = likelihood;
                if iteration == MAX_ITER
                    || (likelihood - previous).abs() < 1e-7 * likelihood.abs().max(1.0)
                {
                    break;
                }
                previous = likelihood;
                let total_clean: f64 = posteriors.iter().sum();
                let broad = n as f64 - total_clean;
                if broad <= 1e-9 {
                    break;
                }
                // Conditional maximizations: each increases the same EM objective.
                let inflation = rows
                    .chunks_exact(d)
                    .zip(&posteriors)
                    .map(|(r, &p)| (1.0 - p) * mahalanobis_sq(r, &model.mean, &model.inverse))
                    .sum::<f64>()
                    / (broad * d as f64);
                model.inflation = inflation.clamp(1.01, 100.0);
                model.clean_weight = (total_clean / n as f64).clamp(0.5, 1.0 - 1e-6);
                let weights: Vec<_> = posteriors
                    .iter()
                    .map(|&p| p + (1.0 - p) / model.inflation)
                    .collect();
                let weight_sum: f64 = weights.iter().sum();
                model.mean.fill(0.0);
                for (row, &w) in rows.chunks_exact(d).zip(&weights) {
                    for j in 0..d {
                        model.mean[j] += w * row[j] / weight_sum;
                    }
                }
                let mut covariance = Sym::zeros(d);
                for (row, &w) in rows.chunks_exact(d).zip(&weights) {
                    for i in 0..d {
                        for j in 0..=i {
                            let value = covariance.get(i, j)
                                + w * (row[i] - model.mean[i]) * (row[j] - model.mean[j])
                                    / n as f64;
                            covariance.set(i, j, value);
                            covariance.set(j, i, value);
                        }
                    }
                }
                (model.inverse, model.log_det) = prepare(covariance, ridge)?;
            }
            // Both models share d means + d(d+1)/2 covariance parameters. The
            // contaminated model adds mixing weight and covariance inflation.
            model.bic_gain = 2.0 * (model.log_lik - single_ll) - 2.0 * (n as f64).ln();
            Some(model)
        })
        .max_by(|a, b| a.log_lik.total_cmp(&b.log_lik))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn normal(seed: &mut u32) -> f32 {
        let mut uniform = || {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 17;
            *seed ^= *seed << 5;
            (*seed as f64 / u32::MAX as f64).clamp(1e-9, 1.0)
        };
        ((-2.0 * uniform().ln()).sqrt() * (std::f64::consts::TAU * uniform()).cos()) as f32
    }
    fn cloud(tails: bool) -> Vec<f32> {
        let mut seed = 572;
        (0..5000)
            .flat_map(|i| {
                let scale = if tails && i >= 4000 { 6.0 } else { 1.0 };
                let x = scale * normal(&mut seed);
                let y = scale * normal(&mut seed) * 0.15;
                [0.8 * x - 0.6 * y, 0.6 * x + 0.8 * y]
            })
            .collect()
    }
    #[test]
    fn recovers_correlated_cloud_and_broad_tail_without_axis_cut() {
        let data = cloud(true);
        let model = fit_contaminated_gaussian(&data, 2).unwrap();
        assert!(model.bic_gain > 10.0);
        assert!((model.clean_weight - 0.8).abs() < 0.06);
        let posterior = model.clean_posteriors(&data).unwrap();
        assert!(posterior[..4000].iter().filter(|&&p| p >= 0.5).count() > 3700);
        assert!(posterior[4000..].iter().filter(|&&p| p >= 0.5).count() < 200);
        assert!(model.inflation > 20.0);
    }
    #[test]
    fn gaussian_is_not_evidence_for_cleanup_and_fit_is_deterministic() {
        let data = cloud(false);
        let a = fit_contaminated_gaussian(&data, 2).unwrap();
        let b = fit_contaminated_gaussian(&data, 2).unwrap();
        assert!(a.bic_gain < 10.0, "{}", a.bic_gain);
        assert_eq!(a.log_lik, b.log_lik);
        assert_eq!(a.clean_posteriors(&data), b.clean_posteriors(&data));
    }
    #[test]
    fn validates_shape_and_degenerate_input() {
        for (data, d) in [
            (vec![], 0),
            (vec![0.0; 200], 2),
            (vec![f32::NAN; 200], 2),
            (vec![1.0; 201], 2),
            (vec![1.0; 1300], 13),
        ] {
            assert!(fit_contaminated_gaussian(&data, d).is_none());
        }
    }
    #[test]
    fn rotated_fit_has_the_same_memberships() {
        let data = cloud(true);
        let rotated: Vec<_> = data.chunks_exact(2).flat_map(|r| [-r[1], r[0]]).collect();
        let labels = |x: &[f32]| {
            fit_contaminated_gaussian(x, 2)
                .unwrap()
                .clean_posteriors(x)
                .unwrap()
                .into_iter()
                .map(|p| p >= 0.5)
                .collect::<Vec<_>>()
        };
        assert_eq!(labels(&data), labels(&rotated));
    }
}
