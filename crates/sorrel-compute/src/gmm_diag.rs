//! Diagonal-covariance Gaussian mixture on row-major `f32` features: k-means
//! initialisation (fixed seed, so deterministic), EM in the log domain, BIC.
//! Used by the core-split proposer in `sorrel-data`; the 1-D amplitude test
//! lives in [`crate::gmm`].

use crate::kmeans;

const TWO_PI_LN: f64 = 1.837_877_066_409_345_3; // ln(2π)

/// A fitted mixture (`means`/`vars` are row-major `(k, d)`).
#[derive(Clone, Debug)]
pub struct DiagGmm {
    pub k: usize,
    pub d: usize,
    pub weights: Vec<f64>,
    pub means: Vec<f64>,
    pub vars: Vec<f64>,
    /// Total log-likelihood of the fitting rows.
    pub log_lik: f64,
    pub n: usize,
}

impl DiagGmm {
    /// Bayesian information criterion (lower is better).
    pub fn bic(&self) -> f64 {
        let params = (self.k * 2 * self.d + self.k - 1) as f64;
        -2.0 * self.log_lik + params * (self.n as f64).ln()
    }

    /// `ln(w_j) + ln N(row | j)` for every component.
    fn log_joint(&self, row: &[f32], out: &mut [f64]) {
        for (j, o) in out.iter_mut().enumerate() {
            let (mu, var) = (&self.means[j * self.d..], &self.vars[j * self.d..]);
            let q: f64 = (0..self.d)
                .map(|i| {
                    let z = row[i] as f64 - mu[i];
                    z * z / var[i] + var[i].ln() + TWO_PI_LN
                })
                .sum();
            *o = self.weights[j].ln() - 0.5 * q;
        }
    }

    /// Highest-posterior component of every row.
    pub fn assign(&self, data: &[f32]) -> Vec<u32> {
        let mut lj = vec![0.0; self.k];
        data.chunks_exact(self.d)
            .map(|r| {
                self.log_joint(r, &mut lj);
                (0..self.k)
                    .max_by(|&a, &b| lj[a].total_cmp(&lj[b]))
                    .unwrap_or(0) as u32
            })
            .collect()
    }
}

/// Fit `k` components to `data` (`d` columns). `None` for too few rows or
/// when a component collapses to under one row of weight.
pub fn fit_diag_gmm(data: &[f32], d: usize, k: usize, max_iter: usize) -> Option<DiagGmm> {
    let n = data.len().checked_div(d)?;
    if k == 0 || n < 2 * k {
        return None;
    }
    // Per-dimension variance floor: 1e-3 of the data's own variance.
    let gm: Vec<f64> = (0..d)
        .map(|i| data.chunks_exact(d).map(|r| r[i] as f64).sum::<f64>() / n as f64)
        .collect();
    let floor: Vec<f64> = (0..d)
        .map(|i| {
            let v = data
                .chunks_exact(d)
                .map(|r| (r[i] as f64 - gm[i]).powi(2))
                .sum::<f64>();
            (1e-3 * v / n as f64).max(1e-12)
        })
        .collect();

    let km = kmeans(data, d, k, 20);
    let mut g = DiagGmm {
        k,
        d,
        weights: vec![0.0; k],
        means: vec![0.0; k * d],
        vars: vec![0.0; k * d],
        log_lik: f64::NEG_INFINITY,
        n,
    };
    let mut resp = vec![0.0; k]; // initial hard responsibilities, then scratch
    let (mut nk, mut sx, mut sxx) = (vec![0.0; k], vec![0.0; k * d], vec![0.0; k * d]);
    for (r, &l) in data.chunks_exact(d).zip(&km.labels) {
        let l = l as usize;
        nk[l] += 1.0;
        for i in 0..d {
            sx[l * d + i] += r[i] as f64;
            sxx[l * d + i] += (r[i] as f64).powi(2);
        }
    }
    let mut prev = f64::NEG_INFINITY;
    for it in 0..=max_iter {
        // M-step from the accumulated statistics (hard ones on the first pass).
        for j in 0..k {
            if nk[j] < 1.0 {
                return None;
            }
            g.weights[j] = nk[j] / n as f64;
            for i in 0..d {
                let m = sx[j * d + i] / nk[j];
                g.means[j * d + i] = m;
                g.vars[j * d + i] = (sxx[j * d + i] / nk[j] - m * m).max(floor[i]);
            }
        }
        // E-step: log-likelihood and soft statistics for the next M-step.
        nk.fill(0.0);
        sx.fill(0.0);
        sxx.fill(0.0);
        let mut ll = 0.0;
        for r in data.chunks_exact(d) {
            g.log_joint(r, &mut resp);
            let m = resp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let s: f64 = resp.iter().map(|&v| (v - m).exp()).sum();
            ll += m + s.ln();
            for j in 0..k {
                let p = (resp[j] - m).exp() / s;
                nk[j] += p;
                for i in 0..d {
                    sx[j * d + i] += p * r[i] as f64;
                    sxx[j * d + i] += p * (r[i] as f64).powi(2);
                }
            }
        }
        g.log_lik = ll;
        if it > 0 && (ll - prev).abs() < 1e-6 * ll.abs().max(1.0) {
            break;
        }
        prev = ll;
    }
    Some(g)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(s: &mut u32) -> f32 {
        let mut u = || {
            *s ^= *s << 13;
            *s ^= *s >> 17;
            *s ^= *s << 5;
            (*s as f64 / u32::MAX as f64).clamp(1e-9, 1.0)
        };
        let (a, b) = (u(), u());
        ((-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()) as f32
    }

    /// `n` 2-D rows around `(cx, 0)` with spread `sd`.
    fn blob(n: usize, cx: f32, sd: f32, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n * 2)
            .map(|i| sd * noise(&mut s) + if i % 2 == 0 { cx } else { 0.0 })
            .collect()
    }

    #[test]
    fn bic_prefers_the_true_component_count_and_assigns_it() {
        let mut x = blob(900, 0.0, 1.0, 3);
        x.extend(blob(100, 8.0, 0.2, 7));
        let one = fit_diag_gmm(&x, 2, 1, 50).unwrap();
        let two = fit_diag_gmm(&x, 2, 2, 50).unwrap();
        assert!(
            two.bic() < one.bic() - 50.0,
            "{} vs {}",
            two.bic(),
            one.bic()
        );
        let l = two.assign(&x);
        let tight = l[900];
        assert!(l[900..].iter().all(|&v| v == tight));
        assert!(l[..900].iter().filter(|&&v| v == tight).count() < 5);
    }

    #[test]
    fn single_blob_is_not_split_by_bic() {
        let x = blob(1000, 0.0, 1.0, 5);
        let best = (1..=4)
            .filter_map(|k| fit_diag_gmm(&x, 2, k, 50))
            .min_by(|a, b| a.bic().total_cmp(&b.bic()))
            .unwrap();
        assert_eq!(best.k, 1);
    }
}
