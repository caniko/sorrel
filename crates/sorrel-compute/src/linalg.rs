//! Tiny linear-algebra helpers — only what the isolation metrics need.
//!
//! We deliberately avoid pulling in `nalgebra` or `ndarray` here: the
//! feature subspaces we operate on are small (typically d=3–6 PCs on the
//! peak channel), so closed-form / Gauss-Jordan is faster than a generic
//! matrix library and lets the rest of the workspace stay dependency-free.

/// Symmetric `d × d` matrix stored row-major as a flat `Vec<f64>`. We keep
/// it as `f64` internally (covariance accumulators benefit from the extra
/// precision) and convert at the boundary.
#[derive(Clone, Debug)]
pub struct Sym {
    pub d: usize,
    pub data: Vec<f64>,
}

impl Sym {
    pub fn zeros(d: usize) -> Self {
        Self { d, data: vec![0.0; d * d] }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.d + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.d + c] = v;
    }

    /// Add `lambda` to the diagonal — Tikhonov regularisation, used to make
    /// rank-deficient covariance invertible.
    pub fn ridge(&mut self, lambda: f64) {
        for i in 0..self.d {
            let v = self.get(i, i);
            self.set(i, i, v + lambda);
        }
    }
}

/// Sample mean of `n` rows of length `d`, supplied row-major in `data`.
pub fn row_mean(data: &[f64], n: usize, d: usize) -> Vec<f64> {
    if n == 0 || d == 0 {
        return vec![0.0; d];
    }
    let mut out = vec![0.0_f64; d];
    for i in 0..n {
        for j in 0..d {
            out[j] += data[i * d + j];
        }
    }
    let inv = 1.0 / n as f64;
    for v in out.iter_mut() {
        *v *= inv;
    }
    out
}

/// Sample covariance (Bessel-corrected) of `n` rows of length `d`. `data`
/// is row-major. Returns a symmetric `d × d` matrix.
pub fn covariance(data: &[f64], n: usize, d: usize) -> Sym {
    let mut s = Sym::zeros(d);
    if n < 2 || d == 0 {
        return s;
    }
    let mean = row_mean(data, n, d);
    let inv = 1.0 / (n - 1) as f64;
    for i in 0..n {
        for r in 0..d {
            let dr = data[i * d + r] - mean[r];
            for c in r..d {
                let dc = data[i * d + c] - mean[c];
                let v = s.get(r, c) + dr * dc;
                s.set(r, c, v);
            }
        }
    }
    // Symmetrise + scale.
    for r in 0..d {
        for c in r..d {
            let v = s.get(r, c) * inv;
            s.set(r, c, v);
            s.set(c, r, v);
        }
    }
    s
}

/// In-place Gauss-Jordan inverse of a symmetric matrix. Returns `None` if
/// the matrix is singular (within `tol`). Operates on `m.data` directly;
/// only the upper triangle is meaningful afterward — we resymmetrise.
pub fn invert(m: &mut Sym, tol: f64) -> Option<()> {
    let d = m.d;
    if d == 0 {
        return Some(());
    }
    // Augment with identity in a separate buffer; we keep the augmented
    // representation as a single 2d×d matrix laid out as [m | I].
    let mut a = vec![0.0_f64; d * 2 * d];
    for r in 0..d {
        for c in 0..d {
            a[r * 2 * d + c] = m.get(r, c);
        }
        a[r * 2 * d + d + r] = 1.0;
    }
    // Forward elimination with partial pivoting.
    for c in 0..d {
        // Find pivot.
        let mut pivot = c;
        let mut best = a[c * 2 * d + c].abs();
        for r in (c + 1)..d {
            let v = a[r * 2 * d + c].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best < tol {
            return None;
        }
        if pivot != c {
            // Swap rows c and pivot.
            for k in 0..2 * d {
                let tmp = a[c * 2 * d + k];
                a[c * 2 * d + k] = a[pivot * 2 * d + k];
                a[pivot * 2 * d + k] = tmp;
            }
        }
        // Normalise pivot row.
        let p = a[c * 2 * d + c];
        let inv_p = 1.0 / p;
        for k in 0..2 * d {
            a[c * 2 * d + k] *= inv_p;
        }
        // Eliminate other rows.
        for r in 0..d {
            if r == c {
                continue;
            }
            let f = a[r * 2 * d + c];
            if f == 0.0 {
                continue;
            }
            for k in 0..2 * d {
                a[r * 2 * d + k] -= f * a[c * 2 * d + k];
            }
        }
    }
    // Copy the inverse (right block) back into m.
    for r in 0..d {
        for c in 0..d {
            m.set(r, c, a[r * 2 * d + d + c]);
        }
    }
    // Resymmetrise to absorb floating-point asymmetry.
    for r in 0..d {
        for c in (r + 1)..d {
            let v = 0.5 * (m.get(r, c) + m.get(c, r));
            m.set(r, c, v);
            m.set(c, r, v);
        }
    }
    Some(())
}

/// Mahalanobis squared distance: `(x - mu)^T Σ⁻¹ (x - mu)`.
/// `inv` is the already-inverted covariance.
pub fn mahalanobis_sq(x: &[f64], mu: &[f64], inv: &Sym) -> f64 {
    let d = inv.d;
    let mut diff = vec![0.0_f64; d];
    for i in 0..d {
        diff[i] = x[i] - mu[i];
    }
    // diff^T inv diff
    let mut acc = 0.0_f64;
    for r in 0..d {
        let mut row = 0.0_f64;
        for c in 0..d {
            row += inv.get(r, c) * diff[c];
        }
        acc += diff[r] * row;
    }
    acc.max(0.0)
}

/// Chi-squared upper-tail survival function `P(X² > x)` for `df` degrees of
/// freedom. Used in the L-ratio metric. Implemented via the regularised
/// upper incomplete gamma `Q(df/2, x/2)`. Series and continued-fraction
/// branches taken straight from Numerical Recipes §6.2.
pub fn chi2_sf(x: f64, df: f64) -> f64 {
    if x <= 0.0 || df <= 0.0 {
        return 1.0;
    }
    gamma_q(df * 0.5, x * 0.5)
}

fn gamma_q(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return 1.0;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - gamma_p_series(a, x)
    } else {
        gamma_q_cf(a, x)
    }
}

fn gamma_p_series(a: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 1e-12;
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut term = sum;
    for _ in 0..max_iter {
        ap += 1.0;
        term *= x / ap;
        sum += term;
        if term.abs() < sum.abs() * eps {
            break;
        }
    }
    let log_g = ln_gamma(a);
    sum * (-x + a * x.ln() - log_g).exp()
}

fn gamma_q_cf(a: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 1e-12;
    let fpmin = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / fpmin;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=max_iter {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = b + an / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < eps {
            break;
        }
    }
    let log_g = ln_gamma(a);
    h * (-x + a * x.ln() - log_g).exp()
}

/// `ln(Γ(z))` — Lanczos approximation. Plenty accurate (≈1e-12) for our
/// degrees-of-freedom range (small integers up to a few hundred).
fn ln_gamma(z: f64) -> f64 {
    let g = 7.0;
    const COEFF: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571_6e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if z < 0.5 {
        // Reflection formula
        return (std::f64::consts::PI / (std::f64::consts::PI * z).sin()).ln() - ln_gamma(1.0 - z);
    }
    let z = z - 1.0;
    let mut a = COEFF[0];
    for (i, &c) in COEFF.iter().enumerate().skip(1) {
        a += c / (z + i as f64);
    }
    let t = z + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + a.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invert_identity_returns_identity() {
        let mut m = Sym::zeros(3);
        for i in 0..3 {
            m.set(i, i, 1.0);
        }
        invert(&mut m, 1e-12).unwrap();
        for r in 0..3 {
            for c in 0..3 {
                let expect = if r == c { 1.0 } else { 0.0 };
                assert!((m.get(r, c) - expect).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn invert_known_2x2() {
        // [[2, 1], [1, 3]]^-1 = (1/5) [[3, -1], [-1, 2]]
        let mut m = Sym::zeros(2);
        m.set(0, 0, 2.0);
        m.set(0, 1, 1.0);
        m.set(1, 0, 1.0);
        m.set(1, 1, 3.0);
        invert(&mut m, 1e-12).unwrap();
        assert!((m.get(0, 0) - 0.6).abs() < 1e-9);
        assert!((m.get(0, 1) + 0.2).abs() < 1e-9);
        assert!((m.get(1, 1) - 0.4).abs() < 1e-9);
    }

    #[test]
    fn invert_singular_matrix_returns_none() {
        let mut m = Sym::zeros(2);
        // Rank-1: rows are linearly dependent.
        m.set(0, 0, 1.0); m.set(0, 1, 2.0);
        m.set(1, 0, 2.0); m.set(1, 1, 4.0);
        assert!(invert(&mut m, 1e-9).is_none());
    }

    #[test]
    fn covariance_of_identity_cloud_is_identity() {
        // Two-d cloud: 4 points at (±1, ±1) — sample covariance should be
        // diagonal with entries 4/3.
        let data = [1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0];
        let cov = covariance(&data, 4, 2);
        assert!((cov.get(0, 0) - 4.0 / 3.0).abs() < 1e-9);
        assert!((cov.get(1, 1) - 4.0 / 3.0).abs() < 1e-9);
        assert!(cov.get(0, 1).abs() < 1e-9);
    }

    #[test]
    fn mahalanobis_zero_at_centroid() {
        let mut inv = Sym::zeros(2);
        inv.set(0, 0, 1.0);
        inv.set(1, 1, 1.0);
        let mu = [0.0_f64, 0.0];
        assert_eq!(mahalanobis_sq(&[0.0, 0.0], &mu, &inv), 0.0);
    }

    #[test]
    fn mahalanobis_equals_euclidean_under_identity_inv() {
        let mut inv = Sym::zeros(2);
        inv.set(0, 0, 1.0);
        inv.set(1, 1, 1.0);
        let mu = [0.0_f64, 0.0];
        let d2 = mahalanobis_sq(&[3.0, 4.0], &mu, &inv);
        assert!((d2 - 25.0).abs() < 1e-9);
    }

    #[test]
    fn chi2_sf_is_one_at_zero_and_decreases() {
        assert!((chi2_sf(0.0, 3.0) - 1.0).abs() < 1e-9);
        let a = chi2_sf(1.0, 3.0);
        let b = chi2_sf(5.0, 3.0);
        let c = chi2_sf(20.0, 3.0);
        assert!(a > b && b > c);
        assert!(c < 0.001);
    }

    #[test]
    fn chi2_sf_matches_known_values() {
        // Known: P(X² > 3.84, df=1) ≈ 0.05.
        let p = chi2_sf(3.841, 1.0);
        assert!((p - 0.05).abs() < 0.005, "got {p}");
        // P(X² > 7.815, df=3) ≈ 0.05.
        let p = chi2_sf(7.815, 3.0);
        assert!((p - 0.05).abs() < 0.005, "got {p}");
    }

    #[test]
    fn ridge_makes_singular_matrix_invertible() {
        let mut m = Sym::zeros(2);
        m.set(0, 0, 1.0); m.set(0, 1, 2.0);
        m.set(1, 0, 2.0); m.set(1, 1, 4.0);
        m.ridge(0.01);
        assert!(invert(&mut m, 1e-12).is_some());
    }
}
