//! The typed Units payload shared by the HDF5 and PyNWB serializers.

pub struct Units {
    pub description: String,
    pub resolution: f64,
    pub id: Vec<i64>,
    pub columns: Vec<Column>,
    pub generated_by: [&'static str; 2],
}

pub struct Column {
    pub name: String,
    pub description: String,
    pub data: Values,
    /// Exclusive end offsets for a ragged column, including empty rows.
    pub index: Option<Vec<u64>>,
}

pub enum Values {
    Float(Vec<f64>),
    Int(Vec<i64>),
    Text(Vec<String>),
}

fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// Sign of the exact sum of `terms` (Shewchuk's grow-expansion: the
/// components stay non-overlapping in increasing magnitude, so the last
/// nonzero one carries the sign).
fn exact_sign(terms: &[f64]) -> std::cmp::Ordering {
    let mut expansion: Vec<f64> = Vec::with_capacity(terms.len());
    for &term in terms {
        let mut q = term;
        let mut next = Vec::with_capacity(expansion.len() + 1);
        for &e in &expansion {
            let (s, err) = two_sum(q, e);
            if err != 0.0 {
                next.push(err);
            }
            q = s;
        }
        next.push(q);
        expansion = next;
    }
    let top = expansion.iter().rev().find(|&&x| x != 0.0).copied();
    top.map_or(std::cmp::Ordering::Equal, |x| x.total_cmp(&0.0))
}

/// `floor((t - t0) × rate + 0.5)` of the exact real product, not of its
/// f64 rounding: near a half-sample boundary the rounded product can land
/// on the wrong side. The difference is split exactly (TwoSum) and each
/// part multiplied exactly (FMA), so the position is a sum of four f64s
/// whose position relative to each half-sample boundary is decided by an
/// exact expansion sign. NaN (refused by the caller) where that is not
/// provably exact: a nonzero product below 2^-960 (its FMA error could
/// underflow), a position of -0.5 samples or less, or 2^51 samples or
/// more (where n ± 0.5 is no longer an f64).
pub fn exact_sample(t: f64, t0: f64, rate: f64) -> f64 {
    const LIMIT: f64 = (1u64 << 51) as f64;
    let tiny = f64::MIN_POSITIVE * 2f64.powi(62); // 2^-960
    let (d_hi, d_lo) = two_sum(t, -t0);
    let parts = [(d_hi, d_hi * rate), (d_lo, d_lo * rate)];
    // |d_lo × rate| <= 2^-52 |d_hi × rate|: bounding the leading product
    // keeps every term, and the expansion sums, far from overflow.
    let leading_in_range = parts[0].1.abs() < 2.0 * LIMIT; // false for NaN
    if !leading_in_range || parts.iter().any(|&(d, p)| d != 0.0 && p.abs() < tiny) {
        return f64::NAN;
    }
    let terms = [
        parts[0].1,
        d_hi.mul_add(rate, -parts[0].1),
        parts[1].1,
        d_lo.mul_add(rate, -parts[1].1),
    ];
    let cmp = |c: f64| exact_sign(&[terms[0], terms[1], terms[2], terms[3], -c]);
    if cmp(-0.5).is_le() || cmp(LIMIT).is_ge() {
        return f64::NAN;
    }
    // In range, n ± 0.5 are exact; the f64 estimate is within a few
    // samples, so step until n - 0.5 <= position < n + 0.5.
    let mut n = terms.iter().sum::<f64>().round().clamp(0.0, LIMIT);
    while cmp(n - 0.5).is_lt() {
        n -= 1.0;
    }
    while cmp(n + 0.5).is_ge() {
        n += 1.0;
    }
    n
}
