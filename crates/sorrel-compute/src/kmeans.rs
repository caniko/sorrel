//! k-means (k-means++ seeding, Lloyd's iterations) on row-major `f32` data.
//!
//! The seed is fixed, so the same input always gives the same partition —
//! suggestions must not flicker between recomputes. Used by the multi-
//! dimensional split proposer in `sorrel-data`.

/// Fixed seed for the k-means++ draws.
const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Result of [`kmeans`].
#[derive(Clone, Debug)]
pub struct KMeans {
    /// Centroid index per input row.
    pub labels: Vec<u32>,
    /// Row-major `(k, d)` centroids.
    pub centroids: Vec<f32>,
    /// Sum of squared distances from each row to its centroid.
    pub inertia: f64,
}

/// splitmix64 step: tiny, good enough for seeding draws.
fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// `(index, squared distance)` of the centroid nearest to `row`.
fn nearest(row: &[f32], centroids: &[f32], d: usize) -> (usize, f32) {
    centroids
        .chunks_exact(d)
        .enumerate()
        .map(|(j, c)| (j, dist2(row, c)))
        .fold(
            (0, f32::INFINITY),
            |best, cur| if cur.1 < best.1 { cur } else { best },
        )
}

/// Label every row of `data` with its nearest centroid (`centroids` is
/// row-major `(k, d)`).
pub fn assign(data: &[f32], d: usize, centroids: &[f32]) -> Vec<u32> {
    if d == 0 || centroids.is_empty() {
        return Vec::new();
    }
    data.chunks_exact(d)
        .map(|row| nearest(row, centroids, d).0 as u32)
        .collect()
}

/// Cluster `data` (row-major, `d` columns) into `k` groups. `k` is clamped
/// to the number of rows; empty input gives an empty result.
pub fn kmeans(data: &[f32], d: usize, k: usize, max_iter: usize) -> KMeans {
    let n = data.len().checked_div(d).unwrap_or(0);
    let k = k.min(n);
    if k == 0 {
        return KMeans {
            labels: Vec::new(),
            centroids: Vec::new(),
            inertia: 0.0,
        };
    }
    let row = |i: usize| &data[i * d..(i + 1) * d];
    let mut rng = SEED;

    // k-means++: each next centre is drawn with probability ∝ squared
    // distance to the nearest centre chosen so far.
    let mut centroids: Vec<f32> = row(next_u64(&mut rng) as usize % n).to_vec();
    let mut near: Vec<f64> = (0..n).map(|i| dist2(row(i), &centroids) as f64).collect();
    while centroids.len() < k * d {
        let total: f64 = near.iter().sum();
        let pick = if total > 0.0 {
            let mut r = (next_u64(&mut rng) >> 11) as f64 / (1u64 << 53) as f64 * total;
            near.iter()
                .position(|&w| {
                    r -= w;
                    r < 0.0
                })
                .unwrap_or(n - 1)
        } else {
            next_u64(&mut rng) as usize % n // all rows identical
        };
        let c = row(pick).to_vec();
        for (i, w) in near.iter_mut().enumerate() {
            *w = w.min(dist2(row(i), &c) as f64);
        }
        centroids.extend(c);
    }

    // Lloyd's.
    let mut labels = vec![u32::MAX; n];
    for _ in 0..max_iter.max(1) {
        let mut changed = false;
        for (i, l) in labels.iter_mut().enumerate() {
            let j = nearest(row(i), &centroids, d).0 as u32;
            changed |= *l != j;
            *l = j;
        }
        if !changed {
            break;
        }
        let mut sums = vec![0.0_f64; k * d];
        let mut counts = vec![0usize; k];
        for (i, &l) in labels.iter().enumerate() {
            counts[l as usize] += 1;
            for (s, &v) in sums[l as usize * d..(l as usize + 1) * d]
                .iter_mut()
                .zip(row(i))
            {
                *s += v as f64;
            }
        }
        for j in (0..k).filter(|&j| counts[j] > 0) {
            for t in 0..d {
                centroids[j * d + t] = (sums[j * d + t] / counts[j] as f64) as f32;
            }
        }
    }
    let inertia = labels
        .iter()
        .enumerate()
        .map(|(i, &l)| dist2(row(i), &centroids[l as usize * d..(l as usize + 1) * d]) as f64)
        .sum();
    KMeans {
        labels,
        centroids,
        inertia,
    }
}

impl KMeans {
    /// Mean simplified silhouette (Hruschka et al. 2004) over `data`: with
    /// `a` the distance to the row's own centroid and `b` the distance to the
    /// nearest other centroid, `(b − a) / max(a, b)`. Centroid-based, so
    /// O(n·k·d) — unlike the pairwise silhouette it stays cheap at 5k rows.
    /// `metrics_iso::simplified_silhouette` is the cluster-vs-pooled-rest
    /// form; it mis-scores k > 2 because the "rest" centroid sits between
    /// the other parts, so k-means gets its own.
    ///
    /// `NaN` for fewer than two centroids.
    pub fn silhouette(&self, data: &[f32], d: usize) -> f32 {
        let k = self.centroids.len().checked_div(d).unwrap_or(0);
        if k < 2 || self.labels.len() * d != data.len() {
            return f32::NAN;
        }
        let mut acc = 0.0_f64;
        for (row, &l) in data.chunks_exact(d).zip(&self.labels) {
            let (mut a, mut b) = (0.0_f32, f32::INFINITY);
            for (j, c) in self.centroids.chunks_exact(d).enumerate() {
                let dist = dist2(row, c).sqrt();
                if j == l as usize {
                    a = dist;
                } else {
                    b = b.min(dist);
                }
            }
            let m = a.max(b);
            if m > 0.0 {
                acc += ((b - a) / m) as f64;
            }
        }
        (acc / self.labels.len() as f64) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic noise in `[-0.5, 0.5)` (xorshift32).
    fn noise(s: &mut u32) -> f32 {
        *s ^= *s << 13;
        *s ^= *s >> 17;
        *s ^= *s << 5;
        *s as f32 / u32::MAX as f32 - 0.5
    }

    #[test]
    fn recovers_three_blobs() {
        let centres = [[0.0_f32, 0.0], [10.0, 0.0], [5.0, 9.0]];
        let mut s = 0xC0FFEE_u32;
        let mut data = Vec::new();
        for c in &centres {
            for _ in 0..100 {
                data.extend([c[0] + noise(&mut s), c[1] + noise(&mut s)]);
            }
        }
        let km = kmeans(&data, 2, 3, 50);
        for b in 0..3 {
            let first = km.labels[b * 100];
            assert!(km.labels[b * 100..(b + 1) * 100]
                .iter()
                .all(|&l| l == first));
        }
        let mut ids: Vec<u32> = (0..3).map(|b| km.labels[b * 100]).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 3, "blobs must get distinct labels");
        assert!(km.inertia < 300.0 * 0.25, "inertia {}", km.inertia);
        assert!(km.silhouette(&data, 2) > 0.8);
        assert_eq!(assign(&data, 2, &km.centroids), km.labels);
        // Fixed seed: identical rerun.
        assert_eq!(kmeans(&data, 2, 3, 50).labels, km.labels);
    }
}
