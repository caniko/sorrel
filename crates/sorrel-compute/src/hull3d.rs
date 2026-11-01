//! Incremental 3-D convex hull, for the Features 3-D view's cluster hulls.
//!
//! Geometry runs in `f64` over the `f32` input. Degenerate input (fewer than
//! four points, all collinear or coplanar) has no volume and gives no faces.

use std::collections::HashSet;

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

struct Face {
    v: [u32; 3],
    /// Unit outward normal (zero for a sliver) and plane offset: `n·p = d`.
    n: V3,
    d: f64,
}

impl Face {
    fn new(v: [u32; 3], p: &[V3]) -> Self {
        let [a, b, c] = v.map(|i| p[i as usize]);
        let n = cross(sub(b, a), sub(c, a));
        let len = dot(n, n).sqrt();
        let n = if len > 0.0 {
            [n[0] / len, n[1] / len, n[2] / len]
        } else {
            [0.0; 3]
        };
        Self { v, n, d: dot(n, a) }
    }

    fn above(&self, q: V3, eps: f64) -> bool {
        dot(self.n, q) - self.d > eps
    }
}

/// Triangles `[a, b, c]` (indices into `points`) of the convex hull, wound
/// counter-clockwise seen from outside. Empty for degenerate input.
///
/// ponytail: O(n·h) incremental hull; n ≤ ~2500 after trimming, switch to
/// quickhull if inputs grow.
pub fn convex_hull_3d(points: &[[f32; 3]]) -> Vec<[u32; 3]> {
    if points.len() < 4 {
        return Vec::new();
    }
    let p: Vec<V3> = points.iter().map(|q| q.map(f64::from)).collect();
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for q in &p {
        for k in 0..3 {
            lo[k] = lo[k].min(q[k]);
            hi[k] = hi[k].max(q[k]);
        }
    }
    let extent = (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f64::max);
    if !extent.is_finite() || extent <= 0.0 {
        return Vec::new();
    }
    let eps = 1e-9 * extent;

    // Initial tetrahedron: extreme point, farthest point, farthest from that
    // line, farthest from that plane.
    let argmax = |f: &dyn Fn(V3) -> f64| {
        (0..p.len())
            .max_by(|&i, &j| f(p[i]).total_cmp(&f(p[j])))
            .unwrap()
    };
    let i0 = argmax(&|q| -q[0]);
    let i1 = argmax(&|q| dot(sub(q, p[i0]), sub(q, p[i0])));
    let e = sub(p[i1], p[i0]);
    let i2 = argmax(&|q| {
        let c = cross(e, sub(q, p[i0]));
        dot(c, c)
    });
    let nrm = cross(e, sub(p[i2], p[i0]));
    let nlen = dot(nrm, nrm).sqrt();
    if nlen <= eps * extent {
        return Vec::new();
    }
    let i3 = argmax(&|q| dot(nrm, sub(q, p[i0])).abs());
    if dot(nrm, sub(p[i3], p[i0])).abs() / nlen <= eps {
        return Vec::new();
    }

    let inside = {
        let s: Vec<V3> = [i0, i1, i2, i3].iter().map(|&i| p[i]).collect();
        [0, 1, 2].map(|k| (s[0][k] + s[1][k] + s[2][k] + s[3][k]) / 4.0)
    };
    let (a, b, c, d) = (i0 as u32, i1 as u32, i2 as u32, i3 as u32);
    let mut faces: Vec<Face> = [[a, b, c], [a, b, d], [a, c, d], [b, c, d]]
        .into_iter()
        .map(|mut v| {
            if Face::new(v, &p).above(inside, 0.0) {
                v.swap(1, 2);
            }
            Face::new(v, &p)
        })
        .collect();

    for (i, &q) in p.iter().enumerate() {
        let vis: Vec<bool> = faces.iter().map(|f| f.above(q, eps)).collect();
        if !vis.contains(&true) {
            continue;
        }
        // Horizon: directed edges of visible faces whose twin is not visible.
        let edges: HashSet<(u32, u32)> = faces
            .iter()
            .zip(&vis)
            .filter(|(_, &v)| v)
            .flat_map(|(f, _)| (0..3).map(|k| (f.v[k], f.v[(k + 1) % 3])))
            .collect();
        let horizon: Vec<(u32, u32)> = edges
            .iter()
            .copied()
            .filter(|&(x, y)| !edges.contains(&(y, x)))
            .collect();
        let mut keep = vis.iter().map(|v| !v);
        faces.retain(|_| keep.next().unwrap());
        faces.extend(
            horizon
                .into_iter()
                .map(|(x, y)| Face::new([x, y, i as u32], &p)),
        );
    }
    faces.into_iter().map(|f| f.v).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 40) as f32) / (1u64 << 24) as f32 * 2.0 - 1.0
    }

    /// Every point on or inside every face plane, and the surface is closed
    /// (each directed edge has exactly its reverse as a twin).
    fn assert_hull(points: &[[f32; 3]], tris: &[[u32; 3]]) {
        for t in tris {
            let p: Vec<V3> = points.iter().map(|q| q.map(f64::from)).collect();
            let f = Face::new(*t, &p);
            for q in &p {
                assert!(dot(f.n, *q) - f.d <= 1e-6, "point outside face {t:?}");
            }
        }
        let edges: HashSet<(u32, u32)> = tris
            .iter()
            .flat_map(|t| (0..3).map(|k| (t[k], t[(k + 1) % 3])))
            .collect();
        assert_eq!(edges.len(), tris.len() * 3, "edge used twice");
        assert!(edges.iter().all(|&(a, b)| edges.contains(&(b, a))));
    }

    #[test]
    fn cube_with_interior_points() {
        let mut pts: Vec<[f32; 3]> = (0..8)
            .map(|i| [(i & 1) as f32, ((i >> 1) & 1) as f32, ((i >> 2) & 1) as f32])
            .collect();
        pts.extend([[0.5, 0.5, 0.5], [0.2, 0.7, 0.4], [0.5, 0.5, 0.0]]);
        let tris = convex_hull_3d(&pts);
        assert_eq!(tris.len(), 12);
        assert!(tris.iter().flatten().all(|&i| i < 8 || i == 10));
        assert_hull(&pts, &tris);
    }

    #[test]
    fn random_points_are_enclosed() {
        let mut s = 7;
        let pts: Vec<[f32; 3]> = (0..400)
            .map(|_| [lcg(&mut s), lcg(&mut s), lcg(&mut s)])
            .collect();
        let tris = convex_hull_3d(&pts);
        assert!(tris.len() >= 4);
        assert_hull(&pts, &tris);
    }

    #[test]
    fn degenerate_input_is_empty() {
        assert!(convex_hull_3d(&[]).is_empty());
        assert!(convex_hull_3d(&[[0.0; 3]; 3]).is_empty());
        let coplanar: Vec<[f32; 3]> = (0..20)
            .map(|i| [i as f32, (i * i % 7) as f32, 2.0])
            .collect();
        assert!(convex_hull_3d(&coplanar).is_empty());
        let line: Vec<[f32; 3]> = (0..9).map(|i| [i as f32, 2.0 * i as f32, 0.0]).collect();
        assert!(convex_hull_3d(&line).is_empty());
        assert!(convex_hull_3d(&[[1.0; 3]; 10]).is_empty());
    }
}
