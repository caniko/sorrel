//! Bounded recursive feature partitioning. Rare units can live inside a broad
//! tail that a majority-cleanup model correctly refuses to call a good unit.
//! Nodes are feature hypotheses; neither compactness nor rpc assigns a label.

use crate::proposers::cluster_feature_space;
use crate::robust_cleanup::robust_boundary_proposal;
use crate::session::Session;
use sorrel_compute::sliding_refractory_argmin;
use sorrel_io::{ClusterId, DataProvider};
use std::collections::VecDeque;

const MAX_DEPTH: usize = 4;
const MAX_FITS: usize = 15;

#[derive(Clone, Debug)]
pub struct DensityNode {
    /// Sorted root-local spike positions of this node, never original NPY rows.
    pub members: Vec<u32>,
    pub depth: usize,
    pub bic_gain: f64,
    pub contamination: f32,
    pub supported: bool,
}

/// Each parent uses the same predeclared contaminated-Gaussian posterior cut
/// as robust cleanup. Recurse into BOTH parts, not just the presumed clean
/// majority. No truth, timing, labels, or desired unit count enters fitting.
pub fn density_tree_memberships(feats: &[f32], d: usize) -> Vec<DensityNode> {
    if d == 0 || !feats.len().is_multiple_of(d) || feats.len() / d > u32::MAX as usize {
        return vec![];
    }
    let n = feats.len() / d;
    let mut queue = VecDeque::from([(0, (0..n as u32).collect::<Vec<_>>())]);
    let mut nodes = Vec::new();
    let mut fits = 0;
    while let Some((depth, members)) = queue.pop_front() {
        if depth >= MAX_DEPTH || members.len() < 200 || fits >= MAX_FITS {
            continue;
        }
        fits += 1;
        let sub: Vec<_> = members
            .iter()
            .flat_map(|&i| feats[i as usize * d..(i as usize + 1) * d].iter().copied())
            .collect();
        let Some(boundary) = robust_boundary_proposal(&sub, d) else {
            continue;
        };
        let (mut kept, mut tail) = (Vec::new(), Vec::new());
        let mut dropped = boundary.minor_idx.iter().copied().peekable();
        for (i, &root_local) in members.iter().enumerate() {
            if dropped.peek() == Some(&(i as u32)) {
                tail.push(root_local);
                dropped.next();
            } else {
                kept.push(root_local);
            }
        }
        for child in [kept, tail] {
            if depth + 1 < MAX_DEPTH {
                queue.push_back((depth + 1, child.clone()));
            }
            nodes.push(DensityNode {
                members: child,
                depth: depth + 1,
                bic_gain: boundary.bic_gain,
                contamination: f32::NAN,
                supported: false,
            });
        }
    }
    nodes
}

/// Refractory checks only veto feature-generated nodes, never select a
/// threshold. Existing cards/ACGs/isolation still decide whether to label it.
pub fn density_candidates<P: DataProvider>(
    session: &Session<P>,
    c: ClusterId,
    globals: &[u32],
) -> Vec<DensityNode> {
    if globals.len() < 200 {
        return vec![];
    }
    let Some((feats, d)) = cluster_feature_space(session, c, globals) else {
        return vec![];
    };
    let mut nodes = density_tree_memberships(&feats, d);
    let times = session.spike_times(c);
    let sr = session.provider.sample_rate();
    if !sr.is_finite() || sr <= 0.0 {
        return vec![];
    }
    let duration = session.provider.n_samples().0;
    let measure = |t: &[_]| {
        let (cont, rp) = sliding_refractory_argmin(
            t,
            (sr * 0.0005) as u64,
            (sr * 0.010) as u64,
            20,
            duration,
            sr,
        );
        if rp > 0 {
            cont
        } else {
            f32::NAN
        }
    };
    let whole = measure(times);
    for node in &mut nodes {
        let selected: Vec<_> = node.members.iter().map(|&i| times[i as usize]).collect();
        node.contamination = measure(&selected);
        node.supported = node.contamination.is_finite()
            && whole.is_finite()
            && node.contamination <= 0.1
            && node.contamination <= whole;
    }
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;
    fn normal(seed: &mut u32) -> f32 {
        let mut u = || {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 17;
            *seed ^= *seed << 5;
            (*seed as f64 / u32::MAX as f64).clamp(1e-9, 1.0)
        };
        ((-2.0 * u().ln()).sqrt() * (std::f64::consts::TAU * u()).cos()) as f32
    }
    #[test]
    fn rare_cloud_can_be_reached_inside_the_first_discarded_tail() {
        let mut seed = 76512;
        let data: Vec<_> = (0..40_000)
            .flat_map(|i| {
                let rare_group = i % 200 < 4;
                let clean = i % 200 < 3;
                let x = normal(&mut seed);
                let y = normal(&mut seed);
                if rare_group {
                    let scale = if clean { 0.12 } else { 2.0 };
                    [8.0 + scale * x, 8.0 + scale * y]
                } else {
                    [x, y]
                }
            })
            .collect();
        let nodes = density_tree_memberships(&data, 2);
        assert!(nodes.len() <= 2 * MAX_FITS);
        assert!(nodes.iter().all(|p| p.depth <= MAX_DEPTH
            && p.members.len() >= 100
            && p.members.windows(2).all(|w| w[0] < w[1])));
        let overlap = |node: &DensityNode| {
            let hit = node.members.iter().filter(|&&i| i % 200 < 3).count();
            hit as f64 / (600 + node.members.len() - hit) as f64
        };
        assert!(
            nodes.iter().any(|n| n.depth > 1 && overlap(n) > 0.8),
            "best {}",
            nodes.iter().map(overlap).fold(0.0, f64::max)
        );
        let again = density_tree_memberships(&data, 2);
        assert_eq!(
            nodes.iter().map(|p| &p.members).collect::<Vec<_>>(),
            again.iter().map(|p| &p.members).collect::<Vec<_>>()
        );
    }
    #[test]
    fn no_tree_on_degenerate_or_unimodal_data() {
        assert!(density_tree_memberships(&[0.0; 400], 2).is_empty());
        assert!(density_tree_memberships(&[f32::NAN; 400], 2).is_empty());
        assert!(density_tree_memberships(&[1.0; 401], 2).is_empty());
        let mut seed = 874652;
        let data: Vec<_> = (0..10_000).map(|_| normal(&mut seed)).collect();
        assert!(density_tree_memberships(&data, 2).is_empty());
    }
}
