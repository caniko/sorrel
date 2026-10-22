//! Per-spike depth along the probe, for a real drift map (depth vs time).
//!
//! Sources, best first:
//!
//! 1. **PC-weighted centre of mass**: Kilosort's first principal component
//!    per channel is a signed projection of the spike on the template's
//!    channels; `max(pc0, 0)²` weights over those channels' `y` positions
//!    give a sub-channel depth estimate, the same recipe phy's and
//!    SpikeInterface's drift maps use.
//! 2. **Template peak channel** (`y` of the channel with the largest
//!    peak-to-peak in the template): coarse, one value per template.
//!
//! `spike_positions.npy` (Kilosort 4) would be a third source; it is not
//! read yet. Neither NWB nor SortingAnalyzer carry PC features today, so
//! they get the template fallback or `None`.

use crate::session::Session;
use sorrel_io::{ClusterId, DataProvider, HasGeometry, HasPcFeatures};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DepthSource {
    PcWeighted,
    TemplatePeak,
}

#[derive(Clone, Debug)]
pub struct SpikeDepths {
    /// `per_cluster[c][i]` is the depth (µm) of the i-th time-sorted spike
    /// of cluster `c`, parallel to `Session::spike_times(c)`.
    pub per_cluster: Vec<Vec<f32>>,
    pub source: DepthSource,
    pub min: f32,
    pub max: f32,
}

/// Centre of mass of `channel_y` weighted by `max(pc0, 0)²`; `None` when no
/// channel has positive loading (the spike is anti-phase on every channel,
/// which happens for noise) or `chans` is empty.
pub fn pc_weighted_depth(
    pc0: &[f32],
    chans: &[u32],
    channel_y: impl Fn(u32) -> f32,
) -> Option<f32> {
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for (&w, &ch) in pc0.iter().zip(chans) {
        let w = w.max(0.0);
        let w = w * w;
        num += w * channel_y(ch);
        den += w;
    }
    (den > 0.0).then(|| num / den)
}

/// Depth per spike for every cluster, or `None` when neither PC features
/// nor templates with geometry are available.
pub fn spike_depths<P>(session: &Session<P>) -> Option<SpikeDepths>
where
    P: DataProvider + HasPcFeatures + HasGeometry,
{
    let positions = session.provider.channel_positions();
    if positions.is_empty() {
        return None;
    }
    let y = |ch: u32| positions.get(ch as usize).map(|p| p[1]).unwrap_or(0.0);
    let (n_pcs, n_ch_tpl) = session.pc_shape();
    let pc_ind = session.provider.pc_feature_ind();
    let has_pcs = n_pcs > 0 && n_ch_tpl > 0 && !pc_ind.is_empty();

    // Template peak-channel depth, computed once per template on demand.
    let (n_templates, n_samples, n_chan) = session.template_shape();
    let mut peak_depth: Vec<Option<f32>> = vec![None; n_templates];
    let mut template_depth = |t: u32| -> Option<f32> {
        let slot = peak_depth.get_mut(t as usize)?;
        if slot.is_none() {
            let wf = session.template_waveform(t)?;
            let peak = (0..n_chan).max_by(|&a, &b| {
                let ptp = |c: usize| {
                    let (lo, hi) = (0..n_samples)
                        .map(|s| wf[s * n_chan + c])
                        .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));
                    hi - lo
                };
                ptp(a).total_cmp(&ptp(b))
            })?;
            *slot = Some(y(peak as u32));
        }
        *slot
    };
    if !has_pcs && n_templates == 0 {
        return None;
    }

    // Walk the global time order once so each (cluster, local) pair gets
    // its global index for the PC lookup.
    let clusters = session.spike_clusters_global();
    let mut per_cluster: Vec<Vec<f32>> = (0..session.n_clusters())
        .map(|c| Vec::with_capacity(session.spike_times(ClusterId(c)).len()))
        .collect();
    let mut any_pc = false;
    let (mut min, mut max) = (f32::MAX, f32::MIN);
    for (g, &c) in clusters.iter().enumerate() {
        let local = per_cluster[c.idx()].len();
        let template = session.spike_templates(c).get(local).copied();
        let from_pc = if has_pcs {
            template.and_then(|t| {
                let pcs = session.pc_feature_for(g as u32)?;
                let chans = pc_ind.get(t as usize * n_ch_tpl..(t as usize + 1) * n_ch_tpl)?;
                pc_weighted_depth(&pcs[..n_ch_tpl], chans, y)
            })
        } else {
            None
        };
        any_pc |= from_pc.is_some();
        let d = from_pc
            .or_else(|| template.and_then(&mut template_depth))
            .unwrap_or(f32::NAN);
        if d.is_finite() {
            min = min.min(d);
            max = max.max(d);
        }
        per_cluster[c.idx()].push(d);
    }
    if min > max {
        return None;
    }
    Some(SpikeDepths {
        per_cluster,
        source: if any_pc {
            DepthSource::PcWeighted
        } else {
            DepthSource::TemplatePeak
        },
        min,
        max,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pc_weighted_depth_interpolates_between_channels() {
        let y = |ch: u32| [0.0f32, 100.0][ch as usize];
        assert_eq!(pc_weighted_depth(&[1.0, 1.0], &[0, 1], y), Some(50.0));
        assert_eq!(pc_weighted_depth(&[1.0, 0.0], &[0, 1], y), Some(0.0));
        // Squared weights: loading 2 on channel 1 vs 1 on channel 0 → 4:1.
        assert_eq!(pc_weighted_depth(&[1.0, 2.0], &[0, 1], y), Some(80.0));
        // Negative loadings carry no weight; all-negative → no estimate.
        assert_eq!(pc_weighted_depth(&[-1.0, 1.0], &[0, 1], y), Some(100.0));
        assert_eq!(pc_weighted_depth(&[-1.0, -2.0], &[0, 1], y), None);
        assert_eq!(pc_weighted_depth(&[], &[], y), None);
    }
}
