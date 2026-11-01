//! Cross-view brushing: what one view points at, every view draws.
//!
//! - [`SpikeBrush`]: a set of spikes (a Features lasso or k-means part, an
//!   Amplitudes lasso or threshold, ISI refractory violators, …). One brush
//!   at a time; the last view to write it wins. Esc clears it.
//! - [`TimeBrush`]: a recording time range (Shift+drag on a time axis);
//!   spike views fade (or hide) what lies outside it.
//! - [`HoverSpike`]: the spike under the pointer in any spike view, shown
//!   in the others on the next frame.
//! - [`Ghost`]: a cluster or history node previewed without selecting it
//!   (Similar / Suggest / History row hover).
//!
//! Spikes are `(cluster, local index)` pairs, the local index being the
//! spike's position in the cluster's time-sorted bucket (the index
//! `CurationCommand::Split` uses), so `session.spike_times(c)[local]` is the
//! spike's sample. A curation edit renumbers buckets, so brushes carry the
//! session version they were made against and [`Brushes::sync`] drops stale
//! ones.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use sorrel_io::ClusterId;

/// Who wrote the spike brush; shown on the brush chip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrushSource {
    FeatureLasso,
    KMeansPart,
    AmplitudeLasso,
    AmplitudeThreshold,
    SpikeDataLasso,
    TemplateFeatureLasso,
    IsiViolators,
}

impl BrushSource {
    pub fn label(self) -> &'static str {
        match self {
            Self::FeatureLasso => "Features lasso",
            Self::KMeansPart => "k-means part",
            Self::AmplitudeLasso => "Amplitudes lasso",
            Self::AmplitudeThreshold => "amplitude threshold",
            Self::SpikeDataLasso => "Spike data lasso",
            Self::TemplateFeatureLasso => "Tmpl features lasso",
            Self::IsiViolators => "ISI violators",
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpikeBrush {
    pub source: BrushSource,
    /// Per cluster, sorted ascending local indices (see [`Self::contains`]).
    pub spikes: Vec<(ClusterId, Arc<[u32]>)>,
    /// Session version the indices refer to.
    pub version: u64,
    /// Unique per built brush (see [`Self::key`]).
    generation: u64,
}

static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);

impl SpikeBrush {
    /// Build a brush; sorts and dedups each cluster's indices and drops
    /// empty clusters. `None` when nothing is left.
    pub fn new(
        source: BrushSource,
        version: u64,
        spikes: impl IntoIterator<Item = (ClusterId, Vec<u32>)>,
    ) -> Option<Self> {
        let spikes: Vec<(ClusterId, Arc<[u32]>)> = spikes
            .into_iter()
            .filter_map(|(c, mut v)| {
                v.sort_unstable();
                v.dedup();
                (!v.is_empty()).then(|| (c, Arc::from(v)))
            })
            .collect();
        (!spikes.is_empty()).then(|| Self {
            source,
            spikes,
            version,
            generation: NEXT_GENERATION.fetch_add(1, Ordering::Relaxed),
        })
    }

    /// The brushed local indices of `cluster` (empty when none).
    pub fn of(&self, cluster: ClusterId) -> &[u32] {
        self.spikes
            .iter()
            .find(|(c, _)| *c == cluster)
            .map_or(&[], |(_, v)| v)
    }

    pub fn contains(&self, cluster: ClusterId, local: u32) -> bool {
        self.of(cluster).binary_search(&local).is_ok()
    }

    pub fn len(&self) -> usize {
        self.spikes.iter().map(|(_, v)| v.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Identity for caches: changes whenever a brush is built anew (a clone
    /// keeps it). The last field is the unique generation, not an address a
    /// later allocation could reuse.
    pub fn key(&self) -> (BrushSource, u64, usize, usize) {
        (
            self.source,
            self.version,
            self.len(),
            self.generation as usize,
        )
    }
}

/// A time range in seconds of the recording.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeBrush {
    pub start: f32,
    pub end: f32,
    /// Hide (rather than fade) spikes outside the range.
    pub only: bool,
}

impl TimeBrush {
    /// Ordered range; `None` when narrower than 1 ms (a click, not a drag).
    pub fn new(a: f32, b: f32) -> Option<Self> {
        let (start, end) = (a.min(b), a.max(b));
        (end - start >= 1e-3).then_some(Self {
            start,
            end,
            only: false,
        })
    }

    pub fn contains(&self, t: f32) -> bool {
        (self.start..=self.end).contains(&t)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HoverSpike {
    pub cluster: ClusterId,
    pub local: u32,
    /// Sample index of the spike (what Trace / Seek use).
    pub sample: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ghost {
    /// A cluster previewed as a merge candidate (Similar / Suggest hover).
    Merge(ClusterId),
    /// A history node previewed (History hover).
    History(u64),
}

/// The app-wide brush state, handed to every pane through `ViewBehavior`.
/// Panes read the current values and write the `*_next` / spike fields.
#[derive(Debug, Default)]
pub struct Brushes {
    pub spikes: Option<SpikeBrush>,
    pub time: Option<TimeBrush>,
    /// Hovered spike / ghost from the previous frame (read these).
    pub hover: Option<HoverSpike>,
    pub ghost: Option<Ghost>,
    /// Written by panes this frame; promoted by [`Brushes::end_frame`].
    pub hover_next: Option<HoverSpike>,
    pub ghost_next: Option<Ghost>,
    /// The session version of the last [`Brushes::sync`].
    pub(crate) synced: Option<u64>,
}

impl Brushes {
    /// Start of frame: drop a spike brush made against an older session
    /// version (its local indices no longer mean the same spikes), and the
    /// hovered spike with it (a bucket index too).
    pub fn sync(&mut self, version: u64) {
        if self.spikes.as_ref().is_some_and(|b| b.version != version) {
            self.spikes = None;
        }
        if self.synced.replace(version).is_some_and(|v| v != version) {
            self.hover = None;
            self.hover_next = None;
        }
    }

    /// End of frame: what was hovered this frame becomes `hover` / `ghost`.
    /// Returns whether either changed (the caller repaints then).
    pub fn end_frame(&mut self) -> bool {
        let (h, g) = (self.hover_next.take(), self.ghost_next.take());
        let changed = h != self.hover || g != self.ghost;
        (self.hover, self.ghost) = (h, g);
        changed
    }

    /// Esc: clear the spike brush, else the time brush. Returns whether
    /// anything was cleared.
    pub fn cancel(&mut self) -> bool {
        self.spikes.take().is_some() || self.time.take().is_some()
    }

    /// Whether a spike at `t` seconds is inside the time brush (always true
    /// without one).
    pub fn in_time(&self, t: f32) -> bool {
        self.time.is_none_or(|b| b.contains(t))
    }
}

/// The brush chip(s) for the top bar: "brush: 24 spikes · ISI violators ×"
/// and "time 12.0–48.5 s ×". Returns nothing; clicks clear in place.
pub fn chips(ui: &mut egui::Ui, b: &mut Brushes) {
    if let Some(s) = &b.spikes {
        let text = format!("brush: {} spikes · {}  ×", s.len(), s.source.label());
        let resp = ui
            .add(egui::Button::new(text).corner_radius(10.0).selected(true))
            .on_hover_text("Highlighted in every spike view · click or Esc clears");
        if resp.clicked() {
            b.spikes = None;
        }
    }
    if let Some(t) = &mut b.time {
        let text = format!("time {:.1}–{:.1} s  ×", t.start, t.end);
        let resp = ui
            .add(egui::Button::new(text).corner_radius(10.0).selected(true))
            .on_hover_text("Spikes outside the range fade · click or Esc clears");
        let mut only = t.only;
        ui.checkbox(&mut only, "only in range")
            .on_hover_text("Hide spikes outside the time range instead of fading them");
        t.only = only;
        if resp.clicked() {
            b.time = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brush_sorts_dedups_and_drops_empty_and_stale() {
        let b = SpikeBrush::new(
            BrushSource::IsiViolators,
            7,
            [
                (ClusterId(1), vec![5, 1, 5, 3]),
                (ClusterId(2), vec![]),
                (ClusterId(4), vec![0]),
            ],
        )
        .unwrap();
        assert_eq!(b.of(ClusterId(1)), &[1, 3, 5]);
        assert!(b.contains(ClusterId(4), 0) && !b.contains(ClusterId(1), 2));
        assert!(b.of(ClusterId(2)).is_empty() && b.len() == 4);
        assert!(SpikeBrush::new(BrushSource::KMeansPart, 0, [(ClusterId(0), vec![])]).is_none());

        let mut all = Brushes {
            spikes: Some(b),
            ..Default::default()
        };
        all.sync(7);
        assert!(all.spikes.is_some(), "same version keeps it");
        all.hover = Some(HoverSpike {
            cluster: ClusterId(1),
            local: 3,
            sample: 9,
        });
        all.sync(8);
        assert!(all.spikes.is_none(), "an edit drops it");
        assert!(all.hover.is_none(), "and the hovered bucket index");
    }

    #[test]
    fn key_is_unique_per_built_brush_and_kept_by_clones() {
        let make =
            || SpikeBrush::new(BrushSource::KMeansPart, 1, [(ClusterId(1), vec![0, 1])]).unwrap();
        let (a, b) = (make(), make());
        assert_ne!(a.key(), b.key(), "same content, different brush");
        assert_eq!(a.key(), a.clone().key());
        drop(a);
        assert_ne!(b.key(), make().key(), "no reuse after a drop");
    }

    #[test]
    fn time_brush_orders_and_ignores_clicks() {
        let t = TimeBrush::new(5.0, 2.0).unwrap();
        assert_eq!((t.start, t.end), (2.0, 5.0));
        assert!(t.contains(2.0) && t.contains(5.0) && !t.contains(5.1));
        assert!(TimeBrush::new(1.0, 1.0005).is_none());
        let mut b = Brushes::default();
        assert!(b.in_time(1e9), "no brush admits everything");
        b.time = Some(t);
        assert!(!b.in_time(1.0) && b.cancel() && !b.cancel());
    }

    #[test]
    fn hover_is_promoted_at_end_of_frame() {
        let mut b = Brushes::default();
        let h = HoverSpike {
            cluster: ClusterId(3),
            local: 9,
            sample: 1000,
        };
        b.hover_next = Some(h);
        assert!(b.end_frame() && b.hover == Some(h));
        assert!(
            b.end_frame() && b.hover.is_none(),
            "nothing hovered clears it"
        );
        assert!(!b.end_frame());
    }
}
