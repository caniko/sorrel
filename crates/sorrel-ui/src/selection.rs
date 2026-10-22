//! Multi-cluster selection state. phy's whole UX is built on selecting
//! several clusters at once — to compare waveforms, run cross-correlograms,
//! and merge — so this is the single primitive that unblocks every
//! comparison view.

use sorrel_io::ClusterId;

/// Ordered set of selected clusters with an "anchor" used for range-extend.
///
/// Order is the order the user added each id; range-extend rewrites it.
/// Lookup is `O(n)` but `n` is typically <= a few dozen even in heavy phy
/// sessions, so a flat `Vec` is the right tradeoff over a hash set.
///
/// # Examples
///
/// ```
/// use sorrel_ui::SelectionSet;
/// use sorrel_io::ClusterId;
///
/// let mut s = SelectionSet::single(ClusterId(3));
/// s.toggle(ClusterId(7));                         // ctrl-click
/// assert_eq!(s.as_slice(), &[ClusterId(3), ClusterId(7)]);
///
/// let order: Vec<ClusterId> = (0..10).map(ClusterId).collect(); // table rows
/// s.extend_to(ClusterId(5), &order);              // shift-click — range from anchor (7)
/// assert_eq!(s.as_slice(), &[ClusterId(5), ClusterId(6), ClusterId(7)]);
///
/// s.replace(ClusterId(99));                       // plain click
/// assert_eq!(s.as_slice(), &[ClusterId(99)]);
/// assert_eq!(s.anchor(), Some(ClusterId(99)));
/// ```
#[derive(Clone, Debug, Default)]
pub struct SelectionSet {
    items: Vec<ClusterId>,
    anchor: Option<ClusterId>,
}

impl SelectionSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Single-cluster initial selection — convenient for app startup.
    pub fn single(c: ClusterId) -> Self {
        Self {
            items: vec![c],
            anchor: Some(c),
        }
    }

    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, ClusterId> {
        self.items.iter()
    }

    #[inline]
    pub fn as_slice(&self) -> &[ClusterId] {
        &self.items
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    #[inline]
    pub fn contains(&self, c: ClusterId) -> bool {
        self.items.contains(&c)
    }

    /// The anchor — the cluster used as the reference for range-extend.
    /// Equals the most recently selected single id, or `None` when empty.
    #[inline]
    pub fn anchor(&self) -> Option<ClusterId> {
        self.anchor
    }

    pub fn clear(&mut self) {
        self.items.clear();
        self.anchor = None;
    }

    /// Replace the selection with a single cluster id (the most common
    /// keyboard / left-click action).
    pub fn replace(&mut self, c: ClusterId) {
        self.items.clear();
        self.items.push(c);
        self.anchor = Some(c);
    }

    /// Toggle membership of `c`. When added, becomes the new anchor;
    /// when removed, the anchor is left alone unless we just removed it,
    /// in which case the new anchor is the last remaining item (or `None`).
    pub fn toggle(&mut self, c: ClusterId) {
        if let Some(pos) = self.items.iter().position(|&x| x == c) {
            self.items.remove(pos);
            if self.anchor == Some(c) {
                self.anchor = self.items.last().copied();
            }
        } else {
            self.items.push(c);
            self.anchor = Some(c);
        }
    }

    /// Replace the selection with the inclusive range between the current
    /// anchor and `c`, walking `order` — the cluster table's visible rows in
    /// their sorted, filtered order — so Shift-click selects what the user
    /// sees between the two rows. Falls back to a raw-id range when either
    /// end is not in `order`. With no anchor, behaves like `replace`.
    pub fn extend_to(&mut self, c: ClusterId, order: &[ClusterId]) {
        let Some(anchor) = self.anchor else {
            return self.replace(c);
        };
        let pos = |id| order.iter().position(|&x| x == id);
        self.items.clear();
        match (pos(anchor), pos(c)) {
            (Some(a), Some(b)) => {
                let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                self.items.extend_from_slice(&order[lo..=hi]);
            }
            _ => {
                let (lo, hi) = if anchor <= c {
                    (anchor, c)
                } else {
                    (c, anchor)
                };
                self.items.extend((lo.0..=hi.0).map(ClusterId));
            }
        }
        // Anchor is preserved so consecutive extend_to operations all pivot
        // around the original click — same as phy.
    }

    /// Move the anchor by `delta` rows of `order` (clamped to its bounds)
    /// and replace the selection with that single id. Used by J/K / arrow
    /// keyboard navigation, so "next" means the next *visible* table row.
    /// An empty `order` clears the selection; an anchor missing from
    /// `order` (filtered out) restarts from the first row.
    pub fn bump(&mut self, delta: i32, order: &[ClusterId]) {
        if order.is_empty() {
            self.clear();
            return;
        }
        let cur = self
            .anchor
            .and_then(|a| order.iter().position(|&x| x == a))
            .unwrap_or(0) as i64;
        let max = order.len() as i64 - 1;
        let next = (cur + delta as i64).clamp(0, max) as usize;
        self.replace(order[next]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity table order `0..n` — the pre-table-order behaviour.
    fn ids(n: u32) -> Vec<ClusterId> {
        (0..n).map(ClusterId).collect()
    }

    #[test]
    fn bump_and_extend_follow_table_order_not_ids() {
        // Table sorted by quality with cluster 4 filtered out.
        let order = [ClusterId(9), ClusterId(2), ClusterId(7), ClusterId(0)];
        let mut s = SelectionSet::single(ClusterId(2));
        s.bump(1, &order);
        assert_eq!(s.as_slice(), &[ClusterId(7)], "next visible row, not id 3");
        s.bump(1, &order);
        s.bump(1, &order);
        assert_eq!(s.as_slice(), &[ClusterId(0)], "clamps at the last row");
        s.bump(-3, &order);
        assert_eq!(s.as_slice(), &[ClusterId(9)]);

        s.extend_to(ClusterId(7), &order);
        assert_eq!(
            s.as_slice(),
            &[ClusterId(9), ClusterId(2), ClusterId(7)],
            "range covers the visible rows between anchor and target"
        );
        // Anchor filtered out of the table: restart from the first row.
        s.replace(ClusterId(4));
        s.bump(1, &order);
        assert_eq!(s.as_slice(), &[ClusterId(2)]);
        // Target not in the table: fall back to a raw-id range.
        s.replace(ClusterId(1));
        s.extend_to(ClusterId(3), &order);
        assert_eq!(s.as_slice(), &[ClusterId(1), ClusterId(2), ClusterId(3)]);
    }

    #[test]
    fn default_is_empty() {
        let s = SelectionSet::default();
        assert!(s.is_empty());
        assert_eq!(s.anchor(), None);
    }

    #[test]
    fn single_seeds_one_id_and_sets_anchor() {
        let s = SelectionSet::single(ClusterId(7));
        assert_eq!(s.as_slice(), &[ClusterId(7)]);
        assert_eq!(s.anchor(), Some(ClusterId(7)));
    }

    #[test]
    fn replace_resets_to_one_id() {
        let mut s = SelectionSet::single(ClusterId(1));
        s.toggle(ClusterId(2));
        s.toggle(ClusterId(3));
        s.replace(ClusterId(99));
        assert_eq!(s.as_slice(), &[ClusterId(99)]);
        assert_eq!(s.anchor(), Some(ClusterId(99)));
    }

    #[test]
    fn toggle_adds_then_removes() {
        let mut s = SelectionSet::new();
        s.toggle(ClusterId(5));
        assert_eq!(s.as_slice(), &[ClusterId(5)]);
        assert_eq!(s.anchor(), Some(ClusterId(5)));
        s.toggle(ClusterId(7));
        assert_eq!(s.as_slice(), &[ClusterId(5), ClusterId(7)]);
        s.toggle(ClusterId(5));
        assert_eq!(s.as_slice(), &[ClusterId(7)]);
        assert_eq!(
            s.anchor(),
            Some(ClusterId(7)),
            "anchor moved to the surviving id"
        );
    }

    #[test]
    fn toggle_removing_anchor_falls_back_to_last() {
        let mut s = SelectionSet::single(ClusterId(1));
        s.toggle(ClusterId(2));
        s.toggle(ClusterId(3));
        // anchor is 3 (last added). Toggle off 3:
        s.toggle(ClusterId(3));
        assert_eq!(s.as_slice(), &[ClusterId(1), ClusterId(2)]);
        assert_eq!(s.anchor(), Some(ClusterId(2)));
    }

    #[test]
    fn toggle_off_last_id_clears_anchor() {
        let mut s = SelectionSet::single(ClusterId(4));
        s.toggle(ClusterId(4));
        assert!(s.is_empty());
        assert_eq!(s.anchor(), None);
    }

    #[test]
    fn extend_to_builds_inclusive_range_in_either_direction() {
        let mut s = SelectionSet::single(ClusterId(3));
        s.extend_to(ClusterId(6), &ids(10));
        assert_eq!(
            s.as_slice(),
            &[ClusterId(3), ClusterId(4), ClusterId(5), ClusterId(6)]
        );
        assert_eq!(
            s.anchor(),
            Some(ClusterId(3)),
            "anchor is preserved across extend"
        );

        // Extend the other way from the same anchor — phy semantics.
        s.extend_to(ClusterId(1), &ids(10));
        assert_eq!(s.as_slice(), &[ClusterId(1), ClusterId(2), ClusterId(3)]);
        assert_eq!(s.anchor(), Some(ClusterId(3)));
    }

    #[test]
    fn extend_to_with_no_anchor_acts_like_replace() {
        let mut s = SelectionSet::new();
        s.extend_to(ClusterId(9), &ids(10));
        assert_eq!(s.as_slice(), &[ClusterId(9)]);
        assert_eq!(s.anchor(), Some(ClusterId(9)));
    }

    #[test]
    fn bump_moves_anchor_within_bounds_and_replaces_selection() {
        let mut s = SelectionSet::single(ClusterId(5));
        s.toggle(ClusterId(7));
        s.bump(1, &ids(10));
        assert_eq!(s.as_slice(), &[ClusterId(8)]); // anchor was 7, +1 -> 8
        s.bump(-100, &ids(10));
        assert_eq!(s.as_slice(), &[ClusterId(0)]);
        s.bump(100, &ids(10));
        assert_eq!(s.as_slice(), &[ClusterId(9)]);
    }

    #[test]
    fn bump_with_zero_clusters_clears_selection() {
        let mut s = SelectionSet::single(ClusterId(3));
        s.bump(1, &ids(0));
        assert!(s.is_empty());
    }

    #[test]
    fn extend_to_self_yields_single_id() {
        let mut s = SelectionSet::single(ClusterId(7));
        s.extend_to(ClusterId(7), &ids(10));
        assert_eq!(s.as_slice(), &[ClusterId(7)]);
        assert_eq!(s.anchor(), Some(ClusterId(7)));
    }

    #[test]
    fn extend_to_adjacent_id_yields_two_id_range() {
        let mut s = SelectionSet::single(ClusterId(4));
        s.extend_to(ClusterId(5), &ids(10));
        assert_eq!(s.as_slice(), &[ClusterId(4), ClusterId(5)]);
        s.extend_to(ClusterId(3), &ids(10));
        assert_eq!(s.as_slice(), &[ClusterId(3), ClusterId(4)]);
    }

    #[test]
    fn toggle_idempotency_pair() {
        // toggle(x); toggle(x) == initial state.
        let mut s = SelectionSet::single(ClusterId(1));
        let before = s.as_slice().to_vec();
        let before_anchor = s.anchor();
        s.toggle(ClusterId(99));
        s.toggle(ClusterId(99));
        assert_eq!(s.as_slice(), &before[..]);
        assert_eq!(s.anchor(), before_anchor);
    }

    #[test]
    fn replace_clears_then_anchors_in_one_step() {
        let mut s = SelectionSet::new();
        s.toggle(ClusterId(1));
        s.toggle(ClusterId(2));
        s.toggle(ClusterId(3));
        s.replace(ClusterId(7));
        assert_eq!(s.len(), 1);
        assert_eq!(s.anchor(), Some(ClusterId(7)));
    }

    #[test]
    fn clear_resets_everything() {
        let mut s = SelectionSet::single(ClusterId(5));
        s.toggle(ClusterId(7));
        s.clear();
        assert!(s.is_empty());
        assert_eq!(s.anchor(), None);
    }

    #[test]
    fn bump_at_boundaries_clamps_correctly() {
        let mut s = SelectionSet::single(ClusterId(0));
        s.bump(-1, &ids(5));
        assert_eq!(s.as_slice(), &[ClusterId(0)]);
        let mut s = SelectionSet::single(ClusterId(4));
        s.bump(1, &ids(5));
        assert_eq!(s.as_slice(), &[ClusterId(4)]);
    }

    #[test]
    fn bump_starts_at_zero_when_no_anchor() {
        let mut s = SelectionSet::new();
        s.bump(1, &ids(10));
        assert_eq!(s.as_slice(), &[ClusterId(1)]);
        assert_eq!(s.anchor(), Some(ClusterId(1)));
    }

    #[test]
    fn extend_to_with_descending_then_ascending_pivots_around_anchor() {
        let mut s = SelectionSet::single(ClusterId(5));
        s.extend_to(ClusterId(1), &ids(10));
        assert_eq!(
            s.as_slice(),
            &[
                ClusterId(1),
                ClusterId(2),
                ClusterId(3),
                ClusterId(4),
                ClusterId(5)
            ]
        );
        // Anchor stayed at 5; extending up should walk back the other way.
        s.extend_to(ClusterId(8), &ids(10));
        assert_eq!(
            s.as_slice(),
            &[ClusterId(5), ClusterId(6), ClusterId(7), ClusterId(8)]
        );
    }

    #[test]
    fn iter_yields_items_in_insertion_order() {
        let mut s = SelectionSet::new();
        s.toggle(ClusterId(7));
        s.toggle(ClusterId(3));
        s.toggle(ClusterId(11));
        let collected: Vec<_> = s.iter().copied().collect();
        assert_eq!(collected, vec![ClusterId(7), ClusterId(3), ClusterId(11)]);
    }

    #[test]
    fn contains_returns_membership_correctly() {
        let mut s = SelectionSet::single(ClusterId(3));
        s.toggle(ClusterId(7));
        assert!(s.contains(ClusterId(3)));
        assert!(s.contains(ClusterId(7)));
        assert!(!s.contains(ClusterId(0)));
        assert!(!s.contains(ClusterId(99)));
    }
}
