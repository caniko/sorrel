use sorrel_data::PhyLabelOp;
use sorrel_io::ClusterId;

/// User intents — a closed enum, dispatched in a single `match`. No heap, no
/// virtual calls in the input path.
///
/// `Move` carries only who and the op, not cluster ids, because the
/// dispatcher resolves the targets from the current best/similar
/// selection. `SelectCluster` and friends drive that selection.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Intent {
    /// Replace the active selection with this single id (plain click).
    SelectCluster(ClusterId),
    /// Toggle membership of `id` in the active selection (Ctrl/Cmd-click).
    ToggleCluster(ClusterId),
    /// Range-extend from the anchor to `id` (Shift-click).
    ExtendCluster(ClusterId),
    /// phy `move_{best,similar,all}_to_{group}`: label those clusters, then
    /// advance the wizard.
    Move(MoveTarget, PhyLabelOp),
    Undo,
    Redo,
    /// Persist the curated state (spike_clusters.npy + cluster_group.tsv).
    Save,
    /// phy `next_best` / `previous_best` / `first` / `last` over the
    /// visible cluster-table rows.
    NextCluster,
    PrevCluster,
    FirstCluster,
    LastCluster,
    /// phy `next` / `previous` (Space): step through the similar clusters.
    NextSimilar,
    PrevSimilar,
    /// phy `select_first_similar` (Ctrl+Space).
    SelectFirstSimilar,
    /// phy `unselect_similar` (Backspace).
    UnselectSimilar,
    /// phy `reset_wizard` (Ctrl+Alt+Space): select the first table row.
    ResetWizard,
    PageBack,
    PageForward,
    TraceStart,
    TraceEnd,
    TraceNarrow,
    TraceWiden,
    /// phy `toggle_spike_reorder` (Ctrl+R).
    ToggleSpikeReorder,
    /// phy v2.1 `toggle_merge_mode` (`v`).
    ToggleMergeMode,
    /// phy TraceView toggles: `switch_origin` (Alt+O), `toggle_highlighted_spikes`
    /// (Alt+S), `toggle_show_labels` (Alt+L), and the gain (Ctrl+Alt+↑/↓).
    TraceOrigin,
    TraceAllSpikes,
    TraceLabels,
    TraceGain(i32),
    /// phy `go_to_next_spike` / `go_to_previous_spike` (Alt+PgDn / Alt+PgUp):
    /// centre the trace on the selection's next / previous spike.
    NextSpike,
    PrevSpike,
    /// Merge every selected cluster into the first-selected one (`g`).
    MergeSelection,
    /// Escape: dismiss the confirm dialog, the help overlay, the snippet
    /// bar, or an in-progress lasso — whichever is active.
    Cancel,
    /// Toggle the keyboard/mouse help overlay.
    ToggleHelp,
    /// phy `split` (`k`): split off the spikes inside the Features lasso.
    Split,
    /// phy `split_init` (Shift+Ctrl+K): one cluster per original template.
    SplitInit,
    /// phy `toggle_automatic_channel_selection` (`c`) in the Features pane.
    ToggleFeatureChannels,
    /// phy `next_amplitudes_type` / `previous_amplitudes_type` (`a` / Shift+`a`).
    CycleAmplitudes(i32),
    /// phy snippet mode (`:`), opened with this text already typed.
    Snippet(&'static str),
    Quit,
}

/// Which clusters a phy `move_*` action labels.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MoveTarget {
    Best,
    Similar,
    All,
}

/// Help-window grouping.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Section {
    Curation,
    Navigation,
    Views,
}

/// One key chord. `shift: None` accepts either state, for symbols that
/// need Shift on common layouts (`?`, `:`, `+`).
#[derive(Copy, Clone, Debug)]
pub struct Binding {
    pub key: egui::Key,
    pub ctrl: bool,
    pub alt: bool,
    pub shift: Option<bool>,
    pub intent: Intent,
    pub section: Section,
    pub help: &'static str,
}

const fn bind(
    key: egui::Key,
    mods: u8,
    intent: Intent,
    section: Section,
    help: &'static str,
) -> Binding {
    Binding {
        key,
        ctrl: mods & CTRL != 0,
        alt: mods & ALT != 0,
        shift: if mods & ANY_SHIFT != 0 {
            None
        } else {
            Some(mods & SHIFT != 0)
        },
        intent,
        section,
        help,
    }
}

const NONE: u8 = 0;
const CTRL: u8 = 1;
const ALT: u8 = 2;
const SHIFT: u8 = 4;
const ANY_SHIFT: u8 = 8;

use egui::Key as K;
use MoveTarget::{All, Best, Similar};
use PhyLabelOp::{SetGood, SetMua, SetNoise, SetUnsorted};
use Section::{Curation as Cur, Navigation as Nav, Views};

/// phy's default TemplateGUI shortcuts (cortex-lab/phy master, v2.1).
/// Entries sharing a `help` text show as one help row.
pub const KEYMAP: &[Binding] = &[
    bind(
        K::G,
        NONE,
        Intent::MergeSelection,
        Cur,
        "merge the selected clusters",
    ),
    bind(
        K::K,
        NONE,
        Intent::Split,
        Cur,
        "split off the spikes inside the lasso",
    ),
    bind(
        K::K,
        CTRL | SHIFT,
        Intent::SplitInit,
        Cur,
        "split the selection by original template",
    ),
    bind(
        K::R,
        CTRL,
        Intent::ToggleSpikeReorder,
        Views,
        "time axes: toggle spike_reorder.npy",
    ),
    bind(
        K::V,
        NONE,
        Intent::ToggleMergeMode,
        Cur,
        "merge mode: stage the selection, G merges, V cancels",
    ),
    bind(
        K::L,
        NONE,
        Intent::Snippet(":l "),
        Cur,
        "set a label (:l name value)",
    ),
    bind(
        K::G,
        ALT,
        Intent::Move(Best, SetGood),
        Cur,
        "move best to good · mua · noise · unsorted",
    ),
    bind(
        K::M,
        ALT,
        Intent::Move(Best, SetMua),
        Cur,
        "move best to good · mua · noise · unsorted",
    ),
    bind(
        K::N,
        ALT,
        Intent::Move(Best, SetNoise),
        Cur,
        "move best to good · mua · noise · unsorted",
    ),
    bind(
        K::U,
        ALT,
        Intent::Move(Best, SetUnsorted),
        Cur,
        "move best to good · mua · noise · unsorted",
    ),
    bind(
        K::G,
        CTRL,
        Intent::Move(Similar, SetGood),
        Cur,
        "move similar to good · mua · noise · unsorted",
    ),
    bind(
        K::M,
        CTRL,
        Intent::Move(Similar, SetMua),
        Cur,
        "move similar to good · mua · noise · unsorted",
    ),
    bind(
        K::N,
        CTRL,
        Intent::Move(Similar, SetNoise),
        Cur,
        "move similar to good · mua · noise · unsorted",
    ),
    bind(
        K::U,
        CTRL,
        Intent::Move(Similar, SetUnsorted),
        Cur,
        "move similar to good · mua · noise · unsorted",
    ),
    bind(
        K::G,
        CTRL | ALT,
        Intent::Move(All, SetGood),
        Cur,
        "move all to good · mua · noise · unsorted",
    ),
    bind(
        K::M,
        CTRL | ALT,
        Intent::Move(All, SetMua),
        Cur,
        "move all to good · mua · noise · unsorted",
    ),
    bind(
        K::N,
        CTRL | ALT,
        Intent::Move(All, SetNoise),
        Cur,
        "move all to good · mua · noise · unsorted",
    ),
    bind(
        K::U,
        CTRL | ALT,
        Intent::Move(All, SetUnsorted),
        Cur,
        "move all to good · mua · noise · unsorted",
    ),
    bind(K::Z, CTRL, Intent::Undo, Cur, "undo"),
    bind(K::Z, CTRL | SHIFT, Intent::Redo, Cur, "redo"),
    bind(K::Y, CTRL, Intent::Redo, Cur, "redo"),
    bind(K::S, CTRL, Intent::Save, Cur, "save"),
    bind(
        K::ArrowDown,
        NONE,
        Intent::NextCluster,
        Nav,
        "next · previous best cluster",
    ),
    bind(
        K::ArrowUp,
        NONE,
        Intent::PrevCluster,
        Nav,
        "next · previous best cluster",
    ),
    bind(
        K::Home,
        NONE,
        Intent::FirstCluster,
        Nav,
        "first · last best cluster",
    ),
    bind(
        K::End,
        NONE,
        Intent::LastCluster,
        Nav,
        "first · last best cluster",
    ),
    bind(
        K::Space,
        NONE,
        Intent::NextSimilar,
        Nav,
        "next · previous similar cluster",
    ),
    bind(
        K::Space,
        SHIFT,
        Intent::PrevSimilar,
        Nav,
        "next · previous similar cluster",
    ),
    bind(
        K::Space,
        CTRL,
        Intent::SelectFirstSimilar,
        Nav,
        "select the first 15 similar clusters",
    ),
    bind(
        K::Backspace,
        NONE,
        Intent::UnselectSimilar,
        Nav,
        "unselect the similar clusters",
    ),
    bind(
        K::Space,
        CTRL | ALT,
        Intent::ResetWizard,
        Nav,
        "reset the wizard (first cluster)",
    ),
    bind(
        K::ArrowLeft,
        ALT,
        Intent::PageBack,
        Nav,
        "trace: previous · next window",
    ),
    bind(
        K::ArrowRight,
        ALT,
        Intent::PageForward,
        Nav,
        "trace: previous · next window",
    ),
    bind(
        K::Home,
        ALT,
        Intent::TraceStart,
        Nav,
        "trace: start · end of recording",
    ),
    bind(
        K::End,
        ALT,
        Intent::TraceEnd,
        Nav,
        "trace: start · end of recording",
    ),
    bind(
        K::Plus,
        ALT | ANY_SHIFT,
        Intent::TraceNarrow,
        Nav,
        "trace: narrow · widen the window",
    ),
    bind(
        K::Equals,
        ALT | ANY_SHIFT,
        Intent::TraceNarrow,
        Nav,
        "trace: narrow · widen the window",
    ),
    bind(
        K::Minus,
        ALT,
        Intent::TraceWiden,
        Nav,
        "trace: narrow · widen the window",
    ),
    bind(
        K::PageDown,
        ALT,
        Intent::NextSpike,
        Nav,
        "trace: next · previous spike of the selection",
    ),
    bind(
        K::PageUp,
        ALT,
        Intent::PrevSpike,
        Nav,
        "trace: next · previous spike of the selection",
    ),
    bind(
        K::T,
        ALT,
        Intent::Snippet(":tg "),
        Nav,
        "trace: go to a time (:tg seconds)",
    ),
    bind(
        K::O,
        ALT,
        Intent::TraceOrigin,
        Nav,
        "trace: channel 0 at the top or bottom",
    ),
    bind(
        K::S,
        ALT,
        Intent::TraceAllSpikes,
        Nav,
        "trace: mark every cluster's spikes or the selection's",
    ),
    bind(
        K::L,
        ALT,
        Intent::TraceLabels,
        Nav,
        "trace: show or hide channel labels",
    ),
    bind(
        K::ArrowUp,
        CTRL | ALT,
        Intent::TraceGain(1),
        Nav,
        "trace: increase · decrease the gain",
    ),
    bind(
        K::ArrowDown,
        CTRL | ALT,
        Intent::TraceGain(-1),
        Nav,
        "trace: increase · decrease the gain",
    ),
    bind(
        K::C,
        NONE,
        Intent::ToggleFeatureChannels,
        Views,
        "Features: fix or release the channels",
    ),
    bind(
        K::A,
        NONE,
        Intent::CycleAmplitudes(1),
        Views,
        "Amplitudes: next · previous type",
    ),
    bind(
        K::A,
        SHIFT,
        Intent::CycleAmplitudes(-1),
        Views,
        "Amplitudes: next · previous type",
    ),
    bind(
        K::Colon,
        ANY_SHIFT,
        Intent::Snippet(":"),
        Views,
        "snippet mode (:g, :c 3 7, :s n_spikes, …)",
    ),
    bind(K::H, NONE, Intent::ToggleHelp, Views, "show this help"),
    bind(
        K::Questionmark,
        ANY_SHIFT,
        Intent::ToggleHelp,
        Views,
        "show this help",
    ),
    bind(K::F1, NONE, Intent::ToggleHelp, Views, "show this help"),
    bind(K::Q, CTRL, Intent::Quit, Views, "quit"),
    bind(
        K::Escape,
        NONE,
        Intent::Cancel,
        Views,
        "close a dialog, the snippet bar or the lasso",
    ),
];

impl Binding {
    fn matches(&self, key: egui::Key, m: egui::Modifiers) -> bool {
        self.key == key
            && self.ctrl == m.command
            && self.alt == m.alt
            && self.shift.is_none_or(|s| s == m.shift)
    }

    /// Fires while a text field has focus: Ctrl chords, Esc and F1.
    fn is_global(&self) -> bool {
        self.ctrl || matches!(self.key, K::Escape | K::F1)
    }

    /// `Ctrl+Alt+G`-style label for the help window.
    pub fn chord(&self) -> String {
        let mut s = String::new();
        if self.ctrl {
            s.push_str("Ctrl+");
        }
        if self.alt {
            s.push_str("Alt+");
        }
        if self.shift == Some(true) {
            s.push_str("Shift+");
        }
        s.push_str(match self.key {
            K::Plus => "+",
            K::Equals => "=",
            K::Minus => "-",
            K::Colon => ":",
            K::Questionmark => "?",
            K::ArrowDown => "↓",
            K::ArrowUp => "↑",
            K::ArrowLeft => "←",
            K::ArrowRight => "→",
            k => k.name(),
        });
        s
    }
}

/// Help rows `(section, chords, help)`, bindings with the same help text
/// joined, in keymap order.
pub fn help_rows() -> Vec<(Section, String, &'static str)> {
    let mut rows: Vec<(Section, String, &'static str)> = Vec::new();
    for b in KEYMAP {
        match rows.iter_mut().find(|r| r.2 == b.help) {
            Some(row) => {
                row.1.push_str(" · ");
                row.1.push_str(&b.chord());
            }
            None => rows.push((b.section, b.chord(), b.help)),
        }
    }
    rows
}

pub fn from_key(key: egui::Key, modifiers: egui::Modifiers) -> Option<Intent> {
    from_key_in_context(key, modifiers, false)
}

/// Like [`from_key`] but suppresses plain-key intents (merge, split,
/// navigation) while a text input has focus, so typing into the filter box
/// never merges clusters mid-keystroke; Ctrl chords, Esc and F1 still fire.
pub fn from_key_in_context(
    key: egui::Key,
    modifiers: egui::Modifiers,
    text_input_active: bool,
) -> Option<Intent> {
    KEYMAP
        .iter()
        .find(|b| b.matches(key, modifiers) && (!text_input_active || b.is_global()))
        .map(|b| b.intent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Key, Modifiers};

    fn mods(ctrl: bool, alt: bool, shift: bool) -> Modifiers {
        Modifiers {
            command: ctrl,
            ctrl,
            alt,
            shift,
            ..Default::default()
        }
    }

    #[test]
    fn every_binding_resolves_to_itself_and_chords_are_unique() {
        for b in KEYMAP {
            let m = mods(b.ctrl, b.alt, b.shift.unwrap_or(false));
            assert_eq!(from_key(b.key, m), Some(b.intent), "{}", b.chord());
            let clashes = KEYMAP.iter().filter(|o| o.matches(b.key, m)).count();
            assert_eq!(clashes, 1, "{} is bound twice", b.chord());
        }
    }

    #[test]
    fn phy_supervisor_keys() {
        let none = Modifiers::default();
        assert_eq!(from_key(Key::G, none), Some(Intent::MergeSelection));
        assert_eq!(from_key(Key::K, none), Some(Intent::Split));
        assert_eq!(
            from_key(Key::G, mods(false, true, false)),
            Some(Intent::Move(Best, SetGood))
        );
        assert_eq!(
            from_key(Key::N, mods(true, false, false)),
            Some(Intent::Move(Similar, SetNoise))
        );
        assert_eq!(
            from_key(Key::M, mods(true, true, false)),
            Some(Intent::Move(All, SetMua))
        );
        assert_eq!(from_key(Key::Space, none), Some(Intent::NextSimilar));
        assert_eq!(
            from_key(Key::Space, mods(false, false, true)),
            Some(Intent::PrevSimilar)
        );
        assert_eq!(
            from_key(Key::Space, mods(true, true, false)),
            Some(Intent::ResetWizard)
        );
        assert_eq!(
            from_key(Key::Questionmark, mods(false, false, true)),
            Some(Intent::ToggleHelp)
        );
        // Sorrel's old single-letter relabel keys are gone: plain M and N
        // do nothing, and Shift+G is not a merge.
        assert_eq!(from_key(Key::M, none), None);
        assert_eq!(from_key(Key::G, mods(false, false, true)), None);
    }

    #[test]
    fn text_focus_keeps_only_global_chords() {
        let none = Modifiers::default();
        assert_eq!(from_key_in_context(Key::G, none, true), None);
        assert_eq!(from_key_in_context(Key::Space, none, true), None);
        assert_eq!(from_key_in_context(Key::ArrowDown, none, true), None);
        assert_eq!(
            from_key_in_context(Key::G, mods(false, true, false), true),
            None
        );
        assert_eq!(
            from_key_in_context(Key::S, mods(true, false, false), true),
            Some(Intent::Save)
        );
        assert_eq!(
            from_key_in_context(Key::Z, mods(true, false, true), true),
            Some(Intent::Redo)
        );
        assert_eq!(
            from_key_in_context(Key::Escape, none, true),
            Some(Intent::Cancel)
        );
    }

    #[test]
    fn help_rows_join_chords_that_share_a_description() {
        let rows = help_rows();
        let redo = rows.iter().find(|r| r.2 == "redo").unwrap();
        assert_eq!(redo.1, "Ctrl+Shift+Z · Ctrl+Y");
        let best = rows.iter().find(|r| r.2.starts_with("move best")).unwrap();
        assert_eq!(best.1, "Alt+G · Alt+M · Alt+N · Alt+U");
    }
}
