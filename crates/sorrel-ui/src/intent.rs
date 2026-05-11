use sorrel_data::PhyLabelOp;
use sorrel_io::ClusterId;

/// User intents — a closed enum, dispatched in a single `match`. No heap, no
/// virtual calls in the input path.
///
/// `Relabel` carries only the op, not a cluster id, because the dispatcher
/// applies it to the entire active selection. `SelectCluster` and friends
/// drive that selection.
#[derive(Copy, Clone, Debug)]
pub enum Intent {
    /// Replace the active selection with this single id (plain click / J/K).
    SelectCluster(ClusterId),
    /// Toggle membership of `id` in the active selection (Ctrl/Cmd-click).
    ToggleCluster(ClusterId),
    /// Range-extend from the anchor to `id` (Shift-click).
    ExtendCluster(ClusterId),
    /// Apply this label transition to every cluster in the active selection.
    Relabel(PhyLabelOp),
    Undo,
    Redo,
    /// Persist the curated state (spike_clusters.npy + cluster_group.tsv).
    Save,
    NextCluster,
    PrevCluster,
    PageBack,
    PageForward,
}

pub fn from_key(key: egui::Key, modifiers: egui::Modifiers) -> Option<Intent> {
    from_key_in_context(key, modifiers, false)
}

/// Like [`from_key`] but suppresses single-letter intents (relabel, navigation,
/// pan) when a text input has focus, while still honouring command-modified
/// shortcuts (Save / Undo / Redo). Without this gate, typing into a filter
/// box would relabel selected clusters mid-keystroke.
pub fn from_key_in_context(
    key: egui::Key,
    modifiers: egui::Modifiers,
    text_input_active: bool,
) -> Option<Intent> {
    use egui::Key::*;
    // Cmd/Ctrl-modified intents are always global — they survive text focus
    // because phy/spike-sorting tooling expects Save/Undo/Redo to work even
    // while a search field is active.
    match key {
        Z if modifiers.command && modifiers.shift => return Some(Intent::Redo),
        Z if modifiers.command => return Some(Intent::Undo),
        S if modifiers.command => return Some(Intent::Save),
        _ => {}
    }
    if text_input_active {
        return None;
    }
    Some(match key {
        G => Intent::Relabel(PhyLabelOp::SetGood),
        M => Intent::Relabel(PhyLabelOp::SetMua),
        N => Intent::Relabel(PhyLabelOp::SetNoise),
        U => Intent::Relabel(PhyLabelOp::SetUnsorted),
        ArrowDown | J => Intent::NextCluster,
        ArrowUp | K => Intent::PrevCluster,
        ArrowLeft | H => Intent::PageBack,
        ArrowRight | L => Intent::PageForward,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Key, Modifiers};

    #[test]
    fn label_keys_emit_relabel_with_just_the_op() {
        let m = Modifiers::default();
        assert!(matches!(
            from_key(Key::G, m),
            Some(Intent::Relabel(PhyLabelOp::SetGood))
        ));
        assert!(matches!(
            from_key(Key::M, m),
            Some(Intent::Relabel(PhyLabelOp::SetMua))
        ));
        assert!(matches!(
            from_key(Key::N, m),
            Some(Intent::Relabel(PhyLabelOp::SetNoise))
        ));
        assert!(matches!(
            from_key(Key::U, m),
            Some(Intent::Relabel(PhyLabelOp::SetUnsorted))
        ));
    }

    #[test]
    fn navigation_keys_emit_cluster_and_pan_intents() {
        let m = Modifiers::default();
        assert!(matches!(from_key(Key::J, m), Some(Intent::NextCluster)));
        assert!(matches!(
            from_key(Key::ArrowDown, m),
            Some(Intent::NextCluster)
        ));
        assert!(matches!(from_key(Key::K, m), Some(Intent::PrevCluster)));
        assert!(matches!(
            from_key(Key::ArrowUp, m),
            Some(Intent::PrevCluster)
        ));
        assert!(matches!(from_key(Key::H, m), Some(Intent::PageBack)));
        assert!(matches!(
            from_key(Key::ArrowLeft, m),
            Some(Intent::PageBack)
        ));
        assert!(matches!(from_key(Key::L, m), Some(Intent::PageForward)));
        assert!(matches!(
            from_key(Key::ArrowRight, m),
            Some(Intent::PageForward)
        ));
    }

    #[test]
    fn cmd_z_undoes_and_cmd_shift_z_redoes() {
        let m = Modifiers {
            command: true,
            ..Default::default()
        };
        assert!(matches!(from_key(Key::Z, m), Some(Intent::Undo)));

        let redo = Modifiers {
            command: true,
            shift: true,
            ..Default::default()
        };
        assert!(matches!(from_key(Key::Z, redo), Some(Intent::Redo)));
    }

    #[test]
    fn cmd_s_emits_save() {
        let m = Modifiers {
            command: true,
            ..Default::default()
        };
        assert!(matches!(from_key(Key::S, m), Some(Intent::Save)));
        // Bare `s` is not an intent.
        let m = Modifiers::default();
        assert!(from_key(Key::S, m).is_none());
    }

    #[test]
    fn shift_alone_does_not_change_label_intents() {
        // Shift+G should still emit a relabel — phy emits the same intent
        // regardless of shift state on label keys.
        let m = Modifiers {
            shift: true,
            ..Default::default()
        };
        assert!(matches!(
            from_key(Key::G, m),
            Some(Intent::Relabel(PhyLabelOp::SetGood))
        ));
    }

    #[test]
    fn cmd_modifier_does_not_swallow_label_keys() {
        // Cmd+G is not a recognised binding; it should still produce the
        // relabel intent because we don't filter on cmd for label keys.
        let m = Modifiers {
            command: true,
            ..Default::default()
        };
        assert!(matches!(
            from_key(Key::G, m),
            Some(Intent::Relabel(PhyLabelOp::SetGood))
        ));
    }

    #[test]
    fn arrow_keys_alias_to_jklh() {
        let m = Modifiers::default();
        let down = from_key(Key::ArrowDown, m);
        let j = from_key(Key::J, m);
        assert!(matches!(down, Some(Intent::NextCluster)));
        assert!(matches!(j, Some(Intent::NextCluster)));

        let up = from_key(Key::ArrowUp, m);
        let k = from_key(Key::K, m);
        assert!(matches!(up, Some(Intent::PrevCluster)));
        assert!(matches!(k, Some(Intent::PrevCluster)));

        let left = from_key(Key::ArrowLeft, m);
        let h = from_key(Key::H, m);
        assert!(matches!(left, Some(Intent::PageBack)));
        assert!(matches!(h, Some(Intent::PageBack)));

        let right = from_key(Key::ArrowRight, m);
        let l = from_key(Key::L, m);
        assert!(matches!(right, Some(Intent::PageForward)));
        assert!(matches!(l, Some(Intent::PageForward)));
    }

    #[test]
    fn ctrl_shift_z_emits_redo_via_command_modifier() {
        // egui's `Modifiers::command` is the platform's "primary" modifier:
        // ctrl on Linux/Windows, cmd on macOS. With shift it's redo.
        let m = Modifiers {
            command: true,
            shift: true,
            ..Default::default()
        };
        assert!(matches!(from_key(Key::Z, m), Some(Intent::Redo)));
    }

    #[test]
    fn shift_alone_z_is_not_an_intent() {
        // Shift-Z without command should not be undo/redo.
        let m = Modifiers {
            shift: true,
            ..Default::default()
        };
        assert!(from_key(Key::Z, m).is_none());
    }

    #[test]
    fn unrecognised_keys_with_modifiers_still_return_none() {
        let m = Modifiers {
            command: true,
            shift: true,
            alt: true,
            ..Default::default()
        };
        assert!(from_key(Key::Q, m).is_none());
    }

    #[test]
    fn bare_z_is_not_an_intent() {
        let m = Modifiers::default();
        assert!(from_key(Key::Z, m).is_none());
    }

    #[test]
    fn unrecognised_key_returns_none() {
        let m = Modifiers::default();
        assert!(from_key(Key::Q, m).is_none());
    }

    #[test]
    fn text_input_focus_suppresses_relabel_keys() {
        let m = Modifiers::default();
        // Without focus, G relabels.
        assert!(matches!(
            from_key_in_context(Key::G, m, false),
            Some(Intent::Relabel(_))
        ));
        // With focus, plain letter keys are passed through to the widget
        // instead of becoming intents.
        assert!(from_key_in_context(Key::G, m, true).is_none());
        assert!(from_key_in_context(Key::M, m, true).is_none());
        assert!(from_key_in_context(Key::J, m, true).is_none());
        assert!(from_key_in_context(Key::ArrowDown, m, true).is_none());
    }

    #[test]
    fn text_input_focus_preserves_command_shortcuts() {
        // Save/Undo/Redo should still work while typing in a filter box —
        // matches phy and standard editor conventions.
        let m = Modifiers {
            command: true,
            ..Default::default()
        };
        assert!(matches!(
            from_key_in_context(Key::S, m, true),
            Some(Intent::Save)
        ));
        assert!(matches!(
            from_key_in_context(Key::Z, m, true),
            Some(Intent::Undo)
        ));
        let redo = Modifiers {
            command: true,
            shift: true,
            ..Default::default()
        };
        assert!(matches!(
            from_key_in_context(Key::Z, redo, true),
            Some(Intent::Redo)
        ));
    }
}
