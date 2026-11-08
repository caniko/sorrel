//! The command registry: defaults, configuration, dispatch and displayed hints
//! all use this table. Mouse gestures and native text editing are not commands.

use crate::intent::{Intent, MoveTarget};
use egui::{Event, Key, Modifiers};
use sorrel_data::PhyLabelOp;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    Application,
    Window,
    Recording,
    Waveforms,
    Ccg,
    Submit,
}

impl Scope {
    pub fn name(self) -> &'static str {
        match self {
            Self::Application => "Application",
            Self::Window => "Window",
            Self::Recording => "Recording",
            Self::Waveforms => "Waveforms (hovered)",
            Self::Ccg => "CCG (hovered)",
            Self::Submit => "Dialogs and text commands",
        }
    }

    fn overlaps(self, other: Self) -> bool {
        self == other
            || !matches!(
                (self, other),
                (Self::Waveforms, Self::Ccg) | (Self::Ccg, Self::Waveforms)
            )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShellAction {
    Open,
    Close,
    Next,
    Previous,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaveformAction {
    Next,
    Previous,
    Mean,
    Overlap,
    Labels,
    AmplitudeUp,
    AmplitudeDown,
    BoxWider,
    BoxNarrower,
    SpreadUp,
    SpreadDown,
    SpreadWider,
    SpreadNarrower,
    Reset,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Action {
    Intent(Intent),
    Shell(ShellAction),
    Waveform(WaveformAction),
    CcgNormalize,
    Fullscreen,
    Settings,
    Submit,
}

pub struct Command {
    pub id: &'static str,
    pub description: &'static str,
    pub scope: Scope,
    pub action: Action,
    pub defaults: &'static [&'static str],
}

macro_rules! command {
    ($id:literal, $description:literal, $scope:ident, $action:expr, $($key:literal),* $(,)?) => {
        Command { id: $id, description: $description, scope: Scope::$scope,
            action: $action, defaults: &[$($key),*] }
    };
}

use Action::{Intent as I, Shell as S, Waveform as W};
use Intent::*;
use MoveTarget::{All, Best, Similar};
use PhyLabelOp::{SetGood, SetMua, SetNoise, SetUnsorted};

/// Stable IDs are the TOML keys. Defaults occur only here.
pub const COMMANDS: &[Command] = &[
    command!(
        "curation.merge",
        "Merge selected clusters",
        Recording,
        I(MergeSelection),
        "G"
    ),
    command!(
        "curation.split",
        "Split off spikes inside the lasso",
        Recording,
        I(Split),
        "K"
    ),
    command!(
        "curation.split_original",
        "Split by original template",
        Recording,
        I(SplitInit),
        "Primary+Shift+K"
    ),
    command!(
        "curation.merge_mode",
        "Toggle merge mode",
        Recording,
        I(ToggleMergeMode),
        "V"
    ),
    command!(
        "curation.label",
        "Set a label (:l name value)",
        Recording,
        I(Snippet(":l ")),
        "L"
    ),
    command!(
        "curation.best_good",
        "Move best clusters to good",
        Recording,
        I(Move(Best, SetGood)),
        "Alt+G"
    ),
    command!(
        "curation.best_mua",
        "Move best clusters to MUA",
        Recording,
        I(Move(Best, SetMua)),
        "Alt+M"
    ),
    command!(
        "curation.best_noise",
        "Move best clusters to noise",
        Recording,
        I(Move(Best, SetNoise)),
        "Alt+N"
    ),
    command!(
        "curation.best_unsorted",
        "Move best clusters to unsorted",
        Recording,
        I(Move(Best, SetUnsorted)),
        "Alt+U"
    ),
    command!(
        "curation.similar_good",
        "Move similar clusters to good",
        Recording,
        I(Move(Similar, SetGood)),
        "Primary+G"
    ),
    command!(
        "curation.similar_mua",
        "Move similar clusters to MUA",
        Recording,
        I(Move(Similar, SetMua)),
        "Primary+M"
    ),
    command!(
        "curation.similar_noise",
        "Move similar clusters to noise",
        Recording,
        I(Move(Similar, SetNoise)),
        "Primary+N"
    ),
    command!(
        "curation.similar_unsorted",
        "Move similar clusters to unsorted",
        Recording,
        I(Move(Similar, SetUnsorted)),
        "Primary+U"
    ),
    command!(
        "curation.all_good",
        "Move all selected clusters to good",
        Recording,
        I(Move(All, SetGood)),
        "Primary+Alt+G"
    ),
    command!(
        "curation.all_mua",
        "Move all selected clusters to MUA",
        Recording,
        I(Move(All, SetMua)),
        "Primary+Alt+M"
    ),
    command!(
        "curation.all_noise",
        "Move all selected clusters to noise",
        Recording,
        I(Move(All, SetNoise)),
        "Primary+Alt+N"
    ),
    command!(
        "curation.all_unsorted",
        "Move all selected clusters to unsorted",
        Recording,
        I(Move(All, SetUnsorted)),
        "Primary+Alt+U"
    ),
    command!("curation.undo", "Undo", Recording, I(Undo), "Primary+Z"),
    command!(
        "curation.redo",
        "Redo",
        Recording,
        I(Redo),
        "Primary+Shift+Z",
        "Primary+Y"
    ),
    command!(
        "curation.save",
        "Save curated state",
        Recording,
        I(Save),
        "Primary+S"
    ),
    command!(
        "navigation.next",
        "Next best cluster",
        Recording,
        I(NextCluster),
        "ArrowDown"
    ),
    command!(
        "navigation.previous",
        "Previous best cluster",
        Recording,
        I(PrevCluster),
        "ArrowUp"
    ),
    command!(
        "navigation.first",
        "First best cluster",
        Recording,
        I(FirstCluster),
        "Home"
    ),
    command!(
        "navigation.last",
        "Last best cluster",
        Recording,
        I(LastCluster),
        "End"
    ),
    command!(
        "navigation.next_similar",
        "Next similar cluster",
        Recording,
        I(NextSimilar),
        "Space"
    ),
    command!(
        "navigation.previous_similar",
        "Previous similar cluster",
        Recording,
        I(PrevSimilar),
        "Shift+Space"
    ),
    command!(
        "navigation.first_similar",
        "Select first 15 similar clusters",
        Recording,
        I(SelectFirstSimilar),
        "Primary+Space"
    ),
    command!(
        "navigation.unselect_similar",
        "Unselect similar clusters",
        Recording,
        I(UnselectSimilar),
        "Backspace"
    ),
    command!(
        "navigation.reset",
        "Reset wizard to first cluster",
        Recording,
        I(ResetWizard),
        "Primary+Alt+Space"
    ),
    command!(
        "trace.previous_window",
        "Trace: previous window",
        Recording,
        I(PageBack),
        "Alt+ArrowLeft"
    ),
    command!(
        "trace.next_window",
        "Trace: next window",
        Recording,
        I(PageForward),
        "Alt+ArrowRight"
    ),
    command!(
        "trace.start",
        "Trace: start of recording",
        Recording,
        I(TraceStart),
        "Alt+Home"
    ),
    command!(
        "trace.end",
        "Trace: end of recording",
        Recording,
        I(TraceEnd),
        "Alt+End"
    ),
    command!(
        "trace.narrow",
        "Trace: narrow window",
        Recording,
        I(TraceNarrow),
        "Alt+Plus",
        "Alt+Equals"
    ),
    command!(
        "trace.widen",
        "Trace: widen window",
        Recording,
        I(TraceWiden),
        "Alt+Minus"
    ),
    command!(
        "trace.next_spike",
        "Trace: next selected spike",
        Recording,
        I(NextSpike),
        "Alt+PageDown"
    ),
    command!(
        "trace.previous_spike",
        "Trace: previous selected spike",
        Recording,
        I(PrevSpike),
        "Alt+PageUp"
    ),
    command!(
        "trace.goto",
        "Trace: go to time (:tg seconds)",
        Recording,
        I(Snippet(":tg ")),
        "Alt+T"
    ),
    command!(
        "trace.origin",
        "Trace: toggle channel origin",
        Recording,
        I(TraceOrigin),
        "Alt+O"
    ),
    command!(
        "trace.all_spikes",
        "Trace: toggle all clusters' spikes",
        Recording,
        I(TraceAllSpikes),
        "Alt+S"
    ),
    command!(
        "trace.labels",
        "Trace: toggle channel labels",
        Recording,
        I(TraceLabels),
        "Alt+L"
    ),
    command!(
        "trace.gain_up",
        "Trace: increase gain",
        Recording,
        I(TraceGain(1)),
        "Primary+Alt+ArrowUp"
    ),
    command!(
        "trace.gain_down",
        "Trace: decrease gain",
        Recording,
        I(TraceGain(-1)),
        "Primary+Alt+ArrowDown"
    ),
    command!(
        "features.channels",
        "Features: fix or release channels",
        Recording,
        I(ToggleFeatureChannels),
        "C"
    ),
    command!(
        "amplitudes.next",
        "Amplitudes: next type",
        Recording,
        I(CycleAmplitudes(1)),
        "A"
    ),
    command!(
        "amplitudes.previous",
        "Amplitudes: previous type",
        Recording,
        I(CycleAmplitudes(-1)),
        "Shift+A"
    ),
    command!(
        "views.spike_reorder",
        "Toggle reordered time axes",
        Recording,
        I(ToggleSpikeReorder),
        "Primary+R"
    ),
    command!(
        "app.snippet",
        "Open snippet bar",
        Recording,
        I(Snippet(":")),
        "Colon"
    ),
    command!(
        "app.help",
        "Show keyboard and mouse help",
        Recording,
        I(ToggleHelp),
        "H",
        "Questionmark",
        "F1"
    ),
    command!(
        "app.quit",
        "Quit recording",
        Recording,
        I(Quit),
        "Primary+Q"
    ),
    command!(
        "views.maximize",
        "Maximize or restore pane under pointer",
        Recording,
        I(ToggleMaximize),
        "Primary+Shift+M"
    ),
    command!(
        "app.cancel",
        "Cancel dialog, snippet, help, lasso or brush; restore pane",
        Recording,
        I(Cancel),
        "Escape"
    ),
    command!(
        "app.open_dataset",
        "Open dataset",
        Application,
        S(ShellAction::Open),
        "Primary+O"
    ),
    command!(
        "app.close_dataset",
        "Close active dataset",
        Application,
        S(ShellAction::Close),
        "Primary+W"
    ),
    command!(
        "app.next_dataset",
        "Next dataset tab",
        Application,
        S(ShellAction::Next),
        "Ctrl+Tab"
    ),
    command!(
        "app.previous_dataset",
        "Previous dataset tab",
        Application,
        S(ShellAction::Previous),
        "Ctrl+Shift+Tab"
    ),
    command!(
        "window.fullscreen",
        "Toggle fullscreen for this window",
        Window,
        Action::Fullscreen,
        "F11"
    ),
    command!(
        "app.settings",
        "Open Settings",
        Window,
        Action::Settings,
        "Primary+Comma"
    ),
    command!(
        "app.submit",
        "Confirm dialog or run text command",
        Submit,
        Action::Submit,
        "Enter"
    ),
    command!(
        "waveforms.next",
        "Next waveform mode",
        Waveforms,
        W(WaveformAction::Next),
        "W"
    ),
    command!(
        "waveforms.previous",
        "Previous waveform mode",
        Waveforms,
        W(WaveformAction::Previous),
        "Shift+W"
    ),
    command!(
        "waveforms.mean",
        "Toggle mean waveforms",
        Waveforms,
        W(WaveformAction::Mean),
        "M"
    ),
    command!(
        "waveforms.overlap",
        "Toggle waveform overlap",
        Waveforms,
        W(WaveformAction::Overlap),
        "O"
    ),
    command!(
        "waveforms.labels",
        "Toggle waveform channel labels",
        Waveforms,
        W(WaveformAction::Labels),
        "Primary+L"
    ),
    command!(
        "waveforms.amplitude_up",
        "Increase waveform amplitude",
        Waveforms,
        W(WaveformAction::AmplitudeUp),
        "Primary+ArrowUp"
    ),
    command!(
        "waveforms.amplitude_down",
        "Decrease waveform amplitude",
        Waveforms,
        W(WaveformAction::AmplitudeDown),
        "Primary+ArrowDown"
    ),
    command!(
        "waveforms.box_wider",
        "Widen waveform boxes",
        Waveforms,
        W(WaveformAction::BoxWider),
        "Primary+ArrowRight"
    ),
    command!(
        "waveforms.box_narrower",
        "Narrow waveform boxes",
        Waveforms,
        W(WaveformAction::BoxNarrower),
        "Primary+ArrowLeft"
    ),
    command!(
        "waveforms.spread_up",
        "Spread probe vertically",
        Waveforms,
        W(WaveformAction::SpreadUp),
        "Shift+ArrowUp"
    ),
    command!(
        "waveforms.spread_down",
        "Shrink probe vertically",
        Waveforms,
        W(WaveformAction::SpreadDown),
        "Shift+ArrowDown"
    ),
    command!(
        "waveforms.spread_wider",
        "Spread probe horizontally",
        Waveforms,
        W(WaveformAction::SpreadWider),
        "Shift+ArrowRight"
    ),
    command!(
        "waveforms.spread_narrower",
        "Shrink probe horizontally",
        Waveforms,
        W(WaveformAction::SpreadNarrower),
        "Shift+ArrowLeft"
    ),
    command!(
        "waveforms.reset",
        "Reset waveform scales",
        Waveforms,
        W(WaveformAction::Reset),
        "R",
        "Shift+R"
    ),
    command!(
        "ccg.normalize",
        "Toggle uniform CCG normalization",
        Ccg,
        Action::CcgNormalize,
        "N"
    ),
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chord {
    pub key: Key,
    primary: bool,
    ctrl: bool,
    alt: bool,
    shift: bool,
}

impl Chord {
    pub fn parse(text: &str) -> Result<Self, String> {
        let parts: Vec<_> = text.split('+').map(str::trim).collect();
        let mut chord = Self {
            key: Key::Space,
            primary: false,
            ctrl: false,
            alt: false,
            shift: false,
        };
        for part in &parts[..parts.len().saturating_sub(1)] {
            match part.to_ascii_lowercase().as_str() {
                "primary" | "cmd" | "command" if !chord.primary => chord.primary = true,
                "ctrl" | "control" if !chord.ctrl => chord.ctrl = true,
                "alt" | "option" if !chord.alt => chord.alt = true,
                "shift" if !chord.shift => chord.shift = true,
                _ => return Err(format!("Invalid modifier in {text:?}")),
            }
        }
        let key = parts.last().copied().unwrap_or_default();
        chord.key = Key::ALL
            .iter()
            .copied()
            .find(|k| k.name().eq_ignore_ascii_case(key))
            .or_else(|| match key.to_ascii_lowercase().as_str() {
                "escape" | "esc" => Some(Key::Escape),
                "arrowup" | "up" => Some(Key::ArrowUp),
                "arrowdown" | "down" => Some(Key::ArrowDown),
                "arrowleft" | "left" => Some(Key::ArrowLeft),
                "arrowright" | "right" => Some(Key::ArrowRight),
                "questionmark" | "?" => Some(Key::Questionmark),
                "colon" | ":" => Some(Key::Colon),
                "plus" => Some(Key::Plus),
                "equals" | "=" => Some(Key::Equals),
                "minus" | "-" => Some(Key::Minus),
                "comma" | "," => Some(Key::Comma),
                _ => None,
            })
            .ok_or_else(|| format!("Unknown key in {text:?}; use Plus for the + key"))?;
        if chord.primary && chord.ctrl && !cfg!(target_os = "macos") {
            return Err("Use either Primary or physical Ctrl, not both".into());
        }
        Ok(chord)
    }

    /// Symbol keys already encode Shift on ordinary keyboard layouts.
    fn symbolic(&self) -> bool {
        matches!(
            self.key,
            Key::Colon | Key::Questionmark | Key::Plus | Key::Equals
        )
    }

    pub fn matches(&self, key: Key, m: Modifiers) -> bool {
        let command = if cfg!(target_os = "macos") {
            self.primary
        } else {
            self.primary || self.ctrl
        };
        let ctrl = if cfg!(target_os = "macos") {
            self.ctrl
        } else {
            command
        };
        self.key == key
            && m.command == command
            && m.ctrl == ctrl
            && m.alt == self.alt
            && (self.symbolic() || m.shift == self.shift)
    }

    pub fn capture(key: Key, m: Modifiers) -> Self {
        Self {
            key,
            primary: m.command,
            ctrl: m.ctrl && (cfg!(target_os = "macos") || !m.command),
            alt: m.alt,
            shift: m.shift,
        }
    }

    pub fn text(&self, display: bool) -> String {
        let mut s = String::new();
        if self.primary {
            s.push_str(if display {
                if cfg!(target_os = "macos") {
                    "Cmd+"
                } else {
                    "Ctrl+"
                }
            } else {
                "Primary+"
            });
        }
        if self.ctrl {
            s.push_str("Ctrl+");
        }
        if self.alt {
            s.push_str("Alt+");
        }
        if self.shift && !self.symbolic() {
            s.push_str("Shift+");
        }
        s.push_str(match self.key {
            Key::Plus => "Plus",
            Key::Equals => "Equals",
            Key::Colon => "Colon",
            Key::Questionmark => "Questionmark",
            Key::Comma => "Comma",
            k => k.name(),
        });
        s
    }

    fn overlaps(&self, other: &Self) -> bool {
        if self.key != other.key || self.alt != other.alt {
            return false;
        }
        let modifier = |c: &Self| {
            if cfg!(target_os = "macos") {
                (c.primary, c.ctrl)
            } else {
                (c.primary || c.ctrl, false)
            }
        };
        modifier(self) == modifier(other) && (self.symbolic() || self.shift == other.shift)
    }
}

#[derive(Clone)]
pub struct Keymap {
    bindings: Vec<Vec<Chord>>,
}

impl Default for Keymap {
    fn default() -> Self {
        Self::compile(&BTreeMap::new()).expect("valid shortcut defaults")
    }
}

impl Keymap {
    pub fn compile(overrides: &BTreeMap<String, Vec<String>>) -> Result<Self, String> {
        for id in overrides.keys() {
            if !COMMANDS.iter().any(|c| c.id == id) {
                return Err(format!("Unknown shortcut action {id:?}"));
            }
        }
        let bindings = COMMANDS
            .iter()
            .map(|c| {
                let strings: Vec<&str> = overrides
                    .get(c.id)
                    .map(|v| v.iter().map(String::as_str).collect())
                    .unwrap_or_else(|| c.defaults.to_vec());
                strings
                    .into_iter()
                    .map(Chord::parse)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| format!("{}: {e}", c.id))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let map = Self { bindings };
        for (i, c) in COMMANDS.iter().enumerate() {
            for (a, chord) in map.bindings[i].iter().enumerate() {
                if map.bindings[i][..a].iter().any(|b| b.overlaps(chord)) {
                    return Err(format!("Duplicate shortcut for {}", c.id));
                }
                for (j, other) in COMMANDS.iter().enumerate().take(i) {
                    if c.scope.overlaps(other.scope)
                        && map.bindings[j].iter().any(|b| b.overlaps(chord))
                    {
                        return Err(format!(
                            "{} conflicts with {} ({})",
                            c.id,
                            other.id,
                            chord.text(true)
                        ));
                    }
                }
            }
        }
        Ok(map)
    }

    pub fn chords(&self, id: &str) -> &[Chord] {
        COMMANDS
            .iter()
            .position(|c| c.id == id)
            .map_or(&[], |i| &self.bindings[i])
    }

    pub fn hint(&self, id: &str) -> String {
        let keys = self
            .chords(id)
            .iter()
            .map(|c| c.text(true))
            .collect::<Vec<_>>();
        if keys.is_empty() {
            "unbound".into()
        } else {
            keys.join(" / ")
        }
    }

    pub fn matches(&self, id: &str, key: Key, modifiers: Modifiers) -> bool {
        self.chords(id).iter().any(|c| c.matches(key, modifiers))
    }

    pub fn actions(&self, events: &[Event], scope: Scope, typing: bool) -> Vec<Action> {
        events
            .iter()
            .filter_map(|event| {
                let Event::Key {
                    key,
                    pressed: true,
                    modifiers,
                    ..
                } = event
                else {
                    return None;
                };
                COMMANDS
                    .iter()
                    .enumerate()
                    .find(|(i, c)| {
                        c.scope == scope
                            && self.bindings[*i].iter().any(|b| {
                                b.matches(*key, *modifiers)
                                    && (!typing
                                        || scope == Scope::Submit
                                        || c.action == Action::Intent(Intent::Cancel)
                                        || c.action == Action::Fullscreen
                                        || (!matches!(scope, Scope::Waveforms | Scope::Ccg)
                                            && (b.primary
                                                || b.ctrl
                                                || matches!(b.key, Key::Escape | Key::F1))))
                            })
                    })
                    .map(|(_, c)| c.action)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_ids_are_valid() {
        let map = Keymap::default();
        for (i, c) in COMMANDS.iter().enumerate() {
            assert!(!COMMANDS[..i].iter().any(|o| o.id == c.id));
            for chord in map.chords(c.id) {
                assert_eq!(Chord::parse(&chord.text(false)).unwrap(), *chord);
                let m = if cfg!(target_os = "macos") {
                    Modifiers {
                        command: chord.primary,
                        mac_cmd: chord.primary,
                        ctrl: chord.ctrl,
                        alt: chord.alt,
                        shift: chord.shift,
                    }
                } else {
                    Modifiers {
                        command: chord.primary || chord.ctrl,
                        ctrl: chord.primary || chord.ctrl,
                        alt: chord.alt,
                        shift: chord.shift,
                        ..Modifiers::NONE
                    }
                };
                let event = Event::Key {
                    key: chord.key,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: m,
                };
                assert_eq!(
                    map.actions(&[event], c.scope, false),
                    [c.action],
                    "{}",
                    c.id
                );
            }
        }
        assert!(map.matches("curation.split", Key::K, Modifiers::NONE));
    }

    #[test]
    fn overrides_replace_defaults_and_can_disable() {
        let map = Keymap::compile(&BTreeMap::from([
            ("curation.split".into(), vec!["F8".into()]),
            ("curation.merge".into(), vec![]),
        ]))
        .unwrap();
        assert!(!map.matches("curation.split", Key::K, Modifiers::NONE));
        assert!(map.matches("curation.split", Key::F8, Modifiers::NONE));
        assert_eq!(map.hint("curation.split"), "F8");
        assert_eq!(map.hint("curation.merge"), "unbound");
    }

    #[test]
    fn conflicts_include_panes_shell_and_symbol_shift() {
        for (id, chord) in [
            ("curation.split", "G"),
            ("waveforms.mean", "K"),
            ("curation.split", "Primary+O"),
            ("curation.split", "Shift+Questionmark"),
        ] {
            assert!(Keymap::compile(&BTreeMap::from([(id.into(), vec![chord.into()])])).is_err());
        }
        assert!(Keymap::compile(&BTreeMap::from([(
            "waveforms.mean".into(),
            vec!["N".into()]
        )]))
        .is_ok());
        assert!(Chord::parse("Hyper+K").is_err());
        assert!(Keymap::compile(&BTreeMap::from([("typo".into(), vec![])])).is_err());
    }

    #[test]
    fn typing_suppresses_plain_and_pane_keys_but_keeps_global_chords() {
        let event = |key, modifiers| Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers,
        };
        let map = Keymap::default();
        assert!(map
            .actions(&[event(Key::G, Modifiers::NONE)], Scope::Recording, true)
            .is_empty());
        assert!(map
            .actions(&[event(Key::M, Modifiers::NONE)], Scope::Waveforms, true)
            .is_empty());
        assert_eq!(
            map.actions(&[event(Key::F11, Modifiers::NONE)], Scope::Window, true),
            [Action::Fullscreen]
        );
        let primary = if cfg!(target_os = "macos") {
            Modifiers {
                command: true,
                mac_cmd: true,
                ..Modifiers::NONE
            }
        } else {
            Modifiers {
                command: true,
                ctrl: true,
                ..Modifiers::NONE
            }
        };
        assert_eq!(
            map.actions(&[event(Key::S, primary)], Scope::Recording, true),
            [Action::Intent(Intent::Save)]
        );
    }
}
