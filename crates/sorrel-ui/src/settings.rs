//! User-wide settings. Recording workspace state stays in `prefs.rs`.

use crate::shortcuts::{Action, Chord, Keymap, Scope, COMMANDS};
use egui::{Context, Event};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub version: u32,
    pub appearance: Appearance,
    pub recent_datasets: Vec<PathBuf>,
    pub keybindings: BTreeMap<String, Vec<String>>,
    /// Retain unrelated settings when this build edits shortcuts.
    #[serde(flatten)]
    extra: BTreeMap<String, toml::Value>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            version: 1,
            appearance: Appearance::default(),
            recent_datasets: vec![],
            keybindings: BTreeMap::new(),
            extra: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Appearance {
    /// None leaves eframe's existing theme preference untouched.
    pub theme: Option<String>,
    #[serde(flatten)]
    extra: BTreeMap<String, toml::Value>,
}

pub fn path() -> Result<PathBuf, String> {
    let var = |name| {
        std::env::var_os(name)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    // Separate from the runtime registry: XDG_RUNTIME_DIR may be ephemeral.
    if let Some(dir) = var("SORREL_CONFIG_DIR") {
        return Ok(dir.join("settings.toml"));
    }
    let home = if cfg!(windows) {
        var("USERPROFILE").or_else(|| var("HOME"))
    } else {
        var("HOME")
    };
    home.map(|p| p.join(".sorrel").join("settings.toml"))
        .ok_or_else(|| "Cannot locate the user home; set SORREL_CONFIG_DIR".into())
}

impl Settings {
    pub fn validate(&self) -> Result<Keymap, String> {
        if self.version != 1 {
            return Err(format!("Unsupported settings version {}", self.version));
        }
        if self
            .appearance
            .theme
            .as_deref()
            .is_some_and(|t| !["system", "dark", "light"].contains(&t))
        {
            return Err("appearance.theme must be system, dark or light".into());
        }
        Keymap::compile(&self.keybindings)
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                let settings: Self = toml::from_str(&text).map_err(|e| e.to_string())?;
                settings.validate()?;
                Ok(settings)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e.to_string()),
        }
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        self.validate()?;
        let text = toml::to_string_pretty(self).map_err(|e| e.to_string())?;
        let parent = path.parent().ok_or("Settings path has no parent")?;
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        // Unique temp files avoid competing writers sharing a staging file.
        let mut tmp = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
        use std::io::Write;
        tmp.write_all(text.as_bytes()).map_err(|e| e.to_string())?;
        tmp.as_file().sync_all().map_err(|e| e.to_string())?;
        sorrel_io::fsutil::replace_file_io(tmp.path(), path).map_err(|e| e.to_string())
    }

    fn apply_appearance(&self, ctx: &Context) {
        let preference = match self.appearance.theme.as_deref() {
            Some("dark") => egui::ThemePreference::Dark,
            Some("light") => egui::ThemePreference::Light,
            Some("system") => egui::ThemePreference::System,
            _ => return,
        };
        ctx.set_theme(preference);
    }
}

/// One shared instance per egui context, including all tabs and viewports.
/// Embedders/tests default to an in-memory instance: no incidental home I/O.
pub struct Runtime {
    pub settings: Settings,
    pub keymap: Keymap,
    pub open: bool,
    pub modal: bool,
    pub shell_owned: bool,
    pub warning: Option<String>,
    path: Option<PathBuf>,
    writable: bool,
    draft: Option<Settings>,
    search: String,
    capture: Option<&'static str>,
    error: Option<String>,
    window_requests: Vec<Action>,
}

impl Default for Runtime {
    fn default() -> Self {
        Self {
            settings: Settings::default(),
            keymap: Keymap::default(),
            open: false,
            modal: false,
            shell_owned: false,
            warning: None,
            path: None,
            writable: true,
            draft: None,
            search: String::new(),
            capture: None,
            error: None,
            window_requests: vec![],
        }
    }
}

pub type Shared = Arc<Mutex<Runtime>>;

pub fn shared(ctx: &Context) -> Shared {
    ctx.data_mut(|d| {
        d.get_temp_mut_or_default::<Shared>(egui::Id::new("sorrel-settings"))
            .clone()
    })
}

pub fn install(ctx: &Context, path: Result<PathBuf, String>, legacy_recents: Vec<PathBuf>) {
    let mut rt = Runtime::default();
    match path {
        Ok(path) => {
            let exists = path.exists();
            match Settings::load(&path) {
                Ok(mut settings) => {
                    if !exists {
                        settings.recent_datasets = legacy_recents;
                    }
                    rt.keymap = settings.validate().expect("validated on load");
                    settings.apply_appearance(ctx);
                    rt.settings = settings;
                }
                Err(e) => {
                    rt.warning = Some(format!(
                        "Cannot load {}: {e}. Using defaults; the file will not be overwritten.",
                        path.display()
                    ));
                    rt.writable = false;
                }
            }
            rt.path = Some(path);
        }
        Err(e) => {
            rt.warning = Some(e);
            rt.writable = false;
        }
    }
    *shared(ctx).lock().unwrap() = rt;
}

pub fn hint(ctx: &Context, id: &str) -> String {
    shared(ctx).lock().unwrap().keymap.hint(id)
}

pub fn split_hint(ctx: &Context) -> String {
    format!(
        "{} splits · {} clears",
        hint(ctx, "curation.split"),
        hint(ctx, "app.cancel")
    )
}

pub fn blocked(ctx: &Context) -> bool {
    let shared = shared(ctx);
    let rt = shared.lock().unwrap();
    rt.open || rt.modal
}

pub fn actions(ctx: &Context, scope: Scope) -> Vec<Action> {
    let shared = shared(ctx);
    let rt = shared.lock().unwrap();
    if rt.open || (rt.modal && scope != Scope::Submit && scope != Scope::Recording) {
        return vec![];
    }
    let typing = ctx.egui_wants_keyboard_input();
    let mut actions = ctx.input(|i| rt.keymap.actions(&i.events, scope, typing));
    if rt.modal && scope == Scope::Recording {
        actions.retain(|a| *a == Action::Intent(crate::intent::Intent::Cancel));
    }
    actions
}

pub fn pressed(ctx: &Context, id: &str) -> bool {
    let shared = shared(ctx);
    let rt = shared.lock().unwrap();
    !rt.open
        && ctx.input(|i| {
            i.events.iter().any(|e| {
                matches!(e,
        Event::Key { key, pressed: true, modifiers, .. } if rt.keymap.matches(id, *key, *modifiers))
            })
        })
}

pub fn consume(ctx: &Context, scope: Scope) -> Vec<Action> {
    let actions = actions(ctx, scope);
    if actions.is_empty() {
        return actions;
    }
    let map = shared(ctx).lock().unwrap().keymap.clone();
    ctx.input_mut(|i| {
        i.events.retain(|e| {
            map.actions(std::slice::from_ref(e), scope, false)
                .is_empty()
        })
    });
    actions
}

pub fn open(ctx: &Context) {
    let shared = shared(ctx);
    let mut rt = shared.lock().unwrap();
    if !rt.open {
        rt.draft = Some(rt.settings.clone());
        rt.open = true;
        rt.error = None;
    }
}

pub fn request_window(ctx: &Context, action: Action) {
    shared(ctx).lock().unwrap().window_requests.push(action);
    ctx.send_viewport_cmd_to(
        egui::ViewportId::ROOT,
        egui::ViewportCommand::Minimized(false),
    );
    ctx.send_viewport_cmd_to(egui::ViewportId::ROOT, egui::ViewportCommand::Visible(true));
    ctx.send_viewport_cmd_to(egui::ViewportId::ROOT, egui::ViewportCommand::Focus);
    ctx.request_repaint_of(egui::ViewportId::ROOT);
}

pub fn take_window_requests(ctx: &Context) -> Vec<Action> {
    std::mem::take(&mut shared(ctx).lock().unwrap().window_requests)
}

pub fn recent(ctx: &Context) -> Vec<PathBuf> {
    shared(ctx).lock().unwrap().settings.recent_datasets.clone()
}

pub fn remember(ctx: &Context, path: PathBuf) {
    let shared = shared(ctx);
    let mut rt = shared.lock().unwrap();
    rt.settings.recent_datasets.retain(|p| *p != path);
    rt.settings.recent_datasets.insert(0, path);
    rt.settings.recent_datasets.truncate(10);
    if rt.writable {
        if let Some(path) = &rt.path {
            if let Err(e) = rt.settings.save(path) {
                rt.warning = Some(format!("Could not save settings: {e}"));
            }
        }
    }
}

/// Call before dispatch: while recording a shortcut, no key or text event
/// reaches a text widget, shell command, pane action or curation operation.
pub fn capture_input(ctx: &Context) {
    let shared = shared(ctx);
    let mut rt = shared.lock().unwrap();
    if !rt.open {
        return;
    }
    if rt.capture.is_none()
        && !egui::Popup::is_any_open(ctx)
        && ctx.input(|i| {
            i.events.iter().any(|e| {
                matches!(e,
            Event::Key { key, pressed: true, modifiers, .. }
                if rt.keymap.matches("app.cancel", *key, *modifiers))
            })
        })
    {
        rt.open = false;
        rt.draft = None;
        consume_keyboard(ctx);
        ctx.request_repaint();
        return;
    }
    if let Some(id) = rt.capture {
        let chord = ctx.input(|i| {
            i.events.iter().find_map(|e| match e {
                Event::Key {
                    key,
                    pressed: true,
                    repeat: false,
                    modifiers,
                    ..
                } => Some(Chord::capture(*key, *modifiers)),
                _ => None,
            })
        });
        if let Some(chord) = chord {
            // A missing draft entry means defaults (e.g. after Reset), not the saved keymap.
            let current = COMMANDS
                .iter()
                .find(|c| c.id == id)
                .map_or_else(Vec::new, |c| {
                    c.defaults.iter().map(|s| (*s).into()).collect()
                });
            let keys = rt
                .draft
                .as_mut()
                .unwrap()
                .keybindings
                .entry(id.into())
                .or_insert(current);
            keys.push(chord.text(false));
            rt.capture = None;
        }
        consume_keyboard(ctx);
    }
}

fn consume_keyboard(ctx: &Context) {
    ctx.input_mut(|i| {
        i.events.retain(|e| {
            !matches!(
                e,
                Event::Key { .. } | Event::Text(_) | Event::Copy | Event::Cut | Event::Paste(_)
            )
        })
    });
}

pub fn show(ctx: &Context) {
    let shared = shared(ctx);
    let mut rt = shared.lock().unwrap();
    if !rt.open {
        return;
    }
    let mut close = false;
    egui::Modal::new(egui::Id::new("sorrel-settings-dialog")).show(ctx, |ui| {
        ui.set_width(780.0_f32.min(ctx.content_rect().width() - 40.0));
        ui.heading("Settings");
        if let Some(warning) = &rt.warning { ui.colored_label(ui.visuals().warn_fg_color, warning); }
        let location = rt.path.clone().or_else(|| path().ok());
        ui.monospace(rt.path.as_ref().map_or_else(|| "~/.sorrel/settings.toml".into(), |p| p.display().to_string()));
        ui.horizontal(|ui| {
            ui.label("Theme");
            let theme = &mut rt.draft.as_mut().unwrap().appearance.theme;
            egui::ComboBox::from_id_salt("settings-theme").selected_text(theme.as_deref().unwrap_or("unchanged"))
                .show_ui(ui, |ui| {
                    for name in ["system", "dark", "light"] { ui.selectable_value(theme, Some(name.into()), name); }
                });
        });
        ui.separator();
        ui.heading("Keyboard shortcuts");
        ui.label("Defaults follow phy. Primary means Ctrl on Windows/Linux and Cmd on macOS. Mouse gestures are unchanged.");
        let label = ui.label("Search shortcuts");
        ui.add(egui::TextEdit::singleline(&mut rt.search).desired_width(ui.available_width()).hint_text("Search actions, IDs or scopes")).labelled_by(label.id);
        if let Some(id) = rt.capture {
            ui.horizontal(|ui| {
                ui.strong(format!("Press a shortcut for {id}…"));
                if ui.button("Cancel recording").clicked() { rt.capture = None; }
            });
        }
        let search = rt.search.to_ascii_lowercase();
        let mut commands: Vec<_> = COMMANDS.iter().collect();
        commands.sort_by_key(|c| (c.scope as u8, c.id));
        let mut previous_scope = None;
        egui::ScrollArea::vertical().max_height((ctx.content_rect().height() * 0.48).max(100.0))
            .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysVisible).show(ui, |ui| {
            for command in commands {
                if !search.is_empty() && !format!("{} {} {}", command.id, command.description, command.scope.name()).to_ascii_lowercase().contains(&search) { continue; }
                if previous_scope != Some(command.scope) {
                    ui.strong(command.scope.name());
                    previous_scope = Some(command.scope);
                }
                ui.push_id(command.id, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(command.description).on_hover_text(format!("{} · {}", command.id, command.scope.name()));
                        let strings = rt.draft.as_ref().unwrap().keybindings.get(command.id).cloned()
                            .unwrap_or_else(|| command.defaults.iter().map(|s| (*s).into()).collect());
                        for (i, text) in strings.iter().enumerate() {
                            let display = Chord::parse(text).map_or_else(|_| text.clone(), |c| c.text(true));
                            if ui.small_button(format!("{display} ×")).on_hover_text("Remove binding").clicked() {
                                let mut keys = strings.clone(); keys.remove(i);
                                rt.draft.as_mut().unwrap().keybindings.insert(command.id.into(), keys);
                            }
                        }
                        if strings.is_empty() { ui.weak("unbound"); }
                        if ui.small_button("Add shortcut").clicked() { rt.capture = Some(command.id); }
                        if ui.small_button("Reset").clicked() { rt.draft.as_mut().unwrap().keybindings.remove(command.id); }
                    });
                });
            }
        });
        let validation = rt.draft.as_ref().unwrap().validate();
        if let Err(e) = &validation { ui.colored_label(ui.visuals().error_fg_color, e); }
        if let Some(e) = &rt.error { ui.colored_label(ui.visuals().error_fg_color, e); }
        ui.horizontal(|ui| {
            if ui.button("Restore all shortcut defaults").clicked() { rt.draft.as_mut().unwrap().keybindings.clear(); rt.capture = None; }
            if ui.add_enabled(validation.is_ok() && rt.writable && rt.capture.is_none(), egui::Button::new("Save settings")).clicked() {
                let mut draft = rt.draft.clone().unwrap();
                draft.recent_datasets = rt.settings.recent_datasets.clone();
                match location.ok_or_else(|| "No settings path".into()).and_then(|p| draft.save(&p).map(|()| p)) {
                    Ok(path) => {
                        rt.path = Some(path); rt.keymap = draft.validate().unwrap();
                        draft.apply_appearance(ctx); rt.settings = draft; rt.warning = None; close = true;
                    }
                    Err(e) => rt.error = Some(e),
                }
            }
            if ui.button("Cancel").clicked() { close = true; }
        });
    });
    if close {
        rt.open = false;
        rt.capture = None;
        rt.draft = None;
        ctx.request_repaint();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_unknown_settings_survive() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        let mut settings: Settings = toml::from_str("version = 1\n[other]\nvalue = 42\n").unwrap();
        settings
            .keybindings
            .insert("curation.split".into(), vec!["F8".into()]);
        settings.save(&path).unwrap();
        let back = Settings::load(&path).unwrap();
        assert_eq!(back.extra["other"]["value"].as_integer(), Some(42));
        assert_eq!(back.validate().unwrap().hint("curation.split"), "F8");
    }

    #[test]
    fn malformed_file_is_not_overwritten_by_recent_updates() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        std::fs::write(&path, "[bad toml").unwrap();
        let ctx = Context::default();
        install(&ctx, Ok(path.clone()), vec![]);
        remember(&ctx, PathBuf::from("recording"));
        assert!(shared(&ctx).lock().unwrap().warning.is_some());
        assert_eq!(std::fs::read_to_string(path).unwrap(), "[bad toml");
    }

    fn key(key: egui::Key) -> Event {
        Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }
    }

    #[test]
    fn configured_dispatch_help_and_capture_use_the_same_map() {
        let ctx = Context::default();
        let shared = shared(&ctx);
        {
            let mut rt = shared.lock().unwrap();
            rt.settings
                .keybindings
                .insert("curation.split".into(), vec!["F8".into()]);
            rt.keymap = rt.settings.validate().unwrap();
        }
        let _ = ctx.run_ui(
            egui::RawInput {
                events: vec![key(egui::Key::K), key(egui::Key::F8)],
                ..Default::default()
            },
            |ui| {
                let ctx = ui.ctx();
                assert_eq!(
                    crate::layout::configured_key_intents(ctx),
                    [crate::intent::Intent::Split]
                );
                assert_eq!(hint(ctx, "curation.split"), "F8");
                open(ctx);
                shared.lock().unwrap().capture = Some("curation.split");
                capture_input(ctx);
                assert!(crate::layout::configured_key_intents(ctx).is_empty());
                assert!(ctx.input(|i| i.events.is_empty()));
            },
        );
        assert!(
            !shared.lock().unwrap().draft.as_ref().unwrap().keybindings["curation.split"]
                .is_empty()
        );
    }

    #[test]
    fn modal_blocks_shell_panes_and_curation_but_keeps_cancel_and_submit() {
        let ctx = Context::default();
        shared(&ctx).lock().unwrap().modal = true;
        let _ = ctx.run_ui(
            egui::RawInput {
                events: vec![
                    key(egui::Key::G),
                    key(egui::Key::N),
                    key(egui::Key::Escape),
                    key(egui::Key::Enter),
                ],
                ..Default::default()
            },
            |ui| {
                let ctx = ui.ctx();
                assert_eq!(
                    crate::layout::configured_key_intents(ctx),
                    [crate::intent::Intent::Cancel]
                );
                assert!(actions(ctx, Scope::Ccg).is_empty());
                assert!(actions(ctx, Scope::Application).is_empty());
                assert!(pressed(ctx, "app.submit"));
            },
        );
    }

    #[test]
    fn missing_file_migrates_recents_and_settings_have_one_owner() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        let ctx = Context::default();
        install(&ctx, Ok(path.clone()), vec![PathBuf::from("old")]);
        assert!(!path.exists(), "loading must not create files");
        remember(&ctx, PathBuf::from("new"));
        install(&ctx, Ok(path), vec![PathBuf::from("stale")]);
        assert_eq!(recent(&ctx), [PathBuf::from("new"), PathBuf::from("old")]);
    }

    #[test]
    fn invalid_save_preserves_previous_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        let mut settings = Settings::default();
        settings.save(&path).unwrap();
        let before = std::fs::read(&path).unwrap();
        settings
            .keybindings
            .insert("curation.split".into(), vec!["G".into()]);
        assert!(settings.save(&path).is_err());
        assert_eq!(std::fs::read(path).unwrap(), before);
    }

    #[test]
    fn gui_can_rebind_save_and_reset_without_touching_home() {
        use egui_kittest::kittest::Queryable;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        let mut h = egui_kittest::Harness::new_ui(|ui| {
            capture_input(ui.ctx());
            show(ui.ctx());
        });
        install(&h.ctx, Ok(path.clone()), vec![]);
        open(&h.ctx);
        h.run_ok();
        h.get_by_label("Search shortcuts").focus();
        h.run_ok();
        h.get_by_label("Search shortcuts")
            .type_text("inside the lasso");
        h.run_ok();
        h.get_by_label("K ×").click();
        h.run_ok();
        h.get_by_label("Add shortcut").click();
        h.run_ok();
        h.key_press(egui::Key::F8);
        h.run_ok();
        h.get_by_label("Save settings").click();
        h.run_ok();
        assert!(!shared(&h.ctx).lock().unwrap().open);
        assert_eq!(hint(&h.ctx, "curation.split"), "F8");
        assert_eq!(
            Settings::load(&path)
                .unwrap()
                .validate()
                .unwrap()
                .hint("curation.split"),
            "F8"
        );
        open(&h.ctx);
        h.run_ok();
        h.get_by_label("Reset").click();
        h.run_ok();
        h.get_by_label("Save settings").click();
        h.run_ok();
        assert_eq!(hint(&h.ctx, "curation.split"), "K");
        assert!(!Settings::load(&path)
            .unwrap()
            .keybindings
            .contains_key("curation.split"));
        open(&h.ctx);
        h.run_ok();
        h.get_by_label("K ×").click();
        h.run_ok();
        h.get_by_label("Add shortcut").click();
        h.run_ok();
        h.key_press(egui::Key::F8);
        h.run_ok();
        h.get_by_label("Save settings").click();
        h.run_ok();
        open(&h.ctx);
        h.run_ok();
        h.get_by_label("Reset").click();
        h.run_ok();
        h.get_by_label("Add shortcut").click();
        h.run_ok();
        h.key_press(egui::Key::F9);
        h.run_ok();
        assert_eq!(
            shared(&h.ctx)
                .lock()
                .unwrap()
                .draft
                .as_ref()
                .unwrap()
                .keybindings["curation.split"],
            vec!["K".to_string(), "F9".to_string()],
            "Add after Reset starts from the defaults"
        );
        h.key_press(egui::Key::Escape);
        h.run_ok();
        open(&h.ctx);
        h.run_ok();
        h.key_press(egui::Key::Escape);
        h.run_ok();
        assert!(
            !shared(&h.ctx).lock().unwrap().open,
            "configured cancel closes Settings"
        );
        open(&h.ctx);
        h.run_ok();
        h.get_by_label("Add shortcut").click();
        h.run_ok();
        h.key_press(egui::Key::Escape);
        h.run_ok();
        assert!(
            shared(&h.ctx).lock().unwrap().open,
            "cancel is recordable during capture"
        );
        assert!(shared(&h.ctx)
            .lock()
            .unwrap()
            .draft
            .as_ref()
            .unwrap()
            .keybindings["curation.split"]
            .contains(&"Escape".into()));
    }
}
