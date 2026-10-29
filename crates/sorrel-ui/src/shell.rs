//! The window: several curation sessions as tabs, or a launcher when none is
//! open. Each tab is a [`Tab`] (a `SorrelApp`); the shell owns everything
//! window-level: close handling, the OS title, GPU pipeline install and
//! requests from `sorrel open` ([`sorrel_data::InstanceRequest`]).

use crate::{RasterPipeline, ScatterPipeline, TracePipeline};
use eframe::egui;
use sorrel_data::{InstanceRequest, Pending};
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

const RECENT_KEY: &str = "recent_datasets";
const RECENT_MAX: usize = 10;

/// Where a tab is in closing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloseState {
    Open,
    /// A "save before closing?" dialog is up.
    Asking,
    Closed,
}

/// One session the shell hosts.
pub trait Tab {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame);
    /// Dataset directory name, for the tab bar.
    fn label(&self) -> String;
    /// The directory curation is saved to; identifies the tab.
    fn dataset(&self) -> &Path;
    fn dirty(&self) -> bool;
    fn window_title(&self) -> String;
    /// Close if clean, else ask: the answer shows in [`Self::close_state`].
    fn request_close(&mut self);
    fn close_state(&self) -> CloseState;
    fn on_exit(&mut self);
}

/// What to open: `sorrel open`'s arguments, relative to `cwd`.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenRequest {
    pub cwd: PathBuf,
    pub args: Vec<String>,
}

pub enum Opened {
    /// A new tab, and the absolute DATA_DIR it was opened from (an `.nwb`
    /// file, not its save directory) for the recent list.
    Tab(Box<dyn Tab>, PathBuf),
    /// The dataset is already open as this tab.
    Existing(usize),
}

/// Builds a tab for a request. Gets the open tabs' datasets so it can answer
/// [`Opened::Existing`] instead of opening a dataset twice.
pub type Opener = Box<
    dyn FnMut(
        &OpenRequest,
        &[PathBuf],
        Option<&eframe::egui_wgpu::RenderState>,
    ) -> anyhow::Result<Opened>,
>;

pub struct Shell {
    tabs: Vec<Box<dyn Tab>>,
    active: usize,
    opener: Opener,
    rs: Option<eframe::egui_wgpu::RenderState>,
    instance: Option<Receiver<Pending>>,
    /// Newest first, at most [`RECENT_MAX`].
    recents: Vec<PathBuf>,
    path_input: String,
    error: Option<String>,
    /// The window asked to close and the tabs are being asked in turn.
    quitting: bool,
    last_title: String,
}

impl Shell {
    /// Installs the GPU pipelines (once, for every tab) and loads the recent
    /// datasets from `cc.storage`.
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        opener: Opener,
        instance: Option<Receiver<Pending>>,
    ) -> Self {
        crate::theme::install_visuals(&cc.egui_ctx);
        crate::theme::install_style(&cc.egui_ctx);
        if let Some(rs) = &cc.wgpu_render_state {
            TracePipeline::install(rs);
            RasterPipeline::install(rs);
            ScatterPipeline::install(rs);
        }
        let recents = cc
            .storage
            .and_then(|s| eframe::get_value(s, RECENT_KEY))
            .unwrap_or_default();
        Self {
            tabs: Vec::new(),
            active: 0,
            opener,
            rs: cc.wgpu_render_state.clone(),
            instance,
            recents,
            path_input: String::new(),
            error: None,
            quitting: false,
            last_title: String::new(),
        }
    }

    /// Open `req` as a tab (or activate the tab already showing it). A
    /// failure is shown in the shell; the current view stays.
    pub fn open(&mut self, req: OpenRequest) {
        if let Err(e) = self.try_open(req) {
            self.error = Some(format!("{e:#}"));
        }
    }

    fn try_open(&mut self, req: OpenRequest) -> anyhow::Result<()> {
        let open: Vec<PathBuf> = self.tabs.iter().map(|t| t.dataset().into()).collect();
        let rs = self.rs.clone();
        let i = match (self.opener)(&req, &open, rs.as_ref())? {
            Opened::Existing(i) if i < self.tabs.len() => i,
            Opened::Existing(i) => anyhow::bail!("no tab {i}"),
            Opened::Tab(tab, root) => {
                self.tabs.push(tab);
                self.recents.retain(|p| *p != root);
                self.recents.insert(0, root);
                self.recents.truncate(RECENT_MAX);
                self.tabs.len() - 1
            }
        };
        self.active = i;
        self.error = None;
        Ok(())
    }

    fn open_path(&mut self, path: &Path) {
        self.open(OpenRequest {
            cwd: std::env::current_dir().unwrap_or_default(),
            args: vec![path.to_string_lossy().into_owned()],
        });
    }

    fn pick(&mut self) {
        // ponytail: blocks the UI thread while the dialog is up (and `sorrel
        // open` times out meanwhile); move to a thread + channel if that bites.
        if let Some(dir) = rfd::FileDialog::new().pick_folder() {
            self.open_path(&dir);
        }
    }

    fn drain_instance(&mut self, ctx: &egui::Context) {
        while let Some(Ok((req, reply))) = self.instance.as_ref().map(|r| r.try_recv()) {
            // Open and Focus raise the window: the user asked for it.
            let mut raise = true;
            let answer = match req {
                InstanceRequest::Open { cwd, args } => {
                    match self.try_open(OpenRequest { cwd, args }) {
                        Ok(()) => serde_json::json!({"ok": true}),
                        Err(e) => {
                            self.error = Some(format!("{e:#}"));
                            serde_json::json!({"error": format!("{e:#}")})
                        }
                    }
                }
                InstanceRequest::Focus => serde_json::json!({"ok": true}),
                InstanceRequest::List => {
                    raise = false;
                    serde_json::json!({
                        "pid": std::process::id(),
                        "tabs": self.tabs.iter().enumerate().map(|(i, t)| serde_json::json!({
                            "dataset": t.dataset().display().to_string(),
                            "label": t.label(),
                            "dirty": t.dirty(),
                            "active": i == self.active,
                        })).collect::<Vec<_>>(),
                    })
                }
            };
            if raise {
                ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(false));
                ctx.send_viewport_cmd(egui::ViewportCommand::Visible(true));
                ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
            }
            let _ = reply.send(answer);
        }
    }

    /// Ctrl-O, Ctrl-W and Ctrl-(Shift-)Tab. None is in the app keymap, and
    /// consuming them here keeps the active tab from seeing them.
    fn shortcuts(&mut self, ctx: &egui::Context) {
        use egui::{Key, Modifiers};
        let (open, close, next, prev) = ctx.input_mut(|i| {
            (
                i.consume_key(Modifiers::COMMAND, Key::O),
                i.consume_key(Modifiers::COMMAND, Key::W),
                i.consume_key(Modifiers::CTRL, Key::Tab),
                i.consume_key(Modifiers::CTRL | Modifiers::SHIFT, Key::Tab),
            )
        });
        let n = self.tabs.len();
        if open {
            self.pick();
        }
        if close && n > 0 {
            self.tabs[self.active].request_close();
        }
        if n > 1 && (next || prev) {
            self.active = if next {
                (self.active + 1) % n
            } else {
                (self.active + n - 1) % n
            };
        }
    }

    fn tab_bar(&mut self, ui: &mut egui::Ui) {
        let (mut switch, mut close, mut add) = (None, None, false);
        egui::ScrollArea::horizontal().show(ui, |ui| {
            ui.horizontal(|ui| {
                for (i, t) in self.tabs.iter().enumerate() {
                    let text = if t.dirty() {
                        format!("{} •", t.label())
                    } else {
                        t.label()
                    };
                    let resp = ui.selectable_label(i == self.active, text);
                    if resp
                        .on_hover_text(t.dataset().display().to_string())
                        .clicked()
                    {
                        switch = Some(i);
                    }
                    if ui
                        .small_button("×")
                        .on_hover_text("Close (Ctrl+W)")
                        .clicked()
                    {
                        close = Some(i);
                    }
                    ui.add_space(6.0);
                }
                add = ui
                    .small_button("+")
                    .on_hover_text("Open dataset (Ctrl+O)")
                    .clicked();
            });
        });
        if let Some(i) = switch {
            self.active = i;
        }
        if let Some(i) = close {
            self.tabs[i].request_close();
        }
        if add {
            self.pick();
        }
    }

    fn launcher(&mut self, ui: &mut egui::Ui) {
        let mut open: Option<PathBuf> = None;
        ui.vertical_centered(|ui| {
            ui.set_max_width(520.0);
            ui.add_space(ui.available_height() * 0.2);
            ui.heading(egui::RichText::new("Sorrel").strong().size(28.0));
            ui.label("Spike-sorting curation");
            ui.add_space(14.0);
            if ui
                .add(egui::Button::new("Open dataset…").min_size(egui::vec2(180.0, 30.0)))
                .on_hover_text("Ctrl+O")
                .clicked()
            {
                self.pick();
            }
            ui.add_space(8.0);
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.path_input)
                    .hint_text("or paste a dataset path and press Enter")
                    .desired_width(f32::INFINITY),
            );
            if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                let p = std::mem::take(&mut self.path_input);
                if !p.trim().is_empty() {
                    open = Some(p.trim().into());
                }
            }
            ui.weak("…or drop a dataset folder on this window");
            if !self.recents.is_empty() {
                ui.add_space(16.0);
                ui.label(egui::RichText::new("Recent").strong());
                for p in &self.recents {
                    let missing = !p.exists();
                    let name = p.file_name().map_or_else(
                        || p.display().to_string(),
                        |n| n.to_string_lossy().into_owned(),
                    );
                    let text = if missing {
                        format!("{name} (missing)")
                    } else {
                        name
                    };
                    let resp = ui
                        .add_enabled(!missing, egui::Button::new(text).frame(false))
                        .on_hover_text(p.display().to_string());
                    if resp.clicked() {
                        open = Some(p.clone());
                    }
                }
            }
        });
        if let Some(p) = open {
            self.open_path(&p);
        }
    }

    /// A dropped `.nwb` / `.mat` opens itself; any other file opens its folder.
    fn drops(&mut self, ctx: &egui::Context) {
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        for path in dropped.into_iter().filter_map(|f| f.path) {
            let leaf = path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("nwb") || e.eq_ignore_ascii_case("mat"));
            let target = if path.is_file() && !leaf {
                path.parent().map(Path::to_path_buf)
            } else {
                Some(path)
            };
            if let Some(p) = target {
                self.open_path(&p);
            }
        }
    }
}

/// Per-frame close bookkeeping: drop `Closed` tabs (after `on_exit`), keep
/// `active` on the same tab, and settle a pending window close. While
/// `quitting`, the first tab still `Asking` is brought to the front; returns
/// `true` once none is left and the window may close, and clears `quitting`
/// when the user cancelled every dialog.
fn reap(tabs: &mut Vec<Box<dyn Tab>>, active: &mut usize, quitting: &mut bool) -> bool {
    let mut i = 0;
    while i < tabs.len() {
        if tabs[i].close_state() == CloseState::Closed {
            tabs.remove(i).on_exit();
            if i < *active {
                *active -= 1;
            }
        } else {
            i += 1;
        }
    }
    *active = (*active).min(tabs.len().saturating_sub(1));
    if !*quitting {
        return false;
    }
    if tabs.is_empty() {
        return true;
    }
    match tabs
        .iter()
        .position(|t| t.close_state() == CloseState::Asking)
    {
        Some(i) => *active = i,
        None => *quitting = false,
    }
    false
}

impl eframe::App for Shell {
    // Also runs while the window is hidden, so `sorrel open` still lands.
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_instance(ctx);
    }

    fn ui(&mut self, root_ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = root_ui.ctx().clone();
        self.shortcuts(&ctx);
        self.drops(&ctx);

        if ctx.input(|i| i.viewport().close_requested()) && !self.tabs.is_empty() {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.quitting = true;
            for t in &mut self.tabs {
                t.request_close();
            }
        }
        if reap(&mut self.tabs, &mut self.active, &mut self.quitting) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }

        let title = self
            .tabs
            .get(self.active)
            .map_or_else(|| "Sorrel".into(), |t| t.window_title());
        if title != self.last_title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.last_title = title;
        }

        if !self.tabs.is_empty() {
            let bar = egui::Frame::side_top_panel(root_ui.style())
                .inner_margin(egui::Margin::symmetric(8, 4));
            egui::Panel::top("shell-tabs")
                .frame(bar)
                .show_inside(root_ui, |ui| self.tab_bar(ui));
        }
        if let Some(err) = self.error.clone() {
            egui::Panel::bottom("shell-error").show_inside(root_ui, |ui| {
                ui.horizontal(|ui| {
                    ui.colored_label(ui.visuals().error_fg_color, err);
                    if ui.small_button("×").clicked() {
                        self.error = None;
                    }
                });
            });
        }
        // ponytail: inactive tabs are not drawn, so agent toasts/highlights
        // show on activation; poll inactive tabs if users miss agent activity.
        match self.tabs.get_mut(self.active) {
            Some(tab) => {
                let id = tab.dataset().to_path_buf();
                egui::CentralPanel::no_frame().show_inside(root_ui, |ui| {
                    ui.push_id(id, |ui| tab.ui(ui, frame));
                });
            }
            None => {
                egui::CentralPanel::default().show_inside(root_ui, |ui| self.launcher(ui));
            }
        }
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, RECENT_KEY, &self.recents);
    }

    fn on_exit(&mut self) {
        for t in &mut self.tabs {
            t.on_exit();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    struct Fake {
        state: Rc<Cell<CloseState>>,
        dirty: bool,
        exited: Rc<Cell<bool>>,
    }

    impl Tab for Fake {
        fn ui(&mut self, _: &mut egui::Ui, _: &mut eframe::Frame) {}
        fn label(&self) -> String {
            String::new()
        }
        fn dataset(&self) -> &Path {
            Path::new("")
        }
        fn dirty(&self) -> bool {
            self.dirty
        }
        fn window_title(&self) -> String {
            String::new()
        }
        fn request_close(&mut self) {
            self.state.set(if self.dirty {
                CloseState::Asking
            } else {
                CloseState::Closed
            });
        }
        fn close_state(&self) -> CloseState {
            self.state.get()
        }
        fn on_exit(&mut self) {
            self.exited.set(true);
        }
    }

    /// Quit with a clean and a dirty tab: the clean one goes at once, the
    /// window waits for the dirty one's answer.
    #[test]
    fn quit_asks_dirty_tabs_then_closes_or_cancels() {
        for answer in [CloseState::Closed, CloseState::Open] {
            let (cs, ds) = (
                Rc::new(Cell::new(CloseState::Open)),
                Rc::new(Cell::new(CloseState::Open)),
            );
            let clean_exit = Rc::new(Cell::new(false));
            let mut tabs: Vec<Box<dyn Tab>> = vec![
                Box::new(Fake {
                    state: cs,
                    dirty: false,
                    exited: clean_exit.clone(),
                }),
                Box::new(Fake {
                    state: ds.clone(),
                    dirty: true,
                    exited: Rc::default(),
                }),
            ];
            let (mut active, mut quitting) = (0, true);
            tabs.iter_mut().for_each(|t| t.request_close());

            assert!(!reap(&mut tabs, &mut active, &mut quitting));
            assert!(clean_exit.get(), "closed tab gets on_exit");
            assert_eq!((tabs.len(), active, quitting), (1, 0, true));

            ds.set(answer);
            let close = reap(&mut tabs, &mut active, &mut quitting);
            assert_eq!(close, answer == CloseState::Closed);
            assert_eq!(quitting, answer == CloseState::Closed);
            assert_eq!(tabs.len(), usize::from(answer == CloseState::Open));
        }
    }
}
