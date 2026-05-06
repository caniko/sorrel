use crate::intent::{from_key, Intent};
use crate::widgets::{cluster_table, trace_view};
use eframe::egui;
use sorrel_data::{CurationCommand, PhyLabelOp, Session};
use sorrel_data::session::ApplyPhyLabel;
use sorrel_io::{ClusterId, DataProvider};

/// Generic eframe app. The whole window is monomorphised against `P`.
pub struct SorrelApp<P: DataProvider + ApplyPhyLabel> {
    session: Session<P>,
    label_str: fn(&P::Label) -> &'static str,
    selected: ClusterId,
    window_start: u64,
    window_len: u32,
    target_points: usize,
    error: Option<String>,
}

impl<P: DataProvider + ApplyPhyLabel> SorrelApp<P> {
    pub fn new(session: Session<P>, label_str: fn(&P::Label) -> &'static str) -> Self {
        let window_len = (session.provider.sample_rate() as u32).max(1) / 10; // 100 ms
        Self {
            session,
            label_str,
            selected: 0,
            window_start: 0,
            window_len,
            target_points: 1024,
            error: None,
        }
    }

    fn dispatch_intent(&mut self, intent: Intent) {
        let n = self.session.provider.n_clusters();
        match intent {
            Intent::SelectCluster(c) => {
                if c < n {
                    self.selected = c;
                }
            }
            Intent::Relabel(c, op) => {
                if let Err(e) = self.session.dispatch(CurationCommand::Relabel { cluster: c, op }) {
                    self.error = Some(format!("journal write failed: {e}"));
                }
            }
            Intent::Undo => { /* V1: no-op */ }
            Intent::Redo => { /* V1: no-op */ }
            Intent::NextCluster => {
                if self.selected + 1 < n {
                    self.selected += 1;
                }
            }
            Intent::PrevCluster => {
                if self.selected > 0 {
                    self.selected -= 1;
                }
            }
            Intent::PageBack => {
                self.window_start = self.window_start.saturating_sub(self.window_len as u64);
            }
            Intent::PageForward => {
                let max = self
                    .session
                    .provider
                    .n_samples()
                    .saturating_sub(self.window_len as u64);
                self.window_start = (self.window_start + self.window_len as u64).min(max);
            }
        }
        // Suppress unused-warning for V1 stubs.
        let _ = PhyLabelOp::SetGood;
    }
}

impl<P: DataProvider + ApplyPhyLabel> eframe::App for SorrelApp<P> {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Input.
        let input_intents: Vec<Intent> = ctx.input(|i| {
            i.events
                .iter()
                .filter_map(|ev| match ev {
                    egui::Event::Key {
                        key,
                        pressed: true,
                        modifiers,
                        ..
                    } => from_key(*key, *modifiers, self.selected),
                    _ => None,
                })
                .collect()
        });
        for it in input_intents {
            self.dispatch_intent(it);
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Sorrel");
                ui.separator();
                ui.label(format!(
                    "{} clusters · {} channels · {:.1} kHz",
                    self.session.provider.n_clusters(),
                    self.session.provider.n_channels(),
                    self.session.provider.sample_rate() / 1000.0
                ));
                if let Some(err) = &self.error {
                    ui.colored_label(egui::Color32::RED, err);
                }
            });
        });

        egui::SidePanel::left("clusters")
            .resizable(true)
            .default_width(260.0)
            .show(ctx, |ui| {
                ui.heading("Clusters");
                cluster_table(ui, &self.session, &mut self.selected, self.label_str);
            });

        egui::TopBottomPanel::bottom("trace").resizable(true).default_height(240.0).show(ctx, |ui| {
            ui.label(format!(
                "trace @ {} samples, len {}",
                self.window_start, self.window_len
            ));
            trace_view(
                ui,
                &self.session,
                self.window_start,
                self.window_len,
                self.target_points,
            );
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading(format!("Cluster {}", self.selected));
            let n_spikes = self.session.provider.spike_times(self.selected).len();
            ui.label(format!("{n_spikes} spikes"));
            ui.label(format!(
                "label: {}",
                (self.label_str)(
                    &self
                        .session
                        .label(self.selected)
                        .unwrap_or_default()
                )
            ));
            ui.separator();
            ui.label("Keys: G=good, M=mua, N=noise, U=unsorted · J/K next/prev · H/L pan");
        });
    }
}
