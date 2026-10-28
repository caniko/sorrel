//! phy ClusterView: the cluster table, its row cache and filter expressions.
//!
//! One widget serves both the Clusters panel and the SimilarityView (phy
//! subclasses one from the other): [`TableCtx::similar`] restricts the rows
//! to a ranked list and adds a `similarity` column.

use crate::selection::SelectionSet;
use crate::theme::{selection_colour, Theme};
use egui::{Color32, Sense, Stroke, Ui, Vec2};
use sorrel_compute::{isi_violations, mean_amplitude, quality_breakdown};
use sorrel_data::Session;
use sorrel_io::{ClusterId, DataProvider, QualityMetrics};

/// Sortable column on the cluster table. The widget mutates the table state
/// on header clicks, so this stays in lockstep with the column rendering.
/// Serialized in the UI prefs: the unit variants keep their old names.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ClusterColumn {
    Id,
    SpikeCount,
    Amplitude,
    IsiViolations,
    Quality,
    Label,
    Tags,
    /// Peak channel of the primary template.
    Channel,
    Shank,
    Depth,
    FiringRate,
    /// SimilarityView only.
    Similarity,
    /// A column of the dataset's quality-metrics table, by name.
    Metric(String),
}

impl ClusterColumn {
    /// Column name (phy's, where phy has the column); the key in the
    /// hidden-columns pref, `sort_by` and filter expressions.
    pub fn name(&self) -> &str {
        match self {
            Self::Id => "id",
            Self::Channel => "ch",
            Self::Shank => "sh",
            Self::Depth => "depth",
            Self::FiringRate => "fr",
            Self::Amplitude => "amp",
            Self::SpikeCount => "n_spikes",
            Self::IsiViolations => "isi",
            Self::Quality => "qual",
            Self::Label => "label",
            Self::Tags => "tags",
            Self::Similarity => "similarity",
            Self::Metric(m) => m,
        }
    }

    fn header(&self) -> &str {
        match self {
            Self::Id => "ID",
            Self::SpikeCount => "# spikes",
            Self::IsiViolations => "ISI<1ms",
            other => other.name(),
        }
    }

    fn tip(&self) -> &'static str {
        match self {
            Self::Id => "Cluster id. Swatch = colour in the plots.",
            Self::Channel => "Peak channel of the primary template",
            Self::Shank => "Shank of the peak channel",
            Self::Depth => "Probe y of the peak channel (µm)",
            Self::FiringRate => "Spikes per second",
            Self::SpikeCount => "Spike count",
            Self::Amplitude => {
                "Peak-to-peak of the primary template (mean spike amplitude without templates)"
            }
            Self::IsiViolations => "Inter-spike intervals under 1 ms (refractory violations)",
            Self::Quality => "Composite quality 0–1: green ≥ 0.7, yellow ≥ 0.4, red below",
            Self::Label => "phy label: good, mua, noise or unsorted",
            Self::Tags => {
                "Provenance from `sorrel auto` (src:*, merged:*, bc:*, conf:*); review:* in red. \
                 Sort puts the review queue first"
            }
            Self::Similarity => "Template similarity to the last selected cluster",
            Self::Metric(_) => "Metric from the dataset's quality-metrics file",
        }
    }

    /// Everything but the id, the label and the tags is a right-aligned number.
    fn numeric(&self) -> bool {
        !matches!(self, Self::Id | Self::Label | Self::Tags)
    }

    fn width(&self) -> f32 {
        match self {
            Self::Id => 44.0,
            Self::SpikeCount => 65.0,
            Self::IsiViolations => 60.0,
            Self::Quality | Self::Channel | Self::Shank => 42.0,
            Self::Label => 56.0,
            Self::Tags => 70.0,
            _ => 56.0,
        }
    }

    /// Direction a first click sorts in: biggest first for magnitudes.
    fn default_desc(&self) -> bool {
        matches!(
            self,
            Self::SpikeCount
                | Self::Amplitude
                | Self::IsiViolations
                | Self::Quality
                | Self::Tags
                | Self::FiringRate
                | Self::Similarity
        )
    }

    fn fmt(&self, v: f64) -> String {
        match self {
            Self::Id
            | Self::Channel
            | Self::Shank
            | Self::SpikeCount
            | Self::IsiViolations
            | Self::Depth => format!("{v:.0}"),
            Self::FiringRate | Self::Quality => format!("{v:.2}"),
            Self::Amplitude => format!("{v:.1}"),
            Self::Similarity => format!("{v:.3}"),
            _ if v.fract() == 0.0 && v.abs() < 1e9 => format!("{v:.0}"),
            _ if v.abs() >= 100.0 => format!("{v:.1}"),
            _ => format!("{v:.3}"),
        }
    }
}

// ---- filter expressions ---------------------------------------------------

/// A value in a filter expression: a cluster's cell or a literal.
#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    Num(f64),
    Str(String),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Ident(String),
    Num(f64),
    Str(String),
    Op(CmpOp),
    And,
    Or,
    Not,
    Open,
    Close,
}

fn tokenize(s: &str) -> Option<Vec<Tok>> {
    let c: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < c.len() {
        let two = |a: char, b: char| c[i] == a && c.get(i + 1) == Some(&b);
        let (tok, len) = match c[i] {
            ch if ch.is_whitespace() => {
                i += 1;
                continue;
            }
            '(' => (Tok::Open, 1),
            ')' => (Tok::Close, 1),
            _ if two('=', '=') => (Tok::Op(CmpOp::Eq), 2),
            _ if two('!', '=') => (Tok::Op(CmpOp::Ne), 2),
            _ if two('<', '=') => (Tok::Op(CmpOp::Le), 2),
            _ if two('>', '=') => (Tok::Op(CmpOp::Ge), 2),
            _ if two('&', '&') => (Tok::And, 2),
            _ if two('|', '|') => (Tok::Or, 2),
            '<' => (Tok::Op(CmpOp::Lt), 1),
            '>' => (Tok::Op(CmpOp::Gt), 1),
            '!' => (Tok::Not, 1),
            q @ ('"' | '\'') => {
                let end = c[i + 1..].iter().position(|&x| x == q)? + i + 1;
                (Tok::Str(c[i + 1..end].iter().collect()), end + 1 - i)
            }
            ch if ch.is_ascii_digit() || ch == '.' || ch == '-' => {
                let mut end = i + 1;
                while end < c.len()
                    && (c[end].is_ascii_digit()
                        || c[end] == '.'
                        || (matches!(c[end], 'e' | 'E')
                            && c.get(end + 1)
                                .is_some_and(|x| x.is_ascii_digit() || *x == '-'))
                        || (c[end] == '-' && matches!(c[end - 1], 'e' | 'E')))
                {
                    end += 1;
                }
                let text: String = c[i..end].iter().collect();
                (Tok::Num(text.parse().ok()?), end - i)
            }
            ch if ch.is_alphabetic() || ch == '_' => {
                let mut end = i + 1;
                while end < c.len() && (c[end].is_alphanumeric() || matches!(c[end], '_' | '.')) {
                    end += 1;
                }
                let word: String = c[i..end].iter().collect();
                let tok = match word.to_ascii_lowercase().as_str() {
                    "and" => Tok::And,
                    "or" => Tok::Or,
                    "not" => Tok::Not,
                    _ => Tok::Ident(word),
                };
                (tok, end - i)
            }
            _ => return None,
        };
        out.push(tok);
        i += len;
    }
    Some(out)
}

/// A parsed phy v2.1 filter expression.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Cmp(String, CmpOp, Val),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Not(Box<Expr>),
}

struct Parser {
    toks: Vec<Tok>,
    i: usize,
}

impl Parser {
    fn eat(&mut self, t: &Tok) -> bool {
        let hit = self.toks.get(self.i) == Some(t);
        self.i += hit as usize;
        hit
    }

    fn or(&mut self) -> Option<Expr> {
        let mut e = self.and()?;
        while self.eat(&Tok::Or) {
            e = Expr::Or(Box::new(e), Box::new(self.and()?));
        }
        Some(e)
    }

    fn and(&mut self) -> Option<Expr> {
        let mut e = self.not()?;
        while self.eat(&Tok::And) {
            e = Expr::And(Box::new(e), Box::new(self.not()?));
        }
        Some(e)
    }

    fn not(&mut self) -> Option<Expr> {
        if self.eat(&Tok::Not) {
            return Some(Expr::Not(Box::new(self.not()?)));
        }
        self.atom()
    }

    fn atom(&mut self) -> Option<Expr> {
        if self.eat(&Tok::Open) {
            let e = self.or()?;
            return self.eat(&Tok::Close).then_some(e);
        }
        let (Some(Tok::Ident(col)), Some(Tok::Op(op))) = (
            self.toks.get(self.i).cloned(),
            self.toks.get(self.i + 1).cloned(),
        ) else {
            return None;
        };
        let val = match self.toks.get(self.i + 2)? {
            Tok::Num(n) => Val::Num(*n),
            Tok::Str(s) => Val::Str(s.clone()),
            _ => return None,
        };
        self.i += 3;
        Some(Expr::Cmp(col, op, val))
    }
}

impl Expr {
    /// `None` when the text is not a filter expression (the caller falls
    /// back to the id/tag substring match).
    pub fn parse(s: &str) -> Option<Self> {
        let mut p = Parser {
            toks: tokenize(s)?,
            i: 0,
        };
        let e = p.or()?;
        (p.i == p.toks.len()).then_some(e)
    }

    /// Evaluate against a row; `get` returns a column's cell. `None` is an
    /// evaluation error (unknown column, number vs string) — phy keeps such
    /// rows visible.
    pub fn eval(&self, get: &dyn Fn(&str) -> Option<Val>) -> Option<bool> {
        match self {
            Self::And(a, b) => Some(a.eval(get)? && b.eval(get)?),
            Self::Or(a, b) => Some(a.eval(get)? || b.eval(get)?),
            Self::Not(a) => Some(!a.eval(get)?),
            Self::Cmp(col, op, rhs) => {
                use CmpOp::*;
                let ord = match (get(col)?, rhs) {
                    (Val::Num(a), Val::Num(b)) => {
                        // NaN compares false except `!=`, like numpy.
                        return Some(match op {
                            Eq => a == *b,
                            Ne => a != *b,
                            Lt => a < *b,
                            Le => a <= *b,
                            Gt => a > *b,
                            Ge => a >= *b,
                        });
                    }
                    (Val::Str(a), Val::Str(b)) => a.cmp(b),
                    _ => return None,
                };
                Some(match op {
                    Eq => ord.is_eq(),
                    Ne => ord.is_ne(),
                    Lt => ord.is_lt(),
                    Le => ord.is_le(),
                    Gt => ord.is_gt(),
                    Ge => ord.is_ge(),
                })
            }
        }
    }
}

// ---- row cache --------------------------------------------------------------

/// Probe geometry and metrics the rows are derived from.
#[derive(Clone, Copy)]
pub struct RowInputs<'a> {
    pub positions: &'a [[f32; 2]],
    pub shanks: &'a [u32],
    pub metrics: &'a QualityMetrics,
}

/// Every table cell, computed once per curation version. Numeric columns
/// are indexed by cluster id; NaN is "no value" (rendered blank, sorted last).
#[derive(Default)]
pub struct ClusterRows {
    key: Option<(u64, usize, usize, usize)>,
    cols: Vec<(String, Vec<f64>)>,
    group: Vec<&'static str>,
    /// phy's display order of the labels, for sorting the label column.
    rank: Vec<f64>,
    has_ch: bool,
    has_sh: bool,
    has_depth: bool,
    has_tags: bool,
    metric_names: Vec<String>,
}

/// Names of the fixed numeric columns, in storage order.
const FIXED: [&str; 9] = [
    "ch", "sh", "depth", "fr", "amp", "n_spikes", "isi", "qual", "tags",
];

/// Peak-to-peak per channel of a `(n_samples, n_channels)` template; the
/// biggest one and its channel.
fn peak_ptp(template: &[f32], n_channels: usize) -> Option<(usize, f64)> {
    if n_channels == 0 || template.len() < n_channels {
        return None;
    }
    let mut lo = vec![f32::INFINITY; n_channels];
    let mut hi = vec![f32::NEG_INFINITY; n_channels];
    for frame in template.chunks_exact(n_channels) {
        for (ch, &v) in frame.iter().enumerate() {
            lo[ch] = lo[ch].min(v);
            hi[ch] = hi[ch].max(v);
        }
    }
    (0..n_channels)
        .map(|ch| (ch, (hi[ch] - lo[ch]) as f64))
        .max_by(|a, b| a.1.total_cmp(&b.1))
}

impl ClusterRows {
    /// Recompute when the curation version (or the inputs' sizes) moved.
    pub fn ensure<P: DataProvider>(
        &mut self,
        session: &Session<P>,
        inputs: RowInputs,
        label_str: impl Fn(&P::Label) -> &'static str,
    ) {
        let key = (
            session.version(),
            inputs.positions.len(),
            inputs.shanks.len(),
            inputs.metrics.len(),
        );
        if self.key == Some(key) {
            return;
        }
        use rayon::prelude::*;
        let n = session.n_clusters() as usize;
        let sr = session.provider.sample_rate().max(1.0);
        let total = session.provider.n_samples().0;
        let duration_s = total as f64 / sr as f64;
        let isi_window = (sr * 0.001).round() as u64;
        let refractory = (sr * 0.0015).round() as u64;
        let n_channels = session.template_shape().2;
        // [ch, sh, depth, fr, amp, n_spikes, isi, qual, tags] per cluster.
        let rows: Vec<[f64; 9]> = (0..n as u32)
            .into_par_iter()
            .map(ClusterId)
            .map(|c| {
                let times = session.spike_times(c);
                let mut r = [f64::NAN; 9];
                r[5] = times.len() as f64;
                if times.is_empty() {
                    return r;
                }
                let amps = session.spike_amplitudes(c);
                r[3] = times.len() as f64 / duration_s.max(f64::MIN_POSITIVE);
                r[4] = if amps.is_empty() {
                    f64::NAN
                } else {
                    mean_amplitude(amps) as f64
                };
                if let Some((ch, ptp)) = session
                    .primary_template(c)
                    .and_then(|t| session.template_waveform(t))
                    .and_then(|w| peak_ptp(w, n_channels))
                {
                    r[0] = ch as f64;
                    r[1] = inputs.shanks.get(ch).map_or(f64::NAN, |&s| s as f64);
                    r[2] = inputs.positions.get(ch).map_or(f64::NAN, |p| p[1] as f64);
                    r[4] = ptp;
                }
                r[6] = isi_violations(times, isi_window) as f64;
                r[7] = quality_breakdown(times, amps, refractory, total, sr, 50).composite() as f64;
                let tags = session.tags(c);
                let review = tags.iter().filter(|t| t.starts_with("review:")).count();
                r[8] = (review * 1000 + tags.len()) as f64;
                r
            })
            .collect();
        self.cols = FIXED
            .iter()
            .enumerate()
            .map(|(k, name)| (name.to_string(), rows.iter().map(|r| r[k]).collect()))
            .collect();
        self.has_ch = rows.iter().any(|r| r[0].is_finite());
        self.has_sh = !inputs.shanks.is_empty() && rows.iter().any(|r| r[1].is_finite());
        self.has_depth = rows.iter().any(|r| r[2].is_finite());
        self.has_tags = (0..n as u32).any(|c| !session.tags(ClusterId(c)).is_empty());
        self.group = (0..n as u32)
            .map(|c| label_str(&session.label(ClusterId(c)).unwrap_or_default()))
            .collect();
        // phy's display order: good, mua, unsorted, noise, anything else.
        self.rank = self
            .group
            .iter()
            .map(|g| match *g {
                "good" => 0.0,
                "mua" => 1.0,
                "unsorted" => 2.0,
                "noise" => 3.0,
                _ => 4.0,
            })
            .collect();
        self.metric_names = inputs.metrics.names().to_vec();
        for name in &self.metric_names {
            let mut col: Vec<f64> = inputs
                .metrics
                .get(name)
                .unwrap_or_default()
                .iter()
                .map(|&v| v as f64)
                .collect();
            // Clusters created by merges/splits have no metric row.
            col.resize(n, f64::NAN);
            self.cols.push((name.clone(), col));
        }
        self.key = Some(key);
    }

    fn get(&self, name: &str) -> Option<&[f64]> {
        self.cols
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_slice())
    }

    /// The columns this dataset can show, in display order. `similar` adds
    /// the similarity column right after the id.
    fn columns(&self, similar: bool) -> Vec<ClusterColumn> {
        use ClusterColumn::*;
        let mut v = vec![Id];
        if similar {
            v.push(Similarity);
        }
        for (show, col) in [
            (self.has_ch, Channel),
            (self.has_sh, Shank),
            (self.has_depth, Depth),
            (true, FiringRate),
            (true, Amplitude),
            (true, SpikeCount),
            (true, IsiViolations),
            (true, Quality),
            (true, Label),
            (self.has_tags, Tags),
        ] {
            if show {
                v.push(col);
            }
        }
        v.extend(self.metric_names.iter().cloned().map(Metric));
        v
    }
}

// ---- table state ------------------------------------------------------------

/// Persistent cluster-table UI state — sort + filter live on the app.
#[derive(Clone, Debug)]
pub struct ClusterTableState {
    pub sort_by: ClusterColumn,
    pub sort_descending: bool,
    pub filter_text: String,
    pub hide_empty: bool,
    pub hide_noise: bool,
    /// Only clusters carrying a `review:*` tag (the queue `sorrel auto` leaves).
    pub review_only: bool,
    /// The rows as last drawn — sorted and filtered. Keyboard navigation
    /// and Shift-click walk this, so "next" means the next row the user
    /// sees, not the next cluster id.
    pub visible: Vec<ClusterId>,
    /// Anchor the table last scrolled to; scroll again only when it
    /// changes so the wheel isn't fought every frame.
    last_scrolled: Option<ClusterId>,
    /// Metric columns of the last drawn rows, for [`Self::sort_by`] names.
    metric_names: Vec<String>,
}

impl Default for ClusterTableState {
    /// phy's default: most spikes first.
    fn default() -> Self {
        Self {
            sort_by: ClusterColumn::SpikeCount,
            sort_descending: true,
            filter_text: String::new(),
            hide_empty: false,
            hide_noise: false,
            review_only: false,
            visible: Vec::new(),
            last_scrolled: None,
            metric_names: Vec::new(),
        }
    }
}

impl ClusterTableState {
    /// The SimilarityView table: best match first.
    pub fn similarity() -> Self {
        Self {
            sort_by: ClusterColumn::Similarity,
            ..Self::default()
        }
    }

    /// Set the filter text (phy `:f`): an expression or an id/tag substring.
    pub fn set_filter(&mut self, expr: &str) {
        self.filter_text = expr.trim().to_owned();
    }

    /// Sort by the column called `col` (phy `:s`): its name or phy's alias,
    /// the first two characters without underscores. `descending == None`
    /// takes the column's natural direction. False when no column matches.
    pub fn sort_by(&mut self, col: &str, descending: Option<bool>) -> bool {
        use ClusterColumn::*;
        let q = col.trim().to_ascii_lowercase();
        let all: Vec<ClusterColumn> = [
            Id,
            Channel,
            Shank,
            Depth,
            FiringRate,
            Amplitude,
            SpikeCount,
            IsiViolations,
            Quality,
            Label,
            Tags,
            Similarity,
        ]
        .into_iter()
        .chain(self.metric_names.iter().cloned().map(Metric))
        .collect();
        let alias = |c: &ClusterColumn| -> [String; 2] {
            let n = c.name().to_ascii_lowercase();
            [
                n.chars().filter(|&x| x != '_').take(2).collect(),
                n.chars().take(2).filter(|&x| x != '_').collect(),
            ]
        };
        let Some(found) = all
            .iter()
            .find(|c| c.name().eq_ignore_ascii_case(&q))
            .or_else(|| all.iter().find(|c| alias(c).contains(&q)))
        else {
            return false;
        };
        self.sort_descending = descending.unwrap_or_else(|| found.default_desc());
        self.sort_by = found.clone();
        true
    }

    fn toggle_sort(&mut self, col: ClusterColumn) {
        if self.sort_by == col {
            self.sort_descending = !self.sort_descending;
        } else {
            self.sort_descending = col.default_desc();
            self.sort_by = col;
        }
    }
}

/// NaN last in either direction; the id breaks ties in the caller.
fn cmp_nan_last(a: f64, b: f64, descending: bool) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Equal,
        (true, false) => Greater,
        (false, true) => Less,
        _ if descending => b.total_cmp(&a),
        _ => a.total_cmp(&b),
    }
}

/// What a row click asks of the selection.
pub enum RowClick {
    Replace(ClusterId),
    Toggle(ClusterId),
    /// Shift-click: extend in table order.
    Extend(ClusterId),
}

impl RowClick {
    /// phy semantics for the Clusters panel.
    pub fn apply(self, selection: &mut SelectionSet, visible: &[ClusterId]) {
        match self {
            Self::Replace(c) => selection.replace(c),
            Self::Toggle(c) => selection.toggle(c),
            Self::Extend(c) => selection.extend_to(c, visible),
        }
    }
}

/// What one table draw reads from the app.
pub struct TableCtx<'a> {
    /// Selected clusters, best first, for the slot swatches.
    pub selected: &'a [ClusterId],
    /// Keyboard anchor to keep in view.
    pub anchor: Option<ClusterId>,
    pub rows: &'a ClusterRows,
    /// Column names the user hid (right-click a header).
    pub hidden: &'a mut Vec<String>,
    /// SimilarityView: list exactly these clusters and show their scores.
    pub similar: Option<&'a [(ClusterId, f32)]>,
}

/// Right-click header menu: one checkbox per column.
fn column_menu(ui: &mut Ui, all: &[ClusterColumn], hidden: &mut Vec<String>) {
    for c in all.iter().filter(|c| **c != ClusterColumn::Id) {
        let pos = hidden.iter().position(|h| h.eq_ignore_ascii_case(c.name()));
        let mut shown = pos.is_none();
        if ui.checkbox(&mut shown, c.name()).changed() {
            match pos {
                Some(p) => {
                    hidden.remove(p);
                }
                None => hidden.push(c.name().to_owned()),
            }
        }
    }
}

/// Header label with the sort arrow. The arrow slot is always laid out
/// (painted only when active) so headers don't change width when the sort
/// moves. Painted rather than a glyph: egui's bundled fonts have no ▲/▼.
fn header_label(ui: &mut Ui, label: &str, active: bool, up: bool, right: bool) -> egui::Response {
    let strong = ui.visuals().strong_text_color();
    let arrow = |ui: &mut Ui| {
        let (r, _) = ui.allocate_exact_size(egui::vec2(8.0, 10.0), Sense::hover());
        if active {
            let c = r.center();
            let (a, b) = if up { (2.5, -2.5) } else { (-2.5, 2.5) };
            ui.painter().add(egui::Shape::convex_polygon(
                vec![
                    egui::pos2(c.x - 4.0, c.y - a),
                    egui::pos2(c.x + 4.0, c.y - a),
                    egui::pos2(c.x, c.y - b),
                ],
                strong,
                Stroke::NONE,
            ));
        }
    };
    let text = egui::RichText::new(label).strong();
    if right {
        arrow(ui);
        ui.add(egui::Label::new(text).sense(Sense::click()))
    } else {
        let r = ui.add(egui::Label::new(text).sense(Sense::click()));
        arrow(ui);
        r
    }
}

/// Cluster table with click-to-sort headers and a small filter row above.
///
/// Multi-select rules match phy: plain click replaces, Cmd/Ctrl-click
/// toggles, Shift-click range-extends from the anchor. The click is
/// returned for the caller to apply.
pub fn cluster_table<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    state: &mut ClusterTableState,
    ctx: TableCtx,
) -> Option<RowClick> {
    use egui_extras::{Column, TableBuilder};

    let TableCtx {
        selected,
        anchor,
        rows,
        hidden,
        similar,
    } = ctx;
    let theme = Theme::from_ui(ui);
    if state.metric_names != rows.metric_names {
        state.metric_names.clone_from(&rows.metric_names);
    }

    // Filter strip above the table: the field on its own row, then the
    // toggles (they wrap when the panel is narrow). The visible/total count
    // lives in the panel heading.
    // The panel has no side margin so the table runs edge to edge; the
    // strip pads itself.
    egui::Frame::new()
        .inner_margin(egui::Margin::symmetric(12, 0))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                let label = ui.label("filter:");
                let edit = ui
                    .add(
                        egui::TextEdit::singleline(&mut state.filter_text)
                            .desired_width(ui.available_width())
                            .hint_text("id, tag or n_spikes > 50 and group == \"good\""),
                    )
                    .labelled_by(label.id);
                // Esc clears the field; egui drops focus on Esc in the same frame.
                if (edit.has_focus() || edit.lost_focus())
                    && ui.input(|i| i.key_pressed(egui::Key::Escape))
                {
                    state.filter_text.clear();
                }
                edit.on_hover_text(
                    "Cluster id or part of a tag (e.g. merged, src:llm, bc:snr), or a phy \
                     expression: columns, == != < <= > >=, and/or/not, parentheses; \
                     `group` is the label. Esc clears",
                );
            });
            ui.horizontal_wrapped(|ui| {
                ui.checkbox(&mut state.hide_empty, "hide empty");
                ui.checkbox(&mut state.hide_noise, "hide noise");
                ui.checkbox(&mut state.review_only, "review").on_hover_text(
                    "Only clusters `sorrel auto` tagged review:*; any edit clears the tag",
                );
            });
        });
    ui.add_space(4.0);

    let n = session.n_clusters() as usize;
    let mut sim: Vec<f64> = Vec::new();
    let ids: Vec<ClusterId> = match similar {
        Some(s) => {
            sim = vec![f64::NAN; n];
            for &(c, v) in s {
                if c.idx() < n {
                    sim[c.idx()] = v as f64;
                }
            }
            s.iter().map(|x| x.0).collect()
        }
        None => (0..n as u32).map(ClusterId).collect(),
    };

    // Filter, then sort.
    let filter = state.filter_text.trim();
    let expr = Expr::parse(filter);
    let lookup = |name: &str, c: ClusterId| match name.to_ascii_lowercase().as_str() {
        "group" | "label" => rows.group.get(c.idx()).map(|g| Val::Str((*g).to_owned())),
        "id" => Some(Val::Num(c.as_f64())),
        "similarity" if similar.is_some() => {
            Some(Val::Num(sim.get(c.idx()).copied().unwrap_or(f64::NAN)))
        }
        _ => rows
            .get(name)
            .map(|v| Val::Num(v.get(c.idx()).copied().unwrap_or(f64::NAN))),
    };
    let mut visible: Vec<ClusterId> = ids
        .into_iter()
        .filter(|&i| {
            if state.hide_empty && session.spike_times(i).is_empty() {
                return false;
            }
            if state.hide_noise && rows.group.get(i.idx()) == Some(&"noise") {
                return false;
            }
            let tags = session.tags(i);
            if state.review_only && !tags.iter().any(|t| t.starts_with("review:")) {
                return false;
            }
            if filter.is_empty() {
                return true;
            }
            match &expr {
                // A row whose predicate errors stays visible, as in phy.
                Some(e) => e.eval(&|name| lookup(name, i)).unwrap_or(true),
                None => i.to_string().contains(filter) || tags.iter().any(|t| t.contains(filter)),
            }
        })
        .collect();

    let sort_vals: Option<&[f64]> = match &state.sort_by {
        ClusterColumn::Id => None,
        ClusterColumn::Label => Some(&rows.rank),
        ClusterColumn::Similarity => Some(&sim),
        c => rows.get(c.name()),
    };
    let key = |c: ClusterId| {
        sort_vals.map_or(c.as_f64(), |v| v.get(c.idx()).copied().unwrap_or(f64::NAN))
    };
    let desc = state.sort_descending;
    visible.sort_by(|&a, &b| cmp_nan_last(key(a), key(b), desc).then(a.cmp(&b)));

    let all = rows.columns(similar.is_some());
    let shown: Vec<ClusterColumn> = all
        .iter()
        .filter(|c| {
            **c == ClusterColumn::Id || !hidden.iter().any(|h| h.eq_ignore_ascii_case(c.name()))
        })
        .cloned()
        .collect();
    // Wider than the panel: scroll sideways instead of squeezing.
    let scroll = shown.iter().map(|c| c.width() + 6.0).sum::<f32>() + 40.0 > ui.available_width();

    // Keep the keyboard anchor in view: scroll minimally (no alignment) and
    // only when the anchor changed, so clicking a visible row or wheeling
    // away from it never snaps the list.
    let mut scroll_to: Option<usize> = None;
    if let Some(anchor) = anchor {
        if state.last_scrolled != Some(anchor) {
            scroll_to = visible.iter().position(|&c| c == anchor);
            state.last_scrolled = Some(anchor);
        }
    }

    let value = |col: &ClusterColumn, i: ClusterId| -> f64 {
        let v = match col {
            ClusterColumn::Similarity => Some(sim.as_slice()),
            c => rows.get(c.name()),
        };
        v.and_then(|v| v.get(i.idx()).copied()).unwrap_or(f64::NAN)
    };

    let sort_col = state.sort_by.clone();
    let up = !state.sort_descending;
    let mut draw = |ui: &mut Ui| {
        let mut clicked: Option<ClusterColumn> = None;
        let mut picked: Option<RowClick> = None;
        // Selectable cell text would swallow the row click (clicking an ID
        // started a text selection instead of selecting the cluster).
        ui.style_mut().interaction.selectable_labels = false;
        // A low-alpha accent keeps the coloured qual/label cells readable on
        // the selected row; the slot bar carries the identity.
        let accent = ui.visuals().selection.bg_fill;
        ui.visuals_mut().selection.bg_fill =
            Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 80);
        let right = egui::Layout::right_to_left(egui::Align::Center);
        ui.spacing_mut().item_spacing.x = 6.0;
        // egui_extras adds item_spacing.y between rows; rows are 24 px flat.
        ui.spacing_mut().item_spacing.y = 0.0;
        // Not resizable: resize handles draw column rules down the whole panel.
        let mut table = TableBuilder::new(ui)
            .striped(true)
            .resizable(false)
            .sense(Sense::click())
            // Shrink vertically so the column rules stop at the last row.
            .auto_shrink([false, true]);
        for (k, c) in shown.iter().enumerate() {
            let w = c.width();
            table = if k + 1 == shown.len() && !scroll {
                table.column(Column::remainder().at_least(40.0))
            } else {
                table.column(Column::initial(w).at_least(w - 14.0))
            };
        }
        if let Some(row) = scroll_to {
            table = table.scroll_to_row(row, None);
        }
        table
            .header(26.0, |mut h| {
                for col in &shown {
                    h.col(|ui| {
                        // Rule under the header, unclipped so it spans the gaps.
                        let r = ui.max_rect();
                        ui.ctx().layer_painter(ui.layer_id()).hline(
                            r.x_range().expand(4.0),
                            r.bottom() - 0.5,
                            Stroke::new(1.0_f32, theme.grid),
                        );
                        let layout = if col.numeric() {
                            right
                        } else {
                            egui::Layout::left_to_right(egui::Align::Center)
                        };
                        ui.with_layout(layout, |ui| {
                            ui.spacing_mut().item_spacing.x = 3.0;
                            if *col == ClusterColumn::Id {
                                ui.add_space(15.0);
                            }
                            let r =
                                header_label(ui, col.header(), sort_col == *col, up, col.numeric())
                                    .on_hover_text(col.tip());
                            r.context_menu(|ui| column_menu(ui, &all, hidden));
                            if r.clicked() {
                                clicked = Some(col.clone());
                            }
                        });
                    });
                }
            })
            .body(|body| {
                body.rows(24.0, visible.len(), |mut row| {
                    let i = visible[row.index()];
                    let slot = selected.iter().position(|&c| c == i);
                    row.set_selected(slot.is_some());
                    let group = rows.group.get(i.idx()).copied().unwrap_or("unsorted");
                    // phy: good in green, mua and noise dimmed. Selected rows
                    // keep the normal text so the selection stays legible.
                    let tint = match (slot, group) {
                        (Some(_), _) => None,
                        (None, "good") => Some(theme.label_colour(group)),
                        (None, "mua" | "noise") => Some(theme.caption),
                        _ => None,
                    };
                    for col in &shown {
                        row.col(|ui| {
                            if tint.is_some() {
                                ui.visuals_mut().override_text_color = tint;
                            }
                            match col {
                                ClusterColumn::Id => {
                                    let cell = ui.max_rect();
                                    ui.horizontal(|ui| {
                                        ui.add_space(4.0);
                                        // The swatch slot is always allocated so ids don't
                                        // shift when a row becomes selected.
                                        let (r, _) = ui
                                            .allocate_exact_size(Vec2::splat(8.0), Sense::hover());
                                        if let Some(slot) = slot {
                                            ui.painter().rect_filled(
                                                r,
                                                2.0,
                                                selection_colour(slot),
                                            );
                                            let bar = egui::Rect::from_min_size(
                                                cell.left_top(),
                                                egui::vec2(2.0, cell.height()),
                                            );
                                            ui.painter().rect_filled(
                                                bar,
                                                0.0,
                                                selection_colour(slot),
                                            );
                                        }
                                        ui.label(format!("{i}"));
                                    });
                                }
                                ClusterColumn::Label => {
                                    // Same vertical centring as the numeric cells.
                                    ui.with_layout(
                                        egui::Layout::left_to_right(egui::Align::Center),
                                        |ui| {
                                            ui.add(
                                                egui::Label::new(
                                                    egui::RichText::new(group)
                                                        .color(theme.label_colour(group)),
                                                )
                                                .truncate(),
                                            );
                                        },
                                    );
                                }
                                ClusterColumn::Tags => {
                                    let tags = session.tags(i);
                                    if tags.is_empty() {
                                        return;
                                    }
                                    // Review reasons first and in red; the rest dimmed.
                                    let (review, rest): (Vec<&String>, Vec<&String>) =
                                        tags.iter().partition(|t| t.starts_with("review:"));
                                    ui.horizontal(|ui| {
                                        if !review.is_empty() {
                                            let text: Vec<&str> = review
                                                .iter()
                                                .map(|t| &t["review:".len()..])
                                                .collect();
                                            ui.colored_label(theme.danger, text.join(" "));
                                        }
                                        let src: Vec<&str> = rest
                                            .iter()
                                            .filter_map(|t| t.strip_prefix("src:"))
                                            .collect();
                                        ui.weak(src.join("+"));
                                    })
                                    .response
                                    .on_hover_text(tags.join("\n"));
                                }
                                num => {
                                    let v = value(num, i);
                                    // NaN is "no value": blank.
                                    if v.is_nan() {
                                        return;
                                    }
                                    let mut text = egui::RichText::new(num.fmt(v)).monospace();
                                    if *num == ClusterColumn::Quality {
                                        text = text.color(theme.quality_colour(v as f32));
                                    }
                                    ui.with_layout(right, |ui| ui.label(text));
                                }
                            }
                        });
                    }
                    // Whole-row click (phy semantics): plain replaces,
                    // Cmd/Ctrl toggles, Shift extends in table order.
                    if row.response().clicked() {
                        let mods = row.response().ctx.input(|inp| inp.modifiers);
                        picked = Some(if mods.command {
                            RowClick::Toggle(i)
                        } else if mods.shift {
                            RowClick::Extend(i)
                        } else {
                            RowClick::Replace(i)
                        });
                    }
                });
            });
        (clicked, picked)
    };
    let (clicked, picked) = if scroll {
        egui::ScrollArea::horizontal()
            .auto_shrink([false, true])
            .show(ui, draw)
            .inner
    } else {
        draw(ui)
    };
    if let Some(col) = clicked {
        state.toggle_sort(col);
    }
    state.visible = visible;
    picked
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(expr: &str, cells: &[(&str, Val)]) -> Option<bool> {
        let get = |name: &str| {
            cells
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, v)| v.clone())
        };
        Expr::parse(expr).expect("parses").eval(&get)
    }

    fn row() -> Vec<(&'static str, Val)> {
        vec![
            ("n_spikes", Val::Num(80.0)),
            ("amp", Val::Num(12.5)),
            ("group", Val::Str("good".into())),
        ]
    }

    #[test]
    fn parser_handles_ops_combinators_and_precedence() {
        let r = row();
        assert_eq!(eval("n_spikes > 50", &r), Some(true));
        assert_eq!(eval("n_spikes >= 80 && amp < 10", &r), Some(false));
        assert_eq!(eval("n_spikes != 80 or group == \"good\"", &r), Some(true));
        assert_eq!(eval("group == 'mua' || amp <= 12.5", &r), Some(true));
        assert_eq!(eval("not group == \"good\"", &r), Some(false));
        assert_eq!(eval("!(n_spikes < 50 and amp > 1)", &r), Some(true));
        // `and` binds tighter than `or`.
        assert_eq!(
            eval("amp > 100 and amp > 1 or n_spikes == 80", &r),
            Some(true)
        );
        assert_eq!(eval("amp > -1e1", &r), Some(true));
    }

    #[test]
    fn eval_errors_are_none_and_nan_compares_false() {
        let r = row();
        assert_eq!(eval("nope > 1", &r), None, "unknown column");
        assert_eq!(eval("group > 1", &r), None, "string vs number");
        let nan = [("x", Val::Num(f64::NAN))];
        assert_eq!(eval("x > 0", &nan), Some(false));
        assert_eq!(eval("x != 0", &nan), Some(true));
    }

    #[test]
    fn plain_text_is_not_an_expression_so_the_substring_filter_applies() {
        for text in [
            "12",
            "merged",
            "src:llm",
            "bc:snr",
            "n_spikes >",
            "(n_spikes > 1",
            "a > 1 b",
            "",
        ] {
            assert!(Expr::parse(text).is_none(), "{text:?} must fall back");
        }
    }

    #[test]
    fn sort_by_takes_names_and_phy_aliases() {
        let mut s = ClusterTableState {
            metric_names: vec!["ContamPct".into()],
            ..Default::default()
        };
        assert!(s.sort_by("n_spikes", None) && s.sort_by == ClusterColumn::SpikeCount);
        assert!(s.sort_descending, "magnitudes sort biggest first");
        assert!(s.sort_by("ns", Some(false)) && s.sort_by == ClusterColumn::SpikeCount);
        assert!(!s.sort_descending);
        assert!(
            s.sort_by("sh", None) && s.sort_by == ClusterColumn::Shank,
            "exact beats alias"
        );
        assert!(s.sort_by("si", None) && s.sort_by == ClusterColumn::Similarity);
        assert!(s.sort_by("AMP", None) && s.sort_by == ClusterColumn::Amplitude);
        assert!(
            s.sort_by("co", Some(true)) && s.sort_by == ClusterColumn::Metric("ContamPct".into())
        );
        assert!(!s.sort_by("bogus", None));
        assert_eq!(
            s.sort_by,
            ClusterColumn::Metric("ContamPct".into()),
            "unknown name changes nothing"
        );
    }

    #[test]
    fn set_filter_trims() {
        let mut s = ClusterTableState::default();
        s.set_filter("  amp > 3 ");
        assert_eq!(s.filter_text, "amp > 3");
    }

    #[test]
    fn nan_sorts_last_in_both_directions() {
        let mut v = [2.0, f64::NAN, 1.0, 3.0];
        v.sort_by(|a, b| cmp_nan_last(*a, *b, false));
        assert_eq!(&v[..3], &[1.0, 2.0, 3.0]);
        assert!(v[3].is_nan());
        v.sort_by(|a, b| cmp_nan_last(*a, *b, true));
        assert_eq!(&v[..3], &[3.0, 2.0, 1.0]);
        assert!(v[3].is_nan());
    }

    #[test]
    fn peak_ptp_finds_the_biggest_channel() {
        // 3 samples x 2 channels: ch0 spans 1, ch1 spans 4.
        let t = [0.0, 1.0, 1.0, -3.0, 0.5, 0.0];
        assert_eq!(peak_ptp(&t, 2), Some((1, 4.0)));
        assert_eq!(peak_ptp(&t, 0), None);
    }

    #[test]
    fn prefs_columns_keep_their_serialized_names() {
        let j = |c: &ClusterColumn| serde_json::to_string(c).unwrap();
        assert_eq!(j(&ClusterColumn::Quality), "\"Quality\"");
        assert_eq!(j(&ClusterColumn::SpikeCount), "\"SpikeCount\"");
        let back: ClusterColumn =
            serde_json::from_str(&j(&ClusterColumn::Metric("x".into()))).unwrap();
        assert_eq!(back, ClusterColumn::Metric("x".into()));
    }

    #[test]
    fn default_state_sorts_by_spike_count_descending() {
        let s = ClusterTableState::default();
        assert_eq!(s.sort_by, ClusterColumn::SpikeCount);
        assert!(s.sort_descending);
        assert!(s.filter_text.is_empty() && !s.hide_empty && !s.hide_noise);
        let sim = ClusterTableState::similarity();
        assert_eq!(sim.sort_by, ClusterColumn::Similarity);
        assert!(sim.sort_descending);
    }

    #[test]
    fn toggle_sort_flips_active_and_picks_natural_direction_for_new() {
        let mut s = ClusterTableState::default();
        s.toggle_sort(ClusterColumn::SpikeCount);
        assert!(!s.sort_descending, "active column flips");
        s.toggle_sort(ClusterColumn::Id);
        assert!(!s.sort_descending, "ids ascend");
        s.toggle_sort(ClusterColumn::Amplitude);
        assert!(s.sort_descending);
        s.toggle_sort(ClusterColumn::Label);
        assert!(!s.sort_descending, "text ascends");
    }
}
