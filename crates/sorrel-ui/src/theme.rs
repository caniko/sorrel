//! Shared colours: the categorical cluster palette plus semantic colours
//! derived from the active egui visuals, so views stop hard-coding greys
//! that only read well in dark mode.

use egui::{Color32, Stroke, Ui};
use sorrel_io::ClusterId;

/// Categorical palette used to colour clusters across views (Tableau-10
/// subset). Index modulo length so we never run out — phy uses the same
/// trick. Tableau's red and purple are swapped (red sits at slot 6, purple
/// at slot 2) so the third selection slot never collides with the danger red
/// used for errors and destructive actions.
pub const PALETTE: [Color32; 8] = [
    Color32::from_rgb(78, 121, 167),
    Color32::from_rgb(242, 142, 43),
    Color32::from_rgb(176, 122, 161),
    Color32::from_rgb(118, 183, 178),
    Color32::from_rgb(89, 161, 79),
    Color32::from_rgb(237, 201, 72),
    Color32::from_rgb(225, 87, 89),
    Color32::from_rgb(255, 157, 167),
];

/// Black or white, whichever reads better on `bg` (WCAG relative luminance), for text on slot-coloured chips.
pub fn contrast_text(bg: Color32) -> Color32 {
    let lin = |c: u8| {
        let c = f32::from(c) / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    let l = 0.2126 * lin(bg.r()) + 0.7152 * lin(bg.g()) + 0.0722 * lin(bg.b());
    // Contrast vs black is (l + .05) / .05, vs white 1.05 / (l + .05).
    if (l + 0.05) / 0.05 > 1.05 / (l + 0.05) {
        Color32::BLACK
    } else {
        Color32::WHITE
    }
}

/// Colour for a cluster by its *position in the selection*: the first
/// selected cluster is always blue, the second orange, and so on, and every
/// view agrees. Two selected clusters can therefore never share a colour
/// (until the ninth), which `id % 8` could not guarantee.
#[inline]
pub fn selection_colour(slot: usize) -> Color32 {
    PALETTE[slot % PALETTE.len()]
}

/// Colour for a cluster by id — only for views that draw *every* cluster
/// (the raster), where selection slots don't apply.
#[inline]
pub fn cluster_colour(c: ClusterId) -> Color32 {
    PALETTE[c.idx() % PALETTE.len()]
}

/// phy's cluster colour fields (`color_field_*`) for clusters outside the
/// selection in the Raster, Scatter and Templates panes. `Neutral` is
/// Sorrel's quiet default; Shift+wheel over those panes or `:cf name`
/// switches.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum ColourScheme {
    #[default]
    Neutral,
    Random,
    Group,
    Depth,
    FiringRate,
}

impl ColourScheme {
    pub const ALL: [ColourScheme; 5] = [
        ColourScheme::Neutral,
        ColourScheme::Random,
        ColourScheme::Group,
        ColourScheme::Depth,
        ColourScheme::FiringRate,
    ];

    /// phy's field names (`:cf group`).
    pub fn name(self) -> &'static str {
        match self {
            ColourScheme::Neutral => "none",
            ColourScheme::Random => "cluster",
            ColourScheme::Group => "group",
            ColourScheme::Depth => "depth",
            ColourScheme::FiringRate => "fr",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.name() == s)
    }

    pub fn step(self, delta: i32) -> Self {
        let i = Self::ALL.iter().position(|&c| c == self).unwrap_or(0) as i32;
        Self::ALL[(i + delta).rem_euclid(Self::ALL.len() as i32) as usize]
    }
}

/// Per-cluster colour under `scheme` (`None` = the view's neutral grey).
/// `depth` and `fr` are per cluster (NaN when unknown); firing rate is
/// log-scaled as phy does.
pub fn scheme_colours(
    scheme: ColourScheme,
    groups: &[&str],
    depth: &[f32],
    fr: &[f32],
) -> Vec<Option<Color32>> {
    let ramp = |vals: &[f32]| -> Vec<Option<Color32>> {
        let (lo, hi) = vals
            .iter()
            .filter(|v| v.is_finite())
            .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        vals.iter()
            .map(|&v| {
                v.is_finite()
                    .then(|| viridis(if hi > lo { (v - lo) / (hi - lo) } else { 0.5 }))
            })
            .collect()
    };
    match scheme {
        ColourScheme::Neutral => vec![None; groups.len()],
        ColourScheme::Random => (0..groups.len())
            .map(|c| Some(PALETTE[c % PALETTE.len()]))
            .collect(),
        ColourScheme::Group => groups
            .iter()
            .map(|g| match *g {
                "good" => Some(Color32::from_rgb(86, 180, 100)),
                "mua" => Some(Color32::from_rgb(240, 160, 60)),
                "noise" => Some(Color32::from_gray(110)),
                _ => None,
            })
            .collect(),
        ColourScheme::Depth => ramp(depth),
        ColourScheme::FiringRate => ramp(
            &fr.iter()
                .map(|&f| (f.max(1e-3)).log10())
                .collect::<Vec<_>>(),
        ),
    }
}

/// Five-stop viridis, `t` in `[0, 1]`.
pub fn viridis(t: f32) -> Color32 {
    const STOPS: [[f32; 3]; 5] = [
        [68.0, 1.0, 84.0],
        [59.0, 82.0, 139.0],
        [33.0, 145.0, 140.0],
        [94.0, 201.0, 98.0],
        [253.0, 231.0, 37.0],
    ];
    let x = t.clamp(0.0, 1.0) * 4.0;
    let i = (x.floor() as usize).min(3);
    let f = x - i as f32;
    let c = |k: usize| (STOPS[i][k] + (STOPS[i + 1][k] - STOPS[i][k]) * f) as u8;
    Color32::from_rgb(c(0), c(1), c(2))
}

/// Shift+wheel over `resp` (phy cycles the colour scheme that way):
/// `+1` / `-1` per notch, 0 otherwise.
pub fn scheme_wheel(ui: &Ui, resp: &egui::Response) -> i32 {
    if !resp.hovered() {
        return 0;
    }
    ui.input(|i| {
        i.events
            .iter()
            .map(|e| match e {
                egui::Event::MouseWheel {
                    delta, modifiers, ..
                } if modifiers.shift => {
                    let d = delta.x + delta.y;
                    -(d.signum() as i32) * (d != 0.0) as i32
                }
                _ => 0,
            })
            .sum()
    })
}

/// egui's stock weak text (dark), warning (light) and error colours miss
/// WCAG AA on its own backgrounds; `text_tokens_meet_wcag_aa_in_both_themes`
/// pins the replacements.
pub fn accessible_visuals(mut v: egui::Visuals) -> egui::Visuals {
    if v.dark_mode {
        // Body text brighter than weak text so the hierarchy stays visible
        // after weak was lifted to AA.
        v.widgets.noninteractive.fg_stroke.color = Color32::from_gray(165);
        v.weak_text_color = Some(Color32::from_gray(132));
        v.error_fg_color = Color32::from_rgb(255, 85, 85);
    } else {
        v.weak_text_color = Some(Color32::from_gray(105));
        v.warn_fg_color = Color32::from_rgb(175, 85, 0);
        v.error_fg_color = Color32::from_rgb(200, 0, 0);
        // Soft shadows instead of the hard bands of egui's stock light ones.
        let soft = egui::Shadow {
            offset: [0, 6],
            blur: 16,
            spread: 0,
            color: Color32::from_black_alpha(64),
        };
        v.window_shadow = soft;
        v.popup_shadow = soft;
    }
    // Unchecked checkboxes and idle buttons need a >=3:1 boundary.
    v.widgets.inactive.bg_stroke = Stroke::new(
        1.0_f32,
        if v.dark_mode {
            Color32::from_gray(105)
        } else {
            Color32::from_gray(135)
        },
    );
    v
}

/// Install [`accessible_visuals`] for both themes.
pub fn install_visuals(ctx: &egui::Context) {
    ctx.set_visuals_of(egui::Theme::Dark, accessible_visuals(egui::Visuals::dark()));
    ctx.set_visuals_of(
        egui::Theme::Light,
        accessible_visuals(egui::Visuals::light()),
    );
}

/// Density pass: type sizes and spacing for both themes. Call after
/// [`install_visuals`] (that one only touches colours).
pub fn install_style(ctx: &egui::Context) {
    use egui::{FontId, TextStyle};
    ctx.all_styles_mut(|style| {
        style
            .text_styles
            .insert(TextStyle::Body, FontId::proportional(14.0));
        style
            .text_styles
            .insert(TextStyle::Button, FontId::proportional(14.0));
        style
            .text_styles
            .insert(TextStyle::Small, FontId::proportional(11.0));
        style
            .text_styles
            .insert(TextStyle::Monospace, FontId::monospace(13.5));
        style
            .text_styles
            .insert(TextStyle::Heading, FontId::proportional(20.0));
        style.spacing.item_spacing = egui::vec2(8.0, 6.0);
        style.spacing.button_padding = egui::vec2(10.0, 4.0);
        style.spacing.interact_size.y = 24.0;
    });
}

/// Semantic colours resolved from the current `Visuals`, so plots follow
/// the light/dark theme instead of always painting on near-black.
#[derive(Clone, Copy, Debug)]
pub struct Theme {
    pub plot_bg: Color32,
    /// Slightly raised surface inside a plot (correlogram cells, probe
    /// sub-plots).
    pub panel: Color32,
    pub grid: Color32,
    pub caption: Color32,
    pub text: Color32,
    pub good: Color32,
    pub warn: Color32,
    pub danger: Color32,
    pub dark: bool,
}

impl Theme {
    pub fn from_ui(ui: &Ui) -> Self {
        Self::from_visuals(ui.visuals())
    }

    pub fn from_visuals(v: &egui::Visuals) -> Self {
        Self {
            plot_bg: v.extreme_bg_color,
            panel: v.faint_bg_color,
            grid: v.widgets.noninteractive.bg_stroke.color,
            caption: v.weak_text_color(),
            text: v.text_color(),
            good: if v.dark_mode {
                Color32::from_rgb(89, 161, 79)
            } else {
                Color32::from_rgb(40, 120, 40)
            },
            warn: v.warn_fg_color,
            danger: v.error_fg_color,
            dark: v.dark_mode,
        }
    }

    /// Colour for a phy label string as shown in the cluster table.
    pub fn label_colour(&self, label: &str) -> Color32 {
        match label {
            "good" => self.good,
            "mua" => self.warn,
            "noise" => self.caption,
            _ => self.text,
        }
    }

    /// Traffic-light colour for a composite quality score in `[0, 1]`.
    pub fn quality_colour(&self, q: f32) -> Color32 {
        if q < 0.4 {
            self.danger
        } else if q < 0.7 {
            self.warn
        } else {
            self.good
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colour_schemes_match_phy_fields() {
        let groups = ["good", "noise", "unsorted"];
        let g = scheme_colours(ColourScheme::Group, &groups, &[], &[]);
        assert!(
            g[0].is_some() && g[1].is_some() && g[2].is_none(),
            "unsorted stays neutral"
        );
        let d = scheme_colours(ColourScheme::Depth, &groups, &[10.0, f32::NAN, 30.0], &[]);
        assert_eq!(
            (d[0], d[1], d[2]),
            (Some(viridis(0.0)), None, Some(viridis(1.0)))
        );
        assert_eq!(ColourScheme::Neutral.step(-1), ColourScheme::FiringRate);
        assert_eq!(ColourScheme::parse("group"), Some(ColourScheme::Group));
    }

    #[test]
    fn selection_slots_are_distinct_up_to_palette_length() {
        let colours: Vec<_> = (0..PALETTE.len()).map(selection_colour).collect();
        for (i, a) in colours.iter().enumerate() {
            for b in &colours[i + 1..] {
                assert_ne!(a, b);
            }
        }
        assert_eq!(selection_colour(PALETTE.len()), selection_colour(0));
    }

    #[test]
    fn contrast_text_is_readable_on_every_slot() {
        assert_eq!(
            contrast_text(Color32::from_rgb(0xf2, 0x8e, 0x2b)),
            Color32::BLACK
        );
        assert_eq!(
            contrast_text(Color32::from_rgb(0xb0, 0x7a, 0xa1)),
            Color32::BLACK
        );
        let lum = |c: Color32| {
            let lin = |v: u8| {
                let v = f32::from(v) / 255.0;
                if v <= 0.04045 {
                    v / 12.92
                } else {
                    ((v + 0.055) / 1.055).powf(2.4)
                }
            };
            0.2126 * lin(c.r()) + 0.7152 * lin(c.g()) + 0.0722 * lin(c.b())
        };
        for bg in PALETTE {
            let (a, b) = (lum(bg) + 0.05, lum(contrast_text(bg)) + 0.05);
            let ratio = a.max(b) / a.min(b);
            assert!(ratio >= 4.5, "{bg:?} only reaches {ratio:.2}:1");
        }
    }

    #[test]
    fn plot_background_differs_between_light_and_dark() {
        // `plot_bg` is sourced from `extreme_bg_color`; if egui ever made
        // it theme-independent our plots would stop following the theme.
        assert_ne!(
            egui::Visuals::dark().extreme_bg_color,
            egui::Visuals::light().extreme_bg_color
        );
    }

    #[test]
    fn quality_colour_is_a_traffic_light() {
        let t = Theme {
            plot_bg: Color32::BLACK,
            panel: Color32::DARK_GRAY,
            grid: Color32::GRAY,
            caption: Color32::GRAY,
            text: Color32::WHITE,
            good: Color32::GREEN,
            warn: Color32::YELLOW,
            danger: Color32::RED,
            dark: true,
        };
        assert_eq!(t.quality_colour(0.1), Color32::RED);
        assert_eq!(t.quality_colour(0.5), Color32::YELLOW);
        assert_eq!(t.quality_colour(0.9), Color32::GREEN);
        assert_eq!(t.label_colour("good"), Color32::GREEN);
        assert_eq!(t.label_colour("unsorted"), Color32::WHITE);
    }

    /// WCAG 2 contrast ratio between two opaque colours.
    fn contrast(a: Color32, b: Color32) -> f32 {
        let lum = |c: Color32| {
            let ch = |v: u8| {
                let v = v as f32 / 255.0;
                if v <= 0.03928 {
                    v / 12.92
                } else {
                    ((v + 0.055) / 1.055).powf(2.4)
                }
            };
            0.2126 * ch(c.r()) + 0.7152 * ch(c.g()) + 0.0722 * ch(c.b())
        };
        let (hi, lo) = (lum(a).max(lum(b)), lum(a).min(lum(b)));
        (hi + 0.05) / (lo + 0.05)
    }

    #[test]
    fn text_tokens_meet_wcag_aa_in_both_themes() {
        // Body and status text need 4.5:1; captions are secondary, 3:1.
        let mut failures = Vec::new();
        let [dark, light] = [egui::Visuals::dark(), egui::Visuals::light()].map(accessible_visuals);
        // Pin the tokens: dark weak 132 under body 165, light weak opaque 105.
        assert_eq!(dark.weak_text_color(), Color32::from_gray(132));
        assert_eq!(dark.text_color(), Color32::from_gray(165));
        assert_eq!(light.weak_text_color(), Color32::from_gray(105));
        for v in [dark, light] {
            let t = Theme::from_visuals(&v);
            let edge = contrast(v.widgets.inactive.bg_stroke.color, v.panel_fill);
            if edge < 3.0 {
                failures.push(format!("widget boundary on panel_fill: {edge:.2} < 3"));
            }
            for (bg_name, bg) in [("plot_bg", t.plot_bg), ("panel_fill", v.panel_fill)] {
                for (name, fg, min) in [
                    ("text", t.text, 4.5),
                    ("good", t.good, 4.5),
                    ("warn", t.warn, 4.5),
                    ("danger", t.danger, 4.5),
                    ("caption", t.caption, 3.0),
                ] {
                    let r = contrast(fg, bg);
                    if r < min {
                        let mode = if t.dark { "dark" } else { "light" };
                        failures.push(format!("{mode} {name} on {bg_name}: {r:.2} < {min}"));
                    }
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}
