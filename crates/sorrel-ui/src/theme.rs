//! Shared colours: the categorical cluster palette plus semantic colours
//! derived from the active egui visuals, so views stop hard-coding greys
//! that only read well in dark mode.

use egui::{Color32, Ui};
use sorrel_io::ClusterId;

/// Categorical palette used to colour clusters across views (Tableau-10
/// subset). Index modulo length so we never run out — phy uses the same
/// trick.
pub const PALETTE: [Color32; 8] = [
    Color32::from_rgb(78, 121, 167),
    Color32::from_rgb(242, 142, 43),
    Color32::from_rgb(225, 87, 89),
    Color32::from_rgb(118, 183, 178),
    Color32::from_rgb(89, 161, 79),
    Color32::from_rgb(237, 201, 72),
    Color32::from_rgb(176, 122, 161),
    Color32::from_rgb(255, 157, 167),
];

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
        let v = ui.visuals();
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
}
