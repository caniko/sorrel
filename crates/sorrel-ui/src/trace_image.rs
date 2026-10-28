//! phy v2.1's TraceImageView: the trace window as an image, one row per
//! channel and one column per pixel, coloured by the signed voltage
//! (blue negative, red positive). It follows the Trace pane's window and
//! HP/CMR toggles, so the two stay in step.

use egui::{Color32, ColorImage, Sense, TextureHandle, TextureOptions, Ui};
use sorrel_data::Session;
use sorrel_io::{DataProvider, SampleIndex};
use sorrel_render::{apply_preproc_cpu, TracePreproc};

use crate::theme::Theme;
use crate::views::{axes_data_rect, draw_axes};

#[derive(Default)]
pub struct TraceImageCache {
    key: Option<(u64, u32, TracePreproc, Vec<u32>, usize, bool)>,
    texture: Option<TextureHandle>,
}

/// Signed value → diverging blue-white-red (or blue-black-red on dark).
fn diverging(v: f32, dark: bool) -> Color32 {
    let t = v.clamp(-1.0, 1.0);
    let base = if dark { 0.0 } else { 255.0 };
    let lerp = |a: f32, b: f32| (a + (b - a) * t.abs()) as u8;
    if t < 0.0 {
        Color32::from_rgb(lerp(base, 40.0), lerp(base, 90.0), lerp(base, 220.0))
    } else {
        Color32::from_rgb(lerp(base, 220.0), lerp(base, 50.0), lerp(base, 40.0))
    }
}

/// Peak-magnitude (signed) per channel and column of `width` columns over
/// the interleaved `buf` (`n_samples × nc`), scaled to `[-1, 1]` by the
/// 99th percentile magnitude.
fn image_columns(buf: &[f32], nc: usize, width: usize) -> Vec<f32> {
    let n = buf.len() / nc.max(1);
    let mut out = vec![0.0f32; nc * width];
    for col in 0..width {
        let (s0, s1) = (
            col * n / width,
            ((col + 1) * n / width).max(col * n / width + 1).min(n),
        );
        for ch in 0..nc {
            out[ch * width + col] = (s0..s1).map(|s| buf[s * nc + ch]).fold(0.0f32, |m, v| {
                if v.abs() > m.abs() {
                    v
                } else {
                    m
                }
            });
        }
    }
    let mut mags: Vec<f32> = out.iter().map(|v| v.abs()).collect();
    let k = (mags.len() * 99 / 100).min(mags.len().saturating_sub(1));
    let scale = if mags.is_empty() {
        1.0
    } else {
        *mags.select_nth_unstable_by(k, f32::total_cmp).1
    }
    .max(f32::MIN_POSITIVE);
    out.iter_mut().for_each(|v| *v /= scale);
    out
}

#[allow(clippy::too_many_arguments)]
pub fn trace_image_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    window_start: u64,
    window_len: u32,
    cfg: TracePreproc,
    shanks: &[u32],
    flip: bool,
    cache: &mut TraceImageCache,
) {
    let theme = Theme::from_ui(ui);
    let p = &session.provider;
    let nc = p.n_channels() as usize;
    let avail = ui.available_size_before_wrap();
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(avail.x.max(120.0), avail.y.max(100.0)),
        Sense::hover(),
    );
    let data = axes_data_rect(rect, false);
    let width = (data.width() as usize).clamp(16, 4096);
    let key = (
        window_start,
        window_len,
        cfg,
        shanks.to_vec(),
        width,
        theme.dark,
    );
    if cache.key.as_ref() != Some(&key) || cache.texture.is_none() {
        let slice = p.trace(SampleIndex(window_start), window_len);
        if nc == 0 || slice.samples.len() != window_len as usize * nc {
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "raw data not available for this window",
                egui::FontId::proportional(13.0),
                theme.caption,
            );
            return;
        }
        let mut buf = slice.samples.to_f32_centred();
        if !cfg.is_off() {
            apply_preproc_cpu(&mut buf, nc, cfg, p.sample_rate(), shanks);
        }
        let cols = image_columns(&buf, nc, width);
        let pixels = (0..nc)
            .flat_map(|row| {
                let ch = if flip { nc - 1 - row } else { row };
                cols[ch * width..(ch + 1) * width]
                    .iter()
                    .map(|&v| diverging(v, theme.dark))
            })
            .collect();
        let image = ColorImage::new([width, nc], pixels);
        match &mut cache.texture {
            Some(t) => t.set(image, TextureOptions::NEAREST),
            None => {
                cache.texture = Some(ui.ctx().load_texture(
                    "trace-image",
                    image,
                    TextureOptions::NEAREST,
                ))
            }
        }
        cache.key = Some(key);
    }
    if let Some(t) = &cache.texture {
        ui.painter().image(
            t.id(),
            data,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            Color32::WHITE,
        );
    }
    let sr = p.sample_rate().max(1.0);
    draw_axes(
        &ui.painter_at(rect),
        rect,
        (
            window_start as f32 / sr,
            (window_start + window_len as u64) as f32 / sr,
        ),
        (0.0, 0.0),
        "time (s)",
        "",
        &theme,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn columns_keep_the_signed_peak_and_normalise() {
        // 4 samples × 2 channels, interleaved; 2 columns.
        let buf = [1.0, -2.0, -3.0, 0.5, 2.0, 0.0, 0.0, 4.0];
        let cols = image_columns(&buf, 2, 2);
        // ch0: [-3, 2], ch1: [-2, 4]; the 99th-percentile magnitude is 4.
        assert_eq!(cols, vec![-0.75, 0.5, -0.5, 1.0]);
        assert_eq!(diverging(0.0, true), Color32::from_rgb(0, 0, 0));
    }
}
