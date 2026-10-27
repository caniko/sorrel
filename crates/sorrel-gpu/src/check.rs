//! `sorrel gpu-check`: list every adapter wgpu can see and validate the
//! kernels against their `sorrel-compute` CPU references on each one.

use crate::{supports_compute, GpuBiquad, GpuCmr, GpuContext, GpuMeanSnippet};
use sorrel_compute::{mean_snippet, subtract_channel_median, Biquad, BiquadState};
use std::fmt::Write;

/// Human-readable report plus `true` unless an adapter produced a wrong result
/// or no adapter exists. A kernel-creation failure ("kernels unavailable", e.g.
/// a shader-compiler bug in one driver) is reported per adapter and only fails
/// the check when no compute-capable adapter could run the kernels at all.
pub fn self_check() -> (String, bool) {
    pollster::block_on(self_check_async())
}

async fn self_check_async() -> (String, bool) {
    let mut out = String::new();
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::all();
    descriptor.flags = crate::instance_flags();
    let instance = wgpu::Instance::new(descriptor.with_env());
    let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
    if adapters.is_empty() {
        out.push_str("no GPU adapters found (install a Vulkan or OpenGL driver)\n");
        return (out, false);
    }
    let mut ok = true;
    let (mut attempted, mut passed) = (0, 0);
    for adapter in &adapters {
        let i = adapter.get_info();
        let _ = writeln!(
            out,
            "{} [{:?} {:?}, vendor 0x{:04x}, driver {} {}]",
            i.name, i.backend, i.device_type, i.vendor, i.driver, i.driver_info
        );
        if i.backend == wgpu::Backend::Gl {
            // Mirror eframe's GL device (WebGL2 limits, so no compute): that
            // is what the GUI gets, and full limits can fail on GL drivers
            // that expose no compute (NVIDIA's headless EGL reports GL 3.3).
            let required_limits = wgpu::Limits {
                max_texture_dimension_2d: 8192,
                ..wgpu::Limits::downlevel_webgl2_defaults()
            };
            let desc = wgpu::DeviceDescriptor {
                required_limits,
                ..Default::default()
            };
            match adapter.request_device(&desc).await {
                Ok(_) => out.push_str("  GUI device: OK (trace preprocessing on CPU)\n"),
                Err(e) => {
                    let _ = writeln!(out, "  GUI device: FAIL ({e})");
                    ok = false;
                }
            }
            continue;
        }
        let ctx = match GpuContext::from_adapter(adapter).await {
            Ok(ctx) => ctx,
            Err(e) => {
                let _ = writeln!(out, "  device: FAIL ({e})");
                ok = false;
                continue;
            }
        };
        if !supports_compute(&ctx.device) {
            out.push_str("  kernels: skipped (no compute support)\n");
            continue;
        }
        attempted += 1;
        let set = match (
            GpuCmr::new(&ctx),
            GpuBiquad::new(&ctx),
            GpuMeanSnippet::new(&ctx),
        ) {
            (Ok(a), Ok(b), Ok(c)) => (a, b, c),
            (a, b, c) => {
                let e = a
                    .err()
                    .or(b.err())
                    .or(c.err())
                    .map_or_else(String::new, |e| e.to_string());
                let _ = writeln!(out, "  kernels unavailable: {e}");
                continue;
            }
        };
        match kernels(set) {
            Ok(()) => {
                passed += 1;
                out.push_str("  kernels: PASS\n");
            }
            Err(e) => {
                let _ = writeln!(out, "  kernels: FAIL ({e})");
                ok = false;
            }
        }
    }
    if attempted > 0 && passed == 0 {
        out.push_str("no adapter can run the compute kernels\n");
        ok = false;
    }
    let pick = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::from_env()
                .unwrap_or(wgpu::PowerPreference::HighPerformance),
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await;
    if let Ok(a) = pick {
        let i = a.get_info();
        let _ = writeln!(out, "preferred adapter: {} ({:?})", i.name, i.backend);
    }
    (out, ok)
}

fn kernels((cmr, biquad, snippets): (GpuCmr, GpuBiquad, GpuMeanSnippet)) -> Result<(), String> {
    // CMR on a Neuropixels-sized row.
    let nc = 384usize;
    let mut gpu: Vec<f32> = (0..nc * 64)
        .map(|i| ((i * 37) % 8191) as f32 - 4000.0)
        .collect();
    let mut cpu = gpu.clone();
    cmr.run(&mut gpu, nc as u32)
        .map_err(|e| format!("CMR: {e}"))?;
    subtract_channel_median(&mut cpu, nc);
    compare("CMR", &gpu, &cpu, |g, c| (g - c).abs() < 1e-3)?;

    // HP biquad, one IIR recursion per channel.
    let b = Biquad::butterworth_hp(300.0, 30_000.0);
    let nc = 8usize;
    let mut gpu: Vec<f32> = (0..nc * 2048)
        .map(|i| ((i as f32 * 0.013).sin() + (i as f32 * 0.057).cos()) * 100.0)
        .collect();
    let mut cpu = gpu.clone();
    biquad
        .run(&mut gpu, nc as u32, b)
        .map_err(|e| format!("biquad: {e}"))?;
    let mut states = vec![BiquadState::default(); nc];
    for row in cpu.chunks_mut(nc) {
        for (x, s) in row.iter_mut().zip(&mut states) {
            *x = b.step(*x, s);
        }
    }
    // The recursion amplifies rounding, and FMA fusion differs per driver
    // (llvmpipe drifts ~1e-3 near zero crossings); scale to the signal peak.
    let tol = 1e-4 * cpu.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    compare("biquad", &gpu, &cpu, |g, c| (g - c).abs() < tol)?;

    // Mean snippet.
    let len = 80usize;
    let rows: Vec<Vec<f32>> = (0..257)
        .map(|s| {
            (0..len)
                .map(|i| ((s * 13 + i * 7) % 91) as f32 - 45.0)
                .collect()
        })
        .collect();
    let gpu = snippets
        .run(&rows.concat(), len as u32)
        .map_err(|e| format!("mean snippet: {e}"))?;
    compare("mean snippet", &gpu, &mean_snippet(&rows), |g, c| {
        (g - c).abs() < 1e-3
    })
}

fn compare(
    name: &str,
    gpu: &[f32],
    cpu: &[f32],
    close: impl Fn(f32, f32) -> bool,
) -> Result<(), String> {
    if gpu.len() != cpu.len() {
        return Err(format!("{name}: length {} vs {}", gpu.len(), cpu.len()));
    }
    match gpu.iter().zip(cpu).position(|(&g, &c)| !close(g, c)) {
        Some(i) => Err(format!("{name}: index {i} gpu={} cpu={}", gpu[i], cpu[i])),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn self_check_passes_on_available_adapters() {
        let (report, ok) = super::self_check();
        eprintln!("{report}");
        // CI sandboxes have no adapter; only assert when one exists.
        if !report.starts_with("no GPU adapters") {
            assert!(ok, "{report}");
        }
    }
}
