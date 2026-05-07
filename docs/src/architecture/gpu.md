# GPU Compute

`sorrel-gpu` is a small set of wgpu compute pipelines that share the
device and queue with eframe's renderer. When the app is installed onto
an `eframe::wgpu` render state, `install_gpu_compute` wires up the GPU
preprocessor; otherwise the trace path silently falls back to the CPU
kernels in `sorrel-compute`.

## Pipelines

| Shader | Pipeline |
|--------|----------|
| `cmr_median.wgsl` | Per-sample channel median for common-median referencing on the trace window. |
| `biquad_hp.wgsl` | Causal biquad high-pass filter applied per-channel after CMR. |
| `mean_snippet.wgsl` | Mean-snippet extraction along a window for waveform / template overlays. |

## Why GPU here

The trace view repaints on every pan and resize. CMR + HP filter on a
1-second × hundreds-of-channels window is fast on the CPU, but pinning
it to the same wgpu device the renderer already uses lets the
downsampled vertex stream stay on-device and avoids the round-trip
through host memory.

## Fallback

The CPU implementations live in `sorrel-compute::cmr` and
`sorrel-compute::filter` and are bit-equivalent up to floating-point
ordering. Headless tools (`--export-qc`) and any host without a wgpu
adapter use them directly.
