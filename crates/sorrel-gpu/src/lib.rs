//! sorrel-gpu: GPU compute kernels mirroring hot paths in `sorrel-compute`.
//!
//! Targets are operations whose CPU cost dominates interactive trace
//! preprocessing on Neuropixels-grade data:
//!
//! * **CMR** — per-time-step median across channels (`cmr::GpuCmr`).
//! * **Biquad HP filter** — IIR filter, embarrassingly parallel across
//!   channels (`filter::GpuBiquad`).
//! * **Mean snippet** — reduce/average over `(n_spikes, snippet_len)` matrix
//!   (`snippets::GpuMeanSnippet`).
//! * **Matched filter** — per-unit spatiotemporal template filter bank for spike
//!   rescue (`matched_filter::GpuMatchedFilter`).
//!
//! All kernels share a [`GpuContext`] which owns the wgpu device/queue. The
//! UI bootstrap should construct the context from the eframe-supplied
//! `RenderState` so we don't pay for a second GPU adapter.

// wgpu's `Device: Send` proof nests deeper than the default 128 on current
// nightlies (future-compat lint `recursion_depth_exceeding_limit`).
#![recursion_limit = "256"]

use std::sync::Arc;

mod check;
pub mod cmr;
pub mod filter;
pub mod matched_filter;
pub mod snippets;

pub use check::self_check;
pub use cmr::GpuCmr;
pub use filter::GpuBiquad;
pub use matched_filter::{GpuMatchedFilter, MatchedFilterBank};
pub use snippets::GpuMeanSnippet;

/// Holds a wgpu `Device`/`Queue` pair. Cheap to clone.
#[derive(Clone)]
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

impl GpuContext {
    /// Create a headless GPU context. Picks a high-performance adapter and
    /// blocks until it's ready. Used by tests and CLI tools that don't have
    /// an existing render context handy.
    pub fn headless() -> Result<Self, GpuInitError> {
        pollster::block_on(Self::headless_async())
    }

    /// Honours the same `WGPU_BACKEND`, `WGPU_ADAPTER_NAME` and
    /// `WGPU_POWER_PREF` overrides eframe applies to the GUI.
    async fn headless_async() -> Result<Self, GpuInitError> {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::PRIMARY;
        descriptor.flags = instance_flags();
        let instance = wgpu::Instance::new(descriptor.with_env());
        let adapter = match wgpu::util::initialize_adapter_from_env(&instance, None).await {
            Ok(adapter) => adapter,
            Err(_) => instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::from_env()
                        .unwrap_or(wgpu::PowerPreference::HighPerformance),
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .map_err(|_| GpuInitError::NoAdapter)?,
        };
        Self::from_adapter(&adapter).await
    }

    /// Open a device on `adapter` with everything it supports, so
    /// [`supports_compute`] reports the adapter's real capability.
    pub async fn from_adapter(adapter: &wgpu::Adapter) -> Result<Self, GpuInitError> {
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("sorrel-gpu.headless"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GpuInitError::DeviceRequest(e.to_string()))?;
        log_uncaptured_errors(&device);
        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }

    /// Build a context from device/queue handles already owned by another
    /// subsystem (typically eframe's render state).
    pub fn from_existing(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self { device, queue }
    }
}

/// wgpu's default instance flags without indirect-call validation, for every
/// Sorrel instance (GUI included). That validation builds an internal compute
/// pipeline on each device, which loses the device on GL contexts without
/// compute (NVIDIA's EGL reports GL 3.3). Sorrel issues no indirect calls.
/// On D3D12 this zeroes `@builtin(num_workgroups)`, so shaders must not use
/// it. `WGPU_VALIDATION_INDIRECT_CALL=1` turns it back on.
pub fn instance_flags() -> wgpu::InstanceFlags {
    wgpu::InstanceFlags::from_build_config()
        .difference(wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL)
        .with_env()
}

/// Whether `device` can run every kernel in this crate. False on eframe's GL
/// fallback, which requests WebGL2 limits (no compute, no storage buffers);
/// callers then keep preprocessing on the CPU.
pub fn supports_compute(device: &wgpu::Device) -> bool {
    let l = device.limits();
    l.max_compute_invocations_per_workgroup >= 256
        && l.max_compute_workgroup_size_x >= 256
        && l.max_storage_buffers_per_shader_stage >= 2
}

/// Reject a dispatch the device would fail validation on (wgpu panics on
/// uncaptured validation errors), so the caller falls back to the CPU.
pub(crate) fn check_limits(
    device: &wgpu::Device,
    bytes: u64,
    workgroups: u32,
) -> Result<(), String> {
    let l = device.limits();
    let max_bytes = l.max_storage_buffer_binding_size.min(l.max_buffer_size);
    if bytes > max_bytes {
        return Err(format!(
            "{bytes} B exceeds the {max_bytes} B storage buffer limit"
        ));
    }
    if workgroups > l.max_compute_workgroups_per_dimension {
        return Err(format!(
            "{workgroups} workgroups exceed the {} per-dimension limit",
            l.max_compute_workgroups_per_dimension
        ));
    }
    Ok(())
}

#[derive(Debug)]
pub enum GpuInitError {
    NoAdapter,
    DeviceRequest(String),
    /// Shader or pipeline creation was rejected (a driver or shader-compiler
    /// bug, e.g. FXC on D3D12). Callers keep preprocessing on the CPU.
    Kernel(String),
}

impl std::fmt::Display for GpuInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "no compatible GPU adapter"),
            Self::DeviceRequest(e) => write!(f, "device request failed: {e}"),
            Self::Kernel(e) => write!(f, "kernel creation failed: {e}"),
        }
    }
}

impl std::error::Error for GpuInitError {}

/// Log wgpu errors that no error scope caught instead of wgpu's default
/// handler, which panics. Sorrel installs it on every device it opens and on
/// eframe's device once the GUI's own pipelines exist; egui-wgpu 0.34 sets no
/// handler of its own.
pub fn log_uncaptured_errors(device: &wgpu::Device) {
    device.on_uncaptured_error(Arc::new(|e| log::error!("wgpu error (not fatal): {e}")));
}

/// Run `build` (shader module, layouts and pipeline creation) under
/// Validation and Internal error scopes, so a backend that rejects a kernel
/// yields `Err` instead of wgpu's fatal uncaptured-error panic. Scopes are
/// per thread and `build` runs on this one. wgpu keeps the first error.
pub(crate) fn guarded<T>(
    device: &wgpu::Device,
    what: &str,
    build: impl FnOnce() -> T,
) -> Result<T, GpuInitError> {
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let out = build();
    let validation = pollster::block_on(validation.pop());
    let internal = pollster::block_on(internal.pop());
    match validation.or(internal) {
        Some(e) => Err(GpuInitError::Kernel(format!("{what}: {e}"))),
        None => Ok(out),
    }
}

/// Block on a `MapAsync` future. wgpu requires polling the device while a
/// readback is in flight; this helper does the standard channel dance.
pub(crate) fn block_on_map(
    device: &wgpu::Device,
    slice: wgpu::BufferSlice<'_>,
) -> Result<(), wgpu::BufferAsyncError> {
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    // `PollType::wait_indefinitely` polls until all submitted work is done, including the
    // map operation. After that the channel must have a value.
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device polling failed while waiting for map_async");
    rx.recv().expect("map_async never returned")
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::{GpuContext, GpuInitError};
    use std::ops::Deref;
    use std::sync::{Mutex, MutexGuard, OnceLock};

    static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());
    static GPU_TEST_CONTEXT: OnceLock<Result<GpuContext, String>> = OnceLock::new();

    pub(crate) struct TestGpuContext {
        _guard: MutexGuard<'static, ()>,
        ctx: &'static GpuContext,
    }

    impl Deref for TestGpuContext {
        type Target = GpuContext;

        fn deref(&self) -> &Self::Target {
            self.ctx
        }
    }

    pub(crate) fn ctx() -> Option<TestGpuContext> {
        let guard = GPU_TEST_LOCK.lock().expect("GPU test lock poisoned");
        match GPU_TEST_CONTEXT.get_or_init(|| {
            GpuContext::headless()
                .map_err(format_gpu_error)
                .and_then(|ctx| match super::supports_compute(&ctx.device) {
                    true => Ok(ctx),
                    false => Err("adapter lacks compute support".into()),
                })
        }) {
            Ok(ctx) => Some(TestGpuContext { _guard: guard, ctx }),
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                None
            }
        }
    }

    fn format_gpu_error(error: GpuInitError) -> String {
        error.to_string()
    }
}

#[cfg(test)]
mod tests {
    /// A shader the backend rejects must come back as `Err`, not wgpu's fatal
    /// uncaptured-error panic (the FXC failure seen under wine).
    #[test]
    fn rejected_pipeline_is_an_error_not_a_panic() {
        let Some(ctx) = crate::test_support::ctx() else {
            return;
        };
        let d = &ctx.device;
        let bad = crate::guarded(d, "bad", || {
            let m = d.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("bad"),
                source: wgpu::ShaderSource::Wgsl("@compute fn main() { nope(); }".into()),
            });
            d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("bad"),
                layout: None,
                module: &m,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        assert!(
            matches!(bad, Err(crate::GpuInitError::Kernel(_))),
            "{bad:?}"
        );
        // The scopes are popped: a good build still succeeds afterwards.
        assert!(crate::cmr::GpuCmr::new(&ctx).is_ok());
    }
}
