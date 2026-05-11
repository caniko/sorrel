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
//!
//! All kernels share a [`GpuContext`] which owns the wgpu device/queue. The
//! UI bootstrap should construct the context from the eframe-supplied
//! `RenderState` so we don't pay for a second GPU adapter.

use std::sync::Arc;

pub mod cmr;
pub mod filter;
pub mod snippets;

pub use cmr::GpuCmr;
pub use filter::GpuBiquad;
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

    async fn headless_async() -> Result<Self, GpuInitError> {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(descriptor);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|_| GpuInitError::NoAdapter)?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("sorrel-gpu.headless"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GpuInitError::DeviceRequest(e.to_string()))?;
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

#[derive(Debug)]
pub enum GpuInitError {
    NoAdapter,
    DeviceRequest(String),
}

impl std::fmt::Display for GpuInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "no compatible GPU adapter"),
            Self::DeviceRequest(e) => write!(f, "device request failed: {e}"),
        }
    }
}

impl std::error::Error for GpuInitError {}

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
        match GPU_TEST_CONTEXT.get_or_init(|| GpuContext::headless().map_err(format_gpu_error)) {
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
