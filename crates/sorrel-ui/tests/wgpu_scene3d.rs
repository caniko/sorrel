//! GPU smoke test for the generic 3-D renderer: a red point in front of a
//! blue triangle must read back red at the centre (depth test), and the
//! clear colour must stay transparent outside both.

use sorrel_ui::wgpu_raster::pack_rgba;
use sorrel_ui::wgpu_scene3d::{Orbit, Point3, Scene3d, Scene3dCallback, Scene3dPipeline, Vert3};
use std::sync::Arc;

const N: u32 = 64;

/// Same lavapipe pinning as `ui_states.rs`; `None` skips the test.
fn gpu() -> Option<sorrel_gpu::GpuContext> {
    if let Some(icd) = std::env::var_os("SORREL_LAVAPIPE_ICD") {
        std::env::set_var("VK_DRIVER_FILES", &icd);
        std::env::set_var("VK_ICD_FILENAMES", &icd);
        std::env::remove_var("VK_LOADER_DRIVERS_SELECT");
        std::env::remove_var("VK_LOADER_DRIVERS_DISABLE");
    }
    match sorrel_gpu::GpuContext::headless() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("skipping scene3d GPU test: {e}");
            None
        }
    }
}

fn read_pixels(gpu: &sorrel_gpu::GpuContext, tex: &wgpu::Texture) -> Vec<u8> {
    let (device, queue) = (&gpu.device, &gpu.queue);
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (N * N * 4) as u64, // N * 4 = 256: already row-aligned
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        tex.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(N * 4),
                rows_per_image: Some(N),
            },
        },
        wgpu::Extent3d {
            width: N,
            height: N,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([enc.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    buf.slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let out = buf.slice(..).get_mapped_range().to_vec();
    out
}

#[test]
fn point_in_front_of_triangle_wins_depth_test() {
    let Some(gpu) = gpu() else { return };
    let red = pack_rgba(255, 0, 0, 255);
    let blue = pack_rgba(0, 0, 255, 255);
    let tri = |x, y| Vert3 {
        pos: [x, y, -0.5],
        color: blue,
    };
    let scene = Arc::new(Scene3d {
        points: Arc::new([Point3 {
            pos: [0.0; 3],
            color: red,
        }]),
        lines: Arc::new([]),
        tris: Arc::new([tri(-0.6, -0.6), tri(0.6, -0.6), tri(0.0, 0.6)]),
    });
    let orbit = Orbit {
        yaw: 0.0,
        pitch: 0.0,
        distance: 3.0,
        target: [0.0; 3],
    };
    let cb = Scene3dCallback {
        id: 1,
        scene,
        generation: 1,
        view_proj: orbit.view_proj(1.0),
        point_radius_px: 6.0,
        size_px: [N, N],
    };
    let mut pipeline = Scene3dPipeline::new(&gpu.device, wgpu::TextureFormat::Rgba8Unorm);
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    pipeline.render_offscreen(&gpu.device, &gpu.queue, &mut enc, &cb);
    gpu.queue.submit([enc.finish()]);
    let px = read_pixels(&gpu, pipeline.color_texture(1).expect("target"));
    let at = |x: u32, y: u32| &px[((y * N + x) * 4) as usize..][..4];
    assert_eq!(at(N / 2, N / 2), [255, 0, 0, 255], "centre should be red");
    // Inside the triangle, off the disc: blue. Corner: untouched.
    assert_eq!(at(N / 2, N / 2 + 12), [0, 0, 255, 255], "triangle body");
    assert_eq!(at(1, 1), [0, 0, 0, 0], "clear is transparent");
}
