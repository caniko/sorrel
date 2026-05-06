use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct TraceVertex {
    pub pos: [f32; 2],
    pub channel: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ScatterVertex {
    pub pos: [f32; 2],
    pub cluster: u32,
}
