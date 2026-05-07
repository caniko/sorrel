// Per-channel direct-form-II-transposed biquad. One invocation = one channel,
// stepping the IIR recursion along the time axis. Across channels the filter
// is independent so we can launch as many threads as we have channels.

struct Params {
    n_channels: u32,
    n_samples: u32,
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ch = gid.x;
    if (ch >= params.n_channels) {
        return;
    }
    let nc = params.n_channels;
    let n = params.n_samples;
    var s1: f32 = 0.0;
    var s2: f32 = 0.0;
    var t: u32 = 0u;
    loop {
        if (t >= n) { break; }
        let idx = t * nc + ch;
        let x = data[idx];
        let y = params.b0 * x + s1;
        s1 = params.b1 * x - params.a1 * y + s2;
        s2 = params.b2 * x - params.a2 * y;
        data[idx] = y;
        t = t + 1u;
    }
}
