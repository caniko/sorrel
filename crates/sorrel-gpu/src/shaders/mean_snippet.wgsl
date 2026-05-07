// Column-mean of a `(n_spikes, snippet_len)` row-major matrix. One thread
// per output position sums down its column; each thread is a stand-alone
// reduction so we don't need any workgroup-shared scratch.

struct Params {
    n_spikes: u32,
    snippet_len: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> snippets: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if (col >= params.snippet_len) {
        return;
    }
    let n = params.n_spikes;
    if (n == 0u) {
        out[col] = 0.0;
        return;
    }
    var acc: f32 = 0.0;
    var i: u32 = 0u;
    let stride = params.snippet_len;
    loop {
        if (i >= n) { break; }
        acc = acc + snippets[i * stride + col];
        i = i + 1u;
    }
    out[col] = acc / f32(n);
}
