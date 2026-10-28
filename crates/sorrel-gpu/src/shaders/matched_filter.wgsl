// Per-unit spatiotemporal matched filter over a (n_time, n_chan) row-major
// chunk:
//   out[u][t] = sum_c sum_k filters[u][c][k] * data[(t + k + shifts[ch]) % n_time][ch]
// with ch = chans[u][c]. gid.x = t, gid.y = u. `shifts` are pre-reduced to
// [0, n_time) on the host (zeros for the real pass, random per-channel
// circular shifts for null replicates).
//
// ponytail: naive loop, one thread per (t, u). Neighbouring threads read
// `data` with stride n_chan (uncoalesced) and re-read the same filter; stage
// a (workgroup + nt) x C data tile in workgroup memory (or transpose the
// chunk to channel-major) if this is the bottleneck.

struct Params {
    n_time: u32,
    n_chan: u32,
    nt: u32,
    c_per_unit: u32,
    n_units: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read> filters: array<f32>;
@group(0) @binding(3) var<storage, read> chans: array<u32>;
@group(0) @binding(4) var<storage, read> shifts: array<u32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    let u = gid.y;
    if (t >= params.n_time || u >= params.n_units) {
        return;
    }
    var acc: f32 = 0.0;
    for (var c: u32 = 0u; c < params.c_per_unit; c = c + 1u) {
        let ch = chans[u * params.c_per_unit + c];
        let base = t + shifts[ch];
        let fbase = (u * params.c_per_unit + c) * params.nt;
        for (var k: u32 = 0u; k < params.nt; k = k + 1u) {
            acc = acc + filters[fbase + k] * data[((base + k) % params.n_time) * params.n_chan + ch];
        }
    }
    out[u * params.n_time + t] = acc;
}
