// CMR via in-workgroup bitonic sort.
//
// Layout: data is [t0_ch0..t0_chN-1, t1_ch0..t1_chN-1, ...] (time-major).
// Each workgroup handles one time step. Inside the workgroup, the 256 threads
// cooperatively bitonic-sort the row (padded to 512) and the median is read
// out at index n_channels/2.
//
// Limit: n_channels ≤ 512. Above that the host falls back to multi-pass or CPU.

const PADDED: u32 = 512u;
const PAD_SENTINEL: f32 = 3.4028235e+38; // f32::MAX — sorts to the high end.

struct Params {
    n_channels: u32,
    n_time_steps: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

var<workgroup> shared_buf: array<f32, 512>;
var<workgroup> shared_median: f32;

fn compare_swap(i: u32, partner: u32, ascending: bool) {
    if (partner > i) {
        let a = shared_buf[i];
        let b = shared_buf[partner];
        let should_swap = (ascending && a > b) || (!ascending && a < b);
        if (should_swap) {
            shared_buf[i] = b;
            shared_buf[partner] = a;
        }
    }
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let t = wg.x;
    if (t >= params.n_time_steps) {
        return;
    }
    let nc = params.n_channels;
    let row = t * nc;
    let tid = lid.x;

    // Load — each thread covers two slots.
    let a0 = tid * 2u;
    let a1 = a0 + 1u;
    shared_buf[a0] = select(PAD_SENTINEL, data[row + a0], a0 < nc);
    shared_buf[a1] = select(PAD_SENTINEL, data[row + a1], a1 < nc);
    workgroupBarrier();

    // Bitonic sort on 512 elements.
    var k: u32 = 2u;
    loop {
        if (k > PADDED) { break; }
        var j: u32 = k >> 1u;
        loop {
            if (j == 0u) { break; }
            // Each thread owns indices a0 and a1.
            let p0 = a0 ^ j;
            let p1 = a1 ^ j;
            let asc0 = (a0 & k) == 0u;
            let asc1 = (a1 & k) == 0u;
            compare_swap(a0, p0, asc0);
            compare_swap(a1, p1, asc1);
            workgroupBarrier();
            j = j >> 1u;
        }
        k = k << 1u;
    }

    // Median: thread 0 picks it.
    if (tid == 0u) {
        if ((nc & 1u) == 1u) {
            shared_median = shared_buf[nc / 2u];
        } else {
            shared_median = 0.5 * (shared_buf[nc / 2u - 1u] + shared_buf[nc / 2u]);
        }
    }
    workgroupBarrier();

    // Subtract — strided write so all 256 threads participate.
    let m = shared_median;
    var i = tid;
    loop {
        if (i >= nc) { break; }
        data[row + i] = data[row + i] - m;
        i = i + 256u;
    }
}
