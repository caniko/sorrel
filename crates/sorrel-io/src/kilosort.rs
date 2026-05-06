//! Concrete Kilosort/phy2 backend. Memory-maps `spike_times.npy`,
//! `spike_clusters.npy`, the raw `.dat`, and `cluster_group.tsv`.

use crate::npy::{read_header, NpyHeader};
use crate::provider::{ClusterId, DataProvider, SampleIndex, TraceSlice};
use anyhow::{anyhow, bail, Context, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

/// V1 phy2 schema as a `#[repr(u8)]` enum so the label vector packs to
/// 1 byte per cluster.
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum PhyLabel {
    #[default]
    Unsorted = 0,
    Good = 1,
    Mua = 2,
    Noise = 3,
}

impl PhyLabel {
    pub fn parse_tsv_value(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "good" => Self::Good,
            "mua" => Self::Mua,
            "noise" => Self::Noise,
            _ => Self::Unsorted,
        }
    }
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unsorted => "unsorted",
            Self::Good => "good",
            Self::Mua => "mua",
            Self::Noise => "noise",
        }
    }
}

pub struct KilosortProvider {
    #[allow(dead_code)]
    root: PathBuf,
    sample_rate: f32,
    n_channels: u32,
    n_samples: SampleIndex,

    // mmaps kept alive for the lifetime of the provider; the slices below
    // borrow into them via raw pointers cast back inside accessor methods.
    _spike_times_mmap: Mmap,
    _spike_clusters_mmap: Mmap,
    _trace_mmap: Mmap,

    // Per-cluster spike-time vectors (we re-bucket once at load time so the
    // hot `spike_times(cluster)` call is a single slice borrow).
    spikes_per_cluster: Vec<Vec<SampleIndex>>,
    initial_labels: Vec<PhyLabel>,

    // Raw pointers + lengths for the memory-mapped trace, exposed as `&[i16]`.
    trace_ptr: *const i16,
    trace_len: usize,
}

// SAFETY: the raw pointer aliases data inside `_trace_mmap` which lives as long
// as the struct. The mmap is read-only and Rust's borrow checker enforces
// non-aliasing on the produced `&[i16]` slices.
unsafe impl Send for KilosortProvider {}
unsafe impl Sync for KilosortProvider {}

#[derive(Debug, Clone)]
pub struct KilosortOpenParams {
    pub sample_rate: f32,
    pub n_channels: u32,
    /// Path to the raw int16 binary (e.g. `recording.dat`); resolved relative
    /// to the kilosort directory if it isn't absolute.
    pub dat_path: PathBuf,
}

impl KilosortProvider {
    pub fn open(root: impl AsRef<Path>, params: KilosortOpenParams) -> Result<Self> {
        let root = root.as_ref().to_path_buf();

        let spike_times_path = root.join("spike_times.npy");
        let spike_clusters_path = root.join("spike_clusters.npy");
        let cluster_group_path = root.join("cluster_group.tsv");

        let st_hdr = read_header(&spike_times_path)?;
        let sc_hdr = read_header(&spike_clusters_path)?;

        if st_hdr.elem_count() != sc_hdr.elem_count() {
            bail!("spike_times and spike_clusters length mismatch");
        }
        let n_spikes = st_hdr.elem_count();

        let st_mmap = unsafe { Mmap::map(&File::open(&spike_times_path)?)? };
        let sc_mmap = unsafe { Mmap::map(&File::open(&spike_clusters_path)?)? };

        let times = read_u64_array(&st_mmap, &st_hdr, n_spikes)?;
        let clusters = read_u32_array(&sc_mmap, &sc_hdr, n_spikes)?;

        let max_cluster = clusters.iter().copied().max().unwrap_or(0);
        let n_clusters = (max_cluster as usize) + 1;

        let mut spikes_per_cluster: Vec<Vec<SampleIndex>> = vec![Vec::new(); n_clusters];
        for (t, &c) in times.iter().zip(clusters.iter()) {
            spikes_per_cluster[c as usize].push(*t);
        }
        // Each bucket is monotonically non-decreasing in Kilosort outputs but
        // we don't trust that strictly:
        for b in spikes_per_cluster.iter_mut() {
            b.sort_unstable();
        }

        let initial_labels = load_cluster_groups(&cluster_group_path, n_clusters);

        let dat_path = if params.dat_path.is_absolute() {
            params.dat_path.clone()
        } else {
            root.join(&params.dat_path)
        };
        let trace_file = File::open(&dat_path)
            .with_context(|| format!("open {}", dat_path.display()))?;
        let trace_mmap = unsafe { Mmap::map(&trace_file)? };
        let bytes = trace_mmap.len();
        if bytes % (params.n_channels as usize * 2) != 0 {
            bail!("dat file size not divisible by n_channels * 2 bytes");
        }
        let n_samples = (bytes / (params.n_channels as usize * 2)) as u64;
        let trace_ptr = trace_mmap.as_ptr() as *const i16;
        let trace_len = bytes / 2;

        Ok(Self {
            root,
            sample_rate: params.sample_rate,
            n_channels: params.n_channels,
            n_samples,
            _spike_times_mmap: st_mmap,
            _spike_clusters_mmap: sc_mmap,
            _trace_mmap: trace_mmap,
            spikes_per_cluster,
            initial_labels,
            trace_ptr,
            trace_len,
        })
    }

    fn trace_slice(&self) -> &[i16] {
        // SAFETY: pointer is valid for the lifetime of `self`; underlying
        // mmap is read-only.
        unsafe { std::slice::from_raw_parts(self.trace_ptr, self.trace_len) }
    }
}

impl DataProvider for KilosortProvider {
    type Label = PhyLabel;

    #[inline]
    fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
    #[inline]
    fn n_channels(&self) -> u32 {
        self.n_channels
    }
    #[inline]
    fn n_samples(&self) -> SampleIndex {
        self.n_samples
    }
    #[inline]
    fn n_clusters(&self) -> u32 {
        self.spikes_per_cluster.len() as u32
    }

    #[inline]
    fn spike_times(&self, cluster: ClusterId) -> &[SampleIndex] {
        self.spikes_per_cluster
            .get(cluster as usize)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    #[inline]
    fn trace(&self, start: SampleIndex, len: u32) -> TraceSlice<'_> {
        let nc = self.n_channels as usize;
        let s = (start as usize).saturating_mul(nc);
        let e = s + (len as usize) * nc;
        let buf = self.trace_slice();
        let e = e.min(buf.len());
        let s = s.min(e);
        TraceSlice {
            start,
            n_channels: self.n_channels,
            samples: &buf[s..e],
        }
    }

    fn initial_labels(&self) -> Vec<Self::Label> {
        self.initial_labels.clone()
    }
}

fn read_u64_array(mmap: &Mmap, hdr: &NpyHeader, n: usize) -> Result<Vec<SampleIndex>> {
    let off = hdr.data_offset as usize;
    let bytes = &mmap[off..];
    match hdr.dtype.as_str() {
        "<i8" | "<u8" => Err(anyhow!("unexpected 1-byte dtype for spike_times")),
        "<i64" | "<u64" => {
            let need = n * 8;
            if bytes.len() < need {
                bail!("spike_times truncated");
            }
            let mut out = Vec::with_capacity(n);
            for chunk in bytes[..need].chunks_exact(8) {
                out.push(u64::from_le_bytes(chunk.try_into().unwrap()));
            }
            Ok(out)
        }
        "<i32" | "<u32" => {
            let need = n * 4;
            if bytes.len() < need {
                bail!("spike_times truncated");
            }
            let mut out = Vec::with_capacity(n);
            for chunk in bytes[..need].chunks_exact(4) {
                out.push(u32::from_le_bytes(chunk.try_into().unwrap()) as u64);
            }
            Ok(out)
        }
        d => bail!("unsupported spike_times dtype {d}"),
    }
}

fn read_u32_array(mmap: &Mmap, hdr: &NpyHeader, n: usize) -> Result<Vec<u32>> {
    let off = hdr.data_offset as usize;
    let bytes = &mmap[off..];
    match hdr.dtype.as_str() {
        "<i32" | "<u32" => {
            let need = n * 4;
            if bytes.len() < need {
                bail!("spike_clusters truncated");
            }
            let mut out = Vec::with_capacity(n);
            for chunk in bytes[..need].chunks_exact(4) {
                out.push(u32::from_le_bytes(chunk.try_into().unwrap()));
            }
            Ok(out)
        }
        "<i64" | "<u64" => {
            let need = n * 8;
            if bytes.len() < need {
                bail!("spike_clusters truncated");
            }
            let mut out = Vec::with_capacity(n);
            for chunk in bytes[..need].chunks_exact(8) {
                out.push(u64::from_le_bytes(chunk.try_into().unwrap()) as u32);
            }
            Ok(out)
        }
        d => bail!("unsupported spike_clusters dtype {d}"),
    }
}

fn load_cluster_groups(path: &Path, n_clusters: usize) -> Vec<PhyLabel> {
    let mut out = vec![PhyLabel::Unsorted; n_clusters];
    let Ok(text) = std::fs::read_to_string(path) else {
        return out;
    };
    let mut lines = text.lines();
    let Some(_header) = lines.next() else {
        return out;
    };
    for line in lines {
        let mut cols = line.split('\t');
        let (Some(id), Some(group)) = (cols.next(), cols.next()) else {
            continue;
        };
        let Ok(id) = id.trim().parse::<usize>() else {
            continue;
        };
        if id < out.len() {
            out[id] = PhyLabel::parse_tsv_value(group);
        }
    }
    out
}

/// Convenience: count spikes per cluster (mostly for tests / status bars).
pub fn spike_counts<P: DataProvider>(p: &P) -> HashMap<ClusterId, usize> {
    (0..p.n_clusters())
        .map(|c| (c, p.spike_times(c).len()))
        .collect()
}
