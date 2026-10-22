//! Wire messages between guest and host. Every message id is pinned so a
//! refactor can never silently change the protocol.

use serde::{Deserialize, Serialize};
use sorrel_data::CurationCommand;
use sorrel_io::TraceDtype;

pub const PROTO_VERSION: u32 = 1;
/// Largest byte range a single `FetchFile` / `FetchTrace` may ask for.
pub const CHUNK: u64 = 1 << 20;

/// Journal position a record was applied against. A reseal resets
/// `n_records`, so the baseline is always part of the cursor.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, piying::Reply)]
pub struct Cursor {
    pub baseline: u64,
    pub n_records: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct FileEntry {
    /// Forward-slash relative path under the dataset root.
    pub path: String,
    pub len: u64,
    pub blake3: [u8; 32],
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct DatInfo {
    /// Bytes after the header offset: exactly the size of the guest mirror.
    pub payload_bytes: u64,
    pub dtype: u8,
    pub n_channels: u32,
    pub sample_rate: f32,
}

impl DatInfo {
    pub fn dtype(&self) -> Option<TraceDtype> {
        dtype_from_tag(self.dtype)
    }
}

pub fn dtype_tag(d: TraceDtype) -> u8 {
    match d {
        TraceDtype::I16 => 0,
        TraceDtype::U16 => 1,
        TraceDtype::I32 => 2,
        TraceDtype::F32 => 3,
    }
}

pub fn dtype_from_tag(t: u8) -> Option<TraceDtype> {
    Some(match t {
        0 => TraceDtype::I16,
        1 => TraceDtype::U16,
        2 => TraceDtype::I32,
        3 => TraceDtype::F32,
        _ => return None,
    })
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HostInfo {
    pub proto: u32,
    pub files: Vec<FileEntry>,
    pub dat: DatInfo,
    pub cursor: Cursor,
    pub n_clusters_at_baseline: u32,
    #[serde(with = "serde_bytes")]
    pub identity_bytes: Vec<u8>,
}

// ---- requests (guest → host) ----

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Hello {
    pub proto: u32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FetchFile {
    pub path: String,
    pub offset: u64,
    pub len: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct JournalTail {
    pub from: u64,
}

/// Payload-relative byte range of the raw recording.
#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct FetchTrace {
    pub offset: u64,
    pub len: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Record {
    pub cursor: Cursor,
    pub cmd: CurationCommand,
}

// ---- replies ----

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct Bytes(#[serde(with = "serde_bytes")] pub Vec<u8>);

impl std::fmt::Debug for Bytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Bytes({} B)", self.0.len())
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub enum HostError {
    /// Protocol version mismatch.
    Version { host: u32 },
    /// The host journal is not where the guest thinks it is.
    Diverged { host: Cursor },
    /// Unknown path, out-of-range offset, oversized request.
    Rejected(String),
    /// Host-side IO or journal failure.
    Failed(String),
}

impl std::fmt::Display for HostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Version { host } => write!(
                f,
                "host speaks protocol v{host}, this build v{PROTO_VERSION}"
            ),
            Self::Diverged { host } => write!(
                f,
                "host journal moved (baseline {:016x}, {} records); re-join to resync",
                host.baseline, host.n_records
            ),
            Self::Rejected(s) => write!(f, "rejected by host: {s}"),
            Self::Failed(s) => write!(f, "host error: {s}"),
        }
    }
}

impl std::error::Error for HostError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_tags_round_trip() {
        for d in [
            TraceDtype::I16,
            TraceDtype::U16,
            TraceDtype::I32,
            TraceDtype::F32,
        ] {
            assert_eq!(dtype_from_tag(dtype_tag(d)), Some(d));
        }
        assert_eq!(dtype_from_tag(9), None);
    }

    #[test]
    fn bytes_encode_compactly() {
        let b = Bytes(vec![7u8; 4096]);
        let enc = rmp_serde::to_vec_named(&b).unwrap();
        assert!(
            enc.len() < 4096 + 16,
            "Vec<u8> must be a bin, not an int array: {}",
            enc.len()
        );
    }
}
