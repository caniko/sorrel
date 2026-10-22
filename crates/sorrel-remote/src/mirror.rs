//! Sparse local copy of the host's raw recording, filled on demand.
//!
//! The file has the host payload's exact length so `KilosortProvider` can
//! mmap it like a real recording. A 64 KiB block bitmap says which bytes are
//! real; everything else reads as zeros and is never shown, because
//! [`TraceFetch::ensure`] reports the window as not resident.
//!
//! ponytail: the bitmap lives in memory only, so a fresh `join` refetches.
//! Persist it atomically next to the file when sessions get long.

use std::fs::{File, OpenOptions};
use std::ops::Range;
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use piying::actor::ActorRef;
use sorrel_io::provider::TraceFetch;

use crate::link::{Fetch, LinkActor};
use crate::protocol::CHUNK;

pub const BLOCK: u64 = 64 * 1024;

pub struct Mirror {
    file: File,
    len: u64,
    resident: Vec<AtomicU64>,
    inflight: Vec<AtomicU64>,
    pending: AtomicUsize,
    link: ActorRef<LinkActor>,
}

fn bit(words: &[AtomicU64], i: u64) -> bool {
    words[(i / 64) as usize].load(Ordering::Acquire) & (1 << (i % 64)) != 0
}

fn set(words: &[AtomicU64], i: u64, on: bool) {
    let mask = 1u64 << (i % 64);
    let w = &words[(i / 64) as usize];
    if on {
        w.fetch_or(mask, Ordering::AcqRel);
    } else {
        w.fetch_and(!mask, Ordering::AcqRel);
    }
}

impl Mirror {
    pub fn create(path: &Path, len: u64, link: ActorRef<LinkActor>) -> Result<Arc<Self>> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .with_context(|| format!("create mirror {}", path.display()))?;
        file.set_len(len)?;
        let n_blocks = len.div_ceil(BLOCK);
        let words = (n_blocks.div_ceil(64)) as usize;
        Ok(Arc::new(Self {
            file,
            len,
            resident: (0..words).map(|_| AtomicU64::new(0)).collect(),
            inflight: (0..words).map(|_| AtomicU64::new(0)).collect(),
            pending: AtomicUsize::new(0),
            link,
        }))
    }

    fn blocks(&self, bytes: &Range<u64>) -> Range<u64> {
        let end = bytes.end.min(self.len);
        if bytes.start >= end {
            return 0..0;
        }
        (bytes.start / BLOCK)..end.div_ceil(BLOCK)
    }

    /// Called by the link actor with the host's reply for one fetch.
    pub fn complete(&self, offset: u64, data: &[u8]) {
        if let Err(e) = write_at(&self.file, offset, data) {
            log::error!("mirror write at {offset}: {e}");
            self.fail(offset..offset + data.len() as u64);
            return;
        }
        // Only blocks fully covered by this write become resident; a short
        // reply at EOF covers its last block fully by construction.
        let start_block = offset.div_ceil(BLOCK);
        let end = offset + data.len() as u64;
        let end_block = if end >= self.len {
            end.div_ceil(BLOCK)
        } else {
            end / BLOCK
        };
        for b in start_block..end_block {
            set(&self.resident, b, true);
        }
        for b in (offset / BLOCK)..end.div_ceil(BLOCK) {
            set(&self.inflight, b, false);
        }
        self.pending.fetch_sub(1, Ordering::AcqRel);
    }

    /// Called by the link actor when a fetch failed; the next `ensure` retries.
    pub fn fail(&self, bytes: Range<u64>) {
        for b in self.blocks(&bytes) {
            set(&self.inflight, b, false);
        }
        self.pending.fetch_sub(1, Ordering::AcqRel);
    }
}

impl TraceFetch for Mirror {
    fn ensure(&self, bytes: Range<u64>) -> bool {
        let blocks = self.blocks(&bytes);
        if blocks.is_empty() {
            return true;
        }
        if blocks.clone().all(|b| bit(&self.resident, b)) {
            return true;
        }
        // Coalesce missing, not-yet-requested blocks into runs of ≤ CHUNK.
        let max_run = CHUNK / BLOCK;
        let mut run: Option<Range<u64>> = None;
        let flush = |run: &mut Option<Range<u64>>| {
            if let Some(r) = run.take() {
                let offset = r.start * BLOCK;
                let len = (r.end * BLOCK).min(self.len) - offset;
                self.pending.fetch_add(1, Ordering::AcqRel);
                if self.link.tell(Fetch { offset, len }).try_send().is_err() {
                    log::error!("link actor gone; trace fetch dropped");
                    self.fail(offset..offset + len);
                }
            }
        };
        for b in blocks {
            if bit(&self.resident, b) || bit(&self.inflight, b) {
                flush(&mut run);
                continue;
            }
            set(&self.inflight, b, true);
            match &mut run {
                Some(r) if r.end == b && r.end - r.start < max_run => r.end = b + 1,
                _ => {
                    flush(&mut run);
                    run = Some(b..b + 1);
                }
            }
        }
        flush(&mut run);
        false
    }

    fn pending(&self) -> usize {
        self.pending.load(Ordering::Acquire)
    }
}

pub(crate) fn write_at(f: &File, offset: u64, data: &[u8]) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        f.write_all_at(data, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0;
        while done < data.len() {
            let n = f.seek_write(&data[done..], offset + done as u64)?;
            if n == 0 {
                return Err(std::io::ErrorKind::WriteZero.into());
            }
            done += n;
        }
        Ok(())
    }
}
