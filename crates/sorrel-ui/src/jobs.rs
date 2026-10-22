//! Background jobs: run a closure against the shared session on a worker
//! thread, hand the result back to the UI thread, and drop it if the
//! inputs it was computed for are no longer what the UI wants.
//!
//! The session lives behind an `Arc<RwLock<_>>`. Workers hold a read lock
//! for the duration of one job; a curation command takes the write lock
//! and therefore waits for in-flight jobs — the UI cancels them first and
//! jobs poll the cancel flag between steps, so that wait stays short.
//!
//! ponytail: a job that ignores the flag inside a long sorrel-data call
//! (`rank_merge_candidates` on 1k+ clusters) can still hold the write
//! lock off for seconds; the upgrade path is a cancel parameter inside
//! those ranking loops.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Read-lock helper that survives a poisoned lock (a panicking worker
/// must not take the UI down with it).
pub fn read<S>(lock: &RwLock<S>) -> RwLockReadGuard<'_, S> {
    lock.read().unwrap_or_else(|p| p.into_inner())
}

pub fn write<S>(lock: &RwLock<S>) -> RwLockWriteGuard<'_, S> {
    lock.write().unwrap_or_else(|p| p.into_inner())
}

/// What a view needs to start a job: the shared session and a way to
/// wake the UI when the result lands.
pub struct JobCtx<S> {
    pub session: Arc<RwLock<S>>,
    pub egui: egui::Context,
}

impl<S> JobCtx<S> {
    pub fn repaint(&self) -> impl Fn() + Send + 'static {
        let ctx = self.egui.clone();
        move || ctx.request_repaint()
    }
}

/// One in-flight computation keyed by the inputs it was started for.
pub struct Job<K, T> {
    key: K,
    rx: Receiver<T>,
    cancel: Arc<AtomicBool>,
}

impl<K, T> Job<K, T> {
    /// Spawn `f` on a worker thread holding a read lock on `session`.
    /// `repaint` is called when the result is ready so an idle UI wakes up.
    pub fn spawn<S, F>(
        key: K,
        session: Arc<RwLock<S>>,
        repaint: impl Fn() + Send + 'static,
        f: F,
    ) -> Self
    where
        S: Send + Sync + 'static,
        T: Send + 'static,
        F: FnOnce(&S, &AtomicBool) -> T + Send + 'static,
    {
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = channel();
        let flag = Arc::clone(&cancel);
        std::thread::Builder::new()
            .name("sorrel-job".into())
            .spawn(move || {
                let guard = read(&session);
                if flag.load(Ordering::Relaxed) {
                    return;
                }
                let out = f(&guard, &flag);
                drop(guard);
                if !flag.load(Ordering::Relaxed) && tx.send(out).is_ok() {
                    repaint();
                }
            })
            .expect("spawn sorrel job thread");
        Self { key, rx, cancel }
    }

    pub fn key(&self) -> &K {
        &self.key
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// The result, once the worker delivered it.
    pub fn try_take(&self) -> Option<T> {
        self.rx.try_recv().ok()
    }
}

impl<K, T> Drop for Job<K, T> {
    fn drop(&mut self) {
        self.cancel();
    }
}

/// Last computed value plus, possibly, a job computing the next one.
pub struct Memo<K, T> {
    value: Option<(K, T)>,
    job: Option<Job<K, T>>,
}

impl<K, T> Default for Memo<K, T> {
    fn default() -> Self {
        Self {
            value: None,
            job: None,
        }
    }
}

impl<K: PartialEq + Clone, T> Memo<K, T> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Make sure a value for `key` exists or is being computed. Returns
    /// the value when it is current; `None` means a job is running (show
    /// a spinner). A finished job for another key is kept as the value so
    /// flipping back to it is instant, but a new job is started.
    pub fn ensure<S, F>(
        &mut self,
        key: K,
        session: &Arc<RwLock<S>>,
        repaint: impl Fn() + Send + 'static,
        f: F,
    ) -> Option<&T>
    where
        S: Send + Sync + 'static,
        T: Send + 'static,
        F: FnOnce(&S, &AtomicBool) -> T + Send + 'static,
    {
        if let Some(job) = &self.job {
            if let Some(v) = job.try_take() {
                self.value = Some((job.key().clone(), v));
                self.job = None;
            }
        }
        if self.value.as_ref().is_some_and(|(k, _)| *k == key) {
            // Current: drop any job still chasing an older key.
            self.job = None;
            return self.value.as_ref().map(|(_, v)| v);
        }
        if !matches!(&self.job, Some(j) if *j.key() == key) {
            self.job = Some(Job::spawn(key, Arc::clone(session), repaint, f));
        }
        None
    }

    /// The last value regardless of key (for views that prefer stale data
    /// over nothing while recomputing).
    pub fn last(&self) -> Option<&T> {
        self.value.as_ref().map(|(_, v)| v)
    }

    pub fn pending(&self) -> bool {
        self.job.is_some()
    }

    /// Cancel the running job (before a curation write) and forget the
    /// value so the next `ensure` recomputes.
    pub fn invalidate(&mut self) {
        self.job = None;
        self.value = None;
    }

    /// Cancel the running job but keep the value.
    pub fn cancel(&mut self) {
        self.job = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn wait_until(mut f: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !f() {
            assert!(Instant::now() < deadline, "timed out");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn memo_delivers_result_and_discards_stale_key() {
        let session = Arc::new(RwLock::new(10u32));
        let mut memo: Memo<u32, u32> = Memo::new();
        // First ask: spawns, nothing yet.
        assert!(memo.ensure(1, &session, || {}, |s, _| *s * 2).is_none());
        assert!(memo.pending());
        wait_until(|| memo.ensure(1, &session, || {}, |s, _| *s * 2).is_some());
        assert_eq!(memo.last(), Some(&20));
        // Different key: the old value is not reported as current.
        assert!(memo.ensure(2, &session, || {}, |s, _| *s * 3).is_none());
        wait_until(|| memo.ensure(2, &session, || {}, |s, _| *s * 3) == Some(&30));
    }

    #[test]
    fn cancel_unblocks_the_writer_and_drops_the_result() {
        let session = Arc::new(RwLock::new(0u32));
        let mut memo: Memo<u32, u32> = Memo::new();
        memo.ensure(
            1,
            &session,
            || {},
            |_, cancel| {
                let start = Instant::now();
                while !cancel.load(Ordering::Relaxed) && start.elapsed() < Duration::from_secs(5) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                42
            },
        );
        // The job holds the read lock; cancelling lets a writer in.
        memo.cancel();
        let t = Instant::now();
        *write(&session) = 5;
        assert!(t.elapsed() < Duration::from_secs(2), "writer was blocked");
        assert!(!memo.pending());
        assert!(memo.last().is_none(), "cancelled result must not land");
    }
}
