//! Background jobs: run a closure against the shared session on a worker
//! thread, hand the result back to the UI thread, and drop it if the
//! inputs it was computed for are no longer what the UI wants.
//!
//! The session lives behind an `Arc<RwLock<_>>`. Session jobs hold a read lock
//! for the duration of one job; a curation command takes the write lock
//! and therefore waits for in-flight jobs — the UI cancels them first and
//! jobs poll the cancel flag between steps, so that wait stays short.
//! Geometry/hit-test/render previews use `spawn_owned` / `ensure_owned` over
//! immutable snapshots instead, so they never hold this lock during curation.
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
    /// Work on an owned immutable snapshot, without holding the session's
    /// read lock. Preview work must never delay a curation write.
    pub fn spawn_owned<F>(key: K, repaint: impl Fn() + Send + 'static, f: F) -> Self
    where
        T: Send + 'static,
        F: FnOnce(&AtomicBool) -> T + Send + 'static,
    {
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = channel();
        let flag = Arc::clone(&cancel);
        std::thread::Builder::new()
            .name("sorrel-preview".into())
            .spawn(move || {
                if flag.load(Ordering::Relaxed) {
                    return;
                }
                let out = f(&flag);
                if !flag.load(Ordering::Relaxed) && tx.send(out).is_ok() {
                    repaint();
                }
            })
            .expect("spawn sorrel preview thread");
        Self { key, rx, cancel }
    }

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
    /// One superseded value, kept only with [`Memo::with_history`]: Undo
    /// restores an earlier key and must not recompute it from scratch.
    previous: Option<(K, T)>,
    history: bool,
    job: Option<Job<K, T>>,
}

impl<K, T> Default for Memo<K, T> {
    fn default() -> Self {
        Self {
            value: None,
            previous: None,
            history: false,
            job: None,
        }
    }
}

impl<K: PartialEq + Clone, T> Memo<K, T> {
    /// Keep the value a newer one supersedes, so flipping back to it is
    /// instant. Opt-in because a value can be large.
    pub fn with_history() -> Self {
        Self {
            history: true,
            ..Self::default()
        }
    }

    /// Take a finished job's value; with history, keep the value it replaces.
    fn settle_job(&mut self) {
        let Some(job) = &self.job else { return };
        let Some(value) = job.try_take() else { return };
        let key = job.key().clone();
        if let Some(old) = self.value.take() {
            if self.history && old.0 != key {
                self.previous = Some(old);
            }
        }
        self.value = Some((key, value));
        self.job = None;
    }

    /// Make a kept superseded value current again (Undo), cancelling any job
    /// chasing the key we just left.
    fn recall_previous(&mut self, key: &K) -> bool {
        if self.previous.as_ref().is_some_and(|(k, _)| k == key) {
            std::mem::swap(&mut self.value, &mut self.previous);
            self.job = None;
            true
        } else {
            false
        }
    }

    /// As `ensure`, but the closure owns its inputs and takes no session lock.
    pub fn ensure_owned<F>(
        &mut self,
        key: K,
        repaint: impl Fn() + Send + 'static,
        f: F,
    ) -> Option<&T>
    where
        T: Send + 'static,
        F: FnOnce(&AtomicBool) -> T + Send + 'static,
    {
        self.settle_job();
        if self.value.as_ref().is_some_and(|(k, _)| *k == key) {
            self.job = None;
            return self.value.as_ref().map(|(_, value)| value);
        }
        if self.recall_previous(&key) {
            return self.value.as_ref().map(|(_, value)| value);
        }
        if !matches!(&self.job, Some(job) if *job.key() == key) {
            self.job = Some(Job::spawn_owned(key, repaint, f));
        }
        None
    }

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
        self.settle_job();
        if self.value.as_ref().is_some_and(|(k, _)| *k == key) {
            // Current: drop any job still chasing an older key.
            self.job = None;
            return self.value.as_ref().map(|(_, v)| v);
        }
        if self.recall_previous(&key) {
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
    /// values so the next `ensure` recomputes.
    pub fn invalidate(&mut self) {
        self.job = None;
        self.value = None;
        self.previous = None;
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
    fn memo_history_reuses_the_previous_value_without_a_job() {
        let session = Arc::new(RwLock::new(10u32));
        let mut memo: Memo<u32, u32> = Memo::with_history();
        assert!(memo.ensure(1, &session, || {}, |s, _| *s).is_none());
        wait_until(|| memo.ensure(1, &session, || {}, |s, _| *s).is_some());
        assert!(memo.ensure(2, &session, || {}, |s, _| *s * 2).is_none());
        wait_until(|| memo.ensure(2, &session, || {}, |s, _| *s * 2) == Some(&20));
        // Back to the superseded key (Undo): the value is kept, no job.
        let ran = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&ran);
        let back = memo
            .ensure(
                1,
                &session,
                || {},
                move |s, _| {
                    flag.store(true, Ordering::Relaxed);
                    *s
                },
            )
            .copied();
        assert_eq!(back, Some(10));
        assert!(!ran.load(Ordering::Relaxed), "must not recompute");
        assert!(!memo.pending());
        // And forward again is still kept too.
        assert_eq!(memo.ensure(2, &session, || {}, |s, _| *s * 2), Some(&20));
    }

    #[test]
    fn a_worker_finishing_after_cancel_cannot_publish_its_old_result() {
        let session = Arc::new(RwLock::new(()));
        let mut memo: Memo<u32, u32> = Memo::new();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let (finished_tx, finished_rx) = std::sync::mpsc::channel();
        memo.ensure(
            1,
            &session,
            || {},
            move |_, cancel| {
                started_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                finished_tx.send(cancel.load(Ordering::Relaxed)).unwrap();
                42
            },
        );
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        memo.cancel();
        release_tx.send(()).unwrap();
        assert!(finished_rx.recv_timeout(Duration::from_secs(5)).unwrap());
        assert!(!memo.pending());
        assert!(
            memo.last().is_none(),
            "the abandoned cut result stays discarded"
        );
    }

    #[test]
    fn owned_preview_does_not_hold_the_session_and_cannot_publish_after_cancel() {
        let session = Arc::new(RwLock::new(10u32));
        let snapshot = *read(&session);
        let mut memo: Memo<u32, u32> = Memo::new();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let (finished_tx, finished_rx) = std::sync::mpsc::channel();
        memo.ensure_owned(
            1,
            || {},
            move |cancel| {
                started_tx.send(()).unwrap();
                release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                finished_tx.send(cancel.load(Ordering::Relaxed)).unwrap();
                snapshot
            },
        );
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        *session
            .try_write()
            .expect("preview must not lock out curation") = 20;
        memo.cancel();
        release_tx.send(()).unwrap();
        assert!(finished_rx.recv_timeout(Duration::from_secs(5)).unwrap());
        assert!(memo.last().is_none());
        wait_until(|| memo.ensure_owned(2, || {}, |_| 20).is_some());
        assert_eq!(memo.last(), Some(&20));
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
