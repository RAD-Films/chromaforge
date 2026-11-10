//! The batch workers: rayon pools of their own for work made ahead of the user (an export's
//! photos, Build Previews, the background preparation, an import's previews), at the lowest normal
//! thread priority.
//!
//! The renders of the photo on screen fan their processor stages out over the shared rayon pool
//! (display conversion, resampling, grain, …). When the batch photos fanned out over the same
//! pool, the loupe's stages queued behind theirs: during an export a slider draft took 0.7–1.3 s
//! instead of 10–20 ms (stack samples: the render workers waiting in `display::finish_render`'s
//! parallel loops). A photo of batch work now runs whole on a batch worker ([`run`]), so its
//! parallel loops stay on the batch workers, and the operating system gives the shared pool's
//! threads the processor first.

use std::cell::{Cell, RefCell};
use std::sync::Arc;

thread_local! {
    /// The calling thread's own batch pool ([`run`]), built on its first batch photo.
    static LANE: RefCell<Option<Arc<rayon::ThreadPool>>> = const { RefCell::new(None) };
    /// This thread is a batch worker.
    static WORKER: Cell<bool> = const { Cell::new(false) };
}

/// Threads of each lane's pool: half the shared pool's (at least two). An export or a build runs
/// two to four lanes at once, all at the lowest priority, so together they still fill the machine
/// when the user is not editing.
fn lane_threads() -> usize {
    // (from a thread outside any pool: the shared pool's size)
    (rayon::current_num_threads() / 2).max(2)
}

fn build() -> Option<Arc<rayon::ThreadPool>> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(lane_threads())
        .thread_name(|i| format!("lc-batch-{i}"))
        .start_handler(|_| {
            let _ = WORKER.try_with(|w| w.set(true));
            lcc_sysmem::lowest_thread_priority();
        })
        .build()
        .map_err(|e| log::warn!("batch workers: {e} (batch work shares the render workers)"))
        .ok()
        .map(Arc::new)
}

/// Run `f`, one photo of batch work, on a batch worker and wait for it: its parallel loops run on
/// the batch workers, not the shared pool. Each calling thread (an export's or a build's lane) has
/// a pool of its own, so one photo runs there at a time and never starts inside another photo's
/// work on a worker that waits in a parallel loop (it would see that photo's thread routing). Call
/// it where the photo's work starts: `f` runs on another thread, which the calling thread's GPU
/// routing ([`lcc_gpu::ThreadState`]) and [`crate::memory::in_background`] are carried over to
/// (its renders count on the calling thread). On a batch worker already, or when no pool can be
/// built, `f` runs here. A panic in `f` continues on the calling thread.
pub fn run<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    if on_batch_worker() {
        return f();
    }
    let pool = LANE
        .try_with(|lane| {
            let mut lane = lane.borrow_mut();
            if lane.is_none() {
                *lane = build();
            }
            lane.clone()
        })
        .ok()
        .flatten();
    let Some(pool) = pool else { return f() };
    let background = crate::memory::is_background();
    let gpu = lcc_gpu::ThreadState::take();
    let (r, carried) = pool.install(move || gpu.run(|| if background { crate::memory::in_background(f) } else { f() }));
    carried.give_back();
    r
}

/// Whether the calling thread is a batch worker.
pub fn on_batch_worker() -> bool {
    WORKER.try_with(Cell::get).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use rayon::prelude::*;
    use std::sync::mpsc;
    use std::time::Duration;

    /// Batch work that keeps every batch worker busy leaves the shared pool free: a render's
    /// parallel loop finishes at once (before, the batch photos' loops held the shared pool's
    /// threads and the loupe's loop waited behind them).
    #[test]
    fn busy_batch_workers_leave_the_render_workers_free() {
        let (release, wait) = mpsc::channel::<()>();
        let wait = std::sync::Mutex::new(wait);
        let (started, busy) = mpsc::channel::<()>();
        let batch = std::thread::spawn(move || {
            super::run(move || {
                assert!(super::on_batch_worker());
                // one blocked task per worker of the pool this runs in
                (0..rayon::current_num_threads()).into_par_iter().for_each(|_| {
                    let _ = started.send(());
                    let _ = wait.lock().unwrap_or_else(std::sync::PoisonError::into_inner).recv_timeout(Duration::from_secs(20));
                });
            })
        });
        // the batch work has started on its pool
        busy.recv_timeout(Duration::from_secs(10)).expect("batch work started");
        std::thread::sleep(Duration::from_millis(100));
        let (done, finished) = mpsc::channel();
        std::thread::spawn(move || {
            let sum: u64 = (0..10_000u64).into_par_iter().sum();
            let _ = done.send(sum);
        });
        let got = finished.recv_timeout(Duration::from_secs(5));
        for _ in 0..1024 {
            let _ = release.send(());
        }
        batch.join().unwrap();
        assert_eq!(got.expect("the render workers waited for the batch work"), 49_995_000);
        assert!(!super::on_batch_worker());
    }

    /// A batch photo that starts more batch work runs it where it is (no second pool, no wait).
    #[test]
    fn batch_work_inside_batch_work_runs_in_place() {
        let outer = super::run(|| {
            let here = std::thread::current().id();
            (super::on_batch_worker(), super::run(move || std::thread::current().id() == here))
        });
        assert_eq!(outer, (true, true));
    }

    #[test]
    fn the_background_flag_goes_with_the_work() {
        assert!(!super::run(crate::memory::is_background));
        assert!(crate::memory::in_background(|| super::run(crate::memory::is_background)));
        assert_eq!(super::run(|| 7), 7);
    }
}
