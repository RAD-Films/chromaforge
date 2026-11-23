//! How long grid and filmstrip cells on screen wait for their first pixels (notes: thumbnails
//! "pop in like popcorn"). Each cell drawn on screen says whether it had a picture; a cell first
//! seen without one is timed until it has one. `ui.inspect` → `perf.thumbFirst`; `ui.thumbFirst`
//! `{reset: true}` starts counting afresh (ported from OneSlips, where the culling measurements of
//! PERF-CULL-TIERS used it).

use std::collections::HashMap;

use lcc_catalog::PhotoId;

/// Samples kept (the latest ones).
const KEEP: usize = 4096;

#[derive(Debug, Default)]
pub struct ThumbFirst {
    /// Cells on screen still without a picture: since when (ms).
    waiting: HashMap<PhotoId, f64>,
    /// Photos already timed (or that had a picture when first seen).
    done: std::collections::HashSet<PhotoId>,
    /// Waits (ms) of the cells timed, latest last; 0 for a cell that had a picture at once.
    samples: Vec<f64>,
    /// The frame being counted, and its empty cells on screen so far.
    frame: u64,
    empty_this_frame: usize,
    /// Empty cells summed over every frame (a cell empty for 10 frames counts 10).
    empty_cell_frames: u64,
}

/// What [`ThumbFirst`] measured since its last reset.
#[derive(Debug, Default, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ThumbFirstSummary {
    /// Cells seen on screen.
    pub seen: usize,
    /// Of those, cells still without a picture.
    pub waiting: usize,
    /// Cells on screen without a picture in the last frame drawn.
    pub empty_now: usize,
    pub empty_cell_frames: u64,
    /// Time to first pixels (ms) of the cells timed: median, 90th percentile, longest.
    pub p50: f64,
    pub p90: f64,
    pub max: f64,
}

impl ThumbFirst {
    /// A cell of photo `id` was drawn on screen in frame `frame` at `now` (ms), with a picture or not.
    pub fn note(&mut self, frame: u64, id: PhotoId, has_pixels: bool, now: f64) {
        if frame != self.frame {
            self.empty_this_frame = 0;
            self.frame = frame;
        }
        if !has_pixels {
            self.empty_this_frame += 1;
            self.empty_cell_frames += 1;
        }
        if self.done.contains(&id) {
            return;
        }
        match (has_pixels, self.waiting.get(&id).copied()) {
            (true, Some(since)) => {
                self.waiting.remove(&id);
                self.push((now - since).max(0.0));
                self.done.insert(id);
            }
            (true, None) => {
                self.push(0.0);
                self.done.insert(id);
            }
            (false, None) => {
                self.waiting.insert(id, now);
            }
            (false, Some(_)) => {}
        }
    }

    fn push(&mut self, ms: f64) {
        if self.samples.len() >= KEEP {
            self.samples.remove(0);
        }
        self.samples.push(if ms.is_finite() { ms } else { 0.0 });
    }

    /// Start counting afresh (a new view to time).
    pub fn reset(&mut self) {
        *self = Self { frame: self.frame, ..Self::default() };
    }

    pub fn summary(&self) -> ThumbFirstSummary {
        let mut v = self.samples.clone();
        v.sort_by(f64::total_cmp);
        let at = |q: f64| -> f64 {
            if v.is_empty() {
                return 0.0;
            }
            let i = ((v.len() - 1) as f64 * q).round() as usize;
            v.get(i.min(v.len() - 1)).copied().unwrap_or(0.0)
        };
        ThumbFirstSummary {
            seen: self.done.len() + self.waiting.len(),
            waiting: self.waiting.len(),
            // (asked between frames: the frame counted last is whole)
            empty_now: self.empty_this_frame,
            empty_cell_frames: self.empty_cell_frames,
            p50: at(0.5),
            p90: at(0.9),
            max: v.last().copied().unwrap_or(0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cells_are_timed_from_first_seen_empty_to_first_pixels() {
        let mut t = ThumbFirst::default();
        t.note(1, PhotoId(1), true, 0.0);
        t.note(1, PhotoId(2), false, 0.0);
        t.note(1, PhotoId(3), false, 0.0);
        t.note(2, PhotoId(2), false, 16.0);
        t.note(2, PhotoId(3), true, 16.0);
        t.note(3, PhotoId(2), true, 100.0);
        let s = t.summary();
        assert_eq!((s.seen, s.waiting, s.empty_cell_frames), (3, 0, 3));
        assert_eq!(s.max, 100.0);
        assert_eq!(s.p50, 16.0);
        t.reset();
        assert_eq!(t.summary(), ThumbFirstSummary::default());
        // junk times never poison the numbers
        t.note(9, PhotoId(4), false, f64::NAN);
        t.note(10, PhotoId(4), true, 5.0);
        assert!(t.summary().max.is_finite());
    }
}
