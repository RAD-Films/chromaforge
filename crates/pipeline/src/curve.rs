//! The parametric tone curve model shared by the pipeline renderer and the UI curve editor.
//!
//! Lightroom-style tone curve parametric model: four regions (shadows, darks, lights, highlights)
//! separated by three split points. Up to process V7 each region contributes a cosine-windowed bump,
//! summed and forced monotone. From V9 each region bends the curve as Lightroom Classic's does,
//! read off tables measured on a grey wedge of our own ([`tables`]), and the region shifts are
//! summed and forced monotone.

mod tables;

use std::sync::OnceLock;

use lcc_color::spline::{Lut1, MonotoneCurve};
use lcc_develop::{Process, ToneCurve};

/// The four shape controls of the parametric curve, in graph order: region 0 is the darkest.
pub const PARAM_AMOUNTS: [&str; 4] = ["curve.shadows", "curve.darks", "curve.lights", "curve.highlights"];

/// Resolution the parametric curve is sampled at, matching the renderer's LUT.
pub const PARAM_LUT: usize = 1024;

/// How the region sliders bend the curve.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Shape {
    /// V1–V7: a cosine bump per region, `0.22 · amount` high at its centre.
    Bumps,
    /// V9 on: Lightroom Classic's measured region curves.
    Measured,
}

/// The parametric (region) curve of a [`ToneCurve`]: the four region amounts bending the diagonal,
/// summed and then forced monotone.
///
/// This is the renderer's curve, not an approximation of it — both `curve_luts` in the pipeline
/// and the curve editor in the UI use this struct so they can never disagree.
#[derive(Clone, Debug)]
pub struct ParametricCurve {
    /// The curve, sampled and already forced monotone.
    lut: [f32; PARAM_LUT],
    /// The three region boundaries as fractions of the graph (0..1).
    pub splits: [f64; 3],
    /// The four region amounts, in graph order (0 = shadows, 3 = highlights).
    pub amounts: [f64; 4],
    shape: Shape,
}

/// The measured region curves as monotone splines: `[region][amount]`, amounts −100, −50, +50, +100.
fn measured() -> &'static [[MonotoneCurve; 4]; 4] {
    static CURVES: OnceLock<[[MonotoneCurve; 4]; 4]> = OnceLock::new();
    CURVES.get_or_init(|| {
        std::array::from_fn(|r| {
            std::array::from_fn(|a| {
                let ys = tables::Y.get(r).and_then(|t| t.get(a));
                let pts: Vec<(f64, f64)> =
                    tables::X.iter().enumerate().map(|(i, &x)| (x as f64, ys.and_then(|y| y.get(i)).copied().unwrap_or(x) as f64)).collect();
                MonotoneCurve::new(&pts)
            })
        })
    })
}

/// Region `region`'s measured shift of the (warped) input `x` at `amount` (−100..100): straight
/// between the amounts measured (0, ±50, ±100).
fn measured_shift(region: usize, amount: f64, x: f64) -> f64 {
    let Some(curves) = measured().get(region) else { return 0.0 };
    let d = |k: usize| curves.get(k).map_or(0.0, |c| c.eval(x) - x);
    let a = amount.clamp(-100.0, 100.0);
    match a {
        a if a >= 50.0 => d(2) + (a - 50.0) / 50.0 * (d(3) - d(2)),
        a if a >= 0.0 => a / 50.0 * d(2),
        a if a >= -50.0 => -a / 50.0 * d(1),
        a => d(1) + (-a - 50.0) / 50.0 * (d(0) - d(1)),
    }
}

impl ParametricCurve {
    /// The V1–V7 curve (cosine bumps); [`Self::for_process`] picks the photo's own.
    pub fn new(c: &ToneCurve) -> ParametricCurve {
        ParametricCurve::for_process(c, Process::V7)
    }

    /// The curve as process `p` draws it.
    pub fn for_process(c: &ToneCurve, p: Process) -> ParametricCurve {
        let s0 = if c.split_shadows.is_nan() { 0.25 } else { (c.split_shadows / 100.0).clamp(0.0, 1.0) };
        let s1 = if c.split_mid.is_nan() { 0.50_f64.max(s0) } else { (c.split_mid / 100.0).clamp(s0, 1.0) };
        let s2 = if c.split_highlights.is_nan() { 0.75_f64.max(s1) } else { (c.split_highlights / 100.0).clamp(s1, 1.0) };
        let clean_amt = |a: f64| if a.is_nan() { 0.0 } else { a.clamp(-100.0, 100.0) };
        let amounts = [clean_amt(c.shadows), clean_amt(c.darks), clean_amt(c.lights), clean_amt(c.highlights)];
        if p < Process::V9 {
            return ParametricCurve::bumps(c);
        }
        let shape = Shape::Measured;
        let mut m = ParametricCurve { lut: [0.0; PARAM_LUT], splits: [s0, s1, s2], amounts, shape };
        let mut prev = 0.0_f32;
        for i in 0..PARAM_LUT {
            let x = i as f64 / (PARAM_LUT - 1) as f64;
            let y = (m.raw_level(x, None).clamp(0.0, 1.0) as f32).max(prev);
            prev = y;
            if let Some(e) = m.lut.get_mut(i) {
                *e = y;
            }
        }
        m
    }

    /// The V1–V8 curve, computed exactly as those processes always have (in `f32`), so their renders never change.
    fn bumps(c: &ToneCurve) -> ParametricCurve {
        let s0 = if c.split_shadows.is_nan() { 0.25 } else { (c.split_shadows / 100.0).clamp(0.0, 1.0) as f32 };
        let s1 = if c.split_mid.is_nan() { 0.50_f32.max(s0) } else { (c.split_mid / 100.0).clamp(s0 as f64, 1.0) as f32 };
        let s2 = if c.split_highlights.is_nan() { 0.75_f32.max(s1) } else { (c.split_highlights / 100.0).clamp(s1 as f64, 1.0) as f32 };
        let clean_amt = |a: f64| if a.is_nan() { 0.0 } else { a.clamp(-100.0, 100.0) };
        let (a0, a1, a2, a3) = (clean_amt(c.shadows), clean_amt(c.darks), clean_amt(c.lights), clean_amt(c.highlights));
        let splits = [s0 as f64, s1 as f64, s2 as f64];
        let amounts = [a0, a1, a2, a3];
        let regions = [(0.0, s0, a0), (s0, s1, a1), (s1, s2, a2), (s2, 1.0, a3)];
        let mut lut = [0.0_f32; PARAM_LUT];
        let mut prev = 0.0_f32;
        for (i, entry) in lut.iter_mut().enumerate() {
            let x = i as f32 / (PARAM_LUT - 1) as f32;
            let mut d = 0.0_f32;
            for (a, b, amt) in regions {
                if amt == 0.0 {
                    continue;
                }
                let (ctr, half) = ((a + b) / 2.0, (b - a) * 0.75 + 0.05);
                if half <= f32::EPSILON {
                    continue;
                }
                let t = ((x - ctr) / half).clamp(-1.0, 1.0);
                d += (amt / 100.0) as f32 * 0.22 * (0.5 + 0.5 * (t * std::f32::consts::PI).cos());
            }
            let y = (x + d * 4.0 * x * (1.0 - x)).clamp(0.0, 1.0);
            prev = prev.max(y);
            *entry = prev;
        }
        ParametricCurve { lut, splits, amounts, shape: Shape::Bumps }
    }

    /// The splits' knots on the input axis, and where the measured tables have them (the default splits).
    fn warp_knots(&self) -> ([f64; 5], [f64; 5]) {
        ([0.0, self.split_x(0), self.split_x(1), self.split_x(2), 1.0], [0.0, 0.25, 0.5, 0.75, 1.0])
    }

    /// Piecewise-straight map taking knots `from` to knots `to` (a segment of no width is skipped).
    fn warp(x: f64, from: &[f64; 5], to: &[f64; 5]) -> f64 {
        for i in 0..4 {
            let (Some(&a), Some(&b), Some(&c), Some(&d)) = (from.get(i), from.get(i + 1), to.get(i), to.get(i + 1)) else { break };
            if x <= b && b - a > 1e-9 {
                return c + (x - a).max(0.0) / (b - a) * (d - c);
            }
        }
        to.get(4).copied().unwrap_or(1.0)
    }

    /// The curve at `x` before the monotone pass and the clamp, with region `replace.0` set to
    /// `replace.1` instead of its slider.
    fn raw_level(&self, x: f64, replace: Option<(usize, f64)>) -> f64 {
        let x = if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) };
        let amount = |i: usize| match replace {
            Some((r, a)) if r == i => a,
            _ => self.amounts.get(i).copied().unwrap_or(0.0),
        };
        match self.shape {
            Shape::Bumps => {
                let mut d = 0.0;
                for i in 0..4 {
                    let amt = amount(i);
                    if amt != 0.0 {
                        d += (amt / 100.0) * 0.22 * self.region_window_at(i, x);
                    }
                }
                x + d * 4.0 * x * (1.0 - x)
            }
            Shape::Measured => {
                let (user, table) = self.warp_knots();
                let w = Self::warp(x, &user, &table);
                let mut y = w;
                for i in 0..4 {
                    let amt = amount(i);
                    if amt != 0.0 {
                        y += measured_shift(i, amt, w);
                    }
                }
                Self::warp(y.clamp(0.0, 1.0), &table, &user)
            }
        }
    }

    /// Convert the evaluated parametric curve into a 1024-entry [`Lut1`].
    pub fn to_lut(&self) -> Lut1 {
        Lut1 { v: self.lut.to_vec() }
    }

    /// The curve at `x` (0 = black, 1 = white), read off the table (so it never overshoots the ends).
    pub fn at(&self, x: f64) -> f64 {
        let x = if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) as f32 };
        let at = x * (PARAM_LUT - 1) as f32;
        let i = at.floor().clamp(0.0, (PARAM_LUT - 1) as f32) as usize;
        let cur = self.lut.get(i).copied().unwrap_or(0.0);
        let next = self.lut.get(i + 1).copied().unwrap_or(cur);
        let f = at - i as f32;
        (cur + (next - cur) * f) as f64
    }

    /// Where region `region`'s handle sits on the input axis: the middle of its own window, so a
    /// handle never has to cross a split to be grabbed.
    pub fn region_x(&self, region: usize) -> f64 {
        let (a, b) = self.region_window(region);
        (a + b) / 2.0
    }

    /// The window (two splits, as fractions of the graph) region `region` owns.
    pub fn region_window(&self, region: usize) -> (f64, f64) {
        let s0 = self.split_x(0);
        let s1 = self.split_x(1);
        let s2 = self.split_x(2);
        match region {
            0 => (0.0, s0),
            1 => (s0, s1),
            2 => (s1, s2),
            _ => (s2, 1.0),
        }
    }

    /// The half-width of region `region`'s own bump, as fraction of the graph.
    pub fn region_half(&self, region: usize) -> f64 {
        let (a, b) = self.region_window(region);
        (b - a) * 0.75 + 0.05
    }

    /// The cosine window region `region` applies at `x` (1 at its centre, 0 at the window edges).
    pub fn region_window_at(&self, region: usize, x: f64) -> f64 {
        let ctr = self.region_x(region);
        let half = self.region_half(region);
        if half <= f64::EPSILON {
            return 0.0;
        }
        let t = ((x - ctr) / half).clamp(-1.0, 1.0);
        0.5 + 0.5 * (t * std::f64::consts::PI).cos()
    }

    /// The region that owns graph position `x`: the one whose own window contains it. The four
    /// windows tile 0..1, so this is always defined and a drag anywhere on the curve moves exactly
    /// one slider — the region under the pointer.
    pub fn region_at(&self, x: f64) -> usize {
        let x = if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) };
        let s0 = self.split_x(0);
        let s1 = self.split_x(1);
        let s2 = self.split_x(2);
        if x <= s0 {
            0
        } else if x <= s1 {
            1
        } else if x <= s2 {
            2
        } else {
            3
        }
    }

    /// How much output level region `region`'s bump gains at `x` per unit of its slider, under V1–V7: the
    /// bump is linear in the amount, so this is a constant for a given `x`. A V1–V7 drag converts the
    /// pointer's vertical motion into a slider change through it ([`Self::amount_after_drag`]).
    pub fn level_per_unit(&self, region: usize, x: f64) -> f64 {
        let x = x.clamp(0.0, 1.0);
        0.22 / 100.0 * self.region_window_at(region, x) * 4.0 * x * (1.0 - x)
    }

    /// The baseline level at `x` from the identity diagonal plus the contributions of all regions
    /// *except* `region`. Used when inverting the model for `region` while holding the other three
    /// regions fixed.
    pub fn base_level_without(&self, region: usize, x: f64) -> f64 {
        let x = x.clamp(0.0, 1.0);
        let mut d = 0.0;
        for i in 0..4 {
            if i == region {
                continue;
            }
            let amt = self.amounts.get(i).copied().unwrap_or(0.0);
            if amt == 0.0 {
                continue;
            }
            d += (amt / 100.0) * 0.22 * self.region_window_at(i, x);
        }
        x + d * 4.0 * x * (1.0 - x)
    }

    /// The curve level region `region` sits at when its slider holds `amount`, at graph position `x`,
    /// holding the other regions fixed at their current settings.
    pub fn handle_level_at(&self, region: usize, amount: f64, x: f64) -> f64 {
        self.raw_level(x, Some((region, amount))).clamp(0.0, 1.0)
    }

    /// The amount that puts region `region`'s curve at `level` at `x`, with the other three
    /// regions held fixed — the inverse of [`Self::handle_level_at`]. A level the region cannot
    /// reach saturates rather than jumping.
    pub fn amount_at_level(&self, region: usize, level: f64, x: f64) -> f64 {
        let current = self.amounts.get(region).copied().unwrap_or(0.0);
        let level = if level.is_nan() { return current } else { level.clamp(0.0, 1.0) };
        match self.shape {
            Shape::Bumps => {
                let per_unit = self.level_per_unit(region, x);
                if per_unit.abs() <= f64::EPSILON {
                    return current;
                }
                let base = self.base_level_without(region, x);
                ((level - base) / per_unit).clamp(-100.0, 100.0).round()
            }
            Shape::Measured => self.measured_amount(region, level, x, current),
        }
    }

    /// The V9 amount that puts region `region`'s curve at `level` at `x`. Where the curve is flat in the amount
    /// (pinned at black or white), several amounts give that level: `prefer` is kept when it is one of them,
    /// so a drag never jumps the slider along a stretch the eye can't see move.
    fn measured_amount(&self, region: usize, level: f64, x: f64, prefer: f64) -> f64 {
        let f = |a: f64| self.handle_level_at(region, a, x);
        let level = level.clamp(0.0, 1.0);
        if (f(prefer) - level).abs() < 1e-6 || f(100.0) - f(-100.0) <= 1e-9 {
            return prefer;
        }
        // the measured level rises with the amount: halve the amount range
        let (mut lo, mut hi) = (-100.0_f64, 100.0_f64);
        for _ in 0..40 {
            let mid = (lo + hi) / 2.0;
            if f(mid) < level {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        ((lo + hi) / 2.0).round()
    }

    /// The amount a drag gives region `region`: it was grabbed at `x` while the slider held `initial`,
    /// and the pointer has since moved the region curve `delta_level` up (down, if negative).
    pub fn amount_after_drag(&self, region: usize, initial: f64, delta_level: f64, x: f64) -> f64 {
        match self.shape {
            Shape::Bumps => {
                let per_unit = self.level_per_unit(region, x);
                if per_unit.abs() > f64::EPSILON { (initial + delta_level / per_unit).clamp(-100.0, 100.0).round() } else { initial }
            }
            Shape::Measured => self.measured_amount(region, self.handle_level_at(region, initial, x) + delta_level, x, initial),
        }
    }

    /// The graph position split `split` (0..=2) sits at, 0..1.
    pub fn split_x(&self, split: usize) -> f64 {
        self.splits.get(split).copied().unwrap_or(0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_finite_or_inverted_splits_do_not_panic_or_poison_curve() {
        let c = ToneCurve {
            split_shadows: 80.0,
            split_mid: 20.0,
            split_highlights: f64::NAN,
            shadows: f64::NAN,
            lights: f64::INFINITY,
            ..Default::default()
        };
        let model = ParametricCurve::new(&c);
        assert!(model.at(0.5).is_finite());
        assert!(model.at(f64::NAN).is_finite());
        assert!(model.region_at(f64::NAN) < 4);
        assert!((model.splits[0] - 0.8).abs() < 1e-6);
        assert!((model.splits[1] - 0.8).abs() < 1e-6);
        assert!((model.splits[2] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn horizontal_inversion_at_x_0_51_with_darks_60_preserves_zero_lights() {
        let c = ToneCurve { darks: 60.0, lights: 0.0, ..Default::default() };
        let model = ParametricCurve::new(&c);
        assert_eq!(model.region_at(0.51), 2, "x=0.51 is inside the Lights region");
        let level = model.at(0.51);
        assert!((level - 0.5619).abs() < 0.001, "extracted model at 0.51 is ~0.5619: {level}");
        // With other regions held fixed, inverting at level ~0.5619 must return Lights = 0, NOT 45!
        let lights = model.amount_at_level(2, level, 0.51);
        assert_eq!(lights, 0.0, "zero vertical delta preserves Lights=0: {lights}");
    }

    #[test]
    fn a_curve_level_and_its_amount_are_inverses_with_other_regions_present() {
        let c = ToneCurve { shadows: 20.0, darks: -30.0, lights: 40.0, highlights: -10.0, ..Default::default() };
        let model = ParametricCurve::new(&c);
        for region in 0..4 {
            let centre = model.region_x(region);
            let (_, upper) = model.region_window(region);
            for x in [centre, centre + (upper - centre) * 0.5] {
                for amount in [-100.0, -40.0, 0.0, 30.0, 100.0] {
                    let level = model.handle_level_at(region, amount, x);
                    let back = model.amount_at_level(region, level, x);
                    assert!((back - amount).abs() <= 1.0, "region {region} at {x}: {amount} -> {level} -> {back}");
                }
            }
        }
    }

    /// The longest run of graph positions in `lo..hi` over which the curve does not rise.
    fn longest_flat_run(m: &ParametricCurve, lo: f64, hi: f64) -> usize {
        let (mut run, mut best) = (0, 0);
        let n = 512;
        let mut prev = m.at(lo);
        for i in 1..=n {
            let y = m.at(lo + (hi - lo) * i as f64 / n as f64);
            run = if y - prev < 1e-5 { run + 1 } else { 0 };
            best = best.max(run);
            prev = y;
        }
        best
    }

    /// Darks and Lights at full strength bend the V1–V7 curve so far that it folds back, and the
    /// monotone pass turns that into a flat band: a range of mid-tones that all come out the same level
    /// (posterised skies and skin). Lightroom's curve keeps rising through the mid-tones, and so does V9's.
    #[test]
    fn v8_darks_and_lights_never_flatten_the_mid_tones() {
        let mut old_flat = 0;
        for (darks, lights) in [(-100.0, 0.0), (100.0, 0.0), (0.0, -100.0), (0.0, 100.0), (-100.0, 100.0), (100.0, -100.0)] {
            let c = ToneCurve { darks, lights, ..Default::default() };
            old_flat = old_flat.max(longest_flat_run(&ParametricCurve::for_process(&c, Process::V7), 0.2, 0.8));
            let new = longest_flat_run(&ParametricCurve::for_process(&c, Process::V9), 0.2, 0.8);
            assert!(new <= 2, "Darks {darks}, Lights {lights}: V9 is flat over {new} of 512 steps");
        }
        assert!(old_flat > 20, "the old curve's flat band, or this proves nothing: {old_flat}");
    }

    /// V9's region curves go through Lightroom Classic's measured levels on the wedge, at every patch and
    /// every amount measured, and halfway between 0 and ±50 they are halfway there.
    #[test]
    fn v8_regions_follow_the_wedge() {
        for r in 0..4 {
            for (k, amount) in [-100.0, -50.0, 50.0, 100.0].into_iter().enumerate() {
                let mut c = ToneCurve::default();
                match r {
                    0 => c.shadows = amount,
                    1 => c.darks = amount,
                    2 => c.lights = amount,
                    _ => c.highlights = amount,
                }
                let m = ParametricCurve::for_process(&c, Process::V9);
                let mut half = c.clone();
                match r {
                    0 => half.shadows = amount / 2.0,
                    1 => half.darks = amount / 2.0,
                    2 => half.lights = amount / 2.0,
                    _ => half.highlights = amount / 2.0,
                }
                let h = ParametricCurve::for_process(&half, Process::V9);
                for (i, &x) in tables::X.iter().enumerate() {
                    let want = tables::Y[r][k][i] as f64;
                    let got = m.at(x as f64);
                    assert!((got - want).abs() < 0.002, "region {r} at {amount}, x {x}: {got} vs Lightroom {want}");
                    if k == 1 || k == 2 {
                        let mid = (x as f64 + want) / 2.0;
                        assert!((h.at(x as f64) - mid).abs() < 0.002, "region {r} at {}, x {x}", amount / 2.0);
                    }
                }
            }
        }
    }

    /// The flat curve stays the diagonal, and V9 handles the same hostile settings as V7.
    #[test]
    fn v8_flat_and_hostile_curves() {
        let m = ParametricCurve::for_process(&ToneCurve::default(), Process::V9);
        for i in 0..=20 {
            let x = i as f64 / 20.0;
            assert!((m.at(x) - x).abs() < 0.002, "no sliders, no change at {x}: {}", m.at(x));
        }
        let c = ToneCurve {
            split_shadows: 80.0,
            split_mid: 20.0,
            split_highlights: f64::NAN,
            shadows: f64::NAN,
            darks: -100.0,
            lights: f64::INFINITY,
            highlights: f64::NEG_INFINITY,
            ..Default::default()
        };
        let m = ParametricCurve::for_process(&c, Process::V9);
        let mut prev = 0.0;
        for i in 0..=100 {
            let y = m.at(i as f64 / 100.0);
            assert!(y.is_finite() && (0.0..=1.0).contains(&y) && y >= prev, "at {i}: {y}");
            prev = y;
        }
        for region in 0..4 {
            assert!(m.amount_at_level(region, f64::NAN, 0.5).is_finite());
            assert!(m.amount_after_drag(region, 0.0, f64::INFINITY, 0.5).abs() <= 100.0);
        }
    }

    /// Moved splits move the regions: the tables are read with the splits taken to where they were
    /// measured (25/50/75), so a Darks bend centred between 15 and 50 peaks lower than at the defaults.
    #[test]
    fn v8_splits_move_the_regions() {
        let wide = ParametricCurve::for_process(
            &ToneCurve { darks: 50.0, split_shadows: 15.0, split_mid: 50.0, split_highlights: 85.0, ..Default::default() },
            Process::V9,
        );
        let plain = ParametricCurve::for_process(&ToneCurve { darks: 50.0, ..Default::default() }, Process::V9);
        assert!(
            (wide.at(0.25) - 0.25) > (plain.at(0.25) - 0.25) + 0.005,
            "lower first split lifts 0.25 more: {} vs {}",
            wide.at(0.25),
            plain.at(0.25)
        );
        for x in [0.0, 1.0] {
            assert!((wide.at(x) - plain.at(x)).abs() < 0.01, "black and white stay put at {x}");
        }
    }

    /// Where V9's curve is pinned at white, a drag that has not moved up or down keeps the slider where it was
    /// (Highlights +100 grabbed near white used to drop to 36 on the first frame).
    #[test]
    fn v8_a_still_drag_keeps_the_slider() {
        let m = ParametricCurve::for_process(&ToneCurve { highlights: 100.0, ..Default::default() }, Process::V9);
        assert_eq!(m.amount_after_drag(3, 100.0, 0.0, 0.875), 100.0);
        assert_eq!(m.amount_at_level(3, m.at(0.875), 0.875), 100.0);
        assert!(m.amount_after_drag(3, 100.0, -0.05, 0.875) < 100.0, "a drag down still lowers it");
    }
}
