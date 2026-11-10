//! The global Highlights and Shadows of process V9: Lightroom Classic's response, one tone curve per picture.
//!
//! Measured on Lightroom Classic exports of our own procedural pictures (240 photo-like scenes with skies, shapes,
//! textures, night, high-key, fog, back-lit and portrait-like layouts; 16-bit sRGB TIFFs, OneSlips, Nov 2026):
//! on flat areas Lightroom's Shadows and Highlights act as one tone curve per picture, and the curve's shape follows
//! the picture's histogram (how far a tone sits below the picture's bright end, and how dark the picture is). The
//! curve here is that fit: a sum of piecewise-linear curves of `E − P` (a tone `E`, in EV over middle grey, against
//! a percentile `P` of the whole picture) whose knot values move linearly with a few histogram features. Fitted on
//! 170 scenes, chosen on a validation slice of them, and scored on 70 scenes it never saw: Shadows +100 is 6.5 8-bit
//! levels rms from Lightroom (the V7 response 9.8), Highlights −100 8.5 (V7 21.5). Guards keep pictures unlike the
//! fitted ones sane (see [`curve`]): their features are held to the fitted range, a far picture's curve is drawn back
//! towards the mean curve, and tones keep their order.
//!
//! The histogram is taken once per photo on a fixed-size render of the whole frame (white-balanced and exposed,
//! [`EDGE`] px on the long side), so the preview, a zoomed window and the export use the same curve. The curve
//! acts, like V1–V7's, on the edge-aware base layer of log luminance, so fine detail keeps its contrast.
//!
//! Lightroom's curve was measured on display-referred pictures (TIFF), where a tone is what is shown. A raw's scene
//! tones go through the camera's tone curve first, so for a raw the histogram and the curve are read in displayed
//! tones (the default rendering of the camera's curve), and a correction fitted on real raws is added to the curve:
//! Lightroom's raw response is not its TIFF one (on our Canon EOS R5 C raws, rounds 7 and 8, the picture-fitted
//! curve alone was 15–23 levels from Lightroom on held-out window scenes; on 7 held-out shots, rendered, the change
//! Highlights −100 makes went from 13.3 to 8.1 levels rms from Lightroom's, Shadows +100 from 12.1 to 12.8 but with
//! no more 3–5× over-lifts of dark frames, which reached 113 levels). In the curve's toe the scene shift is the one that moves
//! the displayed tone by the curve's amount ([`scene_shift`]); elsewhere the shift is given to the scene tone as it
//! is, which keeps the order of tones past the camera curve's white (Highlights then brings them into view, as
//! Lightroom's does).

use std::sync::{Arc, Mutex, PoisonError};

use lcc_develop::{DevelopSettings, Process};
use lcc_raster::Rgb32f;

use crate::SourceInfo;
use crate::geometry::Frame;
use crate::local::{frame_proxy, log_lum, white_balance};
use crate::tone::{GREY, ToneMap};

mod highlights;
mod raw;
mod tables;
mod whites;

/// Long edge of the render the histogram is taken from.
pub const EDGE: usize = 384;

/// The curves' knots: `E − P` (or `E`) from `KNOT_LO` EV in steps of `KNOT_STEP`.
const KNOT_LO: f32 = -20.0;
const KNOT_STEP: f32 = 0.5;
const KN: usize = 57;

/// Tones darker than this (EV over grey) count as this, as in the fit.
const EV_FLOOR: f32 = -16.0;

/// The per-picture table: EV from [`LIFT_LO`] in steps of [`LIFT_STEP`], [`LIFT_N`] entries.
pub const LIFT_LO: f32 = -20.0;
pub const LIFT_STEP: f32 = 1.0 / 32.0;
pub const LIFT_N: usize = 961;

/// The largest shift (EV) the table holds either way.
const MAX_SHIFT: f32 = 8.0;

/// How far (EV) the +100 Shadows curve may darken a tone, and the −100 Highlights curve brighten one.
const SIGN_SLACK: f32 = 0.5;

/// A histogram feature: a percentile relative to the 99th, the 99th itself, or the share of tones under `−k` EV.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Feat {
    P99,
    Pct(f32),
    Dark(f32),
}

/// What a block's curve is a function of: the tone less a percentile of the picture, or the tone itself.
#[derive(Clone, Copy, Debug)]
pub(crate) enum X {
    Pct(f32),
    Abs,
}

/// One piecewise-linear curve: its knot values `c0` at the fit's mean features, moved by `slopes[f]` per unit of
/// feature `f` away from its mean.
pub(crate) struct Block {
    x: X,
    c0: [f32; KN],
    slopes: &'static [[f32; KN]],
}

/// A fitted curve: its features (with their mean, and the range seen while fitting), the inverse covariance of the
/// fitting pictures' features and how far from them (`far.0`, Mahalanobis) a picture may be before its curve is
/// drawn back towards the mean curve (fully at `far.0 + far.1`), and its blocks.
pub(crate) struct Model {
    feats: &'static [Feat],
    fmean: &'static [f32],
    flo: &'static [f32],
    fhi: &'static [f32],
    icov: &'static [&'static [f32]],
    far: (f32, f32),
    blocks: &'static [Block],
}

/// The tones of a picture (EV over grey, sorted, at least one).
#[derive(Clone, Debug, PartialEq)]
pub struct Tones(Vec<f32>);

impl Tones {
    /// The finite tones of `ev` (each at least [`EV_FLOOR`]); `None` when none is finite.
    pub fn new(ev: impl IntoIterator<Item = f32>) -> Option<Tones> {
        let mut v: Vec<f32> = ev.into_iter().filter(|e| e.is_finite()).map(|e| e.max(EV_FLOOR)).collect();
        if v.is_empty() {
            return None;
        }
        v.sort_unstable_by(f32::total_cmp);
        Some(Tones(v))
    }

    /// The `q`th percentile (0–100), linearly interpolated between the sorted tones (numpy's default).
    pub fn percentile(&self, q: f32) -> f32 {
        let v = &self.0;
        let n = v.len();
        let pos = (q.clamp(0.0, 100.0) / 100.0) * (n.saturating_sub(1)) as f32;
        let i = (pos.floor() as usize).min(n.saturating_sub(1));
        let j = (i + 1).min(n.saturating_sub(1));
        let (a, b) = (v.get(i).copied().unwrap_or(0.0), v.get(j).copied().unwrap_or(0.0));
        a + (b - a) * (pos - i as f32)
    }

    /// The share of tones below `ev`.
    pub fn below(&self, ev: f32) -> f32 {
        self.0.partition_point(|&e| e < ev) as f32 / self.0.len().max(1) as f32
    }

    /// The same tones mapped through the increasing function `f`.
    fn mapped(&self, f: impl Fn(f32) -> f32) -> Tones {
        let mut v: Vec<f32> = self.0.iter().map(|&e| f(e)).map(|e| if e.is_finite() { e.max(EV_FLOOR) } else { EV_FLOOR }).collect();
        v.sort_unstable_by(f32::total_cmp);
        Tones(v)
    }
}

/// Linear interpolation of the knot values `k` at `x` (clamped to the knots, as in the fit).
fn knot_at(k: &[f32; KN], x: f32) -> f32 {
    let f = ((x - KNOT_LO) / KNOT_STEP).clamp(0.0, (KN - 1) as f32);
    let i = (f as usize).min(KN - 2);
    let t = f - i as f32;
    let (a, b) = (k.get(i).copied().unwrap_or(0.0), k.get(i + 1).copied().unwrap_or(0.0));
    a + (b - a) * t
}

/// One model's curve for one picture: per block, its offset and knot values.
struct Curve(Vec<(f32, [f32; KN])>);

impl Curve {
    /// The shift (EV) of tone `e` at the fitted amount.
    fn at(&self, e: f32) -> f32 {
        self.0.iter().map(|(off, k)| knot_at(k, e - off)).sum()
    }
}

/// `model`'s curve for a picture with tones `t`. The fit is linear in the features, so it is only trusted near the
/// pictures it was fitted on: each feature is held to the range those had, and a picture far from all of them (an
/// unusual mix, e.g. a bright high-key picture with a long dark tail) gets a curve drawn back towards the mean one.
fn curve(model: &Model, t: &Tones) -> Curve {
    let p99 = t.percentile(99.0);
    let mut feats: Vec<f32> = model
        .feats
        .iter()
        .zip(model.fmean)
        .zip(model.flo.iter().zip(model.fhi))
        .map(|((f, m), (lo, hi))| {
            let v = match *f {
                Feat::P99 => p99,
                Feat::Pct(q) => t.percentile(q) - p99,
                Feat::Dark(k) => t.below(-k),
            };
            v.clamp(*lo, hi.max(*lo)) - m
        })
        .collect();
    let d2: f32 = model.icov.iter().zip(&feats).map(|(row, a)| a * row.iter().zip(&feats).map(|(c, b)| c * b).sum::<f32>()).sum();
    let dist = d2.max(0.0).sqrt();
    let w = if model.far.1 > 0.0 { (1.0 - (dist - model.far.0) / model.far.1).clamp(0.0, 1.0) } else { 1.0 };
    let w = if w.is_finite() { w } else { 0.0 };
    for f in &mut feats {
        *f *= w;
    }
    let blocks: Vec<(f32, [f32; KN])> = model
        .blocks
        .iter()
        .map(|b| {
            let mut k = b.c0;
            for (s, d) in b.slopes.iter().zip(&feats) {
                for (kj, sj) in k.iter_mut().zip(s) {
                    *kj += sj * d;
                }
            }
            let off = match b.x {
                X::Pct(q) => t.percentile(q),
                X::Abs => 0.0,
            };
            (off, k)
        })
        .collect();
    Curve(blocks)
}

/// How much of the +100 Shadows curve a slider fraction `s` (−1 to 1) gives: Lightroom's +50 is 0.46 of its +100
/// and its −100 is −1.02 of it (in EV; chosen on fitting pictures, then 2.7 and 3.6 levels from Lightroom on test
/// pictures, against V7's 5.2 and 8.3). Scaling the +100 curve beat a curve fitted to each amount on its own.
fn shadows_gain(s: f32) -> f32 {
    let a = s.abs().min(1.0);
    let g = a * (0.84 + 0.16 * a);
    if s >= 0.0 { g } else { -SHADOWS_DOWN_SCALE * g }
}

/// Lightroom's Shadows −100 against its +100, in EV.
const SHADOWS_DOWN_SCALE: f32 = 1.02;
/// Lightroom's Highlights +100 against its −100, in EV.
const HIGHLIGHTS_UP_SCALE: f32 = 0.95;

/// How much of the −100 Highlights curve a slider fraction `h` (−1 to 1) gives: its +100 is −0.95 of it (8.7 levels
/// from Lightroom on test pictures, V7 10.9); in between as for Shadows.
fn highlights_gain(h: f32) -> f32 {
    let a = h.abs().min(1.0);
    let g = a * (0.84 + 0.16 * a);
    if h <= 0.0 { g } else { -HIGHLIGHTS_UP_SCALE * g }
}

/// The shift (EV) of each base-layer tone for a picture: [`LIFT_N`] values from [`LIFT_LO`].
#[derive(Clone, Debug, PartialEq)]
pub struct Lift(pub Vec<f32>);

impl Lift {
    /// The shift at tone `ev` (linear between entries, the ends held).
    #[inline]
    pub fn at(&self, ev: f32) -> f32 {
        let f = ((ev - LIFT_LO) / LIFT_STEP).clamp(0.0, (LIFT_N - 1) as f32);
        let f = if f.is_finite() { f } else { 0.0 };
        let i = (f as usize).min(LIFT_N - 2);
        let t = f - i as f32;
        let (a, b) = (self.0.get(i).copied().unwrap_or(0.0), self.0.get(i + 1).copied().unwrap_or(0.0));
        a + (b - a) * t
    }

    /// The table for tones `t` (scene EV) at slider fractions `shadows`, `highlights` (−1 to 1). `display` maps a
    /// scene tone to its displayed tone (EV) for a raw (the camera's default rendering; the curve is read there),
    /// `None` for a display-referred picture, whose tones are shown as they are.
    pub fn new(t: &Tones, shadows: f32, highlights: f32, display: Option<&dyn Fn(f32) -> f32>) -> Lift {
        Lift::with_whites(t, shadows, highlights, 0.0, display)
    }

    /// [`Lift::new`] with a raw's Whites `whites` (−1 to 1): its positive side is Lightroom's picture-dependent
    /// raw Whites ([`Whites`]); a display-referred picture and the negative side leave it to the tone map.
    pub fn with_whites(t: &Tones, shadows: f32, highlights: f32, whites: f32, display: Option<&dyn Fn(f32) -> f32>) -> Lift {
        let (gs, gh) = (shadows_gain(finite(shadows)), highlights_gain(finite(highlights)));
        let gw = if display.is_some() { finite(whites).max(0.0) } else { 0.0 };
        let shown = display.map(|d| t.mapped(d));
        let td = shown.as_ref().unwrap_or(t);
        let mut cs = curve(&tables::SHADOWS_UP, td);
        let mut ch = curve(&tables::HIGHLIGHTS_DOWN, td);
        // a raw: the curve fitted on pictures plus the correction fitted on raws ([`raw`])
        if display.is_some() {
            cs.0.extend(curve(&raw::SHADOWS_UP_RAW, td).0);
            ch.0.extend(curve(&raw::HIGHLIGHTS_DOWN_RAW, td).0);
        }
        // Shadows +100 never darkens much and Highlights −100 never brightens much (a fit's tail can say otherwise)
        let cw = whites::WHITES_UP_RAW.curve(td);
        // a raw's Highlights −: Lightroom's pull of the scene tone ([`highlights`]) in place of the displayed-tone curve
        let pull = (display.is_some() && gh > 0.0).then(|| highlights::Pull::of(t));
        let hl = |d: f32| if pull.is_some() { 0.0 } else { gh * ch.at(d).min(SIGN_SLACK) };
        let shift = |d: f32| gs * cs.at(d).max(-SIGN_SLACK) + hl(d) + gw * cw.at(d);
        let v = (0..LIFT_N)
            .map(|i| {
                let e = LIFT_LO + LIFT_STEP * i as f32;
                // (a raw: the curve read at the displayed tone, its shift turned into a scene shift)
                let s = match display {
                    None => shift(e),
                    Some(d) => scene_shift(d, e, shift(d(e))) + pull.map_or(0.0, |p| gh * p.at(e)),
                };
                if s.is_finite() { s.clamp(-MAX_SHIFT, MAX_SHIFT) } else { 0.0 }
            })
            .collect::<Vec<f32>>();
        // tones keep their order: a darker tone never ends up brighter than a lighter one
        let mut top = f32::MIN;
        let v = v
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let e = LIFT_LO + LIFT_STEP * i as f32;
                top = top.max(e + s);
                (top - e).clamp(-MAX_SHIFT, MAX_SHIFT)
            })
            .collect();
        Lift(v)
    }
}

/// Knots of the Whites curve: `E − P` from `knot_lo` EV in steps of 0.5.
const WK: usize = 23;

/// Lightroom Classic's Whites +100 on a raw, measured on our Canon EOS R5 C raws (rounds 7 and 8, OneSlips, Nov
/// 2026): it lifts the tones a few EV under the picture's bright end (`P`, the `pct` percentile of the displayed
/// tones), and lifts them more the further that end sits under white (`gap`, EV): on a dark frame whose brightest
/// tones are 2 EV under white the midtones rise about 4.5 EV, on a bright frame about 0.5. On held-out raws (7 shots,
/// two scenes, rendered) the change it makes is 16.0 levels rms from Lightroom's, against 39.0 for the fixed
/// white-point move it replaces. The curve's knot values are
/// `c0 + per_gap × gap`, the gap held to the largest the fit saw.
pub(crate) struct Whites {
    pct: f32,
    white: f32,
    gap_max: f32,
    knot_lo: f32,
    c0: [f32; WK],
    per_gap: [f32; WK],
}

/// One picture's Whites curve: its offset `P` and knot values.
struct WhitesCurve {
    off: f32,
    k: [f32; WK],
    lo: f32,
}

impl Whites {
    fn curve(&self, t: &Tones) -> WhitesCurve {
        let p = t.percentile(self.pct);
        let gap = (self.white - p).clamp(0.0, self.gap_max);
        let gap = if gap.is_finite() { gap } else { 0.0 };
        let mut k = self.c0;
        for (kj, g) in k.iter_mut().zip(self.per_gap) {
            *kj += g * gap;
        }
        WhitesCurve { off: p, k, lo: self.knot_lo }
    }
}

impl WhitesCurve {
    fn at(&self, e: f32) -> f32 {
        let f = ((e - self.off - self.lo) / 0.5).clamp(0.0, (WK - 1) as f32);
        let f = if f.is_finite() { f } else { 0.0 };
        let i = (f as usize).min(WK - 2);
        let t = f - i as f32;
        let (a, b) = (self.k.get(i).copied().unwrap_or(0.0), self.k.get(i + 1).copied().unwrap_or(0.0));
        a + (b - a) * t
    }
}

/// A raw's scene shift at scene tone `e` for the displayed shift `shown` of its displayed tone `d(e)`.
///
/// Where the camera curve is steeper than 1:1 (its toe), a scene shift shows several times larger, so the scene
/// tone that displays the moved tone is solved for: on our R5 C raws (round 7, OneSlips, Nov 2026) giving the
/// displayed shift straight to the scene tone lifted a dark room's shadows 3–5× more than Lightroom did, and than
/// the curve asks for. Where the curve is 1:1 or flatter (the shoulder, and past its white, where solving would fold
/// every tone into white) the shift goes to the scene tone as it is, so those tones keep their order and Highlights
/// brings them into view as Lightroom's does.
fn scene_shift(d: &dyn Fn(f32) -> f32, e: f32, shown: f32) -> f32 {
    const H: f32 = 1.0 / 64.0;
    let slope = (d(e + H) - d(e - H)) / (2.0 * H);
    if slope.is_nan() || slope <= 1.0 || !shown.is_finite() {
        return shown;
    }
    let want = d(e) + shown;
    let (mut lo, mut hi) = (e - MAX_SHIFT, e + MAX_SHIFT);
    if !(d(lo) <= want && want <= d(hi)) {
        return shown;
    }
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if d(mid) < want {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi) - e
}

fn finite(v: f32) -> f32 {
    if v.is_finite() { v.clamp(-1.0, 1.0) } else { 0.0 }
}

/// Whether `s` renders its global Highlights / Shadows with this module's curve.
pub fn active(s: &DevelopSettings) -> bool {
    s.process.process() >= Process::V9 && (s.light.shadows != 0.0 || s.light.highlights != 0.0 || s.light.whites > 0.0)
}

/// The share of Lightroom's −100 pull a negative Highlights in `s` gives under process V9 ([`active`]), `None` when
/// Highlights isn't below 0 there.
pub(crate) fn highlights_down(s: &DevelopSettings) -> Option<f32> {
    let h = (s.light.highlights / 100.0) as f32;
    (active(s) && h < 0.0).then(|| highlights_gain(finite(h)))
}

/// A raw's Highlights − pull of `s` ([`highlights`], at the slider's share) as a table on this module's grid, for the
/// whole `frame` of `src`: `None` when Highlights isn't below 0 under process V9 or there is nothing to measure. The
/// table [`for_frame`] gives already holds it at each pixel's base tone; [`crate::hl_edge`] reads it again at a
/// tone that leans towards the surroundings'.
pub(crate) fn pull_table(src: &Rgb32f, info: &SourceInfo, s: &DevelopSettings, frame: &Frame) -> Option<Lift> {
    let gh = highlights_down(s)?;
    let p = highlights::Pull::of(&*tones(src, info, s, frame)?);
    Some(Lift(
        (0..LIFT_N)
            .map(|i| {
                let v = gh * p.at(LIFT_LO + LIFT_STEP * i as f32);
                if v.is_finite() { v.clamp(-MAX_SHIFT, 0.0) } else { 0.0 }
            })
            .collect(),
    ))
}

/// Whether a raw's Whites is drawn by this module's curve rather than the tone map: process V9 and above, Whites up
/// ([`Whites`]). The tone map then leaves its white point where Whites 0 has it ([`crate::finish`]).
pub fn whites_on_raw(s: &DevelopSettings) -> bool {
    s.process.process() >= Process::V9 && s.light.whites > 0.0
}

type Cached = (u64, Arc<Tones>);
static LAST: Mutex<Option<Cached>> = Mutex::new(None);

/// The picture's tones (scene EV, exposure included) measured on the whole `frame` of `src`; `None` when there is
/// nothing to measure. The last result is kept, keyed by everything it depends on.
fn tones(src: &Rgb32f, info: &SourceInfo, s: &DevelopSettings, frame: &Frame) -> Option<Arc<Tones>> {
    let (w, h) = frame.fit(EDGE, EDGE);
    if w == 0 || h == 0 || src.width == 0 || src.height == 0 {
        return None;
    }
    let ev = s.light.exposure as f32;
    let key = {
        use std::hash::{BuildHasher, BuildHasherDefault, DefaultHasher};
        let (t, tint) = crate::local::effective_wb(info, s);
        BuildHasherDefault::<DefaultHasher>::default().hash_one((
            (w, h),
            crate::optics::fingerprint(src),
            format!("{frame:?}"),
            [t.to_bits(), tint.to_bits(), info.as_shot_temp.to_bits(), info.as_shot_tint.to_bits()],
            crate::local::white_balance_key(info, s),
            ev.to_bits(),
        ))
    };
    if let Some((k, t)) = LAST.lock().unwrap_or_else(PoisonError::into_inner).as_ref()
        && *k == key
    {
        return Some(t.clone());
    }
    let mut img = Rgb32f::clone(&frame_proxy(src, frame, w, h));
    white_balance(&mut img, info, s);
    let t = Arc::new(Tones::new(img.data.iter().map(|c| log_lum(*c) + ev))?);
    *LAST.lock().unwrap_or_else(PoisonError::into_inner) = Some((key, t.clone()));
    Some(t)
}

/// The V9 Highlights / Shadows table of `s` for the whole `frame` of `src`, or `None` when `s` doesn't use it
/// ([`active`]) or the frame is empty. `raw_tone` is a raw's default tone map (scene → display luminance), `None`
/// for a display-referred source.
pub fn for_frame(src: &Rgb32f, info: &SourceInfo, s: &DevelopSettings, frame: &Frame, raw_tone: Option<&ToneMap>) -> Option<Arc<Lift>> {
    if !active(s) {
        return None;
    }
    let (sh, hl) = ((s.light.shadows / 100.0) as f32, (s.light.highlights / 100.0) as f32);
    if raw_tone.is_none() && sh == 0.0 && hl == 0.0 {
        return None;
    }
    let t = tones(src, info, s, frame)?;
    let lift = match raw_tone {
        None => Lift::new(&t, sh, hl, None),
        Some(tm) => {
            let d = |e: f32| (tm.apply(GREY * e.exp2()).max(1e-9) / GREY).log2();
            Lift::with_whites(&t, sh, hl, (s.light.whites / 100.0) as f32, Some(&d))
        }
    };
    Some(Arc::new(lift))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A picture's tones: a dark half, a mid ramp and a few bright ones (fixed, so the golden values below hold).
    fn picture() -> Tones {
        let v = (0..1000).map(|i| {
            let x = i as f32 / 999.0;
            if x < 0.5 {
                -6.0 + 4.0 * x
            } else if x < 0.98 {
                -2.0 + 3.0 * (x - 0.5)
            } else {
                2.0
            }
        });
        Tones::new(v).unwrap()
    }

    #[test]
    fn percentiles_follow_numpy() {
        let t = Tones::new([4.0, 1.0, 3.0, 2.0]).unwrap();
        assert_eq!(t.percentile(0.0), 1.0);
        assert_eq!(t.percentile(100.0), 4.0);
        assert!((t.percentile(50.0) - 2.5).abs() < 1e-6);
        assert!((t.percentile(10.0) - 1.3).abs() < 1e-6);
        assert_eq!(t.below(2.5), 0.5);
        assert!(Tones::new([f32::NAN]).is_none());
    }

    #[test]
    fn zero_sliders_shift_nothing() {
        let l = Lift::new(&picture(), 0.0, 0.0, None);
        assert!(l.0.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn shadows_lift_darks_and_highlights_pull_brights() {
        let t = picture();
        let s = Lift::new(&t, 1.0, 0.0, None);
        assert!(s.at(-5.0) > 0.4, "Shadows +100 lifts a dark tone: {}", s.at(-5.0));
        assert!(s.at(2.0).abs() < s.at(-5.0) * 0.5, "and leaves the brightest mostly alone");
        let h = Lift::new(&t, 0.0, -1.0, None);
        assert!(h.at(1.5) < -0.3, "Highlights −100 pulls a bright tone down: {}", h.at(1.5));
        assert!(h.at(-5.0).abs() < h.at(1.5).abs() * 0.2, "and the darks much less: {}", h.at(-5.0));
        // the slider's other side goes the other way
        assert!(Lift::new(&t, -1.0, 0.0, None).at(-5.0) < 0.0);
        assert!(Lift::new(&t, 0.0, 1.0, None).at(1.5) > 0.0);
        // and half the slider does about half
        let half = Lift::new(&t, 0.5, 0.0, None).at(-5.0);
        assert!((half / s.at(-5.0) - 0.46).abs() < 0.01, "{half}");
    }

    #[test]
    fn matches_the_fitted_reference() {
        // the Python reference (fit/final6 + refcurve.design) on `picture()`: shift (EV) at -5, -2, 0, 1.5
        let s = Lift::new(&picture(), 1.0, 0.0, None);
        let h = Lift::new(&picture(), 0.0, -1.0, None);
        for (e, want_s, want_h) in GOLDEN {
            assert!((s.at(e) - want_s).abs() < 2e-3, "shadows at {e}: {} vs {want_s}", s.at(e));
            assert!((h.at(e) - want_h).abs() < 2e-3, "highlights at {e}: {} vs {want_h}", h.at(e));
        }
    }

    /// A bright picture with a long dark tail, far from the fitting pictures: the guards hold its curve.
    fn far_picture() -> Tones {
        Tones::new((0..1000).map(|i| {
            let x = i as f32 / 999.0;
            if x < 0.15 { -9.0 + 3.0 * x / 0.15 } else { 1.0 + 1.2 * (x - 0.15) / 0.85 }
        }))
        .unwrap()
    }

    #[test]
    fn a_picture_unlike_the_fitted_ones_gets_a_guarded_curve() {
        let (s, h) = (Lift::new(&far_picture(), 1.0, 0.0, None), Lift::new(&far_picture(), 0.0, -1.0, None));
        for (e, want_s, want_h) in GOLDEN_FAR {
            assert!((s.at(e) - want_s).abs() < 3e-3, "shadows at {e}: {} vs {want_s}", s.at(e));
            assert!((h.at(e) - want_h).abs() < 3e-3, "highlights at {e}: {} vs {want_h}", h.at(e));
        }
        for l in [&s, &h] {
            assert!(l.0.iter().enumerate().all(|(i, v)| i == 0 || LIFT_STEP + v >= l.0[i - 1] - 1e-5), "monotone");
        }
    }

    const GOLDEN_FAR: [(f32, f32, f32); 4] = [(-8.0, -0.5, -0.42068), (-4.0, 1.84084, 0.06388), (0.0, 1.04332, -0.72077), (1.5, 0.09962, -0.96107)];

    const GOLDEN: [(f32, f32, f32); 4] = [(-5.0, 1.13535, -0.16782), (-2.0, 0.27426, -0.49203), (0.0, -0.01171, -0.82795), (1.5, -0.06334, -1.70216)];

    #[test]
    fn a_raw_keeps_the_order_of_tones_past_the_camera_curves_white() {
        // a camera curve that clips 2 EV over grey: every scene tone above shows as white
        let d = |e: f32| e.min(2.0);
        let l = Lift::new(&picture(), 0.0, -1.0, Some(&d));
        // (Lightroom's pull of a raw compresses tones past white, [`highlights`], but never reverses them)
        let (a, b) = (3.0 + l.at(3.0), 5.0 + l.at(5.0));
        assert!(b - a > 0.5, "tones past white stay apart: {a} and {b}");
        assert!(l.0.iter().enumerate().all(|(i, v)| i == 0 || LIFT_STEP + v >= l.0[i - 1] - 1e-5), "monotone");
    }

    #[test]
    fn a_raw_shifts_its_scene_tones_so_the_shown_tones_follow_the_curve() {
        let t = picture();
        // a display mapping that brightens by 1 EV: the scene shift equals the shown shift of the shown tone
        let d = |e: f32| e + 1.0;
        let raw = Lift::new(&t, 1.0, 0.0, Some(&d));
        let one = |e: f32| e;
        let shown = Lift::new(&t.mapped(d), 1.0, 0.0, Some(&one));
        for e in [-6.0, -3.0, 0.0] {
            assert!((raw.at(e) - shown.at(e + 1.0)).abs() < 1e-3, "{e}: {} vs {}", raw.at(e), shown.at(e + 1.0));
        }
    }

    #[test]
    fn a_raw_adds_the_raw_correction_to_the_curve() {
        // the Python reference (fit/rawgold7.py, from rawres7.json) on `picture()` with a 1:1 display mapping
        let d = |e: f32| e;
        let s = Lift::new(&picture(), 1.0, 0.0, Some(&d));
        let h = Lift::new(&picture(), 0.0, -1.0, Some(&d));
        // (a raw's Highlights − is Lightroom's pull of the scene tone instead, [`highlights`])
        let pull = highlights::Pull::of(&picture());
        for (e, want_s, _) in GOLDEN_RAW {
            assert!((s.at(e) - want_s).abs() < 3e-3, "shadows at {e}: {} vs {want_s}", s.at(e));
            assert!((h.at(e) - pull.at(e)).abs() < 3e-3, "highlights at {e}: {} vs {}", h.at(e), pull.at(e));
        }
        // and a display-referred picture doesn't get it
        let tiff = Lift::new(&picture(), 1.0, 0.0, None);
        assert!((tiff.at(-5.0) - GOLDEN[0].1).abs() < 2e-3);
    }

    const GOLDEN_RAW: [(f32, f32, f32); 4] =
        [(-5.0, 0.98170, -0.18469), (-2.0, 1.03876, -0.49872), (0.0, 0.09104, -0.66611), (1.5, 0.01658, -1.47700)];

    #[test]
    fn a_raw_in_the_camera_curves_toe_shows_the_curves_shift_not_a_multiple() {
        // a camera curve with a steep toe: a scene EV below grey shows 3 EV darker per EV
        let d = |e: f32| if e < 0.0 { 3.0 * e } else { e };
        let t = picture();
        let raw = Lift::new(&t, 1.0, 0.0, Some(&d));
        let one = |e: f32| e;
        let shown = Lift::new(&t.mapped(d), 1.0, 0.0, Some(&one));
        let mut checked = 0;
        for i in 0..12 {
            let e = -3.0 + 0.25 * i as f32;
            let want = shown.at(d(e));
            if want < 0.2 {
                continue;
            }
            let got = d(e + raw.at(e)) - d(e);
            assert!((got - want).abs() < 0.02, "{e}: shows {got}, the curve asks {want}");
            checked += 1;
        }
        assert!(checked >= 4, "the curve lifts the toe's tones: {checked} checked");
    }

    #[test]
    fn hostile_values_stay_finite() {
        let t = Tones::new([f32::NEG_INFINITY, -1e30, 0.0, 1e30, f32::NAN]).unwrap();
        for (s, h) in [(f32::NAN, 1.0), (1e9, -1e9), (1.0, -1.0)] {
            let l = Lift::new(&t, s, h, None);
            assert!(l.0.iter().all(|v| v.is_finite() && v.abs() <= MAX_SHIFT));
            assert!(l.at(f32::NAN).is_finite() && l.at(1e30).is_finite());
        }
    }

    /// The measured tones are kept by what the white balance reads: the same pixels and white under another camera
    /// colour model (re-developed in camera space, or adapted with Bradford) are measured again, not taken from the
    /// last photo's.
    #[test]
    fn the_kept_tones_follow_the_camera_colour_model() {
        use lcc_color::Mat3;
        let mut src = Rgb32f::new(96, 64);
        for (i, p) in src.data.iter_mut().enumerate() {
            let x = (i % 96) as f32 / 95.0;
            *p = [0.02 + 0.6 * x, 0.05 + 0.2 * (1.0 - x), 0.4 * x * x + 0.01];
        }
        let tags = lcc_raw::ColorData {
            illuminant: [17, 21],
            color_matrix: [
                Some(Mat3([[0.9, 0.2, -0.15], [-0.3, 1.25, 0.08], [0.02, -0.12, 0.85]])),
                Some(Mat3([[0.7, 0.3, -0.1], [-0.35, 1.3, 0.1], [0.05, -0.2, 1.0]])),
            ],
            ..Default::default()
        };
        let cc = Arc::new(crate::CameraColor { tags, developed_for: lcc_color::cct::temp_tint_to_xy(5200.0, 5.0), deferred_tables_gain: None });
        let camera = SourceInfo { raw: true, as_shot_temp: 5200.0, as_shot_tint: 5.0, camera_color: Some(cc), ..Default::default() };
        let bradford = SourceInfo { relative_wb: true, ..camera.clone() };
        let mut s = DevelopSettings::default();
        (s.wb.mode, s.wb.temp, s.wb.tint) = (lcc_develop::WbMode::Custom, 2900.0, 30.0);
        let frame = Frame::new(src.width, src.height, &s, true);
        let fresh = |info: &SourceInfo| {
            let mut img = Rgb32f::clone(&frame_proxy(&src, &frame, 96, 64));
            white_balance(&mut img, info, &s);
            img.data.iter().map(|c| log_lum(*c)).sum::<f32>()
        };
        assert!((fresh(&camera) - fresh(&bradford)).abs() > 1e-3, "the two models white-balance differently");
        let median = |info: &SourceInfo| tones(&src, info, &s, &frame).map(|t| t.percentile(50.0));
        let (a, b) = (median(&camera), median(&bradford));
        assert_ne!(a, b, "the second model's tones were taken from the first's");
        assert_eq!(median(&camera), a);
    }
}
