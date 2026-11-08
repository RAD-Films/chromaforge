//! The global Highlights and Shadows of process V8: Lightroom Classic's response, one tone curve per picture.
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
//! Lightroom's raw response is not its TIFF one. Fitted on our Canon EOS R5 C raws (rounds 7–10, 34 scenes, Nov 2026)
//! and scored on flat areas of held-out shots, the change each slider makes is, in 8-bit levels rms from Lightroom's
//! (before this fit → now, and the best one curve per picture can do): Shadows +100 20.7 → 16.1 (3.7), Highlights
//! −100 12.2 → 6.6 (4.2), both together 26.2 → 11.0 (6.5). The Shadows correction also reads how much of the frame
//! is shown white (Lightroom lifts a frame's shadows less the more of it is white), and the pair has its own
//! correction, as Lightroom's pair isn't the sum of its two sliders. In the curve's toe the scene shift is the one
//! that moves the displayed tone by the curve's amount ([`scene_shift`]); elsewhere the shift is given to the scene
//! tone as it is, which keeps the order of tones past the camera curve's white (Highlights then brings them into
//! view, as Lightroom's does).

use std::sync::{Arc, Mutex, PoisonError};

use lcc_develop::{DevelopSettings, Process};
use lcc_raster::Rgb32f;

use crate::SourceInfo;
use crate::geometry::Frame;
use crate::local::{frame_proxy, log_lum, white_balance};
use crate::tone::{GREY, ToneMap};

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

/// The darkest shown tone (EV) the raw fit saw: 8-bit level 1 of an sRGB export. The raw curves were fitted on
/// 8-bit exports, so a raw's curve is read no lower than this, and its histogram counts the tones an export shows
/// as level 0 (under [`RAW_BLACK`]) at [`EV_FLOOR`], as the fit's did. Without both, a dark frame's deep shadows,
/// which a float render keeps, met the curve's unfitted tail: Shadows +100 lifted them 3–4 EV past Lightroom.
const RAW_FIT_LOW: f32 = -9.21;
/// The shown tone (EV) under which an 8-bit sRGB export shows level 0 (level ½).
const RAW_BLACK: f32 = -10.21;

/// The per-picture table: EV from [`LIFT_LO`] in steps of [`LIFT_STEP`], [`LIFT_N`] entries.
pub const LIFT_LO: f32 = -20.0;
pub const LIFT_STEP: f32 = 1.0 / 32.0;
pub const LIFT_N: usize = 961;

/// The largest shift (EV) the table holds either way.
const MAX_SHIFT: f32 = 8.0;

/// How far (EV) the +100 Shadows curve may darken a tone, and the −100 Highlights curve brighten one.
const SIGN_SLACK: f32 = 0.5;

/// A raw's display white for Highlights (EV over grey): 8-bit level 254 of an sRGB export. (At level 250, the
/// near-white tones the curves were fitted on took the pull too: bright frames' near-white went −0.5 EV where
/// Lightroom's went −0.1.)
const SHOWN_WHITE: f32 = 2.461;
/// Highlights down on a raw's tones at and above its display white, which the curves can't reach: they are read
/// at the shown tone, and the camera curve holds every tone past white at white, so a window in a dark room kept
/// its one shift and stayed white. Lightroom pulls those tones down by a scene shift that grows with how dark the
/// picture is: `ABOVE_WHITE_BASE + ABOVE_WHITE_DARK × (share of tones over 2 EV under the scene's mid grey)`, at
/// the Highlights amounts [`ABOVE_WHITE_AT`] ([`highlights_gain`] of 0, −50 and −100), linear in between. Measured
/// on our R5 C raws (rounds 9b and 10, OneSlips, Nov 2026), all on Lightroom's side: each pixel white at zero
/// sliders got its scene tone from Lightroom's Exposure −1/−2/−3 exports (its response recovered from those
/// brackets), and its Highlights output the scene shift it takes. Only pixels Lightroom brings under level 250
/// count; a shot whose whites it leaves white counts as no pull (a white output only bounds the shift, and taking
/// the bound put a −1.45 EV pull on every bright frame). 37–38 shots. Lightroom's HDR exports (no clip) agree: a
/// frame with no dark tones gets about none, one half dark about −1.4 EV. A window in a dark interior needs −3.4 to
/// −5.3 EV; CF gave −0.9 to −2.6.
const ABOVE_WHITE_AT: [f32; 3] = [0.0, 0.46, 1.0];
const ABOVE_WHITE_BASE: [f32; 3] = [0.0, -0.13, -0.74];
const ABOVE_WHITE_DARK: [f32; 3] = [0.0, -3.32, -4.47];

/// A histogram feature: a percentile relative to the 99th, the 99th itself, the share of tones under `−k` EV, or the
/// share of tones over `k` EV (how much of a raw shows white, which its percentiles, all at white, can't tell).
#[derive(Clone, Copy, Debug)]
pub(crate) enum Feat {
    P99,
    Pct(f32),
    Dark(f32),
    Bright(f32),
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
pub struct Tones(Vec<f32>, Option<f32>);

impl Tones {
    /// The finite tones of `ev` (each at least [`EV_FLOOR`]); `None` when none is finite.
    pub fn new(ev: impl IntoIterator<Item = f32>) -> Option<Tones> {
        let mut v: Vec<f32> = ev.into_iter().filter(|e| e.is_finite()).map(|e| e.max(EV_FLOOR)).collect();
        if v.is_empty() {
            return None;
        }
        v.sort_unstable_by(f32::total_cmp);
        Some(Tones(v, None))
    }

    /// The same tones with a raw's clip tone `clip` (EV): the tone its sensor-clipped pixels show at, `None` for none.
    pub fn with_clip(mut self, clip: Option<f32>) -> Tones {
        self.1 = clip.filter(|c| c.is_finite());
        self
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
        Tones(v, self.1.map(&f))
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
                Feat::Bright(k) => 1.0 - t.below(k),
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

/// The shift (EV) of each base-layer tone for a picture, as two tables of [`LIFT_N`] values from [`LIFT_LO`]: Shadows'
/// part (`0`, with the pair's correction) and Highlights' (`2`, with a raw's Whites). They are read at different bases
/// ([`crate::local::NarrowBase`]): Shadows at the brighter of the picture's base and the narrow one, Highlights at the
/// darker, so neither spreads across an edge into the other region. `1`: a raw's past-white term.
#[derive(Clone, Debug, PartialEq)]
pub struct Lift(pub Vec<f32>, pub Option<Regional>, pub Vec<f32>);

/// Highlights down on a raw's tones past its display white, read at the narrow base ([`crate::local::NarrowBase`]: a
/// Gaussian of 0.12 % of the long edge) rather than the tone table: a gain of `amp` EV weighted by how far that base
/// sits past white, from none [`REGION_BELOW`] EV under white to all of it [`REGION_ABOVE`] EV over, and back to none
/// at the raw's clip tone (a clipped sun stays white, as Lightroom keeps it). The narrow base follows a bright region's
/// outline as Lightroom's pull does: read at the wide base (6 %), a window in a dark room sat far under white there and
/// kept its whites (rounds 9b and 10, Nov 2026: at Highlights −100 the white-at-zero pixels went from 28.1 to 25.5
/// levels rms from Lightroom, held-out 17.6 to 15.5; at −50 9.4 to 8.1, held-out 6.4 to 4.9; with
/// [`REGION_MIN_SLOPE`] 0, −100: 23.6, held-out 13.2; −50: 7.8, held-out 5.9). The rise is spread
/// over enough tones that a ramp keeps [`REGION_MIN_SLOPE`] of its slope, so a sky's gradient doesn't fold where its
/// tones cross white.
#[derive(Clone, Debug, PartialEq)]
pub struct Regional {
    pub amp: f32,
    pub rise: (f32, f32),
    pub fall: Option<(f32, f32)>,
}

impl Regional {
    /// The gain (EV, ≤ 0) at narrow-base tone `wb`.
    #[inline]
    pub fn at(&self, wb: f32) -> f32 {
        if !wb.is_finite() {
            return 0.0;
        }
        let up = smoothstep(self.rise.0, self.rise.1, wb);
        let down = match self.fall {
            Some((a, b)) => 1.0 - smoothstep(a, b, wb),
            None => 1.0,
        };
        let v = self.amp * up * down;
        if v.is_finite() { v.clamp(-MAX_SHIFT, 0.0) } else { 0.0 }
    }
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    if e1 <= e0 {
        return if x >= e1 { 1.0 } else { 0.0 };
    }
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// [`Regional`]: the narrow base's tones under white (EV) where the pull starts, and over white where it is complete.
const REGION_BELOW: f32 = 1.0;
const REGION_ABOVE: f32 = 1.0;
/// [`Regional`]: the least slope of a tone's output against its own tone where the base follows it (a smooth sky
/// ramp): the pull's rise is spread over enough tones that the ramp keeps this much of its slope, so it never folds
/// into a contour. Lightroom flattens such a region about this far (its own slope within a pulled white region,
/// rounds 9b / 10: 0.16 to 0.7, one −0.06); at 0.3 a strong pull spread so wide that a window just past white kept
/// little of it (Highlights −100: 25.5 → 23.6 levels rms on the white-at-zero pixels, held-out 15.5 → 13.2; −50:
/// 8.1 → 7.8, held-out 4.9 → 5.9).
const REGION_MIN_SLOPE: f32 = 0.0;
/// [`Regional`]: the tones under the raw's clip tone (EV) over which the pull eases back to none.
const REGION_FALL: f32 = 1.5;
/// A raw counts as clipped when at least this share of its pixels hold its brightest channel value (within
/// [`CLIP_BAND`] of it, before white balance): a sensor's clip piles pixels up at one value, a bright window that
/// isn't clipped doesn't.
const CLIP_SHARE: f32 = 0.002;
const CLIP_BAND: f32 = 0.995;

/// Table `v`'s entry at tone `ev` (linear between entries, the ends held).
#[inline]
fn table_at(v: &[f32], ev: f32) -> f32 {
    let f = ((ev - LIFT_LO) / LIFT_STEP).clamp(0.0, (LIFT_N - 1) as f32);
    let f = if f.is_finite() { f } else { 0.0 };
    let i = (f as usize).min(LIFT_N - 2);
    let t = f - i as f32;
    let (a, b) = (v.get(i).copied().unwrap_or(0.0), v.get(i + 1).copied().unwrap_or(0.0));
    a + (b - a) * t
}

impl Lift {
    /// Shadows' shift at tone `ev` (linear between entries, the ends held).
    #[inline]
    pub fn at(&self, ev: f32) -> f32 {
        table_at(&self.0, ev)
    }

    /// Highlights' shift at tone `ev`.
    #[inline]
    pub fn at_h(&self, ev: f32) -> f32 {
        table_at(&self.2, ev)
    }

    /// The whole shift at tone `ev` where both parts are read at the same base.
    pub fn at_both(&self, ev: f32) -> f32 {
        self.at(ev) + self.at_h(ev)
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
        let shown = display.map(|d| t.mapped(|e| if d(e) < RAW_BLACK { EV_FLOOR } else { d(e) }));
        let td = shown.as_ref().unwrap_or(t);
        let mut cs = curve(&tables::SHADOWS_UP, td);
        let mut ch = curve(&tables::HIGHLIGHTS_DOWN, td);
        // a raw: the curve fitted on pictures plus the correction fitted on raws ([`raw`]); with Shadows up and
        // Highlights down together, the correction fitted on that pair takes over from the two by the product of
        // their amounts (Lightroom's pair isn't the sum of its two sliders: fitted on its own, the pair's held-out
        // shots went from 26.2 to 11.0 levels rms from Lightroom)
        let mut both: Option<Curve> = None;
        if display.is_some() {
            let (rs, rh) = (curve(&raw::SHADOWS_UP_RAW, td), curve(&raw::HIGHLIGHTS_DOWN_RAW, td));
            let mut b = curve(&raw::BOTH_RAW, td);
            b.0.extend(rs.0.iter().map(|(off, k)| (*off, k.map(|v| -v))));
            b.0.extend(rh.0.iter().map(|(off, k)| (*off, k.map(|v| -v))));
            both = Some(b);
            cs.0.extend(rs.0);
            ch.0.extend(rh.0);
        }
        let gb = gs.max(0.0) * gh.max(0.0);
        // Shadows +100 never darkens much and Highlights −100 never brightens much (a fit's tail can say otherwise)
        let cw = whites::WHITES_UP_RAW.curve(td);
        let pair = |d: f32| match &both {
            Some(b) if gb > 0.0 => gb * b.at(d),
            _ => 0.0,
        };
        let s_part = |d: f32| gs * cs.at(d).max(-SIGN_SLACK) + pair(d);
        let h_part = |d: f32| gh * ch.at(d).min(SIGN_SLACK) + gw * cw.at(d);
        // a raw's tones past its display white: Lightroom's pull for them ([`ABOVE_WHITE_BASE`]), read at the narrow base
        let region = display.and_then(|d| regional(t, d, gh));
        // each part on its own: smoothed to a curvature cap, then with tones kept in order
        let table = |shift: &dyn Fn(f32) -> f32| -> Vec<f32> {
            let v = (0..LIFT_N)
                .map(|i| {
                    let e = LIFT_LO + LIFT_STEP * i as f32;
                    // (a raw: the curve read at the displayed tone, its shift turned into a scene shift)
                    let s = match display {
                        None => shift(e),
                        Some(d) => scene_shift(d, e, shift(d(e).max(RAW_FIT_LOW))),
                    };
                    if s.is_finite() { s.clamp(-MAX_SHIFT, MAX_SHIFT) } else { 0.0 }
                })
                .collect::<Vec<f32>>();
            // tones keep their order: a darker tone never ends up brighter than a lighter one
            let mut top = f32::MIN;
            smooth_to_curvature(&v)
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let e = LIFT_LO + LIFT_STEP * i as f32;
                    top = top.max(e + s);
                    (top - e).clamp(-MAX_SHIFT, MAX_SHIFT)
                })
                .collect()
        };
        Lift(table(&s_part), region, table(&h_part))
    }
}

/// Most curvature (change of slope per EV) of tone + shift the table keeps between [`CURVE_FROM`] and [`CURVE_TO`] EV.
/// A table can be smooth and still bend sharply: 433's Highlights -100 took the slope of its sky from 1.0 down to 0.4
/// and back up to 1.2 within an EV around white, a dark and a bright ring around the sun (curvature 1.3 / EV).
const MAX_CURVATURE: f32 = 0.4;
const CURVE_FROM: f32 = -8.0;
const CURVE_TO: f32 = 5.0;
/// The widest smoothing [`smooth_to_curvature`] reaches for (EV, a Gaussian's sigma).
const TABLE_SMOOTH_MAX: f32 = 1.2;

/// The largest |curvature| of tone + shift of table `v` over [`CURVE_FROM`]..[`CURVE_TO`] EV.
fn max_curvature(v: &[f32]) -> f32 {
    let i0 = (((CURVE_FROM - LIFT_LO) / LIFT_STEP).max(1.0)) as usize;
    let i1 = (((CURVE_TO - LIFT_LO) / LIFT_STEP) as usize).min(v.len().saturating_sub(2));
    (i0..i1.max(i0))
        .filter_map(|i| Some((v.get(i + 1)? - 2.0 * v.get(i)? + v.get(i - 1)?) / (LIFT_STEP * LIFT_STEP)))
        .filter(|c| c.is_finite())
        .fold(0.0f32, |m, c| m.max(c.abs()))
}

/// `v` smoothed by [`TABLE_SMOOTH`], or by the least wider Gaussian (steps of a quarter, up to [`TABLE_SMOOTH_MAX`])
/// that keeps its curvature under [`MAX_CURVATURE`]: a table without sharp bends keeps its fitted shape.
fn smooth_to_curvature(v: &[f32]) -> Vec<f32> {
    let mut sigma = TABLE_SMOOTH;
    loop {
        let s = smoothed(v, sigma);
        if max_curvature(&s) <= MAX_CURVATURE || sigma >= TABLE_SMOOTH_MAX {
            return s;
        }
        sigma = (sigma * 1.25).min(TABLE_SMOOTH_MAX);
    }
}

/// Width (EV, a Gaussian's sigma) the table is smoothed over before tones are put back in order. The table sums
/// piecewise-linear curves whose knots sit at different tones (each block's own percentile), so its slope steps every
/// few tenths of an EV; read at a smooth base, every step drew an arc across a smooth wall (round 10, 6719: Lightroom's
/// change on that wall is one smooth hump). Smoothed, those steps go.
const TABLE_SMOOTH: f32 = 0.35;

/// `v` (one value per [`LIFT_STEP`]) smoothed by a Gaussian of `sigma` EV, the ends held.
fn smoothed(v: &[f32], sigma: f32) -> Vec<f32> {
    let r = (3.0 * sigma / LIFT_STEP).ceil() as isize;
    if r <= 0 || v.is_empty() {
        return v.to_vec();
    }
    let k: Vec<f32> = (-r..=r).map(|i| (-0.5 * (i as f32 * LIFT_STEP / sigma).powi(2)).exp()).collect();
    let sum: f32 = k.iter().sum();
    let n = v.len() as isize;
    (0..n)
        .map(|i| {
            let acc: f32 = (-r..=r).zip(&k).map(|(j, w)| w * v.get((i + j).clamp(0, n - 1) as usize).copied().unwrap_or(0.0)).sum();
            let x = acc / sum;
            if x.is_finite() { x } else { 0.0 }
        })
        .collect()
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

/// The scene tone (EV) at which the increasing display mapping `d` shows `shown`, or `None` where it never does
/// within the table's range.
fn scene_tone_showing(d: &dyn Fn(f32) -> f32, shown: f32) -> Option<f32> {
    let (mut lo, mut hi) = (LIFT_LO, LIFT_LO + LIFT_STEP * (LIFT_N - 1) as f32);
    if !(d(lo) <= shown && shown <= d(hi)) {
        return None;
    }
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if d(mid) < shown {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let e = 0.5 * (lo + hi);
    e.is_finite().then_some(e)
}

/// For a raw (display mapping `d`) at Highlights gain `gh`: the scene tone its display white sits at, and the
/// scene shift Lightroom gives the tones from there up ([`ABOVE_WHITE_BASE`]) and its dark-picture part; `None` when Highlights doesn't darken, or the
/// picture's camera curve never reaches white.
fn above_white(t: &Tones, d: &dyn Fn(f32) -> f32, gh: f32) -> Option<(f32, f32, f32)> {
    let g = finite(gh).clamp(0.0, 1.0);
    if g <= 0.0 {
        return None;
    }
    let white = scene_tone_showing(d, SHOWN_WHITE)?;
    let grey = scene_tone_showing(d, 0.0)?;
    let dark = t.below(grey - 2.0);
    let at = |k: &[f32; 3]| {
        let i = if g <= ABOVE_WHITE_AT[1] { 0 } else { 1 };
        let (lo, hi) = (ABOVE_WHITE_AT.get(i).copied().unwrap_or(0.0), ABOVE_WHITE_AT.get(i + 1).copied().unwrap_or(1.0));
        let f = if hi > lo { (g - lo) / (hi - lo) } else { 0.0 };
        let (a, b) = (k.get(i).copied().unwrap_or(0.0), k.get(i + 1).copied().unwrap_or(0.0));
        a + (b - a) * f
    };
    let part = at(&ABOVE_WHITE_DARK) * dark;
    let pull = at(&ABOVE_WHITE_BASE) + part;
    (pull.is_finite() && part.is_finite()).then_some((white, pull.clamp(-MAX_SHIFT, 0.0), part.clamp(-MAX_SHIFT, 0.0)))
}

/// The regional pull of a raw (display mapping `d`) with tones `t` at Highlights gain `gh` ([`Regional`]).
fn regional(t: &Tones, d: &dyn Fn(f32) -> f32, gh: f32) -> Option<Regional> {
    let (white, pull, _) = above_white(t, d, gh)?;
    if pull >= 0.0 {
        return None;
    }
    let fall = t.1.map(|clip| (clip - REGION_FALL, clip));
    // a smoothstep's steepest slope is 1.5 / its width
    let width = (REGION_BELOW + REGION_ABOVE).max(1.5 * -pull / (1.0 - REGION_MIN_SLOPE));
    Some(Regional { amp: pull, rise: (white - REGION_BELOW, white - REGION_BELOW + width), fall })
}

/// A raw's scene shift at scene tone `e` for the displayed shift `shown` of its displayed tone `d(e)`.
///
/// Where the camera curve is steeper than 1:1 over the move (its toe), a scene shift shows several times larger, so
/// the scene tone that displays the moved tone is solved for: on our R5 C raws (round 7, OneSlips, Nov 2026) giving
/// the displayed shift straight to the scene tone lifted a dark room's shadows 3–5× more than Lightroom did, and than
/// the curve asks for. The slope that counts is the curve's over the whole move, not at its start: a deep tone on
/// the 1:1 part under the toe, lifted up into it, showed twice the curve's lift (round 10, Nov 2026). Where the curve
/// is 1:1 or flatter over the move (the shoulder, and past its white, where solving would fold every tone into
/// white) the shift goes to the scene tone as it is, so those tones keep their order and Highlights brings them into
/// view as Lightroom's does.
fn scene_shift(d: &dyn Fn(f32) -> f32, e: f32, shown: f32) -> f32 {
    if !shown.is_finite() || shown == 0.0 {
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
    let solved = 0.5 * (lo + hi) - e;
    // steeper than 1:1 over the move: the solved shift is the smaller one
    if solved.is_finite() && solved.abs() < shown.abs() { solved } else { shown }
}

fn finite(v: f32) -> f32 {
    if v.is_finite() { v.clamp(-1.0, 1.0) } else { 0.0 }
}

/// Whether `s` renders its global Highlights / Shadows with this module's curve.
pub fn active(s: &DevelopSettings) -> bool {
    s.process.process() >= Process::V8 && (s.light.shadows != 0.0 || s.light.highlights != 0.0 || s.light.whites > 0.0)
}

/// Whether a raw's Whites is drawn by this module's curve rather than the tone map: process V8 and above, Whites up
/// ([`Whites`]). The tone map then leaves its white point where Whites 0 has it ([`crate::finish`]).
pub fn whites_on_raw(s: &DevelopSettings) -> bool {
    s.process.process() >= Process::V8 && s.light.whites > 0.0
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
            ev.to_bits(),
        ))
    };
    if let Some((k, t)) = LAST.lock().unwrap_or_else(PoisonError::into_inner).as_ref()
        && *k == key
    {
        return Some(t.clone());
    }
    let mut img = Rgb32f::clone(&frame_proxy(src, frame, w, h));
    // a raw's clipped pixels: those at its brightest channel value before white balance
    let peak = |c: &[f32; 3]| c[0].max(c[1]).max(c[2]);
    let top = img.data.iter().map(peak).filter(|v| v.is_finite()).fold(0.0f32, f32::max);
    let clipped: Vec<bool> = img.data.iter().map(|c| info.raw && top > 0.0 && peak(c) >= CLIP_BAND * top).collect();
    white_balance(&mut img, info, s);
    let n_clip = clipped.iter().filter(|c| **c).count();
    let clip = if n_clip as f32 >= CLIP_SHARE * img.data.len().max(1) as f32 {
        let mut v: Vec<f32> = img.data.iter().zip(&clipped).filter(|(_, c)| **c).map(|(p, _)| log_lum(*p) + ev).filter(|e| e.is_finite()).collect();
        v.sort_unstable_by(f32::total_cmp);
        v.get(v.len() / 2).copied()
    } else {
        None
    };
    let t = Arc::new(Tones::new(img.data.iter().map(|c| log_lum(*c) + ev))?.with_clip(clip));
    *LAST.lock().unwrap_or_else(PoisonError::into_inner) = Some((key, t.clone()));
    Some(t)
}

/// The V8 Highlights / Shadows table of `s` for the whole `frame` of `src`, or `None` when `s` doesn't use it
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
        assert!(s.at_both(-5.0) > 0.4, "Shadows +100 lifts a dark tone: {}", s.at_both(-5.0));
        assert!(s.at_both(2.0).abs() < s.at_both(-5.0) * 0.5, "and leaves the brightest mostly alone");
        let h = Lift::new(&t, 0.0, -1.0, None);
        assert!(h.at_both(1.5) < -0.3, "Highlights −100 pulls a bright tone down: {}", h.at_both(1.5));
        assert!(h.at_both(-5.0).abs() < h.at_both(1.5).abs() * 0.2, "and the darks much less: {}", h.at_both(-5.0));
        // the slider's other side goes the other way
        assert!(Lift::new(&t, -1.0, 0.0, None).at_both(-5.0) < 0.0);
        assert!(Lift::new(&t, 0.0, 1.0, None).at_both(1.5) > 0.0);
        // and half the slider does about half
        let half = Lift::new(&t, 0.5, 0.0, None).at_both(-5.0);
        // (0.46 of the curve; the full slider's table bends more, so it is smoothed wider and its peak drops more:
        // half the slider keeps 0.50 of it here)
        assert!((half / s.at_both(-5.0) - 0.48).abs() < 0.03, "{half}");
    }

    #[test]
    fn matches_the_fitted_reference() {
        // the Python reference (fit/final6 + refcurve.design) on `picture()`, its table smoothed to the curvature cap
        // ([`smooth_to_curvature`]) and kept in order: shift (EV) at -5, -2, 0, 1.5
        let s = Lift::new(&picture(), 1.0, 0.0, None);
        let h = Lift::new(&picture(), 0.0, -1.0, None);
        for (e, want_s, want_h) in GOLDEN {
            assert!((s.at_both(e) - want_s).abs() < 2e-3, "shadows at {e}: {} vs {want_s}", s.at_both(e));
            assert!((h.at_both(e) - want_h).abs() < 2e-3, "highlights at {e}: {} vs {want_h}", h.at_both(e));
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
            assert!((s.at_both(e) - want_s).abs() < 3e-3, "shadows at {e}: {} vs {want_s}", s.at_both(e));
            assert!((h.at_both(e) - want_h).abs() < 3e-3, "highlights at {e}: {} vs {want_h}", h.at_both(e));
        }
        for l in [&s, &h] {
            assert!(l.0.iter().enumerate().all(|(i, v)| i == 0 || LIFT_STEP + v >= l.0[i - 1] - 1e-5), "monotone");
        }
    }

    const GOLDEN_FAR: [(f32, f32, f32); 4] =
        [(-8.0, -0.28447, -0.37753), (-4.0, 1.91148, 0.08584), (0.0, 0.75536, -0.55538), (1.5, 0.27279, -1.00346)];

    const GOLDEN: [(f32, f32, f32); 4] = [(-5.0, 0.97033, -0.17011), (-2.0, 0.26340, -0.54558), (0.0, 0.03489, -1.08222), (1.5, -0.04849, -1.95687)];

    #[test]
    fn a_raw_keeps_the_order_of_tones_past_the_camera_curves_white() {
        // a camera curve that clips 2 EV over grey: every scene tone above shows as white
        let d = |e: f32| e.min(2.0);
        let l = Lift::new(&picture(), 0.0, -1.0, Some(&d));
        let (a, b) = (3.0 + l.at_both(3.0), 5.0 + l.at_both(5.0));
        assert!(b - a > 1.9, "tones past white stay apart: {a} and {b}");
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
            assert!((raw.at_both(e) - shown.at_both(e + 1.0)).abs() < 1e-3, "{e}: {} vs {}", raw.at_both(e), shown.at_both(e + 1.0));
        }
    }

    /// A camera curve that holds every tone past `white` at white, 1:1 below it.
    fn shoulder(white: f32) -> impl Fn(f32) -> f32 {
        move |e: f32| e.min(white)
    }

    /// Tones `n` of which in every hundred sit 4 EV under grey, the rest at grey, and a window 1.5 EV past white.
    fn room(dark_pct: usize) -> Tones {
        let v = (0..100).map(|i| {
            if i < dark_pct {
                -4.0
            } else if i < 97 {
                0.0
            } else {
                SHOWN_WHITE + 1.5
            }
        });
        Tones::new(v.collect::<Vec<f32>>()).unwrap()
    }

    /// A tone's output where the wide base follows it (a ramp filling the frame): its own tone, the table's shift and
    /// the regional pull.
    fn ramp_out(l: &Lift, e: f32) -> f32 {
        e + l.at_both(e) + l.1.as_ref().map_or(0.0, |r| r.at(e))
    }

    #[test]
    fn highlights_pull_tones_past_white_more_in_a_dark_picture() {
        let d = shoulder(SHOWN_WHITE);
        let window = SHOWN_WHITE + 1.5;
        let (bright, dark) = (Lift::new(&room(0), 0.0, -1.0, Some(&d)), Lift::new(&room(80), 0.0, -1.0, Some(&d)));
        let (rb, rd) = (bright.1.clone().unwrap(), dark.1.clone().unwrap());
        // the fitted pull: more in a dark room, read at a window's narrow base
        let wanted = |dark_share: f32| ABOVE_WHITE_BASE[2] + ABOVE_WHITE_DARK[2] * dark_share;
        assert!((rb.amp - wanted(0.0)).abs() < 1e-4 && (rd.amp - wanted(0.8)).abs() < 1e-4, "{} {}", rb.amp, rd.amp);
        // (it eases in over a span wide enough to keep a sky's slope, so a window just past white gets part of it)
        assert!(rd.at(window) < rb.at(window), "{} {}", rd.at(window), rb.at(window));
        assert!((rd.at(rd.rise.1 + 0.1) - rd.amp).abs() < 1e-4 && rd.amp < -1.0, "{} {}", rd.at(rd.rise.1 + 0.1), rd.amp);
        // nothing far under white, and never brighter
        assert_eq!(rd.at(SHOWN_WHITE - 1.5), 0.0);
        assert!((-30..60).all(|i| rd.at(i as f32 * 0.1) <= 0.0));
        // half the slider: less of it
        let half = Lift::new(&room(80), 0.0, -0.5, Some(&d));
        assert!(half.1.as_ref().unwrap().amp > rd.amp && half.1.as_ref().unwrap().amp < 0.0);
        // Highlights up, none, or a display-referred picture: no pull
        for (h, disp) in [(0.5, true), (0.0, true), (-1.0, false)] {
            let l = if disp { Lift::new(&room(80), 0.0, h, Some(&d)) } else { Lift::new(&room(80), 0.0, h, None) };
            assert!(l.1.is_none(), "h {h} raw {disp}");
        }
        // a camera curve that never reaches white: none either
        let dim = |e: f32| e.min(1.0);
        assert!(above_white(&room(80), &dim, 1.0).is_none());
    }

    #[test]
    fn a_sky_ramp_crossing_white_keeps_its_slope() {
        // a smooth sky filling the frame, so the wide base follows each tone: no tone may fold back under a darker
        // one (the contour v4's step drew across a sky), at any dark share or slider amount
        let d = shoulder(SHOWN_WHITE);
        for dark in [0, 30, 60, 90] {
            for h in [-0.3, -0.6, -1.0] {
                let l = Lift::new(&room(dark), 0.0, h, Some(&d));
                let step = 0.01;
                for i in 0..800 {
                    let e = SHOWN_WHITE - 3.0 + i as f32 * step;
                    let slope = (ramp_out(&l, e + step) - ramp_out(&l, e)) / step;
                    let table = (e + step + l.at_both(e + step) - e - l.at_both(e)) / step;
                    // the regional pull takes at most its share of the table's slope, and nothing folds
                    assert!(slope >= table - (1.0 - REGION_MIN_SLOPE) - 0.02, "dark {dark} % h {h}: slope {slope} (table {table}) at {e}");
                    assert!(slope > -1e-3, "dark {dark} % h {h}: folds at {e} (slope {slope})");
                }
            }
        }
    }

    #[test]
    fn a_clipped_sun_stays_white() {
        // a raw whose brightest tones pile up at the sensor's clip (a sun): the pull eases back to none there, so the
        // sun isn't pulled into a grey disc
        let d = shoulder(SHOWN_WHITE);
        let sun = SHOWN_WHITE + 4.0;
        let t = Tones::new((0..1000).map(|i| {
            if i < 600 {
                -4.0
            } else if i < 990 {
                0.0
            } else {
                sun
            }
        }))
        .unwrap()
        .with_clip(Some(sun));
        let l = Lift::new(&t, 0.0, -1.0, Some(&d));
        let r = l.1.as_ref().unwrap();
        assert!(r.fall.is_some());
        assert_eq!(r.at(sun), 0.0);
        assert!(r.at(sun - 2.0) < -0.5, "the glow round it is still pulled: {}", r.at(sun - 2.0));
        // an unclipped bright tone (no pile-up at the top) keeps the whole pull
        assert!(dark_room_window_unclipped() < -1.0);
    }

    fn dark_room_window_unclipped() -> f32 {
        let d = shoulder(SHOWN_WHITE);
        let t = Tones::new((0..1000).map(|i| {
            if i < 600 {
                -4.0
            } else if i < 990 {
                0.0
            } else {
                SHOWN_WHITE + 1.5
            }
        }))
        .unwrap();
        let l = Lift::new(&t, 0.0, -1.0, Some(&d));
        let r = l.1.as_ref().unwrap();
        assert!(r.fall.is_none());
        r.at(SHOWN_WHITE + 5.0)
    }

    #[test]
    fn a_raw_adds_the_raw_correction_to_the_curve() {
        // the Python reference (fit/rawgold10.py, from the r10-final tables) on `picture()` with a 1:1 display mapping, its
        // tables smoothed to the curvature cap ([`smooth_to_curvature`])
        let d = |e: f32| e;
        let s = Lift::new(&picture(), 1.0, 0.0, Some(&d));
        let h = Lift::new(&picture(), 0.0, -1.0, Some(&d));
        let b = Lift::new(&picture(), 1.0, -1.0, Some(&d));
        for (e, want_s, want_h, want_b) in GOLDEN_RAW {
            assert!((s.at_both(e) - want_s).abs() < 3e-3, "shadows at {e}: {} vs {want_s}", s.at_both(e));
            // (the pull past white is read at the narrow base, not in this table: [`Regional`])
            assert!((h.at_both(e) - want_h).abs() < 3e-3, "highlights at {e}: {} vs {want_h}", h.at_both(e));
            assert!((b.at_both(e) - want_b).abs() < 3e-3, "both at {e}: {} vs {want_b}", b.at_both(e));
        }
        // and a display-referred picture doesn't get it
        let tiff = Lift::new(&picture(), 1.0, 0.0, None);
        assert!((tiff.at_both(-5.0) - GOLDEN[0].1).abs() < 2e-3);
    }

    const GOLDEN_RAW: [(f32, f32, f32, f32); 4] = [
        (-5.0, 1.07532, -0.38338, -0.02791),
        (-2.0, 0.98369, -0.87188, -0.26515),
        (0.0, 0.39400, -1.33055, -0.65591),
        (1.5, 0.09293, -1.81559, -1.21474),
    ];

    #[test]
    fn a_raw_in_the_camera_curves_toe_shows_the_curves_shift_not_a_multiple() {
        // a camera curve with a steep toe: a scene EV below grey shows 2 EV darker per EV (kept above the darkest
        // shown tone the raw fit saw, where the curve is read)
        let d = |e: f32| if e < 0.0 { 2.0 * e } else { e };
        let t = picture();
        let raw = Lift::new(&t, 1.0, 0.0, Some(&d));
        let one = |e: f32| e;
        let shown = Lift::new(&t.mapped(d), 1.0, 0.0, Some(&one));
        let mut checked = 0;
        for i in 0..12 {
            let e = -3.0 + 0.25 * i as f32;
            let want = shown.at_both(d(e));
            if want < 0.2 {
                continue;
            }
            let got = d(e + raw.at_both(e)) - d(e);
            // (the table is smoothed in scene EV, twice as wide in shown EV here: within 0.2 EV, never the multiple)
            assert!((got - want).abs() < 0.2 && got < 1.5 * want, "{e}: shows {got}, the curve asks {want}");
            checked += 1;
        }
        assert!(checked >= 4, "the curve lifts the toe's tones: {checked} checked");
    }

    #[test]
    fn a_deep_tone_lifted_into_the_toe_shows_the_curves_lift() {
        // 1:1 under scene -6 EV, then a toe 2.5 EV per EV up to grey: a tone at -7 lifted into the toe used to get the
        // displayed lift as its scene shift (the slope where it starts is 1) and showed about twice the curve's lift
        let d = |e: f32| {
            if e < -6.0 {
                e - 9.0
            } else if e < 0.0 {
                2.5 * e
            } else {
                e
            }
        };
        for (e, shown) in [(-7.0f32, 3.5f32), (-6.5, 4.0), (-6.0, 3.0)] {
            let s = scene_shift(&d, e, shown);
            let moved = d(e + s) - d(e);
            assert!((moved - shown).abs() < 1e-3, "{e}: shows {moved}, the curve asks {shown}");
        }
        // past white the curve is flat: the shift goes to the scene tone as it is
        let flat = |e: f32| e.min(2.0);
        assert_eq!(scene_shift(&flat, 3.0, -1.0), -1.0);
    }

    #[test]
    fn a_dark_raws_deep_shadows_meet_the_curve_where_it_was_fitted() {
        // a dark raw (half its tones 11-15 EV under grey, darker than an 8-bit export shows): how deep those tones
        // sit doesn't change the curve, and a tone under the fit's darkest gets that tone's shift (before, the
        // curve's unfitted tail lifted such a frame's shadows 3-4 EV past Lightroom)
        let one = |e: f32| e;
        let pic = |deep: f32| {
            Tones::new((0..2000).map(|i| if i < 1000 { deep - 4.0 * (i as f32 / 1000.0) } else { -6.0 + 7.0 * ((i - 1000) as f32 / 1000.0) }))
                .unwrap()
        };
        let (a, b) = (Lift::new(&pic(-11.0), 1.0, 0.0, Some(&one)), Lift::new(&pic(-12.0), 1.0, 0.0, Some(&one)));
        for e in [-14.0, -11.0, -9.21, -6.0, -2.0, 0.0] {
            assert!((a.at_both(e) - b.at_both(e)).abs() < 1e-3, "{e}: {} vs {}", a.at_both(e), b.at_both(e));
        }
        // (the shift flattens out under the fit's darkest tone, -9.21, within the table's smoothing)
        assert!((a.at_both(-14.0) - a.at_both(-11.0)).abs() < 0.05, "{} vs {}", a.at_both(-14.0), a.at_both(-11.0));
        assert!(a.at_both(-14.0) < a.at_both(-9.21) + 0.35, "{} vs {}", a.at_both(-14.0), a.at_both(-9.21));
        assert!(a.at_both(-12.0) < 3.0, "the deep shadows lift {} EV", a.at_both(-12.0));
    }

    #[test]
    fn hostile_values_stay_finite() {
        let t = Tones::new([f32::NEG_INFINITY, -1e30, 0.0, 1e30, f32::NAN]).unwrap();
        for (s, h) in [(f32::NAN, 1.0), (1e9, -1e9), (1.0, -1.0)] {
            let l = Lift::new(&t, s, h, None);
            assert!(l.0.iter().all(|v| v.is_finite() && v.abs() <= MAX_SHIFT));
            assert!(l.at_both(f32::NAN).is_finite() && l.at_both(1e30).is_finite());
        }
    }
}
