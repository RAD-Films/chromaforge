//! A raw's Highlights − in process V9, read per pixel: the picture's pull ([`crate::tone_lift::pull_table`]) at the
//! tone of each pixel's surroundings, and a little more next to clipped areas.
//!
//! **The surroundings.** Lightroom Classic keeps the detail inside bright areas when it pulls them: clouds in a
//! partly clipped sky stay 11–14 L* above the sky between them at −100 (rounds 9b and 10 of our own Canon EOS R5 C
//! raws, OneSlips, Nov 2026), and a lamp's cone on a wall stays brighter than the wall. Read at each pixel's own
//! tone (or at a cloud-sized base), a pull that falls by about as much as the tone rises flattened the bright
//! band to one level: clouds went to 5–7 L*, the cone to a flat patch with a hard line where the band began. So
//! the pull is read at a wide base of the scene's log luminance, a bilateral grid [`WIDE`] of the frame's long
//! edge across that merges tones within about [`RANGE`] EV (sky with clouds, a cone with its wall) and keeps
//! larger steps (sky against ground) apart; the pixel's own detail over that base rides on top unchanged. Measured
//! at 1024 px against Lightroom's −100: clouds 0.5 L* under its contrast (was 2.3), the bright unclipped areas
//! 5.7 / 5.3 L* rms on fitted / held-out frames (was 6.4 / 5.9), the whole picture 3.6 / 3.1 (was 3.9 / 3.3).
//!
//! **Next to clipped areas.** Pixels at the edge of a clipped sky came out 2.5 L* lighter than Lightroom's −100,
//! a light rim around the clouds, while pixels 60 px away matched: Lightroom pulls bright tones near clipped areas
//! further, following the areas' outlines. The clipped areas (brightest channel past the camera curve's white)
//! are spread edge-aware, guided by log luminance, over a few hundredths of the frame, and bright tones within
//! [`GATE`] EV of white get up to [`AMOUNT`] EV more pull where the spread is, as far as their brightest channel has
//! reached white ([`AT_WHITE`]). Unclipped sky beside a clipped core doesn't take it: it left the rings 3–48 px from
//! the core 3–5 L* under Lightroom's −100, a blend that ended too soon, while partly clipped pixels need it.
//!
//! Both are taken once per photo on a fixed-size render of the whole frame ([`EDGE`] px long), so previews,
//! zoomed windows and exports agree, and read back at each output pixel from the four nearest grid cells, each by
//! how near it is and how close its surroundings' tone is to the pixel's own ([`EdgePull::at`]). Read bilinearly
//! alone, a dark pixel beside a clipped area took part of the area's much deeper pull: a dark olive line 1–2 px
//! wide (at 3000 px) along the edge, where Lightroom's −100 shows none.

use std::sync::{Arc, Mutex, PoisonError};

use lcc_develop::DevelopSettings;
use lcc_raster::{Plane, Rgb32f};

use crate::SourceInfo;
use crate::geometry::Frame;
use crate::local::{frame_proxy, log_lum, white_balance};
use crate::tone::{GREY, ToneMap};

/// Long edge of the render the clipped areas are found on.
pub const EDGE: usize = 1024;
/// The largest extra pull (EV) at Highlights −100.
pub const AMOUNT: f32 = 0.25;
/// How far the spread reaches: a share of the frame's long edge.
const RADIUS: f32 = 0.03;
/// The guided spread's edge threshold (log luminance, squared).
const LEAK: f32 = 0.1;
/// The spread counts fully from this share of clipped neighbours.
const FULL_AT: f32 = 0.5;
/// Tones from `GATE + 1` EV under white (none) to `GATE` EV under it (all) get the pull.
const GATE: f32 = 2.5;
/// The pull next to clipped areas reaches pixels whose brightest channel lies within this many EV of white (none) to
/// white (all).
const AT_WHITE: f32 = 0.5;
/// The surroundings' base: a bilateral grid this share of the long edge across (its Gaussian's sigma)…
const WIDE: f32 = 0.03;
/// …that merges tones within about this many EV (the width of its tone bins) and keeps larger steps apart.
const RANGE: f32 = 1.5;
/// The most tone bins the grid keeps (a picture with a wider range gets wider bins).
const MAX_BINS: usize = 48;
/// Reading the planes back, a grid cell counts less the further its surroundings' tone lies from the pixel's own: by a
/// Gaussian with this sigma (EV), half the grid's tone bin, so that cells across a step the grid keeps apart don't mix.
pub const READ_TONE: f32 = RANGE * 0.5;
/// …plus this much of its plain nearness (the Gaussian four sigmas out), so that where no cell's tone is near the
/// pixel's own (an edge pixel between a dark beam and the sky) the read falls back to nearness instead of letting the
/// least-far cell (the sky) win outright: that gave whole grid cells of edge pixels the sky's pull, black squares.
pub const READ_FLOOR: f32 = 3.354_626e-4;
/// Across a step in the surroundings (the darkest and brightest surroundings near a pixel at least about this many EV
/// apart, fully from half an EV further: a dark beam against a clipped sky) an edge pixel's pull grades with its own
/// tone, so the edge pixels between the two sides don't each take one side's pull (that left one row brighter than the
/// sky beside one darker than the beam). Smaller steps (a lamp's ring, a specular streak on metal) keep the read by
/// tone, which follows the reference closer there…
pub const STEP_GAP: f32 = 4.0;
/// …where a pixel's own tone counts as itself up to this many EV below the brightest surroundings…
pub const STEP_RAMP: f32 = 2.9;
/// …and as the brightest surroundings from this many EV below them (sky a little darker than its surroundings keeps
/// the sky's pull).
pub const STEP_TOP: f32 = 0.3;

/// The local part of one photo's Highlights pull: planes over the whole output frame.
#[derive(Clone, Debug, PartialEq)]
pub struct EdgePull {
    /// The pull (EV, ≤ 0) at Highlights −100, on the whole frame at [`EDGE`] px long.
    pub plane: Arc<Plane>,
    /// The surroundings' base tone (EV over grey, exposure included), on the same grid.
    pub wide: Arc<Plane>,
    /// The picture's pull at the slider ([`crate::tone_lift`]'s grid), read at the surroundings' tone.
    pub pull: Arc<crate::tone_lift::Lift>,
    /// The Highlights slider's share of the pull next to clipped areas.
    pub gain: f32,
    /// Plane pixels per output pixel, `(x, y)`.
    pub scale: (f32, f32),
    /// Where the rendered pixels sit in the whole output frame.
    pub origin: (usize, usize),
    /// The darkest and the brightest surroundings' tone within a cell of each grid cell (from `wide`), read
    /// bilinearly so they change smoothly from pixel to pixel ([`STEP_GAP`]).
    pub span: Arc<(Plane, Plane)>,
}

impl EdgePull {
    /// The pull from its planes, with the span of the surroundings worked out from `wide`.
    pub fn new(plane: Arc<Plane>, wide: Arc<Plane>, pull: Arc<crate::tone_lift::Lift>, gain: f32, scale: (f32, f32), origin: (usize, usize)) -> Self {
        let span = Arc::new(span_of(&wide));
        Self { plane, wide, pull, gain, scale, origin, span }
    }

    /// The Highlights pull (EV, ≤ 0) at rendered pixel `(x, y)`, whose tone a mask moved by `shift` EV and whose own
    /// base tone (EV over grey, exposure and `shift` included) is `own`. The four nearest grid cells count by how near
    /// they are and by how close their surroundings' tone is to `own` ([`READ_TONE`]), so a pixel beside a much
    /// brighter area (a clipped one) takes the pull of its own side; when none is close, by nearness ([`READ_FLOOR`]).
    ///
    /// Across a step in the surroundings ([`STEP_GAP`]) the weights are not scaled up to a whole: what the cells near
    /// the pixel's tone leave goes to the pixel's own tone, raised towards the brightest surroundings as it nears them
    /// ([`STEP_RAMP`], [`STEP_TOP`]). An edge pixel between a beam and the sky then takes a pull graded by its tone,
    /// and a cell joining the four can't take over at once (whole grid cells of one side's pull: squares and blocks).
    pub fn at(&self, x: usize, y: usize, shift: f32, own: f32) -> f32 {
        let Some(cells) = self.cells(x, y) else { return 0.0 };
        let wide_at = |i: usize| self.wide.data.get(i).copied().unwrap_or(0.0);
        let plane_at = |i: usize| self.plane.data.get(i).copied().unwrap_or(0.0);
        let guided = !self.wide.data.is_empty() && own.is_finite();
        // how close each cell's surroundings' tone is to the pixel's
        let near = cells.map(|(i, _)| {
            let z = if guided { ((wide_at(i) + shift - own) / READ_TONE).abs().min(8.0) } else { 0.0 };
            (-0.5 * z * z).exp()
        });
        let tone = cells.iter().zip(&near).map(|(&(_, b), &g)| {
            let v = b * (g + READ_FLOOR);
            if v.is_finite() { v } else { 0.0 }
        });
        let tone: Vec<f32> = tone.collect();
        let sum: f32 = tone.iter().sum();
        let w: Vec<f32> = if sum > 1e-30 { tone.iter().map(|v| v / sum).collect() } else { cells.iter().map(|c| c.1).collect() };
        let read = |f: &dyn Fn(usize) -> f32| cells.iter().zip(&w).map(|(&(i, _), &k)| k * f(i)).sum::<f32>();
        let (mut surround, mut edge) = (read(&|i| wide_at(i) + shift), read(&plane_at));
        let (lo_p, hi_p) = (&self.span.0, &self.span.1);
        if guided && lo_p.data.len() == self.wide.data.len() && hi_p.data.len() == self.wide.data.len() {
            let bil = |f: &dyn Fn(usize) -> f32| cells.iter().map(|&(i, b)| b * f(i)).sum::<f32>();
            let hi = bil(&|i| hi_p.data.get(i).copied().unwrap_or(0.0)) + shift;
            // each cell's step, read bilinearly: the rule fades in over a whole cell, not along a line where the gap
            // passes the threshold
            let mix = bil(&|i| {
                let gap = hi_p.data.get(i).copied().unwrap_or(0.0) - lo_p.data.get(i).copied().unwrap_or(0.0);
                smooth(STEP_GAP - 0.5, STEP_GAP + 0.5, gap)
            });
            if mix > 0.0 {
                let held = cells.iter().zip(&near).map(|(&(_, b), &g)| b * g);
                let held: Vec<f32> = held.collect();
                let rest = (1.0 - held.iter().sum::<f32>()).clamp(0.0, 1.0);
                let own_side = own + (hi - own) * smooth(hi - STEP_RAMP, hi - STEP_TOP, own);
                let by = |f: &dyn Fn(usize) -> f32| cells.iter().zip(&held).map(|(&(i, _), &k)| k * f(i)).sum::<f32>();
                let s = by(&|i| wide_at(i) + shift) + rest * own_side;
                let e = by(&plane_at) + rest * bil(&plane_at);
                if s.is_finite() && e.is_finite() {
                    surround += (s - surround) * mix;
                    edge += (e - edge) * mix;
                }
            }
        }
        let edge = if self.plane.data.is_empty() { 0.0 } else { self.gain * edge };
        let wide = if self.wide.data.is_empty() { 0.0 } else { self.pull.at(surround) };
        let v = edge + wide;
        if v.is_finite() { v } else { 0.0 }
    }

    /// The four grid cells around rendered pixel `(x, y)`: their index in the planes and bilinear weight, or `None`
    /// on an empty grid.
    fn cells(&self, x: usize, y: usize) -> Option<[(usize, f32); 4]> {
        let plane = if self.wide.data.is_empty() { &self.plane } else { &self.wide };
        let (pw, ph) = (plane.width, plane.height);
        if pw == 0 || ph == 0 {
            return None;
        }
        let fx = ((self.origin.0.saturating_add(x) as f32 + 0.5) * self.scale.0 - 0.5).clamp(0.0, (pw - 1) as f32);
        let fy = ((self.origin.1.saturating_add(y) as f32 + 0.5) * self.scale.1 - 0.5).clamp(0.0, (ph - 1) as f32);
        let (fx, fy) = (if fx.is_finite() { fx } else { 0.0 }, if fy.is_finite() { fy } else { 0.0 });
        let (x0, y0) = (fx as usize, fy as usize);
        let (x1, y1) = ((x0 + 1).min(pw - 1), (y0 + 1).min(ph - 1));
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        Some([(y0 * pw + x0, (1.0 - tx) * (1.0 - ty)), (y0 * pw + x1, tx * (1.0 - ty)), (y1 * pw + x0, (1.0 - tx) * ty), (y1 * pw + x1, tx * ty)])
    }
}

/// The darkest and the brightest value of `wide` within one cell of each cell (non-finite values left out; 0 where a
/// cell has none).
fn span_of(wide: &Plane) -> (Plane, Plane) {
    let (w, h) = (wide.width, wide.height);
    let (mut lo, mut hi) = (Plane::new(w, h), Plane::new(w, h));
    for y in 0..h {
        for x in 0..w {
            let (mut a, mut b) = (f32::MAX, f32::MIN);
            for yy in y.saturating_sub(1)..(y + 2).min(h) {
                for xx in x.saturating_sub(1)..(x + 2).min(w) {
                    if let Some(&v) = wide.data.get(yy * w + xx)
                        && v.is_finite()
                    {
                        a = a.min(v);
                        b = b.max(v);
                    }
                }
            }
            let i = y * w + x;
            let (a, b) = if a <= b { (a, b) } else { (0.0, 0.0) };
            if let (Some(l), Some(u)) = (lo.data.get_mut(i), hi.data.get_mut(i)) {
                *l = a;
                *u = b;
            }
        }
    }
    (lo, hi)
}

fn smooth(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    if t.is_finite() { t * t * (3.0 - 2.0 * t) } else { 0.0 }
}

/// The scene tone (EV over grey) where `tm`'s displayed tone reaches its top, within 0.05 EV.
pub(crate) fn white_of(tm: &ToneMap) -> Option<f32> {
    let d = |e: f32| (tm.apply(GREY * e.exp2()).max(1e-9) / GREY).log2();
    let top = d(12.0);
    (0..=640).map(|i| -8.0 + i as f32 / 32.0).find(|&e| d(e) >= top - 0.05)
}

/// The pull next to clipped areas at Highlights −100 for a white-balanced, exposed `img` (scene linear, exposure
/// `ev` still to add) whose camera curve reaches white at scene tone `white`, with `ppl` pixels per long edge.
#[cfg(test)]
fn unit_plane(img: &Rgb32f, ev: f32, white: f32, ppl: f64) -> Plane {
    unit_planes(img, ev, Some(white), ppl).0
}

/// [`unit_plane`] (zero without a `white`) and the surroundings' base tone (EV over grey, exposure `ev` added).
fn unit_planes(img: &Rgb32f, ev: f32, white: Option<f32>, ppl: f64) -> (Plane, Plane) {
    let log_l = img.map(log_lum);
    // the surroundings without lone pixels (a pixel at the edge of a clipped area that colour reconstruction left at
    // or below black, −20 EV): the grid kept each as surroundings of its own, and edge pixels read them
    let wide = bilateral(&crate::local::median_hv(&log_l, 1), WIDE * ppl as f32, RANGE).map(|v| if (v + ev).is_finite() { v + ev } else { 0.0 });
    let mut plane = Plane::new(img.width, img.height);
    let Some(white) = white else { return (plane, wide) };
    let mut clipped = Plane::new(img.width, img.height);
    for (o, c) in clipped.data.iter_mut().zip(&img.data) {
        let m = c[0].max(c[1]).max(c[2]);
        *o = if log_lum([m, m, m]) + ev >= white { 1.0 } else { 0.0 };
    }
    let near = crate::masks::guided_cross(&log_l, &clipped, (RADIUS * ppl as f32).max(1.0), LEAK);
    for (((o, n), l), c) in plane.data.iter_mut().zip(&near.data).zip(&log_l.data).zip(&img.data) {
        let top = c[0].max(c[1]).max(c[2]);
        let w = (n / FULL_AT).clamp(0.0, 1.0) * smooth(white - AT_WHITE, white, log_lum([top, top, top]) + ev);
        let v = -AMOUNT * w * smooth(white - GATE - 1.0, white - GATE, l + ev);
        *o = if v.is_finite() { v } else { 0.0 };
    }
    (plane, wide)
}

/// A bilateral grid of `l`: each pixel's tone splatted linearly into tone bins `range` wide (wider when the
/// picture's tones need more than [`MAX_BINS`]), each bin's weights and tones blurred with a Gaussian of `sigma`
/// px and softened across neighbouring bins (1, 2, 1), and read back at the pixel's own tone. Tones within a bin
/// or two of each other mix; farther apart they stay apart. A pixel with nothing around keeps its tone.
fn bilateral(l: &Plane, sigma: f32, range: f32) -> Plane {
    let fin = |v: f32| if v.is_finite() { v.clamp(-32.0, 32.0) } else { 0.0 };
    let (lo, hi) = l.data.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(fin(v)), b.max(fin(v))));
    let mut out = Plane::new(l.width, l.height);
    if l.data.is_empty() || lo > hi {
        return out;
    }
    let step = range.max((hi - lo) / (MAX_BINS - 2) as f32).max(1e-3);
    let n = (((hi - lo) / step).ceil() as usize + 2).clamp(2, MAX_BINS);
    // the bin below a tone and how far it sits towards the next
    let bin = |v: f32| {
        let t = ((fin(v) - lo) / step).clamp(0.0, (n - 1) as f32);
        let b = (t as usize).min(n - 2);
        (b, (t - b as f32).clamp(0.0, 1.0))
    };
    let mut grid: Vec<(Plane, Plane)> = (0..n).map(|_| (Plane::new(l.width, l.height), Plane::new(l.width, l.height))).collect();
    for (i, &v) in l.data.iter().enumerate() {
        let (b, f) = bin(v);
        for (k, wt) in [(b, 1.0 - f), (b + 1, f)] {
            if let Some((w, t)) = grid.get_mut(k)
                && let (Some(w), Some(t)) = (w.data.get_mut(i), t.data.get_mut(i))
            {
                *w += wt;
                *t += wt * fin(v);
            }
        }
    }
    let grid: Vec<(Plane, Plane)> = grid.iter().map(|(w, t)| (lcc_raster::blur::gaussian(w, sigma), lcc_raster::blur::gaussian(t, sigma))).collect();
    // bin `k` at pixel `i`, softened over its neighbours
    let read = |k: usize, i: usize| {
        let (mut sw, mut st) = (0.0f32, 0.0f32);
        for (d, wt) in [(-1i64, 0.25f32), (0, 0.5), (1, 0.25)] {
            let Some((w, t)) = usize::try_from(k as i64 + d).ok().and_then(|kk| grid.get(kk)) else { continue };
            sw += wt * w.data.get(i).copied().unwrap_or(0.0);
            st += wt * t.data.get(i).copied().unwrap_or(0.0);
        }
        (sw, st)
    };
    for (i, (o, &v)) in out.data.iter_mut().zip(&l.data).enumerate() {
        let (b, f) = bin(v);
        let ((w0, t0), (w1, t1)) = (read(b, i), read(b + 1, i));
        let (sw, st) = (w0 * (1.0 - f) + w1 * f, t0 * (1.0 - f) + t1 * f);
        let r = if sw > 1e-6 { st / sw } else { fin(v) };
        *o = if r.is_finite() { r } else { fin(v) };
    }
    out
}

type Cached = (u64, Arc<Plane>, Arc<Plane>);
static LAST: Mutex<Option<Cached>> = Mutex::new(None);

/// The extra Highlights pull of `s` for a render of `full` output pixels (the rendered ones from `origin`) of the
/// whole `frame` of `src`, or `None` when it has none: not process V9's Highlights ([`crate::tone_lift::active`]),
/// Highlights not below 0, not a raw (`raw_tone`, its default tone map, `None`), or nothing to measure.
pub fn for_frame(
    src: &Rgb32f,
    info: &SourceInfo,
    s: &DevelopSettings,
    frame: &Frame,
    full: (usize, usize),
    origin: (usize, usize),
    raw_tone: Option<&ToneMap>,
) -> Option<Arc<EdgePull>> {
    let tm = raw_tone?;
    let gain = crate::tone_lift::highlights_down(s)?;
    let (pw, ph) = frame.fit(EDGE, EDGE);
    if pw == 0 || ph == 0 || full.0 == 0 || full.1 == 0 || src.width == 0 || src.height == 0 {
        return None;
    }
    // (without a white the pull next to clipped areas is off; the pull itself stays)
    let white = white_of(tm);
    let ev = s.light.exposure as f32;
    let key = {
        use std::hash::{BuildHasher, BuildHasherDefault, DefaultHasher};
        let (t, tint) = crate::local::effective_wb(info, s);
        BuildHasherDefault::<DefaultHasher>::default().hash_one((
            (pw, ph),
            crate::optics::fingerprint(src),
            format!("{frame:?}"),
            [t.to_bits(), tint.to_bits(), info.as_shot_temp.to_bits(), info.as_shot_tint.to_bits()],
            crate::local::white_balance_key(info, s),
            ev.to_bits(),
            white.map(f32::to_bits),
        ))
    };
    let pull = Arc::new(crate::tone_lift::pull_table(src, info, s, frame, white)?);
    let cached = LAST.lock().unwrap_or_else(PoisonError::into_inner).as_ref().filter(|(k, ..)| *k == key).map(|(_, p, w)| (p.clone(), w.clone()));
    let (plane, wide) = match cached {
        Some(p) => p,
        None => {
            let mut img = Rgb32f::clone(&frame_proxy(src, frame, pw, ph));
            white_balance(&mut img, info, s);
            let (p, w) = unit_planes(&img, ev, white, frame.px_per_long(pw));
            let (p, w) = (Arc::new(p), Arc::new(w));
            *LAST.lock().unwrap_or_else(PoisonError::into_inner) = Some((key, p.clone(), w.clone()));
            (p, w)
        }
    };
    let scale = (pw as f32 / full.0 as f32, ph as f32 / full.1 as f32);
    Some(Arc::new(EdgePull::new(plane, wide, pull, gain, scale, origin)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dark ground under a sky that brightens from 0.5 EV under white (left) through white into a clipped right
    /// part, as a cloud does: tones just under white beside the clipped part are pulled, the same sky far from it
    /// barely, the ground not at all.
    #[test]
    fn bright_tones_next_to_clipped_areas_get_pulled() {
        let (w, h) = (512usize, 128usize);
        let white = 2.0f32;
        let tone = |x: usize, y: usize| {
            if y >= 96 { -4.0 } else { white - 0.5 + 1.5 * ((x as f32 - 256.0) / 128.0).clamp(0.0, 1.0) }
        };
        // scene linear: grey · 2^tone
        let img = Rgb32f::from_fn(w, h, |x, y| [GREY * tone(x, y).exp2(); 3]);
        let p = unit_plane(&img, 0.0, white, w as f64);
        let at = |x: usize, y: usize| p.data[y * w + x];
        assert!(tone(290, 40) < white);
        assert!(at(290, 40) < -0.5 * AMOUNT, "next to the clip: {}", at(290, 40));
        assert!(at(20, 40) > -0.1 * AMOUNT, "far from it: {}", at(20, 40));
        assert!(at(330, 120) == 0.0, "dark ground: {}", at(330, 120));
        assert!(p.data.iter().all(|v| v.is_finite() && *v <= 0.0 && *v >= -AMOUNT));
    }

    /// Sky a full stop under white right beside a clipped area takes none of the pull next to it: that pull made the
    /// recovered rings around a clipped core too dark, so the blend ended too soon.
    #[test]
    fn unclipped_tones_beside_a_clipped_area_get_no_extra_pull() {
        let (w, h) = (512usize, 128usize);
        let white = 2.0f32;
        let img = Rgb32f::from_fn(w, h, |x, _| [GREY * (if x < 256 { white - 1.0 } else { white + 1.0 }).exp2(); 3]);
        let p = unit_plane(&img, 0.0, white, w as f64);
        let at = |x: usize| p.data[64 * w + x];
        assert_eq!(at(250), 0.0, "beside the clip: {}", at(250));
        assert!(at(262) < -0.5 * AMOUNT, "in the clip: {}", at(262));
    }

    #[test]
    fn reads_back_bilinearly_and_never_off_the_plane() {
        let mut plane = Plane::new(2, 1);
        plane.data = vec![0.0, -1.0];
        let none = || Arc::new(Plane::new(0, 0));
        let no_pull = Arc::new(crate::tone_lift::Lift(Vec::new()));
        let e = EdgePull::new(Arc::new(plane), none(), no_pull.clone(), 0.5, (0.5, 0.5), (0, 0));
        assert_eq!(e.at(0, 0, 0.0, 0.0), 0.0);
        assert!((e.at(2, 0, 0.0, 0.0) + 0.375).abs() < 1e-6, "{}", e.at(2, 0, 0.0, 0.0));
        assert_eq!(e.at(usize::MAX, usize::MAX, f32::NAN, f32::NAN), -0.5);
        let empty = EdgePull::new(none(), none(), no_pull, 1.0, (1.0, 1.0), (0, 0));
        assert_eq!(empty.at(3, 3, 0.0, 0.0), 0.0);
    }

    /// The pull is read at the surroundings' tone, whatever the pixel's own: a bright patch in darker surroundings
    /// gets the surroundings' (smaller) pull and keeps its contrast over them.
    #[test]
    fn the_pull_is_read_at_the_surroundings_tone() {
        use crate::tone_lift::{LIFT_LO, LIFT_N, LIFT_STEP, Lift};
        // a pull of −0.5 EV per EV above 0 EV
        let pull = Lift((0..LIFT_N).map(|i| -0.5 * (LIFT_LO + LIFT_STEP * i as f32).max(0.0)).collect());
        let flat = |v: f32| {
            let mut p = Plane::new(4, 4);
            p.data.fill(v);
            Arc::new(p)
        };
        let e = |w: f32| EdgePull::new(flat(0.0), flat(w), Arc::new(pull.clone()), 1.0, (1.0, 1.0), (0, 0));
        assert!((e(2.0).at(1, 1, 0.0, 3.0) + 1.0).abs() < 1e-4, "{}", e(2.0).at(1, 1, 0.0, 3.0));
        assert!(e(-2.0).at(1, 1, 0.0, -1.0).abs() < 1e-6);
    }

    /// Beside a much brighter area on the grid (a clipped core at 3 EV, its pull −1.5 EV), a dark pixel (at −1 EV)
    /// takes the pull of its own side, where a bilinear read gave it about half the core's: a dark line along the
    /// edge. A pixel as bright as the core still takes the core's.
    #[test]
    fn a_dark_pixel_beside_a_bright_area_takes_its_own_sides_pull() {
        use crate::tone_lift::{LIFT_LO, LIFT_N, LIFT_STEP, Lift};
        let pull = Lift((0..LIFT_N).map(|i| -0.5 * (LIFT_LO + LIFT_STEP * i as f32).max(0.0)).collect());
        let mut wide = Plane::new(2, 1);
        wide.data = vec![-1.0, 3.0];
        let e = EdgePull::new(Arc::new(Plane::new(2, 1)), Arc::new(wide), Arc::new(pull), 1.0, (0.5, 0.5), (0, 0));
        // output pixel 2 sits halfway between the two cells
        assert!(e.at(2, 0, 0.0, -1.0).abs() < 0.01, "dark pixel: {}", e.at(2, 0, 0.0, -1.0));
        assert!((e.at(2, 0, 0.0, 3.0) + 1.5).abs() < 0.01, "bright pixel: {}", e.at(2, 0, 0.0, 3.0));
        assert!(e.at(2, 0, 0.0, f32::NAN).is_finite());
    }

    /// An edge pixel (−1.5 EV) whose four cells are a dark beam (−20.8 and −7.1 EV, far below it) and the clipped sky
    /// (2.5 and 2.8 EV, far above it) has no cell near its tone: it gets about no pull, where taking the least-far
    /// cell's (the sky's, −1.3 EV) gave whole grid cells along the edge black squares. A pixel as bright as the sky
    /// still takes the sky's pull.
    #[test]
    fn an_edge_pixel_with_no_cell_near_its_tone_takes_no_sky_pull() {
        use crate::tone_lift::{LIFT_LO, LIFT_N, LIFT_STEP, Lift};
        let pull = Lift((0..LIFT_N).map(|i| -0.5 * (LIFT_LO + LIFT_STEP * i as f32).max(0.0)).collect());
        let mut wide = Plane::new(2, 2);
        wide.data = vec![-20.8, -7.1, 2.5, 2.8];
        let e = EdgePull::new(Arc::new(Plane::new(2, 2)), Arc::new(wide), Arc::new(pull), 1.0, (0.5, 0.5), (0, 0));
        // output pixel (1, 1) sits in the middle of the four cells
        assert!(e.at(1, 1, 0.0, -1.5) > -0.1, "edge pixel: {}", e.at(1, 1, 0.0, -1.5));
        assert!((e.at(1, 1, 0.0, 2.65) + 1.32).abs() < 0.1, "sky pixel: {}", e.at(1, 1, 0.0, 2.65));
    }

    /// A lone pixel far below its neighbours (one colour reconstruction left at black beside a clipped area) is no
    /// surroundings of its own: the plane there holds its neighbours' tone.
    #[test]
    fn a_lone_black_pixel_is_no_surroundings_of_its_own() {
        let mut img = Rgb32f::new(64, 64);
        img.data.fill([0.1, 0.1, 0.1]);
        if let Some(p) = img.data.get_mut(32 * 64 + 32) {
            *p = [0.0, 0.0, 0.0];
        }
        let (_, wide) = unit_planes(&img, 0.0, None, 64.0);
        let at = |x: usize, y: usize| wide.data.get(y * 64 + x).copied().unwrap_or(f32::NAN);
        assert!((at(32, 32) - at(10, 10)).abs() < 0.1, "lone pixel {} vs field {}", at(32, 32), at(10, 10));
    }

    /// On the row where a dark beam (−0.5 EV) meets the clipped sky (2.9 EV) the pull grades with the pixel's own tone:
    /// a pixel a third of the way up takes a little of the sky's pull and one two thirds up most of it, where the read
    /// by tone gave the first the beam's pull and the second the sky's (one row brighter than the sky, the next darker
    /// than the beam). The beam and the sky keep their own pulls, and the pull changes smoothly along the edge.
    #[test]
    fn edge_pixels_between_a_beam_and_the_sky_grade_their_pull() {
        use crate::tone_lift::{LIFT_LO, LIFT_N, LIFT_STEP, Lift};
        let pull = Lift((0..LIFT_N).map(|i| -0.5 * (LIFT_LO + LIFT_STEP * i as f32).max(0.0)).collect());
        let p = |e: f32| pull.at(e);
        let grid = |dark: f32, bright: f32| {
            let mut wide = Plane::new(4, 4);
            for (i, v) in wide.data.iter_mut().enumerate() {
                *v = if i % 4 < 2 { dark } else { bright };
            }
            EdgePull::new(Arc::new(Plane::new(4, 4)), Arc::new(wide), Arc::new(pull.clone()), 1.0, (0.0625, 0.0625), (0, 0))
        };
        // a 4×4 grid: the beam's cells (−1.5 EV) on the left two columns, the clipped sky's (3.5 EV) on the right two;
        // output pixel 32 (of 64, 16 per cell) sits halfway between the beam's last column and the sky's first
        let e = grid(-1.5, 3.5);
        let at = |x: usize, own: f32| e.at(x, 24, 0.0, own);
        assert!((at(32, -1.5) - p(-1.5)).abs() < 0.05, "beam: {}", at(32, -1.5));
        assert!((at(32, 3.5) - p(3.5)).abs() < 0.05, "sky: {}", at(32, 3.5));
        assert!((at(32, 3.3) - p(3.5)).abs() < 0.05, "sky a little darker: {}", at(32, 3.3));
        let (a, b) = (at(32, 0.2), at(32, 1.8));
        assert!(a < p(-1.5) - 0.05 && a > p(3.5) * 0.5, "a third up: {a}");
        assert!(b < p(3.5) * 0.5 && b > p(3.5), "two thirds up: {b}");
        // graded with the tone, and smooth along the grid for an edge pixel
        let steps: Vec<f32> = (0..=50).map(|i| at(32, -1.5 + 0.1 * i as f32)).collect();
        assert!(steps.windows(2).all(|w| w[1] <= w[0] + 1e-4 && w[0] - w[1] < 0.15), "{steps:?}");
        let along: Vec<f32> = (16..=48).map(|x| at(x, 1.0)).collect();
        assert!(along.windows(2).all(|w| (w[1] - w[0]).abs() < 0.25), "{along:?}");
        // a smaller step (a lamp's ring) keeps the read by tone: a pixel near the dark side's tone takes its pull
        let ring = grid(-0.5, 2.9);
        let r = ring.at(32, 24, 0.0, 0.0);
        assert!((r - p(-0.5)).abs() < 0.05, "ring: {r}");
    }

    /// The span planes hold each cell's darkest and brightest neighbour and stay finite on hostile values.
    #[test]
    fn the_span_holds_the_darkest_and_brightest_neighbours() {
        let mut wide = Plane::new(3, 1);
        wide.data = vec![1.0, f32::NAN, -2.0];
        let (lo, hi) = span_of(&wide);
        assert_eq!(lo.data, vec![1.0, -2.0, -2.0]);
        assert_eq!(hi.data, vec![1.0, 1.0, -2.0]);
        let (lo, hi) = span_of(&Plane::new(0, 0));
        assert!(lo.data.is_empty() && hi.data.is_empty());
    }

    /// The bilateral base merges tones a little apart (clouds in a sky) and keeps a large step (sky against ground)
    /// apart, and stays finite on hostile values.
    #[test]
    fn the_surroundings_merge_clouds_and_keep_the_horizon() {
        let (w, h) = (200usize, 200usize);
        let mut l = Plane::new(w, h);
        for y in 0..h {
            for x in 0..w {
                // sky at 2 EV with 0.8 EV clouds in a checker of 20 px cells; ground at -4 EV below row 150
                l.data[y * w + x] = if y >= 150 {
                    -4.0
                } else if (x / 20 + y / 20) % 2 == 0 {
                    2.8
                } else {
                    2.0
                };
            }
        }
        let b = bilateral(&l, 12.0, RANGE);
        let at = |x: usize, y: usize| b.data[y * w + x];
        // cloud and sky cells read nearly the same base, between their tones
        assert!((at(10, 50) - at(30, 50)).abs() < 0.15, "{} {}", at(10, 50), at(30, 50));
        assert!(at(30, 50) > 2.1 && at(10, 50) < 2.7);
        // the sky right above the ground keeps a sky base, the ground a ground base
        assert!(at(100, 145) > 1.5, "{}", at(100, 145));
        assert!(at(100, 155) < -3.0, "{}", at(100, 155));
        let mut bad = Plane::new(3, 1);
        bad.data = vec![f32::NAN, f32::INFINITY, -1e30];
        assert!(bilateral(&bad, 2.0, RANGE).data.iter().all(|v| v.is_finite()));
        assert!(bilateral(&Plane::new(0, 0), 2.0, RANGE).data.is_empty());
    }
}
