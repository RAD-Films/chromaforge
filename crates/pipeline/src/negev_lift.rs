//! Process V9's highlight lift under a negative Exposure, for raws: bright tones near the sensor's clip are rendered
//! lighter than a plain gain puts them, as Lightroom Classic does.
//!
//! **Measured.** Lightroom Classic exports of our own Canon EOS R5 C raws (rounds 10, 23 and 24, OneSlips, Nov 2026),
//! 90 frames at Highlights 0, each at the white balance and Exposure its export records, against our render (radfilms
//! fix/hl-gradient 5043d7399). At Exposure −1 and −3 the gap depends on how far a pixel's brightest raw channel lies
//! below the clip (its headroom), not on how far it lies from a clipped area: about +2.2 L* at the clip, +1.6 at
//! half a stop under it, +1.0 at one stop, +0.9 at two, +0.5 at 3.5, nothing from 4.5. Within each headroom band the gap
//! was the same 4 px and 440 px (at 1500 px) from a clipped area, on either side of an object's outline, beside small
//! and large clipped areas; C* within 1. At Exposure 0 it is +0.2..+0.8 L*, the picture's global offset, so nothing
//! changes there: this lift exists only below Exposure 0 (fully from −1 EV, in proportion above it).
//!
//! **Where.** The decode turns raw levels into scene-linear colour before the pipeline, so a pixel's headroom is read
//! the way [`crate::hl_edge`] finds clipped areas: its brightest white-balanced channel, before the Exposure, against
//! the scene tone where the camera's tone curve reaches white (the sensor's clip renders white at Exposure 0). The read
//! is the white-balanced colour the finish stage starts from (`img`, after the decode's near-clip colour and before
//! the Exposure), so previews, zoomed windows and exports agree; the lift is applied at the tone map, the one place a
//! scene tone becomes a display lightness: the colour is tone mapped as it is and with the lift's EV gain, and the
//! result takes the lifted luminance with the unlifted colour ([`keep_colour`]). An EV gain alone pushed bright warm
//! colours into the curve's shoulder, where they paled: on a −2 EV lamp glow C* fell 1.6 under Lightroom's, which keeps
//! it. Each headroom's lift in L* is turned into EV once per render on the picture's base tone map
//! ([`NegEvLift::new`]), at the scene tone the pixel's luminance has under the Exposure: a saturated colour's luminance
//! lies below its brightest channel (the gap, [`GAP_STEP`]), on a steeper part of the curve, so it takes less EV for the
//! same L*. Pixels past the clip (rebuilt highlights, headroom < 0) take the clip's lift.

use crate::tone::{GREY, ToneMap};

/// The table: headroom (EV under the camera curve's white) from 0 in steps of [`STEP`], [`N`] entries.
pub const N: usize = 41;
pub const STEP: f32 = 0.125;
/// The gap between a pixel's brightest channel and its Rec.2020 luminance (EV) from 0 in steps of [`GAP_STEP`],
/// [`GAPS`] entries (more is held at the last).
pub const GAPS: usize = 7;
pub const GAP_STEP: f32 = 0.5;
/// The most a pixel is lifted (EV).
const MAX_EV: f32 = 2.0;

/// The lift (L*) by headroom (EV), measured at Exposure −1 to −3 (see the module docs); linear between the points.
const TARGET: [(f32, f32); 6] = [(0.0, 2.2), (0.5, 1.6), (1.0, 1.0), (2.0, 0.9), (3.5, 0.5), (4.5, 0.0)];

/// One render's lift: the scene tone (EV over grey, Exposure 0) where the camera curve reaches white, and the lift (EV)
/// at each headroom (rows, [`N`]) and gap (columns, [`GAPS`]): `table[i * GAPS + j]`.
#[derive(Clone, Debug, PartialEq)]
pub struct NegEvLift {
    pub white: f32,
    pub table: Vec<f32>,
}

/// The lift in L* at `headroom` EV under white ([`TARGET`]).
fn target(headroom: f32) -> f32 {
    let h = headroom.max(0.0);
    for w in TARGET.windows(2) {
        let ((h0, l0), (h1, l1)) = (w[0], w[1]);
        if h <= h1 {
            return l0 + (l1 - l0) * ((h - h0) / (h1 - h0)).clamp(0.0, 1.0);
        }
    }
    0.0
}

/// CIE L* of a display-linear luminance (1 = white).
fn lstar(y: f32) -> f32 {
    let y = y.clamp(0.0, 1.0);
    if y > 0.008_856 { 116.0 * y.cbrt() - 16.0 } else { 903.3 * y }
}

/// The lifted pixel: the display colour `plain` (tone mapped without the lift) brightened to the luminance of `lifted`
/// (the same colour tone mapped with it), so the lift raises lightness and keeps the colour, as Lightroom Classic's
/// does, instead of pushing a bright colour into the tone curve's shoulder, where it pales. Where that would take a
/// channel past white the result moves over to `lifted` (fully by 5 % past white).
pub fn keep_colour(plain: [f32; 3], lifted: [f32; 3]) -> [f32; 3] {
    let (y0, y1) = (lcc_color::luminance_2020(plain), lcc_color::luminance_2020(lifted));
    let top = plain[0].max(plain[1]).max(plain[2]);
    if !y0.is_finite() || !y1.is_finite() || !top.is_finite() || y0 <= 1e-9 || y1 <= y0 {
        return lifted;
    }
    let k = y1 / y0;
    let w = ((top * k - 0.95) / 0.1).clamp(0.0, 1.0);
    std::array::from_fn(|n| plain[n] * k + (lifted[n] - plain[n] * k) * w)
}

impl NegEvLift {
    /// The lift for Exposure `ev` on the base tone map `tone`; `None` at Exposure 0 and above, without a white, or
    /// when the table has nothing to add.
    pub fn new(tone: &ToneMap, ev: f32) -> Option<NegEvLift> {
        if !ev.is_finite() || ev >= 0.0 {
            return None;
        }
        let white = crate::hl_edge::white_of(tone)?;
        let strength = (-ev).min(1.0);
        let shown = |t: f32| lstar(tone.apply(GREY * t.exp2()));
        let table: Vec<f32> = (0..N * GAPS)
            .map(|k| {
                let h = (k / GAPS) as f32 * STEP;
                let gap = (k % GAPS) as f32 * GAP_STEP;
                let want = strength * target(h);
                // the scene tone of the luminance of a pixel with this headroom and gap under the Exposure
                let t = white - h - gap + ev;
                let l0 = shown(t);
                // (near white, only as far as the curve still rises within the solve's reach)
                let want = want.min(shown(t + MAX_EV) - l0);
                if want.is_nan() || want <= 0.0 || !l0.is_finite() { 0.0 } else { Self::solve(&shown, t, l0, want) }
            })
            .collect();
        let finite = table.iter().all(|v| v.is_finite() && *v >= 0.0);
        (finite && white.is_finite() && table.iter().any(|v| *v > 0.0)).then_some(NegEvLift { white, table })
    }

    /// The EV (0..[`MAX_EV`]) that raises L* at scene tone `t` (L* `l0`) by `want`, by bisection.
    fn solve(shown: &impl Fn(f32) -> f32, t: f32, l0: f32, want: f32) -> f32 {
        let (mut lo, mut hi) = (0.0f32, MAX_EV);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if shown(t + mid) - l0 < want { lo = mid } else { hi = mid }
        }
        0.5 * (lo + hi)
    }

    /// The lift (EV) for a pixel whose white-balanced linear Rec.2020 colour before the Exposure is `raw`.
    pub fn at(&self, raw: [f32; 3]) -> f32 {
        let m = raw[0].max(raw[1]).max(raw[2]);
        if !m.is_finite() || m <= 0.0 {
            return 0.0;
        }
        let y = lcc_color::luminance_2020(raw);
        let last_gap = (GAPS - 1) as f32;
        let g = if y > 0.0 { ((m / y).log2().max(0.0) / GAP_STEP).min(last_gap) } else { last_gap };
        let g = if g.is_finite() { g } else { last_gap };
        let h = self.white - (m / GREY).log2();
        let f = (h.max(0.0) / STEP).min((N - 1) as f32);
        let f = if f.is_finite() { f } else { (N - 1) as f32 };
        let (i, j) = ((f as usize).min(N - 2), (g as usize).min(GAPS - 2));
        let (t, u) = (f - i as f32, g - j as f32);
        let v = |i: usize, j: usize| self.table.get(i * GAPS + j).copied().unwrap_or(0.0);
        let a = v(i, j) + (v(i, j + 1) - v(i, j)) * u;
        let b = v(i + 1, j) + (v(i + 1, j + 1) - v(i + 1, j)) * u;
        a + (b - a) * t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone() -> ToneMap {
        let s = lcc_develop::DevelopSettings::default();
        assert_eq!(s.process.process(), lcc_develop::Process::V9);
        let info = crate::SourceInfo { raw: true, camera_tone: None, ..Default::default() };
        crate::finish::base_tone(&s, &info, 1.0)
    }

    #[test]
    fn nothing_at_exposure_zero_and_above() {
        let t = tone();
        for ev in [0.0, 0.01, 1.0, 3.0, f32::NAN, f32::INFINITY] {
            assert!(NegEvLift::new(&t, ev).is_none(), "{ev}");
        }
    }

    /// At −1 and −3 EV the lift raises L* by the measured amount at each headroom (a neutral pixel), and at −0.5 EV by
    /// half of it.
    #[test]
    fn the_lift_reaches_the_measured_lightness() {
        let t = tone();
        for (ev, share) in [(-1.0f32, 1.0f32), (-3.0, 1.0), (-0.5, 0.5)] {
            let l = NegEvLift::new(&t, ev).expect("a lift below Exposure 0");
            for h in [0.0f32, 0.5, 1.0, 2.0, 3.5] {
                let m = GREY * (l.white - h).exp2();
                let lift = l.at([m; 3]);
                let tone_at = |x: f32| lstar(t.apply(GREY * x.exp2()));
                let s = l.white - h + ev;
                let got = tone_at(s + lift) - tone_at(s);
                assert!((got - share * target(h)).abs() < 0.05, "ev {ev} headroom {h}: {got} L*");
            }
            assert_eq!(l.at([GREY * (l.white - 5.0).exp2(); 3]), 0.0, "deep tones keep their place");
            // past the clip: the clip's lift; not a colour: none
            assert!((l.at([GREY * (l.white + 1.0).exp2(); 3]) - l.at([GREY * l.white.exp2(); 3])).abs() < 1e-6);
            for bad in [[0.0; 3], [f32::NAN; 3], [-1.0; 3], [f32::INFINITY; 3]] {
                let v = l.at(bad);
                assert!(v.is_finite() && v >= 0.0, "{bad:?}: {v}");
            }
        }
    }

    /// Only a process-V9 raw rendered for SDR under a negative Exposure carries the lift; everything else renders
    /// exactly as before (no lift in its parameters), and the render at −1 EV does lift a pixel near the clip.
    #[test]
    fn only_a_v9_raw_under_a_negative_exposure_is_lifted() {
        let src = lcc_raster::Rgb32f::from_fn(16, 8, |x, _| [0.05 * (x as f32 + 1.0); 3]);
        let raw = crate::SourceInfo { raw: true, ..Default::default() };
        let jpeg = crate::SourceInfo { raw: false, ..Default::default() };
        let fp_of = |s: &lcc_develop::DevelopSettings, info: &crate::SourceInfo| {
            let req = crate::RenderRequest::fit(16, 8);
            let plan = crate::plan(&src, info, s, &req);
            crate::finish::FinishParams::new(s, &plan.frame, info, 16, 8, plan.px_per_long, 1.0, crate::OutputSpace::Srgb).negev_lift
        };
        let at = |ev: f64, process: lcc_develop::Process| {
            let mut s = lcc_develop::DevelopSettings { process: process.version(), ..Default::default() };
            s.light.exposure = ev;
            s
        };
        for ev in [0.0, 0.01, 0.5, 2.0] {
            assert!(fp_of(&at(ev, lcc_develop::Process::V9), &raw).is_none(), "{ev} EV");
        }
        assert!(fp_of(&at(-1.0, lcc_develop::Process::V9), &raw).is_some(), "V9 raw at -1 EV");
        assert!(fp_of(&at(-1.0, lcc_develop::Process::V8), &raw).is_none(), "V8");
        assert!(fp_of(&at(-1.0, lcc_develop::Process::V9), &jpeg).is_none(), "a JPEG");
        let mut hdr = at(-1.0, lcc_develop::Process::V9);
        hdr.hdr.enabled = true;
        if hdr.hdr.peak() > 1.0 {
            assert!(fp_of(&hdr, &raw).is_none(), "an HDR render");
        }
    }

    /// The lifted colour keeps the plain one's chromaticity at the lifted luminance; near white it becomes the lifted
    /// colour; bad inputs give the lifted colour.
    #[test]
    fn the_lift_keeps_the_colour() {
        let plain = [0.6, 0.35, 0.1];
        let lifted = [0.7, 0.5, 0.25];
        let got = keep_colour(plain, lifted);
        let (y0, y1) = (lcc_color::luminance_2020(plain), lcc_color::luminance_2020(lifted));
        assert!((lcc_color::luminance_2020(got) - y1).abs() < 1e-6);
        for n in 0..3 {
            assert!((got[n] - plain[n] * y1 / y0).abs() < 1e-6, "{got:?}");
        }
        let near = keep_colour([0.98, 0.6, 0.2], [0.99, 0.8, 0.5]);
        assert_eq!(near, [0.99, 0.8, 0.5], "past white: the lifted colour");
        assert_eq!(keep_colour([0.0; 3], lifted), lifted);
        assert_eq!(keep_colour([f32::NAN; 3], lifted), lifted);
    }

    /// The brightest channel sets the headroom: a colour with the same luminance and a brighter top channel has less
    /// headroom and takes more lift.
    #[test]
    fn the_lift_follows_the_brightest_channel() {
        let l = NegEvLift::new(&tone(), -2.0).expect("a lift");
        let m = GREY * (l.white - 0.25).exp2();
        assert!(l.at([m; 3]) > l.at([0.25 * m; 3]), "less headroom, more lift");
        // the same luminance (the neutral at 0.25 m) with red near the clip
        let y = 0.25 * m;
        let warm = [m, (y - lcc_color::LUMA_2020[0] * m) / lcc_color::LUMA_2020[1], 0.0];
        assert!((lcc_color::luminance_2020(warm) - y).abs() < 1e-6 * m);
        assert!(l.at(warm) > l.at([y; 3]), "{} vs {}", l.at(warm), l.at([y; 3]));
    }

    /// A saturated colour gets the measured L* on its own luminance, which lies below its brightest channel: less EV
    /// than a neutral with the same headroom, and its lightness rises by the target, not more.
    #[test]
    fn a_saturated_colour_is_lifted_on_its_luminance() {
        let t = tone();
        let ev = -2.0f32;
        let l = NegEvLift::new(&t, ev).expect("a lift");
        let m = GREY * (l.white - 0.25).exp2();
        // luminance exactly one stop under red (on the table's gap grid)
        let warm = [m, (0.5 - lcc_color::LUMA_2020[0] - 0.12 * lcc_color::LUMA_2020[2]) / lcc_color::LUMA_2020[1] * m, 0.12 * m];
        assert!(l.at(warm) < l.at([m; 3]), "{} vs {}", l.at(warm), l.at([m; 3]));
        let y = lcc_color::luminance_2020(warm);
        let s = (y / GREY).log2() + ev;
        let shown = |x: f32| lstar(t.apply(GREY * x.exp2()));
        let got = shown(s + l.at(warm)) - shown(s);
        assert!((got - target(0.25)).abs() < 0.1, "{got} L*");
    }
}
