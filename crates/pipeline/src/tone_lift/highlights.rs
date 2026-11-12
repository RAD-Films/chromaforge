//! A raw's Highlights −, as Lightroom Classic pulls it: one curve of the scene tone, placed and scaled per picture.
//!
//! Measured on Lightroom Classic exports of our own Canon EOS R5 C raws (rounds 9b and 10, 63 shots: skies with
//! clouds, window-lit rooms, a work bench, night and dusk frames; OneSlips, Nov 2026). Each pixel's pull is the stops
//! between Lightroom's Highlights −100 export and its 0 export, read back through the frame's own 0 export as a tone
//! curve, against the scene tone our raw decode rebuilds. Lightroom's pull is one curve of the scene tone: flat in the
//! shadows, about 2 EV at the anchor, and still falling past the camera curve's white by about 0.7 EV per EV, so
//! clipped skies go a little under white and show where recovery ends. Two things move it per picture: its depth
//! (deeper on dark frames, `k`) and where it sits (`a`, from the picture's 50th, 90th and 99th percentile tones).
//!
//! Fitted on 50 shots and scored on 13 it never saw (all-pixel / bright-unclipped L* rms against Lightroom's −100):
//! 3.3 / 6.0, where the fitted per-picture curve of [`super::raw`] was 4.2 / 8.3; at −50 1.9 / 3.0 (was 2.2 / 4.0).
//! Lightroom's −50 is about half its −100 up to white, which [`super::highlights_gain`] already gives.
//!
//! The curve and its per-picture constants were then solved together against the rendered result: one render per
//! constant on 64 shots gave how each moves every pixel, the step was solved on 51 shots with the curve kept smooth,
//! and its size picked by rendering. Change of the L* between each Highlights stage and 0 against Lightroom's, rms
//! over pixels lighter than L* 40 (fitted / held-out 13 shots): −25 1.19 / 0.81 (was 1.27 / 0.87), −50 2.21 / 1.45
//! (2.36 / 1.55), −75 3.29 / 2.37 (3.49 / 2.50), −100 4.10 / 3.36 (4.31 / 3.48). Most of what remains differs picture
//! by picture: the picture's tone statistics did not predict it on the held-out shots, nor did its surroundings at any
//! scale from 1 to 50 % of the long edge.

use super::Tones;

/// The curve at depth 1: pull (EV, negative darkens) at `E − a` (EV), linear between knots.
const KNOTS: [(f32, f32); 13] = [
    (-8.0, 0.000),
    (-6.0, -0.141),
    (-4.0, -0.347),
    (-3.0, -0.458),
    (-2.0, -0.678),
    (-1.0, -1.146),
    (0.0, -1.960),
    (1.0, -2.600),
    (2.0, -3.165),
    (2.25, -3.325),
    (3.0, -4.000),
    (3.75, -4.750),
    (5.0, -5.600),
];

/// Past the last knot the pull keeps growing this much per EV.
const TAIL: f32 = -0.7;

/// Depth `k = K0 + K1 · share of tones under −2 EV`, held to `K_MAX`.
const K0: f32 = 0.294;
const K1: f32 = 1.047;
const K_MAX: f32 = 1.6;

/// Anchor `a = A0 + A50 · p50 + A90 · p90 + A99 · p99` (scene percentiles, EV over grey, held at the white:
/// [`Pull::of`]), held to `A_LO..A_HI`. `A_HI` reaches past the anchors of bright pictures (5.5 EV on our 64 raws, 17 of
/// which a limit of 2.5 held back).
const A0: f32 = 3.516;
const A50: f32 = 0.657;
const A90: f32 = 0.471;
const A99: f32 = -0.292;
const A_LO: f32 = -7.0;
const A_HI: f32 = 6.0;

/// The pull curve at depth 1 (EV) for `x = E − a`.
fn curve(x: f32) -> f32 {
    let (Some(first), Some(last)) = (KNOTS.first(), KNOTS.last()) else {
        return 0.0;
    };
    if !x.is_finite() || x <= first.0 {
        return 0.0;
    }
    if x >= last.0 {
        return last.1 + TAIL * (x - last.0);
    }
    KNOTS
        .windows(2)
        .find_map(|w| match w {
            [p, q] if x < q.0 => Some(p.1 + (q.1 - p.1) * (x - p.0) / (q.0 - p.0)),
            _ => None,
        })
        .unwrap_or(0.0)
}

/// One picture's Highlights −100 pull: depth and anchor from its scene tones.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Pull {
    k: f32,
    a: f32,
}

impl Pull {
    /// The pull for a picture whose scene tones (EV over grey, white-balanced and exposed) are `t`, its percentiles
    /// held at `white` (the scene tone where the raw's camera curve reaches white) when given. Every tone above it shows
    /// as white, and a clipped area's rebuilt colour moves its luminance there: the 99th percentile often lies inside a
    /// clipped area, and changing only the colour of clipped pixels moved the whole picture's pull by up to 10 L*.
    pub(super) fn of(t: &Tones, white: Option<f32>) -> Pull {
        let cap = white.filter(|w| w.is_finite()).unwrap_or(f32::INFINITY);
        let p = |q: f32| t.percentile_held(q, cap);
        let k = K0 + K1 * t.below(-2.0);
        let a = A0 + A50 * p(50.0) + A90 * p(90.0) + A99 * p(99.0);
        Pull { k: if k.is_finite() { k.clamp(0.0, K_MAX) } else { K0 }, a: if a.is_finite() { a.clamp(A_LO, A_HI) } else { 0.0 } }
    }

    /// The pull (EV, ≤ 0) of scene tone `e` at Highlights −100.
    pub(super) fn at(&self, e: f32) -> f32 {
        let v = self.k * curve(e - self.a);
        if v.is_finite() { v.min(0.0) } else { 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curve_is_flat_in_the_shadows_and_falls_past_white() {
        assert_eq!(curve(-20.0), 0.0);
        assert!((curve(0.0) + 1.96).abs() < 1e-6);
        assert!((curve(7.0) - (-5.6 - 1.4)).abs() < 1e-5);
        // never rises: tones keep their order under the pull's slope (> −1)
        let mut last = 0.0f32;
        for i in 0..400 {
            let v = curve(-10.0 + i as f32 * 0.05);
            assert!(v <= last + 1e-6, "rose at {i}");
            last = v;
        }
        assert_eq!(curve(f32::NAN), 0.0);
    }

    #[test]
    fn dark_pictures_get_a_deeper_pull() {
        let bright = Tones::new((0..100).map(|i| -1.0 + i as f32 * 0.03)).unwrap();
        let dark = Tones::new((0..100).map(|i| -6.0 + i as f32 * 0.07)).unwrap();
        let (b, d) = (Pull::of(&bright, None), Pull::of(&dark, None));
        assert!(d.k > b.k);
        assert!(d.at(0.0) < b.at(0.0), "{} vs {}", d.at(0.0), b.at(0.0));
        assert!(b.at(-12.0) == 0.0 && d.at(-12.0) == 0.0);
        // well past a bright picture's anchor the pull is past a stop
        assert!(b.at(8.0) < -1.0, "{}", b.at(8.0));
    }

    /// A clipped area (the top tenth of the tones, above the white at 2.6 EV) whose rebuilt luminance changes, as when
    /// only its colour does: the pull stays the same. Read unheld, the anchor moved with it.
    #[test]
    fn tones_above_the_white_do_not_move_the_pull() {
        let picture = |top: f32| Tones::new((0..100).map(|i| if i < 90 { -7.0 + i as f32 * 0.06 } else { top })).unwrap();
        let (a, b) = (picture(2.8), picture(3.6));
        assert_eq!(Pull::of(&a, Some(2.6)), Pull::of(&b, Some(2.6)));
        assert_ne!(Pull::of(&a, None), Pull::of(&b, None));
    }
}
