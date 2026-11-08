//! Process V9 Split Toning: the Color Grading panel's shadows and highlights wheels as Lightroom Classic applies them.
//! Measured on its own exports of our grey-ramp and colour charts (round 17: each wheel at six hues and two
//! saturations, three balances, two blendings), all in OkLab on display values:
//! - LrC adds an offset that is nearly the same for every colour at a given lightness, not a multiplicative tint.
//! - The offset points along the OkLab direction of the slider hue's ProPhoto HSV colour, with a small per-hue
//!   correction. Its size and its lightness change depend on the hue (two-harmonic Fourier fits over the six hues).
//! - How much each lightness gets is one profile per wheel, whatever the hue (rank one: 99.6 % and 100 % of the
//!   response). The profiles are bumps, not ramps: the shadows wheel peaks near L 0.32 and still gives half its
//!   strength near black; the highlights wheel peaks near L 0.85 and falls off towards white.
//! - Strength grows as (saturation / 50)^1.24.
//! - Balance shifts both profiles (0.21 and 0.14 L per 100) and moves strength between them (∓0.89 per 100);
//!   Blending above 50 narrows them a little (−0.077 per 50) and strengthens them (+0.22 per 50).
//!
//! On the round-17 exports (mean OkLab ΔE×100 on the patches LrC moves) this scores 0.68, V8's wheels 1.75.
//! Midtones, global and the wheels' Luminance sliders were not measured and keep V8's behaviour.

use lcc_color::perceptual::{hsv_to_rgb, oklab_from_2020, oklab_to_2020};
use lcc_color::{D50, D65, PROPHOTO, REC2020, bradford};

/// Lightness of the profiles' first knot, and the knot spacing.
pub const SPLIT_ZONE_L0: f32 = 0.08;
pub const SPLIT_ZONE_DL: f32 = 0.015;
/// The shadows wheel's strength by OkLab lightness (knots from [`SPLIT_ZONE_L0`], [`SPLIT_ZONE_DL`] apart).
#[rustfmt::skip]
pub const SPLIT_ZONE_SH: [f32; 61] = [0.46674, 0.46674, 0.50039, 0.54666, 0.59778, 0.64969, 0.70330, 0.75662, 0.80406, 0.85002, 0.89026, 0.92657, 0.95517, 0.97794, 0.99347, 0.99997, 0.99999, 0.98838, 0.97206, 0.94310, 0.91177, 0.86765, 0.82227, 0.76894, 0.71470, 0.65736, 0.60064, 0.54507, 0.49277, 0.44399, 0.40158, 0.36323, 0.33372, 0.30730, 0.29057, 0.27531, 0.26238, 0.24935, 0.23560, 0.22172, 0.20746, 0.19316, 0.17882, 0.16452, 0.15026, 0.13629, 0.12247, 0.10923, 0.09616, 0.08400, 0.07198, 0.06117, 0.05058, 0.04122, 0.03235, 0.02470, 0.01783, 0.01210, 0.00749, 0.00384, 0.00262];
/// The highlights wheel's strength by OkLab lightness.
#[rustfmt::skip]
pub const SPLIT_ZONE_HL: [f32; 61] = [0.0, 0.0, 0.00129, 0.00306, 0.00310, 0.00285, 0.00411, 0.00562, 0.00777, 0.01013, 0.01334, 0.01682, 0.02084, 0.02565, 0.03143, 0.03792, 0.04492, 0.05315, 0.06188, 0.07201, 0.08241, 0.09417, 0.10614, 0.11966, 0.13352, 0.14849, 0.16383, 0.17981, 0.19610, 0.21273, 0.22960, 0.24664, 0.26357, 0.28046, 0.29718, 0.31387, 0.33895, 0.36567, 0.40363, 0.44464, 0.49459, 0.54786, 0.60650, 0.66674, 0.72851, 0.78794, 0.84605, 0.89533, 0.94196, 0.97134, 0.99810, 0.99913, 0.99525, 0.96420, 0.92305, 0.85607, 0.77369, 0.66912, 0.54454, 0.40274, 0.35540];
/// Fourier coefficients over the slider hue (a0, cos h, sin h, cos 2h, sin 2h): offset size, lightness change, and
/// direction correction (degrees) from the ProPhoto HSV colour's OkLab direction.
const SH_MAG: [f32; 5] = [0.06872, -0.00571, 0.00837, 0.00515, -0.02091];
const SH_DL: [f32; 5] = [0.00232, 0.00934, -0.00362, -0.00210, 0.00326];
const SH_OFF: [f32; 5] = [0.62477, 2.38810, 1.40832, -0.63392, 1.39601];
const HL_MAG: [f32; 5] = [0.05984, 0.00069, -0.00242, 0.00468, -0.01835];
const HL_DL: [f32; 5] = [-0.00120, 0.00946, -0.00124, 0.00361, -0.00649];
const HL_OFF: [f32; 5] = [0.58434, 0.23874, -0.33094, -0.58349, 1.02269];
/// Strength = (saturation / 50)^SAT_POW.
const SAT_POW: f32 = 1.23661;
/// Balance: profile shift per 100 (shadows, highlights); blending: width change and strength change per 50 above
/// 50; balance: strength moved per 100.
const SHIFT: [f32; 2] = [0.20962, 0.14118];
const BLEND_WIDTH: f32 = -0.07721;
const BALANCE_AMP: f32 = 0.89245;
const BLEND_AMP: f32 = 0.22226;
/// The lightness the profiles stretch around under Blending.
pub const SPLIT_CENTRE: f32 = 0.61;

/// Both wheels, ready to apply: each one's OkLab offset (L, a, b) at full strength, where its profile is read
/// (shift), and the profiles' common stretch. Words for the GPU in [`SplitK::words`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitK {
    pub shadows: [f32; 3],
    pub highlights: [f32; 3],
    pub shift: [f32; 2],
    pub scale: f32,
}

/// Number of [`SplitK::words`].
pub const SPLIT_WORDS: usize = 9;

fn fourier(c: &[f32; 5], hue_deg: f32) -> f32 {
    let r = hue_deg.to_radians();
    c[0] + c[1] * r.cos() + c[2] * r.sin() + c[3] * (2.0 * r).cos() + c[4] * (2.0 * r).sin()
}

/// OkLab direction (radians) of a small step from mid grey towards the ProPhoto HSV colour of `hue_deg`, at the
/// grey's luminance.
fn direction(hue_deg: f32) -> f32 {
    let to_pp = REC2020.to_space(&PROPHOTO);
    let from_pp = PROPHOTO.to_space(&REC2020);
    let y = bradford(D50, D65).mul(&PROPHOTO.to_xyz()).0[1];
    let grey = oklab_to_2020([0.5, 0.0, 0.0]);
    let g = to_pp.apply_f32(grey);
    let t = hsv_to_rgb(hue_deg, 1.0, 1.0);
    let lum = |c: [f32; 3]| (y[0] * c[0] as f64 + y[1] * c[1] as f64 + y[2] * c[2] as f64) as f32;
    let k = lum(g) / lum(t).max(1e-6);
    let q: [f32; 3] = std::array::from_fn(|i| g[i] + 0.01 * (t[i] * k - g[i]));
    let a = oklab_from_2020(from_pp.apply_f32(q));
    let b = oklab_from_2020(from_pp.apply_f32(g));
    (a[2] - b[2]).atan2(a[1] - b[1])
}

/// One wheel's OkLab offset (L, a, b) at saturation 50.
fn offset(hue_deg: f32, mag: &[f32; 5], dl: &[f32; 5], off: &[f32; 5]) -> [f32; 3] {
    let a = direction(hue_deg) + fourier(off, hue_deg).to_radians();
    let m = fourier(mag, hue_deg);
    [fourier(dl, hue_deg), m * a.cos(), m * a.sin()]
}

fn strength(sat: f64) -> f32 {
    let s = (sat.clamp(0.0, 100.0) / 50.0) as f32;
    if s > 0.0 { s.powf(SAT_POW) } else { 0.0 }
}

impl SplitK {
    /// The grading's shadows and highlights wheels; `None` when neither has saturation.
    pub fn new(g: &lcc_develop::ColorGrading) -> Option<SplitK> {
        let (ss, sh) = (strength(g.shadows.sat), strength(g.highlights.sat));
        if ss == 0.0 && sh == 0.0 {
            return None;
        }
        let bal = (g.balance.clamp(-100.0, 100.0) / 100.0) as f32;
        let blend = ((g.blending.clamp(0.0, 100.0) - 50.0) / 50.0) as f32;
        let amp = (1.0 + BLEND_AMP * blend).max(0.0);
        let hue = |h: f64| (h.rem_euclid(360.0)) as f32;
        let mut shadows = if ss > 0.0 { offset(hue(g.shadows.hue), &SH_MAG, &SH_DL, &SH_OFF) } else { [0.0; 3] };
        let mut highlights = if sh > 0.0 { offset(hue(g.highlights.hue), &HL_MAG, &HL_DL, &HL_OFF) } else { [0.0; 3] };
        let ks = ss * (1.0 - BALANCE_AMP * bal).max(0.0) * amp;
        let kh = sh * (1.0 + BALANCE_AMP * bal).max(0.0) * amp;
        shadows = shadows.map(|v| v * ks);
        highlights = highlights.map(|v| v * kh);
        let k = SplitK { shadows, highlights, shift: [-SHIFT[0] * bal, -SHIFT[1] * bal], scale: (1.0 + BLEND_WIDTH * blend).max(0.2) };
        k.words().iter().all(|v| v.is_finite()).then_some(k)
    }

    /// The offset added to OkLab `lab` (by its lightness).
    #[inline]
    pub fn offset(&self, l: f32) -> [f32; 3] {
        let ws = zone(&SPLIT_ZONE_SH, SPLIT_CENTRE + (l - self.shift[0] - SPLIT_CENTRE) / self.scale);
        let wh = zone(&SPLIT_ZONE_HL, SPLIT_CENTRE + (l - self.shift[1] - SPLIT_CENTRE) / self.scale);
        std::array::from_fn(|i| ws * self.shadows[i] + wh * self.highlights[i])
    }

    /// For the GPU (`finish.wgsl`, `split_offset`): shadows (L, a, b), highlights (L, a, b), the two shifts, the scale.
    pub fn words(&self) -> [f32; SPLIT_WORDS] {
        let [s, h] = [self.shadows, self.highlights];
        [s[0], s[1], s[2], h[0], h[1], h[2], self.shift[0], self.shift[1], self.scale]
    }
}

/// A profile read at lightness `x` (linear between knots, its ends held outside).
#[inline]
pub fn zone(t: &[f32; 61], x: f32) -> f32 {
    let f = ((x - SPLIT_ZONE_L0) / SPLIT_ZONE_DL).clamp(0.0, (t.len() - 1) as f32);
    if !f.is_finite() {
        return 0.0;
    }
    let i = (f as usize).min(t.len() - 2);
    let u = f - i as f32;
    let (a, b) = (t.get(i).copied().unwrap_or(0.0), t.get(i + 1).copied().unwrap_or(0.0));
    a + (b - a) * u
}

#[cfg(test)]
mod tests {
    use super::*;
    use lcc_color::perceptual::{oklab_from_2020, oklab_to_2020};
    use lcc_develop::{ColorGrading, Wheel};

    // measured data: a value near 2/π is a measurement, not the constant
    #[rustfmt::skip]
    #[allow(clippy::approx_constant)]
    mod r17 {
        include!("split_v9_r17.rs");
    }
    use r17::{R17_SPLIT, R17_SPLIT_BASE};

    /// Mean OkLab error against LrC over the patches LrC moves, and the worst difference from the fitted model, for
    /// `process`'s Color Grading on every round-17 Split Toning export.
    fn r17_error(process: lcc_develop::Process) -> (f32, f32, f32) {
        let (mut sum, mut worst, mut model) = (0.0f32, 0.0f32, 0.0f32);
        for (shh, shs, hlh, hls, bal, blend, lrc, fit) in R17_SPLIT.iter() {
            let mut s = lcc_develop::DevelopSettings { process: process.version(), ..Default::default() };
            s.grading.shadows = Wheel { hue: *shh, sat: *shs, lum: 0.0 };
            s.grading.highlights = Wheel { hue: *hlh, sat: *hls, lum: 0.0 };
            s.grading.balance = *bal;
            s.grading.blending = *blend;
            let ops = crate::colorops::ColorOps::new(&s);
            let (mut e, mut n) = (0.0f32, 0.0f32);
            for ((b, l), f) in R17_SPLIT_BASE.iter().zip(lrc.iter()).zip(fit.iter()) {
                let o = oklab_from_2020(ops.apply(oklab_to_2020(*b), 0.0, 0.0));
                let d = |x: &[f32; 3]| ((o[0] - x[0]).powi(2) + (o[1] - x[1]).powi(2) + (o[2] - x[2]).powi(2)).sqrt();
                model = model.max(d(f));
                let moved = ((l[0] - b[0]).powi(2) + (l[1] - b[1]).powi(2) + (l[2] - b[2]).powi(2)).sqrt();
                if moved > 0.005 {
                    e += d(l);
                    n += 1.0;
                }
            }
            let e = e / n.max(1.0);
            sum += e;
            worst = worst.max(e);
        }
        (sum / R17_SPLIT.len() as f32, worst, model)
    }

    #[test]
    fn v9_split_toning_follows_lightroom_classic_on_our_charts() {
        let (v8, v8_worst, _) = r17_error(lcc_develop::Process::V8);
        let (v9, v9_worst, model) = r17_error(lcc_develop::Process::V9);
        eprintln!(
            "round 17 split toning, mean / worst OkLab error: V8 {v8:.4} / {v8_worst:.4}, V9 {v9:.4} / {v9_worst:.4}; V9 vs the fitted model {model:.4}"
        );
        // measured: V8 0.0169 / 0.0349, V9 0.0041 / 0.0053, V9 within 0.0024 of the model (its profiles on a finer grid)
        assert!(v9 < 0.005 && v9_worst < 0.007, "V9 {v9} / {v9_worst}");
        assert!(v9 < 0.35 * v8 && v9_worst < 0.25 * v8_worst, "V9 {v9} / {v9_worst} vs V8 {v8} / {v8_worst}");
        assert!(model < 0.004, "V9 is {model} off the fitted model");
    }

    #[test]
    fn neutral_wheels_give_nothing_and_odd_values_stay_finite() {
        assert_eq!(SplitK::new(&ColorGrading::default()), None);
        for v in [f64::NAN, f64::INFINITY, -1e9, 1e9, -0.0] {
            let g = ColorGrading {
                shadows: Wheel { hue: v, sat: v, lum: v },
                highlights: Wheel { hue: v, sat: 50.0, lum: 0.0 },
                blending: v,
                balance: v,
                ..Default::default()
            };
            if let Some(k) = SplitK::new(&g) {
                for l in [-1.0, 0.0, 0.5, 1.0, 2.0, f32::NAN] {
                    let o = k.offset(l);
                    assert!(o.iter().all(|x| x.is_finite()), "{v} at {l}: {o:?}");
                }
            }
        }
        assert_eq!(zone(&SPLIT_ZONE_SH, f32::NAN), 0.0);
    }

    #[test]
    fn shadows_and_highlights_wheels_tint_their_own_tones() {
        let g = ColorGrading { shadows: Wheel { hue: 210.0, sat: 50.0, lum: 0.0 }, ..Default::default() };
        let k = SplitK::new(&g).unwrap();
        let (dark, light) = (k.offset(0.3), k.offset(0.9));
        assert!(dark[1].hypot(dark[2]) > 4.0 * light[1].hypot(light[2]), "{dark:?} vs {light:?}");
        // a blue-cyan hue: negative a (towards green-blue) and negative b (towards blue)
        assert!(dark[1] < 0.0 && dark[2] < 0.0, "{dark:?}");
        let g = ColorGrading { highlights: Wheel { hue: 30.0, sat: 50.0, lum: 0.0 }, ..Default::default() };
        let k = SplitK::new(&g).unwrap();
        let (dark, light) = (k.offset(0.3), k.offset(0.85));
        assert!(light[1].hypot(light[2]) > 4.0 * dark[1].hypot(dark[2]), "{dark:?} vs {light:?}");
        assert!(light[1] > 0.0 && light[2] > 0.0, "an orange: {light:?}");
    }
}
